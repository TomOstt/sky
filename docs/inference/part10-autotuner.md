# Part X — Autotuner and search

The autotuner chooses the open parameters that the scheduler ([Part IX](part09-scheduler.md)) leaves to measurement:
- tile shapes and work division inside each skeleton;
- the fusion partition where the cost model cannot separate the candidates;
- model-level settings: speculation shape, KV format, prefill chunk sizes.

It measures on the user's machine, stores the winners in `sky-tune`'s data-only database, and never lets a stored choice move a result outside the numerics contract.

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R7 §10) and live in [`research/`](research/).

## 0. Summary

- **Three levels** (SPINE D11):
  - **kernel**: a pruned BEAM or exhaustive search over skeleton parameters per shape bucket;
  - **block**: a search over the scheduler's tied fusion partitions, measured once on one layer template and reused for all layers;
  - **model**: speculation shape, KV format and chunk sizes, measured end to end.
- **Every parameter is classified** as *order-preserving* (it cannot change any result bit) or *order-changing* (it can change the last bits).
  - Under `--numerics=inference`, both kinds are tuned. An order-changing candidate must pass a tolerance test against the reference before it is stored, and the plan records the tuning entry it used.
  - Under `--numerics=portable`, only order-preserving parameters are tuned, and the spec's guarantee "tuning changes speed, never results" holds unchanged (SPINE D6).
- **Timing is the hard part.** tinygrad's BEAM picked a 4-bit GEMV it had timed at 206 µs that really took about 540 µs, because it measured warm caches and took a minimum of three runs (R2 §6.4). Sky times on rotated DRAM-sized buffers with GPU timestamps, warm clocks, interleaved rounds and medians. It also rejects any measurement faster than the kernel's DRAM floor.
- **Tuning a fixed model is cheap.** Qwen3-32B has about 60 distinct hot kernel instances. An exhaustive overnight search takes 2–5 hours (R7 §10.2). With cost-model pruning, `-O3` for the decode path takes minutes.
- **The database stays in `sky-tune`'s existing data-only text format.** GPU records add keys, which version-1 readers already ignore (`sky-tune/src/db.rs:338-340`). The database never holds code: MSL is regenerated from template IDs and integer parameters (SPINE D19).

## 1. Where the autotuner sits

```text
 Part IX ScheduledPlan ──► open parameters per KernelGroup ──┐
          + tied partitions (Δ predicted < 3%)              │
                                                            ▼
 sky-tune DB lookup (fingerprint, kernel.template, size bucket, contract)
     hit  ──────────────────────────────────────────────► use stored params (-O2, -O3)
     miss ─► -O1/-O2: analytic default from Part XXII's model (best predicted)
          ─► -O3 / overnight: search ──► verify ──► time ──► store ──► use
                                           │
                    reference interpreter (sky-tir) + invariance tests
```

Existing pieces (R1 §5.7, verified in `~/sky/crates/sky-tune/src/`):
- `Tunable` (`measure.rs:31`): a kernel name, a size class, up to 64 candidates (the first is the default), and a bench closure.
- `Config` (`measure.rs:98`): 500 ms budget, 5–31 repeats, 200 µs minimum sample, and a 3% minimum gain before a candidate replaces the default.
- `Db` and `Record` (`db.rs:49`, `db.rs:114`): bounded text, at most 1 MiB, 4,096 records and 1,024-byte lines.
- `lookup(kernel, size_class)` (`lib.rs:138`), keyed by `sky_hw::current_fingerprint()`.

The GPU work extends these; it does not replace them.

## 2. The three levels

### 2.1 Kernel level

**Unit.** One skeleton at one shape bucket, one format and one chip. For example `qgemv` over a 10,240 × 5,120 matrix of MLX 4-bit group-64 weights on an M5 Max.

**Search.**
- Exhaustive over the pruned grid when it holds ≤ 64 candidates, which is `sky-tune`'s `MAX_CANDIDATES`.
- Otherwise BEAM: keep the best 8, and expand each by changing one parameter at a time to a neighboring value, until no candidate improves by more than the noise threshold. tinygrad's BEAM expands every beam member by about 210 actions (R2 §6.4). Sky's spaces are about 10× smaller, because skeletons expose only the knobs that matter (§3).

**Instances for Qwen3-32B** (R7 §10.2):

| Group | Instances | Notes |
|---|---:|---|
| decode GEMVs | 5 | QKV, O, gate/up, down, lm_head |
| verify GEMVs | 5 × 7 widths | shares order-changing parameters with the decode GEMVs (§3.2) |
| prefill GEMMs | 5 × 9 M buckets | 16 … 4096 |
| attention | ≈ 14 | decode by context bucket, prefill by chunk |
| small kernels | ≈ 5 | statistics, combine, sampling |

### 2.2 Block level

**Unit.** One block template: a transformer layer under one specialization (decode M = 1, verify M = k, prefill chunk T). All layers share shapes, so the plan for one is the plan for all (SPINE D11).

**Candidates.**
- The scheduler's tied partitions: the alternatives whose predicted times were within ε = 3% ([Part IX](part09-scheduler.md) §6).
- The named alternatives:
  - head-aligned K1 or P1b ([Part IX](part09-scheduler.md) §3.4);
  - recomputed or stored norm statistics at the crossover widths (§4.2 there);
  - separate or in-kernel split combine.

**Measurement.** A block candidate is timed inside the **full decode step**: the candidate block applied to all 64 layers, with the real weights. A Qwen3-32B step streams 18 GB, far beyond any cache, so the measurement is honest by construction, and every kernel pays its real fill and boundary costs. One step takes about 35 ms on the M5 Max, so 15 interleaved repetitions of four candidates take about 2 seconds.

Kernel-level winners are fixed first. Block-level choices are then made with them in place; if a block choice changes a kernel's shape (head-aligned tiling), the kernel level reruns for that shape.

### 2.3 Model level

**Unit.** The whole program for a workload class.

| Choice | Candidates | Objective | Measured how |
|---|---|---|---|
| Speculation | draft source (EAGLE-3 head, small draft model, prompt lookup); γ ∈ 1…7; tree shape (nodes ≤ 8 in ALU-family plans, ≤ 32 in NA-family plans, [Part IX](part09-scheduler.md) §8) | effective tok/s, S = τ / (1 + γc + γβ) ([Part XXII](part22-performance-model.md) §9) | τ from acceptance traces on a calibration set per workload class; c and β from measured step times |
| KV format | fp16 (default); int8 as a labeled mode | tok/s and memory at a stated context, subject to the quality budget | quality gate first: KLD and top-1 agreement (SPINE D13); then speed |
| Prefill chunk and bucket set | chunk ∈ {256, 512, 1024, 2048}; bucket sets | TTFT on a prompt-length distribution; cold compile time | end to end, with the idle gap fixed (R9 §6.4) |
| Attention split size b_kv | {128, 256, 512} | decode time weighted over the context distribution | full-step replay at three contexts |

Model-level choices never change the model's math. Speculation is exact by construction ([Part XXI](part21-speculative.md)), and the int8 KV format is a separately labeled mode with a quality budget, never a silent substitution (SPINE D5).

## 3. Search spaces, by skeleton

### 3.1 Order-preserving and order-changing parameters

A parameter is **order-preserving (OP)** if every output element is computed by the same sequence of floating-point operations whatever its value. Examples: which threadgroup computes a row, how many rows a simdgroup owns, whether an activation is staged in threadgroup memory.

A parameter is **order-changing (OC)** if it can change the order or grouping of a reduction, or the instruction that performs it. Examples:
- how many values each lane accumulates before the `simd_sum`;
- a split of K across simdgroups;
- the attention split size;
- any parameter of MPP `matmul2d`, whose internal order is opaque (R6 §A2).

| Contract | OP parameters | OC parameters | What a result depends on |
|---|---|---|---|
| `portable` | tuned | fixed at canonical values; no MPP on the decode path, or fixed tiles where MPP is required | chip-independent canonical order: the spec's guarantee "tuning changes speed, never results" holds |
| `inference` (default) | tuned | tuned, each candidate tolerance-tested before storing; shared across an invariance group (§3.2) | (chip, OS and Metal compiler, Sky build, tuning entry): bit-reproducible given these, within tolerance of the reference always |

The tuning entries a plan used are recorded in it, and `sky explain` prints them (SPINE D6; Part VI §8.8).

### 3.2 Invariance groups

Batch and chunk invariance (SPINE D6) constrain OC parameters beyond tolerance. Instantiations that must give identical bits for the same row share one OC choice:

| Invariance group | Members | Why |
|---|---|---|
| per weight matrix, decode | `Gemv` (M = 1) and `VerifyGemv` (M = 2…8) | speculative output must equal plain output byte for byte ([Part IX](part09-scheduler.md) §8) |
| per weight matrix, NA-family plans | `NaGemm` at M = 1 (padded) … 32 | the same, on the neural-accelerator path |
| per weight matrix, prefill | `Gemm` or `NaGemm` at every chunk bucket | chunk invariance |
| attention | decode, verify and prefill attention | b_kv and the block order are shared; prefill blocks align to b_kv (R8 §3.3) |
| per RMSNorm | recomputed and stored statistics | the stored statistic must equal the recomputed one ([Part IX](part09-scheduler.md) §4.2) |

The tuner therefore tunes an invariance group's OC parameters jointly. The objective is the group's time weighted by expected use: for decode, `T(M = 1)` weighted by the share of plain steps plus `T(M = γ + 1)` weighted by the share of verify passes. OP parameters are tuned per instance.

### 3.3 Spaces per skeleton

| Skeleton | Parameter | Values | Class |
|---|---|---|---|
| `qgemv` / `verify_gemv` | rows per simdgroup r | 1, 2, 4, 8 | OP |
| | simdgroups per threadgroup s | 2, 4, 8, 16, 32 | OP |
| | values per lane per step v | 8, 16, 32 | OC (changes each lane's accumulation sequence) |
| | K split across simdgroups | 1, 2, 4 | OC |
| | activation staging in threadgroup memory | off, on | OP |
| | activation registers for M > 1 | fp32, fp16 | OC (fp16 changes products' rounding; tolerance-gated) |
| | weight layout | as loaded, group-interleaved | OP: layout only; dequantized values bit-equal (SPINE D5) |
| `gemm` (simdgroup MMA, M1–M4) | BM, BN | 32, 64, 128 | OP |
| | BK | 16, 32, 64 | OP (sequential K order in 8-wide MMA steps is unchanged) |
| | WM × WN | 1×2, 2×2, 2×4 | OP |
| | split-K | 1, 2, 4 | OC |
| | double-buffered loads, swizzle | off/on; 0–3 | OP |
| `na_gemm` (MPP `matmul2d`, M5) | descriptor M × N × K per call | 16×32×16 (MLX's choice, R3-kernels §d), 32×32×32, 64×64×32 … | OC (opaque internal order) |
| | threadgroup tile BM × BN | 32, 64, 128 | OC (it determines the descriptor, so it is treated as OC) |
| | dequantization target | threadgroup memory, cooperative tensor | OP (bit-equal values) |
| | `relaxed_precision` | off, on | OC; tolerance-gated |
| | swizzle | 0–3 | OP |
| `attn_decode` | split size b_kv | 128, 256, 512 | OC; fixed per (model, chip, entry) ([Part IX](part09-scheduler.md) §10) |
| | GQA packing | query-head-parallel, KV-head-packed | OP (same per-head arithmetic) |
| | simdgroups per threadgroup | 4, 8, 16, 32 | OP if they split heads, OC if they split positions inside a block |
| `attn_prefill` | BQ | 32, 64 | OP |
| | BK (KV block of the online softmax) | 32, 64 | OC; must divide b_kv |
| `sample` | threadgroups and partial count | 64 … 4096 | OP for argmax and Gumbel-max (max with a fixed tie-break is exact); OC for normalized sums (top-p, log-probabilities) |
| generated reductions | tree width | 32, 64, 256 | OC |
| generated elementwise | vector width, threadgroup size | 1, 2, 4; 64 … 1024 | OP |

Raw sizes are 2–400 combinations per skeleton (`qgemv`: 4 × 5 × 3 × 3 × 2 × 2 × 2 = 1,440 before guards). After the guards of [Part IX](part09-scheduler.md) §5 and the model's pruning (§5), 20–100 candidates per instance remain. That matches R7's estimate of about 100 pruned configurations per GEMV and about 300 per GEMM.

## 4. Honest timing

### 4.1 The rules

| Rule | How | Why |
|---|---|---|
| **DRAM-sized working sets** | Each repetition uses a different copy of the weights, rotating through ≥ 512 MB, or the real model inside a full-step replay (§2.2). | The M2 Pro is cache-resident up to about 8 MiB and has a 24 MB system-level cache (R6 §B2, R3 §7). A 28 MB matrix timed warm misled tinygrad's BEAM by 2.6× (R2 §6.4). |
| **GPU timestamps** | Metal 4 `writeTimestamp` on a counter heap around each measured dispatch, or one encoder per measured kernel with the stage-boundary counter. Never wall clock. | Apple GPUs expose no per-dispatch counters; only stage boundaries (R6 §A4). Wall clock includes a 100–160 µs sync floor (R3 §7.B). |
| **Measured in a chain** | A candidate runs N times, dependent, with barriers. The per-instance cost is the slope over N, which includes its fill and boundary costs. | Isolated timings miss the fill term, which is 2.3 µs per kernel on the M2 Pro ([Part XXII](part22-performance-model.md) §2.2). |
| **Warm clock** | 2 s of a sustained GEMV chain before the first candidate; the thermal state recorded before and after. Results are discarded if it left nominal. | The GPU clock is not pinned (R6). Laptops throttle about 20% under sustained load (R6). R9 §6.4 gives the thermal readout. |
| **Interleaved rounds** | Each round runs every surviving candidate once, in a shuffled order; at least 9 rounds. | Drift (thermal, background) hits all candidates equally. |
| **Median and its confidence interval** | Bootstrap 95% interval of the median. A candidate replaces the default only if it is ≥ 3% faster (`sky-tune`'s `min_gain`) and the interval excludes the default's median. | `sky-tune`'s existing rule, plus R9 §6.5's statistics. |
| **DRAM-floor check** | Reject any measurement more than 5% below `bytes / BW_raw` for a streaming kernel. | It is physically impossible from DRAM; it means the cache served the data ([Part XXII](part22-performance-model.md) §10.2). |
| **Winner re-check** | The kernel-level winner is re-timed inside the full-step replay before it is stored. | The final arbiter is the step, not the microbenchmark. |
| **Quiet machine** | AC power, High Power Mode, no other GPU clients; the R9 §6.5 quiet gate for `--autotune=overnight`. | Display composition steals GPU time and bandwidth (R6 §B). |

### 4.2 The tinygrad lesson, specifically

tinygrad's BEAM search for a fused Q4_0 GEMV went wrong in four ways (R2 §6.4, VERIFIED):
1. **No cache flush on Metal.** `invalidate_caches` exists only for NVIDIA (`codegen/opt/search.py:144`).
2. **Minimum of three runs, with early stop.** The minimum is the most cache-favored sample.
3. **Shrunken launch size, extrapolated.** The global size is reduced to ≤ 65,536 work items and the time scaled up.
4. **A 28 MB working set**, which partly fit the system-level cache.

It reported 205.96 µs for the kernel it chose. That kernel ran at about 540 µs, slower than the heuristic's 316–510 µs. For a 100 MB fp16 matrix, which does not fit the cache, the same BEAM was right (530 µs).

Each rule in §4.1 addresses one of these failures: rotation for 1 and 4, medians for 2, full-size launches in a chain for 3, and the DRAM-floor check as a backstop for all of them.

## 5. Cost-model pruning

Before anything is compiled, every candidate gets a predicted time from [Part XXII](part22-performance-model.md): `predict(KernelCost(candidate), GpuFacts)`. That needs no compilation and costs microseconds.

1. **Guards first.** Candidates failing the static guards (threadgroup memory, bindings, the register estimate, the ALU budget) are dropped ([Part IX](part09-scheduler.md) §5).
2. **Predicted-time cut.** Candidates predicted slower than 1.25× the best prediction are dropped. The model's validation error is ±3.4% on whole steps ([Part XXII](part22-performance-model.md) §5), so 25% is a loose cut. It is loosened to 1.5× for skeletons where the model's residual on the last tuning exceeded 15%.
3. **Compile the survivors in parallel** with the asynchronous Metal 4 compiler, and drop any whose pipeline's `maxTotalThreadsPerThreadgroup` is below the requested size (guard G1).
4. **Verify** (§6.4), then **time best-predicted first**.
5. **Branch and bound.** Stop when every untimed candidate's DRAM floor is above the best measured median, because none can win.

The model also allocates the budget across instances, in the manner of Ansor's task scheduler (R7 §2): time goes first to the instances with the largest share of predicted step time. For Qwen3-32B decode those are the gate/up, down and lm_head GEMVs, about 80% of the bytes (R7 §10.2).

## 6. The database: format, keys, verification, invalidation

### 6.1 Format

The database stays the file the spec defines ("Hardware and tuning — Guarantees"):
- a line-oriented text file, `sky-tune-db 1`, then one `result` line per entry;
- bounded: at most 1 MiB, 1,024-byte lines and 4,096 records;
- tokens of at most 64 characters from `[A-Za-z0-9_.^+:=-]`;
- integer parameter values of magnitude at most 2⁴⁰;
- never code.

GPU records use the existing fields plus new keys. The version-1 parser already ignores unknown keys "so a later minor addition to version 1 does not make older readers drop the result" (`sky-tune/src/db.rs:338-340`, VERIFIED). A GPU fingerprint is 16 hex digits like a CPU one, but it hashes a different text (prefix `sky-hw-gpu-fp1`), so it does not match any CPU fingerprint and an older Sky reads the line without ever applying it.

One record, wrapped here for width:

```text
result fp=9c41e2d07a113f5b kernel=qgemv.t3 size=q4g64.n10240.k5120.m1-8 params=r:4,s:32,v:16,ks:1,xs:0,ar:1
       default=r:4,s:8,v:16,ks:1,xs:0,ar:1 best_ns=55210 default_ns=57990 when=1790000000
       hw=3f0a51c9b2e4d687 os=25G42 contract=inference order=7d2e90a1c35b4f68 check=0be3f1a9d4c27e55 err=113
```

| Key | Content | Validation |
|---|---|---|
| `fp` | GPU fingerprint: FNV-1a over `sky-hw-gpu-fp1;name=Apple M5 Max;family=apple10;cores=40;os=<build>;metalc=<id>` | 16 hex digits; exact match required for use |
| `hw` | the same hash without OS and compiler: identifies the hardware alone, for transfer (§8) | 16 hex digits |
| `kernel` | skeleton and **template version** (`qgemv.t3`) | token; the version bumps whenever the generator's output or a reduction order changes |
| `size` | format, shape and width range (`q4g64.n10240.k5120.m1-8`) | token ≤ 64 characters |
| `params` | up to 8 integer parameters | `MAX_PARAMS` = 8 (`params.rs:7`); clamped on read, and `lookup_among` accepts only a value the skeleton offers |
| `contract` | `inference` or `portable` | enumerated |
| `order` | hash of the OC parameter values: the `ReductionSchedule` | 16 hex digits; plans cite it |
| `check` | hash of the outputs on the canonical verification inputs (§6.4) | 16 hex digits |
| `err` | maximum relative error against the reference, in units of 10⁻⁶ | integer ≤ 2⁴⁰ |
| `os` | the OS build, readable copy of what `fp` hashes | token |

The calibrated GPU-model parameters of [Part XXII](part22-performance-model.md) §4 are records in the same file, with `kernel=gpumodel.v1`.

**Where the Metal compiler identity comes from.** Runtime MSL compilation uses the OS's own compiler service (R6 §A3), so the OS build (`kern.osversion`) identifies the compiler. If an AGX driver bundle version is readable, it is hashed in too. `[LIKELY: R6 §A3; MLX's NAX miscompile was tied to a Metal front-end version, mlx#3586, R3-kernels §d]`

### 6.2 Who writes it

The spec says `sky tune` is the only command that writes the database. SPINE D11's `-O3` "tunes missing entries", which needs a write. We propose a one-sentence spec amendment: "`sky tune`, and the explicit tuning flags `-O3` and `--autotune=overnight` of `sky run` and `sky compile`, are the only writers." Each is an explicit user request to tune.
- `sky run` at its default `-O2` never writes.
- `sky check` and `sky repl` never write.
- Nothing tunes behind the user's back, so the spec's "not session saving" argument still holds.

### 6.3 Lookup

A plan asks for (fp, kernel with template version, size, contract):
- A hit is used after clamping.
- A miss falls back to the analytic default, the best predicted candidate under the model, at `-O1` and `-O2`; at `-O3` it triggers the search.
- An entry whose `contract` does not match the requested numerics is a miss. A `portable` run never uses an `inference` entry.

### 6.4 Verification before storing

No candidate is timed, let alone stored, until it passes these checks. They implement R7 §12.2's distilled rules.

1. **Reference.** Outputs are compared with the TIR reference interpreter, in f32, with f64 accumulation for the check ([Part VII](part07-tensor-ir.md)). Never with another candidate's outputs.
2. **Inputs.**
   - Real weight statistics: rows of the actual model's weights.
   - Activations drawn from recorded hidden states, plus adversarial ones: large magnitudes, ±0, values near fp16 overflow.
   - Every M in the instance's range (1…8 for a verify invariance group) and edge shapes: K not a multiple of 512, N not a multiple of the tile.
   - Inputs are generated after compilation, so no kernel can specialize on them.
3. **Poisoned outputs.** Output buffers are filled with NaN before each run, so a kernel that writes nothing fails.
4. **Tolerance.** The contract's per-op tolerance from [Part VII](part07-tensor-ir.md): for a 4-bit GEMV, the error of an fp32-accumulated dot product of length K relative to the f64 reference. Exceeding it rejects the candidate.
5. **Invariance.** For OC candidates, rows computed at M = 1 must equal the same rows at every other M of the group bit for bit. For prefill, the same across chunk buckets.
6. **Determinism.** Two runs on fresh buffers give identical bits.
7. **Record.** `check` and `err` are stored with the entry. On load, `sky tune --verify` reruns step 4 for entries whose `check` it cannot reproduce. An OS update changes `fp`, so old entries are not used at all without re-verification.

Only OP parameters can skip steps 4–5, because they cannot change bits. Their `check` hash must equal the default candidate's. That is `sky-tune`'s existing "same answer for every candidate" test (`measure.rs`, `Tunable::run`), unchanged.

### 6.5 Invalidation

| Change | Effect | Mechanism |
|---|---|---|
| OS update (new Metal compiler) | all GPU entries for this machine unused until re-tuned or re-verified | `fp` includes the OS build |
| Sky release with a changed skeleton or generator | that skeleton's entries unused | template version in `kernel` |
| Sky release with unchanged templates | entries stay valid | template versions unchanged |
| Different chip | entries unused; they may seed transfer (§8) | `fp` and `hw` differ |
| Model swap with the same shapes | entries reused | `size` describes shapes and formats, not models |
| Numerics contract | separate entries | `contract` |
| Database full (4,096 records) | oldest dropped | existing `Db::insert` policy |

`sky tune --show` lists GPU entries grouped by `hw`, marking those whose `fp` no longer matches. `sky tune --prune` (new) removes them.

## 7. Optimization levels and tuning time

| Level | Behavior | Expected time, Qwen3-32B on M5 Max | Writes the DB |
|---|---|---|---|
| `-O1` | analytic defaults only: the best candidate predicted by [Part XXII](part22-performance-model.md)'s model, the scheduler's top partition | none | no |
| `-O2` (default) | DB entries where present, analytic defaults otherwise | none | no |
| `-O3` | `-O2`, then search the missing entries of **this plan's** decode and verify kernels and block choices; prefill uses defaults unless already tuned | 2–10 minutes | yes |
| `--autotune=overnight` | everything: all prefill buckets, all verify widths, model-level choices, an exhaustive search where it fits | 2–5 hours | yes |

Estimates follow R7 §10.2 `[EXPERIMENTAL]`:
- **Decode GEMVs:** about 100 pruned configurations × 5 shapes at 0.1–0.2 s each, compile and timing included: under 2 minutes. With the 7 verify widths sharing OC choices, about 5 minutes.
- **Prefill GEMMs:** about 300 configurations × 45 shapes at 0.2–0.6 s: 1–2 hours.
- **Attention:** a few hundred configurations: minutes.
- **Block level:** 3–6 candidates × about 2 s of full-step replay per specialization: under a minute for decode.
- **Noise control:** interleaving and repeats double or triple the total, giving 2–5 hours overnight.
- **Cold compilation is part of the budget.** Tensor-op pipelines take about 1–2 s each to compile (R6 §A3), and the asynchronous compiler overlaps compilation with timing.

Expected gains (R7 §10.2, LIKELY):
- 0–5% on bandwidth-bound GEMVs that already run near 85–90% of bandwidth;
- 5–30% on small-M prefill GEMMs and attention chunking against fixed heuristics;
- the main value is **robustness** across M5 Pro, Max and Ultra, older chips, and OS and compiler updates.

MLX's fixed routing tables show what a tuner fixes. On the M5 Max, quantized projections at M = 13–32 other than gate/up and lm_head fall to a split-K kernel without neural-accelerator support. A maintainer declined to retune it: "we are unable to optimize for all possible cases" (mlx#4198, R3-kernels §d).

## 8. Transfer across chips

| From → to | What transfers | Procedure | Time |
|---|---|---|---|
| Same chip, new OS | everything, after re-verification | re-run the verification of §6.4 on the stored winner and the default, re-time the top 3 | minutes |
| Within a family (M5 Pro ↔ Max ↔ Ultra) | OP parameters scale with cores and bandwidth; OC choices usually stay good but must be re-verified | seed from the nearest `hw` in the family; re-rank with the target chip's calibrated model; time the top 3 per instance | about 10% of a full search |
| Across generations (M2 → M5) | little: the neural accelerators and dynamic caching change the space | seed the candidate order only | a full search |
| Shipped defaults | per-family tables in Sky releases, in the same data-only format, layered read-only under the user's DB | Sky's CI farm runs `--autotune=overnight` on each supported chip | — |

llama.cpp ships exactly one offline table, for its attention kernel, keyed by Apple GPU family, with 366 rows for Apple10 (R5 §4). It shows that per-family tables are the practical form of transfer. Learned cross-device models such as TLP (7% of target data, R7 §2) solve a harder problem than Sky has: about 60 instances per model, not thousands of open-ended tasks.

## 9. Profile-guided choices

Profiles are histograms collected while running. They are stored as data with the plan cache, never with prompts or outputs.

| Profile | Collected by | Drives |
|---|---|---|
| **MoE expert usage** | router top-k histograms over a calibration corpus | on small-RAM Macs, which experts stay resident and which are paged; gather order in verify passes, where verifying k tokens touches the union of their experts (2.57× the bytes at k = 4 for Qwen3-30B-A3B under independent routing, R8 §7.3); tuning buckets for the per-expert GEMV batch sizes that actually occur |
| **Speculation acceptance** | per-position acceptance on calibration prompts per workload class (chat, code, summarization) | γ and tree shape: maximize S with the measured β curve ([Part XXII](part22-performance-model.md) §9); the per-request adaptive policy is [Part XXI](part21-speculative.md)'s |
| **Context and prompt lengths** | position and prompt-length histograms | which context buckets and prefill chunk buckets get tuned and compiled |
| **Sampling settings** | temperature, top-k, top-p in use | which sampling kernels exist in the plan |

MoE needs the profile more than dense models do. One study found that a 512 MiB LRU expert cache got 0% hits on an iPhone, while random eviction got 18.8% (R8 §7.4): the policy, which the profile chooses, matters more than the capacity.

## 10. A shared performance database, and its privacy questions

**Proposal.** An opt-in exchange of tuning records:
- `sky tune --share` uploads this machine's GPU records;
- `sky tune --fetch` downloads a signed bundle for this machine's `hw` and OS build.

Records arrive in the same data-only format, are verified locally (§6.4) and re-timed (top 3) before use. A bad record can therefore make Sky slower or be rejected. It cannot move a result outside the contract.

**What a record reveals, and the mitigations:**

| Field | Reveals | Mitigation |
|---|---|---|
| `size` shapes | which model architectures the user runs (5120 × 25600 identifies Qwen3-32B) | share only shapes on a published list of public architectures; never shapes of private models |
| `fp`, `os`, timings | hardware model and OS build; timing noise could add to a device fingerprint | upload `hw` and the OS build only; bucket timings to 1%; no machine serial or user identifiers |
| acceptance profiles | the distribution of the user's text | never shared |
| prompts, outputs, weights | — | never collected |

Open questions for the owner, not engineering:
- who hosts the exchange and signs bundles;
- the consent wording;
- the legal review of what counts as personal data (the hardware and OS model of a named account may);
- whether anonymous submissions are accepted at all.

Until decided, Sky ships only the per-family default tables of §8, built on Sky's own machines.

## 11. The self-optimizing loop

```text
 run ──► sample 1 step in 1,000 with per-kernel GPU timestamps ──► residual vs Part XXII prediction
  ▲                                                                        │
  │                 residual > 15% for 3 sampled steps, or fp changed      │
  │                 (OS update), or a new model / chip                     ▼
  └── use new entries ◄── store ◄── verify ◄── search ◄── user approves `sky tune` (or opted in to
                                                          --autotune=background: AC power, nominal
                                                          thermal state, machine idle)
```

- **Telemetry.** A sampled step costs one extra timestamp per kernel, about 386 for Qwen3-32B. It stays in memory, and feeds `sky profile` and the residual check.
- **Refitting.** The GPU-model parameters (`t_fill`, `BW_stream`) are refit from sampled steps in memory. They are persisted only by `sky hw --calibrate-gpu` ([Part XXII](part22-performance-model.md) §4).
- **Consent.** The loop proposes; it does not act. Writing the database stays an explicit user action (§6.2). `--autotune=background` is a persistent opt-in that the user sets, and `sky tune --show` lists what it did.

## 12. AI-proposed variants, under verification

Language models can propose new skeleton variants: an inner loop for the verify GEMV, a different dequantization sequence, a new attention tiling. This happens **offline, in Sky's development process**, never on a user's machine. The proposals enter the same loop as hand-written candidates (R7 §12.3):

```text
for each hot skeleton instance (shape bucket, chip):
    candidates = skeleton × parameters  ∪  model-proposed MSL variants (as new templates)
    for c in candidates:
        compile c; reject on error or guard failure (Part IX §5)
        verify c (§6.4: reference, poisoned outputs, adversarial inputs, tolerance, invariance)
        time c (§4: rotated buffers, GPU timestamps, chain slope, medians, DRAM-floor check)
    keep the Pareto front (time, error); store trace + check hashes
a variant that wins on ≥ 2 chips → code review, licence/provenance check (SPINE D15), template version bump
```

**Why the verification is not optional:**
- **Sakana's "AI CUDA Engineer"** claimed 10–100× speedups. Its evaluation could be bypassed; a claimed 150× kernel was 3× slower (R7 §12.1).
- **KernelBench-style checks** were fooled by fixed shapes and hard-coded outputs into fake 50–120× results (robust-kbench, R7 §12.1).
- **On Metal,** Gimlet Labs' agent kernels beat PyTorch's eager MPS backend by 1.22–1.87×, against a weak baseline, and the authors themselves excluded reward-hacking kernels that always output zeros (R7 §12.1).
- **What worked** were language models as proposers inside a verified harness: AlphaEvolve's 23% on a tiling heuristic (R7 §12.1). Sky takes that pattern and no other.

**Limits of superoptimization here:**
- **Mirage** finds fused kernels by search, but takes up to 4 hours per pattern, targets NVIDIA, and verifies only a multilinear fragment without operators such as ReLU (R7 §6.1).
- **Graph-level saturation** (TASO, Tensat) gains single digits to about 16%. In production, Cranelift's multi-version e-graph bought about 0.1% (R7 §6.2–§6.3).
- **The decode graph is small and repetitive.** The useful rewrites are a handful of known identities ([Part VIII](part08-graph-optimizer.md)).
- **MPP is opaque.** Nothing below a `matmul2d` call can be searched or verified exactly (R6 §A2).
- **OC changes can only be verified to a tolerance**, not proven equivalent. So any bounded search (RMSNorm with GEMV, split-K attention) runs in Sky's build farm, never in the user's compile path (R7 §6.5).

## 13. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| X-T1 | GPU fingerprint | `sky-hw/src/gpu.rs` | `GpuFacts`, `gpu_fingerprint()`, `hardware_fingerprint()` | Stable across runs; changes when the OS build string changes (test with an injected build string); differs from the CPU fingerprint on every machine in CI |
| X-T2 | GPU records in the DB | `sky-tune/src/{db.rs,gpu.rs}` | new keys (`hw`, `os`, `contract`, `order`, `check`, `err`) parsed and validated; version stays 1 | An old-format reader (today's `parse_result`) loads a GPU line without warnings and never matches it; bounds fuzzed for 10⁶ inputs; spec's malformed-line rules hold |
| X-T3 | GPU tunables | `sky-tune/src/measure.rs`, `sky-metal/src/tune.rs` | `GpuTunable` with OP/OC classification, invariance groups, chain-slope timing on GPU timestamps, rotation, warm-up, interleaving | On the M2 Pro, re-tuning the same instance three times gives the same winner or winners within 3% of each other |
| X-T4 | Verification harness | `sky-tune/src/verify.rs`, `sky-tir` reference | §6.4 steps 1–7 | A kernel that writes nothing, one that hard-codes outputs and one with an off-by-one tail are all rejected; an OC candidate with M-dependent rows is rejected |
| X-T5 | Pruning and budget allocation | `sky-tune/src/search.rs` | §5 guards, predicted cut, branch and bound, Ansor-style allocation | On a recorded search, pruning keeps the true winner in ≥ 95% of instances while compiling ≤ 30% of candidates |
| X-T6 | DRAM-floor check | `sky-tune/src/measure.rs` | rejects sub-floor timings | Re-creating tinygrad's mistake (a 28 MB matrix timed warm) is flagged |
| X-T7 | Block-level tuning | `sky-tune/src/block.rs`, `sky-llm` replay harness | full-step replay of tied partitions | Chooses between head-aligned K1 and P1b on the M2 Pro; result reproducible across three runs |
| X-T8 | Levels and CLI | `sky-cli` | `-O1/-O2/-O3`, `--autotune=overnight`, `sky tune --model`, `--prune`, `--verify` | `-O2` never writes (file unchanged byte for byte); `-O3` writes only entries of the current plan; spec amendment of §6.2 merged |
| X-T9 | Model-level tuning | `sky-llm/src/tune.rs` | γ/tree, chunk size and b_kv selection from traces and replays | On recorded acceptance traces, the chosen γ is within one step of the measured best |
| X-T10 | Transfer and shipped tables | `sky-tune/src/transfer.rs`, CI farm job | §8 seeding; per-family default tables | Seeded M5 Pro tuning from M5 Max entries reaches within 3% of a full search at ≤ 20% of its time |

## 14. Measurements that decide

- **Tuning gain over analytic defaults**, per kernel class, on the M2 Pro (G4) and the M5 Max (G5). R7 expects 0–5% for decode GEMVs and 5–30% for small-M prefill. If the decode gain is under 1%, `-O3` should skip the decode GEMVs.
- **Run-to-run stability of winners.** If three tunings of one instance disagree by more than 3%, the timing protocol is not good enough, whatever the winner.
- **The cost of `portable`.** Decode, verify and prefill speed with OC parameters fixed at canonical values against `inference`'s tuned values. This decides whether `portable` is a practical mode or a test mode (SPINE D6, EXPERIMENTAL).
- **Transfer quality within the M5 family**: seeded against full search.
- **Re-verification after an OS update:** how often stored OC entries change bits or fail tolerance.

## 15. Risks and negative results

- **Measurement error outweighs tuning gain** for bandwidth-bound decode GEMVs, where the expected gain is 0–5% and noise is 1–3%. §4's protocol exists for this; if it is not enough, `-O3` tunes only GEMMs and attention.
- **Opaque MPP lowering.** A macOS update can change `matmul2d`'s speed and its bits. The fingerprint forces re-verification, but a regression can still appear between an update and the user's next `sky tune`. MLX hit a miscompile of its NAX GEMM with one Xcode release (mlx#3586, R3-kernels §d).
- **Invariance groups shrink the search.** Sharing OC parameters between M = 1 and M = 8 may cost the decode GEMV a few percent against tuning it alone. That is the price of exact speculation.
- **The database is small:** 4,096 records. About 60 instances × chips × OS builds fit, but a user who runs many models across OS updates will evict entries. `--prune` and the per-family default tables limit the damage.
- **Negative result:** the autotuner cannot recover what the format or the hardware forbids. It will not make a 4.5-bit decode beat the bandwidth ceiling ([Part XXII](part22-performance-model.md)), and it cannot see below MPP.
