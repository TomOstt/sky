# Part XXIV — The first ten implementations

This part says exactly what to build first, in order, and how to know each piece works. It covers gates G0–G4 of the plan, about the first fourteen weeks; [Part XXV](part25-roadmap-12-months.md) continues from there.

For each item:
- the crates and files;
- the architecture;
- pseudocode where it helps;
- the expected benchmark;
- the success criterion;
- the test strategy;
- the failure modes.

Every number comes from the research measurements on the M2 Pro development machine (R3, R4, R5, R6) or from [`roofline.py`](roofline.py). Every expected benchmark below is a target, `[EXPERIMENTAL]` until measured; the measurement that decides it is the item's success criterion.

## Order and parallelism

```text
week   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 #1  harness+baselines ███
 #2  sky-metal             ████████
 #3  quant GEMV family               ██████████                        (needs #2)
 #4  TIR core              ██████████
 #5  loading + relayout       ████████
 #6  Qwen3 builder+oracle               ███████                         (needs #4, #5)
 #7  tokenizer+template      █████
 #8  unfused executor                          ███████████              (needs #3, #6, #7)  → G3
 #9  decode fusion+plan+replay                            ██████████   (needs #8)          → G4
 #10 GQA attention+sampling+exactness                     ██████████   (needs #8)          → G4
```

Items 1, 2, 4, 5 and 7 have no dependencies on each other and start in parallel, one agent each. The critical path is #2 → #3 → #8 → #9.

## #1 Benchmark harness and baselines (gate G0)

**Why first.** Nothing in this report is a result until it is measured with a protocol a hostile reviewer accepts. The competitors' own tools report different things:
- mlx-lm's `generation_tps` counts n+1 tokens over n intervals;
- llama-bench's `tg` decodes from an empty cache without sampling (R9 §6).

**Where.** A separate `sky-evaluation` repository from G0. It is private from week 2 and public at G8, as Part XXIII and [Part XXV](part25-roadmap-12-months.md) §16.2 decide. A separate repository lets outsiders run it without building Sky. `sky bench llm` in the Sky repo drives the Sky side.
- `harness.py`: the driver. Python is acceptable here, because it is outside the measured process.
- `engines/{mlx_lm,llama_cpp,sky}.py`
- `schema.json`: R9 §8.
- `protocol.md`: R9 §7.

**Architecture.**
- One harness drives every engine as a separate process and timestamps every token id as it arrives.
- A metric sampler runs alongside:
  - system memory used (wired + anonymous − purgeable + compressed), sampled with `host_statistics64`;
  - system power from the SMC `PSTR` sensor, sampled at 1 s;
  - thermal state.
- Runs are interleaved in randomized blocks after a quiet-machine gate.

```text
for block in shuffle(engines × workloads), repeated ≥ 5:
    wait_until(quiet_machine)                   # load average, no Spotlight, AC power
    baseline = system_memory_used()
    p = spawn(engine, model_file, prompt_ids, settings)   # same file, same ids
    record t0; for each token id from p: record (id, t)
    record peak(system_memory_used() - baseline), energy(PSTR over the run)
quality gates first (R9 §7): tokenizer ids identical; KLD(engine ‖ other) ≤ 0.1 × KLD(other ‖ BF16); ≥ 99% top-1 agreement
emit JSON (schema.json): ttft, prefill = n_prompt/ttft, decode = (N-1)/(t_last-t_first), peak_ram, joules/token
```

**Expected result (M2 Pro), from R4 and R5:**

| Engine | Model | Decode | Prefill |
|---|---|---:|---:|
| mlx-lm | Qwen3-8B 4-bit | 37–39 tok/s | ~218 tok/s at 2K |
| mlx-lm | Qwen3-0.6B 4-bit | ~267 tok/s | — |
| llama.cpp | Qwen3-8B Q4_0 | 34.8 tok/s (tg128) | 312 tok/s (pp512) |

**Success criterion.**
- The harness reproduces those numbers within 5%.
- Its results are byte-stable in schema.
- A second run on another day agrees within the reported confidence interval.

**Tests.**
- A fake engine that emits tokens at a known rate checks the arithmetic of every metric.
- The memory sampler is checked against a 1 GiB mmapped file read by the GPU (R9's probe).

**Failure modes.**
- `PSTR` unavailable on a machine: use an external meter. `powermetrics` chip counters are recorded only as diagnostics, never reported as energy (D13).
- Swap during runs (the dev box had 4.1 GB in swap; R9): the quiet gate must refuse to run.

## #2 `sky-metal`: the Metal foundation (gate G1)

**Where.** `crates/sky-metal/src/`:
- `{device.rs, buffer.rs, library.rs, pipeline.rs, encode.rs, timing.rs, residency.rs}`
- `crates/sky-metal/benches/` (ported from [`research/microbench/`](research/microbench/))

**Architecture** ([Part XI](part11-metal-backend.md)):
- `objc2-metal` 0.3.2 (R6).
- **Buffers**:
  - `Buffer::shared(len)`;
  - `Buffer::no_copy(mmap: &Mmap, range)`: page-aligned; split at `maxBufferLength` (8.88 GiB on the M2 Pro).
- **Libraries**: compiled from MSL text with `makeLibrary(source:)`. Keyed by a hash of the source and the OS/Metal-compiler version; compiled code is reused through Metal's OS shader cache. Binary archives appear only inside an explicit `sky compile` `.skybin` (D19).
- **Encoding**:
  - Metal 3 path first. Metal 4 is not faster per dispatch (R6); its reusable command buffers come later in #9.
  - Dispatches go to a concurrent encoder with explicit barriers.
- **Timing**: GPU timestamps per command buffer, or per stage boundary. Apple GPUs have no per-dispatch counters (R6).
- **Residency**: a residency set plus a heartbeat (R5).

```rust
let dev = Device::system_default()?;                       // sky-hw records family, cores, NA
let lib = dev.library(&msl_source, LangVersion::V3_2)?;    // cached by (hash, os_build)
let pso = lib.pipeline("qgemv_q4g64", &FunctionConstants::new().set(0, rows_per_sg))?;
let w   = Buffer::no_copy(&weights_mmap, offset..offset + len)?;
let mut cb = dev.queue().command_buffer();
let mut enc = cb.compute_concurrent();
enc.set_pipeline(&pso); enc.set_buffers(&[&w, &x, &y]); enc.dispatch(grid, tg);
enc.barrier_buffers(&[&y]);                                // only where the plan says
enc.end(); cb.commit(); cb.wait();  let ns = cb.gpu_time_ns();
```

**Expected benchmark (M2 Pro), reproducing R6:**

| Measurement | Expected |
|---|---|
| Read bandwidth | ≥ 190 GB/s (R6: 196) |
| Copy | ≈ 183 GB/s |
| Dependent dispatch | ≈ 1.2 µs |
| Independent dispatch | ≈ 0.14 µs |

**Success criterion.**
- The ported microbenchmarks reproduce R6 within 10%.
- `cargo test -p sky-metal` runs on the macOS CI runner (VI-T2).

**Tests.**
- Every buffer path is written and read back.
- An unaligned no-copy buffer is refused at creation. API validation would catch it otherwise, but production must never reach it (R9 §2).
- Pipeline-cache hit and miss.

**Failure modes.**
- `objc2-metal` lacks the OS 26.4/27 enums (R6): use the raw values.
- Watchdog kills under a busy display (R6): keep command buffers short; document `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` as an observed llama.cpp practice, not a fix.

## #3 The quantized GEMV family (gate G1)

**Why.** Decode is weight streaming, so this kernel is most of the decode time:
- 23.9 of 25.8 ms per 8B token in mlx-lm (R4).
- Its small-M variant decides whether speculation pays (Part IV §8).

**Where.**
- `crates/sky-kernel/src/skeletons/qgemv.rs`: parameters and MSL template. Hand-written MSL first; migrated into L3 once the kernel IR exists.
- `crates/sky-metal/benches/qgemv.rs`

**Design** ([Part XVI](part16-decode.md), [Part XII](part12-quantization.md)):
- **Format**: MLX affine, 4 bits, group 64, bf16 scales and biases. Scales may be negative: 50.25% are (R9).
- **Algorithm**:
  - Each simdgroup owns r output rows. Its 32 lanes split K into 16-value chunks in a **fixed order that does not depend on M**.
  - Activations are pre-scaled by 1, 1/16, 1/256 and 1/4096, so nibbles are masked and never shifted. This is MLX's `qmv_fast` trick (R3), also used by llama.cpp's Q4_0 kernel (R5).
  - Each group contributes `s · Σ(q·x) + b · Σx`.
  - The sum is in fp32, then one `simd_sum` per row and per activation row.
- **M = 1…16 in one template.** The K-partition and reduction tree are identical for every M, so row m's result is bit-identical to the M = 1 result. This is **batch invariance by construction** (D6). Only the number of accumulators changes. Registers are bounded by keeping activations in fp16 and lowering r as M grows: R3's 60-line test kernel ran out of registers from M = 4 with r fixed.

```text
kernel qgemv_q4g64<M, R>(W: packed nibbles [N][K/2] (relayout: row-tile contiguous),
                         S, B: bf16 [N][K/64], x: half [M][K], y: half [M][N]):
  rows = my R rows                                   # threadgroup = (R / r) simdgroups
  acc[M][r] = 0 (fp32)
  for k0 in (lane*16) .. K step 512:                 # fixed order, independent of M
      for j in 0..r:
          w   = load8B(W[row_j], k0)                # 16 nibbles; adjacent chunks may be fetched as one 16 B load
          s,b = S[row_j][k0/64], B[row_j][k0/64]
          for m in 0..M:
              xs  = prescaled(x[m], k0)             # x * {1,1/16,1/256,1/4096}, cached per lane
              acc[m][j] += s * dot_masked(w, xs) + b * xsum(x[m], k0)
  for m, j: y[m][row_j] = simd_sum(acc[m][j])        # same tree for every m
```

**Expected benchmark:**

| Chip | Shape / M | Target | Reference point |
|---|---|---|---|
| M2 Pro | 32B shapes at M = 1 | ≥ 185 GB/s, about 95% of raw read | mlx `quantized_matmul` reaches 162–177 GB/s batched (R3); a simple raw-Metal 4-bit GEMV reaches 183–191 (R6) |
| M2 Pro | k/v_proj (5120 → 1024) | ≥ 150 GB/s | mlx 114 GB/s (R3) |
| M2 Pro | M = 8 | ≤ 1.5× the M = 1 time | ALU ridge is about 8 rows, so this is at the edge |
| M5 Max (G5) | M = 8 | ≤ 1.3× the M = 1 time | ridge 8.7 on the ALUs (Part IV) |

**Success criterion (gate G1).**
- ≥ mlx `quantized_matmul` on every Qwen3-8B and 32B shape at M = 1.
- ≥ 88% of measured raw-read bandwidth on the large shapes.
- The batch-invariance test passes.

**Tests.**
- **Oracle**: the reference interpreter (f32) on random inputs, within a tolerance derived from the fp16 input rounding.
- **Dequantization**: bit-exact against `mx.dequantize` on real Qwen3-8B tensors.
- **Batch invariance**: for M = 2…16, every row is bit-identical to the M = 1 result for the same input row.
- **Fuzzing**: random shapes (K a multiple of 64; N any) with poisoned output buffers.
- **Timing**: DRAM-sized rotating weight sets, many calls per command buffer, GPU timestamps, medians. This avoids tinygrad's BEAM mis-ranking (R2) and MLX's single-eval floor (R3).

**Failure modes.**
- The relayout (row-tile-contiguous nibbles with adjacent scales) gains nothing over the original layout. Then keep the original layout and drop relayout from the hot path; the alignment cache from #5 is still needed.
- M ≥ 8 spills registers. Then split the verify widths into two templates.

## #4 TIR core (gate G2)

**Where.** `crates/sky-tir/src/`:
- `{graph.rs, types.rs, dtype.rs, shape.rs, op.rs, verify.rs, text.rs, interp/}`

**Design** ([Part VII](part07-tensor-ir.md)):
- **Graph**: a hash-consed SSA graph in an arena. Types are derived from operands; tinygrad's single interned UOp is the model (R2).
- **`Type { dtype, shape, layout }`**:
  - `DType` includes `Quant(QFormat)`, the descriptor of [Part XII](part12-quantization.md);
  - `Dim` is `Const(u64)` or `Sym(Poly)` with interval bounds, so that `pos: Int<0..MAX_CTX>` exists.
- **Ops**: the minimal set Qwen3 needs, each with a reference decomposition:
  - elementwise, cast, reduce, matmul (with a quantized operand);
  - views (reshape, permute, slice, broadcast);
  - gather;
  - functional `update` (KV append);
  - composites: `rms_norm`, `rope`, `attention`, `swiglu`, `softmax`, `argmax`, `sample`.
- **Verifier**: a total check of shape, dtype and format rules.
- **Text format**: round-trips.
- **Reference interpreter**: f32, with f64 checks.

```rust
let mut g = Graph::new();
let x   = g.param("x", Type::tensor(F16, [Dim::sym("T"), 5120.into()]));
let w   = g.param("w_q", Type::tensor(Quant(QFormat::mlx_affine(4, 64, BF16)), [8192, 5120]));
let n   = g.rms_norm(x, g.param("w_in", Type::tensor(BF16, [5120])), 1e-6);
let q   = g.matmul(n, g.transpose(w));                 // [T, 8192]
g.output("q", q);
assert!(g.verify().is_ok());
println!("{}", g.to_text());                            // stable, parseable
```

**Success criterion.**
- 100% of the op set verified and interpreted.
- Text round-trip on every test graph.
- A differential property test: random small graphs give the interpreter's f32 result equal to a NumPy reference (run as a dev-only test).

**Failure modes.**
- `Poly` from `sky-types` lacks interval bounds. Extend it in `sky-tir` rather than changing scientific Sky.

## #5 Model loading: hub, config, safetensors and the relayout cache (gates G2/G3)

**Where.**
- `crates/sky-hub/`: resolve, pin, shared cache.
- `crates/sky-llm/src/{config.rs, load/safetensors.rs, load/relayout.rs}`
- `crates/sky-quant/src/mlx_affine.rs`

**Design** ([Part XIII](part13-llm-runtime.md), [Part XIX](part19-memory.md)):

- **`hf://org/model[:QUANT]`**:
  - offline-first;
  - shares `~/.cache/huggingface/hub` with flock-compatible locking;
  - never writes to the shared Xet blob store;
  - records the commit (`X-Repo-Commit`) and the sha256 (R9 §10).
- **Config normalization**:
  - accept both `rope_scaling` and `rope_parameters`;
  - fail loudly on unknown keys. mlx-lm silently drops them, and a Llama 3.1 config re-saved in the new format loads with the wrong θ (R9 §11).
- **MLX quantization inference**: follow mlx-lm's per-module rules exactly (R9 §3).
- **Relayout cache.** mlx-community files are unaligned (data at offset ≡ 6 mod 8), so they cannot be used zero-copy (R9 §1–§2). On first use, Sky writes `~/Library/Caches/sky/weights/<key>.skyweights` (under `XDG_CACHE_HOME` when set, like the tuning DB; `SKY_CACHE_DIR` overrides the root, [Part XIII](part13-llm-runtime.md) §3.2). The key includes the source sha256:
  - page-aligned;
  - lossless;
  - optionally in #3's row-tile layout.

  It is then mmapped into no-copy buffers.

```text
relayout(src.safetensors) -> .skyweights:
    hdr = parse_header(src)                       # names, dtypes, offsets (unaligned)
    plan = for t in hdr.tensors: dst_off = align_up(cursor, 16384); cursor = dst_off + size(t)
    write header{sha256(src), sky_version, layout_version, plan}
    for t: copy (or re-tile) bytes → dst_off             # streaming, bounded memory
    fsync; rename into place                              # atomic; concurrent runs safe
```

**Expected benchmark.**
- Qwen3-8B (4.6 GB): relayout bound by SSD write speed, a few seconds, once.
- Later loads are mmap only (milliseconds).

**Success criterion.**
- For every tensor of `mlx-community/Qwen3-8B-4bit` and `Qwen3-0.6B-4bit`, Sky's dequantized values are bit-equal to `mx.dequantize`.
- This is the "same quantization" proof of SPINE D5.

**Failure modes.**
- The disk is too small for the cache. Fall back to copying into Metal buffers, as MLX does, with a warning.

## #6 The Qwen3 builder and oracle parity (gate G2)

**Where.** `crates/sky-llm/src/arch/qwen3.rs`, `crates/sky-llm/tests/oracle_qwen3.rs`

**Design.**
- `build(config) -> Graph` emits a decoder layer 64 times (36 for 8B, 28 for 0.6B) as TIR:
  - embedding gather;
  - the 64 (36, 28) decoder layers;
  - final norm and output head.
- Two programs: prefill (symbolic `T`) and decode (`T = 1`). Both take KV state values.

**Expected result.**
- Qwen3-0.6B on the reference interpreter is slow (seconds per token). This item is about correctness.

**Success criterion.**
- Logits for 8 fixed prompts match HF transformers (fp32, CPU) within a tolerance set by the bf16 storage of the weights.
- Greedy tokens agree for 32 steps, except at documented near-ties.

**Tests.** Reference logits are generated once with transformers and stored by sha256 under `tests/data/`.

**Failure modes.** Disagreements traced to configuration rather than the IR:
- RoPE variants;
- the QK-norm placement;
- the EOS set.

## #7 Tokenizer and chat template (gate G3)

**Where.** `crates/sky-llm/src/{tokenizer.rs, template.rs}`

**Design** (R9 §9):
- **Tokenizer**: the `tokenizers` 0.23.2 crate, fancy-regex build. It matched Python on 168 of 168 cases.
- **Templates**: minijinja with pycompat, a Python-exact `tojson` and `preserve_order`. These rendered 30 of 30 templates byte-identically.
- **Template pitfalls to handle**:
  - date injection (gpt-oss, Mistral);
  - Qwen3 thinking mode (mlx-lm turns it on by default; Sky makes it explicit);
  - no double BOS when re-encoding rendered text;
  - the EOS set is the union of config, generation_config and tokenizer_config.
- **Streaming detokenizer**: emits only at UTF-8 boundaries.

**Success criterion.** The R9 golden-id corpus and template corpus pass byte for byte.

**Failure modes.** The tokenizer differs from llama.cpp's on Qwen's NFC normalization (R9). Benchmarks therefore exchange token ids, never text.

## #8 Unfused Metal executor and the first generation loop (gate G3)

**Where.**
- `crates/sky-kernel/src/lower_unfused.rs`: one kernel per TIR op.
- `crates/sky-llm/src/generate.rs`
- `crates/sky-cli/src/run_llm.rs`

**Design.**
- Every TIR op lowers to:
  - a skeleton: GEMV from #3, a simple attention kernel, argmax;
  - or a generated elementwise/reduction kernel.
- **Memory plan v0**: one buffer per value, then lifetime reuse.
- **KV**: preallocated for the configured maximum context.
- **Greedy argmax on the GPU.**
- **Loop v0**: one command buffer per token, with a host wait. #9 removes the wait.

```text
sky run hf://mlx-community/Qwen3-8B-4bit "Explain unified memory."
  resolve → load (.skyweights) → tokenize+template → prefill(T = n) → loop:
      decode(T = 1) → argmax on GPU → read token → stream text → stop on EOS / max tokens
```

**Expected benchmark (M2 Pro).**
- 8B decode at about 0.7–0.9× mlx-lm: unfused, about 19 kernels per layer like mlx-lm, but with a host wait each token.
- 0.6B lower, because it is overhead-bound.

**Success criterion (gate G3).**
- Greedy output matches mlx-lm for 128 tokens on 20 prompts. Every divergence is explained by a measured near-tie.
- Decode ≥ 0.7× mlx-lm.

**Tests.**
- The oracle comparison of #6, now on Metal, with a tolerance.
- Golden prompts.

**Failure modes.**
- Greedy divergence from mlx-lm caused by bf16 ties (R8: bf16 and fp16 greedy outputs diverge on 49–100% of prompts across stacks). The gate counts *explained* divergences. It never promises identity with MLX.

## #9 Decode fusion, the static memory plan and replay (gate G4)

**Where.**
- `crates/sky-tir/src/passes/{horizontal.rs, composite.rs}`
- `crates/sky-kernel/src/{schedule.rs, epilogue.rs, prologue.rs}`
- `crates/sky-tir/src/plan/memory.rs`
- `crates/sky-metal/src/replay.rs`

**Design** ([Part IX](part09-scheduler.md), [Part XIX](part19-memory.md), [Part XX](part20-persistent-execution.md)):

- **Horizontal fusion through the relayout:**
  - W_q, W_k and W_v are concatenated into one matrix;
  - W_gate and W_up are interleaved.
  - Same values, new layout (D5).
- **Five or six kernels per layer:**
  - K1: rms_norm prologue + QKV GEMV + q/k-norm + RoPE + KV append epilogue;
  - K2: GQA decode attention (from #10);
  - K3: o_proj + residual;
  - K4: rms_norm + gate/up GEMV + SwiGLU;
  - K5: down + residual.
- **The norm prologue recomputes statistics instead of storing them.** Each threadgroup recomputes Σx² over the 5,120-wide input, which is cached in the system-level cache. That replaces a separate kernel and a round trip through memory. This is a recompute-vs-store choice ([Part IX](part09-scheduler.md)).
- **Static memory plan**: lifetime intervals → arena offsets. There is no allocation per token.
- **Replay**:
  - the decode step is planned once and directly re-encoded each step (tens of µs, hidden behind the GPU; Part XX);
  - K steps are in flight, one command buffer per step;
  - the GPU writes the next token, the position and an RNG counter into a small buffer the next step reads;
  - a stop flag turns the remaining steps into empty dispatches through indirect dispatch;
  - an `MTLSharedEvent` tells the CPU streamer which steps are done.

```text
plan_memory(values):                       # interval graph, best-fit by size
    for v in sort_by(size desc, birth asc):
        v.offset = lowest offset such that no placed u with overlapping [birth,death]
                   intersects [offset, offset+v.size)
    arena_size = max(offset+size)          # explain: every byte, with its owner

decode_loop(K):
    cb = queue.command_buffer()
    for s in 0..K: replay(plan.decode, scalars_buffer)      # same dispatches every step
    cb.encode_signal(event, base_step + K); cb.commit()      # CPU never waits per token
```

**Expected benchmark (M2 Pro).**

| Measure | Target | Basis |
|---|---|---|
| Dispatches per 8B token | ≤ 6 × 36 + a few | mlx-lm uses 693 |
| Non-matmul time | about 2% | mlx-lm's 7.6% (R4) |
| 8B decode | ≥ 1.0× mlx-lm, aiming at 1.05–1.10× | R4, R6 |
| 0.6B decode | ≥ 1.3× | mlx-lm is at 54.5% of bandwidth and CPU-heavy (R4) |

**Success criterion (gate G4).**
- 8B decode ≥ 1.0× mlx-lm; 0.6B decode ≥ 1.3×.
- Zero allocations per token, checked in a test.
- `sky explain` lists every kernel with its bytes and FLOPs.

**Tests.**
- Fused and unfused plans give the same output within the inference contract's tolerance.
- The memory plan never overlaps two live values (property test).
- Replay of 1,000 tokens equals 1,000 single-step submissions bit for bit.

**Failure modes.**
- Register pressure in K1 (GEMV + norm + RoPE + KV append). The scheduler's occupancy guard splits it into two kernels; measure both.
- The stop flag through indirect dispatch costs about 0.15 µs per empty slot (R6). Bound K, for example 8–32.

## #10 GQA decode attention, GPU sampling and the exactness suite (gate G4, prerequisite for G6)

**Where.**
- `crates/sky-kernel/src/skeletons/{attn_decode.rs, sample.rs}`
- `crates/sky-llm/tests/{batch_invariance.rs, spec_exactness.rs}`

**Design** ([Part XV](part15-attention.md), [Part XXI](part21-speculative.md)):

- **Decode attention reads each KV head once for all 8 of its query heads.** MLX re-reads K/V per query head and only deduplicates from 8K context (R3). The context is split into blocks of a **fixed size** (for example 256 positions), each block producing partial (max, sum, output). A combine step reduces them in block order. Results therefore do not depend on the context length's split, and are chunk-invariant.
- **Sampling on the GPU**:
  - argmax;
  - temperature;
  - top-k, top-p and min-p;
  - penalties;
  - keyed Gumbel noise from a counter-based generator (Philox) keyed by (seed, position). The same seed gives the same tokens, and speculative output equals plain output at any temperature.
- **The exactness suite**:
  - logits of a sequence are bit-identical whether it is decoded alone, verified in a batch of M = 2…16, or prefilled in chunks of different sizes;
  - mlx-lm #1470's script, adapted, runs as an adversarial test.

**Expected benchmark (M2 Pro).**
- Decode at 4K context ≥ 1.05× mlx-lm, stretch 1.10×. mlx-lm's extra KV reads stream at about 97 GB/s, 53% efficient (R4). At 4K the KV is a small share of the bytes, so the bound is about 1.04–1.11× ([Part XV](part15-attention.md)). The gain grows with context.
- Sampling is ≤ 1–3% of a token either way: fused sampling alone measured 0.7% on 8B (R4).

**Success criterion.**
- The 4K target above, and the per-context bounds of Part XV met within 3%.
- The bit-invariance suite passes on the M2 Pro.
- Seeded sampling reproduces the same tokens across runs.

**Failure modes.**
- The fixed split size underuses the GPU at short contexts. Use a smaller fixed split size or Part XV's aligned-binary-tree combine ([Part XV](part15-attention.md) §4.3). Split sizes keyed by context bucket are not allowed: they would break Part XV's rule I2.

## After the first ten

At G4 the system decodes Qwen3-0.6B and 8B on the M2 Pro at or above mlx-lm, with a static plan and exactness tests. [Part XXV](part25-roadmap-12-months.md) continues from there:
- G5: prefill with `matmul2d`, the autotuner, and the first M5 Max measurements;
- G6: speculation;
- G7: breadth;
- G8: proof.

If any success criterion above fails, the gate's kill or pivot rule in [Part XXVIII](part28-risks.md) applies before work continues.
