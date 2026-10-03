# Part XIX — Memory architecture

This part specifies how Sky decides where every byte lives: the static memory planner, the rule of zero allocation in the hot path, KV allocation, weight mapping and how to account for it honestly, the wired limit and residency sets, the comparison with MLX's caching allocator, the peak-RAM breakdown for Qwen3-32B, the `sky explain` memory report, memory contracts, and what "24 → 20 GB" can and cannot mean.

The KV layout and formats are [Part XIV](part14-kv-cache.md)'s, the weight formats [Part XII](part12-quantization.md)'s, and the measurement protocol [Part XXIII](part23-benchmarks.md)'s. Residency as it affects latency is in [Part XVIII](part18-ttft.md) §5.

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`. Totals come from [`roofline.py`](roofline.py) (`memory_gb`), which assumes 0.3 GB of scratch and 0.2 GB of runtime; this part replaces the scratch assumption with arithmetic and, later, with the planner's output.

## 1. Summary

- **Weights dominate and are the same bytes in every engine.** Qwen3-32B in MLX 4-bit is 18,429,667,328 bytes (R9 §2.3). Under D5 (same quantization), no engine can save memory on weights; differences come from KV format, context length and everything else.
- **Sky plans memory statically.** For each (model, maximum context, batch, KV format, chip), a planner assigns every buffer an offset in one of a few arenas, from lifetime intervals over the dispatch schedule. Nothing is allocated per token; the plan is printed byte by byte by `sky explain --memory`.
- **KV is preallocated for the declared maximum context** (D9). A growth mode that commits KV in slabs is offered for interactive use, outside the per-token loop.
- **Mapped weights are invisible to per-process metrics.** GPU-only reads of a no-copy mapping show up in no footprint, RSS or `ru_maxrss` counter, only as system-wide wired memory (R9 §6.2). Sky never reports those metrics as RAM. Peak RAM is the rise in system memory used over a pre-start baseline.
- **The physical floor for Qwen3-32B is 20.0 GB at 4K context with fp16 KV, 27.5 GB at 32K, 53.3 GB at 128K** (`roofline.py`). "20 GB" therefore means a short context or a smaller KV format, stated as such.
- **What Sky can save at equal settings** is MLX's transient overhead: per-chunk activations, KV growth copies, cached free buffers, prompt-cache duplication and quantized-KV score tensors. The spine's target is ≤ 0.9× mlx-lm at 32K with fp16 KV, measured with the honest metric `[EXPERIMENTAL: S7]`.

## 2. What occupies memory

For one Sky session serving Qwen3-32B:

| Class | Contents | Lifetime | Where it lives | Size (Qwen3-32B) |
|---|---|---|---|---|
| Weights | all tensors of the `.skyweights` file (or GGUF) | process | `mmap(PROT_READ)` + no-copy `MTLBuffer` views, one per layer group | 18.43 GB (MLX 4-bit g64) |
| KV cache | K and V for every layer and position up to the maximum context | session | persistent arena (or slabs, §5) | 256 KiB per position at fp16 (R8 §1.5) |
| Recurrent state (hybrid models) | DeltaNet state and conv state | session | persistent arena | ≈ 151 MB fp32 for Qwen3.6-27B (R8 §1.7) |
| Prefix cache | shared KV blocks of cached prefixes | across requests | persistent arena, byte budget | configured |
| Speculation | drafter weights, draft KV, EAGLE-3 taps, per-position state checkpoints | session | mapped (drafter weights) + persistent arena | §2.1 |
| Generation state | `StepState`, token ring, sampler counts and parameters, stop bitmap | session | persistent arena (shared storage) | < 1 MB |
| Activations | hidden states, Q, attention output, MLP intermediate, logits, attention workspace | one program execution | transient arena, shared by all programs | §9.2 |
| Host I/O | prompt ids, emitted ids | request | host-shared arena | ≤ 4 B per context position |
| Runtime | Metal driver objects, pipelines, command allocators, tokenizer, Rust heap | process | not planned; measured | `roofline.py` assumes 0.2 GB |

### 2.1 Speculation's share

- **EAGLE-3 drafter for Qwen3-32B** (`RedHatAI/Qwen3-32B-speculator.eagle3`): 1.56 B parameters, including a 777.9 M-parameter copy of the embedding (R8 §2.3). In bf16 that is about 3.1 GB (arithmetic). Two reductions are possible because exact verification makes the drafter's numerics irrelevant to the output (D10):
  - share the target's embedding table if the copy is byte-identical (checked by hash at load) `[EXPERIMENTAL: compare the hashes]`;
  - quantize the drafter (labeled; affects acceptance only).
- **Draft model** Qwen3-0.6B 4-bit: a 335,450,584-byte file (R9 §1.2).
- **Hybrid models** checkpoint the recurrent state at each draft position for rollback (R8 §3.5): (γ + 1) × ≈ 151 MB, about 0.6 GB at γ = 3 (arithmetic).
- **Verify activations** at width M are in the transient arena (§9.2).

## 3. The static memory planner

### 3.1 Where it sits

The planner is step 8 of the compile pipeline ([Part VI](part06-architecture.md) §5): `sky_tir::memplan::plan_memory`. Its input is the scheduled program set produced by [Part IX](part09-scheduler.md):
- the prefill programs (one per chunk bucket);
- the decode program;
- the verify programs (one per width);
- the sampler program.

Each program is an ordered list of dispatches. Each dispatch names the logical buffers it reads and writes. Each logical buffer has a byte size (a `Poly` in the symbolic sizes), an alignment requirement and a class (§2).

```rust
pub struct MemoryPlan {
    pub arenas: Vec<Arena>,            // Persistent, Transient, HostShared; each page-aligned
    pub slots: Vec<Slot>,              // one per logical buffer
    pub lower_bound: u64,              // max over time of the bytes live at once (transient)
    pub notes: Vec<Explain>,           // TMEM-xxx rule records for `sky explain`
}
pub struct Slot {
    pub buffer: BufferId, pub arena: ArenaId, pub offset: u64, pub size: u64, pub align: u32,
    pub live: Interval,                // [first writer, last reader] in dispatch order
    pub in_place_of: Option<BufferId>, // TMEM-001
}
```

### 3.2 What it borrows from `sky_ir::lifetime`

Sky already has a last-use analysis for SIR (`crates/sky-ir/src/lifetime.rs`, `Lifetimes::analyze`, `dies`, `reuse`, `reuse_sites`): a backward liveness pass over structured regions that tells an executor where an operand dies so the interpreter can reuse its buffer in place, and that `sky check --explain` reports as `MEM-001` ("`v + dt * a` reuses `v`'s buffer: no allocation in this loop") `[VERIFIED: lifetime.rs module docs; sky-cli/tests/reuse.rs:247]`. Its rules carry over unchanged in spirit:
- a value is live from its definition to its last use on every path;
- an operand used twice in one operation is never reported as dying there (conservative, never wrong);
- an in-place candidate must die at the operation and have the result's shape and representation;
- the analysis is total: on malformed input it answers "nothing dies".

What changes for TIR: the unit is a dispatch, not an SIR op; the output is an offset in an arena, not a move; and the GPU adds a hazard rule (§3.5).

### 3.3 Lifetime intervals

- **Transient buffers** live from their first writer to their last reader in dispatch order. A buffer produced in one decode step and consumed in the next is not transient: the token, position and sampler state live in `StepState`, which is persistent ([Part XIII](part13-llm-runtime.md) §11).
- **All layers share one body.** The builder emits the layer once ([Part XIII](part13-llm-runtime.md) §5.1), so the planner plans the body's transient buffers once and every layer reuses the same offsets. A 64-layer plan costs no more to compute than a 1-layer plan.
- **Programs run one at a time on one queue** in year 1, so prefill, decode, verify and sampler programs share one transient arena, sized by the largest. A draft model that runs on a second queue concurrently ([Part XXI](part21-speculative.md)) is a second concurrency domain with its own transient arena.

### 3.4 Offset assignment

The assignment problem is dynamic storage allocation: place intervals in an address space so that buffers alive at the same time do not overlap, minimizing the arena size. It is NP-hard in general, but compiler planners do well with greedy heuristics. The precedents:
- XLA's buffer assignment, which "aims to produce an optimal buffer allocation for the program, minimizing the amount of intermediate memory consumed" (R7 §2);
- tinygrad's planner: lifetimes plus TLSF suballocation (256-byte blocks) into one arena per device and lane. Its decode JIT went from 6.92 MB in 457 buffers to 1.48 MB in 2, and eager prefill from 260.47 MB in 708 buffers to 5.11 MB in 2 `[VERIFIED: tinygrad schedule/memory.py:20-62; R2 §7.7]`.

Sky's algorithm, `assign_offsets`:
1. Sort transient buffers by size, largest first; ties by start time.
2. Place each at the lowest aligned offset that does not overlap, in address, any already-placed buffer whose interval overlaps it in time.
3. Report the arena size, the lower bound `max_t Σ size(live at t)`, and their ratio as `plan efficiency` (`TMEM-002`).

With one layer body of a few dozen buffers, the quadratic cost is negligible. An exact search (ILP) for one layer body is possible later if the efficiency report shows a gap worth closing `[EXPERIMENTAL]`.

### 3.5 In-place reuse and GPU hazards

An output may take an input's slot (`TMEM-001`) only if all of these hold:
1. **The input dies at this dispatch** (last use, as in `sky_ir::lifetime`).
2. **Same size and alignment.**
3. **The kernel skeleton declares the operand index-aligned with the output**: each output element depends only on the input element at the same index, read before it is written by the same thread. Elementwise kernels and residual-add epilogues qualify; a GEMV does not for its vector input (every output row reads all of it), but it does for the residual it adds into. Skeletons declare this per operand ([Part IX](part09-scheduler.md), [Part XI](part11-metal-backend.md)).
4. **No other dispatch in the same barrier epoch reads the input.** Encoders run dispatches concurrently unless a barrier separates them (R3 §2.1, R5 §1.2), so reusing a slot can add a write-after-read dependency.

Offset reuse between unrelated buffers (step 2 of §3.4) adds such dependencies too. The planner reports every added dependency to the scheduler, which may veto a reuse that would split a concurrent group with a new barrier. Bytes are cheap in the transient arena; a dependent dispatch boundary costs ≈ 1.2 µs of GPU time (R6 B3).

### 3.6 Alignment

- Arena bases are 16 KiB-aligned (the page size), so arenas can be no-copy buffers and residency-set members.
- Every slot is aligned to at least 256 bytes. Stricter requirements come from the consumer: 128 bytes for Metal tensor operands (`tensor_inline` with packed formats; R6 A3); for `device` buffers, the offset must be a multiple of the element size, and vector loads (`uint4`) need 16 bytes, which the Metal debug layer checks and the hardware does not (R9 §1.3).
- Padding is reported per arena (`TMEM-003`).

### 3.7 Symbolic sizes, including zero

Sizes are `Poly` values in the symbolic sizes of the configuration: maximum context `T_max`, chunk length `C`, verify width `M`, prompt length. The plan is instantiated for one configuration, and `sky explain` prints both the polynomial and the number.

**TIR's symbolic domain must include 0.** Sky's existing cost contracts are proven "for every size from 1 up", because a Sky array is never empty (spec, "Performance contracts"; R1 §0, §5.8). Tensor programs do have empty sizes:
- the KV cache before the first prefill;
- a prompt entirely covered by the prefix cache, which leaves zero new tokens to prefill;
- a speculation round with no drafted tokens.

So:
- TIR declares, per symbolic size, whether it may be 0.
- The planner and the contract checker reason over sizes from 0 up for those. The spec's sufficient test substitutes `x = 1 + y` and checks coefficients; for a size that may be 0, the TIR checker substitutes `x = y` instead.
- A zero-sized buffer gets no storage. Sky never creates a zero-length `MTLBuffer`, and a kernel bound to a zero-sized slot is not dispatched.

### 3.8 Verification

- `verify_plan(&MemoryPlan, &Schedule)` checks every pair of slots: if their intervals overlap, their address ranges are disjoint, unless one is an approved `in_place_of` the other. It also checks alignment and arena bounds. It runs after every plan in debug builds and in CI.
- A property test generates random schedules and compares `verify_plan` with a brute-force checker.
- A GPU validation mode fills each slot with a poison pattern (NaN) when its interval ends, so a use after reuse shows up as a non-finite output at the next layer-boundary check (D6 makes finiteness checks a validation mode).

## 4. Zero allocation in the hot path

Rules:
- Everything is allocated when a session starts: arenas, KV, the token ring, sampler state, `StepState`, the host I/O buffers and the residency sets.
- **Per token:** no `MTLBuffer` creation, no residency-set change, no command-allocator growth, and no Rust heap allocation in the decode loop or the streamer thread. The detokenizer and the output string reuse preallocated capacity.
- **Per request:** no Metal allocation. Prompt ids go into a host buffer sized for the maximum context.

Why it matters beyond speed:
- MLX allocates one `MTLBuffer` per array and has hit Metal's resource limit (`Resource limit (499000) exceeded`); a maintainer called exceeding it a sign of "some fatal mistakes", and an mlx-lm leak was acknowledged as "a MLX bug" (mlx#3849, mlx-lm#1662; R3-issues G6). Sky's buffer count is fixed at load: a few arenas, the weight regions and any KV slabs.
- A fixed set of allocations is what makes the memory plan a contract rather than an estimate (§11).

Tests:
- `MTLDevice.currentAllocatedSize` is constant over 10,000 decode tokens and over 100 requests.
- A counting global allocator shows zero allocations per decode token in the loop and the streamer. Sky already has the pattern: `crates/sky-bench/tests/allocations.rs` asserts that a SIR loop allocates nothing (R1 §4.2).

## 5. KV allocation: preallocation vs commit-on-demand

| Policy | How | For | Against |
|---|---|---|---|
| **Preallocate** (default, D9) | all KV blocks for `T_max` created at session start in the persistent arena | zero allocation ever; the plan is exact; a contract on `memory` is provable | reserves `T_max` × 256 KiB even for short chats: 8.59 GB at 32K for Qwen3-32B |
| **Grow in slabs** (`--kv grow`) | KV blocks drawn from a pool backed by slabs (e.g. 256 MiB `MTLBuffer`s) created when the pool runs low, **between submissions**; each slab joins a residency set; attention reads blocks through a block table ([Part XIV](part14-kv-cache.md), [Part XV](part15-attention.md)) | memory follows the conversation; the block table also serves prefix sharing (Part XVIII §6) | one allocation per slab: for Qwen3-32B at fp16, one per 1,024 positions with 256 MiB slabs (arithmetic); the host must keep `free blocks ≥ steps in flight`; an indirection in attention |
| **Sparse mapping** (spike only) | reserve a sparse buffer and map pages on demand (vAttention-style; R7 open question 4); Metal 4 queues have sparse mapping operations (R6 A3) | no indirection | residency sets do not support sparse heaps (Apple's `MTLResidencySet` documentation; R6-work/docs); behaviour on Apple GPUs unknown `[EXPERIMENTAL]` |

`--kv grow` keeps the hot-path rule: a slab is created by the host before the steps that need it are committed, never inside the per-token loop. The memory contract then states the maximum (all slabs), and `sky explain` shows both the reserved and the committed size.

mlx-lm, for comparison, grows each layer's K and V by concatenation in steps of 256 positions: it allocates `ceil(new/256)·256` zeroed slots and concatenates, a full copy of the layer's K and V every 256 tokens (`models/cache.py:339-368`) `[VERIFIED: R4 §4.1]`. The time cost is small; the memory cost is in §8.

## 6. Weights, and the honest metric

### 6.1 How weights are held

| Engine | Mechanism | Counted by per-process footprint? |
|---|---|---|
| MLX / mlx-lm | `pread` into freshly allocated shared `MTLBuffer`s (R3 §4); the buffers are anonymous memory | yes; `mx.get_peak_memory` also includes them (R9 §6.1) |
| llama.cpp | `mmap` + `newBufferWithBytesNoCopy` (R5 §1.5) | **no** |
| Sky | `mmap` of `.skyweights` or GGUF + `newBufferWithBytesNoCopy`, per layer-group region ([Part XIII](part13-llm-runtime.md) §6) | **no** |

### 6.2 What each metric sees

R9's probe (M2 Pro, 1 GiB file) `[VERIFIED: R9 §6.2; R9-work/core/memprobe]`:

| After | phys_footprint | RSS | `task_vm_info.external` | `MTLDevice.currentAllocatedSize` | system wired |
|---|---|---|---|---|---|
| CPU touches every page of an `mmap` | unchanged | +1 GiB | +1 GiB | 0 | ~0 |
| GPU reads a no-copy mapping the CPU never touched | **unchanged** | **unchanged** | **unchanged** | 1 GiB | **+1085 MiB** |
| 1 GiB private buffer filled by the GPU | +1 GiB | unchanged | — | +1 GiB | — |

Consequences:
- Weights that only the GPU reads through a mapping are invisible to Activity Monitor's memory column, RSS, `/usr/bin/time -l` and `vmmap`. llama.cpp's footprint would therefore look about 19 GB smaller than MLX's for the same model, although both need the RAM `[LIKELY: R9 §6.2; to confirm with real binaries]`.
- Sky uses the same mechanism and must not claim that difference.
- R3 §9 recommended reporting `phys_footprint`; R9's probe shows it misses mapped weights, so the spine adopts R9's metric (D13), and so does this part.

### 6.3 The metric Sky reports

`peak_ram_bytes` (R9 §6.2, D13):
- `U(t) = page_size × (wire_count + internal_page_count − purgeable_count + compressor_page_count)`, read with `host_statistics64` at 10 Hz by the external harness. This is Activity Monitor's system "Memory Used".
- `peak_ram_bytes = max_t U(t) − median(U over the 5 s before the engine starts)`.
- It counts copied weights, mapped weights the GPU has wired, KV, activations, allocator caches and driver memory. It does not count clean page cache, which is fair: MLX's page-cache copy of the file is reclaimable after loading.
- Guard rails:
  - if `peak_ram_bytes < 0.95 ×` the weight bytes, the engine is running from unwired file pages; re-measure including `task_vm_info.external` and flag the result;
  - swap-outs, compressions and decompressions must not change during the run;
  - a null engine through the same harness gives the noise floor (expected ±100–300 MB on an idle Mac, `[LIKELY: R9 §6.2]`);
  - load peak and steady state are reported separately; copying loaders can peak at about 2× during load (R9 §6.2).
- Diagnostics, never headlines: `phys_footprint_peak`, `ru_maxrss`, peak `task_vm_info.external`, the engine's own report, and `mincore` residency of the model file.

`mx.get_peak_memory` is not a RAM number: it tracks live MLX buffers only, excluding the buffer cache. One contributor's churn test showed 1.00 GB from it against 60.19 GB of OS footprint (mlx#3896; R3-issues G1).

## 7. The wired limit and residency sets

### 7.1 Facts

| Fact | Value | Label |
|---|---|---|
| `recommendedMaxWorkingSetSize` on the 16 GB M2 Pro | 11.84 GiB, 74% of RAM (MLX reports the same quantity as 12.71 GB decimal) | `[VERIFIED: R6 B1, R4 §0]` |
| On M5 Max machines | unmeasured; ≈ 74% of RAM if the M2 Pro ratio holds | `[LIKELY: R6]` |
| `iogpu.wired_limit_mb` | default 0; raised with `sudo sysctl iogpu.wired_limit_mb=N`, with N above the model size in MB and below the machine's memory (mlx-lm `README.md:286`) | `[VERIFIED: R6 A4]` |
| MLX's limits | memory limit `min(1.5 × recommended working set, 0.95 × RAM)`; above `min(0.95 × recommended, limit)` it evicts cached buffers before allocating (`allocator.cpp:63-65, 132-138`) | `[VERIFIED: R3 §2.3]` |
| MLX wiring | `set_wired_limit` (default 0, nothing wired) is implemented with residency sets and capped at the recommended working set; mlx-lm raises it for each generation call and restores it after (`generate.py:228-263`); it warns when the model exceeds 0.9 × the recommended size | `[VERIFIED: R3 §2.3, R4 §6]` |
| Wiring matters | Llama-3.1-70B fp16 on an M2 Ultra: 0.23 tok/s not wired vs 4.7 wired (mlx-examples#1069) | `[VERIFIED: R3-issues G4]` |
| Residency sets | Metal makes the union of all sets resident; ≤ 32 sets per queue; Metal 4 has no `useResource`, so every resource must be in a set; sets do not support sparse heaps | `[VERIFIED: Apple docs; R6 A4]` |
| macOS 27 | decides residency per set; one monolithic set caused re-wiring stalls (GLM-5.1 on an M3 Ultra: 0.46 → 13.19 tok/s after splitting); MLX now caps each set at 5% of the working set (mlx#4211) | `[VERIFIED: R3-issues G4]` |
| `maxBufferLength` | 8.88 GiB on the 16 GB M2 Pro, so a 32B model needs several buffers | `[VERIFIED: R6 A4]` |
| Kernel panics | "completeMemory() prepare count underflow" across M1–M5 while wiring (mlx#3186, FB22091885); one user avoided it only by not wiring | `[VERIFIED statements: R3-issues G5]` |

### 7.2 Sky's policy

1. **Budget check at load.** The planner's total (weights + arenas) is compared with `recommendedMaxWorkingSetSize`. Over it, `sky run` refuses, prints the plan, and suggests the two fixes that do not change the model: a smaller `--context`, or raising `iogpu.wired_limit_mb` to a computed N (`TMEM-005`). It never changes a sysctl itself. `--allow-over-working-set` proceeds, with a warning that macOS may page or compress GPU memory.
2. **Residency sets by region.** Weight regions (embedding, layer groups, head), the persistent arena, the transient arena and the host arena each join a set. Layer groups are sized so that every set stays at or below 5% of the working set and the total stays at or below 32 sets per queue. For Qwen3-32B: 64 layers in groups of 4 make 16 sets of ≈ 1.1 GB, under 5% of a working set of 35.5 GB (a 48 GB machine at 74%) or more `[arithmetic; LIKELY ratio]`.
3. **Heartbeat and keep-alive** as in [Part XVIII](part18-ttft.md) §5.2; `endResidency()` after the keep-alive returns memory to the system.
4. **No-copy views at layer boundaries** stay far below `maxBufferLength` and give per-group residency.
5. **A kill switch** (`SKY_NO_RESIDENCY=1`) for the panic class above.

## 8. MLX's caching allocator vs a static plan

| Mechanism in MLX / mlx-lm | Memory effect | Source | Static plan |
|---|---|---|---|
| Size-keyed buffer cache; reuse only if `size ≤ cached < min(2·size, size + 2 pages)`; cache limit defaults to the memory limit | freed buffers stay resident until reused or evicted; growing allocations miss the window (the maintainers declined to change the policy: "won't fix", mlx#3886) | R3 §2.3, R3-issues G2 | no cache: every byte has a planned slot |
| `mx.clear_cache()` after every prefill chunk and every 256 decode tokens | bounds the cache's growth to one chunk or 256 tokens of churn | `generate.py:454, 471`; R4 §2.1 | not needed |
| KV growth by concatenation every 256 positions | during each growth the old and new K/V of a layer coexist; the old buffers enter the cache until the next clear. Each 2048-token prefill chunk triggers a growth copy | `cache.py:339-368`; R4 §4.1 | preallocated; no copy |
| Prompt-cache hit: `copy.deepcopy` shares buffers, so the first write copies the whole buffer | 2× the conversation's KV while the cached entry lives; a ≈ 6.3 GB cached sequence killed a 16 GB server on reuse (mlx-lm#1395) | R4 §4.1 | reference-counted blocks; only the last partial block is copied (Part XVIII §6) |
| Quantized KV attention is unfused | prefill materializes (heads × chunk × context) fp32 scores: quantized KV used **more** peak memory than fp16 (Llama-3.2-3B on an M4 Max: +1.41 GB at 8K, +2.38 GB at 32K; 1.208 GB predicted vs 1.216 GB measured for the scores) | mlx-lm#1587; R4 §4.3, §9 | fused quantized-KV attention reads K/V in place ([Part XV](part15-attention.md)); the score tile lives in threadgroup memory |
| One `MTLBuffer` per array | Metal resource-limit failures (mlx#3849) | R3-issues G6 | a fixed handful of buffers |

Measured on the M2 Pro (R4 §7.A), mlx-lm with Qwen3-8B-4bit (a 4,607,835,174-byte file):
- `mx.get_peak_memory`: 4.92 GB in a decode run (128-token prompt), 5.61 GB in a 2048-token prefill run;
- OS lifetime maximum `phys_footprint` (MLX's weights are anonymous, so footprint includes them): 5.25 GB and **6.29 GB**.

At 2,048 tokens the 8B's fp16 KV is 2,048 × 147,456 B ≈ 0.30 GB, so the prefill run's footprint is about 1.4 GB above weights plus KV `[arithmetic on VERIFIED inputs; the footprint also includes Python, Metal and the tokenizer]`. R4 §10.3 concludes that the peak-RAM win available to a static plan is "small" relative to the weights. The spine's ≤ 0.9× target at 32K needs mlx-lm to sit at least ≈ 3 GB above the 27.5 GB physics floor there. The mechanisms above make that plausible at 32K (one 2048-token chunk of a 64-layer, 25,600-wide MLP; KV growth copies of 64 MiB per layer per tensor; cached free buffers), but it is unmeasured `[EXPERIMENTAL: S7 at depth 32K]`.

The cost of a static plan is rigidity: the KV for `T_max` is reserved even when unused (§5), and a new maximum context means a new plan. llama.cpp shows the other extreme: by default its context is the model's trained length (40,960 for Qwen3), auto-fitted to free memory, so its RAM depends on `-c` and every comparison must pin it (R5 §11 item 10).

## 9. Peak-RAM breakdown for Qwen3-32B

### 9.1 Totals (`roofline.py`)

Weights MLX 4-bit group 64 (18.43 GB); scratch 0.3 GB and runtime 0.2 GB assumed by `roofline.py`:

| Context | KV format | Weights | KV | Scratch + runtime | **Total** |
|---:|---|---:|---:|---:|---:|
| 4,096 | fp16 | 18.43 GB | 1.07 GB | 0.50 GB | **20.00 GB** |
| 4,096 | int8 | 18.43 | 0.54 | 0.50 | **19.47** |
| 32,768 | fp16 | 18.43 | 8.59 | 0.50 | **27.52** |
| 32,768 | int8 | 18.43 | 4.29 | 0.50 | **23.22** |
| 131,072 | fp16 | 18.43 | 34.36 | 0.50 | **53.29** |
| 131,072 | int8 | 18.43 | 17.18 | 0.50 | **36.11** |

Notes:
- `roofline.py`'s int8 KV is 1 byte per element with no scales. MLX-style int8 with group-64 scales and biases is 8.5 bits per element: 139,264 B per position, so 4.56 GB at 32K and 18.25 GB at 128K (R8 §1.5). int8 KV is a labeled mode with a quality budget (D5; Apple ships 8-bit KV, R8 §5.2).
- **128K is a different model configuration.** Qwen3-32B's native context is 32,768 (`max_position_embeddings` 40,960); 131,072 needs YaRN ×4, and the model card advises against static YaRN below 32K because it hurts short texts (R8 §1.1; R9-arch §2). A 128K plan is therefore a separate plan whose outputs differ at every length, never a flag on the same plan ([Part XIII](part13-llm-runtime.md) §4.2).

### 9.2 What the planner puts in "scratch"

Arithmetic for the transient arena, assuming the fusions of [Part IX](part09-scheduler.md) (QKV with norm and RoPE writing K/V straight into the cache; a SwiGLU epilogue on gate/up; residual epilogues), a 4-byte residual stream and 2-byte activations (the actual dtypes are [Part VII](part07-tensor-ir.md)'s choice):

| Live at the peak of one layer, per token | Bytes |
|---|---:|
| residual stream (5,120 × 4) | 20,480 |
| normalized input (5,120 × 2) | 10,240 |
| max of: Q + attention output (2 × 8,192 × 2) = 32,768, or SwiGLU output (25,600 × 2) = 51,200 | 51,200 |
| **total per token** | **81,920 (80 KiB)** |

| Program | Transient arena (arithmetic) |
|---|---:|
| prefill chunk C = 512 | 41.9 MB |
| prefill chunk C = 2,048 | 167.8 MB |
| prefill chunk C = 8,192 | 671.1 MB |
| decode, M = 1 (incl. 0.6 MB of fp32 logits) | 0.7 MB |
| verify, M = 32 (incl. 32 rows of logits) | 22.1 MB |
| flash-decode split workspace, per query row, 32K context, assumed split size 1,024 (64 heads × 32 splits × (128 + 2) × 4 B) | 1.1 MB |

Prefill attention may add a workspace that depends on the algorithm: none for a flash kernel that handles each query block over all keys, and possibly hundreds of MB for split-KV prefill at long context ([Part XV](part15-attention.md)). The planner prints the actual figure. With C = 2,048 and no split workspace, the "0.3 GB scratch" assumption is conservative by about 0.13 GB `[arithmetic]`.

### 9.3 Fit on M5 Max machines

The 40-core M5 Max ships with 48, 64 or 128 GB (R6 §0). Assuming the working set is 74% of RAM as on the M2 Pro (35.5, 47.4 and 94.7 GB) `[LIKELY: unmeasured on M5 Max]`:

| Configuration | Total | 48 GB | 64 GB | 128 GB |
|---|---:|---|---|---|
| 4K, fp16 KV | 20.00 GB | fits | fits | fits |
| 32K, fp16 KV | 27.52 GB | fits | fits | fits |
| 32K, int8 KV | 23.22 GB | fits | fits | fits |
| 128K, int8 KV (YaRN) | 36.11 GB | needs `iogpu.wired_limit_mb` | fits | fits |
| 128K, fp16 KV (YaRN) | 53.29 GB | no | needs the sysctl, leaving < 11 GB to macOS | fits |

## 10. The `sky explain` memory report

`sky explain <model> --memory` prints every planned byte, grouped by arena, with each buffer's size formula, number, offset, lifetime and the rule that placed it. Illustrative output for Qwen3-32B, 32K context, fp16 KV. The weight and KV lines are exact arithmetic from the file size and config; the transient line uses §9.2's arithmetic, which the real plan replaces:

```text
$ sky explain hf://mlx-community/Qwen3-32B-4bit --memory --context 32768 --kv f16
memory plan  Qwen3-32B · mlx-affine q4 g64 · max context 32,768 · batch 1 · KV f16 · M5 Max (40c)

weights      mapped, no copy: ~/Library/Caches/sky/weights/9f3c….skyweights      18,429,667,328 B
             18 regions (embedding; 16 groups of 4 layers; norm + lm_head), 16 KiB-aligned
             read per decode token: 17,990,737,920 B (lm_head 2.4%)

persistent   KV  64 layers × {K,V} × 8 heads × 32,768 × 128 × 2 B                   8,589,934,592 B
             sampler counts u32[151,936] · token ring u32[4,096] · StepState · params      628,288 B
             [TMEM-004] KV preallocated for T_max = 32,768 (use --kv grow to commit on demand)

transient    peak = prefill chunk 2,048: 2,048 × 80 KiB + last-row logits                168,379,904 B
             shared by: prefill(512, 2048), decode, verify(2..16), sampler
             plan efficiency 1.00 (arena / lower bound)                              [TMEM-002]
             in place: residual h ← o_proj epilogue, h ← down_proj epilogue (×64)    [TMEM-001]

host         prompt ids u32[32,768]                                                         131,072 B

total planned                                                                       27,188,741,184 B
not planned  Metal driver and pipelines, tokenizer, heap: measured by sky bench llm (S7)
working set  recommendedMaxWorkingSetSize 94.7 GB → fits                            [TMEM-005]
contract     memory <= 28 GB  (proven: 27.19 GB)
```

The report also prints, when a measurement exists, the system-wide `peak_ram_bytes` from the last `sky bench llm` run and the difference "measured − planned", which is the unplanned runtime.

## 11. Memory contracts

### 11.1 Today's contracts

Sky's spec already has performance contracts (`cost flops <= 2 * N^3`) with the measures `flops`, `bytes_read`, `bytes_written`, `bytes`, `allocations` and `allocated_bytes`. `sky check` proves or refutes them statically, for every size from 1 up; a run-time time budget was rejected on purpose ("a contract that is sometimes false is not a contract") `[VERIFIED: spec/v0.1.md, "Performance contracts"]`. A static memory plan fits that rule exactly: it is a fact about the program, not a measurement.

### 11.2 Two new measures for tensor programs

| Measure | Meaning | Proven from |
|---|---|---|
| `memory` | the peak bytes of the static plan: mapped weights + persistent + transient + host arenas | the memory plan |
| `bytes_per_token` | bytes read from memory per decode step: weights + KV at context `T` (+ planned activation traffic) | the plan and the binding ([Part XIII](part13-llm-runtime.md) §6.5) |

Spellings (D12; the spine writes `cost bytes_per_token <= 18.2 GB`):

```text
cost memory <= 20 GB                         # with --context 4096: proven (19.67 GB planned; roofline.py's 0.5 GB overhead gives 20.00)
cost memory <= 20 GB                         # with --context 32768: refuted
cost bytes_per_token <= 18.0 GB + 256 KiB * T
```

Rules:
- **Where they are written.** In a Sky-source model ([Part VII](part07-tensor-ir.md)), on the `def` of the model, like any cost line. For a Hugging Face model built by the Rust builder: `sky compile --contract "memory <= 20 GB"`; the `.skybin` records the contract and its outcome.
- **Outcomes** are the spec's three:
  - proven (a note under `--explain`);
  - refuted (an error with a witness: "memory is 27.19 GB at context 32,768; with this plan's 2,048-token prefill chunk the bound holds for context ≤ 5,345");
  - unproven (a warning naming why).
- **Domain.** Sizes that may be 0 (§3.7) are checked from 0 up.
- **What is not covered.** The unplanned runtime (driver objects, pipelines, the heap) is not in the plan, so `memory` is a statement about planned bytes. `sky bench llm` measures the system-wide peak and reports the gap; a release check can fail if the gap exceeds a configured margin.

### 11.3 Byte units in bounds

Sky's units have no byte unit, and in today's bound grammar a number written against a name is read as a unit literal (`2 N` is two newtons), so `20 GB` does not parse `[VERIFIED: no byte unit under crates/sky-units; spec, "Performance contracts"]`. Proposal, edition-gated and limited to cost bounds:
- decimal byte units `B`, `kB`, `MB`, `GB`, `TB` (powers of 1000);
- binary units `KiB`, `MiB`, `GiB`, `TiB` (powers of 1024);
- no bit units, so that `Gb` cannot be mistaken for `GB`.

Until that lands, the same contracts are written with plain numbers: `cost memory <= 20e9`.

## 12. What "24 → 20 GB" can and cannot mean

**What it cannot mean:** the same model, the same context and the same KV format in 17% less memory.
- The weights are 18.43 GB in every engine (D5).
- The fp16 KV is 256 KiB per position in every engine.
- At 4K context, those two plus `roofline.py`'s 0.5 GB of scratch and runtime already make 20.0 GB.

**What it can mean, each stated with its context:**

| Statement | Condition | Source |
|---|---|---|
| "20 GB" total | with 0.5 GB of other overhead (`roofline.py`): context ≤ ≈ 4.1K at fp16 KV, ≈ 7.7K at MLX-style 8.5-bit int8, ≈ 14.5K at 4-bit; with R8's ≈ 0.4 GB: ≈ 4.4K / 8.4K / 15.8K (R8 §10); ≈ 5.3K at fp16 if the only other bytes are this plan's 0.17 GB transient arena, before the unplanned runtime | arithmetic: (20 GB − 18.43 GB − overhead) ÷ KV bytes per position |
| "≈ 2× more context in 20 GB" | Qwen's own MLX 4-bit uses group 128 (17.41 GB of weights), which roughly doubles those context limits | R8 §1.3, §10 |
| "≤ 0.9× mlx-lm" | same file, same context (32K), same KV format (fp16), the honest metric | spine §3 target |
| "saves 1–3 GB" | equal settings; the saving is mlx-lm's transient overhead (§8) | R8 §10 `[LIKELY]` |
| "fits 32K in 24 GB" | int8 KV (23.2 GB), a labeled mode with a quality budget | `roofline.py` |

The 20 GB owner target is therefore a statement about KV format and context length, and every published RAM number carries both, plus the metric's name.

## 13. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XIX-T1 | Lifetime intervals over schedules | `sky-tir/src/memplan/intervals.rs` | intervals per buffer; layer-body reuse; concurrency domains | Unit tests on hand-written schedules; a 64-layer plan takes no longer than ten times a 1-layer plan |
| XIX-T2 | Offset assignment and in-place rules | `sky-tir/src/memplan/assign.rs` | `assign_offsets`, `TMEM-001/002/003` | Efficiency ≥ 0.95 on the Qwen3 layer body; in-place applied only to index-aligned operands; scheduler veto honored |
| XIX-T3 | Plan verifier and poison mode | `sky-tir/src/memplan/verify.rs`, `sky-metal` validation | `verify_plan`; NaN poisoning of dead slots | Property test vs brute force over 10⁵ random schedules; a deliberately bad plan is caught on the GPU |
| XIX-T4 | Zero-size handling | `sky-tir` symbolic sizes, `memplan` | sizes declared may-be-zero; no zero-length buffers or dispatches | Empty KV, fully cached prompt and zero-draft round run without allocation or dispatch |
| XIX-T5 | Session allocation and hot-path tests | `sky-llm/src/session.rs`, `tests/no_alloc.rs` | everything allocated at session start | `currentAllocatedSize` constant over 10,000 tokens; counting allocator: 0 allocations per token |
| XIX-T6 | KV slab growth | `sky-llm/src/kv/slabs.rs` (with Part XIV) | `--kv grow`; slab creation between submissions | No allocation inside a submission; committed size follows context; memory contract states the maximum |
| XIX-T7 | Residency policy | `sky-metal/src/residency.rs` | sets ≤ 5% each, ≤ 32 per queue; budget check `TMEM-005`; kill switch | A 32B plan on a 48 GB machine produces 18 weight sets + 3 arena sets (≤ 32, each ≤ 5%); over-budget plans refuse with the computed sysctl value |
| XIX-T8 | `sky explain --memory` | `sky-cli/src/llm/explain.rs` | the report of §10 | Every planned byte appears once; totals equal the plan; the measured-vs-planned line appears after a `sky bench llm` run |
| XIX-T9 | Memory contracts | `sky-tir/src/contracts.rs`, `sky-cli` | `memory`, `bytes_per_token`; proven/refuted/unproven | The §11.2 examples give the stated outcomes, with a witness for the refutation |
| XIX-T10 | Byte units in cost bounds (spec PR) | `spec/v0.1.md` (next edition), `sky-types` cost-bound parser | `B`…`TiB` in bounds only | Golden UI tests: `20 GB` parses in a bound and is an error elsewhere; `Gb` is an error with help |
| XIX-T11 | Honest memory measurement | `sky-evaluation` harness, `sky bench llm` | R9's `peak_ram_bytes` with guard rails | Null-engine noise floor recorded; a mapped-weights engine passes the 0.95 × weights guard |
| XIX-T12 | Drafter memory | `sky-llm/src/spec/load.rs` | embedding sharing by hash; optional drafter quantization | EAGLE-3 drafter for Qwen3-32B binds without a second embedding copy when the hashes match |

## 14. Measurements that decide

- **S7 at depth 32K** (and 4K, 16K): `peak_ram_bytes` for Sky and mlx-lm on the same file with fp16 KV, load peak and steady state separately. This decides the ≤ 0.9× target.
- **Planned vs measured**: the unplanned runtime on the M5 Max. It replaces `roofline.py`'s 0.2 GB assumption.
- **Transient-arena size** per program from the real plan, against §9.2's arithmetic and `roofline.py`'s 0.3 GB.
- **`recommendedMaxWorkingSetSize` and `maxBufferLength`** on the 48, 64 and 128 GB M5 Max. They decide §9.3's table.
- **Whether GPU reads of no-copy mappings still wire pages on the M5 Max** (R9 open question 1). If not, the honest metric's guard rail fires for every mapping engine.
- **Slab growth cost**: the time to create and make resident a 256 MiB slab between submissions.

## 15. Risks and negative results

- **The memory win at equal settings is small.** Weights and KV are identical across engines; R4 found mlx-lm's transient overhead to be about 0.3–0.7 GB at 2K on 8B. The ≤ 0.9× target at 32K may not hold if mlx-lm's 32K overhead is under ≈ 3 GB. That is a measured outcome, not a design failure.
- **Accounting tricks are easy and tempting.** A mapped-weights engine looks 18 GB smaller in Activity Monitor. Any published number not from the system-wide metric would be misleading, and a hostile reviewer will check (R9 §10).
- **Preallocation reserves memory** for contexts that are never used. `--kv grow` is the answer, at the cost of a block table in attention.
- **Residency behaviour changes with the OS** (macOS 27 decides per set) and wiring large memory has caused kernel panics (mlx#3186).
- **Over-subscription stalls silently.** Without the budget check, a plan above the working set runs with paging and the user sees only slowness; hence the refusal by default.
- **Zero-sized symbolic sizes** break proofs written for sizes ≥ 1. The TIR domain allows 0 explicitly; contracts imported from scientific Sky keep their ≥ 1 domain.
- **128K context is a different model** (YaRN); quoting its memory next to the 32K configuration's speed would mix two models.
