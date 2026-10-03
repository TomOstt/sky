# R3 — MLX core traced through the code, plus microbenchmarks

Agent R3. Scope: MLX graph/eval, Metal backend, kernels (deep on quantized GEMV), loading, memory accounting,
known issues, and MLX-side microbenchmarks on the local M2 Pro.

- Source: `$S/src/mlx` at HEAD `0840c42a7aab95fcd4a39e8bc56aa6887470c043` (2026-10-01, "Fix reduction over large
  arrays (#4549)"), `mlx/version.h` = 0.32.4 (dev). Shallow clone. mlx-lm clone at `53b9af3` (2026-10-01).
- Benchmarks ran on the installed wheel: **mlx 0.32.3**, mlx-lm 0.32.0, Python 3.12.12 (`$S/venv`).
- Machine: Apple M2 Pro, 16-core GPU, `applegpu_g14s`, 16 GB, macOS 26.6.1. 4K@100 Hz external display attached
  (display scan-out takes a few GB/s of DRAM bandwidth; it is in every number below).
- `$S` = `/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.
  Paths below are relative to `$S/src/mlx/` unless stated; `M/` = `mlx/backend/metal/`, `K/` = `M/kernels/`.
- Companion files written by R3 sub-forks (same evidence standard):
  - `$S/research/R3-kernels.md` — kernel catalog, Steel GEMM, rms_norm/rope/SDPA, Metal 4 / NAX (code reading).
  - `$S/research/R3-issues.md` — GitHub issues/PRs/maintainer statements (Q6).
- Labels: VERIFIED (code path:line, official doc, maintainer link, or our command + raw output), LIKELY (reason
  given), EXPERIMENTAL, MOONSHOT.

---

## 1. Graph and evaluation

### 1.1 Representation (VERIFIED)

- `array` is a handle to a `shared_ptr<ArrayDesc>` (mlx/array.h:546). `ArrayDesc` (mlx/array.h:491-541) holds:
  shape, strides, size, dtype, `shared_ptr<Primitive>`, `Status`, an `Event` (array.h:501), `shared_ptr<Data>`
  (array.h:509; buffer + deleter, array.h:241-256), offset, `data_size` (in elements, not bytes — array.h:355-358),
  `Flags{contiguous,row_contiguous,col_contiguous}` (array.h:258-279), `inputs` (array.h:520), `siblings` and
  `position` for multi-output primitives.
- `Status` is `unscheduled` → `evaluated` (eval_* has run; GPU work maybe not finished) → `available`
  (array.h:397-415).
- `Primitive` holds its `Stream` (primitives.h:49-124) and implements `eval_cpu` / `eval_gpu` (primitives.h:70-75,
  134-135). Building an op allocates an `ArrayDesc` + a `Primitive` on the heap; nothing runs until eval.

### 1.2 eval / async_eval (VERIFIED, mlx/transforms.cpp)

- `eval_impl(outputs, async)` (transforms.cpp:80-365):
  1. Wraps the outputs in a synthetic `Synchronizer` array (transforms.cpp:101-102) and attaches an `Event` to it.
  2. Iterative DFS over unscheduled inputs to count fan-out ("degree") per array (transforms.cpp:115-184). Records
     cross-stream edges that need a `Fence` (transforms.cpp:149-158). Already-evaluated non-tracer arrays are
     `detach()`ed (inputs dropped) on the way (transforms.cpp:178-182).
  3. Builds the execution tape in **BFS order with a width limit** `MLX_BFS_MAX_WIDTH` (default 20,
     mlx/utils.h:188-191), falling back to DFS when the frontier is wider (transforms.cpp:186-232). Purpose: bound
     the number of live intermediates (memory), not maximize GPU concurrency.
  4. Walks the tape (transforms.cpp:236-318). Per array: insert fence/event waits for cross-stream or
     cross-eval inputs (transforms.cpp:255-270), then `gpu::eval(arr)` or `cpu::eval(arr)` (transforms.cpp:272-276).
  5. **Back-pressure:** if more than `MAX_ACTIVE_TASKS = 10` command buffers are in flight (transforms.cpp:25) or
     active memory exceeds the memory limit, it commits all open GPU streams and blocks in
     `scheduler::wait_for_one()` (transforms.cpp:278-292). This is a CPU↔GPU sync point inside eval.
  6. Marks outputs `evaluated` and `detach()`es them (drops `inputs`, which frees intermediates as early as
     possible) (transforms.cpp:308-317).
  7. Signals the per-stream event and calls `gpu::finalize(s)` (= commit) on each open stream
     (transforms.cpp:349-362).
- `eval()` = `eval_impl(...,false).wait()` (transforms.cpp:381-396): the calling thread blocks on the
  synchronizer's `MTLSharedEvent` (`waitUntilSignaledValue`, M/event.cpp:27-29).
- `async_eval()` = the same, without the wait (transforms.cpp:367-379). Events are attached to every array of the
  tape (transforms.cpp:240-252) so a later `item()`/`eval()` waits on them (mlx/array.cpp:132-152).
- **All GPU encoding happens on the calling (Python) thread** inside `eval_impl` — `gpu::eval` is called inline
  (transforms.cpp:272-273). The scheduler's `StreamThread`s (mlx/scheduler.cpp:42-91) run only CPU-stream tasks
  (e.g. `Load`, CPU ops) and CPU-side event waits/signals. So per-token CPU cost (graph build + encode) is on the
  Python thread; mlx-lm hides it by building token t+1 and `async_eval`-ing it before reading token t
  (mlx-lm/mlx_lm/generate.py:456-472).

### 1.3 CPU-side work per GPU primitive (VERIFIED, M/eval.cpp:29-69)

Per primitive, on the Python thread:
- an `NS::AutoreleasePool` (M/eval.cpp:30), command-encoder lookup in an `unordered_map` by stream index
  (M/device.cpp:918-931);
- `primitive.eval_gpu()`: output `malloc` from the buffer cache, kernel name built with string concatenation
  (e.g. `affine_qmv_fast_float16_t_gs_64_b_4_batch_0`, M/quantized.cpp:497-507), pipeline lookup in a hash map
  under a shared mutex (M/device.cpp:843-908), `setComputePipelineState`, `setBuffer`/`setBytes` per argument,
  hazard tracking in `unordered_set`s (`set_input_array`/`register_output_array`, M/device.cpp:345-380),
  `maybeInsertBarrier` (M/device.cpp:393-406), `dispatchThreadgroups`;
- builds an `unordered_set<shared_ptr<Data>>` of input buffers and attaches it to an Obj-C completion handler on
  the command buffer **for every primitive** (M/eval.cpp:49-68), to keep inputs alive until the GPU finishes;
- `detach()` of the array (frees intermediates → allocator `free` → buffer cache).
- Measured cost: see §7 E (µs per dependent op) and §7 F (per-token build/encode time).

### 1.4 mx.compile (VERIFIED, mlx/compile.cpp, M/compiled.cpp)

- Trace → `compile_dfs` → `compile_simplify` (CSE / no-op removal) → `compile_fuse` → `compile_replace`
  (compile.cpp:1120-1190). Cache keyed on function id, input shapes/dtypes (unless shapeless), constants
  (compile.cpp:300-414).
- **Only elementwise ops fuse.** `is_fusable = is_unary || is_binary || is_ternary(Select) || is_broadcast`
  (compile.cpp:28-81). Reductions (`is_reduction` exists, compile.cpp:75-77, but is not fusable), matmuls,
  quantized matmul, norms, rope, SDPA, gathers and slices are never fused. Each fused region is capped at depth
  `max_compile_depth = 11` and `max_compile_arrays = 24` inputs (compile.cpp:25-26, 862-907). A region must have
  a single output shape (compile.cpp:982-990).
- Fused regions become a `Compiled` primitive whose Metal source is **generated as MSL text at runtime**
  (`build_kernel`, M/compiled.cpp:16-250) — contiguous, contiguous-with-N-work-per-thread, large-index, and
  strided variants for ndim 1..3 + dynamic — and compiled with `newLibraryWithSource` on first use, cached by a
  name that hashes the fused tape (M/compiled.cpp:263-330). Runtime compile uses Metal language 4.0 on macOS 26
  (M/device.cpp:64-80).
- `shapeless=True`: `compile_replace` recomputes output shapes via `Primitive::output_shapes` for new input shapes
  instead of re-tracing (compile.cpp:1048-1100); the kernels use dynamic ndim/strides.
- mlx-lm compiles only small activation helpers (`swiglu`, `silu`, ... with `@partial(mx.compile, shapeless=True)`,
  mlx-lm/mlx_lm/models/activations.py:9-41). The Qwen3 decode graph we built (§7 F census) contains exactly one
  `Compiled` node per layer (the SwiGLU), everything else is separate primitives.
- Consequence (LIKELY): MLX's compiler cannot fuse rms_norm into the next matmul, rope + KV-cache write, the
  residual add into a matmul epilogue, or q/k/v (or gate/up) into one wide GEMV. Those are exactly the fusions a
  whole-model compiler would do.

---

## 2. Metal backend

### 2.1 Queues, command buffers, commit policy (VERIFIED)

- One `MTLCommandQueue` per MLX stream (`CommandEncoder` ctor, M/device.cpp:309-324). Command buffers are made with
  `commandBufferWithUnretainedReferences` (M/device.cpp:323, 559): MLX keeps resources alive itself.
- **One compute encoder per command buffer, created with `MTLDispatchTypeConcurrent`** (M/device.cpp:576-584).
  Kernels in an encoder may overlap unless a barrier is inserted.
- **Barriers:** each dispatch's input/output buffers are recorded; before a dispatch, if it reads a buffer written
  in the current barrier epoch (RAW) or writes one read in it (WAR), MLX emits
  `memoryBarrier(MTLBarrierScopeBuffers)` (M/device.cpp:345-406). Granularity = whole MTLBuffer. In a transformer
  decode step almost every op depends on the previous one, so nearly every dispatch is preceded by a barrier
  (LIKELY; q/k/v projections and gate/up are the main independent pairs).
- **Cross-encoder sync:** each encoder gets an `MTLFence`; `end_encoding` makes the new encoder wait on the fences
  of earlier encoders that produced any of its inputs and then `updateFence`s its own (M/device.cpp:428-497).
- **Commit policy:** after each primitive, `gpu::eval` commits if `needs_commit()` (M/eval.cpp:59-68):
  `buffer_ops_ > max_ops_per_buffer || (buffer_sizes_ >> 20) > max_mb_per_buffer` (M/device.cpp:511-514).
  Defaults by architecture letter (M/device.cpp:602-624): `p` 20/40, `g` 40/40, `s` 50/50, `d` 50/50, else 40/40.
  Env overrides `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER` (mlx/utils.h:193-203). Our M2 Pro reports
  `applegpu_g14s` → 50 ops / 50 "MB".
  - **Quirk (VERIFIED):** `buffer_sizes_` accumulates `a.data_size()` of each *new input buffer*
    (M/device.cpp:348-350), and `data_size()` is in **elements**, not bytes (mlx/array.h:355-358). For 4-bit
    weights stored as uint32 the "50 MB" cap is really ~52 M elements ≈ 210 MB of packed weights.
  - Estimated command buffers per decode token (LIKELY, arithmetic from the census in §7 F): Qwen3-8B ≈ 30 M
    input elements/layer and ~19 kernels/layer → a commit every ~1.7 layers → ~20 command buffers/token;
    Qwen3-32B ≈ 76 M elements/layer → ~1 commit/layer → ~64 command buffers/token.
- `finalize(s)` at the end of every eval commits whatever is pending (M/eval.cpp:71-77; transforms.cpp:357-361).
- Completion handlers decrement `n_active_tasks` (M/eval.cpp:62-64; mlx/scheduler.h:39-52) — that counter drives
  the back-pressure in §1.2.

### 2.2 Synchronization points CPU↔GPU (VERIFIED)

| Mechanism | Where | Cost class |
|---|---|---|
| `mx.eval` / `array.item()` / `np.array(x)` | `Event::wait` → `MTLSharedEvent waitUntilSignaledValue` (M/event.cpp:27-29, mlx/array.cpp:145-162, mlx/array.h:595-614) | blocks Python thread until GPU signals |
| eval back-pressure | >10 command buffers in flight or memory > limit → `wait_for_one` (transforms.cpp:278-292) | blocks mid-eval |
| `mx.synchronize(s)` | commit + `waitUntilCompleted` (M/device.cpp:564-574) | full drain |
| cross-stream fence | `Fence` = shared event, or with `MLX_METAL_FAST_SYNCH=1` a spin kernel on a shared buffer (M/fence.cpp:9-135; default off, mlx/utils.h:220-223) | GPU-side wait |
| CPU-stream ops (e.g. weight `Load`) | GPU `encodeWait` on the event (M/event.cpp:52-61, M/device.cpp:505-509) | GPU waits for CPU thread |
| async_eval | none at call; wait deferred to the next `item()` | overlaps CPU encode with GPU |

### 2.3 Allocator (VERIFIED, M/allocator.cpp)

- All buffers: `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked` (M/allocator.cpp:15-16).
  Unified memory, CPU-visible, no driver hazard tracking (MLX does its own barriers/fences).
- Allocations < 256 B come from a 1 MiB `MTLHeap` (M/allocator.h:48-52, M/allocator.cpp:72-76, 148-150);
  larger ones are page-rounded (16 KiB) `newBuffer` (M/allocator.cpp:123-126, 151-153).
- **Buffer cache:** freed buffers go to a size-keyed multimap with LRU eviction (mlx/backend/common/buffer_cache.h:
  30-99). Reuse requires `size <= cached < min(2*size, size + 2 pages)` (buffer_cache.h:32-36). Cache limit
  (`set_cache_limit`) defaults to the memory limit (M/allocator.cpp:63-65; mlx/memory.h:65-74).
- **Memory limit** (`block_limit_`) = `min(1.5 × recommendedMaxWorkingSetSize, 0.95 × RAM)`; GC threshold
  `gc_limit_ = min(0.95 × recommendedMaxWorkingSetSize, block_limit_)`: above it, `malloc` evicts cached buffers
  first (M/allocator.cpp:63-64, 132-138). Also enforces the Metal resource-count limit (499,000 here;
  M/allocator.cpp:135-146).
- **Wired memory / residency sets:** MLX keeps allocations in `MTLResidencySet`s (macOS 15+), up to the wired
  limit set by `mx.set_wired_limit` (default 0 = nothing wired) (M/resident.h:15-26; M/allocator.cpp:100-105).
  Sets are capped at `MLX_RESIDENCY_SET_MAX_PCT` (default 5%) of the working-set size each (≥64 MiB), max 32 sets,
  each with a standing `requestResidency()` and attached to every queue (M/resident.cpp:15-46, 80-111, 221-237).
  New sets are attached to a queue just before a commit (M/device.cpp:516-520). mlx-lm raises the wired limit to
  `max_recommended_working_set_size` for generation (mlx-lm/mlx_lm/utils.py:73-82; generate.py:229-263).

### 2.4 How buffers are created; zero-copy (VERIFIED)

- Normal path: `device->newBuffer(size, Shared|Untracked)` (M/allocator.cpp:152).
- Zero-copy path exists: `make_buffer(ptr, size)` → `newBuffer(ptr, size, options, nullptr)` (= bytesNoCopy)
  (M/allocator.cpp:207-218). Used by `array(void*, shape, dtype, deleter)` (mlx/array.cpp:88-108) and by Python
  numpy/buffer import `cpu_nd_array_to_mlx_no_copy` (python/src/convert.cpp:180-225). Falls back to a copy if the
  pointer/length is not page-aligned. **The safetensors loader does not use it** (§4).

---

## 3. Kernels

Full catalog (every family, JIT vs precompiled), Steel GEMM tiles, rms_norm / rope / SDPA and the Metal 4 / NAX
analysis are in `$S/research/R3-kernels.md` (R3 fork, code-cited). Summary of what matters here:

- Default (release) build: all families precompiled into `mlx.metallib` with `-fno-fast-math`; gather/scatter,
  hadamard, `mx.compile` and custom kernels are always JIT-compiled at runtime (R3-kernels.md §a).
- Steel GEMM on `s`/`c` devices (M2 Pro, and LIKELY M5 Max = `g17s`): 64×64×16 tiles, 2×2 simdgroups,
  single-buffered threadgroup tiles, 8×8 `simdgroup_matrix` with fp32 accumulation (M/matmul.cpp:150-171,
  R3-kernels.md §b).
- Decode attention `sdpa_vector`: 1-pass, 2-pass from kL ≥ 1024 on `s`; fused vector path only when
  `qL ≤ 8` and `qL·gqa ≤ 32` (M/scaled_dot_product_attention.cpp:1001-1006, 1496) → **Qwen3-32B (GQA 8) gets the
  fused decode kernel only for qL ≤ 4**; qL 5-8 falls back to unfused matmul-softmax-matmul (R3-kernels.md §c).
- NAX (M5 Neural Accelerators): used by dense GEMM, quantized `qmm` (prefill), gather_qmm, prefill attention and
  gated-delta; **not** by gemv/qmv, decode attention, norms, rope, elementwise. Needs macOS ≥ 26.2 and GPU gen ≥ 17
  (M/device.cpp:952-971), Metal 4 `mpp::tensor_ops::matmul2d` 16×32×16 fragments, f16/bf16 (f32 via TF32 flag),
  always dequantizing to float first — no int8 MMA path (R3-kernels.md §d).

### 3.1 Quantized matmul: modes, bits, group sizes (VERIFIED)

- Modes: `affine` (bits 2,3,4,5,6,8; group 32/64/128; float32/float16/bfloat16 activations) —
  K/quantized.metal:164-182; `mxfp4` (g32, e2m1 + e8m0 scale), `nvfp4` (g16, e2m1 + e4m3 scale, optional fp32
  global scale), `mxfp8` (g32, e4m3 + e8m0) — K/fp_quantized.metal:250-262; group-size validation
  mlx/ops.cpp:5170-5180, defaults mlx/ops.cpp:4813-4830.
- Kernel variants (K/quantized.h): `qmv_quad` (K = 64/128), `qmv_fast` (aligned), `qmv` (unaligned), `qmv_wide`
  (2–5 vectors per threadgroup), `qvm` / `qvm_split_k` (x @ W, non-transposed), `qmm_t`, `qmm_t_splitk`, `qmm_n`,
  `gather_qmv(_fast)`, `gather_qvm`, `gather_qmm_t/_n`, `gather_qmm_rhs` (MoE), plus `*_nax` versions of
  qmm / gather_qmm (K/quantized_nax.h) and `QQMatmul` (quantized activations, nvfp4/mxfp8).
- Packing for affine 4-bit: 8 values per uint32 (`get_pack_factor`, K/quantized.h:17-20); scales and biases are one
  value each per group in the activation dtype. Bytes per weight at g64 with fp16 scale+bias: 0.5 + 4/64 =
  0.5625 B (4.5 bits).

### 3.2 Dispatch for `QuantizedMatmul::eval_gpu` (VERIFIED, M/quantized.cpp:1868-1962)

- `vector_limit = transpose ? get_qmv_batch_limit(K, N, device) : 4` (M/quantized.cpp:1890).
  `get_qmv_batch_limit` (M/quantized.cpp:89-147), for `K`/`N` = in/out dims:

  | device | K,N ≤ 2048 | ≤ 4096 | larger (all Qwen3-32B projections) |
  |---|---|---|---|
  | gen ≥ 17 (M5), not `d` | 33 | 25 | 13 |
  | gen 15–16 (M3/M4), not `d` | 13 | 15 | 13 |
  | gen 13–14 (M1/M2), `d` | 32 | 18 | 12 |
  | gen 13–14 (M1/M2), other (our M2 Pro `g14s`) | 14 | 10 | **6** |

- `M < vector_limit` → `dispatch_qmv` (M/quantized.cpp:1825-1866):
  - K = 64 or 128 → `qmv_quad`;
  - `M ≥ 2` and (`mode != affine` or gen ≥ 15) → `qmv_wide` (M/quantized.cpp:542-545, 1846-1851): one threadgroup
    streams each weight group once and applies it to up to 5 vectors (`vecs_per_tg = ceil(M/ceil(M/5))`,
    M/quantized.cpp:556-558). So **on M3/M4/M5, affine M=2..5 reads the weights once; M=6..10 twice, etc.**
  - otherwise `qmv` / `qmv_fast` with grid `(M, N/8, B)` — on M1/M2 every x row is a separate threadgroup that
    re-reads the same weight rows (adjacent in dispatch order, so mostly from cache — see §7 C).
- `M ≥ vector_limit`, transposed, non-batched → `qmm_splitk` (M/quantized.cpp:1152-1263): split_k chosen to
  reach ~512 threadgroups of 32×32 tiles; partial sums are stored in the **activation dtype** (fp16) and reduced
  by a separate strided-reduce kernel (M/quantized.cpp:1206-1262, intermediate at 1214). If split_k ≤ 1 → `qmm` (M/quantized.cpp:1042-1150),
  which uses `qmm_nax` when NAX is available and K % 64 == 0 (M/quantized.cpp:1062-1083).
- Non-transposed (`x @ W`): `qvm` for K < 1024, else `qvm_split_k` (M/quantized.cpp:1938-1961).

### 3.3 The decode kernel `affine_qmv_fast` exactly (VERIFIED, K/quantized.h:756-821, M/quantized.cpp:465-540)

Applies to all Qwen3-8B/32B projections at M=1 (needs N % 8 == 0 and K % 512 == 0 for 4-bit;
`qmv_fast_k_alignment`, M/quantized.cpp:149-153, 488).

- **Grid / threadgroup:** threadgroup = 64 threads = 2 simdgroups (`group_dims(32, 2, 1)`); grid =
  `(M, ceil(N/8), B)` threadgroups (M/quantized.cpp:482-496).
- **Outputs:** `results_per_simdgroup = 4` output rows per simdgroup → **8 output rows per threadgroup**
  (K/quantized.h:769-770, 785-787). (nvfp4 on gen-17 `s` with N ≥ 4096 uses 2 rows/simdgroup,
  M/quantized.cpp:489-494.)
- **Threads per output row:** all 32 lanes of the simdgroup cooperate on each of its 4 rows (split along K).
- **Values per thread per iteration:** `packs_per_thread = 2` uint32 → `values_per_thread = 16` (4-bit)
  (K/quantized.h:768-774). One simdgroup iteration covers `block_size = 16 × 32 = 512` K-values
  (K/quantized.h:774, 796-813).
- **Loads per lane per iteration:** 16 consecutive activations (32 B fp16) converted to fp32 into registers
  (`load_vector`, K/quantized.h:28-105); for each of the 4 rows, 8 B of packed weights read as four `uint16`
  (K/quantized.h:235-244) — 32 lanes × 8 B = 256 contiguous bytes per row per iteration (coalesced); one scale and
  one bias per row (4 lanes share a group: `scale_step_per_thread = 64/16 = 4`, K/quantized.h:776, 789-790, 802-803).
- **Dequant arithmetic (the trick):** activations are pre-divided by 1, 16, 256, 4096 (K/quantized.h:62-69), so a
  nibble can be used *masked but unshifted*: `x[4i]*(w&0x000f) + x[4i+1]*(w&0x00f0) + x[4i+2]*(w&0x0f00) +
  x[4i+3]*(w&0xf000)` (K/quantized.h:235-244). Affine scale/bias are applied once per 16-value chunk:
  `scale * Σ(x·q) + bias * Σx` (K/quantized.h:289), with `Σx` computed once in `load_vector` and shared by the
  4 rows. All math in fp32 (`typedef float U`).
- **Reduction:** per-lane partials over the whole K loop, then one `simd_sum` per row at the end and lane 0 writes
  (K/quantized.h:815-820). No threadgroup memory, no atomics, no split-K.
- **Activation reuse:** every simdgroup re-reads (and re-pre-scales) the full x vector from cache; for N = 25600
  that is 3,200 simdgroups × 10 KB of x (~33 MB of L1/L2 traffic per call vs 73.7 MB of DRAM weight traffic).
  (LIKELY minor: hits cache; costs load/ALU issue slots.)
- **Arithmetic per weight:** ≈ 1 AND + 1 int→float + 1 FMA (+ the amortized x pre-scale); at 182 GB/s and 0.5625
  B/weight that is ~324 G weights/s ≈ 1.0–1.3 T simple ALU ops/s against ~2.9 T instructions/s on this GPU
  (16 cores × 128 lanes × 1.398 GHz) — bandwidth-bound with roughly 2× ALU headroom on M2 Pro (LIKELY).

`qmv` (unaligned) is the same with `packs_per_thread = 1` (8 values/lane) and bounds checks (K/quantized.h:823-985).
`qmv_wide` (gen ≥ 15 for affine): k_lanes = 8 lanes per row → 4 rows per simdgroup, 2 simdgroups; decodes 8-value
sub-chunks to fp32 (`scale*q + bias`) and dots them with up to 5 streamed vectors; shuffle-ladder reduction
(K/quantized.h:987-1079, M/quantized.cpp:546-627). fp modes (`fp_qmv_fast`) are structurally identical with an
e2m1/e4m3 value decode and a uint8 scale (K/fp_quantized.h:339-410).

### 3.4 Prefill kernel `affine_qmm_t` (non-NAX) (VERIFIED, K/quantized.h:1191-1317)

- 32×32 output tile, BK = 32, 2×2 simdgroups (128 threads) (K/quantized.h:1191-1222; M/quantized.cpp:1086-1092).
- Per K-step: X tile loaded with steel `BlockLoader`, W tile dequantized into threadgroup memory by
  `QuantizedBlockLoader`, `threadgroup_barrier`, steel `BlockMMA` (8×8 simdgroup_matrix), barrier
  (K/quantized.h:1228-1306). Single-buffered, no async copy.
- On M5 (NAX) prefill goes to `qmm_t_nax` (64×64×64 tiles; BM = 32 when M ≤ 32) — R3-kernels.md §d.

---

## 4. Loading weights (VERIFIED, mlx/io/safetensors.cpp, mlx/io/load.cpp, mlx/backend/common/load.cpp)

- `mx.load(file.safetensors)` reads the 8-byte header length and the JSON header synchronously, then returns one
  **lazy** array per tensor whose primitive is `Load(stream, reader, offset)` on the **CPU stream** (on Metal
  builds `to_stream(s, Device::cpu)`) (io/safetensors.cpp:111-214, 123).
- When evaluated, `Load::eval_cpu` does `out.set_data(allocator::malloc(nbytes))` — a **new Metal shared buffer**
  — and enqueues a task that `pread`s the bytes into `buf->contents()` (backend/common/load.cpp:30-55).
  `ParallelFileReader::read(data, n, offset)` splits reads into 32 MiB batches (io/load.h:120) on a shared pool of
  `clamp(cores/2, 4, 16)` threads (io/load.cpp:340-372, 409-446).
- **So weights are copied, not mmapped.** No `mmap`, no `newBufferWithBytesNoCopy` in the load path (the no-copy
  path exists only for host pointers, §2.4). Implications (LIKELY, from the code path):
  - Load time ≈ file read time (SSD bandwidth when cold; memcpy from page cache when warm) + buffer allocation.
  - Peak physical RAM transiently holds both the page-cache copy of the file and the Metal buffer; the page cache
    is reclaimable, so the process footprint is ~1× model size, but on a 16 GB machine a 9 GB+ model forces page-cache
    eviction/compression while loading. `get_peak_memory` will show ~1× model bytes (Metal buffers only).
  - With `set_wired_limit` (mlx-lm sets it), buffers are added to residency sets as they are allocated
    (M/allocator.cpp:161-163), i.e. wired up to the limit.
  - A runtime that mmaps a page-aligned weight file and wraps it with bytesNoCopy could make load ~instant and avoid
    the double copy, at the price of first-touch page faults on the first forward pass and no control over
    eviction unless wired (EXPERIMENTAL; safetensors offsets are not page-aligned, so it needs a re-packed file).
- mlx-lm: `mx.load(wf)` per shard into one dict (mlx-lm/mlx_lm/utils.py:455), `model.load_weights`, then
  `mx.eval(model.parameters())` (all lazy until then).

## 5. Memory accounting (VERIFIED, M/allocator.cpp, mlx/memory.h)

| API | What it measures |
|---|---|
| `get_active_memory()` | Sum of `MTLBuffer.length` of buffers currently handed out by the MLX allocator, incl. no-copy wrapped buffers (M/allocator.cpp:166-167, 189, 214, 226). Lengths are page-rounded above 16 KiB, so it can exceed logical `nbytes`. Excludes the cache. |
| `get_peak_memory()` | Max of active memory since start or `reset_peak_memory()` (M/allocator.cpp:167, 215; M/allocator.h:29-35). |
| `get_cache_memory()` | Bytes of freed buffers parked in the reuse cache (buffer_cache.h:101-103). Still resident from the OS's view. |
| `get_memory_limit()` | `min(1.5 × recommendedMaxWorkingSetSize, 0.95 × RAM)` default (M/allocator.cpp:63) — a soft limit used by eval back-pressure and GC, not a hard cap. |

None of them include: Metal pipeline/library memory (the 190 MB metallib is mmapped by Metal), command-buffer and
encoder memory, CPU-side graph objects, Python heap, or the file page cache used during loading. "Peak RAM" for a
comparison with llama.cpp should therefore be measured as process physical footprint (e.g. `footprint`/`vmmap`
or `task_info` phys_footprint), not `get_peak_memory` (LIKELY).

Observed: our bandwidth script reports `peak_memory_GB 5.37` while the largest array was 2 GiB (§7 A). This is
an artifact of the script, not MLX: the reshaped view `xr` of the 2 GiB array stayed alive into the copy tests.
Check (`$S/venv/bin/python $S/research/r3_peakcheck.py`, raw output):
```
after full: active 0.537 peak 0.537 cache 0.000
after sum:  active 0.537 peak 0.537 cache 0.000
after x*2:  active 0.537 peak 1.074 cache 0.537
after zeros:active 1.074 peak 1.074 cache 0.000
```
i.e. `mx.full` allocates 1×, an elementwise op adds exactly one output buffer, and the freed output goes to the cache
(VERIFIED).

---

## 6. Known performance issues (Q6)

See `$S/research/R3-issues.md` (R3 fork; ~40 items with links and verbatim maintainer quotes). Items that bear
directly on the measurements below are cited inline in §7–§8.

---

## 7. Measurements (local M2 Pro, MLX 0.32.3 wheel)

Method common to all: scripts under `$S/research/`, run as
`bash $S/benchlock.sh $S/venv/bin/python <script>` (machine-wide lock; other agents' CPU-only jobs may run
concurrently). Wall-clock `time.perf_counter()` around `mx.eval`; warm-up first; medians reported. fp16 activations.
4-bit weights from `mx.quantize(w, group_size=64, bits=4)` on random normal ×0.02 matrices (quantized in row
chunks to bound memory, `r3_common.py:make_qweight`). Where weights could fit in the 24 MB SLC, the benchmark
rotates through enough distinct weight copies (≥ 320 MB working set) that every call streams from DRAM.
Raw outputs are the `r3_out_*.txt` files next to the scripts.

Theoretical numbers for this GPU:
- DRAM: 200 GB/s marketing; LPDDR5-6400 × 256-bit = 204.8 GB/s.
- ALU: 16 cores × 128 FP32 lanes × 2 (FMA) × 1.398 GHz = **5.73 TFLOPS** FP32; FP16 has the same rate on
  Apple GPU family 8 (metal-benchmarks README table: "M2P | 8 | 1.398 GHz | 19 cores | GOPS32 6800 | GOPS16 6800"
  for the 19-core part → 357.9 GFLOPS/core; ×16 = 5.73 TFLOPS) (`$S/src/metal-benchmarks/README.md:57, 76`; L3/SLC 24 MB for M2 Pro,
  third-party measurement, Philip Turner). Core count 16 verified by `system_profiler SPDisplaysDataType`.
  Verified empirically in §7 D (FMA probe).

### 7.A Memory bandwidth seen through MLX (VERIFIED, our measurement)

Command: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_bw.py` → `r3_out_bw_1.txt`.
15 reps each, median (and best).

| Kernel | Size | Median time | GB/s (median) | GB/s (best) | % of 204.8 |
|---|---|---|---|---|---|
| all-reduce sum fp16 | 16 MiB | 0.357 ms | 47.0 | 50.3 | (eval-overhead bound) |
| all-reduce sum fp16 | 64 MiB | 0.710 ms | 94.5 | 97.5 | |
| all-reduce sum fp16 | 256 MiB | 1.835 ms | 146.3 | 150.4 | |
| all-reduce sum fp16 | 1 GiB | 6.013 ms | 178.6 | 180.5 | 87% |
| all-reduce sum fp16 | **2 GiB** | 11.765 ms | **182.5** | 183.9 | **89%** |
| all-reduce max fp16 | 2 GiB | 11.746 ms | 182.8 | 184.2 | 89% |
| row-reduce sum (rows of 4096) | 2 GiB | 13.151 ms | 163.3 | 166.6 | 80% |
| elementwise `x*2` (R+W) | 1 GiB in + 1 GiB out | 11.818 ms | 181.7 | 183.2 | 89% |
| reversed copy `contiguous(x[::-1])` (R+W) | 1 GiB + 1 GiB | 11.829 ms | 181.5 | 182.9 | 89% |

- **Ceiling used below: 182 GB/s** (MLX-visible, large streaming kernels; 91% of the 200 GB/s marketing number,
  89% of 204.8 GB/s). Includes one eval round-trip per rep (~0.1–0.2 ms) — negligible at 2 GiB.
- MLX's own simple kernels reach the same ceiling as each other; MLX row-reduce is ~10% slower than all-reduce.
- Small sizes are dominated by per-eval overhead (16 MiB: 0.36 ms vs 0.09 ms of pure transfer).

### 7.B Decode GEMV, M = 1 (VERIFIED, our measurement)

Command: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_qmv.py` → `r3_out_qmv_1.txt`
(repeat run: `r3_out_qmv_2.txt`). Kernel: `affine_qmv_fast_float16_t_gs_64_b_4_batch_0` (§3.3).
Correctness spot check vs fp32 dequantized reference: max |err| 4.0e-4 at max |ref| 0.37.
- (i) single = one `quantized_matmul` per `mx.eval`, median of 200;
- (ii) batched = 24 independent calls (distinct weights) in one `mx.eval`, time/24, median of 25 evals.
- Bytes = packed weights + fp16 scales + fp16 biases (0.5625 B/weight). % is of the 182 GB/s ceiling.

| Shape (in→out) | MB | single run1 / run2 (µs) | single GB/s (% of 182) | batched run1 / run2 (µs) | batched GB/s (% of 182) |
|---|---|---|---|---|---|
| 8B q/o_proj 4096→4096 | 9.44 | 203.7 / 205.9 | 46 (25%) | 62.0 / 68.9 | 137–152 (75–84%) |
| 8B k/v_proj 4096→1024 | 2.36 | 164.0 / 170.0 | 14 (8%) | 23.4 / 24.2 | 98–101 (54–55%) |
| 8B gate/up 4096→12288 | 28.31 | 339.3 / 320.0 | 83–88 (49%) | 182.4 / 230.4 | 123–155 (68–85%) |
| 8B down 12288→4096 | 28.31 | 317.0 / 321.6 | 88–89 (49%) | 167.9 / 171.4 | 165–169 (91–93%) |
| 8B lm_head 4096→151936 | 350.06 | 2002.6 / 2182.1 | 160–175 (96%) | 2004.6 / 2023.4 | 173–175 (95–96%) |
| 32B q_proj 5120→8192 | 23.59 | 295.6 / 293.9 | 80 (44%) | 145.6 / 141.4 | 162–167 (89–92%) |
| 32B k/v_proj 5120→1024 | 2.95 | 170.9 / 168.2 | 17–18 (10%) | 25.7 / 25.8 | 114–115 (63%) |
| 32B o_proj 8192→5120 | 23.59 | 312.5 / 293.2 | 76–80 (44%) | 142.0 / 139.8 | 166–169 (91–93%) |
| 32B gate/up 5120→25600 | 73.73 | 616.9 / 588.2 | 120–125 (69%) | 446.0 / 429.1 | 165–172 (91–94%) |
| 32B down 25600→5120 | 73.73 | 598.6 / 584.0 | 123–126 (69%) | 416.1 / 419.2 | 176–177 (97%) |
| 32B lm_head 5120→151936 | 437.58 | 2515.5 / 2563.7 | 171–174 (96%) | 2480.8 / 2547.5 | 172–176 (94–97%) |

Reading:
- **Single-call floor ≈ 160–170 µs** (k/v_proj: ~2.4–3 MB of weights takes 164–171 µs). That is one
  build + eval + commit + GPU + `MTLSharedEvent` wake-up round trip, not kernel time (see 7.E: 149–158 µs per
  synchronous eval of a trivial op). Any one-op-per-eval benchmark on Apple GPUs measures this floor
  (compare zcbenz's per-call timings with a ~0.28 ms floor on a machine inferred to be an M5 Max,
  R3-issues.md C5).
- **Batched (≈ kernel throughput): large matrices run at 91–97% of the 182 GB/s ceiling**; lm_head 94–97%.
  Small-N matrices are latency/occupancy-bound: k/v_proj (N = 1024 → 128 threadgroups of 64 threads on 16 cores)
  reaches only 54–63%, 8B q/o (N = 4096) 75–84%.
- Run-to-run spread is up to ~±10% on a few rows (8B gate/up 182 vs 230 µs) — other agents' CPU jobs ran
  concurrently; the lock serializes only benchmarks.
- Sum over one layer using the better of the two batched medians:
  - Qwen3-8B: 703.5 µs for 108.5 MB → 154 GB/s (85%); ×36 + lm_head = 27.3 ms/token of GEMV vs 23.4 ms ideal
    at 182 GB/s → matmuls at **86%** of ceiling.
  - Qwen3-32B: 1606.9 µs for 274.3 MB → 171 GB/s (94%); ×64 + lm_head = 105.3 ms/token vs 98.9 ms ideal
    → **94%** of ceiling (≈ 9.5 tok/s if a 32B model fit in this 16 GB machine).
  These are optimistic: independent calls in one eval run without barriers and may overlap; §7.F measures a real
  dependent layer stack.

### 7.C Verification shapes: M = 1..32 on Qwen3-32B shapes (VERIFIED, our measurement)

Command: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_smallm.py` → `r3_out_smallm_1.txt`.
Batched = 16 independent calls per eval (distinct weights, ≥ 320 MB working set), median of 10 evals.
On this M2 Pro (gen 14, `s`): M ≤ 5 → `qmv_fast` with one threadgroup row per x row; M ≥ 6 → `qmm_splitk` /
`qmm` (vector_limit = 6 for these shapes, §3.2).

Batched µs per call (× vs M=1):

| Shape | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=8 | M=12 | M=16 | M=24 | M=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32B q_proj | 151 (×1.00) | 202 (×1.34) | 282 (×1.87) | 371 (×2.46) | 458 (×3.04) | 667 (×4.43) | 670 (×4.45) | 694 (×4.61) | 684 (×4.54) | 686 (×4.55) | 644 (×4.27) |
| 32B k/v_proj | 30 (×1.00) | 35 (×1.19) | 47 (×1.59) | 55 (×1.87) | 66 (×2.26) | 112 (×3.81) | 113 (×3.85) | 115 (×3.90) | 116 (×3.93) | 120 (×4.08) | 111 (×3.78) |
| 32B o_proj | 200 (×1.00) | 195 (×0.97) | 275 (×1.38) | 356 (×1.78) | 438 (×2.19) | 686 (×3.43) | 683 (×3.41) | 687 (×3.44) | 718 (×3.59) | 686 (×3.43) | 671 (×3.36) |
| 32B gate/up | 435 (×1.00) | 583 (×1.34) | 849 (×1.95) | 1114 (×2.56) | 1396 (×3.21) | 2057 (×4.73) | 2054 (×4.72) | 2081 (×4.79) | 2065 (×4.75) | 2073 (×4.77) | 1918 (×4.41) |
| 32B down | 480 (×1.00) | 532 (×1.11) | 785 (×1.63) | 1088 (×2.26) | 1344 (×2.80) | 2109 (×4.39) | 2126 (×4.43) | 2132 (×4.44) | 2122 (×4.42) | 2176 (×4.53) | 1985 (×4.13) |
| 32B lm_head | 2647 (×1.00) | 3543 (×1.34) | 5015 (×1.89) | 6603 (×2.49) | 8157 (×3.08) | 12072 (×4.56) | 12048 (×4.55) | 12325 (×4.66) | 12477 (×4.71) | 12253 (×4.63) | 11275 (×4.26) |

Single-eval µs (× vs M=1) — includes the ~165 µs sync floor, so ratios are compressed:

| Shape | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=8 | M=16 | M=32 |
|---|---|---|---|---|---|---|---|---|---|
| 32B q_proj | 284 | 330 (×1.16) | 405 (×1.43) | 513 (×1.81) | 590 (×2.08) | 1015 (×3.58) | 846 (×2.98) | 878 (×3.09) | 842 (×2.97) |
| 32B gate/up | 559 | 710 (×1.27) | 966 (×1.73) | 1256 (×2.25) | 1537 (×2.75) | 2277 (×4.07) | 2267 (×4.06) | 2286 (×4.09) | 2127 (×3.80) |
| 32B down | 638 | 683 (×1.07) | 939 (×1.47) | 1229 (×1.93) | 1497 (×2.35) | 2395 (×3.76) | 2328 (×3.65) | 2366 (×3.71) | 2411 (×3.78) |
| 32B lm_head | 2546 | 3386 (×1.33) | 4982 (×1.96) | 6528 (×2.56) | 8165 (×3.21) | 12427 (×4.88) | 12344 (×4.85) | 12680 (×4.98) | 11592 (×4.55) |

Findings:
- **Cost grows from M = 2 on; there is no flat region on this machine.** M=2 +11–34%, M=3 +38–95%, M=4
  +78–156%, M=5 +119–221% (batched). `qmv_fast` launches one threadgroup row per x row, so each extra row redoes the
  dequant work and re-reads the weights (mostly from cache; the time per extra row is ~0.6–0.7× of an M=1 pass).
- **M = 6..32 is a flat "dead zone" at ~4.4–4.8× the M=1 cost**: `qmm` computes a full 32-row tile regardless of M.
  At M=32 it runs at 2·32·5120·25600 / 1.918 ms = **4.4 TFLOPS** (77% of 5.73) — i.e. compute-bound at a decent
  efficiency, but rows 6..31 pay for 32.
- Arithmetic crossover on this GPU (LIKELY): 5.73 TFLOPS / 182 GB/s = 31.5 FLOP/byte; at 0.5625 B and 2 FLOP per
  weight per row the GEMV stays bandwidth-bound up to M ≈ 9 if dequant were free, ≈ 5–6 rows counting the dequant
  ALU (~3 ops/weight). An ideal kernel would be ~flat (≤ 1.1–1.3×) to M ≈ 5 and reach ~2× around M ≈ 8–10; MLX is
  at 3.2× (M=5) and 4.7× (M=8).
- **Emulated M3+ routing on M2 hardware** (`MLX_METAL_GPU_ARCH=applegpu_g15s`, routes affine M ≥ 2 to `qmv_wide`;
  `r3_out_smallm_g15_1.txt`): much worse on M2 — gate/up M=2 ×2.05, M=4 ×6.35, M=12 ×13.99; down M=4 ×4.72.
  This confirms MLX's comment that affine `qmv_wide` only beats `qmv` on gen ≥ 15 (M/quantized.cpp:541-545), and
  means **M2 Pro small-M numbers are not representative of M5 Max**. Third-party M5 Max data (contributor, mlx#3791,
  whole 70B forward): M=3 ×1.12, M=5 ×1.48, M=8 ×2.25, M=16 ×3.9 (R3-issues.md E4).

### 7.D Prefill throughput and ALU peak (VERIFIED, our measurement)

Commands: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_prefill.py` → `r3_out_prefill_1.txt`;
`bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_fma.py` → `r3_out_fma_1.txt`.

Peak-ALU probe (custom Metal kernel, 8 independent `float4`/`half4` FMA chains, 262,144 threads):

| dtype | TFLOPS (median) | % of 5.726 theory |
|---|---|---|
| fp32 | 5.336–5.363 | 93.2–93.7% |
| fp16 | 5.336–5.397 | 93.2–94.3% |

→ The theoretical **16 × 128 × 2 × 1.398 GHz = 5.73 TFLOPS** is consistent with what the hardware delivers
(~94% reachable), and **fp16 runs at the same rate as fp32 on M2 Pro** (VERIFIED). (A first probe with scalar FMAs
reached only 2.08/3.41 TFLOPS — loop overhead; superseded.)

Prefill GEMMs, Qwen3-32B MLP shapes, 7 reps, median:

| Shape | M | 4-bit g64 (`qmm_t`, 32×32 tiles) | fp16 (Steel 64×64×16) |
|---|---|---|---|
| gate/up 5120→25600 | 512 | 30.49 ms, **4.40 TFLOPS (76.9%)** | 26.98 ms, **4.98 TFLOPS (86.9%)** |
| gate/up 5120→25600 | 2048 | 120.95 ms, 4.44 TFLOPS (77.5%) | 106.72 ms, 5.03 TFLOPS (87.9%) |
| down 25600→5120 | 512 | 30.64 ms, 4.38 TFLOPS (76.5%) | 27.74 ms, 4.84 TFLOPS (84.5%) |
| down 25600→5120 | 2048 | 121.09 ms, 4.43 TFLOPS (77.4%) | 108.41 ms, 4.95 TFLOPS (86.5%) |

- fp16 GEMM is at 85–88% of theory (90–93% of the probe peak): near the limit.
- The quantized prefill kernel is 11–13% slower than fp16 at the same shapes (77% of theory): modest headroom on
  non-NAX GPUs (e.g. bigger tiles, or dequantize-to-fp16 then Steel GEMM) (LIKELY).
- No small-M penalty between M=512 and M=2048.
- Implied matmul-only prefill for Qwen3-32B on this M2 Pro: 2 × 31.2 G FLOP/token at 4.4 TFLOPS ≈ 70 tok/s.
  On M5 Max prefill runs on NAX (`qmm_t_nax`, 64×64×64; R3-kernels.md §d); MLX fp16 GEMM on M5 Max measured at
  52.7–59.6 TFLOPS by a maintainer (jagrit06, mlx#3211; R3-issues.md).

### 7.E Dispatch overhead (VERIFIED, our measurement)

Command: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_dispatch.py` → `r3_out_dispatch_1.txt`.
`x = x + 1` on a 1024-element fp16 vector; medians of 200 (N ≤ 100) or 30 evals.

| N ops | graph build only | build + eval, dependent chain | build + eval, N independent ops | `mx.compile`d chain |
|---|---|---|---|---|
| 1 | 1.0 µs | 158.1 µs | 143.6 µs | 157.5 µs |
| 10 | 8.7 µs | 199.0 µs | 175.9 µs | 154.8 µs |
| 100 | 77.3 µs | 472.9 µs | 433.1 µs | 190.9 µs |
| 1000 | 795.6 µs | 3054.4 µs | 3359.7 µs | 575.5 µs |
| 3000 | 2446.6 µs | 9632.7 µs | 11414.5 µs | 1290.6 µs |

- **Marginal cost per dependent tiny op: 2.9–3.2 µs**, of which **~0.8 µs is graph construction** (Python +
  C++ op creation) and **~2.1–2.4 µs is eval + encode + GPU** (N = 100–3000).
- Independent ops cost slightly more (3.4–3.8 µs/op): more live outputs to track.
- **Synchronous eval round trip: 149–158 µs** (100 × (op + `mx.eval`) = 14.9 ms). With `async_eval` pipelining:
  **28.9 µs per eval**.
- `mx.compile` fuses the chain into kernels of ≤ 11 ops (compile.cpp:25), so 1000 adds → ~91 kernels: 575 µs.
- Comparison with real kernels (7.B batched): 2.1–2.4 µs is 0.5% of a 32B gate/up GEMV (430–446 µs), ~9% of a
  32B k/v GEMV (26 µs), and larger than an rms_norm/rope/add on one token (LIKELY a few µs each).

### 7.F Whole decode step: per-token time and the framework floor (VERIFIED, our measurement)

Script `r3_bench_model_overhead.py` builds an mlx-lm `qwen3.Model` (mlx-lm 0.32.0 code) with random fp16
weights, `nn.quantize(group_size=64, bits=4)`, prefills `ctx` tokens, then measures decode exactly like
`mlx_lm.generate_step` (build step t+1, `async_eval`, then `.item()` on step t). Runs: `r3_out_F_1.txt`
(first run; real-dims rows there were perturbed by a 6.9 GB init peak and one OOM, superseded), `r3_out_F_2.txt`,
`r3_out_F_3.txt`, census `r3_out_census_L36.txt` / `r3_out_census_L64.txt`.
Command form: `bash $S/benchlock.sh $S/venv/bin/python $S/research/r3_bench_model_overhead.py <mode> <layers> <ctx>`.

Graph census of one decode token (`mx.export_to_dot`, primitive nodes):

| Primitive | 36 layers (8B) | 64 layers (32B) |
|---|---|---|
| QuantizedMatmul | 253 | 449 |
| RMSNorm | 145 | 257 |
| RoPE | 72 | 128 |
| SliceUpdate (KV write) | 72 | 128 |
| Add (residual) | 72 | 128 |
| ScaledDotProductAttention | 36 | 64 |
| Compiled SwiGLU | 36 | 64 |
| Gather ×3 + dequant, LogSumExp, Subtract, ArgReduce | 7 | 7 |
| **Kernel-launching primitives** | **~693** | **~1,225** |
| Views (Reshape, Transpose, Slice, Squeeze, ...) | 367 | 647 |
| **Total nodes** | 1,060 | 1,872 |

(SDPA may add a second kernel in 2-pass mode at kL ≥ 1024. MLX's compile fuses only the SwiGLU.)

Per-token decode time:

| Config | build only | sync step | **pipelined (mlx-lm style)** | weights streamed / token | effective GB/s | vs 182 GB/s ceiling |
|---|---|---|---|---|---|---|
| tiny dims, 36 layers | 0.73 ms | 4.36 ms | **2.37 ms** (422 tok/s) | ~0 | — | framework floor |
| tiny dims, 64 layers | 1.21–1.28 ms | 5.5–7.5 ms | **4.25–4.35 ms** (230–235 tok/s) | ~0 | — | framework floor |
| tiny 64, single command buffer (env) | 1.21–1.39 ms | 8.0–8.1 ms | 4.14–4.16 ms | ~0 | — | −2…−5% |
| Qwen3-8B dims, 8 layers, ctx 128 | 0.30 ms | 7.76 ms | **7.11 ms** | 1.218 GB | **171.3** | **94%** |
| same, single command buffer (env) | 0.22 ms | 7.74 ms | 7.12 ms | 1.218 GB | 171.1 | 94% (no change) |
| Qwen3-32B dims, 3 layers, ctx 128 | 0.09 ms | 7.99 ms | **7.15 ms** | 1.260 GB | **176.2** | **97%** |
| Qwen3-8B dims, 8 layers, ctx 1024 | 0.19 ms | 8.13 ms | 7.74 ms | 1.218 + 0.034 KV | 162 (incl. KV) | 89% |
| Qwen3-8B dims, 8 layers, ctx 4096 | 0.19 ms | 8.75 ms | 9.49 ms | 1.218 + 0.134 KV | 142 (incl. KV) | 78% |

("single command buffer" = `MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=100000`. With those limits and
the original one-shot model init the GPU ran out of memory — `kIOGPUCommandBufferCallbackErrorOutOfMemory` in
`r3_out_F_1.txt` — the trade-off reported in mlx#4562: bigger command buffers keep intermediates alive.)

Findings:
- **At short context, MLX's whole decode step is at 94–97% of the bandwidth ceiling** for 8B/32B-shaped layers
  (attention, norms, rope, KV writes, lm_head and sampling included). The command-buffer policy makes no
  difference for real-size layers (7.112 vs 7.121 ms).
- **Framework floor ≈ 3.5 µs per kernel**: the 64-layer Qwen3 graph costs ~4.2–4.4 ms/token even with negligible
  math (1,225 kernels, 1,872 nodes). Building the graph in Python is 1.2–1.4 ms of it; the rest is eval/encode and
  GPU launch/barrier time. When pipelined, it hides behind the GPU as long as the GPU needs longer per token —
  true for 32B (≥ 100 ms here, ~30 ms on M5 Max), **false for small draft models** (the floor scales ~0.066 ms per layer, so a 28-layer Qwen3-0.6B drafter is capped near
  ~1.9 ms/token ≈ 500–550 tok/s on M2 Pro before any math, comparable to its ~2 ms of weight streaming;
  LIKELY, scaled from the 36/64-layer tiny runs).
- **Attention is where decode drops off with context**: ctx 128 → 4096 adds 2.38 ms per token for 8 layers while
  the KV bytes read grow by only ~0.1 GB → the attention part runs at ~56 GB/s effective, ~3× below the ceiling
  (VERIFIED time; mechanism LIKELY: `sdpa_vector` re-reads K/V once per query head, 8× for GQA 8, and the
  GQA-deduplicating variant only starts at kL ≥ 8192 — R3-kernels.md §c). Extrapolated to Qwen3-32B
  (64 layers, same 8 KV heads × 128): ~+19 ms/token at 4K context on M2 Pro vs ~6 ms at bandwidth (LIKELY).

### 7.G Experiment: a simple multi-vector 4-bit GEMV (EXPERIMENTAL kernel, VERIFIED timings)

Scripts: `r3_bench_qmv_multi.py` → `r3_out_qmv_multi_1.txt`; `r3_bench_qmv_multi_chunk.py` →
`r3_out_qmv_multi_chunk_1.txt` (both through `bash $S/benchlock.sh`).
~60 lines of MSL in `mx.fast.metal_kernel`, same weight layout as MLX affine 4-bit g64: 2 simdgroups × 4 rows per
threadgroup, 8 nibbles (one uint32) per lane per row per step, each weight word decoded once and applied to MV
activation rows held in registers. Max abs error vs fp32 reference 0.8–1.2e-3 (MLX's own kernel: 2.1–3.0e-3) at
|ref| ≈ 2.4–2.9.

Batched µs per call, × vs MLX at M=1 (same process, same weights):

| Shape | M | MLX | ours, one launch | ours, launches of ≤ 3 rows |
|---|---|---|---|---|
| 32B gate/up | 1 | 419–460 (×1.00) | 441 (×0.96 of MLX) | 427 (×1.02) |
| 32B gate/up | 2 | 590 (×1.28) | 498 (×1.08) | — |
| 32B gate/up | 3 | 852–883 (×1.92–2.03) | 598 (×1.30) | 574 (×1.37) |
| 32B gate/up | 4 | 1116–1160 (×2.52–2.67) | 1030 (×2.24) | 962 (×2.30); ≤2-row launches 891 (×2.13) |
| 32B gate/up | 6 | 2053–2116 (×4.60–4.90) | 5604 (register spill) | 1122 (×2.68) |
| 32B gate/up | 8 | 2047–2071 (×4.50–4.89) | 7290 (register spill) | 1559 (×3.72) |
| 32B down | 3 | 788–795 (×1.90–1.92) | 568 (×1.35) | 533 (×1.30) |
| 32B down | 6 | 2070–2086 (×4.97–5.05) | 5359 (spill) | 1031 (×2.52) |
| 32B down | 8 | 2055–2083 (×4.96–5.01) | 7151 (spill) | 1464 (×3.57) |
| 32B q_proj | 2 / 3 | 200 / 285 (×1.30 / ×1.85) | 162 / 202 (×1.05 / ×1.31 of MLX M=1) | — |

- At M = 1 the simple kernel matches MLX (so MLX's qmv is not leaving bandwidth on the table).
- **At M = 2–3 the same bandwidth-bound design costs 1.05–1.37× of a single-row pass, vs 1.27–2.03× for MLX on
  M2 Pro**: 30–33% less time at M = 3.
- The one-launch kernel spills registers from M = 4 (it keeps 8·M fp32 activations + 4·M accumulators per lane).
  Separate launches do not share weight traffic (each re-streams from DRAM), so chunking gives ×2.5–2.7 at M = 6
  and ×3.6–3.7 at M = 8 — still ~1.4–1.9× faster than MLX. A kernel that keeps register use flat in M (fp16
  activation registers, fewer rows per simdgroup) should get M = 4–8 to ~1.3–2× (LIKELY, not built).
- Caveat: this is M2 Pro. On M5 MLX uses `qmv_wide` for these M, which is much better there (7.C).

---

## 8. Conclusions

### 8.1 Where MLX is already at the hardware limit (negative results)

1. **Large-matrix decode GEMV**: 91–97% of the 182 GB/s MLX-visible ceiling (7.B); a hand-written kernel did not
   beat it at M = 1 (7.G).
2. **Whole decode step at short context**: 94–97% of the ceiling for 8B- and 32B-shaped layers, including attention,
   norms, rope, KV-cache writes, lm_head and sampling (7.F). The commit policy changes nothing at these sizes (7.F),
   consistent with the dense-model report in mlx#4521.
3. **The ceiling itself**: every MLX streaming kernel (reduce, copy, elementwise) tops out at 181.5–184 GB/s =
   89–90% of 204.8 GB/s (7.A). Whether raw Metal gets more is for the raw-Metal agent; MLX is not the limiter.
4. **fp16 prefill GEMM**: 85–88% of the 5.73 TFLOPS ALU peak (90–93% of the probe-measured 5.40) (7.D).
5. **CPU overhead for a 32B model at batch 1**: ~4.3 ms/token of framework work for 64 layers on this M2 Pro,
   hidden by `async_eval` pipelining whenever GPU time per token is larger (7.F).

### 8.2 Where measurable headroom exists

| Area | Evidence | Size of the gap |
|---|---|---|
| Small-M (speculative verification) | 7.C, 7.G; M5 Max data in R3-issues.md E2/E4/E5 | M2 Pro: M=3 costs ×1.9–2.0 vs ×1.3 achievable; M=5 ×3.2; M=6–31 pay the M=32 tile (×4.4–4.8). M5 Max (qmv_wide): ×1.12 / ×1.48 / ×2.25 at M=3/5/8, whole 70B forward, contributor data |
| Decode attention vs context | 7.F | ~56 GB/s effective for attention at 4K context with GQA 8 → +33% per-token time at 4K for 8B-shaped layers |
| Per-kernel framework floor | 7.E, 7.F | ~2–3.5 µs per kernel, ~4.3 ms per 64-layer token; matters for draft models and non-pipelined loops, not for 32B decode |
| Sync round trip | 7.B, 7.E | 149–170 µs per blocking eval (vs 29 µs pipelined); any per-token host sync (CPU sampling, stop checks) pays it |
| Small matrices | 7.B | k/v_proj at 54–63% of ceiling even batched (occupancy); fusing q/k/v (and gate/up) into one wide GEMV fixes it, but k/v are only ~2% of 32B bytes |
| Quantized prefill (non-NAX) | 7.D | 77% of ALU peak vs 88% for fp16 GEMM on the same shapes |
| Loading | §4 | Copy-based (pread into new buffers); no mmap/no-copy path for weights |

### 8.3 What a whole-model compiler could and could not gain against MLX (Qwen3-32B 4-bit, batch 1)

- **Could not (short context):** more than ~3–6% on decode tok/s by fusion, megakernels or better scheduling. The
  step is weight-streaming bound and MLX already streams at 94–97% of what its own kernels can reach on this
  machine. Fusing norms/rope/residuals removes dispatches, not bytes (7.F, 8.1). Maintainers have declined
  megakernels for the same reason (R3-issues.md C3, D7).
- **Could:**
  1. *More tokens per weight pass* — speculative decoding (draft model or prompt lookup) with a verify kernel
     whose cost stays near M=1 up to M ≈ 5–8. With MLX on M5 Max a k=4 verify (M=5) costs ~1.5× a decode step
     (third-party); a flat kernel at ~1.1–1.2× would raise the value of every accepted token (LIKELY).
  2. *Long-context attention*: GQA-aware decode attention that reads K/V once per KV head (7.F; llama.cpp is ~2×
     faster at 146K context per mlx-lm#763, R3-issues.md F).
  3. *Low-overhead execution for small models*: a pre-recorded, fixed-shape dispatch list (no per-token graph
     build/eval) removes the ~4 ms/token floor that caps draft models (7.F).
  4. *Prefill*: quantized GEMM efficiency on non-NAX parts (7.D); on M5, NAX tile selection and int8 paths are
     open questions (R3-issues.md, EXPERIMENTAL).
  5. *Load time, cold TTFT, peak RAM*: page-aligned weight file + mmap + bytesNoCopy instead of pread-copy (§4;
     EXPERIMENTAL; the steady-state RAM gain is small because Metal wires the buffers on use).

### 8.4 Open questions (not answerable on this M2 Pro)

- M5 Max: `qmv_wide` and NAX behaviour for Qwen3-32B shapes at M = 2..32 (vector_limit 13 for these shapes; M ≥ 13
  goes to `qmm_splitk`, which has no NAX version — R3-kernels.md).
- M5 Max: per-kernel GPU gap (barrier drain + launch) — sets the true megakernel upside; here it is ≤ 3% at 32B
  dims (7.F, 97% of ceiling).
- Whether raw Metal reaches more than 182 GB/s on M2 Pro (raw-Metal agent).
- Energy per token (not measured here).

---

## 9. Architect brief (R3: MLX core + microbenchmarks)

**Bottom line.** At batch 1 and short context, MLX decode is weight-streaming-bound and already runs at
94–97% of the bandwidth its own kernels can reach. On our M2 Pro that ceiling is 182 GB/s, i.e. 89% of the
204.8 GB/s theoretical. A whole-model compiler cannot win much there (≤ 3–6%) without changing bytes per token
(quantization) or tokens per weight pass (speculation) (VERIFIED, our measurements; M5 Max extrapolation LIKELY).
The real openings, in order:
1. the cost of verifying several tokens at once (small M);
2. decode attention at longer context with GQA;
3. the per-token framework floor, which matters for draft models and any host round trip;
4. quantized prefill on non-NAX parts (NAX tuning on M5 is unmeasured);
5. load time and cold TTFT via zero-copy weights.

**Key numbers** (M2 Pro 16-core, MLX 0.32.3, fp16 activations, 4-bit g64 affine; all VERIFIED with the commands
in §7):

| Measurement | Result |
|---|---|
| Bandwidth seen through MLX (2 GiB reduce, 1 GiB copy) | 182–184 GB/s (89–90% of 204.8) |
| Qwen3-32B GEMVs at M=1, 24 per eval | 162–177 GB/s (89–97%); k/v_proj 114 GB/s (63%) |
| The same, one call per `mx.eval` | 160–170 µs floor; sync round trip 149–158 µs |
| Cost per dependent tiny op | 2.9–3.2 µs (graph build 0.8, eval/encode/GPU 2.1–2.4) |
| Qwen3 decode graph, 64 layers | 1,872 nodes, ~1,225 kernels; floor 4.2–4.4 ms/token with no math |
| Whole decode step, ctx 128, mlx-lm-style pipelining | 32B dims: 176 GB/s (97%); 8B dims: 171 GB/s (94%) |
| One command buffer per token instead of default policy | 0% on real dims; −2 to −5% on tiny dims |
| 8B dims at ctx 4096 vs 128 | +33% time; attention ≈ 56 GB/s effective |
| Verify cost at M = 2 / 3 / 5 / 8 (32B shapes) | ×1.0–1.3 / ×1.4–2.0 / ×2.2–3.2 / ×3.4–4.7 |
| Simple custom multi-row kernel at M=3 | ×1.30–1.37 (MLX ×1.9–2.0) |
| ALU peak | 5.73 TFLOPS theory; probe 5.36–5.40 (fp16 = fp32 rate) |
| Prefill, 32B MLP, M = 512 / 2048 | 4-bit 4.38–4.44 TFLOPS (77%); fp16 4.84–5.03 (85–88%) |

**How MLX executes (VERIFIED, code at 0840c42).**
- **Graph and eval.** The graph is lazy. `eval` builds a BFS tape (width 20) and encodes every primitive
  synchronously on the calling Python thread (transforms.cpp:80-365).
  - One compute encoder per command buffer, `MTLDispatchTypeConcurrent`, with barriers at whole-buffer
    granularity (M/device.cpp:345-406, 576-584).
  - Commit after 50 ops or ~52 M input *elements* on `s`-class GPUs: M2 Pro, and LIKELY M5 Max (M/device.cpp:511-514, 602-626).
  - At most 10 command buffers in flight; host sync is an `MTLSharedEvent` wait.
- **Per-primitive CPU work.** Every primitive allocates an autorelease pool, looks up its kernel by name, does
  hash-set hazard tracking, and adds a completion handler.
- **`mx.compile`** fuses only elementwise ops, at most 11 deep, into runtime-generated MSL. mlx-lm compiles only
  the SwiGLU.
- **Allocator.** Shared-storage, untracked buffers; a size-bucket cache; residency sets once a wired limit is set.
- **Loading.** Weights are `pread` into freshly allocated buffers, not mmapped (io/safetensors.cpp,
  backend/common/load.cpp).
- **`get_peak_memory`** counts only MLX buffers. Measure peak RAM as process footprint.
- **Decode kernel `affine_qmv_fast`** (K/quantized.h:756-821):
  - 64-thread threadgroups (2 simdgroups × 4 rows = 8 output rows); the 32 lanes split K.
  - Each lane handles 16 nibbles per step: 8 B of weights per row and 32 B of activations, which are reused
    across the 4 rows.
  - Dequant: activations are pre-divided by 1/16/256/4096 so nibbles are masked but never shifted. Scale and
    bias are applied once per 16 values as `s·Σxq + b·Σx`.
  - fp32 accumulation, one `simd_sum` per row.
- **Routing for M > 1.**
  - M3 and later: M ≥ 2 goes to `qmv_wide` (weights read once for ≤ 5 vectors).
  - M2: each row re-runs `qmv`.
  - Above `vector_limit` (6 on M2 Pro, 13 on M5 for 32B shapes): split-K `qmm` (no NAX version), or `qmm_nax`
    on M5.
- **NAX** requires macOS ≥ 26.2 and GPU generation ≥ 17. It uses `matmul2d` 16×32×16 fragments, f16/bf16
  only, no integer MMA. It runs dense GEMM, prefill `qmm` and prefill attention, never qmv, decode attention,
  norms or rope (R3-kernels.md).

**Negative results (MLX at the limit).**
- Large-matrix qmv runs at 91–97% of the ceiling. A hand-written kernel only matched it at M=1.
- The whole decode step is at 94–97% at short context.
- Changing command-buffer policy did nothing at real sizes.
- fp16 GEMM reaches 88% of ALU peak.
- The 182 GB/s ceiling is the same for every MLX streaming kernel, so MLX is not what limits bandwidth.
- Maintainers have rejected megakernels and whole-step compile because they "would end up launching exact same
  graph" (R3-issues.md C3, D7). Our data agree for dense 32B.

**Headroom, with evidence.**
1. **Small M.** On M2 Pro, MLX pays ×1.9–2.0 at M=3, ×2.2–3.2 at M=5, and the full 32-row tile (×4.4–4.8)
   for every M from 6 to 31.
   - A 60-line kernel gets M=3 to ×1.30–1.37 (VERIFIED timing, EXPERIMENTAL kernel). It spills registers from
     M=4.
   - On M5 Max, `qmv_wide` is better but still ×1.12 / ×1.48 / ×2.25 at M = 3 / 5 / 8 for a whole 70B forward
     (contributor data, mlx#3791).
   - Ideal is roughly flat to M ≈ 5 and ×2 around M ≈ 8–10 (arithmetic-intensity estimate, LIKELY).
   - Attention adds a second gap: with GQA 8, MLX's fused decode attention handles only up to 4 queries; 5–8
     fall back to unfused attention (R3-kernels.md).
2. **Attention vs context.** Going from 128 to 4K context costs +2.4 ms per 8 layers, against ~0.7 ms if
   attention ran at bandwidth. The LIKELY cause is K/V being re-read per query head; the GQA-deduplicating path
   only starts at 8K.
   - Extrapolated to Qwen3-32B at 4K on M2 Pro: about +19 ms/token instead of ~6 ms (LIKELY).
3. **Framework floor.** About 3.5 µs per kernel, or ~4.3 ms per 64-layer token. It is hidden behind the GPU
   for 32B decode, but it caps a 28-layer 0.6B drafter near 500–550 tok/s on M2 Pro.
   - Every blocking host sync costs ~150 µs, versus 29 µs pipelined.
4. **Quantized prefill on non-NAX GPUs** runs 11–13% behind fp16 GEMM on the same shapes.
5. **Loading** is copy-based, so zero-copy weights help load time and cold TTFT, not steady-state RAM
   (EXPERIMENTAL).

**Recommendations for Sky.**
1. Treat parity on dense batch-1 decode as table stakes, not the pitch. Large GEMVs must stream at ≥ 95% of
   measured bandwidth. Test with weight sets larger than the SLC (24 MB on M2 Pro) and many calls per
   submission.
2. Make speculative decoding the decode differentiator.
   - Build a verify GEMV whose cost stays ≤ 1.2–1.3× of M=1 up to M ≈ 5–8: weights decoded once,
     register use flat in M (fp16 activation registers, fewer rows per simdgroup).
   - Pair it with GQA-aware fused attention for 2–16 queries.
   - Use prompt-lookup or a small drafter. Qwen3-32B has no MTP head.
3. Write GQA-aware decode attention that reads K/V once per KV head, with split-K over the context. That
   removes the measured 3× gap at 4K and up.
4. Use a static, pre-recorded dispatch schedule for fixed shapes, with no per-token graph build or eval.
   - Never block on the host per token: keep sampling on the GPU and pipeline submission like mlx-lm's
     `async_eval`.
   - This matters most for the drafter.
5. On M5, target NAX for prefill (`matmul2d` tiles, dequantize into registers or threadgroup memory). Measure
   against MLX's `qmm_t_nax`. An INT8 path changes numerics (EXPERIMENTAL).
6. Use a page-aligned weight file with mmap + no-copy buffers, and report peak RAM as `phys_footprint`.
7. Benchmark hygiene: never compare one-op-per-eval timings, which carry a 150–170 µs floor (≈ 0.28 ms on a
   machine that is likely an M5 Max). Rotate weights, report medians, and state the display and background load.

**Open questions (need an M5 Max).**
- `qmv_wide` / `qmm_splitk` / NAX timings for Qwen3-32B shapes at M = 2..32.
- The per-kernel GPU gap; this bounds the real megakernel upside, which is ≤ 3% here.
- Raw-Metal bandwidth vs MLX's 182 GB/s on M2 Pro (raw-Metal agent).
- Energy per token (not measured).
- Whether the M5 Max reports `applegpu_g17s` (third-party evidence says yes). If so, it shares MLX's
  Max-class heuristics with our M2 Pro.

**Notes:** `$S/research/R3-mlx-core.md` (this file: code trace §1–5, measurements §7, conclusions §8).
Companions: `R3-kernels.md` (kernel catalog, Steel, SDPA, NAX), `R3-issues.md` (~40 GitHub items with
maintainer quotes). Scripts `r3_bench_*.py` and raw outputs `r3_out_*.txt` are in the same directory.
