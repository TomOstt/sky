# Part V — Competitor architecture: MLX, mlx-lm, tinygrad, llama.cpp, and the rest

This part describes how the engines Sky must beat actually run a language model on an Apple GPU, measured on our machine where possible and read from their source everywhere else. For each engine it answers four questions:
- how one decode token executes;
- where the engine is already at the hardware limit;
- where it measurably leaves performance on the table;
- what Sky copies from it, and what Sky avoids.

The part is written to be read by the maintainers of the engines it describes. Every number carries its source and label. Where our machine is not representative (the M2 Pro is a generation-14 GPU without neural accelerators), we say so next to the number.

Other parts build on this one:
- [Part IV](part04-hardware-limits.md) gives the hardware ceilings used here.
- [Part VI](part06-architecture.md) gives Sky's architecture.
- [Part XVI](part16-decode.md), [Part XVII](part17-prefill.md), [Part XV](part15-attention.md) and [Part XXI](part21-speculative.md) turn the openings found here into designs.
- [Part XXVIII](part28-risks.md) argues the other side.

**Evidence base.** Research appendices:
- [R3 (MLX core)](research/R3-mlx-core.md), [R3-kernels](research/R3-kernels.md), [R3-issues](research/R3-issues.md);
- [R4 (mlx-lm)](research/R4-mlx-lm.md);
- [R2 (tinygrad)](research/R2-tinygrad.md);
- [R5 (llama.cpp)](research/R5-llamacpp.md);
- [R6 (Apple silicon and Metal)](research/R6-apple-metal.md);
- [R7 (compilers and other systems)](research/R7-compilers.md);
- [R8 (algorithms)](research/R8-algorithms.md);
- [R9 (formats and methodology)](research/R9-formats-methodology.md).

Code citations refer to these pinned commits:

| Project | Commit (date) | Installed for measurements |
|---|---|---|
| MLX | `0840c42` (2026-10-01), 0.32.4-dev | mlx 0.32.3 wheel |
| mlx-lm | `53b9af3` (2026-10-01) | mlx-lm 0.32.0 (the measured files are byte-identical to the clone; R4 §0) |
| llama.cpp | `81e39ad` (2026-10-01) | built from that commit |
| tinygrad | `512e9b2` (2026-10-02) | run from that commit |

**Machine.** All local measurements ran on an Apple M2 Pro (16-core GPU, `applegpu_g14s`, 16 GB, macOS 26.6.1) under a machine-wide benchmark lock. A 4K external display was attached, so its scan-out traffic is inside every bandwidth number (R3 §0).

**Labels.** `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Third-party claims we could not check are marked as such.

---

## 1. Summary

1. **On dense batch-1 decode, MLX and llama.cpp are already near the bandwidth line, and within a few percent of each other per byte.**
   - On the M2 Pro, Qwen3-8B 4-bit decodes at 37.4–39.2 tok/s in mlx-lm and 34.6–36.0 tok/s in llama.cpp (Q4_0). Both stream 153–167 GB/s of weights, 84–91% of the 182.5 GB/s that MLX's own streaming kernels reach. About half of mlx-lm's higher token rate comes from its smaller file (4.26 GB against 4.43 GB per token). `[VERIFIED: R4 §7.A–§7.B, R5 §8.A–§8.B]`
   - The only Qwen3-32B M5 Max run that measured both engines shows the same thing. MLX is 7.7% faster, and llama.cpp's Q4_K_M file streams 7.4% more bytes per token (19.32 GB against 17.99 GB). Both stream ≈ 83% of the 614 GB/s spec. `[LIKELY: one community run, R5 §9.3; bytes from R8]`
2. **The fixed cost per token is the same in both engines, and it is GPU-side.**
   - mlx-lm issues 693 dispatches and 457 barriers per Qwen3-8B token. llama.cpp issues about 724 dispatches and 542 barriers. Both are about 19–20 kernels per layer. `[VERIFIED: R4 §7.C, R5 §8.D2]`
   - Removing everything except the quantized matmuls makes mlx-lm's 8B token 7.6% faster and its 0.6B token 31% faster. `[VERIFIED: R4 §7.E]`
   - llama.cpp's maintainers replayed the graph with indirect command buffers and measured no gain (#14570). R6 measured why: replay removes CPU encoding, not the ≈ 1.2 µs GPU cost of each dependent kernel boundary. `[VERIFIED: R5 §9.5, R6 §B3–B4]`
3. **A generic tensor compiler is far from both.** tinygrad's generated fused-dequant GEMV streams Qwen3-0.6B Q4_0 at about 21 GB/s (≈ 11% of the raw-Metal read). Dequantizing the weights to fp16 at load makes it faster, despite 3.2× the bytes. `[VERIFIED: R2 §8.3–§8.4]` This is the strongest argument in this report for hand-designed kernel skeletons with generated fusion around them (D4), rather than generic code generation.
4. **Verifying several tokens at once is the largest kernel-level gap, and it decides whether speculation pays.**
   - On the M2 Pro a k-token verify forward costs:

     | Tokens verified | mlx-lm | llama.cpp (Q4_0) |
     |---|---|---|
     | 4 | 3.33× | 2.08× |
     | 8 | 6.48× | 3.76× |

     `[VERIFIED: R4 §7.D, R5 §8.G]`
   - As a result, mlx-lm's speculative decoding is 0.37–0.74× of its plain decode here, and llama.cpp's is at best 1.12×.
   - On M5-class GPUs MLX's `qmv_wide` makes small M cheaper, but a contributor still measured +48% at 5 rows and +125% at 8 rows on a 70B forward. The ideal is roughly flat to 8 rows. `[VERIFIED: mlx#3791 via R3-issues E4; ideal LIKELY, R8 §2.2]`
5. **Exactness is a property neither leading engine offers.**
   - mlx-lm's speculative greedy output departs from plain greedy (mlx-lm#1470, M5 Max). Its maintainers documented the behaviour instead of changing it (#1592). We reproduced the divergence in 3 of 9 speculative configurations on the M2 Pro. `[VERIFIED: R4 §7.D, R8 §3.4]`
   - llama.cpp's server documentation states that logits are not bit-identical across batch sizes. `[VERIFIED: R9 §6.1]`
6. **Long-context decode attention leaves measurable bandwidth unused in MLX.**
   - For Qwen3-8B (GQA 4) on the M2 Pro, the extra KV stream from 128 to 4K context runs at ≈ 97 GB/s, about 53% of 182.5. For Qwen3-0.6B (GQA 2) it runs at ≈ 189 GB/s. `[VERIFIED: R4 §7.A]`
   - The cost grows with the GQA factor, which matches the mechanism read in the code: K/V are re-read per query head below 8K context. `[LIKELY: R3-kernels §c]`
   - MLX's duplication-free GQA kernel starts only at 8K context. `[VERIFIED: mlx#4077]`
7. **Prefill: MLX leads on long prompts on M5, but it trails llama.cpp on the M2 Pro and on short prompts.**
   - On the M2 Pro, llama.cpp prefills Qwen3-8B 1.35–1.42× faster than mlx-lm (pp512–pp2048), and Qwen3-0.6B 1.18–1.42× faster. `[VERIFIED: R5 §8.A and R4 §7.A; different files and tools, §9.2]`
   - On an M5 Pro, a vendor report shows mlx-lm at 0.44× of llama.cpp at pp128 on Qwen3-0.6B, and 1.55× ahead at pp2048. `[VERIFIED as published: BaseRT, arXiv 2607.19438, Tables 2–3]`
8. **MLX copies weights; llama.cpp maps them.**
   - MLX `pread`s every tensor into new Metal buffers, and its maintainers are "not considering using mmap" (mlx#4249). `[VERIFIED: R3 §4, R3-issues G3]`
   - mlx-community files cannot be mapped zero-copy: no tensor is even 4-byte aligned, and an unaligned `uint` load silently returns the wrong word. `[VERIFIED: R9 §1.2–§1.3]`
   - llama.cpp maps 32-byte-aligned GGUF files with no copy and keeps them wired with a 5 ms residency heartbeat. Without it, an idle pause made tokens 16× slower (#10119). `[VERIFIED: R5 §1.4–§1.5]`
9. **Others are attacking the same openings.**
   - BaseRT ties MLX on dense 27B decode (0.98×) and wins on short-prompt prefill (1.44× at pp128).
   - Monolith (mpk-apple) has the same runtime shape as Sky's D9 and targets only 1.10×.
   - Mojo reaches the M5 matrix units through undocumented AIR intrinsics and still runs at 0.85–0.96× MLX.
   - MTPLX, built on MLX, claims 65–88 tok/s on Qwen3.6/3.8-27B with MTP heads (unverified).

   `[VERIFIED for the published numbers: R7 §0.2, §8.1; R8 §2.7]`
10. **What Sky takes from each engine:**
    - from llama.cpp, the systems discipline: mapped weights, residency, graph reuse, byte-range hazard tracking, offline per-family tuning tables, in-kernel quantized KV;
    - from MLX, the decode GEMV arithmetic and asynchronous pipelining;
    - from tinygrad, the IR and runtime ideas: a hash-consed SSA graph, range-based fusion, a symbolic-integer layer, a single argument buffer with ICB replay, and a TLSF arena planner.

    What Sky avoids: per-token graph construction, copy-based loading, shape-routing cliffs, non-batch-invariant verification, and generic code generation for the hot kernels.

---

## 2. Reading the numbers in this part

### 2.1 Bandwidth denominators

| Quantity | M2 Pro (ours) | M5 Max, 40-core | Source |
|---|---|---|---|
| Apple's figure | 200 GB/s | 614 GB/s | Apple specs `[VERIFIED: R6 §A1]` |
| Theoretical DRAM | 204.8 GB/s (LPDDR5-6400 × 256-bit) | — | `[VERIFIED: R3 §7]` |
| Best raw-Metal read | **196 GB/s** (scalar loads) | not measured by us; a contributor reports ≈ 530 GB/s (method not shown); Part IV plans with 534 | `[VERIFIED: R6 §B2]`; M5 `[LIKELY: mlx#3837, R6]` |
| Simple 4-bit GEMV, raw Metal | 183–191 GB/s | — | `[VERIFIED: R6 §B2]` |
| MLX streaming kernels (reduce, copy) | **182.5 GB/s** | not measured | `[VERIFIED: R3 §7.A]` |

In this part:
- "% of 182.5" means the share of what MLX's own streaming kernels reach. R3 and R4 use this denominator.
- "% of 196" means the share of the best raw read. [Part IV](part04-hardware-limits.md) §2 uses both.
- The difference matters: MLX at "94–97%" of its own ceiling is 87–90% of the raw read.

### 2.2 Bytes per decode token, per file

Decode streams every weight once, so files of different sizes are not comparable in tokens per second. Each is listed as weights read per token, excluding the one-row embedding gather.

| Model | File | Bits per weight | Weights per token | Source |
|---|---|---|---|---|
| Qwen3-0.6B | `mlx-community/Qwen3-0.6B-4bit` (affine g64, bf16 scales and biases; tied head) | 4.50 | 335.4 MB | `[VERIFIED: R4 §7.B]` |
| Qwen3-0.6B | unsloth Q4_0 GGUF (Q6_K tied head, 3 Q4_1 tensors) | mixed | 376.2 MB | `[VERIFIED: R5 §8.B]` |
| Qwen3-8B | `mlx-community/Qwen3-8B-4bit` | 4.50 | 4,257.7 MB | `[VERIFIED: R4 §7.B]` |
| Qwen3-8B | bartowski Q4_0 GGUF (Q6_K head, 4 Q4_1 `ffn_down`) | 4.67 | 4,431.3 MB | `[VERIFIED: R5 §8.B, R9 §2.3]` |
| Qwen3-8B | Qwen Q4_K_M GGUF | 4.90 | 4,671.8 MB | `[VERIFIED: R5 §8.B]` |
| Qwen3-32B | MLX 4-bit g64 | 4.50 | 17.99 GB | `[VERIFIED: R8 §1.3]` |
| Qwen3-32B | GGUF Q4_0 / Q4_K_M | mixed | 18.26 / 19.32 GB | `[VERIFIED: R8 §1.3]` |

KV reads add, per context position: 114.7 KB (0.6B), 147.5 KB (8B) and 256 KiB (32B) in fp16 or bf16.

### 2.3 The tools measure different things

| Tool | What its decode number is | Source |
|---|---|---|
| `llama-bench tg128` | Decode from an **empty** KV cache, random tokens, **no sampling**; the mean of per-repetition rates | `[VERIFIED: R9 §6.1]` |
| mlx-lm `generation_tps` | Includes sampling, `logsumexp`, detokenization and Python. Counts the first token in the numerator but not its time, so it is biased up by (n+1)/n | `[VERIFIED: R4 §2.6]` |
| R4's wall decode | (N−1)/(t_last − t_first), from external per-token timestamps around `stream_generate`, greedy, after a 128-token prompt. This is the definition Sky's protocol uses (SPINE D13) | `[VERIFIED: R4 §7.A]` |
| tinygrad `--benchmark` | Its own per-token wall time after JIT capture | `[VERIFIED: R2 §11.2]` |

Consequences for this part:
- Where an engine's own tool is the only source, the method is named next to the number.
- The external harness of [Part XXIII](part23-benchmarks.md) replaces all of these at gate G0.

---

## 3. MLX core

MLX is Apple's array framework: a lazy C++ core with Python bindings, a Metal backend of hand-written kernels, and a small JIT for fused elementwise code. Everything mlx-lm does on the GPU goes through it.

### 3.1 Architecture

| Layer | Mechanism | Source |
|---|---|---|
| Array | `array` is a handle to an `ArrayDesc` (shape, strides, dtype, `Primitive`, inputs, `Event`, data buffer). Building an op allocates a descriptor and a primitive on the heap; nothing runs until evaluation | `mlx/array.h:491-546` `[VERIFIED: R3 §1.1]` |
| Evaluation | `eval_impl` counts fan-out by DFS, then builds a BFS tape at most 20 wide (`MLX_BFS_MAX_WIDTH`) to bound live intermediates. It walks the tape, inserting fences for cross-stream inputs, and calls `gpu::eval` inline on the calling thread | `mlx/transforms.cpp:80-365` `[VERIFIED]` |
| Back-pressure | If more than `MAX_ACTIVE_TASKS = 10` command buffers are in flight, or memory exceeds the limit, eval commits and blocks in `wait_for_one` | `mlx/transforms.cpp:25, 278-292` `[VERIFIED]` |
| Threads | All GPU encoding happens on the calling (Python) thread. Scheduler threads run only CPU-stream work, such as weight loads | `mlx/scheduler.cpp:42-91` `[VERIFIED: R3 §1.2]` |
| Compile | `mx.compile` fuses only elementwise, broadcast and select ops, at most 11 deep and 24 inputs, into MSL generated at run time. Reductions, matmuls, norms, RoPE, attention, gathers and slices are never fused | `mlx/compile.cpp:25-81`; `mlx/backend/metal/compiled.cpp:16-330` `[VERIFIED: R3 §1.4]` |
| Queues | One `MTLCommandQueue` per stream. Command buffers come from `commandBufferWithUnretainedReferences` | `mlx/backend/metal/device.cpp:309-324, 559` `[VERIFIED]` |
| Encoders and barriers | One compute encoder per command buffer, created with `MTLDispatchTypeConcurrent`. MLX tracks each dispatch's buffers and emits `memoryBarrier(scope: buffers)` on read-after-write or write-after-read within the current epoch. The granularity is a whole `MTLBuffer` | `mlx/backend/metal/device.cpp:345-406, 576-584` `[VERIFIED]` |
| Commit policy | Commit when ops exceed `max_ops_per_buffer` or input size exceeds `max_mb_per_buffer`: 50/50 on `s`-class GPUs (our M2 Pro; LIKELY the M5 Max too). The "MB" counter sums element counts, not bytes (mlx#3192) | `mlx/backend/metal/device.cpp:511-514, 602-626`; `mlx/array.h:355-358` `[VERIFIED]` |
| Buffers | Every buffer is `StorageModeShared` with `HazardTrackingModeUntracked`: unified memory, with MLX doing its own hazard tracking | `mlx/backend/metal/allocator.cpp:15-16` `[VERIFIED]` |

### 3.2 One decode step through MLX, exactly

The trace below is for Qwen3-8B 4-bit, with mlx-lm's loop driving MLX (the loop itself is in §4.2). Counts are exact Metal-call counts from an interposer on the compute encoder (R4 §7.C).

```text
Python thread (CPU)                                     GPU (one queue)
───────────────────                                     ───────────────
build the graph for step t+1: 1,061 nodes, of which     executing step t
  693 launch kernels; about 0.65 ms of Python
mx.async_eval(t+1) → eval_impl:
  DFS fan-out count; BFS tape
  for each of the 693 kernel primitives:
    autorelease pool; encoder lookup by stream
    output malloc from the buffer cache
    kernel name built as a string, e.g.
      affine_qmv_fast_float16_t_gs_64_b_4_batch_0
    pipeline lookup in a hash map (shared mutex)
    setBuffer / setBytes per argument
    hazard sets → memoryBarrier when needed      ──►    457 barriers per token
    dispatchThreadgroups / dispatchThreads       ──►    693 dispatches per token
    completion handler holding the inputs alive
  commit after 50 ops or ~52 M input elements    ──►    23 command buffers per token
  block if more than 10 command buffers in flight
y_t.item(): MTLSharedEvent wait on step t   ◄──────     step t signals
```

`[VERIFIED: R3 §1.2–§2.2 for the mechanism; R4 §7.C and §7.E for the counts and the 0.65 ms]`

Per layer the 19 dispatches are:
- 7 quantized GEMVs (q, k, v, o, gate, up, down);
- 4 RMSNorms (input, q-norm, k-norm, post-attention);
- 2 RoPEs;
- 2 KV-cache writes;
- 1 attention;
- 2 residual adds;
- 1 compiled SwiGLU.

The 9 dispatches outside the layers are 3 gathers and 1 dequantize for the embedding, the final norm, the LM head, and logsumexp, subtract and argmax for sampling. A 64-layer Qwen3-32B token is about 1,225 dispatches. `[VERIFIED: R4 §3.1, R3 §7.F]`

Synchronization points:

| Mechanism | Cost on the M2 Pro | Source |
|---|---|---|
| Blocking `mx.eval` or `.item()` round trip | 149–158 µs | `[VERIFIED: R3 §7.E]` |
| The same with `async_eval` pipelining | 28.9 µs | `[VERIFIED: R3 §7.E]` |
| Marginal cost per dependent tiny op | 2.9–3.2 µs: 0.8 µs of graph construction, 2.1–2.4 µs of eval, encode and GPU | `[VERIFIED: R3 §7.E]` |
| Cross-stream fence | `MTLSharedEvent`, or a spin kernel with `MLX_METAL_FAST_SYNCH=1` (≈ 180 µs vs ≈ 4 µs on an M3 Ultra; not the default) | `[VERIFIED: mlx#4438 via R3-issues D6]` |

### 3.3 Decode kernels

**The M = 1 kernel, `affine_qmv_fast`** (`mlx/backend/metal/kernels/quantized.h:756-821`; R3 §3.3). It is used for every Qwen3-8B and 32B projection.

| Aspect | Design |
|---|---|
| Threadgroup | 64 threads = 2 simdgroups. Each simdgroup owns 4 output rows, so a threadgroup owns 8 rows |
| Split | The 32 lanes of a simdgroup split K. Each lane handles 16 nibbles per step: 8 bytes of weights per row, read as four `uint16` |
| Dequantization | Activations are pre-divided by 1, 16, 256 and 4096, so a nibble is masked but never shifted. Scale and bias are applied once per 16 values as `s·Σ(x·q) + b·Σx`, with Σx shared by the 4 rows |
| Arithmetic | fp32 accumulation; one `simd_sum` per row at the end; no threadgroup memory, no atomics, no split-K |

**Routing for M > 1** (`mlx/backend/metal/quantized.cpp:89-147, 1825-1962`; R3 §3.2). The batch limit `vector_limit` depends on the GPU generation, its size class and the matrix shape:

| GPU | K, N ≤ 2048 | ≤ 4096 | Larger (all Qwen3-32B projections) |
|---|---|---|---|
| Generation ≥ 17 (M5), not Ultra | 33 | 25 | 13 |
| Generations 15–16 (M3, M4), not Ultra | 13 | 15 | 13 |
| Generations 13–14 (M1, M2), other than Ultra (our M2 Pro) | 14 | 10 | **6** |

- For M < `vector_limit`, generations ≥ 15 route affine M = 2…5 to `qmv_wide`. It streams each weight group once for up to 5 vectors, so M = 6–10 reads the weights twice. `qmv_wide` was adapted from llama.cpp's `kernel_mul_mv_ext` (mlx#3764).
- On generations 13–14 each extra row is a separate threadgroup row that re-reads and re-dequantizes the weights. MLX's comment says `qmv_wide` only wins on generation ≥ 15 for affine. Our emulation of the M3 routing on M2 hardware agreed: it was worse (R3 §7.C).
- For M ≥ `vector_limit`, MLX uses split-K `qmm` with 32-row tiles, so every M from 6 to 31 on the M2 Pro pays for 32 rows. The partial sums are stored in the activation dtype.

**Measured on the M2 Pro** (R3 §7.B; batched calls in one eval, ≥ 320 MB working set):
- Large projections run at 91–97% of 182.5 GB/s.
- k/v projections (N = 1024) reach only 54–63%, because 128 threadgroups of 64 threads do not fill 16 cores.
- A hand-written 60-line kernel matched MLX at M = 1. MLX's M = 1 GEMV is not leaving bandwidth on the table relative to a simple kernel (R3 §7.G).
- R6's simple GEMV, timed with a different harness, reached 183.8–191.4 GB/s on shapes where MLX measured 165–177 GB/s. Whether that 4–13% belongs to the kernel or to the harness is open `[EXPERIMENTAL: time both kernels in one harness; Part XVI]`.

### 3.4 Prefill kernels

| Path | Tile and structure | Measured | Source |
|---|---|---|---|
| `affine_qmm_t` (pre-M5) | 32×32 output tile, BK = 32, 2×2 simdgroups. Weights are dequantized into threadgroup memory, then a Steel `BlockMMA` on 8×8 `simdgroup_matrix`; single-buffered, no async copy | 4.38–4.44 TFLOPS = 77% of the 5.73 TFLOPS peak (Qwen3-32B MLP shapes, M = 512/2048) | `mlx/backend/metal/kernels/quantized.h:1191-1317`; `[VERIFIED: R3 §3.4, §7.D]` |
| Steel GEMM, fp16 (`s`-class) | 64×64×16, 2×2 simdgroups. Fragments are converted to fp32 `simdgroup_matrix`; single-buffered | 4.84–5.03 TFLOPS = 85–88% | `mlx/backend/metal/matmul.cpp:90-171` `[VERIFIED: R3-kernels §b, R3 §7.D]` |
| `qmm_t_nax` (M5) | 64×64×64 tiles (BM = 32 when M ≤ 32). Weights are dequantized into threadgroup memory as fp16/bf16; activations go straight into `matmul2d` fragments | Not measured by us | `mlx/backend/metal/kernels/quantized_nax.h:925-1060` `[VERIFIED: R3-kernels §d]` |
| Dense NAX GEMM (M5) | `s`/`c`: BM = 64, BN = 128, BK = 256, 2×4 simdgroups, swizzle 2. Operands load directly from device memory into registers | M5 Max fp16: 58.2 TFLOPS (4096³), 52.7–59.6 across shapes | `mlx/backend/metal/matmul.cpp:206-293` `[VERIFIED: jagrit06, mlx#3211]` |

Two caveats on M5 routing:
- For 13 ≤ M ≤ 32, only gate/up and the LM head reach `qmm_t_nax` on Qwen3-32B shapes. q, k, v, o and down stay on non-NAX split-K. `[VERIFIED: arithmetic on the routing code, R3-kernels §d]`
- A measured cliff (M = 33 ran 21% faster than M = 32 at N = 6,656) was closed by a maintainer as out of scope. `[VERIFIED: mlx#4198]`

### 3.5 Quantization formats

| Mode | Bits / group | Layout | Source |
|---|---|---|---|
| `affine` | 2, 3, 4, 5, 6, 8 bits; groups 32 / 64 / 128 | `w = scale·q + bias`. 8 nibbles per little-endian `uint32`, least significant first. Separate scale and bias tensors in the weight dtype (bf16 in mlx-community files). **50.25% of the scales are negative**: the quantizer pins the larger-magnitude edge of each group | `mlx/ops.cpp:5122-5233` `[VERIFIED: R9 §1.4]` |
| `mxfp4` | 4 bits, group 32 | E2M1 codes plus an E8M0 scale | `mlx/backend/metal/kernels/fp_quantized.metal:250-262` `[VERIFIED: R3 §3.1]` |
| `nvfp4` | 4 bits, group 16 | E2M1 plus an E4M3 scale and an optional fp32 global scale | same |
| `mxfp8` | 8 bits, group 32 | E4M3 plus E8M0 | same |
| `QQMatmul` | — | Quantized activations (nvfp4, mxfp8) | `[VERIFIED: R3 §3.1]` |

At group 64 with 16-bit scale and bias, affine 4-bit is 4.50 bits per weight. No exact conversion exists between it and any 4.5-bpw GGUF type, which is why Sky loads both formats natively (D5; R9 §2.5).

### 3.6 Model loading

`mx.load` works in two stages (`mlx/io/safetensors.cpp:111-214`; `mlx/backend/common/load.cpp:30-55`; R3 §4):
1. It reads the safetensors header synchronously and returns one lazy array per tensor. Each has a `Load` primitive on the CPU stream.
2. On evaluation, each tensor gets a **new Metal buffer** (`allocator::malloc`). The bytes are `pread` into it in 32 MiB batches on a pool of `clamp(cores/2, 4, 16)` threads.

What follows from this:
- There is no `mmap` and no `newBufferWithBytesNoCopy` on this path. A no-copy path exists only for host pointers.
- Load time is the file read plus allocation.
- The page cache and the Metal buffers coexist during loading.
- `[VERIFIED]`

The maintainers have declined mmap:
- "At the moment we are not considering using mmap" (zcbenz, mlx#4249).
- A contributor's M5 Max data in the same thread showed mapped views running "at resident speed". It listed the caveats: Metal wires the whole no-copy buffer on first use, the wired ceiling counts mapped buffers, and safetensors offsets force a one-time repack. `[VERIFIED: R3-issues G3]`

### 3.7 Memory

| Topic | Mechanism | Source |
|---|---|---|
| Allocation | Allocations under 256 B come from a 1 MiB heap. Larger ones are page-rounded (16 KiB) `newBuffer`s, kept in a size-keyed cache that reuses a buffer if it lies in [size, min(2·size, size + 2 pages)) | `mlx/backend/metal/allocator.cpp`; `mlx/backend/common/buffer_cache.h:30-99` `[VERIFIED: R3 §2.3]` |
| Limits | Memory limit = min(1.5 × `recommendedMaxWorkingSetSize`, 0.95 × RAM). Above the GC limit, `malloc` evicts cached buffers. The Metal resource-count limit (499,000 buffers) is enforced, so one buffer per array can hit it | `[VERIFIED: R3 §2.3, R3-issues G6]` |
| Residency | Allocations go into `MTLResidencySet`s up to the wired limit. The default limit is 0; mlx-lm raises it. Each set is capped at 5% of the working set (`MLX_RESIDENCY_SET_MAX_PCT`) because macOS 27 decides residency per set: one monolithic set caused re-wiring stalls (0.46 → 13.19 tok/s after the fix) | `[VERIFIED: mlx#4211, R3 §2.3]` |
| Reporting | `get_peak_memory` is the high-water mark of *active* MLX buffers only. It excludes the buffer cache, the pipeline library (the 190 MB `mlx.metallib` is mapped by Metal), the Python heap and the page cache. A churn test showed 1.00 GB "peak" against a 60.19 GB OS footprint | `[VERIFIED: R3 §5, mlx#3896]` |
| Reliability | Kernel panics ("completeMemory() prepare count underflow") across M1–M5 when a wired limit is set. One user's A/B test showed no panic without it. Open since 2026-03 | `[VERIFIED as reported: mlx#3186]` |

### 3.8 Attention kernels and KV

| Case | Kernel and condition | Consequence for Qwen3 | Source |
|---|---|---|---|
| Decode, q length ≤ 8 | `sdpa_vector`, fused only if `qL · gqa_factor ≤ 32` | Qwen3-32B (GQA 8) gets the fused kernel only for qL ≤ 4. qL 5–8 falls back to unfused matmul → softmax → matmul | `mlx/backend/metal/scaled_dot_product_attention.cpp:1001-1006, 1496` `[VERIFIED: R3-kernels §c]` |
| 1-pass vs 2-pass | 2-pass from kL ≥ 1024 on `s`/`d` GPUs; on others, GQA models switch at 4,096 | M2 Pro and M5 Max (both `s`) split from 1K | `…:1561-1570` `[VERIFIED]` |
| K/V reuse across a GQA group | The 1-pass kernel fetches a KV head once per query head in its group; reuse relies on caches. The duplication-free `_gqa` 2-pass kernel runs only for qL = 1, no mask, kL ≥ 8,192, and G ∈ {8, 12, 16} with D = 128 (or G = 8 with D = 64) | Qwen3-32B gets it only at ≥ 8K. Qwen3-8B (G = 4) never does | `…:755-763`; `kernels/sdpa_vector.h:65-69, 319-325` `[VERIFIED: mlx#4077, #4380]` |
| Prefill (qL > 8) | `steel_attention`: BQ = 32, causal block skip, **no split over the KV length**. On M5, `attention_nax`: BQ = 64, BK = 32, operands from device memory into NAX registers | — | `kernels/steel/attn/kernels/steel_attention.h`, `steel_attention_nax.h` `[VERIFIED: R3-kernels §c]` |
| Verify widths 9–48 on M5 | The BQ = 64 tile gives a plateau: 0.813 ms at L = 8 vs 1.906 ms at L = 12, flat to 48 (M5 Max, 16K context). Closed as won't-fix | Tree verification above 8 nodes pays a step | `[VERIFIED: mlx#3826]` |
| Quantized KV | No fused kernel in MLX core. mlx-lm composes quantized matmul → softmax → quantized matmul (§4.6). "Quantized SDPA" PR open since 2026-01 | — | `[VERIFIED: R3-kernels §c, mlx#3026]` |

### 3.9 Sampling and speculation

MLX core provides the primitives: `argmax`, `logsumexp`, `random.categorical` (threefry). It has no sampler and no speculative decoding of its own; both live in mlx-lm (§4.7–§4.8). `[VERIFIED: R3-kernels §a]`

### 3.10 M5 neural accelerators ("NAX")

**Gating.** NAX kernels are compiled only with Metal ≥ 4.0, the macOS 26.2 SDK and a 26.2 deployment target. At run time they need macOS ≥ 26.2 and GPU generation ≥ 17 (≥ 18 for phones). There is no Python API to query this. `[VERIFIED: mlx/backend/metal/device.cpp:952-971; R3-kernels §d]`

**Programming model.** `mpp::tensor_ops::matmul2d` with a 16×32×16 descriptor and `relaxed_precision = true`, fed from MLX's own 16×16 fragments; fp32 accumulators. f16 and bf16 inputs; f32 inputs only through TF32 (`MLX_ENABLE_TF32`, on by default). **No integer `matmul2d` anywhere in MLX.** `[VERIFIED: kernels/steel/gemm/nax.h:12, 386-528]`

| Op | Uses NAX? | Note |
|---|---|---|
| Dense GEMM, conv via GEMM, gather GEMM (MoE) | Yes | — |
| Quantized `qmm` (prefill, large M) | Yes | Weights are dequantized to f16/bf16 first |
| Prefill attention | Yes | D ∈ {64, 96, 128, 256, 512} |
| Gated-DeltaNet update (T > 8) | Yes | Used by Qwen3.5/3.6-class hybrid models |
| Decode qmv, `qmv_wide`, GEMV | **No** | All M < `vector_limit` (13 for Qwen3-32B shapes) |
| Decode attention, norms, RoPE, elementwise | **No** | — |

`[VERIFIED: R3-kernels §d]`

Numerics and toolchain:
- With TF32 on, M5 Max f32 matmuls showed a maximum relative error of 1.9e-3, against 6.0e-8 with TF32 off. A maintainer called this "expected behavior" (mlx#3534, #3860). fp16/bf16 LLM paths are unaffected.
- The NAX path needed several correctness fixes after launch: Xcode 26.5 miscompiled the NAX GEMM kernel (#3586); NAX split-K produced bf16 garbage at large sizes (#3797); gather_qmm NAX had corruption cases (#3887); macOS 27 SDK builds broke (#4594).

`[VERIFIED: R3-kernels §d, R3-issues B8–B9]`

### 3.11 Measured on our M2 Pro

| Measurement | Result | Source |
|---|---|---|
| Streaming bandwidth through MLX (2 GiB reduce, 1 GiB copy) | 182.5 GB/s median (89% of 204.8) | `[VERIFIED: R3 §7.A]` |
| Qwen3-32B-shaped GEMVs, M = 1, batched | 162–177 GB/s (91–97%); k/v 114 GB/s (63%) | `[VERIFIED: R3 §7.B]` |
| One GEMV per `mx.eval` | 160–170 µs floor: sync, not kernel | `[VERIFIED: R3 §7.B]` |
| Whole decode step, mlx-lm-style pipelining, ctx 128 | 32B-shaped 3-layer stack 176 GB/s (97%); 8B-shaped 8-layer stack 171 GB/s (94%) | `[VERIFIED: R3 §7.F]` |
| Framework floor, 64 layers, no math | 4.2–4.4 ms per token; ≈ 3.5 µs per kernel | `[VERIFIED: R3 §7.F]` |
| One command buffer per token instead of the default policy | 0% change at real sizes; −2 to −5% at tiny sizes | `[VERIFIED: R3 §7.F]` |
| Verify GEMV, 32B shapes, M = 2 / 3 / 5 / 8 | ×1.0–1.3 / ×1.4–2.0 / ×2.2–3.2 / ×3.4–4.7; M = 6…31 flat at ×4.4–4.8 | `[VERIFIED: R3 §7.C]` |
| Simple custom multi-row kernel, M = 3 | ×1.30–1.37 (spills registers from M = 4) | `[VERIFIED timing, EXPERIMENTAL kernel: R3 §7.G]` |
| Prefill GEMM, 32B MLP shapes | 4-bit 4.38–4.44 TFLOPS (77%); fp16 4.84–5.03 (85–88%) | `[VERIFIED: R3 §7.D]` |

### 3.12 Published M5-class numbers

| Result | Machine | Source | Label |
|---|---|---|---|
| fp16 GEMM 4096³ at 58.2 TFLOPS; 52.7–59.6 across shapes | M5 Max | jagrit06, mlx#3211 | VERIFIED (maintainer) |
| `mx.matmul` fp16 46.7, bf16 36.7, q4 35.0 TFLOPS | M5 Max | mlx#4411 via R6 | EXPERIMENTAL (one measurement) |
| `qmv_wide` over `qmv`, int4: 1.4× (M = 2), 1.7× (M = 4), 1.6× (M = 8) | M5 Max | mlx#3764 | VERIFIED (PR data) |
| 70B 4-bit forward: M = 1 75.9 ms, M = 3 84.8, M = 5 112.7, M = 8 170.8, M = 16 296.0 | M5 Max | mlx#3791 | contributor data |
| GQA decode kernel: 1.08–1.31× at ≥ 8K context | M5 Max | mlx#4077 | VERIFIED (PR data) |
| Decode SDPA, 32K context, 32 q heads: 8 KV heads at 56.2% of peak; 32 KV heads at 88.3% | M5 Max | mlx#3837 | contributor data |
| Measured peak bandwidth 530 GB/s (86% of 614) | M5 Max | mlx#3837 | contributor; method not shown |
| TTFT 3.33–4.06× faster than M4 at 4K prompts; decode 1.19–1.27× (bandwidth-bound) | base M5 | Apple ML Research, 2025-11-19 | VERIFIED |

### 3.13 Strengths

- **Decode GEMV at the bandwidth line.** The `qmv_fast` arithmetic (pre-scaled activations, masked nibbles, `s·Σxq + b·Σx`) costs about 2–4 ALU operations per weight against a budget of about 7.5 on an M5 Max. It runs at 91–97% of the streaming ceiling (R3 §3.3, §7.B; R8 §4.3).
- **CPU work is hidden.** Python graph construction and encoding run while the GPU executes the previous token. That is invisible for models of 8B and up (R3 §7.F; R4 §7.E).
- **First to the M5 matrix units.** MLX shipped NAX support in v0.30.0 (2025-11-19). Large fp16 GEMMs on the M5 Max run near the estimated ≈ 60–65 TFLOPS ceiling (R3-issues B3–B4; R6).
- **Breadth.**
  - Many quantization modes.
  - MoE gather kernels.
  - A fused Gated-DeltaNet kernel for the hybrid Qwen3.5/3.6/3.8 models that SPINE D10 names as the second headline.
  - CPU, CUDA and Metal backends.
- **Responsive to measurements.** Gaps shown with data were closed within weeks: `qmv_wide` (#3764), small-M verify (#3791), GQA decode attention (#4077), residency sets on macOS 27 (#4211).

### 3.14 Weaknesses, with evidence

| Weakness | Evidence | Size | Label |
|---|---|---|---|
| Small-M verification | Covered above: ×1.9–2.0 at M = 3 on the M2 Pro; +48% at M = 5 and +125% at M = 8 on the M5 Max; M = 6…31 pay a 32-row tile on the M2 Pro | The largest kernel gap; decides speculation | VERIFIED (R3 §7.C, mlx#3791) |
| GQA decode attention below 8K | K/V re-read per query head; Qwen3-8B's KV stream at 97 GB/s; 56% of peak at 8 KV heads on the M5 Max | ~2× on the attention share | VERIFIED time, LIKELY mechanism (R4 §7.A, mlx#3837) |
| Verify attention for qL 5–8 (GQA 8) and 9–48 | Unfused fallback; BQ = 64 plateau | Unmeasured on 32B | VERIFIED routing (mlx#3826) |
| Per-kernel fixed cost | ≈ 3.5 µs per kernel, ≈ 4.3 ms per 64-layer token with no math; hidden for 32B, dominant for drafts | 31% of a 0.6B token | VERIFIED (R3 §7.F, R4 §7.E) |
| No fusion across norms, RoPE, KV writes or residuals | `mx.compile` is elementwise-only. The maintainers' policy requires proof "in a popular inference engine" before adding a fused op | 7.6% of an 8B token, by ablation | VERIFIED (R3 §1.4, mlx#4259, R4 §7.E) |
| Copy-based loading | `pread` into new buffers; mmap declined | Load time and cold TTFT | VERIFIED (R3 §4, mlx#4249) |
| Hard-coded routing by device letter | `vector_limit`, 2-pass thresholds and block counts are keyed by architecture letter; shape cliffs closed as out of scope | Local cliffs, 20%+ | VERIFIED (R3-kernels §0, mlx#4198) |
| Not batch-invariant | The verify forward and the sequential forward differ in reduction order (#1470) | Breaks exact speculation | VERIFIED (R8 §3.4) |
| Peak-memory reporting | `get_peak_memory` excludes the cache, driver memory and Python | Misleading RAM claims | VERIFIED (mlx#3896) |
| Quantized prefill on non-NAX GPUs | `qmm_t` at 77% vs fp16 GEMM at 85–88% | 11–13% at kernel level | VERIFIED (R3 §7.D) |

### 3.15 Maintainers' stated positions

Members of ml-explore/mlx: awni (Awni Hannun), angeloskath (Angelos Katharopoulos), jagrit06 (Jagrit Digani), zcbenz (Cheng Zhao).

| Topic | Statement | Link |
|---|---|---|
| Megakernels | "The really hard part with this is to make the mega kernel work with various transformer architectures ... not practical for generic machine learning frameworks or even generic inference engines. So I'm closing this as won't fix, not because it is a bad idea, but because it is engineering not practical at the moment." (zcbenz) | mlx#3313 |
| Whole-step compile | "This is not going to help because you can not fuse them together so even with compilation you would end up launching exact same graph." (zcbenz) | mlx#4296 |
| New fused fast ops | "Currently for us to add a new builtin fast op, you have to deploy a custom kernel in a popular inference engine to prove it is definitely helping first." (zcbenz) | mlx#4259, #4297 |
| Command-buffer reuse | "Sounds a plausible optimization but would need some real world benchmarks." (zcbenz) | mlx#3789 |
| mmap | "At the moment we are not considering using mmap." (zcbenz) | mlx#4249 |
| Fast CPU↔GPU sync by default | "not guaranteed to work and not recommended by the Metal team ... So we can not make it the default." (zcbenz) | mlx#4438 |
| Shape-specific tuning | "we are unable to optimize for all possible cases, currently we are only focusing on real cases happened during inference" (zcbenz) | mlx#4198 |
| Verify attention at 9–48 queries | "We inquired experts who serve models in large scale with MTP and conclusion is that the mentioned case is likely not going to happen in production" (zcbenz) | mlx#3826 |
| Small-M direction | "For small-M we rely on qmv_wide for optimization but this PR tunes for affine_qmm which I think is the wrong direction." (zcbenz) | mlx#3863 |
| Allocator cache policy | "the benefits of tuning cache policy is not high, and I'm closing this as won't fix" (zcbenz) | mlx#3886 |
| Metal 4 adoption | "We will be evaluating additional Metal 4 features on a case-by-case basis and include what makes sense." (awni) | mlx#2693 |
| Prefill limits (pre-M5) | "it is mostly a hardware limitation ... prompt processing is a compute bound computation." (awni) | mlx-lm#92 |
| Quality vs llama.cpp | "It looks like MLX 4-bit is about the same quality as Q4_K_M which is what we expected." (awni) | mlx-examples#1029 |

`[VERIFIED: quotes as collected in R3-issues §C, §D, §E, §F, §H]`

These are coherent positions for a general-purpose framework. They are not oversights. They are also exactly the openings a model-specialized compiler can take (R3-issues §H).

### 3.16 What Sky copies and what Sky avoids

| Copy (with our own implementation; D15) | Why |
|---|---|
| The `qmv_fast` dequantization arithmetic and the 2-simdgroup × 4-row shape as the starting point for the GEMV skeleton | At the bandwidth line, with ALU headroom (§3.3) |
| Concurrent-dispatch encoders, untracked shared buffers, barriers only on real hazards | Necessary for overlap of independent kernels |
| Queuing step t+1 before reading step t; token feedback through data dependence on the GPU | Hides the CPU (§4.2) |
| Residency sets capped per set, with a wired limit | macOS 27 per-set residency (mlx#4211) |
| NAX through documented Metal 4 `matmul2d` for prefill; decode stays on SIMD kernels | Apple's documented route; the M = 1 `matmul2d` is slower than a GEMV (R6) |
| MLX's affine format as a first-class input, with `infer_quant_config`'s rules implemented exactly and negative scales accepted | Same-file comparisons (D5; R9 §1.4) |

| Avoid | Why |
|---|---|
| Per-token graph construction and per-primitive bookkeeping (string kernel names, hash-set hazard tracking, a completion handler per primitive) | ≈ 3.5 µs per kernel; caps drafts (R3 §7.F) |
| Copying weights at load | Load time, cold TTFT, transient double residency |
| Routing thresholds keyed by device letter, with cliffs | Replace with measured, tuned tables (D11) |
| A verify path whose reduction order differs from decode | Breaks exact speculation (#1470) |
| Elements-as-MB commit counters | Unclear policy; Sky plans submission statically ([Part XX](part20-persistent-execution.md)) |
| `get_peak_memory`-style RAM claims | R9 §6.2 metric only |

---

## 4. mlx-lm

mlx-lm is the LLM runtime on top of MLX. It provides:
- a loader;
- one Python file per architecture (`mlx_lm/models/qwen3.py`);
- the generation loops `generate_step`, `stream_generate`, `speculative_generate_step` and `BatchGenerator`;
- samplers, KV caches and prompt caches;
- an OpenAI-compatible server;
- quantization tools (`convert`, dynamic quantization, DWQ, AWQ, GPTQ).

It is the headline baseline (SPINE §1). Everything below is at `53b9af3`. R4 checked that the measured install is byte-identical to that clone for every file that matters (R4 §0).

### 4.1 Architecture

| Component | What it does | Source |
|---|---|---|
| Qwen3 model | `TransformerBlock`, `Attention`, `MLP` as `mlx.nn` modules. q, k and v are three separate quantized linears; gate and up are two. Only the SwiGLU helper is `mx.compile`d | `mlx_lm/models/qwen3.py:55-137`; `mlx_lm/models/activations.py:9-11` `[VERIFIED: R4 §3]` |
| Decode loop | `generate_step`, pipelined with `async_eval` (§4.2) | `mlx_lm/generate.py:304-473` |
| Streaming wrapper | `stream_generate`: per-token EOS check, detokenizer, `GenerationResponse` | `mlx_lm/generate.py:658-760` |
| Speculation | `speculative_generate_step`, draft-model only (§4.8) | `mlx_lm/generate.py:476-655` |
| Batching and server | `BatchGenerator` (continuous batching, `completion_batch_size = 32`); the server batches only without a draft model, seed or KV quantization | `mlx_lm/generate.py:1512-1884`; `mlx_lm/server.py:366-369` `[VERIFIED: R4 §5.2]` |
| Wired memory | `wired_limit()` raises MLX's wired limit to `max_recommended_working_set_size` for every generation call, then synchronizes and restores it | `mlx_lm/generate.py:228-263`; `mlx_lm/utils.py:73-82` |

### 4.2 One decode token through mlx-lm, exactly

```text
generate_step  (mlx_lm/generate.py:456-472)

  y, lp = _step(last prompt token)        build the graph for token 0
  mx.async_eval(y, lp)                    encode + commit; no wait
  loop:
    next_y, next_lp = _step(y)            build the graph for t+1. y is still a lazy array:
                                          the embedding gather reads the token on the GPU,
                                          so the token id never travels to the CPU and back
    mx.async_eval(next_y, next_lp)        encode + commit t+1 while the GPU runs t
    yield y.item(), lp                    the one host sync per token: wait for token t only
    every 256 tokens: mx.clear_cache()

_step(y)  (mlx_lm/generate.py:394-420)
  logits = model(y[None], cache)[:, -1, :]      36 layers × 19 dispatches (§3.2)
  logprobs = logits - logsumexp(logits)         2 dispatches, even for greedy
  token = sampler(logprobs)                     argmax, or a compiled filter chain + categorical
```

`[VERIFIED: source; R4 §2.2]`

Where the time goes (R4 §7.E; ablation on a bit-exact copy of `qwen3.py`, `max|Δ| = 0.0`):

| Per token, context 128 | Qwen3-8B 4-bit | Qwen3-0.6B 4-bit |
|---|---|---|
| Measured inter-token latency (`generate_step`) | 25.79 ms | 3.55 ms |
| Weights + KV at 182.5 GB/s | 23.46 ms (91%) | 1.94 ms (54.5%) |
| Quantized GEMVs below the line | +0.47 ms | +0.47 ms |
| All non-matmul ops | +1.97 ms (7.6%; 435 dispatches) | +1.11 ms (31%; 339 dispatches) |
| CPU: Python build + Metal encode | 0.65 ms build, hidden | 1.94 ms build + encode, against 3.5 ms of GPU: barely hidden |
| Dispatches / barriers / command buffers | 693 / 457 / 23 | 541 / 362 / 11 |

Two subtleties `[VERIFIED: R4 §2.2–§2.3]`:
- **Back-pressure.** For the 8B, `async_eval` itself blocks inside MLX's back-pressure: 23 command buffers per token against a limit of 10 in flight. The call therefore waits on the GPU for about half the step. Pipelining still works because the queue never drains.
- **Host work per token.** It is microseconds and overlapped: an EOS set check, `detokenizer.add_token`, `mx.get_peak_memory`, and a dataclass. The server's extra per-token `logprobs[token].item()` was measured to cost nothing (25.71 vs 25.81 ms).

### 4.3 Prefill

- **Chunking.** `generate_step` processes all prompt tokens except the last in chunks of `prefill_step_size` (default 2048). Each chunk runs:
  - a forward pass with the logits discarded;
  - optional KV quantization;
  - `mx.eval` of the cache state, a full CPU↔GPU sync;
  - `mx.clear_cache()`.

  The last token goes through the decode `_step`, so the LM head runs once. `speculative_generate_step` uses 512-token chunks. `[VERIFIED: mlx_lm/generate.py:425-456, 578-586; R4 §2.1]`
- **Measured on the M2 Pro.**
  - Qwen3-8B reaches 200 / 219.5 / 216.5 / 217.7 tok/s at 128 / 512 / 1,024 / 2,048 tokens. That is 3.3 TFLOPS on the linear layers, 58% of the 5.73 TFLOPS peak, and about 75% of what MLX's `qmm` reaches in isolation.
  - Qwen3-0.6B reaches 2,350 / 2,714 / 2,635 tok/s at 128 / 512 / 2,048.
  - `[VERIFIED: R4 §7.A]`
  - llama.cpp is 1.18–1.42× faster on the same models (§9.2).

### 4.4 Quantization formats and model loading

- **Files.** mlx-community Qwen3 checkpoints are affine 4-bit, group 64, with bf16 scales, biases and norms: 4.500 bits per quantized weight. `[VERIFIED: R4 §0, R9 §1.4]`
  - The files are written by MLX's own safetensors writer with no header padding. No tensor in `Qwen3-8B-4bit` is even 4-byte aligned. `[VERIFIED: R9 §1.2]`
- **Loading.** `mx.load(wf)` per shard (`mlx_lm/utils.py:455`), then `model.load_weights` (`:572`), then `mx.eval(model.parameters())` (`:575`). This is MLX's `pread` path (§3.6). `[VERIFIED]`
- **Tools.**
  - `mlx_lm.convert` (round-to-nearest affine);
  - `dynamic_quant`: 4/5-bit by layer sensitivity, target bpw 4.5–5.5;
  - DWQ: scales and biases distilled against the unquantized model;
  - AWQ and GPTQ.

  No published DWQ quality numbers were found. `[VERIFIED: R8 §4.4]`
- **Dropped weights.** MTP heads are discarded at load (`mlx_lm/models/qwen3_5.py:350` and three other model files). `[VERIFIED: R4 §5.1]`
- **Config drift.** mlx-lm drops unknown config keys without a warning. A Llama 3.1 config re-saved by transformers 5 loads with θ = 1e4 and no llama3 scaling. `[VERIFIED: R9 §0 item 10]`

### 4.5 Memory

| Item | Value or mechanism | Source |
|---|---|---|
| Peak, 8B decode (128 + 256 tokens) | MLX counter 4.92 GB; OS lifetime peak footprint 5.25 GB | `[VERIFIED: R4 §7.A]` |
| Peak, 8B prefill of 2,048 | 5.61 GB; OS 6.29 GB | same |
| KV growth | Every 256 tokens, `concatenate([old, zeros])`: one full copy of each layer's K and V. About 26 µs/token amortized at 4K for 8B | `mlx_lm/models/cache.py:339-375` `[VERIFIED code; LIKELY cost: R4 §4.1]` |
| Prompt-cache reuse | `fetch_nearest_cache` deep-copies. The copy shares buffers, so the first write cannot donate and copies the whole cache: 2× the conversation's KV transiently. A 16 GB Mac ran out of memory on reuse | `mlx_lm/models/cache.py:1656-1670` `[VERIFIED: R4 §4.1; mlx-lm#1395]` |
| Quantized KV | Higher peak than fp16 KV at 8K–32K (+30–73%). The unfused attention materializes fp32 (chunk × context) scores during prefill | `[VERIFIED: mlx-lm#1587, R4 §9]` |
| `peak_memory` as reported | Never reset in a long-lived process; excludes Python and driver memory | `[VERIFIED: R4 §2.6]` |

### 4.6 Attention and KV cache

- **`KVCache`.** Layout `(B, n_kv_heads, capacity, head_dim)` in the model dtype (bf16 here), grown in 256-token steps.
  - Writes are in-place `SliceUpdate`s: one dispatch each, verified by count.
  - Reads are strided views that the attention kernel takes without a copy.
  - `trim(n)` only decrements the offset, so rewind is O(1).
  - `[VERIFIED: R4 §4.1]`
- **`RotatingKVCache`.** A ring buffer for `max_kv_size`. It can be trimmed only until the window wraps, so speculative decoding stops working after that. It cannot be quantized. `[VERIFIED: R4 §4.2]`
- **`QuantizedKVCache`.** Enabled by `--kv-bits` from `quantized_kv_start` (default 5,000). Attention over it is unfused: `quantized_matmul(q, K)` → mask → `softmax(precise)` → `quantized_matmul(scores, V)`. `[VERIFIED: mlx_lm/models/base.py:64-107]`
  - Quantizing from step 0 cost 6–22% decode throughput on an M4 Pro (mlx-lm#1651).
  - At a 5,120-token prompt it was faster than fp16 (#1566).
- **Masks.** Decode uses none. Prefill passes `"causal"`, which selects the fused causal kernel without materializing a mask. `[VERIFIED: R4 §3.2]`
- **Measured decode vs context** (R4 §7.A):

  | Context | 128 | 1,024 | 2,048 | 4,096 | 8,192 |
  |---|---|---|---|---|---|
  | Qwen3-8B (GQA 4) | 39.2 tok/s | 36.6 | 35.1 | 31.7 | — |
  | Qwen3-0.6B (GQA 2) | 267 tok/s | 233 | 208 | 158 | 116 |

  - For 8B, the extra KV bytes from 128 to 4K are streamed at ≈ 97 GB/s, about 53% of 182.5.
  - For 0.6B the same slope is ≈ 189 GB/s.
  - The loss grows with the GQA factor, as the per-query-head K/V re-read predicts (§3.8). `[VERIFIED time; LIKELY mechanism]`
  - R3's 8-layer stack measured ≈ 56 GB/s for the 8B shape at 4K, a larger gap than the whole model shows. Both say the same thing: the KV stream is well below the line once G ≥ 4.

### 4.7 Sampling

- **Where.** Sampling runs on the GPU inside the token's graph. There is no host round trip for it. `[VERIFIED: R4 §2.4]`
- **Greedy.** `argmax(logits − logsumexp(logits))`, so greedy still pays a full-vocabulary logsumexp and subtract: 0.17 ms/token on 8B (0.7%) and 0.10 ms on 0.6B (3%). `[VERIFIED: mlx_lm/generate.py:418-419; R4 §7.E]`
- **With temperature.** A compiled filter chain (top-p → min-p → XTC → top-k), then `mx.random.categorical`. Top-p sorts the whole 151,936-entry vocabulary every token. Logits processors (repetition, presence and frequency penalties, logit bias) also run on the GPU. `[VERIFIED: mlx_lm/sample_utils.py:55-64, 137-273]`
- **Random state.** Draws come from MLX's global random state, not from a key per position. A speculative run therefore consumes random numbers differently from a plain run with the same seed. `[LIKELY: by construction; R8 §3.1]`

### 4.8 Speculative decoding

`speculative_generate_step` (`mlx_lm/generate.py:476-655`; R4 §5.1):

```text
round:
  draft k tokens serially with the draft model         (async_eval each; serial chain)
  verify [y, d1..dk] in one target forward              (M = k+1 rows)
  mx.eval(tokens, drafts); .tolist()                    full host sync, every round
  accept while target_token[i] == draft_token[i]        (Python loop, generate.py:624-635)
  emit the target's token at the first mismatch (or the bonus token)
  trim target cache by k−n, draft cache by max(k−n−1, 0)
```

| Property | Status | Source |
|---|---|---|
| Draft kinds | A separate draft model only. No prompt lookup or n-gram drafting, no EAGLE, Medusa or MTP | `[VERIFIED: R4 §5.1]` |
| Default draft length | 3 in the CLI (`generate.py:216-219`), 2 in the function (`:482`) | `[VERIFIED]` |
| Acceptance rule | The target samples each position independently, and a draft token is accepted only if it equals that sample. This is exact in distribution but accepts Σp·q instead of Σmin(p, q) | `[VERIFIED: R8 §3.1]` |
| Exactness at temperature 0 | Up to bf16 ties only: batched (M = k+1) and sequential (M = 1) logits differ in reduction order (#1470, #1423). Documented as "lossless up to floating-point associativity, not bit-exact" (#1592) | `[VERIFIED: R8 §3.4]` |
| Pipelining across rounds | None. The GPU idles while Python compares tokens, rewinds caches and builds the next draft graph | `[VERIFIED: R4 §2.3]` |
| Hybrid models | Require trimmable caches, so recurrent `ArraysCache` models cannot use it (#1446) | `[VERIFIED: R4 §5.1]` |

Measured and published results:

| Setup | Result | Source |
|---|---|---|
| Our M2 Pro: Qwen3-8B-4bit + Qwen3-0.6B-4bit, greedy, k = 2 / 4 / 6, chat / code / summary prompts | **0.37–0.74× of plain decode** (best 0.74×, code, k = 2). 3 of 9 configurations diverged from plain greedy (at tokens 26 and 99) | `[VERIFIED: R4 §7.D]` |
| Why: one 8B forward over M tokens on the M2 Pro | M = 2: 1.73×; 4: 3.33×; 5: 4.10×; 6: 6.16×; 8: 6.48×; 16: 6.61× | `[VERIFIED: R4 §7.D]` |
| M3 Max: Qwen2.5-32B-4bit + 0.5B-8bit draft, 4 drafts | 19.68 → 35.74 tok/s (**1.82×**), "The outputs are identical" (awni) | `[VERIFIED: mlx-examples#1155]` |
| M5 Max: 70B 4-bit + 0.6B draft, k = 3 chain | ≈ 18 → 24.5 tok/s | `[contributor data: mlx#3791]` |
| LM Studio (MLX engine), M3 Pro: Qwen2.5-32B + 0.5B on a code prompt | 7.30 → 17.74 tok/s (2.43×) | `[VERIFIED: R8 §2.5]` |
| LM Studio mlx-engine, M1 Pro: 7–8B targets with 1–3B drafts | **losses** of 11–39% | `[VERIFIED: R8 §2.2]` |
| EAGLE-3 prototype on mlx-lm, M3 Ultra, Llama-3.1-8B | 1.05× | `[VERIFIED: mlx-lm discussion #890, R8 §2.2]` |

On generation-14 GPUs (M1, M2) MLX has no `qmv_wide` for affine weights, so speculation cannot pay off whatever the acceptance rate. On M5 Max it probably gives 1.2–1.4× for a 32B target with a 0.6B draft `[LIKELY: R4 §7.D, from mlx#3791]`. The SPINE target "≥ 1.2× mlx-lm's own speculative mode" is meaningful only on M5-class hardware.

### 4.9 M5 neural accelerators

mlx-lm has no NAX code of its own. It inherits MLX's routing: prefill `qmm` → `qmm_t_nax`, prefill attention → `attention_nax`, decode on SIMD `qmv` / `qmv_wide` (§3.10). Apple's own M5 study used `mlx_lm.generate` with 4,096-token prompts: TTFT was 3.3–4.1× faster than on an M4, and decode 1.19–1.27× faster, "bounded by memory bandwidth". `[VERIFIED: R4 §8]`

### 4.10 Measured on our M2 Pro

| Metric | Qwen3-0.6B-4bit | Qwen3-8B-4bit | Source |
|---|---|---|---|
| Decode, 128-token prompt + 256 generated, external timestamps | 267.4 tok/s (inter-token latency 3.76 ms) | 37.4 tok/s (26.4 ms) | R4 §7.A |
| Decode, `generate_step` steady state | 281 tok/s | 38.8 tok/s | R4 §7.E |
| Effective bandwidth, % of 182.5 (KV included) | 54.5% | 91% | R4 §7.B |
| TTFT at 128 / 512 / 2,048 tokens (warm) | 55 / 189 / 777 ms | 0.64 / 2.33 / 9.41 s | R4 §7.A |
| Prefill tok/s at 512 / 2,048 | 2,714 / 2,635 | 219.5 / 217.7 | R4 §7.A |
| Dispatches / barriers / encoders per token | 541 / 362 / 11 | 693 / 457 / 23 | R4 §7.C |
| Fusion upper bound (GEMV-only chain vs full step) | 1.46× | 1.08× | R4 §7.E |
| Speculative, 0.6B draft | — | 0.37–0.74× | R4 §7.D |

All rows `[VERIFIED: R4]`.

### 4.11 Published M5-class numbers

| Result | Machine | Source | Label |
|---|---|---|---|
| Qwen3-32B 4-bit: tg128 **28.41** tok/s, pp512 **730.9** tok/s (MLX 0.32.2, mlx-lm 0.31.3) | M5 Max 40-core, 48 GB | mac-llm-bench PR #12, 2026-09-27 | LIKELY (one community run; method documented) |
| Qwen3-0.6B Q4: decode 398.1; prefill pp128 4,600, pp512 12,807, pp2048 16,198 | M5 Pro | BaseRT paper, Tables 1 and 3 | VERIFIED as published (vendor report) |
| Qwen3.6-27B Q4: decode 18.4; prefill pp128 305, pp2048 486 | M5 Pro | same | same |
| Qwen3-4B 4-bit: 134.5 tok/s generation (≈ 55% of 546 GB/s, derived) | M4 Max | mlx-lm `BENCHMARKS.md` | VERIFIED file; percentage LIKELY |
| MTPLX (built on MLX): Qwen3.6-27B 4-bit 30.37 tok/s plain, 81.74 "optimized" with MTP | M5 Max | MTPLX README | UNVERIFIED (self-reported) |

### 4.12 Strengths

- **The CPU is out of the critical path for models of 8B and up.** Token feedback through data dependence, sampling inside the graph, and one pipelined `.item()` per token (§4.2). The batch and server paths keep this property at batch 1 (R4 §7.F).
- **91% of the achievable bandwidth on Qwen3-8B decode**, with the GEMV chain alone at 97.5% (R4 §7.B).
- **Breadth.** Many architectures, a prompt cache with save/load/trim, continuous batching, an OpenAI-compatible server, and quantization tools with mixed bits and distillation.
- **Speculation exists and has paid on Max-class chips** (1.82× on an M3 Max).

### 4.13 Weaknesses, with evidence

| Weakness | Evidence | Label |
|---|---|---|
| Small models are bound by framework overhead | 0.6B at 54.5% of the roofline; 31% of the token is non-matmul dispatch; CPU 1.94 ms against 3.5 ms of GPU, so it would likely be CPU-bound on a 3× faster GPU. BaseRT decodes Qwen3-0.6B 1.33× faster on an M5 Pro | VERIFIED (R4 §7.E, §8); M5 consequence LIKELY |
| Speculation loop | A full sync per round; no cross-round pipelining; sample-and-match acceptance; draft model only | VERIFIED (R4 §2.3, §5.1) |
| Speculation is inexact | #1470, #1423, documented by #1592; 3 of 9 of our configurations diverged | VERIFIED |
| Verify cost on generation 13–14 GPUs | (k+1) tokens cost about (k+1) forwards | VERIFIED (R4 §7.D) |
| Long-context decode | KV stream ≈ 2× below the line at GQA 4; mlx-lm#763 reported ≈ 50% of llama.cpp at 146K context on an M3 Ultra, after which MLX shipped a faster 2-pass kernel (mlx#3023) | VERIFIED |
| Quantized KV | Unfused: slower at short context, higher prefill peak memory | VERIFIED (#1651, #1587) |
| Memory transients | KV growth by concatenation; 2× KV on prompt-cache reuse | VERIFIED code (R4 §4.1) |
| Prefill on M1–M4 | 58% of ALU peak; llama.cpp is 1.18–1.42× faster on the same models | VERIFIED (R4 §7.A; §9.2) |
| Determinism across chips | Temperature-0 output differed between an M3 Ultra and an M5 Max on the same request (mlx-lm#1280, open) | VERIFIED as reported |
| Self-reported metrics | `generation_tps` biased by (n+1)/n; `mlx_lm.benchmark` uses random token ids and arithmetic means | VERIFIED (R4 §2.6) |

### 4.14 Maintainers' stated positions

| Topic | Statement | Link |
|---|---|---|
| Speculative exactness | Documentation only: "lossless up to floating-point associativity, not bit-exact" | mlx-lm#1470 → PR #1592 |
| Native MTP | "sampler is coupled with generation step, and module implementation is coupled with speculative decoding"; later "I will see if I can implement this in a clean and progressive way" (zcbenz). The PR is still open | mlx-lm#990 |
| n-gram self-speculation | "At the moment we don't accept AI written architecture changes" (zcbenz, 2026-08-26) | mlx-lm#1497, #1498 |
| Review capacity | "the number of PRs is way beyond our capacity to review so I'm closing the non-essential ones" (zcbenz) | mlx-lm#1652 |
| Long-context gap | Asked for reproduction commands (awni); MLX then landed a faster 2-pass SDPA | mlx-lm#763, mlx#3023 |

`[VERIFIED: R4 §9, R3-issues E6]`

R4's reading, which we share: mlx-lm is conservative about changes to the generation loop, and performance work happens in MLX core kernels. A fused sampler, a compiled decode step and n-gram drafting proposed by outsiders were not merged in 2026 (R4 §9). For Sky this means the loop-level openings (§12) are likely to stay open longer than the kernel-level ones.

### 4.15 What Sky copies and what Sky avoids

| Copy | Why |
|---|---|
| Queue token t+1 before reading token t, with the token fed back on the GPU | Removes the host from the critical path |
| Sampling inside the device program | No round trip |
| O(1) cache trim for rewind; prompt caches with save, load and trim | Speculation and TTFT |
| The external-timestamp method of R4 and the #1470 script as an adversarial test | Fair measurement; exactness CI ([Part XXI](part21-speculative.md)) |
| Mixed-bit and distillation tools as later, labeled modes | G9 work under a KL budget |

| Avoid | Why |
|---|---|
| A Python loop per token for small models | CPU-bound drafts |
| A blocking host sync per speculative round and Python-side acceptance | Idle GPU every round |
| Sample-and-match acceptance; a global RNG | Lower acceptance; not keyed, so not reproducible under speculation |
| KV growth by concatenation; deep-copied prompt caches | Transient 2× memory |
| Unfused quantized-KV attention | Slower and larger than fp16 |
| Silently dropping unknown config keys | Wrong RoPE without an error |

---

## 5. tinygrad

tinygrad is a small tensor compiler in Python. One hash-consed IR represents the tensor graph, the kernels, the schedule and even the host submission code, and every transformation is a pattern rewrite. It ships an LLM runner (`python -m tinygrad.llm`) that loads GGUF files. Commit `512e9b2`. Much of what is commonly written about tinygrad (ShapeTracker, LazyBuffer, the old OptOps, `MetalGraph`) is gone at this commit (R2 §1).

### 5.1 Architecture

| Stage | Mechanism | Source |
|---|---|---|
| IR | `UOp(op, src, arg, tag)`, hash-consed; dtype and shape are *derived* properties; equality is identity; a sha256 key drives every cache | `tinygrad/uop/ops.py:122-269` `[VERIFIED: R2 §3.1]` |
| Rewrites | `PatternMatcher` rules indexed by root op, with early reject; matchers compiled to Python. Fixed-point and single-pass drivers | `tinygrad/uop/ops.py:1453-1890`; `tinygrad/uop/upat.py` `[VERIFIED: R2 §3.7]` |
| Views | Movement ops become index arithmetic on `RANGE` variables ("rangeify"). Fusion means sharing ranges; materializing means fresh ranges plus `STAGE` → `STORE` | `tinygrad/schedule/indexing.py:169-320` `[VERIFIED: R2 §3.3]` |
| Kernel boundaries | Every `STORE`; multiple consumers with disagreeing ranges; reduce-then-expand; too many buffers; large reductions split in two | `tinygrad/schedule/indexing.py:45-277`; `schedule/rangeify.py:170-348` `[VERIFIED: R2 §5.1]` |
| Recompute vs store | Store if `contiguous()`, if it reads more than 3 buffers, or if a reduce inside reads a buffer; otherwise recompute. No cost model | `tinygrad/schedule/rangeify.py:84-96` `[VERIFIED: R2 §5.2]` |
| Optimization | `OptOps = {TC, SPLIT, PADTO, SWAP}`; hand-coded heuristics, or BEAM search with a disk cache | `tinygrad/codegen/opt/` `[VERIFIED: R2 §6]` |
| Code generation | 30 `graph_rewrite` passes, then an MSL renderer that puts every buffer pointer and scalar into one `args_t` argument buffer | `tinygrad/codegen/__init__.py:275-397`; `renderer/cstyle.py:358-409` `[VERIFIED: R2 §7.1–§7.2]` |
| Compile | The **private** `MTLCompiler.framework` code-generation service, not `newLibraryWithSource`; metallibs cached on disk | `tinygrad/runtime/ops_metal.py:29-71` `[VERIFIED: R2 §7.4]` |
| Runtime ("HCQ2") | Kernels are batched into an `MTLIndirectCommandBuffer` with one argument buffer. The host submission program is itself a UOp program, rendered to C and compiled by clang. Input addresses and scalar variables are patched per call | `tinygrad/runtime/support/hcq2.py`; `runtime/ops_metal.py:91-166, 249-266` `[VERIFIED: R2 §7.5]` |
| JIT | `TinyJit`: call 0 runs eagerly, call 1 captures, later calls replay | `tinygrad/engine/jit.py:104-178` `[VERIFIED: R2 §7.6]` |
| Memory plan | Buffer lifetimes, suballocated with TLSF into one arena per device and lane | `tinygrad/schedule/memory.py:20-62` `[VERIFIED: R2 §7.7]` |

### 5.2 One decode token through tinygrad, exactly

```text
Python: rollout_jit(token, start_pos bound as a symbolic Variable)   (tinygrad/llm/model.py:410-423, 531-547)
  TinyJit replay: check input names, views, dtypes                   (engine/jit.py:169-175)
  exec_hcq: write input GPU addresses into the inputs table;
            write start_pos into the argument buffer
  call the compiled C function hcq_submit():
    hcq_fence: spin until the previous replay of this batch has finished   (replays never overlap)
    mtl_run:  commandBuffer → computeCommandEncoder → waitForFence
              [Apple8 and earlier: setComputePipelineState + an empty dispatch
               per distinct pipeline]
              executeCommandsInBuffer: an ICB of 488 commands,
                                       setBarrier() on every command
              updateFence → endEncoding → encodeSignalEvent → commit
  .item() on the sampled token: MTLSharedEvent wait, one host sync per token   (llm/model.py:547)
GPU: 488 kernels per Qwen3-0.6B token (about 17 per layer), each after a barrier
```

`[VERIFIED: R2 §7.5–§7.6, §8.3]`

Measured costs on the M2 Pro (R2 §7.5):
- one JIT call with one tiny kernel: 155 µs round trip;
- each additional kernel in the same ICB: +4.3 µs;
- an eager `realize` with all caches warm: 453 µs.

Profiling mode puts every kernel in its own command buffer, which inflates a token from 18.4 to 29.4 ms.

### 5.3 Decode and prefill kernels

All kernels are generated. tinygrad has hand-written quantized kernels only for AMD RDNA3/4 (`tinygrad/llm/kernels/amd.py`). On Metal a quantized `Linear` falls through to the generic path. `[VERIFIED: R2 §8.2]`

**The fused-dequant GEMV.** Measured at 12,288 × 4,096 (R2 §8.4); per-call times include the 155 µs round trip:

| Weights | Per call | Effective GB/s on packed bytes | Kernel only |
|---|---|---|---|
| fp16, heuristic matvec path | 751 µs | 134 | 128–149 GB/s |
| fp16, BEAM | 709 µs | 142 | **≈ 189 GB/s** |
| Q8_0, fused | 623 µs | 86 | — |
| Q4_0, fused | 530–544 µs | 52–53 | 315–510 µs, varies by run |
| Q6_K, fused | 1,072 µs | 38.5 | 1.1–1.2 ms |

Why the quantized kernels are slow, read from the generated MSL (R2 §8.4):
- **One thread computes 3 full output rows over all of K.** Lanes of a simdgroup sit 6,912 bytes apart, with no cross-lane reduction, so loads do not coalesce.
- **The nibbles are loaded as 54 scalar `uchar` loads per iteration.** The coalescing pass excludes `uint8` (`tinygrad/codegen/late/coalesce.py:143`).
- **The nibble selector becomes a runtime integer division** (`val/alu5`), because the 2-wide nibble loop is not unrolled.
- **Each weight goes through 4 conversions.**
- **The matvec heuristic never fires for quantized weights.** It requires both multiplicands to be direct loads, and a dequantized weight is an expression (`tinygrad/codegen/opt/heuristic.py:62-64`).

The same generator reaches ≈ 189 GB/s on an fp16 GEMV once BEAM finds the schedule. The weakness lies in quantized loads and in the heuristics, not in the IR.

**Matmul and prefill.** The tensor-core abstraction is a declarative fragment layout (`tinygrad/renderer/tc.py:6-48`). On Metal it is one 8×8×8 `simdgroup_matrix` shape (`:137-141`).
- Fragments load straight from device memory. There is no threadgroup staging, which the maintainers acknowledge (§5.15).
- There is no flash attention on Metal. Attention is composed from tensor ops (`tinygrad/llm/model.py:242`).
- R2 did not measure steady-state prefill. The first prefill call took 7.6–10 s, mostly scheduling and compilation.

`[VERIFIED: R2 §6.3, §8.1, §8.3]`

### 5.4 Quantization formats

`gguf_load` keeps the packed GGUF blocks on the device and returns one lazy decode expression per tensor. Q4_0, for example, is `(q_to_uint8(blocks[:,2:],4).bitcast(int8)-8) * blocks[:,:2].bitcast(f16).cast(f32)` (`tinygrad/llm/gguf.py:52`). `[VERIFIED: R2 §8.2]`

| Aspect | Behaviour |
|---|---|
| Formats | Q4_0/1, Q5_0/1, Q8_0, Q2_K…Q6_K, the IQ family, MXFP4, Q1_0 (`llm/gguf.py:22-38`) |
| Default (`REALIZE=0`) | Dequantization is fused into every consuming kernel, every token |
| `REALIZE=1` | Dequantizes to fp16 at load: 3.2× the bytes, yet **faster** (76.3 vs 54.3 tok/s on 0.6B) |

### 5.5 Model loading

The GGUF file is read through a disk device and copied into device buffers. tinygrad does not use `newBufferWithBytesNoCopy` for weights. The 0.6B model loaded at 0.15–0.32 GB/s. The first steady token arrived after 11–14 s, mostly JIT scheduling, compilation and capture. `[VERIFIED: R2 §8.3, §9.4]`

### 5.6 Memory

- The JIT memory planner cut the decode step's intermediates from 6.92 MB in 457 buffers to 1.48 MB in 2. It cut eager prefill from 260.47 MB in 708 buffers to 5.11 MB in 2. `[VERIFIED: R2 §7.7]`
- Buffers are shared-storage and held in a residency set.
- The generic allocator keeps an LRU free list, and `free` synchronizes the device.

### 5.7 Attention and KV cache

- **Layout.** An fp16 tensor `(2, B, n_kv_heads, max_context, head_dim)`, zero-initialized.
- **Writes and reads.** An explicit `STORE` into a symbolic slice, then `AFTER`, then a read through a symbolic-length slice `0:start_pos+T`.
- **JIT variables.** `start_pos` and `toks` are bounded `Variable`s, so one captured program serves every position.
- **Masks.** Built only when T ≠ 1.

`[VERIFIED: tinygrad/llm/model.py:213-252; R2 §8.1]`

### 5.8 Sampling

Gumbel-max argmax on the GPU inside the JIT (`tinygrad/llm/model.py:418-420`). The CPU reads the token with `.item()` every step (`:547`). `[VERIFIED]`

### 5.9 Speculative decoding

None in `tinygrad/llm`. `[VERIFIED: grep, R5 §12]`

### 5.10 M5 neural accelerators

Not used. Metal tensor cores are the 8×8 `simdgroup_matrix` only. A grep for `tensor_ops`, `matmul2d`, `MetalPerformancePrimitives`, `cooperative_tensor` and `MTL4` finds only the `-std=metal4.0` flag, a comment and an unused enum value. `[VERIFIED: R2 §6.3]`

### 5.11 Measured on our M2 Pro

| Metric | Result | Source |
|---|---|---|
| Qwen3-0.6B Q4_0 GGUF (the file llama.cpp also ran), default fused dequant | **54.3 tok/s** (18.41 ms; n = 26); ≈ 21 GB/s ≈ 11% of the raw read | R2 §8.3 |
| Same, `REALIZE=1` (fp16 weights) | 76.3 tok/s (13.10 ms) at ≈ 1.2 GB/token, ≈ 92 GB/s | R2 §8.3 |
| Kernels per token | **488** (≈ 17 per layer); 287 of them under 20 µs | R2 §8.3 |
| Largest kernels in one token (profiled) | LM head over the Q6_K tied embedding: 5.7 ms (27%) at ≈ 22 GB/s; FFN gate/up: 4.5 ms (21%) | R2 §8.3 |
| BEAM search cost | 4.0–4.35 s per GEMV kernel, cold | R2 §6.4 |
| BEAM ranking on Q4_0 | Reported 206 µs, picked a kernel that takes ≈ 540 µs. Metal timing never flushes caches (`codegen/opt/search.py:144`), so a 28 MB working set partly sits in the system cache | R2 §6.4 |
| Qwen3-8B | Not run (16 GB machine; long capture). Expected ≈ 11 tok/s from the per-format GEMV rates | R2 §10 (LIKELY) |

All rows `[VERIFIED: R2]` except the last.

### 5.12 Published numbers

- The tiny corp on X:
  - "Our baseline is already 178 tok/s, beating LMStudio (160) and llama.cpp (140) out of the box, but with tinygrad's custom kernel feature Claude cranked it to 195.7!"
  - The model is Qwen3.5-0.8B on an M3 Max. Quantization and settings are not stated; the CLI default for that model is Q8_0.
  - [X post](https://x.com/__tinygrad__/status/2046973311977230768). `[claim, not verified: R2 §8.5]`
- R2's cross-check is consistent with this. 178 tok/s on Q8_0 implies about 40% of the M3 Max's bandwidth, close to the 42% that tinygrad's fused Q8_0 GEMV reaches on our M2 Pro. `[LIKELY: R2 §8.4]`
- No M5 numbers exist, and there is no M5-specific path.

### 5.13 Strengths

- **One IR all the way down.** Graph, kernels, schedule and host submission code are all UOps. Each can be rewritten, cached by structural hash, and replayed (`PROCESS_REPLAY`) or visualized (`VIZ`).
- **Rangeify makes views free and fusion simple.** It is a clean model to copy.
- **A symbolic-integer layer.** Variables carry bounds and divisibility, and validity becomes a gate instead of a branch.
- **A small, searchable optimization vocabulary** (SPLIT, TC, PADTO, SWAP) with a persistent cache.
- **Replay with constant CPU cost.** One ICB, one argument buffer, and per-call patching of only input addresses and scalars.
- **Static memory planning** that collapses hundreds of buffers into two.
- **Many backends.**

### 5.14 Weaknesses, with evidence

| Weakness | Evidence | Label |
|---|---|---|
| Fused quantized GEMV | 19–43% of peak bandwidth for Q6_K, Q4_0 and Q8_0 | VERIFIED (R2 §8.4) |
| Too many kernels, each after a barrier | 488 per 0.6B token; `setBarrier()` on every ICB command, so independent kernels never overlap | VERIFIED (`runtime/ops_metal.py:263`) |
| Replay serialization and a host sync per token | `hcq_fence` waits for the previous replay of the same batch; `.item()` every token | VERIFIED (R2 §7.5, §8.1) |
| Autotuning methodology on Apple | No cache flush; minimum of 3 runs; extrapolated grid sizes; mis-ranked Q4_0; cache key without GPU model or OS | VERIFIED (R2 §6.4) |
| Compile latency in Python | 76–151 ms per block-schedule miss; 785 ms for the outer prefill graph; 3.7–7.9 ms per cold kernel | VERIFIED (R2 §3.7) |
| No threadgroup staging; no Metal 4 tensor ops | Matmul fragments come straight from device memory | VERIFIED (R2 §6.3) |
| Private compiler API | `MTLCodeGenService` is not public API ([Part VI](part06-architecture.md) §8.3 rejects it for Sky) | VERIFIED (R2 §7.4) |

### 5.15 Maintainers' stated positions

From `docs/developer/speed.md` at this commit `[VERIFIED: R2 §5.3]`:
- "when is it worth recomputing an arithmetic operation instead of storing… tinygrad doesn't yet have a systematic way to do it" (`:28-43`).
- "we never copy the inputs to on chip SRAM, but this is often quite helpful for kernel speed. Also, we aren't doing a good job with L2 cache awareness" (`:59`).
- Compile speed is "limited largely by the runtime of the Python doing UOp rewrites. Currently it's a bit slow, but on par with torch.compile" (`:14-16`).
- On driver execution speed: "very little to improve here" (`:20-22`). Our Metal measurements of batch-1 decode do not support this: 155 µs per JIT round trip, plus 4.3 µs per kernel × 488 kernels.

### 5.16 What Sky copies and what Sky avoids

| Copy (re-implemented in Rust; D15) | Where in Sky |
|---|---|
| An arena-interned, hash-consed SSA IR with derived types and shapes, rewritten by declarative rules | `sky-tir` ([Part VII](part07-tensor-ir.md)) |
| Rangeify: movement ops as index functions; fuse where consumers agree on indices | [Part IX](part09-scheduler.md) |
| Bounded symbolic integers (`pos: Int<0..MAX_CTX>`) for index proofs and divisibility | `sky_types::poly::Poly` plus bounds |
| SPLIT / TC / PADTO / SWAP as the tuning vocabulary; declarative MMA fragment layouts | `sky-kernel`, `sky-tune` ([Part X](part10-autotuner.md)) |
| One argument buffer per replayed batch; per-call patching of addresses and scalars only | `sky-metal` ([Part XX](part20-persistent-execution.md)) |
| TLSF lifetime planning into one arena; structural-hash schedule caches; replayable rewrite traces | [Part XIX](part19-memory.md), `sky explain` |

| Avoid | Why |
|---|---|
| Generic code generation for quantized GEMV | 53 GB/s against 177 GB/s for a hand-designed kernel |
| A barrier on every command | No overlap of independent kernels |
| Tuning without cache flushes, with minimum-of-3 timing and device-agnostic keys | Mis-ranked kernels |
| Recompute-vs-store by rule of thumb | Sky uses a cost model |
| Private APIs | Product risk |

---

## 6. llama.cpp

llama.cpp is a C/C++ inference engine on the ggml tensor library. Its Metal backend is 100% hand-written MSL. It is the most portable of the four engines, and on Apple GPUs the most carefully engineered at the system level. Commit `81e39ad`. Paths below are relative to the repository root; the Metal backend lives in `ggml/src/ggml-metal/`.

### 6.1 Architecture

| Component | Mechanism | Source |
|---|---|---|
| Model graphs | C++ builds a static ggml graph per micro-batch shape. A graph is **reused** across tokens while its topology holds; the KV view length is padded to 256 for that reason | `src/llama-context.cpp:1398-1470`; `src/llama-kv-cache.cpp:1260-1273` `[VERIFIED: R5 §6]` |
| Encoding | The calling thread encodes the first max(64, 10%) of the nodes and commits them at once, so the GPU starts early. One GCD worker encodes the rest into a second command buffer. Buffers are enqueued in order and use unretained references. The graph call returns without waiting | `ggml/src/ggml-metal/ggml-metal-context.m:474-660` `[VERIFIED: R5 §1.1]` |
| Hazards | One concurrent-dispatch encoder per command buffer. Each node's source and destination byte ranges are tracked, and `memoryBarrierWithScope` is emitted only on a real read-after-write, write-after-read or write-after-write overlap | `ggml-metal-ops.cpp:220-242`; `ggml-metal-common.cpp:51-228`; `ggml-metal-device.m:834-845, 892-893` `[VERIFIED: R5 §1.2]` |
| Reordering | A greedy list scheduler looks up to 64 nodes ahead and hoists independent ones into the current barrier-free group, so Q, K and V land together. Runs once per built graph | `ggml-metal-common.cpp:264-507` `[VERIFIED: R5 §1.3]` |
| Fusion | One table of 9 patterns, at most 16 nodes each: norm + mul (+ add, + scale), add chains, MoE router and reduce, SSM and Gated-DeltaNet helpers. **No GEMV prologue or epilogue, no QKV or gate/up merge, no RoPE + KV-write fusion** | `ggml-metal-fusion.cpp:674-701` `[VERIFIED: R5 §1.3]` |
| Pipelines | 34 per-kind libraries compiled at run time from embedded MSL, in parallel. Pipelines are specialized lazily with function constants and cached in a locked dictionary; no binary archives. The first compile after installation took 4.84 s; later starts 14–48 ms (the OS shader cache) | `ggml-metal-device.m:331-545, 721-818` `[VERIFIED: R5 §1.6]` |

### 6.2 One decode token through llama.cpp, exactly

```text
llama_decode(batch of 1 token)                                   (src/llama-context.cpp:1398-1470)
  previous graph reusable? yes, while n_kv stays inside its 256-padded bucket
      → skip graph build, scheduler reset and allocation
  set_inputs: token id, position, KQ mask, KV write indices
  scheduler: CPU split (token-embedding lookup), then the Metal split
  Metal graph_compute                                            (ggml/src/ggml-metal/ggml-metal-context.m:474-660)
      main thread : encode the first max(64, 10%) nodes → commit (the GPU starts)
      GCD worker  : encode the remaining nodes into a second command buffer
      per node    : snprintf the pipeline name + locked dictionary lookup,
                    setBytes / setBuffer, byte-range hazard check → barrier only if needed
      return without waiting
  logits: asynchronous copy of 151,936 × 4 B = 608 KB to the host
sampler: llama_get_logits_ith → synchronize (waitUntilCompleted)  (src/llama-context.cpp:4001-4006)
      CPU sampling chain → token id on the host → next llama_decode
GPU per Qwen3-8B token: 688 encoded nodes ≈ 724 dispatches, ≈ 542 barriers;
      182 nodes ran concurrently with a predecessor
```

`[VERIFIED: R5 §1, §6, §8.D2]`

Per layer this is 19 nodes, about 20 dispatches (attention is a vector kernel plus a reduce) and about 15 barriers. That is the same structure as MLX's 19 dispatches. A two-point fit of decode time against weight bytes and layers gives t ≈ bytes / 168 GB/s + 64 µs × layers. That is about 1.8 ms (0.6B) and 2.3 ms (8B) per token not explained by streaming, or about 3 µs per dispatch-plus-barrier. `[EXPERIMENTAL: R5 §8.B, §8.D2]`

Feature ablations on decode (R5 §8.F):

| Disabled | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| Graph reuse | **−42%** (≈ 2.9 ms of CPU per token) | −5% |
| Fusion | −0.5% | −4.5% |
| Concurrency | −4.2% | −2.3% |
| Reorder pass | +1.0% (noise) | −0.4% |

`[VERIFIED: R5 §8.F]`

### 6.3 Decode kernels

**Kernel choice by number of token columns** (`ggml/src/ggml-metal/ggml-metal-ops.cpp:2423-2648`):

| Columns | Q4_0, Q8_0, MXFP4, F16, … | K-quants |
|---|---|---|
| 1 | `mul_mv` | `mul_mv` |
| 2–3 | `mul_mv_ext` | `mul_mv`, one grid row per column (weights read 2–3×) |
| 4–8 | `mul_mv_ext` | `mul_mv_ext` |
| > 8 | `mul_mm` | `mul_mm` |

`[VERIFIED: R5 §2.1]`

**The Q4_0 GEMV** (`ggml/src/ggml-metal/kernels/mul_mv.metal:85-100, 218-304`):
- 2 simdgroups per threadgroup, 4 rows per simdgroup (unchanged since July 2023, #2188);
- each thread handles half a 32-weight block, with 16 blocks in flight per simdgroup;
- activations are pre-divided by 1, 256, 16 and 4096, so a nibble needs one mask on a `uint16` load. The −8 zero point folds into Σy;
- the activation registers are reused across the 4 rows;
- one `simd_sum` per row; no threadgroup memory.

This is the same arithmetic idea as MLX's `qmv_fast`. MLX's `qmv_wide` credits llama.cpp's `mul_mv_ext` as its source (mlx#3764). `[VERIFIED: R5 §2.3]`

**`mul_mv_ext`** (columns 2–8). Threads per row and columns per threadgroup are hand-picked; the code says it is "not sure how optimal are those across all different hardware" (`ggml-metal-ops.cpp:2481-2489, 2505`). Columns per threadgroup are 2, 3, 4, 5, 3, 4, 4 for batches 2 to 8, so **weights are read once for 2–5 columns and twice for 6–8**. `[VERIFIED: R5 §2.3]`

Measured kernel throughput, 4096 × 14336 matrix, n = 1 (R5 §8.E):

| Q4_0 | Q8_0 | Q6_K | F16 | Q4_K | MXFP4 |
|---|---|---|---|---|---|
| **177 GB/s** | 176 | 172 | 171 | **147** | **138** |

`[VERIFIED: R5 §8.E]`

### 6.4 Prefill kernels

- **`mul_mm`, simdgroup path (all pre-M5 GPUs).**
  - Tile: 64 weight rows × 32 token columns, K step 32, 4 simdgroups.
  - Each thread dequantizes 16 weights per step into threadgroup memory as half; activations are rounded to half.
  - 8×8 `simdgroup_matrix` multiply into fp32 accumulators.
  - Measured: 3.7–4.9 TFLOPS at n = 512 on the M2 Pro. Qwen3-8B pp512 end to end at 312 tok/s ≈ 4.7 TFLOPS, 83% of the 5.73 peak.
  - `[VERIFIED: ggml/src/ggml-metal/kernels/mul_mm.metal:146-359; R5 §3, §8.E]`
- **Metal 4 tensor path.**
  - `mpp::tensor_ops::matmul2d` with cooperative tensors: 64 × 128 tile, K step 32. Weights are dequantized into threadgroup memory; activations are consumed straight from device memory.
  - Enabled by default only when the GPU name contains M5, M6, A19 or A20. In the code's own words: "M2 Ultra: ~5% slower; M4, M4 Max: no significant difference. TODO: try to update the tensor API kernels to at least match the simdgroup performance" (`ggml-metal-device.m:1145-1162`).
  - The current kernel was contributed by Apple's Developer-Ecosystem-Engineering account (#20962: +26.4% geomean prompt processing on an M5 Max).
  - A bug silently disabled the path on some M5 builds until #27461 (2026-09-01).
  - `[VERIFIED: R5 §3, §9.5]`

### 6.5 Quantization formats

| Type | Block | Bytes | bpw | Dequantization |
|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.5 | `d·(q − 8)`; split-half nibbles |
| Q8_0 | 32 | 34 | 8.5 | `d·q` |
| Q4_K | 256 | 144 | 4.5 | `d·sc·q − dmin·m`; 6-bit sub-scales |
| Q6_K | 256 | 210 | 6.5625 | `d·sc·(q − 32)` |
| MXFP4 | 32 | 17 | 4.25 | E2M1 table × E8M0 |
| NVFP4 | 64 | 36 | 4.5 | 4 × UE4M3 sub-scales |

Also present: Q4_1, Q5_x, Q2_K/Q3_K/Q5_K, the IQ codebook family, TQ, Q1_0/Q2_0. `[VERIFIED: ggml/src/ggml-common.h; R5 §5, R9 §2.2]`

Real files are mixtures. llama-quantize keeps the output head at Q6_K, and with an imatrix it promotes early `ffn_down` tensors to Q4_1. So "Q4_0" Qwen3-8B is 4.67 bpw and Q4_K_M 4.90 bpw (R9 §2.3). GGUF has no group-64 affine type. As ggerganov put it in 2024: "We don't support group size of 64 atm (which is what I think MLX uses), so can't make an apples-to-apples comparison with MLX." `[VERIFIED: R5 §9.6]`

### 6.6 Model loading

- **Mapping.** The default load mode is `mmap`. The mapped tensor region is wrapped directly with `newBufferWithBytesNoCopy`: the pointer is rounded down and the length up to 16 KiB pages. Regions larger than `maxBufferLength` become several overlapping views, so every tensor fits in one. `[VERIFIED: src/llama-model.cpp:1854-1868; ggml-metal-device.m:2176-2270]`
- **Layout.** Tensors are used in their on-disk block layout; repacking exists only for the CPU backends.
- **Measured.** Qwen3-8B loaded in 3.6 s with a cold-ish page cache and 0.42 s warm. `[VERIFIED: R5 §8.D]`
- **Alignment.** GGUF tensors are 32-byte aligned relative to the data section, which is enough for vector loads (R9 §2.1).

### 6.7 Memory

| Topic | Mechanism | Source |
|---|---|---|
| Residency | Every Metal buffer gets its own residency set. A background heartbeat calls `requestResidency` every 5 ms for 180 s after the last compute. It exists because macOS began unwiring GPU memory after about 1 s idle: issue #10119 measured 26.34 ms/token becoming **429.08 ms after a 1.4 s pause** | `ggml-metal-device.m:925-1066` `[VERIFIED: R5 §1.4, §9.5]` |
| Non-weight buffers | KV and compute buffers are `vm_allocate`d host memory wrapped no-copy (shared storage) | `ggml-metal-device.m:2108-2171` `[VERIFIED]` |
| Qwen3-8B at 4,096 context | Model 4,559.9 MiB (mapped) + KV 576 MiB + compute 304.8 MiB on the GPU; 353 MiB on the host | `[VERIFIED: R5 §8.D]` |
| Default context | The model's trained length (40,960 for Qwen3), auto-fitted to free memory. RAM numbers depend on `-c` | `common/common.h:452, 478` `[VERIFIED: R5 §11]` |
| Accounting | Mapped weights read only by the GPU are invisible to per-process footprint and RSS. They show only as system wired memory | `[VERIFIED: R9 §6.2]` |
| Watchdog | Sets the undocumented `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` against display-active GPU timeouts | `ggml/src/ggml-metal/ggml-metal.cpp:1027-1029` `[VERIFIED: R6]` |

### 6.8 Attention and KV cache

- **Selection.** The vector kernel is used if there are fewer than 20 query rows and the head size is a multiple of 32; otherwise the tiled kernel. Flash attention is on by default (`-fa auto`). `[VERIFIED: ggml-metal-ops.cpp:2949-2957; src/llama-context.cpp:557-560, 3769]`
- **Decode (vector kernel).**
  - One query per threadgroup and 32 KV positions per simdgroup step.
  - **Split-K over 32 workgroups**, then a second reduce kernel behind a barrier.
  - Its (Q, NE) parameters come from an **offline-tuned table per GPU family**, with 366 rows for Apple10 (M5 Max).
  - K and V may be F16, BF16, F32, Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0, dequantized in-kernel.
  - `[VERIFIED: ggml-metal-ops.cpp:3575-3830; ggml-metal-tuning.cpp; R5 §4]`
- **Prefill (tiled kernel).** 8 queries per threadgroup, 64 KV per step. A pre-pass classifies mask blocks so fully masked blocks are skipped. With quantized KV and batch ≥ 32, K and V are first dequantized to an F16 scratch buffer. `[VERIFIED: R5 §4]`
- **KV cache.** Preallocated cells, f16 by default.
- **Measured.** Flash attention on was never slower on the M2 Pro: +4% decode on 8B, +8–11% on 0.6B (R5 §8.A).
- **Published.**
  - Split-K vector attention (#15566) took gemma3-1B decode at 32K depth from 98.65 to 172.59 tok/s on an M2 Ultra.
  - At 146K context, llama.cpp decoded 2× faster than MLX on an M3 Ultra (mlx-lm#763).
  - `[VERIFIED: R5 §9.5, R3-issues F1]`

### 6.9 Sampling

- **Default.** Sampling runs on the CPU over the full logits copy, with a chain: penalties → dry → top-n-sigma → top-k → typical → top-p → min-p → xtc → temp-ext → dist.
- **GPU samplers.** Available since #17004 (greedy, dist, top-k, top-p, min-p, temperature, penalties, logit bias). They are off by default for generation (`common/common.h:298`) and on by default for drafting (`:334`).
- **Host round trip.** Even with GPU sampling the token id returns to the host before the next `llama_decode`, so there is one round trip per token. R6 measured that a CPU wait before committing the next command buffer leaves the GPU idle ≈ 105 µs on the M2 Pro.
- `[VERIFIED: R5 §6; R6 §B3]`

### 6.10 Speculative decoding

| Property | Status | Source |
|---|---|---|
| Draft kinds | `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, plus five n-gram modes (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`). Available in the server | `common/common.h:173-186` `[VERIFIED: R5 §7]` |
| Defaults | 3 draft tokens; `ngram-simple` needs a 12-token match | `common/common.h:327-363` |
| Acceptance | Sample the target and stop at `draft[i] != id`: the same sample-and-match rule as mlx-lm | `common/sampling.cpp:678-710` `[VERIFIED: R8 §3.1]` |
| Verify kernels | `mul_mv_ext`: weights read once for 2–5 tokens, twice for 6–8 | R5 §2.3 |

Measured on our M2 Pro: Qwen3-8B Q4_0 target with Qwen3-0.6B Q4_0 draft, greedy code prompt (R5 §8.C, §8.G):

| Configuration | Decode | vs plain | Acceptance |
|---|---|---|---|
| Plain (`llama-completion`) | 34.58 tok/s | 1.00× | — |
| `draft-simple`, 3 drafts | 38.78 | **1.12×** | 79.8% |
| `draft-simple`, 7 drafts | 32.72 | 0.95× | 64.1% |
| `ngram-simple` | 33.31 | 0.96× | no drafts (no 12-token match) |

- A 4-token verify cost 2.08 single-token decodes, and drafting about 0.47.
- With a verifier costing 1.0 decode, the same acceptance would give about 2.3×. Roughly half of the available speedup is lost to the small-batch kernels and per-call overhead on this machine. `[EXPERIMENTAL decomposition of VERIFIED totals: R5 §8.G]`

Published:
- Draft models on Apple: mostly no gain in community reports (#10455). M4 Pro 32B + 3B: 10.78 vs ≈ 11 tok/s.
- MTP (#22673): mixed, from 0.67× (M3 Max) to 1.47× (M1 Ultra, 17 → 25 tok/s).
- DFlash: 1.85× on an M5 Pro (#27342, author's run); 0.97× on an M2 Max despite 76% acceptance.
- A Q = 2 verify attention kernel gave 1.75× on an M5 Max (#23114).

`[VERIFIED as published: R5 §9.4–§9.5]`

### 6.11 M5 neural accelerators

- **Where they are used.** Only through the tensor path of `mul_mm` and `mul_mm_id` (prefill, and verify batches above 8 columns). `mul_mv`, `mul_mv_ext` and flash-attention decode stay on SIMD kernels; a tensor flash-attention kernel is an open PR (#29570).
- **Software gains on M5.** On the same M5 Max, LLaMA-2 7B Q4_0 went from pp512 987 / tg128 103 on the 2023 build to 3,220 / 119.9 on the 2026 build: prefill ×3.3, decode ×1.17.
- **Maintainer status.** ggerganov (2026-03-28): "we did the implementation on M2/M4 and haven't had the chance to optimize it for M5".

`[VERIFIED: R5 §3, §9.1, §13.2]`

### 6.12 Measured on our M2 Pro

| Metric | Qwen3-0.6B Q4_0 | Qwen3-8B Q4_0 | Source |
|---|---|---|---|
| Decode, `llama-bench tg128`, flash attention on (empty cache, no sampling) | 247.5 tok/s (250.2 in a later run) | 34.8 tok/s (35.95 in a later run) | R5 §8.A, §8.F |
| Decode, `llama-completion`, 255 tokens with sampling | — | 34.58 tok/s | R5 §8.C |
| Effective bandwidth on weights | 93 GB/s (51% of 182.5) | 154–159 GB/s (85–87%) | R5 §8.B |
| Prefill pp512 / pp2048 (flash attention on) | 3,849 / 3,117 | 312.5 / 294.1 | R5 §8.A |
| Dispatches / barriers per token | ≈ 564 / ≈ 394 | ≈ 724 / ≈ 542 | R5 §8.D2 |
| Verify cost, n = 2 / 4 / 8 / 16 tokens | 1.22 / 1.53 / 2.49 / 3.36× | 1.52 / 2.08 / 3.76 / 4.29× (Q4_K_M: 1.68 / 2.48 / 4.44 / 4.31×) | R5 §8.G |
| Q4_0 GEMV kernel on an 8B FFN shape (4096 × 14336): n = 1 bandwidth; n = 4 and n = 8 cost | — | 177 GB/s; 2.54×; 4.49× | R5 §8.E |

All rows `[VERIFIED: R5]`.

### 6.13 Published M5-class numbers

| Result | Machine | Source | Label |
|---|---|---|---|
| LLaMA-2 7B Q4_0: pp512 3,220, tg128 119.9 (build `c1d0e7a`) | M5 Max 40-core | discussion #4167 (ggerganov's row) | VERIFIED |
| Same table: M5 Ultra 4,945 / 179.1; M5 Pro 20-core 1,621 / 66.3 | — | #4167 | VERIFIED as posted |
| Qwen3-32B Q4_K_M: tg128 **26.38**, pp512 **667** (b11146, `has tensor = true`) | M5 Max 40-core, 48 GB | mac-llm-bench PR #12 | LIKELY (one community run) |
| qwen35 27B Q4_K_M: pp2048 362.4, tg128 26.60 | M5 Max | #28948 (ggerganov) | VERIFIED |
| Qwen3.6-27B Q4: decode 15.4; pp128 358, pp2048 385. Qwen3-0.6B Q4: decode 386.1; pp128 10,559, pp2048 10,464 | M5 Pro | BaseRT paper, Tables 1–2 | VERIFIED as published |

### 6.14 Strengths

- **Systems engineering.**
  - Mapped, zero-copy weights.
  - A residency heartbeat.
  - Graph reuse.
  - Byte-range hazard tracking with reordering.
  - Parallel encoding with an early first commit.

  The engine's 2023–2026 gains on the same chips came from this system work, not from new GEMV kernels: the same M2 Ultra went from 94.27 to 125.21 tok/s on LLaMA-2 7B Q4_0 (R5 §10).
- **A Q4_0 GEMV at 97% of achievable bandwidth**, unchanged since 2023 (R5 §8.E).
- **Flash attention with split-K decode**, quantized KV inside the kernel, and an offline tuning table per GPU family for the one kernel whose best parameters vary most.
- **The broadest speculative-decoding menu** of any engine on Apple, available in the server.
- **Iteration speed.** One lead maintainer, AI-assisted, with crowd-sourced per-chip tuning (19 SKUs in about two weeks). Apple engineers contribute kernels (#20962). Any gap pointed out publicly is likely to close within weeks (R5 §9.6).

### 6.15 Weaknesses, with evidence

| Weakness | Evidence | Label |
|---|---|---|
| Small-batch kernels | `mul_mv_ext` is issue- and ALU-bound for 4-bit and re-reads weights for 6–8 columns. A 4-token verify costs 2.08×, 8 tokens 3.76× | VERIFIED (R5 §8.E, §8.G) |
| Speculation pays little on Apple | 1.12× at best on the M2 Pro; community reports from 0.67× to 1.47× | VERIFIED |
| A host round trip per token | CPU sampling by default; `waitUntilCompleted` before every sample; 608 KB logits copy per token | VERIFIED (R5 §6) |
| No GEMV-level fusion | q, k, v and gate, up are separate GEMVs; norm, RoPE, KV write and residual are separate kernels: ≈ 20 dispatches and ≈ 15 barriers per layer | VERIFIED (R5 §1.3, §8.D2) |
| K-quant and FP4 GEMVs below the line | Q4_K 147 GB/s, MXFP4 138 GB/s, against 177 for Q4_0. Q4_K_M decodes 13% slower per token than Q4_0 while reading only 5% more | VERIFIED (R5 §8.E, §8.G) |
| Fixed cost for small models | Qwen3-0.6B at 51% of 182.5 | VERIFIED (R5 §8.B) |
| No runtime autotuning | Hand-picked per-type constants, with comments admitting uncertainty; only the flash-attention vector kernel is tuned (offline) | VERIFIED (R5 §2.4) |
| M5 tensor path is young | Slower than SIMD before M5; silently disabled on some M5 builds until 2026-09 | VERIFIED (R5 §3) |
| Bit-identity across batch sizes | The server README warns logits "are not guaranteed to be bit-for-bit identical for different batch sizes" | VERIFIED (R9 §6.1) |

### 6.16 Maintainers' stated positions

| Topic | Statement | Link |
|---|---|---|
| Command replay with indirect command buffers | Graph reuse on Metal "does not lead to any measurable improvement… we already do the CPU/GPU overlap by submitting the first 128 graph nodes… enough to completely mask the CPU overhead"; "`MTLIndirectCommandBuffer` won't cut it." Never merged | #14570 |
| Dependencies and generality | "llama.cpp does not use MPS matrix multiplication as it is too inefficient… Third-party dependencies are highly undesired… a GEMM implementation should make use of the existing dot product kernels… Even if this means sacrificing a bit of performance." (ggerganov, 2023) | #1867 |
| On MLX | "too high level for what we need in ggml. However, the source code has a Metal backend, and we may be able to use it to learn how to better optimize our Metal kernels." (slaren, 2023) | discussion #4345 |
| Universal tuning | "I haven't found a universal solution yet to achieve optimal performance for all batches for all Apple devices and models." (ggerganov, 2024) | #4167 |
| Runtime specialization | Runtime compilation "unlocks… significantly more optimized kernels using the specific shapes of the current computation… through the MTLFunctionConstant mechanism" | #15857 |
| Per-device special cases | Rejected as "would quickly become unmaintainable", in favour of an offline tuning grid: "A table per family sounds good" | #23114, #27668 |
| BF16 | "BF16 does not seem to provide any benefits compared to F16" | #10218 |
| Draft-model speculation | "designed specifically with limited-compute devices in mind" (ggerganov) | #10455 |

`[VERIFIED: R5 §9.5–§9.6]`

The #14570 result is a negative result for the idea that command replay alone speeds up dense decode. R6's measurements explain it: an ICB replay cuts CPU encoding to ≈ 4 µs per token, but the GPU still pays ≈ 1.1–1.25 µs per dependent boundary (R6 §B3–§B4). Sky replays its schedule for other reasons: small drafts, energy, jitter, and GPU-resident sampling ([Part VI](part06-architecture.md) §8.5). It reduces boundaries by fusion, not by replay.

### 6.17 What Sky copies and what Sky avoids

| Copy (our implementation; D15) | Why |
|---|---|
| `mmap` + `newBufferWithBytesNoCopy` with page rounding and overlapping views past `maxBufferLength`; GGUF mapped directly | Zero-copy load (D9) |
| A residency set plus a heartbeat across idle gaps | The 16× first-token penalty after an idle pause (#10119) |
| Byte-range hazard tracking and barriers only on real overlaps; hoisting independent nodes | Fewer barriers, more overlap |
| Split-K decode attention with an offline tuning table per GPU family; mask-block skipping in prefill; quantized KV dequantized inside the kernel | Long context; int8 KV mode |
| Function-constant specialization; parallel library compilation | Fast cold start |
| Breadth of drafters (EAGLE-3, MTP, DFlash, n-gram) as the target menu | D10 |

| Avoid | Why |
|---|---|
| Sampling on the host and a sync per token | ≈ 105 µs of GPU idle per round trip (R6) |
| Re-reading weights per column group in verify | The verify cost curve |
| A fixed fusion table with no GEMV prologue or epilogue | ≈ 20 dispatches per layer |
| Constants picked by hand without measurement | Sky tunes (D11) |
| Using GGUF blocks as the GPU layout without a choice | Sky may repack losslessly (D5); K-quants run at 80% |

---

## 7. The other players

| System | What it is | Best evidence on Apple GPUs | Label |
|---|---|---|---|
| **BaseRT** (Base Compute, July 2026) | C++ runtime with hand-written Metal shaders, a zero-allocation decode loop, low-overhead dispatch, and Metal 4 tensor-op kernels for prefill GEMM, MoE GEMM and prefill attention. Converts checkpoints to its own format at matched quantization, so its comparisons are not same-file | M5 Pro decode vs mlx-lm: 1.02–1.33× on 13 of 15 models, **0.98× on dense Qwen3.6-27B** (18.1 vs 18.4). Qwen3-0.6B 1.33×. Prefill: 1.44× mlx-lm at pp128 on Qwen3.6-27B, **1.02× at pp2048**; up to 3.9× on MoE. Its own conclusion: on decode "BaseRT and MLX both operate close to what the hardware's bandwidth allows" | VERIFIED as published (arXiv 2607.19438; vendor report) |
| **Monolith / mpk-apple** (opened 2026-09-24 by the author of Mirage and MPK) | "The whole generation loop compiled into one GPU-resident static program of fused whole-GPU dispatches, replayed from an indirect command buffer with no CPU work on the critical path." Not a spin-waiting persistent kernel | No results yet (milestone M0). Goals: zero CPU↔GPU syncs per token; plain decode ≥ 1.10× the better of MLX and llama.cpp; stretch ≥ 80% of the bandwidth bound | VERIFIED (github.com/jiazhihao/mpk-apple/issues/1) |
| **Mojo / MAX** (Modular, a Qualcomm company; open-sourced 2026-08-18, Apache-2.0 with LLVM exceptions) | MLIR → LLVM → AIR compiler. Reaches the M5 16×16×16 MMA through the undocumented `llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate`, which accepts fp16, bf16, fp32, fp8 and int8 but not FP4 | W4A16 matmul mix at **0.85–0.90× MLX** as shipped, 0.964× with one slice on `matmul2d`; a measured "~10 TF/s co-issue gap" | VERIFIED (`max/kernels/.../apple/FP4_DESIGN.md`, R7 §5.4) |
| **MTPLX** (open source, built on MLX) | MTP speculation for Qwen3.6/3.8-27B; claims "exact rejection sampling with residual correction" | M5 Max: Qwen3.6-27B 30.37 → 81.74 tok/s "optimized"; Qwen3.8-27B 65.2 "at official Qwen sampling". An independent M3 Max test: +24–28% over mlx-lm (22.17 → 27.5–29.3) | Claims UNVERIFIED; the M3 Max test VERIFIED secondary (R8 §2.7) |
| **MetalRT** (RunAnywhere, March 2026, proprietary) | Small-model decode engine | M4 Max: 1.10–1.19× mlx-lm on 3 of 4 small models; slower than both mlx-lm and llama.cpp on Llama-3.2-3B (184 vs 210 / 222). "Best of 5 runs" | VERIFIED as published (vendor blog; R7 §8.1) |
| **Core ML / Core AI** (Apple) | Core ML: converted models, CPU/GPU/ANE. Core AI (WWDC26, OS 27): PyTorch conversion, ahead-of-time compilation, per-device specialization, in-place KV "states", custom Metal kernels | Core ML's Llama-3.1-8B Int4 showcase: 33.67 tok/s on an M1 Max, ≈ 35% of 400 GB/s. Core AI: ≈ 65 TFLOPS fp16 on the M5 Max GPU (third-party), no int8×int8 path; no LLM decode numbers yet | Core ML VERIFIED numbers, % LIKELY; Core AI EXPERIMENTAL (R6 §A3, R7 §8.1) |
| **ExecuTorch** (PyTorch) | On-device runtime | Deprecated its MPS backend (removal in 1.4); in May 2026 added an **MLX delegate** for Apple GPUs | VERIFIED (R7 §8.3) |
| **MLC-LLM** (TVM Unity / Relax, DLight schedules, MSL emitter) | General ML compiler for LLMs | M2 Ultra study: "MLX achieves the highest sustained generation throughput"; "MLC-LLM delivers consistently lower TTFT for moderate prompt sizes". Secondary sources put it about 17% below MLX on decode | VERIFIED qualitatively (arXiv 2511.05502); gap LIKELY |
| **IREE** | MLIR compiler; Metal through SPIR-V → SPIRV-Cross → MSL | The Apple target declares `mmaCount=0`, so no `simdgroup_matrix` and no neural accelerators. The Metal HAL does not cache GPU programs. No published LLM results on Apple | VERIFIED (`KnownTargets.cpp`, R7 §5.2) |
| **PyTorch MPS / Inductor** | Eager MPS kernels; `torch.compile` on MPS | Inductor-MPS is an "early prototype ... likely to fail" for whole networks. PyTorch 2.13 moved off MPSGraph ("adds compilation and scheduling overhead per dispatch") to native Metal, with hand-written FlexAttention kernels. Some MPS GEMM shapes beat MLX by up to 1.18× (mlx#3196) | VERIFIED (R7 §4b, R3-issues B6–B7) |
| **Metal FlashAttention** (Philip Turner) | JIT-compiled attention kernels | 83% of ALU on an M1 Max. Register-centric tiling, a parameter table per head dimension, undocumented `air.simdgroup_async_copy` through `__asm` (helps on M1/M2, hurts on Apple9+) | VERIFIED (R7 §8.2) |
| **ThunderMittens** (Hazy Research, Dec 2024) | ThunderKittens ported to Metal with 8×8 tiles | M2 Pro: GEMM ≈ 9% faster than MLX; attention within ±15% of MLX | VERIFIED (R7 §8.1) |
| **triton-ext** Apple backend | Out-of-tree Triton → MSL | Minimal backend merged 2026-09-14; abandoned its AIR path as undocumented; no comparison with MLX | VERIFIED (R7 §1) |
| **jax-metal** | StableHLO → MPSGraph plugin (closed source) | Abandoned; last release October 2024 | VERIFIED (R7 §4a) |

Lessons:
1. **Hand-written runtimes tie MLX on dense decode and win on fixed cost.** BaseRT gets 0.98× on 27B and 1.33× on 0.6B. MetalRT's wins are on small models only.
2. **Generic compilers trail.**
   - MLC-LLM is behind on throughput.
   - IREE has no matrix units.
   - Inductor-MPS is a prototype.
   - Mojo is at 0.85–0.96× even with undocumented AIR.
   - This agrees with tinygrad's 11% (§5) and with SPINE D2 and D4.
3. **The openings others chase are the ones §12 lists.** BaseRT's lead is short-prompt prefill. Monolith's is removing the CPU and fusing dispatches. MTPLX's is speculation with exact rejection sampling on a hybrid MTP model. Sky is not alone on any of them. What no competitor offers is the combination of same-file comparisons, exactness under speculation, and a compiler-owned static plan ([Part XXVII](part27-moat.md)).
4. **Apple's own high-level stack is not the frontier for this workload yet.**
   - Core ML reached ≈ 35% of bandwidth.
   - ExecuTorch now delegates to MLX.
   - Core AI is new and unmeasured. It is the competitor most likely to change this picture on OS 27 ([Part XXVIII](part28-risks.md)).

---

## 8. Design philosophy: llama.cpp vs MLX vs tinygrad vs Sky

The Sky column restates decisions D1–D18 of the spine. The first three columns are as measured or read in §3–§6.

| Aspect | llama.cpp / ggml | MLX + mlx-lm | tinygrad | Sky |
|---|---|---|---|---|
| **Graph construction** | C++ builds a static graph per micro-batch shape; reused across tokens while the topology holds (KV view padded to 256) | Python builds a lazy graph every token (1,061 nodes for 8B); nothing is reused | Python builds a lazy UOp graph; TinyJit captures it once and replays | Built ahead of time per (model, max context, batch, chip) from `sky-llm` builders or Sky source into TIR; no per-token graph work (D1, D7, D9) |
| **Laziness** | Eager: build, then run at once; asynchronous submit | Fully lazy; `eval` / `async_eval` | Fully lazy; realize or JIT | None at run time. Shapes and formats are types; only position and context bucket vary (D8, D9) |
| **Fusion** | Fixed table of 9 small-op patterns; no GEMV prologue or epilogue; no QKV or gate/up merge | `mx.compile` fuses elementwise ops only (≤ 11 deep); hand-fused `mx.fast` kernels; mlx-lm compiles only SwiGLU | Rangeify fuses elementwise and reduction ops by range agreement; recompute-vs-store by a 3-rule test | Cost-model fusion partitioning around skeletons, with generated prologues and epilogues (norm → GEMV, residual epilogue, RoPE + KV append); ≤ 6 kernels per layer at G4 (D4, D11) |
| **Kernel source** | 100% hand-written MSL, specialized by function constants, compiled at run time from embedded source | Hand-written MSL templates precompiled into `mlx.metallib`; JIT for `mx.compile`, gather/scatter and custom kernels | Generated MSL compiled through the private `MTLCompiler` service | Hand-written skeletons (later in L3 kernel IR) plus generated fusion code; MSL text through `makeLibrary(source:)`; pipelines cached on disk (D3, D4) |
| **Autotuning** | None at run time; an offline table per GPU family for flash-attention decode only | None; heuristics keyed by device letter | BEAM, opt-in; no cache flush on Metal | Three levels on `sky-tune`: kernel BEAM, block fusion partition, model level; keyed by device fingerprint and OS (D11) |
| **Quantization formats** | GGUF blocks: Q4_0…Q8_0, K-quants, IQ codebooks, MXFP4, NVFP4, TQ | Affine 2–8 bits, groups 32/64/128, separate scale and bias; mxfp4, nvfp4, mxfp8 | GGUF blocks as tensor expressions fused into consumers | MLX affine and GGUF both loaded natively; lossless repack allowed, re-quantization never; same-file comparisons (D5) |
| **Loading** | `mmap` + no-copy buffers; overlapping views | `pread` into new buffers | Disk read, then copy to device buffers | Page-aligned relayout cache (`.skyweights`) mapped no-copy for MLX files; GGUF mapped directly; build time counted in cold start (D9) |
| **Memory planning** | Static plan per graph (ggml-alloc); KV and compute buffers preallocated | Caching allocator and residency sets; KV grown by 256-token concatenation | TLSF arena per JIT (457 buffers → 2) | Static plan per (model, max context, batch, chip); arena offsets; zero allocation in the hot path; KV preallocated (D9) |
| **CPU overhead per token** | Graph reuse; re-encodes ≈ 724 dispatches into 2 command buffers; host sync and CPU sampling every token | Python build + encode, hidden by `async_eval` for 8B and up; 1.94 ms ≈ GPU time on 0.6B | ≈ 155 µs JIT round trip + 4.3 µs per kernel; `.item()` every token | GPU-resident loop: the GPU writes the next token and position; stop checks on the GPU; the static plan directly re-encoded each step (ICBs only where the GPU chooses the command count); K steps in flight, one command buffer per step (D9, D18) |
| **Attention** | Flash attention: tiled prefill; decode split-K over 32 workgroups + reduce; offline-tuned per family | `sdpa_vector` 1- and 2-pass; GQA-deduplicated only at ≥ 8K and G ∈ {8, 12, 16}; fused only for qL·G ≤ 32; Steel or NAX prefill attention | Composed from tensor ops; no flash attention on Metal | GQA-aware flash-decoding skeleton with a fixed split size; flash-attention prefill; neural-accelerator path on M5 (D4, D6) |
| **KV** | Preallocated cells; f16 default; quantized types read inside the kernel | Grown in 256-token steps; quantized KV unfused and slower at short context | Preallocated fp16 tensor; symbolic `start_pos` | Preallocated in the memory plan; fp16 default; int8 a labeled mode; exact rollback for speculation (D9, D10) |
| **Sampling** | CPU chain by default; optional GPU samplers; the token returns to the host | GPU, inside the lazy graph; global RNG state | GPU Gumbel-max inside the JIT; host read per token | GPU sampling with a counter-based RNG (Philox) and keyed Gumbel noise; CPU detokenizes asynchronously (D9, D10) |
| **Speculation** | Draft model, EAGLE-3, MTP, DFlash, DSpark, 5 n-gram modes; sample-and-match acceptance | Draft model only; sample-and-match acceptance; one blocking sync per round | None | Batch-invariant verify GEMV first; EAGLE-3 public weights, draft models, prompt lookup and suffix drafting; trees sized to k\*; rejection sampling or keyed Gumbel; byte-identical to plain decode (D10) |
| **Backends** | CPU, Metal, CUDA, HIP, Vulkan, SYCL, OpenCL, CANN, RPC, … | Metal, CUDA, CPU | Metal, NV/CUDA, AMD, CPU, OpenCL, WebGPU, QCOM, DSP, … | Metal (MSL) plus a CPU reference interpreter as the oracle; no MLIR (D1, D2) |
| **Determinism / batch invariance** | Logits "not guaranteed to be bit-for-bit identical for different batch sizes" | Verify and sequential forwards differ (#1470), documented (#1592); temperature-0 output differed across chips (#1280) | No mechanism found in the code read (LIKELY) | `--numerics=inference`: bit-reproducible per (chip, OS, build, tuning entry); batch- and chunk-invariant by construction (D6; hybrid DeltaNet prefill excepted, VII-M7) |

---

## 9. Head-to-head on our M2 Pro

Same machine, same lock, same Qwen3 models. The files differ: MLX 4-bit for mlx-lm, Q4_0 GGUF for llama.cpp and tinygrad. Bytes are therefore shown with every speed, and efficiency is computed per byte of weights streamed. KV reads add ≤ 1% for 8B at these contexts and ≈ 2–9% for 0.6B.

### 9.1 Decode

| Engine | Model / file | Weights per token | tok/s | ms per token | GB/s | % of 182.5 | % of 196 | Dispatches / barriers per token | Method |
|---|---|---|---|---|---|---|---|---|---|
| mlx-lm | 0.6B, MLX 4-bit | 335.4 MB | **267.4** | 3.74 | 89.7 | 49% | 46% | 541 / 362 | R4 external timestamps, 128 + 256 tokens |
| llama.cpp | 0.6B, Q4_0 GGUF | 376.2 MB | 247.5 | 4.04 | 93.1 | 51% | 48% | ≈ 564 / ≈ 394 | `llama-bench tg128`, flash attention on |
| tinygrad | 0.6B, the same Q4_0 GGUF | ≈ 392 MB (R2's estimate) | 54.3 | 18.41 | ≈ 21 | ≈ 12% | ≈ 11% | 488 kernels, a barrier before each | `tinygrad.llm --benchmark` |
| mlx-lm | 8B, MLX 4-bit | 4,257.7 MB | **37.4** (38.8 steady; 39.2 at context 128) | 26.7 (25.8) | 159.4 (165.1) | 87% (90%) | 81% (84%) | 693 / 457 | R4 |
| llama.cpp | 8B, Q4_0 GGUF | 4,431.3 MB | 34.8 (36.0 in a later run; 34.6 with sampling) | 28.7 (27.8) | 154.3 (159.3) | 85% (87%) | 79% (81%) | ≈ 724 / ≈ 542 | R5 |
| tinygrad | 8B | — | not measured | — | — | — | — | — | R2 §10: ≈ 11 tok/s expected (LIKELY) |

`[VERIFIED: R2 §8.3, R4 §7, R5 §8; arithmetic ours]`

Reading:
- **8B.** mlx-lm decodes 7–8% faster in matched runs. About 4 points of that is its smaller file. The other 3–4 points is a higher rate per byte, close to the ≈ 3% spread R5 saw between repeated runs. Both engines run at 79–84% of the best raw read.
- **0.6B.** Both engines are bound by per-token fixed cost at about half the bandwidth line. mlx-lm is faster per token because it reads 11% fewer bytes (no Q6_K head). The fixed cost is about 1.6–1.8 ms per token in both:
  - llama.cpp: a two-point fit (R5);
  - mlx-lm: the dispatch ablation plus GEMV shortfall, 1.58 ms (R4).
- **tinygrad** is a factor of 4–5 behind on the same file. Its own fp16 path is faster than its 4-bit path.

### 9.2 Prefill

| Engine | 0.6B pp512 | 0.6B pp2048 | 8B pp512 | 8B pp2048 | 8B pp512 as TFLOPS (15.1 GFLOP/token) | Method |
|---|---|---|---|---|---|---|
| mlx-lm | 2,714 | 2,635 | 219.5 | 217.7 | 3.3 (58% of 5.73) | Prompt length ÷ TTFT; includes one decode step (≈ 1–2%) |
| llama.cpp (flash attention on) | 3,849 | 3,117 | 312.5 | 294.1 | 4.7 (83%) | `llama-bench pp`, random tokens |
| **llama.cpp ÷ mlx-lm** | **1.42×** | **1.18×** | **1.42×** | **1.35×** | | |
| tinygrad | not measured; the first prefill call took 7.6–10 s, mostly compilation | | | | | R2 §8.3 |

`[VERIFIED: R4 §7.A, R5 §8.A; arithmetic ours]`

At the kernel level the two engines' quantized GEMMs are close:
- MLX `qmm_t`: 4.40 TFLOPS on 32B MLP shapes (R3 §7.D).
- llama.cpp `mul_mm` Q4_0: 4.28 TFLOPS on an 8B FFN shape (R5 §8.E).

So most of mlx-lm's end-to-end gap lies outside the large GEMMs, as [Part IV](part04-hardware-limits.md) §5 also concludes. Candidates:
- bf16 activations;
- k/v projection shapes;
- attention, norm and RoPE kernels;
- per-chunk evaluation.

`[EXPERIMENTAL: V-T7 decides]`

### 9.3 Long context

| Engine | Evidence on the M2 Pro |
|---|---|
| mlx-lm | 8B: 39.2 → 31.7 tok/s from 128 to 4K context; the extra KV stream runs at ≈ 97 GB/s. 0.6B: 267 → 116 tok/s at 8K; KV at ≈ 189 GB/s (R4 §7.A) |
| llama.cpp | Not measured at depth on this machine (R5 §13.2, open question 5) |
| tinygrad | Not measured |

### 9.4 Verify cost and speculation

Cost of one forward over M tokens, relative to M = 1 (Qwen3-8B, whole model):

| M | 2 | 3 | 4 | 5 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| mlx-lm (MLX 4-bit) | 1.73 | 2.52 | 3.33 | 4.10 | 6.16 | 6.48 | 6.61 |
| llama.cpp Q4_0 | 1.52 | 1.66 | 2.08 | 2.43 | 2.96 | 3.76 | 4.29 |
| llama.cpp Q4_K_M | 1.68 | 2.43 | 2.48 | 2.99 | 3.59 | 4.44 | 4.31 |
| Bandwidth-bound ideal (k\* ≈ 8 on the M2 Pro) | ≈ 1.0–1.1 | | | | | ≈ 1.1–1.3 | ≈ 2 |

`[VERIFIED: R4 §7.D, R5 §8.G; ideal LIKELY: R8 §2.2, Part IV §8]`

| Speculative setup (8B target, 0.6B draft, greedy) | Best result | Worst result | Exact? |
|---|---|---|---|
| mlx-lm | 0.74× (code, k = 2) | 0.37× (chat, k = 6) | No: 3 of 9 configurations diverged |
| llama.cpp | 1.12× (3 drafts) | 0.95× (7 drafts) | Not tested by R5 |

On this generation-14 GPU, MLX's small-M routing is at its worst: `qmv_wide` is disabled for affine weights by design, and M ≥ 6 pays a 32-row tile. The mlx-lm row therefore overstates MLX's verify cost on M3–M5 hardware. The M5 Max data point (+48% at M = 5, +125% at M = 8, mlx#3791) is the fair comparison for the headline.

### 9.5 What these tables do not show

- Energy per token. It was not measured for llama.cpp or tinygrad. For mlx-lm on the 0.6B, R9 found 7.1 W of CPU from the Python loop.
- llama.cpp at long context; tinygrad at 8B.
- Any M5 behaviour: the neural accelerators and `qmv_wide` are off on this GPU.
- Identical methods. llama-bench decodes from an empty cache without sampling, which favours it slightly. mlx-lm's numbers include sampling and context growth.

The harness of [Part XXIII](part23-benchmarks.md) re-measures every engine with one method at gate G0.

---

## 10. Published M5-class numbers, side by side

| Engine | Machine | Model, quantization | Decode | Prefill | Source | Label |
|---|---|---|---|---|---|---|
| MLX 0.32.2 / mlx-lm 0.31.3 | M5 Max 40-core | Qwen3-32B, MLX 4-bit (17.99 GB/token) | **28.41** tok/s (83% of 614 per byte) | pp512 **730.9** (≈ 46 TFLOPS at 63.5 GFLOP/token) | mac-llm-bench PR #12, 2026-09-27 | LIKELY (one run) |
| llama.cpp b11146 | same machine | Qwen3-32B, Q4_K_M (19.32 GB/token) | **26.38** (83% of 614 per byte) | pp512 **667** | same | LIKELY |
| llama.cpp | M5 Max | qwen35 27B, Q4_K_M | 26.60 | pp2048 362.4 | #28948 (ggerganov) | VERIFIED |
| llama.cpp `c1d0e7a` | M5 Max 40-core | LLaMA-2 7B, Q4_0 | 119.9 | pp512 3,220 | #4167 | VERIFIED |
| mlx-lm / llama.cpp / BaseRT | M5 Pro | Qwen3.6-27B (hybrid), 4-bit | 18.4 / 15.4 / 18.1 | pp128 305 / 358 / 440; pp2048 486 / 385 / 497 | BaseRT paper | VERIFIED as published |
| mlx-lm / llama.cpp / BaseRT | M5 Pro | Qwen3-0.6B, 4-bit | 398.1 / 386.1 / 530.8 | pp128 4,600 / 10,559 / 11,873; pp2048 16,198 / 10,464 / 20,105 | same | same |
| MLX | M5 Max | 70B dense 4-bit, whole forward | M = 1: 75.9 ms; M = 8: 170.8 ms | — | mlx#3791 | contributor |
| MLX | M5 Max | fp16 GEMM | — | 52.7–59.6 TFLOPS | mlx#3211 | VERIFIED |
| Mojo / MAX | M5 | W4A16 matmul mix | — | 0.85–0.96× MLX | FP4_DESIGN.md | VERIFIED |
| MTPLX (on MLX) | M5 Max | Qwen3.6-27B 4-bit + MTP | 30.37 plain; 81.74 "optimized" | — | README | UNVERIFIED |
| tinygrad | — | — | no M5 numbers; no M5 path | — | R2 §10 | — |

Per byte, MLX and llama.cpp are equally efficient on the one 32B run that measured both (83.2% vs 83.0% of 614 GB/s, weights only). MLX's 7.7% lead equals llama.cpp's 7.4% larger file. On short prompts MLX trails llama.cpp on the M5 Pro (pp128). It leads from pp512 on the 27B, and from pp2048 on the 0.6B.

---

## 11. Where each engine is already at the hardware limit

These are negative results for Sky. None of them is a place to look for a headline.

| Area | Who is at the limit | Evidence | Consequence for Sky |
|---|---|---|---|
| Large-matrix decode GEMV, M = 1 | MLX and llama.cpp | MLX 91–97% of 182.5 GB/s on 32B shapes; a 60-line kernel only matched it. llama.cpp Q4_0 at 177 GB/s (97%). Raw Metal reads 196 GB/s and a simple GEMV 183–191 GB/s | At most a few percent from GEMV kernel work. Sky's G1 bar is parity, not a win (R3 §7.B, §7.G; R5 §8.E; R6 §B2) |
| Dense short-context decode, whole step | MLX/mlx-lm; llama.cpp within a few percent | mlx-lm 8B at 91% of 182.5 (84% of 196); its GEMV chain alone at 97.5%. On M5 Max both engines at ≈ 83% of spec per byte | 1.05–1.14× is the honest range for fusion and scheduling ([Part IV](part04-hardware-limits.md) §4); R4's upper bound on 8B is 1.08× |
| Command replay as such | llama.cpp (measured), MLX (commit policy) | #14570: no gain from Metal graph or ICB reuse. MLX: one command buffer per token changed nothing at real sizes. R6: ICB replay leaves GPU per-dispatch cost unchanged | Replay serves drafts, energy and GPU-resident sampling, not dense decode |
| Megakernels for a dense 32B | MLX's position; Monolith's goal | A device-wide barrier costs 1.1–1.3 µs, the same as a kernel boundary (R6 §B6). Monolith targets only 1.10× | The megakernel is rejected (R6; SPINE §2); only a G9 feasibility spike remains ([Part XX](part20-persistent-execution.md) §8) |
| Large prefill GEMMs on M5 | MLX | fp16 GEMM 52.7–59.6 TFLOPS against a ≈ 60–65 TFLOPS neural-accelerator estimate. BaseRT gained 2% at pp2048; Mojo trails MLX | Long-prompt prefill on M5 is parity work; INT8 is the only route past it (a labeled mode) |
| Decode on the neural accelerators | Everyone | A one-row `matmul2d` is slower than a GEMV; Apple says decode gains on M5 come from bandwidth | Decode GEMV stays on SIMD; the accelerators serve prefill and wide verification |
| Graph reordering for overlap | llama.cpp | Reorder pass ≈ 0% on Qwen3 decode | Barrier minimization matters only after fusion |
| Attention at GQA 2 | MLX | Qwen3-0.6B's KV stream at ≈ 189 GB/s up to 8K | The long-context opening is specific to G ≥ 4 |
| n-gram drafting on fresh text | llama.cpp | `ngram-simple` found 0 drafts in 256 tokens of new code | Prompt lookup pays only on input-grounded work |

---

## 12. Where measurable headroom exists

| # | Opening | Evidence | Size | Weak engines | Sky part |
|---|---|---|---|---|---|
| 1 | **Verify cost at M = 2–16** | M2 Pro whole forward: mlx-lm 3.33× at 4 and 6.48× at 8; llama.cpp 2.08× and 3.76×. M5 Max MLX: +48% at 5 and +125% at 8 (70B). A 60-line kernel reached 1.30× at M = 3 on the M2 Pro | Decides whether speculation gives ≤ 1× or ≈ 2× (R8 §2.9) | MLX, llama.cpp | [XVI](part16-decode.md), [XXI](part21-speculative.md) |
| 2 | **Exact speculation** | mlx-lm diverges (#1470; 3 of 9 of our runs). Both engines use sample-and-match acceptance (Σp·q, not Σmin(p, q)) | A product property, and higher acceptance at T > 0 | MLX, llama.cpp | [XXI](part21-speculative.md) |
| 3 | **The speculative loop** | mlx-lm: a full sync per round and no cross-round overlap. llama.cpp: graph rebuilds when verify sizes vary (75 of ≈ 150 calls reused a graph) and a host round trip per call | ≈ 0.6 decode-units per cycle in llama.cpp's accounting (R5 §8.G) | mlx-lm, llama.cpp | [XX](part20-persistent-execution.md), [XXI](part21-speculative.md) |
| 4 | **GQA decode attention, 1K–8K and beyond** | 8B KV stream at ≈ 97 GB/s (53%); 56% of peak at 8 KV heads on the M5 Max (mlx#3837); llama.cpp 2× faster than MLX at 146K (mlx-lm#763) | 19% of the 8B token at 4K; about a third of decode time at 32K for 32B ([Part IV](part04-hardware-limits.md) §4) | MLX | [XV](part15-attention.md) |
| 5 | **Verify attention for 5–48 queries** | MLX unfused for qL·G > 32; BQ = 64 plateau from 9 queries; won't-fix (#3826) | Unmeasured on 32B | MLX | [XV](part15-attention.md) |
| 6 | **Fixed per-token cost for small models** (the draft) | 0.6B at 49–54.5% of 182.5 in mlx-lm, 51% in llama.cpp; 31% of the mlx-lm token is non-matmul dispatch; CPU ≈ GPU time; BaseRT 1.33× mlx-lm on Qwen3-0.6B (M5 Pro) | 1.3–1.8× on 0.6B; sets the draft cost c | MLX, llama.cpp, tinygrad | [XVI](part16-decode.md), [XX](part20-persistent-execution.md) |
| 7 | **Fusion of the 19 kernels per layer** | 7.6% of the mlx-lm 8B token; fusion off costs llama.cpp 4.5% on 8B | 1.05–1.08× on 8B; less on 32B | MLX, llama.cpp | [IX](part09-scheduler.md) |
| 8 | **Short-prompt prefill and warm TTFT** | BaseRT 1.44× mlx-lm at pp128 on 27B; mlx-lm 0.44× of llama.cpp at pp128 on 0.6B (M5 Pro) | 1.2–1.5× at ≤ 512 tokens | mlx-lm | [XVII](part17-prefill.md), [XVIII](part18-ttft.md) |
| 9 | **Prefill on M1–M4** | llama.cpp 1.35–1.42× mlx-lm on Qwen3-8B; mlx-lm at 58% of ALU peak; MPP `matmul2d` reaches 95% of peak on the M2 Pro (R6 §B5) | 1.25–1.4× over mlx-lm | mlx-lm | [XVII](part17-prefill.md) |
| 10 | **Cold start and load** | MLX `pread`s every byte and declines mmap; llama.cpp's first shader compile took 4.8 s; tinygrad loaded at 0.15–0.32 GB/s | ≥ 3× cold TTFT (SPINE §3) | MLX, tinygrad; llama.cpp after install | [XVIII](part18-ttft.md), [XIX](part19-memory.md) |
| 11 | **Quantized KV** | MLX: no fused kernel; 6–22% slower decode at short context (#1651); higher peak memory (#1587) | 8-bit KV ≈ 1.05× at 8K, 1.18× at 32K, if fused (R8 §5.3) | MLX | [XIV](part14-kv-cache.md) |
| 12 | **Memory transients** | mlx-lm: KV growth by concatenation; 2× KV on prompt-cache reuse (#1395) | Peak RAM; out-of-memory on 16 GB machines | mlx-lm | [XIX](part19-memory.md) |
| 13 | **Host round trip per token** | llama.cpp syncs and samples on the CPU every token; ≈ 105 µs of GPU idle per wait (R6) | ≈ 0.3–0.5% on 32B; more on drafts | llama.cpp, tinygrad | [XX](part20-persistent-execution.md) |
| 14 | **K-quant decode efficiency** (same-file GGUF pairs) | Q4_K 147 GB/s and MXFP4 138 GB/s against Q4_0 177 | ≈ 1.15–1.2× on Q4_K_M files | llama.cpp | [XII](part12-quantization.md), [XVI](part16-decode.md) |
| 15 | **Energy** | mlx-lm's Python loop drew 7.1 W of CPU during 0.6B decode; nobody publishes J/token | ≤ 0.85× mlx-lm (SPINE target) | mlx-lm | [XX](part20-persistent-execution.md), [XXIII](part23-benchmarks.md) |

---

## 13. Disagreement with the spine

None of the following changes a decision. They are corrections of numbers and wording that the evidence supports.

1. **Plain-decode headroom.** SPINE §2 says "only about 3–6% headroom", and G1 asks for "≥ 88% of measured bandwidth".
   - The 94–97% behind the 3–6% is relative to MLX's own 182 GB/s streaming ceiling, measured on partial layer stacks (R3 §7.F).
   - The whole Qwen3-8B model runs at 91% of 182.5 and 84% of the 196 GB/s raw-Metal read (R4 §7.B, R6 §B2). R4's GEMV-only upper bound is 1.08×.
   - The honest range is the 1.05–1.14× of [Part IV](part04-hardware-limits.md) §4.
   - Proposal: state G1's denominator as the measured raw read (196 GB/s on the M2 Pro), not MLX's streaming ceiling.
   - Resolved: the spine now says plain short-context headroom is about 1.05–1.14× (§2), and G1 measures against the raw-Metal read bandwidth (196 GB/s on the M2 Pro).
2. **Prefill on M1–M4.** SPINE §2 frames the non-NAX opening as the kernel-level 11–13% gap and sets "≥ 1.1× on M1–M4". G4 asks for "prefill ≥ 0.9× mlx-lm on the M2 Pro".
   - llama.cpp already prefills Qwen3-8B 1.35–1.42× faster than mlx-lm on the same machine (§9.2).
   - G4's prefill bar is therefore below an existing competitor.
   - Proposal, consistent with Part IV §11: on M1–M4, prefill ≥ 1.25× mlx-lm on the MLX file and ≥ 1.0× llama.cpp on the GGUF file.
   - Resolved: the spine now says (§3, G5) prefill on M1–M4 ≥ 1.25× mlx-lm **and** ≥ 1.0× llama.cpp, and G4 asks for prefill ≥ 1.0× mlx-lm on the M2 Pro.
3. **Scale dtype.** SPINE §1 describes the MLX 4-bit format as having "fp16 scales and biases". The mlx-community files store bf16 scales and biases (R4 §0, R9 §1.4), as D5 says. Bits per weight are unchanged (4.50). This is a wording fix. Resolved: the spine now says bf16 scales and biases (§1), and the descriptor records the scale dtype because MLX's format also allows fp16.
4. **MLX's share of the M5 Max bandwidth.** SPINE §2 gives "≈ 85% of the 614 GB/s spec".
   - Counting the 17.99 GB actually streamed per token, it is 83.2% (83.9% with the benchmark's KV reads).
   - 85% results from using the 18.43 GB file size, which includes the embedding.
   - llama.cpp's run is 83.0% on the same basis.
   - Resolved: the spine now says ≈ 83% of the 614 GB/s spec (17.99 GB per token).
5. **Size of the attention gap.** SPINE §2 says MLX is "about 3× off bandwidth" at 4K.
   - That figure is R3's 8-layer stack.
   - The whole Qwen3-8B model measures ≈ 2× (97 GB/s). The 0.6B, with GQA 2, is at the line (R4 §7.A).
   - The gap scales with the GQA factor. For Qwen3-32B (GQA 8), 3× remains a LIKELY extrapolation.
   - The ≥ 1.2× long-context target is unaffected.
   - Resolved: the spine now says ≈ 2× end-to-end on 8B (≈ 97 GB/s, R4) and up to 3× only in R3's microbenchmark; the long-context targets are now ≥ 1.05× at 8K, ≥ 1.10× at 16K and ≥ 1.2× at 32K (bounds ≈ 1.08× / 1.14× / 1.24×).
6. **Scope of NAX.** SPINE §2 says MLX's neural-accelerator path "runs prefill GEMM and prefill attention only". It also runs MoE gather GEMMs and the Gated-DeltaNet update for T > 8 (R3-kernels §d). This matters for D10: MLX already has a fused, accelerated DeltaNet kernel for the hybrid 27B models.
7. **Speculative baseline.** SPINE §3 sets "≥ 1.3× mlx-lm's own speculative mode". On M1/M2-class GPUs mlx-lm's speculation is a net loss (0.37–0.74×) and llama.cpp's is ≤ 1.12×, so the target is trivially met there. It must be judged on M5-class hardware only. The spine places it on the M5 Max already; this is a clarification for G6 reviewers. Resolved: the spine now sets ≥ 1.2× (not 1.3×) mlx-lm's own speculative mode with the same draft, on the M5 Max, where mlx-lm's verify cost is β ≈ 0.06–0.18 (mlx#3791) against ≈ 0.5–1 on M1/M2-class GPUs; byte-identical output is the differentiator.

---

## 14. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| V-T1 | Competitor pins and adapters, extending the existing `sky-bench compare` pattern | `benches/compare/llm/pins.toml` (MLX, mlx-lm, llama.cpp, tinygrad commits; model repository commits and sha256); adapters `benches/compare/llm/{mlx_lm.py, llama_cpp.sh, tinygrad.py}`; driver `crates/sky-bench/src/compare/llm.rs` | One command that reruns §9's tables with the protocol of [Part XXIII](part23-benchmarks.md) and writes R9 §8 JSON | Reproduces every §9 number on the dev M2 Pro within its original run spread (≤ 3%); refuses to run on a commit or hash mismatch |
| V-T2 | Metal call census for any engine | `tools/metal-census/census.m`: an encoder-method interposer in the style of R4's `r4count.m` (dispatches, barriers, encoders, commits, ICB executions with their command counts); `sky explain --census` reads the same counters | Dispatches, barriers, encoders and command buffers per token for mlx-lm, llama.cpp, tinygrad and Sky | Reproduces 693 / 457 / 23 (mlx-lm 8B), ≈ 724 / ≈ 542 (llama.cpp 8B), 488 ICB commands (tinygrad 0.6B); Sky's G4 plan shows ≤ 6 dispatches per layer |
| V-T3 | Exactness adversarial suite | `crates/sky-llm/tests/spec_exactness.rs`; `benches/compare/llm/mlx_lm_1470.py` (the #1470 reproduction) | Prompts × draft lengths 1–16 × temperatures {0, 0.6, 1.0}: Sky speculative vs Sky plain; mlx-lm's divergence index recorded | Sky 100% byte-identical (a CI gate, D10). mlx-lm results archived, not asserted |
| V-T4 | Verify-cost curves for every engine | `sky bench llm --verify-curve` (`crates/sky-cli`); MLX adapter (batched `quantized_matmul` as in R3 §7.C, whole forward as in R4 §7.D); llama.cpp adapter (`llama-bench -p 1,…,16 -n 0`) | time(M)/time(1) for M = 1–32, kernel and whole forward, Qwen3-8B/32B shapes, M2 Pro and M5 Max | Reproduces the R3/R4/R5 curves on the M2 Pro; Sky ≤ 1.3× at M ≤ 8 (SPINE §3) |
| V-T5 | Routing-cliff sweep | `benches/compare/llm/cliffs.py` | Both sides of each MLX threshold in §3: `vector_limit` (6 / 13), qL·G ≤ 32, 2-pass at kL ≥ 1024, GQA deduplication at kL ≥ 8192, split-K tile counts, `qmm_nax` K % 64; measured for MLX and Sky | Sky has no step > 1.3× between adjacent M values or context buckets |
| V-T6 | Long-context decode comparison | Adapter options `--depth` for all engines (`llama-bench -d`) | Inter-token latency at contexts 128 … 32K and the effective KV stream (GB/s) by GQA factor: 0.6B G = 2, 8B G = 4, 32B G = 8 | Sky's KV stream ≥ 85% of the measured raw read for G ∈ {2, 4, 8} at ≥ 4K; llama.cpp's M2 Pro numbers filled into §9.3 |
| V-T7 | Explain the M1–M4 prefill gap | Profiles with GPU timestamps at encoder boundaries and Metal System Trace; `benches/compare/llm/prefill_breakdown.py` | Per-op GPU time of mlx-lm and llama.cpp prefill at 512 and 2,048 tokens for 8B: bf16 vs fp16 activations, attention, norms, k/v projections, chunk evaluation | The breakdown accounts for ≥ 90% of the measured 1.35–1.42× difference; it fixes Sky's M1–M4 prefill target (§13 item 2) |
| V-T8 | Cold start and peak RAM, all engines | Harness from [Part XXIII](part23-benchmarks.md) | Process start → first token with the page cache warm and cold: mlx-lm (`pread`), llama.cpp (`mmap`; first compile after install), Sky (`.skyweights`); `peak_ram_bytes` per R9 §6.2 | Null-engine noise ≤ 300 MB; llama.cpp's mapped weights counted through system wired memory; Sky's cold TTFT ≥ 3× better than mlx-lm |
| V-T9 | Energy per token, all engines | Same harness, SMC `PSTR`, ≥ 60 s loops | J/token for mlx-lm, llama.cpp and Sky, decode and prefill | Error ≤ 7% (R9 §6.3); Sky ≤ 0.85× mlx-lm plain decode |
| V-T10 | Competitor watch | Weekly job on the self-hosted macOS runner (D14) running V-T1, V-T4 and V-T6 at llama.cpp master and MLX / mlx-lm main | A dated JSON series; an issue opened when a tracked gap moves by more than 5% | The job runs for 4 consecutive weeks with archived results |
| V-T11 | Reproduce third-party M5 claims | Adapters from V-T1 on the M5 Max host | The community Qwen3-32B run (MLX 28.41 / 730.9; llama.cpp 26.38 / 667), BaseRT's MLX columns where the models overlap, MTPLX's 30.37 / 81.74 on Qwen3.6-27B | Each claim reproduced within 5%, or recorded as not reproducible together with the exact commands. Complements [Part IV](part04-hardware-limits.md) IV-T3 |

---

## 15. Measurements that decide

1. **Bytes-normalized decode efficiency on the M5 Max**, for MLX and llama.cpp on Qwen3-32B, against a raw-Metal read on the same chip. This decides whether plain decode has 0% or about 10% headroom (§11, §13 item 1).
2. **MLX's verify curve on the M5 Max** at Qwen3-32B shapes, M = 1–16, kernel and whole forward. This decides how much of opening 1 survives `qmv_wide` (β in [Part IV](part04-hardware-limits.md) §8).
3. **mlx-lm's speculative mode on the M5 Max** with 0.6B and 1.7B drafts: speed and divergence rate. This sets the "≥ 1.2× mlx-lm's own speculative mode" baseline.
4. **MLX decode attention at 8K, 16K and 32K on the M5 Max** for GQA 8, with and without the mlx#4077 kernel. This decides the long-context targets (≥ 1.05× at 8K, ≥ 1.10× at 16K, ≥ 1.2× at 32K).
5. **Prefill at pp128–pp512 on the M5 Max** for MLX, llama.cpp and a BaseRT-class reference. This decides the warm-TTFT target.
6. **The per-dispatch GPU gap on the M5 Max.** This decides how much fusion is worth at 32B.
7. **The cause of the M1–M4 prefill gap** (V-T7).
8. **Monolith's first results.** If ICB-replayed fused dispatches reach ≥ 1.10×, Sky's D9 runtime is validated, but it is not unique.
9. **MTPLX reproduction.** This sets the bar for the hybrid 27B + MTP headline (D10).

---

## 16. Risks and negative results

- **The openings may close before Sky ships.**
  - MLX closed small-M (`qmv_wide`), GQA attention at ≥ 8K and residency stalls within weeks of being shown data.
  - llama.cpp iterates with AI assistance and crowd-sourced per-chip tuning.
  - Kernel-level openings (§12 items 1, 4, 5, 9, 14) are the most likely to close. Sky's pitch must rest on structural properties: exactness, a static GPU-resident plan, same-file comparisons, explainability, and a verified RAM metric ([Part XXVII](part27-moat.md)). It must not rest on one kernel.
- **M2 Pro evidence overstates some gaps.** MLX disables `qmv_wide` for affine weights on generation 14 by design, and the neural accelerators are absent. Every §12 entry measured only on the M2 Pro is re-measured on the M5 Max at G5 before it is claimed.
- **Third-party numbers are thin.**
  - The M5 Max Qwen3-32B baseline is one community run.
  - BaseRT's tables are a vendor report on non-identical files.
  - MTPLX's numbers are self-reported.
  - Mitigation: V-T11.
- **Others are building the same runtime shape.** Monolith (fused dispatches plus ICB replay, zero syncs per token) is the closest design to D9. If it ships first, Sky's differentiation narrows to the compiler, exactness and formats.
- **Core AI is unmeasured.** Apple's OS 27 inference framework could become the default on-device path. No LLM decode numbers exist yet.
- **Negative results carried into the plan** (§11): no headline from GEMV kernel work on dense decode; command replay alone gains nothing on dense decode; megakernels ≤ 3% at 32B; the neural accelerators do not help batch-1 decode; reordering ≈ 0%; n-gram drafting finds nothing on fresh text; unfused quantized KV makes decode slower; generic code generation for quantized GEMV reaches ≈ 11% of bandwidth.
- **Measurement risk.**
  - The dev machine drives a 4K display, and its scan-out is inside every M2 Pro bandwidth number.
  - Other agents' CPU jobs ran during measurements, with the GPU under a lock.
  - The tools' own metrics are not comparable (§2.3).

  No number in this part is quoted as a Sky result until the [Part XXIII](part23-benchmarks.md) harness re-measures it.
