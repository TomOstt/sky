# Part XVI — Decode optimization

This part designs plain batch-1 decode: one token per step, no speculation. It covers:
- the per-token time budget;
- the quantized GEMV skeleton and what Sky changes relative to MLX and llama.cpp;
- the fused layer of five or six kernels;
- GQA decode attention;
- sampling on the GPU;
- small and draft models;
- long context;
- how each decode target of SPINE §3 is measured.

Speculative decoding builds on everything here and is in [Part XXI](part21-speculative.md). The verify kernels of Part XXI are the same skeleton as the decode GEMV of §3, run with more than one token row, so the two parts must be read together.

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R4 §7.E) and live in [`research/`](research/). Physics numbers come from [`roofline.py`](roofline.py). Per-kernel byte counts below are the same arithmetic on the same `config.json` shapes: bytes = rows × columns × 4.5 / 8.

## 1. Summary

1. **Plain short-context decode of a large dense model has little headroom, and we say so.** On the M2 Pro, mlx-lm decodes Qwen3-8B 4-bit at 91% of MLX's own streaming ceiling (R4 §7.B). On the M5 Max, the one published run (28.4 tok/s for Qwen3-32B) implies about 515 GB/s, 96% of the 534 GB/s planning figure in `roofline.py` (§2.1). Sky's levers are:
   - streaming at ≥ 95% of the raw-Metal read bandwidth rather than MLX's ≈ 91% of its own ceiling;
   - five or six kernels per layer instead of 19;
   - no per-token CPU work.

   Together they are worth 1.09–1.11× on the M2 Pro (§2.2). On the M5 Max they are worth 1.05× or more only if that chip's raw read reaches about 575 GB/s (§2.1). `[EXPERIMENTAL: G0 measures the raw read; G5 measures the ratio]`
2. **The decode GEMV is a hand-written skeleton** with generated prologues and epilogues (SPINE D4):
   - an RMSNorm prologue;
   - residual-add, SwiGLU and partial-sum-of-squares epilogues;
   - row-concatenated Q‖K‖V and interleaved gate‖up weights.

   The skeleton keeps one fixed per-row reduction order for any number of token rows, so the same code is the verify kernel of Part XXI.
3. **The big decode wins are elsewhere:**
   - small and draft models, where mlx-lm is CPU- and dispatch-bound (§7);
   - long context, where MLX's decode attention reads K/V inefficiently (§8);
   - speculation ([Part XXI](part21-speculative.md)).
4. **Sampling moves to the GPU, but not for speed.** A fused sampler is worth ≤ 1–3% per token (R4 §9). It is needed because it removes the CPU from the token loop, enables accept/reject on the GPU for speculation, and makes keyed-Gumbel sampling bit-reproducible (§6).

## 2. The per-token budget

### 2.1 Qwen3-32B on the M5 Max (40-core GPU)

The decode step reads 17.99 GB of weights plus 256 KiB of fp16 KV per context position (`roofline.py`). The table splits the weight stream by Sky's fused kernels (§4). Two bandwidths are shown:
- 534 GB/s is `roofline.py`'s planning number (a contributor's ≈ 530 GB/s measurement; R3, R4);
- 556 GB/s is 95% of a hypothetical 585 GB/s raw read (95% of spec, the top of R6's 520–585 planning range).

| Kernel (per layer unless stated) | Weights read per layer | Per token (× 64 layers) | Time at 534 GB/s | Time at 556 GB/s |
|---|---:|---:|---:|---:|
| Q‖K‖V GEMV, 10,240 × 5,120 | 29.49 MB | 1.887 GB | 3.53 ms | 3.39 ms |
| O GEMV, 5,120 × 8,192 | 23.59 MB | 1.510 GB | 2.83 ms | 2.72 ms |
| gate‖up GEMV, 51,200 × 5,120 | 147.46 MB | 9.437 GB | 17.67 ms | 16.97 ms |
| down GEMV, 5,120 × 25,600 | 73.73 MB | 4.719 GB | 8.84 ms | 8.49 ms |
| LM head, 151,936 × 5,120 (once per token) | — | 0.438 GB | 0.82 ms | 0.79 ms |
| **All weights** | 274.27 MB | **17.99 GB** | **33.69 ms** | **32.36 ms** |
| KV at context 512 / 4K / 32K | 4 KiB per position | 0.134 / 1.074 / 8.590 GB | 0.25 / 2.01 / 16.09 ms | 0.24 / 1.93 / 15.45 ms |
| Kernel boundaries: Sky 6 × 64 + 3 = 387 (§4); MLX ≈ 1,225 | — | — | 0.39 ms (Sky) vs 1.23 ms (MLX) at an assumed 1.0 µs each | same |

Labels:
- shapes and bytes: `[VERIFIED: config.json via R8 §1.2; arithmetic]`;
- the 1.0 µs per dependent boundary on the M5 Max: `[EXPERIMENTAL: R6 measured 1.1–1.25 µs on the M2 Pro and assumes ≈ 1 µs on the M5 Max; G0 measures it]`;
- MLX's ≈ 1,225 kernels per 64-layer token: `[VERIFIED: R3 §7.F census]`.
- The boundary row counts the idle *bubble* only. With the ramp/fill cost of a small kernel that [Part XXII](part22-performance-model.md) fits (≈ 2.3 µs), the effective cost is ≈ 3.5 µs per dependent kernel. The dispatch component is then ≈ 1.4 ms for Sky and ≈ 4.3 ms for MLX per token. Part XXII's full per-kernel model, which includes this term, gives the modeled Sky / MLX-like ratio of 1.03–1.08× at short context quoted in §4. The table below is the conservative bubble-only view.

The gate‖up and down GEMVs are 79% of the bytes. Everything the compiler does outside them (norms, RoPE, attention at short context, sampling, dispatch) is worth at most a few percent.

**What MLX already achieves.** The community M5 Max run measured 28.41 tok/s at tg128 after a 512-token prompt (R5 §9.3; `[LIKELY: one documented community run, 2026-09-27]`). At ≈ 18.13 GB per token, that is ≈ 515 GB/s: 84% of the 614 GB/s spec (≈ 83% on the 17.99 GB of weights alone, as SPINE §2 counts it), and 96% of the 534 GB/s planning figure.

**What Sky can achieve.** At context 512, with weights and KV at 95% of the raw read and 387 boundaries at 1 µs:

| Raw-Metal read on the M5 Max (G0 measures it) | Sky at 95% of it | Sky decode | vs MLX's 28.41 |
|---:|---:|---:|---:|
| 534 GB/s | 507 GB/s | 27.7 tok/s | 0.98× |
| 560 GB/s | 532 GB/s | 29.0 tok/s | 1.02× |
| 585 GB/s | 556 GB/s | 30.3 tok/s | 1.07× |

`[EXPERIMENTAL: arithmetic on assumed bandwidths]`

So a ratio of ≥ 1.05× mlx-lm at ≥ 95% of the raw-Metal read holds only if the M5 Max's raw read is ≥ about 575 GB/s (94% of spec). On the M2 Pro the raw read is 98% of spec (R6), so this is plausible but unproven. SPINE §3 therefore sets ≥ 1.05× on the M2 Pro and ≥ 1.0× on the M5 Max until G0 measures the raw read. The bandwidth form of the target, ≥ 95% of the measured raw read, is the one we commit to on both chips. See §13.

### 2.2 Qwen3-8B on the M2 Pro, next to mlx-lm's measured split

R4 removed classes of operations from a bit-exact copy of mlx-lm's Qwen3 forward and timed what was left (R4 §7.E, `[VERIFIED]`). Context 128; mlx-lm's `generate_step` takes 25.79 ms per token (38.8 tok/s).

| Component | mlx-lm, measured | Sky, budgeted |
|---|---:|---:|
| Weights + KV (4.281 GB) at the streaming rate | 23.46 ms at MLX's 182.5 GB/s ceiling | 22.99 ms at 186.2 GB/s (95% of the 196 GB/s raw read) |
| GEMV shortfall below that rate | +0.47 ms (qmv chain at 97.5% of 182.5) | included above |
| Non-matmul work: dispatches, barriers, norm/RoPE/attention/SwiGLU/residual kernels | +1.97 ms (435 extra dispatches, 7.6%) | +0.26 ms (219 boundaries × 1.2 µs) |
| CPU graph build + encode | ≈ 0 exposed (0.65 ms build, hidden by `async_eval`) | 0 (static plan, replayed) |
| Sampling: the full-vocabulary `logsumexp` + subtract before the argmax | 0.17 ms, inside the total | fused into the LM-head epilogue |
| **Total** | **25.79 ms (38.8 tok/s)** | **≈ 23.25 ms (≈ 43.0 tok/s), 1.11×** |

At 93% of the raw read instead of 95%, Sky's total is 23.75 ms, i.e. 1.09×.

Labels:
- mlx-lm column: `[VERIFIED: R4 §7.E]`;
- the 196 GB/s raw read and 183–191 GB/s simple 4-bit GEMV: `[VERIFIED: R6 §B2]`;
- 1.2 µs per dependent boundary: `[VERIFIED: R6 §B3]`;
- Sky column: `[EXPERIMENTAL: G4 measures it]`.

Two facts make this credible:
- A simple MLX-layout 4-bit GEMV written by R6 already streams at 183–191 GB/s on this GPU (94–98% of the raw read). The skeleton does not need a new algorithm, only to keep that rate while fusing.
- llama.cpp's Q4_0 GEMV streams at 177 GB/s on the same machine (R5 §8.E), yet llama.cpp's whole token reaches only 154–159 GB/s. About 64 µs per layer is not explained by weight bytes (R5 §8.B, two-point fit). That gap is dispatch and barrier time, which fusion removes.

### 2.3 What the budgets mean

| Lever | Size on 8B / M2 Pro | Size on 32B / M5 Max | Label |
|---|---|---|---|
| Stream at 95% of raw read instead of MLX's rate | ≈ 1.02× (23.46 → 22.99 ms) | 0.98–1.07×, depending on the raw read | EXPERIMENTAL |
| 5–6 kernels per layer instead of 19 | ≈ 1.08× (1.97 → 0.26 ms) | ≈ 1.03× (1.23 → 0.39 ms) | LIKELY |
| No CPU in the loop | ≈ 1.00× | ≈ 1.00× (32B is GPU-bound) | VERIFIED for 8B (R4 §7.E) |
| **Combined** | **1.09–1.11×** | **≈ 1.0–1.1×** | EXPERIMENTAL |

This matches SPINE §2 ("about 1.05–1.14×") and BaseRT's result (0.98× MLX on a dense 27B on the M5 Pro; R7 §8.1). It is table stakes, not the pitch.

## 3. The quantized GEMV

### 3.1 The two reference designs

Both are near the bandwidth limit at M = 1 and both are MIT-licensed. We learn from them and write our own (SPINE D15).

| | MLX `affine_qmv_fast` | llama.cpp `kernel_mul_mv_q4_0` |
|---|---|---|
| Threadgroup | 64 threads = 2 simdgroups | 64 threads = 2 simdgroups |
| Output rows | 4 per simdgroup, 8 per threadgroup | 4 per simdgroup, 8 per threadgroup |
| Lane work per iteration | 16 4-bit values (two `uint32`); 32 lanes cover 512 K-values | half a 32-weight block; 16 blocks in flight per simdgroup |
| Weight bytes per row per iteration | 256 contiguous bytes (32 lanes × 8 B) | 18-byte blocks, read as `uint16` |
| Dequantization | activations pre-divided by 1, 16, 256, 4096, so a nibble is used masked but unshifted; `scale · Σ(x·q) + bias · Σx` once per 16 values | same pre-scaling trick; the −8 zero point folded into Σy once per half-block |
| Reduction | per-lane fp32 partials over all of K, then one `simd_sum` per row | one `simd_sum` per row |
| Threadgroup memory, barriers | none | none |
| Measured on the M2 Pro | 91–97% of 182 GB/s on large shapes; k/v projections 54–63% | 177 GB/s at 4096 × 14,336 |
| Source | `mlx/backend/metal/kernels/quantized.h:756-821`; dispatch `quantized.cpp:465-540` (R3 §3.3) | `ggml/src/ggml-metal/kernels/mul_mv.metal:218-304` (R5 §2.3) |

Both designs agree on what matters:
- independent threadgroups that own disjoint output rows;
- no threadgroup memory in the K loop;
- activations in registers, reused across 4 rows;
- dequantization in about two ALU operations per weight;
- one `simd_sum` per row.

That is the design space Sky's skeleton lives in. Neither engine fuses anything into its GEMV: MLX compiles only SwiGLU, and llama.cpp has a fixed table of 9 small fusion patterns (R3 §1.4, R5 §1.3).

Small-M behaviour is where they differ. It matters for [Part XXI](part21-speculative.md):
- MLX on gen-14 GPUs (M1/M2) runs one threadgroup row per token row and re-reads the weights per row. On gen ≥ 15 it runs `qmv_wide` for M = 2–5, which decodes each group once and applies it to up to 5 vectors (R3 §3.2).
- llama.cpp's `mul_mv_ext` reads the weights once for 2–5 columns and twice for 6–8 (R5 §2.3).

### 3.2 Sky's skeleton: `qgemv`

One skeleton, `qgemv`, in `sky-kernel` (`crates/sky-kernel/src/skeletons/qgemv.rs`). The weight formats plug in as codecs (`crates/sky-kernel/src/codecs/`, [Part XII](part12-quantization.md)), and the MSL template is specified in [Part XI](part11-metal-backend.md) §9.1. Its parameters:

| Parameter | Values searched by the tuner ([Part X](part10-autotuner.md)) | Default from R6's measurements |
|---|---|---|
| `rows_per_simdgroup` | 2, 4 | 4 |
| `values_per_lane` per iteration | 8, 16 | 16 |
| `simdgroups_per_threadgroup` | 2, 4, 8 | 8 on M2 Pro (186.3 GB/s on 25,600 × 5,120) |
| `token_rows` M | 1 … 8 on the ALU path | 1 for decode |
| weight format | MLX affine (2–8 bits, group 32/64/128), GGUF Q4_0, Q8_0, Q4_K, Q6_K, MXFP4 | from the file |
| prologue / epilogue | generated (§3.5–§3.7) | from the fusion plan |

The inner loop is fixed by the skeleton:
- lane ℓ of a simdgroup owns K-slices `ℓ·v + i·32·v … +v` for iteration i;
- it accumulates one fp32 partial per (token row, output row) in that fixed order;
- each row ends with one `simd_sum`.

The skeleton never splits K across threadgroups and never uses atomics. That fixed per-row schedule is the "reduction schedule" attribute of SPINE D6. The verifier requires it to be identical in every instantiation of the family (M = 1 decode, M = 2…8 verify). That makes verify logits bit-identical to decode logits ([Part XXI](part21-speculative.md) §3).

### 3.3 Streaming at ≥ 95% of the raw read

What R6 measured on the M2 Pro (`[VERIFIED: R6 §B2, microbench/results/bw.txt]`):

| Configuration of a simple MLX-layout 4-bit GEMV (25,600 × 5,120 unless noted) | GB/s |
|---|---:|
| 4 rows/simdgroup, 16 values/lane, 2 simdgroups/threadgroup | 183.1 |
| 4 rows, 16 values, 8 simdgroups | **186.3** |
| 8 rows, 16 values, 2 simdgroups | 175.3 |
| 8 rows, 32 values, 2 simdgroups (register pressure) | 160.7 |
| Best configuration on 131,072 × 5,120 (378 MB) | **191.4** |
| Best scalar streaming read, for reference | 196.3 |

Design rules that follow:
1. **About 16 bytes in flight per lane per row, and 2–4 rows per simdgroup.** More rows per simdgroup raises register pressure and costs bandwidth. MLX's attempt to double `results_per_simdgroup` was neutral or negative in real decode (mlx#3753, R3-issues C4).
2. **Occupancy comes from many small threadgroups,** not from prefetching tricks (R5 §10). At N = 1,024 (Qwen3-32B's k and v projections), there are too few threadgroups to fill the GPU: MLX reaches 54–63% there (R3 §7.B). Horizontal fusion (§3.6) removes these small shapes.
3. **Activations are read once per threadgroup, from cache.** MLX re-reads and re-scales the input vector in every simdgroup. For N = 25,600 that is about 33 MB of L1/L2 traffic per call against 73.7 MB of DRAM weights (R3 §3.3). Sky loads the vector once per threadgroup into registers or threadgroup memory. `[EXPERIMENTAL: worth ≤ 1–2%]`
4. **Per-shape tuning, timed honestly:** DRAM-sized rotating working sets, GPU timestamps and medians ([Part X](part10-autotuner.md)). tinygrad's BEAM mis-ranked Q4_0 GEMVs because Metal timing never flushed caches (R2 §0, item 9).

Acceptance bar (G1/G5): every Qwen3-8B/32B GEMV of ≥ 20 MB streams at ≥ 95% of the measured raw read on the M2 Pro (≥ 186 GB/s), and at ≥ 93% on the M5 Max.

### 3.4 Relayout

The relayout cache (`.skyweights`, SPINE D9; [Part XII](part12-quantization.md), [Part XIX](part19-memory.md)) does three jobs for decode.

1. **Alignment (mandatory).** mlx-community safetensors files start tensor data at offset ≡ 6 (mod 8). On the M2 Pro an unaligned `uint` load silently reads the rounded-down word (R9 §1.3, `[VERIFIED]`). Every tensor is written at a 16 KiB page boundary.
2. **Concatenation for horizontal fusion:**
   - Q, K and V rows are stored as one `[10,240 × 5,120]` matrix (Qwen3-32B) with their scales and biases.
   - gate and up rows are interleaved in blocks of `rows_per_threadgroup`, so the simdgroup that computes gate row j also computes up row j (§3.6).
3. **A decode-friendly block order (optional).** A threadgroup's rows, codes and their (scale, bias) pairs can be stored contiguously, so one threadgroup reads one sequential stream instead of three (codes, scales, biases).

   R6's 183–191 GB/s were measured on the plain MLX layout, so this layout is adopted only if it measures ≥ 2% faster. `[EXPERIMENTAL: XVI-T2]`

Rule (SPINE D5): **relayout never re-quantizes.** The test `crates/sky-quant/tests/relayout_bits.rs` dequantizes every tensor of the source checkpoint and of the relayout cache and requires bit-equal values.

GGUF is 32-byte aligned and is mapped directly (R5 §1.5). For GGUF the skeleton reads the block format in place, as llama.cpp does. One relayout is worth evaluating: Q4_K into separate code and scale arrays. llama.cpp's Q4_K GEMV reaches only 147 GB/s against 177 GB/s for Q4_0 on the same shape, and Q4_K_M decodes 13% slower per token for 5% more bytes (R5 §8.E, §8.G). `[EXPERIMENTAL]`

### 3.5 The fused RMSNorm prologue

RMSNorm needs the sum of squares of the whole hidden vector before any normalized element exists. A GEMV threadgroup already reads the whole vector, so the question is only where Σx² comes from.

Design:
1. **The producer writes partial sums.** The kernel that writes the residual stream (the embedding gather, the O-GEMV epilogue or the down-GEMV epilogue, §3.7) computes, per threadgroup, the sum of squares of the values it stored. It uses the rounded stored values, not the fp32 pre-rounding values. It writes one fp32 partial per threadgroup: 640 partials for Qwen3-32B's 5,120-wide stream at 8 rows per threadgroup.
2. **The consumer reduces them in a fixed order.** Every consumer threadgroup reads the partials, adds them in index order in fp32, and computes `r = rsqrt(Σ/D + ε)`. It then forms `x̂_k = x_k · r · w_k` in registers while loading the activations. That is 2.5 KB of cached reads per threadgroup, instead of a second pass over x to compute Σx².
3. **No atomics.** The reduction order is fixed by the plan, so the result is bit-reproducible (SPINE D6). Apple GPUs also have no native fp32 atomics (R7 §8.2).

The norm weight cannot be folded into the quantized matrix: that would re-quantize it (D5). So it is applied on the activation side.

Mathematically, the prologue could instead apply r in the epilogue (`r · Σ_k x_k w_k W_nk`). We do not, so that decode, verify and prefill share one formulation (see [Part XVII](part17-prefill.md) §5 for why prefill needs r before the matmul).

This removes 4 RMSNorm dispatches per layer. MLX runs Qwen3-32B's 5,120-wide RMSNorm with its looped kernel, which reads x twice (R3-kernels §c). Removing all RMSNorms saved 0.76 ms per token on 8B in R4's ablation (R4 §7.E).

### 3.6 Horizontal Q‖K‖V and gate‖up

| Fusion | Shapes (Qwen3-32B) | What it removes |
|---|---|---|
| Q‖K‖V | 8,192 + 1,024 + 1,024 = 10,240 rows × 5,120 | 2 dispatches per layer; the two 1,024-row GEMVs that MLX runs at 54–63% of bandwidth (R3 §7.B) |
| gate‖up with SwiGLU epilogue | 2 × 25,600 rows × 5,120, interleaved per threadgroup | 2 dispatches per layer (one GEMV and the SwiGLU kernel); the write and re-read of two 25,600-wide intermediates |

The rewrite lives in the graph optimizer (`crates/sky-tir/src/passes/horizontal.rs`, [Part VIII](part08-graph-optimizer.md)). The matching weight concatenation is produced by the relayout cache. Sky may change the layout; it never re-quantizes (D5).

### 3.7 Epilogues

| Epilogue | Attached to | Work | Notes |
|---|---|---|---|
| residual add | O GEMV, down GEMV | `h = x + y`, stored once | MLX: 2 separate `Add` dispatches per layer (R4 §3.1) |
| partial Σh² | same | per-threadgroup fp32 partial of the stored values | feeds the next prologue (§3.5) |
| SwiGLU | gate‖up GEMV | `silu(g) · u` in registers; only the 25,600-wide product is written | MLX: one compiled kernel (R4 §3.2) |
| KV append | Q‖K‖V GEMV | V rows go straight into the KV cache; q and k rows get QK-norm + RoPE, then K is appended | rules and layout in [Part XIV](part14-kv-cache.md) §4 |
| logits → sampling candidates | LM-head GEMV | per-threadgroup best candidates (§6) | removes the full-vocabulary `logsumexp` pass |

**Where QK-norm and RoPE go.** QK-norm is an RMSNorm over each 128-wide head, so it can run in the Q‖K‖V epilogue only when one threadgroup owns whole heads (a multiple of 128 output rows). With MLX-style tiles of 8 rows it cannot. So:
- the scheduler folds QK-norm + RoPE + K append into K1's epilogue when the tuned tile owns whole heads;
- otherwise they run in the tiny `qk_norm_rope_append` kernel ([Part XIV](part14-kv-cache.md) §4.1). It reads and writes about 20 KB per layer and costs one dependent boundary, ≈ 0.07 ms per 32B token (0.2%).

The choice is measured, because a whole-head tile raises register pressure in K1 ([Part XXIV](part24-roadmap-now.md)'s occupancy guard). Either way, the append happens before attention reads the cache, and decode and verify write identical bytes for the same position ([Part XIV](part14-kv-cache.md) §4.2).

### 3.8 Numerics

The `inference` contract (SPINE D6) applied to the GEMV family:
- fp32 accumulation, FMA allowed;
- the fixed per-row schedule of §3.2;
- reductions across threadgroups only through fixed-order partials (§3.5);
- compiled with the Metal compiler's fast-math reassociation off.

MLX builds its precompiled kernels with `-fno-fast-math` (R3 §3). The Metal runtime compiler's default math mode may reassociate, so `sky-metal` sets the safe math mode explicitly and records it in the tuning key ([Part XI](part11-metal-backend.md)). `[LIKELY: Metal's runtime-compile default has fast math enabled; XVI-T1's bit-equality test detects reassociation]`

Tolerance against the reference interpreter is the correctness gate. Bit-identity with MLX is never promised: even bf16 vs fp16 greedy decoding diverges on 49–100% of prompts (R8 §3.4).

## 4. Five to six kernels per layer

The fusion partition is decided by the scheduler ([Part IX](part09-scheduler.md)). This is the plan it is expected to find for a Qwen3 layer, and the plan we hand-write first (G4):

The kernel names follow [Part IX](part09-scheduler.md) §3, [Part XIV](part14-kv-cache.md) §4.1 and [Part XXII](part22-performance-model.md) §6.

```text
 residual x (+ Σx² partials from the previous kernel)
   │
   ├─K1   Q‖K‖V GEMV        prologue: RMSNorm(input_layernorm) from partials
   │                          epilogue: V append; QK-norm + RoPE + K append when a threadgroup owns whole heads
   │                          (otherwise the tiny qk_norm_rope_append kernel, §3.7)
   ├─K2   decode attention  GQA-packed, fixed split size S, one threadgroup per (KV head, split), all G·q rows
   ├─K2c  split combine     fixed-order merge → attn_out (8,192); only when the context spans more than one split
   ├─K3   O GEMV            epilogue: h = x + o; Σh² partials
   ├─K4   gate‖up GEMV      prologue: RMSNorm(post_attention_layernorm) from partials; epilogue: silu(g)·u → a (25,600)
   └─K5   down GEMV         epilogue: x' = h + d; Σx'² partials ─────────► next layer's K1
 per token: embedding gather (+ Σx² partials) · H1 final RMSNorm + LM-head GEMV + sampling candidates · H2 sample finalize
```

Per token that is 6 × 64 + 2 = 386 kernels ([Part XXII](part22-performance-model.md) §6), plus the embedding gather. Part XXII's per-kernel model of this plan gives 35.30 ms per token (28.3 tok/s) at the 534 GB/s planning bandwidth. That modeled figure is not compared with MLX's measured 28.4 tok/s. The robust comparison is the model's ratio of this plan to an MLX-like plan under the same parameters: 1.03–1.08× at short context ([Part XXII](part22-performance-model.md) §6.3). It is the same conclusion as §2.1: plain short-context decode on the M5 Max gains a few percent at most.

| Engine | Dispatches per token, Qwen3-32B | Per layer | Source |
|---|---:|---:|---|
| MLX / mlx-lm | ≈ 1,225 | 19 | R3 §7.F, R4 §3.1 (`[VERIFIED]` census) |
| llama.cpp (Qwen3-8B, 36 layers) | ≈ 724, with ≈ 542 barriers | ≈ 20 dispatches, ≈ 15 barriers | R5 §8.D2 (`[VERIFIED]`) |
| tinygrad (Qwen3-0.6B) | 488 | ≈ 17 | R2 §8.3 (`[VERIFIED]`) |
| **Sky** | **322–387** | **5–6** | this plan (`[EXPERIMENTAL]`) |

**K2c inside K2.** Each split threadgroup writes its partial, issues a device-scope fence and increments a per-head atomic counter. The threadgroup that observes the last increment performs the combine, in split order.
- Nothing ever waits on another threadgroup, so the co-residency deadlock R6 measured for spinning barriers does not apply (R6 §A4).
- The combine order is fixed, so the result is deterministic.
- R6 verified that device-scope fences plus atomic RMWs give correct cross-threadgroup visibility on the M2 Pro (R6 §B6).

`[EXPERIMENTAL: adopt only if it is measured faster than a separate K2c; [Part XV](part15-attention.md) owns the kernel and specifies the fixed-order combine]`

**What the dispatch reduction is worth.**
- On the M2 Pro, 8B: R4's ablation shows the 435 non-matmul dispatches cost 1.97 ms per token, about 4.5 µs each including their work (R4 §7.E).
- On the M5 Max, 32B: ≈ 0.8 ms per token at the assumed 1 µs per boundary, about 2.5% (§2.1). With the fill term [Part XXII](part22-performance-model.md) fits, the effective cost is ≈ 3.5 µs per dependent kernel (SPINE §2), and the reduction is worth ≈ 3–7% (Part XXII §6.3).

The megakernel alternative buys nothing more for dense 32B decode: a device-wide barrier costs as much as a kernel boundary (R6 §B6). It is rejected for production; only a G9 feasibility spike remains ([Part XX](part20-persistent-execution.md) §8).

## 5. GQA decode attention

The kernel is designed in [Part XV](part15-attention.md). Decode needs four things from it.

1. **Read each K/V byte once per KV head.** Qwen3-32B has 64 query heads and 8 KV heads. MLX's 1-pass and 2-pass vector kernels have every query-head simdgroup stream the same K/V block. Its deduplicating `sdpa_vector_2pass_1_gqa` kernel starts only at a key length of ≥ 8,192 (mlx#4077; R3-kernels §c, R3-issues F3). Sky's kernel packs the 8 query heads of a group into one threadgroup at every context length.
2. **A fixed split size, not a fixed number of splits** (flash-decoding). This keeps results independent of how many tokens are verified together, and lets chunked prefill align to split boundaries (Thinking Machines recipe; R8 §3.3).
3. **K and V appended before attention reads them,** by K1's epilogue or the tiny append kernel (§3.7; [Part XIV](part14-kv-cache.md) §4.2). MLX spends 2 RoPE and 2 KV-write dispatches per layer here (R4 §3.1).
4. **Multi-query support up to the verify width.** MLX's fused vector kernel requires `qL · gqa ≤ 32`, so Qwen3-32B gets it only for qL ≤ 4. Five to eight queries fall back to an unfused path (R3-kernels §c). Sky's kernel takes any number of queries up to the plan's verify width ([Part XXI](part21-speculative.md) §3.4).

At short context attention is small: about 0.25 ms of the 32B step at context 512 (§2.1). At long context it dominates (§8).

## 6. Sampling on the GPU

**Design.** Sampling is the LM-head GEMV's epilogue plus one finalize kernel. Logits are computed in fp32 and never written to memory in full.

| Mode | LM-head epilogue (per threadgroup of rows) | Finalize kernel |
|---|---|---|
| greedy | the best (logit, token id) pair; ties go to the lowest id | fixed-order argmax over the threadgroup winners |
| keyed Gumbel, temperature T | the best (l/T + g(seed, request, position, id), id), with g from a counter-based generator (Philox) | fixed-order argmax |
| top-k (k ≤ 64; Qwen's card uses k = 20) | the threadgroup's top-k list | merge the lists; then top-p and min-p over the k survivors; Gumbel-argmax over the survivors |
| repetition / presence / frequency penalties, logit bias | applied to each logit before selection, from a per-request token-count array on the GPU | — |
| log-probabilities, if requested | per-threadgroup partial log-sum-exp | fixed-order merge; only then is the softmax needed |

The finalize kernel writes:
- the token id into the next step's input slot;
- the position + 1;
- the RNG counter;
- the stop flag (EOS or stop sequence);
- the id into the streamer's ring buffer ([Part XX](part20-persistent-execution.md), SPINE D9).

The keyed-Gumbel construction is what makes sampled speculation byte-identical to sampled plain decoding ([Part XXI](part21-speculative.md) §6).

**An honest note on speed.** A fused sampler is not a decode speedup:
- mlx-lm already samples on the GPU inside its lazy graph (R4 §2.4).
- Skipping its full-vocabulary `logsumexp` saves 0.17 ms per token on 8B (0.7%) and 0.10 ms on 0.6B (3%) (R4 §9, `[VERIFIED]`).
- The outside claim of "5–8%" for skip-logsumexp did not reproduce (mlx-lm #1479, R4 §9).

We expect ≤ 1% on 32B and ≤ 3% on 0.6B. GPU sampling is in the plan for three other reasons:
1. **No CPU in the token loop.** When the CPU must see token t before it can submit token t+1, the GPU idles for about 105 µs (R6 §B3, `[VERIFIED]`). llama.cpp samples on the CPU by default: it copies 608 KB of logits per token and calls `waitUntilCompleted` before every sample (R5 §6). For 32B this costs about 0.3%. For a 0.6B draft model on the M5 Max (≈ 1 ms per step, §7) it costs about 10%.
2. **Accept/reject on the GPU** for speculation, which needs target and draft distributions where they are computed ([Part XXI](part21-speculative.md) §7).
3. **Bit-reproducible seeded sampling** (SPINE D9).

## 7. Small and draft models

Small models are where plain decode has real headroom. A 0.6B model is the draft of [Part XXI](part21-speculative.md), so its step time is the draft cost c.

| Qwen3-0.6B 4-bit, M2 Pro | mlx-lm (R4) | llama.cpp Q4_0 (R5) |
|---|---|---|
| Decode | 267–281 tok/s (3.55–3.76 ms) | 247.5–250 tok/s |
| Share of the bandwidth roofline | 54.5% | 51% of achievable |
| Dispatches per token | 541 (362 barriers) | ≈ 564 |
| CPU per token | 1.94 ms build + encode, against ≈ 3.5 ms of GPU | graph reuse; disabling it costs −42% |
| GEMVs alone (qmv-only chain) | 2.41 ms (76.4% of achievable) | — |

All `[VERIFIED: R4 §7.B–§7.E, R5 §8.A, §8.F]`.

Two conclusions:
- **On a GPU three times faster, mlx-lm becomes CPU-bound on this model.** Its 1.94 ms of Python graph building and encoding per token stays the same while the GPU time shrinks (R4 §7.E). `[LIKELY]` This is consistent with BaseRT's 1.33× over mlx-lm on Qwen3-0.6B on the M5 Pro (R7 §8.1).
- **The win is the static plan,** not kernel heroics:
  - 6 kernels per layer (171 per token instead of 541);
  - no graph build;
  - no per-token encoding (replay; [Part XX](part20-persistent-execution.md));
  - no CPU round trip.

  llama.cpp's own ablation shows what per-token CPU work costs a 0.6B model: −42% without graph reuse (R5 §8.F).

Modeled Sky step for Qwen3-0.6B (0.335 GB of weights; GEMV efficiency assumed 70–80% because the matrices are 1–4 MB):

| Chip | Weights | Boundaries (171) | Step | tok/s | vs mlx-lm |
|---|---:|---:|---:|---:|---:|
| M2 Pro (196 GB/s, 1.2 µs) | 2.14–2.44 ms | 0.21 ms | 2.34–2.65 ms | 377–427 | 1.4–1.6× |
| M5 Max (534 GB/s, 1.0 µs assumed) | 0.79–0.90 ms | 0.17 ms | 0.96–1.07 ms | 936–1,046 | not measured |

`[EXPERIMENTAL: G4 requires ≥ 1.3× on the M2 Pro; SPINE §3 targets ≥ 1.5×]`

Two techniques specific to small models:
- **Fill the GPU.** A 1,024-row GEMV has 128 threadgroups of 8 rows: about 3 per core on a 40-core GPU. The tuner should prefer fewer rows per threadgroup for these shapes, and the scheduler may horizontally fuse more of a layer. `[EXPERIMENTAL]`
- **Truncated draft vocabulary.** Qwen3-0.6B's tied LM head is 87.5 MB, 26% of its bytes. A draft only proposes tokens, so it may use a reduced vocabulary, as EAGLE-3 drafters do with 32,000 entries (R8 §2.3). Exactness is unaffected: the target verifies every token ([Part XXI](part21-speculative.md) §6). `[EXPERIMENTAL: the acceptance loss is measured by XXI-T15]`

Draft cost for a Qwen3-32B target on the M5 Max: with a target step of ≈ 35.9 ms (§2.1, at 507 GB/s) and the step above, c ≈ 0.027–0.034. R8 estimated c ≈ 0.05–0.1 for a 0.6B draft in a latency-bound runtime (R8 §2.5). Sky's plan is what moves it to the 0.03 that Part IV's model assumes ([Part IV](part04-hardware-limits.md) §8).

## 8. Long context

At long context the KV stream competes with the weights. For Qwen3-32B it adds 4 KiB per position per layer: 1.07 GB per token at 4K and 8.59 GB at 32K, against 17.99 GB of weights (`roofline.py`). Decode efficiency is then decided by the attention kernel.

**Evidence on MLX's KV stream:**
- **M2 Pro, Qwen3-8B (GQA 4), real model:** going from 128 to 4,096 context costs +6.04 ms per token for 585 MB of extra KV, an effective 97 GB/s (53% of MLX's achievable) (R4 §7.A, `[VERIFIED]`). R3's synthetic 8-layer stack measured about 56 GB/s for the same step (R3 §7.F).
- **M5 Max, decode SDPA kernel, 32 query / 8 KV heads, 32K keys:** 56.2% of peak. The kernel "is sitting on a fixed cost of roughly 0.3 ms at this context length" (mlx#3837, R3-issues F2, `[VERIFIED statement; contributor data]`).
- **Since then:** mlx#4077 (merged 2026-08-18) added the GQA-deduplicating kernel for GQA 8, head_dim 128 and ≥ 8,192 keys, with 1.08–1.31× kernel speedups on the M5 Max (R3-issues F3). Qwen3-32B qualifies above 8K context.

**Sky's design** is the GQA-packed, fixed-split `flash_decode` kernel of §5, with a target of ≥ 85% of the measured read bandwidth at 4K–32K ([Part XV](part15-attention.md) §4.4). `[EXPERIMENTAL]`

**What better attention can win.** [Part XV](part15-attention.md) §8.2 derives the bound: both engines stream weights at the same rate, Sky's attention runs at 90% of achievable, and MLX's at efficiency e. At e = 0.53, R4's measurement, the speedup from attention alone is ≈ 1.08× at 8K, ≈ 1.14× at 16K and ≈ 1.24× at 32K. The gain grows with context only because the KV share grows (10.7%, 19.3% and 32.3% of the bytes per token). At 8K, even a 2× attention advantage is worth only ≈ 1.08×.

**Targets** (SPINE §3, as restated with Part XV):

| Context | Target vs mlx-lm (fp16 KV) | Bound at MLX's ≈ 53% KV-read efficiency (Part XV §8.2) |
|---|---:|---:|
| 4K, Qwen3-8B on the M2 Pro | ≥ 1.05× (stretch 1.10×) | — |
| 8K, Qwen3-32B on the M5 Max | ≥ 1.05× | ≈ 1.08× |
| 16K | ≥ 1.10× | ≈ 1.14× |
| 32K | ≥ 1.2× | ≈ 1.24× |

Two caveats:
- The bounds assume weight streaming at parity. If Sky also streams weights faster than MLX (§2.1, which depends on the M5 Max's raw read), the two gains multiply.
- They assume MLX at 53%. mlx#4077's GQA kernel (≥ 8K keys, 1.08–1.31× kernel speedups) may raise MLX's efficiency at these lengths. The bounds then shrink: at an assumed e = 0.75 they are 1.02×, 1.04× and 1.07× (Part XV §8.2). G5 measures e before the target is published (XVI-T10).

On the M2 Pro the evidence is direct. For Qwen3-8B at 4K, mlx-lm takes 31.57 ms (R4 §7.A). Sky's budget is about 26.8 ms (weights at 95% of the raw read, KV at 85%, 219 boundaries), i.e. ≈ 1.18×, of which about 1.11× is the short-context gain of §2.2. `[EXPERIMENTAL]`

Two further long-context levers are deliberately outside the plain fp16 target:
- **8-bit KV as a labeled mode.** It is worth 1.05× at 8K and 1.18× at 32K on the roofline (R8 §1.5), and halves KV memory. mlx-lm's unfused quantized-KV attention instead slowed decode by 6–22% (mlx-lm #1651, R4 §9). A fused kernel is the differentiator. Quality is reported as KL ([Part XIV](part14-kv-cache.md)).
- **Speculation at long context gets more expensive.** The free verify width of attention is q\* = R·s/(2G): ≈ 4 queries on the shader ALUs and ≈ 15 on the M5 neural accelerators, halved by int8 KV ([Part XV](part15-attention.md) §2.2). Long-context verify attention therefore runs on the neural accelerators ([Part XXI](part21-speculative.md) §2.3).

## 9. How each decode target is measured

All speed claims follow the protocol of R9 §7 and [Part XXIII](part23-benchmarks.md):
- same-file pairs (D5);
- one external harness timestamping every token for every engine;
- interleaved, randomized blocks;
- a claim counts as a win only when the bootstrap 95% CI of the paired ratio of medians excludes 1.0.

The correctness gates of R9 §7.3 must pass first:
- tokenizer ids identical;
- KLD(Sky ‖ baseline) ≤ 0.1 × KLD(baseline ‖ BF16);
- top-1 agreement ≥ 99%;
- greedy divergences only at near-ties.

| SPINE §3 target | Scenario (R9 §7.4) | Metric (R9 §7.5) | Baseline and file | Pass rule |
|---|---|---|---|---|
| Decode, plain, ≤ 1K context: ≥ 1.05× mlx-lm on the M2 Pro (stretch 1.10×); ≥ 1.0× on the M5 Max until its raw read is measured (≥ 1.05× needs ≈ 575 GB/s); ≥ 95% of the raw read on both | S1: 512-token prompt, 512 tokens decoded | (N − 1)/(t_last − t_first); effective bandwidth = bytes read per token × tok/s | mlx-lm on the MLX 4-bit g64 file; llama.cpp on Q4_0 / Q4_K_M GGUF | CI of the ratio > 1.05 on the M2 Pro and not below 1.0 on the M5 Max, and effective bandwidth ≥ 0.95 × the G0 raw read |
| Decode, small / draft models (0.6B–1.7B): ≥ 1.5× | S1 on Qwen3-0.6B and 1.7B | as above, plus CPU time per token | mlx-lm, same files | CI of the ratio > 1.5 |
| Decode, long context: ≥ 1.05× at 8K, ≥ 1.10× at 16K, ≥ 1.2× at 32K (M5 Max); ≥ 1.05× at 4K (M2 Pro) | S3: prompts of 4K, 8K, 16K and 32K, then 256 tokens | decode tok/s at the stated depth, plus each engine's KV-read efficiency from `sky profile` and a Metal trace | mlx-lm, 16-bit KV on both sides (bf16 vs f16 stated) | CI of the ratio above the target at each context length (§8) |
| Peak memory ≤ 0.9× at 32K | S7 with S3(32K) | max rise in system memory used over a pre-spawn baseline, at 10 Hz | mlx-lm | ratio ≤ 0.9 ([Part XIX](part19-memory.md)) |
| Energy per token ≤ 0.85× (plain decode) | S6: decode loop ≥ 60 s; S4: 10-minute sustained | whole-system J/token from SMC `PSTR`, gross and net of idle | mlx-lm | ratio ≤ 0.85 |
| Sustained performance | S4 | last-5-minutes median ÷ first-minute median | both engines | reported next to every headline |

Each row is printed with the file's bits per weight and bytes per token, and with the quality numbers, next to the speed (R9 §7.6).

Diagnostics for every decode run:
- `sky explain` prints the per-kernel plan: kernels, bytes, predicted and measured µs ([Part XXIII](part23-benchmarks.md));
- `sky profile` shows GPU-timestamped kernel times against the roofline.

A decode win that `sky explain` cannot attribute to bytes, boundaries or CPU time is treated as a measurement error until explained.

## 10. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XVI-T1 | `qgemv` skeleton, M = 1, MLX affine 4-bit g64, aligned MLX layout (with XI-T10 and XII-T7) | `crates/sky-kernel/src/skeletons/qgemv.rs`, codecs in `crates/sky-kernel/src/codecs/`, MSL template per [Part XI](part11-metal-backend.md) §9.1 | Kernel plus tuner parameter space (§3.2) | On every Qwen3-8B/32B shape ≥ 20 MB on the M2 Pro: ≥ 186 GB/s (95% of 196) and ≥ MLX `quantized_matmul` (G1); output within tolerance of the TIR reference; run-to-run bit-identical; built with safe math (fails if reassociation changes bits between two instantiations) |
| XVI-T2 | Relayout cache for decode: alignment, Q‖K‖V concatenation, gate‖up interleave; optional block order | `crates/sky-quant/src/relayout.rs`, `crates/sky-llm/src/weights/cache.rs` | `.skyweights` writer and reader | `sky-quant/tests/relayout_bits.rs`: dequantized values bit-equal to the source for every tensor; the block order kept only if ≥ 2% faster |
| XVI-T3 | Horizontal-fusion pass for Q‖K‖V and gate‖up | `crates/sky-tir/src/passes/horizontal.rs` | Rewrite with rule IDs and explanations | `sky explain` shows 1 Q‖K‖V and 1 gate‖up GEMV per layer; logits within tolerance of the unfused plan |
| XVI-T4 | RMSNorm prologue and partial-Σx² epilogue | `crates/sky-kernel/src/prologues/rms_norm.rs`, `crates/sky-kernel/src/epilogues/norm_partials.rs` | Generated prologue and epilogue code | Bit-identical across runs; within tolerance of the reference; a test that changes the producer's threadgroup count must change the plan (partials are part of the schedule), never silently the result |
| XVI-T5 | Residual-add and SwiGLU epilogues | `crates/sky-kernel/src/epilogues/{residual,swiglu}.rs` | Generated epilogues | Layer output within tolerance; 4 fewer dispatches per layer in `sky explain` |
| XVI-T6 | Decode program: 5–6 kernels per layer, static plan, replayed | `sky-llm/src/decode/plan.rs`, `sky-metal` | First fused decode for Qwen3 | ≤ 6 × L + 3 dispatches per token; Qwen3-8B decode ≥ 1.0× mlx-lm on the M2 Pro (G4), ≥ 1.05× at G5 |
| XVI-T7 | GPU sampler: greedy, keyed Gumbel (Philox), top-k / top-p / min-p, penalties, logit bias | `crates/sky-kernel/src/skeletons/sample.rs`, `crates/sky-llm/src/sample/` | LM-head epilogue plus finalize kernel | Seeded runs bit-reproducible; empirical distribution matches the reference sampler (chi-square over 10⁶ draws on fixed logits); no CPU access to logits in the decode loop |
| XVI-T8 | Small-model plan (Qwen3-0.6B / 1.7B) | `sky-llm/src/decode/plan.rs` | Tuned plan for small shapes | ≥ 1.3× mlx-lm on the M2 Pro (G4); ≥ 1.5× at G5 |
| XVI-T9 | GGUF-native GEMV codecs: Q4_0, Q8_0, Q4_K, Q6_K (with XII-T7) | `crates/sky-kernel/src/codecs/gguf_*.rs` | Kernels that read GGUF blocks in place | Q4_0 ≥ llama.cpp's 177 GB/s on the R5 shape; Q4_K ≥ 165 GB/s (llama.cpp: 147) |
| XVI-T10 | Long-context matrix | `sky-bench`, `sky-evaluation` | S3 at 4K / 8K / 16K / 32K on the M2 Pro (8B) and M5 Max (32B) | Published with KV-stream efficiency for both engines; checks the per-context targets of §8 |
| XVI-T11 | Decode cost contract | `sky-cli` (`sky check`) | `cost bytes_per_token <= 18.2 GB` checked from the plan (SPINE D12) | Contract fails if a plan change adds weight bytes per token |

## 11. Measurements that decide

1. **The M5 Max raw-read bandwidth** (G0, IV-T2). Below ≈ 575 GB/s, a ≥ 1.05× plain-decode ratio is out of reach, the M5 Max ratio target stays ≥ 1.0× (SPINE §3), and the claim becomes "≥ 95% of the raw read".
2. **The dependent-boundary cost on the M5 Max.** At 1 µs, fusion is worth ≈ 2.5% on 32B. At 3 µs or more (MLX's per-op figure on the M2 Pro, R3 §7.E; ≈ 3.5 µs effective once Part XXII's fill term is included, SPINE §2), it is worth ≈ 3–7%.
3. **mlx-lm's KV-stream efficiency on the M5 Max at 4K–32K** (XVI-T10). It sets the long-context bounds of §8; above ≈ 0.6, the 8K and 16K targets are out of reach.
4. **The 0.6B step time on the M5 Max** for Sky and mlx-lm. It decides the small-model claim and the draft cost c of Part XXI.
5. **GEMV efficiency with fused prologue and epilogue vs without.** If fusion costs more than 1% of streaming efficiency on the large shapes, the norm moves back to a separate kernel on those shapes only.

## 12. Risks and negative results

- **Plain decode may be ≈ 1.0× MLX on the M5 Max.** MLX already streams at about 96% of the planning bandwidth there (§2.1). This is the expected worst case, not a failure: the decode story rests on speculation, small models, long context, energy and determinism (SPINE D18).
- **Fused prologues can cost bandwidth.** Extra registers and loads in a GEMV that must keep 16 bytes per lane in flight may reduce streaming (R6 §B2). Mitigation: the scheduler measures fused against unfused per shape ([Part IX](part09-scheduler.md)).
- **"Fuse the small ops" is a small lever on large models.** An outside fused add+RMSNorm kernel for MLX was 0.73× (slower) at 1 × 4,096 because dispatch dominated, and it was worth 0.5–2.5% of a real prefill (mlx#4259, R3-issues D8). Our budget agrees: about 2.5% on 32B at 1 µs per boundary, ≈ 3–7% with the fill term (§4). The plan does not depend on it.
- **Runtime-compiled MSL can change with the OS.** The Metal compiler ships with macOS, so a macOS update can change speed or bits. Tuning entries and bit-reproducibility are keyed by OS and compiler version ([Part X](part10-autotuner.md)).
- **Long-context gains are bounded by the KV share** (§8): ≈ 1.08× at 8K even with a 2× attention advantage. If mlx#4077 lifted MLX's attention well above 53% on the M5 Max, the bounds shrink further, and the long-context story rests on 32K+ and the labeled int8 KV mode ([Part XV](part15-attention.md) §8.2).
- **Negative result carried forward:** command replay alone does not speed up dense 32B decode. llama.cpp measured no gain (#14570), and R3 measured 0% at real sizes (R5 §9.5, R3 §7.F). Replay is in the plan for small models, energy and the GPU-resident loop.

## 13. Disagreement with the spine

1. **Plain decode, short context (SPINE §3: "≥ 1.05× mlx-lm, i.e. ≥ 95% of the raw-Metal read").** The two halves of this target are equivalent only if the M5 Max's raw read is ≥ about 575 GB/s (94% of spec). MLX already reaches ≈ 515 GB/s there (§2.1), and Part XXII's model of Sky's plan gives 28.3 tok/s at the 534 GB/s planning bandwidth. We keep "≥ 95% of the measured raw read" as the commitment. "≥ 1.05× mlx-lm" becomes conditional on G0's bandwidth measurement.

   Resolved: the spine now says ≥ 1.05× mlx-lm on the M2 Pro (stretch 1.10×; modeled 1.09–1.11× on 8B) and ≥ 1.0× on the M5 Max until the raw read is measured, since ≥ 1.05× there needs ≈ 575 GB/s; ≥ 95% of the raw-Metal read is the bandwidth target, and G5 exits on the same terms (SPINE §3, §6).
2. **Long context.** An earlier draft of this part disagreed with "≥ 1.2× at 8K–32K". The spine now states per-context targets (≥ 1.05× / 1.10× / 1.2× at 8K / 16K / 32K), derived in [Part XV](part15-attention.md) §8.2. This part uses them, and there is no remaining disagreement.
