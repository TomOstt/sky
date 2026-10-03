# Part XVII — Prefill optimization

Prefill runs the prompt through the model in chunks of many tokens. It is a sequence of matrix multiplications that are bound by compute, not bandwidth. This part covers:
- the GEMM skeleton, built on Metal Performance Primitives (MPP) `matmul2d` on every chip, with group-scaled weights dequantized into threadgroup tiles;
- the fused prologues and epilogues of the prefill layer;
- chunk pipelining;
- the numerics contract for prefill;
- the measured headroom on the M2 Pro and the M5 Max;
- the int8 / W4A8 mode and its quality cost;
- the targets.

Prefill attention is designed in [Part XV](part15-attention.md). Time to first token, which adds load, compile and cache effects on top of prefill, is in [Part XVIII](part18-ttft.md).

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R6 §B5). Physics numbers come from [`roofline.py`](roofline.py).

## 1. Summary

1. **One GEMM path for every chip: MPP `matmul2d`.**
   - On the M2 Pro it reaches 5.42 TFLOPS fp16, 95% of the ALU peak, and beats MPS (4.41) and MLX's own GEMM (5.0) (R6 §B5, R3 §7.D).
   - On the M5 it is the only route to the neural accelerators (NA): plain `simdgroup_matrix` code leaves them idle (R6 §A2).
   - Tiles must be ≥ 64 × 64 on the M5 Max (R6 §A2).
2. **Quantized weights need our own loader.** MPP has no native format for group-wise fp16/bf16 scales and biases (MLX affine, GGUF). The skeleton dequantizes each weight tile into threadgroup memory, then multiplies, as MLX's NAX kernels do (R3-kernels §d). A native `half × int4b` variant with per-group post-scaling is the alternative the tuner measures. `simdgroup_matrix` is a fallback for systems without MPP, never used on the M5.
3. **Headroom differs by chip generation:**
   - On the M2 Pro, mlx-lm's end-to-end 8B prefill runs at 3.3 TFLOPS against 5.42 for MPP. llama.cpp already reaches about 4.7 TFLOPS of matmul work on the same machine (R4 §7.A, R5 §8.A). So ≥ 1.25× is credible on M1–M4.
   - On the M5 Max, MLX's published pp512 (731 tok/s) is about 71% of the NA peak. BaseRT gained 1.44× at a 128-token prompt but only 1.02× at 2,048 (R7 §8.1). So the M5 commitment is parity, with 1.1–1.2× from tile efficiency and from removing fixed per-prompt costs (§9).
4. **The first generated token comes out of the prefill program.** mlx-lm runs the last prompt token through a separate decode step: about 35 ms for Qwen3-32B on the M5 Max, about 5% of a 512-token TTFT and about 20% of a 128-token one. It also synchronizes the CPU and clears the buffer cache after every 2,048-token chunk (R4 §2.1). Sky submits all chunks back to back and computes the last row's logits and sample inside the last chunk (§6).
5. **int8 activations are the only route to 1,600 tok/s, and they cost quality.** W4A8 cost Qwen3-8B 11.5 MMLU points; W8A8 cost 0.7 (R8 §4.2). It is a labeled mode with a KL report, never the headline (§10).

## 2. Where prefill time goes

Qwen3-32B prefill costs 62.6–66.7 GFLOP per prompt token for 128–4,096-token prompts (`roofline.py`, R8 §1.4). Causal attention grows with prompt length (`roofline.py` `prefill_flops_per_token`, as tabulated in [Part XV](part15-attention.md) §6.2):

| Prompt length | Share of prefill FLOPs in attention |
|---:|---:|
| 512 | 0.9% |
| 2,048 | 3.3% |
| 4,096 | 6.4% |
| 8,192 | 12.1% |
| 32,768 | 35.5% |

So short and medium prompts are a GEMM problem. Long prompts are partly an attention problem ([Part XV](part15-attention.md)).

**Measured on the M2 Pro** (16-core GPU, 5.73 TFLOPS fp16 = fp32 peak):

| What | Rate | Share of peak | Source |
|---|---:|---:|---|
| mlx-lm, Qwen3-8B 4-bit, end to end (pp512 / pp2048) | 219.5 / 217.7 tok/s = 3.3 TFLOPS of linear-layer work | 58% | R4 §7.A `[VERIFIED]` |
| llama.cpp, Qwen3-8B Q4_0, end to end (pp512, FA on) | 312 tok/s ≈ 4.7 TFLOPS of matmul work | ≈ 82% | R5 §8.A `[VERIFIED]` |
| MLX 4-bit GEMM `qmm_t` alone, Qwen3-32B MLP shapes | 4.38–4.44 TFLOPS | 77% | R3 §7.D `[VERIFIED]` |
| MLX fp16 Steel GEMM alone, same shapes | 4.84–5.03 TFLOPS | 85–88% | R3 §7.D `[VERIFIED]` |
| MPP `matmul2d` fp16, 4096³ | **5.42 TFLOPS** | **95%** | R6 §B5 `[VERIFIED]` |
| MPP `matmul2d` half × int4b, 4096³ | 4.90 TFLOPS | 86% | R6 §B5 `[VERIFIED]` |
| `MPSMatrixMultiplication` fp16 | 4.41 TFLOPS | 77% | R6 §B5 `[VERIFIED]` |

mlx-lm loses about a quarter of its GEMM kernel's rate somewhere else: 3.3 end to end against 4.4 in isolation. That gap, plus the kernel gap (4.4 vs 4.9–5.4), is the M1–M4 headroom. llama.cpp shows the end-to-end part is recoverable on this exact machine: the same model at the same prompt length runs 1.42× faster than mlx-lm, even though its own GEMM kernel is no faster than MLX's (R5 §8.E: 4.28 TFLOPS for Q4_0 at n = 512).

**Measured on the M5 Max (40-core GPU):**

| What | Rate | Source |
|---|---|---|
| NA fp16 matrix peak | ≈ 64–65.5 TFLOPS | R6 §A2 `[LIKELY: two independent secondary measurements]` |
| NA int8 × int8 | ≈ 122 TOPS | R6 §A2 `[LIKELY/EXPERIMENTAL: one source]` |
| `simdgroup_matrix` (ALU) path | ≈ 14.6–15.5 TFLOPS | R6 §A2 `[LIKELY]` |
| MLX `mx.matmul` 1024 × 4096 × 4096: fp16 / bf16 / q4 | 46.7 / 36.7 / 35.0 TFLOPS | MLX #4411 via R6 `[EXPERIMENTAL: one measurement]` |
| MLX fp16 GEMM at 4096-class shapes | 52.7–59.6 TFLOPS | mlx#3211 (maintainer) via R3-issues B4 `[VERIFIED]` |
| MLX, Qwen3-32B 4-bit, pp512 | 731 tok/s ≈ 46 TFLOPS effective (731 × 63.0 GFLOP), ≈ 71% of NA peak | R5 §9.3 `[LIKELY: one community run]` |
| llama.cpp, Qwen3-32B Q4_K_M, pp512 | 667 tok/s | same run |

At the model's real shapes, MLX's quantized prefill reaches about 46 TFLOPS. That is well above the 35 TFLOPS of the small q4 benchmark and close to its fp16 GEMM rate. On the M5, large-prompt GEMM is "near-solved by MLX's NAX path" (R7 §8.3).

## 3. The GEMM skeleton: `mpp_qgemm`

One skeleton in `sky-kernel`: `mpp_qgemm` (`crates/sky-kernel/src/skeletons/mpp_qgemm.rs`; MSL template in [Part XI](part11-metal-backend.md) §9.4). Weight formats plug in as codecs (`crates/sky-kernel/src/codecs/`, [Part XII](part12-quantization.md)). It computes C[M × N] = A[M × K] · Wᵀ, where:
- M is the chunk's tokens;
- N is the output features;
- W is a quantized weight in its file format.

### 3.1 MPP `matmul2d` on every chip

Why one path:
- Metal 4 tensor ops compile at run time under MSL 4.0 and run on the M2 Pro, including half × half, bfloat × bfloat and half × int4b (R6 §B1, `[VERIFIED: measured, outputs exact or within 2.1e-5 relative]`).
- Apple: "The same code runs across Apple's entire GPU family from M1 to M5. On older GPUs without neural accelerators, TensorOps falls back to optimized shader implementations" (tech talk 111432, via R6 §A2, `[VERIFIED]`).
- On the M5, `simdgroup_matrix` code runs with "neural accelerator utilization … 0%" (same talk). A kernel written for M1–M4 would be about 4× slower on the M5. MPP is the only way to write one kernel for both.

The caveat: MPP lowers each `matmul2d` to a backend-provided function (`air.externally_defined`), so its code generation is opaque (R6 §A2). Sky controls the tiling, the loads and the epilogue. It does not control the inner MMA.

### 3.2 Tiles

R6's measured tile table on the M2 Pro, fp16 4096³ (`[VERIFIED: R6 §B5, microbench/results/gemm.txt]`):

| Tile (M × N per threadgroup), simdgroups | fp16 TFLOPS | half × int4b TFLOPS |
|---|---:|---:|
| 32 × 32, 1 simdgroup | 5.42 | 4.90 |
| 64 × 32, 2 simdgroups | 5.42 | 4.89 |
| **64 × 64, 4 simdgroups** | **5.38** | **4.88** |
| 128 × 64, 4 simdgroups | 5.04 | 4.75 |
| 64 × 32, 4 simdgroups (Apple's sample) | 4.46 | 3.88 |
| 128 × 128, 4 simdgroups | 4.11 | 3.82 |

On the M5 Max, Apple's sample 64 × 32 tile reaches only about half of the NA peak, and tiles of 64 × 64 or larger are needed (Anemll README, via R6 §A2, `[LIKELY]`). MLX's NAX kernels use 64 × 128 × 256 tiles for dense GEMM on `s`/`c`-class devices, and 64 × 64 × 64 for quantized `qmm_nax` (R3-kernels §d).

Decision:
- The tile family starts at **64 × 64 with 4 simdgroups**. On the M2 Pro it is within 1% of the best tile; on the M5 Max it is the smallest efficient size.
- The tuner searches 64 × 64, 64 × 128 and 128 × 64, a K-step of 64 or 128, and single or double buffering, per (chip, shape, format) ([Part X](part10-autotuner.md)).
- One rule is fixed: no split-K whose split depends on M (§7).

**Small prompts.** A 128-token chunk has two 64-row tiles in M. Occupancy comes from N: for Qwen3-32B gate‖up, 51,200 / 64 = 800 tile columns. Every GEMM of the model has at least 80 tile columns at N ≥ 5,120, so a 40-core GPU is filled for M ≥ 64. `[LIKELY: arithmetic]` Prompts shorter than 64 tokens are padded in M. Their cost is a few milliseconds of wasted MACs, not a correctness issue.

### 3.3 Group-scale dequantization into threadgroup tiles

MPP has no native representation for MLX's affine format: 4-bit codes plus one bf16 scale and one bf16 bias per 64 weights. It has none for GGUF's block formats either. The only natively scaled format is OCP-MX: E8M0 power-of-two scales per 32 elements, from OS 27 (R6 §A3, `[VERIFIED: MSL 4.1 §2.22.2.2]`). So the skeleton builds the weight operand itself.

**Default: dequantize into threadgroup memory, then multiply.** Per K-step of 64 (one group):
1. The threadgroup loads, for its BN weight rows:
   - BN × 32 bytes of codes;
   - BN scales and BN biases.
2. Each thread computes `s·q + b` in fp32 and rounds to fp16.
3. It stores the B tile (BN × 64 × 2 B: 8 KB for BN = 64) into threadgroup memory. The tile is double-buffered, so the next tile's loads overlap the current `matmul2d`.
4. Activations go straight from device memory into the left-input cooperative tensor, as in MLX's NAX path (R3-kernels §d), with the norm prologue applied on the way (§5).

Budget:
- Threadgroup memory is 32 KB per threadgroup on every Apple GPU through the M5 (R6 §A4). BN = 64 double-buffered uses 16 KB; BN = 128 double-buffered uses 32 KB, the whole budget.
- ALU work for dequantization is about 3 operations per weight, amortized over BM = 64 rows: about 2.3% of the MACs. On the M5 it runs on the shader ALUs, beside the NA. `[LIKELY: arithmetic]`

**Alternative: native int4 operand with post-scaling (OS 26.4+).** MPP accepts `half × int4b_format`, exact for the codes (R6 §B1). With a K-step equal to the group size (64):

```text
C[m, n] += s[n, g] · P[m, n] + b[n, g] · Σ_{k ∈ g} A[m, k],   where P = A[:, g] · Q[g, :]ᵀ (int4 codes)
```

This costs one elementwise pass over the accumulator tile per K-step (4,096 FMAs per 262,144 MACs for a 64 × 64 × 64 step, 1.6%). It saves the threadgroup-memory round trip.
- On the M2 Pro, MPP half × int4b ran at 4.90 TFLOPS without scales, 0.9× fp16 (R6 §B5).
- On the M5 Max its NA rate is unmeasured (R6 open questions). For comparison, half × MXFP4 runs at 0.69–0.78× of fp16 there (Anemll, R6 §A3).

The tuner measures both variants per chip. `[EXPERIMENTAL: XVII-T3]`

**Other formats:**
- **GGUF:** the same loader decodes Q4_0 (32-weight blocks, fp16 d), Q8_0, Q4_K and Q6_K blocks into the same fp16 tile.
- **MXFP4 (gpt-oss):** dequantized the same way, because the native MX path is slower than fp16 for compute on the M5 Max (R6 §A3).

**Operand dtype.** fp16 operands with fp32 accumulation:
- the M5 NA exposes no bf16 rate advantage, and MLX's bf16 GEMM runs at 36.7 TFLOPS against 46.7 for fp16 (R6 §A2);
- the mlx-community checkpoints store bf16 scales and biases (R4 §0), so dequantized weights are rounded to fp16.

Weight magnitudes fit fp16. Activations are a risk (§12): they are rounded to fp16 only after normalization (§5).

### 3.4 The `simdgroup_matrix` fallback

`sg_qgemm` ([Part XI](part11-metal-backend.md) §9.3) uses 8 × 8 `simdgroup_matrix` with fp32 accumulation, dequantizing into threadgroup memory like MLX's `qmm_t` and llama.cpp's `mul_mm` (R3 §3.4, R5 §3). It is used only where MPP is unavailable or measured slower:
- OS releases before Metal 4;
- any M1–M4 shape where the tuner finds it faster. llama.cpp found its own tensor-API kernels "~5% slower" on the M2 Ultra and equal on the M4 (R5 §3), but R6's measurement on the M2 Pro points the other way.

It is never used on M5-class GPUs.

Reference rates on the M2 Pro:
- `simdgroup_matrix` register-only peak: 5.44 TFLOPS half → half, 4.67 half → float (R6 §B5);
- MLX's Steel GEMM with this instruction: 85–88% of peak (R3 §7.D).

## 4. The prefill layer

```text
 chunk of M tokens, residual X[M × 5120] (+ per-row Σx² partials)
   │
   ├─P1  Q‖K‖V GEMM     prologue: RMSNorm rows (input_layernorm) from partials, round to fp16
   │                     epilogue (tile = one 128-wide head): QK-norm, RoPE(pos), write K,V into the KV cache; Q to arena
   ├─P2  attention      flash-attention on NA tiles, causal block skip, GQA-packed, fixed split size (Part XV)
   ├─P3  O GEMM          epilogue: + residual, Σx² partials per row
   ├─P4  gate‖up GEMM    prologue: RMSNorm rows (post_attention_layernorm); epilogue: SwiGLU
   └─P5  down GEMM       epilogue: + residual, Σx² partials per row
 last chunk only: final RMSNorm + LM head on the last row (decode skeleton, M = 1) + sampling → first token
```

Five kernels per layer per chunk, against 19 primitives per layer in mlx-lm (R4 §3.1). At prefill sizes dispatch cost is not the issue: a 512-token chunk of Qwen3-32B is about 0.5 s of GEMM on the M5 Max. Fusion matters here for a different reason: it removes intermediate writes and reads.
- MLX's prefill RMSNorm on a 5,120-wide row is a separate looped kernel that reads x twice (R3-kernels §c).
- Its SwiGLU writes two 25,600-wide intermediates per token.

An outside measurement of a fused add + RMSNorm for MLX found 1.07–1.21× on the fused operation at prefill shapes, but only 0.5–2.5% of a real prefill (mlx#4259, R3-issues D8). We expect fusion to be worth a few percent of prefill, not more. `[LIKELY]`

## 5. Fused prologues and epilogues

**RMSNorm prologue.** RMSNorm(x)·W = r_m · ((x ⊙ w_norm) · W), where r_m = rsqrt(mean(x_m²) + ε) is one scalar per token row. Three ways to fuse it:

| Option | Mechanism | Verdict |
|---|---|---|
| Apply r_m in the epilogue | multiply each output row by r_m after the matmul; feed x ⊙ w_norm to the NA | **Rejected**: x ⊙ w_norm is not normalized. Qwen-family residual streams can carry large activations, and rounding them to fp16 risks overflow (fp16 max 65,504) |
| Apply r_m in the prologue, from partials | the previous kernel's epilogue writes per-(row, tile) partial Σx²; the A-tile loader reduces them in fixed order and normalizes `x · r_m · w_k` before rounding to fp16 | **Chosen.** Same formulation as decode ([Part XVI](part16-decode.md) §3.5) |
| Separate RMSNorm kernel | one more dispatch and one more pass over the chunk's activations | fallback |

The residual stream itself stays in fp32 in the activation arena. Only normalized GEMM inputs are rounded to fp16. `[EXPERIMENTAL: XVII-T5 compares KL against an all-fp32 reference]`

**Q‖K‖V epilogue.** In prefill a GEMM tile of BN = 128 columns covers exactly one head (head_dim 128), so the epilogue can do what decode cannot ([Part XVI](part16-decode.md) §3.7):
- QK-norm: a 128-wide RMSNorm per row, with MPP's `reduce_rows` over the cooperative tensor (R6 §A3);
- RoPE: Qwen3's rotate-half pairs (i, i + 64) both lie inside the tile;
- the fp16 K and V rows go straight into the KV cache at positions `chunk_start + m`.

This removes the separate QK-norm, RoPE and KV-write kernels. It requires BN = 128 for the Q‖K‖V GEMM, which the tuner must respect.

**Residual and SwiGLU epilogues.** As in decode:
- the residual add writes the stream once and emits Σx² partials;
- gate‖up rows are interleaved in the relayout cache, so a tile holds gate column j and up column j and writes only silu(g) · u.

**The last row.** Only the last prompt token needs logits. The LM head runs as a one-row GEMV with the decode skeleton on the last chunk's final row, followed by the sampling finalize kernel ([Part XVI](part16-decode.md) §6). The prefill program therefore emits the first generated token.

## 6. Chunk pipelining

What mlx-lm does (R4 §2.1, `[VERIFIED: generate.py at 53b9af3]`):
- `DEFAULT_PREFILL_STEP_SIZE = 2048` (generate.py:48).
- After each chunk, `mx.eval([c.state for c in prompt_cache])`, a full CPU↔GPU synchronization (generate.py:441), then `mx.clear_cache()`, which frees MLX's buffer cache (generate.py:454).
- The last prompt token always goes through the decode-style `_step` (generate.py:456). A 2,048-token prompt costs one 2,047-token forward plus one full 1-token decode step.
- The speculative path uses its own 512-token chunks (generate.py:487).
- KV storage grows by concatenation in 256-slot steps, so a long prefill copies each layer's cache once per 2,048-token chunk (R4 §4.1).

What these cost:

| Cost | Size | Source |
|---|---|---|
| GPU idle per CPU synchronization | ≈ 100–125 µs per chunk | R6 §B3 `[VERIFIED: M2 Pro]` |
| The extra decode step for the last prompt token | ≈ 35 ms on Qwen3-32B / M5 Max; 26 ms on Qwen3-8B / M2 Pro | §2.1 of [Part XVI](part16-decode.md); R4 §7.A |
| Re-allocation after `clear_cache` | not measured | — |

The synchronizations are negligible. The decode step is not negligible for short prompts:
- MLX's 731 tok/s at pp512 is a TTFT-derived rate, because mlx-lm's `prompt_tps` includes the decode step on the last token (R4 §2.6). About 35 ms of its ≈ 700 ms TTFT is that step (≈ 5%).
- At pp128 the same step is ≈ 20% of the TTFT. `[LIKELY: arithmetic]`
- This explains part of BaseRT's short-prompt result. On the M5 Pro, its Qwen3.6-27B pp128 is 440 tok/s against mlx-lm's 305: TTFT ≈ 291 ms vs ≈ 420 ms. One 27B decode step there is ≈ 54 ms (1/18.4 tok/s), about 40% of the gap. `[LIKELY: arithmetic on R7 §8.1; BaseRT does not itemize its gains]`

What Sky does:
1. **Chunk programs from the static plan:** one per chunk-size bucket (for example 64, 512 and 2,048 tokens), with activation arenas planned ahead ([Part XIX](part19-memory.md)). There is no allocation and no cache clearing.

   mlx#4562 shows why a static plan matters. Raising MLX's command-buffer limits made a 30K-token prefill hold +88 GB instead of +8 GB, because in-flight buffers keep intermediates alive (R3-issues D2). A static arena bounds that by construction.
2. **All chunks committed back to back,** with no CPU wait between them. Pre-committed command buffers leave a 0.4 µs GPU gap, against ≈ 105 µs when the CPU waits (R6 §B3).
3. **The last chunk emits the first token** (§5), and the decode program is committed behind it ([Part XX](part20-persistent-execution.md)).
4. **KV written in place** into the preallocated cache by the Q‖K‖V epilogue. There are no growth copies.
5. **Chunk size is a tuning choice with no numeric effect,** because prefill is chunk-invariant (§7). The model-level tuner picks it per (chip, model, prompt length) (SPINE D11c):
   - large chunks fill the GPU with tiles;
   - smaller chunks bound activation memory and keep attention's K/V working set small.

## 7. Numerics: chunk invariance, and what prefill does not promise

The `inference` contract (SPINE D6) requires prefill results not to depend on the chunk size. The one exception is hybrid Gated-DeltaNet layers: their chunked-scan prefill sums in a different order than one-token decode, so they either use a slower recurrent-order prefill or run in a labeled mode that waives chunk invariance for those layers (measurement VII-M7 decides; [Part VII](part07-tensor-ir.md)). For the GEMM this means:
- each token row is computed by the same tile code, with the same K-loop order;
- no split-K whose partitioning depends on M or on the chunk;
- padding rows never influence real rows.

It is LIKELY that the NA computes each output element of a fragment with the same reduction regardless of its row. MPP's lowering is opaque, so this is a property we test, not one we can read in the code (XVII-T6). For attention it means fixed KV-block traversal and split boundaries aligned to chunk boundaries (R8 §3.3; [Part XV](part15-attention.md)).

**Prefill is not bit-identical to decode.** The prefill GEMM (NA tiles) and the decode GEMV (lane-split fp32 dot products) reduce in different orders. Exact speculation does not need them to agree:
- prompt tokens are always written by the prefill program;
- generated tokens are always written by the decode/verify family, which is batch-invariant ([Part XXI](part21-speculative.md) §3).

One feature does need care: **prefix caching over generated tokens.** On a later chat turn, the cached KV of the previous answer was written by decode kernels. A cold run would prefill those same tokens with prefill kernels, so a cache hit and a cache miss can differ in the last bits and, at a near-tie, in the output. Two remedies, for [Part XIV](part14-kv-cache.md) and [Part XVIII](part18-ttft.md) to choose between:
- **Canonical re-prefill:** after a turn, re-prefill the generated tokens in idle time and store prefill-written KV.
- **Tagged entries:** the cache records which kernel family wrote each span and serves a span only to plans that would write it the same way.

Neither is free. The prefill program must make the first possible (chunk-invariant kernels) and the second checkable (the plan records kernel families).

## 8. Attention for prefill

Designed in [Part XV](part15-attention.md). What prefill needs from it:
- **Flash-attention on NA tiles.** MLX's `attention_nax` uses BQ = 64, BK = 32, takes Q/K/V straight from device memory into NA registers, and skips fully masked causal blocks (R3-kernels §c).
- **GQA packing.** MLX's Steel and NAX prefill kernels re-stream K/V for each query-head threadgroup (`kv_head = head / gqa_factor`; R3-kernels §c). Sky's kernel reuses one K/V tile for the 8 query heads of a group.
- **Fixed split size aligned to chunk boundaries,** for chunk invariance (§7).
- **The new chunk's K/V come from the Q‖K‖V epilogue** (§5). The kernel never recomputes them.

Attention is 0.9% of Qwen3-32B prefill FLOPs at 512 tokens and 35.5% at 32K (§2). It matters for long prompts and RAG, not for chat turns.

## 9. Headroom

**M1–M4 (measured on the M2 Pro).**

| | mlx-lm | llama.cpp | Sky plan |
|---|---|---|---|
| Qwen3-8B pp512 | 219.5 tok/s | 312 tok/s (Q4_0 file) | ≈ 290 tok/s (≈ 1.32× mlx-lm) |
| Effective TFLOPS (linear layers) | 3.3 | 4.7 | ≈ 4.4 |
| Label | VERIFIED (R4) | VERIFIED (R5) | EXPERIMENTAL |

The Sky plan assumes:
- the dequant + MPP GEMM runs at ≈ 4.9 TFLOPS (R6's half × int4b rate, or fp16 at 5.42 less ≈ 3% dequant);
- non-GEMM time (attention, epilogue tails, the last-row head) is ≤ 10%.

This supports ≥ 1.25× on M1–M4 (SPINE §3). llama.cpp's 1.42× on the same machine shows the gap is real. The spine's second M1–M4 gate, ≥ 1.0× llama.cpp on its own GGUF file (D5), is tighter: this plan's ≈ 290 tok/s is below llama.cpp's 312, while [Part XXII](part22-performance-model.md) §8.2 models 337–376 tok/s with less non-GEMM time. `[EXPERIMENTAL: G5 decides]`

**M5 Max.**

| | Rate | TTFT at 512 tokens |
|---|---|---|
| MLX (community run) | 731 tok/s, ≈ 71% of NA peak, decode step included | ≈ 0.70 s |
| Sky at 80% of the 65 TFLOPS NA peak | 825 tok/s | ≈ 0.62 s (1.13×) |
| Sky at 85% | 877 tok/s | ≈ 0.58 s (1.2×) |
| Ceiling, 100% of fp16 NA peak | 1,033 tok/s | ≈ 0.50 s |

`[EXPERIMENTAL: the 80–85% efficiency is a target; arithmetic from roofline.py's 63.0 GFLOP/token at 512]`

BaseRT is the reality check: 1.44× MLX at pp128 but 1.02× at pp2048 on the M5 Pro (R7 §8.1, `[VERIFIED: arXiv 2607.19438]`). For long prompts we expect parity to 1.1×; for short prompts more, mostly from §6.

## 10. The int8 / W4A8 mode

The arithmetic:
- The M5 Max NA runs int8 × int8 at ≈ 122 TOPS, about 1.9× its fp16 rate (R6 §A2, `[LIKELY/EXPERIMENTAL: one source]`). On the base M5, a third party measured int8 at 2× fp16 and W4A8 prefill GEMMs 1.10–1.24× faster than MLX's `qmm_t_nax` (R3-kernels §d, scratchy#118, `[not verified]`).
- `roofline.py` gives 1,357 tok/s for Qwen3-32B pp512 at 70% of the int8 peak. The owner's 1,600 tok/s needs 100.8 TOPS, about 83% of the int8 peak.
- MPP has the operand types: char × char → int (OS 26.0) and char × int4b → int (OS 26.4) (R6 §A2, MSL 4.1 Table 7.3). MLX uses no integer `matmul2d` (R3-kernels §d).

The quality cost (arXiv 2505.02214 via R8 §4.2, `[VERIFIED]`):

| Qwen3-8B | Wiki2 perplexity | MMLU |
|---|---:|---:|
| fp16 | 9.71 | 74.7 |
| SmoothQuant W8A8 | 9.62 | 74.0 (−0.7) |
| GPTQ W4 g128 (weights only) | 9.96 | 73.4 |
| **SmoothQuant W4A8** | 12.5 | **63.2 (−11.5)** |
| SmoothQuant W4A4 | 3.4e4 | 24.8 (collapse) |

Int8 activations in prefill change the prompt's KV cache, and with it every later token. So this is a different-output technique.

Decisions:
1. **Not the headline, and not part of any same-quality comparison** (SPINE D5).
2. **`--prefill=int8` is a labeled mode** for 8-bit models (W8A8, MLX 8-bit or GGUF Q8_0 weights), where the measured cost is small. Each run reports KL against fp16 prefill and against BF16 on the R9 quality corpus. The budget is set per model ([Part XII](part12-quantization.md)).
3. **W4A8 on 4-bit models is EXPERIMENTAL** and gated on a quality result we do not have. Rotation-based activation quantization (QuaRot, SpinQuant) does better than SmoothQuant on other models, but no Qwen3 W4A8 rotation numbers exist (R8 §4.5, gap).
4. **The int8 rate itself is unverified on the M5 Max.** XVII-T8 measures sustained int8 `matmul2d` TOPS at Qwen3-32B shapes before any work on the mode.

## 11. Targets

SPINE §3 and G5 set the M1–M4 gate at ≥ 1.25× mlx-lm **and** ≥ 1.0× llama.cpp, which already prefills 1.35–1.42× faster than mlx-lm on the M2 Pro (R5). On the M5 Max the gate is parity:

| Target | Gate (must pass) | Plan | Condition | Measured by |
|---|---|---|---|---|
| Prefill, M1–M4 (M2 Pro) | ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp (same-file pairs, D5) | the model above gives ≈ 1.3× mlx-lm | Qwen3-8B 4-bit g64, prompts 512–8K | R9 S2, prefill = n_prompt / TTFT, n_prompt ≥ 2,048 headline |
| Prefill, M5 Max | ≥ 1.0× mlx-lm | 1.1–1.2× (80–85% of NA peak) | Qwen3-32B 4-bit g64, prompts 512–8K | same |
| Short prompts (128–512) | ≥ 1.1× mlx-lm TTFT | ≥ 1.2× | first token from the prefill program (§6) | R9 S2 at 128 and 512 |
| Prefill, int8 mode | reported separately, with KL | — | 8-bit models first | R9 S2 plus the quality corpus |
| Warm TTFT | follows prefill (SPINE: ≥ 1.15× better) | — | prompt length stated | [Part XVIII](part18-ttft.md) |

The headline is n_prompt ≥ 2,048, where per-request fixed costs are small (R9 §7.5). Short-prompt results are reported separately, so that the decode-step effect of §6 is visible and not hidden inside a single number.

## 12. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XVII-T1 | `mpp_qgemm` fp16 path with the tile family of §3.2 (with XI-T12) | `crates/sky-kernel/src/skeletons/mpp_qgemm.rs`, `sky-metal` (MSL 4.0 runtime compile, tensor support probe) | Dense fp16 GEMM | M2 Pro: ≥ 5.2 TFLOPS at 4096³ (MPP best is 5.42); M5 Max: ≥ 80% of the G0-measured NA peak at Qwen3-32B MLP shapes; results within 1e-3 relative of an fp32 reference |
| XVII-T2 | Quantized loader: dequantize MLX affine (2–8 bits, g32/64/128), Q4_0, Q8_0, Q4_K, Q6_K, MXFP4 into fp16 threadgroup tiles, double-buffered | the codecs' tile-dequant functions (`crates/sky-kernel/src/codecs/`) inside `mpp_qgemm` | Quantized GEMM | Dequantized tile bit-equal to `sky-quant`'s reference dequantizer; ≥ 0.9× of XVII-T1's fp16 rate at the same shapes |
| XVII-T3 | `half × int4b` variant with per-group post-scale (OS 26.4+; with XII-T11) | `crates/sky-kernel/src/skeletons/gemm_na.rs` | Second variant for the tuner | Within tolerance of XVII-T2; rate measured on the M2 Pro and the M5 Max; the faster one becomes the default per chip |
| XVII-T4 | `sg_qgemm` `simdgroup_matrix` fallback (with XI-T13) | `crates/sky-kernel/src/skeletons/sg_qgemm.rs` | Fallback GEMM | ≥ 85% of peak on the M2 Pro at 4096³ (MLX's Steel GEMM reaches 85–88%); never selected on gen ≥ 17 |
| XVII-T5 | Prefill layer with fused prologues and epilogues (§4–§5), including QK-norm + RoPE + KV write in the Q‖K‖V epilogue | `crates/sky-kernel/src/{prologues,epilogues}/`, `crates/sky-llm/src/prefill/plan.rs` | 5 kernels per layer per chunk | Logits and KV within tolerance of the TIR reference; KL vs an all-fp32 reference reported; `sky explain` lists 5 kernels per layer |
| XVII-T6 | Chunk-invariance test | `crates/sky-llm/tests/prefill_chunk_invariance.rs` (with [Part XV](part15-attention.md)'s `prefill_chunk_invariance` for attention) | Test on the M2 Pro CI runner and the M5 Max host | Prefill of a 4,096-token prompt with chunk sizes {64, 192, 512, 2,048, 4,096}: KV cache and last-row logits bit-identical |
| XVII-T7 | Chunk pipelining and first token from the prefill program | `crates/sky-llm/src/prefill/run.rs`, `sky-metal` | Back-to-back chunk submission; LM-head row and sampling in the last chunk | No CPU wait between chunks (GPU timeline shows < 5 µs gaps); TTFT at 128 tokens on Qwen3-8B / M2 Pro ≥ 1.2× mlx-lm |
| XVII-T8 | int8 microbenchmark, then the `--prefill=int8` mode | `research/microbench` (extended); `crates/sky-kernel/src/skeletons/gemm_na_int8.rs` (with XII-T13); `crates/sky-quant/src/act_quant.rs` | Sustained int8 TOPS at real shapes; W8A8 mode | Mode exists only if the measured int8 rate is ≥ 1.5× fp16; every run reports KL vs fp16 prefill and vs BF16 |
| XVII-T9 | Prefill benchmark matrix | `sky-bench`, `sky-evaluation` | pp128 / 512 / 2K / 8K on the M2 Pro (8B) and the M5 Max (32B), vs mlx-lm and llama.cpp | Gate targets of §11 |

## 13. Measurements that decide

1. **NA fp16 throughput by tile on the M5 Max** (IV-T2): `matmul2d` TFLOPS at 64 × 64, 64 × 128 and 128 × 128, sustained for 10 minutes. This decides whether 80–85% of peak is reachable, and so whether the M5 plan is 1.1–1.2× or parity.
2. **Dequant overhead on the NA path:** the XVII-T2 rate against XVII-T1. If dequantization into threadgroup memory costs more than 10%, the int4b post-scale variant (XVII-T3) is the default.
3. **The int8 rate on the M5 Max** (XVII-T8). Below 1.5× fp16, the int8 mode is dropped.
4. **The chunk-invariance test on NA hardware** (XVII-T6). If NA results depend on row position, prefill must use fixed tile-aligned chunk boundaries, and the invariance claim narrows accordingly.
5. **Short-prompt TTFT decomposition for mlx-lm on the M5 Max:** the decode-step share at pp128 and pp512. This tests the §6 explanation of BaseRT's short-prompt gain.

## 14. Risks and negative results

- **Large-prompt GEMM on the M5 is near-solved by MLX.** BaseRT gained 2% at pp2048 and Mojo trails MLX by 4–15% (R7 §8.3). Our M5 plan assumes 80–85% of a peak MLX already reaches 71% of. If the NA peak is lower than the secondary sources say, the plan shrinks toward parity.
- **MPP is opaque and young.** Its lowering is invisible to us (R6 §A2). The NA path had real bugs:
  - an Xcode 26.5 mis-compile of the NAX GEMM;
  - NAX split-K bf16 garbage on the M5 Max;
  - `gather_qmm` corruption (R3-kernels §d);
  - a downstream MLX fork that disabled NAX on gen 17 entirely.

  Mitigation: correctness tests on every OS update, and tuning entries keyed by OS and Metal compiler version ([Part X](part10-autotuner.md)).
- **fp16 activations can overflow.** Normalized GEMM inputs are safe; the SwiGLU output feeding the down projection is not normalized. A finiteness check in validation mode (SPINE D6) catches overflow. The fallback is bf16 operands for that one GEMM, at the bf16 NA rate. `[EXPERIMENTAL]`
- **Thermals.** Sustained laptop prefill fell 19% in llama.cpp's M5 measurements (R6 §M5 Max table). Every prefill number is reported from the 10-minute sustained scenario too ([Part XXIII](part23-benchmarks.md)).
- **Negative result carried forward:** int8 prefill is not same-quality. W4A8 lost 11.5 MMLU points on Qwen3-8B (§10). The owner's 1,600 tok/s is not a fp16 target.

## 15. Disagreement with the spine

One small inconsistency rather than a disagreement. SPINE §3 gives two M1–M4 prefill targets: "≥ 1.25× mlx-lm" (first table) and "≥ 1.1× on M1–M4" (second table). §11 treats ≥ 1.1× as the gate and ≥ 1.25× as the plan. The measured evidence (llama.cpp at 1.42× mlx-lm on the same M2 Pro; MPP at 5.42 TFLOPS) supports the higher number.

Resolved: the spine now says ≥ 1.25× mlx-lm **and** ≥ 1.0× llama.cpp on M1–M4 (SPINE §3, G5); §11 uses that gate.
