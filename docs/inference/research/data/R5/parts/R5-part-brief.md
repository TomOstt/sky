
---------------------------------------------------------------------------------------------------

## 14. Architect brief (R5 — llama.cpp / ggml on Apple Silicon)

Notes: `$S/research/R5-llamacpp.md`; raw data `$S/research/R5-raw/`. llama.cpp @81e39ad (2026-10-01).
Our machine: M2 Pro 16-core GPU, 200 GB/s spec, ≈183 GB/s achievable (R3 stream test).

**Findings (VERIFIED in code unless marked)**
1. *Execution.* Every token the ggml graph is re-encoded into two command buffers: the calling thread
   encodes and commits the first max(64, 10 %) nodes, one GCD worker encodes the rest. Each buffer has
   one concurrent-dispatch encoder; a byte-range hazard tracker emits `memoryBarrier(buffers)` only on
   real overlaps; a reorder pass hoists independent nodes. No ICBs. Graph *build + allocation* are
   skipped while topology is unchanged (`can_reuse`; KV view padded to 256). Each token ends with a host
   sync; sampling is on the CPU by default (GPU samplers exist, off).
2. *Kernels.* 100 % hand-written MSL, specialized by function constants, compiled at runtime from
   embedded source (4.8 s once per install, then 15–50 ms via the OS shader cache). Decode GEMVs read
   GGUF blocks directly; the Q4_0 kernel pre-scales activations by 1/16, 1/256, 1/4096 so a nibble needs
   one AND, folds −8 into Σy, reuses activations across 4 rows, and reduces with one `simd_sum`. Its
   layout is unchanged since July 2023 (#2188). Prefill uses fp16 `simdgroup_matrix` 64×32 tiles. The
   Metal-4 `matmul2d` path (M5 Neural Accelerators) is on by default only for M5/M6/A19/A20 and only
   for mat-mat (prefill). Apple's own engineers wrote its current kernel (#20962).
3. *Fusion* is a fixed 9-pattern table (norm+mul(+add)/scale, add chains, MoE router/reduce, SSM/GDN).
   There is no GEMV prologue/epilogue fusion and no QKV or gate/up merge.
4. *Attention.* Decode uses FA-vec, split-K over 32 workgroups plus a reduce, with quantized KV handled
   in-kernel. (Q, NE) come from an offline table per GPU family, including M5 Max; this is the only
   tuned piece. Prefill FA skips fully-masked blocks.
5. *Memory.* Weights are mmapped and wrapped by `newBufferWithBytesNoCopy` (zero copy). Residency sets
   are kept wired by a 5 ms heartbeat. Without it macOS unwires GPU memory after about 1 s idle; issue
   #10119 shows 26 ms/token becoming 429 ms after a pause. Default context is the trained length,
   auto-fitted.
6. *Speculation.* Supported methods: draft model, EAGLE3, MTP, DFlash, DSpark, and five n-gram modes.
   Available in the server; off by default; 3 drafts.

**Numbers**
- *Ours (VERIFIED).*
  - Qwen3-8B Q4_0: tg128 **34.8 t/s** (33.4 with FA off; 35.95 in a later run); pp512 **312**.
    Qwen3-0.6B: tg128 247.5, pp512 3849.
  - The 8B reads 4.431 GB/token ⇒ **154 GB/s = 77 % of spec ≈ 84 % of achievable**. The 0.6B reaches
    only 47 %.
  - Kernel level: Q4_0 GEMV **177 GB/s ≈ 97 % of achievable**; Q8_0 176, Q6_K 172, **Q4_K 147,
    MXFP4 138**. GEMM 3.7–4.9 TFLOPS.
  - One 8B token is **688 nodes ≈ 724 dispatches ≈ 542 barriers**. A two-point fit gives ≈64 µs per
    layer of non-bandwidth time (EXPERIMENTAL).
  - Ablations on tg128: graph reuse off −42 % (0.6B) / −5 % (8B); fusion off −0.5 % / −4.5 %;
    concurrency off −4 % / −2 %; reorder ≈0.
  - Cost of one call with n tokens, relative to n=1: Q4_0 **n=4 2.08×**, n=8 3.76×. Q4_K_M 2.48× /
    4.44×.
  - Speculation (8B target + 0.6B draft, greedy code prompt): **1.12×** (3 drafts, 80 % accepted),
    0.95× with 7 drafts, ngram-simple 0 drafts. An ideal verifier would give ≈2.3×.
- *Published.*
  - #4167, LLaMA-2 7B Q4_0, **M5 Max 40c: pp512 3220, tg128 119.9** (c1d0e7a, 2026-08-25). M5 Ultra:
    4945 / 179.1. M5 Pro 20c: 1621 / 66.3.
  - Same M5 Max chip, old vs new build: PP ×3.3, TG ×1.17.
  - **Headline workload (LIKELY, community run, 2026-09-27), M5 Max 40c, Qwen3-32B:**
    - llama.cpp Q4_K_M: **tg128 26.38, pp512 667**.
    - MLX 4-bit: **tg128 28.41, pp512 731**.
    - That is ≈82 % / 85 % of 614 GB/s.
  - MLX speculation on M3 Max: Qwen2.5-32B + 0.5B draft, **19.7 → 35.7 t/s (1.82×)**.

**Negative results**
- At batch 1 the GEMV is at the roofline: Q4_0 runs at 97 % of achievable bandwidth. Decode headroom
  over llama.cpp on large dense models is ≈15–20 %, not 2×. MLX already leads llama.cpp by ≈8 % on
  Qwen3-32B on M5 Max.
- **Command replay is not the fix.** The maintainer tried Metal graph/ICB reuse and found "no
  measurable improvement" (#14570). Our data put the fixed cost on the GPU side, at ≈15 barriers per
  layer.
- llama.cpp's reorder pass gives ≈0 for Qwen3 decode; fusion gives ≈4 %.
- Draft-model speculation barely pays with llama.cpp on Apple. The community reports no gain to 0.7×.
  n-gram speculation finds nothing on fresh text.
- The tensor API is not faster than simdgroup MMA before M5. It silently failed in some M5 builds until
  2026-09-01 (#27461).
- Not measured: energy, the host-sync bubble, M5 hardware, contexts ≥16k.

**Recommendations for Sky (Qwen3-32B 4-bit, M5 Max)**
1. **Decode, +15–20 % (EXPERIMENTAL).**
   - Reach Q4_0-class GEMV efficiency on a 4.5-bpw GPU layout, and keep it on any higher-quality
     format: llama.cpp's Q4_K drops to 80 %.
   - Cut each layer to ≈5–6 dispatches with few barriers:
     - QKV GEMV + q/k-norm + RoPE + KV store.
     - FA.
     - O GEMV + residual.
     - Norm + gate/up GEMV + SwiGLU.
     - Down GEMV + residual.
   - Keep sampling and token feedback on the GPU, so step N+1 is queued before step N ends (MLX does
     this).
   - Target ≥93 % of 614 GB/s, i.e. ≈31 t/s against 26.4 (llama.cpp) and 28.4 (MLX).
2. **Speculation is the 2× lever (EXPERIMENTAL→MOONSHOT).**
   - Build a multi-column quantized GEMV/GEMM for n = 2–16 that reads and dequantizes weights once and
     runs on simdgroup MMA or the M5 Neural Accelerators. The goal is a 4–8-token verify costing
     ≈1.1–1.3× one decode; llama.cpp pays 2–4×.
   - Pair it with a 0.6B/1.7B draft or MTP/EAGLE3 heads. MLX's 1.82× on M3 Max shows the headroom.
3. **Prefill/TTFT.** Write a tuned Metal-4 tensor-ops GEMM. On M5 Max, MLX is already 10 % ahead of
   llama.cpp at pp512.
4. **Copy what works:**
   - mmap + NoCopy + residency heartbeat (otherwise idle → 16× slower first tokens);
   - graph reuse;
   - quantized KV in the attention kernel;
   - per-GPU-family offline tuning tables.
5. **Benchmark hygiene:**
   - Pin commits on both sides. Check that `has tensor = true` on M5.
   - Compare equal bits per weight (Q4_0 ≈ MLX 4-bit g64 = 4.5 bpw).
   - FA on, fixed `-c`, ≥5 repetitions.
   - Report bytes/token and % of spec.
   - llama.cpp iterates fast (AI-assisted, crowd-tuned), so re-measure before claiming a win.

**Open questions:**
- Tensor-path kernel quality on M5 Max.
- Whether ≈64 µs per layer really disappears with fused layers.
- Energy per token.
- FA-vec at 16–32k context.
- An upstream, pinned Qwen3-32B run on M5 Max.

**Disk:** I am done with both 8B files. If one must go, delete `Qwen3-8B-Q4_K_M.gguf` (5.0 GB). Keep
`Qwen3-8B-Q4_0.gguf`: it is the canonical file in these notes and uses the same 4.5 bpw as MLX 4-bit.
