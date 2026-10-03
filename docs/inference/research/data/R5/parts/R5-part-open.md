
---------------------------------------------------------------------------------------------------

## 13. Extrapolation to the headline workload and open questions

### 13.1 Qwen3-32B 4-bit on M5 Max — what llama.cpp should do (EXPERIMENTAL arithmetic)

* Qwen3-32B: 64 layers, d_model 5120, FFN 25600, 64 Q heads / 8 KV heads × 128, vocab 151,936, untied
  head. Weights read per decode token ≈ 31.2 B (blocks) + 0.78 B (LM head). At Q4_0 (4.5 bpw) with a
  Q6_K head: **≈ 18.2 GB/token** (Q4_K_M ≈ 19.3 GB). KV cache f16: 64 × 8 × 128 × 2 × 2 B =
  **256 KiB per token of context** (8 GiB at 32k).
* M5 Max: **614 GB/s** spec, up to 40-core GPU with a Neural Accelerator per core, "over 4x the peak GPU
  compute for AI compared to the previous generation" (Apple newsroom text saved by sub-search A,
  `$S/research/tmp-R5a/nr_m5promax.txt`; URL in §9). Ceiling at spec: 33.7 t/s; at a ~91 %-of-spec
  achievable stream (M2 Pro ratio): ≈30.7 t/s.
* My pre-data estimate from M2 Pro efficiency was ≈23–25 t/s for llama.cpp. **Published (LIKELY,
  §9.3): M5 Max 40c, llama.cpp b11146 Qwen3-32B Q4_K_M tg128 26.38 t/s, pp512 667; MLX Qwen3-32B-4bit
  tg128 28.41, pp512 730.9** (same machine, 2026-09-27). 26.38 × ≈19.2 GB ≈ 506 GB/s = 82 % of spec;
  MLX ≈ 85 %. So the M5 Max runs closer to its spec than our M2 Pro (77 %), and **MLX, not llama.cpp,
  is the decode leader to beat on the headline workload (≈+8 %)**.
* Ceilings at 614 GB/s: Q4_K_M ≈ 32 t/s, 4.5-bpw (Q4_0 / MLX 4-bit g64) ≈ 33.7 t/s. A Sky engine at
  ≈93–95 % of spec with Q4_0-class kernel efficiency ⇒ ≈31–32 t/s, i.e. **+18–21 % over llama.cpp
  Q4_K_M and +9–12 % over MLX** (EXPERIMENTAL). Beyond that, only fewer bytes (better ~3.5–4-bit
  formats at equal quality) or speculation can move batch-1 decode.
* Prefill on M5 Max is compute-bound and is where the Neural Accelerators matter; llama.cpp's tensor
  path is enabled there but self-described as immature (§3). TTFT = prefill + first decode + any
  cold costs (shader compile 4.8 s only on first run after install; mmap page-in of 18 GB from SSD on
  a cold page cache: several seconds).

### 13.2 Open questions (could not be settled here)

1. **M5-class behaviour**: per-kernel speed of the tensor-API mul_mm vs simdgroup path on M5 Max;
   whether `mul_mv_ext` / FA-vec could use the Neural Accelerators (today they do not; tensor FA is an
   open PR, #29570); an upstream, pinned-commit Qwen3-32B run (only a community run exists, §9.3);
   whether the M5 build in use actually has the tensor path on (it silently failed in some builds until
   #27461, 2026-09-01).
2. **Size of the per-token host bubble** (sync → sample → set_inputs → first command buffer committed).
   Needs Metal System Trace / GPU timestamps; not measured. #14570 says CPU encode is already masked;
   our D2 + two-point fit say the ≈64 µs/layer is therefore mostly GPU-side — a fusion prototype that
   halves dispatches/barriers per layer is the experiment that settles it.
3. **SLC effects** in the kernel microbenchmarks (E): a 31–33 MB matrix may partially hit the system
   level cache across repetitions; full-model numbers (A/B) are the trustworthy ones.
4. **Energy**: not measured (powermetrics needs sudo). llama.cpp has a 5 ms residency heartbeat thread
   and a CPU threadpool (`--poll 50` default) — worth measuring J/token against MLX.
5. **Long context**: FA-vec split-K behaviour at 16–32k context for Qwen3-32B (tuned rows exist for
   ≥16k on Apple10) — not measured here (16 GB machine, 8B only).
6. **Draft choice**: we measured a 0.6B draft only (1.12×). EAGLE3/MTP/DFlash heads for Qwen3 exist in
   llama.cpp (README examples) but were not tested (no downloads allowed).
