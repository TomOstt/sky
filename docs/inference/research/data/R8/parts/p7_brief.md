---

## Architect brief (R8: algorithms that change the physics)

Full notes: `$S/research/R8-algorithms.md`. Reproducible math and data are in `$S/research/R8-work/` (`model_math.py`, `spec_model.py`, `entropy_sim.py`, `configs/`).

### Findings

**1. Qwen3-32B arithmetic** (VERIFIED: config.json and safetensors headers; the parameter total matches HF exactly at 32,762,123,264).
- Shape: 64 layers, hidden 5120, 64 Q / 8 KV heads, head_dim 128, FFN 25,600, vocab 151,936 untied, QK-norm, no MTP head. Native context 32K (131K with YaRN ×4).
- Streamed per token: 31.98 B params.
- Bytes per token:
  - MLX 4-bit g64: **17.99 GB** (the 18.43 GB file is verified);
  - Qwen's official MLX-4bit, which uses g128: 16.99 GB;
  - GGUF Q4_K_M: 19.32 GB (exact `use_more_bits` recipe verified against the 19.76 GB file);
  - Q4_0: 18.26 GB.
- Decode roofline at 614 GB/s (Apple-verified spec): **34.1 tok/s** at zero context, 30.5 at 8K, 23.1 at 32K.
- KV cache: 256 KiB per token in fp16.
- Compute: prefill is 63.5–66.7 GFLOP/token at 1–4K prompts; decode is 64 GFLOP plus 2.1 MFLOP per context token.

**2. The owner's baseline is already infeasible.** "MLX 35 tok/s" is above the 4-bit roofline. Realistic MLX is about 24–29 tok/s (EXPERIMENTAL). A plain 60 tok/s needs ≤2.56 bits/weight, and Qwen3 collapses at 3-bit PTQ (AWQ W3 g128: MMLU 81.2 → 54.9; VERIFIED, arXiv 2505.02214).

**3. Speculation is the only lossless way past the roofline. Its value on Apple is set by verification cost, not by the algorithm.**
- Generalized speedup: S = τ / (1 + γc + γβ).
- A dense model's "free" verification width is k\* = R·b/16, where R is the ridge point.
  - M5 Max neural accelerators (≈65–70 TFLOPS FP16, secondary source): k\* ≈ 32 at 4.5 bpw.
  - SIMD only: k\* ≈ 8.
  - Check: Sequoia chose tree size 128 on A100 with fp16 weights; the formula gives 153.
- Today's measurements are far from that:
  - MLX qmv with 4 rows costs **2.43×** one row (R3, M2 Pro), i.e. β ≈ 0.48.
  - An MLX EAGLE-3 prototype got **1.05×** (M3 Ultra).
  - LM Studio + MLX on an M1 Pro **lost 11–39%**.
  - MTP gives about +24–28% on M3 Max (independent test), while MTPLX claims 2.2–2.7× on M5 Max. That gap is consistent with the NA making verification cheap.

**4. Public speculators exist** (VERIFIED repos and configs).
- Qwen3-32B: `RedHatAI/Qwen3-32B-speculator.eagle3` (1 layer, 32K draft vocab), `AngelSlim/Qwen3-32B_eagle3`, a RedHat *thinking* variant, plus Zhihu, Nex, Snowflake LSTM and others.
- Qwen3-8B: RedHat EAGLE-3, **DFlash** (15-token block drafter; τ = 5.9 on HumanEval at T=0) and P-EAGLE; Tengyunw and AngelSlim EAGLE-3.
- **Published Qwen3-32B acceptance is modest.**
  - RedHat, T=0.6, k=3: τ = 2.29 (HumanEval), 2.49 (GSM8K), 2.15 (CNN/DM); 3.08 on GSM8K at k=7.
  - AngelSlim, 2 tokens: τ = 1.86–2.01.
- No MLX-format speculators exist. mlx-lm has no EAGLE or MTP support: it drops Qwen3.5 MTP weights, and the community PRs were closed.
- llama.cpp master supports draft-simple, EAGLE-3, MTP, DFlash, DSpark and n-gram drafting.

**5. Modeled result for Qwen3-32B on M5 Max** (EXPERIMENTAL; 27.3 tok/s plain baseline):
- EAGLE-3 with β ≤ 0.05 and a cheap draft (c ≈ 0.03): **1.7–2.2×, i.e. 47–59 tok/s**.
- With today's MLX-like β ≈ 0.48: **0.7–1.0×**.
- An average of 60 tok/s needs τ ≥ 3.5–4.5, i.e. a DFlash/DSpark-class drafter trained for Qwen3-32B (none is public).

**6. Exactness is a real differentiator** (VERIFIED).
- mlx-lm #1470 (M5 Max; Qwen3-32B-6bit + Qwen3-0.6B, T=0): speculative output diverges from plain greedy at token 91.
  - Cause: exact bf16 logit ties combined with batched vs sequential reduction order.
  - Reproduced on three model pairs. MTP PRs saw roughly 1 divergence per 150–190 tokens.
  - MLX resolved it with a documentation-only PR (#1592).
- mlx-lm and llama.cpp both accept a draft only on an exact match with an *independently sampled* target token.
  - This preserves the distribution but gives lower acceptance than Leviathan/Chen rejection sampling (Σp·q instead of Σmin(p,q)).
- Fix: batch-invariant kernels plus keyed Gumbel sampling (as SGLang ships; Daliri et al. 2024 prove drafter-invariance).
  - Result: speculative output is **byte-identical to Sky's own non-speculative output at any temperature**.
- Cost on NVIDIA serving stacks: +34% (SGLang) to +62% (Thinking Machines). It should be small at batch-1 bandwidth-bound decode, because padding M wastes ALU, not bytes (EXPERIMENTAL).
- Never promise identity with MLX or BF16: BF16 vs FP16 greedy outputs diverge on 49–100% of prompts (arXiv 2609.26621).

**7. Quantization numbers** (VERIFIED).
- Qwen3-32B Wiki2 perplexity / MMLU:
  - FP16 7.61 / 81.2
  - W8 7.60 / 81.3 (lossless)
  - GPTQ W4 g128: 7.86 / 80.6
  - AWQ W4 g128: 8.55 / 78.0
  - RTN W4 per-channel: 38.5 (never use per-channel 4-bit)
  - W2: collapse.
- Qwen3-8B: SmoothQuant W8A8 −0.7 MMLU; W4A8 −11.5.
- GGUF KL (Mistral-7B): Q4_K_M median KLD 0.0075 with the top token differing at 5.8% of positions; Q6_K at 3.9%.
- Codebook and trellis 2-bit methods are ALU-bound: QTIP reached 34–43% of bandwidth at batch 1 on a GPU with 3.5× Apple's ALU per byte. On Apple they fall short of their byte savings.
- Apple's own server model uses **ASTC 3.56 bpw decoded by texture hardware at "effectively zero compute cost"** (AFM 2025). This is a moonshot worth one microbenchmark.

**8. KV cache, sparsity and MoE.**
- 8-bit KV is production-safe (Apple ships it). It is worth 1.05× at 8K and 1.18× at 32K, but only with a fused kernel: mlx-lm's unfused version *slowed* decode 6–22% (#1651).
- Eviction (H2O, SnapKV) and KV Prediction change outputs. Eviction fails multi-turn use (SCBench). KV Prediction lost 32% relative TriviaQA accuracy at 2.36× TTFT FLOPs.
- There is no Apple sparse hardware (LIKELY). TEAL-style activation sparsity gives ≤1.5×, changes outputs, and conflicts with speculation.
- MoE verification is *not* free: verifying 4 tokens streams 2.6–3.0× the expert bytes for Qwen3-30B-A3B and gpt-oss-120b.
- MoE decode efficiency today: MLX ≈59% and BaseRT ≈64% of roofline (Qwen3.6-35B-A3B on M5 Pro; my EXPERIMENTAL estimate of active bytes). Fusion is the lever.

**9. The 2026 landscape moved.**
- The Qwen 27B line (3.5/3.6/3.8) ships an MTP head and a 3:1 DeltaNet:attention stack. It streams 20% fewer bytes than Qwen3-32B, with 4× less KV.
- MTPLX claims 65–88 tok/s on M5 Max (UNVERIFIED).
- BaseRT (arXiv 2607.19438) beats MLX prefill by up to 3.9× on M5 Pro.
- These, not just MLX, are the competitors.

### Recommendations (in order)

1. **First kernel milestone:** a batch-invariant quantized GEMV/GEMM family where decode (M=1) and verify (M ≤ 16–32) share one reduction order, with time(M=8) ≤ 1.1× time(M=1) on M5 Max, plus fixed-split attention. Everything below depends on it.
2. **Ship EAGLE-3 chain speculation** for Qwen3-32B and Qwen3-8B using the RedHat weights:
   - exact rejection sampling;
   - optional keyed-Gumbel "deterministic mode";
   - then prompt-lookup and suffix drafting;
   - then trees sized to the measured k\*.
3. **Make "speculative == plain, byte for byte" a CI invariant.** The mlx-lm #1470 script is a ready-made adversarial test. Market it.
4. **Support hybrid DeltaNet + MTP models** (Qwen3.6/3.8-27B) with exact state rollback.
5. **Opt-in modes with a quality certificate:**
   - fused 8-bit KV;
   - a compiler search for mixed-bit weights under a KL budget;
   - INT8-activation prefill.
   Report KL against BF16 for each.
6. **Do not invest in:** KV eviction, KV Prediction, weight or 2:4 sparsity, 2-bit codebook PTQ, BitNet.
7. **Restate the targets.**
   - Decode: plain 29–31 tok/s (bit-exact, ~1.15× MLX); speculative 45–55 typical, ≥60 on input-grounded or greedy work.
   - Prefill: 750–950 tok/s in FP16; 1,600 only with INT8 under a KL budget (about 80% of INT8 peak).
   - RAM: 20 GB only at ≤4.4K context with fp16 KV (≤8.4K with 8-bit KV).
   - TTFT: 420 ms only with prefix-cache hits; a cold 700-token prompt takes ~0.7–0.85 s.

### Open questions (need measurements)

1. The real MLX numbers on M5 Max for Qwen3-32B (decode and prefill).
2. Sustained NA FP16/INT8 TFLOPS and the small-M cost curve. These decide k\* and β.
3. τ for the RedHat EAGLE-3 drafter vs Qwen3-0.6B/1.7B drafts on Sky's workloads at T = 0.6/0.7, in thinking mode, with rejection sampling vs exact-match acceptance.
4. The batch-invariance overhead on Apple.
5. KL of MLX 4-bit g64 vs BF16 for Qwen3-32B (absent from the literature).
6. ASTC texture decode throughput.
7. MTP and DeltaNet rollback cost on Qwen3.6/3.8-27B.
