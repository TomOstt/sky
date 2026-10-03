# Part I — Executive conclusion

This part summarizes. Every claim in it is labeled, with its evidence, in the part it cites ([VERIFIED], [LIKELY], [EXPERIMENTAL] or [MOONSHOT]; see the [README](README.md)). Numbers are measured on our M2 Pro, published by others (cited), or computed by [`roofline.py`](roofline.py).

## The question

Can Sky become a compiler and runtime that runs open LLMs on Apple silicon materially better than MLX, mlx-lm and llama.cpp? Better here means the same weights, quantization, prompt and output quality, with:
- higher tokens per second;
- lower time to first token (TTFT);
- less memory;
- less energy.

Could it become valuable enough that rebuilding it would look worse to a company like Apple than acquiring it?

Ten research agents studied the code of tinygrad, MLX, mlx-lm and llama.cpp, Apple's documentation and the 2025–2026 literature, and measured all three competitors on the M2 Pro development machine. The full evidence is in [`research/`](research/). This part states the conclusions.

## The answer in one paragraph

**Yes, it is worth building, but not for the reason the original target table assumed.**

Dense, short-context, batch-1 decode on Apple GPUs is limited by memory bandwidth, and MLX is already close to that limit:
- On our M2 Pro, mlx-lm decodes Qwen3-8B at 91% of MLX's own streaming ceiling and 84% of what raw Metal can stream (R4, R6).
- On an M5 Max, the published MLX number for Qwen3-32B is 28.4 tok/s, about 83% of the 614 GB/s datasheet bandwidth (R5).

No compiler can make plain decode on an M5 Max reach the table's 60 tok/s for that model: it would need 1.09 TB/s of a 614 GB/s machine.

What Sky *can* do better, measurably and defensibly:
1. **Exact speculative decoding** whose verification is nearly free. One weight pass checks several drafted tokens, so effective speed goes past the bandwidth ceiling. The output is byte-identical to plain decoding at any temperature, which MLX cannot claim today.
2. **Long-context decode, small-model decode, prefill on M1–M4, cold start and memory.** Here the measured gaps range from 5–25% (long-context decode, growing with context) through 25–50% (prefill on M1–M4, small and draft models) to several-fold (cold start).
3. **Determinism, explanation and reach across chips** that a hand-written framework does not offer.

That is a real technology. It is not a guaranteed $100M asset (see "On the acquisition question" below).

## What the physics allows (Part IV)

Qwen3-32B in MLX 4-bit streams 17.99 GB of weights per decoded token, plus 256 KiB of KV cache per token of context.

| Owner's target | Verdict |
|---|---|
| MLX at 35 tok/s, Sky at 60 tok/s plain | **Impossible on an M5 Max.** Its ceiling is 33.9 tok/s at full datasheet bandwidth, about 29.5 at measured bandwidth. 60 tok/s plain is M5 Ultra territory: 1.2 TB/s, ceiling ≈ 60. |
| Sky at 60 tok/s *effective* | **Possible**, with exact speculation, cheap verification and a good drafter. Modeled at 47–59 tok/s typical with the public EAGLE-3 drafter for Qwen3-32B; ≥ 60 on input-grounded work. |
| Prefill 800 → 1,600 tok/s | MLX's published pp512 is 731 tok/s, about 71% of the neural accelerators' fp16 peak. The fp16 ceiling is about 1,030 tok/s. 1,600 needs int8 activations, which cost real quality on 4-bit models (−11.5 MMLU on Qwen3-8B with W4A8; R8). Honest target: ≥ 1.0× MLX on M5, where both engines use the neural accelerators (modeled upside 1.15–1.3×); on M1–M4 ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp, which already prefills 1.35–1.42× faster than mlx-lm on the M2 Pro. |
| RAM 24 → 20 GB | The weights alone are 18.4 GB. At 4K context with fp16 KV the floor is 20.0 GB; at 32K it is 27.5 GB. "20 GB" only means something with a context length. |
| TTFT 900 → 420 ms | A cold 700-token prompt needs ≥ 0.68 s even at 100% of fp16 peak. 420 ms is honest only on a prefix-cache hit. Targets: warm ≥ 1.15× (the prefill gain, plus Sky sampling token 1 inside prefill); repeated prefix ≥ 1.3× with both engines' prefix caches on; cold start (process to first token) ≥ 3×. |

## What the competitors already do (Part V)

| Engine | Strength | Weakness we measured or found |
|---|---|---|
| **MLX / mlx-lm** | Decode kernels at the bandwidth limit on large models; neural accelerators for prefill on M5. | |
| **llama.cpp** | Hand-written kernels at about 97% of bandwidth for Q4_0. | |
| **tinygrad** | | Its generic code generator reaches only about 10% of bandwidth on 4-bit decode (R2), and it has nothing for the M5. Its *design ideas* are worth copying (Part VIII); its kernels are not. |
| **BaseRT** (hand-written Metal 4) | | Only ties MLX on dense decode (R7). |
| **Monolith / mpk-apple** (Sept 2026, from the author of MPK) | Targets about 1.10× over the best of MLX and llama.cpp, a sign the opportunity is real. | |
| **Apple Core AI** (OS 27) | Apple's own on-device inference framework. | |

The measured weaknesses of MLX / mlx-lm:
- Its verification of several tokens costs nearly as much as decoding them one by one on M1/M2-class GPUs (β ≈ 0.5–1, R4), but only +48% for 5 tokens on an M5 Max (β ≈ 0.06–0.18, mlx#3791). There, Sky's verify-kernel advantage over mlx-lm's speculation is about 1.2×; byte-identical output is the differentiator (Part IV §8).
- Decode attention at 4K context streams its KV reads about 2× below bandwidth (R4; up to 3× in R3's microbenchmark).
- Small models run at 54.5% of bandwidth and are CPU-bound (R4).
- Weights are copied with `pread` rather than mapped (R3).
- Speculative output diverges from plain output (#1470), and the maintainers answered with documentation only (R8).

llama.cpp's speculation barely pays (1.12×, R5).

## What Sky should become (Part VI)

A tensor stack beside the scientific language, sharing its infrastructure and its habits: verifiers, spans, explanations with rule IDs, cost contracts.

```text
Sky source / Hugging Face / GGUF → TIR (typed tensor IR) → graph optimizer → global scheduler
   → autotuner (per chip, measured) → kernel IR (hand-designed skeletons + generated fusion)
   → MSL → static execution plan → GPU-resident runtime (exact speculation, GPU sampling)
   → sky explain: every kernel, every byte, every decision
```

Decisions, with the evidence for each in Part VI:
- **A sibling tensor IR, not a bigger SIR.** SIR is f64-only, rank ≤ 2, with no memory model (R1).
- **Our own stack emitting Metal Shading Language, not MLIR.** No production MLIR path reaches Apple's matrix hardware, and every competitive Apple engine emits MSL (R7).
- **Hand-designed kernel skeletons with generated fusion code.** Generic code generation loses badly on quantized decode (R2).
- **Static plans re-encoded directly each step**, with K steps in flight (one command buffer per step) and the GPU writing its own next inputs.
- **A new numerics contract, `inference`**: deterministic, batch-invariant, chunk-invariant (hybrid DeltaNet layers excepted; measurement VII-M7 decides). Sky's scientific `reproducible` default stays as it is.

## The plan (Parts XXIV–XXVI)

The plan is built as gates, each with a measured exit criterion and a kill criterion (Part XXVIII):

| Gate | Weeks | Content |
|---|---|---|
| G0 | 0–2 | A benchmark harness that survives hostile review, and baselines (Part XXIII). |
| G1–G4 | 1–14 | Metal foundation, the quantized GEMV family, the tensor IR with a reference oracle, model loading with an aligned zero-copy cache, tokenizer and templates, first tokens, then fusion with a static memory plan and replay. Exit: Qwen3-8B decode at or above mlx-lm on the M2 Pro, and the 0.6B at ≥ 1.3×. Part XXIV gives the first ten implementations with pseudocode. |
| G5 | 14–20 | Prefill on Metal 4 tensor ops (MPP `matmul2d` on every chip), the autotuner, the M5 Max targets. M5 Max measurements start as soon as the hardware exists (target G1). |
| G6 | 20–28 | Exact speculative decoding: EAGLE-3 with public Qwen3 drafters, draft models, prompt lookup; byte-identity tests in CI. |
| G7 | 20–40 | Breadth: Llama, Gemma, Mistral, Phi, gpt-oss, Qwen3-MoE, the Qwen3.6 hybrid models with multi-token prediction; GGUF; a server. |
| G8 | 30–44 | Proof: a public `sky-evaluation` repository, at least three chips, energy, and the "why faster" report. |

## What the owner needs to decide now

1. **Hardware.** The development Mac is an M2 Pro with 16 GB: no neural accelerators, and no room for a 32B model. The plan needs:
   - **an M5 Max**: 40-core GPU, 128 GB (the benchmark matrix needs it for 70B models and 128K contexts), reachable over SSH as a benchmark host. An M5 Ultra makes the "60 tok/s" demo possible even without speculation;
   - **a self-hosted macOS CI runner** for Metal tests. The current runner is Linux.
2. **The ten in-flight feature branches.** All of their work is uncommitted (R1). Recommendation:
   - commit each to its branch;
   - land the three that tensor authoring needs: arrays you can build, sizes from variables, reductions;
   - park the rest until G4.
3. **The numerics contract.** Accept that tensor programs run under `inference` rather than `reproducible`. A ROADMAP amendment is in Part VI §9.
4. **Openness.** Sky is Apache-2.0 today. Whether the inference stack is open, and when, shapes both adoption and the acquisition story (Part XXVII).

## On the acquisition question

We give no valuation. These are the honest facts that bear on it (Part XXVII):
- Apple already has MLX, Core ML and Core AI, Metal Performance Primitives, and excellent compiler teams. Any single kernel or trick in this report can be copied in months.
- What would take longer to copy is the *combination*:
  - exact, cheap speculation;
  - batch-invariant deterministic kernels;
  - per-chip tuned plans with measured data across M1–M5;
  - a compiler that explains every byte.
- Acquisitions of this size are usually of teams with adoption. The credible path is the owner's Milestones 5–7: beat MLX by 10–20% "without cheating", then 30–50% on several real workloads with less memory, then show an advantage MLX cannot recover by copying one kernel. All of it published and reproducible by outsiders.

## The strongest reasons this may fail (Part XXVIII)

- MLX is near the roofline on dense decode, and it improves quickly.
- Monolith, BaseRT and Core AI are all aiming at the same gap.
- Speculative gains depend on the workload and on drafters per model. A 60+ tok/s average on Qwen3-32B needs a drafter nobody has published.
- The work needs M5 hardware and sustained engineering. A solo developer with AI agents is fast at code but limited by hardware access and verification.

Each risk has a kill criterion tied to a gate, so the project learns early if it is wrong.
