# The spine: decision record for this report

This is the decision record the report was written against. It fixes the decisions (D1–D21), names, numbers, targets, gates (G0–G9) and style that every part follows. The chapters cite it as "SPINE §3" or "D6".

It was revised as the research arrived. Where a chapter's evidence corrected it, the chapter's "Disagreement with the spine" section records the point, followed by "Resolved: …".

The evidence is in [research/](research/README.md); every physics number comes from [roofline.py](roofline.py).

## 1. What we are building

**Sky Inference**: the Sky compiler and runtime extended so that a transformer LLM, given as a Hugging Face checkpoint or as Sky source, is compiled into a hardware-specialized execution plan for Apple GPUs and run with as little CPU involvement as possible. It is driven by:
- `sky run hf://org/model "prompt"`;
- `sky compile` (an ahead-of-time `.skybin`);
- `sky bench llm`;
- `sky explain`.

The year-1 workload is local, batch-1 inference:
- dense and MoE text models;
- Qwen3 first;
- 4-bit to bf16 weights;
- contexts to 128K.

Out of scope for year 1:
- training;
- non-Apple GPUs;
- multi-device;
- multimodal;
- large-batch serving beyond a simple OpenAI-compatible server.

The headline comparison is Qwen3-32B with MLX 4-bit weights against mlx-lm on an M5 Max (40-core GPU). The weights are affine, group 64, with **bf16** scales and biases, as verified on mlx-community's Qwen3 files (R9). MLX's format also allows fp16 scales, so the descriptor records the scale dtype. llama.cpp and PyTorch MPS are secondary baselines.

## 2. The physics we accept (from `roofline.py`; refine with R6/R8 numbers)

- Qwen3-32B has 32.76 B parameters. A batch-1 decode step streams **17.99 GB** of weights at 4.5 bits/weight. All 64 layers plus the untied 0.78 B output head make 31.98 B weights; the embedding is a one-row gather. On top of that it streams **256 KiB of fp16 KV per context position** (64 layers × 8 KV heads × 128 × 2 × 2 B).
- **Decode ceiling at 614 GB/s:**

  | Context | Bandwidth assumed | Ceiling |
  |---|---|---|
  | 512 | full spec bandwidth | **33.9 tok/s** |
  | 512 | ~90% of spec (achievable) | ≈30.5 tok/s |
  | 32K | ~90% of spec | ≈20.8 tok/s |

  **The owner's table ("MLX 35 → Sky 60") is impossible as written:** 35 tok/s is above the datasheet ceiling for these weights, and 60 needs 1.09 TB/s.
- The honest form of "60 tok/s" is **effective** throughput from exact speculative decoding. Take i.i.d. acceptance α, draft length γ, a draft cost ratio of 0.02–0.05 and a verify overhead of 10%, on a base of 27.2 tok/s (90% of the 1K ceiling). Then:

  | Acceptance α | Draft length γ | Effective tok/s |
  |---|---|---|
  | 0.7 | 5 | 59–67 |
  | 0.8 | 5 | 74–84 |

  So 50–70 effective tok/s is plausible on chat and code.
- On a chip with neural-accelerator matmul throughput (assumed 64 TFLOPS fp16; replace with R6's number), one weight stream can verify about 30 tokens before compute dominates. On M4 Max, about 9. **Speculation turns the M5's idle matrix compute into tokens.** That is the thesis of the decode chapters.
- **Prefill** costs about 63 GFLOP per token at short prompts. That is roughly 1,000 tok/s at 100% of an assumed 64 TFLOPS, or ~710 tok/s at 70%. "MLX 800 → Sky 1,600" at fp16 would need more than peak. A 2× prefill is only possible through a faster arithmetic path:
  - INT8 activations on the neural accelerators, if the hardware has a faster int8 rate (R6), with the quality delta reported;
  - or if mlx-lm is far below peak (R3/R4 measure this).
- **MLX is already near the limit on dense decode.** R3 measured on the M2 Pro (MLX 0.32.3):
  - Achievable streaming bandwidth is 182–184 GB/s (89–90% of 204.8 theoretical).
  - Large decode GEMVs run at 91–97% of that.
  - A whole decode step at context 128, pipelined as mlx-lm does it, runs at 94–97% on both 32B and 8B shapes.

  So **plain short-context decode has only about 3–6% headroom.** It is table stakes, not the pitch. The measured openings:
  - **Small-M verification.** MLX costs ×1.9–2.0 at M=3 and ×4.4–4.8 for M = 6…31 on the M2 Pro. On M5 Max, contributor data give ×1.12 at M=3, ×1.48 at M=5 and ×2.25 at M=8 (mlx#3791). A 60-line test kernel reached ×1.30 at M=3.
  - **GQA decode attention at long context.** At 4K context MLX's extra KV reads stream about 2× below bandwidth end-to-end on 8B (≈ 97 GB/s; R4), and up to 3× in R3's 8-layer microbenchmark: K/V are re-read per query head. Its fused decode attention also handles at most 4 queries with GQA 8, so 5–8 fall back to unfused.
  - **Fixed framework cost.** About 3.5 µs per kernel (≈ 1,225 kernels per 64-layer token). It is hidden for 32B but caps a 0.6B draft model.
  - **Loading.** MLX reads weights with `pread` into new buffers, with no mmap. That affects load time and cold TTFT.
  - **Quantized prefill on non-NAX chips.** It runs 11–13% behind fp16 GEMM.

  MLX's M5 neural-accelerator path ("NAX", Metal 4 `matmul2d`, f16/bf16, no int8) runs prefill GEMM, prefill attention, MoE gather-GEMMs and Gated-DeltaNet. It does not run decode GEMV, decode attention, norms or RoPE (R3; Part V §13).
- **End-to-end mlx-lm on the M2 Pro (R4, VERIFIED):**

  | | Qwen3-8B 4-bit | Qwen3-0.6B 4-bit |
  |---|---|---|
  | Decode | 37.4–38.8 tok/s | 267 tok/s |
  | Bandwidth use | 91% of MLX's own 182.5 GB/s ceiling, but 84% of the **196 GB/s raw-Metal read** (R6) | 54.5% |
  | Dispatches per token | 693 (457 barriers) | 541 |
  | CPU vs GPU per token | — | 1.94 ms CPU vs ~3.5 ms GPU; LIKELY CPU-bound on an M5 Max |

  - 8B time split: matmuls run at 97.5% of MLX-achievable; the other 435 dispatches add 7.6%. A simple 4-bit GEMV in raw Metal streams at 183–191 GB/s (R6).
  - **Headroom for plain short-context decode is therefore about 1.05–1.14×** (fusion plus streaming closer to raw read). It is larger for small and draft models.
  - Prefill on 8B end-to-end: 3.3 TFLOPS (58% of peak). On the same GPU, MPP `matmul2d` fp16 GEMM reaches **5.42 TFLOPS (95%)**, against MLX's 5.0 fp16 and 4.4 for MLX's 4-bit GEMM (R6, R3). **Prefill headroom on M1–M4 is about 1.2–1.4×.**
  - mlx-lm's speculation on the M2 Pro is **slower at every setting (0.37–0.74×)**: verifying k+1 tokens costs (k+1) forwards on this GPU generation. Its greedy output diverged in 3 of 9 configurations.
  - On M5 Max (mlx#3791, 70B): verify +12% at 3 tokens, +48% at 5, +125% at 8; speculation took 18 → 24.5 tok/s.
- **M5 family (R6, VERIFIED specs):**

  | Chip | Bandwidth | Memory |
  |---|---|---|
  | M5 Max 40-core | 614 GB/s | 48/64/128 GB |
  | M5 Max 32-core | 460 GB/s | 36 GB |
  | M5 Pro | 307 GB/s | — |
  | M5 Ultra (shipped 2026-09-22) | 1.2 TB/s | up to 512 GB |

  - Neural accelerators: fp16 ≈ 65 TFLOPS and int8 ≈ 122 TOPS on M5 Max (LIKELY, secondary sources); ALU ≈ 16.6 TFLOPS.
  - A contributor measured ~530 GB/s achievable on M5 Max (R3/R4), so roofline.py plans with 0.87.
  - Metal 4 tensor ops (MPP `matmul2d`) compile at run time and run on the M2 Pro too, including half × int4 (R6, VERIFIED). MPP has no native group-scale/bias format (MLX/GGUF style), so dequantize tiles first.
  - A dependent kernel boundary costs ~1.2 µs of idle GPU time; an independent one 0.14 µs. With the ramp/fill cost of a small kernel (~2.3 µs, fitted in Part XXII), the effective cost is ≈ 3.5 µs per dependent kernel. ICB replay cuts CPU cost (about 4 µs for any number of dispatches, vs 72 µs per 1000 direct) but not GPU time.
  - **Megakernel: rejected.** There is no forward-progress guarantee; a grid barrier deadlocked beyond 48 resident threadgroups; and a global barrier costs the same as a kernel boundary (R6).
  - The GPU cannot submit command buffers. So the "GPU-resident loop" means pre-committed command buffers per token, GPU sampling, and an `MTLSharedEvent` to the CPU.
  - Apple announced **Core AI** (OS 27), an on-device inference framework (CPU/GPU/ANE, custom Metal kernels). It is a competitor and must appear in Parts V and XXVIII.
- **The real M5 Max baseline** for Qwen3-32B 4-bit (R5; LIKELY — one community run, 2026-09-27):

  | Engine | Decode (tg128) | Prefill (pp512) | Bandwidth use |
  |---|---|---|---|
  | MLX | **28.4 tok/s** | **731 tok/s** | ≈ 83% of the 614 GB/s spec (17.99 GB per token) |
  | llama.cpp Q4_K_M | 26.4 tok/s | 667 tok/s | — |

  The owner's "MLX 35 / 800" overstates the baseline. We re-measure at G0/G5 on our own M5 Max.
- **Speculation already exists in the competition:**
  - mlx-lm's speculative decoding reached 1.82× on an M3 Max (Qwen2.5-32B with a 0.5B draft: 19.7 → 35.7 tok/s; R5). Sky must beat that, not plain decode.
  - llama.cpp's speculation barely pays on Apple (1.12× at best on the M2 Pro) because its verify costs 2.1× at 4 tokens and 3.8× at 8 tokens (R5).
  - llama.cpp's maintainers found no gain from command replay (indirect command buffers, ICBs; #14570). Its fixed cost is GPU-side barriers (~15 per layer), not CPU encoding.
  - llama.cpp keeps weights resident with a 5 ms residency heartbeat. Without it, the first tokens after an idle pause are about 16× slower (#10119).
- **Others already trying to beat MLX on Apple (R7, VERIFIED).** Part V and Part XXVIII must name them:
  - **BaseRT** (July 2026; hand-written Metal 4 kernels). On dense Qwen3.6-27B Q4 on an M5 Pro it ties MLX on decode (18.1 vs 18.4 tok/s). Its prefill is 1.44× MLX at a 128-token prompt but only 1.02× at 2048.
  - **Monolith / mpk-apple** (Sept 2026, from the author of Mirage Persistent Kernel). An Apple megakernel compiler that aims for ≥ 1.10× over the better of MLX and llama.cpp.
  - **Mojo** (open-sourced Aug 2026). It reaches the M5's 16×16×16 MMA through undocumented `air` intrinsics, yet its W4A16 matmuls run at 0.85–0.96× of MLX.
  - **Core ML's** 8B demo reaches about 35% of bandwidth.
  - **ExecuTorch** deprecated its MPS backend and now delegates to MLX.

  Lesson: hand-written runtimes tie MLX on dense decode. The real openings are short-prompt prefill and TTFT, long-context decode, RAM, energy, and speculation.
- **Memory** for Qwen3-32B 4-bit:

  | Context | KV format | Total |
  |---|---|---|
  | 4K | fp16 | 20.0 GB |
  | 32K | fp16 | 27.5 GB |
  | 32K | int8 | 23.2 GB |

  The 24 → 20 GB target is a statement about non-weight overhead, KV format and context length. It must be stated with the context length.

## 3. Honest targets (replace the owner's table everywhere)

The owner's table rewritten with today's evidence, for Qwen3-32B 4-bit on an M5 Max:

| Metric | Owner's table | Real MLX today | Sky target | Note |
|---|---|---|---|---|
| Decode, plain | 35 → 60 | 28.4 tok/s | ~28–30 tok/s: ≥ 1.0× until raw-read bandwidth is measured; modeled 28.3 (Part XXII) | the hardware ceiling is 33.9 at spec, 29.5 at planning bandwidth |
| Decode, effective with exact speculation | — | Unproven on M5 Max. mlx-lm got 1.82× on M3 Max with a 0.5B draft. Its verify cost per drafted token is β ≈ 0.5–1 on M1/M2-class GPUs (R4) but only **β ≈ 0.06–0.18 on M5 Max** (mlx#3791). It supports draft models only (no EAGLE/MTP), and its output diverges from plain greedy (mlx-lm #1470) | **≥ 1.7× plain: 47–59 tok/s typical** with the public EAGLE-3 drafter (τ 2.3–3.1); **≥ 60 on input-grounded / greedy work** | A 60+ *average* needs τ ≥ 3.5–4.5: a DFlash-class drafter for Qwen3-32B, which nobody has published (G9: train one) |
| Prefill, pp512 | 800 → 1,600 | 731 tok/s | ≥ 731 at fp16 (neural-accelerator parity) | 1,600 needs int8 activations. W4A8 cost Qwen3-8B 11.5 MMLU points (R8), so it is not a headline path; possible only for W8 models under a KL budget |
| TTFT | 900 → 420 ms | — | warm ≥ 1.15× better (prefill gain, plus one saved weight pass: Sky samples token 1 inside the last prefill chunk, while mlx-lm runs an extra full step, about 20% of TTFT at 128 tokens; Part XVIII); cold ≥ 3× better; repeated prefix ≥ 1.3× with both prefix caches on | — |
| RAM | 24 → 20 GB | — | ≤ 0.9× at 32K context | int8 KV as a labeled mode |

Every target is relative to mlx-lm measured on the same machine with the same weights. Re-baseline at gate G0 and again at G5.

| Metric (Qwen3-32B 4-bit, M5 Max) | Target | Condition |
|---|---|---|
| Decode, plain, short context (≤ 1K) | M2 Pro: ≥ 1.05× mlx-lm (stretch 1.10×; modeled 1.09–1.11× on 8B, Part XVI). M5 Max: ≥ 1.0× until raw-read bandwidth is measured, since ≥ 1.05× needs ≥ ~575 GB/s raw read and MLX is at ~96% of the 534 GB/s planning figure. Modeled Sky plan: 28.3 vs MLX 28.4 tok/s (Parts XVI, XXII). Target ≥ 95% of the raw-Metal read bandwidth. Parity is table stakes: mlx-lm is at 84% of raw read and 91% of MLX's own ceiling on 8B (R4) | KV in the baseline's 16-bit type (mlx-lm uses bf16) |
| Decode, small / draft models (0.6B–1.7B) | ≥ 1.5× mlx-lm | mlx-lm is at 54.5% of bandwidth and CPU-heavy (R4); BaseRT is 1.33× on 0.6B (R4/R7) |
| Decode, plain, long context | ≥ 1.05× at 8K, ≥ 1.10× at 16K, ≥ 1.2× at 32K vs mlx-lm (GQA-aware, context-split decode attention). Bounded by MLX's measured ~53% KV-read efficiency: ≈ 1.08× / 1.14× / 1.24× max (Part XV) | 16-bit KV, as baseline |
| Verify cost (the speculative engine's core kernel) | Weight GEMVs: M ≤ 8 tokens at ≤ 1.3× the time of M = 1, on M5 Max and M2 Pro. Attention: free verify width q* = R·s/(2G) is ≈ 15 queries on M5 neural accelerators but ≈ 4 on shader ALUs, so long-context verify attention runs on the accelerators. int8 KV halves q*: KV format and speculation compete for the same compute (Part XV) | Qwen3-32B and 8B shapes |
| Decode, exact speculative | ≥ 1.8× mlx-lm plain; ≥ 1.2× mlx-lm's own speculative mode with the same draft. Most of Sky's speculative lead on M5 comes from exactness and from drafters mlx-lm cannot run (EAGLE-3), not from the verify kernel alone | greedy output bit-identical to Sky's plain decode; sampling distribution-identical |
| Prefill | ≥ 1.0× mlx-lm on M5 (both use the neural accelerators); on M1–M4 ≥ 1.25× mlx-lm **and** ≥ 1.0× llama.cpp, which already runs 1.35–1.42× mlx-lm on the M2 Pro | Headline at prompts ≥ 2,048 (pp2048, R9 protocol). pp512 is reported only to compare with published numbers |
| Prefill, W4A8 mode | reported separately, with perplexity/KL delta | only if the hardware int8 rate is faster |
| TTFT, warm | follows prefill (≥ 1.15× better) | prompt length stated |
| TTFT, repeated prefix | ≥ 1.3× better (≈ 2× at ≤ 8 new tokens) | prefix caches enabled on both sides. ≥ 5× holds only against an engine without prefix reuse, and is reported as such |
| TTFT, cold (process start → first token) | ≥ 3× better | weights in page cache; AOT `.skybin` |
| Peak memory | ≤ 0.9× mlx-lm at 32K, fp16 KV | honest metric from R9 (no mmap accounting tricks) |
| Energy per token | ≤ 0.85× mlx-lm plain decode | whole-system energy (SMC `PSTR`, or an external meter), sustained ≥ 10-minute interleaved runs; never chip-only counters |

## 4. Architecture decisions

**D1. A sibling tensor IR, not a bigger SIR.** Tensors get their own IR, **TIR**, in the new crate `sky-tir`. It is not the same thing as TVM's TensorIR; say so once in Part VII. TIR reuses SIR's infrastructure patterns:
- pass manager, verifier, text format, spans, explanations;
- `sky_types::poly::Poly` for symbolic sizes;
- `sky-diag`.

SIR gains one operation that invokes a compiled tensor program (`tensor.call @graph`), so Sky source can embed tensor code.

Why not extend SIR (R1, VERIFIED)? SIR is:
- f64-only, rank ≤ 2, ≤ 2^24 entries per array;
- without a memory model (no buffers, dtypes or devices);
- built around every array op running through `sky_interp::Value`;
- exhaustively matched, so one new op touched 23 files on an in-flight branch;
- bound by the three-engine byte-identical rule, which cannot hold between a GPU and an f64 oracle.

Extending it would destabilize scientific Sky and slow the critical path. Part VI must state this as a deliberate amendment of ROADMAP principle 2. The principle's intent (no special paths from syntax to machine code; capabilities as passes over an IR) is kept; the "single IR" letter is not.

The levels inside the tensor stack mirror `docs/design/ir.md`:

| Level | Holds | Crate |
|---|---|---|
| TIR graph (L1-like) | tensor types (dtype/storage, symbolic shape, layout/view) and ops | `sky-tir` |
| TIR strategy (L2) | storage formats (quantization), precision, algorithm variants (GEMV / skinny GEMM / GEMM / neural-accelerator GEMM, attention algorithm), KV format, speculation strategy | `sky-tir` |
| Kernel IR (L3) | scheduled kernels: explicit iteration spaces, tiles, memory spaces (device, threadgroup, simdgroup registers, cooperative tensors), barriers, buffer offsets | `sky-kernel` |
| Backends (L4) | MSL for Metal (new); a CPU reference interpreter for TIR (oracle) | `sky-metal`, `sky-tir` |

**D2. No MLIR.** We recommend replacing ROADMAP Phase 5 "GPUs through MLIR" with a native Metal backend that emits MSL text. The reasons (confirm with R7):
- there is no production MLIR path to Metal that reaches simdgroup matrices, Metal 4 tensor ops and the neural accelerators;
- it is a large C++ dependency;
- MLX, llama.cpp and tinygrad all emit or ship MSL directly;
- control over every byte is the product.

**D3. Metal access from Rust** goes through `objc2-metal` 0.3.2 (R6).
- Kernels are compiled at run time from MSL source (`makeLibrary(source:)`).
- Compiled code is reused through Metal's OS shader cache.
- Binary archives appear only inside explicit `.skybin` outputs (D19).
- The offline `metal` toolchain is optional and used only for AOT `.skybin` packaging.

**D4. Kernels are skeletons plus generated fusion code.** A small library of parameterized skeletons, written in L3 kernel IR (hand-written MSL first, migrated into L3 once it exists):
- quantized GEMV;
- skinny GEMM for M ≤ 32;
- GEMM through MPP `matmul2d` on every chip with macOS 26+ (5.42 TFLOPS, 95% of peak on the M2 Pro, R6). It uses the neural accelerators on M5, with `simdgroup_matrix` as the fallback;
- flash-decoding;
- flash-attention prefill;
- reductions;
- sampling.

The scheduler generates the prologue and epilogue code from fused L1 subgraphs. Elementwise and reduction kernels are generated whole. The autotuner searches skeleton parameters and fusion partitions.

**D5. Same quantization.**
- The headline comparison consumes the published checkpoint's exact quantized values.
- Sky may change the layout (repack, interleave, concatenate Q/K/V or gate/up rows) but never re-quantize. A test proves that the dequantized values are bit-equal.
- No lossless conversion *at equal size* exists between MLX 4-bit (4.50 bpw, bf16 scales and biases, about half the scales negative) and the GGUF 4-bit files (bartowski Q4_0 is 4.67 bpw because its head is Q6_K; Q4_K_M is 4.90 bpw). MLX → GGUF Q4_1 is lossless but 11% larger (R9).
- So **Sky loads both formats natively**, and every comparison is a same-file pair:
  - Sky vs mlx-lm on the MLX file;
  - Sky vs llama.cpp on the GGUF file.
  - Bits per weight and bytes per token are printed next to every speed.
- Anything else (W4A8, mixed bits, KV quantization) is a separately labeled mode with a quality budget.

**D6. Numerics for inference.** Sky's default (`reproducible`: f64, no FMA, the same bits everywhere) stays for scientific code. Tensor programs with f16/bf16/quantized storage use a new contract, `inference`:
- fp32 accumulation; FMA allowed;
- every reduction in a fixed order chosen by the kernel configuration;
- **run-to-run bit-reproducible** on the same chip, OS, Sky build and tuning entry. This is enforced in source: kernels are emitted with `#pragma METAL fp math_mode(safe)`, `contract(off)` and explicit `fma`, because `safe` alone still allows contraction (MSL spec §1.6.3; Part XI);
- **batch-invariant**: a sequence's logits do not depend on the batch, the verify width or speculation;
- **chunk-invariant**: prefill results do not depend on the chunk size. One exception: hybrid Gated-DeltaNet layers. Their fast chunked-scan prefill sums in a different order than one-token decode. They either use a slower recurrent-order prefill or run in a labeled mode that waives chunk invariance for those layers; measurement VII-M7 decides (Part VII).

The mode is `--numerics=inference`. Two further modes are proposed:
- `--numerics=portable`: the same bits across Apple GPU generations, by restricting tile choices to a canonical reduction order. EXPERIMENTAL; measure the cost.
- `--numerics=relaxed`, the existing name: nondeterministic shortcuts, used only if measured faster.

The reference interpreter (f32, f64 for checks) is the oracle, with tolerances for f16 kernels.

**Tuning and results.** The spec today guarantees "tuning changes speed, never results" (spec, "Hardware and tuning — Guarantees"). For tensor programs that holds only in `portable` mode:
- Under `inference`, tile shapes, K-splits and the opaque internal order of MPP `matmul2d` can change the last bits.
- So the contract says: results are bit-reproducible *given* (chip, OS/Metal compiler, Sky build, tuning entry), and any tuning entry stays within the contract's tolerance of the reference.
- `sky explain` records the entry used.
- `portable` restricts tuning to order-preserving parameters (no MPP, or fixed tiles), and the original guarantee holds.

Any f16/bf16/quantized storage, i.e. any TIR program, runs under `inference` by default. Scientific f64 code is unchanged.

**libm caveat for scientific Sky.** Transcendental functions (`exp`, `sin`, `log`, …) call the platform libm today (R10). So "the same bits everywhere" holds for arithmetic and linear algebra, but not for transcendentals across OS versions until Roadmap A's F10 (one pure-Rust libm) lands (Part III §11).

Three other scientific-Sky rules do not carry over to tensor programs (R1). The tensor rules replace them; they are not exceptions:
- **"Every non-finite value is an error."** Kernels may use −inf internally (attention masks, softmax). A finiteness check on outputs (logits, hidden states at layer boundaries) is a validation mode, not a per-op cost.
- **"No broadcasting between arrays."** TIR has explicit broadcast views. The language surface allows broadcasting only by matching size names (D8).
- **"Engines byte-identical."** Replaced by:
  - deterministic per (device, build, tuning entry);
  - tolerance against the f32 reference;
  - greedy-token agreement with mlx-lm and HF on fixed prompts.

**D7. Two front doors, one L1.**
- First, a Rust model builder in `sky-llm` (`arch/qwen3.rs`, …) emits L1 directly. This is the fast path to first tokens.
- Second, models written in Sky source (`models/qwen3.sky`), once the front end has tensor types.
- A test requires both to produce identical TIR **after Part VIII's canonicalization passes**. Sky source spells some ops as primitives that the builder emits as composites, and composites have fixed decompositions, so recognition is exact.

**D8. Language surface (proposal; edition-gated).** Sky already parses `Tensor[B, T, D] unit`. The proposal adds:
- storage formats: `x: Tensor[T, 5120] in f16`, `W: Matrix[25600, 5120] in q4(group: 64)`;
- broadcasting only by matching size names (`w: Tensor[D]` times `x: Tensor[T, D]` is unambiguous; an ambiguous match is an error with both spellings, in the spirit of `*` vs `@`);
- reductions over a named axis: `mean(x * x, over: D)`;
- records for weights, if Sky lacks them;
- value-less declarations as model weights bound at run time (Sky's existing "input" rule).

The Tensor IR chapter owns the details and must respect spec style ("intuitive, reject ambiguity").

**D9. Runtime shape.**
- **Weights.**
  - mlx-community safetensors files are *unaligned*: data starts at offset ≡ 6 mod 8, and no tensor is even 4-byte aligned. On the M2 Pro an unaligned `uint` load silently reads the rounded-down word (R9, VERIFIED). Zero-copy over those files is therefore impossible.
  - Sky builds a **page-aligned relayout cache** (`.skyweights`) once, keyed by the source file's sha256. It is lossless and proven bit-identical after dequantization, and it holds the repacked layout too. Sky then mmaps it into no-copy Metal buffers. The build time counts in cold start.
  - GGUF (32-byte aligned) is mapped directly by default. Routing GGUF through the relayout cache (for horizontal fusion, K-quant decode layouts, or the M5 `int4b` prefill path) is EXPERIMENTAL and decided by measurement (Part XII §12).
  - A residency heartbeat or residency set keeps the weights wired across idle gaps (R5).
- A static memory plan per (model, max context, batch, chip) with arena offsets, so there is zero allocation in the hot path; KV is preallocated.
- A GPU-resident decode loop: the GPU writes the next token and position into a uniform buffer, and stop conditions are checked on the GPU. The static plan is *directly encoded* each step, which costs tens of µs and hides behind the GPU, with **K steps in flight, one command buffer per step**. Metal 4 command reuse still re-encodes, so it is no cheaper. ICBs are used only where the GPU must choose the command count (stop, adaptive γ) or where CPU wake-ups cost energy (Part XX).
- GPU sampling with a counter-based RNG (Philox), so seeded sampling is reproducible.
- Detokenization on the CPU, asynchronously.
- **A persistent megakernel is rejected for production** (R6: no forward-progress guarantee, and a grid barrier costs as much as a kernel boundary). Only a G9 feasibility spike remains.

**D10. Speculation.**
- **Order of work:**
  1. The batch-invariant verify GEMV and fixed-split attention (R8's first milestone). Everything else depends on them.
  2. EAGLE-3 with public weights:
     - `RedHatAI/Qwen3-32B-speculator.eagle3` and `AngelSlim/Qwen3-32B_eagle3`;
     - RedHat EAGLE-3 / DFlash for Qwen3-8B;
     - converted to Sky's format (none exist for MLX).
  3. A draft model (Qwen3-0.6B / 1.7B) and prompt-lookup / suffix drafting.
  4. Draft trees sized to the measured free-verification width k\* ≈ R·b/16 tokens. The theory gives ~32 on M5 Max neural accelerators (realistically 16–24 after tile efficiency, Part XXI) and ~8 on plain GPU ALUs; a chip with spare compute verifies more drafted tokens per weight pass.
  - **Byte-identity requires one kernel family for decode and verify.** If verification runs wide on the neural accelerators, decode must use the same arithmetic path, or byte-identity holds only for ALU-width verification (M ≤ ~8). This trades decode speed against verify width, and is decided by measuring the accelerator GEMM at M = 1 (Part XXI).
- **Exactness, as a CI test:**
  - With batch invariance (D6) and keyed Gumbel sampling (seed, position), speculative output is **byte-identical to Sky's own plain output at any temperature**. Daliri et al. 2024; SGLang ships it.
  - mlx-lm cannot claim this: its output departs from plain greedy at token 91 on Qwen3-32B with a 0.6B draft on M5 Max (#1470); its answer was a docs-only PR (#1592).
  - The #1470 script becomes our adversarial test.
  - Standard rejection sampling (accept with min(1, p/q)) is offered too. It accepts more than mlx-lm's and llama.cpp's exact-match acceptance at temperature > 0 (Σmin(p,q) vs Σp·q).
  - Never promise bit-identity with MLX or with a bf16 reference. bf16 and fp16 greedy outputs diverge on 49–100% of prompts (R8).
- **Hybrid models.** Qwen3.6/3.8-27B pairs Gated DeltaNet linear attention with an MTP head; it reads 20% fewer bytes per token and needs 4× less KV than Qwen3-32B, and MTPLX claims 65–88 tok/s on M5 Max (UNVERIFIED). Sky must support these models with exact rollback of the recurrent state. They are the second headline after Qwen3-32B.

**D11. Autotuning in three levels, on `sky-tune` and `sky-hw`:**
- (a) kernel: BEAM over skeleton parameters;
- (b) block: fusion-partition search, pruned by the cost model and then measured; all layers of a model share shapes, so one block is tuned and the result applies to all layers;
- (c) model: speculation tree shape, KV format, prefill chunk size.

Optimization levels:
- `-O1` heuristics only;
- `-O2` uses the cached DB;
- `-O3` tunes missing entries (minutes);
- `--autotune=overnight` runs a full search.

**D12. Explainability is a product feature.** `sky explain` on a model prints, per decode token:
- every kernel, with predicted vs measured time, bytes and FLOPs;
- the memory plan, byte by byte;
- why each fusion was made or refused.

This extends the existing `--explain` and cost contracts. A memory/bandwidth contract (`cost bytes_per_token <= 18.2 GB`) is checked by `sky check`.

**D13. Benchmarks are a separate deliverable.**
- `sky bench llm` in the repo, plus a public `sky-evaluation` repository at gate G8 with:
  - pinned competitor versions and model hashes;
  - the protocol of R9 §7;
  - the JSON results schema of R9 §8.
- Nothing projected is ever shown as measured.
- One external harness timestamps every token for all engines, because the tools' own numbers measure different things (R9).
- **Metric definitions:**
  - TTFT is measured from token ids, with the idle gap stated.
  - Prefill = n_prompt / TTFT, with prompts ≥ 2048.
  - Decode = (N − 1) / (t_last − t_first), at stated context depths.
  - Peak RAM = the maximum rise in system memory used (wired + anonymous − purgeable + compressed) over a pre-start baseline. Per-process footprint and RSS miss memory-mapped no-copy weights.
  - J/token = whole-system energy (SMC `PSTR`) over loops ≥ 60 s. Chip counters undercount by about 2.4×.
- **Correctness gates before any speed claim:**
  - tokenizer ids identical;
  - on identical weights, KLD(Sky ‖ other engine) ≤ 0.1 × KLD(other engine ‖ BF16);
  - ≥ 99% top-1 agreement.
- **Run conditions:**
  - interleaved randomized blocks;
  - fresh reboot, AC power, High Power Mode, quiet-machine gate.
- **Energy as a selling point.** Part of mlx-lm's energy is its Python loop: about 7 W of CPU during 0.6B decode (R9). Sky's GPU-resident loop removes that, and energy per token is a fair place to win.

**D14. Hardware access is a prerequisite the owner must decide.**
- The dev Mac is an M2 Pro with 16 GB. It runs models up to ~8B and has no neural accelerators.
- Required:
  - an M5 Max (40-core GPU, **128 GB**) reachable over SSH as a benchmark host. The benchmark matrix needs 128 GB (70B models, 128K contexts), and so does any M4 Max used as a third chip;
  - a self-hosted macOS CI runner for Metal tests, separate from the dev laptop.

**D15. IP hygiene.**
- Learn from MLX, llama.cpp and tinygrad (all MIT); write our own kernels.
- Any port follows ROADMAP "Building on existing work" (notices, THIRD_PARTY_NOTICES.md).
- Patent consultation for novel techniques is the owner's call. The report identifies candidates and gives no legal conclusions.

**D16. The CPU side (Roadmap A) is secondary.**
- Only cheap fixes from R10 and fixes that also help tokenizer and sampling code.
- No LLVM backend now (confirm with R7/R10).

**D17. In-flight feature branches.**
- R1 found that all ten `worktree-agent-*` branches have **zero commits**: the work is uncommitted in `.claude/worktrees/*`. All ten compile, but three have stale test targets (tuples, readable output, ODE events).
- Commit each worktree's state to its branch so nothing is lost.
- Land the two that tensor authoring needs first, in this order: reductions, then arrays-you-can-build (in-place `store`, `arange`, `concat`). They share 12 files (Part II).
- Sizes-from-variables only changes the checker and cannot express `heads·head_dim`. Commit it, then land it after G4, rebased on symbolic sizes. TIR keeps its own symbolic domain in `sky-tir`, with bounds and sizes that may be 0; `sky-types` is unchanged (Part VII).
- Park the other seven, branch kept, until after G4.
- Freeze new scientific features during G1–G4, except bug fixes.

**D20. One owner per concern in the fusion pipeline.**
- *Legality* (what may fuse) lives in `sky-tir` `passes/legality.rs` (Part VIII).
- *Profitability* (what pays) lives in `sky-kernel` `schedule/` (Part IX), which calls the legality code.

Open language question for the owner: the keyword-argument spelling. The proposals use `over: D`, while the uncommitted readable-output branch uses `sep=`. Pick one spelling for the whole language (Part VII).

**D21. Accepted refinements from the chapters.**
- *Verify fallback.* Part XXI's ≤ 1.15× at M = 4 is a fallback target when M = 8 at ≤ 1.3× is not reached on the shader ALUs.
- *Norm statistics in verify plans.* Part XXII's rule: verify plans store norm statistics (producer partial sums) at M ≥ 6 instead of recomputing them per simdgroup.
- *M1–M4 prefill.* Part XVII's conservative plan (≈ 290 tok/s on the M2 Pro for Qwen3-8B) is below llama.cpp's measured 312. Part XXII's model gives 337–376. The gate stays at ≥ 1.0× llama.cpp, because llama.cpp proves it is reachable. G5 measures which plan holds.

**D19. Caches obey non-negotiable 3** ("caches use a data-only format").

| Artifact | Treatment |
|---|---|
| `.skyweights` relayout cache | Data. Allowed. |
| Plan cache | Stores the execution plan **as validated, bounded data**: kernel choices, parameters, offsets. It never stores MSL or binaries. MSL is regenerated deterministically on load; Metal's OS shader cache reuses compiled code (llama.cpp's later starts take 15–50 ms that way, R5). |
| Metal binary archives | Appear only inside explicit `sky compile` outputs (`.skybin`), which are build artifacts like `sky build` executables, never in implicit caches. |
| Tuning DB | Stays in the existing data-only text format. Its keys are extended for GPU kernels. |
| Hugging Face cache | Uses symlinks. `sky-hub` reads it as an external store; `sky-project`'s no-symlink rule applies to project imports only. |

**D18. Decode parity is table stakes; speculation is the pitch.** MLX's dense decode is at 94–97% of measured bandwidth (R3). The decode chapters lead with the engine that changes tokens per weight pass:
- a verify GEMV that stays ≤ 1.3× of M = 1 up to M ≈ 8;
- GQA-aware attention for 2–16 queries and long contexts;
- a static dispatch schedule with zero per-token graph work, which matters most for the draft model;
- GPU-resident sampling, accept/reject and rewind.

Single-kernel heroics are not the story. The megakernel is rejected for production: ≤ 3% honest upside for dense 32B (R3), and no forward-progress guarantee on Metal (R6). It remains only as a G9 spike.

## 5. Crates (new and extended)

| Crate | Role | New / existing |
|---|---|---|
| `sky-tir` | TIR: tensor types (dtypes incl. quantized blocks, N-d symbolic shapes on `Poly`, views) and ops; verifier, text format, CPU reference interpreter (f32/f64); tensor passes: canonicalize, view folding, composite recognition (rms_norm, rope, attention, swiglu), horizontal fusion, fusion partitioning, memory planning | new |
| `sky-ir`, `sky-lower` | one new SIR op `tensor.call`; lowering of tensor-typed Sky code into TIR (after G4) | existing, extended later |
| `sky-egraph` | block-level rewrite search over TIR (later) | existing, reused |
| `sky-kernel` | L3 kernel IR, skeleton library, index simplifier, cost model hooks | new |
| `sky-metal` | L4: MSL emission, Metal runtime (device, buffers, residency, pipelines, command replay, timestamps) | new |
| `sky-quant` | quantization formats as data: MLX affine, GGUF k-quants, MXFP4, int8; pack/unpack/repack; quality tools | new |
| `sky-llm` | model runtime: config → architecture builder, safetensors/GGUF loading, tokenizer, chat template, sampler config, KV manager, prompt cache, generation loop, speculation | new |
| `sky-hub` | Hugging Face downloads and cache (shares `~/.cache/huggingface/hub`) | new |
| `sky-hw`, `sky-tune` | GPU facts (cores, family, neural accelerators, measured bandwidth), kernel tuning DB | existing, extended |
| `sky-cli` | `run hf://`, `compile`, `bench llm`, `explain`/`profile` for models | existing, extended |
| `sky-serve` | OpenAI-compatible server (year 1, late) | new |

## 6. Gates (relative weeks; re-plan after G0)

| Gate | Weeks | Exit criterion (all measured, all reproducible) |
|---|---|---|
| **G0** ground truth | 0–2 | mlx-lm and llama.cpp baselines and Metal microbenchmarks on the M2 Pro, in JSON; roofline calibrated with measured bandwidth; owner decision on M5 Max and CI hardware |
| **G1** Metal foundation | 1–5 | `sky-metal` works; Sky's Q4 GEMV (M = 1…16, batch-invariant) ≥ mlx `quantized_matmul` on every Qwen3-8B/32B shape on the M2 Pro, and ≥ 88% of the **raw-Metal read bandwidth** (196 GB/s on the M2 Pro; R6) on the large shapes |
| **G2** tensor L1 + oracle | 1–6 | L1 tensor types/ops and the reference interpreter; the Qwen3 builder; Qwen3-0.6B logits match HF transformers (fp32) within tolerance |
| **G3** first tokens | 5–9 | `sky run hf://mlx-community/Qwen3-8B-4bit` on Metal (unfused); tokenizer exact on a test corpus; greedy output matches mlx-lm for 128 tokens on 20 prompts (near-tie divergences explained); decode ≥ 0.7× mlx-lm |
| **G4** parity | 9–14 | decode fusion (≤ 6 kernels per layer), static memory plan, replay, GPU sampling; 8B decode ≥ 1.0× mlx-lm on the M2 Pro; 0.6B decode ≥ 1.3× (overhead-bound). Prefill is measured at G4 and gated at G5 (Part XXV) |
| **G5** lead on M5 Max | 14–20 | GEMV ≥ 90% of achievable bandwidth; GEMM via MPP `matmul2d` on every chip (`simdgroup_matrix` fallback); flash attention; autotuner v1. GGUF loading for the Qwen3 files (needed for the same-file llama.cpp comparison; moved from G7, Part XXV). Prefill on M1–M4 ≥ 1.25× mlx-lm **and ≥ 1.0× llama.cpp**, which already prefills 1.35–1.42× faster than mlx-lm on Qwen3-8B on the M2 Pro (Part V). On M5 Max: 32B decode ≥ 1.0× at short context (≥ 1.05× only if the measured raw read is ≥ ~575 GB/s), long-context targets per §3, prefill ≥ 1.0×, peak RAM ≤ mlx-lm. **M5 Max measurements begin as soon as the hardware exists (target G1)**: bandwidth, dispatch cost, the `matmul2d` rate, and baselines |
| **G6** speculation + TTFT | 20–28 | exact speculation (draft model, prompt lookup; EAGLE-3 if available); batch-invariant verification proven by tests; AOT `.skybin`; prompt cache. Targets in §3 for speculative decode and TTFT |
| **G7** breadth | 20–40 | Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss (MXFP4), Qwen3-MoE; GGUF import; server. Each: correctness suite plus ≥ parity |
| **G8** proof | 30–44 | public `sky-evaluation`; ≥ 3 chips; energy; the "why faster" report from `sky explain`; a paper draft; an outside party reproduces within 5% |
| **G9+** moonshots | ongoing | Megakernel feasibility spike only (rejected for production); e-graph block search; compiler-chosen mixed bits under a KL budget; draft-on-ANE; a trained DFlash-class drafter for Qwen3-32B (needed for a 60+ tok/s average); ASTC texture-compressed weights decoded by the texture units (Apple's own server model uses 3.56 bpw ASTC; R8) |

## 7. Report layout (`docs/inference/`)

| File | Part |
|---|---|
| `README.md` | index, verdict, how to read |
| `part01-executive.md` | I |
| `part02-sky-today.md` | II |
| `part03-current-performance.md` | III |
| `part04-hardware-limits.md` | IV |
| `part05-competitors.md` | V |
| `part06-architecture.md` | VI, plus the one architecture diagram, plus the MLIR/LLVM decisions |
| `part07-tensor-ir.md` | VII |
| `part08-graph-optimizer.md` | VIII |
| `part09-scheduler.md` | IX |
| `part10-autotuner.md` | X |
| `part11-metal-backend.md` | XI |
| `part12-quantization.md` | XII |
| `part13-llm-runtime.md` | XIII |
| `part14-kv-cache.md` | XIV |
| `part15-attention.md` | XV |
| `part16-decode.md` | XVI |
| `part17-prefill.md` | XVII |
| `part18-ttft.md` | XVIII |
| `part19-memory.md` | XIX |
| `part20-persistent-execution.md` | XX |
| `part21-speculative.md` | XXI |
| `part22-performance-model.md` | XXII |
| `part23-benchmarks.md` | XXIII, plus the demo design and profiling/explain |
| `part24-roadmap-now.md` | XXIV: the first ten implementations, with pseudocode |
| `part25-roadmap-12-months.md` | XXV |
| `part26-roadmap-4-years.md` | XXVI, plus staffing |
| `part27-moat.md` | XXVII, plus the acquisition package |
| `part28-risks.md` | XXVIII: the skeptic |
| `top50.md` | top 50 ideas, top 10 breakthrough bets |
| `issues.md` | first 30 GitHub issues |
| `research/R*.md` | appendices |
| `tools/roofline.py` | the model |

## 8. Style rules for every chapter

- Plain, precise English, like `spec/v0.1.md` and `docs/design/ir.md`. No hype, no emoji, no marketing adjectives. "We" means the Sky project.
- Every non-obvious claim carries a label: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what test decides]` or `[MOONSHOT]`.
  - Cite research notes as `(R3 §2)`.
  - Cite code as `mlx/backend/metal/kernels/quantized.h:123`; the research notes record the commit.
  - Cite the web with URLs.
- No invented numbers. A number is measured (with its source), computed by `roofline.py`, or explicitly an assumption (`assume 70% of peak`).
- Concrete over general: name the crate, file, type, function, kernel and test.
- Each design chapter ends with:
  - **Implementation tasks**: each with crate/files, a deliverable and an acceptance test;
  - **Measurements that decide**: what we measure to confirm or kill the design;
  - **Risks and negative results**.
- Cross-reference other chapters by file name, e.g. "see `part16-decode.md` §3".
- Tables over prose for comparisons. ASCII diagrams in fenced `text` blocks.
- Length: as long as the material requires and no longer. A reader should be able to implement from it.
