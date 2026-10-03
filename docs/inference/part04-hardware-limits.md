# Part IV — Hardware limits: what the physics allows

This part calculates the ceilings before any design work: the most that decode, prefill, time to first token (TTFT), memory and energy can be on each Apple chip for the headline model. Every target in this report is checked against these numbers.

Every figure below is either:
- computed by [`roofline.py`](roofline.py) (run `python3 roofline.py --model qwen3-32b --hw m5-max-40`), or
- measured, with its source given.

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`.

## 1. Verdict on the owner's target table

The owner's table (Qwen3-32B, M5 Max, same model and quantization) asked for:

| Metric | MLX in the owner's table | Sky in the owner's table |
|---|---|---|
| Decode | 35 tok/s | 60 tok/s |
| Prefill | 800 tok/s | 1,600 tok/s |
| RAM | 24 GB | 20 GB |
| TTFT | 900 ms | 420 ms |

Against the physics:

| Claim | Possible? | Why |
|---|---|---|
| MLX decodes at 35 tok/s | **No** | 35 tok/s is above the datasheet ceiling of 33.9 tok/s for these weights (§4). The best published MLX number on an M5 Max is 28.4 tok/s (R5, LIKELY: one community run). |
| Sky decodes at 60 tok/s, plain | **No** | It would need 1.09 TB/s; the M5 Max has 614 GB/s. Plain 60 tok/s is M5 Ultra territory: its ceiling is ≈ 60 tok/s at 1.2 TB/s (§4). |
| Sky decodes at 60 tok/s, *effective*, with exact speculation | **Yes, under conditions** | Needs cheap verification (β ≤ 0.05) and a drafter with acceptance length τ ≈ 2.5–3. With the public EAGLE-3 drafter, typical work gives 47–59 tok/s; input-grounded or greedy work ≥ 60 (§8). |
| Prefill at 800 tok/s (MLX) | Close | Published MLX pp512 is 731 tok/s, about 71% of the neural accelerators' fp16 peak (§5). |
| Prefill at 1,600 tok/s (Sky) | **Not at fp16** | The fp16 ceiling is about 1,030 tok/s at 100% of peak. 1,600 tok/s needs int8 activations at ~80% of int8 peak, and W4A8 cost Qwen3-8B 11.5 MMLU points (R8). So not at "same output quality". |
| RAM 24 → 20 GB | Depends on context | 20 GB holds the 18.4 GB of weights plus at most about 4K tokens of fp16 KV, or about 8K of int8 KV (§6). |
| TTFT 900 → 420 ms | **Only with a cached prefix** | A cold 700-token prompt needs ≥ 0.68 s even at 100% of fp16 peak (§7). 420 ms is possible when a prefix cache hit removes most of the prompt. |

The rest of this part shows the arithmetic.

## 2. The machines

| Chip | Spec bandwidth | Planning bandwidth | Matrix peak (fp16) | int8 peak | Shader ALU peak | Max memory |
|---|---:|---:|---:|---:|---:|---:|
| M2 Pro, 16-core GPU (dev machine) | 200 GB/s | **196 GB/s** measured raw read (R6); MLX streams at 182–184 (R3) | 5.73 TFLOPS; MPP `matmul2d` reaches 5.42 (R6) | — | 5.73 | 32 GB (ours: 16) |
| M4 Max, 40-core | 546 GB/s | 491 (planning 90%) | ~16.2 (ALU) | — | ~16.2 | 128 GB |
| M5 Pro, 20-core | 307 GB/s | 276 (planning 90%) | ~32.8 (neural accelerators) | ~61 | ~8.3 | 64 GB |
| M5 Max, 32-core | 460 GB/s | 414 (planning 90%) | ~52 | ~98 | ~13.3 | 36 GB only |
| **M5 Max, 40-core** | **614 GB/s** | **534** (a contributor measured ~530; R3/R4) | **~65** (neural accelerators; R6, two sources) | **~122** (R6, one source) | ~16.6 | 128 GB |
| M5 Ultra (shipped 2026-09-22) | 1,200 GB/s | 1,080 (planning 90%) | ~130 | ~244 | ~33 | 512 GB |

Labels:
- spec bandwidths and memory sizes: `[VERIFIED: Apple specs pages, R6]`;
- M2 Pro measurements: `[VERIFIED: R3, R6]`;
- M5-family planning bandwidth and accelerator rates: `[LIKELY]` — Apple publishes no TFLOPS or clocks for the M5 family (R6); rates for M5 Pro/Ultra are scaled by core count;
- the M5 Max achievable bandwidth must be measured on day 1 (gate G0, [Part XXV](part25-roadmap-12-months.md)).

Two facts shape everything that follows:

- **The neural accelerators help only matrix-shaped work.**
  - Apple shows a plain `simdgroup_matrix` GEMM at "neural accelerator utilization 0%".
  - A one-row `matmul2d` is slower than a GEMV.
  - MLX and llama.cpp both route small batches (M below about 8–33) to ALU kernels. `[VERIFIED: R6]`
  - So decode does not use them; prefill and wide speculative verification can.
- **A kernel boundary costs time even when the work is tiny.** Measured on the M2 Pro (R6, VERIFIED):

  | Event | Cost |
  |---|---|
  | Dependent dispatch (serial, or across a barrier) | ≈ 1.2 µs of GPU time |
  | Independent dispatch | ≈ 0.14 µs |
  | CPU encoding | ≈ 0.08 µs per dispatch |
  | GPU idle when the CPU waits for completion before committing the next command buffer | ≈ 105 µs |

## 3. The model

Qwen3-32B: 64 layers, hidden size 5,120, 64 query heads / 8 KV heads of dimension 128, FFN 25,600, vocabulary 151,936, untied embeddings, QK-norm. 32,762,123,264 parameters, matching Hugging Face exactly. `[VERIFIED: config.json and safetensors headers, R8]`

| Quantity | Value | Note |
|---|---:|---|
| Weights read per decode token | 31.98 B parameters | all layers plus the output head; the embedding is a one-row gather |
| Bytes per decode token, MLX 4-bit group 64 (4.5 bpw) | **17.99 GB** | file size 18.43 GB includes the embedding |
| Bytes per decode token, Qwen's own MLX group 128 | 16.99 GB | (R8) |
| Bytes per decode token, GGUF Q4_0 / Q4_K_M | 18.26 / 19.32 GB | real files are mixes (R8, R9) |
| KV cache per context position, fp16 | **256 KiB** | 64 × 8 × 128 × 2 × 2 B |
| Prefill FLOPs per token | 62.6–66.7 GFLOP | 128 to 4K prompts; attention adds 2.1 MFLOP per token of context |
| Decode FLOPs per token | ≈ 64 GFLOP | |

For the comparisons on our M2 Pro:
- Qwen3-8B streams 4.26 GB per decode token at MLX 4-bit, plus 144 KiB of KV per position.
- Qwen3-0.6B streams 0.34 GB (`roofline.py`).

## 4. Decode: the bandwidth ceiling

Batch-1 decode reads every weight once per token, so tokens per second can be no more than bandwidth divided by bytes per token.

Qwen3-32B, MLX 4-bit, fp16 KV (`roofline.py`):

| Chip | At spec bandwidth, 512 context | At planning bandwidth, 512 | At planning bandwidth, 8K | At planning bandwidth, 32K |
|---|---:|---:|---:|---:|
| M4 Max 40 | 30.1 | 27.1 | 24.4 | 18.5 |
| M5 Pro 20 | 16.9 | 15.2 | 13.7 | 10.4 |
| M5 Max 32 | 25.4 | 22.8 | 20.6 | 15.6 |
| **M5 Max 40** | **33.9** | **29.5** | **26.5** | **20.1** |
| M5 Ultra | 66.2 | 59.6 | 53.6 | 40.6 |

### What competitors already achieve

| Engine | Model / machine | Result | Share of bandwidth | Label |
|---|---|---|---|---|
| mlx-lm | Qwen3-8B, M2 Pro | 37.4–38.8 tok/s | 91% of MLX's own streaming ceiling (182.5 GB/s); 84% of the raw-Metal read (196 GB/s) | VERIFIED (R4, R6) |
| llama.cpp | Qwen3-8B Q4_0, M2 Pro | 34.8 tok/s | 84% of achievable | VERIFIED (R5) |
| mlx-lm | Qwen3-0.6B, M2 Pro | 267 tok/s | 54.5% | VERIFIED (R4) |
| MLX | Qwen3-32B, M5 Max | 28.4 tok/s | about 83% of spec; about 96% of the 530 GB/s contributor measurement | LIKELY (R5) |
| llama.cpp Q4_K_M | Qwen3-32B, M5 Max | 26.4 tok/s | — | LIKELY (R5) |

The 0.6B is a small model, so per-token overhead dominates: 541 dispatches per token, and 1.94 ms of CPU per 3.5 ms of GPU.

### Headroom over MLX for plain decode

- **Short context, large dense model: 1.05–1.14×.**
  - On the M2 Pro the gap between mlx-lm (≈ 165 GB/s effective on 8B) and a raw-Metal 4-bit GEMV (183–191 GB/s, R6) is 1.10–1.16×.
  - Fusion removes most of mlx-lm's 7.6% of non-matmul time (R4).
  - On the M5 Max, MLX is already near the contributor-measured bandwidth. The upside there is unknown until we measure raw-Metal streaming on that chip. `[EXPERIMENTAL: G0/G5]`
- **Long context: 1.05–1.24×, growing with context.**
  - At 4K context, mlx-lm's extra KV reads stream at about 97 GB/s, 53% efficient (R4). MLX's decode attention re-reads K/V per query head (R3).
  - At 32K context on the M5 Max, KV is 8.6 GB per token against 18 GB of weights. Attention efficiency there decides about a third of decode time.
  - With MLX's KV reads at their measured ~53% efficiency, the most a bandwidth-efficient attention can gain over mlx-lm is about 1.08× at 8K, 1.14× at 16K and 1.24× at 32K ([Part XV](part15-attention.md)).
- **Small and draft models: about 1.5× or more.** mlx-lm is at 54.5% on 0.6B and CPU-heavy; BaseRT reports 1.33× on 0.6B (R4, R7).

So plain dense short-context decode is table stakes, not the pitch. This is a negative result, and we state it plainly.

### Where a decode token's time goes

Measured for mlx-lm, Qwen3-8B on the M2 Pro, by removing classes of operations from a bit-exact copy of the model (R4):
- 25.79 ms in total per token;
- 23.46 ms is weights plus KV at full bandwidth;
- the 4-bit matmuls alone take 23.93 ms;
- the other 435 small dependent dispatches add 1.97 ms. The biggest pieces are attention (1.20 ms) and norms (0.76 ms).

A dependent boundary costs 1.2 µs of idle GPU time (R6). Each small kernel also pays a ramp/fill cost of about 2.3 µs, inferred from llama.cpp's data. So the effective cost is about 3.5 µs per dependent kernel, which matches R3's measured ≈ 3.5 µs per MLX kernel. For Qwen3-32B:
- mlx-lm's ≈ 1,225 kernels per 64-layer token cost about 4 ms of a ≈ 30 ms token;
- Sky's plan has 386 dispatches, at five kernels per layer;
- the per-kernel model of [Part XXII](part22-performance-model.md) reproduces mlx-lm on the M2 Pro within −3.4% to +2.6%.

On the M5 Max that model gives Sky 1.03–1.08× over an MLX-like plan at short context, 1.10–1.14× at 8K and 1.24–1.27× at 32K. All of these are modeled, EXPERIMENTAL. They combine the fusion gain with an attention gain that assumes MLX stays at its M2 Pro attention efficiency, so they are model outputs, not targets: the targets are ≥ 1.0× at short context and ≥ 1.05× / 1.10× / 1.2× at 8K / 16K / 32K (§11).

## 5. Prefill: the compute ceiling

Prefill multiplies a block of prompt tokens by every weight, so it is bound by compute: about 63 GFLOP per token for Qwen3-32B.

M5 Max, Qwen3-32B (`roofline.py`):

| Prompt | GFLOP / token | 100% of fp16 peak (65 TFLOPS) | At MLX's measured fp16 matmul rate (46.7 TFLOPS) | At MLX's measured q4 rate (35.0) | int8 at 70% of 122 TOPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 62.6 | 1,039 | 746 | 559 | 1,365 |
| 512 | 63.0 | 1,033 | 742 | 556 | 1,357 |
| 2,048 | 64.6 | 1,007 | 723 | 542 | 1,323 |
| 8,192 | 71.0 | 915 | 658 | 493 | 1,203 |
| 32,768 | 96.8 | 672 | 483 | 362 | 882 |

Sources:
- the MLX matmul rates on M5 Max are `[EXPERIMENTAL: one measurement, R6]`;
- the published MLX pp512 of 731 tok/s (R5) matches the fp16 column, so MLX runs prefill at roughly 71% of the accelerators' peak.

Headroom:
- **M5 Max: about 1.15–1.3× at best**, if Sky's `matmul2d` kernels reach 80–90% of peak. The target is ≥ 1.0×, because MLX also runs prefill on the neural accelerators. R6 found that Apple's sample 64×32 tile gets about half of peak, and that tiles of 64×64 or larger are needed.
  - BaseRT reports 1.44× MLX at 128-token prompts but only 1.02× at 2,048 (R7). Its short-prompt gain comes from overhead, not GEMM speed.
  - `[EXPERIMENTAL: G5]`
- **M1–M4: about 1.25–1.4×.**
  - On the M2 Pro, mlx-lm's end-to-end 8B prefill runs at 3.3 TFLOPS, 58% of peak (R4).
  - MPP `matmul2d` reaches 5.42 TFLOPS fp16 and 4.90 int4 on the same GPU (R6).
  - MLX's own fp16 GEMM reaches 5.0 and its 4-bit GEMM 4.4 (R3).
  - So most of the gap is outside the GEMM: chunk syncs, cache clears, attention, norms.
- **1,600 tok/s needs int8.** The arithmetic allows it only at about 80% or more of the int8 peak (the table shows 1,357 at 70%). The quality cost is real: W4A8 lost 11.5 MMLU points on Qwen3-8B, while W8A8 lost 0.7 (R8). It is therefore a separately labeled mode, possible for 8-bit models, and not the headline.

## 6. Memory

Qwen3-32B at MLX 4-bit (`roofline.py`; assuming 0.3 GB scratch and 0.2 GB runtime, which the memory plan of [Part XIX](part19-memory.md) makes exact):

| Context | KV format | Weights | KV | Total |
|---:|---|---:|---:|---:|
| 4,096 | fp16 | 18.43 GB | 1.07 GB | 20.00 GB |
| 4,096 | int8 | 18.43 | 0.54 | 19.47 |
| 32,768 | fp16 | 18.43 | 8.59 | 27.52 |
| 32,768 | int8 | 18.43 | 4.29 | 23.22 |
| 32,768 | q4 (group 64) | 18.43 | 2.42 | 21.34 |
| 131,072 | fp16 | 18.43 | 34.36 | 53.29 |
| 131,072 | int8 | 18.43 | 17.18 | 36.11 |

What follows:
- A statement such as "24 → 20 GB" means nothing without the context length and KV format.
- At 4K context, 20 GB is the physical floor with fp16 KV. So "20 GB" is achievable only at short context, or with int8 KV (a labeled mode: Apple ships 8-bit KV; R8).
- Comparisons must use one honest metric. Memory-mapped weights read by the GPU are invisible to per-process footprint and RSS. They appear only in system-wide wired memory. So peak RAM is defined as the rise in system memory used over a pre-start baseline (R9 §7; [Part XXIII](part23-benchmarks.md)).

## 7. Time to first token

Warm-process TTFT is roughly prompt length ÷ prefill rate, plus one decode step, plus framework overhead.

On the M5 Max at 100% of fp16 peak:

| Prompt | TTFT floor |
|---:|---:|
| 512 tokens | 0.50 s |
| 700 tokens | 0.68 s |
| 1,024 tokens | 1.0 s |
| 4,096 tokens | 4.2 s |

At MLX's measured rate these are about 1.4× longer.

So:
- **Warm TTFT** improves as much as prefill does (§5), plus one saving. Sky samples the first token inside the last prefill chunk, while mlx-lm runs the last prompt token as a separate decode step: one extra full weight pass, about 35–40 ms on Qwen3-8B on the M2 Pro (R4). That is about 20% of TTFT at 128-token prompts and about 5% at 512 ([Part XVIII](part18-ttft.md)). The warm target is ≥ 1.15× better (SPINE §3).
- **Cold TTFT** (process start to first token, weights in the page cache) is where large factors are honest.
  - mlx-lm reads 18.4 GB with `pread` into new buffers (R3).
  - llama.cpp's first shader compile after install takes 4.8 s (R5).
  - Sky maps an aligned relayout cache with no copy and ships precompiled pipelines ([Part XVIII](part18-ttft.md)).
- **If the weights are not in the page cache**, every engine must read 18.4 GB from SSD. Seconds are unavoidable, and the report never compares a warm Sky with a cold competitor.
- **A cached prefix** (system prompt, earlier turns) removes those tokens from prefill entirely. This is how a 420 ms TTFT becomes honest. Both engines' prefix caches are enabled when comparing, and then Sky's lead shrinks to the saved weight pass and overheads: about 2× at ≤ 8 new tokens, 1.3–1.5× at 64 ([Part XVIII](part18-ttft.md)).
- **The idle gap** before a request changes TTFT by up to 8.8% (R9). Weights un-wired by macOS after idle cost up to 16× on the first tokens unless residency is maintained (R5). Benchmarks state the idle gap.

## 8. Speculation: beyond the bandwidth ceiling, exactly

Plain decode cannot pass §4's ceiling. Speculative decoding can, because one weight stream verifies several drafted tokens.

**How many tokens can be verified for free.** Verifying k tokens costs about k × 64 GFLOP for one 18 GB weight stream. The pass stays bandwidth-bound while k is below the ridge (`roofline.py`, `verify_free_tokens`; R8 gives k* = R·b/16):

| Chip | Free verify width on the matrix peak | Free verify width on shader ALUs |
|---|---:|---:|
| M2 Pro | 8.2 | 8.2 |
| M4 Max | 9.3 | 9.3 |
| M5 Pro / M5 Max / M5 Ultra | 33–35 | 8.4–9.0 |

What this means:
- On every chip, up to about 8 drafted tokens can be verified on the ALUs at almost the cost of one token, **if the verify kernel reads the weights once**.
- On M5 chips, trees of about 16–24 nodes (the theoretical 33–35 less tile inefficiency; [Part XXI](part21-speculative.md)) could be verified on the neural accelerators, **if** `matmul2d` is efficient at those widths. `[EXPERIMENTAL: R6 found tiles of 64×64 or more are needed for efficiency]`
- Byte-identical speculation needs decode and verify to share one arithmetic path. Wide accelerator verification therefore forces decode onto the accelerators too, which may be slower at M = 1 (R6). Otherwise byte-identity is kept only for ALU-width verification. This is measured, not assumed ([Part XXI](part21-speculative.md)).

**Today's verify kernels are far from free:**

| Engine / machine | Extra cost of verifying more tokens |
|---|---|
| mlx-lm, M2 Pro | (k+1) tokens cost (k+1) forwards; β ≈ 0.5–1 (R4) |
| MLX, M5 Max (70B) | +12% at 3 tokens, +48% at 5, +125% at 8 (mlx#3791, R3/R4) |
| llama.cpp, M2 Pro | 2.08× at 4 tokens, 3.76× at 8 (R5) |

That is why mlx-lm's speculation is slower than plain decode on the M2 Pro at every setting, and llama.cpp's gains only 1.12×.

**Modeled effective speed for Qwen3-32B on an M5 Max.** The formula is S = τ / (1 + γc + γβ) (R8), with:
- τ = tokens emitted per round;
- γ = draft length;
- c = draft cost per token, relative to one target step;
- β = extra verify cost per drafted token.

Base: 28.0 tok/s, which is 95% of the 29.5 tok/s planning ceiling.

| Drafter (published τ) | γ | β = 0.02 | β = 0.05 | β = 0.15 (≈ MLX on M5 Max) | β = 0.48 (MLX on M2-class GPUs) |
|---|---:|---:|---:|---:|---:|
| EAGLE-3, τ = 2.29 (HumanEval, temperature 0.6) | 3 | 55.8 | 51.7 | 41.6 | 25.3 |
| EAGLE-3, τ = 2.49 (GSM8K) | 3 | 60.6 | 56.2 | 45.3 | 27.6 |
| EAGLE-3, τ = 3.08 (GSM8K, 7 drafts) | 7 | 63.9 | 55.3 | 38.2 | 18.9 |
| A stronger drafter, τ = 4.5 (DFlash-class; none public for 32B) | 7 | 93.3 | 80.8 | 55.8 | 27.6 |

Sources:
- the τ values are VERIFIED published numbers for `RedHatAI/Qwen3-32B-speculator.eagle3` (R8);
- c = 0.03 is assumed;
- the speeds are modeled, EXPERIMENTAL.

The conclusions:
1. **The verify kernel decides everything.** At β ≈ 0.5 speculation loses; at β ≤ 0.05 it roughly doubles throughput. On the M5 Max, MLX's own verify is already moderately cheap: β ≈ 0.06–0.18 from mlx#3791's +12% at 3 tokens and +48% at 5. So Sky's *kernel* advantage over mlx-lm's speculation there is about 1.2×, not 2×. The rest of Sky's lead must come from exactness and from drafters mlx-lm cannot run: it supports separate draft models only, with no EAGLE-3 or MTP (R4, R8; [Part XXVII](part27-moat.md)).
2. **47–59 tok/s is the honest typical range** with public drafters (SPINE §3; [Part XXII](part22-performance-model.md)'s per-kernel model gives 52–56 at β ≤ 0.05; the idealized table above reaches 64). 60+ on average needs a drafter nobody has published for Qwen3-32B (gate G9).
3. **Exactness is not free elsewhere.** mlx-lm's speculative output diverges from its plain output (#1470). Sky's batch-invariant kernels plus keyed Gumbel sampling make the two byte-identical ([Part XXI](part21-speculative.md)).

## 9. Energy

Energy per token is whole-system power × time ÷ tokens. On Apple silicon, chip-level counters undercount it. During 0.6B decode, system energy per token was 2.4× what the chip counters showed, and 7.1 W of the CPU's draw was mlx-lm's Python loop (R9, VERIFIED with the SMC `PSTR` sensor). Two consequences:

- **Removing per-token CPU work is a real energy gain**, largest for small models: a static, GPU-resident loop ([Part XX](part20-persistent-execution.md)).
- **Speculation reduces DRAM traffic per accepted token** by up to τ×, and DRAM traffic dominates decode energy. This is an honest place to win on joules per token. `[EXPERIMENTAL: measure with PSTR over ≥ 60 s loops]`

## 10. Thermals

Laptops lose about 20% sustained performance to thermals (R6, LIKELY). Every headline number must hold in a 10-minute sustained run on AC power in High Power Mode, interleaved with the competitor ([Part XXIII](part23-benchmarks.md)). A 5-second burst number is never quoted alone.

## 11. Summary: honest targets

| Metric (Qwen3-32B 4-bit, M5 Max 40) | Physics ceiling | Best competitor today | Sky target | Label |
|---|---|---|---|---|
| Decode, plain, short context | 33.9 (spec) / 29.5 (planning) tok/s | MLX 28.4 | ≥ 1.0× until raw-read bandwidth is measured; ≥ 1.05× needs ≈ 575 GB/s raw read. Modeled Sky plan: 28.3 tok/s (Part XXII). On the M2 Pro: ≥ 1.05× (modeled 1.09–1.11×) | EXPERIMENTAL |
| Decode, plain, 8K / 16K / 32K | 26.5 / — / 20.1 tok/s (planning) | unmeasured on M5 Max; mlx-lm 53% efficient on KV reads (M2 Pro) | ≥ 1.05× / 1.10× / 1.2× mlx-lm; bounds ≈ 1.08× / 1.14× / 1.24× | EXPERIMENTAL |
| Decode, effective with exact speculation | none (depends on τ) | mlx-lm 1.82× on M3 Max with a 0.5B draft; slower on M2 Pro; inexact | 47–59 tok/s typical (≥ 1.7× plain); ≥ 60 on input-grounded / greedy work; ≥ 1.2× mlx-lm's own speculation with the same draft | EXPERIMENTAL |
| Prefill (headline at pp2048; pp512 shown for published comparisons) | 1,007 / 1,033 tok/s (fp16 peak) | MLX 731 at pp512; on M1–M4 llama.cpp already 1.35–1.42× mlx-lm (Part V) | ≥ 1.0× (M5: both use the neural accelerators); ≥ 1.25× mlx-lm and ≥ 1.0× llama.cpp (M1–M4) | EXPERIMENTAL |
| Warm TTFT, 512 prompt | 0.50 s | ≈ 0.70 s | follows prefill; ≥ 1.15× better | EXPERIMENTAL |
| Cold TTFT (page cache warm) | dominated by load and compile | `pread` copy of 18.4 GB; shader compiles | ≥ 3× better | EXPERIMENTAL |
| Peak RAM at 32K, fp16 KV | 27.5 GB floor | unmeasured | ≤ 0.9× mlx-lm | EXPERIMENTAL |
| Joules per token | — | unmeasured | ≤ 0.85× mlx-lm plain; lower with speculation | EXPERIMENTAL |

Every target is re-baselined at gate G0 on the M2 Pro, and on an M5 Max as soon as one is available (target: G1, not G5).

## 12. Implementation tasks

| ID | Task | Deliverable | Acceptance test |
|---|---|---|---|
| IV-T1 | Ship `roofline.py` with the report, and add per-kernel functions used by Part XXII | `docs/inference/roofline.py` | Its tables reproduce every number in this part |
| IV-T2 | Day-1 M5 Max measurement kit: raw-Metal read, copy and write bandwidth; dependent and independent dispatch cost; `matmul2d` TFLOPS by tile and dtype; sustained clocks (10 min); system power via `PSTR` | `research/microbench/` extended, plus a `run_m5.sh` | Results JSON committed; `roofline.py`'s M5 Max entry updated from LIKELY to VERIFIED |
| IV-T3 | Baselines on M5 Max: mlx-lm and llama.cpp, Qwen3-32B, at 512 / 8K / 32K context, pp512 / pp8K, peak RAM by the honest metric | `sky-evaluation` baseline run | Medians of ≥ 5 interleaved runs, protocol of [Part XXIII](part23-benchmarks.md) |
| IV-T4 | Verify-cost curve: MLX `quantized_matmul` at M = 1…32 on 32B shapes on M5 Max | Table and plot | Decides β for §8 |

## 13. Measurements that decide

- **Achievable streaming bandwidth on M5 Max** (raw Metal vs MLX). This decides whether plain decode can beat MLX by more than a few percent.
- **`matmul2d` efficiency at M = 8–64 on M5 Max.** This decides the speculative tree width on the neural accelerators.
- **The KV-read efficiency of mlx-lm at 8K and 32K on M5 Max.** This decides the long-context decode target.

## 14. Risks and negative results

- **The plain-decode headroom on the M5 Max may be ≈ 1.0×** if MLX's streaming already reaches raw-Metal bandwidth there. The report then rests on speculation, long context, small models, TTFT, RAM and energy, which is the expected case and not a failure mode.
- **The neural-accelerator rates are secondary-source numbers.** If the fp16 rate is lower, the free verify width on M5 shrinks toward the ALU value (~8–9), and the prefill targets fall.
- **Every figure in §8 is a model.** Acceptance depends on the workload and the temperature, and on whether Qwen3's thinking mode is on. The demo must use declared workloads.
