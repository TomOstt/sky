# R8 — Inference algorithms that change the physics (Qwen3-32B on an M5 Max)

Research agent R8, 2026-10-01/02. These notes are evidence for the Sky acquisition-grade report.
`S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.
Working files: `$S/research/R8-work/`.

**Labels.**
- **VERIFIED**: I read the primary source (paper, official page, model card, config.json, safetensors header, or source code). The URL or path is given.
- **LIKELY**: an inference from verified facts; the reason is given.
- **EXPERIMENTAL**: my own model or estimate; it needs a measurement.
- **MOONSHOT**: high risk.
- **UNVERIFIED**: a third-party claim I could not check.

**Reproducible arithmetic.**
- `R8-work/model_math.py` → `model_math.out.md`: parameters, bytes, FLOPs and KV for 10 models.
- `R8-work/spec_model.py` → `spec_model.out.md`: speculative-decoding speedup model and MoE expert-union arithmetic.
- `R8-work/entropy_sim.py`: entropy of 4-bit codes.
- `R8-work/st_headers.py`: reads safetensors headers over HTTP range requests to get exact tensor shapes.
- `R8-work/configs/`: all config.json files used.

**Method note.** WebSearch/WebFetch hit an account quota midway. After that, sources were fetched with curl:
- arXiv abstract and HTML pages;
- the GitHub API via the authenticated `gh` CLI;
- the Hugging Face API, including safetensors headers;
- OpenAlex for literature search.

The quotes and numbers below come from those fetches. Cached copies are in `R8-work/cache/`.

**Units.** GB = 1e9 bytes. "Roofline tok/s" = bytes per token ÷ 614 GB/s at 100% efficiency. Real kernels reach about 70–90%.

---

## 0. TL;DR

1. **Dense decode ceiling (VERIFIED arithmetic).** Qwen3-32B in MLX 4-bit (group 64) streams **17.99 GB per decoded token**. At 614 GB/s the ceiling is **34.1 tok/s** at zero context and 30.5 tok/s at 8K context.
   - "MLX 35 tok/s" is above this roofline. The owner's baseline is physically impossible for this quantization.
   - A realistic MLX number is about **24–29 tok/s** (EXPERIMENTAL).
2. **Only speculation clears the ceiling without changing the output distribution.** Public EAGLE-3 speculators exist for both Qwen3-32B and Qwen3-8B.
   - Qwen3-32B: RedHatAI, AngelSlim, Zhihu and others.
   - Qwen3-8B: also DFlash and P-EAGLE variants.
   - Published Qwen3-32B acceptance at the model's recommended T=0.6 is only **τ ≈ 2.2–3.1 tokens per round**.
   - With near-free verification this gives **1.6–2.2×**, i.e. 43–59 tok/s from a 27 tok/s base.
   - With MLX's current small-M kernels it gives **≤1×**.
3. **The decisive enabler is a kernel, not a new algorithm.** Verifying k = 4–16 tokens must cost about the same as 1 token.
   - On M5 Max the arithmetic allows this up to k\* ≈ R·b/16 ≈ 32 tokens with the neural accelerators, or about 8 with SIMD units only.
   - Measured MLX today: 4 rows cost **2.43×** one row (R3, M2 Pro). An MLX EAGLE-3 prototype got **1.05×**. llama.cpp's Metal MTP is a net loss on M1 Max.
4. **Exactness is a sellable property MLX gives up.** mlx-lm issue #1470 (M5 Max, Qwen3-32B-6bit + Qwen3-0.6B): speculative greedy diverges from plain greedy at token 91. The cause is a bf16 logit tie combined with batched vs sequential reduction order. MLX's response was to document it.
   - Sky can make decode and verify kernels batch-invariant by construction. Then:
     - greedy speculative output is bit-identical to Sky's own plain greedy;
     - with keyed Gumbel sampling, sampled output is bit-identical too, for any temperature and any drafter.
5. **Prefill 1,600 tok/s on Qwen3-32B needs ≈102–107 TFLOP/s.**
   - M5 Max neural-accelerator FP16 peak is about 65–70 TFLOPS. The FP16 ceiling is therefore about 1,000–1,100 tok/s, so the target is impossible in FP16.
   - It is possible on paper only with INT8 activations (~122–130 TOPS peak). That changes outputs; Qwen3 is quantization-sensitive.
6. **Competition moved in 2026.**
   - The Qwen 27B line (Qwen3.5/3.6/3.8-27B) ships a native MTP head and a 3:1 Gated-DeltaNet:attention stack, with 4× less KV per token.
   - **MTPLX** (MLX-based, open source) self-reports 65–88 tok/s on M5 Max for Qwen3.6/3.8-27B 4-bit.
   - **BaseRT** (arXiv 2607.19438) reports up to 3.9× MLX prefill on M5 Pro.
   - llama.cpp master supports EAGLE-3, MTP, DFlash and DSpark drafting.

---

## 1. Model arithmetic

### 1.1 Qwen3-32B configuration (VERIFIED)

Sources:
- `https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json`, saved as `R8-work/configs/Qwen_Qwen3-32B.json`;
- safetensors headers (`st_headers.py`);
- model card `https://huggingface.co/Qwen/Qwen3-32B` (README).

| field | value |
|---|---|
| architectures / model_type | Qwen3ForCausalLM / qwen3 |
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| num_attention_heads / num_key_value_heads | 64 / 8 (GQA 8:1) |
| head_dim | 128 (so q_proj output is 64·128 = 8192 ≠ hidden 5120) |
| intermediate_size (SwiGLU) | 25600 |
| vocab_size | 151936 |
| tie_word_embeddings | false (separate 151936×5120 lm_head) |
| attention_bias | false (Qwen2's QKV bias is gone) |
| q/k norm | yes: `model.layers.N.self_attn.q_norm.weight [128]`, `k_norm.weight [128]` (RMSNorm over head_dim; verified in safetensors header) |
| rms_norm_eps | 1e-6 |
| rope_theta | 1,000,000 |
| rope_scaling | null in config. Card: "Context Length: 32,768 natively and 131,072 tokens with YaRN"; YaRN is enabled by adding `{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}` |
| max_position_embeddings | 40960 |
| sliding window | `use_sliding_window: false` (full attention in all 64 layers) |
| MTP head | none: no `mtp.*` tensors, and the HF parameter total matches the computed dense total exactly |
| sampling (card) | thinking: T=0.6, TopP=0.95, TopK=20, and "**DO NOT use greedy decoding**"; non-thinking: T=0.7, TopP=0.8, TopK=20 |

**Implication of the last row.** The realistic operating point is *sampling*, not greedy.
- Speculative acceptance at T=0.6–0.7 is the number that matters.
- The exactness claim that matters is "same distribution" or "same sample under a keyed RNG", not "same greedy string".

### 1.2 Parameter count per component (VERIFIED: exact match to HF API `safetensors.total` = 32,762,123,264)

Per layer:

| component | arithmetic | params |
|---|---|---|
| q_proj | 5120 × 8192 | 41,943,040 |
| k_proj, v_proj | 2 × 5120 × 1024 | 10,485,760 |
| o_proj | 8192 × 5120 | 41,943,040 |
| q_norm + k_norm | 2 × 128 | 256 |
| **attention** | sum of the above | **94,372,096** |
| gate, up, down | 3 × 5120 × 25600 | **393,216,000** |
| 2 RMSNorms | 2 × 5120 | 10,240 |
| **per layer** | 94,372,096 + 393,216,000 + 10,240 | **487,598,336** |

Model:

| component | arithmetic | params |
|---|---|---|
| 64 layers | 64 × 487,598,336 | 31,206,293,504 |
| embedding | 151936 × 5120 | 777,912,320 |
| lm_head | 151936 × 5120 | 777,912,320 |
| final norm | | 5,120 |
| **total** | | **32,762,123,264** (exact match to the HF total) |

Card: "Number of Parameters: 32.8B; Non-Embedding: 31.2B". Matches.

Streamed per decode token (matmul weights): 64 × (94,371,840 + 393,216,000) = **31,205,621,760** in the blocks, + **777,912,320** lm_head = **31.98 B**.
- The embedding needs only one row per token, which is negligible.
- Shares of the per-token bytes:
  - MLP: 64 × 393.2 M = 25.17 B = 78.7%;
  - attention projections: 6.04 B = 18.9%;
  - lm_head: 2.4%.

### 1.3 Bytes per decode token (weights only, batch 1)

| format | bits/weight (how) | Qwen3-32B bytes per token | roofline @614 GB/s | check vs real file |
|---|---|---|---|---|
| BF16 | 16 | 63.97 GB | 9.6 tok/s | — |
| MLX 8-bit g64 | 8 + 32/64 = 8.5 | 33.98 GB | 18.1 | mlx-community/Qwen3-32B-8bit = 34.81 GB on disk ✔ (incl. 8-bit embedding) |
| MLX 6-bit g64 | 6.5 | 25.99 GB | 23.6 | -6bit = 26.62 GB ✔ |
| **MLX 4-bit g64** (mlx-lm default) | 4 + (16+16)/64 = **4.5** | **17.99 GB** | **34.1** | mlx-community/Qwen3-32B-4bit = **18.430 GB**; computed 18.43 ✔ |
| MLX 4-bit g128 | 4.25 | 16.99 GB | 36.1 | **Qwen/Qwen3-32B-MLX-4bit (official) uses group_size 128** = 17.406 GB ✔ |
| MLX 3-bit g64 | 3.5 | 13.99 GB | 43.9 | -3bit = 14.335 GB ✔ |
| GGUF Q4_K_M | Q4_K 4.5 + Q6_K 6.5625 mix | **19.32 GB** | 31.8 | Qwen/Qwen3-32B-GGUF Q4_K_M = 19.762 GB; computed file 19.756 GB ✔ |
| GGUF Q4_0 (imatrix) | Q4_0 4.5; first L/8 ffn_down Q4_1 5.0; output Q6_K | 18.26 GB | 33.6 | unsloth Q4_0 = 18.703 GB; computed 18.70 GB ✔ |

All MLX and GGUF file sizes are VERIFIED from the HF tree API.

How the GGUF recipes were derived (VERIFIED from source):
- **Bits per weight**: `ggml/src/ggml-common.h` static_asserts.
  - Q4_K = 144 B / 256 weights = 4.5; Q6_K = 210 / 256 = 6.5625; Q4_0 = 18 / 32 = 4.5; Q8_0 = 34 / 32 = 8.5; MXFP4 = 17 / 32 = 4.25.
  - New in 2026: NVFP4 (ggml) = 36 / 64 = 4.5, with a UE4M3 scale per 16 weights.
- **Q4_K_M tensor choice**: `src/llama-quant.cpp`.
  - `output.weight` → Q6_K ("else if (new_type != GGML_TYPE_Q8_0) new_type = GGML_TYPE_Q6_K").
  - `attn_v` and `ffn_down` → Q6_K where `use_more_bits(i,n) = i < n/8 || i >= 7n/8 || (i − n/8) % 3 == 2`. For n = 64 that is exactly 32 of the 64 layers.
  - Everything else → Q4_K. The token embedding uses the base type.

Arithmetic for Q4_K_M:
- Q6_K: 32 × (131.07 M ffn_down + 5.24 M attn_v) + 777.9 M output = 5.140 B params → 4.216 GB.
- Q4_K: the remaining 26.844 B params → 15.099 GB.
- Embedding at Q4_K: 0.438 GB.
- File ≈ 19.756 GB. Decode bytes = file − embedding ≈ **19.32 GB**.

**Lesson.** Q4_K_M streams **7.4% more bytes** than MLX 4-bit g64 for Qwen3-32B. This alone predicts llama.cpp decode at about 0.93× MLX on equal kernels.

### 1.4 FLOPs per token (EXPERIMENTAL arithmetic from VERIFIED shapes; `model_math.out.md` §5)

Decode, per token:
- **Matmul**: 2 × 31.98 B = **64.0 GFLOP**.
- **Attention**: 4 × n_q × head_dim × T per layer (QKᵀ + PV) = 4 · 64 · 128 · 64 layers = **2.10 MFLOP per context token**.
  - That is 8.6 GFLOP at T = 4K and 68.7 GFLOP at T = 32K, where it equals the matmul cost.
  - These attention FLOPs read the KV cache (bandwidth); they do not use the weights.

Prefill of P tokens: 2·P·31.21 B + 2·V·H (logits for the last token only) + causal attention 2 · n_q · head_dim · P² · 64.
- P = 1K: 65.0 TFLOP (**63.5 GFLOP/token**)
- P = 4K: 273 TFLOP (66.7 GFLOP/token; attention 6.4%)
- P = 32K: 3,171 TFLOP (96.8 GFLOP/token; attention 36%)

### 1.5 KV cache per token of context (VERIFIED shapes)

64 layers × 2 (K, V) × 8 KV heads × 128 = **131,072 elements per token**.

| precision | bytes/token | 8K ctx | 32K ctx | 40,960 (max native) |
|---|---|---|---|---|
| fp16/bf16 | 262,144 B (256 KiB) | 2.15 GB | 8.59 GB | 10.74 GB |
| int8, MLX affine g64 (8.5 bit) | 139,264 B | 1.14 GB | 4.56 GB | 5.70 GB |
| 4-bit, MLX affine g64 (4.5 bit) | 73,728 B | 0.60 GB | 2.42 GB | 3.02 GB |

Decode roofline vs context, 4-bit g64 weights:

| context T | fp16 KV | 8-bit KV | 4-bit KV |
|---|---|---|---|
| 0 | 34.1 | 34.1 | 34.1 |
| 4K | 32.2 | 33.1 | 33.6 |
| 8K | 30.5 | 32.1 | 33.0 |
| 16K | 27.6 | 30.3 | 32.0 |
| 32K | 23.1 | 27.2 | 30.1 |

All values are tok/s. KV quantization is worth up to 1.30× in bandwidth at 32K, and ≤1.05× below 4K.

### 1.6 The other requested models (VERIFIED configs; totals match HF exactly)

| model | total params | streamed params / token | 4-bit g64 GB/token → roofline | Q4_K_M | fp16 KV B/token |
|---|---|---|---|---|---|
| Qwen3-32B | 32.76 B | 31.98 B | 17.99 GB → 34 | 19.32 → 32 | 262,144 |
| Qwen3-8B | 8.19 B | 7.57 B | 4.26 → 144 | 4.67 → 131 | 147,456 |
| Qwen3-4B (tied) | 4.02 B | 4.02 B | 2.26 → 271 | 2.49 → 247 | 147,456 |
| Qwen3-1.7B (tied*) | 1.72 B (+0.31 B duplicate lm_head stored) | 1.72 B | 0.97 → 634 | 1.10 → 558 | 114,688 |
| Qwen3-0.6B (tied*) | 0.60 B (+0.16 B duplicate) | 0.60 B | 0.34 → 1,832 | 0.39 → 1,572 | 114,688 |
| Qwen3-30B-A3B (128 experts, top-8, moe_I = 768) | 30.53 B | 3.04 B active | 1.71 → 359 | 1.88 → 327 | 98,304 |
| Llama-3.1-8B | 8.03 B | 7.51 B | 4.22 → 145 | 4.62 → 133 | 131,072 |
| Llama-3.1-70B | 70.55 B | 69.50 B | 39.09 → 16 | 41.87 → 15 | 327,680 |
| gpt-oss-20b (32 experts, top-4, MXFP4 experts) | 20.91 B | 3.61 B active | native (MXFP4 experts + BF16 rest): 3.71 GB → 166; MXFP4 + 8-bit rest: 2.56 → 240 | — | 24,576 (12 full-attention layers; 12 SWA-128 layers fixed) |
| gpt-oss-120b (128 experts, top-4) | 116.83 B | 5.13 B active | native 5.00 → 123; MXFP4 + 8-bit rest 3.55 → 173 | — | 36,864 (18 full layers + 18 SWA-128) |

Notes on the table:
- \* The Qwen3-0.6B and 1.7B checkpoints store `lm_head.weight` although `tie_word_embeddings: true`. The HF total includes the duplicate (VERIFIED from the header).
- Llama configs come from the ungated mirror `unsloth/Meta-Llama-3.1-*-Instruct`. `meta-llama/*` returns HTTP 401.
- gpt-oss details (VERIFIED from config):
  - `quantization_config.modules_to_not_convert` = attention, router, embed and lm_head, so those stay BF16.
  - Native BF16 lm_head is 0.579 B params = **1.16 GB per token**, more than the active experts' 1.90 GB × 0.6. A runtime that quantizes the non-expert weights to 8 bits gains 1.4×.
  - Alternating `sliding_attention`/`full_attention` layers with `sliding_window: 128` and learned sinks.
- Decode FLOPs per token (matmul): Qwen3-8B 15.1 G, 30B-A3B 6.1 G, Llama-70B 139 G, gpt-oss-120b 10.3 G.

### 1.7 2026 context: the "Qwen 32B" line is now Qwen3.5/3.6/3.8-27B (VERIFIED)

- **Release dates** (HF API `createdAt`):
  - Qwen3.5-27B: 2026-02-24
  - Qwen3.6-27B: 2026-04-21
  - Qwen3.8-27B: 2026-08-05
  - Qwen3.6-35B-A3B and Qwen3.8-Flash-Next are the MoE siblings.
- **Configs** (all three 27B models share the architecture, `qwen3_5`):
  - hidden 5120, 64 layers, `full_attention_interval: 4`;
  - 48 Gated-DeltaNet linear-attention layers: 16 QK heads × 128, 48 V heads × 128, conv kernel 4;
  - 16 gated full-attention layers: 24 Q / 4 KV heads, head_dim 256, partial RoPE 0.25;
  - FFN 17408, vocab 248320, untied, 262,144 native context;
  - **`mtp_num_hidden_layers: 1`**.
- **Tensors** (safetensors headers):
  - MLP 17.11 B, linear attention 5.56 B, full attention 1.68 B, embedding 1.27 B, lm_head 1.27 B;
  - vision 0.46 B;
  - **MTP module 0.425 B**: `mtp.fc` [5120, 10240] + one attention/MLP layer; it reuses the target's embedding and lm_head.
- **Physics vs Qwen3-32B**:
  - Streamed bytes at 4-bit g64 ≈ 25.6 B params × 0.5625 = **14.4 GB → 42.6 tok/s roofline**.
  - KV per token = 16 layers × 2 × 4 × 256 × 2 B = **65,536 B, 4× less than Qwen3-32B**.
  - Plus a fixed recurrent state of 48 × 48 heads × 128 × 128 × 4 B (fp32, `mamba_ssm_dtype: float32`) ≈ **151 MB**, read and written every token (≈1–2% of traffic).
  - One MTP draft token costs about 0.425 B params + the 1.27 B-param lm_head at 4-bit ≈ 0.95 GB, i.e. **≈6.6% of a target step**. The 248K-vocab lm_head dominates the draft cost.
  - Card guidance: SGLang `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4`.
- **Strategic note (LIKELY).** A buyer evaluating "Qwen 32B-class on M5 Max" in late 2026 will run Qwen3.6/3.8-27B with MTP, not Qwen3-32B. Sky's plan should treat the hybrid DeltaNet + MTP architecture as a first-class target.


## 2. Speculative decoding

### 2.1 The math (VERIFIED from Leviathan et al., arXiv 2211.17192, ICML 2023; PDF text extracted to `R8-work/pdf/leviathan.txt`)

**Definitions.**
- Let β be the acceptance probability of one drafted token, and α = E[β].
- Corollary 3.6: α = 1 − E[D_LK(p,q)] = E[Σ_x min(p(x), q(x))], i.e. 1 − total variation distance between target and draft.

**Eq. (1): expected tokens per round** with γ drafted tokens:

**E[#tokens] = (1 − α^(γ+1)) / (1 − α)**

This includes the "bonus" token the target always produces.

**Definition 3.7.** "Let c, the cost coefficient, be the ratio between the time for a single run of Mq and the time for a single run of Mp."

**Theorem 3.8: walltime improvement factor**

**(1 − α^(γ+1)) / ((1 − α)(γc + 1))**

- It rests on an explicit assumption: "we'll assume that we can run γ + 1 concurrent evaluations of Mp in parallel without increasing the walltime."
- **That assumption is exactly what fails on today's Apple Silicon kernels** (§2.3).
- Corollary 3.9: if α > c, some γ gives a speedup of at least (1+α)/(1+c).
- Theorem 3.11: arithmetic operations grow by (1 − α)(γĉ + γ + 1) / (1 − α^(γ+1)).
- Their T5-XXL numbers: speedup of **3.4×** at temp 0 and **2.6×** at temp 1 (EnDe, T5-small draft, γ = 7, α = 0.75 / 0.62).
  - Acceptance drops with temperature in every row of their Table 2.

**Chen et al. (arXiv 2302.01318, DeepMind)** (VERIFIED abstract):
- "2–2.5× decoding speedup in a distributed setup" on Chinchilla-70B;
- the "modified rejection sampling scheme … preserves the distribution of the target model **within hardware numerics**."
- The rule:
  - accept draft x̃ with probability min(1, q(x̃)/p(x̃));
  - on rejection, resample from norm(max(0, q − p)).

**Generalized speedup used in this report** (EXPERIMENTAL model; `spec_model.py`):

  S(γ) = τ / (1 + γ·c + γ·β)

where:
- τ = E[#tokens per round], measured or from Eq. (1);
- c = draft cost per drafted token ÷ T₁;
- β = marginal cost of verifying one more token ÷ T₁;
- T₁ = one plain decode step.

Leviathan's Theorem 3.8 is the special case β = 0. For tree drafts, replace γ·β by (n_nodes − 1)·β and γ·c by depth·c (Sequoia's form below).

### 2.2 How many tokens can be verified "for free" on a bandwidth-bound GPU (EXPERIMENTAL derivation, checked against Sequoia)

For a weight-streaming matmul over k token rows with b-bit weights:
- FLOPs = 2·N·k
- bytes ≈ N·b/8
- arithmetic intensity I(k) = 16·k/b FLOP/byte

The pass stays memory-bound while I(k) < R = F_peak / BW (the ridge point):

**k\* = R · b / 16**   (dense model; ignores the KV term, which grows only in FLOPs)

| device / matmul path | F | BW | R | k\* at b = 4.5 | k\* at b = 16 |
|---|---|---|---|---|---|
| M5 Max neural accelerators FP16 | ~65–70 TFLOPS | 614 | ~110 | **~31–32** | ~110 |
| M5 Max NA INT8 | ~122–130 TOPS | 614 | ~200 | ~56–60 | — |
| M5 Max SIMD FP32 (no NA) | ~16 | 614 | 27 | **~7.5** (FP16 SIMD ≈ 1.5× → ~11) | 27 |
| M2 Pro SIMD | ~6.8 | 200 | 34 | ~9.6 | 34 |
| A100 FP16 tensor cores | 312 | 2,039 | 153 | 43 | 153 |

Sources and checks for the inputs:
- **M5 Max NA FP16 / INT8** (VERIFIED secondary measurement): tzakharko, "Investigating the GPU Neural Accelerators on Apple A19/M5".
  - "Estimated 1024 FLOPS per GPU core per cycle for FP16".
  - "Estimated <~2048 OPS per GPU core per cycle for INT8 (with caveats)".
  - "M5 Max (estimated, 40 cores): 70 TFLOPS FP16, 130 INT8".
  - "Optimal matrix size appears to be 32 × 32 or larger".
  - **No BF16 and no FP8** in first-generation NA hardware. A Metal 4.1 FP8 microbench reports fp8×fp8 at the FP16 rate.
  - Some sources quote ~65 TFLOPS.
- **M5 Max bandwidth: 614 GB/s** — VERIFIED from the Apple newsroom (2026-03): "M5 Max … up to 614GB/s"; the 32-core GPU bin is 460 GB/s.
- **Sequoia (arXiv 2402.12374) cross-check**: its hardware-aware optimizer maximizes Speedup(n,d) = G(n,d)/(t(n) + d·c).
  - It picked **tree size 128 on A100** with fp16 weights. The formula gives ~153 (LIKELY consistent).
  - It picked 768 for offloaded 70B on L40, where verification is nearly free relative to PCIe weight streaming.

**Key insight (LIKELY).** **Weight quantization and speculation compete for the same headroom.**
- 4.5-bit weights cut k\* by 16/4.5 = 3.6× relative to fp16 weights.
- Dequantization ALU work is paid once per weight per pass, regardless of k. Rows 2..k add only FMAs.
- On M5 Max a well-designed kernel should verify **~8 tokens on SIMD units, or up to ~30 with NA-fed tiles**, at roughly the cost of 1. A 32-row tile also matches the NA's preferred 32×32 shape.

**Measured reality on Apple today: verification is far from free.**

| measurement | result | label |
|---|---|---|
| R3 (this project, M2 Pro, MLX 0.32.3, Qwen3-32B-shaped 4-bit layers): MLX qmv vs M rows | q_proj: M=1 153.9 µs, M=2 1.30×, M=3 1.85×, **M=4 2.43×**, M=8 4.39×. gate/up and down show the same shape (M=4 ≈ 2.5×). β ≈ 0.48. R3's own experimental "multi" kernel: M=2 0.98×, M=3 1.22×, then collapses at M ≥ 6. | VERIFIED local (`$S/research/r3_out_qmv_multi_1.txt`) |
| mlx-lm discussion #890 (M3 Ultra, Llama-3.1-8B 4-bit + EAGLE-3 head) | baseline 113.7 tok/s → **118.9 (1.05×)**; "single verification token ~1ms" on an 8.8 ms step; acceptance 0.34; tree verification impossible because the KV cache has a single RoPE offset | VERIFIED (GitHub discussion) |
| lmstudio-ai/mlx-engine #103 (M1 Pro 32 GB) | Llama-3.1-8B + 1B draft: 38 → 33.9 tok/s; + 3B draft: 38.7 → 29.3; Qwen2.5-7B + 1.5B draft: 37.1 → 22.5 (−39%) | VERIFIED (issue) |
| llama.cpp Metal MTP (issue #23752, via modelfit.io) | M1 Max, Qwen3.5-9B: −11% to −24% vs no speculation | UNVERIFIED (secondary) |
| R5 (this project, llama.cpp source) | `mul_mv_ext` (batch 2–8) reads weights **once for batch 2–5, twice for 6–8** | VERIFIED by R5 (`R5-llamacpp.md` §2) |

**Conclusion.** The algorithmic headroom is real on M5 Max (k\* ≈ 8–30). Today's stacks do not capture it. Small-M verification kernels that cost ≈ 1.0–1.1× at k ≤ 8 are the single most valuable decode-side kernel for Sky.

### 2.3 Speculator weights available for Qwen3-32B and Qwen3-8B (VERIFIED via the HF API, model cards and safetensors headers, 2026-10-01)

The HF API search for "eagle3", "speculator" and "eagle-3" returns 1,092 repos. Relevant ones:

| repo (created) | target | architecture (VERIFIED from config / headers) | published acceptance |
|---|---|---|---|
| **RedHatAI/Qwen3-32B-speculator.eagle3** (2025-09-18) | Qwen3-32B | EAGLE-3: 1 Llama-style decoder layer (hidden 5120; attention 146.8 M because the QKV input is concat[emb, hidden]; MLP 393 M); `fc` 3×5120→5120 (78.6 M); **draft vocab 32,000** (lm_head 163.8 M, with d2t/t2d maps); embedding copy 777.9 M. 1.56 B params total, about 0.78 B streamed per draft step | vLLM 0.11.0, 2×A100, **T=0.6, top_p 0.95, top_k 20**. Mean acceptance length by k (speculative tokens): HumanEval k=1 1.67, k=3 **2.29**, k=5 2.47, k=7 2.53; GSM8K k=3 **2.49**, k=7 **3.08**; CNN/DM k=3 2.15, k=7 2.33 |
| RedHatAI/Qwen3-32B-Thinking-speculator.eagle3 (2026-03-10) | Qwen3-32B in thinking mode | EAGLE-3 | not extracted (gap) |
| **AngelSlim/Qwen3-32B_eagle3** (Tencent, 2025-07-11) | Qwen3-32B | LlamaForCausalLMEagle3, 1 layer, hidden 5120, draft vocab 32000 | H20, batch 1, num_speculative_tokens=2, output 1024: accept length **1.86** (MT-bench), 1.86 (HumanEval), 1.90 (Alpaca), **2.01** (GSM8K); 71.6–80.4 tok/s on H20 |
| Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3; nex-agi/SGLANG-EAGLE3-Qwen3-32B-Nex-N1; Zjcxy-SmartAI/Eagle3-Qwen3-32B-zh; thoughtworks/Qwen3-32B-Eagle3; CMBTech/CMB-Qwen3-32B-Eagle3 | Qwen3-32B | EAGLE-3 | not extracted |
| Snowflake/Arctic-LSTM-Speculator-Qwen3-32B-{bird,longcontext} (2026-06) | Qwen3-32B | Arctic LSTM speculator | not extracted |
| **RedHatAI/Qwen3-8B-speculator.eagle3** (2025-09-19; 15k downloads) | Qwen3-8B | 1 layer, hidden 4096, draft vocab 32000, 1.02 B params incl. embedding copy | (card numbers not extracted; same benchmark harness as the 32B card) |
| **RedHatAI/Qwen3-8B-speculator.dflash** (2026-05-07); `.dflash2` (2026-09-16) | Qwen3-8B | **DFlash** block-diffusion drafter: 5 layers (sliding window 2048), block_size 16, **15 draft tokens per forward**, full 151,936-entry lm_head, aux hidden states from layers [2,10,18,26,34]; 2.29 B params | T=0, thinking off, 15 tokens: per-position acceptance HumanEval 87.6/70.5/55.3%…, **avg accepted length 5.88**; math 6.81; QA 2.80 |
| RedHatAI/Qwen3-8B-speculator.peagle (2026-06-09) | Qwen3-8B | P-EAGLE (parallel drafting) | not extracted |
| AngelSlim/Qwen3-8B_eagle3; Tengyunw/qwen3_8b_eagle3 (8.3k downloads); OWSCloud/qwen3_8b_eagle3; RedHatAI/Qwen3-8B-Thinking-speculator.eagle3 | Qwen3-8B | EAGLE-3 | AngelSlim (H20, 2 tokens): accept length 1.97–2.02 |
| For context: RedHatAI/Qwen3.8-27B-speculator.dspark (2026-09-16) | Qwen3.8-27B | **DSpark** = DFlash + Markov head + confidence head; 5 layers, block 8; aux states from 8 target layers; 1.99 B params; no own lm_head | vLLM 0.29, H100, 8 tokens: **mean length 4.42** (HumanEval), **5.74** (math), 4.22 (QA); "consistently delivers higher throughput … than MTP with the same eight speculative tokens" |

Also present (VERIFIED existence):
- Qwen3-30B-A3B: Tengyunw, AngelSlim (Qwen3-a3B_eagle3), RedHatAI eagle3 and dflash, nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3, lmsys SpecForge.
- gpt-oss: lmsys/EAGLE3-gpt-oss-120b-bf16 (47.6k downloads), RedHatAI gpt-oss-20b/120b eagle3, Snowflake LSTM.
- GGUF-packaged EAGLE-3 drafts exist, e.g. `EntityDeletr/EAGLE3-gpt-oss-20b-GGUF`, which means llama.cpp consumes them.

**Negative results and gaps.**
1. **No published MLX-format EAGLE-3/DFlash weights, and no mlx-lm support.**
   - mlx-lm `main` drops Qwen3.5 MTP tensors: `qwen3_5.py: sanitize() … if "mtp." not in k`.
   - Community PRs #1468/#1469 that attached MTP were **closed unmerged**.
   - Converting the safetensors is trivial; the work is the runtime.
2. The **Qwen3-32B EAGLE-3 acceptance is modest at T=0.6** (τ = 2.2–3.1). Implied per-token α ≈ 0.58–0.69 (`spec_model.out.md` §A).
   - This is far below the EAGLE-3 paper's τ ≈ 6 (greedy, trees, H100).
   - The paper's numbers are not transferable to sampled, chain-drafted Qwen3.
3. No acceptance numbers for the RedHat 32B thinking speculator were extracted. Thinking-mode traces at T=0.6 are the dominant real workload for Qwen3 and need a measurement.

### 2.4 EAGLE-1/2/3 (VERIFIED abstracts and paper text)

- **EAGLE-1** (arXiv 2401.15077): autoregresses at the *feature* (second-to-top layer) level with a one-layer draft fed the shifted token sequence.
- **EAGLE-2** (arXiv 2406.16858): a "context-aware dynamic draft tree" that uses draft confidence as a proxy for acceptance.
  - "speedup ratios 3.05x–4.26x, which is 20%–40% faster than EAGLE-1"; lossless.
- **EAGLE-3** (arXiv 2503.01840): drops feature prediction, adds "training-time test", and fuses low/mid/high features. The draft is one decoder layer reusing the target LM head (§2.3 shows public drafts use a 32K draft vocab).
  - Temp 0, H100, tree depth 8. Speedup and τ:

| target | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM |
|---|---|---|---|---|---|
| LLaMA-3.1-8B-Instruct | 4.40× (6.13) | 4.85× (6.74) | 4.48× (6.23) | 4.82× (6.70) | 3.65× (5.34) |
| LLaMA-3.3-70B | 4.11× (5.63) | 4.79× (6.52) | 4.34× (6.15) | 4.30× (6.09) | 3.27× (5.02) |

  - Vicuna-13B at temp 1: 4.33–5.15× (τ = 5.4–6.2).
  - SGLang throughput gain vs batch: 1.81× (bs 2), 1.62× (8), 1.38× (64).
  - No Qwen3 rows in the paper.

### 2.5 Draft-model choice for Qwen3-32B (VERIFIED sizes; acceptance is a gap)

All Qwen3 sizes share the 151,936 vocabulary, so a separate draft model is drop-in.

| draft | params | 4-bit g64 bytes/token | c (bandwidth only) | notes |
|---|---|---|---|---|
| Qwen3-0.6B | 0.60 B (tied) | 0.34 GB | 0.019 | 28 layers. Latency-bound on GPU: about 28 × ~5 kernels per step, so real c ≈ 0.05–0.1 (EXPERIMENTAL) |
| Qwen3-1.7B | 1.72 B | 0.97 GB | 0.054 | |
| Qwen3-4B | 4.02 B | 2.26 GB | 0.126 | too expensive except with very high α |
| EAGLE-3 (RedHat), 4-bit | ~0.78 B streamed | 0.44 GB | 0.024 | 1 layer, so few kernels; needs target hidden states from 3 layers |

Measured data points:
- **Mirror-SD** (Apple, arXiv 2510.13161; VERIFIED HTML), **Qwen3-32B with a 0.6B draft**: "acceptance lengths … approximately 2.5–5.5 tokens" depending on task; peak 5.84× on math.
  - Measured on **8× M2 Ultra GPUs (tensor parallel) plus 8 ANEs for the draft**, not a single chip.
- **LM Studio 0.3.10** (VERIFIED blog, M3 Pro 36 GB, MLX): Qwen2.5-32B-Instruct-4bit + Qwen2.5-0.5B-4bit on "Write a quicksort algorithm in Python": **7.30 → 17.74 tok/s (2.43×)**.
  - Llama-3.1-8B + 1B: 1.71× (both chat prompts).
  - On an RTX 3090 Ti, llama.cpp with Llama-8B Q8 + 1B on "Plan a 1 day trip to DC" got only **1.05×**.
- mlx-lm issue #1470 reproduces Qwen3-32B-6bit + Qwen3-0.6B speculative decoding on an M5 Max but reports no speed numbers.
- **Gap:** no published single-device acceptance rate for Qwen3-0.6B/1.7B drafting Qwen3-32B at T=0.6. This is a one-day measurement for Sky.

### 2.6 Draft-free and alternative methods (VERIFIED abstracts)

| method | mechanism | headline | caveats |
|---|---|---|---|
| Medusa (2401.10774) | k extra decoding heads on the frozen backbone; tree attention | Medusa-1 ">2.2×" lossless; Medusa-2 "2.3–3.6×" | Medusa-2 fine-tunes the backbone and uses "typical acceptance", which **changes the output distribution** |
| Hydra (2402.05109) | sequentially dependent heads | up to 1.31× over Medusa, 2.70× over autoregressive | needs trained heads (none public for Qwen3-32B found) |
| Lookahead (2402.02057) | Jacobi iteration + n-gram pool, no draft | "up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion"; exact | spends log(FLOPs) per step, so it only pays where k\* is large |
| Prompt-lookup decoding (apoorvumang/prompt-lookup-decoding README, VERIFIED) | match the last n-gram (max 3) against the prompt and copy a continuation of up to 10 tokens | "significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality" (summarization, context QA, multi-turn chat; greedy baseline); ~1× on open-ended text | lossless; trivial; llama.cpp ships several n-gram variants (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`; R5 VERIFIED `common/common.h:173-186`) |
| SuffixDecoding (2411.04975 v3, 2025-10) | suffix trees over prompts *and prior outputs*; adaptive speculation length; CPU, model-free | "up to 5.3×, outperforming … 2.8× faster than EAGLE-2/3 and 1.9× faster than Token Recycling" on agentic workloads (SWE-Bench, text-to-SQL); lossless | gains are workload-specific (repetitive agent loops) |
| ReDrafter (Apple, 2403.09919) | RNN draft on target hidden states, beam search, dynamic tree attention, distillation | "up to 2.8×" on H100; **"up to 2.3×" in MLX on Apple Silicon** (chip not named in the abstract) | needs a trained drafter |
| Apple MTP fine-tune (2507.11851) | masked-input MTP via **gated LoRA** that preserves base outputs; quadratic speculative expansion | "nearly 5×" code/math, "almost 2.5×" chat, "without any loss in quality" | requires fine-tuning the model; no Qwen3-32B weights |
| Mirror-SD (Apple, 2510.13161) | draft on NPU in parallel with target on GPU; early-exit-triggered branch rollouts | 2.8–5.8× wall time; ~30% over EAGLE-3; lossless | multi-device M2 Ultra rig; on one M5 Max, ANE and GPU share 614 GB/s (LIKELY contention) |
| Speculative Speculative Decoding (Kumar, Dao, May; 2603.03251) | the draft precomputes speculations for likely verification outcomes, overlapping draft with verify | "~30% faster than optimized speculative decoding baselines … up to 5× faster than autoregressive" | needs concurrency; on a single GPU it must overlap with the target's bandwidth-bound pass |
| DFlash (2602.06036, ICML 2026) | block-diffusion drafter conditioned on target features; one forward drafts a block | "over 6× lossless acceleration … up to 2.5× higher speedup than … EAGLE-3" (GPU) | drafter is 5 layers + full lm_head (0.6–1.3 B streamed per pass), so it needs large k\* (verifies 16 tokens) — a good match for M5 NA, a bad one for SIMD-only chips |

### 2.7 Multi-token-prediction heads in open models (VERIFIED)

- **DeepSeek-V3** (2412.19437 §5.4.3): "the acceptance rate of the second token prediction ranges between 85% and 90% … delivering 1.8 times TPS."
- **MiMo-7B** (2505.07608): "The first MTP layer achieves a remarkably high acceptance rate about 90%, while even the third MTP layer maintains an acceptance rate above 75%" (AIME24).
- **GLM-4.5 / 4.5-Air** (2508.06471): 1 MTP layer each (Table 1).
- **Qwen3-Next-80B-A3B**: MTP. SGLang `NEXTN` steps 3 / draft 4; vLLM `qwen3_next_mtp`, 2 tokens.
- **Qwen3.5/3.6/3.8 (2026)**: `mtp_num_hidden_layers: 1`, MTP tensors shipped (§1.7).
- **Qwen3-32B and Qwen3-8B: no MTP** (VERIFIED).

Measured MTP on Apple Silicon:

| setup | result | label |
|---|---|---|
| MTPLX (MLX-based, `github.com/youssofal/MTPLX`), README | M5 Max: Qwen3.6-27B 4-bit **30.37 tok/s plain → 81.74 tok/s** "optimized" (2.69×). Qwen3.8-27B 4-bit "65.2 tok/s bare speed at official Qwen sampling", 87.6 "optimized". Claims "exact rejection sampling with residual correction". | UNVERIFIED (self-reported) |
| Independent MTPLX test (rotecodefraktion.de), M3 Max 64 GB (no tensor ops), Qwen3.6-27B 4-bit | mlx_lm 22.17 tok/s; MTPLX without MTP 19.86; **MTP depth 3: 27.5–29.3 (+24–28% net vs mlx_lm)**; "with high-entropy, creative text, acceptance drops" | VERIFIED secondary |
| MTPLX per-depth acceptance (modelfit.io) | 73% / 48% / 32% at depths 1/2/3, so τ ≈ 2.53 | UNVERIFIED |

**Reading (LIKELY).** The same software gives about 1.3× on M3 Max but a claimed 2.2–2.7× on M5 Max. That matches §2.2: the M5's neural accelerators make the 4-token verify pass nearly free, while M3's SIMD-only path does not.

### 2.8 Tree verification (VERIFIED abstracts and HTML)

- **SpecInfer** (2305.09781, ASPLOS'24): candidate sequences from small speculative models are organized as a token tree and "verified against the LLM in parallel using a novel tree-based parallel decoding mechanism … provably preserving model quality".
  - 1.5–2.8× for distributed inference; 2.6–3.5× for offloading-based inference.
- **Sequoia** (2402.12374): DP-optimal tree; "number of tokens generated … continues to grow with the tree size". Speedups:
  - Llama2-7B / 13B / Vicuna-33B on A100: 4.04× / 3.73× / 2.27×;
  - 9.96× for Llama2-70B offloaded on L40 (0.56 s/token vs 5.6).
  - Tree size 128 (A100) and 768 (L40 offload).
- **How tree size should scale with hardware** (EXPERIMENTAL).
  - Choose n_nodes ≤ k\* = R·b/16, and grow the tree until the marginal accepted token per node falls below the marginal cost β(n).
  - M5 Max with NA-fed 4-bit kernels: n ≈ 16–32.
  - M3/M4 class (SIMD only): n ≈ 4–8. A chain of 3–4 is near-optimal there.
  - For MoE, n must be much smaller (§7.3).

### 2.9 Model results for Qwen3-32B on M5 Max (EXPERIMENTAL; `spec_model.out.md` §B)

Baseline T₁ = 17.99 GB / (0.8 × 614 GB/s) = 36.6 ms, i.e. **27.3 tok/s**.

| τ source (T=0.6) | γ | c | β = 0.02 (target kernel) | β = 0.05 | β = 0.15 | β = 0.48 (MLX today) |
|---|---|---|---|---|---|---|
| EAGLE-3 HumanEval τ=2.29 | 3 | 0.03 | 1.99× (54 t/s) | 1.85× (50) | 1.49× (41) | **0.91× (25)** |
| EAGLE-3 GSM8K τ=2.49 | 3 | 0.03 | 2.17× (59) | 2.01× (55) | 1.62× (44) | 0.98× (27) |
| EAGLE-3 GSM8K τ=3.08 | 7 | 0.03 | 2.28× (62) | 1.97× (54) | 1.36× (37) | 0.67× (18) |
| EAGLE-3 CNN/DM τ=2.15 | 3 | 0.03 | 1.87× (51) | 1.73× (47) | 1.40× (38) | 0.85× (23) |
| same, draft cost c = 0.10 (0.6B draft model) | 3 | 0.10 | 1.58–1.83× | 1.48–1.72× | 1.23–1.42× | 0.78–0.91× |
| hypothetical DSpark-class τ=4.4 | 8 | 0.03 | 3.16× (86) | 2.70× (74) | 1.81× (49) | 0.87× (24) |

**Reading.**
- With public Qwen3-32B drafts, **≈45–60 tok/s is the realistic band** on predictable prompts, but only if β ≤ 0.05.
- 60 tok/s on average needs a better drafter (τ ≥ 3.5–4.5). DFlash/DSpark-class drafters reach that on 8B/27B targets, but none is public for Qwen3-32B. Training one is possible: Red Hat `speculators` and SGLang SpecForge are open.


## 3. Exactness and determinism

### 3.1 When is speculative output identical to standard decoding?

**Greedy (T=0).**
- Exactly identical **iff the verifier's logits at every position are bit-identical to the logits plain decoding would compute, and ties break the same way**.
- The accept rule is "accept the draft token if it equals argmax(target logits)". The output is then the target's greedy string, whatever the drafter proposes.
- Exactness is a property of the *kernels*, not of the algorithm.

**Sampling (Leviathan/Chen rejection sampling).**
- Same *distribution* as the target, "within hardware numerics" (Chen et al., VERIFIED quote).
- Not the same *sample* as a non-speculative run with the same seed, because the random draws are consumed differently.

**What mlx-lm and llama.cpp actually do** (VERIFIED from source):
- **mlx-lm** (`mlx_lm/generate.py`, `speculative_generate_step`):
  - Samples the target token independently at each verified position (`sampler` → `mx.random.categorical`).
  - Accepts the draft token only if it is **equal** (`if tn != dtn: break`).
  - CLI default `--num-draft-tokens 3`.
- **llama.cpp**: `common_sampler_sample_and_accept_n` (`common/sampling.cpp:678-710`) does the same. It samples the target and breaks on `draft[i] != id`.

Why this matters (EXPERIMENTAL arithmetic):
- The scheme is **distribution-preserving**: every emitted token is the target's own sample.
- But it is **acceptance-suboptimal**:
  - Its per-position acceptance is Σ_x p_draft(x)·p_target(x), the collision probability.
  - Rejection sampling achieves Σ_x min(p_draft, p_target) = 1 − TV.
- Example: if draft = target = (0.5, 0.5), independent matching accepts 50%, while rejection sampling accepts 100%. At p_max = 0.9 the figures are 82% vs 100%.
- At the T=0.6–0.7 that Qwen recommends, positions with real entropy lose a large share of acceptance.
- MTPLX states it uses "exact rejection sampling with residual correction" (UNVERIFIED implementation). **That is an easy lossless win over both mlx-lm and llama.cpp.**

### 3.2 Bit-identical speculation for *any* temperature: keyed Gumbel sampling (VERIFIED theory; EXPERIMENTAL for Sky)

Construction:
- Sample token t as argmax_x(logits_t(x)/T + g_t(x)).
- The Gumbel noise g_t is generated from a counter-based hash of (seed, request, **position t**). It does not depend on how many tokens were processed together.
- The target's sampled sequence is then a deterministic function of (prompt, seed).
- A speculative run that verifies "draft token == argmax(logits/T + g_t)" emits **exactly the same tokens** as the non-speculative run, provided the logits are bit-identical (§3.3).

Supporting sources:
- **Daliri, Musco & Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding"** (arXiv 2408.07978; VERIFIED abstract):
  - Gumbel coupling gives "Pr[a = b] ≥ (1 − D_TV)/(1 + D_TV)";
  - schemes "whose output is fixed given a fixed random seed, regardless of what drafter is used for speculation."
- **Cost:** acceptance falls from the optimal 1 − TV to between (1−TV)/(1+TV) and 1 − TV. When the draft also uses the same g_t, acceptance is usually close to optimal in practice (LIKELY).
- **SGLang already ships seeded-hash Gumbel sampling.** The deterministic mode uses "Gumbel noise generated from a seeded hash function" so that "the same (inputs, seed) pair always yields the same sample, even when temperature > 0" (VERIFIED, LMSYS blog 2025-09-22).

**Product claim this enables** (needs batch-invariant kernels):
- "Sky with speculation produces byte-for-byte the same text as Sky without speculation, at any temperature, for a given seed."
- The output is also independent of the drafter, the draft length, and the tree shape.

### 3.3 Batch invariance (VERIFIED: Thinking Machines blog, 2025-09; SGLang blog)

**The claim.** "the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies!"
- Kernels are run-to-run deterministic but **not batch-invariant**: reduction order changes with batch size and shape.
- Demonstration: Qwen3-235B-A22B-Instruct-2507, 1000 completions at T=0 → **80 unique completions**, diverging at token 103. With their kernels, all 1000 are identical.

**How to build batch-invariant kernels** (their recipe):
- **RMSNorm**: data-parallel, one row per core; never split a row's reduction differently as batch changes.
- **Matmul**: data-parallel tiles. **No split-K that depends on shape.** One fixed tile configuration and instruction for all M.
- **Attention**:
  - parallelize over queries;
  - use the **same KV-cache layout for prefill and decode** (update the cache before the attention kernel);
  - use a **fixed split *size* (not a fixed number of splits)** for split-KV / FlashDecoding.
  - Chunked prefill and prefix caching must align chunk boundaries with the split size. SGLang: "align the truncation point with an integer multiple of the split_kv_size".

**Measured cost on NVIDIA serving stacks:**
- Thinking Machines, vLLM, Qwen3-8B, 1000 sequences: default 26 s → 55 s unoptimized deterministic → **42 s** with an improved attention kernel (1.62×).
- SGLang: average **34.35%** slowdown on FlashInfer/FA3, range 24–55% by backend and length.
- Both are at serving batch sizes, where split-K and variable tiles matter for throughput.

**Why the cost should be small for Sky at batch 1** (EXPERIMENTAL, LIKELY):
- Decode is bandwidth-bound.
- Running M=1 through the *same* tiled kernel as M=2..16, with rows padded to the tile height and a fixed K-blocking and accumulation order, wastes ALU work, not bandwidth.
- The verify pass then reproduces the decode pass bit for bit.
- Prefill can use a different, compute-optimal kernel as long as K/V written for the prompt are *the same values* whichever chunking produced them. That requires chunk-invariant GEMM and attention.
- To be measured: the cost of padding M=1 to the NA tile, against a specialized GEMV. At ~8 FMAs of ALU budget per weight (§4.3), padding to 8 rows fits, but padding to 32 rows may not on the SIMD path.

### 3.4 Evidence that this matters on Apple today (VERIFIED: GitHub ml-explore/mlx-lm#1470, 2026-07-04, open)

- **Setup:** "speculative_generate_step diverges from plain generation at temp=0 with num_draft_tokens=4" on **M5 Max 128 GB, mlx 0.31.2**, target `mlx-community/Qwen3-32B-6bit`, draft `Qwen3-0.6B-MLX-6bit`.
- **Result:** "First divergence at token index 91." num_draft = 1–3 matched; 4 diverged.
- **Root cause, independently reproduced on three model pairs:**
  - Qwen3-32B-4bit + 0.6B: logits "both tokens logit 33.75 … logit_gap = 0.0 exactly";
  - Qwen3-8B-6bit + 0.6B: diverged at num_draft = 8, index 99;
  - Qwen3-4B + 0.6B.
  - Mechanism: exact bf16 ties combined with "batched forward (num_draft+1 wide)" vs "sequential single-token forwards — same math, different reduction order."
- **MTP work in mlx-lm PRs #1468/#1469** saw "a rare divergence (roughly 1 per 150 to 190 tokens) between speculative and plain generation on quantized models, at temp=0", including a self-speculative control test.
- **Maintainers' disposition:** document it (PR #1592: "lossless up to floating-point associativity, not bit-exact"); no code fix.

Two practical notes (LIKELY):
1. bf16 logits near 32–64 have a spacing of 0.25, so exact ties at the top are common. Computing the final logits in fp32 makes ties rare. It does not fix batch variance.
2. **Cross-precision divergence is large.** "Greedy Decoding Is Not Precision-Invariant" (arXiv 2609.26621, 2026-09; VERIFIED abstract): BF16 vs FP16 on the same hardware → "49–100% of prompts diverge". Selective FP32 LM-head recompute gives "+22–36 pp exact agreement" at <4% overhead at batch ≤ 4.
   - Therefore **never promise bit-identity with MLX, llama.cpp or a BF16 reference.** Promise bit-identity with *Sky's own* non-speculative path, and same-distribution or quality parity with the reference.

### 3.5 Design rules for Sky (EXPERIMENTAL; derived from the above)

1. One decode/verify GEMV-GEMM family with **M-invariant reduction order**. M=1 is the M=k kernel with masked rows. Make this a compile-time invariant: the IR carries a "reduction schedule" attribute that must be identical for the decode and verify instantiations.
2. Attention with a fixed KV-block traversal order and a fixed split size, identical for 1 query and k queries. Tree verification uses masks only, never a different traversal order.
3. KV written by verify must equal KV written by decode for the same position. On rollback, truncate; never recompute with a different kernel.
4. **Hybrid linear attention (Qwen3.5/3.6/3.8):**
   - The chunked parallel scan used for prefill and multi-token verify has a different FP order from the 1-token recurrent update.
   - For bit-exactness, run the recurrent update per position inside the verify pass. Its state math is small: ~6 × 48 × 128² FLOPs per layer per token.
   - Checkpoint the state per draft position for rollback.
5. Logits: compute the final reduction in fp32. Break argmax ties by the lowest token id. Sample with keyed Gumbel noise.
6. **Test harness.** For N prompts × draft lengths 1..16 × temperatures {0, 0.6, 1.0}, assert `spec_tokens == plain_tokens`. The mlx-lm #1470 script is a ready-made adversarial test. It is a cheap, demonstrable differentiator.

Related 2026 work:
- **LLM-42** (arXiv 2601.17768; VERIFIED abstract) gets determinism by "a non-deterministic fast path and … a lightweight verify-rollback loop" that "replays candidate tokens under a fixed-shape reduction schedule".
- The same verify-replay idea can certify Sky outputs when a fast non-invariant kernel is used.


## 4. Quantization

### 4.1 Formats and exact bits per weight

| format | storage per group | bpw | decode work per weight | source |
|---|---|---|---|---|
| MLX affine, bits ∈ {2,3,4,5,6,8}, group 32/64/128 | b-bit codes + fp16 scale + fp16 bias per group | b + 32/g (4-bit: 5.0 / **4.5** / 4.25) | extract bits + 1 FMA; the bias is folded via Σx | VERIFIED `mlx/ops.cpp`: "The supported bits are 2, 3, 4, 5, 6 and 8" |
| MLX `mxfp4` / `nvfp4` / `mxfp8` modes | E2M1 codes + E8M0 scale per 32 / E2M1 + scale per 16 / E4M3 per 32 | 4.25 / ~4.5 / 8.25 | LUT or convert + scale | VERIFIED `QuantizationMode::{Affine,Nvfp4,Mxfp4,Mxfp8}`; expected group 16 for nvfp4, else 32 |
| GGUF Q8_0 / Q6_K / Q5_K / Q4_K / Q3_K / Q2_K | super-blocks of 256 (K-quants) with 6–8-bit sub-scales | 8.5 / 6.5625 / 5.5 / 4.5 / 3.4375 / 2.625 | bit-extract + sub-scale + FMA | VERIFIED `ggml-common.h` |
| GGUF Q4_0 / IQ4_NL / IQ4_XS | 32-blocks, fp16 scale / non-linear 16-entry LUT | 4.5 / 4.5 / 4.25 | IQ4: 16-entry LUT | VERIFIED |
| GGUF IQ3_XXS / IQ3_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ1_S / IQ1_M | E8-lattice-style grids with sign bits (codebooks of 256–2048 entries) | 3.0625 / 3.4375 / 2.0625 / 2.3125 / 2.5625 / 1.5625 / 1.75 | codebook gather + signs | VERIFIED |
| GGUF MXFP4 / NVFP4 / TQ1_0 / TQ2_0 / new Q1_0 / Q2_0 | — | 4.25 / 4.5 / 1.6875 / 2.0625 / 1.125 / 2.25 | — | VERIFIED |
| GPTQ / AWQ | uniform codes, group 128, with error compensation / activation-aware scaling | ~4.125–4.25 | same as affine | papers |
| AQLM (2401.06118) | additive multi-codebook | ~2–3 | multiple codebook lookups + adds | abstract: "Pareto optimal … below 3 bits"; "match or outperform optimized FP16 implementations for speed" (GPU) |
| QuIP# (2402.04396) | randomized Hadamard + E8P lattice codebook | 2–4 | Hadamard transform + lattice decode | abstract |
| QTIP (2406.11235, NeurIPS'24) | trellis-coded quantization (bitshift trellis); codes 1MAD/3INST/HYB | 2–4 | "2 instructions (MAD and &)" + vabsdiff4 (1MAD); "3 ALU instructions" (3INST); HYB "amortized 2 instructions per weight" with a 2 KiB (2⁹-entry) codebook | VERIFIED HTML |
| Apple ASTC (AFM 2025 server model) | 6×6 ASTC HDR blocks of 128 bits | **3.56** | "fixed-function hardware for ASTC decompression within Apple GPUs … effectively zero compute cost" | VERIFIED arXiv 2507.13575 |
| BitNet b1.58 | ternary, trained from scratch | ~1.58–2 | add/sub | 2504.12285: only a 2B4T model is released |

### 4.2 Quality numbers

**(a) Qwen3 itself.** "An Empirical Study of Qwen3 Quantization", arXiv 2505.02214. VERIFIED by parsing Tables 2 and 4 of the HTML. Post-trained (instruct) models; Wiki2 / C4 perplexity and 5-shot MMLU.

| Qwen3-32B | Wiki2 | C4 | MMLU |
|---|---|---|---|
| FP16 | 7.61 | 10.8 | 81.2 |
| W8 RTN / AWQ / GPTQ (per-channel or g128) | 7.60–7.61 | 10.8 | 81.2–81.4 (**lossless**) |
| W4 **RTN per-channel** | **38.5** | 35.0 | 78.4 (perplexity collapses; never use per-channel 4-bit) |
| W4 AWQ per-channel / GPTQ per-channel | 8.21 / 8.33 | 11.3 / 11.4 | 79.9 / 79.1 |
| W4 AWQ g128 / **GPTQ g128** | 8.55 / **7.86** | 11.6 / 11.0 | 78.0 / **80.6** |
| W3 AWQ per-channel / AWQ g128 | 12.3 / 19.1 | 14.9 / 20.7 | 70.0 / **54.9** |
| W2 (any) | ≥38 or NaN | — | 24–28 (collapse) |
| Bi-LLM 1.06-bit g128 | 17.1 | 17.6 | 57.5 |

| Qwen3-8B | Wiki2 | MMLU |
|---|---|---|
| FP16 | 9.71 | 74.7 |
| W8 | 9.69–9.73 | 74.5–74.7 |
| **SmoothQuant W8A8** | 9.62 | **74.0** (−0.7) |
| W4 GPTQ g128 / AWQ g128 | 9.96 / 11.3 | 73.4 / 69.3 |
| **SmoothQuant W4A8** | 12.5 | **63.2** (−11.5) |
| SmoothQuant W4A4 | 3.4e4 | 24.8 (collapse) |
| W3 AWQ g128 | 27.5 | 33.2 |

Authors' conclusions (quote):
- "Qwen3 exhibits more pronounced performance degradation under low-bit quantization (3 bits or fewer)" than LLaMA3.
- "in LLaMA3-8B, AWQ with w3a16g128 … C4 from 9.2 to only 11.6, whereas in Qwen3-8B-Base … from 10.4 to 23.8."

Caveats:
- MLX's default (RTN, affine, **g64**) is not in the tables.
- Expected quality of MLX g64 (LIKELY): between "RTN per-channel" (bad) and GPTQ g128 (good). g64 halves the group size, which helps substantially.
- **Measuring KL(MLX-4bit-g64 ‖ BF16) on Qwen3-32B is an open, cheap experiment.**

**(b) GGUF K-quants / I-quants: KL divergence vs FP16.** Mistral-7B with imatrix (Artefact2 gist, 2024-02-27; VERIFIED page). The "top-token differs" column is the fraction of positions where greedy picks a different token than FP16.

| type | bpw | median KLD | KLD p99 | top-token differs | ln(PPL_q/PPL) |
|---|---|---|---|---|---|
| Q6_K | 6.57 | 0.0032 | 0.022 | **3.9%** | −0.0008 |
| Q5_K_M | 5.67 | 0.0043 | 0.037 | 4.4% | 0.0005 |
| **Q4_K_M** | 4.83 | **0.0075** | 0.089 | **5.8%** | 0.0060 |
| IQ4_XS | 4.32 | 0.0088 | 0.108 | 6.1% | 0.0079 |
| Q3_K_M | 3.89 | 0.0171 | 0.255 | 8.4% | 0.0258 |
| IQ3_XXS | 3.21 | 0.0330 | 0.549 | 11.4% | 0.0589 |
| Q2_K | 3.00 | 0.0588 | 1.034 | 14.9% | 0.110 |
| IQ2_XXS | 2.20 | 0.175 | 2.50 | 23.1% | 0.299 |
| IQ1_S | 1.78 | 0.550 | 5.52 | 38.4% | 0.924 |

**Reading.** Even Q6_K changes the greedy token at ~4% of positions, so *every* quantized model's text diverges from BF16 within tens of tokens. "Same model and quantization" is the only fair comparison basis, and quality must be stated as KL or task deltas.

**(c) 2-bit codebook and trellis state of the art** (QTIP Table 5, Llama-2, ctx 4096, Wiki2):

| model | FP16 | QTIP 2-bit | QuIP# 2-bit | AQLM 2-bit |
|---|---|---|---|---|
| Llama-2-70B | 3.12 | 3.70 | 3.91 | 3.83 |
| Llama-2-7B | 5.12 | 5.86 | 6.19 | 6.14 |

These use fine-tuning. Given Qwen3's measured sensitivity, expect larger 2-bit gaps on Qwen3 (LIKELY).

**(d) Apple Foundation Models 2025** (arXiv 2507.13575; VERIFIED HTML):
- On-device ~3B: "compressed to 2 bits-per-weight using Quantization-Aware-Training"; embedding 4-bit; **KV cache 8-bit**; LoRA adapters recover quality. MMLU 67.8 → 64.4, IFEval 85.1 → 82.3.
- Server PT-MoE: **ASTC 3.56 bpw**, decoded by GPU texture hardware "with effectively zero compute cost"; singular vectors moved into LoRA before compression. MMLU 80.0 → 79.2, IFEval 89.1 → 90.2.

**(e) QAT at 2–3 bits.** ParetoQ (2502.02631; VERIFIED abstract): "a notable learning transition between 2 and 3 bits: For 3-bits and above, the fine-tuned models stay close to their original pre-trained distributions, whereas for learning 2-bit networks or below, the representations change drastically."

### 4.3 What is practical for Apple GPU decode? An ALU budget per weight (EXPERIMENTAL arithmetic)

At bandwidth BW and b bits per weight, the GPU must consume BW·8/b weights per second.

M5 Max SIMD: 40 cores × 128 lanes × ~1.6 GHz ≈ **8.2 × 10¹² lane-ops/s**, with FP16 possibly ~1.5× that.

| b (bpw) | weights/s at 614 GB/s | lane-ops available per weight |
|---|---|---|
| 8.5 | 0.58 × 10¹² | ~14 |
| 4.5 | 1.09 × 10¹² | **~7.5** |
| 3.5 | 1.40 × 10¹² | ~5.8 |
| 2.0 | 2.46 × 10¹² | **~3.3** |

What each scheme needs per weight:
- Affine 4-bit: 1 FMA for the dot product plus ~1–2 ops of amortized unpack (shift/mask, int→fp). Total ~2–4 of 7.5: **fits.** MLX reaches about 77–88% of bandwidth on M2 Pro per R3.
- A 2-bit trellis or lattice code needs ~2–3 decode ops plus an FMA plus unpack, i.e. ≥4–5 ops against a budget of **3.3**: **ALU-bound**.

GPU evidence that codebook methods are ALU-limited at batch 1 (VERIFIED QTIP Table 4, RTX 6000 Ada, 960 GB/s):
- Llama-2-70B 2-bit QTIP: **23.5 tok/s** = 17.5 GB × 23.5 = 411 GB/s, i.e. **43% of peak bandwidth**.
- Llama-2-7B: 188 tok/s ≈ 1.75 GB × 188 = 329 GB/s (34%).
- That GPU has ~3.5× more ALU per byte than an M5 Max: 91 TFLOPS fp32 / 960 GB/s ≈ 95 FLOP/B vs ~27.

**Conclusion (LIKELY).** On Apple GPUs, codebook, lattice and trellis 2-bit formats would recover only part of their 2.25× byte saving. Affine/K-quant-style formats at 3–4.5 bpw, plus small LUTs (IQ4_NL-style), are the practical family.

llama.cpp corroborates this in its own README table (VERIFIED `tools/quantize/README.md`, Llama-3.1-8B, hardware unspecified): text generation **IQ3_S (3.66 bpw) 69.3 t/s < IQ4_XS (4.46 bpw) 77.5 t/s**, and IQ2_XXS (2.38 bpw) 79.9 t/s ≈ Q4_K_S (4.67 bpw) 76.7 t/s. Fewer bytes did not mean faster for codebook types.

**ASTC (MOONSHOT/EXPERIMENTAL).**
- Apple's own server model streams 3.56-bpw weights through texture decompression at no ALU cost.
- If texture sampling can sustain ~600 GB/s of compressed reads on M5 Max (unknown; needs a microbenchmark), Qwen3-32B would stream 31.98 B × 3.56 / 8 = **14.2 GB/token → 43 tok/s roofline**.
- It changes outputs. Apple needed LoRA recovery; Qwen3 is quantization-sensitive.

### 4.4 Mixed-bit allocation under a KL budget

Already shipping (VERIFIED):
- **mlx-lm**: `mlx_lm.dynamic_quant` estimates "the sensitivity of a model's outputs to each layer" and uses "higher precision (default 5 bits) for the more sensitive layers and lower precision (default 4 bits) for the rest". `--target-bpw` range [4.5, 5.5] with defaults.
  - `mlx_lm.dwq` (Distilled Weight Quantization) fine-tunes scales and biases against the unquantized teacher: "DWQ works best distilling to lower precision, anywhere from 2-bit to 4-bit."
  - mlx-lm also has AWQ and GPTQ.
  - **No published DWQ quality numbers** were found in the docs or the mlx-community DWQ model cards (gap).
- **llama.cpp**: fixed K-quant mixes (`use_more_bits`, §1.3), imatrix-weighted rounding, and per-tensor `--tensor-type` overrides.
- **EvoPress** (2410.14649; VERIFIED abstract): evolutionary search over per-layer compression levels, because "the independence assumption does not generally hold"; state of the art "for … quantization with dynamic bitwidths" on Llama/Mistral/Phi.

**Could a compiler search this under a KL budget? Yes** (LIKELY, with precedent).
- Objective: minimize streamed bytes subject to E_calib[KL(p_BF16 ‖ p_q)] ≤ ε, or a task-accuracy proxy.
- Search space: per-tensor (bits ∈ {3,4,5,6,8}, group ∈ {32,64,128}).
- Method: greedy sensitivity first (mlx-lm style), then evolutionary refinement (EvoPress style), with KL measured end to end.
- This is a natural "compiler pass" with a quality certificate.
- Expected payoff for Qwen3-32B: about 4.5 → 3.8–4.0 average bpw at Q4_K_M-like KL, i.e. **1.12–1.18× decode** (EXPERIMENTAL).
- 3-bit-average models are not credible for Qwen3 without QAT: AWQ W3 g128 already gives MMLU 81.2 → 54.9.

### 4.5 Activation quantization and INT8 prefill (relevant to the M5 neural accelerators)

- The M5 NA runs **INT8 at ~2× its FP16 rate** (tzakharko estimate).
- W8A8 is near-lossless on Qwen3-8B: SmoothQuant gives −0.7 MMLU and unchanged perplexity.
- W4A8 with SmoothQuant is poor: −11.5 MMLU. Rotation-based methods (QuaRot, SpinQuant) do better.
  - SpinQuant W4A4KV4 leaves "merely 2.9 points" on LLaMA-2 7B zero-shot.
  - Qwen3-specific W4A8 numbers with rotations are a gap.

INT8 prefill is a **different-output** technique:
- The prompt's KV cache, and hence all later tokens, change.
- Use it only with a measured KL budget, or as an opt-in "fast prefill" mode.
- It is the *only* route to >1,100 tok/s prefill on Qwen3-32B (§10).

### 4.6 FP8, MXFP4/NVFP4, BitNet

- **FP8.** No FP8 (or BF16) matrix support in first-gen M5 NA per tzakharko. A Metal 4.1 microbench reports fp8×fp8 at FP16 rate, so FP8 only saves bytes (VERIFIED secondary; UNVERIFIED for M5 Max).
- **MXFP4.** 4.25 bpw with power-of-two (E8M0) scales.
  - Native and lossless for **gpt-oss**, which was trained/released in MXFP4 for its MoE weights (VERIFIED config).
  - As post-hoc quantization of Qwen3, the coarse scales are LIKELY worse than affine g64 at similar bits. No Qwen3 numbers found (gap).
- **NVFP4** (16-element groups, E4M3 scales): better post-hoc quality than MXFP4 on NVIDIA studies (LIKELY; not verified here). Apple has no FP4 compute, so on Apple it is just another dequant format.
- **BitNet b1.58.**
  - Requires native ternary pretraining. Only "BitNet b1.58 2B4T" is released by Microsoft (VERIFIED).
  - There is no Qwen3-32B BitNet.
  - MTPLX mentions a "Ternary Bonsai 2 27B" running in 11.4 GB on M5 Max at 64 tok/s (UNVERIFIED; worth a look as a 2026 ternary 27B model).

### 4.7 Lossless bandwidth reduction: entropy-coded weights (EXPERIMENTAL)

Simulation (`entropy_sim.py`): MLX-style affine g64 codes have entropy
- **3.59–3.75 bits** at 4-bit (Laplace … Gaussian weights; Student-t(4) 3.67);
- **2.54–2.69** at 3-bit.

Entropy coding the codes, keeping fp16 scales and biases, would cut 4.5 bpw to ≈ 4.09–4.25 bpw, i.e. **5.6–9% fewer bytes, bit-identical output**.

Difficulty and precedent:
- Variable-length decoding at ~10¹² symbols/s eats into the ~7.5 op/weight budget (§4.3). Use many independent streams with table-driven decode (LUT in threadgroup memory).
- Precedent: **DFloat11** (2504.11651; VERIFIED abstract) Huffman-codes the BF16 exponent for "around 30% model size reduction while preserving bit-for-bit identical outputs" with a GPU decompression kernel. That is the BF16 analogue.
- Rank: low gain, high effort, but genuinely lossless.


## 5. KV cache

### 5.1 Size and bandwidth

See §1.5 for the numbers. For Qwen3-32B the KV cache is 256 KiB per token in fp16. Reading it equals the 4-bit weight bytes at T ≈ 68K tokens; it is 48% of them at 32K and 6% at 4K.

### 5.2 Quantized KV quality (VERIFIED abstracts)

- **KIVI** (2402.02750, ICML'24): tuning-free **2-bit**, "the key cache should be quantized per-channel" and "the value cache … per-token".
  - Llama/Falcon/Mistral keep "almost the same quality".
  - 2.6× lower peak memory, up to 4× batch, 2.35–3.47× throughput (serving).
- **KVQuant** (2401.18079, NeurIPS'24): per-channel and **pre-RoPE** keys, non-uniform levels, dense-and-sparse outliers.
  - "< 0.1 perplexity degradation with 3-bit quantization" (LLaMA, Llama-2/3, Mistral); 1M context on one A100.
- **Apple Foundation Models (2025)** ship an **8-bit KV cache** in production on device and on the server (VERIFIED).
- **Qwen3-specific KV-quant studies:** none found (gap). Note that Qwen3 has QK-norm, which should make key quantization *easier* (bounded key norms) — LIKELY, untested.

### 5.3 MLX quantized KV (VERIFIED)

- mlx-lm options: `--kv-bits`, `--kv-group-size`, `--quantized-kv-start`.
  - `generate.py`: `DEFAULT_QUANTIZED_KV_START = 5000`; `maybe_quantize_kv_cache` converts the cache once `offset >= quantized_kv_start`.
- **R3: MLX core has no fused quantized-KV SDPA kernel.** mlx-lm runs attention through quantized matmuls (`R3-kernels.md`).
- **ml-explore/mlx-lm#1651** (2026-08-03): "`quantized_kv_start` default disagrees between `generate_step` and the CLI" (library 0 vs CLI 5000).
  - Setup: M4 Pro, `Qwen3-0.6B-4bit`, `kv_bits=8`, 8-token prompt.
  - Quantizing from step 0 was **slower**: 325.4 vs 375.4 tok/s at 64 generated tokens (−13%), 314.2 vs 382.3 at 128 (−18%), 322.8 vs 341.9 at 1024 (−6%).
  - It saved memory only at length: 169 MB at 1024 tokens.
  - Closed by PR #1819, which aligns to the CLI default.
- **Lesson.** Without a fused dequant-in-attention kernel, KV quantization costs speed at short context and helps only at long context. A fused 8-bit KV attention kernel is a clean win for Sky at ≥8K context (1.05–1.18× decode, §1.5). It changes outputs slightly.

### 5.4 Sliding windows and attention sinks (VERIFIED)

- **StreamingLLM** (2309.17453): keeping the initial tokens' KV ("attention sinks") stabilizes window attention up to 4 M tokens; 22.2× over sliding-window recompute.
  - Evicted tokens are forgotten. This is not long-context retrieval.
- **gpt-oss** ships it natively:
  - config `layer_types` alternates `sliding_attention`/`full_attention`, `sliding_window: 128`, plus learned per-head sinks;
  - KV grows only in half the layers (§1.6).
- **Qwen3 dense:** `use_sliding_window: false`, full attention everywhere.
- **Qwen3.5/3.6/3.8:** 3:1 Gated-DeltaNet:attention, so KV shrinks 4× (§1.7).
- Retro-fitting a window onto Qwen3-32B changes outputs and is not recommended.

### 5.5 KV eviction (VERIFIED abstracts): risky

- **H2O** (2306.14048): keep "heavy hitters" plus recent tokens. With 20% heavy hitters, up to 29× throughput vs FlexGen-class systems, and latency up to 1.9× lower at the same batch.
- **SnapKV** (2404.14469): an observation window at the end of the prompt selects per-head KV positions. 3.6× faster generation and 8.2× memory efficiency at 16K inputs, "comparable accuracy" on LongBench.
- **SCBench** (2412.10319): "sub-O(n) memory methods suffer in multi-turn scenarios"; "attention distribution shift issues in long-generation scenarios".

**Verdict.** Eviction changes outputs and fails on shared-context and multi-turn use (agent loops, chat history), which is the typical local workload. Do not make it a default; at most offer an opt-in long-document mode.

### 5.6 Prefix and prompt caching (exact, if batch-invariant)

- Mechanisms: vLLM automatic prefix caching, SGLang RadixAttention, mlx-lm prompt cache (`make_prompt_cache`, `load_prompt_cache`, `trim_prompt_cache` used in `generate.py`; VERIFIED), llama.cpp slot save/restore and `--cache-reuse` (R5).
- TTFT then scales with **new** tokens only.
- **Exactness:** reused KV equals freshly computed KV only if prefill is chunk/batch-invariant (§3.3). SGLang had to align chunk boundaries to the split-KV size to keep its radix cache deterministic.
- A persistent on-disk prompt cache is ~0.26 MB per token (fp16) for Qwen3-32B, or 0.07 MB at 4-bit. A 4K-token system prompt costs 1.07 GB at fp16.

### 5.7 Apple "KV Prediction for Improved TTFT" (arXiv 2410.08391; VERIFIED HTML)

**Mechanism.** A small auxiliary model processes the prompt and *predicts the base model's KV cache*. Generation then continues "exclusively with the base model". Training uses a consistency loss L1(predicted, true KV).

**Numbers** (OpenELM):
- OE-1.1B with a 450M auxiliary: **2.36× fewer TTFT FLOPs**, TriviaQA 23.57% → **16.09%**.
- OE-1.1B layer-pruned 0.50: 1.93×, TriviaQA 17.59%, HumanEval 15.79 → 11.51.
- Layer-pruned 0.75: 1.34×, HumanEval 13.82.
- Measured on an **M2 Pro CPU**: 5.59 s → 3.17 s (**1.76×**) and 3.33 s (1.68×).

**Verdict (negative).**
- It changes outputs, with large accuracy losses.
- It needs a trained predictor per base model; none exists for Qwen3.
- It has only been shown at 1–3B scale.

---

## 6. Sparsity

### 6.1 Hardware

**No evidence of structured-sparsity support** (2:4 or otherwise) on Apple GPUs, the M5 neural accelerators, or the ANE.
- Checked: Apple newsroom M5 Pro/Max pages, the MLX M5 post, the tzakharko NA characterization (formats only), and web search. All negative.
- Label: LIKELY none. Apple has not documented any.
- A community project (`Tekraj15/metalSPARSE`) claims software 2:4 kernels with "up to 2.85×" (UNVERIFIED).

**On a bandwidth-bound GPU, sparsity only helps if fewer bytes are read.**

### 6.2 Activation (contextual) sparsity (VERIFIED abstracts and HTML)

**TEAL** (2408.14690, ICLR'25): training-free magnitude thresholding of the hidden states feeding the projections.

| sparsity | Llama-3-8B PPL (base 5.87) | 6-task avg (base 68.07) |
|---|---|---|
| 25% | 5.94 | — |
| 40% | 6.21 | 66.21 |
| 50% | 6.67 | 63.42 |

- Decode speedups at batch 1: **1.53× at 40% and 1.8× at 50% on A6000**; about 1.25× / 1.40× on A100.
- Kernel: "weights in column major format", loads only the active columns.
- Compatible with 4-bit AWQ and 2/3-bit QuIP#; the errors "compound somewhat independently".
- **At batch 4**, "roughly 38% sparsity [is needed] versus 60% in single batch" for the same error. Gains concentrate at batch 1.

**CATS** (2404.08763): within 1–2% of base on downstream tasks at 50% MLP activation sparsity; "~15% improvement in wall-clock inference latency".

**R-Sparse** (ICLR'25): 50% model-level sparsity, "43% end-to-end efficient improvements".

**Deja Vu** (2310.17157): predicts contextual sparsity; >2× over FasterTransformer on OPT-175B (ReLU-era models).

**For Sky on Qwen3-32B (EXPERIMENTAL).**
- At 40% input sparsity on all projections, bytes drop ~35–40%, i.e. **≤1.5× decode**.
- It changes outputs. No Qwen3 TEAL numbers were found (gap; Qwen3's quantization sensitivity suggests caution).
- Weights must be stored with input-channel-contiguous layout and groups along the output dimension.
- **It conflicts with speculation:** k verified tokens activate the *union* of their channels, which destroys sparsity as k grows (TEAL's batch-4 result).
- Treat it as an alternative to speculative decoding for unpredictable, high-entropy text, not as an addition.

### 6.3 Weight sparsity (VERIFIED abstracts)

- **SparseGPT** (2301.00774): one-shot 50–60% unstructured on OPT-175B/BLOOM-176B "with negligible increase in perplexity"; supports 2:4 and 4:8.
- **Wanda** (2306.11695): magnitude × activation pruning, no retraining.

**Without sparse hardware:**
- A 50%-sparse 4-bit tensor needs ~2 bits of values + 1 bit of mask + scales ≈ 3.5 bpw. That is the same bytes as dense 3-bit affine.
- "Pruning vs Quantization: Which is Better?" (2307.02973): "in most cases quantization outperforms pruning."

**Verdict:** no bandwidth win over simply using fewer quantization bits. Not recommended.

---

## 7. MoE on Apple

### 7.1 Active-parameter economics (VERIFIED configs; `model_math.out.md`)

**Qwen3-30B-A3B** (128 experts, top-8, moe_I = 768, 48 layers):
- 30.53 B total, **3.04 B streamed per token** (the card's 3.3 B includes the embedding).
- 4-bit g64: **1.71 GB/token → 359 tok/s roofline** at 614 GB/s.
- Expert weights total 28.99 B (16.3 GB at 4.5 bpw), all resident on any ≥24 GB Mac.

**gpt-oss-120b** (128 experts, top-4, 36 layers):
- 116.83 B total, 5.13 B active.
- Native MXFP4 experts + BF16 rest: 5.00 GB/token → 123 tok/s.
- With 8-bit non-expert weights: 3.55 GB → 173 tok/s. **The BF16 lm_head alone is 1.16 GB per token.**

**gpt-oss-20b:** native 3.71 GB → 166 tok/s; 2.56 GB → 240 tok/s with 8-bit non-expert weights.

**Measured (VERIFIED, BaseRT arXiv 2607.19438, M5 Pro 48 GB, 307 GB/s):** Qwen3.6-35B-A3B 4-bit decode

| engine | tok/s |
|---|---|
| BaseRT | 110.7 |
| **MLX** | **101.9** |
| llama.cpp | 63.1 |

- BaseRT prefill on Qwen3-30B-A3B at pp2048: 3,834 tok/s vs llama.cpp 1,740.
- Roofline check (EXPERIMENTAL, from safetensors shapes):
  - Qwen3.6-35B-A3B streams about 2.95 B params per token: 8 of 256 routed experts (1.007 B over 40 layers), shared experts 0.126 B, router 0.021 B, linear attention 1.012 B, full attention 0.273 B, lm_head 0.509 B.
  - At 4.5 bpw that is 1.66 GB, plus ~0.13 GB of fp32 DeltaNet state traffic, i.e. **≈1.79 GB/token → ≈172 tok/s roofline at 307 GB/s**.
  - **MLX reaches ≈59% of it; BaseRT ≈64%.**
- MoE decode loses efficiency to many small GEMVs (moe_I = 512–768), the router, and gather/scatter. This is a kernel and fusion problem, i.e. compiler territory.

### 7.2 What mlx-lm and llama.cpp do (VERIFIED source)

- **mlx-lm** `models/switch_layers.py`: `SwitchGLU` runs `mx.gather_qmm` for gate, up and down as **three separate gathered quantized matmuls**.
  - "When we have many tokens, then sort them" (`do_sort = indices.size >= 64`): prefill sorts tokens by expert; decode does not.
- **llama.cpp**: `ggml_mul_mat_id`.
  - `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` puts "MoE tensors → MXFP4, other tensors → Q8_0" (`llama-quant.cpp`).
  - CPU and expert offload flags (R5 covers these).

### 7.3 Speculative decoding on MoE at batch 1 (EXPERIMENTAL arithmetic + VERIFIED papers)

Verifying k tokens touches the **union** of their experts. Under independent routing (an upper bound; real routing is correlated), the expected number of distinct experts is E·(1 − (1 − K/E)^k).

| model | k=1 | k=2 | k=4 | k=8 | streamed bytes at k=4 vs k=1 | at k=8 vs k=1 |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (128, top-8) | 8 | 15.5 | 29.1 | 51.6 | **2.57×** | 4.25× |
| gpt-oss-120b (128, top-4) | 4 | 7.9 | 15.3 | 28.7 | **2.97×** | 5.31× |
| gpt-oss-20b (32, top-4) | 4 | 7.5 | 13.2 | 21.0 | 2.53× | 3.82× |

**Verification is not free for MoE**, unlike dense models (1.00× until k\*). The speedup collapses unless acceptance is very high or drafts share experts.

Supporting papers (VERIFIED abstracts):
- **MoESD** (2505.19645): MoE "benefits more from SD than dense models" **at medium batch sizes**, where expert loads are shared anyway.
- **EcoSpec** (2607.12696, 2026): "confidence-driven SD can introduce *expert scattering*". Cost-aware draft selection gives up to 1.62× on DeepSeek-V3.1, Qwen3-235B and gpt-oss-120b.
- **"The Limits of Speculation"** (2609.22156) formalizes the marginal-cost rule.

Claims for MoE on Apple:
- MTPLX claims 125.8 tok/s on M5 Max for "Qwen 3.8 Flash Next" (an MoE) with MTP depth 3 (UNVERIFIED).

### 7.4 Expert caching, prefetch and offload

- **Unified memory changes the problem.** On a 64–128 GB M5 Max every target model's experts are resident, so offload is irrelevant. It matters on 16–36 GB Macs, iPhones, and for ≥120B models.
- Literature: Mixtral offloading (2312.17238), HOBBIT (2411.01433, mixed-precision expert offloading), MoE-APEX (2026), HybriMoE (DAC'25).
- Apple-device data, "Paging the Experts" (arXiv 2609.29032, iPhone, Qwen3.6-35B-A3B; VERIFIED abstract):
  - 512 MiB LRU expert cache: **0.00%** hits;
  - seeded random eviction: 18.80%;
  - 576 MiB LRU: 38.58%.
  - "The apparent capacity cliff is … a policy/workload interaction."
- Apple "LLM in a flash" (2312.11514): windowing plus row-column bundling. "Models up to twice the size of the available DRAM, with a 4–5x and 20–25x increase in inference speed compared to naive loading approaches in CPU and GPU."

### 7.5 Where a compiler helps MoE (EXPERIMENTAL)

1. Fuse router GEMV + softmax + top-k + gather(gate‖up) + SwiGLU + gather(down) + weighted combine into one or two kernels per layer. mlx-lm uses about 3 gathers plus several elementwise kernels.
2. Concatenate gate and up per expert in the weight layout, so one gather reads one contiguous block.
3. Expert-deduplicated verify for speculation: load each distinct expert once per pass; choose drafts by expected marginal expert cost (EcoSpec idea).
4. Keep non-expert weights at 8-bit, not BF16, for gpt-oss (1.4× on its decode).

---

## 8. TTFT techniques

| technique | effect on TTFT | output | notes |
|---|---|---|---|
| Chunked prefill (Sarathi-Serve, 2403.02310) | none at batch 1 | identical if chunk-invariant | Built for mixed prefill/decode batches ("2.6× higher serving capacity" Mistral-7B/A100). Locally it bounds activation memory: mlx-lm `DEFAULT_PREFILL_STEP_SIZE = 2048` (VERIFIED). |
| Prefix/prompt cache | TTFT ∝ new tokens only; e.g. chat turn N pays only for its new message | identical if batch-invariant (§3.3, §5.6) | the biggest legitimate TTFT lever for chat and agents |
| Faster prefill kernels (NA) | MLX M5 post: TTFT "3.3x to 4.06x" vs M4 at 4096-token prompts (Qwen3 1.7B–14B, gpt-oss-20b, Qwen3-30B-A3B); "under 10 seconds for a dense 14B" on base M5. BaseRT: up to 3.9× MLX prefill on M5 Pro | identical (FP16) | VERIFIED. Kernel work, not algorithm. |
| INT8 (W8A8 / W4A8) prefill | up to ~2× compute ceiling (NA INT8 ≈ 2× FP16) | **different** (KV changes) | Qwen3-8B W8A8: −0.7 MMLU (§4.5) |
| Sparse-attention prefill (MInference 2407.02490; FlexPrefill, XAttention) | "up to 10x for pre-filling" at 1M tokens (A100) | different | Qwen3-32B attention is only 6.4% of prefill FLOPs at 4K and 36% at 32K, so ≤1.07× at 4K and ≤1.5× at 32K. Irrelevant for short prompts. |
| KV Prediction (Apple) | 1.7–2.4× fewer TTFT FLOPs | different; large accuracy loss | §5.7: no Qwen3 predictor |
| LLM in a flash / mmap + wired weights | cold start only | identical | Qwen3-32B 4-bit = 18.4 GB to page in. At a few GB/s of SSD that is several seconds. **TTFT claims must specify warm vs cold.** |


## 9. Ranked techniques that raise effective tok/s beyond the bandwidth roofline (Qwen3-32B, M5 Max)

Baseline assumptions:
- Plain-decode roofline: 34.1 tok/s for 4-bit g64 at short context.
- Realistic plain: about 27 tok/s at 80% of peak.
- Speedups are relative to a well-implemented plain decoder. "Effective tok/s" means emitted tokens ÷ wall time.

| # | technique | expected gain (EXPERIMENTAL unless noted) | output guarantee | difficulty |
|---|---|---|---|---|
| **1** | **EAGLE-3 speculation** with public weights (`RedHatAI/Qwen3-32B-speculator.eagle3`, `AngelSlim/Qwen3-32B_eagle3`). Requires **near-free small-M verify kernels** (β ≤ 0.05, k ≤ 8 on SIMD / ≤ 16–32 with NA tiles) and **exact rejection sampling** (or keyed-Gumbel verify). | **1.6–2.2×** on code/math-like text at T=0.6 (τ 2.3–3.1 published; 43–59 tok/s). **1.4–1.8×** on chat. ~1.2–1.4× on creative/high-entropy text. Up to ~2.5× greedy with trees if NA verify holds. With MLX-class kernels (β ≈ 0.48): **≤1×**. | **Same distribution** (rejection sampling). **Bit-identical to Sky's own plain path** for greedy, and for any temperature with keyed Gumbel, *if* kernels are batch-invariant. | High. Hidden-state taps from 3 target layers, draft KV, 32K→151K vocab map, tree mask + RoPE positions, rollback, small-M kernels. Chain first, trees second. |
| **2** | **Model-free speculation**: prompt lookup + suffix trees over the session and prior outputs. Composes with #1 by taking the best proposal. | 2–4× on input-grounded tasks (PLD README); up to 5.3× on agentic loops (SuffixDecoding); ~1.0× on free text | Same distribution / bit-identical (as #1) | **Low**. CPU-side structures reuse #1's verify path. |
| **3** | **Separate small draft model** (Qwen3-0.6B / 1.7B; shared vocabulary) | 1.3–2.4× (LM Studio: 2.43× for a 32B + 0.5B code prompt on M3 Pro); can be negative on weak chips or small targets | Same distribution | Low–medium. The draft is latency-bound, so it needs a fused small-model step (c ≈ 0.05–0.1). |
| **4** | **Train a DFlash/DSpark-class block drafter** for Qwen3-32B (none public; Red Hat `speculators` and SpecForge are open) | τ 4.4–5.9 seen on 8B/27B targets → **2.5–3×** if a 16-token verify is ~free on NA | Same distribution | High. Needs GPU training plus a block-draft runtime. |
| **5** | **Fused 8-bit KV attention** | 1.03× at 4K, 1.05× at 8K, 1.10× at 16K, **1.18× at 32K** (fp16→8-bit, §1.5); also −47% KV memory | Different output (small; Apple ships 8-bit KV) | Medium |
| **6** | **Mixed-bit weights under a KL budget** (compiler search, plus DWQ-style scale tuning) | 1.10–1.18× (4.5 → ~3.8–4.1 bpw) | **Different output**, KL-bounded. Not allowed under "same quantization" comparisons. | Medium |
| **7** | **Lossless entropy-coded 4-bit codes** | ≤1.06–1.10× (code entropy 3.59–3.75 bits) | **Bit-identical** | High (GPU variable-length decode inside the GEMV) |
| **8** | **Activation sparsity** (TEAL-style, 40%) | ≤1.3–1.5×; erodes when combined with speculation | **Different output** (Llama-3-8B PPL 5.87→6.21 at 40%); no Qwen3 data | Medium–high (column-major quantized layout, sparse GEMV) |
| **9** | **ASTC hardware-decoded weights**, Apple-FM style (MOONSHOT) | ≤1.26× (3.56 vs 4.5 bpw) if the texture path sustains bandwidth | **Different output**; needs LoRA/QAT recovery | Very high; unknown throughput |

**Explicitly not recommended** (negative results):
- KV eviction (H2O, SnapKV): changes outputs; fails multi-turn (SCBench).
- KV Prediction: large accuracy loss; no Qwen3 predictor.
- 2:4 or unstructured weight sparsity: no Apple hardware; quantization beats pruning at equal size.
- 2-bit codebook/trellis PTQ on Qwen3: ALU-bound on Apple, and Qwen3 collapses at ≤3-bit PTQ (W3 AWQ g128 MMLU 81.2→54.9).
- BitNet: needs pretraining.
- Medusa-2 typical acceptance: changes the distribution.
- Sparse-attention prefill at short prompts: attention is only 6% of FLOPs at 4K.

**Strategic (LIKELY).** The single biggest "physics" lever in 2026 is the *model*.
- Qwen3.6/3.8-27B streams 14.4 GB/token instead of 18.0 GB (−20%).
- KV is 4× smaller.
- It ships a trained MTP head; DSpark drafters reach τ ≈ 4.4–5.7.
- MTPLX (MLX-based) already claims 65–88 tok/s on M5 Max (UNVERIFIED).
- Sky must support hybrid DeltaNet layers and MTP with exact rollback to be competitive.

---

## 10. The owner's target table: what is physically impossible, and the honest version

Hardware basis (VERIFIED unless noted):
- M5 Max 40-core GPU: 614 GB/s.
- NA ≈ 65–70 TFLOPS FP16 and ≈ 122–130 TOPS INT8 (secondary-source estimates).
- Qwen3-32B: 17.99 GB/token (MLX 4-bit g64); 63.5–66.7 GFLOP/token prefill at P = 1–4K.

| metric | owner (MLX → Sky) | physics | verdict | honest target |
|---|---|---|---|---|
| **Decode** | 35 → 60 tok/s | roofline **34.1** at 0 ctx, 32.2 at 4K, 30.5 at 8K | **MLX 35 is impossible** (>100% of peak). Realistic MLX is 24–29 (EXPERIMENTAL; M3 Max MLX ran Qwen3.6-27B at ~80% of roofline; MTPLX's M5 Max plain number implies ~71%). **Plain 60 needs ≤2.56 bpw**, i.e. a different, degraded model (W3 already collapses on Qwen3). | Plain: **29–31 tok/s** (85–90% of peak), bit-exact, ≈1.1–1.25× MLX. Speculative (EAGLE-3, same distribution): **45–55 tok/s** typical on code/math at T=0.6 and **≥60 on input-grounded or greedy workloads**. Always quote workload, temperature and context. |
| **Prefill** | 800 → 1,600 tok/s | 1,600 × 64–67 GFLOP = **102–107 TFLOP/s** > FP16 NA peak | **Impossible in FP16** (ceiling ~1,050–1,100 tok/s at 100% utilization). INT8 needs ~80% of INT8 peak and changes outputs. | FP16: **750–950 tok/s** (70–90% NA utilization); first measure what MLX actually gets. Opt-in INT8 "fast prefill": 1,200–1,500 tok/s under a stated KL budget. |
| **RAM** | 24 → 20 GB | weights 18.43 GB (g64) / 17.41 (g128); KV 0.256 MB/token fp16 | Possible **only at bounded context**. 20 GB at 32K is impossible with fp16 KV (27 GB). | 20 GB total: **≤4.4K ctx** (fp16 KV, g64), ≤8.4K (8-bit KV), ≤15.8K (4-bit KV); about 2× those with g128. At equal settings, savings vs MLX come from static memory planning and allocator slack (~1–3 GB, LIKELY). |
| **TTFT** | 900 → 420 ms | 900 ms at 800 tok/s ⇒ prompt ≈ 690 tokens (+~37 ms first decode). 420 ms for the same prompt needs **≈1,800 tok/s** | **Impossible cold in FP16** (floor ≈ 0.66 s at 100% NA; ~0.82 s at 80%). INT8 would need ~87% of peak. | Cold, FP16: **~0.7–0.85 s** for ~700 tokens. Warm with prefix cache (chat or agent turn with cached history): **≪420 ms, bit-exact**. State "cold vs cached" and the prompt length. |

---

## 11. Open questions and the experiments that settle them

1. **Measure the real MLX baseline** on M5 Max: Qwen3-32B 4-bit g64 decode at 0/4K/8K context, and prefill at P = 512/2K/8K.
   - Expected: decode 24–29 tok/s.
   - Prefill is unknown. If MLX is already near 800, FP16 headroom is ≤1.35×.
2. **Small-M kernel sweep** on M5 Max: time(M)/time(1) for M = 1..32 on Qwen3-32B shapes, SIMD vs NA (Metal 4 tensor ops). This sets β and k\*, and therefore the whole speculative plan.
3. **Acceptance study.** RedHat EAGLE-3 vs Qwen3-0.6B/1.7B drafts on Sky's target prompts at T = 0.6/0.7, thinking on/off. Compare exact-match acceptance (mlx-lm/llama.cpp style) with rejection sampling and keyed Gumbel.
4. **Batch-invariance tax** on Apple: M=1 GEMV vs padded-tile kernel with an M-invariant reduction order. Target: <3% decode cost.
5. **Quality baselines:** KL and top-token agreement for MLX 4-bit g64, Q4_K_M, mlx-lm dynamic quant and DWQ on Qwen3-32B vs BF16.
6. **INT8 NA sustained throughput** on prefill GEMM shapes, including MacBook thermals over 10+ s.
7. **Hybrid models:** MTP acceptance and DeltaNet state checkpoint/rollback cost for Qwen3.6/3.8-27B.

---

## 12. Key sources (all fetched 2026-10-01/02)

**Configs, tensors and weights:**
- `huggingface.co/{Qwen/Qwen3-32B, Qwen3-8B, Qwen3-0.6B, Qwen3-30B-A3B, openai/gpt-oss-20b, gpt-oss-120b, unsloth/Meta-Llama-3.1-8B/70B-Instruct, Qwen/Qwen3.6-27B, Qwen3.8-27B}/resolve/main/config.json`
- HF API `/api/models/<repo>` (param totals), `/api/models/<repo>/tree/main` (file sizes)
- Safetensors headers via range requests

**Speculators:**
- `huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3`
- `huggingface.co/AngelSlim/Qwen3-32B_eagle3`
- `huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash`
- `huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark`

**Speculative decoding papers:**
- arXiv 2211.17192 (Leviathan), 2302.01318 (Chen)
- 2401.15077 / 2406.16858 / 2503.01840 (EAGLE 1/2/3)
- 2401.10774 (Medusa), 2402.05109 (Hydra), 2402.02057 (Lookahead), 2411.04975 (SuffixDecoding)
- 2305.09781 (SpecInfer), 2402.12374 (Sequoia)
- 2403.09919 (ReDrafter), 2507.11851 (Apple MTP), 2510.13161 (Mirror-SD), 2603.03251 (SSD), 2602.06036 (DFlash)
- 2412.19437 (DeepSeek-V3), 2505.07608 (MiMo), 2508.06471 (GLM-4.5)

**Determinism:**
- thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- lmsys.org/blog/2025-09-22-sglang-deterministic/
- arXiv 2408.07978 (drafter-invariant SD), 2601.17768 (LLM-42), 2609.26621 (precision divergence)
- github.com/ml-explore/mlx-lm/issues/1470 and PRs #1468, #1469, #1592

**Quantization:**
- arXiv 2505.02214 (Qwen3 quantization study), 2406.11235 (QTIP), 2402.04396 (QuIP#), 2401.06118 (AQLM)
- 2405.16406 (SpinQuant), 2504.12285 (BitNet 2B4T), 2502.02631 (ParetoQ), 2410.14649 (EvoPress)
- 2507.13575 (Apple FM 2025), 2504.11651 (DFloat11)
- gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
- llama.cpp `ggml/src/ggml-common.h`, `src/llama-quant.cpp`, `tools/quantize/README.md`, `common/sampling.cpp`, `common/common.h` (master, 2026-10-01)
- mlx `mlx/ops.cpp`; mlx-lm `mlx_lm/{generate.py, sample_utils.py, models/switch_layers.py, models/qwen3_5.py, LEARNED_QUANTS.md}` (main, 2026-10-01)

**KV cache, sparsity, MoE, TTFT:**
- arXiv 2402.02750 (KIVI), 2401.18079 (KVQuant), 2309.17453 (StreamingLLM), 2306.14048 (H2O), 2404.14469 (SnapKV), 2412.10319 (SCBench), 2410.08391 (KV Prediction)
- 2408.14690 (TEAL), 2404.08763 (CATS), 2310.17157 (Deja Vu), 2301.00774 (SparseGPT), 2306.11695 (Wanda), 2307.02973 (pruning vs quantization)
- 2505.19645 (MoESD), 2607.12696 (EcoSpec), 2609.22156 (Limits of Speculation), 2609.29032 (Paging the Experts)
- 2312.11514 (LLM in a flash), 2403.02310 (Sarathi-Serve), 2407.02490 (MInference)
- github.com/ml-explore/mlx-lm/issues/1651

**Apple hardware and competitors:**
- apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-…
- machinelearning.apple.com/research/exploring-llms-mlx-m5
- tzakharko.github.io/apple-neural-accelerators-benchmark/
- arXiv 2607.19438 (BaseRT)
- github.com/youssofal/MTPLX
- rotecodefraktion.de MTPLX M3 Max test
- lmstudio.ai/blog/lmstudio-v0.3.10
- github.com/lmstudio-ai/mlx-engine/issues/103
- github.com/ml-explore/mlx-lm/discussions/890

**Other agents' local evidence:**
- R3: `$S/research/r3_out_qmv_multi_1.txt` (MLX small-M qmv on M2 Pro)
- R5: `$S/research/R5-llamacpp.md` (llama.cpp mul_mv_ext and speculative types)


---

## Appendix A — generated model arithmetic (`R8-work/model_math.py`)

Bandwidth used for rooflines: 614 GB/s (decimal GB). Rooflines assume 100% of peak; real kernels reach ~70-90%.

### 1. Architecture (from config.json)

| model | hidden | layers | q heads | kv heads | head_dim | FFN (I or moe I) | vocab | tied | experts (top-k) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 5120 | 64 | 64 | 8 | 128 | 25600 | 151936 | False | - |
| Qwen3-8B | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | False | - |
| Qwen3-4B | 2560 | 36 | 32 | 8 | 128 | 9728 | 151936 | True | - |
| Qwen3-1.7B | 2048 | 28 | 16 | 8 | 128 | 6144 | 151936 | True | - |
| Qwen3-0.6B | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | True | - |
| Qwen3-30B-A3B | 2048 | 48 | 32 | 4 | 128 | 768 | 151936 | False | 128 (top-8) |
| Llama-3.1-8B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | False | - |
| Llama-3.1-70B | 8192 | 80 | 64 | 8 | 128 | 28672 | 128256 | False | - |
| gpt-oss-20b | 2880 | 24 | 64 | 8 | 64 | 2880 | 201088 | False | 32 (top-4) |
| gpt-oss-120b | 2880 | 36 | 64 | 8 | 64 | 2880 | 201088 | False | 128 (top-4) |

### 2. Parameter counts (computed) vs HF safetensors totals

| model | embed | lm_head | attn/layer | FFN/layer (all experts) | total computed | HF total | match | active linear params / token |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 777,912,320 | 777,912,320 | 94,372,096 | 393,216,000 | 32,762,123,264 | 32,762,123,264 | exact | 31.984 B |
| Qwen3-8B | 622,329,856 | 622,329,856 | 41,943,296 | 150,994,944 | 8,190,735,360 | 8,190,735,360 | exact | 7.568 B |
| Qwen3-4B | 388,956,160 | 0 | 26,214,656 | 74,711,040 | 4,022,468,096 | 4,022,468,096 | exact | 4.022 B |
| Qwen3-1.7B | 311,164,928 | 0 | 12,583,168 | 37,748,736 | 1,720,574,976 | 2,031,739,904 | diff -311,164,928 | 1.720 B |
| Qwen3-0.6B | 155,582,464 | 0 | 6,291,712 | 9,437,184 | 596,049,920 | 751,632,384 | diff -155,582,464 | 0.596 B |
| Qwen3-30B-A3B | 311,164,928 | 311,164,928 | 18,874,624 | 603,979,776 | 30,532,122,624 | 30,532,122,624 | exact | 3.042 B |
| Llama-3.1-8B | 525,336,576 | 525,336,576 | 41,943,040 | 176,160,768 | 8,030,261,248 | 8,030,261,248 | exact | 7.505 B |
| Llama-3.1-70B | 1,050,673,152 | 1,050,673,152 | 150,994,944 | 704,643,072 | 70,553,706,496 | 70,553,706,496 | exact | 69.502 B |
| gpt-oss-20b | 579,133,440 | 579,133,440 | 26,550,144 | 796,262,400 | 20,914,757,184 | 20,914,757,184 | exact | 3.607 B |
| gpt-oss-120b | 579,133,440 | 579,133,440 | 26,550,144 | 3,185,049,600 | 116,829,156,672 | 116,829,156,672 | exact | 5.131 B |

### 3. Weight bytes streamed per decoded token (GB) and roofline tok/s at 614 GB/s (context ~0)

| model | BF16 | MLX 8b g64 (8.5) | MLX 6b g64 (6.5) | MLX 4b g64 (4.5) | MLX 4b g128 (4.25) | MLX 3b g64 (3.5) | GGUF Q4_K_M | GGUF Q4_0(+imatrix) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 63.97 GB / 10 | 33.98 GB / 18 | 25.99 GB / 24 | 17.99 GB / 34 | 16.99 GB / 36 | 13.99 GB / 44 | 19.32 GB / 32 | 18.26 GB / 34 |
| Qwen3-8B | 15.14 GB / 41 | 8.04 GB / 76 | 6.15 GB / 100 | 4.26 GB / 144 | 4.02 GB / 153 | 3.31 GB / 185 | 4.67 GB / 131 | 4.43 GB / 139 |
| Qwen3-4B | 8.04 GB / 76 | 4.27 GB / 144 | 3.27 GB / 188 | 2.26 GB / 271 | 2.14 GB / 287 | 1.76 GB / 349 | 2.49 GB / 247 | 2.37 GB / 259 |
| Qwen3-1.7B | 3.44 GB / 178 | 1.83 GB / 336 | 1.40 GB / 439 | 0.97 GB / 634 | 0.91 GB / 672 | 0.75 GB / 816 | 1.10 GB / 558 | 1.05 GB / 585 |
| Qwen3-0.6B | 1.19 GB / 515 | 0.63 GB / 970 | 0.48 GB / 1268 | 0.34 GB / 1832 | 0.32 GB / 1939 | 0.26 GB / 2355 | 0.39 GB / 1572 | 0.38 GB / 1633 |
| Qwen3-30B-A3B | 6.08 GB / 101 | 3.23 GB / 190 | 2.47 GB / 248 | 1.71 GB / 359 | 1.62 GB / 380 | 1.33 GB / 461 | 1.88 GB / 327 | 1.80 GB / 342 |
| Llama-3.1-8B | 15.01 GB / 41 | 7.97 GB / 77 | 6.10 GB / 101 | 4.22 GB / 145 | 3.99 GB / 154 | 3.28 GB / 187 | 4.62 GB / 133 | 4.37 GB / 140 |
| Llama-3.1-70B | 139.00 GB / 4 | 73.85 GB / 8 | 56.47 GB / 11 | 39.09 GB / 16 | 36.92 GB / 17 | 30.41 GB / 20 | 41.87 GB / 15 | 39.51 GB / 16 |

gpt-oss (experts MXFP4 4.25 bpw on disk):

| model | native (MXFP4 experts + BF16 rest) | MXFP4 experts + 8-bit rest | all 4-bit g64 | BF16 |
|---|---|---|---|---|
| gpt-oss-20b | 3.71 GB / 166 | 2.56 GB / 240 | 2.03 GB / 303 | 7.21 GB / 85 |
|  (expert weights total 19.11 B params; active 2.389 B; non-expert linear 0.639 B; lm_head 0.579 B) |||||
| gpt-oss-120b | 5.00 GB / 123 | 3.55 GB / 173 | 2.89 GB / 213 | 10.26 GB / 60 |
|  (expert weights total 114.66 B params; active 3.583 B; non-expert linear 0.969 B; lm_head 0.579 B) |||||

### 4. KV cache bytes per token of context

fp16 = 2 B/elem; int8 (MLX-style affine g64, fp16 scale+bias) = 8.5 bits/elem; 4-bit g64 = 4.5 bits/elem.

| model | KV layers (full) | elems/token | fp16 B/token | int8(g64) B/token | 4-bit(g64) B/token | fp16 GB @8K | fp16 GB @32K | fp16 GB @128K |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 64 | 131,072 | 262,144 | 139,264 | 73,728 | 2.15 | 8.59 | 34.36 |
| Qwen3-8B | 36 | 73,728 | 147,456 | 78,336 | 41,472 | 1.21 | 4.83 | 19.33 |
| Qwen3-4B | 36 | 73,728 | 147,456 | 78,336 | 41,472 | 1.21 | 4.83 | 19.33 |
| Qwen3-1.7B | 28 | 57,344 | 114,688 | 60,928 | 32,256 | 0.94 | 3.76 | 15.03 |
| Qwen3-0.6B | 28 | 57,344 | 114,688 | 60,928 | 32,256 | 0.94 | 3.76 | 15.03 |
| Qwen3-30B-A3B | 48 | 49,152 | 98,304 | 52,224 | 27,648 | 0.81 | 3.22 | 12.88 |
| Llama-3.1-8B | 32 | 65,536 | 131,072 | 69,632 | 36,864 | 1.07 | 4.29 | 17.18 |
| Llama-3.1-70B | 80 | 163,840 | 327,680 | 174,080 | 92,160 | 2.68 | 10.74 | 42.95 |
| gpt-oss-20b | 12 (+12 SWA-128) | 12,288 | 24,576 | 13,056 | 6,912 | 0.20 | 0.81 | 3.22 |
| gpt-oss-120b | 18 (+18 SWA-128) | 18,432 | 36,864 | 19,584 | 10,368 | 0.31 | 1.21 | 4.84 |

### 5. FLOPs

Decode FLOPs/token = 2*(active matmul params incl. lm_head) + attention 4*n_q_heads*head_dim*T per full-attn layer (QK^T and PV), T = context.

Prefill FLOPs for P tokens = 2*P*(active matmul params excl. lm_head) + 2*V*H (last-token logits) + causal attention 2*n_q_heads*head_dim*P^2 per layer.

| model | matmul GFLOP/token (decode) | attn MFLOP per token per ctx-token | attn GFLOP/token @4K | @32K | prefill TFLOP @P=1K | @P=4K | @P=32K | attn share @P=32K |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | 64.0 | 2.10 | 8.6 | 68.7 | 65.0 | 273.2 | 3171 | 36% |
| Qwen3-8B | 15.1 | 0.59 | 2.4 | 19.3 | 14.5 | 61.8 | 772 | 41% |
| Qwen3-4B | 8.0 | 0.59 | 2.4 | 19.3 | 7.8 | 34.7 | 555 | 57% |
| Qwen3-1.7B | 3.4 | 0.23 | 0.9 | 7.5 | 3.0 | 13.5 | 216 | 57% |
| Qwen3-0.6B | 1.2 | 0.23 | 0.9 | 7.5 | 1.0 | 5.5 | 152 | 81% |
| Qwen3-30B-A3B | 6.1 | 0.79 | 3.2 | 25.8 | 6.0 | 29.0 | 601 | 70% |
| Llama-3.1-8B | 15.0 | 0.52 | 2.1 | 17.2 | 14.6 | 61.6 | 739 | 38% |
| Llama-3.1-70B | 139.0 | 2.62 | 10.7 | 85.9 | 141.6 | 582.7 | 5893 | 24% |
| gpt-oss-20b | 7.2 | 0.20 | 0.8 | 6.4 | 6.3 | 26.6 | 305 | 35% |
| gpt-oss-120b | 10.3 | 0.29 | 1.2 | 9.7 | 9.5 | 39.9 | 458 | 35% |

### 6. Qwen3-32B decode roofline vs context (MLX 4-bit g64 weights)

| context T | weights GB | KV fp16 GB | total GB | roofline tok/s fp16-KV | KV 8-bit total GB | tok/s | KV 4-bit total GB | tok/s |
|---|---|---|---|---|---|---|---|---|
| 0 | 17.99 | 0.00 | 17.99 | 34.1 | 17.99 | 34.1 | 17.99 | 34.1 |
| 1024 | 17.99 | 0.27 | 18.26 | 33.6 | 18.13 | 33.9 | 18.07 | 34.0 |
| 4096 | 17.99 | 1.07 | 19.06 | 32.2 | 18.56 | 33.1 | 18.29 | 33.6 |
| 8192 | 17.99 | 2.15 | 20.14 | 30.5 | 19.13 | 32.1 | 18.59 | 33.0 |
| 16384 | 17.99 | 4.29 | 22.29 | 27.6 | 20.27 | 30.3 | 19.20 | 32.0 |
| 32768 | 17.99 | 8.59 | 26.58 | 23.1 | 22.55 | 27.2 | 20.41 | 30.1 |
| 40960 | 17.99 | 10.74 | 28.73 | 21.4 | 23.69 | 25.9 | 21.01 | 29.2 |

### 7. Required bits/weight for a given decode rate on Qwen3-32B (no speculation, T~0)

- 35 tok/s at 100% of 614 GB/s needs <= 4.39 bits/weight averaged over 31.98 B streamed params; at 85% efficiency: 3.73 bpw
- 40 tok/s at 100% of 614 GB/s needs <= 3.84 bits/weight averaged over 31.98 B streamed params; at 85% efficiency: 3.26 bpw
- 50 tok/s at 100% of 614 GB/s needs <= 3.07 bits/weight averaged over 31.98 B streamed params; at 85% efficiency: 2.61 bpw
- 60 tok/s at 100% of 614 GB/s needs <= 2.56 bits/weight averaged over 31.98 B streamed params; at 85% efficiency: 2.18 bpw


## Appendix B — generated speculative/MoE model (`R8-work/spec_model.py`)

Plain-decode step for Qwen3-32B 4-bit g64 at 80% of 614 GB/s: T1 = 36.6 ms -> 27.3 tok/s.

### A. Per-token acceptance alpha implied by published mean acceptance lengths (i.i.d. chain model)

| source (target, temp) | k drafted | mean accepted length tau | implied alpha |
|---|---|---|---|
| RedHatAI Qwen3-32B EAGLE-3, HumanEval, T=0.6 | 3 | 2.29 | 0.63 |
| same, HumanEval | 7 | 2.53 | 0.61 |
| same, GSM8K | 3 | 2.49 | 0.69 |
| same, GSM8K | 7 | 3.08 | 0.69 |
| same, CNN/DM | 3 | 2.15 | 0.59 |
| same, CNN/DM | 7 | 2.33 | 0.58 |
| AngelSlim Qwen3-32B EAGLE-3, MT-bench (H20, bs=1) | 2 | 1.86 | 0.55 |
| AngelSlim Qwen3-32B EAGLE-3, GSM8K | 2 | 2.01 | 0.62 |
| RedHatAI Qwen3-8B DFlash, HumanEval, T=0, think off | 15 | 5.88 | 0.84 |
| RedHatAI Qwen3.8-27B DSpark, HumanEval | 8 | 4.42 | 0.81 |
| RedHatAI Qwen3.8-27B DSpark, math | 8 | 5.74 | 0.88 |
| MTPLX Qwen3.6-27B MTP (per-position 73/48/32%) | 3 | 2.53 | 0.70 |

Note: per-position acceptance is not i.i.d. in practice (it decays with depth), so 'implied alpha' is only a summary.

### B. Speedup S = tau / (1 + gamma*c + beta*gamma) for a chain draft

c = draft cost per drafted token / T1; beta = marginal cost of verifying one more token / T1.
beta ~0.02-0.05: kernel that streams weights once for M<=k* (target design); beta ~0.48: MLX 0.32 qmv at M=4 on M2 Pro (R3 measurement: 2.43x time for 4 rows).

| tau source | gamma | c | beta=0.02 | beta=0.05 | beta=0.15 | beta=0.48 (MLX-like) |
|---|---|---|---|---|---|---|
| Qwen3-32B EAGLE-3 HumanEval T=0.6 | 3 | 0.03 | 1.99x (54 t/s) | 1.85x (50 t/s) | 1.49x (41 t/s) | 0.91x (25 t/s) |
| Qwen3-32B EAGLE-3 HumanEval T=0.6 | 3 | 0.1 | 1.68x (46 t/s) | 1.58x (43 t/s) | 1.31x (36 t/s) | 0.84x (23 t/s) |
| Qwen3-32B EAGLE-3 GSM8K T=0.6 | 3 | 0.03 | 2.17x (59 t/s) | 2.01x (55 t/s) | 1.62x (44 t/s) | 0.98x (27 t/s) |
| Qwen3-32B EAGLE-3 GSM8K T=0.6 | 3 | 0.1 | 1.83x (50 t/s) | 1.72x (47 t/s) | 1.42x (39 t/s) | 0.91x (25 t/s) |
| Qwen3-32B EAGLE-3 GSM8K T=0.6 | 7 | 0.03 | 2.28x (62 t/s) | 1.97x (54 t/s) | 1.36x (37 t/s) | 0.67x (18 t/s) |
| Qwen3-32B EAGLE-3 GSM8K T=0.6 | 7 | 0.1 | 1.67x (46 t/s) | 1.50x (41 t/s) | 1.12x (31 t/s) | 0.61x (17 t/s) |
| Qwen3-32B EAGLE-3 CNN/DM T=0.6 | 3 | 0.03 | 1.87x (51 t/s) | 1.73x (47 t/s) | 1.40x (38 t/s) | 0.85x (23 t/s) |
| Qwen3-32B EAGLE-3 CNN/DM T=0.6 | 3 | 0.1 | 1.58x (43 t/s) | 1.48x (40 t/s) | 1.23x (34 t/s) | 0.78x (21 t/s) |
| hypothetical tree, tau=3.5 (greedy code) | 15 | 0.03 | 2.00x (55 t/s) | 1.59x (43 t/s) | 0.95x (26 t/s) | 0.40x (11 t/s) |
| hypothetical tree, tau=3.5 (greedy code) | 15 | 0.1 | 1.25x (34 t/s) | 1.08x (29 t/s) | 0.74x (20 t/s) | 0.36x (10 t/s) |
| hypothetical tau=4.4 (DSpark-class) | 8 | 0.03 | 3.16x (86 t/s) | 2.70x (74 t/s) | 1.81x (49 t/s) | 0.87x (24 t/s) |
| hypothetical tau=4.4 (DSpark-class) | 8 | 0.1 | 2.26x (62 t/s) | 2.01x (55 t/s) | 1.47x (40 t/s) | 0.78x (21 t/s) |

### C. 'Free' verification width k* on a bandwidth-bound GPU

Weight-streaming matmul over k tokens: FLOPs = 2*N*k, bytes = N*b/8 -> intensity = 16k/b FLOP/byte.
It stays bandwidth-bound while 16k/b < R = F/BW (ridge point), i.e. k* = R*b/16.

| device / path | F (TFLOP/s) | BW (GB/s) | R=F/BW | k* at b=4.5 | k* at b=16 |
|---|---|---|---|---|---|
| M5 Max NA fp16 (est. 70 TFLOPS, tzakharko) | 70 | 614 | 114 | 32.1 | 114 |
| M5 Max NA int8 (est. 130 TOPS) | 130 | 614 | 212 | 59.5 | 212 |
| M5 Max SIMD fp32 (40c*128*2*1.6GHz) | 16.4 | 614 | 27 | 7.5 | 27 |
| M2 Pro SIMD fp32 (19c, ~6.8 TFLOPS) | 6.8 | 200 | 34 | 9.6 | 34 |
| A100 fp16 tensor (312 TFLOPS, 2.0 TB/s) | 312 | 2039 | 153 | 43.0 | 153 |

(Sequoia's empirically optimal tree size on A100 with fp16 weights was 128; the formula gives ~153.)

### D. MoE: distinct experts touched when verifying k tokens (independent routing = upper bound)

| model | experts E | top-K | k=1 | k=2 | k=4 | k=8 | bytes(k=4)/bytes(k=1) | bytes(k=8)/bytes(k=1) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 128 | 8 | 8.0 | 15.5 | 29.1 | 51.6 | 2.57 | 4.25 |
| gpt-oss-120b | 128 | 4 | 4.0 | 7.9 | 15.3 | 28.7 | 2.97 | 5.31 |
| gpt-oss-20b | 32 | 4 | 4.0 | 7.5 | 13.2 | 21.0 | 2.53 | 3.82 |
| Qwen3.6-35B-A3B (+1 shared) | 256 | 8 | 8.0 | 15.8 | 30.5 | 57.4 | n/a | n/a |

Dense model: bytes(k)/bytes(1) = 1.00 for all k <= k* (only KV/activations grow).


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
