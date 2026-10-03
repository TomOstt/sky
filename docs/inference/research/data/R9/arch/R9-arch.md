# R9-ARCH: what each config.json means for a model compiler

This covers task item 3: how the fields in a Hugging Face `config.json` map to the architecture features a model compiler must support. It has a table, per-family gotchas, a feature checklist and the Qwen3-32B arithmetic.

## Sources

- **Configs.** 31 `config.json` files, fetched on 2026-10-01 into `$S/research/R9-work/arch/configs/`.
  - The meta-llama and google repos are gated (HTTP 401). For those I used the `unsloth/*` mirrors, which carry `"unsloth_fixed": true`. The `mlx-community/*` mirrors are sparser.
- **Code.** All at their clone HEADs:
  - mlx-lm `53b9af3`;
  - llama.cpp `81e39ad`;
  - MLX `0840c42`;
  - transformers 5.18.0 (from the venv).
- **Scripts.** `resave.py` re-serialises configs with transformers 5. `qwen3_32b_math.py` does the arithmetic.
- **Remote headers.** I read safetensors headers over HTTP Range requests, without downloading the weights.

## 1. Compact table

Unless marked otherwise, every value comes from the fetched configs and is VERIFIED; semantics are checked against the code cited in section 2.

Abbreviations:
- **pre** — pre-norm RMSNorm: `input_layernorm` sits before attention, and `post_attention_layernorm` sits before the MLP.
- **NeoX** — rotate-half RoPE layout, as in the HF weights.
- **dc** — `head_dim` is decoupled, meaning heads × head_dim ≠ hidden.
- **GQA** — query heads divided by KV heads.

| Family | Norm | QK-norm | Biases | MLP act | RoPE | Sliding window | Softcap / sinks | Tied emb | Emb scale | Heads / head_dim | MoE | Release dtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 dense (0.6B–32B) | pre, eps 1e-6 | yes: per-head RMSNorm (dim 128), applied **before** RoPE | none | SwiGLU | θ 1e6, no scaling (YaRN ×4 is opt-in for 128k), NeoX | none (keys are inert) | none | ≤4B yes; 8B/32B no | – | dc: 32B is 64×128 = 8192 vs hidden 5120; GQA 8 (32B), 4 (8B) | – | BF16 |
| Qwen3-MoE (30B-A3B, 235B-A22B) | same | same | none | SwiGLU experts | same | none | none | no | – | dc (32×128 vs 2048); GQA 8 / 16 | 128 experts, top-8, no shared expert; softmax(fp32) → top-k → renorm (`norm_topk_prob`); all layers sparse | BF16 |
| Qwen2.5 / R1-Distill-Qwen | pre, eps 1e-6 (R1-32B: 1e-5) | no | **q, k, v bias**; no o bias | SwiGLU | θ 1e6 (R1-Distill-Qwen-7B: **1e4**) | none (inert) | none | ≤3B yes | – | coupled; GQA **7** (7B), **5** (32B) | – | BF16 |
| Llama 3.1/3.3 / R1-Distill-Llama | pre, eps 1e-5 | no | none | SwiGLU | θ 5e5, `llama3` scaling (×8, low 1, high 4, orig 8192) | none | none | no | – | coupled; GQA 4 (8B), 8 (70B) | – | BF16 |
| Llama 3.2 1B/3B | same | no | none | SwiGLU | `llama3` ×32 | none | none | **yes** | – | 1B: head_dim 64; 3B: GQA 3 | – | BF16 |
| Gemma 3 (1B/4B/27B) | **sandwich** (pre+post attention, pre+post FFN), **(1+w)**, eps 1e-6 | yes, with (1+w) | none | **GeGLU** (tanh GELU) | local layers θ 1e4, no scaling; global layers θ 1e6 + **linear ×8** (4B and up); NeoX | **5 local : 1 global**; window 512 (1B) / 1024 | none (both softcap keys null) | yes (no `lm_head` tensor) | **×√hidden, rounded to bf16** | dc (27B 32×128 vs 5376; 4B 8×256 vs 2560); scale = `query_pre_attn_scalar`^-½ (27B: 168); GQA 2; 1B is MQA | – | BF16 |
| Mistral 7B v0.3 | pre, 1e-5 | no | none | SwiGLU | θ 1e6 | none (`sliding_window` null) | none | no | – | coupled; GQA 4 | – | BF16 |
| Ministral 8B 2410 | pre, 1e-5 | no | none | SwiGLU | θ **1e8** | `layer_types` [full, swa, swa, swa]×9, window 32768 (= max_pos, so inert up to 32k) | none | no | – | coupled; GQA 4 | – | BF16 |
| Mistral Small 3.x 24B | pre, 1e-5 | no | none | SwiGLU | θ 1e8 (2501) / **1e9** (3.1) | none | none | no (key absent in 3.1's `text_config`) | – | dc 32×128 = 4096 vs 5120; GQA 4 | – | BF16 |
| Phi-4 14B | pre, 1e-5 | no | none | SwiGLU; fused `gate_up_proj` (halves) | θ 2.5e5, no scaling; fused `qkv_proj` | none | none | no | – | coupled; GQA 4 (40/10) | – | BF16 |
| Phi-4-mini 3.8B | pre, 1e-5 | no | none | same as Phi-4 | θ 1e4; **LongRoPE** (48 short/long factors); **partial rotary 0.75** (96 of 128 dims); attention factor 1.190 | inert (262144) | none | yes | – | GQA 3 | – | BF16 |
| DeepSeek-V3/R1 671B | pre, plus RMSNorm on the q/kv latents | no | none | SwiGLU | MLA with decoupled RoPE: 64 of 192 qk dims, **interleaved**, θ 1e4, YaRN ×40 (orig 4096); softmax scale ×1.874 | none | none | no | – | 128 heads; q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v 128 | 256 routed + 1 shared expert, top-8; **sigmoid** + `e_score_correction_bias` (used for selection only); 8 groups → top-4 groups (top-2 sum); renorm; ×2.5; first 3 layers dense; plus 1 MTP layer | **FP8 e4m3** with 128×128 block scales |
| gpt-oss 20b/120b | pre, 1e-5 | no | **q, k, v, o bias**; router bias; expert biases | **clamped SwiGLU-OAI**: (clip(up, ±7) + 1)·g·σ(1.702g), with g = min(gate, 7) | θ 1.5e5, YaRN ×32 (orig 4096, β 32/1, **truncate=false**); attention factor 1.347 | alternating swa/full, swa first; window **128** | **sinks**: a learned logit per head | no | – | dc 64×64 = 4096 vs 2880; GQA 8 | 32 / 128 experts, top-4; top-k then softmax over the k; no shared expert; `gate_up` is **interleaved** | **MXFP4 experts** (U8 blocks + E8M0 scales, group 32); everything else BF16 |

## 2. Gotchas, with keys and citations

Unless a line is marked otherwise, everything below is VERIFIED from the cited file or line.

Path shorthands used below:
- `mlxlm/` = `$S/src/mlx-lm/mlx_lm/models/`
- `lcpp/` = `$S/src/llama.cpp/`
- `tf/` = the venv's `site-packages/transformers/`

### Qwen3 (dense and MoE)

- **QK-norm before RoPE.** `q_norm` and `k_norm` are RMSNorms over `head_dim`, applied before RoPE. See `mlxlm/qwen3.py:60-61, 80-94`. llama.cpp uses the same order (`lcpp/src/models/qwen3.cpp:88-100`).
- **Biases.** mlx-lm hard-codes `bias=False` (`qwen3.py:55-58`). The `attention_bias` key is not read.
- **Decoupled head_dim.** The real header of mlx-community/Qwen3-32B-4bit shows `q_proj.weight` as U32 [8192, 640], so the projection is 5120 → 8192.
- **Inert sliding-window keys.** `sliding_window` and `max_window_layers` do nothing because `use_sliding_window: false`. transformers sets the window to null unless `use_sliding_window` is true (`tf/models/qwen2/configuration_qwen2.py:84-91`; qwen3 has identical lines). A compiler that reads `sliding_window` literally is wrong. R1-Distill-Qwen-7B is a trap here: it ships `"sliding_window": 4096`, which is also inert.
- **Context length and YaRN.** `max_position_embeddings` 40960 is 32768 for output plus 8192 for the prompt. YaRN is static, and the card says not to enable it below 32k because it can hurt short texts (Qwen3-32B README L267-315). Benchmark rule: never enable YaRN in only one runtime.
- **Stop tokens.** `generation_config.json` has `eos [151645, 151643]`, while `config.json` has the single id 151645. mlx-lm overrides the config value with the generation_config list (`mlx_lm/utils.py:375-376`).
- **MoE routing is the same math as gpt-oss.** Qwen3-MoE does a full softmax, then top-k, then renormalises (`qwen3_moe.py:134-142`). Mathematically that equals a softmax over the top-k logits, which is what gpt-oss does (`gpt_oss.py:129-131`). The numerics differ, though.
  - LIKELY: ties among bf16 router logits can pick different experts across runtimes (`argpartition` vs `topk`), but this should be rare.
- **Unstacked experts.** HF stores each expert separately: the 30B-A3B index has 18,867 tensors. The loader must stack them into [E, out, in] for gather-matmul. mlx-lm sorts tokens by expert once there are at least 64 indices (`switch_layers.py:181`).

### Qwen2.5 and R1-Distill-Qwen

- **Hard-coded biases.** QKV bias is on and o_proj bias is off, both hard-coded (`qwen2.py:45-48`). The config has no `attention_bias` key.
- **R1-Distill differences:**
  - R1-Distill-Qwen-7B uses θ = 1e4 (it was distilled onto a Qwen2.5-Math base).
  - R1-Distill-Qwen-32B uses eps 1e-5, where Qwen2.5-32B uses 1e-6.
  - In R1-Distill-Qwen-32B, `generation_config` has `bos 151646` while `config` has 151643.
- **Different tie default.** mlx-lm's `qwen2` defaults to `tie_word_embeddings=True` (`qwen2.py:30`), and so does `llama.py:33`. HF's `LlamaConfig` defaults to False. This only matters when the key is absent; that risk is LIKELY.

### Llama 3.x

- **llama3 RoPE.** The frequency remap uses factor, low/high frequency factors and the original length (`rope_utils.py:86-106`). llama.cpp instead bakes a `rope_freqs` tensor into the GGUF (`lcpp/conversion/llama.py:296-326`).
- **RoPE layout depends on the file format.** llama.cpp's converter permutes q/k for `LlamaForCausalLM` and `MistralForCausalLM` (`conversion/llama.py:17-20, 176-181, 260-264`). llama.cpp therefore runs NORM (interleaved) RoPE for LLAMA, DEEPSEEK2 and MISTRAL3 (`src/llama-model.cpp:3061-3104`), and NEOX for QWEN2/3, QWEN3MOE, PHI3, GEMMA3 and OPENAI_MOE (`:3126-3186`).
  - **The RoPE layout is a property of the file, not of the model.** A Sky loader that reads both GGUF and safetensors must track it per file.
- **o_proj bias.** In `llama.py`, `attention_bias` also applies to `o_proj` (`llama.py:61-64`).

### Gemma 3

- **(1+w) norms, and the file format changes them.** The norms compute `rms_norm(x, 1+w)` (`gemma3_text.py:104-111`). The GGUF converter has already added 1 (`conversion/gemma.py:131-132`).
- **Same tensor name, different graph position.**
  - In Gemma, `post_attention_layernorm` normalises the attention output before the residual add (a sandwich layout; `gemma3_text.py:142-160`).
  - In Llama and Qwen, the tensor with the same name is the pre-MLP norm.
- **Attention scale.** The scale is `query_pre_attn_scalar`^-0.5 (`gemma3_text.py:46`), and 27B uses 168, not head_dim 128.
  - llama.cpp ignores the key and hard-codes the scale by layer count: 27B gets n_embd/n_head, all others get head_dim (`src/models/gemma3.cpp:28-31`).
  - Released checkpoints are fine. A new size or fine-tune with a different scalar would break silently.
- **Embedding scale rounds to bf16.**
  - HF and mlx-lm cast √hidden to bf16 (`tf/models/gemma3/modeling_gemma3.py:114-117`; `gemma3_text.py:190`).
  - Measured: 27B goes 73.32 → 73.5 (+0.24%); 4B goes 50.60 → 50.5 (−0.19%).
  - llama.cpp uses fp32 `sqrtf(n_embd)` (`gemma3.cpp:91`). That is a systematic ~0.2% difference in the residual stream compared with the reference.
- **Layer pattern.** Layer *i* is global when (i+1) % 6 == 0 (`gemma3_text.py:55`).
  - Local layers use θ 1e4 with no scaling.
  - Global layers use θ 1e6 with linear ×8 (`gemma3_text.py:57-70`).
  - transformers 5 now stores this as nested `rope_parameters {sliding_attention, full_attention}` (`configuration_gemma3.py:108-115, 136-163`).
- **fp16 clipping.** In fp16, mlx-lm clips the residual (`gemma3_text.py:125-132`), so run in bf16.
- **Checkpoint quirks.**
  - Weights live under the `language_model.model.*` prefix, and there is no `lm_head`.
  - Vocab is 262208 for 4B/27B and 262144 for 1B.
  - mlx-community/gemma-3-27b-it-4bit's `text_config` has only 10 keys; eps, θ, local θ, pattern and vocab are missing and come from code defaults (`gemma3.py:18-27`, `gemma3_text.py:15-32`).
- **No softcapping in Gemma 3.** Both soft-caps are null. Gemma 2 used 50 (attention) and 30 (final) (`gemma2.py:25-26`). For Gemma 4, `gemma4_text.py:41` defaults the final cap to 30; LIKELY, configs not fetched.

### Mistral and Ministral

- **Mistral runs through `llama.py`.** mlx-lm remaps model type `mistral` to `llama` (`mlx_lm/utils.py:48`). `llama.py` reads `layer_types` and gives sliding layers a `RotatingKVCache` (`llama.py:162, 187-194, 266-270`).
- **Window semantics agree.** Key *j* is visible to query *i* when i − j < W, in both mlx-lm (`base.py:35-37`) and llama.cpp (`src/llama-hparams.h:486-491`).
- **Mistral Small 3.1.** Its `text_config` lacks the tie key, and mlx-lm forces `tie=False` (`mistral3.py:19-20`).
- **High θ.** θ of 1e8–1e9 means RoPE angles should be computed in fp32. LIKELY needed for parity.

### Phi-4 and Phi-4-mini

- **Fused weights split in halves.** The checkpoint has a fused `qkv_proj` [q; k; v] and a fused `gate_up_proj` [gate; up] (checkpoint index; `phi3.py:61-62, 95-99, 124-130`).
- **LongRoPE short/long selection differs across all three runtimes:**
  - HF picks by the *current* sequence length (`tf/modeling_rope_utils.py:570-573`).
  - llama.cpp picks by the *configured* `n_ctx_seq > n_ctx_orig` (`src/llama-model.cpp:2337-2350`).
  - mlx-lm **always uses `long_factor`**; `short_factor` is never used (`rope_utils.py:49-50`).
  - Result: below 4096 tokens the RoPE factors are 1.0 in HF but up to 47.77 in mlx-lm, so the outputs are not comparable.
- **Partial rotary.** Only the first 96 of 128 dims are rotated (`phi3.py:65`).

### DeepSeek-V3 (noted only)

- **MLA structure.** q = q_b(norm(q_a(x))); the KV latent is 512 values plus a shared `k_pe` of 64. Decode uses absorbed weights (`deepseek_v3.py:88-92, 155-165`).
- **Small cache.** The KV cache is (512+64)×2 = **1,152 B per token per layer**. Ordinary MHA at these dims would be 128×(192+128)×2 = 81,920 B.
- **RoPE.** It applies only to the 64-dim part and is interleaved (`traditional=True`, `deepseek_v3.py:109-115`; HF `rope_interleave=True`, `configuration_deepseek_v3.py:114`).
- **"head_dim" is not universal.** transformers sets `head_dim := qk_rope_head_dim = 64` (`configuration_deepseek_v3.py:124`), so the key means different things in different families.
- **Softmax scale.** It is multiplied by mscale², where mscale = 0.1·ln 40 + 1, giving ×1.874 (`deepseek_v3.py:101-107`).
- **Router.** mlx-lm implements it as `group_expert_select` (`deepseek_v3.py:195-227`); llama.cpp has the equivalent (`src/llama-graph.cpp:2063-2172`; `models/deepseek2.cpp:28`).
- **Release weights.** FP8 with `weight_scale_inv`, which mlx-lm dequantises (`deepseek_v3.py:384-422`). The MTP layer (layer 61) is dropped (`deepseek_v3.py:479`).

### gpt-oss

- **Sinks.**
  - A per-head bf16 logit is concatenated to the scores before softmax and sliced off afterwards (`mlx/mlx/fast.cpp:867-885`); llama.cpp uses `ggml_soft_max_add_sinks` (`llama-graph.cpp:2731`).
  - mlx-lm's quantized-KV attention path **rejects sinks** (`base.py:119-121`).
- **Activation.** The constants are α = 1.702 and limit = 7, with a "+1" on the up branch; the gate is clamped only from above (`activations.py:21-28`). With those constants the gate is effectively a QuickGELU.
- **Interleaved `gate_up`.** Even rows are gate and odd rows are up (`gpt_oss.py:226-237`). Phi uses halves instead.
- **MXFP4 layout** (checked from the real header of openai/gpt-oss-20b):
  - `*_blocks` are U8 [E, out, in/32, 16] and `*_scales` are U8 [E, out, in/32].
  - value = LUT[nibble] · 2^(scale−127), with LUT {0, .5, 1, 1.5, 2, 3, 4, 6, −0, …, −6}.
  - The **low nibble is the even element** (`tf/integrations/mxfp4.py:29-46, 271, 298-308`).
- **YaRN `truncate=false` is ignored by mlx-lm and llama.cpp.**
  - HF honours it (`modeling_rope_utils.py:451-475`).
  - mlx-lm (`rope_utils.py:150-153`) and llama.cpp (`ggml/src/ggml.c:4482-4490`) always floor/ceil.
  - For gpt-oss I computed the correction range as [8, 18] with truncation versus [8.09, 17.40] without, so the RoPE frequencies of dims ~9–17 are a few percent off the reference.
- **Shard naming.** The official index lists 3 shards named `model-0000{0,1,2}-of-00002`, so don't infer the shard count from file names.
- **Mixed quant in the MLX conversion.** mlx-community/gpt-oss-20b-MXFP4-Q8 sets a global quantization of mxfp4 with group 32 in its config, with per-module overrides of 8-bit affine group 64 for attention and embeddings.

### Cross-cutting config parsing (pitfalls for Sky)

- **transformers 5 renamed keys** (VERIFIED by re-serialising with 5.18.0):
  - `rope_theta` + `rope_scaling` become `rope_parameters`, with θ inside.
  - `torch_dtype` becomes `dtype`.
  - `layer_types` is added.
  - Gemma 3's `rope_parameters` becomes nested per layer type.
- **mlx-lm drops unknown keys silently** (`base.py:13-21`). I fed it the transformers-5 re-serialised configs:
  - **Llama 3.1** came out as `rope_theta = 10000, rope_scaling = None`; the true values are 5e5 with llama3 ×8.
  - **Gemma 3** global RoPE scale came out as 1.0 instead of 0.125.
  - Only 16 of 135 mlx-lm model files read `rope_parameters` (qwen3 does, `qwen3.py:33-38`).
  - Sky should normalise both schemas and **fail loudly** on unknown or missing keys.
- **`type` vs `rope_type`.** Both appear in the wild: DeepSeek-V3 and Phi-4-mini use `type`. mlx-lm accepts both (`rope_utils.py:299-301`).
- **Missing keys fall back to code defaults.** Those defaults live in transformers' config classes, so pin the version used to build Sky's default tables.
- **Other quirks to handle:**
  - Inert keys, such as Phi-4-mini's `sliding_window` of 262144.
  - Padded vocab: Qwen2.5 has 152064 rows for a tokenizer of about 151.6k.
  - MLX 4-bit files have both `.biases` (affine zero-points) and `.bias` (linear bias); Qwen2.5 4-bit has both on q_proj.
  - Checkpoints can be a single file or a sharded index; Llama-3.2-1B has no index.
- **Default runtime numerics differ.**
  - mlx-lm runs **bf16 activations and a bf16 KV cache** (checked with Qwen3-0.6B-4bit: embed, logits and KV dtype are all bf16; KV is allocated in steps of 256).
  - llama.cpp runs f32 activations and an **f16 KV cache** (`common/common.h:343-344`).

## 3. Minimum op/feature checklist (ordered by Qwen3-32B first)

**Tier 0: Qwen3-32B, MLX 4-bit.** Sharded safetensors are the main path; GGUF is secondary.

1. **Config parsing.**
   - Normalise both the legacy and transformers-5 schemas.
   - Keep per-architecture default tables.
   - Take stop tokens from the `generation_config` list.
2. **Embedding.** Gather from a quantized table and dequantise one row.
3. **RMSNorm.** Accumulate in fp32 with eps 1e-6. It is needed on the hidden state (5120) and per head (128) for QK-norm.
4. **4-bit affine quantized GEMV/GEMM.**
   - Group 64, bf16 scales and biases.
   - Shapes: 5120→8192, 5120→1024 (×2), 8192→5120, 5120→25600 (×2), 25600→5120 and 5120→151936.
   - Fusions: QKV into one GEMV; gate+up with a SwiGLU epilogue.
5. **RoPE.** NeoX/rotate-half layout, θ 1e6, dim 128, fp32 angles, position offset.
6. **GQA attention.**
   - Ratio 8, head_dim 128, scale 1/√128, causal.
   - A single-query decode kernel plus a flash-style prefill kernel.
   - KV append in bf16/f16; the cache layout must be preallocated or paged.
7. **Residual adds and final norm.** Then the LM head, then sampling: greedy for the benchmark, plus temperature/top-k/top-p for parity.

**Tier 1: Llama 3.x, Mistral, Qwen2.5, R1-Distill and Phi-4.**

8. Linear biases on Q/K/V.
9. **RoPE variants:**
   - llama3 remap (a precomputed freqs table);
   - linear;
   - YaRN, with and without truncation, plus the attention factor;
   - LongRoPE with the short/long switch;
   - partial rotary.
10. Tied embeddings, with a quantized LM head.
11. **Load-time splitting of fused weights.** Phi uses halves; gpt-oss interleaves.
12. **Odd dimensions.** Non-power-of-two head counts and GQA ratios (3, 5, 7), and head_dim 64, 128 and 256.
13. **Per-file layout flags.** GGUF Llama/Mistral need interleaved RoPE; GGUF Gemma norms come pre-offset.

**Tier 2: Gemma 3, Ministral and gpt-oss.**

14. **Sliding windows.** Per-layer sliding window from `layer_types`, a rotating KV cache, and a different θ per layer type.
15. **Gemma specifics:** (1+w) RMSNorm, sandwich norms, GeGLU-tanh, a bf16-rounded embedding scale, and `query_pre_attn_scalar`.
16. Attention sinks (gpt-oss) and tanh soft-capping (Gemma 2/4).
17. **MoE.**
    - Router: softmax or sigmoid, top-k, renorm, group-limited selection, bias correction and routed scaling.
    - A gather/grouped GEMM with tokens sorted by expert for prefill.
    - Shared experts and expert biases.
18. **Low-precision formats.**
    - MXFP4 dequant: E2M1 LUT, E8M0 scale, group 32, low nibble first.
    - FP8 e4m3 dequant with 128×128 block scales.
19. **MLA (DeepSeek).**
    - Low-rank q/kv with latent norms.
    - Decoupled interleaved RoPE.
    - Absorbed decode.
    - A 576-value latent KV cache.
    - The YaRN mscale² factor.

**Also in mlx-lm/models (2026); not deep-dived.** The files are VERIFIED to exist; the one-line semantics are LIKELY, based on keyword greps:

- `qwen3_next.py`, `qwen3_5.py`, `qwen3_5_moe.py`: hybrid Gated-DeltaNet linear attention plus periodic full attention, partial rotary, shared experts, MTP.
- `gemma3n.py`, `gemma4_text.py`: per-layer inputs and cross-layer KV sharing (plus AltUp/LAuReL in 3n). Gemma 4 has final soft-cap 30.
- `llama4_text.py`: QK-norm, NoPE layers, MoE.
- `deepseek_v32.py`, `deepseek_v41.py`, `glm_moe_dsa.py`: a DSA sparse-attention indexer on top of MLA.
- `mistral4.py`: MLA plus MoE.
- `ministral3.py`: Llama-4-style position-dependent query scaling.
- `kimi_k3.py`: gated delta plus MLA plus MoE.

## 4. Qwen3-32B: dims and arithmetic

All numbers are VERIFIED from `Qwen/Qwen3-32B/config.json` and the `qwen3_32b_math.py` output.

**Config.**
- Architecture: 64 layers, hidden 5120, 64 query heads and 8 KV heads, head_dim 128, intermediate 25600.
- Vocabulary and context: vocab 151936, RoPE θ 1,000,000, `max_position_embeddings` 40960.
- Other fields: `tie_word_embeddings` false, dtype bfloat16, eps 1e-6, `attention_bias` false, `rope_scaling` null.

**Parameters.**
- Per layer:
  - q: 5120×8192 = 41,943,040;
  - k and v: 5,242,880 each;
  - o: 41,943,040;
  - MLP: 3×5120×25600 = 393,216,000;
  - linear total: 487,587,840.
- Across 64 layers that is 31,205,621,760.
- Embedding and LM head: 777,912,320 each, untied.
- Norms: 676,864.
- **Total 32,762,123,264**, which matches the model card's 32.8B (31.2B without embeddings).

**MLX 4-bit, group 64, bf16 scale + bias.**
- That is 4 + 32/64 = **4.5 bits per weight**, or 0.5625 bytes.
- Predicted file size: (31,205,621,760 + 2×777,912,320)×0.5625 + 676,864×2 = **18,429,667,328 B**. This exactly equals the `total_size` in mlx-community/Qwen3-32B-4bit's index.
- Remote headers show 450 U32 weights, 450 BF16 scales, 450 BF16 biases and 257 BF16 norm vectors.

**Decode bytes per token (batch 1, weights only).**
- (all linear layers + LM head) × 0.5625, plus norms, plus one embedding row = **17,992,094,528 B ≈ 17.99 GB (16.76 GiB)**.
- The LM head is 2.4% of that.
- For comparison, in Qwen3-0.6B the tied 151936×1024 head is about 26% of the bytes, so small-model benchmarks stress different kernels.

**FLOPs.**
- Matmuls: 2×31,983,534,080 = **63.97 GFLOP per token**.
- Attention: 64 layers × 64 heads × 128 × 2 (QKᵀ and PV) × 2 FLOP = **2,097,152·T FLOP per token**, where T is the context length. At T = 4096 that is 8.59 GFLOP.
- Arithmetic intensity is about 3.6 FLOP per byte, so decode is purely bandwidth-bound.

**KV cache in bf16.**
- 2 × 64 × 8 × 128 × 2 = **262,144 B (256 KiB) per token**.
- That is 1 GiB at 4k context, 8 GiB at 32k and 10 GiB at 40,960.
- Each decode step also reads T × 256 KiB: +0.134 GB at T = 512 and +1.07 GB (+6%) at T = 4096.

**Bandwidth roofline.**
- tok/s ≤ BW / (17.99 GB + T × 262,144 B).
- Examples: 30.3 tok/s at 546 GB/s, or 28.6 tok/s at T = 4096. The 546 GB/s figure is the M4 Max spec (LIKELY); other agents should supply M5 Max bandwidth.
- Q4_0 is also 4.5 bits per weight, so its weight bytes are comparable. The GGUF output-tensor quant type is not verified here.
