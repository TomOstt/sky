# Part XXI — Speculative decoding

Plain decode cannot pass the bandwidth ceiling: 33.9 tok/s for Qwen3-32B 4-bit on an M5 Max ([Part IV](part04-hardware-limits.md) §4). Speculative decoding can. A cheap drafter proposes several tokens, and one pass of the target model's weights checks them all. This part designs Sky's speculative engine:
- the theory and the per-chip arithmetic;
- the batch-invariant verify kernels;
- the drafters, ranked;
- tree sizing;
- the acceptance rules, including keyed-Gumbel coupling, which makes speculative output byte-identical to plain output at any temperature;
- accept, reject and rewind on the GPU, including DeltaNet state;
- pipelined rounds;
- the exactness test;
- the modeled speedups;
- the competition;
- what it takes to reach an honest "60 tok/s".

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R8 §2.2). The primary source for this part is [R8](research/R8-algorithms.md).

## 1. Summary

1. **Speedup is S = τ / (1 + γc + γβ)** (R8 §2.1), where:
   - τ is the tokens emitted per round;
   - γ is the number of drafted tokens;
   - c is the cost of one draft token, relative to one target step;
   - β is the extra cost of verifying one more token, relative to one target step.

   The verify kernel sets β. Its ceiling is the free verify width k\* = R·b/16, where R is the ridge point and b the bits per weight: about 8–9 tokens on the shader ALUs of every chip, and about 33–35 on the M5 neural accelerators (`roofline.py`).
2. **The core kernel is a batch-invariant verify GEMV.** Verifying M tokens must cost about the same as decoding one, and must produce bit-identical logits rows.
   - For the weight GEMVs on the ALUs, the target is ≤ 1.3× the M = 1 time up to M = 8 (SPINE §3). That is at the arithmetic edge once dequantization is counted (§2.3), so M = 5–8 may need the neural accelerators.
   - On the M5, the neural accelerators serve M = 9–32.
   - Attention has its own free width, q\* = R·s/(2G): ≈ 4 queries on the ALUs and ≈ 15 on the M5 neural accelerators ([Part XV](part15-attention.md) §2.2). Long-context verify attention therefore runs on the accelerators.
3. **On the M5 Max, the verify kernel alone is not a large lead over mlx-lm.**
   - MLX's verify on that chip is already moderately cheap: β ≈ 0.06–0.18 per drafted token, from mlx#3791's +12% at 3 tokens and +48% at 5. On M2-class GPUs MLX pays β ≈ 0.5–1.
   - Sky's kernel advantage over mlx-lm's speculation on the M5 Max is therefore about 1.2× (§10).
   - The larger lead comes from three things mlx-lm does not have:
     - exactness: output byte-identical to plain output;
     - drafters mlx-lm cannot run: EAGLE-3, DFlash, prompt lookup, and MTP for the hybrid models, where mlx-lm supports separate draft models only;
     - pipelined rounds without a host synchronization.
4. **Exactness is a kernel property plus a sampling rule.**
   - Batch-invariant kernels (SPINE D6) plus keyed-Gumbel sampling, keyed by (seed, request, position), make speculative output byte-identical to Sky's own plain output at any temperature, for any drafter, draft length or tree shape (Daliri et al. 2024; R8 §3.2).
   - mlx-lm cannot claim this: its speculative output departs from plain greedy at token 91 on Qwen3-32B with a 0.6B draft on an M5 Max (mlx-lm #1470). That script becomes our adversarial CI test (§9).
5. **Honest numbers for Qwen3-32B on the M5 Max** (modeled, [Part IV](part04-hardware-limits.md) §8):
   - with the public EAGLE-3 drafter, 1.7–2.2× plain if β ≤ 0.05: 47–59 tok/s typical, ≈ 60 or more on predictable or input-grounded work (§10, §15);
   - about 1.5× at MLX-on-M5 verify costs, and nothing at MLX-on-M2 costs;
   - an average of 60 tok/s or more needs a DFlash-class drafter for Qwen3-32B, which nobody has published (gate G9; §12);
   - the most credible "60+" in 2026 is the Qwen3.6/3.8-27B hybrid with its own MTP head (§14).

## 2. Theory

### 2.1 The speedup model

Leviathan et al. give the expected tokens per round with γ drafts and i.i.d. per-token acceptance α as E = (1 − α^(γ+1)) / (1 − α). Their speedup theorem assumes that "γ + 1 concurrent evaluations of Mp" cost no more wall time than one (arXiv 2211.17192, Thm 3.8; R8 §2.1, `[VERIFIED]`). That assumption is exactly what fails on Apple GPUs today. R8 generalizes it with an explicit verify cost:

```text
S(γ) = τ / (1 + γ·c + γ·β)

τ  tokens emitted per round (measured, or E above)
c  time of one draft token / time of one plain target step T₁
β  extra time to verify one more token / T₁      (β = 0 is Leviathan's assumption)
trees: replace γ·c by depth·c and γ·β by (nodes − 1)·β   (Sequoia's form; R8 §2.1)
```

Published τ for Qwen3-32B drafters imply a per-token acceptance α ≈ 0.55–0.69 at the model's recommended temperature of 0.6 (R8 App. B, §A). Real acceptance decays with draft depth, so α is only a summary.

### 2.2 Free verify width

A weight-streaming matmul over k token rows with b-bit weights has an arithmetic intensity of 16k/b FLOP per byte. It stays bandwidth-bound while that is below the ridge point R = F_peak / BW, so:

**k\* = R · b / 16** (R8 §2.2, `[EXPERIMENTAL derivation; cross-checked against Sequoia's tree size of 128 on an A100, where the formula gives ≈ 153]`)

`roofline.py` (`verify_free_tokens`) gives, for 4.5-bit weights:

| Chip | Bandwidth used | k\* on shader ALUs | k\* on the matrix units |
|---|---:|---:|---:|
| M2 Pro, 16-core | 196 GB/s (measured raw read) | 8.2 | 8.2 (no neural accelerators) |
| M4 Max, 40-core | 491 GB/s (planning) | 9.3 | 9.3 (no neural accelerators) |
| M5 Pro, 20-core | 276 GB/s (planning) | 8.4 | 33.4 |
| M5 Max, 32-core | 414 GB/s (planning) | 9.0 | 35.3 |
| **M5 Max, 40-core** | **534 GB/s (planning)** | **8.7** | **34.2** |
| M5 Ultra, 80-core | 1,080 GB/s (planning) | 8.6 | 33.9 |

The ALU column is nearly the same on every chip: Apple scales shader ALUs and bandwidth together. The matrix column is what changed with the M5. **Speculation is how a decode workload uses the M5's otherwise idle neural accelerators** (SPINE §2). `[LIKELY: the M5 neural-accelerator rates are secondary-source measurements, R6 §A2]`

Weight quantization and speculation compete for the same headroom. At 16-bit weights, k\* is 3.6× larger. Dequantization is paid once per weight per pass, whatever k is (R8 §2.2).

### 2.3 What the ridge arithmetic leaves out

**Dequantization on the ALU path.** The ALU budget per streamed weight is 8.7 FMA slots on the M5 Max (16.6 TFLOPS / 2 per FMA ÷ 0.95 T weights/s at 534 GB/s), and the same 8.7 on the M2 Pro. A verify kernel spends, per weight:
- about 2 operations of dequantization, shared by all M rows (mask and convert; R3 §3.3);
- M FMAs, plus the per-group scale and bias work (≈ M/8);
- activation pre-scaling (≈ M/8 when 8 output rows share one threadgroup).

At M = 4 that is ≈ 7 operations, within budget. At M = 8 it is ≈ 12, so the kernel becomes compute-bound at roughly 1.4× the bandwidth time unless it runs near peak ALU efficiency. `[EXPERIMENTAL: arithmetic; XXI-T1 measures]`

Measured kernels agree on the direction:
- R3's 60-line multi-row kernel reached ×1.30 at M = 3 on the M2 Pro, and spilled registers from M = 4 (R3 §7.G).
- MLX's `qmv_wide` costs ×2.25 for a whole 70B forward at M = 8 on the M5 Max (mlx#3791).

So **SPINE's "M ≤ 8 at ≤ 1.3×" is reachable on the ALUs for M ≈ 4–5 with confidence, and for M = 8 only with a near-peak kernel.** On the M5, rows 5–8 can move to the neural-accelerator tier (§3.3). This is recorded in §19.

**Attention at long context.** Each K or V element of a KV head serves all G query heads of its group, once per verified query. [Part XV](part15-attention.md) §2.2 derives the intensity, 2·G·q / s FLOP per KV byte (s bytes per KV element), and from it the **free verify width of attention**:

**q\* = R · s / (2G)**, the attention counterpart of k\* = R·b/16.

| Path | q\*, Qwen3-32B, fp16 KV | q\*, Qwen3-32B, int8 KV | q\*, Qwen3-8B, fp16 KV |
|---|---:|---:|---:|
| M5 Max neural accelerators | **15.2** | 7.6 | 30 |
| M5 Max shader ALUs | **3.9** | 1.9 | 7.8 |
| M2 Pro shader ALUs | 3.7 | 1.8 | 7.3 |

(Part XV §2.2; `[LIKELY: arithmetic on roofline.py rates]`)

At short context this does not matter: attention is ≈ 0.25 ms of a 36 ms step at context 512. At 32K it does. Part XV's verify-step ratios T(q)/T(1) from attention alone, at 100% of peak compute (optimistic), for Qwen3-32B on the M5 Max at 32K:

| Path | q = 4 | q = 8 | q = 16 |
|---|---:|---:|---:|
| shader ALUs | 1.01 | **1.34** | **2.01** |
| neural accelerators | 1.00 | 1.00 | 1.02 |

At a more realistic 70% of ALU peak, q = 8 costs ≈ 1.5× per round. `[EXPERIMENTAL: arithmetic]`

Consequences:
- **Long-context verify attention runs on the neural accelerators** (Part XV AT-D4). GQA packing makes this possible: the 8 query heads × q tokens of a KV group form an (8q × 128) query tile, which is NA-shaped for q ≥ 4.
- **On M1–M4 the verify width is capped by context.** The tree sizer of §5 takes context length as an input.
- **int8 KV halves q\*.** The KV format and speculation compete for the same compute ([Part XIV](part14-kv-cache.md) §12).
- **SPINE §3's "≤ 1.3× at M = 8" applies to the weight GEMVs.** Attention has its own targets in Part XV §5.3: ≤ 1.1× for q ≤ 8 on the M5 Max NA path at 4K–32K, and ≤ 1.3× at q = 4 on M1–M4 up to 16K.

### 2.4 β measured today

| Engine, chip, model | Verify cost | β per drafted token | Source |
|---|---|---:|---|
| mlx-lm, M2 Pro, Qwen3-8B forward | M = 2: 1.73×, 4: 3.33×, 8: 6.48× | ≈ 0.7–1.0 | R4 §7.D `[VERIFIED]` |
| MLX `qmv`, M2 Pro, Qwen3-32B q_proj | M = 4: 2.43–2.46×; M = 6–32: flat ≈ 4.4–4.8× | ≈ 0.48 | R3 §7.C `[VERIFIED]` |
| MLX `qmv_wide`, M5 Max, 70B 4-bit forward | M = 3: +12%; M = 5: +48%; M = 8: +125%; M = 16: 3.9× | **≈ 0.06–0.18** | mlx#3791 via R3-issues E4 `[VERIFIED statement; contributor data]` |
| llama.cpp, M2 Pro, Qwen3-8B Q4_0 | M = 4: 2.08×; M = 8: 3.76× | ≈ 0.36–0.39 | R5 §8.G `[VERIFIED]` |
| R3's experimental kernel, M2 Pro, 32B gate/up | M = 2: 1.08×; M = 3: 1.30× | ≈ 0.04–0.15 | R3 §7.G `[VERIFIED]` |
| **Sky target** | M ≤ 8 at ≤ 1.3× | **≤ 0.04** | SPINE §3 |

## 3. The batch-invariant verify-kernel family

### 3.1 The rule

A verify pass at width M must produce, for each row, the logits that plain decode would have produced at that position, bit for bit. The same holds for every intermediate value that feeds later tokens: the KV entries and the hidden-state taps an EAGLE-3 drafter reads.

Plain decode is the M = 1 kernel. So every kernel of the decode/verify family uses one per-row schedule ([Part XVI](part16-decode.md) §3.2):
- the lane-to-K mapping;
- the loop order;
- fp32 accumulation with the same FMA structure;
- one `simd_sum` per row;
- no split-K across threadgroups;
- reductions across threadgroups only through fixed-order partials.

M changes how many rows a pass computes, never how one row is computed.

Implementation (R8 §3.5, rule 1):
- The kernel IR carries a `reduction_schedule` attribute.
- The verifier in `sky-kernel` requires decode and verify instantiations of a family to carry the same attribute.
- They are separate instantiations (M = 1, 2, …, 8), so M = 1 decode pays nothing for padding.
- A bit-equality test (XXI-T1) proves the attribute means what it says on real hardware, because the Metal compiler is compiled with safe math ([Part XVI](part16-decode.md) §3.8).
- If the test ever fails, the fallback is the conservative form: decode runs through the M = k kernel with masked rows. That wastes ALU work, not bandwidth (R8 §3.3).

### 3.2 ALU tier: M = 1 … 8 (all chips)

`qgemv` with `token_rows = M`: the verify GEMV family of [Part XI](part11-metal-backend.md) §9.2 (instances MMAX 2/4/8/16). The weight word is decoded once and applied to all M rows held in registers.

Design points, from R3's experiment (R3 §7.G):
- Keep activations in fp16 registers and handle fewer output rows per simdgroup as M grows. R3's one-launch kernel spilled at M = 4 with 8·M fp32 activations plus 4·M accumulators per lane.
- Never split M across separate launches. That re-streams the weights: ×2.5–2.7 at M = 6 and ×3.6–3.7 at M = 8 in R3's measurements.
- Fuse every prologue and epilogue of the decode kernel ([Part XVI](part16-decode.md) §3.5–§3.7). The verify pass is the decode program with M rows.

Expected cost: about ×1.0–1.15 up to M ≈ 4–5, and ×1.3–1.6 at M = 8, depending on achieved ALU efficiency (§2.3). `[EXPERIMENTAL: XXI-T1]`

### 3.3 Neural-accelerator tier: M = 9 … 32 (M5 only)

`mpp_qgemm` from [Part XVII](part17-prefill.md) §3, with the token dimension padded to the fragment size:
- the weight tile is dequantized into threadgroup memory, or used as a native int4 operand with post-scaling;
- the neural accelerators do the MACs;
- the ALUs do dequantization in parallel.

At M = 16 the step needs ≈ 1.0 TFLOP per 18 GB weight pass: about 30 TFLOPS sustained to stay at the bandwidth time, 46% of the 65 TFLOPS peak. At M = 32 it needs ≈ 94%. So the realistic free width on the M5 Max is about 16–24 tokens, not 34. `[EXPERIMENTAL: depends on matmul2d efficiency at 16–32-row tiles; R6 found ≥ 64 × 64 tiles are needed for peak]`

MLX does not route these widths to its neural-accelerator kernels:
- quantized M ≤ 12 stays on `qmv` for Qwen3-32B shapes;
- M = 13–32 mostly goes to non-NA split-K;
- only gate/up and the LM head reach `qmm_t_nax` (R3-kernels §d, worked routing example).

Maintainers declined both routing fixes and verify-window attention fixes (mlx#4198, mlx#3826). Owning this routing is a concrete Sky advantage on the M5.

### 3.4 Byte-identity across tiers

The two tiers reduce differently. The ALU tier uses lane-split fp32 dot products. The neural-accelerator tier uses an MMA unit whose internal order is hidden behind `air.externally_defined` functions (R6 §A2). Their logits will differ in the last bits. If decode runs on the ALU tier and verification on the neural-accelerator tier, near-ties will flip exactly as in mlx-lm #1470.

Decision:
- **Byte-identical mode** (the default): decode and verify use one family. On M1–M4, and on the M5 unless the next condition holds, that is the ALU tier, and verify width is capped at 8.
- **Neural-accelerator family for both,** on the M5, if a `mpp_qgemm` instantiation at M = 1 (padded) streams 4-bit weights at ≥ 95% of the raw read, and [Part XI](part11-metal-backend.md)'s `verify_mpp` probe passes its bitwise row-invariance test on that chip (XI-T12). Then plain decode also runs on it, and wide trees stay byte-identical. R6 notes that a one-row `matmul2d` is slower than a GEMV in compute terms. Whether it can still stay bandwidth-bound is the open question. `[EXPERIMENTAL: XXI-T2 decides]`
- **Distribution-exact wide mode,** if the condition fails: verification on neural-accelerator tiles, decode on the ALU tier. Output remains a correct sample of the target distribution within hardware numerics (Chen et al.; R8 §2.1), but it is no longer byte-identical to plain output. `sky explain` labels the plan as such.

[Part XV](part15-attention.md) applies the same rule to attention (AT-D4, XV-T5): decode and verify attention share one arithmetic path, chosen by measurement.

### 3.5 Verify attention

Designed in [Part XV](part15-attention.md) §5. Verify widths up to 32 queries use the same `flash_decode` kernel as decode (AT-D3). The invariants this part relies on:
1. **Same traversal as decode.** The context is cut into splits of a fixed size S aligned to absolute positions, with a fixed-order combine (AT-D2). A verified row at position p + i uses exactly the splits decode would use at that position (R8 §3.5, rule 2).
2. **Chains use a causal window mask. Trees use ancestor bitmasks plus a per-node gather of the window block** in depth order, i.e. by logical position (Part XV rule I7). Masks alone are not enough for trees: hardware fragments may sum in pairwise order, so moving a nonzero term within a block can change rounding. Fully masked partials never enter the combine (rules N2, N3).
3. **Per-row positions.** RoPE and the mask use the node's position p + depth, not its slot index. mlx-lm's KV cache has a single RoPE offset, which is why an MLX EAGLE-3 prototype could not verify trees (discussion #890, R8 §2.2).
4. **K and V for a position are written once, before attention reads them,** and verify writes the same bytes as decode ([Part XIV](part14-kv-cache.md) §4.2; Part XV rule I8). After acceptance, the accepted tree path is compacted to its final slots (§7.1).
5. **More than about 4 queries per KV group at long context run on neural-accelerator tiles** (§2.3). MLX's fused vector kernel takes at most 4 queries for GQA-8 models and falls back to unfused attention for 5–8 (R3-kernels §c).
6. **Decode attention uses the same arithmetic path as verify** (Part XV AT-D4). On the M5, whether that path is the neural accelerators or the ALUs is decided by one measurement: XV-T5 requires NA decode attention to reach ≥ 95% of the ALU variant's bandwidth at q = 1, or the plan uses the ALUs for both. This is the attention half of §3.4's rule.

## 4. Drafting options, ranked

| Rank | Drafter | Weights | Published τ (setting) | Draft cost c on the M5 Max | Verify width | Status in mlx-lm / llama.cpp |
|---|---|---|---|---|---|---|
| 1 | **EAGLE-3** for Qwen3-32B | `RedHatAI/Qwen3-32B-speculator.eagle3`; `AngelSlim/Qwen3-32B_eagle3` | RedHat, T = 0.6, top-p 0.95, top-k 20: HumanEval 2.29 (k = 3) / 2.53 (k = 7); GSM8K 2.49 / 3.08; CNN/DM 2.15 / 2.33. AngelSlim, 2 tokens: 1.86–2.01 | ≈ 0.027 at 4-bit; ≈ 0.096 at bf16 | γ + 1 = 4–8 | none / supported |
| 2 | **DFlash** for Qwen3-8B | `RedHatAI/Qwen3-8B-speculator.dflash` (and `.dflash2`) | 5.88 on HumanEval, 6.81 on math, 2.80 on QA (T = 0, thinking off, 15 tokens) | one forward per 15-token block | 16 | open request (mlx-lm #1135) / supported |
| 3 | **Draft model** Qwen3-0.6B / 1.7B | the Qwen checkpoints; same 151,936-token vocabulary | no single-device number for Qwen3-32B at T = 0.6 (gap). Mirror-SD on 8× M2 Ultra: 2.5–5.5 | 0.6B: 0.027–0.034 with Sky's plan; 1.7B: ≈ 0.054 bandwidth-only | γ + 1 | supported / supported |
| 4 | **Prompt lookup / suffix trees** | none | 2–4× on input-grounded tasks, greedy (PLD README); up to 5.3× on agentic loops (SuffixDecoding); ≈ 1× on open-ended text; llama.cpp's `ngram-simple` found 0 drafts in 256 tokens of fresh code | ≈ 0 (CPU) | γ + 1 | none (PRs #1497/#1498 closed) / supported |
| 5 | **MTP head** of the Qwen3.6/3.8-27B hybrids | shipped in the checkpoints (`mtp_num_hidden_layers: 1`) | per-depth acceptance 73/48/32%, τ ≈ 2.53 (modelfit.io, UNVERIFIED) | ≈ 0.066 (the 248K-vocabulary LM head dominates) | γ + 1 | MTP weights dropped at load; PR #990 open / supported |

Sources: R8 §2.3–§2.7, R4 §5.1, R5 §7 (`[VERIFIED]` unless marked).

The c column is EXPERIMENTAL arithmetic from [Part XVI](part16-decode.md) §7. Weights are read at 70–85% of the planning bandwidth, against a target step of ≈ 35.9 ms.

**1. EAGLE-3 (first, at G6).**
- What the checkpoint holds (RedHat, `[VERIFIED: config and safetensors headers, R8 §2.3]`):
  - one Llama-style decoder layer at hidden size 5,120, whose QKV input is concat[embedding, hidden];
  - an `fc` layer from 3 × 5,120 to 5,120;
  - a 32,000-entry draft vocabulary with `d2t`/`t2d` maps;
  - an embedding copy.

  That is 1.56 B parameters in total, about 0.78 B read per draft step. The embedding copy is a one-row gather and is not streamed.
- **Runtime needs:**
  - hidden-state taps from three target layers (low, middle, high), written by the verify pass for every verified row;
  - a draft KV cache;
  - the vocabulary map;
  - per-row positions.
- **Quantize the drafter.** The drafter is not the model under comparison, so D5's "never re-quantize" does not bind it. At 4 bits, c falls from ≈ 0.096 to ≈ 0.027. The τ loss is measured (XXI-T15). Exactness is unaffected, because the target verifies everything (§6).
- No MLX-format EAGLE-3 or DFlash weights exist, and mlx-lm has no EAGLE support (R8 §2.3). Converting the safetensors is trivial; the work is the runtime (XXI-T8).
- At T = 0.6, published τ is 2.2–3.1, far below the EAGLE-3 paper's ≈ 6, which was greedy, tree-drafted and measured on an H100 (R8 §2.3–§2.4). τ in Qwen3's thinking mode is unmeasured. A RedHat thinking-mode speculator exists without published numbers (`RedHatAI/Qwen3-32B-Thinking-speculator.eagle3`).

**2. DFlash for 8B.** A block-diffusion drafter (arXiv 2602.06036):
- 5 layers, sliding window 2,048;
- block size 16, so one forward drafts 15 tokens;
- full 151,936-entry LM head;
- auxiliary hidden states from target layers [2, 10, 18, 26, 34];
- 2.29 B parameters.

τ = 5.88 on HumanEval (T = 0) makes it the strongest public drafter for any Qwen3 size. Its 16-token verify needs the neural-accelerator tier. On an ALU-only chip the same verify costs two weight passes. Modeled speedup:
- with a 16-wide verify at 1.2× and a block cost of 0.3: S ≈ 3.9 on HumanEval and ≈ 1.9 on QA;
- at a 2.2× verify cost: ≈ 2.35 and ≈ 1.1. `[EXPERIMENTAL]`

DFlash for 8B is the M2 Pro / M5 demonstration path. DFlash for 32B does not exist (§12).

**3. Draft models.** Drop-in, because all Qwen3 sizes share the vocabulary (R8 §2.5).
- In mlx-lm the 0.6B draft is CPU-bound: about 1.94 ms of Python graph building and encoding per draft token on the M2 Pro (R4 §7.E). R8 therefore estimates c ≈ 0.05–0.1 in a latency-bound runtime.
- Sky's static plan targets c ≈ 0.03 ([Part XVI](part16-decode.md) §7).
- A truncated draft vocabulary (EAGLE-3 style) cuts the 0.6B's tied LM head, 26% of its bytes. `[EXPERIMENTAL]`
- Qwen3-1.7B costs about 2.8× more per token. It pays only if its τ is much higher.

**4. Prompt lookup and suffix trees.**
- Prompt-lookup decoding matches the last n-gram (n ≤ 3) against the prompt and copies up to 10 continuation tokens (R8 §2.6).
- SuffixDecoding keeps suffix trees over prompts and previous outputs, with adaptive draft lengths.

Both run on the CPU in microseconds. They pay on summarization, retrieval-augmented QA, code editing and agent loops, and do nothing on fresh text. llama.cpp's `ngram-simple` needs a 12-token exact match and found nothing in fresh code generation (R5 §8.C).

For a 32B target a CPU in the loop is affordable: one CPU round trip leaves the GPU idle about 105 µs (R6 §B3), 0.3% of a 36 ms step. It composes with rank 1–3: each round takes the longer of the two proposals, or a tree containing both.

**5. MTP for the Qwen3.6/3.8 hybrids.** The second headline model (SPINE D10). It needs the Gated-DeltaNet state handling of §7.3 (§14).

## 5. Tree sizing

**Chains first.** With i.i.d. acceptance α, the best chain length γ\* maximizes (1 − α^(γ+1)) / ((1 − α)(1 + γ(c + β))):

| α (per token) | c + β = 0.05 | 0.08 | 0.13 | 0.25 | 0.5 |
|---|---|---|---|---|---|
| 0.60 | γ\* = 4, S = 1.92 | 3, 1.75 | 3, 1.57 | 2, 1.31 | 1, 1.07 |
| 0.65 (EAGLE-3 on Qwen3-32B, T = 0.6) | 5, 2.11 | 4, 1.91 | 3, 1.69 | 2, 1.38 | 1, 1.10 |
| 0.70 | 6, 2.35 | 5, 2.10 | 4, 1.82 | 2, 1.46 | 1, 1.13 |
| 0.80 | 8, 3.09 | 6, 2.67 | 5, 2.24 | 3, 1.69 | 2, 1.22 |

`[EXPERIMENTAL: i.i.d. model; real acceptance decays with depth, so measured per-position acceptance replaces α in the tuner]`

At Qwen3-32B's measured α and Sky's target costs (c ≈ 0.03, β ≈ 0.02–0.05), the best chain is 4–5 drafts. That is a verify width of 5–6, inside the ALU tier. Wider trees pay only when β stays small past M = 8, which means the neural-accelerator tier on the M5.

**Trees.** Sequoia's rule (R8 §2.8): grow the tree greedily by the marginal acceptance probability of each candidate node, and stop when the expected accepted tokens added by one more node fall below its marginal cost β(n). The node budget is the measured free width:
- 4–8 on ALU-only chips, where a chain of 3–5 is near-optimal (R8 §2.8);
- 16–24 on the M5 Max neural-accelerator tier (§3.3);
- fewer at long context (§2.3).

Inputs to the sizer: the measured β(M) curve per chip and context bucket, from `sky-tune` (SPINE D11c); the drafter's per-depth acceptance on a calibration set; c.

**Dynamic trees** (EAGLE-2 style, draft confidence as an acceptance proxy) change the shape per round. In a static plan, shape changes must be chosen by the GPU, through indirect dispatch or indirect command-buffer ranges, where the GPU picks the count (R6 §A4). Static shapes first; dynamic shapes are EXPERIMENTAL.

## 6. Acceptance rules

Logits are always computed in fp32. bf16 logits near 32–64 have a spacing of 0.25, so exact ties are common: #1470's divergence was an exact tie at 33.75 (R8 §3.4). Argmax ties break to the lowest token id.

| Rule | How it accepts | Acceptance per position | Output guarantee | Where used |
|---|---|---|---|---|
| **Keyed Gumbel coupling** (Sky default) | The target's token at position t is y_t = argmax_x (l_t(x)/T + g(seed, request, t, x)), with Gumbel noise from a counter-based hash (Philox). The draft is accepted iff d_t = y_t; the first mismatch emits y_t. A drafter that uses the same noise proposes argmax(log q_t + g). | between (1 − TV)/(1 + TV) and 1 − TV, where TV is the total-variation distance between target and draft (Daliri et al., arXiv 2408.07978) | **byte-identical to Sky's plain output** at any temperature, given batch-invariant logits; independent of drafter, draft length and tree shape | default; CI-tested |
| **Rejection sampling** (Leviathan; Chen et al.) | Accept d_t with probability min(1, p(d_t)/q(d_t)); on rejection, sample from normalize(max(0, p − q)); bonus token from p | Σ min(p, q) = 1 − TV (optimal) | the target distribution "within hardware numerics"; not the same sample as a plain run with the same seed | `--accept=rejection` |
| Sample-and-match (mlx-lm, llama.cpp) | Sample the target independently at each position; accept iff equal | Σ p·q (collision probability) | the target distribution | competitors (`generate.py`, `common/sampling.cpp:678-710`) |

Sources: R8 §3.1–§3.2, R4 §5.1, R5 §7.

Why keyed Gumbel is the default:
1. **It makes exactness testable byte for byte.** Plain decode uses the same rule with no draft, so the sampled text of a plain run is a deterministic function of (prompt, seed). A speculative run must reproduce it exactly. SGLang ships seeded-hash Gumbel sampling for its deterministic mode (LMSYS blog, 2025-09-22; R8 §3.2).
2. **It turns sampled tree verification into greedy tree verification.** At each tree node the target's choice is a deterministic function of that node's prefix: the noise depends on the position, not on the path. The accepted path walks from the root, following the child whose token equals the target's choice. No residual distributions, no per-branch random draws.
3. **It beats sample-and-match.** Coupled acceptance is near 1 − TV in practice (R8 §3.2, `[LIKELY]`), against Σ p·q. If draft = target = (0.5, 0.5), sample-and-match accepts 50% and coupling accepts 100%. At p_max = 0.9 the figures are 82% against 100% (R8 §3.1).

Filters compose: top-k, top-p and min-p select a candidate set S_t from the target's logits, and y_t = argmax over S_t of (l/T + g). That is an exact sample of the filtered, renormalized distribution. Qwen's card recommends top-k 20 with T = 0.6 / top-p 0.95 (thinking) or T = 0.7 / top-p 0.8 (non-thinking), and warns against greedy decoding in thinking mode (R8 §1.1). **The operating point that matters is sampling, so the exactness claim that matters is the sampled one.**

Never promised: bit-identity with MLX, llama.cpp or a bf16 reference. bf16 and fp16 greedy outputs diverge on 49–100% of prompts (arXiv 2609.26621; R8 §3.4).

## 7. Accept, reject and rewind on the GPU

One round, entirely on the GPU, with no CPU in the dependency chain:

```text
 inputs (GPU buffer): last token, position p, RNG key, draft state, stop flag
   │
   ├─ draft      γ steps of the drafter (EAGLE-3: 1 layer + 32K head, reads hidden taps)  ─► d₁…d_γ, (q rows if rejection mode)
   ├─ verify     target, M = γ+1 rows (chain) or n nodes (tree): decode program with M rows
   │              writes KV at positions p…p+γ (tree: scratch slots), hidden taps, fp32 logits candidates per row
   ├─ accept     per row: y_i = keyed-Gumbel choice (or rejection test); n = accepted prefix length
   │              emits d₁…d_n, y_{n+1}; writes new last token, p ← p+n+1, RNG counter, stop flag, ring-buffer entries
   ├─ commit     KV length ← p+n+1 (chain: nothing to move; tree: copy the accepted path's KV into place)
   │              draft cache length ← matching value; DeltaNet / conv state ← committed state (§7.3)
   └─ next round reads the same buffers (pre-committed command buffer, §8)
```

### 7.1 KV trim

The KV cache is preallocated (SPINE D9), so trimming is a length update:
- **Chain drafts:** the verify pass already wrote the accepted tokens' K and V at their final positions, and they are exactly what plain decode would have written (§3.5, rule 4). Rejected positions are overwritten by the next round.
- **Tree drafts:** the verify pass writes node K/V into scratch slots, and a small copy kernel moves the accepted path into place. That is 256 KiB per accepted token for Qwen3-32B, trivial.

mlx-lm's `trim(n)` is also O(1) (R4 §4.1). The difference is that Sky's runs on the GPU, without the per-round host synchronization mlx-lm performs (§8).

### 7.2 Drafter state

- The EAGLE-3 draft KV and a draft model's KV trim the same way.
- After a full accept, the drafter still needs the target's last emitted token. mlx-lm re-feeds it (generate.py:646-649; R4 §5.1). Sky folds it into the next round's first draft step.

### 7.3 Gated-DeltaNet state (Qwen3.6/3.8 hybrids)

The hybrid models keep a recurrent state per linear-attention layer:
- 48 layers × 48 heads × 128 × 128 fp32 values, ≈ 151 MB;
- plus a short convolution state.

Plain decode reads and writes it every token (R8 §1.7). Two rules (R8 §3.5, rule 4):
1. **Verify runs the per-position recurrent update,** not the chunked parallel scan used for prefill. Their floating-point orders differ, so only the per-position update reproduces plain decode bit for bit. Its arithmetic is small: about 6 × 48 × 128² FLOP per layer per token.
2. **Rollback without checkpoints.** The verify kernel reads S₀, the state at the start of the round, and walks the γ+1 positions in registers. Each 128-float state row is independent given the token's k, v, gate and β, so a row block fits in registers. It writes no state.

   After acceptance, a commit kernel recomputes S_{n+1} from S₀ with the same per-row update code: one read and one write of the state (≈ 302 MB), whatever n is. Checkpointing every draft position would instead write γ+1 states (≈ 151 MB each).

At γ = 3 the commit costs ≈ 0.57 ms at 534 GB/s, about 2% of a 27B round. `[EXPERIMENTAL: XXI-T13]`

mlx-lm cannot run speculative decoding on these models: its speculation requires trimmable caches, and the recurrent `ArraysCache` is not trimmable (mlx-lm #1446; R4 §5.1).

## 8. Pipelined rounds

mlx-lm performs one full synchronization per round: `mx.eval(tokens, draft_tokens)` and two `.tolist()` calls (generate.py:620-622). "The GPU idles while Python compares tokens, rewinds caches and builds the next draft graph" (R4 §2.3, `[VERIFIED]`).

Sky's rounds are a static plan ([Part XX](part20-persistent-execution.md)):
- **Pre-committed command buffers.** The CPU commits K rounds ahead. Each round reads its inputs (last token, position, RNG counter, draft state) from GPU buffers written by the previous round's accept kernel. A pre-committed buffer leaves a 0.4 µs gap, against ≈ 105 µs when the CPU waits (R6 §B3).
- **Fixed shapes.** The verify width is fixed per plan. A round that needs fewer tokens (near `max_tokens`) pads. Positions are data, so variable acceptance needs no re-encoding.
- **Stop flags.** EOS or a stop sequence sets a flag that turns the remaining pre-committed rounds into no-ops. The CPU learns of it through an `MTLSharedEvent` and the token ring buffer.
- **CPU-side drafters** (prompt lookup, suffix trees) are the exception. Their round boundary includes one event-driven CPU step, about 0.1 ms, which is affordable for 32B targets (§4).

Overlapping the draft of round r+1 with the verify of round r ("speculative speculative decoding", arXiv 2603.03251), or drafting on the Neural Engine (Mirror-SD), would add concurrency on one memory bus. On one M5 Max the GPU and the ANE share the 614 GB/s (R8 §2.6). Both are [MOONSHOT] items for G9 (SPINE §6: draft-on-ANE).

## 9. The exactness CI test

Built from mlx-lm #1470 (`[VERIFIED: R8 §3.4]`). On an M5 Max 128 GB with MLX 0.31.2, target `mlx-community/Qwen3-32B-6bit` and draft `Qwen3-0.6B-MLX-6bit`, temperature 0:
- `num_draft_tokens` 1–3 matched plain generation; 4 diverged at token index 91;
- the logits at the divergence were tied exactly (33.75 and 33.75, gap 0.0);
- the same failure was reproduced on Qwen3-8B-6bit (draft 8, index 99) and on Qwen3-4B;
- mlx-lm's MTP pull requests saw about one divergence per 150–190 tokens, including a self-speculation control;
- the maintainers' response was documentation: "lossless up to floating-point associativity, not bit-exact" (PR #1592).

R4 reproduced the effect on the M2 Pro: 3 of 9 speculative greedy configurations diverged (R4 §7.D).

Sky's suite, `crates/sky-llm/tests/spec_exactness.rs`, runs on the self-hosted macOS runner (M2 Pro, Qwen3-8B with a 0.6B draft) and nightly on the M5 Max host (Qwen3-32B):

| Test | What it asserts | Why it is strong |
|---|---|---|
| **#1470 port** | The same prompts, models (Sky-loaded MLX 6-bit and 4-bit files) and draft lengths 1–16: speculative tokens == plain tokens, byte for byte | The case mlx-lm fails, in public |
| **Self-draft** | Drafter = target, with the same kernels at M = 1. Acceptance must be exactly 100% at T = 0, and at any T under keyed Gumbel. Any rejection is a batch-invariance bug | Needs no reference text; detects a single flipped bit anywhere in the verify path |
| **Tie injection** | A copy of the model with two LM-head rows made identical, so exact logit ties are guaranteed. Speculative and plain output must agree, and the tie must go to the lowest id | Exercises the tie-break rule that #1470 tripped over |
| **Adversarial drafter** | A drafter that proposes the target's runner-up token at every near-tie | Maximizes the chance of exposing any difference between verify and decode |
| **Temperatures and filters** | T ∈ {0, 0.6, 1.0}; top-k 20 with top-p 0.95; min-p; repetition penalty | The operating point is sampling (§6) |
| **Tree shapes** | Chains and trees of up to the plan's node budget, including prefixes that end one token before a split boundary | Exercises logical-position split membership (§3.5) |
| **Long context** | Prompts of 4K and 32K tokens | Attention dominates there (§2.3) |
| **Hybrid models** | Qwen3.6-27B with MTP drafts | DeltaNet commit (§7.3) |
| **Kernel level** | For M = 1…16: each verify-kernel logits row is bit-equal to the M = 1 decode row on random and real inputs | Localizes failures to a kernel |
| **Distribution** | Rejection mode: chi-square of 10⁶ sampled tokens against the target distribution on fixed contexts | The non-byte-identical mode is still exact in distribution |

A single failure blocks the release. The suite's results are published with `sky-evaluation` ([Part XXIII](part23-benchmarks.md)).

## 10. Modeled speedups

The modeled table for Qwen3-32B on the M5 Max is in [Part IV](part04-hardware-limits.md) §8, and we use its numbers unchanged: base 28.0 tok/s (95% of the 29.5 tok/s planning ceiling), c = 0.03. As speedup factors over plain:

| Drafter (published τ) | γ | β = 0.02 | β = 0.05 | β = 0.15 (≈ MLX on M5 Max) | β = 0.48 (MLX `qmv` on M2-class; ≈ 0.5–1 end to end) |
|---|---:|---:|---:|---:|---:|
| EAGLE-3, τ = 2.29 (HumanEval, T = 0.6) | 3 | 1.99× (55.8) | 1.85× (51.7) | 1.49× (41.6) | 0.90× (25.3) |
| EAGLE-3, τ = 2.49 (GSM8K) | 3 | 2.16× (60.6) | 2.01× (56.2) | 1.62× (45.3) | 0.99× (27.6) |
| EAGLE-3, τ = 3.08 (GSM8K, 7 drafts) | 7 | 2.28× (63.9) | 1.98× (55.3) | 1.36× (38.2) | 0.68× (18.9) |
| DFlash-class, τ = 4.5 (none public for 32B) | 7 | 3.33× (93.3) | 2.89× (80.8) | 1.99× (55.8) | 0.99× (27.6) |

Tok/s in parentheses. `[EXPERIMENTAL: modeled; τ values VERIFIED from the RedHat card (R8 §2.3)]`

Reading:
- With Sky's target verify cost (β ≤ 0.05) and the public EAGLE-3 drafter: **1.7–2.2× plain, about 47–61 tok/s**, depending on workload. This is SPINE §3's "≥ 1.7× plain: 47–59 tok/s typical"; the top of the range is predictable work such as GSM8K, where the spine claims ≥ 60.
- At MLX-like verify costs on the M5 Max (β ≈ 0.15): 1.4–1.6×.
- At MLX-on-M2 costs (β ≈ 0.48 for the kernel, ≈ 0.5–1 end to end): no gain. That is what R4 measured on the M2 Pro: 0.37–0.74× (R4 §7.D).
- With a 0.6B draft model at c = 0.10 instead of 0.03 (a latency-bound drafter): τ = 2.29 → 1.68× (β = 0.02), τ = 2.49 → 1.83× (R8 App. B). The draft's step time matters as much as β.

**Against mlx-lm's own speculative mode, with the same 0.6B draft** (SPINE target: ≥ 1.2×). Same τ, γ = 3:

| | c | β | S |
|---|---:|---:|---:|
| mlx-lm on the M5 Max | 0.05–0.10 (CPU-bound draft, R8 §2.5) | ≈ 0.10 (mlx#3791 interpolated at M = 4) | τ / 1.45–1.60 |
| Sky | 0.03 | 0.02–0.05 | τ / 1.15–1.24 |

That is about 1.2–1.35× at T = 0, where both engines accept the same tokens. At T > 0, rejection sampling or coupling accepts more than mlx-lm's Σ p·q (§6). `[EXPERIMENTAL]`

The verify kernel is worth about 1.2× of this. The rest is drafter cost, the acceptance rule and no per-round synchronization. Separately from speed, Sky's output is byte-identical to its plain output, and mlx-lm's is not.

## 11. What the competition achieves

| Engine | Chip, models | Result | Source |
|---|---|---|---|
| mlx-lm | M2 Pro, Qwen3-8B + 0.6B draft, greedy | **0.37–0.74×** at every draft length (2, 4, 6); 3 of 9 configurations diverged from plain greedy | R4 §7.D `[VERIFIED]` |
| mlx-lm | M3 Max, Qwen2.5-32B-4bit + 0.5B-8bit draft, 4 drafts | 19.68 → 35.74 tok/s (**1.82×**); "The outputs are identical" on that prompt | mlx-examples #1155 (awni) `[VERIFIED]` |
| mlx-lm (main) | M5 Max, 70B 4-bit + 0.6B draft, k = 3 chain | ≈ 18 → 24.5 tok/s (≈ 1.36×) | mlx#3791 `[VERIFIED statement; contributor data]` |
| LM Studio (MLX) | M3 Pro, Qwen2.5-32B + 0.5B, code prompt | 7.30 → 17.74 tok/s (2.43×) | R8 §2.5 `[VERIFIED blog]` |
| LM Studio (MLX) | M1 Pro, Llama-3.1-8B and Qwen2.5-7B with draft models | 0.61–0.89× | lmstudio-ai/mlx-engine #103 via R8 §2.2 `[VERIFIED]` |
| MLX EAGLE-3 prototype | M3 Ultra, Llama-3.1-8B | 1.05×; no trees (single RoPE offset) | mlx-lm discussion #890 via R8 §2.2 `[VERIFIED]` |
| llama.cpp | M2 Pro, Qwen3-8B Q4_0 + 0.6B, `--spec-draft-n-max 3` | **1.12×** at 79.8% acceptance; with a ≈ 1.0× verifier the same acceptance would give ≈ 2.3× | R5 §8.C, §8.G `[VERIFIED]` |
| llama.cpp DFlash | M5 Pro, Qwen3.8-27B Q4_K_M | 10.42 → 19.31 tok/s (1.85×); an M2 Max user saw 0.97× at 76% acceptance | PR #27342 via R5 §9.4 `[author's run]` |
| llama.cpp MTP | various | M1 Ultra 27B Q8_0 17 → 25; M4 Pro 0.69–0.84×; M3 Max 0.67× | PR #22673 via R5 §9.4 |
| MTPLX (MLX-based) | M5 Max, Qwen3.6-27B 4-bit with MTP | 30.37 → 81.74 tok/s "optimized" (2.69×); Qwen3.8-27B "65.2 tok/s bare speed at official Qwen sampling" | README, R8 §2.7 `[UNVERIFIED: self-reported]` |
| MTPLX, independent test | M3 Max, Qwen3.6-27B, MTP depth 3 | +24–28% net over mlx-lm; acceptance drops on creative text | rotecodefraktion.de via R8 §2.7 `[VERIFIED secondary]` |

The pattern is consistent: speculation on Apple GPUs pays in proportion to how cheap verification is. The M3 Max (ALU-only) and M5 Max results for the same MTPLX software differ by about 2×, which is what §2.2 predicts from the M5's neural accelerators. `[LIKELY]`

Sky must beat:
- mlx-lm's speculative mode with the same draft (≥ 1.2×, §10);
- MTPLX on the hybrid models, once its claims are measured by us.

## 12. A DFlash-class drafter for Qwen3-32B (gate G9)

A 60 tok/s **average** needs τ ≥ 3.5–4.5 at T = 0.6 (R8 §2.9; §10: τ = 4.5 gives 81–93 tok/s at β ≤ 0.05). No public Qwen3-32B drafter reaches that. Block drafters do, on other targets:
- DFlash on Qwen3-8B: τ = 5.88 on HumanEval at T = 0;
- DSpark on Qwen3.8-27B: τ = 4.42 on HumanEval and 5.74 on math with 8 tokens (R8 §2.3).

Plan (gate G9; SPINE §6):
1. **Training framework:** Red Hat `speculators` or SGLang SpecForge. Both are open and both produced the published Qwen3 drafters (R8 §2.9).
2. **Data:** Qwen3-32B's own outputs at T = 0.6, in thinking and non-thinking modes, over a chat, code and math mix. The drafter must learn the target's sampled distribution, not a dataset's text.
3. **Shape:** DFlash-style, about 5 layers with a 16-token block, auxiliary hidden states from about 5 target layers. Its verify width of 16 fits the M5 neural-accelerator tier (§3.3), not the ALU tier.
4. **Compute budget:** unknown. The Red Hat model cards do not state training cost (gap). It needs data-center GPUs, outside Sky's Apple-only scope. `[EXPERIMENTAL: estimate after a pilot on the 8B recipe]`
5. **Acceptance target:** τ ≥ 4.0 at T = 0.6 averaged over the R9 workload mix, measured by XXI-T15 before any runtime work.

Exactness does not depend on the drafter (§6), so a better drafter is purely a speed upgrade with no new correctness risk. Owning the only strong Qwen3-32B drafter for Apple is also a moat item ([Part XXVII](part27-moat.md)).

## 13. The MoE caveat

For a dense model, verifying k ≤ k\* tokens reads the same weight bytes as verifying one. For a mixture-of-experts model, verifying k tokens reads the **union** of the experts they route to. With independent routing, an upper bound because real routing is correlated (R8 §7.3, `[EXPERIMENTAL arithmetic]`):

| Model | Experts, top-k | Distinct experts at k = 1 / 2 / 4 / 8 | Bytes at k = 4 / k = 8, relative to k = 1 |
|---|---|---|---|
| Qwen3-30B-A3B | 128, top-8 | 8 / 15.5 / 29.1 / 51.6 | 2.57× / 4.25× |
| gpt-oss-120b | 128, top-4 | 4 / 7.9 / 15.3 / 28.7 | 2.97× / 5.31× |
| gpt-oss-20b | 32, top-4 | 4 / 7.5 / 13.2 / 21.0 | 2.53× / 3.82× |

So on batch-1 MoE, β is not small: it grows with the expert union. mlx-lm users report speculation at −25% to −45% on a 397B-A17B MoE target with a 9B draft (mlx-lm #1132; R4 §9).

Rules for MoE plans:
- small γ (1–3);
- an expert-deduplicated verify kernel that loads each distinct expert once per pass;
- draft selection weighted by expected marginal expert cost (EcoSpec, arXiv 2607.12696: up to 1.62× on large MoE models; R8 §7.3);
- the tree sizer uses the measured union curve instead of dense β.

MTPLX claims 125.8 tok/s for an MoE model with MTP depth 3 on the M5 Max (UNVERIFIED). We treat MoE speculation as EXPERIMENTAL until measured.

## 14. The hybrid models: Qwen3.6/3.8-27B with MTP

Why they matter (R8 §1.7, `[VERIFIED: configs and safetensors headers]`):
- 64 layers in a 3:1 pattern of Gated-DeltaNet and gated full attention;
- about 25.6 B parameters read per token, 14.4 GB at 4-bit g64: 20% fewer bytes than Qwen3-32B, for a 42.6 tok/s roofline at 614 GB/s;
- 4× less KV per token, plus a fixed ≈ 151 MB recurrent state;
- an MTP head of 0.425 B parameters that reuses the target's embedding and LM head. One MTP draft token costs ≈ 0.95 GB of reads, ≈ 6.6% of a target step, so c ≈ 0.066.

Modeled for the M5 Max, with plain at 95% of the planning bandwidth (≈ 35.2 tok/s) and τ = 2.53 at γ = 3 (the UNVERIFIED MTPLX per-depth acceptance):
- β = 0.03 gives S ≈ 1.96, ≈ 69 tok/s;
- β = 0.05 gives S ≈ 1.88, ≈ 66 tok/s.

`[EXPERIMENTAL]` MTPLX claims 65–88 tok/s on this model and chip (UNVERIFIED).

Sky's requirements, in order:
1. Gated-DeltaNet layers in TIR with a per-position recurrent kernel and a chunked prefill kernel ([Part VII](part07-tensor-ir.md), [Part XV](part15-attention.md)).
2. The MTP head as a drafter.
3. Checkpoint-free commit of the recurrent and convolution state (§7.3).
4. The exactness suite on this model (§9).

mlx-lm cannot do this today: it drops MTP weights at load, and its speculation needs trimmable caches (R4 §5.1).

## 15. The honest version of "60 tok/s"

| Claim | Verdict | Why |
|---|---|---|
| 60 tok/s **plain** decode, Qwen3-32B 4-bit, M5 Max | **Impossible** | It needs 1.09 TB/s. The ceiling is 33.9 tok/s at the 614 GB/s spec ([Part IV](part04-hardware-limits.md) §1). Plain 60 is M5 Ultra territory: 59.6 tok/s at that chip's planning bandwidth (`roofline.py`) |
| 60 tok/s plain with fewer bits | Not at the same quality | It would need ≤ 2.56 bits per weight. Qwen3-32B collapses under 3-bit post-training quantization (AWQ W3 g128: MMLU 81.2 → 54.9; R8 §4.2) |
| 60 tok/s **effective** with exact speculation, public EAGLE-3 drafter | **On predictable or input-grounded work, if β ≤ 0.02–0.05** | GSM8K-like τ = 2.49 gives 56–61 tok/s; 3.08 at 7 drafts gives 55–64 (§10). Prompt lookup adds 2–4× on input-grounded tasks (R8 §2.6) |
| 60 tok/s **average** across chat, code and math at T = 0.6 | **Needs a better drafter** | Public EAGLE-3 gives 47–59 typical. τ ≥ 3.5–4.5 needs a DFlash-class drafter for Qwen3-32B: G9 (§12) |
| 60+ tok/s on a "32B-class" model in 2026 | **Most credible on Qwen3.6/3.8-27B with MTP** | 66–69 tok/s modeled (§14); MTPLX claims 65–88 (UNVERIFIED) |

What Sky claims, and how it is stated: effective tok/s = emitted tokens ÷ wall time, always with:
- the workload, the temperature and sampler, and thinking mode on or off;
- the context length;
- the drafter;
- the acceptance rule;
- the statement that the output is byte-identical to Sky's plain output (keyed Gumbel) or distribution-identical (rejection mode).

## 16. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XXI-T1 | ALU verify GEMV, M = 1…8, one `reduction_schedule` across instantiations (with XI-T11) | `crates/sky-kernel/src/skeletons/qgemv.rs` (`token_rows`), `crates/sky-kernel/src/verify/schedule.rs` (attribute and verifier check) | Batch-invariant verify family | Rows bit-equal to the M = 1 kernel for M = 2…8 on random and Qwen3 weights (M2 Pro and M5 Max). Cost on Qwen3-8B/32B shapes: ≤ 1.15× at M = 4; ≤ 1.3× at M = 8 (SPINE target; §2.3 risk) |
| XXI-T2 | Neural-accelerator verify GEMM, M = 9…32 (M5), plus the M = 1 streaming test (with XI-T12's `verify_mpp` probe) | `crates/sky-kernel/src/skeletons/mpp_qgemm.rs` (verify widths) | Second tier, and the decision of §3.4 | Cost at M = 16 ≤ 1.3× of the M = 1 GEMV. Record whether `mpp_qgemm` at M = 1 streams at ≥ 95% of the raw read: this decides byte-identical wide trees |
| XXI-T3 | Verify attention: multi-query, GQA-packed, logical positions, tree bitmasks with per-node window gather, NA tiles at long context (with XV-T2, XV-T5) | `crates/sky-kernel/src/skeletons/flash_decode.rs`, `crates/sky-kernel/src/skeletons/flash_decode_na.rs` | Attention for M ≤ 32 | Per-row output bit-equal to decode attention at the same position (chains, trees, prefixes one token before a split boundary); cost at 32K context and q ≤ 8 ≤ 1.1× of q = 1 on the M5 Max NA path (Part XV §5.3) |
| XXI-T4 | Acceptance kernels: keyed Gumbel (Philox), rejection sampling with residual resampling, filters | `crates/sky-kernel/src/skeletons/accept.rs`, `crates/sky-llm/src/spec/accept.rs` | GPU accept step | Self-draft acceptance = 100% (keyed Gumbel, all T); rejection-mode chi-square passes on 10⁶ draws |
| XXI-T5 | Commit kernels: KV length update, tree-path KV compaction, draft-cache rewind | `crates/sky-llm/src/spec/commit.rs` | GPU rewind | Byte equality of the KV cache after n rounds of speculation and after plain decode of the same tokens |
| XXI-T6 | Draft-model drafting (Qwen3-0.6B / 1.7B) as a static plan, optional truncated draft vocabulary | `crates/sky-llm/src/spec/draft_model.rs` | First drafter (G6) | c ≤ 0.035 for 0.6B against a 32B target on the M5 Max, measured with GPU timestamps; exactness suite green |
| XXI-T7 | Prompt lookup and suffix-tree drafting on the CPU, with event-driven round submission | `crates/sky-llm/src/spec/lookup.rs`, `crates/sky-llm/src/spec/suffix.rs` | Model-free drafter | ≥ 2× effective tok/s on the summarization and code-editing workloads of [Part XXIII](part23-benchmarks.md), greedy; ≈ 1.0× (never < 0.97×) on fresh text |
| XXI-T8 | EAGLE-3 runtime and converter for the RedHat and AngelSlim checkpoints | `crates/sky-llm/src/spec/eagle3.rs`, `sky-quant` (drafter quantization) | EAGLE-3 chains for Qwen3-32B and 8B | τ within 5% of the RedHat card's numbers at the card's settings (bf16 drafter); τ loss of the 4-bit drafter reported; ≥ 1.7× plain on the M5 Max at β ≤ 0.05 (G6) |
| XXI-T9 | Exactness CI suite (§9) | `crates/sky-llm/tests/spec_exactness.rs`, `.github/workflows/metal.yml` | Release gate | All rows of §9 green on the M2 Pro runner per PR and on the M5 Max nightly |
| XXI-T10 | Tree drafting and the tree sizer from measured β(M, context) | `crates/sky-llm/src/spec/tree.rs`, `sky-tune` (model-level entry) | Trees sized per chip and context bucket | On the M5 Max, the tree plan beats the best chain by ≥ 10% on the R9 workloads, or trees are disabled for that chip |
| XXI-T11 | Pipelined rounds: K rounds in flight, one command buffer per round, stop flag, ring buffer | `crates/sky-llm/src/spec/run.rs`, [Part XX](part20-persistent-execution.md) | No CPU in the round dependency chain | GPU timeline shows < 5 µs between rounds (except CPU-drafter rounds); ≥ 1.2× mlx-lm's speculative mode with the same 0.6B draft on the M5 Max (SPINE §3) |
| XXI-T12 | DFlash runtime for Qwen3-8B; G9 training of a DFlash-class Qwen3-32B drafter | `crates/sky-llm/src/spec/dflash.rs`; external training repository | Block drafting; then the 32B drafter | 8B: τ within 5% of the card; on the M5 Max ≥ 2.5× plain on HumanEval-style prompts. 32B drafter: τ ≥ 4.0 at T = 0.6 on the workload mix (§12) |
| XXI-T13 | MTP drafting and DeltaNet commit for Qwen3.6/3.8-27B | `crates/sky-llm/src/arch/qwen3_5.rs`, `crates/sky-llm/src/spec/mtp.rs` | Hybrid-model speculation | Exactness suite green on the hybrid; commit cost ≤ 3% of a round; effective tok/s reported against MTPLX on the same machine |
| XXI-T14 | Expert-deduplicated verify for MoE | `sky-kernel` (gather GEMV), `crates/sky-llm/src/spec/moe.rs` | MoE speculation that never loses | On Qwen3-30B-A3B: speculative ≥ plain at every tested γ, or speculation off by default for that model |
| XXI-T15 | Acceptance study | `sky-evaluation` | τ per drafter, workload, temperature, thinking mode and acceptance rule | Published table that replaces every assumed τ in this part |

## 17. Measurements that decide

1. **The β(M) curve on the M5 Max** for XXI-T1 and XXI-T2 at Qwen3-32B shapes, M = 1…32, contexts of 512, 8K and 32K. It decides the chain length, the tree budget and whether the neural-accelerator tier exists.
2. **Whether the neural-accelerator GEMM streams at ≥ 95% at M = 1** (XXI-T2). It decides whether wide trees can be byte-identical (§3.4).
3. **τ at T = 0.6 and 0.7, thinking on and off** (XXI-T15), for EAGLE-3 (bf16 and 4-bit), Qwen3-0.6B / 1.7B and prompt lookup, under each acceptance rule. Every modeled number in §10 depends on it.
4. **The 0.6B draft step on the M5 Max** (c), for Sky and for mlx-lm.
5. **mlx-lm's speculative mode on the M5 Max with the same draft,** measured by us. It is the baseline of the ≥ 1.2× target; today we rely on one contributor's 70B data (mlx#3791).
6. **DeltaNet commit cost and MTP acceptance** on Qwen3.6-27B (XXI-T13).

## 18. Risks and negative results

- **The neural-accelerator tier may not materialize.** The NA rates are secondary-source numbers, and `matmul2d` efficiency at 16–32-row tiles is unmeasured. If it fails, the M5 behaves like an ALU-only chip for verification: k\* ≈ 5–8, chains of 3–5, and τ ≈ 2.3–2.6 with EAGLE-3. The 1.7× target still holds at β ≤ 0.05 for chains (§10); the trees and DFlash do not.
- **The ALU tier may miss ≤ 1.3× at M = 8** (§2.3). The chains that matter (γ = 3–5) need only M = 4–6.
- **Acceptance at Qwen's recommended sampling is modest.** τ = 2.2–3.1 is far below paper headlines measured greedy on an H100. Thinking-mode τ is unmeasured. A workload with low acceptance gets ≈ 1.2–1.4× (R8 §9).
- **Long context erodes the gain.** Verify attention becomes compute-bound past about 4 queries per KV group on the ALUs (§2.3). At 32K context a chain of 7 would cost 1.5× per round unless attention runs on the neural accelerators.
- **MoE:** verification is not free (§13).
- **The competition moves.** mlx-lm already has β ≈ 0.06–0.18 on the M5 Max and an open MTP pull request; llama.cpp supports every drafter type; MTPLX claims large numbers on the hybrids. Sky's durable advantages are exactness, drafter breadth and owning the verify-path routing that MLX's maintainers declined (§3.3). Raw kernel speed alone is about 1.2×.
- **Negative results carried forward:**
  - mlx-lm's speculation is slower than plain decode on the M2 Pro at every setting (0.37–0.74×; R4 §7.D);
  - llama.cpp's speculation gains 1.12× at best there (R5 §8.C);
  - an MLX EAGLE-3 prototype gained 1.05× (R8 §2.2);
  - n-gram drafting finds nothing on fresh text (R5 §8.C);
  - Medusa-2's typical acceptance changes the output distribution and is excluded (R8 §2.6).

## 19. Disagreement with the spine

1. **Neural-accelerator verification and byte-identity are coupled** (§3.4). SPINE D10 sizes trees to k\* ≈ 32 on the M5's neural accelerators, and promises byte-identical speculative output. Both hold together only if one kernel family serves plain decode (M = 1) and verification (M up to 32). The neural-accelerator lowering is opaque (R6 §A2), so an ALU decode GEMV and an NA verify GEMM cannot be made to agree bitwise. We keep byte-identity as the default and limit it to the ALU tier unless XXI-T2 shows the NA family can also decode at ≥ 95% of the raw read. Wide NA trees otherwise run in a labeled, distribution-exact mode. [Part XV](part15-attention.md) (AT-D4) adopts the same rule for attention, so the remaining gap is with SPINE D10's wording only.

   Resolved: the spine now says byte-identity requires one kernel family for decode and verify: if verification runs wide on the neural accelerators, decode must use the same arithmetic path, or byte-identity holds only for ALU-width verification (M ≤ ~8), decided by measuring the accelerator GEMM at M = 1 (SPINE D10).
2. **"M ≤ 8 at ≤ 1.3×" for the weight GEMVs on the ALUs is at the arithmetic edge** (§2.3). We keep the target and add a fallback: ≤ 1.15× at M = 4 on all chips, with M = 5–8 on the neural-accelerator tier where present.
3. **The realistic free width on the M5 Max neural accelerators is about 16–24 tokens, not 33–35.** At M = 32 the pass needs about 94% of NA peak to stay at bandwidth time (§3.3). At long context it is smaller still (§2.3). The tree sizer uses measured β(M, context), never the ridge number.

   Resolved: the spine now sizes trees to the measured free width, realistically 16–24 tokens on the M5 neural accelerators against the theoretical ≈ 33–35 of `roofline.py` (SPINE D10).
