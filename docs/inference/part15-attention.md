# Part XV — Attention architecture

This part specifies Sky's attention kernels:
- decode attention, GQA-aware and split over the context with a fixed split size;
- verify attention for 2–32 queries, chains and trees;
- prefill flash attention on `simdgroup_matrix` (M1–M4) and on Metal 4 tensor ops and the neural accelerators (M5);
- the numerics of −∞ masks and online softmax;
- the bandwidth physics of long context, 32K–128K;
- sliding windows and sinks;
- the kernels' tuning spaces;
- what to copy from prior art;
- the exact rules for batch and chunk invariance.

The KV cache these kernels read is in [Part XIV](part14-kv-cache.md). The decode and verify GEMVs that surround attention are in [Part XVI](part16-decode.md), prefill GEMMs in [Part XVII](part17-prefill.md), speculation in [Part XXI](part21-speculative.md), and tuning in [Part X](part10-autotuner.md).

Evidence labels follow the report's convention. Research notes are cited as (R3 §7) and live in [`research/`](research/). Hardware rates come from [`roofline.py`](roofline.py), current version:
- M5 Max 40-core: 614 GB/s spec, 534 GB/s planning-achievable; neural accelerators (NA) ≈ 65 TFLOPS fp16; shader ALUs ≈ 16.6 TFLOPS. All three are LIKELY, from R6.
- M2 Pro: 196 GB/s measured read; 5.73 TFLOPS ALU.

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| AT-D1 | **One decode/verify kernel family** (`flash_decode`). It reads each KV head once and serves all of that head's query rows (G heads × q queries). | MLX re-reads K/V once per query head below 8K context, and its attention ran at 53% of achievable bandwidth at 4K (R4 §7.A). |
| AT-D2 | The context is cut into **splits of fixed size S**, aligned to absolute positions, with a **sequential, fixed-order combine**. S is a plan constant, never a function of the context length. | Batch, verify and chunk invariance (SPINE D6; R8 §3.3). MLX and llama.cpp fix the *number* of splits, so the split size changes with the context. |
| AT-D3 | Verify widths up to 32 queries use the same kernel. Chains use a causal window mask. Trees use ancestor bitmasks plus a per-node gather of the window block. | Exact speculation at any tree shape. MLX's fused path stops at 4 queries for GQA 8 (R3-kernels §c). |
| AT-D4 | On M5, verify attention at long context runs on the **neural accelerators**. Decode uses the same arithmetic path as verify. Whether that path is NA or ALU on M5 is decided by one measurement (§13). | At GQA 8, attention does 8 FLOP per KV byte per verified query. Beyond about 4 queries that exceeds the ALU ridge (≈ 31 FLOP/B), but stays under the NA ridge (≈ 122 FLOP/B) up to about 15 queries (§2). |
| AT-D5 | Prefill is **FA2-style tiles** with causal block skipping and an fp32 online softmax, GQA-grouped tiles, on `simdgroup_matrix` or MPP `matmul2d` (M1–M4) and NA `matmul2d` (M5). | Standard; MLX already has an NA prefill path, so the M5 target is parity (§6). |
| AT-D6 | −∞ is legal inside kernels, under explicit NaN rules (§7). Output finiteness is a validation mode. | SPINE D6. |
| AT-D7 | Kernel parameters are split into **numerics-class** parameters (fixed per plan; they define the bits) and **free** parameters (tuned at will; they never change bits). | sky-tune's rule "tuning may change speed, never results" (R1 §5.7) then still holds for the free parameters. |

## 2. The physics of attention on Apple GPUs

### 2.1 Bytes and FLOPs per decode token

Qwen3-32B has 64 layers, 64 query heads, 8 KV heads and head_dim 128 (G = 8). Per token of context, attention reads 262,144 B of KV and does 4 · 64 · 128 · 64 = 2.10 MFLOP for one query (R8 §1.4). At the planning bandwidth (`roofline.py`):

| Context | KV read per token | Attention GFLOP (1 query) | KV read time at 534 GB/s | Weight read time |
|---|---|---|---|---|
| 1,024 | 0.27 GB | 2.1 | 0.50 ms | 33.7 ms |
| 4,096 | 1.07 GB | 8.6 | 2.01 ms | 33.7 ms |
| 8,192 | 2.15 GB | 17.2 | 4.02 ms | 33.7 ms |
| 32,768 | 8.59 GB | 68.7 | 16.08 ms | 33.7 ms |
| 131,072 | 34.36 GB | 274.9 | 64.32 ms | 33.7 ms |

### 2.2 Arithmetic intensity, and the free verify width of attention

Each K element is used in one multiply-add per query row (QKᵀ), and each V element in one multiply-add per query row (PV). A KV head serves G query heads, and each of q verified queries is a row. So, with s bytes per KV element:

```text
FLOP per KV byte  =  2 · G · q / s
Qwen3-32B, fp16 KV (G = 8, s = 2):   8 · q
Qwen3-8B,  fp16 KV (G = 4, s = 2):   4 · q
Qwen3-32B, int8 KV (s ≈ 1.016):     ≈ 16 · q
```

Attention stays bandwidth-bound while this is below the ridge R = peak FLOP/s ÷ bandwidth. The **free verify width of attention** is therefore q\* = R · s / (2G), the counterpart of R8's free width for weights, k\* = R · b / 16:

| Path | Ridge (FLOP/B) | q\*, Qwen3-32B fp16 | q\*, Qwen3-32B int8 | q\*, Qwen3-8B fp16 |
|---|---|---|---|---|
| M5 Max neural accelerators (65 TFLOPS / 534 GB/s) | 122 | **15.2** | 7.6 | 30 |
| M5 Max shader ALUs (16.6 TFLOPS / 534 GB/s) | 31 | **3.9** | 1.9 | 7.8 |
| M2 Pro shader ALUs (5.73 TFLOPS / 196 GB/s) | 29 | 3.7 | 1.8 | 7.3 |

`[LIKELY: arithmetic on roofline.py rates; the M5 rates are secondary measurements (R6 §A2)]`

Consequences:
1. **Decode (q = 1) is bandwidth-bound everywhere,** provided each KV byte is read once. MLX's long-context loss is a bytes problem, not a compute problem: its kernels fetch a KV head's data once per query head and rely on caches (§3.1).
2. **Verifying several queries at long context is compute-bound on shader ALUs.** The table below assumes the GEMVs stay flat (Part XVI's target) and attention runs at the better of bandwidth and 100% of peak compute. That is optimistic; real kernels reach less.

   | Model and path | Context | q = 2 | q = 4 | q = 8 | q = 16 |
   |---|---|---|---|---|---|
   | Qwen3-32B, M5 Max ALU | 4,096 | 1.00 | 1.00 | 1.06 | 1.18 |
   | Qwen3-32B, M5 Max ALU | 16,384 | 1.00 | 1.01 | 1.20 | 1.60 |
   | Qwen3-32B, M5 Max ALU | 32,768 | 1.00 | 1.01 | **1.34** | **2.01** |
   | Qwen3-32B, M5 Max NA | 32,768 | 1.00 | 1.00 | 1.00 | 1.02 |
   | Qwen3-8B, M2 Pro ALU | 32,768 | 1.00 | 1.00 | 1.05 | 1.63 |

   These are verify-step time ratios T(q)/T(1) from attention alone (`roofline.py` rates; computed with the formula above).
3. **On M5 the neural accelerators turn this cost into nothing up to q ≈ 15.** On M1–M4 the planner caps the verify width by context ([Part XXI](part21-speculative.md)), and int8 KV halves q\*. That is why the KV format and speculation compete ([Part XIV](part14-kv-cache.md) §12).
4. The softmax itself is small. G·q/512 scores per KV byte for head_dim 128, at roughly 8 ALU operations per score (scale, max, subtract, exp2, sum, convert), is ≈ q/8 operations per byte at G = 8. That stays well below the ALU budget `[LIKELY]`.

## 3. What MLX and llama.cpp do today

### 3.1 MLX (`mx.fast.scaled_dot_product_attention`)

Source: `mlx/backend/metal/scaled_dot_product_attention.cpp` and `kernels/sdpa_vector.h`, MLX 0840c42 (R3-kernels §c) `[VERIFIED]`.

- **Vector mode** applies iff the query length is ≤ 8. The fused kernel also needs `qL · gqa ≤ 32` (`:1001-1005`). So **Qwen3-32B (GQA 8) gets fused decode attention only for qL ≤ 4.** qL 5–8 falls back to an unfused composition: scale, matmul QKᵀ, mask, softmax, matmul V (`mlx/fast.cpp:819-870`).
- **1-pass kernel:** one 1024-thread threadgroup per (query head, query position). Simdgroup g walks keys g, g+32, …. **K/V for a KV head are fetched by each of its G query-head threadgroups** (`sdpa_vector.h:65-69`).
- **2-pass kernel:** on "s"-class devices (M2 Pro, and LIKELY M5 Max) from kL ≥ 1024. Block b walks keys b, b + blocks, …. The *number* of blocks comes from a device-letter table keyed by context bucket: for "s", 64, or 128 / 256 / 512 / 1024 blocks for N ≤ 8K / 32K / 64K / above when gqa·qL > 4 (`:774-812`). The split size therefore changes with the context.
- **GQA-deduplicating variant** `sdpa_vector_2pass_1_gqa`: only for qL = 1, no mask, no sinks, head dims equal, **kL ≥ 8192**, and G = 8 with D ∈ {64, 128} (or G ∈ {12, 16} at D = 128) (`:755-763`). For Qwen3-32B it reads K/V 2× instead of 8×. mlx#4077 (merged 2026-08-18) reports kernel speedups of 1.08–1.31× on M5 Max (R3-issues F3).
- **Verify window:** qL 9–48 goes to `attention_nax` with BQ = 64 and no split over the KV length. mlx#3826 measured on M5 Max (GQA 4, D = 128, 16K context): 0.813 ms at L = 8, 1.906 ms at L = 12, then flat to L = 48. It was closed as won't-fix: "likely not going to happen in production" (R3-kernels §d).
- **Prefill `steel_attention`** (non-NAX): BQ = 32 and BK = 16 for D ≥ 128, 4 simdgroups (128 threads), grid (⌈qL/32⌉, heads, batch), no split over KV. Q sits in threadgroup memory; K and V share one threadgroup buffer. The softmax uses exp2 with log₂e folded into the scale. Causal blocks past the diagonal are skipped. Each query-head threadgroup re-streams K/V.
- **Prefill `attention_nax`** (M5): BQ = 64, BK = 32, 4 simdgroups; Q, K and V go straight from device memory into NA registers; there is a head-dim split (`dsplit`) for D = 256/512.
- **Quantized KV:** no fused kernel. mlx-lm runs `quantized_matmul` → mask → softmax → `quantized_matmul`.

**Measured gaps:**

| Measurement | Result | Source |
|---|---|---|
| Qwen3-8B on M2 Pro (GQA 4), mlx-lm, context 128 → 4,096 | +6.04 ms for 585 MB more KV per token: the KV stream runs at **97 GB/s, 53% of achievable** | R4 §7.A `[VERIFIED]` |
| 8-layer Qwen3-8B-shaped stack, M2 Pro, 4K | attention at ≈ 56 GB/s effective, **≈ 3× below the 182 GB/s ceiling**; extrapolated to Qwen3-32B: ≈ +19 ms per token at 4K instead of ≈ 6 ms | R3 §7.F `[VERIFIED time; mechanism LIKELY]` |
| mlx#3837, M5 Max, 32 query / 8 KV heads, L = 32,768, D = 128 | 56.2% of peak bandwidth; "a fixed cost of roughly 0.3 ms at this context length" (before mlx#4077) | R3-issues F2 |
| mlx-lm#763, M3 Ultra, MiniMax-M2.1 4-bit, 146K context | 5.95 tok/s (MLX) vs 12.12 (llama.cpp flash attention) | R3-issues F1 |

### 3.2 llama.cpp (`flash_attn_ext`, `flash_attn_ext_vec`)

Source: `ggml/src/ggml-metal/ggml-metal-ops.cpp:2949-3830`, `kernels/fa_vec_common.metal`, `ggml-metal-tuning.cpp`, llama.cpp 81e39ad (R5 §4) `[VERIFIED]`.

- **The vec kernel** runs iff there are fewer than 20 query rows and the head size is a multiple of 32. It handles 1 query per threadgroup by default and 32 KV positions per simdgroup step.
- **Split-K ("flash decoding"):** always **32 workgroups** along the KV axis. Simdgroups per workgroup grow 1 → 2 → 4 with the KV length. Partial (O, m, l) go to a temporary buffer, and a second `vec_reduce` kernel combines them after a barrier. The number of splits is fixed, so the split size varies with the length.
- **Offline-tuned table:** (queries per threadgroup, elements per lane) per GPU family (Apple7/8/9/10), KV dtype, head dims, KV-length bucket (< 1K, < 4K, < 16K, ≥ 16K) and query-row bucket. For Qwen3 fp16 on M5 Max it is (Q = 1, NE = 4), and (Q = 2, NE = 2) at ≥ 16K (`ggml-metal-tuning.cpp:747+`). A Q = 2 verify variant (#23114) measured 0.84× on M2 Ultra, 1.47× on M4 Max and 1.75× on M5 Max (R5 §9.5).
- **The tiled prefill kernel:** 8 queries per threadgroup, 64 KV per simdgroup step. A pre-pass (`blk`) classifies mask blocks as skip, no-mask or partial. A pad kernel handles KV lengths that are not multiples of 64.
- **KV types:** F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 are dequantized in the kernel; for batches ≥ 32 a quantized KV is first dequantized to an F16 scratch. Sinks, ALiBi, softcap and MLA are supported.
- **History:** the split-K change (#15566) took Gemma 3 1B tg128 at depth 32,768 on an M2 Ultra from 98.65 to 172.59 tok/s.

### 3.3 Other Apple prior art

- **Metal FlashAttention** (Philip Turner): 83% of ALU on M1 Max for forward and backward. Its methods are register-centric tiling, a third blocking dimension along D, intentional spilling at large D, and blocks of "16-32 along the parallelization dimension and 80-128 along the traversal dimension". Parameter tables per head dim have separate rows for Apple9+ and earlier. It uses the undocumented `air.simdgroup_async_copy`, whose behavior changes across GPU generations (R7 §8.2–§8.3). `[VERIFIED: README and source]`
- **ThunderMittens** (Hazy Research): attention "within ±15% of MLX" on the M2 Pro, with 8×8 base tiles. Its finding: "shared memory isn't as crucial" (R7 §8.1–§8.2).

## 4. Decode attention: `flash_decode`

Skeleton: `crates/sky-kernel/src/skeletons/flash_decode.rs`. Combine kernel: `crates/sky-kernel/src/kernels/attn_combine.rs`.

### 4.1 The canonical reduction

For a query row at absolute position t (one query head of one query), the visible keys are [lo, t], where lo = 0 for full attention or t − W + 1 for a sliding window. The row's result is defined as:

```text
blocks   b = [b·B_k, (b+1)·B_k)        aligned to absolute positions; B_k ∈ {32, 64}
splits   j = [j·S,   (j+1)·S)          aligned to absolute positions; S a multiple of B_k
within split j (blocks in increasing order), fp32 online softmax:
    s     = (q · scale · log2 e) · K_bᵀ          dot product over head_dim in the fragment's fixed K-order
    s     = −∞ where the key is masked (future, outside window, not an ancestor, padding)
    m'    = max(m, rowmax(s));   p = exp2(s − m');   l = l·exp2(m − m') + rowsum(p)
    O     = O·exp2(m − m') + p · V_b
    (split 0 starts from the sink state if the layer has learned sinks, §9; otherwise m = −∞, l = 0, O = 0)
partials P_j = (m_j, l_j, O_j)                    fp32
combine  P = (((P_0 ⊕ P_1) ⊕ P_2) ⊕ … ⊕ P_J),  J = ⌊t / S⌋,   increasing j
         (m_a, l_a, O_a) ⊕ (m_b, l_b, O_b) = (m, l_a·2^(m_a−m) + l_b·2^(m_b−m), O_a·2^(m_a−m) + O_b·2^(m_b−m)),  m = max(m_a, m_b)
output   O / l
```

The result depends only on B_k, S, the fragment's dot-product order, the exp2 variant and the precision of p in the PV product. These are the **numerics-class parameters** (§10). It does not depend on the context length, the number of rows, the batch, the chunk or the thread mapping. §12 turns this into rules.

### 4.2 Mapping onto the GPU

- **Grid:** one unit of work is (KV head h, split j). Qwen3-32B has 8 × ⌈T/S⌉ units per layer.
- **A unit of work is one simdgroup.** It loads K and V for its split once (B_k positions at a time) and keeps all of the KV head's rows in matrix fragments. For decode those are G = 8 rows, one per query head; for verify, 8·q rows (§5). It writes one partial per row.
- **A threadgroup holds 1–8 such simdgroups**, each on a *different* split. The count is a free parameter: it changes occupancy, not arithmetic. Two simdgroups never split one split between them, because that would add a merge to the tree.
- **Arithmetic path:**
  - M1–M4: `simdgroup_matrix<float, 8, 8>` fragments with fp32 accumulation (MLX's steel kernels also use fp32 fragments, R3-kernels §b), or MPP `matmul2d`, which reached 95% of ALU peak for f16 on the M2 Pro (R6 §B5).
  - M5: NA `matmul2d`, or the same ALU path. One of the two is chosen per plan and used for every row count (§12 rule I4).
  - At decode, G = 8 rows fill exactly one 8-row fragment. An NA descriptor needs 16 or 32 rows, so the rows are padded. Padding wastes compute, not bandwidth: 32 padded rows is 32 FLOP/B, against an NA ridge of 122.
- **KV formats:** fp16/bf16 directly. int8 is converted to fp16 on load, with the per-token scales folded into scores and probabilities ([Part XIV](part14-kv-cache.md) §5.2).
- **Combine.** By default a separate dispatch merges each row's partials in increasing j. That is one dependent dispatch per layer: ≈ 1.1–1.25 µs of boundary on the M2 Pro (R6 §B3) plus the kernel's own ≈ 2.5 µs, about 0.2 ms per Qwen3-32B token over 64 layers ([Part XXII](part22-performance-model.md) §6.1). A free alternative lets the last unit of a KV head to finish (elected by an atomic counter behind a device-scope fence, a pattern R6 §B6 verified) do the merge. The merge order is still by j, not by arrival, so it is equally deterministic.

### 4.3 Choosing S: occupancy against partial traffic

Each partial is fp32 O (128) + m + l = 520 B per row. Units per layer and partial traffic (written and read back), relative to the KV bytes, for Qwen3-32B:

| S | Units per layer at 1K / 4K / 32K / 128K | Partial traffic, q = 1 (r+w) | Partial traffic, q = 8 (r+w) |
|---|---|---|---|
| 256 | 32 / 128 / 1,024 / 4,096 | 6.3% | 51% |
| 512 | 16 / 64 / 512 / 2,048 | 3.2% | 25% |
| 1,024 | 8 / 32 / 256 / 1,024 | 1.6% | 13% |
| 2,048 | 8 / 16 / 128 / 512 | 0.8% | 6.3% |

Partials of one layer are small at moderate contexts (2.1 MB at 32K with S = 512 and q = 1). They are read back immediately and LIKELY come from the system-level cache. On the M2 Pro, R6 found re-reads cache-resident only up to about 8 MiB (R6 §B2). The table's traffic is therefore an upper bound on DRAM cost.

S is a numerics-class parameter. It is chosen once per (model, chip) by measurement over the context range the plan supports, and then fixed. Candidates are {256, 512, 1024}; the starting point is 256, Part XXIV's example. Small S helps occupancy at short context (M5 Max has 40 cores; 1K context at S = 1024 gives only 8 units per layer). Large S helps verify traffic at long context.

**Variant** `[EXPERIMENTAL]`: merge split partials in an **aligned binary tree** instead of a left-to-right chain. The prefix of n splits is the left-to-right combination of the aligned power-of-two subtrees given by the binary digits of n. A threadgroup may then pre-merge any aligned group of 2^l splits without changing the result. That makes occupancy and partial traffic free parameters, at the cost of a slightly more complex combine. It is worth building only if the fixed-S tradeoff above costs more than 3% anywhere.

### 4.4 Targets

| Target | Machine | Baseline |
|---|---|---|
| Decode attention ≥ 85% of measured read bandwidth at 4K–32K, Qwen3-8B shapes | M2 Pro (G1–G4) | MLX at 53% (R4) |
| Decode attention ≥ 85% of measured read bandwidth at 8K–128K, Qwen3-32B | M5 Max (G5) | MLX 56% at 32K before mlx#4077 (R3-issues F2) |
| `decode_verify_invariance` passes (§12) | both | — |

## 5. Verify attention for 2–32 queries

### 5.1 Chains

A chain verify of γ drafts processes q = γ + 1 queries at positions p … p + γ. They are rows of the same `flash_decode` launch: G·q rows per KV head, in fragments of 8 rows, or 16/32 on the NA. Their K and V were appended before attention ([Part XIV](part14-kv-cache.md) §4.2).
- **Mask.** Query p + i sees keys ≤ p + i. Inside the window block, later drafts are masked to −∞; blocks wholly after the row's position are skipped. The row's split count is ⌊(p + i)/S⌋ + 1, exactly what decode at that position uses.
- **Cost.** Each KV byte is read once for all G·q rows. Compute grows with q, and §2.2 says where it stops being free.
- **Path.** On M5, G·q ≥ 32 rows is the NA's preferred tile (tzakharko: 32×32 or larger, R8 §2.2; Anemll: tiles of at least 64×64 for peak, R6 §A2). On M1–M4 the ALU path applies, and the planner caps q by context.

### 5.2 Trees

Tree verification (SpecInfer, Sequoia; R8 §2.8) puts n nodes in the window:
- **Positions.** Node k at depth d_k has position p + d_k. RoPE uses that position, not the slot index. The slot index is the node's order.
- **Mask.** Each query node carries a bitmask over the window: bit k′ is set iff k′ is an ancestor of k or k itself (≤ 64 bits per node for windows ≤ 64). Prefix blocks before the window are visible to every node.
- **Exactness.** In chain decode at position p + d, the window block holds the path's keys at consecutive slots. In a tree layout the same keys sit at scattered slots with masked siblings between them. Hardware fragments may sum in a pairwise order, so moving a nonzero term within a block can change rounding. **Rule I7:** the window block for node k is *gathered* through k's ancestor list in depth order, i.e. by logical position. Prefix blocks stay shared; only the window block (≤ 64 keys) is per node.
- **After acceptance**, the accepted path is compacted to slots p … p + a ([Part XIV](part14-kv-cache.md) §9), and later decode steps see it at its absolute positions.

mlx-lm cannot verify trees at all: its KV cache has a single RoPE offset (mlx-lm discussion #890, R8 §2.2).

### 5.3 Targets

| Target | Condition |
|---|---|
| Verify attention for q ≤ 8 costs ≤ 1.1× the q = 1 time | M5 Max, Qwen3-32B, 4K–32K, NA path |
| Verify attention for q ≤ 8 costs ≤ 1.1× the q = 1 time | M2 Pro, Qwen3-8B (G = 4), up to 32K |
| ≤ 1.3× at q = 4 | M1–M4, Qwen3-32B-class (G = 8), up to 16K |
| `tree_verify_exact` and `spec_equals_plain` pass (§12) | all chips |

## 6. Prefill flash attention: `flash_prefill`

Skeleton: `crates/sky-kernel/src/skeletons/flash_prefill.rs`.

### 6.1 Structure (FA2)

- **Tile.** A threadgroup takes a tile of query rows and walks the visible key blocks in increasing order. It keeps per-row (m, l, O) in fp32 and writes O/l once at the end. Rows are BQ positions of **all G heads of one KV group** (GQA grouping, a free parameter), so a K/V tile loaded once serves G·BQ rows. MLX instead re-streams K/V per query head (§3.1).
- **Causal block skipping.** Blocks entirely after the tile's last query are not visited. Blocks at the diagonal are masked. With BQ ≠ BK the diagonal spans several blocks; the mask is computed from absolute positions, never materialized.
- **Scale.** scale · log₂e is folded into Q once (f32), and the kernel uses exp2.
- **Chunked prefill.** A chunk of C new positions attends to the cached prefix plus itself. The kernel receives the chunk's absolute start position; key blocks stay aligned to absolute positions (rule I9).
- **Tiles by chip:**

| Chip | Matrix path | Starting tile | Notes |
|---|---|---|---|
| M1–M4 | `simdgroup_matrix` 8×8, fp32 accumulators | BQ = 32, BK = 16–32, 4 simdgroups (MLX steel) | Q in threadgroup memory; MLX's steel attention is single-buffered, so double-buffered K/V is an EXPERIMENTAL gain |
| M1–M4 | MPP `matmul2d` | 32×32 or 64×32 with 1–2 simdgroups (95% of ALU peak in R6's GEMM) | a single code path for all chips (R6 §B5) |
| M5 | NA `matmul2d` | BQ·G ≥ 64 rows, BK = 32–64 | operands straight from device memory into cooperative tensors (MLX `attention_nax`); 64×64 or larger tiles for peak (R6 §A2) |

- **Threadgroup memory is 32 KB on every family, including M5** (R6 §A4). A K and a V tile of 64 positions × 128 × 2 B fill it exactly (16 KB each). So double buffering needs BK = 32, or direct device-to-register loads.
- **Head dims:**
  - 128: Qwen3, Llama.
  - 64: gpt-oss.
  - 256: Gemma 3 4B and Qwen3.5+ full-attention layers. These need a split along D (MFA's third blocking dimension; MLX's `dsplit`).

### 6.2 How much prefill attention matters

Attention's share of prefill FLOPs for Qwen3-32B (`roofline.py` `prefill_flops_per_token`) is 0.9% at 512 tokens, 3.3% at 2K, 6.4% at 4K, 12.1% at 8K and 35.5% at 32K.

So prefill attention matters only for long prompts. MLX already has an NA attention kernel, and the M5 target is **parity at 2K–32K**. On M1–M4, MLX's single-buffered steel kernel may leave headroom `[EXPERIMENTAL: measure at G5]`.

INT8 attention (SageAttention-style; Draw Things' "Metal Quantized Attention" uses INT8 matmul on M5 Max, R3-issues A4) changes outputs. It is research, behind a labeled mode, like W8A8 prefill ([Part XII](part12-quantization.md) §9).

## 7. −∞, NaN and the `inference` contract

Scientific Sky treats every non-finite value as an error. Tensor programs may use −∞ inside kernels; output finiteness is a validation mode (SPINE D6). The exact rules:

| Rule | Statement |
|---|---|
| N1 | Masks enter as −∞ on scores, or as skipped blocks. A causal or window mask is computed from positions, never materialized as a [q, T] tensor. Tree masks are per-node bitmasks. |
| N2 | **Safe online softmax.** When the running max is still −∞ after a block (every key so far masked), the reference used for exponentials is 0: p = exp2(s − m_ref) with m_ref = (m′ = −∞ ? 0 : m′), and the rescale factor exp2(m − m_ref) is 0 when m = −∞. The kernel never evaluates −∞ − (−∞). |
| N3 | A partial with m = −∞ (all keys masked) never enters the combine. A row's splits beyond its position are skipped (I6). |
| N4 | The final division O / l is guarded. l = 0 can occur only for padding rows, which output 0 by selection, not by dividing. |
| N5 | Every V the kernel can touch is finite: slabs are zero-initialized and never receive non-finite values ([Part XIV](part14-kv-cache.md) §3.1). A masked probability is exactly 0, so masked keys contribute exactly 0. |
| N6 | The exp2 variant (fast or precise) and the precision of p in the PV product are fixed per plan and are part of the numerics key. |
| N7 | `--validate=finite` checks attention outputs and hidden states at layer boundaries, and reports the first non-finite value with its layer, head and position. It is never a per-operation cost in production plans. |
| N8 | Logit soft-capping (Gemma 2/4: c·tanh(s/c)) and sinks (§9) are applied in fp32 before the max. |

## 8. Long context: 32K to 128K

### 8.1 Bandwidth

KV dominates the bytes at long context ([Part XIV](part14-kv-cache.md) §2.3): it is 32% of the bytes per token at 32K and 66% at 128K for Qwen3-32B in fp16. Decode ceilings (`roofline.py`, spec / achievable):

| Context | fp16 KV | int8 KV (labeled mode) |
|---|---|---|
| 32,768 | 23.1 / 20.1 tok/s | 27.5 / 23.9 |
| 65,536 | 17.5 / 15.2 | 23.0 / 20.0 |
| 131,072 | 11.7 / 10.2 | 17.3 / 15.1 |

128K needs YaRN ×4 for Qwen3-32B, enabled identically in every engine compared (R9 §3.1). With fp16 KV it needs 53 GB of memory ([Part XIV](part14-kv-cache.md) §2.2).

### 8.2 What better attention can win against MLX

Assume both engines stream weights at the same rate. MLX's measured 28.4 tok/s on M5 Max implies ≈ 515 GB/s (R5 §13.1). Sky's attention runs at 90% of achievable (a target). MLX's attention runs at efficiency e. Then the speedup from attention alone, for Qwen3-32B, is:

| Context | KV share | e = 0.31 (R3, 4K) | e = 0.53 (R4, 4K) | e = 0.75 (assumed, after mlx#4077) | e at which Sky reaches 1.2× | Sky int8 vs MLX fp16 at e = 0.53 |
|---|---|---|---|---|---|---|
| 4,096 | 5.6% | 1.11 | 1.04 | 1.01 | — | 1.07 |
| 8,192 | 10.7% | 1.22 | 1.08 | 1.02 | 0.33 | 1.14 |
| 16,384 | 19.3% | 1.39 | 1.14 | 1.04 | 0.45 | 1.27 |
| 32,768 | 32.3% | 1.64 | **1.24** | 1.07 | 0.57 | 1.48 |
| 65,536 | 48.8% | 1.96 | 1.35 | 1.10 | — | 1.80 |
| 131,072 | 65.6% | 2.28 | **1.47** | 1.13 | — | 2.19 |

`[EXPERIMENTAL: e for MLX on M5 Max at Qwen3-32B shapes is not measured; the e values are the measurements and assumption named in each column header]`

Reading:
- The gain grows with context because KV's share grows.
- At 8K, even a 2× attention advantage is worth only ≈ 1.08×.
- These bounds are attention alone. [Part XXII](part22-performance-model.md) §6.3's modeled 1.10–1.14× at 8K and 1.24–1.27× at 32K compare whole plans and add Sky's dispatch and fill savings on top.
- The fused int8 KV mode adds the byte reduction on top.
- This bounds the long-context target; see §13.

## 9. Sliding windows and sinks

- **Window.** Row t sees keys j with t − j < W (mlx-lm `base.py:35-37`, llama.cpp `llama-hparams.h:486-491`; R9-ARCH). Blocks entirely before t − W + 1 are skipped; the block containing t − W + 1 is masked. Splits are still aligned to absolute positions, so a window row's reduction is the canonical one restricted to its visible blocks.
- **Ring buffers.** A sliding layer stores K/V in a ring of capacity C, a multiple of B_k ([Part XIV](part14-kv-cache.md) §7.2). The kernel walks *logical* blocks in increasing order and maps logical block b to physical block b mod (C / B_k). Since C is a multiple of B_k and blocks are aligned, every logical block is one contiguous physical block.
- **Learned sinks** (gpt-oss): split 0 of every row starts from (m = sink_h·log₂e, l = 1, O = 0), i.e. a virtual key with V = 0, before the first real block. Later splits start empty. This is exactly the softmax MLX computes by concatenating the sink logit and slicing it off (`mlx/fast.cpp:867-885`), up to rounding order.
- **Per-layer patterns:**
  - Gemma 3: 5 local : 1 global; local window 1024 with RoPE θ 1e4; global layers θ 1e6 with linear ×8.
  - gpt-oss: alternating, window 128, sinks.
  
  W, sinks and soft-capping are per-layer function constants, so each pattern instantiates its own pipelines.

## 10. Kernel parameters and search spaces

| Parameter | Class | Candidates | Applies to |
|---|---|---|---|
| Key block B_k | **numerics** | 32, 64 | decode, verify, prefill |
| Split size S | **numerics** | 256, 512, 1024, 2048 | decode, verify |
| Matrix path | **numerics** | `simdgroup_matrix` 8×8 fp32; MPP `matmul2d` (descriptor M×N×K, `relaxed_precision` off or on); NA `matmul2d` | all |
| Precision of p in PV | **numerics** | fp32 (ALU fragments); fp16 (required for NA half operands) | all |
| exp2 variant | **numerics** | fast, precise | all |
| Combine order | **numerics** | sequential (default); aligned binary tree (EXPERIMENTAL, §4.3) | decode, verify |
| Splits per threadgroup | free | 1, 2, 4, 8 | decode, verify |
| Rows per fragment group | free | 8, 16, 32 (NA) | decode, verify |
| K/V staging | free | device → registers; threadgroup memory, single or double buffered | all |
| Load width per lane | free | 8 B, 16 B | all |
| Combine placement | free | separate dispatch; elected last threadgroup | decode, verify |
| BQ (query positions per tile) | free | 16, 32, 64, 128 | prefill |
| GQA grouping of the tile | free | on, off | prefill |
| Simdgroup arrangement WM × WN | free | 1×1 … 4×2 | prefill |
| D split | free | none; 2 (D = 256) | prefill |
| Threadgroup swizzle | free | 0–3 | prefill |

- **Free parameters** are searched by the autotuner (BEAM over skeleton parameters, [Part X](part10-autotuner.md)). They are keyed by (kernel template version, KV dtype, head_dim, G, context bucket, chip family, OS and Metal compiler version). The table follows llama.cpp's per-family offline table for its vec kernel, but tunes a fixed split *size* instead of a split *count*.
- **Numerics-class parameters** are chosen once per (model, chip) when a plan is built (`-O3` or overnight), recorded in the plan, and included in the numerics key. Changing them creates a new determinism domain, which SPINE D6 allows per tuning entry.
- **A free parameter's bit-neutrality is itself tested.** For every pair of candidate values, the same inputs must produce identical bits (§12). A parameter that fails this test becomes numerics-class.

## 11. Prior art and what we copy

| Source | Copy | Avoid |
|---|---|---|
| Flash-Decoding (pytorch.org/blog/flash-decoding; up to 8× at long sequences; 64K: 64.4 µs vs 2,300.6 µs for FA2 at B = 1) | split-KV with a log-sum-exp combine | a split count that depends on length |
| FlashAttention-2 | tiling, online softmax, causal skip | — |
| FlashAttention-3 | the ideas of overlap and low precision | its mechanisms (TMA, WGMMA, warp specialization) do not exist on Apple (R7 §9.1) |
| FlashDecoding++ (arXiv 2311.01282) | dataflow chosen per shape, i.e. the tuner | the "unified max" softmax: numerically risky, and it changes the reduction |
| FlashInfer (arXiv 2501.01005) | plan/run split: plan per context bucket, run a fixed command stream; templates over variants | — |
| LeanAttention (arXiv 2405.10480; 2.6× over FA2, 8.33× at 512K) | stream-K decomposition, equivalent to tuned split-KV | — |
| MLX `sdpa_vector_2pass_1_gqa` | one simdgroup set per KV head serving several query heads | its gates (qL = 1, no mask, ≥ 8K) and device-letter block counts |
| MLX `steel_attention` / `attention_nax` | exp2 with log₂e folded; causal `kb_lim`; NA operands from device memory | per-query-head K/V re-streaming |
| llama.cpp `flash_attn_ext_vec` / tiled | per-family offline parameter tables; mask-block pre-classification; the reduce kernel | a fixed number of workgroups (32) |
| Metal FlashAttention | register-centric tiling; D blocking; per-head-dim parameter tables per family | `air.simdgroup_async_copy` and other undocumented intrinsics |
| SGLang deterministic mode / Thinking Machines | fixed split *size*; same KV layout for prefill and decode; chunk alignment | — |

Licenses: MLX, llama.cpp and MFA are MIT, ThunderMittens is MIT and BaseRT is Apache-2.0. We learn from them and write our own kernels; any port follows ROADMAP's notice rules (SPINE D15).

## 12. Batch and chunk invariance

Definitions:
- **Canonical reduction** of a row: §4.1 for decode and verify; rule I9 for prefill.
- **Level L1** (required by the `inference` contract, SPINE D6):
  - **L1a** run-to-run bit reproducibility on the same chip, OS, Sky build and tuning entry;
  - **L1b** batch and verify invariance: a row's result does not depend on how many queries are verified with it, on the tree shape, on speculation, or on other sessions in the batch;
  - **L1c** chunk invariance: prefill results, including the K/V written, do not depend on the chunk size or on where prefix-cache boundaries fall.
- **Level L2** (not promised): prefill equals decode bit for bit for the same position. Prefill uses compute-optimal kernels; [Part XIV](part14-kv-cache.md) §10.3 states what follows for the prefix cache.

The rules:

| Rule | Statement | Enforced by |
|---|---|---|
| **I1** | Key blocks (B_k) and splits (S) are aligned to absolute positions in every program (decode, verify, prefill) and every KV layout (slab, ring, pages). Ring capacities and page sizes are multiples of B_k; S is a multiple of the page size. | plan verifier (`crates/sky-tir/src/verify/attn.rs`) |
| **I2** | B_k and S are fixed per plan. They are never derived from the context length, the number of queries, the batch, the chunk or the occupancy. | plan verifier: they are constants in the plan, not runtime arguments |
| **I3** | **Row independence.** A row's result depends only on its own query vector, its visible keys and the canonical reduction. Kernels may pack any rows together (query heads, verified queries, sessions). This requires matrix fragments whose per-row result does not depend on the other rows. | test `attn_row_independence`: random contents in the other rows; the probed row is bit-identical. It runs per path (simdgroup_matrix, MPP, NA) and chip family |
| **I4** | **One arithmetic path per program family.** Decode (q = 1) and verify (q ≤ 32) use the same fragment path and dot-product order, padding rows as needed. Switching paths by q (ALU for small q, NA for large q) is forbidden under `inference`. It is allowed under `relaxed`, which loses byte-identity. | plan verifier: one `AttnPath` per plan |
| **I5** | Masked keys contribute exact zeros: −∞ score, p = 0, finite V. Masked positions keep their place inside a block; blocks are never compacted. | N1–N5; slab zero-initialization |
| **I6** | A row at position t uses splits 0 … ⌊t/S⌋ and combines them in increasing order. Splits beyond t are skipped. All-masked partials never enter the combine. | kernel structure; test `decode_verify_invariance` |
| **I7** | Tree nodes gather their window block through the ancestor list in depth order (position-indexed). The content of the node's window block then equals the chain-decode block at that position. Prefix blocks are shared. | test `tree_verify_exact` |
| **I8** | K and V for a position are written once, by the program that first processes the position, before attention reads them. They are never recomputed by a different kernel. Rewind truncates the logical length ([Part XIV](part14-kv-cache.md) §4.2, §9). | test `kv_append_invariance`, `kv_rewind_exact` |
| **I9** | **Prefill's canonical reduction.** For row t: key blocks of size B_k^pre aligned to absolute positions, visited in increasing order from the window start, with the fp32 online softmax. It is independent of the chunk size, the query tile size (BQ) and the GQA grouping. A chunk may start in the middle of a block; that block is then partially masked, not shifted. | test `prefill_chunk_invariance`: chunk sizes 1, 7, 64, 512, 2048 and random prefix boundaries; K/V and last-position logits byte-identical |
| **I10** | Attention alone does not give chunk invariance. The prefill GEMMs, norms and RoPE must also be chunk-invariant ([Part XVII](part17-prefill.md)), and the QKV GEMV must be row-invariant across verify widths ([Part XVI](part16-decode.md)). | the end-to-end tests below |
| **I11** | In the server, a session's decode rows are never computed by a prefill program shared with another session's prompt. Batching happens only within a program family. | `sky-serve` scheduler |
| **I12** | Window, sinks and soft-cap are per-layer constants. A sink enters split 0 only. | plan verifier |

**End-to-end tests** (CI on the self-hosted macOS runner, VI-T2):
- `spec_equals_plain`: for N prompts × draft lengths 1–16 × temperatures {0, 0.6, 1.0} with keyed Gumbel sampling, speculative output bytes equal plain output bytes (R8 §3.5 rule 6). mlx-lm #1470's script (Qwen3-32B-6bit + 0.6B, divergence at token 91 in mlx-lm) is one of the adversarial cases.
- `decode_verify_invariance`: logits at each position are bit-identical whether computed by decode or within verify passes of width 2–32, at contexts 1K–32K.
- `prefill_chunk_invariance` (I9).
- `paged_equals_slab` ([Part XIV](part14-kv-cache.md) §11).
- `attn_row_independence` (I3), per chip family.
- `free_param_bit_neutrality` (§10): every free attention parameter, every candidate pair.

**What is not promised:** bit-identity across chip families, OS or Metal-compiler versions, Sky builds or tuning entries (the proposed `portable` contract would address chip families, SPINE D6), and L2.

## 13. Disagreement with the spine

1. **The long-context decode target.** SPINE §3 sets "decode, plain, long context (8K–32K): ≥ 1.2× mlx-lm, fp16 KV". §8.2 shows what attention alone can give against MLX's weight streaming at parity. With MLX's attention at the 53% R4 measured, the bound is ≈ 1.08× at 8K, 1.14× at 16K and 1.24× at 32K. Reaching 1.2× at 8K needs MLX's attention at ≤ 33% efficiency, and at 16K ≤ 45%. MLX's ≥ 8K GQA kernel (mlx#4077) makes such low values unlikely at those lengths. **Proposed restatement:** fp16 KV ≥ 1.05× at 8K, ≥ 1.1× at 16K, ≥ 1.2× at 32K and ≥ 1.4× at 128K; with the labeled int8 KV mode, ≥ 1.2× from 16K. MLX's attention efficiency on M5 Max should be measured at G0/G5 before the target is published.

   The same arithmetic bounds Part XXIV's "decode at 4K context ≥ 1.15× mlx-lm" to ≈ 1.04–1.11× on Qwen3-32B. On Qwen3-8B on the M2 Pro (KV share 12% at 4K) the bound is ≈ 1.10×.

   Resolved: the spine now says ≥ 1.05× at 8K, ≥ 1.10× at 16K and ≥ 1.2× at 32K against mlx-lm with 16-bit KV, bounded at ≈ 1.08× / 1.14× / 1.24× by MLX's measured ≈ 53% KV-read efficiency (SPINE §3). The 128K and int8-KV targets proposed here are not spine targets. Part XXIV now states ≥ 1.05× (stretch 1.10×) at 4K.
2. **The verify-cost target.** SPINE §3 asks for "M ≤ 8 tokens verified at ≤ 1.3× the time of M = 1, on M5 Max and M2 Pro". That is reachable for the GEMVs. For attention at long context on shader ALUs it is not: Qwen3-32B (G = 8) on the M5 Max ALU path costs 1.34× at q = 8 and 32K from attention alone (§2.2), and more in practice. The target holds if (a) M5 verify attention runs on the neural accelerators, which AT-D4 requires, and (b) for ALU-only chips it is stated for GQA-8 models at contexts ≤ 8K, or for GQA-4 models (Qwen3-8B on the M2 Pro: 1.05× at 32K).

   Resolved: the spine now applies "≤ 1.3× at M ≤ 8" to the weight GEMVs only and states attention separately: the free verify width q\* = R·s/(2G) is ≈ 15 queries on the M5 neural accelerators and ≈ 4 on shader ALUs, so long-context verify attention runs on the accelerators (SPINE §3).

## 14. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XV-T1 | Attention reference semantics and the row-independence probe | `crates/sky-tir/src/interp/attention.rs`; `crates/sky-metal/tests/attn_row_independence.rs` | The canonical reduction (§4.1, I9) in the interpreter (f32, with an f64 check); a probe kernel per matrix path | The interpreter matches a naive f64 softmax-attention within tolerance; `attn_row_independence` reports per path and chip whether I3 holds (the M2 Pro first; the M5 Max at G5) |
| XV-T2 | `flash_decode` (fp16, ALU path) + `attn_combine` | `crates/sky-kernel/src/skeletons/flash_decode.rs`, `crates/sky-kernel/src/kernels/attn_combine.rs` | Fixed-S split decode attention, GQA-packed | Qwen3-8B shapes on the M2 Pro: ≥ 85% of measured read bandwidth at 4K–32K; ≥ 1.5× MLX's attention time at 4K; `decode_verify_invariance` passes |
| XV-T3 | Verify: chain masks, tree bitmasks, window gather | same skeleton; `crates/sky-llm/src/spec/tree_layout.rs` | q ≤ 32 rows; per-node window gather | `tree_verify_exact`; `spec_equals_plain` on Qwen3-0.6B/8B (with [Part XXI](part21-speculative.md)) |
| XV-T4 | int8 KV in `flash_decode` and `flash_prefill` | codec in `crates/sky-kernel/src/codecs/kv_int8.rs` | Fused int8 attention with folded scales | With XIV-T4: speed at 1K–32K and the quality budget |
| XV-T5 | M5 paths: NA `matmul2d` for decode and verify | `crates/sky-kernel/src/skeletons/flash_decode_na.rs` | NA variant with padded rows | On M5 Max: (a) NA decode attention ≥ 95% of the ALU variant's bandwidth at q = 1, or else the plan uses ALU for both; (b) verify q ≤ 8 at 32K ≤ 1.1× q = 1 on NA |
| XV-T6 | `flash_prefill` for M1–M4 (simdgroup_matrix and MPP variants) | `crates/sky-kernel/src/skeletons/flash_prefill.rs` | FA2 tiles, causal skip, GQA grouping | `prefill_chunk_invariance`; ≥ 1.0× MLX `steel_attention` on Qwen3-8B shapes at 2K–8K on the M2 Pro |
| XV-T7 | `flash_prefill` on M5 NA | same, NA variant | NA tiles ≥ 64×64 | ≥ 1.0× MLX `attention_nax` at 2K–32K on M5 Max |
| XV-T8 | Windows, rings, sinks, soft-cap | mask and sink support in both skeletons | Per-layer constants | gpt-oss-20b and Gemma 3 4B pass Part XII's G-same against llama.cpp; exact rewind across ring wrap |
| XV-T9 | Head dims 64 and 256 | D-split variants | Instantiations for 64/128/256 | Correctness and ≥ 85% bandwidth at decode for each |
| XV-T10 | Attention tuning entries | `sky-tune` tunables for both skeletons | Free-parameter search plus the bit-neutrality test | `free_param_bit_neutrality` passes; the tuned entry beats the heuristic or ties |
| XV-T11 | Aligned-binary-tree combine (EXPERIMENTAL) | `attn_combine` variant | Free split grouping | Built only if the fixed-S tradeoff costs > 3% somewhere (§4.3) |
| XV-T12 | Long-context benchmark and MLX efficiency measurement | `sky bench llm --context 4k..128k` ([Part XXIII](part23-benchmarks.md)) | Decode at 4K–128K for Sky, mlx-lm, llama.cpp; MLX's attention efficiency e | Numbers for the §8.2 table; the restated target of §13 is checked |

## 15. Measurements that decide

| Measurement | Decides |
|---|---|
| Row independence of each matrix path (XV-T1), on M2 Pro and M5 Max | Whether I3 holds for MPP / NA fragments; if not, which paths are allowed under `inference` |
| NA vs ALU decode attention bandwidth at q = 1 on M5 Max (XV-T5) | The M5 attention path (AT-D4) |
| Bandwidth vs S ∈ {256, 512, 1024} at 1K–128K, and partial traffic | The plan's S |
| MLX's attention efficiency on M5 Max for Qwen3-32B at 8K–32K (XV-T12) | The long-context target (§13) |
| Verify attention cost vs q at 4K–32K on M5 Max and M2 Pro | The verify width cap per context ([Part XXI](part21-speculative.md)) |
| Prefill attention vs MLX at 2K–32K on M2 Pro and M5 Max | Whether prefill attention is a lead or a parity item |

## 16. Risks and negative results

- **Row independence of hardware fragments is assumed, not known.** If NA `matmul2d` or MPP fragments do not give per-row results independent of other rows, batch invariance on that path needs one query per fragment. That wastes compute and may make verify attention compute-bound again. The probe (XV-T1) comes before any kernel work on that path.
- **A 1-row `matmul2d` is slower than a GEMV** (R6 negative results). If NA decode attention cannot stream at ALU speed (XV-T5 (a)), M5 plans must choose between a slower decode and a slower verify. Under the `inference` contract they may not mix paths.
- **Long-context gains are bounded by KV share** (§8.2). Better attention alone cannot deliver large wins at 4K–16K. The long-context story needs 32K+, or the labeled int8 KV mode.
- **Fixed S trades occupancy against partial traffic.** If no single S is good across 1K–128K, the aligned-binary-tree variant adds complexity.
- **Prefill attention parity on M5 is the realistic goal**: MLX already uses the NA there, and Apple engineers contribute to both MLX and llama.cpp's tensor paths (R5 §9.5).
- **Negative results kept in view:**
  - MLX and llama.cpp both fix the split *count*, which ties the reduction order to the context length.
  - MLX's maintainers declined verify-window attention fixes (#3826).
  - Unified-max softmax tricks are numerically risky.
  - Undocumented async copies changed behavior across GPU generations (MFA).
