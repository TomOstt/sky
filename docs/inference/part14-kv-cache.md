# Part XIV — KV-cache architecture

This part specifies how Sky stores and manages attention state:
- the KV layout per layer and its place in the static memory plan;
- the append, fused into the QKV epilogue;
- int8 KV fused into attention, and 4-bit KV as research;
- a decision framework for KV formats;
- rotating, sliding-window and sink caches (Gemma 3, gpt-oss);
- the recurrent state of hybrid DeltaNet models and its exact rollback;
- trim and rewind for speculation;
- the persistent prefix cache;
- paged KV for the server;
- what we reject (eviction, KV Prediction).

The attention kernels that read the cache are in [Part XV](part15-attention.md). The memory plan that places the cache is in [Part XIX](part19-memory.md). Time-to-first-token policy for the prefix cache (what to cache, when to look it up) is in [Part XVIII](part18-ttft.md) §6. Speculation, which drives rewind, is in [Part XXI](part21-speculative.md).

Evidence labels follow the report's convention. Research notes are cited as (R8 §5) and live in [`research/`](research/). Memory and bandwidth numbers come from [`roofline.py`](roofline.py), current version: M5 Max 40-core at 614 GB/s spec and 534 GB/s planning-achievable (87%, LIKELY; measure on day 1).

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| KV-D1 | Per layer, K and V are each a slab `[kv_heads, capacity, head_dim]`, fp16 by default (bf16 when matching mlx-lm's default in a comparison). Slabs are zero-initialized, 16 KiB-aligned, and indexed by absolute position. | One layout for prefill, decode and verify, which batch invariance requires (R8 §3.3). Head-major rows give each attention threadgroup a long contiguous stream. |
| KV-D2 | The KV cache is **preallocated in the static memory plan** for a context bucket. Growth to the next bucket is a re-plan plus one copy. Commit-on-demand through Metal placement-sparse buffers is EXPERIMENTAL. | Zero allocation in the hot path (SPINE D9), without paying 128K of RAM for a 4K conversation. |
| KV-D3 | K and V are written by the **epilogue of the QKV projection** (V directly; K after QK-norm and RoPE), before the attention kernel runs. Verify and decode write identical bytes for the same position. | Saves dispatches; rollback becomes a counter change (R8 §3.5 rule 3). |
| KV-D4 | **int8 KV** (symmetric, one fp16 scale per token and head) is a **labeled mode** (`--kv=int8`), always fused into attention, with a declared KL budget. | Apple ships 8-bit KV in production (R8 §5.2). Its value is 1.05× at 8K and 1.19× at 32K, but only when fused: mlx-lm's unfused path made decode 6–18% slower (#1651). |
| KV-D5 | **4-bit KV is research** until Qwen3 quality data exist. | No Qwen3 KV-quantization study exists (R8 §5.2). |
| KV-D6 | Sliding-window layers use **ring buffers sized for rollback**: capacity ≥ W − 1 + the largest batch of new positions. Learned sinks (gpt-oss) enter the softmax as a virtual key. | mlx-lm's rotating cache cannot rewind once it wraps (R4 §4.2). |
| KV-D7 | Hybrid DeltaNet state is **double-buffered and replayed** on partial acceptance, using the same sequential recurrence as decode. | Exact rollback at ≤ 2.1% extra traffic per rejected round, instead of writing one state checkpoint per drafted position. |
| KV-D8 | Rewind is a **GPU-resident length counter**; nothing is copied, except the compaction of an accepted tree path. | No CPU round trip in the speculation loop ([Part XX](part20-persistent-execution.md)). |
| KV-D9 | The **prefix cache** keys 256-token blocks by a chained sha256 over token ids within a namespace that includes the numerics key. Blocks persist to disk and are mapped and copied into the slab on a hit. | Exact reuse requires identical kernels (R8 §5.6). Restoring is cheap compared with prefill. |
| KV-D10 | **Paged KV** (fixed pages, block tables) only in `sky-serve`. Pages are multiples of the attention block, so paged and contiguous layouts give identical bits. | Paging buys nothing at batch 1 (R7 §9.1). |
| KV-D11 | **Rejected:** eviction (H2O, SnapKV), KV Prediction, retrofitting sliding windows onto full-attention models. | They change outputs and fail multi-turn use (§13). |

## 2. Sizes

### 2.1 KV bytes per context position

`roofline.py` (`Model.kv_bytes_per_position`), K and V, all layers:

| Model | Layers × KV heads × head_dim | fp16 per position | int8 (Sky) per position |
|---|---|---|---|
| Qwen3-32B | 64 × 8 × 128 | **256 KiB** (262,144 B) | 133,120 B |
| Qwen3-8B | 36 × 8 × 128 | 144 KiB | 74,880 B |
| Qwen3-0.6B / 1.7B | 28 × 8 × 128 | 112 KiB | 58,240 B |
| Qwen3-30B-A3B | 48 × 4 × 128 | 96 KiB | 49,920 B |
| Llama-3.1-8B | 32 × 8 × 128 | 128 KiB | 66,560 B |
| Llama-3.1-70B | 80 × 8 × 128 | 320 KiB | 166,400 B |
| gpt-oss-20b | 12 full layers × 8 × 64, plus 12 sliding layers with a fixed ring | 24,576 B, plus a fixed ring (§7) | 12,672 B |
| Qwen3.6-27B (hybrid) | 16 full layers × 4 × 256, plus 48 DeltaNet layers with fixed state | 65,536 B, plus ≈ 151 MB of state (§8) | 33,024 B |

"int8 (Sky)" is 8 bits per element plus one fp16 scale per head vector: 8.125 bits at head_dim 128 (8.25 at 64, 8.0625 at 256). The last two rows come from R8 §1.6–§1.7 (VERIFIED configs).

Two facts for Qwen3-32B:
- Reading the KV equals reading the 4-bit weights at T ≈ 68.6 K tokens. It is 10% of the weight bytes at 6.9 K tokens, 48% at 32 K, and 6% at 4 K (R7 §0.1, R8 §5.1).
- Qwen3-32B's native context is 32,768 tokens (`max_position_embeddings` 40,960). 131,072 needs YaRN ×4, which must be enabled identically in every engine being compared (R8 §1.1, R9 §3.1).

### 2.2 Memory at 4K, 32K and 128K

`roofline.py` `memory_gb`: weights in MLX 4-bit g64 (18.43 GB, including the embedding), with 0.3 GB of scratch and 0.2 GB of runtime assumed. The "int8 (Sky)" rows use the same function at 1.0156 bytes per element. Roofline's own "int8" row has no scale overhead.

**Qwen3-32B:**

| Context | KV format | KV GB | Total GB |
|---|---|---|---|
| 4,096 | fp16 | 1.07 | **20.00** |
| 4,096 | int8 (roofline, no scales) | 0.54 | 19.47 |
| 4,096 | int8 (Sky) | 0.55 | 19.47 |
| 4,096 | 4-bit g64 | 0.30 | 19.23 |
| 32,768 | fp16 | 8.59 | **27.52** |
| 32,768 | int8 (roofline) | 4.29 | 23.22 |
| 32,768 | int8 (Sky) | 4.36 | 23.29 |
| 32,768 | 4-bit g64 | 2.42 | 21.34 |
| 131,072 | fp16 | 34.36 | **53.29** |
| 131,072 | int8 (roofline) | 17.18 | 36.11 |
| 131,072 | int8 (Sky) | 17.45 | 36.38 |
| 131,072 | 4-bit g64 | 9.66 | 28.59 |

Practical consequences:
- At 128K, fp16 KV pushes Qwen3-32B to 53 GB. The default GPU working set is about 74% of RAM (R6 hardware model, LIKELY for M5), about 47 GB on a 64 GB machine. So fp16 KV at 128K needs a 128 GB M5 Max or a raised `iogpu.wired_limit_mb`. int8 KV (36.4 GB) fits a 64 GB machine.
- The SPINE's RAM target is relative ("≤ 0.9× mlx-lm at 32K, fp16 KV"). KV is 8.59 GB on both sides, so the gain must come from everything else ([Part XIX](part19-memory.md)). The KV design contributes three things mlx-lm lacks (§3.4).

**Qwen3-8B on the 16 GB M2 Pro development machine** (`recommendedMaxWorkingSetSize` 11.84 GiB = 12.7 GB, R6 §B1):

| Context | fp16 total GB | int8 (Sky) total GB |
|---|---|---|
| 4,096 | 5.71 | 5.41 |
| 32,768 | 9.94 | 7.56 |
| 40,960 | 11.15 | 8.17 |

The full native context fits only just in fp16. Gates G2–G4 test long-context kernels on this machine with 8B shapes.

### 2.3 Bandwidth: decode ceilings by context and KV format

`roofline.py` `decode_ceiling_tps`, Qwen3-32B 4-bit g64 weights, tok/s at 614 GB/s spec / 534 GB/s achievable:

| Context | fp16 | int8 (Sky) | 4-bit g64 | int8 ÷ fp16 | KV share of bytes (fp16) |
|---|---|---|---|---|---|
| 1,024 | 33.6 / 29.3 | 33.9 / 29.5 | 34.0 / 29.6 | 1.007 | 1.5% |
| 4,096 | 32.2 / 28.0 | 33.1 / 28.8 | 33.6 / 29.2 | 1.029 | 5.6% |
| 8,192 | 30.5 / 26.5 | 32.2 / 28.0 | 33.0 / 28.7 | **1.055** | 10.7% |
| 16,384 | 27.6 / 24.0 | 30.4 / 26.5 | 32.0 / 27.8 | 1.105 | 19.3% |
| 32,768 | 23.1 / 20.1 | 27.5 / 23.9 | 30.1 / 26.2 | **1.189** | 32.3% |
| 65,536 | 17.5 / 15.2 | 23.0 / 20.0 | 26.9 / 23.4 | 1.317 | 48.8% |
| 131,072 | 11.7 / 10.2 | 17.3 / 15.1 | 22.2 / 19.3 | 1.477 | 65.6% |

These ceilings assume attention reads each KV byte once at full bandwidth. Getting there is [Part XV](part15-attention.md)'s job. MLX's attention reached only 53% of achievable bandwidth on Qwen3-8B at 4K (R4 §7.A).

## 3. Layout and allocation

### 3.1 The slab

For each layer ℓ:

```text
K[ℓ] : [kv_heads, capacity, head_dim]   fp16 (or bf16)      e.g. Qwen3-32B: [8, C, 128]
V[ℓ] : [kv_heads, capacity, head_dim]   fp16 (or bf16)
                                       int8 mode adds:
Ks[ℓ], Vs[ℓ] : [kv_heads, capacity]     fp16 scales; K, V become int8
```

- **Indexing.** Row `[h, slot]` holds the key or value of position p for KV head h. For a contiguous slab, slot = p. For a ring, slot = p mod C (§7). For pages, slot comes from a block table (§11). Kernels never see positions relative to a chunk.
- **Alignment.** Each slab starts on a 16 KiB page boundary. One position of one head is 256 B in fp16. An attention block of B_k = 32 positions is 8 KiB, and capacities are multiples of B_k, so no block straddles a page or a head boundary. The same alignment serves `bytesNoCopy` mapping of persisted prefix blocks (§10) and MPP tensor operands, which need 128-byte alignment (R6 §A3).
- **GQA.** KV heads are stored once. Query head j reads KV head ⌊j / G⌋, where G = 8 for Qwen3-32B and 4 for Qwen3-8B. Attention reads each KV head once for all of its G query heads ([Part XV](part15-attention.md) §4).
- **Dtype.** The KV dtype is the plan's activation dtype: fp16 for Qwen3 by default, bf16 for Gemma (fp16 residual clipping in mlx-lm, R9-ARCH). mlx-lm defaults to a bf16 cache and llama.cpp to f16 (R9 §3.1). Benchmarks set both engines to the same dtype or report both (R9 §7.3, G4).
- **Zero initialization.** Slabs are zero-filled when created. Attention masks unused and future positions to −∞, but a probability of 0 times a non-finite V would still give NaN. With zero-filled slabs, every V the kernel can touch is finite. tinygrad zero-initializes its cache for the same reason ("so the flash kernels can safely read whole tiles past the valid region", R2 §8.1). `[VERIFIED: R2]`
- **Head-major vs position-major.** Head-major (above) gives each attention threadgroup, which owns one KV head and one span of positions, a contiguous stream of S × 256 B. Position-major `[capacity, kv_heads, head_dim]` would make the decode append contiguous (2 KB instead of 8 × 256 B) and growth stride-free, but splits the attention stream into 256-byte pieces 2 KB apart. Appends are tiny and attention reads are large, so we choose head-major. The commit-on-demand variant (§3.3) removes the growth argument.

### 3.2 Preallocation in the static memory plan

The memory plan ([Part XIX](part19-memory.md)) is computed per (model, context bucket, batch, chip). The KV region is one arena range per layer, fixed at plan time. Nothing is allocated per token. The buckets are 4K, 8K, 16K, 32K, 64K and 128K, capped at the model's maximum context.

- `sky run` picks the smallest bucket that fits the prompt plus `--max-tokens` (default policy: the prompt rounded up to a bucket, at least 8K).
- When the logical length reaches the bucket's capacity, the runtime switches to the next bucket's plan. Plans are compiled ahead, and only strides differ, so the switch is one GPU blit of the existing KV into the larger slab: one read and one write of the current KV. For Qwen3-32B at 8K → 16K that is 2.15 GB read plus 2.15 GB written, a few milliseconds at M5 Max copy speeds, once per doubling `[LIKELY: copy speed scaled from the M2 Pro's 183 GB/s blit, R6 §B2]`.
- `--max-context N` preallocates N up front for latency-critical runs.

### 3.3 Commit-on-demand (vAttention-style)

`[EXPERIMENTAL: Metal mechanism exists in the API; behavior and cost for compute buffers on Apple GPUs unmeasured]`

vAttention keeps the KV virtually contiguous and commits physical memory as it grows; it reports up to 1.23× over PagedAttention systems (arXiv 2405.04437, R7 §9.1). On Metal the candidate mechanism is the **placement-sparse buffer**:
- `MTLDevice.newBuffer(length:options:placementSparsePageSize:)` with 16, 64 or 256 KiB sparse pages (`Metal/MTLDevice.hpp:408` in Apple's macOS 26 metal-cpp);
- backed by an `MTLHeap` created with `maxCompatiblePlacementSparsePageSize` (`MTLHeap.hpp:56`);
- pages mapped and unmapped by `MTL4CommandQueue.updateBufferMappings(buffer, heap, operations, count)` (`MTL4CommandQueue.hpp:144`).

`[VERIFIED: the declarations exist in the headers R6 fetched]`

Design: reserve each slab at the model's maximum capacity, and map 16 KiB pages for `[h, 0 : L]` as the logical length L grows. In fp16 that is one page per 64 positions per (layer, KV head, K or V): 1,024 page mappings every 64 tokens for Qwen3-32B, batched into one `updateBufferMappings` call per command buffer. The strides never change, so bucket growth needs no copy.

Why not simply rely on lazy anonymous memory? Metal wires the **entire** `bytesNoCopy` buffer on first kernel use (contributor finding in mlx#2878, R3-issues G3), so untouched pages do not stay uncommitted.

Kill criteria for the spike (XIV-T10):
- mapping cost per page > 1 µs amortized;
- GPU read bandwidth on sparse buffers < 98% of a plain buffer;
- residency-set interaction breaks wiring.

### 3.4 What this layout fixes compared with mlx-lm

| mlx-lm behavior (R4 §4) | Effect | Sky |
|---|---|---|
| `KVCache` grows in 256-slot steps by allocating zeros and concatenating (cache.py:349-364) | A full copy of each layer's K and V every 256 tokens: O(n²/256) traffic, about 26 µs/token at 4K for 8B, amortized (LIKELY) | Preallocated bucket; growth copies once per doubling |
| A prompt-cache hit deep-copies the cache; the first write after that cannot donate and copies the whole buffer (cache.py:1656-1670; mlx-lm#1395) | 2× the conversation's KV exists transiently | Restore copies into the slab once (§10); nothing shares live buffers |
| `QuantizedKVCache` attention materializes `(heads × L × kL)` fp32 scores (base.py:64-107) | +30–73% peak memory at 8K–32K during prefill (mlx-lm#1587) | Fused int8 attention never materializes scores (§5) |
| `RotatingKVCache` is not trimmable once it wraps; `to_quantized` is not implemented (cache.py:544-554; mlx-lm#1573) | No speculation on sliding-window models past the window | Ring capacity rule (§7) |
| Hybrid/SSM `ArraysCache` cannot be trimmed, so no speculation on hybrid models (mlx-lm#1446) | Qwen3.5+ cannot speculate in mlx-lm | State double-buffer + replay (§8) |

## 4. The append, fused into the QKV epilogue

### 4.1 Data flow in one decode layer (Qwen3-32B)

```text
h (5120, f16)
  └─ K1: rms_norm prologue → QKV GEMV (rows: q 8192 | k 1024 | v 1024, one concatenated weight, RL3 in Part XII)
          epilogue:  v rows ─────────────────────────────► V[ℓ][kv_head][slot(p)][:]      (f16, or int8 + scale)
                     q, k rows (raw) ─► scratch
  └─ K2: per head: q_norm / k_norm (RMSNorm over 128) → RoPE(p; θ = 1e6, NeoX, f32 angles)
          q ─► attention scratch (pre-scaled by softmax scale · log2 e)
          k ─► K[ℓ][kv_head][slot(p)][:]                                                  (f16, or int8 + scale)
  └─ attention (Part XV) reads K[ℓ], V[ℓ] for positions [0, p]
```

K2 is tiny: it reads 20 KB and writes 20 KB per layer. When the GEMV tile owns whole heads (a multiple of 128 output rows per threadgroup), the scheduler folds K2 into K1's epilogue. Otherwise K2 stays separate. Per-head RMSNorm needs all 128 outputs of a head, so a tile of 8 rows per threadgroup (MLX's `qmv_fast`) cannot fold it.

What a separate K2 costs: one dependent dispatch per layer, ≈ 1.1–1.25 µs on the M2 Pro (R6 §B3) and assumed ≈ 1 µs on M5. For 64 layers that is ≈ 0.07 ms per token, about 0.2% of a 33.7 ms Qwen3-32B step. It matters more for a 0.6B draft model ([Part XXI](part21-speculative.md)). The scheduler decides by measurement ([Part IX](part09-scheduler.md); Part XXIV's K1 and its occupancy guard).

### 4.2 Rules

1. **Write before read.** The append for positions [p, p + n) happens before the attention kernel of the same layer reads them, in every program: decode (n = 1), verify (n = γ + 1 or tree nodes) and prefill chunks (n = chunk). Prefill and decode use the same cache layout. That is part of the batch-invariance recipe (R8 §3.3: "update the cache before the attention kernel").
2. **Same bytes for the same position.** K and V written for position p by a verify pass must equal, bit for bit, what a decode step would write (R8 §3.5 rule 3). This needs:
   - the QKV projection to be row-invariant across M ([Part XVI](part16-decode.md));
   - QK-norm with a fixed reduction order per head, independent of how many positions the kernel processes;
   - RoPE angles computed in f32 from the absolute position, with the same formula and table in all programs.

   Test: `kv_append_invariance` (XIV-T2).
3. **Positions come from the GPU.** The position p and the slot mapping are read from the GPU-resident counters ([Part XX](part20-persistent-execution.md)), never patched in by the CPU between tokens.
4. **Prefill chunks.** The prefill GEMM's epilogue performs the same append for C positions. The QKV GEMM of prefill is chunk-invariant but is not required to match the decode GEMV bit for bit ([Part XVII](part17-prefill.md)). So K/V of prompt positions are a function of the prompt alone, whatever the chunking (§10.3).

## 5. int8 KV, fused into attention

### 5.1 Format

| Field | Value |
|---|---|
| Code | `Int{8}`, values in [−127, 127] ([Part XII](part12-quantization.md) §2.3) |
| Block | one head vector: 128 elements (head_dim) of one token, for K and V separately |
| Scale | fp16, one per (token, head): s = fp16(absmax / 127) |
| Offset | none (symmetric) |
| Planes | `K` int8 `[kv_heads, C, 128]`; `Ks` fp16 `[kv_heads, C]`; the same for V |
| Bits per element | 8.125; Qwen3-32B: 133,120 B per position (−49% vs fp16) |

**Quantize on append** (in K2 for keys, in K1's epilogue for values; both have the whole 128-vector):
- `absmax` is a max-reduction, which is exact and order-independent, so the result is deterministic under any tiling;
- s = fp16(absmax / 127); if absmax = 0, then s = 0 and every code is 0;
- q = clamp(rint(x / s), −127, 127), using the *stored* fp16 s and round-half-to-even.

Why per (token, head) and not the alternatives:
- **llama.cpp's q8_0 KV** uses one scale per 32 elements along head_dim: 8.5 bits, four scales per head vector (R5 §4). It is more accurate per element, at 4.6% more bytes.
- **Per-channel K scales (KIVI)** need groups over tokens, so a residual fp16 window is kept until a group fills (arXiv 2402.02750). That complicates append, rewind and invariance.
- **Per (token, head) scales fold into the attention math for free** (§5.2).
- Qwen3 applies an RMSNorm to every key head before RoPE, which bounds key norms. That should make per-token key quantization easier than on models without QK-norm `[LIKELY, untested: R8 §5.2]`.

If quality fails (§5.4), the first fallback is a fixed Hadamard rotation of q and k per head (rotation preserves q·k and spreads outlier channels), applied in K2 before quantization. The second is per-32 scales.

### 5.2 Fusion into attention

With per-token scales s_kⱼ and s_vⱼ:

```text
score_ij = s_kj · (q_i · k_int_j)          one multiply per score, after the integer-valued dot product
O_i      = Σ_j (p_ij · s_vj) · v_int_j     scale folded into the probability before the PV product
```

The kernel loads int8 K and V tiles (half the bytes), converts them to fp16 in registers (one conversion per element loaded), and runs the same matrix-fragment code as the fp16 path ([Part XV](part15-attention.md) §4). Scores are never materialized in device memory. On M5, `matmul2d` also accepts `char` operands mixed with `half` (MSL 4.1 Table 7.3, OS 26.0 combinations, R6 §A2), which may remove the conversion. Its rate is unmeasured `[EXPERIMENTAL]`.

### 5.3 Evidence for fusion

| Evidence | Finding | Source |
|---|---|---|
| mlx-lm #1651, M4 Pro, Qwen3-0.6B-4bit, `kv_bits=8` from step 0, unfused | Decode 13% slower at 64 generated tokens, 18% slower at 128, 6% slower at 1,024 (equivalently fp16 was 6–22% faster). It saved memory only past ~256 tokens: 169 MB at 1,024 | R8 §5.3, R4 §9 `[VERIFIED]` |
| mlx-lm #1566 (companion) | Cause: 3 kernels plus a materialized score tensor instead of fused SDPA. On a 5,120-token prompt the quantized cache was *faster*: +3.7% (0.5B), +18.3% (Llama-1B) | R4 §9 |
| mlx-lm #1587 | Quantized decode −3…−7% on dense 32B; peak memory +30–73% at 8K–32K from prefill score materialization | R4 §9 |
| MLX core | No fused quantized-KV SDPA; PR mlx#3026 "Quantized SDPA" open since 2026-01-20 | R3-kernels §c, R3-issues F4 |
| llama.cpp | q8_0/q4_0/q4_1/q5_x KV dequantized inside the FA kernels; for batch ≥ 32 dequantized to an F16 scratch first | R5 §4 |
| Apple Foundation Models 2025 | 8-bit KV cache in production, on device and on the server | arXiv 2507.13575, R8 §5.2 |

Expected value with fusion, from the table in §2.3: 1.03× at 4K, **1.055× at 8K, 1.105× at 16K, 1.19× at 32K, 1.48× at 128K** (roofline; ceilings, not measurements). Plus −49% KV memory.

### 5.4 Quality gate

int8 KV changes outputs, so it is a labeled mode with [Part XII](part12-quantization.md) §7.4's budget:

```text
KLD(Sky int8-KV ‖ Sky fp16-KV, same weights) ≤ 0.1 × KLD(Sky fp16-KV ‖ BF16)      at 8K and 32K contexts
top-1 agreement ≥ 99%
```

It is measured teacher-forced on long-context corpora, because the KV error grows with the number of positions attended. Qwen3-8B is measured on the M2 Pro and Qwen3-32B on the M5 Max. `[EXPERIMENTAL: no Qwen3 KV-quantization data exist]`

## 6. 4-bit KV (research)

`[EXPERIMENTAL]` Candidate formats:
- int4 per (token, head) with an fp16 scale: 4.125 bits;
- MLX-style affine g64: 4.5 bits.

For Qwen3-32B at 32K that is 2.2–2.4 GB instead of 8.6 GB, and a 30.1–30.4 tok/s ceiling instead of 23.1 (spec; §2.3).

**Literature, all on non-Qwen models:**
- KIVI: tuning-free 2-bit, keys per channel, values per token, "almost the same quality" on Llama, Falcon and Mistral (arXiv 2402.02750).
- KVQuant: pre-RoPE per-channel keys, non-uniform levels and outliers, "< 0.1 perplexity degradation with 3-bit" (arXiv 2401.18079).

`[VERIFIED abstracts, R8 §5.2]`

Sky's position:
- 4-bit KV is research until three things hold: a Qwen3 KLD measurement passes the §5.4 budget; the fused kernel keeps decode attention bandwidth-bound (§12, compute); and rewind and prefix caching work with whatever grouping the format uses.
- Formats with token-group scales (KIVI's keys) need a residual fp16 window and change the append and rewind design. Per-token formats do not.

## 7. Sliding-window, rotating and sink caches

### 7.1 Which models

| Family | Pattern | Window | Sinks | Source |
|---|---|---|---|---|
| gpt-oss 20b / 120b | alternating sliding / full, sliding first | **128** | learned logit per head | R9 §3, R8 §5.4 |
| Gemma 3 | 5 local : 1 global; layer i is global when (i+1) % 6 = 0 | 1,024 (512 for 1B) | none | R9-ARCH |
| Ministral 8B | `layer_types` [full, swa, swa, swa] × 9 | 32,768 (= max position, so inert up to 32K) | none | R9 §3 |
| Mistral 7B v0.3, Qwen3 | none (`sliding_window` keys inert unless `use_sliding_window` is true) | — | — | R9 §3.1 |

Window semantics, agreed by mlx-lm (`base.py:35-37`) and llama.cpp (`llama-hparams.h:486-491`): key j is visible to query i when **i − j < W** (R9-ARCH).

### 7.2 Ring buffers sized for rollback

A sliding layer's K and V live in a ring of capacity C, with slot = p mod C. The capacity rule is:

```text
C = round_up( W − 1 + n_new_max , B_k )        n_new_max = max(prefill chunk, verify width, tree nodes)
```

Why. A batch of n_new positions [p, p + n) is written before attention reads it (§4.2). Query p needs keys p − W + 1 … p, and the last query needs p + n − W … p + n − 1. All of them must be present at once: W − 1 + n slots.

The same rule makes **rewind safe after wrap-around**. Rejected drafts at positions up to p + γ overwrite slots of positions up to p + γ − C ≤ p + 1 − W. Those positions are older than any future query's window start (≥ p + 2 − W). So rewinding the logical length never exposes a slot that a rejected draft has destroyed. mlx-lm's `RotatingKVCache` has no such rule and refuses to trim once it wraps (R4 §4.2).

Determinism: C is a multiple of the attention block B_k and blocks are aligned to absolute positions, so each logical block maps to exactly one physical block. The attention kernel walks logical blocks in increasing order and maps each to its slot range ([Part XV](part15-attention.md) §9).

Memory is small. For gpt-oss-20b (12 sliding layers × 8 KV heads × 64 × K,V × 2 B = 24,576 B per slot), with W = 128 and a 512-token prefill chunk: C = 640 and 15.7 MB in total.

### 7.3 Learned sinks (gpt-oss)

A sink is a per-head learned logit that takes part in the softmax normalization with no value vector:
- MLX concatenates it to the scores before the softmax and slices it off afterwards (`mlx/fast.cpp:867-885`).
- llama.cpp uses `ggml_soft_max_add_sinks` (`llama-graph.cpp:2731`).
- mlx-lm's quantized-KV attention path **rejects** sinks (`base.py:119-121`).

`[VERIFIED: R9-ARCH]`

Sky stores sinks as a per-layer `[heads]` weight, not in the KV cache. The attention kernel initializes split 0 of every row with (m = sink_h·log₂e, l = 1, O = 0), a virtual key with V = 0, before the first real block ([Part XV](part15-attention.md) §9). That works the same for fp16 and int8 KV.

### 7.4 StreamingLLM-style "keep first tokens"

StreamingLLM keeps the first tokens' KV ("attention sinks") with a window. That stabilizes generation, but evicted tokens are forgotten (arXiv 2309.17453, R8 §5.4). mlx-lm uses it when `max_kv_size` is set (`RotatingKVCache(max_size, keep=4)`, cache.py:35-38). Applying a window to a model trained with full attention, such as Qwen3, changes outputs (R8 §5.4). **Not offered in year 1.** If a user need appears, it is a labeled mode.

## 8. Hybrid DeltaNet state and exact rollback

### 8.1 What must be stored

Qwen3.5/3.6/3.8-27B use 48 Gated DeltaNet layers and 16 full-attention layers (R8 §1.7):
- **Recurrent state:** 48 layers × 48 V heads × 128 × 128 × fp32 (`mamba_ssm_dtype: float32`) = **151 MB**, read and written every token (≈ 1–2% of traffic, R8 §1.7).
- **Convolution state:** the kernel width is 4, so the last 3 inputs of the convolved channels are kept. If those are q, k, v (16·128·2 + 48·128 = 10,240 channels) in bf16, that is 61 KB per layer and 2.9 MB in total `[LIKELY: channel set and dtype from the qwen3_next design; confirm from the checkpoint]`.
- **KV of the full-attention layers:** 16 layers × 4 KV heads × head_dim 256 → 65,536 B per position, 4× less than Qwen3-32B.
- **The MTP head:** one attention layer with its own small KV cache, under the same rules as any layer.

Each head's state (128 × 128 × 4 B = 64 KB) exceeds the 32 KB threadgroup-memory limit, which still holds on M5 (R6 §A4). The recurrence kernel therefore splits a head's state across threadgroups by value columns. Each column is independent given k ([Part XVI](part16-decode.md)).

### 8.2 Exactness of the recurrence

The chunked parallel scan used for prefill has a different floating-point order from the one-token recurrent update. MLX uses chunk 16 on NAX, 8 without, and sequential chunk 1 for T ≤ 8 (`gated_delta_update.cpp:17-37`, R3-kernels §c). Hence:
- **Verify runs the decode recurrence**, one position after another inside the verify pass, for any verify width, never the chunked scan. Then the output and state for each verified position equal plain decode's bit for bit (R8 §3.5 rule 4). Per token the state math is ~6 × 48 × 128² FLOPs per layer (R8 §3.5), so this costs little.
- **Prefill uses the chunked scan** with scan chunks aligned to absolute positions (multiples of the scan chunk size), so the result does not depend on the prefill chunk size. A prefill that starts at an unaligned position (a new turn after decode) first runs sequential steps up to the next aligned boundary. That depends only on the position, not on the chunking.

### 8.3 Rollback

Verify processes the last accepted token y_p and the drafts d₁…d_γ (γ + 1 steps). If a drafts are accepted, the state needed next is the state after y_p, d₁, …, d_a: a + 1 steps from the state before the round.

| Option | Extra memory | Extra traffic per round | Exact? |
|---|---|---|---|
| A. Checkpoint the state after every verified position | (γ + 1) × 151 MB: 604 MB at γ = 3 | (γ + 1) × 151 MB written every round | yes |
| **B. Double-buffer; on partial acceptance replay a + 1 steps from the saved state** (chosen) | 1 × 151 MB (second buffer) + per-position inputs (q, k, v, gates: KB) | 0 on full acceptance; on rejection 151 MB read + 151 MB written + a + 1 recurrent steps | yes: replay uses the decode kernel |
| C. Recompute with the chunked scan | — | — | **no** (different FP order) |

Option B costs at most ≈ 302 MB per rejected round, about 2.1% of a 14.4 GB step. Full acceptance costs nothing extra. Verify reads S_before and writes S_after into the other buffer. Full acceptance swaps the buffers. Partial acceptance runs the replay kernel from S_before. The convolution state needs no replay: it is the last three inputs, which are kept for the window anyway.

`[EXPERIMENTAL: replay vs checkpoint cost at γ = 3…7 on M5 Max; Part XXI picks γ with it]`

### 8.4 Prefix caching on hybrids

A hybrid model's state is a 151 MB snapshot, not per-token data. The prefix cache (§10) therefore stores state snapshots only at turn ends (optionally every N blocks for long prompts). A hit resumes at the last snapshot at or before the matched length and prefills the rest.

## 9. Trim and rewind for speculation

| Situation | Action | Data movement |
|---|---|---|
| Chain verify, a of γ drafts accepted | logical length L := p + a + 1 (target); the draft's own cache rewinds the same way | none; rejected slots are overwritten later and are invisible meanwhile, because attention reads only positions < L plus the current window |
| Tree verify | Tree nodes occupy window slots in node order. After acceptance, the accepted path's K and V are **compacted** to slots p … p + a, so later decode steps see them at their absolute positions | ≤ γ positions × 256 KiB for Qwen3-32B: ≤ 2 MB per round, one blit |
| Sliding-window layer | the same as chain or tree; the capacity rule (§7.2) keeps every needed position | none |
| DeltaNet layer | double buffer + replay (§8.3) | ≤ 302 MB per rejected round |
| EAGLE-3 draft head | its own one-layer KV, plus the hidden-state taps of the target; rewind by counter | none |
| Separate draft model (Qwen3-0.6B) | its own cache (112 KiB per position); rewind by counter | none |

Rules:
1. **Counters live on the GPU.** The accept/reject kernel writes the new L for target and draft into the uniform buffer the next step reads ([Part XX](part20-persistent-execution.md)). The CPU never sits in the loop.
2. **Masked stale slots contribute exact zeros.** A stale slot inside a block that attention reads is masked to −∞ and has a finite V (zero-initialized slab or a real rejected value), so its contribution is exactly 0 ([Part XV](part15-attention.md) §7).
3. **Never mutate shared data.** Prefix-cache blocks restored into the slab are copies at batch 1 and copy-on-write pages in the server (§11). Rewind therefore never corrupts a cached block.
4. **Exactness test.** `kv_rewind_exact` (XIV-T3) runs speculation with random accept patterns and asserts that, after every round, the K/V bytes for positions < L equal those of a plain decode run.

## 10. The prefix (prompt) cache

### 10.1 Keys

The cache unit is a **block of B_p = 256 tokens**. 256 is a multiple of the attention block B_k and of the example decode split size ([Part XV](part15-attention.md) §4), which SGLang found necessary for deterministic radix caching ("align the truncation point with an integer multiple of the split_kv_size", R8 §3.3). Sky's attention blocks are aligned to absolute positions, so any multiple of B_k would be exact. 256 amortizes hashing and file overhead.

```text
namespace = sha256( model source-key ‖ layout-id ‖ architecture-config hash (RoPE/YaRN, window, sinks)
                    ‖ KV format ‖ numerics key )
numerics key = ( Sky build id, GPU family, OS + Metal compiler version,
                 tuning-entry hashes of the QKV, norm/RoPE and attention kernels, prefill mode (fp16 | w8a8) )
h_0 = sha256( namespace ‖ token ids of block 0 as u32 LE )
h_b = sha256( h_{b−1} ‖ token ids of block b as u32 LE )
```

Block b is stored under h_b. A prompt hits the longest chain of stored blocks that matches its first tokens. A partial last block is not cached; its tokens are prefilled. The numerics key is part of the namespace because KV bits depend on the kernels (SPINE D6). After an OS or Sky update, old entries simply miss.

### 10.2 Storage tiers

| Tier | Where | Content | Eviction |
|---|---|---|---|
| RAM | a KV block pool inside the memory plan's budget | recent blocks, as fp16 or int8 per the plan | LRU by bytes |
| Disk | `~/Library/Caches/sky/kv.noindex/<namespace>/<h_b>.skykv` (root per [Part XIII](part13-llm-runtime.md); `*.noindex` keeps Spotlight out) | one file per block: a 16 KiB header (format version, namespace, h_b, length, dtype, per-layer offsets, sha256 of the payload), then K and V per layer, every plane page-aligned | LRU by bytes under a configurable budget; `sky cache gc` |

A Qwen3-32B block of 256 tokens is 64 MiB in fp16 (33 MiB in int8). A 4K-token system prompt is 1.07 GB (R8 §5.6). On a hit the block file is mmapped and blitted into the slab at its positions: one read of the block. That costs milliseconds warm, or the SSD read time cold. Either way it is far below prefill: 4,096 tokens at ~700 tok/s is ~6 s ([Part XVII](part17-prefill.md)).

Restoring copies into the slab rather than attaching file pages to the slab, because the batch-1 attention kernel reads a contiguous slab. The server's paged layout (§11) can map block files without copying `[EXPERIMENTAL]`.

### 10.3 What "exact" means for a restored prefix

| Claim | Holds? | Why |
|---|---|---|
| Restoring and continuing reproduces the live session bit for bit | **yes**, under the same numerics key | the cached bytes are the session's own K/V; all kernels are deterministic |
| Prompt K/V do not depend on how the prompt was chunked | **yes** | prefill is chunk-invariant ([Part XV](part15-attention.md) §12, [Part XVII](part17-prefill.md)) |
| Re-prefilling a conversation from scratch reproduces the K/V that decode wrote for earlier *generated* tokens | **not guaranteed** | prefill and decode use different matmul kernels, and bit-equality between them is not part of the `inference` contract ([Part XV](part15-attention.md) §12, level L2). A cold re-prefill and a cache hit may therefore continue differently at near-ties; both are valid samples of the model |

The TTFT policy (what to store, when to look up, how the server reuses entries) is [Part XVIII](part18-ttft.md) §6. mlx-lm's equivalents for comparison:
- `save_prompt_cache` / `load_prompt_cache` / `trim_prompt_cache` (safetensors);
- the server's `LRUPromptCache` keyed by a token trie, with deep copies on hit.

llama.cpp has slot save/restore and `--cache-reuse`; `llama-server` defaults to `cache_prompt: true` (R4 §4.4, R5, R9 §0).

## 11. Paged KV for the server

`sky-serve` runs several sessions in one process. It uses pages:
- **Page:** P positions for all KV heads of one layer, `[kv_heads, P, head_dim]` for K and for V. P ∈ {64, 128, 256} is a multiple of B_k, and the decode split size S is a multiple of P.
- **Block table:** per sequence, a GPU buffer of `u32` page ids per layer. Attention reads K and V through it, one page at a time.
- **Bit-identical to the slab.** The reduction structure of attention is defined on absolute positions and blocks, and no block straddles a page, so a paged sequence gives the same bits as the same sequence in a slab. Test: `paged_equals_slab` (XIV-T9).
- **Prefix sharing.** Pages referenced by several sessions carry reference counts. Appending into a shared, partially filled page copies it first (copy-on-write). This is the RadixAttention idea (arXiv 2312.07104; up to 6.4× throughput in SGLang's serving setting, R7 §9.1).
- **Batch invariance in the server.** A session's decode rows always go through the decode programs, never through a prefill GEMM shared with another session's prompt ([Part XV](part15-attention.md) §12).

Batch-1 local runs keep the slab: paging buys nothing at batch 1 (R7 §9.1) and adds an indirection per block.

## 12. A decision framework for KV formats

Four quantities decide a KV format: memory saved, extra compute, bandwidth gained, and accuracy lost. For Qwen3-32B (G = 8, head_dim 128):

| KV format | Bytes per position | KV at 32K | Extra ALU per KV element | Attention FLOP per KV byte, q = 1 / q = 8 | Decode ceiling at 32K / 128K (spec) | Accuracy evidence | Status |
|---|---|---|---|---|---|---|---|
| fp16 / bf16 | 262,144 | 8.59 GB | — | 8 / 64 | 23.1 / 11.7 | reference | **default** |
| int8, fp16 scale per (token, head) | 133,120 | 4.36 GB | 1 conversion; 1 multiply per score and per probability | 16 / 126 | 27.5 / 17.3 | Apple ships 8-bit KV; no Qwen3 study | **labeled mode** `--kv=int8` |
| int8, scale per 32 (llama.cpp q8_0 style) | 139,264 | 4.56 GB | 1 conversion + 1 multiply per element | 15 / 120 | 27.2 / 16.9 | llama.cpp ships it | fallback if per-head scales fail §5.4 |
| int4, fp16 scale per (token, head) | 67,584 | 2.21 GB | shift, mask, conversion | 31 / 248 | 30.4 / 22.9 | KIVI / KVQuant on Llama-class models only | research |

The FLOP-per-byte column is 2·G·q/s, where s is the bytes per KV element ([Part XV](part15-attention.md) §2).

The decision rules, applied by the planner (`crates/sky-llm/src/kv/policy.rs`) and printed by `sky explain`:
1. **Fit.** Pick the smallest bucket and format such that weights + KV + scratch fit the GPU working set: ≈ 74% of RAM by default, raised with `iogpu.wired_limit_mb` (R6). If only a quantized format fits, say so and require the labeled mode explicitly.
2. **Pay only where KV matters.** int8 is worth ≤ 1.03× below 4K, 1.055× at 8K and 1.19× at 32K (§2.3). Below ~7K tokens, where KV is under 10% of the weight bytes (R7 §0.1), the planner never suggests it.
3. **Watch the compute ridge.** Quantized KV doubles (int8) or quadruples (int4) the attention intensity. On the ALU path (M1–M4, and M5 if attention runs on shader ALUs) the ridge is ≈ 29–31 FLOP/B, so int8 KV becomes compute-bound at q ≥ 2 verified queries. On the M5 neural accelerators the ridge is ≈ 122 FLOP/B: q ≤ 7 for int8, q ≤ 15 for fp16. **KV format and speculative verify width compete for the same compute**, and [Part XXI](part21-speculative.md) sizes trees with the KV format as an input.
4. **Accuracy is a budget, not a hope.** Every non-default format passes [Part XII](part12-quantization.md) §7.4's gate on the model at hand, at the context lengths used.
5. **Never unfused.** A quantized KV format without a fused attention kernel for the current chip is not offered (mlx-lm #1651, #1587).

## 13. Rejected

| Technique | What it does | Why rejected | Source |
|---|---|---|---|
| H2O (arXiv 2306.14048) | keeps "heavy hitter" tokens plus recent ones | Changes outputs. Evicted tokens are gone, which fails shared-context and multi-turn use (agent loops, chat history), the typical local workload | R8 §5.5 |
| SnapKV (arXiv 2404.14469) | selects per-head KV positions from an observation window at the end of the prompt | Same. Its "comparable accuracy" is on LongBench single-turn tasks; SCBench (arXiv 2412.10319) finds "sub-O(n) memory methods suffer in multi-turn scenarios" | R8 §5.5 |
| KV Prediction (Apple, arXiv 2410.08391) | a small auxiliary model predicts the base model's KV to cut TTFT | Changes outputs with large accuracy losses: OE-1.1B TriviaQA 23.57% → 16.09% at 2.36× fewer TTFT FLOPs. Needs a trained predictor per base model; none exists for Qwen3; shown only at 1–3B | R8 §5.7 |
| Retrofitting a sliding window or StreamingLLM keep-tokens onto a full-attention model | bounded KV | Changes outputs; long-context retrieval is lost | R8 §5.4 |
| Unfused quantized KV (mlx-lm's design) | dequantize outside attention | Slower than fp16 below several thousand tokens and higher prefill peak memory | §5.3 |

## 14. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XIV-T1 | KV slab layout and bucketed preallocation | `crates/sky-llm/src/kv/{layout.rs, slab.rs}`; memory planner hooks in `sky-tir` ([Part XIX](part19-memory.md)) | Slabs per layer in the arena, zero-filled, 16 KiB-aligned; bucket selection | Qwen3-32B at 32K allocates exactly 8,589,934,592 B of KV plus ≤ one page of padding per plane; an allocation counter shows 0 allocations per decode token |
| XIV-T2 | Append in the QKV epilogue + the `qk_norm_rope_append` kernel | `crates/sky-kernel/src/epilogues/kv_append.rs`, `crates/sky-kernel/src/kernels/qk_norm_rope_append.rs` | Both fused and two-kernel variants | `kv_append_invariance`: decode, verify (γ = 1…16) and prefill chunks (64…2048) write identical K/V bytes for the same positions; values match the reference interpreter within tolerance |
| XIV-T3 | GPU-resident length counters and rewind | `crates/sky-llm/src/kv/cursor.rs` | Counters in the uniform buffer; accept/reject writes them | `kv_rewind_exact`: 1,000 random accept patterns; K/V for positions < L equal a plain run's after every round |
| XIV-T4 | int8 KV: quantize-on-append + fused attention (with XV-T4) | `crates/sky-kernel/src/epilogues/kv_quant.rs`, attention skeleton | `--kv=int8` mode | §5.4 budget on Qwen3-8B (8K, 32K); M5 Max Qwen3-32B decode at 32K ≥ 1.12× fp16 KV (ceiling 1.19×); at 1K ≥ 0.98× (no regression) |
| XIV-T5 | Ring caches and learned sinks | `crates/sky-llm/src/kv/ring.rs`; attention mask/sink support | Capacity rule; logical→physical block mapping | gpt-oss-20b passes Part XII's G-same vs llama.cpp; speculation with rewind across a wrap is exact |
| XIV-T6 | DeltaNet state manager: double buffer + replay | `crates/sky-llm/src/kv/recurrent.rs`, `crates/sky-kernel/src/kernels/gdn_replay.rs` | Rollback by replay; the conv-state window | Qwen3.6-27B speculative (MTP) output byte-identical to plain on 32 prompts × T ∈ {0, 0.6}; state memory = 2 × 151 MB |
| XIV-T7 | Tree-path compaction | `crates/sky-llm/src/kv/compact.rs` | Blit of accepted nodes to p … p + a | `tree_verify_exact`: tree speculation output == chain decode output |
| XIV-T8 | Prefix cache | `crates/sky-llm/src/prefix/{hash.rs, pool.rs, disk.rs}` | Hash chain, RAM pool, `.skykv` files, restore by blit | Restore + continue == the uninterrupted session, byte for byte, on 32 multi-turn transcripts; a changed numerics key misses; a corrupted file is detected by its sha256 and ignored |
| XIV-T9 | Paged KV for `sky-serve` | `crates/sky-serve/src/kv_pages.rs`; attention block-table path | Pages, block tables, copy-on-write prefix sharing | `paged_equals_slab`: identical output bytes; two sessions with a shared system prompt store its pages once |
| XIV-T10 | Commit-on-demand spike | `crates/sky-metal/benches/sparse_kv.rs` | Placement-sparse slab with `updateBufferMappings` | Decision record against the kill criteria of §3.3 |
| XIV-T11 | Bucket growth | `crates/sky-llm/src/kv/grow.rs` | Re-plan + blit | Growing 4K → 32K mid-session gives output identical to a run preallocated at 32K; growth time per doubling reported |
| XIV-T12 | 4-bit KV research harness | `crates/sky-quant/src/kvq/` | int4 per-head and affine g64 KV codecs behind `--kv=int4-research` | KLD report on Qwen3-8B at 8K/32K; no shipping decision before it |

## 15. Measurements that decide

| Measurement | Decides |
|---|---|
| Fused int8 vs fp16 attention, 1K–128K, Qwen3-8B on the M2 Pro and Qwen3-32B on the M5 Max | Whether `--kv=int8` delivers its roofline gain, and from which context the planner suggests it |
| ΔKLD of int8 KV on Qwen3-8B/32B: per-head scale vs per-32 scale vs Hadamard + per-head | The int8 format, or whether int8 ships at all |
| Placement-sparse buffer mapping cost and read bandwidth | Commit-on-demand vs bucket growth |
| DeltaNet replay vs checkpoint-all at γ = 3…7 | §8.3 option B, and the γ for hybrids |
| Fused append (K1 with K2 folded) vs two kernels, per layer, on M2 Pro and M5 Max | The epilogue fusion choice ([Part IX](part09-scheduler.md)) |
| Prefix restore time for a 4K prefix, warm and cold page cache | The TTFT claims of [Part XVIII](part18-ttft.md) |

## 16. Risks and negative results

- **int8 KV quality on Qwen3 is unknown.** Apple's production use is on Apple's own models with LoRA recovery for other components. If the budget fails, int8 KV stays a research flag and the long-context story rests on attention efficiency alone ([Part XV](part15-attention.md) §8).
- **KV formats compete with speculation.** Quantized KV raises attention's arithmetic intensity, and on chips without neural accelerators that shrinks the free verify width at long context (§12 rule 3).
- **Placement-sparse buffers may not behave as hoped** for compute (residency, mapping cost, OS version). Bucket growth is the fallback and works today.
- **Hybrid state is large per snapshot** (151 MB). Prefix caching for hybrids is coarse (turn boundaries only), and rollback costs real traffic when drafts are rejected often.
- **The numerics key makes caches short-lived.** Every OS or Metal-compiler update invalidates every prefix-cache entry. That is the price of exactness.
- **Disk.** The prefix cache can grow large (64 MiB per 256-token block for Qwen3-32B in fp16); the default budget must be conservative.
- **Negative results kept in view:**
  - unfused quantized KV is slower than fp16 at short context (mlx-lm #1651);
  - eviction methods fail multi-turn use (SCBench);
  - KV Prediction trades large accuracy losses for TTFT FLOPs;
  - paging buys nothing at batch 1.
