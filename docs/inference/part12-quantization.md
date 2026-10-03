# Part XII — Quantization engine

This part specifies how Sky represents, loads, stores, checks and computes with quantized weights. It covers:
- quantized storage formats as types in the tensor IR (TIR);
- the format list, with priorities and bytes per token for Qwen3-32B;
- native loading of MLX and GGUF checkpoints, and why the two cannot be converted into each other at equal size;
- the relayout cache (`.skyweights`) and the test that proves it holds the same quantized values;
- dequantization inside the decode GEMV, per format;
- quality evaluation and the gates every quantized result must pass;
- the experimental and rejected directions: mixed-bit search, activation quantization, codebook formats, BitNet, sparsity, and texture-compressed weights.

KV-cache formats are in [Part XIV](part14-kv-cache.md). The verify GEMV skeleton and its batch invariance are in [Part XVI](part16-decode.md); prefill GEMM tiling is in [Part XVII](part17-prefill.md); the memory plan is in [Part XIX](part19-memory.md). The dtype system that carries `QFormat` is in [Part VII](part07-tensor-ir.md).

Evidence labels follow the report's convention. Research notes are cited as (R8 §4) and live in [`research/`](research/). Numbers marked "roofline" come from [`roofline.py`](roofline.py).

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| Q-D1 | A quantized tensor's type carries a **`QFormat`** (numeric meaning) and a **`QPacking`** (bit and byte arrangement). Layout transformations may change `QPacking` and tensor layout, never `QFormat`. | Lets the compiler repack weights freely while a type-level rule forbids changing their values (SPINE D5). |
| Q-D2 | Sky loads **MLX safetensors and GGUF natively**. Every speed claim is a same-file pair: Sky vs mlx-lm on the MLX file, Sky vs llama.cpp on the GGUF file. | No conversion between the 4.5-bit MLX and GGUF formats is exact (§4.3). |
| Q-D3 | MLX files go through a **page-aligned relayout cache** (`.skyweights`), built once and proven bit-identical after dequantization. GGUF files are mapped directly for decode. | mlx-community files are not even 4-byte aligned, and unaligned Metal loads silently return wrong words (R9 §1.3). |
| Q-D4 | Each format gets its own **dequant codec** inside one GEMV skeleton. The codec follows the pre-scaled-activation pattern of MLX and llama.cpp. The goal is that the higher-quality formats (Q4_K, Q5_K, Q6_K, MXFP4, IQ4_NL) stream at ≥ 88% of measured bandwidth, against llama.cpp's 75–80% for Q4_K and MXFP4. | Format efficiency, not just bytes, sets decode speed: Q4_K_M decodes 13% slower than Q4_0 in llama.cpp while reading only 5% more bytes (R5 §8 G). |
| Q-D5 | Quality is measured **teacher-forced** against a BF16 reference: KLD, top-1 agreement, perplexity. The R9 gates come before any speed number. | Greedy text diverges from BF16 within tens of tokens for every quantized model (R8 §4.2). |
| Q-D6 | Anything that changes values is a **labeled mode** with a declared KL budget: mixed bits, activation quantization, KV quantization, texture compression. | SPINE D5. |
| Q-D7 | **Rejected:** 2-bit codebook and trellis formats, BitNet, weight sparsity. **Deferred:** entropy-coded lossless codes. **Moonshot:** ASTC texture-compressed weights. | §10–§11. |

## 2. Formats as types

### 2.1 Three separate things: format, packing, layout

| Concept | Meaning | Example | Who may change it |
|---|---|---|---|
| **Format** (`QFormat`) | The map from stored numbers (codes, scales, offsets) to real values | "4-bit unsigned codes, groups of 64 along the input axis, one bf16 scale and one bf16 bias per group: `w = s·q + b`" | Nobody, in an inference graph under the default contract. Only `sky quant` pipelines, which produce a new checkpoint |
| **Packing** (`QPacking`) | Where each code's bits sit inside a word or block, and where the scales sit relative to the codes | MLX: 8 codes per little-endian `u32`, least significant first, scales in a separate tensor. GGUF Q4_0: 18-byte blocks, scale first, nibble *j* and *j*+16 sharing a byte | The relayout pass, losslessly |
| **Layout** | How whole rows, blocks and planes are placed in memory | row-major `[N, K/8]` words; Q, K and V rows concatenated; gate and up rows interleaved; every plane page-aligned | The graph optimizer and relayout pass ([Part VIII](part08-graph-optimizer.md)), losslessly |

The test in §5.5 proves, for every tensor, that a packing or layout change kept every dequantized value bit-identical.

### 2.2 The `QFormat` and `QPacking` descriptors

`QFormat` and `QPacking` are plain data in `sky-tir`, because the TIR reference interpreter needs them to define dequantization. `sky-quant` builds them from file metadata and implements packing, relayout and quality tools on top. The following is the proposed definition (`crates/sky-tir/src/dtype/qformat.rs`):

```rust
/// Numeric meaning of a quantized tensor. Equal QFormat + equal stored numbers
/// = equal real values, whatever the packing or layout.
pub struct QFormat {
    pub code: Code,                  // what one stored code means before scaling
    pub axis: BlockAxis,             // which logical axis is blocked (Reduction for weights, HeadDim for KV)
    pub block: u32,                  // elements per block: 16, 32, 64, 128 or 256
    pub sub: Option<SubBlocks>,      // k-quants and NVFP4: sub-blocks with quantized sub-scales
    pub scale: Scale,                // per-block multiplier
    pub offset: Offset,              // per-block additive term
    pub tensor_scale: Option<FloatTy>, // one per tensor (NVFP4's optional f32 global scale)
}

pub enum Code {
    Uint { bits: u8 },                     // q in [0, 2^bits)      MLX affine, Q4_1, Q4_K, Q5_K
    Biased { bits: u8, zero: u16 },        // q - zero              Q4_0 (zero 8), Q5_0 (16), Q6_K (32)
    Int { bits: u8 },                      // two's complement      Q8_0, int8 KV
    Table { bits: u8, table: TableId },    // table[q]              IQ4_NL (int8 table), E2M1 (MXFP4, NVFP4)
    MiniFloat(MiniFloatTy),                // E4M3, E5M2             MXFP8, FP8 checkpoints
}

pub struct SubBlocks {
    pub size: u32,                   // 32 (Q4_K, Q5_K), 16 (Q6_K, NVFP4)
    pub scale: SubScale,             // Uint6 (Q4_K/Q5_K), Int8 (Q6_K, signed), Ue4m3 (NVFP4)
    pub min: Option<SubScale>,       // Uint6 mins (Q4_K/Q5_K)
}

pub enum Scale {
    Float { ty: FloatTy, sign: SignRule }, // bf16 / f16 / f32 per block
    PowerOfTwo,                            // E8M0: 2^(e - 127), MXFP4 / MXFP8
    SuperTimesSub { super_ty: FloatTy },   // d * sub_scale (k-quants)
}

pub enum SignRule { Any, NonNegative }     // what the producing quantizer guarantees (§2.4)

pub enum Offset {
    None,                                  // symmetric formats
    Add { ty: FloatTy },                   // w = s*c + b         MLX affine (b = group edge), Q4_1 (b = min)
    SubMin { super_ty: FloatTy },          // w = d*sc*q - dmin*m  Q4_K, Q5_K
}

/// Bit and byte arrangement. Repacking may change this; never QFormat.
pub struct QPacking {
    pub codes: CodeOrder,
    pub side: SideData,
    pub align: u32,                  // guaranteed byte alignment of every block / plane start
}

pub enum CodeOrder {
    Lsb32,        // MLX: element j of a little-endian u32 at bits [b*j, b*j+b); 3/5/6-bit codes
                  //      form one little-endian bit stream that may cross word boundaries
    SplitHalf16,  // GGUF Q4_0/Q4_1/Q5_x/IQ4_NL/MXFP4: byte i = element i (low) | element i+16 (high)
    KQuant,       // GGUF Q4_K/Q5_K: per 64 elements, low nibbles = first 32, high nibbles = next 32
    Q6K,          // GGUF Q6_K: 4-bit ql plus 2-bit qh, 4-way interleave per 128-element half
    EvenLow,      // gpt-oss MXFP4 and Metal int4b operands: byte i = element 2i (low) | 2i+1 (high)
    Bytes,        // one code per byte (int8, E4M3)
}

pub enum SideData {
    Planes,                               // codes, scales and offsets are separate tensors (MLX, .skyweights)
    InBlock { block_bytes: u16, layout: BlockFieldsId }, // fields interleaved per block (GGUF)
}
```

Canonical names print as `mlx.affine(b4, g64, bf16)`, `gguf.q4_k`, `mx.fp4(g32)` and so on in TIR text, `sky explain` and error messages. Part VII owns the text syntax; this part owns the payload.

### 2.3 Every supported format as a `QFormat`

| Format | `Code` | Block (sub) | Scale | Offset | Native packing | Bits per weight |
|---|---|---|---|---|---|---|
| MLX affine, b ∈ {2,3,4,5,6,8}, g ∈ {32,64,128} | `Uint{b}` | g | `Float{bf16 or f16, Any}` | `Add{same dtype}` | `Lsb32`, `Planes` | b + 32/g |
| GGUF Q4_0 | `Biased{4, 8}` | 32 | `Float{f16, Any}` | none | `SplitHalf16`, 18-byte blocks | 4.5 |
| GGUF Q4_1 | `Uint{4}` | 32 | `Float{f16, NonNegative}` | `Add{f16}` (the min) | `SplitHalf16`, 20-byte blocks | 5.0 |
| GGUF Q5_0 | `Biased{5, 16}` | 32 | `Float{f16, Any}` | none | `SplitHalf16` + 4-byte high-bit plane, 22-byte blocks | 5.5 |
| GGUF Q8_0 | `Int{8}` | 32 | `Float{f16, NonNegative}` | none | `Bytes`, 34-byte blocks | 8.5 |
| GGUF Q4_K | `Uint{4}` | 256 (8 × 32) | `SuperTimesSub{f16}`, `Uint6` | `SubMin{f16}`, `Uint6` | `KQuant`, 144-byte blocks | 4.5 |
| GGUF Q5_K | `Uint{5}` | 256 (8 × 32) | `SuperTimesSub{f16}`, `Uint6` | `SubMin{f16}`, `Uint6` | `KQuant` + `qh` plane, 176-byte blocks | 5.5 |
| GGUF Q6_K | `Biased{6, 32}` | 256 (16 × 16) | `SuperTimesSub{f16}`, `Int8` (signed) | none | `Q6K`, 210-byte blocks | 6.5625 |
| GGUF IQ4_NL | `Table{4, iq4nl}` | 32 | `Float{f16, Any}` | none | `SplitHalf16`, 18-byte blocks | 4.5 |
| GGUF IQ4_XS | `Table{4, iq4nl}` | 256 (8 × 32) | `SuperTimesSub{f16}`, 6-bit sub-scales | none | 136-byte blocks | 4.25 |
| MXFP4 (OCP; gpt-oss, GGUF, MLX `mxfp4`) | `Table{4, e2m1}` | 32 | `PowerOfTwo` (E8M0) | none | GGUF: 17-byte blocks, `SplitHalf16`; HF/MLX: `EvenLow` or `Lsb32`, `Planes` | 4.25 |
| NVFP4 (MLX `nvfp4`, GGUF NVFP4) | `Table{4, e2m1}` | 16 (MLX) / 64 with 4 × 16 (GGUF) | `Float{ue4m3}` per 16, optional f32 tensor scale | none | `Planes` / 36-byte blocks | 4.5 |
| MXFP8 (MLX `mxfp8`) | `MiniFloat(e4m3)` | 32 | `PowerOfTwo` | none | `Bytes`, `Planes` | 8.25 |
| int8 KV (Sky; [Part XIV](part14-kv-cache.md)) | `Int{8}` | 128 = head_dim | `Float{f16, NonNegative}` | none | `Bytes`, `Planes` | 8.125 |

Sources: MLX `affine_quantize` and modes, `mlx/ops.cpp:5122-5345` (R9 §1.4); GGUF structs, `ggml/src/ggml-common.h:181-370`, reference dequantization in `ggml/src/ggml-quants.c` (R9 §2.2, R5 §5); the IQ4_NL table `kvalues_iq4nl = {-127, -104, …, 113}` and the doubled E2M1 table `kvalues_fp4` in `ggml-common.h:1120-1129`; gpt-oss's MXFP4 nibble order in `transformers/integrations/mxfp4.py:29-46` (R9-ARCH). `[VERIFIED]`

### 2.4 Reference semantics: what "the same value" means

The reference interpreter in `sky-tir` (`crates/sky-tir/src/interp/dequant.rs`) defines the value of every element as **the exact real number given by the format's formula, rounded once to f32**:

| Format | Exact formula | Significant bits needed (exact in f64) |
|---|---|---|
| MLX affine | s·q + b | bf16 s (8) × q (≤ 8) = 16, plus b |
| Q4_0, Q5_0 | d·(q − zero) | 11 + 5 |
| Q4_1 | d·q + m | 11 + 4, plus m |
| Q4_K, Q5_K | d·sc·q − dmin·m | 11 + 6 + 5 = 22 |
| Q6_K | d·sc·(q − 32) | 11 + 8 + 6 = 25 |
| IQ4_NL | d·T[q] | 11 + 8 |
| MXFP4 | 2^(e−127)·E2M1[q] | 2 + exponent |
| NVFP4 | g·s·E2M1[q] | 24 + 4 + 2 = 30 |

All products fit in f64's 53-bit significand, so the interpreter computes them in f64 and rounds once to f32. That value is the oracle for kernels (with a tolerance; [Part VII](part07-tensor-ir.md)) and the exact target of the same-quantization test (§5.5).

**Negative scales are normal.**
- MLX's quantizer pins the larger-magnitude edge of each group exactly. That makes the scale negative for about half of all groups (50.25% in `mlx-community/Qwen3-8B-4bit`), and the bias is then the group's *maximum* (`mlx/ops.cpp:5170-5225`). `[VERIFIED: R9 §1.4]`
- GGUF Q4_0 does the same: `d = max / -8`, where `max` is the signed value of largest magnitude (`ggml/src/ggml-quants.c:132`). Q5_0 uses `max / -16` (`:206`). `[VERIFIED: llama.cpp 81e39ad]`
- Q6_K sub-scales are signed `int8`.
- `SignRule` records what the producing quantizer guarantees. Kernels never rely on it: no kernel may assume a positive scale or bias = group minimum. The loader uses it only for validation (a negative Q8_0 scale means a non-standard file and produces a warning).

### 2.5 How operations type-check on quantized tensors

TIR's type checker ([Part VII](part07-tensor-ir.md)) enforces these rules. Each has a diagnostic code in `sky-diag`.

| Rule | Statement | Example error |
|---|---|---|
| QT1 | A quantized tensor is storage for a real tensor. Its logical element type is the format's value type (f32 in the interpreter). | — |
| QT2 | Only quantization-aware operations accept a quantized operand: `qmatmul` (weight operand), `gather_rows` (embedding lookup), `dequant`, `relayout`. Any other operation needs an explicit `dequant`. | `add(%w: mlx.affine(b4,g64,bf16), %x: f16)`: "quantized operand to `add`; dequantize it first, or use it as a matmul weight" |
| QT3 | The blocked axis of a weight must be the reduction axis of the `qmatmul`. Its length must be a multiple of the block size (and of the super-block, 256, for k-quants). Scale and offset planes must have shape `[N, K/g]`. | "K = 5120 is not a multiple of the Q4_K super-block (256)" (it is; 5120 = 20 × 256) |
| QT4 | `qmatmul(x: f16 or bf16, W: quant) -> x's dtype`, with f32 accumulation under the `inference` contract. The op carries the `QFormat`, so strategy selection can only pick a kernel that implements exactly that format. | — |
| QT5 | Under the default contract an inference graph contains **no `quantize` of weights**. `quantize` exists only in `sky quant` pipelines, which write a new checkpoint with new hashes. | "re-quantizing weights changes the model; this is only allowed in `sky quant`" |
| QT6 | `relayout` must name a recipe from the lossless list (§5.3). The verifier rejects any recipe not on the list. | — |
| QT7 | Activation quantization (`quantize_dynamic`) and KV quantization (`kv_append` into an int8 cache) are allowed only inside a labeled mode. The mode's name is recorded in the plan and printed next to every result. | "int8 KV is a labeled mode: pass `--kv=int8`" |
| QT8 | Two `QFormat`s are equal if every field is equal. Packing differences never make two tensors type-incompatible; they only change which codec a kernel instantiates. | — |

The surface syntax proposed in SPINE D8 (`W: Matrix[25600, 5120] in q4(group: 64)`) maps onto these types. `q4(group: 64)` means `mlx.affine(b4, g64, bf16)`. GGUF formats are written by name, for example `in gguf.q4_k`.

## 3. Supported formats, priorities and bytes per token

### 3.1 Uniform formats on Qwen3-32B

Bytes per decode token are computed with `roofline.py`'s `decode_weight_bytes`: 31.98 B streamed parameters (64 layers plus the untied 0.78 B output head; the embedding is a one-row gather). The ceilings use the M5 Max 40-core: 614 GB/s spec, and 534 GB/s achievable (87%, the current planning value in `roofline.py` [LIKELY: one contributor measurement of ≈530 GB/s; R6 plans 520–585]).

| Format | bpw | GB per token | Ceiling at 614 / 534 GB/s (tok/s) | Priority | Why |
|---|---|---|---|---|---|
| MLX affine 4-bit g64 | 4.50 | 17.99 | 34.1 / 29.7 | **P0** | the headline file (`mlx-community/Qwen3-32B-4bit`) |
| MLX affine 4-bit g128 | 4.25 | 16.99 | 36.1 / 31.4 | **P0** | Qwen's official `Qwen3-32B-MLX-4bit` uses group 128 (R8 §1.3) |
| MLX affine 4-bit g32 | 5.00 | 19.99 | 30.7 / 26.7 | P1 | |
| MLX affine 6-bit g64 | 6.50 | 25.99 | 23.6 / 20.6 | **P0** | mlx-lm#1470's divergence test uses `Qwen3-32B-6bit` (R8 §3.4) |
| MLX affine 8-bit g64 | 8.50 | 33.98 | 18.1 / 15.7 | **P0** | 8-bit targets and drafts; the W8A8 candidate (§9) |
| MLX affine 3-bit g64 | 3.50 | 13.99 | 43.9 / 38.2 | P1 | exists on the Hub; quality risk on Qwen3 (§7.3) |
| MLX affine 5-bit g64 | 5.50 | 21.99 | 27.9 / 24.3 | P1 | `mlx_lm.dynamic_quant` raises sensitive layers to 5 bits (R8 §4.4) |
| MLX affine 2-bit g64 | 2.50 | 9.99 | 61.4 / 53.4 | P2 | collapses on Qwen3 without QAT (§7.3) |
| GGUF Q4_0 | 4.50 | 17.99 | 34.1 / 29.7 | **P0** | llama.cpp's fastest 4-bit kernel; baseline file |
| GGUF Q4_1 | 5.00 | 19.99 | 30.7 / 26.7 | **P0** | imatrix Q4_0 files promote the first n/8 `ffn_down` tensors to Q4_1 (R9 §2.3) |
| GGUF Q8_0 | 8.50 | 33.98 | 18.1 / 15.7 | **P0** | |
| GGUF Q4_K | 4.50 | 17.99 | 34.1 / 29.7 | **P0** | Q4_K_M, the most common GGUF recipe |
| GGUF Q6_K | 6.5625 | 26.24 | 23.4 / 20.4 | **P0** | every Q4_K_M file and most Q4_0 files store the output head (and Q4_K_M half the `ffn_down`/`attn_v`) as Q6_K |
| GGUF Q5_K | 5.50 | 21.99 | 27.9 / 24.3 | P1 | Q5_K_M files |
| GGUF Q5_0 / Q5_1 | 5.5 / 6.0 | 21.99 / 23.99 | 27.9 / 25.6 (spec) | P2 | older files |
| GGUF IQ4_NL | 4.50 | 17.99 | 34.1 / 29.7 | P1 | better quality than Q4_0 at the same size (lookup table) |
| GGUF IQ4_XS | 4.25 | 16.99 | 36.1 / 31.4 | P1 | KLD close to Q4_K_M at fewer bytes (§7.3) |
| MXFP4 | 4.25 | 16.99 | 36.1 / 31.4 | P1 (G7) | gpt-oss experts are released in MXFP4 (R8 §4.6) |
| NVFP4 | 4.50 | 17.99 | 34.1 / 29.7 | P2 | |
| MXFP8 | 8.25 | 32.98 | 18.6 / 16.2 | P2 | |
| bf16 / fp16 | 16.00 | 63.97 | 9.6 / 8.3 | **P0** | norms, reference runs, the BF16 quality baseline |
| Q2_K, Q3_K | 2.625 / 3.4375 | 10.49 / 13.74 | 58.5 / 44.7 (spec) | P2 | loadable for completeness; quality poor on Qwen3 |
| IQ1/IQ2/IQ3, TQ1_0/TQ2_0, Q1_0/Q2_0 | — | — | — | **not supported** | §10.1: ALU-bound codebooks; the loader explains why |

Reproduce with:

```text
cd docs/inference && python3 -c "import roofline as r; \
m=r.MODELS['qwen3-32b']; print(r.decode_weight_bytes(m, r.Quant('x', 6.5625, ''))/1e9)"
```

The bits-per-weight column is exact from the block sizes (`ggml-common.h` static asserts; MLX: b + 2·16/g). To reach 60 tok/s plain at 614 GB/s, the streamed weights would need ≤ 2.56 bpw (R8 App. A §7), a regime where Qwen3 collapses under post-training quantization (§7.3). Bytes per token are a quality decision, not a free speed knob.

### 3.2 Real files: mixed recipes

Files labeled "4-bit" are mixtures, and their sizes differ. Sky prints the per-tensor format histogram and the streamed bytes per token next to every speed number (SPINE D5, D13).

| File (Qwen3-32B) | Recipe | GB per token | File size | Source |
|---|---|---|---|---|
| `mlx-community/Qwen3-32B-4bit` | all linear layers and head affine b4 g64, bf16 scales; embedding b4 | **17.99** | 18.430 GB | R8 §1.3, R9 §2.3 `[VERIFIED: HF tree API, index]` |
| `Qwen/Qwen3-32B-MLX-4bit` | affine b4 g128 | 16.99 | 17.406 GB | R8 §1.3 `[VERIFIED]` |
| Q4_0, no imatrix | Q4_0 + Q6_K output head | 18.19 | 18.63 GB | R9 §2.3 `[VERIFIED arithmetic]` |
| Q4_0, imatrix (unsloth) | + first L/8 `ffn_down` in Q4_1 | 18.26 | 18.703 GB | R8 §1.3 `[VERIFIED]` |
| `Qwen/Qwen3-32B-GGUF` Q4_K_M | Q4_K; Q6_K for the head and for `attn_v`/`ffn_down` in 32 of 64 layers (`use_more_bits`) | 19.32 | 19.762 GB | R8 §1.3 `[VERIFIED: src/llama-quant.cpp]` |
| `mlx-community/Qwen3-32B-6bit` | affine b6 g64 | 25.99 | 26.62 GB | R8 §1.3 |
| `mlx-community/Qwen3-32B-8bit` | affine b8 g64 | 33.98 | 34.81 GB | R8 §1.3 |

Q4_K_M streams 7.4% more bytes per token than MLX 4-bit g64. On equal kernels that alone predicts llama.cpp at about 0.93× of MLX on this model (R8 §1.3) `[LIKELY]`.

## 4. Native loading of MLX and GGUF

### 4.1 MLX safetensors

`crates/sky-llm/src/load/safetensors.rs` reads the 8-byte header length, the JSON header and the tensor table. It never trusts alignment.

**Facts the loader must handle** `[VERIFIED: R9 §1]`:
- MLX's writer pads nothing. In `mlx-community/Qwen3-8B-4bit` the data starts at file offset 103,462 (≡ 6 mod 8), so **0 of 907 tensors are 4-byte aligned**. That includes the packed `uint32` weights.
- Data order is not header order (MLX writes in `unordered_map` order).
- Each quantized `Linear` at path `p` stores `p.weight` (U32 `[out, in·bits/32]`), `p.scales` and `p.biases` (`[out, in/group]`, in the weight dtype, bf16 in practice). An MLX `.biases` tensor is a zero point; it is not a Linear's `.bias`. gpt-oss files contain both.
- Norms (`input_layernorm`, `q_norm`, `k_norm`, …) stay bf16.

**Format resolution**, implemented exactly as mlx-lm does it, failing on anything ambiguous (`crates/sky-quant/src/mlx.rs`):
1. `config["quantization"] = {group_size, bits, mode?}` gives the default. A missing `mode` means affine. An identical `quantization_config` copy may exist.
2. Per-module overrides are keys equal to the module path, holding parameters or `false` (`mlx_lm/utils.py:470-560`; `class_predicate` checks overrides first, `:487-492`).
3. For a module with `.scales` and no entry, infer from shapes (`infer_quant_config`, `utils.py:380-405`): `group = in_dims / scales.shape[-1]`, `bits = weight.shape[-1]·32 / in_dims`. Non-`uint8` scales mean affine. `uint8` scales mean nvfp4 for (bits 4, group 16), mxfp4 for (4, 32) and mxfp8 for (8, 32).
4. Legacy `quantization_config.quant_method` values (`mxfp4` for gpt-oss, `fp8` with UE8M0 32×32 blocks, `compressed-tensors`, `awq`/`gptq`, `bitnet`) are converted by mlx-lm at load time (`utils.py:509-547`). Sky supports `mxfp4` natively (P1), `fp8` at P2, and rejects the rest with a message naming the format.

Then the loader builds the relayout cache (§5) on first use and maps the cache.

### 4.2 GGUF

`crates/sky-llm/src/load/gguf.rs` follows the reference reader (`ggml/src/gguf.cpp`, R9 §2.1): magic, version 3, tensor count, KV pairs, tensor infos, then data at `GGML_PAD(header_end, general.alignment)` with a default alignment of 32. Each tensor's offset must **equal** the padded sum of all earlier tensor sizes (`gguf.cpp:790-815`), and `ne[0] % block_size == 0` is enforced. `[VERIFIED: R9 §2.1]`

What the GGUF path adds over the MLX path:
- **Per-tensor formats.** A "Q4_0" file is Q4_0 + Q4_1 + Q6_K + F32; Q4_K_M is Q4_K + Q6_K + F32 (R9 §2.3). The plan instantiates one GEMV codec per (format, shape) pair. Q, K and V can only be concatenated (§5.3, R3) when they share a format, which Q4_K_M's Q6_K `attn_v` layers do not.
- **Per-file layout flags.** llama.cpp permutes q/k rows for Llama and Mistral at conversion and then uses interleaved RoPE. Qwen, Phi, Gemma 3 and gpt-oss use NeoX RoPE. GGUF Gemma norms already include the +1 (`conversion/gemma.py:131-132`). The loader reads `general.architecture` and records the flags in the TIR graph; it never assumes HF layout (R9 §3.1).
- **Tokenizer metadata.** GGUF carries one EOS; HF's `generation_config.json` has a list (Qwen3: [151645, 151643]). Sky takes the union ([Part XIII](part13-llm-runtime.md)).
- **Zero copy for decode.** Tensors are 32-byte aligned relative to the data section. The loader maps the file with `mmap(PROT_READ, MAP_SHARED)`, rounds the mapping to 16 KiB pages, wraps it with `newBufferWithBytesNoCopy`, and splits it into overlapping views when it exceeds `maxBufferLength` (8.88 GiB on the 16 GB M2 Pro), as llama.cpp does (`ggml-metal-device.m:2176-2270`, R5 §1.5). 32 bytes is enough alignment for the `uint4` loads of the decode GEMV. `[VERIFIED: R9 §1.3, R5 §1.5]`

### 4.3 Why no lossless conversion exists at equal size

The two families quantize differently, so their stored numbers do not map onto each other without growing:

| Conversion | Exact? | Cost | Evidence |
|---|---|---|---|
| MLX affine g64 (bf16 s, b) → GGUF Q4_1 (g32, f16 d, m) | **Yes**, if every scale and bias is fp16-representable: split each group into two 32-blocks with the same d = s and m = b, and permute nibbles to split-half order | 5.0 bpw instead of 4.5 (+11% bytes) | All 128 M scales and biases of `mlx-community/Qwen3-8B-4bit` are fp16-representable `[VERIFIED: R9 §2.5]` |
| GGUF Q4_0 → MLX affine g32 with fp16 scales | **Yes**: s = d, b = −8d | 5.0 bpw instead of 4.5 | arithmetic `[VERIFIED: R9 §2.5]` |
| MLX affine g64 ↔ Q4_0 / Q4_K / IQ4_NL (all 4.5 bpw) | **No.** Q4_0 has no bias and a block of 32; Q4_K has 6-bit sub-scales relative to an fp16 super-scale; neither can express 64 independent bf16 (s, b) pairs in 4.5 bits | — | R9 §2.5 |

So a "same weights" comparison exists only within one file format, unless one side pays 11% more bytes, which is itself an 11% decode-speed handicap on a bandwidth-bound workload. That is why Sky loads both formats natively and every comparison is a same-file pair (SPINE D5).

## 5. The relayout cache (`.skyweights`)

### 5.1 Purpose

The relayout cache exists for four reasons:
1. **Alignment.** mlx-community tensors are 2-byte aligned. On the M2 Pro, binding a `device const uint*` at a buffer offset of 2 or 6 bytes silently returns the word at the rounded-down address; only Metal API validation catches it (R9 §1.3, `core/memprobe`). MPP tensor operands need 128-byte-aligned pointers and row strides (R6 §A3).
2. **Zero copy.** `makeBuffer(bytesNoCopy:)` needs page-aligned memory in one VM region. MLX instead copies every weight with `pread` into fresh buffers (`mlx/backend/common/load.cpp:30-55`), which costs load time and makes the weights footprint-charged memory (R3 §4).
3. **Kernel-friendly layout.** Concatenated Q/K/V rows, gate/up rows adjacent, wide aligned loads (§5.3).
4. **One verified artifact** that several processes can map.

GGUF files skip it for decode (§4.2). The extension in §6.5 may also route GGUF files through it, if a measurement shows it pays.

### 5.2 File format and key

```text
~/Library/Caches/sky/weights/<key>.skyweights   (Part XIII §6.2; SKY_CACHE_DIR moves the root, SKY_WEIGHTS_CACHE the directory)

source-key = sha256( sorted source shard sha256s )
layout-id  = relayout recipe id ‖ .skyweights format version
key        = sha256( sorted source shard sha256s ‖ layout-id )

offset 0        header, 16 KiB: magic "SKYW", format version, the key's inputs in clear,
                tensor count, table offset and length, total size, header sha256
offset 16 KiB   tensor table: per tensor: name, logical shape, QFormat, QPacking, layout recipe,
                plane descriptors {kind: codes | scales | offsets | high bits, offset, length, sha256},
                provenance {source shard, offset, length}
next 16 KiB     data planes, each starting on a 16 KiB page boundary
end             manifest: per-tensor result of the same-quantization test (§5.5), build log
```

Rules:
- The file is written to `model.skyweights.partial` under `flock`, fsynced, and renamed only after the §5.5 test has passed, so a reader never sees a half-built or unverified file ([Part XIII](part13-llm-runtime.md) §6.2 owns the directory and locking rules).
- The source sha256 values come from the Hub resolve headers (`X-Linked-Etag`) or are computed once. Sky never trusts a file name (R9 §5).
- Disk cost is a second copy of the weights: 18.43 GB for Qwen3-32B 4-bit. `sky weights gc` removes caches whose sources are gone. Build time counts in cold start (SPINE D9) and is measured in [Part XVIII](part18-ttft.md).
- Whether the relayout cache counts as "the same file" for benchmarking was an open question in R9 (§10, Q6). Our answer: yes, because §5.5 proves bit-identical dequantized values, and its build time is part of cold start.

### 5.3 Lossless relayout recipes

`crates/sky-quant/src/relayout/` implements a closed list. The TIR verifier rejects any other recipe (QT6).

| ID | Recipe | Applies to | Changes | Notes |
|---|---|---|---|---|
| RL1 | Align every plane to 16 KiB | all MLX files | layout | Required |
| RL2 | Split planes | GGUF blocks → codes / scales / offsets planes | packing | Optional for GGUF (§6.5) |
| RL3 | Concatenate rows of tensors with equal `QFormat` and equal K | Q, K, V → one `[N_q+N_k+N_v, K]`; gate, up → `[2N, K]` | layout | Fewer, larger GEMVs; enables the fused epilogues of [Part XVI](part16-decode.md) |
| RL4 | Interleave rows | gate/up as (g₀, u₀, g₁, u₁, …) in row tiles | layout | One threadgroup owns gᵢ and uᵢ, so SwiGLU fuses into the epilogue (R7 design sketch (b)) |
| RL5 | Reorder codes within a block | split-half → sequential, or → `EvenLow` for MPP `int4b`/`uint4b` operands | packing | The MPP probe verified low nibble = even element (R6 §B1) |
| RL6 | Widen scale dtype | bf16 → f32 always; bf16 → f16 only if exact for every element | packing (value-preserving) | Only for small tensors; wider scales cost bandwidth |
| RL7 | Code offset | unsigned 4-bit q → signed q − 8 (XOR 0x8 per nibble), with b′ = b + 8s | packing (value-preserving) | Only if b + 8s is exact in the stored dtype for every group; otherwise not applied |

Forbidden: anything that changes a dequantized value, such as re-rounding, merging groups, dropping offsets, or converting a scale to a narrower dtype when that is inexact.

### 5.4 Build procedure

`sky weights build hf://mlx-community/Qwen3-32B-4bit` (also run implicitly by `sky run` and `sky compile`):
1. Resolve and hash the sources (`sky-hub`).
2. Plan the layout. The graph optimizer decides which tensors to concatenate or interleave (RL3/RL4), because the recipe depends on the kernels chosen for the plan. The layout recipe ID is part of the key.
3. Stream each source tensor once: read with `pread` into an aligned buffer, apply the recipe, write the planes with `F_NOCACHE` so the page cache is not doubled.
4. Run the same-quantization test (§5.5) over every tensor in parallel.
5. Write the manifest and rename the file into place.

`[EXPERIMENTAL: build time for 18.43 GB on an M5 Max SSD, measured at G5; the alternative of building at sky compile time only is decided by that number (Part VI §12)]`

### 5.5 The same-quantization test (SPINE D5)

**Statement.** For every tensor *t* and every element *i*:

```text
f32_bits( dequant_ref(QFormat_t, QPacking_src, layout_src, bytes_src, i) )
  ==
f32_bits( dequant_ref(QFormat_t, QPacking_cache, layout_cache, bytes_cache, i) )
```

`dequant_ref` is the interpreter's exact-then-round-once dequantization (§2.4), decoding each side through its own packing and layout.

**Implementation** (`crates/sky-quant/src/proof.rs`, test `same_quantization_exhaustive`):
- Exhaustive, not sampled. It runs over all 32.76 B elements of Qwen3-32B with NEON-vectorized decoders, in parallel over tensors. `[EXPERIMENTAL: runtime; the target is under 60 s on an M5 Max]`
- It also checks that scales and offsets were copied exactly and that `QFormat` is equal on both sides.
- It writes a per-tensor result (element count, mismatch count, which must be 0, and the sha256 of the source and cache planes) into the manifest.
- `sky weights verify <file>` re-runs it at any time.

**GPU leg** (`crates/sky-metal/tests/codec_exact.rs`). For each codec, a kernel dequantizes random blocks of real files into f32. The test asserts bit equality with `dequant_ref`. This tests the codec, not the GEMV: the GEMV's accumulation order is checked against the reference with a tolerance ([Part VII](part07-tensor-ir.md)), because pre-scaled activations and f32 FMA change rounding by design (§6.2).

**Adversarial fixtures** (`crates/sky-quant/tests/fixtures/`):
- groups with negative scales;
- bias = max groups;
- a 3-bit and a 6-bit MLX tensor (bit streams that cross words);
- a Q4_K block whose sub-scales use the high-bit path of `get_scale_min_k4` (sub-blocks 4–7, `ggml-quants.c:880-887`);
- Q6_K with negative sub-scales.

A deliberately wrong recipe (an MSB-first nibble order, which gives max |Δ| = 0.55 on real data, R9 §1.4) must fail the test.

## 6. Dequantization inside the GEMV

### 6.1 What the fastest kernels do, and what they reach

**MLX `affine_qmv_fast`** (`mlx/backend/metal/kernels/quantized.h:756-821`; R3 §3.3):
- Threadgroups of 2 simdgroups. Each simdgroup owns 4 output rows, and all 32 lanes split K.
- Each lane loads 16 activations once (32 B) and reuses them for its 4 rows, reading 8 B of codes per row per step.
- **Activations are pre-divided by 1, 16, 256 and 4096** (`:62-69`), so a nibble is used masked but never shifted: `x[4i]·(w & 0x000f) + x[4i+1]·(w & 0x00f0) + …` (`:235-244`).
- Scale and bias are applied once per 16 values as `s·Σ(x·q) + b·Σx`, with Σx computed once per lane and shared by the 4 rows (`:289`).
- All math is f32, with one `simd_sum` per row at the end.

**llama.cpp `mul_vec_q_n_f32_impl`** for Q4_0 (`ggml/src/ggml-metal/kernels/mul_mv.metal:218-304`; R5 §2.3) uses the same pre-scaling (1, 1/256, 1/16, 1/4096 for its split-half order) and folds the −8 zero point into one multiply by Σy per half-block. Its layout (4 rows per simdgroup, 2 simdgroups) has been unchanged since July 2023 (#2188). Its Q4_K kernel runs 2 rows per simdgroup and unpacks 6-bit sub-scales per sub-block (`:1511-1616`). Its MXFP4 kernel copies a 16-entry table to threadgroup memory and uses byte loads, also at 2 rows per simdgroup (`:3034-3110`).

**Measured streaming efficiency at M = 1 on the M2 Pro.** The best raw read is 196.3 GB/s (R6 §B2). MLX's own streaming kernels reach 182–184 GB/s (R3 §7.A).

| Kernel / format | bpw | GB/s | % of 196 GB/s | Source |
|---|---|---|---|---|
| R6 test kernel, MLX-style affine 4-bit g64, 32B gate/up shape (25600×5120) | 4.5 | **186.3** | 95% | R6 §B2 `[VERIFIED]` |
| same kernel, 32B down shape (5120×25600) | 4.5 | 183.8 | 94% | R6 §B2 |
| same kernel, 131072×5120 (378 MB) | 4.5 | 191.4 | 98% | R6 §B2 |
| MLX `qmv_fast`, 32B gate/up / down / lm_head (24 independent calls per eval) | 4.5 | 165–172 / 176–177 / 172–176 | 84–90% | R3 §7.B `[VERIFIED]` |
| llama.cpp Q4_0 | 4.5 | **177.4** | 91% | R5 §8 E `[VERIFIED]` |
| llama.cpp Q8_0 | 8.5 | 176.2 | 90% | R5 §8 E |
| llama.cpp Q6_K | 6.5625 | 172.2 | 88% | R5 §8 E |
| llama.cpp F16 | 16 | 171.1 | 87% | R5 §8 E |
| llama.cpp **Q4_K** | 4.5 | **147.3** | **75%** | R5 §8 E |
| llama.cpp **MXFP4** | 4.25 | **137.7** | **70%** | R5 §8 E |
| tinygrad, generic fused dequant: Q8_0 / Q4_0 / Q6_K | — | 86 / 53 / 38.5 | 20–44% | R2 §8 `[VERIFIED]` |

llama.cpp's kernel numbers use a 4096×14336 matrix (31–48 MB, 117 MB for F16), so a few percent of system-level-cache reuse is possible. R6 found the M2 Pro cache effective only up to about 8 MiB, which makes the effect small. `[LIKELY]`

Three conclusions:
- A simple, well-shaped kernel reaches the roofline for the simple formats. MLX and llama.cpp already do (R6 §B2: "Decode GEMV is fine with simple kernels as long as each lane keeps about 16 bytes in flight per row and handles 2–4 rows").
- **The gap is in the higher-quality formats.** Q4_K loses 25% of bandwidth and MXFP4 30% in llama.cpp. As a result Q4_K_M decodes 13% slower than Q4_0 per token while reading only 5% more bytes (R5 §8 G).
- Generic code generation does not reach it. This is why Sky uses hand-designed skeletons plus generated codecs (SPINE D4; Part VI §8.4).

### 6.2 The dequantization algebra

Every supported format except the lookup-table ones has a linear code map inside a block: w = s·c(q) + o, with c(q) = q − zero. For one output row:

```text
y = Σ_blocks Σ_{i in block} x_i · (s · (q_i − zero) + o)
  = Σ_blocks [ s · Σ_i x_i·q_i  +  (o − s·zero) · Σ_i x_i ]
```

So per block, per row, the kernel needs one dot product of x with the raw codes, plus two multiply-adds. Σx per block is shared by every row the threadgroup owns. Pre-scaling x by 2^(−4j) lets the dot product consume codes masked but unshifted. Per weight that is about 1 AND, 1 integer→float conversion and 1 FMA (R3 §3.3).

Two-level k-quant scales fit the same pattern at sub-block granularity:
- Q4_K: s = d·scᵢ and o = −dmin·mᵢ per 32-element sub-block. Unpacking the 6-bit scᵢ and mᵢ costs a few operations per sub-block per row, amortized over 32 weights.
- Q6_K: s = d·scᵢ with a signed `int8` scᵢ per 16 elements, and zero = 32. Rebuilding a 6-bit code from a 4-bit `ql` and 2 bits of `qh` costs about 2 more operations per weight.

Lookup-table formats cannot use the masked trick, because the map is non-linear:
- **IQ4_NL:** extract the nibble, look up an `int8` table, multiply by d.
- **MXFP4 / NVFP4:** extract the nibble, then either look up E2M1 or build the half value from its bits (sign, 2 exponent bits, 1 mantissa bit). The E8M0 scale is an exact power of two applied once per 32-element block partial sum, so it costs nothing per weight.

**ALU budget.** At 614 GB/s and 4.5 bpw an M5 Max must consume 1.09 × 10¹² weights per second. With 40 cores × 128 lanes × ~1.6 GHz ≈ 8.2 × 10¹² lane-operations per second, that leaves **≈ 7.5 operations per weight** (R8 §4.3; the clock is an estimate, R6 gives ~1.62 GHz [E]). The M2 Pro at 196 GB/s has a similar budget: 2.86 × 10¹² lane-operations/s over 0.35 × 10¹² weights/s ≈ 8.2. `[LIKELY: arithmetic on estimated clocks]`

| Format | Operations per weight, M = 1 (estimate) | Budget at its bpw (M5 Max, 614 GB/s) | Fits? |
|---|---|---|---|
| MLX affine 4-bit, Q4_0, Q4_1 | ~3 (AND, convert, FMA) | 7.5 | yes, room for 3–4 extra verify rows |
| MLX affine 8-bit, Q8_0 | ~2 (convert, FMA) | 14 | yes |
| Q4_K | ~3.3 (adds amortized sub-scale unpack) | 7.5 | yes |
| Q5_K | ~4.5 (adds high-bit merge) | 9.2 | yes |
| Q6_K | ~5 (ql/qh merge, signed sub-scale) | 10.9 | yes |
| IQ4_NL | ~4–5 (shift, mask, table, convert, FMA) | 7.5 | yes; the table access decides it [EXPERIMENTAL] |
| MXFP4 | ~4 (shift, mask, E2M1 decode, FMA) | 7.9 | yes [EXPERIMENTAL] |
| 2-bit codebook / trellis | ≥ 4–5 (decode, unpack, FMA) | 3.3 | **no** (§10.1) |

The operation counts are our estimates from the published inner loops (R3 §3.3, R5 §2.3) `[LIKELY]`. Two things follow:
- The higher-quality formats are not ALU-limited at M = 1. **llama.cpp's Q4_K and MXFP4 deficits are kernel-shape problems** (2 rows per simdgroup, so half the activation reuse; threadgroup-memory table lookups; byte loads), not format limits. `[LIKELY: R5 §11 item 7 reaches the same conclusion]`
- The budget left after dequantization is what multi-row verification spends (§6.4).

### 6.3 Per-format codec design

Each codec is a small MSL generator in `crates/sky-kernel/src/codecs/<format>.rs`, plugged into the one quantized-GEMV skeleton (`crates/sky-kernel/src/skeletons/qgemv.rs`). The skeleton owns the threadgroup and simdgroup shape, the K loop, the activation pre-scaling, the row reuse and the reduction. The codec owns three functions:

```text
codec.load(block_ptr, lane, step)      -> registers holding raw codes + side data for this lane's slice
codec.dot(codes, x_prescaled, row)     -> partial Σ x·q (or Σ x·T[q]) in f32
codec.finish(partial, sums_x, side)    -> s·partial + (o − s·zero)·Σx, accumulated into the row's f32 sum
```

| Codec | Loads per lane per step (each row) | Side data | Activation reuse | Starting parameters (tuned, Part X) | Notes |
|---|---|---|---|---|---|
| `mlx_affine<b, g>` | 8 B of codes as 4 × `u16` (b = 4); 16 B for b = 8; bit-stream window for 3/5/6 | separate bf16 s, b planes | 16 values × 4 pre-scaled copies, 4 rows | 2 SG × 4 rows (MLX); R6 found 8 SG × 4 rows best on M2 Pro | b ∈ {3, 5, 6}: codes straddle `u32` words; load a 64-bit window |
| `gguf_q4_0` / `q4_1` | 8 B of nibbles | in-block f16 d (and m) | pre-scaled in split-half order | 2 SG × 4 rows | zero point folded via Σx |
| `gguf_q8_0` | 16 B of `int8` | in-block f16 d | 1 copy | 4 SG × 2 rows, split K (llama.cpp) | convert `char4` → `float4` |
| `gguf_q4_k` | 16 B of nibbles | 12-byte packed scales/mins per super-block, decoded once per sub-block into registers | Σx per 32-element sub-block | **4 rows per SG** (llama.cpp uses 2) | sub-scale decode amortized over 32 weights × rows |
| `gguf_q5_k` | 16 B of nibbles + 4 B of `qh` | as Q4_K | as Q4_K | 4 rows per SG | high bit merged with one shift + OR per weight |
| `gguf_q6_k` | 8 B of `ql` + 4 B of `qh` | 16 × `int8` sub-scales + f16 d | Σx per 16 | 4 rows per SG | signed sub-scales |
| `iq4_nl` | 8 B of nibbles | in-block f16 d | none (non-linear) | 4 rows per SG | 16-entry table: candidates are registers + `simd_shuffle`, a constant-memory table, or threadgroup memory [EXPERIMENTAL] |
| `mxfp4` | 8 B of nibbles | E8M0 byte per 32 | none | 4 rows per SG | E2M1 → half by bit construction (no table); the 2^(e−127) scale is applied per block partial |
| `f16` / `bf16` | 16 B | none | 1 copy | 2–4 rows per SG | bf16 → f32 is a 16-bit shift |

Rules that apply to every codec:
- **Loads are aligned vectors from planes.** GGUF blocks are 18, 20, 144, 176 or 210 bytes, so block-interleaved formats force 16-bit loads. On the decode path Sky keeps GGUF's native blocks (Q4_0 already streams at 91% of read bandwidth that way). If a K-quant codec misses its target (§6.6), RL2 is applied to that file: codes and side data are split into planes so each lane issues aligned 16-byte loads. That is still bit-exact (§5.5).
- **f32 accumulation everywhere**, one fixed reduction order per (plan, device): the per-lane partials, then one `simd_sum` per row, never split-K that depends on shape. This is the batch-invariance rule of R8 §3.5; [Part XVI](part16-decode.md) owns it.
- **No format-specific branching inside the K loop.** Mixed-format files instantiate one codec per (format, shape) pair at plan time.

### 6.4 Multi-row verification and the M5 neural accelerators

Verification multiplies the same weights by M = 2–32 activation rows. The dequantization work is paid once per weight per pass; each extra row adds one FMA per weight (R8 §2.2). Two consequences for codecs:

1. **On the ALU path** (every chip; on M5, the path decode uses today), the pre-scaling trick costs one pre-scaled copy of x per row. R3's 60-line test kernel kept 8·M f32 activations plus 4·M accumulators per lane and spilled registers from M = 4 (R3 §7.G). Codecs therefore offer a second form for M > 2: **decode the codes once to f32 (or f16) registers and FMA them against M plain activation rows**. That costs ≈ 3 operations plus M FMAs per weight, which fits the ≈ 7.5–8.2 operation budget up to M ≈ 4–5 for 4-bit formats `[LIKELY: arithmetic; Part XVI measures it]`. Which form a given M uses is a numerics-class choice: the decode (M = 1) and verify (M = k) instantiations must use the same per-row arithmetic ([Part XVI](part16-decode.md)).
2. **On the M5 neural accelerators** a weight tile must reach `matmul2d` as a tensor operand. There are two ways:
   - **Dequantize into a half/bf16 tile** in threadgroup memory or a cooperative tensor, then `matmul2d(x, W_tile)`. This is what MLX's NAX kernels do (`quantized_nax.h:488`, R6 §A2). It rounds each dequantized weight to fp16, which the f32 ALU path does not.
   - **Feed the codes directly as an `int4b`/`uint4b` operand** (OS 26.4+; half/bfloat × int4b/uint4b → half/bfloat/float, MSL 4.1 Table 7.3, R6 §A2) and apply the group scale outside the matmul. With a K-tile equal to the group size (64): `y += s_g · (x_g · q_g) + o_g · Σx_g`. The weights stay exact (no fp16 rounding), and no ALU work is spent on decoding. The cost is one scale FMA per output element per 64 K, and the alignment rules of format tensors: 128-byte-aligned pointers and row strides; int4 row strides a multiple of 256 elements (R6 §A3). Qwen3-32B's K values (5120, 8192, 25600) are all multiples of 256, and RL1 provides the alignment.

   `[EXPERIMENTAL: the int4b path's speed on M5 Max is unknown (R6 open question). On the M2 Pro, MPP half×int4b GEMM reached 4.90 TFLOPS against 5.42 for half×half, exact on a full 4096³ check (R6 §B5). MLX's dequantize-to-tile q4 GEMM reaches 35.0 TFLOPS against 46.7 for fp16 on M5 Max (MLX #4411, R6 §A2 [E]). That gap is the prize.]`

   GGUF formats reach the `int4b` path only after RL2 (plane split) and RL5 (nibble order). Q4_0's −8 is folded like a bias: `uint4b` codes plus −8·d·Σx.

The only block-scaled `matmul2d` format is OCP-MX: E8M0 scales per 32 elements, from OS 27 (MSL 4.1 `tensor_blockwise`). There is no native representation of fp16 group scales and biases (R6 §A3). On M5 Max, half × MXFP4 runs at 0.69–0.78× and MXFP4 × MXFP4 at 0.5–0.6× of the FP16 rate (Anemll, R6 §A2) `[LIKELY]`. **Sky therefore does not convert MLX or GGUF weights to MXFP4 for compute**; MXFP4 matters only for gpt-oss, whose weights already are MXFP4.

### 6.5 Prefill GEMM

Prefill multiplies many rows (512 or more) by each weight tile, so dequantization is amortized over the M-tile and the GEMM is compute-bound. The prefill skeleton ([Part XVII](part17-prefill.md)) takes the same codecs in a third form, `codec.dequant_tile(block_ptr) -> half tile`, used by:
- **M1–M4:** dequantize into threadgroup memory, then MPP `matmul2d` (95% of ALU peak for f16 on the M2 Pro, R6 §B5), with `simdgroup_matrix` as the fallback. MLX's non-NAX quantized GEMM is 11–13% behind its fp16 GEMM on the same shapes (R3 §7.D); that gap is a design target for G5, not a format limit.
- **M5:** NA `matmul2d` on a dequantized tile, or `int4b` operands (§6.4).

Codecs for prefill must produce exactly the values `dequant_ref` defines, rounded to the tile's dtype. The prefill GEMM is chunk-invariant ([Part XVII](part17-prefill.md)); it is not required to match the decode GEMV bit for bit.

### 6.6 Efficiency targets

Measured on the M2 Pro at G1 and on the M5 Max at G5. "Read" is the best measured streaming read on that machine (196 GB/s on the M2 Pro, R6 §B2).

| Codec | Target at M = 1, % of read bandwidth | Reference point |
|---|---|---|
| `mlx_affine<4, 64>`, `<4, 128>` | ≥ 92%, and ≥ MLX `quantized_matmul` on every Qwen3-8B/32B shape (G1) | R6 test kernel 94–98%; MLX 84–90% |
| `mlx_affine<8, 64>`, `<6, 64>`, `gguf_q4_0`, `gguf_q8_0` | ≥ 90% | llama.cpp Q4_0 91%, Q8_0 90% |
| `gguf_q4_k`, `gguf_q5_k`, `gguf_q6_k` | **≥ 88%** | llama.cpp Q4_K 75%, Q6_K 88% |
| `mxfp4`, `iq4_nl` | **≥ 88%** | llama.cpp MXFP4 70% |
| `f16`, `bf16` | ≥ 90% | llama.cpp F16 87% |

Targets are design goals, not measurements. If a K-quant or table codec stays below 85% after tuning, the fallback is RL2 plus a plane-layout codec (§6.3), measured against the native-block codec on the same file.

## 7. Quality evaluation

### 7.1 Metrics and harness

R9 §6.6 defines the method, and Sky implements it in `crates/sky-quant/src/quality/` with the CLI `sky bench llm --quality`:
- **Teacher forcing.** Identical token-id sequences go into every engine, and the next-token distributions are compared at every position. Greedy free-running decode is a smoke test only, because the first bf16 near-tie changes every later token.
- **Reference.** The BF16 base model at a pinned commit, computed by two independent implementations (Sky's reference interpreter in f32 and transformers or llama.cpp BF16). They must agree to mean KLD ≲ 1e-4 before either is trusted. `[LIKELY achievable; calibrate]`
- **Corpora.**
  - A Wikitext-2-style corpus: fixed 512- or 2048-token chunks, scoring the second half (the llama-perplexity convention).
  - A chat corpus: 100 fixed prompts in non-thinking mode, responses generated once greedily by the BF16 reference, then teacher-forced in every engine.
  - Both are stored as token ids with a sha256.
- **Per position, dumped** in a compact cross-runtime format (`.skylogits`): logsumexp (f32), the target token's logit, and the top-64 ids and logits, about 0.5 KB per position for ~64 K positions. The full fp32 vocabulary is kept on a 2,048-position subset (~1.2 GB) for exact KLD. llama-perplexity's own KLD stores base log-probs as `uint16` and skips p < e⁻¹⁶ (`tools/perplexity/perplexity.cpp:191-240`), so Sky does not use it as the reference, but it writes llama.cpp's `.kld` format too so llama.cpp's tool can check our numbers.
- **Statistics:**
  - mean KLD (nats/token) with a 95% CI; median, p99 and max KLD;
  - **top-1 agreement %**;
  - PPL and PPL ratio;
  - RMS Δp of the correct token;
  - per-position max |Δlogit| (diagnostic).

### 7.2 Gates

These are R9 §7.3's gates. They are pre-registered and must pass before any speed number is published:

| Gate | Pairs | Pass |
|---|---|---|
| G-tok | Sky's tokenizer vs `tokenizers` (Python), on the prompt texts and R9's 28-string adversarial corpus | identical ids |
| G-same | (Sky, mlx-lm) on the MLX file; (Sky, llama.cpp) on the Q4_0 GGUF; (Sky, llama.cpp) on the Q4_K_M GGUF. 2,048 positions with full vocabulary, 64 K positions with top-64 | **KLD(Sky‖baseline) ≤ 0.1 × KLD(baseline‖BF16)**; **top-1 agreement ≥ 99%**; Sky's KLD vs BF16 ≤ the baseline's + 2σ |
| G-greedy | 32 prompts × 128 tokens | report the matching prefix length; each divergence must sit at a reference top-2 logit margin < 0.1 (a near-tie); a larger margin is a bug |
| G-settings | from engine logs | same KV dtype (mlx-lm bf16, llama.cpp f16 by default: set both, or report both), flash attention on, no speculation, no prompt cache, argmax sampler, batch 1 |

Gate G-same is also the acceptance test of every new codec: a codec that changes results beyond 0.1× of the quantization error itself is wrong, even if it is fast.

### 7.3 What is known about quality today

**Qwen3 under post-training quantization** (arXiv 2505.02214, Tables 2 and 4, instruct models, Wiki2 perplexity and 5-shot MMLU; R8 §4.2) `[VERIFIED]`:

| Qwen3-32B | Wiki2 | MMLU |
|---|---|---|
| FP16 | 7.61 | 81.2 |
| W8 (RTN / AWQ / GPTQ) | 7.60–7.61 | 81.2–81.4 |
| W4 GPTQ g128 | 7.86 | 80.6 |
| W4 AWQ g128 | 8.55 | 78.0 |
| W4 RTN per-channel | **38.5** | 78.4 |
| W3 AWQ g128 | 19.1 | **54.9** |
| W2 (any) | ≥ 38 or NaN | 24–28 |

**GGUF types, KL divergence vs FP16** (Mistral-7B with imatrix; Artefact2 gist, R8 §4.2) `[VERIFIED page]`:

| Type | bpw | Median KLD | Top token differs |
|---|---|---|---|
| Q6_K | 6.57 | 0.0032 | 3.9% |
| Q5_K_M | 5.67 | 0.0043 | 4.4% |
| Q4_K_M | 4.83 | 0.0075 | 5.8% |
| IQ4_XS | 4.32 | 0.0088 | 6.1% |
| Q3_K_M | 3.89 | 0.0171 | 8.4% |
| IQ2_XXS | 2.20 | 0.175 | 23.1% |

**Gaps that Sky measures first:**
1. KLD(MLX 4-bit g64 ‖ BF16) on Qwen3-32B. MLX's default (RTN, affine, g64) is not in the Qwen3 study. It is LIKELY between RTN per-channel and GPTQ g128, and MLX's maintainers expect "about the same quality as Q4_K_M" (mlx-examples#1029, R3-issues F6).
2. The same for Q4_0, Q4_K_M, `mlx_lm.dynamic_quant` and DWQ outputs on Qwen3-32B. No DWQ quality numbers are published (R8 §4.4).

The BF16 reference for Qwen3-32B needs 64 GB of weights, so it is computed on the M5 Max host (128 GB) at G5. On the 16 GB M2 Pro, Qwen3-8B (16.4 GB in BF16) is also too large, so G2/G3 quality runs on the M2 Pro use Qwen3-0.6B to 4B in BF16, and the 8B BF16 reference comes from transformers on the benchmark host. `[VERIFIED sizes: roofline.py]`

### 7.4 Quality budgets for labeled modes

A labeled mode (§1, Q-D6) passes if its extra error is small compared with the error of the weight quantization it builds on:

```text
KLD(Sky in mode M ‖ Sky default, same file)  ≤  β_M × KLD(Sky default ‖ BF16)
top-1 agreement(M vs default)               ≥  τ_M
```

The proposal is β = 0.1 and τ = 99% for int8 KV ([Part XIV](part14-kv-cache.md)) and the W8A8 prefill (§9). That is the same ratio as gate G-same: the mode's error must be one tenth of the quantization error. For mixed bits (§8) the budget is stated against BF16 directly, because the mode replaces the base file. `[EXPERIMENTAL: calibrate β against BF16-vs-BF16 cross-runtime noise, R9 open question 4]`

## 8. Mixed-bit search under a KL budget

`[EXPERIMENTAL for models up to 8B; MOONSHOT for 32B on one machine]`

**What already exists** (R8 §4.4) `[VERIFIED]`:
- mlx-lm's `dynamic_quant` estimates per-layer sensitivity and gives sensitive layers 5 bits and the rest 4, with `--target-bpw` in [4.5, 5.5].
- mlx-lm's `dwq` (distilled weight quantization) fine-tunes scales and biases against the unquantized teacher.
- llama.cpp has fixed mixes (`use_more_bits`), imatrix-weighted rounding and per-tensor `--tensor-type` overrides.
- EvoPress (arXiv 2410.14649) searches per-layer levels by evolution, because "the independence assumption does not generally hold".

**What Sky adds** is a compiler pass with a quality certificate (`sky quant search`, `crates/sky-quant/src/search/`):

```text
minimize   Σ_t bytes(t, bits_t, group_t)                over per-tensor choices
subject to E_calib[ KL(p_BF16 ‖ p_quantized) ] ≤ ε      measured end to end
choices    bits ∈ {3, 4, 5, 6, 8}, group ∈ {32, 64, 128}, format family fixed (MLX affine)
method     1. per-tensor sensitivity proxy (mlx-lm style: output error per layer on calibration activations)
           2. greedy allocation to the byte target
           3. evolutionary refinement (EvoPress style) with true end-to-end KL on the top candidates
output     a new MLX-format checkpoint + a certificate: ε achieved, calibration set hash, bpw, bytes/token
```

**Expected value:** about 4.5 → 3.8–4.1 average bpw at Q4_K_M-like KL, i.e. **1.10–1.18× decode** (R8 §4.4, EXPERIMENTAL). 3-bit averages are not credible on Qwen3 without QAT: AWQ W3 g128 already drops MMLU from 81.2 to 54.9.

**Constraints:**
- This is re-quantization, so the result is a different model. It is never used in a same-file comparison. The fair comparison is against mlx-lm's `dynamic_quant` and DWQ outputs at equal bits per weight: same bpw, compare KLD and speed.
- The search needs the BF16 weights (64 GB for Qwen3-32B) and many end-to-end evaluations. On one M5 Max that is overnight to days for 32B `[LIKELY: evaluation count × prefill time]`. The order is 0.6B–8B first, 32B later.
- Kernels: a mixed file instantiates one `mlx_affine<b, g>` codec per distinct pair, a bounded set (≤ 15 pairs).

## 9. Activation quantization

**Evidence on quality** (arXiv 2505.02214, Qwen3-8B; R8 §4.5) `[VERIFIED]`:

| Qwen3-8B | Wiki2 | MMLU | Δ MMLU |
|---|---|---|---|
| FP16 | 9.71 | 74.7 | — |
| SmoothQuant **W8A8** | 9.62 | 74.0 | **−0.7** |
| SmoothQuant **W4A8** | 12.5 | 63.2 | **−11.5** |
| SmoothQuant W4A4 | 3.4e4 | 24.8 | collapse |

**Evidence on speed.** On M5 Max the neural accelerators run INT8×INT8 at ≈ 122 TOPS, about 2× the FP16 rate (Anemll; tzakharko's ~2048 INT8 ops per core per clock agrees; R6 §A2) `[LIKELY]`. Draw Things measured INT8 matmul 1.61–1.87× faster than FP16 on M5 Max (R3-issues A4). MLX uses no integer `matmul2d` (R3-kernels §d).

**Policy:**
- Activation quantization only helps **prefill** (compute-bound). Decode is bandwidth-bound and gains nothing.
- **W4A8 is not a headline path.** It cost Qwen3-8B 11.5 MMLU points with SmoothQuant. Rotation methods (QuaRot, SpinQuant) do better on Llama, with no Qwen3 numbers (R8 §4.5). It is research.
- **W8A8 is acceptable** as a labeled "fast prefill" mode for 8-bit checkpoints (MLX 8-bit, Q8_0), and only if the §7.4 budget passes on the model at hand. Its −0.7 MMLU on Qwen3-8B is the reason.
- INT8 prefill changes the prompt's KV cache and therefore every later token. Plans in this mode get a different numerics key, so prefix-cache entries never mix with the default mode ([Part XIV](part14-kv-cache.md) §12).
- Kernel shape ([Part XVII](part17-prefill.md)):
  - per-token dynamic activation scales: absmax per row, an exact, order-independent reduction, so it is deterministic under any tiling;
  - `char × char → int` `matmul2d` (OS 26.0);
  - a dequantizing epilogue s_x·s_w.
  - For W4 weights, `char × int4b → int` exists from OS 26.4 (R6 §A2).

The prefill target of 1,600 tok/s on Qwen3-32B needs ≈ 102–107 TFLOP/s, above the FP16 peak (R8 §10). So that number is reachable, if at all, only in this labeled mode. SPINE §3 keeps it out of the headline.

## 10. Rejected directions

### 10.1 2-bit codebook and trellis formats (AQLM, QuIP#, QTIP, GGUF IQ1/IQ2/IQ3)

**Rejected** for Apple GPUs at batch 1.
- **ALU-bound.** At 2 bpw an M5 Max must consume 2.46 × 10¹² weights/s, leaving ≈ 3.3 lane-operations per weight. Lattice and trellis decoding plus unpacking plus the FMA need ≥ 4–5 (R8 §4.3). `[LIKELY]`
- **Measured elsewhere:**
  - QTIP 2-bit on an RTX 6000 Ada reached only 43% (70B) and 34% (7B) of its bandwidth at batch 1, on a GPU with ~3.5× Apple's ALU per byte (QTIP Table 4, R8 §4.3). `[VERIFIED]`
  - llama.cpp's own table: IQ3_S (3.66 bpw) generates 69.3 t/s, *slower* than IQ4_XS (4.46 bpw) at 77.5; IQ2_XXS (2.38 bpw) at 79.9 t/s barely beats Q4_K_S (4.67 bpw) at 76.7 (`tools/quantize/README.md`, R8 §4.3). Fewer bytes did not mean faster. `[VERIFIED]`
- **Quality on Qwen3.** Qwen3 is more sensitive than Llama 3 at ≤ 3 bits: "in Qwen3-8B-Base … C4 from 10.4 to 23.8" for AWQ w3g128, against 9.2 → 11.6 on Llama-3-8B (2505.02214). 2-bit PTQ collapses (MMLU 24–28). `[VERIFIED]`

The loader rejects these types with a message that cites this section.

### 10.2 BitNet and ternary models

**Rejected** as a conversion target. BitNet b1.58 needs native ternary pre-training. Microsoft has released only a 2B model (BitNet b1.58 2B4T, arXiv 2504.12285); there is no Qwen3-32B BitNet (R8 §4.6) `[VERIFIED]`. If a strong ternary model appears (MTPLX mentions a "Ternary Bonsai 2 27B", UNVERIFIED), it is a new model to support, not a quantization feature. ParetoQ explains why: below 3 bits "the representations change drastically", so ternary is a training decision (R8 §4.2 (e)).

### 10.3 Weight sparsity (2:4, unstructured) and activation sparsity

**Rejected** for weights; **not planned** for activations.
- There is no evidence of structured-sparsity hardware in Apple GPUs, the M5 neural accelerators or the ANE (R8 §6.1) `[LIKELY]`. On a bandwidth-bound GPU, sparsity helps only by reading fewer bytes.
- A 50%-sparse 4-bit tensor needs about 2 bits of values, plus 1 bit of mask, plus scales: ≈ 3.5 bpw, the same bytes as dense 3-bit affine. "In most cases quantization outperforms pruning" (arXiv 2307.02973). (R8 §6.3) `[VERIFIED]`
- Activation sparsity (TEAL, arXiv 2408.14690) gives ≤ 1.5× at 40% sparsity but changes outputs (Llama-3-8B PPL 5.87 → 6.21) and has no Qwen3 data. **It also conflicts with speculation**: k verified tokens activate the union of their channels, and TEAL itself needs about 38% sparsity at batch 4 against 60% at batch 1 for the same error. It would compete with Sky's main lever (R8 §6.2). `[VERIFIED abstracts; conflict LIKELY]`

### 10.4 Deferred: entropy-coded lossless codes

MLX-style 4-bit codes have an entropy of 3.59–3.75 bits (R8 §4.7, simulation). Entropy-coding them would cut 4.5 bpw to ≈ 4.09–4.25 bpw, i.e. **5.6–9% fewer bytes with bit-identical output**. DFloat11 (arXiv 2504.11651) does the BF16 analogue. Variable-length decoding at ~10¹² symbols/s eats into the 7.5-operation budget (§6.2). **Deferred** to G9+, `[EXPERIMENTAL]`: it is the only remaining byte reduction that keeps exact outputs, and it would be allowed in same-file comparisons, because it is lossless.

## 11. Moonshot: ASTC texture-compressed weights

`[MOONSHOT]`

**What Apple did.** Apple's 2025 server foundation model stores weights as **ASTC (6×6 HDR blocks of 128 bits, 3.56 bpw)**, decompressed by the GPU's texture hardware "with effectively zero compute cost". Singular vectors were moved into LoRA adapters before compression, and LoRA recovery kept quality close: MMLU 80.0 → 79.2 (arXiv 2507.13575, R8 §4.2 (d)). `[VERIFIED]`

**Why it is interesting.** It is the only route found to fewer bytes per weight without spending ALU on decoding.
- Qwen3-32B at 3.56 bpw would stream 31.98 B × 3.56 / 8 = **14.2 GB per token**, a 43 tok/s roofline at 614 GB/s, against 34.1 at 4.5 bpw: **≤ 1.26×** (R8 §4.3).
- Metal exposes the format: `MTLPixelFormatASTC_6x6_HDR` (raw value 226, `Metal/MTLPixelFormat.hpp:151` in Apple's metal-cpp). `[VERIFIED: header]`

**What decides it** (a feasibility spike, G9+; `crates/sky-metal/benches/astc_gemv.rs`):
1. **Bandwidth.** Can texture reads of ASTC 6×6 HDR data sustain ≥ 90% of the read bandwidth a buffer GEMV gets, measured as compressed bytes per second, on M5 Max? Unknown. The texture units may decode slower than DRAM delivers.
2. **Kernel.** A weight matrix becomes one or more 2D textures, with 6×6 blocks spanning 6 rows × 6 input columns, read with `texture2d<half>::read()` (no filtering) inside a GEMV. Large matrices need tiling across textures, because 2D textures have a maximum dimension. `[LIKELY: 16384 on Apple GPUs; check the Feature Set Tables]`
3. **Quality.** Compress Qwen3-8B with an ASTC HDR encoder (astcenc), measure KLD vs BF16, then measure again with a LoRA recovery step. Apple needed LoRA recovery, and Qwen3 is more quantization-sensitive than Llama 3 (§7.3).

It changes outputs, so it is a labeled mode with a §7.4 budget. Kill criteria: texture bandwidth < 90% of buffer bandwidth, or KLD worse than MLX 4-bit at equal bytes after recovery.

## 12. Disagreement with the spine

Two points of precision; neither changes a decision.

1. **"No lossless conversion exists."** SPINE D5 says no lossless conversion exists between MLX 4-bit and the GGUF 4-bit files. R9 §2.5 shows one direction does exist: MLX affine g64 → GGUF **Q4_1** is exact whenever every bf16 scale and bias is fp16-representable (true for all 128 M groups of Qwen3-8B-4bit), and Q4_0 → MLX affine g32 with fp16 scales is exact. Both cost 5.0 bpw instead of 4.5. The accurate statement, used in §4.3, is: **no lossless conversion exists at equal size**, and a size-increasing conversion is an 11% bandwidth handicap, so it cannot serve a speed comparison. The decision (load both natively; same-file pairs) stands. Resolved: the spine now says (D5) that no lossless conversion exists *at equal size*, and that MLX → GGUF Q4_1 is lossless but 11% larger (R9).
2. **"GGUF is mapped directly."** SPINE D9 maps GGUF directly. That holds for decode with native-block codecs. Two measured facts may require GGUF tensors to go through the relayout cache too:
   - (a) llama.cpp's block-interleaved Q4_K and MXFP4 kernels reach only 75% and 70% of read bandwidth (R5 §8 E). If Sky's native-block codecs also miss the 88% target (§6.6), RL2 (plane split) is the fallback.
   - (b) The M5 `int4b` prefill path (§6.4) needs plane-split, 128-byte-aligned codes.
   
   This chapter therefore keeps direct mapping as the default and makes the GGUF relayout an EXPERIMENTAL extension, decided by the measurements in §13. It is an extension of D9, not a reversal.

## 13. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XII-T1 | `QFormat` / `QPacking` data types and the exact reference dequantization for all P0 formats | `crates/sky-tir/src/dtype/qformat.rs`, `crates/sky-tir/src/interp/dequant.rs` | Descriptors, printers and parsers for canonical names; `dequant_ref` | Golden vectors from `mx.dequantize` (MLX affine b4/b6/b8, g64/g128) and from ggml's `dequantize_row_*` (Q4_0, Q4_1, Q8_0, Q4_K, Q6_K), bit-exact in f32 |
| XII-T2 | Type rules QT1–QT8 with diagnostics | `crates/sky-tir/src/verify/quant.rs`, `sky-diag` codes | Verifier pass | UI goldens: `add` on a quantized tensor, wrong blocked axis, K not a multiple of the block, `quantize` in an inference graph, int8 KV without a mode — each gives its diagnostic |
| XII-T3 | MLX safetensors loader with exact format resolution | `crates/sky-llm/src/load/safetensors.rs`, `crates/sky-quant/src/mlx.rs` | Loader for `mlx-community` and `Qwen/*-MLX-*` files; overrides; `infer_quant_config`; legacy rejection messages | Loads Qwen3-0.6B/8B 4-bit and the 32B 4-bit index; per-tensor formats equal mlx-lm's (`R9-work/core/mlx_q_check.py` as the oracle); a crafted ambiguous file fails loudly |
| XII-T4 | GGUF reader with per-tensor formats and layout flags | `crates/sky-llm/src/load/gguf.rs`, `crates/sky-quant/src/gguf.rs` | Header/KV/tensor-info reader enforcing exact padded offsets; mmap + no-copy views | Reproduces R9 §2.3's tensor-type histograms for the three 8B/0.6B files; rejects IQ2 with the §10.1 message |
| XII-T5 | `.skyweights` writer/reader with recipes RL1–RL7 (the same deliverable as XIII-T7; this part owns the recipes and the proof) | `crates/sky-llm/src/load/skyweights.rs`, `crates/sky-quant/src/relayout/` | `sky weights build/verify/gc` | Every plane 16 KiB-aligned; atomic rename; a killed build leaves no `.skyweights` |
| XII-T6 | Same-quantization test (D5) | `crates/sky-quant/src/proof.rs`; `crates/sky-quant/tests/same_quantization.rs`; fixtures | Exhaustive CPU proof + manifest; GPU codec leg in `crates/sky-metal/tests/codec_exact.rs` | 0 mismatches on Qwen3-0.6B/8B 4-bit and Qwen3-8B Q4_K_M; an MSB-first recipe and a sub-block-4–7 scale bug both fail |
| XII-T7 | Quantized GEMV skeleton + P0 codecs | `crates/sky-kernel/src/skeletons/qgemv.rs`, `crates/sky-kernel/src/codecs/*.rs` | MSL for `mlx_affine<4,64|128>`, `<6,64>`, `<8,64>`, `gguf_q4_0/q4_1/q8_0/q4_k/q6_k`, `f16/bf16` | `sky bench kernel qgemv --format F --shape S` on the M2 Pro meets §6.6 for each codec on Qwen3-8B shapes and on the 32B shapes that fit; correctness within tolerance vs `dequant_ref`; G-same passes on 8B |
| XII-T8 | P1 codecs | same | `gguf_q5_k`, `iq4_nl`, `mxfp4` (both GGUF and gpt-oss layouts), `mlx_affine<2|3|5, g>` | §6.6 targets; gpt-oss-20b MXFP4 experts load and pass G-same vs llama.cpp |
| XII-T9 | K-quant plane-split experiment (RL2) | `crates/sky-quant/src/relayout/plane_split.rs`, a `gguf_q4_k_planes` codec | Both codecs on the same Q4_K_M file | Decision record: plane-split kept only if ≥ 5% faster at M = 1 or M = 4 |
| XII-T10 | Quality harness | `crates/sky-quant/src/quality/`, `sky bench llm --quality` | `.skylogits` dump, KLD/PPL/top-1/Δp statistics, `.kld` export | Reproduces llama-perplexity's KLD on a GGUF pair within its uint16 quantization noise; two BF16 references agree to mean KLD ≲ 1e-4; gates G-same and G-greedy run in CI on 0.6B |
| XII-T11 | `int4b`/`uint4b` operand path for M5 GEMM | `crates/sky-kernel/src/skeletons/gemm_na.rs` (with Part XVII) | NA GEMM taking codes directly, group scale outside the matmul | Bit-exact against the dequantize-to-tile path in f32-exact cases; TFLOPS vs the dequant-tile path on M5 Max reported |
| XII-T12 | Mixed-bit search (EXPERIMENTAL) | `crates/sky-quant/src/search/`, `sky quant search` | Search + certificate | On Qwen3-1.7B/8B: at mlx-lm `dynamic_quant`'s bpw, Sky's KLD is ≤ dynamic_quant's; or at equal KLD, fewer bytes |
| XII-T13 | W8A8 prefill mode (EXPERIMENTAL) | `crates/sky-kernel/src/skeletons/gemm_na_int8.rs`, `--prefill=w8a8` | INT8 activation quantization + int GEMM + epilogue | §7.4 budget on Qwen3-8B 8-bit; prefill speedup vs FP16 on M5 Max reported |
| XII-T14 | ASTC spike (MOONSHOT) | `crates/sky-metal/benches/astc_gemv.rs` | Texture-GEMV microbenchmark + an 8B quality run | Kill or continue on the two criteria of §11 |

## 14. Measurements that decide

| Measurement | Decides | Where |
|---|---|---|
| GB/s at M = 1 of each codec on the M2 Pro (G1) and M5 Max (G5), against the best raw read on the same machine | Whether native-block GGUF codecs suffice (§6.6), or RL2 is needed | XII-T7, T9 |
| Cost of M = 2…8 per codec on ALU, and the register-spill point of each codec form | The ALU verify width per format ([Part XVI](part16-decode.md)) | XII-T7 |
| `int4b` vs dequantize-to-tile GEMM TFLOPS on M5 Max, Qwen3-32B shapes | Whether prefill uses codes directly; whether GGUF gets RL2 for prefill | XII-T11 |
| KLD(MLX 4-bit g64 ‖ BF16), KLD(Q4_K_M ‖ BF16), KLD(Q4_0 ‖ BF16) on Qwen3-32B | The quality line printed next to every speed number; the β of §7.4 | XII-T10 |
| `.skyweights` build time and exhaustive-proof time for Qwen3-32B | Build on first run vs at `sky compile` only | XII-T5, T6 |
| INT8 NA sustained throughput, and W8A8 KLD on Qwen3-8B-8bit | Whether the fast-prefill mode ships | XII-T13 |
| ASTC texture-read bandwidth on M5 Max | Whether §11 continues | XII-T14 |

## 15. Risks and negative results

- **Most format work does not move the headline.** The headline file is MLX 4-bit g64, where MLX already streams at 94–97% of what its kernels reach (R3). Format breadth wins the GGUF comparisons and the quality story, not the MLX decode number.
- **Q4_K efficiency may be harder than the ALU budget suggests.** The operation counts in §6.2 are estimates. llama.cpp's maintainers hand-tune per type and still sit at 75%. RL2 is the fallback; if it fails too, Q4_K_M comparisons are reported honestly as format-limited.
- **The relayout cache doubles disk use** (18.4 GB more for 32B 4-bit) and adds to cold start. If its build time is large, cold-TTFT targets ([Part XVIII](part18-ttft.md)) suffer on first run.
- **The same-quantization test proves values, not speed or kernels.** A codec can be value-exact and still sum in a different order. Gate G-same guards end-to-end equivalence, and Part XVI's invariance tests guard verify/decode equality.
- **Quality numbers for Qwen3 are thin.** There are no published KLD numbers for MLX g64 or DWQ on Qwen3-32B, and no Qwen3 W4A8-with-rotation or KV-quant studies. Sky must produce them before claiming parity of quality.
- **MPP is opaque.** `matmul2d` lowers to backend-defined functions (`air.externally_defined`, R6 §A2). The `int4b` path's speed and its rounding behavior can change with an OS update, so tuning entries and bit-reproducibility are keyed by OS and Metal compiler version ([Part X](part10-autotuner.md)).
- **Negative results kept in view:**
  - Codebook formats are slower than their byte count suggests on GPUs (§10.1).
  - Generic fused dequantization reaches 20–44% of bandwidth (R2).
  - Converting to MX formats loses compute rate on M5 (§6.4).
  - Re-quantization cannot appear in a fair comparison (§4.3).
