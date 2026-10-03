# R9 — Model formats, the Hugging Face ecosystem, model architectures, and benchmark methodology on macOS

Research notes for the Sky acquisition-grade report.

- **Agent:** R9.
- **Dates:** 2026-10-01 to 02.
- **Status:** complete; the architect brief is at the end (section 10).

**Labels.**

| Label | Meaning |
|---|---|
| VERIFIED | Spec or official doc URL, source `path:line`, or our own check with the exact command and output. |
| LIKELY | Reasoned, with the reason given. |
| EXPERIMENTAL | Not yet proven. |
| MOONSHOT | Speculative. |

**Machine used for every check.** VERIFIED via `sysctl`/`sw_vers`:
- MacBook Pro Mac14,9, Apple M2 Pro, 16 GB, macOS 26.6.1 (25G76).
- Metal `maxBufferLength` 8.88 GiB; `recommendedMaxWorkingSetSize` 11.84 GiB; 16 KiB pages.
- This is **not** the M5 Max target. Absolute speeds and power figures here are not M5 Max numbers.

**Versions.** VERIFIED from `pip` dist-info and `git log -1`:

| Component | Version |
|---|---|
| mlx (pip) | 0.32.3 |
| mlx-lm (pip) | 0.32.0 |
| mlx source | `0840c42` (2026-10-01) |
| mlx-lm source | `53b9af3` (2026-10-01) |
| llama.cpp | `81e39ad` (2026-10-01) |
| huggingface_hub | 1.33.0 |
| hf_xet | 1.6.0 |
| safetensors (py) | 0.8.0 |
| tokenizers | 0.23.2 |
| transformers | 5.18.0 |

All four source clones are shallow, so no release tags are recorded.

**Scratch evidence.** Everything is under `S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`, in `$S/research/R9-work/`:

| Path | Topic |
|---|---|
| `core/` | safetensors, GGUF, memory probe, quant checks |
| `arch/` | architectures; full notes in `arch/R9-arch.md` |
| `tok/` | tokenizers and templates |
| `hub/` | HF Hub |
| `power/` | power and thermal |

---

## 0. Top findings

1. **MLX-written safetensors are not aligned at all.** VERIFIED.
   - In `mlx-community/Qwen3-8B-4bit` the header length is 103,454 bytes, so the data section starts at file offset 103,462 (≡ 6 mod 8).
   - Every one of the 907 tensors therefore sits at a file offset that is only 2-byte aligned. That includes the `uint32`-packed 4-bit weights.
   - Files written by the HF safetensors library are different: the header is padded to 8 bytes and tensors are sorted by dtype size, so they are aligned to their element size, up to 8 bytes.
   - Neither writer guarantees 16 bytes.
2. **A zero-copy MLX loader on Metal gives silently wrong answers.** VERIFIED on M2 Pro.
   - Bind a `device const uint*` at a buffer offset of 2 or 6 bytes and the GPU returns data from the address rounded down to 4 bytes. No error is raised.
   - Metal API validation does catch it: "offset … must be a multiple of 4" for `uint`, and "multiple of 16" for `uint4`.
   - Consequence: Sky cannot `mmap` + `makeBuffer(bytesNoCopy:)` an mlx-community file and read the packed words directly. It must copy, repack once into an aligned cache, or use byte/ushort loads.
   - GGUF does not have this problem. Its tensors are 32-byte aligned by construction (`ggml/src/gguf.cpp:780-815`).
3. **MLX affine 4-bit packing.** VERIFIED against `mx.dequantize`.
   - The format is `w = scale·q + bias`.
   - Eight 4-bit values per little-endian `uint32`, element *j* in bits [4j, 4j+4), least significant first.
   - Group size is 64 along the input dimension. Scales and biases are bf16.
   - **50.25 % of the scales are negative.** The quantizer anchors the larger-magnitude edge of each group exactly (`mlx/ops.cpp:5170-5225`). Kernels must not assume positive scales or bias = group minimum.
4. **Lossless conversion from MLX to GGUF exists, but costs size.** VERIFIED for Qwen3-8B-4bit.
   - All 128 M scales and 128 M biases are exactly representable in fp16. MLX group-64 affine therefore converts losslessly to GGUF Q4_1 (group 32, fp16 `d`/`m`).
   - The cost is 5.0 instead of 4.5 bits per weight.
   - No 4.5-bpw GGUF type can hold MLX weights exactly, and the reverse is also true.
   - So the "same weights" comparison must be done **per format**: Sky and mlx-lm on the identical MLX file; Sky and llama.cpp on the identical GGUF file.
5. **"Q4_0" and "4-bit" files are mixed precision, and their sizes differ.** VERIFIED.
   - bartowski's Qwen3-8B Q4_0:
     - `output.weight` is Q6_K;
     - 4 `ffn_down` tensors are Q4_1, because llama-quantize upgrades them when an imatrix is used (`src/llama-quant.cpp:661-666`);
     - overall 4.67 bpw.
   - Qwen's official Q4_K_M: 4.90 bpw. Its metadata name is "Qwen3 8B Awq Compatible Instruct", so its provenance is unclear.
   - mlx-community 4-bit: 4.50 bpw, with a 4-bit LM head.
   - Decode is bandwidth-bound, so a 9 % size difference becomes a free speed difference.
   - The protocol must report **weight bytes read per decode token**. It must also build all quants itself from one pinned BF16 base commit.
6. **The tools' own speed numbers are not comparable with each other.** VERIFIED from source.
   - **llama-bench `tg128`:**
     - decodes from an *empty* KV cache;
     - feeds random tokens with no sampling;
     - reports the mean of per-repetition rates.
   - **mlx-lm `generation_tps`:**
     - runs after the prompt;
     - includes sampling, `logsumexp`, detokenization and Python;
     - counts the first token in the numerator but not in the elapsed time.
   - **llama.cpp perf counters:** classify a decode call as "eval" or "prompt eval" by batch size, not by phase.
   - **Prompt caching:** llama-server defaults to `cache_prompt: true`.
   - **Our approach:** one external harness with per-token timestamps.
7. **The usual memory metrics can be gamed, and accidentally are.** VERIFIED with our probe.
   - Weights reached through `mmap` + a Metal no-copy buffer that only the GPU reads appear in **none** of these: `phys_footprint` (Activity Monitor), RSS, `ru_maxrss`, `task_vm_info.external`, `footprint`, `vmmap` resident.
   - They do appear as **system-wide wired memory** and in `MTLDevice.currentAllocatedSize`.
   - GPU-private Metal memory is the opposite: it shows in `phys_footprint` but not in RSS.
   - The proposed honest metric is in section 6.2.
8. **Quality equivalence must use a teacher-forced comparison** against a BF16 reference: KLD, top-1 agreement, Δp. Greedy token-match is only a smoke test, because bf16 near-ties flip.
   - llama-perplexity's KLD stores the base log-probs as uint16 and ignores p < e⁻¹⁶ (`tools/perplexity/perplexity.cpp:191-240`).
   - It also needs huge files: 37 GiB for Llama 3 on Wikitext-2 (README).
   - So we need our own compact, cross-runtime logit dump.
9. **The HF cache changed two weeks ago.** VERIFIED by R9-HUB.
   - huggingface_hub 1.32 added a cross-repo shared blob store, on by default and already live on this machine. It keys payloads by the **Xet hash**, not the sha256, and makes them read-only.
   - Consequences: `du` on a repo reports 4.3 MB for a 4.6 GB model; copying without dereferencing symlinks gives broken files.
   - Sky must read the cache, never write store payloads, and pin hashes in a lockfile.
10. **Config-schema drift silently breaks loaders.** VERIFIED by R9-ARCH.
    - transformers 5 moves `rope_theta`/`rope_scaling` into `rope_parameters`.
    - mlx-lm drops unknown keys without a warning: a re-saved Llama 3.1 config loads with θ = 1e4 and no llama3 scaling.
    - Sky's config loader must normalise both schemas and fail loudly.

---

## 1. safetensors

### 1.1 Exact format

VERIFIED; sources: https://github.com/huggingface/safetensors README "Format"; reference implementation `safetensors/src/tensor.rs` (fetched to `core/st_tensor.rs`).

**File layout:**
- 8 bytes: `N`, a u64 little-endian header size.
- N bytes: UTF-8 JSON header.
  - "MUST begin with `{`" and "MAY be trailing padded with whitespace (0x20)".
  - Content: `{"NAME": {"dtype": "F16", "shape": [...], "data_offsets": [BEGIN, END]}, ...}` plus an optional `"__metadata__": {str: str}`.
- The data buffer.
  - Offsets are relative to the start of the buffer.
  - END is exclusive.
  - "The byte buffer needs to be entirely indexed, and cannot contain holes."
  - Little-endian, row-major.
- The header is limited to 100 MB (`MAX_HEADER_SIZE = 100_000_000`, `tensor.rs:10`; MLX applies the same limit in `mlx/io/safetensors.cpp:126`).

**dtypes:**
- `Dtype` enum in `tensor.rs:808-861`, "MUST be in increasing alignment order":
  - BOOL, F4 (MX FP4);
  - F6_E2M3, F6_E3M2;
  - U8, I8;
  - F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ;
  - I16, U16, F16, BF16;
  - I32, U32, F32, C64, F64, I64, U64.
- Sub-byte dtypes (F4, F6) must total a whole number of bytes (`MisalignedSlice`, `tensor.rs:647-653`).
- MLX reads and writes only the subset in `mlx/io/safetensors.cpp:18-31`, which includes F8_E4M3 and F8_E8M0.

**Reader validation** (`Metadata::validate`, `tensor.rs:626-660`):
- The reader sorts tensors by `data_offsets`.
- It requires each tensor to start exactly where the previous one ended.
- It checks `END-BEGIN == numel·bits/8`.
- It requires `8+N+buffer_end == file_size` (`:420`).
- It does **not** check any alignment.

**Writer** (`prepare`, `tensor.rs:218-262`):
- Sorts by "descending dtype alignment, then by name" (`:227-233`).
- "Force alignment to 8 bytes": pads the JSON with spaces to a multiple of 8 (`:253-255`).

Consequences, all VERIFIED by reading the code:
- With the HF writer, the data start (8+N) is 8-aligned.
- Each tensor's offset is a multiple of its own element size, because earlier tensors have equal or larger power-of-two element sizes.
- **The maximum guaranteed alignment is therefore 8 bytes, and only for 8-byte dtypes.** U32 tensors are 4-aligned, BF16 tensors 2-aligned. Nothing guarantees 16-byte alignment.

### 1.2 What MLX actually writes, checked on real files

**MLX's own writer** (`mlx/io/safetensors.cpp:220-275`): VERIFIED by reading the code.
- No header padding: `header_len = header.length()`.
- Tensors are emitted in `std::unordered_map` iteration order.
- The header is a `nlohmann::json`, so its keys are sorted alphabetically.
- Result: data order ≠ header order, and there is no alignment.
- mlx-lm's `save_model` calls `mx.save_safetensors(..., metadata={"format":"mlx"})` (`mlx_lm/utils.py:885-940`).

**The check.** Command: `$S/venv/bin/python core/st_inspect.py <snapshot>/model.safetensors`. Full outputs are in `core/st_inspect_*.txt`.

| | Qwen3-8B-4bit (commit 545dc42) | Qwen3-0.6B-4bit (commit 73e3e38) |
|---|---|---|
| file size | 4,607,835,174 | 335,450,584 |
| header N | 103,454 (N%8 = 6) | 78,288 (N%8 = 0, by luck) |
| data start 8+N | 103,462: ≡6 mod 8, ≡38 mod 64, ≡1062 mod 4096 | 78,296: ≡8 mod 16 |
| trailing header spaces | 0 | 0 |
| `__metadata__` | `{"format":"mlx"}` | same |
| tensors | 907 (653 BF16, 254 U32) | 704 (507 BF16, 197 U32) |
| bytes | U32 3.814 GiB, BF16 0.477 GiB | U32 0.278 GiB, BF16 0.035 GiB |
| contiguous, no holes | yes | yes |
| data order follows header order | **no** | **no** |
| tensors whose absolute offset is ≥4-aligned | **0 / 907** (all are 2-aligned) | 704/704 (8-aligned) |
| tensors whose absolute offset is 16-aligned | **0 / 907** | **0 / 704** |
| relative offset (from data start) 16-aligned | 907/907 | 704/704 |

Notes on the table:
- Relative offsets are 16-aligned simply because every tensor size happens to be a multiple of 16.
- **Implication:** if Sky copies the data buffer to a page-aligned allocation (one `pread` of the whole buffer), every tensor becomes ≥16-aligned for these models. If Sky maps the file, they do not.

**Tensor naming.** Each quantized `Linear` or `Embedding` at path `p` stores three tensors:
- `p.weight`: U32, shape `[out, in·bits/32]`;
- `p.scales`: BF16, `[out, in/group]`;
- `p.biases`: BF16, `[out, in/group]`.

**Example, layer 0 of Qwen3-8B:** `q_proj.weight` is `U32 [4096,512]`; scales and biases are `[4096,64]`; `mlp.down_proj.weight` is `U32 [4096,1536]`.

**Other layout facts:**
- `lm_head` and `embed_tokens` are both 4-bit, group 64. The 8B is untied, so `lm_head` is present; the 0.6B is tied, so it has no `lm_head`.
- Norms (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm`) stay BF16.
- The MLX `.biases` (zero-points) must not be confused with a Linear's `.bias`. R9-ARCH found both names in the same files for gpt-oss.

### 1.3 Zero-copy on Metal: rules and our GPU test

**Apple docs (VERIFIED, developer.apple.com JSON docs fetched 2026-10-01):**
- `makeBuffer(bytesNoCopy:length:options:deallocator:)`:
  - "A page-aligned pointer to the starting memory address."
  - The length must give "a page-aligned region of memory".
  - "The existing memory allocation needs to exist within a single virtual memory (VM) region."
  - Pages are 16 KiB here (`vm_page_size=16384`).
- `setBuffer(_:offset:index:)`:
  - "For buffers binding to an argument using the `device` address space, align the offset to the data type's size. The maximum size for an offset is 16 bytes."
  - The alignment needed for `constant` buffers depends on the GPU family.
- `maxBufferLength` is "at least 256 MB". It is 8.88 GiB on this 16 GB M2 Pro, so a 19 GB model does not fit in one buffer here. On large-memory machines it is bigger (LIKELY; check on the M5 Max). llama.cpp splits big mappings into several overlapping views.

**Our test.** Source: `core/memprobe/memprobe.m`.
- Build: `clang -O2 -fobjc-arc -framework Foundation -framework Metal`.
- Run: `./memprobe mem1g.bin alignonly`.
- Setup: a 64 KiB page-aligned no-copy buffer holding bytes 0..255; we bind it at offset *o* and load.

| Offset | `uint` load | `uint4` load | byte loads |
|---|---|---|---|
| 0 | OK | OK | OK |
| 2 | **0x03020100 instead of 0x05040302**: rounded down, silent | wrong | OK |
| 4 | OK | OK, but API-invalid | OK |
| 6 | **0x07060504 instead of 0x09080706**: silent | wrong | OK |
| 8 | OK | OK, but API-invalid | OK |

With `MTL_DEBUG_LAYER=1`:
- `failed assertion 'Compute Function(ld32): the offset into the buffer a … must be a multiple of 4 but was set to 2'`;
- for `uint4`, `'… must be a multiple of 16 but was set to 4'`.

VERIFIED (M2 Pro, GPU family Apple 8). The tolerance of `uint4` loads at 4-aligned offsets is **undefined behaviour**. Do not rely on it on M5 (LIKELY to vary).

**Design consequences for Sky:**

| Option | Assessment |
|---|---|
| (a) `pread`/copy into aligned `MTLBuffer`s, as MLX does | Simple. Costs one full read and makes the weights anonymous, footprint-charged memory. VERIFIED that MLX does this: `Load::eval_cpu` → `allocator::malloc` + `pread`, `mlx/backend/common/load.cpp:30-55`, `mlx/io/load.cpp:409-430`. |
| (b) First-run **relayout cache** | Recommended. Write a Sky-native file keyed by the source sha256 (e.g. `~/.cache/sky/relayout/<sha256>/`), with every tensor 16 KiB-aligned (or at least 256 B), weights pre-interleaved for Sky's kernels, then `mmap` + `bytesNoCopy`. This gives the fastest cold start after the first run, the same file can serve several processes, and Sky owns the layout. Cost: a second ~19 GB copy on disk for Qwen3-32B. EXPERIMENTAL. |
| (c) Map the original file and read with byte or `ushort` loads | Zero copy, but all mlx-community tensors are only 2-byte aligned, and it limits kernel vector widths. Kernel throughput impact unknown. EXPERIMENTAL. |
| (d) GGUF | Map directly. 32-byte alignment is fine for `uint4` loads. VERIFIED. |

### 1.4 MLX quantized checkpoint layout

All VERIFIED from source, plus our check on real data.

**The quantizer** (`mlx/ops.cpp:5122-5233`, `affine_quantize`):
- Group sizes 32/64/128; bits 2, 3, 4, 5, 6, 8.
- `scale = (max−min)/(2^b−1)` (minimum 1e-7).
- The sign trick: choose `edge` = whichever of max and min has the larger magnitude, `q0 = round(edge/scale)`, `scale = edge/q0`, `bias = edge`.
  - This makes one edge exactly representable.
  - Scales can be negative, and `bias` is then the group **max**, not the min.
- Values are stored as `q ∈ [0, 2^b−1]`.
- Dequantize: `w = scale·q + bias`.
- Scales and biases are stored in the weight dtype.

**Packing** (`pack_and_quantize`, `ops.cpp:5122-5168`):
- Power-of-two bit widths: element *j* of each `uint32` sits in bits [b·j, b·(j+1)), least significant first.
- 3, 5 and 6 bits: a little-endian bit stream across `uint32`s, so a value can straddle words.

**Check on real data** (`core/mlx_q_check.py`, on `model.layers.0.self_attn.q_proj`):
- LSB-first numpy dequant vs `mx.dequantize`: max |Δ| = 0.00195, which is bf16 half-ulp rounding of MLX's bf16 output.
- An MSB-first decoding gives max |Δ| = 0.55, i.e. clearly wrong.
- Whole model:
  - 127,975,424 scales, 50.25 % negative;
  - |scale| range [1.52e-5, 0.208];
  - 0 scales and 0 biases not exactly representable in fp16.

**Modes** (`mlx_lm/utils.py:972-979`; `fp_quantize` in `ops.cpp:5235-5345`):

| mode | group | bits | Values | Scales |
|---|---|---|---|---|
| `affine` | 64 (default; 32/128 allowed) | 4 (default; 2–8) | as above | bf16 scale + bias |
| `mxfp4` | 32 | 4 | E2M1 LUT {0,.5,1,1.5,2,3,4,6, −0,…,−6}; 8 codes per `uint32`, LSB-first | `uint8` E8M0 = exponent+127, rounded **up** so the block maximum stays representable; no biases |
| `nvfp4` | 16 | 4 | E2M1 | E4M3 + optional global scale |
| `mxfp8` | 32 | 8 | E4M3 viewed as `uint32` | E8M0 |

**Config entries** (`utils.py:470-560`):
- `config["quantization"] = {"group_size", "bits", "mode"?}` is the default. `mode` is absent in older files and means affine.
- An identical `"quantization_config"` copy is kept "for the HF model tree" (#957).
- **Per-layer overrides** are keys equal to the module path, holding a dict of params or `false`. Example: `"model.layers.3.mlp.down_proj": {"group_size": 64, "bits": 6}`.
  - `class_predicate` checks `p in config["quantization"]` first (`:487-492`).
- For unlisted modules that have `.scales`, `infer_quant_config` (`:380-405`) recovers the parameters from shapes:
  - `group = in_dims // scales.shape[-1]`;
  - `bits = weight.shape[-1]·32 // in_dims`;
  - the mode is affine if the scales are not `uint8`; otherwise (4,16) means nvfp4, (4,32) mxfp4, (8,32) mxfp8.
  - Sky should implement exactly this inference, and fail on anything ambiguous.
- Legacy `quantization_config.quant_method` values are converted at load time (`:509-547`):
  - `mxfp4` (gpt-oss), `mxfp8`, `fp8` with ue8m0 32×32 blocks;
  - `compressed-tensors`;
  - `awq`/`gptq`, which are repacked;
  - `bitnet`.
- mlx-community Qwen3-8B-4bit has `{"group_size":64,"bits":4}` and no per-layer entries. It was converted with mlx-lm 0.24.0 (README). Its total size of 4,607,731,712 B for 8,190,735,360 parameters is 4.5004 bpw.
- **mxfp4 differs between MLX and GGUF.**
  - GGUF `block_mxfp4` interleaves the E8M0 byte with 16 bytes of codes.
  - GGUF nibble order is low = element *j*, high = element *j+16*.
  - MLX stores the scales as a separate tensor, with sequential nibbles.
  - The two are losslessly convertible.

---

## 2. GGUF

### 2.1 Container

VERIFIED: `ggml/include/gguf.h:1-46`; reader `ggml/src/gguf.cpp`.

**Header:**
- magic `"GGUF"`;
- `u32` version (3);
- `i64` n_tensors;
- `i64` n_kv.

**Key/value pairs** (`string key, i32 type, value`):
- Types: u8, i8, u16, i16, u32, i32, f32, bool (`i8`), string (`u64` length + bytes, no NUL), array (`i32` element type, `u64` count, elements), u64, i64, f64.

**Tensor info** (repeated):
- name, `u32` n_dims (≤4), `i64` ne[], `i32` ggml_type, `u64` offset within the data blob.

**Data blob:** begins at `GGML_PAD(header_end, alignment)`.

**Alignment:**
- `alignment = general.alignment` (must be `u32` and a power of two), default **32** (`gguf.cpp:622-636`).
- The reference reader is stricter than "offsets are multiples of the alignment": tensor *i*'s offset must **equal** the padded sum of all earlier tensor sizes (`gguf.cpp:790-815`). Tensors are stored in info order, each padded to the alignment.
- Alignment is relative to the start of the GGUF, not the file (`:371`), which matters for embedded GGUFs.

**Validation:**
- duplicate names are rejected;
- names are limited to `GGML_MAX_NAME`;
- `ne[0] % block_size == 0` is enforced (`:640-760`).

### 2.2 Block layouts

VERIFIED: `ggml/src/ggml-common.h:181-370`; reference dequant in `ggml/src/ggml-quants.c`.

Type ids come from `ggml.h:390-433`. New in 2026: NVFP4 = 40, Q1_0 = 41, Q2_0 = 42. The Q4_0_4_4-style repack ids are gone.

| Type (id) | Block | Bytes | bpw | Struct | Dequant |
|---|---|---|---|---|---|
| Q4_0 (2) | 32 | 18 | 4.5 | `{f16 d; u8 qs[16]}` | `x[j]=(qs[j]&0xF)-8)*d`, `x[j+16]=((qs[j]>>4)-8)*d` (`ggml-quants.c:459-476`): split-half nibbles, symmetric |
| Q4_1 (3) | 32 | 20 | 5.0 | `{f16 d, m; u8 qs[16]}` | `d·q + m`, same nibble order |
| Q8_0 (8) | 32 | 34 | 8.5 | `{f16 d; i8 qs[32]}` | `qs[j]·d` (`:553-567`) |
| Q4_K (12) | 256 | 144 | 4.5 | `{f16 d, dmin; u8 scales[12]; u8 qs[128]}` | 8 sub-blocks of 32. Six-bit scale and min packed into 12 bytes (`get_scale_min_k4`, `:880-887`). `y = d·sc·q − dmin·m`. Each 64-element chunk uses the low nibbles for the first 32 and the high nibbles for the next 32 (`:1529-1550`). |
| Q6_K (14) | 256 | 210 | 6.5625 | `{u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}` | `q = (ql nibble | 2 bits of qh << 4) − 32`. `y = d·scales[is]·q`, in 128-element halves with a 4-way interleave (`:1939-1968`). |
| MXFP4 (39) | 32 | 17 | 4.25 | `{u8 e (E8M0); u8 qs[16]}` | `kvalues_fp4 = {0,1,2,3,4,6,8,12,0,-1,…,-12}` (doubled E2M1) × `GGML_E8M0_TO_FP32_HALF(e)` = 2^(e−128) (`ggml-common.h:1124-1129`, `ggml-impl.h:489-510`). Split-half nibbles. |
| NVFP4 (40) | 64 | 36 | 4.5 | `{u8 d[4] UE4M3; u8 qs[32]}` | 4 sub-blocks of 16 |

### 2.3 Real files

Command:
```
PYTHONPATH=$S/src/llama.cpp/gguf-py $S/venv/bin/python core/gguf_inspect.py <file> <gguf-py>
```
Outputs are in `core/gguf_inspect_*.txt`.

| File | data_offset | % 32 / 64 / 4096 / 16384 | Tensors | Types | bpw |
|---|---|---|---|---|---|
| unsloth Qwen3-0.6B-Q4_0 | 5,952,192 | 0 / 0 / 704 / 4800 | 310 | 193 Q4_0, 3 **Q4_1**, 1 **Q6_K** (token_embd, tied), 113 F32 | — |
| bartowski Qwen3-8B-Q4_0 (sha256 c9bd6e55…, commit 0b69f75) | 5,957,152 | 0 / 32 / 1568 / 9760 | 399 | 249 Q4_0, 4 **Q4_1** (`ffn_down`, layers 0–3), 1 **Q6_K** (`output.weight`), 145 F32 | **4.67** |
| Qwen/Qwen3-8B-GGUF Q4_K_M (sha256 d98cdcbd…) | 5,956,416 | 0 / 0 / 832 / 9024 | 399 | 217 Q4_K, 37 Q6_K, 145 F32 | **4.90** |

- In every file, all tensors are 32-aligned and none is 4096-aligned. No file sets `general.alignment`.
- The two imatrix files carry `quantize.imatrix.*` keys.
- The Q4_K_M's `general.name` is "Qwen3 8B Awq Compatible Instruct" with finetune "awq-compatible-Instruct". It is not obviously a conversion of `Qwen/Qwen3-8B`, so its provenance is uncertain.

**Why "Q4_0" is mixed precision** (VERIFIED, `src/llama-quant.cpp`):
- The output tensor, or `token_embd` when tied, becomes Q6_K for most file types unless `--output-tensor-type` or `--pure` is given (`:488-510`). Exceptions: a dimension that is not a multiple of the block size goes to Q8_0, and MXFP4_MOE also uses Q8_0.
- With an imatrix, Q4_0/Q5_0 upgrade `ffn_down` in the first `n_layer/8` layers to Q4_1/Q5_1 (`:661-666`).
- `--pure` disables both (`:764`).

**For Qwen3-32B** (computed in Python; the parameter counts are VERIFIED by R9-ARCH):

| Variant | Weight bytes |
|---|---|
| MLX 4-bit group 64, bf16 scales | 18,429,667,328 B (matches the mlx-community index) |
| Q4_0 with Q6_K output | 18,631,576,576 B (+1.1 %) |
| + imatrix Q4_1 `ffn_down` | +1.45 % |
| Q4_K_M | about +9 % (LIKELY, scaled from the 8B) |

- Decode weight bytes per token: MLX 17.99 GB, Q4_0 18.19 GB.
- The bf16 KV cache costs 262,144 B per token of context.

### 2.4 Tokenizer metadata in GGUF

VERIFIED from the 8B Q4_0 file:
- `tokenizer.ggml.model = "gpt2"` (byte-level BPE) and `tokenizer.ggml.pre = "qwen2"`. The `pre` value selects llama.cpp's hard-coded pre-tokenizer regex, not the regex string from `tokenizer.json`.
- `tokens[151936]`, which includes padding entries up to the embedding size.
- `token_type[]` and `merges[151387]`.
- `eos_token_id = 151645`, `bos_token_id = padding_token_id = 151643`, `add_bos_token = false`.
- `tokenizer.chat_template` is a 4,614-character Jinja string. It differs from the 4,905-character template in the unsloth 0.6B file, so publishers patch templates.
- **Pitfall:** the GGUF carries one EOS. HF's `generation_config.json` has a list (Qwen3: [151645, 151643]).

### 2.5 Exact cross-format conversions

VERIFIED arithmetic; the fp16 exactness is VERIFIED on the Qwen3-8B data.

- **MLX affine group-64 bf16 → GGUF Q4_1:**
  - Split each group into two 32-element blocks with the same `d=scale` and `m=bias`, and permute nibbles into split-half order.
  - Exact iff every scale and bias is fp16-representable, which holds for Qwen3-8B-4bit.
  - Cost: 5.0 instead of 4.5 bpw, +11 % bytes.
- **GGUF Q4_0 → MLX affine group-32 with fp16 scales:** `scale=d`, `bias=−8d`, which is exact. Cost: 5.0 bpw on the MLX side.
- **No exact conversion exists between the 4.5-bpw types** (MLX g64 ↔ Q4_0/Q4_K).
- **Conclusion:** identical-weights comparisons are only possible **within one file format** without a size penalty. That is why Sky must run both formats natively (section 7).

---

## 3. HF `config.json` → architecture (summary of R9-ARCH; full notes in `R9-work/arch/R9-arch.md`)

All entries are VERIFIED by R9-ARCH from 31 fetched configs and code citations, unless marked otherwise.

**Sources.**
- **Configs:** in `R9-work/arch/configs/`. Gated meta-llama and google repos were read via `unsloth/*` mirrors, which carry `"unsloth_fixed": true`.
- **Code:** mlx-lm `53b9af3`, llama.cpp `81e39ad`, transformers 5.18.0.

**Abbreviations:**
- **pre** — pre-norm.
- **NeoX** — rotate-half RoPE.
- **dc** — head_dim decoupled, meaning heads × head_dim ≠ hidden.
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

### 3.1 Gotchas a model compiler must handle

Citations are in `arch/R9-arch.md` §2.

**RoPE layout is per file, not per model.**
- llama.cpp permutes q/k for Llama and Mistral at conversion (`conversion/llama.py:17-20,176-181,260-264`) and then uses NORM (interleaved) RoPE. Qwen, Phi, Gemma3 and gpt-oss use NEOX (`src/llama-model.cpp:3061-3186`).
- GGUF Gemma norms already have the +1 folded in (`conversion/gemma.py:131-132`).
- A Sky GGUF loader must carry a per-file layout flag rather than assume HF layout.

**Config schema drift.**
- transformers 5 changes several keys:
  - `rope_theta`/`rope_scaling` become `rope_parameters`;
  - `torch_dtype` becomes `dtype`;
  - `layer_types` is new;
  - Gemma's RoPE becomes nested per layer type.
- mlx-lm silently drops unknown keys (`models/base.py:13-21`). A re-saved Llama 3.1 config loads with θ = 1e4 and no llama3 scaling; a Gemma 3 global RoPE scale comes out as 1.0 instead of 0.125.
- Only 16 of 135 mlx-lm model files read `rope_parameters`.
- **Sky must normalise both schemas and fail on unknown or missing keys.**

**Defaults when a key is absent.**
- mlx-community's Gemma-3-27B text config has only 10 keys.
- mlx-lm's llama/qwen2 default to tied embeddings; HF's LlamaConfig defaults to untied. This is a risk only when the key is absent (LIKELY).

**Inert keys.**
- Qwen's `sliding_window` is ignored unless `use_sliding_window` is true. R1-Distill-Qwen-7B ships 4096.
- Phi-4-mini ships `sliding_window` 262144.

**Runtimes disagree numerically with HF.**
- **gpt-oss YaRN `truncate=false`:** mlx-lm (`rope_utils.py:150-153`) and llama.cpp (`ggml.c:4482-4490`) floor/ceil the correction range anyway.
- **Phi LongRoPE short/long factor choice:** HF picks by the current length; llama.cpp by `n_ctx`; mlx-lm always uses long.
- **Gemma 3 embedding scale:** HF and mlx-lm round 73.32 → 73.5 in bf16; llama.cpp uses fp32 (`gemma3.cpp:91`).
- **Gemma 3 attention scale:** llama.cpp hard-codes it by model size (`gemma3.cpp:28-31`).

**Default numerics.**
- mlx-lm: bf16 activations and a bf16 KV cache.
- llama.cpp: f32 activations and an f16 KV cache (`common/common.h:343-344`).
- These must be stated or matched in benchmarks. KV dtype changes both bandwidth and quality.

**Checkpoint traps.**
- Fused weights are split differently:
  - Phi `gate_up` is split into halves;
  - gpt-oss `gate_up` is interleaved even/odd (`models/gpt_oss.py:226-237`).
- gpt-oss MXFP4 is stored as `U8 blocks [E,out,in/32,16]` plus E8M0 scales, low nibble = even element. Its 3 shards are named `-of-00002`.
- Qwen3-MoE experts are stored unstacked: 18,867 tensors.
- Stop tokens are a list in `generation_config.json`; Qwen3 has [151645, 151643].
- mlx-lm's quantized-KV attention rejects sinks (`base.py:119-121`).
- Qwen3 YaRN must never be enabled in only one runtime. The model card says not to use static YaRN below 32k.

### 3.2 Feature tiers for Sky

**Tier 0 (Qwen3-32B):**
- config normalisation and stop-token lists;
- quantized embedding gather;
- fp32-accumulated RMSNorm, including per-head QK-norm (dim 128, before RoPE);
- 4-bit affine group-64 GEMV/GEMM with bf16 scale and bias (fused QKV and fused gate+up+SwiGLU);
- NeoX RoPE with θ 1e6 and fp32 angles;
- GQA-8 attention with head_dim 128 (decode kernel, flash prefill, KV append);
- LM head and sampling.

**Tier 1:**
- QKV bias;
- llama3, linear, YaRN and LongRoPE scaling; partial rotary;
- tied quantized LM head;
- fused-weight splitting;
- GQA ratios 3, 5, 7 and head_dim 64/256;
- GGUF layout flags.

**Tier 2:**
- per-layer sliding windows with a rotating cache;
- Gemma (1+w) norms, sandwich norms, GeGLU and scales;
- sinks and softcap;
- MoE routing and grouped GEMM (softmax→top-k→renorm for Qwen3-MoE; top-k→softmax for gpt-oss; sigmoid + bias with groups for DeepSeek);
- MXFP4 and FP8 dequant;
- MLA.

**Newer files in mlx-lm, not deep-dived** (the files exist; their semantics are LIKELY as named): `qwen3_next`/`qwen3_5` (Gated DeltaNet), `gemma3n`/`gemma4_text`, `llama4_text`, `deepseek_v32`/`v41` (sparse attention), `mistral4`, `ministral3`, `kimi_k3`.

### 3.3 Qwen3-32B numbers

VERIFIED from the config and arithmetic.

**Config:** 64 layers, hidden 5120, 64 query and 8 KV heads, head_dim 128, intermediate 25600, vocab 151936, θ 1e6, `max_position_embeddings` 40960, untied, bf16, eps 1e-6.

**Size:**
- 32,762,123,264 parameters.
- MLX 4-bit group 64: 18,429,667,328 B, which equals mlx-community's `total_size`.

**Decode, per token:**
- 17.99 GB of weight reads; the LM head is 2.4 % of that.
- 63.97 GFLOP of matmuls, plus 2,097,152·T FLOP of attention at context T.
- The bf16 KV cache costs 256 KiB per token.
- Decode is bandwidth-bound. Roofline: tok/s ≤ BW / (17.99e9 + 262,144·T). Get the M5 Max BW figure from the hardware agent; do not assume it.


---

## 4. Tokenizers, chat templates, Rust crates (R9-TOK)

Source: R9-TOK. Evidence is in `R9-work/tok/`: `corpus.py`, `tok_test.py`, `llamacpp_tok.py/.out`, `special_literal.py/.out`, `tpl_test.py`, `tpl_results.json`, `crates.json`, and the Rust test crate `rs/`.

Versions: transformers 5.18.0, tokenizers 0.23.2, minijinja-py 2.24.0, llama.cpp `81e39ad`.

### 4.1 tokenizer.json structure (VERIFIED)

**Qwen3-8B (the actual mlx-community file):**
- **Model:**
  - `BPE`, `byte_fallback=false`, `ignore_merges=false`.
  - 151,643 vocab entries and 151,387 merges.
  - Merges are stored as `[a,b]` pairs. Older Qwen2.5 files use `"a b"` strings, so parsers must accept both.
- **Pipeline:**
  - normalizer `NFC`;
  - pre-tokenizer `Split(regex, Isolated)` → `ByteLevel(add_prefix_space=false, use_regex=false)`;
  - post-processor `ByteLevel` (no BOS template);
  - decoder `ByteLevel`.
- **Regex:** `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
  - Digits are split one at a time.
  - The `\s+(?!\S)` lookahead needs a backtracking engine: Oniguruma or fancy-regex.
- **Added tokens:** 26, ids 151643–151668.
  - 14 have `special=true`: `<|endoftext|>` 151643, `<|im_start|>` 151644, `<|im_end|>` 151645, and the vision tokens.
  - 12 have `special=false`: `<tool_call>` 151657, the fim tokens, `<tool_response>`, `<think>` 151667, `</think>` 151668.
- **Vocab size:** 151,669 ids are used, but `vocab_size` is 151,936. That leaves 267 padded embedding rows; GGUF stores them as `[PAD…]` with type 5.
- **tokenizer_config:** `add_bos_token=false`, `bos=null`, `eos=<|im_end|>`, `split_special_tokens=false`.

**Other families:**

| Family | Pre-tokenizer | ignore_merges | Tokenizer adds BOS | EOS list (generation_config) | ids used / vocab_size |
|---|---|---|---|---|---|
| Qwen3 / Qwen2.5 | regex above, `\p{N}`, NFC | false | no | [151645, 151643] | 151,669 / 151,936 |
| Llama-3.1, R1-Distill-Llama | same regex but `\p{N}{1,3}`, no normalizer | **true** | yes | [128001, 128008, 128009] | 128,256 / 128,256 |
| gpt-oss (o200k_harmony) | case-aware `\p{Lu}…\p{M}`, `\p{N}{1,3}` | true | no | [200002, 199999, 200012] | 200,019 / 201,088 |
| Phi-4 | cl100k-style | – | no | [100257, 100265] | 100,352 |
| Gemma-3 | SentencePiece BPE: `Replace(" "→"▁")`, no regex, byte_fallback; `.model` has `add_dummy_prefix=false`, `split_digits=true` | false | yes (2) | [1, 106] | 262,145 / 262,208 |
| Mistral-7B-v0.3 | Metaspace `prepend_scheme=first`, byte_fallback | false | yes | 2 | 32,768 |
| Mistral-Small-3.2 | byte-level BPE, but the Hub file carries the **Llama-3 regex** instead of Tekken's (transformers warns and offers `fix_mistral_regex`) | true | yes | 2 | 131,072 |

**mlx-community/Qwen3-8B-4bit has no `generation_config.json`.** mlx-lm builds its EOS set from config, generation_config and the tokenizer (`utils.py:366-376`, `tokenizer_utils.py:358-360`). With this repo it therefore stops only on 151645, whereas the Qwen original also stops on 151643. **Throughput runs must use a fixed N with EOS ignored in every runtime.**

### 4.2 Exactness results on a 28-string adversarial corpus

The corpus covers whitespace, digits, contractions, CJK/RTL, ZWJ emoji, NFC/NFD, zero-width and control characters, code/URLs, and literal special-token strings.

**Where the tokenizers agree:**
- Python `tokenizers` vs transformers (Qwen3): 28/28 identical. VERIFIED.
- Rust `tokenizers` 0.23.2 built pure-Rust (`default-features=false`, fancy-regex) vs the Python wheel (Oniguruma): **168/168 identical** (6 tokenizers × 28 strings). It builds on cargo 1.84.1 with a 90-crate lockfile. VERIFIED.

**HF vs llama.cpp** (`llama-server /tokenize`, Qwen3-0.6B Q4_0 GGUF): 26/28 identical. VERIFIED.
- Both differences are NFD inputs. **llama.cpp skips the NFC normalizer**: NFD "café naïve Å" gives 5 ids in HF and 11 in llama.cpp; NFD Hangul gives 3 vs 8.
- llama.cpp also never reads the HF regex. The converter fingerprints the tokenizer by a hash of token ids (`conversion/base.py:1629`), and the hash selects a hard-coded regex, e.g. `qwen2` at `llama-vocab.cpp:383-391`. VERIFIED.

**Special tokens typed by a user:**
- By default HF turns a typed `<|im_end|>` into control id 151645, which is an injection vector.
- `split_special_tokens=True` splits only `special=true` tokens.
- llama.cpp `parse_special=false` matches HF except for `<|fim_prefix|>`, which the GGUF converter re-typed as CONTROL: 7 tokens vs 1.
- `/tokenize` defaults to `parse_special=true` (`server-context.cpp:5090`).

All VERIFIED:
- **`ignore_merges`:** Llama-3.1 has **588 vocab tokens reachable only with `ignore_merges=true`** (e.g. `" января"`). There was no difference on 49k natural tokens, so it is rare but real.
- **Double BOS:** encoding a rendered template with `add_special_tokens=True` gives `[128000,128000,…]` for Llama-3.1, `[2,2,…]` for Gemma-3 and `[1,1,…]` for Mistral-Small-3.2. The raw `tokenizers` library ignores tokenizer_config's `add_bos_token`.
- **R1-Distill-Qwen BOS:** 151643 in `config.json`, but 151646 in the tokenizer and in generation_config.
- **transformers v5 regression:** it aliases `LlamaTokenizerFast` to `LlamaTokenizer` (`tokenization_llama.py:149`), which rebuilds a SentencePiece Metaspace pipeline.
  - Effect: "The quick brown fox" becomes "Thequickbrownfox" (13 ids vs 11) for `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` **and** its mlx-community 4-bit copy.
  - mlx-lm requires transformers ≥ 5.7 and uses AutoTokenizer, so it is LIKELY affected; not run end to end.
  - **Never use v5 AutoTokenizer as the golden reference for repos that declare `LlamaTokenizer(Fast)`.**
- **Streaming detokenization:** Qwen3 tokens can end mid-codepoint (id 61804 = `b' \xf0\x9f\x91'`). Per-token decoding of a ZWJ family emoji plus a flag yields 11 U+FFFD characters.
  - mlx-lm buffers on a trailing U+FFFD (`tokenizer_utils.py:249`); llama-server uses `validate_utf8` (`server-common.cpp:903`).
  - Sky should buffer bytes to a UTF-8 boundary.
- **`clean_up_tokenization_spaces=true`** (Llama-3.1) is ignored for BPE by transformers 5.18, which calls it "destructive"; v4 honoured it. **Compare ids, never decoded strings.**
- **SentencePiece `.model` vs tokenizer.json:** Gemma-3 0/28 differ; Mistral-7B-v0.3 4/28 differ (all with leading whitespace).

### 4.3 Chat templates (VERIFIED by rendering)

**Where the template lives:**
- `chat_template.jinja` (and `additional_chat_templates/`) take precedence over `tokenizer_config.json["chat_template"]` (`tokenization_utils_base.py:1784`).
- gpt-oss has no template in tokenizer_config at all.
- unsloth's Qwen3 GGUF embeds a rewritten template (4,905 vs 4,116 characters) with one semantic change.

**How transformers renders** (`chat_template_utils.py:487-500`):
- a sandboxed environment with `trim_blocks` and `lstrip_blocks`, plus the `loopcontrols` and `{% generation %}` extensions;
- `tojson` is overridden to `json.dumps(ensure_ascii=False)` with no HTML escaping;
- globals `raise_exception` and `strftime_now`.

**Qwen3 thinking:**
- `enable_thinking=false` appends `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n`. Undefined means thinking is on.
- mlx-lm passes `enable_thinking=True` by default (`tokenizer_utils.py:393`).
- Reasoning from earlier turns is stripped, except after the last real user query during a tool loop.
- The model card says not to use greedy decoding in thinking mode. For speed this is irrelevant (fixed-N, EOS ignored). For quality evaluation, use the card's sampling or the non-thinking mode.

**Non-deterministic templates:**
- gpt-oss injects "Current date: <today>".
- Mistral-Small-3.2's default system prompt uses `strftime_now`.
- Llama-3.1 hard-codes "26 Jul 2024".
- R1-Distill forces thinking (ends with `<｜Assistant｜><think>\n`).
- **Benchmarks must pin `strftime_now`, or better, exchange pre-tokenized prompts.**

**minijinja vs jinja2** (10 templates × 3 conversations, Qwen3 with thinking on and off: 33 cases):
- With **pycompat plus a Python-compatible `tojson`**, all 30 renderable cases were **byte-identical**. VERIFIED.
- The other 3 raise identically in both engines: Gemma-3 rejects the tool role, and the Mistral templates need 9-character tool-call ids. Those tool paths are untested.
- minijinja's built-in `tojson` breaks 5 tool cases: compact separators and HTML-escaped `<`/`&`.
- Without pycompat, Qwen3 fails on `.startswith` and R1-Distill on `.split`.
- In the compiled Rust crate, identity also needs `preserve_order` (otherwise map keys come out sorted) and a kwargs-aware `tojson` (serde prints `1e-5` as `0.00001`; Python prints `1e-05`).
- Remaining differences: printing a list gives a Python repr in jinja2 vs JSON in minijinja; `{% generation %}` is unsupported.
- llama.cpp ships its own C++ Jinja engine (`common/jinja/`). It is on by default and marks strings from message content so they cannot become special tokens.

### 4.4 Rust crates (crates.io API, VERIFIED)

| Crate | Version (date) | License | MSRV / edition | Required deps | Note |
|---|---|---|---|---|---|
| tokenizers | 0.23.2 (2026-09-03); 1.0.0-rc.2 (2026-09-21) | Apache-2.0 | 2018 | 24 | Default features pull C code (onig, esaxx). Use `default-features=false` + fancy-regex: exact (§4.2). |
| safetensors | 0.8.0 | Apache-2.0 | 1.80 | 4 | Zero-copy `deserialize(&[u8])`. |
| minijinja / minijinja-contrib | 2.24.0 (2026-08-12) | Apache-2.0 | 1.70 | 2 / 2 | Need `json`, `loop_controls`, `preserve_order`, `unicode`, and contrib `pycompat`. |
| hf-hub | 1.0.0 (2026-07-10) | Apache-2.0 | 2024 (Rust ≥ 1.85) | 21 (tokio, reqwest, hf-xet) | Its README says it "does not read environment variables" (R9-HUB found that it reads `HF_TOKEN`/`HF_HOME` in `constants.rs:7-14`; the version needs re-checking). Lockfile has 356 packages. |
| candle-core (`gguf_file`) | 0.11.0 | MIT/Apache | – | 18 | Heavy if you only need the parser. |
| ggus / gguf-rs / gguf-rs-lib / gguf | 0.5.1 / 0.1.8 / 0.3.0 / 0.1.2 | MIT | – | – | All weak or stale. Write our own ~200-line GGUF parser. |
| memmap2 | 0.9.11 | MIT/Apache | 1.65 | 1 | – |
| openai-harmony | 0.0.8 | Apache-2.0 | – | – | Oracle for gpt-oss rendering. |
| mistralrs-core | 0.8.1 | MIT | 1.88 | 86 | Precedent: Rust LLM engine on tokenizers + minijinja + contrib + hf-hub. |

**Recommendations:**
- `tokenizers` (fancy-regex build, pinned) on tokenizer.json, plus a union of EOS ids from config, generation_config and tokenizer_config.
- A golden-ids conformance gate built from the raw Python `tokenizers` library, with llama.cpp as a second oracle for GGUF models.
- minijinja + pycompat + a Python-exact `tojson`.
- Pin Rust ≥ 1.85, since this machine has 1.84.1.

---

## 5. The Hugging Face Hub (summary of R9-HUB; evidence in `R9-work/hub/`)

### 5.1 Resolve API

VERIFIED with curl and no token on 2026-10-01; code is huggingface_hub 1.33 unless stated.

**URL:** `{HF_ENDPOINT}/{""|datasets/|spaces/}{repo}/resolve/{quote(rev,safe="")}/{path}` (`file_download.py:289-291`).
- `rev` may be a branch, a tag, a full or short commit, or `refs%2Fpr%2FN`. The ref must be %-encoded: an unencoded `refs/pr/1` gives 404 `RevisionNotFound`.

**Headers:**
- `X-Repo-Commit` is always the full 40-hex commit, even for short-SHA requests. Record this in results.
- `X-Linked-Etag` is the sha256 for LFS/Xet files and the git-blob SHA-1 for plain files.
- `X-Linked-Size` is the file size.
- `X-Xet-Hash` is a Merkle hash, not the sha256.

**Redirects:**
- Plain files: 307 to `/api/resolve-cache/...`.
- Xet files: 302 to a pre-signed `us.aws.cdn.hf.co/xet-bridge-us/...` URL, which supports **Range**.
  - VERIFIED: the first 16 MiB of the 8B Q4_0 GGUF came back as 206 with `Content-Range: bytes 0-16777215/4787332640`.
  - Parsing that prefix as a sparse file with gguf-py works. The 16 MiB prefix files were deleted afterwards to save disk; the full-file parse is in `core/gguf_inspect_Qwen3-8B-Q4_0.txt`. **Remote header inspection is cheap.** huggingface_hub reads safetensors headers with `range: bytes=0-100000` (`hf_api.py:7088`).

**Metadata APIs:**
- `/api/models/{repo}/revision/{rev}?blobs=true` returns `sha` and siblings with `lfs.sha256` and size.
- `/api/models/{repo}/tree/{rev}?recursive=true` returns `oid`, `lfs.oid` and `xetHash`.
- For gated repos the metadata is public, but `lfs.oid` and `xetHash` are redacted to 64 `*` characters. **Validate that hashes are hex**, as llama.cpp does (`common/hf-cache.cpp:160-162`).

**Errors:**

| Case | Response |
|---|---|
| Gated repo without a token | 401, `X-Error-Code: GatedRepo` |
| Nonexistent repo | 401 "Invalid username or password", indistinguishable from a private repo |
| Missing branch | 404 `RevisionNotFound` |
| Missing file | 404 `EntryNotFound`, still carrying `X-Repo-Commit` |
| An invalid token on a public file | Still served |

**Rate limits** (unauthenticated, from headers): 3,000 resolves per 300 s and 500 API calls per 300 s.

### 5.2 Auth

Never put secrets in code.

**Token order** (VERIFIED, `utils/_auth.py:79-84,145-147`):
1. An OIDC exchange, if `HF_OIDC_RESOURCE` is set. This is CI "Trusted Publishers": a short-lived token, with no stored secret.
2. `HF_TOKEN`, or the legacy `HUGGING_FACE_HUB_TOKEN`.
3. The token file `HF_TOKEN_PATH`, default `$HF_HOME/token`, written by `hf auth login`.
4. A Colab secret.

**Rules for Sky:**
- Read the same sources, and never log the token.
- Strip `Authorization` on cross-host redirects (`file_download.py:1817-1821`).
- Use fine-grained read-only tokens.
- The public `./benchmark.sh` should use **ungated** repos only (Qwen3 is ungated), so no token is needed.

### 5.3 Xet

VERIFIED from docs and behaviour.

**What it is:**
- Hub storage is content-defined-chunk dedup: chunks grouped into xorbs of at most 64 MiB, plus shards and a content-addressed store. The spec is v1.1.0 at huggingface.co/docs/xet.
- Clients without Xet support get a single reconstructed URL from the "bridge" (docs: hub/xet/legacy-git-lfs).

**Throughput** (EXPERIMENTAL, one sample, this link, 409 MB file):

| Method | Throughput |
|---|---|
| Bridge, 1 stream | 23.7 MB/s |
| Bridge, 8 parallel Range requests | 45.4 MB/s |
| `hf_hub_download` with hf_xet | 29.6 MB/s |

**Recommendation:** Sky v1 does not need a native Xet client. Use HTTPS resolve plus parallel Range requests plus sha256 verification. The native protocol (xet-read-token, then `/v2/reconstructions/{xet_hash}`) only adds dedup and perhaps peak speed. Its dependency cost is large: the hf-hub 1.0.0 crate hard-depends on hf-xet, with 356 packages in its Cargo.lock.

### 5.4 Cache layout

VERIFIED on disk and in code. Sky must share this cache.

**Standard layout:**
- Each repo lives at `hub/models--{org}--{name}/` and contains:
  - `refs/<rev>` (holding the commit);
  - `blobs/<etag>`;
  - `snapshots/<commit>/<path>`, a symlink to `../../blobs/<etag>`;
  - `trees/<commit>.json` (tree cache, v1.22+: size, `blob_id`, `lfs_sha256`, `xet_hash`);
  - `.no_exist/<commit>/<path>`.
- Hub level: `.locks/models--…/<etag>.lock` (flock) and `CACHEDIR.TAG`.
- Partial downloads are named `blobs/<etag>.incomplete`.

**New shared blob store (v1.32.0, 2026-09-17, on by default):** `utils/_shared_blobs.py`.
- Payloads live at `hub/blobs/<xet[:2]>/<xet_hash>` and are **read-only** (0444).
- Bookkeeping:
  - `<xet>.refs` manifests, each line appended and fsync'd before the symlink is visible;
  - `<xet>.lock`;
  - a marker file `hub/blobs/.huggingface-shared-blobs` containing `1\n`.
- Each per-repo `blobs/<sha256>` becomes a symlink to `../../blobs/xx/<xet>`.
- It is disabled by `HF_HUB_DISABLE_SHARED_BLOBS` or `HF_HUB_DISABLE_XET`.
- Only Xet-verified downloads are published to it.

**Observed here:**
- The 4.6 GB `model.safetensors` resolves through two symlinks to `hub/blobs/09/09490e15…`.
- The `tokenizer.json` payload is shared by the 0.6B and 8B repos; its `.refs` file lists both.

**Pitfalls:**
- `du` on the repo directory reports 4.3 MB.
- `find -type f` on a snapshot finds 0 files.
- `rsync`/`tar` without dereferencing produce broken symlink skeletons.
- Benchmarks must use `du -L` or `stat -L` and hash through the symlinks.

**Other clients** (VERIFIED from source):
- The hf-hub 1.0.0 crate and llama.cpp (which now uses the HF cache, `common/hf-cache.cpp`, order `LLAMA_CACHE` → `HF_HUB_CACHE` → … → `~/.cache/huggingface/hub`) both read the shared store fine, because `exists()` follows symlinks. They write per-repo blobs, so there is no dedup but no corruption either.
- llama.cpp's `remove_cached_repo` deletes only the symlinks, leaving payloads orphaned until `hf cache prune`.
- llama.cpp's `-hf` always resolves `main`, so a revision cannot be pinned.
  - Its quant selection is regex `<quant>[.-]`; the default order is Q4_K_M, then Q8_0, then the first GGUF.
  - **Use `-m <snapshot path>` in benchmarks.**
- mlx-lm uses `snapshot_download(allow_patterns=…)` (`utils.py:251-292`). Its CLIs have no `--revision` flag, so **pass the snapshot path to pin**.

**Environment variables** (`constants.py`):
- `HF_HOME`; `HF_HUB_CACHE` (then `HUGGINGFACE_HUB_CACHE`); `HF_HUB_OFFLINE` (and `TRANSFORMERS_OFFLINE`); `HF_ENDPOINT`.
- `HF_TOKEN`, `HF_TOKEN_PATH`.
- `HF_HUB_DISABLE_SYMLINKS`, `HF_HUB_DISABLE_XET`, `HF_HUB_DISABLE_SHARED_BLOBS`, `HF_XET_HIGH_PERFORMANCE`.
- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated and ignored.

**Upstream drift (VERIFIED):**
- huggingface_hub **2.0.0** (2026-09-24) replaced the HTTP stack and removed 1.x deprecations.
- transformers 5.18 accepts `>=1.31,<3.0`, so an unpinned install can pick up either major version. **Pin it.**

### 5.5 Design: `sky run hf://org/model[@rev][:QUANT|/path]`

1. **Grammar.**
   - huggingface_hub's own `hf://[<type>/]<id>[@<rev>][/<path>]` (`utils/_hf_uris.py:259-319`), plus a llama.cpp-style `:QUANT` suffix allowed only without `/path`. Git refs cannot contain `:` (LIKELY unambiguous).
   - Examples: `hf://mlx-community/Qwen3-32B-4bit@<40hex>` and `hf://bartowski/Qwen_Qwen3-32B-GGUF:Q4_0`.
   - In benchmark mode, require a 40-hex `@rev` and an exact filename.
2. **Cache resolution.** Same order as Python. Do not invent `SKY_CACHE` for the hub cache. Sky's *derived* artefacts live in `~/.cache/sky/` and are keyed by source sha256 (§1.3 b).
3. **Offline first.**
   - A full commit plus a complete snapshot (checked against `trees/<commit>.json` or Sky's lockfile) means zero network calls.
   - A branch or tag with `--offline`/`HF_HUB_OFFLINE=1` reads `refs/<rev>`.
   - Otherwise make one `/api/models/{repo}/revision/{rev}` call.
4. **File set.**
   - **Safetensors repos:**
     - `config.json`, `generation_config.json`;
     - `tokenizer.json`, or else `tokenizer.model`;
     - `tokenizer_config.json`, `special_tokens_map.json`;
     - `chat_template.jinja` or `chat_template.json`;
     - only the shards listed in `model.safetensors.index.json`.
   - **GGUF repos:** the selected quant plus all its split parts.
5. **Writes.**
   - Take a flock on `.locks/<repo>/<etag>.lock`.
   - Reuse `blobs/<etag>` if it exists (as a file or a symlink).
   - Otherwise download `.incomplete` with parallel, resumable Range requests.
   - Verify the sha256 (or git SHA-1).
   - Rename atomically, then create the snapshot symlink.
   - **Never write `hub/blobs/`** in v1. Payloads are read-only and shared, so mmap them `PROT_READ`.
6. **Security.** Strip auth on cross-host redirects, reject non-hex oids, and reject absolute paths or `..` in repo paths (`hf-cache.cpp:164-173`).
7. **Provenance.** Every result records `{repo_id, requested_revision, resolved_commit, endpoint, offline, files:[{path,size,sha256|git_sha1,xet_hash}]}`.

---

## 6. Benchmark methodology on macOS

### 6.1 How the existing tools define their numbers

All VERIFIED from source at the pinned commits.

**mlx-lm `stream_generate`** (`mlx_lm/generate.py:658-760`):
- **What is not timed:** the prompt is tokenized *before* `tic` (`:700-708`), and model load happens earlier still.
- **`prompt_time`** runs from `tic` until the first token is yielded. It covers:
  - every prefill chunk (`prefill_step_size`: 2048 by default in `generate_step` (`:314`), 512 in `speculative_generate_step` (`:487`)), each `mx.eval`'d;
  - the final `_step` on the last prompt token, including sampling and `logsumexp` over the vocabulary (`:415-420`);
  - `mx.eval` of the first token;
  - plus `async_eval` dispatch of step 2 (`:455-466`).
- **`prompt_tps`** = `prompt.size / prompt_time`. It is really a TTFT-based rate (tokenization excluded).
- **`generation_tps`** = `(n+1) / (now − tic₂)`, where `tic₂` is reset right after token 0. The numerator counts token 0 but the denominator does not include its time, which biases the rate up by (n+1)/n: +0.4 % at 256 tokens. Python, detokenizer and consumer time (e.g. printing) are included.
- **`peak_memory`** = `mx.get_peak_memory()/1e9`, in decimal GB. It is the maximum of MLX's `active_memory_` since the process started (or since `reset_peak_memory`) (`mlx/backend/metal/allocator.cpp:166-167,214-215`).
  - It **excludes** MLX's buffer cache, the Python heap, Metal driver and pipeline memory, and the page cache.
  - It includes the weights, because MLX copies them (§1.3).
- **Wired limit:** `wired_limit()` sets MLX's wired limit to `max_recommended_working_set_size` during generation (`generate.py:229-264`; `utils.py:73-82`). MLX keeps buffers in Metal residency sets (`allocator.cpp:100-103,162,213`).

**`mlx_lm.benchmark`** (`benchmark.py:1-180`):
- Defaults: `-p 512 -g 1024 -n 5`, one warm-up, `prefill_step_size` 2048.
- **Random-token prompts** drawn uniformly from the vocabulary, including special tokens. That is harmless for a dense model but distorts MoE routing (LIKELY).
- EOS disabled (`_eos_token_ids = {}`).
- Reports the **mean** of the per-trial `prompt_tps`, `generation_tps` and `peak_memory`. Peak memory is never reset between trials, so it is cumulative.

**`llama-bench`** (`tools/llama-bench/llama-bench.cpp`):
- Defaults (`:381-425`): `-p 512 -n 128 -b 2048 -ub 512`, 5 repetitions, KV cache F16/F16, flash-attention auto, all layers on the GPU, warm-up on.
- `pp` and `tg` are **separate tests**.
- **`test_prompt`** (`:2168-2196`):
  - random tokens (BOS first if the vocabulary adds one);
  - processed in `n_batch` chunks;
  - timed until `llama_synchronize`;
  - no sampling.
- **`test_gen`** (`:2198-2222`):
  - feeds **random tokens** one at a time;
  - `llama_synchronize` after each;
  - **no logits processing or sampling**;
  - the KV cache is **cleared before each repetition** (`:2444`), so `tg128` measures decode at context positions 0–127 unless `-d <depth>` is given (depth state cached, `:2446-2473`).
- **Reported rate** = mean over repetitions of `n_tokens / t_rep`, ± stdev (`:1560-1575`): a mean of rates, not total tokens over total time. JSON output includes `samples_ns`.

**llama.cpp perf counters** (`src/llama-context.cpp:775-790`):
- A decode call with exactly one queued token counts as "eval"; one with more than one counts as "prompt eval". The split is by batch size, not by phase; the code carries a FIXME saying so.
- Sampling time is tracked separately.
- `eval` runs exclude the first generated token.

**llama-server** (`tools/server`):
- `timings` fields: `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `*_per_second`, `cache_n` (`server-common.cpp:86-96`).
- **`cache_prompt` defaults to `true`** (README:587), so repeated prompts skip prefill. The README also warns that logits "are not guaranteed to be bit-for-bit identical for different batch sizes".
- mlx_lm.server has an `LRUPromptCache` (`server.py:42,433-435`).
- Prefix caching must be defeated, or recorded, in every TTFT test.

**Industry definitions:**
- NVIDIA AIPerf (VERIFIED, https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference):
  - `ttft = first_response.perf_ns − request.start_perf_ns` (includes queueing, network, prefill and first token);
  - `ITL = (request_latency − TTFT)/(output_tokens − 1)`;
  - `prefill_throughput_per_user = ISL / TTFT`;
  - `output_token_throughput_per_user = 1/ITL`.
- MLPerf Inference uses TTFT and TPOT = (t_end − t_first)/(n − 1), with pre-tokenized inputs (LIKELY; https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc).

**What does not line up:**

| Quantity | mlx-lm | llama-bench | Proposed Sky protocol |
|---|---|---|---|
| Prompt content | user text (CLI) or random ids (benchmark) | random ids | fixed real text, chat-templated, exchanged as **ids** |
| Decode context | after the prompt | **empty** cache (unless `-d`) | stated depth d ∈ {512, 4096, 16384} |
| Sampling in the decode loop | yes (logsumexp + sampler) | **no** | argmax in every engine, timed |
| First token in the decode rate | in the numerator, not the time | n/a | excluded: (N−1)/(t_last − t_first) |
| Aggregation | mean of trial rates | mean of rep rates | median + bootstrap CI, all raw runs published |
| Tokenization in TTFT | excluded | n/a | reported separately; the headline TTFT starts at ids |

### 6.2 Memory: what each metric sees

**Apple definitions** (VERIFIED, `man footprint`, macOS 26.6.1):
- "A process's 'footprint' is equal to the total of all Dirty memory".
- Dirty excludes memory backed by a file.
- "file-backed memory, even if dirty, can be automatically evicted".
- `(Wired)` covers GPU-shared memory.
- Activity Monitor's Memory column, top's MEM and Xcode's gauge "all report values from the process footprint ledger".
- `phys_footprint_peak` is the "maximum value … since process launch".
- "Memory dirtied on behalf of a process is deliberately excluded" for some drivers (`nofootprint`).
- `footprint -a` reports system data: `sys_wired`, `sys_unwired`, `sys_footprint`.

**Our probe** (VERIFIED). Source `core/memprobe/memprobe.m`; full logs `core/memprobe/memprobe_run1.txt` (CPU touches the mapping) and `memprobe_run2_gpuonly.txt` (only the GPU reads it). Command: `bash $S/benchlock.sh ./memprobe mem1g.bin [gpuonly]`, M2 Pro, 1 GiB file of random data. The test file was deleted to save disk; recreate it with `head -c 1073741824 /dev/urandom > mem1g.bin`. All values in MiB:

| Stage | phys_footprint | RSS (`resident_size`) | `task_vm_info.external` | `ru_maxrss` | `MTLDevice.currentAllocatedSize` | System wired Δ |
|---|---|---|---|---|---|---|
| Baseline | 6 | 14 | 9 | 14 | 0 | – |
| `mmap` 1 GiB file (`PROT_READ`, `MAP_SHARED`), untouched | 6 | 14 | 9 | 14 | 0 | ~0 |
| **CPU touches every page** (run 1) | **6** | **1038** | **1033** | 1038 | 0 | ~0 |
| `makeBuffer(bytesNoCopy:)` over the mapping | 6 | (same) | (same) | – | **1024** | ~0 |
| **GPU reads all of it, CPU never touched** (run 2) | **6** | **14** | **9** | **14** | 1024 | **+1085** (file pages −1051) |
| Release buffer + `munmap` | 6 | 14 | 9 | 14 | 0 | back to baseline |
| 1 GiB Shared `MTLBuffer`, untouched | 6 | 14 | – | – | 1024 | – |
| … CPU `memset` | **1030** (graphics ledger 1025) | 1038 | 9 | 1038 | 1024 | – |
| + 1 GiB **Private** buffer, GPU-filled | **2054** | **1038**, unchanged | – | – | 2048 | – |
| Both released (autorelease pool drained) | **1030**, private charge persists | 14 | – | – | 0 | – |
| … 3 s later | 6 | 14 | – | – | 0 | – |
| `malloc` 1 GiB + touch | 1030 | 1038 | – | – | – | – |

The Private buffer shows in `footprint` as "Owned physical footprint (unmapped) (graphics)" 1,073,758,208 B.

**Conclusions** (VERIFIED on M2 Pro; whether they hold on the M5 Max is LIKELY but unchecked):
1. **Weights reached through `mmap` + a GPU no-copy buffer are invisible to every per-process metric.** That includes footprint/Activity Monitor, RSS, `ru_maxrss` (`/usr/bin/time -l`), `task_vm_info.external`, and footprint/vmmap resident.
   - They show up only as **system-wide wired memory**: `host_statistics64` `wire_count`, `vm_stat` "Pages wired down", Activity Monitor's system "Memory Used".
   - They also show in `MTLDevice.currentAllocatedSize`, which counts the buffer's *length* whether or not it is resident.
   - This is how llama.cpp's default path (mmap + no-copy Metal buffers) works, so llama.cpp's footprint will look about 19 GB smaller than MLX's for the same model even though both need the RAM (LIKELY; check with real binaries).
2. CPU-touched file pages count in RSS and `external`, but not in footprint.
3. GPU-private memory counts in footprint (graphics ledger) but **not** in RSS.
4. Footprint drops lag release by up to seconds (deferred Metal free). `phys_footprint_peak` includes transient peaks.
5. `currentAllocatedSize` over-reports lazily-backed buffers. Only the engine can read it, not an outside observer.
6. `mx.get_peak_memory` covers only live MLX buffers.

**So every single metric can be gamed, even by accident.**

**Proposed honest headline metric, `peak_ram_bytes`:**

A peak **system-wide unreclaimable-memory delta**, sampled outside the engine.

- `U(t) = page_size × (wire_count + internal_page_count − purgeable_count + compressor_page_count)`.
  - Read via `host_statistics64(HOST_VM_INFO64)` at 10 Hz by the harness.
  - This is Activity Monitor's "Memory Used" (App + Wired + Compressed).
- `peak_ram_bytes = max_t U(t) − median(U over the 5 s before the engine process is spawned)`.
- It counts:
  - weights the engine copied (anonymous or graphics memory);
  - weights the GPU reads through a no-copy file mapping (they become wired);
  - the KV cache, activations, caches and driver memory, including `nofootprint`.
- It does **not** count clean, reclaimable page cache. That is fair: MLX's page-cache copy of the file is not needed after loading.

**Guard rails:**
- (a) **CPU-only mapped weights:** if `peak_ram_bytes < 0.95 × weight_bytes_total`, the engine is running from unwired file pages, which happens with CPU-side mmap use.
  - Re-measure as `max_t[U(t) + external_engine(t)] − U₀`, where `external_engine` is `task_vm_info.external` of the engine process tree.
  - Flag the result.
- (b) **Run validity:**
  - Require `vm_stat` swapouts, compressions and decompressions Δ = 0 during the run. Otherwise memory pressure has distorted U, and the run is invalid.
  - Require the quiet gate (§6.5).
- (c) **Noise:**
  - Run a null engine (`sleep`) through the same harness and report its spread as the noise floor. Expect about ±100–300 MB on an idle Mac (LIKELY).
  - Repeat 3× and report the median.

**Always also report** these per-process diagnostics, labelled as diagnostics:
- `phys_footprint_peak` of the engine process tree, from `proc_pid_rusage(RUSAGE_INFO_V6).ri_lifetime_max_phys_footprint`, read on the child before it exits;
- `ru_maxrss` from `wait4`;
- peak `task_vm_info.external`;
- the engine's own report (`mx.get_peak_memory` / llama.cpp buffer sizes / Sky's allocator);
- `model_file_resident_bytes` from `mincore` on an observer mapping. R9-POWER verified that this does not touch pages.

**Phases:** report peak during load and steady state during decode at depth d separately. Load-time peaks can be about 2× for copying loaders.

**Optional audit (EXPERIMENTAL):**
- **Memory-knee test:** a balloon process allocates and keeps touching X GB of anonymous memory. Raise X until decode tok/s drops more than 5 % or swap starts. Then `RAM_needed ≈ total − X_max`.
- It cannot be gamed by any accounting trick. It is slow, so use it only to validate the headline number once.

### 6.3 Power and energy (from R9-POWER; evidence in `R9-work/power/`)

**powermetrics:**
- Needs root (VERIFIED: "must be invoked as the superuser"; `sudo -n` was unavailable here).
- The man page says the values are "**estimated and may be inaccurate** … should not be used for any comparison between devices". Comparing Sky with MLX on one Mac is allowed. VERIFIED quote.
- Samplers: `tasks, battery, network, disk, interrupts, cpu_power, thermal, sfi, gpu_power, ane_power` (`-h`, VERIFIED).
- Plist keys: `elapsed_ns`, `cpu_energy`, `gpu_energy`, `ane_energy`, `combined_power`, `thermal_pressure`, `gpu.freq_hz` and `dvfm_states` (VERIFIED via `strings`).
- "Combined" means CPU+GPU+ANE, so it **excludes DRAM** (LIKELY from the label).
- Windows run long: 1000 ms requested gives 1009.3 ms (VERIFIED from macmon's trace notes). Use `elapsed_ns`.

**Sudoless sources:**

| Source | Covers | Refresh (measured) | Label |
|---|---|---|---|
| IOReport "Energy Model" (macmon reads it with no root) | CPU, GPU, ANE, DRAM, GPU SRAM | ≤200 ms on M2 Pro | VERIFIED |
| SMC `PSTR` | whole-system power | ~1 s | VERIFIED |
| `ioreg AppleSmartBattery` → `PowerTelemetryData` | SystemPowerIn etc. | 60 s windows | VERIFIED behaviour; EXPERIMENTAL to rely on |

- macmon v0.8.2: `all_power = cpu+gpu+ane` **excludes DRAM**, and `sys_power = max(PSTR, all_power)` (VERIFIED, `src_lib/metrics.rs:185,473-475,561-564`).
- asitop is *not* sudoless: it runs `sudo powermetrics` (VERIFIED, `asitop/utils.py:47-60`). It divides mJ by 1000 and calls the result W, which is correct only at `-i 1000` (LIKELY bug).
- **Cross-check:** PSTR agreed with battery SystemPowerIn within 0.5–2.2 % over 60 s windows (VERIFIED). The two may share a sensor, so this shows consistency, not independent validation.

**Measured on Qwen3-0.6B-4bit (mlx-lm), M2 Pro** (VERIFIED, `power/demo2/`):
- Decode: chip (CPU+GPU+ANE) 20.7 W; chip + DRAM 25.0 W; **PSTR system 49.2 W**.
- Energy per decode token: 111 / 134 / **264 mJ** gross. Above idle: 93 / 113 / 170 mJ.
- **System energy was 2.4× the chip counters.** About 10 W of the increase appears in no CPU/GPU/ANE/DRAM channel.
- The CPU drew 7.1 W during decode, from mlx-lm's Python host loop. That is a legitimate target for Sky.

**M5-specific risks** (LIKELY, all-smi #410): M5 Max counters update in ~2.1 s batches. Substring channel matching double-counts sub-channels: 106.7 W reported against a true 36.2 W. Channel names change by generation; on M5, `PCPU` means "Super" cores (VERIFIED from macmon comments).

**Definitions:**
- Counter sources: `E[a,b] = Σᵢ P̄ᵢ·|windowᵢ ∩ [a,b]|`. Each sample is already a window average, so no trapezoid is needed. Use the trapezoid only for instantaneous meters.
- **Decode J/token** = E_system[t_first, t_last]/(N−1).
- **Prefill J/token** = E_system over a ≥60 s loop of prefills / total prompt tokens.
- **Energy to first token** = mean E_system[t_submit, t_first] over a ≥60 s loop with a fixed idle gap.
- **Net** = gross − P_idle·duration, where P_idle is measured for ≥60 s *with the model loaded and the process alive*.
- Headline is **gross system** energy; net and chip-only are diagnostics.
- Error: ≤ 2τ/T, which is ≤ 7 % at τ = 2.1 s and T = 60 s; also report the spread over ≥5 windows.

**Negative result** (VERIFIED): one 0.58 s TTFT window cannot be measured. PSTR gave 2.26 vs 8.37 mJ/token from the chip counters. **Loop short phases to ≥60 s.**

**Gold standard:** a wall-power AC meter or an inline USB-C PD meter, with the battery at 100 % and not charging (`BatteryPower=0`, `IsCharging=No`), following MLPerf Power (arXiv 2410.12032). The 16" M5 Max may pull more than its 140 W adapter at full CPU+GPU, draining the battery while plugged in (LIKELY, secondary sources). Detect this from battery telemetry and invalidate the run.

### 6.4 Thermal state and power modes (from R9-POWER)

**Thermal pressure:**
- `ProcessInfo.thermalState`: nominal/fair/serious/critical. "Serious" "reduces performance. Fans are running at maximum speed." (VERIFIED, Apple docs).
- CLI read: `osascript -l JavaScript -e 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'`.
- Finer levels via notify key `com.apple.system.thermalpressurelevel`: 0 Nominal, 1 Moderate, 2 Heavy, 3 Trapping, 4 Sleeping (VERIFIED, `OSThermalNotification.h:46-61`). It moved 0→1→0 under multi-agent load here, matching the system log.
- **`pmset -g therm` is useless on Apple Silicon** (VERIFIED): "No thermal warning level" while pressure was 1.

**Throttling detector:**
- Log pressure at 10 Hz and flag any window at level ≥ 1.
- Track GPU frequency and top-state residency (macmon/IOReport). Falling frequency at ~100 % activity means it is power- or thermally-limited.
- Fit a slope of tok/s over time.

**Sustained runs** of 1/10/30 min: report the median of the last 5 min and its ratio to minute 1.

**High Power Mode** (VERIFIED, Apple Support 101613, 2026-09-21):
- Available on 2024-and-later MacBook Pros with Pro or Max chips, which includes both the 14" and 16" M5 Max.
- It "allows the fans to run at higher speeds".
- Set in System Settings → Battery → Energy Mode, separately for battery and adapter.
- `pmset` key `powermode`: 0 auto, 1 low, 2 high. Changing it needs sudo (LIKELY). This M2 Pro 14" has no `powermode` key (VERIFIED).
- Secondary reports say the 14" M5 Max loses ~7.4 % sustained GPU performance in High Power Mode and ~25 % in Automatic (LIKELY, unverified). Report chassis size and mode, and **run the headline in High Power Mode on AC**.
- Low Power Mode must be off: `notifyutil -g com.apple.system.lowpowermode` → 0.

**Idle gap matters for TTFT** (VERIFIED, `power/ttft_gap.py`, warm process, 1,097-token prompt, 6 runs per gap, randomised):
- Median TTFT was 386.5 ms back-to-back and 420.6 ms after an 8 s idle: **+8.8 %**.
- Fix the idle gap (1.0 s) before every TTFT measurement, identically for all engines.

### 6.5 Noise control and statistics

**Quiet gate before every trial**, enforced by the harness and logged:
- CPU ≥97 % idle over 30 s (`top -l 2 -s 5 -n 0`).
- GPU <1 % active.
- Thermal pressure 0.
- PSTR within 10 % of the idle baseline.
- No change in `vm_stat` swapins/swapouts.
- `pmset -g assertions` recorded. Here it showed audio assertions; a CoreAudio stream keeps the SoC out of deep idle.

**Machine state:**
- **Power:** AC, battery 100 %, High Power Mode, Low Power Mode off.
- **Sleep:** `caffeinate -i -s -w <pid>` (VERIFIED flags).
- **Display:** fixed brightness. Record `ioreg -rc AppleARMBacklight`; auto-brightness and True Tone off; external displays disconnected.
- **Spotlight:** store models in a `*.noindex` directory (LIKELY effective) or `sudo mdutil -i off`.
- **Time Machine:** off (`tmutil status`).
- **Other:** no other user apps. Close the network after downloads (optional).

**This development box cannot produce publishable numbers** (VERIFIED): 4.1 GB of swap in use and ~13.8 GB in the compressor. Use a dedicated, freshly rebooted M5 Max.

**Cold page cache without sudo** (VERIFIED by R9-POWER with `mincore`):
- An APFS clone (`cp -c`) of a cached file is **0 % resident** and costs no disk space. So `cp -c model.safetensors /tmp/cold/…` gives a cold-load path.
- Not a substitute for `sudo purge` when reasoning about other cached files.

**Run ordering:**
- Interleave engines in **randomised blocks** (ABCABC… with each block's order shuffled from a fixed seed).
- Fixed cool-down between trials: pressure back to 0 and PSTR within 10 % of idle, or 60 s.
- One discarded warm-up per process. JIT engines compile shaders at first use, and Sky has a JIT.
- Publish **all** runs, including warm-ups. Warm-up may never settle (Barrett et al., OOPSLA 2017).

**Run counts:**
- Measured here: llama-bench decode CV ≈ 1.1 %, prefill ≈ 0.5 % (VERIFIED from `$S/research/R5-raw/A2-8B-Q4_0-fa0.md`, 5 repetitions, Qwen3-8B Q4_0).
- So **N = 10 interleaved trials per cell** gives ≈ ±0.8 % on the median (LIKELY).
- Thermal drift dominates, so interleaving is mandatory.

**Statistics:**
- Per cell, report median, p5/p95, mean ± stdev, and **bootstrap 95 % CI of the median** (10,000 resamples, percentile method).
- For claims, report the **ratio of medians Sky/baseline with a bootstrap CI** (paired by block).
- **Claim a win only if the CI excludes 1.0** (Kalibera & Jones, ISMM 2013; Chen & Revels, arXiv 1608.04295).
- Pre-register the metrics, scenarios, N and thresholds in the repo before collecting the headline data.

### 6.6 Quality equivalence: how to show the same output quality

**Principles:**
1. **Compare distributions, teacher-forced.** Feed identical token-id sequences to every engine and compare the next-token distributions at every position.
   - Greedy free-running decode diverges at the first bf16 near-tie, and every later token then differs. It is a smoke test only.
2. **Use a high-precision reference.**
   - The **BF16 base model** at the pinned commit.
   - Computed once with two independent implementations: llama.cpp on a BF16 GGUF, and mlx-lm on BF16 MLX, or transformers.
   - The two references must agree (mean KLD ≲ 1e-4; LIKELY achievable) before either is trusted.
3. **Standard corpus and chat corpus.**
   - (a) A Wikitext-2-style corpus: fixed chunks of 512 or 2048 tokens, scoring only the second half (the llama-perplexity convention, `perplexity.cpp:530-540`).
   - (b) A chat corpus: 100 fixed prompts in non-thinking mode, with responses generated once greedily by the BF16 reference and then teacher-forced in every engine.
   - Publish both as **token ids** with a sha256.

**Metrics per engine × file vs the BF16 reference:**
- mean KLD (nats/token) ± 95 % CI;
- median, p99 and max KLD;
- **top-1 agreement %**;
- PPL and PPL ratio;
- RMS Δp of the correct token;
- per-position max |Δlogit| (diagnostic).

**The decisive comparison is on identical weights:**
- **Sky vs mlx-lm on the identical MLX file, and Sky vs llama.cpp on the identical GGUF.**
- Report KLD(engine‖engine) and KLD of each against BF16.
- **Pass criterion (pre-registered):**
  - Sky's KLD vs BF16 ≤ the baseline engine's KLD vs BF16 + 2σ;
  - **and** KLD(Sky‖baseline) ≤ 0.1 × KLD(baseline‖BF16).
  - In words: Sky's deviation from the baseline must be small compared with quantization error itself.

**Tooling facts** (VERIFIED):
- `llama-perplexity --kl-divergence-base` stores base log-probs as **uint16 with a per-token scale and minimum**, so the reference carries quantization noise. It skips p < e⁻¹⁶ in the KLD sum (`perplexity.cpp:191-240`).
  - Files are huge: "37 GiB for LLaMA 3" on Wikitext-2 (README). For Qwen's 151,936-token vocabulary, budget ~300 KB per scored position.
- `mlx_lm.perplexity`:
  - default dataset `allenai/tulu-3-sft-mixture`, sequence length 512, 256 samples, seed 123;
  - **scores all positions** in each window;
  - standard error by the delta method (`perplexity.py:20-100`).
  - Its numbers are therefore not comparable with llama-perplexity's.

**Our harness should:**
- Emit per-position `{logsumexp (f32), target logit, top-64 ids + logits}`, about 0.5 KB per position, for ~64k positions.
- Also keep the full fp32 vocabulary on a 2,048-position subset (~1.2 GB in fp16) for exact KLD.
- Compute every statistic with the same code for all engines.
- Optionally, also write llama.cpp's `.kld` format so llama.cpp's own tool can verify our numbers.

**Optional task-level check:**
- GSM8K, 250 problems, non-thinking mode, greedy, identical prompts, through each engine's OpenAI-compatible server (`mlx_lm.server`, `llama-server`, Sky).
- Differences should be within the binomial CI (±~6 pp at n=250, so it is weak evidence).
- mlx-lm has `mlx_lm.evaluate` (lm-eval).
- **Quality runs never share a process with speed runs.**

### 6.7 Packaging: `git clone sky-evaluation && ./benchmark.sh`

**Pin everything:**
- `uv.lock` with exact `mlx`, `mlx-metal`, `mlx-lm`, `transformers`, `tokenizers`, `huggingface_hub` (1.x vs 2.x matters, §5.4), `hf_xet`, `numpy`.
- The llama.cpp commit, plus the exact CMake flags (`-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`; record `GGML_NATIVE`, BLAS and so on). Alternatively an official release binary with its sha256.
- The Sky release binary and its sha256.
- macmon version and sha256.

**`models.lock.json`:**
- For every input file: repo, **40-hex commit**, path, size, sha256, and xet hash.
- For every derived file (our MLX conversion and GGUF quantizations): the exact command, tool versions, and expected sha256.
  - Check that the conversions are deterministic by running them twice (LIKELY true for llama-quantize without an imatrix; check MLX).
- Verify by hashing through symlinks (`shasum -a 256 "$(realpath …)"`). Hashing warms the page cache, so do it **before** cold-start steps or use the APFS clone.

**Prompts:** `prompts/*.json` hold token-id arrays plus the source text, the template, the renderer version and a pinned date. Every engine is fed the ids.

**Environment capture**, into the result JSON:
- `sw_vers`; `uname -a`; `sysctl hw.model machdep.cpu.brand_string hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`.
- `system_profiler SPHardwareDataType SPPowerDataType SPDisplaysDataType -json`, with the serial number hashed.
- `pmset -g`, `pmset -g custom`, `pmset -g batt`, `pmset -g ac`, `pmset -g assertions`.
- Thermal pressure and `thermalState`.
- Backlight level.
- Metal device name, `maxBufferLength` and `recommendedMaxWorkingSetSize`.
- `clang --version`, `rustc --version`, Python version, `pip freeze`/`uv.lock` hash.
- Engine versions and git SHAs; macmon version; whether `HF_HUB_OFFLINE` was set.

**Offline execution:** prefetch, then run everything with `HF_HUB_OFFLINE=1` against snapshot paths:
- mlx-lm: `--model <snapshot dir>`;
- llama.cpp: `-m <snapshot>/<file>`, never `-hf`, which tracks `main`.

**`./benchmark.sh` phases:**
1. `doctor`: check the hardware and OS, AC power, power modes, free RAM and disk; refuse to run under swap.
2. `fetch`: resumable, verified.
3. `build`, or download binaries.
4. `gates`: tokenizer conformance and identical-weights logit equivalence.
5. `speed`.
6. `memory`.
7. `energy`, with an optional external meter.
8. `quality`.
9. `report`: JSON plus Markdown/HTML, including raw samples.

**Practical notes:**
- The full run takes hours, so `--quick` runs a reduced matrix.
- Disk budget: Qwen3-32B BF16 is ~65.5 GB, plus MLX 4-bit at 18.4 GB, GGUF Q4_0 at ~18.6 GB, optionally Q4_K_M at ~20 GB, plus Sky's relayout cache at ~18.4 GB. **Total about 140 GB.**
  - Offer a "consumer" mode that downloads the published quantized files (pinned by hash) and skips the BF16 reference. Use the pre-computed, published reference logits (top-k file, hashed) for the quality step.

---

## 7. A benchmark protocol meant to survive hostile review

**Target:** Qwen3-32B, 4-bit, batch 1, M5 Max MacBook Pro.

**Engines:**
- Sky;
- mlx-lm (and MLX);
- llama.cpp, using its Metal backend.

Every step below is executable and logged. "Pre-register" means: commit the parameters to the repo before collecting headline data.

### 7.0 Pre-registration (commit before data collection)
- Metrics and their exact definitions (§7.5).
- Scenarios, N, randomisation seed, pass thresholds, analysis code.
- **The claim rule:** a "win" requires the bootstrap 95 % CI of the paired ratio of medians to exclude 1.0, on the same file, same prompt ids, same sampler and same KV dtype.
  - The claim must also pass the quality gate (§7.3).

### 7.1 Artefacts: one base model, every quantization built by us

1. **Base model.** `Qwen/Qwen3-32B` @ a pinned 40-hex commit, as BF16 safetensors. Record sha256 for every shard.
2. **MLX file.** Built with the pinned mlx-lm:
   ```
   mlx_lm.convert --hf-path <base snapshot> -q --q-bits 4 --q-group-size 64
   ```
   - Record its sha256.
   - Also list `mlx-community/Qwen3-32B-4bit` @ commit and report whether it is byte-identical. If not, use **ours**.
3. **GGUF files.** Built with the pinned llama.cpp:
   ```
   convert_hf_to_gguf.py --outtype bf16 <base snapshot>
   llama-quantize <bf16.gguf> <out.gguf> Q4_0
   llama-quantize <bf16.gguf> <out.gguf> Q4_K_M
   ```
   - No imatrix, so no data dependence. Otherwise publish the imatrix file and its hash.
   - Record each file's tensor-type histogram. Q4_0 will have a Q6_K output tensor (§2.3).
4. **Size metadata.** For each file record:
   - `weight_bytes_total`;
   - `weight_bytes_read_per_decode_token` (all linear weights + LM head + one embedding row);
   - effective bpw.
5. **Prompts.**
   - Fixed, licence-clean English text, wrapped in the Qwen3 chat template in **non-thinking mode** (`enable_thinking=false`).
   - Rendered once with pinned transformers/jinja2, with a pinned date.
   - Lengths n_prompt ∈ {128, 512, 2048, 8192, 32768}, each with 3 different texts.
   - Stored as **token-id arrays** with a sha256. Exactly one BOS (Qwen has none).
6. **Quality corpora** (§6.6), as ids with a sha256.

### 7.2 Machine preparation (recorded in `machine` and `power_state`)
- **Hardware and OS:**
  - M5 Max MacBook Pro: record 14" or 16", GPU core count, RAM.
  - Pinned macOS build. Freshly rebooted, logged in, no other apps.
- **Power:**
  - On AC (record the adapter wattage), battery 100 % and not charging.
  - **High Power Mode on** (`pmset -g` `powermode`=2, if exposed); Low Power Mode off.
- **Display and ambient:** lid open; fixed brightness; auto-brightness and True Tone off; no external display. Record ambient temperature with a thermometer.
- **Background services:**
  - Spotlight excluded (`*.noindex` directory) or `mdutil -i off`.
  - Time Machine off.
  - Notifications off.
  - `caffeinate -i -s` for the whole session.
- **Run `./benchmark.sh doctor`.** It refuses to proceed if swap is in use, free RAM is below model + KV + 8 GB, thermal pressure is above 0, Low Power Mode is on, or the machine is on battery (for the headline).

### 7.3 Correctness gates (must pass before any speed number is published)

**G1 Tokenizer.**
- Sky's ids equal the golden ids (from the raw Python `tokenizers` library) on the prompt texts and the 28-string adversarial corpus.
- Engines are fed ids anyway; this gate guards Sky's text API.

**G2 Identical-weights equivalence (teacher-forced).** Pairs:
- (Sky, mlx-lm) on the MLX file;
- (Sky, llama.cpp) on the Q4_0 GGUF;
- (Sky, llama.cpp) on the Q4_K_M GGUF.

Measure on 2,048 positions with the full vocabulary and on 64k positions with top-64.

**Pass:**
- KLD(Sky‖baseline) ≤ 0.1 × KLD(baseline‖BF16);
- top-1 agreement ≥ 99 %;
- Sky's KLD vs BF16 ≤ the baseline's + 2σ.

**G3 Greedy smoke test.** 32 prompts × 128 tokens.
- Report the token-match prefix length.
- For each divergence, the reference's top-2 logit margin must be < 0.1 (a near-tie). A larger margin means a bug: stop and fix.

**G4 Settings parity.** Asserted from engine logs:
- KV cache dtype. All bf16 or all f16; state which. llama.cpp's default is f16 and mlx-lm's is bf16 (§3.1), so set llama.cpp `-ctk/-ctv` to match, or report both.
- Flash attention on.
- Context size ≥ n_prompt + n_gen.
- No speculative decoding.
- No prompt cache.
- Sampler = argmax.
- EOS ignored.
- Batch 1.

### 7.4 Speed scenarios

**Run order:** interleaved randomised blocks, N=10 measured trials per cell, plus 1 discarded warm-up per engine process.

| ID | Scenario | Purpose |
|---|---|---|
| S1 | n_prompt=512 → decode **512** tokens | Headline decode tok/s at depth ~512–1024; also TTFT@512 |
| S2 | n_prompt ∈ {128, 2048, 8192, 32768} → decode 1 token | Headline prefill tok/s and TTFT curve |
| S3 | n_prompt ∈ {4096, 16384} → decode 256 tokens | Decode at depth (KV-bound regime) |
| S4 | Sustained: S1 in a loop for 10 min (and 30 min, once) | Thermal stability: last-5-min median ÷ minute-1 median |
| S5 | Cold start: exec → first token, n_prompt=512 | (a) warm page cache; (b) cold, via a fresh APFS clone of the model files |
| S6 | Energy loops: decode loop ≥60 s; prefill loop ≥60 s (n=2048); TTFT loop ≥60 s with a fixed 1.0 s idle gap | J/token and energy to first token |
| S7 | Memory: S1 and S3(16384) with the 10 Hz memory sampler | Peak RAM at load and at steady state |

**Two driver modes**, both published:
- **(A) Native in-process driver:**
  - mlx-lm through `stream_generate`, timestamping each yielded token on the harness side;
  - llama.cpp through a small C driver on `llama_decode`, with `llama_synchronize` per token and argmax on the host — the same work as llama-bench plus sampling;
  - Sky through its API.
- **(B) Server end-to-end:**
  - `mlx_lm.server`, `llama-server` (`cache_prompt=false`) and `sky serve`, all on 127.0.0.1;
  - one identical OpenAI-compatible streaming client sending `prompt` as token ids, with timestamps at the socket (AIPerf definitions).
- **The headline is (A).** (B) shows that no engine wins by skipping work that a real user pays for.

### 7.5 Exact metric definitions

**Clock and timestamps:**
- The clock is `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` (mach absolute), read in the harness.
- `t_submit`: immediately before the prompt ids are handed to the engine.
- `t_tok[i]`: when the id of output token *i* is available to the harness, for i = 0…N−1.

**Latency and throughput:**
- **TTFT** = `t_tok[0] − t_submit`, with the engine process warm, the model loaded, and a fixed idle gap of 1.0 s before submission.
- **TTFT_text** (secondary) additionally includes template rendering and tokenization, starting from raw messages.
- **Prefill tok/s** = `n_prompt / TTFT`, reported for n_prompt ≥ 2048 as the headline. At those lengths the first-token sampling cost is negligible; the engines' internal prefill timers are recorded as diagnostics.
- **Decode tok/s** = `(N − 1) / (t_tok[N−1] − t_tok[0])`, with N = 512 (S1) or 256 (S3). The depth is stated as `n_prompt`.
- **ITL percentiles** (p50/p90/p99/max) are taken over `t_tok[i] − t_tok[i−1]` for i ≥ 1. The TPOT/ITL mean equals 1/decode tok/s.
- **E2E latency** = `t_tok[N−1] − t_submit`.
- **Cold-start TTFT** = first token time − `exec` time of the engine process (S5). Report page-cache state (a) warm or (b) cold.

**Memory:**
- **Peak RAM** = `peak_ram_bytes` (§6.2): max ΔUsed against a 5 s pre-spawn baseline, sampled at 10 Hz. Guard rails as defined there.
- Load peak and decode steady state are reported separately.

**Energy:**
- **J/token (decode)** = E_system over the decode loop ÷ (generated tokens − 1 per request).
- **J/prefill-token** = E_system over the prefill loop ÷ total prompt tokens.
- **Energy to first token** = mean E_system[t_submit, t_tok[0]] over the TTFT loop.
- E_system comes from SMC PSTR (sudoless, about 1 s resolution) or a wall/USB-C meter, which is preferred and recorded as `power.source`.
- Report gross (headline) and net-of-idle. Idle is measured for ≥60 s with the engine loaded and idle.
- Chip and DRAM counters are diagnostics.

**Effective bandwidth** (diagnostic, exposes size games) = `weight_bytes_read_per_decode_token × decode tok/s`, compared with the device's peak bandwidth.

### 7.6 Analysis and reporting
- **Per cell:** median, p5, p95, mean, stdev, n, and a bootstrap CI of the median.
- **Per comparison:** paired-by-block ratio of medians with a bootstrap CI. Claim win/tie/loss by the pre-registered rule.
- **Always show** file bpw and bytes per token next to every speed number. A faster engine on a smaller file proves nothing.
- **Always show** quality (KLD vs BF16, top-1 agreement) next to every speed number.
- **Publish:**
  - raw per-token timestamps;
  - power and memory sample streams;
  - environment capture;
  - all trials including warm-ups and invalidated runs, with the invalidation reasons (quiet-gate failures, thermal events, swap).
- **Known confounders to state explicitly:**
  - KV dtype (bf16 vs f16).
  - Flash-attention implementations.
  - llama.cpp's f32 activations vs MLX's bf16 (§3.1).
  - mlx-lm's Python host loop (~7 W of CPU during decode on M2 Pro, §6.3): a legitimate win for Sky, but it must be named.
  - Prompt cache disabled.
  - The Q6_K LM head in Q4_0 files (+1.1 % bytes for 32B).

---

## 8. Results JSON schema (v1)

One file per (engine, model file, scenario) cell. Field names are normative.

```json
{
  "schema_version": "sky-bench/1.0",
  "run_id": "uuid", "created_utc": "2026-10-02T12:00:00Z",
  "preregistration": {"git_sha": "40hex", "path": "PREREG.md"},
  "harness": {"name": "sky-evaluation", "version": "1.0.0", "git_sha": "40hex",
              "clock": "CLOCK_UPTIME_RAW", "driver_mode": "native|server"},
  "machine": {"model_identifier": "Mac17,x", "chassis": "14in|16in", "chip": "Apple M5 Max",
              "cpu_cores": {"super": 0, "performance": 0, "efficiency": 0}, "gpu_cores": 40,
              "memory_bytes": 0, "serial_sha256": "hex",
              "metal": {"device": "Apple M5 Max", "max_buffer_length": 0,
                        "recommended_max_working_set_size": 0},
              "page_size": 16384},
  "os": {"product_version": "26.x", "build": "25Gxx", "kernel": "Darwin 25.x"},
  "power_state": {"on_ac": true, "adapter_w": 140, "battery_percent": 100, "is_charging": false,
                  "powermode": 2, "lowpowermode": 0, "display_brightness_raw": 32768,
                  "external_displays": 0, "lid_open": true, "ambient_c": 22.0},
  "engine": {"name": "sky|mlx-lm|llama.cpp", "version": "x.y.z", "git_sha": "40hex",
             "binary_sha256": "hex", "build_flags": ["-DGGML_METAL=ON"],
             "python": "3.12.x", "lockfile_sha256": "hex",
             "settings": {"kv_cache_dtype": "bf16|f16", "flash_attention": true,
                          "prefill_chunk": 2048, "ubatch": 512, "threads": 0,
                          "prompt_cache": false, "speculative": false,
                          "wired_limit_bytes": 0, "mmap": true}},
  "model": {"base": {"repo": "Qwen/Qwen3-32B", "commit": "40hex"},
            "file_format": "mlx-safetensors|gguf",
            "source": {"repo": "org/name", "commit": "40hex", "derived_by": "cmd + tool versions"},
            "files": [{"path": "model-00001-of-00004.safetensors", "size": 0,
                       "sha256": "hex", "xet_hash": "hex"}],
            "quant": {"scheme": "mlx-affine|Q4_0|Q4_K_M", "bits": 4, "group_size": 64,
                      "tensor_type_histogram": {"Q4_0": 0, "Q6_K": 1, "F32": 0},
                      "bpw_effective": 4.5},
            "weight_bytes_total": 0, "weight_bytes_read_per_decode_token": 0,
            "tokenizer": {"files": [{"path": "tokenizer.json", "sha256": "hex"}],
                          "eos_ids": [151645, 151643]}},
  "workload": {"scenario": "S1|S2|S3|S4|S5a|S5b|S6-decode|S6-prefill|S6-ttft|S7",
               "prompt_id": "en-512-a", "prompt_ids_sha256": "hex", "n_prompt": 512,
               "n_gen": 512, "depth": 512, "batch": 1, "sampler": "argmax",
               "ignore_eos": true, "chat_template": {"mode": "non-thinking",
               "renderer": "transformers 5.18.0", "date_pinned": "2026-01-01"},
               "idle_gap_s": 1.0, "seed": 0},
  "trials": [{
    "index": 0, "block": 0, "position_in_block": 0, "warmup": false, "valid": true,
    "invalid_reasons": [],
    "quiet_gate": {"cpu_idle_pct": 98.7, "gpu_active_pct": 0.2, "thermal_pressure": 0,
                   "pstr_w": 9.1, "swap_delta_pages": 0},
    "timestamps_ns": {"t_submit": 0, "t_tok_first": 0, "t_tok_last": 0,
                      "per_token_file": "raw/tok_ts_0.bin"},
    "latency": {"ttft_ms": 0, "ttft_text_ms": null, "e2e_ms": 0,
                "itl_ms": {"p50": 0, "p90": 0, "p99": 0, "max": 0}},
    "throughput": {"prefill_tps": 0, "decode_tps": 0},
    "engine_reported": {"prompt_tps": 0, "generation_tps": 0, "peak_memory_gb": 0},
    "output": {"token_ids_sha256": "hex", "n_tokens": 512},
    "memory": {"peak_ram_bytes": 0, "steady_ram_bytes": 0, "load_peak_ram_bytes": 0,
               "baseline_used_bytes": 0, "sampler_hz": 10,
               "diagnostics": {"phys_footprint_peak": 0, "ru_maxrss": 0,
                               "task_external_peak": 0, "engine_self_report": 0,
                               "model_file_resident_bytes": 0, "metal_allocated_peak": null},
               "guard": {"weights_counted": true, "swapouts_delta": 0, "compressions_delta": 0}},
    "power": {"source": "pstr|wall_meter|usb_c_meter", "sampler": "macmon 0.8.2",
              "interval_ms": 200, "idle_w": 0,
              "energy_j": {"system": 0, "soc": 0, "soc_dram": 0},
              "j_per_token_gross": 0, "j_per_token_net": 0, "ettft_j": null,
              "battery_power_mw_max": 0},
    "thermal": {"pressure_max": 0, "seconds_at_level": {"0": 0, "1": 0},
                "thermal_state_max": 0, "gpu_freq_mhz_median": 0, "fan_rpm_max": 0}
  }],
  "summary": {"decode_tps": {"n": 10, "median": 0, "p5": 0, "p95": 0, "mean": 0, "stdev": 0,
                             "ci95_median": [0, 0], "method": "bootstrap-percentile-10000"},
              "prefill_tps": {}, "ttft_ms": {}, "peak_ram_bytes": {}, "j_per_token_gross": {},
              "sustained_ratio_last5_over_first1": null,
              "effective_bandwidth_gbps": 0},
  "quality": {"reference": {"impl": "llama.cpp-bf16+mlx-bf16", "agreement_kld": 0},
              "corpus": {"id": "wikitext2-chunks-512", "ids_sha256": "hex", "n_positions": 0},
              "vs_reference": {"kld_mean": 0, "kld_ci95": [0, 0], "kld_p99": 0, "kld_max": 0,
                               "top1_agreement": 0, "ppl": 0, "ppl_ratio": 0, "rms_dp": 0},
              "identical_weights_pair": {"other_engine": "mlx-lm", "kld_mean": 0,
                                         "top1_agreement": 0, "max_abs_logit_diff_p99": 0},
              "greedy_smoke": {"prompts": 32, "mean_match_prefix": 0, "max_divergence_margin": 0},
              "gate_pass": true},
  "comparisons": [{"metric": "decode_tps", "baseline": "mlx-lm", "candidate": "sky",
                   "ratio_of_medians": 0, "ci95": [0, 0], "pairing": "block",
                   "claim": "win|tie|loss"}],
  "raw_files": {"timestamps": "raw/", "power": "raw/power.jsonl", "memory": "raw/mem.jsonl",
                "env": "raw/env.json"}
}
```

---

## 9. Open questions
1. **M5 Max behaviour still unverified** — we have no M5 Max hardware:
   - the alignment tolerance of misaligned `uint4` loads;
   - whether GPU access to no-copy file mappings still wires the pages (§6.2);
   - `maxBufferLength`;
   - the batching period of the IOReport counters;
   - the `powermode` key;
   - whether PSTR works.
2. Are `mlx_lm.convert` outputs byte-deterministic across runs and machines? Does our conversion equal `mlx-community/Qwen3-32B-4bit`? If not, which do we headline?
3. KV-cache dtype parity. mlx-lm runs bf16; llama.cpp defaults to f16 (and supports bf16/q8). Do we match all engines to one dtype, or report each engine's default?
4. What threshold makes "the same quality" credible to Apple reviewers? The proposal is KLD(Sky‖baseline) ≤ 0.1 × KLD(baseline‖BF16) plus top-1 agreement ≥ 99 %. It should be sanity-checked against MLX-vs-llama.cpp BF16 noise.
5. Which reference implementation produces the BF16 logits? It needs ~65.5 GB, which fits on a 128 GB M5 Max; CPU fp32 does not fit. Should we also publish reference logits computed on a CUDA box for independence?
6. Does Sky's relayout cache count as a "different file"? Pre-register that it is a lossless re-layout (bit-identical tensor values; verify by hashing dequantized tensors) and that its build time is reported in cold start S5.
7. Tokenizer API: does Sky's `sky run` text path match HF exactly on NFD input? HF normalizes NFC; llama.cpp doesn't.
8. huggingface_hub 2.x adoption. Will the shared-blob cache or the HTTP stack change again before the public release? Pin the version and test both.

---

## 10. Architect brief

**Notes:** `$S/research/R9-formats-methodology.md`, where `S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.
- Evidence: `$S/research/R9-work/{core,arch,tok,hub,power}/`. All checks ran on an **M2 Pro 16 GB, macOS 26.6.1**, not an M5 Max.

### Findings

**1. MLX-written safetensors are unaligned.** VERIFIED.
- MLX writes its own safetensors with no header padding (`mlx/io/safetensors.cpp:220-275`). In `mlx-community/Qwen3-8B-4bit` the data starts at file offset 103,462 (≡ 6 mod 8).
- As a result, **0 of 907 tensors are even 4-byte aligned**, including the packed `uint32` weights.
- The HF safetensors writer pads to 8 bytes and sorts by dtype, so its tensors are aligned to their element size. Nothing guarantees 16 bytes.
- GGUF guarantees 32-byte alignment, and its reference reader enforces exact padded offsets (`gguf.cpp:780-815`).

**2. Zero-copy loading of MLX files is a correctness trap.** VERIFIED on M2 Pro.
- A `uint` loaded at a buffer offset of 2 or 6 **silently returns the word at the address rounded down**.
- Metal API validation does catch it: "must be a multiple of 4" for `uint`, "of 16" for `uint4`.
- So `mmap` + `makeBuffer(bytesNoCopy:)` cannot serve mlx-community files. Sky must copy them (MLX itself `pread`s into fresh buffers), repack them into an aligned cache, or use byte loads.

**3. MLX affine 4-bit layout.** VERIFIED against `mx.dequantize`.
- Dequantization is `w = scale·q + bias`.
- 8 nibbles per little-endian `uint32`, least-significant first; group 64; bf16 scales and biases.
- **50.25 % of scales are negative**, because the quantizer pins the larger-magnitude edge (`ops.cpp:5170-5225`).
- The config holds the defaults plus per-module overrides. Missing parameters are inferred from tensor shapes (`utils.py:380-405`).
- Supported modes: affine, mxfp4, nvfp4, mxfp8.

**4. Files labelled "4-bit" differ in size and content.** VERIFIED.
- bartowski Q4_0 is 4.67 bpw:
  - its `output.weight` is Q6_K;
  - its first-1/8-layer `ffn_down` tensors are Q4_1 because of the imatrix (`llama-quant.cpp:488-510,661-666`).
- Qwen's official Q4_K_M is 4.90 bpw, and its metadata name is "Awq Compatible Instruct", which leaves its provenance unclear.
- MLX 4-bit is 4.50 bpw.
- Decode speed is bounded by memory bandwidth, so size differences turn directly into speed differences.

**5. No exact conversion exists between the 4.5-bpw formats.**
- MLX group-64 → Q4_1 is exact on Qwen3-8B (every scale and bias fits in fp16; VERIFIED), but it costs 5.0 bpw.
- **"Same weights" therefore means a pair per format:** Sky vs mlx-lm on the same MLX file, and Sky vs llama.cpp on the same GGUF. Sky must run both formats natively.

**6. The tools' own speed numbers measure different things.** VERIFIED.
- **llama-bench `tg`** decodes from an **empty** KV cache, feeds random tokens without sampling, and averages per-repetition rates.
- **mlx-lm `generation_tps`** includes sampler, Python and detokenization, and counts the first token without its time.
- llama.cpp perf counters split "eval"/"prompt eval" by batch size; llama-server caches prompts by default; mlx_lm.benchmark uses random ids.
- **Consequence:** one external harness with per-token timestamps.

**7. Every standard memory metric can be fooled.** VERIFIED with our probe (`core/memprobe`).
- Weights read **only by the GPU** through an `mmap` + no-copy buffer (llama.cpp's default path) are invisible to `phys_footprint` (Activity Monitor), RSS/`ru_maxrss`, `task_vm_info.external`, and `footprint`/`vmmap`.
- They show up only as **system-wide wired memory** (+1085 MiB per GiB read) and in `MTLDevice.currentAllocatedSize`.
- GPU-private buffers appear in footprint but not in RSS.
- Footprint lags buffer release by seconds.

**8. Power counters undercount, and short phases can't be measured.** VERIFIED by R9-POWER.
- powermetrics needs root and Apple calls its values "estimated". macmon reads IOReport without root, but leaves DRAM out of its total.
- Whole-system power (SMC `PSTR`) agreed with battery telemetry to within about 2 %.
- During Qwen3-0.6B decode, **system energy per token was 2.4× the chip counters**. The CPU drew 7.1 W of that from mlx-lm's Python loop.
- PSTR updates roughly once a second; M5 counters reportedly arrive in ~2 s batches (LIKELY). Sub-second phases are therefore not measurable: loop each phase for ≥60 s.
- The idle time before a request changes time-to-first-token by **up to +8.8 %**.

**9. Tokenizers and chat templates.** VERIFIED by R9-TOK.
- **Rust `tokenizers` 0.23.2**, built pure-Rust with fancy-regex, matches Python on **168/168** cases.
- **llama.cpp** skips Qwen's NFC normalizer (2/28 differ) and hard-codes its regex by fingerprint instead of reading HF's.
- **transformers v5** breaks R1-Distill-Llama tokenization ("Thequickbrownfox").
- **Double BOS** appears when rendered templates are re-encoded with `add_special_tokens=True`.
- **Templates:** minijinja + pycompat + a Python-exact `tojson` (+ `preserve_order`) renders **30/30 cases byte-identically** to transformers. gpt-oss and Mistral templates inject today's date. mlx-lm turns Qwen3 thinking on by default.

**10. The HF Hub changed under us.** VERIFIED by R9-HUB.
- huggingface_hub **1.32** (two weeks ago) added a read-only **shared blob store keyed by the Xet hash**, on by default. `du` reports 4.3 MB for a 4.6 GB model, and copying without following symlinks breaks it.
- huggingface_hub **2.0** exists and transformers accepts both majors, so pin it.
- The resolve endpoint gives the commit (`X-Repo-Commit`) and sha256 (`X-Linked-Etag`). The Xet bridge serves HTTP Range requests: 8 parallel ranges ran ≥ hf_xet here.
- llama.cpp `-hf` always fetches `main`; mlx-lm's CLIs have no `--revision`. Pin by snapshot path.

**11. Architectures.** VERIFIED by R9-ARCH.
- The table covers 13 families. **Config-schema drift is silent:** after transformers 5's move to `rope_parameters`, mlx-lm loads Llama 3.1 with θ=1e4 and no scaling.
- Runtimes disagree numerically: llama.cpp uses f32 activations and an f16 KV cache, mlx-lm bf16; YaRN, LongRoPE and Gemma scaling differ.
- **Qwen3-32B:**
  - **17.99 GB of weights read per decode token** at MLX 4-bit (18,429,667,328 B in total);
  - 256 KiB of KV cache per token;
  - bound: tok/s ≤ BW/(17.99e9 + 262,144·T).

### Pitfalls a hostile reviewer will catch
- Comparing a different-size file: Q4_0's Q6_K LM head adds +1.1 % bytes/token at 32B; Q4_K_M about +9 %; imatrix mixes vary.
- **KV dtype mismatch** (bf16 vs f16), or a different flash-attention setting.
- llama-bench `tg` from an empty cache presented as decode speed. Using engine-reported rates instead of external timestamps.
- Prompt or prefix caching left on.
- TTFT with an unstated idle gap or an unstated tokenization boundary.
- Reporting footprint or RSS for an mmap engine. Reporting `mx.get_peak_memory` as RAM.
- Chip-only power reported as energy.
- Measuring sub-second phases with ~1–2 s counters.
- Charging during the run, or the battery draining on a 140 W adapter.
- Thermal drift without interleaving. 14" vs 16" chassis, or High Power Mode off.
- Greedy token-match offered as proof of quality.
- Injected template dates; thinking on in one engine only; differing EOS lists (the mlx-community repo lacks `generation_config.json`).
- Publishing from a swapping machine (this dev box had 4.1 GB of swap).

### Recommendations

**Loader:**
- Read both formats natively.
- On first use, build a **lossless, page-aligned relayout cache** keyed by the source sha256, then `mmap` + no-copy (EXPERIMENTAL).
- GGUF files can be mapped directly.
- Implement MLX's `infer_quant_config` rules exactly, and accept negative scales.

**Config:** normalise both the `rope_scaling` and `rope_parameters` schemas, and fail loudly on unknown keys. Take EOS as the union of config, generation_config and tokenizer_config.

**Text stack:**
- `tokenizers` (fancy-regex build, pinned) plus a golden-ids gate.
- minijinja + pycompat with a Python-exact `tojson`.
- UTF-8-boundary buffering for streamed output.
- Benchmarks exchange **token ids**.

**Hub:** `hf://` grammar plus `:QUANT`; offline-first; share `~/.cache/huggingface/hub` with flock-compatible writes, never writing the shared store; record commit and sha256 per result; plain HTTPS with parallel Range requests, no Xet client.

**Benchmark** — follow §7 of the notes:
- Pre-register.
- Build every quant ourselves from one pinned BF16 commit.
- Pass the correctness gates first: tokenizer ids, then a teacher-forced identical-weights KLD of ≤0.1× the quantization KLD with ≥99 % top-1 agreement.
- Run interleaved randomized blocks of N=10, with fixed sampler, KV dtype and idle gap.
- Headline metrics: **TTFT** from ids; **prefill** = n_prompt/TTFT (n ≥ 2048); **decode** = (N−1)/(t_last−t_first) at stated depths; **peak RAM** = max ΔUsed (wired + anon − purgeable + compressed) against a pre-spawn baseline; **J/token** = whole-system energy over ≥60 s loops.
- Report every speed number together with its bpw, bytes per token and KLD.
- Run on a rebooted M5 Max, on AC power, in High Power Mode, under a quiet gate.
- The JSON schema is in §8.

### Open questions
1. **What can only be checked on M5 Max hardware:**
   - whether unaligned `uint4` loads are tolerated;
   - whether no-copy GPU reads still wire pages;
   - `maxBufferLength`;
   - the IOReport counter batching interval;
   - whether `PSTR` works;
   - the `powermode` key.
2. Is `mlx_lm.convert` deterministic, and does its output equal `mlx-community/Qwen3-32B-4bit`?
3. **KV dtype:** force parity, or report each engine's default?
4. Is the KLD threshold credible to Apple reviewers? Calibrate it against BF16-vs-BF16 cross-runtime noise.
5. Who computes the BF16 reference logits? Should we also publish an independently computed (CUDA) reference?
6. Does the relayout cache count as the "same file"? Proposal: yes if it is proven bit-identical after dequantization, with its build time included in cold start.
