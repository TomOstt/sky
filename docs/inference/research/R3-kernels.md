# R3-kernels: MLX Metal kernels, Steel GEMM, fast kernels, Metal 4 / NAX (code reading)

Fork of the R3 agent. Scope: Q3 parts (a) kernel catalog + JIT vs precompiled, (b) Steel GEMM / gemv routing,
(c) fast kernels (rms_norm, rope, sdpa_vector, steel attention, sdpa_blocked, gated_delta), (d) Metal 4 / M5
Neural Accelerator ("NAX") support, plus `device.cpp` arch parsing. Quantized qmv/qmm algorithms are covered by
the parent; they appear here only where NAX routing depends on them. No GPU timing was run for these notes.

- Source: `$S/src/mlx` at HEAD `0840c42` (2026-10-01, "Fix reduction over large arrays (#4549)"), `mlx/version.h`
  = 0.32.4 (dev). The clone is shallow (one commit), so there is no git history to consult.
- Installed wheel in `$S/venv`: mlx / mlx-metal 0.32.3, tag `macosx_26_0_arm64`.
- All paths below are relative to `$S/src/mlx/` unless stated. `K/` = `mlx/backend/metal/kernels/`,
  `M/` = `mlx/backend/metal/`.
- Labels: VERIFIED means code at path:line, a maintainer statement, or our own command output. LIKELY gives the
  reasoning. Third-party claims are marked as such.

---------------------------------------------------------------------------------------------------------------

## 0. Device classes: how MLX reads the GPU architecture (VERIFIED)

- `Device::Device()` takes `arch_` from env `MLX_METAL_GPU_ARCH` if set, otherwise from
  `MTLDevice.architecture.name` (M/device.cpp:589-591). The env var is defined at mlx/utils.h:235-238.
- Generation = the two digits before the last character: "applegpu_g14s" gives 14 (M/device.cpp:592-601).
- Size class = the last character (M/device.cpp:602-624):

  | letter | MLX comment | max_ops_per_buffer / max_mb_per_buffer |
  |---|---|---|
  | `p` | phone | 20 / 40 |
  | `g` | base, pro | 40 / 40 |
  | `s` | max | 50 / 50 |
  | `d` | ultra | 50 / 50 |
  | other (e.g. `c`) | medium | 40 / 40 |

  The env vars `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` override these (mlx/utils.h:193-203,
  M/device.cpp:625-626).
- **Our M2 Pro reports `applegpu_g14s`** (VERIFIED, `mx.device_info()` in the venv). MLX's comment calls `s` "max",
  so an M2 Pro takes MLX's Max-class branches. Third-party reports say the **M5 Max is `applegpu_g17s`**
  (mlx-lm issue #1206, "Apple M5 Max (applegpu_g17s)") and the M5 Pro is also `g17s`
  (github.com/nathansutton/chad/issues/45). MLX's own comment at M/quantized.cpp:489-492 ("...on M5 Max GPUs",
  gated on `gen == 17 && arch == 's'`) agrees. So the M2 Pro and the M5 Max share every `devc == 's'` heuristic in
  MLX (LIKELY; the M5 string is third-party plus an MLX comment, not our own measurement). The heuristics they share:
  - 50 ops / 50 MB per command buffer;
  - 2-pass sdpa_vector from kL >= 1024;
  - split-K threshold 2048;
  - the `s` row of the 2-pass blocks table.
- `MLX_METAL_GPU_ARCH` changes routing at runtime. It lets you emulate other device classes (for example `...g14g`)
  on the M2 Pro for experiments. Do NOT set gen >= 17 on pre-M5 hardware. `is_nax_available()` would return true:
  macOS 26.6 satisfies the OS check, and the wheel contains the NAX kernels. MLX's NAX fragment layout is hard-coded
  (see section 4), so results would be wrong or the kernels would fail (LIKELY). A fork used the opposite trick,
  `MLX_METAL_GPU_ARCH=applegpu_g15`, to turn NAX off on M5 (github.com/PrismML-Eng/mlx/pull/4).
- `is_nax_available()` is computed once and cached in a function-static (M/device.cpp:952-970).

---------------------------------------------------------------------------------------------------------------

## (a) Kernel catalog and how each family is compiled

### Build modes (VERIFIED)

- **Default build (`MLX_METAL_JIT=OFF`, CMakeLists.txt:44).** Every family is compiled offline to `.air` and linked
  into one `mlx.metallib` (K/CMakeLists.txt:12-48, 50-196, 203-209).
  - Flags: `-fno-fast-math -Wall ...` (K/CMakeLists.txt:13-21).
  - The host looks kernels up by name (`d.get_kernel(name)`, M/nojit_kernels.cpp:9-60 and following).
  - Specialization happens when the pipeline is created, via MTL function constants
    (for example `align_M/N/K`, `has_batch` at M/matmul.cpp:409-416).
- **Always JIT-compiled at runtime, even in the default build** (sources embedded by `make_jit_source`,
  M/CMakeLists.txt:24-36):
  - gather, scatter, gather_axis, scatter_axis, masked_scatter (M/indexing.cpp:98-147, 320-324, 462-475, 706);
  - Hadamard (M/hadamard.cpp:98-101);
  - `mx.compile` fused kernels (M/compiled.cpp:263-265);
  - `mx.fast.metal_kernel` custom kernels (M/custom_kernel.cpp:50-53).
- **`MLX_METAL_JIT=ON`.** Most families become runtime-compiled strings (M/CMakeLists.txt:38-117), using
  M/jit_kernels.cpp instead of M/nojit_kernels.cpp (M/CMakeLists.txt:39, 119). The families listed at
  K/CMakeLists.txt:50-64 stay precompiled even then: arg_reduce, conv (naive/winograd/depthwise), dot,
  gather_mm_offsets, layer_norm, random, rms_norm, rope, scaled_dot_product_attention (vector + blocked),
  sdpa_vjp, gated_delta_update (+vjp), and fence (Metal >= 3.2).
- **Runtime compile settings.** Metal language 4.0 on macOS 26 and 4.1 on macOS 27 (M/device.cpp:64-80,
  662). Default math mode is Safe (mlx/backend/common/metal_kernel.h:13-14).
- **Release wheels** use toolkit `metal` (non-JIT; .github/actions/setup/action.yml:225-233) at three deployment
  targets: 14.0, 15.0, 26.2 (.github/workflows/release.yml:150-172). Only the 26.2 build can contain NAX kernels
  (section d).
- **Our installed wheel** (command: `strings mlx.metallib | grep -c _nax_`; `otool -l libmlx.dylib`):
  - `mlx.metallib` is 190,319,536 bytes and has 5,878 strings containing `_nax_`;
  - `LC_BUILD_VERSION`: `minos 26.2`, `sdk 26.5`;
  - wheel tag `macosx_26_0_arm64`.

  So the NAX kernels ship in the macOS-26 wheel and are turned off at runtime on the M2 Pro by the generation check.

### Families (VERIFIED from file names, instantiation strings and K/CMakeLists.txt; "P" = precompiled in the default build)

| Family (files) | Purpose | Build |
|---|---|---|
| unary, binary, binary_two, ternary (+`*_ops.h`) | Elementwise ops. Variants by layout: `v`, `v2`, `vn` (contiguous), `ss/sv/vs/vv`, `g1..g3`, `gn2/gn4large` (strided/broadcast). binary_two = two-output ops such as DivMod (M/binary.cpp:107, 220). | P |
| copy | Copy and cast: contiguous, general, dynamic-offset, `large` (int64 index) variants | P |
| arange | Fill with a range | P |
| reduce (+ reduction/{reduce_all,row,col,init,ops}.h) | all_reduce, row_reduce, col_reduce (small / looped / 2pass / longcolumn) | P |
| arg_reduce | argmin / argmax | P always |
| softmax (block/looped, `precise`), logsumexp, cross_entropy (+vjp) | Row-wise reductions | P |
| scan | cumsum-style scans, contiguous and strided | P |
| sort (block merge sort "mbsort"), searchsorted | Sorting | P |
| random | threefry `rbits` | P always |
| fft (+fft/radix.h, readwrite.h) | Stockham radix, Bluestein, Rader, four-step | P |
| conv.metal | naive unfold (im2col), winograd 2-D transforms, depthwise 1-D/2-D | P always |
| steel/conv | implicit-GEMM conv 2-D / 3-D / general | P |
| steel/gemm: fused, splitk(+accum), gather (gather_mm, gather_mm_rhs), masked, segmented | Dense GEMM and MoE/gather GEMMs (simdgroup_matrix 8x8) | P |
| steel/gemm `*_nax` (fused, splitk, gather, segmented) | Same, using Metal 4 `matmul2d` (NAX) | P only if Metal 4, SDK >= 26.2 and target >= 26.2 (K/CMakeLists.txt:166-194) |
| gemv, gemv_masked | Dense mat-vec: `gemv`, `gemv_t`, `gemv_wide` (2-15 vectors), block-masked | P |
| dot | `dot_product` for M=N=1 | P always |
| gather_mm_offsets | Binary search for segment offsets in sorted-rhs gather (MoE) (M/matmul.cpp:2191-2200) | P always |
| quantized (+quantized_utils.h) | affine: qmv, qmv_fast, qmv_quad, qmv_wide, qvm, qvm_split_k, qmm_t, qmm_n, qmm_t_splitk, gather_{qmv,qmv_fast,qvm,qmm_t,qmm_n,qmm_rhs}, quantize, dequantize (K/quantized.h:1549-2682) | P |
| fp_quantized (+fp4.h, fp8.h) | The same set for mxfp4 / nvfp4 / mxfp8 | P |
| quantized_nax, fp_quantized_nax | NAX qmm_t / qmm_n / gather_qmm_t / gather_qmm_rhs | NAX-gated P |
| rms_norm, layer_norm (+vjp) | Normalization | P always |
| rope | RoPE: single, freqs and large variants | P always |
| scaled_dot_product_attention.metal (sdpa_vector.h, sdpa_blocked.h) | Decode attention (1-pass, 2-pass, 2-pass-GQA, aggregation) and blocked-D512 helpers | P always |
| steel/attn (steel_attention, steel_attention_nax incl. dsplit) | Fused flash-style prefill attention | P (NAX: gated) |
| scaled_dot_product_attention_vjp | Attention backward | P always |
| gated_delta_update (+vjp, +nax, +nax_vjp, gated_delta_nax_ops.h) | Gated DeltaNet linear-attention recurrence | P always (NAX: gated) |
| hadamard.h | Fast Hadamard transform | JIT always |
| indexing/* | gather, scatter, gather_axis, scatter_axis, masked_scatter, gather_front | JIT always |
| fence.metal | `input_coherent`, `fence_update`, `fence_wait`: system-scope coherent spin fences used by M/fence.cpp:73-124 (CPU-GPU fast sync) | P if Metal >= 3.2 |
| Helper headers | bf16, bf16_math, complex, defines, erf, expm1f, fp4, fp8, atomic, logging, utils, steel/utils | - |

---------------------------------------------------------------------------------------------------------------

## (b) Steel GEMM and the routing for small M

### Matmul::eval_gpu routing, in order (VERIFIED, M/matmul.cpp:1497-1634)

1. Copy non-contiguous inputs (`check_transpose`, M/matmul.cpp:27-43).
2. Collapse batches into M when possible (M/matmul.cpp:1532-1544).
3. `M==1 && N==1`: `dot_product`, 512-thread groups with 32 items per thread, then all_reduce and a copy
   (M/matmul.cpp:1090-1136, 1552-1564).
4. **`gemv_wide`**: used when `!a_transposed && b_transposed` (x @ W.T, the nn.Linear layout) and `gemv_wide_config`
   accepts, which needs all of:
   - **gen >= 15 (M3+)**. Comment: "Pre-M3 generations are limited by load issue rate rather than bandwidth"
     (M/matmul.cpp:1357-1362).
   - fp16 or bf16 output;
   - 2 <= M with `passes=(M+4)/5 <= 3`, i.e. **M ≤ 15**;
   - N > 1, K % 4 == 0, 8-byte-aligned offsets.

   Each threadgroup streams the weight matrix once for up to 5 vectors (M/matmul.cpp:1345-1400, 1564-1589).
   **Not available on the M2 Pro (gen 14).**
5. `min(M,N)==1`: `gemv` (M/matmul.cpp:1590-1609). For x @ W.T with M=1, `transpose_mat = false`, so the kernel is
   `gemv_<T>`:
   - `bm = 8` simdgroups if out >= 4096, else 4; `sn=32`, `tm=tn=4` (M/matmul.cpp:1209-1225);
   - each simdgroup produces 4 output rows, and each lane loads 4 consecutive elements per row per step;
   - simd_sum reduction; 32 outputs per 256-thread threadgroup when out >= 4096 (K/gemv.h:29-140).

   `gemv_t` (matrix transposed): `sm/sn` = 4/8 (when K >= 8192 and out >= 2048) or 8/4; `bn` = 16/4/2 by output
   size (M/matmul.cpp:1186-1207).
6. Otherwise `steel_matmul` goes to `steel_matmul_axpby` (M/matmul.cpp:889-1087):
   - **Case 1, non-NAX split-K**: `!use_nax && batch==1 && tiles16(M)*tiles16(N) <= T && K/16 >= 8 && K >= max(M,N)`.
     T is 2048 for `s`/`d` devices and 1024 otherwise (M/matmul.cpp:966-990).
     - Tiles: bm, bn ∈ {16, 32}, bk=16, wm=wn=2.
     - Partitions: `min(max(2, next_pow2(K/16 / (tiles32(M)*tiles32(N)))), 32)`.
     - Writes fp32 partials, then a `splitk_accum` kernel sums them (M/matmul.cpp:518-669).
   - **Case 2, NAX split-K**: `use_nax && batch==1 && (K >= 3*max(M,N) || (max(M,N) <= 1024 && K > 2*max(M,N)))`
     (M/matmul.cpp:993-1015). Partition size 4096 (2048/1024/K/2 for small K), tiles 128x128x512 or 64x64x256,
     1-D grid (M/matmul.cpp:676-794).
   - Otherwise regular NAX if `use_nax` (M/matmul.cpp:1031-1056), else regular steel (M/matmul.cpp:1058-1084).
   - `use_nax = is_nax_available() && !complex && (MLX_ENABLE_TF32 || dtype != f32)` (M/matmul.cpp:963-965).

### Regular Steel GEMM tiles (VERIFIED, `GEMM_TPARAM_MACRO`, M/matmul.cpp:90-171)

| devc | Case | BM×BN×BK, WM×WN |
|---|---|---|
| `g`/`p` (small) | complex64 | 64×32×8, 4×1 |
| | nt (x @ W.T) | 64×32×32, 2×2 |
| | fp16/bf16, other layouts | 64×64×16, 1×2 |
| | fp32, other layouts | default 64×64×16, 2×2 |
| `d` (large) | large fp16/bf16 GEMM, M·N·batch >= 2^20 | 64×64×16 1×2 (2·max(M,N) > K); nt with large K: 64×32×32 2×2; nn with large K: 32×64×16 1×2 |
| | small fp16/bf16 | nt 64×32×32 2×2; nn 64×64×16 1×2 |
| | small fp32 | nt 32×64×16 1×2; nn 64×32×32 2×2 |
| everything else, incl. **`s` (M2 Pro, M5 Max) and `c`** | all | **64×64×16, WM=2, WN=2** (4 simdgroups = 128 threads) |

Kernel structure (VERIFIED):
- Threadgroup memory tiles for A and B, padded by 16 bytes (K/steel/gemm/gemm.h:37-43).
- The loader issues `n_reads = BCOLS·BROWS/threads` contiguous elements as one aligned vector
  (K/steel/gemm/loader.h:22-43).
- The main loop is **single-buffered**: barrier, load A/B tiles, barrier, MMA. There is no double buffering or
  async copy (K/steel/gemm/gemm.h:96-120).
- MMA uses `simdgroup_matrix<T,8,8>` with `simdgroup_multiply_accumulate` (K/steel/gemm/mma.h:33-47, 205).
  **The A/B fragments are `AccumType`, i.e. float**: fp16/bf16 inputs are converted to fp32 when loaded from
  threadgroup memory, so the multiply runs on fp32 8x8 simdgroup matrices (K/steel/gemm/mma.h:456, 478-480, 58-66).
  Each warp tile is (BM/(8·WM)) × (BN/(8·WN)) fragments (mma.h:459-466).
- No threadgroup swizzle in the non-NAX regular path: `swizzle_log = 0` (M/matmul.cpp:456).
- Precompiled tile sets:
  - fused: 64/64/16 2×2, 64/64/16 1×2, 64/32/32 2×2, 32/64/16 1×2, 32/32/16 2×2, 64/32/8 4×1, each for
    nn/nt/tn/tt and f16/bf16/f32/c64 (K/steel/gemm/kernels/steel_gemm_fused.metal:21-33);
  - splitk: 16/32 × 16/32 × 16 (K/steel/gemm/kernels/steel_gemm_splitk.metal:54-63).
- Other variants:
  - `AddMM` (epilogue via `use_out_source` / `do_axpby` function constants);
  - `BlockMaskedMM` (steel_gemm_masked, gemv_masked);
  - `GatherMM` (M/matmul.cpp:2880-2930): for M==1 with sorted rhs, `gather_mm_rhs` (NAX version if available);
    otherwise `gather_mv`, then `gather_mm_wide`, then `gather_mm`;
  - `SegmentedMM` (M/matmul.cpp:2935-3110, NAX-aware at 2987-3047).
- Convolutions take NAX only through `steel_matmul`: explicit-GEMM conv (M/conv.cpp:191), the stride-2
  transposed 3-D conv (M/conv.cpp:969) and Winograd (M/conv.cpp:1259). The implicit-GEMM `steel_conv` kernels have
  no NAX version (VERIFIED: grep).

**Consequence for our microbenchmarks (LIKELY):** on the M2 Pro (`s`, gen 14), every fp16 prefill GEMM measured
through MLX runs the 64×64×16, 2×2 tile, with fp32 simdgroup-matrix math and single-buffered loads. dense M=2..15
has no `gemv_wide`.

---------------------------------------------------------------------------------------------------------------

## (c) Fast kernels

### RMSNorm (VERIFIED)
- `RMS_N_READS = 4` and `RMS_LOOPED_LIMIT = 4096` (K/defines.h:15-16).
- **axis ≤ 4096**: `rms_single_row`. One threadgroup per row, `threads = 32·ceil(ceil(axis/4)/32)`
  (M/normalization.cpp:68-75). Each thread loads 4 values into registers once, then simd_sum, a threadgroup
  reduction, `precise::rsqrt`, and the write (K/rms_norm.metal:12-80). The input is read once.
- **axis > 4096**: `rms_looped` with `maxTotalThreadsPerThreadgroup` threads (M/normalization.cpp:59-61, 76-80).
  It loops once to accumulate and a **second time to normalize, re-reading x** (K/rms_norm.metal:84-160).
  **Qwen3-32B (hidden 5120) uses the looped kernel; Qwen3-8B (4096) uses single-row.**
- Output is `w * T(x * inv_rms)`: normalize in fp32, cast, then multiply by the weight in T
  (K/rms_norm.metal:144-151).
- In decode (one row) the whole op is one threadgroup, i.e. one GPU core (LIKELY negligible bytes;
  dispatch-dominated).

### RoPE (VERIFIED)
- Decode fast path `rope_single`: used when the input is row-contiguous, T==1 and there is one offset
  (M/rope.cpp:96).
  - One thread per rotation pair; grid (dims/2, B·N) (M/rope.cpp:133-138).
  - `inv_freq = exp2(-(i/(dims/2)) · log2(base))`, computed in-kernel; the host passes `log2(base)`
    (M/rope.cpp:123; K/rope.metal:56-66). Precomputed `freqs` arrays are also supported (K/rope.metal:68-81).
  - `fast::cos/sin` per pair, fp32 math, scalar loads (no vectorization) (K/rope.metal:11-52).
- Layout: `traditional` = interleaved pairs (2i, 2i+1); non-traditional = halves (i, i+dims/2), which is the
  HF "rotate_half" style used by Qwen/Llama (K/rope.metal:28-35).
- Multi-token path: `n_per_thread = 4` heads per thread share one cos/sin (M/rope.cpp:8, 141-152;
  K/rope.metal:83-140). Donates the input buffer when possible (M/rope.cpp:26-57).

### Decode attention: sdpa_vector (VERIFIED)
- **Vector mode iff q length ≤ 8** (M/scaled_dot_product_attention.cpp:1496). A fused kernel also requires
  `qL·gqa_factor ≤ 32` (M/scaled_dot_product_attention.cpp:1001-1005). Supported head dims are
  {64, 96, 128, 192, 256, 512} with matching V, or (192,128), (96,64) (:979-998).
  - **Qwen3-32B (64 q heads, 8 KV heads, GQA 8): fused vector attention only for qL ≤ 4.** qL 5-8 falls back to
    the unfused composition: scale-mul, matmul QKᵀ, where/add mask, softmax, matmul V, with GQA by
    unflatten/broadcast (mlx/fast.cpp:819-870, 930-940). Qwen3-8B (GQA 4) is fused up to qL 8.
- **1-pass vs 2-pass** (M/scaled_dot_product_attention.cpp:1561-1570): 2-pass if
  `((devc=='d' || devc=='s') && kL >= 1024) || (n_kv_heads < n_q_heads && kL >= 4096)`, else 1-pass. So on `s`
  devices (M2 Pro and LIKELY the M5 Max) 2-pass starts at kL = 1024; on `g`/`c`, GQA models switch at kL = 4096.
- **1-pass** (`sdpa_vector`, K/sdpa_vector.h:16-189):
  - Threadgroup of 1024 threads (32 simdgroups) per (batch·q_head, q_pos) (M/scaled_dot_product_attention.cpp:678-679).
  - Simdgroup `g` walks keys g, g+32, ...; each lane holds D/32 elements of q and k (4 for D=128).
  - The score is a lane dot product plus `simd_sum`, then an online softmax in fp32 (K/sdpa_vector.h:43-48, 99-140).
  - Cross-simdgroup merge goes through threadgroup memory (:146-180).
  - **K/V for a KV head is fetched by each of its gqa_factor q-head threadgroups** (no sharing in-kernel;
    reuse only via cache) (:65-69).
- **2-pass** (`sdpa_vector_2pass_1` + `_2`):
  - Grid (n_kv_heads, B, blocks), threadgroup (32, gqa_factor, qL): one simdgroup per (q head of the group, q pos)
    (M/scaled_dot_product_attention.cpp:817-818).
  - Block b walks keys b, b+blocks, ... (K/sdpa_vector.h:263-300). Writes partial o/sum/max, then pass 2
    (1024 threads) merges them (M/scaled_dot_product_attention.cpp:897-918).
  - Blocks (M/scaled_dot_product_attention.cpp:774-812):
    - `s`: 64, or with N > 1024 and gqa·qL > 4: 128 (N ≤ 8K), 256 (≤ 32K), 512 (≤ 64K), 1024 above;
    - `d`: 128, 256, 512 or 1024 by rule;
    - other: 64 if gqa·qL >= 4, else 32;
    - `MLX_SDPA_BLOCKS` overrides (rounded up to a multiple of 32).
- **Duplication-free GQA variant** `sdpa_vector_2pass_1_gqa`: each simdgroup computes HPT heads over a contiguous
  token chunk, so each K/V byte is read G/HPT times instead of G times (K/sdpa_vector.h:319-325).
  - Used only for qL == 1, no mask, no sinks, D==V, kL >= 8192 and (G=8 with D ∈ {64,128}, or G ∈ {12,16} with
    D=128) (M/scaled_dot_product_attention.cpp:755-763).
  - Instantiated HPT: (64,G8,HPT8), (128,G8,4), (128,G12,4), (128,G16,2)
    (K/scaled_dot_product_attention.metal:68-71).
  - **Qwen3-32B at kL >= 8192 gets it: K/V read 2× instead of 8×.**
- **Quantized KV cache: no fused kernel in MLX core** (VERIFIED: no "quant" in the sdpa sources). mlx-lm runs
  quantized-KV attention as `quantized_matmul`, then mask, `softmax(precise=True)`, then `quantized_matmul`. That
  materializes the scores ($S/src/mlx-lm/mlx_lm/models/base.py:64-102).

### Prefill attention (VERIFIED)
- Full mode for qL > 8 (M/scaled_dot_product_attention.cpp:1572-1622).
  - Fused head dims: {64, 72, 80, 96, 128, 192, 256}, (96,64), and 512 on NAX (:950-967).
  - `use_fallback` still picks the unfused path when qL > 8 and D ∈ {192, 256}, apart from the NAX/causal special
    cases (:1431-1451).
- **Non-NAX `steel_attention`** (M/scaled_dot_product_attention.cpp:503-646; kernel
  K/steel/attn/kernels/steel_attention.h):
  - Tiles: BQ=32; BK=16 for D >= 128 (32 for D < 128; 32 for D=256 half on `d`); WM=4, WN=1 (2 for D=256).
    That is 128 threads for D=128.
  - Grid (ceil(qL/32), H, B): **no split over the KV length**.
  - Q in threadgroup memory; K and V share one threadgroup buffer (:134-139).
  - exp2 softmax with log2(e) folded into the scale (:177).
  - **Causal**: KV blocks past the diagonal are skipped (`kb_lim`); the diagonal block is masked
    (steel_attention.h:282-296, 386-400).
  - GQA: `kv_head = head / gqa_factor`, with each q-head threadgroup re-streaming K/V (:97).
  - Instantiated: (BQ,BK,D) = (32,16,192), (32,16,128), (32,32,{96,80,72,64}), (32,32,96/64), and 256 with WN=2
    (K/steel/attn/kernels/steel_attention.metal:18-35).
- **NAX attention** `attention_nax`, used on NAX devices for D ∈ {64, 96, 128, 256, 512}
  (M/scaled_dot_product_attention.cpp:412-427):
  - BQ=64 (32 for D=512), BK=32; WM=4 (2 for D=512); WN=1, or D/128 for the head-dim-split `dsplit` kernel at
    D=256/512 (:218-225).
  - Q/K/V tiles go straight from device memory into NAX registers; no threadgroup memory except the dsplit
    exchange (K/steel/attn/kernels/steel_attention_nax.h:111, 155-229, 655).
  - Causal block skip as above (:188-209).
  - D=72/80 can be zero-padded to 96 to reach NAX, by default only for qL, kL >= 512 with no mask
    (M/scaled_dot_product_attention.cpp:429-494).
- **`sdpa_blocked.h` (new)**: a NON-NAX fallback for **D=512 causal prefill**.
  - Condition: f16/bf16, qL >= 1024, qL ≤ kL, no array mask, no sinks, no LSE
    (M/scaled_dot_product_attention.cpp:20-33).
  - Processes 256 query rows at a time: scale-copy Q, steel GEMM QKᵀ into a [B,H,256,kL] score buffer, a causal-mask
    kernel, softmax, steel GEMM with V, then copy out (M/scaled_dot_product_attention.cpp:70-200;
    K/sdpa_blocked.h:11-45).
  - Not flash attention: it materializes scores per block.

### gated_delta_update (VERIFIED)
- Fused Gated DeltaNet recurrence used by the mlx-lm models qwen3_next, qwen3_5, kimi_linear, kimi_k3,
  bailing_moe_v3 and olmo_hybrid (grep of $S/src/mlx-lm/mlx_lm/models).
- Chunked: chunk 16 on NAX, 8 without NAX, for T > 8; chunk 1 (sequential) for decode T ≤ 8. Overrides:
  `GATED_DELTA_CHUNK`, `GATED_DELTA_CKPT` (M/gated_delta_update.cpp:17-37).
- Shapes are limited to Dk=Dv=128 and (Hk,Hv) ∈ {(24,24), (32,32), (16,32), (16,48), (16,16), (16,64)}
  (M/gated_delta_update.cpp:39-46).
- The NAX version uses `matmul2d` 16×16×32 and 16×32×16 (K/gated_delta_nax_ops.h:249-281, 306-335, 360-396).
- **Not relevant to dense Qwen3-32B.**

---------------------------------------------------------------------------------------------------------------

## (d) Metal 4 / M5 Neural Accelerators ("NAX")

### Detection and requirements (VERIFIED)
- **Build time.** NAX kernels are compiled only if `MLX_METAL_VERSION >= 400` (Metal 4), the macOS SDK is >= 26.2
  and `CMAKE_OSX_DEPLOYMENT_TARGET >= 26.2`. Otherwise the build warns and defines `MLX_METAL_NO_NAX`
  (K/CMakeLists.txt:166-194; JIT build: M/CMakeLists.txt:89-116). The Metal version is probed via `__METAL_VERSION__`
  with `-mmacosx-version-min=<target>` (CMakeLists.txt:217-224).
- **Run time.** `is_nax_available()` (M/device.cpp:952-970) returns false under `MLX_METAL_NO_NAX`; otherwise it
  requires `__builtin_available(macOS 26.2, iOS 26.2, ...)` **and** `gen >= 17` (`gen >= 18` for phones `p`). M5
  family = gen 17; A19-class phones = gen 18 (LIKELY from the gate).
  - In a 26.2-target binary the OS check folds to true. Issue ml-explore/mlx#4525 reports that the 26.2-built wheel
    is tagged `macosx_26_0` and installs on 26.0/26.1. Maintainer zcbenz: "Documenting the behavior or checking real
    OS version sound good to me, but I think in practice it is not going to be worth the efforts fixing 26.0/26.1"
    (https://github.com/ml-explore/mlx/issues/4525).
- **No Python API.** No `is_nax_available` binding (grep of python/src: none). Maintainer jagrit06: use Metal GPU
  captures to see it, or read the dispatch code (https://github.com/ml-explore/mlx/issues/3182). zcbenz: "MLX makes
  uses of Neural Accelerators, you can search 'nax' in the source code"
  (https://github.com/ml-explore/mlx/issues/3217).

### Programming model (VERIFIED)
- `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`; `mpp::tensor_ops::matmul2d<desc,
  metal::execution_simdgroup>`. Operands are cooperative tensors from `get_left/right_input_cooperative_tensor`
  and `get_destination_cooperative_tensor`, filled element-wise from MLX register fragments and run with
  `gemm_op.run(ct_a, ct_b, ct_c)` (K/steel/gemm/nax.h:12, 386-528).
- **Descriptor: `matmul2d_descriptor(16, 32, 16, transpose_a, transpose_b, /*relaxed_precision*/ true,
  multiply_accumulate)`.** M×N×K = 16×32×16 per simdgroup call (K/steel/gemm/nax.h:401-411, 473-483). Attention
  uses the same in K/steel/attn/nax.h:401-411, 473-483; gated-delta also uses 16×16×32.
- MLX's fragment is 16×16 with 8 elements per lane (2 rows × 4 cols); the lane-to-element map is hard-coded in
  `BaseNAXFrag::get_coord` (K/steel/gemm/nax.h:27-58). Accumulators are float (`NAXTile<AccumType=float,...>`,
  K/steel/gemm/gemm_nax.h:45).
- **Dense NAX GEMM.**
  - Loads A and B **directly from device memory into registers**; no threadgroup staging (K/steel/gemm/gemm_nax.h:52-99).
  - Per-simdgroup tile SM×SN = (BM/WM)×(BN/WN), inner step SK=32 (K/steel/gemm/kernels/steel_gemm_fused_nax.h:150-158).
  - Host tiles: default 128×128×512 with 4×4 simdgroups (512 threads). For `s`/`c` (M5 Max): **BM=64, BN=128,
    BK=256** (BK=64 when K >= 8192 and K > M+N), WM=2, WN=4 (256 threads, 32×32 output per simdgroup),
    `swizzle_log=2`. For `d`: 128×64 or 64×128 (M/matmul.cpp:206-230, 287-293).
  - Precompiled set: K/steel/gemm/kernels/steel_gemm_fused_nax.metal:23-35.
- **Dtypes:** fp16, bf16, fp32 inputs (f32 only when `MLX_ENABLE_TF32` = 1, the default, mlx/utils.h:225-228);
  complex excluded (M/matmul.cpp:963-965).
  - Maintainer angeloskath: "TF32 is only supported by NAX. So this flag enables TF32 if NAX is available for this
    operation." (https://github.com/ml-explore/mlx/issues/3235). Third-party M5 Max errors under default TF32:
    max rel err 1.9e-3 vs 6.0e-8 with `MLX_ENABLE_TF32=0` (https://github.com/ml-explore/mlx/issues/3860).
  - **No integer `matmul2d` anywhere in MLX:** quantized NAX kernels dequantize to T first (section below).

### Which ops use NAX (VERIFIED)

| Op | NAX? | Condition / note |
|---|---|---|
| Dense GEMM (`Matmul`, `AddMM`, conv via explicit GEMM/Winograd) | yes | `steel_matmul_axpby` when use_nax; NAX split-K for K ≫ M,N (M/matmul.cpp:963-1056) |
| Dense GEMV (M=1 or N=1), `gemv_wide`, `dot_product` | **no** | routed before `steel_matmul` (M/matmul.cpp:1552-1609) |
| GatherMM (MoE) | partly | M==1 with sorted rhs gives `gather_mm_rhs_nax` (M/matmul.cpp:2906-2911); SegmentedMM NAX (2987-3047) |
| Quantized matmul, prefill/large M | yes | `qmm` goes to `qmm_nax` when `is_nax && (transpose \|\| affine) && K%64==0 && (transpose \|\| N%64==0)` (M/quantized.cpp:1062-1083). Tiles BM=64 (32 if transpose and M ≤ 32), BN=64, BK=64, WM=WN=2 (M/quantized.cpp:836-844). Weights are dequantized into threadgroup memory as T by `QuantizedBlockLoader`; activations load straight into NAX fragments; `matmul2d` runs on T (K/quantized_nax.h:925-1060). Modes: affine 2/3/4/5/6/8-bit × gs 32/64/128, mxfp4 (gs32), nvfp4 (gs16), mxfp8 (gs32); types f32/f16/bf16 (K/quantized_nax.metal:94-112; K/fp_quantized_nax.metal:113-120) |
| Gather qmm (MoE) | yes | `gather_qmm_nax` (transpose, K%64==0) and `gather_qmm_rhs_nax` (M/quantized.cpp:1283-1300, 1678-1690) |
| **Decode qmv (M=1 … vector_limit−1)** | **no** | `QuantizedMatmul::eval_gpu` sends `M < vector_limit` to `dispatch_qmv` (qmv_quad / qmv_wide / qmv / qmv_fast, plain SIMD) (M/quantized.cpp:1825-1866, 1890-1935). On gen ≥ 17 non-ultra, `vector_limit` = 33 (K,N ≤ 2048), 25 (≤ 4096), **13 otherwise** (M/quantized.cpp:89-99). For Qwen3-32B shapes, M ≤ 12 never touches NAX |
| Quantized M ≥ vector_limit with **few output tiles** | **no** | Transposed non-batched first tries `qmm_splitk` (no NAX variant), aiming for about 512 threadgroups: `split_k = max(1, 512/(ceil(N/32)·ceil(M/32)))`. Only when that ends ≤ 1 does it go to `qmm`, and then NAX (M/quantized.cpp:1152-1200, 1893-1900). See below |
| Prefill attention | yes | `attention_nax` for D ∈ {64, 96, 128, 256, 512} (and D 72/80 padded) |
| Decode attention (sdpa_vector) | **no** | SIMD kernels only |
| gated_delta_update (T > 8) | yes | chunk 16 with matmul2d |
| RMSNorm, RoPE, softmax, elementwise, reductions | no | — |

**Worked routing example, Qwen3-32B 4-bit on an M5 Max (`g17s`), derived from the code above (VERIFIED
arithmetic, not measured).** For 13 ≤ M ≤ 32:
- q_proj (N=8192): 256 tiles, split_k 2, non-NAX split-K.
- k/v_proj (N=1024): split_k 16, non-NAX.
- o_proj (K=8192, N=5120): split_k 2, non-NAX.
- down_proj (K=25600, N=5120): split_k 3, reduced to 2 by the K-divisibility loop, non-NAX.
- **Only gate/up (N=25600: 800 tiles) and lm_head reach `qmm_t_nax`.**

From M ≥ 33 (two M-tiles), q_proj (512 tiles) also reaches NAX; k/v_proj, o_proj and down_proj stay on split-K
until the tile count passes 256.
- An issue measured this cliff on an M5 Max: M=33 ran 21% faster than M=32 at N=6656. Maintainer zcbenz closed it:
  "we are unable to optimize for all possible cases, currently we are only focusing on real cases happened during
  inference" (https://github.com/ml-explore/mlx/issues/4198).
- An earlier report found qmm_splitk 1.5-1.8× slower than fp16 at M=96/128 on NAX hardware (closed,
  https://github.com/ml-explore/mlx/issues/3584).

**Attention in the speculative-verify window (VERIFIED code; numbers are third-party).**
- qL ≤ 8 (and qL·gqa ≤ 32) uses the vector kernels. Above that, `attention_nax` with BQ=64 and no split-KV.
- Issue #3826 (M5 Max, GQA 32/8, D=128, fp16): L=8 takes 0.813 ms, L=12 1.906 ms, flat until L=48, at S=16K.
- A contributor's prototype extending the 2-pass kernel to qL ≤ 16 gave 1.7-1.8× at L=12 on an M3 Max.
- Maintainer zcbenz closed it as won't-fix: "We inquired experts who serve models in large scale with MTP and
  conclusion is that the mentioned case is likely not going to happen in production"
  (https://github.com/ml-explore/mlx/issues/3826).

### NAX correctness and toolchain history (VERIFIED issue titles and states, `gh issue list -R ml-explore/mlx --search NAX`)
- #3586: Xcode 26.5 / metalfe-32023.883 mis-compiles the NAX GEMM kernel (closed).
- #3797: NAX split-K bf16 garbage on the M5 Max at the M·N ≥ 2048² boundary (closed).
- #3887 / #3856: gather_qmm NAX corruption for K % 64 ≠ 0 or seq % 32 ≠ 0 (closed).
- #4597 / PR #4594: cooperative-tensor build fixes for the macOS 27 SDK (closed / merged).
- A downstream fork disabled NAX on gen 17 entirely, citing wrong fp16 GEMM and qmm_t results on "M5-class
  (applegpu_g17s)" (https://github.com/PrismML-Eng/mlx/pull/4). Upstream did not do this; upstream still gates at
  gen >= 17 (M/device.cpp:965). The fork's symptom matches #3586's compiler issue (LIKELY, unconfirmed).

### Third-party NAX throughput data (NOT VERIFIED by us; base M5, 10-core GPU, macOS 27)
https://github.com/AI-native-Systems-Research/scratchy/issues/118 reports, with operands in registers or threadgroup
memory:
- f16 `matmul2d` 16.3-16.6 TFLOP/s;
- **int8×int8 31-33 TOPS (2× f16)**;
- fp8 not double-rate (16.4); fp4 slower (12.9-13.5);
- stream-read bandwidth 124.6 GB/s;
- W4A8 prefill GEMMs 1.10-1.24× faster than MLX's `affine_qmm_t_nax`.

MLX uses only float operands for `matmul2d` (above), so if the int8 rate holds it is untapped headroom for
prefill (EXPERIMENTAL).

---------------------------------------------------------------------------------------------------------------

## Implications for a whole-model compiler (labelled)

1. **Decode GEMV is outside NAX entirely** (VERIFIED routing). On the M5 the headline decode path is plain-SIMD
   qmv, the same class of kernel as on the M2 Pro. NAX helps prefill and large verify batches only.
2. **Verify-window gaps on the M5 (VERIFIED routing; magnitude from third-party issues).**
   - qL 5-8 on GQA-8 models uses unfused attention;
   - qL 9-48 hits the BQ=64 plateau with no split-KV;
   - M 13-32 quantized projections (all but gate/up/lm_head) use non-NAX split-K.

   Maintainers have declined both attention and routing fixes (#3826, #4198). A compiler that owns its own
   routing and tile tables for spec-decode shapes can target these directly (EXPERIMENTAL; benefit unmeasured on
   the M5).
3. **Non-NAX Steel GEMM** (the M2 Pro and every pre-M5 Mac) uses one 64×64×16 tile for `s`/`c` devices: single
   buffering and fp32 simdgroup-matrix math. LIKELY headroom exists for prefill on pre-M5 hardware, but measure
   before claiming. Ask the parent's measurement D for the TFLOPS vs peak.
4. **RMSNorm looped path at 5120** re-reads the input (2 passes), and decode RoPE / RMSNorm are tiny single-purpose
   dispatches. These are fusion candidates (LIKELY small in bytes; mostly dispatch count).
5. **1-pass sdpa_vector re-reads K/V per q head** for GQA (8× for Qwen3-32B) up to kL 1024 on `s`. The
   duplication-free GQA kernel only starts at kL ≥ 8192. For mid contexts (1K-8K) the 2-pass kernel still issues
   per-head K/V loads that rely on cache hits (LIKELY modest; measure).
6. **No fused quantized-KV attention** in MLX (VERIFIED). A fused kernel is a clear differentiator if KV
   quantization is used.
7. **int8 `matmul2d` (W8A8 / W4A8 prefill) is unused by MLX** (VERIFIED). The 2× rate is third-party (MOONSHOT
   until measured on M5 Max).
