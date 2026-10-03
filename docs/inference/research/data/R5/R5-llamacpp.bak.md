# R5 — llama.cpp / ggml on Apple Silicon: code trace + measurements

Author: research agent R5 (one of ten). Date: 2026-10-01.
Scope: how llama.cpp's Metal backend works, why it is fast on Apple GPUs, where it leaves performance on
the table, and what we measured on this machine. Evidence labels: **VERIFIED** (code `path:line` at the
pinned commit, official URL, maintainer statement, or our own measurement with command + raw output),
**LIKELY** (reason given), **EXPERIMENTAL** (our inference or extrapolation; needs a test), **MOONSHOT**.

All `path:line` citations are relative to the llama.cpp clone at
`$S/src/llama.cpp`, commit **81e39ad34368329b5db77620721510fd91107e80** (2026-10-01 19:55 +0300,
"llama : clamp kpool re-pool bound to existing pools (#29805)"; shallow clone, depth 1; `llama-bench`
reports `build: 81e39ad (1)`). `S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`.

---------------------------------------------------------------------------------------------------

## 0. Provenance, environment, and two setup fixes

* Machine: Apple M2 Pro, 16-core GPU, 16 GB, 200 GB/s spec; macOS 26.6.1. llama.cpp reports
  `GPU family: MTLGPUFamilyApple8 (1008)`, `MTLGPUFamilyMetal4 (5002)`, `has bfloat = true`,
  `has tensor = false`, `use residency sets = true`, `use shared buffers = true`,
  `recommendedMaxWorkingSetSize = 12713.12 MB` (VERIFIED, our run of `llama-bench --help`).
* **Build fix (VERIFIED).** The shared build failed with `fatal error: 'array' file not found`
  ($S/llama-build.log). Root cause: a stale, partial libc++ header directory
  `/Library/Developer/CommandLineTools/usr/include/c++/v1` (11 leftover entries dated 2022/2023) sits
  first on clang's search path and shadows the SDK's complete `MacOSX.sdk/usr/include/c++/v1` (185
  entries). Workaround without sudo: `-DCMAKE_CXX_FLAGS="-nostdinc++ -isystem $SDK/usr/include/c++/v1"`
  (same for OBJCXX). Script: `$S/research/R5-build.sh`, log `$S/research/R5-build.log`; built under the
  bench lock in ~2 min; `$S/llama.done` created. Targets: llama-bench, llama-cli, llama-server,
  llama-perplexity, llama-speculative, llama-speculative-simple, llama-lookup, llama-batched-bench,
  test-backend-ops, llama-completion. (Note for Sky's own toolchain: any C++ in Sky's build on this
  machine will hit the same shadowing.)
* **Model fix (VERIFIED).** `unsloth/Qwen3-8B-GGUF` has **no Q4_0 file** (repo tree lists Q4_1, Q4_K_M,
  Q4_K_S, IQ4_NL, ... but not Q4_0), so `hf download --include "*Q4_0.gguf"` silently fetched nothing.
  I fetched `bartowski/Qwen_Qwen3-8B-GGUF/Qwen_Qwen3-8B-Q4_0.gguf` (4,787,332,640 B) and renamed it to
  `$S/models/Qwen3-8B-Q4_0.gguf` (provenance note in `$S/models/.dl-8b-q4_0.lock/README`). Another agent
  separately fetched `Qwen3-8B-Q4_K_M.gguf` (unsloth). Consequences for comparisons:
  - the 8B "Q4_0" file is **not pure Q4_0**: 249 tensors Q4_0, 4 tensors Q4_1 (some `ffn_down`), the
    `output.weight` head is **Q6_K** (510.5 MB) — standard llama-quantize behaviour (B-bytes.txt).
  - the 0.6B (unsloth) has **tied embeddings**: no `output.weight`; `token_embd` is Q6_K (127.6 MB) and is
    read in full every token as the LM head.

---------------------------------------------------------------------------------------------------

## 1. Metal backend architecture (ggml/src/ggml-metal/*)

File map at this commit (VERIFIED, `wc -l`): `ggml-metal.cpp` (1053, ggml-backend glue),
`ggml-metal-context.m` (789, graph → command buffers), `ggml-metal-ops.cpp` (5751, one encoder function
per op), `ggml-metal-device.m` (2515, MTLDevice/library/pipelines/buffers/residency),
`ggml-metal-device.cpp` (2576, pipeline selection + specialization), `ggml-metal-fusion.cpp` (1122,
fusion table), `ggml-metal-common.cpp` (507, memory ranges + graph reorder), `ggml-metal-tuning.cpp`
(1172, offline-tuned FA-vec table), `kernels/*.metal` (34 per-kind libraries; `mul_mv.metal` 3398 lines,
`mul_mm.metal` 973, FA split into `fa_*.metal` / `fa_vec_*.metal` per KV type).

### 1.1 How a ggml graph becomes Metal command buffers (VERIFIED)

* `ggml_backend_metal_init` sets `n_cb = 1` (ggml-metal.cpp:627): **one extra encoder thread**, i.e. two
  command buffers per graph. Max is `GGML_METAL_MAX_COMMAND_BUFFERS 8` (ggml-metal-context.m:21); values
  >2 log a warning "not recommended and can degrade the performance" (ggml-metal-context.m:718-720).
* `ggml_metal_graph_compute` (ggml-metal-context.m:474-660):
  - `n_main = MAX(64, 0.1*n_nodes)` nodes are encoded **by the calling thread first**, into a command
    buffer that is `enqueue`d and committed as soon as encoding finishes (ggml-metal-context.m:485,
    556-568, 768-770) so the GPU starts while the rest is still being encoded. Comment: "tests on M1 Pro
    and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2" (:498).
  - the remaining nodes are split evenly across `n_cb` command buffers encoded **in parallel** with
    `dispatch_apply(n_cb, ctx->d_queue, encode_async)` on a concurrent GCD queue (:595, :140).
    All command buffers are `enqueue`d up front in order (:575-593), which fixes their GPU execution order
    regardless of which thread finishes encoding first.
  - command buffers are created with `commandBufferWithUnretainedReferences` (:557, :576) — no
    per-resource retain/release cost.
  - **graph_compute returns without waiting** (`waitUntilCompleted` only in capture/debug mode, :598-656).
    `ggml_metal_synchronize` waits on `cmd_buf_last` (:254-259) and checks status.
* Each command buffer gets **one compute encoder** for its whole node range
  (`ggml_metal_op_init` → `ggml_metal_encoder_init`, ggml-metal-ops.cpp:43). The encoder is created with
  `computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent` when concurrency is on
  (ggml-metal-device.m:834-845). Kernel args go through `setBytes` (inline constants), buffers through
  `setBuffer:offset:` per dispatch (ggml-metal-device.m:877-883); no argument buffers, **no indirect
  command buffers** anywhere in ggml-metal (`grep IndirectCommand` → 0 hits).

### 1.2 Concurrency / barrier handling (VERIFIED)

* Default on; `GGML_METAL_CONCURRENCY_DISABLE` turns it off (ggml-metal-context.m:142).
* Hazard tracking per encoder (ggml-metal-ops.cpp:220-242, 529-533; ggml-metal-common.cpp:51-228): every
  encoded node adds its src ranges and dst range (buffer id + [begin,end) from `tensor->data` and the
  buffer type's alloc size). Before encoding a node, it is checked against the accumulated set: it may run
  concurrently only if it does not write any previously read/written range and does not read any
  previously written range. Otherwise the encoder emits
  `memoryBarrierWithScope:MTLBarrierScopeBuffers` (ggml-metal-device.m:892-893) and clears the set.
  RAW/WAR/WAW on overlapping byte ranges ⇒ barrier; read-read overlap is allowed (:176-179).
* Some ops force a barrier explicitly (e.g. FA vec's two-kernel split + reduce,
  ggml-metal-ops.cpp:3808 "sync the 2 kernels"; FA pad kernel; ssm/cumsum).

### 1.3 Graph optimization (reordering) and op fusion (VERIFIED)

* **Reorder pass** `ggml_graph_optimize` (ggml-metal-common.cpp:453-507), invoked via the backend's
  `graph_optimize` hook from the scheduler at graph-split time (ggml-backend.cpp:1463-1478;
  ggml-metal.cpp:564-574), i.e. **once per (re)built graph, not per token** when the graph is reused.
  Algorithm (:264-451): greedy list scheduling over "fused node groups": when the next node conflicts with
  the current concurrent set, look ahead up to `N_FORWARD = 64` nodes (:404) for nodes that are
  independent of both the current set and all skipped nodes, and hoist them into the current concurrent
  set. Only "safe" ops may be moved (MUL_MAT, MUL_MAT_ID, ROPE, (RMS_)NORM, GET/SET_ROWS, CPY, binary
  ops, GLU, UNARY, ... :337-369). Effect: e.g. Q, K, V projections (+ their norms/ropes) land in one
  barrier-free group. Disable with `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` (ggml-metal-context.m:151).
* **Fusion** — single source of truth table `ggml_metal_fusions` (ggml-metal-fusion.cpp:674-701,
  ids in ggml-metal-fusion.h:29-41), shared by the optimizer (structural match) and the encoders (full
  match incl. buffer placement). Max 16 nodes per fused kernel (ggml-metal-fusion.h:19). Complete list:
  1. `NORM_MUL`: NORM or RMS_NORM + MUL (weight)                         — e.g. attn_norm·γ
  2. `NORM_MUL_ADD`: NORM or RMS_NORM + MUL + ADD (bias)
  3. `NORM_SCALE`: NORM or RMS_NORM + SCALE
  4. `ADD_CHAIN`: 2–7 consecutive ADDs (multi-input add)
  5. `SNAKE`: MUL+SIN+SQR+MUL+ADD (audio models)
  6. `GDN_CACHE`: GATED_DELTA_NET + CPY into the recurrent cache (Qwen3-Next-style linear attention)
  7. `TOPK_MOE`: SOFT_MAX + ARGSORT + GET_ROWS (+norm/scale) — MoE router (4 variants)
  8. `MOE_REDUCE`: MUL + expert VIEWs + ADD chain (2–8 experts) — MoE output weighted sum
  9. `SSM_CONV_SILU`: SSM_CONV + SILU
  Disable with `GGML_METAL_FUSION_DISABLE`; stats with `GGML_METAL_FUSION_DEBUG` (ggml-metal-device.m:
  1300-1304). The fused norm kernel takes up to two extra operand buffers and writes only the last node's
  output (ggml-metal-ops.cpp:4156-4232).
* **What is NOT fused (VERIFIED by absence in the table + Qwen3 graph):** no GEMV/GEMM epilogues
  (bias, activation, residual), no RMSNorm→GEMV prologue, no QKV merge for GGUFs that store Q/K/V
  separately (Qwen3 GGUFs do; `create_tensor_qkv` only uses a fused `attn_qkv` tensor if the file has one,
  src/llama-model.cpp:3405-3436), no gate+up merge, no RoPE+KV-cache-write fusion. SwiGLU is a single
  ggml op (`ggml_swiglu_split`, src/llama-graph.cpp:1872) — one kernel, but separate from both GEMVs.

### 1.4 Residency sets (VERIFIED)

* Every Metal buffer (allocated or mapped) gets its own `MTLResidencySet` (macOS 15+): `addAllocation`
  for each underlying MTLBuffer, `commit`, `requestResidency` (ggml-metal-device.m:2036-2066).
* A device-wide collection plus a **background heartbeat** (GCD block) calls `requestResidency` on all
  sets every 5 ms for `keep_alive = 180 s` after the last graph compute
  (ggml-metal-device.m:925-1031; env `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`); each `graph_compute` re-arms the
  timer (`ggml_metal_device_rsets_keep_alive`, ggml-metal-context.m:491; ggml-metal-device.m:1439-1445).
  Purpose: keep model memory wired so macOS does not page/compress it between requests. A dummy blit is
  issued at init as a workaround for residency-set memory not being released (links to
  developer.apple.com/forums/thread/839089 and llama.cpp issue #25937, ggml-metal-device.m:1018-1024).
  Disable: `GGML_METAL_NO_RESIDENCY` (:1264-1267).

### 1.5 Buffers from the mmapped GGUF (VERIFIED)

* Load mode `auto` (default) ⇒ mmap (src/llama-model-loader.cpp:559). If the device supports
  `buffer_from_host_ptr` (Metal: yes, ggml-metal.cpp:698, 823) and the buffer type is the default one, the
  **mmapped file region holding the tensors is wrapped directly** (src/llama-model.cpp:1854-1868).
* `ggml_metal_buffer_map` (ggml-metal-device.m:2176-2270): page-aligns the pointer *down* and the size
  *up* to `sysconf(_SC_PAGESIZE)` (16 KiB on Apple Silicon), then
  `newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared deallocator:nil` — **zero copy**.
  If the region exceeds `maxBufferLength`, it creates several **overlapping views** with overlap
  ≥ the largest tensor + 1 page so every tensor fits in exactly one view (:2226-2252). Tensor → buffer
  lookup returns (MTLBuffer, offset) at encode time.
* GGUF alignment (32 B default) is only relative to the data section; page alignment is handled by the
  down-rounding above. Tensors are used **in their on-disk block layout** — no repacking for Metal
  (repacking exists only for CPU backends: `--repack`).
* Non-mapped buffers (KV cache, compute buffers) use `vm_allocate`'d host memory wrapped NoCopy (shared
  storage) on unified-memory devices, private storage otherwise (ggml-metal-device.m:2110-2171,
  `use_shared_buffers = has_unified_memory`, :1268-1279).

### 1.6 Pipeline compilation (VERIFIED)

* Default build embeds the **MSL source** (`GGML_METAL_EMBED_LIBRARY`), 34 per-kind libraries
  (mul_mv, mul_mm, fa_vec_q8_0, ...). At device init all 34 are compiled **at runtime** in parallel with
  `newLibraryWithSource` on a GCD global queue (ggml-metal-device.m:331-430, 455-495), with preprocessor
  macros `GGML_METAL_HAS_BF16` / `GGML_METAL_HAS_TENSOR` decided per device (:452-462) and MSL language 4.0
  when the tensor API is on (:201-207).
  - Measured (VERIFIED): first-ever run on this machine
    `loaded 34 libraries from embedded data in 4.841 sec (max single = 4.841 sec)`; every later process
    start `0.014–0.048 sec` (A*.err files) — the OS Metal shader cache absorbs it. So cold TTFT after an
    install/upgrade pays ~5 s once; warm starts pay ~nothing.
  - Alternative: precompiled `default.metallib` (+ separate `ggml-tensor.metallib` for tensor-API mul_mm)
    when not embedding (:497-545).
* **Specialization by function constants**: concrete pipelines are created lazily the first time an op
  shape needs them: `newFunctionWithName:constantValues:` + `newComputePipelineStateWithFunction`,
  cached in a dictionary under a lock keyed by a formatted name string (ggml-metal-device.m:721-818). E.g.
  mul_mv bakes `nsg, ne12, r2, r3, split` (ggml-metal-device.cpp:1072-1088); mul_mv_ext adds `nxpsg`;
  FA bakes mask/sinks/bias/softcap/kvpad/sparse/ns10/ns20/nsg/nwg (:1843-1869). Function-constant ids are
  namespaced per kernel family (ggml-metal-impl.h:103-123). No `MTLBinaryArchive` use (grep: 0 hits).
* Per-dispatch CPU cost therefore includes an `snprintf` of the pipeline name + locked dictionary lookup
  per node per token (ggml-metal-device.cpp:1072-1075 → ggml-metal-device.m:700-718). (EXPERIMENTAL: a
  small but non-zero part of encode time; ICB/command reuse would remove it.)

---------------------------------------------------------------------------------------------------

## 2. Decode kernels (GEMV, `kernels/mul_mv.metal`)

### 2.1 Which kernel runs (VERIFIED, ggml-metal-ops.cpp:2423-2648, ggml-metal-common.cpp:35-42)

| src1 columns `ne11` (= tokens in the ubatch) | non-K quants (Q4_0/Q4_1/Q5_x/Q8_0/MXFP4/IQ4_NL/F16/BF16/F32) | K-quants (Q2_K..Q6_K) |
|---|---|---|
| 1 | `mul_mv` | `mul_mv` |
| 2–3 | `mul_mv_ext` | `mul_mv` (one column per threadgroup-y) |
| 4–8 | `mul_mv_ext` | `mul_mv_ext` |
| > 8 (and ne00 ≥ 64, simdgroup MM available) | `mul_mm` | `mul_mm` |

(`mul_mv_ext` also requires src1 F32 and `ne00 % 128 == 0`.)

### 2.2 `mul_mv` parameters per type (VERIFIED, ggml-metal-impl.h:26-100; ggml-metal-device.cpp:856-1096)

| type | block | bytes/block | bpw | NSG (simdgroups/TG) | NR0 | rows per TG | split-K across simdgroups? | threadgroup mem |
|---|---|---|---|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.5 | 2 | 4 per simdgroup | 8 | no | none |
| Q4_K | 256 | 144 | 4.5 | 2 | 2 per simdgroup | 4 | no | none |
| Q6_K | 256 | 210 | 6.5625 | 2 | 2 per simdgroup | 4 | no | none |
| Q8_0 | 32 | 34 | 8.5 | 4 | 2 per TG | 2 | **yes** (4 SGs stride over K) | 32·4·NR0 B |
| MXFP4 | 32 | 17 | 4.25 | 2 | 2 per simdgroup | 4 | no | 16-entry LUT (128 B) |

Threadgroup = NSG × 32 threads; grid.x = ceil(rows / (NR0·NSG)) (Q8_0/F16/F32: ceil(rows/NR0)),
grid.y = src1 columns, grid.z = batch (ggml-metal-ops.cpp:2637-2644).

### 2.3 Inner loops (VERIFIED)

* **Q4_0** (`mul_vec_q_n_f32_impl`, mul_mv.metal:218-304; dot helper :85-100): within a simdgroup,
  thread `t` handles half a block: `ix = t/2` selects one of 16 blocks in flight, `il = (t%2)*8`
  selects the half. Per iteration a simdgroup consumes 16 blocks = 512 weights of each of its 4 rows.
  The activation slice (16 floats) is loaded once into registers and **reused for all 4 rows**.
  Trick: activations are **pre-divided by 1, 256, 16, 4096** so a nibble can be used *in place* with a
  single mask on a `uint16` load (`qs & 0x0F00` etc.) — no shifts; the `-8` zero point is applied once per
  half-block as `sumy * -8` (`d * (sumy*-8 + Σacc)`). Loads: 4× `uint16` (8 B) + the fp16 `d` per
  half-block per row; activations read directly from device memory (no threadgroup staging). Reduction:
  one `simd_sum` per row, lane 0 writes (no threadgroup memory, no barriers).
* **Q4_K** (mul_mv.metal:1511-1616): 32 lanes = 4 groups of 8; each group of 8 handles one 256-weight
  super-block (stride 4 super-blocks); each lane does 32 weights (two 16-float register tiles `yl/yh`
  from 4 positions in the super-block); 6-bit scales/mins unpacked with 16-bit masks (`kmask1..3`); same
  "mask without shift, rescale by 1/256 and 1/16" trick; mins applied via per-sub-block Σy. 2 rows per
  simdgroup, `simd_sum` reduction.
* **Q6_K** (mul_mv.metal:1763-1860): pairs of lanes per super-block (stride 2), each lane 16 weights;
  6-bit values rebuilt from `ql` nibble + 2 bits of `qh`, `-32` offset, int8 sub-scales; 2 rows/SG.
* **Q8_0** (mul_mv.metal:513-585): `NQ = 8` int8 per lane, 4 lanes per block, 4 simdgroups split the K
  dimension (`ib0 = sgitg*NQ + ix`, stride `NSG*NQ` blocks); 2 rows per threadgroup; cross-simdgroup
  reduction via threadgroup memory + barriers (`helper_mv_reduce_and_write`, :169-209).
* **MXFP4** (mul_mv.metal:3034-3110): 16-entry E2M1 table copied to threadgroup memory, byte loads, 4
  table lookups per byte pair, E8M0 scale via `e8m0_to_fp32`; 2 lanes per block, 16 blocks in flight;
  2 rows/SG. Comment "[TAG_MUL_MV_WEIRD]": a redundant bound check makes it faster, cause unknown (:3075).
* **mul_mv_ext** (batch 2–8; mul_mv.metal:602-809, host ggml-metal-ops.cpp:2450-2551): NSG = 2 fixed (with
  a TODO admitting the choice is not understood, :2481-2489); `nxpsg` = 16/8/4 lanes along K per row
  (so 2/4/8 rows per simdgroup), each thread dequantizes 4 chunks of float4 (`deq_t4`) — or one float4x4
  for K-quants — and dots them with `r1ptg` activation columns; reduction via `simd_shuffle_down`.
  `r1ptg` by batch: 2→2, 3→3, 4→4, 5→5, 6→3, 7→4, 8→4 (:2506-2520).
  **Consequence (VERIFIED by reading the dispatch, :2551):** grid.y = ceil(ne11/r1ptg), and each
  threadgroup re-streams its weight rows, so weights are read **once for batch 2–5 and twice for batch
  6–8** (unless they survive in the SLC; EXPERIMENTAL). This matters for speculative verification
  (n_draft+1 = 4–8 tokens). See measurement E.

### 2.4 How decode GEMV was tuned
See §9 (PR history from the gh sub-search). In-code evidence: per-type NR0/NSG constants are hand-picked
(ggml-metal-impl.h:26-100), the ext-path parameters are hand-picked with "not sure how optimal are those
across all different hardware" (ggml-metal-ops.cpp:2505). The only *offline autotuned* table in the
Metal backend is for the **FA vec kernel** (§4), keyed by Apple GPU family, generated by
`tools/tuning` (`ggml-metal-tuning fa-vec`, tools/tuning/README.md; ggml-metal-tuning.cpp:69-75 cites
PR #27824 and discussion #27668). No runtime autotuning anywhere (VERIFIED by reading; LIKELY complete).

---------------------------------------------------------------------------------------------------

## 3. Prefill kernels (`kernels/mul_mm.metal`) and the M5 tensor path

* **simdgroup_matrix path (all pre-M5 GPUs; VERIFIED mul_mm.metal:146-359):** threadgroup tile
  **64 (src0 rows) × 32 (src1 columns)**, K step 32, 4 simdgroups (128 threads). Each thread dequantizes
  16 weights per K-step (`dequantize_func` → `half4x4`) into threadgroup memory (`sa`, 4 KiB) and converts
  8 activations to half (`sb`); each simdgroup then does 4 (A) × 2 (B) `simdgroup_load` of 8×8 tiles and
  8 `simdgroup_multiply_accumulate` per 8-deep K slice into **8 fp32 accumulators** (32×16 sub-tile).
  So prefill is **f16×f16→f32 on the simdgroup MMA units, with weights dequantized per tile**; the
  activations are rounded to half (bf16 variants exist). Output written straight from registers when the
  tile is full, via threadgroup staging otherwise. Comment at :248: "NOTE: this is massively slower..
  WTF?" (an indexing form that the compiler handles badly) — evidence of empirical, hand-tuned code.
* **Metal 4 tensor API path (VERIFIED, mul_mm.metal:13-142; common.h:7-10):** under
  `GGML_METAL_HAS_TENSOR`, `mul_mm` uses `mpp::tensor_ops::matmul2d` (MetalPerformancePrimitives) with
  cooperative tensors: tile NRA=64 × NRB=128 (`SZ_SIMDGROUP 16 × N_MM_BLOCK_{X=4,Y=2} ×
  N_MM_SIMD_GROUP_{X,Y}=2`, ggml-metal-impl.h:8-15), K step 32; A is still dequantized into threadgroup
  memory; **B (activations) is consumed straight from device memory** as a strided tensor; K is a dynamic
  extent (PR #27064 comment). MoE `mul_mm_id` has a matching tensor variant (:589-614).
* **When enabled (VERIFIED ggml-metal-device.m:1145-1162):** `has_tensor = supportsFamily(Metal4)` but
  then **forced off unless the GPU name contains "M5", "M6", "A19" or "A20"** (override
  `GGML_METAL_TENSOR_ENABLE`), with the in-code justification:
  "M2 Ultra: ~5% slower; M4, M4 Max: no significant difference. TODO: try to update the tensor API kernels
  to at least match the simdgroup performance". A tiny dummy kernel is compiled at init to verify f16 and
  bf16 tensor support (:1164-1260). Our M2 Pro: "tensor API disabled for pre-M5 and pre-A19 devices".
  (The M5 GPU "neural accelerators" are reached through this tensor/MPP path; LIKELY — Apple documents
  the tensor APIs as the way to use them; see sub-search §9 for Apple's statement.)
* No MPS / MPSGraph / Core ML usage anywhere in ggml-metal (VERIFIED grep). Everything is hand-written MSL.
* bfloat: `has_bfloat = supportsFamily(Metal3) || supportsFamily(Apple6)` (ggml-metal-device.m:1139-1143);
  BF16 kernels are compiled under `GGML_METAL_HAS_BF16`.

---------------------------------------------------------------------------------------------------

## 4. Attention: `flash_attn_ext` and the vec (decode) variant

* **Selection (VERIFIED, ggml-metal-ops.cpp:2949-2957):** vec kernel iff `ne01 < 20` (query rows) and
  head size % 32 == 0. Otherwise the tiled kernel.
* **Tiled (prefill) kernel:** 8 queries per threadgroup (`OP_FLASH_ATTN_EXT_NQPSG 8`), 64 KV per
  simdgroup step (`NCPSG 64`) (ggml-metal-impl.h:126-127); a pre-pass `blk` kernel classifies mask blocks
  so fully masked blocks are skipped; a `pad` kernel handles KV not multiple of 64
  (ggml-metal-ops.cpp:3419-3573). For batch ≥ 32 with a quantized KV cache, K/V are first **dequantized to
  an F16 scratch** and the F16 kernel is used (PR #27390, :2959-2984).
* **Vec (decode) kernel** (`kernels/fa_vec_common.metal`; host ggml-metal-ops.cpp:3575-3830):
  - 1 query per threadgroup by default (`VEC_NQPSG 1`), 32 KV positions per simdgroup step
    (`VEC_NCPSG 32`); `NE` head elements per lane (NL = 32/NE lanes per row).
  - **Split-K ("flash decoding"):** always `nwg = 32` workgroups along the KV axis (non-sparse case);
    `nsg` grows 1→2→4 while `2*nwg*nsg*32 < n_kv` (:3686-3717); partial (O, m, l) go to a temp buffer and
    a second `vec_reduce` kernel combines the 32 partials, separated by a barrier (:3800-3825).
  - (Q, NE) per (GPU family, KV dtype, dk, dv, KV-length bucket {<1k, <4k, <16k, ≥16k}, query-rows
    bucket) come from the **offline-tuned table** (ggml-metal-tuning.cpp; families Apple7 = M1,
    Apple8 = M2, Apple9 = A18 Pro/M3/M4, **Apple10 = M5 Max**, 366 rows for Apple10). E.g. Qwen3
    (dk=dv=128, F16 KV) decode on M5 Max: default (Q=1, NE=4); KV ≥ 16k: (Q=2, NE=2)
    (ggml-metal-tuning.cpp:747+). Short KV (< 1024) always uses the baseline (:fa_vec_pick).
* **KV types (VERIFIED, kernel files):** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (one `fa_*` and
  `fa_vec_*` library per type; K/V dequantized on the fly by templated `deq_k_t4`/`deq_v_t4`). Also a
  "sparse" path driven by an `n_kv_max` hint + mask (for DSA-style sparse attention), sinks, ALiBi bias,
  logit softcap, MLA (V a view of K).
* **When FA is used (VERIFIED):** `-fa auto` is the default (`LLAMA_FLASH_ATTN_TYPE_AUTO`,
  src/llama-context.cpp:3769); at context creation it is resolved by probing whether the backend
  supports the FA op for this model (`resolve(llm_fused_op_flash_attn_probe, ...)`,
  src/llama-context.cpp:557-560). On Metal with standard head sizes ⇒ **on**.

---------------------------------------------------------------------------------------------------

## 5. GGUF format and block layouts

* **File layout (VERIFIED ggml/include/gguf.h:1-30, 41-46):** magic `"GGUF"`, `uint32 version` (=3),
  `int64 n_tensors`, `int64 n_kv`; KV pairs (string key, `gguf_type`, value or typed array); per-tensor
  info: name, `uint32 n_dims`, `int64 ne[n_dims]`, `ggml_type` (int32), `uint64 offset` into the data
  blob; then padding to `general.alignment` (default **32**, must be a power of 2,
  ggml/src/gguf.cpp:622-635, 781); data blob with each tensor at an aligned offset. Strings are
  `uint64 len + bytes`. Our files: GGUF v3, no explicit `general.alignment` ⇒ 32 B (B-bytes.txt).
* **Blocks (VERIFIED ggml/src/ggml-common.h):**

| type | weights/block | layout | bytes | bits/weight | dequant |
|---|---|---|---|---|---|
| Q4_0 (:194-198) | 32 | fp16 `d`; 16 B nibbles (`qs[j]` low nibble = w[j], high = w[j+16]) | 18 | 4.5 | `d·(q−8)` |
| Q8_0 (:251-255) | 32 | fp16 `d`; 32× int8 | 34 | 8.5 | `d·q` |
| MXFP4 (:214-218) | 32 | uint8 E8M0 exponent `e`; 16 B of E2M1 nibbles | 17 | 4.25 | `2^(e−127)·E2M1[q]` (table `kvalues_mxfp4`) |
| Q4_K (:327-337) | 256 | fp16 `d`, fp16 `dmin`; 12 B of 6-bit scales+mins for 8 sub-blocks of 32; 128 B nibbles | 144 | 4.5 | `d·sc_i·q − dmin·m_i` |
| Q6_K (:362-367) | 256 | 128 B low nibbles `ql`; 64 B high 2-bit `qh`; 16× int8 sub-scales (sub-blocks of 16); fp16 `d` | 210 | 6.5625 | `d·sc_i·(q−32)` |

  (Also present: Q4_1 5.0 bpw, Q5_0 5.5, Q5_1 6.0, Q5_K 5.5, Q2_K/Q3_K, IQ-family, TQ, Q1_0/Q2_0,
  **NVFP4** (64 weights, 4× UE4M3 sub-scales + 32 B = 36 B = 4.5 bpw, :221-226).)
* Practical note (VERIFIED B-bytes.txt): "Q4_0" and "Q4_K_M" files are mixtures — llama-quantize keeps
  `output.weight` at Q6_K and promotes some `ffn_down`/`attn_v` (Q4_K_M: `attn_v` and `ffn_down` → Q6_K;
  Q4_0: 4 `ffn_down` → Q4_1). Effective weight bytes/token for Qwen3-8B: Q4_0 4.431 GB, Q4_K_M 4.672 GB.

---------------------------------------------------------------------------------------------------

## 6. Per-token CPU overhead, sync points, sampling

* **Graph reuse (VERIFIED).** `llama_context::process_ubatch` (src/llama-context.cpp:1397-1470) keeps
  the previous `llm_graph_result` and, if `res->can_reuse(gparams)` holds, **skips graph construction,
  `ggml_backend_sched_reset` and `ggml_backend_sched_alloc_graph`** — only `set_inputs` (token ids,
  positions, KQ mask, KV write indices) and `graph_compute` run. `can_reuse` asks every input object
  (src/llama-graph.cpp:1411-1441); for attention this requires the same ubatch size and the same KQ-mask
  shape (`can_reuse_kq_mask`, :48-65). The KV view length `n_kv` is **padded to 256**
  ("pad the n_kv value so that the graph remains constant across batches and can be reused",
  src/llama-kv-cache.cpp:1260-1273), so a decode graph is rebuilt about once per 256 tokens.
  `LLAMA_GRAPH_REUSE_DISABLE=1` turns it off (:281-286). Measured: baseline `llama-completion` reports
  `graphs reused = 253` for 255 decode calls (C0 log).
  - Ablation (measurement F, VERIFIED): disabling reuse costs **−42 % tg on Qwen3-0.6B (250 → 144 t/s,
    i.e. ≈2.9 ms extra CPU per token)** and −5 % on 8B (35.95 → 34.18 t/s, ≈1.4 ms/token). Graph
    build + allocation is therefore a ms-scale cost; reuse is llama.cpp's main defence against it.
* **Scheduler (VERIFIED).** `ggml_backend_sched_compute_splits` (ggml/src/ggml-backend.cpp:1662+) walks
  splits, copies split inputs (with events/synchronize between backends) and calls each backend's
  `graph_compute`. With full offload the Metal split dominates; the reorder/optimize step runs at split
  time only (ggml-backend.cpp:1463-1478). (Exact split count for Qwen3: see measurement D.)
* **Re-encoding every token (VERIFIED).** Even when the ggml graph is reused, the Metal backend
  re-encodes every node into fresh command buffers each token (ggml-metal-context.m:474-660): pipeline
  lookup by name, hazard check, `setComputePipelineState`/`setBytes`/`setBuffer`/`dispatchThreadgroups`
  per node. No ICB replay, no MTL4 argument tables. Mitigations: 2-thread encoding, early commit of the
  first ≈10 % of nodes, unretained references.
* **Sync points per token (VERIFIED).**
  1. `llama_decode` → `graph_compute` is asynchronous; logits are queued with
     `ggml_backend_tensor_get_async` (src/llama-context.cpp:1563): a blit into a NoCopy-wrapped host
     array (ggml-metal-context.m:372-416) — for Qwen3 that is 151,936 × 4 B ≈ **608 KB per token**.
  2. The sampler then calls `llama_get_logits_ith` or `llama_get_sampled_token_ith`, both of which
     call `ctx->synchronize()` → `waitUntilCompleted` (src/llama-context.cpp:4001-4006, 4075-4079;
     common/sampling.cpp:153, 612).
  3. The next `llama_decode` receives the sampled token id **from the host**. So there is one
     GPU→CPU→GPU round trip per token, and the GPU idles while the CPU samples, writes inputs and
     encodes the first command buffer. (EXPERIMENTAL: bubble size not measured; estimate 0.1–0.5 ms.)
     MLX/mlx-lm avoids this structurally: sampling is part of the lazy GPU graph and `generate_step`
     builds step N+1 and calls `mx.async_eval` *before* blocking on token N's `.item()`
     ($S/src/mlx-lm/mlx_lm/generate.py:457-466) (VERIFIED in that clone).
* **Sampling (VERIFIED).** Default is CPU sampling over the full logits vector (common/sampling.cpp).
  Since 2025/26 there is **backend (GPU) sampling**: samplers that implement `backend_apply` are appended
  to the compute graph — greedy, dist, top-k, top-p, min-p, temp, temp-ext, penalties, logit-bias
  (src/llama-sampler.cpp:1101, 1393, 1513, 1713, 1876, 2098, 2301, 3197, 4037; wiring in
  src/llama-context.cpp:1282-1330). It is **off by default** for normal generation
  (`backend_sampling = false`, common/common.h:298; flag `-bs/--backend-sampling`, common/arg.cpp:2306)
  but **on by default for speculative drafting** (common/common.h:334). Even with backend sampling the
  token still goes back to the host before the next `llama_decode` (LIKELY, from the API shape: the
  batch is a host `llama_batch`; I found no device-side token feedback path).
* **Residency heartbeat** costs a background thread waking every 5 ms (§1.4) — negligible CPU but
  worth knowing for energy measurements (EXPERIMENTAL).

---------------------------------------------------------------------------------------------------

## 7. Speculative decoding

* **Types at this commit (VERIFIED common/common.h:173-186):** `draft-simple` (separate draft model),
  `draft-eagle3`, `draft-mtp` (multi-token-prediction heads, incl. shared KV), `draft-dflash`,
  `draft-dspark` (DFlash + Markov head), and self-speculative `ngram-simple`, `ngram-map-k`,
  `ngram-map-k4v`, `ngram-mod`, `ngram-cache` (3-level n-gram cache, the old `llama-lookup`).
  Defaults: `n_max = 3` draft tokens, `n_min = 0`, `p_min = 0.0`, draft sampling offloaded to the
  backend (common/common.h:327-334). `ngram-simple` needs a `size_n = 12`-token match
  (common/common.h:362-363).
* **Where:** `examples/speculative` (tree-based, multiple drafts; README cites PRs #2926, #3624, #5625),
  `examples/speculative-simple` (all `--spec-type`s; README shows draft-mtp with
  `ggml-org/Qwen3.6-27B-GGUF`, draft-dflash with `ggml-org/Qwen3-8B-GGUF`), `examples/lookup`
  (prompt-lookup decoding, PR #4484), and the **server** (`common_speculative_init_from_params`,
  `common_speculative_draft`, tools/server/server-context.cpp:1121, 1261, 3044) — so `llama-server -md`
  or `--spec-type ...` does speculative decoding in production.
* Verification batches of n_draft+1 = 2–8 tokens go through `mul_mv_ext` (§2.3) — weights read once
  for 2–5 columns, twice for 6–8; K-quants with 2–3 columns use plain `mul_mv` with one grid row per
  column (weights read 2–3×). Measurement G quantifies the resulting verification cost.
* Our measurement (C) and published numbers: §8 and §9.

---------------------------------------------------------------------------------------------------

## 8. Measurements on this machine (all timing runs via `bash $S/benchlock.sh ...`)

Raw files: `$S/research/R5-raw/`. Drivers: `$S/research/R5-bench{A,C,D,E,F,G}.sh`.
Caveat for every number: other agents ran CPU-heavy work concurrently (load average 3–7 during runs;
the GPU itself was exclusive under the lock). Repeat runs of the same config differed by up to ~3 %
(8B tg128: 34.83 in A3 vs 35.95 in F-BASE).

### A. llama-bench, Qwen3 0.6B / 8B Q4_0, pp128/512/2048 + tg128, FA off/on, r=5 (VERIFIED)

Commands (exact; from the `# cmd:` lines in A*.md):
```
B=$S/src/llama.cpp/build/bin
$B/llama-bench -m $S/models/Qwen3-0.6B-Q4_0.gguf -p 128,512,2048 -n 128 -fa off,on -r 5 -o md -oe json
$B/llama-bench -m $S/models/Qwen3-8B-Q4_0.gguf  -p 128,512,2048 -n 128 -fa off    -r 5 -o md -oe json
$B/llama-bench -m $S/models/Qwen3-8B-Q4_0.gguf  -p 128,512,2048 -n 128 -fa on     -r 5 -o md -oe json
```
Defaults in effect (JSON): n_batch 2048, n_ubatch 512, type_k/v f16, load_mode auto (mmap), -ngl -1 (all).
Raw output:
```
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 0 |  pp128 |  3510.98 ±  42.72 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 0 |  pp512 |  3473.97 ± 140.76 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 0 | pp2048 |  2978.70 ±   2.78 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 0 |  tg128 |   222.20 ±   4.63 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 1 |  pp128 |  3776.59 ±  23.84 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 1 |  pp512 |  3849.31 ± 150.05 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 1 | pp2048 |  3116.88 ±   4.21 |
| qwen3 0.6B Q4_0 | 358.78 MiB | 596.05 M | MTL,BLAS | 6 | 1 |  tg128 |   247.54 ±   2.11 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 0 |  pp128 |   306.67 ±   0.78 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 0 |  pp512 |   307.16 ±   1.50 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 0 | pp2048 |   290.36 ±   1.08 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 0 |  tg128 |    33.43 ±   0.36 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 1 |  pp128 |   310.51 ±   0.84 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 1 |  pp512 |   312.49 ±   1.37 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 1 | pp2048 |   294.11 ±   2.40 |
| qwen3 8B Q4_0   |   4.45 GiB |   8.19 B | MTL,BLAS | 6 | 1 |  tg128 |    34.83 ±   0.94 |
build: 81e39ad (1)
```
Per-rep samples (JSON, t/s): 8B tg128 FA off [33.1, 33.0, 33.6, 33.6, 33.8], FA on
[35.8, 35.4, 34.5, 35.1, 33.4]; 0.6B tg128 FA off [219.4, 224.4, 227.6, 223.8, 215.8], FA on
[248.5, 247.9, 250.2, 244.5, 246.7].
Observations: FA on is never slower here (+1–2 % pp and +4 % tg on 8B; +8–11 % on 0.6B); pp drops
5–6 % from pp512 to pp2048 (attention cost growing with context).

### B. Bandwidth efficiency of tg128 (VERIFIED arithmetic on VERIFIED inputs)

Bytes/token from GGUF tensor info via gguf-py (`$S/research/R5-bytes.py`, output
`R5-raw/B-bytes.txt`): sum of all tensors' `n_bytes`, minus `token_embd` when `output.weight`
exists (then only one embedding row is read per token), plus nothing else (KV reads at tg128 are
≤ 0.2 % for 8B, ≈2 % for 0.6B: 36 layers × 8 KV heads × 128 × 2 (K,V) × 2 B = 147 KB per cached
token, avg 64 cached tokens ⇒ 9.4 MB; 0.6B: 115 KB/token ⇒ 7.3 MB).

| model | output.weight? | weight bytes/token | tg128 t/s | GB/s | % of 200 GB/s spec | % of ≈183 GB/s achievable (R3) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B Q4_0 | no (tied, Q6_K emb is the head) | 376.21 MB | 222.20 (FA off) | 83.6 | 41.8 % | 45.7 % |
| " | " | " | 247.54 (FA on) | 93.1 | 46.6 % | 50.9 % |
| Qwen3-8B Q4_0 | yes (Q6_K, 510.5 MB) | 4,431.32 MB | 33.43 (FA off) | 148.1 | 74.1 % | 80.9 % |
| " | " | " | 34.83 (FA on) | 154.3 | 77.2 % | 84.3 % |
| " | " | " | 35.95 (F, FA on) | 159.3 | 79.7 % | 87.0 % |
| Qwen3-8B Q4_K_M (for reference) | yes | 4,671.77 MB | (see G) | | | |

Two-point model (EXPERIMENTAL): fitting `t = bytes/BW + c·n_layers` to (376 MB, 28 layers, 4.04 ms)
and (4431 MB, 36 layers, 28.71 ms) gives **BW ≈ 168 GB/s and c ≈ 64 µs per layer** — i.e. ≈1.8 ms
(0.6B) / 2.3 ms (8B) per token not explained by weight streaming.

### C. Speculative decoding, Qwen3-8B Q4_0 target + Qwen3-0.6B Q4_0 draft (VERIFIED)

Prompt (README example): `Here is a quick sort implementation in C++. Just code, no comments:\n\n#include`;
`-n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -c 4096` (+ `-md ... -ngld 99` for the draft). Raw logs
`R5-raw/C*.log`.

| run | command (tool / spec flags) | decode t/s | drafted | accepted | acceptance | speedup vs C0 |
|---|---|---|---|---|---|---|
| C0 baseline | `llama-completion ... -no-cnv` | **34.58** (eval 255 runs, 28.92 ms/token); `graphs reused = 253` | – | – | – | 1.00× |
| C1 | `llama-speculative-simple --spec-type draft-simple --spec-draft-n-max 7` | 32.72 (258 tok / 7.885 s) | 329 | 211 | 64.1 % | **0.95×** |
| C2 | `... --spec-draft-n-max 3` (= default n_max) | **38.78** (258 tok / 6.653 s) | 228 | 182 | 79.8 % | **1.12×** |
| C3 | `--spec-type ngram-simple` (no draft model) | 33.31 (257 / 7.715 s) | 0 | 0 | n/a | 0.96× |

Why so small (EXPERIMENTAL analysis from the C2 log): the target's batched verification calls are
logged as "prompt eval": 5,348.9 ms for 321 tokens over ≈76 calls ⇒ **≈70 ms per verification call of
≈4 tokens, 2.4× a 1-token decode (28.9 ms)**. The draft costs ≈4 ms/token. With 3.4 tokens gained per
cycle, the cycle costs ≈87 ms ⇒ 25.8 ms/token vs 28.9 ms. Kernel-level cause: measurement E (multi-column
GEMV is far from free). Also `graphs reused` drops to 46–75 because verification batch sizes vary
(graph rebuilds). C3: ngram-simple needs a 12-token exact match; fresh code generation had none.

### D. Graph size / splits / memory (VERIFIED, `R5-raw/D-graphdebug-Qwen3-8B-Q4_0.log`)

`sched_reserve: graph: nodes = 1266, splits = 2, input objects = 4, input tensors = 6`;
`sched copies = 1`; `MTL0 compute buffer size = 304.75 MiB`, `CPU compute buffer size = 20.01 MiB`;
`CPU_Mapped model buffer size = 333.84 MiB` (token_embd stays on the CPU: the input layer is not
offloaded, hence 2 splits) and `MTL0_Mapped model buffer size = 4559.88 MiB`; KV (4096 ctx, f16)
576 MiB. Memory breakdown at exit: MTL0 self 5440 MiB = 4559 model + 576 context + 304 compute; Host
353 MiB. Load (cold-ish page cache) 3,626 ms; warm 422 ms (C1/C2). Default sampler chain on CPU:
`penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist`.
(Per-node dispatch/barrier counts: D2 below.)

### E. Kernel-level GEMV/GEMM throughput (VERIFIED; `test-backend-ops perf`)

Command: `test-backend-ops perf -b MTL0 -o MUL_MAT -p 'type_a=(q4_0|q4_K|q8_0|mxfp4|q6_K|f16),type_b=f32,m=4096,n=(1|2|3|4|5|8|512),k=14336'`
(raw `R5-raw/E-mulmat-perf.log`, summary `R5-raw/E-summary.txt`). Matrix 4096×14336 (an 8B FFN
shape); "GB/s" = weight bytes / kernel time; "×T1" = time relative to n=1.

```
type        MB | n=1   us   GB/s  xT1 | n=2   us   GB/s  xT1 | n=3   us   GB/s  xT1 | n=4   us   GB/s  xT1 | n=5   us   GB/s  xT1 | n=8   us   GB/s  xT1 | n=512 us
f16     117.44 |    686.3  171.1 1.00 |    666.1  176.3 0.97 |    670.5  175.2 0.98 |    686.0  171.2 1.00 |    798.5  147.1 1.16 |    834.1  140.8 1.22 |  12317.7 (4.88 TFLOPS)
q4_0     33.03 |    186.2  177.4 1.00 |    313.4  105.4 1.68 |    343.9   96.1 1.85 |    472.2   69.9 2.54 |    589.9   56.0 3.17 |    836.9   39.5 4.49 |  14059.1 (4.28 TFLOPS)
q8_0     62.39 |    354.2  176.2 1.00 |    381.9  163.4 1.08 |    425.8  146.5 1.20 |    514.1  121.4 1.45 |    701.7   88.9 1.98 |    901.6   69.2 2.55 |  14024.2 (4.29 TFLOPS)
mxfp4    31.20 |    226.6  137.7 1.00 |    376.7   82.8 1.66 |    481.1   64.8 2.12 |    620.5   50.3 2.74 |    653.6   47.7 2.88 |    996.0   31.3 4.40 |  14271.9 (4.21 TFLOPS)
q4_K     33.03 |    224.2  147.3 1.00 |    419.1   78.8 1.87 |    625.2   52.8 2.79 |    701.5   47.1 3.13 |    783.4   42.2 3.49 |   1163.5   28.4 5.19 |  15898.3 (3.78 TFLOPS)
q6_K     48.17 |    279.7  172.2 1.00 |    458.4  105.1 1.64 |    683.8   70.4 2.44 |    609.3   79.1 2.18 |    785.0   61.4 2.81 |   1197.3   40.2 4.28 |  16189.6 (3.71 TFLOPS)
```
Findings:
* **Batch-1 Q4_0 GEMV runs at 177 GB/s ≈ 97 % of the achievable ≈183 GB/s** (89 % of spec). Q8_0/Q6_K/
  F16 also ≥171 GB/s. **Q4_K only 147 GB/s (80 %) and MXFP4 138 GB/s (75 %)** — the K-quant and FP4
  kernels are measurably less efficient than Q4_0. (Caveat, LIKELY ≤5 %: a 31–33 MB matrix may get
  some reuse from the system-level cache across repeated runs; F16 at 117 MB is unaffected.)
* **Small batches are not free.** For 4-bit formats a 4-column call costs 2.5–3.1× a 1-column call and
  8 columns 4.4–5.2× — the multi-column kernels are issue/ALU-bound, not bandwidth-bound (F16, whose
  weight bytes dominate, stays flat up to n=4). K-quants at n=2–3 scale linearly (separate column
  passes, as read in §2.3). This is the kernel-level reason speculative decoding barely pays (C).
* Prefill GEMM (n=512) reaches 3.7–4.9 TFLOPS on the 16-core M2 Pro GPU (simdgroup-matrix path).
  The end-to-end 8B pp512 (312 t/s ≈ 4.7 TFLOPS of matmul work) is consistent with that.
* End-to-end decode (154–159 GB/s) vs the kernel's 177 GB/s ⇒ ≈10–13 % of decode time is outside
  the big GEMVs (attention, norms, ropes, small ops, launch gaps, host round-trip).

### F. Feature ablation on decode (VERIFIED; `R5-raw/F-ablation-*.md`)

Command per config: `env <VAR>=1 llama-bench -m <model> -p 0 -n 128 -fa on -r <5|3> -o md`.

| config | 0.6B tg128 (r=5) | Δ | 8B tg128 (r=3) | Δ |
|---|---|---|---|---|
| baseline | 250.22 ± 2.38 | – | 35.95 ± 0.34 | – |
| `GGML_METAL_FUSION_DISABLE=1` | 248.85 ± 1.37 | −0.5 % | 34.32 ± 0.82 | −4.5 % |
| `GGML_METAL_CONCURRENCY_DISABLE=1` | 239.60 ± 6.40 | −4.2 % | 35.13 ± 0.09 | −2.3 % |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | 252.78 ± 2.25 | +1.0 % (noise) | 35.79 ± 0.14 | −0.4 % |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | **144.09 ± 17.78** | **−42 %** | 34.18 ± 0.15 | −4.9 % |

Reading: graph reuse is the big CPU-side lever for small models (≈2.9 ms/token of graph build+alloc);
fusion and concurrency are worth a few % each on 8B; the reorder pass is ~0 for Qwen3 decode.

### D2. What one decode token costs in GPU work items (VERIFIED; `GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_FUSION_DEBUG=2 llama-completion ... -n 4 -lv 5`, parsed by `$S/research/R5-parseD.py`)

With fusion stats on, the backend forces single-threaded encoding (n_cb = 0), so each graph is one
pass. Every decode graph of Qwen3-8B (identical across the 5 passes):

| | Qwen3-8B (36 layers) | Qwen3-0.6B (28 layers) |
|---|---|---|
| ggml graph nodes (incl. views) | 1266 | 986 |
| encoded nodes (non-empty, after fusion) | **688** | 536 |
| — MUL_MAT | 253 (7/layer + LM head) | 197 |
| — RMS_NORM (each fused with its MUL) | 145 (4/layer + final) | 113 |
| — ROPE / SET_ROWS (KV write) / ADD | 72 / 72 / 72 | 56 / 56 / 56 |
| — FLASH_ATTN_EXT (vec: 2 dispatches each, vec + reduce) | 36 | 28 |
| — GLU (SwiGLU) / GET_ROWS | 36 / 2 | 28 / 2 |
| Metal dispatches per token | **≈724** (688 + 36 FA reduce) | ≈564 |
| nodes encoded after a memory barrier | **506** (+36 FA-internal ⇒ ≈542 barriers) | 366 (+28) |
| nodes that ran concurrently with predecessors | 182 | 170 |
| fusions fired | RMS_NORM+MUL ×145 per graph (725 over 5 graphs) | ×113 |

So a Qwen3 layer is **19 nodes / ≈20 dispatches / ≈15 barriers** in llama.cpp. Combined with the
two-point fit in B (≈64 µs/layer not explained by weight streaming), that is ≈3 µs of overhead per
dispatch+barrier on this GPU (EXPERIMENTAL inference). A hand-fused layer (QKV-GEMV + q/k-norm + RoPE +
KV store; FA; O-GEMV + residual; norm + gate/up-GEMV + SwiGLU; down-GEMV + residual) is 5–6 dispatches.

### G. Verification-cost curve: one `llama_decode` call with n tokens (VERIFIED; `R5-raw/G-smallbatch.md`)

Command: `llama-bench -m <model> -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md` (each "pp n" is
exactly one decode call with n tokens). Time per call = n / (t/s):

```
Qwen3-8B-Q4_0    n=1: 27.5 ms (x1.00)  n=2: 42.0 (x1.52)  n=3: 45.8 (x1.66)  n=4: 57.4 (x2.08)  n=5: 67.0 (x2.43)  n=6: 81.5 (x2.96)  n=7: 106.8 (x3.88)  n=8: 103.4 (x3.76)  n=16: 118.0 (x4.29)
Qwen3-8B-Q4_K_M  n=1: 31.6 ms (x1.00)  n=2: 53.1 (x1.68)  n=3: 76.8 (x2.43)  n=4: 78.5 (x2.48)  n=5: 94.6 (x2.99)  n=6: 113.4 (x3.59)  n=7: 134.9 (x4.27)  n=8: 140.4 (x4.44)  n=16: 136.3 (x4.31)
Qwen3-0.6B-Q4_0  n=1:  4.3 ms (x1.00)  n=2:  5.2 (x1.22)  n=3:  5.6 (x1.31)  n=4:  6.6 (x1.53)  n=5:  7.3 (x1.70)  n=6:  8.6 (x2.01)  n=7:  10.4 (x2.42)  n=8:  10.7 (x2.49)  n=16:  14.4 (x3.36)
```
Raw t/s (8B Q4_0): pp1 36.32±0.67, pp2 47.65±0.10, pp3 65.47±0.44, pp4 69.71±0.16, pp5 74.67±0.20,
pp6 73.66±2.58, pp7 65.53±3.12, pp8 77.34±0.38, pp16 135.54±0.40. (Q4_K_M: pp1 31.63±0.25 … pp16
117.36±0.31; 0.6B: pp1 233.30±23.48 … pp16 1112.58±3.26.)

Findings:
* **Verifying 4 tokens costs 2.1× (Q4_0) / 2.5× (Q4_K_M) a single-token decode; 8 tokens 3.8× / 4.4×.**
  An ideal memory-bound verifier would be ≈1.0–1.2× up to n≈8. pp16 (GEMM path) costs barely more
  than pp8 (GEMV-ext path) ⇒ the n = 2–8 kernels are the weak spot (cf. E).
* Q4_K_M decodes 13 % slower than Q4_0 at n=1 (31.6 vs 27.5 ms) although it reads only 5 % more bytes —
  consistent with E (Q4_K GEMV 147 GB/s vs Q4_0 177 GB/s).
* Simple model of draft-model speculation (EXPERIMENTAL): with k = 3 drafts, acceptance α = 0.8 ⇒
  expected tokens/cycle (1 − α^4)/(1 − α) = 2.95; cycle cost = 2.08 (verify 4) + 3 × 4.3/27.5 (draft)
  = 2.55 single-decode units ⇒ predicted **1.16×**, measured **1.12×** (C2). With a verifier costing
  1.0× the same acceptance gives ≈2.0×. **≈1.7× of available speculative speedup is lost to
  small-batch kernel inefficiency on this machine.**
