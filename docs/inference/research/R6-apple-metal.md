# R6: Apple Silicon and Metal 4 for an LLM inference runtime (Sky)

**Written:** 2026-10-02, research agent R6.

**Scope:**
- **Part A** sets out hardware and API facts with sources.
- **Part B** reports local Metal microbenchmarks on an M2 Pro.
- **The hardware model** is the set of numbers the roofline and cost model should use.
- **The architect brief** comes last.

**Headline workload:** Qwen3-32B in 4-bit, batch 1, on an M5 Max.

## Evidence labels

| Label | Meaning |
|---|---|
| **[V]** VERIFIED | Backed by an official Apple URL, WWDC/tech-talk transcript, spec section and page, source `path:line`, or our own measurement. Our measurements name the raw file under `$S/research/microbench/results/`. |
| **[L]** LIKELY | Reasoning is given. Independent third-party measurements from a single source carry this label unless two independent sources agree. |
| **[E]** EXPERIMENTAL | Plausible, but untested or measured only once. |
| **[M]** MOONSHOT | Speculative. |

**Path shorthand:**
- `$S` = `/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad`

**Local primary sources saved for re-checking:**
- Text of the MSL 4.1 spec (dated 2026-06-04): `$S/research/tools/MSL-spec.txt`
- Apple doc pages: `$S/research/R6-work/docs/`, rendered with `$S/research/tools/appledoc.py`, which reads the developer.apple.com JSON endpoints
- WWDC26 transcripts: `$S/research/R6-work/docs/wwdc26_*.txt`
- Apple's metal-cpp, which lets us diff the 2026 APIs: `$S/research/R6-work/metal-cpp` (tags macOS26, 26.4 and 27)
- Helper-agent page dumps: `$S/research/R6-work/chips/` and `$S/research/R6-work/neural/`

---

## 0. Key findings

1. **M5 Max memory bandwidth depends on the GPU bin.**
   - 614 GB/s is only for the 40-core GPU part [V]. That part ships with 48, 64 or 128 GB.
   - The 32-core M5 Max is 460 GB/s and is sold only with 36 GB [V].
   - Source: apple.com/macbook-pro/specs and the Mac Studio specs: "M5 Max with 18-core CPU and 32-core GPU (460GB/s memory bandwidth) or … 40-core GPU (614GB/s memory bandwidth)".
   - **Decode ceiling for Qwen3-32B 4-bit** (≈18 GB of weights per token) is ≈34 tok/s on a 40-core M5 Max [L, arithmetic].
2. **The GPU Neural Accelerators are reachable only through Metal 4 TensorOps (MPP) and the frameworks built on them.**
   - Apple's own tech talk shows plain `simdgroup_matrix` code at "neural accelerator utilization is 0%" [V].
   - Measured M5 Max ceilings: ≈64–65.5 TFLOPS FP16 and ≈122 TOPS INT8 [L]. Two independent sources agree on the per-core rate.
   - They accelerate prefill (≈3.3–4×), not batch-1 decode [V].
3. **On this M2 Pro, Metal 4 tensors and MPP compile and run correctly.**
   - MPP `matmul2d` compiles under MSL 4.0 with the runtime compiler, for half, bfloat and **int4b_format** [V, measured].
   - It reaches **5.42 TFLOPS FP16**, 95% of the 5.73 TFLOPS ALU peak. That beats `MPSMatrixMultiplication` (4.41) [V, measured].
   - MSL 4.1 (fp8/fp4/int2, block scales, acquire/release) needs OS 27 and is rejected by this macOS 26.6.1 [V, measured].
4. **Measured M2 Pro numbers:**
   - Best read bandwidth: **196 GB/s**, 98% of the 200 GB/s spec.
   - A simple MLX-style 4-bit GEMV streams at **183–191 GB/s**.
   - Shared and private storage show no bandwidth difference [V, measured].
5. **Dispatch cost on M2 Pro:**
   - A **dependent dispatch costs ≈1.1–1.25 µs** of GPU time. That is the same with a serial encoder, explicit barriers, or Metal 4 intra-pass barriers.
   - Independent dispatches pipeline at ≈0.14 µs.
   - A CPU→GPU→CPU round trip is ≈100–125 µs.
   - Replaying an ICB costs ≈4 µs of CPU for any N [V, measured].
6. **Persistent kernels are possible but unattractive on this hardware.**
   - The MSL 4.1 spec never mentions forward progress [V].
   - A relaxed `atomic_load` spin **never observes** other threadgroups' increments. The spin needs a device-scope fence or an RMW.
   - A working device-wide barrier costs **1.1–1.3 µs**, the same as a dispatch boundary.
   - Oversubscribing residency (more than 48 × 1024-thread threadgroups on the 16-core M2 Pro) deadlocks [V, measured].
   - macOS kills long GPU work while a display is active (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`); llama.cpp sets the undocumented `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` [V, source].
7. **WWDC26 (June 2026) changes:**
   - Metal 4.1 / OS 27 adds MXFP-style block-scaled tensors (`tensor_blockwise`, E8M0 scales over 32-element blocks), fp8/fp4/int2 MPP operands, and acquire/release atomics.
   - **Core AI** is a new framework (CPU/GPU/ANE, PyTorch conversion, custom Metal kernels, OS 27) [V].
8. **Rust bindings:**
   - `objc2-metal` 0.3.2 already exposes `MTL4*` and `MTLTensor` from the macOS 26 SDK. It lacks the 26.4/27 additions: Int4 data types and auxiliary planes [V].
   - `metal-rs` is deprecated [V].

---

## Part A — Facts with sources

### A1. Chip table: M1 to M5 and M6

Compiled by helper agent R6a from Apple newsroom and spec pages. Every cell is sourced in `$S/research/R6-work/chips/`. Key sources: support.apple.com tech-spec pages; apple.com/{macbook-pro,mac-studio}/specs; the Metal Feature Set Tables PDF (dated May 21 2026) for the families; Philip Turner `metal-benchmarks/README.md:57-78` for M1/M2 clocks and SLC.

**How compute is derived:**
- FP32 = cores × 128 lanes × 2 × clock.
- **FP16 equals FP32 on Apple7/8** [V: metal-benchmarks README.md:124-127].
- **Apple9+ can co-issue FP16, FP32 and INT for up to 2× on mixed code** [V: Apple tech talk 111375]. It is not 2× on pure FP16.
- **Apple publishes no GPU clock or TFLOPS for any M3–M6 chip** [V: R6a grepped every M5/M6 newsroom and spec page].

| Chip | Family | GPU cores | GPU clock (MHz) | FP32 TFLOPS (all bins) | Memory, bandwidth (per bin) | Max RAM | SLC |
|---|---|---|---|---|---|---|---|
| M1 | Apple7 | 7/8 | 1278 [V-ind] | 2.29/2.62 (Apple: "2.6 teraflops") | LPDDR4X-4266, ~68 GB/s [L] | 16 GB | 8 MB [L] |
| M1 Pro / Max / Ultra | Apple7 | 16 / 32 / 64 (top bins) | 1296 [V-ind] | 5.3 / 10.6 / 21.2 | LPDDR5-6400: 200 / 400 / 800 GB/s | 32 / 64 / 128 GB | 24 / 48 / 96 MB [L] |
| M2 | Apple8 | 8/10 | 1398 [V-ind] | 2.86/3.58 | LPDDR5-6400, 100 GB/s | 24 GB | 8 MB [L] |
| **M2 Pro (this machine: 16-core)** | Apple8 | 16/19 | 1398 [V-ind] | **5.73**/6.80 | LPDDR5-6400, **200 GB/s** | 32 GB | 24 MB [L] |
| M2 Max / Ultra | Apple8 | 30–38 / 60–76 | 1398 | 10.7–13.6 / 21.5–27.2 | 400 / 800 GB/s | 96 / 192 GB | 48 / ~96 MB [L] |
| M3 / Pro / Max / Ultra | Apple9 | 10 / 18 / 30–40 / 60–80 | ~1380 [L] | 3.5 / 6.4 / 10.6–14.1 / 21–28 | 100 / 150 / 300–400 / 819 GB/s | 24 / 36 / 128 / 512 GB | unknown |
| M4 / Pro / Max | Apple9 | 10 / 20 / 32–40 | ~1578 [L] | 4.0 / 8.1 / 12.9–16.2 | LPDDR5X: 120 / 273 / 410–546 GB/s | 32 / 64 / 128 GB | unknown |
| M5 | Apple10 | 8/10 | 1578 [E] | 3.2/4.0 | LPDDR5X-9600, 153 GB/s | 32 GB | unknown |
| **M5 Pro** | Apple10 | 16/20 | ~1580–1620 [L] | ~6.5/~8.2 | **307 GB/s** (both bins) | 48 GB (16c) / 64 GB (20c) | unknown |
| **M5 Max** | Apple10 | **32 / 40** | ~1620 [E: FB23754032 powermetrics] | ~13.3 / **~16.6** | **460 GB/s (32c) / 614 GB/s (40c)** | **36 GB (32c only) / 128 GB (40c)** | unknown |
| M5 Ultra (Aug 2026) | Apple10 [L] | 64/80 | ~1550–1620 [L] | ~25–33 | **1.2 TB/s (both bins)** | 96 GB / 512 GB | unknown |
| M6 (Aug 2026) | Apple11 [L] | 12 | ~1700 [L] | ~5.2 | 153–170 GB/s | 32 GB | unknown |

Notes on the table:
- **[V-ind]** marks values verified by independent measurement (metal-benchmarks README, or eclecticlight powermetrics for M3 Pro / M4 Pro).
- **Families** [V]: Metal Feature Set Tables p.2 lists "M1-series … Apple7 / M2-series … Apple8 / M3-series, M4-series … Apple9 / M5-series … Apple10". `MTLGPUFamily.apple10` and `.apple11` exist in the docs with no abstract (developer.apple.com/documentation/metal/mtlgpufamily/apple11). Our probe on the M2 Pro returns apple1–8 = true, apple9–12 = false, metal3 = true, metal4 = true [V: results/probe.txt].

#### M5 Pro / M5 Max official specs [V]

Sources:
- N-M5PM: apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- support.apple.com/en-us/126318 and /126319
- apple.com/macbook-pro/specs/

| | M5 Pro | M5 Max |
|---|---|---|
| CPU | "15-core CPU with 5 super cores and 10 performance cores" or "18-core … 6 super cores and 12 performance cores" | 18-core (6 super + 12 performance) in both bins |
| GPU | 16 or 20 cores | 32 or 40 cores |
| Bandwidth | **307 GB/s for both bins** | **460 GB/s (32-core GPU) or 614 GB/s (40-core GPU)** |
| Memory | 24 GB base; 48 GB; 64 GB with the 20-core GPU only | 36 GB with the 32-core GPU only; 48, 64 or 128 GB with the 40-core GPU |
| Neural Engine | 16-core | 16-core |

Architecture quotes (N-M5PM):
- "Fusion Architecture that connects two dies … into a single SoC": two third-generation 3 nm dies.
- "With a Neural Accelerator in each GPU core and higher unified memory bandwidth, M5 Pro and M5 Max are over 4x the peak GPU compute for AI compared to the previous generation."
- "second-generation dynamic caching", "third-generation ray-tracing engine", and "a faster 16-core Neural Engine with a higher bandwidth connection to memory".
- **Die split** [L, Ars Technica via mirror]: die 1 holds the CPU, Neural Engine and I/O; die 2 holds the GPU and memory controller.

Apple's AI claims (exact):
- **MacBook Pro:** "up to 4x faster LLM prompt processing than M4 Pro and M4 Max". Footnote: "Time to first token measured with a 16K-token prompt using a 14-billion parameter model with 4-bit weights and FP16 activations, mlx-lm, and MLX framework". **This is a prefill claim, not decode.**
- **Mac Studio (Aug 2026):** "Up to 10.7x faster LLM prompt processing in LM Studio when compared to Mac Studio with M1 Max, and 3.9x faster than M4 Max". Footnote: 8K-token prompt, 14B 4-bit, LM Studio 0.4.19.
- **Decode** is bounded by bandwidth. M4 Max to M5 Max is at most 614/546 = 1.125× [L].

#### M5 Ultra, Mac Studio and Mac Pro status [V]

Sources:
- apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
- apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
- apple.com/mac-studio/specs

Facts:
- M5 Ultra was announced 2026-08-25 and shipped 2026-09-22.
- "M5 Ultra uses UltraFusion to connect two dual-die M5 Max chips to form the quad-die architecture"; inter-die bandwidth is "over 4.4TB/s".
- Configurations:

  | Bin | CPU | GPU | Neural Engine | Bandwidth | Memory |
  |---|---|---|---|---|---|
  | Base | 30-core | 64-core | 32-core | 1.2 TB/s | 96 GB |
  | Top | 36-core | 80-core | 32-core | 1.2 TB/s | 96, 256 or 512 GB |

- The 512 GB configuration ships "late October".
- The Mac Studio now offers M5 Max or M5 Ultra.
- The Mac Pro is discontinued: apple.com/mac-pro/ returns a 301 redirect (observed).
- M6 (first 2 nm part) is in the Mac mini: 12-core GPU, Neural Accelerators, up to 170 GB/s.

### A2. GPU Neural Accelerators (M5 family, A19 / A19 Pro)

Helper R6b collected the sources. I re-checked the key quotes myself.

**What they are**
- "The neural accelerator is a new hardware block in M5, located directly in each shader core. It sits alongside the other GPU pipelines and is designed to accelerate dense compute-bound work such as the prefill stage of an LLM." [V: WWDC26-330 transcript, developer.apple.com/videos/play/wwdc2026/330/]
- There is one accelerator per GPU core: M5 has 10; M5 Pro 16/20; M5 Max 32/40; M5 Ultra 64/80; A19 has 5 and A19 Pro 6 [V: newsroom].

**Per-core throughput**
- FP16 matrix MAC: **≈1024 FLOP per core per clock**, with FP16 or FP32 accumulate.
- INT8→INT32: ≈2048 OP per core per clock.
- BF16 is not exposed.
- Source: tzakharko.github.io/apple-neural-accelerators-benchmark/, measured on an A19 at ≈7.4 TFLOPS on 5 cores [L].

**M5 Max measurement** [L]
- Anemll (github.com/Anemll/fp8-mlp-metal41-bench, macOS 27, Mac17,6, 40-core, 128 GB) reports FP16 GEMM **63.8 TFLOPS** and **≈65 TFLOPS** on a pure-compute probe.
- FP8 runs at the same rate, about 65 TFLOPS. FP4 is about 50–57. **INT8×INT8 is about 122 TOPS.**
- The `simdgroup_matrix` ALU path reaches ≈14.6–15.5 TFLOPS.
- Cross-check: 1024 × 40 cores × 1.6 GHz = 65.5 TFLOPS, which agrees with the Anemll numbers.
- Tile shape matters: Apple's sample 64×32 tile reaches only about half of peak, and 64×64 or larger tiles are needed (Anemll README:30).

**Framework-level M5 Max numbers** [E]
- These come from MLX issue #4411 (`mx.matmul` 1024×4096×4096), comparing the NA path against NA forced off.
- fp16: 46.7 vs 14.0 TFLOPS. bf16: 36.7 vs 13.9. q4: 35.0 vs 12.3.

**Official claims and Apple's measured LLM results** [V]
- "Matrix multiplication operations … are up to 4 to 8 times faster, depending on precision" (Apple tech talk 111432, developer.apple.com/videos/play/tech-talks/111432/).
- Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (2025-11-19, machinelearning.apple.com/research/exploring-llms-mlx-m5), M5 vs M4, mlx_lm:
  - TTFT is 3.3–4.1× faster; Qwen3-14B-4bit gains 4.06×.
  - Generation speed improves only 1.19–1.27×, which tracks the +28% bandwidth.
  - The post notes "MLX requires macOS 26.2 or later".

**Data types**
- MSL 4.1 Table 7.3 (spec pp.351–354) lists every `matmul2d` A×B→C combination and the OS that introduced it [V]:
  - **Metal 4 (OS 26.0):** char/half/float combinations, plus char×char→int.
  - **OS 26.1:** bfloat combinations.
  - **OS 26.4:** uchar combinations, half/bfloat × int4b/uint4b → half/bfloat/float, and char×int4b→int.
  - **Metal 4.1 (OS 27):** int2b, `metal_fp4_e2m1`, `metal_fp8_e4m3`/`e5m2` operands, and fp8×fp8 / fp4×fp4.
- Relative rates on M5 Max (Anemll) [L]:
  - int8×int8 ≈2×.
  - fp8 ≈1×; fp4 ≈0.8–1×.
  - half×MXFP4 0.69–0.78×; MXFP4×MXFP4 0.5–0.6×.
  - float×float is accelerated only with `relaxed_precision`, a TF32-like mode.
  - MLX defaults to `MLX_ENABLE_TF32=1` (`mlx/utils.h:225-227`).

**How software reaches them** [V]
- The tech talk: "TensorOps … The API is portable. The same code runs across Apple's entire GPU family from M1 to M5. On older GPUs without neural accelerators, TensorOps falls back to optimized shader implementations."
- WWDC26-330: TensorOps "automatically uses any available hardware acceleration across all Apple Silicon GPU generations".
- MPS, MPSGraph, Core ML and Core AI use them automatically. MLX, llama.cpp and PyTorch already do too (tech talk 111432).
- **The MPP header delegates every matmul to backend-provided functions**, `extern "C" __attribute__((section("air.externally_defined")))` with names like `__tensorops_impl_matmul2d_op_run_dv_f16_dv_f16_dv_f32`.
  - Source: `/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/__impl/MPPTensorOpsMatMul2dImpl.h:18-20,47-60` [V].
  - The GPU-specific compiler backend picks the implementation, so **we cannot see or control how MPP is lowered.**

**Does `simdgroup_matrix` use them?** **No** [V].
- Tech talk 111432: "during v1 [simdgroup_matrix GEMM], our neural accelerator utilization is 0%. All of this compute work is happening on the ALU … dedicated matrix hardware is sitting completely idle." Moving to TensorOps gave about 4×.
- Apple's advice: "if you're already writing your own custom kernels in metal using SIMD Group matrix API, you should move your workloads over to adopt TensorOps instead."

**Batch-1 decode: no gain** [V]
- Apple's ML post attributes the decode gain to bandwidth.
- In llama.cpp the tensor matmul is used only when the batch exceeds 8 (`ggml/src/ggml-metal/ggml-metal-common.cpp:35-42`).
- MLX keeps small-M on the ALU `qmv` kernel (`mlx/backend/metal/quantized.cpp:89-99`).
- On M5, llama.cpp tg128 is unchanged with the tensor API on or off (22.88 vs 22.56 tok/s; PR #16634).
- The NA *can* help **speculative-decoding verification** and **batched decode**. The ridge is ≈30 rows on M5 Max (see the hardware model) [L].

**Gating in existing runtimes** [V]
- llama.cpp disables the tensor API on pre-M5 chips. The comment reads "M2 Ultra: ~5% slower / M4, M4 Max: no significant difference" (`ggml/src/ggml-metal/ggml-metal-device.m:1148-1162`). It also test-compiles a probe kernel (`:1165-1210`).
- MLX enables NAX only on macOS ≥ 26.2 with GPU arch generation ≥ 17 (Mac) or ≥ 18 (phone): `mlx/backend/metal/device.cpp:952-971`.
- MLX's quantized NAX kernels **dequantize to half/bf16 in threadgroup memory** and do not use MPP's native int4 (`quantized_nax.h:488`).
- These citations come from R6b. MLX clone HEAD is 0840c42.

### A3. Metal 4 (WWDC25) and Metal 4.1 / WWDC26

All of this is [V] unless marked. Doc URLs are `developer.apple.com/documentation/metal/<lowercased name>`. Text copies are in `$S/research/R6-work/docs/`.

**Availability**
- MTL4 types are macOS/iOS 26.0+.
- Feature set tables:
  - Metal 4 requires Apple7+ (M1+).
  - "Tensors Metal 4 Apple7"; "Machine learning encoding Metal 4 Apple7".
  - "Acquire and release memory ordering atomics Metal 3 & 4 Apple7"; this needs MSL 4.1 / OS 27.
- On our M2 Pro (macOS 26.6.1):
  - `newMTL4CommandQueue` returns `AGXG14XFamilyCommandQueue_mtlnext`.
  - `newCommandBuffer` returns `AGXG14XFamilyCommandBuffer_mtlnext`.
  - Probe output: `results/probe.txt`. 51 MTL4/tensor protocols and 108 classes are present at runtime.

**Command model** (article "Understanding the Metal 4 core API")

| Area | Detail |
|---|---|
| `MTL4CommandQueue` (`makeMTL4CommandQueue()`) | Commits arrays of command buffers: `commit(_:options:)`, `commit:count:`. Has `signalEvent`/`waitForEvent`, `addResidencySet`, sparse mapping ops. Apple says it "reduces CPU runtime and memory overhead by sending work to the GPU when you commit". |
| `MTL4CommandBuffer` | Made by the **device** (`makeCommandBuffer()`). **Reusable**: "reuse and repurpose each command buffer indefinitely by starting over" (`beginCommandBuffer(allocator:)` … `endCommandBuffer()`). **Holds no strong references to resources.** Has `useResidencySet`, `writeTimestamp(counterHeap:index:)` and `resolveCounterHeap`. |
| `MTL4CommandAllocator` | Owns the encoding memory. `reset()` once the GPU has finished. One allocator per frame or token in flight. |
| `MTL4ComputeCommandEncoder` | Merges compute, blit and acceleration-structure work. Has dispatch, indirect dispatch (by `MTLGPUAddress`), `copyFromBuffer`, `fillBuffer`, **`executeCommandsInBuffer` (ICB, including an indirect-range variant)**, `copyFromTensor`, `writeTimestamp(granularity:…)` and `setArgumentTable`. There are no per-slot `setBuffer` calls. Exact signatures: metal-cpp `Metal/MTL4ComputeCommandEncoder.hpp`. |
| `MTL4ArgumentTable` (descriptor `maxBufferBindCount`, …) | `setAddress(_:index:)` takes a GPU address (`MTLBuffer.gpuAddress` plus offset), and also `setResource`/`setTexture`. One table can be shared by many encoders. |
| Hazards | "**In Metal 4, the framework considers all resources untracked.**" `hazardTrackingMode` "has no effect on the work you submit to an `MTL4CommandQueue`" (article "Resource synchronization"). |
| Synchronization, smallest scope first | Intra-pass barriers (`barrierAfterEncoderStages(_:beforeEncoderStages:visibilityOptions:)` with `MTLStages.dispatch`, `.blit`, `.machineLearning` …, `MTL4VisibilityOptions.device` / `.resourceAlias`); fences; consumer and producer queue barriers (`barrierAfterQueueStages…`, `barrierAfterStages…`); `MTLEvent`; `MTLSharedEvent`. Events work across MTL3 and MTL4 queues. |
| `MTL4CommitFeedback` | `gpuStartTime`/`gpuEndTime`/`error`, delivered by `MTL4CommitOptions.addFeedbackHandler`. Errors are `MTL4CommandQueueError`: none/**timeout**/notPermitted/outOfMemory/deviceRemoved/accessRevoked/internal (metal-cpp `MTL4CommandQueue.hpp`). |
| Timing | `MTL4CounterHeap` (type `.timestamp`), with `writeTimestamp(granularity .relaxed/.precise)`. |

**Compilation** (article "Using the Metal 4 compilation API")
- **Compiler:** `MTL4Compiler`, created with `makeCompiler(descriptor:)`. It offers synchronous and asynchronous `makeComputePipelineState(descriptor:dynamicLinkingDescriptor:compilerTaskOptions:)`, `makeLibrary(descriptor:)` and `makeMachineLearningPipelineState(descriptor:)`. Async compilation runs on a QoS dispatch queue.
- **Descriptors and serialization:** `MTL4LibraryDescriptor`, `MTL4LibraryFunctionDescriptor`, `MTL4SpecializedFunctionDescriptor`, `MTL4ComputePipelineDescriptor`; `MTL4PipelineDataSetSerializer` for harvesting; `MTL4Archive` (`makeArchive(url:)`).
- **Archives:** "both Metal 3 and 4 can load binary archives".
- **Flexible pipeline states** (unspecialized pixel formats and so on) are a **render** feature only.
- **Offline compile needs Xcode.** The `metal` CLI and `metal-package-builder` ship with Xcode 26 [V: doc text; locally `xcrun -f metal` fails]. Runtime source compilation works without Xcode [V, measured].
- **Cold start** [L]: compiling a pipeline that uses tensor ops takes ≈1–2 s (llama.cpp PR #16634 comment). Cache the binaries (archives) for TTFT.

**`MTLTensor` (`MTLTensorDescriptor`)**
- Descriptor fields: `dimensions`/`strides` as `MTLTensorExtents` (rank 0–16), `dataType`, `usage` (`.compute` / `.render` / `.machineLearning`), plus storage and hazard options.
- Created with `device.makeTensor(descriptor:)` or from a buffer.
- `MTLTensorDataType` values (metal-cpp `Metal/MTLTensor.hpp:41-59`):

  | Values | Raw value | Since |
  |---|---|---|
  | Float32 | 3 | macOS 26.0 |
  | Float16 | 16 | macOS 26.0 |
  | BFloat16 | 121 | macOS 26.0 |
  | Int8 / UInt8 | 45 / 49 | macOS 26.0 |
  | Int16 / UInt16 | 37 / 41 | macOS 26.0 |
  | Int32 / UInt32 | 29 / 33 | macOS 26.0 |
  | Int4 / UInt4 | 143 / 144 | **macOS 26.4** (metal-cpp diff 26 → 26.4) |
  | MetalFloat8E5M2 / E4M3 | 141 / 142 | **macOS 27** |
  | MetalFloat8UE8M0 | 145 | **macOS 27** |
  | MetalFloat4E2M1 | 148 | **macOS 27** |
  | UInt2 / Int2 | 149 / 150 | **macOS 27** |

- **Alignment for format types** (Feature Set Tables, "Tensor limits by data type"):
  - extents[0] must be a multiple of 32 elements.
  - The row stride must be a multiple of 256 elements for int4, 512 for int2 and 128 for fp8.
  - Buffer offsets must be 128-byte aligned.
- MSL side (§2.22): `tensor<ElementType, Extents, DescriptorType, Tags…>` with `tensor_handle` (host-bound), `tensor_inline` (built in the shader from a pointer, §2.22.2.7) or `tensor_offset`, plus `dextents<>`.
  - **Format types need 128-byte-aligned pointers and 128-byte row strides** (§2.22.2.7, p.101).
  - The `tensor_inline` data pointer for `int4b_format` is `device uchar*` [V: compiler error text in results/probe.txt].

**Metal Performance Primitives (MSL 4.1 §7, pp.349–367; the spec's §1.3 calls it "17.2.1")**
- **Execution scopes:** `execution_thread`, `execution_simdgroup`, and `execution_simdgroups<N>` with N = 1 or simdgroups_per_threadgroup.
- **matmul2d:**
  - `tensor_ops::matmul2d<matmul2d_descriptor(M,N,K, transpose_left, transpose_right, relaxed_precision, mode::multiply|multiply_accumulate), Scope>`, called through `run(A,B,C)`.
  - C may be a `cooperative_tensor`, obtained from `get_destination_cooperative_tensor<…>()`.
  - Cooperative inputs come from `get_left_input_cooperative_tensor` and `get_right_input_cooperative_tensor` (OS 26.1+), checked with `is_compatible_as_left_input`.
  - Row and column reductions are `reduce_rows`/`reduce_columns` (sum/max/min) and `map_iterator`.
- **convolution2d:** `convolution2d_descriptor`, NHWC activations, HWIO weights; the only supported scope is `execution_simdgroups<N>` with N = simdgroups_per_threadgroup.
- **Barrier rule:** results written to device or threadgroup tensors need a barrier at the matching scope (§7.2, p.350).

**Quantized and block-scaled formats**
- **Formats** (§2.21 "Packed Numeric Type", §2.22.2.2 "Tensor Blockwise"):
  - `int4b_format`/`uint4b_format` arrived with "Metal 4.0 SDK 26.4".
  - `int2b`/`uint2b`, `metal_fp4_e2m1`, `metal_fp8_e4m3`/`e5m2` and `metal_fp8_ue8m0` are Metal 4.1.
- **Block scaling** (§2.22.2.2 Table 2.28, p.89):
  - The PlaneTag is "Only `tensor_plane_scales`".
  - The ElementType is "Only `metal_fp8_ue8m0_format`".
  - For BlockSizes, "The first dimension must be 32. All other dimensions must be 1".
  - **The only scaled format is OCP-MX-style: power-of-two E8M0 scales per 32 elements along the innermost dimension.**
  - **No fp16 scales or biases.** MLX affine 4-bit (group 64, fp16 scale and bias) and GGUF Q4_K/Q4_0 have no native representation. You either repack to MXFP4/int4 or dequantize yourself, for example into a cooperative tensor (WWDC26-330).
- **Hardware rates** for these formats on M5 Max (Anemll) [L]: FP8 ≈ FP16, FP4 slower, MXFP4 0.5–0.78× of FP16. **There is no compute win below 8 bits.** The gain is only in memory bandwidth on thin (decode-like) shapes.

**MTL4MachineLearningCommandEncoder** (article "Machine learning passes")
- Runs **Core ML models** converted to a "Metal ML package" (`metal-package-builder`, Xcode 26) on the GPU timeline.
- API: `setPipelineState(MTL4MachineLearningPipelineState)`, `setArgumentTable`, `dispatchNetwork(intermediatesHeap:)`.
- "The system automatically chooses an inference engine, such as a device's GPU or Apple Neural Engine (ANE) for each machine learning model."
- It **cannot build or modify networks**. Not useful for a compiler that generates its own kernels [L].

**What changed at WWDC 2026 (June 2026; MSL spec v4.1 dated 2026-06-04)**
- **MSL 4.1** (§1.3):
  - placement new
  - packed numeric block-scaling formats
  - multiplane tensors and `tensor_blockwise`
  - interleave and deinterleave
  - **memory order including acquire/release for barriers and atomics** (§6.10, §6.16)
  - new texture read modes
  - new MPP matmul dtypes
- **API diff** (metal-cpp `release/metal-cpp_macOS26.4_iOS26.4` → `release/metal-cpp_macOS27_iOS27`, 19 files):
  - tensor auxiliary planes: `MTLTensorAuxiliaryPlaneDescriptor`/`Map` and `MTLTensorPlaneType` (data/scales)
  - `MTLTensorBufferAttachments` and `newTensorWithDescriptor:attachments:error:`
  - `copyFromTensor(…plane…)` on MTL4 compute and blit encoders
  - the fp8, fp4, int2 and ue8m0 data types
  - **`MTLLanguageVersion.version4_1` is macOS 27.0+** [V: doc]
- **MSL 4.1 is unavailable on our machine** [V, measured]. On macOS 26.6.1, MSL 4.1, the `metal_fp8_e4m3_format` type and `memory_order_release` are all rejected: "invalid value 'metal4.1' in '-std=metal4.1'" and "use of undeclared identifier 'memory_order_release'" (results/probe.txt).
- **Sessions:** WWDC26-330 "Optimize custom machine learning operations with Metal tensors", 359 "Build real-time neural rendering pipelines with Metal", and 388 "Find and fix performance issues in your Metal games".

**Core AI** (new at WWDC26) [V]
- **What it is:**
  - "Core AI is a brand new framework built right into the platform … designed to be the best way to bring and run models on device" (Platforms State of the Union, WWDC26-102).
  - "the inference framework powering on-device Apple Intelligence … fast inference across the CPU, GPU, and Neural Engine … modern Swift API" (WWDC26-324).
- **Toolchain:** PyTorch → Core AI conversion with `coreai-torch`; ahead-of-time compilation; per-device "specialization" with a model cache; "states" for in-place KV caches; Instruments support; a debugger.
- **Custom kernels:** custom Metal 4 kernels through `TorchMetalKernel` (WWDC26-325/330).
- **Availability:** models target "iOS 27.0 and macOS 27.0" (WWDC26-326).
- **Relation to Core ML:** described as "familiar" to Core ML users (WWDC26-325). There is no explicit deprecation statement for Core ML.
- **Relation to MLX:** MLX remains Apple's research and local-inference framework. The State of the Union says "MLX … now supports Metal 4, GPU Neural Accelerators, and … RDMA over Thunderbolt".
- **Performance** [E]: Anemll measured Core AI reaching ≈65 TFLOPS FP16 on the M5 Max GPU, but with no native INT8×INT8 path.

### A4. Compute details for kernel design

**Basic limits**
- SIMD width is 32. Max threads per threadgroup is 1024. Threadgroup memory is **32 KB** per threadgroup [V]:
  - Probe: `threadExecutionWidth=32`, `maxThreadgroupMemoryLength=32768`.
  - Feature Set Tables: "Maximum total threadgroup memory allocation … 32 KB" for Apple4–Apple10, so **still 32 KB on M5**.
  - `setBytes` is limited to 4 KB; there are 31 buffer argument slots.
- **Register file:** ≈208 KB per core on Apple7/8, which bounds residency at 384–3072 threads per core (metal-benchmarks README.md:83-85) [V-ind].
- **Measured co-residency on M2 Pro** [V, measured: results/persist.txt]:
  - **48 threadgroups × 1024 threads were co-resident** (3 × 1024 per core × 16 cores). With 64 such threadgroups, the extra 16 were never scheduled until others exited.
  - Metal has **no API to query residency or occupancy** beyond `maxTotalThreadsPerThreadgroup` [L].
- **Apple9+ "Dynamic Caching"** allocates registers and threadgroup memory dynamically, so occupancy depends on actual use [V: Apple tech talk 111375; we have not measured it].
- **`simdgroup_matrix`** is 8×8 only, with T = half, float, or bfloat (Metal 3.1+) (MSL 4.1 §2.4 p.38; functions in §6.7). It runs on the ALUs [V].
  - Measured register-only peak on M2 Pro: 5.44 TFLOPS for half accumulate and 4.67 TFLOPS for float accumulate [V, measured: results/gemm.txt].
- **SIMD shuffles and reductions** (`simd_sum`, `simd_shuffle_xor`, quad ops) are in §6.10.2–6.10.3.

**Memory model**
- **Coherence** (§4.8, p.122–123):
  - "**By default, memory in the device address space has threadgroup coherence.**"
  - Metal 3.2+ adds the `coherent(device)` buffer qualifier (and `memory_coherence_device` for textures) for device coherence.
- **Atomics** (§6.16.1, p.299–300):
  - Before Metal 4.1, "for atomic operations other than `atomic_thread_fence`, `memory_order_relaxed` is the only enumeration value".
  - `atomic_thread_fence(mem_flags, memory_order_seq_cst, thread_scope_device)` is Metal 3.2+.
  - acquire, release and acq_rel are **Metal 4.1+**.
- Thread scopes are thread, simdgroup, threadgroup and device (§6.16.2, Metal 3.2+).
- Docs: "you don't need to synchronize compute or render passes when they access an instance of an atomic type" (article "Resource synchronization").

**Persistent kernels and megakernels: negative results**
1. **No forward-progress guarantee** [V]. The phrase "forward progress" (or "deadlock" or "spin") appears **nowhere** in the MSL 4.1 spec (grep of the full text). Apple documents no ordering or residency guarantee for threadgroups.
2. **Relaxed atomic loads can spin forever on a stale value** [V, measured: results/persist.txt].
   - A device-wide barrier whose spin was `while (atomic_load_explicit(ctr, relaxed) < target)` **never completed** for 16 threadgroups. Every threadgroup hit our spin bound, with or without `coherent(device)` on the counter.
   - What works:
     - adding `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` inside the loop: 1.26 µs per barrier
     - spinning on an RMW (`atomic_fetch_add(ctr, 0)`): **1.11 µs per barrier**
   - Neither variant showed visibility violations.
3. **Grids beyond residency deadlock.** 64 × 1024-thread threadgroups produced 48 timeouts [V, measured]. A persistent kernel must size its grid to the co-resident capacity, and that capacity is not queryable.
4. **The GPU watchdog** [V, source and issues].
   - macOS kills GPU work that blocks display composition with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`.
   - llama.cpp sets `setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true)`, commented "workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)" (`ggml/src/ggml-metal/ggml-metal.cpp:1027-1029`; ref llama.cpp issue #20141).
   - MLX issue #3267 ("Metal GPU watchdog kills LoRA training when display is active") reports 100% reproducible kills with the display on, none with it off. On an **M5 Max** (macOS 26.5.1), sustained training escalated to `watchdogd` kernel panics while the display was active, even with the env var set.
   - The env var is **undocumented** [L: it does not appear in Apple's docs]. There is no documented time limit; `MTL4CommandQueueError.timeout` and `MTLCommandBufferError` timeout codes exist.
   - **We did not run a watchdog test**, because it could disrupt the shared machine.
5. **Cost.** A working device-wide barrier (1.1–1.3 µs) costs the same as a dependent dispatch boundary (1.1–1.25 µs). **A megakernel buys nothing for global syncs on M2 Pro.** Its only possible wins are finer-grained producer/consumer flags that avoid full-grid syncs, and no CPU in the loop [L].

**Indirect command buffers and GPU-driven work**
- Compute ICBs are supported from Apple3 with "Maximum indirect command buffer length: No limit" [V: Feature Set Tables].
- Descriptor: `MTLIndirectCommandBufferDescriptor` with `commandTypes = .concurrentDispatch`, `inheritBuffers`, `inheritPipelineState`, `maxKernelBufferBindCount`.
- Pipelines need `supportIndirectCommandBuffers = true`.
- Commands are `MTLIndirectComputeCommand` with `concurrentDispatchThreadgroups` and **`setBarrier`**.
- GPU-side encoding uses MSL `command_buffer` / `compute_command`. We verified it works [V, measured].
- Metal 4 keeps ICBs: `MTL4ComputeCommandEncoder.executeCommandsInBuffer(_:range:)` and the variant taking an **indirect range buffer**, so the GPU decides how many commands run [V: metal-cpp header].
- **Indirect dispatch** lets the GPU choose grid sizes (`dispatchThreadgroups(indirectBuffer:…)` in MTL3; by GPU address in MTL4).
- **The GPU cannot submit command buffers or create new work outside pre-allocated ICB slots** [L: there is no such API].

**Storage, zero-copy and memory limits**
- Shared and private storage modes showed **no bandwidth difference** on M2 Pro (181–182 GB/s for both with float4 reads) [V, measured: results/bw.txt].
- `makeBuffer(bytesNoCopy:length:options:deallocator:)`:
  - "A page-aligned pointer", with a length "that results in a page-aligned region".
  - "The existing memory allocation needs to exist within a single virtual memory (VM) region" [V: doc]. Page size is 16 KB.
- **Read-only mmapped files work in production** [V]:
  - llama.cpp maps GGUF with `mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)` (`src/llama-mmap.cpp:485-495`).
  - It wraps the mapping with `newBufferWithBytesNoCopy:… options:MTLResourceStorageModeShared` (`ggml/src/ggml-metal/ggml-metal-device.m:2214`).
  - When a mapping exceeds `maxBufferLength` it splits it into **overlapping views** (`:2228-2241`).
- `maxBufferLength` is "at least 256 MB" per the doc. On our 16 GB M2 Pro it is **8.88 GiB** [V: probe], so Qwen3-32B weights need several buffers or views.
- `recommendedMaxWorkingSetSize` is "An approximation of how much memory … this GPU device can allocate without affecting its runtime performance". On our machine it is **11.84 GiB, 74% of 16 GiB** [V: doc, probe].
- `iogpu.wired_limit_mb` (default 0) is raised by `sudo sysctl iogpu.wired_limit_mb=N` to fit large models (mlx-lm `README.md:286`) [V: source].
- MLX's `set_wired_limit` is implemented with **residency sets** and capped at the recommended working set (`mlx/backend/metal/allocator.cpp:100-104,254-262`) [V].
- **Residency sets:** `MTLResidencySet` (macOS 15+) with `requestResidency`; MTL4 queues and command buffers add them, up to 32 per queue (Feature Set Tables) [V]. **In Metal 4 every resource must be made resident through a residency set**, because there is no `useResource` on MTL4 encoders [L, from the API surface].

**Timing and profiling without Xcode**
- What works [V, probe]:
  - `MTLCommandBuffer.gpuStartTime`/`gpuEndTime` (used throughout) and `MTL4CommitFeedback`.
  - The `MTLCounterSampleBuffer` **timestamp counter set only, at stage boundaries only**. On M2 Pro, `supportsCounterSampling(atStageBoundary)=true`; atDispatch, atDraw, atTileDispatch and atBlit are false. Per-kernel GPU times therefore need one encoder per kernel, or MTL4 `writeTimestamp`.
  - `MTL4CounterHeap` timestamps.
- [L]:
  - The Metal Performance HUD (`MTL_HUD_ENABLED=1`).
  - `powermetrics --samplers gpu_power` (needs sudo) for GPU clock and power.
  - Programmatic `MTLCaptureManager` → `.gputrace`, though viewing it needs Xcode.
  - Instruments' Metal System Trace needs Xcode.

### A5. Rust bindings

- **objc2-metal** [V: crates.io API, docs.rs]:
  - Latest is **0.3.2**, released 2025-10-04. It includes `MTL4CommandQueue`, `MTL4ComputeCommandEncoder`, `MTL4MachineLearningCommandEncoder`, `MTLTensor` and others behind per-header Cargo features (`MTL4CommandQueue`, `MTL4ArgumentTable`, `MTL4Compiler`, `MTLTensor`, …; 105 features).
  - **Missing:** the OS 26.4/27 additions. `MTLTensorDataType` has only the 26.0 cases, and there are no auxiliary-plane types.
  - `MTLTensorDataType` is a newtype over `NSInteger`, so raw values (143 = Int4, …) can be passed.
- **metal-rs** (crate `metal` 0.33.0, 2025-12-17) [V]. Its README says: "Use of this crate is deprecated as the objc ecosystem … are unmaintained. For new development, please use objc2 and objc2-metal instead."
- **Other options:**
  - Apple's **metal-cpp** covers the full macOS 27 API. We used it here because the local SDK lacks MTL4 headers.
  - A thin Objective-C or Swift shim.
- **Recommendation** [L]:
  - Drive Metal from Rust with `objc2-metal` 0.3.x: Metal 3 path everywhere, and the MTL4 path behind a runtime check (`supportsFamily(.metal4)` = raw value 5002, plus `respondsToSelector`).
  - Declare missing 27.0 selectors with `objc2::extern_protocol!`/`msg_send!` until objc2 regenerates.
  - Generate MSL as text and compile at runtime with `newLibraryWithSource`. This needs no Xcode.
  - Cache pipelines with `MTLBinaryArchive` / `MTL4Archive` to hide the 1–2 s tensor-op compile.
  - Keep one backend abstraction so M1–M4 run `simdgroup_matrix` or MPP kernels and M5+ run MPP TensorOps.

---

## Part B — Local experiments (Apple M2 Pro 16-core GPU, macOS 26.6.1)

Sources, build and run commands are in `$S/research/microbench/README.md`. Raw outputs are in `$S/research/microbench/results/`.

**Conditions:**
- Every timing run was serialized through `benchlock.sh`.
- Each run had 2–3 warmups followed by 7–12 timed repetitions; medians are reported.
- Three displays were attached, so WindowServer took some GPU time and bandwidth.
- The GPU clock was not pinned.
- **Run-to-run variance between separate runs is about 10–20% for µs-scale dispatch numbers** (compare dispatch.txt with icb.txt). Bandwidth numbers vary by about 2%.

### B1. Feature probe: `bin/probe` → results/probe.txt

- **Families:** apple1–8, mac1/2, common1–3, metal3 and **metal4** are true; apple9–12 are false. The arch name is `applegpu_g14s`.
- **Metal 4 objects exist at runtime and can be created:** queue `AGXG14XFamilyCommandQueue_mtlnext`, command buffer and allocator. A full MTL4 dispatch benchmark ran (B3).
- **Compile results:**

  | Test | MSL version | Result |
  |---|---|---|
  | `#include <metal_tensor>` | 3.2 | Fails: `dextents` undeclared |
  | `#include <metal_tensor>` with `tensor_inline` | 4.0 | OK |
  | MPP `matmul2d` half×half→float, 64×32 tile, 4 simdgroups | 4.0 | Compiles, runs, **max abs error 0 vs CPU** (M=128..256) |
  | MPP `matmul2d` bfloat×bfloat→float | 4.0 | Exact; pipeline `maxTotalThreadsPerThreadgroup` drops to 768 |
  | MPP `matmul2d` **half×int4b_format→float** | 4.0 | **Exact.** Packing is **low nibble = even element**; the `tensor_inline` pointer type must be `device uchar*` |
  | `metal_fp8_e4m3_format` | 4.0 | Unknown type |
  | Any source | 4.1 | Rejected: "invalid value 'metal4.1'" |
  | `memory_order_release` | 4.0 | Undeclared |
  | Device-scope fence plus relaxed atomics | 3.2 | OK |

- **Counters:** only the `timestamp` counter set exists, and only `atStageBoundary` is supported. `recommendedMaxWorkingSetSize` is 11.84 GiB (74%); `maxBufferLength` is 8.88 GiB.

### B2. Bandwidth: `bin/bw` → results/bw.txt, results/bw_confirm.txt

**Spec:** 200 GB/s (LPDDR5-6400 × 256-bit = 204.8 GB/s raw).

| Test (1 GiB unless noted) | Best median | Notes |
|---|---|---|
| Read, scalar `float`, unroll 4, TG 512, grid 16384 | **196.3 GB/s** (interleaved recheck, 5 rounds: 195.7–196.7) | 98% of spec, 96% of raw |
| Read, `float4` (unroll 1–8), best config | 181–186 GB/s | Scalar 4-byte grid-stride loads beat 16-byte loads by about 5%, reproduced in interleaved rounds |
| Read, private vs shared storage | 181.3 vs 181.8 GB/s (float4) | **No difference** |
| Copy 512 MiB → 512 MiB (read + write counted) | 183.0 GB/s | Blit-encoder copy: 187.0 |
| Write | 187.3 GB/s | |
| Re-read working set of 1 / 4 / 8 / 16 / 24 / 32 / 64 / 256 MiB / 1 GiB | 313\* / 487 / 405 / 207 / 192 / 190 / 191 / 183 / 182 GB/s | \*1 MiB is dispatch-overhead bound. **Cache-resident only up to about 8 MiB; at 16 MiB it is effectively DRAM** |
| **4-bit GEMV** (MLX-style affine, group 64, fp16 scale+bias, 4.5 bits/weight), 25600×5120 (73.8 MB) | **186.3 GB/s** (396 µs) | 4 rows per simdgroup, 16 values per lane, 8 simdgroups per threadgroup |
| Same kernel, 5120×25600 (down projection) | 183.8 GB/s (402 µs) | |
| Same kernel, 131072×5120 (378 MB) | **191.4 GB/s** (1.98 ms) | **94–98% of the best streaming read** |

- **GEMV sensitivity:** 8 rows per simdgroup with 32 values per lane drops to 150–173 GB/s, from register pressure. 1 row per simdgroup with 32 values per lane gives 160–166. **Decode GEMV is fine with simple kernels as long as each lane keeps about 16 bytes in flight per row and handles 2–4 rows** [V, measured].

### B3. Dispatch overhead: `bin/dispatch` and `bin/mtl4` → results/dispatch.txt, results/mtl4.txt

The kernel is tiny: 1 threadgroup × 32 threads doing `buf[i] += 1`, so every dispatch depends on the previous one.

| Mode (Metal 3, Swift) | N=1000: GPU µs per dispatch | CPU encode µs per dispatch | Marginal GPU cost (N=100→1000 slope) |
|---|---|---|---|
| Serial encoder | 1.148 | 0.081 | **1.139 µs** |
| Serial + `memoryBarrier(scope:.buffers)` | 1.169 | 0.094 | 1.175 |
| Concurrent, no barrier (independent work) | **0.140** | 0.076 | **0.137** |
| Concurrent + `memoryBarrier(scope:)` (MLX pattern) | 1.248 | 0.233 | 1.248 |
| Concurrent + `memoryBarrier(resources:)` | 1.233 | 0.301 | 1.238 |
| One encoder per dispatch | 1.161 | 0.320 | 1.156 |
| **One command buffer per dispatch** | **13.9** | **5.74** | **14.1** |
| Untracked buffer, serial | 1.165 | 0.082 | 1.169 |
| "Small" kernel (64 threadgroups × 256 threads, ≈0.9 µs of work), serial vs concurrent | 1.88 vs 0.94 | 0.085 | ≈1 µs of dependency bubble on top of the kernel's own time |

| Path (C++, metal-cpp) | N=1000: GPU µs per dispatch | CPU encode µs per dispatch |
|---|---|---|
| **MTL4** encoder + `barrierAfterEncoderStages(dispatch→dispatch, .device)` | **1.184** | 0.194 |
| MTL4 encoder, no barriers | 0.139 | 0.070 |
| MTL3 serial encoder | 1.223 | 0.069 |
| MTL3 concurrent + memoryBarrier | 1.254 | 0.212 |

**Submission latency** (wall clock, median of 40):

| Measurement | Median |
|---|---|
| Empty command buffer, commit → `waitUntilCompleted` | 13.7 µs |
| One tiny dispatch, commit → completed (wait, spin on status, shared-event spin, or completion handler) | **106–125 µs** |
| MTL4 commit + `signalEvent` → CPU sees the event | 84.8 µs |
| **GPU idle gap when the CPU waits for CB *k* and then commits CB *k*+1** | **104.6 µs** (95–135) |
| Gap when both CBs are committed up front | **0.4 µs** |

Conclusions [V, measured on M2 Pro]:
1. A dependent kernel boundary costs ≈1.1–1.25 µs on the GPU whichever way it is expressed: serial, barrier or MTL4 barrier. Independent kernels overlap at ≈0.14 µs each.
2. **Metal 4 does not reduce GPU-side dispatch cost on M2 Pro**, and its barrier calls cost CPU time like Metal 3's. MTL4's advantages are structural: reusable command buffers, no per-CB retains, explicit residency, and one encoder for compute plus blit.
3. Never use one command buffer per kernel: 14 µs on the GPU and 5.7 µs of CPU each.
4. Keep the CPU out of the per-token critical path. A CPU round trip leaves the GPU idle ≈100 µs. Pre-commit the next token's work and do sampling on the GPU.

### B4. Indirect command buffer replay: `bin/icb` → results/icb.txt

| N dependent tiny dispatches | Direct serial: GPU / CPU µs | Direct concurrent + barrier | **ICB with setBarrier, replayed** | ICB without barriers |
|---|---|---|---|---|
| 10 | 17.4 / 4.5 | 18.2 / 6.7 | 20.2 / **4.4** | 8.9 / 4.4 |
| 100 | 141.6 / 12.7 | 162.5 / 29.5 | 142.5 / **4.0** | 25.3 / 3.9 |
| 1000 | 1399 / 72.0 | 1559 / 204 | 1383 / **3.8** | 193.8 / 3.6 |

- One-time CPU encoding of the ICB costs 0.163 µs per command.
- **Replay costs ≈4 µs of CPU for any N, but GPU time per dispatch does not improve**: 1.38 µs with barriers, 0.19 µs without.
- **GPU-encoded ICB** (an encode kernel writes `compute_command`s, then they are replayed in the same command buffer) is correct: the check buffer is 12×n as expected.
  - It costs 0.5 ms extra for 1000 commands, and **≈0.15 µs per reset (empty) slot**. With n=10 in a 1000-slot range, the GPU time was 161.6 µs.
  - Use the indirect-range variant so the GPU decides the count [L].

### B5. Matmul throughput: `bin/gemm` → results/gemm.txt, results/gemm_check.txt

The theoretical peak for a 16-core M2 Pro at 1.398 GHz is **5.73 TFLOPS** (FP16 = FP32 on Apple8).

| Kernel (4096³ unless noted) | TFLOPS | % of peak |
|---|---|---|
| `simdgroup_matrix` register-only, half×half→half | 5.44 | 95% |
| `simdgroup_matrix` register-only, half×half→float | 4.67 | 82% |
| **MPP `matmul2d` half×half→float**: tile 32×32 with 1 simdgroup, or 64×32 with 2 simdgroups | **5.42** | **95%** |
| MPP: 64×64 with 4 simdgroups / 128×64 with 4 / Apple's sample 64×32 with 4 / 128×128 with 4 | 5.38 / 5.04 / 4.46 / 4.11 | |
| **MPP half×int4b_format→float** (best tile) | **4.90** | 86% (≈0.9× FP16) |
| `MPSMatrixMultiplication` f16 | 4.41 | 77% |

- **Correctness:** the full 4096×4096 output was verified against the closed-form result for the 32×32/1, 64×64/4 and 64×32/4 tiles, for f16 and int4. Max relative error is 2.1e-5 (f16) and 0 (int4) [V].
- **Implication:** on pre-M5 hardware MPP is a **fast, portable GEMM** that beats MPS by 23% here and handles int4 operands. This contrasts with llama.cpp's finding that its tensor-API kernels are about 5% slower than its own on M2 Ultra; that difference is likely kernel design, not MPP. MPP is a credible single code path for prefill across M1–M5 [L].

### B6. Device-wide barrier inside one dispatch: `bin/persist` → results/persist.txt

| Spin variant (thread 0 of each threadgroup) | 16–64 threadgroups × 256, 1000 barriers | 48 threadgroups × 1024 | 64 threadgroups × 1024 |
|---|---|---|---|
| Relaxed `atomic_load`, plain counter | **Never completes** (all threadgroups time out) | Times out | Times out |
| Relaxed `atomic_load`, `coherent(device)` counter | **Never completes** | Times out | Times out |
| Relaxed `atomic_load` + device seq_cst fence inside the loop | **1.26 µs per barrier**, 0 violations | 1.43 µs | **48 timeouts (deadlock)** |
| RMW spin `atomic_fetch_add(ctr, 0)` | **1.11 µs per barrier**, 0 violations | 1.28 µs | **48 timeouts** |

---

## Hardware model (numbers for the roofline and the cost model)

### M2 Pro, 16-core GPU (measured here unless marked)

| Quantity | Value | Label |
|---|---|---|
| DRAM bandwidth, achievable read | 196 GB/s best, ~185 GB/s typical vector loads | [V, measured] |
| DRAM bandwidth, 4-bit GEMV effective | 183–191 GB/s | [V, measured] |
| Copy (read + write) | 183 GB/s | [V, measured] |
| Cache-resident re-read (≤ 8 MiB) | 400–490 GB/s | [V, measured] |
| FP16 / FP32 ALU peak | 5.73 TFLOPS (theory); 5.44 measured register-only | [V] |
| GEMM achievable | 5.42 TFLOPS (MPP f16), 4.90 (MPP int4) | [V, measured] |
| Ridge point | 5.42 TFLOPS ÷ 0.196 TB/s ≈ 28 FLOP/byte. For 4.5-bit weights, 2M FLOP per 0.5625 byte gives M ≈ 8 rows | [L, derived] |
| Dependent dispatch bubble | 1.1–1.25 µs | [V, measured] |
| Independent dispatch | 0.14 µs | [V, measured] |
| CPU encode | ≈0.07–0.09 µs per dispatch; +0.15 µs per explicit barrier | [V, measured] |
| ICB replay | ≈4 µs CPU per execute, any N | [V, measured] |
| CPU↔GPU round trip | 100–125 µs | [V, measured] |
| Device-wide barrier inside a kernel | 1.1–1.3 µs | [V, measured] |
| Co-residency | 48 × 1024 threads (3072 per core) | [V, measured] |

### M5 Max (official numbers plus estimates; nothing measured locally)

| Quantity | Value | Label |
|---|---|---|
| DRAM bandwidth (spec) | **614.4 GB/s (40-core)**, 460.8 GB/s (32-core) | [V, Apple] |
| Achievable read bandwidth | **≈520–585 GB/s (85–95%)**. M2 Pro reaches 96–98% here; base M5 STREAM copy was 79% (michaelstinkerings.org, single source). **Measure on day 1.** | [L] |
| Neural Accelerator FP16 matmul peak | **≈64–65.5 TFLOPS**: Anemll measured; tzakharko gives 1024 FLOP/clk/core × 40 × ~1.6 GHz | [L] |
| Neural Accelerator INT8×INT8 | **≈122 TOPS** | [L/E, one source] |
| FP8 / FP4 / MXFP4 | ≈1× / 0.8–0.9× / 0.5–0.8× of FP16 | [E] |
| Framework-achieved GEMM (MLX `mx.matmul`) | fp16 46.7, bf16 36.7, **q4 35.0 TFLOPS** | [E, MLX #4411] |
| ALU SIMD FP32 | ≈16.6 TFLOPS (40 × 128 × 2 × 1.62 GHz); `simdgroup_matrix` ≈14.6–15.5 | [L] |
| Clock under sustained load | Laptops can drop to 0.84–1.52 GHz depending on power mode and fans (FB23754032). Sustained pp512 fell 19% (llama.cpp PR #16634) | [E] |
| Ridge point (Neural Accelerator FP16) | 65 TFLOPS ÷ 0.6 TB/s ≈ 108 FLOP/byte, so **M ≈ 30 rows** for 4.5-bit weights (≈ 7.6 rows on the ALU path) | [L] |
| Per-dispatch bubble | Unknown. **Assume ≈1 µs** (M2 Pro 1.1–1.25 µs; higher clock). Measure. | [E] |
| Memory | 128 GB max (40-core). Default GPU working set ≈ 74% of RAM (≈95 GB at 128 GB if the M2 Pro ratio holds); raise with `iogpu.wired_limit_mb` | [L] |

**Qwen3-32B 4-bit decode, batch 1, on M5 Max 40-core** [L, derived]
- **Weights read per token:**
  - 32.8B parameters × 4.5 bits ≈ 18.5 GB.
  - Excluding the input embedding table (only one row is read) gives ≈18.0 GB. The LM head is untied, so it is read every token and is included.
  - Model shape [V: huggingface.co/Qwen/Qwen3-32B config.json]: 64 layers, 64 query and 8 KV heads, head_dim 128, hidden 5120, intermediate 25600, vocab 151936, `tie_word_embeddings: false`.
- **KV cache:**
  - 64 layers × 8 KV heads × 128 × 2 × 2 bytes = **256 KiB per token** of context.
  - That is +1.07 GB per step at 4K context and +8.6 GB at 32K.
- **Decode speed:**
  - Theoretical: 614.4 / 18.0 = **34 tok/s**.
  - At 90% efficiency with 4K context: ≈**29 tok/s**.
  - Dispatch overhead: 64 layers × ~10 dependent kernels × ~1 µs ≈ 0.6 ms per token, about 2% (with ~6 fused kernels per layer it is about 1%).
- **Prefill:**
  - Compute is ≈2 × 32.8 GFLOP = 65.6 GFLOP per token.
  - The ceiling is ≈1000 tok/s at 65 TFLOPS, or ≈530 tok/s at MLX's measured q4 rate of 35 TFLOPS.
  - TTFT for a 2K prompt is ≈2–4 s plus pipeline compile (cold 1–2 s unless cached).

### Per-dispatch cost model (fit to M2 Pro; re-fit on M5)

```
T_token ≈ Σ_kernels max(bytes_k / BW_eff, flops_k / F_eff(k))      # roofline per kernel
        + N_dep × t_dep (≈1.2 µs)  + N_indep × t_indep (≈0.14 µs)    # GPU-side dispatch bubbles
        + [CPU in loop] × t_rt (≈100 µs)                             # only if CPU waits for token k before submitting k+1
CPU_encode ≈ N × 0.08 µs (+0.15 µs per explicit barrier)  — or ≈4 µs per ICB replay; overlaps GPU if pipelined
```

### What Metal allows and does not allow

| Capability | Allowed? | Details |
|---|---|---|
| **Persistent / megakernel execution** | Technically yes; no guarantee | No forward-progress or residency guarantee [V]. The grid must not exceed co-resident capacity, which cannot be queried (measured 48 × 1024 threads on M2 Pro). Spin-waits need device-scope fences or RMWs [V, measured]. Acquire/release atomics only arrive in MSL 4.1 / OS 27. A display-active watchdog kills long GPU work [V]. A global barrier costs about as much as a kernel boundary [V, measured]. **Verdict:** use per-token command buffers and dataflow-level fusion, not run-forever kernels [L]. |
| **Command replay** | Yes | ICBs: no length limit, CPU or GPU encoded, ≈4 µs CPU per replay, GPU cost unchanged [V, measured]. MTL4 command buffers are reusable by re-encoding [V]. Indirect dispatch and indirect ICB ranges let the GPU pick sizes and counts [V]. |
| **Zero-copy weights** | Yes | `bytesNoCopy` over a page-aligned, single-VM-region `mmap(PROT_READ, MAP_SHARED)` of the weight file works (llama.cpp) [V]. Split into ≤ `maxBufferLength` views. Make them resident with residency sets, and raise `iogpu.wired_limit_mb` for very large models [V]. MTL4 requires explicit residency [V]. |
| **GPU-resident decode loop** | Partly | The GPU can encode ICB commands, choose dispatch sizes and counts, and run sampling. It **cannot submit new command buffers** [L]. Practical design: the CPU commits token *k*+1's pre-built command buffer (or ICB replay) before token *k* finishes. GPU-side argmax/sampling writes the next token id into a buffer the next command buffer reads. `MTLSharedEvent` signals the CPU asynchronously. This removes the ≈100 µs round-trip bubble (0.4 µs gap measured when pre-committed) [V, measured; design L]. |

---

## Recommendations for Sky [L]

1. **Two kernel tiers.**
   - Decode GEMV/GEMM with M ≤ ~8 (ALU) or M ≤ ~30 (Neural Accelerator ridge) uses plain MSL streaming kernels. Our simple kernel already hits 94–98% of the read roofline.
   - Prefill and batched or speculative verification use MPP `matmul2d` with tiles of at least 64×64 (Apple's 64×32 sample tile halves throughput on M5 Max).
   - MPP is a good path even on M1–M4 (95% of peak on M2 Pro).
2. **Weight format.**
   - Keep an affine group format for the decode kernels.
   - For MPP prefill on OS 26.4+, int4b with per-channel or per-group scales applied outside the matmul works.
   - OS 27 MXFP4 (E8M0 per 32) is the only *native* scaled format, and on M5 Max it is slower than FP16 for compute. Do not chase it for prefill. It is mainly a bandwidth play.
3. **Submission.**
   - One command buffer per token (or per N tokens), pre-built and replayed (MTL4 reuse or ICB).
   - GPU-side sampling.
   - Never put the CPU in the per-token dependency chain.
   - Fuse to fewer than 10 dependent kernels per layer.
4. **Do not build a persistent megakernel as the primary design.** If one is explored, it must size its grid from a calibrated residency probe, use fence or RMW spins, and stay well under watchdog-relevant durations.
5. **Use `objc2-metal`** with a runtime MTL4 check and a Metal 3 fallback. Compile MSL at runtime; it needs no Xcode. Cache binary archives to hide tensor-op compile time.
6. **First-day measurements on an M5 Max:**
   - achievable read and GEMV bandwidth
   - dependent-dispatch bubble
   - MPP fp16/int8/int4 GEMM TFLOPS by tile
   - sustained clocks under each power mode
   - energy (with R9)

## Open questions

- **M5 Max:** actual achievable bandwidth, SLC size, sustained neural-accelerator clocks in the 14-inch vs 16-inch laptops vs Mac Studio, and NUMA effects on the quad-die M5 Ultra.
- **Neural-accelerator throughput** for int4b operands (OS 26.4) and BF16. We have no clean microbenchmark.
- **Dispatch bubble on Apple9/10:** does second-generation dynamic caching change the ~1.2 µs figure?
- **Watchdog:** the actual time limit, and whether it applies with no display attached (headless Mac Studio).
- **MTL4 commit latency** vs MTL3 on M5. On M2 Pro, MTL4 event-observed completion was 85 µs vs about 106 µs for MTL3.
- **Core AI:** whether its LLM path (KV "states") outperforms MLX on M5 Max for decode and prefill, and whether its custom-kernel hooks are usable from a non-Python toolchain.

---

## Architect brief (R6: Apple silicon + Metal 4)

**Notes:** `$S/research/R6-apple-metal.md`. **Benchmarks:** `$S/research/microbench/` (README, sources, `results/`). **Sources:** `$S/research/R6-work/`.

### Findings

- **M5 Max bandwidth depends on the bin** [V, Apple specs]:
  - 614 GB/s only on the 40-core GPU part (48, 64 or 128 GB).
  - The 32-core M5 Max is 460 GB/s and comes with 36 GB only.
  - M5 Pro is 307 GB/s.
  - M5 Ultra (shipped 2026-09-22) is 1.2 TB/s on both bins, up to 512 GB.
  - The Mac Pro is discontinued.
  - Apple publishes no GPU clock or TFLOPS for M5.
- **The GPU Neural Accelerators are one per GPU core.** Apple: "designed to accelerate dense compute-bound work such as the prefill stage" (WWDC26-330).
  - Reachable **only via Metal 4 TensorOps (MPP `matmul2d`/`convolution2d`)** or frameworks built on them.
  - Apple's tech talk shows plain `simdgroup_matrix` GEMM at "neural accelerator utilization is 0%" [V].
  - MPP lowers to backend-defined `air.externally_defined` functions, so its codegen is opaque [V].
- **Measured M5 Max ceilings** [L; Anemll, consistent with tzakharko's A19 per-core rate]:
  - ≈64–65 TFLOPS FP16 and ≈122 TOPS INT8×INT8.
  - FP8 runs at the FP16 rate; FP4 and MXFP4 are slower.
  - MLX reaches 46.7 TFLOPS fp16 and 35 TFLOPS q4 [E].
  - Prefill and TTFT speed up 3.3–4× vs M4. **Decode only tracks bandwidth (+19–27%)** [V, Apple ML Research].
- **WWDC26 / OS 27:**
  - MSL 4.1 brings fp8/fp4/int2 MPP operands, `tensor_blockwise` with **only E8M0 scales per 32 elements (MX-style)**, multiplane tensors, and acquire/release atomics.
  - Int4/UInt4 tensors arrived in OS 26.4.
  - **Core AI** is a new on-device inference framework (CPU/GPU/ANE, PyTorch conversion, custom Metal kernels, OS 27).
  - Metal 4 core: unretained, reusable `MTL4CommandBuffer`; `MTL4CommandAllocator`; argument tables (GPU addresses); mandatory residency sets; **all resources untracked**; one compute encoder for compute and blit; ICBs retained.

### Local numbers (M2 Pro 16-core, macOS 26.6.1, all through the bench lock)

- **Metal 4 tensors and MPP compile and run correctly on this Apple8 GPU** with MSL 4.0 runtime compilation, for half, bfloat and **int4b_format** operands (low nibble first; pointer type `device uchar*`).
  - MPP half GEMM 4096³ reaches **5.42 TFLOPS, 95% of the 5.73 TFLOPS peak**. MPS reaches 4.41. MPP int4 reaches 4.90. All outputs were verified.
  - MSL 4.1 and fp8 are rejected (OS 27 needed).
- **Bandwidth:**
  - Read **196 GB/s (98% of 200)**; copy 183; write 187.
  - Shared = private.
  - Caching helps only for working sets ≤ 8 MiB.
  - A simple 4-bit affine GEMV gets **183–191 GB/s (94–98% of read)**.
- **Dispatch:**
  - A dependent kernel boundary costs **1.1–1.25 µs** of GPU time, whether serial, barrier or **MTL4** intra-pass barrier.
  - Independent dispatches cost 0.14 µs.
  - One command buffer per kernel costs 14 µs, so never do it.
  - CPU encode is about 0.08 µs per dispatch.
  - A CPU→GPU→CPU round trip is **100–125 µs**: the GPU sits idle about 105 µs if the CPU waits before committing the next command buffer, versus 0.4 µs when pre-committed.
  - Metal 4 is not faster than Metal 3 here.
- **ICB replay:**
  - CPU cost is ≈4 µs regardless of N (vs 72 µs for 1000 direct dispatches); GPU per-dispatch cost is unchanged.
  - GPU-encoded ICBs work, at ≈0.15 µs per empty slot.

### Negative results

- **No forward-progress guarantee:** the MSL 4.1 spec never mentions it.
- **Co-residency:** 48 × 1024-thread threadgroups were co-resident; 64 deadlocked a grid barrier (caught by a bounded spin).
- **A relaxed `atomic_load` spin never sees other threadgroups' updates**, even on a `coherent(device)` counter. A device fence inside the loop, or an RMW spin, is required.
- **A working global barrier costs 1.1–1.3 µs, no cheaper than a kernel boundary.** Persistent megakernels buy nothing for global syncs.
- **macOS kills long-running GPU work while a display is active** (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`):
  - llama.cpp sets the undocumented `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (`ggml-metal.cpp:1027-1029`).
  - An M5 Max report shows kernel panics under sustained load with the display on.
- **Neural Accelerators do not help batch-1 decode:**
  - A 1-row `matmul2d` is slower than a GEMV.
  - llama.cpp and MLX route M < 8–33 to ALU kernels.
- **Data-format limits:**
  - No native format exists for fp16 group scales or biases (MLX/GGUF style). The only native scaled format is E8M0-per-32 (MX), and on M5 Max it is slower than FP16 for compute.
  - Per-dispatch GPU timestamps are not available on Apple GPUs; only the stage-boundary counter is.
  - Offline `metal` compilation and `.gputrace` viewing need Xcode.

### Hardware model for the roofline

- **M2 Pro:** 190 GB/s effective, 5.4 TFLOPS; ridge ≈ 8 rows for 4.5-bit weights; t_dep ≈ 1.2 µs, t_indep ≈ 0.14 µs.
- **M5 Max (40-core):**
  - 614 GB/s spec; plan for 520–585 GB/s effective [L].
  - Neural Accelerator FP16 ≈ 65 TFLOPS and INT8 ≈ 122 TOPS [L]; ALU ≈ 16.6 TFLOPS [L].
  - Ridge ≈ 30 rows on the Neural Accelerator, ≈ 8 rows on the ALU.
  - t_dep ≈ 1 µs [E].
- **Qwen3-32B 4-bit decode:** about 18 GB of weights per token plus 256 KiB of KV per context token, so **≈34 tok/s ceiling and ≈29 tok/s at 90% efficiency with 4K context**. Dispatch overhead is about 1–2% with ≤ 10 kernels per layer.
- **Prefill ceiling** ≈1000 tok/s at peak, ≈530 tok/s at MLX's q4 rate.
- Thermals can cost about 20% sustained on laptops.

### Recommendations

1. **Decode:** hand-written streaming GEMV kernels. Simple kernels already reach the roofline, so beating MLX and llama.cpp will come from fusion, fewer kernels and KV handling, not GEMV heroics.
2. **Prefill and verification:** MPP `matmul2d` with tiles of at least 64×64, on every chip. It is portable from M1 to M5 and already beats MPS on M2.
3. **Submission:** pre-built, reused command buffers or ICBs per token; GPU-side sampling; the CPU never in the token chain; residency sets plus mmap and `bytesNoCopy` weights.
4. **Persistence:** no run-forever megakernel.
5. **Rust:** `objc2-metal` 0.3.2 (has MTL4 and `MTLTensor`; patch in the 26.4/27 enums), runtime MSL compilation, binary-archive caching to hide the ≈1–2 s tensor-op compile.
6. **On M5 Max hardware, measure first:** bandwidth, t_dep, MPP TFLOPS by tile and dtype, sustained clocks, energy.

### Open questions

- Real M5 Max achievable bandwidth and SLC size.
- Neural Accelerator rates for int4b and bf16.
- Watchdog limits with no display attached.
- Whether Core AI's LLM path is a serious competitor.
- The dispatch bubble on Apple10.
