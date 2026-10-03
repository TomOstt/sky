# Part XI — The Metal backend

This part specifies `sky-metal`, the L4 backend of the tensor stack ([Part VI](part06-architecture.md) §4). It covers how Sky drives Metal from Rust, how it holds memory, how it turns kernel IR into Metal Shading Language (MSL), how pipelines are compiled and reused, the kernel skeletons in MSL, capability gating, timing, and how kernels are tested.

It does not decide which kernels exist or how they are fused (scheduler, [Part IX](part09-scheduler.md)), their parameters (autotuner, [Part X](part10-autotuner.md)), weight formats ([Part XII](part12-quantization.md)), the attention algorithm ([Part XV](part15-attention.md)) or the decode loop ([Part XX](part20-persistent-execution.md)). It provides the mechanisms those parts use.

Evidence labels follow the report: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R6 §B3) and ship as appendices: [R2](research/R2-tinygrad.md), [R3](research/R3-mlx-core.md) and [R3-kernels](research/R3-kernels.md), [R4](research/R4-mlx-lm.md), [R5](research/R5-llamacpp.md), [R6](research/R6-apple-metal.md) (the primary source for this part), [R7](research/R7-compilers.md), [R8](research/R8-algorithms.md), [R9](research/R9-formats-methodology.md). The Metal microbenchmarks we ran ship in [`research/microbench/`](research/microbench/), with raw outputs in [`research/microbench/results/`](research/microbench/results/). Unless stated otherwise, "measured" means measured by R6 on an Apple M2 Pro (16-core GPU, Apple8 family) under macOS 26.6.1 (build 25G76).

## 1. Decisions

| # | Decision | Evidence |
|---|---|---|
| 1 | Metal is driven from Rust through **`objc2-metal` 0.3.2**. The OS 26.4/27 additions it lacks are reached through raw enum values and `msg_send!` shims in one module. | It already exposes `MTL4*` and `MTLTensor`; `metal-rs` is deprecated (R6 §A5) `[VERIFIED]` |
| 2 | **Two command paths behind one interface.** The Metal 3 path (`MTLCommandQueue`, serial compute encoder) is the default through G4. The Metal 4 path (`MTL4CommandQueue`, argument tables, intra-pass barriers) sits behind a runtime check. The default changes only if the M5 Max measurements of §15 favour Metal 4. | Metal 4 is not cheaper per dispatch on the GPU (1.184 vs 1.223 µs) or on the CPU (0.194 vs 0.212 µs) (R6 §B3) `[VERIFIED]` |
| 3 | **One kernel ABI.** Every kernel takes one `constant` argument struct holding GPU addresses and scalars. The same execution plan then encodes to Metal 3, Metal 4 and indirect command buffers (ICBs) without change. | tinygrad's Metal runtime does the same, which is what lets an ICB command bind one buffer (R2 §7.2, §7.5) `[VERIFIED]` |
| 4 | **All memory is `shared`, hazard-untracked**, planned statically into arenas. Weights are mmapped from a page-aligned `.skyweights` file into no-copy buffers, split into views no larger than `maxBufferLength`, kept resident by a residency set plus a heartbeat. | Shared and private storage stream at the same rate (181.8 vs 181.3 GB/s); mlx-community files are unaligned and silently misread (R6 §B2, R9 §1.3) `[VERIFIED]` |
| 5 | **MSL text is generated from kernel IR and compiled at run time.** There are three language tiers: MSL 3.2 (macOS 15), 4.0 (macOS 26) and 4.1 (macOS 27). Tensor operations are emitted only at 4.0 or later. | Runtime compilation needs no Xcode; MSL 4.1 is rejected on macOS 26.6.1 (R6 §B1) `[VERIFIED]` |
| 6 | **Caches are data only (SPINE D19).** The plan cache stores validated data and never MSL or binaries. On load, MSL is regenerated deterministically and the OS shader cache supplies compiled code. Metal binary archives appear only in explicit `sky compile` outputs (`.skybin`). | llama.cpp's warm starts take 15–50 ms through the OS shader cache (R5 §1.6) `[VERIFIED]` |
| 7 | **Kernels for the `inference` contract are compiled in `safe` math mode with contraction off and explicit `fma`.** The reduction order is then written in the source, which is what makes batch invariance something the emitter can guarantee and a test can check. | The default MSL math mode is `fast`, which allows reassociation; `safe` still allows contraction within a statement (MSL 4.1 spec §1.6.3, pp.15–17) `[VERIFIED]` |
| 8 | **Skeletons:** a decode GEMV for MLX affine q4 g64; a verify GEMV family for M ≤ 16 that shares the decode GEMV's reduction schedule; an MPP `matmul2d` GEMM that dequantizes group-scaled tiles into threadgroup memory (all chips, macOS 26+); a `simdgroup_matrix` GEMM as the fallback; split-K decode attention with a fixed split size. | §9 |
| 9 | **One kernel family per plan for all verify widths.** MPP's internal reduction order is opaque, so the ALU GEMV and MPP cannot agree bit for bit. If the M5 plan verifies on the neural accelerators, plain decode must run the same MPP family at M = 1. | MPP lowers to backend-provided `air.externally_defined` functions (R6 §A2) `[VERIFIED]`; R8 §3.5 rule 1 |
| 10 | **Timing.** Per command buffer: `gpuStartTime`/`gpuEndTime` or `MTL4CommitFeedback`. Per kernel: a profiling mode with one compute encoder per kernel and stage-boundary timestamps, because Apple GPUs expose no per-dispatch counters. | Only the `timestamp` counter set exists, and only `atStageBoundary` (R6 §B1) `[VERIFIED]` |
| 11 | **Tests:** comparison against the TIR reference interpreter; shape fuzzing; poisoned outputs and canary guards; bitwise invariance tests (run-to-run, M, chunk); real-weight goldens; API validation in CI; a compiler-regression self-check whenever the OS build or the GPU compiler changes. | §12 |

## 2. Where the backend sits

```text
 sky-kernel (L3)                                   sky-metal (L4)
 ───────────────                                   ──────────────
 KernelIr { skeleton, params,       ─── emit ───►  msl::Emitter ──► MslUnit { source, entry, fn_consts,
            prologue, epilogue,                                                min_lang, schedule_hash }
            args, reduction schedule }                                   │
                                                                         ▼
 sky-tir plan (data) ───────────────────────────►  compile::Compiler (language gating, options, parallel)
   dispatch list, barriers, arena offsets                                │   OS shader cache (implicit)
                                                                         ▼   .skybin archive (explicit only)
                                                   pipeline::Pipelines ──► encode::{Mtl3, Mtl4, Icb}
                                                                         │
 sky-llm runtime (Part XIII, XX) ──── uses ──────► device::Gpu, buffer::{Arena, WeightMap},
                                                   residency::Keeper, timing::Profiler, caps::GpuCaps
```

Module layout of the new crate:

| File | Contents |
|---|---|
| `crates/sky-metal/src/device.rs` | `Gpu::open()`: device, queues, `GpuCaps` |
| `crates/sky-metal/src/caps.rs` | family, OS and neural-accelerator detection; compile probes; `GpuCaps` → `sky_hw::GpuInfo` |
| `crates/sky-metal/src/raw.rs` | raw enum values for OS 26.4/27; `msg_send!` shims for selectors `objc2-metal` 0.3.2 lacks |
| `crates/sky-metal/src/buffer.rs` | `Arena`, `BufferView`, `WeightMap` (no-copy mmap, view splitting) |
| `crates/sky-metal/src/residency.rs` | `Keeper`: residency sets, heartbeat thread, wired-limit check |
| `crates/sky-metal/src/msl/` | `Emitter`, the prelude, skeleton templates (`gemv_q4.metal.in`, …), golden snapshots |
| `crates/sky-metal/src/compile.rs` | language gating, `MTLCompileOptions`, parallel compilation |
| `crates/sky-metal/src/pipeline.rs` | `PipelineKey`, `Pipelines`, `.skybin` archives, self-check digests |
| `crates/sky-metal/src/encode/{mtl3,mtl4,icb}.rs` | the three encoders of one `DispatchList` ([Part XX](part20-persistent-execution.md) §4) |
| `crates/sky-metal/src/timing.rs` | commit feedback, counter sampling, `sky profile` modes |
| `crates/sky-metal/src/debug.rs` | validation switches, poison, canaries, labels and debug groups |
| `crates/sky-metal/tests/` | kernel tests; they need a GPU and run on the self-hosted macOS runner (VI-T2) |

## 3. Metal from Rust

### 3.1 Binding choice

| Option | For | Against |
|---|---|---|
| **`objc2-metal` 0.3.2 (chosen)** | Generated from the macOS 26 SDK: `MTL4CommandQueue`, `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable`, `MTL4Compiler`, `MTLTensor` behind per-header Cargo features; maintained; pure Rust (R6 §A5) | Lacks the OS 26.4/27 additions: Int4 tensor types, tensor planes, `MTLLanguageVersion` 4.1 (R6 §A5) |
| `metal` (metal-rs) 0.33 | Familiar | "Use of this crate is deprecated" (its README; R6 §A5) |
| Apple's metal-cpp behind a C++ shim | Complete through macOS 27; R6 used it for `mtl4.cpp` | A C++ build step and an FFI layer to maintain |
| Swift or Objective-C shim | Full SDK | A second language in the build |

`objc2-metal` method names are the Objective-C selectors with colons replaced by underscores: `newBufferWithLength_options`, `newLibraryWithSource_options_error` (returns `Result<_, Retained<NSError>>`), `newResidencySetWithDescriptor_error`, `newMTL4CommandQueue`, `newCommandAllocator`, `queryTimestampFrequency`. `newBufferWithBytesNoCopy_length_options_deallocator` and `newIndirectCommandBufferWithDescriptor_maxCommandCount_options` are `unsafe` `[VERIFIED: docs.rs objc2-metal 0.3.2, MTLDevice trait]`. Encoder selectors follow the same rule, for example `dispatchThreadgroups_threadsPerThreadgroup`, `barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions` and `executeCommandsInBuffer_withRange` `[LIKELY: the selector list in metal-cpp's MTLHeaderBridge.hpp plus objc2's naming rule]`. The sketches below wrap encoder calls in `unsafe` blocks; the precise unsafety of each binding is checked when XI-T1 is written.

### 3.2 Device and queue setup

```rust
// crates/sky-metal/src/device.rs (sketch)
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue3: Retained<ProtocolObject<dyn MTLCommandQueue>>,          // default path
    pub queue4: Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>>, // macOS 26+, Apple7+
    pub caps: GpuCaps,
}

impl Gpu {
    pub fn open(opts: &GpuOptions) -> Result<Gpu, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
        let caps = caps::probe(&device)?;                       // §4
        let queue3 = device.newCommandQueue().ok_or(GpuError::Queue)?;
        let queue4 = if caps.metal4 && opts.allow_mtl4 { device.newMTL4CommandQueue() } else { None };
        Ok(Gpu { device, queue3, queue4, caps })
    }
}
```

Command buffers on the Metal 3 path are created with `commandBufferWithUnretainedReferences`, as MLX and llama.cpp do (R3 §2.1, R5 §1.1). The plan owns every buffer for the life of a session, so retaining them per command buffer buys nothing.

### 3.3 Raw values for OS 26.4 and 27

`objc2` enums are newtypes over the integer, so values the crate does not name can still be constructed (R6 §A5). `raw.rs` holds the values Sky needs, each tied to its source:

| Type | Name | Raw value | Available from | Source |
|---|---|---|---|---|
| `MTLGPUFamily` | Apple7 / 8 / 9 / 10 | 1007 / 1008 / 1009 / 1010 | — | metal-cpp `MTLDevice.hpp` (macOS 26) |
| `MTLGPUFamily` | Metal3 / Metal4 | 5001 / 5002 | Metal 4: macOS 26 | same; probe (R6 §B1) |
| `MTLLanguageVersion` | 3.2 / 4.0 / 4.1 | 196610 (0x30002) / 262144 (0x40000) / 262145 (0x40001) | 4.1: macOS 27 | metal-cpp `MTLLibrary.hpp`, 26 → 27 diff |
| `MTLTensorDataType` | Int4 / UInt4 | 143 / 144 | macOS 26.4 | metal-cpp 26 → 26.4 diff (R6 §A3) |
| `MTLTensorDataType` | MetalFloat8E5M2 / E4M3 / UE8M0 / MetalFloat4E2M1 / UInt2 / Int2 | 141 / 142 / 145 / 148 / 149 / 150 | macOS 27 | R6 §A3 |
| `MTLFloatingPointConversionRoundingMode` (on `MTLCompileOptions`) | ToNearestEven / TowardZero | 0 / 1 | macOS 27 | metal-cpp 26 → 27 diff of `MTLLibrary.hpp` |
| `MTLStages` | Dispatch / Blit / MachineLearning | 1<<27 / 1<<28 / 1<<30 | macOS 26 | metal-cpp `MTLCommandEncoder.hpp` |
| `MTL4VisibilityOptions` | None / Device / ResourceAlias | 0 / 1 / 2 | macOS 26 | metal-cpp `MTL4CommandEncoder.hpp` |
| `MTL4CounterHeapType`, `MTL4TimestampGranularity` | Timestamp; Relaxed / Precise | 1; 0 / 1 | macOS 26 | metal-cpp `MTL4Counters.hpp` |
| `MTLResourceOptions` | StorageModeShared / Private / HazardTrackingModeUntracked | 0x0 / 0x20 / 0x100 | — | metal-cpp `MTLResource.hpp` |

A selector the crate does not bind is called through a guarded `msg_send!`:

```rust
// crates/sky-metal/src/raw.rs (sketch)
use objc2::{msg_send, sel, runtime::AnyObject};

pub const LANG_3_2: usize = 0x30002;
pub const LANG_4_0: usize = 0x40000;
pub const LANG_4_1: usize = 0x40001;       // macOS 27
pub const TENSOR_INT4: isize = 143;        // macOS 26.4
pub const TENSOR_UINT4: isize = 144;

/// macOS 27: pin float-to-float conversions to round-to-nearest-even explicitly.
/// Returns false on older systems, where RTNE is the only behavior.
pub fn set_conversion_rounding_rtne(opts: &AnyObject) -> bool {
    let s = sel!(setFloatingPointConversionRoundingMode:);
    let ok: bool = unsafe { msg_send![opts, respondsToSelector: s] };
    if ok {
        let _: () = unsafe { msg_send![opts, setFloatingPointConversionRoundingMode: 0isize] };
    }
    ok
}
```

When `objc2-metal` regenerates from a newer SDK, the shims are deleted and a test (`raw_values_match_crate`) compares every raw constant with the crate's named constant.

### 3.4 Metal 3 and Metal 4

| | Metal 3 path | Metal 4 path | Measured (R6 §B3, §B4) |
|---|---|---|---|
| Queue / command buffer | `MTLCommandQueue`; one-shot command buffers | `MTL4CommandQueue`; device-made `MTL4CommandBuffer`, reusable after `beginCommandBufferWithAllocator`; memory in `MTL4CommandAllocator`, reset after completion | — |
| Bindings | per dispatch `setBuffer` / `setBytes` | `MTL4ArgumentTable` holding GPU addresses (`setAddress_atIndex`), set on the encoder | — |
| Ordering between dependent dispatches | serial encoder | `barrierAfterEncoderStages(.dispatch, .dispatch, .device)` | GPU 1.223 vs 1.184 µs per dependent dispatch; CPU 0.069 vs 0.194 µs |
| Independent dispatches | concurrent encoder, no barrier | no barrier | 0.140 vs 0.139 µs GPU |
| Residency | `useResource` or residency sets (macOS 15) | residency sets only; no `useResource` | — |
| Hazard tracking | optional per resource | none: "the framework considers all resources untracked" | — |
| Submission | `commit` | `commit_count` of an array; `signalEvent_value` between commits | commit + event seen by CPU: 106 µs (Metal 3) vs 84.8 µs (Metal 4) for a one-dispatch buffer |
| Compute + blit | separate encoders | one `MTL4ComputeCommandEncoder` | — |
| ICB execution | `executeCommandsInBuffer` (range, or indirect range buffer) | same (range, or indirect range by GPU address) | replay ≈ 4 µs CPU for any length |
| Timing | `gpuStartTime`/`gpuEndTime`; stage-boundary counters | `MTL4CommitFeedback`; `MTL4CounterHeap` + `writeTimestamp` | — |

Metal 4 brings structure, not speed: reusable command buffers and allocators, argument tables, explicit residency and one encoder for compute and blit (R6 §B3, conclusion 2) `[VERIFIED]`. Its event-to-CPU latency was about 20 µs lower in one measurement, which matters only if the CPU waits on the GPU, and the design of [Part XX](part20-persistent-execution.md) avoids that. So the Metal 3 path ships first: every production engine on Apple uses it (R3 §2.1, R5 §1.1), and our kernel microbenchmarks ran through it, including the MSL 4.0 MPP kernels (the command API and the language version are independent; R6 [`gemm.swift`](research/microbench/gemm.swift)). The Metal 4 path is built at G5 for the persistent runtime and becomes the default only by measurement (§15, item 5).

### 3.5 Synchronization rules

Buffers are created with `MTLResourceHazardTrackingModeUntracked`, as MLX does (R3 §2.3); untracked and tracked buffers have the same GPU cost per dispatch (1.169 vs 1.139 µs) (R6 §B3). The runtime is therefore responsible for every ordering:

| Scope | Metal 3 path | Metal 4 path |
|---|---|---|
| Dispatch after a dependent dispatch, same step | serial compute encoder (dispatches run in order) | `barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(Dispatch, Dispatch, Device)` from the plan's barrier list |
| Independent dispatches | concurrent encoder only where the plan proves independence; otherwise serial | no barrier |
| Across encoders or command buffers | `MTLFence`: the producer encoder `updateFence`, the consumer `waitForFence` (MLX's scheme, R3 §2.1; tinygrad's, R2 §7.4) | queue barrier `barrierAfterQueueStages(Dispatch, Dispatch, Device)` at the start of the consumer's encoder |
| GPU → CPU | `MTLSharedEvent` signal; CPU reads results only after observing the value | same (`signalEvent_value` on the queue) |
| CPU → GPU | write shared memory before `commit` | same |
| Inside an ICB | `setBarrier` on the command | same; note that `executeCommands` itself is in no stage, so a barrier cannot wait on "the ICB" as a unit; each command keeps its own stages (Apple, `MTL4ComputeCommandEncoder` "Command stages") |

The measured bubble for one dependent kernel boundary is 1.1–1.25 µs whichever way it is expressed; independent dispatches cost 0.14 µs (R6 §B3) `[VERIFIED]`. Barriers are therefore a cost to minimize ([Part IX](part09-scheduler.md)), and the encoders above only realize the plan's barrier list.

### 3.6 The kernel ABI: one argument struct

Every generated kernel has this signature shape:

```metal
struct GemvArgs {                       // emitted per kernel from the KernelIr argument list
  device const uint*      w;            // GPU addresses written by the host (u64)
  device const ushort*    s;            // bf16 bits
  device const ushort*    b;
  device const half*      x;
  device const ushort*    gamma;
  device       half*      y;
  device const half*      resid;
  device const StepState* st;           // GPU-written step scalars (Part XX §3)
  float eps;                            // plan constants
};
[[kernel]] void k(constant GemvArgs& a [[buffer(0)]], /* thread position attributes */) { ... }
```

The host lays out all argument structs of a plan in one shared **argument arena**, at 256-byte-aligned offsets (tinygrad uses the same alignment for its single argument buffer, R2 §7.5). Pointers are `MTLBuffer.gpuAddress() + offset`. A dispatch is then four things: pipeline, argument offset, grid, threadgroup size, plus a barrier bit:

| Encoder | How a dispatch binds its arguments |
|---|---|
| Metal 3 | `setBuffer_offset_atIndex(arena, off, 0)` |
| Metal 4 | `argument_table.setAddress_atIndex(arena.gpuAddress() + off, 0)`, then dispatch. Whether the table is snapshotted per dispatch at encode time is `[EXPERIMENTAL: XI-T6 checks it with a 1000-dispatch chain whose dispatches each bind a different offset]`; if not, one table per dispatch is used |
| ICB | `setKernelBuffer_offset_atIndex(arena, off, 0)` once, when the ICB is built |

Scalars that change per token (position, token id, RNG counter, verify width, stop flag) are not in the argument structs. They live in `StepState`, which kernels reach through `a.st` and which the GPU itself updates ([Part XX](part20-persistent-execution.md) §3). Argument structs are therefore immutable for the life of a plan. `setBytes` (≤ 4 KB, R6 §A4) is not used on the hot path, because ICB commands cannot use it.

Pointers in argument buffers need argument-buffer tier 2; the M2 Pro reports tier 2 (`argumentBuffersSupport=1`, R6 [`results/probe.txt`](research/microbench/results/probe.txt)) `[VERIFIED]`. Every buffer referenced only through such pointers must be made resident through a residency set (§6.5).

## 4. Capability detection and gating

### 4.1 What is probed

`caps::probe` runs once per process (about a millisecond for queries; compile probes are cached as data per OS build) and fills:

```rust
// crates/sky-hw/src/gpu.rs: plain data, no Metal dependency (sky-hw stays free of unsafe code, R1 §5.7)
pub struct GpuInfo {
    pub name: String,                 // "Apple M2 Pro"
    pub arch: String,                 // MTLDevice.architecture.name: "applegpu_g14s"
    pub generation: Option<u32>,      // 14 from "g14s" (MLX's parse, R3-kernels §0)
    pub size_class: Option<char>,     // 's'
    pub apple_family: Option<u32>,    // highest n with supportsFamily(1000 + n): 7..10
    pub metal4: bool,                 // supportsFamily(5002)
    pub gpu_cores: Option<u32>,       // IORegistry "gpu-core-count" [LIKELY], else None
    pub neural_accelerators: NaState, // Absent | Present | PresentUnverified
    pub os_version: (u32, u32, u32),  // 26.6.1
    pub os_build: String,             // kern.osversion: "25G76"
    pub gpu_compiler: Option<String>, // "32023.886", see §8.3
    pub max_buffer_length: u64,       // 8.88 GiB on the 16 GB M2 Pro
    pub recommended_working_set: u64, // 11.84 GiB (74% of RAM) on the M2 Pro
    pub max_threadgroup_memory: u32,  // 32 KiB, Apple4..Apple10
    pub simd_width: u32,              // 32
    pub counters: CounterSupport,     // stage boundary only, on the M2 Pro
    pub measured: Option<Calibration>,// bandwidth, t_dep, MPP TFLOPS by tile: from `sky hw --gpu --calibrate`
}
```

The probe values on our M2 Pro are in R6 [`results/probe.txt`](research/microbench/results/probe.txt) `[VERIFIED]`. The Apple GPU family answers coarse questions; the architecture string separates chips within a family (M5 Max is reported as `applegpu_g17s` by third parties and by an MLX comment, R3-kernels §0 `[LIKELY]`).

**Neural accelerators.** No API reports them. Sky sets `Present` when `supportsFamily(Apple10)` holds and the generation is at least 17, which matches the M5 family in Apple's feature-set tables (R6 §A1) and MLX's gate (`gen >= 17`, `mlx/backend/metal/device.cpp:952-971`, R3-kernels §d). `sky hw --gpu --calibrate` then confirms the state by measurement: on a chip with accelerators, MPP fp16 GEMM throughput exceeds the `simdgroup_matrix` peak several times over (≈64–65 vs ≈14.6–15.5 TFLOPS on M5 Max, R6 §A2 `[LIKELY]`). Until calibrated, the state is `PresentUnverified` and the autotuner treats both GEMM paths as candidates.

**Compile probes.** As llama.cpp does with a dummy kernel (`ggml-metal-device.m:1164-1260`, R5 §3), Sky compiles one tiny kernel per feature tier once per (OS build, device) and records the result as data: MSL 4.0 tensors, MPP `matmul2d` for half, bfloat and `int4b_format`, and MSL 4.1. llama.cpp's history shows why this is needed: its tensor path was silently disabled on some M5 builds until 2026-09-01 (#27461, R5 §9.5) `[VERIFIED]`.

### 4.2 Gates

| Feature | Requirement | Gate in `caps.rs` | Fallback |
|---|---|---|---|
| Residency sets, MSL 3.2, device-scope fences | macOS 15 | OS version | none: macOS 15 is Sky's minimum for the tensor stack |
| Metal 4 command path | macOS 26, Apple7+ | `supportsFamily(5002)` and `respondsToSelector(newMTL4CommandQueue)` | Metal 3 path |
| MSL 4.0 tensors, MPP `matmul2d` half/float/char | macOS 26.0 | compile probe | `simdgroup_matrix` GEMM |
| MPP bfloat operands | macOS 26.1 | compile probe | convert to half, or `simdgroup_matrix` |
| Cooperative-tensor inputs to `matmul2d` | macOS 26.1 per the spec (Table 7.5) | compile probe | threadgroup-tile variant |
| MPP `int4b`/`uint4b` operands, `MTLTensorDataType` 143/144 | macOS 26.4 | compile probe | dequantized tiles (the default anyway, §9.4) |
| MSL 4.1: fp8/fp4/int2, `tensor_blockwise`, acquire/release atomics | macOS 27 | compile probe | not used before year 2 |
| Neural-accelerator code paths | M5 family | `neural_accelerators != Absent` and calibration | ALU paths |

On macOS 26.6.1 the MSL 4.1 probe fails with "invalid value 'metal4.1' in '-std=metal4.1'" and `memory_order_release` is undeclared (R6 §B1) `[VERIFIED]`. The gates are evaluated once and stored in `GpuCaps`; no kernel checks the OS at run time.

## 5. Data types by chip and OS

| Feature | OS / MSL | M1 (Apple7) | M2 (Apple8) | M3, M4 (Apple9) | M5 (Apple10) | Source |
|---|---|---|---|---|---|---|
| `half` arithmetic | all | same rate as fp32 | same rate as fp32 | fp16/fp32/int co-issue, up to 2× on mixed code | as Apple9 `[LIKELY]` | R6 §A1 |
| `bfloat` type | MSL 3.1 | yes | yes | yes | yes | llama.cpp gates bf16 on Metal3/Apple6 (R5 §3) |
| bf16 → f32 by bit shift (no `bfloat` needed) | all | yes | yes | yes | yes | arithmetic identity |
| `simdgroup_matrix` 8×8 half/float | MSL 2.3+ | ALU | ALU: 5.44 TFLOPS (half acc.), 4.67 (float acc.) measured | ALU | ALU; accelerators idle ("utilization is 0%") | R6 §A2, §B5 |
| `simdgroup_matrix` bfloat | MSL 3.1 | yes | yes | yes | yes | MSL 4.1 spec §2.4 (R6 §A4) |
| MPP `matmul2d` half × half → float | macOS 26.0, MSL 4.0 | shader fallback | **5.42 TFLOPS (95% of peak), exact** | shader fallback | accelerators, ≈64–65 TFLOPS `[LIKELY]` | R6 §A2, §B5 |
| MPP bfloat × bfloat → float | macOS 26.1 | fallback | exact; pipeline limit drops to 768 threads | fallback | accelerated; MLX measured bf16 36.7 vs fp16 46.7 TFLOPS `[EXPERIMENTAL]` | R6 §A2, §B1 |
| MPP half/bfloat × `int4b`/`uint4b` | macOS 26.4 | fallback | exact; 4.90 TFLOPS; low nibble = even element; `device uchar*` | fallback | accelerated `[LIKELY]`; rate unmeasured | R6 §B1, §B5 |
| MPP char × char → int | macOS 26.0 | fallback | not measured | fallback | ≈122 TOPS, about 2× fp16 `[LIKELY: one source]` | R6 §A2 |
| MPP fp8 / fp4 / int2; `tensor_blockwise` (E8M0 per 32 only) | macOS 27, MSL 4.1 | — | rejected on 26.6.1 | — | fp8 ≈ fp16 rate; fp4 0.8–0.9×; MXFP4 0.5–0.78× `[EXPERIMENTAL]` | R6 §A2, §A3 |
| fp32 × fp32 on accelerators | macOS 26 | — | — | — | only with `relaxed_precision` (TF32-like) | R6 §A2 |
| Group-scaled formats (fp16/bf16 scale and bias per group: MLX affine, GGUF) | — | no native form | no native form | no native form | **no native form** | MSL 4.1 spec Table 2.28 (R6 §A3) |
| Device-scope fence, `coherent(device)` | MSL 3.2 | yes | yes | yes | yes | R6 §A4 |
| acquire/release atomics | MSL 4.1, macOS 27 | per tables | rejected on 26.6.1 | per tables | per tables | R6 §A3, §B1 |

Two consequences shape the skeletons. First, no chip can feed MLX affine or GGUF weights to MPP natively, so the GEMM dequantizes tiles itself (§9.4); Apple's session on custom quantized formats recommends exactly that, through threadgroup memory or cooperative tensors (WWDC26-330) `[VERIFIED]`. Second, MPP runs on every chip from M1 on macOS 26 ("The same code runs across Apple's entire GPU family from M1 to M5", tech talk 111432, R6 §A2) and on the M2 Pro it beat both `MPSMatrixMultiplication` (4.41) and our register-only `simdgroup_matrix` with float accumulation (4.67) `[VERIFIED]`.

## 6. Buffers and memory

### 6.1 Storage and arenas

All buffers use `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`. Shared storage costs nothing in bandwidth on unified memory (181.8 vs 181.3 GB/s, R6 §B2) `[VERIFIED]`, and the CPU can read sampled tokens and write step inputs without blits. llama.cpp measured private buffers at 0.76× and moved to shared (#15906, R5 §9.5).

The static memory plan ([Part XIX](part19-memory.md)) assigns every activation, scratch tile, KV block and argument struct an offset in a few arenas. `sky-metal` allocates each arena once:

```rust
// crates/sky-metal/src/buffer.rs (sketch)
pub struct Arena {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    base: u64,                      // buf.gpuAddress()
    len: usize,
}
impl Arena {
    pub fn new(gpu: &Gpu, len: usize, label: &str) -> Result<Arena, GpuError> {
        let opts = MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        let buf = gpu.device.newBufferWithLength_options(len, opts).ok_or(GpuError::Alloc(len))?;
        buf.setLabel(Some(&NSString::from_str(label)));
        Ok(Arena { base: buf.gpuAddress(), buf, len })
    }
    /// GPU address of a planned slot. The planner guarantees 256-byte alignment.
    pub fn addr(&self, off: usize) -> u64 { debug_assert!(off % 256 == 0 && off < self.len); self.base + off as u64 }
}
```

Alignment rules the planner enforces, each with its source:

| Rule | Value | Source |
|---|---|---|
| Device buffer offsets aligned to the element size | `uint` 4 B, `uint4` 16 B | Apple `setBuffer(_:offset:index:)`; violations read the rounded-down word silently (R9 §1.3) `[VERIFIED]` |
| Format-type tensors (`int4b_format`, …) | 128-byte pointer and 128-byte row stride | MSL 4.1 spec §2.22.2.7, p.101 (R6 §A3) |
| `MTLTensor` int4 row stride | multiple of 256 elements; offsets 128-byte aligned | Feature Set Tables (R6 §A3) |
| Argument structs | 256 B | `[LIKELY: tinygrad's choice; constant-buffer alignment depends on the GPU family]` |
| Everything in an arena | 256 B minimum | Sky rule; makes the three rows above automatic |

### 6.2 Weights: no-copy mmap

Weights come from two sources ([Part XII](part12-quantization.md), [Part XIX](part19-memory.md)):

- **`.skyweights`**, Sky's relayout of an mlx-community safetensors file. Those files start their data at offset ≡ 6 mod 8, so no tensor is even 4-byte aligned, and a zero-copy mapping would be silently wrong (R9 §0, §1.2) `[VERIFIED]`. The relayout is lossless and is written once, keyed by the source sha256 (SPINE D9).
- **GGUF**, mapped directly: tensor offsets are 32-byte aligned (R9 §1.3).

`makeBuffer(bytesNoCopy:length:options:deallocator:)` needs a page-aligned pointer, a page-multiple length and one VM region (R6 §A4; pages are 16 KiB) `[VERIFIED]`. One read-only `mmap` of the whole file is one VM region, which is what llama.cpp does (`src/llama-mmap.cpp:485-495`, `ggml-metal-device.m:2214`, R5 §1.5).

**Splitting at `maxBufferLength`.** The limit is 8.88 GiB on the 16 GB M2 Pro and only "at least 256 MB" by contract (R6 §A4). Qwen3-32B's weights are about 18 GB, so they need several buffers. The two sources are handled differently:

- `.skyweights` is laid out in **segments of at most 1 GiB**, and no tensor crosses a segment boundary. The largest tensor, Qwen3-32B's `lm_head`, is 151,936 × 5,120 × 0.5625 B ≈ 0.44 GB. At load time consecutive segments are grouped into views of at most `maxBufferLength`. Views never overlap, and every tensor lies in exactly one.
- GGUF tensors sit where the file puts them, so Sky uses llama.cpp's **overlapping views**: each view overlaps the next by at least the largest tensor plus a page, so every tensor fits in one view (`ggml-metal-device.m:2226-2252`, R5 §1.5) `[VERIFIED]`.

```rust
// crates/sky-metal/src/buffer.rs (sketch)
pub struct WeightMap { _map: memmap2::Mmap, views: Vec<BufferView> }
pub struct BufferView { buf: Retained<ProtocolObject<dyn MTLBuffer>>, file_off: u64, len: u64 }

impl WeightMap {
    pub fn map_skyweights(gpu: &Gpu, f: &std::fs::File, seg: &[Segment]) -> Result<WeightMap, GpuError> {
        let map = unsafe { memmap2::MmapOptions::new().map(f)? };     // PROT_READ, MAP_SHARED
        let page = 16 * 1024;
        assert!(map.as_ptr() as usize % page == 0);
        let limit = gpu.caps.max_buffer_length as u64;
        let mut views = Vec::new();
        for group in seg.group_consecutive_up_to(limit) {           // segments are page-aligned in the file
            let ptr = unsafe { map.as_ptr().add(group.file_off as usize) } as *mut std::ffi::c_void;
            let len = round_up(group.len, page as u64) as usize;
            let opts = MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
            let buf = unsafe {
                gpu.device.newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(ptr).unwrap(), len, opts, None)
            }.ok_or(GpuError::NoCopy { off: group.file_off, len })?;
            views.push(BufferView { buf, file_off: group.file_off, len: len as u64 });
        }
        Ok(WeightMap { _map: map, views })
    }
    /// GPU address of a tensor: one view, by construction.
    pub fn addr(&self, file_off: u64) -> u64 { /* binary search over views */ unimplemented!() }
}
```

The relayout writer also asserts, for every tensor, the alignment the consuming kernel needs (16 B for `uint4` loads, 128 B for any tensor bound as an MPP format tensor), and the loader checks it again; misalignment is a load error, never a silent wrong answer.

### 6.3 Residency and the heartbeat

On macOS 15 and later, macOS can unwire GPU memory after about a second of idling, and the next token pays page faults. llama.cpp's issue #10119 shows 26.34 ms per token becoming 429.08 ms after a 1.4 s pause on an M2 Ultra; its fix is a residency set per buffer plus a background "heartbeat" that calls `requestResidency` every 5 ms for 180 s after the last compute (`ggml-metal-device.m:925-1031`, R5 §1.4, §9.5) `[VERIFIED]`. MLX implements its wired limit with residency sets too (R3 §2.3).

Sky does the same, with one difference: the plan knows all its buffers up front, so it needs few sets.

```rust
// crates/sky-metal/src/residency.rs (sketch)
pub struct Keeper { sets: Vec<Retained<ProtocolObject<dyn MTLResidencySet>>>, beat: Option<std::thread::JoinHandle<()>> }

impl Keeper {
    pub fn new(gpu: &Gpu, groups: &[&[&ProtocolObject<dyn MTLAllocation>]]) -> Result<Keeper, GpuError> {
        let mut sets = Vec::new();
        for g in groups {                                // weights, KV, arenas: ≤ 32 sets per queue (Feature Set Tables)
            let d = MTLResidencySetDescriptor::new();
            let rs = gpu.device.newResidencySetWithDescriptor_error(&d)?;
            for a in *g { rs.addAllocation(a); }
            rs.commit();
            rs.requestResidency();
            gpu.queue3.addResidencySet(&rs);             // Metal 3 queues accept sets since macOS 15
            if let Some(q4) = &gpu.queue4 { q4.addResidencySet(&rs); }
            sets.push(rs);
        }
        Ok(Keeper { sets, beat: None })
    }
    /// Re-request residency every `period` until `keep_alive` after the last submission.
    pub fn start_heartbeat(&mut self, period: Duration, keep_alive: Duration, last_submit: Arc<AtomicU64>) { /* ... */ }
}
```

Defaults are llama.cpp's (5 ms, 180 s), exposed as `--keep-resident=<seconds>`. Whether the heartbeat is still needed on macOS 26/27, and what it costs in energy, are measurements (§15, item 6); R5 lists the energy cost as unmeasured. llama.cpp also issues a dummy blit at initialization to work around residency-set memory not being released (`ggml-metal-device.m:1022-1026`; Apple forums thread 839089) `[VERIFIED in code]`; XI-T5 tests whether Sky needs it.

### 6.4 The wired limit

`recommendedMaxWorkingSetSize` is "an approximation of how much memory … this GPU device can allocate without affecting its runtime performance": 11.84 GiB, 74% of RAM, on the M2 Pro (R6 §A4) `[VERIFIED]`. Users raise it with `sudo sysctl iogpu.wired_limit_mb=N` (mlx-lm `README.md:286`, R6 §A4).

Sky computes the plan's resident bytes (weights, KV at maximum context, scratch, arguments) before loading and compares them with the limit:

| Plan bytes vs limit | Behavior |
|---|---|
| fits | load |
| exceeds | refuse with a message giving the exact `sysctl` command and the context length that would fit; `--allow-overcommit` loads anyway with a warning |

Sky never runs `sudo` itself. A worked case: Qwen3-32B 4-bit at 32K context with fp16 KV needs 27.5 GB (SPINE §2). The 32-core M5 Max ships only with 36 GB (R6 §0); if its limit is also 74% of RAM, that is 26.6 GB, and Sky would refuse and suggest int8 KV (23.2 GB) or a shorter context `[LIKELY: the 74% ratio is measured only on the M2 Pro; XI-T5 records it on every machine]`.

### 6.5 Accounting

`MTLDevice.currentAllocatedSize` includes no-copy weight buffers; per-process footprint and RSS do not see them, but system-wide wired memory does (R9 §0, finding 7). `sky explain` reports both numbers and labels which is which; [Part XIX](part19-memory.md) and [Part XXIII](part23-benchmarks.md) define the honest peak-memory metric.

## 7. MSL emission

### 7.1 From kernel IR to text

`sky_kernel::Kernel` ([Part IX](part09-scheduler.md)) names a skeleton, its parameters, generated prologue and epilogue statements, an argument list and a **reduction schedule**: the lane-to-element map, the chunk order and the combine tree of every reduction. `msl::Emitter` turns it into an `MslUnit`:

```rust
// crates/sky-metal/src/msl/mod.rs (sketch)
pub struct MslUnit {
    pub source: String,             // complete, self-contained MSL
    pub entry: String,              // "gemv_q4g64__r4_v16_s8"
    pub fn_consts: Vec<(u32, ConstValue)>,
    pub min_lang: MslVersion,       // V3_2 | V4_0 | V4_1
    pub schedule_hash: u64,         // hash of the reduction schedule (§9.6)
    pub provenance: Provenance,     // skeleton id + version, plan id, rule ids: printed by `sky explain`
}
```

Emission is deterministic: the same IR yields the same bytes, which is what makes the data-only plan cache possible (§8) and what golden snapshot tests (`crates/sky-metal/src/msl/snapshots/`) check.

### 7.2 Layout of a generated source file

```text
 1  // sky-metal <emitter version> | skeleton gemv_q4g64 v3 | plan <hash> | rules F12 F31   (provenance)
 2  #pragma METAL fp math_mode(safe)        // contract `inference` (§7.4); `relaxed` emits fast instead
 3  #pragma METAL fp contract(off)
 4  #include <metal_stdlib>
 5  #include <metal_tensor>                 // only if min_lang >= 4.0
 6  #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>   // only for MPP skeletons
 7  using namespace metal;
 8  <Sky prelude, versioned: bf16 decode, butterfly sums, StepState>
 9  <function constants: shapes and flags>
10  <argument struct>
11  <generated prologue and epilogue as static inline functions>
12  <skeleton body with literal tile sizes, calling the hooks>
13  [[kernel]] entry point
```

### 7.3 What is a literal, a function constant, or a runtime value

| Kind | Examples | Why |
|---|---|---|
| Literal in the source | tile sizes, rows per simdgroup, values per lane, MPP descriptor arguments, register array sizes, `MMAX` | template arguments and array sizes must be compile-time constants; MPP descriptors are `constexpr` |
| Function constant (`[[function_constant(i)]]`) | `K`, `N`, group size, `NORM`, epilogue selector `EPI`, `HAS_RESIDUAL` | one library specializes into many pipelines; dead branches are removed at pipeline creation; llama.cpp specializes this way (`newFunctionWithName:constantValues:`, R5 §1.6) |
| Runtime value in `StepState` | position, verify width M, stop flag, RNG counter | changes every token; written by the GPU |
| Runtime value in the argument struct | addresses, `eps`, strides | fixed per plan |

### 7.4 Numerics flags

The MSL default math mode is `fast`: no NaNs or infinities, reassociation allowed, contraction "fast". `safe` "disables unsafe floating-point optimizations … This sets the FP contract to on", which still lets the compiler fuse `a*b+c` within a statement. Contraction is turned off with `#pragma METAL fp contract(off)` (MSL 4.1 spec §1.6.3, pp.15–17) `[VERIFIED]`. MLX builds its kernels with `-fno-fast-math` (R3 §3) and tinygrad compiles with `-fno-fast-math` (R2 §7.4); our microbenchmarks used `fast`.

| Contract ([Part VI](part06-architecture.md) §8.8) | Emitted | Compile options |
|---|---|---|
| `inference` | `math_mode(safe)`, `contract(off)`, every reduction step written as an explicit `fma` or as an add of non-products, `precise::` for `exp`/`rsqrt`/`sin`/`cos` | `mathMode = .safe`; on macOS 27 `FloatingPointConversionRoundingMode = ToNearestEven` set explicitly |
| `relaxed` | `math_mode(fast)` | `mathMode = .fast` |

With contraction off and explicit `fma`, the arithmetic of every output element is fixed by the source text. Two kernels that share a reduction schedule then produce the same bits for the same inputs, whatever else differs between them. This is the mechanism behind batch invariance (§9.2) and chunk invariance. The cost of `safe` + `contract(off)` relative to `fast` is measured per skeleton (§15, item 3); for bandwidth-bound kernels it is `[LIKELY]` near zero.

### 7.5 Language version by OS

| MSL | `MTLLanguageVersion` | OS | Sky uses it for |
|---|---|---|---|
| 3.2 | 0x30002 | macOS 15 | every skeleton that does not need tensors |
| 4.0 | 0x40000 | macOS 26 | MPP GEMM, cooperative tensors |
| 4.1 | 0x40001 | macOS 27 | nothing in year 1; later fp8/fp4 operands, acquire/release |

The emitter computes `min_lang` from the features a kernel uses and compiles at exactly that version, so a kernel that does not need Metal 4 still runs on macOS 15. `<metal_tensor>` fails to compile under MSL 3.2 ("use of undeclared identifier 'dextents'", R6 §B1) `[VERIFIED]`, so the includes are gated too.

## 8. Pipeline compilation and caching

### 8.1 Costs

| Event | Cost | Source |
|---|---|---|
| llama.cpp, first run after install: 34 libraries compiled in parallel | 4.841 s | R5 §1.6 `[VERIFIED]` |
| llama.cpp, every later process start (OS shader cache) | 0.014–0.048 s | R5 §1.6 `[VERIFIED]` |
| One pipeline using tensor ops, cold | ≈ 1–2 s | llama.cpp PR #16634 comment (R6 §A3) `[LIKELY]` |
| Concurrent compilation tasks the device allows | 10 on the M2 Pro | `maximumConcurrentCompilationTaskCount`, R6 probe `[VERIFIED]` |

### 8.2 Policy (SPINE D19)

The spec's non-negotiable 3 requires caches in a data-only format, so:

| Artifact | Contents | When |
|---|---|---|
| **Plan cache** (implicit) | validated, bounded data: kernel choices, skeleton parameters, function-constant values, arena offsets, the dispatch list. **No MSL, no binaries.** | written by `sky run` after planning; read on the next start |
| **OS shader cache** (Apple's, implicit) | compiled code for MSL Sky submitted before | maintained by macOS |
| **`.skybin`** (explicit) | the plan, the MSL, and a Metal binary archive harvested at `sky compile` time | only from `sky compile`; a build artifact like a `sky build` executable |
| **Self-check digests** (implicit, data) | per kernel: a hash of its outputs on fixed inputs (§8.4) | written after a verified run |

A warm start therefore runs: read the plan cache (data) → regenerate MSL with the deterministic emitter (milliseconds) → `newLibraryWithSource_options_error` → the OS shader cache supplies compiled code (llama.cpp's 15–50 ms path) → create pipelines. The cold tensor-op compile of 1–2 s is a first-run cost, or one avoided by shipping a `.skybin`.

The `.skybin` archive uses `MTLBinaryArchive` (loadable by Metal 3 and Metal 4: "both Metal 3 and 4 can load binary archives", Apple, "Using the Metal 4 compilation API") or, on the Metal 4 path, an `MTL4Archive` written by an `MTL4PipelineDataSetSerializer` attached to the `MTL4Compiler` (`serializeAsArchiveAndFlush`) `[VERIFIED: Apple docs in R6-work/docs]`. An archive holds GPU- and OS-specific binaries; Sky loads it only when its key (§8.3) matches and otherwise compiles from the MSL in the same `.skybin`. Pipelines are created with the archive attached; a miss falls back to compilation and is reported by `sky explain`.

### 8.3 Keys

```rust
// crates/sky-metal/src/pipeline.rs (sketch)
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct PipelineKey {
    pub source_hash: [u8; 32],      // sha256 of the regenerated MSL
    pub entry: String,
    pub fn_consts: Vec<(u32, ConstValue)>,
    pub lang: MslVersion,
    pub math: MathContract,         // inference | relaxed
    pub device_arch: String,        // "applegpu_g14s"
    pub os_build: String,           // "25G76"
    pub gpu_compiler: Option<String>, // "32023.886"
}
```

The OS build always identifies the system's Metal compiler. The compiler version itself is best effort: it is the directory name under `/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/`, which on our machine is `32023.886` and which also appears in compiler error paths (R6 [`results/probe.txt`](research/microbench/results/probe.txt)) `[VERIFIED locally; it is a private path, read only for a key, never linked]`. The same key fields extend the `sky-tune` database keys ([Part X](part10-autotuner.md)), because a compiler update can change a kernel's speed.

### 8.4 Compiler-regression self-check

The runtime compiler belongs to the OS, so an OS update can change generated code, its speed, or its results. This has happened: Xcode 26.5's front end (`metalfe-32023.883`) mis-compiled MLX's NAX GEMM (mlx#3586), and a fork disabled NAX on M5 after wrong fp16 GEMM results (R3-kernels §d) `[VERIFIED: issue states]`. The Metal 4.1 tensor path has eleven undocumented behaviors (Rigel, arXiv 2606.12765, R7 §5.5).

So when `(os_build, gpu_compiler)` differs from the last verified pair, `sky-metal` runs every pipeline of the plan once on fixed small inputs before first use and compares output hashes with digests recorded on the last verified pair (and checks them against the TIR reference within tolerance when no digest exists). A mismatch disables the plan cache and tuning entries for that key, falls back to the next skeleton variant, and tells the user. The check is bounded to about a second for a Qwen3 plan `[EXPERIMENTAL: XI-T9 measures it]`.

## 9. Kernel skeletons

The skeletons are written by hand in MSL first and move into L3 kernel IR once `sky-kernel` exists (SPINE D4). This section gives their structure and the hooks the scheduler fills. Parameters (R, VPT, SG, tiles) belong to the autotuner; the values shown are the measured starting points.

### 9.0 Prelude

```metal
// Sky prelude (excerpt). MSL 3.2+.
// bf16 → f32 exactly, without depending on the `bfloat` type.
static inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

// Fixed-order simdgroup sum. An xor butterfly leaves the identical bits in every lane,
// because IEEE addition is commutative; the order is written here, not chosen by simd_sum.
static inline float sky_simd_sum(float v) {
  v += simd_shuffle_xor(v, 16);
  v += simd_shuffle_xor(v, 8);
  v += simd_shuffle_xor(v, 4);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 1);
  return v;
}

struct StepState {               // Part XX §3.2 gives the full layout
  uint pos;                      // position of the first token of this step
  uint verify_m;                 // tokens in this step (1 for plain decode)
  uint stop;                     // nonzero: this step is a no-op
  uint _pad;
  // ...
};
```

### 9.1 Decode GEMV: MLX affine q4, group 64

The format: eight 4-bit values per little-endian `uint32`, element j in bits [4j, 4j+4); per group of 64 inputs one scale and one bias in bf16; `w = s·q + b`. About half of all scales are negative and the bias is then the group maximum (R9 §1.4) `[VERIFIED]`, so the kernel assumes nothing about signs.

The structure is the one that measured best in raw Metal: 4 rows per simdgroup, 16 values per lane per iteration, 8 simdgroups per threadgroup, 186.3 GB/s on the 25600×5120 shape and 191.4 GB/s on a 378 MB matrix, 94–98% of the best streaming read (R6 §B2; source [`bw.swift`](research/microbench/bw.swift) `genGemvKernel`) `[VERIFIED]`. MLX's `affine_qmv_fast` uses the same shape and the identity `Σ w·x = s·Σq·x + b·Σx` (R3 §3.3).

The **norm prologue** fuses the RMSNorm that precedes QKV, gate/up and the LM head. Every simdgroup streams all of `x` for its rows anyway, so it accumulates `Σx²` in the same loop and applies `1/rms` once per row at the end: `y = (1/rms) · W·(x ⊙ γ)`. That costs no memory traffic and no extra dispatch. It is not bit-equal to normalizing first and rounding to bf16 (which is what MLX and HF do) but it is closer to the f32 oracle.

```metal
#define R   4      // rows per simdgroup
#define VPT 16     // values per lane per iteration (VPT divides 64: a lane's chunk is inside one group)
#define SG  8      // simdgroups per threadgroup
constant uint K    [[function_constant(0)]];   // in_features, multiple of 32*VPT (else the masked-tail variant)
constant uint N    [[function_constant(1)]];   // out_features
constant bool NORM [[function_constant(2)]];
constant uint EPI  [[function_constant(3)]];   // 0 store | 1 +residual | 2 SwiGLU pair | 3 head block

[[kernel]] void gemv_q4g64(constant GemvArgs& a [[buffer(0)]],
                           threadgroup float* yb [[threadgroup(0)]],          // EPI 3 only
                           uint tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort ln [[thread_index_in_simdgroup]])
{
  if (a.st->stop != 0) return;                         // no-op step (Part XX §5.4)
  const uint row0 = (tg * SG + sg) * R;
  const uint wpr = K / 8, gpr = K / 64;
  float acc[R];
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  float ss = 0.0f;                                     // Σ x², norm prologue

  for (uint c = ln * VPT; c < K; c += 32 * VPT) {      // lane→k map and chunk order: the reduction schedule
    float xv[VPT];
    float xs = 0.0f;
    for (int i = 0; i < VPT; ++i) {
      const float x0 = float(a.x[c + i]);
      float xi = x0;
      if (NORM) { ss = fma(x0, x0, ss); xi = x0 * bf16_to_f32(a.gamma[c + i]); }
      xv[i] = xi;
      xs += xi;                                        // an add of non-products: nothing to contract
    }
    const uint g = c / 64;
    for (int r = 0; r < R; ++r) {
      const uint n = row0 + r;
      const uint2 wv = *(device const uint2*)(a.w + n * wpr + c / 8);   // 8-byte aligned by layout
      float d = 0.0f;
      for (int t = 0; t < 8; ++t) d = fma(float((wv.x >> (4 * t)) & 0xFu), xv[t], d);
      for (int t = 0; t < 8; ++t) d = fma(float((wv.y >> (4 * t)) & 0xFu), xv[8 + t], d);
      const float s = bf16_to_f32(a.s[n * gpr + g]);
      const float b = bf16_to_f32(a.b[n * gpr + g]);
      acc[r] = fma(s, d, fma(b, xs, acc[r]));
    }
  }
  float inv_rms = 1.0f;
  if (NORM) inv_rms = precise::rsqrt(sky_simd_sum(ss) / float(K) + a.eps);
  for (int r = 0; r < R; ++r) {
    const float v = sky_simd_sum(acc[r]) * inv_rms;
    if (EPI == 3) { if (ln == 0) yb[sg * R + r] = v; }           // head block: finished below
    else if (ln == 0) epilogue_row(a, row0 + r, v);               // generated: store | +resid | (pairs) SwiGLU
  }
  if (EPI == 3) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) epilogue_head(a, tg, yb, ln);                     // generated: q/k-norm, RoPE, KV append
  }
}
```

The epilogue hooks the scheduler can fill:

| `EPI` | Hook | Requirement | Used for |
|---|---|---|---|
| 0 | `y[n] = v` | — | plain projections |
| 1 | `y[n] = resid[n] + v` | same thread reads and writes element n, so in-place is allowed | o_proj and down_proj with the residual add |
| 2 | `h[i] = silu(g)·u` for rows (2i, 2i+1) | gate and up rows interleaved by the relayout (a layout change D5 allows) | gate/up + SwiGLU |
| 3 | per-head finish: Qwen3's q/k RMSNorm over 128 values, RoPE, then q to scratch and k, v into the KV cache at `st->pos` | threadgroup rows are head-aligned: `SG·R = head_dim` (128) | fused QKV |

`EPI 3` needs 128 rows per threadgroup, for example R = 4 and SG = 32 (1,024 threads) or R = 8 and SG = 16. Larger threadgroups and R = 8 both risk bandwidth: R = 8 dropped to 170–175 GB/s in the measured sweep (R6 §B2). Whether head-aligned blocks keep ≥ 90% of bandwidth is `[EXPERIMENTAL: §15 item 4]`; if they do not, the per-head work moves into the attention kernel's prologue ([Part XV](part15-attention.md)), which reads q anyway, and the k/v append moves with it.

A head-block epilogue for the rotate-half RoPE of Qwen3 (head_dim 128, pairs (i, i+64)):

```metal
static inline void epilogue_head(constant GemvArgs& a, uint head, threadgroup float* yb, ushort ln) {
  float v[4];
  for (int j = 0; j < 4; ++j) v[j] = yb[ln + 32 * j];           // lane holds elements ln, ln+32, ln+64, ln+96
  if (head < NQ_HEADS + NKV_HEADS) {                            // q and k heads: per-head RMSNorm, then RoPE
    float ss = 0.0f;
    for (int j = 0; j < 4; ++j) ss = fma(v[j], v[j], ss);
    const float inv = precise::rsqrt(sky_simd_sum(ss) / 128.0f + a.eps);
    const device ushort* nw = head < NQ_HEADS ? a.q_norm : a.k_norm;
    for (int j = 0; j < 4; ++j) v[j] = (v[j] * inv) * bf16_to_f32(nw[ln + 32 * j]);
    const float p = float(a.st->pos);
    for (int j = 0; j < 2; ++j) {                               // element i = ln + 32j pairs with i + 64
      const float ang = p * a.inv_freq[ln + 32 * j];            // fp32, as the reference computes it
      const float c = precise::cos(ang), s = precise::sin(ang);
      const float x1 = v[j], x2 = v[j + 2];
      v[j]     = x1 * c - x2 * s;
      v[j + 2] = x2 * c + x1 * s;
    }
  }
  store_head(a, head, v, ln);                                   // generated: q scratch | K cache | V cache
}
```

### 9.2 The verify GEMV: M ≤ 16, batch-invariant by construction

Speculative decoding verifies 1 + γ tokens per target pass. MLX costs ×1.9–2.0 at M = 3 and ×4.4–4.8 for M = 6…31 on the M2 Pro (R3 §7.C); llama.cpp's verify of 4 tokens costs 2.1× (R5 §8 G). A 60-line kernel that decodes each weight word once and applies it to several activation rows reached ×1.30 at M = 3, but spilled registers from M = 4 because it held 8·M activations per lane (R3 §7.G) `[VERIFIED]`. The skeleton below keeps register use flat in M by staging activations in threadgroup memory.

**The invariance argument.** For token m and row n, the kernel performs exactly the operations of `gemv_q4g64` for that row: the same lane→k map (`c = c0 + ln·VPT`), the same chunk order, the same `fma` sequence over i = 0…VPT−1, the same per-chunk `Σx` order, the same update `fma(s, d, fma(b, xs, acc))`, and the same butterfly. Tokens never share an accumulator. With contraction off and no reassociation (§7.4), the bits of `Y[m][n]` therefore do not depend on M, on the other tokens, or on which instance ran. What may differ between instances is anything outside the per-row arithmetic: rows per simdgroup, simdgroups per threadgroup, `MMAX`. What may not differ is `VPT`, the lane map, the chunk order and the combine tree; the emitter refuses to put kernels with different `schedule_hash` values into one family (§9.6).

```metal
#define VPT  16            // must equal gemv_q4g64's VPT: part of the family's reduction schedule
#define MMAX 8             // literal per instance: 2, 4, 8, 16
#define R    2             // may differ from gemv_q4g64: does not touch the per-row order
#define SG   4

[[kernel]] void verify_gemv_q4g64(constant VerifyArgs& a [[buffer(0)]],
    threadgroup float* xt  [[threadgroup(0)]],   // [MMAX][32*VPT] activations (normed and γ-scaled as in gemv): 16 KiB at MMAX 8
    threadgroup float* xst [[threadgroup(1)]],   // [MMAX][32]     per-lane-chunk Σx, summed in gemv's order
    threadgroup float* sst [[threadgroup(2)]],   // [MMAX][32]     per-lane Σx², carried across chunks (NORM)
    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort ln [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]])
{
  if (a.st->stop != 0) return;
  const uint M = a.st->verify_m;                                // 1 ≤ M ≤ MMAX, written by the GPU
  const uint row0 = (tg * SG + sg) * R, wpr = K / 8, gpr = K / 64;
  float acc[MMAX][R];
  for (int m = 0; m < MMAX; ++m) for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;

  for (uint c0 = 0; c0 < K; c0 += 32 * VPT) {
    // Stage: the thread owning (m, lane ℓ) converts x[m][c0 + ℓ·VPT + i] for i = 0..VPT-1 in order,
    // exactly as gemv's lane ℓ does: writes xt, the chunk's Σx to xst, and continues Σx² in sst.
    stage_tokens(a, c0, M, xt, xst, sst, tid);                  // generated from the same prologue IR as gemv
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = c0 + ln * VPT, g = c / 64;
    for (int r = 0; r < R; ++r) {
      const uint n = row0 + r;
      const uint2 wv = *(device const uint2*)(a.w + n * wpr + c / 8);
      float q[VPT];                                             // decoded once, reused by every token
      for (int t = 0; t < 8; ++t) { q[t] = float((wv.x >> (4 * t)) & 0xFu); q[8 + t] = float((wv.y >> (4 * t)) & 0xFu); }
      const float s = bf16_to_f32(a.s[n * gpr + g]), b = bf16_to_f32(a.b[n * gpr + g]);
      for (uint m = 0; m < MMAX; ++m) {
        if (m >= M) break;
        float d = 0.0f;
        for (int i = 0; i < VPT; ++i) d = fma(q[i], xt[m * 32 * VPT + ln * VPT + i], d);
        acc[m][r] = fma(s, d, fma(b, xst[m * 32 + ln], acc[m][r]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint m = 0; m < M; ++m) {
    const float inv_rms = NORM ? precise::rsqrt(sky_simd_sum(sst[m * 32 + ln]) / float(K) + a.eps) : 1.0f;
    for (int r = 0; r < R; ++r) {
      const float v = sky_simd_sum(acc[m][r]) * inv_rms;
      if (ln == 0) epilogue_row_m(a, m, row0 + r, v);
    }
  }
}
```

`fma(q, x, d)` and gemv's `fma(x', q', d)` compute the same correctly rounded result, because the product inside an `fma` is exact before the single rounding. Registers per lane are `MMAX·R` accumulators plus `VPT` decoded values: 16 + 16 at MMAX 8, R 2. Threadgroup memory for staged activations limits MMAX: 32·VPT·MMAX·4 B is 16 KiB at MMAX 8 and 32 KiB (the whole limit) at MMAX 16, so the M = 16 instance stages half-chunks or stores activations as `half` after the same conversion `[EXPERIMENTAL: register pressure and staging overhead are measured per instance]`.

The targets this kernel serves are the gate G1 (Sky's Q4 GEMV for M = 1…16 batch-invariant and ≥ MLX on every Qwen3-8B/32B shape on the M2 Pro) and the verify target of SPINE §3 (M ≤ 8 at ≤ 1.3× the time of M = 1). The arithmetic: weights are 0.5625 B per element and each extra token adds 2 FLOP per weight plus the staged loads, so on the M2 Pro (ridge ≈ 28 FLOP/byte for fp16 GEMM, R6 hardware model) the extra ALU work at M = 8 is far below the ridge; the risk is instruction issue, not FLOPs, which is why R3's kernel stopped at ×1.30 for M = 3 `[EXPERIMENTAL]`.

### 9.3 GEMM on `simdgroup_matrix` (fallback)

This path serves macOS 15 (no MPP) and any shape or chip where the autotuner finds it faster than §9.4. It follows llama.cpp's `mul_mm` structure: a 64×32 output tile, K steps of 32, 4 simdgroups, weights dequantized per tile into threadgroup memory, 8×8 `simdgroup_matrix` products with fp32 accumulators (`mul_mm.metal:146-359`, R5 §3) `[VERIFIED in code]`.

```metal
#define BN 64      // weight rows (outputs) per threadgroup
#define BM 32      // tokens per threadgroup
#define BK 32
[[kernel]] void sg_qgemm_q4g64(constant QGemmArgs& a [[buffer(0)]],
    threadgroup half* sw [[threadgroup(0)]],                    // [BN][BK] dequantized weights, 4 KiB
    threadgroup half* sx [[threadgroup(1)]],                    // [BM][BK] activations, 2 KiB
    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
  const uint n0 = tg.x * BN, m0 = tg.y * BM;
  const uint nb = (sg & 1) * 32, mb = (sg >> 1) * 16;           // each simdgroup: 32 outputs × 16 tokens
  simdgroup_float8x8 acc[4][2];
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < a.k; k0 += BK) {
    dequant_tile_q4g64(a, sw, n0, k0, BN, BK, tid);             // row (tid/2), 16 values: two words, one group
    load_act_tile(a, sx, m0, k0, BM, BK, tid);                  // M is padded to BM in the arena (Part XIX)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 wa[4], xb[2];
      for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], sw + (nb + 8 * i) * BK + kk, BK);
      for (int j = 0; j < 2; ++j) simdgroup_load(xb[j], sx + (mb + 8 * j) * BK + kk, BK, ulong2(0, 0), true); // Xᵀ
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j)
        simdgroup_multiply_accumulate(acc[i][j], wa[i], xb[j], acc[i][j]);   // C[n][m] += W[n][k]·X[m][k]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  store_tile_transposed(a, acc, n0 + nb, m0 + mb);              // generated epilogue, Y[m][n] row-major
}
```

MLX's equivalent quantized kernel runs at 77% of the M2 Pro's ALU peak against 87% for its fp16 GEMM (R3 §7.D); the same gap is the opening for §9.4 on M1–M4.

### 9.4 GEMM on MPP `matmul2d` with dequantized tiles

MPP is the prefill GEMM on every chip with macOS 26 (§5): 5.42 TFLOPS fp16 on the M2 Pro, and the only path to the M5's neural accelerators. MPP has no native group-scale format (§5), so the kernel dequantizes each K-slice of the weights into a threadgroup tile and passes it as a `tensor_inline` threadgroup tensor; Apple describes this as "the simplest approach" for custom formats (WWDC26-330) and MLX's `qmm_t_nax` does the same (`quantized_nax.h:925-1060`, R3-kernels §d) `[VERIFIED]`.

Tile shapes: on the M2 Pro the best MPP tiles were 32×32 with 1 simdgroup and 64×32 with 2 (5.42 TFLOPS), 64×64 with 4 reached 5.38, and Apple's sample 64×32 with 4 only 4.46 (R6 §B5) `[VERIFIED]`. On the M5 Max the 64×32 sample tile reaches about half of peak and tiles of at least 64×64 are needed (Anemll README:30, R6 §A2) `[LIKELY]`; llama.cpp's Apple-written kernel uses 64×128 (R5 §3) and MLX uses BM 64 × BN 128 for `s`-class M5 parts (R3-kernels §d). The skeleton therefore starts at BM 64 × BN 128 on M5 and at 64×32 / 2 simdgroups on M1–M4, and the autotuner searches from there.

The API usage below follows what compiled and ran on the M2 Pro (R6 [`gemm.swift`](research/microbench/gemm.swift), [`probe.swift`](research/microbench/probe.swift)) and the MSL 4.1 spec's `matmul2d` examples (§7, pp.354–360).

```metal
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define BM  64     // tokens per threadgroup
#define BN  128    // weight rows (outputs) per threadgroup
#define BK  64     // = group size: one scale and one bias per row per K-slice
#define NSG 4

[[kernel]] void mpp_qgemm_q4g64(constant QGemmArgs& a [[buffer(0)]],
    threadgroup half* wt [[threadgroup(0)]],                    // [BN][BK] half, K innermost: 16 KiB
    uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]])
{
  const int n0 = tg.x * BN, m0 = tg.y * BM;                     // M padded to BM in the arena
  // Destination tile C[m0.., n0..] as a strided inline tensor: extents (inner, outer) = (BN, BM).
  array<int, 2> c_str = {1, a.n};
  auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(a.y + m0 * a.n + n0, dextents<int32_t, 2>(BN, BM), c_str);
  auto tW = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(wt, dextents<int32_t, 2>(BK, BN));

  // Right operand is W (N × K, K innermost) used as Wᵀ: transpose_right = true.
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, /*transpose_left*/ false, /*transpose_right*/ true,
                                            /*relaxed_precision*/ false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using TA = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
  auto cT = op.get_destination_cooperative_tensor<TA, decltype(tW), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

  for (int k0 = 0; k0 < a.k; k0 += BK) {
    // 1. Dequantize W[n0 .. n0+BN)[k0 .. k0+BK) into wt. Thread t handles word (t % 8) of rows (t / 8) + 16·p,
    //    so 8 consecutive threads read one row's 32 bytes and write its 128 bytes contiguously.
    for (int p = 0; p < BN / 16; ++p) {
      const int row = tid / 8 + 16 * p, wi = tid % 8, n = n0 + row;
      const uint word = a.w[n * (a.k / 8) + k0 / 8 + wi];
      const float s = bf16_to_f32(a.s[n * (a.k / 64) + k0 / 64]);
      const float b = bf16_to_f32(a.b[n * (a.k / 64) + k0 / 64]);
      threadgroup half* dst = wt + row * BK + wi * 8;
      for (int t = 0; t < 8; ++t) dst[t] = half(fma(s, float((word >> (4 * t)) & 0xFu), b));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. Multiply-accumulate this K-slice: A tile = x[m0.., k0..] with extents (BK, BM), row stride K.
    array<int, 2> a_str = {1, a.k};
    TA tA(a.x + m0 * a.k + k0, dextents<int32_t, 2>(BK, BM), a_str);
    op.run(tA, tW, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);            // every simdgroup is done reading wt
  }
  // 3. Epilogue on the accumulator, element by element, then one store.
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (!cT.is_valid_element(i)) continue;
    auto idx = cT.get_multidimensional_index(i);                // idx[0] = n offset, idx[1] = m offset
    cT[i] = epilogue_elem(a, m0 + idx[1], n0 + idx[0], cT[i]); // generated: residual, activation, ...
  }
  cT.store(tC);
}
```

Points that are not yet verified, each with its test:

| Point | Status |
|---|---|
| `transpose_right = true` with a threadgroup right operand | `[LIKELY: MLX's NAX kernels pass transpose flags in their descriptors (R3-kernels §d); XI-T12 compares against the oracle]` |
| Static `K = BK` with a K-slice tensor of extent exactly BK, accumulated across slices in `multiply_accumulate` mode | `[LIKELY: the spec's gemmBias example accumulates into a loaded cooperative tensor; our measured kernels used a dynamic K instead]` |
| Partial M tiles | avoided: the arena pads M to BM and padded rows are zero |
| Rounding: dequantized weights are rounded to `half` before the product | the same as MLX's NAX path; allowed by the `inference` contract against the f32 oracle |
| Register path: dequantize straight into `get_right_input_cooperative_tensor` (macOS 26.1+), skipping threadgroup memory | `[EXPERIMENTAL]`; Apple recommends it (WWDC26-330), but the element-to-thread layout is opaque (Rigel's "8x8 cooperative_tensor fragment layout", R7 §5.5), so the per-element gather may cost more than it saves |
| Threadgroup memory | 16 KiB single-buffered; double buffering uses the whole 32 KiB limit, which M5 keeps (R6 §A4) |

### 9.5 Decode attention (outline)

[Part XV](part15-attention.md) owns the algorithm. The skeleton contract `sky-metal` provides:

```metal
// attn_decode_split: flash-decoding, GQA-packed, FIXED split size S (positions), M ≤ MMAX query tokens.
// Grid = (n_splits, n_kv_heads) read from an indirect-arguments buffer the GPU wrote this step
// (n_splits = ceil((pos + M) / S)); see Part XX §3.3.
#define D    128
#define G    8        // query heads per KV head (Qwen3-32B: 64 / 8)
#define S    512      // split size: a plan constant shared by decode and verify
#define BKV  32       // keys per block
[[kernel]] void attn_decode_split(constant AttnArgs& a [[buffer(0)]],
    threadgroup half* kt [[threadgroup(0)]], threadgroup half* vt [[threadgroup(1)]],
    threadgroup half* qt [[threadgroup(2)]],                    // G × M query rows, already normed and roped
    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort ln [[thread_index_in_simdgroup]])
{
  if (a.st->stop != 0) return;
  const uint split = tg.x, hkv = tg.y, M = a.st->verify_m, p0 = a.st->pos;
  const uint lo = split * S, hi = min(lo + S, p0 + M);         // absolute boundaries: independent of M and context
  // per owned query row (g, m): running max mr, sum lr, output o[D/32] in fp32
  for (uint kb = lo; kb < hi; kb += BKV) {                     // fixed block order inside the split
    // load K, V block (hkv, kb .. kb+BKV) once for all G·M query rows: GQA K/V read once per KV head
    // scores with the causal mask key ≤ p0 + m; online-softmax update in fp32
  }
  // write partials (o, mr, lr) of each query row to its fixed slot (split, g, m)
}
// The combine reads a row's partials in split order 0..n-1. It runs as its own kernel (K2c);
// Part IX §4.2(c) prices the o_proj-prologue variant and refuses it.
```

The invariance rules are R8's: a fixed split *size* rather than a fixed number of splits, a fixed key-block order, the same KV layout for prefill and decode, and tree verification through masks only (R8 §3.3, §3.5) `[VERIFIED: as published recipes; EXPERIMENTAL for Sky until XI-T14]`. MLX re-reads K/V once per query head; at 4K context its KV reads run about 2× below bandwidth end to end on Qwen3-8B (≈ 97 GB/s, R4) and up to 3× in R3's 8-layer microbenchmark (R3 §7.F); reading each KV head once for all G query heads is the point of the skeleton.

### 9.6 Families and the invariance rule

A **kernel family** is a set of instances sharing one `schedule_hash`. Within a family, bit-equality of each output row across instances follows from §7.4 and is checked by tests (§12). Across families there is no such guarantee, and with MPP there cannot be one: its reduction order is chosen by backend code we cannot see (§5).

The rule for plans: **a decode plan uses one family for every width it can run, 1 … k_max.** This is R8's first design rule ("M=1 is the M=k kernel with masked rows", R8 §3.5). Two candidate families exist:

| Family | Widths | Where | Invariance |
|---|---|---|---|
| `verify_gemv` (ALU, §9.2), with `gemv_q4g64` as its M = 1 member | 1 … 16 | every chip | by construction, tested |
| `verify_mpp`: §9.4 with a fixed small tile (BM 8, 16 or 32), M padded to the tile | 1 … 32 | M5 (accelerators); possibly M1–M4 | `[EXPERIMENTAL]`: MPP must be shown to give the same bits for a row whatever the other rows of the tile hold; XI-T12 tests it |

On the M5 Max the thesis of SPINE §2 is to verify about 30 tokens per weight pass on the accelerators. That requires `verify_mpp`, and then plain decode must run `verify_mpp` at M = 1 too, or speculative output stops being byte-identical to plain output. It pays only if the padded MPP kernel streams weights at M = 1 nearly as fast as the ALU GEMV. Nobody has shown that: R6 lists a one-row `matmul2d` as slower than a GEMV, and both MLX and llama.cpp route small M to ALU kernels (R6 §0, negative results) `[LIKELY]`. This is the decisive M5 measurement (§15, item 1).

## 10. Timing and `sky profile`

### 10.1 What Metal provides

| Source | Granularity | Notes |
|---|---|---|
| `MTLCommandBuffer.gpuStartTime` / `gpuEndTime` | command buffer | used by every R6 benchmark |
| `MTL4CommitFeedback` (`gpuStartTime`, `gpuEndTime`, `error`) | one commit | delivered through `MTL4CommitOptions.addFeedbackHandler` |
| `MTLCounterSampleBuffer`, timestamp set | **stage boundaries only** on Apple GPUs | M2 Pro: `atStageBoundary` true; dispatch, draw, tile and blit boundaries false (R6 §B1) `[VERIFIED]` |
| `MTL4CounterHeap` (type Timestamp) + `MTL4ComputeCommandEncoder.writeTimestamp(granularity:counterHeap:index:)`; `MTLDevice.queryTimestampFrequency` | per write point; `.precise` or `.relaxed` | whether a precise write between two dispatches of one encoder brackets each dispatch is `[EXPERIMENTAL: XI-T15]` |
| `powermetrics` (sudo), IOReport via macmon (no sudo) | 0.2–2 s | GPU clock and power; methodology in R9 §6.3 |

### 10.2 What `sky profile` does

There are no per-dispatch counters, so `sky profile` attributes time in three ways and reports which one produced each number:

1. **Production timing** (always on, near zero cost): GPU time per command buffer, so per decode step, from commit feedback. Feeds `sky bench llm` and the residual "measured vs predicted" in `sky explain`.
2. **Attribution mode** (`sky profile --per-kernel`): the plan is encoded with one compute encoder per kernel and a stage-boundary timestamp after each. One encoder per dispatch cost 1.161 µs per dependent dispatch against 1.148 µs for a serial encoder (R6 §B3), so the distortion is small `[VERIFIED for tiny kernels]`. Never one command buffer per kernel: that costs 14 µs of GPU time each (R6 §B3), and tinygrad's profile mode does exactly that and inflates a 0.6B decode token from 18.4 to 29.4 ms (R2 §7.5) `[VERIFIED]`. On the Metal 4 path the same mode uses `writeTimestamp` once §10.1's experiment confirms its granularity.
3. **Isolation and ablation**: each kernel alone in a loop over a DRAM-sized rotating set of weight copies, as R3 measured MLX (≥ 320 MB working sets, R3 §7); and remove-one-class ablations of a whole step, as R4 did (R4 §7.E). These give per-kernel bandwidth and the cost of dependency bubbles.

`sky profile` prints per kernel: measured µs, predicted µs from the cost model, bytes, FLOPs, achieved GB/s as a fraction of the calibrated read bandwidth, and the attribution method.

## 11. Watchdog, errors and debugging

### 11.1 The watchdog

macOS kills GPU work that blocks display composition (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). llama.cpp sets the undocumented `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (`ggml/src/ggml-metal/ggml-metal.cpp:1027-1029`). MLX issue #3267 reports 100% reproducible kills of long training work with a display attached and none without, and on an M5 Max under macOS 26.5.1 sustained load escalated to `watchdogd` kernel panics even with that variable set. No time limit is documented (R6 §A4) `[VERIFIED]`.

Sky's policy:

| Rule | Mechanism |
|---|---|
| Bound the GPU time of every command buffer | The cost model estimates each command buffer; prefill chunks are split so no command buffer exceeds a budget (default 100 ms `[assumption, to be measured]`). Decode steps of a 32B model are ~30 ms each. |
| Detect and adapt | On a timeout or interactivity error, halve the budget, re-plan the chunking, retry once, and report |
| Do not set the undocumented variable by default | `SKY_METAL_RELAX_TIMEOUT=1` sets it for users who accept undocumented behavior |
| Never run unbounded GPU loops | no persistent kernels (Part XX §8); every spin in tests is bounded, as in R6's `persist.swift` |

R6 did not run a watchdog test, to avoid disrupting the shared machine; the limit with and without a display is an open question (R6, open questions).

### 11.2 Errors

The Metal 3 path checks `MTLCommandBuffer.status` and `error` in the completion handler; the Metal 4 path reads `MTL4CommitFeedback.error`, whose `MTL4CommandQueueError` codes include timeout, notPermitted, outOfMemory, deviceRemoved, accessRevoked and internal (metal-cpp `MTL4CommandQueue.hpp`, R6 §A3). An error invalidates the steps in flight; [Part XX](part20-persistent-execution.md) §5.7 describes how the step state is restored. Out-of-memory errors also appear when command buffers keep too many intermediates alive: MLX hit `kIOGPUCommandBufferCallbackErrorOutOfMemory` when forcing one command buffer per token (R3 §7.F).

### 11.3 Debugging aids

| Aid | Use | Status |
|---|---|---|
| `MTL_DEBUG_LAYER=1` (API validation) | catches misaligned offsets: "must be a multiple of 4 but was set to 2" (R9 §1.3) and other API misuse | `[VERIFIED]`; on in every CI run |
| Shader validation (`MTL_SHADER_VALIDATION=1`) | out-of-bounds device memory accesses inside kernels | `[LIKELY: Apple's "Validating your app's Metal shader usage"]`; nightly CI, it is slow |
| Metal 4 debug classes | `MTL4Debug*` and `MTL4GPUDebug*` are present at run time on the M2 Pro | `[VERIFIED: probe.txt class list]` |
| `MTL_HUD_ENABLED=1` | quick on-screen GPU timing | `[LIKELY]` |
| `MTLCaptureManager` → `.gputrace` | full capture; viewing needs Xcode | `[LIKELY]` (R6 §A4) |
| Labels and debug groups | every buffer, pipeline and step labelled with plan names | Sky rule |
| `SKY_METAL_SYNC=1` | commit and wait after every step, so a failure points at one step | Sky rule |
| Poison and canaries (`debug.rs`) | §12 | Sky rule |

## 12. Kernel test strategy

Kernels are tested on the self-hosted macOS runner (VI-T2) with these layers, all under `crates/sky-metal/tests/`:

| Layer | What it checks | How |
|---|---|---|
| Oracle | values | every instance against the TIR reference interpreter ([Part VII](part07-tensor-ir.md)) in f32, with f64 checks; tolerances per contract and dtype, for example max relative error over rows and a cap on outliers. MLX's own qmv is within 4.0e-4 absolute of fp32 on R3's check (R3 §7.B), a useful bound for the q4 GEMV |
| Shape fuzzing | boundaries | random shapes inside each skeleton's constraints plus the edges: K not a multiple of 32·VPT (masked-tail variant), N not a multiple of rows per threadgroup, M = 1 … MMAX, M not a multiple of the GEMM tile, group boundaries, context lengths at split boundaries ± 1 |
| Adversarial values | numerics | scales with mixed signs (half are negative in real files), bias equal to the group maximum, scale magnitudes over the observed range 1.5e-5 … 0.208 (R9 §1.4), large activations near fp16 limits, −inf in masks |
| Poisoned outputs | coverage | output buffers are filled with a NaN pattern before each run; any unwritten element is caught |
| Canaries | out-of-bounds writes | every test buffer has guard regions filled with a pattern, checked after the run |
| Invariance, bitwise | determinism contracts | run-to-run (same inputs twice); M-invariance (each token at width M equals width 1, for every family instance); chunk invariance (prefill results equal across chunk sizes); split invariance for attention |
| Real weights | format and layout | Qwen3-0.6B tensors from the relayout cache, GPU dequantization against CPU dequantization bit for bit (SPINE D5: layout may change, values may not) |
| Compiler regression | OS updates | the self-check of §8.4, plus a CI job that runs the suite on each new macOS beta |
| Performance | regressions | achieved bandwidth fraction per skeleton and shape against stored thresholds (for example ≥ 88% of calibrated read for the q4 GEMV on 32B shapes, gate G1); a drop beyond run-to-run noise fails the job |

Run-to-run spread is about 10–20% for microsecond-scale dispatch timings and about 2% for bandwidth on the M2 Pro (R6 §B), so performance tests compare medians of interleaved runs, never single runs. R7's lesson from AI-generated kernels applies to any kernel search: only a harness with an oracle, poisoned outputs and adversarial shapes is trustworthy (R7 §12.2).

## 13. Disagreement with the spine

1. **D4 lists "GEMM on simdgroup matrices (M1–M4)".** The evidence puts MPP first on every chip with macOS 26: 5.42 TFLOPS on the M2 Pro, above our register-only `simdgroup_matrix` with float accumulation (4.67) and above MPS (4.41) (R6 §B5). SPINE §3 already relies on it for the M1–M4 prefill target. This part makes the `simdgroup_matrix` GEMM the fallback (macOS 15, or shapes where it measures faster), not the M1–M4 primary. llama.cpp found its own tensor kernels about 5% slower than its `simdgroup_matrix` kernels on an M2 Ultra (R5 §3), so the autotuner keeps both. Resolved: the spine now says (D4) that GEMM goes through MPP `matmul2d` on every chip with macOS 26+, with `simdgroup_matrix` as the fallback.
2. **SPINE §1 describes the headline weights as "fp16 scales and biases".** The mlx-community files store bf16 scales and biases (R9 §0, §1.4; SPINE D5 says bf16). The kernels read bf16. R9 found every Qwen3-8B scale and bias exactly representable in fp16, so a relayout may store fp16 only after a per-tensor exactness check. Resolved: the spine now says bf16 scales and biases (§1, D5), with the scale dtype recorded in the descriptor.
3. **Not a contradiction, but a constraint the spine does not state:** the M5 thesis (about 30 tokens verified per weight pass on the accelerators) and the exactness CI test hold together only if plain decode runs the same MPP family at M = 1 (§9.6). Whether that costs plain-decode speed is unmeasured. Resolved: the spine now says (D10) that byte-identity requires one kernel family for decode and verify, decided by measuring the accelerator GEMM at M = 1 ([Part XXI](part21-speculative.md)).

## 14. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XI-T1 | Crate, device, queues, capability probe | `sky-metal/src/{device,caps}.rs`; `sky-hw/src/gpu.rs` (`GpuInfo`) | `Gpu::open`; `sky hw --gpu` | On the M2 Pro prints the values in R6 `probe.txt` (families 1–8 and metal4 true, 32 KiB threadgroup memory, 8.88 GiB `maxBufferLength`, 11.84 GiB working set) |
| XI-T2 | Raw values and selector shims | `sky-metal/src/raw.rs` | constants of §3.3, guarded `msg_send!` helpers | `raw_values_match_crate`; on macOS 26.6 a 4.1 compile fails with "requires macOS 27"; on 26.4+ an Int4 `MTLTensor` is created |
| XI-T3 | Arenas and the argument ABI | `buffer.rs`, `msl/prelude` | `Arena`, 256-B planner contract, argument structs with GPU addresses | property test on offsets and alignment; a 10-kernel chain with pointer arguments runs clean under `MTL_DEBUG_LAYER=1` |
| XI-T4 | No-copy weight mapping | `buffer.rs` (`WeightMap`) | `.skyweights` segment views; GGUF overlapping views | Qwen3-0.6B `.skyweights` and a GGUF map without copies (RSS does not grow by the file size); GPU dequantization equals CPU dequantization bit for bit; validation clean |
| XI-T5 | Residency and wired limit | `residency.rs` | `Keeper`, heartbeat, plan-fit check | decode, idle 2 s, decode: first token after the gap within 10% of steady state (llama.cpp's #10119 shows 16× without a heartbeat); records `recommendedMaxWorkingSetSize`/RAM on every CI machine |
| XI-T6 | Encoders and synchronization | `encode/{mtl3,mtl4}.rs` | serial-encoder path with fences; Metal 4 path with barriers and argument tables | a 1,000-dispatch dependent chain with a different argument offset per dispatch produces the exact count on both paths; randomized dependency graphs with poisoned buffers pass |
| XI-T7 | MSL emitter | `msl/mod.rs`, `msl/snapshots/` | `Emitter`, layout of §7.2, pragmas, function constants, `min_lang` | golden snapshots; same IR → identical bytes; every snapshot compiles at its `min_lang` on macOS 26 |
| XI-T8 | Compilation and caching per D19 | `compile.rs`, `pipeline.rs` | parallel compile; plan cache as data; `.skybin` archives | the plan-cache file contains no MSL and no binary (format test); warm start ≤ 100 ms for a Qwen3-8B plan `[target]`; a `.skybin` loads on the machine that built it without a compile |
| XI-T9 | Compiler-regression self-check | `pipeline.rs` | digests per (OS build, compiler) | a deliberately perturbed kernel is caught; the check takes ≤ 2 s for a Qwen3-32B plan |
| XI-T10 | Decode GEMV q4g64 with prologue and epilogues | `msl/skeletons/gemv_q4.metal.in` | §9.1, `EPI` 0–3 | oracle within tolerance; ≥ 88% of calibrated read bandwidth on Qwen3-8B/32B shapes on the M2 Pro (gate G1); mixed-sign scale tests pass |
| XI-T11 | Verify GEMV family | `msl/skeletons/verify_gemv_q4.metal.in` | §9.2, instances MMAX 2/4/8/16 | bitwise M-invariance for M = 1…16 on 1,000 fuzzed shapes and on real Qwen3 weights; time(M = 8) ≤ 1.3 × time(M = 1) on 32B shapes `[target, SPINE §3]` |
| XI-T12 | MPP GEMM with dequantized tiles; `verify_mpp` probe | `msl/skeletons/mpp_qgemm.metal.in` | §9.4; tile sweep | oracle within tolerance; ≥ 85% of MPP fp16 GEMM throughput at the same shapes on the M2 Pro (≥ 4.6 of 5.42 TFLOPS) `[target]`; the bitwise row-invariance test for `verify_mpp` reports pass or fail on each chip |
| XI-T13 | `simdgroup_matrix` GEMM fallback | `msl/skeletons/sg_qgemm.metal.in` | §9.3 | oracle; runs with MPP disabled; within 10% of MLX's `qmm_t` on Qwen3 shapes `[target]` |
| XI-T14 | Decode attention skeleton (with Part XV) | `msl/skeletons/attn_decode.metal.in` | §9.5 | bitwise invariance across M and across context lengths for a fixed query; each K/V byte read once per KV head (counted in a debug build) |
| XI-T15 | Timing and `sky profile` | `timing.rs`, `sky-cli` | the three modes of §10.2 | per-kernel attribution sums to within 5% of the step's command-buffer time; the report states each number's method |
| XI-T16 | Test infrastructure | `debug.rs`, `tests/` | poison, canaries, fuzz harness, validation jobs | an injected out-of-bounds write and an injected unwritten element are both caught |
| XI-T17 | Watchdog policy | `pipeline.rs`, `sky-llm` prefill planner | command-buffer time budget, retry | an injected interactivity error (mocked) triggers re-chunking; a real 32K-token prefill runs without errors with a display attached on the CI runner |

## 15. Measurements that decide

1. **`verify_mpp` on the M5 Max:** weight-streaming bandwidth of the MPP GEMM with tiles of 8, 16 and 32 rows at M = 1…32, against the ALU GEMV at M = 1 and the verify GEMV at M = 2…16, plus its bitwise row invariance. Decides which family the M5 decode plan uses (§9.6) and how many tokens speculation can verify per weight pass.
2. **Dequantized-tile MPP GEMM vs fp16 MPP vs MLX** on the M2 Pro and the M5 Max, over tiles including 64×64 and 64×128 and Qwen3 prefill shapes. Decides the prefill GEMM and its tiles ([Part XVII](part17-prefill.md)).
3. **`safe` + `contract(off)` vs `fast`** for each skeleton. Decides whether the `inference` contract costs speed.
4. **Head-aligned GEMV blocks** (1,024-thread threadgroups for `EPI 3`) vs plain blocks. Decides where Qwen3's q/k-norm and RoPE are fused.
5. **Metal 4 vs Metal 3 on the M5 Max:** dependent-dispatch bubble, CPU encode cost, commit-to-event latency. Decides the default command path.
6. **Residency:** first-token latency after idle gaps of 1, 5 and 60 s with and without the heartbeat on macOS 26 and 27, and the heartbeat's energy cost.
7. **Limits on M5 Max machines with 36, 64 and 128 GB:** `maxBufferLength`, `recommendedMaxWorkingSetSize`. Decides segment grouping and the plan-fit check.
8. **Compile times:** cold and warm (OS shader cache) per skeleton family, including tensor-op kernels. Decides whether `sky run` needs a progress message and how much `.skybin` saves for cold TTFT ([Part XVIII](part18-ttft.md)).
9. **`MTL4` `writeTimestamp` granularity** between dispatches. Decides whether attribution mode needs one encoder per kernel.

## 16. Risks and negative results

**Negative results this design accepts (all measured or verified):**
- Metal 4 is not faster per dispatch than Metal 3 on the GPU or the CPU (R6 §B3).
- `simdgroup_matrix` code leaves the M5 accelerators idle; Apple's talk shows "neural accelerator utilization is 0%" (R6 §A2).
- No native format exists for fp16/bf16 group scales and biases; the only native scaled format is E8M0 per 32 elements, OS 27 only (R6 §A3).
- Apple GPUs expose no per-dispatch timestamps, only stage boundaries (R6 §B1).
- Shared and private storage have the same bandwidth, so copying weights into private memory buys nothing (R6 §B2).
- Unaligned `uint` loads at 2-byte offsets return the rounded-down word without error (R9 §1.3).
- llama.cpp ships no binary archives and relies on the OS shader cache (R5 §1.6).
- tinygrad's fast compile path uses the private `MTLCompiler` service; Sky will not, because it is private API (Part VI §8.3).

**Risks:**

| Risk | Effect | Mitigation |
|---|---|---|
| MPP is opaque: its code generation is invisible, its reduction order unknown, and its behavior has changed with compiler releases (MLX #3586; Rigel's undocumented behaviors) | wrong results after an OS update; no batch invariance with ALU kernels | §8.4 self-check; the `simdgroup_matrix` fallback is kept working; one family per plan (§9.6) |
| The OS compiler changes speed or results | tuning entries and digests go stale | keys include the OS build and compiler version (§8.3); CI on macOS betas |
| `objc2-metal` lags the SDK | missing APIs | `raw.rs` shims; a metal-cpp shim is the fallback binding |
| The watchdog kills long command buffers, and on an M5 Max sustained load has produced kernel panics | lost work; worse, a crashed machine | command-buffer time budget; adaptive re-chunking; no persistent kernels |
| Threadgroup memory stays at 32 KiB on M5 | caps MPP tile double-buffering and verify-GEMV staging at M = 16 | single-buffered tiles; half-chunk staging; measured per instance |
| Register pressure in the verify family at M = 16 | spills, as R3's kernel spilled from M = 4 | activations staged in threadgroup memory; R tuned per instance |
| Head-aligned GEMV blocks lose bandwidth | QKV fusion falls back to a separate epilogue kernel | §15 item 4 decides; Part XV's attention prologue takes the work |
| MPP's speed relative to `simdgroup_matrix` varies by chip (llama.cpp: about 5% slower on an M2 Ultra) | prefill regression on some chips | the autotuner keeps both paths per chip |
| Requiring macOS 26 for MPP excludes macOS 15 users from the fast prefill | slower prefill there | the fallback GEMM; support policy stated in `sky hw` output |
