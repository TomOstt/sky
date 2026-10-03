// probe.swift - feature probe: GPU families, limits, counters, Metal 4 runtime classes,
// and runtime compilation of MSL 4.0 tensor / Metal Performance Primitives kernels.
import Foundation
import Metal
import ObjectiveC

func section(_ s: String) { print("\n=== \(s) ===") }

func tryCompile(_ name: String, lang: UInt, _ src: String) -> MTLLibrary? {
    do {
        let lib = try compileLibrary(src, lang: lang)
        print("[COMPILE OK ] \(name) (lang 0x\(String(lang, radix: 16)))")
        return lib
    } catch {
        let msg = "\(error)".split(separator: "\n").prefix(12).joined(separator: "\n      ")
        print("[COMPILE ERR] \(name) (lang 0x\(String(lang, radix: 16))):\n      \(msg)")
        return nil
    }
}

func tryPipeline(_ lib: MTLLibrary?, _ fn: String) -> MTLComputePipelineState? {
    guard let lib = lib, let f = lib.makeFunction(name: fn) else { return nil }
    do {
        let p = try device.makeComputePipelineState(function: f)
        print("[PIPELINE OK] \(fn): maxTotalThreadsPerThreadgroup=\(p.maxTotalThreadsPerThreadgroup) threadExecutionWidth=\(p.threadExecutionWidth) staticThreadgroupMemoryLength=\(p.staticThreadgroupMemoryLength)")
        return p
    } catch {
        print("[PIPELINE ERR] \(fn): \(error)")
        return nil
    }
}

let mppGemmSrc = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// C[M,N] (float) = A[M,K] (half) x B[K,N] (half), all row-major. 64x32 output tile per threadgroup, 4 simdgroups.
kernel void mpp_gemm(device half  *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],
                     constant uint3 &d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = d.x, N = d.y, K = d.z;
  auto tA = tensor<device half,  dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>(K, M));
  auto tB = tensor<device half,  dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>(N, K));
  auto tC = tensor<device float, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, (int)tgid.y * 64);
  auto mB = tB.slice((int)tgid.x * 32, 0);
  auto mC = tC.slice((int)tgid.x * 32, (int)tgid.y * 64);
  op.run(mA, mB, mC);
}
"""

let mppQ4Src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// C[M,N] (float) = A[M,K] (half) x B[K,N] (int4, packed 2 per byte, row-major K x N).
kernel void mpp_q4(device half *A [[buffer(0)]], device uchar *B [[buffer(1)]], device float *C [[buffer(2)]],
                   constant uint3 &d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = d.x, N = d.y, K = d.z;
  auto tA = tensor<device half,  dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>(K, M));
  auto tB = tensor<device int4b_format, dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>(N, K));
  auto tC = tensor<device float, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, (int)tgid.y * 64);
  auto mB = tB.slice((int)tgid.x * 32, 0);
  auto mC = tC.slice((int)tgid.x * 32, (int)tgid.y * 64);
  op.run(mA, mB, mC);
}
"""

let mppBf16Src = mppGemmSrc.replacingOccurrences(of: "device half  *A", with: "device bfloat *A")
    .replacingOccurrences(of: "device half *B", with: "device bfloat *B")
    .replacingOccurrences(of: "tensor<device half,  dextents<int32_t,2>, tensor_inline>(A", with: "tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(A")
    .replacingOccurrences(of: "tensor<device half,  dextents<int32_t,2>, tensor_inline>(B", with: "tensor<device bfloat, dextents<int32_t,2>, tensor_inline>(B")
    .replacingOccurrences(of: "mpp_gemm", with: "mpp_bf16")

let fp8Src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mpp_fp8(device half *A [[buffer(0)]], device metal_fp8_e4m3_format *B [[buffer(1)]], device float *C [[buffer(2)]],
                   uint2 tgid [[threadgroup_position_in_grid]]) {
  auto tA = tensor<device half,  dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>(64, 64));
  auto tB = tensor<device metal_fp8_e4m3_format, dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>(32, 64));
  auto tC = tensor<device float, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(32, 64));
  constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  op.run(tA, tB, tC);
}
"""

let plainSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = float(i); }
"""

let tensorOnlySrc = """
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
kernel void t(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  auto t = tensor<device float, dextents<int32_t,2>, tensor_inline>(o, dextents<int32_t,2>(16, 16));
  o[i] = float(t.get_extent(0) * t.get_extent(1));
}
"""

let memOrderSrc = """
#include <metal_stdlib>
using namespace metal;
// MSL 3.2+: device-scope fence + relaxed atomics; MSL 4.1 adds acquire/release orders.
kernel void mo(device atomic_uint *flag [[buffer(0)]], device uint *data [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  data[i] = i;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_store_explicit(flag, 1u, memory_order_relaxed);
}
"""

let acqRelSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void ar(device atomic_uint *flag [[buffer(0)]], device uint *data [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  data[i] = i;
  atomic_store_explicit(flag, 1u, memory_order_release);
  uint v = atomic_load_explicit(flag, memory_order_acquire);
  data[i] += v;
}
"""

func runMppGemmCheck(_ p: MTLComputePipelineState, M: Int, N: Int, K: Int, bf16: Bool = false) {
    // fill A, B with small integers so fp16 products are exact
    var a = [Float](repeating: 0, count: M * K), b = [Float](repeating: 0, count: K * N)
    for i in 0..<(M * K) { a[i] = Float((i * 7 + 3) % 5) - 2 }
    for i in 0..<(K * N) { b[i] = Float((i * 5 + 1) % 7) - 3 }
    func toHalfBits(_ x: Float) -> UInt16 { Float16(x).bitPattern }
    func toBf16Bits(_ x: Float) -> UInt16 { UInt16(x.bitPattern >> 16) }
    let ah = a.map { bf16 ? toBf16Bits($0) : toHalfBits($0) }, bh = b.map { bf16 ? toBf16Bits($0) : toHalfBits($0) }
    let bufA = device.makeBuffer(bytes: ah, length: ah.count * 2, options: .storageModeShared)!
    let bufB = device.makeBuffer(bytes: bh, length: bh.count * 2, options: .storageModeShared)!
    let bufC = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
    memset(bufC.contents(), 0, M * N * 4)
    var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))
    let q = device.makeCommandQueue()!
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    e.setBuffer(bufA, offset: 0, index: 0); e.setBuffer(bufB, offset: 0, index: 1); e.setBuffer(bufC, offset: 0, index: 2)
    e.setBytes(&dims, length: 16, index: 3)
    e.dispatchThreadgroups(MTLSize(width: N / 32, height: M / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("  run error: \(err)"); return }
    let c = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
    var maxErr: Float = 0
    for i in 0..<M { for j in 0..<N {
        var s: Float = 0
        for k in 0..<K { s += a[i * K + k] * b[k * N + j] }
        maxErr = max(maxErr, abs(s - c[i * N + j]))
    } }
    print("  RUN \(p.label ?? "") M=\(M) N=\(N) K=\(K): max |err| vs CPU = \(maxErr)  (C[0]=\(c[0]))")
}

func runMppQ4Check(_ p: MTLComputePipelineState, M: Int, N: Int, K: Int) {
    var a = [Float](repeating: 0, count: M * K), b = [Int](repeating: 0, count: K * N)
    for i in 0..<(M * K) { a[i] = Float((i * 7 + 3) % 5) - 2 }
    for i in 0..<(K * N) { b[i] = ((i * 5 + 1) % 16) - 8 }       // int4 range [-8, 7]
    let ah = a.map { Float16($0).bitPattern }
    var packed = [UInt8](repeating: 0, count: K * N / 2)
    for i in stride(from: 0, to: K * N, by: 2) {                   // low nibble = even element (assumed)
        packed[i / 2] = UInt8(b[i] & 0xF) | (UInt8(b[i + 1] & 0xF) << 4)
    }
    let bufA = device.makeBuffer(bytes: ah, length: ah.count * 2, options: .storageModeShared)!
    let bufB = device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared)!
    let bufC = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
    memset(bufC.contents(), 0, M * N * 4)
    var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))
    let q = device.makeCommandQueue()!
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    e.setBuffer(bufA, offset: 0, index: 0); e.setBuffer(bufB, offset: 0, index: 1); e.setBuffer(bufC, offset: 0, index: 2)
    e.setBytes(&dims, length: 16, index: 3)
    e.dispatchThreadgroups(MTLSize(width: N / 32, height: M / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { print("  run error: \(err)"); return }
    let c = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
    var maxErr: Float = 0, maxErrSwapped: Float = 0
    for i in 0..<M { for j in 0..<N {
        var s: Float = 0, s2: Float = 0
        for k in 0..<K { s += a[i * K + k] * Float(b[k * N + j]); s2 += a[i * K + k] * Float(b[k * N + (j ^ 1)]) }
        maxErr = max(maxErr, abs(s - c[i * N + j])); maxErrSwapped = max(maxErrSwapped, abs(s2 - c[i * N + j]))
    } }
    print("  RUN int4 M=\(M) N=\(N) K=\(K): max|err| (low nibble first) = \(maxErr); (high nibble first) = \(maxErrSwapped)")
}

@main struct Probe {
    static func main() {
        print("R6 probe  \(osVersionString())  \(Date())")
        print(deviceSummary())

        section("GPU families (supportsFamily, raw values)")
        let fams: [(String, Int)] = (1...12).map { ("apple\($0)", 1000 + $0) } +
            [("mac1", 2001), ("mac2", 2002), ("common1", 3001), ("common2", 3002), ("common3", 3003),
             ("macCatalyst1", 4001), ("macCatalyst2", 4002), ("metal3", 5001), ("metal4", 5002), ("metal5?", 5003)]
        for (n, v) in fams {
            if let f = MTLGPUFamily(rawValue: v) { print("  \(n) (\(v)): \(device.supportsFamily(f))") }
        }

        section("Device limits")
        print("  name=\(device.name) arch=\(device.architecture.name) registryID=\(device.registryID)")
        print("  maxThreadsPerThreadgroup=\(device.maxThreadsPerThreadgroup)")
        print("  maxThreadgroupMemoryLength=\(device.maxThreadgroupMemoryLength)")
        print("  maxBufferLength=\(device.maxBufferLength) (\(Double(device.maxBufferLength) / 1073741824) GiB)")
        print("  recommendedMaxWorkingSetSize=\(device.recommendedMaxWorkingSetSize) (\(Double(device.recommendedMaxWorkingSetSize) / 1073741824) GiB)")
        print("  physical memory=\(ProcessInfo.processInfo.physicalMemory) ratio=\(Double(device.recommendedMaxWorkingSetSize) / Double(ProcessInfo.processInfo.physicalMemory))")
        print("  hasUnifiedMemory=\(device.hasUnifiedMemory) location=\(device.location.rawValue) lowPower=\(device.isLowPower) headless=\(device.isHeadless)")
        print("  currentAllocatedSize=\(device.currentAllocatedSize)")
        print("  argumentBuffersSupport=\(device.argumentBuffersSupport.rawValue) maxArgumentBufferSamplerCount=\(device.maxArgumentBufferSamplerCount)")
        print("  supportsFunctionPointers=\(device.supportsFunctionPointers) supportsDynamicLibraries=\(device.supportsDynamicLibraries) supportsRaytracing=\(device.supportsRaytracing)")
        print("  supportsBFloat16? (no direct API)  supportsShaderBarycentricCoordinates=\(device.supportsShaderBarycentricCoordinates)")
        print("  sparseTileSizeInBytes=\(device.sparseTileSizeInBytes)  peerGroupID=\(device.peerGroupID) maxTransferRate=\(device.maxTransferRate)")
        print("  shouldMaximizeConcurrentCompilation=\(device.shouldMaximizeConcurrentCompilation) maximumConcurrentCompilationTaskCount=\(device.maximumConcurrentCompilationTaskCount)")
        let pts: [(String, MTLCounterSamplingPoint)] = [("atStageBoundary", .atStageBoundary), ("atDrawBoundary", .atDrawBoundary),
            ("atDispatchBoundary", .atDispatchBoundary), ("atTileDispatchBoundary", .atTileDispatchBoundary), ("atBlitBoundary", .atBlitBoundary)]
        for (n, p) in pts { print("  supportsCounterSampling(\(n))=\(device.supportsCounterSampling(p))") }
        if let sets = device.counterSets { for s in sets { print("  counterSet \(s.name): \(s.counters.map { $0.name })") } }
        let ts = device.sampleTimestamps()
        print("  sampleTimestamps cpu=\(ts.cpu) gpu=\(ts.gpu)")

        section("Objective-C runtime: Metal 4 protocols/classes present")
        var pc: UInt32 = 0
        if let protos = objc_copyProtocolList(&pc) {
            var names: [String] = []
            for i in 0..<Int(pc) { let n = String(cString: protocol_getName(protos[i])); if n.hasPrefix("MTL4") || n.contains("MTLTensor") || n.hasPrefix("MTLResidency") { names.append(n) } }
            print("  protocols (\(names.count)): \(names.sorted().joined(separator: ", "))")
        }
        var cc: Int32 = 0
        cc = objc_getClassList(nil, 0)
        let classes = UnsafeMutablePointer<AnyClass>.allocate(capacity: Int(cc))
        let got = objc_getClassList(AutoreleasingUnsafeMutablePointer(classes), cc)
        var cnames: [String] = []
        for i in 0..<Int(got) {
            let n = String(cString: class_getName(classes[i]))
            if n.contains("MTL4") || n.contains("Tensor") && (n.hasPrefix("MTL") || n.hasPrefix("AGX") || n.hasPrefix("_MTL") || n.hasPrefix("IOGPU")) { cnames.append(n) }
        }
        print("  classes (\(cnames.count)): \(cnames.sorted().joined(separator: ", "))")
        let devObj = device as AnyObject
        print("  device class = \(String(cString: class_getName(object_getClass(devObj)!)))")
        for sel in ["newMTL4CommandQueue", "newMTL4CommandQueueWithDescriptor:error:", "newCommandAllocator", "newCommandAllocatorWithDescriptor:error:",
                    "newCommandBuffer", "newArgumentTableWithDescriptor:error:", "newCompilerWithDescriptor:error:",
                    "newTensorWithDescriptor:error:", "tensorSizeAndAlignWithDescriptor:", "newCounterHeapWithDescriptor:error:",
                    "newResidencySetWithDescriptor:error:", "newLogStateWithDescriptor:error:", "newArchiveWithURL:error:",
                    "newPipelineDataSetSerializerWithDescriptor:", "supportsPlacementSparse", "newTensorWithDescriptor:attachments:error:"] {
            print("  device responds to \(sel): \(devObj.responds(to: NSSelectorFromString(sel)))")
        }
        if devObj.responds(to: NSSelectorFromString("newMTL4CommandQueue")),
           let q = devObj.perform(NSSelectorFromString("newMTL4CommandQueue"))?.takeRetainedValue() {
            print("  newMTL4CommandQueue -> \(String(cString: class_getName(object_getClass(q)!)))")
            for sel in ["commit:count:", "commit:count:options:", "signalEvent:value:", "waitForEvent:value:", "addResidencySet:"] {
                print("    queue responds to \(sel): \((q as AnyObject).responds(to: NSSelectorFromString(sel)))")
            }
        }
        if devObj.responds(to: NSSelectorFromString("newCommandBuffer")),
           let cb = devObj.perform(NSSelectorFromString("newCommandBuffer"))?.takeRetainedValue() {
            print("  newCommandBuffer (MTL4) -> \(String(cString: class_getName(object_getClass(cb)!)))")
        }
        if devObj.responds(to: NSSelectorFromString("newCommandAllocator")),
           let a = devObj.perform(NSSelectorFromString("newCommandAllocator"))?.takeRetainedValue() {
            print("  newCommandAllocator -> \(String(cString: class_getName(object_getClass(a)!)))")
        }

        section("Runtime MSL compilation")
        _ = tryPipeline(tryCompile("plain kernel MSL 3.2", lang: 0x30002, plainSrc), "k")
        _ = tryPipeline(tryCompile("plain kernel MSL 4.0", lang: 0x40000, plainSrc), "k")
        _ = tryCompile("plain kernel MSL 4.1 (needs OS 27)", lang: 0x40001, plainSrc)
        _ = tryCompile("<metal_tensor> under MSL 3.2 (expect error)", lang: 0x30002, tensorOnlySrc)
        _ = tryPipeline(tryCompile("<metal_tensor> tensor_inline MSL 4.0", lang: 0x40000, tensorOnlySrc), "t")
        _ = tryPipeline(tryCompile("device-scope fence + relaxed atomics MSL 3.2", lang: 0x30002, memOrderSrc), "mo")
        _ = tryPipeline(tryCompile("acquire/release atomics MSL 4.0 (spec says 4.1)", lang: 0x40000, acqRelSrc), "ar")
        let lib = tryCompile("MPP matmul2d half x half -> float MSL 4.0", lang: 0x40000, mppGemmSrc)
        if let p = tryPipeline(lib, "mpp_gemm") {
            runMppGemmCheck(p, M: 128, N: 64, K: 256)
            runMppGemmCheck(p, M: 256, N: 256, K: 512)
        }
        let libbf = tryCompile("MPP matmul2d bfloat x bfloat -> float MSL 4.0 (OS 26.1+)", lang: 0x40000, mppBf16Src)
        if let p = tryPipeline(libbf, "mpp_bf16") { runMppGemmCheck(p, M: 128, N: 64, K: 256, bf16: true) }
        if let p = tryPipeline(tryCompile("MPP matmul2d half x int4b_format -> float MSL 4.0 (OS/SDK 26.4+)", lang: 0x40000, mppQ4Src), "mpp_q4") {
            runMppQ4Check(p, M: 128, N: 256, K: 256)
        }
        _ = tryCompile("MPP matmul2d half x fp8_e4m3 MSL 4.0 (spec: 4.1)", lang: 0x40000, fp8Src)
        _ = tryCompile("MPP matmul2d half x fp8_e4m3 MSL 4.1", lang: 0x40001, fp8Src)
    }
}
