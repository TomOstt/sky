// gemm.swift - matrix-multiply throughput probe on this GPU:
//  (1) register-only simdgroup_matrix multiply-accumulate peak (half x half -> float, and half accum)
//  (2) Metal 4 MPP tensor_ops::matmul2d GEMM (half x half -> float; half x int4b_format -> float)  [MSL 4.0]
//  (3) MPSMatrixMultiplication fp16 GEMM as a vendor reference
// TFLOPS = 2*M*N*K / GPU time. Medians of 7 timed runs after 2 warmups.
import Foundation
import Metal
import MetalPerformanceShaders

let peakSrc = """
#include <metal_stdlib>
using namespace metal;
// Each simdgroup: 4 independent 8x8 accumulators, ITERS x 4 simdgroup_multiply_accumulate (8x8x8 = 1024 FLOP each).
template <typename TA, typename TC>
inline void peak_body(device TC *out, uint gid, uint sg, uint lane, int iters) {
  simdgroup_matrix<TA, 8, 8> a0 = simdgroup_matrix<TA, 8, 8>(TA(0.5h + 0.001h * lane));
  simdgroup_matrix<TA, 8, 8> a1 = simdgroup_matrix<TA, 8, 8>(TA(0.25h));
  simdgroup_matrix<TA, 8, 8> b0 = simdgroup_matrix<TA, 8, 8>(TA(0.125h));
  simdgroup_matrix<TA, 8, 8> b1 = simdgroup_matrix<TA, 8, 8>(TA(0.0625h));
  simdgroup_matrix<TC, 8, 8> c0 = simdgroup_matrix<TC, 8, 8>(TC(0)), c1 = c0, c2 = c0, c3 = c0;
  for (int i = 0; i < iters; i++) {
    simdgroup_multiply_accumulate(c0, a0, b0, c0);
    simdgroup_multiply_accumulate(c1, a0, b1, c1);
    simdgroup_multiply_accumulate(c2, a1, b0, c2);
    simdgroup_multiply_accumulate(c3, a1, b1, c3);
  }
  simdgroup_store(c0, out + (gid * 256), 8);
  simdgroup_store(c1, out + (gid * 256 + 64), 8);
  simdgroup_store(c2, out + (gid * 256 + 128), 8);
  simdgroup_store(c3, out + (gid * 256 + 192), 8);
}
kernel void peak_hf(device float *out [[buffer(0)]], constant int &iters [[buffer(1)]],
                    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
  peak_body<half, float>(out, gid * sgs + sg, sg, lane, iters);
}
kernel void peak_hh(device half *out [[buffer(0)]], constant int &iters [[buffer(1)]],
                    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
  peak_body<half, half>(out, gid * sgs + sg, sg, lane, iters);
}
"""

func mppSrc(tm: Int, tn: Int, nsg: Int) -> String {
    return """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;
    kernel void mpp_f16(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],
                        constant uint3 &d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
      const int M = d.x, N = d.y, K = d.z;
      auto tA = tensor<device half,  dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>(K, M));
      auto tB = tensor<device half,  dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>(N, K));
      auto tC = tensor<device float, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(N, M));
      constexpr auto desc = matmul2d_descriptor(\(tm), \(tn), static_cast<int>(dynamic_extent));
      matmul2d<desc, execution_simdgroups<\(nsg)>> op;
      auto mA = tA.slice(0, (int)tgid.y * \(tm));
      auto mB = tB.slice((int)tgid.x * \(tn), 0);
      auto mC = tC.slice((int)tgid.x * \(tn), (int)tgid.y * \(tm));
      op.run(mA, mB, mC);
    }
    kernel void mpp_q4(device half *A [[buffer(0)]], device uchar *B [[buffer(1)]], device float *C [[buffer(2)]],
                       constant uint3 &d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
      const int M = d.x, N = d.y, K = d.z;
      auto tA = tensor<device half,  dextents<int32_t,2>, tensor_inline>(A, dextents<int32_t,2>(K, M));
      auto tB = tensor<device int4b_format, dextents<int32_t,2>, tensor_inline>(B, dextents<int32_t,2>(N, K));
      auto tC = tensor<device float, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(N, M));
      constexpr auto desc = matmul2d_descriptor(\(tm), \(tn), static_cast<int>(dynamic_extent));
      matmul2d<desc, execution_simdgroups<\(nsg)>> op;
      auto mA = tA.slice(0, (int)tgid.y * \(tm));
      auto mB = tB.slice((int)tgid.x * \(tn), 0);
      auto mC = tC.slice((int)tgid.x * \(tn), (int)tgid.y * \(tm));
      op.run(mA, mB, mC);
    }
    """
}

func timeCB(_ queue: MTLCommandQueue, runs: Int = 7, warm: Int = 2, _ enc: (MTLCommandBuffer) -> Void) -> Double {
    var ts: [Double] = []
    for it in 0..<(warm + runs) {
        let cb = queue.makeCommandBuffer()!
        enc(cb)
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { print("CB error: \(e)"); return .nan }
        if it >= warm { ts.append(gpuSeconds(cb)) }
    }
    return median(ts)
}

@main struct Gemm {
    static func main() {
        print("R6 gemm  \(osVersionString())  \(Date())")
        print(deviceSummary())
        let queue = device.makeCommandQueue()!
        warmupGPU(queue: queue, ms: 400)

        // (1) simdgroup_matrix peak
        let lib = try! compileLibrary(peakSrc)
        let out = device.makeBuffer(length: 64 << 20, options: .storageModePrivate)!
        for name in ["peak_hf", "peak_hh"] {
            let p = makePipeline(lib, name)
            var iters: Int32 = 4096
            let tgs = 16 * 64, sgPerTG = 4
            let t = timeCB(queue) { cb in
                let e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1)
                e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * sgPerTG, height: 1, depth: 1))
                e.endEncoding()
            }
            let flops = Double(tgs * sgPerTG) * Double(iters) * 4 * 1024
            print(String(format: "simdgroup_matrix peak %@: %.3f ms -> %.2f TFLOPS", name, t * 1e3, flops / t / 1e12))
        }

        // (2) MPP matmul2d GEMM
        let M = 4096, N = 4096, K = 4096
        let A = device.makeBuffer(length: M * K * 2, options: .storageModePrivate)!
        let B = device.makeBuffer(length: K * N * 2, options: .storageModePrivate)!
        let Bq = device.makeBuffer(length: K * N / 2, options: .storageModePrivate)!
        let C = device.makeBuffer(length: M * N * 4, options: .storageModePrivate)!
        do {
            let cb = queue.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!
            b.fill(buffer: A, range: 0..<A.length, value: 0x3C); b.fill(buffer: B, range: 0..<B.length, value: 0x30)
            b.fill(buffer: Bq, range: 0..<Bq.length, value: 0x21)
            b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))
        for (tm, tn, nsg) in [(64, 32, 4), (64, 64, 4), (128, 64, 4), (128, 128, 4), (32, 32, 1), (64, 32, 2)] {
            guard let l = try? compileLibrary(mppSrc(tm: tm, tn: tn, nsg: nsg), lang: 0x40000) else { print("MPP compile failed \(tm)x\(tn)"); continue }
            for kname in ["mpp_f16", "mpp_q4"] {
                let p = makePipeline(l, kname)
                let t = timeCB(queue) { cb in
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p)
                    e.setBuffer(A, offset: 0, index: 0); e.setBuffer(kname == "mpp_f16" ? B : Bq, offset: 0, index: 1); e.setBuffer(C, offset: 0, index: 2)
                    e.setBytes(&dims, length: 16, index: 3)
                    e.dispatchThreadgroups(MTLSize(width: N / tn, height: M / tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * nsg, height: 1, depth: 1))
                    e.endEncoding()
                }
                print(String(format: "MPP matmul2d %@ tile %dx%d simdgroups=%d (%dx%dx%d): %.2f ms -> %.2f TFLOPS", kname, tm, tn, nsg, M, N, K, t * 1e3, 2.0 * Double(M * N * K) / t / 1e12))
            }
        }

        if CommandLine.arguments.contains("check") {
            // A filled with 0x3C bytes -> half 0x3C3C, B with 0x30 -> half 0x3030, Bq with 0x21 -> int4 (1 low nibble, 2 high nibble)
            let a = Float(Float16(bitPattern: 0x3C3C)), b = Float(Float16(bitPattern: 0x3030))
            let rb = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
            for (tm, tn, nsg) in [(32, 32, 1), (64, 64, 4), (64, 32, 4)] {
                let l = try! compileLibrary(mppSrc(tm: tm, tn: tn, nsg: nsg), lang: 0x40000)
                for kname in ["mpp_f16", "mpp_q4"] {
                    let p = makePipeline(l, kname)
                    let cb = queue.makeCommandBuffer()!
                    let bl0 = cb.makeBlitCommandEncoder()!; bl0.fill(buffer: C, range: 0..<C.length, value: 0); bl0.endEncoding()
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p)
                    e.setBuffer(A, offset: 0, index: 0); e.setBuffer(kname == "mpp_f16" ? B : Bq, offset: 0, index: 1); e.setBuffer(C, offset: 0, index: 2)
                    e.setBytes(&dims, length: 16, index: 3)
                    e.dispatchThreadgroups(MTLSize(width: N / tn, height: M / tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * nsg, height: 1, depth: 1))
                    e.endEncoding()
                    let bl = cb.makeBlitCommandEncoder()!; bl.copy(from: C, sourceOffset: 0, to: rb, destinationOffset: 0, size: M * N * 4); bl.endEncoding()
                    cb.commit(); cb.waitUntilCompleted()
                    let c = rb.contents().bindMemory(to: Float.self, capacity: M * N)
                    var maxRel: Float = 0, bad = 0
                    for i in 0..<M { for j in 0..<N {
                        let expect = kname == "mpp_f16" ? Float(K) * a * b : Float(K) * a * (j % 2 == 0 ? 1 : 2)
                        let r = abs(c[i * N + j] - expect) / expect
                        if r > 1e-3 { bad += 1 }
                        maxRel = max(maxRel, r)
                    } }
                    print("CHECK \(kname) tile \(tm)x\(tn) sg=\(nsg): max rel err \(maxRel), entries off by >1e-3: \(bad) of \(M * N); C[0]=\(c[0])")
                }
            }
            return
        }

        // (3) MPS reference
        let descA = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * 2, dataType: .float16)
        let descB = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N * 2, dataType: .float16)
        let Ch = device.makeBuffer(length: M * N * 2, options: .storageModePrivate)!
        let descC = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * 2, dataType: .float16)
        let mA = MPSMatrix(buffer: A, descriptor: descA), mB = MPSMatrix(buffer: B, descriptor: descB), mC = MPSMatrix(buffer: Ch, descriptor: descC)
        let mm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: M, resultColumns: N, interiorColumns: K, alpha: 1, beta: 0)
        let t = timeCB(queue) { cb in mm.encode(commandBuffer: cb, leftMatrix: mA, rightMatrix: mB, resultMatrix: mC) }
        print(String(format: "MPSMatrixMultiplication f16 (%dx%dx%d): %.2f ms -> %.2f TFLOPS", M, N, K, t * 1e3, 2.0 * Double(M * N * K) / t / 1e12))
    }
}
