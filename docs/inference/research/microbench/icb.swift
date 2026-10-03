// icb.swift - Indirect command buffer (ICB) replay vs direct encoding (Metal 3 API).
// Encode up to 1000 tiny dependent compute dispatches into an ICB once (CPU-side), then replay with
// executeCommandsInBuffer; compare CPU encode time and GPU time with direct encoding.
// Also: GPU-side encoding of the ICB from a kernel (compute_command in MSL), then replay.
import Foundation
import Metal

let icbSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device uint *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) { buf[tid] += 1u; }

struct ICBArgs {
  command_buffer cmdbuf [[id(0)]];
  compute_pipeline_state pso [[id(1)]];
};
// GPU-side encoding: thread i writes command i. n = number of commands to emit (GPU-decided count).
kernel void encode_cmds(device ICBArgs &args [[buffer(0)]], device uint *target [[buffer(1)]],
                        constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  compute_command cmd(args.cmdbuf, i);
  if (i >= n) { cmd.reset(); return; }
  cmd.set_compute_pipeline_state(args.pso);
  cmd.set_kernel_buffer(target, 0);
  cmd.set_barrier();
  cmd.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(32, 1, 1));
}
"""

@main struct ICB {
    static func main() {
        print("R6 icb  \(osVersionString())  \(Date())")
        print(deviceSummary())
        let queue = device.makeCommandQueue()!
        let lib = try! compileLibrary(icbSrc)
        let pTiny = makePipeline(lib, "tiny", icb: true)
        let buf = device.makeBuffer(length: 1 << 16, options: .storageModePrivate)!
        let maxN = 1000
        warmupGPU(queue: queue, ms: 400)

        // ---- CPU-encoded ICB ----
        let d = MTLIndirectCommandBufferDescriptor()
        d.commandTypes = [.concurrentDispatch]
        d.inheritBuffers = false
        d.inheritPipelineState = false
        d.maxKernelBufferBindCount = 2
        let icbBar = device.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: maxN, options: [])!
        let icbNoBar = device.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: maxN, options: [])!
        let tEnc0 = nowNs()
        for i in 0..<maxN {
            let c = icbBar.indirectComputeCommandAt(i)
            c.setComputePipelineState(pTiny)
            c.setKernelBuffer(buf, offset: 0, at: 0)
            c.setBarrier()
            c.concurrentDispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        let tEnc1 = nowNs()
        for i in 0..<maxN {
            let c = icbNoBar.indirectComputeCommandAt(i)
            c.setComputePipelineState(pTiny)
            c.setKernelBuffer(buf, offset: 0, at: 0)
            c.concurrentDispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        print(String(format: "one-time CPU encode of %d ICB commands: %.1f us (%.3f us/cmd)", maxN, Double(tEnc1 - tEnc0) / 1e3, Double(tEnc1 - tEnc0) / 1e3 / Double(maxN)))

        func timed(_ body: (MTLComputeCommandEncoder) -> Void, conc: Bool) -> (Double, Double) {
            var g: [Double] = [], c: [Double] = []
            for it in 0..<12 {
                let t0 = nowNs()
                let cb = queue.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder(dispatchType: conc ? .concurrent : .serial)!
                body(e)
                e.endEncoding()
                let t1 = nowNs()
                cb.commit(); cb.waitUntilCompleted()
                if let err = cb.error { fatalError("\(err)") }
                if it >= 3 { g.append(gpuSeconds(cb)); c.append(Double(t1 - t0) * 1e-9) }
            }
            return (median(g) * 1e6, median(c) * 1e6)
        }
        print("\n# N dependent tiny dispatches: GPU us (median of 9) | CPU encode us (median)")
        print("N      direct-serial        direct-conc+barrier    ICB(setBarrier) replay  ICB(no barrier) replay")
        for n in [10, 100, 1000] {
            let a = timed({ e in
                e.setComputePipelineState(pTiny); e.setBuffer(buf, offset: 0, index: 0)
                for _ in 0..<n { e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
            }, conc: false)
            let b = timed({ e in
                e.setComputePipelineState(pTiny); e.setBuffer(buf, offset: 0, index: 0)
                for i in 0..<n { if i > 0 { e.memoryBarrier(scope: .buffers) }
                    e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
            }, conc: true)
            let c = timed({ e in
                e.useResource(buf, usage: [.read, .write])
                e.executeCommandsInBuffer(icbBar, range: 0..<n)
            }, conc: true)
            let dd = timed({ e in
                e.useResource(buf, usage: [.read, .write])
                e.executeCommandsInBuffer(icbNoBar, range: 0..<n)
            }, conc: true)
            print(String(format: "%-5d  %8.1f | %7.1f   %8.1f | %7.1f    %8.1f | %7.1f      %8.1f | %7.1f", n, a.0, a.1, b.0, b.1, c.0, c.1, dd.0, dd.1))
            print(String(format: "       per-dispatch GPU us: direct-serial %.3f  direct-conc+barrier %.3f  ICB+barrier %.3f  ICB-nobarrier %.3f",
                         a.0 / Double(n), b.0 / Double(n), c.0 / Double(n), dd.0 / Double(n)))
        }

        // ---- GPU-encoded ICB ----
        print("\n# GPU-side ICB encoding (kernel writes compute_commands), then replay in the same command buffer")
        let icbGPU = device.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: maxN, options: .storageModePrivate)!
        let encFn = lib.makeFunction(name: "encode_cmds")!
        let pEnc = makePipeline(lib, "encode_cmds")
        let ae = encFn.makeArgumentEncoder(bufferIndex: 0)
        let argBuf = device.makeBuffer(length: ae.encodedLength, options: .storageModeShared)!
        ae.setArgumentBuffer(argBuf, offset: 0)
        ae.setIndirectCommandBuffer(icbGPU, index: 0)
        ae.setComputePipelineState(pTiny, index: 1)
        let check = device.makeBuffer(length: 4 * 32, options: .storageModeShared)!
        for n in [10, 100, 1000] {
            var g: [Double] = [], cpu: [Double] = []
            var nn = UInt32(n)
            memset(check.contents(), 0, 128)
            for it in 0..<12 {
                let t0 = nowNs()
                let cb = queue.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                e.setComputePipelineState(pEnc)
                e.setBuffer(argBuf, offset: 0, index: 0); e.setBuffer(check, offset: 0, index: 1); e.setBytes(&nn, length: 4, index: 2)
                e.useResource(icbGPU, usage: .write)
                e.dispatchThreads(MTLSize(width: maxN, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                e.endEncoding()
                let e2 = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
                e2.useResource(check, usage: [.read, .write])
                e2.useResource(icbGPU, usage: .read)
                e2.executeCommandsInBuffer(icbGPU, range: 0..<maxN)
                e2.endEncoding()
                let t1 = nowNs()
                cb.commit(); cb.waitUntilCompleted()
                if let err = cb.error { print("GPU-encoded ICB error: \(err)"); break }
                if it >= 3 { g.append(gpuSeconds(cb)); cpu.append(Double(t1 - t0) * 1e-9) }
            }
            let v = check.contents().bindMemory(to: UInt32.self, capacity: 32)[0]
            if !g.isEmpty {
                print(String(format: "GPU-encoded n=%d (ICB range 0..<%d, rest reset): GPU %.1f us total (encode kernel + replay), CPU %.1f us; check buf[0]=%u (expect %d)",
                             n, maxN, median(g) * 1e6, median(cpu) * 1e6, v, 12 * n))
            }
        }
    }
}
