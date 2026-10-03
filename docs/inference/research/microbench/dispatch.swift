// dispatch.swift - Metal 3 dispatch-overhead and submission-latency microbenchmarks.
// For N in {1,10,100,1000} dependent tiny dispatches (1 threadgroup x 32 threads, each does buf[i] += 1):
//   serial encoder / serial + memoryBarrier / concurrent (no barrier) / concurrent + memoryBarrier(scope:.buffers)
//   / concurrent + memoryBarrier(resources:) / one encoder per dispatch / one command buffer per dispatch
// Reports medians over 9 timed reps (after 3 warmups): GPU time (gpuEndTime-gpuStartTime), CPU encode time,
// commit->completion wall time. Then submission latency tests.
import Foundation
import Metal

let tinySrc = """
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device uint *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) { buf[tid] += 1u; }
kernel void tiny_off(device uint *buf [[buffer(0)]], constant uint &k [[buffer(1)]], uint tid [[thread_position_in_grid]]) { buf[k * 32u + tid] += 1u; }
// ~fixed work kernel: 64 threadgroups x 256 threads, ~a few microseconds
kernel void small(device float *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  float x = buf[tid];
  for (int i = 0; i < 64; i++) x = fma(x, 1.0001f, 0.5f);
  buf[tid] = x;
}
"""

enum Mode: String, CaseIterable {
    case serial, serialBarrier, concurrent, concurrentBarrier, concurrentBarrierRes, encoderPerDispatch, cbPerDispatch
}

struct Result { var gpu: Double; var cpu: Double; var wall: Double }

@main struct Dispatch {
    static func main() {
        print("R6 dispatch  \(osVersionString())  \(Date())")
        print(deviceSummary())
        let queue = device.makeCommandQueue(maxCommandBufferCount: 2048)!
        let lib = try! compileLibrary(tinySrc)
        let pTiny = makePipeline(lib, "tiny")
        let pSmall = makePipeline(lib, "small")
        let tracked = device.makeBuffer(length: 1 << 20, options: [.storageModePrivate])!
        let untracked = device.makeBuffer(length: 1 << 20, options: [.storageModePrivate, .hazardTrackingModeUntracked])!
        let bigbuf = device.makeBuffer(length: 64 * 256 * 4, options: [.storageModePrivate])!
        warmupGPU(queue: queue, ms: 400)

        func run(_ mode: Mode, _ n: Int, _ buf: MTLBuffer, _ p: MTLComputePipelineState, tg: Int, tpt: Int) -> Result {
            let grid = MTLSize(width: tg, height: 1, depth: 1), tpg = MTLSize(width: tpt, height: 1, depth: 1)
            let t0 = nowNs()
            var cbs: [MTLCommandBuffer] = []
            switch mode {
            case .serial, .serialBarrier, .concurrent, .concurrentBarrier, .concurrentBarrierRes:
                let cb = queue.makeCommandBuffer()!
                let conc = !(mode == .serial || mode == .serialBarrier)
                let e = cb.makeComputeCommandEncoder(dispatchType: conc ? .concurrent : .serial)!
                e.setComputePipelineState(p)
                e.setBuffer(buf, offset: 0, index: 0)
                for i in 0..<n {
                    if i > 0 {
                        if mode == .serialBarrier || mode == .concurrentBarrier { e.memoryBarrier(scope: .buffers) }
                        if mode == .concurrentBarrierRes { e.memoryBarrier(resources: [buf]) }
                    }
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                }
                e.endEncoding()
                cbs.append(cb)
            case .encoderPerDispatch:
                let cb = queue.makeCommandBuffer()!
                for _ in 0..<n {
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p); e.setBuffer(buf, offset: 0, index: 0)
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                    e.endEncoding()
                }
                cbs.append(cb)
            case .cbPerDispatch:
                for _ in 0..<n {
                    let cb = queue.makeCommandBuffer()!
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p); e.setBuffer(buf, offset: 0, index: 0)
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
                    e.endEncoding()
                    cbs.append(cb)
                }
            }
            let t1 = nowNs()
            for cb in cbs { cb.commit() }
            let t2 = nowNs()
            cbs.last!.waitUntilCompleted()
            let t3 = nowNs()
            for cb in cbs { if let e = cb.error { fatalError("\(e)") } }
            let gpu = cbs.last!.gpuEndTime - cbs.first!.gpuStartTime
            return Result(gpu: gpu, cpu: Double(t1 - t0) * 1e-9, wall: Double(t3 - t2) * 1e-9)
        }

        func sweep(_ label: String, _ buf: MTLBuffer, _ p: MTLComputePipelineState, tg: Int, tpt: Int, modes: [Mode], ns: [Int]) {
            print("\n# \(label)")
            print("mode                 N     GPU_us(med)  GPU_us/disp  CPUenc_us(med) CPUenc_us/disp  commit->done_us(med)")
            for m in modes {
                var perN: [Int: Double] = [:]
                for n in ns {
                    if m == .cbPerDispatch && n > 1000 { continue }
                    var rs: [Result] = []
                    for it in 0..<12 { let r = run(m, n, buf, p, tg: tg, tpt: tpt); if it >= 3 { rs.append(r) } }
                    let g = median(rs.map { $0.gpu }) * 1e6, c = median(rs.map { $0.cpu }) * 1e6, w = median(rs.map { $0.wall }) * 1e6
                    perN[n] = g
                    print(String(format: "%-20@ %5d  %11.1f  %11.3f  %14.1f  %14.3f  %20.1f", m.rawValue as NSString, n, g, g / Double(n), c, c / Double(n), w))
                }
                if let a = perN[100], let b = perN[1000] {
                    print(String(format: "  -> marginal GPU cost per dispatch (N=100..1000 slope): %.3f us", (b - a) / 900.0))
                }
            }
        }

        let ns = [1, 10, 100, 1000]
        sweep("tiny kernel (1 TG x 32 threads), tracked buffer", tracked, pTiny, tg: 1, tpt: 32, modes: Mode.allCases, ns: ns)
        sweep("tiny kernel (1 TG x 32 threads), UNTRACKED buffer", untracked, pTiny, tg: 1, tpt: 32,
              modes: [.serial, .concurrent, .concurrentBarrier, .encoderPerDispatch], ns: ns)
        sweep("small kernel (64 TG x 256 threads, 64 FMAs/thread), tracked", bigbuf, pSmall, tg: 64, tpt: 256,
              modes: [.serial, .concurrent, .concurrentBarrier], ns: ns)

        // ---------------- Submission latency ----------------
        print("\n# Submission latency (wall clock, 40 reps after 5 warmups; GPU kept warm between reps)")
        func lat(_ label: String, _ body: () -> Double) {
            var xs: [Double] = []
            for it in 0..<45 { let x = body(); if it >= 5 { xs.append(x) } }
            let s = xs.sorted()
            print(String(format: "%-62@ median %7.1f us  p10 %7.1f  p90 %7.1f  min %7.1f", label as NSString,
                         median(xs) * 1e6, s[s.count / 10] * 1e6, s[s.count * 9 / 10] * 1e6, s[0] * 1e6))
        }
        lat("empty CB: commit -> waitUntilCompleted returns") {
            let cb = queue.makeCommandBuffer()!
            let t0 = nowNs(); cb.commit(); cb.waitUntilCompleted(); return Double(nowNs() - t0) * 1e-9
        }
        lat("1 tiny dispatch CB: commit -> waitUntilCompleted returns") {
            let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pTiny); e.setBuffer(tracked, offset: 0, index: 0)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            e.endEncoding()
            let t0 = nowNs(); cb.commit(); cb.waitUntilCompleted(); return Double(nowNs() - t0) * 1e-9
        }
        lat("1 tiny dispatch CB: commit -> spin on cb.status == .completed") {
            let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pTiny); e.setBuffer(tracked, offset: 0, index: 0)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            e.endEncoding()
            let t0 = nowNs(); cb.commit(); while cb.status != .completed && cb.status != .error {}; return Double(nowNs() - t0) * 1e-9
        }
        let ev = device.makeSharedEvent()!
        var evVal: UInt64 = 0
        lat("1 tiny dispatch CB + encodeSignalEvent: commit -> CPU spin sees value") {
            evVal += 1
            let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pTiny); e.setBuffer(tracked, offset: 0, index: 0)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            e.endEncoding(); cb.encodeSignalEvent(ev, value: evVal)
            let t0 = nowNs(); cb.commit(); while ev.signaledValue < evVal {}; return Double(nowNs() - t0) * 1e-9
        }
        lat("1 tiny dispatch CB: commit -> addCompletedHandler fires (semaphore)") {
            let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pTiny); e.setBuffer(tracked, offset: 0, index: 0)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            e.endEncoding()
            let sem = DispatchSemaphore(value: 0)
            cb.addCompletedHandler { _ in sem.signal() }
            let t0 = nowNs(); cb.commit(); sem.wait(); return Double(nowNs() - t0) * 1e-9
        }
        // GPU-side gap between back-to-back command buffers committed together vs CPU round trip in between
        print("\n# GPU timeline gap between consecutive command buffers (us, median of 40)")
        var gapsPipelined: [Double] = [], gapsRoundTrip: [Double] = []
        for it in 0..<45 {
            let a = queue.makeCommandBuffer()!, b = queue.makeCommandBuffer()!
            for cb in [a, b] { let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(pSmall); e.setBuffer(bigbuf, offset: 0, index: 0)
                e.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding() }
            a.commit(); b.commit(); b.waitUntilCompleted()
            if it >= 5 { gapsPipelined.append((b.gpuStartTime - a.gpuEndTime) * 1e6) }
            let c = queue.makeCommandBuffer()!, d = queue.makeCommandBuffer()!
            for cb in [c, d] { let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(pSmall); e.setBuffer(bigbuf, offset: 0, index: 0)
                e.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding() }
            c.commit(); c.waitUntilCompleted(); d.commit(); d.waitUntilCompleted()
            if it >= 5 { gapsRoundTrip.append((d.gpuStartTime - c.gpuEndTime) * 1e6) }
        }
        print(String(format: "both committed up front:        gap median %.1f us (min %.1f, max %.1f)", median(gapsPipelined), gapsPipelined.min()!, gapsPipelined.max()!))
        print(String(format: "CPU waits for 1st, then commits: gap median %.1f us (min %.1f, max %.1f)", median(gapsRoundTrip), gapsRoundTrip.min()!, gapsRoundTrip.max()!))
    }
}
