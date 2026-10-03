// persist.swift - persistent-kernel feasibility probe: device-wide barrier across threadgroups of ONE dispatch,
// built from relaxed atomics + atomic_thread_fence(seq_cst, thread_scope_device) + coherent(device) buffers (MSL 3.2).
// Every spin loop is bounded, so a non-resident threadgroup produces a recorded timeout instead of a GPU hang.
// Measures per-barrier latency vs. the ~1.2 us dependent-dispatch boundary, checks cross-threadgroup visibility,
// and shows what happens when the grid exceeds what can be co-resident (no forward-progress guarantee in MSL).
import Foundation
import Metal

let persistSrc = """
#include <metal_stdlib>
using namespace metal;
// stats[0] = timeouts, stats[1] = visibility violations
kernel void gbar(CTRQUAL device atomic_uint *ctr [[buffer(0)]], COHERENT device uint *data [[buffer(1)]],
                 device atomic_uint *stats [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                 uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                 uint tid [[thread_index_in_threadgroup]]) {
  const uint nbar = p.x, spinMax = p.y;
  threadgroup uint abort_flag;
  if (tid == 0) abort_flag = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint b = 1; b <= nbar; b++) {
    if (tid == 0) {
      data[tg] = b;                                         // "work" of phase b
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);  // release
      atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
      const uint target = b * ntg;
      uint spins = 0;
      while (SPINLOAD < target) {
        if (++spins > spinMax) { atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed); abort_flag = 1; break; }
      }
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);  // acquire
      if (abort_flag == 0) {
        uint other = data[(tg + 1) % ntg];                  // must already hold phase >= b
        if (other < b) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (abort_flag != 0) return;
  }
}
"""

@main struct Persist {
    static func main() {
        print("R6 persist  \(osVersionString())  \(Date())")
        print(deviceSummary())
        let queue = device.makeCommandQueue()!
        warmupGPU(queue: queue, ms: 300)
        let ctr = device.makeBuffer(length: 256, options: .storageModeShared)!
        let data = device.makeBuffer(length: 1 << 20, options: .storageModeShared)!
        let stats = device.makeBuffer(length: 256, options: .storageModeShared)!
        // spin variants: (label, counter qualifier, spin load expression, data coherent)
        let variants: [(String, String, String, Bool)] = [
            ("relaxed atomic_load, plain ctr", "", "atomic_load_explicit(ctr, memory_order_relaxed)", true),
            ("relaxed atomic_load + fence in loop", "", "(atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device), atomic_load_explicit(ctr, memory_order_relaxed))", true),
            ("relaxed atomic_load, coherent(device) ctr", "coherent(device)", "atomic_load_explicit(ctr, memory_order_relaxed)", true),
            ("atomic_fetch_add(ctr,0) RMW spin", "", "atomic_fetch_add_explicit(ctr, 0u, memory_order_relaxed)", true),
            ("atomic_fetch_add(ctr,0) RMW spin, NON-coherent data", "", "atomic_fetch_add_explicit(ctr, 0u, memory_order_relaxed)", false),
        ]
        for (label, ctrq, spin, coherent) in variants {
            let src = persistSrc.replacingOccurrences(of: "COHERENT", with: coherent ? "coherent(device)" : "")
                .replacingOccurrences(of: "CTRQUAL", with: ctrq).replacingOccurrences(of: "SPINLOAD", with: spin)
            print("\n# variant: \(label); data buffer coherent(device)=\(coherent)")
            guard let lib = try? compileLibrary(src, lang: 0x30002) else { print("  compile failed"); continue }
            let p = makePipeline(lib, "gbar")
            // (ntg, threadsPerTG, nbar, spinMax)
            let cases: [(Int, Int, UInt32, UInt32)] = [(16, 256, 1000, 200_000), (32, 256, 1000, 200_000), (64, 256, 1000, 200_000),
                                                       (16, 1024, 1000, 200_000), (48, 1024, 200, 200_000),
                                                       (64, 1024, 1, 100_000), (256, 1024, 1, 100_000)]
            for (ntg, tpt, nbar, spinMax) in cases where tpt <= p.maxTotalThreadsPerThreadgroup {
                var times: [Double] = []
                var to: UInt32 = 0, vis: UInt32 = 0
                for it in 0..<(nbar > 1 ? 6 : 2) {
                    memset(ctr.contents(), 0, 256); memset(stats.contents(), 0, 256); memset(data.contents(), 0, 1 << 20)
                    var prm = SIMD2<UInt32>(nbar, spinMax)
                    let cb = queue.makeCommandBuffer()!
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p)
                    e.setBuffer(ctr, offset: 0, index: 0); e.setBuffer(data, offset: 0, index: 1); e.setBuffer(stats, offset: 0, index: 2)
                    e.setBytes(&prm, length: 8, index: 3)
                    e.dispatchThreadgroups(MTLSize(width: ntg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpt, height: 1, depth: 1))
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    if let err = cb.error { print("  CB error: \(err)"); break }
                    let s = stats.contents().bindMemory(to: UInt32.self, capacity: 2)
                    to = max(to, s[0]); vis = max(vis, s[1])
                    if it >= 1 { times.append(gpuSeconds(cb)) }
                }
                if times.isEmpty { continue }
                let t = median(times)
                print(String(format: "ntg=%4d x %4d thr, %4u barriers: GPU %.1f us total, %.3f us/barrier | timeouts=%u visibility_violations=%u",
                             ntg, tpt, nbar, t * 1e6, t * 1e6 / Double(nbar), to, vis))
            }
        }
    }
}
