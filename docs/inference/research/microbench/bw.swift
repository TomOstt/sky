// bw.swift - GPU memory bandwidth microbenchmarks (read / copy / write / 4-bit GEMV-like streaming).
// Usage: bin/bw [quick]
// Every config: 2 warmup runs + 7 timed runs (1 dispatch per command buffer), GPU time = gpuEndTime-gpuStartTime.
import Foundation
import Metal

struct Variant { let type: String; let bytes: Int; let unroll: Int }

func genStreamKernels(_ vs: [Variant]) -> String {
    var s = "#include <metal_stdlib>\nusing namespace metal;\n"
    s += "inline float hsum(float x){return x;} inline float hsum(float2 x){return x.x+x.y;} inline float hsum(float4 x){return x.x+x.y+x.z+x.w;}\n"
    for v in vs {
        let n = "\(v.type)_u\(v.unroll)"
        var loads = "", stores = ""
        for u in 0..<v.unroll { loads += " acc += src[i + \(u) * g];" ; stores += " dst[i + \(u) * g] = src[i + \(u) * g];" }
        var writes = ""
        for u in 0..<v.unroll { writes += " dst[i + \(u) * g] = val;" }
        s += """
        kernel void read_\(n)(device const \(v.type) *src [[buffer(0)]], device float *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                              uint gid [[thread_position_in_grid]], uint g [[threads_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
          \(v.type) acc = \(v.type)(0);
          uint i = gid;
          for (; i + \(v.unroll - 1) * g < n; i += \(v.unroll) * g) {\(loads) }
          for (; i < n; i += g) acc += src[i];
          float r = simd_sum(hsum(acc));
          if (lane == 0) out[gid / 32] = r;
        }
        kernel void copy_\(n)(device const \(v.type) *src [[buffer(0)]], device \(v.type) *dst [[buffer(1)]], constant uint &n [[buffer(2)]],
                              uint gid [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
          uint i = gid;
          for (; i + \(v.unroll - 1) * g < n; i += \(v.unroll) * g) {\(stores) }
          for (; i < n; i += g) dst[i] = src[i];
        }
        kernel void write_\(n)(device \(v.type) *dst [[buffer(0)]], constant uint &n [[buffer(1)]],
                               uint gid [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
          const \(v.type) val = \(v.type)(float(gid & 7));
          uint i = gid;
          for (; i + \(v.unroll - 1) * g < n; i += \(v.unroll) * g) {\(writes) }
          for (; i < n; i += g) dst[i] = val;
        }

        """
    }
    return s
}

// 4-bit affine quantized GEMV (MLX-like layout): W[rows][cols/8] uint32 (8 nibbles, low first),
// scales/biases [rows][cols/64] half, x[cols] half, y[rows] float.
// Each simdgroup handles R consecutive rows; each lane handles VPT values per row per iteration.
func genGemvKernel(R: Int, VPT: Int, NSG: Int) -> String {
    let words = VPT / 8
    let wtype = words == 1 ? "uint" : "uint\(words)"
    var unpack = ""
    for j in 0..<words {
        let wj = words == 1 ? "w" : "w[\(j)]"
        for k in 0..<8 { unpack += "d += float((\(wj) >> \(4 * k)) & 0xFu) * xv[\(j * 8 + k)]; " }
    }
    return """
    kernel void gemv_q4_r\(R)_v\(VPT)_s\(NSG)(device const uint *W [[buffer(0)]], device const half *S [[buffer(1)]], device const half *B [[buffer(2)]],
                        device const half *x [[buffer(3)]], device float *y [[buffer(4)]], constant uint2 &dims [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
      const uint rows = dims.x, cols = dims.y;
      const uint row0 = (tgid * \(NSG) + sgid) * \(R);
      if (row0 >= rows) return;
      const uint wpr = cols / 8, gpr = cols / 64;
      float acc[\(R)];
      for (int r = 0; r < \(R); r++) acc[r] = 0.0f;
      for (uint c = lane * \(VPT); c < cols; c += 32 * \(VPT)) {
        float xv[\(VPT)]; float xs = 0.0f;
        for (int i = 0; i < \(VPT); i += 4) { half4 h = *((device const half4 *)(x + c + i)); float4 f = float4(h);
          xv[i] = f.x; xv[i+1] = f.y; xv[i+2] = f.z; xv[i+3] = f.w; xs += f.x + f.y + f.z + f.w; }
        for (int r = 0; r < \(R); r++) {
          const uint row = row0 + r;
          \(wtype) w = *((device const \(wtype) *)(W + row * wpr + c / 8));
          const float s = float(S[row * gpr + c / 64]);
          const float b = float(B[row * gpr + c / 64]);
          float d = 0.0f;
          \(unpack)
          acc[r] += s * d + b * xs;
        }
      }
      for (int r = 0; r < \(R); r++) { float v = simd_sum(acc[r]); if (lane == 0) y[row0 + r] = v; }
    }

    """
}

func timeKernel(queue: MTLCommandQueue, runs: Int = 7, warm: Int = 2, encode: (MTLComputeCommandEncoder) -> Void) -> [Double] {
    var ts: [Double] = []
    for it in 0..<(warm + runs) {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        encode(e)
        e.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let err = cb.error { fatalError("cb error \(err)") }
        if it >= warm { ts.append(gpuSeconds(cb)) }
    }
    return ts
}

@main struct BW {
    static func main() {
        let quick = CommandLine.arguments.contains("quick")
        print("R6 bw  \(osVersionString())  \(Date())")
        print(deviceSummary())
        let queue = device.makeCommandQueue()!
        let variants = [Variant(type: "float", bytes: 4, unroll: 1), Variant(type: "float", bytes: 4, unroll: 4),
                        Variant(type: "float2", bytes: 8, unroll: 4), Variant(type: "float4", bytes: 16, unroll: 1),
                        Variant(type: "float4", bytes: 16, unroll: 2), Variant(type: "float4", bytes: 16, unroll: 4),
                        Variant(type: "float4", bytes: 16, unroll: 8)]
        var src = genStreamKernels(variants)
        let gemvCfgs: [(Int, Int, Int)] = [(1, 16, 2), (2, 16, 2), (4, 16, 2), (8, 16, 2), (4, 32, 2), (4, 16, 4), (4, 16, 8), (8, 32, 2), (2, 32, 4), (1, 32, 8)]
        for (r, v, s) in gemvCfgs { src += genGemvKernel(R: r, VPT: v, NSG: s) }
        let lib: MTLLibrary
        do { lib = try compileLibrary(src) } catch { print(error); exit(1) }

        let bytes = 1 << 30   // 1 GiB
        let shared = device.makeBuffer(length: bytes, options: .storageModeShared)!
        let priv = device.makeBuffer(length: bytes, options: .storageModePrivate)!
        let out = device.makeBuffer(length: 64 << 20, options: .storageModePrivate)!
        do { // initialize (touch) all pages on the GPU
            let cb = queue.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!
            b.fill(buffer: shared, range: 0..<bytes, value: 1); b.fill(buffer: priv, range: 0..<bytes, value: 1)
            b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        warmupGPU(queue: queue, ms: 500)

        if CommandLine.arguments.contains("confirm") {
            // Interleaved re-check of scalar vs vector loads (rules out ordering/thermal artifacts in the sweep)
            print("\n# CONFIRM: interleaved rounds, 1 GiB private read, median of 7 runs per config per round")
            let cfgs: [(String, Int, Int)] = [("float_u4", 512, 16384), ("float4_u4", 256, 1048576), ("float4_u1", 128, 16384),
                                               ("float_u1", 512, 32768), ("float4_u8", 128, 1048576), ("float2_u4", 256, 1048576)]
            var acc: [String: [Double]] = [:]
            for round in 0..<5 {
                for (v, tg, g) in cfgs {
                    let p = makePipeline(lib, "read_\(v)")
                    let bytesPer = v.hasPrefix("float4") ? 16 : (v.hasPrefix("float2") ? 8 : 4)
                    var n = UInt32(bytes / bytesPer)
                    let ts = timeKernel(queue: queue) { e in
                        e.setComputePipelineState(p); e.setBuffer(priv, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
                        e.setBytes(&n, length: 4, index: 2)
                        e.dispatchThreads(MTLSize(width: g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                    }
                    let med = median(ts.map { Double(bytes) / $0 / 1e9 })
                    acc["\(v) tg=\(tg) grid=\(g)", default: []].append(med)
                    print("round \(round) \(v) tg=\(tg) grid=\(g): \(fmt(med, 1)) GB/s")
                }
            }
            for (k, xs) in acc.sorted(by: { $0.key < $1.key }) { print("CONFIRM \(k): median of rounds \(fmt(median(xs), 1)) GB/s (min \(fmt(xs.min()!, 1)) max \(fmt(xs.max()!, 1)))") }
            return
        }

        // ---------- READ sweep ----------
        print("\n# READ 1 GiB: variant tg grid -> median GB/s (min..max) [storage]")
        let tgs = quick ? [256, 1024] : [64, 128, 256, 512, 1024]
        let grids = quick ? [65536, 262144] : [16384, 32768, 65536, 131072, 262144, 1048576]
        var bestRead = (0.0, "")
        for (bufName, buf) in [("private", priv), ("shared", shared)] {
            for v in variants where bufName == "private" || v.type == "float4" {
                let p = makePipeline(lib, "read_\(v.type)_u\(v.unroll)")
                var n = UInt32(bytes / v.bytes)
                for tg in tgs where tg <= p.maxTotalThreadsPerThreadgroup {
                    for g in grids {
                        let ts = timeKernel(queue: queue) { e in
                            e.setComputePipelineState(p); e.setBuffer(buf, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
                            e.setBytes(&n, length: 4, index: 2)
                            e.dispatchThreads(MTLSize(width: g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                        }
                        let gbs = ts.map { Double(bytes) / $0 / 1e9 }
                        let med = median(gbs)
                        print("read \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g) -> \(fmt(med, 1)) GB/s (\(fmt(gbs.min()!, 1))..\(fmt(gbs.max()!, 1))) [\(bufName)]")
                        if med > bestRead.0 { bestRead = (med, "read \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g) [\(bufName)]") }
                    }
                }
            }
        }
        print("BEST READ: \(fmt(bestRead.0, 1)) GB/s  \(bestRead.1)")

        // ---------- COPY / WRITE ----------
        print("\n# COPY 512 MiB -> 512 MiB (GB/s counts read+write bytes) and WRITE 1 GiB")
        let half = bytes / 2
        var bestCopy = (0.0, ""), bestWrite = (0.0, "")
        for v in variants where v.type == "float4" || (v.type == "float" && v.unroll == 4) {
            let pc = makePipeline(lib, "copy_\(v.type)_u\(v.unroll)")
            let pw = makePipeline(lib, "write_\(v.type)_u\(v.unroll)")
            for tg in (quick ? [256] : [256, 1024]) {
                for g in (quick ? [262144] : [65536, 262144, 1048576]) {
                    var nc = UInt32(half / v.bytes)
                    let tc = timeKernel(queue: queue) { e in
                        e.setComputePipelineState(pc); e.setBuffer(priv, offset: 0, index: 0); e.setBuffer(priv, offset: half, index: 1)
                        e.setBytes(&nc, length: 4, index: 2)
                        e.dispatchThreads(MTLSize(width: g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                    }
                    let cg = tc.map { Double(2 * half) / $0 / 1e9 }
                    print("copy \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g) -> \(fmt(median(cg), 1)) GB/s (\(fmt(cg.min()!, 1))..\(fmt(cg.max()!, 1)))")
                    if median(cg) > bestCopy.0 { bestCopy = (median(cg), "copy \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g)") }
                    var nw = UInt32(bytes / v.bytes)
                    let tw = timeKernel(queue: queue) { e in
                        e.setComputePipelineState(pw); e.setBuffer(priv, offset: 0, index: 0)
                        e.setBytes(&nw, length: 4, index: 1)
                        e.dispatchThreads(MTLSize(width: g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                    }
                    let wg = tw.map { Double(bytes) / $0 / 1e9 }
                    print("write \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g) -> \(fmt(median(wg), 1)) GB/s (\(fmt(wg.min()!, 1))..\(fmt(wg.max()!, 1)))")
                    if median(wg) > bestWrite.0 { bestWrite = (median(wg), "write \(v.type)_u\(v.unroll) tg=\(tg) grid=\(g)") }
                }
            }
        }
        // blit copy for reference
        do {
            var ts: [Double] = []
            for it in 0..<9 {
                let cb = queue.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!
                b.copy(from: priv, sourceOffset: 0, to: priv, destinationOffset: half, size: half)
                b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                if it >= 2 { ts.append(gpuSeconds(cb)) }
            }
            let g = ts.map { Double(2 * half) / $0 / 1e9 }
            print("blit copy 512 MiB -> \(fmt(median(g), 1)) GB/s (\(fmt(g.min()!, 1))..\(fmt(g.max()!, 1)))")
        }
        print("BEST COPY: \(fmt(bestCopy.0, 1)) GB/s  \(bestCopy.1)")
        print("BEST WRITE: \(fmt(bestWrite.0, 1)) GB/s  \(bestWrite.1)")

        // ---------- READ vs working-set size (cache effects) ----------
        print("\n# READ vs size (float4_u4, tg=256, grid=min(n/4, 262144)), 15 runs each")
        let pr = makePipeline(lib, "read_float4_u4")
        let sizesMiB: [Int] = [1, 4, 8, 16, 24, 32, 48, 64, 128, 256, 1024]
        for mib in sizesMiB {
            let sz: Int = mib << 20
            var n = UInt32(sz / 16)
            let g = min(Int(n) / 4, 262144)
            // many dispatches per command buffer to amortize CB overhead for small sizes
            let reps = max(1, (64 << 20) / sz)
            var ts: [Double] = []
            for it in 0..<17 {
                let cb = queue.makeCommandBuffer()!
                let e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(pr); e.setBuffer(priv, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
                e.setBytes(&n, length: 4, index: 2)
                for _ in 0..<reps { e.dispatchThreads(MTLSize(width: g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                if it >= 2 { ts.append(gpuSeconds(cb) / Double(reps)) }
            }
            let gb = ts.map { Double(sz) / $0 / 1e9 }
            print("size \(sz >> 20) MiB reps=\(reps) -> \(fmt(median(gb), 1)) GB/s (\(fmt(gb.min()!, 1))..\(fmt(gb.max()!, 1)))  per-dispatch \(fmt(median(ts) * 1e6, 1)) us")
        }

        // ---------- 4-bit GEMV-like ----------
        print("\n# GEMV q4 (MLX-like affine 4-bit, group 64, fp16 scale+bias): rows x cols -> effective GB/s over weight+scale+bias bytes")
        let shapes: [(Int, Int, String)] = [(25600, 5120, "Qwen3-32B up/gate-like"), (5120, 25600, "Qwen3-32B down-like"),
                                            (131072, 5120, "large 0.36 GB")]
        for (rows, cols, label) in shapes {
            let wBytes = rows * cols / 2, sBytes = rows * (cols / 64) * 2
            let W = device.makeBuffer(length: wBytes, options: .storageModePrivate)!
            let Sb = device.makeBuffer(length: sBytes, options: .storageModePrivate)!
            let Bb = device.makeBuffer(length: sBytes, options: .storageModePrivate)!
            let X = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let Y = device.makeBuffer(length: rows * 4, options: .storageModePrivate)!
            let xp = X.contents().bindMemory(to: UInt16.self, capacity: cols)
            for i in 0..<cols { xp[i] = Float16(Float(i % 13) * 0.01).bitPattern }
            do {
                let cb = queue.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!
                b.fill(buffer: W, range: 0..<wBytes, value: 0x5A); b.fill(buffer: Sb, range: 0..<sBytes, value: 0x3C); b.fill(buffer: Bb, range: 0..<sBytes, value: 0x00)
                b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
            let total = Double(wBytes + 2 * sBytes + cols * 2 + rows * 4)
            for (r, v, s) in gemvCfgs {
                let p = makePipeline(lib, "gemv_q4_r\(r)_v\(v)_s\(s)")
                var dims = SIMD2<UInt32>(UInt32(rows), UInt32(cols))
                let tgs = (rows + r * s - 1) / (r * s)
                let ts = timeKernel(queue: queue, runs: 9) { e in
                    e.setComputePipelineState(p)
                    e.setBuffer(W, offset: 0, index: 0); e.setBuffer(Sb, offset: 0, index: 1); e.setBuffer(Bb, offset: 0, index: 2)
                    e.setBuffer(X, offset: 0, index: 3); e.setBuffer(Y, offset: 0, index: 4); e.setBytes(&dims, length: 8, index: 5)
                    e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * s, height: 1, depth: 1))
                }
                let gb = ts.map { total / $0 / 1e9 }
                print("gemv \(label) \(rows)x\(cols) R=\(r) VPT=\(v) SG/TG=\(s): \(fmt(median(ts) * 1e6, 1)) us -> \(fmt(median(gb), 1)) GB/s (\(fmt(gb.min()!, 1))..\(fmt(gb.max()!, 1)))")
            }
        }
    }
}
