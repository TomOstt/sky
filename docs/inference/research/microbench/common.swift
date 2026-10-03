// common.swift - shared helpers for the R6 Metal microbenchmarks.
// Build: see README.md (swiftc -O -parse-as-library common.swift <bench>.swift -framework Metal)
import Foundation
import Metal

let device: MTLDevice = {
    guard let d = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
    return d
}()

/// MTLLanguageVersion raw values: (major << 16) | minor. 4.0 = 0x40000, 4.1 = 0x40001.
func compileLibrary(_ src: String, lang: UInt = 0x30002, fast: Bool = true) throws -> MTLLibrary {
    let o = MTLCompileOptions()
    guard let lv = MTLLanguageVersion(rawValue: lang) else { throw NSError(domain: "lang", code: Int(lang)) }
    o.languageVersion = lv
    o.mathMode = fast ? .fast : .safe
    return try device.makeLibrary(source: src, options: o)
}

func makePipeline(_ lib: MTLLibrary, _ name: String, icb: Bool = false,
                  tgSizeIsMultipleOfSIMD: Bool = true) -> MTLComputePipelineState {
    guard let f = lib.makeFunction(name: name) else { fatalError("no function \(name)") }
    let d = MTLComputePipelineDescriptor()
    d.computeFunction = f
    d.supportIndirectCommandBuffers = icb
    d.threadGroupSizeIsMultipleOfThreadExecutionWidth = tgSizeIsMultipleOfSIMD
    do {
        return try device.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
    } catch { fatalError("pipeline \(name): \(error)") }
}

@inline(__always) func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

func median(_ xs: [Double]) -> Double {
    precondition(!xs.isEmpty)
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
}
func minOf(_ xs: [Double]) -> Double { xs.min()! }
func fmt(_ x: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", x) }

/// GPU time of a completed command buffer in seconds (gpuEndTime - gpuStartTime).
func gpuSeconds(_ cb: MTLCommandBuffer) -> Double { cb.gpuEndTime - cb.gpuStartTime }

/// Run some heavy GPU work for `ms` milliseconds so DVFS ramps the GPU clock up before measuring.
func warmupGPU(queue: MTLCommandQueue, ms: Double = 300) {
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void spin(device float *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        float x = float(gid) * 1e-7f, y = 1.0001f;
        for (int i = 0; i < 4096; i++) { x = fma(x, y, 1e-7f); y = fma(y, 0.99999f, 1e-7f); }
        o[gid] = x + y;
    }
    """
    let lib = try! compileLibrary(src)
    let p = makePipeline(lib, "spin")
    let n = 1 << 20
    let buf = device.makeBuffer(length: n * 4, options: .storageModePrivate)!
    let t0 = nowNs()
    while Double(nowNs() - t0) / 1e6 < ms {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p)
        e.setBuffer(buf, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

func deviceSummary() -> String {
    var s = "device=\(device.name) arch=\(device.architecture.name) "
    s += "unified=\(device.hasUnifiedMemory) maxTG=\(device.maxThreadsPerThreadgroup.width) "
    s += "tgMem=\(device.maxThreadgroupMemoryLength) "
    s += "recommendedMaxWorkingSet=\(device.recommendedMaxWorkingSetSize) maxBuffer=\(device.maxBufferLength)"
    return s
}

func osVersionString() -> String {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
}
