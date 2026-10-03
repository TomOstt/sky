// mtl4.cpp - Metal 4 (MTL4*) vs Metal 3 dispatch overhead, using Apple's metal-cpp headers
// (github.com/apple/metal-cpp, tag release/metal-cpp_macOS26_iOS26) because the local SDK (15.4) has no MTL4 headers.
// Build: see README.md.  N dependent tiny dispatches (1 TG x 32 threads, buf[i] += 1).
//   MTL4: one MTL4CommandBuffer, one MTL4ComputeCommandEncoder, argument table, residency set,
//         intra-pass barrier (barrierAfterEncoderStages dispatch->dispatch, visibility device) between dispatches,
//         GPU time from MTL4CommitFeedback (GPUStartTime/GPUEndTime), completion via MTLSharedEvent.
//   MTL3: same work through MTLCommandQueue (serial encoder, and concurrent encoder + memoryBarrier).
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <time.h>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device uint *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) { buf[tid] += 1u; }
kernel void spin(device float *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float x = float(gid) * 1e-7f, y = 1.0001f;
  for (int i = 0; i < 4096; i++) { x = fma(x, y, 1e-7f); y = fma(y, 0.99999f, 1e-7f); }
  o[gid] = x + y;
}
)";

static inline uint64_t nowNs() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static NS::String* S(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    printf("R6 mtl4  device=%s  metal4=%d\n", dev->name()->utf8String(), (int)dev->supportsFamily((MTL::GPUFamily)5002));
    NS::Error* err = nullptr;
    MTL::CompileOptions* co = MTL::CompileOptions::alloc()->init();
    co->setLanguageVersion((MTL::LanguageVersion)0x30002);
    MTL::Library* lib = dev->newLibrary(S(kSrc), co, &err);
    if (!lib) { printf("compile error %s\n", err->localizedDescription()->utf8String()); return 1; }
    MTL::Function* fn = lib->newFunction(S("tiny"));
    MTL::Function* fnSpin = lib->newFunction(S("spin"));
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &err);
    MTL::ComputePipelineState* psoSpin = dev->newComputePipelineState(fnSpin, &err);
    MTL::Buffer* buf = dev->newBuffer(1 << 16, MTL::ResourceStorageModePrivate);
    MTL::Buffer* spinBuf = dev->newBuffer(4 << 20, MTL::ResourceStorageModePrivate);

    // ---------------- Metal 4 objects ----------------
    MTL4::CommandQueue* q4 = dev->newMTL4CommandQueue();
    MTL4::CommandAllocator* alloc = dev->newCommandAllocator();
    MTL4::CommandBuffer* cb4 = dev->newCommandBuffer();
    MTL4::ArgumentTableDescriptor* atd = MTL4::ArgumentTableDescriptor::alloc()->init();
    atd->setMaxBufferBindCount(4);
    MTL4::ArgumentTable* at = dev->newArgumentTable(atd, &err);
    if (!q4 || !alloc || !cb4 || !at) { printf("MTL4 object creation failed\n"); return 1; }
    MTL::ResidencySetDescriptor* rsd = MTL::ResidencySetDescriptor::alloc()->init();
    MTL::ResidencySet* rs = dev->newResidencySet(rsd, &err);
    rs->addAllocation(buf); rs->addAllocation(spinBuf); rs->commit(); rs->requestResidency();
    q4->addResidencySet(rs);
    MTL::SharedEvent* ev = dev->newSharedEvent();
    uint64_t evVal = 0;

    struct R { double gpu, cpu, commit, wall; };
    auto runMTL4 = [&](int n, bool barrier, MTL::ComputePipelineState* p, MTL::Buffer* b, MTL::Size tg, MTL::Size tpt) -> R {
        uint64_t t0 = nowNs();
        alloc->reset();  // safe: previous submission completed (we wait on the event below)
        cb4->beginCommandBuffer(alloc);
        MTL4::ComputeCommandEncoder* enc = cb4->computeCommandEncoder();
        enc->setComputePipelineState(p);
        at->setAddress(b->gpuAddress(), 0);
        enc->setArgumentTable(at);
        for (int i = 0; i < n; i++) {
            if (i > 0 && barrier) enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            enc->dispatchThreadgroups(tg, tpt);
        }
        enc->endEncoding();
        cb4->endCommandBuffer();
        uint64_t t1 = nowNs();
        std::atomic<bool> fbDone{false};
        double gs = 0, ge = 0;
        MTL4::CommitOptions* opts = MTL4::CommitOptions::alloc()->init();
        opts->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
            gs = fb->GPUStartTime(); ge = fb->GPUEndTime();
            if (fb->error()) printf("feedback error: %s\n", fb->error()->localizedDescription()->utf8String());
            fbDone.store(true);
        });
        const MTL4::CommandBuffer* cbs[1] = {cb4};
        q4->commit(cbs, 1, opts);
        q4->signalEvent(ev, ++evVal);
        uint64_t t2 = nowNs();
        while (ev->signaledValue() < evVal) {}
        uint64_t t3 = nowNs();
        while (!fbDone.load()) {}
        opts->release();
        return R{ge - gs, (t1 - t0) * 1e-9, (t2 - t1) * 1e-9, (t3 - t2) * 1e-9};
    };

    // ---------------- Metal 3 objects ----------------
    MTL::CommandQueue* q3 = dev->newCommandQueue();
    auto runMTL3 = [&](int n, bool concurrentBarrier) -> R {
        uint64_t t0 = nowNs();
        MTL::CommandBuffer* cb = q3->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder(concurrentBarrier ? MTL::DispatchTypeConcurrent : MTL::DispatchTypeSerial);
        enc->setComputePipelineState(pso);
        enc->setBuffer(buf, 0, 0);
        for (int i = 0; i < n; i++) {
            if (i > 0 && concurrentBarrier) enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
        }
        enc->endEncoding();
        uint64_t t1 = nowNs();
        cb->commit();
        uint64_t t2 = nowNs();
        cb->waitUntilCompleted();
        uint64_t t3 = nowNs();
        return R{cb->GPUEndTime() - cb->GPUStartTime(), (t1 - t0) * 1e-9, (t2 - t1) * 1e-9, (t3 - t2) * 1e-9};
    };

    // warm up GPU clocks via MTL4 path
    {
        uint64_t t0 = nowNs();
        while (nowNs() - t0 < 400000000ull) runMTL4(1, false, psoSpin, spinBuf, MTL::Size(4096, 1, 1), MTL::Size(256, 1, 1));
    }

    printf("\n# N dependent tiny dispatches (1 TG x 32 thr). medians of 9 reps after 3 warmups. times in us\n");
    printf("%-34s %5s %10s %10s %10s %10s %10s\n", "path", "N", "GPU", "GPU/disp", "CPUenc", "CPU/disp", "commit");
    for (int n : {1, 10, 100, 1000}) {
        for (int mode = 0; mode < 4; mode++) {
            std::vector<double> g, c, cm, w;
            for (int it = 0; it < 12; it++) {
                NS::AutoreleasePool* p2 = NS::AutoreleasePool::alloc()->init();
                R r = mode == 0 ? runMTL4(n, true, pso, buf, MTL::Size(1, 1, 1), MTL::Size(32, 1, 1))
                    : mode == 1 ? runMTL4(n, false, pso, buf, MTL::Size(1, 1, 1), MTL::Size(32, 1, 1))
                    : mode == 2 ? runMTL3(n, false) : runMTL3(n, true);
                p2->release();
                if (it >= 3) { g.push_back(r.gpu); c.push_back(r.cpu); cm.push_back(r.commit); w.push_back(r.wall); }
            }
            const char* name = mode == 0 ? "MTL4 encoder + intra-pass barriers" : mode == 1 ? "MTL4 encoder, no barriers"
                             : mode == 2 ? "MTL3 serial encoder" : "MTL3 concurrent + memoryBarrier";
            double G = median(g) * 1e6, C = median(c) * 1e6;
            printf("%-34s %5d %10.1f %10.3f %10.1f %10.3f %10.1f\n", name, n, G, G / n, C, C / n, median(cm) * 1e6);
        }
    }

    // Submission latency for MTL4: empty-ish command buffer (1 tiny dispatch), commit -> event observed on CPU
    {
        std::vector<double> w;
        for (int it = 0; it < 45; it++) {
            R r = runMTL4(1, false, pso, buf, MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
            if (it >= 5) w.push_back(r.wall);
        }
        std::sort(w.begin(), w.end());
        printf("\nMTL4 1-dispatch CB: commit+signalEvent -> CPU spin sees event: median %.1f us p10 %.1f p90 %.1f min %.1f\n",
               median(w) * 1e6, w[w.size() / 10] * 1e6, w[w.size() * 9 / 10] * 1e6, w[0] * 1e6);
    }
    pool->release();
    return 0;
}
