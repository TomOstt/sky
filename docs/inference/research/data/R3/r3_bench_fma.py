# D'. Peak ALU probe: long FMA chains in a custom Metal kernel (fp32 and fp16),
# float4/half4 vectors x 8 independent accumulators (32 scalar FMAs per loop trip)
# to amortize loop overhead. Verifies the 16 x 128 x 2 x 1.398 GHz = 5.73 TFLOPS figure.
# Usage: python r3_bench_fma.py
import sys
import mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import env_header, med, time_evals

env_header("D_fma_probe_v2")
PEAK = 16 * 128 * 2 * 1.398e9
src = """
    uint tid = thread_position_in_grid.x;
    typedef vec<T, 4> V;
    V a0 = V(T(tid) * T(1e-7)), a1 = a0 + V(T(1e-3)), a2 = a0 + V(T(2e-3)), a3 = a0 + V(T(3e-3));
    V a4 = a0 + V(T(4e-3)), a5 = a0 + V(T(5e-3)), a6 = a0 + V(T(6e-3)), a7 = a0 + V(T(7e-3));
    V b = V(T(0.999)), c = V(T(1e-4));
    #pragma unroll(4)
    for (int i = 0; i < ITERS; i++) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    V s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[tid] = s.x + s.y + s.z + s.w;
"""
k = mx.fast.metal_kernel(name="fmaprobe4", input_names=["inp"], output_names=["out"], source=src)
dummy = mx.zeros((1,), dtype=mx.float32)
ITERS = 2048
for NT in [16 * 1024 * 4, 16 * 1024 * 16]:
    for dt in [mx.float32, mx.float16]:
        for tg in [256, 1024]:
            def run():
                o = k(inputs=[dummy], template=[("T", dt), ("ITERS", ITERS)], grid=(NT, 1, 1),
                      threadgroup=(tg, 1, 1), output_shapes=[(NT,)], output_dtypes=[dt])
                mx.eval(o)
            ts = time_evals(run, 9, warmup=3)
            fl = NT * ITERS * 32 * 2
            t = med(ts)
            print(f"threads={NT:7d} tg={tg:4d} {str(dt):<14s} {fl/t/1e12:6.3f} TFLOPS (best {fl/min(ts)/1e12:6.3f}) "
                  f"= {100*fl/t/PEAK:5.1f}% of {PEAK/1e12:.3f}  [t={t*1e3:.2f} ms]", flush=True)
