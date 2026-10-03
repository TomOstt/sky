# D. Prefill throughput on Qwen3-32B MLP shapes, M = 512 and 2048.
# Quantized (4-bit g64 affine, transpose=True) vs fp16 dense (x @ W.T).
# Plus a peak-FLOPS probe: custom Metal FMA kernel (fp32 and fp16) to verify
# the theoretical 16 cores x 128 ALU x 2 x 1.398 GHz = 5.73 TFLOPS.
# Usage: python r3_bench_prefill.py
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, make_qweight, med, time_evals

env_header("D_prefill")
GS, BITS = 64, 4
REPS = 7
PEAK_THEORY = 16 * 128 * 2 * 1.398e9

# ---- peak FLOPS probe ----
src = """
    uint tid = thread_position_in_grid.x;
    T a0 = T(tid) * T(1e-7), a1 = a0 + T(1e-3), a2 = a0 + T(2e-3), a3 = a0 + T(3e-3);
    T a4 = a0 + T(4e-3), a5 = a0 + T(5e-3), a6 = a0 + T(6e-3), a7 = a0 + T(7e-3);
    T b = T(0.999), c = T(1e-4);
    for (int i = 0; i < ITERS; i++) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    out[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
"""
kern = mx.fast.metal_kernel(name="fmaprobe", input_names=["inp"], output_names=["out"], source=src)
ITERS = 4096
NT = 16 * 1024 * 16
dummy = mx.zeros((1,), dtype=mx.float32)
for dt in [mx.float32, mx.float16]:
    def run():
        o = kern(inputs=[dummy], template=[("T", dt), ("ITERS", ITERS)], grid=(NT, 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(NT,)], output_dtypes=[dt])
        mx.eval(o)
    ts = time_evals(run, REPS, warmup=3)
    flops = NT * ITERS * 8 * 2
    t = med(ts)
    print(f"FMA probe {str(dt):<14s} {flops/t/1e12:6.3f} TFLOPS (best {flops/min(ts)/1e12:6.3f}); "
          f"theory {PEAK_THEORY/1e12:5.3f} -> {100*flops/t/PEAK_THEORY:5.1f}%", flush=True)

# ---- prefill GEMMs ----
SHAPES = [("32B gate/up", 5120, 25600), ("32B down", 25600, 5120)]
for name, K, N in SHAPES:
    wq = make_qweight(N, K, seed=7)
    w16 = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
    mx.eval(w16)
    for M in [512, 2048]:
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.float16)
        mx.eval(x)
        flops = 2.0 * M * K * N
        tq = time_evals(lambda: mx.eval(mx.quantized_matmul(x, *wq, transpose=True, group_size=GS, bits=BITS)), REPS)
        tf = time_evals(lambda: mx.eval(x @ w16.T), REPS)
        for lab, ts in [("q4 g64", tq), ("fp16", tf)]:
            t = med(ts)
            print(f"{name:<12s} M={M:5d} K={K:6d} N={N:6d} {lab:<7s} med {t*1e3:8.2f} ms  "
                  f"{flops/t/1e12:6.3f} TFLOPS ({100*flops/t/PEAK_THEORY:5.1f}% of theory)  "
                  f"tok/s-equiv {M/t:9.0f}", flush=True)
    del wq, w16
    mx.clear_cache()
print("peak_memory_GB", mx.get_peak_memory() / GB)
