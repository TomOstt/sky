# B. Decode GEMV: mx.quantized_matmul with M=1 (4-bit, group 64, affine, transpose=True)
# Variant (i): one call per mx.eval (synchronous latency incl. launch + sync)
# Variant (ii): BATCH independent calls in one mx.eval (amortized; ~GPU throughput)
# Weights rotate through enough distinct copies (>= ~320 MB working set) so that
# every call streams its weights from DRAM, not from the 24 MB SLC.
# Usage: python r3_bench_qmv.py [fp16]
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, make_qweight, med

env_header("B_qmv")
GS, BITS = 64, 4
BATCH = 24
REPS_SINGLE = 200
REPS_BATCH = 25
WS_TARGET = 320e6

SHAPES = [
    ("8B q/o_proj", 4096, 4096),
    ("8B k/v_proj", 4096, 1024),
    ("8B gate/up", 4096, 12288),
    ("8B down", 12288, 4096),
    ("8B lm_head", 4096, 151936),
    ("32B q_proj", 5120, 8192),
    ("32B k/v_proj", 5120, 1024),
    ("32B o_proj", 8192, 5120),
    ("32B gate/up", 5120, 25600),
    ("32B down", 25600, 5120),
    ("32B lm_head", 5120, 151936),
]

do_fp16 = len(sys.argv) > 1 and sys.argv[1] == "fp16"


def bench(name, K, N, make, call, nbytes_one):
    ncopies = max(2, int(-(-WS_TARGET // nbytes_one)))
    ncopies = min(ncopies, 400)
    ws = [make(i) for i in range(ncopies)]
    x = (mx.random.normal((1, K)) * 0.1).astype(mx.float16)
    mx.eval(x)
    # warmup
    for i in range(10):
        mx.eval(call(x, ws[i % ncopies]))
    # (i) one call per eval
    ts = []
    for it in range(REPS_SINGLE):
        w = ws[it % ncopies]
        t0 = time.perf_counter()
        y = call(x, w)
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    t1 = med(ts)
    # (ii) BATCH independent calls per eval
    tb = []
    j = 0
    for it in range(REPS_BATCH + 2):
        sel = [ws[(j + b) % ncopies] for b in range(BATCH)]
        j += BATCH
        t0 = time.perf_counter()
        ys = [call(x, w) for w in sel]
        mx.eval(ys)
        dt = time.perf_counter() - t0
        if it >= 2:
            tb.append(dt / BATCH)
    t2 = med(tb)
    print(f"{name:<14s} K={K:6d} N={N:6d} MB={nbytes_one/1e6:8.2f} copies={ncopies:3d} | "
          f"single: {t1*1e6:9.1f} us {nbytes_one/t1/GB:6.1f} GB/s | "
          f"batched x{BATCH}: {t2*1e6:9.1f} us {nbytes_one/t2/GB:6.1f} GB/s "
          f"(min {min(tb)*1e6:8.1f} us)", flush=True)
    del ws
    mx.clear_cache()
    return t1, t2


if not do_fp16:
    # correctness spot check vs dequantized reference on one shape
    wq, sc, bi = make_qweight(1024, 4096, seed=123)
    xx = (mx.random.normal((1, 4096)) * 0.1).astype(mx.float16)
    ref = xx.astype(mx.float32) @ mx.dequantize(wq, sc, bi, group_size=GS, bits=BITS).astype(mx.float32).T
    got = mx.quantized_matmul(xx, wq, sc, bi, transpose=True, group_size=GS, bits=BITS)
    print("# correctness max|err| =", mx.max(mx.abs(got.astype(mx.float32) - ref)).item(),
          " max|ref| =", mx.max(mx.abs(ref)).item())
    for name, K, N in SHAPES:
        nb = sum(a.nbytes for a in make_qweight(8, K))  # per-8-rows, scale up
        nbytes = nb * N // 8

        def make(i, K=K, N=N):
            return make_qweight(N, K, seed=i)

        def call(x, w):
            return mx.quantized_matmul(x, *w, transpose=True, group_size=GS, bits=BITS)

        bench(name, K, N, make, call, nbytes)
else:
    for name, K, N in SHAPES:
        if N == 151936:
            continue
        nbytes = N * K * 2

        def make(i, K=K, N=N):
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
            mx.eval(w)
            return w

        def call(x, w):
            return x @ w.T

        bench("fp16 " + name, K, N, make, call, nbytes)

print("peak_memory_GB", mx.get_peak_memory() / GB)
