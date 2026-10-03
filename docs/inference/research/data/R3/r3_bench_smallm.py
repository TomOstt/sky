# C. Verification shapes: quantized_matmul (4-bit, g64, affine, transpose=True)
# on Qwen3-32B shapes with M = 1..32 rows. Reports per-call time (single-eval and
# batched-eval) and the ratio vs M=1. Weights rotate through distinct copies
# (>= ~320 MB working set) so weights stream from DRAM.
# Usage: python r3_bench_smallm.py
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, make_qweight, med

env_header("C_small_M")
GS, BITS = 64, 4
BATCH = 16
REPS_SINGLE = 60
REPS_BATCH = 10
WS_TARGET = 320e6
MS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]

SHAPES = [
    ("32B q_proj", 5120, 8192),
    ("32B k/v_proj", 5120, 1024),
    ("32B o_proj", 8192, 5120),
    ("32B gate/up", 5120, 25600),
    ("32B down", 25600, 5120),
    ("32B lm_head", 5120, 151936),
]
only = sys.argv[1:]

for name, K, N in SHAPES:
    if only and name not in only:
        continue
    nb = sum(a.nbytes for a in make_qweight(8, K)) * N // 8
    ncopies = min(400, max(2, int(-(-WS_TARGET // nb))))
    ws = [make_qweight(N, K, seed=i) for i in range(ncopies)]
    base1 = base2 = None
    for M in MS:
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.float16)
        mx.eval(x)

        def call(w):
            return mx.quantized_matmul(x, *w, transpose=True, group_size=GS, bits=BITS)

        for i in range(6):
            mx.eval(call(ws[i % ncopies]))
        ts = []
        for it in range(REPS_SINGLE):
            w = ws[it % ncopies]
            t0 = time.perf_counter()
            mx.eval(call(w))
            ts.append(time.perf_counter() - t0)
        t1 = med(ts)
        tb = []
        j = 0
        for it in range(REPS_BATCH + 2):
            sel = [ws[(j + b) % ncopies] for b in range(BATCH)]
            j += BATCH
            t0 = time.perf_counter()
            mx.eval([call(w) for w in sel])
            dt = time.perf_counter() - t0
            if it >= 2:
                tb.append(dt / BATCH)
        t2 = med(tb)
        if M == 1:
            base1, base2 = t1, t2
        print(f"{name:<13s} K={K:6d} N={N:6d} M={M:3d} | single {t1*1e6:9.1f} us (x{t1/base1:5.2f}) | "
              f"batched {t2*1e6:9.1f} us (x{t2/base2:5.2f}) {nb/t2/GB:6.1f} GB/s-weights",
              flush=True)
    del ws
    mx.clear_cache()

print("peak_memory_GB", mx.get_peak_memory() / GB)
