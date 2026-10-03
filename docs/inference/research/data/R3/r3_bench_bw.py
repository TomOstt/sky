# A. Achievable memory bandwidth as seen through MLX (reductions and copies).
# Usage: python r3_bench_bw.py
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, med, time_evals

env_header("A_bandwidth")
REPS = 15


def report(label, nbytes, ts):
    t = med(ts)
    print(f"{label:<44s} bytes={nbytes/GB:7.3f}GB  med={t*1e3:9.3f} ms  "
          f"min={min(ts)*1e3:9.3f} ms  BW_med={nbytes/t/GB:7.1f} GB/s  BW_best={nbytes/min(ts)/GB:7.1f} GB/s",
          flush=True)


# --- Reductions (read-only): sweep sizes to expose the SLC/cache cliff ---
for mb in [16, 64, 256, 1024, 2048]:
    n = mb * (1 << 20) // 2  # fp16 elements
    x = mx.full((n,), 0.5, dtype=mx.float16)
    mx.eval(x)
    nbytes = x.nbytes
    ts = time_evals(lambda: mx.eval(mx.sum(x)), REPS)
    report(f"all-reduce sum fp16 {mb} MiB", nbytes, ts)
    if mb >= 256:
        ts = time_evals(lambda: mx.eval(mx.max(x)), REPS)
        report(f"all-reduce max fp16 {mb} MiB", nbytes, ts)
        xr = x.reshape(-1, 4096)
        ts = time_evals(lambda: mx.eval(mx.sum(xr, axis=-1)), REPS)
        report(f"row-reduce sum fp16 {mb} MiB (rows of 4096)", nbytes, ts)
    del x
    mx.clear_cache()

# --- Copies (read N + write N) ---
for mb in [64, 256, 1024]:
    n = mb * (1 << 20) // 2
    x = mx.full((n,), 0.5, dtype=mx.float16)
    mx.eval(x)
    nbytes = 2 * x.nbytes
    ts = time_evals(lambda: mx.eval(x * 2), REPS)
    report(f"elementwise x*2 fp16 {mb} MiB (R+W)", nbytes, ts)
    ts = time_evals(lambda: mx.eval(mx.contiguous(x[::-1])), REPS)
    report(f"reversed copy fp16 {mb} MiB (R+W)", nbytes, ts)
    del x
    mx.clear_cache()

print("peak_memory_GB", mx.get_peak_memory() / GB)
