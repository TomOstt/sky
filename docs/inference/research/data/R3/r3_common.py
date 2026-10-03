# Shared helpers for the R3 MLX microbenchmarks.
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import mlx.core as mx

GB = 1e9


def env_header(name):
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    hdr = {
        "bench": name,
        "mlx": mx.__version__,
        "python": sys.version.split()[0],
        "device": info.get("device_name"),
        "arch": info.get("architecture"),
        "macos": platform.mac_ver()[0],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_")},
    }
    print("# " + json.dumps(hdr), flush=True)
    return hdr


def med(xs):
    return statistics.median(xs)


def time_evals(fn, reps, warmup=3):
    """fn() must build + eval and block. Returns list of seconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def qweight_bytes(N, K, bits=4, group_size=64, scale_bytes=2, has_bias=True):
    w = N * K * bits // 8
    s = N * (K // group_size) * scale_bytes
    b = s if has_bias else 0
    return w + s + b


def make_qweight(N, K, bits=4, group_size=64, dtype=mx.float16, chunk_elems=1 << 25, seed=0):
    """Quantize a random (N, K) fp16 matrix with mx.quantize, in row chunks
    to bound peak memory. Returns (w_q, scales, biases)."""
    mx.random.seed(seed)
    rows = max(1, chunk_elems // K)
    wq, sc, bi = [], [], []
    for r0 in range(0, N, rows):
        r = min(rows, N - r0)
        w = (mx.random.normal((r, K)) * 0.02).astype(dtype)
        q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(q, s, b)
        wq.append(q); sc.append(s); bi.append(b)
        del w
    if len(wq) == 1:
        out = (wq[0], sc[0], bi[0])
    else:
        out = (mx.concatenate(wq, 0), mx.concatenate(sc, 0), mx.concatenate(bi, 0))
    mx.eval(*out)
    return out
