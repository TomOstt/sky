# G. EXPERIMENT: a simple multi-vector 4-bit GEMV ("qmv_multi") written as an
# mx.fast.metal_kernel, vs MLX's quantized_matmul at M = 1..8 on Qwen3-32B shapes.
# Same weight layout as MLX affine 4-bit g64 (uint32-packed nibbles, fp16 scale+bias).
# Design: MLX qmv structure (2 simdgroups x 4 rows per threadgroup, 32 lanes split K,
# simd_sum at the end) but each weight chunk is decoded ONCE per lane and applied to
# MV activation rows held in registers. 8 values (one uint32) per lane per row per step.
# Usage: python r3_bench_qmv_multi.py [check]
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, make_qweight, med

env_header("G_qmv_multi")
GS, BITS = 64, 4

SRC = """
    const int ROWS = 4;
    const int VPT = 8;
    const int BLOCK = VPT * 32;
    const int Kw = KD / 8;
    const int Kg = KD / 64;
    uint tg = threadgroup_position_in_grid.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint ln = thread_index_in_simdgroup;
    int row0 = tg * 8 + sg * ROWS;

    float res[ROWS][MV];
    for (int r = 0; r < ROWS; r++) for (int m = 0; m < MV; m++) res[r][m] = 0.0f;

    const device uint32_t* wp = w + row0 * Kw + ln;
    const device half* sp = scales + row0 * Kg + ln / 8;
    const device half* bp = biases + row0 * Kg + ln / 8;
    const device half* xp = x + ln * VPT;

    for (int k = 0; k < KD; k += BLOCK) {
        float xs[MV][VPT];
        float xsum[MV];
        for (int m = 0; m < MV; m++) {
            const device half4* xv = (const device half4*)(xp + m * KD + k);
            half4 a = xv[0];
            half4 b = xv[1];
            float v0 = a.x, v1 = a.y, v2 = a.z, v3 = a.w, v4 = b.x, v5 = b.y, v6 = b.z, v7 = b.w;
            xsum[m] = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7));
            xs[m][0] = v0;
            xs[m][1] = v1 * 0.0625f;
            xs[m][2] = v2 * 0.00390625f;
            xs[m][3] = v3 * 0.000244140625f;
            xs[m][4] = v4 * 1.52587890625e-05f;
            xs[m][5] = v5 * 9.5367431640625e-07f;
            xs[m][6] = v6 * 5.9604644775390625e-08f;
            xs[m][7] = v7 * 3.725290298461914e-09f;
        }
        const int kq = k / 8;
        const int kg = k / 64;
        for (int r = 0; r < ROWS; r++) {
            uint q = wp[r * Kw + kq];
            float sc = float(sp[r * Kg + kg]);
            float bi = float(bp[r * Kg + kg]);
            float wf[VPT];
            wf[0] = float(q & 0x0000000fu);
            wf[1] = float(q & 0x000000f0u);
            wf[2] = float(q & 0x00000f00u);
            wf[3] = float(q & 0x0000f000u);
            wf[4] = float(q & 0x000f0000u);
            wf[5] = float(q & 0x00f00000u);
            wf[6] = float(q & 0x0f000000u);
            wf[7] = float(q & 0xf0000000u);
            for (int m = 0; m < MV; m++) {
                float acc = xs[m][0] * wf[0];
                for (int i = 1; i < VPT; i++) acc = fma(xs[m][i], wf[i], acc);
                res[r][m] = fma(sc, acc, fma(bi, xsum[m], res[r][m]));
            }
        }
    }
    for (int r = 0; r < ROWS; r++) {
        for (int m = 0; m < MV; m++) {
            float v = simd_sum(res[r][m]);
            if (ln == 0) y[m * ND + row0 + r] = half(v);
        }
    }
"""

kern = mx.fast.metal_kernel(
    name="qmv_multi_g64_b4",
    input_names=["w", "scales", "biases", "x"],
    output_names=["y"],
    source=SRC,
)


def qmv_multi(x, wq, sc, bi):
    M, K = x.shape
    N = wq.shape[0]
    assert N % 8 == 0 and K % 256 == 0
    return kern(
        inputs=[wq, sc, bi, x],
        template=[("MV", M), ("KD", K), ("ND", N)],
        grid=(64 * (N // 8), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
    )[0]


def mlx_qmm(x, wq, sc, bi):
    return mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=GS, bits=BITS)


# --- correctness ---
wq, sc, bi = make_qweight(1024, 5120, seed=3)
for M in [1, 2, 3, 4, 8]:
    x = (mx.random.normal((M, 5120)) * 0.5).astype(mx.float16)
    a = qmv_multi(x, wq, sc, bi).astype(mx.float32)
    b = mlx_qmm(x, wq, sc, bi).astype(mx.float32)
    ref = x.astype(mx.float32) @ mx.dequantize(wq, sc, bi, group_size=GS, bits=BITS).astype(mx.float32).T
    print(f"# check M={M}: max|ours-ref| {mx.max(mx.abs(a-ref)).item():.3e}  "
          f"max|mlx-ref| {mx.max(mx.abs(b-ref)).item():.3e}  max|ref| {mx.max(mx.abs(ref)).item():.3f}", flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "check":
    sys.exit(0)

BATCH, REPS, WS = 16, 10, 320e6
SHAPES = [("32B q_proj", 5120, 8192), ("32B gate/up", 5120, 25600), ("32B down", 25600, 5120)]
for name, K, N in SHAPES:
    nb = sum(a.nbytes for a in make_qweight(8, K)) * N // 8
    ncopies = min(400, max(2, int(-(-WS // nb))))
    ws = [make_qweight(N, K, seed=i) for i in range(ncopies)]
    base = {}
    for M in [1, 2, 3, 4, 5, 6, 8]:
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.float16)
        mx.eval(x)
        row = []
        for lab, fn in [("mlx", mlx_qmm), ("multi", qmv_multi)]:
            for i in range(4):
                mx.eval(fn(x, *ws[i % ncopies]))
            tb, j = [], 0
            for it in range(REPS + 2):
                sel = [ws[(j + b) % ncopies] for b in range(BATCH)]
                j += BATCH
                t0 = time.perf_counter()
                mx.eval([fn(x, *w) for w in sel])
                dt = time.perf_counter() - t0
                if it >= 2:
                    tb.append(dt / BATCH)
            t = med(tb)
            if M == 1:
                base[lab] = t
            row.append(f"{lab:5s} {t*1e6:8.1f} us (x{t/base[lab]:4.2f}, {nb/t/GB:6.1f} GB/s-w)")
        print(f"{name:<12s} M={M:2d} | " + " | ".join(row), flush=True)
    del ws
    mx.clear_cache()
print("peak_memory_GB", mx.get_peak_memory() / GB)
