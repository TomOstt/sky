# R4: decode ITL vs context length (mlx-lm generate_step, greedy), plus TTFT at that length.
# Usage: python r4_ctx.py <model-name> ctx1 ctx2 ...
import sys, os, time, statistics, json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
import r4_common as C

name = sys.argv[1]
ctxs = [int(x) for x in sys.argv[2:]]
N, WARM = 48, 8
model, tok = load(C.model_path(name))
all_ids = C.prompt_ids(tok, max(ctxs))
generate_step(mx.array(all_ids[:64]), model, max_tokens=4)  # generator object (lazy)
for _ in generate_step(mx.array(all_ids[:64]), model, max_tokens=8):
    pass  # warm-up
res = {}
for L in ctxs:
    mx.reset_peak_memory()
    s = []
    t0 = time.perf_counter()
    for _ in generate_step(mx.array(all_ids[:L]), model, max_tokens=N + WARM):
        s.append(time.perf_counter())
    d = [b - a for a, b in zip(s[WARM:], s[WARM + 1:])]
    res[L] = dict(ttft_s=s[0] - t0, itl_ms=1e3 * statistics.median(d), peak_gb=mx.get_peak_memory() / 1e9)
    print(f"[{name}] ctx={L:6d}  TTFT {res[L]['ttft_s']:7.3f} s  decode ITL median {res[L]['itl_ms']:7.3f} ms "
          f"({1e3/res[L]['itl_ms']:6.2f} tok/s)  peak {res[L]['peak_gb']:.2f} GB", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"raw_ctx_{name}_{time.strftime('%H%M%S')}.json"), "w"))
