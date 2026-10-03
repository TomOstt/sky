# R4: does the mlx_lm.server per-token pattern cost decode throughput?
# server.py:867 (batched path) does r.logprobs[r.token].item() for every generated token,
# after BatchGenerator has already async_eval'd the NEXT step. Compare steady-state ITL of:
#  (a) generate_step (stream_generate's engine)
#  (b) BatchGenerator, 1 sequence, plain loop over next_generated()
#  (c) (b) + r.logprobs[r.token].item() per token (exactly what server.py:867 does)
# Usage: python r4_server_pattern.py <model-name>
import sys, os, time, statistics, json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, BatchGenerator
import r4_common as C

name = sys.argv[1]
N, WARM = int(os.environ.get("N", "96")), 8
model, tok = load(C.model_path(name))
ids = C.prompt_ids(tok, 128)

def itl(stamps):
    d = [b - a for a, b in zip(stamps[WARM:], stamps[WARM + 1:])]
    return 1e3 * statistics.median(d)

def gen_step():
    s = []
    for _ in generate_step(mx.array(ids), model, max_tokens=N + WARM):
        s.append(time.perf_counter())
    return itl(s)

def batch(item):
    g = BatchGenerator(model, max_tokens=N + WARM, stop_tokens=[])
    g.insert([ids])
    s = []
    while rs := g.next_generated():
        for r in rs:
            if item:
                r.logprobs[r.token].item()
            s.append(time.perf_counter())
    g.close()
    return itl(s)

res = {}
for rep in range(3):
    for k, f in (("generate_step", gen_step), ("BatchGenerator", lambda: batch(False)),
                 ("BatchGenerator + logprobs[token].item() (server.py:867)", lambda: batch(True))):
        res.setdefault(k, []).append(f())
for k, v in res.items():
    print(f"[{name}] {k:58s} ITL median-of-3 {statistics.median(v):7.3f} ms  runs {[round(x, 3) for x in v]}")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"raw_server_pattern_{name}_{time.strftime('%H%M%S')}.json"), "w"))
