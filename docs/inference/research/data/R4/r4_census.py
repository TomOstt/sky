# R4 measurement C: what one mlx-lm decode step dispatches on the GPU.
#  (1) graph census: mx.export_to_dot on the outputs of one generate_step-equivalent
#      decode step (logits -> logsumexp -> argmax) with a warm KV cache;
#  (2) exact Metal counts: dispatchThreads/dispatchThreadgroups/memoryBarrier/endEncoding
#      calls intercepted by libr4count.dylib (ObjC swizzle of the AGX compute encoder).
# Usage: python r4_census.py <model-name> [ctx]
import sys, os, io, re, ctypes, collections, json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import r4_common as C

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "libr4count.dylib"))
lib.r4_install()
cnt = (ctypes.c_long * 5)()

def counts():
    lib.r4_counts(cnt)
    return dict(dispatchThreadgroups=cnt[0], dispatchThreads=cnt[1], barriers=cnt[2],
                barriers_res=cnt[3], encoders=cnt[4])

name = sys.argv[1]
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 128
model, tok = load(C.model_path(name))
ids = mx.array(C.prompt_ids(tok, ctx))
cache = make_prompt_cache(model)
model(ids[None, :-1], cache=cache)
mx.eval([c.state for c in cache])
y = ids[-1:]

def step(y):
    # identical to mlx_lm.generate.generate_step._step (no logits processors, greedy)
    logits = model(y[None], cache=cache)[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    return mx.argmax(logprobs, axis=-1), logprobs.squeeze(0)

# warm-up steps (compile kernels, settle cache growth)
for _ in range(3):
    y2, lp = step(y); mx.eval(y2, lp); y = y2
for c in cache:
    c.trim(3)

# (1) graph census
y2, lp = step(y)
f = io.StringIO()
mx.export_to_dot(f, y2, lp)
labels = re.findall(r'\[label ="([^"]+)"', f.getvalue())
census = collections.Counter(labels)
open(os.path.join(here, f"census_{name}_ctx{ctx}.dot"), "w").write(f.getvalue())
# (2) exact dispatch counts for evaluating exactly this graph
lib.r4_reset()
mx.eval(y2, lp)
exact = counts()
for c in cache:
    c.trim(1)
# (3) same, but evaluate model logits only (no sampling) to separate sampling cost
lib.r4_reset()
logits = model(y[None], cache=cache)
mx.eval(logits)
exact_fwd = counts()
for c in cache:
    c.trim(1)

L = len(model.layers)
print(f"== {name}  layers={L}  ctx={ctx}  offset_after={cache[0].offset}")
print(f"graph primitive nodes: {sum(census.values())}")
for k, v in census.most_common():
    print(f"  {k:60s} {v:5d}  ({v/L:.2f}/layer)")
print("exact Metal calls for one full decode step (forward+logsumexp+argmax):", exact)
print("exact Metal calls for forward only (logits):                       ", exact_fwd)
tot = exact["dispatchThreadgroups"] + exact["dispatchThreads"]
print(f"total dispatches/token = {tot}  ({tot/L:.2f} per layer); barriers/token = {exact['barriers']}; "
      f"compute encoders (~command buffers) = {exact['encoders']}")
json.dump(dict(model=name, ctx=ctx, census=census, exact=exact, exact_fwd=exact_fwd),
          open(os.path.join(here, f"census_{name}_ctx{ctx}.json"), "w"), indent=1)
