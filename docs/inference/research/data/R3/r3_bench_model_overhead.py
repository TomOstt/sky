# F. Per-token overhead of an mlx-lm Qwen3 decode step (random weights).
#   mode "census": build one decode-step graph and count primitives (no timing).
#   mode "tiny":  Qwen3 layer structure with tiny dims -> math is negligible, so
#                 per-token time = MLX framework/dispatch floor for that many layers.
#   mode "real":  Qwen3-8B dims (hidden 4096, 32/8 heads, inter 12288, vocab 151936)
#                 with a reduced number of layers; per-token time vs. weight bytes.
# Usage: python r3_bench_model_overhead.py census|tiny|real [n_layers] [ctx]
import re
import sys
import time
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r3_common import GB, env_header, med

mode = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 36
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 128
env_header(f"F_model_overhead_{mode}_L{n_layers}_ctx{ctx}")

if mode == "real":
    dims = dict(hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, vocab_size=151936)
elif mode == "real32":
    dims = dict(hidden_size=5120, intermediate_size=25600, num_attention_heads=64,
                num_key_value_heads=8, head_dim=128, vocab_size=151936)
else:
    dims = dict(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, vocab_size=1024)

args = qwen3.ModelArgs(model_type="qwen3", num_hidden_layers=n_layers, rms_norm_eps=1e-6,
                       max_position_embeddings=40960, tie_word_embeddings=False,
                       rope_theta=1e6, **dims)
# Initialize directly in fp16 (patch the RNG helpers nn layers call), quantize lazily,
# then materialize one parameter at a time: bounded peak memory even when the
# command-buffer limits are raised (one huge eval OOM'd the GPU, see r3_out_F_1.txt).
_u, _n = mx.random.uniform, mx.random.normal
mx.random.uniform = lambda *a, **k: _u(*a, **{**k, "dtype": mx.float16})
mx.random.normal = lambda *a, **k: _n(*a, **{**k, "dtype": mx.float16})
model = qwen3.Model(args)
mx.random.uniform, mx.random.normal = _u, _n
model.set_dtype(mx.float16)
nn.quantize(model, group_size=64, bits=4)
from mlx.utils import tree_flatten
for _k, _v in tree_flatten(model.parameters()):
    mx.eval(_v)
mx.clear_cache()
mx.reset_peak_memory()
wbytes = sum(v.nbytes for _, v in nn.utils.tree_flatten(model.parameters()))
emb_bytes = sum(v.nbytes for _, v in nn.utils.tree_flatten(model.model.embed_tokens.parameters()))
print(f"# params bytes {wbytes/1e9:.3f} GB (embedding table {emb_bytes/1e9:.3f} GB, read 1 row/token)")

cache = make_prompt_cache(model)
prompt = mx.random.randint(0, dims["vocab_size"], (1, ctx))
logits = model(prompt, cache=cache)
mx.eval(logits, [c.state for c in cache])


def step(y):
    logits = model(y[None], cache=cache)[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    return mx.argmax(logprobs, axis=-1), logprobs


y = mx.array([1])
mx.eval(y)

if mode == "census":
    out, lp = step(y)
    mx.export_to_dot("r3_census.dot", out, lp)
    txt = open("r3_census.dot").read()
    prims = re.findall(r'label ="([^"]+)", shape=rectangle', txt)
    c = Counter(p.split(" ")[0] for p in prims)
    print("total primitive nodes:", len(prims))
    for k, v in c.most_common():
        print(f"  {k:<28s} {v}")
    sys.exit(0)

# warmup
for _ in range(5):
    y, _ = step(y)
    mx.eval(y)

# (a) graph build only (python + C++ graph construction), no eval.
# (cache offset advances; we roll it back by trimming afterwards)
tb = []
for _ in range(10):
    t0 = time.perf_counter()
    o, lp = step(y)
    tb.append(time.perf_counter() - t0)
    for c in cache:
        c.trim(1)
t_build = med(tb)

# (b) synchronous: build + eval each token
ts = []
for _ in range(30):
    t0 = time.perf_counter()
    y, lp = step(y)
    mx.eval(y)
    ts.append(time.perf_counter() - t0)
t_sync = med(ts)

# (c) pipelined like mlx_lm.generate_step: async_eval next before reading current
N = 60
y, lp = step(y)
mx.async_eval(y, lp)
tp = []
t_prev = time.perf_counter()
for i in range(N):
    ny, nlp = step(y)
    mx.async_eval(ny, nlp)
    _ = y.item()
    t_now = time.perf_counter()
    tp.append(t_now - t_prev)
    t_prev = t_now
    y, lp = ny, nlp
mx.eval(y)
t_pipe = med(tp[5:])

line = (f"L={n_layers:3d} ctx={ctx:5d} | build-only {t_build*1e3:7.3f} ms | sync step {t_sync*1e3:7.3f} ms | "
        f"pipelined {t_pipe*1e3:7.3f} ms/token ({1/t_pipe:7.1f} tok/s)")
if mode.startswith("real"):
    streamed = wbytes - emb_bytes
    line += (f" | streamed weights {streamed/1e9:.3f} GB -> {streamed/t_pipe/GB:6.1f} GB/s effective; "
             f"ideal @182GB/s {streamed/182e9*1e3:.3f} ms")
print(line, flush=True)
print("peak_memory_GB", mx.get_peak_memory() / GB)
