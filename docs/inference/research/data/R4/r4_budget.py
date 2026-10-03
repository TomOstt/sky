# R4 measurement E: per-token time budget of mlx-lm decode, by ablation.
# A forward function that replicates mlx_lm/models/qwen3.py op-for-op (validated bit-exact
# against model(...)), with switches that remove classes of non-matmul ops while keeping
# every quantized matmul (same weights, same shapes, same dependency structure).
# Timings:
#   pipelined  = steady-state inter-token latency with mlx-lm's own async_eval pipelining
#                (build t+1 -> async_eval(t+1) -> wait t), i.e. what generate_step does
#   sync       = one step at a time: build | encode(async_eval call) | wait
# Usage: python r4_budget.py <model-name> [ctx]
import sys, os, time, json, ctypes, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.activations import swiglu
import r4_common as C

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "libr4count.dylib")); lib.r4_install()
_cnt = (ctypes.c_long * 5)()
def ndispatch():
    lib.r4_counts(_cnt); return _cnt[0] + _cnt[1], _cnt[2]

name = sys.argv[1]
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NPIPE = int(os.environ.get("NPIPE", "64"))
NSYNC = int(os.environ.get("NSYNC", "20"))
model, tok = load(C.model_path(name))
tied = model.args.tie_word_embeddings
ids = mx.array(C.prompt_ids(tok, ctx))

def fresh_cache():
    c = make_prompt_cache(model)
    model(ids[None, :-1], cache=c)
    mx.eval([x.state for x in c])
    return c

def fwd(y, cache, norms=True, rope=True, attn=True, adds=True, act=True, sample=True, extras=None):
    m = model.model
    h = m.embed_tokens(y[None])
    for layer, c in zip(m.layers, cache):
        a = layer.self_attn
        x = layer.input_layernorm(h) if norms else h
        B, L, D = x.shape
        q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q = q.reshape(B, L, a.n_heads, -1)
        k = k.reshape(B, L, a.n_kv_heads, -1)
        if norms:
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if attn:
            if rope:
                q = a.rope(q, offset=c.offset); k = a.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            o = scaled_dot_product_attention(q, k, v, cache=c, scale=a.scale, mask=None)
        else:
            o = q
            extras.extend([k, v])  # keep k/v projections alive and evaluated
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        r = a.o_proj(o)
        h = (h + r) if adds else r
        x = layer.post_attention_layernorm(h) if norms else h
        g, u = layer.mlp.gate_proj(x), layer.mlp.up_proj(x)
        if act:
            g = swiglu(g, u)
        else:
            extras.append(u)
        r = layer.mlp.down_proj(g)
        h = (h + r) if adds else r
    h = m.norm(h) if norms else h
    logits = (m.embed_tokens.as_linear(h) if tied else model.lm_head(h))[:, -1, :]
    if sample:
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return mx.argmax(logprobs, axis=-1), logprobs
    return mx.argmax(logits, axis=-1), logits

VARIANTS = {
    "full (= qwen3.py)": dict(),
    "no logsumexp (argmax on logits)": dict(sample=False),
    "no q/k/input/post RMSNorm": dict(norms=False),
    "no RoPE": dict(rope=False),
    "no attention (rope+cache+sdpa)": dict(attn=False),
    "no residual adds": dict(adds=False),
    "no SwiGLU": dict(act=False),
    "qmv only (all non-matmul removed)": dict(norms=False, attn=False, adds=False, act=False, sample=False),
}

# ---- validation: bit-exact vs mlx-lm model() ----
cA, cB = fresh_cache(), fresh_cache()
ref = model(ids[-1:][None], cache=cA)[:, -1, :]
mine, mine_lp = fwd(ids[-1:], cB, sample=False)
mx.eval(ref, mine_lp)
maxdiff = mx.max(mx.abs(ref.astype(mx.float32) - mine_lp.astype(mx.float32))).item()
print(f"[validate] max|model() - fwd()| = {maxdiff}")
del cA, cB

def run_pipelined(opts, n=NPIPE, warm=8):
    cache = fresh_cache()
    extras = []
    y = ids[-1:]
    cur, _ = fwd(y, cache, extras=extras, **opts)
    mx.async_eval(cur, extras); extras = []
    stamps = []
    for i in range(n + warm):
        nxt, lp = fwd(cur, cache, extras=extras, **opts)
        mx.async_eval(nxt, lp, extras); extras = []
        cur.item()
        stamps.append(time.perf_counter())
        cur = nxt
    cur.item()
    d = [b - a for a, b in zip(stamps[warm:], stamps[warm + 1:])]
    return statistics.median(d), statistics.mean(d)

def run_sync(opts, n=NSYNC, warm=4):
    cache = fresh_cache()
    y = ids[-1:]
    tb, te, tw, tt, nd = [], [], [], [], []
    for i in range(n + warm):
        extras = []
        lib.r4_reset()
        t0 = time.perf_counter()
        nxt, lp = fwd(y, cache, extras=extras, **opts)
        t1 = time.perf_counter()
        mx.async_eval(nxt, lp, extras)
        t2 = time.perf_counter()
        mx.eval(nxt, lp, extras)
        t3 = time.perf_counter()
        if i >= warm:
            tb.append(t1 - t0); te.append(t2 - t1); tw.append(t3 - t2); tt.append(t3 - t0)
            nd.append(ndispatch())
        if opts.get("attn", True):
            for c in cache:
                c.trim(1)
    med = statistics.median
    return dict(build_ms=1e3 * med(tb), encode_ms=1e3 * med(te), wait_ms=1e3 * med(tw),
                total_ms=1e3 * med(tt), gpu_upper_ms=1e3 * med([e + w for e, w in zip(te, tw)]),
                dispatches=nd[-1][0], barriers=nd[-1][1])

# ---- reference: mlx-lm's own generate_step, steady-state ITL ----
def run_generate_step(n=NPIPE, warm=8):
    stamps = []
    for i, (t, _) in enumerate(generate_step(ids, model, max_tokens=n + warm + 1)):
        stamps.append(time.perf_counter())
    d = [b - a for a, b in zip(stamps[warm:], stamps[warm + 1:])]
    return statistics.median(d), statistics.mean(d)

res = dict(model=name, ctx=ctx, validate_maxdiff=maxdiff, env=C.env_info(), variants={})
g_med, g_mean = run_generate_step()
res["generate_step_itl_ms"] = (1e3 * g_med, 1e3 * g_mean)
print(f"[{name} ctx={ctx}] mlx-lm generate_step ITL: median {1e3*g_med:.3f} ms  mean {1e3*g_mean:.3f} ms "
      f"({1/g_med:.1f} tok/s)")
for label, opts in VARIANTS.items():
    p_med, p_mean = run_pipelined(opts)
    s = run_sync(opts)
    res["variants"][label] = dict(pipelined_med_ms=1e3 * p_med, pipelined_mean_ms=1e3 * p_mean, **s)
    print(f"  {label:36s} pipelined ITL med {1e3*p_med:8.3f} ms | sync: build {s['build_ms']:6.3f} "
          f"encode {s['encode_ms']:6.3f} wait {s['wait_ms']:7.3f} total {s['total_ms']:7.3f} ms | "
          f"dispatches {s['dispatches']:4d} barriers {s['barriers']:4d}", flush=True)
fn = os.path.join(here, f"raw_E_{name}_ctx{ctx}_{time.strftime('%H%M%S')}.json")
json.dump(res, open(fn, "w"), indent=1, default=str)
print("raw ->", fn)
