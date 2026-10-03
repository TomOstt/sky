# R4 measurement D: mlx-lm speculative decoding, Qwen3-8B-4bit target + Qwen3-0.6B-4bit draft.
# Usage: python r4_spec.py chat|code|summary      (one prompt type per lock acquisition)
#        python r4_spec.py verify                  (cost of an M-token target forward, M=1..8)
# Greedy (temp 0), EOS enabled, max_tokens=GEN. Records wall-clock per-token stamps, mlx-lm's
# own generation_tps, acceptance (from_draft fraction), and whether the token sequence equals
# the non-speculative baseline.
import sys, os, time, json, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import r4_common as C

here = os.path.dirname(os.path.abspath(__file__))
kind = sys.argv[1]
TRIALS = int(os.environ.get("TRIALS", "5"))
GEN = int(os.environ.get("GEN", "192"))

model, tok = load(C.model_path("Qwen3-8B-4bit"))

if kind == "verify":
    ids = mx.array(C.prompt_ids(tok, 256))
    cache = make_prompt_cache(model)
    model(ids[None, :-8], cache=cache); mx.eval([c.state for c in cache])
    res = {}
    for M in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]:
        ts = []
        for i in range(4 + 15):
            x = ids[-M:][None]
            t0 = time.perf_counter()
            out = model(x, cache=cache)
            mx.eval(out)
            t1 = time.perf_counter()
            for c in cache:
                c.trim(M)
            if i >= 4:
                ts.append(t1 - t0)
        res[M] = 1e3 * statistics.median(ts)
        print(f"  target forward M={M:2d} tokens: median {res[M]:7.3f} ms  ({res[M]/res[1]:.2f}x of M=1)",
              flush=True)
    json.dump(res, open(os.path.join(here, f"raw_D_verify_{time.strftime('%H%M%S')}.json"), "w"))
    sys.exit(0)

draft, dtok = load(C.model_path("Qwen3-0.6B-4bit"))

passage = open(os.path.join(C.SRC, "mlx/README.md")).read()
passage = passage[passage.find("MLX is"):][:2200]
PROMPTS = {
    "chat": "What are the main differences between a cold and the flu? Answer in a few short paragraphs.",
    "code": "Write a Python function that reads a CSV file and returns the mean of every numeric column "
            "as a dict. Include type hints, a docstring and error handling.",
    "summary": "Summarize the following passage in five bullet points.\n\n" + passage,
}
msgs = [{"role": "user", "content": PROMPTS[kind]}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok.encode(prompt, add_special_tokens=False)

def run(k):
    stamps, toks, fd = [], [], []
    t_call = time.perf_counter()
    kw = dict(draft_model=draft, num_draft_tokens=k) if k else {}
    for r in stream_generate(model, tok, ids, max_tokens=GEN, **kw):
        stamps.append(time.perf_counter()); toks.append(r.token); fd.append(r.from_draft); last = r
    n = len(stamps)
    return dict(n=n, ttft=stamps[0] - t_call,
                wall_tps=(n - 1) / (stamps[-1] - stamps[0]),
                e2e_tps=n / (stamps[-1] - t_call),
                mlxlm_tps=last.generation_tps, accept=sum(fd) / n, tokens=toks,
                peak_gb=last.peak_memory)

out = dict(kind=kind, prompt_tokens=len(ids), gen=GEN, env=C.env_info(), configs={})
run(0); run(3)  # warm-up both paths
base_tokens = None
for k in [0, 2, 4, 6]:
    mx.reset_peak_memory()
    rs = [run(k) for _ in range(TRIALS)]
    if k == 0:
        base_tokens = rs[0]["tokens"]
    same = all(r["tokens"] == base_tokens for r in rs)
    first_div = next((i for i, (a, b) in enumerate(zip(rs[0]["tokens"], base_tokens)) if a != b), None)
    med = lambda key: statistics.median(r[key] for r in rs)
    out["configs"][k] = dict(trials=[{kk: vv for kk, vv in r.items() if kk != "tokens"} for r in rs],
                             same_as_baseline=same, first_divergence=first_div,
                             median_wall_tps=med("wall_tps"), median_e2e_tps=med("e2e_tps"),
                             median_mlxlm_tps=med("mlxlm_tps"), median_accept=med("accept"),
                             n=rs[0]["n"])
    print(f"[{kind}] prompt={len(ids)} tok  k={k}: n={rs[0]['n']}  wall decode tok/s med="
          f"{med('wall_tps'):.2f} (all {[round(r['wall_tps'], 2) for r in rs]})  mlx-lm gen tps med="
          f"{med('mlxlm_tps'):.2f}  e2e tok/s med={med('e2e_tps'):.2f}  accept={med('accept'):.3f}  "
          f"identical-to-baseline={same} first_div={first_div}  peak={rs[-1]['peak_gb']:.2f}GB", flush=True)
print("baseline text:", repr(tok.decode(base_tokens)[:300]))
json.dump(out, open(os.path.join(here, f"raw_D_{kind}_{time.strftime('%H%M%S')}.json"), "w"),
          indent=1, default=str)
