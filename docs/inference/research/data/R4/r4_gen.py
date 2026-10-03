# R4 measurement A: mlx-lm end-to-end decode / prefill / TTFT / peak memory.
# Usage: python r4_gen.py <model-name> decode|prefill [lengths...]
# Uses mlx-lm's own stream_generate (the public generation path) and records BOTH
# mlx-lm's self-reported numbers and our own wall-clock timestamps per token.
import sys, time, json, os
import mlx.core as mx
from mlx_lm import load, stream_generate
import r4_common as C

name = sys.argv[1]
mode = sys.argv[2]
TRIALS = int(os.environ.get("TRIALS", "5"))
GEN = int(os.environ.get("GEN", "256"))

t0 = time.perf_counter()
model, tok = load(C.model_path(name))
load_s = time.perf_counter() - t0
tok.eos_token_ids = []  # never stop early: always generate exactly max_tokens

def run(ids, max_tokens):
    mx.reset_peak_memory()
    stamps = []
    t_call = time.perf_counter()
    last = None
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens):  # greedy (temp 0) default
        stamps.append(time.perf_counter())
        last = r
    return dict(
        ttft_wall=stamps[0] - t_call,
        decode_wall_tps=(len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
        itl_median_ms=(1e3 * sorted(b - a for a, b in zip(stamps, stamps[1:]))[(len(stamps) - 1) // 2]
                       if len(stamps) > 2 else None),
        n=len(stamps),
        mlxlm_prompt_tps=last.prompt_tps,
        mlxlm_generation_tps=last.generation_tps,
        mlxlm_peak_gb=last.peak_memory,
        mlx_peak_gb_after=mx.get_peak_memory() / 1e9,
        active_gb_after=mx.get_active_memory() / 1e9,
        cache_gb_after=mx.get_cache_memory() / 1e9,
        prompt_tokens=last.prompt_tokens,
    )

out = dict(env=C.env_info(), model=name, mode=mode, load_s=load_s, trials=TRIALS, results={})
if mode == "decode":
    ids = C.prompt_ids(tok, 128)
    run(ids, 32)  # warm-up
    rs = [run(ids, GEN) for _ in range(TRIALS)]
    out["results"]["decode_p128_g%d" % GEN] = rs
    print(f"[{name}] decode prompt=128 gen={GEN}")
    for k in ("decode_wall_tps", "itl_median_ms", "mlxlm_generation_tps", "ttft_wall",
              "mlxlm_prompt_tps", "mlxlm_peak_gb"):
        vals = [r[k] for r in rs]
        print(f"  {k:22s} median={C.med(vals):.4f}  all={[round(v, 4) for v in vals]}")
elif mode == "prefill":
    lengths = [int(x) for x in sys.argv[3:]] or [128, 512, 2048]
    ids_all = C.prompt_ids(tok, max(lengths))
    run(ids_all[:128], 4)  # warm-up (kernels for both prefill and decode shapes)
    run(ids_all[:max(lengths)], 1)
    for L in lengths:
        rs = [run(ids_all[:L], 1) for _ in range(TRIALS)]
        out["results"][f"prefill_{L}"] = rs
        print(f"[{name}] prefill L={L}")
        for k in ("ttft_wall", "mlxlm_prompt_tps", "mlxlm_peak_gb"):
            vals = [r[k] for r in rs]
            print(f"  {k:18s} median={C.med(vals):.4f}  all={[round(v, 4) for v in vals]}")
        print(f"  wall prefill tok/s (L/ttft_wall) median={L / C.med([r['ttft_wall'] for r in rs]):.1f}")
out["rusage_end"] = C.rusage()
print("  os lifetime_max_phys_footprint GB:", out["rusage_end"].get("lifetime_max_phys_footprint", 0) / 1e9)
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  f"raw_A_{name}_{mode}_{time.strftime('%H%M%S')}.json")
json.dump(out, open(fn, "w"), indent=1, default=str)
print("raw ->", fn)
