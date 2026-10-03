import json, sys, datetime as dt, statistics as st
W = sys.argv[1]
samples = [(dt.datetime.fromisoformat(json.loads(l)["timestamp"]).timestamp(), json.loads(l)) for l in open(f"{W}/demo_macmon.jsonl")]
marks = [json.loads(l[6:]) for l in open(f"{W}/demo_out.txt") if l.startswith("MARKS ")][0]
idle_a = float(open(f"{W}/demo_idle_start.txt").read()); idle_b = float(open(f"{W}/demo_idle_end.txt").read())
CH = {"cpu": "cpu_power", "gpu": "gpu_power", "ane": "ane_power", "dram": "ram_power", "pstr": "sys_power"}
def energy(a, b):
    E = {k: 0.0 for k in CH}
    for (t0, _), (t1, d) in zip(samples, samples[1:]):
        lo, hi = max(t0, a), min(t1, b)
        if hi > lo:
            for k, f in CH.items(): E[k] += d[f] * (hi - lo)
    return E
def agg(E): return {"soc": E["cpu"] + E["gpu"] + E["ane"], "soc_dram": E["cpu"] + E["gpu"] + E["ane"] + E["dram"], "pstr": E["pstr"]}
Ti = idle_b - idle_a; Pidle = {k: v / Ti for k, v in agg(energy(idle_a, idle_b)).items()}
print("idle W:", {k: round(v, 2) for k, v in Pidle.items()}, f"T={Ti:.1f}s")
def phase(name, a, b, ntok):
    T = b - a; E = agg(energy(a, b))
    print(f"{name:12s} T={T:6.2f}s tokens={ntok:6d}  avg W: " + " ".join(f"{k}={v/T:6.2f}" for k, v in E.items()))
    print(f"{'':12s} J/token gross: " + " ".join(f"{k}={v/ntok*1000:7.2f}mJ" for k, v in E.items()) +
          " | net: " + " ".join(f"{k}={(v - Pidle[k]*T)/ntok*1000:7.2f}mJ" for k, v in E.items()))
phase("prefill-loop", marks["prefill_loop_start"], marks["prefill_loop_end"], marks["prefill_tokens_total"])
phase("ttft(1 run)", marks["t_start"], marks["t_first_token"], marks["prompt_tokens"])
phase("decode", marks["t_first_token"], marks["t_end"], marks["generation_tokens"] - 1)
dec = [d for t, d in samples if marks["t_first_token"] + 0.5 < t <= marks["t_end"]]
g = [d["gpu_power"] for d in dec]; c = [d["cpu_power"] for d in dec]; p = [d["sys_power"] for d in dec]
print(f"decode 200ms-sample stats: n={len(dec)} gpu W median={st.median(g):.2f} p5={sorted(g)[len(g)//20]:.2f} p95={sorted(g)[len(g)*19//20]:.2f}; cpu W median={st.median(c):.2f}; PSTR distinct={len(set(round(x,3) for x in p))}")
print("gpu_freq_mhz set:", sorted(set(d['gpu_freq_mhz'] for d in dec)), "gpu_active median:", st.median(d['gpu_active_ratio'] for d in dec))
print("marks:", json.dumps({k: marks[k] for k in ("prefill_reps","prefill_tokens_total","prompt_tokens","prompt_tps","generation_tokens","generation_tps","peak_memory_gb","finish_reason")}))
