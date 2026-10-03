"""Integrate macmon (IOReport + SMC PSTR) power samples over phase windows.
Each macmon sample i is an average over (t_{i-1}, t_i] (timestamp = end of window, src_app/main.rs:205-211),
so energy over [a,b] = sum_i P_i * |(t_{i-1},t_i] ∩ [a,b]|  (exact for counter deltas; no trapezoid needed).
PSTR is a held ~1 s value; treated the same way (approximation)."""
import json, sys, datetime as dt
W = sys.argv[1]
samples = []
for line in open(f"{W}/demo_macmon.jsonl"):
    d = json.loads(line)
    t = dt.datetime.fromisoformat(d["timestamp"]).timestamp()
    samples.append((t, d))
marks = None
for line in open(f"{W}/demo_out.txt"):
    if line.startswith("MARKS "):
        marks = json.loads(line[6:])
idle_a = float(open(f"{W}/demo_idle_start.txt").read()); idle_b = float(open(f"{W}/demo_idle_end.txt").read())
CH = {"cpu": "cpu_power", "gpu": "gpu_power", "ane": "ane_power", "dram": "ram_power", "gpu_sram": "gpu_ram_power", "pstr": "sys_power"}
def energy(a, b):
    E = {k: 0.0 for k in CH}
    for (t0, _), (t1, d) in zip(samples, samples[1:]):
        lo, hi = max(t0, a), min(t1, b)
        if hi > lo:
            for k, f in CH.items():
                E[k] += d[f] * (hi - lo)
    return E
def show(name, a, b, ntok=None, base=None):
    E = energy(a, b); T = b - a
    soc = E["cpu"] + E["gpu"] + E["ane"]; soc_dram = soc + E["dram"]
    line = f"{name:8s} T={T:6.2f}s  P_avg: cpu={E['cpu']/T:5.2f} gpu={E['gpu']/T:5.2f} dram={E['dram']/T:5.2f} gpu_sram={E['gpu_sram']/T:5.2f} | SoC(cpu+gpu+ane)={soc/T:5.2f}W SoC+DRAM={soc_dram/T:5.2f}W PSTR(system)={E['pstr']/T:5.2f}W"
    print(line)
    if ntok:
        r = {"soc": soc, "soc_dram": soc_dram, "pstr": E["pstr"]}
        out = "         J/token gross: " + "  ".join(f"{k}={v/ntok:.4f}" for k, v in r.items())
        if base:
            out += " | net of idle: " + "  ".join(f"{k}={(v - base[k]*T)/ntok:.4f}" for k, v in r.items())
        print(out)
    return E
Ei = show("idle", idle_a, idle_b)
Ti = idle_b - idle_a
base = {"soc": (Ei["cpu"]+Ei["gpu"]+Ei["ane"])/Ti, "soc_dram": (Ei["cpu"]+Ei["gpu"]+Ei["ane"]+Ei["dram"])/Ti, "pstr": Ei["pstr"]/Ti}
print("marks:", json.dumps(marks))
show("prefill", marks["t_start"], marks["t_first_token"], marks["prompt_tokens"], base)
show("decode", marks["t_first_token"], marks["t_end"], marks["generation_tokens"] - 1, base)
show("total", marks["t_start"], marks["t_end"], None, base)
# count of samples in decode window, and cadence check (zero-valued gpu samples while busy => batching)
dec = [d for t, d in samples if marks["t_first_token"] < t <= marks["t_end"]]
print(f"decode samples={len(dec)}  zero-GPU samples in decode={sum(1 for d in dec if d['gpu_power']==0.0)}  distinct PSTR values={len(set(round(d['sys_power'],3) for d in dec))}")
print("gpu_freq_mhz in decode:", sorted(set(d['gpu_freq_mhz'] for d in dec))[:10], " gpu_active_ratio median:", sorted(d['gpu_active_ratio'] for d in dec)[len(dec)//2] if dec else None)
