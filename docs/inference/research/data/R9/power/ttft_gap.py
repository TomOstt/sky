# TTFT vs idle gap before the request (GPU DVFS / power-gating ramp). Interleaved randomized order, warm process.
import time, json, random, statistics as st
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load("mlx-community/Qwen3-0.6B-4bit")
filler = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
          "energy report and the benchmark harness records timestamps for every phase. ") * 40
prompt = tok.apply_chat_template([{"role": "user", "content": filler + "\nSummarize."}], add_generation_prompt=True, enable_thinking=False)
sampler = make_sampler(temp=0.0)
for _ in range(3):
    for _ in stream_generate(model, tok, prompt, max_tokens=1, sampler=sampler): pass
gaps = [0.0, 0.25, 1.0, 3.0, 8.0]
order = [g for g in gaps for _ in range(6)]
random.seed(1); random.shuffle(order)
res = {g: [] for g in gaps}
for g in order:
    time.sleep(g)
    t0 = time.perf_counter()
    for r in stream_generate(model, tok, prompt, max_tokens=1, sampler=sampler):
        t1 = time.perf_counter()
        break
    res[g].append(t1 - t0)
print("prompt_tokens", r.prompt_tokens)
for g in gaps:
    v = res[g]
    print(f"gap={g:5.2f}s  TTFT median={st.median(v)*1000:7.1f} ms  min={min(v)*1000:7.1f}  max={max(v)*1000:7.1f}  n={len(v)}")
print("RAW " + json.dumps({str(k): v for k, v in res.items()}))
