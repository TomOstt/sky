# Energy-per-token demo: phase timestamps for alignment with an external power sampler (macmon).
import time, json, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
model, tok = load(MODEL)
filler = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
          "energy report and the benchmark harness records timestamps for every phase. ") * 40
msgs = [{"role": "user", "content": filler + "\nNow write the integers from 1 to 5000 separated by single spaces, nothing else."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
sampler = make_sampler(temp=0.0)
for _ in stream_generate(model, tok, prompt, max_tokens=4, sampler=sampler):  # warmup (compiles kernels)
    pass
mx.clear_cache()
marks = {"t_start": time.time()}
n = 0
for r in stream_generate(model, tok, prompt, max_tokens=NTOK, sampler=sampler):
    n += 1
    if n == 1:
        marks["t_first_token"] = time.time()
marks["t_end"] = time.time()
marks.update(prompt_tokens=r.prompt_tokens, prompt_tps=r.prompt_tps, generation_tokens=r.generation_tokens,
             generation_tps=r.generation_tps, peak_memory_gb=r.peak_memory, finish_reason=r.finish_reason, model=MODEL)
print("MARKS " + json.dumps(marks), flush=True)
