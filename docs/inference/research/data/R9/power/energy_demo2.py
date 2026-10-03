# Energy demo v2: (B) looped prefill-only phase >=10 s, then (C) long decode with EOS masked. Prints phase timestamps.
import time, json, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
MODEL = sys.argv[1]; NTOK = int(sys.argv[2]); PREFILL_SECONDS = float(sys.argv[3])
model, tok = load(MODEL)
filler = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly "
          "energy report and the benchmark harness records timestamps for every phase. ") * 40
msgs = [{"role": "user", "content": filler + "\nNow write the integers from 1 to 5000 separated by single spaces, nothing else."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
sampler = make_sampler(temp=0.0)
eos = list(tok.eos_token_ids)
def no_eos(tokens, logits):
    logits[:, eos] = -float("inf")
    return logits
for _ in stream_generate(model, tok, prompt, max_tokens=4, sampler=sampler):
    pass
marks = {}
# (B) prefill loop
marks["prefill_loop_start"] = time.time(); reps = 0; ptoks = 0
while time.time() - marks["prefill_loop_start"] < PREFILL_SECONDS:
    for r in stream_generate(model, tok, prompt, max_tokens=1, sampler=sampler):
        pass
    reps += 1; ptoks += r.prompt_tokens
marks["prefill_loop_end"] = time.time(); marks["prefill_reps"] = reps; marks["prefill_tokens_total"] = ptoks
time.sleep(3.0)
# (C) long decode
marks["t_start"] = time.time(); n = 0
for r in stream_generate(model, tok, prompt, max_tokens=NTOK, sampler=sampler, logits_processors=[no_eos]):
    n += 1
    if n == 1:
        marks["t_first_token"] = time.time()
marks["t_end"] = time.time()
marks.update(prompt_tokens=r.prompt_tokens, prompt_tps=r.prompt_tps, generation_tokens=r.generation_tokens,
             generation_tps=r.generation_tps, peak_memory_gb=r.peak_memory, finish_reason=r.finish_reason, model=MODEL)
print("MARKS " + json.dumps(marks), flush=True)
