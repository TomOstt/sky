import json, os, sys, time, subprocess, urllib.request, signal, glob
S = "/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad"
T = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
from tokenizers import Tokenizer
Q = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/*"))[0]
at = AutoTokenizer.from_pretrained(Q); tk = Tokenizer.from_file(f"{Q}/tokenizer.json")
CASES = ["a<|fim_prefix|>b", "a<tool_call>b", "a<|vision_pad|>b", "a<|endoftext|>b", "a<think>b", "a<|im_start|>b"]
PORT = 18792
srv = subprocess.Popen([f"{S}/src/llama.cpp/build/bin/llama-server", "-m", f"{S}/models/Qwen3-0.6B-Q4_0.gguf", "--port", str(PORT), "-ngl", "0", "-c", "512", "--no-warmup", "-t", "2", "--host", "127.0.0.1"], stdout=open(f"{T}/llama-server2.log", "w"), stderr=subprocess.STDOUT)
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
try:
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200: break
        except Exception: time.sleep(0.5)
    for s in CASES:
        hf_def = tk.encode(s, add_special_tokens=False).ids
        hf_split = at(s, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        l_ps = post("/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
        l_nps = post("/tokenize", {"content": s, "add_special": False, "parse_special": False})["tokens"]
        print(f"{s!r:22s} HF default={hf_def} HF split_special={hf_split} | llama parse_special={l_ps} no_parse={l_nps}  match(default)={hf_def==l_ps} match(split)={hf_split==l_nps}")
finally:
    srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()
