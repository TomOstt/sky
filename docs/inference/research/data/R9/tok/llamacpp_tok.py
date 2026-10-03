import json, os, sys, time, subprocess, urllib.request, signal
sys.path.insert(0, os.path.dirname(__file__))
from corpus import CORPUS
S = "/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad"
T = os.path.dirname(os.path.abspath(__file__))
PORT = 18791
srv = subprocess.Popen([f"{S}/src/llama.cpp/build/bin/llama-server", "-m", f"{S}/models/Qwen3-0.6B-Q4_0.gguf",
                        "--port", str(PORT), "-ngl", "0", "-c", "512", "--no-warmup", "-t", "2", "--host", "127.0.0.1"],
                       stdout=open(f"{T}/llama-server.log", "w"), stderr=subprocess.STDOUT)
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())
try:
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200: break
        except Exception: time.sleep(0.5)
    hf = json.load(open(f"{T}/qwen3_hf_tokens.json"))
    res = {}
    for name, s in CORPUS:
        ps = post("/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
        nps = post("/tokenize", {"content": s, "add_special": False, "parse_special": False})["tokens"]
        addsp = post("/tokenize", {"content": s, "add_special": True, "parse_special": True})["tokens"]
        det = post("/detokenize", {"tokens": ps})["content"]
        res[name] = {"parse_special": ps, "no_parse_special": nps, "add_special": addsp, "detok": det}
        h = hf[name]
        m1 = "OK " if ps == h["hf_tokenizers"] else "DIFF"
        m2 = "OK " if nps == h["transformers_split_special"] else "DIFF"
        print(f"{name:24s} llama(parse_special) vs HF default: {m1}   llama(no_parse) vs HF split_special: {m2}   add_special_prefix={addsp[:len(addsp)-len(ps)]}  detok_roundtrip={'OK' if det==s else 'DIFF'}")
        if ps != h["hf_tokenizers"]:
            print("    HF   :", h["hf_tokenizers"][:40]); print("    llama:", ps[:40])
    json.dump(res, open(f"{T}/qwen3_llamacpp_tokens.json", "w"), ensure_ascii=False, indent=0)
finally:
    srv.send_signal(signal.SIGINT); time.sleep(1); srv.kill()
