import json, sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
from corpus import CORPUS
from tokenizers import Tokenizer
from transformers import AutoTokenizer
Q = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/*"))[0]
tk = Tokenizer.from_file(f"{Q}/tokenizer.json")
at = AutoTokenizer.from_pretrained(Q)
out = {}
for name, s in CORPUS:
    a = tk.encode(s, add_special_tokens=False).ids
    b = at.encode(s, add_special_tokens=False)
    c = at(s, add_special_tokens=False, split_special_tokens=True)["input_ids"] if hasattr(at, "split_special_tokens") else None
    dec_full = tk.decode(a, skip_special_tokens=False)
    out[name] = {"text": s, "hf_tokenizers": a, "transformers": b, "transformers_split_special": c,
                 "roundtrip_ok": dec_full == s, "decoded": dec_full}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "qwen3_hf_tokens.json"), "w"), ensure_ascii=False, indent=0)
for k, v in out.items():
    flag = "" if v["hf_tokenizers"] == v["transformers"] else "  <-- tokenizers!=transformers"
    rt = "" if v["roundtrip_ok"] else f"  ROUNDTRIP-MISMATCH decoded={v['decoded']!r}"
    print(f"{k:24s} n={len(v['hf_tokenizers']):4d}{flag}{rt}")
