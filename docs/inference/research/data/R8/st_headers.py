# Read safetensors headers via HTTP range requests and print parameter counts by category.
import json, struct, sys, urllib.request, collections, re
repo = sys.argv[1]
base = f"https://huggingface.co/{repo}/resolve/main/"
def get(url, rng=None):
    req = urllib.request.Request(url)
    if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=60) as r: return r.read()
try:
    idx = json.loads(get(base + "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
except Exception:
    shards = ["model.safetensors"]
tensors = {}
for s in shards:
    n = struct.unpack("<Q", get(base + s, (0, 7)))[0]
    hdr = json.loads(get(base + s, (8, 8 + n - 1)))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        tensors[k] = v
def numel(shape):
    p = 1
    for d in shape: p *= d
    return p
cat = collections.Counter()
dt = collections.Counter()
for k, v in tensors.items():
    n = numel(v["shape"]); dt[v["dtype"]] += n
    kk = k
    if "visual" in kk or "vision" in kk: c = "vision"
    elif kk.startswith("mtp."): c = "mtp"
    elif "embed_tokens" in kk: c = "embed"
    elif "lm_head" in kk: c = "lm_head"
    elif "linear_attn" in kk: c = "linear_attn"
    elif "self_attn" in kk: c = "full_attn"
    elif ".mlp.experts" in kk or "shared_expert" in kk or ".mlp.gate." in kk or "router" in kk: c = "moe"
    elif ".mlp." in kk: c = "mlp"
    elif "norm" in kk: c = "norm"
    else: c = "other:" + re.sub(r"\d+", "N", kk)
    cat[c] += n
tot = sum(cat.values())
print(repo, "total params", tot, "dtypes", dict(dt))
for c, n in sorted(cat.items(), key=lambda x: -x[1]): print(f"  {c:20s} {n:>15,d}  {n/1e9:8.3f} B")
if len(sys.argv) > 2:
    pat = sys.argv[2]
    for k, v in sorted(tensors.items()):
        if re.search(pat, k): print("   ", k, v["shape"], v["dtype"])
