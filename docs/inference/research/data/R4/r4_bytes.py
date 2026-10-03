# R4: bytes streamed per decode token for MLX 4-bit Qwen3 checkpoints.
# Reads tensor sizes straight from the safetensors headers (no weights loaded).
import json, struct, sys, glob, os, collections

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

DT = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I32": 4}

def analyze(name):
    d = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--mlx-community--{name}/snapshots/*/"))[0]
    cfg = json.load(open(d + "config.json"))
    tensors = {}
    for f in sorted(glob.glob(d + "*.safetensors")):
        for k, v in header(f).items():
            if k == "__metadata__":
                continue
            nb = v["data_offsets"][1] - v["data_offsets"][0]
            tensors[k] = (v["dtype"], v["shape"], nb)
    cats = collections.Counter()
    dtypes = collections.Counter()
    for k, (dt, shp, nb) in tensors.items():
        dtypes[(k.rsplit(".", 1)[-1], dt)] += 1
        if "embed_tokens" in k:
            cats["embed_tokens"] += nb
        elif "lm_head" in k:
            cats["lm_head"] += nb
        elif any(p in k for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            cats["attn_linear"] += nb
        elif any(p in k for p in ("gate_proj", "up_proj", "down_proj")):
            cats["mlp_linear"] += nb
        elif "norm" in k:
            cats["norms"] += nb
        else:
            cats["other:" + k] += nb
    total_file = sum(nb for _, _, nb in tensors.values())
    tied = cfg.get("tie_word_embeddings", False)
    # Bytes streamed per decode token (weights only):
    linear = cats["attn_linear"] + cats["mlp_linear"]
    head = cats["embed_tokens"] if tied else cats["lm_head"]
    norms = cats["norms"]
    per_tok = linear + head + norms
    L = cfg["num_hidden_layers"]; nkv = cfg["num_key_value_heads"]; hd = cfg["head_dim"]
    kv_per_pos = 2 * L * nkv * hd * 2  # K+V, fp16/bf16 bytes, per context position
    print(f"== {name}  (tie_word_embeddings={tied})")
    print(f"  dtypes: {dict(dtypes)}")
    for k, v in sorted(cats.items()):
        print(f"  {k:14s} {v/1e6:10.2f} MB")
    print(f"  file total     {total_file/1e6:10.2f} MB")
    print(f"  per-token weight bytes (linears + {'tied embed as lm_head' if tied else 'lm_head'} + norms): "
          f"{per_tok/1e6:.2f} MB = {per_tok/2**30:.4f} GiB")
    print(f"  KV cache bytes read per context position per token: {kv_per_pos} B "
          f"({kv_per_pos/1e3:.1f} KB)")
    # effective bits per weight of linear layers
    n_params = 0
    for k, (dt, shp, nb) in tensors.items():
        if k.endswith(".weight") and dt == "U32" and ("embed" not in k):
            n_params += shp[0] * shp[1] * 8
    lin_bytes = sum(nb for k, (dt, shp, nb) in tensors.items()
                    if ("embed" not in k) and ("norm" not in k))
    print(f"  quantized (non-embed) params: {n_params/1e9:.4f} B; bits/weight incl. scales+biases: "
          f"{8*lin_bytes/n_params:.3f}")
    return dict(name=name, per_tok=per_tok, kv_per_pos=kv_per_pos, cats=dict(cats))

if __name__ == "__main__":
    out = [analyze(n) for n in ("Qwen3-0.6B-4bit", "Qwen3-8B-4bit")]
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "r4_bytes.json"), "w"), indent=1)
