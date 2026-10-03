#!/usr/bin/env python3
"""R8: exact parameter / byte / FLOP / KV arithmetic for the models in the report.
All architecture numbers come from the config.json files saved in ./configs (fetched from huggingface.co
on 2026-10-01). Parameter totals are cross-checked against the HF API safetensors totals.
Run: python3 model_math.py > model_math.out.md
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
def cfg(name):
    return json.load(open(os.path.join(HERE, "configs", name)))

BW_GBs = 614.0   # M5 Max peak memory bandwidth (task brief; see notes for verification status)

HF_TOTAL = {  # from https://huggingface.co/api/models/<repo> 'safetensors.total'
    "Qwen3-32B": 32762123264, "Qwen3-8B": 8190735360, "Qwen3-0.6B": 751632384,
    "Qwen3-1.7B": 2031739904, "Qwen3-4B": 4022468096, "Qwen3-30B-A3B": 30532122624,
    "Llama-3.1-8B": 8030261248, "Llama-3.1-70B": 70553706496,
    "gpt-oss-20b": 20914757184, "gpt-oss-120b": 116829156672,
}

def dense(name, c, qk_norm):
    H, L, nh, nkv = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or H // nh
    I, V = c["intermediate_size"], c["vocab_size"]
    tied = c.get("tie_word_embeddings", False)
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H + (2 * hd if qk_norm else 0)
    mlp = 3 * H * I
    norms = 2 * H
    per_layer = attn + mlp + norms
    emb = V * H
    head = 0 if tied else V * H
    total = emb + head + L * per_layer + H
    linear = L * (attn - (2 * hd if qk_norm else 0) + mlp)          # quantizable matmul weights in the blocks
    return dict(name=name, H=H, L=L, nh=nh, nkv=nkv, hd=hd, I=I, V=V, tied=tied,
                emb=emb, head=head, attn_layer=attn, mlp_layer=mlp, total=total,
                linear=linear, active_linear=linear, full_attn_layers=L, kv_heads=nkv,
                attn_q_heads=nh, window_layers=0, window=0, moe=None)

def qwen3_moe(name, c):
    H, L, nh, nkv, hd = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k, Im, V = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], c["vocab_size"]
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H + 2 * hd
    router = E * H
    experts = E * 3 * H * Im
    per_layer = attn + router + experts + 2 * H
    emb = V * H; head = 0 if c["tie_word_embeddings"] else V * H
    total = emb + head + L * per_layer + H
    linear = L * (attn - 2 * hd + router + experts)
    active_linear = L * (attn - 2 * hd + router + k * 3 * H * Im)
    return dict(name=name, H=H, L=L, nh=nh, nkv=nkv, hd=hd, I=Im, V=V, tied=c["tie_word_embeddings"],
                emb=emb, head=head, attn_layer=attn, mlp_layer=experts, total=total, linear=linear,
                active_linear=active_linear, full_attn_layers=L, kv_heads=nkv, attn_q_heads=nh,
                window_layers=0, window=0,
                moe=dict(E=E, k=k, expert_params=3 * H * Im, expert_bytes_note="per expert per layer"))

def gpt_oss(name, c):
    H, L, nh, nkv, hd = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    E, k, I, V = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"], c["vocab_size"]
    attn_w = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    attn_b = nh * hd + 2 * nkv * hd + H + nh            # q,k,v,o biases + sinks
    router = E * H + E
    exp_w = E * (H * 2 * I + I * H)                      # gate_up fused + down (MXFP4 on disk)
    exp_b = E * (2 * I + H)                              # BF16 biases
    norms = 2 * H
    per_layer = attn_w + attn_b + router + exp_w + exp_b + norms
    emb = V * H; head = V * H
    total = emb + head + L * per_layer + H
    n_sw = sum(1 for t in c["layer_types"] if t == "sliding_attention")
    return dict(name=name, H=H, L=L, nh=nh, nkv=nkv, hd=hd, I=I, V=V, tied=False, emb=emb, head=head,
                attn_layer=attn_w + attn_b, mlp_layer=exp_w, total=total,
                linear=L * (attn_w + router + exp_w), active_linear=L * (attn_w + router + k * (H * 2 * I + I * H)),
                exp_w_total=L * exp_w, exp_w_active=L * k * (H * 2 * I + I * H),
                nonexp_linear=L * (attn_w + router),
                full_attn_layers=L - n_sw, window_layers=n_sw, window=c["sliding_window"], kv_heads=nkv,
                attn_q_heads=nh, moe=dict(E=E, k=k, expert_params=H * 2 * I + I * H))

M = {}
M["Qwen3-32B"] = dense("Qwen3-32B", cfg("Qwen_Qwen3-32B.json"), True)
M["Qwen3-8B"] = dense("Qwen3-8B", cfg("Qwen_Qwen3-8B.json"), True)
M["Qwen3-4B"] = dense("Qwen3-4B", cfg("Qwen_Qwen3-4B.json"), True)
M["Qwen3-1.7B"] = dense("Qwen3-1.7B", cfg("Qwen_Qwen3-1.7B.json"), True)
M["Qwen3-0.6B"] = dense("Qwen3-0.6B", cfg("Qwen_Qwen3-0.6B.json"), True)
M["Qwen3-30B-A3B"] = qwen3_moe("Qwen3-30B-A3B", cfg("Qwen_Qwen3-30B-A3B.json"))
M["Llama-3.1-8B"] = dense("Llama-3.1-8B", cfg("unsloth_Meta-Llama-3.1-8B-Instruct.json"), False)
M["Llama-3.1-70B"] = dense("Llama-3.1-70B", cfg("unsloth_Meta-Llama-3.1-70B-Instruct.json"), False)
M["gpt-oss-20b"] = gpt_oss("gpt-oss-20b", cfg("openai_gpt-oss-20b.json"))
M["gpt-oss-120b"] = gpt_oss("gpt-oss-120b", cfg("openai_gpt-oss-120b.json"))

def gb(x): return x / 1e9

print("# R8 model arithmetic (generated by model_math.py)\n")
print(f"Bandwidth used for rooflines: {BW_GBs:.0f} GB/s (decimal GB). Rooflines assume 100% of peak; real kernels reach ~70-90%.\n")

print("## 1. Architecture (from config.json)\n")
print("| model | hidden | layers | q heads | kv heads | head_dim | FFN (I or moe I) | vocab | tied | experts (top-k) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in M.values():
    e = f"{m['moe']['E']} (top-{m['moe']['k']})" if m["moe"] else "-"
    print(f"| {m['name']} | {m['H']} | {m['L']} | {m['nh']} | {m['nkv']} | {m['hd']} | {m['I']} | {m['V']} | {m['tied']} | {e} |")

print("\n## 2. Parameter counts (computed) vs HF safetensors totals\n")
print("| model | embed | lm_head | attn/layer | FFN/layer (all experts) | total computed | HF total | match | active linear params / token |")
print("|---|---|---|---|---|---|---|---|---|")
for k, m in M.items():
    hf = HF_TOTAL.get(k)
    ok = "exact" if hf == m["total"] else (f"diff {m['total']-hf:+,d}" if hf else "n/a")
    act = m["active_linear"] + m["head"] if not m["tied"] else m["active_linear"] + m["emb"]
    print(f"| {k} | {m['emb']:,} | {m['head']:,} | {m['attn_layer']:,} | {m['mlp_layer']:,} | {m['total']:,} | {hf:,} | {ok} | {act/1e9:.3f} B |")

# ---------- bytes per decode token ----------
def bytes_per_token(m, fmt):
    """Weight bytes streamed per decoded token (batch 1). Embedding: one row only (negligible).
    lm_head always read in full (if tied, the embedding matrix is read as lm_head)."""
    head_params = m["V"] * m["H"]   # lm_head (or tied embedding used as lm_head)
    if m["name"].startswith("gpt-oss"):
        exp = m["exp_w_active"]; nonexp = m["nonexp_linear"]
        if fmt == "native":   # MXFP4 experts (4.25 bpw), BF16 everything else
            return exp * 4.25 / 8 + nonexp * 2 + head_params * 2
        if fmt == "mlx_mxfp4_q8":  # MXFP4 experts, 8-bit g64 (8.5bpw) others incl head
            return exp * 4.25 / 8 + (nonexp + head_params) * 8.5 / 8
        if fmt == "mlx4":
            return (exp + nonexp + head_params) * 4.5 / 8
        if fmt == "bf16":
            return (exp + nonexp + head_params) * 2
        return None
    lin = m["active_linear"]
    if fmt == "bf16": return (lin + head_params) * 2
    if fmt == "mlx8": return (lin + head_params) * 8.5 / 8
    if fmt == "mlx6": return (lin + head_params) * 6.5 / 8
    if fmt == "mlx4": return (lin + head_params) * 4.5 / 8
    if fmt == "mlx4g128": return (lin + head_params) * 4.25 / 8
    if fmt == "mlx3": return (lin + head_params) * 3.5 / 8
    if fmt == "mlx2": return (lin + head_params) * 2.5 / 8
    if fmt in ("q4_k_m", "q4_0"):
        L = m["L"]; n_more = sum(1 for i in range(L) if (i < L // 8 or i >= 7 * L // 8 or (i - L // 8) % 3 == 2))
        H, I = m["H"], m["I"]
        if m["moe"]:   # down proj of *active* experts
            down_layer = m["moe"]["k"] * H * I
        else:
            down_layer = H * I
        v_layer = H * m["nkv"] * m["hd"]
        head = head_params * 6.5625 / 8      # output.weight -> Q6_K (llama-quant.cpp)
        if fmt == "q4_k_m":
            q6 = n_more * (down_layer + v_layer)
            return (lin - q6) * 4.5 / 8 + q6 * 6.5625 / 8 + head
        else:  # Q4_0 (+imatrix: first L/8 ffn_down in Q4_1 = 5.0 bpw)
            q41 = (L // 8) * down_layer
            return (lin - q41) * 4.5 / 8 + q41 * 5.0 / 8 + head
    return None

fmts = [("bf16", "BF16"), ("mlx8", "MLX 8b g64 (8.5)"), ("mlx6", "MLX 6b g64 (6.5)"), ("mlx4", "MLX 4b g64 (4.5)"),
        ("mlx4g128", "MLX 4b g128 (4.25)"), ("mlx3", "MLX 3b g64 (3.5)"), ("q4_k_m", "GGUF Q4_K_M"), ("q4_0", "GGUF Q4_0(+imatrix)")]
print("\n## 3. Weight bytes streamed per decoded token (GB) and roofline tok/s at %.0f GB/s (context ~0)\n" % BW_GBs)
print("| model | " + " | ".join(f for _, f in fmts) + " |")
print("|---|" + "---|" * len(fmts))
for k, m in M.items():
    if k.startswith("gpt-oss"): continue
    cells = []
    for f, _ in fmts:
        b = bytes_per_token(m, f)
        cells.append(f"{gb(b):.2f} GB / {BW_GBs*1e9/b:.0f}")
    print(f"| {k} | " + " | ".join(cells) + " |")
print("\ngpt-oss (experts MXFP4 4.25 bpw on disk):\n")
print("| model | native (MXFP4 experts + BF16 rest) | MXFP4 experts + 8-bit rest | all 4-bit g64 | BF16 |")
print("|---|---|---|---|---|")
for k in ("gpt-oss-20b", "gpt-oss-120b"):
    m = M[k]
    cells = []
    for f in ("native", "mlx_mxfp4_q8", "mlx4", "bf16"):
        b = bytes_per_token(m, f); cells.append(f"{gb(b):.2f} GB / {BW_GBs*1e9/b:.0f}")
    print(f"| {k} | " + " | ".join(cells) + " |")
    print(f"|  (expert weights total {m['exp_w_total']/1e9:.2f} B params; active {m['exp_w_active']/1e9:.3f} B; non-expert linear {m['nonexp_linear']/1e9:.3f} B; lm_head {m['V']*m['H']/1e9:.3f} B) |||||")

# ---------- KV cache ----------
print("\n## 4. KV cache bytes per token of context\n")
print("fp16 = 2 B/elem; int8 (MLX-style affine g64, fp16 scale+bias) = 8.5 bits/elem; 4-bit g64 = 4.5 bits/elem.\n")
print("| model | KV layers (full) | elems/token | fp16 B/token | int8(g64) B/token | 4-bit(g64) B/token | fp16 GB @8K | fp16 GB @32K | fp16 GB @128K |")
print("|---|---|---|---|---|---|---|---|---|")
for k, m in M.items():
    el = m["full_attn_layers"] * 2 * m["kv_heads"] * m["hd"]
    fixed = m["window_layers"] * 2 * m["kv_heads"] * m["hd"] * m["window"] * 2  # bytes, fp16, sliding layers
    def at(T): return gb(el * 2 * T + fixed)
    print(f"| {k} | {m['full_attn_layers']}" + (f" (+{m['window_layers']} SWA-{m['window']})" if m['window_layers'] else "") +
          f" | {el:,} | {el*2:,} | {el*8.5/8:,.0f} | {el*4.5/8:,.0f} | {at(8192):.2f} | {at(32768):.2f} | {at(131072):.2f} |")

# ---------- FLOPs ----------
print("\n## 5. FLOPs\n")
print("Decode FLOPs/token = 2*(active matmul params incl. lm_head) + attention 4*n_q_heads*head_dim*T per full-attn layer (QK^T and PV), T = context.\n")
print("Prefill FLOPs for P tokens = 2*P*(active matmul params excl. lm_head) + 2*V*H (last-token logits) + causal attention 2*n_q_heads*head_dim*P^2 per layer.\n")
print("| model | matmul GFLOP/token (decode) | attn MFLOP per token per ctx-token | attn GFLOP/token @4K | @32K | prefill TFLOP @P=1K | @P=4K | @P=32K | attn share @P=32K |")
print("|---|---|---|---|---|---|---|---|---|")
for k, m in M.items():
    head = m["V"] * m["H"]
    mm = 2 * (m["active_linear"] + head)
    a_coef = 4 * m["nh"] * m["hd"] * m["full_attn_layers"]      # FLOPs per context token per decoded token
    def pre(P):
        lin = 2 * P * m["active_linear"] + 2 * head
        att = 2 * m["nh"] * m["hd"] * P * P * m["full_attn_layers"]  # causal: (4*nh*hd*P^2)/2
        if m["window_layers"]:
            att += 4 * m["nh"] * m["hd"] * P * min(P, m["window"]) * m["window_layers"]
        return lin, att
    l32, a32 = pre(32768)
    print(f"| {k} | {mm/1e9:.1f} | {a_coef/1e6:.2f} | {a_coef*4096/1e9:.1f} | {a_coef*32768/1e9:.1f} | "
          f"{sum(pre(1024))/1e12:.1f} | {sum(pre(4096))/1e12:.1f} | {sum(pre(32768))/1e12:.0f} | {a32/(l32+a32)*100:.0f}% |")

# ---------- Qwen3-32B decode time model ----------
m = M["Qwen3-32B"]
print("\n## 6. Qwen3-32B decode roofline vs context (MLX 4-bit g64 weights)\n")
w = bytes_per_token(m, "mlx4")
el = m["full_attn_layers"] * 2 * m["kv_heads"] * m["hd"]
print("| context T | weights GB | KV fp16 GB | total GB | roofline tok/s fp16-KV | KV 8-bit total GB | tok/s | KV 4-bit total GB | tok/s |")
print("|---|---|---|---|---|---|---|---|---|")
for T in (0, 1024, 4096, 8192, 16384, 32768, 40960):
    kv16 = el * 2 * T; kv8 = el * 8.5 / 8 * T; kv4 = el * 4.5 / 8 * T
    print(f"| {T} | {gb(w):.2f} | {gb(kv16):.2f} | {gb(w+kv16):.2f} | {BW_GBs*1e9/(w+kv16):.1f} | {gb(w+kv8):.2f} | {BW_GBs*1e9/(w+kv8):.1f} | {gb(w+kv4):.2f} | {BW_GBs*1e9/(w+kv4):.1f} |")

print("\n## 7. Required bits/weight for a given decode rate on Qwen3-32B (no speculation, T~0)\n")
params = m["active_linear"] + m["V"] * m["H"]
for tps in (35, 40, 50, 60):
    bpw = BW_GBs * 1e9 / tps * 8 / params
    print(f"- {tps} tok/s at 100% of {BW_GBs:.0f} GB/s needs <= {bpw:.2f} bits/weight averaged over {params/1e9:.2f} B streamed params; at 85% efficiency: {bpw*0.85:.2f} bpw")
