# Qwen3-32B decode/prefill traffic & compute budget (config from HF Qwen/Qwen3-32B config.json)
L, H, NQ, NKV, D, I, V = 64, 5120, 64, 8, 128, 25600, 151936
q = H*NQ*D; k = H*NKV*D; v = H*NKV*D; o = NQ*D*H
gate = H*I; up = H*I; down = I*H
per_layer = q+k+v+o+gate+up+down
lin = per_layer*L
emb = V*H; head = V*H
print(f"per-layer linear params {per_layer/1e6:.1f}M; all linear {lin/1e9:.2f}B; embed {emb/1e9:.3f}B; lm_head {head/1e9:.3f}B; total {(lin+emb+head)/1e9:.2f}B")
for name,bits in [("MLX q4 g64 (4.5 b/w)",4.5),("Q4_0 (4.5 b/w)",4.5),("Q4_K (~4.5 b/w)",4.5),("pure 4.0 b/w",4.0)]:
    wbytes = (lin+head)*bits/8
    print(f"{name}: weight bytes read per decode token = {wbytes/1e9:.2f} GB")
wbytes = (lin+head)*4.5/8
kv_tok_fp16 = 2*L*NKV*D*2
print(f"KV bytes/token fp16 = {kv_tok_fp16} B = {kv_tok_fp16/1024:.0f} KiB; 8-bit ~{kv_tok_fp16/2/1024:.0f} KiB (+scales)")
for ctx in [1024, 4096, 8192, 32768, 40960, 131072]:
    a = ctx*kv_tok_fp16
    print(f"ctx {ctx:6d}: KV read {a/1e9:6.2f} GB = {100*a/wbytes:5.1f}% of weight bytes (fp16 KV); {50*a/wbytes:5.1f}% with 8-bit KV")
for frac in [0.10,0.25,0.5,1.0]:
    print(f"attention traffic = {int(frac*100)}% of weights at ctx ~ {frac*wbytes/kv_tok_fp16:,.0f} tokens (fp16 KV)")
for bw in [546, 614, 820]:
    print(f"BW {bw} GB/s -> decode upper bound at short ctx {bw/ (wbytes/1e9):.1f} tok/s; at 80% util {0.8*bw/(wbytes/1e9):.1f}")
# prefill flops
for P in [512, 2048, 4096, 16384]:
    lin_flops = 2*lin*P + 2*head*1  # lm_head only for last token
    # attention: QK^T and PV, causal ~ half: 2 matmuls * 2 flops * P^2/2 * D * NQ * L
    attn_flops = 2*2*(P*P/2)*D*NQ*L
    print(f"prefill P={P}: linear {lin_flops/1e12:.1f} TFLOP, attention {attn_flops/1e12:.2f} TFLOP ({100*attn_flops/(lin_flops+attn_flops):.1f}% of total)")
# decode arithmetic intensity
print(f"decode GEMV intensity ~ 2 flops / {4.5/8} B = {2/(4.5/8):.2f} flop/B")
