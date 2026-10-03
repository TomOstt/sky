import json
c=json.load(open("configs/qwen3-32b.json"))
L,h,H,KV,hd,I,V=c["num_hidden_layers"],c["hidden_size"],c["num_attention_heads"],c["num_key_value_heads"],c["head_dim"],c["intermediate_size"],c["vocab_size"]
q=h*H*hd; k=h*KV*hd; v=k; o=H*hd*h
mlp=3*h*I
attn=q+k+v+o
norms=2*h+2*hd
per_layer_lin=attn+mlp
emb=V*h; head=V*h
total=L*(per_layer_lin+norms)+emb+head+h
print(f"q={q:,} k={k:,} v={v:,} o={o:,} attn={attn:,} mlp={mlp:,} per_layer_linear={per_layer_lin:,}")
print(f"total params={total:,}  non-embedding(excl emb+head)={total-emb-head:,}")
bpw = 0.5 + 4/64   # 4-bit + bf16 scale + bf16 bias per 64
lin = L*per_layer_lin
wbytes = (lin+head)*bpw + L*norms*2 + h*2 + (h//8*4 + 2*(h//64)*2)  # + one embedding row
print(f"bits/weight={bpw*8}  decode weight bytes/token={wbytes:,.0f} B = {wbytes/1e9:.3f} GB = {wbytes/2**30:.3f} GiB")
file_bytes = (lin+emb+head)*bpw + (L*norms+h)*2
print(f"predicted MLX file bytes={file_bytes:,.0f} (actual index total_size 18,429,667,328)")
flops = 2*(lin+head)
print(f"decode matmul FLOPs/token={flops/1e9:.2f} GFLOP; intensity={flops/wbytes:.2f} FLOP/B")
attn_flops_per_ctx = L*H*hd*2*2   # QK^T and PV, 2 FLOP per MAC
print(f"attention FLOPs/token = {attn_flops_per_ctx:,} * T  (T=4096 -> {attn_flops_per_ctx*4096/1e9:.2f} GFLOP)")
kv = 2*L*KV*hd*2
print(f"KV bytes/token bf16={kv:,} B = {kv/1024:.0f} KiB; 4k ctx={kv*4096/2**30:.2f} GiB; 32k={kv*32768/2**30:.2f} GiB; 40960={kv*40960/2**30:.2f} GiB")
for T in [512,4096,32768]:
    print(f"  bytes/token at ctx {T}: weights {wbytes/1e9:.2f} GB + KV read {kv*T/1e9:.3f} GB = {(wbytes+kv*T)/1e9:.2f} GB")
for bw in [400,546,614,800]:
    print(f"  roofline decode @ {bw} GB/s, ctx~0: {bw/(wbytes/1e9):.1f} tok/s; ctx 4096: {bw/((wbytes+kv*4096)/1e9):.1f} tok/s")
