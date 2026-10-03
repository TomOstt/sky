# R5 measurement B: bytes of weights read per decoded token, from GGUF tensor info (gguf-py)
import sys, collections
from gguf import GGUFReader, GGMLQuantizationType as Q
def analyze(path):
    r = GGUFReader(path)
    kv = {f.name: f for f in r.fields.values()}
    def kvval(name):
        f = r.fields.get(name)
        if f is None: return None
        try: return f.contents()
        except Exception: return None
    arch = kvval('general.architecture')
    print(f"== {path.split('/')[-1]}  arch={arch}  gguf_version={kvval('GGUF.version')}  alignment={kvval('general.alignment')}  n_tensors={len(r.tensors)} n_kv={len(r.fields)}")
    for k in ['block_count','embedding_length','feed_forward_length','attention.head_count','attention.head_count_kv','attention.key_length','vocab_size','context_length']:
        print(f"   {arch}.{k} = {kvval(arch+'.'+k)}")
    print("   general.file_type =", kvval('general.file_type'), " quantization_version =", kvval('general.quantization_version'))
    by_type = collections.Counter(); by_type_n = collections.Counter(); total = 0
    names = {t.name: t for t in r.tensors}
    has_output = 'output.weight' in names
    rows = []
    for t in r.tensors:
        nb = int(t.n_bytes); ne = int(t.n_elements)
        tn = Q(t.tensor_type).name
        by_type[tn] += nb; by_type_n[tn] += 1; total += nb
        rows.append((t.name, tn, ne, nb))
    print(f"   output.weight present: {has_output}")
    emb = names['token_embd.weight']
    emb_bytes = int(emb.n_bytes); emb_type = Q(emb.tensor_type).name
    n_embd = int(emb.shape[0]) if len(emb.shape) else None
    print(f"   token_embd: type={emb_type} shape={list(map(int,emb.shape))} bytes={emb_bytes/1e6:.2f} MB")
    if has_output:
        o = names['output.weight']; print(f"   output: type={Q(o.tensor_type).name} shape={list(map(int,o.shape))} bytes={int(o.n_bytes)/1e6:.2f} MB")
    print("   bytes by type:")
    for k,v in sorted(by_type.items(), key=lambda x:-x[1]):
        print(f"     {k:8s} {by_type_n[k]:4d} tensors {v/1e6:10.2f} MB")
    # per-layer-role breakdown (blk.0)
    print("   blk.0 tensors:")
    for (n,tn,ne,nb) in rows:
        if n.startswith('blk.0.'):
            print(f"     {n:32s} {tn:6s} ne={ne:>12d} bytes={nb:>12d}  bpw={8*nb/ne:.3f}")
    # bytes read per decode token
    per_tok = total
    if has_output:
        per_tok -= emb_bytes  # token_embd only used for get_rows (one row) when an output head exists
        per_tok += emb_bytes // int(emb.shape[1])  # one row read by get_rows
        note = "total - token_embd + 1 row of token_embd"
    else:
        note = "total (tied embeddings: token_embd is the output head, read fully) + 1 row"
        per_tok += emb_bytes // int(emb.shape[1])
    print(f"   file tensor-data total = {total/1e6:.2f} MB ; weight bytes read per token = {per_tok/1e6:.2f} MB  [{note}]")
    return per_tok, kvval
if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
