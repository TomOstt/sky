# Parse GGML_METAL_GRAPH_DEBUG=1 output: per encoded node lines look like
#   ggml_metal_op_encode_impl: node[  123] - MUL_MAT      blk.0.attn_q ... (concurrent)
# Count, per encode pass (index restarts at 0), the number of encoded (non-noop) nodes, barriers
# (= non-concurrent non-noop nodes), op histogram, and fusion lines.
import re, sys, collections
txt = open(sys.argv[1], errors='replace').read().splitlines()
pat = re.compile(r'node\[\s*(\d+)\] - (\S+)\s+(.*)$')
passes = []; cur = None; last = -1
fus = collections.Counter()
for ln in txt:
    if 'fuse:' in ln:
        fus[ln.split('fuse:')[1].strip()] += 1
    m = pat.search(ln)
    if not m: continue
    idx = int(m.group(1)); op = m.group(2); rest = m.group(3)
    if idx <= last or cur is None:
        cur = {'nodes':0,'noop':0,'conc':0,'barrier':0,'ops':collections.Counter()}; passes.append(cur)
    last = idx
    if '(noop)' in rest: cur['noop'] += 1; continue
    cur['nodes'] += 1; cur['ops'][op] += 1
    if '(concurrent)' in rest: cur['conc'] += 1
    else: cur['barrier'] += 1
print(f"{len(passes)} encode passes")
for i,p in enumerate(passes):
    print(f"pass {i:2d}: encoded={p['nodes']:5d} noop={p['noop']:5d} concurrent={p['conc']:5d} non-concurrent(barrier before)={p['barrier']:5d}  top ops={p['ops'].most_common(12)}")
print("fusion lines:", fus.most_common(20))
for k in ['graph nodes','graph splits','graphs reused','fusion stats','- NORM','- ADD','NORM_MUL','ADD_CHAIN']:
    for ln in txt:
        if k in ln and ('llama_context' in ln or 'ggml_metal' in ln or 'perf' in ln):
            print('  >', ln.strip()[:200])
