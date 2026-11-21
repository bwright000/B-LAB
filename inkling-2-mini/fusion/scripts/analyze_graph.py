"""Attribute every op of one decode graph (GGML_SCHED_DEBUG=2 dump) to the Inkling helper that built it.
Usage: python analyze_graph.py sched.log
Segments use the cb() names in src/models/inkling2.cpp (Qcur/Kcur/Vcur/Rcur, kq_b, attn_out, ffn_inp, ffn_out, l_out).
"""
import re, sys
from collections import Counter, defaultdict

lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().splitlines()
node_re = re.compile(r'node #\s*(\d+) \(\s*([^)]*?)\s*\):\s*(\S*)')
graphs, cur = [], []
for l in lines:
    m = node_re.search(l)
    if not m:
        continue
    idx, op, name = int(m.group(1)), m.group(2), m.group(3)
    if idx == 0 and cur:
        graphs.append(cur); cur = []
    cur.append((idx, op, name))
if cur:
    graphs.append(cur)
print(f"graphs dumped: {len(graphs)}; node counts: {[len(g) for g in graphs]}")
g = graphs[-1]   # last = a single-token decode
print(f"analysing last graph: {len(g)} non-view nodes")

ops = Counter(op for _, op, _ in g)
print("\n== op types (top 25) ==")
for op, n in ops.most_common(25):
    print(f"{n:6d}  {op}")

# segment: the op belongs to the region that ends at the next named checkpoint
marks = ('Qcur', 'Kcur', 'Vcur', 'Rcur', 'kq_b', 'attn_out', 'ffn_inp', 'ffn_out', 'l_out', 'result_norm', 'result_output')
region_of = {'Qcur': 'A1 inputs: attn-res mix + loop control + norms + Q proj + Q clip',
             'Kcur': 'A2 K proj + K conv + K clip', 'Vcur': 'A3 V proj + V conv + V norm',
             'Rcur': 'A4 rel-pos proj + R clip', 'kq_b': 'A5 rel-pos bias table + sinks',
             'attn_out': 'A6 attention + gate + out proj + post-conv',
             'ffn_inp': 'B1 residual', 'ffn_out': 'C  MoE (ffn-res mix + loop control + norm + experts + post-conv)',
             'l_out': 'D  residual', 'result_norm': 'E  output mix + norm', 'result_output': 'F  lm head + softcap'}
regions = defaultdict(Counter); pending = []
for _, op, name in g:
    pending.append(op)
    base = name.rsplit('-', 1)[0]
    if base in marks:
        regions[region_of[base]].update(pending); pending = []
if pending:
    regions['Z  tail'].update(pending)

print("\n== ops per region (whole token) ==")
MM = ('MUL_MAT', 'MUL_MAT_ID', 'FLASH_ATTN_EXT')
for r in sorted(regions):
    c = regions[r]; tot = sum(c.values()); mm = sum(v for k, v in c.items() if k in MM)
    top = ', '.join(f"{k}:{v}" for k, v in c.most_common(6))
    print(f"{tot:6d} ops ({tot - mm:5d} glue)  {r}\n        {top}")
