"""Bytes read per generated token for Inkling 2 Mini, from real GGUF tensor sizes.
Usage: python ceiling.py <model.gguf> [bandwidth_GBps]
"""
import re, sys
from collections import defaultdict
from gguf import GGUFReader

path = sys.argv[1]; bw = float(sys.argv[2]) if len(sys.argv) > 2 else None
r = GGUFReader(path)
LOOP = set(range(6, 18)); PASSES = 3; TOPK, NEXP = 8, 128

per_layer = defaultdict(lambda: {'dense': 0, 'expert': 0})
other = defaultdict(int)
for t in r.tensors:
    m = re.match(r'blk\.(\d+)\.(.+)', t.name)
    if m:
        kind = 'expert' if '_exps' in m.group(2) else 'dense'
        per_layer[int(m.group(1))][kind] += int(t.n_bytes)
    else:
        other[t.name] += int(t.n_bytes)

GB = 1e9
dense = expert_active = 0.0
for L, s in per_layer.items():
    reps = PASSES if L in LOOP else 1
    dense += s['dense'] * reps
    expert_active += s['expert'] * TOPK / NEXP * reps
lm_head = other.get('output.weight', other.get('token_embd.weight', 0))
total_file = sum(int(t.n_bytes) for t in r.tensors)
expert_total = sum(s['expert'] for s in per_layer.values())

print(f"file tensors        : {total_file/GB:6.2f} GB  (experts {expert_total/GB:.2f} GB = {expert_total/total_file:.0%})")
print(f"per token, dense    : {dense/GB:6.3f} GB  (attention/norms/router, looped layers x{PASSES})")
print(f"per token, experts  : {expert_active/GB:6.3f} GB  ({TOPK}/{NEXP} of each layer's experts)")
print(f"per token, LM head  : {lm_head/GB:6.3f} GB")
tok = dense + expert_active + lm_head
print(f"TOTAL bytes/token   : {tok/GB:6.3f} GB")
if bw:
    print(f"decode ceiling      : {bw/(tok/GB):6.0f} tok/s at {bw:.0f} GB/s measured bandwidth")
