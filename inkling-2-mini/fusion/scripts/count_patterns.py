"""Count consecutive op patterns in the last decode graph of a GGML_SCHED_DEBUG=2 dump.
Usage: python count_patterns.py sched.log"""
import re, sys
from collections import Counter
node_re = re.compile(r'node #\s*(\d+) \(\s*([^)]*?)\s*\):\s*(\S*)')
graphs, cur = [], []
for l in open(sys.argv[1], encoding='utf-8', errors='replace'):
    m = node_re.search(l)
    if not m: continue
    if int(m.group(1)) == 0 and cur: graphs.append(cur); cur = []
    cur.append(m.group(2))
graphs.append(cur)
ops = graphs[-1]
s = ' '.join(ops)
patterns = {
    'conv tail: SSM_CONV ADD SCALE': 'SSM_CONV ADD SCALE',
    'dyn tap: CONT MUL ADD': 'CONT MUL ADD',
    'dyn tap w/ proj: CONT MUL_MAT MUL ADD': 'CONT MUL_MAT MUL ADD',
    'dyn tail: ADD ADD SCALE': 'ADD ADD SCALE',
    'conv head: CONCAT CPY': 'CONCAT CPY',
    'conv head: CONCAT SET_ROWS': 'CONCAT SET_ROWS',
    'control: RMS_NORM RMS_NORM CONCAT MUL_MAT': 'RMS_NORM RMS_NORM CONCAT MUL_MAT',
    'control tail: MUL_MAT TANH MUL_MAT TANH': 'MUL_MAT TANH MUL_MAT TANH',
    'control combine: MUL SUB MUL ADD': 'MUL SUB MUL ADD',
    'mix: RMS_NORM REPEAT MUL_MAT SOFT_MAX': 'RMS_NORM REPEAT MUL_MAT SOFT_MAX',
    'mix tail: SOFT_MAX CONT MUL_MAT': 'SOFT_MAX CONT MUL_MAT',
}
for k, p in patterns.items():
    print(f"{s.count(p):5d}  {k}")
# what precedes each SSM_CONV (3 ops) and each CONCAT
ctx = Counter(' '.join(ops[i-3:i]) for i, o in enumerate(ops) if o == 'SSM_CONV')
print('\nbefore SSM_CONV:'); [print(f"{n:5d}  {k}") for k, n in ctx.most_common(4)]
ctx = Counter(' '.join(ops[i:i+3]) for i, o in enumerate(ops) if o == 'CONCAT')
print('\nCONCAT followed by:'); [print(f"{n:5d}  {k}") for k, n in ctx.most_common(6)]
