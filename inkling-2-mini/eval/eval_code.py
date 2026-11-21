"""Coding eval against a running llama-server (OpenAI API).
Usage: python eval_code.py --label NAME --effort low|medium|high [--port 8091] [--mbpp 100] [--max-tokens 4096]
Appends a summary row to ../bench/code_eval.md and per-problem JSONL to results/NAME.jsonl
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True); ap.add_argument('--effort', default='low')
ap.add_argument('--port', type=int, default=8091); ap.add_argument('--mbpp', type=int, default=100)
ap.add_argument('--humaneval', type=int, default=164); ap.add_argument('--max-tokens', type=int, default=4096)
ap.add_argument('--temp', type=float, default=1.0)
a = ap.parse_args()

SYS = "You are a precise Python programmer. Reply with one complete ```python code block containing the full solution (imports + function). No tests, no explanation."
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

def ask(prompt):
    body = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": a.temp, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "chat_template_kwargs": {"reasoning_effort": a.effort}}
    t = time.time()
    r = requests.post(f"http://127.0.0.1:{a.port}/v1/chat/completions", json=body, timeout=900).json()
    dt = time.time() - t
    ch = r["choices"][0]; txt = ch["message"].get("content") or ""
    u = r.get("usage", {}); tm = r.get("timings", {})
    return txt, ch.get("finish_reason"), u.get("completion_tokens", 0), dt, tm.get("predicted_per_second", 0)

def extract(txt):
    blocks = CODE_RE.findall(txt)
    return max(blocks, key=len) if blocks else txt

def run_py(src, timeout=15):
    with tempfile.TemporaryDirectory(dir=os.path.join(HERE, 'sandbox')) as td:
        f = os.path.join(td, 'prog.py'); open(f, 'w', encoding='utf-8').write(src)
        try:
            p = subprocess.run([sys.executable, '-I', f], cwd=td, capture_output=True, timeout=timeout)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False

def tasks():
    he = pd.read_parquet(os.path.join(HERE, 'data', 'humaneval.parquet')).head(a.humaneval)
    for _, r in he.iterrows():
        prompt = f"Complete this Python function:\n\n```python\n{r['prompt']}```"
        def build(code, r=r):
            header = r['prompt'] if f"def {r['entry_point']}" not in code else ''
            imports = '\n'.join(l for l in r['prompt'].splitlines() if l.startswith(('import ', 'from ')))
            return f"{imports}\n{header}{code}\n\n{r['test']}\ncheck({r['entry_point']})\n"
        yield 'humaneval', r['task_id'], prompt, build
    mb = pd.read_parquet(os.path.join(HERE, 'data', 'mbpp.parquet')).head(a.mbpp)
    for _, r in mb.iterrows():
        tests = list(r['test_list']); imps = '\n'.join(r['test_imports'])
        prompt = f"{r['prompt']}\nYour code should pass this test:\n{tests[0]}"
        def build(code, tests=tests, imps=imps):
            return f"{imps}\n{code}\n\n" + '\n'.join(tests) + '\n'
        yield 'mbpp', f"mbpp/{r['task_id']}", prompt, build

os.makedirs(os.path.join(HERE, 'sandbox'), exist_ok=True); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
stats = {}; t0 = time.time()
with open(os.path.join(HERE, 'results', f'{a.label}.jsonl'), 'w', encoding='utf-8') as out:
    for suite, tid, prompt, build in tasks():
        try:
            txt, fin, ntok, dt, tps = ask(prompt)
            ok = fin != 'length' and run_py(build(extract(txt)))
        except Exception as e:
            fin, ntok, dt, tps, ok = f'error:{e}', 0, 0, 0, False
        s = stats.setdefault(suite, {'n': 0, 'pass': 0, 'tok': 0, 'sec': 0.0, 'trunc': 0, 'tps': []})
        s['n'] += 1; s['pass'] += ok; s['tok'] += ntok; s['sec'] += dt; s['trunc'] += fin == 'length'
        if tps: s['tps'].append(tps)
        out.write(json.dumps({'id': tid, 'pass': ok, 'finish': fin, 'tokens': ntok, 'sec': round(dt, 2)}) + '\n'); out.flush()

md = os.path.join(HERE, '..', 'bench', 'code_eval.md')
if not os.path.exists(md):
    open(md, 'w').write("| label | effort | suite | pass@1 | avg tokens | avg sec/task | sec per correct | truncated | gen tok/s |\n|---|---|---|---|---|---|---|---|---|\n")
with open(md, 'a') as f:
    for suite, s in stats.items():
        p = s['pass'] / s['n']; tps = sum(s['tps']) / len(s['tps']) if s['tps'] else 0
        spc = s['sec'] / s['pass'] if s['pass'] else float('inf')
        f.write(f"| {a.label} | {a.effort} | {suite} | {p:.1%} | {s['tok']/s['n']:.0f} | {s['sec']/s['n']:.1f} | {spc:.1f} | {s['trunc']} | {tps:.1f} |\n")
print(f"done {a.label} {a.effort} in {(time.time()-t0)/60:.1f} min")
