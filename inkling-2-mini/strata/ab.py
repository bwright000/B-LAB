"""Token-level engine A/B: identical prompt token ids into llama.cpp (HTTP /completion) and strata-inkling (stdin line protocol).

  ab.py prep   --url http://127.0.0.1:8080            -> prompts.json (chat-templated token ids, via the llama.cpp server)
  ab.py llama  --url http://127.0.0.1:8080 --label X  -> one JSON line per prompt in results/X.jsonl
  ab.py strata --label X -- <engine args...>           -> same, driving strata-inkling.exe --serve

Decode tok/s excludes prefill in both: llama.cpp `timings.predicted_per_second`; Strata's DONE line (dec_ms).
Wall-clock tok/s (first token -> last token, measured here) is recorded for both as a cross-check.
"""
import argparse, json, os, statistics, subprocess, sys, threading, time, queue
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, 'engine', 'strata-inkling.exe')
SYS = ("You are a precise Python programmer. Reply with one complete ```python code block containing the full "
       "solution (imports + function). No tests, no explanation.")
CHAT = [  # bench.py's fixed panel
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]
CODE = [  # small coding tasks in eval_code.py's format (system prompt + function to complete)
    'Complete this Python function:\n\n```python\nfrom typing import List\n\n\ndef merge_intervals(intervals: List[List[int]]) -> List[List[int]]:\n    """ Merge all overlapping intervals and return them sorted by start.\n    >>> merge_intervals([[1,3],[2,6],[8,10]])\n    [[1, 6], [8, 10]]\n    """\n```',
    'Complete this Python function:\n\n```python\ndef roman_to_int(s: str) -> int:\n    """ Convert a Roman numeral (I, V, X, L, C, D, M, with subtractive pairs) to an integer.\n    >>> roman_to_int("MCMXCIV")\n    1994\n    """\n```',
    'Write a function to find the longest common subsequence of two strings and return it as a string.\nYour code should pass this test:\nassert lcs("AGGTAB", "GXTXAYB") == "GTAB"',
    'Complete this Python function:\n\n```python\nfrom typing import Dict, List\n\n\ndef topo_sort(graph: Dict[str, List[str]]) -> List[str]:\n    """ Return a topological order of the DAG given as adjacency lists; raise ValueError on a cycle.\n    >>> topo_sort({"a": ["b"], "b": ["c"], "c": []})\n    [\'a\', \'b\', \'c\']\n    """\n```',
]


def prep(a):
    out = []
    for kind, msgs in ([('chat', [{"role": "user", "content": p}]) for p in CHAT] +
                       [('code', [{"role": "system", "content": SYS}, {"role": "user", "content": p}]) for p in CODE]):
        t = requests.post(a.url + '/apply-template', json={"messages": msgs, "chat_template_kwargs": {"reasoning_effort": "low"}}, timeout=30).json()['prompt']
        ids = requests.post(a.url + '/tokenize', json={"content": t, "add_special": True, "parse_special": True}, timeout=30).json()['tokens']
        out.append({"kind": kind, "text": t, "ids": ids})
        print(kind, len(ids), repr(t[:90]))
    json.dump(out, open(os.path.join(HERE, 'prompts.json'), 'w'))


def summarise(rows, label):
    for kind in ('chat', 'code'):
        v = [r['tps'] for r in rows if r['kind'] == kind and r['tps']]
        w = [r['tps_wall'] for r in rows if r['kind'] == kind and r['tps_wall']]
        acc = sum(r.get('acc', 0) for r in rows if r['kind'] == kind); off = sum(r.get('off', 0) for r in rows if r['kind'] == kind)
        if v:
            print(f"== {label} {kind}: median {statistics.median(v):.1f} tok/s (min {min(v):.1f} max {max(v):.1f}, wall median {statistics.median(w):.1f})"
                  + (f", drafts {acc}/{off} = {acc/off:.0%}" if off else ""))


def save(rows, label):
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', label + '.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def llama(a):
    P = json.load(open(os.path.join(HERE, 'prompts.json'))); rows = []
    for rep in range(a.reps):
        for i, p in enumerate(P):
            body = {"prompt": p['ids'], "n_predict": a.gen, "temperature": a.temp, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                    "cache_prompt": False, "return_tokens": True, "stream": True, "seed": 1234 + rep}
            t0 = time.monotonic(); tf = None; toks = []; tm = {}
            with requests.post(a.url + '/completion', json=body, stream=True, timeout=900) as r:
                for raw in r.iter_lines():
                    if not raw.startswith(b'data:'):
                        continue
                    ev = json.loads(raw[5:])
                    if ev.get('tokens'):
                        if tf is None:
                            tf = time.monotonic()
                        toks += ev['tokens']
                    if ev.get('timings'):
                        tm = ev['timings']
            te = time.monotonic(); n = tm.get('predicted_n', len(toks))
            row = {"i": i, "rep": rep, "kind": p['kind'], "n": n, "tps": tm.get('predicted_per_second'),
                   "tps_wall": (len(toks) - 1) / (te - tf) if tf and len(toks) > 1 and te > tf else None,
                   "pp_tps": tm.get('prompt_per_second'), "off": tm.get('draft_n', 0), "acc": tm.get('draft_n_accepted', 0), "ids": toks}
            rows.append(row)
            print(f"{a.label} #{i} {p['kind']} n={n} {row['tps']:.1f} tok/s wall {row['tps_wall'] or 0:.1f} drafts {row['acc']}/{row['off']}", flush=True)
    save(rows, a.label); summarise(rows, a.label)


def shared_mb(pid):
    """System RAM borrowed by the GPU for this process (WDDM 'Shared Usage'), MiB. ~450 MiB is the CUDA baseline; much more = VRAM spill."""
    cmd = (r"$s=(Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | "
           "Where-Object { $_.InstanceName -match 'pid_%d_' }; [int64](($s | Measure-Object CookedValue -Sum).Sum / 1MB)" % pid)
    try:
        return int(subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, timeout=30).stdout.strip() or 0)
    except Exception:
        return -1


def vram_mb():
    try:
        return int(subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout.strip())
    except Exception:
        return -1


def strata(a, engine_args):
    P = json.load(open(os.path.join(HERE, 'prompts.json'))); rows = []
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '1'   # GPU 1 only (GPU 0 is off-limits); engine sees it as --gpu 0
    for kv in a.env or []:
        k, v = kv.split('=', 1); env[k] = v
    errf = open(os.path.join(HERE, 'results', a.label + '.err'), 'w') if os.path.isdir(os.path.join(HERE, 'results')) else subprocess.DEVNULL
    pr = subprocess.Popen([ENGINE, '--serve'] + engine_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
                          text=True, bufsize=1, env=env, cwd=os.path.join(HERE, 'engine'))
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.rstrip('\n')) for l in pr.stdout] + [q.put(None)], daemon=True).start()

    def wait_for(pred, timeout):
        end = time.monotonic() + timeout
        while True:
            try:
                l = q.get(timeout=max(0.1, end - time.monotonic()))
            except queue.Empty:
                raise TimeoutError('engine timeout')
            if l is None:
                raise RuntimeError('engine exited')
            if l.startswith('ERR'):
                raise RuntimeError(l)
            r = pred(l)
            if r is not None:
                return r

    try:
        t0 = time.monotonic()
        wait_for(lambda l: (print(l, flush=True) or (True if l.startswith('READY') else None)) if l.startswith(('INFO', 'READY')) else None, 600)
        print(f"load {time.monotonic() - t0:.1f}s  vram {vram_mb()} MiB  shared {shared_mb(pr.pid)} MiB", flush=True)
        for rep in range(a.reps):
            for i, p in enumerate(P):
                toks = []; tf = [None]; done = {}
                pr.stdin.write(f"GEN {a.gen} temperature={a.temp} top_p=1.0 top_k=128 seed={1234 + rep} " + ','.join(map(str, p['ids'])) + '\n'); pr.stdin.flush()

                def on(l):
                    if l.startswith('T '):
                        if tf[0] is None:
                            tf[0] = time.monotonic()
                        toks.append(int(l[2:])); return None
                    if l.startswith('DONE'):
                        w = l.split(); done.update(gen=int(w[1]), plen=int(w[2]), pms=float(w[3]), dms=float(w[4]), fin=w[5],
                                                   acc=int(w[6]) if len(w) > 6 else 0, off=int(w[7]) if len(w) > 7 else 0)
                        return True
                    return None
                wait_for(on, 900); te = time.monotonic()
                row = {"i": i, "rep": rep, "kind": p['kind'], "n": done['gen'], "fin": done['fin'],
                       "tps": (done['gen'] - 1) / (done['dms'] / 1000) if done['dms'] > 0 and done['gen'] > 1 else None,
                       "tps_wall": (len(toks) - 1) / (te - tf[0]) if tf[0] and len(toks) > 1 and te > tf[0] else None,
                       "pp_tps": done['plen'] / (done['pms'] / 1000) if done['pms'] > 0 else None, "off": done['off'], "acc": done['acc'], "ids": toks}
                rows.append(row)
                print(f"{a.label} #{i} {p['kind']} n={row['n']} {row['tps'] or 0:.1f} tok/s wall {row['tps_wall'] or 0:.1f} fin={done['fin']} drafts {row['acc']}/{row['off']} pp {row['pp_tps'] or 0:.0f}", flush=True)
        pr.stdin.write('STATS\n'); pr.stdin.flush()
        try:
            wait_for(lambda l: (print(l) or True) if l.startswith('STATS') else None, 10)
        except Exception as e:
            print('stats:', e)
    finally:
        try:
            pr.stdin.write('QUIT\n'); pr.stdin.flush(); pr.wait(60)
        except Exception:
            pr.kill()
    save(rows, a.label); summarise(rows, a.label)


if __name__ == '__main__':
    argv = sys.argv[1:]; extra = []
    if '--' in argv:
        k = argv.index('--'); argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--label', default='run'); ap.add_argument('--gen', type=int, default=400); ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--reps', type=int, default=1); ap.add_argument('--env', action='append')
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    {'prep': prep, 'llama': llama}.get(a.mode, lambda a: strata(a, extra))(a)
