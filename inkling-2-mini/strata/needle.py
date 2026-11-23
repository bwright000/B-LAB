"""Needle + turn-2 cache test against an OpenAI-compatible server. needle.py <port> <approx_tokens> [max_tokens]"""
import importlib.util, json, sys, time, os, requests
spec = importlib.util.spec_from_file_location('bench', os.path.expanduser(r'~\.claude\skills\inference-optimisation\scripts\bench.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
port, L = int(sys.argv[1]), int(sys.argv[2]); mt = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
url = f'http://127.0.0.1:{port}/v1/chat/completions'
doc, codes = b.document(L, seed=7)
def ask(msgs):
    t = time.time()
    r = requests.post(url, json={"messages": msgs, "max_tokens": mt, "temperature": 0.0, "chat_template_kwargs": {"reasoning_effort": "low"}}, timeout=900).json()
    dt = time.time() - t; m = r['choices'][0]['message']; txt = m.get('content') or ''
    return txt, r.get('usage', {}), r.get('timings', {}), dt, r['choices'][0].get('finish_reason')
m1 = [{"role": "user", "content": doc}]
txt, u, tm, dt, fin = ask(m1)
print(f"turn1: prompt {u.get('prompt_tokens')} tok, cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')}, completion {u.get('completion_tokens')}, {dt:.1f}s, fin={fin}")
print(f"  prefill {tm.get('prompt_per_second')} tok/s over {tm.get('prompt_n')}, decode {tm.get('predicted_per_second')} tok/s, drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')}")
print(f"  retrieval {sum(c in txt for c in codes)}/{len(codes)}  answer={txt[:160]!r}")
m2 = m1 + [{"role": "assistant", "content": txt}, {"role": "user", "content": "Which of those codes comes first alphabetically? Answer with just the code."}]
txt2, u2, tm2, dt2, fin2 = ask(m2)
print(f"turn2: prompt {u2.get('prompt_tokens')} tok, cached {(u2.get('prompt_tokens_details') or {}).get('cached_tokens')} / cache_n {tm2.get('cache_n')}, {dt2:.2f}s total, prompt_ms {tm2.get('prompt_ms')}, answer={txt2[:80]!r} (expected {sorted(codes)[0]})")
