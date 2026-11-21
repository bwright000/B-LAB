"""Follow-up tests against a running llama-server on :8091.
  python followup.py ab --label NAME          # fixed 6-prompt coding panel at temp 1.0 and 0; appends to ../bench/ledger.jsonl
  python followup.py needle --ctx 32768       # 3 codes hidden in a ~ctx-token document; then turn-2 cache test
"""
import argparse, json, os, random, statistics, time, requests

HERE = os.path.dirname(os.path.abspath(__file__)); URL = 'http://127.0.0.1:8091'
LEDGER = os.path.join(HERE, '..', 'bench', 'ledger.jsonl')
PANEL = [
  'Write a Python function merge_intervals(intervals) that merges overlapping intervals. Include type hints.',
  'Write a Python class LRUCache with get and put in O(1), using OrderedDict.',
  'Write a function that validates an IPv4 address string without using regex or ipaddress.',
  'Write a Python function to compute the Levenshtein distance between two strings using O(min(n,m)) memory.',
  'Write a Python generator that yields prime numbers indefinitely using an incremental sieve.',
  'Write a Python function that flattens an arbitrarily nested list without recursion.',
]

def chat(content, temp, max_tokens=800, effort='low'):
    body = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'temperature': temp,
            'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'seed': 1234, 'chat_template_kwargs': {'reasoning_effort': effort}}
    r = requests.post(f'{URL}/v1/chat/completions', json=body, timeout=900).json()
    return r['choices'][0]['message'].get('content') or '', r.get('timings', {}), r.get('usage', {})

def ab(label):
    for temp in (1.0, 0.0):
        tps, acc = [], []
        for p in PANEL:
            _, tm, _ = chat(p, temp)
            tps.append(tm.get('predicted_per_second', 0))
            if tm.get('draft_n'): acc.append(tm['draft_n_accepted'] / tm['draft_n'])
        row = {'t': time.strftime('%H:%M:%S'), 'label': label, 'temp': temp, 'median_tps': round(statistics.median(tps), 1),
               'samples': [round(x, 1) for x in tps], 'draft_acc': round(statistics.mean(acc), 2) if acc else None}
        open(LEDGER, 'a').write(json.dumps(row) + '\n'); print(row)

def needle(ctx):
    words = open(os.path.join(HERE, 'data', 'wiki.test.raw'), encoding='utf-8').read().split()
    rng = random.Random(ctx); codes = [f'{rng.choice("ABCDEFGHJK")}{rng.randint(1000, 9999)}' for _ in range(3)]
    n_tok = lambda s: len(requests.post(f'{URL}/tokenize', json={'content': s}).json()['tokens'])
    # size the document with the real tokenizer (skill: never assume words-per-token)
    n_words = int(ctx * 0.6); doc = ' '.join(words[:n_words])
    while n_tok(doc) > ctx - 1500: n_words = int(n_words * 0.93); doc = ' '.join(words[:n_words])
    parts = doc.split(' '); L = len(parts)
    for code, frac in zip(codes, (0.1, 0.5, 0.9)):
        parts.insert(int(L * frac), f'\n[The secret code number {codes.index(code)+1} is {code}.]\n')
    doc = ' '.join(parts); ntok = n_tok(doc)
    q1 = doc + '\n\nWhat are the three secret codes in the text above? Reply with only the three codes, comma separated.'
    t = time.time(); out, tm, _ = chat(q1, 0.0, max_tokens=2000); t1 = time.time() - t
    found = [c for c in codes if c in out]
    q2 = doc + '\n\nWhich secret code appears in the middle of the text? Reply with only that code.'
    t = time.time(); out2, tm2, _ = chat(q2, 0.0, max_tokens=2000); t2 = time.time() - t
    row = {'t': time.strftime('%H:%M:%S'), 'label': f'needle_{ctx}', 'doc_tokens': ntok, 'found': f'{len(found)}/3',
           'turn1_prompt_tps': round(tm.get('prompt_per_second', 0)), 'turn1_prefill_s': round(tm.get('prompt_ms', 0) / 1e3, 1),
           'turn1_total_s': round(t1, 1), 'turn2_cached_tokens': tm2.get('cache_n'), 'turn2_prefill_s': round(tm2.get('prompt_ms', 0) / 1e3, 2),
           'turn2_total_s': round(t2, 1), 'turn2_correct': codes[1] in out2, 'decode_tps_at_depth': round(tm.get('predicted_per_second', 0), 1)}
    open(LEDGER, 'a').write(json.dumps(row) + '\n'); print(row)

ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--label', default='x'); ap.add_argument('--ctx', type=int, default=32768)
a = ap.parse_args()
ab(a.label) if a.mode == 'ab' else needle(a.ctx)
