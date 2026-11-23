"""(22 Nov) The article's example answers for the final model: the same selection rules as the original examples.json
(scratchpad blog/prep.py: per task, 3 'trim' / 'loop' / 'cost' pairs, rng seed 7), with the trained model under the key
'od5c' so the article's player reads it unchanged; plus the trained model's answers to the article's quoted questions.
Prints EXAMPLESJSON <json> and QUOTEJSON <json> (one line each) for the Colab log.

    python article_final_examples.py --base runs/mvtf/full-base/records.jsonl --new runs/mvtf/full-od5n-r22/records.jsonl \
        --quotes svamp-chal-646:1,svamp-chal-883:4 --out runs/smarter-od5n22/article
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

TYPES = ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance']
WR = re.compile(r"[A-Za-z']+")


def think(r):
    t = r['text']; i = t.find('<think>'); j = t.find('</think>')
    return t[i + 7:j] if i >= 0 and j > i else (t[i + 7:] if i >= 0 else '')


def looped(words):
    if len(words) < 40: return False
    g = Counter(' '.join(words[k:k + 8]) for k in range(len(words) - 7))
    return max(g.values()) >= 4


def load(path):
    R = [json.loads(l) for l in open(path) if l.strip()]
    for r in R:
        w = WR.findall(think(r).lower())
        r['_loop'] = bool(r['truncated']) or looped(w)
        r['correct'] = bool(r['correct'])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--new', required=True)
    ap.add_argument('--quotes', default=''); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r['id'], r['sample_index'])
    idx = {'base': {key(r): r for r in load(a.base)}, 'od5c': {key(r): r for r in load(a.new)}}
    rng = random.Random(7)
    CAP = 9000
    def clip(s):
        return s if len(s) <= CAP else s[:CAP - 1400] + '\n\n[ … ' + f'{len(s) - CAP + 1400:,}' + ' characters cut … ]\n\n' + s[-1400:]
    ex = []
    for t in TYPES:
        pools = {'trim': [], 'loop': [], 'cost': []}
        for (i, s), b in idx['base'].items():
            if b['dataset'] != t: continue
            o = idx['od5c'].get((i, s))
            if not o: continue
            if b['correct'] and o['correct'] and not b['_loop'] and 250 <= b['output_tokens'] <= 2500 and o['output_tokens'] <= 0.45 * b['output_tokens']:
                pools['trim'].append((b, o))
            if b['_loop'] and o['correct'] and o['output_tokens'] < 1500:
                pools['loop'].append((b, o))
            if b['correct'] and not o['correct'] and o['output_tokens'] < 0.5 * b['output_tokens'] and b['output_tokens'] < 3000:
                pools['cost'].append((b, o))
        for kind, P in pools.items():
            rng.shuffle(P)
            seen = set()
            for b, o in P:
                if b['id'] in seen: continue
                seen.add(b['id'])
                ex.append({'type': t, 'kind': kind, 'id': b['id'], 's': b['sample_index'],
                           'base': {'text': clip(b['text']), 'tok': b['output_tokens'], 'ok': b['correct'], 'cut': bool(b['truncated']), 'loop': b['_loop']},
                           'od5c': {'text': clip(o['text']), 'tok': o['output_tokens'], 'ok': o['correct'], 'cut': bool(o['truncated']), 'loop': o['_loop']},
                           'qacc': {m: sum(idx[m][(b['id'], k)]['correct'] for k in range(8)) for m in ('base', 'od5c')}})
                if len(seen) >= 3: break
    quotes = []
    for q in [x for x in a.quotes.split(',') if x]:
        i, s = q.rsplit(':', 1); s = int(s)
        for m in ('base', 'od5c'):
            r = idx[m][(i, s)]
            quotes.append({'model': m, 'id': i, 's': s, 'tok': r['output_tokens'], 'ok': r['correct'],
                           'wait': think(r).lower().count('wait'), 'think': think(r)[:6000]})
        # the trained model's other answers to the same question (to pick a representative one)
        quotes.append({'model': 'od5c-all', 'id': i, 'answers': [{'s': k, 'tok': idx['od5c'][(i, k)]['output_tokens'],
                       'ok': idx['od5c'][(i, k)]['correct'], 'wait': think(idx['od5c'][(i, k)]).lower().count('wait')} for k in range(8)]})
    (out / 'examples_final.json').write_text(json.dumps(ex, separators=(',', ':')))
    print('EXAMPLESJSON ' + json.dumps(ex, separators=(',', ':')), flush=True)
    print('QUOTEJSON ' + json.dumps(quotes, separators=(',', ':')), flush=True)
    print('examples', len(ex), {k: sum(1 for e in ex if e['kind'] == k) for k in ('trim', 'loop', 'cost')}, flush=True)


if __name__ == '__main__':
    main()
