"""(22 Nov, user: make the article 100% on the final model) Recompute, for the original vs ONE trained model on the FULL TEST
(2,759 x 8 records with text), every number the ThinkingCap article still showed for od5c16, with the article's own
definitions (scratchpad blog/prep.py + dens.py, article_metrics.py, the 18 Nov retention/rescue and moved-beyond-noise
snippets). Thinking-off parts (needing the original's thinking-off records) are NOT computed here: the caller merges them.

Writes <out>/stats_final.json (same schema as the article's stats.json, trained model under the key 'od5c' so the
article's charts read it unchanged; no 'nothink' entries) and <out>/extra_final.json, and prints both as one line each
(prefixes STATSJSON / EXTRAJSON) so they can be read from the Colab log.

    python article_final_stats.py --base runs/mvtf/full-base/records.jsonl --new runs/mvtf/full-od5n-r22/records.jsonl --out runs/smarter-od5n22/article
"""
import argparse, json, math, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance']
WR = re.compile(r"[A-Za-z']+")
FILL = ['wait', 'hmm', 'alternatively', 'actually', 'double-check', 'maybe', 'perhaps', 'verify', 'again', 'okay']
WRL = re.compile(r"[a-z']+")
MARK = ['wait', 'hmm', 'however', 'perhaps', 'alternatively', 'maybe', 'actually']
PHR = ['let me verify', 'let me check', 'double-check', 'double check', 'let me make sure']


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
        th = think(r)
        w = WR.findall(th.lower())
        r['_w'] = w
        r['_loop'] = bool(r['truncated']) or looped(w)
        r['_resp'] = r['output_tokens'] - (r.get('reasoning_tokens') or 0)
        paras = [p for p in re.split(r'\n\s*\n', th) if p.strip()]
        r['_paras'] = len(paras)
        r['_vocab'] = len(set(w))
        g4 = [' '.join(w[k:k + 4]) for k in range(len(w) - 3)]
        c4 = Counter(g4)
        r['_rep4'] = (sum(v for v in c4.values() if v > 1) / len(g4)) if g4 else 0.0
        pv = []
        for p in paras:
            pw = WR.findall(p.lower())
            if len(pw) >= 10: pv.append(len(set(pw)) / len(pw))
        r['_pv'] = pv
        r['_th'] = th
        r['correct'] = bool(r['correct'])
    return R


def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M = {'base': load(a.base), 'od5c': load(a.new)}  # 'od5c' = the trained model (article chart key)
    key = lambda r: (r['id'], r['sample_index'])
    idx = {m: {key(r): r for r in R} for m, R in M.items()}
    MODELS = ('base', 'od5c')
    stats = {'types': TYPES, 'per': {}, 'hist': {}, 'dec': {}, 'fill': {}, 'vol': {}, 'mass': {}}
    for t in TYPES + ['all']:
        per = {}
        for m in MODELS:
            S = [r for r in M[m] if t == 'all' or r['dataset'] == t]
            tok = [r['output_tokens'] for r in S]
            per[m] = {'n': len(S), 'acc': 100 * sum(r['correct'] for r in S) / len(S),
                      'mean': st.mean(tok), 'median': q(tok, .5), 'p90': q(tok, .9),
                      'think': st.mean((r.get('reasoning_tokens') or 0) for r in S), 'resp': st.mean(r['_resp'] for r in S),
                      'right': st.mean([r['output_tokens'] for r in S if r['correct']] or [0]),
                      'wrong': st.mean([r['output_tokens'] for r in S if not r['correct']] or [0]),
                      'runaway': 100 * sum(bool(r['truncated']) for r in S) / len(S),
                      'loop': 100 * sum(r['_loop'] for r in S) / len(S),
                      'paras': st.mean(r['_paras'] for r in S), 'words': st.mean(len(r['_w']) for r in S)}
            tot = sum(tok)
            per[m]['share_loop'] = 100 * sum(r['output_tokens'] for r in S if r['_loop']) / tot
            per[m]['share_wrong'] = 100 * sum(r['output_tokens'] for r in S if not r['correct']) / tot
        stats['per'][t] = per
        edges = [16 * 2 ** (k / 2) for k in range(21)]
        h = {}
        for m in MODELS:
            S = [r['output_tokens'] for r in M[m] if t == 'all' or r['dataset'] == t]
            c = [0] * (len(edges) - 1)
            for x in S:
                k = min(len(c) - 1, max(0, int(2 * math.log2(max(x, 16) / 16))))
                c[k] += 1
            h[m] = [round(100 * v / len(S), 2) for v in c]
        stats['hist'][t] = {'edges': [round(e) for e in edges], **h}
        byq = defaultdict(list)
        for r in M['base']:
            if t == 'all' or r['dataset'] == t: byq[r['id']].append(r)
        qs = sorted(byq, key=lambda i: st.median(x['output_tokens'] for x in byq[i]))
        dec = []
        for d in range(10):
            part = qs[d * len(qs) // 10:(d + 1) * len(qs) // 10]
            row = {'q': len(part)}
            for m in MODELS:
                rr = [idx[m][(i, s)] for i in part for s in range(8) if (i, s) in idx[m]]
                row[m + '_acc'] = 100 * sum(x['correct'] for x in rr) / max(1, len(rr))
                row[m + '_tok'] = st.mean(x['output_tokens'] for x in rr) if rr else 0
            dec.append(row)
        stats['dec'][t] = dec
        f = {}
        for m in MODELS:
            S = [r for r in M[m] if t == 'all' or r['dataset'] == t]
            W = sum(len(r['_w']) for r in S) or 1
            c = Counter(x for r in S for x in r['_w'] if x in FILL)
            f[m] = {w: round(1000 * c[w] / W, 2) for w in FILL}
            f[m]['_total_words_per_answer'] = round(W / len(S), 1)
        stats['fill'][t] = f
        v = {}
        for m in MODELS:
            g = defaultdict(list)
            for r in M[m]:
                if t == 'all' or r['dataset'] == t: g[r['id']].append(r['output_tokens'])
            ratios = [max(x) / max(1, min(x)) for x in g.values() if len(x) == 8]
            v[m] = {'median_ratio': q(ratios, .5), 'p90_ratio': q(ratios, .9),
                    'right_and_wrong': 100 * sum(1 for i, x in g.items() if 0 < sum(idx[m][(i, s)]['correct'] for s in range(8)) < 8) / len(g)}
        stats['vol'][t] = v
    B = M['base']; tot = sum(r['output_tokens'] for r in B)
    stats['mass'] = {t: {
        'loop': 100 * sum(r['output_tokens'] for r in B if r['_loop'] and r['dataset'] == t) / tot,
        'wrong': 100 * sum(r['output_tokens'] for r in B if not r['_loop'] and not r['correct'] and r['dataset'] == t) / tot,
        'right': 100 * sum(r['output_tokens'] for r in B if not r['_loop'] and r['correct'] and r['dataset'] == t) / tot,
    } for t in TYPES}
    stats['write'] = {}
    for t in TYPES + ['all']:
        stats['write'][t] = {}
        for m in MODELS:
            S = [r for r in M[m] if t == 'all' or r['dataset'] == t]
            stats['write'][t][m] = {'words': st.mean(len(r['_w']) for r in S), 'vocab': st.mean(r['_vocab'] for r in S),
                                    'ttr': st.mean(r['_vocab'] / len(r['_w']) for r in S if len(r['_w']) >= 20) if any(len(r['_w']) >= 20 for r in S) else 0,
                                    'rep4': 100 * st.mean(r['_rep4'] for r in S), 'paras': st.mean(r['_paras'] for r in S),
                                    'wpp': st.mean(len(r['_w']) / r['_paras'] for r in S if r['_paras'])}
    G = lambda n: 0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3 if n <= 7 else 4
    qids = sorted({r['id'] for r in B}); qtype = {r['id']: r['dataset'] for r in B}
    stats['trans'] = {}
    for t in TYPES + ['all']:
        mat = [[0] * 5 for _ in range(5)]
        for i in qids:
            if t != 'all' and qtype[i] != t: continue
            x = sum(idx['base'][(i, s)]['correct'] for s in range(8) if (i, s) in idx['base'])
            y = sum(idx['od5c'][(i, s)]['correct'] for s in range(8) if (i, s) in idx['od5c'])
            mat[G(x)][G(y)] += 1
        stats['trans'][t] = mat
    stats['quint'] = {}
    for t in TYPES:
        byq = defaultdict(list)
        for r in B:
            if r['dataset'] == t: byq[r['id']].append(r)
        qs = sorted(byq, key=lambda i: st.median(x['output_tokens'] for x in byq[i]))
        row = []
        for d in range(5):
            part = qs[d * len(qs) // 5:(d + 1) * len(qs) // 5]
            bb = [idx['base'][(i, s)] for i in part for s in range(8)]; oo = [idx['od5c'][(i, s)] for i in part for s in range(8)]
            row.append({'saved': 100 * (1 - sum(x['output_tokens'] for x in oo) / sum(x['output_tokens'] for x in bb)),
                        'dacc': 100 * (sum(x['correct'] for x in oo) - sum(x['correct'] for x in bb)) / len(bb),
                        'tok': st.mean(x['output_tokens'] for x in bb)})
        stats['quint'][t] = row
    stats['cat'] = {}
    for t in TYPES + ['all']:
        stats['cat'][t] = {}
        for m in MODELS:
            S = [r for r in M[m] if t == 'all' or r['dataset'] == t]; tt = sum(r['output_tokens'] for r in S)
            stats['cat'][t][m] = {'loop': 100 * sum(r['output_tokens'] for r in S if r['_loop']) / tt,
                                  'wrong': 100 * sum(r['output_tokens'] for r in S if not r['_loop'] and not r['correct']) / tt,
                                  'right': 100 * sum(r['output_tokens'] for r in S if not r['_loop'] and r['correct']) / tt}
    stats['density'] = {}
    for t in TYPES + ['all']:
        stats['density'][t] = {}
        for m in MODELS:
            S = [r for r in M[m] if t == 'all' or r['dataset'] == t]
            tok = sum(r['output_tokens'] for r in S); th = sum((r.get('reasoning_tokens') or 0) for r in S)
            acc = 100 * sum(r['correct'] for r in S) / len(S)
            pv = [x for r in S for x in r['_pv']]
            stats['density'][t][m] = {'acc': acc, 'tok': tok / len(S), 'think': th / len(S), 'final': (tok - th) / len(S),
                                      'density': acc / (tok / len(S)) * 100, 'para_ttr': st.mean(pv) if pv else 0}

    # ---- extra: the article's other measures ----
    ex = {'flips': {}, 'gpass': {}, 'og': {}, 'kappa': {}, 'retention': {}, 'moved': {}, 'per100': {}}
    Q = sorted(qids)
    for t in TYPES + ['all']:
        qs = [i for i in Q if t == 'all' or qtype[i] == t]; N = len(qs) * 8
        def flips(A, Bm, shift=0):
            nf = pf = 0
            for i in qs:
                for s in range(8):
                    x = idx[A][(i, s)]['correct']; y = idx[Bm][(i, (s + shift) % 8)]['correct']
                    nf += x and not y; pf += (not x) and y
            return 100 * nf / N, 100 * pf / N
        ex['flips'][t] = {'trained': flips('base', 'od5c'), 'noise': flips('base', 'base', 4)}
        ex['gpass'][t] = {m: {k: 100 * st.mean(math.comb(sum(idx[m][(i, s)]['correct'] for s in range(8)), k) / math.comb(8, k) for i in qs)
                              for k in (1, 2, 4, 8)} for m in MODELS}
        og = {'base': [], 'od5c': []}
        for i in qs:
            cor = [idx[m][(i, s)]['output_tokens'] for m in MODELS for s in range(8) if idx[m][(i, s)]['correct']]
            if not cor: continue
            mn = min(cor)
            for m in MODELS:
                og[m].append(st.mean(idx[m][(i, s)]['output_tokens'] for s in range(8)) - mn)
        ex['og'][t] = {**{m: st.mean(v) for m, v in og.items()}, 'n': len(og['base'])}
        kp = {}
        for m in MODELS:
            W = c = 0
            for i in qs:
                for s in range(8):
                    th = idx[m][(i, s)]['_th'].lower(); w = WRL.findall(th); W += len(w)
                    c += sum(1 for x in w if x in MARK) + sum(th.count(p) for p in PHR)
            kp[m] = 100 * c / max(1, W)
        ex['kappa'][t] = kp
        b = {i: sum(idx['base'][(i, s)]['correct'] for s in range(8)) for i in qs}
        o = {i: sum(idx['od5c'][(i, s)]['correct'] for s in range(8)) for i in qs}
        ex['retention'][t] = {'keep': 100 * sum(min(b[i], o[i]) for i in qs) / max(1, sum(b.values())),
                              'rescue': 100 * sum(max(0, o[i] - b[i]) for i in qs) / max(1, sum(8 - b[i] for i in qs))}
        nz = [0, 0, 0]; tr = [0, 0, 0]
        for i in qs:
            x = sum(idx['base'][(i, s)]['correct'] for s in range(4))
            yn = sum(idx['base'][(i, s)]['correct'] for s in range(4, 8)); yt = sum(idx['od5c'][(i, s)]['correct'] for s in range(4, 8))
            nz[0 if yn < x else 1 if yn == x else 2] += 1; tr[0 if yt < x else 1 if yt == x else 2] += 1
        ex['moved'][t] = {'noise_down_same_up': nz, 'trained_down_same_up': tr, 'excess_down': tr[0] - nz[0], 'excess_up': tr[2] - nz[2]}
        rr = rw = wr = ww = 0
        for i in qs:
            for s in range(8):
                x = idx['base'][(i, s)]['correct']; y = idx['od5c'][(i, s)]['correct']
                rr += x and y; rw += x and not y; wr += (not x) and y; ww += (not x) and (not y)
        ex['per100'][t] = {k: 100 * v / N for k, v in (('right_right', rr), ('right_wrong', rw), ('wrong_right', wr), ('wrong_wrong', ww))}
        both = sum(min(b[i], o[i]) for i in qs)  # the article's 'out of every 100 answers': per question, k0 vs k1 of 8
        ex.setdefault('per100q', {})[t] = {'right_right': 100 * both / N, 'right_wrong': 100 * (sum(b.values()) - both) / N,
                                           'wrong_right': 100 * (sum(o.values()) - both) / N,
                                           'wrong_wrong': 100 * (N - sum(b.values()) - sum(o.values()) + both) / N}
    NUM = re.compile(r'-?\d+(?:\.\d+)?')
    gold = {}
    for r in B:
        if r['dataset'] == 'svamp' and r['correct']:
            mm = re.findall(r'[Ff]inal answer:\s*\$?(-?[\d,]+(?:\.\d+)?)', r.get('response') or '')
            if mm: gold.setdefault(r['id'], mm[-1].replace(',', ''))
    ex['xio'] = {}
    for m in MODELS:
        xs = []
        for i, gv in gold.items():
            for s in range(8):
                r = idx[m][(i, s)]
                if not r['correct']: continue
                th = r['_th']
                pos = next((mm.start() for mm in NUM.finditer(th) if mm.group().rstrip('.') in (gv, gv + '.0') or (gv.endswith('.0') and mm.group() == gv[:-2])), None)
                if pos is None or not th: continue
                xs.append(pos / len(th))
        ex['xio'][m] = {'xi_O': 100 * st.mean(xs) if xs else None, 'n': len(xs)}

    (out / 'stats_final.json').write_text(json.dumps(stats, separators=(',', ':')))
    (out / 'extra_final.json').write_text(json.dumps(ex, separators=(',', ':')))
    print('STATSJSON ' + json.dumps(stats, separators=(',', ':')), flush=True)
    print('EXTRAJSON ' + json.dumps(ex, separators=(',', ':')), flush=True)
    a_ = stats['per']['all']
    print('per all', {m: {k: round(v, 1) for k, v in a_[m].items()} for m in a_}, flush=True)


if __name__ == '__main__':
    main()
