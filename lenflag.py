"""(23 Nov, user) Is thinking length a usable "send this to a lawyer" signal? For each model's full-test records (2,759 x 8),
per task and overall:
 (a) AUROC of thinking length for predicting a WRONG answer (raw length; overall uses length / task median);
 (b) the same WITHIN question: length / that question's median over its 8 answers, only questions with both right and
     wrong answers ("a long answer to the same question is more likely wrong", not just "hard questions are long");
 (c) escalation: flag the longest X% of answers per task (X = 5, 10, 20): share of all wrong answers caught, error rate
     among flagged, accuracy of the unflagged rest vs overall;
 (d) question level: coefficient of variation of the 8 lengths vs "answered inconsistently" (0 < right < 8), AUROC.
(a), (b) and (d) carry bootstrap 95% intervals over questions.

    python lenflag.py --model base=runs/mvtf/full-base/records.jsonl --model od5n22=runs/mvtf/full-od5n-r22/records.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = [("svamp", "Math"), ("mbpp", "Code"), ("ifeval", "Instructions"), ("bfcl_simple_python", "One function"),
         ("bfcl_multiple", "Several functions"), ("bfcl_irrelevance", "No function fits")]


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.sort(neg)
    lo = np.searchsorted(s, pos, side="left")
    hi = np.searchsorted(s, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def load(path):
    by = defaultdict(list)  # (dataset, id) -> list of (length, wrong, truncated)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        by[(r["dataset"], r["id"])].append((float(r.get("reasoning_tokens") or 0), 0 if r["correct"] else 1, bool(r.get("truncated"))))
    return by


def boot(qs, fn, rng, n=400):
    """95% bootstrap interval over questions; fn(list of question entries) -> statistic."""
    vals = []
    idx = np.arange(len(qs))
    for _ in range(n):
        v = fn([qs[i] for i in rng.choice(idx, len(idx))])
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def raw_auc(qs):
    L = np.array([a[0] for q in qs for a in q]); W = np.array([a[1] for q in qs for a in q])
    return auroc(L[W == 1], L[W == 0])


def within_auc(qs):
    rel, wr = [], []
    for q in qs:
        L = np.array([a[0] for a in q]); W = np.array([a[1] for a in q])
        if 0 < W.sum() < len(W):
            m = np.median(L) or 1.0
            rel += list(L / m); wr += list(W)
    rel, wr = np.array(rel), np.array(wr)
    return auroc(rel[wr == 1], rel[wr == 0]) if len(rel) else float("nan")


def cv_auc(qs):
    cv, inc = [], []
    for q in qs:
        L = np.array([a[0] for a in q]); W = np.array([a[1] for a in q])
        cv.append(L.std() / (L.mean() or 1.0)); inc.append(1 if 0 < W.sum() < len(W) else 0)
    cv, inc = np.array(cv), np.array(inc)
    return auroc(cv[inc == 1], cv[inc == 0])


def escalate(qs, frac):
    L = np.array([a[0] for q in qs for a in q]); W = np.array([a[1] for q in qs for a in q])
    k = max(1, int(round(frac * len(L))))
    order = np.argsort(-L, kind="stable")
    flag = np.zeros(len(L), bool); flag[order[:k]] = True
    caught = W[flag].sum() / max(1, W.sum())
    return {"caught": float(caught), "err_flagged": float(W[flag].mean()), "acc_rest": float(1 - W[~flag].mean()),
            "acc_all": float(1 - W.mean())}


def analyse(by, rng):
    out = {}
    pooled = []  # overall: lengths divided by the task median, so tasks of different length scales can be pooled
    for ds, name in TASKS:
        qs = [v for (d, _), v in by.items() if d == ds]
        if not qs:
            continue
        med = float(np.median([a[0] for q in qs for a in q])) or 1.0
        pooled += [[(a[0] / med, a[1], a[2]) for a in q] for q in qs]
        n_ans = sum(len(q) for q in qs)
        out[name] = {
            "answers": n_ans, "acc": 1 - sum(a[1] for q in qs for a in q) / n_ans,
            "trunc": sum(a[2] for q in qs for a in q) / n_ans,
            "median_len": med,
            "a": raw_auc(qs), "a_ci": boot(qs, raw_auc, rng),
            "b": within_auc(qs), "b_ci": boot(qs, within_auc, rng),
            "b_questions": sum(1 for q in qs if 0 < sum(a[1] for a in q) < len(q)),
            "c": {str(x): escalate(qs, x / 100) for x in (5, 10, 20)},
            "d": cv_auc(qs), "d_ci": boot(qs, cv_auc, rng),
        }
    qs = pooled
    n_ans = sum(len(q) for q in qs)
    out["All"] = {"answers": n_ans, "acc": 1 - sum(a[1] for q in qs for a in q) / n_ans,
                  "trunc": sum(a[2] for q in qs for a in q) / n_ans, "median_len": 1.0,
                  "a": raw_auc(qs), "a_ci": boot(qs, raw_auc, rng, 200),
                  "b": within_auc(qs), "b_ci": boot(qs, within_auc, rng, 200),
                  "b_questions": sum(1 for q in qs if 0 < sum(a[1] for a in q) < len(q)),
                  "c": {str(x): escalate(qs, x / 100) for x in (5, 10, 20)},
                  "d": cv_auc(qs), "d_ci": boot(qs, cv_auc, rng, 200)}
    return out


def fmt(res):
    lines = []
    names = list(res)
    first = res[names[0]]
    lines.append("== (a) raw length -> wrong answer, AUROC [95%]   |   (b) within question (relative length), AUROC [95%] (n questions)")
    for t in first:
        row = f"   {t:<18}"
        for n in names:
            r = res[n].get(t)
            if r:
                row += (f" | {n}: a {r['a']:.3f} [{r['a_ci'][0]:.3f}, {r['a_ci'][1]:.3f}]  b {r['b']:.3f} "
                        f"[{r['b_ci'][0]:.3f}, {r['b_ci'][1]:.3f}] ({r['b_questions']})")
        lines.append(row)
    lines.append("== (c) flag the longest X% of answers per task: wrong answers caught / error rate among flagged / accuracy of the rest (all)")
    for t in first:
        for x in ("5", "10", "20"):
            row = f"   {t:<18} {x:>2}%"
            for n in names:
                r = res[n].get(t)
                if r:
                    c = r["c"][x]
                    row += f" | {n}: caught {100 * c['caught']:5.1f}%  err {100 * c['err_flagged']:5.1f}%  rest {100 * c['acc_rest']:5.1f} ({100 * c['acc_all']:.1f})"
            lines.append(row)
    lines.append("== (d) question level: spread of the 8 lengths (CV) -> answered inconsistently, AUROC [95%]")
    for t in first:
        row = f"   {t:<18}"
        for n in names:
            r = res[n].get(t)
            if r:
                row += f" | {n}: {r['d']:.3f} [{r['d_ci'][0]:.3f}, {r['d_ci'][1]:.3f}]  acc {100 * r['acc']:.1f}  trunc {100 * r['trunc']:.1f}%  median {r['median_len']:.0f}t"
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, help="NAME=records.jsonl (repeatable)")
    ap.add_argument("--out", default="runs/lenflag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {}
    for m in args.model:
        name, path = m.split("=", 1)
        res[name] = analyse(load(path), rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lenflag.json").write_text(json.dumps(res, indent=1))
    text = fmt(res)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
