"""(23 Nov, user: "one function deserves an explanation, measured not asserted") Why does the final model lose on one
function (BFCL simple)? Splits the WRONG bfcl_simple_python answers of the original and of a trained model by kind of
mistake. No ground truth is needed: each wrong call is compared with the RIGHT calls either model gave to the same
question (up to 16 answers), which show the expected function name and arguments.

Kinds: no call (nothing parseable), wrong function name, right name but a required argument missing, an extra
argument, a wrong argument value (same keys), or unclassifiable (no right answer to compare with on that question).

    python onefn_split.py --base runs/mvtf/full-base/records.jsonl --new runs/mvtf/full-od5n-r22/records.jsonl
"""
import argparse
import ast
import json
from collections import Counter, defaultdict


def parse(resp):
    """First function call in the visible answer -> (name, {arg: value repr}) or None."""
    text = (resp or "").strip()
    start = text.find("[")
    cands = [text[start:]] if start >= 0 else []
    cands.append(text)
    for c in cands:
        for end in range(len(c), 0, -1):
            if c[end - 1] not in "])":
                continue
            try:
                node = ast.parse(c[:end].strip(), mode="eval").body
            except Exception:
                continue
            if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
                node = node.elts[0]
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                args = {}
                for kw in node.keywords:
                    try:
                        args[kw.arg] = repr(ast.literal_eval(kw.value))
                    except Exception:
                        args[kw.arg] = ast.unparse(kw.value)
                return name, args
            break
    return None


def norm(v):
    return v.replace(" ", "").replace('"', "'").lower()


def classify(call, rights):
    if call is None:
        return "no call / unparseable"
    if not rights:
        return "unclassifiable (no right answer)"
    name, args = call
    names = {r[0] for r in rights}
    if name not in names:
        return "wrong function name"
    same = [r for r in rights if r[0] == name]
    keys_all = set.intersection(*[set(r[1]) for r in same])
    keys_any = set.union(*[set(r[1]) for r in same])
    if keys_all - set(args):
        return "missing argument"
    if set(args) - keys_any:
        return "extra argument"
    for r in same:
        if all(norm(args.get(k, "")) == norm(v) for k, v in r[1].items() if k in args):
            return "same call as a right answer (scorer detail)"
    return "wrong argument value"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--dataset", default="bfcl_simple_python")
    a = ap.parse_args()
    rows = {}
    for label, path in (("original", a.base), ("trained", a.new)):
        rows[label] = [r for r in map(json.loads, open(path)) if r["dataset"] == a.dataset]
    rights = defaultdict(list)
    for label in rows:
        for r in rows[label]:
            if r["correct"]:
                c = parse(r.get("response"))
                if c:
                    rights[r["id"]].append(c)
    print(f"== {a.dataset}: kinds of WRONG answers (expected call taken from the right answers to the same question)")
    res = {}
    for label in rows:
        n = len(rows[label])
        wrong = [r for r in rows[label] if not r["correct"]]
        cnt = Counter(classify(parse(r.get("response")), rights[r["id"]]) for r in wrong)
        trunc = sum(1 for r in wrong if r.get("truncated"))
        res[label] = (n, len(wrong), cnt, trunc)
        print(f"-- {label}: {len(wrong)} wrong of {n} answers ({100 * len(wrong) / n:.1f}%), truncated {trunc}")
        for k, v in cnt.most_common():
            print(f"   {k:<44} {v:5d}  ({100 * v / n:.2f} per 100 answers)")
    kinds = sorted(set(res["original"][2]) | set(res["trained"][2]))
    print("== change per 100 answers (trained - original)")
    for k in kinds:
        d = 100 * res["trained"][2][k] / res["trained"][0] - 100 * res["original"][2][k] / res["original"][0]
        print(f"   {k:<44} {res['original'][2][k]:5d} -> {res['trained'][2][k]:5d}   {d:+.2f}")
    # examples of the kind that grew most
    grew = max(kinds, key=lambda k: res["trained"][2][k] - res["original"][2][k])
    print(f"== examples of '{grew}' (trained model)")
    shown = 0
    for r in rows["trained"]:
        if not r["correct"] and classify(parse(r.get("response")), rights[r["id"]]) == grew:
            print(f"   {r['id']}: {r.get('response', '').strip()[:160]!r}  | a right one: {rights[r['id']][0] if rights[r['id']] else None}")
            shown += 1
            if shown >= 6:
                break


if __name__ == "__main__":
    main()
