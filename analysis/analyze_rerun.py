"""Score the contextual-benchmark rerun (analysis/rerun/out/*.jsonl) against the frozen manifest.

Unlike the original run, every response is stored in full, so all items are re-extractable.
Intention-to-evaluate: items with no successful response, or with an empty answer, count as
incorrect. Responses cut off by the output limit are reported separately (and count as incorrect
unless they still contain a scorable final answer).

Output: analysis/out/ctx_rerun_results.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ctx import extract_mcq, extract_numeric, wilson  # noqa: E402
from scoring import numeric_equal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "rerun")
CONFIGS = ["gpt", "claude", "gemini_minimal", "gemini_high"]
TRUNC = {"length", "max_tokens", "MAX_TOKENS"}
RNG = np.random.default_rng(20261012)


def load():
    man = [json.loads(l) for l in open(os.path.join(RUN, "manifest.jsonl"))]
    res = {}
    for c in CONFIGS:
        p = os.path.join(RUN, "out", f"{c}.jsonl")
        best = {}
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                if r["status"] == "ok" or r["key"] not in best:
                    best[r["key"]] = r
        res[c] = best
    return man, res


_OPS = {"+": "+", "plus": "+", "-": "-", "minus": "-", "*": "*", "times": "*", "multiplied by": "*",
        "/": "/", "divided by": "/"}


def exact_roundings(item):
    """For NumericBench arithmetic items, the exact result rounded half-up and half-even to two
    decimals. Some reference answers were produced with binary floating point and are off by
    0.01, so these items are scored against the exact roundings instead of the reference label."""
    import ast
    import re
    from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
    if item.get("subset") not in ("arithmetic", "context_arithmetic", "different_digit"):
        return set()
    m = re.search(r"Data:\n(\{.*?\})", item["user"], re.S)
    q = re.search(r"result of (\w+) (.+?) (\w+)\? Please round", item["user"])
    if not q:
        q = re.search(r"result of (\w+)\s*([+\-*/])\s*(\w+)\? Please round", item["user"])
    if not (m and q) or q.group(2).strip() not in _OPS:
        return set()
    vals = {k: Decimal(str(v)) for k, v in ast.literal_eval(m.group(1)).items()}
    a, b, op = vals[q.group(1)], vals[q.group(3)], _OPS[q.group(2).strip()]
    x = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
    return {str(x.quantize(Decimal("0.01"), ROUND_HALF_UP)), str(x.quantize(Decimal("0.01"), ROUND_HALF_EVEN))}


def score_item(item, r):
    if r is None or r.get("status") != "ok":
        return {"status": "failed", "correct": False, "pred": "", "truncated": False}
    text = r.get("text") or ""
    pred = extract_mcq(text) if item["is_mcq"] else extract_numeric(text)
    ok_label = (pred != "" and pred == item["gt"].strip().upper()) if item["is_mcq"] else numeric_equal(pred, item["gt"])
    exact = exact_roundings(item)
    ok = any(numeric_equal(pred, g) for g in exact) if exact else ok_label
    legacy = (not item["is_mcq"]) and numeric_equal(pred, item["gt"],
                                                  tol=0.005 if item["bench"] == "numericbench" else 0.01)
    return {"status": "ok", "correct": bool(ok), "correct_label": bool(ok_label), "correct_tol": bool(ok or legacy), "pred": pred,
            "truncated": r.get("finish") in TRUNC, "marker": "####" in text if item["bench"] == "gsm_symbolic" else None}


def usage_tokens(c, r):
    u = r.get("usage") or {}
    if c.startswith("gemini"):
        return u.get("promptTokenCount", 0), u.get("thoughtsTokenCount", 0) or 0, u.get("candidatesTokenCount", 0) or 0
    if c == "claude":
        return (u.get("input_tokens", 0) or 0) + (u.get("cache_read_input_tokens", 0) or 0) + (u.get("cache_creation_input_tokens", 0) or 0), 0, u.get("output_tokens", 0)
    d = u.get("completion_tokens_details") or {}
    think = d.get("reasoning_tokens", 0) or 0
    return u.get("prompt_tokens", 0), think, (u.get("completion_tokens", 0) or 0) - think


def main():
    man, res = load()
    by_sub = collections.defaultdict(list)
    for it in man:
        by_sub[(it["bench"], it["subset"])].append(it)
    import decimal
    n_fix = 0
    for it in man:
        er = exact_roundings(it)
        if er and not any(numeric_equal(it["gt"], g) for g in er):
            n_fix += 1
    out_fix = {"n_arith_items": sum(1 for it in man if exact_roundings(it)), "n_reference_off": n_fix}
    out = {"label_check": out_fix, "manifest_sha256": hashlib.sha256(open(os.path.join(RUN, "manifest.jsonl"), "rb").read()).hexdigest(),
           "n_items": len(man), "numericbench": {}, "gsm_symbolic": {}, "runs": {}}
    scored = {c: {it["key"]: score_item(it, res[c].get(it["key"])) for it in man} for c in CONFIGS}
    for (bench, sub), items in sorted(by_sub.items()):
        cell = {"n": len(items)}
        for c in CONFIGS:
            s = [scored[c][it["key"]] for it in items]
            k = sum(x["correct"] for x in s)
            cell[c] = {"acc": k / len(s), "ci": wilson(k, len(s)), "n_failed": sum(x["status"] == "failed" for x in s),
                       "n_truncated": sum(x["truncated"] for x in s), "n_unparsed": sum(x["status"] == "ok" and x["pred"] == "" for x in s),
                       "acc_tol": sum(x.get("correct_tol", False) for x in s) / len(s),
                       "acc_original_labels": sum(x.get("correct_label", False) for x in s) / len(s)}
            if bench == "gsm_symbolic":
                cell[c]["share_marker"] = sum(bool(x.get("marker")) for x in s) / len(s)
        if bench == "gsm_symbolic":
            tids = sorted({it["template"] for it in items})
            groups = {t: [i for i, it in enumerate(items) if it["template"] == t] for t in tids}
            for c in CONFIGS:
                y = np.array([scored[c][it["key"]]["correct"] for it in items], float)
                per_t = [y[groups[t]].mean() for t in tids]
                boots = [np.concatenate([y[groups[tids[j]]] for j in RNG.integers(0, len(tids), len(tids))]).mean()
                         for _ in range(2000)]
                cell[c]["ci"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
                cell[c]["template_sd"] = float(np.mean([y[groups[t]].std() for t in tids]))
                cell[c]["share_templates_all_correct"] = float(np.mean(np.array(per_t) == 1.0))
            cell["n_templates"] = len(tids)
        out[bench][sub] = cell
    for c in CONFIGS:
        rs = [r for r in res[c].values() if r.get("status") == "ok"]
        toks = np.array([usage_tokens(c, r) for r in rs], float) if rs else np.zeros((1, 3))
        lat = np.array([r.get("latency_s", 0) for r in rs]) if rs else np.zeros(1)
        ts = sorted(r["ts"] for r in res[c].values() if r.get("ts"))
        out["runs"][c] = {"n_ok": len(rs), "n_attempted": len(man),
                          "model_versions": dict(collections.Counter(r.get("model_version") for r in rs)),
                          "mean_tokens": {"input": float(toks[:, 0].mean()), "thinking": float(toks[:, 1].mean()), "output": float(toks[:, 2].mean())},
                          "total_tokens": {"input": float(toks[:, 0].sum()), "thinking": float(toks[:, 1].sum()), "output": float(toks[:, 2].sum())},
                          "latency_s": {"mean": float(lat.mean()), "p50": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
                          "first_ts": ts[0] if ts else None, "last_ts": ts[-1] if ts else None,
                          "share_truncated": float(np.mean([r.get("finish") in TRUNC for r in rs])) if rs else 0.0}
    with open(os.path.join(HERE, "out", "ctx_rerun_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    for bench in ("numericbench", "gsm_symbolic"):
        for sub, cell in out[bench].items():
            print(f"{bench[:3]} {sub:<20} n={cell['n']:<5}", "  ".join(
                f"{c}:{cell[c]['acc']:.3f}(fail {cell[c]['n_failed']}, trunc {cell[c]['n_truncated']}, unp {cell[c]['n_unparsed']})" for c in CONFIGS))
    for c, r in out["runs"].items():
        print(c, r["n_ok"], r["model_versions"], {k: round(v) for k, v in r["mean_tokens"].items()}, r["latency_s"], r["first_ts"], r["last_ts"])


if __name__ == "__main__":
    main()
