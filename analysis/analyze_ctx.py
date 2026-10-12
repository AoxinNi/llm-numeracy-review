"""Re-score NumericBench and GSM-Symbolic results with one scorer and matched item sets.

Raw responses were stored truncated to 500 characters by the original evaluation scripts. A
response is therefore *re-extractable* only if its stored text is shorter than 500 characters;
otherwise the prediction stored at run time is used ("stored-only"), and for multiple-choice items
whose stored prediction is not a valid option the item is *unscorable*.

Primary rules
    * Numeric answers: value after the last "The answer is"/"####" marker, else the last number
      in the response; thousands separators are handled; exact numerical equality.
    * Multiple choice: the option letter after the last "answer" marker, a bare letter, or a
      letter alone on the last line; anything else is unparsed (incorrect).
    * Matched items: only items attempted for every configuration enter cross-model tables.
    * Uncertainty: Wilson 95% intervals for single proportions; cluster bootstrap over GSM
      templates (B = 2000) for GSM-Symbolic.
Sensitivity: the original extraction and absolute tolerances (0.005 NumericBench, 0.01 GSM).
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scoring import _clean, extract_last_number, normalize_number, numeric_equal  # noqa: E402

HERE = os.path.dirname(__file__)
BENCH = os.path.join(HERE, "..", "data", "contextual_v1")
OUT = os.path.join(HERE, "out")
MODELS = ["gpt", "claude", "gemini_minimal", "gemini_high"]
RNG = np.random.default_rng(20261006)
STORE_LIMIT = 500
NB_MATCHED = ["arithmetic", "context_arithmetic", "different_digit", "num_list_100", "mixed_number_string"]

_MARK = re.compile(r"(?:the answer is|####)\s*:?\s*", re.I)
_MCQ_MARK = re.compile(r"answer(?:\s+is)?\s*[:：]?\s*\(?\**\s*([A-H])\b", re.I)
OPTIONS = list("ABCDEFGH")
MCQ_STORED_MAX = 0.05  # above this share of stored-only MCQ predictions a cell is not scorable
NUM_STORED_MAX = 0.20  # same for numeric answers, whose stored extraction is less fragile


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - r) / d, (c + r) / d]


def extract_numeric(text: str) -> str:
    marks = list(_MARK.finditer(text))
    if marks:
        tail = _clean(text[marks[-1].end():])
        m = re.match(r"[+-]?\$?\d[\d,]*(?:\.\d+)?", tail)
        if m:
            return normalize_number(m.group().replace("$", ""))
    return extract_last_number(text)


def extract_mcq(text: str) -> str:
    t = text.strip()
    marks = list(_MCQ_MARK.finditer(t))
    if marks:
        return marks[-1].group(1).upper()
    bare = _clean(t).strip(".()")
    if re.fullmatch(r"[A-H]", bare):
        return bare
    last = _clean(t.splitlines()[-1]).strip(".()") if t else ""
    if re.fullmatch(r"[A-H]", last):
        return last
    return ""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def score_nb():
    out = {"per_subset": {}, "coverage": {}, "legacy": {}}
    data = {m: load_jsonl(os.path.join(BENCH, "numericbench_results", f"{m}_results.jsonl")) for m in MODELS}
    by = {m: {(r["subset"], r["idx"]): r for r in rows} for m, rows in data.items()}
    subsets = sorted({r["subset"] for rows in data.values() for r in rows})
    for sub in subsets:
        keysets = [{k for k in by[m] if k[0] == sub} for m in MODELS]
        matched = set.intersection(*keysets)
        res = {"n_matched": len(matched), "n_per_model": {m: len(k) for m, k in zip(MODELS, keysets)},
               "complete_for_all": sub in NB_MATCHED}
        for m in MODELS:
            c = collections.Counter()
            for k in sorted(matched):
                r = by[m][k]
                raw = r["raw_response"] or ""
                full = len(raw) < STORE_LIMIT and not raw.startswith("ERROR")
                c["n"] += 1
                c["error"] += raw.startswith("ERROR")
                if r["is_mcq"]:
                    if full:
                        pred = extract_mcq(raw)
                    elif r["predicted"] in OPTIONS:
                        pred, c["stored_only"] = r["predicted"], c["stored_only"] + 1
                    else:
                        pred, c["unscorable"] = "", c["unscorable"] + 1
                    ok = pred != "" and pred == r["ground_truth"].strip().upper()
                else:
                    if full:
                        pred = extract_numeric(raw)
                    else:
                        pred, c["stored_only"] = r["predicted"], c["stored_only"] + 1
                    ok = numeric_equal(pred, r["ground_truth"])
                c["parsed"] += pred != ""
                c["correct"] += ok
                if full:
                    c["n_full"] += 1
                    c["correct_full"] += ok
                c["legacy_correct"] += bool(r["correct"])
            acc = c["correct"] / c["n"] if c["n"] else float("nan")
            is_mcq = by[m][next(iter(matched))]["is_mcq"] if matched else False
            stored_share = (c["stored_only"] + c["unscorable"]) / c["n"] if c["n"] else 0.0
            scorable = stored_share <= (MCQ_STORED_MAX if is_mcq else NUM_STORED_MAX)
            res[m] = {"acc": acc if scorable else float("nan"), "scorable": scorable,
                      "acc_reextractable": c["correct_full"] / c["n_full"] if c["n_full"] else float("nan"),
                      "ci": wilson(c["correct"], c["n"]) if scorable else [float("nan")] * 2, "legacy_acc": c["legacy_correct"] / c["n"] if c["n"] else float("nan"),
                      **{k: int(v) for k, v in c.items()}}
        out["per_subset"][sub] = res
    return out


def score_gsm():
    out = {}
    for split in ("main", "p1"):
        data = {m: {(r["id"], r["instance"]): r for r in
                    load_jsonl(os.path.join(BENCH, "gsm_symbolic_results", f"{m}_{split}_results.jsonl"))}
                for m in MODELS}
        matched = sorted(set.intersection(*[set(d) for d in data.values()]))
        templates = sorted({k[0] for k in matched})
        tidx = {t: i for i, t in enumerate(templates)}
        res = {"n_matched": len(matched), "n_templates": len(templates),
               "n_per_model": {m: len(d) for m, d in data.items()}}
        corr = {}
        for m in MODELS:
            c = collections.Counter()
            y = []
            for k in matched:
                r = data[m][k]
                raw = r["raw_response"] or ""
                full = len(raw) < STORE_LIMIT and not raw.startswith("ERROR")
                if full:
                    pred = extract_numeric(raw)
                    c["has_marker"] += "####" in raw
                else:
                    pred = r["predicted"]
                    c["stored_only"] += 1
                ok = numeric_equal(pred, r["ground_truth"])
                y.append(ok)
                c["correct"] += ok
                c["legacy_correct"] += bool(r["correct"])
                c["incomplete_no_marker"] += full and "####" not in raw
                done = (not full) or "####" in raw
                c["completed"] += done
                c["correct_completed"] += ok and done
            y = np.array(y, float)
            corr[m] = y
            # per-template accuracy and fragility
            per_t = collections.defaultdict(list)
            for k, v in zip(matched, y):
                per_t[k[0]].append(v)
            tacc = np.array([np.mean(v) for v in per_t.values()])
            res[m] = {"acc": float(y.mean()), "legacy_acc": c["legacy_correct"] / len(matched),
                      "share_completed": c["completed"] / len(matched),
                      "acc_completed": c["correct_completed"] / c["completed"] if c["completed"] else float("nan"),
                      "template_sd": float(np.mean([np.std(v) for v in per_t.values()])),
                      "share_templates_all_correct": float(np.mean(tacc == 1.0)),
                      **{k: int(v) for k, v in c.items()}}
        # cluster bootstrap over templates
        groups = collections.defaultdict(list)
        for j, k in enumerate(matched):
            groups[tidx[k[0]]].append(j)
        glist = [np.array(groups[i]) for i in range(len(templates))]
        boots = {m: [] for m in MODELS}
        for _ in range(2000):
            pick = RNG.integers(0, len(glist), len(glist))
            idx = np.concatenate([glist[i] for i in pick])
            for m in MODELS:
                boots[m].append(corr[m][idx].mean())
        for m in MODELS:
            res[m]["ci"] = [float(x) for x in np.percentile(boots[m], [2.5, 97.5])]
        out[split] = res
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    import glob
    import hashlib
    inputs = sorted(glob.glob(os.path.join(BENCH, "numericbench_results", "*_results.jsonl"))
                    + glob.glob(os.path.join(BENCH, "gsm_symbolic_results", "*_results.jsonl")))
    results = {"numericbench": score_nb(), "gsm_symbolic": score_gsm(),
               "input_sha256": {os.path.relpath(p, BENCH): hashlib.sha256(open(p, "rb").read()).hexdigest()
                                for p in inputs}}
    with open(os.path.join(OUT, "ctx_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for sub, r in results["numericbench"]["per_subset"].items():
        print(f"NB {sub:<20} n={r['n_matched']:<5}", "  ".join(
            f"{m}:{r[m]['acc']:.3f}(full {r[m]['acc_reextractable']:.3f}, leg {r[m]['legacy_acc']:.3f}, stored {r[m].get('stored_only', 0)})"
            for m in MODELS))
    for sp, r in results["gsm_symbolic"].items():
        print(f"GSM {sp} n={r['n_matched']} templates={r['n_templates']}", "  ".join(
            f"{m}:{r[m]['acc']:.3f}[{r[m]['ci'][0]:.3f},{r[m]['ci'][1]:.3f}](leg {r[m]['legacy_acc']:.3f}, done {r[m]['share_completed']:.2f}, acc|done {r[m]['acc_completed']:.3f})"
            for m in MODELS))


if __name__ == "__main__":
    main()
