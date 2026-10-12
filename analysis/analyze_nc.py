"""Number Cookbook analysis from the frozen snapshot (analysis/snapshot/nc_items.jsonl.gz).

Primary analysis
    * Item set: tasks where every configuration returned a successful API response for >= 90% of
      items; within those tasks, only items answered (status ok) by all four configurations.
      Infrastructure failures (e.g. depleted API credits) are therefore excluded symmetrically,
      while model behaviour such as truncation or unparsable output counts as incorrect.
    * Scorer: analysis/scoring.py (sign-aware, last "The answer is" marker, strict format).
    * Aggregates: unweighted mean over tasks of item-level accuracy, split into in-domain and
      out-of-domain operand lengths using Number Cookbook's thresholds.
    * Uncertainty: 95% percentile intervals from a stratified item bootstrap (items resampled
      within task x range cells, identical resamples across models for paired contrasts) and,
      for RG-PG gaps, a task-level bootstrap (tasks resampled within the RG and PG groups).

Sensitivity analyses: intention-to-evaluate on all items, first-marker extraction, lenient
extraction, the legacy scorer with failures dropped (reproduces the earlier manuscript), and
three alternative RG/PG codings.
"""

from __future__ import annotations

import collections
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
SNAP = os.path.join(HERE, "snapshot", "nc_items.jsonl.gz")
OUT = os.path.join(HERE, "out")
MODELS = ["gpt", "claude", "gemini_minimal", "gemini_high"]
LABELS = {"gpt": "GPT-5.4", "claude": "Claude Opus 4.6", "gemini_minimal": "Gemini 3 Flash MINIMAL",
          "gemini_high": "Gemini 3 Flash HIGH"}
RNG = np.random.default_rng(20261006)
B = 1000
COVERAGE_MIN = 0.90

# ── RG / PG coding (identical to section5_analysis.py, the coding used in the manuscript) ──
RG_A = {
    "max_Integer_Integer_Integer", "max_Float_Float_Float", "max_Fraction_Fraction_Fraction",
    "max_ScientificNotation_ScientificNotation_ScientificNotation", "max_hard_Integer_Integer_Integer",
    "max_hard_Float_Float_Float", "max_hard_ScientificNotation_ScientificNotation_ScientificNotation",
    "to_float_Fraction_none_Float", "to_float_ScientificNotation_none_Float",
    "to_scient_Float_none_ScientificNotation", "to_scient_Integer_none_ScientificNotation",
    "sig_Float_int_ScientificNotation", "sig_Integer_int_ScientificNotation",
    "length_Float_none_int", "length_Integer_none_int", "count_Integer_int_int",
}
CONVERSION = {t for t in RG_A if t.startswith(("to_float", "to_scient", "sig_"))}
STRUCTURAL_RG = {t for t in RG_A if t.startswith(("length", "count"))}
DIGIT_ACCESS = {"get_digit_Float_int_int", "get_digit_Integer_int_int",
                "digit_max_Float_Float_Float", "digit_max_Integer_Integer_Integer"}


def codings(tasks: set[str]) -> dict[str, dict[str, set[str]]]:
    rg_a = RG_A & tasks
    pg_a = tasks - RG_A
    arith = {t for t in pg_a if not t.startswith(("digit_", "get_digit"))}
    return {
        "A_original": {"RG": rg_a, "PG": pg_a},
        "B_conversion_as_PG": {"RG": rg_a - CONVERSION, "PG": pg_a | (CONVERSION & tasks)},
        "C_core_only": {"RG": {t for t in rg_a if t.startswith("max")}, "PG": arith},
        "D_digit_access_as_RG": {"RG": rg_a | (DIGIT_ACCESS & tasks), "PG": pg_a - DIGIT_ACCESS},
    }


def range_of(task_max_digit: int, digit: int) -> str:
    """Number Cookbook in-/out-of-domain split, identical to MetricsRecorder.statistics().

    The thresholds depend on the largest operand length in the task, which distinguishes short
    from long tasks and the full from the half-size test set used here.
    """
    if task_max_digit <= 15:
        cut = 5
    elif task_max_digit <= 25:
        cut = 9
    elif task_max_digit <= 70:
        cut = 11
    else:
        cut = 21
    return "in" if digit < cut else "out"


def load():
    rows = collections.defaultdict(dict)  # (id) -> model -> row
    meta = {}
    with gzip.open(SNAP, "rt") as f:
        for line in f:
            d = json.loads(line)
            rows[d["id"]][d["model"]] = d
            meta[d["id"]] = (d["task"], d["digit"])
    return rows, meta


def cell_arrays(rows, meta, ids, field):
    """task -> range -> array[n_items, n_models]"""
    maxd = collections.defaultdict(int)
    for i, (t, dgt) in meta.items():
        maxd[t] = max(maxd[t], dgt)
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in ids:
        t, dgt = meta[i]
        cells[t][range_of(maxd[t], dgt)].append([rows[i][m][field] for m in MODELS])
    return {t: {r: np.asarray(v, dtype=float) for r, v in rr.items()} for t, rr in cells.items()}


def task_means(cells):
    return {t: {r: a.mean(axis=0) for r, a in rr.items()} for t, rr in cells.items()}


def group_mean(tm, tasks, rng_key):
    vals = [tm[t][rng_key] for t in tasks if rng_key in tm[t]]
    return np.mean(vals, axis=0) if vals else np.full(len(MODELS), np.nan)


def bootstrap_task_means(cells):
    """B resamples of per-task means: task -> range -> array[B, n_models]."""
    out = {}
    for t, rr in cells.items():
        out[t] = {}
        for r, a in rr.items():
            n = len(a)
            idx = RNG.integers(0, n, size=(B, n))
            out[t][r] = a[idx].mean(axis=1)
    return out


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=0)
    return lo, hi


def summarize(cells, groups):
    # Sorted lists, not sets: str hashing is randomized per process, and the bootstrap draws
    # must map to the same tasks on every run for the outputs to be reproducible.
    groups = {k: sorted(v) for k, v in groups.items()}
    tm = task_means(cells)
    bt = bootstrap_task_means(cells)
    res = {"per_model": {}, "contrasts": {}}
    tasks_all = sorted(cells)
    for rk in ("in", "out"):
        allm = group_mean(tm, tasks_all, rk)
        boot_all = np.mean([bt[t][rk] for t in tasks_all if rk in bt[t]], axis=0)
        rg = group_mean(tm, groups["RG"], rk)
        pg = group_mean(tm, groups["PG"], rk)
        b_rg = np.mean([bt[t][rk] for t in groups["RG"] if rk in bt[t]], axis=0)
        b_pg = np.mean([bt[t][rk] for t in groups["PG"] if rk in bt[t]], axis=0)
        gap_lo, gap_hi = ci(b_rg - b_pg)
        all_lo, all_hi = ci(boot_all)
        rg_lo, rg_hi = ci(b_rg)
        pg_lo, pg_hi = ci(b_pg)
        # task-level bootstrap for the RG-PG gap
        rgt = [t for t in groups["RG"] if rk in tm[t]]
        pgt = [t for t in groups["PG"] if rk in tm[t]]
        rg_arr = np.array([tm[t][rk] for t in rgt])
        pg_arr = np.array([tm[t][rk] for t in pgt])
        tb = (rg_arr[RNG.integers(0, len(rgt), (B, len(rgt)))].mean(1)
              - pg_arr[RNG.integers(0, len(pgt), (B, len(pgt)))].mean(1))
        tlo, thi = ci(tb)
        for j, m in enumerate(MODELS):
            pm = res["per_model"].setdefault(m, {})
            pm[rk] = {
                "all": float(allm[j]), "all_ci": [float(all_lo[j]), float(all_hi[j])],
                "RG": float(rg[j]), "RG_ci": [float(rg_lo[j]), float(rg_hi[j])],
                "PG": float(pg[j]), "PG_ci": [float(pg_lo[j]), float(pg_hi[j])],
                "gap": float(rg[j] - pg[j]), "gap_ci_item": [float(gap_lo[j]), float(gap_hi[j])],
                "gap_ci_task": [float(tlo[j]), float(thi[j])],
                "n_RG_tasks": len(rgt), "n_PG_tasks": len(pgt),
            }
        # paired contrast: HIGH - MINIMAL
        hi_i, mi_i = MODELS.index("gemini_high"), MODELS.index("gemini_minimal")
        for name, b in (("all", boot_all), ("RG", b_rg), ("PG", b_pg)):
            d = b[:, hi_i] - b[:, mi_i]
            point = {"all": allm, "RG": rg, "PG": pg}[name]
            lo, hi = ci(d)
            res["contrasts"][f"high_minus_minimal_{name}_{rk}"] = {
                "delta": float(point[hi_i] - point[mi_i]), "ci": [float(lo), float(hi)]}
        dd = (b_pg[:, hi_i] - b_pg[:, mi_i]) - (b_rg[:, hi_i] - b_rg[:, mi_i])
        lo, hi = ci(dd)
        res["contrasts"][f"dPG_minus_dRG_{rk}"] = {
            "delta": float((pg[hi_i] - pg[mi_i]) - (rg[hi_i] - rg[mi_i])), "ci": [float(lo), float(hi)]}
    return res, tm


def well_learned(rows, meta, ids, model, thr=0.9):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in ids:
        t, dgt = meta[i]
        per[t][dgt].append(rows[i][model]["exact"])
    out = {}
    for t, dd in per.items():
        ok = [d for d, v in dd.items() if np.mean(v) >= thr]
        out[t] = {"well_learned_digit": max(ok) if ok else 0,
                  "n_per_digit": int(np.median([len(v) for v in dd.values()]))}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, meta = load()
    all_ids = sorted(rows)
    # coverage per task and configuration
    cov = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    for i in all_ids:
        t = meta[i][0]
        tot[t] += 1
        for m in MODELS:
            cov[t][m] += rows[i][m]["status"] == "ok"
    excluded = sorted(t for t in tot if min(cov[t][m] / tot[t] for m in MODELS) < COVERAGE_MIN)
    kept = set(tot) - set(excluded)
    matched = [i for i in all_ids if meta[i][0] in kept and all(rows[i][m]["status"] == "ok" for m in MODELS)]
    results = {
        "design": {
            "n_items_total": len(all_ids), "n_tasks_total": len(tot), "coverage_min": COVERAGE_MIN,
            "excluded_tasks": {t: {m: round(cov[t][m] / tot[t], 3) for m in MODELS} for t in excluded},
            "n_tasks_primary": len(kept), "n_items_primary": len(matched), "bootstrap_reps": B,
        },
        "coverage": json.load(open(os.path.join(HERE, "snapshot", "nc_coverage.json"))),
    }
    groups_all = codings(kept)
    for name, g in groups_all.items():
        cells = cell_arrays(rows, meta, matched, "exact")
        res, tm = summarize(cells, g)
        results.setdefault("codings", {})[name] = res
        if name == "A_original":
            results["primary"] = res
            results["per_task_exact"] = {t: {r: dict(zip(MODELS, map(float, v))) for r, v in rr.items()}
                                         for t, rr in tm.items()}
            results["groups"] = {k: sorted(v) for k, v in g.items()}
    # digit match (primary set)
    dres, _ = summarize(cell_arrays(rows, meta, matched, "digit_match"), groups_all["A_original"])
    results["primary_digit_match"] = dres["per_model"]
    # sensitivity analyses on the primary item set
    for field in ("exact_first", "exact_lenient", "exact_value"):
        r, _ = summarize(cell_arrays(rows, meta, matched, field), groups_all["A_original"])
        results.setdefault("sensitivity", {})[field] = r["per_model"]
    # intention-to-evaluate on all items and all tasks
    groups_full = codings(set(tot))["A_original"]
    r, _ = summarize(cell_arrays(rows, meta, all_ids, "exact"), groups_full)
    results["sensitivity"]["itt_all_items"] = r["per_model"]
    results["sensitivity"]["itt_all_items_contrasts"] = r["contrasts"]
    # legacy reproduction: legacy scorer on all items and tasks. As in text_model/metrics.py,
    # failed Gemini requests count as wrong and the single failed Claude request is dropped.
    legacy = {}
    for m in MODELS:
        ids_m = [i for i in all_ids if m != "claude" or rows[i][m]["status"] == "ok"]
        maxd = collections.defaultdict(int)
        for i in all_ids:
            maxd[meta[i][0]] = max(maxd[meta[i][0]], meta[i][1])
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in ids_m:
            t, dgt = meta[i]
            per[t][range_of(maxd[t], dgt)].append(rows[i][m]["exact_legacy"])
        legacy[m] = {}
        for rk in ("in", "out"):
            tmean = {t: np.mean(v[rk]) for t, v in per.items() if v.get(rk)}
            legacy[m][rk] = {
                "all": float(np.mean(list(tmean.values()))),
                "RG": float(np.mean([v for t, v in tmean.items() if t in RG_A])),
                "PG": float(np.mean([v for t, v in tmean.items() if t not in RG_A])),
            }
    results["sensitivity"]["legacy_scorer_failures_dropped"] = legacy
    # scorer change impact on the primary set
    diff = collections.Counter()
    for i in matched:
        for m in MODELS:
            r = rows[i][m]
            diff[(m, "legacy_correct_new_wrong")] += r["exact_legacy"] and not r["exact"]
            diff[(m, "legacy_wrong_new_correct")] += (not r["exact_legacy"]) and r["exact"]
            diff[(m, "sign_mismatch")] += not r["sign_ok"]
            diff[(m, "truncated")] += r["truncated"]
    results["scorer_impact_primary"] = {f"{m}:{k}": int(v) for (m, k), v in diff.items()}
    # tokens (primary items)
    tok = {}
    for m in MODELS:
        a = np.array([[rows[i][m]["in_tok"], rows[i][m]["think_tok"], rows[i][m]["out_tok"]] for i in matched], float)
        tok[m] = {"input": a[:, 0].mean(), "thinking": a[:, 1].mean(), "output": a[:, 2].mean(),
                  "total": a.sum(1).mean()}
    tok["high_over_minimal_total"] = tok["gemini_high"]["total"] / tok["gemini_minimal"]["total"]
    results["tokens_primary"] = json.loads(json.dumps(tok, default=float))
    # truncation-free accuracy (responses not cut by the token limit)
    nt = {}
    for m in MODELS:
        ok = [rows[i][m]["exact"] for i in matched if not rows[i][m]["truncated"]]
        nt[m] = {"n": len(ok), "share_truncated": 1 - len(ok) / len(matched)}
    results["truncation_primary"] = nt
    results["well_learned_digit"] = {m: well_learned(rows, meta, matched, m) for m in MODELS}
    # --- A-tier additions -------------------------------------------------------------
    maxd = collections.defaultdict(int)
    for i in all_ids:
        maxd[meta[i][0]] = max(maxd[meta[i][0]], meta[i][1])
    # micro (item-level) accuracy on the primary item set
    micro = {}
    for m in MODELS:
        for rk in ("in", "out"):
            v = [rows[i][m]["exact"] for i in matched if range_of(maxd[meta[i][0]], meta[i][1]) == rk]
            micro.setdefault(m, {})[rk] = float(np.mean(v))
    results["micro_exact"] = micro
    # response type: visible reasoning (> 20 output tokens) versus a direct answer
    rtype = {}
    for m in MODELS:
        vis = [rows[i][m]["out_tok"] > 20 for i in matched]
        acc_vis = [rows[i][m]["exact"] for i in matched if rows[i][m]["out_tok"] > 20]
        acc_dir = [rows[i][m]["exact"] for i in matched if rows[i][m]["out_tok"] <= 20]
        rtype[m] = {"share_visible_reasoning": float(np.mean(vis)),
                    "acc_visible": float(np.mean(acc_vis)) if acc_vis else float("nan"),
                    "acc_direct": float(np.mean(acc_dir)) if acc_dir else float("nan")}
    results["response_type"] = rtype
    # ceiling check: RG-PG gap on the logit scale of task accuracies, task-level bootstrap
    tm = results["per_task_exact"]
    g = results["groups"]
    lg = lambda p: np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))  # noqa: E731
    logit = {}
    for j, m in enumerate(MODELS):
        for rk in ("in", "out"):
            rg = np.array([lg(tm[t][rk][m]) for t in g["RG"] if rk in tm[t]])
            pg = np.array([lg(tm[t][rk][m]) for t in g["PG"] if rk in tm[t]])
            bt = (rg[RNG.integers(0, len(rg), (B, len(rg)))].mean(1) - pg[RNG.integers(0, len(pg), (B, len(pg)))].mean(1))
            logit.setdefault(m, {})[rk] = {"gap": float(rg.mean() - pg.mean()),
                                           "ci": [float(x) for x in np.percentile(bt, [2.5, 97.5])]}
    results["logit_gap"] = logit
    # well-learned digit with bootstrap intervals (items resampled within each length)
    wl_ci = {}
    fig_tasks = ["max_Integer_Integer_Integer", "max_Float_Float_Float", "max_hard_Float_Float_Float",
                 "length_Integer_none_int", "length_Float_none_int", "count_Integer_int_int",
                 "to_scient_Integer_none_ScientificNotation", "sig_Integer_int_ScientificNotation"]
    for m in MODELS:
        for t in fig_tasks:
            per = collections.defaultdict(list)
            for i in matched:
                if meta[i][0] == t:
                    per[meta[i][1]].append(rows[i][m]["exact"])
            digits = sorted(per)
            arrs = [np.array(per[d], float) for d in digits]
            draws = []
            for _ in range(200):
                ok = [d for d, a in zip(digits, arrs) if a[RNG.integers(0, len(a), len(a))].mean() >= 0.9]
                draws.append(max(ok) if ok else 0)
            wl_ci.setdefault(m, {})[t] = [int(x) for x in np.percentile(draws, [2.5, 97.5])]
    results["well_learned_ci"] = wl_ci
    # per task and operand length: accuracy per configuration (primary items), for figures
    ptd = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in matched:
        t, dgt = meta[i]
        ptd[t][dgt].append([rows[i][m]["exact"] for m in MODELS])
    results["per_task_digit_exact"] = {
        t: {str(d): [round(float(x), 4) for x in np.mean(v, axis=0)] for d, v in sorted(dd.items())}
        for t, dd in ptd.items()}
    with open(os.path.join(OUT, "nc_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    p = results["primary"]["per_model"]
    for m in MODELS:
        print(f"{LABELS[m]:<26} all in {p[m]['in']['all']:.3f} out {p[m]['out']['all']:.3f} | "
              f"RG {p[m]['in']['RG']:.3f}/{p[m]['out']['RG']:.3f} PG {p[m]['in']['PG']:.3f}/{p[m]['out']['PG']:.3f} | "
              f"gap in {p[m]['in']['gap']:+.3f} {p[m]['in']['gap_ci_task']} out {p[m]['out']['gap']:+.3f}")
    print(json.dumps(results["primary"]["contrasts"], indent=0))
    print("excluded:", list(results["design"]["excluded_tasks"]))
    print("legacy:", json.dumps(legacy))


if __name__ == "__main__":
    main()
