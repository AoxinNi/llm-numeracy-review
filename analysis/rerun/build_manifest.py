"""Freeze the item manifest for the contextual-benchmark rerun (NumericBench + GSM-Symbolic).

NumericBench: the same items as the original run (numericbench_eval.load_and_sample, seed 42).
GSM-Symbolic main: the same 1,000 items as the original run (identical for every model).
GSM-Symbolic P1/P2: 10 instances per template, chosen deterministically per template; instances
the original GPT-5.4 run already answered are preferred so that old GPT results overlap maximally.

Output: analysis/rerun/manifest.jsonl (one request per line) and manifest.sha256.
Run from the repository root. Requires the NumericBench data in numericbench_data/ (see
numericbench_eval.py) and the apple/GSM-Symbolic dataset in the Hugging Face cache.
"""
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())
import numericbench_eval as nb  # noqa: E402  (reads api_config at import; no calls are made)

HERE = os.path.dirname(os.path.abspath(__file__))
GSM = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--apple--GSM-Symbolic/snapshots/*/"))[0]
OLD = "data/contextual_v1/gsm_symbolic_results"
PER_TEMPLATE = 10

GSM_SYSTEM = ("Solve the following math problem step by step. End your response with the final numerical "
              "answer on a new line preceded by ####. For example: #### 42")


def gsm_gt(answer: str) -> str:
    import re
    m = re.search(r"####\s*(.+)", answer)
    return m.group(1).strip() if m else answer.strip().split("\n")[-1].strip()


def old_keys(model: str, split: str) -> set:
    p = f"{OLD}/{model}_{split}_results.jsonl"
    return {(r["id"], r["instance"]) for r in map(json.loads, open(p))} if os.path.exists(p) else set()


def main():
    rows = []
    for name, cfg in nb.SUBSETS.items():
        for item in nb.load_and_sample(name, cfg):
            system, user = nb.format_prompt(item)
            rows.append({"key": f"nb|{name}|{item['idx']}", "bench": "numericbench", "subset": name,
                         "system": system, "user": user, "gt": str(item["answer"]), "is_mcq": cfg["is_mcq"],
                         "grounding": cfg["grounding"]})
    main_keys = old_keys("claude", "main")
    assert main_keys == old_keys("gpt", "main") == old_keys("gemini_high", "main")
    for split in ("main", "p1", "p2"):
        items = [json.loads(l) for l in open(os.path.join(GSM, split, "test.jsonl"))]
        by_t = {}
        for it in items:
            by_t.setdefault(it["original_id"], []).append(it)
        prefer = old_keys("gpt", split)
        chosen = []
        for tid in sorted(by_t):
            pool = sorted(by_t[tid], key=lambda it: it["instance"])
            if split == "main":
                pick = [it for it in pool if (it["id"], it["instance"]) in main_keys]
            else:
                rng = random.Random(f"{split}:{tid}")
                rng.shuffle(pool)
                pool.sort(key=lambda it: (it["id"], it["instance"]) not in prefer)  # stable: preferred first
                pick = pool[:PER_TEMPLATE]
            chosen += pick
        for it in chosen:
            rows.append({"key": f"gsm|{split}|{it['id']}|{it['instance']}", "bench": "gsm_symbolic",
                         "subset": split, "template": it["original_id"], "system": GSM_SYSTEM,
                         "user": it["question"], "gt": gsm_gt(it["answer"]), "is_mcq": False})
    out = os.path.join(HERE, "manifest.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    open(os.path.join(HERE, "manifest.sha256"), "w").write(h + "  manifest.jsonl\n")
    from collections import Counter
    print(Counter(r["subset"] for r in rows), len(rows), h[:12])


if __name__ == "__main__":
    main()
