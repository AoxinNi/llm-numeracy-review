"""
GSM-Symbolic Evaluation Pipeline

[Release note] Script of the first, superseded contextual run. Its outputs are in
data/contextual_v1/ and are used in the paper only for a provenance note; the reported results come
from analysis/rerun/. numericbench_eval.py also supplies the item sampling used by
analysis/rerun/build_manifest.py.

Tests fragility (RG) by measuring performance variance across parametric
variations of the same word problem template.

Sampling: 100 templates × 10 instances = 1,000 items per model (from 'main' split).
Also evaluates p1 (added clause) and p2 (added clause + extra operation) at same rate.
"""

import json
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download

import api_config as _cfg

REPO_ID = "apple/GSM-Symbolic"
OUTPUT_DIR = Path("data/contextual_v1/gsm_symbolic_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES_PER_TEMPLATE = 10
random.seed(42)


def load_and_sample(split: str) -> list[dict]:
    path = hf_hub_download(repo_id=REPO_ID, filename=f"{split}/test.jsonl", repo_type="dataset")
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))

    by_template = defaultdict(list)
    for item in items:
        by_template[item["original_id"]].append(item)

    sampled = []
    for tid in sorted(by_template):
        pool = by_template[tid]
        random.shuffle(pool)
        sampled.extend(pool[:INSTANCES_PER_TEMPLATE])

    return sampled


def extract_ground_truth(answer_str: str) -> str:
    match = re.search(r'####\s*(.+)', answer_str)
    if match:
        return match.group(1).strip()
    return answer_str.strip().split('\n')[-1].strip()


def format_prompt(item: dict) -> tuple[str, str]:
    system = (
        "Solve the following math problem step by step. "
        "End your response with the final numerical answer on a new line "
        "preceded by ####. For example: #### 42"
    )
    return system, item["question"]


def extract_answer(response: str) -> str:
    match = re.search(r'####\s*(.+)', response)
    if match:
        ans = match.group(1).strip()
        num_match = re.search(r'-?\d+(?:\.\d+)?', ans)
        if num_match:
            return num_match.group(0)
        return ans

    lines = response.strip().split('\n')
    for line in reversed(lines):
        num_match = re.search(r'-?\d+(?:\.\d+)?', line)
        if num_match:
            return num_match.group(0)
    return response.strip()


def evaluate(predicted: str, ground_truth: str) -> bool:
    try:
        return abs(float(predicted) - float(ground_truth)) < 0.01
    except (ValueError, TypeError):
        return predicted.strip() == ground_truth.strip()


# --- API Callers ---

def _get_openai_client(api_key, base_url=""):
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    if base_url:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def call_gpt(system, user):
    client = _get_openai_client(_cfg.OPENAI_API_KEY, _cfg.OPENAI_BASE_URL)
    r = client.chat.completions.create(
        model=_cfg.OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0, max_tokens=512,
    )
    return r.choices[0].message.content or ""


def call_claude(system, user):
    import anthropic
    kwargs = {"api_key": _cfg.ANTHROPIC_API_KEY}
    if _cfg.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = _cfg.ANTHROPIC_BASE_URL
    client = anthropic.Anthropic(**kwargs)
    r = client.messages.create(
        model=_cfg.ANTHROPIC_MODEL, system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0, max_tokens=512,
    )
    return r.content[0].text


def call_gemini(system, user, thinking_level="none"):
    client = _get_openai_client(_cfg.GOOGLE_API_KEY, _cfg.GOOGLE_BASE_URL)
    sys_msg = system
    if thinking_level == "high":
        sys_msg = "Think step by step carefully before answering. Show your reasoning process.\n\n" + system
    r = client.chat.completions.create(
        model=_cfg.GEMINI_MODEL,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        temperature=0, max_tokens=1024 if thinking_level == "high" else 512,
    )
    return r.choices[0].message.content or ""


MODEL_CALLERS = {
    "gpt": lambda s, u: call_gpt(s, u),
    "claude": lambda s, u: call_claude(s, u),
    "gemini_minimal": lambda s, u: call_gemini(s, u, "none"),
    "gemini_high": lambda s, u: call_gemini(s, u, "high"),
}


def run_evaluation(model_key: str, splits: list[str] = None, workers: int = 1):
    caller = MODEL_CALLERS[model_key]
    splits = splits or ["main"]
    write_lock = threading.Lock()

    def process_item(item, split):
        system, user = format_prompt(item)
        gt = extract_ground_truth(item["answer"])
        try:
            response = caller(system, user)
            predicted = extract_answer(response)
            is_correct = evaluate(predicted, gt)
        except Exception as e:
            print(f"  ERROR item {item['id']}: {e}")
            response = f"ERROR: {e}"
            predicted = ""
            is_correct = False
            time.sleep(2)
        return {
            "id": item["id"],
            "instance": item["instance"],
            "original_id": item["original_id"],
            "split": split,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": is_correct,
            "raw_response": response[:500],
        }

    for split in splits:
        results_file = OUTPUT_DIR / f"{model_key}_{split}_results.jsonl"

        completed_keys = set()
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    r = json.loads(line)
                    completed_keys.add((r["id"], r["instance"]))
            print(f"Resuming {model_key}/{split}: {len(completed_keys)} items done")

        items = load_and_sample(split)
        pending = [item for item in items if (item["id"], item["instance"]) not in completed_keys]
        print(f"\n=== {model_key} / {split}: {len(items)} items ({len(pending)} remaining) ===")

        correct_count = 0
        total_count = 0

        if workers <= 1:
            for i, item in enumerate(pending):
                result = process_item(item, split)
                with write_lock:
                    with open(results_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                total_count += 1
                correct_count += int(result["correct"])
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(pending)}] running acc: {correct_count}/{total_count} = {correct_count/max(total_count,1):.3f}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_item, item, split): i
                           for i, item in enumerate(pending)}
                for future in as_completed(futures):
                    result = future.result()
                    with write_lock:
                        with open(results_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                    total_count += 1
                    correct_count += int(result["correct"])
                    if total_count % 100 == 0:
                        print(f"  [{total_count}/{len(pending)}] running acc: {correct_count}/{total_count} = {correct_count/max(total_count,1):.3f}")

        if total_count > 0:
            print(f"  Final: {correct_count}/{total_count} = {correct_count/total_count:.3f}")


def analyze_results():
    """Analyze GSM-Symbolic results with focus on fragility (per-template variance)."""
    for split in ["main", "p1", "p2"]:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        model_data = {}
        for model_key in MODEL_CALLERS:
            results_file = OUTPUT_DIR / f"{model_key}_{split}_results.jsonl"
            if not results_file.exists():
                continue
            results = []
            with open(results_file) as f:
                for line in f:
                    r = json.loads(line)
                    if "ERROR" not in r.get("raw_response", ""):
                        results.append(r)
            if results:
                model_data[model_key] = results

        if not model_data:
            print("  No data.")
            continue

        # Overall accuracy
        print(f"\n{'Model':<20} {'Accuracy':>10} {'N':>6}")
        print("-" * 40)
        for m, results in model_data.items():
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"{m:<20} {acc:>10.3f} {len(results):>6}")

        # Per-template variance (fragility metric)
        print(f"\n{'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
        print("-" * 60)
        for m, results in model_data.items():
            by_template = defaultdict(list)
            for r in results:
                by_template[r["original_id"]].append(int(r["correct"]))

            template_accs = []
            for tid, corrects in by_template.items():
                if len(corrects) >= 3:
                    template_accs.append(sum(corrects) / len(corrects))

            if template_accs:
                import statistics
                mean = statistics.mean(template_accs)
                std = statistics.stdev(template_accs) if len(template_accs) > 1 else 0
                mn = min(template_accs)
                mx = max(template_accs)
                rng = mx - mn
                print(f"{m:<20} {mean:>8.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f} {rng:>8.3f}")

    # Save summary
    summary = {}
    for split in ["main", "p1", "p2"]:
        summary[split] = {}
        for model_key in MODEL_CALLERS:
            results_file = OUTPUT_DIR / f"{model_key}_{split}_results.jsonl"
            if not results_file.exists():
                continue
            results = []
            with open(results_file) as f:
                for line in f:
                    r = json.loads(line)
                    if "ERROR" not in r.get("raw_response", ""):
                        results.append(r)
            if not results:
                continue

            by_template = defaultdict(list)
            for r in results:
                by_template[r["original_id"]].append(int(r["correct"]))

            template_accs = [sum(v)/len(v) for v in by_template.values() if len(v) >= 3]

            import statistics
            summary[split][model_key] = {
                "overall_accuracy": sum(r["correct"] for r in results) / len(results),
                "n": len(results),
                "template_mean": statistics.mean(template_accs) if template_accs else None,
                "template_std": statistics.stdev(template_accs) if len(template_accs) > 1 else None,
                "template_min": min(template_accs) if template_accs else None,
                "template_max": max(template_accs) if template_accs else None,
            }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSM-Symbolic Evaluation")
    parser.add_argument("--model", choices=list(MODEL_CALLERS.keys()), help="Model to evaluate")
    parser.add_argument("--splits", nargs="+", default=["main"], choices=["main", "p1", "p2"])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent API calls (default: 8)")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.dry_run:
        for split in args.splits:
            items = load_and_sample(split)
            templates = len(set(i["original_id"] for i in items))
            print(f"{split}: {len(items)} items ({templates} templates × {INSTANCES_PER_TEMPLATE} instances)")
    elif args.model:
        run_evaluation(args.model, args.splits, workers=args.workers)
    else:
        parser.print_help()
