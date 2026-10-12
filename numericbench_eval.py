"""
NumericBench Evaluation Pipeline

[Release note] Script of the first, superseded contextual run. Its outputs are in
data/contextual_v1/ and are used in the paper only for a provenance note; the reported results come
from analysis/rerun/. numericbench_eval.py also supplies the item sampling used by
analysis/rerun/build_manifest.py.

Runs GPT-5.4, Claude Opus 4.6, and Gemini 3 (MINIMAL + HIGH) on sampled NumericBench tasks.

Sampling strategy (cost-aware):
  - arithmetic_operation: 500 items (sampled across digit levels and operations)
  - context_arithmetic_operation: 500 items
  - different_digit_arithmetic_operation: 500 items
  - num_list (100-length): all 1500 items (cheap, RG-critical)
  - mixed_number_string: all 2000 items (cheap)
  - sequence: all 500 items
  - stock single-turn: 300 items (expensive per item ~11K tokens)
  - weather single-turn: 300 items (expensive per item ~11K tokens)

Total: ~6,100 items × 4 conditions = ~24,400 API calls
"""

import json
import os
import random
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("numericbench_data")
OUTPUT_DIR = Path("data/contextual_v1/numericbench_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ─── Subset Configuration ────────────────────────────────────────────────────

SUBSETS = {
    "arithmetic": {
        "file": "arithmetic_operation/arithmetic_operation.json",
        "sample_n": 500,
        "grounding": "PG",
        "is_mcq": False,
    },
    "context_arithmetic": {
        "file": "arithmetic_operation/context_arithmetic_operation.json",
        "sample_n": 500,
        "grounding": "PG+RG",
        "is_mcq": False,
    },
    "different_digit": {
        "file": "arithmetic_operation/different_digit_arithmetic_operation.json",
        "sample_n": 500,
        "grounding": "PG",
        "is_mcq": False,
    },
    "num_list_100": {
        "file": "num_list/num_list_500_per_sample_100_length.json",
        "sample_n": None,  # all
        "grounding": "RG",
        "is_mcq": True,
    },
    "num_list_1000": {
        "file": "num_list/num_list_500_per_sample_1000_length.json",
        "sample_n": 300,  # expensive
        "grounding": "RG",
        "is_mcq": True,
    },
    "mixed_number_string": {
        "file": "mixed_number_string/mixed_number_string_500_per_sample.json",
        "sample_n": None,  # all
        "grounding": "RG",
        "is_mcq": False,
    },
    "sequence": {
        "file": "sequence/sequence_500_sample_100_length.json",
        "sample_n": None,  # all
        "grounding": "PG",
        "is_mcq": True,
    },
    "stock": {
        "file": "stock/single-turn/stock_500_per_sample_150_length.json",
        "sample_n": 300,
        "grounding": "RG",
        "is_mcq": True,
    },
    "weather": {
        "file": "weather/single-turn/weather_500_per_sample_200_length.json",
        "sample_n": 300,
        "grounding": "RG",
        "is_mcq": True,
    },
}


def load_and_sample(subset_name: str, config: dict) -> list[dict]:
    """Load a subset and optionally sample items."""
    filepath = DATA_DIR / config["file"]
    with open(filepath) as f:
        raw = json.load(f)

    system_prompt = raw.get("system_prompt", "")
    system_prompt_cot = raw.get("system_prompt_cot", "")
    description = raw.get("description", "")
    items = raw["data"]

    rng = random.Random(SEED)
    if config["sample_n"] is not None and config["sample_n"] < len(items):
        items = rng.sample(items, config["sample_n"])

    for item in items:
        item["_subset"] = subset_name
        item["_system_prompt"] = system_prompt
        item["_system_prompt_cot"] = system_prompt_cot
        item["_description"] = description
        item["_grounding"] = config["grounding"]
        item["_is_mcq"] = config["is_mcq"]

    return items


def format_prompt(item: dict) -> tuple[str, str]:
    """Format a NumericBench item into (system_message, user_message)."""
    system = item["_system_prompt"]
    description = item["_description"]

    struct_data = item.get("struct_data", "")
    if isinstance(struct_data, (list, dict)):
        struct_str = json.dumps(struct_data)
    else:
        struct_str = str(struct_data)

    question = item["question"]

    user_msg = ""
    if description:
        user_msg += f"{description}\n\n"
    if struct_str and struct_str != "{}":
        user_msg += f"Data:\n{struct_str}\n\n"
    user_msg += f"Question: {question}"

    return system, user_msg


def extract_answer(response: str, is_mcq: bool) -> str:
    """Extract the answer from model response."""
    if is_mcq:
        match = re.search(r'\b([A-H])\b', response.strip())
        if match:
            return match.group(1)
        return response.strip()[:1]
    else:
        lines = response.strip().split("\n")
        last_line = lines[-1].strip()
        match = re.search(r'-?\d+(?:\.\d+)?', last_line)
        if match:
            return match.group(0)
        return last_line


def evaluate_response(predicted: str, ground_truth: str, is_mcq: bool) -> bool:
    """Check if predicted answer matches ground truth."""
    pred = predicted.strip().upper()
    gt = ground_truth.strip().upper()

    if is_mcq:
        return pred == gt

    try:
        return abs(float(pred) - float(gt)) < 0.005
    except (ValueError, TypeError):
        return pred == gt


# ─── API Config ───────────────────────────────────────────────────────────────

try:
    import api_config as _cfg
except ImportError:
    raise RuntimeError(
        "api_config.py not found. Copy api_config.py and fill in your API keys."
    )


# ─── API Callers ──────────────────────────────────────────────────────────────

def _get_openai_client(api_key: str, base_url: str = ""):
    from openai import OpenAI
    kwargs = {"api_key": api_key, "timeout": 120.0}
    if base_url:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def call_gpt(system: str, user: str) -> str:
    """Call GPT via OpenAI-compatible proxy."""
    client = _get_openai_client(_cfg.OPENAI_API_KEY, _cfg.OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=_cfg.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


def call_claude(system: str, user: str) -> str:
    """Call Claude via Anthropic-compatible proxy."""
    import anthropic
    kwargs = {"api_key": _cfg.ANTHROPIC_API_KEY, "timeout": 120.0}
    if _cfg.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = _cfg.ANTHROPIC_BASE_URL
    client = anthropic.Anthropic(**kwargs)
    response = client.messages.create(
        model=_cfg.ANTHROPIC_MODEL,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0,
        max_tokens=256,
    )
    return response.content[0].text


def call_gemini(system: str, user: str, thinking_level: str = "none") -> str:
    """Call Gemini via OpenAI-compatible proxy.

    For MINIMAL: standard call, no thinking.
    For HIGH: pass thinking instructions in system prompt to encourage step-by-step.
    """
    client = _get_openai_client(_cfg.GOOGLE_API_KEY, _cfg.GOOGLE_BASE_URL)

    sys_msg = system
    if thinking_level == "high":
        sys_msg = (
            "Think step by step carefully before answering. "
            "Show your reasoning process.\n\n" + system
        )

    response = client.chat.completions.create(
        model=_cfg.GEMINI_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=1024 if thinking_level == "high" else 256,
    )
    return response.choices[0].message.content or ""


MODEL_CALLERS = {
    "gpt": lambda s, u: call_gpt(s, u),
    "claude": lambda s, u: call_claude(s, u),
    "gemini_minimal": lambda s, u: call_gemini(s, u, "none"),
    "gemini_high": lambda s, u: call_gemini(s, u, "high"),
}


# ─── Main Evaluation Loop ────────────────────────────────────────────────────

def run_evaluation(model_key: str, subsets_to_run: list[str] | None = None,
                   resume: bool = True, workers: int = 1):
    """Run evaluation for a single model on all subsets."""
    caller = MODEL_CALLERS[model_key]
    results_file = OUTPUT_DIR / f"{model_key}_results.jsonl"

    completed_ids = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add((r["subset"], r["idx"]))
        print(f"Resuming {model_key}: {len(completed_ids)} items already completed")

    subsets_to_process = subsets_to_run or list(SUBSETS.keys())

    write_lock = threading.Lock()
    total_items = 0
    total_correct = 0
    subset_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    def process_item(subset_name, config, item):
        system, user = format_prompt(item)
        ground_truth = str(item["answer"])
        is_mcq = config["is_mcq"]

        try:
            response = caller(system, user)
            predicted = extract_answer(response, is_mcq)
            correct = evaluate_response(predicted, ground_truth, is_mcq)
        except Exception as e:
            print(f"  ERROR on item {item['idx']}: {e}")
            response = f"ERROR: {e}"
            predicted = ""
            correct = False
            time.sleep(2)

        return {
            "subset": subset_name,
            "idx": item["idx"],
            "grounding": config["grounding"],
            "is_mcq": is_mcq,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "raw_response": response[:500],
            "ability": item.get("ability", ""),
        }

    for subset_name in subsets_to_process:
        config = SUBSETS[subset_name]
        items = load_and_sample(subset_name, config)
        pending = [(subset_name, config, item) for item in items
                    if (subset_name, item["idx"]) not in completed_ids]
        print(f"\n--- {subset_name}: {len(items)} items, {len(pending)} remaining [{config['grounding']}] ---")

        if workers <= 1:
            for i, (sn, cfg, item) in enumerate(pending):
                result = process_item(sn, cfg, item)
                with write_lock:
                    with open(results_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    total_items += 1
                    total_correct += int(result["correct"])
                    subset_stats[sn]["total"] += 1
                    subset_stats[sn]["correct"] += int(result["correct"])
                    if (i + 1) % 50 == 0:
                        acc = subset_stats[sn]["correct"] / subset_stats[sn]["total"]
                        print(f"  [{i+1}/{len(pending)}] {sn} running acc: {acc:.3f}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_item, sn, cfg, item): i
                           for i, (sn, cfg, item) in enumerate(pending)}
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    with write_lock:
                        with open(results_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        total_items += 1
                        total_correct += int(result["correct"])
                        sn = result["subset"]
                        subset_stats[sn]["total"] += 1
                        subset_stats[sn]["correct"] += int(result["correct"])
                        if total_items % 50 == 0:
                            acc = subset_stats[sn]["correct"] / max(subset_stats[sn]["total"], 1)
                            print(f"  [{total_items} done] {sn} running acc: {acc:.3f}")

        if subset_stats[subset_name]["total"] > 0:
            acc = subset_stats[subset_name]["correct"] / subset_stats[subset_name]["total"]
            print(f"  Final: {acc:.3f} ({subset_stats[subset_name]['correct']}/{subset_stats[subset_name]['total']})")

    if total_items > 0:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"Overall: {total_correct}/{total_items} = {total_correct/total_items:.3f}")
        for s, stats in subset_stats.items():
            if stats["total"] > 0:
                print(f"  {s}: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.3f}")


# ─── Results Analysis ────────────────────────────────────────────────────────

def analyze_results():
    """Analyze all model results and produce summary tables."""
    model_results = {}

    for model_key in MODEL_CALLERS:
        results_file = OUTPUT_DIR / f"{model_key}_results.jsonl"
        if not results_file.exists():
            continue

        results = []
        with open(results_file) as f:
            for line in f:
                results.append(json.loads(line))

        model_results[model_key] = results
        print(f"Loaded {model_key}: {len(results)} results")

    if not model_results:
        print("No results found.")
        return

    # Per-subset accuracy
    print(f"\n{'Subset':<25}", end="")
    for m in model_results:
        print(f"  {m:>15}", end="")
    print()
    print("-" * (25 + 17 * len(model_results)))

    all_subsets = sorted(set(r["subset"] for results in model_results.values() for r in results))
    subset_accs = defaultdict(dict)

    for subset in all_subsets:
        print(f"{subset:<25}", end="")
        for model_key, results in model_results.items():
            subset_results = [r for r in results if r["subset"] == subset]
            if subset_results:
                acc = sum(r["correct"] for r in subset_results) / len(subset_results)
                subset_accs[subset][model_key] = acc
                print(f"  {acc:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    # Per-grounding-type accuracy
    print(f"\n{'Grounding':<15}", end="")
    for m in model_results:
        print(f"  {m:>15}", end="")
    print()
    print("-" * (15 + 17 * len(model_results)))

    for grounding in ["RG", "PG", "PG+RG"]:
        print(f"{grounding:<15}", end="")
        for model_key, results in model_results.items():
            g_results = [r for r in results if r["grounding"] == grounding]
            if g_results:
                acc = sum(r["correct"] for r in g_results) / len(g_results)
                print(f"  {acc:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    # Per-ability accuracy (for arithmetic)
    print(f"\n{'Ability':<40}", end="")
    for m in model_results:
        print(f"  {m:>15}", end="")
    print()
    print("-" * (40 + 17 * len(model_results)))

    all_abilities = sorted(set(r.get("ability", "") for results in model_results.values()
                               for r in results if r.get("ability")))
    for ability in all_abilities:
        print(f"{ability:<40}", end="")
        for model_key, results in model_results.items():
            a_results = [r for r in results if r.get("ability") == ability]
            if a_results:
                acc = sum(r["correct"] for r in a_results) / len(a_results)
                print(f"  {acc:>14.3f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

    # Save summary JSON
    summary = {}
    for model_key, results in model_results.items():
        summary[model_key] = {
            "overall": sum(r["correct"] for r in results) / len(results),
            "by_subset": {},
            "by_grounding": {},
            "by_ability": {},
        }
        for subset in all_subsets:
            sr = [r for r in results if r["subset"] == subset]
            if sr:
                summary[model_key]["by_subset"][subset] = {
                    "accuracy": sum(r["correct"] for r in sr) / len(sr),
                    "n": len(sr),
                }
        for grounding in ["RG", "PG", "PG+RG"]:
            gr = [r for r in results if r["grounding"] == grounding]
            if gr:
                summary[model_key]["by_grounding"][grounding] = {
                    "accuracy": sum(r["correct"] for r in gr) / len(gr),
                    "n": len(gr),
                }
        for ability in all_abilities:
            ar = [r for r in results if r.get("ability") == ability]
            if ar:
                summary[model_key]["by_ability"][ability] = {
                    "accuracy": sum(r["correct"] for r in ar) / len(ar),
                    "n": len(ar),
                }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NumericBench Evaluation")
    parser.add_argument("--model", choices=list(MODEL_CALLERS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Subsets to evaluate (default: all)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results instead of running evaluation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample count without running")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent API calls (default: 8)")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    elif args.dry_run:
        print("Dry run — items per subset:")
        total = 0
        for name, config in SUBSETS.items():
            items = load_and_sample(name, config)
            total += len(items)
            print(f"  {name}: {len(items)} [{config['grounding']}]")
        print(f"  TOTAL: {total}")
    elif args.model:
        run_evaluation(args.model, args.subsets, workers=args.workers)
    else:
        print("Usage: python numericbench_eval.py --model gpt|claude|gemini_minimal|gemini_high")
        print("       python numericbench_eval.py --analyze")
        print("       python numericbench_eval.py --dry-run")
