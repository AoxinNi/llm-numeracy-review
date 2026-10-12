"""Build the frozen item-level Number Cookbook snapshot from the released API outputs.

Every item in the shared manifest is scored for every model (intention-to-evaluate): missing,
failed, truncated, or unparsable responses count as incorrect. Output:

    snapshot/nc_items.jsonl.gz   one row per (model, item)
    snapshot/nc_coverage.json    attempted / succeeded / truncated / parsed counts per model
    snapshot/MANIFEST.json       SHA-256 of every input file, model version strings, run dates

Inputs (data/number_cookbook/): items.jsonl.gz and one normalized response file per configuration
(responses/<config>.jsonl.gz, exported from the providers' raw batch outputs with the parsers below).

Run from the repository root:  python analysis/build_nc_snapshot.py
"""

from __future__ import annotations

import collections
import datetime as dt
import glob
import gzip
import hashlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scoring import extract_nc, legacy_extract, score_nc, value_equal_nc  # noqa: E402

BENCH = os.path.join("data", "number_cookbook")
OUT = os.path.join(os.path.dirname(__file__), "snapshot")

MODELS = {
    "gpt": {"responses": "gpt_responses", "format": "openai", "max_tokens": 256},
    "claude": {"responses": "claude_responses", "format": "anthropic", "max_tokens": 256},
    "gemini_minimal": {"responses": "gemini_lowthinklevel_responses", "format": "gemini", "max_tokens": None},
    "gemini_high": {"responses": "gemini_highthinklevel_responses", "format": "gemini", "max_tokens": None},
}
_ID = re.compile(r"request-(\d+)$")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[int, dict]:
    items = {}
    with gzip.open(f"{BENCH}/items.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            cid = int(_ID.match(d["custom_id"]).group(1))
            items[cid] = {"task": d["task"], "digit": int(d["digit"]), "gt": d["groundtruth"]}
    return items


# The parsers below document how the raw provider outputs (OpenAI / Anthropic / Gemini batch
# formats) were normalized into data/number_cookbook/responses/. They are not needed to rebuild
# the snapshot from the released files.
def parse_openai(d: dict) -> dict:
    resp = d.get("response") or {}
    body = resp.get("body") or {}
    ok = d.get("error") is None and resp.get("status_code") == 200 and body.get("choices")
    if not ok:
        return {"status": "api_error"}
    ch = body["choices"][0]
    usage = body.get("usage", {})
    think = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return {
        "status": "ok", "text": ch["message"].get("content") or "",
        "truncated": ch.get("finish_reason") == "length",
        "in_tok": usage.get("prompt_tokens", 0), "think_tok": think,
        "out_tok": usage.get("completion_tokens", 0) - think,
        "model_version": body.get("model"), "created": body.get("created"),
    }


def parse_anthropic(d: dict, max_tokens: int) -> dict:
    res = d.get("result") or {}
    if res.get("type") != "succeeded":
        return {"status": "api_error"}
    msg = res.get("message", {})
    content = msg.get("content") or []
    text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
    usage = msg.get("usage", {})
    stop = msg.get("stop_reason")
    out = usage.get("output_tokens", 0)
    return {
        "status": "ok", "text": text,
        "truncated": stop == "max_tokens" if stop else out >= max_tokens,
        "in_tok": usage.get("input_tokens", 0), "think_tok": 0, "out_tok": out,
        "model_version": msg.get("model"), "created": None,
    }


def parse_gemini(d: dict) -> dict:
    resp = d.get("response") or {}
    cands = resp.get("candidates") or []
    if not cands:
        return {"status": "api_error"}
    c = cands[0]
    parts = (c.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = resp.get("usageMetadata", {})
    return {
        "status": "ok", "text": text, "truncated": c.get("finishReason") == "MAX_TOKENS",
        "in_tok": u.get("promptTokenCount", 0), "think_tok": u.get("thoughtsTokenCount", 0),
        "out_tok": u.get("candidatesTokenCount", 0),
        "model_version": resp.get("modelVersion"), "created": None,
    }


def load_responses(model: str, cfg: dict, manifest_files: dict) -> tuple[dict[int, dict], collections.Counter]:
    """Return the normalized response per item and the line counts of the raw files."""
    path = f"{BENCH}/responses/{model}.jsonl.gz"
    manifest_files[path] = sha256(path)
    chosen = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            chosen[d.pop("id")] = d
    with open(f"{BENCH}/responses/raw_stats.json") as f:
        stats = collections.Counter(json.load(f)[model])
    return chosen, stats


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest_files: dict[str, str] = {}
    manifest_files[f"{BENCH}/items.jsonl.gz"] = sha256(f"{BENCH}/items.jsonl.gz")
    items = load_manifest()
    print(f"manifest: {len(items)} items")
    coverage, versions = {}, {}
    # mtime=0 keeps the gzip byte-identical across rebuilds, so its checksum is meaningful.
    gz = gzip.GzipFile(os.path.join(OUT, "nc_items.jsonl.gz"), "wb", mtime=0)
    with io.TextIOWrapper(gz, encoding="utf-8") as wf:
        for model, cfg in MODELS.items():
            resp, stats = load_responses(model, cfg, manifest_files)
            cov = collections.Counter(attempted=len(items))
            vers, created = collections.Counter(), []
            for cid, it in sorted(items.items()):
                domain = it["task"].split("_")[-1]
                r = resp.get(cid, {"status": "missing"})
                text = r.get("text", "") if r["status"] == "ok" else ""
                pred = extract_nc(text, domain)
                s = score_nc(pred, it["gt"], domain)
                pred_first = extract_nc(text, domain, marker="first")
                pred_len = extract_nc(text, domain, lenient=True)
                pred_leg = legacy_extract(text, domain)
                legacy_gt = "".join(c for c in it["gt"] if c.isdigit() or c in "./e")
                cov[r["status"]] += 1
                cov["truncated"] += int(bool(r.get("truncated")))
                cov["parsed"] += int(bool(pred))
                if r.get("model_version"):
                    vers[r["model_version"]] += 1
                if r.get("created"):
                    created.append(r["created"])
                wf.write(json.dumps({
                    "model": model, "id": cid, "task": it["task"], "digit": it["digit"],
                    "status": r["status"], "truncated": int(bool(r.get("truncated"))),
                    "in_tok": r.get("in_tok", 0), "think_tok": r.get("think_tok", 0),
                    "out_tok": r.get("out_tok", 0),
                    "pred": pred, "exact": s["exact_match"], "digit_match": round(s["digit_match"], 6),
                    "sign_ok": s["sign_ok"],
                    "exact_first": score_nc(pred_first, it["gt"], domain)["exact_match"],
                    "exact_lenient": score_nc(pred_len, it["gt"], domain)["exact_match"],
                    "exact_legacy": int(bool(pred_leg) and pred_leg == legacy_gt),
                    "exact_value": value_equal_nc(pred, it["gt"], domain, it["task"]),
                    "text_sha1": hashlib.sha1(text.encode()).hexdigest() if text else "",
                }) + "\n")
            cov.update({f"raw_{k}": v for k, v in stats.items()})
            coverage[model] = dict(cov)
            versions[model] = {
                "model_versions": dict(vers),
                "first_created_utc": dt.datetime.utcfromtimestamp(min(created)).isoformat() if created else None,
                "last_created_utc": dt.datetime.utcfromtimestamp(max(created)).isoformat() if created else None,
            }
            print(model, dict(cov))
    with open(os.path.join(OUT, "nc_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump({
            "built_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "benchmark": "Number Cookbook (Yang et al., 2025) test split, 44 tasks",
            "n_items": len(items),
            "configs": {m: {k: v for k, v in c.items()} for m, c in MODELS.items()},
            "versions": versions,
            "input_sha256": manifest_files,
            "output_sha256": {"nc_items.jsonl.gz": sha256(os.path.join(OUT, "nc_items.jsonl.gz"))},
        }, f, indent=2)


if __name__ == "__main__":
    main()
