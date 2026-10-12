"""Run one configuration over the frozen contextual manifest, in parallel across API keys.

    python analysis/rerun/run_ctx.py <config> [--workers-per-key 6] [--limit N]

Configs: claude, gemini_minimal, gemini_high, gpt (model from api_config / --model).
Every response is stored in full (text, usage, finish reason, model version) in
analysis/rerun/out/<config>.jsonl. Already successful keys are skipped, so the script resumes.
Keys are never written to disk or logs; only the key index (1-3) is recorded.
"""
import argparse
import importlib.util
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cfg", os.path.join(HERE, "..", "..", "api_config.py"))
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

MAX_TOKENS = {"claude": 4096, "gpt": 4096, "gemini_minimal": 8192, "gemini_high": 32768}


def keys(prefix):
    """<PREFIX>_API_KEY plus optional <PREFIX>_API_KEY1..3 for parallel keys."""
    names = [f"{prefix}_API_KEY"] + [f"{prefix}_API_KEY{i}" for i in (1, 2, 3)]
    ks = [getattr(cfg, n, "") for n in names]
    return list(dict.fromkeys(k for k in ks if k))


def call_claude(item, key, model):
    import anthropic
    c = anthropic.Anthropic(api_key=key, base_url=cfg.ANTHROPIC_BASE_URL, timeout=600, max_retries=0)
    r = c.messages.create(model=model, max_tokens=MAX_TOKENS["claude"], temperature=0,
                          system=item["system"], messages=[{"role": "user", "content": item["user"]}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return {"text": text, "finish": r.stop_reason, "usage": r.usage.model_dump(), "model_version": r.model}


def call_gpt(item, key, model):
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url=cfg.OPENAI_BASE_URL.rstrip("/") + "/v1", timeout=600, max_retries=0)
    r = c.chat.completions.create(model=model, max_completion_tokens=MAX_TOKENS["gpt"], temperature=0,
                                  messages=[{"role": "system", "content": item["system"]},
                                            {"role": "user", "content": item["user"]}])
    return {"text": r.choices[0].message.content or "", "finish": r.choices[0].finish_reason,
            "usage": r.usage.model_dump(), "model_version": r.model}


def call_gemini(item, key, model, level):
    body = {"systemInstruction": {"parts": [{"text": item["system"]}]},
            "contents": [{"role": "user", "parts": [{"text": item["user"]}]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": level},
                                 "maxOutputTokens": MAX_TOKENS["gemini_high" if level == "HIGH" else "gemini_minimal"]}}
    url = cfg.GOOGLE_BASE_URL.rstrip("/") + f"/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    cand = (d.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return {"text": text, "finish": cand.get("finishReason"), "usage": d.get("usageMetadata"),
            "model_version": d.get("modelVersion")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["claude", "gpt", "gemini_minimal", "gemini_high"])
    ap.add_argument("--workers-per-key", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    conf = a.config
    if conf == "claude":
        ks, model, fn = keys("ANTHROPIC"), a.model or cfg.ANTHROPIC_MODEL, call_claude
    elif conf == "gpt":
        ks, model, fn = keys("OPENAI"), a.model or cfg.OPENAI_MODEL, call_gpt
    else:
        level = "HIGH" if conf == "gemini_high" else "MINIMAL"
        ks, model = keys("GOOGLE"), a.model or cfg.GEMINI_MODEL
        fn = lambda item, key, model: call_gemini(item, key, model, level)  # noqa: E731
    items = [json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"))]
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out_path = os.path.join(HERE, "out", f"{conf}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l)
            if r.get("status") == "ok":
                done.add(r["key"])
    todo = [it for it in items if it["key"] not in done]
    random.Random(0).shuffle(todo)  # spread subsets over time so partial runs stay balanced
    if a.limit:
        todo = todo[:a.limit]
    lock = threading.Lock()
    stats = {"ok": 0, "err": 0, "t0": time.time()}
    sems = [threading.Semaphore(a.workers_per_key) for _ in ks]
    print(f"[{conf}] model={model} keys={len(ks)} todo={len(todo)} done={len(done)}", flush=True)

    def work(i, item):
        ki = i % len(ks)
        with sems[ki]:
            delay = 5
            for attempt in range(8):
                t = time.time()
                try:
                    res = fn(item, ks[ki], model)
                    rec = {"key": item["key"], "status": "ok", "attempts": attempt + 1, "key_index": ki + 1,
                           "latency_s": round(time.time() - t, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "config": conf, "model": model, **res}
                    break
                except Exception as e:  # noqa: BLE001
                    msg = f"{type(e).__name__}: {str(e)[:300]}"
                    if isinstance(e, urllib.error.HTTPError):
                        try:
                            msg += " | " + e.read().decode()[:300]
                        except Exception:  # noqa: BLE001
                            pass
                    for k in ks:
                        msg = msg.replace(k, "<key>")
                    rec = {"key": item["key"], "status": "error", "error": msg, "attempts": attempt + 1,
                           "key_index": ki + 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": conf, "model": model}
                    time.sleep(delay + random.random() * delay)
                    delay = min(delay * 2, 120)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["ok" if rec["status"] == "ok" else "err"] += 1
            n = stats["ok"] + stats["err"]
            if n % 100 == 0 or n == len(todo):
                el = time.time() - stats["t0"]
                print(f"[{conf}] {n}/{len(todo)} ok={stats['ok']} err={stats['err']} "
                      f"elapsed={el/60:.1f}min rate={n/el*60:.1f}/min", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers_per_key * len(ks)) as ex:
        for f in as_completed([ex.submit(work, i, it) for i, it in enumerate(todo)]):
            f.result()
    print(f"[{conf}] finished ok={stats['ok']} err={stats['err']}", flush=True)


if __name__ == "__main__":
    main()
