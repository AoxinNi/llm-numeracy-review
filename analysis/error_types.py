"""Classify wrong GSM-Symbolic answers from the rerun into arithmetic slips versus setup errors.

For every incorrect, non-truncated response we extract the arithmetic the model wrote down
("a op b ... = c") and recompute it exactly.
  * arithmetic slip  -- at least one written step is computed incorrectly (a PG failure);
  * setup error      -- every written step is correct, yet the final answer is wrong, so the model
                        operated on the wrong quantities or relations (operand selection /
                        problem modelling, outside NGF);
  * unclassified     -- no checkable step was found.
Output: analysis/out/error_types.json
"""
import json
import os
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
from analyze_rerun import CONFIGS, TRUNC, load, score_item  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = r"\d[\d,]*(?:\.\d+)?"
OP = r"(?:\+|-|−|\*|×|x|÷|/|\\times|\\cdot|\\div)"
STEP = re.compile(rf"((?:\(?\$?{NUM}\)?\s*{OP}\s*)+\(?\$?{NUM}\)?)\s*=\s*\$?(-?{NUM})")


def _val(tok):
    return Fraction(tok.replace(",", ""))


def evaluate(expr):
    e = (expr.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*")
         .replace("÷", "/").replace("−", "-").replace("$", ""))
    e = re.sub(r"(?<=\d)\s*x\s*(?=\d|\()", "*", e)
    toks = re.findall(rf"{NUM}|[+\-*/()]", e)
    if not toks or any(t not in "+-*/()" and not re.fullmatch(NUM, t) for t in toks):
        return None
    # shunting-yard with exact fractions
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    out, ops = [], []
    for t in toks:
        if t in prec:
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[t]:
                out.append(ops.pop())
            ops.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if not ops:
                return None
            ops.pop()
        else:
            out.append(_val(t))
    while ops:
        if ops[-1] == "(":
            return None
        out.append(ops.pop())
    st = []
    try:
        for t in out:
            if isinstance(t, Fraction):
                st.append(t)
            else:
                b, a = st.pop(), st.pop()
                st.append(a + b if t == "+" else a - b if t == "-" else a * b if t == "*" else a / b)
    except (IndexError, ZeroDivisionError):
        return None
    return st[0] if len(st) == 1 else None


def step_verdicts(text):
    """Yield (step_text, correct) for every checkable written step."""
    for m in STEP.finditer(text):
        after = text[m.end():m.end() + 30].lower()
        before = text[max(0, m.start() - 12):m.start()]
        # skip percentages, remainders, approximations, and partial matches inside longer
        # expressions (implicit multiplication before, or the expression continuing after "=")
        if "remainder" in after or after.lstrip().startswith(("%", "\\%")) or "approx" in before or "≈" in before:
            continue
        prev = before.rstrip()[-1:] if before.rstrip() else ""
        if prev and (prev.isalnum() or prev in "()*/×÷+-−^\\{"):
            continue
        if re.match(r"\s*(?:\+|-|−|\*|×|÷|/|\(|\\times|\\cdot|\\div|\\frac)", text[m.end():]):
            continue
        nxt = text[m.end():m.end() + 3]
        if nxt[:1].isalpha() or nxt.startswith(("...", "…")):
            continue  # coefficient of a variable (19n) or an approximation (0.428...)
        v = evaluate(m.group(1))
        if v is None:
            continue
        stated = _val(m.group(2).replace("−", "-"))
        dec = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        # a stated value that rounds or truncates the exact result to the written precision is
        # not a slip (e.g. 145/6 = 24.16, or 190/7 = 27 people)
        unit = Fraction(1, 10 ** dec)
        yield m.group(0), abs(v - stated) < unit


def check_steps(text):
    v = [ok for _, ok in step_verdicts(text)]
    return sum(v), len(v) - sum(v)


def main():
    man, res = load()
    gsm = [it for it in man if it["bench"] == "gsm_symbolic"]
    out = {}
    for c in CONFIGS:
        counts = {"wrong": 0, "truncated_or_failed": 0, "arithmetic_slip": 0, "setup_error": 0, "unclassified": 0}
        examples = {"arithmetic_slip": [], "setup_error": []}
        for it in gsm:
            r = res[c].get(it["key"])
            s = score_item(it, r)
            if s["correct"]:
                continue
            counts["wrong"] += 1
            if s["status"] != "ok" or r.get("finish") in TRUNC:
                counts["truncated_or_failed"] += 1
                continue
            ok, bad = check_steps(r.get("text") or "")
            kind = "arithmetic_slip" if bad else ("setup_error" if ok else "unclassified")
            counts[kind] += 1
            if kind in examples and len(examples[kind]) < 3:
                examples[kind].append(it["key"])
        out[c] = {"counts": counts, "examples": examples}
        print(c, counts)
    with open(os.path.join(HERE, "out", "error_types.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
