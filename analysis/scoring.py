"""Answer extraction and scoring for the Section 5 re-analysis.

Replaces the scorer in ``text_model/metrics.py`` for the paper's numbers. Differences from the
legacy scorer (kept here as ``legacy_extract`` for the sensitivity analysis):

* Signs are kept. The legacy scorer removed every ``+``/``-`` from predictions and every
  non-digit from ground truths, so ``1.2e-3`` was scored equal to ``1.2e3``.
* The answer is taken after the *last* "The answer is" marker (the final answer), not the
  first one. ``marker="first"`` reproduces the legacy choice.
* A response consisting only of a correctly shaped number is accepted without the marker.
* Markdown/LaTeX wrappers (``**``, ``$``, ``\\boxed{}``) and trailing punctuation are removed
  before matching, so formatting alone does not turn a correct answer into a miss.
* Failed, missing, and unparsable responses are scored as incorrect by the callers
  (intention-to-evaluate); nothing is dropped from the denominator.
"""

from __future__ import annotations

import math
import re

NC_MARKER = "the answer is"

# Strict patterns follow the format each Number Cookbook prompt asks for, with an optional sign.
_STRICT = {
    "Integer": r"[+-]?\d+",
    "int": r"[+-]?\d+",
    "Float": r"[+-]?\d+\.\d+",
    "Fraction": r"[+-]?\d+/\d+",
    "ScientificNotation": r"[+-]?\d+\.\d+[eE][+-]?\d+",
}
# Lenient patterns also accept thousands separators and near-miss formats.
_LENIENT = {
    "Integer": r"[+-]?\d[\d,]*",
    "int": r"[+-]?\d[\d,]*",
    "Float": r"[+-]?\d[\d,]*(?:\.\d+)?",
    "Fraction": r"[+-]?\d[\d,]*\s*/\s*\d[\d,]*",
    "ScientificNotation": r"[+-]?\d+(?:\.\d+)?\s*[eE]\s*[+-]?\d+",
}

_WRAPPERS = re.compile(r"\\boxed\{|\\\(|\\\)|\\\[|\\\]|\$|\*\*|\*|`|\\text\{|\{|\}")


def _clean(text: str) -> str:
    return _WRAPPERS.sub("", text).strip().lstrip(":").strip()


def normalize_number(s: str) -> str:
    """Canonical string form: no spaces/commas, lower-case e, no '+' signs."""
    s = s.replace(",", "").replace(" ", "").replace("E", "e")
    s = s.replace("e+", "e")
    if s.startswith("+"):
        s = s[1:]
    return s


def extract_nc(text: str, domain: str, marker: str = "last", lenient: bool = False) -> str:
    """Extract a Number Cookbook answer. Returns '' when nothing valid is found."""
    if not text:
        return ""
    low = text.lower()
    idx = low.rfind(NC_MARKER) if marker == "last" else low.find(NC_MARKER)
    if idx < 0:
        # A bare answer (the whole response is one number of the requested shape) is accepted.
        bare = _clean(text).rstrip(".")
        if re.fullmatch(_STRICT[domain], bare):
            return normalize_number(bare)
        if not lenient:
            return ""
        # Fallback: last number of the expected shape anywhere in the text.
        found = re.findall(_LENIENT[domain], _clean(text))
        return normalize_number(found[-1]) if found else ""
    rest = _clean(text[idx + len(NC_MARKER):])
    pattern = (_LENIENT if lenient else _STRICT)[domain]
    m = re.match(pattern, rest)
    return normalize_number(m.group()) if m else ""


def legacy_extract(text: str, domain: str) -> str:
    """The original ``MetricsRecorder.retrieve_answer`` (first marker, signs stripped)."""
    marker = "The answer is"
    try:
        text = text[text.index(marker) + len(marker):]
    except ValueError:
        pass
    text = text.strip()
    pattern = {"Integer": r"\d+", "int": r"\d+", "Float": r"\d+\.\d+", "Fraction": r"\d+/\d+",
               "ScientificNotation": r"\d+\.\d+[eE][+-]?\d+"}[domain]
    m = re.match(pattern, text)
    if m is None:
        return ""
    return m.group().replace("+", "").replace("-", "").replace("E", "e")


def _parts(s: str, domain: str) -> list[str]:
    """Split into the digit parts used by Number Cookbook's digit-match metric."""
    s = s.lstrip("+-")
    try:
        if domain in ("Integer", "int"):
            return [s]
        if domain == "Float":
            a, b = s.split(".")
            return [a, b]
        if domain == "Fraction":
            a, b = s.split("/")
            return [a, b]
        if domain == "ScientificNotation":
            mant, exp = s.split("e")
            a, b = mant.split(".")
            return [a, b, exp.lstrip("+-")]
    except ValueError:
        pass
    return [""] * {"Integer": 1, "int": 1, "Float": 2, "Fraction": 2, "ScientificNotation": 3}[domain]


# Number Cookbook reads integer parts right-to-left and fractional parts left-to-right.
_READ_DIRECTION = {
    "Integer": ["right"], "int": ["right"], "Float": ["right", "left"],
    "Fraction": ["right", "right"], "ScientificNotation": ["right", "left", "right"],
}


def _signs(s: str, domain: str) -> tuple[bool, ...]:
    neg_mant = s.startswith("-")
    if domain == "ScientificNotation" and "e" in s:
        return (neg_mant, s.split("e", 1)[1].startswith("-"))
    return (neg_mant,)


def score_nc(pred: str, gt: str, domain: str) -> dict[str, float]:
    """Exact match (sign-aware) and digit match for one Number Cookbook item."""
    gt_n = normalize_number(gt)
    exact = int(bool(pred) and pred == gt_n)
    gp, pp = _parts(gt_n, domain), _parts(pred, domain)
    hits = 0
    for g, p, direction in zip(gp, pp, _READ_DIRECTION[domain]):
        if direction == "right":
            g, p = g[::-1], p[::-1]
        hits += sum(int(a == b) for a, b in zip(g, p))
    total = sum(len(g) for g in gp) or 1
    sign_ok = (not pred) or _signs(pred, domain) == _signs(gt_n, domain)
    return {"exact_match": exact, "digit_match": hits / total, "sign_ok": int(sign_ok)}


# ── Contextual benchmarks (NumericBench, GSM-Symbolic) ─────────────────────────────────────

_NUM = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def to_float(s: str) -> float | None:
    try:
        return float(normalize_number(s))
    except (ValueError, AttributeError):
        return None


def extract_last_number(text: str) -> str:
    found = _NUM.findall(_clean(text or ""))
    return normalize_number(found[-1]) if found else ""


def numeric_equal(pred: str, gt: str, tol: float | None = None) -> bool:
    """Exact decimal equality by default (no floating point: values in these benchmarks reach
    10^10, where a relative float tolerance would accept answers that are off by 0.01);
    ``tol`` reproduces the legacy absolute tolerances."""
    from decimal import Decimal, InvalidOperation
    p, g = to_float(pred), to_float(gt)
    if p is None or g is None:
        return normalize_number(pred or "").upper() == normalize_number(gt or "").upper() != ""
    if tol is not None:
        return abs(p - g) < tol
    try:
        return Decimal(normalize_number(pred)) == Decimal(normalize_number(gt))
    except InvalidOperation:
        return math.isclose(p, g, rel_tol=1e-12, abs_tol=0.0)


def value_equal_nc(pred: str, gt: str, domain: str, task: str) -> int:
    """Numerical equality ignoring formatting conventions (trailing zeros, leading zeros,
    unreduced fractions). Significant-figure tasks keep the strict rule, because there the
    number of written digits is the answer."""
    from decimal import Decimal, InvalidOperation
    from fractions import Fraction
    if not pred:
        return 0
    if task.startswith("sig_"):
        return int(pred == normalize_number(gt))
    try:
        if domain == "Fraction":
            a, b = pred.split("/")
            c, d = normalize_number(gt).split("/")
            return int(int(b) != 0 and Fraction(int(a), int(b)) == Fraction(int(c), int(d)))
        return int(Decimal(pred) == Decimal(normalize_number(gt)))
    except (ValueError, InvalidOperation, ZeroDivisionError):
        return 0
