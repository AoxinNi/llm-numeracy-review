"""Adversarial tests for analysis/scoring.py. Run: python -m pytest analysis/test_scoring.py"""

import pytest

from scoring import extract_nc, legacy_extract, numeric_equal, score_nc, extract_last_number, value_equal_nc


# ── Signs ────────────────────────────────────────────────────────────────────────────────

def test_negative_exponent_is_not_equal_to_positive():
    pred = extract_nc("The answer is 1.2e-3", "ScientificNotation")
    assert pred == "1.2e-3"
    assert score_nc(pred, "1.2e3", "ScientificNotation")["exact_match"] == 0


def test_legacy_scorer_had_the_exponent_sign_bug():
    pred = legacy_extract("The answer is 1.2e-3", "ScientificNotation")
    assert pred == "1.2e3"  # sign silently dropped: this is the bug being fixed


def test_explicit_plus_exponent_matches():
    pred = extract_nc("The answer is 1.2e+3", "ScientificNotation")
    assert score_nc(pred, "1.2e3", "ScientificNotation")["exact_match"] == 1


def test_negative_integer_kept_and_wrong_against_positive_gt():
    pred = extract_nc("The answer is -952", "Integer")
    assert pred == "-952"
    s = score_nc(pred, "952", "Integer")
    assert s["exact_match"] == 0 and s["sign_ok"] == 0


def test_negative_integer_correct_against_negative_gt():
    assert score_nc(extract_nc("The answer is -14", "Integer"), "-14", "Integer")["exact_match"] == 1


def test_missing_sign_is_wrong():
    assert score_nc(extract_nc("The answer is 14", "Integer"), "-14", "Integer")["exact_match"] == 0


# ── Marker handling ──────────────────────────────────────────────────────────────────────

def test_last_marker_is_used():
    text = "The answer is 1609.9448749340.\n\nWait, recalculating.\n\nThe answer is 1609.944874934"
    assert extract_nc(text, "Float") == "1609.944874934"
    assert extract_nc(text, "Float", marker="first") == "1609.9448749340"


def test_marker_is_case_insensitive():
    assert extract_nc("so the answer is 42", "Integer") == "42"


def test_bare_number_is_accepted():
    assert extract_nc("1014", "Integer") == "1014"
    assert extract_nc(" 42.892. ", "Float") == "42.892"
    assert extract_nc("1014 apples", "Integer") == ""


def test_no_marker_strict_is_empty_lenient_falls_back():
    assert extract_nc("I think it's 42.", "Integer") == ""
    assert extract_nc("I think it's 42.", "Integer", lenient=True) == "42"


# ── Formatting wrappers ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The answer is **952**", "The answer is $952$", "The answer is \\boxed{952}",
    "The answer is: 952.", "The answer is `952`",
])
def test_wrappers_are_removed(text):
    assert extract_nc(text, "Integer") == "952"


def test_commas_rejected_strict_accepted_lenient():
    assert extract_nc("The answer is 1,234", "Integer") == "1"
    assert extract_nc("The answer is 1,234", "Integer", lenient=True) == "1234"


def test_fraction_and_float_shapes():
    assert extract_nc("The answer is 2107937/357246", "Fraction") == "2107937/357246"
    assert extract_nc("The answer is 14", "Float") == ""
    assert extract_nc("The answer is 14", "Float", lenient=True) == "14"


# ── Refusals, errors, truncation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["", None, "ERROR: rate limit", "Let me compute 2107937 / 357"])
def test_unparsable_is_empty_and_scores_zero(text):
    pred = extract_nc(text or "", "Integer")
    assert pred == ""
    assert score_nc(pred, "952", "Integer")["exact_match"] == 0


# ── Digit match ──────────────────────────────────────────────────────────────────────────

def test_digit_match_reads_integers_from_the_right():
    assert score_nc("1952", "952", "Integer")["digit_match"] == 1.0
    assert score_nc("953", "952", "Integer")["digit_match"] == pytest.approx(2 / 3)


# ── Contextual benchmarks ────────────────────────────────────────────────────────────────

def test_numeric_equal_strict_and_legacy_tolerance():
    assert numeric_equal("14.38", "14.38")
    assert numeric_equal("14.380", "14.38")
    assert not numeric_equal("14.384", "14.38")
    assert numeric_equal("14.384", "14.38", tol=0.005)


def test_numeric_equal_handles_commas_and_signs():
    assert numeric_equal("1,314,228.21", "1314228.21")
    assert not numeric_equal("511647.55", "-511647.55")


def test_extract_last_number():
    assert extract_last_number("Total: 1,234 apples, so #### 1,234") == "1234"
    assert extract_last_number("no number") == ""


def test_value_equal_ignores_format_conventions():
    assert value_equal_nc("2.80e4", "2.8e4", "ScientificNotation", "to_scient_Integer_none_ScientificNotation") == 1
    assert value_equal_nc("07.977", "7.977", "Float", "digit_add_Float_Float_Float") == 1
    assert value_equal_nc("322/49", "46/7", "Fraction", "truediv_Integer_Integer_Fraction") == 1
    assert value_equal_nc("2.81e4", "2.8e4", "ScientificNotation", "to_scient_Integer_none_ScientificNotation") == 0


def test_value_equal_keeps_significant_figures_strict():
    assert value_equal_nc("4.00e1", "4.000e1", "ScientificNotation", "sig_Float_int_ScientificNotation") == 0


def test_numeric_equal_is_exact_for_large_values():
    assert not numeric_equal("11207337192.25", "11207337192.24")
    assert numeric_equal("11207337192.240", "11207337192.24")
