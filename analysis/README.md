# Section 5 re-analysis

Rebuilds every number, table, and data figure in Section 5 of the survey from the released API
outputs in `../data/number_cookbook/` (Number Cookbook) and `rerun/out/` (NumericBench and
GSM-Symbolic). Nothing in `../data/` is modified.

```sh
PYTHON=python3 bash analysis/run_all.sh     # from the repository root
```

| Step | Script | Output |
|---|---|---|
| Scorer tests | `test_scoring.py` | 28 adversarial cases (signs, exponents, markers, markup, refusals, exact equality) |
| Number Cookbook snapshot | `build_nc_snapshot.py` | `snapshot/nc_items.jsonl.gz`, `nc_coverage.json`, `MANIFEST.json` (SHA-256 of all inputs) |
| Number Cookbook analysis | `analyze_nc.py` | `out/nc_results.json` |
| Earlier contextual run (provenance only) | `analyze_ctx.py` | `out/ctx_results.json` |
| Contextual rerun | `analyze_rerun.py` | `out/ctx_rerun_results.json` |
| GSM-Symbolic error types | `error_types.py` | `out/error_types.json` (arithmetic slip / setup error / unclassified) |
| Paper assets | `make_paper_assets.py` | `paper/generated/*.tex` (one macro per reported number, tables), `paper/figures/generated/*.pdf` |
| Publication figures | `paper/generate_publication_figures.py` | `paper/figures/publication/*` (the figures in the paper) |

## Contextual rerun (`rerun/`)

- `build_manifest.py` freezes `rerun/manifest.jsonl` (8,900 requests): the original NumericBench
  items for all nine subsets (seed 42), the original 1,000 GSM-Symbolic main items, and 10
  instances per template for P1 (1,000) and P2 (500).
- `run_ctx.py <config>` runs one configuration (`gpt`, `claude`, `gemini_minimal`, `gemini_high`)
  over the manifest, in parallel across the keys in `api_config.py`, and resumes after interruption.
  GPT and Claude: temperature 0, 4,096 output tokens. Gemini: native `generateContent` endpoint
  with `thinkingConfig.thinkingLevel` MINIMAL (8,192 tokens) or HIGH (32,768 tokens). Every response
  is stored in full with usage, finish reason and returned model version; keys are never logged.
- `analyze_rerun.py` recomputes NumericBench arithmetic answers exactly: 24 of 1,500 reference
  answers carry floating-point rounding errors, so these items are scored against the exact value
  (half-up or half-even rounding).
  Original-label accuracy is kept as `acc_original_labels`.
- `error_types.py` recomputes every written step "a op b = c" in wrong GSM-Symbolic answers with
  exact fractions, skipping percentages, remainders, approximations and partial expressions.

The pipeline is deterministic: rebuilding from the released outputs reproduces the snapshot byte for
byte and the results hash `2730878f3071` used in the submitted manuscript.

## What changed relative to the earlier scripts

- **Signs.** `text_model/metrics.py` stripped `+`/`-` from predictions and non-digits from ground
  truths, so `1.2e-3` scored as `1.2e3`. Ground truths in this test set are never negative, so the
  bug only affected exponent signs.
- **Answer marker.** The last "The answer is" is used (the earlier scorer took the first). A bare
  number of the requested shape is accepted.
- **Denominators.** Every manifest item is scored for every configuration. The earlier Claude path
  skipped failed requests; the GPT path iterated `range(len(cache))`.
- **Gemini HIGH billing failures.** 4,969 Number Cookbook requests failed with "prepayment credits
  are depleted", almost all in six multiplication tasks. The primary analysis drops tasks where
  any configuration has < 90% successful responses and uses items answered by all four.
- **NumericBench multiple choice.** The earlier extraction fell back to the first character of the
  response, so Claude answers beginning "I need to..." were read as option "I". Raw responses were
  stored truncated to 500 characters, so these cells are reported as not scorable.
- **GSM-Symbolic.** Without a `####` line the earlier extraction took the first number on the last
  line (68 from "Total: $68 + $616 + $702 = $1,386") and did not handle thousands separators.
  Claude's main-split accuracy changes from 72.4% to 95.4%.
- **Gemini on contextual benchmarks** was run through an OpenAI-compatible proxy: "minimal" is a
  plain call and "high" adds a step-by-step instruction and a larger token limit. Most default
  Gemini GSM responses are cut off by the token limit; they are reported, not interpreted.
- **Matched items.** GSM-Symbolic P1 instances differ between GPT and the other models; only the
  733 shared items are compared. P2 (72 shared items) and NumericBench subsets not completed for
  every model are excluded from comparisons.

## Superseded

The earlier contextual run (`analyze_ctx.py`) is kept only for provenance: its stored responses
were truncated, Gemini was called through an OpenAI-compatible interface without native thinking,
and its GPT-5.4 counting results (99.8% / 98.7%) could not be reproduced by the rerun (9.9% / 65.3%).

## Not done (would need new API calls)

- Gemini HIGH on the six excluded Number Cookbook multiplication tasks (4,969 items).
- Claude on Number Cookbook items truncated at 256 output tokens, with a larger limit.
- Repeated runs to estimate API nondeterminism.
