# Numeracy in Large Language Models — evaluation code and data

Code, prompts, and complete model outputs for the empirical evaluation (Section 5) of

> Aoxin Ni. *Numeracy in Large Language Models: Fundamental Limitations and Paths to Improvement.*
> Preprint: [arXiv:2608.13129](https://arxiv.org/abs/2608.13129). The results in this repository
> correspond to the revised manuscript, which supersedes preprint version 1.

The evaluation covers GPT-5.4, Claude Opus 4.6, and Gemini 3 Flash (at the MINIMAL and HIGH
thinking levels) on three benchmarks:

| Benchmark | Items per configuration | What is released |
|---|---|---|
| [Number Cookbook](https://github.com/GraphPKU/number_cookbook) (44 task variants, half-size test split) | 119,425 | items, batch requests as submitted, every response |
| NumericBench (all nine subsets) | 6,400 | frozen request manifest, every full response |
| [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) (main, P1, P2) | 2,500 | frozen request manifest, every full response |

Every number, table, and data figure in Section 5 of the paper is generated from these files.

## Reproduce the paper's results

```sh
pip install -r requirements.txt
bash analysis/run_all.sh          # or: PYTHON=/path/to/python bash analysis/run_all.sh
```

This runs the scorer tests, rebuilds the item-level Number Cookbook snapshot, scores all
benchmarks, and writes:

- `analysis/snapshot/nc_items.jsonl.gz` — one scored row per (configuration, item);
- `analysis/out/*.json` — all results, bootstrap intervals, and sensitivity analyses;
- `paper/generated/results_macros.tex` — one LaTeX macro per number reported in the paper;
- `paper/generated/tab_*.tex` and `paper/figures/publication/*` — the tables and figures.

No API calls are made. The pipeline is deterministic: it reproduces the snapshot byte for byte and
the results hash `2730878f3071` printed by `make_paper_assets.py`. It takes a few minutes on a
laptop. All outputs are also committed, so they can be inspected without running anything.

## Repository layout

```
analysis/                 scoring, analysis, and asset generation (see analysis/README.md)
  scoring.py              answer extraction and exact-match scoring (+ test_scoring.py)
  build_nc_snapshot.py    Number Cookbook: score every item for every configuration
  analyze_nc.py           RG/PG analysis, bootstrap intervals, alternative codings, sensitivity
  analyze_rerun.py        NumericBench and GSM-Symbolic
  error_types.py          arithmetic slips vs. setup errors in wrong GSM-Symbolic answers
  analyze_ctx.py          first (superseded) contextual run, used only for a provenance note
  make_paper_assets.py    LaTeX macros, tables, and draft figures
  rerun/                  contextual benchmarks: manifest builder, API runner, manifest, outputs
data/
  number_cookbook/        items, batch requests, and normalized responses per configuration
  contextual_v1/          outputs of the first contextual run (superseded)
paper/                    figure script and generated tables/figures
numericbench_eval.py      scripts of the first contextual run; numericbench_eval.py also
gsm_symbolic_eval.py      defines the NumericBench item sampling (seed 42)
api_config.example.py     API configuration template (keys are read from the environment)
```

## Data

**Number Cookbook** (`data/number_cookbook/`). Items were generated with the benchmark's code
release. Requests were submitted through each provider's batch interface with default sampling
and one response per item; GPT-5.4 and Claude were limited to 256 output tokens.

- `items.jsonl.gz`: `custom_id`, `task`, `digit` (operand length), `data` (user prompt),
  `groundtruth`.
- `requests/<config>.jsonl.gz`: the batch request files exactly as submitted (system prompt, user
  prompt, model, token limit, and Gemini thinking level).
- `responses/<config>.jsonl.gz`: one record per item: `id`, `status` (`ok` or `api_error`),
  `text` (visible answer text; Gemini thought summaries are not included), `truncated`, token
  counts (`in_tok`, `think_tok`, `out_tok`), `model_version`, `created`.

The raw provider outputs (2.4 GB, mostly Gemini response metadata) were normalized into these files
with the parsers in `analysis/build_nc_snapshot.py`. For Gemini HIGH, 4,969 requests failed with a
billing error, mostly in six multiplication tasks; the primary analysis therefore uses the 38 tasks
answered by every configuration.

**NumericBench and GSM-Symbolic** (`analysis/rerun/`). `manifest.jsonl` (SHA-256 in
`manifest.sha256`) holds all 8,900 requests with system prompt, user prompt, and reference answer.
`out/<config>.jsonl` holds every response in full, with token usage, finish reason, returned model
version, and latency. GPT and Claude were called with temperature 0 and 4,096 output tokens;
Gemini through the native `generateContent` endpoint with `thinkingLevel` MINIMAL or HIGH.
24 of the 1,500 NumericBench arithmetic reference answers contain floating-point rounding errors,
so the three arithmetic subsets are scored against the exactly recomputed value (either exact
rounding is accepted); accuracy against the original labels is reported separately.

## Running new evaluations

```sh
cp api_config.example.py api_config.py      # keys are read from environment variables
python analysis/rerun/run_ctx.py claude     # or gpt, gemini_minimal, gemini_high
```

`run_ctx.py` resumes after interruption and can spread requests over several keys
(`<PROVIDER>_API_KEY`, optionally `<PROVIDER>_API_KEY1..3` in `api_config.py`). Keys are never
written to outputs or logs. Rebuilding the manifest (`analysis/rerun/build_manifest.py`) requires the
NumericBench data in `numericbench_data/` and the GSM-Symbolic dataset from Hugging Face.

## Licence

The code is released under the MIT License. Benchmark items (Number Cookbook, NumericBench,
GSM-Symbolic) remain under the licences of their original releases; please cite those benchmarks
when using the corresponding files. Model outputs are provided for research and reproducibility.

## Citation

```bibtex
@article{ni2026numeracy,
  title   = {Numeracy in Large Language Models: Fundamental Limitations and Paths to Improvement},
  author  = {Ni, Aoxin},
  journal = {arXiv preprint arXiv:2608.13129},
  year    = {2026}
}
```
