#!/usr/bin/env bash
# Rebuild every Section 5 number, table, and figure of the paper from the released API outputs.
# Usage (from the repository root): bash analysis/run_all.sh   [PYTHON=/path/to/python]
set -euo pipefail
PY="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

"$PY" -m pytest -q analysis/test_scoring.py
"$PY" analysis/build_nc_snapshot.py
"$PY" analysis/analyze_nc.py
"$PY" analysis/analyze_ctx.py
"$PY" analysis/analyze_rerun.py
"$PY" analysis/error_types.py
"$PY" analysis/make_paper_assets.py
"$PY" paper/generate_publication_figures.py
