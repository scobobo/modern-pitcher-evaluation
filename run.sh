#!/usr/bin/env bash
# Convenience wrapper: runs the analysis inside the project venv.
#   ./run.sh                          -> full 11-season four-seam run value study
#   ./run.sh --target is_whiff        -> same, predicting whiffs on swings
#   ./run.sh --pitch-type SL          -> sliders instead of four-seamers
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python run_analysis.py "$@"
