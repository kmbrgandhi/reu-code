#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY run type2 --length 14 --codes 8 --algo memetic --metric psl --param ts_iters=0 --time 3600
