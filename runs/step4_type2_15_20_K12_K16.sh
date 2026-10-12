#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY run type2 --length 13,15-20 --codes 12 --algo saw --time 7200
$PY run type2 --length 13,15-20 --codes 16 --algo saw --time 7200
