#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY barker --length 71,73,74,75 --time 14400
