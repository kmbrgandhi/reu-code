#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY vsearch --length 13-16 --codes 12 --time 1800
