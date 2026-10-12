#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY run psl --length 106-300 --algo saw --time 900 --beat-record
