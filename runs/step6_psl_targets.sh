#!/bin/bash
# Run from anywhere; always works in the repository root.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python -u -m fastcodes"

$PY run psl --length 251,135,164,196,234,272 --algo saw --time 7200 --beat-record
