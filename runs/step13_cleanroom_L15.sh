#!/bin/bash
# Clean-room exhaustive search, 8 binary codes, length 15 (independent C program),
# in 40 chunks. If it is interrupted, start it again: finished chunks are skipped.
cd "$(dirname "$0")/.." || exit 1
bash runs/t2search_chunked.sh 15 40
