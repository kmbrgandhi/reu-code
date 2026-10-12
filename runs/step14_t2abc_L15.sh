#!/bin/bash
# Complete search at length 15, 8 codes, with the composition-split program
# (exhaustive/t2abc.c), in 20 chunks.
cd "$(dirname "$0")/.." || exit 1
bash runs/t2abc_chunked.sh 15 32 20
