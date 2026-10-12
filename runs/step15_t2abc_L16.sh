#!/bin/bash
# Complete search at length 16, 8 codes, with the composition-split program
# (exhaustive/t2abc.c), in 40 chunks. Tables use 128 MB per thread.
cd "$(dirname "$0")/.." || exit 1
bash runs/t2abc_chunked.sh 16 128 40
