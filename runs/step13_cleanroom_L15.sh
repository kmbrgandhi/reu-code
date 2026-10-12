#!/bin/bash
# Clean-room exhaustive search, 8 binary codes, length 15 (independent C program).
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/cleanroom
cleanroom/t2search 15 -o results/cleanroom/solutions_N15.txt > results/cleanroom/runlog15.txt 2>&1
