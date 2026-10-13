#!/bin/bash
# Run the clean-room search (cleanroom/t2search) in chunks of tasks.
# If the run is interrupted, start it again: finished chunks are skipped, so
# at most one chunk is lost. At the end, cleanroom/merge_chunks.py runs the
# program's consistency checks over all chunks together.
# Usage: bash runs/t2search_chunked.sh N [chunks]
# Set OUT to write somewhere other than results/cleanroom (for tests), and
# THREADS to use fewer than 8 threads.
cd "$(dirname "$0")/.." || exit 1
N="$1"; K="${2:-40}"; OUT="${OUT:-results/cleanroom}"; TH="${THREADS:-8}"
[ -n "$N" ] || { echo "Usage: bash runs/t2search_chunked.sh N [chunks]"; exit 1; }
if [ ! -x cleanroom/t2search ] || [ cleanroom/t2search.c -nt cleanroom/t2search ]; then
    /usr/bin/clang -O3 -mcpu=native -pthread -o cleanroom/t2search cleanroom/t2search.c || exit 1
fi
dir="${OUT}/N${N}"; log="${OUT}/runlog${N}.txt"
mkdir -p "$dir"
P=$(cleanroom/t2search "$N" -n 2>&1 | sed -n 's/.* tasks=\([0-9]*\)$/\1/p')
[ -n "$P" ] || { echo "could not get the number of tasks"; exit 1; }
cleanroom/t2search "$N" -n >> "$log" 2>&1
echo "chunks=$K started $(date)" >> "$log"
# total search time of finished chunks, so the progress line stays correct after a restart
spent() { cat "$dir"/chunk_*.log 2>/dev/null | sed -n 's/^CHUNK.* search_time=\([0-9.]*\) s.*/\1/p' | awk '{s += $1} END {printf "%.1f", s}'; }
for ((i = 0; i < K; i++)); do
    first=$((P * i / K)); last=$((P * (i + 1) / K))
    clog="$dir/chunk_${i}.log"
    if grep -q "^wrote " "$clog" 2>/dev/null; then continue; fi
    cleanroom/t2search "$N" -t "$TH" -r "$first:$last" -o "$dir/chunk_${i}.txt" > "$clog" 2>&1 || { echo "chunk $i failed; see $clog" >> "$log"; exit 1; }
    done_tasks=0
    for ((j = 0; j < K; j++)); do
        grep -q "^wrote " "$dir/chunk_${j}.log" 2>/dev/null && done_tasks=$((done_tasks + P * (j + 1) / K - P * j / K))
    done
    echo "  [N=${N}] ${done_tasks}/${P} tasks, $(spent) s  (chunk $((i + 1)) of $K done $(date '+%a %H:%M'))" >> "$log"
done
python3 cleanroom/merge_chunks.py "$N" "$dir" "${OUT}/solutions_N${N}.txt" >> "$log" 2>&1 || exit 1
