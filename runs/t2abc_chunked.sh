#!/bin/bash
# Run the composition-split search (exhaustive/t2abc) in chunks of tasks.
# If the run is interrupted, start it again: finished chunks are skipped, so
# at most one chunk is lost.
# Usage: bash runs/t2abc_chunked.sh N [table MB per thread] [chunks]
cd "$(dirname "$0")/.." || exit 1
N="$1"; MB="${2:-32}"; K="${3:-20}"
[ -n "$N" ] || { echo "Usage: bash runs/t2abc_chunked.sh N [MB] [chunks]"; exit 1; }
if [ ! -x exhaustive/t2abc ] || [ exhaustive/t2abc.c -nt exhaustive/t2abc ]; then
    /usr/bin/clang -O3 -mcpu=native -pthread -o exhaustive/t2abc exhaustive/t2abc.c || exit 1
fi
dir="results/t2abc/N${N}"; log="results/t2abc/runlog${N}.txt"
mkdir -p "$dir"
P=$(exhaustive/t2abc "$N" -m "$MB" -n 2>&1 | sed -n 's/.* prime=\([0-9]*\) .*/\1/p')
[ -n "$P" ] || { echo "could not get the number of tasks"; exit 1; }
exhaustive/t2abc "$N" -m "$MB" -n >> "$log" 2>&1
echo "chunks=$K table=${MB}MB started $(date)" >> "$log"
# total search time of finished chunks, so the progress line stays correct after a restart
spent() { cat "$dir"/chunk_*.log 2>/dev/null | sed -n 's/^RESULT.* search_time=\([0-9.]*\) s.*/\1/p' | awk '{s += $1} END {printf "%.1f", s}'; }
for ((i = 0; i < K; i++)); do
    first=$((P * i / K)); last=$((P * (i + 1) / K))
    clog="$dir/chunk_${i}.log"
    if grep -q "^RESULT" "$clog" 2>/dev/null; then continue; fi
    exhaustive/t2abc "$N" -m "$MB" -r "$first:$last" -o "$dir/chunk_${i}.txt" > "$clog" 2>&1 || { echo "chunk $i failed; see $clog" >> "$log"; exit 1; }
    done_tasks=0
    for ((j = 0; j < K; j++)); do
        grep -q "^RESULT" "$dir/chunk_${j}.log" 2>/dev/null && done_tasks=$((done_tasks + P * (j + 1) / K - P * j / K))
    done
    echo "  [N=${N}] ${done_tasks}/${P} tasks, $(spent) s  (chunk $((i + 1)) of $K done $(date '+%a %H:%M'))" >> "$log"
done
# merge the chunk files into one sorted solution file
sort "$dir"/chunk_*.txt > "results/t2abc/solutions_N${N}.txt"
count=$(wc -l < "results/t2abc/solutions_N${N}.txt" | tr -d ' ')
echo "RESULT N=${N} count=${count} search_time=$(spent) s chunks=${K} (merged from $dir)" >> "$log"
