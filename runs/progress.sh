#!/bin/bash
# Show progress of the exhaustive searches at one length (both C programs).
# Usage: bash runs/progress.sh 15
cd "$(dirname "$0")/.." || exit 1
N="${1:-15}"
shown=0
for prog in t2abc t2search; do
    case $prog in
        t2abc)    LOG="results/t2abc/runlog${N}.txt";     SOL="results/t2abc/solutions_N${N}.txt";     name="composition-split search (exhaustive/t2abc)" ;;
        t2search) LOG="results/cleanroom/runlog${N}.txt"; SOL="results/cleanroom/solutions_N${N}.txt"; name="clean-room search (cleanroom/t2search)" ;;
    esac
    [ -f "$LOG" ] || continue
    shown=1
    echo "== $name"
    if grep -q "^RESULT N=${N}" "$LOG"; then
        grep "^RESULT N=${N}" "$LOG" | tail -1
        echo "Solutions: $(wc -l < "$SOL" | tr -d ' ') sets ($SOL)"
        continue
    fi
    pid=""
    for p in $(pgrep -x "$prog"); do
        ps -o args= -p "$p" | grep -q "^[^ ]*$prog ${N}\b" && pid=$p
    done
    if [ -z "$pid" ]; then state="NOT running (stopped or interrupted)"
    elif ps -o stat= -p "$pid" | grep -q T; then state="PAUSED (resume with: bash runs/resume.sh)"
    else state="running"; fi
    line=$(grep "tasks" "$LOG" | grep "elapsed\|tasks, " | tail -1)
    if [ -z "$line" ]; then echo "Length ${N}: ${state}; no progress line yet (setup takes a minute)."; continue; fi
    recent=$(grep "tasks" "$LOG" | grep "elapsed\|tasks, " | tail -30 | head -1)
    python3 - "$line" "$state" "$N" "$recent" <<'PY'
import re, sys, datetime
line, state, N, recent = sys.argv[1:5]
# t2search:  [N=15] 183512/262145 tasks, 897.9 s
# t2abc:     [N=15] 1234/74279 tasks (1.7%), 60 s elapsed, about 3500 s left, ...
pat = r'(\d+)/(\d+) tasks(?: \([\d.]+%\))?, ([\d.]+) s'
m = re.search(pat, line)
done, total, el = int(m.group(1)), int(m.group(2)), float(m.group(3))
# remaining time from the recent rate (last ~30 progress lines), so pauses don't distort it
m0 = re.search(pat, recent)
d0, e0 = int(m0.group(1)), float(m0.group(3))
rate = (done - d0) / (el - e0) if el > e0 and done > d0 else done / el
rem = (total - done) / rate
eta = datetime.datetime.now() + datetime.timedelta(seconds=rem)
print(f"Length {N}: {state}")
print(f"  {100 * done / total:5.1f}% done  ({done:,}/{total:,} tasks)")
print(f"  elapsed {el / 3600:5.2f} h, estimated remaining {rem / 3600:5.2f} h, finish around {eta:%a %H:%M}")
PY
done
[ "$shown" = 1 ] || { echo "No log yet for length ${N} (results/t2abc/runlog${N}.txt or results/cleanroom/runlog${N}.txt)."; exit 1; }
