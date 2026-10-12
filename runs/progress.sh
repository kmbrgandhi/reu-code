#!/bin/bash
# Show progress of the clean-room exhaustive search.
# Usage: bash runs/progress.sh 15
cd "$(dirname "$0")/.." || exit 1
N="${1:-15}"
LOG="results/cleanroom/runlog${N}.txt"
[ -f "$LOG" ] || { echo "No log yet: $LOG"; exit 1; }
if grep -q "^RESULT N=${N}" "$LOG"; then
    grep "^RESULT N=${N}" "$LOG"
    echo "Solutions: $(wc -l < results/cleanroom/solutions_N${N}.txt | tr -d ' ') sets (results/cleanroom/solutions_N${N}.txt)"
    exit 0
fi
pgrep -f "t2search ${N}" >/dev/null && state="running" || state="NOT running (stopped or interrupted)"
ps -o stat= -p $(pgrep -f "t2search ${N}" | head -1) 2>/dev/null | grep -q T && state="PAUSED (resume with: bash runs/resume.sh)"
line=$(grep "tasks," "$LOG" | tail -1)
if [ -z "$line" ]; then echo "Length ${N}: ${state}; no progress line yet (setup takes a minute)."; exit 0; fi
recent=$(grep "tasks," "$LOG" | tail -30 | head -1)
# line looks like:  [N=15] 183512/262145 tasks, 897.9 s
python3 - "$line" "$state" "$N" "$recent" <<'PY'
import re, sys, datetime
line, state, N, recent = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pat = r'(\d+)/(\d+) tasks, ([\d.]+) s'
m = re.search(pat, line)
done, total, el = int(m.group(1)), int(m.group(2)), float(m.group(3))
frac = done / total
# remaining time from the recent rate (last ~30 progress lines), so pauses don't distort it
m0 = re.search(pat, recent)
d0, e0 = int(m0.group(1)), float(m0.group(3))
rate = (done - d0) / (el - e0) if el > e0 and done > d0 else done / el
rem = (total - done) / rate
eta = datetime.datetime.now() + datetime.timedelta(seconds=rem)
print(f"Length {N}: {state}")
print(f"  {100 * frac:5.1f}% done  ({done:,}/{total:,} tasks)")
print(f"  elapsed {el / 3600:5.2f} h, estimated remaining {rem / 3600:5.2f} h, finish around {eta:%a %H:%M}")
PY
