#!/bin/bash
# Start a runbook step in the background at normal priority.
# Usage:  bash runs/start.sh step3
#
# Why this script: zsh lowers the priority of jobs you start with "&" (its
# BG_NICE option), and macOS then gives them only leftover CPU time. Starting
# the job from bash avoids that. The script also refuses to start a second
# search while one is running (set FORCE=1 to override).
cd "$(dirname "$0")/.." || exit 1
step="$1"
script=$(ls runs/"${step}"_*.sh 2>/dev/null | head -1)
if [ -z "$step" ] || [ -z "$script" ]; then
    echo "Usage: bash runs/start.sh <step>. Available steps:"
    ls runs/step*_*.sh | sed 's|runs/||; s|_.*||' | sort -u | sed 's/^/  /'
    exit 1
fi
# A search that was just stopped (pkill) can take a few seconds to exit.
for i in 1 2 3 4 5 6 7 8 9 10; do
    pgrep -f "m fastcodes (run|barker|exhaust-type2|vsearch)|t2search [0-9]" >/dev/null || break
    sleep 1
done
if [ -z "$FORCE" ] && pgrep -f "m fastcodes (run|barker|exhaust-type2|vsearch)|t2search [0-9]" >/dev/null; then
    echo "A search is already running. Wait for it to finish, or stop it with:"
    echo '  pkill -f "m fastcodes"'
    exit 1
fi
mkdir -p results
nohup caffeinate -i bash "$script" > "results/$step.log" 2>&1 &
echo "Started $script at normal priority (process $!)."
echo "Log: results/$step.log   Check: pgrep -fl \"m fastcodes\""
