#!/bin/bash
# Pause every running search (frees the CPU; nothing is lost). Resume with runs/resume.sh.
pids=$(pgrep -f "t2search [0-9]|m fastcodes (run|barker|exhaust-type2|vsearch)|multiprocessing.spawn")
[ -z "$pids" ] && { echo "No search is running."; exit 0; }
kill -STOP $pids && echo "Paused: $(echo $pids | wc -w | tr -d ' ') process(es). Resume with: bash runs/resume.sh"
