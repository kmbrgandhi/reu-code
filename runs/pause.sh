#!/bin/bash
# Pause every running search (frees the CPU; nothing is lost). Resume with runs/resume.sh.
pids=$(bash "$(dirname "$0")/searchpids.sh")
[ -z "$pids" ] && { echo "No search is running."; exit 0; }
kill -STOP $pids && echo "Paused: $(echo $pids | wc -w | tr -d ' ') process(es). Resume with: bash runs/resume.sh"
