#!/bin/bash
# Resume searches paused with runs/pause.sh.
pids=$(pgrep -f "t2search [0-9]|m fastcodes (run|barker|exhaust-type2|vsearch)|multiprocessing.spawn")
[ -z "$pids" ] && { echo "No paused search found (if the Mac restarted, the run is gone; start it again)."; exit 0; }
kill -CONT $pids && echo "Resumed: $(echo $pids | wc -w | tr -d ' ') process(es)."
