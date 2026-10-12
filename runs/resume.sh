#!/bin/bash
# Resume searches paused with runs/pause.sh.
pids=$(bash "$(dirname "$0")/searchpids.sh")
[ -z "$pids" ] && { echo "No paused search found (if the Mac restarted, the run is gone; start it again)."; exit 0; }
kill -CONT $pids && echo "Resumed: $(echo $pids | wc -w | tr -d ' ') process(es)."
