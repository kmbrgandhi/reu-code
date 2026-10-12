#!/bin/bash
# Print the process IDs of this repository's searches and their worker processes.
# It matches the programs themselves (not shells that mention them), so other
# projects' Python jobs and wrapper shells are left alone.
main=$( (pgrep -x t2search; pgrep -x t2abc;
         pgrep -f "^[^ ]*[Pp]ython[0-9.]* -m fastcodes (run|barker|exhaust-type2|vsearch)") 2>/dev/null)
for p in $main; do
    echo "$p"
    pgrep -P "$p"
done | sort -u
