# Runbook: new type-2 alternating codes and PSL records

Each step is a script in `runs/`. Start a step with `bash runs/start.sh stepN`.
The launcher runs the step in the background at normal priority, keeps the Mac
awake, writes the log to `results/stepN.log`, and refuses to start a second
search while one is running.

Don't start steps with `nohup ... &` typed directly in zsh. zsh lowers the
priority of background jobs, so macOS gives them only leftover CPU time.

## Before you start

- Run every command from the repository:

  ```sh
  cd ~/Documents/Personal/CodingProjects/reu-code
  ```

- Keep the Mac plugged in, with the lid open. `caffeinate -i` stops idle
  sleep, but closing the lid can still put the Mac to sleep.
- Run one step at a time. Each step uses all 8 cores.
- Every result is also appended to `results/runs.jsonl`, so you lose nothing if
  a step stops early.

### Commands you use between steps

| To do this | Run this |
|---|---|
| Check whether a step is still running | `pgrep -fl "m fastcodes"` (no output means it has finished) |
| Read a step's log | `cat results/stepN.log` |
| Watch a log as it grows (press Ctrl+C to stop watching; the step keeps running) | `tail -f results/stepN.log` |
| See the best result so far for every problem | `.venv/bin/python -m fastcodes report` |
| Stop the current step | `pkill -f "m fastcodes"` |

A log gets one new block each time a length (or search setting) finishes, so
it stays empty for a while at the start.

While a step runs, every worker also saves its best result so far to
`results/checkpoints/` every 30 seconds. If a run is interrupted (for example,
the battery runs out), those files remain, and `report` includes them, marked
"from a checkpoint". When a length finishes normally, its checkpoint files are
deleted.

## Step 0: Check the setup (about 2 minutes)

```sh
mkdir -p results
.venv/bin/python -m fastcodes warmup
.venv/bin/python -m pytest fastcodes/tests -q
```

The last line should say `57 passed`.

## Step 1: Type-2 calibration at length 14 (about 4 hours)

A published type-2 set exists at length 14 (Markkanen et al. 2008). This step
tries four search settings, each for up to an hour, and each stops early if it
finds a set.

```sh
bash runs/start.sh step1
```

If you started step 1 with the older multi-line command and the memetic line
broke, run the memetic setting on its own after step 1 finishes:

```sh
bash runs/start.sh step1b
```

**When it finishes:** run `cat results/step1.log` (or `results/step1_type2_L14.log`
if you used the older command). A line with `peak 0 (target reached)` means
that setting found a type-2 set.

**Result (2026-10-10):** the self-avoiding walk with the default metric (`isl`)
found a length-14 set in 38 minutes. `--metric psl` and tabu search reached
only peak 2 in 1 hour. Steps 2–4 therefore use `saw` with the default metric.

## Step 2: Type-2 at length 13 (about 2 hours)

Length 13 is not in Sulzer's (1993) table (lengths 8–12) or in Markkanen et al.
(2008) (length 14). If you find a set, check the literature again before you
call it new.

```sh
bash runs/start.sh step2
```

## Step 3: Type-2 lengths 15–20 with 8 codes (about 12 hours)

This is the open problem: no type-2 set longer than 14 was known to Markkanen et
al. (2008). Each length runs for up to 2 hours and stops early when it finds a
set.

```sh
bash runs/start.sh step3
```

## Step 4: Type-2 lengths 13 and 15–20 with 12 and 16 codes (about 28 hours)

The number of codes must be a multiple of 4 (Sulzer 1993). More codes give the
search more freedom.

```sh
bash runs/start.sh step4
```

## Step 5: PSL calibration at lengths 106–110 (about 5 hours)

This checks whether the search can at least tie the published records. Each
length runs for 1 hour.

```sh
bash runs/start.sh step5
```

**When it finishes:** each result has a `published:` line that says whether it
`ties the best published value` or is above it. If most lengths are above the
record, the search needs tuning before steps 6–7. Tell Claude the results.

## Step 6: PSL target lengths (about 12 hours)

- **L = 251:** no valid published code has PSL 10. The published code gives 11.
- **L = 135, 164, 196, 234, 272:** at each of these, the record is 1 higher
  than at the length just below it.

Each length runs for 2 hours.

```sh
bash runs/start.sh step6
```

## Step 7: Full PSL sweep, lengths 106–300 (about 49 hours)

The sweep covers 195 lengths at 15 minutes each.

```sh
bash runs/start.sh step7
```

If the sweep stops partway, edit `--length` in `runs/step7_psl_106_300.sh` to
the remaining range (for example `180-300`), then start it again.

## Step 8: Barker calibration at lengths 30, 40, 50, 60 (about 2 hours)

Unimodular (polyphase) Barker sequences are known for every length up to 70
and for 72, 76 and 77 (Jedwab 2008, Proposition 7.1; Nunn & Coxson 2009). This
step measures how far the search reaches on your Mac. Each length runs for up
to 30 minutes and stops at the first certified sequence.

```sh
bash runs/start.sh step8
```

**When it finishes:** each length prints `CERTIFIED Barker sequence` or
`not found`, with the best peak. If it can't certify length 50 or 60, the
search needs more work before step 9 is worth running. Tell Claude the results.

## Step 9: Barker target lengths 71, 73, 74, 75 (about 16 hours)

No unimodular Barker sequence is published for these lengths. The 2009 search
that found 72, 76 and 77 ran for months on several computers, so expect a long
search, and run this only after step 8 looks good. Each length runs for up to
4 hours.

```sh
bash runs/start.sh step9
```

A result counts only if the log says `CERTIFIED Barker sequence`. That line
means the interval-arithmetic upper bound on every sidelobe is at most 1. To
re-check saved results at any time, run
`.venv/bin/python -m fastcodes barker-certify`. Before claiming a new length,
check section 7.5 of Coxson & Russo's 2025 book *Optimal-Peak-Sidelobe
Polyphase Codes*, which we could not read.

## Step 10: Complete search at length 13, 8 codes (estimate: about 45 minutes)

This step checks every possible set of 8 binary codes of length 13. It
either finds type-2 sets or proves that none exist. Either result is new:
Virtanen et al. (2013) list type-2 codes as known only for lengths 8–12
and 14. The time is an estimate scaled from the length-12 run (169 s); each
extra length costs about 16 times more. Close other heavy programs first.

```sh
bash runs/start.sh step10
```

The method is checked against Sulzer (1993): it reproduces his counts of
86, 78 and 150 sets for lengths 8, 9 and 10 exactly, and it finds his
published sets. The search doesn't save checkpoints; if it's interrupted,
start it again.

## Step 11: Complete search at length 14 (estimate: about 12 hours)

This validates the method on the known length-14 set (Markkanen et al. 2008)
and counts every length-14 set. Overnight, plugged in. It needs about 3 GB
of memory.

```sh
bash runs/start.sh step11
```

## Step 12: Vector-space search, 12 codes, lengths 13–16 (about 2 hours)

Sets of 12 codes would use fewer codes than the 16 that type-1 codes need at
these lengths. Each length runs for up to 30 minutes.

```sh
bash runs/start.sh step12
```

## Step 13: Complete search at length 15, 8 codes (about 7 hours)

This uses the independent C program in `cleanroom/`, which is about 30 times
faster than `exhaust-type2`. It agrees with `exhaust-type2` on every set at
lengths 8–14. A result at length 15 is new either way: a type-2 set longer
than the published record of 14, or a proof that 8 codes can't reach 15. It
uses about 6 GB of memory.

```sh
bash runs/start.sh step13
```

To see the percentage done, the time elapsed and the estimated finish time,
run:

```sh
bash runs/progress.sh 15
```

## If you find something

- **Type-2:** `report` prints `type-2 alternating set found:` with the codes
  in hex. To check them independently, run:

  ```sh
  .venv/bin/python -m fastcodes check set --length <L> <hex1> <hex2> ...
  ```

  The type-2 peak must be 0.
- **PSL:** a line that says `BELOW THE BEST PUBLISHED VALUE` gives a possible
  new record. To check the code, run:

  ```sh
  .venv/bin/python -m fastcodes check psl --length <L> <hex>
  ```

In both cases, keep the log and `results/runs.jsonl`, and recheck the latest
literature before you claim a new result.
