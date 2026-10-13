"""Merge the chunks of a t2search run (-r first:last) and run its consistency checks.

A full run checks three things: every 4-multiset was generated once, the
canonical splits equal the distinct solutions, and every 4+4 split of every
solution was found. A chunk can't check these on its own, because the splits
of one solution can fall in different chunks. This script checks them over
all chunks together, and also checks that the chunks used the same setup and
cover every task exactly once.

Usage: python3 merge_chunks.py N chunk_dir solutions_out
Reads chunk_dir/chunk_*.log and chunk_dir/chunk_*.txt. Exits 1 if a check fails.
"""
import re
import sys
from itertools import combinations
from pathlib import Path


def fail(msg):
    print(f"MERGE FAILED: {msg}")
    sys.exit(1)


def twice_splits(sol):
    """Twice the number of unordered 4+4 splits of one solution (as t2search counts them)."""
    ys = {tuple(sorted(sol[i] for i in c)) for c in combinations(range(8), 4)}
    n = 0
    for y in ys:
        rest = list(sol)
        for w in y:
            rest.remove(w)
        n += 1 + (tuple(sorted(rest)) == y)
    return n


N, cdir, out = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
logs = sorted(cdir.glob('chunk_*.log'), key=lambda p: int(p.stem.split('_')[1]))
if not logs:
    fail(f"no chunk logs in {cdir}")

setup, ranges, sols = None, [], set()
quads = canon = tw = 0
search_time = 0.0
for log in logs:
    text = log.read_text()
    lines = text.splitlines()
    head = [re.sub(r' threads=\d+', '', l) for l in lines
            if l.startswith(f'N={N} ') or l.startswith('pairs=')]
    if setup is None:
        setup = head
    elif head != setup:
        fail(f"{log.name} used a different setup: {head} vs {setup}")
    m = re.search(r'^CHUNK N=(\d+) tasks=(\d+)\.\.(\d+) of (\d+) 4-multisets=(\d+) canonical=(\d+) '
                  r'twice_splits=(\d+) unique=(\d+) search_time=([\d.]+) s$', text, re.M)
    if not m or int(m.group(1)) != N:
        fail(f"{log.name} has no CHUNK line for N={N}")
    if not re.search(r'^wrote \d+ solutions to ', text, re.M):
        fail(f"{log.name}: the solution file was not written")
    a, b, total = int(m.group(2)), int(m.group(3)), int(m.group(4))
    ranges.append((a, b, total))
    quads += int(m.group(5)); canon += int(m.group(6)); tw += int(m.group(7))
    search_time += float(m.group(9))
    rows = (cdir / f'{log.stem}.txt').read_text().splitlines()
    if len(rows) != int(m.group(8)):
        fail(f"{log.stem}.txt has {len(rows)} lines, the log says {m.group(8)}")
    sols.update(rows)

total = ranges[0][2]
pos = 0
for a, b, t in sorted(ranges):
    if t != total or a != pos:
        fail(f"task ranges don't tile 0..{total}: {sorted(ranges)}")
    pos = b
if pos != total:
    fail(f"tasks {pos}..{total} are missing")

T4 = int(re.search(r'4-multisets=(\d+)', ' '.join(setup)).group(1))
tw2 = sum(twice_splits(s.split()) for s in sols)
ok = quads == T4 and canon == len(sols) and tw == tw2
print(f"merged {len(logs)} chunks covering tasks 0..{total}")
print(f"4-multisets generated={quads} (expected {T4}) {'OK' if quads == T4 else 'MISMATCH'}")
print(f"canonical={canon} unique solutions={len(sols)} "
      f"{'(canonical==unique OK)' if canon == len(sols) else '(CANONICAL/UNIQUE MISMATCH)'}")
print(f"all 4+4 splits found: {tw // 2}, expected {tw2 // 2} {'OK' if tw == tw2 else 'MISMATCH'}")
if not ok:
    fail("consistency check failed")
out.write_text(''.join(s + '\n' for s in sorted(sols)))
print(f"RESULT N={N} count={len(sols)} search_time={search_time:.2f} s chunks={len(logs)} (merged from {cdir})")
print(f"wrote {len(sols)} solutions to {out}")
