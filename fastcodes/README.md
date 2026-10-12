# fastcodes

`fastcodes` is a fast, tested rewrite of the 2017 `REU_code` search heuristics.
It searches for:

- Binary, m-phase, and continuous-phase codes with low peak sidelobe level
  (PSL), optionally over Doppler bins.
- Complementary code sets: K codes whose autocorrelations sum to zero away
  from the peak. The 2017 code called these "alternating codes".
- Type-2 alternating code sets (Sulzer 1993), the incoherent-scatter radar
  codes that the 2017 work was aiming for.

Every reported code is re-checked with an independent NumPy evaluator before
it is returned.

## Set up

The repository already has a virtual environment in `.venv`. To recreate it:

```sh
cd ~/Documents/Personal/CodingProjects/reu-code
python3 -m venv .venv
.venv/bin/pip install -r fastcodes/requirements.txt
```

Then compile everything once and run the tests. The first compile takes one to
two minutes; numba caches the result, so later runs start immediately.

```sh
.venv/bin/python -m fastcodes warmup
.venv/bin/python -m pytest fastcodes/tests -q
```

Run all commands from the repository root (`~/Documents/Personal/CodingProjects/reu-code`).

## Run searches from the command line

Searches use every CPU core by default. Each core runs an independent search
with its own random seed, and the best verified result is reported. To use
fewer cores, add `--workers N`. You don't need to set anything else to run in
parallel.

| Goal | Command |
|---|---|
| Binary code with optimal PSL | `.venv/bin/python -m fastcodes run psl --length 64 --algo memetic --time 60 --target 4` |
| Sweep many lengths | `.venv/bin/python -m fastcodes run psl --length 106-300 --algo saw --time 600` |
| Quadriphase code, 3 Doppler bins | `.venv/bin/python -m fastcodes run psl --length 40 --phases 4 --doppler 3 --algo tabu --time 60` |
| Continuous-phase code | `.venv/bin/python -m fastcodes run psl --length 45 --phases 0 --algo deluge --time 60` |
| Complementary set of 4 codes | `.venv/bin/python -m fastcodes run set --length 13 --codes 4 --algo tabu --time 60` |
| Type-2 alternating set | `.venv/bin/python -m fastcodes run type2 --length 15 --codes 8 --algo saw --time 3600` |
| Summarize all results so far | `.venv/bin/python -m fastcodes report` |
| Check codes you have (hex) | `.venv/bin/python -m fastcodes check set --length 14 406 b72 4df bab 24c d38 295 de1` |
| List algorithms and parameters | `.venv/bin/python -m fastcodes algos` |
| Speed benchmarks | `.venv/bin/python -m fastcodes bench --quick` |

Useful options for `run`:

- `--target X`: stop when the peak is at most X. Searches for sets and type-2
  codes stop at 0 by default. When one core reaches the target, the others stop
  within 10 seconds.
- `--time S`: seconds per length.
- `--beat-record`: for binary PSL, stop a length early only when the result is
  below the best published value. Otherwise each length runs for `--time`.
- `--param key=value`: set an algorithm parameter, for example
  `--param walk_factor=256` or `--param schedule=lin_add`. Use
  `python -m fastcodes algos` to see the names and defaults.
- `--metric psl|isl|pnorm` and `--p`: the cost the search minimizes. `psl` is
  the peak, with a tiny energy term to break ties. `isl` is the sidelobe
  energy. `pnorm` is the sum of |sidelobe|^p, a smooth stand-in for the peak.
- `--out FILE`: results are appended as JSON lines (default
  `results/runs.jsonl`).

For binary PSL runs, each result is compared with the best published value for
that length. The output says whether the result ties the record, is above it,
or is below it.

To stop a run, press Ctrl+C or run `pkill -f "fastcodes run"`. The worker
processes stop too. For a step-by-step campaign, see [../RUNBOOK.md](../RUNBOOK.md).

To run a long campaign after you close the terminal, use `nohup`:

```sh
mkdir -p results
nohup .venv/bin/python -m fastcodes run psl --length 106-300 --algo saw --time 900 --beat-record > results/psl_106_300.log 2>&1 &
```

## Search for polyphase Barker sequences

`python -m fastcodes barker --length 60 --time 1800` searches for a
unimodular Barker sequence: phases phi[n] with every aperiodic sidelobe
|C(k)| <= 1. The method follows Nunn & Coxson (2009): it lowers the ISL by
basin hopping, then pushes every sidelobe under a ceiling that drops step by
step.

A result counts only after certification. The phases are rounded to exact
fractions m[n] / 2^40 of a turn, and interval arithmetic (`mpmath.iv`) bounds
every |C(k)|. Results go to `results/barker.jsonl`. To re-check them, run
`python -m fastcodes barker-certify`. To check the published codes, run
`python -m fastcodes barker-certify --known`; all nine Nunn & Coxson (2009)
codes pass, and these codes are stored in `data/nunn_coxson_2009_barker.json`.

In short tests on one shared core, the search certified lengths 13, 17 and 20
within seconds, but did not reach length 35 within a minute. Run
`runs/step8_barker_calibration.sh` to measure its reach on all cores before you
aim at the open lengths 71, 73, 74 and 75.

## Use the Python API

```python
from fastcodes import psl, type2, complementary_set
from fastcodes.search import search, search_parallel

r = search(psl(51), 'memetic', time_limit=30, target=3)          # one core
r = search_parallel(type2(14, 8), 'saw', time_limit=600, target=0)  # all cores
print(r.summary())
print(r.code)        # signs and hex for binary codes
```

## Problems

| Builder | What it minimizes |
|---|---|
| `psl(L, phases=2, doppler=1)` | max over delays τ ≠ 0 and Doppler bins q of \|Σ a[n] conj(a[n+τ]) e^(−2πiqn/N)\|. N = 1 is ordinary PSL. This is the 2017 `fast_autocorrelations` definition. |
| `complementary_set(L, K, phases=2)` | peak of the summed aperiodic autocorrelation of K codes. 0 means a complementary set. |
| `type2(L, K, phases=2)` | max over l, k ≥ 1 of \|Σ_codes Σ_j a[j] conj(a[j+l]) conj(a[j+k]) a[j+l+k]\|. 0 means a type-2 alternating set (Sulzer 1993; polyphase form as in Markkanen et al. 2008). |

`phases=2` is binary, `phases=m` uses the m-th roots of unity, and `phases=0`
is continuous phase. Symmetries are removed by fixing values: the first element
of each code (all problems), and also the second element of each code for
type-2 problems. `test_type2_symmetries_justify_frozen_positions` checks that
this loses no solutions.

## Algorithms

| Name | 2017 source | Notes |
|---|---|---|
| `random` | `rand_select_algorithm_doppler`, `random_arbitrary_polyphase` | baseline |
| `sdls` | `hill_climbing_iterative` + `sdls_local_search` | restarted steepest descent |
| `first` | `simple_local_search` | restarted first-improvement climbing |
| `stochastic` | `stochastic_local_search` | |
| `anneal` (`anneal_tabu`, `anneal_local`) | `anneal_full.py` | all ten cooling schedules (`schedule=`) |
| `threshold` | `threshold_accept.py` | |
| `tabu` | `ts_local_search` | restarted tabu search |
| `memetic` (`memetic_sdls`, `memetic_simple`) | `memetic_tabu_search.py` | |
| `evolution` (`evolution_naive`) | `evolut_full.py` | |
| `genetic` (`genetic_part_random`, `genetic_random`, `genetic_tournament`) | `genetic_full.py`, `crossover_algs.py` | `crossover=uniform\|half\|randpoint` |
| `saw` | new | self-avoiding walk, the idea behind lssOrel (Bošković, Brglez & Brest 2017) |
| `deluge` | `polyphase.great_deluge` | continuous phase |
| `pso` | `particle_swarm.py` | continuous phase |
| `anneal_phases` | `polyphase.anneal_phases` | continuous phase |

`ant_colony.py` was an empty stub and is not ported. Exhaustive search is not
ported; `REU_code/exhaustive.py` and the review notes describe its issues.

## How it is faster

The 2017 code copied the code and recomputed the whole ambiguity function for
every candidate move. `fastcodes` keeps the correlation values of the current
code in memory and updates only the terms a single-element change affects:

- Binary PSL: O(L) per candidate move, instead of O(L²) Python operations.
- Doppler PSL: O(L·N) per move.
- Type-2 sets: O(L²) per move, instead of O(K·L³).

When a candidate can't beat the best move found so far, the evaluation stops
early. Everything is compiled with numba, and searches run on all cores.

Measured on an Apple M3 with `python -m fastcodes bench --quick` (one core,
exact cost of one candidate move):

| Problem | 2017 code (evals/s) | fastcodes (evals/s) | Speedup |
|---|---:|---:|---:|
| Binary PSL, L = 13 | 22,416 | 6,758,205 | 301× |
| Binary PSL, L = 64 | 6,302 | 3,468,784 | 550× |
| Binary PSL, L = 256 | 1,716 | 1,181,290 | 688× |
| Quadriphase, L = 40, 3 Doppler bins | 3,288 | 1,082,255 | 329× |
| Type-2, L = 14, 8 codes | — | 2,145,194 | — |

The parallel runner multiplies throughput by roughly the number of cores.
Early exit makes real searches faster still. Your numbers will vary with the
machine and its load.

## Verification

- `fastcodes/verify.py` holds independent NumPy evaluators. The type-2 check
  builds Sulzer's lag-product sequences explicitly, unlike the kernels. Every
  search result is re-scored with them (`Result.verified`).
- `fastcodes/tests` (62 tests) checks the following:
  - Incremental updates match a full recompute after random moves, for every
    problem type and metric.
  - Early-exit evaluation is exact whenever it matters.
  - The Barker codes have PSL 1.
  - All 104 published example codes for L = 2–105 have their listed PSL. The
    exception is the L = 103 code, which is printed with a typo in its source.
  - All 195 published codes for L = 106–300 have their listed PSL, except the
    flagged L = 251 row.
  - The Sulzer (1993) and Markkanen et al. (2008) type-2 sets score 0.
  - A Golay pair is complementary.
  - Every algorithm returns a verified result.

## Reference data

- `data/binary_min_psl.tsv`: minimum or best-known binary PSL for L = 2–105.
  Each row has an example code and its source:
  - Leukhin & Potekhin 2012/2013/2015 (exhaustive, L ≤ 82)
  - Dimitrov et al. 2021 (optimality at 83–84)
  - Nunn & Coxson 2008 (best known, 83–105)
- `data/binary_best_known_106_300.tsv`: best published PSL for L = 106–300.
  Every value comes from Dimitrov, Baicheva & Nikolov 2021 (IEEE Access
  9:112400, Tables 11–16). Earlier reports from Brest & Bošković 2021 and
  Dimitrov et al. 2020 are in the `other_reports` column. All printed codes
  were recomputed. At L = 251 the claimed PSL 10 is not backed by the printed
  code, which gives 11; that row is flagged.
- `records.py`: the published type-2 sets:
  - Sulzer 1993, Table 4: lengths 8–12, 8 codes.
  - Markkanen, Vierinen & Markkanen 2008, p. 2237, footnote 1: length 14,
    8 codes.

## Changes from the 2017 code

These 2017 bugs are fixed:

- Polyphase "alternating" searches computed autocorrelations without the
  complex conjugate. `complementary_set` uses the conjugate.
- `anneal` and `anneal_phases` never updated the current cost after accepting a
  better code.
- Memetic: crossover always returned the father, because `random.sample` returns
  a list that never equals `1`. The mutation probability was also inverted.
- Genetic: `mutate_all` rebound a loop variable, so it never mutated anything.
- Threshold accepting divided the threshold by j^β cumulatively, so it fell
  like 1/(j!)^β.
- Some test harnesses ran the wrong variant: "smart" evolution ran the naive
  one, and "half crossover" ran random-point crossover.
- Memetic search on 3-phase and 5-phase codes produced values outside the
  alphabet.

Deliberate differences from 2017:

- Memetic and evolution searches replace the worst members during partial
  restarts instead of growing the population.
- Threshold accepting uses the corrected threshold schedule.
- `tabu` has automatic run lengths by default. The 2017 values (20 iterations,
  tenure 2–3) are still the defaults inside `memetic`.

## Limitations

- Doppler bins follow the 2017 definition: N bins spread across the whole
  Doppler band. Sidelobes between bins are not controlled.
- Binary complementary sets use complex arithmetic, which is about 2–4× slower
  than an integer version would be.
- Time limits are checked periodically: every few hundred to few thousand
  evaluations, or once per child in population methods.
