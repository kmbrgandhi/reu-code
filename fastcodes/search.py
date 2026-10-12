"""Python driver: parameters, deadlines, multicore runs, and verification.

    from fastcodes import psl, type2
    from fastcodes.search import search, search_parallel

    r = search(psl(64), 'memetic', time_limit=30, target=4)
    r = search_parallel(type2(15, 8), 'tabu', time_limit=600, target=0)

Every result is checked with the independent evaluators in verify.py before
it is returned (Result.verified).
"""
import json
import math
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import continuous as CO
from . import discrete as DI
from . import kernels as KN
from . import verify as VF


@dataclass(frozen=True)
class Algo:
    fn: object
    names: tuple
    defaults: tuple
    continuous: bool = False   # True: needs a phases=0 problem
    both: bool = False         # True: works for discrete and continuous problems
    doc: str = ''


ALGOS = {
    'random': Algo(DI.random_search, ('max_restarts',), (0,), both=True,
                   doc='random sampling baseline'),
    'sdls': Algo(DI.sdls, ('max_restarts',), (0,),
                 doc='restarted steepest descent (2017 hill climbing, sdls)'),
    'first': Algo(DI.first_improvement, ('max_restarts',), (0,),
                  doc='restarted first-improvement climbing (2017 simple climb)'),
    'stochastic': Algo(DI.stochastic_climb, ('max_restarts', 'iters', 'temp'), (0, 40, 2.0),
                       doc='2017 stochastic climb'),
    'anneal': Algo(DI.anneal,
                   ('max_restarts', 't0_factor', 'gen_factor', 'ntemp_factor', 'kmove', 'alpha',
                    'schedule', 'beta', 'final_temp', 'stuck_kappa', 'stuck_mult', 'tabu',
                    'tabu_len', 'local'),
                   (0, 2.0, 25.0, 2.9, 3, 1.5, 0, -1.0, 0.25, 0.2, 2.0, 0, 100, 0),
                   doc='simulated annealing (2017 anneal_full; tabu=1, local=1 variants)'),
    'threshold': Algo(DI.threshold_accepting,
                      ('max_restarts', 'gen_factor', 'kmove', 'beta', 'init_mult', 'stuck_kappa',
                       'stuck_mult', 'nthr_factor'),
                      (0, 30.0, 3, 1.3, 2.0, 0.2, 2.0, 0.5),
                      doc='threshold accepting (2017 threshold_accept)'),
    'tabu': Algo(DI.tabu, ('max_restarts', 'max_iters', 'min_tenure', 'extra_tenure'),
                 (0, 0, 0, -1), doc='restarted tabu search (auto lengths when 0/-1)'),
    'saw': Algo(DI.self_avoiding_walk, ('max_restarts', 'walk_factor'), (0, 64.0),
                doc='self-avoiding walk (new; lssOrel-style)'),
    'memetic': Algo(DI.memetic,
                    ('max_runs', 'pop_size', 'n_gens', 'offspring', 'p_cross', 'p_mut', 'mut_k',
                     'restart_every', 'restart_mult', 'ts_iters', 'ts_min', 'ts_extra',
                     'local_search'),
                    (0, 100, 30, 500, 0.1, -1.0, 2, 8, 1.5, 20, 2, 1, 0),
                    doc='memetic + tabu search (2017 run_evolution_memet)'),
    'evolution': Algo(DI.evolution,
                      ('max_runs', 'mode', 'pop', 'n_gens', 'offspring', 'mut_k', 'restart_every',
                       'restart_count'),
                      (0, 1, 100, 25, 500, 3, 8, 15),
                      doc='(mu+lambda) evolution (2017 evolut_algo; mode=0 is the naive one)'),
    'genetic': Algo(DI.genetic,
                    ('max_runs', 'pop', 'n_gens', 'selection', 'crossover', 'retain',
                     'random_select', 'mutate', 'k'),
                    (0, 100, 25, 0, 0, 0.2, 0.05, 0.02, 2),
                    doc='generational GA (2017 genetic_full)'),
    'deluge': Algo(CO.great_deluge, ('max_restarts', 'n_inner', 'min_step', 'divisor', 'rain'),
                   (0, -1, 1e-5, 1.3, 0.03), continuous=True,
                   doc='great deluge, continuous phases (2017 polyphase.great_deluge)'),
    'pso': Algo(CO.particle_swarm,
                ('max_restarts', 'swarm_mult', 'omega', 'c1', 'c2', 'vnorm', 'max_iters'),
                (0, 20.0, 1.0, 2.0, 2.0, math.pi, 0), continuous=True,
                doc='particle swarm, continuous phases (2017 particle_swarm)'),
    'anneal_phases': Algo(CO.anneal_phases,
                          ('max_restarts', 't0_factor', 'gen_factor', 'ntemp_factor', 'step_norm',
                           'alpha', 'beta', 'stuck_kappa', 'stuck_mult', 'ls_min_step',
                           'ls_divisor', 'ls_rain'),
                          (0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.3, 0.2, 2.0, 0.01, 1.6, 0.4),
                          continuous=True,
                          doc='annealing in phase space + deluge local search (2017 anneal_phases)'),
}

# Named variants that match the 2017 experiment names.
PRESETS = {
    'anneal_tabu': ('anneal', dict(tabu=1)),
    'anneal_local': ('anneal', dict(local=1)),
    'evolution_naive': ('evolution', dict(mode=0)),
    'genetic_part_random': ('genetic', dict(selection=0)),
    'genetic_random': ('genetic', dict(selection=1)),
    'genetic_tournament': ('genetic', dict(selection=2)),
    'memetic_sdls': ('memetic', dict(local_search=1)),
    'memetic_simple': ('memetic', dict(local_search=2)),
}

SCHEDULES = ['semi_exponential', 'mult_exponential', 'log_mult', 'lin_mult', 'quad_mult',
             'lin_add', 'quad_add', 'expon_add', 'trigon_add', 'quenching']
SELECTIONS = ['part_random', 'random', 'tournament']
LOCAL_SEARCHES = ['tabu', 'sdls', 'first']
CROSSOVERS = ['uniform', 'half', 'randpoint']


def resolve(algo, params=None):
    """Return (Algo, parameter vector, parameter dict) for a name or preset."""
    params = dict(params or {})
    if algo in PRESETS:
        base, over = PRESETS[algo]
        params = {**over, **params}
        algo = base
    if algo not in ALGOS:
        raise ValueError(f'unknown algorithm {algo!r}; choose from {sorted(ALGOS) + sorted(PRESETS)}')
    a = ALGOS[algo]
    for key, table in (('schedule', SCHEDULES), ('selection', SELECTIONS), ('crossover', CROSSOVERS),
                       ('local_search', LOCAL_SEARCHES)):
        if isinstance(params.get(key), str):
            params[key] = table.index(params[key])
    unknown = set(params) - set(a.names)
    if unknown:
        raise ValueError(f'{algo}: unknown parameters {sorted(unknown)}; valid: {list(a.names)}')
    prm = np.array([float(params.get(n, d)) for n, d in zip(a.names, a.defaults)])
    return a, prm, {n: float(v) for n, v in zip(a.names, prm)}


@dataclass
class Result:
    problem: dict
    algo: str
    params: dict
    seed: int
    time: float
    evals: int
    restarts: int
    cost: float
    peak: float
    verified_peak: float
    verified: bool
    code: dict = field(default_factory=dict)
    hit_target: bool = False
    workers: int = 1

    @property
    def evals_per_sec(self):
        return self.evals / self.time if self.time > 0 else float('nan')

    def to_json(self):
        d = asdict(self)
        d['evals_per_sec'] = self.evals_per_sec
        return json.dumps(d, default=float)

    def summary(self):
        tgt = ' (target reached)' if self.hit_target else ''
        ok = 'verified' if self.verified else 'VERIFICATION FAILED'
        return (f"{self.problem['kind']} L={self.problem['L']} K={self.problem['K']} "
                f"m={self.problem['phases']} N={self.problem['doppler']} | {self.algo}: "
                f"peak {self.peak:.6g}{tgt} [{ok}: {self.verified_peak:.6g}] | "
                f"{self.time:.1f}s, {self.evals:,} move evals ({self.evals_per_sec:,.0f}/s), "
                f"{self.restarts} restarts, {self.workers} worker(s)")


def _check_compat(problem, a, algo):
    if a.both:
        return
    if a.continuous and not problem.continuous:
        raise ValueError(f'{algo} needs continuous phases (build the problem with phases=0)')
    if not a.continuous and problem.continuous:
        raise ValueError(f'{algo} needs a discrete alphabet (phases=2 or more)')


def _result(problem, algo, pdict, seed, elapsed, evals, restarts, best, bpk, B, tgt):
    vpk = VF.evaluate(problem, B[0], B[1])
    return Result(problem=problem.describe(), algo=algo, params=pdict, seed=seed, time=elapsed,
                  evals=evals, restarts=restarts, cost=float(best), peak=float(bpk),
                  verified_peak=float(vpk), verified=bool(abs(vpk - bpk) < 1e-6 * max(1.0, vpk)),
                  code=problem.export(B[0], B[1]), hit_target=bool(bpk <= tgt + 1e-9))


def _write_checkpoint(path, result):
    """Atomically replace `path` with the result as JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(result.to_json())
    os.replace(tmp, path)


def search(problem, algo='sdls', time_limit=10.0, target=None, seed=None, params=None,
           chunk=None, stop_event=None, checkpoint=None, checkpoint_every=30.0):
    """Run one search on this process.

    target      stop when the peak objective (PSL, set peak, or type-2 peak) is
                <= target. None runs until the time limit.
    chunk       optional seconds per compiled call. Between calls the driver
                checks stop_event and writes checkpoints. Restart-based
                algorithms are unaffected; population and annealing runs are
                cut at chunk boundaries.
    checkpoint  optional file path. The best result so far (verified) is
                written there at most every `checkpoint_every` seconds, so an
                interrupted run keeps its progress.
    """
    a, prm, pdict = resolve(algo, params)
    _check_compat(problem, a, algo)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    KN.seed_rng(seed)
    P = problem.arrays()
    S = problem.new_state()
    Bc = problem.new_state()
    B = problem.new_state()
    tgt = -1.0 if target is None else float(target)
    t0 = time.perf_counter()
    deadline = t0 + float(time_limit)
    best, bpk, evals, restarts = math.inf, math.inf, 0, 0
    one_call = chunk is None or prm[0] > 0
    last_ck = t0
    while True:
        dl = deadline if one_call else min(deadline, time.perf_counter() + chunk)
        c, pk, ev, r = a.fn(P, S, prm, tgt, dl, Bc)
        evals += int(ev)
        restarts += int(r)
        if c < best:
            best, bpk = c, pk
            for dst, src in zip(B, Bc):
                dst[...] = src
        now = time.perf_counter()
        if checkpoint and best < math.inf and now - last_ck >= checkpoint_every:
            _write_checkpoint(checkpoint, _result(problem, algo, pdict, seed, now - t0, evals,
                                                  restarts, best, bpk, B, tgt))
            last_ck = now
        if (one_call or bpk <= tgt + 1e-9 or now >= deadline
                or (stop_event is not None and stop_event.is_set())):
            break
    if best == math.inf:
        raise RuntimeError('search returned no state; increase the time limit')
    return _result(problem, algo, pdict, seed, time.perf_counter() - t0, evals, restarts,
                   best, bpk, B, tgt)


def _worker(args):
    problem, algo, time_limit, target, seed, params, chunk, ev, ck = args
    res = search(problem, algo, time_limit, target, seed, params, chunk, ev, checkpoint=ck)
    if res.hit_target:
        ev.set()
    return res


def _stop_workers_on_signal(ex, mgr):
    """Make SIGTERM (pkill) and SIGINT (Ctrl+C) stop the worker processes too.

    Without this, workers keep running until their time limit after the main
    process is killed. SIGHUP is left alone so that `nohup` keeps working.
    """
    def handler(signum, frame):
        for p in list((getattr(ex, '_processes', None) or {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        try:
            mgr.shutdown()
        except Exception:
            pass
        os._exit(128 + signum)

    old = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, handler)
    return old


# Algorithms whose single run is long; cutting them into short chunks would
# restart them, so they get longer chunks (and checkpoints) by default.
LONG_RUNS = {'memetic', 'evolution', 'genetic', 'anneal', 'threshold', 'pso', 'anneal_phases'}


def checkpoint_path(problem, algo, worker, directory='results/checkpoints'):
    d = problem.describe()
    name = f"{d['kind']}_L{d['L']}_K{d['K']}_m{d['phases']}_N{d['doppler']}_{algo}_w{worker}.json"
    return os.path.join(directory, name)


def search_parallel(problem, algo='sdls', time_limit=10.0, workers=None, target=None, seed=None,
                    params=None, chunk=None, checkpoint_dir='results/checkpoints'):
    """Independent searches on `workers` processes (default: all cores).

    Returns the best Result, with evals and restarts summed over workers. When
    one worker reaches the target, the others stop at their next chunk boundary.
    Default chunk: 10 s, or 300 s for the long-run algorithms in LONG_RUNS.
    Each worker writes its best result so far to `checkpoint_dir` every 30 s;
    the files are removed when the run completes (set checkpoint_dir=None to
    turn this off). Files left there come from interrupted runs.
    """
    workers = workers or os.cpu_count()
    warmup(algo)  # compile once here so the workers load the cache
    base_algo = PRESETS[algo][0] if algo in PRESETS else algo
    if chunk is None:
        chunk = min(300.0 if base_algo in LONG_RUNS else 10.0, float(time_limit))
    base = seed if seed is not None else int.from_bytes(os.urandom(4), 'little')
    ctx = mp.get_context('spawn')
    with ctx.Manager() as mgr:
        ev = mgr.Event()
        cks = [checkpoint_path(problem, algo, i, checkpoint_dir) if checkpoint_dir else None
               for i in range(workers)]
        jobs = [(problem, algo, time_limit, target, (base + 7919 * i) % 2 ** 32, params, chunk, ev,
                 cks[i]) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            old = _stop_workers_on_signal(ex, mgr)
            try:
                results = list(ex.map(_worker, jobs))
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    for ck in cks:
        if ck and os.path.exists(ck):
            os.remove(ck)
    best = min(results, key=lambda r: (r.peak, r.cost))
    best.evals = sum(r.evals for r in results)
    best.restarts = sum(r.restarts for r in results)
    best.time = max(r.time for r in results)
    best.workers = workers
    return best


def warmup(algo=None):
    """Compile (or load from cache) the kernels and the given algorithms."""
    from .problems import psl, type2
    names = [algo] if algo else list(ALGOS)
    for name in names:
        a, _, _ = resolve(name)
        pb = type2(6, 4, phases=0) if a.continuous else type2(6, 4)
        search(pb, name, time_limit=0.05, seed=1, params={a.names[0]: 1})
        if not a.continuous:
            search(psl(5), name, time_limit=0.05, seed=1, params={a.names[0]: 1})


def save(result, path='results/runs.jsonl'):
    """Append a result as one JSON line."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(result.to_json() + '\n')
