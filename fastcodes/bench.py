"""Benchmarks: `python -m fastcodes bench [--quick]`.

1. Move-evaluation throughput: the 2017 approach (copy the code, recompute the
   whole ambiguity function with scipy.signal.convolve, then scan it in Python)
   against the incremental kernels.
2. Time to reach the published optimum PSL for a few binary lengths.
3. Time to find type-2 alternating sets at Sulzer's lengths.
"""
import cmath
import time

import numpy as np
from numba import njit
from scipy import signal

from . import kernels as KN
from .problems import psl, type2
from .records import binary_min_psl
from .search import search


# --- the 2017 evaluation, ported to Python 3 with no other changes -------------
def _second_max(conv, optarg=None):
    first = second = float('-inf')
    count = 0
    for x in conv:
        count += 1
        if abs(x) > second:
            if abs(x) >= first:
                first, second = abs(x), first
            else:
                second = abs(x)
    return second if count >= 2 else None


def fast_autocorrelations_2017(code, N, num_elem=None, op_on_list=_second_max):
    imag = cmath.sqrt(-1)
    overall = []
    for freq_delay in range(N):
        shifted = []
        for index in range(len(code)):
            shifted.append(cmath.exp(-2 * cmath.pi * imag * index * freq_delay / N) * code[index].conjugate())
        lst = signal.convolve(code, shifted[::-1])
        lst = lst[:(len(lst) // 2 + 1)]
        for i in lst:
            overall.append(abs(i))
    return op_on_list(overall, num_elem)


def _old_rate(L, m, N, seconds=1.0):
    """Evaluations per second of the 2017 neighbor evaluation (copy + full recompute)."""
    rng = np.random.default_rng(0)
    code = np.exp(2j * np.pi * rng.integers(0, m, L) / m)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        i = n % L
        new = np.copy(code)
        new[i] = new[i] * cmath.exp(2j * cmath.pi / m)
        fast_autocorrelations_2017(new, N)
        n += 1
    return n / (time.perf_counter() - t0)


@njit(cache=True)
def _scan(P, S, reps):
    nv = P[0][KN.IP_NV]
    nm = P[0][KN.IP_NMOVES]
    acc = 0.0
    n = 0
    for r in range(reps):
        for v in range(nv):
            for t in range(1, nm + 1):
                acc += KN.eval_move(P, S, v, t, np.inf)
                n += 1
    return acc, n


def _new_rate(pb, seconds=1.0):
    P, S = pb.arrays(), pb.new_state()
    KN.seed_rng(0)
    KN.random_init(P, S)
    _scan(P, S, 1)
    reps, n_done, t_used = 1, 0, 0.0
    while t_used < seconds:
        t0 = time.perf_counter()
        _, n = _scan(P, S, reps)
        t_used += time.perf_counter() - t0
        n_done += n
        reps *= 2
    return n_done / t_used


def throughput(quick):
    print('1. Move evaluations per second (exact cost of one candidate move, one core)')
    print(f"   {'problem':34s} {'2017 code':>12s} {'fastcodes':>14s} {'speedup':>9s}")
    cases = [(13, 2, 1), (64, 2, 1), (256, 2, 1), (40, 4, 3)]
    if not quick:
        cases += [(1024, 2, 1), (40, 3, 5)]
    for L, m, N in cases:
        old = _old_rate(L, m, N, 0.5 if quick else 2.0)
        new = _new_rate(psl(L, m, N), 0.5 if quick else 2.0)
        name = f'psl L={L} m={m} Doppler N={N}'
        print(f'   {name:34s} {old:12,.0f} {new:14,.0f} {new / old:8,.0f}x')
    for L, K in [(14, 8), (20, 16)]:
        new = _new_rate(type2(L, K), 0.5 if quick else 2.0)
        print(f"   {f'type-2 L={L} K={K}':34s} {'(no 2017 code)':>12s} {new:14,.0f}")


def time_to_optimum(quick):
    t = binary_min_psl()
    lengths = [21, 40, 51] if quick else [21, 28, 40, 48, 51, 60, 64]
    algos = ['sdls', 'tabu', 'memetic', 'saw']
    limit = 10.0 if quick else 60.0
    print(f'\n2. Seconds to reach the published optimum PSL, binary, one core, limit {limit:g} s')
    print('   ' + f"{'L':>4s} {'PSL':>4s} " + ' '.join(f'{a:>10s}' for a in algos))
    for L in lengths:
        row = []
        for a in algos:
            r = search(psl(L), a, time_limit=limit, target=t[L]['psl'], seed=12345)
            row.append(f'{r.time:10.2f}' if r.hit_target else f"{'>' + format(limit, 'g'):>10s}")
        print(f"   {L:4d} {t[L]['psl']:4d} " + ' '.join(row))


def type2_known(quick):
    limit = 10.0 if quick else 60.0
    algos = ['tabu', 'sdls', 'saw']
    print(f'\n3. Seconds to find a type-2 set with 8 codes at Sulzer (1993) lengths, one core, limit {limit:g} s')
    print('   ' + f"{'L':>4s} " + ' '.join(f'{a:>10s}' for a in algos))
    for L in ([8, 9, 10, 11] if quick else [8, 9, 10, 11, 12]):
        row = []
        for a in algos:
            r = search(type2(L, 8), a, time_limit=limit, target=0, seed=12345)
            row.append(f'{r.time:10.2f}' if r.hit_target else f"{'>' + format(limit, 'g'):>10s}")
        print(f'   {L:4d} ' + ' '.join(row))


def main(quick=False):
    throughput(quick)
    time_to_optimum(quick)
    type2_known(quick)
