"""Vector-space tabu search for binary type-2 alternating code sets.

Every code c has a fixed contribution vector v(c) to the fourth-order sums
S[l, k] (see exact.code_vectors). A set of K codes is type-2 exactly when its
vectors sum to zero. Codes with the same vector are interchangeable, so the
search works with the distinct vectors only (about 2^(N-3) of them).

State: K slots, each holding one distinct vector. Energy: ||sum of vectors||^2,
which equals the sidelobe energy (ISL metric) of the set; 0 means type-2.
Move: replace the vector in one slot by any other distinct vector. One step
scores every possible move with a matrix-vector product per slot, so a single
move can change a whole code. That neighborhood is far larger than the
single-bit flips of the other algorithms.

Tabu rule: a slot that just changed stays fixed for a few steps, and a vector
that just left a slot can't come back into that slot for a while (unless the
move gives a new best energy). Runs restart from random sets.

Composition: the coordinates S[1, N-3] and S[2, N-3] of a code depend only on
the products t_j = c_j c_{j+N-3}, j = 0, 1, 2. Each code has type A
(t0 = t1 = t2), B (t0 != t2) or C (t0 = t2 != t1), with values (2, 1), (0, -1)
and (-2, 1). For these two sums to vanish, a set of K codes needs exactly
K/4 codes of type A, K/2 of type B and K/4 of type C, so K must be a multiple
of 4 (N >= 4). With composition=True each slot keeps its type, which removes
the infeasible compositions and makes each step scan fewer vectors.
"""
import os
import time

import numpy as np
from numba import njit

from . import verify
from .exact import code_vectors
from .kernels import now


@njit(cache=True)
def _run(V, norms, typ, want, K, iters, slot_tenure, val_tenure, seed, deadline, best_c):
    """One tabu run from a random set. Slot r holds only vectors u with
    typ[u] == want[r] (want[r] < 0: any vector). Returns (best energy, steps)."""
    np.random.seed(seed)
    U, D = V.shape
    c = np.random.randint(0, U, K)
    for r in range(K):
        while want[r] >= 0 and typ[c[r]] != want[r]:
            c[r] = np.random.randint(0, U)
    S = np.zeros(D, dtype=np.int64)
    for r in range(K):
        S += V[c[r]]
    E = int(np.sum(S * S))
    best = E
    best_c[:] = c
    slot_until = np.zeros(K, dtype=np.int64)
    # tabu[u] = step until which vector u may not re-enter (any slot)
    val_until = np.zeros(U, dtype=np.int64)
    base = np.zeros(D, dtype=np.int64)
    for it in range(iters):
        bestE = np.int64(1) << 62
        bslot = -1
        bu = -1
        for r in range(K):
            if it < slot_until[r]:
                continue
            a = c[r]
            for d in range(D):
                base[d] = S[d] - V[a, d]
            const = np.int64(0)
            for d in range(D):
                const += base[d] * base[d]
            for u in range(U):
                if u == a or (want[r] >= 0 and typ[u] != want[r]):
                    continue
                dot = np.int64(0)
                for d in range(D):
                    dot += base[d] * V[u, d]
                e = const + 2 * dot + norms[u]
                if e < bestE and (it >= val_until[u] or e < best):
                    bestE = e
                    bslot = r
                    bu = u
        if bslot < 0:
            break
        a = c[bslot]
        for d in range(D):
            S[d] += V[bu, d] - V[a, d]
        c[bslot] = bu
        E = bestE
        slot_until[bslot] = it + slot_tenure
        val_until[a] = it + val_tenure
        if E < best:
            best = E
            best_c[:] = c
            if best == 0:
                return best, it + 1
        if (it & 31) == 0 and now() > deadline:
            return best, it + 1
    return best, iters


def code_types(codes):
    """Type of each binary code (rows of +-1): 0 = A, 1 = B, 2 = C (see the module docstring)."""
    c = np.asarray(codes)
    N = c.shape[1]
    t = c[:, :3] * c[:, N - 3:]
    return np.where(t[:, 0] != t[:, 2], 1, np.where(t[:, 0] == t[:, 1], 0, 2)).astype(np.int64)


def search(N, K, time_limit=60.0, seed=None, iters=0, slot_tenure=0, val_tenure=0, stop_event=None,
           composition=True):
    """Restarted vector-space tabu search. Returns a result dict; 'codes' holds
    a verified type-2 set if one was found. composition=True fixes the forced
    K/4 A + K/2 B + K/4 C composition (it needs K to be a multiple of 4)."""
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), 'little')
    rng = np.random.default_rng(seed)
    B, Vall = code_vectors(N)
    V, first = np.unique(Vall, axis=0, return_index=True)
    V = np.ascontiguousarray(V, dtype=np.int64)
    norms = np.sum(V * V, axis=1).astype(np.int64)
    typ = code_types(B[first]) if N >= 4 else np.zeros(len(V), dtype=np.int64)
    if composition:
        if K % 4 or N < 4:
            raise ValueError('a binary type-2 set needs a multiple of 4 codes (N >= 4)')
        want = np.array([0] * (K // 4) + [1] * (K // 2) + [2] * (K // 4), dtype=np.int64)
    else:
        want = np.full(K, -1, dtype=np.int64)
    iters = iters or 50 * K * N
    slot_tenure = slot_tenure or max(1, K // 4)
    val_tenure = val_tenure or 2 * K
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    best_c = np.zeros(K, dtype=np.int64)
    run_best = np.zeros(K, dtype=np.int64)
    best = None
    runs = steps = 0
    while time.perf_counter() < deadline and not (stop_event is not None and stop_event.is_set()):
        e, st = _run(V, norms, typ, want, K, iters, slot_tenure, val_tenure,
                     int(rng.integers(2 ** 31)), deadline, run_best)
        runs += 1
        steps += st
        if best is None or e < best:
            best = int(e)
            best_c[:] = run_best
        if best == 0:
            break
    codes = B[first[best_c]]
    res = dict(N=N, K=K, composition=bool(composition), found=bool(best == 0), energy=best,
               time=time.perf_counter() - t0,
               runs=runs, steps=steps, seed=seed, distinct_vectors=len(V),
               peak=float(np.abs(V[best_c].sum(axis=0)).max()))
    res['verified_peak'] = verify.type2_peak(codes)
    if res['found']:
        res['codes'] = codes.tolist()
        res['hex'] = [format(int(''.join('0' if b > 0 else '1' for b in row), 2), 'X').zfill((N + 3) // 4)
                      for row in codes]
    return res


def _worker(args):
    N, K, time_limit, seed, params, ev = args
    r = search(N, K, time_limit, seed, stop_event=ev, **params)
    if r['found']:
        ev.set()
    return r


def search_parallel(N, K, time_limit=60.0, workers=None, seed=None, **params):
    """Independent restarted searches on all cores; all stop when one finds a set."""
    import multiprocessing as mp
    import signal
    from concurrent.futures import ProcessPoolExecutor
    from .search import _stop_workers_on_signal
    workers = workers or os.cpu_count()
    search(6, 4, 0.01, 1)  # compile once here so the workers load the cache
    base = seed if seed is not None else int.from_bytes(os.urandom(4), 'little')
    ctx = mp.get_context('spawn')
    with ctx.Manager() as mgr:
        ev = mgr.Event()
        jobs = [(N, K, time_limit, (base + 7919 * i) % 2 ** 32, params, ev) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            old = _stop_workers_on_signal(ex, mgr)
            try:
                results = list(ex.map(_worker, jobs))
            finally:
                for sig, h in old.items():
                    signal.signal(sig, h)
    found = [r for r in results if r['found']]
    best = found[0] if found else min(results, key=lambda r: r['energy'])
    best['workers'] = workers
    best['runs'] = sum(r['runs'] for r in results)
    best['steps'] = sum(r['steps'] for r in results)
    best['time'] = max(r['time'] for r in results)
    return best
