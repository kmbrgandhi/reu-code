"""Search for unimodular (polyphase) Barker sequences, with certified results.

A unimodular Barker sequence of length N has a[n] = exp(i phi[n]) and
|C(k)| <= 1 for every lag k = 1..N-1, where C(k) = sum_n a[n] conj(a[n+k]).
Lag N-1 always has |C| = 1 exactly, so only lags 1..N-2 constrain the search.

Known results (see README.md): such sequences exist for every N <= 70 and for
N = 72, 76, 77 (Jedwab 2008, Proposition 7.1; Nunn & Coxson 2009). No sequence
is published for N = 71, 73, 74, 75 or N >= 78.

Method (after Nunn & Coxson 2009, who found theirs "using constrained
optimization starting at optimal-ISL codes", preprint abstract and Sect. II):

1. From random phases, find a local minimum of the ISL with L-BFGS, then
   lower it by basin hopping (perturb a few phases, re-minimize, keep drops).
2. Every few hops, run a level descent from the current low-ISL code: lower a
   ceiling step by step, pushing every |C(k)| under it by minimizing the
   excess sum_k max(0, |C(k)|^2 - ceiling^2)^2 with L-BFGS, until every
   |C(k)| <= 1 - margin or the ceiling stops dropping.

Certification: the phases are rounded to exact fractions m[n] / 2^40 of a full
turn, and every |C(k)|^2 is bounded with interval arithmetic (mpmath.iv). A
result counts as certified only if every upper bound is <= 1.
"""
import json
import math
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

H_DEFAULT = 2 ** 40   # phase grid for published results: phi = 2 pi m / H


# ---------------------------------------------------------------------------
# Objective and exact gradient (compiled)
# ---------------------------------------------------------------------------

@njit(cache=True)
def objective(phi, mode, p, tau):
    """Return (value, gradient, max interior |C(k)|) for full phases phi.

    mode 0: sum_{k=1}^{N-2} |C(k)|^p
    mode 1: sum_{k=1}^{N-2} max(0, |C(k)|^2 - tau)^2   (Barker excess)
    """
    N = phi.shape[0]
    a = np.cos(phi) + 1j * np.sin(phi)
    g = np.zeros(N)
    f = 0.0
    peak2 = 0.0
    for k in range(1, N - 1):
        C = 0.0 + 0.0j
        for n in range(N - k):
            C += a[n] * np.conj(a[n + k])
        m2 = C.real * C.real + C.imag * C.imag
        if m2 > peak2:
            peak2 = m2
        if mode == 0:
            f += m2 ** (0.5 * p)
            coef = 0.5 * p * m2 ** (0.5 * p - 1.0) if m2 > 0 else 0.0
        else:
            e = m2 - tau
            if e <= 0.0:
                continue
            f += e * e
            coef = 2.0 * e
        if coef == 0.0:
            continue
        cC = np.conj(C)
        for n in range(N - k):
            t = a[n] * np.conj(a[n + k])
            # d|C|^2/dphi[n] from this term = 2 Re(conj(C) * i t) = -2 Im(conj(C) t)
            w = coef * (-2.0) * (cC * t).imag
            g[n] += w
            g[n + k] -= w
    return f, g, np.sqrt(peak2)


def interior_peak(phi):
    """Largest |C(k)| over k = 1..N-2 (float64)."""
    return float(objective(np.asarray(phi, dtype=np.float64), 0, 2.0, 0.0)[2])


# ---------------------------------------------------------------------------
# Local optimization
# ---------------------------------------------------------------------------

def _minimize(phi, mode, p, tau, maxiter, scale=1.0):
    """L-BFGS on phi[2:] (phi[0] = phi[1] = 0 fixes the constant phase and the
    linear phase ramp, which leave every |C(k)| unchanged)."""
    from scipy.optimize import minimize
    full = np.zeros_like(phi)

    def fun(x):
        full[2:] = x
        f, g, _ = objective(full, mode, p, tau)
        return f * scale, g[2:] * scale

    res = minimize(fun, phi[2:], jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=0.0, gtol=1e-14, maxls=50))
    out = np.zeros_like(phi)
    out[2:] = res.x
    return out


def isl(phi):
    return float(objective(phi, 0, 2.0, 0.0)[0])


def level_descent(phi, margin=1e-7, shrink=0.995, rounds=80, maxiter=3000):
    """Lower a ceiling step by step: each round pushes every |C(k)| below
    shrink * (current peak) by minimizing the Barker excess. Stops at the
    target 1 - margin or when a round fails to lower the peak."""
    target = 1.0 - margin
    pk = interior_peak(phi)
    for _ in range(rounds):
        if pk <= target:
            break
        ceiling = max(shrink * pk, target)
        cand = _minimize(phi, 1, 2.0, ceiling ** 2, maxiter, scale=1e6)
        cpk = interior_peak(cand)
        if cpk < pk - 1e-12:
            phi, pk = cand, cpk
        else:
            break
    return phi, pk


def attempt(N, rng, deadline, margin=1e-7, k=3, sigma=0.4, stall_limit=60, descent_every=15,
            maxiter=3000):
    """One run: random phases -> local ISL minimum -> basin hopping on the ISL
    (perturb k phases by N(0, sigma), re-minimize, keep if the ISL drops), with
    a level descent toward |C| <= 1 every `descent_every` hops and at the end.
    Nunn & Coxson (2009) likewise started their Barker searches from low-ISL
    codes. Returns (best interior peak, its phases, number of level descents)."""
    target = 1.0 - margin
    phi = rng.uniform(0, 2 * np.pi, N)
    phi[:2] = 0.0
    phi = _minimize(phi, 0, 2.0, 0.0, maxiter)
    cur = isl(phi)
    best_pk, best_phi, descents = math.inf, phi, 0
    stall = hops = 0
    while stall < stall_limit and time.time() < deadline:
        cand = phi.copy()
        idx = rng.choice(np.arange(2, N), size=min(k, N - 2), replace=False)
        cand[idx] += rng.normal(0.0, sigma, size=idx.size)
        cand = _minimize(cand, 0, 2.0, 0.0, maxiter)
        v = isl(cand)
        hops += 1
        if v < cur - 1e-9:
            phi, cur, stall = cand, v, 0
        else:
            stall += 1
        if hops % descent_every == 0:
            q, pk = level_descent(phi.copy(), margin, maxiter=maxiter)
            descents += 1
            if pk < best_pk:
                best_pk, best_phi = pk, q
            if pk <= target:
                return best_pk, best_phi, descents
    q, pk = level_descent(phi.copy(), margin, maxiter=maxiter)
    descents += 1
    if pk < best_pk:
        best_pk, best_phi = pk, q
    return best_pk, best_phi, descents


# ---------------------------------------------------------------------------
# Certification with interval arithmetic
# ---------------------------------------------------------------------------

def quantize(phi, H=H_DEFAULT):
    """Integers m with phi ~= 2 pi m / H (exact rational phases for publication)."""
    return [int(v) % H for v in np.rint(np.asarray(phi) / (2 * np.pi) * H).astype(np.int64)]


def certify(m=None, H=H_DEFAULT, radians=None, dps=40):
    """Interval-arithmetic bound on max_{k=1..N-2} |C(k)|^2.

    Give either integers m (phases 2 pi m / H, exact) or decimal strings in
    radians (exact decimals as printed). Returns a dict with the upper bound
    and certified = (upper bound <= 1). Lag N-1 has |C| = 1 exactly and is not
    checked. An exact |C(k)| = 1 (as in binary Barker codes) cannot be
    certified this way; the search keeps a margin below 1 for that reason.
    """
    from mpmath import iv
    iv.dps = dps
    if m is not None:
        ph = [2 * iv.pi * iv.mpf(int(x)) / iv.mpf(H) for x in m]
    else:
        ph = [iv.mpf(str(x)) for x in radians]
    c = [iv.cos(x) for x in ph]
    s = [iv.sin(x) for x in ph]
    N = len(ph)
    worst = iv.mpf(0)
    worst_k = 0
    for k in range(1, N - 1):
        re = iv.mpf(0)
        im = iv.mpf(0)
        for n in range(N - k):
            re += c[n] * c[n + k] + s[n] * s[n + k]
            im += s[n] * c[n + k] - c[n] * s[n + k]
        m2 = re ** 2 + im ** 2
        if m2.b > worst.b:
            worst, worst_k = m2, k
    ub = float(worst.b)
    return dict(max_c2_upper=ub, max_c_upper=math.sqrt(ub), worst_lag=worst_k,
                certified=bool(worst.b <= 1), dps=dps)


def known_codes():
    """Published polyphase Barker codes (Nunn & Coxson 2009), as decimal strings."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'nunn_coxson_2009_barker.json')
    with open(path) as f:
        return {int(k): v for k, v in json.load(f)['codes'].items()}


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def search(N, time_limit=60.0, seed=None, stop_event=None, **params):
    """Search until a certified Barker sequence is found or time runs out."""
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), 'little')
    rng = np.random.default_rng(seed)
    margin = params.get('margin', 1e-7)
    t0 = time.time()
    deadline = t0 + time_limit
    best_pk, best_phi = math.inf, None
    runs = descents = 0
    while time.time() < deadline and not (stop_event is not None and stop_event.is_set()):
        pk, phi, d = attempt(N, rng, deadline, **params)
        runs += 1
        descents += d
        if pk < best_pk:
            best_pk, best_phi = pk, phi
        if pk <= 1.0 - margin:
            break
    res = dict(N=N, found=bool(best_pk <= 1.0 - margin), interior_peak=float(best_pk),
               time=time.time() - t0, runs=runs, level_descents=descents, seed=seed,
               params=params, H=H_DEFAULT)
    if best_phi is not None:
        m = quantize(best_phi)
        res['m'] = m
        res['interior_peak_quantized'] = interior_peak(2 * np.pi * np.array(m, dtype=np.float64) / H_DEFAULT)
        if res['found']:
            res['certification'] = certify(m=m)
    return res


def _worker(args):
    N, time_limit, seed, params, ev = args
    r = search(N, time_limit, seed, ev, **params)
    if r['found'] and r.get('certification', {}).get('certified'):
        ev.set()
    return r


def search_parallel(N, time_limit=60.0, workers=None, seed=None, **params):
    """Independent searches on all cores; stops everyone when one is certified."""
    workers = workers or os.cpu_count()
    objective(np.zeros(4), 0, 2.0, 0.0)   # compile once before the workers start
    base = seed if seed is not None else int.from_bytes(os.urandom(4), 'little')
    ctx = mp.get_context('spawn')
    with ctx.Manager() as mgr:
        ev = mgr.Event()
        jobs = [(N, time_limit, (base + 104729 * i) % 2 ** 32, params, ev) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            from .search import _stop_workers_on_signal
            old = _stop_workers_on_signal(ex, mgr)
            try:
                results = list(ex.map(_worker, jobs))
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    certified = [r for r in results if r.get('certification', {}).get('certified')]
    best = certified[0] if certified else min(results, key=lambda r: r['interior_peak'])
    best['workers'] = workers
    best['runs'] = sum(r['runs'] for r in results)
    best['level_descents'] = sum(r['level_descents'] for r in results)
    return best
