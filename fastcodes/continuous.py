"""Continuous-phase (arbitrary polyphase) algorithms, compiled with numba.

Ports of the 2017 polyphase.py and particle_swarm.py. They work for any
complex problem kind built with phases=0: single codes with Doppler bins,
complementary sets, and polyphase type-2 sets. The signature matches
discrete.py.
"""
import numpy as np
from numba import njit

from .kernels import (IP_NV, apply_value, eval_value, full_cost, init_entries, now,
                      restore_state, save_state)
from .discrete import _improve, _done, _alloc_like

INF = np.inf


@njit(cache=True)
def _set_phases(P, S, ph):
    fr = P[4]
    vals = S[1]
    for v in range(ph.shape[0]):
        if fr[v]:
            ph[v] = 0.0
        vals[v] = np.cos(ph[v]) + 1j * np.sin(ph[v])
    init_entries(P, S)
    return full_cost(P, S)


@njit(cache=True)
def _random_phases(P, ph, lo, hi):
    fr = P[4]
    for v in range(ph.shape[0]):
        ph[v] = 0.0 if fr[v] else lo + (hi - lo) * np.random.random()


@njit(cache=True)
def _norm(x):
    return np.sqrt(np.sum(x * x))


@njit(cache=True)
def _deluge(P, S, ph, steps, cur, min_step, divisor, rain, best, bp, B, target, deadline,
            W, Wph, keep_local_best):
    """One great-deluge run (2017 polyphase.great_deluge).

    Each sweep tries ph[v] + steps[v] for every position. A move is accepted
    when its cost is below the water level; every accepted move lowers the
    level by `rain`. A rejected move reverses the direction of that step. After
    two sweeps in a row without an accepted move, all steps shrink by
    `divisor`. The run ends when the step-vector norm drops below min_step.

    Returns (cur, best, bp, hit, evals, local_best). When keep_local_best is
    set, the best state of this run is restored into S/ph at the end.
    """
    nv = P[0][IP_NV]
    fr = P[4]
    water = cur
    unsucc = 0
    ev = 0
    hit = False
    local_best = cur
    if keep_local_best:
        save_state(S, W[0], W[1], W[2], W[3])
        Wph[:] = ph
    while _norm(steps) > min_step:
        dry = 0
        for v in range(nv):
            if fr[v]:
                continue
            newp = ph[v] + steps[v]
            z = np.cos(newp) + 1j * np.sin(newp)
            c = eval_value(P, S, v, z, water)
            ev += 1
            if c < water:
                apply_value(P, S, v, z)
                ph[v] = newp
                cur = c
                dry += 1
                water -= rain
                best, bp, hit = _improve(P, S, cur, best, bp, B, target)
                if keep_local_best and cur < local_best:
                    local_best = cur
                    save_state(S, W[0], W[1], W[2], W[3])
                    Wph[:] = ph
                if hit:
                    break
            else:
                steps[v] = -steps[v]
        if hit:
            break
        if dry == 0:
            unsucc += 1
            if unsucc == 2:
                steps /= divisor
                unsucc = 0
        else:
            unsucc = 0
        if now() > deadline:
            break
    if keep_local_best:
        restore_state(S, W[0], W[1], W[2], W[3])
        ph[:] = Wph
        cur = local_best
    init_entries(P, S)  # resync after many complex updates
    cur = full_cost(P, S)
    return cur, best, bp, hit, ev, local_best


@njit(cache=True)
def great_deluge(P, S, prm, target, deadline, B):
    """2017 great_deluge_handler (after Friese & Zottmann 1994).

    prm: [max_restarts, n_inner=-1 (= int(10 sqrt(n))), min_step=1e-5,
          divisor=1.3, rain=0.03]
    From each random start, run n_inner deluges, each with a fresh random step
    vector (uniform in [0, pi) per position), continuing from the current code.
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    n_inner = int(prm[1]) if prm[1] > 0 else int(10 * np.sqrt(nv))
    ph = np.zeros(nv)
    steps = np.zeros(nv)
    W = _alloc_like(S)
    Wph = np.zeros(nv)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    while True:
        _random_phases(P, ph, 0.0, 2 * np.pi)
        cur = _set_phases(P, S, ph)
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        for it in range(n_inner):
            if hit:
                break
            _random_phases(P, steps, 0.0, np.pi)
            cur, best, bp, hit, e, lb = _deluge(P, S, ph, steps, cur, prm[2], prm[3], prm[4],
                                               best, bp, B, target, deadline, W, Wph, False)
            ev += e
            if now() > deadline:
                break
        r += 1
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def particle_swarm(P, S, prm, target, deadline, B):
    """2017 particle_swarm_polyphase.

    prm: [max_restarts, swarm_mult=20, omega=1, c1=2, c2=2, vnorm=pi, max_iters=0]
    Swarm size is swarm_mult * n. Velocities are rescaled to norm `vnorm` each
    step, as in 2017. Each particle is evaluated in full, because every
    position moves at once. max_iters = 0 runs until the deadline.
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    fr = P[4]
    ns = max(2, int(prm[1] * nv))
    omega, c1, c2, vnorm = prm[2], prm[3], prm[4], prm[5]
    max_it = int(prm[6])
    X = np.zeros((ns, nv))
    V = np.zeros((ns, nv))
    PB = np.zeros((ns, nv))
    PBc = np.zeros(ns)
    G = np.zeros(nv)
    tmp = np.zeros(nv)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    while True:
        Gc = INF
        for i in range(ns):
            _random_phases(P, tmp, 0.0, 2 * np.pi)
            X[i] = tmp
            for v in range(nv):
                V[i, v] = 0.0 if fr[v] else np.random.random()
            c = _set_phases(P, S, tmp)
            ev += 1
            PB[i] = X[i]
            PBc[i] = c
            if c < Gc:
                Gc = c
                G[:] = X[i]
            best, bp, hit = _improve(P, S, c, best, bp, B, target)
            if hit:
                break
        it = 0
        while not hit:
            for i in range(ns):
                r1 = c1 * np.random.random()
                r2 = c2 * np.random.random()
                for v in range(nv):
                    if fr[v]:
                        continue
                    V[i, v] = omega * V[i, v] + r1 * (PB[i, v] - X[i, v]) + r2 * (G[v] - X[i, v])
                nrm = _norm(V[i])
                if nrm > 0:
                    V[i] *= vnorm / nrm
                for v in range(nv):
                    X[i, v] = np.fmod(X[i, v] + V[i, v], 2 * np.pi)
                tmp[:] = X[i]
                c = _set_phases(P, S, tmp)
                ev += 1
                if c < PBc[i]:
                    PBc[i] = c
                    PB[i] = X[i]
                best, bp, hit = _improve(P, S, c, best, bp, B, target)
                if hit:
                    break
            for i in range(ns):  # global best is refreshed after each sweep, as in 2017
                if PBc[i] < Gc:
                    Gc = PBc[i]
                    G[:] = PB[i]
            it += 1
            if (max_it > 0 and it >= max_it) or now() > deadline:
                break
        r += 1
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def anneal_phases(P, S, prm, target, deadline, B):
    """2017 anneal_phases_handler: annealing in continuous phase space with a
    short great-deluge local search after every proposal.

    prm: [max_restarts, t0_factor=1, gen_factor=1.5, ntemp_factor=2,
          step_norm=2, alpha=1.5, beta=1.3, stuck_kappa=0.2, stuck_mult=2,
          ls_min_step=0.01, ls_divisor=1.6, ls_rain=0.4]
    A proposal adds a random vector of norm step_norm (positive components,
    as in 2017) to all phases. Temperature follows T0 / j^beta.
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    T0 = prm[1] * nv
    ngen = max(1, int(prm[2] * nv))
    ntemps = max(2, int(prm[3] * T0))
    step_norm, alpha, beta = prm[4], prm[5], prm[6]
    upp = prm[8] * nv ** prm[7]
    ph = np.zeros(nv)
    cand = np.zeros(nv)
    steps = np.zeros(nv)
    snap = _alloc_like(S)
    snap_ph = np.zeros(nv)
    W = _alloc_like(S)
    Wph = np.zeros(nv)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        _random_phases(P, ph, 0.0, 2 * np.pi)
        cur = _set_phases(P, S, ph)
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit:
            break
        T = T0
        stuck = 0
        for j in range(1, ntemps):
            for i in range(ngen):
                save_state(S, snap[0], snap[1], snap[2], snap[3])
                snap_ph[:] = ph
                _random_phases(P, steps, 0.0, np.pi)
                nrm = _norm(steps)
                if nrm > 0:
                    steps *= step_norm / nrm
                cand[:] = ph + steps
                ph[:] = cand
                c = _set_phases(P, S, ph)
                _random_phases(P, steps, 0.0, np.pi)
                c, best, bp, hit, e, lb = _deluge(P, S, ph, steps, c, prm[9], prm[10], prm[11],
                                                 best, bp, B, target, deadline, W, Wph, True)
                ev += e + 1
                if hit:
                    stop = True
                    break
                if c < cur or np.random.random() < np.exp(-(((c - cur) / T) ** alpha)):
                    cur = c
                    stuck = 0
                else:
                    restore_state(S, snap[0], snap[1], snap[2], snap[3])
                    ph[:] = snap_ph
                    stuck += 1
                if stuck > upp or now() > deadline:
                    break
            if stop or stuck > upp or now() > deadline:
                break
            T = max(T0 / (j ** beta), 1e-12)
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r
