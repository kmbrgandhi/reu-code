"""Discrete-alphabet search algorithms (binary and m-phase), compiled with numba.

These are ports of the 2017 REU_code heuristics onto the incremental kernels,
with the 2017 bugs fixed (see README.md, "Changes from the 2017 code"), plus a
self-avoiding walk, which is the strongest published approach for long
low-autocorrelation binary sequences.

Every algorithm has the signature

    algo(P, S, prm, target, deadline, B) -> (best_cost, best_peak, evals, restarts)

P, S      problem and state tuples (see kernels.py)
prm       float64 parameter vector; names and defaults live in search.py
target    stop as soon as the peak objective is <= target
deadline  absolute time.perf_counter() value at which to stop
B         buffers (xi, vals, Ci, Cc) that receive the best state found

Each call keeps restarting until it reaches the target, the deadline, or its
restart limit (prm[0]; 0 means unlimited).
"""
import numpy as np
from numba import njit

from .kernels import (IP_NV, IP_NMOVES, TOL, apply_move, eval_move, full_cost, init_entries,
                      inverse_t, is_binary, load_code, n_free, now, peak_value, random_free_var,
                      random_init, random_t, restore_state, save_state, state_hash, move_hash,
                      hs_clear, hs_insert, hs_contains)

INF = np.inf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _improve(P, S, cur, best, best_peak, B, target):
    """Save S into B if it beats `best`. Returns (best, best_peak, target_hit)."""
    if cur < best - TOL:
        save_state(S, B[0], B[1], B[2], B[3])
        best = cur
        best_peak = peak_value(P, S)
    return best, best_peak, best_peak <= target + 1e-9


@njit(cache=True)
def _done(r, max_restarts, deadline):
    return (max_restarts > 0 and r >= max_restarts) or now() > deadline


@njit(cache=True)
def _resync(P, S, counter):
    """Rebuild complex entries now and then to stop rounding drift."""
    if (not is_binary(P[0])) and (counter & 4095) == 0:
        init_entries(P, S)


@njit(cache=True)
def descend_steepest(P, S, cur):
    """Steepest descent: apply the best improving single move until none
    exists (the 2017 'sdls' local search). Returns (cost, evals)."""
    nv = P[0][IP_NV]
    nm = P[0][IP_NMOVES]
    fr = P[4]
    ev = 0
    while True:
        bc = cur
        bv = -1
        bt = 0
        off = np.random.randint(0, nv)
        for vv in range(nv):
            v = vv + off
            if v >= nv:
                v -= nv
            if fr[v]:
                continue
            for t in range(1, nm + 1):
                c = eval_move(P, S, v, t, bc)
                ev += 1
                if c < bc - TOL:
                    bc = c
                    bv = v
                    bt = t
        if bv < 0:
            return cur, ev
        apply_move(P, S, bv, bt)
        cur = bc


@njit(cache=True)
def descend_first(P, S, cur):
    """First-improvement descent in random order (the 2017 'simple' climb)."""
    nv = P[0][IP_NV]
    nm = P[0][IP_NMOVES]
    fr = P[4]
    ev = 0
    improved = True
    while improved:
        improved = False
        order = np.random.permutation(nv)
        for v in order:
            if fr[v]:
                continue
            for t in range(1, nm + 1):
                c = eval_move(P, S, v, t, cur)
                ev += 1
                if c < cur - TOL:
                    apply_move(P, S, v, t)
                    cur = c
                    improved = True
                    break
    return cur, ev


@njit(cache=True)
def climb_once(P, S, cur):
    """One sweep: move to the best neighbor if it improves (2017 'climb_ineff'),
    trying one random phase change per position."""
    nv = P[0][IP_NV]
    fr = P[4]
    bc = cur
    bv = -1
    bt = 0
    for v in range(nv):
        if fr[v]:
            continue
        t = random_t(P)
        c = eval_move(P, S, v, t, bc)
        if c < bc - TOL:
            bc = c
            bv = v
            bt = t
    if bv >= 0:
        apply_move(P, S, bv, bt)
        return bc, nv
    return cur, nv


@njit(cache=True)
def random_kmove(P, S, k, mv_v, mv_t):
    """Apply k moves at distinct random free positions; remember them."""
    n = 0
    while n < k:
        v = random_free_var(P)
        dup = False
        for a in range(n):
            if mv_v[a] == v:
                dup = True
                break
        if dup:
            continue
        t = random_t(P)
        apply_move(P, S, v, t)
        mv_v[n] = v
        mv_t[n] = t
        n += 1


@njit(cache=True)
def revert_kmove(P, S, k, mv_v, mv_t):
    for a in range(k - 1, -1, -1):
        apply_move(P, S, mv_v[a], inverse_t(P, mv_t[a]))


@njit(cache=True)
def tabu_search(P, S, cur, max_iters, min_ten, extra_ten, tabu_until, W, target):
    """Tabu search in the style of Gallardo, Cotta & Fernandez (2009).

    Each step applies the best move whose position is not tabu (a tabu move is
    allowed if it beats the best cost of this search). The moved position stays
    tabu for min_ten + U{0..extra_ten} steps. Ends in the best state seen.
    W holds work buffers. Returns (best_cost, evals).
    """
    nv = P[0][IP_NV]
    nm = P[0][IP_NMOVES]
    fr = P[4]
    best = cur
    save_state(S, W[0], W[1], W[2], W[3])
    tabu_until[:] = 0
    ev = 0
    for it in range(max_iters):
        bc = INF
        bv = -1
        bt = 0
        off = np.random.randint(0, nv)
        for vv in range(nv):
            v = vv + off
            if v >= nv:
                v -= nv
            if fr[v]:
                continue
            allowed = it >= tabu_until[v]
            for t in range(1, nm + 1):
                bound = bc if allowed else min(bc, best - TOL)
                c = eval_move(P, S, v, t, bound)
                ev += 1
                if c < bc and (allowed or c < best - TOL):
                    bc = c
                    bv = v
                    bt = t
        if bv < 0:
            break
        apply_move(P, S, bv, bt)
        cur = bc
        tabu_until[bv] = it + min_ten + np.random.randint(0, extra_ten + 1)
        if cur < best - TOL:
            best = cur
            save_state(S, W[0], W[1], W[2], W[3])
            if peak_value(P, S) <= target + 1e-9:
                break
    restore_state(S, W[0], W[1], W[2], W[3])
    return best, ev


@njit(cache=True)
def _alloc_like(S):
    return (S[0].copy(), S[1].copy(), S[2].copy(), S[3].copy())


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

@njit(cache=True)
def random_search(P, S, prm, target, deadline, B):
    """Random sampling baseline (2017 rand_select_algorithm_doppler)."""
    max_r = int(prm[0])
    best = INF
    bp = INF
    ev = 0
    r = 0
    while True:
        random_init(P, S)
        c = full_cost(P, S)
        ev += 1
        r += 1
        best, bp, hit = _improve(P, S, c, best, bp, B, target)
        if hit:
            break
        if (r & 255) == 0 and _done(r, max_r, deadline):
            break
        if max_r > 0 and r >= max_r:
            break
    return best, bp, ev, r


@njit(cache=True)
def sdls(P, S, prm, target, deadline, B):
    """Restarted steepest descent (2017 hill_climbing_iterative + sdls_local_search)."""
    max_r = int(prm[0])
    best = INF
    bp = INF
    ev = 0
    r = 0
    while True:
        random_init(P, S)
        cur, e = descend_steepest(P, S, full_cost(P, S))
        ev += e
        r += 1
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def first_improvement(P, S, prm, target, deadline, B):
    """Restarted first-improvement climbing (2017 simple_local_search)."""
    max_r = int(prm[0])
    best = INF
    bp = INF
    ev = 0
    r = 0
    while True:
        random_init(P, S)
        cur, e = descend_first(P, S, full_cost(P, S))
        ev += e
        r += 1
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def stochastic_climb(P, S, prm, target, deadline, B):
    """2017 stochastic_local_search: `iters` steps; each step tries positions in
    random order with a random phase change and takes the first move that is
    better, or worse with probability exp((old - new) / temp).
    prm: [max_restarts, iters=40, temp=2]"""
    max_r = int(prm[0])
    iters = int(prm[1])
    temp = prm[2]
    nv = P[0][IP_NV]
    fr = P[4]
    best = INF
    bp = INF
    ev = 0
    r = 0
    while True:
        random_init(P, S)
        cur = full_cost(P, S)
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        for it in range(iters):
            order = np.random.permutation(nv)
            for v in order:
                if fr[v]:
                    continue
                t = random_t(P)
                c = eval_move(P, S, v, t, INF)
                ev += 1
                if c < cur or np.random.random() < np.exp((cur - c) / temp):
                    apply_move(P, S, v, t)
                    cur = c
                    break
            best, bp, hit = _improve(P, S, cur, best, bp, B, target)
            if hit:
                break
        r += 1
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def _temperature(sched, T0, Tf, n, j, beta):
    """The ten cooling schedules from 2017 anneal_full.py (beta < 0 = 2017 default)."""
    if sched == 0:      # semi_exponential (power law)
        b = 1.3 if beta < 0 else beta
        return T0 / (j ** b)
    elif sched == 1:    # mult_exponential
        b = 1.3 if beta < 0 else beta
        return T0 / (b ** j)
    elif sched == 2:    # log_mult
        b = 1.3 if beta < 0 else beta
        return T0 / (1.0 + b * np.log(1.0 + j))
    elif sched == 3:    # lin_mult
        b = 0.8 if beta < 0 else beta
        return T0 / (1.0 + b * j)
    elif sched == 4:    # quad_mult
        b = 0.8 if beta < 0 else beta
        return T0 / (1.0 + b * j * j)
    elif sched == 5:    # lin_add
        return Tf + (T0 - Tf) * ((n - j) / n)
    elif sched == 6:    # quad_add
        return Tf + (T0 - Tf) * ((n - j) / n) ** 2
    elif sched == 7:    # expon_add
        return Tf + (T0 - Tf) / (1.0 + np.exp(2.0 * np.log(max(T0 - Tf, 1.0 + 1e-9)) * (j - 0.5 * n) / n))
    elif sched == 8:    # trigon_add
        return Tf + (T0 - Tf) * 0.5 * (1.0 + np.cos(j * np.pi / n))
    # quenching
    return T0 / np.exp((1.0 - 0.7) * j)


@njit(cache=True)
def anneal(P, S, prm, target, deadline, B):
    """Simulated annealing (2017 anneal / anneal_with_tabu / anneal_with_local_search).

    prm: [max_restarts, t0_factor=2, gen_factor=25, ntemp_factor=2.9, kmove=3,
          alpha=1.5, schedule=0, beta=-1, final_temp=0.25, stuck_kappa=0.2,
          stuck_mult=2, tabu=0, tabu_len=100, local=0]

    T0 = t0_factor * n, where n is the number of variables. Each temperature
    runs gen_factor * n proposals; a proposal changes kmove random positions.
    A worse proposal is accepted with probability exp(-((new - cur) / T)^alpha).
    An attempt ends when stuck_mult * n^stuck_kappa proposals in a row are
    rejected, as in 2017. tabu=1 skips recently visited states; local=1 runs a
    first-improvement descent after each proposal.
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    nf = n_free(P)
    T0 = prm[1] * nv
    ngen = max(1, int(prm[2] * nv))
    ntemps = max(2, int(prm[3] * T0))
    k = int(min(prm[4], max(nf - 1, 1)))
    alpha = prm[5]
    sched = int(prm[6])
    beta = prm[7]
    Tf = prm[8]
    upp = prm[10] * nv ** prm[9]
    use_tabu = prm[11] > 0
    tabu_len = max(1, int(prm[12]))
    local = prm[13] > 0
    mv_v = np.zeros(k, dtype=np.int64)
    mv_t = np.zeros(k, dtype=np.int64)
    snap = _alloc_like(S)
    ring = np.zeros(tabu_len, dtype=np.uint64)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        random_init(P, S)
        cur = full_cost(P, S)
        ev += 1
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit:
            break
        T = T0
        stuck = 0
        ring[:] = 0
        rn = 0
        for j in range(1, ntemps):
            for i in range(ngen):
                if local:
                    save_state(S, snap[0], snap[1], snap[2], snap[3])
                random_kmove(P, S, k, mv_v, mv_t)
                c = full_cost(P, S)
                if local:
                    c, e = descend_first(P, S, c)
                    ev += e
                ev += 1
                h = np.uint64(0)
                if use_tabu:
                    h = state_hash(P, S)
                    seen = False
                    for a in range(tabu_len):
                        if ring[a] == h:
                            seen = True
                            break
                    if seen:
                        if local:
                            restore_state(S, snap[0], snap[1], snap[2], snap[3])
                        else:
                            revert_kmove(P, S, k, mv_v, mv_t)
                        continue
                if c < cur or np.random.random() < np.exp(-(((c - cur) / T) ** alpha)):
                    cur = c
                    stuck = 0
                    best, bp, hit = _improve(P, S, cur, best, bp, B, target)
                    if hit:
                        stop = True
                        break
                else:
                    if local:
                        restore_state(S, snap[0], snap[1], snap[2], snap[3])
                    else:
                        revert_kmove(P, S, k, mv_v, mv_t)
                    stuck += 1
                if use_tabu:
                    ring[rn % tabu_len] = h
                    rn += 1
                _resync(P, S, ev)
                if stuck > upp:
                    break
                if (ev & 1023) == 0 and now() > deadline:
                    stop = True
                    break
            if stop or stuck > upp:
                break
            T = max(_temperature(sched, T0, Tf, ntemps, j, beta), 1e-12)
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def threshold_accepting(P, S, prm, target, deadline, B):
    """Threshold accepting (2017 threshold_accept.py).

    prm: [max_restarts, gen_factor=30, kmove=3, beta=1.3, init_mult=2,
          stuck_kappa=0.2, stuck_mult=2, nthr_factor=0.5]

    As in 2017, a proposal is accepted when its cost is below an absolute
    threshold that starts at init_mult * (initial cost). The threshold during
    generation j is thr0 / j^beta. (The 2017 code divided the current threshold
    by j^beta every generation, so it fell like 1/(j!)^beta and stopped
    accepting moves after a few generations.)
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    nf = n_free(P)
    ngen = max(1, int(prm[1] * nv))
    k = int(min(prm[2], max(nf - 1, 1)))
    beta = prm[3]
    upp = prm[6] * nv ** prm[5]
    nthr = max(2, int(prm[7] * nv))
    mv_v = np.zeros(k, dtype=np.int64)
    mv_t = np.zeros(k, dtype=np.int64)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        random_init(P, S)
        cur = full_cost(P, S)
        thr0 = prm[4] * cur
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit:
            break
        stuck = 0
        for j in range(1, nthr):
            thr = thr0 / (j ** beta)
            for i in range(ngen):
                random_kmove(P, S, k, mv_v, mv_t)
                c = full_cost(P, S)
                ev += 1
                if c < thr:
                    cur = c
                    stuck = 0
                    best, bp, hit = _improve(P, S, cur, best, bp, B, target)
                    if hit:
                        stop = True
                        break
                else:
                    revert_kmove(P, S, k, mv_v, mv_t)
                    stuck += 1
                _resync(P, S, ev)
                if stuck > upp:
                    break
                if (ev & 1023) == 0 and now() > deadline:
                    stop = True
                    break
            if stop or stuck > upp:
                break
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def _ts_params(nv, prm_iters, prm_min, prm_extra):
    if prm_iters > 0:
        iters = int(prm_iters)
    else:  # random in [n/2, 3n/2]
        iters = nv // 2 + np.random.randint(0, nv + 1)
    mn = int(prm_min) if prm_min > 0 else max(1, iters // 10)
    ex = int(prm_extra) if prm_extra >= 0 else max(1, iters // 50)
    return max(iters, 1), mn, ex


@njit(cache=True)
def tabu(P, S, prm, target, deadline, B):
    """Restarted tabu search. prm: [max_restarts, max_iters=0 (auto), min_tenure=0
    (auto), extra_tenure=-1 (auto)]. Auto values: max_iters uniform in
    [n/2, 3n/2], min_tenure = max_iters/10, extra_tenure = max_iters/50."""
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    W = _alloc_like(S)
    tabu_until = np.zeros(nv, dtype=np.int64)
    best = INF
    bp = INF
    ev = 0
    r = 0
    while True:
        random_init(P, S)
        iters, mn, ex = _ts_params(nv, prm[1], prm[2], prm[3])
        cur, e = tabu_search(P, S, full_cost(P, S), iters, mn, ex, tabu_until, W, target)
        ev += e
        r += 1
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def self_avoiding_walk(P, S, prm, target, deadline, B):
    """Self-avoiding walk (the idea behind lssOrel, Boskovic, Brglez & Brest 2017).

    From a random start, repeatedly move to the best neighbor that the walk has
    not visited (Zobrist hashes in a hash set), for walk_factor * n steps, then
    restart. prm: [max_restarts, walk_factor=64]
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    nm = P[0][IP_NMOVES]
    fr = P[4]
    wl = max(1, int(prm[1] * nv))
    cap = 1
    while cap < 4 * wl + 16:
        cap *= 2
    tab = np.zeros(cap, dtype=np.uint64)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        random_init(P, S)
        cur = full_cost(P, S)
        hs_clear(tab)
        h = state_hash(P, S)
        hs_insert(tab, h)
        best, bp, hit = _improve(P, S, cur, best, bp, B, target)
        if hit:
            break
        for step in range(wl):
            bc = INF
            bv = -1
            bt = 0
            bh = np.uint64(0)
            for v in range(nv):
                if fr[v]:
                    continue
                for t in range(1, nm + 1):
                    nh = move_hash(P, S, h, v, t)
                    if hs_contains(tab, nh):
                        continue
                    c = eval_move(P, S, v, t, bc)
                    ev += 1
                    if c < bc:
                        bc = c
                        bv = v
                        bt = t
                        bh = nh
            if bv < 0:
                break
            apply_move(P, S, bv, bt)
            cur = bc
            h = bh
            hs_insert(tab, h)
            best, bp, hit = _improve(P, S, cur, best, bp, B, target)
            if hit:
                stop = True
                break
            _resync(P, S, step + 1)
            if (step & 63) == 0 and now() > deadline:
                stop = True
                break
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


# ---------------------------------------------------------------------------
# Population-based algorithms
# ---------------------------------------------------------------------------

@njit(cache=True)
def _store(S, popx, popz, i):
    popx[i, :] = S[0]
    popz[i, :] = S[1]


@njit(cache=True)
def _tournament(popc, n, k):
    best = np.random.randint(0, n)
    for a in range(1, k):
        c = np.random.randint(0, n)
        if popc[c] < popc[best]:
            best = c
    return best


@njit(cache=True)
def _crossover(P, popx, popz, a, b, mode, cx, cz):
    """mode 0 uniform, 1 half-half, 2 random point (2017 crossover_algs.py)."""
    nv = P[0][IP_NV]
    point = nv // 2
    if mode == 2:
        point = np.random.randint(0, nv)
    for v in range(nv):
        if mode == 0:
            take_a = np.random.random() < 0.5
        else:
            take_a = v < point
        if take_a:
            cx[v] = popx[a, v]
            cz[v] = popz[a, v]
        else:
            cx[v] = popx[b, v]
            cz[v] = popz[b, v]


@njit(cache=True)
def _mutate_k(P, S, k):
    """Change k distinct random free positions (state entries are updated)."""
    nf = n_free(P)
    k = min(k, nf)
    mv_v = np.zeros(k, dtype=np.int64)
    mv_t = np.zeros(k, dtype=np.int64)
    random_kmove(P, S, k, mv_v, mv_t)


@njit(cache=True)
def memetic(P, S, prm, target, deadline, B):
    """Memetic algorithm with tabu-search local improvement (2017
    run_evolution_memet, after Gallardo, Cotta & Fernandez 2009).

    prm: [max_runs, pop_size=100, n_gens=30, offspring=500, p_cross=0.1,
          p_mut=-1 (= 1/n), mut_k=2, restart_every=8, restart_mult=1.5,
          ts_iters=20, ts_min=2, ts_extra=1]

    Each child is a tournament-selected parent (or, with probability p_cross,
    a uniform crossover of two parents), mutated at mut_k positions with
    probability p_mut, then improved by tabu search. The children replace the
    population, as in 2017. Every restart_every generations, restart_mult * L
    random members are added. Set ts_iters = 0 for the automatic tabu length.
    local_search: 0 tabu search, 1 steepest descent, 2 first improvement
    (the three 2017 memetic variants).
    """
    max_r = int(prm[0])
    ip = P[0]
    nv = ip[IP_NV]
    L = ip[2]
    pop0 = int(prm[1])
    ngens = int(prm[2])
    offspring = int(prm[3])
    p_cross = prm[4]
    p_mut = prm[5] if prm[5] >= 0 else 1.0 / nv
    mut_k = int(prm[6])
    r_every = max(1, int(prm[7]))
    r_add = int(prm[8] * L)
    cap = max(offspring, pop0) + r_add + 1
    popx = np.zeros((cap, nv), dtype=np.int64)
    popz = np.zeros((cap, nv), dtype=np.complex128)
    popc = np.zeros(cap)
    chx = np.zeros((cap, nv), dtype=np.int64)
    chz = np.zeros((cap, nv), dtype=np.complex128)
    chc = np.zeros(cap)
    cx = np.zeros(nv, dtype=np.int64)
    cz = np.zeros(nv, dtype=np.complex128)
    W = _alloc_like(S)
    tabu_until = np.zeros(nv, dtype=np.int64)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        n = pop0
        for i in range(n):
            random_init(P, S)
            popc[i] = full_cost(P, S)
            _store(S, popx, popz, i)
            best, bp, hit = _improve(P, S, popc[i], best, bp, B, target)
            if hit:
                break
        if hit:
            break
        for g in range(ngens):
            if g % r_every == 0:
                for i in range(n, n + r_add):
                    random_init(P, S)
                    popc[i] = full_cost(P, S)
                    _store(S, popx, popz, i)
                n += r_add
            nch = min(offspring, n)
            for ci in range(nch):
                if np.random.random() > p_cross:
                    a = _tournament(popc, n, 2)
                    cx[:] = popx[a]
                    cz[:] = popz[a]
                else:
                    a = _tournament(popc, n, 2)
                    b = _tournament(popc, n, 2)
                    _crossover(P, popx, popz, a, b, 0, cx, cz)
                load_code(P, S, cx, cz)
                if np.random.random() < p_mut:
                    _mutate_k(P, S, mut_k)
                ls = int(prm[12])
                if ls == 1:
                    c, e = descend_steepest(P, S, full_cost(P, S))
                elif ls == 2:
                    c, e = descend_first(P, S, full_cost(P, S))
                else:
                    iters, mn, ex = _ts_params(nv, prm[9], prm[10], prm[11])
                    c, e = tabu_search(P, S, full_cost(P, S), iters, mn, ex, tabu_until, W, target)
                ev += e
                chc[ci] = c
                _store(S, chx, chz, ci)
                best, bp, hit = _improve(P, S, c, best, bp, B, target)
                if hit or ((ci & 7) == 0 and now() > deadline):
                    stop = True
                    break
            if stop:
                break
            n = nch
            popx[:n] = chx[:n]
            popz[:n] = chz[:n]
            popc[:n] = chc[:n]
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def _sort_truncate(popx, popz, popc, n, keep):
    order = np.argsort(popc[:n])
    nx = popx[order[:keep]].copy()
    nz = popz[order[:keep]].copy()
    nc = popc[order[:keep]].copy()
    popx[:keep] = nx
    popz[:keep] = nz
    popc[:keep] = nc


@njit(cache=True)
def evolution(P, S, prm, target, deadline, B):
    """(mu + lambda) evolution (2017 evolut_full.py).

    prm: [max_runs, mode=1, pop=100, n_gens=25, offspring=500, mut_k=3,
          restart_every=8, restart_count=15]

    mode 1 ('smart', evolut_algo): members are improved by one best-neighbor
    sweep; children are a random parent mutated at mut_k positions and then
    improved the same way; random improved members are added every
    restart_every generations.
    mode 0 ('naive', evolut_algo_naive): children are random-point crossovers
    of two random parents, with no mutation or local search.
    The best `pop` of parents + children survive.
    """
    max_r = int(prm[0])
    smart = prm[1] > 0
    nv = P[0][IP_NV]
    pop = int(prm[2])
    ngens = int(prm[3])
    off = int(prm[4])
    mut_k = int(prm[5])
    r_every = max(1, int(prm[6]))
    r_add = int(prm[7])
    cap = pop + off + r_add + 1
    popx = np.zeros((cap, nv), dtype=np.int64)
    popz = np.zeros((cap, nv), dtype=np.complex128)
    popc = np.zeros(cap)
    cx = np.zeros(nv, dtype=np.int64)
    cz = np.zeros(nv, dtype=np.complex128)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    while True:
        n = pop
        for i in range(n):
            random_init(P, S)
            c = full_cost(P, S)
            if smart:
                c, e = climb_once(P, S, c)
                ev += e
            popc[i] = c
            _store(S, popx, popz, i)
            best, bp, hit = _improve(P, S, c, best, bp, B, target)
            if hit:
                break
        if hit:
            break
        for g in range(1, ngens + 1):
            if smart and g % r_every == 0:
                for i in range(n, n + r_add):
                    random_init(P, S)
                    c, e = climb_once(P, S, full_cost(P, S))
                    ev += e
                    popc[i] = c
                    _store(S, popx, popz, i)
                n += r_add
            m = n
            for ci in range(off):
                if smart:
                    a = np.random.randint(0, n)
                    load_code(P, S, popx[a], popz[a])
                    _mutate_k(P, S, mut_k)
                    c, e = climb_once(P, S, full_cost(P, S))
                    ev += e
                else:
                    a = np.random.randint(0, n)
                    b = np.random.randint(0, n)
                    while b == a and n > 1:
                        b = np.random.randint(0, n)
                    _crossover(P, popx, popz, a, b, 2, cx, cz)
                    load_code(P, S, cx, cz)
                    c = full_cost(P, S)
                    ev += 1
                popc[m] = c
                _store(S, popx, popz, m)
                m += 1
                best, bp, hit = _improve(P, S, c, best, bp, B, target)
                if hit or ((ci & 15) == 0 and now() > deadline):
                    stop = True
                    break
            if stop:
                break
            _sort_truncate(popx, popz, popc, m, pop)
            n = pop
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r


@njit(cache=True)
def genetic(P, S, prm, target, deadline, B):
    """Generational genetic algorithm (2017 genetic_full.run_evolution).

    prm: [max_runs, pop=100, n_gens=25, selection=0, crossover=0, retain=0.2,
          random_select=0.05, mutate=0.02, k=2]

    selection: 0 elitist + random survivors (select_parents_part_random),
               1 fitness-proportional (select_parents_random),
               2 tournament of size k (tournament_selection).
    crossover: 0 uniform, 1 half-half, 2 random point.
    Selected parents are mutated at one position with probability `mutate`,
    and crossover children refill the population.
    """
    max_r = int(prm[0])
    nv = P[0][IP_NV]
    pop = max(int(prm[1]), 4)
    ngens = int(prm[2])
    sel = int(prm[3])
    xmode = int(prm[4])
    retain = prm[5]
    rsel = prm[6]
    pm = prm[7]
    k = max(1, int(prm[8]))
    popx = np.zeros((pop, nv), dtype=np.int64)
    popz = np.zeros((pop, nv), dtype=np.complex128)
    popc = np.zeros(pop)
    parx = np.zeros((pop, nv), dtype=np.int64)
    parz = np.zeros((pop, nv), dtype=np.complex128)
    cx = np.zeros(nv, dtype=np.int64)
    cz = np.zeros(nv, dtype=np.complex128)
    best = INF
    bp = INF
    ev = 0
    r = 0
    hit = False
    stop = False
    nret = max(2, int(pop * retain))
    while True:
        for i in range(pop):
            random_init(P, S)
            popc[i] = full_cost(P, S)
            ev += 1
            _store(S, popx, popz, i)
            best, bp, hit = _improve(P, S, popc[i], best, bp, B, target)
            if hit:
                break
        if hit:
            break
        for g in range(ngens):
            # --- parent selection
            npar = 0
            order = np.argsort(popc)
            if sel == 0:
                for a in range(pop):
                    i = order[a]
                    if a < nret or np.random.random() < rsel:
                        parx[npar] = popx[i]
                        parz[npar] = popz[i]
                        npar += 1
            elif sel == 1:
                cmin = popc[order[0]]
                tries = 0
                while npar < nret and tries < 1000 * pop:
                    tries += 1
                    i = np.random.randint(0, pop)
                    prob = 1.0 if popc[i] <= 0 else cmin / popc[i]
                    if np.random.random() < prob:
                        parx[npar] = popx[i]
                        parz[npar] = popz[i]
                        npar += 1
            else:
                for a in range(nret):
                    i = _tournament(popc, pop, k)
                    parx[npar] = popx[i]
                    parz[npar] = popz[i]
                    npar += 1
            while npar < 2:
                i = np.random.randint(0, pop)
                parx[npar] = popx[i]
                parz[npar] = popz[i]
                npar += 1
            # --- mutation of parents
            for a in range(npar):
                if np.random.random() < pm:
                    load_code(P, S, parx[a], parz[a])
                    _mutate_k(P, S, 1)
                    parx[a] = S[0]
                    parz[a] = S[1]
            # --- new population: parents + crossover children
            for a in range(pop):
                if a < npar:
                    cx[:] = parx[a]
                    cz[:] = parz[a]
                else:
                    f = np.random.randint(0, npar)
                    mo = np.random.randint(0, npar)
                    while mo == f:
                        mo = np.random.randint(0, npar)
                    _crossover(P, parx, parz, f, mo, xmode, cx, cz)
                load_code(P, S, cx, cz)
                c = full_cost(P, S)
                ev += 1
                popc[a] = c
                _store(S, popx, popz, a)
                best, bp, hit = _improve(P, S, c, best, bp, B, target)
                if hit:
                    stop = True
                    break
            if stop or now() > deadline:
                stop = True
                break
        r += 1
        if stop or hit or _done(r, max_r, deadline):
            break
    return best, bp, ev, r
