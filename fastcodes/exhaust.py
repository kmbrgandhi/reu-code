"""Exhaustive meet-in-the-middle search for binary type-2 sets of 8 codes.

A set of 8 codes is type-2 exactly when the codes' fourth-order contribution
vectors sum to zero (see exact.code_vectors). Codes with the same vector are
interchangeable, so the search ranges over multisets of 8 distinct vectors.

Method
------
1. Hash each distinct vector to 64 bits with a random linear map
   h(v) = sum_d w_d v_d (mod 2^64). The map is linear, so a zero vector sum
   gives a zero hash sum (no false negatives). Every hash hit is re-checked
   exactly, so collisions can't produce false positives.
2. Form all pairs i <= j of distinct vectors and their hashes. Every 8-multiset
   splits into pairs (i1,i2), (i3,i4), (i5,i6), (i7,i8), so it is found as
   A + B = 0 with A = pair1 + pair2 and B = pair3 + pair4.
3. Split the work by the low bits of the hash: for each residue r, list every
   A whose hash is r (mod M) and every B whose hash is -r (mod M), and match
   them through a hash table. Each residue is independent, so the residues run
   in parallel on all cores.

The total work grows like (number of pairs)^2: about 3e11 pair combinations at
N = 13, 5e12 at N = 14, and 7e13 at N = 15.
"""
import os
import time

import numpy as np
from numba import njit, prange, get_num_threads

from . import verify
from .exact import code_vectors


@njit(cache=True)
def _vec_hash(V, w):
    U, D = V.shape
    h = np.zeros(U, dtype=np.uint64)
    for u in range(U):
        acc = np.uint64(0)
        for d in range(D):
            acc += w[d] * np.uint64(np.int64(V[u, d]))
        h[u] = acc
    return h


@njit(cache=True)
def _pairs(h):
    U = h.shape[0]
    P = U * (U + 1) // 2
    pi = np.empty(P, dtype=np.int32)
    pj = np.empty(P, dtype=np.int32)
    ph = np.empty(P, dtype=np.uint64)
    p = 0
    for i in range(U):
        for j in range(i, U):
            pi[p] = i
            pj[p] = j
            ph[p] = h[i] + h[j]
            p += 1
    return pi, pj, ph


@njit(cache=True)
def _gen(r, ph, low, order, start, M, out_v, out_a, out_b):
    """All pair-of-pairs (a <= b) whose residues sum to r modulo M. Returns count."""
    P = ph.shape[0]
    n = 0
    for a in range(P):
        t = (r - low[a]) % M
        for q in range(start[t], start[t + 1]):
            b = order[q]
            if b < a:
                continue
            if n < out_v.shape[0]:
                out_v[n] = ph[a] + ph[b]
                out_a[n] = a
                out_b[n] = b
            n += 1
    return n


@njit(cache=True)
def _residue(rr, M, ph, low, order, start, xv, xa, xb, yv, ya, yb, tab, tab_bits, hits, nh, t,
             max_hits):
    """Match residue rr against -rr. Hits go to hits[t] / nh[t]. Returns 0, or
    the buffer size needed if the buffers were too small (nothing is lost:
    the caller retries this residue with bigger buffers)."""
    sr = (M - rr) % M
    nx = _gen(rr, ph, low, order, start, M, xv, xa, xb)
    ny = nx if rr == sr else _gen(sr, ph, low, order, start, M, yv, ya, yb)
    if nx > xv.shape[0] or ny > yv.shape[0] or 2 * nx > tab.shape[0]:
        return max(nx, ny, 1)
    tmask = np.uint64(tab.shape[0] - 1)
    sh = np.uint64(64 - tab_bits)
    g = np.uint64(0x9E3779B97F4A7C15)
    for i in range(nx):
        s = (xv[i] * g) >> sh
        while tab[s] != -1:
            s = (s + np.uint64(1)) & tmask
        tab[s] = i
    for i in range(ny):
        v = yv[i] if rr != sr else xv[i]
        want = np.uint64(0) - v
        s = (want * g) >> sh
        while tab[s] != -1:
            j = tab[s]
            if xv[j] == want:
                if nh[t] < max_hits:
                    k = nh[t]
                    hits[t, k, 0] = xa[j]
                    hits[t, k, 1] = xb[j]
                    if rr != sr:
                        hits[t, k, 2] = ya[i]
                        hits[t, k, 3] = yb[i]
                    else:
                        hits[t, k, 2] = xa[i]
                        hits[t, k, 3] = xb[i]
                nh[t] += 1
            s = (s + np.uint64(1)) & tmask
    for i in range(nx):                 # clear only the slots we used
        s = (xv[i] * g) >> sh
        while tab[s] != -1:
            tab[s] = -1
            s = (s + np.uint64(1)) & tmask
    return 0


@njit(cache=True)
def _count(rr, M, ph, low, order, start):
    """Buffer size needed for residue rr and its partner -rr."""
    sr = (M - rr) % M
    empty_v = np.empty(0, dtype=np.uint64)
    empty_i = np.empty(0, dtype=np.int32)
    return max(_gen(rr, ph, low, order, start, M, empty_v, empty_i, empty_i),
               _gen(sr, ph, low, order, start, M, empty_v, empty_i, empty_i))


@njit(cache=True, parallel=True)
def _search(ph, low, order, start, M, cap, tab_bits, max_hits, r_lo, r_hi, max_over):
    """Residues r in [r_lo, r_hi) (each paired with -r), split across threads.
    Returns hits, hit counts, and the residues whose buffers overflowed."""
    T = get_num_threads()
    hits = np.full((T, max_hits, 4), -1, dtype=np.int64)
    nh = np.zeros(T, dtype=np.int64)
    over = np.full((T, max_over), -1, dtype=np.int64)
    nover = np.zeros(T, dtype=np.int64)
    half = M // 2
    for t in prange(T):
        xv = np.empty(cap, dtype=np.uint64)
        xa = np.empty(cap, dtype=np.int32)
        xb = np.empty(cap, dtype=np.int32)
        yv = np.empty(cap, dtype=np.uint64)
        ya = np.empty(cap, dtype=np.int32)
        yb = np.empty(cap, dtype=np.int32)
        tab = np.full(1 << tab_bits, -1, dtype=np.int64)
        for r in range(r_lo + t, min(r_hi, half + 1), T):
            need = _residue(np.int64(r), M, ph, low, order, start, xv, xa, xb, yv, ya, yb,
                            tab, tab_bits, hits, nh, t, max_hits)
            if need > 0:
                if nover[t] < max_over:
                    over[t, nover[t]] = r
                nover[t] += 1
    return hits, nh, over, nover


def _compact(hits, nh, max_hits):
    """The filled rows of a per-thread hit buffer, as one small array."""
    rows = [hits[t, :min(int(nh[t]), max_hits)] for t in range(hits.shape[0])]
    return np.concatenate(rows) if rows else np.zeros((0, 4), dtype=np.int64)


def _next_prime(n):
    n = max(3, n | 1)
    while any(n % q == 0 for q in range(3, int(n ** 0.5) + 1, 2)):
        n += 2
    return n


def _retry_overflows(rs, M, ph, low, order, start, max_hits):
    """Process overflowed residues one at a time with exactly sized buffers."""
    hits = np.full((1, max_hits, 4), -1, dtype=np.int64)
    nh = np.zeros(1, dtype=np.int64)
    for r in rs:
        need = int(_count(np.int64(r), M, ph, low, order, start)) + 16
        tb = int(np.ceil(np.log2(2 * need)))
        bufs = [np.empty(need, np.uint64), np.empty(need, np.int32), np.empty(need, np.int32),
                np.empty(need, np.uint64), np.empty(need, np.int32), np.empty(need, np.int32)]
        tab = np.full(1 << tb, -1, dtype=np.int64)
        assert _residue(np.int64(r), M, ph, low, order, start, *bufs, tab, tb, hits, nh, 0,
                        max_hits) == 0
    return hits, nh


def search(N, K=8, log2_M=None, seed=12345, max_hits=200000, batches_hint=1, progress=False):
    """Exhaustively find every 8-code type-2 set of length N (up to the symmetries
    folded into the distinct vectors). Returns a dict with the solutions found,
    each re-checked exactly, and timing."""
    if K != 8:
        raise ValueError('the meet-in-the-middle search is written for K = 8')
    t0 = time.time()
    B, Vall = code_vectors(N)
    V, first = np.unique(Vall, axis=0, return_index=True)
    V = np.ascontiguousarray(V, dtype=np.int64)
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 2 ** 63, size=V.shape[1], dtype=np.uint64) | np.uint64(1)
    h = _vec_hash(V, w)
    pi, pj, ph = _pairs(h)
    P = len(ph)
    if log2_M is None:
        # about 8 pairs per bucket (fast) up to ~1M pairs; 4 per bucket above (less memory)
        log2_M = max(4, int(np.ceil(np.log2(P))) - (3 if P < 2 ** 20 else 2))
    t1 = time.time()
    # Residues come from a separate random linear map modulo a prime. The low
    # bits of the 64-bit hash are NOT uniform (every vector coordinate has a
    # fixed parity), which made some residues huge and others empty.
    M = _next_prime(1 << log2_M)
    c = rng.integers(1, M, size=V.shape[1]).astype(np.int64)
    rv = (V @ c) % M
    low = (rv[pi] + rv[pj]) % M
    order = np.argsort(low, kind='stable').astype(np.int64)
    counts = np.bincount(low, minlength=M)
    start = np.zeros(M + 1, dtype=np.int64)
    start[1:] = np.cumsum(counts)
    expected = P * P / 2 / M
    cap = int(1.5 * expected + 10 * np.sqrt(expected)) + 4096
    tab_bits = int(np.ceil(np.log2(2 * cap)))
    half = M // 2                      # residues 0..half pair up with M - r
    batches = max(1, min(int(batches_hint), half + 1))
    edges = np.linspace(0, half + 1, batches + 1).astype(np.int64)
    all_hits, all_nh, n_retried, truncated = [], 0, 0, False
    max_over = 1 << 16
    tb = time.time()
    for bi in range(batches):
        hits, nh, over, nover = _search(ph, low, order, start, M, cap, tab_bits, max_hits,
                                        int(edges[bi]), int(edges[bi + 1]), max_over)
        if (nover > max_over).any():
            raise RuntimeError('too many overflowed residues; rerun with a larger log2_M')
        all_hits.append(_compact(hits, nh, max_hits))      # keep only real hits
        truncated |= bool((nh > max_hits).any())
        all_nh += int(nh.sum())
        rs = [int(r) for t in range(over.shape[0]) for r in over[t, :nover[t]]]
        if rs:
            h2, n2 = _retry_overflows(rs, M, ph, low, order, start, max_hits)
            all_hits.append(_compact(h2, n2, max_hits))
            truncated |= bool((n2 > max_hits).any())
            all_nh += int(n2.sum())
            n_retried += len(rs)
        if progress and batches > 1:
            done = (bi + 1) / batches
            el = time.time() - tb
            print(f'    progress {100 * done:5.1f}%  elapsed {el / 60:6.1f} min  '
                  f'estimated remaining {el / done * (1 - done) / 60:6.1f} min  '
                  f'hash hits so far {all_nh}  residues retried {n_retried}', flush=True)
    t2 = time.time()
    # exact re-check and dedupe
    sols = set()
    for rows in all_hits:
        for a, b, c, d in rows:
            idx = tuple(sorted([pi[a], pj[a], pi[b], pj[b], pi[c], pj[c], pi[d], pj[d]]))
            if not V[list(idx)].sum(axis=0).any():
                sols.add(idx)
    sol_codes = [B[first[list(s)]] for s in sorted(sols)]
    for S in sol_codes:
        assert verify.type2_peak(S) == 0
    # classes modulo the global chirp b[j] -> b[j] (-1)^(j(j-1)/2), which maps
    # every vector coordinate S[l, k] to (-1)^(l k) S[l, k]
    sign = np.array([(-1) ** (l * k) for l in range(1, N) for k in range(l, N - l)])
    index = {tuple(v): u for u, v in enumerate(V)}
    classes = set()
    for sidx in sols:
        img = tuple(sorted(index[tuple(V[u] * sign)] for u in sidx))
        classes.add(min(sidx, img))
    return dict(N=N, K=K, distinct_vectors=len(V), pairs=P, M=M, chirp_classes=len(classes),
                residues_retried=n_retried,
                hash_hits=all_nh, hits_truncated=truncated,
                solutions=len(sols), solution_indices=sorted(sols),
                solution_codes=[S.tolist() for S in sol_codes],
                setup_s=t1 - t0, search_s=t2 - t1, threads=get_num_threads())
