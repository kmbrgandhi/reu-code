"""Exact search for binary type-2 alternating code sets with a constraint solver.

The type-2 condition (Sulzer 1993) for K binary codes of length N is

    S[l, k] = sum_r sum_j b[r,j] b[r,j+l] b[r,j+k] b[r,j+l+k] = 0
    for all 1 <= l <= k with l + k <= N - 1.

With b = (-1)^x for bits x, each product is (-1)^(XOR of the bits), so
S[l, k] = 0 says that exactly half of the XOR terms in each group are 1. That
is a pure Boolean model: XOR constraints plus one exact-count constraint per
(l, k). Google OR-Tools CP-SAT solves it on all cores and either finds a set,
proves that none exists (INFEASIBLE), or runs out of time (UNKNOWN).

Symmetry breaking (each keeps at least one solution from every class):
- Per code: negation and alternate-sign multiplication leave every term
  unchanged, so the first two bits of each code are fixed to 0 (+1).
- Across codes: the order of the codes does not matter, so the codes (read
  as binary numbers) are required to be non-decreasing.
"""
import math
import os
import time

import numpy as np

from . import verify


def build_model(N, K):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f'x{r}_{j}') for j in range(N)] for r in range(K)]
    for r in range(K):
        m.Add(x[r][0] == 0)
        m.Add(x[r][1] == 0)
    cache = {}
    n_terms = 0
    for l in range(1, N):
        for k in range(l, N - l):
            n = K * (N - l - k)
            if n % 2:
                return None  # odd number of +-1 terms can never sum to 0
            ts = []
            for r in range(K):
                for j in range(N - l - k):
                    pos = [j, j + l, j + k, j + l + k]
                    # positions that appear twice cancel (l == k gives b[j+l]^2 = 1)
                    odd = tuple(sorted(p for p in set(pos) if pos.count(p) % 2 == 1))
                    key = (r, odd)
                    if key not in cache:
                        t = m.NewBoolVar('')
                        # t = 1 exactly when the product is -1 (odd parity)
                        m.AddBoolXOr([x[r][p] for p in odd] + [t.Not()])
                        cache[key] = t
                    ts.append(cache[key])
                    n_terms += 1
            m.Add(sum(ts) == n // 2)
    weights = [2 ** (N - 1 - j) for j in range(2, N)]
    vals = [sum(w * x[r][j] for w, j in zip(weights, range(2, N))) for r in range(K)]
    for r in range(K - 1):
        m.Add(vals[r] <= vals[r + 1])
    return m, x, dict(xor_vars=len(cache), terms=n_terms)


def solve(N, K, time_limit=600.0, workers=None, log=False):
    """Returns dict(status, codes (K x N array of +-1 or None), time, model stats,
    verified_peak)."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    built = build_model(N, K)
    if built is None:
        return dict(N=N, K=K, status='INFEASIBLE (parity)', codes=None, time=0.0)
    m, x, stats = built
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = workers or os.cpu_count()
    s.parameters.log_search_progress = bool(log)
    st = s.Solve(m)
    name = s.StatusName(st)
    out = dict(N=N, K=K, status=name, time=time.time() - t0, model=stats, codes=None)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        codes = np.array([[1 - 2 * s.Value(x[r][j]) for j in range(N)] for r in range(K)])
        out['codes'] = codes.tolist()
        out['verified_peak'] = verify.type2_peak(codes)
        out['hex'] = [format(int(''.join('0' if b > 0 else '1' for b in c), 2), 'X').zfill(math.ceil(N / 4))
                      for c in codes]
    return out


# ---------------------------------------------------------------------------
# Count model: choose how many times each distinct code vector is used
# ---------------------------------------------------------------------------

def code_vectors(N):
    """All codes with the first two elements +1, and each code's contribution
    to every S[l, k] (l <= k, l + k <= N - 1). Returns (codes, vectors)."""
    n = N - 2
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n - 1, -1, -1)) & 1
    B = np.concatenate([np.ones((len(idx), 2), dtype=np.int64), 1 - 2 * bits], axis=1)
    cols = []
    for l in range(1, N):
        for k in range(l, N - l):
            J = N - l - k
            cols.append((B[:, :J] * B[:, l:l + J] * B[:, k:k + J] * B[:, l + k:l + k + J]).sum(axis=1))
    return B, np.stack(cols, axis=1)


def solve_counts(N, K, time_limit=600.0, workers=None, log=False):
    """Type-2 set as an integer program: n_u >= 0 copies of each distinct code
    vector u, sum n_u = K, and sum_u n_u * v_u = 0 in every coordinate.

    Codes with the same vector are interchangeable, and the order of codes does
    not matter, so this model has no code-order or equal-vector symmetry.
    INFEASIBLE is a proof that no type-2 set of K binary codes of length N
    exists."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    B, V = code_vectors(N)
    U, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    m = cp_model.CpModel()
    n = [m.NewIntVar(0, K, f'n{u}') for u in range(len(U))]
    m.Add(sum(n) == K)
    for d in range(U.shape[1]):
        col = U[:, d]
        nz = np.nonzero(col)[0]
        m.Add(sum(int(col[u]) * n[u] for u in nz) == 0)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = workers or os.cpu_count()
    s.parameters.log_search_progress = bool(log)
    st = s.Solve(m)
    out = dict(N=N, K=K, status=s.StatusName(st), time=time.time() - t0,
               model=dict(distinct_vectors=len(U), dims=U.shape[1]), codes=None)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        rows = []
        for u in range(len(U)):
            rows += [B[first[u]]] * s.Value(n[u])
        codes = np.array(rows)
        out['codes'] = codes.tolist()
        out['verified_peak'] = verify.type2_peak(codes)
        out['hex'] = [format(int(''.join('0' if b > 0 else '1' for b in c), 2), 'X').zfill(math.ceil(N / 4))
                      for c in codes]
    return out
