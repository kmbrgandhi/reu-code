"""Incremental objective kernels shared by every search algorithm.

The 2017 code recomputed the full autocorrelation (O(L^2) Python operations)
for every candidate move. These kernels keep the correlation entries of the
current code in memory and update them after a single-element change in O(L)
(binary PSL), O(L * N) (Doppler PSL), or O(L^2) (type-2 alternating sets).
Everything here is compiled with numba.

Problem kinds
-------------
BIN_PSL     one binary code, aperiodic autocorrelation C[k], k = 1..L-1.
BIN_TYPE2   K binary codes, Sulzer (1993) type-2 alternating condition:
            S[l, k] = sum_r sum_j b[j] b[j+l] b[j+k] b[j+l+k], 1 <= l <= k,
            l + k <= L - 1. A type-2 alternating set has S == 0 everywhere.
CPLX_PSL    one unimodular code (m-phase or continuous phase) with N Doppler
            bins spread over the full band:
            C[q, k] = sum_n a[n] conj(a[n+k]) exp(-2 pi i q n / N).
            N = 1 is the ordinary PSL. Same definition as the 2017 code's
            fast_autocorrelations.
CPLX_SET    K unimodular codes, summed aperiodic autocorrelation (complementary
            sets; what the 2017 code called "alternating" codes), with the
            complex conjugate that the 2017 code omitted.
CPLX_TYPE2  K unimodular codes, polyphase type-2 condition:
            S[l, k] = sum_r sum_j a[j] conj(a[j+l]) conj(a[j+k]) a[j+l+k].

Data layout (all numba-friendly tuples)
---------------------------------------
P = (ip, fp, dop, roots, frozen, zob)   problem description (read only)
S = (xi, vals, Ci, Cc)                  mutable search state

xi holds +1/-1 for binary kinds and phase indices 0..m-1 for m-phase kinds.
vals holds the complex values for complex kinds.
Ci / Cc hold the correlation entries for integer / complex kinds.
"""
import time

import numpy as np
from numba import njit, objmode

# Problem kinds.
BIN_PSL = 0
BIN_TYPE2 = 1
CPLX_PSL = 2
CPLX_SET = 3
CPLX_TYPE2 = 4

# Metrics.
M_PSL = 0     # peak |entry| (+ a tiny energy tie-break)
M_ISL = 1     # sum |entry|^2 (integrated sidelobe energy)
M_PNORM = 2   # sum |entry|^p (smooth proxy for the peak)

# Layout of the integer parameter vector ip.
IP_KIND = 0
IP_NV = 1       # number of variables (K * L)
IP_L = 2        # code length
IP_K = 3        # number of codes (1 for single-code problems)
IP_M = 4        # number of phases (2 for binary, 0 for continuous phase)
IP_N = 5        # number of Doppler bins
IP_METRIC = 6
IP_NMOVES = 7   # moves per variable: 1 (binary), m - 1 (m-phase), 0 (continuous)
IP_SIZE = 8

# Layout of the float parameter vector fp.
FP_P = 0        # exponent for M_PNORM
FP_EPS = 1      # weight of the energy tie-break in M_PSL
FP_SIZE = 2

TOL = 1e-12


@njit(cache=True)
def seed_rng(seed):
    """Seed numba's internal random generator (separate from NumPy's)."""
    np.random.seed(seed)


@njit(cache=True)
def now():
    """Wall-clock seconds, callable from compiled code (used for deadlines)."""
    with objmode(t='float64'):
        t = time.perf_counter()
    return t


@njit(cache=True)
def is_binary(ip):
    return ip[IP_KIND] == BIN_PSL or ip[IP_KIND] == BIN_TYPE2


# ---------------------------------------------------------------------------
# Metric accumulation
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _finish(metric, peak2, s2, sp, eps):
    if metric == M_PSL:
        return np.sqrt(peak2) + eps * s2
    elif metric == M_ISL:
        return s2
    return sp


@njit(cache=True, inline='always')
def _exceeds(metric, peak2, s2, sp, bound, bound2):
    """True once the partial cost provably exceeds `bound` (early exit)."""
    if metric == M_PSL:
        return peak2 > bound2
    elif metric == M_ISL:
        return s2 > bound
    return sp > bound


@njit(cache=True, inline='always')
def _bound2(bound):
    if bound == np.inf:
        return np.inf
    if bound < 0.0:
        return -1.0
    return bound * bound * (1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Initialization from scratch
# ---------------------------------------------------------------------------

@njit(cache=True)
def init_entries(P, S):
    ip = P[0]
    dop = P[2]
    xi, vals, Ci, Cc = S
    kind = ip[IP_KIND]
    L = ip[IP_L]
    K = ip[IP_K]
    N = ip[IP_N]
    if kind == BIN_PSL:
        Ci[0, 0] = 0
        for k in range(1, L):
            s = 0
            for n in range(L - k):
                s += xi[n] * xi[n + k]
            Ci[0, k] = s
    elif kind == BIN_TYPE2:
        Ci[:, :] = 0
        for r in range(K):
            o = r * L
            for l in range(1, L):
                for k in range(l, L - l):
                    s = 0
                    for j in range(L - l - k):
                        s += xi[o + j] * xi[o + j + l] * xi[o + j + k] * xi[o + j + l + k]
                    Ci[l, k] += s
    elif kind == CPLX_PSL:
        for q in range(N):
            for k in range(L):
                if q == 0 and k == 0:
                    Cc[0, 0] = 0.0
                    continue
                s = 0.0 + 0.0j
                for n in range(L - k):
                    s += vals[n] * np.conj(vals[n + k]) * dop[q, n]
                Cc[q, k] = s
    elif kind == CPLX_SET:
        Cc[0, 0] = 0.0
        for k in range(1, L):
            s = 0.0 + 0.0j
            for r in range(K):
                o = r * L
                for n in range(L - k):
                    s += vals[o + n] * np.conj(vals[o + n + k])
            Cc[0, k] = s
    elif kind == CPLX_TYPE2:
        Cc[:, :] = 0.0
        for r in range(K):
            o = r * L
            for l in range(1, L):
                for k in range(l, L - l):
                    s = 0.0 + 0.0j
                    for j in range(L - l - k):
                        s += (vals[o + j] * np.conj(vals[o + j + l])
                              * np.conj(vals[o + j + k]) * vals[o + j + l + k])
                    Cc[l, k] += s


# ---------------------------------------------------------------------------
# Cost of the current state
# ---------------------------------------------------------------------------

@njit(cache=True)
def full_cost(P, S):
    """Metric over the stored entries (O(number of entries))."""
    ip = P[0]
    fp = P[1]
    Ci = S[2]
    Cc = S[3]
    kind = ip[IP_KIND]
    L = ip[IP_L]
    N = ip[IP_N]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    if kind == BIN_PSL:
        for k in range(1, L):
            m2 = float(Ci[0, k] * Ci[0, k])
            peak2 = max(peak2, m2)
            s2 += m2
            if metric == M_PNORM:
                sp += m2 ** hp
    elif kind == BIN_TYPE2:
        for l in range(1, L):
            for k in range(l, L - l):
                m2 = float(Ci[l, k] * Ci[l, k])
                peak2 = max(peak2, m2)
                s2 += m2
                if metric == M_PNORM:
                    sp += m2 ** hp
    elif kind == CPLX_PSL:
        for q in range(N):
            for k in range(L):
                if q == 0 and k == 0:
                    continue
                c = Cc[q, k]
                m2 = c.real * c.real + c.imag * c.imag
                peak2 = max(peak2, m2)
                s2 += m2
                if metric == M_PNORM:
                    sp += m2 ** hp
    elif kind == CPLX_SET:
        for k in range(1, L):
            c = Cc[0, k]
            m2 = c.real * c.real + c.imag * c.imag
            peak2 = max(peak2, m2)
            s2 += m2
            if metric == M_PNORM:
                sp += m2 ** hp
    else:  # CPLX_TYPE2
        for l in range(1, L):
            for k in range(l, L - l):
                c = Cc[l, k]
                m2 = c.real * c.real + c.imag * c.imag
                peak2 = max(peak2, m2)
                s2 += m2
                if metric == M_PNORM:
                    sp += m2 ** hp
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def peak_value(P, S):
    """Largest |entry| (the PSL-style objective), independent of the metric."""
    ip = P[0]
    Ci = S[2]
    Cc = S[3]
    kind = ip[IP_KIND]
    L = ip[IP_L]
    N = ip[IP_N]
    peak2 = 0.0
    if kind == BIN_PSL:
        for k in range(1, L):
            peak2 = max(peak2, float(Ci[0, k] * Ci[0, k]))
    elif kind == BIN_TYPE2:
        for l in range(1, L):
            for k in range(l, L - l):
                peak2 = max(peak2, float(Ci[l, k] * Ci[l, k]))
    elif kind == CPLX_PSL:
        for q in range(N):
            for k in range(L):
                if q == 0 and k == 0:
                    continue
                c = Cc[q, k]
                peak2 = max(peak2, c.real * c.real + c.imag * c.imag)
    elif kind == CPLX_SET:
        for k in range(1, L):
            c = Cc[0, k]
            peak2 = max(peak2, c.real * c.real + c.imag * c.imag)
    else:
        for l in range(1, L):
            for k in range(l, L - l):
                c = Cc[l, k]
                peak2 = max(peak2, c.real * c.real + c.imag * c.imag)
    return np.sqrt(peak2)


# ---------------------------------------------------------------------------
# BIN_PSL: flip one element
# ---------------------------------------------------------------------------

@njit(cache=True)
def _eval_binpsl(ip, fp, xi, Ci, i, bound):
    L = ip[IP_L]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    b2 = _bound2(bound)
    bi = xi[i]
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    for k in range(1, L):
        d = 0
        if i + k < L:
            d += xi[i + k]
        if i - k >= 0:
            d += xi[i - k]
        c = Ci[0, k] - 2 * bi * d
        m2 = float(c * c)
        if m2 > peak2:
            peak2 = m2
        s2 += m2
        if metric == M_PNORM:
            sp += m2 ** hp
        if _exceeds(metric, peak2, s2, sp, bound, b2):
            return _finish(metric, peak2, s2, sp, fp[FP_EPS])
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def _apply_binpsl(ip, xi, Ci, i):
    L = ip[IP_L]
    bi = xi[i]
    for k in range(1, L):
        d = 0
        if i + k < L:
            d += xi[i + k]
        if i - k >= 0:
            d += xi[i - k]
        Ci[0, k] -= 2 * bi * d
    xi[i] = -bi


# ---------------------------------------------------------------------------
# BIN_TYPE2: flip one element of one code
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _t2bin(xi, o, j, l, k):
    return xi[o + j] * xi[o + j + l] * xi[o + j + k] * xi[o + j + l + k]


# Flipping bit p changes the terms with j = p, p-l, p-k, p-l-k. A term flips
# sign when p appears in it an odd number of times. For l == k, the term with
# j = p - l contains p twice and does not change.

@njit(cache=True)
def _eval_bint2(ip, fp, xi, Ci, v, bound):
    L = ip[IP_L]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    b2 = _bound2(bound)
    o = (v // L) * L
    p = v - o
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    for l in range(1, L):
        for k in range(l, L - l):
            jmax = L - 1 - l - k
            d = 0
            if p <= jmax:
                d += _t2bin(xi, o, p, l, k)
            j = p - l - k
            if j >= 0:
                d += _t2bin(xi, o, j, l, k)
            if l != k:
                j = p - l
                if j >= 0 and j <= jmax:
                    d += _t2bin(xi, o, j, l, k)
                j = p - k
                if j >= 0 and j <= jmax:
                    d += _t2bin(xi, o, j, l, k)
            c = Ci[l, k] - 2 * d
            m2 = float(c * c)
            if m2 > peak2:
                peak2 = m2
            s2 += m2
            if metric == M_PNORM:
                sp += m2 ** hp
            if _exceeds(metric, peak2, s2, sp, bound, b2):
                return _finish(metric, peak2, s2, sp, fp[FP_EPS])
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def _apply_bint2(ip, xi, Ci, v):
    L = ip[IP_L]
    o = (v // L) * L
    p = v - o
    for l in range(1, L):
        for k in range(l, L - l):
            jmax = L - 1 - l - k
            d = 0
            if p <= jmax:
                d += _t2bin(xi, o, p, l, k)
            j = p - l - k
            if j >= 0:
                d += _t2bin(xi, o, j, l, k)
            if l != k:
                j = p - l
                if j >= 0 and j <= jmax:
                    d += _t2bin(xi, o, j, l, k)
                j = p - k
                if j >= 0 and j <= jmax:
                    d += _t2bin(xi, o, j, l, k)
            Ci[l, k] -= 2 * d
    xi[v] = -xi[v]


# ---------------------------------------------------------------------------
# CPLX_PSL: set one element to a new unimodular value z
# ---------------------------------------------------------------------------

@njit(cache=True)
def _eval_cpsl(ip, fp, dop, vals, Cc, i, z, bound):
    L = ip[IP_L]
    N = ip[IP_N]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    b2 = _bound2(bound)
    dd = z - vals[i]
    cdd = np.conj(dd)
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    for q in range(N):
        a1 = dd * dop[q, i]
        for k in range(L):
            if k == 0:
                if q == 0:
                    continue
                c = Cc[q, 0]          # zero-delay, nonzero Doppler: code independent
            else:
                c = Cc[q, k]
                if i + k < L:
                    c += a1 * np.conj(vals[i + k])
                if i - k >= 0:
                    c += vals[i - k] * cdd * dop[q, i - k]
            m2 = c.real * c.real + c.imag * c.imag
            if m2 > peak2:
                peak2 = m2
            s2 += m2
            if metric == M_PNORM:
                sp += m2 ** hp
            if _exceeds(metric, peak2, s2, sp, bound, b2):
                return _finish(metric, peak2, s2, sp, fp[FP_EPS])
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def _apply_cpsl(ip, dop, vals, Cc, i, z):
    L = ip[IP_L]
    N = ip[IP_N]
    dd = z - vals[i]
    cdd = np.conj(dd)
    for q in range(N):
        a1 = dd * dop[q, i]
        for k in range(1, L):
            c = 0.0 + 0.0j
            if i + k < L:
                c += a1 * np.conj(vals[i + k])
            if i - k >= 0:
                c += vals[i - k] * cdd * dop[q, i - k]
            Cc[q, k] += c
    vals[i] = z


# ---------------------------------------------------------------------------
# CPLX_SET: set one element of one code
# ---------------------------------------------------------------------------

@njit(cache=True)
def _eval_cset(ip, fp, vals, Cc, v, z, bound):
    L = ip[IP_L]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    b2 = _bound2(bound)
    o = (v // L) * L
    i = v - o
    dd = z - vals[v]
    cdd = np.conj(dd)
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    for k in range(1, L):
        c = Cc[0, k]
        if i + k < L:
            c += dd * np.conj(vals[o + i + k])
        if i - k >= 0:
            c += vals[o + i - k] * cdd
        m2 = c.real * c.real + c.imag * c.imag
        if m2 > peak2:
            peak2 = m2
        s2 += m2
        if metric == M_PNORM:
            sp += m2 ** hp
        if _exceeds(metric, peak2, s2, sp, bound, b2):
            return _finish(metric, peak2, s2, sp, fp[FP_EPS])
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def _apply_cset(ip, vals, Cc, v, z):
    L = ip[IP_L]
    o = (v // L) * L
    i = v - o
    dd = z - vals[v]
    cdd = np.conj(dd)
    for k in range(1, L):
        if i + k < L:
            Cc[0, k] += dd * np.conj(vals[o + i + k])
        if i - k >= 0:
            Cc[0, k] += vals[o + i - k] * cdd
    vals[v] = z


# ---------------------------------------------------------------------------
# CPLX_TYPE2: set one element of one code
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _t2c(vals, o, j, l, k, p, z):
    """Fourth-order term with position p (relative to the code) replaced by z.
    Pass p = -1 for the unmodified term."""
    a = z if j == p else vals[o + j]
    b = z if j + l == p else vals[o + j + l]
    c = z if j + k == p else vals[o + j + k]
    d = z if j + l + k == p else vals[o + j + l + k]
    return a * np.conj(b) * np.conj(c) * d


@njit(cache=True, inline='always')
def _t2c_delta(vals, o, p, l, k, z, L):
    jmax = L - 1 - l - k
    d = 0.0 + 0.0j
    if p <= jmax:
        d += _t2c(vals, o, p, l, k, p, z) - _t2c(vals, o, p, l, k, -1, z)
    j = p - l - k
    if j >= 0:
        d += _t2c(vals, o, j, l, k, p, z) - _t2c(vals, o, j, l, k, -1, z)
    j = p - l
    if j >= 0 and j <= jmax:
        d += _t2c(vals, o, j, l, k, p, z) - _t2c(vals, o, j, l, k, -1, z)
    if l != k:
        j = p - k
        if j >= 0 and j <= jmax:
            d += _t2c(vals, o, j, l, k, p, z) - _t2c(vals, o, j, l, k, -1, z)
    return d


@njit(cache=True)
def _eval_ct2(ip, fp, vals, Cc, v, z, bound):
    L = ip[IP_L]
    metric = ip[IP_METRIC]
    hp = 0.5 * fp[FP_P]
    b2 = _bound2(bound)
    o = (v // L) * L
    p = v - o
    peak2 = 0.0
    s2 = 0.0
    sp = 0.0
    for l in range(1, L):
        for k in range(l, L - l):
            c = Cc[l, k] + _t2c_delta(vals, o, p, l, k, z, L)
            m2 = c.real * c.real + c.imag * c.imag
            if m2 > peak2:
                peak2 = m2
            s2 += m2
            if metric == M_PNORM:
                sp += m2 ** hp
            if _exceeds(metric, peak2, s2, sp, bound, b2):
                return _finish(metric, peak2, s2, sp, fp[FP_EPS])
    return _finish(metric, peak2, s2, sp, fp[FP_EPS])


@njit(cache=True)
def _apply_ct2(ip, vals, Cc, v, z):
    L = ip[IP_L]
    o = (v // L) * L
    p = v - o
    for l in range(1, L):
        for k in range(l, L - l):
            Cc[l, k] += _t2c_delta(vals, o, p, l, k, z, L)
    vals[v] = z


# ---------------------------------------------------------------------------
# Generic move interface used by the algorithms
# ---------------------------------------------------------------------------
# A discrete move is (v, t): add t (1..m-1) to the phase index of variable v.
# For binary kinds there is one move per variable (a flip) and t is ignored.

@njit(cache=True)
def eval_value(P, S, v, z, bound):
    """Cost after setting variable v of a complex kind to the value z."""
    ip = P[0]
    kind = ip[IP_KIND]
    if kind == CPLX_PSL:
        return _eval_cpsl(ip, P[1], P[2], S[1], S[3], v, z, bound)
    elif kind == CPLX_SET:
        return _eval_cset(ip, P[1], S[1], S[3], v, z, bound)
    return _eval_ct2(ip, P[1], S[1], S[3], v, z, bound)


@njit(cache=True)
def apply_value(P, S, v, z):
    ip = P[0]
    kind = ip[IP_KIND]
    if kind == CPLX_PSL:
        _apply_cpsl(ip, P[2], S[1], S[3], v, z)
    elif kind == CPLX_SET:
        _apply_cset(ip, S[1], S[3], v, z)
    else:
        _apply_ct2(ip, S[1], S[3], v, z)


@njit(cache=True)
def eval_move(P, S, v, t, bound):
    """Cost after move (v, t), without changing the state.

    If the cost provably exceeds `bound`, the function may stop early and
    return any value larger than `bound`. Pass np.inf for the exact cost.
    """
    ip = P[0]
    kind = ip[IP_KIND]
    if kind == BIN_PSL:
        return _eval_binpsl(ip, P[1], S[0], S[2], v, bound)
    elif kind == BIN_TYPE2:
        return _eval_bint2(ip, P[1], S[0], S[2], v, bound)
    z = P[3][(S[0][v] + t) % ip[IP_M]]
    return eval_value(P, S, v, z, bound)


@njit(cache=True)
def apply_move(P, S, v, t):
    ip = P[0]
    kind = ip[IP_KIND]
    if kind == BIN_PSL:
        _apply_binpsl(ip, S[0], S[2], v)
    elif kind == BIN_TYPE2:
        _apply_bint2(ip, S[0], S[2], v)
    else:
        m = ip[IP_M]
        nidx = (S[0][v] + t) % m
        apply_value(P, S, v, P[3][nidx])
        S[0][v] = nidx


@njit(cache=True)
def inverse_t(P, t):
    """Move that undoes (v, t)."""
    m = P[0][IP_M]
    if is_binary(P[0]):
        return t
    return m - t


# ---------------------------------------------------------------------------
# Random states and copies
# ---------------------------------------------------------------------------

@njit(cache=True)
def random_init(P, S):
    """Uniformly random code; frozen variables get value +1 (phase index 0)."""
    ip = P[0]
    roots = P[3]
    frozen = P[4]
    xi, vals = S[0], S[1]
    nv = ip[IP_NV]
    m = ip[IP_M]
    binary = is_binary(ip)
    for v in range(nv):
        if binary:
            if frozen[v]:
                xi[v] = 1
            else:
                xi[v] = 1 if np.random.random() < 0.5 else -1
        elif m > 0:
            idx = 0 if frozen[v] else np.random.randint(0, m)
            xi[v] = idx
            vals[v] = roots[idx]
        else:
            if frozen[v]:
                vals[v] = 1.0 + 0.0j
            else:
                ph = 2.0 * np.pi * np.random.random()
                vals[v] = np.cos(ph) + 1j * np.sin(ph)
    init_entries(P, S)


@njit(cache=True)
def load_code(P, S, x, z):
    """Load a code (xi part x, complex part z) into S and rebuild the entries."""
    S[0][:] = x
    S[1][:] = z
    init_entries(P, S)


@njit(cache=True)
def save_state(S, bx, bz, bCi, bCc):
    bx[:] = S[0]
    bz[:] = S[1]
    bCi[:, :] = S[2]
    bCc[:, :] = S[3]


@njit(cache=True)
def restore_state(S, bx, bz, bCi, bCc):
    S[0][:] = bx
    S[1][:] = bz
    S[2][:, :] = bCi
    S[3][:, :] = bCc


@njit(cache=True)
def n_free(P):
    c = 0
    fr = P[4]
    for v in range(fr.shape[0]):
        if not fr[v]:
            c += 1
    return c


@njit(cache=True)
def random_free_var(P):
    fr = P[4]
    nv = fr.shape[0]
    while True:
        v = np.random.randint(0, nv)
        if not fr[v]:
            return v


@njit(cache=True)
def random_t(P):
    nm = P[0][IP_NMOVES]
    if nm <= 1:
        return 1
    return np.random.randint(1, nm + 1)


# ---------------------------------------------------------------------------
# Zobrist hashing and a fixed-size hash set (for tabu-by-state and SAW)
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def zidx(P, S, v):
    """Hash-table column for the current value of variable v."""
    if is_binary(P[0]):
        return (1 - S[0][v]) // 2
    return S[0][v]


@njit(cache=True)
def state_hash(P, S):
    zob = P[5]
    h = np.uint64(0)
    for v in range(P[0][IP_NV]):
        h ^= zob[v, zidx(P, S, v)]
    return h | np.uint64(1)


@njit(cache=True)
def move_hash(P, S, h, v, t):
    """Hash of the state after move (v, t), given the current hash h."""
    zob = P[5]
    cur = zidx(P, S, v)
    if is_binary(P[0]):
        new = 1 - cur
    else:
        new = (S[0][v] + t) % P[0][IP_M]
    return (h ^ zob[v, cur] ^ zob[v, new]) | np.uint64(1)


@njit(cache=True)
def hs_clear(tab):
    tab[:] = 0


@njit(cache=True)
def hs_contains(tab, h):
    mask = np.uint64(tab.shape[0] - 1)
    i = (h * np.uint64(0x9E3779B97F4A7C15)) & mask
    while True:
        x = tab[i]
        if x == 0:
            return False
        if x == h:
            return True
        i = (i + np.uint64(1)) & mask


@njit(cache=True)
def hs_insert(tab, h):
    """Insert h; returns False if it was already present."""
    mask = np.uint64(tab.shape[0] - 1)
    i = (h * np.uint64(0x9E3779B97F4A7C15)) & mask
    while True:
        x = tab[i]
        if x == 0:
            tab[i] = h
            return True
        if x == h:
            return False
        i = (i + np.uint64(1)) & mask
