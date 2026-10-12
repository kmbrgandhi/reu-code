"""Independent reference evaluators (plain NumPy, no shared code with the kernels).

Use these to check any code the searches report. They are deliberately written
in a different form from the incremental kernels: the type-2 check builds the
lag-product sequences explicitly, as in Sulzer (1993), instead of using the
fourth-order sums.
"""
import numpy as np


def doppler_psl(code, N=1):
    """Peak off-peak |ambiguity| over N Doppler bins spread over the full band.

    |chi(tau, q)| = |sum_n c[n] conj(c[n+tau]) exp(-2 pi i q n / N)|, all delays,
    q = 0..N-1, excluding (tau = 0, q = 0). N = 1 is the ordinary PSL.
    """
    c = np.asarray(code, dtype=complex)
    L = len(c)
    n = np.arange(L)
    best = 0.0
    for q in range(N):
        d = c * np.exp(-2j * np.pi * q * n / N)
        r = np.abs(np.correlate(d, c, mode='full'))
        if q == 0:
            r[L - 1] = 0.0
        best = max(best, r.max())
    return best


def summed_acf(codes):
    """Sum of the aperiodic autocorrelations (with conjugate), lags 0..L-1."""
    codes = np.asarray(codes, dtype=complex)
    L = codes.shape[1]
    tot = np.zeros(L, dtype=complex)
    for c in codes:
        for k in range(L):
            tot[k] += np.dot(c[:L - k], np.conj(c[k:]))
    return tot


def set_peak(codes):
    """Peak |summed autocorrelation| at nonzero lags (0 = complementary set)."""
    return float(np.abs(summed_acf(codes)[1:]).max()) if np.asarray(codes).shape[1] > 1 else 0.0


def type2_peak(codes):
    """Sulzer (1993) type-2 check. For each lag l, form the lag-product codes
    t[j] = s[j] conj(s[j+l]) and require their summed autocorrelation to vanish
    at every nonzero shift. Returns the largest violation (0 = type-2 set)."""
    S = np.asarray(codes, dtype=complex)
    L = S.shape[1]
    worst = 0.0
    for l in range(1, L - 1):
        T = S[:, :L - l] * np.conj(S[:, l:])
        worst = max(worst, float(np.abs(summed_acf(T)[1:]).max()))
    return worst


def lehtinen_weak_violation(codes):
    """Lehtinen (1986) weak alternating condition: for every lag l and every pair
    of positions i != i', sum_k t_k[i] conj(t_k[i']) = 0 with t = lag-l products.
    Returns the largest violation (0 = weak alternating code set)."""
    S = np.asarray(codes, dtype=complex)
    L = S.shape[1]
    worst = 0.0
    for l in range(1, L):
        T = S[:, :L - l] * np.conj(S[:, l:])
        G = T.T @ np.conj(T)
        np.fill_diagonal(G, 0)
        if G.size:
            worst = max(worst, float(np.abs(G).max()))
    return worst


def evaluate(problem, xi, vals):
    """Independent objective value for a state of `problem`."""
    from . import kernels as KN
    c = problem.complex_code(xi, vals).reshape(problem.K, problem.L)
    if problem.kind in (KN.BIN_PSL, KN.CPLX_PSL):
        return doppler_psl(c[0], problem.N)
    if problem.kind == KN.CPLX_SET:
        return set_peak(c)
    return type2_peak(c)
