"""Exact small-K search for m-phase type-2 alternating sets (light CPU).

Each code's fourth-order terms a_j conj(a_{j+l}) conj(a_{j+k}) a_{j+l+k} are m-th roots of unity.
Their sums are stored exactly as integer coordinates:
  m = 2: integer;  m = 3: a + b*w (w = e^{2 pi i/3}, w^2 = -1 - w);  m = 4: a + b*i.
A K-multiset of codes is type-2 iff its integer vectors sum to zero.
"""
import itertools, sys, time
import numpy as np

def int_vectors(N, m):
    pairs = [(l, k) for l in range(1, N) for k in range(l, N - l)]
    X = np.array(list(itertools.product(range(m), repeat=N - 2)), dtype=np.int64).reshape(-1, N - 2)
    X = np.concatenate([np.zeros((len(X), 2), np.int64), X], axis=1)  # constant phase + ramp fixed
    cols = []
    for l, k in pairs:
        J = N - l - k
        e = (X[:, :J] - X[:, l:l+J] - X[:, k:k+J] + X[:, l+k:l+k+J]) % m
        cnt = [(e == r).sum(1) for r in range(m)]
        if m == 2:
            cols.append(cnt[0] - cnt[1])
        elif m == 3:
            cols += [cnt[0] - cnt[2], cnt[1] - cnt[2]]
        elif m == 4:
            cols += [cnt[0] - cnt[2], cnt[1] - cnt[3]]
    V = np.stack(cols, 1)
    U, first = np.unique(V, axis=0, return_index=True)
    return X, U, first

def ksum_zero(U, K, seed=1):
    """All K-multisets (sorted index tuples) of rows of U summing to zero, for K in 1..6.
    Meet in the middle: K = a + b with a = K // 2; exact linear 64-bit hash, then exact check."""
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 2**63, size=U.shape[1], dtype=np.uint64) | np.uint64(1)
    h = (U.astype(np.uint64) * w).sum(1)              # wraps mod 2^64: exactly additive
    def multisets(k):
        return np.array(list(itertools.combinations_with_replacement(range(len(U)), k)), dtype=np.int64).reshape(-1, k)
    a = K // 2; b = K - a
    A = multisets(a); B = A if b == a else multisets(b)
    hA = h[A].sum(1) if a else np.zeros(1, np.uint64)
    hB = h[B].sum(1)
    order = np.argsort(hB); hBs = hB[order]
    want = (np.uint64(0) - hA)
    lo = np.searchsorted(hBs, want, 'left'); hi = np.searchsorted(hBs, want, 'right')
    sols = set()
    for i in np.nonzero(hi > lo)[0]:
        for jpos in range(lo[i], hi[i]):
            idx = tuple(sorted(list(A[i]) + list(B[order[jpos]])))
            if not U[list(idx)].sum(0).any():
                sols.add(idx)
    return sols

if __name__ == '__main__':
    jobs = [(2, 4, range(6, 10)),        # validation: binary 4-code sets exist up to N = 7 only (Sulzer 1993)
            (3, 3, range(4, 10)), (3, 6, range(5, 9)),
            (4, 2, range(4, 10)), (4, 4, range(5, 9))]
    for m, K, Ns in jobs:
        for N in Ns:
            t = time.time()
            X, U, first = int_vectors(N, m)
            if K >= 6 and len(U) > 400 or K >= 4 and len(U) > 9000:
                print(f'm={m} K={K} N={N}: skipped (U={len(U)} too large for this light run)'); continue
            sols = ksum_zero(U, K)
            ex = '' if not sols else f'  e.g. phase exponents {[X[first[u]].tolist() for u in sorted(sols)[0]]}'
            print(f'm={m} K={K} N={N}: distinct vectors {len(U):5d}, type-2 sets {len(sols):5d}  ({time.time()-t:.1f}s){ex}', flush=True)
