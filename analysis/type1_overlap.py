"""Which 8-code type-2 sets are also type-1 (Lehtinen weak) or complementary sets?

For each solution in results/cleanroom/solutions_N*.txt, the script tries every
choice of codes that share each code's fourth-order vector (the normalized
codes with c0 = c1 = +1). It reports:
  - type-1: some choice meets Lehtinen's weak condition (for every lag, the
    lag products at different positions are orthogonal across the codes);
  - complementary (N = 8 only, the slow part): some choice, with per-code
    sign alternation, has autocorrelations that sum to zero at every lag.

Usage: python analysis/type1_overlap.py > results/type1_overlap.txt
"""
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def classes(N):
    idx = [(l, k) for l in range(1, N) for k in range(l, N - l)]

    def vec(c):
        return tuple(int(sum(c[j] * c[j + l] * c[j + k] * c[j + l + k] for j in range(N - l - k))) for l, k in idx)
    cls = defaultdict(list)
    for m in range(1 << (N - 2)):
        c = [1, 1] + [-1 if (m >> (N - 1 - j)) & 1 else 1 for j in range(2, N)]
        cls[vec(c)].append(np.array(c))
    return vec, cls


def weak_ok(S):
    N = S.shape[1]
    for l in range(1, N):
        T = S[:, :N - l] * S[:, l:]
        G = T.T @ T
        np.fill_diagonal(G, 0)
        if np.abs(G).max():
            return False
    return True


def complementary_possible(choices, N):
    odd = np.arange(1, N) % 2 == 1
    for ch in itertools.product(*choices):
        A = [np.array([int(np.dot(c[:N - t], c[t:])) for t in range(1, N)]) for c in ch]
        if sum(a[~odd] for a in A).any():        # alternation can't change even lags
            continue
        parts = [a[odd] for a in A]
        for signs in itertools.product([1, -1], repeat=len(A) - 1):
            if not sum(s * o for s, o in zip((1,) + signs, parts)).any():
                return True
    return False


for N in (8, 9, 10, 11, 12, 14):
    vec, cls = classes(N)
    rows = (ROOT / 'results' / 'cleanroom' / f'solutions_N{N}.txt').read_text().split('\n')[:-1]
    n1 = ndc = nboth = 0
    for row in rows:
        vs = [vec([-1 if (int(h, 16) >> (N - 1 - j)) & 1 else 1 for j in range(N)]) for h in row.split()]
        choices = [cls[v] for v in vs]
        t1 = any(weak_ok(np.array(ch)) for ch in itertools.product(*choices))
        n1 += t1
        if N == 8:
            dc = complementary_possible(choices, N)
            ndc += dc
            nboth += t1 and dc
            if t1:
                print(f'N=8 type-1 set: {row}  complementary: {dc}')
    extra = f'  doubly complementary: {ndc}  type-1 and doubly complementary: {nboth}' if N == 8 else ''
    print(f'N={N}: sets={len(rows)}  type-1 for some code choice: {n1}{extra}', flush=True)
