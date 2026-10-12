"""Exhaustive and vector-space type-2 searches."""
import numpy as np

from fastcodes import exhaust, records, verify, vsearch


def test_exhaustive_counts_match_sulzer_1993():
    """Sulzer (1993), Table 3: 86, 78 and 150 sets of 8 codes for N = 8, 9, 10."""
    for N, n in ((8, 86), (9, 78), (10, 150)):
        r = exhaust.search(N)
        assert r['solutions'] == n, N
        for S in r['solution_codes']:
            assert verify.type2_peak(np.array(S)) == 0


def test_exhaustive_includes_sulzer_table4_sets():
    """Each published Sulzer set appears among the exhaustive solutions
    (compared as multisets of code vectors)."""
    from fastcodes.exact import code_vectors
    for N in (8, 9, 10):
        r = exhaust.search(N)
        B, V = code_vectors(N)
        U = np.unique(V, axis=0)
        index = {tuple(v): u for u, v in enumerate(U)}

        def vec(c):
            c = c * c[0]                       # make the first element +1
            c = c * ((-1) ** np.arange(N)) if c[1] < 0 else c   # and the second
            cols = [int((c[:N - l - k] * c[l:N - k] * c[k:N - l] * c[l + k:]).sum())
                    for l in range(1, N) for k in range(l, N - l)]
            return tuple(cols)
        pub = records.type2_known_sets()[N]
        key = tuple(sorted(index[vec(c)] for c in pub))
        assert key in set(map(tuple, r['solution_indices'])), N


def test_vector_search_finds_short_lengths():
    for N in (9, 10, 11, 12):
        r = vsearch.search(N, 8, time_limit=30, seed=1)
        assert r['found'] and r['verified_peak'] == 0, N
