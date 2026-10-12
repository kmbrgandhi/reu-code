"""Correctness tests. Run from the repo root: .venv/bin/python -m pytest fastcodes/tests -q"""
import numpy as np
import pytest

from fastcodes import complementary_set, kernels as KN, psl, records, type2, verify
from fastcodes.problems import binary_from_hex, binary_to_hex
from fastcodes.search import ALGOS, PRESETS, search

ALL_KINDS = [
    psl(23), psl(31, metric='isl'), psl(19, metric='pnorm', p=6),
    psl(17, phases=4, doppler=3), psl(15, phases=0, doppler=2), psl(12, phases=2, doppler=4),
    complementary_set(9, 4), complementary_set(7, 3, phases=5), complementary_set(6, 2, phases=0),
    type2(11, 8), type2(10, 8, metric='psl'), type2(9, 4, phases=3), type2(7, 4, phases=0),
]


def _random_walk(pb, steps, seed):
    P, S = pb.arrays(), pb.new_state()
    KN.seed_rng(seed)
    rng = np.random.default_rng(seed)
    KN.random_init(P, S)
    worst_pred = worst_fresh = 0.0
    for _ in range(steps):
        v = KN.random_free_var(P)
        if pb.continuous:
            z = np.exp(1j * rng.uniform(0, 2 * np.pi))
            pred = KN.eval_value(P, S, v, z, np.inf)
            KN.apply_value(P, S, v, z)
        else:
            t = KN.random_t(P)
            pred = KN.eval_move(P, S, v, t, np.inf)
            KN.apply_move(P, S, v, t)
        inc = KN.full_cost(P, S)
        S2 = pb.new_state()
        KN.load_code(P, S2, S[0], S[1])
        worst_pred = max(worst_pred, abs(pred - inc))
        worst_fresh = max(worst_fresh, abs(inc - KN.full_cost(P, S2)))
    return P, S, worst_pred, worst_fresh


@pytest.mark.parametrize('pb', ALL_KINDS, ids=lambda p: p.name + ' ' + p.metric)
def test_incremental_updates_match_full_recompute(pb):
    P, S, e_pred, e_fresh = _random_walk(pb, 200, 7)
    tol = 0.0 if (pb.binary and pb.metric != 'pnorm') else 1e-8
    assert e_pred <= tol and e_fresh <= tol
    # and the kernel's peak matches the independent NumPy evaluator
    assert abs(KN.peak_value(P, S) - verify.evaluate(pb, S[0], S[1])) < 1e-9


@pytest.mark.parametrize('pb', ALL_KINDS[:6] + ALL_KINDS[9:11], ids=lambda p: p.name)
def test_early_exit_bound(pb):
    """eval_move(bound) is exact when the cost is <= bound, else returns > bound."""
    P, S = pb.arrays(), pb.new_state()
    KN.seed_rng(3)
    KN.random_init(P, S)
    for v in range(pb.nv):
        if pb.frozen[v] or pb.continuous:
            continue
        exact = KN.eval_move(P, S, v, 1, np.inf)
        for bound in (exact * 0.5, exact - 1e-9, exact, exact * 2):
            got = KN.eval_move(P, S, v, 1, bound)
            if exact <= bound:
                assert got == pytest.approx(exact, abs=1e-12)
            else:
                assert got > bound


@pytest.mark.parametrize('L,code', [
    (2, [1, 1]), (3, [1, 1, -1]), (4, [1, 1, -1, 1]), (5, [1, 1, 1, -1, 1]),
    (7, [1, 1, 1, -1, -1, 1, -1]), (11, [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
    (13, [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])])
def test_barker_codes_have_psl_1(L, code):
    pb = psl(L)
    P, S = pb.arrays(), pb.new_state()
    KN.load_code(P, S, np.array(code), np.zeros(L, complex))
    assert KN.peak_value(P, S) == 1


def test_published_binary_table_codes():
    """Every example code in data/binary_min_psl.tsv has the listed PSL,
    except the known L = 103 typo in the source."""
    table = records.binary_min_psl()
    assert sorted(table) == list(range(2, 106))
    for L, row in table.items():
        c = binary_from_hex(row['example_hex'], L)
        pb = psl(L)
        P, S = pb.arrays(), pb.new_state()
        KN.load_code(P, S, c, np.zeros(L, complex))
        got = KN.peak_value(P, S)
        assert got == verify.doppler_psl(c)
        if L == 103:
            assert got == 10 and 'typo' in row['note']
        else:
            assert got == row['psl'], L


def test_published_best_known_106_300():
    """Every example code in data/binary_best_known_106_300.tsv has the listed
    PSL, except rows flagged MISMATCH (L = 251: the claimed 10 is not backed by
    the printed code, which gives 11)."""
    table = records.binary_best_known()
    assert all(L in table for L in range(106, 301))
    flagged = []
    for L in range(106, 301):
        row = table[L]
        got = verify.doppler_psl(binary_from_hex(row['example_hex'], L))
        if 'MISMATCH' in row['note']:
            flagged.append(L)
            assert got > row['psl']
        else:
            assert got == row['psl'], L
    assert flagged == [251]


def test_published_type2_sets():
    """Sulzer (1993) Table 4 and Markkanen et al. (2008) sets satisfy the
    type-2 condition, and (as those papers say) they are neither complementary
    sets nor Lehtinen type-1 codes."""
    for N, codes in records.type2_known_sets().items():
        assert verify.type2_peak(codes) == 0, N
        assert verify.set_peak(codes) > 0, N
        assert verify.lehtinen_weak_violation(codes) > 0, N
        pb = type2(N, 8)
        P, S = pb.arrays(), pb.new_state()
        KN.load_code(P, S, codes.ravel(), np.zeros(8 * N, complex))
        assert KN.full_cost(P, S) == 0 and KN.peak_value(P, S) == 0


def test_type2_symmetries_justify_frozen_positions():
    """Negating a code or multiplying it by (-1)^n leaves the type-2 terms
    unchanged, so fixing the first two elements of each code loses nothing."""
    rng = np.random.default_rng(5)
    codes = rng.choice([-1, 1], size=(8, 13))
    base = verify.type2_peak(codes)
    alt = (-1) ** np.arange(13)
    t = codes.copy()
    t[0] *= -1
    t[3] *= alt
    t[5] = -t[5] * alt
    assert verify.type2_peak(t) == base
    # and the m-phase analogue: constant phase and linear phase ramp per code
    z = np.exp(2j * np.pi * rng.integers(0, 5, size=(4, 9)) / 5)
    ramp = np.exp(2j * np.pi * 2 * np.arange(9) / 5)
    z2 = z.copy()
    z2[1] *= np.exp(2j * np.pi / 5) * ramp
    assert abs(verify.type2_peak(z2) - verify.type2_peak(z)) < 1e-9


def test_golay_pair_is_complementary():
    a = [1, 1, 1, -1, 1, 1, -1, 1]
    b = [1, 1, 1, -1, -1, -1, 1, -1]
    pb = complementary_set(8, 2)
    P, S = pb.arrays(), pb.new_state()
    KN.load_code(P, S, np.zeros(16, np.int64), np.array(a + b, complex))
    assert KN.peak_value(P, S) == 0


def test_hex_roundtrip():
    rng = np.random.default_rng(1)
    for L in (5, 13, 64, 107):
        c = rng.choice([-1, 1], L)
        assert np.array_equal(binary_from_hex(binary_to_hex(c), L), c)


@pytest.mark.parametrize('algo', sorted(ALGOS) + sorted(PRESETS))
def test_every_algorithm_returns_a_verified_result(algo):
    from fastcodes.search import resolve
    a, _, _ = resolve(algo)
    pb = psl(11, phases=0) if a.continuous else psl(15)
    r = search(pb, algo, time_limit=0.3, seed=11)
    assert r.verified
    pb = type2(6, 4, phases=0) if a.continuous else type2(8, 8)
    r = search(pb, algo, time_limit=0.3, seed=11)
    assert r.verified


def test_searches_reach_known_optima():
    r = search(psl(13), 'sdls', time_limit=10, target=1, seed=2)
    assert r.hit_target and r.verified
    r = search(psl(40), 'memetic', time_limit=30, target=3, seed=2)
    assert r.hit_target and r.verified
    r = search(type2(9, 8), 'tabu', time_limit=30, target=0, seed=2)
    assert r.hit_target and r.verified and r.verified_peak == 0
