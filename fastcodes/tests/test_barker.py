"""Tests for the polyphase Barker mode (fastcodes/barker.py)."""
import numpy as np

from fastcodes import barker as BK


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for mode, p, tau in [(0, 2.0, 0.0), (0, 6.0, 0.0), (1, 2.0, 0.5)]:
        phi = rng.uniform(0, 2 * np.pi, 19)
        _, g, _ = BK.objective(phi, mode, p, tau)
        fd = np.zeros_like(phi)
        for i in range(len(phi)):
            e = np.zeros_like(phi)
            e[i] = 1e-6
            fd[i] = (BK.objective(phi + e, mode, p, tau)[0] - BK.objective(phi - e, mode, p, tau)[0]) / 2e-6
        assert np.abs(g - fd).max() <= 1e-6 * max(1.0, np.abs(g).max())


def test_published_nunn_coxson_codes_certify():
    """All nine published codes (N = 65-70, 72, 76, 77) pass the interval check."""
    codes = BK.known_codes()
    assert sorted(codes) == [65, 66, 67, 68, 69, 70, 72, 76, 77]
    for N, ph in codes.items():
        assert len(ph) == N
        assert BK.certify(radians=ph)['certified'], N


def test_certification_rejects_a_violation():
    ph = [float(x) for x in BK.known_codes()[69]]
    ph[30] += 0.05
    assert BK.interior_peak(np.array(ph)) > 1.0
    assert not BK.certify(radians=[repr(x) for x in ph])['certified']


def test_quantized_phases_roundtrip():
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, 30)
    m = BK.quantize(phi)
    back = 2 * np.pi * np.array(m, dtype=np.float64) / BK.H_DEFAULT
    assert abs(BK.interior_peak(back) - BK.interior_peak(phi)) < 1e-9


def test_search_finds_and_certifies_short_lengths():
    for N in (13, 17):
        r = BK.search(N, time_limit=60, seed=3)
        assert r['found'] and r['certification']['certified'], N
