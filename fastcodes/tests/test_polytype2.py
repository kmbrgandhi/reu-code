"""Small exact polyphase type-2 searches and saved example sets."""
import json
from pathlib import Path

import numpy as np

from fastcodes import verify
from fastcodes.polytype2 import int_vectors, ksum_zero

RESULTS = Path(__file__).resolve().parents[2] / 'results'


def test_binary_four_code_limit_matches_sulzer():
    """Sulzer (1993, p. 998): 4-code binary type-2 sets exist at N = 7, not above."""
    _, U7, _ = int_vectors(7, 2)
    _, U8, _ = int_vectors(8, 2)
    assert len(ksum_zero(U7, 4)) > 0
    assert len(ksum_zero(U8, 4)) == 0


def test_four_phase_four_code_sets_at_length_8():
    X, U, first = int_vectors(8, 4)
    sols = ksum_zero(U, 4)
    assert len(sols) == 8
    for s in sols:
        codes = np.exp(0.5j * np.pi * X[first[list(s)]])
        assert verify.type2_peak(codes) < 1e-9


def test_saved_doubly_complementary_examples():
    data = json.loads((RESULTS / 'type2_and_complementary.json').read_text())
    for N, d in data.items():
        N = int(N)
        codes = np.array([[1 if b == '0' else -1 for b in bin(int(h, 16))[2:].zfill(N)[-N:]]
                          for h in d['example_hex']])
        assert verify.type2_peak(codes) == 0
        assert verify.set_peak(codes) == 0
