"""fastcodes: fast, verified searches for low-sidelobe radar codes.

A numba rewrite of the 2017 REU_code heuristics (see fastcodes/README.md).
"""
from .problems import Problem, psl, complementary_set, type2, binary_from_hex, binary_to_hex

__all__ = ['Problem', 'psl', 'complementary_set', 'type2', 'binary_from_hex', 'binary_to_hex']
