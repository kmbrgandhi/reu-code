"""Published reference values and known codes, with sources.

Every value here comes from a primary source named next to it. The tests in
tests/ recompute every listed code with fastcodes.verify.
"""
import csv
import os

import numpy as np

from .problems import binary_from_hex

_DATA = os.path.join(os.path.dirname(__file__), 'data')


def binary_min_psl():
    """Minimum (or best known) binary PSL for lengths 2-105.

    Returns {L: dict(psl, status, example_hex, source, note)}. Sources:
    L = 2-68  Leukhin & Potekhin 2012, arXiv:1212.4930, Table 2 (exhaustive);
              also matches Leukhin & Potekhin 2013 (EuRAD), Table II, p. 497.
    L = 69-82 Leukhin & Potekhin 2015, J. Phys. Conf. Ser. 613:012006, Table 1
              (exhaustive).
    L = 83-84 PSL 5, optimal per Dimitrov et al. 2021, arXiv:2104.10477, p. 1.
    L = 85-105 PSL 5, best known (Nunn & Coxson 2008, IEEE TAES 44(1),
              Table 2). Not proven optimal. The printed L = 103 code has a typo.
    """
    out = {}
    with open(os.path.join(_DATA, 'binary_min_psl.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out[int(r['L'])] = dict(psl=int(r['PSL']), status=r['status'],
                                    example_hex=r['example_hex'], source=r['source'],
                                    note=r.get('note', ''))
    return out


def binary_best_known():
    """Best published binary PSL by length: binary_min_psl() for 2-105, plus
    data/binary_best_known_106_300.tsv when present (each row cites its source).
    Returns {L: dict(psl, status, source)}."""
    out = {L: dict(psl=r['psl'], status=r['status'], source=r['source'], note=r['note'],
                   example_hex=r['example_hex'])
           for L, r in binary_min_psl().items()}
    path = os.path.join(_DATA, 'binary_best_known_106_300.tsv')
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r.get('best_psl', '').strip():
                    out[int(r['L'])] = dict(psl=int(r['best_psl']), status='best known (published)',
                                            source=r['source'], note=r.get('note', ''),
                                            example_hex=r['example_hex'].replace(' ', ''))
    return out


def binary_min_psl_code(L):
    """The example code for length L from binary_min_psl(), as +1/-1 values."""
    return binary_from_hex(binary_min_psl()[L]['example_hex'], L)


# Type-2 alternating code sets (Sulzer 1993 condition), 8 codes each.

# Sulzer, M. P. (1993), A new type of alternating code for incoherent scatter
# measurements, Radio Science 28(6), 995-1001, Table 4 (p. 1001). Octal
# numbers; expand each to N bits.
SULZER_1993_TABLE4 = {
    8: ['0', '41', '15', '64', '11', '33', '23', '26'],
    9: ['161', '17', '124', '62', '13', '46', '64', '21'],
    10: ['7', '306', '142', '364', '11', '121', '73', '64'],
    11: ['7', '621', '22', '713', '274', '326', '206', '243'],
    12: ['167', '1101', '262', '1663', '524', '236', '554', '371'],
}

# Markkanen, Vierinen & Markkanen (2008), Polyphase alternating codes,
# Ann. Geophys. 26, 2237-2243, p. 2237, footnote 1: "The 8 sequences forming
# the 14-bit code are in hexadecimal format". The paper calls this the longest
# type-2 alternating code the authors knew of.
MARKKANEN_2008_TYPE2_14 = ['406', 'b72', '4df', 'bab', '24c', 'd38', '295', 'de1']


def type2_known_sets():
    """Published type-2 sets as {length: array (8, length) of +1/-1}."""
    out = {}
    for N, octs in SULZER_1993_TABLE4.items():
        out[N] = np.array([[1 if b == '0' else -1 for b in format(int(c, 8), f'0{N}b')]
                           for c in octs])
    out[14] = np.array([[1 if b == '0' else -1 for b in format(int(h, 16), '014b')]
                        for h in MARKKANEN_2008_TYPE2_14])
    return out
