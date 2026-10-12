"""The composition-split C search (exhaustive/t2abc.c) against the clean-room results."""
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'exhaustive' / 't2abc.c'
CLANG = shutil.which('clang') or '/usr/bin/clang'


@pytest.fixture(scope='module')
def t2abc(tmp_path_factory):
    if not Path(CLANG).exists():
        pytest.skip('clang not found')
    exe = tmp_path_factory.mktemp('t2abc') / 't2abc'
    subprocess.run([CLANG, '-O2', '-pthread', '-o', str(exe), str(SRC)], check=True)
    return exe


def test_same_sets_as_cleanroom(t2abc, tmp_path):
    """Identical solution files for N = 8-12, including the representative code words."""
    for N in range(8, 13):
        out = tmp_path / f'sol_N{N}.txt'
        subprocess.run([str(t2abc), str(N), '-t', '2', '-m', '1', '-o', str(out)],
                       check=True, capture_output=True)
        ref = ROOT / 'results' / 'cleanroom' / f'solutions_N{N}.txt'
        assert out.read_text() == ref.read_text(), N


def test_task_ranges_add_up(t2abc, tmp_path):
    """Two task ranges together find the same sets as one full run."""
    N, P = 11, 101
    lines = []
    for r in ('0:40', '40:101'):
        out = tmp_path / f'part_{r.replace(":", "_")}.txt'
        subprocess.run([str(t2abc), str(N), '-t', '2', '-p', str(P), '-r', r, '-o', str(out)],
                       check=True, capture_output=True)
        lines += out.read_text().splitlines()
    ref = (ROOT / 'results' / 'cleanroom' / f'solutions_N{N}.txt').read_text().splitlines()
    assert sorted(lines) == ref


def test_forced_composition():
    """Every 8-code type-2 set at N = 8-14 has 2 codes of type A, 4 of type B and 2 of type C."""
    for N in (8, 9, 10, 11, 12, 14):
        rows = (ROOT / 'results' / 'cleanroom' / f'solutions_N{N}.txt').read_text().split('\n')[:-1]
        for row in rows:
            types = []
            for h in row.split():
                c = np.array([-1 if (int(h, 16) >> (N - 1 - j)) & 1 else 1 for j in range(N)])
                t = c[:3] * c[N - 3:]                    # lag-(N-3) products
                types.append({(2, 1): 'A', (0, -1): 'B', (-2, 1): 'C'}[(t[0] * t[1] + t[1] * t[2], t[0] * t[2])])
            assert sorted(types) == list('AABBBBCC'), (N, row)
