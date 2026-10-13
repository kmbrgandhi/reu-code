"""Chunked runs of the clean-room C search (cleanroom/t2search.c -r) and their merge checks."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'cleanroom' / 't2search.c'
MERGE = ROOT / 'cleanroom' / 'merge_chunks.py'
CLANG = shutil.which('clang') or '/usr/bin/clang'


@pytest.fixture(scope='module')
def t2search(tmp_path_factory):
    if not Path(CLANG).exists():
        pytest.skip('clang not found')
    exe = tmp_path_factory.mktemp('t2search') / 't2search'
    subprocess.run([CLANG, '-O2', '-pthread', '-o', str(exe), str(SRC)], check=True)
    return exe


def n_tasks(exe, N):
    out = subprocess.run([str(exe), str(N), '-n'], check=True, capture_output=True, text=True).stdout
    return int(re.search(r'tasks=(\d+)', out).group(1))


def run_chunks(exe, N, cuts, d):
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        with open(d / f'chunk_{i}.log', 'w') as log:
            subprocess.run([str(exe), str(N), '-t', '2', '-q', '-r', f'{a}:{b}', '-o', str(d / f'chunk_{i}.txt')],
                           check=True, stdout=log, stderr=subprocess.STDOUT)


def merge(N, d):
    return subprocess.run([sys.executable, str(MERGE), str(N), str(d), str(d / 'merged.txt')],
                          capture_output=True, text=True)


def test_chunks_merge_to_full_result(t2search, tmp_path):
    """Uneven chunks merge to the same solution file as one full run, and all checks pass."""
    for N in (10, 11, 12):
        P = n_tasks(t2search, N)
        d = tmp_path / f'N{N}'
        d.mkdir()
        run_chunks(t2search, N, [0, P // 3, P // 3 + 1, P // 2, P], d)
        r = merge(N, d)
        assert r.returncode == 0, r.stdout
        assert r.stdout.count(' OK') == 3, r.stdout
        ref = ROOT / 'results' / 'cleanroom' / f'solutions_N{N}.txt'
        assert (d / 'merged.txt').read_text() == ref.read_text(), N


def test_merge_rejects_missing_tasks(t2search, tmp_path):
    P = n_tasks(t2search, 11)
    run_chunks(t2search, 11, [0, P // 2], tmp_path)
    r = merge(11, tmp_path)
    assert r.returncode == 1 and 'missing' in r.stdout
    assert not (tmp_path / 'merged.txt').exists()


def test_merge_rejects_mixed_seeds(t2search, tmp_path):
    P = n_tasks(t2search, 11)
    run_chunks(t2search, 11, [0, P // 2, P], tmp_path)
    with open(tmp_path / 'chunk_1.log', 'w') as log:
        subprocess.run([str(t2search), '11', '-t', '2', '-q', '-s', '7', '-r', f'{P // 2}:{P}',
                        '-o', str(tmp_path / 'chunk_1.txt')], check=True, stdout=log, stderr=subprocess.STDOUT)
    r = merge(11, tmp_path)
    assert r.returncode == 1 and 'different setup' in r.stdout
