import sys
def vec(x, N):
    c = [-1 if (x >> (N-1-j)) & 1 else 1 for j in range(N)]
    return tuple(sum(c[j]*c[j+l]*c[j+k]*c[j+l+k] for j in range(N-l-k))
                 for l in range(1, N) for k in range(l, N-l))
for N in map(int, sys.argv[1:]):
    lines = open(f"solutions_N{N}.txt").read().split("\n")[:-1]
    seen = set(); bad = 0
    for ln in lines:
        codes = [int(h, 16) for h in ln.split()]
        assert len(codes) == 8 and all(len(h) == (N+3)//4 for h in ln.split())
        assert all(x >> (N-2) == 0 for x in codes)          # c0 = c1 = +1
        vs = [vec(x, N) for x in codes]
        if any(sum(col) for col in zip(*vs)): bad += 1
        seen.add(tuple(sorted(vs)))
    print(f"N={N}: lines={len(lines)} distinct vector-multisets={len(seen)} nonzero-sum lines={bad} sorted={lines == sorted(lines)}")
