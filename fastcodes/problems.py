"""Problem definitions: build the arrays the compiled kernels need.

Use the builder functions:

    psl(L, phases=2, doppler=1)          one code, (Doppler) peak sidelobe level
    complementary_set(L, K, phases=2)    K codes, summed autocorrelation
    type2(L, K, phases=2)                K codes, Sulzer type-2 alternating condition

phases=2 is binary, phases=m >= 3 uses the m-th roots of unity, and phases=0
means continuous (arbitrary) phases.
"""
import numpy as np

from . import kernels as KN

METRICS = {'psl': KN.M_PSL, 'isl': KN.M_ISL, 'pnorm': KN.M_PNORM}
KIND_NAMES = {KN.BIN_PSL: 'bin_psl', KN.BIN_TYPE2: 'bin_type2', KN.CPLX_PSL: 'cplx_psl',
              KN.CPLX_SET: 'cplx_set', KN.CPLX_TYPE2: 'cplx_type2'}


class Problem:
    """A search problem. Instances are picklable, so they can go to worker processes."""

    def __init__(self, kind, L, K=1, m=2, N=1, metric='psl', p=4.0, frozen=None, name=''):
        if L < 2:
            raise ValueError('code length must be at least 2')
        self.kind, self.L, self.K, self.m, self.N = kind, L, K, m, N
        self.metric, self.p, self.name = metric, float(p), name
        nv = K * L
        self.nv = nv
        ip = np.zeros(KN.IP_SIZE, dtype=np.int64)
        ip[KN.IP_KIND], ip[KN.IP_NV], ip[KN.IP_L], ip[KN.IP_K] = kind, nv, L, K
        ip[KN.IP_M], ip[KN.IP_N], ip[KN.IP_METRIC] = m, N, METRICS[metric]
        ip[KN.IP_NMOVES] = 1 if self.binary else (m - 1 if m > 0 else 0)
        self.ip = ip

        n_act, maxmag = self._entry_stats()
        integer_valued = self.binary or (kind == KN.CPLX_SET and m == 2)
        scale = 0.25 if integer_valued else 1e-6
        fp = np.zeros(KN.FP_SIZE)
        fp[KN.FP_P] = self.p
        fp[KN.FP_EPS] = scale / (n_act * maxmag ** 2)
        self.fp = fp

        self.roots = (np.exp(2j * np.pi * np.arange(m) / m) if m > 0
                      else np.zeros(0, dtype=np.complex128))
        n = np.arange(L)
        self.dop = np.exp(-2j * np.pi * np.outer(np.arange(N), n) / N).astype(np.complex128)
        self.frozen = (np.zeros(nv, dtype=np.bool_) if frozen is None
                       else np.asarray(frozen, dtype=np.bool_).copy())
        rng = np.random.default_rng(0x5EED + 7919 * kind + 104729 * L + 1299709 * K)
        self.zob = rng.integers(1, 2 ** 63, size=(nv, max(m, 2)), dtype=np.uint64)

    # -- derived properties ------------------------------------------------
    @property
    def binary(self):
        return self.kind in (KN.BIN_PSL, KN.BIN_TYPE2)

    @property
    def continuous(self):
        return self.m == 0

    def _entry_stats(self):
        L, K, N = self.L, self.K, self.N
        if self.kind in (KN.BIN_TYPE2, KN.CPLX_TYPE2):
            n_act = sum(max(0, L - 2 * l) for l in range(1, L))
            return max(n_act, 1), K * L
        if self.kind == KN.CPLX_PSL:
            return N * L - 1, L
        if self.kind == KN.CPLX_SET:
            return L - 1, K * L
        return L - 1, L

    def arrays(self):
        return (self.ip, self.fp, self.dop, self.roots, self.frozen, self.zob)

    def new_state(self):
        L, nv = self.L, self.nv
        xi = np.zeros(nv, dtype=np.int64)
        vals = np.zeros(nv, dtype=np.complex128)
        if self.kind == KN.BIN_PSL:
            Ci, Cc = np.zeros((1, L), np.int64), np.zeros((1, 1), np.complex128)
        elif self.kind == KN.BIN_TYPE2:
            Ci, Cc = np.zeros((L, L), np.int64), np.zeros((1, 1), np.complex128)
        elif self.kind == KN.CPLX_PSL:
            Ci, Cc = np.zeros((1, 1), np.int64), np.zeros((self.N, L), np.complex128)
        elif self.kind == KN.CPLX_SET:
            Ci, Cc = np.zeros((1, 1), np.int64), np.zeros((1, L), np.complex128)
        else:
            Ci, Cc = np.zeros((1, 1), np.int64), np.zeros((L, L), np.complex128)
        return (xi, vals, Ci, Cc)

    def describe(self):
        return dict(kind=KIND_NAMES[self.kind], L=self.L, K=self.K, phases=self.m,
                    doppler=self.N, metric=self.metric, p=self.p, name=self.name)

    def __repr__(self):
        d = self.describe()
        return 'Problem(' + ', '.join(f'{k}={v}' for k, v in d.items() if v not in ('', None)) + ')'

    # -- code conversion ---------------------------------------------------
    def complex_code(self, xi, vals):
        """Complex values of the code (flattened K*L array)."""
        if self.binary:
            return np.asarray(xi, dtype=np.complex128)
        return np.asarray(vals, dtype=np.complex128)

    def export(self, xi, vals):
        """JSON-friendly representation of a code (or code set)."""
        L, K = self.L, self.K
        out = {}
        if self.binary or self.m == 2:
            b = (np.asarray(xi) if self.binary else np.round(np.real(vals)).astype(int))
            rows = [b[r * L:(r + 1) * L] for r in range(K)]
            out['signs'] = [''.join('+' if s > 0 else '-' for s in row) for row in rows]
            out['hex'] = [binary_to_hex(row) for row in rows]
        elif self.m > 0:
            rows = [np.asarray(xi)[r * L:(r + 1) * L] for r in range(K)]
            out['phase_indices'] = [row.tolist() for row in rows]
        else:
            ph = np.angle(np.asarray(vals))
            out['phases'] = [ph[r * L:(r + 1) * L].tolist() for r in range(K)]
        return out

    def state_from_code(self, code):
        """Build xi/vals arrays from +-1 values, phase indices, or complex values."""
        code = np.asarray(code).ravel()
        if code.size != self.nv:
            raise ValueError(f'expected {self.nv} values, got {code.size}')
        xi = np.zeros(self.nv, dtype=np.int64)
        vals = np.zeros(self.nv, dtype=np.complex128)
        if self.binary:
            xi[:] = np.where(np.real(code) > 0, 1, -1)
        elif np.iscomplexobj(code):
            vals[:] = code
            if self.m > 0:
                idx = np.round(np.angle(code) * self.m / (2 * np.pi)).astype(np.int64) % self.m
                xi[:] = idx
                vals[:] = self.roots[idx]
        else:  # integer phase indices
            xi[:] = code.astype(np.int64) % self.m
            vals[:] = self.roots[xi]
        return xi, vals


def binary_to_hex(signs):
    """Hex string, first element = most significant bit, +1 -> 0 and -1 -> 1."""
    bits = ''.join('0' if s > 0 else '1' for s in signs)
    return format(int(bits, 2), 'X').zfill((len(bits) + 3) // 4)


def binary_from_hex(h, L):
    """Inverse of binary_to_hex (the PSL is the same for either bit convention)."""
    bits = bin(int(h, 16))[2:].zfill(L)[-L:]
    return np.array([1 if c == '0' else -1 for c in bits], dtype=np.int64)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def psl(L, phases=2, doppler=1, metric='psl', p=4.0):
    """One code; minimize the (Doppler) peak sidelobe level.

    The first element is frozen to +1: multiplying a code by a constant phase
    does not change any sidelobe magnitude.
    """
    frozen = np.zeros(L, dtype=bool)
    frozen[0] = True
    if phases == 2 and doppler == 1:
        return Problem(KN.BIN_PSL, L, 1, 2, 1, metric, p, frozen, name=f'psl L={L}')
    return Problem(KN.CPLX_PSL, L, 1, phases, doppler, metric, p, frozen,
                   name=f'psl L={L} m={phases} N={doppler}')


def complementary_set(L, K, phases=2, metric='psl', p=4.0):
    """K codes of length L; minimize the peak of the summed autocorrelation.

    A value of 0 means a complementary set (Golay pairs are the K = 2 case).
    The first element of every code is frozen (a constant phase per code does
    not change its autocorrelation).
    """
    frozen = np.zeros(K * L, dtype=bool)
    frozen[::L] = True
    return Problem(KN.CPLX_SET, L, K, phases, 1, metric, p, frozen,
                   name=f'complementary set L={L} K={K} m={phases}')


def type2(L, K, phases=2, metric='isl', p=4.0):
    """K codes of length L; Sulzer (1993) type-2 alternating condition.

    Cost 0 means a type-2 alternating code set. The first two elements of each
    code are frozen: a constant phase and a linear phase ramp per code both
    leave every fourth-order term unchanged (for binary codes these are
    negation and alternate-sign multiplication).
    """
    frozen = np.zeros(K * L, dtype=bool)
    frozen[::L] = True
    frozen[1::L] = True
    kind = KN.BIN_TYPE2 if phases == 2 else KN.CPLX_TYPE2
    return Problem(kind, L, K, phases, 1, metric, p, frozen,
                   name=f'type-2 alternating L={L} K={K} m={phases}')
