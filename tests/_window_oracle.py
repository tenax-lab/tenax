"""Window oracle: C(r) = <O^dag_r O_0> on rectangular CTM windows, by direct sum.

A reference for N_k that shares no code with compute_spectrum: it does no
differentiation and no perturbed CTM. It is converged only at D=2. Certified on
the Ising PEPS against Onsager in test_ipeps_spectrum_contractions.py.
"""

import numpy as np
from scipy.special import ellipk


class Window:
    """All tensors are plain numpy; site tensors are closed (u, d, l, r) of dim D^2."""

    def __init__(self, env):
        self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4 = [
            np.asarray(x) for x in env
        ]

    def left(self, h):
        L = self.C1.T  # (t, a)
        for _ in range(h):
            # (t, x.., a) (a, x, a') -> (t, x.., x, a')
            L = np.tensordot(L, self.T4, axes=(L.ndim - 1, 0))
        return np.tensordot(L, self.C4, axes=(L.ndim - 1, 0))  # (t, x1..xh, b)

    def right(self, h):
        R = self.C2  # (t, a)
        for _ in range(h):
            R = np.tensordot(R, self.T2, axes=(R.ndim - 1, 0))
        return np.tensordot(R, self.C3, axes=(R.ndim - 1, 0))  # (t, r1..rh, k=b)

    def apply(self, L, col):
        """Absorb one column; col[i] is the closed site tensor of row i."""
        h = len(col)
        # (t, x1..xh, b) -> (t', u, x1..xh, b)
        L = np.tensordot(self.T1, L, axes=(0, 0))  # (u, t', x1..xh, b)
        L = np.moveaxis(L, 1, 0)  # (t', u, x1..xh, b)
        for i, a in enumerate(col):
            # L = (t', u, r1..r_i, x_{i+1}..x_h, b); contract u and x_{i+1}
            xpos = 2 + i
            L = np.tensordot(L, a, axes=((1, xpos), (0, 2)))  # a: (u,d,l,r)
            # now (t', r1..r_i, x_{i+2}..x_h, b, d, r_new)
            L = np.moveaxis(L, -2, 1)  # d -> slot 1
            L = np.moveaxis(L, -1, 2 + i)  # r_new -> after r1..r_i
        # (t', d, r1..rh, b) with T3 (b, d, b')
        L = np.tensordot(L, self.T3, axes=((h + 2, 1), (0, 1)))
        return L  # (t', r1..rh, b')

    def close(self, L, h):
        return np.tensordot(L, self.right(h), axes=L.ndim)


def correlators(env, aa, ket, bra, both, R):
    """C[(m, n)] for |m|,|n| <= R, r = pos(bra) - pos(ket), y pointing down.

    aa/ket/bra/both are closed double-layer tensors: no operator, O on the ket,
    O^dag on the bra, and both on one site.  C(-r) = conj C(r) fills the rest.
    """
    w = Window(env)
    C = {}
    for n in range(R + 1):
        h = n + 1
        Rc = w.right(h)
        L0 = w.left(h)
        for ket_row, bra_row in [(0, n)] if n == 0 else [(0, n), (n, 0)]:
            sgn = 1 if ket_row == 0 else -1  # bra below ket: +n; bra above: -n
            col = [aa] * h
            nv = w.apply(L0, col)
            if ket_row == 0:  # the m = 0 column holds both insertions
                c0 = list(col)
                if n == 0:
                    c0[0] = both
                else:
                    c0[0], c0[n] = ket, bra
                val = np.tensordot(w.apply(L0, c0), Rc, axes=nv.ndim)
                C[(0, n)] = val / np.tensordot(nv, Rc, axes=nv.ndim)
            c1 = list(col)
            c1[ket_row] = ket
            v = w.apply(L0, c1)
            cb = list(col)
            cb[bra_row] = bra
            for m in range(1, R + 1):
                val = np.tensordot(w.apply(v, cb), Rc, axes=nv.ndim)
                norm = np.tensordot(w.apply(nv, col), Rc, axes=nv.ndim)
                C[(m, sgn * n)] = val / norm
                if m < R:
                    v = w.apply(v, col)
                    nv = w.apply(nv, col)
                    # keep the scale bounded on long sweeps
                    s = np.linalg.norm(nv)
                    v, nv = v / s, nv / s
    for (m, n), val in list(C.items()):
        C[(-m, -n)] = np.conj(val)
    return C


def closed_double_layer(K, Kb):
    """``(u, d, l, r)`` of dimension ``D^2``, with the physical leg traced."""
    D2 = K.shape[0] ** 2
    return np.einsum("udlrs,UDLRs->uUdDlLrR", K, Kb).reshape(D2, D2, D2, D2)


def norm_form(env, A, B, one_site_expectation, a, b, R):
    """``N_k(B, B)`` at ``k = (a pi, b pi) != 0``: the sum, less the disconnected part.

    ``one_site_expectation`` is ``<B>_ket <B>_bra``, the constant that ``C(r)``
    approaches at large ``r``.
    """
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        R,
    )
    return sum(
        np.cos(np.pi * (a * m + b * n)) * (np.real(value) - one_site_expectation)
        for (m, n), value in C.items()
    )


def ising_peps(beta):
    """``|psi|^2`` of this PEPS is the classical 2D Ising partition function."""
    W = np.array(
        [
            [np.exp(beta / 2), np.exp(-beta / 2)],
            [np.exp(-beta / 2), np.exp(beta / 2)],
        ]
    )
    e, V = np.linalg.eigh(W)
    M = V @ np.diag(np.sqrt(e)) @ V.T
    A = np.einsum("su,sd,sl,sr->udlrs", M, M, M, M)
    return A / np.linalg.norm(A)


def onsager_c10(beta):
    """Exact nearest-neighbour correlation of the 2D Ising model."""
    t = np.tanh(2 * beta)
    kappa = 2 * np.sinh(2 * beta) / np.cosh(2 * beta) ** 2
    return 0.5 / t * (1 + (2 / np.pi) * (2 * t**2 - 1) * ellipk(kappa**2))
