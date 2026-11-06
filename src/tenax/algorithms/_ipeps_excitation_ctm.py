"""Independent ket and bra iPEPS on a 2x2 cell, for H_k and N_k at k = (a pi, b pi).

Private contractions behind :func:`tenax.algorithms.ipeps_excitations.compute_spectrum`.
Every closed network here must stay invariant under random gauges on the
environment bonds; see ``tests/test_ipeps_spectrum_contractions.py``.  A reversed
T4/C4 ring matches the energy to 10 digits on a C4v state, so a symmetric check is
not enough.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_multisite,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.core.tensor import DenseTensor

CELL = [(x, y) for x in range(2) for y in range(2)]
NEIGHBORS = {
    (x, y): {
        "left": ((x - 1) % 2, y),
        "right": ((x + 1) % 2, y),
        "top": (x, (y - 1) % 2),
        "bottom": (x, (y + 1) % 2),
    }
    for x, y in CELL
}


def cell_phases(a: int, b: int) -> dict:
    """Phases ``(-1)^(a x + b y)`` of the momentum ``(a pi, b pi)`` on the cell."""
    return {c: (-1.0) ** (a * c[0] + b * c[1]) for c in CELL}


def _ordered(t, labels):
    have = [str(i.label) for i in t.indices]
    return jnp.transpose(t.todense(), [have.index(label) for label in labels])


def unpack_env(env, D: int) -> dict:
    """Raw arrays with each double-layer edge leg split into (ket, bra)."""

    def edge(T):
        return T.reshape(T.shape[0], D, D, T.shape[2])

    return {
        "C1": _ordered(env.C1, ["c1_d", "c1_r"]),
        "C2": _ordered(env.C2, ["c2_l", "c2_d"]),
        "C3": _ordered(env.C3, ["c3_u", "c3_l"]),
        "C4": _ordered(env.C4, ["c4_r", "c4_u"]),
        "T1": edge(_ordered(env.T1, ["t1_l", "u2", "t1_r"])),
        "T2": edge(_ordered(env.T2, ["t2_u", "r2", "t2_d"])),
        "T3": edge(_ordered(env.T3, ["t3_r", "d2", "t3_l"])),
        "T4": edge(_ordered(env.T4, ["t4_d", "l2", "t4_u"])),
    }


def window_1x1(E, K, Kb):
    """Closed one-site window: Z11."""
    return jnp.einsum(
        E["C1"], [0, 1],
        E["T1"], [1, 2, 3, 4],
        E["C2"], [4, 5],
        E["T4"], [0, 7, 8, 6],
        K, [2, 9, 7, 10, 11],
        Kb, [3, 12, 8, 13, 11],
        E["T2"], [5, 10, 13, 14],
        E["C4"], [6, 15],
        E["T3"], [15, 9, 12, 16],
        E["C3"], [14, 16],
        [],
        optimize="greedy",
    )  # fmt: skip


def mixed_rdm_h(EL, ER, K1, Kb1, K2, Kb2):
    """Horizontal bond, K1 left and K2 right: ``(s1, s2, s1_bra, s2_bra)``."""
    L = jnp.einsum(
        EL["C1"], [0, 1],
        EL["T1"], [1, 2, 3, 4],
        EL["T4"], [0, 6, 7, 5],
        EL["C4"], [5, 8],
        EL["T3"], [8, 9, 10, 11],
        K1, [2, 9, 6, 12, 13],
        Kb1, [3, 10, 7, 14, 15],
        [4, 12, 14, 11, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    R = jnp.einsum(
        ER["C2"], [0, 1],
        ER["T1"], [2, 3, 4, 0],
        ER["T2"], [1, 5, 6, 7],
        ER["C3"], [7, 8],
        ER["T3"], [9, 10, 11, 8],
        K2, [3, 10, 12, 5, 13],
        Kb2, [4, 11, 14, 6, 15],
        [2, 12, 14, 9, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    return jnp.einsum(L, [0, 1, 2, 3, 4, 5], R, [0, 1, 2, 3, 6, 7], [4, 6, 5, 7])


def mixed_rdm_v(ET, EB, K1, Kb1, K2, Kb2):
    """Vertical bond, K1 top and K2 bottom: ``(s1, s2, s1_bra, s2_bra)``."""
    U = jnp.einsum(
        ET["C1"], [0, 1],
        ET["T1"], [1, 2, 3, 4],
        ET["C2"], [4, 5],
        ET["T4"], [0, 7, 8, 6],
        ET["T2"], [5, 9, 10, 11],
        K1, [2, 12, 7, 9, 13],
        Kb1, [3, 14, 8, 10, 15],
        [6, 12, 14, 11, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    D_ = jnp.einsum(
        EB["T4"], [0, 1, 2, 3],
        EB["C4"], [3, 4],
        EB["T3"], [4, 5, 6, 7],
        EB["C3"], [8, 7],
        EB["T2"], [9, 10, 11, 8],
        K2, [12, 5, 1, 10, 13],
        Kb2, [14, 6, 2, 11, 15],
        [0, 12, 14, 9, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    return jnp.einsum(U, [0, 1, 2, 3, 4, 5], D_, [0, 1, 2, 3, 6, 7], [4, 6, 5, 7])


def _phys(open_, a, b):
    """Einsum labels of the ket and bra physical legs: traced, or left open."""
    return (a, b) if open_ else (a, a)


def mixed_rdm_diag(Es, kets, bras, *, main: bool):
    """Diagonal pair on a plaquette: ``(s1, s2, s1_bra, s2_bra)``.

    ``Es``, ``kets`` and ``bras`` are ordered ``(TL, TR, BL, BR)``.  ``main``
    leaves TL and BR open (TL=s1, BR=s2); otherwise TR and BL are open
    (TR=s1, BL=s2).
    """
    (E_tl, E_tr, E_bl, E_br), (k_tl, k_tr, k_bl, k_br), (b_tl, b_tr, b_bl, b_br) = (
        Es, kets, bras,
    )  # fmt: skip
    s, t = _phys(main, 10, 13)
    TL = jnp.einsum(
        E_tl["C1"], [0, 1],
        E_tl["T1"], [1, 2, 3, 4],
        E_tl["T4"], [0, 5, 6, 7],
        k_tl, [2, 8, 5, 9, s],
        b_tl, [3, 11, 6, 12, t],
        [4, 7, 8, 11, 9, 12] + ([s, t] if main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(not main, 10, 13)
    TR = jnp.einsum(
        E_tr["T1"], [0, 1, 2, 3],
        E_tr["C2"], [3, 4],
        E_tr["T2"], [4, 5, 6, 7],
        k_tr, [1, 8, 9, 5, s],
        b_tr, [2, 11, 12, 6, t],
        [0, 7, 8, 11, 9, 12] + ([s, t] if not main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(not main, 10, 13)
    BL = jnp.einsum(
        E_bl["T4"], [0, 1, 2, 3],
        E_bl["C4"], [3, 4],
        E_bl["T3"], [4, 5, 6, 7],
        k_bl, [8, 5, 1, 9, s],
        b_bl, [11, 6, 2, 12, t],
        [7, 0, 8, 11, 9, 12] + ([s, t] if not main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(main, 10, 13)
    BR = jnp.einsum(
        E_br["T3"], [0, 1, 2, 3],
        E_br["C3"], [4, 3],
        E_br["T2"], [5, 6, 7, 4],
        k_br, [8, 1, 9, 6, s],
        b_br, [11, 2, 12, 7, t],
        [0, 5, 8, 11, 9, 12] + ([s, t] if main else []),
        optimize="greedy",
    )  # fmt: skip
    if main:
        left = jnp.einsum(
            TL,
            [0, 1, 2, 3, 4, 5, 6, 7],
            BL,
            [8, 1, 2, 3, 9, 10],
            [0, 4, 5, 8, 9, 10, 6, 7],
        )
        right = jnp.einsum(
            TR,
            [0, 1, 2, 3, 4, 5],
            BR,
            [6, 1, 2, 3, 7, 8, 9, 10],
            [0, 4, 5, 6, 7, 8, 9, 10],
        )
        return jnp.einsum(
            left,
            [0, 1, 2, 3, 4, 5, 6, 7],
            right,
            [0, 1, 2, 3, 4, 5, 8, 9],
            [6, 8, 7, 9],
        )
    left = jnp.einsum(
        TL, [0, 1, 2, 3, 4, 5], BL, [6, 1, 2, 3, 7, 8, 9, 10], [0, 4, 5, 6, 7, 8, 9, 10]
    )
    right = jnp.einsum(
        TR, [0, 1, 2, 3, 4, 5, 6, 7], BR, [8, 1, 2, 3, 9, 10], [0, 4, 5, 8, 9, 10, 6, 7]
    )
    return jnp.einsum(
        left, [0, 1, 2, 3, 4, 5, 8, 9], right, [0, 1, 2, 3, 4, 5, 6, 7], [6, 8, 7, 9]
    )


def z00(c1, c2, c3, c4):
    return jnp.einsum(
        c1, [0, 1], c2, [1, 2], c3, [2, 3], c4, [0, 3], [], optimize="greedy"
    )


def z10(top, bot):
    """``C1 T1 C2`` of the row below the cut, closed with ``C3 T3 C4`` of the row above."""
    return jnp.einsum(
        top["C1"], [0, 1],
        top["T1"], [1, 2, 3, 4],
        top["C2"], [4, 5],
        bot["C3"], [5, 6],
        bot["T3"], [7, 2, 3, 6],
        bot["C4"], [0, 7],
        [],
        optimize="greedy",
    )  # fmt: skip


def z01(left, right):
    """``C1 T4 C4`` of the column right of the cut, closed with ``C2 T2 C3`` of the column left."""
    return jnp.einsum(
        left["C1"], [0, 1],
        right["C2"], [1, 2],
        right["T2"], [2, 3, 4, 5],
        right["C3"], [5, 6],
        left["C4"], [7, 6],
        left["T4"], [0, 3, 4, 7],
        [],
        optimize="greedy",
    )  # fmt: skip


class ExcitationCTM:
    """Energy and log-norm densities of a ket and bra that differ by cell phases."""

    def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
        self.A = jnp.asarray(A)
        self.D, self.d = self.A.shape[0], self.A.shape[-1]
        self.gate = jnp.asarray(gate).reshape((self.d,) * 4)
        self.gate_nnn = (
            None if gate_nnn is None else jnp.asarray(gate_nnn).reshape((self.d,) * 4)
        )
        self.chi, self.sweeps = chi, sweeps
        site = _wrap_as_dense_tensor(self.A)
        self._template = _build_double_layer_tensor(site).indices
        envs, self.ctm_converged = _converged_env({c: site for c in CELL}, chi)
        self.envs0 = jax.tree_util.tree_map(jax.lax.stop_gradient, envs)
        self._sweep = jax.jit(
            lambda e, dl: _ctm_tensor_sweep_multisite(
                e, dl, NEIGHBORS, chi, True, "svd", "flow", "2x2"
            )[0]
        )

    def _double_layer(self, K, Kb):
        D2 = self.D**2
        data = jnp.einsum("udlrs,UDLRs->uUdDlLrR", K, Kb).reshape(D2, D2, D2, D2)
        return DenseTensor(data, self._template)

    def environments(self, kets, bras):
        dl = {c: self._double_layer(kets[c], bras[c]) for c in CELL}
        e = self.envs0
        for _ in range(self.sweeps):
            e = self._sweep(e, dl)
        return {c: unpack_env(e[c], self.D) for c in CELL}

    def energy_density(self, E, kets, bras):
        total = 0.0
        for c in CELL:
            for direction, fn in (("right", mixed_rdm_h), ("bottom", mixed_rdm_v)):
                n = NEIGHBORS[c][direction]
                rho = fn(E[c], E[n], kets[c], bras[c], kets[n], bras[n])
                total = total + _bond_value(rho, self.gate)
            if self.gate_nnn is not None:
                cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
                cxy = NEIGHBORS[cx]["bottom"]
                sites = (c, cx, cy, cxy)
                Es = tuple(E[x] for x in sites)
                ks = tuple(kets[x] for x in sites)
                bs = tuple(bras[x] for x in sites)
                for main in (True, False):
                    total = total + _bond_value(
                        mixed_rdm_diag(Es, ks, bs, main=main), self.gate_nnn
                    )
        return total / len(CELL)

    def log_norm_density(self, E, kets, bras):
        total = 0.0
        for c in CELL:
            cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
            cxy = NEIGHBORS[cx]["bottom"]
            total = (
                total
                + jnp.log(jnp.abs(window_1x1(E[c], kets[c], bras[c])))
                + jnp.log(
                    jnp.abs(z00(E[cxy]["C1"], E[cy]["C2"], E[c]["C3"], E[cx]["C4"]))
                )
                - jnp.log(jnp.abs(z10(E[cy], E[c])))
                - jnp.log(jnp.abs(z01(E[cx], E[c])))
            )
        return total / len(CELL)

    def densities(self, phases, eps, X, Y):
        """``(energy density, log-norm density)`` at ket ``A + eps phi X``, bra ``A + phi Y``."""
        kets = {c: self.A + eps * phases[c] * X for c in CELL}
        bras = {c: self.A + phases[c] * Y for c in CELL}
        E = self.environments(kets, bras)
        return self.energy_density(E, kets, bras), self.log_norm_density(E, kets, bras)

    def ket_derivative(self, phases, X, h):
        """``d/deps d/dY`` of both densities at the base point: central difference on the ket."""
        zero = jnp.zeros_like(self.A)
        grads = []
        for sign in (1.0, -1.0):
            _, pull = jax.vjp(lambda Y: self.densities(phases, sign * h, X, Y), zero)
            grads.append(
                (
                    pull((jnp.ones(()), jnp.zeros(())))[0],
                    pull((jnp.zeros(()), jnp.ones(())))[0],
                )
            )
        scale = 2 * h
        return (grads[0][0] - grads[1][0]) / scale, (grads[0][1] - grads[1][1]) / scale


def _bond_value(rho, gate):
    return jnp.einsum("ijkl,klij->", rho, gate) / jnp.einsum("ijij->", rho)


def _converged_env(sites, chi):
    envs, info, _ = _ctm_tensor_multisite(
        sites, NEIGHBORS, chi, max_iter=600, conv_tol=1e-11,
        _return_status=True, _warn_unconverged=False,
    )  # fmt: skip
    return envs, bool(info.converged)
