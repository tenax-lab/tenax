"""Implicit AD through the charged C4v (2-sublattice) CTM sweep (spec 2).

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.

``ctm_tensor_c4v`` runs the #1211 sweep on a charged C4v ``SymmetricTensor``
A, as the 2-sublattice state (A, negate(A)).  It ranks its projector on the
host, so it is eager-only.  This module differentiates the sweep at a
frozen layout L:

- ``_sweep_at`` is F_L: the sweep with the traced projector
  ``_c4v_staggered_projectors_at(Qf, L)``, both outputs normalised by
  ``_normalize_by_max_abs``.
- ``_align`` fixes the one free sign per chi index against a reference T:
  F~(e; A, ref) = align(F_L(e, a(A)), ref).
- ``charged_c4v_cold_start`` runs the #1211 loop to sv-convergence and
  freezes the layout of its last sweep.  ``_elementwise_phase`` iterates
  e <- F~(e; A, ref=e) to an element-wise fixed point.
- ``charged_c4v_converge`` is that phase as a ``jax.custom_vjp``.  Its
  backward solves (I - J^T + tau I) lambda = g with ``gmres_eager`` and
  returns dA = (dF~/dA)^T lambda.

The step, the adjoint matvec and the site VJP compile once per layout.
"""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_project,
    _c4v_staggered_projectors_at,
    _leg,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ctm_tensor_moves import _normalize_by_max_abs
from tenax.algorithms.ad_utils import (
    _bond_blocks,
    _bond_component_phases,
    _bond_rebuild,
    _bond_ref_blocks,
)

__all__ = [
    "_align",
    "_aligned_sweep",
    "_chi_components",
    "_layout_of",
    "_rel_change",
    "_scale_chi",
    "_step",
    "_step_impl",
    "_sweep_at",
]


def _layout_of(C) -> BondLayout:
    """The layout of a 2-sublattice corner: the kept states per ``c_a`` charge."""
    return BondLayout.from_dims(Counter(np.asarray(_leg(C, "c_a").charges).tolist()))


def _chi_components(T) -> tuple[np.ndarray, ...]:
    """The chi positions that the blocks of T connect, as static index arrays.

    Union-find over the positions of t_l and t_r: a block joins every
    position it touches.  ``_bond_component_phases`` fixes the signs of a
    component up to one sign, which cancels on T (both chi legs carry the
    same sign).
    """
    labels = tuple(i.label for i in T.indices)
    a, b = labels.index("t_l"), labels.index("t_r")
    parent = list(range(T.indices[a].dim))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _key, blk, pos in _bond_blocks(T):
        if blk.size == 0:
            continue
        members = [int(p) for p in pos[a]] + [int(p) for p in pos[b]]
        root = find(members[0])
        for m in members[1:]:
            parent[find(m)] = root
    groups: dict[int, list[int]] = {}
    for i in range(len(parent)):
        groups.setdefault(find(i), []).append(i)
    return tuple(np.asarray(g, dtype=np.int64) for g in sorted(groups.values()))


def _scale_chi(X, g, chi_labels):
    """``X`` with the factor ``g[i]`` on chi position i of each leg in ``chi_labels``."""
    labels = tuple(i.label for i in X.indices)
    axes = [labels.index(lab) for lab in chi_labels]
    out = []
    for _key, blk, pos in _bond_blocks(X):
        for ax in axes:
            shape = [1] * blk.ndim
            shape[ax] = -1
            blk = blk * g[pos[ax]].reshape(shape)
        out.append(blk)
    return _bond_rebuild(X, out)


def _align(C_new, T_new, T_ref):
    """Fix the free sign of each chi index against ``T_ref`` (spec 2, section 1).

    Each projector column has a free sign.  Column i of P_right is column i
    of P_left, so a sign g_i acts on chi position i of both legs:
    T_new = g T g and C_new = g C g.  Then
    X[i, k] = sum_d T_new[i, d, k] T_ref[i, d, k] = g_i g_k W[i, k] with
    W >= 0 at the fixed point, the form that ``_bond_component_phases``
    reads.  The signs are piecewise constant, so they come from
    ``stop_gradient`` inputs, and ``_bond_ref_blocks`` stops the reference.
    C is diagonal at a fresh projection, so the signs leave it unchanged;
    they are applied anyway, to keep the map a gauge for any C.
    """
    labels = tuple(i.label for i in T_new.indices)
    a, b = labels.index("t_l"), labels.index("t_r")
    n = T_new.indices[a].dim
    refs = _bond_ref_blocks(T_new, T_ref)
    X = jnp.zeros((n, n), dtype=T_new._data.dtype)
    for key, blk, pos in _bond_blocks(T_new):
        rblk = refs.get(key)
        if rblk is None or blk.size == 0:
            continue
        M = jax.lax.stop_gradient(blk) * rblk
        rest = tuple(k for k in range(M.ndim) if k not in (a, b))
        M = jnp.sum(M, axis=rest)
        if a > b:
            M = M.T
        X = X.at[np.ix_(pos[a], pos[b])].add(M)
    g = jax.lax.stop_gradient(_bond_component_phases(X + X.T, _chi_components(T_new)))
    return _scale_chi(C_new, g, ("c_a", "c_b")), _scale_chi(T_new, g, ("t_l", "t_r"))


def _sweep_at(A, C, T, layout: BondLayout):
    """F_L: one sweep at the frozen layout, both outputs normalised by max |.|.

    ``_normalize_by_max_abs`` differentiates the scale as c |X|_F, so a
    cotangent along the output has a zero VJP; the plain ``max_abs()`` of
    the eager sweep does not (spec 2, section 2).
    """
    a = _build_double_layer_tensor(A)
    Qf = _c4v_staggered_enlarged_corner(C, T, a)
    P_left, P_right = _c4v_staggered_projectors_at(Qf, layout)
    C_new, T_new = _c4v_staggered_project(Qf, T, a, P_left, P_right)
    return _normalize_by_max_abs(C_new), _normalize_by_max_abs(T_new)


def _aligned_sweep(A, C, T, layout: BondLayout):
    """F~(e; A, ref=T) = align(F_L(e, a(A)), T)."""
    C_new, T_new = _sweep_at(A, C, T, layout)
    return _align(C_new, T_new, T)


def _rel_change(X_new, X_old) -> jax.Array:
    """max |X_new - X_old| / max |X_old| over the block data (one layout)."""
    return jnp.max(jnp.abs(X_new._data - X_old._data)) / jnp.max(jnp.abs(X_old._data))


def _step_impl(A, C, T, layout: BondLayout):
    """One element-wise step: ``(C1, T1, residual)``."""
    C1, T1 = _aligned_sweep(A, C, T, layout)
    return C1, T1, jnp.maximum(_rel_change(C1, C), _rel_change(T1, T))


_step = jax.jit(_step_impl, static_argnames=("layout",))
