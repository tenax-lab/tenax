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

import math
from collections import Counter
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tenax.algorithms._ctm_convergence_policy import (
    CTMConvergeInfo,
    CTMNotConvergedError,
)
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_loop,
    _c4v_staggered_project,
    _c4v_staggered_projectors_at,
    _leg,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ctm_tensor_moves import _normalize_by_max_abs
from tenax.algorithms._gmres_eager import gmres_eager
from tenax.algorithms.ad_utils import (
    _bond_blocks,
    _bond_component_phases,
    _bond_rebuild,
    _bond_ref_blocks,
)

__all__ = [
    "_KRYLOV_DIM",
    "_adjoint_matvec",
    "_align",
    "_aligned_sweep",
    "_chi_components",
    "_elementwise_phase",
    "_layout_of",
    "_rel_change",
    "_scale_chi",
    "_site_vjp",
    "_step",
    "_step_impl",
    "_sweep_at",
    "charged_c4v_cold_start",
    "charged_c4v_converge",
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


_KRYLOV_DIM = 30  # GMRES restart length, the C4v root engine's default


def charged_c4v_cold_start(A, ctm):
    """The #1211 loop to sv-convergence: ``(C, T, layout)``.

    The layout of the last sweep is the frozen layout L of an AD run.

    Raises:
        CTMNotConvergedError: no ``sv_diff < ctm.conv_tol`` in ``ctm.max_iter``
            sweeps.
    """
    C, T, _a, info = _c4v_staggered_loop(A, ctm.chi, ctm.max_iter, ctm.conv_tol)
    if not info.converged:
        raise CTMNotConvergedError(
            info,
            site="gradient",
            conv_tol=ctm.conv_tol,
            chi=ctm.chi,
            detail="charged C4v AD: the cold start (the #1211 sweep) did not converge",
        )
    return C, T, _layout_of(C)


def _elementwise_phase(A, warm, layout, ctm, record):
    """Iterate e <- F~(e; A, ref=e) from ``warm`` to an element-wise fixed point.

    Stops when max(|dC| / max|C|, |dT| / max|T|) < ``ctm.conv_tol``, and
    writes ``forward_sweeps``, ``forward_residual`` and ``forward_rate``
    (the last residual ratio, an estimate of rho(J)) into ``record``.

    Raises:
        CTMNotConvergedError: no fixed point in ``ctm.max_iter`` sweeps, or a
            non-finite residual.
    """
    C, T = warm
    res, prev, sweep = float("inf"), float("inf"), 0
    for sweep in range(1, ctm.max_iter + 1):
        C, T, r = _step(A, C, T, layout)
        prev, res = res, float(r)
        if not math.isfinite(res):
            break
        if res < ctm.conv_tol:
            rate = res / prev if math.isfinite(prev) and prev > 0 else float("nan")
            record.update(forward_sweeps=sweep, forward_residual=res, forward_rate=rate)
            return C, T
    info = CTMConvergeInfo(
        converged=False,
        iterations=sweep,
        sv_diff=res,
        best_iteration=sweep,
        step_multiplier=float("nan"),
    )
    raise CTMNotConvergedError(
        info,
        site="gradient",
        conv_tol=ctm.conv_tol,
        chi=ctm.chi,
        detail=(
            "charged C4v AD: no element-wise fixed point at the frozen layout "
            f"{layout.dims} (residual max(|dC|/max|C|, |dT|/max|T|) = {res:.3e})"
        ),
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def charged_c4v_converge(A, warm, layout, ctm, record):
    """The element-wise fixed point e* = F~(e*; A, ref=e*), with an implicit VJP.

    Args:
        A: the charged C4v site tensor.
        warm: ``(C, T)`` at ``layout`` to start from.  Its cotangent is zero.
        layout: the frozen ``BondLayout`` (``charged_c4v_cold_start``).
        ctm: the ``CTMConfig``.  The forward reads ``max_iter`` and
            ``conv_tol``; the backward reads ``adjoint_tol``,
            ``adjoint_maxiter`` and ``adjoint_tikhonov``.
        record: a dict that receives the forward and backward metadata.
    """
    return _elementwise_phase(A, warm, layout, ctm, record)


def _converge_fwd(A, warm, layout, ctm, record):
    C, T = _elementwise_phase(A, warm, layout, ctm, record)
    if jax.tree_util.tree_structure(warm) != jax.tree_util.tree_structure((C, T)):
        raise ValueError(
            "charged C4v AD: the warm env does not have the layout of the fixed point"
        )
    return (C, T), (A, C, T)


def _converge_bwd(layout, ctm, record, res, cot):
    """Solve (I - J^T + tau I) lambda = g; return (dF~/dA)^T lambda (spec 2, section 3)."""
    A, C, T = res
    b, _unravel = ravel_pytree(cot)
    tau = float(ctm.adjoint_tikhonov or 0.0)
    matvecs = [0]

    def matvec(v):
        matvecs[0] += 1
        return _adjoint_matvec(v, A, C, T, layout, tau)

    lam, resid = gmres_eager(
        matvec, b, tol=ctm.adjoint_tol, maxiter=ctm.adjoint_maxiter, restart=_KRYLOV_DIM
    )
    gate = max(10.0 * ctm.adjoint_tol, 1e-12)
    record.update(adjoint_matvecs=matvecs[0], adjoint_residual=float(resid))
    if not (math.isfinite(resid) and resid <= gate):
        raise RuntimeError(
            "charged C4v AD: the adjoint solve stopped at relative residual "
            f"{resid:.3e} > {gate:.1e} after {matvecs[0]} matvecs "
            f"(adjoint_tol={ctm.adjoint_tol}, adjoint_maxiter={ctm.adjoint_maxiter})"
        )
    if not bool(jnp.all(jnp.isfinite(lam))):
        raise RuntimeError("charged C4v AD: the adjoint solution is not finite")
    dA = _site_vjp(lam, A, C, T, layout)
    if not bool(jnp.all(jnp.isfinite(dA._data))):
        raise RuntimeError("charged C4v AD: the site cotangent dA is not finite")
    return dA, jax.tree_util.tree_map(jnp.zeros_like, (C, T))


charged_c4v_converge.defvjp(_converge_fwd, _converge_bwd)


def _adjoint_matvec_impl(v, A, C, T, layout, tau):
    """v - J^T v + tau v, J the Jacobian of e -> F~(e; A, ref=T) at e = (C, T)."""
    _flat, unravel = ravel_pytree((C, T))
    _out, vjp = jax.vjp(lambda e: _align(*_sweep_at(A, *e, layout), T), (C, T))
    (jt_v,) = vjp(unravel(v))
    return v - ravel_pytree(jt_v)[0] + tau * v


_adjoint_matvec = jax.jit(_adjoint_matvec_impl, static_argnames=("layout",))


def _site_vjp_impl(lam, A, C, T, layout):
    """(dF~/dA)^T lambda at the fixed point (C, T)."""
    _flat, unravel = ravel_pytree((C, T))
    _out, vjp = jax.vjp(lambda A_: _align(*_sweep_at(A_, C, T, layout), T), A)
    (dA,) = vjp(unravel(lam))
    return dA


_site_vjp = jax.jit(_site_vjp_impl, static_argnames=("layout",))
