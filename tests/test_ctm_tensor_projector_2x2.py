"""Tests for the 2x2 plaquette CTM projector (multisite path)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _dense_tensor_5leg(D: int, d: int, seed: int = 0) -> DenseTensor:
    """Random rank-5 iPEPS site tensor with labels (u, d, l, r, phys).

    Uses the trivial U(1) charge sector and the (OUT, IN, OUT, IN, IN) flow
    convention used elsewhere in the test suite (see ``test_coarse_grain``).
    """
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal(
        (D, D, D, D, d)
    )
    arr = arr / np.linalg.norm(arr)

    sym = U1Symmetry()
    charges_D = np.zeros(D, dtype=np.int32)
    charges_phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges_D.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, charges_phys.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(jnp.asarray(arr), indices)


def test_build_enlarged_corner_top_left_shape():
    """Q_TL = C1 . T1 . T4 . a should be a rank-4 tensor with seam labels
    (chi_R, r2, chi_B, d2). Open legs are the right + bottom seams that
    connect Q_TL to Q_TR (right) and Q_BL (bottom) in the 2x2 plaquette.
    """
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)  # (u2, d2, l2, r2)
    env = initialize_ctm_tensor_env(A, chi)  # identity-init env

    Q_TL = _build_enlarged_corner(env.C1, env.T1, env.T4, a, position="top_left")

    assert Q_TL.ndim == 4
    shapes = {ind.label: ind.dim for ind in Q_TL.indices}
    assert shapes == {"chi_R": chi, "r2": D * D, "chi_B": chi, "d2": D * D}, (
        f"unexpected Q_TL shape: {shapes}"
    )


def test_build_enlarged_corner_top_left_numerical():
    """Numerical check: ``_build_enlarged_corner`` for the top-left quarter
    must match a direct ``np.einsum`` reference contraction.

    The shape-only test above would still pass if any of the following bugs
    were present: relabel direction swapped (e.g. c1_r↔c1_d), seam labels
    chi_R↔chi_B swapped, missing flow flip, ket/bra fuse axis swap, or
    u2/l2 absorbed by the wrong edge. This test catches all of them.

    Strategy:
      * build a random PEPS site tensor A and the double-layer ``a``,
      * replace the identity-initialised env C1/T1/T4 with random data
        (keeping the same TensorIndex / flow conventions) so the contraction
        is non-trivial,
      * run ``_build_enlarged_corner`` and ``np.einsum`` on the SAME raw
        arrays and compare after permuting axes to a common ordering.

    Index labels (per ``CTMTensorEnv`` docstring + ``_STD_EDGE_SPECS``):
      C1: (c1_d, c1_r)        T1: (t1_l, u2, t1_r)
      T4: (t4_d, l2, t4_u)    a:  (u2,   d2, l2,   r2)

    The shared bonds in the contraction are:
      C1.c1_r — T1.t1_l   (chi)
      C1.c1_d — T4.t4_d   (chi)
      T1.u2   — a.u2      (D^2)
      T4.l2   — a.l2      (D^2)
    Remaining open legs: T1.t1_r, T4.t4_u, a.d2, a.r2.
    """
    D, chi, d = 2, 3, 2
    A = _dense_tensor_5leg(D, d, seed=42)
    a = _build_double_layer_tensor(A)  # (u2, d2, l2, r2)
    a_arr = np.asarray(a.todense())  # axis order: u2, d2, l2, r2

    env = initialize_ctm_tensor_env(A, chi)

    # Build random raw arrays for the env tensors. Use complex128 to match
    # the iPEPS dtype of A.
    rng = np.random.default_rng(7)

    def _randc(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    C1_arr = _randc(chi, chi).astype(np.complex128)  # (c1_d, c1_r)
    T1_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t1_l, u2, t1_r)
    T4_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t4_d, l2, t4_u)

    # Wrap as DenseTensor sharing the same TensorIndex objects as the
    # identity-initialised env. This guarantees label/flow conventions
    # exactly match _build_enlarged_corner's expectations.
    C1 = DenseTensor(jnp.asarray(C1_arr), env.C1.indices)
    T1 = DenseTensor(jnp.asarray(T1_arr), env.T1.indices)
    T4 = DenseTensor(jnp.asarray(T4_arr), env.T4.indices)

    # --- Tenax path ---
    Q = _build_enlarged_corner(C1, T1, T4, a, position="top_left")
    # Determine axis order from the returned tensor (the existing shape test
    # only checks the set of labels, not their order, so we discover the
    # order here and use it explicitly).
    q_labels = [ind.label for ind in Q.indices]
    assert set(q_labels) == {"chi_R", "r2", "chi_B", "d2"}, (
        f"unexpected Q label set: {q_labels}"
    )
    Q_dense = np.asarray(Q.todense())

    # --- Reference einsum on the same raw arrays ---
    # Q_ref[t1_r, t4_u, d2, r2] =
    #   sum_{cd, cr=t1l, u2, l2, t4d=cd}
    #     C1[cd, cr] * T1[cr, u2, t1_r] * T4[cd, l2, t4_u]
    #     * a[u2, d2, l2, r2]
    # Index legend for the einsum below:
    #   a = c1_d / t4_d (chi)             b = c1_r / t1_l (chi)
    #   c = u2 (D^2)                      d = l2 (D^2)
    #   e = t1_r (chi)                    f = t4_u (chi)
    #   g = d2 (D^2)                      h = r2 (D^2)
    Q_ref = np.einsum(
        "ab,bce,adf,cgdh->efhg",
        C1_arr,
        T1_arr,
        T4_arr,
        a_arr,
        optimize=True,
    )
    # Q_ref axes: (e=t1_r=chi_R, f=t4_u=chi_B, h=r2, g=d2)
    ref_label_to_axis = {"chi_R": 0, "chi_B": 1, "r2": 2, "d2": 3}
    perm = tuple(ref_label_to_axis[lbl] for lbl in q_labels)
    Q_ref = np.transpose(Q_ref, perm)

    assert Q_dense.shape == Q_ref.shape, (
        f"shape mismatch: tenax {Q_dense.shape}, ref {Q_ref.shape}"
    )
    assert np.allclose(Q_dense, Q_ref, atol=1e-10, rtol=1e-10), (
        f"Q_TL mismatch: max abs diff = {np.max(np.abs(Q_dense - Q_ref))}"
    )


@pytest.mark.parametrize("position", ["top_right", "bottom_left", "bottom_right"])
def test_build_enlarged_corner_other_positions_shape(position):
    """Q_TR / Q_BL / Q_BR all rank-4 with appropriate seam labels."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    if position == "top_right":
        Q = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position=position)
        expected = {"chi_L": chi, "l2": D * D, "chi_B": chi, "d2": D * D}
    elif position == "bottom_left":
        Q = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position=position)
        expected = {"chi_R": chi, "r2": D * D, "chi_T": chi, "u2": D * D}
    elif position == "bottom_right":
        Q = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position=position)
        expected = {"chi_L": chi, "l2": D * D, "chi_T": chi, "u2": D * D}

    assert Q.ndim == 4
    shapes = {ind.label: ind.dim for ind in Q.indices}
    assert shapes == expected, f"unexpected Q[{position}] shape: {shapes}"


def test_build_enlarged_corner_top_right_numerical():
    """Numerical cross-check for ``position="top_right"``.

    Index labels:
      C2: (c2_l, c2_d)         T1: (t1_l, u2, t1_r)
      T2: (t2_u, r2, t2_d)     a:  (u2,   d2, l2,   r2)

    Shared bonds in the contraction:
      C2.c2_l — T1.t1_r   (chi)
      C2.c2_d — T2.t2_u   (chi)
      T1.u2   — a.u2      (D^2)
      T2.r2   — a.r2      (D^2)
    Remaining open legs: T1.t1_l (chi_L), T2.t2_d (chi_B), a.l2, a.d2.
    """
    D, chi, d = 2, 3, 2
    A = _dense_tensor_5leg(D, d, seed=42)
    a = _build_double_layer_tensor(A)  # (u2, d2, l2, r2)
    a_arr = np.asarray(a.todense())

    env = initialize_ctm_tensor_env(A, chi)

    rng = np.random.default_rng(11)

    def _randc(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    C2_arr = _randc(chi, chi).astype(np.complex128)  # (c2_l, c2_d)
    T1_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t1_l, u2, t1_r)
    T2_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t2_u, r2, t2_d)

    C2 = DenseTensor(jnp.asarray(C2_arr), env.C2.indices)
    T1 = DenseTensor(jnp.asarray(T1_arr), env.T1.indices)
    T2 = DenseTensor(jnp.asarray(T2_arr), env.T2.indices)

    Q = _build_enlarged_corner(C2, T1, T2, a, position="top_right")
    q_labels = [ind.label for ind in Q.indices]
    assert set(q_labels) == {"chi_L", "l2", "chi_B", "d2"}, (
        f"unexpected Q label set: {q_labels}"
    )
    Q_dense = np.asarray(Q.todense())

    # Index legend:
    #   a = c2_l / t1_r (chi)             b = c2_d / t2_u (chi)
    #   c = u2 (D^2)                      e = r2 (D^2)
    #   f = t1_l (chi_L)                  g = t2_d (chi_B)
    #   h = l2 (D^2)                      k = d2 (D^2)
    Q_ref = np.einsum(
        "ab,fca,beg,ckhe->fghk",
        C2_arr,
        T1_arr,
        T2_arr,
        a_arr,
        optimize=True,
    )
    # Q_ref axes: (f=chi_L, g=chi_B, h=l2, k=d2)
    ref_label_to_axis = {"chi_L": 0, "chi_B": 1, "l2": 2, "d2": 3}
    perm = tuple(ref_label_to_axis[lbl] for lbl in q_labels)
    Q_ref = np.transpose(Q_ref, perm)

    assert Q_dense.shape == Q_ref.shape, (
        f"shape mismatch: tenax {Q_dense.shape}, ref {Q_ref.shape}"
    )
    assert np.allclose(Q_dense, Q_ref, atol=1e-10, rtol=1e-10), (
        f"Q_TR mismatch: max abs diff = {np.max(np.abs(Q_dense - Q_ref))}"
    )


def test_build_enlarged_corner_bottom_left_numerical():
    """Numerical cross-check for ``position="bottom_left"``.

    Index labels:
      C4: (c4_r, c4_u)         T3: (t3_r, d2, t3_l)
      T4: (t4_d, l2, t4_u)     a:  (u2,   d2, l2,   r2)

    Shared bonds in the contraction:
      C4.c4_u — T4.t4_u   (chi)
      C4.c4_r — T3.t3_r   (chi)
      T3.d2   — a.d2      (D^2)
      T4.l2   — a.l2      (D^2)
    Remaining open legs: T4.t4_d (chi_T), T3.t3_l (chi_R), a.u2, a.r2.
    """
    D, chi, d = 2, 3, 2
    A = _dense_tensor_5leg(D, d, seed=42)
    a = _build_double_layer_tensor(A)
    a_arr = np.asarray(a.todense())

    env = initialize_ctm_tensor_env(A, chi)

    rng = np.random.default_rng(13)

    def _randc(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    C4_arr = _randc(chi, chi).astype(np.complex128)  # (c4_r, c4_u)
    T3_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t3_r, d2, t3_l)
    T4_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t4_d, l2, t4_u)

    C4 = DenseTensor(jnp.asarray(C4_arr), env.C4.indices)
    T3 = DenseTensor(jnp.asarray(T3_arr), env.T3.indices)
    T4 = DenseTensor(jnp.asarray(T4_arr), env.T4.indices)

    Q = _build_enlarged_corner(C4, T3, T4, a, position="bottom_left")
    q_labels = [ind.label for ind in Q.indices]
    assert set(q_labels) == {"chi_R", "r2", "chi_T", "u2"}, (
        f"unexpected Q label set: {q_labels}"
    )
    Q_dense = np.asarray(Q.todense())

    # Index legend:
    #   a = c4_r / t3_r (chi)             b = c4_u / t4_u (chi)
    #   c = d2 (D^2)                      e = l2 (D^2)
    #   f = t4_d (chi_T)                  g = t3_l (chi_R)
    #   h = u2 (D^2)                      k = r2 (D^2)
    Q_ref = np.einsum(
        "ab,acg,feb,hcek->fghk",
        C4_arr,
        T3_arr,
        T4_arr,
        a_arr,
        optimize=True,
    )
    # Q_ref axes: (f=chi_T, g=chi_R, h=u2, k=r2)
    ref_label_to_axis = {"chi_T": 0, "chi_R": 1, "u2": 2, "r2": 3}
    perm = tuple(ref_label_to_axis[lbl] for lbl in q_labels)
    Q_ref = np.transpose(Q_ref, perm)

    assert Q_dense.shape == Q_ref.shape, (
        f"shape mismatch: tenax {Q_dense.shape}, ref {Q_ref.shape}"
    )
    assert np.allclose(Q_dense, Q_ref, atol=1e-10, rtol=1e-10), (
        f"Q_BL mismatch: max abs diff = {np.max(np.abs(Q_dense - Q_ref))}"
    )


def test_compute_2x2_projector_left_shape_and_isometry():
    """For a converged-style env, _compute_2x2_projector returns a (P_top, P_bot)
    pair satisfying P_top^dagger . P_bot ~ I on the chi_new = chi truncated space."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    Q_TL = _build_enlarged_corner(env.C1, env.T1, env.T4, a, position="top_left")
    Q_TR = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position="top_right")
    Q_BL = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position="bottom_left")
    Q_BR = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position="bottom_right")

    from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector

    P_top, P_bot = _compute_2x2_projector(Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="left")

    # Both projectors should be rank-3 with shared labels chi_outer + fused_D2
    # and distinct chi_new labels (chi_new_top, chi_new_bot).
    assert P_top.ndim == 3
    assert P_bot.ndim == 3

    # Closure: contract P_top with P_bot. Shared labels auto-pair; free legs
    # are (chi_new_top, chi_new_bot). Result should be ~ identity.
    from tenax.contraction.contractor import contract

    closure = contract(P_top, P_bot)
    assert closure.ndim == 2
    closure_dense = closure.todense()
    eye = jnp.eye(closure_dense.shape[0], dtype=closure_dense.dtype)
    err = float(jnp.linalg.norm(closure_dense - eye))
    assert err < 1e-6, f"P_top . P_bot != I; Frobenius err = {err:.2e}"


def test_build_enlarged_corner_bottom_right_numerical():
    """Numerical cross-check for ``position="bottom_right"``.

    Index labels:
      C3: (c3_u, c3_l)         T3: (t3_r, d2, t3_l)
      T2: (t2_u, r2, t2_d)     a:  (u2,   d2, l2,   r2)

    Shared bonds in the contraction:
      C3.c3_l — T3.t3_l   (chi)
      C3.c3_u — T2.t2_d   (chi)
      T3.d2   — a.d2      (D^2)
      T2.r2   — a.r2      (D^2)
    Remaining open legs: T3.t3_r (chi_L), T2.t2_u (chi_T), a.l2, a.u2.
    """
    D, chi, d = 2, 3, 2
    A = _dense_tensor_5leg(D, d, seed=42)
    a = _build_double_layer_tensor(A)
    a_arr = np.asarray(a.todense())

    env = initialize_ctm_tensor_env(A, chi)

    rng = np.random.default_rng(17)

    def _randc(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    C3_arr = _randc(chi, chi).astype(np.complex128)  # (c3_u, c3_l)
    T3_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t3_r, d2, t3_l)
    T2_arr = _randc(chi, D * D, chi).astype(np.complex128)  # (t2_u, r2, t2_d)

    C3 = DenseTensor(jnp.asarray(C3_arr), env.C3.indices)
    T3 = DenseTensor(jnp.asarray(T3_arr), env.T3.indices)
    T2 = DenseTensor(jnp.asarray(T2_arr), env.T2.indices)

    Q = _build_enlarged_corner(C3, T3, T2, a, position="bottom_right")
    q_labels = [ind.label for ind in Q.indices]
    assert set(q_labels) == {"chi_L", "l2", "chi_T", "u2"}, (
        f"unexpected Q label set: {q_labels}"
    )
    Q_dense = np.asarray(Q.todense())

    # Index legend:
    #   a = c3_u / t2_d (chi)             b = c3_l / t3_l (chi)
    #   c = d2 (D^2)                      e = r2 (D^2)
    #   f = t3_r (chi_L)                  g = t2_u (chi_T)
    #   h = l2 (D^2)                      k = u2 (D^2)
    Q_ref = np.einsum(
        "ab,fcb,gea,kche->fghk",
        C3_arr,
        T3_arr,
        T2_arr,
        a_arr,
        optimize=True,
    )
    # Q_ref axes: (f=chi_L, g=chi_T, h=l2, k=u2)
    ref_label_to_axis = {"chi_L": 0, "chi_T": 1, "l2": 2, "u2": 3}
    perm = tuple(ref_label_to_axis[lbl] for lbl in q_labels)
    Q_ref = np.transpose(Q_ref, perm)

    assert Q_dense.shape == Q_ref.shape, (
        f"shape mismatch: tenax {Q_dense.shape}, ref {Q_ref.shape}"
    )
    assert np.allclose(Q_dense, Q_ref, atol=1e-10, rtol=1e-10), (
        f"Q_BR mismatch: max abs diff = {np.max(np.abs(Q_dense - Q_ref))}"
    )
