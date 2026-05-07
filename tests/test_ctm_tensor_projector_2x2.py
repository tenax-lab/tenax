"""Tests for the 2x2 plaquette CTM projector (multisite path)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
