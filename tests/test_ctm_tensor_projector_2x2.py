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
