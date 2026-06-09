"""Tests for the dense reduced-corner QR projector (issue #570, Phase 1).

The reduced-corner QR-CTMRG projector (Yang/Zhang/Corboz, arXiv:2505.00494)
builds an isometry directly from the enlarged corners with no large (chi D^2)
truncating SVD.  ``_reduced_qr_projector`` productionizes the spike-validated
"Candidate C" construction (concat both corners -> unpivoted QR -> tiny
2*chi x 2*chi Hermitian eigendecomposition -> top-chi directions).

Phase 1 is dense + forward only; these tests exercise the isometry property of
the returned projector on representative dense enlarged corners.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _reduced_qr_projector
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _build_dense_enlarged_corners(chi: int, D: int = 2, seed: int = 0):
    """Build a representative dense enlarged-corner pair ``(C1g, C4g)``.

    Mirrors the spike's view of the 1x1 left move: each enlarged corner is a
    dense ``(fused | cut)`` matrix where ``dim(fused) = chi * D**2`` and the
    cut leg (``t1_r`` / ``t3_l``) is already dimension ``chi`` (the reduced
    corner).  ``C1g`` carries labels ``(fused, t1_r)``; ``C4g`` carries
    ``(fused, t3_l)``.  The two corners are made close-but-not-identical so the
    both-corners (off-C4v) path is genuinely exercised.
    """
    sym = U1Symmetry()
    fused_dim = chi * D * D
    rng = np.random.default_rng(seed)

    fused_idx = TensorIndex.from_charges(
        sym, np.zeros(fused_dim, dtype=np.int32), FlowDirection.IN, label="fused"
    )
    t1_r_idx = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t1_r"
    )
    t3_l_idx = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t3_l"
    )

    base = rng.standard_normal((fused_dim, chi))
    # C4g close to C1g (small asymmetric perturbation) so both corners matter.
    pert = 0.05 * rng.standard_normal((fused_dim, chi))
    C1g = DenseTensor(jnp.asarray(base), (fused_idx, t1_r_idx))
    C4g = DenseTensor(jnp.asarray(base + pert), (fused_idx, t3_l_idx))
    return C1g, C4g


@pytest.mark.parametrize("chi", [4, 8])
def test_reduced_qr_projector_is_isometry(chi):
    """``P`` has labels ``(fused, chi_new)`` and ``P^dagger P = I_chi``."""
    C1g, C4g = _build_dense_enlarged_corners(chi)

    P = _reduced_qr_projector(C1g, C4g, chi)

    assert P.labels() == ("fused", "chi_new")
    assert P._data.shape == (chi * 2 * 2, chi)

    # Isometry: contracting P.bar() with P over the fused leg yields I_chi.
    gram = P._data.conj().T @ P._data
    np.testing.assert_allclose(
        np.asarray(gram), np.eye(chi), atol=1e-9
    )
