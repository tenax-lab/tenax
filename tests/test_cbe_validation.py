"""Validation of CBE's ``expand_bond``: the added directions are orthogonal.

(A slow "1-site+CBE vs 2-site DMRG" comparison used to live here; it never
called CBE -- CBE is not wired into the DMRG sweep -- and was deleted in the
slow-bucket prune.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.cbe import expand_bond
from tenax.algorithms.dmrg import (
    _build_right_environments_list,
    _build_trivial_left_env,
    _build_trivial_right_env,
    _effective_hamiltonian_matvec,
    _lanczos_solve,
    _one_site_update_symmetric,
    _update_left_env_symmetric,
    _update_right_env_symmetric,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


class TestCBEValidation:
    def test_expand_bond_produces_orthogonal_directions(self):
        """CBE expansion adds columns orthogonal to the original site tensor.

        Uses a deterministic left-canonical site (via QR) and a non-trivial
        diagonal matvec to ensure the sketch has components outside the
        original column space.
        """
        chi_l, d, chi_r = 4, 2, 3
        d2, chi_rr = 2, 3
        k = 2

        # Build a left-canonical site tensor: A_mat has orthonormal columns
        key0, key1 = jax.random.split(jax.random.PRNGKey(0))
        A_mat, _ = jnp.linalg.qr(jax.random.normal(key0, (chi_l * d, chi_r)))
        site_3d = A_mat.reshape(chi_l, d, chi_r)
        right_3d = jax.random.normal(key1, (chi_r, d2, chi_rr))

        # Non-trivial matvec: diagonal H with distinct eigenvalues.
        # Identity matvec is broken for CBE because the sketch Y = A @ M lies
        # entirely in A's column space, making Y_perp = 0 and QR degenerate.
        def diagonal_matvec(v, shape):
            n = v.shape[0]
            return v * jnp.arange(1, n + 1, dtype=v.dtype)

        sym = U1Symmetry()

        def _wrap(data, prefix):
            s = data.shape
            return DenseTensor(
                data,
                (
                    TensorIndex.from_charges(
                        sym, np.zeros(s[0], np.int32), FlowDirection.IN, f"{prefix}l"
                    ),
                    TensorIndex.from_charges(
                        sym, np.zeros(s[1], np.int32), FlowDirection.IN, "p"
                    ),
                    TensorIndex.from_charges(
                        sym, np.zeros(s[2], np.int32), FlowDirection.OUT, f"{prefix}r"
                    ),
                ),
            )

        site_t = _wrap(site_3d, "v")
        right_t = _wrap(right_3d, "w")

        expanded = expand_bond(
            site_t, diagonal_matvec, right_t, k=k, key=jax.random.PRNGKey(42)
        )
        exp = expanded.todense()

        assert exp.shape == (chi_l, d, chi_r + k)

        # Verify orthogonality: new columns should be perpendicular to original
        new_cols = exp.reshape(chi_l * d, chi_r + k)[:, chi_r:]
        overlap = jnp.conj(A_mat).T @ new_cols
        assert float(jnp.max(jnp.abs(overlap))) < 1e-10, (
            f"New columns should be orthogonal to original, max overlap={float(jnp.max(jnp.abs(overlap)))}"
        )
