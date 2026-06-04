"""Frozen accuracy spine for the stacked-block migration (#566, P0)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

FP_RTOL = 1e-12
FP_ATOL = 1e-12


def assert_tiered(ref, got, *, tier: str) -> None:
    ref = jnp.asarray(ref)
    got = jnp.asarray(got)
    assert ref.shape == got.shape, f"shape {ref.shape} != {got.shape}"
    if tier == "bit":
        assert bool(jnp.array_equal(ref, got)), (
            f"not bit-identical: max|d|={float(jnp.max(jnp.abs(ref - got))):.3e}"
        )
    elif tier == "fp":
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref), rtol=FP_RTOL, atol=FP_ATOL
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")


def assert_svd_invariants(A_ref, U, S, Vh, *, tier: str = "fp") -> None:
    """Compare gauge-INVARIANT SVD outputs only: singular values + reconstruction."""
    recon = (U * S[..., None, :]) @ Vh
    assert_tiered(A_ref, recon, tier=tier)


def canonical_tensors():
    """Yield (name, tensor-or-array) covering the required golden cases."""
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _build_initial_fpeps_tensor,
    )

    key = jax.random.PRNGKey(0)
    yield "ferm_D2", _build_initial_fpeps_tensor(FPEPSConfig(D=2), key)
    yield "ferm_D4", _build_initial_fpeps_tensor(FPEPSConfig(D=4), key)
    yield "dense_D2", _dense_u1_trivial(D=2, key=key)
    yield "degenerate_sv", jnp.diag(jnp.array([2.0, 2.0, 1.0, 0.5]))
    v = jnp.array([1.0, 2.0, 3.0, 4.0])
    yield "rank_deficient", jnp.outer(v, v)


def _dense_u1_trivial(D, key):
    """U(1)-trivial single-block DenseTensor iPEPS site.

    Replicates ``make_dense_site`` from ``examples/profile_ctm_ad_wall_566.py``
    inline so the harness does not depend on the (non-package) ``examples/``
    directory being importable.
    """
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor, FlowDirection, TensorIndex

    data = jax.random.normal(key, (D, D, D, D, 2))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(2, dtype=np.int32)
    return DenseTensor(
        data,
        (
            TensorIndex.from_charges(sym, zD, FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, zD, FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, zD, FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, zD, FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, zd, FlowDirection.IN, label="phys"),
        ),
    )
