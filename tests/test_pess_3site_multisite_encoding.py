"""Sanity tests for the 3-site multisite kagome encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_3site_multisite,
)


@pytest.mark.core
@pytest.mark.parametrize("D, d", [(2, 2), (3, 2), (2, 3)])
def test_pess_to_kagome_3site_multisite_returns_3_site_dict(D, d):
    """Encoding produces 3 sites with the documented shapes/dtypes and
    finite, non-zero entries.

    Structural equivalence to :func:`pess_to_kagome_supersite` is validated
    by Task B.3 (D=2 energy parity), not at this task.
    """
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    assert set(sites.keys()) == {"u", "v", "w"}
    for name in ("u", "v", "w"):
        A = sites[name]
        assert A.shape == (D, D, D, D, d), f"{name}: got {A.shape}"
        assert A.dtype == jnp.complex128
        assert jnp.all(jnp.isfinite(A)), f"{name} has NaN/Inf"
        assert float(jnp.linalg.norm(A)) > 0.0, f"{name} has zero norm"
