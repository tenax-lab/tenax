"""Honeycomb CTM convergence-check tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_convergence import check_honeycomb_convergence
from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


@pytest.mark.parametrize("method", ["elementwise", "svd"])
def test_identical_envs_are_converged(method: str):
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    assert check_honeycomb_convergence(envs, envs, method=method, tol=1e-10)


@pytest.mark.parametrize("method", ["elementwise", "svd"])
def test_different_envs_not_converged(method: str):
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs1 = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    envs2 = initialize_honeycomb_env(sites, chi_init=4, seed=43)
    assert not check_honeycomb_convergence(envs1, envs2, method=method, tol=1e-10)


def test_unknown_method_raises():
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    sites = {(0, 0): A, (1, 0): A}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    with pytest.raises(ValueError, match="Unknown convergence method"):
        check_honeycomb_convergence(envs, envs, method="bogus", tol=1e-10)
