"""Tests for metric preconditioning (norm environment) for iPEPS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._metric_precond import (
    _contract_single_site_environment,
    norm_environment_matvec,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_random_ipeps_tensor(D=2, d=2, seed=0):
    """Create a random DenseTensor iPEPS site tensor."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(data, indices)


def _converge_ctm(A, chi=4, max_iter=50):
    """Run CTM to convergence and return the single-site environment."""
    return ctm_tensor(A, chi=chi, max_iter=max_iter, conv_tol=1e-10)


# ------------------------------------------------------------------ #
# Tests: _contract_single_site_environment                             #
# ------------------------------------------------------------------ #


class TestContractSingleSiteEnv:
    """Tests for the single-site environment contraction."""

    def test_output_shape(self):
        """E has shape (D^2, D^2, D^2, D^2)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)
        assert E.shape == (D**2, D**2, D**2, D**2)

    def test_reproduces_norm(self):
        """E contracted with double-layer tensor gives a positive norm."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)
        E = _contract_single_site_environment(env)

        # Build the double-layer tensor (u2, d2, l2, r2) and contract with E
        from tenax.contraction.contractor import contract as tensor_contract

        a2 = _build_double_layer_tensor(A)
        a2_dense = a2.todense()  # (D^2, D^2, D^2, D^2)

        # Norm = E_{u2,d2,l2,r2} * a2_{u2,d2,l2,r2}
        norm = jnp.einsum("ijkl,ijkl->", E, a2_dense)
        assert norm.real > 0, f"Expected positive norm, got {norm}"


# ------------------------------------------------------------------ #
# Tests: norm_environment_matvec                                       #
# ------------------------------------------------------------------ #


class TestNormEnvironmentMV:
    """Tests for the norm-environment matvec."""

    def test_output_shape(self):
        """N . v has shape (D, D, D, D, d)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, (D, D, D, D, d))
        Nv = norm_environment_matvec(A, env, v)
        assert Nv.shape == (D, D, D, D, d)

    def test_positive_definite(self):
        """<v|N|v> > 0 for random v."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, (D, D, D, D, d))
        Nv = norm_environment_matvec(A, env, v)

        vNv = jnp.sum(v * Nv)
        assert vNv.real > 0, f"Expected <v|N|v> > 0, got {vNv}"

    def test_hermitian(self):
        """<u|N|v> = conj(<v|N|u>)."""
        D, d = 2, 2
        A = _make_random_ipeps_tensor(D=D, d=d)
        env = _converge_ctm(A)

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(99)
        u = jax.random.normal(key1, (D, D, D, D, d))
        v = jax.random.normal(key2, (D, D, D, D, d))

        Nu = norm_environment_matvec(A, env, u)
        Nv = norm_environment_matvec(A, env, v)

        uNv = jnp.sum(u * Nv)
        vNu = jnp.sum(v * Nu)
        np.testing.assert_allclose(
            uNv,
            jnp.conj(vNu),
            rtol=1e-5,
            atol=1e-10,
            err_msg="Norm environment is not Hermitian",
        )
