"""Padding invariance for chi-schedule unification (issue #453).

If we converge CTM at logical chi=chi_small and pad envs to chi=chi_big,
evaluating energy and gradient on the same A must yield the same
answer as the unpadded chi=chi_small evaluation. This invariant is the
load-bearing requirement of the unified scheduled-bump refactor.

This invariant is already true in production (the variPEPS Sec 2.8.2
reactive auto-chi_E bump path uses it via ``_maybe_bump_chi`` in
``ipeps_optimize.py``), so these tests should PASS on ``main``. They
are committed as the green baseline that gates the upcoming refactor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_env_pad import pad_dense_env_chi
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _heisenberg_gate() -> jnp.ndarray:
    """Two-site Heisenberg gate (d=2), shape (d, d, d, d)."""
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _make_site_tensor(data: jnp.ndarray, D: int, d: int) -> DenseTensor:
    """Wrap raw (D, D, D, D, d) data as a DenseTensor with trivial-U(1) indices.

    Mirrors the convention used by ``tests/test_chi_auto_bump.py`` and
    the AD ``_params_to_A_norm`` path in ``ipeps_optimize.py``.
    """
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


@pytest.mark.core
def test_pad_dense_env_chi_preserves_energy():
    """Pad chi=4 envs to chi=8; energy must match unpadded chi=4 evaluation."""
    d = 2
    D = 2
    chi_small = 4
    chi_big = 8

    rng = np.random.default_rng(0)
    A_data = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A_data = A_data / jnp.linalg.norm(A_data)
    A = _make_site_tensor(A_data, D, d)
    gate = _heisenberg_gate()

    # Converge CTM at chi_small.
    cfg_small = CTMConfig(chi=chi_small, max_iter=80, conv_tol=1e-10)
    envs_small, _ = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg_small, env_init=None),
    )
    env_small = envs_small[(0, 0)]

    E_small = float(compute_energy_ctm_tensor(A, env_small, gate, d))

    # Pad to chi_big.
    env_padded = pad_dense_env_chi(env_small, chi_big, base_charges=None)
    E_padded = float(compute_energy_ctm_tensor(A, env_padded, gate, d))

    assert abs(E_padded - E_small) < 1e-12, (
        f"padded energy {E_padded:.16e} != unpadded {E_small:.16e}, "
        f"diff={E_padded - E_small:.3e}"
    )


@pytest.mark.core
def test_pad_dense_env_chi_preserves_gradient():
    """Pad chi=4 envs to chi=8; energy gradient w.r.t. A must match."""
    d = 2
    D = 2
    chi_small = 4
    chi_big = 8

    rng = np.random.default_rng(1)
    A_data = jnp.asarray(
        rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    )
    A_data = A_data / jnp.linalg.norm(A_data)
    A = _make_site_tensor(A_data, D, d)
    gate = _heisenberg_gate()

    # Converge CTM at chi_small.
    cfg_small = CTMConfig(chi=chi_small, max_iter=80, conv_tol=1e-10)
    envs_small, _ = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(cfg_small, env_init=None),
    )
    env_small = envs_small[(0, 0)]
    env_padded = pad_dense_env_chi(env_small, chi_big, base_charges=None)

    # grad takes a jax.Array; build the DenseTensor inside so the closed-over
    # env is the only difference between the two evaluations.
    def E_small_fn(A_data_):
        A_ = _make_site_tensor(A_data_, D, d)
        return compute_energy_ctm_tensor(A_, env_small, gate, d).real

    def E_padded_fn(A_data_):
        A_ = _make_site_tensor(A_data_, D, d)
        return compute_energy_ctm_tensor(A_, env_padded, gate, d).real

    g_small = jax.grad(E_small_fn, holomorphic=False)(A_data)
    g_padded = jax.grad(E_padded_fn, holomorphic=False)(A_data)

    max_abs_diff = float(jnp.max(jnp.abs(g_padded - g_small)))
    assert max_abs_diff < 1e-10, (
        f"padded gradient differs from unpadded by max abs {max_abs_diff:.3e}"
    )
