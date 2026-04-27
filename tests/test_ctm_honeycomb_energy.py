"""Tests for the honeycomb 2-vertex bond RDM.

The basic sanity tests (hermiticity / positivity / trace 1) won't catch
contraction-topology bugs by themselves — they pass for any symmetric,
positive, normalized matrix. The D=1 product-state exact comparison is
the strong gate: it verifies the RDM equals the literal tensor-product
of single-site density matrices, which is the only thing it CAN equal
when all virtual bonds are trivial.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_energy import _rdm2_bond
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


# ------------------------------------------------------------------ #
# Sanity tests                                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_bond_rdm_shape_hermitian_trace1(alpha: int):
    """Shape, hermiticity, trace 1 with a random (unphysical) env.

    NOTE: positivity is NOT checked here. A random env is not a valid
    converged CTM environment — the resulting "RDM" has no physical
    interpretation, so its eigenvalues can be negative. Only the
    explicit symmetrize + trace-normalize steps in ``_rdm2_bond`` are
    being exercised. The strong physical gate is the D=1 product-state
    test below.
    """
    D, d, chi = 2, 2, 4
    A = _make_random_honeycomb_site(D=D, d=d, key=jax.random.PRNGKey(100 + alpha))
    B = _make_random_honeycomb_site(D=D, d=d, key=jax.random.PRNGKey(200 + alpha))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    rho = _rdm2_bond(sites, envs, alpha=alpha)
    assert rho.shape == (d * d, d * d)

    herm_err = float(jnp.max(jnp.abs(rho - rho.conj().T)))
    assert herm_err < 1e-10, f"hermiticity err = {herm_err}"

    trace_err = float(jnp.abs(jnp.trace(rho) - 1.0))
    assert trace_err < 1e-10, f"trace = {jnp.trace(rho)}"


# ------------------------------------------------------------------ #
# Exact-comparison gate: D=1 product state                             #
# ------------------------------------------------------------------ #


def _make_d1_site(d: int, key: jax.Array) -> DenseTensor:
    """Rank-4 site with D=1 virtual legs — a pure product state on phys."""
    sym = U1Symmetry()
    virt = np.zeros(1, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (1, 1, 1, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (1, 1, 1, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_bond_rdm_d1_equals_tensor_product_of_1site(alpha: int):
    """D=1 → product state → bond RDM = ρ_A ⊗ ρ_B exactly.

    With all virtual legs of dim 1, the iPEPS state factorizes:
    ``|ψ⟩ = |ψ_A⟩ ⊗ |ψ_B⟩ ⊗ ...`` so ρ_AB = ρ_A ⊗ ρ_B. Any wrong
    contraction topology (extra/missing tensors, wrong leg pairings,
    swapped sublattice roles) will fail this comparison — it cannot be
    "almost" right.
    """
    d = 2
    chi = 1
    A = _make_d1_site(d=d, key=jax.random.PRNGKey(7))
    B = _make_d1_site(d=d, key=jax.random.PRNGKey(8))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    rho = _rdm2_bond(sites, envs, alpha=alpha)

    # Expected: build ρ_A and ρ_B from the raw site vectors (D=1 collapses
    # to a d-vector after contracting the trivial virtual legs).
    a_vec = A.todense().reshape(d)
    b_vec = B.todense().reshape(d)
    rho_A = jnp.outer(a_vec, jnp.conj(a_vec))
    rho_A = rho_A / (jnp.trace(rho_A) + 1e-30)
    rho_B = jnp.outer(b_vec, jnp.conj(b_vec))
    rho_B = rho_B / (jnp.trace(rho_B) + 1e-30)
    expected = jnp.kron(rho_A, rho_B)

    err = float(jnp.max(jnp.abs(rho - expected)))
    assert err < 1e-10, (
        f"alpha={alpha}: bond RDM differs from ρ_A ⊗ ρ_B by max |Δ| = {err:.3e}"
    )
