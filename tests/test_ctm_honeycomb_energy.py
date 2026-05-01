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

from tenax.algorithms._ctm_honeycomb_energy import (
    _rdm1,
    _rdm2_bond,
    compute_honeycomb_energy,
    compute_honeycomb_triangle_energy,
)
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


# ------------------------------------------------------------------ #
# Task 10: 1-site RDM + triangle-energy helper                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("sublattice", [(0, 0), (1, 0)])
def test_rdm1_shape_hermitian_trace1(sublattice):
    """Shape, hermiticity, trace 1 with random env (positivity not asserted)."""
    d, chi = 2, 4
    A = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(300))
    B = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(301))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    rho = _rdm1(sites, envs, sublattice=sublattice)
    assert rho.shape == (d, d)
    herm_err = float(jnp.max(jnp.abs(rho - rho.conj().T)))
    assert herm_err < 1e-10
    trace_err = float(jnp.abs(jnp.trace(rho) - 1.0))
    assert trace_err < 1e-10


@pytest.mark.parametrize("sublattice", [(0, 0), (1, 0)])
def test_rdm1_d1_equals_single_site_density(sublattice):
    """D=1 product state → ρ_s = a_s a_s^† / |a_s|² exactly."""
    d, chi = 2, 1
    A = _make_d1_site(d=d, key=jax.random.PRNGKey(50))
    B = _make_d1_site(d=d, key=jax.random.PRNGKey(51))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    rho = _rdm1(sites, envs, sublattice=sublattice)

    vec = sites[sublattice].todense().reshape(d)
    expected = jnp.outer(vec, jnp.conj(vec))
    expected = expected / (jnp.trace(expected) + 1e-30)

    err = float(jnp.max(jnp.abs(rho - expected)))
    assert err < 1e-10, f"sublattice {sublattice}: max |Δ| = {err:.3e}"


def test_triangle_energy_equals_sum_of_traces():
    """``compute_honeycomb_triangle_energy = Tr(ρ_A H) + Tr(ρ_B H)`` exactly."""
    d, chi = 2, 4
    A = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(400))
    B = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(401))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    # Random Hermitian Hamiltonian.
    H_re = jax.random.normal(jax.random.PRNGKey(999), (d, d))
    H_im = jax.random.normal(jax.random.PRNGKey(1000), (d, d))
    H = H_re + 1j * H_im
    H = 0.5 * (H + H.conj().T)

    E = compute_honeycomb_triangle_energy(sites, envs, H)

    rho_A = _rdm1(sites, envs, sublattice=(0, 0))
    rho_B = _rdm1(sites, envs, sublattice=(1, 0))
    expected = (jnp.trace(rho_A @ H) + jnp.trace(rho_B @ H)).real

    assert jnp.allclose(E, expected, atol=1e-12)


def test_triangle_energy_d1_recovers_classical_expectation():
    """D=1 → triangle_energy = ⟨a|H|a⟩/⟨a|a⟩ + ⟨b|H|b⟩/⟨b|b⟩.

    With trivial virtual bonds the iPEPS state factorizes into a product
    of single-site states; the triangle energy reduces to the sum of
    classical 1-site expectation values, which is the cheapest sanity
    check on the kagome ``energy_fn`` plumbing.
    """
    d, chi = 2, 1
    A = _make_d1_site(d=d, key=jax.random.PRNGKey(500))
    B = _make_d1_site(d=d, key=jax.random.PRNGKey(501))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    # Pauli-z as the test Hamiltonian.
    H = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    E = compute_honeycomb_triangle_energy(sites, envs, H)

    a_vec = A.todense().reshape(d)
    b_vec = B.todense().reshape(d)
    a_norm2 = jnp.sum(jnp.abs(a_vec) ** 2)
    b_norm2 = jnp.sum(jnp.abs(b_vec) ** 2)
    e_a = jnp.vdot(a_vec, H @ a_vec) / a_norm2
    e_b = jnp.vdot(b_vec, H @ b_vec) / b_norm2
    expected = (e_a + e_b).real

    assert jnp.allclose(E, expected, atol=1e-12), (
        f"E={float(E)}, expected={float(expected)}"
    )


# ------------------------------------------------------------------ #
# Task 11: default 3-edge NN bond energy                                #
# ------------------------------------------------------------------ #


def test_default_energy_equals_sum_of_three_bond_traces():
    """``compute_honeycomb_energy = Σ_α Tr(ρ_α · H_bond)`` exactly."""
    d, chi = 2, 4
    A = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(600))
    B = _make_random_honeycomb_site(D=2, d=d, key=jax.random.PRNGKey(601))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    # Random Hermitian (d², d²) bond Hamiltonian.
    H_re = jax.random.normal(jax.random.PRNGKey(700), (d * d, d * d))
    H_im = jax.random.normal(jax.random.PRNGKey(701), (d * d, d * d))
    H = H_re + 1j * H_im
    H = 0.5 * (H + H.conj().T)

    E = compute_honeycomb_energy(sites, envs, H)

    expected = 0.0 + 0.0j
    for alpha in (0, 1, 2):
        rho = _rdm2_bond(sites, envs, alpha=alpha)
        expected = expected + jnp.trace(rho @ H)
    expected = expected.real

    assert jnp.allclose(E, expected, atol=1e-12), (
        f"E={float(E)}, expected={float(expected)}"
    )


def test_default_energy_d1_recovers_per_bond_classical_expectation():
    """D=1 product state → Σ_α ⟨ψ_A ψ_B|H|ψ_A ψ_B⟩ (same on every α)."""
    d, chi = 2, 1
    A = _make_d1_site(d=d, key=jax.random.PRNGKey(800))
    B = _make_d1_site(d=d, key=jax.random.PRNGKey(801))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)

    # XX coupling: H = sx ⊗ sx for spin-1/2.
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    H = jnp.kron(sx, sx)

    E = compute_honeycomb_energy(sites, envs, H)

    a_vec = A.todense().reshape(d)
    b_vec = B.todense().reshape(d)
    a_norm2 = jnp.sum(jnp.abs(a_vec) ** 2)
    b_norm2 = jnp.sum(jnp.abs(b_vec) ** 2)
    psi = jnp.kron(a_vec, b_vec) / jnp.sqrt(a_norm2 * b_norm2)
    e_per_bond = jnp.vdot(psi, H @ psi)
    expected = (3.0 * e_per_bond).real

    assert jnp.allclose(E, expected, atol=1e-12), (
        f"E={float(E)}, expected={float(expected)}"
    )
