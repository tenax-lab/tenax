"""Implicit-AD gradient tests for honeycomb CTM.

The strong gate is FD-vs-AD agreement on a D=1 product state, where the CTM
fixed point is trivial and the energy gradient is exact. The grad-finite
smoke test at D=2 confirms the GMRES backward survives random-input
non-convergence (PR #343 contract: warn-not-raise).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_ad import honeycomb_ctm_energy_implicit
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _heisenberg_bond_xxz(d: int = 2, delta: float = 1.0) -> jnp.ndarray:
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H = np.kron(sx, sx) + np.kron(sy, sy) + delta * np.kron(sz, sz)
    return jnp.asarray(H, dtype=jnp.complex128)


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


def _make_d1_site(d: int, key: jax.Array) -> DenseTensor:
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


# ------------------------------------------------------------------ #
# Forward smoke                                                        #
# ------------------------------------------------------------------ #


def test_forward_returns_scalar_finite_energy():
    """Smoke: D=1 product state → finite scalar energy."""
    A = _make_d1_site(d=2, key=jax.random.PRNGKey(0))
    B = _make_d1_site(d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        E = honeycomb_ctm_energy_implicit(
            sites,
            H,
            chi=4,
            max_iter=20,
            conv_tol=1e-8,
            projector_method="biorthogonal",
            forward_gauge="phase",
        )
    assert jnp.isfinite(E)
    assert E.shape == ()


# ------------------------------------------------------------------ #
# Gradient finiteness on random tensors                                #
# ------------------------------------------------------------------ #


def test_grad_finite_on_random_d2_sites():
    """jax.grad through the implicit backward returns finite gradients.

    Random tensors do not in general make CTM converge, so GMRES may emit a
    convergence warning — that's expected and surface as a finite (possibly
    inaccurate) λ. We only assert *finiteness* here; the FD-AD numerical
    accuracy gate is the D=1 test below.
    """
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites_template = {(0, 0): A, (1, 0): B}
    coords = sorted(sites_template.keys())
    H = _heisenberg_bond_xxz()

    A_data = A.todense()
    B_data = B.todense()

    def loss(A_d, B_d):
        A_local = DenseTensor(A_d, A.indices)
        B_local = DenseTensor(B_d, B.indices)
        st = {(0, 0): A_local, (1, 0): B_local}
        return honeycomb_ctm_energy_implicit(
            st,
            H,
            chi=4,
            max_iter=10,
            conv_tol=1e-8,
            projector_method="biorthogonal",
            forward_gauge="phase",
            gmres_tol=1e-4,
            gmres_maxiter=20,
            gmres_restart=10,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E = loss(A_data, B_data)
        gA, gB = jax.grad(loss, argnums=(0, 1), holomorphic=False)(A_data, B_data)

    assert jnp.isfinite(E), f"E = {E}"
    assert jnp.all(jnp.isfinite(gA)), "grad A has non-finite entries"
    assert jnp.all(jnp.isfinite(gB)), "grad B has non-finite entries"
    _ = coords


# ------------------------------------------------------------------ #
# FD-vs-AD strict gate (D=1 product state)                             #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_fd_vs_ad_d1_product_state():
    """At D=1 the energy is exact in the env; AD grad must match central FD.

    Compares grad on a few elements of A's data against central finite
    differences. Median relative error must be < 5e-2 (matches the existing
    ``test_ctm_energy_implicit_gradient_matches_fd`` tolerance).
    """
    A = _make_d1_site(d=2, key=jax.random.PRNGKey(2))
    B = _make_d1_site(d=2, key=jax.random.PRNGKey(3))
    H = _heisenberg_bond_xxz()
    A_data = A.todense()

    def energy_fn(params_data):
        A_local = DenseTensor(params_data, A.indices)
        st = {(0, 0): A_local, (1, 0): B}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return honeycomb_ctm_energy_implicit(
                st,
                H,
                chi=4,
                max_iter=30,
                conv_tol=1e-10,
                projector_method="biorthogonal",
                forward_gauge="phase",
                gmres_tol=1e-9,
                gmres_maxiter=60,
                gmres_restart=20,
            )

    grad_ad = jax.grad(lambda x: energy_fn(x).real)(A_data)

    flat = A_data.ravel()
    grad_ad_flat = grad_ad.ravel()
    rel_errs = []
    for i in range(flat.size):
        fd_i = None
        for eps in (1e-4, 1e-5, 1e-6):
            e_plus = energy_fn(flat.at[i].add(eps).reshape(A_data.shape))
            e_minus = energy_fn(flat.at[i].add(-eps).reshape(A_data.shape))
            cand = float(((e_plus - e_minus) / (2 * eps)).real)
            if abs(cand) < 100:
                fd_i = cand
                break
        if fd_i is None:
            continue
        if abs(fd_i) < 5e-3:
            continue
        rel_errs.append(abs(float(grad_ad_flat[i].real) - fd_i) / abs(fd_i))

    assert len(rel_errs) >= 2, (
        f"Too few valid FD gradients: {len(rel_errs)} (likely all near zero)"
    )
    median_err = float(jnp.median(jnp.array(rel_errs)))
    max_err = max(rel_errs)
    assert median_err < 0.05, (
        f"D=1 median FD-AD rel err {median_err:.3e} > 0.05 "
        f"(max {max_err:.3e}, n={len(rel_errs)})"
    )
