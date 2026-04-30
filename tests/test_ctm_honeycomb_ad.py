"""Implicit-AD gradient tests for honeycomb CTM.

The strong gate is FD-vs-AD agreement on a D=2 *near-product* state — D=1
collapses the env legs to dim 1 and decouples the env from the physical
site, so the implicit machinery (GMRES on ``(I - J_env^T)`` plus
``J_sites^T λ``) is bypassed. A near-product D=2 site keeps the energy near
the classical product expectation while still routing gradients through the
full env Jacobian, which is what we need to verify is wired correctly.

The D=1 case is kept as a direct-gradient smoke (forward + grad finite) to
catch dtype / dispatch regressions cheaply.
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


def _make_near_product_d2_site(
    d: int, eps: float, key: jax.Array
) -> tuple[DenseTensor, tuple[TensorIndex, ...]]:
    """D=2 site obtained by perturbing a normalized product state by ``eps``.

    Layout: ``data[0, 0, 0, :]`` is the (random unit) product-state vector;
    every other entry is ``eps * complex_normal``. After construction the
    full tensor is renormalized to unit Frobenius norm.
    """
    sym = U1Symmetry()
    virt = np.zeros(2, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    data = np.zeros((2, 2, 2, d), dtype=np.complex128)
    re0 = np.array(jax.random.normal(key, (d,)))
    im0 = np.array(jax.random.normal(jax.random.fold_in(key, 1), (d,)))
    v0 = re0 + 1j * im0
    data[0, 0, 0, :] = v0 / np.linalg.norm(v0)

    pkey = jax.random.fold_in(key, 99)
    pre = np.array(jax.random.normal(pkey, (2, 2, 2, d)))
    pim = np.array(jax.random.normal(jax.random.fold_in(pkey, 1), (2, 2, 2, d)))
    pert = (pre + 1j * pim).astype(np.complex128)
    pert[0, 0, 0, :] = 0.0
    data = data + eps * pert
    data = data / np.linalg.norm(data)
    return DenseTensor(jnp.asarray(data), indices), indices


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
    accuracy gate is the D=2 near-product test below.
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
# FD-vs-AD strict gate (D=2 near-product, uniform A=B)                 #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_fd_vs_ad_d2_near_product_state():
    """D=2 near-product, uniform A=B: AD grad must match central FD.

    The implicit-AD machinery (GMRES on ``(I − J_env^T)`` and ``J_sites^T λ``)
    only does real work when env legs of dim ``D² > 1`` couple to the site.
    A near-product D=2 state stays close enough to the classical fixed point
    that the energy stabilizes (env elements may still oscillate by a U(1)
    gauge — that's fine, the energy is gauge-invariant), while exercising
    the full Jacobian.

    Filter ``|AD| < 1e-3`` (FD signal-to-noise floor at this CTM regularity)
    and require median FD-AD relative error ``< 5e-2`` over ≥ 3 valid
    elements — same tolerance as
    :func:`test_ctm_energy_implicit_gradient_matches_fd` for the square CTM.
    """
    A, A_indices = _make_near_product_d2_site(d=2, eps=0.05, key=jax.random.PRNGKey(7))
    H = _heisenberg_bond_xxz()
    A_data = A.todense()

    def energy_fn(params_data):
        A_local = DenseTensor(params_data, A_indices)
        st = {(0, 0): A_local, (1, 0): A_local}  # uniform A=B
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return honeycomb_ctm_energy_implicit(
                st,
                H,
                chi=4,
                max_iter=60,
                conv_tol=1e-8,
                projector_method="biorthogonal",
                forward_gauge="phase",
                gmres_tol=1e-7,
                gmres_maxiter=60,
                gmres_restart=20,
            )

    grad_ad = jax.grad(lambda x: energy_fn(x).real)(A_data)

    flat = A_data.ravel()
    grad_ad_flat = grad_ad.ravel()
    rel_errs = []
    for i in range(flat.size):
        fd_i = None
        for eps in (1e-5, 1e-6):
            e_plus = energy_fn(flat.at[i].add(eps).reshape(A_data.shape))
            e_minus = energy_fn(flat.at[i].add(-eps).reshape(A_data.shape))
            cand = float(((e_plus - e_minus) / (2 * eps)).real)
            if abs(cand) < 100:
                fd_i = cand
                break
        if fd_i is None:
            continue
        ad_i = float(grad_ad_flat[i].real)
        # Skip elements where both grads are below the noise floor — relative
        # error there is dominated by FD truncation, not implicit-AD error.
        if abs(ad_i) < 1e-3 and abs(fd_i) < 1e-3:
            continue
        # Use max(|fd|, |ad|) in the denominator to keep rel_err meaningful
        # when one side is near zero.
        rel_errs.append(abs(ad_i - fd_i) / max(abs(ad_i), abs(fd_i)))

    assert len(rel_errs) >= 3, (
        f"Too few valid FD gradients: {len(rel_errs)} (likely all near zero)"
    )
    median_err = float(jnp.median(jnp.array(rel_errs)))
    max_err = max(rel_errs)
    assert median_err < 0.05, (
        f"D=2 near-product median FD-AD rel err {median_err:.3e} > 0.05 "
        f"(max {max_err:.3e}, n={len(rel_errs)})"
    )
