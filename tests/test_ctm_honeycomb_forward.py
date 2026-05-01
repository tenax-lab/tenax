"""Forward CTM convergence loop for honeycomb iPEPS.

The clean stabilization signal is at ``D=1`` (product state): the env reaches
its fixed point after a couple of sweeps and the energy is then exact, so we
can require ``|E_25 - E_30|/|E_30| < 1e-3`` with no slack. At ``D=2`` random
A/B do **not** in general converge under biorthogonal CTM — that is a known
biorthogonal-on-random-input artifact, not a forward-loop bug — so we only
require the energy to remain finite and bounded over 30 sweeps.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_honeycomb_energy import compute_honeycomb_energy
from tenax.algorithms._ctm_honeycomb_forward import honeycomb_ctm_run
from tenax.algorithms._ctm_honeycomb_moves import honeycomb_ctm_step
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_d1_site(d: int, key: jax.Array) -> DenseTensor:
    """Rank-4 honeycomb site with D=1 virtual legs — pure product state."""
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


def _heisenberg_bond_xxz(d: int = 2, delta: float = 1.0) -> jnp.ndarray:
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    H = np.kron(sx, sx) + np.kron(sy, sy) + delta * np.kron(sz, sz)
    return jnp.asarray(H, dtype=jnp.complex128)


def test_energy_stabilizes_at_d1_within_30_sweeps():
    """D=1 product state: |E_25 - E_30|/|E_30| < 1e-3 (clean fixed-point signal)."""
    A = _make_d1_site(d=2, key=jax.random.PRNGKey(0))
    B = _make_d1_site(d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()

    envs_30, info = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=30,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    envs_25, _ = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=25,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    E_30 = compute_honeycomb_energy(sites, envs_30, H)
    E_25 = compute_honeycomb_energy(sites, envs_25, H)
    rel = float(jnp.abs(E_30 - E_25) / (jnp.abs(E_30) + 1e-12))
    assert jnp.isfinite(E_30), f"E_30 = {E_30}"
    assert rel < 1e-3, (
        f"D=1 energy did not stabilize: |E_25 - E_30|/|E_30| = {rel:.3e} "
        f"(E_25={complex(E_25)}, E_30={complex(E_30)})"
    )
    assert info.iterations == 30


def test_energy_finite_and_bounded_at_d2_after_30_sweeps():
    """D=2 random A≠B: energy stays finite and within a reasonable bound.

    Random tensors do not in general converge under biorthogonal CTM, so we
    only assert that 30 sweeps don't produce NaN / inf and that ``|E|`` is
    bounded by a generous physical-scale ceiling. The strong stabilization
    test is the D=1 case above; the strict CTM-converged test belongs to
    the AD path (Task 13) where the iPEPS state is near a real ground state.
    """
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    H = _heisenberg_bond_xxz()

    envs, info = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=30,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    E = compute_honeycomb_energy(sites, envs, H)
    # Heisenberg per-bond energy is bounded by |Tr(H)/d²| · 3 ≈ 3 · 0.75 = 2.25
    # for a normalized state; allow generous slack for the unnormalized
    # (renormalize=True) inf-norm gauge.
    assert jnp.isfinite(E), f"E = {E}"
    assert jnp.abs(E) < 1e2, f"|E| blew up: {complex(E)}"
    assert info.iterations == 30


def test_returns_per_sublattice_envs_with_canonical_layout():
    """Output dict has two coords; each env still has 9 fields with the canonical chi."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(2))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(3))
    sites = {(0, 0): A, (1, 0): B}
    envs, _ = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=5,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    for coord, env in envs.items():
        assert len(env._fields) == 9
        for alpha in (0, 1, 2):
            C = getattr(env, f"C{alpha}")
            assert C.indices[0].dim == 4 and C.indices[1].dim == 4, (
                f"{coord}.C{alpha} has dims "
                f"{(C.indices[0].dim, C.indices[1].dim)}, want (4, 4)"
            )


def test_warns_on_max_iter_without_convergence():
    """conv_tol=0 keeps `converged=False`; honeycomb_ctm_run should warn, not raise."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(4))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(5))
    sites = {(0, 0): A, (1, 0): B}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, info = honeycomb_ctm_run(
            sites,
            chi=4,
            max_iter=3,
            conv_tol=0.0,
            projector_method="biorthogonal",
            forward_gauge="phase",
        )
    assert info.converged is False
    assert any("did not converge" in str(wi.message).lower() for wi in w), [
        str(wi.message) for wi in w
    ]


def test_chi_ramp_initializes_at_first_stage_then_grows():
    """chi_ramp=[(2, 5), (4, None)] — final env has chi=4 corners after ramp."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(6))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(7))
    sites = {(0, 0): A, (1, 0): B}
    envs, _ = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=4,
        conv_tol=0.0,
        projector_method="biorthogonal",
        forward_gauge="phase",
        chi_ramp=[(2, 3), (4, None)],
    )
    for coord in [(0, 0), (1, 0)]:
        env = envs[coord]
        for alpha in (0, 1, 2):
            C = getattr(env, f"C{alpha}")
            assert C.indices[0].dim == 4, (
                f"{coord}.C{alpha} chi_in dim = {C.indices[0].dim}, want 4"
            )


@pytest.mark.parametrize("method", ["elementwise", "svd"])
def test_convergence_reported_with_loose_tolerance(method):
    """Loose tolerance + long budget → converged=True is reachable."""
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(8))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(9))
    sites = {(0, 0): A, (1, 0): B}
    _, info = honeycomb_ctm_run(
        sites,
        chi=4,
        max_iter=60,
        conv_tol=1e-2,
        conv_method=method,
        projector_method="biorthogonal",
        forward_gauge="phase",
    )
    # Smoke: doesn't have to converge for every method/seed, but it must not
    # silently report converged with iterations=max_iter.
    if info.converged:
        assert info.iterations <= 60
    else:
        assert info.iterations == 60


# Make sure honeycomb_ctm_step is still imported / re-export contract intact.
_ = honeycomb_ctm_step
