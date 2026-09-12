"""Tests for ctm_energy_implicit: FD-AD gradient comparison."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit, ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate, ipeps
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_minimal_site_tensors_for_validation(seed: int = 0):
    """Build a minimal trivial-U(1) (D=2, d=2) site tensor for validation tests.

    Used by validation smoke tests that only need a syntactically valid
    site_tensors dict — the validation in ctm_energy_{implicit,explicit}
    fires before any CTM work, so a random tensor is fine.
    """
    import numpy as np

    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    rng = np.random.default_rng(seed)
    D, d = 2, 2
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys_charges.copy(), FlowDirection.IN, label="p"),
    )
    site = DenseTensor(
        rng.standard_normal((D, D, D, D, d)).astype(np.float64),
        indices,
    )
    return {(0, 0): site}


def _make_su_tensor(D=2, d=2):
    """Create a simple-update initialized tensor (well-conditioned for CTM)."""
    gate = heisenberg_gate()
    config = iPEPSConfig(max_bond_dim=D, num_imaginary_steps=100, dt=0.05)
    _, (A_su, _), _ = ipeps(gate, None, config)
    data = A_su.todense()
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_ctm_energy_implicit_gradient_is_finite_and_nontrivial():
    """Gradient from GMRES backward is finite, non-zero, and well-scaled.

    Captures the production contract for implicit-AD: the optimizer needs
    a finite, non-zero gradient with magnitude comparable to the energy.
    Strict FD-parity is *not* checked here — at D=2 χ=4 the 2x2 plaquette
    projector's stop_gradient (PR #447) drops the basis-rotation
    contribution from ∂(projector)/∂A, giving ~25% FD bias.  The bias
    shrinks at larger bond dimension and does not block L-BFGS
    convergence (verified empirically up to D=3, χ=24).
    """
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS
    A_data = A.todense()

    def energy_fn(params_data):
        A_local = _wrap_as_dense_tensor(params_data)
        site_tensors = {(0, 0): A_local}
        return ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=chi,
            max_iter=100,
            conv_tol=1e-12,
            gmres_tol=1e-10,
            gmres_maxiter=300,
            gmres_restart=50,
        )

    E = float(energy_fn(A_data))
    grad_ad = jax.grad(energy_fn)(A_data)

    assert jnp.all(jnp.isfinite(grad_ad)), "Gradient has NaN/Inf entries"
    g_norm = float(jnp.linalg.norm(grad_ad))
    assert g_norm > 1e-6 * (abs(E) + 1e-12), (
        f"Gradient norm too small: g_norm={g_norm:.3e}, E={E:.3e}"
    )
    assert g_norm < 1e6 * (abs(E) + 1e-12), (
        f"Gradient norm too large: g_norm={g_norm:.3e}, E={E:.3e}"
    )


@pytest.mark.slow
def test_ctm_energy_implicit_forward_runs():
    """Smoke test: forward pass produces a reasonable energy."""
    A = _make_su_tensor(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    site_tensors = {(0, 0): A}
    energy = ctm_energy_implicit(
        site_tensors,
        neighbors,
        gate,
        chi=chi,
        max_iter=40,
        conv_tol=1e-8,
    )
    assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
    assert energy.shape == (), f"Energy should be scalar, got shape {energy.shape}"


def test_ctm_energy_implicit_chi_max_required_when_bump_enabled():
    """Passing ctmrg_heuristic_increase_chi=True without chi_max must raise.

    Validates that the four bump kwargs are plumbed through
    ctm_energy_implicit → dispatch → _make_implicit_vjp_fn →
    _sigma_gauged_ctm_converge.  The validation in
    _sigma_gauged_ctm_converge fires before any CTM work happens, so we
    only need a minimal valid call to confirm the ValueError surfaces.
    """
    site_tensors = _make_minimal_site_tensors_for_validation()
    gate = heisenberg_gate()

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )


def test_ctm_energy_explicit_chi_max_required_when_bump_enabled():
    """ctm_energy_explicit(..., ctmrg_heuristic_increase_chi=True, chi_max=None) raises.

    Mirrors the implicit-AD validation smoke (Task 5).  The explicit-AD
    forward (#514 Task 6) plumbs the same four bump kwargs and reuses the
    same validation contract, so the ValueError must fire before any CTM
    work happens.
    """
    site_tensors = _make_minimal_site_tensors_for_validation()
    gate = heisenberg_gate()

    with pytest.raises(ValueError, match="chi_max"):
        ctm_energy_explicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            chi_max=None,
        )


# ---------------------------------------------------------------------------
# #841 honesty guard: the implicit-AD forward must not silently hand the
# backward a non-stationary environment.
#
# The implicit fixed-point backward linearizes gauge_fix(step(.)) around the
# environment the forward returns; its premise is env* = gauge_fix(step(env*)).
# Before the guard, `_sigma_gauged_ctm_converge` discarded the loop's
# convergence flag, and neither conv_method could certify the premise anyway:
# 'sv' certifies singular-value spectra only, and 'elementwise' can exit on a
# coincidental dip of a residual-gauge (Z2^chi bond-sign) limit cycle -- the
# consecutive-sweep diff touches 1e-13 on a sweep where the sign pattern
# happens to realign, then jumps back to O(0.3) on the very next sweep
# (measured on a D=2 SU state at chi=4: dips at sweeps 3, 5, 9, 17 of an
# otherwise flat 3.3e-01 plateau).  The guard measures the LITERAL residual
# ||gauge_fix(step(env*)) - env*|| with one extra gauged sweep and warns when
# it exceeds max(100*conv_tol, 1e-8).
# ---------------------------------------------------------------------------


def _normed_random_d2_site(seed: int = 0):
    """Deterministic normalized D=2 site tensor (trivial U(1) charges)."""
    site = _make_minimal_site_tensors_for_validation(seed)[(0, 0)]
    data = site.todense()
    return _wrap_as_dense_tensor(data / jnp.linalg.norm(data))


def test_forward_nonstationary_warns_when_starved():
    """A max_iter-starved forward must warn instead of silently returning.

    max_iter=6 on a state that needs ~40 sweeps at chi=4: the loop exits
    non-converged with a stationarity residual of O(0.6), far above the
    max(100*conv_tol, 1e-8) threshold.  Before #841's guard this outcome
    was discarded at the `_sigma_gauged_ctm_converge` call site and the
    backward would linearize around the non-stationary environment with
    no diagnostic at all.
    """
    site = _normed_random_d2_site()
    gate = heisenberg_gate()

    with pytest.warns(RuntimeWarning, match="stationarity residual"):
        ctm_energy_implicit(
            {(0, 0): site},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=6,
            min_iter=2,
            conv_tol=1e-9,
            conv_method="elementwise",
        )


def test_forward_stationary_does_not_warn():
    """A truly stationary forward fixed point must NOT trigger the guard.

    recipe="1x1" is the one configuration measured to reach a literally
    stationary environment (the #841 campaign\'s control: its implicit
    gradient has slope_fd/|g| = 1.0000 on the very state where 2x2 gives
    0.13).  On the random D=2 fixture it converges element-wise at sweep
    ~79 and the residual keeps SHRINKING over further sweeps (7.7e-10 ->
    3.1e-10 over five extra sweeps, measured) -- a stable fixed point,
    not a coincidence dip.  The guard must stay quiet here; this pins it
    to the literal residual rather than to "2x2 always warns".

    (The 1x1 recipe is deprecated (#911) because its fixed point is a
    rank-collapsed environment -- stationary but physically wrong for
    D > 1.  That makes it unfit for production and exactly fit for this
    control: stationarity is the property under test, not energy
    quality.  Its deprecation UserWarning is swallowed by the recorder.)
    """
    import warnings as _warnings

    site = _normed_random_d2_site()
    gate = heisenberg_gate()

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ctm_energy_implicit(
            {(0, 0): site},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=120,
            conv_tol=1e-9,
            conv_method="elementwise",
            recipe="1x1",
        )
    # Match the guard's distinctive phrase, not bare "stationar": the 1x1
    # deprecation text (#911) itself contains "no environment is stationary".
    stationarity_warnings = [
        w for w in caught if "stationarity residual" in str(w.message).lower()
    ]
    assert not stationarity_warnings, (
        f"Guard fired on a stationary fixed point: {stationarity_warnings}"
    )


@pytest.mark.slow
def test_sv_convergence_does_not_certify_elementwise_premise():
    """#841 signature: 'sv' reports converged while the premise fails.

    On a D=3 SU-relaxed state at chi=9, conv_method='sv' crosses its
    tolerance in ~11 sweeps (the corner spectra genuinely converge), but
    the literal stationarity residual stays at O(0.5): the phase gauge
    leaves Z2^chi bond signs and near-degenerate multiplet rotations
    unpinned, so the environment never reaches an element-wise fixed
    point.  This is exactly the configuration in which issue #841
    measured slope_fd/|g| = 0.13: the forward reports success, the
    backward premise is violated, and before the guard nothing warned.
    """
    import warnings as _warnings

    from tenax.algorithms._ctm_energy_ad import get_last_implicit_ad_diagnostics
    from tenax.algorithms.ipeps import ipeps
    from tenax.algorithms.ipeps_config import iPEPSConfig

    gate = heisenberg_gate()
    config = iPEPSConfig(max_bond_dim=3, num_imaginary_steps=30, dt=0.05)
    with _warnings.catch_warnings():
        # ipeps()'s own #839 CTM warning is not under test here.
        _warnings.simplefilter("ignore")
        _, (A_su, _), _ = ipeps(gate, None, config)
    data = A_su.todense()
    site = _wrap_as_dense_tensor(data / jnp.linalg.norm(data))

    with pytest.warns(RuntimeWarning, match="stationarity residual"):
        ctm_energy_implicit(
            {(0, 0): site},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=9,
            max_iter=25,
            conv_tol=1e-5,
            conv_method="sv",
        )

    diag = get_last_implicit_ad_diagnostics()
    # The loop itself must have reported success -- that is the point:
    # 'sv' certifies converged spectra while the element-wise premise fails.
    assert diag["forward_converged"] is True
    assert diag["forward_stationarity_residual"] > 0.1
