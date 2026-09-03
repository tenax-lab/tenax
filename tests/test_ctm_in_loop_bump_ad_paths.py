"""In-CTM bump integration tests for ctm_energy_implicit + ctm_energy_explicit (#514).

Task 8 of #514: end-to-end coverage that the four bump kwargs are correctly
plumbed through the AD entry points and surface their validation errors
*before* any CTM work runs.  Also covers the boundary mutex check (Part A:
``chi_ramp + bump`` rejected at ``ctm_energy_implicit``) and the explicit-AD
chi-lock invariant (backprop sweeps must all use the same fixed chi for
tape integrity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as _energy_ad_mod
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_explicit,
    ctm_energy_implicit,
)
from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

# Bucket comes from ``_FILE_MARKERS`` in conftest, not from a module-level
# ``pytestmark`` (#933): conftest *adds* its marker, so a module-level core
# mark would override the per-test ``@pytest.mark.slow`` withholding below.


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _build_site_tensor(seed: int = 0, D: int = 2, d: int = 2) -> DenseTensor:
    """Single trivial-U(1) (D, D, D, D, d) DenseTensor.

    Used by validation-error tests where the actual values do not matter
    because the bump-validation raises before any CTM step runs.
    """
    rng = np.random.default_rng(seed)
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
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    data = rng.standard_normal((D, D, D, D, d)).astype(np.float64)
    return DenseTensor(data, indices)


def _wrap_as_dense_tensor(arr: jax.Array) -> DenseTensor:
    """Wrap a raw ``jax.Array`` site tensor as a DenseTensor (D,D,D,D,d)."""
    arr = jnp.asarray(arr)
    D = arr.shape[0]
    d = arr.shape[4]
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
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(arr, indices)


# ---------------------------------------------------------------------------
# Implicit-AD bump tests
# ---------------------------------------------------------------------------


def test_implicit_ad_chi_max_required_raises():
    """ctm_energy_implicit(bump=True, chi_max=None) raises before any CTM work."""
    site_tensors = {(0, 0): _build_site_tensor()}
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


def test_implicit_ad_step_size_zero_raises():
    """ctm_energy_implicit(bump=True, step_size=0) raises."""
    site_tensors = {(0, 0): _build_site_tensor()}
    gate = heisenberg_gate()
    with pytest.raises(ValueError, match="positive integer"):
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_step_size=0,
            chi_max=8,
        )


def test_implicit_ad_chi_ramp_plus_bump_raises():
    """ctm_energy_implicit(bump=True, chi_ramp=[(4,5),(8,None)]) raises at the
    ctm_energy_implicit boundary (not deep inside the loop).  Regression for
    Task 5 review: previously this combination silently no-op'd because the
    chi_ramp branch in _run_forward drops the bump kwargs.
    """
    site_tensors = {(0, 0): _build_site_tensor()}
    gate = heisenberg_gate()
    with pytest.raises(ValueError, match="chi_ramp"):
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            chi_ramp=[(4, 5), (8, None)],
            ctmrg_heuristic_increase_chi=True,
            chi_max=8,
        )


def test_implicit_ad_env_init_above_chi_max_raises():
    """env_init with chi=8 + chi_max=6 raises with bump enabled."""
    A = _build_site_tensor()
    site_tensors = {(0, 0): A}
    env_init = {(0, 0): initialize_ctm_tensor_env(A, 8)}
    gate = heisenberg_gate()
    with pytest.raises(ValueError, match="exceeds the configured chi_max"):
        ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            max_iter=2,
            ctmrg_heuristic_increase_chi=True,
            chi_max=6,
            env_init=env_init,
        )


# ---------------------------------------------------------------------------
# Explicit-AD bump tests
# ---------------------------------------------------------------------------


def test_explicit_ad_chi_max_required_raises():
    """ctm_energy_explicit(bump=True, chi_max=None) raises."""
    site_tensors = {(0, 0): _build_site_tensor()}
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


def test_explicit_ad_step_size_zero_raises():
    """ctm_energy_explicit(bump=True, step_size=0) raises."""
    site_tensors = {(0, 0): _build_site_tensor()}
    gate = heisenberg_gate()
    with pytest.raises(ValueError, match="positive integer"):
        ctm_energy_explicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_step_size=0,
            chi_max=8,
        )


def test_explicit_ad_env_init_above_chi_max_raises():
    """env_init.chi > chi_max raises for explicit-AD."""
    A = _build_site_tensor()
    site_tensors = {(0, 0): A}
    env_init = {(0, 0): initialize_ctm_tensor_env(A, 8)}
    gate = heisenberg_gate()
    with pytest.raises(ValueError, match="exceeds the configured chi_max"):
        ctm_energy_explicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=2,
            backprop_steps=1,
            ctmrg_heuristic_increase_chi=True,
            chi_max=6,
            env_init=env_init,
        )


def test_explicit_ad_grad_flows_through_bump():
    """jax.grad through ctm_energy_explicit with bump enabled returns finite grads."""
    rng = np.random.default_rng(0)
    A_data = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)).astype(np.float64))
    gate = heisenberg_gate()

    def _loss(params):
        site_tensors = {(0, 0): _wrap_as_dense_tensor(params)}
        return ctm_energy_explicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
            warmup_steps=3,
            backprop_steps=2,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=8,
        )

    grad = jax.grad(_loss)(A_data)
    assert np.all(np.isfinite(np.asarray(grad))), (
        "explicit-AD grad through bump-enabled ctm_energy_explicit must be finite"
    )


def test_explicit_ad_backprop_uses_fixed_chi(monkeypatch):
    """After bump in warmup, backprop sweeps all use chi_post_warmup.

    Verified by spy: count distinct chi values seen by jit_step during the
    backprop phase — must be exactly 1.  (The explicit-AD bump kwargs only
    affect warmup; backprop must trace once for tape integrity.)

    Spy mechanism: monkey-patch ``_make_jit_ctm_step`` in
    ``_ctm_energy_ad`` to return a wrapper that records the ``chi`` kwarg
    each call.  The warmup goes through ``_run_ctm_loop_with_bump`` (which
    passes the same ``jit_step`` we hand out), and the backprop loop uses
    a closure that calls the same ``jit_step``.  We split warmup vs
    backprop by capturing the last warmup-phase ``chi`` value: the
    backprop phase begins after that.
    """
    rng = np.random.default_rng(0)
    A = _wrap_as_dense_tensor(
        jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)).astype(np.float64))
    )
    site_tensors = {(0, 0): A}
    gate = heisenberg_gate()

    # Wrap _make_jit_ctm_step to return a recording proxy.
    chi_history: list[int] = []
    real_make = _make_jit_ctm_step

    def _spy_make(neighbors):
        inner = real_make(neighbors)

        def _wrapped(st, envs, *, chi, **kw):
            chi_history.append(int(chi))
            return inner(st, envs, chi=chi, **kw)

        return _wrapped

    monkeypatch.setattr(_energy_ad_mod, "_make_jit_ctm_step", _spy_make)

    warmup_steps = 3
    backprop_steps = 2

    ctm_energy_explicit(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=4,
        warmup_steps=warmup_steps,
        backprop_steps=backprop_steps,
        ctmrg_heuristic_increase_chi=True,
        ctmrg_heuristic_increase_chi_threshold=1e-9,
        ctmrg_heuristic_increase_chi_step_size=2,
        chi_max=8,
    )

    # The last ``backprop_steps`` entries of ``chi_history`` are the
    # backprop-phase ``chi`` values.  They must all be identical.
    assert len(chi_history) >= backprop_steps, (
        f"spy recorded too few calls: {len(chi_history)} < {backprop_steps}"
    )
    backprop_chis = chi_history[-backprop_steps:]
    assert len(set(backprop_chis)) == 1, (
        f"explicit-AD backprop must use a single fixed chi for tape integrity; "
        f"observed chi values across {backprop_steps} backprop sweeps: "
        f"{backprop_chis}.  Full chi_history: {chi_history}"
    )
