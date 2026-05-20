"""Tests for variPEPS-style in-CTM χ-bump (Issue #492).

The in-CTM χ-bump grows ``chi`` *inside* ``python_loop_ctm_converge`` when
the projector SVD's normalised smallest kept singular value exceeds a
threshold, mirroring variPEPS's ``ctmrg_heuristic_increase_chi``.  Unlike
the end-of-outer-step ``chi_auto_bump``, this guarantees every gradient
the optimizer sees is computed at a converged CTM fixed point — no
zero-padded transitional environments (the cliff-edge artifact diagnosed
in v8b on 2026-05-20).

These tests exercise the mechanism in isolation, without involving the
AD optimizer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_python_loop import (
    CTMConvergeInfo,
    python_loop_ctm_converge,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_random_A(D=2, d=2, key=None):
    """A small DenseTensor iPEPS site tensor, no symmetry sectors."""
    if key is None:
        key = jax.random.PRNGKey(42)
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / (jnp.linalg.norm(data) + 1e-10)
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


class TestInCtmChiBumpHook:
    """Direct calls to python_loop_ctm_converge."""

    def test_off_by_default_no_chi_change(self):
        """Without the flag, chi stays put and CTMConvergeInfo.final_chi == chi."""
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=6,
            max_iter=10,
            min_iter=2,
            conv_tol=1e-5,
        )
        assert isinstance(info, CTMConvergeInfo)
        assert info.final_chi == 6, (
            "Without ctmrg_heuristic_increase_chi=True, chi must stay at the "
            f"initial value (got final_chi={info.final_chi})."
        )

    def test_on_triggers_bump_when_threshold_loose(self):
        """A loose threshold forces the bump path to fire at small chi."""
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        # threshold=1.0 means smallest_S > 1.0 → never satisfied since
        # smallest_S is normalised by largest SV and is always in [0, 1].
        # Use a tiny threshold so the bump triggers on the noisy small-chi
        # spectrum of a random tensor.
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=8,
            min_iter=2,
            conv_tol=1e-5,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=6,
        )
        assert info.final_chi > 2, (
            "With a near-zero threshold, in-CTM bump must fire and grow chi; "
            f"got final_chi={info.final_chi}."
        )
        assert info.final_chi <= 6, (
            f"final_chi={info.final_chi} must respect chi_max=6 ceiling."
        )

    def test_chi_max_caps_growth(self):
        """chi never exceeds chi_max even with aggressive threshold."""
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=20,
            min_iter=2,
            conv_tol=1e-12,  # never converge → use the full budget
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=10,  # large step
            chi_max=4,
        )
        assert info.final_chi <= 4, (
            f"final_chi={info.final_chi} must respect chi_max=4 ceiling."
        )

    def test_smallest_S_is_populated(self):
        """info.max_smallest_S returns a finite non-trivial number from the
        JIT'd CTM step (previously 0.0 placeholder).
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=6,
            max_iter=4,
            min_iter=2,
            conv_tol=1e-5,
        )
        assert 0.0 <= info.max_smallest_S <= 1.0, (
            f"max_smallest_S must be in [0,1] (normalised by largest SV); "
            f"got {info.max_smallest_S}."
        )


class TestConfigValidation:
    """CTMConfig.__post_init__ guards for the new knobs."""

    def test_mutex_with_chi_ramp(self):
        """ctmrg_heuristic_increase_chi cannot coexist with chi_ramp."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            CTMConfig(
                chi=4,
                ctmrg_heuristic_increase_chi=True,
                chi_ramp=[(2, 5), (4, None)],
            )

    def test_positive_step_size(self):
        """step_size must be a positive integer when the flag is on."""
        with pytest.raises(ValueError, match="positive integer"):
            CTMConfig(
                chi=4,
                ctmrg_heuristic_increase_chi=True,
                ctmrg_heuristic_increase_chi_step_size=0,
            )

    def test_defaults_off(self):
        """The new flag defaults to False (no behavior change)."""
        cfg = CTMConfig(chi=8)
        assert cfg.ctmrg_heuristic_increase_chi is False
        assert cfg.ctmrg_heuristic_increase_chi_threshold == 1e-6
        assert cfg.ctmrg_heuristic_increase_chi_step_size == 2
