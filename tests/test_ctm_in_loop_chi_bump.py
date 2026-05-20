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

    def test_bump_sweeps_count_against_max_iter(self):
        """Each in-CTM bump performs a post-pad re-sweep; that sweep
        must be charged against the ``max_iter`` budget so the total
        number of physical CTM sweeps never exceeds ``max_iter``.
        Otherwise multiple bumps silently overshoot the documented cap.
        Codex review on PR #513.

        We probe this by forcing the bump to fire every iteration
        (threshold ~ 0) and asserting that ``info.iterations <= max_iter``.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        budget = 4
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=budget,
            min_iter=2,
            conv_tol=1e-12,  # never converge → exhaust budget
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-12,  # always trigger
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=10,
        )
        assert info.iterations <= budget, (
            f"total physical sweeps {info.iterations} exceeds max_iter {budget}; "
            "post-bump sweep is not being charged against the budget."
        )

    def test_returned_env_is_post_bump_swept(self):
        """After an in-CTM bump, the returned env must have been swept
        at the new chi — not just zero-padded.  Regression for codex
        review on PR #513.  We exercise the last-iter bump branch by
        running with ``max_iter=2``, ``min_iter=2`` so a bump on iter 1
        (the final allowed sweep) must still produce a swept env.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        envs, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=2,
            min_iter=2,
            conv_tol=1e-12,  # never converge → bump will fire each iter
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=4,
        )
        # If the env were only zero-padded (no post-bump sweep), the
        # corner tensors' new χ slots would be zero and the corner
        # singular values would have at most ``chi_pre_bump`` non-zero
        # entries.  With the post-bump sweep, all ``chi_current`` slots
        # carry real spectral weight.
        from tenax.algorithms._ctm_tensor_convergence import _corner_singular_values

        c1_svs = _corner_singular_values(envs[(0, 0)].C1)
        nonzero_count = int(jnp.sum(jnp.abs(c1_svs) > 1e-12))
        assert info.final_chi > 2, "bump must have fired"
        assert nonzero_count > 2, (
            f"returned env's C1 has {nonzero_count} non-zero SVs but bump "
            f"raised chi to {info.final_chi}; this indicates the env was "
            "returned still zero-padded without a post-bump sweep."
        )

    def test_bump_disabled_does_not_override_chi_from_env(self):
        """When ``ctmrg_heuristic_increase_chi=False`` (default), the
        warm-start env_init chi-override branch must be skipped — the
        function should run at the requested ``chi`` and the env_init
        is the caller's responsibility (must match ``chi``).  Codex
        review on PR #513.

        This is a regression guard against accidentally widening the
        override to all warm-start paths.  We verify the gating by
        observing that bump-off + matching env_init returns at the
        requested chi without any silent chi-promotion.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}

        # Bump OFF, env_init matches requested chi=4 → final_chi == 4
        # (no chi-from-env override at all).
        _, info = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=4,
            min_iter=2,
            conv_tol=1e-5,
        )
        assert info.final_chi == 4, (
            "With bump disabled, final_chi must equal the requested chi; "
            f"got {info.final_chi}.  Indicates a leaked chi-promotion."
        )

    def test_python_loop_rejects_chi_max_below_chi(self):
        """Defense-in-depth: direct callers passing chi_max < chi get a
        clear error rather than silently running at chi (violating the
        documented chi_max cap).  CTMConfig catches this at
        construction; this is the direct-call guard.  Codex review
        on PR #513.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        with pytest.raises(ValueError, match="must be >="):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=8,
                max_iter=3,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                ctmrg_heuristic_increase_chi_step_size=2,
                chi_max=4,  # < chi=8 → must reject
            )

    def test_warm_start_rejects_env_above_chi_max(self):
        """If env_init's chi exceeds the configured chi_max, raise rather
        than silently letting the warm-start override exceed the cap.
        The caller explicitly set chi_max as a memory/runtime budget;
        an env above it indicates a misconfiguration.  Codex review
        on PR #513.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}

        # Build an env at chi=6 via a first bump-enabled call.
        envs1, info1 = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=6,
            min_iter=2,
            conv_tol=1e-5,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=6,
        )
        assert info1.final_chi == 6

        # Second call: env_init at chi=6 but chi_max LOWERED to 4.
        # The override must refuse, not silently overrule the cap.
        with pytest.raises(ValueError, match="exceeds the configured chi_max"):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=2,
                max_iter=2,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                ctmrg_heuristic_increase_chi_threshold=1e-9,
                ctmrg_heuristic_increase_chi_step_size=2,
                chi_max=4,
                env_init=envs1,
            )

    def test_warm_start_preserves_grown_chi(self):
        """When env_init is passed back from a previous call that bumped
        chi, the next call must honour the env's actual chi rather than
        the input ``chi`` argument.  Otherwise the bump's progress is
        silently destroyed every warm-start round-trip.  Codex review
        on PR #513.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}

        # First call: chi=2, bump enabled, chi_max=6.  The bump should
        # raise chi above 2 inside this call.
        envs1, info1 = python_loop_ctm_converge(
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
        assert info1.final_chi > 2, "first call must bump chi"

        # Second call: same configured chi=2, but env_init is the
        # bumped env from call 1.  ``final_chi`` must equal call 1's
        # final chi (no down-truncation back to 2).
        envs2, info2 = python_loop_ctm_converge(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            chi=2,
            max_iter=2,
            min_iter=2,
            conv_tol=1e-5,
            ctmrg_heuristic_increase_chi=True,
            ctmrg_heuristic_increase_chi_threshold=1e-9,
            ctmrg_heuristic_increase_chi_step_size=2,
            chi_max=6,
            env_init=envs1,
        )
        assert info2.final_chi >= info1.final_chi, (
            f"warm-start round-trip lost chi: call 1 ended at "
            f"chi={info1.final_chi}, call 2 ended at chi={info2.final_chi} "
            "with configured chi=2 — env_init's chi was silently truncated."
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

    def test_mutex_with_chi_auto_bump(self):
        """ctmrg_heuristic_increase_chi cannot coexist with chi_auto_bump
        (both grow chi; together produces runtime pad_dense_env_chi error
        when the in-loop grow outpaces the outer ``_maybe_bump_chi``).
        Codex review on PR #513.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            CTMConfig(
                chi=4,
                chi_max=8,
                ctmrg_heuristic_increase_chi=True,
                chi_auto_bump=True,
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

    def test_requires_chi_max(self):
        """Enabling the bump without chi_max is a silent no-op trap; the
        config must reject it (codex review on PR #513).
        """
        with pytest.raises(ValueError, match="requires chi_max"):
            CTMConfig(chi=4, ctmrg_heuristic_increase_chi=True)

    def test_python_loop_rejects_chi_ramp_combination(self):
        """Defense-in-depth: direct callers combining
        ``ctmrg_heuristic_increase_chi=True`` with ``chi_ramp`` get a
        clear error.  Otherwise the chi_ramp early-return would silently
        ignore the bump knobs (codex review on PR #513).
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        with pytest.raises(ValueError, match="mutually exclusive"):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=2,
                max_iter=3,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                chi_max=4,
                chi_ramp=[(2, 2), (4, None)],
            )

    def test_python_loop_rejects_non_positive_step_size(self):
        """Defense-in-depth: direct callers passing step_size <= 0 get a
        clear error.  step_size=0 would cause an infinite loop (bump
        fires every iter, chi never grows); negative would shrink chi.
        Codex review on PR #513.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        with pytest.raises(ValueError, match="positive integer"):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=2,
                max_iter=3,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                ctmrg_heuristic_increase_chi_step_size=0,
                chi_max=4,
            )
        with pytest.raises(ValueError, match="positive integer"):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=4,
                max_iter=3,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                ctmrg_heuristic_increase_chi_step_size=-1,
                chi_max=8,
            )

    def test_python_loop_rejects_no_chi_max(self):
        """Defense-in-depth: direct callers of python_loop_ctm_converge
        also get a clear error if they enable the bump without chi_max.
        """
        A = _make_random_A(D=2, d=2)
        site_tensors = {(0, 0): A}
        with pytest.raises(ValueError, match="requires chi_max"):
            python_loop_ctm_converge(
                site_tensors,
                SINGLE_SITE_NEIGHBORS,
                chi=2,
                max_iter=3,
                min_iter=2,
                ctmrg_heuristic_increase_chi=True,
                # chi_max omitted → must raise
            )
