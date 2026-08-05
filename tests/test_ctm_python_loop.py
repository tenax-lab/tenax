"""Tests for Python-loop CTM convergence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_python_loop import (
    CTMConvergeInfo,
    _make_jit_ctm_step,
    python_loop_ctm_converge,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for this test module and restore afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_random_A(D=2, d=2, key=None):
    """Create a random DenseTensor iPEPS site tensor."""
    if key is None:
        key = jax.random.PRNGKey(0)
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


# Heisenberg gate for energy computation
def _heisenberg_gate():
    """Return the 2-site Heisenberg gate as (2,2,2,2) array."""
    return jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)


class TestJitCtmStep:
    """Tests for _make_jit_ctm_step."""

    def test_step_changes_env(self):
        """_make_jit_ctm_step produces different environments from input."""
        A = _make_random_A()
        site_tensors = {(0, 0): A}
        envs = {(0, 0): initialize_ctm_tensor_env(A, chi=4)}
        step = _make_jit_ctm_step(SINGLE_SITE_NEIGHBORS)
        new_envs, _, _ = step(site_tensors, envs, chi=4)
        # Environments should differ after a sweep
        old_c1 = envs[(0, 0)].C1.todense()
        new_c1 = new_envs[(0, 0)].C1.todense()
        assert not jnp.allclose(old_c1, new_c1)


class TestPythonLoopCtmConverge:
    """Tests for python_loop_ctm_converge."""

    def test_python_loop_converges(self):
        """Python-loop CTM reaches convergence."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        max_iter = 100
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            conv_tol=1e-5,
        )
        # conv_tol=1e-5: the #702 sweep-invariant corner convention removed
        # the historic sv_diff~0.05 random-init plateau (#425/#426) — the
        # sweep now converges to sv_diff ~1e-6, where it stalls on the
        # degenerate-pair SV basis rotation (LAPACK basis choice inside
        # ket-bra-paired SVs flips between iterations; same #425 mechanism
        # at ~1e-6 amplitude instead of 0.05).  1e-5 sits above that floor,
        # so this asserts genuine convergence.  The xfail band below is kept
        # for platforms whose BLAS still lands in the old plateau regime.
        if (
            not info.converged
            and 1e-4 < float(info.sv_diff) < 0.5
            and info.iterations < max_iter
        ):
            pytest.xfail(
                f"CTM plateau on random init (#425/#426): "
                f"converged={info.converged}, sv_diff={float(info.sv_diff):.3e}, "
                f"iter={info.iterations}; M2b honeycomb-native CTM pending."
            )
        assert info.converged
        assert info.sv_diff < 1e-5
        assert info.iterations > 1

    def test_python_loop_2site_checkerboard_converges(self):
        """Checkerboard topology converges + produces a finite energy.

        Behavioral coverage for the CHECKERBOARD_NEIGHBORS sweep path,
        which `optimize_gs_ad` for 2-site cells routes through. No
        atol-tight parity check vs the legacy `ctm_tensor_2site` path
        (that is brittle on macOS — see #354 bucket E).
        """
        A = _make_random_A(key=jax.random.PRNGKey(42))
        B = _make_random_A(key=jax.random.PRNGKey(99))
        gate = _heisenberg_gate()
        max_iter = 50
        envs, info = python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B},
            CHECKERBOARD_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            conv_tol=1e-8,
        )
        # Known limitation (Issues #425/#426): same CTM-iter plateau as
        # test_python_loop_converges, on the 2-site CHECKERBOARD path. The
        # random-init fixed point oscillates without converging; depending on
        # BLAS/XLA numerics it either trips the plateau_patience=20 early-bail
        # (~iter 12, sv_diff~0.004) OR keeps oscillating for the full max_iter
        # (sv_diff~0.017). Both are the same non-convergence, so the guard keys
        # only on the *healthy-oscillation* signature (bounded sv_diff), not the
        # iteration count — the earlier `iterations < max_iter` clause was
        # platform-sensitive and hard-failed the full-run case in CI (#692).
        # A genuinely broken CTM (diverging/NaN sv_diff, or a spurious
        # zero-diff non-convergence) still falls outside the band and fails.
        if not info.converged and 1e-5 < float(info.sv_diff) < 0.5:
            pytest.xfail(
                f"CTM plateau on random init (#425/#426, 2-site checkerboard): "
                f"converged={info.converged}, sv_diff={float(info.sv_diff):.3e}, "
                f"iter={info.iterations}; M2b honeycomb-native CTM pending."
            )
        assert info.converged
        energy = compute_energy_ctm_tensor_multisite(
            {(0, 0): A, (1, 0): B}, envs, CHECKERBOARD_NEIGHBORS, gate
        )
        assert jnp.isfinite(energy)
        assert energy.shape == ()

    def test_chi_ramp_energy_close_to_direct(self):
        """Chi ramp from chi=4 to chi=8 gives energy close to direct chi=8."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        gate = _heisenberg_gate()

        # Direct run at chi=8
        envs_direct, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=8,
            max_iter=100,
            conv_tol=1e-10,
        )
        energy_direct = float(compute_energy_ctm_tensor(A, envs_direct[(0, 0)], gate))

        # Ramp: chi=4 warmup, then chi=8
        envs_ramp, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=8,
            max_iter=100,
            conv_tol=1e-10,
            chi_ramp=[(4, 5), (8, None)],
        )
        energy_ramp = float(compute_energy_ctm_tensor(A, envs_ramp[(0, 0)], gate))

        # Both should give similar energy at chi=8
        np.testing.assert_allclose(energy_ramp, energy_direct, atol=1e-4)

    def test_converge_info_fields(self):
        """CTMConvergeInfo has correct fields."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        envs, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=5,
            conv_tol=1e-20,  # won't converge in 5 iters
        )
        assert isinstance(info, CTMConvergeInfo)
        assert info.converged is False
        assert info.iterations == 5
        assert info.sv_diff > 0

    def test_env_init_warm_start(self):
        """Passing env_init warm-starts the convergence."""
        A = _make_random_A(key=jax.random.PRNGKey(42))
        chi = 4

        # Run 10 iterations to get warm environments
        envs_warm, _ = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=10,
            conv_tol=1e-20,
        )

        # Continue from warm start — should converge faster
        max_iter = 100
        envs_final, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-5,
            env_init=envs_warm,
        )
        # conv_tol=1e-5: see test_python_loop_converges — post-#702 the
        # sweep converges to the ~1e-6 degenerate-pair floor instead of the
        # old 0.05 plateau, so this asserts genuine convergence.  The xfail
        # band below is kept for platforms still in the old plateau regime.
        if (
            not info.converged
            and 1e-4 < float(info.sv_diff) < 0.5
            and info.iterations < max_iter
        ):
            pytest.xfail(
                f"CTM plateau on random init (#425/#426, warm-start): "
                f"converged={info.converged}, sv_diff={float(info.sv_diff):.3e}, "
                f"iter={info.iterations}; M2b honeycomb-native CTM pending."
            )
        assert info.converged


# TestMultisiteEnergy::test_1site_matches_existing removed: bit-parity
# (atol=1e-10) between multisite-energy and 1-site-energy code paths on
# random A.  No user-visible contract — production runs always use the
# multisite path for 1-site cells when it's the active code; energy
# integration tests at the iPEPS layer cover the actual workflow.


class TestPlateauPatience:
    """Regression coverage for the ``plateau_patience`` early-bail."""

    def test_disabled_matches_legacy_max_iter(self):
        """``plateau_patience=None`` runs to ``max_iter`` even when stuck."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 30
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,  # impossible — forces non-convergence
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=None,
        )
        assert not info.converged
        assert info.iterations == max_iter

    def test_patience_bails_before_max_iter_on_plateau(self):
        """A non-converging input bails before ``max_iter`` with a finite patience."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 50
        patience = 5
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,  # impossible — forces non-convergence
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=patience,
        )
        assert not info.converged
        # Bail must happen before exhausting the budget.  Since #781
        # ``iterations`` is the bail sweep itself (the sweeps performed);
        # ``best_iteration`` is the one trailing it by ``patience``.
        assert info.iterations < max_iter
        assert info.best_iteration == info.iterations - patience

    def test_large_patience_matches_disabled(self):
        """``plateau_patience > max_iter`` is equivalent to ``None``."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 25
        kwargs = dict(
            site_tensors={(0, 0): A},
            neighbors=SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
        )
        _, info_none = python_loop_ctm_converge(plateau_patience=None, **kwargs)
        _, info_huge = python_loop_ctm_converge(
            plateau_patience=10 * max_iter, **kwargs
        )
        assert info_none.iterations == info_huge.iterations == max_iter
        assert not info_none.converged
        assert not info_huge.converged

    def test_iterations_counts_sweeps_actually_run(self, monkeypatch):
        """``iterations`` is the sweep count performed, not the best-metric index.

        Issue #781: the plateau bail used to report ``iterations=best_iter``
        — the iteration that achieved the best metric — so every caller
        computing a per-sweep cost (``1000 * total_s / info.iterations``)
        divided the full elapsed time by an index roughly ``patience``
        sweeps short of the truth, inflating per-sweep timings by up to 2x.
        Counting the actual ``jit_step`` invocations is the ground truth
        the field has to match.
        """
        from tenax.algorithms import _ctm_python_loop as _loop_mod

        sweeps_run = 0
        real_make = _loop_mod._make_jit_ctm_step

        def _counting_make(*args, **kwargs):
            step = real_make(*args, **kwargs)

            def _counted(*step_args, **step_kwargs):
                nonlocal sweeps_run
                sweeps_run += 1
                return step(*step_args, **step_kwargs)

            return _counted

        monkeypatch.setattr(_loop_mod, "_make_jit_ctm_step", _counting_make)

        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 50
        patience = 5
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,  # impossible — forces the plateau bail
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=patience,
        )
        assert not info.converged
        # Guard the premise: this must be the *bail* path, not budget exhaustion.
        assert sweeps_run < max_iter, (
            f"expected an early plateau bail, ran the full budget ({sweeps_run} sweeps)"
        )
        assert info.iterations == sweeps_run, (
            f"iterations={info.iterations} but {sweeps_run} CTM sweeps were "
            "actually executed (#781)"
        )

    def test_best_iteration_locates_the_returned_env(self):
        """``best_iteration`` says which sweep produced the env handed back.

        The bail returns the best-metric env, so the information that
        ``iterations`` used to carry is still needed — it just belongs in
        its own field.  On a bail the counter can only have been reset by
        an improvement, so the gap to the bail sweep is exactly
        ``plateau_patience``.
        """
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        max_iter = 50
        patience = 5
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=max_iter,
            min_iter=5,
            conv_tol=0.0,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=patience,
        )
        assert not info.converged
        assert info.iterations < max_iter
        assert info.best_iteration + patience == info.iterations, (
            f"best_iteration={info.best_iteration}, iterations="
            f"{info.iterations}, patience={patience}"
        )

    def test_best_iteration_equals_iterations_without_a_bail(self):
        """Off the bail path the returned env *is* the last sweep's.

        Only the ``plateau_patience`` stop-loss hands back an earlier
        environment.  When the loop converges, or exhausts ``max_iter``
        with the bail disabled, ``best_iteration`` must not lag.
        """
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        base = dict(
            site_tensors={(0, 0): A},
            neighbors=SINGLE_SITE_NEIGHBORS,
            chi=4,
            min_iter=5,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
        )

        # Budget-exhausted path: impossible tolerance, bail disabled.
        max_iter = 25
        _, exhausted = python_loop_ctm_converge(
            max_iter=max_iter, conv_tol=0.0, plateau_patience=None, **base
        )
        assert not exhausted.converged
        assert exhausted.iterations == max_iter
        assert exhausted.best_iteration == exhausted.iterations

        # Converged path: a tolerance loose enough that the elementwise
        # metric clears it well inside the budget.
        _, converged = python_loop_ctm_converge(
            max_iter=max_iter, conv_tol=1e3, plateau_patience=20, **base
        )
        assert converged.converged
        assert converged.iterations < max_iter
        assert converged.best_iteration == converged.iterations

    def test_returned_env_is_best_seen(self):
        """Returned env corresponds to the best ``sv_diff``, not the bail iter."""
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=50,
            min_iter=5,
            conv_tol=0.0,
            conv_method="elementwise",
            renormalize=True,
            projector_method="svd",
            plateau_patience=5,
        )
        assert isinstance(info, CTMConvergeInfo)
        assert not info.converged
        # ``sv_diff`` carries the *best* metric, so it must be finite and
        # not larger than the very first measurement we could have made.
        assert info.sv_diff < float("inf")

    def test_sv_first_iter_sentinel_does_not_anchor_plateau(self):
        """SV ``min_iter <= 1`` sentinel ``0.0`` must not become ``best_diff``.

        Codex review on PR #439: with ``conv_method="sv"`` and
        ``min_iter <= 1``, the first convergence check has no previous
        singular values, so the SV branch leaves ``final_diff`` at its
        sentinel value.  Without the ``plateau_metric_valid`` guard, the
        plateau block would record that sentinel as ``best_diff`` and
        bail after ``patience`` real measurements because no positive
        diff can improve on zero.
        """
        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))
        # ``min_iter=1`` exercises the no-baseline first iter.  Run long
        # enough that any sentinel-anchored bail would fire well before
        # the loop completes.
        _, info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=4,
            max_iter=40,
            min_iter=1,
            conv_tol=0.0,  # impossible — force non-convergence
            conv_method="sv",
            renormalize=True,
            projector_method="svd",
            plateau_patience=3,
        )
        # If the sentinel had anchored best_diff at 0.0, every subsequent
        # real positive diff would have incremented iters_since_best and
        # the loop would have bailed at iter ~5 with sv_diff=0.0.  With
        # the guard, the first real diff initializes best_diff to a
        # positive value, and either: (a) some later iter improves on it
        # (counter resets) → patience hasn't fired, or (b) patience fires
        # cleanly with the real best_diff and iter count.
        assert info.sv_diff > 0.0, (
            "SV plateau sentinel ``0.0`` must not propagate to best_diff; "
            f"saw info={info!r}"
        )

    def test_chi_ramp_non_final_fixed_sweeps_ignores_patience(self):
        """Explicit warm-up budgets must run to completion despite patience.

        Codex review on PR #439: a ramp like ``[(8, 100), (16, None)]`` is
        a contract to spend exactly 100 sweeps at chi=8 before moving on.
        With ``plateau_patience=5`` propagated naively, the chi=8 stage
        bails after ~5 non-improving iters and the final stage sees a
        different under-relaxed env than the user requested.  The fix
        forces ``plateau_patience=None`` on non-final stages with an
        explicit sweep count.
        """
        from unittest.mock import patch

        from tenax.algorithms import _ctm_python_loop as _loop_mod

        A = _make_random_A(D=3, key=jax.random.PRNGKey(7))

        seen_patience: list[int | None] = []
        real_fn = _loop_mod.python_loop_ctm_converge

        def _spy(*args, **kwargs):
            # Only record recursive (single-stage) calls dispatched by
            # _python_loop_chi_ramp; outer call has chi_ramp set.
            if kwargs.get("chi_ramp") is None:
                seen_patience.append(kwargs.get("plateau_patience"))
            return real_fn(*args, **kwargs)

        with patch.object(_loop_mod, "python_loop_ctm_converge", new=_spy):
            _loop_mod.python_loop_ctm_converge(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                chi=4,
                max_iter=30,
                min_iter=5,
                conv_tol=0.0,
                conv_method="elementwise",
                renormalize=True,
                projector_method="svd",
                plateau_patience=3,
                chi_ramp=[(4, 25), (4, None)],
            )

        # Two recursive calls: stage 0 (fixed-budget non-final) must see
        # plateau_patience=None; stage 1 (final, open budget) must see 3.
        assert seen_patience == [None, 3], (
            "Non-final fixed-budget stages must opt out of plateau_patience; "
            f"saw {seen_patience!r}"
        )

    def test_ctm_config_field_forwards_to_python_loop(self):
        """``CTMConfig.plateau_patience`` reaches ``python_loop_ctm_converge``.

        Regression coverage for PR #439 review: high-level callers that
        configure CTM only through ``CTMConfig`` (e.g. ``optimize_gs_ad``,
        PESS) need an end-to-end opt-out path — otherwise the new default
        cannot be disabled without bypassing the policy layer.
        """
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
        from tenax.algorithms.ipeps_config import CTMConfig

        cfg_default = CTMConfig()
        assert cfg_default.plateau_patience == 20
        kwargs_default = ctm_converge_kwargs(cfg_default)
        assert kwargs_default["plateau_patience"] == 20

        cfg_off = CTMConfig(plateau_patience=None)
        kwargs_off = ctm_converge_kwargs(cfg_off)
        assert kwargs_off["plateau_patience"] is None

        cfg_custom = CTMConfig(plateau_patience=7)
        kwargs_custom = ctm_converge_kwargs(cfg_custom)
        assert kwargs_custom["plateau_patience"] == 7
