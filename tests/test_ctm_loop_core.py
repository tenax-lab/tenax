"""Smoke tests for _run_ctm_loop_with_bump helper."""

from __future__ import annotations

import pytest


def test_ctmloopresult_fields_present():
    """CTMLoopResult exposes all required fields with correct types."""
    from tenax.algorithms._ctm_loop_core import CTMLoopResult

    r = CTMLoopResult(
        envs={},
        converged=True,
        iterations=5,
        sv_diff=1e-9,
        max_truncation_error=0.0,
        max_smallest_S=0.0,
        final_chi=8,
        bump_extra_sweeps=0,
    )
    assert r.converged is True
    assert r.iterations == 5
    assert r.final_chi == 8
    assert r.bump_extra_sweeps == 0


def test_helper_runs_one_sweep_no_bump():
    """Helper runs `max_iter` sweeps with bump disabled and returns final_chi=chi_current."""
    import numpy as np

    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    rng = np.random.default_rng(0)
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
    site_tensors = {(0, 0): site, (1, 0): site}
    neighbors = CHECKERBOARD_NEIGHBORS
    envs = {c: initialize_ctm_tensor_env(A, 4) for c, A in site_tensors.items()}
    jit_step = _make_jit_ctm_step(neighbors)

    result = _run_ctm_loop_with_bump(
        jit_step,
        site_tensors,
        envs,
        chi_current=4,
        chi_max=None,
        bump_enabled=False,
        bump_threshold=1e-6,
        bump_step_size=2,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        gauge_fix_fn=None,
        max_iter=3,
        min_iter=10,  # never converges → uses full budget
        conv_tol=1e-12,
        conv_method="sv",
        plateau_patience=None,
        bump_base_charges=None,
    )

    assert result.iterations == 3
    assert result.converged is False
    assert result.final_chi == 4
    assert result.bump_extra_sweeps == 0


def _build_site_tensors_2site():
    """Shared 2-site checkerboard fixture used by gauge-arg regression tests."""
    import numpy as np

    from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    rng = np.random.default_rng(0)
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
    site_tensors = {(0, 0): site, (1, 0): site}
    envs_init = {c: initialize_ctm_tensor_env(A, 4) for c, A in site_tensors.items()}
    return site_tensors, envs_init


def test_helper_gauge_fn_receives_start_of_iter_as_envs_old_no_bump():
    """gauge_fix_fn(envs_new, envs_old): envs_old must be the env at the start
    of the iteration (i.e. the same env passed into jit_step), NOT the
    post-step env_new.

    Without this invariant, sigma-gauge alignment uses the wrong reference.
    Regression for the Task 4 review finding (bump-off path).
    """
    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors, envs_init = _build_site_tensors_2site()
    raw_jit_step = _make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)

    # Record the id() of the envs dict passed into jit_step on every sweep,
    # and the (envs_new_id, envs_old_id) pair passed to gauge_fix_fn.
    passed_to_step_ids: list[int] = []
    received_gauge_args: list[tuple[int, int]] = []

    def _spy_step(site_tensors, envs, **kwargs):
        passed_to_step_ids.append(id(envs))
        return raw_jit_step(site_tensors, envs, **kwargs)

    def _spy_gauge(envs_new, envs_old):
        received_gauge_args.append((id(envs_new), id(envs_old)))
        # Return envs_new unchanged so the loop progresses naturally.
        return envs_new

    _run_ctm_loop_with_bump(
        _spy_step,
        site_tensors,
        envs_init,
        chi_current=4,
        chi_max=None,
        bump_enabled=False,
        bump_threshold=1e-6,
        bump_step_size=2,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        gauge_fix_fn=_spy_gauge,
        max_iter=3,
        min_iter=10,  # never converges → uses full budget
        conv_tol=1e-12,
        conv_method="sv",
        plateau_patience=None,
        bump_base_charges=None,
    )

    # Three iterations, three sweeps, three gauge calls.
    assert len(passed_to_step_ids) == 3
    assert len(received_gauge_args) == 3

    # Structural contract: in each iteration, the env passed as the 2nd
    # positional arg to gauge_fix_fn (envs_old) must be the *same object* as
    # the env that was passed into jit_step that iteration.  i.e. start-of-iter.
    received_old_ids = [old for (_new, old) in received_gauge_args]
    assert received_old_ids == passed_to_step_ids, (
        "gauge_fix_fn second arg must be the start-of-iter env (the env "
        "passed to jit_step), not the post-step env."
    )


def test_helper_gauge_fn_receives_start_of_iter_as_envs_old_with_bump():
    """Same contract on the bump path: gauge_fix_fn(envs_post_resweep,
    envs_old) must receive the start-of-iter env as envs_old, NOT the
    pre-bump envs_new from the same iteration.

    Load-bearing case for sigma-gauge + bump (#514 follow-up).
    """
    from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump
    from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    site_tensors, envs_init = _build_site_tensors_2site()
    raw_jit_step = _make_jit_ctm_step(CHECKERBOARD_NEIGHBORS)

    # The first jit_step call of each iteration receives the start-of-iter env.
    # The second jit_step call (post-bump resweep) receives the padded env.
    # We record the start-of-iter envs by id on the first call per iter.
    sweep_envs_ids: list[int] = []
    received_gauge_args: list[tuple[int, int]] = []

    def _spy_step(site_tensors, envs, **kwargs):
        sweep_envs_ids.append(id(envs))
        return raw_jit_step(site_tensors, envs, **kwargs)

    def _spy_gauge(envs_new, envs_old):
        received_gauge_args.append((id(envs_new), id(envs_old)))
        return envs_new

    # bump_threshold=0.0 forces every iteration to bump (max_smallest_S > 0).
    _run_ctm_loop_with_bump(
        _spy_step,
        site_tensors,
        envs_init,
        chi_current=4,
        chi_max=8,
        bump_enabled=True,
        bump_threshold=0.0,
        bump_step_size=2,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        gauge_fix_fn=_spy_gauge,
        max_iter=2,
        min_iter=10,
        conv_tol=1e-12,
        conv_method="sv",
        plateau_patience=None,
        bump_base_charges=None,
    )

    # With bump firing, each iteration calls jit_step twice (pre-bump +
    # post-resweep).  The start-of-iter env is the first call of each iter.
    assert len(received_gauge_args) >= 1, "bump should have fired at least once"
    # First sweep of iter 0 received envs_init.
    assert sweep_envs_ids[0] == id(envs_init)
    # The first gauge call's envs_old must equal the iter-0 start env.
    assert received_gauge_args[0][1] == sweep_envs_ids[0], (
        "bump-path gauge_fix_fn second arg must be the start-of-iter env, "
        "NOT the pre-bump envs_new from the same iteration."
    )
    # And envs_old must NOT equal the post-resweep env (sweep idx 1) or the
    # pre-bump envs_new (which is a fresh dict from the first jit_step).
    # We can check the negative-id case: envs_old should differ from the
    # second-sweep input id (the padded env constructed inside the helper).
    if len(sweep_envs_ids) >= 2:
        assert received_gauge_args[0][1] != sweep_envs_ids[1], (
            "envs_old must not be the padded-and-resweep input env."
        )
