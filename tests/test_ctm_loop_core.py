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
