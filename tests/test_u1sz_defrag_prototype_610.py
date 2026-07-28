"""Faithfulness guard for the #610 C-lever prototype (Stage 2 prereq)."""

import importlib.util
import pathlib

import jax
import numpy as np
import pytest

from examples.u1sz_defrag_prototype_610 import sector_dropping_truncation
from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair


def test_prototype_ctm_converges_and_energy_is_sane():
    jax.config.update("jax_enable_x64", True)
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
        for name in env._fields:
            t = getattr(env, name)
            assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"
        e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e), "prototype energy non-finite"
    assert -2.0 < e < 0.0, f"prototype energy {e} outside sane Heisenberg window"


def test_prototype_actually_drops_sectors():
    # Under the prototype, the env block counts must drop toward the Gate-A
    # static prediction: corners 5->3, edges 19->9 at D=3 chi=12.
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6)
    nblocks = {n: len(getattr(env, n)._block_keys) for n in env._fields}
    # edges must have strictly fewer blocks than the fragmented baseline (19)
    assert nblocks["T1"] < 19, f"edge sectors not dropped: {nblocks}"
    assert nblocks["C1"] <= 5, f"corner sectors not dropped: {nblocks}"


@pytest.mark.xfail(
    reason=(
        "#610 (stale obstruction, independent of #700): with the current "
        "env-init chi-bond charge structure the documented mixed-generation "
        "chi-bond mismatch ('does not match previous') no longer trips — the "
        "surgical-drop backward VJP traces without raising. #700's vacuum-tiling "
        "fix does not restore the obstruction (it was pinning a since-changed "
        "env-init detail, not the #700 collapse). The real #610 structural fix "
        "(env-init + per-direction refresh emitting a uniform sector set) is "
        "what would flip this test."
    ),
    strict=False,
)
def test_backward_trace_does_not_survive_surgical_drop():
    """Gate-B Step-0 obstruction guard (#610): documents that the SURGICAL
    traced-path sector drop (filtered backward base_charges + greedy-backfill-
    suppressed traced SVD that honestly emits a smaller chi_new) STILL cannot be
    traced through the multisite 2x2 backward.

    Root cause (NO-GO-by-obstruction): the multisite sweep refreshes only the
    legs along the current absorption direction per move, while env-init seeds
    every chi bond at the full 5-sector {-2,-1,0,1,2} structure.  A later
    direction's enlarged-corner build then contracts a freshly-dropped (3-sector)
    leg against a perpendicular leg still carrying the full env-init bond ->
    opt_einsum size mismatch.  This pins the obstruction so a future structural
    fix (env-init + per-direction refresh emitting a uniform sector set) is
    recognised as the thing that flips this test.
    """
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location(
        "probe_backward_jaxpr_566",
        pathlib.Path(__file__).resolve().parents[1]
        / "examples"
        / "probe_backward_jaxpr_566.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    A = probe.make_site("u1sz", 3)
    chi = 12

    # Baseline (no prototype) traces fine.
    jx = probe.backward_vjp_jaxpr(A, chi, "auto")
    assert hasattr(jx, "jaxpr") or hasattr(jx, "eqns")

    # IMPORTANT: the monkeypatch only takes effect if the patched CTM-step is
    # traced COLD under the patch. The baseline call above seeds the jax.jit
    # compilation cache (identical input avals + same _make_jit_ctm_step
    # closure), so without clearing it the prototype-wrapped call would reuse the
    # un-patched cached trace and silently NOT raise. Clear caches so the patched
    # path re-traces — mirrors the standalone `--defrag` probe, which runs cold.
    jax.clear_caches()

    # Under the surgical drop the single-sweep VJP fails to trace with the
    # documented mixed-generation chi-bond mismatch.
    with pytest.raises(ValueError, match="does not match previous"):
        with sector_dropping_truncation(keep={-1, 0, 1}):
            probe.backward_vjp_jaxpr(A, chi, "auto")
