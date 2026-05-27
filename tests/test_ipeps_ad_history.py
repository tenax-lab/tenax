"""Tests for the optimize_gs_ad history-capture hook."""

import jax.numpy as jnp
import pytest

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad


def _heisenberg_gate():
    """Real-dtype Heisenberg gate, matching tests/test_ipeps_excitations.py."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _fast_config(*, return_history: bool, num_steps: int = 3) -> iPEPSConfig:
    """Minimal-cost config that still exercises the AD optimizer loop.

    Uses the **explicit-AD** path (``gs_implicit_ad=False``) with a tiny
    CTM unroll so the JIT graph is small enough to compile in seconds.
    The history accumulator code is path-agnostic — it lives in the Python
    optimizer loop, outside ``value_and_grad``, so this test validates the
    return-shape contract regardless of implicit vs explicit AD.  Implicit
    AD on this same config is the slow path (bicgstab adjoint compile) and
    is exercised in real workloads, not in this unit test.
    """
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=5),
        gs_num_steps=num_steps,
        gs_learning_rate=1e-2,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        gs_optimizer="adam",  # Adam doesn't trigger L-BFGS line-search JIT
        return_history=return_history,
    )


@pytest.mark.algorithm
def test_optimize_gs_ad_returns_history_1site():
    """When return_history=True, the optimizer appends a history dict to its return tuple.

    The 1-site and 2-site paths share identical accumulator logic, so this
    test exercises the contract; the 2-site path is covered indirectly by
    the benchmark harness.
    """
    gate = _heisenberg_gate()
    out = optimize_gs_ad(gate, None, _fast_config(return_history=True))

    # 1-site path default: (A, env, E) → with flag: (A, env, E, history)
    assert isinstance(out[-1], dict)
    history = out[-1]
    for key in ("energies", "step_times", "jit_compile_time", "num_steps", "converged"):
        assert key in history, f"history missing {key!r}"

    # step_times excludes step 0 (which lands in jit_compile_time), so
    # len(step_times) == len(energies) - 1 when num_steps > 1, or 0 if num_steps == 1.
    assert len(history["energies"]) == history["num_steps"]
    assert len(history["step_times"]) == max(history["num_steps"] - 1, 0)
    assert history["num_steps"] >= 1

    # All energy/time entries are Python floats (the benchmark JSON serializer requires this).
    assert all(isinstance(e, float) for e in history["energies"])
    assert all(isinstance(t, float) for t in history["step_times"])
    assert isinstance(history["jit_compile_time"], float)
    assert isinstance(history["converged"], bool)
    assert isinstance(history["num_steps"], int)


@pytest.mark.algorithm
def test_optimize_gs_ad_default_no_history():
    """Default return shape is unchanged when return_history flag is off."""
    gate = _heisenberg_gate()
    out = optimize_gs_ad(gate, None, _fast_config(return_history=False, num_steps=2))
    # 1-site path: (A, env, E_gs) — last element is a JAX scalar, NOT a dict.
    assert not isinstance(out[-1], dict)
    assert len(out) == 3


def _fast_multisite_config(*, return_history: bool, num_steps: int = 3) -> iPEPSConfig:
    """Minimal-cost multisite (Lattice) config; mirrors _fast_config for the
    1-site path.  Used to validate the #458 return_history extension to
    ``_optimize_gs_ad_multisite``."""
    from tenax.core.lattice import checkerboard

    return iPEPSConfig(
        max_bond_dim=2,
        unit_cell=checkerboard(),
        ctm=CTMConfig(chi=4, max_iter=5),
        gs_num_steps=num_steps,
        gs_learning_rate=1e-2,
        gs_implicit_ad=False,
        gs_explicit_ad_steps=2,
        gs_explicit_ad_warmup=1,
        gs_optimizer="adam",
        return_history=return_history,
    )


@pytest.mark.algorithm
def test_optimize_gs_ad_returns_history_multisite():
    """The multisite dispatcher (#458 follow-up) now surfaces the same
    history dict as the 1-site and 2-site paths.

    Pre-#458 this raised ``NotImplementedError`` before any optimization
    work.  The 4-tuple return shape and dict keys must match the 1-site
    contract so downstream consumers (benchmark harnesses, finite-χ
    scaling tooling) can treat the three paths interchangeably.
    """
    gate = _heisenberg_gate()
    out = optimize_gs_ad(gate, None, _fast_multisite_config(return_history=True))

    # Multisite default: (sites_dict, envs_dict, E_gs) → with flag adds history
    assert len(out) == 4
    assert isinstance(out[-1], dict)
    history = out[-1]
    for key in ("energies", "step_times", "jit_compile_time", "num_steps", "converged"):
        assert key in history, f"history missing {key!r}"

    assert len(history["energies"]) == history["num_steps"]
    assert len(history["step_times"]) == max(history["num_steps"] - 1, 0)
    assert history["num_steps"] >= 1

    assert all(isinstance(e, float) for e in history["energies"])
    assert all(isinstance(t, float) for t in history["step_times"])
    assert isinstance(history["jit_compile_time"], float)
    assert isinstance(history["converged"], bool)
    assert isinstance(history["num_steps"], int)


@pytest.mark.algorithm
def test_optimize_gs_ad_default_no_history_multisite():
    """Default multisite return shape stays a 3-tuple when the flag is off."""
    gate = _heisenberg_gate()
    out = optimize_gs_ad(
        gate, None, _fast_multisite_config(return_history=False, num_steps=2)
    )
    # Multisite: (sites_dict, envs_dict, E_gs).
    assert len(out) == 3
    assert not isinstance(out[-1], dict)
