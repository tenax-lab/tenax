"""The metric preconditioner is invariant to the CTM environment's scale.

``precondition_gradient`` solves ``(N̂ + δI) g' = g`` with the local norm
metric ``N`` of Rader et al. (arXiv:2511.09546, Eq. 11).  ``N`` is the
contracted single-site environment, whose overall scale is a CTM
normalisation *convention*: two exact gauges of one environment
(``forward_gauge="phase"`` vs ``"bond_phase"``) differ in it by ~40%.  Before
the fix ``N`` was inverted raw, so ``δ`` was measured against that arbitrary
scale and the two gauges produced different search directions from
bit-identical energies and gradients.  ``N̂ = N ‖A‖² / ⟨A|N|A⟩`` is the metric
of the normalised state; these tests pin

* invariance of ``g'`` under ``env -> c·env`` (real, complex, multisite) and
  under ``A -> a·A``;
* that the normaliser is the Rayleigh quotient ``⟨A|N|A⟩/⟨A|A⟩`` (not, e.g.,
  ``‖N‖`` -- also scale invariant, but a different δ meaning);
* agreement of ``g'`` across ``phase`` / ``bond_phase`` envs (the symptom);
* the degenerate-norm guard (warn + unpreconditioned gradient, no NaN).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, heisenberg_gate, iPEPSConfig  # noqa: E402
from tenax.algorithms import _metric_precond as mp  # noqa: E402
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    CHECKERBOARD_NEIGHBORS,
)
from tenax.algorithms.ipeps import ipeps  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402
from tenax.core.tensor import DenseTensor, SymmetricTensor  # noqa: E402

CHI = 8
DELTA = 1e-2
SITES = ((0, 0), (1, 0))
# Tight GMRES so a comparison against a direct solve is meaningful; the
# invariance tests hold at any tolerance (same Krylov sequence up to rounding).
CFG = iPEPSConfig(metric_gmres_tol=1e-12, metric_gmres_maxiter=200)


def _scale_env(env, c):
    return env._replace(**{f: getattr(env, f) * c for f in env._fields})


def _envs(sites, gauge):
    envs, info = python_loop_ctm_converge(
        sites,
        CHECKERBOARD_NEIGHBORS,
        **ctm_converge_kwargs(CTMConfig(chi=CHI, forward_gauge=gauge)),
    )
    assert bool(info.converged), gauge  # a converged env, not an oscillating one
    return envs


def _shape(T):
    return tuple(idx.dim for idx in T.indices)


def _like(A, data):
    return DenseTensor(jnp.asarray(data), A.indices)


@pytest.fixture(scope="module")
def su_state():
    """Heisenberg 2-site D=2 simple-update state + envs under both gauges."""
    su = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=50,
        dt=0.05,
        unit_cell="2site",
        ctm=CTMConfig(chi=CHI),
    )
    _, (A, B), _ = ipeps(heisenberg_gate(), None, su, compute_energy=False)
    sites = {(0, 0): A * (1.0 / A.norm()), (1, 0): B * (1.0 / B.norm())}
    rng = np.random.default_rng(7)
    grads = {k: _like(sites[k], rng.normal(size=_shape(sites[k]))) for k in SITES}
    return {
        "sites": sites,
        "grads": grads,
        "phase": _envs(sites, "phase"),
        "bond_phase": _envs(sites, "bond_phase"),
    }


@pytest.fixture(scope="module")
def complex_state(su_state):
    """A genuinely complex state (complex site tensors, complex env)."""
    rng = np.random.default_rng(11)
    sites = {}
    for k in SITES:
        A = su_state["sites"][k]
        data = A.todense() + 0.2j * rng.normal(size=_shape(A))
        data = data / jnp.linalg.norm(data)
        sites[k] = _like(A, data)
    grads = {
        k: _like(
            sites[k],
            rng.normal(size=_shape(sites[k])) + 1j * rng.normal(size=_shape(sites[k])),
        )
        for k in SITES
    }
    envs = _envs(sites, "phase")
    assert jnp.iscomplexobj(envs[(0, 0)].T1.todense())  # the regime under test
    return {"sites": sites, "grads": grads, "envs": envs}


def _psi_norm(A, env):
    """⟨A|N|A⟩ via the double layer, independently of the module's E_mat."""
    from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor

    E = mp._contract_single_site_environment(env)
    return jnp.einsum("ijkl,ijkl->", E, _build_double_layer_tensor(A).todense())


def _assert_close(a, b, rtol):
    err = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    assert err < rtol, err


# ---------------------------------------------------------------------------
# Invariance under env -> c·env and A -> a·A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.1, 0.57, 10.0])
def test_env_rescale_invariance_real(su_state, c):
    k = (0, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    ref = mp.precondition_gradient(A, env, g, DELTA, CFG)
    out = mp.precondition_gradient(A, _scale_env(env, c), g, DELTA, CFG)
    _assert_close(out, ref, 1e-10)


@pytest.mark.parametrize("c", [-1.0, 0.3])
@pytest.mark.parametrize("field", ["C1", "T2"])
def test_single_env_tensor_rescale_invariance(su_state, field, c):
    """CTM normalises each C/T separately; one of them alone may move.

    Scaling *every* tensor by c moves N by c**8, so a sign flip of N is only
    reachable this way.
    """
    k = (0, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    ref = mp.precondition_gradient(A, env, g, DELTA, CFG)
    scaled = env._replace(**{field: getattr(env, field) * c})
    out = mp.precondition_gradient(A, scaled, g, DELTA, CFG)
    _assert_close(out, ref, 1e-10)


@pytest.mark.parametrize("c", [0.1, 0.57, 10.0, complex(np.exp(0.7j))])
def test_env_rescale_invariance_complex(complex_state, c):
    k = (1, 0)
    A, g = complex_state["sites"][k], complex_state["grads"][k]
    env = complex_state["envs"][k]
    ref = mp.precondition_gradient(A, env, g, DELTA, CFG)
    out = mp.precondition_gradient(A, _scale_env(env, c), g, DELTA, CFG)
    assert jnp.iscomplexobj(ref)
    _assert_close(out, ref, 1e-10)


@pytest.mark.parametrize("a", [0.3, 4.0])
def test_site_rescale_invariance(su_state, a):
    """The operator N̂ is independent of ‖A‖ (the env is held fixed)."""
    k = (1, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    ref = mp.precondition_gradient(A, env, g, DELTA, CFG)
    out = mp.precondition_gradient(A * a, env, g, DELTA, CFG)
    _assert_close(out, ref, 1e-10)


def test_multisite_independent_env_rescale(su_state):
    """Each site's env carries its own scale; each is normalised by its own."""
    sites, grads, envs = su_state["sites"], su_state["grads"], su_state["phase"]
    ref = mp.precondition_gradient_multisite(sites, envs, grads, DELTA, CFG)
    scaled = {
        (0, 0): _scale_env(envs[(0, 0)], 0.1),
        (1, 0): _scale_env(envs[(1, 0)], 10.0),
    }
    out = mp.precondition_gradient_multisite(sites, scaled, grads, DELTA, CFG)
    for k in SITES:
        _assert_close(out[k], ref[k], 1e-10)


def test_multisite_complex_env_rescale(complex_state):
    sites, grads, envs = (complex_state[x] for x in ("sites", "grads", "envs"))
    ref = mp.precondition_gradient_multisite(sites, envs, grads, DELTA, CFG)
    scaled = {
        (0, 0): _scale_env(envs[(0, 0)], 0.57),
        (1, 0): _scale_env(envs[(1, 0)], 10.0),
    }
    out = mp.precondition_gradient_multisite(sites, scaled, grads, DELTA, CFG)
    for k in SITES:
        _assert_close(out[k], ref[k], 1e-10)


# ---------------------------------------------------------------------------
# What the normaliser is (and hence what delta means)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["real", "complex"])
def test_normalised_metric_has_unit_rayleigh_quotient(su_state, complex_state, which):
    """⟨A|N̂|A⟩/⟨A|A⟩ == 1, so δ is relative to the metric along the state."""
    if which == "real":
        A, env = su_state["sites"][(0, 0)], su_state["phase"][(0, 0)]
    else:
        A, env = complex_state["sites"][(0, 0)], complex_state["envs"][(0, 0)]
    a = A * 2.5  # not unit norm: the ‖A‖² factor is exercised
    N_hat = mp._normalized_metric_matrix(a, env)
    D, d = _shape(A)[0], _shape(A)[-1]
    a_mat = a.todense().reshape(D**4, d)
    rq = jnp.vdot(a_mat, N_hat @ a_mat) / jnp.vdot(a_mat, a_mat)
    assert abs(complex(rq) - 1.0) < 1e-12, rq


def test_matches_direct_solve_with_psi_norm(su_state):
    """g' solves (N/⟨ψ|ψ⟩ + δ) g' = g, ⟨ψ|ψ⟩ from the double layer."""
    k = (1, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    D, d = _shape(A)[0], _shape(A)[-1]
    n = _psi_norm(A, env)  # ‖A‖ = 1 here
    E_mat = mp._metric_matrix(mp._contract_single_site_environment(env), D)
    M = jnp.kron(E_mat / n, jnp.eye(d)) + DELTA * jnp.eye(D**4 * d)
    expected = jnp.linalg.solve(M, g.todense().reshape(-1)).reshape(_shape(A))
    out = mp.precondition_gradient(A, env, g, DELTA, CFG)
    _assert_close(out, expected, 1e-8)


# ---------------------------------------------------------------------------
# The symptom: phase vs bond_phase envs give the same g'
# ---------------------------------------------------------------------------


def test_gauge_invariance_phase_vs_bond_phase(su_state):
    sites, grads = su_state["sites"], su_state["grads"]
    envs_p, envs_b = su_state["phase"], su_state["bond_phase"]
    for k in SITES:
        # Regime: the two gauges really do carry different env scales (else
        # this test is vacuous), while describing the same environment.
        ratio = float(_psi_norm(sites[k], envs_b[k]) / _psi_norm(sites[k], envs_p[k]))
        assert abs(ratio - 1.0) > 0.1, ratio
    zp = mp.precondition_gradient_multisite(sites, envs_p, grads, DELTA, CFG)
    zb = mp.precondition_gradient_multisite(sites, envs_b, grads, DELTA, CFG)
    for k in SITES:
        _assert_close(zb[k], zp[k], 1e-8)
    # The L-BFGS step-1 direction (empty history) is H_0 g = g'.
    flat = jnp.concatenate([grads[k].todense().reshape(-1) for k in SITES])
    n_A = sites[(0, 0)].todense().size

    def h0(envs):
        def f(v):
            gv = {
                (0, 0): _like(sites[(0, 0)], v[:n_A].reshape(_shape(sites[(0, 0)]))),
                (1, 0): _like(sites[(1, 0)], v[n_A:].reshape(_shape(sites[(1, 0)]))),
            }
            z = mp.precondition_gradient_multisite(sites, envs, gv, DELTA, CFG)
            return jnp.concatenate([z[k].reshape(-1) for k in SITES])

        return f

    dp = mp.lbfgs_two_loop(flat, [], h0(envs_p))
    db = mp.lbfgs_two_loop(flat, [], h0(envs_b))
    _assert_close(db, dp, 1e-8)


# ---------------------------------------------------------------------------
# Block-sparse input
# ---------------------------------------------------------------------------


def test_symmetric_tensor_site_matches_dense(su_state):
    k = (0, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    A_sym = SymmetricTensor.from_dense(A.todense(), A.indices)
    g_sym = SymmetricTensor.from_dense(g.todense(), g.indices)
    ref = mp.precondition_gradient(A, env, g, DELTA, CFG)
    out = mp.precondition_gradient(A_sym, env, g_sym, DELTA, CFG)
    _assert_close(out, ref, 1e-12)


# ---------------------------------------------------------------------------
# Degenerate norm guard
# ---------------------------------------------------------------------------


def _degenerate_cases(su_state):
    k = (0, 0)
    A, env = su_state["sites"][k], su_state["phase"][k]
    nan_env = env._replace(C1=env.C1 * jnp.nan)
    return {
        "zero_env": (A, _scale_env(env, 0.0)),
        "nan_env": (A, nan_env),
        "zero_site": (A * 0.0, env),
    }


@pytest.mark.parametrize("case", ["zero_env", "nan_env", "zero_site"])
def test_degenerate_norm_returns_unpreconditioned_gradient(su_state, case):
    A, env = _degenerate_cases(su_state)[case]
    g = su_state["grads"][(0, 0)]
    with pytest.warns(RuntimeWarning, match="degenerate state norm"):
        out = mp.precondition_gradient(A, env, g, DELTA, CFG)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(g.todense()))


def test_degenerate_norm_multisite_is_per_site(su_state):
    """One degenerate site falls back alone; the other is still preconditioned."""
    sites, grads, envs = su_state["sites"], su_state["grads"], su_state["phase"]
    ref = mp.precondition_gradient_multisite(sites, envs, grads, DELTA, CFG)
    bad = {(0, 0): _scale_env(envs[(0, 0)], 0.0), (1, 0): envs[(1, 0)]}
    with pytest.warns(RuntimeWarning, match="degenerate state norm"):
        out = mp.precondition_gradient_multisite(sites, bad, grads, DELTA, CFG)
    np.testing.assert_array_equal(
        np.asarray(out[(0, 0)]), np.asarray(grads[(0, 0)].todense())
    )
    _assert_close(out[(1, 0)], ref[(1, 0)], 1e-12)


def test_healthy_norm_does_not_warn(su_state):
    k = (0, 0)
    A, g, env = su_state["sites"][k], su_state["grads"][k], su_state["phase"][k]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = mp.precondition_gradient(A, env, g, DELTA, CFG)
    assert bool(jnp.all(jnp.isfinite(out)))
