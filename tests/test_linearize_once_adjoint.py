"""Linearise once, jit only the transpose, for the implicit-AD adjoint.

The default backward (``adjoint_chain_rule="joint"``) builds the gauge-fixed
sweep's pullback once (``_jit_linearize_joint``) and hands it to one joint
adjoint solve (``_jit_joint_adjoint_solve``): the Neumann loop and GMRES as
phases of one program, with one copy of the transpose.  The oracle is
``"separate"``, whose programs are unchanged.  Specs:
docs/superpowers/specs/2026-10-25-linearize-once-adjoint-design.md,
docs/superpowers/specs/2026-10-25-single-transpose-adjoint-design.md
"""

from __future__ import annotations

import collections
import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _block_sparse_jit
from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

# ----------------------------------------------------------- _neumann_loop


def _contractive(rho, n=6, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    M *= rho / np.max(np.abs(np.linalg.eigvals(M)))
    b = rng.standard_normal(n)
    return M, b


def _loop(M, b, init=None, use_init=False, tol=1e-13, maxiter=500):
    J = jnp.asarray(M)
    init = jnp.zeros(len(b)) if init is None else jnp.asarray(init)
    return cea._neumann_loop(
        lambda v: (J.T @ v[0],),
        (jnp.asarray(b),),
        (init,),
        jnp.array(use_init),
        tol=tol,
        maxiter=maxiter,
    )


def test_the_neumann_loop_solves_a_contractive_system():
    M, b = _contractive(0.5)
    diverged, converged, n_iter, lam, abs_resid, b_norm = _loop(M, b)
    assert bool(converged) and not bool(diverged)
    assert int(n_iter) > 5  # regime: it iterated
    np.testing.assert_allclose(
        np.asarray(lam[0]), np.linalg.solve(np.eye(6) - M.T, b), rtol=1e-10
    )
    np.testing.assert_allclose(float(b_norm), np.linalg.norm(b), rtol=1e-14)
    assert float(abs_resid) < 1e-12


def test_the_neumann_loop_flags_divergence():
    M, b = _contractive(1.5)
    diverged, converged, n_iter, *_ = _loop(M, b)
    assert bool(diverged) and not bool(converged)
    assert int(n_iter) <= 10


def test_the_warm_start_is_used_only_when_asked():
    M, b = _contractive(0.5)
    exact = np.linalg.solve(np.eye(6) - M.T, b)
    n_warm = int(_loop(M, b, init=exact, use_init=True)[2])
    n_cold = int(_loop(M, b, init=exact, use_init=False)[2])
    assert n_warm == 1
    assert n_cold > 5


# --------------------------------------------------------- _fusion_variant

needs_no_fusion = pytest.mark.skipif(
    jax.default_backend() != "cpu"
    or not _block_sparse_jit.JIT_TAKES_COMPILER_OPTIONS
    or not _block_sparse_jit.no_fusion_compiles(),
    reason="the no-fusion variant is CPU-only and needs compiler_options",
)


@pytest.fixture
def any_size(monkeypatch):
    """Drop the size threshold, so a small U(1) state dispatches as a large one."""
    monkeypatch.setattr(_block_sparse_jit, "NO_FUSION_MIN_SITE_BLOCKS", 1)


@needs_no_fusion
@pytest.mark.parametrize("by_bond_dim", [True, False])
def test_a_large_symmetric_site_picks_the_no_fusion_variant(any_size, by_bond_dim):
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(prog, (A, B), by_bond_dim=by_bond_dim)
    assert got is prog.block_sparse


@needs_no_fusion
def test_dense_sites_pick_the_default_variant(any_size):
    prog = _block_sparse_jit.block_sparse_jit(lambda x: x)
    got = cea._fusion_variant(
        prog, (_wrap_as_dense_tensor(jnp.ones((2, 2, 2, 2, 2))),), by_bond_dim=True
    )
    assert got is prog.dense


# ------------------------------------------------------- the default backward

#: One gate object: the gate fingerprint is part of the dispatch cache key.
_GATE = heisenberg_gate()

#: tests/test_joint_chain_rule.py's fixture: at chi=4 the fused loop solves
#: seed 0.
_FUSED_SEED = 0

_SOLVERS = ("_jit_joint_adjoint_solve",)
_NEW_PROGRAMS = ("_jit_linearize_joint", *_SOLVERS)


@pytest.fixture
def fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.fixture
def spy(monkeypatch):
    """Record every program the backward calls through ``_fusion_variant``:
    ``(name, "block_sparse" | "dense", variant, args, kwargs)``."""
    calls = []
    real = cea._fusion_variant

    def spying(prog, params_data_tuple, *, by_bond_dim):
        variant = real(prog, params_data_tuple, by_bond_dim=by_bond_dim)
        kind = "block_sparse" if variant is prog.block_sparse else "dense"

        def call(*args, **kwargs):
            calls.append((prog.__name__, kind, variant, args, kwargs))
            return variant(*args, **kwargs)

        return call

    monkeypatch.setattr(cea, "_fusion_variant", spying)
    return calls


def _site(seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((2, 2, 2, 2, 2))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A0 = A0 + 0.1j * rng.standard_normal((2, 2, 2, 2, 2))
    A0 = jnp.asarray(A0, dtype=dtype)
    return A0 / jnp.linalg.norm(A0)


def _energy(arrs, mode, *, method="fixed_point", tol=1e-11, neighbors=None, **kw):
    if not isinstance(arrs, (tuple, list)):
        arrs = (arrs,)
    neighbors = neighbors or SINGLE_SITE_NEIGHBORS
    coords = sorted(neighbors)
    return jnp.real(
        ctm_energy_implicit(
            {c: _wrap_as_dense_tensor(a) for c, a in zip(coords, arrs)},
            neighbors,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=method,
            gmres_tol=tol,
            adjoint_chain_rule=mode,
            **kw,
        )
    )


def _grad(arrs, mode, **kw):
    return jax.grad(lambda a: _energy(a, mode, **kw))(arrs)


_SVD = re.compile(r"\bsvd\[")


def _svd_count(call):
    _name, _kind, variant, args, kwargs = call
    jaxpr = jax.make_jaxpr(lambda *a: variant(*a, **kwargs))(*args)
    return len(_SVD.findall(str(jaxpr)))


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_adjoint_solvers_hold_no_forward_sweep(spy):
    """The forward projector SVDs live in the linearise program only.  A
    solver that re-linearised the sweep (main's GMRES matvec did) would carry
    every one of them again."""
    _grad(_site(_FUSED_SEED), "joint")
    by_name = {c[0]: c for c in spy}
    assert set(_NEW_PROGRAMS) <= set(by_name), sorted(by_name)
    forward = _svd_count(by_name["_jit_linearize_joint"])
    assert forward > 0  # regime: the forward sweep has SVDs to leak
    for name in _SOLVERS:
        assert _svd_count(by_name[name]) == 0, name


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_one_backward_linearises_once(spy):
    _grad(_site(_FUSED_SEED), "joint")
    names = [c[0] for c in spy]
    assert names.count("_jit_linearize_joint") == 1, names
    assert names.count("_jit_joint_adjoint_solve") == 1, names


from tests.test_neumann_then_gmres import _assert_same, _oracle  # noqa: E402


def _dot_count(fn, *args):
    """dot_general sites, counting every occurrence of a repeated sub-jaxpr.

    The jaxpr printer names a repeated identical sub-jaxpr once and refers
    to it afterwards, so a text count misses a second copy of a pullback.
    """
    count = 0

    def walk(jaxpr):
        nonlocal count
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                count += 1
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None and hasattr(inner, "eqns"):
                        walk(inner)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return count


def _joint_solve_call(spy):
    (call,) = [c for c in spy if c[0] == "_jit_joint_adjoint_solve"]
    return call


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_joint_solve_holds_one_transpose(spy):
    """Neumann and GMRES share one matvec call site, so the program's
    contractions are one pullback's plus the Krylov bookkeeping."""
    _grad(_site(_FUSED_SEED), "joint")
    _name, _kind, variant, args, kwargs = _joint_solve_call(spy)
    vjp_joint, rhs = args[0], args[1]
    one = _dot_count(lambda f, v: f(tuple(v)), vjp_joint, rhs)
    assert one > 16  # regime: a second copy would stand out
    assert _dot_count(lambda *a: variant(*a, **kwargs), *args) <= one + 16


def _check_against_oracle(call):
    _name, _kind, variant, args, kwargs = call
    got = variant(*args, **kwargs)
    kw = {k: kwargs[k] for k in ("tol", "maxiter", "restart")}

    def oracle(f, b, i, u):
        def jt_aux(v):
            params_ct, env_ct = f(tuple(v))
            return env_ct, params_ct

        return _oracle(jt_aux, b, i, u, **kw)

    want = jax.jit(oracle)(*args)
    _assert_same(got, want)
    return got


@pytest.mark.usefixtures("fresh_vjp_cache")
@pytest.mark.parametrize("case", ["cold", "warm", "unsolved", "complex"])
def test_the_joint_solve_matches_the_two_solvers_composed(spy, case):
    dtype = jnp.complex128 if case == "complex" else jnp.float64
    A0 = _site(_FUSED_SEED, dtype=dtype)
    kw = dict(gmres_maxiter=3) if case == "unsolved" else {}
    _grad(A0, "joint", **kw)
    if case == "warm":
        spy.clear()
        _grad(A0 * (1.0 + 1e-3), "joint")
    call = _joint_solve_call(spy)
    assert call[4]["neumann"] is True
    if case == "warm":
        assert bool(call[3][3])  # regime: the warm start is used
    got = _check_against_oracle(call)
    if case == "unsolved":
        assert not bool(got[1])  # regime: the Neumann phase did not solve
    else:
        assert bool(got[1]) and not bool(got[0])


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_u1_joint_solve_matches_the_two_solvers_composed(spy):
    from tests.test_block_sparse_ctm_ad import _make_u1_symmetric_ipeps

    A_sym = _make_u1_symmetric_ipeps()
    jax.grad(
        lambda A: jnp.real(
            ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _GATE,
                chi=4,
                max_iter=30,
                conv_tol=1e-8,
                gmres_tol=1e-11,
                adjoint_chain_rule="joint",
            )
        )
    )(A_sym)
    got = _check_against_oracle(_joint_solve_call(spy))
    assert int(got[2]) > 0  # regime: the Neumann phase ran


@pytest.fixture
def rule_spy(monkeypatch):
    """``{program name: by_bond_dim}`` for each ``_fusion_variant`` call."""
    rules = {}
    real = cea._fusion_variant

    def spying(prog, params_data_tuple, *, by_bond_dim):
        rules[prog.__name__] = by_bond_dim
        return real(prog, params_data_tuple, by_bond_dim=by_bond_dim)

    monkeypatch.setattr(cea, "_fusion_variant", spying)
    return rules


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_joint_solve_takes_the_joint_fusion_rule(rule_spy):
    """It holds the params VJP, so the joint programs' rule applies."""
    _grad(_site(_FUSED_SEED), "joint")
    assert rule_spy == {
        "_jit_linearize_joint": False,
        "_jit_joint_adjoint_solve": True,
    }, rule_spy


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_later_backwards_on_one_layout_trace_nothing_new():
    A0 = _site(_FUSED_SEED)
    mark0 = len(_TRACED)
    _grad(A0, "joint")
    assert set(_NEW_PROGRAMS) <= set(_TRACED[mark0:]), sorted(set(_TRACED[mark0:]))
    mark = len(_TRACED)
    for k in (1, 2):
        _grad(A0 * (1.0 + 1e-3 * k), "joint")
    retraced = set(_TRACED[mark:]) & set(_NEW_PROGRAMS)
    assert not retraced, sorted(retraced)


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_second_backward_with_new_values_matches_separate():
    """A pullback reused from the first backward would give the first
    state's gradient."""
    A0 = _site(_FUSED_SEED)
    _grad(A0, "joint")
    A1 = 0.9 * A0 + 0.1 * _site(5)
    A1 = A1 / jnp.linalg.norm(A1)
    g_joint = _grad(A1, "joint")
    g_sep = _grad(A1, "separate")
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(dtype=jnp.complex128),
        dict(projector_backward="flow", forward_gauge="bond_phase"),
    ],
    ids=["complex", "flow"],
)
def test_the_fixed_point_path_matches_separate(kw):
    """Review Focus 1 and 3 on the default (fixed-point) method; the gmres
    method's cases live in tests/test_joint_chain_rule.py."""
    kw = dict(kw)
    dtype = kw.pop("dtype", jnp.float64)
    A0 = _site(_FUSED_SEED, dtype=dtype)
    g_sep = _grad(A0, "separate", **kw)
    g_joint = _grad(A0, "joint", **kw)
    if "projector_backward" in kw:  # regime: the flowing backward really ran
        used = cea.get_last_implicit_ad_diagnostics().get("projector_backward_used")
        assert used == "flow", used
    np.testing.assert_allclose(
        np.asarray(g_joint), np.asarray(g_sep), rtol=1e-10, atol=1e-12
    )


@needs_no_fusion
@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_solvers_take_the_variant_the_site_tensors_call_for(spy, any_size):
    from tests.test_block_sparse_ctm_ad import _make_u1_symmetric_ipeps

    A_sym = _make_u1_symmetric_ipeps()
    jax.grad(
        lambda A: jnp.real(
            ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _GATE,
                chi=4,
                max_iter=30,
                conv_tol=1e-8,
                gmres_tol=1e-11,
                adjoint_chain_rule="joint",
            )
        )
    )(A_sym)
    kinds = {c[0]: c[1] for c in spy}
    assert kinds == dict.fromkeys(_NEW_PROGRAMS, "block_sparse"), kinds


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_dense_sites_keep_the_default_variant(spy):
    _grad(_site(_FUSED_SEED), "joint")
    kinds = {c[0]: c[1] for c in spy}
    assert kinds == dict.fromkeys(_NEW_PROGRAMS, "dense"), kinds


# ------------------------------------------------------------ SU states (slow)


def _assert_tree_close(a, b, rtol=1e-10, atol=1e-12):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb) and la
    for x, y in zip(la, lb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


@pytest.mark.slow
def test_a_u1_su_state_matches_separate():
    from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, ipeps, xxz_gate
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    gate = xxz_gate(0.3)
    A0, B0 = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    cfg = iPEPSConfig(
        max_bond_dim=2, num_imaginary_steps=100, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(gate, (A0, B0), cfg, compute_energy=False)
    A, B = A * (1.0 / A.norm()), B * (1.0 / B.norm())

    def grad(mode):
        return jax.grad(
            lambda p: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): p[0], (1, 0): p[1]},
                    CHECKERBOARD_NEIGHBORS,
                    gate,
                    chi=8,
                    max_iter=200,
                    conv_tol=1e-10,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )((A, B))

    _assert_tree_close(grad("joint"), grad("separate"))


@pytest.mark.slow
def test_a_fermionic_su_state_matches_separate():
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _fpeps_simple_update,
        _initialize_fpeps,
        spinless_fermion_gate,
    )

    cfg = FPEPSConfig(D=2, t=1.0, V=1.0, mu=2.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(0))
    B = _initialize_fpeps(cfg, jax.random.PRNGKey(1))
    A, B, _ = _fpeps_simple_update(A, H, max_D=2, dt=0.05, steps=100, B=B)

    def grad(mode):
        return jax.grad(
            lambda p: jnp.real(
                ctm_energy_implicit(
                    {(0, 0): p[0], (1, 0): p[1]},
                    CHECKERBOARD_NEIGHBORS,
                    H,
                    chi=8,
                    max_iter=200,
                    conv_tol=1e-10,
                    gmres_tol=1e-11,
                    adjoint_chain_rule=mode,
                )
            )
        )((A, B))

    _assert_tree_close(grad("joint"), grad("separate"))


@pytest.mark.slow
@pytest.mark.usefixtures("fresh_vjp_cache")
def test_a_fermionic_joint_solve_matches_the_two_solvers_composed(spy):
    from tenax.algorithms.fermionic_ipeps import (
        FPEPSConfig,
        _fpeps_simple_update,
        _initialize_fpeps,
        spinless_fermion_gate,
    )

    cfg = FPEPSConfig(D=2, t=1.0, V=1.0, mu=2.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A = _initialize_fpeps(cfg, jax.random.PRNGKey(0))
    B = _initialize_fpeps(cfg, jax.random.PRNGKey(1))
    A, B, _ = _fpeps_simple_update(A, H, max_D=2, dt=0.05, steps=100, B=B)
    jax.grad(
        lambda p: jnp.real(
            ctm_energy_implicit(
                {(0, 0): p[0], (1, 0): p[1]},
                CHECKERBOARD_NEIGHBORS,
                H,
                chi=8,
                max_iter=200,
                conv_tol=1e-10,
                gmres_tol=1e-11,
                adjoint_chain_rule="joint",
            )
        )
    )((A, B))
    got = _check_against_oracle(_joint_solve_call(spy))
    assert int(got[2]) > 0  # regime: the Neumann phase ran
