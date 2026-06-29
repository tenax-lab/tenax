import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_config import CTMConfig

pytestmark = pytest.mark.core


def test_fuse_virtual_legs_defaults_true():
    cfg = CTMConfig(chi=8)
    assert cfg.fuse_virtual_legs is True


def test_fuse_virtual_legs_can_disable():
    cfg = CTMConfig(chi=8, fuse_virtual_legs=False)
    assert cfg.fuse_virtual_legs is False


from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_site(D, d, seed):
    """Single-site U(1)-trivial DenseTensor with labels (u,d,l,r,phys)."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(d, dtype=np.int32)
    idx = [
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, zd.copy(), FlowDirection.IN, label="phys"),
    ]
    return DenseTensor(data, idx)


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _split_explicit_loss(D):
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_explicit

    chi = D * D  # lossless: rank of D×D corner ≤ D²
    A = _make_site(D, 2, seed=7)
    gate = _heisenberg_gate()

    def loss(a):
        return ctm_energy_split_explicit(
            {(0, 0): a},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            warmup_steps=2,
            backprop_steps=3,
            chi_I=chi,
        ).real

    return loss, A


@pytest.mark.parametrize("D", [2, 3])
def test_split_explicit_energy_finite(D):
    """ctm_energy_split_explicit returns a finite energy (forward correctness).

    The double-layer corner-pair forward (#463 doublelayer) gives the
    physically-correct split energy; this checks the explicit-AD entry point's
    *forward* value is finite. Gradient finiteness is a separate, currently
    deferred concern — see ``test_split_explicit_grad_finite``.
    """
    loss, A = _split_explicit_loss(D)
    e = loss(A)
    assert jnp.isfinite(e), f"energy is not finite: {e}"


@pytest.mark.parametrize("D", [2, 3])
def test_split_explicit_grad_finite(D):
    """ctm_energy_split_explicit returns a finite, non-zero gradient.

    The degenerate 1-site double-layer corner (spectrum ``[0.5, 0.5, 0, 0]``)
    is AD-stable now that the production bounded edge path applies the
    biorthogonal projector pair directly instead of routing through the
    vestigial ``_factorize_projector`` SVD round-trip, whose ``sqrt(s)`` /
    rank-deficient-SVD backward divided by the projector's zero singular
    values (#463 split backward AD-stability).
    """
    loss, A = _split_explicit_loss(D)
    _, g = jax.value_and_grad(loss)(A)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.all(jnp.isfinite(gs)), "gradient contains non-finite values"
    assert float(jnp.sum(jnp.abs(gs))) > 0, "gradient is identically zero"


def test_split_explicit_raises_for_multisite():
    """ctm_energy_split_explicit raises NotImplementedError for >1 site."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_explicit

    A = _make_site(2, 2, seed=0)
    gate = _heisenberg_gate()

    with pytest.raises(NotImplementedError, match="single-site"):
        ctm_energy_split_explicit(
            {(0, 0): A, (1, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
        )


# --------------------------------------------------------------------------- #
# Implicit (fixed-point) split-CTM AD (#463 Phase 2, Task 3)                   #
# --------------------------------------------------------------------------- #


def _split_implicit_loss(D, seed=7, chi=None):
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit

    if chi is None:
        chi = D * D  # lossless: rank of D×D corner ≤ D²
    A = _make_site(D, 2, seed=seed)
    gate = _heisenberg_gate()

    def loss(a):
        return ctm_energy_split_implicit(
            {(0, 0): a},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=60,
            conv_tol=1e-12,
            min_iter=2,
        ).real

    return loss, A


@pytest.mark.parametrize("D", [2, 3])
def test_split_implicit_energy_finite(D):
    """ctm_energy_split_implicit returns a finite energy (forward correctness)."""
    loss, A = _split_implicit_loss(D)
    e = loss(A)
    assert jnp.isfinite(e), f"energy is not finite: {e}"


@pytest.mark.parametrize("D", [2, 3])
def test_split_implicit_grad_finite(D):
    """ctm_energy_split_implicit returns a finite, non-zero gradient."""
    loss, A = _split_implicit_loss(D)
    _, g = jax.value_and_grad(loss)(A)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.all(jnp.isfinite(gs)), "gradient contains non-finite values"
    assert float(jnp.sum(jnp.abs(gs))) > 0, "gradient is identically zero"


def test_split_implicit_grad_matches_explicit():
    """Implicit (Neumann) gradient matches the trusted explicit-AD gradient.

    Both paths differentiate the same gauge-invariant split-CTM energy at the
    same lossless ``chi_I=chi`` fixed point, so the implicit fixed-point
    gradient and the explicit unrolled gradient must agree once the explicit
    reference is itself converged (#463 Phase 2 validation — implicit==explicit,
    not implicit==FD: the split energy_fn carries a pre-existing Wirtinger gap
    that AD-vs-FD inherits and explicit shares).
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_implicit,
    )

    D, seed = 2, 11  # well-converging lossless case
    chi = D * D
    A = _make_site(D, 2, seed=seed)
    gate = _heisenberg_gate()

    def loss_imp(a):
        return ctm_energy_split_implicit(
            {(0, 0): a},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            max_iter=80,
            conv_tol=1e-13,
            min_iter=2,
        ).real

    def loss_exp(a):
        return ctm_energy_split_explicit(
            {(0, 0): a},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            chi_I=chi,
            warmup_steps=40,
            backprop_steps=40,
        ).real

    e_i, g_i = jax.value_and_grad(loss_imp)(A)
    e_e, g_e = jax.value_and_grad(loss_exp)(A)
    gi = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_i)])
    ge = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_e)])

    assert jnp.allclose(e_i, e_e, atol=1e-9), f"energy mismatch: {e_i} vs {e_e}"
    cos = float(
        jnp.real(jnp.vdot(gi, ge)) / (jnp.linalg.norm(gi) * jnp.linalg.norm(ge))
    )
    rel = float(jnp.linalg.norm(gi - ge) / jnp.linalg.norm(ge))
    assert cos > 1 - 1e-9, f"gradient direction mismatch: cos={cos}"
    assert rel < 1e-6, f"gradient magnitude mismatch: rel={rel}"


def test_split_implicit_warm_start_matches_cold():
    """env_init warm-start yields the identical energy + gradient as cold start.

    The Γ-gauge-fixed split fixed point is seed-independent, so warm-starting
    the forward — even from a *different* tensor's converged env — must not
    change the energy or the implicit gradient; it only speeds convergence.
    The custom_vjp returns a zero cotangent for env_init.
    """
    from tenax.algorithms._split_ctm_energy_ad import (
        converge_split_env,
        ctm_energy_split_implicit,
    )

    D, seed = 2, 11
    chi = D * D
    A = _make_site(D, 2, seed=seed)
    A_seed = _make_site(D, 2, seed=5)
    gate = _heisenberg_gate()
    kw = dict(chi=chi, chi_I=chi, max_iter=80, conv_tol=1e-12, min_iter=2)
    warm_env = converge_split_env(A_seed, **kw)

    def cold(a):
        return ctm_energy_split_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate, **kw
        ).real

    def warm(a):
        return ctm_energy_split_implicit(
            {(0, 0): a}, SINGLE_SITE_NEIGHBORS, gate, env_init=warm_env, **kw
        ).real

    e_c, g_c = jax.value_and_grad(cold)(A)
    e_w, g_w = jax.value_and_grad(warm)(A)
    gc = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_c)])
    gw = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g_w)])

    assert jnp.allclose(e_c, e_w, atol=1e-10), f"energy: cold={e_c} warm={e_w}"
    assert float(jnp.linalg.norm(gc - gw) / jnp.linalg.norm(gc)) < 1e-8


def test_make_ctm_energy_fn_dispatches_to_split_implicit():
    """make_ctm_energy_fn routes to the split path when fuse_virtual_legs=False.

    With ``fuse_virtual_legs=False`` + ``recipe='1x1'`` the implicit branch
    must produce the same energy as a direct ``ctm_energy_split_implicit``
    call, and a finite gradient through the dispatch closure.
    """
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn

    D = 2
    chi = D * D
    A = _make_site(D, 2, seed=7)
    gate = _heisenberg_gate()
    cfg = CTMConfig(
        chi=chi,
        chi_I=chi,
        fuse_virtual_legs=False,
        max_iter=60,
        conv_tol=1e-12,
        min_iter=2,
    )

    fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=2,
        explicit_steps=3,
        recipe="1x1",
    )
    e, g = jax.value_and_grad(lambda a: fn({(0, 0): a}).real)(A)
    e_direct = ctm_energy_split_implicit(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=chi,
        chi_I=chi,
        max_iter=60,
        conv_tol=1e-12,
        min_iter=2,
    ).real

    assert jnp.allclose(e, e_direct, atol=1e-10)
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.all(jnp.isfinite(gs)) and float(jnp.sum(jnp.abs(gs))) > 0


def test_make_ctm_energy_fn_split_rejects_non_1x1_recipe():
    """The split dispatch guards against unsupported recipes."""
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn

    A = _make_site(2, 2, seed=0)
    gate = _heisenberg_gate()
    cfg = CTMConfig(chi=4, fuse_virtual_legs=False)

    fn = make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=2,
        explicit_steps=3,
        recipe="2x2",
    )
    with pytest.raises(NotImplementedError, match="recipe"):
        fn({(0, 0): A})


def test_split_implicit_raises_for_multisite():
    """ctm_energy_split_implicit raises NotImplementedError for >1 site."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit

    A = _make_site(2, 2, seed=0)
    gate = _heisenberg_gate()

    with pytest.raises(NotImplementedError, match="single-site"):
        ctm_energy_split_implicit(
            {(0, 0): A, (1, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
        )


# --------------------------------------------------------------------------- #
# End-to-end optimizer integration (#463 Phase 2, Task 4)                      #
# --------------------------------------------------------------------------- #


def _split_opt_config(fuse, **overrides):
    from tenax.algorithms.ipeps_config import CTMConfig as _CTMConfig
    from tenax.algorithms.ipeps_config import iPEPSConfig

    D = 2
    chi = D * D
    ctm_kw = dict(
        chi=chi,
        chi_I=chi,
        fuse_virtual_legs=fuse,
        max_iter=40,
        conv_tol=1e-10,
        min_iter=2,
    )
    ctm_kw.update(overrides.pop("ctm", {}))
    cfg_kw = dict(
        unit_cell="1x1",
        gs_recipe="1x1",
        gs_implicit_ad=True,
        gs_num_steps=2,
        gs_log_interval=1,
        su_init=False,
    )
    cfg_kw.update(overrides)
    return iPEPSConfig(ctm=_CTMConfig(**ctm_kw), **cfg_kw)


def _split_opt_inputs():
    A = jax.random.normal(jax.random.PRNGKey(3), (2, 2, 2, 2, 2))
    A = A / jnp.linalg.norm(A)
    return A, _heisenberg_gate()


def test_optimize_gs_ad_split_returns_split_env():
    """fuse_virtual_legs=False drives the whole 1-site optimizer through split.

    The returned environment must be a ``SplitCTMTensorEnv`` (not the fused
    ``CTMTensorEnv``), confirming the warm-start / line-search probe /
    final-env eval all run the split χ²·D⁴ forward — not just the gradient.
    """
    from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    _, env, E = optimize_gs_ad(gate, A, _split_opt_config(fuse=False))
    assert isinstance(env, SplitCTMTensorEnv)
    assert jnp.isfinite(E)


def test_optimize_gs_ad_fused_still_returns_fused_env():
    """Regression: the default fused path is unchanged (returns CTMTensorEnv)."""
    from tenax.algorithms._ctm_tensor import CTMTensorEnv
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    _, env, E = optimize_gs_ad(gate, A, _split_opt_config(fuse=True))
    assert isinstance(env, CTMTensorEnv)
    assert jnp.isfinite(E)


def test_optimize_gs_ad_split_rejects_chi_auto_bump():
    """The split optimizer path rejects fused-env-coupled accelerators."""
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    cfg = _split_opt_config(fuse=False, ctm={"chi_auto_bump": True, "chi_max": 6})
    with pytest.raises(NotImplementedError, match="auto-bump"):
        optimize_gs_ad(gate, A, cfg)


def test_optimize_gs_ad_split_rejects_non_1x1_recipe():
    """The split optimizer path requires gs_recipe='1x1'."""
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    cfg = _split_opt_config(fuse=False, gs_recipe="2x2")
    with pytest.raises(NotImplementedError, match="1x1"):
        optimize_gs_ad(gate, A, cfg)


def test_optimize_gs_ad_split_rejects_chi_schedule():
    """Split path rejects gs_chi_schedule_steps up front.

    Without the guard the schedule advances into _apply_chi_bump ->
    pad_dense_env_chi, which assumes a fused CTMTensorEnv and would
    AttributeError on the cached SplitCTMTensorEnv (Codex review on #651).
    """
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    cfg = _split_opt_config(fuse=False, gs_chi_schedule_steps=[(4, 1)])
    with pytest.raises(NotImplementedError, match="schedule"):
        optimize_gs_ad(gate, A, cfg)


def test_optimize_gs_ad_split_cg_metric_does_not_crash():
    """Split + CG + gs_metric_precond falls back to plain CG (no crash).

    The metric inner product needs the fused CTMTensorEnv; on the split path
    the cache holds a SplitCTMTensorEnv, so metric preconditioning must be
    disabled for the CG optimizer too — not just L-BFGS (Codex review on #651).
    """
    from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, gate = _split_opt_inputs()
    cfg = _split_opt_config(fuse=False, gs_optimizer="cg", gs_metric_precond=True)
    _, env, E = optimize_gs_ad(gate, A, cfg)
    assert isinstance(env, SplitCTMTensorEnv)
    assert jnp.isfinite(E)
