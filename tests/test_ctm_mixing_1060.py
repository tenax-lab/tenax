"""#1060: linear mixing in the CTM forward loop, and the signed step multiplier.

A production spinless t-V D=3 state at chi=20 sat in an exact two-state cycle
(``|e_n - e_{n-2}| = 5e-13`` while ``|e_n - e_{n-1}| = 3.9e-2`` for 600 sweeps)
around a fixed point the plain iteration cannot reach; mixing 0.3 converged it
to 6e-9.  The loop is exercised here on toy linear maps
``F(x) = x* + lam (x - x*)``, whose multiplier is known exactly, so the tests
pin the mechanism without a CTM: mixing turns ``lam`` into
``(1 - beta) lam + beta``, convergence is certified on the *undamped*
residual, and ``step_multiplier`` recovers ``lam`` with its sign.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_loop_core import _run_ctm_loop_with_bump

jax.config.update("jax_enable_x64", True)

COORDS = [(0, 0), (1, 0)]
X_STAR = {
    (0, 0): {
        "C": jnp.array([0.3, -1.2, 0.7]),
        "T": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    },
    (1, 0): {
        "C": jnp.array([2.0, 0.1, -0.4]),
        "T": jnp.array([[0.5, -0.5], [1.5, 0.0]]),
    },
}


def _linear_step(lam):
    def step(site_tensors, envs, **_):
        new = {
            c: jax.tree.map(lambda x, s: s + lam * (x - s), envs[c], X_STAR[c])
            for c in envs
        }
        return new, jnp.asarray(0.0), jnp.asarray(0.0)

    return step


def _start():
    return {
        c: jax.tree.map(lambda s: s + 0.01 * jnp.ones_like(s), X_STAR[c])
        for c in COORDS
    }


def _identity_gauge(envs_new, _envs_old):
    return envs_new


def _run(
    lam,
    *,
    mixing=0.0,
    max_iter=200,
    conv_tol=1e-10,
    gauge=_identity_gauge,
    conv_method="elementwise",
):
    return _run_ctm_loop_with_bump(
        _linear_step(lam),
        {},
        _start(),
        chi_current=4,
        chi_max=None,
        bump_enabled=False,
        bump_threshold=1e-6,
        bump_step_size=2,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        gauge_fix_fn=gauge,
        max_iter=max_iter,
        min_iter=0,
        conv_tol=conv_tol,
        conv_method=conv_method,
        plateau_patience=None,
        mixing=mixing,
    )


def _undamped_residual(lam, envs):
    out, _, _ = _linear_step(lam)({}, envs)
    return max(
        float(jnp.max(jnp.abs(a - b)))
        for c in envs
        for a, b in zip(jax.tree.leaves(out[c]), jax.tree.leaves(envs[c]), strict=True)
    )


def test_mixing_converges_a_flip_unstable_fixed_point():
    """lam = -1.5 diverges plainly; beta = 0.5 gives -0.25 and converges to x*."""
    plain = _run(-1.5, mixing=0.0, max_iter=60)
    assert not plain.converged

    mixed = _run(-1.5, mixing=0.5)
    assert mixed.converged
    for c in COORDS:
        for a, b in zip(
            jax.tree.leaves(mixed.envs[c]), jax.tree.leaves(X_STAR[c]), strict=True
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)


@pytest.mark.parametrize("mixing", [0.0, 0.5, 0.9])
def test_convergence_is_certified_on_the_undamped_residual(mixing):
    """A converged result is a fixed point of the plain map, at any mixing.

    With mixing 0.9 consecutive *mixed* iterates differ by only a tenth of
    the undamped residual, so a loop that tested the mixed difference would
    certify an environment ten times further from stationary than conv_tol.
    """
    lam, tol = 0.5, 1e-8
    res = _run(lam, mixing=mixing, conv_tol=tol, max_iter=2000)
    assert res.converged
    assert _undamped_residual(lam, res.envs) < tol


@pytest.mark.parametrize("lam", [-1.5, -0.9, 0.9, 0.5])
def test_step_multiplier_recovers_the_signed_multiplier(lam):
    """The multiplier's sign separates a cycle (lam < 0) from slow decay."""
    res = _run(lam, mixing=0.0, max_iter=6, conv_tol=0.0)
    assert res.step_multiplier == pytest.approx(lam, rel=1e-9)


def test_step_multiplier_reports_the_plain_map_under_mixing():
    """Under mixing the residual ratio is (1 - beta) lam + beta; lam is reported.

    The diagnostic answers "does the plain step need mixing?", so it must not
    change meaning when mixing is already on.
    """
    lam, beta = -1.5, 0.5
    res = _run(lam, mixing=beta, max_iter=6, conv_tol=0.0)
    assert res.step_multiplier == pytest.approx(lam, rel=1e-9)


def test_step_multiplier_is_nan_without_two_measured_sweeps():
    res = _run(0.5, mixing=0.0, max_iter=1, conv_tol=0.0)
    assert np.isnan(res.step_multiplier)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mixing": 1.0}, r"\[0, 1\)"),
        ({"mixing": -0.1}, r"\[0, 1\)"),
        ({"mixing": 0.3, "conv_method": "sv"}, "elementwise"),
        ({"mixing": 0.3, "gauge": None}, "gauge"),
    ],
)
def test_loop_refuses_meaningless_mixing(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _run(0.5, **kwargs)


def test_ctmconfig_validates_ctm_mixing():
    from tenax.algorithms.ipeps_config import CTMConfig

    assert CTMConfig().ctm_mixing == 0.0
    CTMConfig(forward_gauge="bond_phase", ctm_mixing=0.3)
    with pytest.raises(ValueError, match="bond_phase"):
        CTMConfig(forward_gauge="phase", ctm_mixing=0.3)
    with pytest.raises(ValueError, match="bond_phase"):
        CTMConfig(forward_gauge="bond_phase", ctm_conv_method="sv", ctm_mixing=0.3)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CTMConfig(forward_gauge="bond_phase", ctm_mixing=1.0)


def test_mixing_reaches_every_forward(monkeypatch):
    """ctm_converge_kwargs and python_loop_ctm_converge hand mixing to the loop."""
    import tenax.algorithms._ctm_python_loop as pl
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
    from tenax.algorithms.ipeps_config import CTMConfig

    cfg = CTMConfig(chi=4, forward_gauge="bond_phase", ctm_mixing=0.3)
    kw = ctm_converge_kwargs(cfg)
    assert kw["mixing"] == 0.3

    seen = []

    class _Stop(Exception):
        pass

    def _spy(*args, **kwargs):
        seen.append(kwargs["mixing"])
        raise _Stop

    monkeypatch.setattr(pl, "_run_ctm_loop_with_bump", _spy)
    site = _dense_site()
    with pytest.raises(_Stop):
        pl.python_loop_ctm_converge(
            {(0, 0): site, (1, 0): site}, CHECKERBOARD_NEIGHBORS, **kw
        )
    assert seen == [0.3]


def test_mixing_reaches_the_implicit_ad_forward_per_value(monkeypatch):
    """Each mixing value reaches the loss forward: it is part of the VJP cache key.

    Were it missing from the key, the second call would reuse the closure
    built for the first and run the old mixing.
    """
    import tenax.algorithms._ctm_energy_ad as ead
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS

    seen = []

    class _Stop(Exception):
        pass

    def _spy(*args, **kwargs):
        seen.append(kwargs["mixing"])
        raise _Stop

    monkeypatch.setattr(ead, "_sigma_gauged_ctm_converge", _spy)
    site = _dense_site()
    gate = jnp.zeros((2, 2, 2, 2))
    for beta in (0.25, 0.5):
        with pytest.raises(_Stop):
            ead.ctm_energy_implicit(
                {(0, 0): site, (1, 0): site},
                CHECKERBOARD_NEIGHBORS,
                gate,
                chi=4,
                forward_gauge="bond_phase",
                conv_method="elementwise",
                mixing=beta,
            )
    assert seen == [0.25, 0.5]


def test_mixing_reaches_the_ad_loss(monkeypatch):
    """make_ctm_energy_fn forwards CTMConfig.ctm_mixing to the implicit loss."""
    import tenax.algorithms._ctm_energy_ad as ead
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
    from tenax.algorithms.ipeps_config import CTMConfig

    seen = []
    monkeypatch.setattr(
        ead, "ctm_energy_implicit", lambda *a, **k: seen.append(k["mixing"]) or 0.0
    )
    cfg = CTMConfig(chi=4, forward_gauge="bond_phase", ctm_mixing=0.4)
    fn = make_ctm_energy_fn(
        neighbors=CHECKERBOARD_NEIGHBORS,
        gate=None,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=0,
        explicit_steps=0,
    )
    fn({})
    assert seen == [0.4]


def _dense_site():
    from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry

    sym = U1Symmetry()
    z2, z3 = np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32)
    indices = tuple(
        TensorIndex.from_charges(sym, q.copy(), f, label=lbl)
        for q, f, lbl in (
            (z2, FlowDirection.OUT, "u"),
            (z2, FlowDirection.IN, "d"),
            (z2, FlowDirection.OUT, "l"),
            (z2, FlowDirection.IN, "r"),
            (z3, FlowDirection.IN, "p"),
        )
    )
    data = np.random.default_rng(0).standard_normal((2, 2, 2, 2, 2))
    return DenseTensor(data, indices)
