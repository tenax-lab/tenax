"""``CTMConfig.forward_gauge="auto"``: bond_phase where supported, phase elsewhere.

The default gauge is the sentinel ``"auto"``, resolved per path by
``ipeps_config.resolve_forward_gauge``:

* ``"bond_phase"`` (#841) on the fused implicit-AD path with no ``chi_ramp``
  and ``ctm_ad_mode=None`` -- the loss closure, the warm-start / probe /
  final-evaluation forwards, and direct config-driven ``ctm_energy_implicit``
  callers (``pess_optimize``);
* ``"phase"`` on every other path -- explicit AD, split CTM, ``chi_ramp``,
  the ``ctm_ad_mode`` engines, and the legacy ``ad_utils`` paths, where the
  int encoding maps unknown spellings to ``"qr"`` and an unresolved ``"auto"``
  would silently have switched a default run to the QR gauge.

Explicit values are never promoted: explicit ``"phase"`` stays ``"phase"`` on
the implicit path, explicit ``"bond_phase"`` is still refused where it cannot
be honoured.  These tests spy on the hand-off to the CTM (mechanism, not
convergence).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._ctm_energy_ad as ead  # noqa: E402
import tenax.algorithms._ctm_python_loop as pl  # noqa: E402
import tenax.algorithms._split_ctm_energy_ad as sead  # noqa: E402
from tenax.algorithms.ad_utils import (  # noqa: E402
    _config_from_tuple,
    _config_to_tuple,
    _legacy_forward_gauge,
)
from tenax.algorithms.ipeps_ad_policy import (  # noqa: E402
    build_ad_ctm_config,
    ctm_converge_kwargs,
    make_ctm_energy_fn,
    validate_ctm_for_implicit_ad,
    validate_split_ctm_config,
)
from tenax.algorithms.ipeps_config import (  # noqa: E402
    CTMConfig,
    iPEPSConfig,
    resolve_forward_gauge,
)

_RAMP = [(4, 2), (6, None)]


def _ctm(**kw):
    """CTMConfig, silencing the chi_ramp deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return CTMConfig(chi=4, **kw)


# --------------------------------------------------------------------------
# 1. The default and the resolver
# --------------------------------------------------------------------------


def test_default_is_the_auto_sentinel():
    assert CTMConfig().forward_gauge == "auto"


@pytest.mark.parametrize(
    "kw,implicit,expected",
    [
        ({}, True, "bond_phase"),
        ({}, False, "phase"),
        ({"fuse_virtual_legs": False}, True, "phase"),
        ({"chi_ramp": _RAMP}, True, "phase"),
        ({"ctm_ad_mode": "c4v_reference"}, True, "phase"),
        ({"ctm_ad_mode": "root_implicit"}, True, "phase"),
    ],
)
def test_auto_resolves_per_path(kw, implicit, expected):
    assert _ctm(**kw).effective_forward_gauge(implicit_ad=implicit) == expected


@pytest.mark.parametrize("gauge", ["phase", "bond_phase", "qr", "sigma", "none"])
@pytest.mark.parametrize("implicit", [True, False])
def test_explicit_values_are_never_promoted(gauge, implicit):
    """Even where ``"auto"`` would pick something else."""
    for kw in ({}, {"fuse_virtual_legs": False}, {"chi_ramp": _RAMP}):
        assert resolve_forward_gauge(gauge, implicit_ad=implicit, **kw) == gauge


def test_unknown_gauge_spelling_is_refused_at_construction():
    """A typo must not reach ``_config_to_tuple``'s ``.get(..., 0) == "qr"``."""
    with pytest.raises(ValueError, match="forward_gauge"):
        CTMConfig(forward_gauge="Auto")


# --------------------------------------------------------------------------
# 2. The optimizer's entry point: build_ad_ctm_config
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctm_kw,cfg_kw,expected",
    [
        ({}, {"gs_implicit_ad": True}, "bond_phase"),
        ({}, {"gs_implicit_ad": False}, "phase"),
        ({"fuse_virtual_legs": False}, {"gs_implicit_ad": True}, "phase"),
        ({"chi_ramp": _RAMP}, {"gs_implicit_ad": True}, "phase"),
        ({"ctm_ad_mode": "root_implicit"}, {"gs_implicit_ad": True}, "phase"),
        ({"forward_gauge": "phase"}, {"gs_implicit_ad": True}, "phase"),
        ({"forward_gauge": "bond_phase"}, {"gs_implicit_ad": False}, "bond_phase"),
    ],
)
def test_build_ad_ctm_config_resolves_auto(ctm_kw, cfg_kw, expected):
    ctm = _ctm(**ctm_kw)
    config = iPEPSConfig(ctm=ctm, **cfg_kw)
    resolved = build_ad_ctm_config(config)
    assert resolved.forward_gauge == expected
    # The user's config is not mutated.
    assert config.ctm.forward_gauge == ctm_kw.get("forward_gauge", "auto")


def test_build_ad_ctm_config_does_not_rewarn_chi_ramp():
    """Resolution copies rather than ``replace``s, so ``__post_init__`` (and
    its chi_ramp DeprecationWarning) does not run a second time."""
    config = iPEPSConfig(ctm=_ctm(chi_ramp=_RAMP))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert build_ad_ctm_config(config).forward_gauge == "phase"


def test_implicit_ad_validation_accepts_the_default():
    validate_ctm_for_implicit_ad(CTMConfig(chi=4))
    validate_ctm_for_implicit_ad(build_ad_ctm_config(iPEPSConfig()))


# --------------------------------------------------------------------------
# 3. The forward-only CTMs: ctm_converge_kwargs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, "bond_phase"),
        ({"forward_gauge": "phase"}, None),
        ({"chi_ramp": _RAMP}, None),
        ({"ctm_ad_mode": "root_implicit"}, None),
        ({"ctm_ad_mode": "c4v_reference"}, None),
    ],
)
@pytest.mark.parametrize("for_probe", [False, True])
def test_ctm_converge_kwargs_resolves_auto(kw, expected, for_probe):
    out = ctm_converge_kwargs(_ctm(**kw), for_probe=for_probe)
    assert out["forward_gauge"] == expected


def test_explicit_ad_config_reaches_the_forwards_as_phase():
    resolved = build_ad_ctm_config(iPEPSConfig(gs_implicit_ad=False))
    assert ctm_converge_kwargs(resolved)["forward_gauge"] is None


# --------------------------------------------------------------------------
# 4. The loss closure: make_ctm_energy_fn
# --------------------------------------------------------------------------


def _spy(monkeypatch, module, name):
    seen = []

    def spy(*a, **k):
        seen.append(k)
        return jnp.zeros(())

    monkeypatch.setattr(module, name, spy)
    return seen


def _closure(cfg, *, use_explicit=False):
    return make_ctm_energy_fn(
        neighbors={},
        gate=None,
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=use_explicit,
        explicit_warmup=1,
        explicit_steps=1,
    )


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, "bond_phase"),
        ({"forward_gauge": "phase"}, "phase"),
        ({"forward_gauge": "bond_phase"}, "bond_phase"),
        ({"chi_ramp": _RAMP}, "phase"),
    ],
)
def test_implicit_closure_hands_the_resolved_gauge_to_the_ctm(
    kw, expected, monkeypatch
):
    seen = _spy(monkeypatch, ead, "ctm_energy_implicit")
    _closure(_ctm(**kw))({(0, 0): None})
    assert [k["forward_gauge"] for k in seen] == [expected]


def test_explicit_closure_runs_the_default_without_refusing(monkeypatch):
    seen = _spy(monkeypatch, ead, "ctm_energy_explicit")
    _closure(_ctm(), use_explicit=True)({(0, 0): None})
    assert len(seen) == 1
    with pytest.raises(NotImplementedError, match="bond_phase"):
        _closure(_ctm(forward_gauge="bond_phase"), use_explicit=True)({(0, 0): None})


def test_split_closure_runs_the_default_without_refusing(monkeypatch):
    seen = _spy(monkeypatch, sead, "ctm_energy_split_implicit")
    _closure(_ctm(fuse_virtual_legs=False))({(0, 0): None})
    assert len(seen) == 1
    validate_split_ctm_config(_ctm(fuse_virtual_legs=False), "2x2")
    with pytest.raises(NotImplementedError, match="bond_phase"):
        validate_split_ctm_config(
            _ctm(fuse_virtual_legs=False, forward_gauge="bond_phase"), "2x2"
        )


# --------------------------------------------------------------------------
# 5. ctm_energy_implicit called straight from a config (pess_optimize)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gauge,chi_ramp,expected",
    [
        ("auto", None, "bond_phase"),
        ("auto", _RAMP, "phase"),
        ("phase", None, "phase"),
        ("sigma", None, "sigma"),
    ],
)
def test_ctm_energy_implicit_resolves_auto_before_the_cache_key(
    gauge, chi_ramp, expected, monkeypatch
):
    seen = []

    def spy(*args):
        seen.append(args[13])  # forward_gauge, positional in the dispatch
        return jnp.zeros(())

    monkeypatch.setattr(ead, "_ctm_energy_implicit_dispatch", spy)
    ead.ctm_energy_implicit(
        {(0, 0): None}, {}, None, chi=4, forward_gauge=gauge, chi_ramp=chi_ramp
    )
    assert seen == [expected]


@pytest.mark.parametrize("chi_ramp,expected", [(None, "bond_phase"), (_RAMP, "phase")])
def test_ctm_energy_implicit_called_without_a_gauge_defaults_like_the_config(
    chi_ramp, expected, monkeypatch
):
    """A direct call that names no gauge gets what ``CTMConfig()`` would:
    one meaning of "the default" whether or not a config is in the way."""
    seen = []

    def spy(*args):
        seen.append(args[13])
        return jnp.zeros(())

    monkeypatch.setattr(ead, "_ctm_energy_implicit_dispatch", spy)
    ead.ctm_energy_implicit({(0, 0): None}, {}, None, chi=4, chi_ramp=chi_ramp)
    assert seen == [expected]


# --------------------------------------------------------------------------
# 6. Legacy ad_utils paths: "auto" is phase, never qr
# --------------------------------------------------------------------------


def test_legacy_config_tuple_encodes_auto_as_phase_not_qr():
    tup = _config_to_tuple(CTMConfig())
    assert _config_from_tuple(tup).forward_gauge == "phase"
    assert tup == _config_to_tuple(CTMConfig(forward_gauge="phase"))
    # Regime: "qr" really has a different code, so the test above can fail.
    assert tup != _config_to_tuple(CTMConfig(forward_gauge="qr"))


def test_legacy_gauge_reader_resolves_auto_to_phase():
    assert _legacy_forward_gauge(CTMConfig()) == "phase"
    assert _legacy_forward_gauge(CTMConfig(forward_gauge="sigma")) == "sigma"


# --------------------------------------------------------------------------
# 7. End to end: optimize_gs_ad with the default config
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dense_2site():
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

    keys = jax.random.split(jax.random.PRNGKey(1060), 2)
    sites = {}
    for c, k in zip(((0, 0), (1, 0)), keys):
        A = jax.random.normal(k, (2, 2, 2, 2, 2))
        sites[c] = _wrap_as_dense_tensor(A / jnp.linalg.norm(A))
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.kron(sz, sz) + 0.5 * (np.kron(sp, sp.T) + np.kron(sp.T, sp))
    return sites, jnp.asarray(h.reshape(2, 2, 2, 2))


@pytest.mark.parametrize(
    "implicit,loss_name,expected_loss,expected_fwd",
    [
        (True, "ctm_energy_implicit", "bond_phase", "bond_phase"),
        (False, "ctm_energy_explicit", None, None),
    ],
)
def test_optimize_gs_ad_default_hands_the_resolved_gauge_everywhere(
    implicit, loss_name, expected_loss, expected_fwd, dense_2site, monkeypatch
):
    """Every CTM ``optimize_gs_ad`` runs under the default config: the loss
    (spied, returning a constant so no backward is built) and every
    forward-only ``python_loop_ctm_converge`` (warm start, probes, final
    evaluation)."""
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    sites, gate = dense_2site
    loss_calls = []

    def loss_spy(site_tensors, *a, **k):
        loss_calls.append(k.get("forward_gauge"))
        return sum(jnp.sum(jnp.abs(t.todense()) ** 2) for t in site_tensors.values())

    monkeypatch.setattr(ead, loss_name, loss_spy)
    fwd = []
    orig = pl.python_loop_ctm_converge

    def fwd_spy(*a, **k):
        fwd.append(k.get("forward_gauge"))
        return orig(*a, **k)

    monkeypatch.setattr(pl, "python_loop_ctm_converge", fwd_spy)
    cfg = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=implicit,
        gs_num_steps=1,
        gs_verbose=False,
        ctm=CTMConfig(chi=4, max_iter=4, min_iter=1, conv_tol=1e-8),
    )
    assert cfg.ctm.forward_gauge == "auto"  # regime: the default, unset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize_gs_ad(gate, (sites[(0, 0)], sites[(1, 0)]), cfg)
    assert loss_calls, "the loss was never evaluated"
    assert all(g == expected_loss for g in loss_calls), loss_calls
    assert fwd, "no forward-only CTM ran"
    assert all(g == expected_fwd for g in fwd), fwd


def test_resolve_forward_gauge_is_public():
    """The resolver is the documented meaning of ``"auto"``, so it is exported."""
    import tenax

    assert "resolve_forward_gauge" in tenax.__all__
    assert tenax.resolve_forward_gauge("auto", implicit_ad=True) == "bond_phase"
    assert tenax.resolve_forward_gauge("auto", implicit_ad=False) == "phase"


# --------------------------------------------------------------------------
# ctm_mixing (#1060) with the "auto" default
# --------------------------------------------------------------------------


def test_mixing_accepts_the_auto_default():
    """``"auto"`` is bond_phase where mixing is supported, so it must construct."""
    cfg = CTMConfig(ctm_mixing=0.3)
    resolved = build_ad_ctm_config(iPEPSConfig(ctm=cfg))
    assert resolved.forward_gauge == "bond_phase"
    assert resolved.ctm_mixing == 0.3


def test_mixing_with_auto_still_refuses_where_auto_is_not_bond_phase():
    """Off the fused implicit path ``"auto"`` resolves to phase: mixing must raise."""
    cfg = CTMConfig(ctm_mixing=0.3)
    with pytest.raises(ValueError, match="ctm_mixing > 0 requires"):
        build_ad_ctm_config(iPEPSConfig(ctm=cfg, gs_implicit_ad=False))


@pytest.mark.parametrize("gauge", ["phase", "qr", "sigma", "none"])
def test_mixing_still_refuses_an_explicit_non_bond_gauge(gauge):
    with pytest.raises(ValueError, match="ctm_mixing > 0 requires"):
        CTMConfig(forward_gauge=gauge, ctm_mixing=0.3)


def test_legacy_paths_refuse_mixing_rather_than_drop_it():
    """``"auto"`` + mixing now constructs; the legacy paths resolve it to phase
    and have no mixing, so they must refuse, not silently run unmixed."""
    with pytest.raises(ValueError, match="ctm_mixing"):
        _legacy_forward_gauge(CTMConfig(ctm_mixing=0.3))
    assert _legacy_forward_gauge(CTMConfig()) == "phase"
