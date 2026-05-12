"""Unit tests for iPEPS AD dispatch/config policy helpers."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from tenax.algorithms.ipeps_ad_policy import (
    build_ad_ctm_config,
    ctm_converge_kwargs,
    make_ctm_energy_fn,
    resolve_projector_backward,
    use_reference_c4v_path,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def test_build_ad_ctm_config_applies_projector_override_without_mutation():
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="svd", forward_gauge="sigma"),
        gs_projector_method="eigh",
        gs_implicit_ad=True,
    )

    updated = build_ad_ctm_config(config)

    assert updated.projector_method == "eigh"
    assert updated.forward_gauge == "sigma"
    assert config.ctm.projector_method == "svd"


def test_build_ad_ctm_config_preserves_explicit_qr_no_silent_promotion():
    """Explicit user choice of ``forward_gauge="qr"`` is preserved — no silent override."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, forward_gauge="qr"),
    )

    updated = build_ad_ctm_config(config)

    assert updated.forward_gauge == "qr"
    assert config.ctm.forward_gauge == "qr"


def test_build_ad_ctm_config_preserves_explicit_phase():
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, forward_gauge="phase"),
    )

    updated = build_ad_ctm_config(config)

    assert updated.forward_gauge == "phase"


def test_use_reference_c4v_path_requires_all_conditions():
    cfg_ok = iPEPSConfig(
        unit_cell="1x1",
        gs_c4v=True,
        gs_implicit_ad=True,
        ctm=CTMConfig(chi=8, ctm_ad_mode="c4v_reference"),
    )
    assert use_reference_c4v_path(cfg_ok)

    assert not use_reference_c4v_path(iPEPSConfig(unit_cell="2site", gs_c4v=True))
    assert not use_reference_c4v_path(iPEPSConfig(unit_cell="1x1", gs_c4v=False))


def test_resolve_projector_backward_rejects_implicit_non_svd():
    """Implicit AD requires SVD projector."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="eigh"),
        gs_implicit_ad=True,
    )
    with pytest.raises(ValueError, match="Implicit AD requires CTM settings"):
        resolve_projector_backward(config)


def test_resolve_projector_backward_rejects_implicit_non_phase_gauge():
    """Implicit AD requires phase gauge."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="svd", forward_gauge="sigma"),
        gs_implicit_ad=True,
    )
    with pytest.raises(ValueError, match="forward_gauge"):
        resolve_projector_backward(config)


def test_resolve_projector_backward_rejects_implicit_non_elementwise_conv():
    """Implicit AD requires element-wise convergence check."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="svd", ctm_conv_method="sv"),
        gs_implicit_ad=True,
    )
    with pytest.raises(ValueError, match="ctm_conv_method"):
        resolve_projector_backward(config)


def test_resolve_projector_backward_rejects_implicit_gs_projector_override():
    """Implicit AD validates effective projector after gs override."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="svd"),
        gs_projector_method="qr",
        gs_implicit_ad=True,
    )
    with pytest.raises(ValueError, match="projector_method='svd'"):
        resolve_projector_backward(config)


def test_resolve_projector_backward_accepts_implicit_stable_combo():
    """Implicit AD stable combo passes unchanged."""
    config = iPEPSConfig(
        ctm=CTMConfig(
            chi=8,
            projector_method="svd",
            forward_gauge="phase",
            ctm_conv_method="elementwise",
        ),
        gs_implicit_ad=True,
    )
    out = resolve_projector_backward(config)
    assert out is config


def test_ctm_converge_kwargs_propagates_min_iter_and_conv_method():
    cfg = CTMConfig(
        chi=12,
        max_iter=37,
        min_iter=5,
        conv_tol=1e-7,
        ctm_conv_method="elementwise",
        projector_method="svd",
        qr_warmup_steps=2,
        projector_backward="lorentzian",
    )
    kw = ctm_converge_kwargs(cfg, env_init=None)
    # The point of the helper is that warm-start CTM no longer silently
    # falls back to function defaults (min_iter=10, conv_method="sv");
    # callers always see ctm_cfg-derived values.
    assert kw["min_iter"] == 5
    assert kw["conv_method"] == "elementwise"
    assert kw["chi"] == 12
    assert kw["projector_method"] == "svd"
    assert kw["env_init"] is None


def test_make_ctm_energy_fn_resolves_ctm_cfg_at_call_time():
    """``conv_tol`` schedule rebindings must reach the AD loss closure.

    Dispatchers update their local ``ctm_cfg`` via
    ``ctm_cfg = _replace(ctm_cfg, conv_tol=new_tol)`` when
    ``gs_ctm_conv_tol_schedule`` is active.  Capturing the dataclass by
    value at helper-construction time (a previous version of this code)
    silently froze ``conv_tol`` on the AD path while the warm-start CTM
    saw updates — exactly the kind of split-policy drift #351 set out to
    eliminate.  The helper takes a callable so the closure resolves the
    live binding on every invocation.
    """
    seen_conv_tols: list[float] = []

    def _fake_ctm_energy_implicit(*_args, **kwargs):
        seen_conv_tols.append(kwargs["conv_tol"])
        return 0.0

    ctm_cfg = CTMConfig(
        chi=8,
        max_iter=10,
        conv_tol=1e-4,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
    )

    # Patch BEFORE constructing the helper so the import inside
    # make_ctm_energy_fn resolves to the fake.
    with patch(
        "tenax.algorithms._ctm_energy_ad.ctm_energy_implicit",
        new=_fake_ctm_energy_implicit,
    ):
        energy_fn = make_ctm_energy_fn(
            neighbors={(0, 0): {}},
            gate=None,
            get_ctm_cfg=lambda: ctm_cfg,
            env_cache={},
            use_explicit=False,
            explicit_warmup=0,
            explicit_steps=0,
        )
        energy_fn({})
        # Simulate the conv_tol schedule rebinding the dispatcher's local
        # ctm_cfg between AD steps.
        ctm_cfg = replace(ctm_cfg, conv_tol=1e-9)
        energy_fn({})

    assert seen_conv_tols == [1e-4, 1e-9], (
        "make_ctm_energy_fn must observe the rebinding of ctm_cfg "
        f"(saw {seen_conv_tols!r})"
    )


def test_make_ctm_energy_fn_forces_plateau_patience_none_on_implicit_path():
    """Implicit-AD energy must not early-bail on a CTM plateau.

    Codex follow-up review on PR #439: the implicit-AD backward solves
    ``(I - J^T) λ = ∂L/∂env`` around the returned env, which is only
    well-defined when env is a fixed point.  If we let
    ``ctm_cfg.plateau_patience`` propagate into ``ctm_energy_implicit``,
    a plateau bail returns a best-iter (non-fixed-point) env and the
    implicit-function-theorem premise breaks → inconsistent gradients.
    Forcing ``plateau_patience=None`` at the policy layer keeps the AD
    gradient math correct.  Warm-start CTM calls outside the custom_vjp
    boundary (``ctm_converge_kwargs``) still honor the field.
    """
    seen_plateau: list[int | None] = []

    def _fake_ctm_energy_implicit(*_args, **kwargs):
        seen_plateau.append(kwargs.get("plateau_patience"))
        return 0.0

    # Even with the most aggressive bail, the implicit path must see None.
    ctm_cfg = CTMConfig(
        chi=8,
        max_iter=10,
        conv_tol=1e-4,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        plateau_patience=3,
    )

    with patch(
        "tenax.algorithms._ctm_energy_ad.ctm_energy_implicit",
        new=_fake_ctm_energy_implicit,
    ):
        energy_fn = make_ctm_energy_fn(
            neighbors={(0, 0): {}},
            gate=None,
            get_ctm_cfg=lambda: ctm_cfg,
            env_cache={},
            use_explicit=False,
            explicit_warmup=0,
            explicit_steps=0,
        )
        energy_fn({})

    assert seen_plateau == [None], (
        "make_ctm_energy_fn must force plateau_patience=None on the "
        f"implicit-AD path regardless of ctm_cfg (saw {seen_plateau!r})"
    )

    # Warm-start path (ctm_converge_kwargs) still honors the field.
    kw = ctm_converge_kwargs(ctm_cfg)
    assert kw["plateau_patience"] == 3


from tenax.algorithms.ipeps_ad_policy import validate_ctm_for_implicit_ad


def test_validate_ctm_for_implicit_ad_rejects_non_svd():
    cfg = CTMConfig(chi=8, projector_method="eigh")
    with pytest.raises(ValueError, match="projector_method"):
        validate_ctm_for_implicit_ad(cfg)


def test_validate_ctm_for_implicit_ad_rejects_non_phase_gauge():
    cfg = CTMConfig(chi=8, projector_method="svd", forward_gauge="sigma")
    with pytest.raises(ValueError, match="forward_gauge"):
        validate_ctm_for_implicit_ad(cfg)


def test_validate_ctm_for_implicit_ad_rejects_non_elementwise_conv():
    cfg = CTMConfig(chi=8, projector_method="svd", ctm_conv_method="sv")
    with pytest.raises(ValueError, match="ctm_conv_method"):
        validate_ctm_for_implicit_ad(cfg)


def test_validate_ctm_for_implicit_ad_accepts_stable_combo():
    cfg = CTMConfig(
        chi=8,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
    )
    # No exception:
    validate_ctm_for_implicit_ad(cfg)
