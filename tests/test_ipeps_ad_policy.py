"""Unit tests for iPEPS AD dispatch/config policy helpers."""

import pytest

from tenax.algorithms.ipeps_ad_policy import (
    build_ad_ctm_config,
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
