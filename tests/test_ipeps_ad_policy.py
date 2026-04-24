"""Unit tests for iPEPS AD dispatch/config policy helpers."""

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


def test_build_ad_ctm_config_promotes_qr_to_phase():
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, forward_gauge="qr"),
    )

    updated = build_ad_ctm_config(config)

    assert updated.forward_gauge == "phase"
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


def test_resolve_projector_backward_is_noop():
    """resolve_projector_backward no longer auto-promotes; defaults are correct."""
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="eigh"),
        gs_implicit_ad=True,
    )
    result = resolve_projector_backward(config)
    # No silent override — user gets what they asked for
    assert result.ctm.projector_method == "eigh"
    assert result.gs_projector_method is None
