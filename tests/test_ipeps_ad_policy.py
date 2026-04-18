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


def test_build_ad_ctm_config_auto_promotes_qr_to_phase_for_explicit_ad():
    config = iPEPSConfig(
        ctm=CTMConfig(chi=8, projector_method="eigh", forward_gauge="qr"),
        gs_implicit_ad=False,
    )

    updated = build_ad_ctm_config(config)

    assert updated.forward_gauge == "phase"
    assert config.ctm.forward_gauge == "qr"


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


def test_resolve_projector_backward_promotes_only_auto_explicit_eigh():
    promoted = resolve_projector_backward(
        iPEPSConfig(ctm=CTMConfig(chi=8, projector_method="eigh"), gs_implicit_ad=False)
    )
    assert promoted.ctm.projector_backward == "lorentzian"

    unchanged = resolve_projector_backward(
        iPEPSConfig(
            ctm=CTMConfig(
                chi=8,
                projector_method="eigh",
                projector_backward="standard",
            ),
            gs_implicit_ad=False,
        )
    )
    assert unchanged.ctm.projector_backward == "standard"
