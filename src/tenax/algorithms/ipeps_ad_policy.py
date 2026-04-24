"""Policy helpers for iPEPS AD dispatch and CTM configuration.

This module centralizes lightweight decision logic used by
``ipeps_optimize`` so routing and config overrides are defined in one place.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def resolve_projector_backward(
    config: iPEPSConfig,
    *,
    logger: logging.Logger | None = None,
) -> iPEPSConfig:
    """No-op — kept for backward compatibility.

    Previously auto-promoted projector settings; now the defaults
    (``projector_method="svd"``, ``forward_gauge="phase"``) are correct
    for all AD paths.  Users who explicitly set ``projector_method="eigh"``
    get eigh without silent overrides.
    """
    return config


def use_reference_c4v_path(config: iPEPSConfig) -> bool:
    """Return ``True`` when the strict reference-mode dispatch gate is met."""
    return (
        config.unit_cell == "1x1"
        and config.gs_c4v
        and config.gs_implicit_ad
        and getattr(config.ctm, "ctm_ad_mode", None) == "c4v_reference"
    )


def build_ad_ctm_config(config: iPEPSConfig) -> CTMConfig:
    """Return the effective CTMConfig used by iPEPS AD optimizers.

    Applies AD-only policy overrides while leaving ``config.ctm`` unchanged.
    """
    ctm_cfg = config.ctm
    if config.gs_projector_method is not None:
        ctm_cfg = replace(ctm_cfg, projector_method=config.gs_projector_method)
    if ctm_cfg.forward_gauge == "qr":
        ctm_cfg = replace(ctm_cfg, forward_gauge="phase")
    return ctm_cfg
