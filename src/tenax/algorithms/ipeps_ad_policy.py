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
    """Validate AD policy invariants and return ``config`` unchanged.

    No silent promotion is applied. For the implicit-AD path we enforce the
    empirically stable CTM combination:
    - ``projector_method == "svd"``
    - ``forward_gauge == "phase"`` (Frobenius + phase fixing path)
    - ``ctm_conv_method == "elementwise"``

    Explicit-AD keeps user choices unchanged.
    """
    if not config.gs_implicit_ad:
        return config

    ctm_cfg = build_ad_ctm_config(config)
    errors: list[str] = []
    if ctm_cfg.projector_method != "svd":
        errors.append(f"projector_method={ctm_cfg.projector_method!r} (expected 'svd')")
    if ctm_cfg.forward_gauge != "phase":
        errors.append(f"forward_gauge={ctm_cfg.forward_gauge!r} (expected 'phase')")
    if ctm_cfg.ctm_conv_method != "elementwise":
        errors.append(
            f"ctm_conv_method={ctm_cfg.ctm_conv_method!r} (expected 'elementwise')"
        )

    if errors:
        msg = (
            "Implicit AD requires CTM settings "
            "(projector_method='svd', forward_gauge='phase', "
            "ctm_conv_method='elementwise'). Got: " + ", ".join(errors)
        )
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg)

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
    No silent gauge promotion: explicit user choices are preserved.  The
    defaults (``projector_method="svd"``, ``forward_gauge="phase"``) are
    already correct for AD paths.
    """
    ctm_cfg = config.ctm
    if config.gs_projector_method is not None:
        ctm_cfg = replace(ctm_cfg, projector_method=config.gs_projector_method)
    return ctm_cfg
