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
    """Auto-promote projector settings for AD paths.

    - Implicit AD: promote to SVD (Fishman) projectors, which have a
      numerically safe S^{-1/2} backward via the S_safe pattern.  The
      eigh projector with standard backward is unstable at chi > D^2
      due to degenerate eigenvalues.
    - Explicit AD + eigh: promote projector_backward to lorentzian.
    """
    ctm_cfg = config.ctm
    effective_projector = config.gs_projector_method or ctm_cfg.projector_method

    if config.gs_implicit_ad:
        # Implicit AD needs SVD (Fishman) projectors for stable gradients
        # at chi > D^2.  eigh backward has degenerate-eigenvalue issues
        # that corrupt the GMRES solve.
        if effective_projector == "eigh" and config.gs_projector_method is None:
            new_config = replace(config, gs_projector_method="svd")
            if logger is not None:
                logger.info(
                    "projector_method auto-promoted: eigh -> svd "
                    "(implicit AD, Fishman projector for stable backward)"
                )
            return new_config
        return config

    # Explicit AD + eigh: promote backward to lorentzian
    if effective_projector != "eigh":
        return config
    if ctm_cfg.projector_backward != "auto":
        return config

    new_ctm = replace(ctm_cfg, projector_backward="lorentzian")
    if logger is not None:
        logger.info(
            "projector_backward auto-promoted: auto -> lorentzian "
            "(explicit AD + projector_method=eigh)"
        )
    return replace(config, ctm=new_ctm)


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
