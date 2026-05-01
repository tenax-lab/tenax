"""Policy helpers for iPEPS AD dispatch and CTM configuration.

This module centralizes lightweight decision logic used by
``ipeps_optimize`` so routing and config overrides are defined in one place.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def ctm_converge_kwargs(
    ctm_cfg: CTMConfig,
    *,
    env_init=None,
) -> dict:
    """Return the policy kwargs forwarded to ``python_loop_ctm_converge``.

    Centralizing this dict means the 1-site, 2-site, and multisite iPEPS
    dispatchers can warm-start CTM with identical settings — fixing the
    historical issue where some warm-start paths silently fell back to
    ``conv_method="sv"`` while the implicit-AD energy used
    ``conv_method="elementwise"`` (#351).
    """
    return {
        "chi": ctm_cfg.chi,
        "max_iter": ctm_cfg.max_iter,
        "min_iter": ctm_cfg.min_iter,
        "conv_tol": ctm_cfg.conv_tol,
        "conv_method": ctm_cfg.ctm_conv_method,
        "renormalize": ctm_cfg.renormalize,
        "projector_method": ctm_cfg.projector_method,
        "qr_warmup_steps": ctm_cfg.qr_warmup_steps,
        "projector_backward": ctm_cfg.projector_backward,
        "chi_ramp": ctm_cfg.chi_ramp,
        "env_init": env_init,
    }


def make_ctm_energy_fn(
    *,
    neighbors,
    gate,
    get_ctm_cfg: Callable[[], CTMConfig],
    env_cache: dict,
    use_explicit: bool,
    explicit_warmup: int,
    explicit_steps: int,
    energy_fn=None,
):
    """Build a ``site_tensors → energy`` closure used by every iPEPS AD dispatcher.

    Encapsulates the explicit-vs-implicit branch and the kwarg unpacking
    from ``ctm_cfg`` so a policy change touches one place rather than three
    (issue #351, item 4).

    ``get_ctm_cfg`` is a zero-arg callable that resolves the current
    ``CTMConfig`` at every invocation.  Dispatchers rebind their local
    ``ctm_cfg`` when ``gs_ctm_conv_tol_schedule`` is active, so the
    callable must read the live binding (typically ``lambda: ctm_cfg``)
    rather than capturing a snapshot.  Passing the dataclass directly
    would freeze ``conv_tol`` on the AD path while the warm-start CTM
    saw updates — the very kind of split-policy bug #351 set out to
    eliminate.

    Args:
        neighbors:        Coord → direction → coord neighbor graph.
        gate:             Two-site Hamiltonian gate.
        get_ctm_cfg:      Zero-arg callable returning the current
                          effective ``CTMConfig`` (e.g.
                          ``lambda: ctm_cfg``).
        env_cache:        Dict storing the warm-start envs under key ``"envs"``.
        use_explicit:     True for the explicit-AD path, False for implicit.
        explicit_warmup:  Number of forward-only warmup CTM sweeps (explicit).
        explicit_steps:   Number of backprop-tracked CTM sweeps (explicit).
        energy_fn:        Optional energy callback for non-default
                          (e.g. coarse-grain) energy evaluation.
    """
    # Deferred to avoid pulling the CTM/AD stack at module import time.
    from tenax.algorithms._ctm_energy_ad import (
        ctm_energy_explicit,
        ctm_energy_implicit,
    )

    def _ctm_energy_fn(site_tensors):
        ctm_cfg = get_ctm_cfg()
        env_init = env_cache.get("envs", None)
        if use_explicit:
            return ctm_energy_explicit(
                site_tensors,
                neighbors,
                gate,
                chi=ctm_cfg.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                env_init=env_init,
                energy_fn=energy_fn,
            )
        return ctm_energy_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            projector_method=ctm_cfg.projector_method,
            renormalize=ctm_cfg.renormalize,
            projector_backward=ctm_cfg.projector_backward,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=env_init,
            forward_gauge=ctm_cfg.forward_gauge,
            conv_method=ctm_cfg.ctm_conv_method,
            min_iter=ctm_cfg.min_iter,
            gmres_tol=ctm_cfg.gmres_tol,
            gmres_maxiter=ctm_cfg.gmres_maxiter,
            gmres_restart=ctm_cfg.gmres_restart,
            arnoldi_precheck=False,
            energy_fn=energy_fn,
        )

    return _ctm_energy_fn
