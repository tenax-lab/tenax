"""Policy helpers for iPEPS AD dispatch and CTM configuration.

This module centralizes lightweight decision logic used by
``ipeps_optimize`` so routing and config overrides are defined in one place.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def validate_ctm_for_implicit_ad(ctm_cfg: CTMConfig) -> None:
    """Raise ``ValueError`` unless ``ctm_cfg`` matches the empirically stable
    implicit-AD CTM combination: SVD projectors, phase forward-gauge,
    element-wise convergence check.

    Centralizing this check lets the iPEPS dispatchers (operating on
    :class:`iPEPSConfig`) and the PESS multisite path (operating on a raw
    :class:`CTMConfig`) share a single source of truth for the invariant.
    """
    errors: list[str] = []
    if ctm_cfg.projector_method not in ("svd", "qr"):
        errors.append(
            f"projector_method={ctm_cfg.projector_method!r} (expected 'svd' or 'qr')"
        )
    if ctm_cfg.forward_gauge != "phase":
        errors.append(f"forward_gauge={ctm_cfg.forward_gauge!r} (expected 'phase')")
    if ctm_cfg.ctm_conv_method != "elementwise":
        errors.append(
            f"ctm_conv_method={ctm_cfg.ctm_conv_method!r} (expected 'elementwise')"
        )
    if errors:
        raise ValueError(
            "Implicit AD requires CTM settings "
            "(projector_method in ('svd', 'qr'), forward_gauge='phase', "
            "ctm_conv_method='elementwise'). Got: " + ", ".join(errors)
        )


def validate_split_ctm_config(ctm_cfg: CTMConfig, recipe: str) -> None:
    """Reject ``CTMConfig`` combinations the split-CTM path cannot honor.

    The split (``fuse_virtual_legs=False``) forward is fixed-χ (single-site or
    2-site checkerboard), so every χ-changing knob is unsupported.  Raising up
    front beats the failure modes of letting one through: ``chi_ramp`` would be
    *silently ignored*
    (the split forward never reads it), and the end-of-step ``chi_auto_bump``
    would *crash* when ``_maybe_bump_chi`` pads a ``SplitCTMTensorEnv`` with
    the fused-only ``pad_dense_env_chi``.

    Shared single source of truth for ``make_ctm_energy_fn``'s split branch and
    the ``_optimize_gs_ad_tensor`` dispatcher so both entry points reject
    identically.  Covers only the ``CTMConfig``-level knobs; ``cg_gates`` and
    the ``iPEPSConfig`` schedules are guarded by the optimizer dispatcher.
    """
    if recipe not in ("1x1", "2x2"):
        raise NotImplementedError(
            "fuse_virtual_legs=False (split CTM) supports gs_recipe in "
            f"('1x1', '2x2'); got recipe={recipe!r}."
        )
    if ctm_cfg.ctmrg_heuristic_increase_chi:
        raise NotImplementedError(
            "in-CTM chi auto-bump (ctmrg_heuristic_increase_chi) is not "
            "supported on the split-CTM path; use fuse_virtual_legs=True."
        )
    # TODO(#512): chi_auto_bump and chi_ramp are deprecated and slated for
    # removal next release; drop these two checks once the fields are gone.
    if ctm_cfg.chi_auto_bump:
        raise NotImplementedError(
            "chi_auto_bump is not supported on the split-CTM path; "
            "use fuse_virtual_legs=True."
        )
    if ctm_cfg.chi_ramp is not None:
        raise NotImplementedError(
            "chi_ramp is not supported on the split-CTM path; "
            "use fuse_virtual_legs=True."
        )


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
    try:
        validate_ctm_for_implicit_ad(ctm_cfg)
    except ValueError as exc:
        if logger is not None:
            logger.error(str(exc))
        raise
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
    for_probe: bool = False,
) -> dict:
    """Return the policy kwargs forwarded to ``python_loop_ctm_converge``.

    Centralizing this dict means the 1-site, 2-site, and multisite iPEPS
    dispatchers can warm-start CTM with identical settings — fixing the
    historical issue where some warm-start paths silently fell back to
    ``conv_method="sv"`` while the implicit-AD energy used
    ``conv_method="elementwise"`` (#351).

    Set ``for_probe=True`` from forward-only line-search probes
    (``loss_fn_fwd``).  When ``ctm_cfg.probe_max_iter`` /
    ``ctm_cfg.probe_conv_tol`` are also set, those override the
    accepted-step ``max_iter`` / ``conv_tol`` so HZ φ probes bail in
    ~10-15 sweeps instead of 30-60 (issue #503).  Either probe override
    may be ``None`` independently — the un-overridden field keeps the
    accepted-step value.
    """
    _max_iter = ctm_cfg.max_iter
    _conv_tol = ctm_cfg.conv_tol
    if for_probe:
        if ctm_cfg.probe_max_iter is not None:
            _max_iter = ctm_cfg.probe_max_iter
        if ctm_cfg.probe_conv_tol is not None:
            _conv_tol = ctm_cfg.probe_conv_tol
    return {
        "chi": ctm_cfg.chi,
        "max_iter": _max_iter,
        "min_iter": ctm_cfg.min_iter,
        "conv_tol": _conv_tol,
        "conv_method": ctm_cfg.ctm_conv_method,
        "renormalize": ctm_cfg.renormalize,
        "projector_method": ctm_cfg.projector_method,
        "qr_warmup_steps": ctm_cfg.qr_warmup_steps,
        "projector_backward": ctm_cfg.projector_backward,
        "chi_ramp": ctm_cfg.chi_ramp,
        "env_init": env_init,
        "plateau_patience": ctm_cfg.plateau_patience,
        # In-CTM χ-bump (variPEPS §2.8.2; Issue #492).
        "ctmrg_heuristic_increase_chi": ctm_cfg.ctmrg_heuristic_increase_chi,
        "ctmrg_heuristic_increase_chi_threshold": (
            ctm_cfg.ctmrg_heuristic_increase_chi_threshold
        ),
        "ctmrg_heuristic_increase_chi_step_size": (
            ctm_cfg.ctmrg_heuristic_increase_chi_step_size
        ),
        "chi_max": ctm_cfg.chi_max,
        # GSPMD device mesh (#632 rung 2): shards the warm-start / probe /
        # final-env forward CTM evals too, so the whole optimize loop runs
        # sharded (not just the value_and_grad). None (default) → single-device.
        "device_mesh": ctm_cfg.device_mesh,
        # Chunked edge absorption (#632 chunk×shard): lowers peak memory ÷K
        # (1×1 recipe, dense envs). Composes with device_mesh (÷N·K total).
        # None (default) → single monolithic contraction, unchanged.
        "ctm_chunk_size": ctm_cfg.ctm_chunk_size,
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
    explicit_backward_steps: int | None = None,
    energy_fn=None,
    recipe: str = "2x2",
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
        explicit_backward_steps:
                          TBPTT cutoff (#506): when set, only the last K
                          of ``explicit_steps`` sweeps are differentiated;
                          the leading ``explicit_steps - K`` sweeps run
                          under ``jax.lax.stop_gradient``.  ``None``
                          (default) preserves full backward.
        energy_fn:        Optional energy callback for non-default
                          (e.g. coarse-grain) energy evaluation.
        recipe:           CTM sweep recipe for the implicit-AD path:
                          ``"2x2"`` (Fishman, default) or ``"1x1"`` (enables
                          ``projector_method``, incl. reduced-corner "qr").
                          Sourced from ``iPEPSConfig.gs_recipe``.  No-op for
                          the explicit-AD branch (it uses the hardcoded
                          ``"2x2"`` step).
    """
    # Deferred to avoid pulling the CTM/AD stack at module import time.
    from tenax.algorithms._ctm_energy_ad import (
        ctm_energy_explicit,
        ctm_energy_implicit,
    )
    from tenax.algorithms._split_ctm_energy_ad import (
        ctm_energy_split_explicit,
        ctm_energy_split_explicit_2site,
        ctm_energy_split_implicit,
        ctm_energy_split_implicit_2site,
    )

    def _split_ctm_energy_fn(site_tensors, ctm_cfg):
        """Dispatch to the split (``fuse_virtual_legs=False``) path.

        Single-site (``recipe='1x1'``) and 2-site checkerboard (``recipe='2x2'``)
        split forwards exist (#463 Phase 2); guard unsupported combinations up
        front.  Energy-callback handling differs by branch: the single-site
        branch forwards ``energy_fn`` (a genuinely custom CG callback is rejected
        downstream), while the 2-site branch always computes its own split-aware
        energy and ignores any ``energy_fn`` (the fused default, or a custom one).
        """
        validate_split_ctm_config(ctm_cfg, recipe)
        if len(site_tensors) == 2:
            # The 2-site split forward is the genuine joint 2×2 plaquette CTM
            # (#463 Phase 2); every 2-site split AD call hard-codes
            # ``recipe="2x2"`` downstream.  ``recipe="1x1"`` is only wired for
            # the single-site split path, so accept-then-silently-run-2x2 would
            # mislabel the experiment — reject it instead of threading a recipe
            # the 2-site kernels do not implement.
            if recipe != "2x2":
                raise NotImplementedError(
                    "The 2-site split-CTM AD path only supports gs_recipe='2x2'; "
                    f"got recipe={recipe!r}. Use unit_cell='1x1' for the '1x1' "
                    "(single-site) split recipe."
                )
            # TBPTT (#506): the 2-site split explicit path always backprops
            # through every step (``ctm_energy_split_explicit_2site`` does not
            # thread ``backward_steps`` and swallows unknown kwargs).  Full
            # backward — ``K is None`` or ``K == explicit_steps`` — is therefore
            # already correct and must be allowed; only genuine truncation
            # (``K < explicit_steps``) would be silently ignored and defeat the
            # memory cap, so reject only that until the 2-site path implements
            # truncation.
            if (
                use_explicit
                and explicit_backward_steps is not None
                and explicit_backward_steps < explicit_steps
            ):
                raise NotImplementedError(
                    "Truncated backprop (gs_explicit_ad_backward_steps < "
                    "gs_explicit_ad_steps / TBPTT) is not wired into the 2-site "
                    "split explicit-AD path; set gs_explicit_ad_backward_steps to "
                    "None or gs_explicit_ad_steps (full backward), or use the "
                    "single-site split / fused explicit path."
                )
            if use_explicit:
                return ctm_energy_split_explicit_2site(
                    site_tensors,
                    neighbors,
                    gate,
                    chi=ctm_cfg.chi,
                    warmup_steps=explicit_warmup,
                    backprop_steps=explicit_steps,
                    chi_I=ctm_cfg.chi_I,
                    renormalize=ctm_cfg.renormalize,
                    energy_fn=None,
                )
            _cached = env_cache.get("envs", None)
            envs_init = _cached if _cached else None
            return ctm_energy_split_implicit_2site(
                site_tensors,
                neighbors,
                gate,
                chi=ctm_cfg.chi,
                max_iter=ctm_cfg.max_iter,
                conv_tol=ctm_cfg.conv_tol,
                chi_I=ctm_cfg.chi_I,
                renormalize=ctm_cfg.renormalize,
                min_iter=ctm_cfg.min_iter,
                energy_fn=None,
                envs_init=envs_init,
            )
        if use_explicit:
            # The explicit split forward re-initializes internally; no
            # env_init warm-start yet (perf follow-up).
            return ctm_energy_split_explicit(
                site_tensors,
                neighbors,
                gate,
                chi=ctm_cfg.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                backward_steps=explicit_backward_steps,
                chi_I=ctm_cfg.chi_I,
                renormalize=ctm_cfg.renormalize,
                energy_fn=energy_fn,
            )
        # Warm-start the gauge-fixed forward from the cached split env (the
        # optimizer stores a {coord: SplitCTMTensorEnv} dict).  The seed is
        # gradient-free, so this only speeds convergence.
        _cached = env_cache.get("envs", None)
        split_env_init = _cached.get((0, 0)) if _cached else None
        return ctm_energy_split_implicit(
            site_tensors,
            neighbors,
            gate,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            chi_I=ctm_cfg.chi_I,
            renormalize=ctm_cfg.renormalize,
            min_iter=ctm_cfg.min_iter,
            energy_fn=energy_fn,
            env_init=split_env_init,
        )

    def _ctm_energy_fn(site_tensors):
        ctm_cfg = get_ctm_cfg()
        if not ctm_cfg.fuse_virtual_legs:
            return _split_ctm_energy_fn(site_tensors, ctm_cfg)
        env_init = env_cache.get("envs", None)
        # In-CTM χ-bump knobs (variPEPS §2.8.2; Issue #492) are inlined at
        # both call sites for grep-ability and type-checker visibility.
        # Defaults in CTMConfig keep the bump off, so this is a no-op for
        # existing callers; downstream ctm_energy_{explicit,implicit}
        # validate the combination (chi_max required, step_size > 0, etc.)
        # so we don't re-check here.
        if use_explicit:
            return ctm_energy_explicit(
                site_tensors,
                neighbors,
                gate,
                chi=ctm_cfg.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                backward_steps=explicit_backward_steps,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                env_init=env_init,
                energy_fn=energy_fn,
                ctmrg_heuristic_increase_chi=ctm_cfg.ctmrg_heuristic_increase_chi,
                ctmrg_heuristic_increase_chi_threshold=(
                    ctm_cfg.ctmrg_heuristic_increase_chi_threshold
                ),
                ctmrg_heuristic_increase_chi_step_size=(
                    ctm_cfg.ctmrg_heuristic_increase_chi_step_size
                ),
                chi_max=ctm_cfg.chi_max,
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
            adjoint_method=ctm_cfg.adjoint_method,
            energy_fn=energy_fn,
            # Flows through to the implicit-AD forward CTM.  Default 20
            # (variPEPS-style approximate AD: cheap on the plateau, the
            # gradient is wrong either way during early optimization
            # because the CTM doesn't reach a true fixed point regardless).
            # Set ``CTMConfig.plateau_patience=None`` at the final chi
            # stage when strict variational gradients matter.
            plateau_patience=ctm_cfg.plateau_patience,
            ctmrg_heuristic_increase_chi=ctm_cfg.ctmrg_heuristic_increase_chi,
            ctmrg_heuristic_increase_chi_threshold=(
                ctm_cfg.ctmrg_heuristic_increase_chi_threshold
            ),
            ctmrg_heuristic_increase_chi_step_size=(
                ctm_cfg.ctmrg_heuristic_increase_chi_step_size
            ),
            chi_max=ctm_cfg.chi_max,
            # CTM recipe (gs_recipe): "2x2" (default) or "1x1" (enables qr).
            recipe=recipe,
            # GSPMD device mesh (#632): when set, shards the implicit-AD forward
            # CTM + backward across the mesh. None (default) → single-device.
            device_mesh=ctm_cfg.device_mesh,
            # Chunked edge absorption (#632 chunk×shard): lowers peak memory ÷K
            # (1×1 recipe, dense envs). None (default) → monolithic, unchanged.
            ctm_chunk_size=ctm_cfg.ctm_chunk_size,
        )

    return _ctm_energy_fn
