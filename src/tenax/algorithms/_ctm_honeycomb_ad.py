"""Honeycomb CTM-to-energy AD wrapper: Python-loop forward + GMRES backward.

Mirrors :func:`tenax.algorithms._ctm_energy_ad.ctm_energy_implicit` for the
rank-4, 6-corner, 3-direction, 2-sublattice honeycomb topology. Forward is
:func:`honeycomb_ctm_run`; backward is implicit differentiation through the
converged env via JIT-fused GMRES on ``(I - J_env^T) λ = dE/denv``.

The biorthogonal projector path is the default (``projector_method="biorthogonal"``,
``forward_gauge="phase"``); the per-column phase fix is already applied inside
the projector build, so the converged env is an element-wise fixed point and
no extra post-step gauge fix is needed (memory: ``feedback_phase_gauge_default.md``).
"""

from __future__ import annotations

__all__ = ["honeycomb_ctm_energy_implicit"]

from typing import Any

import jax
import jax.numpy as jnp

from tenax.algorithms._arnoldi import arnoldi_spectral_radius_pytree
from tenax.algorithms._ctm_honeycomb_energy import compute_honeycomb_energy
from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_forward import (
    _jit_honeycomb_step,
    honeycomb_ctm_run,
)
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.algorithms._gmres_lax import gmres_pytree_jax
from tenax.algorithms.ad_utils import CTMRGGradientError


def honeycomb_ctm_energy_implicit(
    sites: dict[Coord, Any],
    hamiltonian: jnp.ndarray,
    *,
    chi: int = 16,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    projector_method: str = "biorthogonal",
    forward_gauge: str = "phase",
    renormalize: bool = True,
    conv_method: str = "elementwise",
    min_iter: int = 4,
    chi_ramp: list[tuple[int, int | None]] | None = None,
    env_init: dict[Coord, HoneycombCTMEnv] | None = None,
    energy_fn=None,
    gmres_tol: float = 1e-6,
    gmres_maxiter: int = 200,
    gmres_restart: int = 30,
    arnoldi_precheck: bool = False,
) -> jnp.ndarray:
    """Honeycomb iPEPS energy with implicit-differentiation backward (GMRES).

    Forward: :func:`honeycomb_ctm_run` (Python-loop CTM convergence).
    Backward: JIT-fused GMRES solve of ``(I - J_env^T) λ = dE/denv``,
    chained to ``∂E/∂sites = ∂E/∂sites + J_sites^T λ``.

    Args:
        sites: ``{(0, 0): A, (1, 0): B}`` rank-4 honeycomb site tensors.
        hamiltonian: 2-site bond Hamiltonian, shape ``(d², d²)`` for the
            default ``energy_fn`` (3-edge NN sum). The honeycomb caller may
            override ``energy_fn`` to consume a different operator shape.
        chi: Environment bond dimension.
        max_iter: Maximum CTM iterations per chi-ramp stage.
        conv_tol: Convergence tolerance for ``conv_method``.
        projector_method: ``"biorthogonal"`` (default), ``"eigh"`` or ``"svd"``
            (the latter two are isometric A=B opt-in paths).
        forward_gauge: ``"phase"`` (default) — phase fix is applied inside the
            projector build.
        renormalize: Inf-norm normalize all 9 fields after each sweep.
        conv_method: ``"elementwise"`` (default) or ``"svd"``.
        min_iter: Minimum sweeps before the first convergence check.
        chi_ramp: Optional ``[(stage_chi, stage_sweeps), ...]`` schedule.
        env_init: Optional initial envs.
        energy_fn: Optional ``(sites, envs, hamiltonian) -> scalar`` override.
            Defaults to :func:`compute_honeycomb_energy`.
        gmres_tol: GMRES relative tolerance.
        gmres_maxiter: GMRES maximum iterations.
        gmres_restart: GMRES restart parameter.
        arnoldi_precheck: If True, run a 20-step Arnoldi pre-estimate of
            ``ρ(J^T)`` and raise :class:`CTMRGGradientError` if ``ρ ≥ 1``.

    Returns:
        Scalar energy per honeycomb unit cell (sum over 3 NN bonds by default).
    """
    coords = sorted(sites.keys())
    params_data = tuple(sites[c] for c in coords)

    return _honeycomb_ctm_energy_implicit_dispatch(
        params_data,
        coords,
        hamiltonian,
        chi,
        max_iter,
        conv_tol,
        projector_method,
        forward_gauge,
        renormalize,
        conv_method,
        min_iter,
        chi_ramp,
        env_init,
        energy_fn,
        gmres_tol,
        gmres_maxiter,
        gmres_restart,
        arnoldi_precheck,
    )


# ---------------------------------------------------------------------------
# Dispatch + cache (one compiled custom_vjp per static-config key)
# ---------------------------------------------------------------------------


_VJP_CACHE: dict = {}


def _honeycomb_ctm_energy_implicit_dispatch(
    params_data_tuple,
    coords,
    hamiltonian,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    forward_gauge,
    renormalize,
    conv_method,
    min_iter,
    chi_ramp,
    env_init,
    energy_fn,
    gmres_tol,
    gmres_maxiter,
    gmres_restart,
    arnoldi_precheck,
):
    """Cache the compiled custom_vjp keyed on static configuration.

    ``hamiltonian``, ``chi_ramp``, ``env_init`` and ``energy_fn`` are stored
    in a ``mutables`` dict and updated per call so the JIT'd backward
    survives optimizer steps even when the gate or initial env changes.
    """
    cache_key = (
        tuple(coords),
        chi,
        max_iter,
        conv_tol,
        projector_method,
        forward_gauge,
        renormalize,
        conv_method,
        min_iter,
        gmres_tol,
        gmres_maxiter,
        gmres_restart,
        id(hamiltonian),
        id(energy_fn),
        arnoldi_precheck,
    )

    entry = _VJP_CACHE.get(cache_key)
    if entry is not None:
        f, mutables = entry
        mutables["hamiltonian"] = hamiltonian
        mutables["chi_ramp"] = chi_ramp
        mutables["env_init"] = env_init
        mutables["energy_fn"] = energy_fn
        return f(params_data_tuple)

    mutables = {
        "hamiltonian": hamiltonian,
        "chi_ramp": chi_ramp,
        "env_init": env_init,
        "energy_fn": energy_fn,
    }
    f = _make_honeycomb_implicit_vjp_fn(
        coords=coords,
        mutables=mutables,
        chi=chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        projector_method=projector_method,
        forward_gauge=forward_gauge,
        renormalize=renormalize,
        conv_method=conv_method,
        min_iter=min_iter,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_restart=gmres_restart,
        arnoldi_precheck=arnoldi_precheck,
    )
    _VJP_CACHE[cache_key] = (f, mutables)
    return f(params_data_tuple)


def _make_honeycomb_implicit_vjp_fn(
    coords,
    mutables,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    forward_gauge,
    renormalize,
    conv_method,
    min_iter,
    gmres_tol,
    gmres_maxiter,
    gmres_restart,
    arnoldi_precheck,
):
    """Build a custom_vjp-decorated function closed over static config."""

    _cached: dict = {}

    def _run_forward(site_tensors):
        envs, _ = honeycomb_ctm_run(
            site_tensors,
            chi=chi,
            max_iter=max_iter,
            conv_tol=conv_tol,
            projector_method=projector_method,
            forward_gauge=forward_gauge,
            renormalize=renormalize,
            conv_method=conv_method,
            min_iter=min_iter,
            chi_ramp=mutables["chi_ramp"],
            env_init=mutables["env_init"],
        )
        return envs

    def _compute_energy(site_tensors, envs):
        H = mutables["hamiltonian"]
        e_fn = mutables["energy_fn"]
        if e_fn is not None:
            return e_fn(site_tensors, envs, H)
        return compute_honeycomb_energy(site_tensors, envs, H)

    @jax.custom_vjp
    def f(params_data_tuple):
        site_tensors = dict(zip(coords, params_data_tuple))
        envs = _run_forward(site_tensors)
        return _compute_energy(site_tensors, envs)

    def f_fwd(params_data_tuple):
        site_tensors = dict(zip(coords, params_data_tuple))
        envs = _run_forward(site_tensors)
        energy = _compute_energy(site_tensors, envs)

        _cached["env_treedef"] = jax.tree.structure(envs)
        env_leaves = tuple(jax.tree.leaves(envs))
        residuals = (params_data_tuple, env_leaves)
        return energy, residuals

    @jax.jit
    def _jit_dE_denv(params_data_tuple, env_leaves):
        """Step 1: dE/denv at converged env (GMRES RHS)."""
        H_ = mutables["hamiltonian"]
        e_fn_ = mutables["energy_fn"]
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]

        def energy_from_env(env_leaves_flat):
            envs = jax.tree.unflatten(env_treedef, env_leaves_flat)
            if e_fn_ is not None:
                return e_fn_(site_tensors, envs, H_)
            return compute_honeycomb_energy(site_tensors, envs, H_)

        _, vjp_fn = jax.vjp(energy_from_env, env_leaves)
        return vjp_fn(jnp.ones(()))[0]

    @jax.jit
    def _jit_apply_I_minus_Jt(params_data_tuple, env_leaves, v):
        """Step 2: one ``(I - J^T)`` matvec for the GMRES iteration."""
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]

        def step_from_env(env_leaves_flat):
            envs = jax.tree.unflatten(env_treedef, env_leaves_flat)
            envs_new = _jit_honeycomb_step(
                envs,
                site_tensors,
                chi=chi,
                projector_method=projector_method,
                forward_gauge=forward_gauge,
                renormalize=renormalize,
            )
            return tuple(jax.tree.leaves(envs_new))

        _, vjp_fn = jax.vjp(step_from_env, env_leaves)
        jt_v = vjp_fn(v)[0]
        return tuple(vi - ji for vi, ji in zip(v, jt_v))

    @jax.jit
    def _jit_chain_rule(params_data_tuple, env_leaves, lam, g_scalar):
        """Steps 3-4: direct ∂E/∂sites + indirect ``J_sites^T λ``."""
        H_ = mutables["hamiltonian"]
        e_fn_ = mutables["energy_fn"]
        env_treedef = _cached["env_treedef"]
        envs = jax.tree.unflatten(env_treedef, env_leaves)

        def energy_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            if e_fn_ is not None:
                return e_fn_(st, envs, H_)
            return compute_honeycomb_energy(st, envs, H_)

        _, vjp_energy_params = jax.vjp(energy_from_params, params_data_tuple)
        direct = vjp_energy_params(jnp.ones(()))[0]

        def step_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            envs_new = _jit_honeycomb_step(
                envs,
                st,
                chi=chi,
                projector_method=projector_method,
                forward_gauge=forward_gauge,
                renormalize=renormalize,
            )
            return tuple(jax.tree.leaves(envs_new))

        _, vjp_step_params = jax.vjp(step_from_params, params_data_tuple)
        indirect = vjp_step_params(lam)[0]

        total = jax.tree.map(lambda d, ind: g_scalar * (d + ind), direct, indirect)
        return (total,)

    def f_bwd(residuals, g):
        params_data_tuple, env_leaves = residuals

        # Step 1: dE/denv
        dE_denv = _jit_dE_denv(params_data_tuple, env_leaves)

        # Step 1.5: optional Arnoldi pre-check on ``ρ(J^T)``.
        if arnoldi_precheck:

            def apply_Jt_only(v):
                i_minus_jt_v = _jit_apply_I_minus_Jt(params_data_tuple, env_leaves, v)
                return tuple(vi - ri for vi, ri in zip(v, i_minus_jt_v))

            rho = arnoldi_spectral_radius_pytree(apply_Jt_only, dE_denv, n_iter=20)
            if rho >= 1.0:
                raise CTMRGGradientError(rho)

        # Step 2: GMRES solve via ``jax.scipy.sparse.linalg.gmres`` (eager
        # wrapper). Honors ``gmres_maxiter`` per PR #343 — failed convergence
        # is warned, not raised; the chain rule still uses whatever ``λ`` GMRES
        # returns so the optimizer can recover.
        def _matvec(v):
            return _jit_apply_I_minus_Jt(params_data_tuple, env_leaves, v)

        lam, _info = gmres_pytree_jax(
            _matvec,
            dE_denv,
            dE_denv,
            tol=gmres_tol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        lam_leaves = tuple(jax.tree.leaves(lam))

        # Steps 3-4: chain rule
        return _jit_chain_rule(params_data_tuple, env_leaves, lam_leaves, g)

    f.defvjp(f_fwd, f_bwd)
    return f
