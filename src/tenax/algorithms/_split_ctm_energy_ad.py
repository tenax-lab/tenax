"""Single-site split-CTM AD energy entry points (#463 Phase 2).

Mirrors ``ctm_energy_explicit`` / ``ctm_energy_implicit`` from
``_ctm_energy_ad`` but on the split (ket/bra-separate) double layer.
Single-site (``ctm_split_tensor``, isolated 1×1 iPEPS) and 2-site
checkerboard multisite (``ctm_split_tensor_2site``, coupled forward) are
both supported here: the single-site forward-only/explicit/implicit entries
plus the 2-site forward-only + explicit-AD converge siblings below.  The
2-site implicit VJP lands in #463 Phase 2 Task 2.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

__all__ = [
    "converge_split_env",
    "converge_split_env_2site",
    "ctm_energy_split_explicit",
    "ctm_energy_split_explicit_2site",
    "ctm_energy_split_implicit",
    "ctm_energy_split_implicit_2site",
]


def _extract_single_site(site_tensors):
    if len(site_tensors) != 1:
        raise NotImplementedError(
            "split-CTM (fuse_virtual_legs=False) supports only the single-site "
            f"(recipe='1x1') path; got {len(site_tensors)} sites."
        )
    ((_coord, A),) = site_tensors.items()
    return A


def ctm_energy_split_explicit(
    site_tensors,
    neighbors,
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    backward_steps: int | None = None,
    chi_I: int | None = None,
    renormalize: bool = True,
    energy_fn=None,
    **_ignored,
):
    """Single-site iPEPS energy with explicit (unrolled) split-CTM AD."""
    A = _extract_single_site(site_tensors)
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn (e.g. coarse-grain) is not supported on the split "
            "path yet; use fuse_virtual_legs=True."
        )
    if backward_steps is not None:
        raise ValueError(
            "backward_steps (TBPTT) is not supported on the split explicit path; "
            "set gs_explicit_ad_backward_steps=None or use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor,
    )
    from tenax.algorithms.ad_utils import ctm_split_tensor_converge_explicit

    env = ctm_split_tensor_converge_explicit(
        A,
        chi=chi,
        chi_I=chi_I,
        renormalize=renormalize,
        num_steps=backprop_steps,
        warmup_steps=warmup_steps,
    )
    return compute_energy_split_ctm_tensor(A, env, gate)


# ---------------------------------------------------------------------------
# Implicit (fixed-point) split-CTM AD (#463 Phase 2, Task 3)
# ---------------------------------------------------------------------------
#
# Custom-VJP over A -> converged split env.  Forward: gauge-fixed split-CTM
# (sweep + Γ phase-fix per iteration) run to an *element-wise* fixed point.
# Backward: Neumann series ``λ = Σ_n (J^T)^n g`` in env space, then a single
# projection to A space (variPEPS Eq. 18-19; mirrors ``ctm_tensor_converge``
# in ``ad_utils``).  Validated against the trusted explicit-AD gradient.


def _split_step(A, env, chi, chi_I, renormalize):
    """One gauge-fixed split-CTM sweep: ``Γ ∘ sweep``.

    This is the fixed-point map ``f`` whose Jacobian drives the implicit
    backward.  The Γ phase-fix is what makes the env converge element-wise
    (and hence makes ``f`` a contraction in the physical subspace).
    """
    from tenax.algorithms._split_ctm_tensor import _split_ctm_tensor_sweep
    from tenax.algorithms.ad_utils import _phase_fix_split_ctm_tensor

    env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)
    return _phase_fix_split_ctm_tensor(env)


def _split_env_max_diff(env_new, env_old) -> float:
    """Max element-wise abs difference between two split envs (all 12 tensors)."""
    leaves_new = jax.tree.leaves(env_new)
    leaves_old = jax.tree.leaves(env_old)
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves_new, leaves_old))


def _converge_split_gauge_fixed(
    A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter, env_init=None
):
    """Run gauge-fixed split-CTM to an element-wise fixed point.

    Returns the converged ``SplitCTMTensorEnv``.  Convergence is measured
    element-wise on the Γ-phase-fixed env (not the corner spectrum, which
    plateaus on the degenerate 1-site corner — see ``ctm_split_tensor``).

    When *env_init* (a ``SplitCTMTensorEnv``) is given it seeds the sweep
    (warm-start) instead of the fresh rank-1 corner init.  The Γ phase-fix
    canonicalizes the gauge each sweep, so a warm seed converges to the
    identical fixed point as a cold start — only faster.
    """
    if env_init is not None:
        env = env_init
    else:
        from tenax.algorithms._split_ctm_tensor_init import (
            initialize_split_ctm_tensor_env,
        )

        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
    prev = None
    for it in range(max_iter):
        env = _split_step(A, env, chi, chi_I, renormalize)
        if prev is not None and it + 1 >= min_iter:
            if _split_env_max_diff(env, prev) < conv_tol:
                break
        prev = env
    return env


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _split_ctm_converge(A, env_init, static):
    """Converge gauge-fixed split-CTM; custom-VJP via implicit differentiation.

    ``static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)``
    is a hashable tuple of non-differentiable CTM settings.  *env_init* is an
    optional ``SplitCTMTensorEnv`` warm-start seed (``None`` → fresh init);
    it carries **zero** gradient because the fixed point is seed-independent
    (mirrors ``env_init_leaves`` in ``ad_utils.ctm_tensor_converge``).
    Returns the converged ``SplitCTMTensorEnv`` pytree.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    return _converge_split_gauge_fixed(
        A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter, env_init=env_init
    )


def _split_ctm_converge_fwd(A, env_init, static):
    env = _split_ctm_converge(A, env_init, static)
    return env, (A, env, env_init)


def _split_ctm_converge_bwd(static, residuals, g):
    """Backward via Neumann series ``λ = Σ_n (J^T)^n g`` at the fixed point.

    ``g`` is the cotangent on the converged env (a ``SplitCTMTensorEnv``
    pytree).  We accumulate ``λ`` in env space using ``J^T = (∂f/∂env)^T``,
    then project once to A space with ``(∂f/∂A)^T λ``.  Mirrors the
    YASTN-style iterative VJP in ``ad_utils._ctm_tensor_converge_bwd``.
    The warm-start seed ``env_init`` gets a zero cotangent.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    A, env, env_init = residuals

    # J^T in env space (A fixed) and (∂f/∂A)^T projector (env fixed).
    _, vjp_env_fn = jax.vjp(lambda e: _split_step(A, e, chi, chi_I, renormalize), env)
    _, vjp_A_fn = jax.vjp(lambda a: _split_step(a, env, chi, chi_I, renormalize), A)

    max_fp_iter = min(max_iter, 50)

    grads = g  # running (J^T)^n g
    lam = g  # accumulated Neumann sum
    for _ in range(max_fp_iter):
        grads = vjp_env_fn(grads)[0]
        grads_inf = max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(grads))
        if grads_inf < conv_tol:
            break
        lam = jax.tree.map(lambda li, gi: li + gi, lam, grads)
        lam_norm = sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(lam)) ** 0.5
        if not math.isfinite(lam_norm) or lam_norm > 1e15:
            lam = jax.tree.map(lambda li, gi: li - gi, lam, grads)  # undo last
            break

    dA = vjp_A_fn(lam)[0]
    # Warm-start seed is gradient-free (fixed point is seed-independent).
    d_env_init = None if env_init is None else jax.tree.map(jnp.zeros_like, env_init)
    return (dA, d_env_init)


_split_ctm_converge.defvjp(_split_ctm_converge_fwd, _split_ctm_converge_bwd)


def ctm_energy_split_implicit(
    site_tensors,
    neighbors,
    gate,
    *,
    chi: int = 20,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
    energy_fn=None,
    env_init=None,
    **_ignored,
):
    """Single-site iPEPS energy with implicit (fixed-point) split-CTM AD.

    The split-CTM forward is run to a gauge-fixed element-wise fixed point;
    the gradient is obtained by implicit differentiation (Neumann series),
    avoiding storage of the unrolled CTM iterations.  Single-site only
    (``recipe="1x1"``); multisite has no split forward yet.

    *env_init* is an optional ``SplitCTMTensorEnv`` warm-start seed for the
    forward CTM (gradient-free); ``None`` cold-starts from a fresh init.
    """
    A = _extract_single_site(site_tensors)
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn (e.g. coarse-grain) is not supported on the split "
            "path yet; use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor,
    )

    static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)
    env = _split_ctm_converge(A, env_init, static)
    return compute_energy_split_ctm_tensor(A, env, gate)


def converge_split_env(
    A,
    *,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
    env_init=None,
):
    """Forward-only gauge-fixed split-CTM converge (no gradient).

    Returns the *same* Γ-phase-fixed element-wise fixed-point
    ``SplitCTMTensorEnv`` that :func:`ctm_energy_split_implicit`
    differentiates.  Forward-only energy evaluations on the split path
    (the optimizer's warm-start, line-search probe, and final-env eval)
    must use this rather than the bare :func:`ctm_split_tensor` so they
    land on the identical fixed point as the AD loss — keeping the
    line-search φ(α) and the gradient dφ/dα mutually consistent.

    *env_init* is an optional ``SplitCTMTensorEnv`` warm-start seed; ``None``
    cold-starts from a fresh init.
    """
    if chi_I is None:
        chi_I = chi
    return _converge_split_gauge_fixed(
        A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter, env_init=env_init
    )


# ---------------------------------------------------------------------------
# Multisite (2-site checkerboard) split-CTM AD (#463 Phase 2)
# ---------------------------------------------------------------------------
#
# The single-site machinery above is written entirely against jax.tree leaves,
# so it generalizes to a {coord: SplitCTMTensorEnv} dict pytree by swapping the
# fixed-point step for the multisite sweep and per-coord Γ phase-fix.  The
# custom_vjp now differentiates w.r.t. a {coord: Tensor} site dict.


def _split_step_multisite(site_tensors, envs, neighbors, chi, chi_I, renormalize):
    """One gauge-fixed multisite split-CTM sweep: ``Γ ∘ sweep`` per coord.

    Fixed-point map ``f`` for the coupled ``{coord: env}`` system.  The per-coord
    Γ phase-fix is what makes the joint env converge element-wise.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_sweep_multisite,
    )
    from tenax.algorithms.ad_utils import _phase_fix_split_ctm_tensor

    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _split_ctm_sweep_multisite(
        envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, recipe="2x2"
    )
    return {c: _phase_fix_split_ctm_tensor(e) for c, e in envs.items()}


def _converge_split_multisite_gauge_fixed(
    site_tensors,
    neighbors,
    chi,
    chi_I,
    max_iter,
    conv_tol,
    renormalize,
    min_iter,
    envs_init=None,
):
    """Run gauge-fixed multisite split-CTM to an element-wise fixed point.

    Convergence is measured element-wise on the Γ-phase-fixed ``{coord: env}``
    dict (all tensors, all coords), mirroring the single-site
    :func:`_converge_split_gauge_fixed`.
    """
    if envs_init is not None:
        envs = envs_init
    else:
        from tenax.algorithms._split_ctm_tensor_convergence import (
            _initialize_split_multisite_env,
        )

        envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    prev = None
    for it in range(max_iter):
        envs = _split_step_multisite(
            site_tensors, envs, neighbors, chi, chi_I, renormalize
        )
        if prev is not None and it + 1 >= min_iter:
            if _split_env_max_diff(envs, prev) < conv_tol:
                break
        prev = envs
    return envs


def converge_split_env_2site(
    site_tensors,
    neighbors,
    *,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
    envs_init=None,
):
    """Forward-only gauge-fixed 2-site split-CTM converge (no gradient).

    Returns the ``{coord: SplitCTMTensorEnv}`` dict at the same Γ-phase-fixed
    element-wise fixed point that :func:`ctm_energy_split_implicit_2site`
    differentiates.  Forward-only energy evaluations on the 2-site split path
    (optimizer warm-start, line-search probe, final-env eval) must use this so
    the line-search φ(α) and the gradient dφ/dα stay mutually consistent —
    the 2-site analogue of :func:`converge_split_env`.
    """
    if chi_I is None:
        chi_I = chi
    return _converge_split_multisite_gauge_fixed(
        site_tensors,
        neighbors,
        chi,
        chi_I,
        max_iter,
        conv_tol,
        renormalize,
        min_iter,
        envs_init=envs_init,
    )


def _explicit_split_multisite_converge(
    site_tensors,
    neighbors,
    *,
    chi,
    chi_I=None,
    renormalize=True,
    warmup_steps=0,
    backprop_steps=20,
):
    """Explicit (unrolled) multisite split-CTM converge for warm-start AD.

    ``warmup_steps`` sweeps run under ``stop_gradient``; ``backprop_steps``
    sweeps are fully differentiable.  Uses the raw multisite sweep (no Γ
    phase-fix) — the energy is gauge-invariant, matching the single-site
    :func:`ad_utils.ctm_split_tensor_converge_explicit`.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _initialize_split_multisite_env,
        _split_ctm_sweep_multisite,
    )

    if chi_I is None:
        chi_I = chi
    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    for _ in range(warmup_steps):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, "2x2"
        )
    if warmup_steps > 0:
        envs = jax.tree.map(jax.lax.stop_gradient, envs)
    for _ in range(backprop_steps):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, "2x2"
        )
    return envs


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static):
    """Converge gauge-fixed multisite split-CTM; custom-VJP via implicit diff.

    ``static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)``.
    Differentiates w.r.t. the ``{coord: Tensor}`` site dict; ``envs_init`` is a
    gradient-free warm-start seed (fixed point is seed-independent).  Returns the
    converged ``{coord: SplitCTMTensorEnv}`` dict.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    return _converge_split_multisite_gauge_fixed(
        site_tensors,
        neighbors,
        chi,
        chi_I,
        max_iter,
        conv_tol,
        renormalize,
        min_iter,
        envs_init=envs_init,
    )


def _split_ctm_converge_multisite_fwd(site_tensors, envs_init, neighbors, static):
    envs = _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static)
    return envs, (site_tensors, envs, envs_init)


def _split_ctm_converge_multisite_bwd(neighbors, static, residuals, g):
    """Backward via Neumann series ``λ = Σ_n (J^T)^n g`` at the coupled fixed point.

    ``g`` is the cotangent on the converged ``{coord: env}`` dict.  Accumulate λ
    in env space with ``J^T = (∂f/∂envs)^T`` (site dict fixed), then project once
    to site space with ``(∂f/∂site_tensors)^T λ``.  Direct mirror of the
    single-site :func:`_split_ctm_converge_bwd`, now over the dict pytree.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    site_tensors, envs, envs_init = residuals

    _, vjp_env_fn = jax.vjp(
        lambda e: _split_step_multisite(
            site_tensors, e, neighbors, chi, chi_I, renormalize
        ),
        envs,
    )
    _, vjp_site_fn = jax.vjp(
        lambda s: _split_step_multisite(s, envs, neighbors, chi, chi_I, renormalize),
        site_tensors,
    )

    max_fp_iter = min(max_iter, 50)
    grads = g
    lam = g
    for _ in range(max_fp_iter):
        grads = vjp_env_fn(grads)[0]
        grads_inf = max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(grads))
        if grads_inf < conv_tol:
            break
        lam = jax.tree.map(lambda li, gi: li + gi, lam, grads)
        lam_norm = sum(float(jnp.sum(x**2)) for x in jax.tree.leaves(lam)) ** 0.5
        if not math.isfinite(lam_norm) or lam_norm > 1e15:
            lam = jax.tree.map(lambda li, gi: li - gi, lam, grads)
            break

    d_site = vjp_site_fn(lam)[0]
    d_envs_init = None if envs_init is None else jax.tree.map(jnp.zeros_like, envs_init)
    return (d_site, d_envs_init)


_split_ctm_converge_multisite.defvjp(
    _split_ctm_converge_multisite_fwd, _split_ctm_converge_multisite_bwd
)


def ctm_energy_split_explicit_2site(
    site_tensors,
    neighbors,
    gate,
    *,
    chi=20,
    warmup_steps=3,
    backprop_steps=20,
    chi_I=None,
    renormalize=True,
    energy_fn=None,
    **_ignored,
):
    """2-site checkerboard iPEPS energy with explicit (unrolled) split-CTM AD."""
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn is not supported on the split path; "
            "use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_multisite,
    )

    envs = _explicit_split_multisite_converge(
        site_tensors,
        neighbors,
        chi=chi,
        chi_I=chi_I,
        renormalize=renormalize,
        warmup_steps=warmup_steps,
        backprop_steps=backprop_steps,
    )
    return compute_energy_split_ctm_tensor_multisite(
        site_tensors, envs, neighbors, gate
    )


def ctm_energy_split_implicit_2site(
    site_tensors,
    neighbors,
    gate,
    *,
    chi=20,
    max_iter=100,
    conv_tol=1e-8,
    chi_I=None,
    renormalize=True,
    min_iter=2,
    energy_fn=None,
    envs_init=None,
    **_ignored,
):
    """2-site checkerboard iPEPS energy with implicit (fixed-point) split-CTM AD.

    The coupled ``(env_A, env_B)`` forward is run to a gauge-fixed element-wise
    fixed point; the gradient comes from implicit differentiation (Neumann
    series) over the joint ``{coord: env}`` pytree.  *envs_init* is an optional
    gradient-free ``{coord: SplitCTMTensorEnv}`` warm-start seed.
    """
    if energy_fn is not None:
        raise NotImplementedError(
            "custom energy_fn is not supported on the split path; "
            "use fuse_virtual_legs=True."
        )
    if chi_I is None:
        chi_I = chi

    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_multisite,
    )

    static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)
    envs = _split_ctm_converge_multisite(site_tensors, envs_init, neighbors, static)
    return compute_energy_split_ctm_tensor_multisite(
        site_tensors, envs, neighbors, gate
    )
