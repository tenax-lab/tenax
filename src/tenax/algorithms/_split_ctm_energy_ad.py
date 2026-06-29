"""Single-site split-CTM AD energy entry points (#463 Phase 2).

Mirrors ``ctm_energy_explicit`` / ``ctm_energy_implicit`` from
``_ctm_energy_ad`` but on the split (ket/bra-separate) double layer.
Single-site only: the split forward (``ctm_split_tensor``) converges one
site as an isolated 1×1 iPEPS.  Multisite has no split forward yet.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

__all__ = [
    "converge_split_env",
    "ctm_energy_split_explicit",
    "ctm_energy_split_implicit",
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
    A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter
):
    """Run gauge-fixed split-CTM to an element-wise fixed point.

    Returns the converged ``SplitCTMTensorEnv``.  Convergence is measured
    element-wise on the Γ-phase-fixed env (not the corner spectrum, which
    plateaus on the degenerate 1-site corner — see ``ctm_split_tensor``).
    """
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


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _split_ctm_converge(A, static):
    """Converge gauge-fixed split-CTM; custom-VJP via implicit differentiation.

    ``static = (chi, chi_I, max_iter, conv_tol, renormalize, min_iter)``
    is a hashable tuple of non-differentiable CTM settings.  Returns the
    converged ``SplitCTMTensorEnv`` pytree.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    return _converge_split_gauge_fixed(
        A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter
    )


def _split_ctm_converge_fwd(A, static):
    env = _split_ctm_converge(A, static)
    return env, (A, env)


def _split_ctm_converge_bwd(static, residuals, g):
    """Backward via Neumann series ``λ = Σ_n (J^T)^n g`` at the fixed point.

    ``g`` is the cotangent on the converged env (a ``SplitCTMTensorEnv``
    pytree).  We accumulate ``λ`` in env space using ``J^T = (∂f/∂env)^T``,
    then project once to A space with ``(∂f/∂A)^T λ``.  Mirrors the
    YASTN-style iterative VJP in ``ad_utils._ctm_tensor_converge_bwd``.
    """
    chi, chi_I, max_iter, conv_tol, renormalize, min_iter = static
    A, env = residuals

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
    return (dA,)


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
    **_ignored,
):
    """Single-site iPEPS energy with implicit (fixed-point) split-CTM AD.

    The split-CTM forward is run to a gauge-fixed element-wise fixed point;
    the gradient is obtained by implicit differentiation (Neumann series),
    avoiding storage of the unrolled CTM iterations.  Single-site only
    (``recipe="1x1"``); multisite has no split forward yet.
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
    env = _split_ctm_converge(A, static)
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
):
    """Forward-only gauge-fixed split-CTM converge (no gradient).

    Returns the *same* Γ-phase-fixed element-wise fixed-point
    ``SplitCTMTensorEnv`` that :func:`ctm_energy_split_implicit`
    differentiates.  Forward-only energy evaluations on the split path
    (the optimizer's warm-start, line-search probe, and final-env eval)
    must use this rather than the bare :func:`ctm_split_tensor` so they
    land on the identical fixed point as the AD loss — keeping the
    line-search φ(α) and the gradient dφ/dα mutually consistent.
    """
    if chi_I is None:
        chi_I = chi
    return _converge_split_gauge_fixed(
        A, chi, chi_I, max_iter, conv_tol, renormalize, min_iter
    )
