"""Split CTM with Tensor protocol — sweep loop, convergence, and main entry."""

from __future__ import annotations

__all__ = [
    "_renormalize_split_env",
    "_split_ctm_tensor_sweep",
    "ctm_split_tensor",
]

import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    _corner_singular_values,
    _ctm_sv_diff,
)
from tenax.algorithms._split_ctm_tensor_init import (
    SplitCTMTensorEnv,
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms._split_ctm_tensor_moves import (
    _split_ctm_move_bottom,
    _split_ctm_move_left,
    _split_ctm_move_right,
    _split_ctm_move_top,
)
from tenax.algorithms._tensor_utils import max_abs_normalize
from tenax.core import EPS
from tenax.core.tensor import Tensor

# ------------------------------------------------------------------ #
# Sweep + convergence                                                  #
# ------------------------------------------------------------------ #


def _split_ctm_tensor_sweep(
    env: SplitCTMTensorEnv,
    A: Tensor,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMTensorEnv:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    A_bar = A.bar()
    # variPEPS sweep order (L/T/R/B), matching the fused ``_ctm_tensor_sweep``
    # so the split path tracks the same fixed point as the oracle.
    env = _split_ctm_move_left(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_top(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_right(env, A, A_bar, chi, chi_I)
    env = _split_ctm_move_bottom(env, A, A_bar, chi, chi_I)

    if renormalize:
        env = _renormalize_split_env(env)

    return env


def _renormalize_split_env(env: SplitCTMTensorEnv) -> SplitCTMTensorEnv:
    """Renormalize all 12 tensors in a SplitCTMTensorEnv."""
    C1, _ = max_abs_normalize(env.C1)
    C2, _ = max_abs_normalize(env.C2)
    C3, _ = max_abs_normalize(env.C3)
    C4, _ = max_abs_normalize(env.C4)

    def normalize_pair(T_ket: Tensor, T_bra: Tensor) -> tuple[Tensor, Tensor]:
        nk = T_ket.max_abs()
        nb = T_bra.max_abs()
        shared = jnp.sqrt(nk * nb) + EPS
        return T_ket * (1.0 / shared), T_bra * (1.0 / shared)

    T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
    T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
    T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
    T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

    return SplitCTMTensorEnv(
        C1=C1,
        C2=C2,
        C3=C3,
        C4=C4,
        T1_ket=T1k,
        T1_bra=T1b,
        T2_ket=T2k,
        T2_bra=T2b,
        T3_ket=T3k,
        T3_bra=T3b,
        T4_ket=T4k,
        T4_bra=T4b,
    )


def ctm_split_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    min_iter: int = 2,
) -> SplitCTMTensorEnv:
    """Run split-CTM to convergence using the Tensor protocol.

    Convergence is measured on the corner (``C1``) singular-value spectrum
    via the same :func:`_corner_singular_values` / :func:`_ctm_sv_diff`
    helpers the fused :func:`ctm_tensor` uses, so the split path tracks the
    same fixed point as the oracle.

    .. note::
        The corner singular-value criterion can **plateau** for degenerate
        or low-rank corners — e.g. the boundary of a 1-site uniform iPEPS is
        genuinely rank-1/2, so the *normalized* corner spectrum is constant
        from the first sweep even while the environment (and energy) are
        still relaxing toward the fixed point. The ``min_iter`` floor stops
        the loop from breaking on that initial transient, but it cannot
        detect convergence beyond it. For exact convergence studies on such
        inputs (e.g. the lossless-``chi_I`` parity oracle), pass
        ``conv_tol=0.0`` with a sufficiently large ``max_iter`` to force a
        fixed number of full sweeps instead of relying on the spectral break.

    Args:
        A:          iPEPS site tensor (DenseTensor or SymmetricTensor) with
                    5 legs ``(u, d, l, r, phys)``.
        chi:        Environment bond dimension.
        max_iter:   Maximum number of CTM iterations.
        conv_tol:   Convergence tolerance on corner singular values. Use
                    ``0.0`` to disable the early break (run all ``max_iter``
                    sweeps).
        chi_I:      Interlayer bond dimension. Defaults to ``chi``.
        renormalize: Renormalize environment at each step.
        min_iter:   Minimum number of sweeps before the ``conv_tol`` early
                    break may fire. Guards against a premature break on the
                    initial transient plateau of a degenerate corner.
                    Effectively floored at 2: the first sweep has no previous
                    spectrum to compare against, so the earliest possible break
                    is the second sweep regardless of ``min_iter``.

    Returns:
        Converged SplitCTMTensorEnv.
    """
    if chi_I is None:
        chi_I = chi

    env = initialize_split_ctm_tensor_env(A, chi, chi_I)

    prev_sv = None
    for iteration in range(max_iter):
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, renormalize)

        current_sv = _corner_singular_values(env.C1)
        if prev_sv is not None and iteration + 1 >= min_iter:
            diff = _ctm_sv_diff(current_sv, prev_sv)
            if float(diff) < conv_tol:
                break
        prev_sv = current_sv

    return env
