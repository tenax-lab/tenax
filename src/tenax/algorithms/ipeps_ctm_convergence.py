"""CTM algorithm for iPEPS — sweep loops, convergence, and entry points."""

from __future__ import annotations

__all__ = [
    "_ctm_sweep",
    "_ctm_sv_diff",
    "CTMConvergenceInfo",
    "ctm",
    "_ctm_via_2x2",
    "_renormalize_env",
    "_ctm_2site_sweep",
    "ctm_2site",
    "_split_ctm_sweep",
    "_split_env_to_standard",
    "ctm_split",
]


import warnings

import jax
import jax.numpy as jnp

# Defined in ``_ctm_tensor_convergence`` rather than here, and re-exported so
# every existing import path keeps working.  It lives in the lower module
# because ``ctm_tensor`` returns it too (#919) and this module already imports
# *from* that one -- defining it here forced ``ctm_tensor``'s annotation to be
# TYPE_CHECKING-only, which broke ``typing.get_type_hints`` on public API.
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: F401
    CTMConvergenceInfo as CTMConvergenceInfo,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_sv_diff,
    _forced_corner_rank,
    _max_virtual_bond_dim,
)
from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
)
from tenax.algorithms.ipeps_ctm_init import (
    _build_double_layer,
    _initialize_ctm_env,
    _initialize_split_ctm_env,
)
from tenax.algorithms.ipeps_ctm_moves import (
    _ctm_bottom_move,
    _ctm_bottom_move_2site,
    _ctm_left_move,
    _ctm_left_move_2site,
    _ctm_right_move,
    _ctm_right_move_2site,
    _ctm_top_move,
    _ctm_top_move_2site,
    _split_ctm_move,
)
from tenax.core import EPS
from tenax.linalg import _dense_svd


def _ctm_sweep(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    renormalize: bool,
    projector_method: str = "eigh",
) -> CTMEnvironment:
    """One full CTM sweep: left, right, top, bottom moves + optional renormalize."""
    env = _ctm_left_move(env, a, chi, projector_method)
    env = _ctm_right_move(env, a, chi, projector_method)
    env = _ctm_top_move(env, a, chi, projector_method)
    env = _ctm_bottom_move(env, a, chi, projector_method)
    if renormalize:
        env = _renormalize_env(env)
    return env


# ``_ctm_sv_diff`` is imported, not defined here.
#
# This module used to carry its own copy, and the copy is what the **public**
# path ran: ``ipeps()`` imports ``ctm_2site`` from here (``ipeps.py:30``, called
# at ``:462``), so #898's guard in ``_ctm_tensor_convergence`` protected the
# tensor loops while every ``ipeps()`` call kept the unguarded comparison --
# the rank-1 corner still normalised to ``[1, 0, ..., 0]``, still compared
# equal to an unrelated environment, and still reported convergence early.
# A fix that leaves the production path broken is worse than no fix, because
# the closed issue stops anyone looking.
#
# The guarded implementation is written to be shared: it uses ``jnp.where``
# rather than a Python ``if`` precisely so it can run inside the
# ``jax.lax.while_loop`` in this module, and its own docstring says the guard
# belongs in one place so "a future tenth loop inherits the guard instead of
# re-acquiring the bug".  This is that inheritance, and it also picks up the
# #670 zero-padding, which is a no-op on the dense path where the spectrum
# length is fixed by ``chi``.


def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
    *,
    return_meta: bool = False,
) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]:
    """Compute the CTM environment of a 1x1 PEPS with the 2x2 recipe.

    .. deprecated:: 0.9.3
        Use :func:`ctm_tensor`.  ``ctm()`` now runs ``ctm_tensor(recipe="2x2")``
        and returns its environment as the raw-array :class:`CTMEnvironment`
        that ``ctm()`` always returned.  The 1-site sweep it ran before reached
        no fixed point: on the Ising PEPS at beta=0.2 it stopped at chi=12, 16
        and 32 without converging, with C(1,0) errors of 7.6e-4, 4.3e-4 and
        4.3e-4, where the 2x2 recipe converges with errors below 4e-16.  So
        environments from ``ctm()`` changed in 0.9.3, and ``ctm()`` no longer
        runs under ``jax.jit``, because ``ctm_tensor`` is a Python loop.

    Args:
        A:           Raw site array with 5 legs ``(u, d, l, r, s)`` and four
                     virtual legs of one dimension, or a ``DenseTensor``.  Any
                     other shape raises ``ValueError``.  A ``SymmetricTensor``
                     raises ``TypeError``: call :func:`ctm_tensor`, which keeps
                     the blocks.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` apply.  A ``projector_method`` other than
                     ``"svd"`` is ignored with a ``UserWarning``, because the
                     2x2 recipe always uses Fishman SVD; ``qr_warmup_steps`` is
                     ignored too.
        initial_env: Must be ``None``.  ``ctm_tensor`` has no warm start, so
                     anything else raises ``ValueError`` rather than start cold
                     in silence.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or exhausted
                     ``max_iter`` (#839).  Its fields are Python scalars.
                     Keyword-only and off by default so the return arity of
                     this public entry point is unchanged.

    Returns:
        The CTMEnvironment, or ``(env, info)`` when ``return_meta=True``.
        **The environment is not necessarily converged** -- that is what
        ``info`` is for.
    """
    warnings.warn(
        "ctm is deprecated and will be removed in a later release: call "
        "ctm_tensor(A, chi, ...) on a DenseTensor or SymmetricTensor site tensor "
        "with legs (u, d, l, r, phys). ctm now runs ctm_tensor(recipe='2x2'), so "
        "its environments changed in 0.9.3 (the old 1-site sweep reached no "
        "fixed point), and it no longer runs under jax.jit.",
        DeprecationWarning,
        stacklevel=2,
    )
    if initial_env is not None:
        raise ValueError(
            "ctm() no longer takes initial_env: it runs ctm_tensor(recipe='2x2'), "
            "which has no warm start. Leave initial_env as None."
        )
    if config.projector_method != "svd":
        warnings.warn(
            f"ctm() ignores projector_method={config.projector_method!r}: the 2x2 "
            "recipe always uses Fishman SVD projectors.",
            UserWarning,
            stacklevel=2,
        )
    return _ctm_via_2x2(A, config, return_meta=return_meta)


def _ctm_via_2x2(
    A: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]:
    """``ctm()`` without its warnings: ``ctm_tensor(recipe="2x2")`` with the
    ``CTMConfig`` fields mapped, returned as a raw-array ``CTMEnvironment``.

    For callers that need ``ctm()``'s result and must not trigger its
    ``DeprecationWarning``.

    Args:
        A:           Raw site array with 5 legs ``(u, d, l, r, s)`` and four
                     virtual legs of one dimension, or a ``DenseTensor``, which
                     is passed through.  ``_wrap_as_dense_tensor`` reads
                     ``shape[0]`` for all four virtual legs without a check, so
                     any other shape raises ``ValueError`` here.  A
                     ``SymmetricTensor`` raises ``TypeError``: the raw result
                     would need ``todense()``.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` are passed on.  ``projector_method`` is not:
                     the 2x2 recipe always uses Fishman SVD, and ``"svd"`` is
                     passed because ``"qr"`` would still start the QR warm-up
                     loop of ``ctm_tensor``.  So ``qr_warmup_steps`` has no
                     effect.
        return_meta: When True, also return the :class:`CTMConvergenceInfo`
                     of ``ctm_tensor``, whose fields are Python scalars.

    Returns:
        The raw-array ``CTMEnvironment``, or ``(env, info)``.  Every field has
        its legs in the order ``CTMTensorEnv`` documents, which is the order the
        raw consumers contract.
    """
    from tenax.algorithms._ctm_env_dense import _as_dense_env, _in_documented_leg_order
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.core.tensor import SymmetricTensor, Tensor

    if isinstance(A, SymmetricTensor):
        raise TypeError(
            "ctm() returns a raw-array CTMEnvironment, so a SymmetricTensor site "
            "tensor would have to be densified. Call ctm_tensor(A, chi, ...) "
            "instead: it keeps the blocks and returns a CTMTensorEnv."
        )
    if not isinstance(A, Tensor):
        A = jnp.asarray(A)
        if A.ndim != 5 or len(set(A.shape[:4])) != 1:
            raise ValueError(
                "Expected a raw site tensor with 5 legs (u, d, l, r, s) and four "
                f"virtual legs of one dimension; got shape {A.shape}."
            )
        A = _wrap_as_dense_tensor(A)
    env, _eps, info = ctm_tensor(
        A,
        config.chi,
        max_iter=config.max_iter,
        conv_tol=config.conv_tol,
        renormalize=config.renormalize,
        projector_method="svd",
        recipe="2x2",
        min_iter=config.min_iter,
        return_meta=True,
    )
    # The 2x2 sweep stores each corner with its two legs reversed, and the raw
    # consumers contract by position: put every field in the documented order.
    env = _as_dense_env(_in_documented_leg_order(env))
    if return_meta:
        return env, info
    return env


def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:
    """Normalize environment tensors to prevent exponential growth."""

    def normalize(x: jax.Array) -> jax.Array:
        norm = jnp.max(jnp.abs(x))
        return x / (norm + EPS)

    return CTMEnvironment(
        C1=normalize(env.C1),
        C2=normalize(env.C2),
        C3=normalize(env.C3),
        C4=normalize(env.C4),
        T1=normalize(env.T1),
        T2=normalize(env.T2),
        T3=normalize(env.T3),
        T4=normalize(env.T4),
    )


def _ctm_2site_sweep(
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    a_A: jax.Array,
    a_B: jax.Array,
    chi: int,
    # NOTE: Legacy dense 2-site sweep, used by simple update only.
    # AD optimization uses _ctm_tensor_sweep_multisite instead.
    renormalize: bool,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """One full 2-site CTM sweep: L/R/T/B moves for both sublattices + renormalize."""
    # Left moves
    env_A = _ctm_left_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_left_move_2site(env_B, env_A, a_A, chi)
    # Right moves
    env_A = _ctm_right_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_right_move_2site(env_B, env_A, a_A, chi)
    # Top moves
    env_A = _ctm_top_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_top_move_2site(env_B, env_A, a_A, chi)
    # Bottom moves
    env_A = _ctm_bottom_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_bottom_move_2site(env_B, env_A, a_A, chi)
    if renormalize:
        env_A = _renormalize_env(env_A)
        env_B = _renormalize_env(env_B)
    return env_A, env_B


def ctm_2site(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> (
    tuple[CTMEnvironment, CTMEnvironment]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo]
):
    """Compute CTM environments for a 2-site checkerboard unit cell.

    .. note::
        This is the **legacy dense** 2-site CTM used by simple update
        (``ipeps()``).  For AD-based optimization, use
        ``optimize_gs_ad()`` with ``unit_cell="2site"``, which routes
        through the Tensor-protocol multisite CTM
        (``ctm_tensor_converge_2site``).

    On a checkerboard, all neighbors of A are B and vice versa. Each
    absorption move for env_A uses B's double-layer tensor and T's from
    env_B, and vice versa.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A: Site tensor for sublattice A, shape (D, D, D, D, d).
        B: Site tensor for sublattice B, shape (D, D, D, D, d).
        config: CTMConfig.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or exhausted
                     ``max_iter`` (#839).  Keyword-only and off by default so
                     the return arity of this public entry point is unchanged.

    Returns:
        ``(env_A, env_B)``, or ``(env_A, env_B, info)`` when
        ``return_meta=True``.  **The environments are not necessarily
        converged** -- that is what ``info`` is for.  A single ``info``
        covers both sublattices because the loop's criterion is their
        maximum, so they converge or fail together.
    """
    chi = config.chi

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    D_A = A.shape[0]
    D_B = B.shape[0]
    if a_A.ndim == 8:
        a_A = a_A.reshape(D_A**2, D_A**2, D_A**2, D_A**2)
    if a_B.ndim == 8:
        a_B = a_B.reshape(D_B**2, D_B**2, D_B**2, D_B**2)

    env_A = _initialize_ctm_env(a_A, chi)
    env_B = _initialize_ctm_env(a_B, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    # See ``ctm`` above: cap ``min_iter`` at the budget so a small ``max_iter``
    # stays convergeable (#976).
    min_iter = min(config.min_iter, max_iter)
    renormalize = config.renormalize

    # Initial singular values (zeros — first iteration never converges).
    # Seeded with the REAL dtype, not ``C1.dtype``, for the same reason as in
    # ``ctm`` above: ``body_fn`` assigns ``_dense_svd(..., compute_uv=False)``,
    # which is always real, and ``lax.while_loop`` requires an invariant carry
    # type.  ``ctm`` was fixed for this; ``ctm_2site`` was not, so it raised
    # ``TypeError`` on any complex site tensor -- including through public
    # ``ipeps()`` with a complex ``initial_peps`` (#842).
    _sv_dtype_A = jnp.zeros((), dtype=env_A.C1.dtype).real.dtype
    _sv_dtype_B = jnp.zeros((), dtype=env_B.C1.dtype).real.dtype
    prev_sv_A = jnp.zeros(min(chi, env_A.C1.shape[0]), dtype=_sv_dtype_A)
    prev_sv_B = jnp.zeros(min(chi, env_B.C1.shape[0]), dtype=_sv_dtype_B)
    # See ``ctm`` above (#903 review, P1).  Both sublattices share one bound:
    # the smaller, so neither is certified on the other's headroom.
    # Per sublattice, not aggregated (#903 review): each corner is judged by
    # the reachable rank of the site sitting under it.
    max_rank_A = _forced_corner_rank(_max_virtual_bond_dim(a_A))
    max_rank_B = _forced_corner_rank(_max_virtual_bond_dim(a_B))

    # Carry: (env_A, env_B, prev_sv_A, prev_sv_B, iteration, converged, diff)
    # ``diff`` rides along only so it can be reported (#839); ``converged`` is
    # still derived from it in the body, so loop behaviour is unchanged.
    # Seeded at inf -- no comparison exists before two sweeps, and inf never
    # satisfies ``< conv_tol``.  Its dtype is the promotion of the two
    # sublattice singular-value dtypes because the body reports
    # ``max(diff_A, diff_B)``, which promotes.
    _diff_dtype = jnp.result_type(_sv_dtype_A, _sv_dtype_B)
    init_carry = (
        env_A,
        env_B,
        prev_sv_A,
        prev_sv_B,
        jnp.array(0, dtype=jnp.int32),
        jnp.bool_(False),
        jnp.array(jnp.inf, dtype=_diff_dtype),
    )

    def cond_fn(carry):
        _, _, _, _, iteration, converged, _ = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        eA, eB, psA, psB, iteration, _, _ = carry
        eA, eB = _ctm_2site_sweep(eA, eB, a_A, a_B, chi, renormalize)
        sv_A = _dense_svd(eA.C1, compute_uv=False)
        sv_B = _dense_svd(eB.C1, compute_uv=False)
        diff_A = _ctm_sv_diff(sv_A, psA, max_rank=max_rank_A)
        diff_B = _ctm_sv_diff(sv_B, psB, max_rank=max_rank_B)
        diff = jnp.maximum(diff_A, diff_B)
        # Honor ``min_iter`` before certifying convergence (#976).  No warm-up
        # on this path, so the loop count is the total sweep count.
        converged = (diff < conv_tol) & ((iteration + 1) >= min_iter)
        return (eA, eB, sv_A, sv_B, iteration + 1, converged, diff)

    env_A, env_B, _, _, n_iter, converged, diff = jax.lax.while_loop(
        cond_fn, body_fn, init_carry
    )
    if return_meta:
        return (
            env_A,
            env_B,
            CTMConvergenceInfo(converged=converged, n_iter=n_iter, diff=diff),
        )
    return env_A, env_B


# ---------------------------------------------------------------------------
# Split-CTMRG: ket/bra layers kept separate (arXiv:2502.10298)
# ---------------------------------------------------------------------------


def _split_ctm_sweep(
    env: SplitCTMEnvironment,
    A: jax.Array,
    chi: int,
    chi_I: int,
    renormalize: bool,
) -> SplitCTMEnvironment:
    """One full split-CTM sweep: L/R/T/B moves + optional renormalize."""
    env = _split_ctm_move(env, A, chi, chi_I, "left")
    env = _split_ctm_move(env, A, chi, chi_I, "right")
    env = _split_ctm_move(env, A, chi, chi_I, "top")
    env = _split_ctm_move(env, A, chi, chi_I, "bottom")
    if renormalize:

        def normalize(x: jax.Array) -> jax.Array:
            norm = jnp.max(jnp.abs(x))
            return x / (norm + EPS)

        def normalize_pair(
            T_ket: jax.Array, T_bra: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            """Normalize ket/bra pair using a shared factor.

            Uses the geometric mean of the max-abs norms to preserve the
            relative scaling set by the SVD split.
            """
            nk = jnp.max(jnp.abs(T_ket))
            nb = jnp.max(jnp.abs(T_bra))
            shared = jnp.sqrt(nk * nb) + EPS
            return T_ket / shared, T_bra / shared

        T1k, T1b = normalize_pair(env.T1_ket, env.T1_bra)
        T2k, T2b = normalize_pair(env.T2_ket, env.T2_bra)
        T3k, T3b = normalize_pair(env.T3_ket, env.T3_bra)
        T4k, T4b = normalize_pair(env.T4_ket, env.T4_bra)

        env = SplitCTMEnvironment(
            C1=normalize(env.C1),
            C2=normalize(env.C2),
            C3=normalize(env.C3),
            C4=normalize(env.C4),
            T1_ket=T1k,
            T1_bra=T1b,
            T2_ket=T2k,
            T2_bra=T2b,
            T3_ket=T3k,
            T3_bra=T3b,
            T4_ket=T4k,
            T4_bra=T4b,
        )
    return env


def _split_env_to_standard(
    env: SplitCTMEnvironment,
) -> CTMEnvironment:
    """Convert SplitCTMEnvironment to standard CTMEnvironment.

    Contracts each ``(T_ket, T_bra)`` pair over the interlayer bond::

        T_full[a, (uU), b] = sum_c T_ket[a, u, c] * T_bra[c, U, b]
    """
    chi = env.C1.shape[0]

    def merge(T_ket, T_bra):
        D = T_ket.shape[1]
        T = jnp.einsum("auc,cUb->auUb", T_ket, T_bra)
        return T.reshape(chi, D * D, chi)

    return CTMEnvironment(
        C1=env.C1,
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=merge(env.T1_ket, env.T1_bra),
        T2=merge(env.T2_ket, env.T2_bra),
        T3=merge(env.T3_ket, env.T3_bra),
        T4=merge(env.T4_ket, env.T4_bra),
    )


def ctm_split(
    A: jax.Array,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> SplitCTMEnvironment | tuple[SplitCTMEnvironment, CTMConvergenceInfo]:
    """Compute split-CTM environment for a PEPS with 1x1 unit cell.

    Uses the split-CTMRG algorithm (arXiv:2502.10298) where ket and bra
    layers are kept separate, reducing projector cost from O(chi^3 * D^6)
    to O(chi^3 * D^3).

    Args:
        A:      Site tensor of shape ``(D, D, D, D, d)``.
        config: CTMConfig with ``chi_I`` set.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or ran out of
                     iterations (#839).  Keyword-only and off by default so
                     the return arity is unchanged.

    Returns:
        The SplitCTMEnvironment, or ``(env, info)`` when ``return_meta=True``.
        **The environment is not necessarily converged** -- that is what
        ``info`` is for.  Unlike :func:`ctm` / :func:`ctm_2site` this is a
        Python loop, so ``info``'s fields are Python scalars.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi
    # See ``ctm`` above: cap ``min_iter`` at the budget (#976).
    min_iter = min(config.min_iter, config.max_iter)

    env = _initialize_split_ctm_env(A, chi, chi_I)

    prev_sv = None
    converged = False
    n_iter = 0
    diff_val = float("inf")
    for _ in range(config.max_iter):
        env = _split_ctm_sweep(env, A, chi, chi_I, config.renormalize)
        n_iter += 1

        current_sv = _dense_svd(env.C1, compute_uv=False)
        if prev_sv is not None:
            diff_val = float(
                _ctm_sv_diff(
                    current_sv,
                    prev_sv,
                    max_rank=_forced_corner_rank(_max_virtual_bond_dim(A) ** 2),
                )
            )
            # Honor ``min_iter`` before breaking on convergence (#976); keep
            # updating ``prev_sv`` through the minimum window (no warm-up here).
            if diff_val < config.conv_tol and n_iter >= min_iter:
                converged = True
                break
        prev_sv = current_sv

    if return_meta:
        return env, CTMConvergenceInfo(
            converged=converged, n_iter=n_iter, diff=diff_val
        )
    return env
