"""CTM algorithm for iPEPS — sweep loops, convergence, and entry points."""

from __future__ import annotations

__all__ = [
    "_ctm_sv_diff",
    "CTMConvergenceInfo",
    "ctm",
    "_ctm_via_2x2",
    "_ctm_2site_via_2x2",
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
    _initialize_split_ctm_env,
)
from tenax.algorithms.ipeps_ctm_moves import _split_ctm_move
from tenax.core import EPS
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import _dense_svd

# ``_ctm_sv_diff`` is imported, not defined here.
#
# This module used to carry its own copy, and the copy is what the **public**
# path ran: until 0.9.3 ``ipeps()`` measured with ``ctm_2site`` from here, so
# #898's guard in ``_ctm_tensor_convergence`` protected the tensor loops while
# every ``ipeps()`` call kept the unguarded comparison -- the rank-1 corner
# still normalised to ``[1, 0, ..., 0]``, still compared equal to an unrelated
# environment, and still reported convergence early.  A fix that leaves the
# production path broken is worse than no fix, because the closed issue stops
# anyone looking.
#
# The guarded implementation is written to be shared: it uses ``jnp.where``
# rather than a Python ``if`` precisely so it could run inside the
# ``jax.lax.while_loop`` that ``ctm`` and ``ctm_2site`` ran before 0.9.3, and
# its own docstring says the guard belongs in one place so "a future tenth
# loop inherits the guard instead of re-acquiring the bug".  ``ctm_split``,
# the one loop left here, is that inheritance, and it also picks up the #670
# zero-padding, which is a no-op on the dense path where the spectrum length
# is fixed by ``chi``.


def ctm(
    A: jax.Array | DenseTensor,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
    *,
    return_meta: bool = False,
) -> CTMEnvironment | tuple[CTMEnvironment, CTMConvergenceInfo]:
    """Compute the CTM environment of a 1x1 PEPS with the 2x2 recipe.

    .. deprecated:: 0.9.3
        Use :func:`ctm_tensor`.  ``ctm()`` now runs ``ctm_tensor(recipe="2x2")``
        and returns its environment as the raw-array :class:`CTMEnvironment`
        that ``ctm()`` always returned.  The 1-site sweep it ran before often
        did not converge: on the Ising PEPS at beta=0.2 it stopped at chi=12,
        16 and 32 without converging, with C(1,0) errors of 7.6e-4, 4.3e-4 and
        4.3e-4, where the 2x2 recipe converges and agrees with Onsager's exact
        value to roundoff (below 1e-15).  So environments from ``ctm()``
        changed in 0.9.3, and ``ctm()`` no longer runs under ``jax.jit``,
        because ``ctm_tensor`` is a Python loop.

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
        "its environments changed in 0.9.3 (the old 1-site sweep often did not "
        "converge), and it no longer runs under jax.jit.",
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
    A: jax.Array | DenseTensor,
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
    from tenax.algorithms._ctm_env_dense import _as_dense_env
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor

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
    # consumers contract by position: _as_dense_env puts every field in the
    # documented order, by label.
    env = _as_dense_env(env)
    if return_meta:
        return env, info
    return env


def _ctm_2site_via_2x2(
    A: jax.Array | DenseTensor,
    B: jax.Array | DenseTensor,
    config: CTMConfig,
    *,
    return_meta: bool = False,
    _return_detail: bool = False,
) -> (
    tuple[CTMEnvironment, CTMEnvironment]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo, str | None]
):
    """``ctm_2site()`` without its warnings: the 2x2 recipe of
    ``ctm_tensor_2site`` with the ``CTMConfig`` fields mapped, returned as two
    raw-array ``CTMEnvironment`` named tuples.

    For callers that need ``ctm_2site()``'s result and must not trigger its
    ``DeprecationWarning``: ``ipeps()`` and the fixture-only tests.

    Args:
        A:           Site tensor of sublattice A: a raw array with 5 legs
                     ``(u, d, l, r, s)`` and four virtual legs of one dimension,
                     or a ``DenseTensor``, which is passed through.
                     ``_wrap_as_dense_tensor`` reads ``shape[0]`` for all four
                     virtual legs without a check, so any other shape raises
                     ``ValueError`` here.  A ``SymmetricTensor`` raises
                     ``TypeError``: the raw result would need ``todense()``.
        B:           Site tensor of sublattice B, checked the same way.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` are passed on.  ``projector_method`` is not:
                     the 2x2 recipe always uses Fishman SVD, and ``"svd"`` is
                     passed because ``"qr"`` would still start the QR warm-up
                     loop.  So ``qr_warmup_steps`` has no effect.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`,
                     whose fields are Python scalars.
        _return_detail: Private.  With ``return_meta``, also return the loop's
                     not-converged diagnosis (``None`` when converged), which
                     ``ipeps()`` quotes when a rank-1 corner blinds the
                     criterion (#898).

    Returns:
        ``(env_A, env_B)``, ``(env_A, env_B, info)``, or ``(env_A, env_B,
        info, detail)``.  Every field has its legs in the order
        ``CTMTensorEnv`` documents, which is the order the raw 2-site RDMs of
        ``ipeps_rdm`` contract.  The loop's own not-converged warning is off:
        ``ipeps()`` warns itself, and ``ctm_2site()`` reports through ``info``
        as it always did.
    """
    from tenax.algorithms._ctm_env_dense import _as_dense_env
    from tenax.algorithms._ctm_tensor_convergence import (
        CHECKERBOARD_NEIGHBORS,
        _ctm_tensor_multisite,
    )
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor

    sites = {}
    for name, coord, X in (("A", (0, 0), A), ("B", (1, 0), B)):
        if isinstance(X, SymmetricTensor):
            raise TypeError(
                f"ctm_2site() returns raw-array CTMEnvironments, so the "
                f"SymmetricTensor site tensor {name} would have to be densified. "
                "Call ctm_tensor_2site(A, B, chi, ...) instead: it keeps the "
                "blocks and returns two CTMTensorEnvs."
            )
        if not isinstance(X, Tensor):
            X = jnp.asarray(X)
            if X.ndim != 5 or len(set(X.shape[:4])) != 1:
                raise ValueError(
                    "Expected raw site tensors with 5 legs (u, d, l, r, s) and "
                    f"four virtual legs of one dimension; got shape {X.shape} "
                    f"for {name}."
                )
            X = _wrap_as_dense_tensor(X)
        sites[coord] = X
    envs, status, detail = _ctm_tensor_multisite(
        sites,
        CHECKERBOARD_NEIGHBORS,
        config.chi,
        max_iter=config.max_iter,
        conv_tol=config.conv_tol,
        renormalize=config.renormalize,
        projector_method="svd",
        recipe="2x2",
        min_iter=config.min_iter,
        _return_status=True,
        _warn_unconverged=False,
    )
    # The 2x2 sweep stores each corner with its two legs reversed, and the raw
    # consumers contract by position: _as_dense_env puts every field in the
    # documented order, by label.
    env_A = _as_dense_env(envs[(0, 0)])
    env_B = _as_dense_env(envs[(1, 0)])
    if return_meta:
        info = CTMConvergenceInfo(
            converged=status.converged, n_iter=status.iterations, diff=status.sv_diff
        )
        if _return_detail:
            return env_A, env_B, info, detail
        return env_A, env_B, info
    return env_A, env_B


def ctm_2site(
    A: jax.Array | DenseTensor,
    B: jax.Array | DenseTensor,
    config: CTMConfig,
    *,
    return_meta: bool = False,
) -> (
    tuple[CTMEnvironment, CTMEnvironment]
    | tuple[CTMEnvironment, CTMEnvironment, CTMConvergenceInfo]
):
    """Compute the CTM environments of a 2-site checkerboard pair with the 2x2 recipe.

    .. deprecated:: 0.9.3
        Use :func:`ctm_tensor_2site`.  ``ctm_2site()`` now runs its 2x2 recipe
        and returns the two environments as the raw-array
        :class:`CTMEnvironment` named tuples that ``ctm_2site()`` always
        returned.  The sweep it ran before built each sublattice's new boundary
        from that sublattice's own corners and edges, with a projector of its
        own, so the two environments did not share a basis at the A|B seams and
        the energy moved with the bond gauge of the pair: on three D=2
        simple-update states, each in two bond gauges at chi=6 and 8, the
        energy of ``ipeps()``, which measured with it, ranged from -0.7815 to
        +0.8171, while ``ctm_tensor_2site`` agreed with the 1-site CTM of each
        state to 4e-11.  So environments from ``ctm_2site()`` changed in 0.9.3,
        and ``ctm_2site()`` no longer runs under ``jax.jit``, because the 2x2
        recipe is a Python loop.

    Args:
        A:           Site tensor of sublattice A: a raw array with 5 legs
                     ``(u, d, l, r, s)`` and four virtual legs of one
                     dimension, or a ``DenseTensor``.  Any other shape raises
                     ``ValueError``.  A ``SymmetricTensor`` raises
                     ``TypeError``: call :func:`ctm_tensor_2site`, which keeps
                     the blocks.
        B:           Site tensor of sublattice B, the same way.
        config:      ``chi``, ``max_iter``, ``conv_tol``, ``renormalize`` and
                     ``min_iter`` apply.  A ``projector_method`` other than
                     ``"svd"`` is ignored with a ``UserWarning``, because the
                     2x2 recipe always uses Fishman SVD; ``qr_warmup_steps`` is
                     ignored too.
        return_meta: When True, also return a :class:`CTMConvergenceInfo`
                     saying whether the loop converged or exhausted
                     ``max_iter`` (#839).  Its fields are Python scalars.
                     Keyword-only and off by default so the return arity of
                     this public entry point is unchanged.

    Returns:
        ``(env_A, env_B)``, or ``(env_A, env_B, info)`` when
        ``return_meta=True``.  Each field has its legs in the order
        ``CTMTensorEnv`` documents, which is the order
        :func:`compute_energy_ctm_2site` contracts.  **The environments are not
        necessarily converged** -- that is what ``info`` is for.  One ``info``
        covers both sublattices: the loop converges when the C1 spectrum of
        each sublattice stops changing (the criterion reads only C1).
    """
    warnings.warn(
        "ctm_2site is deprecated and will be removed in a later release: call "
        "ctm_tensor_2site(A, B, chi, ...) on DenseTensor or SymmetricTensor site "
        "tensors with legs (u, d, l, r, phys). ctm_2site now runs the 2x2 recipe "
        "of ctm_tensor_2site, so its environments changed in 0.9.3 (the old "
        "sweep gave energies that moved with the bond gauge), and it no longer "
        "runs under jax.jit.",
        DeprecationWarning,
        stacklevel=2,
    )
    if config.projector_method != "svd":
        warnings.warn(
            f"ctm_2site() ignores projector_method={config.projector_method!r}: "
            "the 2x2 recipe always uses Fishman SVD projectors.",
            UserWarning,
            stacklevel=2,
        )
    return _ctm_2site_via_2x2(A, B, config, return_meta=return_meta)


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
        ``info`` is for.  Its fields are Python scalars.
    """
    chi = config.chi
    chi_I = config.chi_I if config.chi_I is not None else chi
    # Cap ``min_iter`` at the budget so a small ``max_iter`` stays convergeable
    # (#976).
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
