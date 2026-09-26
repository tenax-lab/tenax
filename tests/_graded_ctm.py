"""Finite-patch references for the graded Tensor CTM (#1035, design §5 step 4).

An untruncated CTM started from the *vacuum boundary* (every boundary
double-layer leg fixed to its parity-even slot 0) is an exact contraction of a
finite open-boundary patch: after ``n`` sweeps the environment around a
``1 x 1`` centre is the ``(2n+1) x (2n+1)`` patch, and around a ``1 x 2``
centre the ``(2n+1) x (2n+2)`` one.  That patch is exactly what #1038's Fock
oracle and Phase 2's ``double_layer_value`` compute, so the CTM's RDMs can be
checked against them to roundoff.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _fermionic_fock_oracle import _H2, bonds_of, sites_of

from tenax.algorithms._ctm_tensor_convergence import (
    SINGLE_SITE_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.tensor import SymmetricTensor


def vacuum_boundary_env(A: SymmetricTensor, chi: int) -> CTMTensorEnv:
    """Production's initial environment with every edge's D² leg restricted
    to fused slot 0 (ket 0, bra 0): the open boundary of a finite patch."""
    env = initialize_ctm_tensor_env(A, chi)
    edges = {}
    for name in ("T1", "T2", "T3", "T4"):
        T = getattr(env, name)
        dense = np.array(T.todense())
        dense[:, 1:, :] = 0.0
        edges[name] = SymmetricTensor.from_dense(
            jnp.asarray(dense), T.indices, tol=float("inf")
        )
    return env._replace(**edges)


def sweep(env: CTMTensorEnv, a, chi: int, n: int) -> CTMTensorEnv:
    """``n`` full sweeps of the production 2x2-recipe CTM on a 1-site cell."""
    for _ in range(n):
        envs, _eps, _s = _ctm_tensor_sweep_multisite(
            {(0, 0): env}, {(0, 0): a}, SINGLE_SITE_NEIGHBORS, chi, True, "svd"
        )
        env = envs[(0, 0)]
    return env


def untruncated_env(A: SymmetricTensor, n: int) -> tuple[CTMTensorEnv, int]:
    """The exact environment of the ``(2n+1)``-ring patch: ``chi = D**(2n)``
    keeps every singular value of every projector."""
    D = A.indices[0].dim
    chi = D ** (2 * n)
    a = _build_double_layer_tensor(A)
    return sweep(vacuum_boundary_env(A, chi), a, chi, n), chi


def patch(R: int, C: int, A: np.ndarray) -> dict:
    """The uniform ``R x C`` patch of ``A[u, d, l, r, p]`` with every outward
    leg sliced to index 0 -- the vacuum boundary as dense site tensors."""
    out = {}
    for i, j in sites_of(R, C):
        sl = [slice(None)] * 5
        if i == 0:
            sl[0] = slice(0, 1)
        if i == R - 1:
            sl[1] = slice(0, 1)
        if j == 0:
            sl[2] = slice(0, 1)
        if j == C - 1:
            sl[3] = slice(0, 1)
        out[(i, j)] = A[tuple(sl)]
    return out


def centre_bond(R: int, C: int, horizontal: bool):
    """The bond the CTM's ``2x1`` (``1x2``) RDM measures on the patch."""
    s = (R // 2, C // 2 - 1) if horizontal else (R // 2 - 1, C // 2)
    t = (s[0], s[1] + 1) if horizontal else (s[0] + 1, s[1])
    assert any(b[0] == s and b[2] == t for b in bonds_of(R, C)), (s, t)
    return s, t


HOP = _H2
