"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian).

Structurally this mirrors ``_ctm_root_implicit_asym`` function for function.
The difference is that every environment tensor stays a ``SymmetricTensor``
and every contraction goes through ``contract``, so charge bookkeeping is the
library's job and a wrong flow raises instead of silently mis-gluing the
network — which is the #718 failure mode.

Charge arithmetic at the cut lives in ``_ctm_root_implicit_sym_sectors``; this
file never touches a charge directly.
"""

from __future__ import annotations

from typing import NamedTuple

from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core import SymmetricTensor, Tensor

__all__ = [
    "SymEnv",
    "init_env_sym",
    "swap_env_convention_sym",
]


class SymEnv(NamedTuple):
    """Eight environment tensors in this module's rotation-uniform frame."""

    C1: SymmetricTensor
    C2: SymmetricTensor
    C3: SymmetricTensor
    C4: SymmetricTensor
    T1: SymmetricTensor
    T2: SymmetricTensor
    T3: SymmetricTensor
    T4: SymmetricTensor


def swap_env_convention_sym(env: SymEnv) -> SymEnv:
    """Between this module's uniform frame and ``CTMTensorEnv``'s.

    Same map as ``_ctm_root_implicit_asym.swap_env_convention``, and the same
    reason it exists: this module closes the ring uniformly (corner ``k`` is
    always ``(leg towards k-1, leg towards k)``), while ``CTMTensorEnv`` closes
    it with ``C4`` transposed and ``T3``, ``T4`` reversed.  Reinterpreting one
    as the other glues the network wrongly — 1.5% on the energy at D=2 chi=4,
    and the +-2.121e-3 per-bond antisymmetry that was #718.

    An involution, so one function converts either way.  It is a *no-op* on a
    symmetric initialiser, which is why the mismatch stayed invisible until the
    environment became genuinely asymmetric.
    """
    return env._replace(
        C4=env.C4.transpose((1, 0)),
        T3=env.T3.transpose((2, 1, 0)),
        T4=env.T4.transpose((2, 1, 0)),
    )


def init_env_sym(A: Tensor, chi: int) -> tuple[SymEnv, SymmetricTensor]:
    """Seed the environment and build the double layer, without densifying.

    ``_ctm_root_implicit_asym._init_env`` calls ``.todense()`` on all nine
    tensors here.  Keeping them symmetric is most of the symmetric forward.
    """
    env_t = initialize_ctm_tensor_env(A, chi)
    a_t = _build_double_layer_tensor(A)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    a = a_t.transpose(perm)
    env = SymEnv(
        C1=env_t.C1,
        C2=env_t.C2,
        C3=env_t.C3,
        C4=env_t.C4,
        T1=env_t.T1,
        T2=env_t.T2,
        T3=env_t.T3,
        T4=env_t.T4,
    )
    return swap_env_convention_sym(env), a
