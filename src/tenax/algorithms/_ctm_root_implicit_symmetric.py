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
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core import SymmetricTensor, Tensor
from tenax.core.index import FlowDirection

__all__ = [
    "SymEnv",
    "half_infinite_sym",
    "init_env_sym",
    "lower_left_quadrant_sym",
    "swap_env_convention_sym",
    "upper_left_quadrant_sym",
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


# ---------------------------------------------------------------------------
# Enlarged corners and the half-infinite environment (paper Eq. 65)
# ---------------------------------------------------------------------------


def upper_left_quadrant_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """``C1 T1 T4 a`` with legs ``(chi_r, a_r, chi_d, a_d)``.

    Same network as ``_ctm_root_implicit_asym._upper_left_quadrant``; the
    einsum's index letters become labels.  ``(chi_d, a_d)`` is the vertical
    bond to be truncated, ``(chi_r, a_r)`` is what stays open to the right.

    ``output_labels`` already fixes the axis order, so no transpose is needed
    afterwards.
    """
    c1 = env.C1.relabels(dict(zip(env.C1.labels(), ("c", "e"), strict=True)))
    t1 = env.T1.relabels(dict(zip(env.T1.labels(), ("e", "f", "chi_r"), strict=True)))
    t4 = env.T4.relabels(dict(zip(env.T4.labels(), ("h", "i", "c"), strict=True)))
    a4 = a.relabels(dict(zip(a.labels(), ("f", "a_d", "i", "a_r"), strict=True)))
    q = contract(c1, t1, t4, a4, output_labels=("chi_r", "a_r", "h", "a_d"))
    return q.relabels({"h": "chi_d"})


def lower_left_quadrant_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """``C4 T3 T4 a`` with legs ``(chi_u, a_u, chi_r, a_r)``.

    Same network as ``_ctm_root_implicit_asym._lower_left_quadrant``, whose
    einsum is ``"mn,pqm,nit,uqik->tupk"``.  Here ``(chi_u, a_u)`` — the
    einsum's ``(t, u)`` — is the vertical bond to be truncated, and it is the
    partner of :func:`upper_left_quadrant_sym`'s ``(chi_d, a_d)``: ``t`` is
    ``T4.u`` against the other quadrant's ``T4.d``, and ``u`` is ``a.u``
    against its ``a.d``.
    """
    # The einsum letter ``q`` is a-down / T3's a-leg; the Python name ``quad``
    # is used for the result so the two never collide.
    c4 = env.C4.relabels(dict(zip(env.C4.labels(), ("m", "n"), strict=True)))
    t3 = env.T3.relabels(dict(zip(env.T3.labels(), ("p", "q", "m"), strict=True)))
    t4 = env.T4.relabels(dict(zip(env.T4.labels(), ("n", "i", "t"), strict=True)))
    a4 = a.relabels(dict(zip(a.labels(), ("u", "q", "i", "k"), strict=True)))
    quad = contract(c4, t3, t4, a4, output_labels=("t", "u", "p", "k"))
    return quad.relabels(
        {"t": "chi_u", "u": "a_u", "p": "chi_r", "k": "a_r"},
    )


def _as_matrix_sym(quadrant: SymmetricTensor) -> SymmetricTensor:
    """Fuse a quadrant's last two legs to ``row`` and its first two to ``col``.

    The fused leg lands at the position of the first of the two axes, so the
    result's axis order is ``(col, row)`` = ``(axes 0-1, axes 2-3)`` — exactly
    what the dense code's plain ``.reshape(chi * d2, chi * d2)`` produces, and
    ``fuse_indices`` keeps the parent legs' row-major ordering inside the fused
    charge array, so ``todense()`` of the result equals that reshape element
    for element.

    Fusion happens *only* here, to feed the SVD.  Everywhere the singular
    values have to act, the cut leg stays split — that is what replaces the
    dense code's ``jnp.kron(root, eye_d2)``, which is not a per-sector kron
    once the leg is fused.
    """
    fused = fuse_indices(quadrant, 2, 3, "row", FlowDirection.IN)
    return fuse_indices(fused, 0, 1, "col", FlowDirection.OUT)


def half_infinite_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """Paper Eq. 65: upper-left quadrant glued to lower-left along the cut.

    The dense reference reshapes each quadrant in its *native* axis order, so
    the cut sits in the upper quadrant's **columns** (axes 2-3, ``chi_d, a_d``)
    and in the lower quadrant's **rows** (axes 0-1, ``chi_u, a_u``).
    :func:`_as_matrix_sym` fuses axes 2-3 and axes 0-1 of whatever it is given,
    so it is the right fusion for *both* operands — but it fuses by position,
    not by meaning, so the cut comes out labelled ``row`` on the top quadrant
    and ``col`` on the bottom one.  Renaming ``col -> row`` on the bottom is
    what pairs them.

    The flows work out for the same reason: ``_as_matrix_sym`` fuses axes 2-3
    with ``IN`` and axes 0-1 with ``OUT``, and the cut's constituent legs carry
    opposite flows on the two quadrants (``T4.d`` vs ``T4.u``, ``a.d`` vs
    ``a.u``).  Two sign flips cancel, so the fused charge arrays agree while
    the fused flows stay opposite — which is exactly the contractible case.
    """
    top = _as_matrix_sym(upper_left_quadrant_sym(env, a))
    bot = _as_matrix_sym(lower_left_quadrant_sym(env, a))
    # ``top`` is (col = chi_r/a_r, row = the cut); ``bot`` is (col = the cut,
    # row = chi_r/a_r).  Rename so only the cut is shared.
    top = top.relabels({"col": "m_row", "row": "cut"})
    bot = bot.relabels({"col": "cut", "row": "m_col"})
    return contract(top, bot, output_labels=("m_row", "m_col"))
