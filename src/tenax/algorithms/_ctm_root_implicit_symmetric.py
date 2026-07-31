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

import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_root_implicit_asym import (
    _denman_beavers,
    _inv_sqrt,
    _quartic_root,
)
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    SectorSVD,
    bond_index_from_layout,
    sector_map,
    sector_svd,
    tensor_from_sector_matrices,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.contraction.contractor import contract
from tenax.core import SymmetricTensor, Tensor
from tenax.core.index import FlowDirection
from tenax.linalg import _group_blocks_by_bond_charge

__all__ = [
    "SymEnv",
    "SymProjectors",
    "SymRoot",
    "absorb_inverse_roots_sym",
    "all_projectors_sym",
    "apply_bond_matrix",
    "characteristic_residual_sym",
    "converge_sym",
    "half_infinite_sym",
    "init_env_sym",
    "lower_left_quadrant_sym",
    "remove_inverse_roots_sym",
    "renormalised_corner_sym",
    "renormalised_edge_sym",
    "root_parametrize_sym",
    "rotate_a_sym",
    "rotate_env_sym",
    "swap_env_convention_sym",
    "sweep_sym",
    "sym_energy",
    "sym_root_implicit_energy_and_grad",
    "sym_root_to_covariant_convention",
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


def rotate_env_sym(env: SymEnv) -> SymEnv:
    """Rotate the environment 90 degrees counter-clockwise.

    Exactly ``_ctm_root_implicit_asym.rotate_env``: old top-right becomes new
    top-left, and so on.  Every tensor already carries the rotated axis order
    (``C2[l, d]`` read as ``C1[d, r]`` is the same tensor), so this is a pure
    relabel of the eight slots with no transpose and no charge arithmetic —
    which is what lets one left move serve all four directions.
    """
    return SymEnv(
        C1=env.C2,
        C2=env.C3,
        C3=env.C4,
        C4=env.C1,
        T1=env.T2,
        T2=env.T3,
        T3=env.T4,
        T4=env.T1,
    )


def rotate_a_sym(a: SymmetricTensor) -> SymmetricTensor:
    """Rotate the double-layer tensor 90 degrees counter-clockwise.

    ``a`` is stored ``(u, d, l, r)``, and the rotation sends new-up = old-right,
    new-down = old-left, new-left = old-up, new-right = old-down — the same
    ``(3, 2, 0, 1)`` permutation ``_ctm_root_implicit_asym.rotate_a`` applies to
    the dense array.  ``SymmetricTensor.transpose`` takes the identical
    "new order of legs" convention as ``jnp.transpose``, so the two agree
    element for element (checked in the tests).
    """
    return a.transpose((3, 2, 0, 1))


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
    top, bot = _cut_halves_sym(env, a)
    return contract(top, bot, output_labels=("m_row", "m_col"))


def _cut_halves_sym(
    env: SymEnv, a: SymmetricTensor
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The two factors of Eq. 65, as matrices sharing only the ``cut`` label.

    ``top`` comes out ``(m_row, cut)`` and ``bot`` ``(cut, m_col)``, matching
    the dense ``top``/``bot`` of ``_ctm_root_implicit_asym._fishman_projectors``
    axis for axis.  Factored out of :func:`half_infinite_sym` because
    :func:`all_projectors_sym` needs the *same* two halves — the sector-key
    alignment below rests on these exact labels and flows, so recomputing them
    independently would be a place for the two to drift apart.
    """
    top = _as_matrix_sym(upper_left_quadrant_sym(env, a))
    bot = _as_matrix_sym(lower_left_quadrant_sym(env, a))
    # ``top`` is (col = chi_r/a_r, row = the cut); ``bot`` is (col = the cut,
    # row = chi_r/a_r).  Rename so only the cut is shared.
    top = top.relabels({"col": "m_row", "row": "cut"})
    bot = bot.relabels({"col": "cut", "row": "m_col"})
    return top, bot


# ---------------------------------------------------------------------------
# Fishman projectors, per charge sector (paper Eqs. 66-67)
# ---------------------------------------------------------------------------


class SymProjectors(NamedTuple):
    """One direction's Fishman pair, per charge sector, plus the frozen data.

    ``P_left`` is the dense module's ``P_top`` and ``P_right`` its ``P_bot``;
    both live on the *cut* leg, so ``P_right[q] @ P_left[q]`` closes on the
    retained subspace of sector ``q`` and ``P_left[q] @ P_right[q]`` projects
    onto it.

    Both carry a factor of ``norm(s_k)`` — the norm of the *unnormalised*
    retained spectrum of the whole cut — because ``S`` is normalised and the
    projectors each take one ``S^-1/2``.  So the closure is ``norm(s_k) * 1``
    rather than ``1``, and ``P_left @ P_right`` squares to ``norm(s_k)`` times
    itself.  That is the dense module's behaviour, reproduced here on purpose:
    it is harmless in the forward sweep, where every renormalised corner and
    edge is rescaled immediately afterwards, and ``S`` is a *variable* of the
    characteristic equations, so quietly dividing the factor out would move the
    root rather than tidy it.

    ``S`` is a full ``chi_q x chi_q`` **matrix** per sector, not a vector of
    singular values, for the reason spelled out in
    :func:`_fishman_projectors_sym`.  ``sectors`` carries the decomposition the
    pair was built from — with ``U`` and ``Vh`` already rotated into the pinned
    bond gauge, so projectors rebuilt from them inside a characteristic
    equation land in the same gauge as these.

    ``P_left_t`` and ``P_right_t`` are the same two objects assembled into
    ``SymmetricTensor``s on the cut leg, which is the form the sweep actually
    contracts; see :func:`_projector_tensors_sym` for why they are built here,
    where the two cut indices are in hand, rather than rebuilt at the use site.
    """

    P_left: dict[int, jax.Array]  # per sector, (n_q, chi_q)
    P_right: dict[int, jax.Array]  # per sector, (chi_q, n_q)
    S: dict[int, jax.Array]  # per sector, (chi_q, chi_q) MATRIX
    sectors: dict[int, SectorSVD]
    layout: BondLayout
    P_left_t: SymmetricTensor  # (cut, chi_new)
    P_right_t: SymmetricTensor  # (chi_new, cut)


def _sector_blocks(
    matrix: SymmetricTensor, *, row_axis: int, col_axis: int
) -> dict[int, jax.Array]:
    """Dense blocks of a 2-leg tensor, keyed by bond charge, in ``(row, col)``.

    Same key convention and same orientation step as
    :func:`~tenax.algorithms._ctm_root_implicit_sym_sectors.sector_svd`, which
    is what makes the three key sets line up: ``sector_svd`` keys the
    half-infinite matrix ``M`` by ``flow(m_row) * charge(m_row)``, and

    * ``top`` shares ``M``'s ``m_row`` leg outright, so keying it on the same
      axis reproduces ``M``'s key exactly;
    * ``bot`` is keyed on its ``cut`` leg, and charge conservation on ``top``
      pins ``flow_top(cut) * charge(cut) = -q`` while the two quadrants carry
      *opposite* flows on the cut, so ``flow_bot(cut) * charge(cut) = +q``.

    Verified empirically on the D=2 fixture rather than trusted: ``M``, ``top``
    and ``bot`` all key ``{-1, 0, 1}`` and the cut charge ``top`` reports as its
    column key is the one ``bot`` reports as its row key.
    """
    grouped = _group_blocks_by_bond_charge(matrix, [row_axis], [col_axis])
    out: dict[int, jax.Array] = {}
    for q, entries in grouped.items():
        if len(entries) != 1:  # pragma: no cover - a fused matrix has one per q
            raise ValueError(
                f"sector {q} has {len(entries)} blocks; expected a fused matrix"
            )
        (_row_key, _col_key, block) = entries[0]
        out[q] = jnp.transpose(block, (row_axis, col_axis))
    return out


def _pin_bond_gauge_sector(
    P_left: jax.Array,
    P_right: jax.Array,
    U: jax.Array,
    Vh: jax.Array,
    k_q: int,
    prev_P_left: jax.Array | None = None,
):
    """Pin the residual phase freedom on each renormalised bond of one sector.

    ``_ctm_root_implicit_asym._pin_bond_gauge`` restricted to a charge sector.
    An SVD fixes the singular subspaces but leaves one phase per retained index
    free (one *sign*, for real input).  That phase is a gauge of the CTM bond,
    so the corner spectra converge without it being fixed — and indeed the
    environment converges element-wise in ``|.|`` to 1e-14 while individual
    signs keep flipping from sweep to sweep.  A characteristic equation cannot
    have a root under those conditions: ``F`` compares tensors, not their
    magnitudes.

    Fixing the phase on ``U`` alone is not enough, because the projectors also
    inherit the *previous* bond gauge through the quadrants.  Pinning the
    largest-magnitude entry of each ``P_left`` column to be real-positive fixes
    the new bond directly, and pushing the conjugate phase onto ``P_right``
    keeps ``P_right @ P_left = 1``.  The same phase is folded into
    ``(U, Vh)`` so that projectors rebuilt from them inside the characteristic
    equations land in the same gauge.

    Warm alignment to the previous sweep is preferred over ``argmax`` because
    ``argmax`` is discontinuous: when two retained singular values are close its
    row index hops between sweeps and the pinned phase oscillates with period
    two.  Per sector this is if anything more likely, since a sector's retained
    block is small and near-degenerate pairs inside it are common.
    """
    if prev_P_left is None:
        # Cold start: pin the largest-magnitude entry of each column.
        idx = jnp.argmax(jnp.abs(P_left), axis=0)
        ref = P_left[idx, jnp.arange(P_left.shape[1])]
    else:
        ref = jnp.sum(jnp.conj(prev_P_left) * P_left, axis=0)
    psi = jnp.where(jnp.abs(ref) > 0, jnp.conj(ref) / jnp.abs(ref), 1.0)
    P_left = P_left * psi[None, :]
    P_right = jnp.conj(psi)[:, None] * P_right
    U = U.at[:, :k_q].multiply(psi[None, :])
    Vh = Vh.at[:k_q, :].multiply(jnp.conj(psi)[:, None])
    return P_left, P_right, U, Vh


def _retained_S_sym(
    sectors: dict[int, SectorSVD], layout: BondLayout, dtype
) -> dict[int, jax.Array]:
    """The retained spectrum as one ``chi_q x chi_q`` matrix per sector.

    The dense line being ported is

        S_keep = diag(s_k / (norm(s_k) + 1e-300)).astype(M.dtype)

    with ``s_k`` the globally floored top-``chi`` singular values of the *whole*
    cut.  Both the floor and the normalisation are therefore global, not
    per-sector: ``sector_svd`` already applied the global floor to
    ``S_keep_diag``, and the norm here runs over every sector's retained slice.
    Normalising sector by sector instead would rescale each sector's projector
    pair independently — ``P_left @ P_right`` carries one factor of ``S^-1`` —
    and the retained-subspace projector would stop matching the dense one.

    The cast to the environment's dtype is the dense module's, for the dense
    module's reason: ``S`` is a *variable* of the characteristic equations and
    the reverse pass needs it free to leave the reals, while
    ``jnp.linalg.svd`` only ever returns a real spectrum (#721).
    """
    retained = {q: sectors[q].S_keep_diag[: layout.dim_of(q)] for q in layout.sectors}
    norm = jnp.sqrt(sum(jnp.sum(v**2) for v in retained.values()))
    return {q: jnp.diag(v / (norm + 1e-300)).astype(dtype) for q, v in retained.items()}


def _fishman_projectors_sym(
    top_blocks: dict[int, jax.Array],
    bot_blocks: dict[int, jax.Array],
    sectors: dict[int, SectorSVD],
    S: dict[int, jax.Array],
    layout: BondLayout,
) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
    """Paper Eqs. 66-67 per charge sector, from a *given* decomposition.

    Sector by sector this is ``_ctm_root_implicit_asym._fishman_projectors``
    verbatim::

        P_left  = bot @ Vh[:chi].conj().T @ inv_sqrt        # (n_q, chi_q)
        P_right = inv_sqrt @ (U[:, :chi].conj().T @ top)    # (chi_q, n_q)

    ``S`` is a general ``chi_q x chi_q`` matrix and ``inv_sqrt`` a genuine
    *matrix* inverse square root (Denman-Beavers, so no decomposition enters
    ``F``; deliberately not ``eigh``, whose VJP divides by eigenvalue
    differences and would NaN exactly where this method is meant to be safe).
    That is what makes the pair gauge-covariant: under the bond rotation
    ``U -> U W``, ``Vh -> W† Vh``, ``S -> W† S W`` the matrix root is
    equivariant, ``(W† S W)^-1/2 = W† S^-1/2 W``, so ``P_right -> W† P_right``,
    ``P_left -> P_left W`` and

        P_right @ P_left = S^-1/2 (U† M Vh†) S^-1/2 = S^-1/2 S S^-1/2 = 1

    survives for *any* ``S``, diagonal or not.  (With the *normalised* ``S``
    the caller actually passes, the right-hand side is ``norm(s_k) * 1``; the
    covariance argument is unaffected, since a scalar commutes with ``W``.
    See :class:`SymProjectors`.)  With a diagonal ``S`` the
    in-space rotation is not representable, the closure breaks at first order,
    and Eq. 88's null-space restriction then discards a physical contribution
    instead of a gauge one — 120% gradient error in Phase 1.

    The two-sided roots of paper Eq. 73, ``(S S†)^-1/4`` and ``(S† S)^-1/4``,
    do **not** belong here: they are equivariant under independent left/right
    rotations but their product is not ``S^-1``, so the closure breaks for a
    non-diagonal ``S`` (2e-1 vs 2.5e-2 gradient error when Phase 1 tried it).
    In the paper they sit on the *cut legs* of the modified environment, where
    no closure condition applies.
    """
    inv_sqrt = sector_map(_inv_sqrt, S)
    P_left: dict[int, jax.Array] = {}
    P_right: dict[int, jax.Array] = {}
    for q in layout.sectors:
        # A sector retaining nothing never reaches here (``layout.sectors``
        # excludes it); a sector with ``n_q == chi_q`` has an empty null space,
        # which is legal — ``U[:, :chi_q]`` is then all of ``U``.
        if q not in top_blocks or q not in bot_blocks:  # pragma: no cover
            raise ValueError(
                f"retained charge {q} of the cut has no block in "
                f"{'top' if q not in top_blocks else 'bot'}; the sector keys of "
                "M, top and bot have stopped agreeing."
            )
        k_q = layout.dim_of(q)
        svd = sectors[q]
        P_left[q] = bot_blocks[q] @ svd.Vh[:k_q].conj().T @ inv_sqrt[q]
        P_right[q] = inv_sqrt[q] @ (svd.U[:, :k_q].conj().T @ top_blocks[q])
    return P_left, P_right


def _projector_tensors_sym(
    P_left: dict[int, jax.Array],
    P_right: dict[int, jax.Array],
    layout: BondLayout,
    top: SymmetricTensor,
    bot: SymmetricTensor,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Assemble the per-sector Fishman pair into two ``SymmetricTensor``s.

    Applying the projectors *as tensors* rather than sector by sector is the
    choice this module makes, and the reason is the renormalised **corner**.
    Its two projectors come from two different moves — ``P_left`` of move ``k``
    and ``P_right`` of move ``k+1`` — so a per-sector implementation would have
    to know how the sector keys of two independently decomposed cuts line up
    against the same quadrant.  As a tensor there is nothing to line up:
    ``contract`` matches the charge sectors itself and raises if they cannot be
    matched, which is exactly the class of bug (#718) this module exists to
    make loud.

    Which index goes where is fixed by the identity insertion, not by which
    half each projector was *built* from.  Eq. 65 factorises the cut as
    ``M = top . bot`` and inserts ``P_left . P_right`` on the shared bond, so
    ``P_left``'s cut leg contracts a leg playing ``top``'s role and
    ``P_right``'s contracts one playing ``bot``'s.  The flows follow: the two
    quadrants carry opposite flows on the cut (``_cut_halves_sym``), so
    ``P_left`` takes ``bot``'s cut index and ``P_right`` takes ``top``'s, and
    each then meets its partner with the opposite arrow.

    The truncated bond is built from :func:`bond_index_from_layout` twice, OUT
    on ``P_left`` and IN on ``P_right``.  That is forced, not a convention:
    ``tensor_from_sector_matrices`` pins the new leg's charge to
    ``-q * flow``, so the two flows give the same sector list with opposite
    arrows — the exact dual pair the renormalised ring needs, since a corner
    takes its first leg from a ``P_left`` and its second from a ``P_right``
    while the edge that shares each of those bonds takes the other one.
    """
    cut_top = top.indices[top.labels().index("cut")]
    cut_bot = bot.indices[bot.labels().index("cut")]
    sym = cut_top.symmetry
    new_out = bond_index_from_layout(layout, sym, FlowDirection.OUT, "chi_new")
    new_in = bond_index_from_layout(layout, sym, FlowDirection.IN, "chi_new")
    P_left_t = tensor_from_sector_matrices(
        {q: P_left[q] for q in layout.sectors},
        row_index=cut_bot,
        col_index=new_out,
        row_axis=0,
        col_axis=1,
    )
    P_right_t = tensor_from_sector_matrices(
        {q: P_right[q] for q in layout.sectors},
        row_index=new_in,
        col_index=cut_top,
        row_axis=0,
        col_axis=1,
    )
    return P_left_t, P_right_t


def _check_layout_override(
    wanted: BondLayout, svds: dict[int, SectorSVD], k: int
) -> BondLayout:
    """Check a forced layout is a legal truncation of ``svds``, and return it.

    Test-only; see ``layout_override`` on :func:`all_projectors_sym`.  "Legal"
    means every retained charge is one the cut actually carries and no sector is
    asked for more directions than it has, so that what the caller gets is a
    *different truncation of the same decomposition* rather than an index error
    — which is the whole point of the hook.
    """
    for q in wanted.sectors:
        if q not in svds:
            raise ValueError(
                f"layout_override[{k}] retains charge {q}, which the cut does not "
                f"carry (it has {sorted(svds)}); an override has to be a different "
                "truncation of the same decomposition, not a different cut."
            )
        available = min(svds[q].U.shape[1], svds[q].Vh.shape[0])
        if wanted.dim_of(q) > available:
            raise ValueError(
                f"layout_override[{k}] asks sector {q} for {wanted.dim_of(q)} "
                f"states; it only has {available}."
            )
    return wanted


def all_projectors_sym(
    env: SymEnv,
    a: SymmetricTensor,
    chi: int,
    prev: list[SymProjectors] | None = None,
    *,
    layout_override: tuple[BondLayout | None, ...] | None = None,
) -> list[SymProjectors]:
    """Decompose the cut in all four directions, from the *same* environment.

    ``_ctm_root_implicit_asym.all_projectors`` with the dense SVD replaced by a
    per-sector one.  Paper Eq. 65 and "their corresponding rotated versions":
    every projector in a sweep is built from one environment, and every corner
    and edge is then renormalised simultaneously.  That matters — a sequential
    (Gauss-Seidel) sweep, where move ``k+1`` sees the output of move ``k``, has
    a fixed point that does *not* satisfy Eqs. 76-77, because those equations
    evaluate all four moves at the same ``y``.

    ``prev`` is the previous sweep's return value and only feeds the bond-gauge
    pin.  A sector whose retained dimension moved between sweeps is pinned cold
    instead: the charge distribution over the cut is data-dependent, so unlike
    the dense case the two sweeps' blocks need not even have the same shape.

    ``layout_override`` **exists solely for the trap test**
    ``test_a_wrong_bond_layout_breaks_the_root`` and has no production caller.
    It is a per-direction ``BondLayout | None`` replacing the one
    :func:`sector_svd` chose, so a test can retain a *deliberately wrong* set
    of states — one dimension moved from one charge sector to another, total
    still ``chi`` — and ask whether anything downstream notices.  Nothing else
    may use it: the layout is what the truncation *chose*, and choosing it from
    outside is exactly the bug the trap test is built to catch.
    """
    if layout_override is not None and len(layout_override) != 4:
        raise ValueError(
            f"layout_override must have one entry per direction, got "
            f"{len(layout_override)}"
        )
    out: list[SymProjectors] = []
    env_k, a_k = env, a
    for k in range(4):
        top, bot = _cut_halves_sym(env_k, a_k)
        M = contract(top, bot, output_labels=("m_row", "m_col"))
        svds, layout = sector_svd(M, chi, row_axis=0, col_axis=1)
        if layout_override is not None and layout_override[k] is not None:
            layout = _check_layout_override(layout_override[k], svds, k)
        # ``top`` keyed on ``m_row`` and ``bot`` keyed on ``cut`` reproduce
        # ``M``'s sector keys; see :func:`_sector_blocks`.
        top_blocks = _sector_blocks(top, row_axis=0, col_axis=1)
        bot_blocks = _sector_blocks(bot, row_axis=0, col_axis=1)

        S = _retained_S_sym(svds, layout, M.dtype)
        P_left, P_right = _fishman_projectors_sym(
            top_blocks, bot_blocks, svds, S, layout
        )

        pinned = dict(svds)
        for q in layout.sectors:
            prev_left = None
            if prev is not None:
                candidate = prev[k].P_left.get(q)
                if candidate is not None and candidate.shape == P_left[q].shape:
                    prev_left = candidate
            P_left[q], P_right[q], U_q, Vh_q = _pin_bond_gauge_sector(
                P_left[q],
                P_right[q],
                svds[q].U,
                svds[q].Vh,
                layout.dim_of(q),
                prev_left,
            )
            pinned[q] = svds[q]._replace(U=U_q, Vh=Vh_q)

        P_left_t, P_right_t = _projector_tensors_sym(P_left, P_right, layout, top, bot)
        out.append(
            SymProjectors(
                P_left=P_left,
                P_right=P_right,
                S=S,
                sectors=pinned,
                layout=layout,
                P_left_t=P_left_t,
                P_right_t=P_right_t,
            )
        )
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    return out


# ---------------------------------------------------------------------------
# Forward: one left move, four rotations to a sweep (paper Eqs. 68-69)
# ---------------------------------------------------------------------------


def _unrotate_index_sym(slot: int, k: int) -> int:
    """Which original tensor sits in ``slot`` after ``k`` rotations.

    ``_ctm_root_implicit_asym._unrotate_index`` verbatim.  Slots and tensors
    are both numbered 1..4 in the ``C1..C4`` order and one rotation advances
    the label by one, so this is modular arithmetic and needs no symmetric
    counterpart of its own — it is repeated here only so the symmetric sweep
    reads without a cross-module hop.
    """
    return (slot - 1 + k) % 4 + 1


def _normalize_sym(t: SymmetricTensor) -> SymmetricTensor:
    """``t / max|t|`` — max-abs, not Frobenius, matching the dense module.

    ``_ctm_root_implicit_asym._normalize`` divides by ``jnp.max(jnp.abs(x))``,
    and the two do not agree up to a constant: the ratio depends on how the
    weight is spread over the tensor, so it changes from sweep to sweep and
    from tensor to tensor.  Since ``S`` is a variable of the characteristic
    equations and the renormalisation is what fixes the corner scale, a
    different normaliser is a different root, not a different presentation.

    The max is taken over the flat block buffer rather than ``todense()``:
    every entry outside a block is exactly zero, so the two agree, and the
    dense array is what this module exists to avoid materialising.
    """
    scale = jnp.max(jnp.abs(t._data))
    return t * (1.0 / (scale + 1e-300))


def renormalised_corner_sym(
    env_k: SymEnv,
    a_k: SymmetricTensor,
    projs_k: SymProjectors,
    projs_next: SymProjectors,
) -> SymmetricTensor:
    """Paper Eq. 68: the upper-left quadrant projected on *both* open legs.

    ``_ctm_root_implicit_asym._renormalised_corner`` with the two
    ``jnp.einsum`` index letters become labels.  The corner sits in the ``C1``
    slot of move ``k`` — above its vertical bond, hence ``P_left`` of move
    ``k`` — and in the ``C4`` slot of move ``k+1``, below its bond in the
    rotated frame, hence ``P_right`` of move ``k+1``.

    :func:`_as_matrix_sym` fuses the quadrant exactly as the dense
    ``.reshape(n, n)`` does, and the two fused legs are then *literally* the
    two cut indices the projectors expect:

    * axes 2-3 ``(chi_d, a_d)``, fused IN, are move ``k``'s ``top`` cut;
    * axes 0-1 ``(chi_r, a_r)``, fused OUT, are the ``(chi_u, a_u)`` of move
      ``k+1``'s *lower* quadrant — the rotation sends this quadrant's
      right-facing pair to the next move's upward-facing one — so they are
      move ``k+1``'s ``bot`` cut.

    That second identification is what makes the corner contract at all, and
    it is the one place where two different moves' charge sectors have to
    agree.  They do because ``T1.l``/``T1.r`` and ``a.l``/``a.r`` are dual
    pairs, so the two fusions (IN of the duals, OUT of the originals) produce
    the same charges.
    """
    quad = _as_matrix_sym(upper_left_quadrant_sym(env_k, a_k))
    quad = quad.relabels({"col": "cut_next", "row": "cut_k"})
    p_left = projs_k.P_left_t.relabels({"cut": "cut_k", "chi_new": "new_l"})
    p_right = projs_next.P_right_t.relabels({"cut": "cut_next", "chi_new": "new_r"})
    return contract(p_right, quad, p_left, output_labels=("new_l", "new_r"))


def renormalised_edge_sym(
    env_k: SymEnv,
    a_k: SymmetricTensor,
    projs_k: SymProjectors,
) -> SymmetricTensor:
    """Paper Eq. 69: the edge absorbs one ``a`` and is projected on both bonds.

    The dense reference is ``_renormalised_edge``, whose only ingredient is
    ``_left_move_pieces``' ``t4g``::

        jnp.einsum("hit,ujik->tukhj", T4, a).reshape(chi*d2, d2, chi*d2)

    — the left edge with one ``a`` absorbed, legs ``((chi,a), a_r, (chi,a))``.
    Here the two fused pairs are built with :func:`fuse_indices` instead of a
    reshape, with the *same* flows :func:`_as_matrix_sym` gives the quadrants:
    ``(T4.u, a.u)`` OUT, which is the lower quadrant's ``(chi_u, a_u)`` and so
    the ``bot`` cut, and ``(T4.d, a.d)`` IN, which is the upper quadrant's
    ``(chi_d, a_d)`` and so the ``top`` cut.  ``P_right`` therefore lands on
    the upper bond and ``P_left`` on the lower one — the dense
    ``einsum("ui,ixj,jd->dxu", P_bot, t4g, P_top)``, which reads backwards
    until the insertion ``Q_upper (P_left P_right) Q_lower`` is written out:
    the piece *above* a bond carries ``P_left``, the piece below carries
    ``P_right``.

    The open ``a.r`` leg becomes the new edge's middle leg.  It is left exactly
    as ``a`` hands it over — no relabel, no flow flip — because ``a.l`` and
    ``a.r`` are a dual pair, so next sweep's ``T4.a-leg x a.l`` contraction
    finds matching charges and opposite arrows on its own.
    """
    t4 = env_k.T4.relabels(
        dict(zip(env_k.T4.labels(), ("chi_d", "a_l", "chi_u"), strict=True))
    )
    a4 = a_k.relabels(
        dict(zip(a_k.labels(), ("a_u", "a_d", "a_l", "a_r"), strict=True))
    )
    g = contract(t4, a4, output_labels=("chi_u", "a_u", "a_r", "chi_d", "a_d"))
    g = fuse_indices(g, 0, 1, "cut_u", FlowDirection.OUT)
    g = fuse_indices(g, 2, 3, "cut_d", FlowDirection.IN)
    p_left = projs_k.P_left_t.relabels({"cut": "cut_d", "chi_new": "new_l"})
    p_right = projs_k.P_right_t.relabels({"cut": "cut_u", "chi_new": "new_r"})
    return contract(p_right, g, p_left, output_labels=("new_l", "a_r", "new_r"))


def sweep_sym(
    env: SymEnv,
    a: SymmetricTensor,
    chi: int,
    prev: list[SymProjectors] | None = None,
    *,
    layout_override: tuple[BondLayout | None, ...] | None = None,
) -> tuple[SymEnv, list[SymProjectors]]:
    """One simultaneous CTMRG sweep: all four directions from one environment.

    ``_ctm_root_implicit_asym.sweep`` function for function.  Every projector
    comes from the *same* environment and every corner and edge is renormalised
    against it; a sequential (Gauss-Seidel) sweep would have a fixed point that
    does not satisfy Eqs. 76-77, since those evaluate all four moves at one
    ``y``.

    Returns ``(env, projectors)``; the projectors feed the next sweep's
    per-sector gauge alignment.

    ``layout_override`` is forwarded to :func:`all_projectors_sym` and is
    test-only; see that function.
    """
    projs = all_projectors_sym(env, a, chi, prev, layout_override=layout_override)
    corners: list = [None] * 4
    edges: list = [None] * 4
    env_k, a_k = env, a
    for k in range(4):
        corners[_unrotate_index_sym(1, k) - 1] = _normalize_sym(
            renormalised_corner_sym(env_k, a_k, projs[k], projs[(k + 1) % 4])
        )
        edges[_unrotate_index_sym(4, k) - 1] = _normalize_sym(
            renormalised_edge_sym(env_k, a_k, projs[k])
        )
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    return SymEnv(*corners, *edges), projs


def _same_block_structure(x: SymmetricTensor, y: SymmetricTensor) -> bool:
    """Whether two symmetric tensors can be compared entry by entry.

    The dense module skips a sweep pair "whose shapes changed".  Here the shape
    is not the whole story: the retained charge distribution over the cut is
    data-dependent, so two sweeps can produce the same total ``chi`` split
    differently over sectors — same dense shape, different blocks.  Comparing
    the flat buffers in that case would subtract unrelated entries.

    Keys and buffer length are *not* sufficient, which is subtle enough to have
    shipped once: a layout that merely **permutes** multiplicities between
    sectors, ``{0: 1, 1: 2} -> {0: 2, 1: 1}``, keeps the diagonal key set
    ``((0,0), (1,1))`` and keeps the packed length (``1² + 2² == 2² + 1²``)
    while every block's shape and offset moves.  So the per-leg charge layout
    is compared too — it is what actually determines the packing.
    """
    if x._data.shape != y._data.shape or x._block_keys != y._block_keys:
        return False
    return all(
        np.array_equal(xi.sectors, yi.sectors)
        and np.array_equal(xi.multiplicities, yi.multiplicities)
        and xi.flow == yi.flow
        for xi, yi in zip(x.indices, y.indices, strict=True)
    )


def _max_abs_diff_sym(x: SymmetricTensor, y: SymmetricTensor) -> float:
    """Largest entry-wise difference, over the flat block buffer.

    Every entry outside a block is zero in both operands once
    :func:`_same_block_structure` holds, so this equals the max-abs difference
    of the dense arrays without building either.
    """
    return float(jnp.max(jnp.abs(x._data - y._data)))


def _phase_aligned_max_diff(prev: SymmetricTensor, cur: SymmetricTensor) -> float:
    """``max|cur/λ - prev|`` with ``λ = ⟨prev, cur⟩`` — the *root* residual.

    This is the corner/edge block of the characteristic equations themselves
    (``X'/λ - X``, :func:`characteristic_residual_sym`), evaluated on one sweep
    of the forward map.  It is deliberately *weaker* than the raw element-wise
    difference by exactly one complex scalar per tensor, and the reason is
    measured, not stylistic.

    On an unboosted D=2 Z2 state the sweep settles onto a **period-2 orbit**
    rather than a fixed point: ``sweep(x) = σ(x)`` with ``σ`` flipping the sign
    of ``C2`` and ``C4`` and leaving the other six tensors alone, reproduced to
    1e-12 while the raw residual sits at 1.880 for 200 sweeps.  That ``σ`` is a
    *bond gauge*, not a physical difference — with per-direction signs
    ``s = (+, +, -, -)`` on the four renormalised bonds, corner ``C_{j}`` picks
    up ``s_{j-1} s_j`` and every edge picks up ``s_j² = +1``, which is exactly
    the observed pattern (and the product of the four corner signs is ``+1``,
    the gauge-invariant consistency check).  The mechanism is that the pin of
    :func:`_pin_bond_gauge_sector` fixes the *projectors* — ``P_left`` and
    ``P_right`` are stationary to 3e-11 at every direction — while the sign
    that reaches a renormalised corner comes from the enlarged **quadrant**,
    which carries the previous sweep's corner sign directly.  Measured: ``top``
    flips at directions 1 and 3, ``bot`` at 0 and 2, projectors nowhere.

    Diagnosis note for the record: this is *not* the "a sector's retained
    dimension moved between sweeps, so the pin cold-restarts" mechanism the
    Phase 3 plan expected.  The layout is ``((-1, 2), (0, 2))`` on all four
    directions at every one of the 200 sweeps, so the warm pin never restarts.

    Since ``F`` is built from ``X'/λ - X`` tensor by tensor, such an orbit *is*
    a root — ``λ`` comes out ``-1`` on the two flipping corners and the residual
    is zero — so stopping on the raw difference rejects an environment the
    characteristic equations are perfectly happy with.  Aligning the phase is
    what makes the forward criterion agree with the equations it feeds.
    """
    lam = jnp.vdot(prev._data, cur._data)
    if float(jnp.abs(lam)) < 1e-12:
        # Orthogonal to the previous sweep: nothing to align, and dividing
        # would manufacture a tiny residual out of a large motion.
        return _max_abs_diff_sym(prev, cur)
    return float(jnp.max(jnp.abs(cur._data / lam - prev._data)))


def converge_sym(
    A: Tensor,
    chi: int,
    *,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    return_projectors: bool = False,
):
    """Sweep until the environment stops moving, element by element.

    ``_ctm_root_implicit_asym.converge`` with the dense sweep replaced by
    :func:`sweep_sym`.  The convergence test is deliberately **element-wise,
    not spectral**: corner singular values are invariant under independent
    rotations of each bond, so a spectral criterion calls convergence while
    the tensors are still moving — and the characteristic equations compare
    tensors, not spectra.  Each tensor is normalised before the comparison,
    and a pair whose block structure moved between sweeps is skipped rather
    than compared (see :func:`_same_block_structure`).

    The comparison is element-wise *up to one complex scalar per tensor*
    (:func:`_phase_aligned_max_diff`), which is precisely the freedom
    ``F``'s ``X'/λ - X`` normalisation quotients out.  Without it a residual
    bond-sign orbit — a genuine root — is reported as a permanent stall; with
    it the criterion says exactly "the characteristic equations hold".

    With ``return_projectors`` the final projector set comes back as a fourth
    element.  That is not a diagnostic: the converged environment sits in the
    bond gauge of the chain that built it, and the root parametrisation needs
    the same chain.  A cold re-pin fixes a *different* gauge and leaves ``y*``
    describing an environment it was not extracted from — a real state
    survives that (the gauge is a sign, one sweep absorbs it), a complex one
    does not (#721).
    """
    env, a = init_env_sym(A, chi)
    prev = None
    prev_projs: list[SymProjectors] | None = None
    residual = float("inf")
    converged = False
    iters = 0
    for it in range(int(max_iter)):
        env, prev_projs = sweep_sym(env, a, chi, prev_projs)
        iters = it + 1
        cur = tuple(t * (1.0 / (jnp.linalg.norm(t._data) + 1e-300)) for t in env)
        if prev is not None and all(
            _same_block_structure(c, q) for c, q in zip(cur, prev, strict=True)
        ):
            residual = max(
                _phase_aligned_max_diff(q, c) for c, q in zip(cur, prev, strict=True)
            )
            if iters >= min_iter and residual < conv_tol:
                converged = True
                break
        prev = cur
    meta: dict[str, Any] = {
        "iters": iters,
        "residual": residual,
        "converged": converged,
    }
    if return_projectors:
        return env, a, meta, prev_projs
    return env, a, meta


# ---------------------------------------------------------------------------
# Bond matrices on a *split* chi leg — the deleted ``kron``
# ---------------------------------------------------------------------------


def apply_bond_matrix(
    t: SymmetricTensor,
    mats: dict[int, jax.Array],
    *,
    axis: int,
    side: str = "left",
) -> SymmetricTensor:
    """Contract a per-sector matrix onto one renormalised chi leg.

    This replaces the dense modules' ``jnp.kron(root, eye_d2)``, and it is a
    replacement rather than a port because the identity behind that ``kron``
    **breaks under charge fusion**.  Dense, the cut leg is the flat product
    ``n = chi * d2`` with ``chi`` the slow index, so a matrix acting on the
    ``chi`` factor alone is ``kron(root, I_d2)``.  Fused, sector ``q`` of the
    cut leg is a direct sum over every ``(q_chi, q_d2)`` with
    ``q_chi + q_d2 = q``, and ``todense()`` orders it by charge rather than
    row-major in ``(chi, d2)`` — so within one sector the ``chi`` index is not
    the slow one, is not even contiguous, and no per-sector ``kron`` reproduces
    the operator.

    The fix is to never fuse in the first place.  The quadrant keeps its legs
    split as ``(chi_r, a_r, chi_d, a_d)``, the bond matrix is contracted onto
    the ``chi`` leg here, and :func:`_as_matrix_sym` fuses only afterwards, to
    feed the per-sector ``U``/``Vh`` algebra.  Nothing else in this module
    needs the fused form.

    ``mats`` is keyed by the **charge value carried on** ``axis``, which for a
    renormalised chi bond is the :class:`BondLayout` key ``q`` itself:
    :func:`bond_index_from_layout` builds the index with
    ``sectors = layout.sectors`` for either flow, so ``q`` and the charge
    coincide on both ends of the bond.

    ``side="left"`` is ``M @ t`` on that axis, ``side="right"`` is ``t @ M``;
    for the Hermitian Eq. 73 roots the two agree, for ``s`` they do not.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    blocks: dict[Any, jax.Array] = {}
    for key, block in t.blocks.items():
        q = int(key[axis])
        m = mats.get(q)
        if m is None:
            raise ValueError(
                f"no bond matrix for charge {q} on axis {axis} of a tensor whose "
                f"leg carries sectors {[int(c) for c in t.indices[axis].sectors]}; "
                f"the bond matrices are keyed {sorted(mats)}.  This is the "
                "environment and the truncation disagreeing about the retained "
                "charges — the bond the tensor lives on came from an earlier "
                "sweep, and the current cut keeps a different set.  It cannot "
                "happen at a fixed point, and :func:`root_parametrize_sym` "
                "sweeps rather than extracting when it does."
            )
        if side == "left":
            new = jnp.tensordot(m, block, axes=((1,), (axis,)))
            new = jnp.moveaxis(new, 0, axis)
        else:
            new = jnp.tensordot(block, m, axes=((axis,), (0,)))
            new = jnp.moveaxis(new, -1, axis)
        blocks[key] = new
    return SymmetricTensor._from_blocks_unchecked(blocks, t.indices)


def _map_env_roots_sym(env: SymEnv, roots: tuple, *, normalize: bool) -> SymEnv:
    """Absorb ``roots[k]`` onto corner/edge environment legs (paper Eq. 82).

    ``_ctm_root_implicit_asym._map_env_roots`` with ``left @ C @ right`` and
    ``einsum("ai,ixj,jb->axb")`` replaced by :func:`apply_bond_matrix`, and the
    same direction bookkeeping: corner ``k`` has its first chi leg on direction
    ``k-1``'s bond and its second on direction ``k``'s, while edge ``k`` takes
    ``roots[k]`` on both.  The indices are **covariant** (§V.3) direction
    numbers throughout, which is what :func:`sym_root_to_covariant_convention`
    hands over.
    """
    corners, edges = [], []
    for k in range(4):
        prev_dir, own_dir = (k - 1) % 4, k
        C = getattr(env, f"C{k + 1}")
        C = apply_bond_matrix(C, roots[prev_dir], axis=0, side="left")
        C = apply_bond_matrix(C, roots[own_dir], axis=1, side="right")
        E = getattr(env, f"T{k + 1}")
        E = apply_bond_matrix(E, roots[k], axis=0, side="left")
        E = apply_bond_matrix(E, roots[k], axis=2, side="right")
        if normalize:
            C = C * (1.0 / (jnp.linalg.norm(C._data) + 1e-300))
            E = E * (1.0 / (jnp.linalg.norm(E._data) + 1e-300))
        corners.append(C)
        edges.append(E)
    return SymEnv(*corners, *edges)


def remove_inverse_roots_sym(env: SymEnv, S_all: tuple) -> SymEnv:
    """Regular ``x`` -> modified ``(C̃, Ẽ)``: multiply by ``sqrt(S)``.

    Undoes the inverse square roots the forward projectors put on the
    environment legs, leaving the modified tensors of paper Eq. 82.  ``S_all``
    is covariant-indexed and per sector; the root is Denman-Beavers, never
    ``eigh``, for the reason in :func:`_fishman_projectors_sym`.
    """
    roots = tuple(sector_map(lambda m: _denman_beavers(m)[0], S) for S in S_all)
    return _map_env_roots_sym(env, roots, normalize=True)


def absorb_inverse_roots_sym(env_tilde: SymEnv, S_all: tuple) -> SymEnv:
    """Modified ``(C̃, Ẽ)`` -> regular ``x``: multiply by ``sqrt(S^-1)``.

    The **differentiable** direction.  ``S`` enters here, so the energy — which
    is evaluated on the regular environment — depends on ``S`` through this map
    and ``S̆`` picks that up.  Writing ``F`` in the regular variables instead
    sets that adjoint to zero, which is measurably wrong: dense, ``|S̆|`` comes
    out the same order as ``|C̆|``.
    """
    roots = tuple(sector_map(_inv_sqrt, S) for S in S_all)
    return _map_env_roots_sym(env_tilde, roots, normalize=True)


# ---------------------------------------------------------------------------
# Characteristic equations, covariant form (paper §V.3, Eqs. 71-88)
# ---------------------------------------------------------------------------


class SymRoot(NamedTuple):
    """Root variables plus the constants held fixed while differentiating.

    ``_ctm_root_implicit_asym.AsymRoot`` with every dense ``chi``-shaped array
    replaced by a ``{charge: block}`` dict.  ``s`` is a full ``chi_q x chi_q``
    **matrix** per sector, not a vector — see :func:`_fishman_projectors_sym`
    for why a diagonal ``S`` breaks the projector closure at first order.

    ``layouts`` has no dense counterpart and is not optional: "chi is an int"
    is exactly what stops being true here, and every shape in ``u``, ``v``,
    ``S`` and the reassembled bond index is read off it.  It is a frozen
    dataclass, hence a pytree *leaf*, so it stays static under AD.
    """

    env: SymEnv
    u: tuple[dict[int, jax.Array], ...]
    s: tuple[dict[int, jax.Array], ...]
    v: tuple[dict[int, jax.Array], ...]
    U_star: tuple[dict[int, jax.Array], ...]
    U_perp: tuple[dict[int, jax.Array], ...]
    Vh_star: tuple[dict[int, jax.Array], ...]
    Vh_perp: tuple[dict[int, jax.Array], ...]
    s_star_inv: tuple[dict[int, jax.Array], ...]
    layouts: tuple[BondLayout, ...]

    @property
    def y(self):
        return (self.env, self.u, self.s, self.v)


def sym_root_to_covariant_convention(root: SymRoot) -> SymRoot:
    """Relabel a forward-convention root into the one paper §V.3 uses.

    ``_ctm_root_implicit_asym.asym_root_to_covariant_convention`` per sector.
    The forward sweep truncates with the **left** half of the plane, gluing the
    upper-left quadrant to the lower-left one at the same rotation.  §V.3
    truncates with the **upper** half, gluing enlarged corner ``k`` to enlarged
    corner ``k+1``.  Those are the same truncation, because

        lower_left_quadrant_sym(env_k) == upper_left_quadrant_sym(env_{k-1})

    holds exactly, and therefore ``M_left(k) == M_up(k-1).T``.  Transposing a
    decomposition exchanges its isometries — ``(U S Vh).T = Vh.T S.T U.T``, and
    both factors stay isometric since ``Vh Vh† = 1`` gives
    ``(Vh.T)† Vh.T = conj(Vh Vh†) = 1`` — so §V.3's data at direction ``j`` is
    the forward data at rotation ``j+1`` with ``U`` and ``Vh`` swapped and every
    block transposed.  **Plain transpose, not adjoint**: no conjugation enters.

    Sector keys are carried across **unchanged**, and that is a measured
    statement rather than an omission.  Transposing a block does swap which leg
    its key was read off, so a remap ``q -> -q`` is the obvious hazard; it is
    not needed here because of which leg each layer keys by:

    * forward, :func:`sector_svd` keys ``M_left(k)`` by its row leg, which is
      the upper-left quadrant's fused ``(chi_r, a_r)`` — the **outer** leg,
      flow ``OUT``, so ``q = -charge``;
    * covariant, :func:`_cut_outer_blocks` keys the enlarged corner by that
      same outer leg.

    A fused quadrant matrix carries equal charges on its two legs (measured:
    block keys ``(0, 0)`` and ``(1, 1)`` on the D=2 Z2 fixture, with the
    ``OUT``/``IN`` flow pair making that charge conservation), so keying by the
    outer leg gives the same integer on both sides of every glue, and the
    forward layout keys ``{-1, 0}`` are exactly the covariant ones.  The
    alternative — keying the covariant layer by the *cut* leg, which is the
    literal transpose convention — would have needed ``q -> -q`` and, on
    ``ZnSymmetry(2)``, a modular one rather than an integer negation, since the
    live keys are ``{-1, 0}`` and not ``{0, 1}``.

    The environment itself is untouched; only which cut each ``S`` belongs to
    changes.  Feed the result to :func:`remove_inverse_roots_sym` and
    :func:`characteristic_residual_sym`, which are §V.3-indexed.
    """

    def shift(xs: tuple) -> tuple:
        return tuple(xs[(j + 1) % 4] for j in range(4))

    def shiftT(xs: tuple) -> tuple:
        return tuple({q: m.T for q, m in d.items()} for d in shift(xs))

    return SymRoot(
        env=root.env,
        u=shiftT(root.v),
        s=shiftT(root.s),
        v=shiftT(root.u),
        U_star=shiftT(root.Vh_star),
        U_perp=shiftT(root.Vh_perp),
        Vh_star=shiftT(root.U_star),
        Vh_perp=shiftT(root.U_perp),
        s_star_inv=shiftT(root.s_star_inv),
        layouts=shift(root.layouts),
    )


def _cut_outer_blocks(
    matrix: SymmetricTensor,
) -> dict[int, tuple[jax.Array, int, int]]:
    """``(cut, outer)`` block plus both legs' charges, keyed by the OUTER leg.

    :func:`_as_matrix_sym` returns a quadrant as ``(col, row)`` = ``(outer,
    cut)``; §V.3 orders an enlarged corner ``(cut | outer)``, hence the
    transpose.  The dense module pins that same transpose to 7.7e-16 against
    the reference implementation's dumped fixed point, and getting it backwards
    is an O(1) residual, not a small one.

    Keying by the outer leg — not the row, which is what
    :func:`_sector_blocks` would do — is what makes
    :func:`sym_root_to_covariant_convention` a pure transpose with no charge
    remap; see its docstring.  The two charges are returned alongside because
    the projector tensors have to be reassembled with explicit block keys, and
    deriving those from ``q`` by hand is exactly the arithmetic this module
    exists to avoid.
    """
    grouped = _group_blocks_by_bond_charge(matrix, [0], [1])
    out: dict[int, tuple[jax.Array, int, int]] = {}
    for q, entries in grouped.items():
        if len(entries) != 1:  # pragma: no cover - a fused matrix has one per q
            raise ValueError(
                f"sector {q} has {len(entries)} blocks; expected a fused matrix"
            )
        (outer_key, cut_key, block) = entries[0]
        out[q] = (jnp.transpose(block, (1, 0)), int(outer_key[0]), int(cut_key[0]))
    return out


def _decorated_quadrant(
    quad: SymmetricTensor,
    *,
    cut_left: dict[int, jax.Array] | None = None,
    outer_right: dict[int, jax.Array] | None = None,
) -> dict[int, tuple[jax.Array, int, int]]:
    """One enlarged corner with bond matrices on its two split chi legs.

    ``quad`` is :func:`upper_left_quadrant_sym`'s ``(chi_r, a_r, chi_d, a_d)``.
    ``cut_left`` left-multiplies the **cut** leg ``chi_d`` (axis 2) and
    ``outer_right`` right-multiplies the **outer** leg ``chi_r`` (axis 0),
    matching the dense ``K_L[k-1] @ EC[k] @ K_is[k]`` with ``EC`` in
    ``(cut | outer)`` order.  Both act on the *split* leg, which is the whole
    point (:func:`apply_bond_matrix`); the fusion happens afterwards.
    """
    if cut_left is not None:
        quad = apply_bond_matrix(quad, cut_left, axis=2, side="left")
    if outer_right is not None:
        quad = apply_bond_matrix(quad, outer_right, axis=0, side="right")
    return _cut_outer_blocks(_as_matrix_sym(quad))


def _projector_tensor_right(
    P_R: dict[int, jax.Array],
    charges: dict[int, tuple[int, int]],
    outer_index,
    layout: BondLayout,
) -> SymmetricTensor:
    """``P_R[k]`` as a tensor on ``(outer_k, chi_new)``, ``chi_new`` flowing OUT.

    ``P_R`` is the §V.3 partner of the forward ``P_left`` (the bridge maps one
    to the other), so its renormalised leg is the corner's ``new_l`` — flow
    ``OUT`` — and the corner it feeds is the one the forward sweep builds from
    ``P_left`` of the matching direction.  That agreement is not decorative:
    the residual subtracts against ``C̃`` from the forward chain, so a swapped
    flow is a subtraction between different index sets.
    """
    sym = outer_index.symmetry
    chi_new = bond_index_from_layout(layout, sym, FlowDirection.OUT, "chi_new")
    blocks = {(charges[q][0], int(q)): P_R[q].T for q in layout.sectors if P_R[q].size}
    return SymmetricTensor._from_blocks_unchecked(blocks, (outer_index, chi_new))


def _projector_tensor_left(
    P_L: dict[int, jax.Array],
    charges: dict[int, tuple[int, int]],
    cut_index,
    layout: BondLayout,
) -> SymmetricTensor:
    """``P_L[k]`` as a tensor on ``(cut_{k+1}, chi_new)``, ``chi_new`` IN.

    Mirror of :func:`_projector_tensor_right`: the bridge sends this to the
    forward ``P_right``, whose renormalised leg is the corner's ``new_r``.
    """
    sym = cut_index.symmetry
    chi_new = bond_index_from_layout(layout, sym, FlowDirection.IN, "chi_new")
    blocks = {(charges[q][1], int(q)): P_L[q] for q in layout.sectors if P_L[q].size}
    return SymmetricTensor._from_blocks_unchecked(blocks, (cut_index, chi_new))


def _vdot_sym(x: SymmetricTensor, y: SymmetricTensor) -> jax.Array:
    """``⟨x, y⟩`` over the blocks the two share.

    Deliberately **not** real-projected.  ``λ`` is genuinely complex for a
    complex state — dense at D=2, chi=4 the corner ``λ`` comes out around
    ``-2.1e2 - 3.6e2j`` — and taking the real part alone moved dense ``|F1|``
    from 2e-13 to 1.6e0.
    """
    keys = set(x.blocks) & set(y.blocks)
    xb, yb = x.blocks, y.blocks
    return sum(jnp.vdot(xb[k], yb[k]) for k in sorted(keys))


def _normalised_residual_sym(
    new: SymmetricTensor, cur: SymmetricTensor
) -> SymmetricTensor:
    """``new/λ - cur`` with ``λ = ⟨cur, new⟩``, on ``cur``'s block structure.

    The reference implementation's ``X'/λ - X`` rather than ``X' - λ X``.  The
    two differ by a row scaling of ``F``, which leaves the implicit gradient
    invariant but not the Jacobian conditioning, and it relies on every tensor
    in ``y`` having unit Frobenius norm — which
    :func:`root_parametrize_sym` arranges via
    :func:`remove_inverse_roots_sym`'s normalisation.

    Blocks are taken from ``cur``: the residual has to be a pytree matching
    ``y``, so a block ``new`` grew or lost would make ``F(y)`` un-subtractable
    from ``y`` in the adjoint solve.
    """
    lam = _vdot_sym(cur, new)
    nb = new.blocks
    extra = set(nb) - set(cur.blocks)
    if extra:  # pragma: no cover - guards a silent drop
        raise ValueError(
            f"the renormalised tensor has blocks {sorted(extra)} the environment "
            "does not; dropping them would make F(y) quietly ignore part of the "
            "network, which is the failure mode this module exists to avoid."
        )
    blocks = {
        key: (nb[key] / lam - block) if key in nb else -block
        for key, block in cur.blocks.items()
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, cur.indices)


def _isometries_sym(consts: SymRoot, u_all: tuple, v_all: tuple, k: int):
    """Paper Eqs. 71-72 per sector: ``U = U* + U_perp u``, ``Vh = V* + v V_perp``."""
    U = {
        q: consts.U_star[k][q] + consts.U_perp[k][q] @ u_all[k][q]
        for q in consts.U_star[k]
    }
    Vh = {
        q: consts.Vh_star[k][q] + v_all[k][q] @ consts.Vh_perp[k][q]
        for q in consts.Vh_star[k]
    }
    return U, Vh


def _north_edge_column(env_k: SymEnv, a_k: SymmetricTensor) -> SymmetricTensor:
    """The north edge with one ``a`` absorbed, both chi pairs fused.

    Dense this is ``einsum("xfy,fjlr->xljyr", T1, a).reshape(n, d2, n)``, whose
    two fused pairs are ``(T1.l, a.l)`` and ``(T1.r, a.r)``.  ``P_R[k]`` is
    built on the enlarged corner's *outer* leg — ``(T1.r, a.r)`` — and is
    applied to ``(T1.l, a.l)``; that reads like a mismatch and is not, because
    a 1x1 unit cell's two enlarged corners are translated copies of the same
    edge, so the glue joins one copy's right end to the next copy's left end.
    The flows follow: the left pair fuses ``IN`` against ``P_R``'s ``OUT`` and
    the right pair ``OUT`` against ``P_L``'s ``IN``, which is the *same*
    duals-fuse-to-matching-charges argument :func:`renormalised_corner_sym`
    already rests on.
    """
    t1 = env_k.T1.relabels(
        dict(zip(env_k.T1.labels(), ("chi_l", "a_u", "chi_r"), strict=True))
    )
    a4 = a_k.relabels(
        dict(zip(a_k.labels(), ("a_u", "a_d", "a_l", "a_r"), strict=True))
    )
    g = contract(t1, a4, output_labels=("chi_l", "a_l", "a_d", "chi_r", "a_r"))
    g = fuse_indices(g, 0, 1, "col_l", FlowDirection.IN)
    g = fuse_indices(g, 2, 3, "col_r", FlowDirection.OUT)
    return g


def characteristic_residual_sym(y, a: SymmetricTensor, consts: SymRoot, chi: int):
    """``F(y, p)`` in the covariant parametrisation of paper §V.3.

    ``y = (env_tilde, u, S, v)`` with *modified* corners and edges, so that
    holding ``U*`` and ``V*`` constant is licensed by Eq. 88.  Returns a pytree
    matching ``y``: four corners, four edges, and per direction one ``u``, one
    ``S`` and one ``v``, each a ``{charge: block}`` dict.

    The assembly mirrors
    ``_ctm_root_implicit_asym.asym_characteristic_residual_covariant``, which
    in turn mirrors the reference implementation's
    ``contract_asymmetric_characteristic_equation`` — and it uses a **different
    half of the plane from the forward sweep in this module**:

    * :func:`sweep_sym` truncates with the left half, gluing the upper-left
      quadrant to the lower-left one at the same rotation.
    * §V.3 truncates with the upper half, gluing enlarged corner ``k`` to
      enlarged corner ``k+1``.  So :func:`lower_left_quadrant_sym` is **not
      used here at all**, and the enlarged corner is the upper-left quadrant
      *transposed* into ``(cut | outer)`` order.

    A sweep and an equation that disagree give a fixed point that is not a
    root; that is how #718 started, and it is why
    :func:`sym_root_to_covariant_convention` exists rather than the two halves
    being reconciled by adjusting placements.

    The quartic roots of Eq. 73 go on the **cut legs**, never inside the
    projectors: they are equivariant under independent left/right rotations but
    their product is not ``S^-1``, so the closure breaks for a non-diagonal
    ``S`` (dense measured 2e-1 against 2.5e-2 gradient error when Phase 1 put
    them in the projectors).  They are attached by contraction on the split
    ``chi`` leg — see :func:`apply_bond_matrix` for why not ``kron``.
    """
    env_tilde, u_all, S_all, v_all = y
    layouts = consts.layouts
    for k, layout in enumerate(layouts):
        if layout.total != int(chi):  # pragma: no cover - guards a caller error
            raise ValueError(
                f"direction {k} retains {layout.total} states, not chi={chi}; "
                "the layout and the truncation have come apart."
            )

    # Eq. 73.  ``s = S^-1``, and the roots are Denman-Beavers (matmuls and
    # inverses only), so no decomposition re-enters ``F``.
    s_all = tuple(sector_map(jnp.linalg.inv, S) for S in S_all)
    root_L = tuple(
        {q: _quartic_root(m.conj().T @ m) for q, m in s.items()} for s in s_all
    )
    root_R = tuple(
        {q: _quartic_root(m @ m.conj().T) for q, m in s.items()} for s in s_all
    )
    env_mod = _modified_env_sym(env_tilde, s_all)

    # Pass 1: per direction the bare enlarged corner, the two decorated ones,
    # and the column the renormalised edge is built from.
    EC, As, B, sB, cols, outer_idx, cut_idx = [], [], [], [], [], [], []
    env_k, a_k = env_mod, a
    for k in range(4):
        km = (k - 1) % 4
        quad = upper_left_quadrant_sym(env_k, a_k)
        matrix = _as_matrix_sym(quad)
        outer_idx.append(matrix.indices[0])
        cut_idx.append(matrix.indices[1])
        EC.append(matrix)
        As.append(_decorated_quadrant(quad, cut_left=root_L[km], outer_right=s_all[k]))
        B.append(_decorated_quadrant(quad, outer_right=root_R[k]))
        sB.append(_decorated_quadrant(quad, cut_left=s_all[km], outer_right=root_R[k]))
        cols.append(_north_edge_column(env_k, a_k))
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)

    # Pass 2: the half-infinite environment at direction ``k`` glues the
    # decorated ``EC[k]`` to ``EC[k+1]`` across the cut, carrying exactly one
    # ``s_k`` (already absorbed into ``As[k]``'s outer leg).
    W, P_R_t, P_L_t = [], [], []
    U_all, Vh_all = [], []
    for k in range(4):
        kp = (k + 1) % 4
        layout = layouts[k]
        U, Vh = _isometries_sym(consts, u_all, v_all, k)
        U_all.append(U)
        Vh_all.append(Vh)
        for q in layout.sectors:
            if q not in As[k] or q not in B[kp]:  # pragma: no cover - guard
                missing = "EC[k]" if q not in As[k] else "EC[k+1]"
                raise ValueError(
                    f"direction {k} retains charge {q} but {missing} has no such "
                    "block; the two enlarged corners' sector keys have stopped "
                    "agreeing, which they cannot do at a fixed point."
                )
        W.append({q: As[k][q][0] @ B[kp][q][0] for q in layout.sectors})
        P_R = {q: U[q].conj().T @ As[k][q][0] for q in layout.sectors}
        P_L = {q: sB[kp][q][0] @ Vh[q].conj().T for q in layout.sectors}
        P_R_t.append(
            _projector_tensor_right(
                P_R,
                {q: As[k][q][1:] for q in layout.sectors},
                outer_idx[k],
                layout,
            )
        )
        P_L_t.append(
            _projector_tensor_left(
                P_L,
                {q: sB[kp][q][1:] for q in layout.sectors},
                cut_idx[kp],
                layout,
            )
        )

    # Pass 3: residuals.  Both projectors of a corner belong to the *same*
    # enlarged corner, one per group, so the corner takes ``P_R[k-1]`` — the
    # partner of the cut that ``EC[k]``'s first group sits on.
    corners: list = [None] * 4
    edges: list = [None] * 4
    R_u: list = [None] * 4
    R_S: list = [None] * 4
    R_v: list = [None] * 4
    for k in range(4):
        km = (k - 1) % 4
        layout = layouts[k]
        U, Vh = U_all[k], Vh_all[k]
        core = {q: U[q].conj().T @ W[k][q] @ Vh[q].conj().T for q in layout.sectors}
        # One ``λ`` per direction, summed over sectors, because ``S`` is
        # normalised over the whole cut and not sector by sector
        # (:func:`_retained_S_sym`); a per-sector ``λ`` would rescale each
        # sector's equation independently and move the root.
        lam_S = sum(jnp.vdot(S_all[k][q], core[q]) for q in layout.sectors)
        R_S[k] = {q: core[q] / lam_S - S_all[k][q] for q in layout.sectors}
        R_u[k] = {
            q: (consts.U_perp[k][q].conj().T @ W[k][q] @ Vh[q].conj().T)
            @ consts.s_star_inv[k][q]
            / lam_S
            - u_all[k][q]
            for q in layout.sectors
        }
        R_v[k] = {
            q: consts.s_star_inv[k][q]
            @ (U[q].conj().T @ W[k][q] @ consts.Vh_perp[k][q].conj().T)
            / lam_S
            - v_all[k][q]
            for q in layout.sectors
        }

        kp = (k + 1) % 4
        idx = _unrotate_index_sym(1, k)
        ec = EC[k].relabels({"col": "outer_k", "row": "cut_k"})
        p_r = P_R_t[km].relabels({"col": "cut_k", "chi_new": "new_l"})
        p_l = P_L_t[k].relabels({"row": "outer_k", "chi_new": "new_r"})
        C_new = contract(p_r, ec, p_l, output_labels=("new_l", "new_r"))
        C_cur = getattr(env_tilde, f"C{idx}")
        corners[idx - 1] = _normalised_residual_sym(C_new, C_cur)

        col = cols[k].relabels({"col_l": "cut_k", "col_r": "outer_k"})
        p_r_e = P_R_t[k].relabels({"col": "cut_k", "chi_new": "new_l"})
        E_new = contract(p_r_e, col, p_l, output_labels=("new_l", "a_d", "new_r"))
        E_cur = getattr(env_tilde, f"T{idx}")
        edges[idx - 1] = _normalised_residual_sym(E_new, E_cur)

    return (SymEnv(*corners, *edges), tuple(R_u), tuple(R_S), tuple(R_v))


def _modified_env_sym(env_tilde: SymEnv, s_all: tuple) -> SymEnv:
    """``iCi = s_{k-1} C̃_k s_k`` with the edges left alone.

    The full inverse goes on *both* corner legs, which is what puts the
    singular values explicitly into the contraction environment so that it
    transforms covariantly.  Edges already carry their ``s`` from the Eq. 82
    map, so :func:`remove_inverse_roots_sym` has done their half already.
    """
    corners = []
    for k in range(4):
        prev_dir, own_dir = (k - 1) % 4, k
        C = getattr(env_tilde, f"C{k + 1}")
        C = apply_bond_matrix(C, s_all[prev_dir], axis=0, side="left")
        C = apply_bond_matrix(C, s_all[own_dir], axis=1, side="right")
        corners.append(C)
    edges = [getattr(env_tilde, f"T{k + 1}") for k in range(4)]
    return SymEnv(*corners, *edges)


def _env_matches_layouts(env: SymEnv, projs: list[SymProjectors]) -> bool:
    """Whether every chi leg of ``env`` carries the layout the cut now chooses.

    The environment's bonds were built by the *previous* sweep's truncation and
    the layouts come from the current one.  At a fixed point they agree; away
    from one they need not, because the retained charge distribution is
    data-dependent — and then ``S`` is a matrix on charges the corner does not
    have, which is not a small error to absorb but a mis-shaped one.

    Forward-convention indices throughout: direction ``k``'s bond is leg 0 of
    corner ``unrot(1, k)``, leg 1 of corner ``unrot(1, k-1)``, and both chi legs
    of edge ``unrot(4, k)``.
    """
    for k in range(4):
        want = projs[k].layout
        sectors = want.sectors
        mults = [want.dim_of(q) for q in sectors]
        edge = getattr(env, f"T{_unrotate_index_sym(4, k)}")
        legs = (
            getattr(env, f"C{_unrotate_index_sym(1, k)}").indices[0],
            getattr(env, f"C{_unrotate_index_sym(1, (k - 1) % 4)}").indices[1],
            edge.indices[0],
            edge.indices[2],
        )
        for leg in legs:
            if [int(c) for c in leg.sectors] != sectors:
                return False
            if [int(m) for m in leg.multiplicities] != mults:
                return False
    return True


def root_parametrize_sym(
    env: SymEnv,
    a: SymmetricTensor,
    chi: int,
    *,
    prev_projs: list[SymProjectors] | None = None,
    pinv_rtol: float = 1e-10,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
    layout_override: tuple[BondLayout | None, ...] | None = None,
) -> tuple[SymRoot, float]:
    """Extract ``y* = (C̃, Ẽ, 0, S*, 0)`` and the frozen isometries.

    Returns the **covariant** root — already shifted, transposed and swapped by
    :func:`sym_root_to_covariant_convention` — together with
    ``‖F(y*)‖`` measured by :func:`characteristic_residual_sym`, so the number
    reported is the residual of the equations Phase 3's adjoint will actually
    differentiate rather than a proxy for it.

    Each environment tensor is rescaled to unit Frobenius norm so that the
    ``λ`` defined as an inner product really is the eigenvalue Eqs. 76-77 need;
    the modified tensors are renormalised again by
    :func:`remove_inverse_roots_sym`.  Rescaling is harmless: the energy is a
    ratio with equal numbers of corners and edges above and below.

    Pass ``prev_projs`` — :func:`converge_sym`'s fourth return value — whenever
    the environment came from a sweep chain.  A converged environment carries
    that chain's bond gauge, and a cold re-pin fixes a *different* one, which
    leaves ``y*`` describing an environment it was not extracted from.  For a
    real state one polish sweep absorbs that, because there the gauge is a sign
    and one sweep reproduces it exactly; for a complex state the gauge is a
    continuous phase, and the **corner** residual plateaus while the edges fall
    geometrically, because the corner is the only equation built from two
    directions' projectors and so the only one that sees a *relative* bond
    phase (#721).

    ``layout_override`` is forwarded to :func:`all_projectors_sym`, including
    to the polish sweeps so the forced truncation is applied consistently, and
    is test-only; see that function.
    """
    best: tuple[SymRoot, float] | None = None
    for _step in range(max(int(polish_steps), 1)):
        env = SymEnv(*[t * (1.0 / (jnp.linalg.norm(t._data) + 1e-300)) for t in env])
        projs = all_projectors_sym(
            env, a, chi, prev_projs, layout_override=layout_override
        )
        prev_projs = projs
        if not _env_matches_layouts(env, projs):
            # Away from a fixed point the cut can retain a different charge
            # distribution than the bonds the environment was built on.  One
            # sweep makes them agree by construction, so sweep rather than
            # extract a root whose ``S`` lives on charges the corners lack.
            env, prev_projs = sweep_sym(
                env, a, chi, projs, layout_override=layout_override
            )
            continue
        U_star, U_perp, Vh_star, Vh_perp, s_list, s_inv = [], [], [], [], [], []
        u_zero, v_zero = [], []
        for k in range(4):
            p = projs[k]
            layout = p.layout
            U_star.append(
                {q: p.sectors[q].U[:, : layout.dim_of(q)] for q in layout.sectors}
            )
            U_perp.append(
                {q: p.sectors[q].U[:, layout.dim_of(q) :] for q in layout.sectors}
            )
            Vh_star.append(
                {q: p.sectors[q].Vh[: layout.dim_of(q)] for q in layout.sectors}
            )
            Vh_perp.append(
                {q: p.sectors[q].Vh[layout.dim_of(q) :] for q in layout.sectors}
            )
            s_list.append(p.S)
            # Constant right/left preconditioner for Eqs. 78 and 80, diagonal
            # because the root ``S*`` is; the floor is relative to the largest
            # retained value of the *whole* cut, not of each sector, for the
            # reason :func:`sector_svd` floors globally.
            biggest = max(
                float(jnp.max(jnp.abs(jnp.diag(p.S[q])))) for q in layout.sectors
            )
            cutoff = pinv_rtol * biggest
            inv = {}
            for q in layout.sectors:
                diag = jnp.diag(p.S[q]).real
                inv_diag = jnp.where(
                    diag > cutoff, 1.0 / jnp.where(diag > cutoff, diag, 1.0), 0.0
                )
                inv[q] = jnp.diag(inv_diag).astype(p.S[q].dtype)
            s_inv.append(inv)
            # ``u`` is a coordinate on ``U``'s null space and ``v`` on ``Vh``'s,
            # and those live on *different* legs of the cut: ``U`` spans the
            # rows of sector ``q`` and ``Vh`` its columns.  Reading both
            # dimensions off ``U`` is right only when every sector block is
            # square, which the Z2 fixture is and the U(1) one is not — it fails
            # as a 6-against-4 matmul inside Eq. 71 rather than as a wrong
            # number, but only on the fragmenting layout.
            n_row = {q: p.sectors[q].U.shape[0] for q in layout.sectors}
            n_col = {q: p.sectors[q].Vh.shape[1] for q in layout.sectors}
            dtype = env.C1._data.dtype
            u_zero.append(
                {
                    q: jnp.zeros(
                        (n_row[q] - layout.dim_of(q), layout.dim_of(q)), dtype=dtype
                    )
                    for q in layout.sectors
                }
            )
            v_zero.append(
                {
                    q: jnp.zeros(
                        (layout.dim_of(q), n_col[q] - layout.dim_of(q)), dtype=dtype
                    )
                    for q in layout.sectors
                }
            )

        root = SymRoot(
            env=env,
            u=tuple(u_zero),
            s=tuple(s_list),
            v=tuple(v_zero),
            U_star=tuple(U_star),
            U_perp=tuple(U_perp),
            Vh_star=tuple(Vh_star),
            Vh_perp=tuple(Vh_perp),
            s_star_inv=tuple(s_inv),
            layouts=tuple(p.layout for p in projs),
        )
        root_cov = sym_root_to_covariant_convention(root)
        tilde = remove_inverse_roots_sym(root_cov.env, root_cov.s)
        y_star = (tilde, root_cov.u, root_cov.s, root_cov.v)
        R = characteristic_residual_sym(y_star, a, root_cov, chi)
        residual = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.abs(_leaf_data(leaf)) ** 2)
                    for leaf in jax.tree.leaves(R)
                )
            )
        )
        if best is None or residual < best[1]:
            best = (root_cov._replace(env=tilde), residual)
        if residual <= polish_tol:
            break
        env, prev_projs = sweep_sym(env, a, chi, projs)

    if best is None:
        raise ValueError(
            f"root_parametrize_sym: after {polish_steps} sweeps the environment's "
            "chi bonds never carried the charge distribution the truncation "
            "chooses, so no root could be extracted.  That is a state whose CTM "
            "has not settled — the retained charges are still moving between "
            "sweeps — not a bug in the equations; converge further first."
        )
    return best


def _leaf_data(leaf):
    """The flat buffer of a pytree leaf, whether it is a tensor or an array.

    ``SymmetricTensor`` flattens to a *single* leaf — its packed ``_data``
    buffer (PR #87) — whose L2 norm is the Frobenius norm, so a residual norm
    taken this way needs no adapter and no ``todense()``.
    """
    return leaf._data if isinstance(leaf, SymmetricTensor) else leaf


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

# Positional leg names of :class:`CTMTensorEnv`, in this module's post-swap
# axis order.  ``compute_energy_ctm_tensor`` contracts purely by label, so the
# boundary between the two conventions is a rename plus the axis swap of
# :func:`swap_env_convention_sym` — nothing else.  The D² leg of each edge is
# renamed too (a sweep leaves it carrying whichever of ``a``'s four legs the
# absorption left open) but is *not* flow-flipped: opposite legs of ``a`` are a
# dual pair, so the renamed leg already meets the open double layer with
# matching charges and the opposite arrow.
_CTM_LABELS = {
    "C1": ("c1_d", "c1_r"),
    "C2": ("c2_l", "c2_d"),
    "C3": ("c3_u", "c3_l"),
    "C4": ("c4_r", "c4_u"),
    "T1": ("t1_l", "u2", "t1_r"),
    "T2": ("t2_u", "r2", "t2_d"),
    "T3": ("t3_r", "d2", "t3_l"),
    "T4": ("t4_d", "l2", "t4_u"),
}


def _to_ctm_env_sym(env: SymEnv) -> CTMTensorEnv:
    """This module's uniform ring -> ``CTMTensorEnv``'s.

    The swap is load-bearing and must not be dropped: this module closes the
    ring uniformly while ``CTMTensorEnv`` closes it with ``C4`` transposed and
    ``T3``, ``T4`` reversed.  Reinterpreting one as the other glues the network
    wrongly — 1.5% on the energy at D=2 chi=4 and the ±2.121e-3 per-bond
    antisymmetry that was #718, which read as an unlicensed gauge in Eq. 88 and
    was this relabelling all along.  It is a *no-op* on the initialiser, whose
    ``C4`` is symmetric and whose ``T3``/``T4`` are palindromic, so the mistake
    is invisible until the environment is genuinely asymmetric — i.e. after the
    first sweep.

    Unlike the dense ``_to_ctm_env`` this takes no template.  A converged
    symmetric environment's chi legs carry the charge layout the truncation
    chose, which is data-dependent and generally *not* the initialiser's, so
    borrowing the template's indices would assert a layout the data does not
    have.  The tensors carry their own indices; only the labels are renamed.
    """
    env = swap_env_convention_sym(env)
    fields = {}
    for name, labels in _CTM_LABELS.items():
        t = getattr(env, name)
        fields[name] = t.relabels(dict(zip(t.labels(), labels, strict=True)))
    return CTMTensorEnv(**fields)


def sym_energy(A: Tensor, env: SymEnv, gate) -> jax.Array:
    """Nearest-neighbour energy per site from a symmetric CTM environment."""
    return compute_energy_ctm_tensor(A, _to_ctm_env_sym(env), gate)


# ---------------------------------------------------------------------------
# The adjoint (paper Eq. 18)
# ---------------------------------------------------------------------------


def _pytree_norm(tree) -> float:
    """``‖·‖_2`` over a pytree, Frobenius on every ``SymmetricTensor`` leaf.

    A ``SymmetricTensor`` flattens to a *single* leaf — its packed ``_data``
    buffer — and every entry outside a block is exactly zero, so the flat L2
    norm of that buffer already is the Frobenius norm of the tensor.  No
    adapter, and nothing is densified.
    """
    return float(
        jnp.sqrt(
            sum(jnp.sum(jnp.abs(_leaf_data(x)) ** 2) for x in jax.tree.leaves(tree))
        )
    )


def _zeros_like_sectors(xs: tuple) -> tuple:
    """A tuple of ``{charge: zeros}`` dicts shaped like ``xs``."""
    return tuple({q: jnp.zeros_like(m) for q, m in d.items()} for d in xs)


def _sub_jaxprs(params: dict):
    """Every jaxpr nested inside an equation's parameters.

    ``pjit``, ``closed_call``, ``scan``, ``cond`` and the custom-derivative
    primitives all hide their bodies in ``eqn.params``, sometimes wrapped in a
    ``ClosedJaxpr`` and sometimes in a list.  Duck-typed rather than matched
    against ``jax.core`` classes, which have moved between JAX versions.
    """
    for value in params.values():
        items = value if isinstance(value, (list, tuple)) else (value,)
        for item in items:
            jaxpr = getattr(item, "jaxpr", item)
            if hasattr(jaxpr, "eqns"):
                yield jaxpr


def _primitive_names(closed_jaxpr) -> set[str]:
    """The set of primitives a jaxpr applies, sub-jaxprs included."""
    names: set[str] = set()
    stack = [getattr(closed_jaxpr, "jaxpr", closed_jaxpr)]
    while stack:
        jaxpr = stack.pop()
        for eqn in jaxpr.eqns:
            names.add(eqn.primitive.name)
            stack.extend(_sub_jaxprs(eqn.params))
    return names


def _double_layer_sym(A: Tensor) -> SymmetricTensor:
    """``a`` in this module's ``(u, d, l, r)`` order, still block sparse.

    The same two lines :func:`init_env_sym` uses, factored out because the
    gradient needs ``a`` as a *function of* ``A`` — that dependence is the
    whole of ``∂_p F``.
    """
    a_t = _build_double_layer_tensor(A)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    return a_t.transpose(perm)


def sym_root_implicit_energy_and_grad(
    A: Tensor,
    gate,
    *,
    chi: int = 8,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    polish_steps: int = 40,
    polish_tol: float = 1e-10,
    solve_tol: float = 1e-8,
    solve_maxiter: int = 400,
    solve_restart: int = 30,
    root_residual_warn: float = 1e-6,
):
    """Energy and ``dE/dA`` for a 1x1 unit cell, block sparse throughout.

    ``_ctm_root_implicit_asym.asym_root_implicit_energy_and_grad`` with every
    dense array replaced by a :class:`~tenax.core.SymmetricTensor` or a
    ``{charge: block}`` dict.  Returns ``(energy, dE/dA, diagnostics)``, with
    the gradient a ``SymmetricTensor`` on ``A``'s own indices — the charge
    structure is part of the answer, not scaffolding around it.

    Root implicit differentiation of paper §V: the environment is characterised
    by ``F(y, p) = 0`` in the modified variables ``y = (C̃, Ẽ, u, S, v)`` of
    Eqs. 76-80, and Eq. 18 turns that into a gradient without back-propagating
    a single decomposition.  On the symmetric path that is not a convenience
    but the point: the block-sparse SVD/eigh VJPs are the #566 compile wall and
    the #687 accuracy floor, and ``F`` is contractions, inverses and
    Denman-Beavers roots only — see
    ``diagnostics["backward_jaxpr"]``, which the tests assert contains neither
    ``svd`` nor ``eigh``.

    Structure, mirroring the dense function step for step:

    1. :func:`converge_sym` with ``return_projectors=True``.  The chain is not
       a diagnostic — the converged environment sits in *its* bond gauge and a
       cold re-pin describes a different one (#721).
    2. :func:`root_parametrize_sym` for ``y*`` and the frozen constants.
    3. the energy on the **regular** environment recovered by
       :func:`absorb_inverse_roots_sym`.  That is the only path by which ``S``
       reaches the energy, and it is what makes ``S̆`` nonzero; writing ``F``
       in the regular variables sets that adjoint to zero, which was #718.
    4. ``ў`` = VJP of the energy with respect to ``y``.
    5. ``F̆ ∂_y F = ў`` by GMRES.
    6. ``dE/dA = ∂_p E - F̆ ∂_p F``.

    On the gauge, and why there is no phase-fixing condition here.  An
    independent phase on *each* environment tensor is an exact null direction
    of ``∂_y F``: the normalisation is a ratio, ``X'/⟨X, X'⟩`` is invariant
    under a phase on ``X'`` while picking up ``e^{iγ}`` from ``X``, so
    ``R = X'/⟨X, X'⟩ - X`` is phase-*covariant* tensor by tensor and vanishes
    along all eight orbits at once.  ``∂_y F`` is therefore singular, and two
    things make that harmless.  The adjoint system is consistent because ``ў``
    is orthogonal to every null direction — the energy is invariant along each
    orbit — and the leftover freedom in ``F̆`` cannot reach the gradient at
    all, because differentiating ``F(y*(p), p) = 0`` puts ``∂_p F`` in
    **range**(``∂_y F``), to which the cokernel is orthogonal.  GMRES converges
    on the singular system.  Only the first of the two can fail, and it fails
    at the *energy* boundary rather than in this module — which is exactly
    where #718 went wrong — so ``diagnostics["gauge_consistency"]`` measures it
    rather than assuming it.

    The pairing there is ``Re Σ g·δz``, **unconjugated**: that is how JAX pairs
    a complex cotangent with a tangent.  The conjugated form manufactures
    violations that are not there — it reported 2.59e-2 against a true
    -1.5e-16 on the dense path.

    The adjoint solve runs on the *real embedding* of the pytree
    (``_ctm_c4v_root_implicit._solve_root_adjoint``, shared with the dense
    path).  ``F`` conjugates, so its VJP is only real-linear and a
    complex-linear Krylov method does not apply; for a real state the embedding
    is the plain real solve at no extra cost.  No ``SymmetricTensor`` adapter
    is needed on top of that — the tensors flatten to a single array leaf and
    ``jax.tree.unflatten`` puts them back.
    """
    from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint

    if not isinstance(A, SymmetricTensor):
        raise TypeError(
            "sym_root_implicit_energy_and_grad expects a SymmetricTensor; the "
            "dense 1x1 path is _ctm_root_implicit_asym."
            "asym_root_implicit_energy_and_grad."
        )

    A_const = jax.lax.stop_gradient(A)
    env, a_const, meta, forward_projs = converge_sym(
        A_const,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        min_iter=min_iter,
        return_projectors=True,
    )
    root_cov, root_residual = root_parametrize_sym(
        env,
        a_const,
        chi,
        prev_projs=forward_projs,
        polish_steps=polish_steps,
        polish_tol=polish_tol,
    )
    if root_residual > root_residual_warn:
        warnings.warn(
            f"Symmetric root implicit AD: ‖F(y*)‖ = {root_residual:.3e} exceeds "
            f"{root_residual_warn:.1e}; the implicit-function gradient is "
            "correspondingly inaccurate (paper Fig. 1).",
            RuntimeWarning,
            stacklevel=2,
        )

    # ``root_parametrize_sym`` already returns the covariant convention and
    # already carries the *modified* environment in ``.env``.
    tilde = root_cov.env
    S_star = root_cov.s
    y_star = (tilde, root_cov.u, S_star, root_cov.v)

    def energy_of(A_live, env_tilde, S_all):
        return sym_energy(A_live, absorb_inverse_roots_sym(env_tilde, S_all), gate)

    energy, vjp_energy = jax.vjp(energy_of, A, tilde, S_star)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    # ``u`` and ``v`` carry no cotangent: the energy does not see the
    # null-space coordinates, only their effect through the root.
    y_bar = (
        tilde_bar,
        _zeros_like_sectors(root_cov.u),
        S_bar,
        _zeros_like_sectors(root_cov.v),
    )

    y_bar_norm = _pytree_norm(y_bar)
    gauge_consistency = 0.0
    for bar, tensor in zip(tilde_bar, tilde, strict=True):
        # JAX pairs cotangents unconjugated: ``Re Σ g·δz`` with the tangent
        # ``δz = i x`` of the phase orbit.  Conjugating instead reports a
        # violation that is not there.
        pairing = float(jnp.real(jnp.sum(bar._data * (1j * tensor._data))))
        scale = y_bar_norm * float(jnp.linalg.norm(tensor._data)) + 1e-300
        gauge_consistency = max(gauge_consistency, abs(pairing) / scale)
    if gauge_consistency > 1e-8:
        warnings.warn(
            f"Symmetric root implicit AD: the energy cotangent has a "
            f"{gauge_consistency:.3e} relative component along an environment "
            "tensor's phase, which is a null direction of ∂F/∂y. The adjoint "
            "system is inconsistent by that much and the gradient is "
            "correspondingly unreliable; the energy is supposed to be "
            "invariant under every such phase (#721).",
            RuntimeWarning,
            stacklevel=2,
        )

    def F_of_y(y):
        return characteristic_residual_sym(y, a_const, root_cov, chi)

    F_at_root, vjp_y = jax.vjp(F_of_y, y_star)
    covariant_residual = _pytree_norm(F_at_root)
    if covariant_residual > root_residual_warn:
        warnings.warn(
            f"Symmetric root implicit AD: the covariant ‖F(y*)‖ = "
            f"{covariant_residual:.3e} exceeds {root_residual_warn:.1e}. The "
            "gradient solves the adjoint of equations that y* does not "
            "satisfy, so it is correspondingly inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )
    F_bar, solve_resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0],
        y_bar,
        tol=solve_tol,
        maxiter=solve_maxiter,
        restart=solve_restart,
    )

    def F_of_p(A_live):
        return characteristic_residual_sym(
            y_star, _double_layer_sym(A_live), root_cov, chi
        )

    _, vjp_p = jax.vjp(F_of_p, A)
    grad = grad_direct - vjp_p(F_bar)[0]

    # The claim the whole method rests on, in a form a test can assert: the
    # three functions this gradient differentiates, and the three pullbacks it
    # runs.  The forward decomposition that built ``U*``/``Vh*`` happens in
    # ``all_projectors_sym``, outside all six, and enters them as a constant.
    #
    # The **forward** jaxprs are the load-bearing half, which is not obvious.
    # The VJP of a decomposition contains no decomposition — it is the stored
    # ``U``, ``s``, ``Vh`` and a divided difference — so an ``svd`` smuggled
    # into ``F`` would leave the pullback's primitive list clean while putting
    # exactly the ``1/(s_i^2 - s_j^2)`` this method exists to avoid back into
    # the gradient.  The pullbacks are listed too, to catch a custom rule that
    # introduces one.
    #
    # What is recorded is the **primitive listing**, not the pretty-printed
    # jaxpr, and that is not tidiness either.  ``jaxpr.__str__`` names
    # variables ``a, b, ..., z, aa, ab, ...``; these programs run to ~40k
    # equations, so the namer gets as far as a variable literally called
    # ``svd``, and ``"svd" not in str(jaxpr)`` is then false against a jaxpr
    # that applies no such primitive.  The dense module's version of this test
    # is safe only because its jaxpr is small enough.  Listing primitives says
    # exactly what is meant, and is 300 bytes instead of 4 MB.
    primitives: set[str] = set()
    for fn, args in (
        (energy_of, (A, tilde, S_star)),
        (F_of_y, (y_star,)),
        (F_of_p, (A,)),
    ):
        primitives |= _primitive_names(jax.make_jaxpr(fn)(*args))
    for pullback, cotangent in (
        (vjp_energy, jnp.ones((), dtype=energy.dtype)),
        (vjp_y, y_bar),
        (vjp_p, F_bar),
    ):
        primitives |= _primitive_names(jax.make_jaxpr(pullback)(cotangent))
    backward_jaxpr = "\n".join(sorted(primitives))

    return (
        energy,
        grad,
        {
            **meta,
            "root_residual": root_residual,
            "covariant_residual": covariant_residual,
            "gmres_residual": float(solve_resid),
            "gauge_consistency": gauge_consistency,
            # ``|S̆| / |C̆, Ĕ|``.  The energy reaches ``S`` only through the
            # Eq. 82 absorption, so writing ``F`` in the regular variables
            # would make this exactly zero — the #718 bug.  Measured ~0.4
            # here, i.e. the same order as the environment's own cotangent.
            "singular_value_adjoint_ratio": _pytree_norm(S_bar)
            / (_pytree_norm(tilde_bar) + 1e-300),
            "backward_jaxpr": backward_jaxpr,
        },
    )
