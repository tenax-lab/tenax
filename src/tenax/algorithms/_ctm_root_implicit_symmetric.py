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

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_root_implicit_asym import _inv_sqrt
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
    "all_projectors_sym",
    "converge_sym",
    "half_infinite_sym",
    "init_env_sym",
    "lower_left_quadrant_sym",
    "renormalised_corner_sym",
    "renormalised_edge_sym",
    "rotate_a_sym",
    "rotate_env_sym",
    "swap_env_convention_sym",
    "sweep_sym",
    "sym_energy",
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


def all_projectors_sym(
    env: SymEnv,
    a: SymmetricTensor,
    chi: int,
    prev: list[SymProjectors] | None = None,
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
    """
    out: list[SymProjectors] = []
    env_k, a_k = env, a
    for k in range(4):
        top, bot = _cut_halves_sym(env_k, a_k)
        M = contract(top, bot, output_labels=("m_row", "m_col"))
        svds, layout = sector_svd(M, chi, row_axis=0, col_axis=1)
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
) -> tuple[SymEnv, list[SymProjectors]]:
    """One simultaneous CTMRG sweep: all four directions from one environment.

    ``_ctm_root_implicit_asym.sweep`` function for function.  Every projector
    comes from the *same* environment and every corner and edge is renormalised
    against it; a sequential (Gauss-Seidel) sweep would have a fixed point that
    does not satisfy Eqs. 76-77, since those evaluate all four moves at one
    ``y``.

    Returns ``(env, projectors)``; the projectors feed the next sweep's
    per-sector gauge alignment.
    """
    projs = all_projectors_sym(env, a, chi, prev)
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
    """
    return x._data.shape == y._data.shape and x._block_keys == y._block_keys


def _max_abs_diff_sym(x: SymmetricTensor, y: SymmetricTensor) -> float:
    """Largest entry-wise difference, over the flat block buffer.

    Every entry outside a block is zero in both operands once
    :func:`_same_block_structure` holds, so this equals the max-abs difference
    of the dense arrays without building either.
    """
    return float(jnp.max(jnp.abs(x._data - y._data)))


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
                _max_abs_diff_sym(c, q) for c, q in zip(cur, prev, strict=True)
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
