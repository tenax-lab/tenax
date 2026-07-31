"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3).

Nothing here knows about CTM.  It knows about charge sectors: how a chi bond
is split across them, how to decompose a block-diagonal matrix sector by
sector, and how to get back to a :class:`SymmetricTensor`.  A future CTM
layer (``_ctm_root_implicit_symmetric``, not yet written) will call into this
at the cut and nowhere else; the decomposition/reassembly helpers implied by
that split are forthcoming and not part of this module yet.

The split exists because the two halves fail differently.  A bug here is a
wrong *shape* or a wrong charge and shows up as an exception; a bug in the CTM
layer is a mis-glued network and shows up as a wrong number.  Keeping them
apart keeps the tests that catch them apart too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core import BlockKey, SymmetricTensor
from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.symmetry import BaseSymmetry, ProductSymmetry
from tenax.linalg import _group_blocks_by_bond_charge


@dataclass(frozen=True, slots=True)
class BondLayout:
    """How a truncated chi bond is distributed over charge sectors.

    This is the symmetric analogue of "chi is an int".  Every downstream shape
    — ``u``, ``v``, ``S``, ``U*``, ``U_perp`` — is read off it, and it is
    *frozen* at the converged point: if it moved under AD the adjoint would be
    solving a different-sized system than the forward built.

    Deliberately **not** a JAX pytree: a ``NamedTuple`` with a ``dict`` field
    flattens its dimensions to pytree leaves, which makes them tracers under
    ``jax.jit``/``jax.grad`` and makes the layout itself unhashable (dicts
    aren't hashable), so it cannot be closed over as static metadata either
    — exactly where this needs to live, since ``custom_vjp``'s
    ``nondiff_argnums`` requires hashability. A frozen dataclass over a tuple
    of ``(charge, dim)`` pairs is opaque to ``jax.tree_util`` and hashable,
    matching the repo's existing pattern for static shape metadata
    (:class:`~tenax.core.index.TensorIndex`).

    Attributes:
        dims: Sorted ``(charge, dim)`` pairs, zero-dim entries excluded.
              Construct via :meth:`from_dims` rather than directly, so the
              sort/filter invariant is guaranteed rather than assumed.
    """

    dims: tuple[tuple[int, int], ...]

    @classmethod
    def from_dims(cls, dims: Mapping[int, int]) -> BondLayout:
        """Build a :class:`BondLayout` from a ``{charge: dim}`` mapping.

        Normalises once: drops zero-dim entries, sorts by charge, and
        rejects negative dims. Downstream reads (``total``, ``sectors``,
        ``dim_of``) then trust the invariant instead of re-deriving it.
        """
        bad = {q: d for q, d in dims.items() if d < 0}
        if bad:
            raise ValueError(
                f"BondLayout dims must be non-negative, got negative entries {bad} "
                f"(full input: {dict(dims)})"
            )
        pairs = tuple(sorted((q, d) for q, d in dims.items() if d > 0))
        return cls(dims=pairs)

    @property
    def total(self) -> int:
        return sum(d for _, d in self.dims)

    @property
    def sectors(self) -> list[int]:
        """Retained charges, sorted, excluding sectors that kept nothing."""
        return [q for q, _ in self.dims]

    def dim_of(self, charge: int) -> int:
        """Return the retained dimension for ``charge`` (0 if not retained)."""
        for q, d in self.dims:
            if q == charge:
                return d
        return 0


def bond_index_from_layout(
    layout: BondLayout,
    symmetry: BaseSymmetry,
    flow: FlowDirection,
    label: Label,
) -> TensorIndex:
    """Build the truncated chi bond described by ``layout``.

    Sectors that retained nothing are dropped rather than kept at width zero.
    A zero multiplicity is a legal but poisonous index: it survives every
    charge check and then produces zero-size blocks that contract to zero.
    """
    sectors = layout.sectors
    if not sectors:
        raise ValueError(
            f"BondLayout retained no charges; the cut is empty (dims={layout.dims})."
        )
    return TensorIndex(
        symmetry=symmetry,
        sectors=np.asarray(sectors, dtype=np.int32),
        multiplicities=np.asarray([layout.dim_of(q) for q in sectors], dtype=np.int32),
        flow=flow,
        label=label,
    )


class SectorSVD(NamedTuple):
    """A full dense SVD of one charge sector of the cut, plus its keys.

    ``U`` and ``Vh`` are *full* (``full_matrices=True``): the null-space
    columns ``U[:, k:]`` and rows ``Vh[k:]`` are the ``U_perp`` / ``Vh_perp``
    of paper Eq. 71-72, restricted to this sector.  #715 planned to avoid
    materialising them; per sector these matrices are small, and the
    complementary-projector form would add a redundant null direction to
    ``d_yF`` for nothing.
    """

    U: jax.Array
    s: jax.Array
    Vh: jax.Array
    S_keep_diag: jax.Array
    row_key: BlockKey
    col_key: BlockKey


def sector_svd(
    matrix: SymmetricTensor,
    chi: int,
    *,
    row_axis: int,
    col_axis: int,
    floor_rtol: float = 1e-12,
) -> tuple[dict[int, SectorSVD], BondLayout]:
    """Full SVD per charge sector, then one global top-``chi`` truncation.

    The matrix is block diagonal in the bond charge, so each sector decomposes
    independently — but the *truncation* is global, over the union of every
    sector's spectrum.  That is what makes the retained charge distribution
    data-dependent, and hence what :class:`BondLayout` has to record.

    The floor is taken against the largest singular value of the **whole cut**,
    not of each sector.  Flooring per sector would rescale a sector whose
    values are all numerically zero up to the retained range and promote its
    noise to a kept direction.  Phase 1 floors for the same reason — an early
    environment is rank deficient and a singular ``S`` makes the matrix inverse
    square root produce NaNs (``_ctm_root_implicit_asym.all_projectors``).
    """
    grouped = _group_blocks_by_bond_charge(matrix, [row_axis], [col_axis])

    raw: dict[int, tuple] = {}
    for q, entries in grouped.items():
        if len(entries) != 1:  # pragma: no cover - a fused matrix has one per q
            raise ValueError(
                f"sector {q} has {len(entries)} blocks; expected a fused matrix"
            )
        (row_key, col_key, block) = entries[0]
        # Orientation contract: U spans row_axis, Vh spans col_axis (standard
        # SVD convention M = U diag(s) Vh, where U spans M's row space).
        #
        # `_group_blocks_by_bond_charge` returns `block` in the tensor's own
        # (native) axis order, *not* reordered to (row_axis, col_axis) --
        # `row_key`/`col_key` are correctly labelled, but the array itself
        # isn't. `jnp.transpose(block, (row_axis, col_axis))` moves the
        # tensor's `row_axis` to position 0 and `col_axis` to position 1, so
        # after this line `block` really is (row, col) and the SVD below
        # honours the names it's given. Skipping this step is exactly the
        # #718-flavoured bug this function used to have: parameter names that
        # don't match behaviour, invisible on square sectors, silently wrong
        # on non-square ones.
        block = jnp.transpose(block, (row_axis, col_axis))
        U, s, Vh = jnp.linalg.svd(block, full_matrices=True)
        raw[q] = (U, s, Vh, row_key, col_key)

    # Global truncation across sectors.
    ranked = sorted(
        (
            (float(sv), q, i)
            for q, (_U, s, *_r) in raw.items()
            for i, sv in enumerate(s)
        ),
        key=lambda t: -t[0],
    )
    dims: dict[int, int] = dict.fromkeys(raw, 0)
    for _sv, q, _i in ranked[: int(chi)]:
        dims[q] += 1
    layout = BondLayout.from_dims(dims)

    biggest = max((float(raw[q][1][0]) for q in raw), default=0.0)
    floor = floor_rtol * biggest

    sectors: dict[int, SectorSVD] = {}
    for q, (U, s, Vh, row_key, col_key) in raw.items():
        sectors[q] = SectorSVD(
            U=U,
            s=s,
            Vh=Vh,
            S_keep_diag=jnp.maximum(s, floor),
            row_key=row_key,
            col_key=col_key,
        )
    return sectors, layout


def sector_map(
    fn: Callable[[jax.Array], jax.Array], mats: dict[int, jax.Array]
) -> dict[int, jax.Array]:
    """Apply a dense matrix function to every charge sector.

    Phase 1's ``_denman_beavers`` / ``_inv_sqrt`` / ``_quartic_root`` go
    through here untouched.  They are ``jnp.linalg.inv`` plus matmuls, so they
    put no decomposition back into ``F`` — which is the whole point of the
    method — and they are already correct on a general (non-diagonal, complex)
    matrix, which is what ``S`` becomes in the reverse pass.

    Charge conservation forbids entries between sectors, so a block-diagonal
    matrix function *is* the per-sector function.  There is nothing to
    generalise.
    """
    return {q: fn(m) for q, m in mats.items()}


def tensor_from_sector_matrices(
    mats: dict[int, jax.Array],
    *,
    row_index: TensorIndex,
    col_index: TensorIndex,
    row_axis: int,
    col_axis: int,
) -> SymmetricTensor:
    """Rebuild a 2-leg :class:`SymmetricTensor` from per-sector matrices.

    Inverse of :func:`sector_svd`'s orientation step: ``mats[q]`` is in
    ``(row, col)`` orientation and is written back in the tensor's native
    axis order.

    Block keys are *charge values* per axis, not bare copies of the dict key
    ``q``.  ``q`` is the bond charge :func:`_group_blocks_by_bond_charge`
    (called by :func:`sector_svd` with ``left_leg_positions=[row_axis]``)
    computes as ``flow_charge(row_index.flow, key[row_axis])``, seeded with the
    identity.  Inverting it recovers the row charge, because ``flow_charge`` is
    an involution on canonical representatives; charge conservation then pins
    the column charge, since ``fuse(q, flow_charge(col_flow, q_col))`` must be
    the identity:

    * ``key[row_axis] = flow_charge(row_flow, q)``
    * ``key[col_axis] = flow_charge(col_flow, dual(q))``

    Both reduce to plain ``q`` only in the IN-row/OUT-col case (flows +1/-1)
    that every fixture in this module happens to use — verified empirically
    against ``sector_svd`` + ``._validate()`` + a numeric round trip for all
    four IN/OUT combinations, not assumed from the IN/OUT case alone.

    This goes through the symmetry rather than negating the integer (#734).
    Raw negation agreed with ``dual`` for U(1) and, modulo ``n``, for ``Z_n``;
    what it did *not* agree with is the canonical representative a
    :class:`~tenax.core.index.TensorIndex` now carries.  With both flows the
    same way a ``Z2`` partner came out ``-1`` where the leg's sectors are
    ``[0, 1]``, so the key named a charge the leg did not have and every later
    contraction silently dropped that block — the projector pair in
    ``_ctm_root_implicit_symmetric`` collapsed its corners to a single sector.

    For :class:`~tenax.core.symmetry.ProductSymmetry` the failure was worse
    still: charges are bit-packed, so ``-encode(1, 2)`` decodes as ``(-1, -3)``
    and ``fuse(q, -q)`` is ``-65536`` rather than the identity — a genuine
    conservation violation, not a relabelling.  The arithmetic above is
    group-correct for product symmetries too, but the rest of the Phase 3 path
    has not been exercised on them, so they stay refused here until #734's
    ProductSymmetry enablement verifies the whole path end to end.
    """
    if {row_axis, col_axis} != {0, 1}:
        raise ValueError("row_axis and col_axis must be 0 and 1 in some order")

    sym = row_index.symmetry
    if isinstance(sym, ProductSymmetry):
        raise NotImplementedError(
            "tensor_from_sector_matrices now derives both block-key charges "
            "through symmetry.flow_charge/dual, which is group-correct for "
            "ProductSymmetry as well, but the rest of #715 Phase 3 has never "
            "been run on bit-packed charges -- where a wrong partner is a "
            "conservation violation (-encode(1, 2) decodes as (-1, -3), not "
            "(-1, -2)) rather than a relabelling, and "
            "_from_blocks_unchecked would pass it straight to a contraction. "
            "Until #734's ProductSymmetry enablement verifies that path, "
            "Phase 3 supports U(1) and Z_n only."
        )

    indices: list[TensorIndex] = [None, None]  # type: ignore[list-item]
    indices[row_axis] = row_index
    indices[col_axis] = col_index

    blocks: dict[BlockKey, jax.Array] = {}
    for q, mat in mats.items():
        if mat.size == 0:
            continue
        q_arr = np.array([int(q)], dtype=np.int32)
        key = [0, 0]
        key[row_axis] = int(sym.flow_charge(row_index.flow, q_arr)[0])
        key[col_axis] = int(sym.flow_charge(col_index.flow, sym.dual(q_arr))[0])
        blocks[tuple(key)] = mat if row_axis == 0 else mat.T
    return SymmetricTensor._from_blocks_unchecked(blocks, tuple(indices))
