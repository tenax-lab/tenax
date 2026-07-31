"""Tensor index (leg) metadata with sector-based charge information.

Each leg of a tensor is described by a TensorIndex, which carries:
- The symmetry group governing charges on this leg
- Sorted unique charge sectors and their multiplicities
- The flow direction (incoming/outgoing)
- A label (string or integer) for identification and label-based contraction
- Optional FuseInfo recording how fused legs can be split

Labels are the primary user-facing API for specifying contractions.
Two legs with the same label on different tensors will be automatically
contracted when contract() or TensorNetwork.contract() is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from tenax.core.symmetry import BaseSymmetry

# Label type: strings (descriptive) or integers (positional)
Label = str | int


class FlowDirection(IntEnum):
    """Flow direction of a tensor leg.

    IN (+1):  Incoming leg — corresponds to a "ket" index, arrow pointing
              into the tensor. Positive charge flows in.
    OUT (-1): Outgoing leg — corresponds to a "bra" index, arrow pointing
              out of the tensor. Positive charge flows out.

    Conservation law: for any valid block of a symmetric tensor, fusing each
    leg's flow-weighted charge (``symmetry.flow_charge(flow_i, charge_i)``,
    i.e. the charge itself on an IN leg and its group inverse on an OUT leg)
    must yield ``symmetry.identity()``.  See :func:`_net_charge`, which is the
    one place that evaluates it.  This is *not* the same as
    ``sum_i(flow_i * charge_i)``, which only coincides for U(1) (#734).
    """

    IN = 1
    OUT = -1


@dataclass(frozen=True)
class FuseInfo:
    """Records parent indices of a fused leg for split_index.

    When two legs are merged via ``fuse_indices``, the resulting fused leg
    stores a ``FuseInfo`` so that ``split_index`` can reverse the operation.
    Recursive: parents may themselves have ``fuse_info`` (e.g. ``((a,b), c)``).

    Attributes:
        parent_indices: The two TensorIndex objects that were fused to
                        produce this leg, in the order (axis_a, axis_b).
        fused_flow:     The flow the fused leg had at fusion time.  Recorded so
                        ``split_index`` can invert the fusion using the original
                        sign even if the fused leg's flow was later flipped
                        (e.g. by ``bar``/``flip_flow``).  ``None`` for legacy
                        fuse_info, in which case split falls back to the fused
                        leg's current flow.
    """

    parent_indices: tuple[TensorIndex, ...]
    fused_flow: FlowDirection | None = None


@dataclass(frozen=True, slots=True)
class TensorIndex:
    """Metadata for one leg (index) of a symmetric tensor.

    TensorIndex is frozen and slots-based for memory efficiency — large
    networks create millions of these objects.

    The internal representation stores sorted unique charge sectors and
    their multiplicities alongside the dense charges array. This makes
    sector-level operations explicit and lays the foundation for
    non-abelian symmetries.

    Attributes:
        symmetry:        The symmetry group governing charges on this leg.
        sectors:         Sorted unique charges, shape (n_sectors,), dtype int32.
        multiplicities:  Number of basis states per sector, shape (n_sectors,),
                         dtype int32. ``dim == sum(multiplicities)``.
        flow:            Whether this leg is incoming (IN) or outgoing (OUT).
        label:           Human-readable or integer identifier for this leg.
        fuse_info:       If this leg was produced by ``fuse_indices``, records
                         the parent indices so ``split_index`` can reverse it.

    Example:
        >>> u1 = U1Symmetry()
        >>> idx = TensorIndex.from_charges(u1, np.array([-1, 0, 1], dtype=np.int32),
        ...                                FlowDirection.IN, label="left")
        >>> idx.dim
        3
        >>> idx.sectors
        array([-1,  0,  1], dtype=int32)
    """

    symmetry: BaseSymmetry
    sectors: np.ndarray  # shape (n_sectors,), dtype int32, sorted
    multiplicities: np.ndarray  # shape (n_sectors,), dtype int32
    flow: FlowDirection
    label: Label = ""
    fuse_info: FuseInfo | None = None
    # Original charges array preserved for from_dense/todense compatibility.
    # Set by from_charges(); None when constructed directly (sorted order used).
    _charges_cache: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.sectors.ndim != 1:
            raise ValueError(f"sectors must be 1-D, got shape {self.sectors.shape}")
        if self.multiplicities.ndim != 1:
            raise ValueError(
                f"multiplicities must be 1-D, got shape {self.multiplicities.shape}"
            )
        if len(self.sectors) != len(self.multiplicities):
            raise ValueError(
                f"sectors and multiplicities must have same length, "
                f"got {len(self.sectors)} vs {len(self.multiplicities)}"
            )
        if self.sectors.dtype != np.int32:
            object.__setattr__(self, "sectors", self.sectors.astype(np.int32))
        if self.multiplicities.dtype != np.int32:
            object.__setattr__(
                self, "multiplicities", self.multiplicities.astype(np.int32)
            )

        # Canonicalise sectors through the symmetry and merge duplicates.
        #
        # Charge representatives are a basis convention, not physics, so
        # rewriting them is free.  But ``flow_charge`` is an involution only on
        # canonical representatives, and label *equality* is how blocks get
        # paired during contraction -- so two representatives of one sector
        # (Z2 ``-1`` and ``1``) must never coexist on a leg (#734).
        #
        # Every construction path funnels through __post_init__, so this also
        # repairs ``dual()``, which sorted without merging and could emit
        # sectors=[0, 1, 1] in violation of the sorted-unique invariant.
        #
        # Sorting is part of the same repair, not a nicety: ``multiplicity``,
        # ``has_sector`` and ``sector_offset`` all use ``np.searchsorted``, so
        # unsorted-but-canonical sectors return silently wrong answers rather
        # than raising.  The short-circuit below is also the fast path -- the
        # overwhelmingly common case is already canonical and already sorted.
        sectors = self.sectors
        canon = self.symmetry.canonicalize_charges(sectors)
        if np.array_equal(canon, sectors) and (
            len(sectors) < 2 or bool(np.all(sectors[1:] > sectors[:-1]))
        ):
            return
        uniq, inverse = np.unique(canon, return_inverse=True)
        merged = np.zeros(len(uniq), dtype=np.int32)
        np.add.at(merged, inverse, self.multiplicities)
        object.__setattr__(self, "sectors", uniq.astype(np.int32))
        object.__setattr__(self, "multiplicities", merged)

    @classmethod
    def from_charges(
        cls,
        symmetry: BaseSymmetry,
        charges: np.ndarray,
        flow: FlowDirection,
        label: Label = "",
    ) -> TensorIndex:
        """Construct a TensorIndex from a dense charges array.

        Extracts sorted unique sectors and their multiplicities from the
        charges array while preserving the original charge ordering for
        backward-compatible dense tensor operations.

        Args:
            symmetry: Symmetry group for this leg.
            charges:  1-D integer array of charge per basis state.
            flow:     Flow direction (IN or OUT).
            label:    Leg label for contraction matching.

        Returns:
            New TensorIndex with sectors/multiplicities derived from charges.
        """
        charges = np.asarray(charges, dtype=np.int32)
        if charges.ndim != 1:
            raise ValueError(f"charges must be 1-D, got shape {charges.shape}")
        # Canonicalise before deriving sectors so the cached dense array and the
        # sector table agree on representatives (#734).  Element-wise, so basis
        # ordering within the leg is preserved.
        charges = symmetry.canonicalize_charges(charges)
        sectors, multiplicities = np.unique(charges, return_counts=True)
        obj = cls(
            symmetry=symmetry,
            sectors=sectors.astype(np.int32),
            multiplicities=multiplicities.astype(np.int32),
            flow=flow,
            label=label,
        )
        # Preserve original ordering for from_dense/todense compatibility
        object.__setattr__(obj, "_charges_cache", charges)
        return obj

    @property
    def charges(self) -> np.ndarray:
        """Dense charges array for this leg.

        Returns the original ordering if constructed via ``from_charges``,
        otherwise returns sorted charges (``np.repeat(sectors, multiplicities)``).
        """
        if self._charges_cache is not None:
            return self._charges_cache
        return np.repeat(self.sectors, self.multiplicities)

    @property
    def dim(self) -> int:
        """Bond dimension of this leg (number of basis states)."""
        return int(np.sum(self.multiplicities))

    @property
    def n_sectors(self) -> int:
        """Number of distinct charge sectors."""
        return len(self.sectors)

    def multiplicity(self, charge: int) -> int:
        """Return multiplicity of a charge sector (0 if not present)."""
        idx = np.searchsorted(self.sectors, charge)
        if idx < len(self.sectors) and self.sectors[idx] == charge:
            return int(self.multiplicities[idx])
        return 0

    def sector_offset(self, charge: int) -> int:
        """Return starting position of charge sector in the *sorted* charges array.

        Note: this gives the offset in the sorted representation
        (``np.repeat(sectors, multiplicities)``), not in the original charges
        array from ``from_charges``.
        """
        idx = np.searchsorted(self.sectors, charge)
        if idx < len(self.sectors) and self.sectors[idx] == charge:
            return int(np.sum(self.multiplicities[:idx]))
        raise KeyError(f"Charge {charge} not in sectors {self.sectors}")

    def has_sector(self, charge: int) -> bool:
        """Check if this index has a given charge sector."""
        idx = np.searchsorted(self.sectors, charge)
        return bool(idx < len(self.sectors) and self.sectors[idx] == charge)

    def dual(self) -> TensorIndex:
        """Return a new TensorIndex with flipped flow and dual charges.

        Used when reversing a leg direction. The dual of an IN leg is an
        OUT leg with negated (for U(1)) or modular-negated (for Zn) charges.

        Returns:
            New TensorIndex with opposite flow and dual charges.
        """
        dual_sectors = self.symmetry.dual(self.sectors)
        sort_perm = np.argsort(dual_sectors)
        obj = TensorIndex(
            symmetry=self.symmetry,
            sectors=dual_sectors[sort_perm],
            multiplicities=self.multiplicities[sort_perm],
            flow=FlowDirection(-int(self.flow)),
            label=self.label,
        )
        # Compute dual charges preserving original ordering if available.
        #
        # The canonicalisation is defence in depth, not a fix: for every
        # symmetry currently in the tree ``dual`` already maps canonical
        # representatives to canonical representatives, so it never fires
        # (reverting it fails no test).  It is kept so that a future symmetry
        # whose ``dual`` leaves the fundamental domain cannot silently
        # desynchronise the cached dense charges from ``sectors``, which
        # __post_init__ does canonicalise.  It costs one extra ``% n``
        # allocation per ``dual()`` for Z_n.
        if self._charges_cache is not None:
            object.__setattr__(
                obj,
                "_charges_cache",
                self.symmetry.canonicalize_charges(
                    self.symmetry.dual(self._charges_cache)
                ),
            )
        return obj

    def flip_flow(self) -> TensorIndex:
        """Return a new TensorIndex with flipped flow but unchanged charges.

        Unlike :meth:`dual`, this does **not** negate the charge values.
        Used by :meth:`Tensor.bar` to produce opposite flows for contraction
        while keeping charges identical for equality-based block matching.

        Returns:
            New TensorIndex with opposite flow and identical charges.
        """
        obj = TensorIndex(
            symmetry=self.symmetry,
            sectors=self.sectors,
            multiplicities=self.multiplicities,
            flow=FlowDirection(-int(self.flow)),
            label=self.label,
            fuse_info=self.fuse_info,
        )
        if self._charges_cache is not None:
            object.__setattr__(obj, "_charges_cache", self._charges_cache)
        return obj

    def relabel(self, new_label: Label) -> TensorIndex:
        """Return a new TensorIndex with a different label, otherwise identical.

        Args:
            new_label: The replacement label.

        Returns:
            New frozen TensorIndex with the updated label.
        """
        obj = TensorIndex(
            symmetry=self.symmetry,
            sectors=self.sectors,
            multiplicities=self.multiplicities,
            flow=self.flow,
            label=new_label,
            fuse_info=self.fuse_info,
        )
        if self._charges_cache is not None:
            object.__setattr__(obj, "_charges_cache", self._charges_cache)
        return obj

    def is_dual_of(self, other: TensorIndex) -> bool:
        """Check if this index is the exact dual of other.

        Strict check: requires opposite flows AND charge sectors that are
        dual of each other with matching multiplicities.

        Args:
            other: Another TensorIndex to compare against.

        Returns:
            True if self and other are exact duals.
        """
        if type(self.symmetry) is not type(other.symmetry):
            return False
        if self.flow == other.flow:
            return False
        if self.dim != other.dim:
            return False
        return np.array_equal(self.charges, self.symmetry.dual(other.charges))

    def compatible_with(self, other: TensorIndex) -> bool:
        """Check if this index can be connected to other in a network.

        Looser than is_dual_of: requires same symmetry type, same bond
        dimension, and opposite flows. Does not require exact charge matching
        (useful for connecting tensors where charges may differ by relabeling).

        Args:
            other: Another TensorIndex to check compatibility with.

        Returns:
            True if the two indices can be connected.
        """
        return (
            type(self.symmetry) is type(other.symmetry)
            and self.dim == other.dim
            and self.flow != other.flow
        )

    def __hash__(self) -> int:
        return hash((self.symmetry, self.charges.tobytes(), int(self.flow), self.label))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorIndex):
            return NotImplemented
        return (
            type(self.symmetry) is type(other.symmetry)
            and self.symmetry == other.symmetry
            and np.array_equal(self.charges, other.charges)
            and self.flow == other.flow
            and self.label == other.label
        )

    def __repr__(self) -> str:
        return (
            f"TensorIndex(sym={self.symmetry!r}, dim={self.dim}, "
            f"n_sectors={self.n_sectors}, flow={self.flow.name}, "
            f"label={self.label!r})"
        )


def _net_charge(
    indices: Sequence[TensorIndex],
    key: Sequence[int],
) -> int:
    """Return the net fused charge of a block key, as a Python int.

    This is the only sanctioned way to evaluate a block's conservation law: a
    block is valid exactly when ``_net_charge(indices, key) == sym.identity()``.

    ``sum(int(idx.flow) * int(q) for ...)`` is **not** equivalent. It assumes the
    group inverse is integer negation and the group operation is integer
    addition — true for U(1), accidentally true for ``Z_n`` whenever two or more
    legs fuse afterwards, and false for the bit-packed charges of
    :class:`~tenax.core.symmetry.ProductSymmetry` (#734).

    This is a thin adapter: it reads the flow off each index and delegates the
    arithmetic to :meth:`~tenax.core.symmetry.BaseSymmetry.net_charge`, so that
    no charge is ever inverted or combined outside the symmetry class.

    Args:
        indices: Tensor indices, one per leg.
        key:     One charge per leg, in the same order.

    Returns:
        The fused net charge.

    Raises:
        ValueError: If ``indices`` is empty, or if ``key`` has a different
            length than ``indices``.
    """
    if len(indices) == 0:
        raise ValueError("_net_charge requires at least one index")
    pairs = list(zip(indices, key, strict=True))
    return indices[0].symmetry.net_charge(
        [q for _, q in pairs], [idx.flow for idx, _ in pairs]
    )
