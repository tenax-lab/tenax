"""Tensor index (leg) metadata with labels and charge information.

Each leg of a tensor is described by a TensorIndex, which carries:
- The symmetry group governing charges on this leg
- The charge of each basis state along this leg
- The flow direction (incoming/outgoing)
- A label (string or integer) for identification and label-based contraction

Labels are the primary user-facing API for specifying contractions.
Two legs with the same label on different tensors will be automatically
contracted when contract() or TensorNetwork.contract() is called.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    Conservation law: sum_i(flow_i * charge_i) == symmetry.identity()
    for any valid block of a symmetric tensor.
    """

    IN = 1
    OUT = -1


@dataclass(frozen=True, slots=True)
class TensorIndex:
    """Metadata for one leg (index) of a symmetric tensor.

    TensorIndex is frozen and slots-based for memory efficiency — large
    networks create millions of these objects.

    Attributes:
        symmetry:  The symmetry group governing charges on this leg.
        charges:   1-D numpy int32 array of length D (bond dimension).
                   charges[i] is the charge of basis state i.
        flow:      Whether this leg is incoming (IN) or outgoing (OUT).
        label:     Human-readable or integer identifier for this leg.
                   Shared labels across tensors drive automatic contraction.

    Example:
        >>> u1 = U1Symmetry()
        >>> idx = TensorIndex(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN, label="left")
        >>> idx.dim
        3
        >>> idx.dual().flow
        <FlowDirection.OUT: -1>
    """

    symmetry: BaseSymmetry
    charges: np.ndarray  # shape (D,), dtype int32
    flow: FlowDirection
    label: Label = ""

    def __post_init__(self) -> None:
        if self.charges.ndim != 1:
            raise ValueError(f"charges must be 1-D, got shape {self.charges.shape}")
        # Coerce to int32 if needed (use object.__setattr__ since frozen)
        if self.charges.dtype != np.int32:
            object.__setattr__(self, "charges", self.charges.astype(np.int32))

    @property
    def dim(self) -> int:
        """Bond dimension of this leg (number of basis states)."""
        return len(self.charges)

    def dual(self) -> TensorIndex:
        """Return a new TensorIndex with flipped flow and dual charges.

        Used when reversing a leg direction. The dual of an IN leg is an
        OUT leg with negated (for U(1)) or modular-negated (for Zn) charges.

        Returns:
            New TensorIndex with opposite flow and dual charges.
        """
        return TensorIndex(
            symmetry=self.symmetry,
            charges=self.symmetry.dual(self.charges),
            flow=FlowDirection(-int(self.flow)),
            label=self.label,
        )

    def flip_flow(self) -> TensorIndex:
        """Return a new TensorIndex with flipped flow but unchanged charges.

        Unlike :meth:`dual`, this does **not** negate the charge values.
        Used by :meth:`Tensor.bar` to produce opposite flows for contraction
        while keeping charges identical for equality-based block matching.

        Returns:
            New TensorIndex with opposite flow and identical charges.
        """
        return TensorIndex(
            symmetry=self.symmetry,
            charges=self.charges,
            flow=FlowDirection(-int(self.flow)),
            label=self.label,
        )

    def relabel(self, new_label: Label) -> TensorIndex:
        """Return a new TensorIndex with a different label, otherwise identical.

        Args:
            new_label: The replacement label.

        Returns:
            New frozen TensorIndex with the updated label.
        """
        return TensorIndex(
            symmetry=self.symmetry,
            charges=self.charges,
            flow=self.flow,
            label=new_label,
        )

    def is_dual_of(self, other: TensorIndex) -> bool:
        """Check if this index is the exact dual of other.

        Strict check: requires opposite flows AND charge arrays that are
        dual of each other. Used to validate that two legs can be contracted
        while preserving exact charge conservation.

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
        # Use value-based hash consistent with __eq__: two TensorIndex objects
        # that compare equal (same symmetry type+params, same charges, same flow,
        # same label) must have the same hash. id(self.symmetry) would break this
        # invariant when two separately-constructed symmetry instances are equal.
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
            f"flow={self.flow.name}, label={self.label!r})"
        )
