"""Tensor storage classes: DenseTensor and SymmetricTensor.

DenseTensor wraps a plain JAX array with index metadata (labels, flows, charges).
SymmetricTensor stores only the symmetry-allowed charge sectors (block-sparse).

Both are registered as JAX pytree nodes, making them compatible with
jax.jit, jax.grad, jax.vmap, etc.

Block-sparse design (SymmetricTensor):
- Blocks are stored as a dict[BlockKey, jax.Array]
- BlockKey = tuple of one representative charge per leg
- Only blocks satisfying the conservation law are stored
- Block arrays are the pytree leaves (traced by JAX)
- Block keys and index metadata are pytree aux data (static)
- jax.jit recompiles only when block structure changes (bond dim change)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, Label, TensorIndex, _net_charges

# Block key: tuple of one charge value per leg identifying a charge sector
BlockKey = tuple[int, ...]


def _charge_summary(idx: TensorIndex) -> str:
    """Format index sectors as ``{charge: count, ...}``."""
    parts = [f"{int(q)}:{int(m)}" for q, m in zip(idx.sectors, idx.multiplicities)]
    return "{" + ", ".join(parts) + "}"


def _tensor_box_repr(
    type_name: str,
    indices: tuple[TensorIndex, ...],
    info_lines: list[str],
    charge_lines: list[str] | None = None,
) -> str:
    """Build an ASCII box with legs extending left (IN) and right (OUT).

    Args:
        type_name:    Short type name shown inside the box.
        indices:      Tuple of TensorIndex objects for the legs.
        info_lines:   Additional lines inside the box (dtype, block stats).
        charge_lines: Optional charge summary lines appended below the box.
    """
    in_legs = [(idx.label, idx.dim) for idx in indices if idx.flow == FlowDirection.IN]
    out_legs = [
        (idx.label, idx.dim) for idx in indices if idx.flow == FlowDirection.OUT
    ]

    # Format leg strings
    in_strs = [f"{lbl} ({dim}) ──>" for lbl, dim in in_legs]
    out_strs = [f"<── {lbl} ({dim})" for lbl, dim in out_legs]

    # Box content: type name + info lines
    box_content = [type_name] + info_lines

    # Number of rows = max(in legs, out legs, content lines)
    n_rows = max(len(in_strs), len(out_strs), len(box_content))

    # Pad lists to n_rows
    while len(in_strs) < n_rows:
        in_strs.append("")
    while len(out_strs) < n_rows:
        out_strs.append("")
    while len(box_content) < n_rows:
        box_content.append("")

    # Compute widths
    left_w = max((len(s) for s in in_strs), default=0)
    box_w = max(len(s) for s in box_content) + 2  # 1 space padding each side

    # Build lines
    lines = []
    # Top border
    lines.append(f"{'':<{left_w}}┌{'─' * box_w}┐")
    for i in range(n_rows):
        left = f"{in_strs[i]:>{left_w}}"
        content = f" {box_content[i]:<{box_w - 1}}"
        right_edge = "├" if out_strs[i] else "│"
        left_edge = "┤" if in_strs[i] else "│"
        right = out_strs[i]
        lines.append(f"{left}{left_edge}{content}{right_edge}{right}")
    # Bottom border
    lines.append(f"{'':<{left_w}}└{'─' * box_w}┘")

    # Charge summary below the box
    if charge_lines:
        lines.append(" charges: " + charge_lines[0])
        for cl in charge_lines[1:]:
            lines.append("          " + cl)

    return "\n".join(lines)


def _koszul_sign(parities: list[int] | tuple[int, ...], perm: tuple[int, ...]) -> int:
    """Compute the Koszul sign for a permutation of graded (fermionic) objects.

    Counts inversions where both elements have odd parity. Each such
    inversion contributes a factor of -1.

    Args:
        parities: Parity (0 or 1) of each element in the *original* ordering.
        perm: The permutation (indices into the original ordering).

    Returns:
        +1 or -1.
    """
    sign = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j] and parities[perm[i]] and parities[perm[j]]:
                sign = -sign
    return sign


def _check_add_indices(a: tuple[TensorIndex, ...], b: tuple[TensorIndex, ...]) -> None:
    """Validate that two tensors have matching indices for addition."""
    if len(a) != len(b):
        raise ValueError(
            f"Cannot add tensors with different ranks: {len(a)} vs {len(b)}."
        )
    for i, (ai, bi) in enumerate(zip(a, b)):
        if ai.label != bi.label:
            raise ValueError(
                f"Index {i} label mismatch: {ai.label!r} vs {bi.label!r}. "
                f"Relabel tensors to match before adding."
            )
        if ai.dim != bi.dim:
            raise ValueError(
                f"Index {i} ({ai.label!r}) dimension mismatch: {ai.dim} vs {bi.dim}."
            )
        if ai.flow != bi.flow:
            raise ValueError(
                f"Index {i} ({ai.label!r}) flow mismatch: "
                f"{ai.flow.name} vs {bi.flow.name}."
            )
        if ai.symmetry != bi.symmetry:
            raise ValueError(
                f"Index {i} ({ai.label!r}) symmetry mismatch: "
                f"{ai.symmetry!r} vs {bi.symmetry!r}."
            )
        if not (
            np.array_equal(ai.sectors, bi.sectors)
            and np.array_equal(ai.multiplicities, bi.multiplicities)
        ):
            raise ValueError(
                f"Index {i} ({ai.label!r}) sector/multiplicity mismatch: "
                f"sectors {ai.sectors} vs {bi.sectors}, "
                f"mults {ai.multiplicities} vs {bi.multiplicities}."
            )


def _compute_valid_blocks(
    indices: tuple[TensorIndex, ...],
    target: int | None = None,
) -> list[BlockKey]:
    """Find all charge-sector tuples satisfying the symmetry conservation law.

    Uses incremental fused-sector propagation: builds up partial fused charges
    one leg at a time, then solves the last leg in closed form via
    ``q = flow_charge(flow, fuse(target, dual(partial)))``.  That inversion is
    valid for any abelian group because ``flow_charge`` is an involution on
    canonical representatives, so a single path now covers every rank and every
    symmetry (#734).

    Both of the branches this replaces were broken for the bit-packed charges of
    :class:`~tenax.core.symmetry.ProductSymmetry`: the ``n_values() is None``
    branch solved ``q_last = (target - prev) * flow_last`` with integer algebra,
    and the finite branch weighted charges as ``flow_last * q`` -- neither of
    which is the group operation once charges are packed bitfields rather than
    integers.

    Args:
        indices: Tuple of TensorIndex objects, one per tensor leg.
        target:  Target charge for the conservation law. If None, uses the
                 symmetry identity (standard conservation).  Setting target=Q
                 selects blocks whose net charge is Q instead of the identity;
                 used at MPS boundaries to enforce a specific quantum number.

    Returns:
        List of BlockKey tuples (one charge per leg) for valid sectors.
    """
    if not indices:
        return [()]

    sym = indices[0].symmetry
    effective_target = int(
        sym.canonicalize_charges(
            np.array(
                [int(target if target is not None else sym.identity())], dtype=np.int32
            )
        )[0]
    )

    # Sectors are canonical, sorted and unique (guaranteed by TensorIndex).
    unique_charges_per_leg = [idx.sectors.tolist() for idx in indices]
    n_legs = len(indices)

    def _flow_weighted(leg_i: int, q: int) -> np.ndarray:
        return sym.flow_charge(indices[leg_i].flow, np.array([q], dtype=np.int32))

    # partial maps running fused charge -> list of partial BlockKey prefixes.
    # Seeding with the identity (rather than the first leg) makes the rank-1
    # case fall out of the same closed form below (#733).
    partial: dict[int, list[tuple[int, ...]]] = {sym.identity(): [()]}
    for leg_i in range(n_legs - 1):
        next_partial: dict[int, list[tuple[int, ...]]] = {}
        for q in unique_charges_per_leg[leg_i]:
            eff_q = _flow_weighted(leg_i, q)
            for prev_fused, prev_combos in partial.items():
                new_fused = int(
                    sym.fuse(np.array([prev_fused], dtype=np.int32), eff_q)[0]
                )
                extended = [combo + (q,) for combo in prev_combos]
                if new_fused in next_partial:
                    next_partial[new_fused].extend(extended)
                else:
                    next_partial[new_fused] = extended
        partial = next_partial

    # Closed form for the last leg. In an abelian group the solution is unique,
    # so this replaces enumeration for finite groups too.
    last_charge_set = set(unique_charges_per_leg[n_legs - 1])
    flow_last = indices[n_legs - 1].flow
    target_arr = np.array([effective_target], dtype=np.int32)
    valid_keys: list[BlockKey] = []

    for prev_fused, prev_combos in partial.items():
        needed = sym.fuse(target_arr, sym.dual(np.array([prev_fused], dtype=np.int32)))
        q_last = int(sym.flow_charge(flow_last, needed)[0])
        if q_last in last_charge_set:
            for combo in prev_combos:
                valid_keys.append(combo + (q_last,))

    return valid_keys


def _block_slices(
    indices: tuple[TensorIndex, ...],
    key: BlockKey,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Find the positions (boolean mask) and block shape for a given BlockKey.

    For each leg, finds indices where charges[i] == key[leg]. These
    positions form the slice of the dense tensor corresponding to this block.

    Args:
        indices: Tuple of TensorIndex per leg.
        key:     BlockKey (one charge per leg).

    Returns:
        Tuple of (masks_per_leg, block_shape) where masks_per_leg[i] is a
        boolean array selecting positions along leg i, and block_shape[i] is
        the number of True entries (number of states with this charge).
    """
    masks = tuple(idx.charges == q for idx, q in zip(indices, key))
    shape = tuple(int(m.sum()) for m in masks)
    return masks, shape


# ---------- Tensor Protocol ----------


class Tensor(ABC):
    """Abstract base class for tensor objects.

    Both DenseTensor and SymmetricTensor satisfy this interface.
    Users should type-hint with Tensor for polymorphic code.
    """

    @property
    @abstractmethod
    def indices(self) -> tuple[TensorIndex, ...]: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> Any: ...

    @abstractmethod
    def todense(self) -> jax.Array: ...

    @abstractmethod
    def conj(self) -> Tensor: ...

    @abstractmethod
    def dagger(self) -> Tensor: ...

    @abstractmethod
    def bar(self) -> Tensor:
        """Element-wise conjugate with all flows flipped (no charge dual).

        Unlike :meth:`dagger`, this keeps the original charge values and
        applies no fermionic twist phases.  The result has opposite flows
        (enabling contraction) but identical charges (enabling equality-based
        block matching in the contraction engine).

        Used as the bra operation in split CTM to avoid the charge-mismatch
        problem that arises with :meth:`dagger` for nontrivial U(1) charges.
        """
        ...

    @abstractmethod
    def transpose(self, axes: tuple[int, ...]) -> Tensor: ...

    @abstractmethod
    def norm(self) -> jax.Array: ...

    def labels(self) -> tuple[Label, ...]:
        """Return the label of each leg in order."""
        return tuple(idx.label for idx in self.indices)

    @abstractmethod
    def relabel(self, old: Label, new: Label) -> Tensor:
        """Return a new tensor with one label renamed.

        Args:
            old: The label to replace.
            new: The replacement label.

        Returns:
            New tensor with updated index metadata.

        Raises:
            KeyError: If old label not found.
        """
        ...

    def relabels(self, mapping: dict[Label, Label]) -> Tensor:
        """Return a new tensor with multiple labels renamed.

        Args:
            mapping: Dict of {old_label: new_label}.

        Returns:
            New tensor with updated index metadata.
        """
        raise NotImplementedError

    def __mul__(self, scalar: float) -> Tensor:
        """Scalar multiplication: T * scalar."""
        raise NotImplementedError

    def __rmul__(self, scalar: float) -> Tensor:
        """Scalar multiplication: scalar * T."""
        return self.__mul__(scalar)

    def __add__(self, other: Tensor) -> Tensor:
        """Element-wise addition of two tensors with the same indices."""
        raise NotImplementedError

    def __sub__(self, other: Tensor) -> Tensor:
        """Element-wise subtraction."""
        return self.__add__(other.__mul__(-1.0))

    def __neg__(self) -> Tensor:
        """Negate all elements."""
        return self.__mul__(-1.0)

    def max_abs(self) -> jax.Array:
        """Maximum absolute value across all elements."""
        raise NotImplementedError


def inner(a: Tensor, b: Tensor) -> jax.Array:
    """Compute the inner product (full contraction) of two tensors.

    Contracts all shared labels to produce a scalar. Both tensors must
    have the same set of labels.

    Args:
        a: First tensor.
        b: Second tensor (will be conjugated).

    Returns:
        Scalar JAX array: sum of a_conj * b over all elements.
    """
    if isinstance(a, DenseTensor) and isinstance(b, DenseTensor):
        return jnp.sum(jnp.conj(a._data) * b._data)
    if isinstance(a, SymmetricTensor) and isinstance(b, SymmetricTensor):
        # Fast path: identical block structure → single buffer dot
        if a._block_keys == b._block_keys and a._block_shapes == b._block_shapes:
            return jnp.sum(jnp.conj(a._data) * b._data)
        # Slow path: iterate over matching keys
        a_blocks = a.blocks
        b_blocks = b.blocks
        total = jnp.zeros((), dtype=a.dtype)
        for key in a._block_keys:
            if key in b_blocks:
                total = total + jnp.sum(jnp.conj(a_blocks[key]) * b_blocks[key])
        return total
    # Mixed types: fall back to dense
    warnings.warn(
        "inner() called with mixed tensor types (DenseTensor + SymmetricTensor). "
        "Falling back to todense() which may be slow for large tensors.",
        stacklevel=2,
    )
    return jnp.sum(jnp.conj(a.todense()) * b.todense())


# ---------- DenseTensor ----------


@jax.tree_util.register_pytree_node_class
class DenseTensor(Tensor):
    """A tensor stored as a plain JAX array with index metadata.

    Used when no symmetry structure is exploited. Full compatibility with
    jax.jit, jax.vmap, jax.grad via pytree registration.

    Pytree structure:
        Leaves:     (data_array,)
        Aux data:   indices tuple (static, not traced by JAX)

    Args:
        data:    JAX array of shape matching the dimension of each index.
        indices: Tuple of TensorIndex objects, one per leg.

    Example:
        >>> data = jnp.ones((2, 3))
        >>> t = DenseTensor(data, (idx_a, idx_b))
        >>> t.norm()
        DeviceArray(2.4494898, dtype=float32)
    """

    def __init__(
        self,
        data: jax.Array,
        indices: tuple[TensorIndex, ...],
    ) -> None:
        if data.ndim != len(indices):
            raise ValueError(
                f"data has {data.ndim} dims but {len(indices)} indices given"
            )
        for i, (dim, idx) in enumerate(zip(data.shape, indices)):
            if dim != idx.dim:
                raise ValueError(
                    f"data.shape[{i}]={dim} but indices[{i}].dim={idx.dim}"
                )
        self._data = data
        self._indices = tuple(indices)

    # --- Pytree interface (JAX jit/vmap/grad compatibility) ---

    def tree_flatten(self) -> tuple[tuple[jax.Array], tuple[TensorIndex, ...]]:
        return (self._data,), self._indices

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[TensorIndex, ...],
        children: tuple[jax.Array],
    ) -> DenseTensor:
        # Bypass validation for JAX-internal dummy objects (e.g., custom_vjp
        # backward pass creates object() placeholders to probe pytree structure).
        obj = object.__new__(cls)
        obj._data = children[0]
        obj._indices = tuple(aux)
        return obj

    # --- Tensor interface ---

    @property
    def indices(self) -> tuple[TensorIndex, ...]:
        return self._indices

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> Any:
        return self._data.dtype

    def todense(self) -> jax.Array:
        return self._data

    def conj(self) -> DenseTensor:
        return DenseTensor(jnp.conj(self._data), self._indices)

    def dagger(self) -> DenseTensor:
        """Conjugate transpose: conjugate data and dual all indices."""
        new_indices = tuple(idx.dual() for idx in self._indices)
        return DenseTensor(jnp.conj(self._data), new_indices)

    def bar(self) -> DenseTensor:
        """Element-wise conjugate with flipped flows. No charge dual."""
        new_indices = tuple(idx.flip_flow() for idx in self._indices)
        return DenseTensor(jnp.conj(self._data), new_indices)

    def transpose(self, axes: tuple[int, ...]) -> DenseTensor:
        """Permute tensor legs.

        Args:
            axes: New ordering of leg indices.

        Returns:
            New DenseTensor with permuted data and reordered indices.
        """
        return DenseTensor(
            jnp.transpose(self._data, axes),
            tuple(self._indices[i] for i in axes),
        )

    def norm(self) -> jax.Array:
        """Frobenius norm."""
        return jnp.linalg.norm(self._data.ravel())

    def relabel(self, old: Label, new: Label) -> DenseTensor:
        """Return a copy with one leg label renamed.

        Args:
            old: Current label to replace.
            new: New label value.

        Returns:
            New DenseTensor with the specified label changed.

        Raises:
            KeyError: If *old* is not found among the tensor's labels.
        """
        found = False
        new_indices = []
        for idx in self._indices:
            if idx.label == old:
                new_indices.append(idx.relabel(new))
                found = True
            else:
                new_indices.append(idx)
        if not found:
            raise KeyError(
                f"Label {old!r} not found in tensor with labels {self.labels()}"
            )
        return DenseTensor(self._data, tuple(new_indices))

    def relabels(self, mapping: dict[Label, Label]) -> DenseTensor:
        """Return a copy with multiple leg labels renamed at once.

        Args:
            mapping: ``{old_label: new_label}`` pairs.  Labels not present
                in the mapping are left unchanged.

        Returns:
            New DenseTensor with the specified labels changed.
        """
        new_indices = tuple(
            idx.relabel(mapping[idx.label]) if idx.label in mapping else idx
            for idx in self._indices
        )
        return DenseTensor(self._data, new_indices)

    def __mul__(self, scalar: float) -> DenseTensor:
        return DenseTensor(self._data * scalar, self._indices)

    def __rmul__(self, scalar: float) -> DenseTensor:
        return self.__mul__(scalar)

    def __add__(self, other: Tensor) -> DenseTensor:
        if not isinstance(other, DenseTensor):
            other = DenseTensor(other.todense(), other.indices)
        _check_add_indices(self._indices, other._indices)
        return DenseTensor(self._data + other._data, self._indices)

    def __sub__(self, other: Tensor) -> DenseTensor:
        if not isinstance(other, DenseTensor):
            other = DenseTensor(other.todense(), other.indices)
        _check_add_indices(self._indices, other._indices)
        return DenseTensor(self._data - other._data, self._indices)

    def __neg__(self) -> DenseTensor:
        return DenseTensor(-self._data, self._indices)

    def max_abs(self) -> jax.Array:
        return jnp.max(jnp.abs(self._data))

    def __repr__(self) -> str:
        return _tensor_box_repr(
            "Dense",
            self._indices,
            [str(self.dtype)],
        )


# ---------- SymmetricTensor ----------


@jax.tree_util.register_pytree_node_class
class SymmetricTensor(Tensor):
    """Block-sparse tensor storing only symmetry-allowed charge sectors.

    Storage model:

    - ``_blocks``: ``dict[BlockKey, jax.Array]`` --
      Key is a tuple of one representative charge per leg.
      Value is a JAX array of shape ``(n_states_leg0, ..., n_states_legN)``
      for that charge sector.
    - ``_indices``: ``tuple[TensorIndex, ...]`` --
      Full index metadata per leg.

    Conservation law enforced on all stored blocks::

        sum_i(flow_i * charge_i) == symmetry.identity()

    Pytree structure:
        Leaves:     list of block arrays [blocks[k] for k in sorted_keys]
        Aux data:   (sorted_keys, indices) — static, not traced by JAX

    Note on JAX JIT compatibility:
        Block structure (keys) is static Python data. jax.jit recompiles
        only when the set of keys changes (i.e., when bond dimension changes
        after SVD truncation). Within a DMRG sweep at fixed bond dim, no
        recompilation occurs.

    Args:
        blocks:  Dict mapping BlockKey -> JAX array for each allowed sector.
        indices: Tuple of TensorIndex objects, one per leg.

    Example:
        >>> t = SymmetricTensor.zeros(indices=(idx_in, idx_out))
        >>> t.n_blocks
        3  # one block per unique charge value
    """

    def __init__(
        self,
        blocks: dict[BlockKey, jax.Array],
        indices: tuple[TensorIndex, ...],
    ) -> None:
        self._indices = tuple(indices)
        self._init_flat_buffer(blocks)
        self._validate()

    def _init_flat_buffer(self, blocks: dict[BlockKey, jax.Array]) -> None:
        """Pack block arrays into a single flat 1D buffer with index metadata."""
        sorted_keys = sorted(blocks.keys())
        if sorted_keys:
            flat_parts = [blocks[k].ravel() for k in sorted_keys]
            self._data: jax.Array = jnp.concatenate(flat_parts)
        else:
            self._data = jnp.zeros(0, dtype=jnp.float64)
        self._block_keys: tuple[BlockKey, ...] = tuple(sorted_keys)
        shapes = [blocks[k].shape for k in sorted_keys]
        self._block_shapes: tuple[tuple[int, ...], ...] = tuple(shapes)
        offsets = []
        offset = 0
        for s in shapes:
            offsets.append(offset)
            size = 1
            for d in s:
                size *= d
            offset += size
        self._block_offsets: tuple[int, ...] = tuple(offsets)

    @classmethod
    def _raw(
        cls,
        *,
        indices: tuple[TensorIndex, ...],
        data: jax.Array,
        block_keys: tuple[BlockKey, ...],
        block_shapes: tuple[tuple[int, ...], ...],
        block_offsets: tuple[int, ...],
    ) -> SymmetricTensor:
        """Construct without validation from pre-computed flat-buffer fields."""
        obj = object.__new__(cls)
        obj._indices = indices
        obj._data = data
        obj._block_keys = block_keys
        obj._block_shapes = block_shapes
        obj._block_offsets = block_offsets
        return obj

    @classmethod
    def _from_blocks_unchecked(
        cls,
        blocks: dict[BlockKey, jax.Array],
        indices: tuple[TensorIndex, ...],
    ) -> SymmetricTensor:
        """Construct without validation from a blocks dict."""
        obj = object.__new__(cls)
        obj._indices = tuple(indices)
        obj._init_flat_buffer(blocks)
        return obj

    def _validate(self) -> None:
        """Verify all block keys satisfy the symmetry conservation law.

        Delegates to :func:`~tenax.core.index._net_charges`, the vectorised twin
        of :func:`~tenax.core.index._net_charge`: the whole key table is fused
        leg by leg, so the symmetry sees one ``(n_blocks,)`` array per leg
        instead of one scalar per (block, leg) pair.  The per-key adapter
        remains the right call for scalar sites; here it dominated
        ``SymmetricTensor`` construction (~27% on a 489-block rank-4 U(1)
        tensor).
        """
        if not self._indices or not self._block_keys:
            return
        identity = self._indices[0].symmetry.identity()
        net = _net_charges(self._indices, self._block_keys)

        bad = np.flatnonzero(net != identity)
        if bad.size:
            key = self._block_keys[int(bad[0])]
            fused = int(net[int(bad[0])])
            raise ValueError(
                f"Block {key} violates charge conservation: "
                f"fused={fused}, expected identity={identity}"
            )

    def _get_block(self, idx: int) -> jax.Array:
        """Return the block array at position idx in sorted key order."""
        offset = self._block_offsets[idx]
        shape = self._block_shapes[idx]
        size = 1
        for d in shape:
            size *= d
        return self._data[offset : offset + size].reshape(shape)

    def stacked_blocks(self):
        """Return a StackedView: blocks grouped by shape, one array per group."""
        from tenax.core.stacked_view import build_stacked

        return build_stacked(
            self._data,
            self._block_keys,
            self._block_shapes,
            self._block_offsets,
            self._indices,
        )

    def from_stacked_blocks(self, view):
        """Rebuild a SymmetricTensor from a StackedView (canonical sorted-key layout)."""
        from tenax.core.stacked_view import scatter_stacked

        if not self._block_keys:
            total = 0
        else:
            last_shape = self._block_shapes[-1]
            last_size = int(np.prod(last_shape)) if last_shape else 1
            total = self._block_offsets[-1] + last_size
        data = scatter_stacked(
            view,
            self._block_keys,
            self._block_shapes,
            self._block_offsets,
            total,
            self._data.dtype,
        )
        return SymmetricTensor._raw(
            indices=self._indices,
            data=data,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    # --- Pytree interface ---

    def tree_flatten(
        self,
    ) -> tuple[
        list[jax.Array],
        tuple[
            tuple[BlockKey, ...],
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[TensorIndex, ...],
        ],
    ]:
        # Single leaf: the flat data buffer
        return [self._data], (
            self._block_keys,
            self._block_shapes,
            self._block_offsets,
            self._indices,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[
            tuple[BlockKey, ...],
            tuple[tuple[int, ...], ...],
            tuple[int, ...],
            tuple[TensorIndex, ...],
        ],
        children: list[jax.Array],
    ) -> SymmetricTensor:
        block_keys, block_shapes, block_offsets, indices = aux
        return cls._raw(
            indices=indices,
            data=children[0],
            block_keys=block_keys,
            block_shapes=block_shapes,
            block_offsets=block_offsets,
        )

    # --- Factory methods ---

    @classmethod
    def zeros(
        cls,
        indices: tuple[TensorIndex, ...],
        dtype: Any = jnp.float64,
        target: int | None = None,
    ) -> SymmetricTensor:
        """Create a zero tensor with all valid charge sectors initialized to zero.

        Args:
            indices: Tuple of TensorIndex objects.
            dtype:   Data type for block arrays.
            target:  Target charge for block selection (construction-time only).
                     If None, uses the symmetry identity (standard conservation).

        Returns:
            SymmetricTensor with all valid blocks set to zero.
        """
        valid_keys = _compute_valid_blocks(indices, target=target)
        blocks: dict[BlockKey, jax.Array] = {}
        for key in valid_keys:
            _, shape = _block_slices(indices, key)
            if all(s > 0 for s in shape):
                blocks[key] = jnp.zeros(shape, dtype=dtype)
        if target is not None and target != 0:
            # Non-identity target: blocks satisfy sum(flow*q) == target,
            # which would fail standard validation. Bypass it.
            return cls._from_blocks_unchecked(dict(blocks), indices)
        return cls(blocks, indices)

    @classmethod
    def random_normal(
        cls,
        indices: tuple[TensorIndex, ...],
        key: jax.Array,
        dtype: Any = jnp.float64,
        stddev: float = 1.0,
        target: int | None = None,
    ) -> SymmetricTensor:
        """Create a random tensor with blocks drawn from N(0, stddev).

        Splits the JAX random key sequentially over blocks.

        Args:
            indices: Tuple of TensorIndex objects.
            key:     JAX random key.
            dtype:   Data type for block arrays.
            stddev:  Standard deviation of the normal distribution.
            target:  Target charge for block selection (construction-time only).
                     If None, uses the symmetry identity (standard conservation).

        Returns:
            SymmetricTensor with random entries in all valid blocks.
        """
        valid_keys = _compute_valid_blocks(indices, target=target)
        blocks: dict[BlockKey, jax.Array] = {}
        for i, block_key in enumerate(sorted(valid_keys)):
            _, shape = _block_slices(indices, block_key)
            if all(s > 0 for s in shape):
                subkey = jax.random.fold_in(key, i)
                data = jax.random.normal(subkey, shape, dtype=dtype) * stddev
                blocks[block_key] = data
        if target is not None and target != 0:
            # Non-identity target: blocks satisfy sum(flow*q) == target,
            # which would fail standard validation. Bypass it.
            return cls._from_blocks_unchecked(dict(blocks), indices)
        return cls(blocks, indices)

    @classmethod
    def random_normal_np(
        cls,
        indices: tuple[TensorIndex, ...],
        rng: np.random.RandomState,
        dtype: Any = jnp.float64,
        stddev: float = 1.0,
        target: int | None = None,
    ) -> SymmetricTensor:
        """Create a random tensor using numpy (no JAX compilation overhead).

        Useful in tight loops (e.g. iDMRG) where JAX random_normal triggers
        costly recompilation on each call.

        Args:
            indices: Tuple of TensorIndex objects.
            rng:     numpy RandomState for reproducibility.
            dtype:   Data type for block arrays.
            stddev:  Standard deviation of the normal distribution.
            target:  Target charge for block selection.

        Returns:
            SymmetricTensor with random entries in all valid blocks.
        """
        valid_keys = _compute_valid_blocks(indices, target=target)
        blocks: dict[BlockKey, jax.Array] = {}
        for block_key in sorted(valid_keys):
            _, shape = _block_slices(indices, block_key)
            if all(s > 0 for s in shape):
                data = jnp.array(
                    rng.randn(*shape).astype(np.float64) * stddev, dtype=dtype
                )
                blocks[block_key] = data
        if target is not None and target != 0:
            return cls._from_blocks_unchecked(dict(blocks), indices)
        return cls(blocks, indices)

    @classmethod
    def from_dense(
        cls,
        data: jax.Array,
        indices: tuple[TensorIndex, ...],
        tol: float = 1e-12,
    ) -> SymmetricTensor:
        """Extract block-sparse structure from a dense JAX array.

        Elements outside valid charge sectors must be zero (within tol)
        or a ValueError is raised.

        Args:
            data:    Dense JAX array of shape matching index dimensions.
            indices: Tuple of TensorIndex objects.
            tol:     Tolerance for checking zero elements outside blocks.

        Returns:
            SymmetricTensor with blocks extracted from data.

        Raises:
            ValueError: If data has non-zero elements outside valid sectors.
        """
        if data.shape != tuple(idx.dim for idx in indices):
            raise ValueError(
                f"data.shape {data.shape} does not match index dims "
                f"{tuple(idx.dim for idx in indices)}"
            )

        valid_keys = _compute_valid_blocks(indices)

        # Extract blocks using JAX-compatible indexing so that from_dense()
        # works under JAX tracing (e.g. inside jax.grad / jax.vjp).
        # Masks and index arrays are static (derived from charges), only
        # the data values may be JAX tracers.
        full_mask = np.zeros(data.shape, dtype=bool)
        blocks: dict[BlockKey, jax.Array] = {}

        for key in sorted(valid_keys):
            masks, shape = _block_slices(indices, key)
            if not all(s > 0 for s in shape):
                continue
            idx_arrays = [np.where(m)[0] for m in masks]
            grid = np.ix_(*idx_arrays)
            blocks[key] = data[grid]

            # Mark these positions as valid (static mask for validation)
            full_mask[grid] = True

        # Validation: check for non-zero elements outside valid blocks.
        # Skip when tol is infinite (used by CTM wrapping) or when data
        # is a JAX tracer (validation requires concrete values).
        if tol != float("inf") and not isinstance(data, jax.core.Tracer):
            data_np = np.asarray(data)
            outside = data_np[~full_mask]
            if np.any(np.abs(outside) > tol):
                raise ValueError(
                    f"data has {np.sum(np.abs(outside) > tol)} non-zero elements "
                    f"outside symmetry-allowed sectors (max abs value: "
                    f"{np.max(np.abs(outside)):.3e})"
                )

        return cls(blocks, indices)

    # --- Tensor interface ---

    @property
    def indices(self) -> tuple[TensorIndex, ...]:
        return self._indices

    @property
    def ndim(self) -> int:
        return len(self._indices)

    @property
    def dtype(self) -> Any:
        return self._data.dtype if self._data.size > 0 else jnp.float64

    @property
    def n_blocks(self) -> int:
        """Number of non-empty charge sectors."""
        return len(self._block_keys)

    @property
    def blocks(self) -> dict[BlockKey, jax.Array]:
        """Block dict from flat buffer, cached for repeated access.

        The cache is valid because ``_data`` (a JAX array) is immutable.
        Eliminates redundant slice+reshape calls when ``blocks`` is
        accessed multiple times (e.g. during ``_blockwise_contract``).
        """
        try:
            return self._blocks_cache
        except AttributeError:
            cache = {
                self._block_keys[i]: self._get_block(i)
                for i in range(len(self._block_keys))
            }
            object.__setattr__(self, "_blocks_cache", cache)
            return cache

    def todense(self) -> jax.Array:
        """Materialize the full dense tensor (for testing/debugging only).

        Warning: Creates an array of full size; avoid for large tensors.

        Returns:
            Dense JAX array of shape tuple(idx.dim for idx in indices).
        """
        shape = tuple(idx.dim for idx in self._indices)
        if self.n_blocks == 0:
            return jnp.zeros(shape, dtype=self.dtype)

        # Start from zeros and scatter blocks using JAX-compatible ops
        # so that todense() works under JAX tracing (e.g. inside jax.grad).
        result = jnp.zeros(shape, dtype=self.dtype)
        for key, block in self.blocks.items():
            masks, _ = _block_slices(self._indices, key)
            idx_arrays = [np.where(m)[0] for m in masks]
            grid = np.ix_(*idx_arrays)
            result = result.at[grid].set(block)

        return result

    def conj(self) -> SymmetricTensor:
        """Return conjugate tensor (single flat buffer op)."""
        return SymmetricTensor._raw(
            indices=self._indices,
            data=jnp.conj(self._data),
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    def dagger(self) -> SymmetricTensor:
        """Conjugate transpose with fermionic twist phases.

        For each block, applies complex conjugation and reverses all leg
        flows (via dual).  For fermionic symmetries, multiplies by the
        super-algebra sign ``(-1)^{sum_{i<j} p_i p_j}`` where ``p_i``
        is the parity of the i-th charge.  For bosonic symmetries this
        is equivalent to ``conj()`` with dualled indices (sign is always +1).

        Returns:
            New SymmetricTensor with conjugated data, dual indices, and
            twist phase corrections.
        """
        sym = self._indices[0].symmetry if self._indices else None
        new_indices = tuple(idx.dual() for idx in self._indices)
        new_blocks: dict[BlockKey, jax.Array] = {}
        for key, block in self.blocks.items():
            new_key = (
                tuple(int(sym.dual(np.array([q]))[0]) for q in key) if sym else key
            )
            val = jnp.conj(block)
            if sym is not None and sym.is_fermionic:
                # Super-algebra dagger: (-1)^{sum_{i<j} p_i * p_j}
                # where p_i is the parity of the i-th charge.
                parities = [int(sym.parity(np.array([q]))[0]) for q in key]
                n_sign = 0
                for i in range(len(parities)):
                    for j in range(i + 1, len(parities)):
                        n_sign += parities[i] * parities[j]
                if n_sign % 2 == 1:
                    val = -val
            new_blocks[new_key] = val
        return SymmetricTensor._from_blocks_unchecked(new_blocks, new_indices)

    def bar(self) -> SymmetricTensor:
        """Element-wise conjugate with flipped flows. No charge dual, no twist."""
        new_indices = tuple(idx.flip_flow() for idx in self._indices)
        return SymmetricTensor._raw(
            indices=new_indices,
            data=jnp.conj(self._data),
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    def transpose(self, axes: tuple[int, ...]) -> SymmetricTensor:
        """Permute tensor legs.

        For fermionic symmetries, each block acquires a Koszul sign
        determined by the charges' parities and the permutation.

        Args:
            axes: New ordering of leg indices.

        Returns:
            New SymmetricTensor with permuted blocks and reordered indices.
        """
        new_indices = tuple(self._indices[i] for i in axes)
        sym = self._indices[0].symmetry if self._indices else None
        is_ferm = sym is not None and sym.is_fermionic

        new_blocks: dict[BlockKey, jax.Array] = {}
        for key, block in self.blocks.items():
            new_key = tuple(key[i] for i in axes)
            transposed = jnp.transpose(block, axes)
            if is_ferm:
                parities = tuple(int(sym.parity(np.array([q]))[0]) for q in key)
                sign = _koszul_sign(parities, axes)
                if sign < 0:
                    transposed = -transposed
            new_blocks[new_key] = transposed
        return SymmetricTensor._from_blocks_unchecked(new_blocks, new_indices)

    def norm(self) -> jax.Array:
        """Frobenius norm across all blocks."""
        if self.n_blocks == 0:
            return jnp.zeros((), dtype=self.dtype)
        return jnp.sqrt(jnp.sum(jnp.abs(self._data) ** 2))

    def block_shapes(self) -> dict[BlockKey, tuple[int, ...]]:
        """Return the shape of each stored block."""
        return {
            self._block_keys[i]: self._block_shapes[i]
            for i in range(len(self._block_keys))
        }

    def relabel(self, old: Label, new: Label) -> SymmetricTensor:
        """Return a copy with one leg label renamed.

        Args:
            old: Current label to replace.
            new: New label value.

        Returns:
            New SymmetricTensor sharing the same block data.

        Raises:
            KeyError: If *old* is not found among the tensor's labels.
        """
        found = False
        new_indices = []
        for idx in self._indices:
            if idx.label == old:
                new_indices.append(idx.relabel(new))
                found = True
            else:
                new_indices.append(idx)
        if not found:
            raise KeyError(
                f"Label {old!r} not found in tensor with labels {self.labels()}"
            )
        return SymmetricTensor._raw(
            indices=tuple(new_indices),
            data=self._data,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    def relabels(self, mapping: dict[Label, Label]) -> SymmetricTensor:
        """Return a copy with multiple leg labels renamed at once.

        Args:
            mapping: ``{old_label: new_label}`` pairs.  Labels not present
                in the mapping are left unchanged.

        Returns:
            New SymmetricTensor sharing the same block data.
        """
        new_indices = tuple(
            idx.relabel(mapping[idx.label]) if idx.label in mapping else idx
            for idx in self._indices
        )
        return SymmetricTensor._raw(
            indices=new_indices,
            data=self._data,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    def __mul__(self, scalar: float) -> SymmetricTensor:
        return SymmetricTensor._raw(
            indices=self._indices,
            data=self._data * scalar,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
        )

    def __rmul__(self, scalar: float) -> SymmetricTensor:
        return self.__mul__(scalar)

    def __add__(self, other: Tensor) -> SymmetricTensor:
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"Cannot add SymmetricTensor and {type(other).__name__}; "
                f"convert to matching type first."
            )
        _check_add_indices(self._indices, other._indices)
        # Fast path: identical block structure → single buffer add
        if (
            self._block_keys == other._block_keys
            and self._block_shapes == other._block_shapes
        ):
            return SymmetricTensor._raw(
                indices=self._indices,
                data=self._data + other._data,
                block_keys=self._block_keys,
                block_shapes=self._block_shapes,
                block_offsets=self._block_offsets,
            )
        # Slow path: union of block keys
        self_blocks = self.blocks
        other_blocks = other.blocks
        new_blocks: dict[BlockKey, jax.Array] = {}
        all_keys = set(self._block_keys) | set(other._block_keys)
        for key in all_keys:
            if key in self_blocks and key in other_blocks:
                new_blocks[key] = self_blocks[key] + other_blocks[key]
            elif key in self_blocks:
                new_blocks[key] = self_blocks[key]
            else:
                new_blocks[key] = other_blocks[key]
        return SymmetricTensor._from_blocks_unchecked(new_blocks, self._indices)

    def __sub__(self, other: Tensor) -> SymmetricTensor:
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from SymmetricTensor."
            )
        return self.__add__(other.__mul__(-1.0))

    def __neg__(self) -> SymmetricTensor:
        return self.__mul__(-1.0)

    def max_abs(self) -> jax.Array:
        if self.n_blocks == 0:
            return jnp.zeros((), dtype=self.dtype)
        return jnp.max(jnp.abs(self._data))

    def __repr__(self) -> str:
        total_elements = int(self._data.size)
        sym_name = self._indices[0].symmetry.__class__.__name__ if self._indices else ""
        # Short symmetry label: U1Symmetry -> U(1), ZnSymmetry(3) -> Z(3)
        sym = self._indices[0].symmetry if self._indices else None
        if sym_name == "U1Symmetry":
            sym_label = "U(1)"
        elif sym_name == "ZnSymmetry":
            sym_label = f"Z({sym.n})"
        elif sym_name == "ProductSymmetry":
            sym_label = "Product"
        else:
            sym_label = sym_name

        # Build charge summary lines
        charge_parts = [f"{idx.label}{_charge_summary(idx)}" for idx in self._indices]
        # Group into lines of roughly 50 chars
        charge_lines = []
        current = ""
        for part in charge_parts:
            if current and len(current) + len(part) + 1 > 50:
                charge_lines.append(current)
                current = part
            else:
                current = current + " " + part if current else part
        if current:
            charge_lines.append(current)

        return _tensor_box_repr(
            "Symmetric",
            self._indices,
            [
                f"{sym_label} {self.dtype}",
                f"{self.n_blocks}blk nnz={total_elements}",
            ],
            charge_lines,
        )
