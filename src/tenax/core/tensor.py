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

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.index import Label, TensorIndex

# Block key: tuple of one charge value per leg identifying a charge sector
BlockKey = tuple[int, ...]


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


def _compute_valid_blocks(
    indices: tuple[TensorIndex, ...],
    target: int | None = None,
) -> list[BlockKey]:
    """Find all charge-sector tuples satisfying the symmetry conservation law.

    Uses incremental fused-sector propagation: instead of testing all N-leg
    combinations, builds up partial charge sums one leg at a time, pruning
    incompatible branches early.  For finite groups (Zn), every intermediate
    sum is kept.  For infinite groups (U1), the last leg is constrained to
    exactly cancel the running sum, avoiding enumeration of its charges.

    Args:
        indices: Tuple of TensorIndex objects, one per tensor leg.
        target:  Target charge for the conservation law. If None, uses the
                 symmetry identity (standard conservation). For U(1), setting
                 target=Q selects blocks where sum(flow_i * charge_i) == Q
                 instead of 0. Used at MPS boundaries to enforce a specific
                 quantum number sector.

    Returns:
        List of BlockKey tuples (one charge per leg) for valid sectors.
    """
    if not indices:
        return [()]

    sym = indices[0].symmetry
    effective_target = target if target is not None else sym.identity()

    # Collect unique charge values per leg (sorted for determinism)
    unique_charges_per_leg = [sorted(set(idx.charges.tolist())) for idx in indices]

    n_legs = len(indices)

    # For infinite symmetries (U1), the last leg's charge is fully determined
    # by the running sum of the previous legs. We can skip enumeration.
    is_infinite = sym.n_values() is None

    if n_legs == 1:
        # Single leg: only target charge is valid
        return [
            (q,)
            for q in unique_charges_per_leg[0]
            if int(indices[0].flow) * q == effective_target
        ]

    # Incremental propagation: partial_combos maps
    #   running_fused_charge -> list of partial BlockKey tuples
    # Start with the first leg
    flow0 = int(indices[0].flow)
    partial: dict[int, list[tuple[int, ...]]] = {}
    for q in unique_charges_per_leg[0]:
        fused = flow0 * q
        partial.setdefault(fused, []).append((q,))

    # Process legs 1 .. n_legs-2 (all except the last)
    last_leg_idx = n_legs - 1
    for leg_i in range(1, last_leg_idx):
        flow_i = int(indices[leg_i].flow)
        next_partial: dict[int, list[tuple[int, ...]]] = {}
        for q in unique_charges_per_leg[leg_i]:
            effective_q = flow_i * q
            # For each existing partial sum, fuse with this leg's charge
            for prev_fused, prev_combos in partial.items():
                new_fused = int(
                    sym.fuse(
                        np.array([prev_fused], dtype=np.int32),
                        np.array([effective_q], dtype=np.int32),
                    )[0]
                )
                extended = [combo + (q,) for combo in prev_combos]
                if new_fused in next_partial:
                    next_partial[new_fused].extend(extended)
                else:
                    next_partial[new_fused] = extended
        partial = next_partial

    # Process the last leg: only keep combos where total fuses to identity
    flow_last = int(indices[last_leg_idx].flow)
    valid_keys: list[BlockKey] = []

    if is_infinite:
        # For U1: the required effective charge for the last leg is
        # determined by: fuse(prev_fused, flow_last * q_last) == effective_target
        # For U1: prev_fused + flow_last * q_last == effective_target
        # => q_last = (effective_target - prev_fused) * flow_last
        last_charge_set = set(unique_charges_per_leg[last_leg_idx])
        for prev_fused, prev_combos in partial.items():
            if flow_last == 0:
                continue
            # needed = effective_target - prev_fused (what flow_last * q_last must equal)
            needed = effective_target - prev_fused
            # q_last = needed / flow_last; for flow IN(+1)/OUT(-1): q_last = needed * flow_last
            q_last = needed * flow_last
            if q_last in last_charge_set:
                for combo in prev_combos:
                    valid_keys.append(combo + (q_last,))
    else:
        # For finite groups: enumerate last leg charges
        for q in unique_charges_per_leg[last_leg_idx]:
            effective_q = flow_last * q
            for prev_fused, prev_combos in partial.items():
                total = int(
                    sym.fuse(
                        np.array([prev_fused], dtype=np.int32),
                        np.array([effective_q], dtype=np.int32),
                    )[0]
                )
                if total == effective_target:
                    for combo in prev_combos:
                        valid_keys.append(combo + (q,))

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


class Tensor:
    """Structural base class (duck-typed protocol) for tensor objects.

    Both DenseTensor and SymmetricTensor satisfy this interface.
    Users should type-hint with Tensor for polymorphic code.
    """

    @property
    def indices(self) -> tuple[TensorIndex, ...]:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def dtype(self) -> Any:
        raise NotImplementedError

    def todense(self) -> jax.Array:
        raise NotImplementedError

    def conj(self) -> Tensor:
        raise NotImplementedError

    def dagger(self) -> Tensor:
        raise NotImplementedError

    def bar(self) -> Tensor:
        """Element-wise conjugate with all flows flipped (no charge dual).

        Unlike :meth:`dagger`, this keeps the original charge values and
        applies no fermionic twist phases.  The result has opposite flows
        (enabling contraction) but identical charges (enabling equality-based
        block matching in the contraction engine).

        Used as the bra operation in split CTM to avoid the charge-mismatch
        problem that arises with :meth:`dagger` for nontrivial U(1) charges.
        """
        raise NotImplementedError

    def transpose(self, axes: tuple[int, ...]) -> Tensor:
        raise NotImplementedError

    def norm(self) -> jax.Array:
        raise NotImplementedError

    def labels(self) -> tuple[Label, ...]:
        """Return the label of each leg in order."""
        return tuple(idx.label for idx in self.indices)

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
        raise NotImplementedError

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
        # Both must have matching block keys
        total = jnp.zeros((), dtype=a.dtype)
        for key in a._blocks:
            if key in b._blocks:
                total = total + jnp.sum(jnp.conj(a._blocks[key]) * b._blocks[key])
        return total
    # Mixed types: fall back to dense
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
        return DenseTensor(self._data + other._data, self._indices)

    def __sub__(self, other: Tensor) -> DenseTensor:
        if not isinstance(other, DenseTensor):
            other = DenseTensor(other.todense(), other.indices)
        return DenseTensor(self._data - other._data, self._indices)

    def __neg__(self) -> DenseTensor:
        return DenseTensor(-self._data, self._indices)

    def max_abs(self) -> jax.Array:
        return jnp.max(jnp.abs(self._data))

    def __repr__(self) -> str:
        return (
            f"DenseTensor(shape={self._data.shape}, dtype={self.dtype}, "
            f"labels={self.labels()})"
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
        self._blocks: dict[BlockKey, jax.Array] = dict(blocks)
        self._validate()

    def _validate(self) -> None:
        """Verify all block keys satisfy the symmetry conservation law."""
        if not self._indices:
            return
        sym = self._indices[0].symmetry
        identity = sym.identity()

        for key in self._blocks:
            effective = [
                np.array([int(idx.flow) * int(charge)], dtype=np.int32)
                for idx, charge in zip(self._indices, key)
            ]
            fused_val = int(sym.fuse_many(effective)[0])
            if fused_val != identity:
                raise ValueError(
                    f"Block {key} violates charge conservation: "
                    f"fused={fused_val}, expected identity={identity}"
                )

    # --- Pytree interface ---

    def tree_flatten(
        self,
    ) -> tuple[list[jax.Array], tuple[list[BlockKey], tuple[TensorIndex, ...]]]:
        # Sort keys for deterministic ordering
        keys = sorted(self._blocks.keys())
        arrays = [self._blocks[k] for k in keys]
        return arrays, (keys, self._indices)

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[list[BlockKey], tuple[TensorIndex, ...]],
        children: list[jax.Array],
    ) -> SymmetricTensor:
        keys, indices = aux
        blocks = dict(zip(keys, children))
        # Skip validation (keys are already validated at construction time)
        obj = object.__new__(cls)
        obj._indices = indices
        obj._blocks = blocks
        return obj

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
            obj = object.__new__(cls)
            obj._indices = tuple(indices)
            obj._blocks = dict(blocks)
            return obj
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
            obj = object.__new__(cls)
            obj._indices = tuple(indices)
            obj._blocks = dict(blocks)
            return obj
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
        if not self._blocks:
            return jnp.float64
        return next(iter(self._blocks.values())).dtype

    @property
    def n_blocks(self) -> int:
        """Number of non-empty charge sectors."""
        return len(self._blocks)

    @property
    def blocks(self) -> dict[BlockKey, jax.Array]:
        """Read-only view of the block dict."""
        return self._blocks

    def todense(self) -> jax.Array:
        """Materialize the full dense tensor (for testing/debugging only).

        Warning: Creates an array of full size; avoid for large tensors.

        Returns:
            Dense JAX array of shape tuple(idx.dim for idx in indices).
        """
        shape = tuple(idx.dim for idx in self._indices)
        if not self._blocks:
            return jnp.zeros(shape, dtype=self.dtype)

        # Start from zeros and scatter blocks using JAX-compatible ops
        # so that todense() works under JAX tracing (e.g. inside jax.grad).
        result = jnp.zeros(shape, dtype=self.dtype)
        for key, block in self._blocks.items():
            masks, _ = _block_slices(self._indices, key)
            idx_arrays = [np.where(m)[0] for m in masks]
            grid = np.ix_(*idx_arrays)
            result = result.at[grid].set(block)

        return result

    def conj(self) -> SymmetricTensor:
        """Return conjugate tensor (conjugate all block arrays)."""
        new_blocks = {k: jnp.conj(v) for k, v in self._blocks.items()}
        obj = object.__new__(SymmetricTensor)
        obj._indices = self._indices
        obj._blocks = new_blocks
        return obj

    def dagger(self) -> SymmetricTensor:
        """Conjugate transpose with fermionic twist phases.

        For each block, applies complex conjugation, reverses all leg flows
        (via dual), and multiplies by the product of twist phases for all
        charges in the block key. For bosonic symmetries this is equivalent
        to ``conj()`` with dualled indices.

        Returns:
            New SymmetricTensor with conjugated data, dual indices, and
            twist phase corrections.
        """
        sym = self._indices[0].symmetry if self._indices else None
        new_indices = tuple(idx.dual() for idx in self._indices)
        new_blocks: dict[BlockKey, jax.Array] = {}
        for key, block in self._blocks.items():
            new_key = (
                tuple(int(sym.dual(np.array([q]))[0]) for q in key) if sym else key
            )
            val = jnp.conj(block)
            if sym is not None and sym.is_fermionic:
                twist = 1.0
                for q in key:
                    twist *= sym.twist_phase(q)
                if twist != 1.0:
                    val = val * twist
            new_blocks[new_key] = val
        obj = object.__new__(SymmetricTensor)
        obj._indices = new_indices
        obj._blocks = new_blocks
        return obj

    def bar(self) -> SymmetricTensor:
        """Element-wise conjugate with flipped flows. No charge dual, no twist."""
        new_indices = tuple(idx.flip_flow() for idx in self._indices)
        new_blocks = {k: jnp.conj(v) for k, v in self._blocks.items()}
        obj = object.__new__(SymmetricTensor)
        obj._indices = new_indices
        obj._blocks = new_blocks
        return obj

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
        for key, block in self._blocks.items():
            new_key = tuple(key[i] for i in axes)
            transposed = jnp.transpose(block, axes)
            if is_ferm:
                parities = tuple(int(sym.parity(np.array([q]))[0]) for q in key)
                sign = _koszul_sign(parities, axes)
                if sign < 0:
                    transposed = -transposed
            new_blocks[new_key] = transposed
        obj = object.__new__(SymmetricTensor)
        obj._indices = new_indices
        obj._blocks = new_blocks
        return obj

    def norm(self) -> jax.Array:
        """Frobenius norm across all blocks."""
        if not self._blocks:
            return jnp.zeros((), dtype=self.dtype)
        sq_norms = [jnp.sum(jnp.abs(v) ** 2) for v in self._blocks.values()]
        return jnp.sqrt(sum(sq_norms))

    def block_shapes(self) -> dict[BlockKey, tuple[int, ...]]:
        """Return the shape of each stored block."""
        return {k: v.shape for k, v in self._blocks.items()}

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
        obj = object.__new__(SymmetricTensor)
        obj._indices = tuple(new_indices)
        obj._blocks = self._blocks
        return obj

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
        obj = object.__new__(SymmetricTensor)
        obj._indices = new_indices
        obj._blocks = self._blocks
        return obj

    def __mul__(self, scalar: float) -> SymmetricTensor:
        new_blocks = {k: v * scalar for k, v in self._blocks.items()}
        obj = object.__new__(SymmetricTensor)
        obj._indices = self._indices
        obj._blocks = new_blocks
        return obj

    def __rmul__(self, scalar: float) -> SymmetricTensor:
        return self.__mul__(scalar)

    def __add__(self, other: Tensor) -> SymmetricTensor:
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"Cannot add SymmetricTensor and {type(other).__name__}; "
                f"convert to matching type first."
            )
        # Union of block keys: add where both exist, copy where only one exists
        new_blocks: dict[BlockKey, jax.Array] = {}
        all_keys = set(self._blocks.keys()) | set(other._blocks.keys())
        for key in all_keys:
            if key in self._blocks and key in other._blocks:
                new_blocks[key] = self._blocks[key] + other._blocks[key]
            elif key in self._blocks:
                new_blocks[key] = self._blocks[key]
            else:
                new_blocks[key] = other._blocks[key]
        obj = object.__new__(SymmetricTensor)
        obj._indices = self._indices
        obj._blocks = new_blocks
        return obj

    def __sub__(self, other: Tensor) -> SymmetricTensor:
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from SymmetricTensor."
            )
        return self.__add__(other.__mul__(-1.0))

    def __neg__(self) -> SymmetricTensor:
        return self.__mul__(-1.0)

    def max_abs(self) -> jax.Array:
        if not self._blocks:
            return jnp.zeros((), dtype=self.dtype)
        return jnp.max(jnp.stack([jnp.max(jnp.abs(v)) for v in self._blocks.values()]))

    def __repr__(self) -> str:
        total_elements = sum(v.size for v in self._blocks.values())
        return (
            f"SymmetricTensor(ndim={self.ndim}, n_blocks={self.n_blocks}, "
            f"nnz={total_elements}, dtype={self.dtype}, labels={self.labels()})"
        )
