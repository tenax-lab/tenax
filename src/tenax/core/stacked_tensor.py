"""Lazy/persisting stacked SymmetricTensor (#566 P1d slice 1).

``StackedSymmetricTensor`` carries the per-shape-group stacked arrays (a
:class:`~tenax.core.stacked_view.StackedView`) as its PRIMARY representation and
DEFERS materialization of the flat ``_data`` buffer until first access. This lets
a chain of consecutive stacked contractions inside one jit trace skip the
per-call gather (``stacked_blocks()`` reading ``_data``) and scatter
(``_from_blocks_unchecked``/``scatter_stacked`` writing ``_data``): the stacked
arrays persist on the Python tensor object between contractions.

Correctness invariants (do NOT regress):

* It is a *subclass* of :class:`SymmetricTensor`, so every consumer (energy fn,
  fuse, svd, bar, relabels, pytree, AD) treats it identically. Any base method
  that reads ``self._data`` transparently triggers lazy materialization (via
  :func:`~tenax.core.stacked_view.scatter_stacked`) — correctness preserved, the
  persist win is simply lost across that op.
* ``tree_flatten`` emits the MATERIALIZED ``_data`` leaf, so jit/grad boundaries
  are byte-identical to a plain ``SymmetricTensor``. The stacked cache is a
  within-trace optimization only; it does NOT survive a pytree round-trip
  (``tree_unflatten`` rebuilds a plain ``SymmetricTensor`` from ``_data``).
* This class is ONLY produced when ``TENAX_STACK_BLOCKSPARSE`` is truthy (the
  contractor gates its construction). With the flag off, nothing here is reached
  and behavior is byte-identical to today.

Label-only / data-elementwise ops (``relabel``, ``relabels``, ``bar``,
``__mul__``) are overridden to PRESERVE the stacked cache without materializing
``_data`` — these are exactly the ops the CTM sweep interleaves between
contractions (e.g. relabelling the output of one absorption to feed the next).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.core.stacked_view import StackedView, scatter_stacked
from tenax.core.tensor import BlockKey, SymmetricTensor


@jax.tree_util.register_pytree_node_class
class StackedSymmetricTensor(SymmetricTensor):
    """A SymmetricTensor whose primary representation is a cached StackedView.

    The flat ``_data`` buffer is materialized lazily on first access (and then
    cached). Constructed only via :meth:`from_stacked`.
    """

    __slots__ = ()  # keep instance dict from base; sentinel lives in __dict__

    # ------------------------------------------------------------------ build
    @classmethod
    def from_stacked(
        cls,
        *,
        view: StackedView,
        indices: tuple,
        block_keys: tuple[BlockKey, ...],
        block_shapes: tuple[tuple[int, ...], ...],
        block_offsets: tuple[int, ...],
        total_size: int,
        dtype,
    ) -> StackedSymmetricTensor:
        """Build from per-shape-group stacked arrays with DEFERRED ``_data``.

        ``view`` must already be in canonical sorted-key layout matching
        ``block_keys`` (the contractor guarantees this).
        """
        obj = object.__new__(cls)
        obj._indices = tuple(indices)
        obj._block_keys = tuple(block_keys)
        obj._block_shapes = tuple(block_shapes)
        obj._block_offsets = tuple(block_offsets)
        obj._stacked_view = view
        obj._total_size = int(total_size)
        obj._dtype = dtype
        obj._data_cache = None  # sentinel: not yet materialized
        return obj

    # ----------------------------------------------------------- lazy _data
    def _materialize(self) -> jax.Array:
        """Scatter the cached StackedView into the flat ``_data`` buffer once."""
        cache = self.__dict__.get("_data_cache")
        if cache is None:
            cache = scatter_stacked(
                self._stacked_view,
                self._block_keys,
                self._block_shapes,
                self._block_offsets,
                self._total_size,
                self._dtype,
            )
            self.__dict__["_data_cache"] = cache
        return cache

    @property
    def _data(self) -> jax.Array:  # type: ignore[override]
        return self._materialize()

    @_data.setter
    def _data(self, value) -> None:
        # Base __init__/_raw paths never run on this subclass, but keep a setter
        # so any defensive ``obj._data = ...`` stays well-defined.
        self.__dict__["_data_cache"] = value

    # --------------------------------------------------- stacked fast path
    def stacked_blocks(self) -> StackedView:  # type: ignore[override]
        """Return the CACHED StackedView with NO gather from ``_data``."""
        return self._stacked_view

    # --------------------------------------- cache-preserving metadata ops
    def _rebuild_view_with_indices(self, new_indices: tuple) -> StackedView:
        return StackedView(groups=self._stacked_view.groups, indices=tuple(new_indices))

    def _with_indices(self, new_indices: tuple) -> StackedSymmetricTensor:
        """Return a new StackedSymmetricTensor sharing arrays, new indices.

        Used by label-only ops (relabel/relabels): the stacked arrays and the
        materialized ``_data`` (if any) are unchanged, only leg metadata changes.
        """
        obj = StackedSymmetricTensor.from_stacked(
            view=self._rebuild_view_with_indices(new_indices),
            indices=new_indices,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
            total_size=self._total_size,
            dtype=self._dtype,
        )
        # Carry over an already-materialized buffer if present (data is identical
        # under a pure relabel).
        cached = self.__dict__.get("_data_cache")
        if cached is not None:
            obj.__dict__["_data_cache"] = cached
        return obj

    def relabel(self, old, new):  # type: ignore[override]
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
        return self._with_indices(tuple(new_indices))

    def relabels(self, mapping):  # type: ignore[override]
        new_indices = tuple(
            idx.relabel(mapping[idx.label]) if idx.label in mapping else idx
            for idx in self._indices
        )
        return self._with_indices(new_indices)

    def __mul__(self, scalar):  # type: ignore[override]
        new_groups = {
            shape: type(grp)(keys=grp.keys, array=grp.array * scalar)
            for shape, grp in self._stacked_view.groups.items()
        }
        obj = StackedSymmetricTensor.from_stacked(
            view=StackedView(groups=new_groups, indices=self._indices),
            indices=self._indices,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
            total_size=self._total_size,
            dtype=self._dtype,
        )
        return obj

    def bar(self):  # type: ignore[override]
        new_indices = tuple(idx.flip_flow() for idx in self._indices)
        new_groups = {
            shape: type(grp)(keys=grp.keys, array=jnp.conj(grp.array))
            for shape, grp in self._stacked_view.groups.items()
        }
        return StackedSymmetricTensor.from_stacked(
            view=StackedView(groups=new_groups, indices=new_indices),
            indices=new_indices,
            block_keys=self._block_keys,
            block_shapes=self._block_shapes,
            block_offsets=self._block_offsets,
            total_size=self._total_size,
            dtype=self._dtype,
        )

    # ----------------------------------------------------------- pytree
    def tree_flatten(self):  # type: ignore[override]
        # Emit the MATERIALIZED _data leaf so jit/grad boundaries are identical
        # to a plain SymmetricTensor. The stacked cache is intentionally dropped
        # (within-trace optimization only).
        return [self._materialize()], (
            self._block_keys,
            self._block_shapes,
            self._block_offsets,
            self._indices,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):  # type: ignore[override]
        # Return a PLAIN SymmetricTensor; the cache does not cross pytree edges.
        block_keys, block_shapes, block_offsets, indices = aux
        return SymmetricTensor._raw(
            indices=indices,
            data=children[0],
            block_keys=block_keys,
            block_shapes=block_shapes,
            block_offsets=block_offsets,
        )
