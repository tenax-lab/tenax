"""Jit the static-shape pieces of an eager CTM sweep (#1098 compile cost).

Outside ``jax.jit`` every JAX primitive compiles one executable per new
shape.  An eager symmetric CTM sweep runs thousands of block-level
primitives, so at chi=16 the #1098 layout phase spends hours compiling
(measured 2026-10-23: 3249 compiles in one D=3 chi=8 sweep).  Only the M'
truncation needs concrete values; :func:`call` runs every other piece under
one compiled executable per function and input structure, kept in a small
LRU cache (:data:`_MAX_EXECUTABLES`).

Under an enclosing trace (the jitted ``_step``, AD) :func:`call` calls the
function directly, so the traced program is the one it was before.  It also
calls the function directly when no input is a :class:`SymmetricTensor`, so
the dense eager path does not change.

The contraction path reads some ``TENAX_*`` environment flags per call (see
:data:`_CONTRACTION_ENV_FLAGS`).  Inside a jitted piece such a flag is read
only at trace time and its value is baked into the executable.  So the flag
values are part of the cache key: a new value gives a new executable.  While
``TENAX_STRICT_CONTRACT`` is armed, :func:`call` does not jit at all, so the
#834 audit checks run on every call, as they do without this module.
"""

from __future__ import annotations

import collections
import functools
import os
from collections.abc import Callable

import jax

from tenax.contraction import contractor as _contractor
from tenax.core.tensor import SymmetricTensor

# Module switch: ``False`` runs every piece eagerly, as before this module.
_ENABLED = True


# The env flags that the contraction path reads per call, so a trace bakes
# them in.  ``TENAX_STRICT_CONTRACT`` is not here: :func:`call` does not jit
# while it is armed.
_CONTRACTION_ENV_FLAGS = (
    "TENAX_BATCH_BLOCKSPARSE",
    "TENAX_BLOCKSPARSE_BACKEND",
    "TENAX_STACK_BLOCKSPARSE",
    "TENAX_STACK_PERSIST_RETURN",
    "TENAX_USE_CUTENSOR_BLOCKSPARSE",
    "TENAX_USE_CUTENSORNET",
)


def _env_flags() -> tuple[str | None, ...]:
    return tuple(os.environ.get(name) for name in _CONTRACTION_ENV_FLAGS)


# Bounded LRU of compiled pieces, keyed by function, static values, flag
# snapshot and input structure.  ``jax.jit`` keeps every executable it has
# compiled until ``jax.clear_caches()``, and an eager sweep rarely repeats a
# layout: in ``test_ctm_symmetric_rdm_positive_853[20]`` 482 of 562
# executables ran once, and one ``_symmetric_projector_prime`` executable
# held ~130 MB, so an unbounded cache peaked at 14.7 GB RSS.  Evicting the
# oldest entry frees its executable.  Measured on that test (CPU), peak RSS /
# wall time: 4 -> 4.6 GB / 12.5 min, 16 -> 5.0 GB / 11.4 min, 64 -> 6.0 GB /
# 11.3 min; unbounded 14.7 GB / 11.6 min; jit off 5.3 GB / 90 min.
_MAX_EXECUTABLES = 16
_executables: collections.OrderedDict = collections.OrderedDict()


def _clear() -> None:
    """Drop every cached executable."""
    _executables.clear()


def _key(fn, static_argnames, args, kwargs):
    static = tuple((k, kwargs[k]) for k in static_argnames if k in kwargs)
    dynamic = {k: v for k, v in kwargs.items() if k not in static_argnames}
    leaves, treedef = jax.tree.flatten((args, dynamic))
    avals = tuple(jax.typeof(x) for x in leaves)
    # An ahead-of-time executable is bound to its input shardings, so the
    # same layout on another device needs its own entry.
    shardings = tuple(getattr(x, "sharding", None) for x in leaves)
    return (fn, static, _env_flags(), treedef, avals, shardings), dynamic


def _compiled(fn: Callable, static_argnames: tuple[str, ...], args, kwargs):
    """The compiled piece for these inputs and the dynamic kwargs to call it with."""
    key, dynamic = _key(fn, static_argnames, args, kwargs)
    hit = _executables.get(key)
    if hit is not None:
        _executables.move_to_end(key)
        return hit, dynamic

    # A fresh wrapper per entry, so JAX's trace cache for it dies with it.
    @functools.wraps(fn)
    def piece(*a, **kw):
        return fn(*a, **kw)

    compiled = (
        jax.jit(piece, static_argnames=static_argnames).lower(*args, **kwargs).compile()
    )
    _executables[key] = compiled
    while len(_executables) > _MAX_EXECUTABLES:
        _executables.popitem(last=False)
    return compiled, dynamic


def _has_tracer(tree) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def _is_symmetric(x) -> bool:
    return isinstance(x, SymmetricTensor)


def _has_symmetric(tree) -> bool:
    return any(_is_symmetric(x) for x in jax.tree.leaves(tree, is_leaf=_is_symmetric))


def call(fn: Callable, *args, static_argnames: tuple[str, ...] = (), **kwargs):
    """``fn(*args, **kwargs)``, compiled and cached (bounded) for concrete symmetric inputs."""
    if not _ENABLED or _has_tracer((args, kwargs)):
        return fn(*args, **kwargs)
    if not _has_symmetric((args, kwargs)):
        # The dense eager path stays as it was (plan Global Constraint).
        return fn(*args, **kwargs)
    if _contractor._strict_contract():
        # A cached executable would skip the per-call #834 checks.
        return fn(*args, **kwargs)
    compiled, dynamic = _compiled(fn, tuple(static_argnames), args, kwargs)
    return compiled(*args, **dynamic)
