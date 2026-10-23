"""Jit the static-shape pieces of an eager CTM sweep (#1098 compile cost).

Outside ``jax.jit`` every JAX primitive compiles one executable per new
shape.  An eager symmetric CTM sweep runs thousands of block-level
primitives, so at chi=16 the #1098 layout phase spends hours compiling
(measured 2026-10-23: 3249 compiles in one D=3 chi=8 sweep).  Only the M'
truncation needs concrete values; :func:`call` runs every other piece under
one ``jax.jit`` per function and input structure.

Under an enclosing trace (the jitted ``_step``, AD) :func:`call` calls the
function directly, so the traced program is the one it was before.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax

# Module switch: ``False`` runs every piece eagerly, as before this module.
_ENABLED = True


@functools.cache
def _jitted(fn: Callable, static_argnames: tuple[str, ...]) -> Callable:
    return jax.jit(fn, static_argnames=static_argnames)


def _has_tracer(tree) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def call(fn: Callable, *args, static_argnames: tuple[str, ...] = (), **kwargs):
    """``fn(*args, **kwargs)``, under a cached ``jax.jit`` when the inputs are concrete."""
    if not _ENABLED or _has_tracer((args, kwargs)):
        return fn(*args, **kwargs)
    return _jitted(fn, tuple(static_argnames))(*args, **kwargs)
