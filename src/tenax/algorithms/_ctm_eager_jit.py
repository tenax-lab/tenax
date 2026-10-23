"""Jit the static-shape pieces of an eager CTM sweep (#1098 compile cost).

Outside ``jax.jit`` every JAX primitive compiles one executable per new
shape.  An eager symmetric CTM sweep runs thousands of block-level
primitives, so at chi=16 the #1098 layout phase spends hours compiling
(measured 2026-10-23: 3249 compiles in one D=3 chi=8 sweep).  Only the M'
truncation needs concrete values; :func:`call` runs every other piece under
one ``jax.jit`` per function and input structure.

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


@functools.cache
def _jitted(
    fn: Callable, static_argnames: tuple[str, ...], flags: tuple[str | None, ...]
) -> Callable:
    # JAX keys its trace cache on the function object, so a second
    # ``jax.jit(fn)`` would reuse the first trace and the flags baked into it.
    # A fresh wrapper per flag snapshot gets its own trace and executable.
    del flags

    @functools.wraps(fn)
    def piece(*args, **kwargs):
        return fn(*args, **kwargs)

    return jax.jit(piece, static_argnames=static_argnames)


def _has_tracer(tree) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def _is_symmetric(x) -> bool:
    return isinstance(x, SymmetricTensor)


def _has_symmetric(tree) -> bool:
    return any(_is_symmetric(x) for x in jax.tree.leaves(tree, is_leaf=_is_symmetric))


def call(fn: Callable, *args, static_argnames: tuple[str, ...] = (), **kwargs):
    """``fn(*args, **kwargs)``, under a cached ``jax.jit`` for concrete symmetric inputs."""
    if not _ENABLED or _has_tracer((args, kwargs)):
        return fn(*args, **kwargs)
    if not _has_symmetric((args, kwargs)):
        # The dense eager path stays as it was (plan Global Constraint).
        return fn(*args, **kwargs)
    if _contractor._strict_contract():
        # A cached executable would skip the per-call #834 checks.
        return fn(*args, **kwargs)
    return _jitted(fn, tuple(static_argnames), _env_flags())(*args, **kwargs)
