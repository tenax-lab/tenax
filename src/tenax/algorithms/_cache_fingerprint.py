"""Value fingerprints for the implicit-AD compile caches (#1049).

``_VJP_CACHE`` (``_ctm_energy_ad``, ``_ctm_honeycomb_ad``) maps a static
configuration to a compiled ``custom_vjp``.  The gate and the energy callback
are part of that configuration -- the JIT'd backward reads them at trace time
and bakes them in as compile-time constants -- and they used to enter the key
as ``id(gate)`` / ``id(energy_fn)``.  ``optimize_gs_ad`` builds its energy
callback as a closure inside the optimizer, so every call had a fresh id,
missed the cache, re-traced and re-compiled the whole backward, and left the
old entry behind.

:func:`fingerprint` replaces those ids with what actually decides the traced
program: a function's code object and the fingerprints of what it closes over,
an array's shape, dtype and contents, a pytree's structure and leaves.  Two
objects with equal fingerprints trace to the same program, so reusing the
compiled backward is exact, not approximate.

Anything it cannot see into falls back to ``id(obj)``, which can only cause a
cache *miss*, never a wrong hit -- provided the object stays alive while a key
holds its id, or a recycled id could match.  So :func:`fingerprint` also
returns the objects it keyed by id, and the cache entry must keep them.
"""

from __future__ import annotations

import functools
import hashlib
import types

import jax
import numpy as np

#: Arrays larger than this are keyed by identity instead of a content hash: a
#: per-call device-to-host copy of a big captured constant would cost more than
#: it saves, and identity is the conservative choice.
_MAX_HASHED_ELEMENTS = 1 << 20

#: Recursion bound for closures that reach themselves (or deep object graphs).
_MAX_DEPTH = 8

_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def fingerprint(obj) -> tuple[tuple, list]:
    """Return ``(key, keepalive)`` for ``obj``.

    ``key`` is hashable and equal for objects that trace to the same program.
    ``keepalive`` lists every object keyed by ``id`` -- the caller must keep
    them alive for as long as ``key`` is in a cache.
    """
    keep: list = []
    return _fp(obj, keep, 0), keep


def _by_id(obj, keep):
    keep.append(obj)
    return ("id", id(obj))


def _fp(obj, keep, depth):
    if depth > _MAX_DEPTH:
        return _by_id(obj, keep)
    d = depth + 1
    if isinstance(obj, _SCALARS):
        return ("v", type(obj).__name__, obj)
    if isinstance(obj, (np.generic, np.ndarray, jax.Array)):
        return _fp_array(obj, keep)
    if isinstance(obj, (tuple, list)):
        return (type(obj).__name__, tuple(_fp(x, keep, d) for x in obj))
    if isinstance(obj, dict):
        items = [(_fp(k, keep, d), _fp(v, keep, d)) for k, v in obj.items()]
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(obj, (set, frozenset)):
        return ("set", tuple(sorted((_fp(x, keep, d) for x in obj), key=repr)))
    if isinstance(obj, functools.partial):
        return (
            "partial",
            _fp(obj.func, keep, d),
            _fp(obj.args, keep, d),
            _fp(obj.keywords, keep, d),
        )
    if isinstance(obj, types.MethodType):
        return ("method", _fp(obj.__func__, keep, d), _fp(obj.__self__, keep, d))
    if isinstance(obj, types.FunctionType):
        cells = []
        for cell in obj.__closure__ or ():
            try:
                cells.append(_fp(cell.cell_contents, keep, d))
            except ValueError:  # an empty cell (referenced before assignment)
                cells.append(("empty-cell",))
        return (
            "fn",
            obj.__code__,  # equal code objects compare by content
            _by_id(obj.__globals__, keep),  # the module the code resolves names in
            _fp(obj.__defaults__, keep, d),
            _fp(obj.__kwdefaults__, keep, d),
            tuple(cells),
        )
    # A registered pytree (DenseTensor, SymmetricTensor, ...): its structure
    # (including static aux data) plus its leaves.
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    if not (len(leaves) == 1 and leaves[0] is obj):
        try:
            hash(treedef)
        except TypeError:
            return _by_id(obj, keep)
        return ("tree", treedef, tuple(_fp(x, keep, d) for x in leaves))
    return _by_id(obj, keep)


def _fp_array(x, keep):
    if isinstance(x, jax.core.Tracer) or getattr(x, "size", 0) > _MAX_HASHED_ELEMENTS:
        return _by_id(x, keep)
    a = np.ascontiguousarray(np.asarray(x))
    return ("arr", a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).hexdigest())


#: Bound on each implicit-AD ``_VJP_CACHE``.  An entry pins compiled
#: executables (the D=3 fused backward is a ~120 MB artifact) plus its gate,
#: energy callback and env, so an unbounded cache grows with every distinct
#: configuration a process visits (#1049).  Eight covers a chi schedule or a
#: short parameter scan without recompiling.
VJP_CACHE_MAXSIZE = 8


def lru_get(cache, key):
    """``cache[key]`` or ``None``; a hit becomes the most recently used."""
    entry = cache.get(key)
    if entry is not None and hasattr(cache, "move_to_end"):
        cache.move_to_end(key)
    return entry


def lru_put(cache, key, entry, maxsize: int = VJP_CACHE_MAXSIZE) -> None:
    """Insert ``entry`` and evict least-recently-used entries beyond ``maxsize``.

    Only real configuration keys (tuples) are evicted, so a test's planted
    sentinel entry is never dropped by this.
    """
    cache[key] = entry
    if hasattr(cache, "move_to_end"):
        cache.move_to_end(key)
    while sum(isinstance(k, tuple) for k in cache) > maxsize:
        oldest = next(k for k in cache if isinstance(k, tuple))
        del cache[oldest]
