"""Value fingerprints for the implicit-AD compile caches (#1049).

``_VJP_CACHE`` (``_ctm_energy_ad``, ``_ctm_honeycomb_ad``) maps a static
configuration to a compiled ``custom_vjp``.  The gate and the energy callback
are part of that configuration -- the JIT'd backward reads them at trace time
and bakes them in as compile-time constants -- and they used to enter the key
as ``id(gate)`` / ``id(energy_fn)``.  ``optimize_gs_ad`` builds its energy
callback as a closure inside the optimizer, so every call had a fresh id,
missed the cache, re-traced and re-compiled the whole backward, and left the
old entry behind.

:func:`cache_key_part` replaces those ids with a fingerprint of what decides
the traced program, when one can be taken **exactly**: a function's code
object, defaults and the fingerprints of what it closes over; an array's
shape, dtype, weak-type bit and contents; a pytree's node types, static
(auxiliary) data and leaves; a plain object's type and attribute state.  Two
objects with equal exact fingerprints trace to the same program, so reusing the
compiled backward is exact, not approximate.

If any part cannot be fingerprinted exactly -- an object with no inspectable
state, an array too large to hash per call, a recursion past the depth bound --
the whole component falls back to ``id(obj)``, exactly the old key.  That can
only cause a cache *miss*, never a wrong hit, as long as the object stays alive
while a key holds its id; the returned ``keepalive`` list is for that.

Equality of static data is decided here, field by field, and never delegated
to ``__eq__``: ``TensorIndex.__eq__`` ignores ``fuse_info``, which
``split_index`` branches on, so two gates it calls equal can trace differently.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import types

import jax
import numpy as np

#: Arrays larger than this are not content-hashed: a per-call device-to-host
#: copy of a big captured constant would cost more than it saves, and an
#: identity key would not see an in-place mutation.  Such a component falls
#: back to the identity of the object that holds it.
_MAX_HASHED_ELEMENTS = 1 << 20

#: Recursion bound for closures that reach themselves or deep object graphs.
_MAX_DEPTH = 12

_SCALARS = (type(None), bool, int, float, complex, str, bytes)


class _Inexact(Exception):
    """Raised inside the walk when a component cannot be fingerprinted exactly."""


def cache_key_part(obj) -> tuple[tuple, list]:
    """``(key, keepalive)`` for one component of a compile-cache key.

    ``key`` is a value fingerprint when ``obj`` can be fingerprinted exactly,
    else ``("id", id(obj))``.  ``keepalive`` holds every object the key refers
    to by identity; the cache entry must keep them alive for as long as the
    key is cached.
    """
    keep: list = []
    try:
        return ("fp", _fp(obj, keep, 0)), keep
    except _Inexact:
        return ("id", id(obj)), [obj]


def _fp(obj, keep, depth):
    if depth > _MAX_DEPTH:
        raise _Inexact
    d = depth + 1
    if isinstance(obj, _SCALARS):
        return ("v", type(obj).__name__, obj)
    if isinstance(obj, enum.Enum):
        return ("enum", type(obj), obj.value)
    if isinstance(obj, type):
        return ("type", obj)
    if isinstance(obj, np.dtype):
        return ("dtype", obj.str)
    if isinstance(obj, (np.generic, np.ndarray, jax.Array)):
        return _fp_array(obj)
    if isinstance(obj, (tuple, list)):
        return (type(obj), tuple(_fp(x, keep, d) for x in obj))
    if isinstance(obj, dict):
        # Insertion order is observable (``.items()``/``.values()``), so it is
        # part of the key.
        return (
            type(obj),
            tuple((_fp(k, keep, d), _fp(v, keep, d)) for k, v in obj.items()),
        )
    if isinstance(obj, (set, frozenset)):
        return (type(obj), tuple(sorted((_fp(x, keep, d) for x in obj), key=repr)))
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
                contents = cell.cell_contents
            except ValueError:  # an empty cell (referenced before assignment)
                cells.append(("empty-cell",))
                continue
            cells.append(_fp(contents, keep, d))
        keep.append(obj.__globals__)
        return (
            "fn",
            obj.__code__,  # equal code objects compare by content
            ("globals", id(obj.__globals__)),  # the namespace names resolve in
            _fp(obj.__defaults__, keep, d),
            _fp(obj.__kwdefaults__, keep, d),
            tuple(cells),
        )
    if isinstance(obj, (types.BuiltinFunctionType, types.ModuleType)):
        keep.append(obj)
        return ("id-stable", type(obj).__name__, id(obj))
    # A registered pytree node (DenseTensor, SymmetricTensor, CTMTensorEnv,
    # ...): node types, static data and leaves, walked here rather than
    # compared with ``PyTreeDef.__eq__`` (see the module docstring).
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    if not (len(leaves) == 1 and leaves[0] is obj):
        return _fp_tree(treedef, iter(leaves), keep, d)
    if dataclasses.is_dataclass(obj):
        return (
            "dc",
            type(obj),
            tuple(
                (f.name, _fp(getattr(obj, f.name), keep, d))
                for f in dataclasses.fields(obj)
            ),
        )
    state = _object_state(obj)
    if state is not None:
        return ("obj", type(obj), _fp(state, keep, d))
    raise _Inexact


def _fp_tree(treedef, leaves, keep, depth):
    if depth > _MAX_DEPTH:
        raise _Inexact
    node = treedef.node_data()
    if node is None:  # a leaf
        return _fp(next(leaves), keep, depth + 1)
    node_type, aux = node
    return (
        "node",
        node_type,
        _fp(aux, keep, depth + 1),
        tuple(_fp_tree(c, leaves, keep, depth + 1) for c in treedef.children()),
    )


def _object_state(obj):
    """Attribute state of a plain object, or ``None`` if it has none to read."""
    d = getattr(obj, "__dict__", None)
    has_dict = isinstance(d, dict)
    state = dict(d) if has_dict else {}
    has_slots = False
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if name in ("__dict__", "__weakref__"):
                continue
            has_slots = True
            if hasattr(obj, name):
                state[name] = getattr(obj, name)
    if not (has_dict or has_slots):
        return None
    return state


def _fp_array(x):
    if isinstance(x, jax.core.Tracer) or getattr(x, "size", 0) > _MAX_HASHED_ELEMENTS:
        raise _Inexact
    weak = bool(getattr(x, "weak_type", False))
    a = np.ascontiguousarray(np.asarray(x))
    return ("arr", a.shape, a.dtype.str, weak, hashlib.sha1(a.tobytes()).hexdigest())


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
