"""Value keys for the implicit-AD compile caches (#1049).

``_VJP_CACHE`` (``_ctm_energy_ad``, ``_ctm_honeycomb_ad``) maps a static
configuration to a compiled ``custom_vjp``.  The gate, the lattice
``neighbors`` and the energy callback are part of that configuration -- the
JIT'd backward reads them at trace time and bakes them in -- and they used to
enter the key as ``id(...)``.  ``optimize_gs_ad`` builds its energy callback as
a closure inside the optimizer, so every call had a fresh id, missed the
cache, re-traced and re-compiled the whole backward, and left the old entry
behind.

:func:`cache_key_part` keys a component by **value only where the value is
the whole story**:

* **Data** -- scalars (floats and complexes by bit pattern, so ``-0.0`` is not
  ``0.0``), strings, ``None``, tuples, lists, dicts (in insertion order, which
  a trace can observe), arrays (shape, dtype, JAX weak-type bit, contents),
  and objects whose class is defined in ``tenax`` (tensors, ``TensorIndex``,
  ``FuseInfo``, symmetries, ``CGGates``, ...), walked field by field.  A tenax
  pytree node is walked node by node; static data is never compared with
  ``__eq__`` (``TensorIndex.__eq__`` ignores ``fuse_info``, which
  ``split_index`` branches on).
* **Declared callbacks** -- a function tenax builds and marks with
  :func:`declare_cache_key` is keyed by its code object plus the data it
  declares.  That declaration is a contract: the callback's trace depends on
  nothing else (tenax module-level functions and constants it calls are taken
  as fixed).

Everything else -- user callbacks, bound methods, sets, objects from other
packages, arrays too large to hash per call -- keeps the **old identity key**
for that whole component.  Behaviour can depend on state no fingerprint can
see (module globals, class attributes, external registries), so for those a
value key could hit a backward traced against stale state, while an identity
key can only miss.  The returned ``keepalive`` list keeps every object an
identity key names alive, so a recycled ``id`` cannot match.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import struct

import jax
import numpy as np

#: Arrays larger than this are not content-hashed: a per-call device-to-host
#: copy would cost more than it saves.  Their component keeps its identity key.
_MAX_HASHED_ELEMENTS = 1 << 20

#: Recursion bound for deep object graphs.
_MAX_DEPTH = 16

_ATTR = "_tenax_cache_key"


class _Inexact(Exception):
    """Raised inside the walk when a component cannot be keyed by value."""


def declare_cache_key(fn, *data):
    """Mark a tenax-built callback as keyed by its code object plus ``data``.

    Use only where ``fn``'s trace is determined by its code and ``data`` --
    the values its closure captures, e.g. ``d_phys``.  Returns ``fn``.

    The contract is checked here, on every declaration: each captured cell
    must be one of ``data`` (by identity) or a module-level tenax function
    (e.g. an energy routine imported locally by the optimizer).  Anything
    else raises ``ValueError`` -- a capture left out of the key would let a
    later call reuse a backward traced against a different value.
    """
    functions = []
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            continue  # an empty cell carries no value
        if any(value is x for x in data):
            continue
        if _is_tenax_module_function(value):
            # Keyed by identity: a factory that binds a different function
            # here, or a monkeypatched/reloaded one, gets a different key.
            functions.append(value)
            continue
        raise ValueError(
            f"declare_cache_key({fn.__qualname__}): the closure captures "
            f"{name!r}, which is not among the declared values"
        )
    setattr(fn, _ATTR, (data, tuple(functions)))
    return fn


def _is_tenax_module_function(obj) -> bool:
    import sys

    mod = sys.modules.get(getattr(obj, "__module__", None) or "")
    return (
        callable(obj)
        and mod is not None
        and mod.__name__.split(".")[0] == "tenax"
        and getattr(mod, getattr(obj, "__name__", ""), None) is obj
    )


def cache_key_part(obj) -> tuple[tuple, list]:
    """``(key, keepalive)`` for one component of a compile-cache key.

    ``key`` is ``("val", ...)`` when ``obj`` is data or a declared callback, else
    ``("id", id(obj))``.  The cache entry must keep ``keepalive`` alive for as
    long as ``key`` is cached.
    """
    try:
        return ("val", _key(obj, 0)), []
    except _Inexact:
        return ("id", id(obj)), [obj]


def _tenax_owned(t: type) -> bool:
    return (getattr(t, "__module__", "") or "").split(".")[0] == "tenax"


def _key(obj, depth):
    if depth > _MAX_DEPTH:
        raise _Inexact
    d = depth + 1
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        return (type(obj).__name__, obj)
    if isinstance(obj, float):
        return ("float", struct.pack("<d", obj))
    if isinstance(obj, complex):
        return ("complex", struct.pack("<dd", obj.real, obj.imag))
    if isinstance(obj, enum.Enum):
        return ("enum", type(obj), obj.value)
    if isinstance(obj, np.dtype):
        return ("dtype", obj.str)
    if isinstance(obj, (np.generic, np.ndarray, jax.Array)):
        return _key_array(obj)
    if callable(obj) and hasattr(obj, _ATTR) and hasattr(obj, "__code__"):
        data, functions = getattr(obj, _ATTR)
        return (
            "declared",
            obj.__code__,
            _key(data, d),
            # Module-level tenax functions: immortal, so an id is stable.
            tuple((f.__module__, f.__qualname__, id(f)) for f in functions),
        )
    if type(obj) in (tuple, list):
        return (type(obj).__name__, tuple(_key(x, d) for x in obj))
    if type(obj) is dict:
        return ("dict", tuple((_key(k, d), _key(v, d)) for k, v in obj.items()))
    if not _tenax_owned(type(obj)):
        raise _Inexact
    # A tenax pytree node (DenseTensor, SymmetricTensor, CTMTensorEnv, ...):
    # node types, static data and leaves.
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    if not (len(leaves) == 1 and leaves[0] is obj):
        return _key_tree(treedef, iter(leaves), d)
    if dataclasses.is_dataclass(obj):
        return (
            "dc",
            type(obj),
            tuple(
                (f.name, _key(getattr(obj, f.name), d)) for f in dataclasses.fields(obj)
            ),
        )
    state = _object_state(obj)
    if state is None:
        raise _Inexact
    return ("obj", type(obj), _key(state, d))


def _key_tree(treedef, leaves, depth):
    if depth > _MAX_DEPTH:
        raise _Inexact
    node = treedef.node_data()
    if node is None:  # a leaf
        return _key(next(leaves), depth + 1)
    node_type, aux = node
    if not (_tenax_owned(node_type) or node_type in (tuple, list, dict, type(None))):
        raise _Inexact
    return (
        "node",
        node_type,
        _key(aux, depth + 1),
        tuple(_key_tree(c, leaves, depth + 1) for c in treedef.children()),
    )


def _object_state(obj):
    """Instance state of a tenax object, or ``None`` if it has none to read."""
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


def _key_array(x):
    if isinstance(x, jax.core.Tracer) or getattr(x, "size", 0) > _MAX_HASHED_ELEMENTS:
        raise _Inexact
    # The array type is part of the key: a callback may branch on
    # ``isinstance(gate, np.ndarray)``.
    kind = "jax" if isinstance(x, jax.Array) else type(x).__name__
    weak = bool(getattr(x, "weak_type", False))
    # A NumPy array's layout is observable too (``x.flags``, ``x.strides``);
    # a jax.Array exposes none, so its key carries no layout.
    layout = None
    if isinstance(x, np.ndarray):
        layout = (x.flags.c_contiguous, x.flags.f_contiguous, x.strides)
    # The shape is read before the conversion: ``np.ascontiguousarray``
    # promotes a 0-d array to shape ``(1,)``.
    shape = tuple(np.shape(x))
    a = np.ascontiguousarray(np.asarray(x))
    return (
        "arr",
        kind,
        shape,
        a.dtype.str,
        weak,
        layout,
        hashlib.sha1(a.tobytes()).hexdigest(),
    )


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
