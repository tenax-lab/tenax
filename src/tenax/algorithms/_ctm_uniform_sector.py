"""Process-local "keep sectors" context for the U(1)-Sz uniform-sector env (#615).

Default OFF. When a keep set is active, the symmetric CTM env-init and the
projector truncation restrict chi-bond charges to the keep set, so the 2x2
multisite backward carries a *uniform* sector set and becomes traceable under
the sector drop (#610 NO-GO-by-obstruction).

The keep set is a STATIC, structural choice (it changes block sectors/shapes),
so it is read at TRACE time as a plain Python value via ``current_keep_sectors``
— never threaded as a traced array. Mirrors the codebase's existing module-level
toggles (``set_implicit_ad_norm_diagnostics``, ``_batch_blocksparse_enabled``).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterable

import numpy as np

_KEEP_SECTORS: frozenset[int] | None = None


def current_keep_sectors() -> frozenset[int] | None:
    """The active keep set, or ``None`` (default: no restriction)."""
    return _KEEP_SECTORS


@contextlib.contextmanager
def keep_sectors_context(keep: frozenset[int] | Iterable[int] | None):
    """Activate a keep set for the duration of the ``with`` block.

    ``keep=None`` is a no-op pass-through (restores the default path), so callers
    can wrap unconditionally: ``with keep_sectors_context(keep_sectors): ...``.
    """
    global _KEEP_SECTORS
    prev = _KEEP_SECTORS
    _KEEP_SECTORS = None if keep is None else frozenset(int(q) for q in keep)
    try:
        yield
    finally:
        _KEEP_SECTORS = prev


def restrict_charges_to_keep(charges, keep: frozenset[int] | None) -> np.ndarray:
    """Return ``charges`` with entries outside ``keep`` removed.

    Degenerate guard: if filtering would empty the array, return the original
    (an env bond / SVD sector list must never be empty).
    """
    arr = np.asarray(charges, dtype=np.int32)
    if keep is None:
        return arr
    mask = np.array([int(q) in keep for q in arr], dtype=bool)
    return arr[mask] if mask.any() else arr
