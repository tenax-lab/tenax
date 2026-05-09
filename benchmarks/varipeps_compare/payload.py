"""Cross-library .npz payload — init iPEPS tensor + Hamiltonian gate + metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_payload(
    path: Path | str, *, init: np.ndarray, gate: np.ndarray, meta: dict
) -> None:
    """Write init+gate+meta to a single .npz file.

    Args:
        path: Output filename (typically ``<key>.npz``).
        init: Initial iPEPS site tensor as a numpy array.  Shape depends on path:
              ``single_site`` → ``(D, D, D, D, d)``;
              ``bipartite_2site`` → ``(2, D, D, D, D, d)`` stacking A and B.
        gate: Two-site Hamiltonian ``(d, d, d, d)``.  For the ``single_site``
              path this is the sublattice-rotated gate; for ``bipartite_2site``
              it is the bare Heisenberg gate.
        meta: JSON-serializable metadata dict.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, init=init, gate=gate, meta=np.array(json.dumps(meta)))


def load_payload(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a payload .npz, returning ``(init, gate, meta)``."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        init = f["init"]
        gate = f["gate"]
        meta = json.loads(str(f["meta"]))
    return init, gate, meta
