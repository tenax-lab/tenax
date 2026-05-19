"""Checkpoint primitives for iPEPS AD long runs.

Serializes optimizer state to disk so a run can be resumed from a
mid-point after a crash, OOM, or manual interrupt.  The schema is a
flat dict (no dataclass) so adding new state fields is a forward-
compatible change: older checkpoints loaded by newer code simply
return ``None`` for missing keys, and the caller falls back to its
fresh-init default.

Two files are written per checkpoint path:

* ``ckpt.last.pkl`` — overwritten every K steps and at every
  chi-schedule stage boundary; this is what ``gs_resume=True``
  reloads.
* ``ckpt.best.pkl`` — overwritten only when the current energy beats
  the prior best-seen value; this is the snapshot the user typically
  wants to publish.

Writes are atomic: pickle goes to ``<path>.tmp`` first, then
``os.replace()`` swaps it into place.  A crash mid-write leaves the
previous good file intact.

Format: pickle (cloudpickle if available, otherwise stdlib pickle).
Pickle was chosen over HDF5 because the bundle contains dataclasses
(iPEPSConfig, CTMConfig), variable-length L-BFGS history, and
Tensor objects with nested structure that HDF5 cannot self-describe
without a brittle flatten/unflatten layer.  Recovery checkpoints are
Python-only and ephemeral, so the standard HDF5 selling points
(cross-language, partial loads, archival) do not apply.

Config-compatibility on resume is enforced by ``validate_config``:
fields that would silently corrupt the run (``max_bond_dim``,
``unit_cell``, ``gs_c4v``, hamiltonian shape) raise; everything else
is allowed to differ with a warning.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import warnings
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1


def _tenax_git_sha() -> str | None:
    """Return the tenax git SHA if discoverable, else ``None``.

    Best-effort identifier so a checkpoint file records which code
    version produced it.  Failure is non-fatal — the SHA is metadata,
    not load-bearing.
    """
    try:
        # Run from the tenax package directory so we resolve the
        # tenax repo's HEAD, not the caller's CWD.
        import tenax

        pkg_dir = os.path.dirname(os.path.abspath(tenax.__file__))
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _config_to_dict(config: Any) -> dict:
    """Serialise an iPEPSConfig (dataclass tree) to a plain dict.

    Uses ``dataclasses.asdict`` which recurses into nested dataclasses
    (e.g. ``CTMConfig``) and returns a JSON-pickleable structure.

    Known limitations (unreachable in the 2-site-only scope wired in
    PR #497; relevant to 1-site / multisite follow-ups):

    * ``unit_cell=Lattice(...)`` — ``Lattice.neighbor_map`` is a
      ``MappingProxyType`` set in ``Lattice.__post_init__``; ``asdict``
      deepcopies through it and raises
      ``TypeError: cannot pickle 'mappingproxy' object``.  Fix when the
      multisite path is wired: convert ``neighbor_map`` to a plain dict
      before snapshotting (or store a stable Lattice identifier).
    * ``cg_gates=CGGates(...)`` — ``CGGates`` is a dataclass with
      ``jnp.ndarray`` fields; ``asdict`` produces nested arrays and
      ``validate_config`` below then raises
      ``ValueError: truth value of an array...`` on dict-eq comparison.
      Fix when the 1-site cg_gates path is wired: special-case
      ``cg_gates`` with an array-aware (hash or ``np.array_equal``)
      comparator.

    Both paths are currently blocked by the dispatch guard at
    ``optimize_gs_ad`` (``ipeps_optimize.py``), which raises
    ``NotImplementedError`` if ``gs_checkpoint_path`` is set with a
    non-``"2site"`` ``unit_cell``.  See PR #497 review threads.
    """
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def save_checkpoint(
    state: dict,
    path: str,
    *,
    is_best: bool = False,
) -> str:
    """Atomically write ``state`` to ``path`` as a pickle.

    Adds metadata (schema version, tenax SHA, timestamp) under the
    keys ``"schema_version"``, ``"tenax_sha"``, ``"timestamp"`` if
    they are not already present in ``state``.

    Args:
        state: Plain dict of state to serialise.  Values must be
            pickleable; JAX arrays are auto-handled by pickle.
        path: Directory that holds the checkpoint files.  Created if
            it does not exist.
        is_best: When ``True``, writes to ``ckpt.best.pkl``; otherwise
            writes to ``ckpt.last.pkl``.

    Returns:
        The absolute path of the file that was written.
    """
    os.makedirs(path, exist_ok=True)
    fname = "ckpt.best.pkl" if is_best else "ckpt.last.pkl"
    final_path = os.path.join(path, fname)
    tmp_path = final_path + ".tmp"

    bundle = dict(state)
    bundle.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    bundle.setdefault("tenax_sha", _tenax_git_sha())
    bundle.setdefault(
        "timestamp",
        datetime.now(UTC).isoformat(timespec="seconds"),
    )

    with open(tmp_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, final_path)
    return final_path


def load_checkpoint(path: str, *, prefer_best: bool = False) -> dict | None:
    """Load a checkpoint from ``path``.

    Args:
        path: Directory that holds the checkpoint files.
        prefer_best: When ``True``, loads ``ckpt.best.pkl``; otherwise
            loads ``ckpt.last.pkl``.

    Returns:
        The bundled state dict, or ``None`` if no checkpoint file
        exists at ``path``.

    Raises:
        ValueError: If the file's ``schema_version`` is newer than
            ``CHECKPOINT_SCHEMA_VERSION`` (would be loaded under an
            unknown schema).
    """
    fname = "ckpt.best.pkl" if prefer_best else "ckpt.last.pkl"
    final_path = os.path.join(path, fname)
    if not os.path.isfile(final_path):
        return None

    with open(final_path, "rb") as f:
        bundle = pickle.load(f)

    version = bundle.get("schema_version", 0)
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint at {final_path!r} has schema_version={version}, "
            f"but this tenax build supports up to "
            f"{CHECKPOINT_SCHEMA_VERSION}. Update tenax to load this "
            f"checkpoint."
        )
    return bundle


_FATAL_CONFIG_FIELDS = (
    "max_bond_dim",
    "unit_cell",
    "gs_c4v",
    "gs_implicit_ad",
)


def validate_config(saved_config: dict, current_config: Any) -> None:
    """Verify a fresh ``iPEPSConfig`` is compatible with a saved one.

    Fields whose change would silently corrupt the run
    (``max_bond_dim``, ``unit_cell``, ``gs_c4v``, ``gs_implicit_ad``)
    raise ``ValueError`` on mismatch.  All other differences warn but
    proceed — this lets a user resume with a longer
    ``gs_num_steps`` or a different log interval without re-running
    from scratch.

    Args:
        saved_config: The ``config`` dict that was pickled into the
            checkpoint (as produced by ``_config_to_dict``).
        current_config: The fresh ``iPEPSConfig`` instance the caller
            wants to resume with.
    """
    current_dict = _config_to_dict(current_config)
    fatal_diffs = []
    soft_diffs = []
    for k in saved_config.keys() | current_dict.keys():
        if saved_config.get(k) != current_dict.get(k):
            if k in _FATAL_CONFIG_FIELDS:
                fatal_diffs.append(
                    f"{k}: saved={saved_config.get(k)!r} -> "
                    f"current={current_dict.get(k)!r}"
                )
            else:
                soft_diffs.append(k)

    if fatal_diffs:
        raise ValueError(
            "Cannot resume: checkpoint config differs from current "
            "config on load-bearing fields:\n  - " + "\n  - ".join(fatal_diffs)
        )

    if soft_diffs:
        warnings.warn(
            "Resuming with config changes on non-fatal fields: "
            + ", ".join(sorted(soft_diffs))
            + ". The checkpoint will be loaded; current-config values "
            "take effect from the next step onward.",
            stacklevel=2,
        )


def checkpoint_exists(path: str | None, *, prefer_best: bool = False) -> bool:
    """Return ``True`` iff a checkpoint file exists at ``path``.

    Convenience for the resume guard in ``_optimize_gs_ad_*``.  A
    ``None`` path returns ``False`` (no checkpoint configured).
    """
    if path is None:
        return False
    fname = "ckpt.best.pkl" if prefer_best else "ckpt.last.pkl"
    return os.path.isfile(os.path.join(path, fname))
