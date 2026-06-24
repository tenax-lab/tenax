"""Tests for the checkpoint primitive (``_checkpoint.py``).

Covers round-trip, atomic-write semantics, schema versioning, and
config-compatibility validation.  Resume-integration tests for the
2-site AD path live separately to keep this file unit-scoped.
"""

from __future__ import annotations

import os
import pickle
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_exists,
    gate_fingerprint,
    load_checkpoint,
    save_checkpoint,
    validate_config,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.core
def test_save_load_round_trip(tmp_path):
    state = {
        "step": 7,
        "best_energy": -0.5,
        "params": jnp.array([1.0, 2.0, 3.0]),
        "history": [jnp.array([1.0]), jnp.array([2.0, 3.0])],
        "config": {"max_bond_dim": 2, "ctm": {"chi": 4}},
    }
    written = save_checkpoint(state, str(tmp_path))
    assert os.path.isfile(written)
    assert os.path.basename(written) == "ckpt.last.pkl"

    loaded = load_checkpoint(str(tmp_path))
    assert loaded is not None
    assert loaded["step"] == 7
    assert loaded["best_energy"] == -0.5
    np.testing.assert_array_equal(loaded["params"], np.array([1.0, 2.0, 3.0]))
    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert "timestamp" in loaded


@pytest.mark.core
def test_load_missing_file_returns_none(tmp_path):
    assert load_checkpoint(str(tmp_path)) is None
    assert load_checkpoint(str(tmp_path), prefer_best=True) is None


@pytest.mark.core
def test_checkpoint_exists(tmp_path):
    assert not checkpoint_exists(str(tmp_path))
    assert not checkpoint_exists(None)
    save_checkpoint({"step": 0}, str(tmp_path))
    assert checkpoint_exists(str(tmp_path))
    assert not checkpoint_exists(str(tmp_path), prefer_best=True)


@pytest.mark.core
def test_best_and_last_are_independent(tmp_path):
    save_checkpoint({"step": 1, "energy": -0.1}, str(tmp_path))
    save_checkpoint({"step": 1, "energy": -0.5}, str(tmp_path), is_best=True)
    save_checkpoint({"step": 2, "energy": -0.3}, str(tmp_path))

    last = load_checkpoint(str(tmp_path))
    best = load_checkpoint(str(tmp_path), prefer_best=True)
    assert last["step"] == 2 and last["energy"] == -0.3
    assert best["step"] == 1 and best["energy"] == -0.5


@pytest.mark.core
def test_atomic_write_no_partial_file_on_crash(tmp_path, monkeypatch):
    """A crash mid-write must leave any prior good file intact.

    Forces ``pickle.dump`` to raise after the tmp file is created but
    before ``os.replace`` runs; the prior good ``ckpt.last.pkl`` must
    still be readable and the tmp file must not exist as the final
    file.
    """
    save_checkpoint({"step": 1, "marker": "good"}, str(tmp_path))

    real_dump = pickle.dump
    call_count = {"n": 0}

    def _failing_dump(obj, fp, *args, **kwargs):
        call_count["n"] += 1
        # Write a few bytes so the tmp file exists & is non-empty
        fp.write(b"partial")
        raise RuntimeError("simulated crash mid-dump")

    monkeypatch.setattr(pickle, "dump", _failing_dump)
    with pytest.raises(RuntimeError, match="simulated crash"):
        save_checkpoint({"step": 2, "marker": "bad"}, str(tmp_path))
    monkeypatch.setattr(pickle, "dump", real_dump)

    # Final file untouched, .tmp left over (caller can clean up).
    loaded = load_checkpoint(str(tmp_path))
    assert loaded["step"] == 1 and loaded["marker"] == "good"
    assert call_count["n"] == 1


@pytest.mark.core
def test_load_rejects_newer_schema(tmp_path):
    """A checkpoint with schema_version > current must raise."""
    bogus = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION + 99,
        "step": 0,
    }
    final_path = os.path.join(str(tmp_path), "ckpt.last.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(bogus, f)

    with pytest.raises(ValueError, match="schema_version"):
        load_checkpoint(str(tmp_path))


@pytest.mark.core
def test_validate_config_fatal_mismatch_raises():
    saved = {
        "max_bond_dim": 2,
        "unit_cell": "2site",
        "gs_c4v": True,
        "gs_implicit_ad": True,
    }
    current = iPEPSConfig(
        max_bond_dim=3,  # changed - fatal
        unit_cell="2site",
        gs_c4v=True,
        gs_implicit_ad=True,
    )
    with pytest.raises(ValueError, match="max_bond_dim"):
        validate_config(saved, current)


@pytest.mark.core
def test_validate_config_soft_mismatch_warns():
    current = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        gs_c4v=True,
        gs_implicit_ad=True,
        gs_num_steps=50,  # different from saved (soft)
    )
    saved = {
        "max_bond_dim": 2,
        "unit_cell": "2site",
        "gs_c4v": True,
        "gs_implicit_ad": True,
        "gs_num_steps": 20,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_config(saved, current)
    assert any("gs_num_steps" in str(w.message) for w in caught)


@pytest.mark.core
def test_gate_fingerprint_round_trip_and_changes_with_bytes():
    """Same bytes → same fingerprint; different bytes → different.

    Also verifies that ``jnp.array`` and ``np.array`` with identical
    values produce the same fingerprint (both go through
    ``np.asarray``).
    """
    g1 = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    g2 = jnp.array([[1.0, 0.0], [0.0, -1.0]])  # same bytes
    g3 = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # different bytes (sign flip)
    fp1, fp2, fp3 = (gate_fingerprint(g) for g in (g1, g2, g3))
    assert fp1 == fp2
    assert fp1 != fp3
    # Cross-typesystem stability: jnp ↔ np.
    fp1_np = gate_fingerprint(np.asarray(g1))
    assert fp1 == fp1_np


@pytest.mark.core
def test_validate_config_compatible_no_warning():
    cfg = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        gs_c4v=True,
        gs_implicit_ad=True,
        ctm=CTMConfig(chi=4),
    )
    saved = {**cfg.__dict__}
    # asdict recurses through nested dataclasses; for the round-trip
    # comparison validate_config uses asdict on the live one, so make
    # the saved snapshot match that shape.
    from tenax.algorithms._checkpoint import _config_to_dict

    saved = _config_to_dict(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_config(saved, cfg)  # no warning expected


def test_cg_gates_fingerprint_round_trip_and_changes_with_bytes():
    import jax.numpy as jnp

    from tenax.algorithms._checkpoint import cg_gates_fingerprint
    from tenax.algorithms.coarse_grain import CGGates

    h_intra = jnp.eye(4)
    h_inter = {"h": jnp.ones((4, 4, 4, 4)), "v": jnp.zeros((4, 4, 4, 4))}
    g1 = CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=2, map_fn=None, init_fn=None)
    # same arrays, DIFFERENT callable -> same fingerprint (callables not hashed)
    g2 = CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=2,
                 map_fn=lambda x: x, init_fn=None)
    assert cg_gates_fingerprint(g1) == cg_gates_fingerprint(g2)
    # perturb h_intra -> different fingerprint
    g3 = CGGates(h_intra=h_intra.at[0, 0].add(1.0), h_inter=h_inter, n_sites=2)
    assert cg_gates_fingerprint(g3) != cg_gates_fingerprint(g1)
    # perturb an h_inter entry -> different fingerprint
    g4 = CGGates(h_intra=h_intra,
                 h_inter={"h": h_inter["h"].at[0, 0, 0, 0].add(1.0), "v": h_inter["v"]},
                 n_sites=2)
    assert cg_gates_fingerprint(g4) != cg_gates_fingerprint(g1)
