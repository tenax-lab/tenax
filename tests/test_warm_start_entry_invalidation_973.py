"""Entry-point invalidation of the implicit-AD warm-start cache (#973).

``_VJP_CACHE`` in ``_ctm_energy_ad`` is module-level, and its λ warm-start
seed used to survive across optimizer invocations: the second of two
back-to-back ``optimize_gs_ad`` calls with bit-identical configs seeded its
adjoint solves from the first call's state and returned a different energy
(~8e-5 after three steps).  The fix drops the seed at the entry of every
run that evaluates gradients through the cached implicit backward.

These tests pin the *wiring*: each run-starting implementation must
invalidate the seed before it touches its tensor arguments.  A planted
cache entry records whether the real ``invalidate_implicit_ad_warm_start``
reached it; junk tensor arguments then abort the run immediately after,
so each test costs milliseconds.  Deleting any one entry-point call, or
moving it after the argument handling, turns the recorded count to zero.

End-to-end determinism (repeated-run and A/B/A bit-equality) lives in
``test_optimizer_run_independence_973.py`` (slow bucket).
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig
from tenax.algorithms._ctm_energy_ad import _VJP_CACHE

_SENTINEL_KEY = "_test_973_sentinel"

# Junk args abort each optimizer right after its entry invalidation, so the
# expected failure is whatever the first real touch of the argument raises.
_JUNK_ERRORS = (TypeError, AttributeError, ValueError, KeyError)


def _plant_seed() -> dict:
    """Insert a fake ``_VJP_CACHE`` entry whose invalidation callback counts.

    Mirrors the real entry layout ``(compiled_fn, mutables)``:
    ``invalidate_implicit_ad_warm_start`` iterates values and calls
    ``mutables["_invalidate_warm_start"]``.  The string key can never
    collide with a real static-config cache key (a tuple).
    """
    fired = {"n": 0}

    def _cb() -> None:
        fired["n"] += 1

    _VJP_CACHE[_SENTINEL_KEY] = (None, {"_invalidate_warm_start": _cb})
    return fired


@pytest.fixture(autouse=True)
def _clean_sentinel():
    yield
    _VJP_CACHE.pop(_SENTINEL_KEY, None)


def _cfg(unit_cell: str = "1x1") -> iPEPSConfig:
    return iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=5),
        unit_cell=unit_cell,
        gs_c4v=True,
        gs_implicit_ad=True,
        gs_num_steps=1,
        gs_verbose=False,
    )


def test_1site_entry_drops_stale_seed_before_tensor_work():
    from tenax.algorithms.ipeps_optimize import _optimize_gs_ad_tensor

    fired = _plant_seed()
    with pytest.raises(_JUNK_ERRORS):
        _optimize_gs_ad_tensor(object(), object(), _cfg())
    assert fired["n"] == 1, (
        "#973: _optimize_gs_ad_tensor must invalidate the implicit-AD "
        "warm-start seed at entry, before touching its arguments "
        f"(callback fired {fired['n']} times)"
    )


def test_2site_entry_drops_stale_seed_before_tensor_work():
    from tenax.algorithms.ipeps_optimize import _optimize_gs_ad_tensor_2site

    fired = _plant_seed()
    with pytest.raises(_JUNK_ERRORS):
        _optimize_gs_ad_tensor_2site(object(), object(), _cfg("2site"))
    assert fired["n"] == 1, (
        "#973: _optimize_gs_ad_tensor_2site must invalidate the implicit-AD "
        "warm-start seed at entry, before touching its arguments "
        f"(callback fired {fired['n']} times)"
    )


def test_multisite_entry_drops_stale_seed_before_tensor_work():
    from tenax.algorithms.ipeps_optimize import _optimize_gs_ad_multisite

    fired = _plant_seed()
    with pytest.raises(_JUNK_ERRORS):
        # unit_cell is a plain string, so the Lattice handling right after
        # the entry sequence aborts the run.
        _optimize_gs_ad_multisite(object(), None, _cfg())
    assert fired["n"] == 1, (
        "#973: _optimize_gs_ad_multisite must invalidate the implicit-AD "
        "warm-start seed at entry, before touching its arguments "
        f"(callback fired {fired['n']} times)"
    )


def test_pess_supersite_entry_drops_stale_seed_before_tensor_work():
    from tenax.algorithms.pess_optimize import optimize_pess_ad

    fired = _plant_seed()
    with pytest.raises(_JUNK_ERRORS):
        optimize_pess_ad(object(), object(), object())
    assert fired["n"] == 1, (
        "#973: optimize_pess_ad must invalidate the implicit-AD warm-start "
        f"seed at entry (callback fired {fired['n']} times)"
    )


def test_pess_3site_multisite_entry_drops_stale_seed_before_tensor_work():
    from tenax.algorithms.pess_optimize import optimize_pess_3site_multisite_ad

    fired = _plant_seed()
    with pytest.raises(_JUNK_ERRORS):
        optimize_pess_3site_multisite_ad(object(), object(), object())
    assert fired["n"] == 1, (
        "#973: optimize_pess_3site_multisite_ad must invalidate the "
        "implicit-AD warm-start seed at entry "
        f"(callback fired {fired['n']} times)"
    )
