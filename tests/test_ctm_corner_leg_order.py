"""Corners leave the 2x2 multisite CTM sweep in the cold-init leg order.

The ``_ctm_tensor_absorb_*_2plaq`` moves used to return the corners with their
two legs swapped.  ``jax.jit`` keys its cache on the pytree treedef, which
holds the leg order, so every cold start compiled the jitted step and each
#1142 eager-jit piece twice.  The values were never affected: only the order.
"""

from __future__ import annotations

import copy

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
import tenax.algorithms._ctm_tensor_moves as moves
from tenax import DenseTensor
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    CORNER_LEG_ORDER,
    _to_corner_order,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

CHI = 6
_EDGE_ORDER = {
    "T1": ("t1_l", "u2", "t1_r"),
    "T2": ("t2_u", "r2", "t2_d"),
    "T3": ("t3_r", "d2", "t3_l"),
    "T4": ("t4_d", "l2", "t4_u"),
}


def _u1_pair():
    A, B = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    return {(0, 0): A, (1, 0): B}


def _dense_pair():
    return {c: DenseTensor(t.todense(), t.indices) for c, t in _u1_pair().items()}


def _fermionic_pair():
    """A random FermionicU1 pair, normalised.

    Copied from ``_fermionic_pair`` in ``tests/test_ctm_eager_sweep_jit.py``
    (itself from ``tests/test_ctm_674_fermionic_fused.py``).
    """
    sym = FermionicU1()

    def idx(charges, flow, label):
        return TensorIndex.from_charges(
            sym, np.array(charges, dtype=np.int32), flow, label=label
        )

    lA = [0, 1, -1]
    rA = [0, 0, 1]
    ud = [0, 1, -1]
    phys = [0, 1]

    def make(l_charges, r_charges, key):
        return SymmetricTensor.random_normal(
            (
                idx(ud, FlowDirection.OUT, "u"),
                idx(ud, FlowDirection.IN, "d"),
                idx(l_charges, FlowDirection.OUT, "l"),
                idx(r_charges, FlowDirection.IN, "r"),
                idx(phys, FlowDirection.IN, "phys"),
            ),
            jax.random.PRNGKey(key),
        )

    A = make(lA, rA, 0)
    B = make(rA, lA, 1)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


PAIRS = {"dense": _dense_pair, "u1": _u1_pair, "fermionic_u1": _fermionic_pair}


def _order(t):
    return tuple(ix.label for ix in t.indices)


def _cold(sites):
    return {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}


def _one_sweep(sites, envs):
    dls = {c: cpl._build_double_layer_tensor(t) for c, t in sites.items()}
    return _ctm_tensor_sweep_multisite(
        envs, dls, CHECKERBOARD_NEIGHBORS, CHI, True, recipe="2x2"
    )[0]


def _dense_by_label(t, labels):
    perm = tuple(_order(t).index(lab) for lab in labels)
    arr = t.transpose(perm).todense()
    return np.asarray(arr)


def test_the_table_is_the_cold_init_order():
    env = initialize_ctm_tensor_env(_u1_pair()[(0, 0)], CHI)
    assert {f: _order(getattr(env, f)) for f in CORNER_LEG_ORDER} == CORNER_LEG_ORDER


def test_to_corner_order_is_a_noop_on_the_canonical_order():
    env = initialize_ctm_tensor_env(_u1_pair()[(0, 0)], CHI)
    assert _to_corner_order(env.C1, "C1") is env.C1


@pytest.mark.parametrize("kind", ["dense", "u1"])
def test_2x2_sweep_returns_the_cold_init_order(kind, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = PAIRS[kind]()
    out = _one_sweep(sites, _cold(sites))
    for c, env in out.items():
        for f, labels in CORNER_LEG_ORDER.items():
            assert _order(getattr(env, f)) == labels, (kind, c, f)
        for f, labels in _EDGE_ORDER.items():
            assert _order(getattr(env, f)) == labels, (kind, c, f)


@pytest.mark.parametrize("kind", ["dense", "u1", "fermionic_u1"])
def test_sweep_values_match_the_old_order(kind, monkeypatch):
    """Same values as the swapped-order sweeps, once legs are aligned by label.

    Two sweeps, so every corner that a move transposed is consumed by a later
    move: the compared tensors include ones built from transposed corners,
    which is where a wrong Koszul sign would show.
    """
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = PAIRS[kind]()
    new = _one_sweep(sites, _one_sweep(sites, _cold(sites)))
    monkeypatch.setattr(moves, "_to_corner_order", lambda t, field: t)
    old = _one_sweep(sites, _one_sweep(sites, _cold(sites)))
    swapped = 0
    for c in new:
        for f in new[c]._fields:
            a, b = getattr(new[c], f), getattr(old[c], f)
            labels = _order(a)
            swapped += labels != _order(b)
            np.testing.assert_allclose(
                _dense_by_label(a, labels), _dense_by_label(b, labels), atol=1e-12
            )
    assert swapped > 0, "the identity patch must reproduce the old swapped order"


def test_old_order_input_comes_back_canonical(monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", False)
    sites = _u1_pair()
    envs = _cold(sites)
    flipped = {
        c: env._replace(
            **{f: getattr(env, f).transpose((1, 0)) for f in CORNER_LEG_ORDER}
        )
        for c, env in envs.items()
    }
    out = _one_sweep(sites, flipped)
    for env in out.values():
        for f, labels in CORNER_LEG_ORDER.items():
            assert _order(getattr(env, f)) == labels


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            "dense",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "a second, separate retrace: the dense cold init gives each "
                    "edge's D2 leg a one-sector index with the opposite flow, and "
                    "the sweep returns the double layer's three-sector index"
                ),
            ),
        ),
        "u1",
    ],
)
def test_jitted_step_compiles_once_from_a_cold_start(kind):
    """Calls 2 and 3 from a cold start retrace nothing.

    Counts ``jaxpr_trace_duration`` events: a trace fires even when the
    persistent compilation cache serves the executable, and ``_cache_size()``
    reads 0 under pytest-cov's tracer, so neither stands in for it.
    """
    sites = PAIRS[kind]()
    envs = _cold(sites)
    step = cpl._make_jit_ctm_step(copy.deepcopy(CHECKERBOARD_NEIGHBORS), "2x2")
    envs = step(sites, envs, chi=CHI)[0]
    traces = []

    def listener(name, _dur, **_kw):
        if name == "/jax/core/compile/jaxpr_trace_duration":
            traces.append(name)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        for _ in range(2):
            envs = step(sites, envs, chi=CHI)[0]
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert len(traces) == 0, len(traces)
