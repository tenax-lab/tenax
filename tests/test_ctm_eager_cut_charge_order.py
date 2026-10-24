"""The eager M' cut emits its new chi bond in ascending-charge order.

The cut keeps a global top-chi of the full SVD, whose bond runs in
singular-value order, so the bond's charge *sequence* depended on the values.
Moves inside one sweep consume the tensors earlier moves wrote, before the
end-of-sweep charge sort of the #1098 layout phase, so every eager sweep fed
new index metadata to the #1142 jitted pieces and retraced them all.  The
jitted projector and the cold init already emit ascending charge order.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
import tenax.algorithms._ctm_tensor_projector_2x2 as p2x2
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

CHI = 8


def _u1_pair():
    A, B = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    return {(0, 0): A, (1, 0): B}


def _fermionic_pair():
    """A random FermionicU1 pair, normalised (``tests/test_ctm_eager_sweep_jit.py``)."""
    sym = FermionicU1()

    def idx(charges, flow, label):
        return TensorIndex.from_charges(
            sym, np.array(charges, dtype=np.int32), flow, label=label
        )

    lA, rA, ud, phys = [0, 1, -1], [0, 0, 1], [0, 1, -1], [0, 1]

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

    A, B = make(lA, rA, 0), make(rA, lA, 1)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


PAIRS = {"u1": _u1_pair, "fermionic_u1": _fermionic_pair}


def _sweeps(sites, n):
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}
    dls = {c: cpl._build_double_layer_tensor(t) for c, t in sites.items()}
    for _ in range(n):
        envs = _ctm_tensor_sweep_multisite(
            envs, dls, CHECKERBOARD_NEIGHBORS, CHI, True, recipe="2x2"
        )[0]
    return envs


def test_charge_ordered_slots_sorts_by_charge_and_keeps_sector_order():
    charges = np.array([0, 2, -2, 0, 2, -2, 4], dtype=np.int32)
    keep = [0, 1, 2, 3, 5, 6]
    out = p2x2._charge_ordered_slots(keep, charges)
    assert sorted(out) == keep
    assert out == [2, 5, 0, 3, 1, 6]


def _chi_legs(envs):
    for c, env in envs.items():
        for f, t in zip(env._fields, env):
            for ix in t.indices:
                if ix.label not in cpl._D2_LEG_LABELS:
                    yield c, f, ix


@pytest.mark.parametrize("kind", ["u1", "fermionic_u1"])
def test_eager_sweep_chi_legs_are_in_ascending_charge_order(kind, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", False)
    for c, f, ix in _chi_legs(_sweeps(PAIRS[kind](), 2)):
        q = np.asarray(ix.charges)
        assert np.array_equal(q, np.sort(q, kind="stable")), (c, f, ix.label, q)


def _su_u1_pair():
    A0, B0 = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=2, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=CHI)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


def _converged(sites):
    cfg = dataclasses.replace(
        CTMConfig(chi=CHI),
        conv_tol=1e-10,
        max_iter=200,
        min_iter=5,
        plateau_patience=None,
        on_unconverged="warn",
    )
    envs, _ = cpl.python_loop_ctm_converge(
        sites, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
    )
    g = xxz_gate(0.3).todense()
    e = compute_energy_ctm_tensor_2site(
        sites[(0, 0)], sites[(1, 0)], envs[(0, 0)], envs[(1, 0)], g, 2
    )
    return float(e), cpl._chi_layout(envs)


def test_sorted_cut_converges_to_the_same_energy_and_layout(monkeypatch):
    """Converged from an SU state: same energy and chi layout as the unsorted cut.

    Not tensor by tensor along the way: from a cold init the early cuts are
    exactly degenerate, ``_select_chi_slots`` breaks exact ties by bond
    position, and the sort changes the positions, so the trajectories differ
    (measured: corner spectra 1.5% apart after two raw sweeps).  The fixed
    point must not.
    """
    sites = _su_u1_pair()
    e_new, layout_new = _converged(sites)
    monkeypatch.setattr(p2x2, "_charge_ordered_slots", lambda keep, charges: keep)
    e_old, layout_old = _converged(sites)
    assert layout_new == layout_old
    assert abs(e_new - e_old) < 1e-9, (e_new, e_old)


def test_a_settled_eager_sweep_retraces_no_jitted_piece():
    """With the layout settled, the next eager sweep traces no jitted piece.

    A guard against value-dependent metadata in any piece's input.  It does
    not by itself pin the charge sort: in this settled D=2 state the
    singular-value order is stable from sweep to sweep, so the unsorted cut
    passes too (measured).  The sort's own property is tested directly above.

    Asserts the regime first: sweeps until one whole sweep leaves the chi
    layout unchanged, then measures the next sweep, whose layout must not move
    either, so any trace there is metadata churn, not a layout change.
    """
    sites = _su_u1_pair()
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}
    dls = {c: cpl._build_double_layer_tensor(t) for c, t in sites.items()}

    def sweep(e):
        return _ctm_tensor_sweep_multisite(
            e, dls, CHECKERBOARD_NEIGHBORS, CHI, True, recipe="2x2"
        )[0]

    # Settle: a whole sweep must leave the layout unchanged (input == output),
    # because pieces inside a sweep consume tensors written earlier in it.
    for _ in range(12):
        nxt = sweep(envs)
        settled = cpl._chi_layout(nxt) == cpl._chi_layout(envs)
        envs = nxt
        if settled:
            break
    assert settled, "regime: the layout never settled in 12 sweeps"
    layout_before = cpl._chi_layout(envs)
    traces = []
    inside = [None]
    real_call = ej.call

    def call(fn, *args, **kwargs):
        inside[0] = fn.__name__
        try:
            return real_call(fn, *args, **kwargs)
        finally:
            inside[0] = None

    def listener(name, _dur, **_kw):
        # Only traces inside a jitted piece: the eager code around the cut
        # traces its own small ops every sweep, before and after this change.
        if name == "/jax/core/compile/jaxpr_trace_duration" and inside[0]:
            traces.append(inside[0])

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ej, "call", call)
            envs = sweep(envs)
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert cpl._chi_layout(envs) == layout_before, "regime: layout must be settled"
    assert traces == [], sorted(set(traces))
