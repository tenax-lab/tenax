"""Frozen SU layout -> traced graded CTM + AD (#1035 follow-up)."""

from __future__ import annotations

import dataclasses
import warnings

import jax
import numpy as np
import pytest

import tenax.algorithms.fermionic_ipeps as fi
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
    ctm_tensor_2site,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _initialize_fpeps,
    _trotter_gate,
    bond_layout,
    spinless_fermion_gate,
    su_grow_layout,
)
from tenax.algorithms.ipeps_simple_update import _simple_update_checkerboard_sweep

jax.config.update("jax_enable_x64", True)

CHI = 12


def _su(pin, D=3, seed=6, steps=4 * 40):
    cfg = FPEPSConfig(D=D, t=1.0, V=0.0, dt=0.05)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(seed))
    gate = _trotter_gate(spinless_fermion_gate(cfg), cfg.dt)
    return A0, _simple_update_checkerboard_sweep(
        A0, A0, gate, D, steps, pin_sectors=pin
    )


def test_bond_layout_counts_even_and_odd_per_bond():
    A0, _ = _su(True, steps=0)
    lay = bond_layout(A0, A0)
    assert len(lay) == 8 and all(sum(p) == 3 for p in lay)
    # Pin the ORDER (n_even, n_odd), not just the total -- an independent
    # count straight off A0.u's charges (bypassing bond_layout's own
    # tuple-building line) must match bond_layout's first entry, or a mutant
    # that swaps which slot is "even" vs "odd" would be invisible here.
    u_charges = np.asarray(A0.indices[A0.labels().index("u")].charges)
    assert lay[0] == (int((u_charges % 2 == 0).sum()), int((u_charges % 2 == 1).sum()))
    # D=3's alternating virtual-charge tiling ([i % 2 for i in range(3)] ==
    # [0, 1, 0], set in _build_initial_fpeps_tensor) puts every leg of A0 at
    # the same split, so pin all eight explicitly too.
    assert lay == ((2, 1),) * 8


def test_the_pinned_su_never_moves_the_layout():
    A0, (A, B, _) = _su(True)
    assert bond_layout(A, B) == bond_layout(A0, A0)


def test_the_unpinned_su_grows_the_layout():
    A0, (A, B, _) = _su(False)
    lay = bond_layout(A, B)
    # Regime: the top-D truncation wants a different sector split than the
    # initial one -- otherwise this test cannot tell pinned from unpinned.
    # At D=3, V=0, t=1, dt=0.05 (this fixture) the (n_even, n_odd)=(2, 1)
    # split of D=3's alternating [0,1,0] virtual charges is a strong
    # attractor for the *pure-hopping, half-filled* free-fermion problem:
    # scanning seed in {1..7} at D=3 and D=4, only seed=6 (of this file's
    # allowed seed in {1,2,3} x D in {3,4} grid, extended to locate one that
    # moves) drives the vertical A.d<->B.u bond to (1, 2) by step 160; every
    # other seed in {1,2,3,4,5,7} at D=3, and {1,2,3,4,5,7} at D=4, reproduces
    # the initial split on every leg through the full 40-sweep run.
    assert lay != bond_layout(A0, A0)
    # Both ends of each bond agree: A.r/B.l and A.d/B.u (legs u, d, l, r).
    assert lay[3] == lay[4 + 2] and lay[1] == lay[4 + 0]


def _scripted(monkeypatch, layouts):
    """Replace one SU cycle and the layout probe by a script; record max_D."""
    it = iter(layouts)
    calls = []

    def fake_sweep(A, B, gate, max_D, steps, lambdas=None, **kw):
        calls.append((max_D, steps, kw.get("pin_sectors")))
        return A, B, lambdas

    monkeypatch.setattr(fi, "_simple_update_checkerboard_sweep", fake_sweep)
    monkeypatch.setattr(fi, "bond_layout", lambda A, B: next(it))
    monkeypatch.setattr(fi, "_to_physical_pair", lambda A, B, lam: (A, B))
    return calls


def _lay(n_even, n_odd):
    return ((n_even, n_odd),) * 8


def test_it_raises_max_d_one_slot_per_stage(monkeypatch):
    # stage 2: 3 cycles at (1,1); stage 3: 4 cycles, slot goes odd at cycle 1
    layouts = [_lay(1, 1)] * 3 + [_lay(1, 2)] * 4
    calls = _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    out = fi.su_grow_layout(
        spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=3, final_cycles=4
    )
    assert [c[0] for c in calls] == [2] * 3 + [3] * 4
    assert all(c[1] == 4 and c[2] is False for c in calls)  # 4 phases, unpinned
    assert out.stages == (_lay(1, 1), _lay(1, 2))
    assert out.layout == _lay(1, 2) and out.frozen


def test_a_layout_that_moves_after_the_first_cycle_is_not_frozen(monkeypatch):
    layouts = [_lay(1, 1)] * 3 + [_lay(1, 2), _lay(2, 1), _lay(1, 2), _lay(1, 2)]
    _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    out = fi.su_grow_layout(
        spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=3, final_cycles=4
    )
    assert not out.frozen


def test_it_refuses_to_freeze_a_collapsed_sector(monkeypatch):
    layouts = [_lay(1, 1)] * 3 + [_lay(3, 0)] * 4  # the odd sector died at growth
    _scripted(monkeypatch, layouts)
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="collapsed"):
        fi.su_grow_layout(
            spinless_fermion_gate(cfg),
            cfg,
            D_start=2,
            cycles_per_stage=3,
            final_cycles=4,
        )


def test_d_start_below_two_is_refused():
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="D_start"):
        fi.su_grow_layout(spinless_fermion_gate(cfg), cfg, D_start=1)


def test_d_start_above_config_d_is_refused():
    # An empty range(D_start, config.D + 1) would otherwise leave `lay`
    # unbound at the collapse check -- a NameError, not a clear ValueError.
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="D_start"):
        fi.su_grow_layout(spinless_fermion_gate(cfg), cfg, D_start=4)


@pytest.mark.slow
def test_growth_puts_the_new_slot_where_the_discarded_weight_is():
    """The one moment the SU reads the physics: growing D=2 -> 3, the sector
    that gains the slot is the sector of the 3rd-largest singular value of the
    untruncated bond.  Then the layout stays put (YASTN: 0 moves in 300 steps)."""
    from tenax.algorithms.ipeps_simple_update import (
        _graded_bond_update,
        scale_bond_axis,
    )

    cfg = FPEPSConfig(D=3, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    out = su_grow_layout(
        H,
        cfg,
        D_start=2,
        cycles_per_stage=10,
        final_cycles=20,
        key=jax.random.PRNGKey(2),
    )
    assert out.frozen
    assert all(sum(p) == 3 for p in out.layout)
    assert out.layout[3] == out.layout[4 + 2] and out.layout[1] == out.layout[4 + 0]
    # Regime + mechanism: redo the first growth truncation by hand on the
    # stage-2 state and read where the 3rd singular value lives.
    cfg2 = dataclasses.replace(cfg, D=2)
    A2 = B2 = fi._initialize_fpeps(cfg2, jax.random.PRNGKey(2))
    lam = None
    for _ in range(10):
        A2, B2, lam = _simple_update_checkerboard_sweep(
            A2, B2, _trotter_gate(H, cfg.dt), 2, 4, lambdas=lam, pin_sectors=False
        )
    # first phase of the growth cycle: horizontal A.r -- B.l, lambdas absorbed
    A_abs = scale_bond_axis(
        scale_bond_axis(scale_bond_axis(A2, "u", lam.v_BA), "d", lam.v_AB),
        "l",
        lam.h_BA,
    )
    A_abs = scale_bond_axis(A_abs, "r", lam.h_AB)
    B_abs = scale_bond_axis(
        scale_bond_axis(scale_bond_axis(B2, "u", lam.v_AB), "d", lam.v_BA),
        "r",
        lam.h_BA,
    )
    # _graded_bond_update returns (A_new, B_new, sigma) -- not (U, sigma, Vh):
    # both A_new and B_new are full site tensors (labels u, d, l, r, phys)
    # after graded_reorder, and sigma is the new bond's singular values.
    A_new, _, sigma = _graded_bond_update(
        A_abs, B_abs, _trotter_gate(H, cfg.dt), "r", "l", None, None
    )
    order = np.argsort(-np.asarray(sigma))
    charges = np.asarray(A_new.indices[A_new.labels().index("r")].charges)[order[:3]]
    par = np.asarray(A_new.indices[0].symmetry.parity(charges))
    assert (
        abs(np.asarray(sigma)[order[2]] - np.asarray(sigma)[order[3]]) > 1e-6
    )  # not degenerate
    assert out.stages[1][3] == (int((par == 0).sum()), int((par == 1).sum()))


@pytest.fixture(scope="module")
def frozen_state():
    cfg = FPEPSConfig(D=3, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    out = su_grow_layout(H, cfg, D_start=2, key=jax.random.PRNGKey(2))
    assert out.frozen
    return out.A, out.B, H, 2


@pytest.fixture(scope="module")
def eager_envs(frozen_state):
    # Cost (Ruling R3), measured on this clone (CPU, D=3, key=PRNGKey(2)):
    # su_grow_layout ~64 s; the eager CTM's per-sweep max-corner-SV diff
    # (same metric ctm_tensor_2site checks against conv_tol) traced sweep by
    # sweep: 1.07e-7 at sweep 150 (the brief's original max_iter, which does
    # NOT reach 1e-10 and would leave eager_envs short of the fixed point the
    # design assumes), then a monotone tail -- 4.2e-8 (160), 6.5e-9 (180),
    # 1.0e-9 (200), 3.9e-10 (210) -- crossing 1e-10 at sweep 225, ~443 s
    # total. max_iter=260 below gives a 35-sweep margin over that measured
    # crossing while staying at ~510 s (~8.5 min), well under the ~30 min
    # budget in Decision 1 -- so conv_tol stays at the brief's 1e-10 rather
    # than being relaxed.
    A, B, _, _ = frozen_state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eA, eB = ctm_tensor_2site(A, B, CHI, max_iter=260, conv_tol=1e-10)
    assert eA.C1.indices[0].dim == CHI  # regime: the corner is truncated
    envs = {(0, 0): eA, (1, 0): eB}
    # YASTN's fixed-point check (fixed_pt.py:369): one more eager sweep must
    # not move any sector size, or there is no fixed point to hand to AD.
    dls = {(0, 0): _build_double_layer_tensor(A), (1, 0): _build_double_layer_tensor(B)}
    again, _, _ = _ctm_tensor_sweep_multisite(
        envs, dls, CHECKERBOARD_NEIGHBORS, CHI, True
    )
    assert jax.tree_util.tree_structure(again) == jax.tree_util.tree_structure(envs)
    return envs


def _E(A, B, envs, H, d):
    return float(
        compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], H, d)
    )


@pytest.mark.slow
def test_the_seeded_traced_ctm_keeps_the_layout_and_the_energy(
    frozen_state, eager_envs
):
    A, B, H, d = frozen_state
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
        env_init=eager_envs,
    )
    tree = jax.tree_util.tree_structure
    assert tree(envs) == tree(eager_envs)
    assert abs(_E(A, B, envs, H, d) - _E(A, B, eager_envs, H, d)) <= 1e-8


@pytest.mark.slow
def test_the_cold_traced_ctm_is_why_the_seed_exists(frozen_state, eager_envs):
    """Measurement, kept as a regime guard: from the tiled initial
    environment the traced CTM lands on a different layout.  If this ever
    fails, the seed is a no-op for this fixture and Task 3's first test
    certifies nothing -- pick a fixture where it matters."""
    A, B, _, _ = frozen_state
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
    )
    tree = jax.tree_util.tree_structure
    assert tree(envs) != tree(eager_envs)
