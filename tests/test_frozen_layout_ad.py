"""Frozen SU layout -> traced graded CTM + AD (#1035 follow-up)."""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_energy_ad as ea
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
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.algorithms.ipeps_simple_update import _simple_update_checkerboard_sweep

jax.config.update("jax_enable_x64", True)

CHI = 12
# Ruling R16 (Task 5): the plumbing/end-to-end tests use their OWN D=2, chi=8
# fixture (``seeded_d2`` below), not ``frozen_state``/``eager_envs`` (D=3,
# chi=12): the latter costs ~10 min to build and its implicit-AD forward does
# not reliably reach a #841 element-wise fixed point (Task 4 rounds 1-5).
CHI_D2 = 8


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


def test_cycles_per_stage_below_one_is_refused():
    # Same class as the D_start guard above: a zero-cycle stage leaves
    # `lay` unbound (first stage) or silently reuses the previous stage's
    # layout (a later stage) -- final-review finding M5.
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="cycles_per_stage"):
        fi.su_grow_layout(
            spinless_fermion_gate(cfg), cfg, D_start=2, cycles_per_stage=0
        )


def test_final_cycles_below_two_is_refused():
    # final_cycles=1 makes `frozen` trivially True: the freeze check
    # compares cycle 0's layout against every later cycle in the final
    # stage, so with only one cycle there is nothing to compare against
    # -- final-review finding M5.
    cfg = FPEPSConfig(D=3)
    with pytest.raises(ValueError, match="final_cycles"):
        fi.su_grow_layout(spinless_fermion_gate(cfg), cfg, D_start=2, final_cycles=1)


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


def _layout(envs):
    """The sector layout of an environment, with basis order quotiented out.

    Controller ruling (#1035 Task 3, debug round 2): the charge ORDER along a
    chi leg is a basis gauge, not a layout -- the eager cut emits the bond in
    descending-SV order (sectors interleaved), the traced SVD in sector-block
    order, and both are the same environment.  The design freezes sector
    SIZES, so the layout is: per leg (label, flow, sorted {charge: count}),
    plus the sorted block keys (a key is a per-leg charge tuple, independent
    of basis order).  All legs are listed; the D^2 legs never move, so this is
    the chi-leg layout plus a constant.
    """
    out = {}
    for c in sorted(envs):
        env = envs[c]
        for name in env._fields:
            t = getattr(env, name)
            legs = tuple(
                (
                    ix.label,
                    int(ix.flow),
                    tuple(
                        (int(q), int(n))
                        for q, n in zip(*np.unique(ix.charges, return_counts=True))
                    ),
                )
                for ix in t.indices
            )
            keys = tuple(sorted(tuple(int(q) for q in k) for k in t.blocks))
            out[(c, name)] = (legs, keys)
    return out


@pytest.mark.slow
def test_the_seeded_traced_ctm_keeps_the_layout_and_the_energy(
    frozen_state, eager_envs
):
    A, B, H, d = frozen_state
    site = {(0, 0): A, (1, 0): B}
    envs, _ = python_loop_ctm_converge(
        site,
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
        env_init=eager_envs,
    )
    # Layout, not raw tree_structure, against the eager seed: basis order along
    # a chi leg is a gauge (see ``_layout``; controller ruling, Task 3 round 2).
    assert _layout(envs) == _layout(eager_envs)
    assert abs(_E(A, B, envs, H, d) - _E(A, B, eager_envs, H, d)) <= 1e-8
    # The property the jit cache needs is traced->traced stability: one more
    # traced sweep must reproduce the FULL tree_structure (basis order
    # included), or every AD step would retrace.
    again, _ = python_loop_ctm_converge(
        site,
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=1,
        conv_tol=1e-10,
        env_init=envs,
    )
    tree = jax.tree_util.tree_structure
    assert tree(again) == tree(envs)


@pytest.mark.slow
def test_the_cold_traced_ctm_is_why_the_seed_exists(frozen_state, eager_envs):
    """Measurement, kept as a regime guard: from the tiled initial
    environment the traced CTM lands on a different sector layout (sizes, not
    basis order -- see ``_layout``).  If this ever fails, the seed is a no-op
    for this fixture and Task 3's first test certifies nothing -- pick a
    fixture where it matters."""
    A, B, _, _ = frozen_state
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        chi=CHI,
        max_iter=50,
        conv_tol=1e-10,
    )
    assert _layout(envs) != _layout(eager_envs)


@pytest.mark.slow
def test_the_graded_energy_gradient_matches_finite_differences_at_a_fixed_environment(
    frozen_state, eager_envs
):
    """R15 (Task 4, round 6): FD-vs-AD certification of the graded energy
    *readout's* backward, at a fixed (constant, un-differentiated) CTM
    environment -- not the full implicit-AD CTM gradient the original Task 4
    brief targeted.

    Rounds 1-5 (see the Task 4 report) established that the full pipeline
    cannot be certified this way on current code: no fixture found across a
    D=3 key and a 16-point D=2 (key, chi) grid has *both* a converged,
    element-wise phase-gauge CTM fixed point (#841) and a full-rank kept
    corner spectrum (round 4/5's ``min_sv_ratio`` floor) at the same time --
    the two failure modes the implicit backward's SVD/fixed-point VJPs are
    sensitive to. Round 5's conclusion was that this is a design-level gap in
    the phase-gauge 2x2 recipe on these fixtures, not something a test-body
    change can route around.

    What *can* be certified without that fixture: the plan's Review Focus 3
    is specifically about a graded sign the forward applies (in
    ``dense_rdm``, ``_ctm_graded.py``) but whose backward the AD path could
    silently drop -- a defect that lives entirely in the energy *readout*
    (``compute_energy_ctm_tensor_2site`` -> ``_rdm2x1_tensor_2site`` /
    ``_rdm1x2_tensor_2site`` -> ``G.contract`` for the graded double-layer
    contractions, then ``G.dense_rdm`` for the bra-leg sign), not in the CTM
    fixed point. Holding ``eager_envs`` CONSTANT (a plain input, never
    recomputed or differentiated) removes both round 1-5 failure modes at
    once: there is no CTM sweep and no SVD anywhere in the differentiated
    path, so neither the #841 stationarity question nor the corner-rank
    floor applies here. This does NOT certify the implicit CTM fixed-point
    backward (the GMRES/Neumann adjoint solve and its own SVD-projector
    VJPs) -- that remains uncertified by an FD test on current fixtures, per
    the round 5 finding.

    Real-fixture check (this file's own module-scope ``frozen_state``/
    ``eager_envs``, D=3, chi=12, key=PRNGKey(2)): seed 0 (below) passed the
    regime and comparison asserts on the first try, confirmed beforehand
    against a bit-identical cached copy of the same fixture (Task 3's
    ``eager_cache.pkl``, same D/chi/key/max_iter/conv_tol -- see the Task 4
    report round 1) -- so no second ``V`` seed was needed.

    Dtype/pairing note (memory: complex cotangents pair UNCONJUGATED): the
    site tensors and the Hamiltonian gate here are real (float64) --
    ``SymmetricTensor.random_normal``'s default dtype -- so ``.real`` after
    ``jnp.vdot(a, b)`` is a no-op and conjugation does not matter; the
    ordinary real inner product is used.
    """
    A, B, H, d = frozen_state

    def f(A_):
        # A_.norm() only -- SymmetricTensor has no __truediv__, so multiply
        # by the reciprocal; mathematically A_ / A_.norm(), as R15 specifies
        # (no +eps: A's norm stays well away from 0 for every A_ this test
        # constructs).  B and the environments are constant closure inputs,
        # never touched by the differentiated argument.
        A_n = A_ * (1.0 / A_.norm())
        return compute_energy_ctm_tensor_2site(
            A_n, B, eager_envs[(0, 0)], eager_envs[(1, 0)], H, d
        )

    rng = np.random.default_rng(0)
    V = jax.tree_util.tree_map(lambda x: jnp.asarray(rng.standard_normal(x.shape)), A)
    g = jax.grad(f)(A)
    slope = float(
        sum(
            jnp.vdot(a, b).real
            for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(V))
        )
    )
    fd = {}
    for h in (1e-3, 3e-4, 1e-4, 3e-5):
        Ap = jax.tree_util.tree_map(lambda x, v: x + h * v, A, V)
        Am = jax.tree_util.tree_map(lambda x, v: x - h * v, A, V)
        fd[h] = float((f(Ap) - f(Am)) / (2 * h))
    vals = np.array(list(fd.values()))
    fd_unc = vals.max() - vals.min()  # h-scan spread (memory: FD needs an h scan)
    assert abs(slope) > 10 * fd_unc, (slope, fd)  # regime: the slope is measurable
    assert abs(slope - np.median(vals)) <= 3 * fd_unc + 1e-7, (slope, fd)


# ------------------------------------------------------------------ #
# Task 5: envs_init on optimize_gs_ad (2-site)                       #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def seeded_d2():
    """Ruling R16: Task 5's own D=2, chi=8, key=4 fixture -- NOT
    ``frozen_state``/``eager_envs`` (D=3, chi=12, ~10 min to build, and whose
    implicit-AD forward does not reliably reach a #841 element-wise fixed
    point per the Task 4 report).  ``key=4`` is the D=2/chi=8 combination
    Task 4 round 4 found where the implicit-AD forward genuinely converges
    (stationarity residual 7.9e-11).

    Cost (measured on this clone, CPU, `taskset -c 0-7`): ``su_grow_layout``
    ~27-28 s (``D_start == config.D`` makes the stage range one value, no
    growth to trace).  The eager CTM's own max-corner-SV-diff criterion
    (Task 4's ``probe_d2_key4.py``) crosses ``conv_tol=1e-10`` between
    ``max_iter=100`` (still short) and ``130`` (converged); ``max_iter=150``
    below gives a 20-sweep margin.  Total module-fixture build (both
    fixtures share it): ~2-3 min.
    """
    cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    out = su_grow_layout(H, cfg, D_start=2, key=jax.random.PRNGKey(4))
    assert out.frozen
    A, B = out.A, out.B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eA, eB = ctm_tensor_2site(A, B, CHI_D2, max_iter=150, conv_tol=1e-10)
    assert eA.C1.indices[0].dim == CHI_D2  # regime: the corner is truncated
    return A, B, H, 2, {(0, 0): eA, (1, 0): eB}


def _ad_config(steps, **ctm):
    return iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        su_init=False,
        gs_implicit_ad=True,
        gs_num_steps=steps,
        gs_verbose=False,
        ctm=CTMConfig(chi=CHI_D2, max_iter=50, conv_tol=1e-9, **ctm),
    )


@pytest.fixture(scope="module")
def light_AB():
    """Cheap (D=2, no SU sweep, no CTM) site tensors for the refusal tests
    that call ``A.norm()``/``B.norm()`` before ever raising, but never run
    any CTM or inspect envs_init's structure.  Building the full
    ``seeded_d2`` fixture (~155 s: a real 28 s SU growth plus a genuine
    ~126 s eager CTM to conv_tol=1e-10) for a test that raises before any
    of that matters would be pure waste (review item 6)."""
    cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.05)
    H = spinless_fermion_gate(cfg)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(0))
    return A0, A0, H


def _fake_envs(chi):
    """A minimal stand-in for ``{(0, 0): CTMTensorEnv, (1, 0): CTMTensorEnv}``
    for refusal tests whose check either fires before envs_init is ever
    inspected, or reads only ``.C1.indices[0].dim`` from it (the chi-mismatch
    check) -- avoids building a real, converged environment for a test that
    never runs any CTM."""
    env = SimpleNamespace(C1=SimpleNamespace(indices=[SimpleNamespace(dim=chi)]))
    return {(0, 0): env, (1, 0): env}


def test_envs_init_is_the_first_forward_seed(light_AB, monkeypatch):
    """R4: the spy must not pay an AD compile -- it records ``env_init`` and
    aborts via a sentinel exception before the (expensive) implicit-AD
    backward is ever traced/compiled.

    Final-review finding I2: uses ``light_AB`` + ``_fake_envs(CHI_D2)`` (no
    SU sweep, no eager CTM), not the ``seeded_d2`` fixture (~155 s: a real
    28 s SU growth plus a genuine ~126 s eager CTM to conv_tol=1e-10) the
    first version of this test paid for a 0.26 s body.  Safe because the
    spy fires strictly before anything reads ``envs_init`` beyond
    ``.C1.indices[0].dim`` -- traced by hand through every hop between
    ``optimize_gs_ad`` and the spy target:
    ``_optimize_gs_ad_tensor_2site``'s ``envs_init`` validation block reads
    only ``.keys()`` and ``.C1.indices[0].dim`` (both of which
    ``_fake_envs``'s ``SimpleNamespace`` supplies) and then stores the dict
    unread into ``_env_cache_2s["envs"]``; ``make_ctm_energy_fn``'s
    ``_ctm_energy_fn`` does ``env_cache.get("envs")`` with no read;
    ``ctm_energy_implicit``'s ``_validate_chi_bump_args`` short-circuits on
    ``bump_enabled=False`` (``_ad_config``'s default) before ever touching
    ``env_init``; ``_ctm_energy_implicit_dispatch`` stores it into
    ``mutables["env_init"]`` with no read; and ``_run_forward`` passes
    ``mutables["env_init"]`` straight into ``_sigma_gauged_ctm_converge``
    -- the spy target -- as the very first thing that would read deeper.
    Confirmed by running this test: it passes against a ``SimpleNamespace``
    that has no attributes beyond ``C1.indices[0].dim``.

    The spy targets ``_ctm_energy_ad._sigma_gauged_ctm_converge``, not
    ``python_loop_ctm_converge`` (the brief's literal target): instrumented
    with a scratch probe (print timestamps on both names), the DEFAULT
    2-site implicit-AD forward (``gs_implicit_ad=True``, default
    ``forward_gauge="phase"``, no ``chi_ramp``) calls
    ``_sigma_gauged_ctm_converge`` directly from ``_run_forward`` -- that
    call IS the first forward the design seeds.  ``python_loop_ctm_converge``
    is reached only via ``chi_ramp`` (unused here) or via the grad-free
    ``_update_env_cache_2s`` warm-start refresh that runs AFTER the gradient
    step -- too late to avoid paying for the backward compile, which
    defeats R4's purpose.  Confirmed eager: ``_run_forward`` is called by
    ``jax.value_and_grad(loss_fn)`` with no enclosing ``jit``, so its
    ``custom_vjp`` forward rule runs with concrete values, not abstract
    tracers -- ``env_init is envs`` holds by identity, not merely by value.
    """
    A, B, H = light_AB
    envs = _fake_envs(CHI_D2)

    class _Seen(Exception):
        pass

    seeds = []

    def _spy(*_a, **kw):
        seeds.append(kw.get("env_init"))
        raise _Seen

    monkeypatch.setattr(ea, "_sigma_gauged_ctm_converge", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(_Seen):
            optimize_gs_ad(H, (A, B), _ad_config(1), envs_init=envs)
    assert seeds and seeds[0] is envs


def test_envs_init_refuses_a_chi_bump(light_AB):
    A, B, H = light_AB
    with pytest.raises(ValueError, match="envs_init.*chi_auto_bump"):
        optimize_gs_ad(
            H,
            (A, B),
            _ad_config(1, chi_auto_bump=True),
            envs_init=_fake_envs(CHI_D2),
        )


def test_envs_init_refuses_a_mismatched_chi(light_AB):
    A, B, H = light_AB
    cfg = _ad_config(1)
    cfg = dataclasses.replace(cfg, ctm=dataclasses.replace(cfg.ctm, chi=CHI_D2 + 2))
    with pytest.raises(ValueError, match="envs_init.*chi"):
        optimize_gs_ad(H, (A, B), cfg, envs_init=_fake_envs(CHI_D2))


def test_envs_init_refuses_a_ctmrg_heuristic_increase_chi(light_AB):
    """Final-review finding I1: the in-CTM bump (#512's recommended,
    non-deprecated chi-growth mode) grows chi *inside* the forward, which
    defeats the frozen layout and re-opens #518 (the stall-rollback
    restore, ruling R17, assumes chi is fixed for the whole run).  Needs
    ``chi_max`` set -- ``CTMConfig.__post_init__`` refuses
    ``ctmrg_heuristic_increase_chi=True`` without one -- so the config
    itself constructs cleanly and the ``envs_init``-specific refusal below
    is the one that actually fires."""
    A, B, H = light_AB
    with pytest.raises(ValueError, match="envs_init.*ctmrg_heuristic_increase_chi"):
        optimize_gs_ad(
            H,
            (A, B),
            _ad_config(1, ctmrg_heuristic_increase_chi=True, chi_max=CHI_D2 + 2),
            envs_init=_fake_envs(CHI_D2),
        )


def test_envs_init_refuses_a_chi_ramp(light_AB):
    """Final-review finding I1: a ``chi_ramp`` schedule changes chi between
    stages, the same class of hazard ``chi_auto_bump`` is already refused
    for.  ``CTMConfig`` only warns (``DeprecationWarning``, filtered
    globally in ``pyproject.toml``) on a bare ``chi_ramp``, so this reaches
    the ``envs_init``-specific refusal rather than failing at construction."""
    A, B, H = light_AB
    with pytest.raises(ValueError, match="envs_init.*chi_ramp"):
        optimize_gs_ad(
            H,
            (A, B),
            _ad_config(1, chi_ramp=[(CHI_D2, 5)]),
            envs_init=_fake_envs(CHI_D2),
        )


def test_envs_init_refuses_a_non_2site_unit_cell():
    """The refusals the ruling lists: bump, mismatch, non-2site, split-CTM,
    root-implicit, and (fix round 2, I1/M8) ctmrg_heuristic_increase_chi,
    chi_ramp, and a key mismatch.  All raise ValueError before any CTM/AD
    work.  This one and the root-implicit one below fire in
    ``optimize_gs_ad`` itself, before ``A_init``/``hamiltonian_gate`` are
    ever touched -- so they need no real tensors at all, not even the
    cheap ``light_AB`` stand-in."""
    cfg = dataclasses.replace(_ad_config(1), unit_cell="1x1")
    with pytest.raises(ValueError, match="envs_init.*2site"):
        optimize_gs_ad(None, None, cfg, envs_init=_fake_envs(CHI_D2))


def test_envs_init_refuses_the_split_ctm(light_AB):
    A, B, H = light_AB
    cfg = _ad_config(1, fuse_virtual_legs=False)
    with pytest.raises(ValueError, match="envs_init.*split"):
        optimize_gs_ad(H, (A, B), cfg, envs_init=_fake_envs(CHI_D2))


def test_envs_init_refuses_the_root_implicit_path():
    cfg = dataclasses.replace(
        _ad_config(1),
        ctm=dataclasses.replace(_ad_config(1).ctm, ctm_ad_mode="root_implicit"),
    )
    with pytest.raises(ValueError, match="envs_init.*root.implicit"):
        optimize_gs_ad(None, None, cfg, envs_init=_fake_envs(CHI_D2))


def test_envs_init_refuses_a_key_mismatch(light_AB):
    """Final-review finding M8: ``envs_init``'s keys must equal the 2-site
    cell's, ``{(0, 0), (1, 0)}`` (``CHECKERBOARD_NEIGHBORS.keys()`` --
    what ``make_ctm_energy_fn``/``_energy_fn_2site`` actually index by).  A
    wrong-keyed dict used to fail deep inside the CTM instead of at this
    entry point."""
    A, B, H = light_AB
    bad_envs = {(0, 0): _fake_envs(CHI_D2)[(0, 0)]}  # missing (1, 0)
    with pytest.raises(ValueError, match="envs_init.*keys"):
        optimize_gs_ad(H, (A, B), _ad_config(1), envs_init=bad_envs)


@pytest.mark.slow
def test_frozen_layout_ad_lowers_the_energy_and_keeps_the_layouts(
    seeded_d2, monkeypatch
):
    """R16/R10, fix round 1 (Task 5 review, ruling R17): 5 AD steps (the
    brief's 10, halved per the ruling).

    Deviates from R10's literal ``_layout(envs1) == _layout(seed)`` on the
    optimizer's RETURN value for a reason this task's own probing surfaced
    (scratch/probe_final_layout.py): ``optimize_gs_ad``'s 2-site return is
    always a fresh, COLD CTM re-evaluation on the final tensors
    (``_eval_fresh_2site`` calls ``python_loop_ctm_converge`` with
    ``env_init=None`` unconditionally -- issue #899, "the seed was the
    line-search-reverted cache... evaluated cold for the same reason").
    Measured directly: even after a single AD step, this fixture's seed
    ((0,0).C1 chi split {0:4,1:4} on one leg, {0:5,1:3} on the other) and the
    cold-recomputed return ({0:6,1:2} on both) differ -- a genuine,
    physics-driven difference in a from-scratch CTM run, not a bug in
    ``envs_init``. Comparing ``_layout`` (or raw ``tree_structure``) against
    that return value would therefore fail regardless of whether
    ``envs_init`` seeding works, so it is not evidence about the property
    #1035 actually claims.

    That property -- the traced CTM keeps the χ-sector layout it is seeded
    with -- lives entirely in the environment fed to the real per-step
    differentiated forward, ``_ctm_energy_ad._sigma_gauged_ctm_converge``
    (confirmed the actual first-forward call site by the spy tests above,
    and by direct instrumentation: scratch/probe_spy_order.py). So this test
    spies on it non-invasively (calls straight through) and records every
    ``env_init`` it was ever seeded with, across all 5 steps including any
    interior line-search/HZ-probe re-evaluations.

    Round-1 history (a first implementation of this test, superseded here):
    an early version tolerated one ``env_init=None`` cold call (an HZ
    line-search probe failed to converge and triggered the L-BFGS
    line-search-stall reset -- an HZ ``phi=4 dphi=0 alpha=1.7
    converged=False`` on step 4-5, which clears ``_env_cache_2s`` via
    ``_drop_env_cache_for_reset``, *not* the ``CTMRGGradientError``/
    grad-spike-guard reset paths, neither of which fired) and accepted that
    every later call landed on a new, self-consistent-with-itself layout.
    The review adjudicated this as a defect, not a caveat: with
    ``envs_init`` set, `chi_auto_bump`/`gs_chi_schedule_steps` are already
    refused, so the reset's own justification (issue #518: a reactive/
    scheduled χ bump can leave the cached env at a stale χ) cannot apply,
    and clearing the seed anyway was pure loss of the feature's whole
    point. Correction to that first version's claim: the stall-reset branch
    IS gated by ``gs_stall_recovery`` -- ``_normalize_stall_recovery``
    resolves the unset default to ``"reset"`` for any non-1x1 unit cell
    (``ipeps_optimize.py`` ``_normalize_stall_recovery``, called with
    ``unit_cell="2site"``), so it is *on by default* for every 2-site
    ``envs_init`` caller, not an edge case.

    Fixed (ruling R17): ``_optimize_gs_ad_tensor_2site`` now restores
    ``best_env_cache_2s or {"envs": envs_init}`` after every
    ``_drop_env_cache_for_reset`` when ``envs_init`` is set (the
    ``_reset_env_cache_2s`` helper, used at all three rollback sites: the
    ``CTMRGGradientError`` reset, the grad-spike guard, and the L-BFGS
    line-search-stall reset). So this test is now strict: no call is ever
    cold, and every seeded call's layout equals the original seed's --
    confirmed to still exercise the L-BFGS stall path on this fixture (see
    the report's Fix round 1 section for whether it fired and how many
    calls were recorded).
    """
    A, B, H, d, envs = seeded_d2
    E0 = _E(A, B, envs, H, d)
    real_sigma = ea._sigma_gauged_ctm_converge
    seen_layouts = []

    def _spy(*a, **kw):
        env_init = kw.get("env_init")
        assert env_init is not None, "a forward ran cold: the reset lost the seed"
        seen_layouts.append(_layout(env_init))
        return real_sigma(*a, **kw)

    monkeypatch.setattr(ea, "_sigma_gauged_ctm_converge", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        (A1, B1), (_env_A1, _env_B1), E1 = optimize_gs_ad(
            H, (A, B), _ad_config(5), envs_init=envs
        )
    seed_layout = _layout(envs)
    assert seen_layouts and all(lay == seed_layout for lay in seen_layouts)
    tree = jax.tree_util.tree_structure
    assert tree((A1, B1)) == tree((A, B))  # site layout frozen
    assert E1 <= E0 + 1e-6, (E0, E1)
    assert E1 >= -8 / np.pi**2 - 1e-3  # V=0 free-fermion bound, one-sided only
