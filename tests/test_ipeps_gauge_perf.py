"""The dense BP solve is traced, and the design rests on it staying that way.

The rewrite (#882) re-gauges after **every** simple-update step, which puts a BP
solve on the per-step budget rather than the per-run one.  Run eagerly that is
18.9 ms per sweep and tens of sweeps per solve -- 490.6 ms for one D=2 solve,
measured -- i.e. the cadence is unaffordable; run as one ``lax.while_loop`` it
is 0.034 ms per sweep and 2.44 ms per solve, and the binding term becomes the
one-off compile.  **The whole gate is still missed** as this file stands:
533-535 ms for a 100-step run against a 450 ms budget, of which 292 ms is the
compile.  See ``test_re_gauging_every_step_fits_the_simple_update_budget`` at
the bottom of this file, which records that rather than hiding it -- and #882
Task 7b, which closes it by tracing ``gauge_fix`` end to end.  So the two
things worth guarding here are not "is it fast" but:

* **tracing did not move the answer** -- checked against the eager driver on the
  same input, through a *gauge-invariant* witness (see below);
* **every solve hits the same compiled entry** -- a solve that re-traces costs
  ~0.3 s and blows the whole run's budget on its own, while still producing the
  right answer.  Wall-clock alone cannot see that: it passes on a fast machine
  and fails mysteriously on a slow one.

**Why the state comparison is gauge-invariant and not elementwise.**  At
``D >= 3`` the returned ``Gamma`` carries a genuine +-1 sign freedom, inherited
from the eigh/SVD phase ambiguity inside ``_sqrt_pinv``.  Measured here at
``D=3 seed=1``: ``|A_eager - A_traced| = 2.000e+00`` while
``||A_eager| - |A_traced|| = 1.0e-13``, with the elementwise ratio taking
exactly ``{-1, +1}``; the bond weights agree to 6.9e-15 and the torus witness to
1.1e-14.  It is not a tracing artefact -- an unjitted rewrite that differs only
in the residual reduction shows the same flip -- so an elementwise ``Gamma``
assertion would be flaky by construction.  ``_torus_2x2`` closes every bond, so
a gauge that fails to cancel shows up and a sign that does cancel does not.
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_gauge_helpers import (  # tests/ is on sys.path
    _dense_pair,
    _torus_2x2,
    retraced,  # noqa: F401  -- a fixture; pytest reads it off this namespace
)

import tenax.algorithms.ipeps_bp_gauge as bp
from tenax.algorithms.ipeps import (
    _make_trotter_gate_tensor,
    heisenberg_gate,
    sublattice_rotate_gate,
)
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.algorithms.ipeps_gauge import absorb_weights, gauge_fix
from tenax.algorithms.ipeps_simple_update import (
    BondWeights,
    _simple_update_checkerboard_sweep,
)

#: Same witness tolerance ``test_ipeps_bp_gauge.py`` uses, and for the same
#: reason: an exact gauge is exact, so anything above f64 noise is a defect.
GAUGE_TOL = 1e-13


def _unit(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    n = float(np.linalg.norm(a))
    return a / n if n > 0.0 else a


def _solve_eager(A, B, w, **kw):
    """The same solve, forced down the Python-loop driver.

    ``_use_traced_loop`` is the dispatch itself, not a test hook bolted on: it
    is what sends ``SymmetricTensor`` to the eager loop.  Overriding it is the
    only way to run *both* drivers on the same dense input, which is what makes
    this a comparison rather than two unrelated numbers.
    """
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bp, "_use_traced_loop", lambda A, B: False)
        return bp.bp_gauge_checkerboard(A, B, w, **kw)


#: The nine random draws, plus the one input where the two drivers genuinely
#: differ.  ``_dense_pair``'s flows already match the convention ``_gauge_bond``
#: stamps, so on all nine the structure restoration is a no-op and the arm of
#: the code that #882's flow defect lives in is never compared across drivers.
#: The simple-update-evolved pair -- the one Phase 2 actually hands over -- has
#: four of its five flows inverted, so it is the only fixture here that
#: exercises it.  ``"su"`` is spelled as a seed rather than a second test so the
#: assertions below cover it verbatim.
_PARITY_FIXTURES = [(D, seed) for D in (2, 3, 4) for seed in (0, 1, 2)]
_PARITY_FIXTURES.append((2, "su"))


@pytest.mark.parametrize("D,seed", _PARITY_FIXTURES)
def test_the_traced_driver_reproduces_the_eager_one(D, seed):
    """Same fixed point, same sweep count, same state -- on ten fixtures.

    The sweep count is asserted **exactly**.  It is the sharpest available
    check on the carry: the health gate, the ``done`` counter and the
    convergence test all feed it, and a rollback that fires one sweep early or
    a residual reduced over the wrong axis would change it while leaving a
    perfectly plausible spectrum behind.
    """
    A, B = _su_evolved_pair(D) if seed == "su" else _dense_pair(D=D, seed=seed)
    w0 = BondWeights.ones(D, D)
    kw = {"max_iter": 400, "tol": 1e-13}

    A_t, B_t, w_t, info_t = bp.bp_gauge_checkerboard(A, B, w0, **kw)
    A_e, B_e, w_e, info_e = _solve_eager(A, B, w0, **kw)

    assert info_t.converged and info_e.converged, f"{info_t} / {info_e}"
    assert info_t.iterations == info_e.iterations, (
        f"D={D} seed={seed}: traced took {info_t.iterations} sweeps, eager "
        f"{info_e.iterations}; the two drivers must follow the same trajectory"
    )
    # Both stop *below* the tolerance, and agree to within a few percent.  Not
    # tighter than that on purpose: the reported residual is a converged fixed
    # point's round-off, accumulated over tens of sweeps down two different
    # fusion paths.  Measured spread across these fixtures: 7e-3 to 1.3e-2
    # relative, so 0.1 is ~8x headroom.  The sweep count above is what pins the
    # trajectory; this only catches a residual reduced over the wrong axis.
    assert max(info_t.residual, info_e.residual) < kw["tol"]
    assert info_t.residual == pytest.approx(info_e.residual, rel=0.1)

    for name, a, b in zip(w_t._fields, w_t, w_e, strict=True):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert d < 1e-12, f"D={D} seed={seed}: {name} moved {d:.3e} under trace"

    # 1e-12 rather than ``GAUGE_TOL`` here: this compares two round-off paths
    # against each other, not a gauge against exactness.  Measured max 3.2e-14.
    rel = float(
        np.linalg.norm(
            _unit(_torus_2x2(A_t, B_t, w_t)) - _unit(_torus_2x2(A_e, B_e, w_e))
        )
    )
    assert rel < 1e-12, (
        f"D={D} seed={seed}: the traced driver landed on a different state "
        f"({rel:.3e}); tracing must not change the physics"
    )
    # ... and it is still a gauge of the *input*, which is the guarantee the
    # module actually sells.  Comparing only against the eager run would pass
    # if both drivers were wrong in the same way.
    moved = float(
        np.linalg.norm(_unit(_torus_2x2(A_t, B_t, w_t)) - _unit(_torus_2x2(A, B, w0)))
    )
    assert moved < GAUGE_TOL, (
        f"D={D} seed={seed}: the solve moved the state {moved:.3e}"
    )


# ``usefixtures``, not an argument: see the note in ``test_ipeps_bp_gauge.py``.
@pytest.mark.usefixtures("retraced")
def test_the_two_drivers_roll_back_identically():
    """The rollback is the one path the two drivers implement *differently*.

    Eager keeps the previous iterate by never overwriting it; traced computes
    the candidate unconditionally and then rejects it with
    ``where(healthy, candidate, carry_in)``.  ``tests/test_ipeps_bp_gauge.py``
    checks each driver's rollback against the invariance witness, but on
    *different* pair kinds -- dense goes traced, symmetric goes eager -- so
    nothing there compares the two rewrites on one state.  This does.

    Injected at ``_sweep_is_healthy`` and keyed on the sweep index, because a
    host-side call counter fires once inside a ``while_loop`` body and would
    silently never reach the third sweep.  ``retraced`` is what makes the patch
    visible to the jitted driver at all.
    """
    D = 3
    A, B = _dense_pair(D=D)
    w0 = BondWeights.ones(D, D)
    kw = {"max_iter": 50, "tol": 0.0}
    dies_at_sweep_2 = lambda gam, weights, sweep: sweep < 2  # noqa: E731

    with pytest.MonkeyPatch.context() as m:
        m.setattr(bp, "_sweep_is_healthy", dies_at_sweep_2)
        A_t, B_t, w_t, info_t = bp.bp_gauge_checkerboard(A, B, w0, **kw)
        A_e, B_e, w_e, info_e = _solve_eager(A, B, w0, **kw)

    assert info_t == info_e == (2, float("inf"), False), f"{info_t} / {info_e}"
    assert float(A_t.norm()) > 0.0 and float(B_t.norm()) > 0.0

    rel = float(
        np.linalg.norm(
            _unit(_torus_2x2(A_t, B_t, w_t)) - _unit(_torus_2x2(A_e, B_e, w_e))
        )
    )
    assert rel < GAUGE_TOL, (
        f"the two rollbacks returned different states ({rel:.3e}); the "
        f"where-select must reject the candidate, not blend it"
    )
    # And the rejected iterate is still an exact gauge of the input -- the
    # rollback hands back a usable state, not the corpse.
    moved = float(
        np.linalg.norm(_unit(_torus_2x2(A_t, B_t, w_t)) - _unit(_torus_2x2(A, B, w0)))
    )
    assert moved < GAUGE_TOL, f"the rollback moved the state by {moved:.3e}"


def test_the_reported_info_is_still_python_scalars():
    """``BPGaugeInfo`` is ``(int, float, bool)``, whatever the driver.

    Leaking 0-d arrays here fails in two different ways and only one of them is
    visible: ``assert info.converged`` is satisfied by *any* 0-d array, so a
    ``converged=array(False)`` would pass silently, while
    ``info.residual == float("inf")`` fails loudly
    (``tests/test_ipeps_bp_gauge.py``).
    """
    A, B = _dense_pair(D=2)
    _, _, _, info = bp.bp_gauge_checkerboard(A, B, BondWeights.ones(2, 2), max_iter=20)
    assert type(info.iterations) is int
    assert type(info.residual) is float
    assert type(info.converged) is bool


# --- the compile-cache gate ------------------------------------------------
#
# The budget below is derived from ``iPEPSConfig``'s own defaults rather than
# hard-coded, so that a change to either number is visible here.

#: Wall-clock target for a default simple-update run, and the measured cost of
#: that run without any re-gauging.  The difference is what re-gauging every
#: step is allowed to cost in total.
_SU_TARGET_S = 0.75
_SU_BASELINE_S = 0.30

#: What a solve is gated at.  ``tol`` is ``bp_gauge_checkerboard``'s own default
#: rather than ``gauge_fix``'s looser ``1e-6``: the loose tolerance stops in
#: fewer sweeps, so gating there would let the budget be met by asking for less.
_GATE_KW = {"tol": 1e-12, "max_iter": 100}


def _budget() -> tuple[int, int, float]:
    """``(n_solves, D, seconds)`` the re-gauging cadence has to fit inside."""
    cfg = iPEPSConfig()
    # One BP solve per simple-update step: the step evolves all four bonds and
    # the gauge is re-derived once afterwards.
    return cfg.num_imaginary_steps, cfg.max_bond_dim, _SU_TARGET_S - _SU_BASELINE_S


def _su_evolved_pair(D: int, phases: int = 40):
    """A *simple-update-evolved* pair, in the absorbed form ``gauge_fix`` takes.

    Sweep counts are a property of the state, and the states this will actually
    be handed have decaying bond spectra rather than a random pair's flat ones.
    A fixture that solves in 30 sweeps proves nothing about a state that runs to
    ``max_iter``.

    Absorbed rather than Vidal because that is the boundary the rewrite draws
    (#882 §3): the simple update holds no lambdas, so what reaches
    :func:`gauge_fix` is the pair with every weight already split into its two
    ends.
    """
    A, B = _dense_pair(D=D)
    gate = _make_trotter_gate_tensor(
        sublattice_rotate_gate(heisenberg_gate()), 0.05, site_tensor=A
    )
    A, B, stored = _simple_update_checkerboard_sweep(A, B, gate, D, phases)
    return absorb_weights(A, B, stored)


def test_every_solve_in_a_run_hits_one_compiled_entry():
    """100 solves, one trace.  This is the failure mode wall-clock cannot see.

    Compile is the binding term in the budget below -- 292 ms against a 450 ms
    budget for all hundred solves -- so a second compilation is not a slowdown,
    it is a doubling.

    The inputs are **independently built pairs**, cycled, not one pair with a
    scalar wobble on it: a rescaled copy of ``A`` is the same state once
    ``_rescale`` has had it, and -- more to the point -- it reuses ``A``'s very
    ``TensorIndex`` *objects*, so it could not distinguish "the cache key is the
    shape/dtype/treedef" from "the cache key happens to hold those objects".
    Freshly built pairs carry fresh ``TensorIndex`` instances, and
    ``__hash__``/``__eq__`` are value-based (``core/index.py``), which is the
    property being pinned.

    The two probes at the end are not decoration: without them an assertion that
    the counter did not move would also pass if the counter were dead.  They
    also pin the two things that *do* split the key, and the second one is a
    live constraint on Phase 2 rather than a curiosity.
    """
    n_solves, D, _ = _budget()
    pairs = [_dense_pair(D=D, seed=k) for k in range(5)]

    bp._bp_solve_traced.clear_cache()
    assert bp._bp_solve_traced._cache_size() == 0, "cache not cleared"

    for i in range(n_solves):
        Ai, Bi = pairs[i % len(pairs)]
        info = gauge_fix(Ai, Bi, **_GATE_KW)[3]
        assert info.iterations > 0, f"solve {i} did nothing: {info}"

    assert bp._bp_solve_traced._cache_size() == 1, (
        f"{n_solves} solves produced {bp._bp_solve_traced._cache_size()} compiled "
        f"entries; every one of them must reuse the same trace or the XLA "
        f"compile is paid again and the re-gauging budget is gone"
    )

    # A different bond dimension is a different key -- obvious, and it is what
    # makes the assertion above non-vacuous.
    A3, B3 = _dense_pair(D=3)
    gauge_fix(A3, B3, **_GATE_KW)
    assert bp._bp_solve_traced._cache_size() == 2, (
        "a different bond dimension did not produce a second entry, so the "
        "assertion above cannot distinguish 'one compile' from 'no counter'"
    )

    # And so is a different *flow convention* at the same D, which is much less
    # obvious and is a constraint on the caller.  ``TensorIndex`` is pytree aux
    # data, so the pair's flows are part of the carry treedef; a
    # simple-update-evolved pair has its four virtual flows inverted relative to
    # ``_wrap_as_dense_tensor``, and mixing the two conventions in one run costs
    # a second 292 ms compile.  A real run hands over one convention throughout
    # -- which is exactly what ``_restore_caller_structure`` now guarantees, by
    # handing each caller its own metadata back instead of stamping this
    # module's onto it.  This assertion is what would notice if that stopped
    # being true.
    A_su, B_su = _su_evolved_pair(D)
    gauge_fix(A_su, B_su, **_GATE_KW)
    assert bp._bp_solve_traced._cache_size() == 3, (
        "a pair with inverted virtual flows reused an existing entry; the flows "
        "are aux data and must be part of the key"
    )


def test_the_solve_converges_on_an_su_evolved_state_well_inside_max_iter():
    """Sweep count on a *real* state, which is what sets the per-solve cost.

    The warm budget allows roughly 100 sweeps per solve at ``D=2``; a state that
    runs to ``max_iter`` costs several times that and blows the budget on its
    own.  Asserting convergence *and* a sweep-count ceiling keeps the timing
    gate below from being satisfied by a solve that simply gave up.
    """
    _, D, _ = _budget()
    A, B = _su_evolved_pair(D)
    _, _, _, info = gauge_fix(A, B, tol=_GATE_KW["tol"], max_iter=400)
    assert info.converged, f"BP did not converge on an SU-evolved state: {info}"
    assert info.iterations <= _GATE_KW["max_iter"], (
        f"an SU-evolved D={D} state needed {info.iterations} sweeps against a "
        f"cap of {_GATE_KW['max_iter']}; the re-gauging budget assumes it "
        f"converges well inside that"
    )


def _warm_solve_seconds(A, B, repeats: int = 20) -> float:
    """Median wall-clock of one warm ``gauge_fix``, after the compile is paid.

    Median rather than minimum: the minimum reports the machine's best moment
    rather than its typical one, which is exactly the flattery a budget test
    should not accept.
    """
    gauge_fix(A, B, **_GATE_KW)  # pay the compile
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = gauge_fix(A, B, **_GATE_KW)
        jax.block_until_ready(jax.tree_util.tree_leaves(r))
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


# NOTE: the ``timing`` marker is currently **decorative**.  It is registered in
# ``pyproject.toml``, and that file's comment says these tests "can be deselected
# with -m 'not timing'" -- but that string appears nowhere else in the repo.
# ``.github/workflows/ci.yml`` runs the non-core bucket as
# ``-m "not core and not slow"``, so both tests below execute on a 2-core GitHub
# runner in a bucket this project already tracks as chronically red.  Wiring the
# deselection into the workflow needs a token with the ``workflow`` scope, which
# this branch does not have.  So the assertion below is written to be
# *machine-independent* instead of merely generous: it compares the traced solve
# against the eager one **measured in the same process**, so a runner that is 20x
# slower slows both arms and the ratio survives.  If you later add
# ``-m "not timing"`` to ci.yml, delete this note, not the ratio.


@pytest.mark.timing
def test_the_warm_solve_is_orders_faster_than_the_eager_one(record_property):
    """Steady state is inside the budget.  The compile is not -- see below.

    Arithmetic, not a constant::

        n_solves = iPEPSConfig().num_imaginary_steps      # 100
        D        = iPEPSConfig().max_bond_dim             # 2
        budget   = 0.75 s target - 0.30 s un-gauged baseline

    Measured on a quiet 128-core machine (load1 ~6-11), three fresh processes:
    2.44 ms per warm solve at 26 sweeps, so 100 solves is 244 ms -- 54% of the
    budget -- against 490.6 ms per eager solve (18.9 ms/sweep).  That is the
    ~200x tracing bought, and it is the part that is under control.

    Asserted as a **ratio against the eager driver in this same process**, not
    as an absolute millisecond count.  The absolute number is what the budget is
    written in, but it is also the one thing a CI runner cannot reproduce; the
    ratio is a property of the code.  20x is a tenth of the measured 200x, so
    what fails here is a regression that puts per-sweep host work back into the
    loop, not a busy afternoon.  The absolute numbers are recorded either way.
    """
    n_solves, D, budget_s = _budget()
    A, B = _su_evolved_pair(D)
    w0 = BondWeights.ones(D, D)

    warm_s = _warm_solve_seconds(A, B)
    t0 = time.perf_counter()
    r = _solve_eager(A, B, w0, **_GATE_KW)
    jax.block_until_ready(jax.tree_util.tree_leaves(r))
    eager_s = time.perf_counter() - t0

    record_property("warm_solve_ms", round(warm_s * 1e3, 3))
    record_property("eager_solve_ms", round(eager_s * 1e3, 1))
    record_property("hundred_warm_solves_ms", round(n_solves * warm_s * 1e3))
    record_property("budget_ms", round(budget_s * 1e3))

    assert eager_s / warm_s > 20.0, (
        f"the traced solve is only {eager_s / warm_s:.1f}x the eager one "
        f"({warm_s * 1e3:.2f} ms vs {eager_s * 1e3:.0f} ms); it was ~200x, and "
        f"{n_solves} warm solves are budgeted at {budget_s * 1e3:.0f} ms "
        f"(measured 244 ms).  Something is dispatching per sweep again."
    )


@pytest.mark.timing
@pytest.mark.xfail(
    strict=False,
    reason=(
        "MEASURED AT 533-535 ms AGAINST A 450 ms BUDGET, and 707-724 ms if the "
        "process is cold (quiet machine, three fresh processes each, load1 "
        "~7-11 of 128).  Not a flaky threshold and not relaxed: the compile "
        "alone is 292 ms, 65% of the budget, and it is flat in D.  Closing it "
        "needs a decision outside this module -- either jit gauge_fix end to "
        "end (its eager boundary is 1.5 of the 2.44 ms warm solve; an "
        "all-traced equivalent measures 0.93 ms, taking 100 solves from 244 ms "
        "to 93 ms) or revisit the every-step re-gauging cadence, which is a "
        "design change and the user's call (#882 §2).  See task-7-report.md."
    ),
)
def test_re_gauging_every_step_fits_the_simple_update_budget(record_property):
    """The gate, whole: one compile plus ``num_imaginary_steps`` solves.

    Kept as a live measurement rather than deleted or re-pointed at a number
    that passes, because the target is the thing under review and the
    measurement is what the review needs.  ``xfail`` non-strict, so a machine
    or a change that closes it reports ``XPASS`` instead of turning green
    silently.

    The compile is flat in ``D`` -- 214 ms of XLA at D=2, 216 at D=3, 224 at
    D=4, on a sweep body of 324 jaxpr equations -- so it is structure-bound
    rather than array-size-bound, unlike #633's CTM finding.  Lowering
    ``jax_persistent_cache_min_compile_time_secs`` below it (it defaults to 1 s,
    so this compile is never stored) was tried and moved nothing: 711/721/710 ms
    on the cold-process measurement.

    **What is charged, and why it is the generous accounting.**  Only
    ``_bp_solve_traced``'s own cache is evicted, so the first solve here pays
    the trace (84 ms) and the XLA compile (214 ms) and nothing else.
    ``jax.clear_caches()`` would also evict the *eager* boundary's little
    compiles, which a real simple-update run has already paid for; charging
    those puts the same measurement at 707-724 ms instead of 533.  Not
    evicting anything is the trap this test fell into first: earlier tests in
    this file share the ``(D=2, max_iter=100, tol=1e-12)`` key, so the gate
    silently measured a warm solve and reported XPASS at ~250 ms.
    """
    n_solves, D, budget_s = _budget()
    A, B = _su_evolved_pair(D)

    gauge_fix(A, B, **_GATE_KW)  # warm the eager boundary, as a real run has
    bp._bp_solve_traced.clear_cache()
    assert bp._bp_solve_traced._cache_size() == 0, (
        "the traced solve is still cached, so the line below would time a warm "
        "call and report a compile that never happened"
    )

    t0 = time.perf_counter()
    first = gauge_fix(A, B, **_GATE_KW)
    jax.block_until_ready(jax.tree_util.tree_leaves(first))
    compile_s = time.perf_counter() - t0

    warm_s = _warm_solve_seconds(A, B)
    total = compile_s + (n_solves - 1) * warm_s

    # Recorded and printed unconditionally.  On XFAIL the number is in the
    # assertion message, but on XPASS pytest reports the bare word and the good
    # news would arrive with no evidence for it -- and the whole point of
    # keeping this test alive is to learn what the number became.
    record_property("compile_ms", round(compile_s * 1e3, 1))
    record_property("warm_solve_ms", round(warm_s * 1e3, 3))
    record_property("total_ms", round(total * 1e3))
    record_property("budget_ms", round(budget_s * 1e3))
    print(
        f"\n[#882 gate] {n_solves} solves at D={D}: {total * 1e3:.0f} ms "
        f"(compile {compile_s * 1e3:.0f} + {n_solves - 1} x "
        f"{warm_s * 1e3:.2f} ms) vs {budget_s * 1e3:.0f} ms budget"
    )

    assert total < budget_s, (
        f"re-gauging {n_solves} steps at D={D} costs {total * 1e3:.0f} ms "
        f"(first solve incl. compile {compile_s * 1e3:.0f} ms + "
        f"{n_solves - 1} x {warm_s * 1e3:.2f} ms) against a "
        f"{budget_s * 1e3:.0f} ms budget"
    )


def test_the_eager_driver_is_still_reachable_and_agrees():
    """``SymmetricTensor`` has no traced path, so the Python loop is live code.

    Dispatch is on the tensor type, and the symmetric arm is the one that keeps
    the eager loop from rotting into something nothing executes.
    """
    from _ipeps_gauge_helpers import _symmetric_pair

    A, B = _symmetric_pair()
    assert not bp._use_traced_loop(A, B)
    _, _, w, info = bp.bp_gauge_checkerboard(
        A, B, BondWeights.ones(3, 3), max_iter=400, tol=1e-13
    )
    assert info.converged, f"the eager driver stopped converging: {info}"
    for name in w._fields:
        assert float(jnp.max(getattr(w, name))) > 0.0, f"{name} came back zero"
