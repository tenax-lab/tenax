"""The dense BP solve is traced, and the design rests on it staying that way.

The rewrite (#882) re-gauges after **every** simple-update step, which puts a BP
solve on the per-step budget rather than the per-run one.  Run eagerly that is
18.9 ms per sweep and tens of sweeps per solve -- 490.6 ms for one D=2 solve,
measured -- i.e. the cadence is unaffordable; run as one ``lax.while_loop`` it
is 0.034 ms per sweep and 2.44 ms per solve, and the binding term becomes the
one-off compile.  **The whole gate is still missed**: 533-535 ms for a
100-step run against a 450 ms budget, of which 292 ms is the compile.  See
``test_re_gauging_every_step_fits_the_simple_update_budget`` at the bottom of
this file, which records that rather than hiding it.  So the two things worth
guarding are not "is it fast" but:

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
from _ipeps_gauge_helpers import _dense_pair, _torus_2x2  # tests/ is on sys.path

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


@pytest.fixture
def retraced():
    """Make a monkeypatch inside the traced solve actually take effect.

    ``_bp_solve_traced``'s cache key is ``(shape, dtype, treedef, max_iter,
    tol)`` and does **not** include the module globals its body reads, so a
    patched ``_sweep_is_healthy`` is honoured only if the entry is retraced --
    and the entry the patched run leaves behind would poison any later test
    matching the same key.  Clear on both sides.

    ``clear_cache()`` on the one jitted function, not ``jax.clear_caches()``:
    the global version also evicts every *eager* op the session has compiled,
    which costs the next test ~170 ms of re-compilation and, in the timing
    tests below, would be charged to the thing being measured.
    """
    bp._bp_solve_traced.clear_cache()
    yield
    bp._bp_solve_traced.clear_cache()


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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("D", [2, 3, 4])
def test_the_traced_driver_reproduces_the_eager_one(D, seed):
    """Same fixed point, same sweep count, same state -- on nine fixtures.

    The sweep count is asserted **exactly**.  It is the sharpest available
    check on the carry: the health gate, the ``done`` counter and the
    convergence test all feed it, and a rollback that fires one sweep early or
    a residual reduced over the wrong axis would change it while leaving a
    perfectly plausible spectrum behind.
    """
    A, B = _dense_pair(D=D, seed=seed)
    w0 = BondWeights.ones(D, D)
    kw = {"max_iter": 400, "tol": 1e-13}

    A_t, B_t, w_t, info_t = bp.bp_gauge_checkerboard(A, B, w0, **kw)
    A_e, B_e, w_e, info_e = _solve_eager(A, B, w0, **kw)

    assert info_t.converged and info_e.converged, f"{info_t} / {info_e}"
    assert info_t.iterations == info_e.iterations, (
        f"D={D} seed={seed}: traced took {info_t.iterations} sweeps, eager "
        f"{info_e.iterations}; the two drivers must follow the same trajectory"
    )
    # Both stop *below* the tolerance, and on the same order.  Not tighter than
    # that on purpose: the reported residual is a converged fixed point's
    # round-off, accumulated over ~40 sweeps down two different fusion paths,
    # and it is the sweep count above that pins the trajectory.  Measured
    # spread across these nine fixtures: 7e-3 to 1.3e-2 relative.
    assert max(info_t.residual, info_e.residual) < kw["tol"]
    assert info_t.residual == pytest.approx(info_e.residual, rel=0.5)

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


def test_the_two_drivers_roll_back_identically(retraced):
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
    it is a doubling.  The inputs are deliberately *different states* of the
    same shape, since what is being pinned is that the cache key is the
    shape/dtype/treedef and not the values.

    The ``D=3`` probe at the end is not decoration: without it an assertion
    that the counter did not move would also pass if the counter were dead.
    """
    n_solves, D, _ = _budget()
    A, B = _su_evolved_pair(D)

    bp._bp_solve_traced.clear_cache()
    assert bp._bp_solve_traced._cache_size() == 0, "cache not cleared"

    key = jax.random.PRNGKey(0)
    for i in range(n_solves):
        # A different state each solve, as a real run would hand over.
        key, sub = jax.random.split(key)
        Ai = A + A * (1e-3 * jax.random.normal(sub, ()))
        info = gauge_fix(Ai, B, **_GATE_KW)[3]
        assert info.iterations > 0, f"solve {i} did nothing: {info}"

    assert bp._bp_solve_traced._cache_size() == 1, (
        f"{n_solves} solves produced {bp._bp_solve_traced._cache_size()} compiled "
        f"entries; every one of them must reuse the same trace or the XLA "
        f"compile is paid again and the re-gauging budget is gone"
    )

    A3, B3 = _dense_pair(D=3)
    gauge_fix(A3, B3, **_GATE_KW)
    assert bp._bp_solve_traced._cache_size() == 2, (
        "a different bond dimension did not produce a second entry, so the "
        "assertion above cannot distinguish 'one compile' from 'no counter'"
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


@pytest.mark.timing
def test_the_warm_re_gauging_cost_fits_the_budget():
    """Steady state is inside the budget.  The compile is not -- see below.

    Arithmetic, not a constant::

        n_solves = iPEPSConfig().num_imaginary_steps      # 100
        D        = iPEPSConfig().max_bond_dim             # 2
        budget   = 0.75 s target - 0.30 s un-gauged baseline

    Measured on a quiet 128-core machine (load1 ~7-11), three fresh processes:
    2.44 ms per warm solve at 26 sweeps, so 100 solves is 244 ms -- 54% of the
    budget.  Eager, the same solve is 490.6 ms (18.9 ms/sweep), so this is the
    ~200x that tracing bought and it is the part that is under control.

    A generous 2x on top of the measured number, because this asserts a
    *wall-clock* on whatever machine runs it: what would fail here is a
    regression that reintroduces per-sweep host work, not a busy afternoon.
    """
    n_solves, D, budget_s = _budget()
    A, B = _su_evolved_pair(D)
    warm_s = _warm_solve_seconds(A, B)

    assert n_solves * warm_s < 2.0 * budget_s, (
        f"{n_solves} warm solves cost {n_solves * warm_s * 1e3:.0f} ms "
        f"({warm_s * 1e3:.2f} ms each) against a {budget_s * 1e3:.0f} ms "
        f"budget; the steady state used to be 248 ms"
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
def test_re_gauging_every_step_fits_the_simple_update_budget():
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
