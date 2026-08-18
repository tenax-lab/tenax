"""``gauge_fix`` is traced end to end, and the design rests on it staying that way.

The rewrite (#882) re-gauges after **every** simple-update step, which puts a BP
solve on the per-step budget rather than the per-run one.  Run eagerly that is
18.9 ms per sweep and tens of sweeps per solve -- 490.6 ms for one D=2 solve,
measured -- i.e. the cadence is unaffordable.

Getting inside the budget took **two** boundaries, and the second one is the
part that is easy to undo by accident.  Task 7 made the solve one
``lax.while_loop`` (0.034 ms per sweep, 0.89 ms per solve) and still missed the
gate at 528 ms against 450, because what was left around it was eager: the
``BondWeights.ones``, the entry point's validation and casts, and above all
``absorb_weights``' eight ``scale_bond_axis`` calls.  Those cost 1.5 ms of a
2.47 ms warm solve -- more than the solve.  Task 7b put the whole of
``gauge_fix`` behind one jit, ``absorb_weights`` included, taking a warm solve
to **~0.92 ms** and the gate to **379-391 ms, inside budget**;
``test_re_gauging_every_step_fits_the_simple_update_budget`` in this file now
asserts it rather than recording a shortfall -- **but only when it is run with
``--no-cov``**, which this repo's ``addopts`` do not do by default.  See that
test's own docstring for the exact invocation and for why the gate withdraws
instead of failing.

So the three things worth guarding here are not "is it fast" but:

* **tracing did not move the answer** -- checked against the eager route on the
  same input, at *both* boundaries, through a *gauge-invariant* witness (see
  below);
* **every solve hits the same compiled entry** -- a solve that re-traces costs
  ~0.3 s and blows the whole run's budget on its own, while still producing the
  right answer.  Wall-clock alone cannot see that: it passes on a fast machine
  and fails mysteriously on a slow one;
* **the pair comes back in the caller's own flow convention** -- which is what
  keeps a run to *one* cache entry, since ``TensorIndex`` is pytree aux data.

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
import sys
import time
from functools import partial

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
import tenax.algorithms.ipeps_gauge as ig
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


def _gauge_fix_eager(A, B, **kw):
    """``gauge_fix`` forced down its eager route, on a dense pair.

    Same trick as :func:`_solve_eager` and for the same reason, one boundary
    further out: ``gauge_fix`` dispatches on ``bp._use_traced_loop`` too --
    reaching through the module rather than binding the name at import, so that
    this patch is honoured -- and its eager arm is the pre-Task-7b code path,
    ``bp_gauge_checkerboard`` followed by ``absorb_weights``.  Overriding the
    dispatch is what lets the two routes be compared on the *same* dense input,
    which is the only way to check that the wider jit did not move the answer.
    """
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bp, "_use_traced_loop", lambda A, B: False)
        return gauge_fix(A, B, **kw)


#: The nine random draws, plus the one input where the two drivers genuinely
#: differ.  ``_dense_pair``'s flows already match the convention ``_gauge_bond``
#: stamps, so on all nine the structure restoration is a no-op and the arm of
#: the code that #882's flow defect lives in is never compared across drivers.
#: The simple-update-evolved pair -- the one Phase 2 actually hands over -- has
#: **all five** of its flows inverted, ``phys`` included, so it is the only
#: fixture here that exercises it.  Measured on both tensors at D=2, 3 and 4:
#: ``[-1, 1, -1, 1, 1]`` for ``('u', 'd', 'l', 'r', 'phys')`` from
#: ``_dense_pair`` against ``[1, -1, 1, -1, -1]`` after the sweep.  ``"su"`` is
#: spelled as a seed rather than a second test so the assertions below cover it
#: verbatim.
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


@pytest.mark.parametrize("D,seed", _PARITY_FIXTURES)
def test_the_traced_gauge_fix_reproduces_the_eager_one(D, seed):
    """The same ten fixtures, one boundary further out.

    The test above pins the *solve*; this pins ``gauge_fix``, which since Task 7b
    is a wider jit around it -- ``BondWeights.ones``, the solve, and
    ``absorb_weights``, all compiled together.  The extra span is what makes it
    a separate claim: the parts that moved inside the trace are the ones that
    convert Vidal back to absorbed form, i.e. exactly the arithmetic
    ``test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one`` exists to
    protect, and the eager arm here is the pre-Task-7b code path verbatim.

    Measured on this branch the two routes are **bit-identical** on all ten
    fixtures -- weights 0.0 apart, torus 0.0 apart -- which is what one expects,
    since the ``while_loop`` is the same computation either way and
    ``absorb_weights`` is eight multiplies.  The tolerances below are the
    driver-level test's, not that zero: an XLA release is free to fuse the eight
    multiplies differently, and a test that demanded bit-identity would be
    reporting the compiler rather than the code.
    """
    A, B = _su_evolved_pair(D) if seed == "su" else _dense_pair(D=D, seed=seed)
    kw = {"max_iter": 400, "tol": 1e-13}
    ones = BondWeights.ones(D, D)

    A_t, B_t, w_t, info_t = gauge_fix(A, B, **kw)
    A_e, B_e, w_e, info_e = _gauge_fix_eager(A, B, **kw)

    assert info_t.converged and info_e.converged, f"{info_t} / {info_e}"
    assert info_t.iterations == info_e.iterations, (
        f"D={D} seed={seed}: traced took {info_t.iterations} sweeps, eager "
        f"{info_e.iterations}; the two routes must follow the same trajectory"
    )
    assert info_t.residual == pytest.approx(info_e.residual, rel=0.1)

    for name, a, b in zip(w_t._fields, w_t, w_e, strict=True):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert d < 1e-12, f"D={D} seed={seed}: {name} moved {d:.3e} under trace"

    # Read with ``ones``: ``gauge_fix`` returns the pair already absorbed, so
    # passing ``w_t`` here would count every bond twice (#667's mechanism) and
    # would compare two wrong readings against each other.
    rel = float(
        np.linalg.norm(
            _unit(_torus_2x2(A_t, B_t, ones)) - _unit(_torus_2x2(A_e, B_e, ones))
        )
    )
    assert rel < 1e-12, (
        f"D={D} seed={seed}: the traced gauge_fix landed on a different state "
        f"({rel:.3e}); tracing must not change the physics"
    )
    moved = float(
        np.linalg.norm(
            _unit(_torus_2x2(A_t, B_t, ones)) - _unit(_torus_2x2(A, B, ones))
        )
    )
    assert moved < GAUGE_TOL, (
        f"D={D} seed={seed}: gauge_fix moved the state {moved:.3e}"
    )


def test_gauge_fix_hands_back_the_callers_leg_flows():
    """The absorbed output must carry the *caller's* flows, not this module's.

    ``tests/test_ipeps_bp_gauge.py`` pins this for ``bp_gauge_checkerboard``.
    It needs pinning here too, and not as duplication: since Task 7b
    ``gauge_fix`` has its own jit boundary and reaches ``_bp_solve`` directly,
    so it no longer inherits the entry point's restoration, and
    ``absorb_weights`` runs *after* it inside the same trace.

    The stake is not cosmetic.  ``TensorIndex`` is pytree **aux** data, so a
    pair's flow convention is part of ``_gauge_fix_traced``'s cache key: a
    ``gauge_fix`` that handed back its own convention would make step *n+1* of a
    simple update a different key from step *n*, and every step after the first
    would pay a fresh ~285 ms compile -- 100 of them against a 450 ms budget.
    That is why this sits in the perf file.  The fixture is the
    simple-update-evolved pair precisely because its virtual flows are inverted
    relative to ``_wrap_as_dense_tensor``; the assertion that it *is* still
    non-canonical is what keeps the test from going vacuous if that changes.
    """
    A, B = _su_evolved_pair(2)
    canonical = [int(i.flow) for i in _dense_pair(D=2)[0].indices]
    assert [int(i.flow) for i in A.indices] != canonical, (
        "the simple update's output now uses the BP module's flow convention, "
        "so this test no longer probes anything -- find another non-canonical "
        "pair or delete it, do not weaken it"
    )

    A_g, B_g, _, info = gauge_fix(A, B, **_GATE_KW)
    assert info.converged, f"BP did not converge: {info}"
    for original, gauged, tag in ((A, A_g, "A"), (B, B_g, "B")):
        assert gauged.labels() == original.labels(), f"{tag}: axis order changed"
        assert [int(i.flow) for i in gauged.indices] == [
            int(i.flow) for i in original.indices
        ], f"{tag}: flows changed"


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

#: One eager ``gauge_fix`` on the gate's own fixture, on the machine class the
#: budget was measured on: 477-483 ms across three fresh processes at load1 ~4.6
#: of 128 (26 sweeps at 18.3-18.6 ms).  Used only to decide whether an absolute
#: millisecond budget can be evaluated at all here -- see the gate test.
_REFERENCE_EAGER_SOLVE_S = 0.48

#: How much slower than the reference a machine may be before the wall-clock
#: gate withdraws instead of failing.  2x is deliberately tight: it is the point
#: past which "this box is busy" and "this code regressed" stop being
#: distinguishable by that number, and the ratio-based tests above keep covering
#: the regression case at any speed.
_QUIETNESS_FACTOR = 2.0


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


def _via_gauge_fix(A, B):
    return gauge_fix(A, B, **_GATE_KW)[3]


def _via_bp_gauge_checkerboard(A, B):
    labels = A.labels()
    w = BondWeights.ones(
        A.indices[labels.index("r")].dim, A.indices[labels.index("d")].dim
    )
    return bp.bp_gauge_checkerboard(A, B, w, **_GATE_KW)[3]


def _shared_pjit_cache_is_full() -> bool:
    """Is the process-wide ``PjitFunctionCache`` *these jits use* at capacity?

    ``jax._src.pjit`` builds two of these at capacity 8192 apiece and hands each
    ``jit`` one of them, and while one is *at* capacity every insertion also
    evicts.  That is the single regime in which ``_cache_size()`` *differences*
    stop being exact, so it is the only thing the guards below are allowed to
    withdraw on.  See the comment in
    :func:`test_every_solve_in_a_run_hits_one_compiled_entry` for the
    measurements.

    **Which** of the two is not a detail: asking ``any()`` of both would let a
    cache these jits never touch license a withdrawal, and a withdrawal granted
    on an irrelevant precondition hides exactly the cache-key regression this
    file exists to catch.  So the cache is *measured*, not assumed -- a
    throwaway ``jit`` is traced and whichever cache grew is the one in use.

    Two things make the measurement worth doing rather than reading off jax's
    own rule.  ``_get_cpp_global_cache`` selects on ``contains_explicit
    attributes``, and the module comment beside it says a bare ``jax.jit(f)``
    goes to ``_cpp_pjit_cache_fun_only`` -- but on jax 0.10.2 it does not:
    a plain ``jit`` and a ``static_argnums`` ``jit`` both land in
    ``_cpp_pjit_cache_explicit_attributes``, measured at ``(1, 0)`` apiece over
    three fresh functions each, with ``fun_only`` flat at 0 the whole time.  So
    the documented rule and the shipped behaviour disagree, and a probe believes
    the right one.  The probe must also hold a reference to its throwaway
    ``jit``: drop it and the entry goes with it, and every delta reads ``(0, 0)``.

    Fails **closed** in both halves.  If a future jax stops exposing the module
    globals, or they stop answering ``size()``/``capacity()``, or the probe
    cannot tell which cache grew, this reports ``False`` -- "saturation was not
    established" -- and the caller fails loudly instead of skipping on an excuse
    it could not check.  A withdrawal that cannot verify its own precondition is
    a dead guard.
    """
    try:
        from jax._src import pjit as _pjit

        caches = (
            _pjit._cpp_pjit_cache_explicit_attributes,
            _pjit._cpp_pjit_cache_fun_only,
        )
        before = [c.size() for c in caches]
        # Same shape as the boundaries under test -- ``static_argnums`` and a
        # never-before-seen function -- so it is routed the way they are.  The
        # reference has to outlive the measurement; see the docstring.
        probe = partial(jax.jit, static_argnums=(1,))(lambda a, k: a * k)
        probe(jnp.ones((1,)), 2)
        grew = [i for i, c in enumerate(caches) if c.size() > before[i]]
        if len(grew) == 1:
            # Judge on the size *before* the probe.  The probe is an insertion,
            # so on a cache sitting one entry below capacity it fills it, and a
            # post-insertion reading would report a saturation this function
            # itself caused -- granting the withdrawal on its own side effect,
            # one entry early.  A cache that had room for the probe was not
            # saturated, which is what this comparison says.
            used = caches[grew[0]]
            return before[grew[0]] >= used.capacity()
        if not grew:
            # A guaranteed-fresh key that grows *neither* cache is the
            # definition of an evicting insert, and the probe only ever touches
            # the cache these jits use: if that one had room it would have
            # grown.  So this is saturation of the relevant cache, which is the
            # one thing a withdrawal is allowed to rest on.  (Reading it as
            # "inconclusive" would invert the guard precisely in the regime it
            # exists to detect.)
            return True
        return False  # both grew -- the probe is not isolating one cache
    except Exception:  # private jax layout moved -- do not guess
        return False


#: The two compiled boundaries around the one solve, each with the call that
#: reaches it.  ``gauge_fix`` is what a simple-update step calls and what the
#: budget is written against; ``bp_gauge_checkerboard`` is the narrower entry
#: point, still live for callers that want Vidal form back.  They are separate
#: ``jit``\\ s with separate caches, so a key that split on one and not the other
#: would be invisible to a test that only watched the one it inherited.
_COMPILED_ENTRIES = {
    "gauge_fix": (_via_gauge_fix, lambda: ig._gauge_fix_traced),
    "bp_gauge_checkerboard": (_via_bp_gauge_checkerboard, lambda: bp._bp_solve_traced),
}


@pytest.mark.parametrize("entry", sorted(_COMPILED_ENTRIES))
def test_every_solve_in_a_run_hits_one_compiled_entry(entry):
    """100 solves, one trace.  This is the failure mode wall-clock cannot see.

    Compile is the binding term in the budget below -- ~285 ms against a 450 ms
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

    Run against **both** compiled boundaries.  ``gauge_fix`` acquired its own in
    Task 7b so that ``absorb_weights`` is compiled with the solve rather than
    dispatched eagerly around it, and it inlines the driver instead of nesting
    a second ``jit`` -- so the key it is compiled under is genuinely its own and
    inheriting ``bp_gauge_checkerboard``'s coverage would prove nothing about
    it.  ``gauge_fix`` is also the one a simple-update step calls.

    The two probes at the end pin the two things that *do* split the cache key
    -- the bond dimension and the flow convention -- and the second is a live
    constraint on Phase 2 rather than a curiosity.  They are **not** what rules
    out a dead counter, which is what this docstring used to claim: measured, a
    ``_cache_size()`` stubbed to a constant still fails the *first* assertion,
    so the probes' own assertions never get to speak.  Nothing here can tell a
    dead counter from a lost compile -- both read zero -- so the first message
    names both readings instead of asserting one.  What the probes do buy is
    the calibration below: all three deltas are measured before any is judged,
    so a run whose counter has stopped tracking traces is distinguishable from
    a run that lost a compile, and only the first is withdrawn.
    """
    solve, cache_of = _COMPILED_ENTRIES[entry]
    n_solves, D, _ = _budget()
    pairs = [_dense_pair(D=D, seed=k) for k in range(5)]

    # Everything below counts entries **relative to this baseline**, never
    # against zero.  ``_cache_size()`` is a private nanobind counter with no
    # documented post-clear value, and it is not a live count of this function's
    # entries: this file failed CI at ``assert -10 == 0`` on its very first line
    # (``_bp_solve``; ``-7`` for ``_gauge_fix_traced``).
    #
    # **Reproduced on Linux at the same jax/jaxlib 0.10.2**, so it is not the
    # macOS arm64 build, and the earlier claim here that it was is withdrawn.
    # Every ``jit`` in the process shares one of two ``PjitFunctionCache``es
    # (``jax._src.pjit``, capacity 8192 each).  Below capacity the counter is
    # exact and ``clear_cache()`` removes only this function's entries and
    # zeroes it.  Once the shared LRU has had to **evict**, ``clear_cache()``
    # empties the whole shared cache and charges every removal to its caller, so
    # it lands at minus the number of entries the other functions still hold:
    # shrinking the capacity and overflowing it measured ``-3, -20, -3, -40,
    # -3`` for co-tenants holding ``3, 20, 3, 40, 3`` at capacities ``64, 64,
    # 128, 128, 256``.  The magnitude tracks the co-tenants, not the capacity,
    # which is why CI read ``-10`` and not ``-8000``.
    #
    # ``conftest.py``'s RSS-gated global ``jax.clear_caches()`` is not the
    # trigger the earlier version of this comment blamed.  It books the same
    # deficit the eviction already created, and the ``clear_cache()`` below
    # produces the identical value without it: with ten co-tenant entries both
    # read ``-10``, in either order.  What creates the deficit is the eviction.
    #
    # From a constant negative baseline the **differences** are exact -- the
    # counter still moves by one per new key -- and that is the regime CI was
    # in, which is what makes the three claims below well-posed.  They do not,
    # however, "survive an arbitrary constant offset", as this comment also used
    # to say: while the shared LRU is held *at* capacity by other functions,
    # every insert evicts too and the counter stops moving at all -- three
    # genuinely fresh keys measured ``0, 0, 0`` against a side-effect trace
    # counter's ``1, 2, 3``.  That regime is detected and refused below rather
    # than reported as a compile that never happened.
    # Sampled **before** the workload, and it is the only saturation reading the
    # withdrawal below is allowed to use.  Asking after the run would let the
    # regression grant its own excuse: the failure this test exists to catch is
    # "every solve compiles", which on a nearly-full shared cache inserts a
    # hundred entries and saturates it, making an after-the-fact predicate true
    # and skipping the very run that proved the defect.  A precondition a defect
    # can create is not a precondition.
    saturated_before = _shared_pjit_cache_is_full()

    cache_of().clear_cache()
    base = cache_of()._cache_size()

    for i in range(n_solves):
        info = solve(*pairs[i % len(pairs)])
        assert info.iterations > 0, f"solve {i} did nothing: {info}"
    after_run = cache_of()._cache_size() - base

    # A different bond dimension is a different key -- obvious, and it is what
    # makes the assertion below non-vacuous.
    solve(*_dense_pair(D=3))
    after_dim = cache_of()._cache_size() - base

    # And so is a different *flow convention* at the same D, which is much less
    # obvious and is a constraint on the caller.  ``TensorIndex`` is pytree aux
    # data, so the pair's flows are part of the key; a simple-update-evolved
    # pair has **all five** of its flows inverted relative to ``_dense_pair``'s
    # -- the four virtual legs and ``phys``, measured ``[-1, 1, -1, 1, 1]``
    # against ``[1, -1, 1, -1, -1]`` for ``('u', 'd', 'l', 'r', 'phys')`` on
    # both tensors at D=2, 3 and 4 -- and mixing the two conventions in one run
    # costs a second ~285 ms compile.  A real run hands over one convention
    # throughout -- which is exactly what ``_restore_caller_structure``
    # guarantees, by handing each caller its own metadata back instead of
    # stamping the BP module's onto it
    # (``test_gauge_fix_hands_back_the_callers_leg_flows``).  This assertion is
    # what would notice if that stopped being true.
    solve(*_su_evolved_pair(D))
    after_flows = cache_of()._cache_size() - base

    # Measured first, judged second.  Each probe above is a guaranteed-fresh key
    # after the clear, so each must move the counter by exactly one; if either
    # fails to, the counter is not counting this function's traces and none of
    # the three numbers means anything.  Withdraw only on the one cause that can
    # be *confirmed* -- a shared LRU that was already at capacity when this test
    # started -- so a counter broken for any other reason, including a cache
    # this run's own compiles filled, still fails loudly below instead of
    # skipping.
    calibrated = (after_dim - after_run, after_flows - after_dim) == (1, 1)
    if not calibrated and saturated_before:
        pytest.skip(
            f"the PjitFunctionCache these jits use was already at capacity "
            f"when this test started, so "
            f"_cache_size() has stopped counting {entry}'s traces: two "
            f"guaranteed-fresh keys moved it by {after_dim - after_run} and "
            f"{after_flows - after_dim} instead of 1 and 1.  Once the shared "
            f"LRU evicts on every insert the counter is non-conservative and "
            f"no baseline repairs it, so this run cannot measure the property.  "
            f"Re-run this file in its own process to evaluate it."
        )

    assert after_run == 1, (
        f"{n_solves} solves through {entry} produced {after_run} compiled "
        f"entries, not 1; every one of them must reuse the same trace or the "
        f"XLA compile is paid again and the re-gauging budget is gone.  The two "
        f"probes after it moved the counter by {after_dim - after_run} and "
        f"{after_flows - after_dim} against 1 and 1 expected, so "
        + (
            "it was tracking traces and this is a real miscount."
            if calibrated
            else "it was not tracking traces either -- and the shared cache "
            "these jits use was not already at capacity when this test "
            "started, so this is a dead counter, a jit that is never entered, "
            "or a cache this run's own compiles filled -- not a pre-existing "
            "LRU, which is the only thing that would excuse it."
        )
    )
    assert after_dim == 2, (
        f"a different bond dimension produced {after_dim - after_run} new "
        f"entries, not 1, so the assertion above cannot distinguish 'one "
        f"compile' from 'no counter'"
    )
    assert after_flows == 3, (
        f"a pair with inverted flows produced {after_flows - after_dim} new "
        f"entries, not 1; it reused an existing entry, and the flows are aux "
        f"data that must be part of the key"
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


def _eager_reference_seconds(A, B) -> float:
    """One **warm** eager ``gauge_fix``: this machine's speed, in one number.

    Warm, i.e. the second call, and that is the whole subtlety.  The first eager
    solve in a fresh process pays ~150 ms of first-touch compilation for tiny
    ops (``_validate_weights`` alone measures 150.2 ms then 0.45 ms), which has
    nothing to do with how fast the box is and would make a quiet machine look
    like a busy one.
    """
    _gauge_fix_eager(A, B, **_GATE_KW)
    t0 = time.perf_counter()
    r = _gauge_fix_eager(A, B, **_GATE_KW)
    jax.block_until_ready(jax.tree_util.tree_leaves(r))
    return time.perf_counter() - t0


def _coverage_is_active() -> bool:
    """Is a Python tracer -- coverage.py, or a debugger -- watching this process?

    Asked directly of the interpreter rather than of ``pytest``'s config,
    because what invalidates the measurement is the tracer, not the flag: a
    ``--cov`` that ``--no-cov`` later turned off does not slow anything down,
    and a debugger sets no pytest option at all.  Both interpreter mechanisms
    are checked -- ``sys.settrace`` on 3.11 and below, and ``sys.monitoring``,
    which coverage 7 uses on 3.12+ under ``COVERAGE_CORE=sysmon``.
    """
    if sys.gettrace() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)  # absent on 3.11
    if monitoring is None:
        return False
    # Tool ids are 0..5 (DEBUGGER, COVERAGE, PROFILER, two reserved, OPTIMIZER);
    # the range is fixed by the language, not by a constant worth importing.
    return any(monitoring.get_tool(tool_id) is not None for tool_id in range(6))


# NOTE: the ``timing`` marker is currently **decorative**.  It is registered in
# ``pyproject.toml``, and that file's comment says these tests "can be deselected
# with -m 'not timing'" -- but that string appears nowhere else in the repo.
# ``.github/workflows/ci.yml`` runs the non-core bucket as
# ``-m "not core and not slow"``, so both tests below execute on a 2-core GitHub
# runner in a bucket this project already tracks as chronically red.  Wiring the
# deselection into the workflow needs a token with the ``workflow`` scope, which
# this branch does not have.  So neither test below is allowed to fail merely
# because the machine is slow, and they buy that in two different ways:
#
#   * the warm test asserts a **ratio** against the eager route measured in the
#     same process, so a runner 20x slower slows both arms and the ratio
#     survives.  Machine-independent by construction.
#   * the gate is an absolute 450 ms and no ratio expresses it, so it
#     **calibrates and skips**: it times the eager route first and withdraws the
#     claim if this box is more than 2x the machine class the number was
#     measured on.  The budget is never widened.
#
# If you later add ``-m "not timing"`` to ci.yml, delete this note -- not the
# ratio, and not the calibration.


@pytest.mark.timing
def test_the_warm_solve_is_orders_faster_than_the_eager_one(record_property):
    """Steady state, against the same call run eagerly in the same process.

    Arithmetic, not a constant::

        n_solves = iPEPSConfig().num_imaginary_steps      # 100
        D        = iPEPSConfig().max_bond_dim             # 2
        budget   = 0.75 s target - 0.30 s un-gauged baseline

    Measured on a quiet 128-core machine (load1 ~5), three fresh processes:
    **0.91-0.94 ms** per warm ``gauge_fix`` at 26 sweeps, so 100 solves is
    ~92 ms -- 20% of the budget -- against ~480 ms for the same call forced down
    the eager route (18.4 ms/sweep).  That is ~540x, and it is the part that is
    under control; the compile below is the rest of the budget.

    Both arms are :func:`gauge_fix`, not the bare solve: since Task 7b the jit
    spans ``absorb_weights`` too, and timing only the ``while_loop`` would stop
    seeing the 1.5 ms of eager boundary that Task 7b exists to remove.  Both are
    also **warm** -- :func:`_eager_reference_seconds` discards the first eager
    call.  Timing the first one instead scores ~1980 ms and a ratio over 2000x,
    which is not a steady-state comparison but a compile: the eager route's own
    first-touch is ~150 ms on top of a cold eager loop.

    Asserted as a **ratio measured in this same process**, not as an absolute
    millisecond count.  The absolute number is what the budget is written in,
    but it is also the one thing a CI runner cannot reproduce; the ratio is a
    property of the code.  20x is a small fraction of the measured ~540x, so
    what fails here is a regression that puts per-sweep host work back into the
    loop, not a busy afternoon.  The absolute numbers are recorded either way.
    """
    n_solves, D, budget_s = _budget()
    A, B = _su_evolved_pair(D)

    warm_s = _warm_solve_seconds(A, B)
    eager_s = _eager_reference_seconds(A, B)

    record_property("warm_solve_ms", round(warm_s * 1e3, 3))
    record_property("eager_solve_ms", round(eager_s * 1e3, 1))
    record_property("hundred_warm_solves_ms", round(n_solves * warm_s * 1e3))
    record_property("budget_ms", round(budget_s * 1e3))

    assert eager_s / warm_s > 20.0, (
        f"the traced gauge_fix is only {eager_s / warm_s:.1f}x the eager one "
        f"({warm_s * 1e3:.2f} ms vs {eager_s * 1e3:.0f} ms); it was ~540x, and "
        f"{n_solves} warm solves are budgeted at {budget_s * 1e3:.0f} ms "
        f"(measured ~92 ms).  Something is dispatching per sweep again."
    )


@pytest.mark.timing
def test_re_gauging_every_step_fits_the_simple_update_budget(record_property):
    """The gate, whole: one compile plus ``num_imaginary_steps`` solves.

    **Asserted, not recorded.**  It shipped as a live ``xfail`` while Task 7 had
    the solve traced and the boundary around it eager (533-535 ms against 450);
    Task 7b put ``gauge_fix``'s own boundary inside the jit and it now measures
    **379-391 ms**, six fresh processes, load1 ~5-9 of 128 on a quiet 128-core
    machine.  Of that, 288-300 ms is the one compile and ~92 ms is the 99 warm
    solves.  The budget was not moved to meet it.

    **Asserted in no configuration this repo runs by default**, which is a
    property of the repo rather than of the gate, and is why it is spelled out
    here.  ``pyproject.toml``'s ``addopts`` carry ``--cov=tenax`` and condition
    1 below skips under coverage, so of the four ways this repo runs pytest:

    * required CI (``-m core``) -- **not collected**; ``conftest.py`` maps this
      file to ``algorithm``;
    * ``fast-other`` (``-m "not core and not slow"``, with ``--cov``) --
      collected, **skipped**;
    * the ``slow`` bucket (``-m slow``) -- **not collected**; this test carries
      ``timing``, not ``slow``;
    * plain local ``uv run pytest`` -- collected, **skipped**.

    To actually evaluate the budget::

        JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_gauge_perf.py \\
            -k re_gauging_every_step --no-cov -s

    ``-s`` because the number is ``print``ed on a pass as well as a fail.  Do
    not "fix" the skip by weakening the budget or by dropping ``--cov`` from
    ``addopts``; the measurement genuinely cannot be made under a Python
    tracer, and the two withdraw conditions below are what keep that honest.

    The compile is flat in ``D`` -- 214 ms of XLA at D=2, 216 at D=3, 224 at
    D=4, on a sweep body of 324 jaxpr equations -- so it is structure-bound
    rather than array-size-bound, unlike #633's CTM finding.  It is therefore
    ~75% of what is left, and any further headroom has to come from *not
    compiling*, not from a faster solve.  The persistent compilation cache is
    the obvious lever and is deliberately not pulled here: it is gated on
    ``jax_persistent_cache_min_compile_time_secs``, which ``tenax/__init__.py``
    pins at 1 s, and lowering that globally makes tenax write a disk entry for
    every small eager op -- a library-wide decision, not this test's (#882
    Task 7b brief).

    **What is charged, and why it is the generous accounting.**  Only
    ``_gauge_fix_traced``'s own cache is evicted, so the first solve here pays
    the trace and the XLA compile of the gauge and nothing else.
    ``jax.clear_caches()`` would also evict every *eager* op the process has
    compiled.  The cold-process accounting -- fresh interpreter, nothing warmed
    -- is the stricter one and is **also** inside budget now, at 402-422 ms
    (Task 7: 707-724).  It improved by more than the warm term alone explains,
    because the eager boundary this removed was itself ~150 ms of first-touch
    compilation: ``_validate_weights`` alone measures 150.2 ms on its first call
    in a fresh process and 0.45 ms thereafter, tiny ops costing 25-50 ms apiece
    to compile, and the traced route never calls it (its weights are
    ``BondWeights.ones``, built in and unrejectable).

    Not evicting anything is the trap this test fell into first: earlier tests
    in this file share the ``(D=2, max_iter=100, tol=1e-12)`` key, so the gate
    silently measured a warm solve and reported a passing ~250 ms.

    **It withdraws rather than lying, on three conditions, and none of them
    widens the budget.**  Unlike every other assertion in this file this one is
    an absolute wall-clock number: the budget is 450 milliseconds and no ratio
    expresses it.  So it first establishes that the number is *measurable here*.

    1. **Under a Python tracer it is not.**  ``pyproject.toml``'s ``addopts``
       carry ``--cov=tenax`` and ``.github/workflows/ci.yml`` runs every bucket
       with coverage, and ``coverage.py`` instruments exactly the Python-bound
       half of this measurement -- jaxpr tracing and lowering.  Measured on this
       branch, same process, same fixture: **288 ms compile without coverage,
       373 ms with**, while the warm term is untouched at 0.92 ms because it is
       XLA.  That turns 386 ms into 463 ms and fails a budget the code meets.
       The gate is a claim about a ``python -m`` run, and it is checked as one:
       ``-p no:cacheprovider``-style purity is not needed, ``--no-cov`` is.
    2. **On a machine several times slower it is not either.**  The eager route
       is timed in the same process as a speed reference, warm, and the claim is
       withdrawn past ``_QUIETNESS_FACTOR`` x the value this machine class
       measures -- the same cross-check the task's own measurement discipline
       demanded before any number was believed.

    3. **And if the timed call cannot be shown to have compiled, it is not
       either.**  The compile term is only a compile if the cleared cache
       actually gained an entry across it, and the counter that answers that is
       the same non-conservative ``_cache_size()`` the entry-count guard above
       documents.  When it fails to move by one *and* jax's shared
       ``PjitFunctionCache`` is confirmed at capacity, the claim is withdrawn;
       when it fails to move by one and the cache is **not** saturated, that is
       a real defect and the assertion fires.

    All three are "the number cannot be measured here", never "the number is
    allowed to be bigger here".  What still runs everywhere are the guards that
    do not depend on the machine: one compiled entry, traced-vs-eager parity,
    flow preservation, and the eager/warm ratio.
    """
    n_solves, D, budget_s = _budget()
    if _coverage_is_active():
        pytest.skip(
            "coverage.py is tracing this process, which inflates the jaxpr "
            "trace-and-lower half of the measurement by ~30% (288 -> 373 ms "
            "measured) while leaving the XLA half alone.  Re-run with --no-cov "
            "to evaluate the budget; the structural guards in this file are the "
            "machine-independent ones and did run."
        )

    A, B = _su_evolved_pair(D)
    eager_s = _eager_reference_seconds(A, B)
    record_property("eager_solve_ms", round(eager_s * 1e3, 1))
    if eager_s > _QUIETNESS_FACTOR * _REFERENCE_EAGER_SOLVE_S:
        pytest.skip(
            f"one warm eager solve took {eager_s * 1e3:.0f} ms against the "
            f"{_REFERENCE_EAGER_SOLVE_S * 1e3:.0f} ms this machine class "
            f"measures, so a {budget_s * 1e3:.0f} ms wall-clock budget cannot "
            f"be evaluated here.  The structural guards in this file -- one "
            f"compiled entry, traced-vs-eager parity, the eager/warm ratio -- "
            f"are the machine-independent ones and did run."
        )

    gauge_fix(A, B, **_GATE_KW)  # warm the eager remainder, as a real run has
    ig._gauge_fix_traced.clear_cache()
    cleared = ig._gauge_fix_traced._cache_size()

    t0 = time.perf_counter()
    first = gauge_fix(A, B, **_GATE_KW)
    jax.block_until_ready(jax.tree_util.tree_leaves(first))
    compile_s = time.perf_counter() - t0

    # Same counter, same treatment as the entry-count guard above: a difference,
    # never an absolute.  This used to read ``_cache_size() == 0`` before the
    # timed call, which is the assertion that failed CI at ``-7`` -- and it only
    # escaped notice because condition 1 skips this test under ``--cov``, so the
    # one invocation its own skip message recommends (``--no-cov``) was also the
    # one that reached it.  Asked afterwards it is a stronger claim anyway: an
    # empty cache before says the next call *may* compile, whereas one added
    # entry says it *did*.
    compiled = ig._gauge_fix_traced._cache_size() - cleared
    if compiled != 1 and _shared_pjit_cache_is_full():
        pytest.skip(
            f"jax's process-wide PjitFunctionCache is at capacity, so "
            f"_cache_size() moved by {compiled} rather than 1 across a call "
            f"that had just had its cache cleared.  The counter cannot confirm "
            f"that {compile_s * 1e3:.0f} ms is a compile rather than a warm "
            f"call, so the budget is not evaluated here.  Re-run this file in "
            f"its own process."
        )
    assert compiled == 1, (
        f"the timed call added {compiled} entries to the traced gauge's cache "
        f"instead of 1, so {compile_s * 1e3:.0f} ms is not one compile: the "
        f"clear did not take and this timed a warm call, or the jit was never "
        f"entered.  jax's shared cache is not at capacity, so the counter is "
        f"not the excuse."
    )

    warm_s = _warm_solve_seconds(A, B)
    total = compile_s + (n_solves - 1) * warm_s

    # Recorded and printed unconditionally.  A failure puts the number in its
    # own message, but a *pass* would otherwise report the bare word, and the
    # margin is what says whether the next change to this module still fits.
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
