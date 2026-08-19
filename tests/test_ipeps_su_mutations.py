"""Every guard in ``test_ipeps_su.py`` killed by the defect it is named for (#882 Task 13).

A guard that has not killed a mutant is not yet a guard.  This repo has shipped
the other outcome: three tests encoded #667's product-state collapse as
*expected behaviour* and had to be re-pointed when it was fixed, and on this
very branch the split and truncation guards passed four review rounds while
``_su_evolve`` scored 0 of 9 on energy, because their references were re-derived
in the same wrong metric as the code.  So each mutation below is a **faithful
re-introduction of an original defect**, and each is required to kill a named
guard, at a named number.

**Every cell is two-sided, and that is not decoration.**  Task 12's acceptance
suite on this branch is deliberately red -- six cells fail on purpose, listed in
:data:`_RED_CELLS` -- so the plan's own

.. code-block:: python

    with mutant():
        with pytest.raises(AssertionError):
            guard()

passes *without the mutation doing anything at all* for any guard mapped onto a
red cell.  That is an assertion that cannot fail, written into the plan for a
task whose entire subject is assertions that cannot fail.  Each cell here
therefore runs the guard **unmutated first** and requires it to pass, on exactly
the parametrisation the mutant will use; if it does not, the cell fails loudly
naming that fact rather than reporting a kill or skipping.

Three further disciplines, each of them a bug this project has already paid for:

* **``AssertionError`` is not a kill.**  A mutant can make a guard fail for the
  wrong reason -- a ``TypeError`` from a shape mismatch, or a precondition
  firing before the claim is reached -- and both look identical to a real kill.
  Each cell matches the *specific* assertion message and extracts the number the
  guard reported, then requires that number to be in the range the defect
  produces.  "Killed" without a number is not a result.
* **The mutation has to be proved to have applied.**  A harness that silently
  no-ops reports exactly what a real hole reports; a shared ``mutate.py`` did
  precisely that on an earlier round of this task and printed ``*** SURVIVED
  ***``.  :func:`_installed` checks the patched attribute *is* the mutant before
  the guard runs, restores in a ``finally`` so an exception cannot leave the
  mutation in the tree, and checks the restore took.  :func:`_pristine` re-checks
  every hook before and after every cell.
* **The guards are called, not re-implemented.**  Each cell imports the real test
  function out of ``test_ipeps_su`` and calls it.  A local paraphrase of a guard
  is a second thing to keep in step with the first, and it is the paraphrase
  that would be mutated into agreement.

Measured kills (``JAX_PLATFORMS=cpu ... --no-cov -p no:randomly``; the first
five re-measured in fix round 1 at box load 3.9 -> 6.8 and unchanged to every
digit, the sixth taken at load 6.4 -> 4.0):

======  ===========================  =====================================  =================
mutant  re-introduces                killed by                              reading
======  ===========================  =====================================  =================
667     bond carries ``lam**1.5``    ``..._d2_reaches_the_heisenberg_...``  E = -0.509176
851     one spectrum per             ``..._su_evolve_has_no_steps_mod_4_``  d = 4.903e-01
        orientation, stamped onto    ``dependence``
        the twin bond
865     ``base_charges`` pinned to   ``..._the_vidal_metric_matches_a_...`` ratio = 15.266305x
        the old bond layout          ``spectrum_derived_outside_tenax``
865     ``base_charges`` pinned to   ``..._su_step_keeps_the_largest_``    err = 1.787e-01
        the old bond layout          ``singular_values`` (``slow``)
869     the metric never re-derived  ``..._su_step_truncates_in_the_...``   ratio = 1.010944x
        (weights flat at one)
6.2a    all of ``sigma`` on the      ``..._su_step_splits_sqrt_sigma_...``  |G_i-G_j| = 3.450e+00
        left factor
======  ===========================  =====================================  =================

Every one of those numbers is the guard's **own** formatting of its **own**
reading, lifted out of the assertion message rather than recomputed here.

**#865 appears twice on purpose.**  The guard the tree names for it --
``test_su_step_keeps_the_largest_singular_values`` -- could not demonstrate a
kill at ``01cd473``, because its symmetric meta-assertion builds its reference
with the same pin the defect applies and so pre-empted the claim; fix round 1
reordered the two in ``test_ipeps_su.py`` and the pinned negative finding became
the second row, at 1.787e-01 (the number the meta-assertion held before the
swap).  The cheap external-anchor kill stays: two independent kills on one
defect, one of them at 23 s, is worth more than either alone.

Two of them reproduce figures already recorded in the tree from probes that no
longer exist, which is the point of committing this file rather than quoting a
scratch script: ``test_the_bond_guards_see_different_mutations`` quotes
``3.4e+00`` for a fully one-sided split, and
``test_su_step_truncates_in_the_state_s_own_basis`` quotes
``1.009201, 1.009628, 1.010690, 1.010944`` as the pre-fix engine's four bonds.
The 869 mutant reproduces **all four** to six decimals -- see
:func:`test_the_869_mutant_reproduces_the_pre_fix_engine_bond_for_bond`, which
also establishes that the docstring's four numbers are not in ``_BONDS`` order.

**The harness has been watched failing in each of the ways it must.**  A
mutation-testing file that can only report success is the very thing it is
auditing, so each branch was driven out of band before the file was kept:

=========================================  ==========================================
made to happen                             reported
=========================================  ==========================================
a mutation that wraps but changes nothing  ``mutant noop SURVIVED``
the guard pointed at a red cell            ``the UNMUTATED guard did not pass``
   (``..._d3_actually_uses_..._direction``     -- BP did not converge, 7.174e-02
   at seed 0, one of :data:`_RED_CELLS`)
the kill landing on another assertion      ``killed the guard, but on the wrong
                                           assertion``
the kill landing at the wrong size         ``killed the guard at 3.45, outside
                                           [1000, 10000]``
a "mutant" that is the original object     ``the mutant ... IS the original object``
an exception raised inside the context     restored; :func:`_pristine` clean
a mutation leaked past the context         ``ipeps_su.gauge_fix is <lambda>, not the
                                           object it held at import``
=========================================  ==========================================

Row 2 is C-1 in one line: against a cell that is already red, the plan's
``pytest.raises(AssertionError)`` would have reported a kill for a mutation that
did nothing.

**What this file does not do.**  It does not mutate ``ipeps_su.py`` on disk, and
it neither edits nor re-implements a guard: it imports each one and calls it.
(The one edit fix round 1 *did* make to ``test_ipeps_su.py`` is the reorder
described above -- two assertions moved earlier in a single guard, changing
nothing that guard accepts.)  Every mutation is a monkeypatch of a
module-level name that :func:`~tenax.algorithms.ipeps_su._su_step` looks up *at
call time*, so it bites however the guard reached the step -- through
``test_ipeps_su``'s own ``from ... import _su_step`` binding, or through
``_su_evolve``'s internal call.  Two mutants additionally wrap ``_su_step``
itself, and those wrap it in **both** namespaces, because a guard that calls the
imported binding would otherwise not see it (see :func:`_su_step_patches`).
"""

from __future__ import annotations

import contextlib
import re

import jax.numpy as jnp
import numpy as np
import pytest

# ``tests/`` is on ``sys.path`` -- the same route ``test_ipeps_su`` itself uses
# to reach ``_ipeps_gauge_helpers`` and ``test_ipeps_gauge``.  The guards are
# imported as a module rather than by name because two mutants have to patch
# ``test_ipeps_su._su_step``, which needs the module object anyway.
import test_ipeps_su as guards

import tenax.algorithms.ipeps_su as ipeps_su
from tenax.algorithms.ipeps_bp_gauge import BondWeights
from tenax.core._tensor_utils import scale_bond_axis

#: The six cells of Task 12's acceptance sweep that are red at ``49b65b8``, and
#: which no mutant may be mapped onto.
#:
#: They are red **on purpose** -- the user's explicit decision, with no
#: ``xfail`` -- and they are why every cell below is two-sided.  ``-m slow`` sits
#: at 6 failed / 11 passed (675.21 s); the D=4 residue and the D=3-seed-0 residue
#: are unexplained and out of scope for #882 Phase 2
#: (``task-10-reopen-report.md`` §"what did not close").  Mapping a mutant onto
#: any of these would give a cell that passes with the mutation removed.
#:
#: Two of them matter for the mapping the brief hands over:
#:
#: * ``test_d3_actually_uses_its_third_bond_direction`` is red **only at seed
#:   0**, so the guard is usable at seeds 1 and 2 -- it is not used here, for the
#:   separate reason given in :data:`_MUTANTS`' 865 row;
#: * ``test_d2_reaches_the_heisenberg_energy_not_the_product_state`` is absent
#:   from this list, i.e. green at every seed, so the 667 row's target is sound
#:   as the brief has it.  This cell uses seed 0, which is also the only seed of
#:   that test outside the ``slow`` bucket.
_RED_CELLS = (
    "test_su_evolve_reaches_the_simple_update_reference_energy[3-0]",
    "test_su_evolve_reaches_the_simple_update_reference_energy[4-0]",
    "test_su_evolve_reaches_the_simple_update_reference_energy[4-1]",
    "test_su_evolve_reaches_the_simple_update_reference_energy[4-2]",
    "test_the_energy_does_not_drift_away_with_more_steps[3-0]",
    "test_d3_actually_uses_its_third_bond_direction[0]",
)

#: Every module attribute any mutant replaces, with the object it holds now.
#:
#: Captured once at import, before anything can have been patched, and used for
#: three things: to build the mutants out of the *originals* rather than out of
#: whatever is installed (a mutant that called the patched name would recurse),
#: to prove a patch changed something, and to prove the restore put the same
#: object back.  ``ipeps_su._su_step`` and ``guards._su_step`` are the same
#: object here and are listed separately because they stop being so under a
#: patch.
_HOOKS: dict[tuple[object, str], object] = {
    (ipeps_su, "absorb_sqrt_singular_values"): ipeps_su.absorb_sqrt_singular_values,
    (ipeps_su, "gauge_fix"): ipeps_su.gauge_fix,
    (ipeps_su, "truncated_svd"): ipeps_su.truncated_svd,
    (ipeps_su, "_su_step"): ipeps_su._su_step,
    (guards, "_su_step"): guards._su_step,
}

assert ipeps_su._su_step is guards._su_step, (
    "test_ipeps_su binds _su_step to something other than the module's own "
    "function, so patching one namespace no longer implies the other and the "
    "two-namespace patching below needs re-deriving"
)


def _pristine(where: str) -> None:
    """Every hook holds the object it held at import, or say which does not.

    C-4.  A mutation left installed by a crashed cell is indistinguishable from
    a genuine hole in the *next* cell, and that is not hypothetical on this
    branch: a ``timeout``-killed run once left a mutation in the tree.  Called
    by the autouse fixture on both sides of every test in this file.
    """
    for (module, attr), original in _HOOKS.items():
        current = getattr(module, attr)
        assert current is original, (
            f"{where}: {module.__name__}.{attr} is {current!r}, not the object "
            f"it held at import ({original!r}).  A mutation has leaked out of "
            f"its context manager, so any reading taken after this point is a "
            f"reading of mutated code."
        )


@pytest.fixture(autouse=True)
def _no_leaked_mutation():
    """Fail the cell if a mutation is live on entry or survives on exit."""
    _pristine("before the test")
    yield
    _pristine("after the test")


@contextlib.contextmanager
def _installed(patches: dict[tuple[object, str], object]):
    """Install ``{(module, attr): replacement}``; verify, then restore and verify.

    The three checks are the whole of C-4, and each of them corresponds to a way
    this project has already been misled:

    * **the replacement differs from the original** -- otherwise a mutant built
      by a typo'd factory installs the function it was meant to replace and the
      guard passes, which reads exactly like a hole;
    * **the attribute is the replacement after ``setattr``** -- a module that
      re-exports through a property, or a name the caller misspelled onto a
      fresh attribute, silently mutates nothing;
    * **the attribute is the original again after the ``finally``** -- so an
      exception raised by the guard (which is the *expected* outcome here)
      cannot leave the mutation installed for the next cell.

    The restores all run before any of them is checked, so one bad restore
    cannot strand the others.
    """
    installed: dict[tuple[object, str], object] = {}
    try:
        for (module, attr), replacement in patches.items():
            original = _HOOKS[(module, attr)]
            assert getattr(module, attr) is original, (
                f"{module.__name__}.{attr} was already patched before this "
                f"mutation installed itself"
            )
            assert replacement is not original, (
                f"the mutant for {module.__name__}.{attr} IS the original "
                f"object -- this harness would report a survival that means "
                f"nothing"
            )
            setattr(module, attr, replacement)
            installed[(module, attr)] = replacement
            assert getattr(module, attr) is replacement, (
                f"setattr on {module.__name__}.{attr} did not take; the "
                f"mutation is not live and anything measured under it is a "
                f"measurement of unmutated code"
            )
        yield
    finally:
        for module, attr in installed:
            setattr(module, attr, _HOOKS[(module, attr)])
        stranded = [
            f"{module.__name__}.{attr}"
            for module, attr in installed
            if getattr(module, attr) is not _HOOKS[(module, attr)]
        ]
        if stranded:
            raise RuntimeError(
                f"could not restore {stranded} -- a mutation is still installed"
            )


# --- the fixtures the guards want, built fresh per call ---------------------


def _fresh_su_cache():
    """A new ``test_ipeps_su.su`` cache, never shared between the two sides.

    ``su`` is module-scoped and memoises pairs, gauge solves and *stepped
    states*, which is what keeps that file under ten minutes.  Reusing one cache
    across the unmutated and mutated halves of a cell would hand the mutated
    half the unmutated step out of the cache and report a survival -- the exact
    false negative this file exists to make impossible.  So each side builds its
    own, and pays the gauge solve twice.
    """
    return _unwrap(guards.su)()


def _fresh_chain_anchor():
    """A new ``test_ipeps_su.chain_anchor``, likewise never shared.

    This one caches no stepped state -- only the chain pair, its BP gauge, the
    identity gate and the external ``_CHAIN_TRUTH`` constants -- and its gauge
    comes from ``test_ipeps_su``'s own ``ipeps_gauge.gauge_fix`` binding, which
    no mutant here patches.  So sharing it across the two sides of a cell would
    in fact be sound today, and it is still not done: "the fixture happens to be
    downstream of nothing I patched" is a property of the current mutant set,
    checked by hand, and it would go on reading as true after a mutant that
    broke it was added.  The symmetric chain solve is ~6 s; that is what the
    rule costs.
    """
    return _unwrap(guards.chain_anchor)()


def _unwrap(fixture):
    """The undecorated function behind a pytest fixture object.

    ``__wrapped__`` is how a fixture exposes it; calling the fixture object
    itself is an error by design.  If that stops working, say so here rather
    than four frames down inside ``su.pair``.
    """
    factory = getattr(fixture, "__wrapped__", None)
    assert callable(factory), (
        f"{fixture!r} no longer exposes its undecorated function as "
        f"__wrapped__, so this file cannot build the fixture the guards want"
    )
    return factory


# --- the mutants -------------------------------------------------------------
#
# Each factory returns the ``{(module, attr): replacement}`` map for one defect.
# They are factories rather than module-level constants so that the ones
# carrying state (a captured spectrum, a captured charge layout) get a fresh
# holder per cell.

_ORIG_ABSORB = _HOOKS[(ipeps_su, "absorb_sqrt_singular_values")]
_ORIG_GAUGE = _HOOKS[(ipeps_su, "gauge_fix")]
_ORIG_SVD = _HOOKS[(ipeps_su, "truncated_svd")]
_ORIG_STEP = _HOOKS[(ipeps_su, "_su_step")]


def _su_step_patches(wrapper):
    """Install ``wrapper`` as ``_su_step`` in **both** namespaces.

    ``test_ipeps_su`` does ``from tenax.algorithms.ipeps_su import _su_step``, so
    its guards hold their own binding and patching ``ipeps_su._su_step`` alone
    would not reach them; ``_su_evolve``, by contrast, looks the name up in
    ``ipeps_su``'s globals on every call and would not see a patch of the test
    module alone.  Both are needed and neither is redundant.

    This is only necessary for the two mutants that have to wrap the step as a
    whole.  The other three patch a name ``_su_step`` *reads at call time*
    (``absorb_sqrt_singular_values``, ``gauge_fix``, ``truncated_svd``), which
    bites through either route without this.
    """
    return {(ipeps_su, "_su_step"): wrapper, (guards, "_su_step"): wrapper}


def _mutant_667():
    """#667: the bond carries ``lambda**1.5`` instead of ``lambda``.

    The shipped defect was arithmetic, not storage: ``Gamma`` kept a single
    ``sqrt(sigma)`` from its own SVD while the same ``sigma``, normalised to max
    1, was stored as ``lambda`` and re-absorbed *in full* by the next sweep, so
    the bond ended up carrying ``sqrt(sigma) * lambda``.  ``_SUState`` has
    nowhere to store a spectrum, so the two-sweep round trip cannot be rebuilt
    here -- but its *net power* can, and the net power is the defect.  Each end
    of the bond gets one extra ``lambda**0.25`` on top of its ``sqrt(sigma)``,
    which puts ``lambda**0.5`` more on the bond than belongs there.

    ``lambda = sigma / max(sigma)`` reproduces ``_normalise_lambda`` from the
    shipped engine, so the leading weight is untouched and only the relative
    suppression of the smaller ones is wrong -- which is what drove the state to
    the product state.

    **Not ``lambda**2``.**  That is the *other* member of this defect class --
    re-absorbing ``gauge_fix``'s weights inside ``_su_step`` -- and
    ``_su_step``'s stage 1 exists to keep the two apart.  This mutant is #667's
    own power.
    """

    def _absorb(U, s, Vh, bond_label):
        F_left, F_right = _ORIG_ABSORB(U, s, Vh, bond_label)
        s = jnp.asarray(s)
        lam = s / jnp.max(s)
        extra = jnp.where(lam > 0.0, lam**0.25, 0.0)
        return (
            scale_bond_axis(F_left, bond_label, extra),
            scale_bond_axis(F_right, bond_label, extra),
        )

    return {(ipeps_su, "absorb_sqrt_singular_values"): _absorb}


#: Which bond shares a stored slot with which, under #851's two-spectrum
#: bookkeeping: one ``lam_h`` for both horizontal bonds and one ``lam_v`` for
#: both vertical ones.  Written out rather than derived from ``_BOND_ENDS``,
#: because it is a statement about the *old* engine's storage and not about this
#: one's topology.
_TWIN = {"h_AB": "h_BA", "h_BA": "h_AB", "v_AB": "v_BA", "v_BA": "v_AB"}


def _mutant_851():
    """#851: two stored spectra for four inequivalent bonds.

    The shipped sweep kept one horizontal and one vertical spectrum, so phases 0
    and 2 wrote the same slot and whichever ran last was stamped onto *both*
    horizontal bonds -- which is why ``steps % 4`` chose the answer.  Rebuilt
    here as exactly that stamp: after each step, the spectrum the SVD just
    produced on ``bond`` is absorbed, ``sqrt`` into each end, onto ``bond``'s
    twin.

    Two hooks, because the spectrum and the stamp live in different places:
    ``absorb_sqrt_singular_values`` is where ``sigma`` is visible, and the step
    wrapper is where the output pair is.  The wrapper asserts it saw a spectrum
    rather than stamping a stale one, so a change that stopped routing through
    ``absorb_sqrt_singular_values`` fails the mutant instead of silently
    weakening it.

    The stamp is a genuine state change, not a gauge: it multiplies both ends of
    the twin bond, so the contracted network changes.  That is what makes it
    visible to a torus reading, and it is why the guard it kills can be run
    under a ``dt=0`` gate where every other movement is zero.
    """
    seen: dict[str, np.ndarray] = {}

    def _absorb(U, s, Vh, bond_label):
        seen["sigma"] = np.asarray(s)
        return _ORIG_ABSORB(U, s, Vh, bond_label)

    def _step(state, gate, max_D, bond):
        seen.pop("sigma", None)
        out = _ORIG_STEP(state, gate, max_D=max_D, bond=bond)
        sigma = seen.pop("sigma", None)
        assert sigma is not None, (
            "the #851 mutant never saw a spectrum -- _su_step no longer routes "
            "its split through absorb_sqrt_singular_values, so this mutation "
            "stamps nothing and a survival would mean nothing"
        )
        twin = _TWIN[bond]
        pair = {"A": out.A, "B": out.B}
        root = jnp.asarray(np.sqrt(sigma / sigma.max()))
        for site, leg in ipeps_su._BOND_ENDS[twin]:
            axis = pair[site].labels().index(leg)
            assert pair[site].indices[axis].dim == root.shape[0], (
                f"the {bond} spectrum has {root.shape[0]} entries and its twin "
                f"{twin} has dimension {pair[site].indices[axis].dim}; #851's "
                f"stamp only exists where the two slots have the same shape"
            )
            pair[site] = scale_bond_axis(pair[site], leg, root)
        return ipeps_su._SUState(A=pair["A"], B=pair["B"])

    return {
        (ipeps_su, "absorb_sqrt_singular_values"): _absorb,
        **_su_step_patches(_step),
    }


def _mutant_865():
    """#865: ``base_charges`` pinned to the old bond's layout on a bosonic path.

    ``_truncation_base_charges`` returns the bond leg's charges for a fermionic
    symmetry and ``None`` otherwise; imposing it anyway pins the new bond's
    per-sector *keep counts* to the old layout, which stops the SVD keeping the
    globally largest singular values.  Measured on U(1)-Sz D=3 at step 0 it kept
    ``[4.611, 1.428, 0.159]`` where the top three were
    ``[6.378, 4.611, 4.183]`` -- discarding the largest singular value and
    retaining 25.6% of the weight against an optimal 87.0%.

    Two hooks again, and for a structural reason rather than for convenience:
    the pin is a property of the *bond*, and by the time ``truncated_svd`` sees
    ``theta`` the bond leg has been contracted away.  The step wrapper reads the
    layout off the input pair -- which is what ``_truncation_base_charges``
    does, and what ``test_su_step_keeps_the_largest_singular_values``' own
    meta-assertion builds its pin from -- and publishes it; the SVD wrapper
    consumes it.  ``base_charges`` reaches ``_derive_charges`` as a *multiset*,
    so reading it off the ungauged input rather than off ``gauge_fix``'s
    permutation of the same leg pins identically.

    **Dense is structurally blind to this**, which is not a weakness of the
    guard: ``linalg.svd`` documents ``base_charges`` as ignored on the dense
    path, so the mutation is a no-op there.  The cell asserts that blindness as
    a control, because it is the entire reason the expensive symmetric arm
    cannot be traded for a cheap dense one.
    """
    pin: dict[str, np.ndarray] = {}

    def _svd(tensor, *args, **kwargs):
        assert "pin" in pin, (
            "the #865 mutant's SVD hook fired without a published bond layout, "
            "so it would have pinned nothing"
        )
        kwargs["base_charges"] = pin["pin"]
        return _ORIG_SVD(tensor, *args, **kwargs)

    def _step(state, gate, max_D, bond):
        (site_i, leg_i), _end_j = ipeps_su._BOND_ENDS[bond]
        src = getattr(state, site_i)
        pin["pin"] = np.asarray(src.indices[src.labels().index(leg_i)].charges)
        try:
            return _ORIG_STEP(state, gate, max_D=max_D, bond=bond)
        finally:
            pin.pop("pin", None)

    return {(ipeps_su, "truncated_svd"): _svd, **_su_step_patches(_step)}


def _mutant_869():
    """#869: the truncation metric is never re-derived -- weights flat at one.

    The rewrite's cadence is that every step re-derives its gauge from the pair,
    because the previous step's non-unitary gate invalidated it.  This mutant
    keeps the gauge itself -- ``gauge_fix``'s *pair* is passed through untouched
    -- and replaces only the ``BondWeights`` it returns with ones, i.e. the
    "initialise the weights and never re-derive them" shape.

    What that does is delete ``_su_step``'s stage 2: with every weight at 1 the
    six outer legs are multiplied by 1 and divided by 1, so the SVD is taken of
    the **absorbed** two-site tensor rather than the Vidal one.  Its environment
    is not the identity, its singular vectors are not Schmidt vectors, and the
    subspace it keeps is not the state's -- which is exactly the state
    ``_su_step`` shipped in before Task 10 was reopened.

    Deliberately *not* patched: ``ipeps_gauge.gauge_fix``.  The guard this kills
    builds its own reference by calling ``gauge_fix`` itself, and a mutation
    that flattened both sides would move the reference with the code and read
    1.000000 -- which is the shared-error failure mode the whole task is about.
    Patching ``ipeps_su.gauge_fix`` reaches the step and nothing else.
    """

    def _gauge(A, B, *args, **kwargs):
        A_g, B_g, weights, info = _ORIG_GAUGE(A, B, *args, **kwargs)
        flat = BondWeights(
            **{
                field: jnp.ones_like(jnp.asarray(getattr(weights, field)))
                for field in BondWeights._fields
            }
        )
        return A_g, B_g, flat, info

    return {(ipeps_su, "gauge_fix"): _gauge}


def _mutant_62a():
    """The §6.2a hole: all of ``sigma`` left on one factor.

    ``absorb_sqrt_singular_values`` exists to put ``sqrt(sigma)`` into *both*
    ends of the new bond.  Putting the whole of ``sigma`` on the left factor and
    nothing on the right leaves the same physical **state** -- a diagonal weight
    factors arbitrarily between the two legs it joins without changing the
    contracted value -- so the torus reading, the gate-application guard and
    every other closed-network probe in this tree call the two equal.  The
    output is then not in absorbed form, which is the convention ``_SUState``
    and ``gauge_fix`` share, and the next step's gauge is taken of something the
    pair is not.

    Only a reading of each end *separately* sees it, which is what the guard it
    kills is: each end's bond Gram matrix, in the Vidal metric, must be diagonal
    and the same at both ends.
    """

    def _absorb(U, s, Vh, bond_label):
        s = jnp.asarray(s)
        ones = jnp.ones_like(s)
        return (
            scale_bond_axis(U, bond_label, s),
            scale_bond_axis(Vh, bond_label, ones),
        )

    return {(ipeps_su, "absorb_sqrt_singular_values"): _absorb}


# --- the mapping -------------------------------------------------------------
#
# ``(id, factory, guard runner, message pattern, [lo, hi] the number must be
# in)``.  The runner is a callable so a cell can build the fixtures the guard
# wants -- freshly, per side.
#
# **The guard names differ from the brief's table, which named three functions
# that do not exist.**  Re-derived against ``tests/test_ipeps_su.py`` at
# ``49b65b8``:
#
#   brief                                        this file
#   -------------------------------------------  ------------------------------
#   test_d2_reaches_the_heisenberg_energy...      unchanged (:2802)
#   test_no_steps_mod_4_dependence                test_su_evolve_has_no_steps_
#                                                 mod_4_dependence (:1430)
#   test_d3_actually_uses_its_third_bond_...      exists (:2958), NOT used --
#                                                 see the 865 row below
#   test_returned_weights_are_bp_consistent       does not exist; ``_SUState``
#                                                 carries no weights by design.
#                                                 The 869 mutant is pointed at
#                                                 the truncation-basis guard
#                                                 instead -- see its row.
#   test_su_step_output_is_still_absorbed_form    test_su_step_splits_sqrt_
#                                                 sigma_into_both_ends (:710)
#
# Two rows depart from the brief on grounds of *what the mutant can reach*
# rather than of naming, and both departures were forced by a measurement:
#
# * **865 -> two guards, neither of them the D=3 rank guard.**  The brief maps
#   the ``base_charges`` pin onto
#   ``test_d3_actually_uses_its_third_bond_direction``, which runs on the
#   **dense** arm -- where ``linalg.svd`` ignores ``base_charges`` outright, so
#   the mutation is a no-op there and the cell could only ever report a
#   survival.  ``test_the_865_mutant_is_invisible_on_the_dense_arm`` is that
#   fact, measured.
#
#   The guard the brief's table should have named is
#   ``test_su_step_keeps_the_largest_singular_values[symmetric-h_AB]``, whose own
#   docstring calls it "the only executable coverage of ``base_charges=None`` ...
#   (#865)".  **At 01cd473 it could not demonstrate the kill**, for a reason
#   that is this task's own subject one level down: that cell built its #865
#   reference by pinning the *same* charges the mutant pins, so under the
#   mutation reference and code coincided, its "this cell can see #865"
#   meta-assertion fired at a separation of 6.237e-16, and the claim --
#   ``kept == top-D`` -- was never reached.  It was a detector, not a
#   demonstration that its claim is watched, and this file's first round pinned
#   that as a negative finding rather than closing it, because the task
#   constraints forbade editing ``test_ipeps_su.py``.
#
#   **Fix round 1 closed it instead.**  With that constraint lifted for the one
#   edit, the claim assertions were moved *above* the meta-assertion block in
#   ``test_ipeps_su.py``; nothing was weakened, because the meta-assertion still
#   runs on every green pass and still fails loudly on a cell that has gone
#   blind.  The claim was never 16 orders from firing -- it was 16 orders *past*
#   its own gate and simply not evaluated: under the pin the two readings swap
#   exactly, ``sep`` 1.787e-01 -> 6.237e-16 and ``err`` 6.237e-16 -> 1.787e-01,
#   both against gates of 1e-3 and 1e-11 respectively.  The pinned finding is
#   therefore now a real kill -- the row below whose id is
#   ``865-base-charges-pinned-on-the-truncation-guard`` -- at **1.787e-01**, for
#   the same ~240 s the pin cost.  It is the one row here that carries ``slow``.
#
#   #865 keeps its second, cheap kill as well, and two independent kills on one
#   defect is the point rather than an accident:
#   ``test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax``
#   (collected id ``[h_AB-horizontal-symmetric]``; stacked ``parametrize``
#   renders bottom-decorator-first) scores ``_su_step``'s truncation in its
#   reading 2 against ``||truth[max_D:]|| / ||truth||`` -- computed from the
#   infinite chain's Schmidt spectrum, rebuilt in Python ``decimal`` outside
#   tenax.  A reference that no pin can move, and 23 s rather than 240.  All
#   four of its symmetric cells kill (15.266305x on both ``h_AB`` cells,
#   6.230252x on both ``h_BA`` ones); this file runs the first.
# * **869 -> the truncation-basis guard.**  There is no "returned weights"
#   guard to kill, because there are no returned weights: ``_SUState`` has two
#   fields and nowhere to put a spectrum, which is the design premise
#   ``test_su_state_has_no_lambda_fields`` asserts.  The nearest existing guard,
#   ``test_the_stored_spectra_are_closer_to_the_bp_messages_than_the_plan_says``,
#   measures the **shipped** engine (``_simple_update_checkerboard_sweep``), so
#   no mutation of ``ipeps_su`` reaches it at all -- it would report a survival
#   for a mutant that never ran.  What a never-re-derived metric actually breaks
#   is the truncation basis, and ``test_su_step_truncates_in_the_state_s_own_
#   basis`` is the guard that reads it against Eckart-Young rather than against
#   anything ``_su_step`` computed.

#: ``(id, mutant factory, guard runner, message pattern, [lo, hi])``.
#:
#: Cost per cell, **both sides**, measured
#: ``JAX_PLATFORMS=cpu ... --no-cov -p no:randomly`` on this box at load
#: 3.9 -> 6.8: 865 22.56 s, 667 17.69 s, 851 2.35 s, 869 0.46 s, 6.2a 0.03 s.
#: None is above the ~60 s C-5 puts in ``slow``, so those five stay in
#: ``-m "not slow"``; with the two auxiliary cells the file adds **45.56 s**
#: there, against a +120 s budget.
#:
#: The sixth row is the exception and carries ``slow`` for it:
#: ``865-base-charges-pinned-on-the-truncation-guard`` runs the symmetric
#: ``D=3`` truncation guard and costs **234.34 s** (load 6.4 -> 4.0),
#: effectively all of it eager BP solves on that pair, twice over because C-1
#: wants the unmutated side.  There is no cheaper symmetric arm for that guard,
#: and it is the *same* wall-clock the negative finding it replaces used to
#: cost -- bought back as a kill.  The mark sits on the ``pytest.param``, not on
#: the function, because ``_MUTANTS`` is a single parametrised test; verified by
#: collection that ``-m "not slow"`` deselects exactly this id and ``-m slow``
#: selects exactly it.
_MUTANTS = [
    pytest.param(
        (
            "667",
            _mutant_667,
            lambda: guards.test_d2_reaches_the_heisenberg_energy_not_the_product_state(
                0
            ),
            r"E=(-?\d+\.\d+) -- at or above the product state",
            (-0.60, 0.0),
        ),
        id="667-lambda-1.5-on-the-bond",
    ),
    pytest.param(
        (
            "851",
            _mutant_851,
            lambda: guards.test_su_evolve_has_no_steps_mod_4_dependence(
                _fresh_su_cache()
            ),
            r"differs from steps=4 by ([0-9.e+-]+) under a dt=0 gate",
            (1e-3, 10.0),
        ),
        id="851-one-spectrum-per-orientation",
    ),
    pytest.param(
        (
            "865",
            _mutant_865,
            lambda: (
                guards.test_the_vidal_metric_matches_a_spectrum_derived_outside_tenax(
                    "symmetric", "horizontal", "h_AB", _fresh_chain_anchor()
                )
            ),
            r"truncation error is ([0-9.]+)x the best achievable",
            (1.01, 1e3),
        ),
        id="865-base-charges-pinned",
    ),
    pytest.param(
        (
            "869",
            _mutant_869,
            lambda: guards.test_su_step_truncates_in_the_state_s_own_basis("h_AB"),
            r"\(([0-9.]+)x\)\.  It truncates",
            (1.0001, 10.0),
        ),
        id="869-metric-never-re-derived",
    ),
    pytest.param(
        (
            "6.2a",
            _mutant_62a,
            lambda: guards.test_su_step_splits_sqrt_sigma_into_both_ends(
                "dense", "h_AB", _fresh_su_cache()
            ),
            r"max \|G_i - G_j\| = ([0-9.e+-]+)\)",
            (1e-3, 100.0),
        ),
        id="6.2a-sigma-all-on-one-factor",
    ),
    pytest.param(
        (
            "865 (own guard)",
            _mutant_865,
            lambda: guards.test_su_step_keeps_the_largest_singular_values(
                "symmetric", "h_AB", _fresh_su_cache()
            ),
            r"\(relative error ([0-9.e+-]+)\)\.  The truncation is not taking "
            r"the globally largest singular values",
            (1e-3, 10.0),
        ),
        marks=pytest.mark.slow,
        id="865-base-charges-pinned-on-the-truncation-guard",
    ),
]


@pytest.mark.parametrize("mutant", _MUTANTS)
def test_the_guard_dies_on_a_faithful_reintroduction_of_its_defect(mutant, capsys):
    """A guard that passes on a re-introduction of its own defect is not a guard.

    Two-sided, in this order, and the order is the point (C-1): the unmutated
    guard has to be watched **passing** on this exact parametrisation before the
    mutation goes in, or "the guard failed" is not evidence that the mutation
    did anything.  Six cells of this file's sibling are red by design, and
    against a red cell ``pytest.raises(AssertionError)`` is satisfied by the
    empty mutation.

    The kill is then required to be the *right* failure (C-3): an
    ``AssertionError`` whose message matches the specific claim the defect
    breaks, carrying a number in the range the defect produces.  A ``TypeError``
    from a shape mismatch, a precondition firing before the claim is reached, or
    the correct assertion firing at ``1.0000001x`` would each look like a kill
    and none of them would be one.
    """
    name, factory, run_guard, pattern, (lo, hi) = mutant

    # --- side 1: the guard passes with nothing patched ----------------------
    try:
        run_guard()
    except BaseException as exc:  # noqa: BLE001 -- pytest.fail re-raises cleanly
        pytest.fail(
            f"mutant {name}: the UNMUTATED guard did not pass -- "
            f"{type(exc).__name__}: {exc}\n\n"
            f"Nothing can be concluded about the mutation from a cell whose "
            f"unmutated half is already red: `pytest.raises(AssertionError)` "
            f"is satisfied by a mutation that does nothing.  This cell is "
            f"reporting that fact rather than a kill.  If this is one of the "
            f"six cells that are red on purpose ({', '.join(_RED_CELLS)}), the "
            f"mapping is wrong and must be re-pointed, not skipped."
        )

    # --- side 2: the same guard, with the defect back in --------------------
    patches = factory()
    with _installed(patches):
        try:
            run_guard()
        except AssertionError as exc:
            message = str(exc)
        else:
            pytest.fail(
                f"mutant {name} SURVIVED: {sorted(a for _m, a in patches)} were "
                f"replaced and verified live, and the guard still passed.  The "
                f"guard does not watch the defect it is named for."
            )

    match = re.search(pattern, message)
    assert match, (
        f"mutant {name} killed the guard, but on the wrong assertion.  Expected "
        f"a message matching {pattern!r}; got:\n\n{message}\n\n"
        f"A guard can fail from a precondition or a shape error under a "
        f"mutation without ever reaching the claim it exists to make, and that "
        f"is indistinguishable from a real kill unless the message is checked."
    )
    reading = float(match.group(1))
    assert lo <= reading <= hi, (
        f"mutant {name} killed the guard at {reading:.6g}, outside the "
        f"[{lo:.6g}, {hi:.6g}] this defect produces.  The guard fired, but not "
        f"at the size of the defect that was injected -- so either the mutation "
        f"is not the one documented or the guard is reading something else."
    )
    with capsys.disabled():
        print(f"\n  [mutation] {name}: killed at {reading:.6g}")


#: The pre-fix engine's truncation-quality ratios on seed 0's four bonds, as a
#: **set**.
#:
#: Quoted in ``test_su_step_truncates_in_the_state_s_own_basis``' docstring as
#: what ``_su_step`` read before Task 10's reopening -- taken out of band, with a
#: private script that no longer exists, which is the state of affairs this task
#: is here to end.  As a set rather than a mapping because the sentence they come
#: from does not say which is which, and this file found out: the docstring's
#: order is not ``_BONDS``'.
_PRE_FIX_RATIOS = (1.009201, 1.009628, 1.010690, 1.010944)


def test_the_869_mutant_reproduces_the_pre_fix_engine_bond_for_bond():
    """The 869 mutant is not *a* wrong metric -- it is the one that shipped.

    The cell above requires the mutant to kill one bond's guard at a number in a
    range.  That is enough to call it a kill and not enough to call the mutation
    faithful: many wrong metrics would push the ratio above 1.  This pins the
    stronger statement, which is the one that makes the mutation evidence about
    the *defect* rather than about mutation in general -- with the weights held
    flat, all four bonds reproduce
    ``test_su_step_truncates_in_the_state_s_own_basis``' recorded pre-fix
    readings to six decimal places:

    ========  ========
    bond      ratio
    ========  ========
    ``h_AB``  1.010944
    ``h_BA``  1.009201
    ``v_AB``  1.009628
    ``v_BA``  1.010690
    ========  ========

    Compared as a **set**, because the docstring those four numbers come from
    lists them without saying which bond is which, and they are not in
    ``_BONDS`` order -- which this file is how anybody found out.  Matching all
    four to 1e-6 is a far stronger statement than matching one to a range, and
    it also serves C-4: a harness that silently failed to mutate could not
    produce these numbers at all.

    **Two-sided on every bond**, not just on the one ``_MUTANTS`` covers.  The
    parametrised cell establishes an unmutated pass for ``h_AB`` alone; the
    other three had none, so on those the "it failed" half rested on nothing
    and would have read the same way against a red cell.  Each bond is
    therefore run unmutated first here as well, with the same C-1 refusal.  The
    guard is cheap (~0.15 s a bond), so this is a fifth of a second, not a
    trade.
    """
    got = []
    for bond in guards._BONDS:
        # C-1, once per bond.  ``_MUTANTS`` establishes an unmutated pass for
        # ``h_AB`` only, and on a bond that is already red "the guard raised
        # AssertionError" is satisfied by a mutation that does nothing -- the
        # same hole the parametrised cell's side 1 exists to close, and it does
        # not stop being a hole because the discriminator below is a 1e-6
        # multiset rather than ``pytest.raises``.  ~0.15 s per bond.
        try:
            guards.test_su_step_truncates_in_the_state_s_own_basis(bond)
        except BaseException as exc:  # noqa: BLE001 -- pytest.fail re-raises cleanly
            pytest.fail(
                f"mutant 869: the UNMUTATED guard did not pass on {bond} -- "
                f"{type(exc).__name__}: {exc}\n\n"
                f"Nothing can be concluded about the mutation from a bond whose "
                f"unmutated half is already red: the kill asserted below is "
                f"satisfied by a mutation that does nothing.  This cell is "
                f"reporting that fact rather than a kill.  If this is one of "
                f"the six cells that are red on purpose "
                f"({', '.join(_RED_CELLS)}), the mapping is wrong and must be "
                f"re-pointed, not skipped."
            )

        with _installed(_mutant_869()):
            try:
                guards.test_su_step_truncates_in_the_state_s_own_basis(bond)
            except AssertionError as exc:
                match = re.search(r"\(([0-9.]+)x\)\.  It truncates", str(exc))
                assert match, f"{bond}: unexpected failure\n{exc}"
                got.append(float(match.group(1)))
            else:
                pytest.fail(
                    f"{bond}: the flat-weights mutant survived, so the guard "
                    f"does not read the truncation basis on this bond"
                )

    for want, have in zip(_PRE_FIX_RATIOS, sorted(got)):
        assert abs(have - want) < 1e-6, (
            f"the flat-weights mutant reads {sorted(got)} where the engine as "
            f"shipped read {list(_PRE_FIX_RATIOS)}.  The mutation is a wrong "
            f"metric but not the wrong metric that shipped, so a kill under it "
            f"is evidence about mutation testing and not about #869."
        )


def test_the_865_mutant_is_invisible_on_the_dense_arm():
    """Why #865 has to be killed on a symmetric cell, measured.

    ``linalg.svd`` ignores ``base_charges`` on the dense path -- its own
    docstring says so -- so the pin the mutant restores is a no-op there.  This
    is that sentence as a measurement, and it is why the brief's mapping of #865
    onto ``test_d3_actually_uses_its_third_bond_direction`` (dense, 1200 steps,
    three seeds, ~20 minutes) could only ever have reported a survival.

    Asserting a **survival** is normally the thing this file exists to stop, and
    it is only admissible here because the parametrised cell above watches the
    same mutant object kill a symmetric guard at 15.266305x.  On its own this
    would be indistinguishable from a harness that failed to mutate; together
    the two say the mutation is real *and* that dense cannot see it.
    """
    guards.test_su_step_keeps_the_largest_singular_values(
        "dense", "h_AB", _fresh_su_cache()
    )
    with _installed(_mutant_865()):
        guards.test_su_step_keeps_the_largest_singular_values(
            "dense", "h_AB", _fresh_su_cache()
        )
