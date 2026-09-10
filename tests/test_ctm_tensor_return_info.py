"""``ctm_tensor(return_meta=True)``: whether the CTM converged, as a measurement.

Single-site ``ctm_tensor`` returned ``(env, eps)`` and nothing else.  A run that
reached ``conv_tol`` and a run that spent its whole budget on a limit cycle
returned the same shape of result, and the difference is not visible in the
returned tensors -- so a caller reading an energy off ``env`` had no way to ask
whether the number was a fixed point.  That is the mechanism behind #901: a
``recipe="1x1"`` cycle of ~6e-3 amplitude sat underneath a passing ``< 1e-3``
assertion for weeks because nothing in the call chain said otherwise.

``_ctm_tensor_multisite`` grew a *warning* for this in #910.  ``ctm_tensor``
takes the opt-in route already established by #839 for :func:`ctm`,
:func:`ctm_2site` and :func:`ctm_split`: the same ``return_meta`` flag returning
the same :class:`CTMConvergenceInfo`.  The default path is unchanged.

Two things are pinned that the shared type does *not* itself guarantee, because
they are properties of this loop:

* ``diff`` must be ``inf`` -- never ``0.0`` -- when fewer than two sweeps ran;
* a rank-collapsed corner (#898) must report ``converged=False`` rather than
  reading "no change" as convergence.
"""

import inspect
import math
import typing

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_diagnostics import env_is_collapsed
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo


@pytest.fixture(scope="module")
def su_state():
    """A physical D=2 Heisenberg state -- converges under the 2x2 recipe.

    Deliberately not a random tensor: the collapse case below needs a state on
    which ``2x2`` genuinely converges, or its control proves nothing.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    cfg = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=60,
        dt=0.05,
        unit_cell="1x1",
        # This CTM is dead weight: ``ipeps()`` runs simple update first and the
        # tensors are fixed before it starts, so ``config.ctm`` cannot affect
        # them -- and this fixture discards both the energy and the env.  It
        # was spending its whole budget without converging (the
        # "CTM did not converge in ipeps()" warning), then throwing the
        # result away.  chi is unchanged; only the sweep count is cut (#933).
        ctm=CTMConfig(chi=8, max_iter=2, conv_tol=1e-10),
    )
    _E, tensors, _envs = ipeps(gate, None, cfg)
    return tensors[0]


# ------------------------------------------------------------------ #
# The default path must not move                                      #
# ------------------------------------------------------------------ #


def test_default_call_still_returns_a_two_tuple(su_state):
    """``return_meta`` is opt-in; every existing caller unpacks two values."""
    result = ctm_tensor(su_state, chi=8, max_iter=20, conv_tol=1e-10)
    assert len(result) == 2
    _env, eps = result
    assert isinstance(eps, float)


def test_return_meta_does_not_change_the_first_two_elements(su_state):
    """The info is *additional*, not a different computation."""
    env_a, eps_a = ctm_tensor(su_state, chi=8, max_iter=20, conv_tol=1e-10)
    env_b, eps_b, info = ctm_tensor(
        su_state, chi=8, max_iter=20, conv_tol=1e-10, return_meta=True
    )
    assert isinstance(info, CTMConvergenceInfo)
    assert eps_a == eps_b
    delta = jax.numpy.max(jax.numpy.abs(env_a.C1._data - env_b.C1._data))
    assert float(delta) == 0.0


def test_the_public_return_annotation_is_resolvable_at_runtime():
    """#920 review P2: a `TYPE_CHECKING`-only import breaks introspection.

    `ctm_tensor` is public API, so `typing.get_type_hints` and
    `inspect.signature(..., eval_str=True)` must work on it -- runtime validators
    and doc tooling call them.  The first version imported
    `CTMConvergenceInfo` only under `TYPE_CHECKING` (to dodge a circular import)
    and inside the `return_meta` branch, so the name was absent from the
    function's runtime globals and both calls raised `NameError`.

    Fixed by moving the type down into this module rather than papering over the
    cycle -- `ipeps_ctm_convergence` already imports *from* here, so defining it
    here removes the cycle instead of deferring it.  No CTM runs in this test.
    """
    hints = typing.get_type_hints(ctm_tensor)
    assert "return" in hints
    assert inspect.signature(ctm_tensor, eval_str=True) is not None


def test_there_is_exactly_one_CTMConvergenceInfo_class():  # noqa: N802
    """Every import path must reach the *same* class object.

    A second same-named class is not a hypothetical: the first attempt at this
    feature defined one in `_ctm_tensor_convergence` while `tenax` was already
    bound to the #839 NamedTuple, and it failed its own isinstance check with an
    identical-looking repr.  Nothing in a traceback distinguishes them.
    """
    from tenax import CTMConvergenceInfo as via_tenax
    from tenax.algorithms.ipeps_ctm import CTMConvergenceInfo as via_shim
    from tenax.algorithms.ipeps_ctm_convergence import CTMConvergenceInfo as via_ipeps

    assert via_tenax is via_shim is via_ipeps is CTMConvergenceInfo


def test_return_meta_is_keyword_only(su_state):
    """Positional callers must not be able to hit it by accident.

    ``ctm_tensor`` has nine positional-or-keyword parameters ahead of it; a
    tenth positional would be a live trap for anyone passing ``recipe``
    positionally.
    """
    with pytest.raises(TypeError):
        ctm_tensor(su_state, 8, 20, 1e-10, True, "svd", 3, "auto", "2x2", True)


# ------------------------------------------------------------------ #
# converged / not converged, both measured                            #
# ------------------------------------------------------------------ #


def test_reports_convergence_when_it_happens(su_state):
    conv_tol = 1e-10
    _env, _eps, info = ctm_tensor(
        su_state, chi=8, max_iter=60, conv_tol=conv_tol, return_meta=True
    )
    assert info.converged
    assert info.diff < conv_tol
    # Regime guard: it must have converged *early*, not been certified by
    # running out of budget.  If the fixture ever drifts so that 60 sweeps are
    # only just enough, this fires instead of the test quietly weakening.
    assert info.n_iter < 60


def test_reports_non_convergence_on_an_exhausted_budget(su_state):
    """A budget that ran out reports a real measured residual, not ``inf``."""
    conv_tol = 1e-14
    _env, _eps, info = ctm_tensor(
        su_state, chi=8, max_iter=4, conv_tol=conv_tol, return_meta=True
    )
    assert not info.converged
    assert math.isfinite(info.diff)
    assert info.diff >= conv_tol
    # The shared type documents "n_iter == max_iter exactly when converged is
    # False"; this loop must honour that, and it is also the regime guard --
    # the case is only about exhaustion if every sweep was used.
    assert info.n_iter == 4


def test_converged_is_never_true_without_a_measurement_under_tol(su_state):
    """Cross-check: ``converged`` and ``diff`` are set separately, so they must agree."""
    for max_iter, conv_tol in [(60, 1e-10), (4, 1e-14), (1, 1e-10), (0, 1e-10)]:
        _env, _eps, info = ctm_tensor(
            su_state, chi=8, max_iter=max_iter, conv_tol=conv_tol, return_meta=True
        )
        if info.converged:
            assert info.diff < conv_tol
            assert info.n_iter >= 2


# ------------------------------------------------------------------ #
# "Not measured" must not read as "perfectly converged"               #
# ------------------------------------------------------------------ #


def test_a_single_sweep_reports_no_criterion_rather_than_a_perfect_one(su_state):
    """The criterion compares a *pair* of spectra, so one sweep produces none.

    The trap this pins is an initialiser: ``diff = 0.0`` would make this case
    report the most perfectly converged number available, in the same breath as
    ``converged=False``.
    """
    _env, _eps, info = ctm_tensor(
        su_state, chi=8, max_iter=1, conv_tol=1e-10, return_meta=True
    )
    assert info.n_iter == 1
    assert info.diff == math.inf
    assert not info.converged


def test_zero_budget_reports_instead_of_raising(su_state):
    """Everything the reporting path reads is assigned before the loop (#901)."""
    _env, _eps, info = ctm_tensor(
        su_state, chi=8, max_iter=0, conv_tol=1e-10, return_meta=True
    )
    assert info.n_iter == 0
    assert not info.converged
    assert info.diff == math.inf


def test_a_warmup_that_consumes_the_whole_budget_still_reports(su_state):
    """``qr_warmup_steps == max_iter`` leaves the measured loop with nothing.

    Three sweeps really ran, so ``n_iter`` is 3 -- not 0 -- and the criterion is
    a real measured number, not ``inf`` (#920 review P2, both rounds).  Counting
    the sweeps while discarding their spectra made the report contradict itself:
    "three sweeps ran" alongside "nothing was measured".
    """
    _env, _eps, info = ctm_tensor(
        su_state,
        chi=8,
        max_iter=3,
        conv_tol=1e-10,
        projector_method="qr",
        qr_warmup_steps=3,
        return_meta=True,
    )
    assert info.n_iter == 3, "warm-up sweeps moved the environment; count them"
    assert not info.converged
    assert math.isfinite(info.diff), "warm-up spectra must be measured, not discarded"


def test_a_pure_warmup_run_measures_what_the_same_sweeps_measure_without_one(su_state):
    """The sharpest form of #920 review P2, on the default recipe.

    ``recipe="2x2"`` ignores ``projector_method`` -- its sweep wrapper hardcodes
    the plaquette projector -- so a warm-up that consumes the whole budget runs
    the *same* sweeps that otherwise converge.  Before the fix this reported
    ``converged=False, diff=inf``: a genuine fixed point, certified as nothing.

    The two runs do not agree on ``n_iter``, and that is by design rather than a
    loose end.  The warm-up deliberately does not break on ``conv_tol`` -- having
    converged under eigh is not the same as having converged under the projector
    the caller asked for, and breaking there would silently skip QR -- so it
    spends the whole budget where the plain run exits early.  It therefore ends
    up *more* converged, which is the direction that cannot mislead.
    """
    common = dict(chi=8, max_iter=30, conv_tol=1e-10, recipe="2x2", return_meta=True)
    _e, _x, warmed = ctm_tensor(
        su_state, projector_method="qr", qr_warmup_steps=30, **common
    )
    _e, _x, plain = ctm_tensor(su_state, projector_method="svd", **common)

    # The control must actually converge, or "warmed converged too" proves nothing.
    assert plain.converged and plain.diff < 1e-10
    assert warmed.converged, "a converged fixed point must not report uncertified"
    assert math.isfinite(warmed.diff) and warmed.diff < 1e-10
    # It ran the full budget precisely because the warm-up does not early-break.
    assert warmed.n_iter == 30 and plain.n_iter < 30


def test_n_iter_equals_the_callers_max_iter_when_the_budget_is_exhausted(su_state):
    """The field's documented invariant, under a warm-up that does not consume it.

    ``n_iter == max_iter`` exactly when ``converged`` is False -- against the
    ``max_iter`` the *caller* passed, not the post-warm-up remainder.  With
    ``qr_warmup_steps=6, max_iter=10`` the old behaviour reported 4.
    """
    _env, _eps, info = ctm_tensor(
        su_state,
        chi=8,
        max_iter=10,
        conv_tol=1e-14,
        projector_method="qr",
        qr_warmup_steps=6,
        return_meta=True,
    )
    assert not info.converged
    assert info.n_iter == 10


# ------------------------------------------------------------------ #
# Collapse: the other reason the criterion can be inf                 #
# ------------------------------------------------------------------ #


def test_a_collapsed_corner_is_never_certified_as_converged(su_state):
    """``recipe="1x1"`` collapses the corner to rank 1 (#723/#911).

    The criterion returns ``inf`` on such a spectrum (#898) rather than reading
    an unchanging rank-1 corner as perfect convergence, so ``converged`` must be
    ``False`` however long the loop runs.  The ``2x2`` control is what makes
    this non-vacuous: without it the test passes if *everything* is reported
    uncertified.
    """
    env_bad, _eps, bad = ctm_tensor(
        su_state, chi=8, max_iter=40, conv_tol=1e-12, recipe="1x1", return_meta=True
    )
    env_good, _eps, good = ctm_tensor(
        su_state, chi=8, max_iter=40, conv_tol=1e-12, recipe="2x2", return_meta=True
    )
    assert env_is_collapsed(env_bad)
    assert not env_is_collapsed(env_good)
    assert not bad.converged
    assert bad.diff == math.inf


def test_the_two_infinite_criteria_need_the_env_to_be_told_apart(su_state):
    """``inf`` from "never evaluated" vs ``inf`` from "evaluated on a collapse".

    Same ``diff``, opposite diagnosis: the first wants more sweeps, the second
    wants a different recipe and will never be fixed by more sweeps.  The shared
    ``CTMConvergenceInfo`` cannot separate them -- ``n_iter`` distinguishes the
    two here, but only because the collapsed run was given a large budget, so
    the honest separator is the environment itself.  This pins that
    ``env_is_collapsed`` is the thing to reach for, which is what the docstring
    tells callers to do.
    """
    env_unmeasured, _eps, unmeasured = ctm_tensor(
        su_state, chi=8, max_iter=1, conv_tol=1e-10, return_meta=True
    )
    env_collapsed, _eps, collapsed = ctm_tensor(
        su_state, chi=8, max_iter=40, conv_tol=1e-12, recipe="1x1", return_meta=True
    )
    assert unmeasured.diff == collapsed.diff == math.inf
    assert not unmeasured.converged and not collapsed.converged
    # The info alone is ambiguous; the env is not.
    assert not env_is_collapsed(env_unmeasured)
    assert env_is_collapsed(env_collapsed)
