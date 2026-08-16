"""#785: gradient accuracy can be measured, and cannot be predicted.

``measure_gradient_error`` is the only thing in the library that says whether a
root-implicit gradient is accurate.  These tests pin two separate claims:

1. It measures what it says -- it agrees with a hand-rolled finite difference,
   it reports a *wrong* gradient as wrong, and it refuses to call an
   unconverged finite difference converged.
2. The cheap surrogates really do fail, so nobody re-derives one from the
   diagnostics dict.  The rank-matched pair below is the evidence: same
   ``usable_rank``, the state with the **larger** residual has the **better**
   gradient.

The states here are D=2 at chi=4 and each root-implicit call is a full CTM
convergence, so the surrogate test is marked slow; the measurement-contract
tests use a trivial closed-form map and cost nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._root_implicit_gradient_check import (  # noqa: E402
    GradientErrorReport,
    measure_gradient_error,
)
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402


def _wrap(data, D=2, d=2):
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(jnp.asarray(data), idx)


def _quartic_state(seed=0):
    rng = np.random.RandomState(seed)
    return _wrap(rng.standard_normal((2, 2, 2, 2, 2)))


def _exact_pair(scale=1.0):
    """A closed-form energy with an exactly known gradient.

    ``E = sum(x^4)`` has ``dE/dx = 4 x^3``; ``scale`` multiplies the returned
    gradient so a *deliberately wrong* one can be handed over.
    """

    def energy_and_grad(A):
        x = jnp.asarray(A.todense())
        return jnp.sum(x**4), 4.0 * scale * x**3

    return energy_and_grad


# --------------------------------------------------------------------------- #
# 1. It measures what it claims to measure                                      #
# --------------------------------------------------------------------------- #


def test_a_correct_gradient_measures_as_correct():
    """The exact gradient of a smooth map comes back at the FD noise floor."""
    report = measure_gradient_error(_exact_pair(), _quartic_state())

    assert isinstance(report, GradientErrorReport)
    # Better than the scan can resolve -- the good case, and NOT a failure.
    assert not report.is_resolved, report.summary()
    assert report.relative_error < 1e-6, report.summary()
    assert report.resolution < 1e-6, report.summary()


@pytest.mark.parametrize("scale, want", [(1.15, 0.15), (2.0, 1.0), (0.5, 0.5)])
def test_a_wrong_gradient_measures_as_wrong_by_the_right_amount(scale, want):
    """A gradient scaled by ``s`` reports relative error ``|s - 1|``.

    Not merely "reports something nonzero": the magnitude has to be right, or
    the number cannot be used to decide whether a state is usable.  This is the
    check that would have caught #841's 15x if it had existed on that path.
    """
    report = measure_gradient_error(_exact_pair(scale), _quartic_state())

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(want, rel=1e-3), report.summary()


@pytest.mark.parametrize("scale", [1e-6, 1e6, 1e-200, 1e200])
def test_the_direction_is_normalised_so_the_error_is_scale_free(scale):
    """A caller-supplied direction of any length gives the same relative error.

    The outer two scales are the load-bearing ones.  ``np.linalg.norm`` squares
    before summing, so it halves the usable exponent range: at ``1e200`` it
    overflows to ``inf`` and ``v / inf`` collapses the direction to exactly
    zero, at which point ``analytic``, the finite difference and ``resolution``
    are all 0 and a 30%-wrong gradient reports as unresolvably accurate.  At
    ``1e-200`` the same square underflows the norm to 0 instead.  This is the
    #870 trap; normalising by max-abs first is the fix.
    """
    A = _quartic_state()
    v = np.random.RandomState(3).standard_normal(A.todense().shape)

    report = measure_gradient_error(_exact_pair(1.3), A, direction=v * scale)

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(0.3, rel=1e-3), report.summary()


def test_a_non_finite_direction_is_refused():
    A = _quartic_state()
    v = np.random.RandomState(3).standard_normal(A.todense().shape)
    v[0, 0, 0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite entries"):
        measure_gradient_error(_exact_pair(), A, direction=v)


def test_the_best_step_is_reported_not_the_worst():
    """A scan containing one useless step still reports the good one.

    Every step agrees on a smooth map at sane step sizes, so best-vs-worst is
    invisible there -- a mutant swapping ``min`` for ``max`` survived the rest
    of this file.  ``h=0.1`` on a quartic carries real O(h^2) truncation
    (~1e-2 relative), which is exactly the regime a real CTM map lands in, so
    picking the worst step would report the scan's own truncation as the
    gradient's error.
    """
    report = measure_gradient_error(_exact_pair(), _quartic_state(), steps=(1e-1, 1e-6))

    assert report.relative_error < 1e-6, (
        f"reported {report.relative_error:.2e}, which is the h=0.1 truncation "
        f"rather than the h=1e-6 measurement -- {report.summary()}"
    )
    assert report.step == pytest.approx(1e-6)


# --------------------------------------------------------------------------- #
# 2. It refuses to report a number the finite difference cannot support         #
# --------------------------------------------------------------------------- #


def test_an_unconverged_finite_difference_is_reported_not_hidden():
    """A map with no usable FD step must come back with a huge ``resolution``.

    An FD reference that has not converged reports *its own* truncation as the
    gradient's error.  Returning that as a bare number is how a check agrees to
    four digits with a reference that is itself wrong, so the scan has to say
    how well it could resolve anything at all.
    """
    rng = np.random.RandomState(0)

    def noisy(A):
        x = jnp.asarray(A.todense())
        # Energy contaminated at a level that swamps any central difference:
        # the step-to-step disagreement is then huge and nothing has plateaued.
        noise = rng.standard_normal() * 1e-3
        return jnp.sum(x**4) + noise, 4.0 * x**3

    report = measure_gradient_error(noisy, _quartic_state())

    # Nothing measurable: the floor is enormous, so no claim can stand clear.
    assert not report.is_resolved, report.summary()
    assert report.resolution > 1e-3, report.summary()


def test_one_step_is_refused_because_it_cannot_detect_its_own_truncation():
    with pytest.raises(ValueError, match="at least two distinct magnitudes"):
        measure_gradient_error(_exact_pair(), _quartic_state(), steps=(1e-5,))


@pytest.mark.parametrize("steps", [(1e-5, 1e-5), (1e-5, -1e-5), (-1e-5, 1e-5)])
def test_two_steps_of_one_magnitude_are_refused(steps):
    """A one-step scan wearing two entries defeats the resolution guard.

    The central difference is even in ``h``, so both of these evaluate the
    *same* difference twice: ``resolution`` comes out exactly 0, every nonzero
    disagreement satisfies ``best_rel > tol * 0``, and a pure truncation
    artifact would be reported as a resolved measurement -- the precise thing
    the single-step check exists to prevent.
    """
    with pytest.raises(ValueError, match="at least two distinct magnitudes"):
        measure_gradient_error(_exact_pair(1.2), _quartic_state(), steps=steps)


def test_a_zero_step_is_refused():
    with pytest.raises(ValueError, match="nonzero"):
        measure_gradient_error(_exact_pair(), _quartic_state(), steps=(0.0, 1e-5))


def test_a_symmetric_tensor_is_refused_rather_than_silently_perturbed():
    """A dense shift of a block-sparse buffer leaves the allowed sectors."""

    class _NotDense:
        pass

    with pytest.raises(TypeError, match="DenseTensor"):
        measure_gradient_error(_exact_pair(), _NotDense())


# --------------------------------------------------------------------------- #
# 3. The cheap surrogates fail -- measured, so nobody rebuilds one              #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_residual_is_anticorrelated_with_gradient_error_on_a_rank_matched_pair():
    """Same ``usable_rank``; larger residual, better gradient (#785).

    This is the whole content of #785 in one assertion.  If it ever fails --
    if the residual starts ordering these two states correctly -- then a
    residual-based gate has become possible and #785 should be reopened rather
    than this test deleted.
    """
    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_root_implicit_energy_and_grad,
    )

    try:
        from _su_fixtures import physical_su_d2
    except ImportError:  # pragma: no cover - fixture lives beside the tests
        pytest.skip("_su_fixtures not importable")

    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = (jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)).reshape(
        2, 2, 2, 2
    )

    rng = np.random.RandomState(42)
    spiked = 0.03 * rng.standard_normal((2, 2, 2, 2, 2))
    spiked[0, 0, 0, 0, 0] = 1.0
    spiked = spiked / np.linalg.norm(spiked)

    kw = dict(chi=4, max_iter=300, conv_tol=1e-13, on_root_residual="warn")

    def run(A):
        _E, g, diag = asym_root_implicit_energy_and_grad(
            A, H, return_diagnostics=True, **kw
        )
        # A random direction and exactly these two steps, both load-bearing.
        #
        # Probing along ``g`` instead maximises ``|g.v|`` and drives BOTH
        # states' errors below their own resolution floors (8.7e-08 against a
        # 9.6e-07 floor), at which point the two are indistinguishable and the
        # comparison this test exists for is unsupported by its own
        # measurement.  Widening the scan to 1e-4/1e-7 does the same from the
        # other side: this random direction is near-orthogonal to ``g`` on the
        # spiked state (``g.v = -1.5e-05``), so the outer steps are roundoff-
        # and truncation-dominated and the floor grows to 2.1e-05.
        del g
        err = measure_gradient_error(
            lambda t: asym_root_implicit_energy_and_grad(t, H, **kw)[:2],
            A,
            steps=(1e-5, 1e-6),
        )
        return diag, err

    phys_diag, phys_err = run(physical_su_d2())
    spike_diag, spike_err = run(_wrap(spiked))

    assert phys_diag["usable_rank"] == spike_diag["usable_rank"], (
        "the pair is only evidence about #785 while the ranks match; "
        f"got {phys_diag['usable_rank']} and {spike_diag['usable_rank']}"
    )
    # The physical state has the LARGER residual and the BETTER gradient.
    assert phys_diag["covariant_residual"] > spike_diag["covariant_residual"]
    assert phys_err.relative_error < spike_err.relative_error
    # And the gap has to exceed what either scan can resolve, or "one is better
    # than the other" is a statement about finite-difference noise.
    gap = spike_err.relative_error - phys_err.relative_error
    floor = max(phys_err.resolution, spike_err.resolution)
    assert gap > floor, (
        f"the {gap:.2e} gap between the two gradients is inside the finite "
        f"difference's own {floor:.2e} resolution, so this measurement does "
        f"not establish that one is more accurate than the other\n"
        f"  physical: {phys_err.summary()}\n"
        f"  spiked:   {spike_err.summary()}"
    )


# --------------------------------------------------------------------------- #
# 4. Complex states -- a real probe direction is blind to half the gradient     #
# --------------------------------------------------------------------------- #


def _complex_state(seed=0):
    rng = np.random.RandomState(seed)
    data = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal(
        (2, 2, 2, 2, 2)
    )
    return _wrap(data)


def _complex_pair(imag_scale=1.0):
    """Real-valued energy of a complex tensor, with its exact JAX gradient.

    ``imag_scale != 1`` corrupts **only** the imaginary components, which is
    precisely the error a real probe direction cannot see: it pairs to exactly
    zero in ``Re(sum(g * v))`` when ``v`` is real.
    """

    def _energy(z):
        return jnp.sum(jnp.abs(z) ** 2) + jnp.real(jnp.sum(z**3))

    def energy_and_grad(A):
        x = jnp.asarray(A.todense())
        g = jax.grad(_energy)(x)
        g = g.real + 1j * imag_scale * g.imag
        return _energy(x), g

    return energy_and_grad


def test_a_complex_state_is_probed_with_a_complex_direction():
    """The exact complex gradient measures as correct.

    Also pins the pairing convention: JAX's complex cotangents pair
    **unconjugated**, so the directional derivative is ``Re(sum(g * v))``.
    Measured against a finite difference, the conjugated form gives an
    unrelated number (0.888 against -1.893), so getting this backwards would
    invent an enormous error on every complex state.
    """
    report = measure_gradient_error(_complex_pair(), _complex_state())

    assert report.relative_error < 1e-6, report.summary()


def test_an_error_only_in_the_imaginary_part_is_detected():
    """The #884 review finding: a real direction cannot see it at all.

    With a real ``v`` the corrupted imaginary components contribute exactly
    zero to ``Re(sum(g * v))``, so this check would report a badly wrong
    gradient as accurate.  The default direction is complex when the state is,
    and this is the regression test for that.
    """
    A = _complex_state()
    report = measure_gradient_error(_complex_pair(imag_scale=4.0), A)

    assert report.is_resolved, report.summary()
    assert report.relative_error > 1e-2, (
        f"an imaginary-part error of 4x measured as {report.relative_error:.2e} "
        f"-- the probe direction is not seeing the imaginary half"
    )

    # And the blindness is real, not hypothetical.  Demonstrated on the
    # arithmetic rather than through the API, which now refuses a real-valued
    # direction outright: with real ``v`` the corrupted gradient's directional
    # derivative still matches the finite difference exactly, so nothing in the
    # comparison could ever notice the 4x error.
    corrupted = _complex_pair(imag_scale=4.0)
    x = np.asarray(A.todense())
    real_v = np.random.RandomState(1).standard_normal(x.shape)
    real_v = real_v / np.linalg.norm(real_v)

    _E, g_bad = corrupted(A)
    blind_analytic = float(np.real(np.sum(np.asarray(g_bad) * real_v)))

    h = 1e-6
    e_plus = float(np.real(corrupted(_wrap(x + h * real_v))[0]))
    e_minus = float(np.real(corrupted(_wrap(x - h * real_v))[0]))
    blind_fd = (e_plus - e_minus) / (2 * h)

    assert blind_analytic == pytest.approx(blind_fd, rel=1e-6), (
        "a real direction was expected to be blind to the imaginary error -- "
        f"g.v={blind_analytic:.6e} against fd={blind_fd:.6e}. If these now "
        "disagree, the blindness argument needs rechecking."
    )


@pytest.mark.parametrize("cast_to_complex", [False, True])
def test_a_real_valued_direction_on_a_complex_state_is_refused(cast_to_complex):
    """Judged on the values, not the dtype.

    A real-valued array cast to ``complex128`` has all-zero imaginary
    components and is exactly as blind as a real one, so accepting it because
    ``iscomplexobj`` is True would guard the dtype rather than the property
    that matters.
    """
    v = np.random.RandomState(2).standard_normal((2, 2, 2, 2, 2))
    if cast_to_complex:
        v = v.astype(complex)

    with pytest.raises(ValueError, match="no imaginary component"):
        measure_gradient_error(_complex_pair(), _complex_state(), direction=v)


def test_a_genuinely_complex_direction_is_accepted():
    """The guard must not reject the directions it exists to require."""
    rng = np.random.RandomState(5)
    v = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))

    report = measure_gradient_error(_complex_pair(), _complex_state(), direction=v)

    assert report.relative_error < 1e-6, report.summary()


# --------------------------------------------------------------------------- #
# 5. Non-finite results fail CLOSED, not into the "good" branch                 #
# --------------------------------------------------------------------------- #


def test_a_nan_gradient_is_reported_not_classified_as_unresolved():
    """The #884 review's second finding, and a defect class this repo knows.

    ``nan > x`` is False, so a NaN reaching the ``is_resolved`` comparison sets
    it to ``False`` -- the branch documented as "the gradient is better than
    the scan can resolve" -- and ``summary()`` then reports an error below a
    NaN floor.  That is #787 (a NaN cotangent reported as 0.0, i.e. perfect)
    and the #772 residual gate failing open, one level up.

    A NaN gradient here is reachable: #772 *was* the asymmetric root-implicit
    engine returning them on a physical simple-update state.
    """

    def nan_grad(A):
        x = jnp.asarray(A.todense())
        g = 4.0 * x**3
        return jnp.sum(x**4), g.at[0, 0, 0, 0, 0].set(jnp.nan)

    with pytest.raises(ValueError, match="non-finite entries"):
        measure_gradient_error(nan_grad, _quartic_state())


def test_an_infinite_gradient_is_refused_too():
    def inf_grad(A):
        x = jnp.asarray(A.todense())
        g = 4.0 * x**3
        return jnp.sum(x**4), g.at[0, 0, 0, 0, 0].set(jnp.inf)

    with pytest.raises(ValueError, match="non-finite entries"):
        measure_gradient_error(inf_grad, _quartic_state())


def test_a_nan_energy_at_a_perturbed_state_is_reported():
    """The gradient can be finite while the shifted energy is not."""
    calls = {"n": 0}

    def nan_at_shift(A):
        x = jnp.asarray(A.todense())
        calls["n"] += 1
        # The unshifted call comes first and must stay finite, so the failure
        # is specifically in the finite difference rather than the gradient.
        energy = jnp.sum(x**4) if calls["n"] == 1 else jnp.nan
        return energy, 4.0 * x**3

    with pytest.raises(ValueError, match="energy at the perturbed state"):
        measure_gradient_error(nan_at_shift, _quartic_state())


# --------------------------------------------------------------------------- #
# 6. Both halves of a complex direction, and steps that survive the state       #
# --------------------------------------------------------------------------- #


def test_a_purely_imaginary_direction_is_refused():
    """The symmetric case to the real-direction one, and equally blind.

    ``Re(sum(g * v))`` with ``v = i*w`` samples only the imaginary coordinates,
    so an arbitrarily corrupted *real* part pairs to exactly zero. The guard
    has to require both halves, not just the one that was found first.
    """
    w = np.random.RandomState(6).standard_normal((2, 2, 2, 2, 2))

    with pytest.raises(ValueError, match="no real component"):
        measure_gradient_error(_complex_pair(), _complex_state(), direction=1j * w)


def test_a_complex_direction_on_a_real_state_is_refused():
    """The shifted state would leave the space the gradient was taken in."""
    rng = np.random.RandomState(7)
    v = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))

    with pytest.raises(ValueError, match="state is real"):
        measure_gradient_error(_exact_pair(), _quartic_state(), direction=v)


def test_steps_that_round_away_against_the_state_are_reported():
    """A shift below the state's ULP measures nothing, and says so.

    With entries around 1e20 the default steps do not change the buffer at all,
    so both signs return the same energy and every difference is exactly zero.
    ``relative_error`` then becomes ``|0 - analytic| / 1e-300`` -- inf -- while
    ``resolution`` is 0, so without this check ``is_resolved`` is True and a
    *correct* gradient is reported as infinitely wrong.
    """
    huge = _wrap(np.full((2, 2, 2, 2, 2), 1e20))

    with pytest.raises(ValueError, match="rounds away"):
        measure_gradient_error(_exact_pair(), huge)


def test_a_state_small_enough_for_the_steps_still_works():
    """The rounds-away guard must not reject ordinary states."""
    report = measure_gradient_error(_exact_pair(1.25), _quartic_state())

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(0.25, rel=1e-3), report.summary()


# --------------------------------------------------------------------------- #
# 7. Same map, same direction, and a step chosen without peeking at the answer  #
# --------------------------------------------------------------------------- #


def test_the_step_is_chosen_without_consulting_the_gradient():
    """Selecting the step by agreement with the gradient is circular.

    For ``E = x^4`` the central difference is ``4 + 4h**2`` *exactly*, so at
    ``x=1``, ``h=0.2`` returns 4.16.  A supplied gradient of 4.16 is 4% wrong,
    and picking the step whose difference best matches it selects h=0.2 and
    reports the wrong gradient as perfect.  The step is chosen by ``|h|`` alone
    instead, and ``resolution`` reports whether that choice was any good.

    The two steps are 20x apart so the separation requirement is satisfied --
    the circularity is about which step is *selected*, not about how close the
    steps are.
    """
    A = _wrap(np.ones((2, 2, 2, 2, 2)))

    def four_percent_wrong(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**4), 4.16 * jnp.ones_like(x)

    report = measure_gradient_error(
        four_percent_wrong, A, direction=np.ones((2, 2, 2, 2, 2)), steps=(0.2, 0.01)
    )

    assert report.step == pytest.approx(0.01), (
        f"expected the smallest magnitude, got h={report.step:.3e}"
    )
    assert report.relative_error > 1e-2, (
        "the h=0.2 truncation lands exactly on the wrong gradient, so a "
        f"selection that consults it reports ~0 -- got {report.summary()}"
    )


def test_a_partially_frozen_shift_is_rejected():
    """One frozen coordinate distorts the direction without freezing the buffer.

    A whole-array equality check cannot see this: the 1e20 entry swallows the
    step while the 1.0 entries move, so the difference follows a different
    direction from the one ``g.v`` is projected along, and a correct gradient
    is reported as wrong.
    """
    mixed = np.ones((2, 2, 2, 2, 2))
    mixed[0, 0, 0, 0, 0] = 1e20

    with pytest.raises(ValueError, match="rounds away"):
        measure_gradient_error(_exact_pair(), _wrap(mixed))


def test_the_shifted_state_keeps_the_input_dtype():
    """float32 in, float32 at every finite-difference evaluation.

    A float64 direction promotes the shifted buffer, and because tenax enables
    x64 it stays promoted -- so the gradient would be taken at one precision
    and the differences evaluated at another, which is a different numerical
    map with different CTM tolerances.
    """
    seen = []

    def recording(t):
        x = jnp.asarray(t.todense())
        seen.append(np.dtype(x.dtype).name)
        return jnp.sum(x**4), 4.0 * x**3

    data = np.random.RandomState(0).standard_normal((2, 2, 2, 2, 2)).astype(np.float32)
    measure_gradient_error(recording, _wrap(data), steps=(1e-3, 1e-4))

    assert set(seen) == {"float32"}, (
        f"the map was evaluated at mixed precisions: {sorted(set(seen))}"
    )
    assert len(seen) == 5, f"expected 1 gradient + 2 steps x 2 signs, got {len(seen)}"


# --------------------------------------------------------------------------- #
# 8. Cancellation in the subtraction, and steps too close to bound each other   #
# --------------------------------------------------------------------------- #


def test_an_energy_offset_that_swallows_the_difference_is_reported():
    """Cancellation in the SUBTRACTION, which the displacement guard cannot see.

    The shifted tensors genuinely differ, but ``ULP(1e20) = 1.6e+04`` swallows
    the ~1e-05 energy change, so every difference is exactly 0, ``resolution``
    is 0, and a *correct* gradient reports as ~1e300 wrong with
    ``is_resolved=True``. The same shape arises near a stationary point.
    """

    def offset_energy(t):
        x = jnp.asarray(t.todense())
        return 1e20 + jnp.sum(x**2), 2.0 * x

    with pytest.raises(ValueError, match="rounding floor of the energy"):
        measure_gradient_error(offset_energy, _wrap(np.ones((2, 2, 2, 2, 2))))


def test_the_same_energy_without_the_offset_measures_fine():
    """The offset is the problem, not the map -- the guard must not over-reject."""

    def plain_energy(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 2.0 * x

    report = measure_gradient_error(plain_energy, _wrap(np.ones((2, 2, 2, 2, 2))))

    assert report.relative_error < 1e-6, report.summary()


@pytest.mark.parametrize("steps", [(0.1, 0.099), (1e-5, 1.5e-5)])
def test_steps_too_close_together_are_refused(steps):
    """Nearby steps share their truncation, so their spread cannot bound it.

    The central difference's error is ``C*h^2``; two steps a factor ``r`` apart
    differ in bias by ``r^2 - 1``, which at ``r = 1.01`` is 2%. Measured on
    ``E = x^4`` with ``(0.1, 0.099)``: spread 1.97e-04 against a *common* error
    of 9.71e-03, so ``resolution`` understates the truncation ~50x and the
    report calls a correct gradient resolvedly wrong.
    """
    with pytest.raises(ValueError, match="separated by at least"):
        measure_gradient_error(_exact_pair(), _quartic_state(), steps=steps)


def test_well_separated_steps_are_accepted():
    """The separation guard must not reject the default scan."""
    report = measure_gradient_error(
        _exact_pair(1.4), _quartic_state(), steps=(1e-4, 1e-6)
    )

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(0.4, rel=1e-3), report.summary()


# --------------------------------------------------------------------------- #
# 9. The guards must not reject VALID scans                                     #
# --------------------------------------------------------------------------- #


def test_a_state_with_tiny_entries_is_not_mistaken_for_rounding_away():
    """Rounding scales with the perturbed value, not with ``base`` alone.

    At ``base = 3.85e-08`` with ``h*v = 1e-05`` the realised shift is off by
    1.69e-21 -- ordinary rounding of a sum dominated by the *step*. A bound of
    ``4*eps*|base|`` is 3.42e-23 there, so the scan was refused as "rounds
    away" despite being perfectly good.
    """
    # This value is not arbitrary: with a normalised all-ones direction and
    # h=1e-5 it puts the realised-vs-intended drift at 2.12e-22, above the
    # 2.87e-23 that ``4*eps*|base|`` allows and below the 1.57e-21 that
    # ``4*eps*max(|base|, |intended|)`` allows. A value picked by eye rounds
    # exactly and exercises nothing -- the first draft of this test did, and a
    # mutant reverting the bound survived it.
    tiny = np.full((2, 2, 2, 2, 2), 3.2321179903596804e-08)

    # A quadratic, so the central difference is EXACT at any step and the only
    # thing under test is whether the rounding guard lets the scan run.  With a
    # quartic the h >> x regime is dominated by higher-order terms and the scan
    # honestly reports "unresolvable", which would not isolate the guard.
    def quadratic(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 2.0 * x

    report = measure_gradient_error(
        quadratic,
        _wrap(tiny),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1e-5, 1e-7),
    )

    assert report.relative_error < 1e-6, report.summary()


def test_an_energy_returned_at_higher_precision_than_the_state_is_allowed():
    """The cancellation floor belongs to the energy's precision, not the state's.

    A float32 state whose map evaluates in float64 has a real rounding floor of
    ~8.9e-10 at ``|E| ~ 1e6``; charging it the float32 floor of 4.8e-01 rejects
    a span of 4e-03 that is seven orders above the noise.
    """

    def f64_energy_from_f32_state(t):
        x = jnp.asarray(t.todense())
        x64 = x.astype(jnp.float64)
        return 1e6 + jnp.sum(x64**2), (2.0 * x64).astype(x.dtype)

    data = np.full((2, 2, 2, 2, 2), 0.5, dtype=np.float32)

    report = measure_gradient_error(
        f64_energy_from_f32_state, _wrap(data), steps=(1e-3, 1e-4)
    )

    assert report.relative_error < 1e-2, report.summary()


# --------------------------------------------------------------------------- #
# 10. A malformed gradient, and one bad step among good ones                    #
# --------------------------------------------------------------------------- #


def test_a_broadcastable_gradient_is_refused_rather_than_projected():
    """``grad * v`` would broadcast a scalar into a plausible answer.

    For ``E = sum(x)`` the true gradient is all-ones, and a returned scalar
    ``1`` projects to *exactly* the same directional derivative -- so a
    callback that never produced a tensor-shaped gradient reports as perfectly
    accurate.
    """

    def scalar_grad(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x), jnp.asarray(1.0)

    with pytest.raises(ValueError, match="expected"):
        measure_gradient_error(scalar_grad, _quartic_state())


def test_one_unusable_step_does_not_discard_the_whole_scan():
    """Two well-separated steps survive; the third is skipped, not fatal.

    With ``E = 5e10 + sum(x**2)`` the energy floor is 4.4e-05. The spans are
    2.3e-03 at h=1e-4 and 2.3e-04 at h=1e-5 (both usable) but 2.3e-05 at
    h=1e-6, which falls below it. Aborting there would throw away a perfectly
    good 10x-separated scan.
    """

    def offset_quadratic(t):
        x = jnp.asarray(t.todense())
        return 5e10 + jnp.sum(x**2), 2.0 * x

    report = measure_gradient_error(
        offset_quadratic,
        _wrap(np.ones((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1e-4, 1e-5, 1e-6),
    )

    assert report.step == pytest.approx(1e-5), (
        f"h=1e-6 should have been skipped, leaving 1e-5 as the smallest "
        f"usable magnitude; got {report.step:.1e}"
    )
    # The surviving span is only ~5x above the energy floor, so the difference
    # is genuinely noisy and the scan says so.  What this test asserts is that
    # a report exists at all and is CONSISTENT with a correct gradient -- the
    # error sits within the scan's own resolution rather than standing clear of
    # it.  Demanding a tight number here would be demanding precision the
    # offset destroyed.
    assert not report.is_resolved, report.summary()
    assert report.relative_error <= 2.0 * report.resolution, report.summary()


def test_the_scan_still_fails_when_too_few_steps_survive():
    """Skipping is not the same as tolerating: two separated steps are required.

    Here every step but one falls below the floor, so nothing can bound the
    resolution and the aggregate error names each dropped step.
    """

    def big_offset(t):
        x = jnp.asarray(t.todense())
        return 1e18 + jnp.sum(x**2), 2.0 * x

    with pytest.raises(ValueError, match="usable step magnitude"):
        measure_gradient_error(
            big_offset,
            _wrap(np.ones((2, 2, 2, 2, 2))),
            direction=np.ones((2, 2, 2, 2, 2)),
            steps=(1e-4, 1e-5, 1e-6),
        )


# --------------------------------------------------------------------------- #
# 11. The unperturbed state, and the knob that decides how the report reads     #
# --------------------------------------------------------------------------- #


def test_a_non_finite_unperturbed_energy_is_rejected():
    """The map must be defined at the state whose gradient is measured.

    A NaN here with finite neighbours would otherwise yield a confident report
    about a map that is undefined exactly where it was asked.
    """
    calls = {"n": 0}

    def nan_at_base(t):
        x = jnp.asarray(t.todense())
        calls["n"] += 1
        energy = jnp.nan if calls["n"] == 1 else jnp.sum(x**4)
        return energy, 4.0 * x**3

    with pytest.raises(ValueError, match="unperturbed state"):
        measure_gradient_error(nan_at_base, _quartic_state())


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("-inf")])
def test_an_unusable_spread_tolerance_is_rejected(bad):
    """``fd_spread_tol`` decides which branch the report is read as.

    Negative makes ``best_rel > tol * resolution`` true for essentially any
    error, so everything reads as a resolved measurement; NaN makes it false
    always, so everything lands in the branch that means "better than the scan
    can resolve" -- the one that reads as good news.
    """
    with pytest.raises(ValueError, match="fd_spread_tol"):
        measure_gradient_error(_exact_pair(), _quartic_state(), fd_spread_tol=bad)


def test_zero_spread_tolerance_is_allowed():
    """Zero is degenerate but coherent: any nonzero error counts as resolved."""
    report = measure_gradient_error(
        _exact_pair(1.5), _quartic_state(), fd_spread_tol=0.0
    )

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(0.5, rel=1e-3), report.summary()


def test_the_advertised_cost_matches_the_number_of_evaluations():
    """The docstring promises seven convergences by default; count them.

    The module docstring said "four" while the function docstring said "six",
    and neither counted the unperturbed call -- a stale claim someone would
    budget a run against.
    """
    calls = {"n": 0}

    def counting(t):
        x = jnp.asarray(t.todense())
        calls["n"] += 1
        return jnp.sum(x**4), 4.0 * x**3

    measure_gradient_error(counting, _quartic_state())

    assert calls["n"] == 7, (
        f"default scan made {calls['n']} evaluations, not the documented 7 "
        "(1 unperturbed + 2 per step x 3 steps)"
    )


# --------------------------------------------------------------------------- #
# 12. Complex energies, imaginary gradients on real states                      #
# --------------------------------------------------------------------------- #


def test_a_complex_energy_with_a_nan_imaginary_part_is_rejected():
    """``np.real`` before the finiteness check accepts ``complex(finite, nan)``.

    Every energy in the scan could then be non-finite while the report reads as
    ordinary -- the check has to see the whole scalar first.
    """

    def nan_imag_energy(t):
        x = jnp.asarray(t.todense())
        return jnp.asarray(complex(float(jnp.sum(x**4)), float("nan"))), 4.0 * x**3

    with pytest.raises(ValueError, match="unperturbed state"):
        measure_gradient_error(nan_imag_energy, _quartic_state())


def test_an_imaginary_gradient_on_a_real_state_is_rejected():
    """``Re(sum(g * v))`` would drop it silently.

    A cotangent for a real input space has no imaginary part, so ``2x + 1j*1e30``
    is a malformed gradient rather than one to project -- and projecting it
    reports the gradient as accurate.
    """

    def imag_grad(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 2.0 * x + 1j * 1e30

    with pytest.raises(ValueError, match="imaginary part"):
        measure_gradient_error(imag_grad, _quartic_state())


def test_a_real_valued_complex_dtype_gradient_on_a_real_state_is_allowed():
    """Zero imaginary content is a real cotangent, whatever the dtype says.

    The guard is on the values, like the direction guard -- rejecting on dtype
    alone would refuse a legitimate gradient that merely came back promoted.
    """

    def complex_dtype_grad(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), (2.0 * x).astype(jnp.complex128)

    report = measure_gradient_error(complex_dtype_grad, _quartic_state())

    assert report.relative_error < 1e-6, report.summary()


def test_a_frozen_coordinate_that_dominates_the_projection_is_caught():
    """The freeze check cannot key off the direction alone.

    ``A[0]=1e20`` with ``v[0]=1e-13`` and ``g[0]=1e20``: that coordinate carries
    ~1e6 of ``g.v`` while sitting far below any direction-relative cutoff, so a
    direction-only mask excludes it. It is frozen at every shifted state, its
    contribution drops out of the difference but stays in ``analytic``, and the
    correct gradient reads as resolvedly wrong.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e20
    weights = np.ones(n)
    weights[0] = 1e20

    def affine(t):
        x = jnp.asarray(t.todense()).reshape(-1)
        return jnp.sum(jnp.asarray(weights) * x), jnp.asarray(weights).reshape(
            (2, 2, 2, 2, 2)
        )

    v = np.ones(n)
    v[0] = 1e-13

    with pytest.raises(ValueError, match="rounds away"):
        measure_gradient_error(
            affine,
            _wrap(base.reshape((2, 2, 2, 2, 2))),
            direction=v.reshape((2, 2, 2, 2, 2)),
        )


def test_a_coordinate_dominating_the_signed_projection_after_cancellation_is_caught():
    """The L1-share mask missed this; the projection test does not.

    Contributions ``g_i*v_i`` of 1e-13, 1 and -1+5e-14 sum to 1.5e-13 while
    their L1 total is 2, so the first coordinate supplies two thirds of
    ``analytic`` yet is 5e-14 of the L1 -- excluded by any share-of-total
    cutoff. Freezing it with a large base then drops that contribution from the
    difference while it stays in ``analytic``.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e18  # large enough to freeze coordinate 0 at every step
    weights = np.zeros(n)
    weights[0] = 1e-13
    weights[1] = 1.0
    weights[2] = -1.0 + 5e-14

    def affine(t):
        x = jnp.asarray(t.todense()).reshape(-1)
        return jnp.sum(jnp.asarray(weights) * x), jnp.asarray(weights).reshape(
            (2, 2, 2, 2, 2)
        )

    with pytest.raises(ValueError, match="rounds away"):
        measure_gradient_error(
            affine,
            _wrap(base.reshape((2, 2, 2, 2, 2))),
            direction=np.ones((2, 2, 2, 2, 2)),
        )


def test_the_unresolved_summary_quotes_the_bound_it_compared_against():
    """``is_resolved`` tests against ``fd_spread_tol * resolution``.

    Quoting ``resolution`` alone would claim the gradient is ``fd_spread_tol``
    times more accurate than the scan established.
    """
    report = measure_gradient_error(_exact_pair(), _quartic_state())

    assert not report.is_resolved, report.summary()
    assert report.unresolved_bound == pytest.approx(10.0 * report.resolution)
    assert f"{report.unresolved_bound:.2e}" in report.summary()
    assert "below the measurement resolution" not in report.summary()


def test_the_displacement_is_validated_without_using_the_gradient():
    """The experiment cannot be validated by the quantity it is testing.

    The projection check weights by ``grad_arr``, so a gradient that is wrong
    by being *zero* on a frozen coordinate makes both projections omit it and
    the drift reads as ~0. Measured on this state: 3.1e-02 with the true
    gradient, 5.4e-12 with the returned one, while the gradient-free norm check
    reads 1.8e-01 either way.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e20

    def centred_affine_with_hole(t):
        x = jnp.asarray(t.todense()).reshape(-1)
        g = np.ones(n)
        g[0] = 0.0  # wrong exactly where the step is about to freeze
        return jnp.sum(x - jnp.asarray(base)), jnp.asarray(g).reshape((2, 2, 2, 2, 2))

    with pytest.raises(ValueError, match="rounds away"):
        measure_gradient_error(
            centred_affine_with_hole,
            _wrap(base.reshape((2, 2, 2, 2, 2))),
            direction=np.ones((2, 2, 2, 2, 2)),
        )
