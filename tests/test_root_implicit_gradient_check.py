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


def test_zero_spread_tolerance_is_refused():
    """Zero bypasses the floor, including the divergence folded into it.

    ``resolved = rel > tol * resolution`` with ``tol = 0`` marks every nonzero
    error resolved however badly the differences diverge: the exact gradient of
    ``sum(x**3)`` at zero has ``relative_error`` 1 and ``fd_divergence`` 0.9999
    and would be declared definitively 100% wrong. It was legal until the
    divergence moved into the floor and the separate convergence gate went
    away -- so this is a consequence of that simplification, not an oversight
    in it.
    """
    with pytest.raises(ValueError, match="strictly positive"):
        measure_gradient_error(_exact_pair(1.5), _quartic_state(), fd_spread_tol=0.0)


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


# --------------------------------------------------------------------------- #
# 13. Compared along the direction the difference actually followed             #
# --------------------------------------------------------------------------- #


def test_a_frozen_coordinate_does_not_create_a_phantom_error():
    """A distorted direction is measured along, not bounded or refused.

    ``base[0]=1e20`` swallows the step, so the difference follows a direction
    without that coordinate. Projecting the gradient along the *intended* one
    then invents an error that is not the gradient's -- earlier versions
    reported this correct gradient as resolvedly wrong, then refused the scan
    outright. Both projections now use the realised direction, so the answer is
    simply right.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e20
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def centred_affine(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), jnp.ones((2, 2, 2, 2, 2))

    report = measure_gradient_error(
        centred_affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
    )

    assert report.relative_error < 1e-9, report.summary()
    assert report.direction_distortion > 1e-2, (
        "the probe direction was distorted by a frozen coordinate and the "
        f"report does not say so -- {report.summary()}"
    )


def test_the_distortion_says_which_subspace_was_probed():
    """It is information, not an accuracy floor.

    A coordinate the arithmetic cannot move is simply not tested, and no floor
    repairs that -- three earlier attempts to bound it all failed, because a
    gradient can always align with the residual. Reporting which direction was
    actually probed is the honest alternative, so an undistorted scan must read
    ~0 and a distorted one must not.
    """
    clean = measure_gradient_error(_exact_pair(), _quartic_state())
    assert clean.direction_distortion < 1e-9, clean.summary()

    n = 32
    base = np.ones(n)
    base[0] = 1e20
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def centred_affine(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), jnp.ones((2, 2, 2, 2, 2))

    distorted = measure_gradient_error(
        centred_affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
    )
    assert distorted.direction_distortion == pytest.approx(1 / np.sqrt(n), rel=1e-6), (
        "one frozen coordinate out of 32 should distort the unit direction by "
        f"1/sqrt(32) -- got {distorted.direction_distortion:.3e}"
    )


def test_a_wrong_gradient_is_still_caught_on_a_distorted_direction():
    """Measuring along the realised direction must not blunt the check.

    The gradient is scaled by 1.5 on every coordinate, so it is wrong along any
    direction -- including the distorted one.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e20
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def wrong_by_half(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), 1.5 * jnp.ones((2, 2, 2, 2, 2))

    report = measure_gradient_error(
        wrong_by_half,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
    )

    assert report.is_resolved, report.summary()
    assert report.relative_error == pytest.approx(0.5, rel=1e-6), report.summary()


def test_steps_probing_different_directions_still_give_a_sound_answer():
    """Each step is compared along its own direction, so they need not agree.

    ``base[0]=1e14`` moves at h=1 and freezes at h=1e-2 -- ULP(1e14) is 1.56e-02,
    between the two displacements -- so the steps probe genuinely different
    directions (unit distance 1.73e-01). Pooling their *differences* was
    unsound at any tolerance; pooling their *relative errors* is not, because
    each is measured against its own ``analytic_h``.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e14
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))
    weights = np.ones(n)
    weights[0] = 100.0
    w_j = jnp.asarray(weights.reshape((2, 2, 2, 2, 2)))

    def weighted_affine(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(w_j * (x - base_j)), w_j

    report = measure_gradient_error(
        weighted_affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1.0, 1e-2),
    )

    assert report.relative_error < 1e-9, (
        f"a correct gradient must measure correctly even when the steps probe "
        f"different directions -- {report.summary()}"
    )


def test_a_uniform_error_is_resolved_across_differing_directions():
    """A gradient uniformly 5% high reads 0.05 along ANY direction.

    That is what makes the relative errors poolable where the differences were
    not: on float32 with the default steps the unit directions drift from
    5.2e-05 to 4.4e-02 apart, and pooling derivatives let that drift mark the
    real error unresolved.
    """
    data = np.full((2, 2, 2, 2, 2), 0.5, dtype=np.float32)

    def five_percent_high(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 1.05 * 2.0 * x

    report = measure_gradient_error(five_percent_high, _wrap(data), steps=(1e-2, 1e-3))

    assert report.relative_error == pytest.approx(0.05, rel=5e-2), report.summary()
    assert report.is_resolved, report.summary()


def test_ordinary_rounding_still_counts_as_the_same_direction():
    """The guard must not reject scans whose steps merely round differently.

    Every step distorts the direction slightly -- float32 reaches ~1e-2 on the
    smallest components -- so the tolerance has to sit above that or ordinary
    float32 scans are refused.
    """
    data = np.full((2, 2, 2, 2, 2), 0.5, dtype=np.float32)

    def quadratic(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 2.0 * x

    report = measure_gradient_error(quadratic, _wrap(data), steps=(1e-3, 1e-4))

    assert report.relative_error < 1e-2, report.summary()


def test_a_small_real_error_is_not_hidden_by_effective_norm_drift():
    """Steps with parallel but differently-scaled displacements are commensurable.

    The direction guard makes the unit directions agree, but rounding still
    changes the effective *norm* between steps. Pooling raw differences then
    reads that scaling difference as noise: on this float32 state it produced a
    resolution of ~1.0e-02 and marked a genuine 5% error unresolved against a
    10.4% bound, even though every per-step comparison measured exactly 5%.
    """
    data = np.full((2, 2, 2, 2, 2), 0.5, dtype=np.float32)

    def five_percent_high(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2), 1.05 * 2.0 * x

    # Steps sized for float32: the default 1e-6 is ~3 ULPs on a 0.5 entry, so
    # the displacement is quantised and the scan is roundoff-dominated for
    # reasons that have nothing to do with the pooling under test.
    report = measure_gradient_error(
        five_percent_high,
        _wrap(data),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1e-2, 1e-3),
    )

    assert report.relative_error == pytest.approx(0.05, rel=1e-2), report.summary()
    assert report.is_resolved, (
        "a real 5% error was hidden by effective-norm drift between steps -- "
        f"{report.summary()}"
    )


def test_resolution_pools_relative_errors_not_differences():
    """The pooled quantity has to be direction-local, not a derivative.

    ``base = 1e8`` with steps ``(1e-4, 1e-6)`` quantises the fine step's
    displacement, so the two effective norms differ by ~1% while the gradient
    is uniformly 5% high. Pooling the differences reads that 1% as noise --
    measured ``resolution`` 1.01e-02, which marks the real 5% error unresolved.
    Pooling the relative errors gives 0.00, because each is measured against
    its own step's projection and a uniform 5% error is 5% along any direction.
    """
    n = 32
    base = np.full(n, 1e8)
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def five_percent_high(t):
        x = jnp.asarray(t.todense())
        d = x - base_j
        # Algebraically sum(x^2 - base^2), written to avoid forming x^2 ~ 1e16
        # and losing the 1e8-scale difference to cancellation -- that would put
        # ~5.5e-02 in relative_error for reasons unrelated to the 5% error.
        return jnp.sum(d * d + 2.0 * base_j * d), 1.05 * 2.0 * x

    # A uniform direction, so the two steps stay commensurable (pairwise unit
    # distance 1.6e-16 against 2.5e-02 for the default random one). Without
    # that they carry no convergence evidence and the scan is unresolved by
    # construction, which would test the wrong thing.
    report = measure_gradient_error(
        five_percent_high,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1e-4, 1e-6),
    )

    assert report.relative_error == pytest.approx(0.05, rel=1e-3), report.summary()
    assert report.is_resolved, (
        "a real 5% error was marked unresolved by a floor built from the "
        f"differences rather than the relative errors -- {report.summary()}"
    )


def test_an_unconverged_scan_is_not_resolved_by_agreeing_errors():
    """The relative errors share the gradient, so they can agree while lying.

    ``E = sum(x**3)`` at ``x = 0`` has an exact gradient of zero and every
    central difference proportional to ``h**2``: each ``rel`` is exactly 1 and
    their spread is 0, so pooling them alone reports an *exact* gradient as a
    resolved 100% error. The differences themselves span 1e4x across the
    default steps, which is the gradient-independent signal that catches it.
    """
    zero_state = _wrap(np.zeros((2, 2, 2, 2, 2)))

    def cubic(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**3), 3.0 * x**2  # exact: zero at x = 0

    report = measure_gradient_error(cubic, zero_state)

    assert report.fd_divergence > 0.25, report.summary()
    assert not report.is_resolved, (
        "an exact gradient was reported as a resolved 100% error because the "
        f"relative errors agreed while the scan had not converged -- "
        f"{report.summary()}"
    )
    assert "set by the differences" in report.summary(), report.summary()


def test_a_converged_scan_is_not_blocked_by_the_convergence_signal():
    """The bound must not reinstate the false-unresolved behaviour.

    The #884 norm-drift case has ``fd_divergence`` ~1.0e-02 -- real scale drift,
    not non-convergence -- and must still resolve its genuine 5% error.
    """
    n = 32
    base = np.full(n, 1e8)
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def five_percent_high(t):
        x = jnp.asarray(t.todense())
        d = x - base_j
        return jnp.sum(d * d + 2.0 * base_j * d), 1.05 * 2.0 * x

    # A uniform direction, so the two steps stay commensurable (pairwise unit
    # distance 1.6e-16 against 2.5e-02 for the default random one). Without
    # that they carry no convergence evidence and the scan is unresolved by
    # construction, which would test the wrong thing.
    report = measure_gradient_error(
        five_percent_high,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1e-4, 1e-6),
    )

    # Its two steps are not commensurable -- the fine one quantises, so the
    # realised directions differ by more than the grouping tolerance -- and the
    # convergence check therefore stands down rather than blocking on an
    # artefact. ``nan`` here means "no evidence", not "diverged".
    assert np.isnan(report.fd_divergence) or report.fd_divergence < 0.25, (
        report.summary()
    )
    assert report.is_resolved, report.summary()


def test_incommensurable_steps_leave_the_result_unresolved():
    """No commensurable pair means no convergence evidence, so nothing is settled.

    ``base[0]=1e14`` moves at h=1 and freezes at h=1e-2, so the differences are
    different directional derivatives. The relative errors still agree -- 0.5 at
    both steps -- but agreement among them proves nothing on its own, since they
    share the gradient under test: ``sum((x-base)**3)`` on the same fixture has
    an exact gradient and produces a consistent ``rel`` of 1.

    So the error is reported and the result stays unresolved.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e14
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))
    weights = np.ones(n)
    weights[0] = 100.0
    w_j = jnp.asarray(weights.reshape((2, 2, 2, 2, 2)))

    def half_too_high(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(w_j * (x - base_j)), 1.5 * w_j

    report = measure_gradient_error(
        half_too_high,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1.0, 1e-2),
    )

    assert np.isnan(report.fd_divergence), report.summary()
    assert not report.is_resolved, (
        "no convergence evidence must not license a definitive measurement -- "
        f"{report.summary()}"
    )
    assert report.relative_error == pytest.approx(0.5, rel=1e-6), report.summary()


def test_an_exact_gradient_is_not_declared_wrong_without_evidence():
    """The counter-example that makes the rule necessary.

    Same incommensurable fixture, but a cubic whose exact gradient is zero at
    ``base``: every ``rel`` is exactly 1 with zero spread. Treating "no
    evidence" as converged reported an EXACT gradient as a definitive 100%
    error.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e14
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def cubic_at_base(t):
        x = jnp.asarray(t.todense())
        d = x - base_j
        return jnp.sum(d**3), 3.0 * d**2  # exact, and zero at base

    report = measure_gradient_error(
        cubic_at_base,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1.0, 1e-2),
    )

    assert not report.is_resolved, (
        f"an exact gradient was declared definitively wrong -- {report.summary()}"
    )


def test_an_unconverged_bound_reflects_the_divergence():
    """A non-converged scan must not advertise a tiny bound.

    The relative errors agree to ~1e-16 here, so pooling them alone gave
    ``unresolved_bound`` ~1e-15 beside a 100% error -- a bound fifteen orders
    too optimistic. Folding ``fd_divergence`` into the floor fixes that at the
    source: the bound now reflects how far the derivative is still moving with
    h, rather than needing to be read together with a separate flag.
    """
    zero_state = _wrap(np.zeros((2, 2, 2, 2, 2)))

    def cubic(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**3), 3.0 * x**2

    report = measure_gradient_error(cubic, zero_state)

    assert not report.is_resolved, report.summary()
    assert report.fd_divergence > 0.25, report.summary()
    assert report.unresolved_bound > 1.0, (
        "the bound must carry the divergence, not sit at the relative errors' "
        f"own agreement -- {report.summary()}"
    )
    assert "set by the differences" in report.summary(), report.summary()


def test_the_reported_step_comes_from_the_group_the_evidence_is_about():
    """``best_group`` can exclude the smallest |h|, leaving the report unchecked.

    With ``base[0]=1e14`` and steps ``(0.5, 0.25, 0.125)`` the two coarse steps
    realise one direction and the finest realises another, so the convergence
    evidence covers the coarse pair while the reported number came from the
    step nothing had examined.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e14
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def affine(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), jnp.ones((2, 2, 2, 2, 2))

    report = measure_gradient_error(
        affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(0.5, 0.25, 0.125),
    )

    assert report.step in (0.5, 0.25), (
        "the reported step must come from the commensurable group the "
        f"convergence check examined, got h={report.step:.3g}"
    )


def test_an_indeterminate_scan_says_so_rather_than_quoting_a_floor():
    """NaN divergence is indeterminate, and the summary must not imply a bound."""
    n = 32
    base = np.ones(n)
    base[0] = 1e14
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def half_too_high(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), 1.5 * jnp.ones((2, 2, 2, 2, 2))

    report = measure_gradient_error(
        half_too_high,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1.0, 1e-2),
    )

    assert np.isnan(report.fd_divergence), report.summary()
    assert not report.is_resolved, report.summary()
    assert "indeterminate" in report.summary(), report.summary()


def test_duplicate_magnitudes_are_not_convergence_evidence():
    """``(0.5, -0.5)`` is one central difference counted twice.

    Its divergence is identically zero, which would manufacture convergence
    evidence from a single measurement. The up-front ``steps`` check cannot
    catch this: the whole set has two separated magnitudes, but the
    commensurable *subset* does not.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e14  # 0.5/-0.5 realise one direction, 0.125 another
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def cubic_at_base(t):
        x = jnp.asarray(t.todense())
        d = x - base_j
        return jnp.sum(d**3), 3.0 * d**2  # exact, zero at base

    report = measure_gradient_error(
        cubic_at_base,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(0.5, -0.5, 0.125),
    )

    assert not report.is_resolved, (
        "a duplicated coarse step supplied its own convergence evidence and "
        f"declared an exact gradient 100% wrong -- {report.summary()}"
    )


def test_resolution_is_scoped_to_the_evidence_group():
    """A step outside the group must not inflate the spread.

    ``base[0]=1e14`` puts the two coarse steps on one realised direction and
    the fine step on another. The fine step's relative error is unrelated to
    the reported measurement, and pooling it could mark a cleanly measured
    error unresolved.
    """
    n = 32
    base = np.ones(n)
    # 1e12: the coarse pair agrees to 5.6e-17 in unit direction while the fine
    # step sits 1.8e-01 away, so the group is exactly {1.0, 0.5}.
    base[0] = 1e12
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))
    weights = np.ones(n)
    weights[0] = 100.0
    w_j = jnp.asarray(weights.reshape((2, 2, 2, 2, 2)))

    # The error lives ONLY on the coordinate the fine step freezes, so the
    # coarse group measures it while the out-of-group step sees a correct
    # gradient. Pooling that unrelated rel inflates the spread and blocks a
    # cleanly measured error -- a uniform error would not show this, because
    # every step would report the same rel.
    wrong = np.ones(n)
    wrong[0] = 10.0 * weights[0]
    wrong_j = jnp.asarray(wrong.reshape((2, 2, 2, 2, 2)))

    def wrong_on_the_big_coordinate(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(w_j * (x - base_j)), wrong_j

    report = measure_gradient_error(
        wrong_on_the_big_coordinate,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        direction=np.ones((2, 2, 2, 2, 2)),
        steps=(1.0, 0.5, 1e-4),
    )

    assert report.step in (1.0, 0.5), report.summary()
    assert report.is_resolved, (
        "the out-of-group fine step inflated the spread and blocked a cleanly "
        f"measured error -- {report.summary()}"
    )


def test_a_partly_converged_scan_is_not_called_measured():
    """A fixed convergence cutoff can always be met by construction.

    Two commensurable steps whose normalised differences are 1.2 and 1.0 give a
    divergence of 0.167 -- under any 0.25-style cutoff -- with agreeing relative
    errors of 0.0909, while the true derivative is 0.5 and the supplied
    gradient is ~118% wrong. Folding the divergence into the floor removes the
    cutoff entirely: how far the derivative is still moving with h is itself a
    limit on what can be attributed to the gradient.
    """
    # E(q) = 0.5q + (73/30)q^3 - (26/15)q^5 along a single coordinate, so the
    # central differences at h=1 and h=0.5 are exactly 1.2 and 1.0.
    base = np.zeros((2, 2, 2, 2, 2))
    e = np.zeros((2, 2, 2, 2, 2))
    e[0, 0, 0, 0, 0] = 1.0
    e_j = jnp.asarray(e)

    def tuned(t):
        q = jnp.sum(jnp.asarray(t.todense()) * e_j)
        energy = 0.5 * q + (73.0 / 30.0) * q**3 - (26.0 / 15.0) * q**5
        return energy, (12.0 / 11.0) * e_j

    report = measure_gradient_error(tuned, _wrap(base), direction=e, steps=(1.0, 0.5))

    assert report.fd_divergence == pytest.approx(1.0 / 6.0, rel=1e-3), report.summary()
    assert not report.is_resolved, (
        "a scan whose derivative is still moving 17% between steps must not "
        f"yield a definitive error -- {report.summary()}"
    )
    # 0.167 sits below the retired 0.25 gate, so this also pins that the
    # summary explains the floor without one: a threshold here would make
    # 0.24 and 0.26 read qualitatively differently after the classifier
    # stopped using a cutoff at all.
    assert "set by the differences" in report.summary(), report.summary()


def test_the_report_does_not_depend_on_the_order_of_steps():
    """Equal-sized commensurable groups must not be settled by encounter order.

    With ``base[0]=1e15`` the magnitudes split into two disjoint groups of the
    same size. Retaining whichever came first made the answer depend on how
    ``steps`` happened to be written -- ``(4, 2, .25, .125)`` reported h=2 and
    error 0.2, reversed it reported h=.125 and 9.8e-04. Ties now go to the
    finest step, matching the documented selection policy.
    """
    n = 32
    base = np.ones(n)
    base[0] = 1e15
    base_j = jnp.asarray(base.reshape((2, 2, 2, 2, 2)))

    def affine(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x - base_j), 1.2 * jnp.ones((2, 2, 2, 2, 2))

    kw = dict(
        direction=np.ones((2, 2, 2, 2, 2)),
    )
    forward = measure_gradient_error(
        affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        steps=(4.0, 2.0, 0.25, 0.125),
        **kw,
    )
    reverse = measure_gradient_error(
        affine,
        _wrap(base.reshape((2, 2, 2, 2, 2))),
        steps=(0.125, 0.25, 2.0, 4.0),
        **kw,
    )

    assert forward.step == reverse.step, (
        f"reordering steps changed the reported step: {forward.summary()} vs "
        f"{reverse.summary()}"
    )
    assert forward.relative_error == pytest.approx(reverse.relative_error), (
        f"reordering steps changed the answer: {forward.summary()} vs "
        f"{reverse.summary()}"
    )
    assert forward.step == pytest.approx(0.125), (
        f"ties must go to the finest step, got h={forward.step:.3g}"
    )


def test_a_complex_energy_is_refused_rather_than_reduced_to_its_real_part():
    """``float(np.real(...))`` would validate the gradient of Re(E) alone.

    ``E = sum(x**2) + 1j*sum(x)`` with the gradient ``2x`` reports as accurate
    although that gradient does not differentiate the callable's output.
    """

    def complex_energy(t):
        x = jnp.asarray(t.todense())
        return jnp.sum(x**2) + 1j * jnp.sum(x), 2.0 * x

    with pytest.raises(ValueError, match="imaginary part"):
        measure_gradient_error(complex_energy, _quartic_state())


def test_roundoff_sized_imaginary_residue_is_tolerated():
    """A real objective computed in complex arithmetic is still real.

    The guard has to be relative and well above roundoff, or every complex-dtype
    CTM energy would be refused for carrying a ~1e-16 residue.
    """

    def nearly_real(t):
        x = jnp.asarray(t.todense())
        e = jnp.sum(x**2)
        return e + 1j * e * 1e-15, 2.0 * x

    report = measure_gradient_error(nearly_real, _quartic_state())

    assert report.relative_error < 1e-6, report.summary()


def test_a_real_offset_cannot_hide_an_imaginary_derivative():
    """The guard is on the imaginary VARIATION, not its size.

    ``1e20 + 1e10*sum(x) + 1j*5e9*sum(x)`` has an imaginary value only 5e-11 of
    the real offset, so a value-scaled threshold passes it -- and the scan would
    then differentiate the real part alone while accepting a gradient that omits
    an imaginary derivative half as large. A constant cancels in a difference,
    which is why the difference is what gets checked.
    """
    ones = jnp.ones((2, 2, 2, 2, 2))

    def offset_complex(t):
        x = jnp.asarray(t.todense())
        q = jnp.sum(x)
        return 1e20 + 1e10 * q + 1j * 5e9 * q, 1e10 * ones

    with pytest.raises(ValueError, match="imaginary part varies"):
        measure_gradient_error(
            offset_complex, _wrap(np.ones((2, 2, 2, 2, 2))), steps=(1e-2, 1e-4)
        )


def test_a_large_real_offset_alone_still_measures():
    """The same objective without the imaginary term must still work.

    Otherwise the guard would be rejecting the offset rather than the complex
    part -- the failure mode the previous version had in reverse.
    """
    ones = jnp.ones((2, 2, 2, 2, 2))

    def offset_real(t):
        x = jnp.asarray(t.todense())
        return 1e20 + 1e10 * jnp.sum(x), 1e10 * ones

    report = measure_gradient_error(
        offset_real, _wrap(np.ones((2, 2, 2, 2, 2))), steps=(1e-2, 1e-4)
    )

    # It measures rather than raising. The number is not tight -- |E| ~ 1e20
    # swamps a 1e10-scale difference, so the scan honestly reports the error as
    # sitting at its own floor -- but that is precision, not the complex guard.
    assert report.relative_error <= report.unresolved_bound, report.summary()


def test_a_rounding_floor_cannot_hide_an_imaginary_derivative_either():
    """The absolute floor was the same offset defect one level down.

    ``1e20 + 1e6*sum(x) + 1j*3e5*sum(x)`` at steps ``(2e-2, 1e-2)`` keeps both
    imaginary spans (~1.2e+04) under a ``4*eps*|E|`` floor of ~8.9e+04, so the
    scan accepted a gradient omitting an imaginary derivative 30% the size of
    the real one. The floor is gone: the comparison is against the real span,
    which the cancellation check has already shown to be above the noise.
    """
    ones = jnp.ones((2, 2, 2, 2, 2))

    def offset_complex(t):
        x = jnp.asarray(t.todense())
        q = jnp.sum(x)
        return 1e20 + 1e6 * q + 1j * 3e5 * q, 1e6 * ones

    with pytest.raises(ValueError, match="imaginary part varies"):
        measure_gradient_error(
            offset_complex,
            _wrap(np.ones((2, 2, 2, 2, 2))),
            direction=np.ones((2, 2, 2, 2, 2)),
            steps=(2e-2, 1e-2),
        )
