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


def test_the_direction_is_normalised_so_the_error_is_scale_free():
    """A caller-supplied direction of any length gives the same relative error."""
    A = _quartic_state()
    v = np.random.RandomState(3).standard_normal(A.todense().shape)

    small = measure_gradient_error(_exact_pair(1.3), A, direction=v * 1e-6)
    large = measure_gradient_error(_exact_pair(1.3), A, direction=v * 1e6)

    assert small.relative_error == pytest.approx(large.relative_error, rel=1e-6)


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

    # And the blindness is real, not hypothetical: force a real direction and
    # the same corrupted gradient measures as perfect.
    real_v = np.real(np.asarray(A.todense())) * 0 + np.random.RandomState(
        1
    ).standard_normal(A.todense().shape)
    blind = measure_gradient_error(
        _complex_pair(imag_scale=4.0), A, direction=real_v.astype(complex)
    )
    assert blind.relative_error < 1e-6, (
        "a real direction was expected to be blind to the imaginary error; if "
        "this now fails the blindness argument needs rechecking"
    )


def test_a_real_direction_on_a_complex_state_is_refused():
    with pytest.raises(ValueError, match="real but the state is complex"):
        measure_gradient_error(
            _complex_pair(),
            _complex_state(),
            direction=np.random.RandomState(2).standard_normal((2, 2, 2, 2, 2)),
        )


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
