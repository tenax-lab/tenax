"""Measuring root-implicit gradient accuracy, because no cheap signal predicts it (#785).

The root-implicit engines report a root residual, a covariant residual, an
adjoint residual and a ``usable_rank``.  None of them is a gradient-quality
signal, and the residual is *anti-correlated* with gradient error across states:
measured on a rank-matched pair at ``chi=4``, the physical simple-update state
carries the **larger** covariant residual (8.49e-06 against 6.85e-06) and a
gradient **13x more accurate** (2.9e-07 against 3.8e-06).  A gate tightened onto
the residual would reject the good state and admit the bad one.

Six candidate surrogates were measured against a directional finite difference
across nine states spanning ten orders of gradient error, and every one failed:

===========================  ==========================================
candidate                    why it fails
===========================  ==========================================
``root_residual``            anti-correlated across states (above)
``covariant_residual``       same
``usable_rank``              ties states whose gradients differ 13x;
                             #785 reports a pair differing 1400x
``s[usable_rank-1]/s[0]``    degenerate -- identically 1 when every
                             direction collapses, so the worst states
                             score the best possible value
``retained_smin_rtol``       inverts on a rank-matched pair, and moves
                             five orders with ``chi`` (3.0e-08 -> 2.1e-13)
                             on a state whose gradient error does not
                             move at all, so no fixed threshold exists
adjoint amplification        structurally ~1: ``F = X'/<X,X'> - X`` makes
``||F_bar||/||y_bar||``      ``dF/dy`` a small perturbation of ``-I``,
                             measured 1.00-1.07, so it collapses to the
                             bare residual
site conditioning            2200x spread in gradient error across a
``s_min/s_max`` of ``A``     1.3x range of conditioning (3.4e-06 to
                             7.7e-03 at ``s_min/s_max ~ 3e-02``), purely
                             from the random seed
===========================  ==========================================

So the honest position is that the accuracy of a root-implicit gradient can be
*measured* but not *predicted*, and this module measures it.  It is deliberately
not called from the engines: it costs four extra CTM convergences, which is far
too much per optimizer step, and #785 rejected exactly that.  Run it once on a
representative state before committing to a long optimization, the way one
checks a discretization before a production run.

Scope of the measurements above: ``D=2``, ``chi`` in {4, 8}, dense asymmetric
engine.  The symmetric and multisite engines are untested -- #785's "Not
established" still stands for them.
"""

from __future__ import annotations

__all__ = ["GradientErrorReport", "measure_gradient_error"]

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from tenax.core.tensor import DenseTensor, Tensor


class GradientErrorReport(NamedTuple):
    """What a directional finite difference says about a gradient.

    Two numbers, and the second is what keeps the first honest.

    ``relative_error`` is ``|g.v - dE/dt| / |dE/dt|`` at the best step tried.
    ``resolution`` is how much the steps disagree among *themselves*, which is
    the floor below which this scan cannot measure anything.

    ``is_resolved`` says which regime you are in, and both are useful:

    * ``True``  -- ``relative_error`` is a measurement.  A gradient that is
      wrong by 15% reports 0.15 here.
    * ``False`` -- the disagreement is at or below the floor, so the honest
      reading is "the gradient is accurate to about ``resolution``".  A tiny
      ``resolution`` is then good news; a large one means this scan could not
      tell you anything and the steps need adjusting.

    Collapsing those two into one "converged" flag is wrong: an *exact*
    gradient has every step at the roundoff floor, and calling that
    unconverged would report the best possible outcome as a failure.
    """

    relative_error: float
    analytic: float
    finite_difference: float
    step: float
    resolution: float
    is_resolved: bool

    def summary(self) -> str:
        if self.is_resolved:
            verdict = f"gradient error {self.relative_error:.2e}"
        else:
            verdict = (
                f"gradient error below the measurement resolution "
                f"{self.resolution:.2e} (reported {self.relative_error:.2e})"
            )
        return (
            f"{verdict} at h={self.step:.1e} "
            f"(analytic {self.analytic:.10e}, fd {self.finite_difference:.10e})"
        )


def _unit_direction(shape, seed: int, *, complex_valued: bool) -> np.ndarray:
    """A random unit direction, complex when the state is.

    A real direction on a complex state samples only the real coordinates, and
    ``Re(sum(g * v))`` with real ``v`` cannot see the gradient's imaginary
    components at all -- an arbitrarily large error confined to them pairs to
    exactly zero and this check would report it as below its resolution.  The
    asymmetric engine takes complex states (#721), so that is reachable, not
    hypothetical.
    """
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(shape)
    if complex_valued:
        v = v + 1j * rng.standard_normal(shape)
    return v / np.linalg.norm(v)


def measure_gradient_error(
    energy_and_grad: Callable[[Tensor], tuple[Any, Any]],
    A: Tensor,
    *,
    direction: np.ndarray | None = None,
    steps: tuple[float, ...] = (1e-4, 1e-5, 1e-6),
    seed: int = 0,
    fd_spread_tol: float = 10.0,
) -> GradientErrorReport:
    """Measure a gradient's relative error along one direction, by finite difference.

    ``energy_and_grad`` takes a site tensor and returns ``(energy, gradient)`` --
    :func:`~tenax.algorithms._ctm_root_implicit_asym.asym_root_implicit_energy_and_grad`
    with its keywords bound, or any of the other engines' entry points.  The
    finite difference is taken of *that same callable*, so this compares the
    gradient against the map it claims to differentiate and needs no external
    reference implementation.

    **Cost: two energy evaluations per step**, each a full CTM convergence, so
    six with the default three steps.  That is why nothing calls this
    automatically.

    ``steps`` is scanned rather than fixed because a single step cannot tell a
    wrong gradient from an unresolvable difference.  The scan yields both
    numbers in :class:`GradientErrorReport`: the smallest disagreement with the
    finite difference, and the ``resolution`` the steps agree to among
    themselves.  ``is_resolved`` says whether the first stands clear of the
    second by ``fd_spread_tol``.

    On a **complex** state the default direction is complex too, and a
    real-*valued* one passed explicitly is refused -- judged on the values, so
    a real array cast to a complex dtype is refused as well.  ``Re(sum(g * v))`` with real ``v`` pairs the
    gradient's imaginary components to exactly zero, so an arbitrarily large
    error confined to them would be reported as below the resolution -- the
    asymmetric engine takes complex states (#721), so that is reachable.  The
    pairing is deliberately *unconjugated*, which is JAX's convention for
    complex cotangents; the conjugated form gives an unrelated number (measured
    0.888 against a finite difference of -1.893) and would invent an enormous
    error on every complex state.

    Only :class:`~tenax.core.tensor.DenseTensor` is supported.  Perturbing a
    ``SymmetricTensor``'s buffer by a dense direction leaves the symmetry
    sectors, so the shifted state would not be a valid tensor at all -- and the
    root-implicit path this exists for is dense anyway.

    Args:
        energy_and_grad: ``A -> (energy, gradient)``.
        A:               The state to measure at.
        direction:       Perturbation direction; a seeded random unit vector by
                         default, complex when the state is.  Normalised on
                         entry either way, since the relative error is
                         scale-free only if it is.
        steps:           Finite-difference steps to scan.  At least two
                         distinct magnitudes: the central difference is even in
                         ``h``, so repeated or sign-flipped entries measure the
                         same difference twice and report zero resolution.
        seed:            Seed for the default random direction.
        fd_spread_tol:   How far the error must stand clear of the resolution
                         floor to count as measured rather than bounded.

    Returns:
        A :class:`GradientErrorReport`.
    """
    if not isinstance(A, DenseTensor):
        raise TypeError(
            f"measure_gradient_error needs a DenseTensor, got {type(A).__name__}. "
            "A dense perturbation of a block-sparse tensor leaves the "
            "symmetry-allowed sectors, so the shifted state is not a valid "
            "tensor and the finite difference would not be of this map."
        )
    # Two *distinct magnitudes*, not two entries.  A central difference is even
    # in h, so (1e-5, 1e-5) and (1e-5, -1e-5) both produce the identical
    # difference twice: ``resolution`` comes out exactly 0, every nonzero
    # disagreement then satisfies ``best_rel > tol * 0``, and a one-step
    # truncation artifact is reported as a resolved measurement -- defeating
    # the guard this check exists to be.
    magnitudes = {abs(float(h)) for h in steps}
    if 0.0 in magnitudes:
        raise ValueError(f"steps must all be nonzero; got {steps!r}.")
    # Distinct is not enough -- they have to be SEPARATED.  The central
    # difference's truncation is ``C*h^2``, so two steps a factor ``r`` apart
    # have biases differing by ``r^2 - 1``: at ``r = 1.01`` the spread is 2% of
    # the bias they SHARE, so ``resolution`` understates the real truncation by
    # ~50x and the report calls a correct gradient resolvedly wrong.  Measured
    # on E = x^4 with steps (0.1, 0.099): spread 1.97e-04 against a common
    # error of 9.71e-03.  ``r >= 2`` makes ``r^2 - 1 = 3``, so the spread is the
    # same order as the bias and can stand in for it.
    if len(magnitudes) >= 2:
        ordered = sorted(magnitudes)
        ratio = ordered[-1] / ordered[0]
        if ratio < 2.0:
            raise ValueError(
                f"steps span only a factor {ratio:.3g} in magnitude; they must "
                f"be separated by at least 2x. Nearby steps share almost all of "
                f"their O(h^2) truncation, so their spread measures how close "
                f"they are to each other rather than how wrong either is, and "
                f"``resolution`` would understate the real error. Got {steps!r}."
            )
    if len(magnitudes) < 2:
        raise ValueError(
            f"steps must contain at least two distinct magnitudes so the scan "
            f"can measure its own resolution; got {steps!r}, which has "
            f"{len(magnitudes)}. The central difference is even in h, so "
            "repeated or sign-flipped steps evaluate the same difference twice "
            "and report a resolution of exactly zero -- which would mark any "
            "one-step truncation artifact as a resolved measurement."
        )

    base = np.asarray(A.todense())
    is_complex = np.iscomplexobj(base)
    v = (
        _unit_direction(base.shape, seed, complex_valued=is_complex)
        if direction is None
        else np.asarray(direction)
    )
    if v.shape != base.shape:
        raise ValueError(f"direction has shape {v.shape}, expected {base.shape}")
    # Rescale by max-abs BEFORE taking the norm.  ``norm`` squares before it
    # sums, so it halves the usable exponent range: a finite direction with
    # components above ~1e154 (float64) overflows it to ``inf``, the zero-vector
    # check passes, ``v / inf`` collapses every component to exactly 0, and then
    # ``analytic``, the finite difference and ``resolution`` are all 0 -- so an
    # arbitrarily wrong gradient reports as more accurate than the scan can
    # resolve.  Below ~1e-162 the same square underflows the norm to 0 instead.
    # This is the #870 trap (four sites in the BP gauge) and the fix is the
    # same: divide by ``max_abs`` first, after which every component is <= 1 and
    # the norm lands in ``[1, sqrt(N)]``.
    max_abs = float(np.max(np.abs(v)))
    if max_abs == 0.0:
        raise ValueError("direction is the zero vector")
    if not np.isfinite(max_abs):
        raise ValueError(
            "direction has non-finite entries, so it cannot be normalised."
        )
    v = v / max_abs
    v = v / float(np.linalg.norm(v))
    if is_complex:
        # Judged on the VALUES, not the dtype, and on BOTH halves.  The two
        # failures are symmetric: ``Re(sum(g * v))`` with a real-valued ``v``
        # pairs the gradient's imaginary components to exactly zero, and with a
        # purely imaginary ``v`` it pairs the real ones to zero instead.  Either
        # way half the gradient is unsampled and an arbitrarily large error
        # living there reports as perfect.  A real array cast to a complex
        # dtype is still real-valued, which is why this is not an
        # ``iscomplexobj`` check.
        v_scale = float(np.max(np.abs(v)))
        halves = (
            ("real", float(np.max(np.abs(v.real)))),
            ("imaginary", float(np.max(np.abs(v.imag)))),
        )
        for missing, content in halves:
            if content < 1e-12 * v_scale:
                # The half that is ABSENT from ``v`` is the half of the
                # gradient that goes unsampled; what remains is the other one.
                sampled = "imaginary" if missing == "real" else "real"
                raise ValueError(
                    f"direction has no {missing} component but the state is "
                    f"complex, so the finite difference would sample only the "
                    f"{sampled} coordinates and any error in the gradient's "
                    f"{missing} components would pair to exactly zero -- a "
                    "wrong gradient would be reported as perfect. Pass a "
                    "direction with both, or omit it for a random one. (A "
                    "real-valued array cast to a complex dtype is still "
                    "real-valued and is refused for the same reason.)"
                )
    elif np.iscomplexobj(v) and float(np.max(np.abs(v.imag))) > 0.0:
        raise ValueError(
            "direction has an imaginary component but the state is real, so "
            "the shifted state would leave the space the gradient was taken "
            "in and the finite difference would not be of this map."
        )

    # Only NOW cast to the state's own precision -- after the real/imaginary
    # validation above, which the cast would otherwise silently satisfy by
    # discarding the very imaginary part it is checking for.  This is the ONLY
    # cast: ``analytic`` and every shifted state then use the identical
    # direction, rather than one projecting in float64 and the other shifting
    # in its float32 rounding.  A float32 state
    # plus a float64 direction promotes the shifted buffer, and since tenax
    # enables x64 the DenseTensor stays promoted, so the gradient would be
    # taken at one precision while every finite-difference energy is evaluated
    # at another, where CTM convergence and tolerances behave differently.
    # That is not the same numerical map, which is the one thing this function
    # promises to compare against.
    v = v.astype(base.dtype, copy=False)

    _energy, grad = energy_and_grad(A)
    grad_arr = np.asarray(grad)
    if not np.all(np.isfinite(grad_arr)):
        # Fail closed.  ``nan > x`` is False, so letting a NaN through would set
        # ``is_resolved=False`` -- the branch documented as "the gradient is
        # better than the scan can resolve" -- and ``summary()`` would report
        # an error below a NaN floor.  A NaN gradient is the #772 failure the
        # optimizer explicitly handles, so it is reachable and must not be
        # dressed up as the good case.
        n_bad = int(np.count_nonzero(~np.isfinite(grad_arr)))
        raise ValueError(
            f"the gradient has {n_bad} non-finite entries, so its accuracy "
            "cannot be measured. This is the #772 shape -- a root-implicit "
            "gradient going NaN on a physical state -- and it is reported "
            "rather than classified, because a NaN compares False against "
            "every threshold and would otherwise be indistinguishable from a "
            "gradient too accurate to resolve."
        )
    analytic = float(np.real(np.sum(grad_arr * v)))
    if not np.isfinite(analytic):
        raise ValueError(
            f"the directional derivative g.v is {analytic}, so there is "
            "nothing to compare a finite difference against."
        )

    def energy_at(t: float) -> float:
        # ``v`` already carries ``base``'s precision, so this inherits it.
        # Casting here INSTEAD would leave ``analytic`` projecting along a
        # float64 direction while the difference shifts along its float32
        # rounding -- two slightly different directions, compared as one.
        intended = t * v
        shifted_data = base + intended
        # Coordinatewise, not whole-array.  A single frozen coordinate is enough
        # to break the measurement and does NOT make the buffer bit-identical:
        # on a mixed-scale tensor a 1e20 entry can swallow the step while a
        # 1.0 entry moves, so the difference follows a *distorted* direction
        # while ``analytic`` still projects along the intended one -- reporting
        # a correct gradient as wrong.  Compare the realised displacement to the
        # intended one on the coordinates where the intended shift is
        # meaningful; ordinary rounding leaves this around 1e-10.
        realised = shifted_data - base
        # Rounding a sum costs about ``eps * |base|`` in absolute terms, so that
        # -- not a fixed relative figure -- is the bound a healthy coordinate
        # must meet.  A dtype-blind threshold rejects float32 outright, where
        # ordinary rounding is already 2.5e-04 relative.
        eps = float(np.finfo(base.dtype).eps)
        tol_abs = 4.0 * eps * np.abs(base)
        # A coordinate is "frozen" when it carries real weight in the direction
        # yet is too small for ``base`` to represent: it then contributes to
        # ``g.v`` but not to the difference, so the two follow different
        # directions.  Whole-array equality cannot see this -- on a mixed-scale
        # tensor a 1e20 entry freezes while the 1.0 entries move.
        meaningful = np.abs(intended) > 1e-12 * float(np.max(np.abs(intended)))
        frozen = meaningful & (np.abs(intended) <= tol_abs)
        drifted = meaningful & (np.abs(realised - intended) > tol_abs)
        n_bad = int(np.count_nonzero(frozen | drifted))
        if n_bad:
            raise ValueError(
                f"the step h={t:.1e} rounds away against this state on "
                f"{n_bad} of {int(np.count_nonzero(meaningful))} weighted "
                f"coordinates: the shift there is below what base can "
                f"represent, so the finite difference follows a different "
                f"direction from the one g.v is projected along and a correct "
                f"gradient would be reported as wrong. The state's largest "
                f"entry is {float(np.max(np.abs(base))):.2e}; use steps large "
                "enough to change it."
            )
        shifted = DenseTensor(jnp.asarray(shifted_data), A.indices)
        energy, _g = energy_and_grad(shifted)
        out = float(np.real(energy))
        if not np.isfinite(out):
            raise ValueError(
                f"the energy at the perturbed state (t={t:.1e}) is {out}, so "
                "the finite difference is undefined. Reported rather than "
                "propagated: a non-finite difference makes every comparison "
                "below fail open."
            )
        return out

    energy_eps = float(np.finfo(base.dtype).eps)
    results = []
    for h in steps:
        e_plus, e_minus = energy_at(h), energy_at(-h)
        # Cancellation in the SUBTRACTION, which the displacement guard above
        # cannot see: the tensors genuinely differ, but the energies do not.
        # Near a stationary point, or with a large constant offset, the change
        # falls under the spacing of the energy value itself -- for
        # E = 1e20 + sum(x^2), ULP(1e20) is 1.6e+04 and a 1e-05 change vanishes,
        # so every difference is exactly 0, ``resolution`` is 0, and a correct
        # gradient is reported as ~1e300 wrong with ``is_resolved=True``.
        span = abs(e_plus - e_minus)
        floor = 4.0 * energy_eps * max(abs(e_plus), abs(e_minus))
        if span <= floor:
            raise ValueError(
                f"at h={h:.1e} the two perturbed energies differ by {span:.3e}, "
                f"at or below the {floor:.3e} rounding floor of the energy "
                f"itself (|E| ~ {max(abs(e_plus), abs(e_minus)):.3e}), so the "
                "difference carries no signal. This is cancellation in the "
                "subtraction rather than in the state -- the shifted tensors "
                "did differ. Near a stationary point, or with a large additive "
                "constant in the energy, use a direction with more signal or "
                "remove the offset."
            )
        fd = (e_plus - e_minus) / (2.0 * h)
        if not np.isfinite(fd):
            raise ValueError(
                f"the finite difference at h={h:.1e} is {fd}; the energies "
                "were finite, so this is a cancellation or overflow in the "
                "difference itself."
            )
        rel = abs(fd - analytic) / max(abs(fd), 1e-300)
        results.append((rel, fd, h))

    # Pick the step by |h| ALONE, never by how well its difference agrees with
    # the gradient.  Selecting on agreement is circular: it hands the answer to
    # whichever truncation error happens to imitate the gradient's error, and
    # the two coincide exactly more often than one would guess.  For E = x^4 at
    # x = 1 the central difference is 4 + 4h^2 exactly, so h=0.1 returns 4.04 --
    # and a supplied gradient of 4.04, which is 1% wrong, was matched to 1.8e-15
    # and reported as perfect.  The smallest step carries the least truncation;
    # whether it was also small enough to be roundoff-dominated is exactly what
    # ``resolution`` below reports, so the honest answer survives either way.
    best_rel, best_fd, best_h = min(results, key=lambda r: abs(r[2]))

    # The scan's own noise floor: how far the finite differences sit from each
    # other, relative to the scale of the derivative.  Measured on the
    # differences themselves rather than on their errors -- when the gradient
    # is right every step lands at roundoff, and the ratio of two floor values
    # (1e-11 against 2e-10) is noise that would read as a 20x spread and report
    # an exact gradient as a failure.
    fds = [r[1] for r in results]
    scale = max(abs(analytic), max(abs(f) for f in fds), 1e-300)
    resolution = (max(fds) - min(fds)) / scale

    # The error is a *measurement* only when it stands clear of that floor by
    # ``fd_spread_tol``.  Below the floor it is an upper bound instead, which
    # is the good case and must not be reported as a failed measurement.
    resolved = best_rel > fd_spread_tol * resolution

    return GradientErrorReport(
        relative_error=best_rel,
        analytic=analytic,
        finite_difference=best_fd,
        step=best_h,
        resolution=resolution,
        is_resolved=resolved,
    )
