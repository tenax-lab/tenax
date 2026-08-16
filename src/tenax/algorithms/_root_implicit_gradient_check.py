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
not called from the engines: with the default three steps it costs **seven** CTM
convergences -- two per step plus one at the unperturbed state -- which is far
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
    * ``False`` -- and it has **two** causes, which must not be read alike:

      - ``fd_divergence`` is a small number: the disagreement did not stand
        clear of the floor, so the gradient is accurate to about
        ``unresolved_bound`` -- the larger of the two thresholds the error is
        tested against, so ``is_resolved`` is exactly
        ``relative_error > unresolved_bound`` and the published floor can never
        sit below an error that was rejected.  A tiny bound is good news; a
        large one means the steps need adjusting.
      - ``fd_divergence`` is large enough to dominate the floor: **the scan
        did not converge**.  The bound then carries that divergence, so it is
        honest but wide -- a 50% error left unresolved this way has a bound of
        at least 50%, not a misleadingly tiny one.
      - ``fd_divergence`` is ``nan``: **indeterminate**, no two steps probed
        commensurable directions.  ``unresolved_bound`` is ``nan`` too, because
        nothing constrained it and a finite number here would be invented --
        relative errors that share the gradient under test agree with each
        other for free, so the invented number could even be small.
        :meth:`summary` says which case applies.

      ``nan`` means no two steps probed commensurable directions.  That is an
      absence of evidence, and it leaves the result **unresolved** and the
      bound meaningless -- not the good case.  Agreeing relative errors cannot
      establish an error on their own, because they share the gradient being
      tested.

    The convergence check is evidence, not proof: two steps can agree by
    coincidence rather than by converging, and no pair can tell those apart.
    More commensurable steps make it less likely.

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
    unresolved_bound: float
    direction_distortion: float
    fd_divergence: float

    def summary(self) -> str:
        if self.is_resolved:
            verdict = f"gradient error {self.relative_error:.2e}"
        else:
            # Quote the bound that was actually compared against, not
            # ``resolution`` alone: an error of 1e-02 against a resolution of
            # 2e-03 lands here, and saying "below 2e-03" would claim the
            # gradient is five times better than the scan established.
            # ``unresolved_bound`` is the threshold ``is_resolved`` itself
            # uses -- both clauses of it -- so the two cannot disagree.
            if self.fd_divergence != self.fd_divergence:  # nan
                # No floor to quote -- the bound is NaN because nothing
                # constrained it.  Printing "a floor of nan" would invite the
                # reader to treat it as a number that came out badly rather
                # than as a measurement that was never established.
                verdict = (
                    f"gradient error {self.relative_error:.2e} not resolved: no "
                    "two steps probed commensurable directions, so the scan is "
                    "indeterminate and no floor was established"
                )
            else:
                verdict = (
                    f"gradient error {self.relative_error:.2e} not resolved "
                    f"against a floor of {self.unresolved_bound:.2e}"
                )
                if self.fd_divergence > 0.0 and self.fd_divergence >= self.resolution:
                    # No threshold: say so exactly when the differences, rather
                    # than the errors' own scatter, are what set the floor.  A
                    # ``0.25`` gate here would still make 0.24 and 0.26 read
                    # qualitatively differently after the classifier stopped
                    # using one.
                    verdict += (
                        f"; that floor is set by the differences still moving "
                        f"{self.fd_divergence:.2e} between steps rather than by "
                        "the errors' own scatter"
                    )
        return (
            f"{verdict} at h={self.step:.1e} "
            f"(analytic {self.analytic:.10e}, fd {self.finite_difference:.10e})"
        )


#: How many times the reported error must exceed the finite differences' own
#: divergence before it is called measured.  Not caller-tunable on purpose: it
#: bounds the fraction of the error that non-convergence could explain, and a
#: caller loosening it would be loosening exactly the guarantee the number is
#: supposed to carry.
_ARTEFACT_MARGIN = 3.0

#: How many ULP of the state the factor-2 separation may fall short by.
#:
#: The separation is measured on the *realised* displacements, and each endpoint
#: rounds to the state's own grid, so the coarse span comes back a little under
#: twice the fine one: a requested ratio of exactly 2 measures 1.9999999999999993
#: on a float64 state with a 1e12 coordinate.  The deficit is ABSOLUTE -- about
#: one ULP of the largest coordinate -- which is why the slack is expressed in
#: those units rather than as a relative constant.
#:
#: A relative constant is what this had first, and it was wrong: 1e-06 is
#: float64 reasoning, while the same scan on complex64 falls short by 6.7e-05
#: (34x more) and was refused outright as unable to bound its own resolution.
#: The shortfall grows as the step shrinks relative to the state, so no fixed
#: multiple of ``eps`` works either -- only a bound in the units the deficit
#: actually has.
#:
#: Four ULP leaves margin over the one observed while staying far from the
#: degeneracy this guards: two steps of nearly equal size fall short by about a
#: whole span, which is many orders larger than any ULP count.
#:
#: The ULP is taken at the coordinate that SETS the span, not at the largest
#: coordinate of the state.  With a 1e18 entry beside unit ones the span is
#: 3.5e-01 at a unit coordinate (ULP 2.2e-16) while the state's max scale gives
#: 8.9e+02 -- eighteen orders too big, which refuses the scan outright.
_SPAN_SLACK_ULP = 4.0

#: The smallest separation an accepted pair can have once rounding is allowed
#: for.  The allowance corrects the factor-2 requirement; it must never erase
#: it, and subtracting it unconditionally does exactly that when the spans are
#: themselves a few ULP -- 2.00 and 1.50 ULP is a ratio of 1.333, which a
#: 4-ULP allowance accepts.  Capping the allowance at ``2 - this`` guarantees
#: every accepted pair really is separated by at least this factor.
_MIN_EFFECTIVE_RATIO = 1.5


def _separated(coarse: float, fine: float, coarse_ulp: float, fine_ulp: float) -> bool:
    """Is ``coarse`` at least twice ``fine``, allowing for endpoint rounding?

    The allowance is a *correction*, so it may never erode the requirement into
    meaninglessness.  Subtracting it unconditionally does exactly that once the
    displacements are themselves only a few ULP: on a float32 unit state, spans
    of 2.00 and 1.50 ULP have a ratio of 1.333 and a 4-ULP allowance accepts
    them, after which a smooth odd polynomial makes those two neighbouring
    secants agree and reports its exact zero gradient as a resolved 100% error.

    So the allowance is capped at what still leaves a real factor:
    ``_MIN_EFFECTIVE_RATIO`` is the smallest separation any accepted pair can
    have, whatever the rounding.  Above the cap the spans are quantisation, not
    measurement, and the pair is refused rather than corrected.
    """
    if coarse < _MIN_EFFECTIVE_RATIO * fine:
        # Hard floor, taken BEFORE any allowance: whatever the rounding, an
        # accepted pair is separated by at least this factor.  Comparing the
        # allowance against ``fine`` instead does not work -- the allowance
        # belongs to whichever step set the span, so a coarse step on a 1e15
        # coordinate carries a 0.5 allowance while the fine step's span is
        # 4.4e-02, and a scan separated by a factor of 33.9 was refused.
        return False
    # Propagated, not the larger of the two: the margin ``coarse - 2 * fine``
    # inherits one rounding from ``coarse`` and two from ``fine``.
    slack = _SPAN_SLACK_ULP * (coarse_ulp + 2.0 * fine_ulp)
    return coarse >= 2.0 * fine - slack


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

    **Cost: two energy evaluations per step**, each a full CTM convergence, plus
    one at the unperturbed state -- seven with the default three steps.  That is
    why nothing calls this automatically.

    ``steps`` is scanned rather than fixed because a single step cannot tell a
    wrong gradient from an unresolvable difference.  The scan yields both
    numbers in :class:`GradientErrorReport`: the disagreement at the **smallest
    usable** step magnitude, and the ``resolution`` the steps agree to among
    themselves.  ``is_resolved`` says whether the first stands clear of the
    second by ``fd_spread_tol``.

    The step is chosen by ``|h|`` alone and never by which difference agrees
    best with the gradient -- that would be circular, handing the answer to
    whichever truncation error imitates the gradient's error.  For ``E = x^4``
    the central difference is ``4 + 4h^2`` exactly, so ``h=0.1`` returns 4.04
    and a supplied gradient of 4.04, 1% wrong, was matched to 1.8e-15 and
    reported as perfect.  Whether the smallest step was itself
    roundoff-dominated is what ``resolution`` reports, so the honest answer
    survives either way.

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
    if not (np.isfinite(fd_spread_tol) and fd_spread_tol > 1.0):
        # This number decides which branch the report is read as.  Negative
        # makes ``best_rel > tol * resolution`` true for essentially any error,
        # so everything looks resolved; NaN makes it false always, so every
        # result lands in the documented "better than resolvable" branch.
        raise ValueError(
            f"fd_spread_tol must be finite and greater than 1, got "
            f"{fd_spread_tol!r}. It is the factor by which the error must "
            "EXCEED the floor, so anything at or below 1 lets an error merely "
            "equal to the floor count as measured: at 1.0 the exact gradient "
            "of sum(x**3) at zero, whose relative error is 1 against a "
            "resolution of 0.9999, is declared definitively 100% wrong. "
            "Negative marks any error resolved, and NaN marks every one "
            "unresolved -- the branch that reads as good news."
        )
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
    # The STATE, not just the energies.  A non-finite coordinate the callback
    # happens not to use passes every energy check -- an affine objective can
    # read only the finite coordinates and return a finite gradient -- while
    # ``realised = shifted - base`` is NaN there, carrying NaN into
    # ``effective``, ``analytic_h`` and every ``rel``.  The scan then returns a
    # report whose error, resolution, distortion and bound are all NaN and
    # blames incommensurable steps, which is a wrong diagnosis of a broken
    # input.  It also fails OPEN on the way: the separation check compares
    # ``nan >= 2*fine - slack``, which is False, so the scan proceeds.
    if not np.all(np.isfinite(base)):
        bad = int(np.count_nonzero(~np.isfinite(base)))
        raise ValueError(
            f"the state has {bad} non-finite entr{'y' if bad == 1 else 'ies'}, "
            "so every displacement taken from it is undefined. Reported rather "
            "than measured: the energies can still come back finite if the "
            "callback does not read those coordinates, and the scan would then "
            "return an all-NaN report blaming the steps."
        )
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

    energy0, grad = energy_and_grad(A)
    # Finiteness of the WHOLE scalar, before ``np.real``.  Checking after the
    # projection accepts ``complex(finite, nan)``: every energy in the scan
    # could be non-finite while the report reads as ordinary.
    e0_arr = np.asarray(energy0)
    if not np.all(np.isfinite(e0_arr)):
        # Checked for the same reason every perturbed energy is: the map has to
        # be defined at the state whose gradient is being measured.  Without
        # this, a NaN here with finite neighbours yields a confident-looking
        # report about a map that is undefined where it was asked.
        raise ValueError(
            f"the energy at the unperturbed state is {e0_arr}, so the map is "
            "undefined at the state whose gradient is being measured."
        )
    grad_arr = np.asarray(grad)
    if not np.iscomplexobj(base) and np.iscomplexobj(grad_arr):
        imag = float(np.max(np.abs(grad_arr.imag)))
        if imag > 0.0:
            # ``Re(sum(g * v))`` would silently drop it.  A cotangent for a real
            # input space has no imaginary part, so this is a malformed
            # gradient rather than one to project: ``2x + 1j*1e30`` would
            # otherwise report as accurate.
            raise ValueError(
                f"the gradient has a non-zero imaginary part (max {imag:.3e}) "
                "but the state is real, so it is not a valid cotangent for "
                "this input space. Taking its real part would discard that "
                "component silently and report the gradient as accurate."
            )
    if grad_arr.shape != base.shape:
        # ``grad_arr * v`` would happily BROADCAST a scalar or a lower-rank
        # array, and the projection then looks right: for ``E = sum(x)`` a
        # returned scalar ``1`` gives exactly the directional derivative of the
        # true all-ones gradient, so a callback that never produced a
        # tensor-shaped gradient at all is reported as perfectly accurate.
        raise ValueError(
            f"the gradient has shape {grad_arr.shape}, expected "
            f"{base.shape}. A broadcast-compatible shape is not accepted: it "
            "would project to a plausible directional derivative and hide the "
            "fact that no gradient of the right shape was returned."
        )
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

    class _StepUnusable(Exception):
        """This step cannot produce a difference; others still might.

        Distinct from the hard errors, which are about the *map* being broken
        (a NaN gradient, a non-finite energy) and cannot be fixed by choosing a
        different step.  A step that rounds away, or whose energies cancel, is
        a property of that step against this state -- discarding the whole scan
        for it throws away the other steps' perfectly good data.
        """

    energy_dtypes: list = []

    def _energy_eps() -> float:
        """Machine epsilon of the precision the energies actually came back in."""
        if not energy_dtypes:
            return float(np.finfo(base.dtype).eps)
        dt = np.result_type(*energy_dtypes)
        if not np.issubdtype(dt, np.inexact):
            return float(np.finfo(np.float64).eps)
        return float(np.finfo(dt).eps)

    def energy_at(t: float) -> float:
        # ``v`` already carries ``base``'s precision, so this inherits it.
        # Casting here INSTEAD would leave ``analytic`` projecting along a
        # float64 direction while the difference shifts along its float32
        # rounding -- two slightly different directions, compared as one.
        # ``t`` is a plain Python float by the time it gets here (the step is
        # normalised at the top of the loop), so it is a WEAK scalar under
        # NEP 50 and leaves this at the state's dtype.  Casting it to the
        # state's dtype explicitly would be redundant -- and a redundancy no
        # test can distinguish is worse than the one-line invariant above.
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
        # Compare the projected derivative the difference will actually follow
        # against the one ``analytic`` assumes.  This replaces every
        # per-coordinate mask: those kept guarding an adjacent property --
        # first the direction's own magnitude, which misses a coordinate with
        # tiny ``v_i`` and huge ``g_i``; then the contribution's share of the
        # L1 total, which misses a coordinate that dominates the *signed* sum
        # after cancellation (contributions 1e-13, 1, -1+5e-14 sum to 1.5e-13
        # while the L1 total is 2). The projection is the quantity that has to
        # survive, so it is the one to test.
        shifted = DenseTensor(jnp.asarray(shifted_data), A.indices)
        energy, _g = energy_and_grad(shifted)
        energy_arr = np.asarray(energy)
        energy_dtypes.append(energy_arr.dtype)
        # Whole scalar first -- see the unperturbed check above.  ``np.real``
        # applied before this would let ``complex(finite, nan)`` through.
        if not np.all(np.isfinite(energy_arr)):
            raise ValueError(
                f"the energy at the perturbed state (t={t:.1e}) is "
                f"{energy_arr}, so "
                "the finite difference is undefined. Reported rather than "
                "propagated: a non-finite difference makes every comparison "
                "below fail open."
            )
        # The REALISED displacement comes back with the energy.  Everything
        # downstream is compared along the direction the arithmetic actually
        # produced, so this is data rather than a validation input.
        return (
            float(np.real(energy_arr)),
            float(np.imag(energy_arr)) if np.iscomplexobj(energy_arr) else 0.0,
            realised,
        )

    results = []
    skipped: list[str] = []
    for h in steps:
        # Normalised to a plain Python float FIRST.  A ``np.float64`` step --
        # what ``tuple(np.logspace(...))`` yields -- is a strong scalar under
        # NEP 50, so it promotes anything it touches: ``t * v`` and the shifted
        # buffer, and separately ``(realised_plus - realised_minus) / (2 * h)``,
        # which decides the precision of the direction the projection is taken
        # along.  A Python float is weak and leaves each at the state's own
        # dtype, so the scan measures the map it was given rather than a
        # float64 lift of it -- and the answer stops depending on whether the
        # caller wrote ``1e-2`` or ``np.float64(1e-2)``.
        h = float(h)
        try:
            (e_plus, i_plus, realised_plus), (e_minus, i_minus, realised_minus) = (
                energy_at(h),
                energy_at(-h),
            )
            # Cancellation in the SUBTRACTION, which the displacement guard above
            # cannot see: the tensors genuinely differ, but the energies do not.
            # Near a stationary point, or with a large constant offset, the change
            # falls under the spacing of the energy value itself -- for
            # E = 1e20 + sum(x^2), ULP(1e20) is 1.6e+04 and a 1e-05 change vanishes,
            # so every difference is exactly 0, ``resolution`` is 0, and a correct
            # gradient is reported as ~1e300 wrong with ``is_resolved=True``.
            span = abs(e_plus - e_minus)
            # The floor belongs to the precision the ENERGY came back in, not the
            # state's.  A map may evaluate at float64 from a float32 state, and
            # charging it the float32 floor rejects a scan whose span is seven
            # orders above its real rounding: |E| ~ 1e6 gives 4.8e-01 against
            # 8.9e-10.
            floor = 4.0 * _energy_eps() * max(abs(e_plus), abs(e_minus))
            if span <= floor:
                raise _StepUnusable(
                    f"at h={h:.1e} the two perturbed energies differ by {span:.3e}, "
                    f"at or below the {floor:.3e} rounding floor of the energy "
                    f"itself (|E| ~ {max(abs(e_plus), abs(e_minus)):.3e}), so the "
                    "difference carries no signal. This is cancellation in the "
                    "subtraction rather than in the state -- the shifted tensors "
                    "did differ. Near a stationary point, or with a large additive "
                    "constant in the energy, use a direction with more signal or "
                    "remove the offset."
                )
            # A genuinely complex objective is refused -- but on the imaginary
            # part's VARIATION, not its size.  Scaling against ``|Re(E)|`` lets
            # an additive real constant hide it: ``1e20 + 1e10*sum(x) +
            # 1j*5e9*sum(x)`` has an imaginary value only 5e-11 of the offset,
            # and the scan would differentiate the real part alone while
            # accepting a gradient that omits an imaginary derivative half as
            # large.  A constant cancels in a difference, so the difference is
            # what to look at.  The floor keeps the roundoff residue that any
            # real objective evaluated in complex arithmetic carries.
            # Compared against the REAL span alone.  An absolute floor derived
            # from ``|E|`` reintroduces exactly the defect this guard was
            # rewritten to fix, one level down: at ``|E| = 1e20`` that floor is
            # ~8.9e+04 and swallows an imaginary variation of 1.2e+04 -- an
            # imaginary derivative 30% the size of the real one, hidden by an
            # offset that changes no derivative at all.
            #
            # No floor is needed.  The cancellation check above has already
            # established that ``span`` is above the energy's own rounding, so
            # a ratio against it is meaningful; and a real objective evaluated
            # in complex arithmetic keeps its imaginary residue proportional to
            # the energy, so the ratio stays at ~1e-15 rather than at 1e-6.
            # Where the residue really is within 1e-6 of the signal, the scan
            # cannot tell an imaginary derivative from noise, and refusing is
            # the honest answer.
            imag_span = abs(i_plus - i_minus)
            if imag_span > 1e-6 * span:
                raise ValueError(
                    f"the energy's imaginary part varies by {imag_span:.3e} "
                    f"across h={h:.1e}, against {span:.3e} for the real part, "
                    "so this is not a real objective. Differencing the real "
                    "part alone would validate the gradient of Re(E) while "
                    "reporting it as the gradient of the callable. Return a "
                    "real energy, or its real part explicitly."
                )
            fd = (e_plus - e_minus) / (2.0 * h)
            if not np.isfinite(fd):
                raise ValueError(
                    f"the finite difference at h={h:.1e} is {fd}; the energies "
                    "were finite, so this is a cancellation or overflow in the "
                    "difference itself."
                )
            # Project the gradient along the direction the difference REALLY
            # followed, not the one that was asked for.  ``base + h*v`` and
            # ``base - h*v`` round differently, so the central difference
            # corresponds to ``(realised_plus - realised_minus) / 2h`` -- and
            # comparing a projection along ``v`` against a difference along
            # that is comparing two different derivatives.  Three rounds of
            # floors tried to bound the gap; none could, because a gradient can
            # always align with the residual.  Measuring both along the same
            # direction removes the gap instead of bounding it.
            effective = (realised_plus - realised_minus) / (2.0 * h)
            analytic_h = float(np.real(np.sum(grad_arr * effective)))
            # Scale BEFORE subtracting.  ``abs(fd - analytic_h)`` overflows
            # for finite operands of opposite sign near the dtype limit: with
            # a slope of -1e308 and a returned gradient of +1e308 the
            # difference is 2e308, so ``rel`` came out ``inf``, ``inf - inf``
            # made ``resolution`` and the bound NaN, and a gradient wrong by a
            # factor of -1 was filed as *unresolved* while ``fd_divergence``
            # sat at 4.9e-12 -- a converged scan reported as no measurement.
            # Dividing each term first keeps ``fd/denom`` at +-1 and gives the
            # correct 2.0.
            denom = max(abs(fd), 1e-300)
            rel = abs(fd / denom - analytic_h / denom)
            if not np.isfinite(rel):
                raise _StepUnusable(
                    f"the relative error is {rel} -- the projected gradient "
                    f"{analytic_h:.3e} is beyond what a difference of "
                    f"{fd:.3e} can be compared against at this scale"
                )
            # Kept as information, NOT as an accuracy floor: it says which
            # direction was actually probed.  A coordinate the arithmetic
            # cannot move is simply not tested, and no floor can repair that --
            # saying so is the honest report.
            step_distortion = float(np.linalg.norm(effective - v)) / max(
                float(np.linalg.norm(v)), 1e-300
            )
            # Per unit of displacement, so steps with different effective
            # NORMS stay commensurable.  Raw differences under different
            # scalings are different derivatives, and pooling them reads that
            # difference as noise.
            eff_norm = max(float(np.linalg.norm(effective)), 1e-300)
            # How far the two endpoints REALLY sit apart.  This, not the
            # requested ``h``, is what makes two steps distinct measurements:
            # distinct requested magnitudes can round to the same pair of
            # states, and then the same secant is counted twice as independent
            # evidence.  On a float32 scalar at 0.75, steps of 1.2 and 0.6 ULP
            # both realise the same +-1-ULP endpoints, so both normalised
            # differences are identical, the divergence is 0, and an exact
            # gradient is reported as a *resolved* 100% error.
            #
            # ``max`` of the absolute values rather than a norm: a norm squares
            # before it sums and so halves the exponent range it can represent,
            # which is how a displacement at 1e200 becomes ``inf`` and one at
            # 1e-200 becomes exactly 0.
            displacement = np.abs(realised_plus - realised_minus)
            realised_span = float(np.max(displacement))
            # ``np.spacing`` on the ARRAY element, not on ``float(...)`` of it:
            # the Python cast promotes to float64 and returns 2.2e-16 for a
            # float32 state, which is the allowance for the wrong grid by
            # nine orders and refuses every single-precision scan.
            # The grid the span was rounded onto -- the ULP of the coordinate
            # that SETS the span, not of the largest coordinate in the state.
            # Those differ wildly: with a 1e18 entry alongside unit ones the
            # span is 3.5e-01 at a unit coordinate whose ULP is 2.2e-16, while
            # the state's own max-scale allowance would be 8.9e+02 -- eighteen
            # orders too big, and enough to refuse the scan outright.
            span_ulp = float(
                np.spacing(np.abs(base).flat[int(np.argmax(displacement))])
            )
            # The identity of a secant is the WHOLE displacement, not its
            # largest component.  Two steps can share a maximum while moving a
            # different coordinate differently -- measured on a float32 state,
            # two steps with span 8.156300e-04 apart at coordinate 0 by
            # 1.4529e-07 against 1.4342e-07 -- and collapsing them onto the
            # scalar makes the retained one depend on which came first, which
            # moved the reported difference and the resolution.
            # Canonicalised across sign, because ``h`` and ``-h`` evaluate the
            # SAME central difference -- the whole point of it being even in
            # ``h`` -- while their raw displacement bytes are exact negatives.
            # Keyed on the signed bytes, adding ``-4`` to ``(4, 2, .25, .125)``
            # counted the coarse secant twice, let that group win on size, and
            # moved the report off the fine direction it should have used.
            forward = (realised_plus - realised_minus).tobytes()
            reverse = (realised_minus - realised_plus).tobytes()
            secant = min(forward, reverse)
            if realised_span == 0.0:
                # Unreachable while the cancellation check above runs first: a
                # zero span means the two endpoints are the same state, so the
                # energies are identical and that check has already rejected
                # the step.  Kept because the separation test below DIVIDES by
                # the smallest span, so if that check is ever moved or
                # narrowed, the failure here would be a ZeroDivisionError from
                # a line that has nothing to do with the cause.  No mutant is
                # carried for it -- it cannot be reached through the public
                # API, and a test that pretended otherwise would be fiction.
                raise _StepUnusable(
                    "the two endpoints are the same state, so the difference "
                    "is identically zero"
                )
        except _StepUnusable as exc:
            # One bad step is not a bad scan.  Keep going; the survivors are
            # checked for separation below, which is the property that actually
            # has to hold.
            skipped.append(f"h={h:.1e}: {exc}")
            continue
        results.append(
            (
                rel,
                fd,
                h,
                step_distortion,
                analytic_h,
                effective,
                fd / eff_norm,
                realised_span,
                span_ulp,
                secant,
            )
        )

    # Separation is required on the REALISED displacements, not the requested
    # magnitudes -- two ``h`` values that land on the same endpoints are one
    # measurement however far apart they were asked to be.
    # Keyed on the span for the extremes check, but the retained ULP is the
    # smallest rather than the first seen, so equal spans with different grids
    # cannot make the allowance depend on the order of ``steps``.
    ulp_of: dict[float, float] = {}
    for r in results:
        ulp_of[r[7]] = min(ulp_of.get(r[7], r[8]), r[8])
    surviving = sorted(ulp_of)
    if len(surviving) < 2 or not _separated(
        surviving[-1], surviving[0], ulp_of[surviving[-1]], ulp_of[surviving[0]]
    ):
        detail = "\n  ".join(skipped) if skipped else "(none skipped)"
        raise ValueError(
            f"only {len(surviving)} distinct realised displacement(s) remain"
            + (
                f", spanning a factor {surviving[-1] / surviving[0]:.3g}"
                if len(surviving) >= 2
                else ""
            )
            + ", which cannot bound its own resolution -- at least two "
            "magnitudes a factor 2 apart are needed. Steps dropped:\n  " + detail
        )

    # Pick the step by |h| ALONE, never by how well its difference agrees with
    # the gradient.  Selecting on agreement is circular: it hands the answer to
    # whichever truncation error happens to imitate the gradient's error, and
    # the two coincide exactly more often than one would guess.  For E = x^4 at
    # x = 1 the central difference is 4 + 4h^2 exactly, so h=0.1 returns 4.04 --
    # and a supplied gradient of 4.04, which is 1% wrong, was matched to 1.8e-15
    # and reported as perfect.  The smallest step carries the least truncation;
    # whether it was also small enough to be roundoff-dominated is exactly what
    # ``resolution`` below reports, so the honest answer survives either way.
    direction_distortion = max(r[3] for r in results)

    # The scan's own noise floor: how far the finite differences sit from each
    # other, relative to the scale of the derivative.  Measured on the
    # differences themselves rather than on their errors -- when the gradient
    # is right every step lands at roundoff, and the ratio of two floor values
    # (1e-11 against 2e-10) is noise that would read as a 20x spread and report
    # an exact gradient as a failure.
    # Pool the per-step RELATIVE ERRORS, not the differences themselves.
    #
    # Each ``rel`` is measured against that step's own ``analytic_h``, along the
    # direction that step actually followed, so it is dimensionless and
    # direction-local: a gradient that is uniformly 5% high reads 0.05 along
    # ANY direction.  Their spread therefore answers the question resolution is
    # for -- does the answer depend on the step? -- without ever comparing two
    # different derivatives.
    #
    # Pooling the differences could not do that at any tolerance.  Raw values
    # mixed different scalings, and normalising fixed the scale but not the
    # direction, while any gradient can amplify a direction difference
    # arbitrarily.  The quantity was wrong, not the tolerance -- which is why
    # the per-step relative errors are pooled instead, and why the tolerance
    # on the directions themselves (``direction_tol``) is a separate matter:
    # it decides which steps may be pooled AT ALL, not how the pooled numbers
    # are compared.  On float32 the pairwise unit distance with a random
    # direction degrades from 5.2e-05 to 4.4e-02 as h shrinks, so a float32
    # scan at fine steps is genuinely indeterminate rather than merely
    # imprecise; coarser steps still resolve.
    # The EVIDENCE GROUP is the single source of truth for everything below:
    # the reported step, the error spread and the convergence signal all come
    # from the same commensurable set, so they describe one measurement rather
    # than three different ones.
    #
    # Grouped by realised unit direction, because differences taken along
    # different directions are different derivatives -- a coordinate that moves
    # at the coarse step and freezes at the fine one is not a convergence
    # sequence.  Deduplicated by REALISED DISPLACEMENT and required to span a
    # factor of two, because ``(0.5, -0.5)`` is the SAME central difference
    # counted twice: its divergence is identically zero, which would
    # manufacture convergence evidence out of one measurement.  Keying on the
    # requested ``|h|`` misses the other way that happens -- two magnitudes
    # that round to the same endpoints are also one measurement, and float32
    # reaches that regime with ordinary-looking steps.  The up-front ``steps``
    # validation cannot cover either -- a subset can violate what the whole
    # satisfies.
    # How closely two steps must agree in direction before either can bound the
    # other's convergence.  ``sqrt(eps)`` of the STATE's dtype, not a fixed
    # constant: the realised direction's error is dominated by cancellation in
    # ``(base + h*v) - base``, which is ``O(eps * |base| / (h |v|))``, so at the
    # step a central difference actually wants -- ``h |v| ~ sqrt(eps) |base|``
    # -- the legitimate wobble is itself ``O(sqrt(eps))``.  A gap above that is
    # not rounding; it is coordinates quantising differently at the two steps.
    #
    # The fixed 1e-3 this replaces was five orders too loose on float64, and
    # the slack was not free.  ``fd_divergence`` cannot separate "the step
    # changed" from "the direction changed" -- doing so needs the true
    # gradient, which is the thing under test -- so any pooled direction gap is
    # charged to the floor as though the differences had not converged.  With a
    # 9.8e-04 gap (13609 of 60000 sampled mixed-scale states reach that band
    # while still passing the separation check) an affine objective whose
    # gradient is nearly orthogonal to the probe drives the divergence to 1.00
    # and the bound to 1.0e+01, marking a *uniform, cleanly measured* 50% error
    # unresolved.  Those steps now land in separate groups and the scan reports
    # itself indeterminate, which is what actually happened.
    #
    # This narrows the exposure by five orders; it does not remove it. The
    # inflation factor is ``g.delta / g.u`` for the true ``g``, unbounded for
    # any nonzero tolerance, and no gradient-free quantity can bound it.  The
    # residual is charged to the floor, so the failure direction is
    # conservative: a measurable error can be left unresolved, never the
    # reverse.
    direction_tol = float(np.sqrt(np.finfo(base.dtype).eps))
    units = [r[5] / max(float(np.linalg.norm(r[5])), 1e-300) for r in results]
    best_group: list[int] = []
    best_key: tuple[Any, ...] | None = None
    for anchor in range(len(units)):
        seen_secants: set[bytes] = set()
        accepted: list[int] = []
        for j in range(len(units)):
            if float(np.linalg.norm(units[j] - units[anchor])) > direction_tol:
                continue
            if results[j][9] in seen_secants:
                continue
            # PAIRWISE, not just against the anchor.  A ball of radius ``tol``
            # admits members up to ``2 * tol`` from each other, so "close to
            # the anchor" is not "close to each other" -- the property that
            # actually has to hold, since every pair in the group is used to
            # bound every other.  Measured on a float32 three-step scan: two
            # steps 2.2556e-04 from the anchor, both inside
            # ``sqrt(eps)=3.4527e-04``, sitting 4.5113e-04 apart.  Commensurable
            # is a relation between the steps, not between each step and a
            # representative.
            #
            # Greedy, so the group is a clique rather than the largest clique.
            # Erring small is the safe direction here: a smaller group can only
            # withhold a measurement, never license one.
            if any(
                float(np.linalg.norm(units[j] - units[m])) > direction_tol
                for m in accepted
            ):
                continue
            seen_secants.add(results[j][9])
            accepted.append(j)
        # Extremes by VALUE, with the ULP as the tiebreak, so two members that
        # share a span cannot make the pair depend on which was stored first.
        members = accepted
        if len(members) >= 2:
            coarse = max(members, key=lambda j: (results[j][7], results[j][8]))
            fine = min(members, key=lambda j: (results[j][7], results[j][8]))
        magnitudes = sorted(results[j][7] for j in members)
        if len(members) >= 2 and _separated(
            results[coarse][7],
            results[fine][7],
            results[coarse][8],
            results[fine][8],
        ):
            # Ranked by a TOTAL order, so no comparison can end in a tie that
            # encounter order then settles.  Size and finest step alone are not
            # total: two cliques that share their finest member have identical
            # keys, and a strict ``>`` keeps whichever ``steps`` happened to
            # produce first -- which changed ``fd_divergence``, ``resolution``
            # and even ``is_resolved`` with the same steps written backwards.
            # The last component is arbitrary but deterministic, and it only
            # ever decides between groups that are equal in every measured
            # property above it.
            #
            # Every component is a MEASURED value, never an index: indices
            # move when ``steps`` is reordered, so a tiebreak on them would
            # reintroduce the order dependence it is meant to remove.  The
            # trailing tuples are sorted, hence permutation-invariant, and
            # together they fix every quantity the report is built from -- so
            # a group that ties on all of them produces an identical report
            # and the choice between them cannot be observed.
            #
            #   more members            -- more evidence
            #   finer smallest span     -- matches the reported-step policy
            #   tighter agreement       -- better commensurability
            #   then the measurements themselves
            key = (
                len(members),
                -min(magnitudes),
                -max(
                    float(np.linalg.norm(units[a] - units[b]))
                    for a in members
                    for b in members
                ),
                tuple(magnitudes),
                tuple(sorted(results[j][6] for j in members)),
                tuple(sorted(results[j][0] for j in members)),
                tuple(sorted(abs(results[j][2]) for j in members)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_group = members

    evidence = [results[j] for j in best_group] if best_group else results

    # Pool the per-step RELATIVE ERRORS, not the differences themselves.  Each
    # ``rel`` is measured against that step's own ``analytic_h``, along the
    # direction that step actually followed, so it is dimensionless and
    # direction-local: a gradient uniformly 5% high reads 0.05 along ANY
    # direction.  Pooling the differences could not do that at any tolerance --
    # raw values mixed scalings, normalising fixed the scale but not the
    # direction, and requiring the directions to agree cannot work on float32,
    # where the pairwise unit distance with a random direction degrades from
    # 5.2e-05 to 4.4e-02 as h shrinks.  The quantity was wrong, not the
    # tolerance.
    rels = [r[0] for r in evidence]
    resolution = max(rels) - min(rels)
    # ``fd_divergence`` is folded in below rather than compared against a
    # separate cutoff.  Any fixed cutoff can be met by construction -- with a
    # 0.25 bound, two commensurable steps whose differences are 1.2 and 1.0
    # give divergence 0.167, agreeing relative errors of 0.0909, and a
    # "measured" verdict on a gradient that is 118% wrong.  How far the
    # derivative is still moving with h is itself a limit on what can be
    # attributed to the gradient, so it belongs in the floor.

    # ...but the relative errors SHARE the gradient under test, so they can
    # agree perfectly while the differences have not converged at all.
    # ``E = sum(x^3)`` at ``x = 0`` has an exact gradient of zero and every
    # central difference proportional to ``h^2``: each ``rel`` is exactly 1, the
    # spread is 0, and an exact gradient would be reported as a *resolved* 100%
    # error.  So a gradient-independent signal is kept alongside: do the
    # differences themselves agree, per unit displacement?  ``nan`` when no
    # group qualified -- an absence of evidence, which leaves the result
    # unresolved rather than licensing it.
    if best_group:
        fdu = [r[6] for r in evidence]
        fd_divergence = (max(fdu) - min(fdu)) / max(max(abs(f) for f in fdu), 1e-300)
    else:
        fd_divergence = float("nan")

    resolution = max(resolution, 0.0 if np.isnan(fd_divergence) else fd_divergence)

    # Reported from the same group.  This does not make a two-point check
    # sound: two steps can agree by coincidence rather than convergence --
    # ``E(q) = q^3 - 102.4 q^5`` is constructed so h=0.5 and h=0.25 give
    # identical normalised differences -- and no pair can distinguish that from
    # a converged sequence.  More commensurable steps reduce the chance;
    # nothing removes it, and the docstring says so rather than implying proof.
    (best_rel, best_fd, best_h, _dist, best_analytic, _eff, _fdu, _span, _ulp, _sig) = (
        min(evidence, key=lambda r: abs(r[2]))
    )

    direction_distortion = max(r[3] for r in results)

    # A tolerated direction distortion is a floor on what this scan can resolve.
    # Every step carries the same one, so it never shows up in their spread: a
    # 0.5% drift with a near-zero spread would otherwise report a correct
    # gradient as resolvedly 0.5% wrong.
    # No distortion term here: the projections are already taken along the
    # realised directions, so there is no residual gap for a floor to cover.

    # The error is a *measurement* only when it stands clear of that floor by
    # ``fd_spread_tol``.  Below the floor it is an upper bound instead, which
    # is the good case and must not be reported as a failed measurement.
    # ``nan`` -- no commensurable pair, so no convergence evidence -- must NOT
    # count as converged.  Agreeing relative errors prove nothing on their own,
    # since they share the gradient under test: ``E = sum((x-base)^3)`` with
    # ``base[0]=1e14`` and steps ``(1, 1e-2)`` makes the realised directions
    # incommensurable, every ``rel`` exactly 1 with zero spread, and an EXACT
    # gradient would be reported as a definitive 100% error.  Absence of
    # evidence is not evidence of convergence.
    # Two conditions, and the second is deliberately NOT scaled by the
    # caller's multiplier.
    #
    # ``fd_spread_tol`` says how far the error must exceed the errors' own
    # scatter.  Letting it also govern the convergence requirement means any
    # value close enough to 1 admits a scan whose derivative is still moving as
    # much as the error itself: for the exact gradient of ``sum(x**3)`` at zero
    # with steps ``(1, 0.5)``, ``relative_error`` is 1 against a divergence of
    # 0.75, so ``fd_spread_tol=1.1`` declared an exact gradient 100% wrong.
    # Raising the minimum multiplier only moves that boundary -- 4/3 clears
    # this example and the next construction sits under it.
    #
    # ``_ARTEFACT_MARGIN`` fixes the share of the reported error that
    # non-convergence could account for.  At 3, at most a third of it can be
    # artefact, so what remains is a real statement about the gradient however
    # the multiplier is set.
    #
    # The ``isnan`` clause is redundant -- ``best_rel > margin * nan`` is
    # already False -- and kept explicit anyway: leaning on NaN comparison
    # semantics silently is what produced the earlier fail-open bugs here, and
    # a reader should not have to rederive it.  Its mutant is therefore an
    # equivalent one, reported rather than kept.
    #
    # There are now two thresholds, so the *reported* bound has to be whichever
    # one the error failed to clear.  Quoting only the first would put the
    # published floor BELOW an error rejected by the second: with
    # ``fd_spread_tol=1.5`` and ``resolution == fd_divergence == 0.1``, a
    # ``best_rel`` of 0.2 clears ``0.15`` but not ``0.3``, and a bound of 0.15
    # would say "accurate to 1.5e-01" about a scan that measured 2.0e-01.  So
    # ``resolved`` is defined *from* the bound rather than alongside it, and the
    # two cannot drift apart again.  ``max`` is not used to fold in the
    # divergence, because ``max`` with ``nan`` returns whichever argument came
    # first -- the ordering trap this module has already been bitten by.
    if np.isnan(fd_divergence):
        # Indeterminate: no two steps probed commensurable directions, so
        # nothing constrained the floor.  Publishing ``fd_spread_tol *
        # resolution`` here would put a *finite* number on a scan that
        # established none -- and it can be small, since relative errors that
        # share the gradient under test agree with each other for free.  NaN is
        # both the honest value and the one that keeps the invariant intact:
        # ``best_rel > nan`` is False, so no separate clause is needed to hold
        # the verdict down, and ``is_resolved`` stays exactly
        # ``relative_error > unresolved_bound`` with no exception to document.
        unresolved_bound = float("nan")
    else:
        unresolved_bound = max(
            fd_spread_tol * resolution, _ARTEFACT_MARGIN * fd_divergence
        )
    resolved = best_rel > unresolved_bound

    return GradientErrorReport(
        relative_error=best_rel,
        analytic=best_analytic,
        finite_difference=best_fd,
        step=best_h,
        resolution=resolution,
        is_resolved=resolved,
        fd_divergence=fd_divergence,
        unresolved_bound=unresolved_bound,
        direction_distortion=direction_distortion,
    )
