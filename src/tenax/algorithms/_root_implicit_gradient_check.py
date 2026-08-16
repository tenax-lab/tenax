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

    On a **complex** state the default direction is complex too, and a real one
    passed explicitly is refused.  ``Re(sum(g * v))`` with real ``v`` pairs the
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
        steps:           Finite-difference steps to scan.
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
    if len(steps) < 2:
        raise ValueError(
            f"steps must contain at least two values so the scan can say whether "
            f"the finite difference converged; got {steps!r}. A single step "
            "cannot separate a wrong gradient from an unconverged difference."
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
    if is_complex and not np.iscomplexobj(v):
        # A real direction on a complex state is blind to the whole imaginary
        # half of the gradient, so it is refused rather than quietly measuring
        # less than the caller thinks.  Passing an explicitly complex-typed
        # array of real values is allowed -- that is a deliberate choice.
        raise ValueError(
            "direction is real but the state is complex, so the finite "
            "difference would sample only the real coordinates and any error "
            "in the gradient's imaginary components would pair to exactly "
            "zero. Pass a complex direction, or omit it for a random one."
        )
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("direction is the zero vector")
    v = v / norm

    _energy, grad = energy_and_grad(A)
    analytic = float(np.real(np.sum(np.asarray(grad) * v)))

    def energy_at(t: float) -> float:
        shifted = DenseTensor(jnp.asarray(base + t * v), A.indices)
        energy, _g = energy_and_grad(shifted)
        return float(np.real(energy))

    results = []
    for h in steps:
        fd = (energy_at(h) - energy_at(-h)) / (2.0 * h)
        rel = abs(fd - analytic) / max(abs(fd), 1e-300)
        results.append((rel, fd, h))

    best_rel, best_fd, best_h = min(results, key=lambda r: r[0])

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
