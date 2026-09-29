"""CTM hold test: a point that successive sweeps agree on is not yet an attractor.

The convergence criteria in this package compare *successive* sweeps.  They
cannot tell an attractor from a saddle: at a saddle consecutive sweeps agree
to the loop's tolerance, yet a small displacement along the unstable direction
grows every sweep, and the loop -- given more sweeps -- walks away to a
different fixed point.  Measured (#1035, fermionic t-V D=3 chi=12 V=1 mu=2,
``su_grow_layout`` key=2): ``ctm_tensor_2site`` certifies a point S with
successive-sweep corner diff < 1e-10 after ~225 sweeps, and S departs at
x1.041/sweep to a stable fixed point B 1.4e-2 away (E_B - E_S = -4.6e-7).
Implicit AD seeded at S linearises the CTM step around a non-attractor.

The hold test perturbs the claimed point by a small relative random amount and
runs the step again.  An attractor pulls the perturbation back in; a saddle
lets its unstable component grow.  The distance is measured in a
**gauge-invariant** metric (:func:`env_spectral_invariants`): the CTM step's
output is defined only up to a gauge on every chi bond (basis order, signs,
and the bond-sign period-2 cycle of the phase-gauged step), and an
element-wise distance would read that gauge motion as instability.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from typing import Any, NamedTuple

import jax
import numpy as np

from tenax.core.tensor import DenseTensor, SymmetricTensor

__all__ = [
    "HoldResult",
    "env_invariant_distance",
    "env_spectral_invariants",
    "hold_test",
    "perturb_env",
]

Invariants = dict[Hashable, np.ndarray]


def _leg_gram_spectra(t) -> dict[Hashable, np.ndarray]:
    """Per leg, per charge sector: singular values of the leg's matricisation.

    For leg ``i`` the matricisation ``(leg i | all other legs)`` has singular
    values ``sqrt(eig(M M^dagger))``.  Any unitary acting on another leg
    cancels in ``M M^dagger``, and a unitary on leg ``i`` itself (within a
    charge sector, which is all a symmetric gauge can do) leaves the
    eigenvalues fixed -- so the result is blind to every chi-bond gauge,
    including slot order and signs.  Block-sparse on ``SymmetricTensor``: the
    Gram matrix of sector ``q`` on leg ``i`` is the sum over the blocks whose
    leg-``i`` charge is ``q`` -- no dense round-trip.
    """
    out: dict[Hashable, np.ndarray] = {}
    if isinstance(t, SymmetricTensor):
        blocks = {k: np.asarray(v) for k, v in t.blocks.items()}
        ndim = len(t.indices)
        for leg in range(ndim):
            grams: dict[int, np.ndarray] = {}
            for key, b in blocks.items():
                q = key[leg]
                q = q.item() if hasattr(q, "item") else q
                m = np.moveaxis(b, leg, 0).reshape(b.shape[leg], -1)
                g = m @ m.conj().T
                grams[q] = grams[q] + g if q in grams else g
            for q, g in grams.items():
                if not np.all(np.isfinite(g)):
                    out[(leg, q)] = np.full(g.shape[0], np.nan)
                    continue
                ev = np.linalg.eigvalsh(g)
                out[(leg, q)] = np.sqrt(np.clip(ev, 0.0, None))[::-1]
        return out
    a = np.asarray(t.todense() if isinstance(t, DenseTensor) else t)
    for leg in range(a.ndim):
        m = np.moveaxis(a, leg, 0).reshape(a.shape[leg], -1)
        if not np.all(np.isfinite(m)):
            out[(leg, None)] = np.full(m.shape[0], np.nan)
            continue
        out[(leg, None)] = np.linalg.svd(m, compute_uv=False)
    return out


def env_spectral_invariants(envs: dict[Any, Any]) -> Invariants:
    """Gauge-invariant fingerprint of a CTM environment.

    For every site, every environment tensor (corners and edges) and every
    leg: the per-charge-sector singular values of that leg's matricisation,
    normalised by the leg's largest singular value (the overall scale is set
    by ``renormalize`` and is not physical).  Two environments related by any
    chi-bond gauge -- slot permutation, signs, a unitary within a sector --
    have identical invariants.

    Args:
        envs: ``{coord: CTMTensorEnv}`` (or any NamedTuple of tensors).

    Returns:
        ``{(coord, field, leg, charge): descending singular values}``.
    """
    inv: Invariants = {}
    for c in sorted(envs):
        env = envs[c]
        for f in env._fields:
            spectra = _leg_gram_spectra(getattr(env, f))
            by_leg: dict[int, float] = {}
            for (leg, _q), s in spectra.items():
                top = float(s[0]) if s.size else 0.0
                # Not ``max``: max(0.0, nan) is 0.0 and would drop the NaN.
                prev = by_leg.get(leg, 0.0)
                by_leg[leg] = top if (top > prev or not math.isfinite(top)) else prev
            for (leg, q), s in spectra.items():
                top = by_leg[leg]
                inv[(c, f, leg, q)] = s / top if top > 0 or math.isnan(top) else s
    return inv


def env_invariant_distance(a: Invariants, b: Invariants) -> float:
    """Sup-norm distance between two invariant fingerprints.

    A sector present on one side only, or shorter on one side, is compared
    against zeros: a new sector carrying singular values ``~x`` is a distance
    ``~x``, continuous in the environment.  Non-finite values give ``inf``
    (fail closed).
    """
    d = 0.0
    for k in set(a) | set(b):
        x = a.get(k, np.zeros(0))
        y = b.get(k, np.zeros(0))
        n = max(x.size, y.size)
        if n == 0:
            continue
        xp = np.zeros(n)
        yp = np.zeros(n)
        xp[: x.size] = x
        yp[: y.size] = y
        v = float(np.max(np.abs(xp - yp)))
        if not math.isfinite(v):
            return math.inf
        d = max(d, v)
    return d


def perturb_env(envs: dict[Any, Any], rel: float, key: jax.Array) -> dict[Any, Any]:
    """Add ``rel * max|leaf| * N(0, 1)`` to every numeric leaf of every env.

    Deterministic in ``key``.  Block-sparse structure is preserved: on a
    ``SymmetricTensor`` the leaves are the stored blocks.
    """
    out = {}
    for c in sorted(envs):
        leaves, tdef = jax.tree.flatten(envs[c])
        keys = jax.random.split(key, len(leaves) + 1)
        key = keys[0]
        new = []
        for x, k in zip(leaves, keys[1:]):
            scale = float(np.max(np.abs(np.asarray(x)))) if x.size else 0.0
            noise = jax.random.normal(k, x.shape, dtype=x.real.dtype)
            if jax.numpy.iscomplexobj(x):
                k2 = jax.random.fold_in(k, 1)
                noise = noise + 1j * jax.random.normal(k2, x.shape, x.real.dtype)
            new.append(x + rel * scale * noise.astype(x.dtype))
        out[c] = jax.tree.unflatten(tdef, new)
    return out


class HoldResult(NamedTuple):
    """Outcome of :func:`hold_test`.

    Attributes:
        passed:    True iff the point held (see :func:`hold_test`).
        rate:      Fitted per-sweep growth factor of the displacement over the
                   tail window (``< 1`` contracting, ``> 1`` escaping);
                   ``inf`` on a non-finite environment.
        distances: Invariant distance between the perturbed and the
                   unperturbed trajectory after ``k = 0..K`` sweeps each.
        envs:      The perturbed trajectory's last environment.  On failure
                   this is the natural place to continue iterating from: it
                   has already been pushed off the saddle along the unstable
                   direction.
        sweeps:    CTM steps spent -- two per hold sweep (both trajectories).
                   A rejection spends ``2 * max_sweeps``.
    """

    passed: bool
    rate: float
    distances: tuple[float, ...]
    envs: dict[Any, Any]
    sweeps: int


#: Default number of hold sweeps (each runs both trajectories, so the cost is
#: twice this in CTM steps).  See :func:`hold_test` for the measurement.
DEFAULT_HOLD_SWEEPS = 40

#: Default relative size of the perturbation.  Small enough to stay linear
#: (the #1035 saddle's escape is still exponential at displacement 1e-4),
#: large enough to sit orders above float noise.
DEFAULT_HOLD_PERTURBATION = 1e-6

#: Early pass: the displacement fell to this fraction of its running peak.
DEFAULT_HOLD_CONTRACTION = 1e-3

#: A growing fit is re-tested every ``sweeps // 2`` sweeps, up to this many
#: times ``sweeps``, before the point is rejected.
DEFAULT_HOLD_EXTENSION = 3


def hold_test(
    step: Callable[[dict[Any, Any]], dict[Any, Any]],
    envs: dict[Any, Any],
    *,
    sweeps: int = DEFAULT_HOLD_SWEEPS,
    perturbation: float = DEFAULT_HOLD_PERTURBATION,
    key: jax.Array | None = None,
    contraction: float = DEFAULT_HOLD_CONTRACTION,
    invariants: Callable[[dict[Any, Any]], Invariants] = env_spectral_invariants,
    max_sweeps: int | None = None,
) -> HoldResult:
    """Is ``envs`` an attractor of ``step``, or only a point it passes through?

    Runs two trajectories for up to ``max_sweeps`` sweeps each: the claimed point
    ``x_k`` and a copy ``y_k`` perturbed by ``perturbation`` (relative,
    deterministic ``key``), and tracks their gauge-invariant distance
    ``d_k = d(y_k, x_k)`` -- a finite-difference estimate of how the step's
    linearisation acts on a random direction.  At an attractor ``d_k`` decays
    geometrically; at a saddle its unstable component grows geometrically.

    Verdict:

    * **pass early** as soon as ``d_k <= contraction * max(d_0..d_k)``: the
      displacement collapsed by orders of magnitude, which an unstable
      component cannot do;
    * at ``k = sweeps`` fit ``log d_k`` over the last ``sweeps // 2`` sweeps
      and **pass iff the per-sweep rate is < 1**.  The earlier sweeps are
      discarded because the step is non-normal: the #1035 attractor B
      amplifies the perturbation ~25x before it decays;
    * a fit that still grows is not yet a rejection: a transient can outlast
      the first window (measured: B, perturbed from a point 3e-8 off it,
      peaks at k~35 and fits 1.019 at K=40, 0.959 at K=60).  The fit is
      repeated every ``sweeps // 2`` sweeps on the latest window, and the
      point is **rejected only if it still grows at** ``max_sweeps`` (default
      ``3 * sweeps``).  A saddle's growth does not saturate; a transient's
      does.

    Measured on #1035 (fermionic t-V D=3 chi=12 V=1 mu=2, perturbation
    1e-6), fitted over ``[K/2, K]``: at K=20/30/40/60 the saddle S reads
    1.026/1.037/1.053/1.057 and the attractor B 0.993/0.971/0.946/0.971; a
    dense D=2 chi=12 Heisenberg environment passes early at sweep 4-5.

    Three statistics that were measured and rejected:

    * ``d_K < d_0`` -- B is still at 1.35 d_0 after 100 sweeps (transient).
    * distance to the *claimed* point instead of the co-evolved one -- the
      claimed point is converged only to the loop's ``conv_tol``, so at an
      attractor the perturbed trajectory plateaus at that residual and the
      fitted rate reads ~1: a false saddle (seen on the mocked saddle map in
      ``tests/test_ctm_hold.py``).
    * successive steps ``d(y_k, y_{k-1})`` -- no floor, but the saddle's
      drift enters a step only as ``(lambda - 1) ~ 0.04`` of the displacement
      and the stable decay swamps it: S still reads 0.988 at K=40, while B
      reads 1.027 (per-sweep spectral jitter).

    Args:
        step:          One CTM sweep, ``envs -> envs``.
        envs:          The claimed fixed point.
        sweeps:        First verdict at ``K`` sweeps (>= 4) per trajectory.
        perturbation:  Relative perturbation size (> 0).
        key:           PRNG key for the perturbation (default ``PRNGKey(0)``).
        contraction:   Early-pass threshold, see above.
        invariants:    Gauge-invariant fingerprint; default
                       :func:`env_spectral_invariants`.
        max_sweeps:    Longest hold before rejecting a still-growing fit
                       (default ``3 * sweeps``; >= ``sweeps``).  A saddle
                       costs ``2 * max_sweeps`` steps.

    Returns:
        :class:`HoldResult`.
    """
    if sweeps < 4:
        raise ValueError(f"hold_test: sweeps must be >= 4, got {sweeps}")
    if max_sweeps is None:
        max_sweeps = DEFAULT_HOLD_EXTENSION * sweeps
    if max_sweeps < sweeps:
        raise ValueError(
            f"hold_test: max_sweeps ({max_sweeps}) must be >= sweeps ({sweeps})"
        )
    if not perturbation > 0:
        raise ValueError(f"hold_test: perturbation must be > 0, got {perturbation}")
    if key is None:
        key = jax.random.PRNGKey(0)
    x = envs
    y = perturb_env(envs, perturbation, key)
    d = [env_invariant_distance(invariants(y), invariants(x))]
    peak = d[0]
    half = sweeps // 2
    rate = math.nan
    for k in range(1, max_sweeps + 1):
        x = step(x)
        y = step(y)
        dk = env_invariant_distance(invariants(y), invariants(x))
        d.append(dk)
        if not math.isfinite(dk):
            return HoldResult(False, math.inf, tuple(d), y, 2 * k)
        peak = max(peak, dk)
        if dk <= contraction * peak:
            rate = (dk / d[0]) ** (1.0 / k) if d[0] > 0 else 0.0
            return HoldResult(True, rate, tuple(d), y, 2 * k)
        if k >= sweeps and ((k - sweeps) % half == 0 or k == max_sweeps):
            ks = np.arange(k - half, k + 1)
            logs = np.log(np.maximum(np.asarray(d)[ks], np.finfo(float).tiny))
            rate = float(np.exp(np.polyfit(ks, logs, 1)[0]))
            if rate < 1.0:
                return HoldResult(True, rate, tuple(d), y, 2 * k)
    return HoldResult(False, rate, tuple(d), y, 2 * max_sweeps)
