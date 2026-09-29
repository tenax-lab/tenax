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


#: The fixed (non-gauge) leg of a 3-leg edge tensor ``T[chi, D^2, chi]``.
_D2_LEG = 1


def _d2_grams(t) -> dict[Hashable, np.ndarray]:
    """Per charge sector, the full Gram matrix of an edge tensor's D^2 leg.

    ``G[m, m'] = sum_{a, b} T[a, m, b] conj(T[a, m', b])``.  A unitary on
    either chi leg cancels in the sum, so ``G`` is blind to every chi-bond
    gauge (order, signs, the period-2 sign cycle); but unlike its spectrum it
    keeps the D^2 basis, which is NOT a gauge -- that leg contracts against
    the fixed double layer.  ``D^2 x D^2`` per sector; block-sparse on
    ``SymmetricTensor`` (sum over the blocks with that D^2 charge).
    """
    out: dict[Hashable, np.ndarray] = {}
    if isinstance(t, SymmetricTensor):
        for key, v in t.blocks.items():
            q = key[_D2_LEG]
            q = q.item() if hasattr(q, "item") else q
            b = np.asarray(v)
            m = np.moveaxis(b, _D2_LEG, 0).reshape(b.shape[_D2_LEG], -1)
            g = m @ m.conj().T
            out[q] = out[q] + g if q in out else g
        return out
    a = np.asarray(t.todense() if isinstance(t, DenseTensor) else t)
    m = np.moveaxis(a, _D2_LEG, 0).reshape(a.shape[_D2_LEG], -1)
    out[None] = m @ m.conj().T
    return out


def _edge_blocks(t) -> list[tuple[tuple, np.ndarray]]:
    """``[((q_a, q_m, q_b), block)]`` of a 3-leg tensor; one block if dense."""
    if isinstance(t, SymmetricTensor):
        out = []
        for key, v in t.blocks.items():
            q = tuple(k.item() if hasattr(k, "item") else k for k in key)
            out.append((q, np.asarray(v)))
        return out
    a = np.asarray(t.todense() if isinstance(t, DenseTensor) else t)
    return [((None, None, None), a)]


#: Number of random probes in the fourth-order sketch (per sector pair).
_SKETCH_R = 32

#: Fixed seed of the sketch coefficients (the fingerprint must be a function
#: of the environment alone, identical on every call).
_SKETCH_SEED = 1035


def _sketch_coeffs(role: int, q: Hashable, d: int) -> np.ndarray:
    """Deterministic ``(R, d)`` Gaussian coefficients for one D^2 sector.

    ``role`` (0..3) gives A, B, C, E their own independent draws.  Seeded from
    the role, a stable hash of the sector charge and its size -- never from
    Python's salted ``hash`` -- so every call and every process agrees.
    """
    import zlib

    seed = [_SKETCH_SEED, role, zlib.crc32(repr(q).encode()), d]
    return np.random.default_rng(seed).standard_normal((_SKETCH_R, d))


def _d2_sketch(t) -> dict[Hashable, np.ndarray]:
    """Random sketch of the fourth-order slice invariant, bounded in size.

    With the D^2 slices ``T_m = T[:, m, :]``, the full fourth-order invariant
    ``Tr(T_m T_m'^+ T_n T_n'^+)`` is D^8 numbers per edge (256 MiB at D=8;
    Codex P2 on #1058).  Instead, for R fixed random coefficient vectors per
    D^2 sector and role, form ``A_r = sum_m a_rm T_m`` (likewise ``B_r``,
    ``C_r``, ``E_r``) and keep ``s_r = Tr(A_r B_r^+ C_r E_r^+)``.

    * ``T_m -> U T_m V^+`` maps every ``A_r -> U A_r V^+``: each ``s_r`` is
      blind to both chi-leg gauges (order, signs, the period-2 sign cycle).
    * A D^2 rotation ``R`` maps ``a_r -> R^T a_r``: ``s_r`` is a generic
      quartic form in the coefficients, so a generic rotation moves it even
      where the Gram ``G`` is degenerate (``G = I``).
    * Size: ``R`` complex numbers per sector pair ``(q, p)`` with
      ``A, B`` in sector ``q`` and ``C, E`` in sector ``p`` -- independent of
      D.  Rotations of a symmetric tensor stay inside a sector, so the
      ``(q, q)`` pairs see them; the ``(q, p)`` pairs add cross-sector
      correlations.  Cost ``O(R chi^3)`` per pair.

    Block-sparse: ``A_r`` is built per ``(q_a, q_b)`` block; ``A_r B_r^+`` is
    summed over the shared ``q_b`` into ``(q_a, q_a')`` blocks, and the trace
    pairs ``(q_a, q_a')`` with ``(q_a', q_a)`` so the chi charge closes.
    """
    # Per D^2 sector: its (q_a, q_b) blocks as (d_a, d_m, d_b) arrays.
    sectors: dict[Hashable, list] = {}
    for (qa, qm, qb), x in _edge_blocks(t):
        sectors.setdefault(qm, []).append((qa, qb, x))

    def combos(q, role):
        c = _sketch_coeffs(role, q, sectors[q][0][2].shape[1])
        return [(qa, qb, np.einsum("xmb,rm->rxb", x, c)) for qa, qb, x in sectors[q]]

    def pair(q, r1, r2):
        """``A_r B_r^+`` per (q_a, q_a'), summed over the shared q_b."""
        out: dict[tuple, np.ndarray] = {}
        left, right = combos(q, r1), combos(q, r2)
        for qa, qb, a in left:
            for qa2, qb2, b in right:
                if qb2 != qb:
                    continue
                p = np.einsum("rxb,ryb->rxy", a, b.conj())
                out[(qa, qa2)] = out[(qa, qa2)] + p if (qa, qa2) in out else p
        return out

    ab = {q: pair(q, 0, 1) for q in sectors}
    ce = {q: pair(q, 2, 3) for q in sectors}
    out: dict[Hashable, np.ndarray] = {}
    for q, P1 in ab.items():
        for p, P2 in ce.items():
            acc = None
            for (qa, qa2), m1 in P1.items():
                m2 = P2.get((qa2, qa))
                if m2 is not None:
                    term = np.einsum("rxy,ryx->r", m1, m2)
                    acc = term if acc is None else acc + term
            if acc is not None:
                out[(q, p)] = acc
    return out


def env_spectral_invariants(envs: dict[Any, Any]) -> Invariants:
    """Gauge-invariant fingerprint of a CTM environment.

    For every site, every environment tensor (corners and edges) and every
    leg: the per-charge-sector singular values of that leg's matricisation,
    normalised by the leg's largest singular value (the overall scale is set
    by ``renormalize`` and is not physical).  Two environments related by any
    chi-bond gauge -- slot permutation, signs, a unitary within a sector --
    have identical invariants.

    Per-leg spectra are ALSO blind to a unitary on an edge tensor's D^2 leg,
    which is not a gauge (Codex P1 on #1058): an unstable mode that only
    rotates that leg would be invisible.  So every 3-leg (edge) tensor also
    contributes its full D^2 Gram matrix per sector (:func:`_d2_grams`),
    normalised by the same leg's largest singular value squared, as its real
    and imaginary parts.  It fixes the D^2 basis and quotients only the chi
    legs: under ``T -> T x_2 U`` it moves as ``U G U^dagger``, so a rotation
    is seen unless ``U`` commutes with ``G`` -- which ANY unitary within a
    degenerate eigenspace of ``G`` does (e.g. orthonormal equal-norm slices,
    ``G = I``; SU(2)-symmetric states can have a degenerate ``G`` by
    symmetry).  So each edge tensor also contributes a bounded random sketch
    of the fourth-order slice invariant ``Tr(T_m T_m'^+ T_n T_n'^+)``
    (:func:`_d2_sketch`: ``R`` = 32 fixed random probes per sector pair,
    independent of D), normalised by the leg scale to the fourth power,
    which is basis-sensitive where ``G`` is not.

    **Known limits.**  (1) The sketch detects a *generic* D^2 rotation with
    probability 1 over its fixed random coefficients, but not every one: a
    rotation that happens to leave all ``R`` probes of a sector unchanged is
    invisible, and the probes are fixed, so that is a fixed (measure-zero)
    set, not a per-call chance.  (2) Even the full fourth-order tensor would
    not be complete: spectra, ``G`` and degree-4 trace words do not separate
    orbits of the chi-leg gauge group in general, so a rotation detected only
    by trace words of degree >= 6 stays invisible.  (A rotation that IS
    realised by chi-leg unitaries, ``sum_k R_mk T_k = U T_m V^+``, is a
    gauge-equivalent environment, and no invariant should see it.)  RDM-based
    invariants would close both, at about one extra CTM contraction per
    sweep per trajectory; not used.

    Args:
        envs: ``{coord: CTMTensorEnv}`` (or any NamedTuple of tensors).

    Returns:
        ``{(coord, field, leg, charge): descending singular values}`` plus,
        per edge tensor, ``{(coord, field, "d2gram", charge): [Re G, Im G]}``
        and ``{(coord, field, "d2sketch", (q, p)): [Re s, Im s]}``, flattened.
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
            t = getattr(env, f)
            ndim = len(t.indices) if hasattr(t, "indices") else np.ndim(t)
            if ndim == 3:
                top = by_leg.get(_D2_LEG, 0.0)
                scale = top * top if top > 0 or math.isnan(top) else 1.0
                for q, g in _d2_grams(t).items():
                    g = np.asarray(g) / scale
                    inv[(c, f, "d2gram", q)] = np.concatenate(
                        [g.real.ravel(), g.imag.ravel()]
                    )
                for q, k in _d2_sketch(t).items():
                    k = np.asarray(k) / (scale * scale)
                    inv[(c, f, "d2sketch", q)] = np.concatenate(
                        [k.real.ravel(), k.imag.ravel()]
                    )
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


def _diff_norm(y: dict[Any, Any], x: dict[Any, Any]) -> float:
    """Element-wise 2-norm of ``y - x`` (``inf`` if the pytrees differ)."""
    tot = 0.0
    for c in x:
        lx, tx = jax.tree.flatten(x[c])
        ly, ty = jax.tree.flatten(y[c])
        if tx != ty or any(a.shape != b.shape for a, b in zip(lx, ly)):
            return math.inf
        for a, b in zip(lx, ly):
            tot += float(np.sum(np.abs(np.asarray(b) - np.asarray(a)) ** 2))
    return math.sqrt(tot)


def _rescale(x: dict[Any, Any], y: dict[Any, Any], s: float) -> dict[Any, Any]:
    """``x + s (y - x)``, leaf by leaf (same pytree structure required)."""
    return {c: jax.tree.map(lambda a, b: a + s * (b - a), x[c], y[c]) for c in x}


class HoldResult(NamedTuple):
    """Outcome of :func:`hold_test`.

    Attributes:
        passed:    True iff every perturbation direction's tail contracted.
        rate:      The worst direction's fitted per-sweep growth factor over
                   its last window (``< 1`` contracting, ``> 1`` escaping);
                   ``inf`` on a non-finite environment.
        rates:     The same, per direction.
        distances: The worst direction's displacement from the unperturbed
                   trajectory after ``k = 0..K`` sweeps, with renormalisation
                   undone (``d_0 * exp(cumulative log growth)``).
        envs:      The worst direction's last environment.  On failure this
                   is the natural place to continue iterating from: it sits
                   off the saddle along the (renormalised) unstable
                   direction.
        sweeps:    CTM steps spent -- ``1 + directions`` per hold sweep.
    """

    passed: bool
    rate: float
    rates: tuple[float, ...]
    distances: tuple[float, ...]
    envs: dict[Any, Any]
    sweeps: int


#: Default first-verdict window (hold sweeps per trajectory).
DEFAULT_HOLD_SWEEPS = 40

#: Default relative size of the perturbation.  Small enough to stay linear
#: (the #1035 saddle's escape is still exponential at displacement 1e-4),
#: large enough to sit orders above float noise.
DEFAULT_HOLD_PERTURBATION = 1e-6

#: Renormalise a displacement once it has shrunk (or grown) by this factor.
DEFAULT_HOLD_CONTRACTION = 1e-3

#: A growing fit is re-tested every ``sweeps // 2`` sweeps, up to this many
#: times ``sweeps``, before the point is rejected.
DEFAULT_HOLD_EXTENSION = 3

#: A displacement at or below this fraction of ``d_0`` has collapsed onto the
#: reference trajectory (float noise); the direction is re-seeded.
_COLLAPSE = 1e-9

#: The contraction recorded for a collapse step (``log`` of this, relative to
#: ``d_0``): large, since the true factor is unbounded.
_COLLAPSE_LOG = 1e-12

#: Independent perturbation directions (deterministic keys); all must pass.
DEFAULT_HOLD_DIRECTIONS = 2


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
    directions: int = DEFAULT_HOLD_DIRECTIONS,
) -> HoldResult:
    """Is ``envs`` an attractor of ``step``, or only a point it passes through?

    Runs the claimed point ``x_k`` and ``directions`` copies ``y_k``, each
    perturbed by ``perturbation`` (relative; independent deterministic keys
    split from ``key``), side by side, and tracks each gauge-invariant
    distance ``d_k = d(y_k, x_k)`` -- a finite-difference estimate of how the
    step's linearisation acts on that direction.

    **Renormalisation (power iteration).**  Whenever ``d_k`` has shrunk below
    ``contraction * d_0`` (or grown above ``d_0 / contraction``) the
    displacement is rescaled back to its initial size, ``y <- x + s (y -
    x)``, and the log-growth is accumulated across the rescale.  This is
    Benettin's Lyapunov estimate: stable components are deflated each time
    while an unstable one keeps its share, so even a weakly excited unstable
    direction comes to dominate, and ``d`` never sinks into float noise on a
    fast attractor.  A rescale is skipped when the element-wise difference is
    far larger than the invariant one suggests (the two trajectories took
    different chi-bond gauges) or the pytrees differ.  A displacement that
    collapses onto the reference trajectory (``d_k <= 1e-9 d_0``, float
    noise; e.g. a locally constant step) cannot be rescaled: the step is
    recorded as a contraction by 1e-12 and the direction is re-seeded with a
    fresh deterministic perturbation, keeping its accumulated log growth.

    **Verdict.**  No early acceptance: a displacement that contracts early
    is exactly what a weakly excited saddle does while its stable components
    decay (Codex P1 on #1058).  At ``k = sweeps`` fit the accumulated log
    growth over the last ``sweeps // 2`` sweeps; **pass iff every direction's
    rate is < 1**.  A fit that still grows is re-tested every ``sweeps // 2``
    sweeps, on the latest window, up to ``max_sweeps`` (default ``3 *
    sweeps``) -- the #1035 attractor B, perturbed from a point 3e-8 off it,
    amplifies a perturbation until k~35 -- and the point is rejected only if
    some direction still grows there.

    **Known limit: a pass is "no growth seen in the window", not a proof.**
    Any fixed-length hold can pass a saddle whose unstable mode is weakly
    excited and barely faster than slowly decaying stable modes: the
    unstable share must grow by ``1 / (initial share)`` before it dominates,
    which takes ``~ log(1/share) / log(lambda_u / lambda_s)`` sweeps.  Codex
    (P1 on #1058): 9,999 modes at x0.99 plus one at x1.01 pass at
    ``sweeps=40`` (it needs ~230).  Renormalisation does not help while the
    total displacement has not shrunk.  A leading-eigenvalue estimate of the
    linearised step (Arnoldi on gauge-aligned finite-difference JVPs) would
    isolate such an outlier; not implemented.  The measured #1035 saddle is
    rejected (rates 1.006/1.045) because its unstable mode is well excited.

    Measured on #1035 (fermionic t-V D=3 chi=12 V=1 mu=2, perturbation
    1e-6, one direction, no renormalisation), fitted over ``[K/2, K]``: at
    K=20/30/40/60 the saddle S reads 1.026/1.037/1.053/1.057 and the
    attractor B 0.993/0.971/0.946/0.971.

    Statistics measured and rejected: ``d_K < d_0`` (B is still at 1.35 d_0
    after 100 sweeps); distance to the *claimed* point (plateaus at the
    point's own ``conv_tol`` residual and fits ~1 -- a false saddle); and
    successive steps ``d(y_k, y_{k-1})`` (S reads 0.988, B 1.027 at K=40).

    Args:
        step:          One CTM sweep, ``envs -> envs``.
        envs:          The claimed fixed point.
        sweeps:        First verdict at ``K`` sweeps (>= 4) per trajectory.
        perturbation:  Relative perturbation size (> 0).
        key:           PRNG key for the perturbations (default ``PRNGKey(0)``).
        contraction:   Renormalisation threshold (0 < contraction < 1).
        invariants:    Gauge-invariant fingerprint; default
                       :func:`env_spectral_invariants`.
        max_sweeps:    Longest hold before rejecting a still-growing fit
                       (default ``3 * sweeps``; >= ``sweeps``).
        directions:    Independent perturbations, all of which must pass
                       (>= 1, default 2).  Cost ``(1 + directions)`` steps
                       per hold sweep.

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
    if not 0 < contraction < 1:
        raise ValueError(f"hold_test: contraction must be in (0, 1), got {contraction}")
    if directions < 1:
        raise ValueError(f"hold_test: directions must be >= 1, got {directions}")
    if key is None:
        key = jax.random.PRNGKey(0)
    per = 1 + directions
    x = envs
    Ix = invariants(x)
    dir_keys = list(jax.random.split(key, directions))
    reseeds = [0] * directions
    ys = [perturb_env(envs, perturbation, k) for k in dir_keys]
    d0 = [env_invariant_distance(invariants(y), Ix) for y in ys]
    e0 = [_diff_norm(y, x) for y in ys]
    dcur = list(d0)
    logs = [[0.0] for _ in ys]
    half = sweeps // 2
    rates = [math.nan] * directions
    tiny = np.finfo(float).tiny

    def _result(passed, k):
        worst = int(np.nanargmax(rates)) if not all(map(math.isnan, rates)) else 0
        dist = tuple(d0[worst] * math.exp(v) for v in logs[worst])
        return HoldResult(passed, rates[worst], tuple(rates), dist, ys[worst], per * k)

    for k in range(1, max_sweeps + 1):
        x = step(x)
        Ix = invariants(x)
        for i in range(directions):
            ys[i] = step(ys[i])
            dk = env_invariant_distance(invariants(ys[i]), Ix)
            if not math.isfinite(dk) or not d0[i] > 0:
                rates[i] = math.inf
                return _result(False, k)
            if dk <= _COLLAPSE * d0[i]:
                # Codex P2 on #1058: the step mapped y onto x (to float
                # noise).  ``y - x`` is gone, so no rescale can restore it,
                # and log(tiny / tiny) = 0 would read as a flat tail -- a
                # false rejection of the strongest attractor there is.
                # Count the step as a collapse by at least 1/_COLLAPSE_LOG,
                # then re-seed the direction with a fresh deterministic
                # perturbation, keeping the accumulated log growth.
                floor = _COLLAPSE_LOG * d0[i]
                logs[i].append(logs[i][-1] + math.log(floor / max(dcur[i], tiny)))
                reseeds[i] += 1
                fresh = jax.random.fold_in(dir_keys[i], reseeds[i])
                ys[i] = perturb_env(x, perturbation, fresh)
                e0[i] = _diff_norm(ys[i], x)
                dcur[i] = env_invariant_distance(invariants(ys[i]), Ix)
                continue
            logs[i].append(logs[i][-1] + math.log(max(dk, tiny) / max(dcur[i], tiny)))
            dcur[i] = dk
            if dk < contraction * d0[i] or dk > d0[i] / contraction:
                s = d0[i] / max(dk, tiny)
                ek = _diff_norm(ys[i], x)
                # Linear regime: e scales with d.  An element-wise difference
                # far above that is a gauge mismatch -- do not amplify it.
                if math.isfinite(ek) and ek * s <= 100.0 * e0[i]:
                    ys[i] = _rescale(x, ys[i], s)
                    dcur[i] = env_invariant_distance(invariants(ys[i]), Ix)
        if k >= sweeps and ((k - sweeps) % half == 0 or k == max_sweeps):
            ks = np.arange(k - half, k + 1)
            for i in range(directions):
                rates[i] = float(np.exp(np.polyfit(ks, np.asarray(logs[i])[ks], 1)[0]))
            if all(r < 1.0 for r in rates):
                return _result(True, k)
    return _result(False, max_sweeps)
