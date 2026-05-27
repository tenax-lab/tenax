"""Hager-Zhang line search with approximate Wolfe conditions.

Reference: Hager & Zhang, SIAM J. Optim. 16(1), 170-192, 2005.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def hager_zhang_line_search(
    phi: Callable[[float], float],
    dphi: Callable[[float], float],
    phi0: float,
    dphi0: float,
    alpha_init: float = 1.0,
    delta: float = 0.1,
    sigma: float = 0.9,
    eps_factor: float = 1e-6,
    theta: float = 0.5,
    gamma: float = 0.66,
    rho: float = 5.0,
    max_iter: int = 40,
    max_step: float | None = None,
    energy_bound: float | None = None,
    bracket_only_phi: bool = False,
) -> tuple[float, float, bool]:
    """Hager-Zhang line search with approximate Wolfe conditions.

    Args:
        phi: Function alpha -> f(x + alpha*d). Python-level, not JIT-traced.
        dphi: Function alpha -> nabla f(x + alpha*d) . d. Python-level.
        phi0: phi(0) = f(x).
        dphi0: dphi(0) = nabla f(x) . d. Must be < 0 (descent direction).
        alpha_init: Initial step size guess.
        delta: Sufficient decrease parameter (0 < delta < sigma).
        sigma: Curvature parameter (delta < sigma < 1).
        eps_factor: Approximate Wolfe tolerance factor.
        theta: Bisection balance parameter.
        gamma: Interval shrinkage parameter.
        rho: Expansion factor for initial bracketing.
        max_iter: Maximum number of iterations.
        max_step: Maximum allowed step size. If set, alpha is clipped to this.
        energy_bound: Reject trial points where ``|phi(alpha)| > energy_bound``.
        bracket_only_phi: When True, skip ``dphi`` evaluation during the
            bracket-expansion phase.  Bracket detection then relies
            solely on the energy-excess criterion ``phi(c) > phi0 +
            eps``; the slope-sign-change shortcut and the Wolfe-OK
            early exit are unavailable until the zoom phase.  Saves one
            implicit-AD backward (Neumann/GMRES adjoint + chain rule)
            per bracket probe.  Default is ``False`` (legacy behavior):
            monotonically-decreasing ``phi`` (e.g. ``exp(-a) - 1``) can
            satisfy Wolfe at a small ``alpha`` without ever crossing
            ``phi0 + eps``, so opt-in on this flag is reserved for
            callers whose probes are expensive enough to justify the
            extra zoom-phase iterations.  iPEPS HZ call sites pass
            ``True`` explicitly to capture the implicit-AD backward
            savings (issue #504).
    """
    # Not a descent direction
    if dphi0 >= 0:
        return 0.0, phi0, False

    eps = eps_factor * abs(phi0)

    # Track best point seen (lowest phi value) as fallback
    best_alpha = 0.0
    best_phi = phi0

    def _is_finite(x: float) -> bool:
        return math.isfinite(x)

    def _safe_phi(alpha: float) -> float:
        try:
            val = float(phi(alpha))
        except Exception:
            return float("inf")
        if not _is_finite(val):
            return float("inf")
        if energy_bound is not None and abs(val) > energy_bound:
            return float("inf")
        return val

    def _safe_dphi(alpha: float) -> float:
        try:
            val = float(dphi(alpha))
        except Exception:
            return float("nan")
        return val

    def _update_best(alpha: float, f: float) -> None:
        nonlocal best_alpha, best_phi
        if _is_finite(f) and f < best_phi:
            best_alpha = alpha
            best_phi = f

    def _standard_wolfe(alpha: float, f: float, df: float) -> bool:
        """Check standard Wolfe conditions."""
        return f <= phi0 + delta * alpha * dphi0 and df >= sigma * dphi0

    def _approx_wolfe(alpha: float, f: float, df: float) -> bool:
        """Check approximate Wolfe conditions (relaxed decrease)."""
        return f <= phi0 + eps and sigma * dphi0 <= df <= (2 * delta - 1) * dphi0

    def _wolfe_ok(alpha: float, f: float, df: float) -> bool:
        return _standard_wolfe(alpha, f, df) or _approx_wolfe(alpha, f, df)

    # ------------------------------------------------------------------
    # Secant step between two points with known derivatives
    # ------------------------------------------------------------------
    def _secant_step(a: float, da: float, b: float, db: float) -> float:
        denom = da - db
        if abs(denom) < 1e-30:
            return (a + b) / 2.0
        return b - db * (b - a) / denom

    # ------------------------------------------------------------------
    # "update" procedure: given bracket [a, b] and a trial point c,
    # return a tighter bracket.
    # Precondition: dphi(a) < 0, phi(a) <= phi0 + eps,
    #               dphi(b) >= 0 or phi(b) > phi0 + eps.
    # ------------------------------------------------------------------
    def _update(
        a: float, fa: float, da: float, b: float, fb: float, db: float, c: float
    ) -> tuple[float, float, float, float, float, float]:
        fc = _safe_phi(c)
        _update_best(c, fc)
        if not _is_finite(fc):
            # Treat as too high — shrink right endpoint
            return a, fa, da, c, fc, 0.0

        if fc > phi0 + eps:
            # c is too high — it becomes the new right endpoint
            return a, fa, da, c, fc, 0.0

        dc = _safe_dphi(c)
        if dc >= 0:
            # c has non-negative slope — it becomes right endpoint
            return a, fa, da, c, fc, dc

        # phi(c) is low and dphi(c) < 0 — c becomes the new left endpoint
        return c, fc, dc, b, fb, db

    # ------------------------------------------------------------------
    # Bisection on the bracket to guarantee sufficient shrinkage.
    # ------------------------------------------------------------------
    def _bisect(
        a: float,
        fa: float,
        da: float,
        b: float,
        fb: float,
        db: float,
        max_bisect: int = 50,
    ) -> tuple[float, float, float, float, float, float]:
        for _ in range(max_bisect):
            if abs(b - a) < 1e-15:
                break
            mid = (1 - theta) * a + theta * b
            fm = _safe_phi(mid)
            _update_best(mid, fm)
            if fm > phi0 + eps:
                b, fb, db = mid, fm, 0.0
            else:
                dm = _safe_dphi(mid)
                if dm >= 0:
                    return a, fa, da, mid, fm, dm
                a, fa, da = mid, fm, dm
        return a, fa, da, b, fb, db

    # ==================================================================
    # Phase 1: Bracket — expand from alpha_init
    # ==================================================================
    c_prev = 0.0
    f_prev = phi0
    d_prev = dphi0
    c = alpha_init
    if max_step is not None:
        c = min(c, max_step)

    for it in range(max_iter):
        fc = _safe_phi(c)
        _update_best(c, fc)

        if not _is_finite(fc):
            # NaN/inf — treat as too high, bracket is [c_prev, c]
            a, fa, da = c_prev, f_prev, d_prev
            b, fb, db = c, fc, 0.0
            break

        # Issue #504: phi-only bracket skips the per-probe dphi call.
        # Bracket detection then relies solely on the energy-excess
        # branch (``fc > phi0 + eps``) below; the slope-sign-change
        # shortcut and the Wolfe-OK early exit are unavailable until
        # the zoom phase.  ``d_prev`` is propagated as NaN so the zoom
        # secant degrades gracefully (``da if finite else 0.0``).
        if bracket_only_phi:
            dc = float("nan")
        else:
            dc = _safe_dphi(c)

            # Check Wolfe at this point
            if _wolfe_ok(c, fc, dc):
                return c, fc, True

            if dc >= 0:
                # Found bracket: slope changed sign
                # The bracket is [c_prev, c] but we need dphi(a) < 0
                a, fa, da = c_prev, f_prev, d_prev
                b, fb, db = c, fc, dc
                break

        if fc > phi0 + eps:
            # phi too high — bracket using bisection from [0, c]
            a, fa, da = 0.0, phi0, dphi0
            b, fb, db = c, fc, 0.0
            a, fa, da, b, fb, db = _bisect(a, fa, da, b, fb, db)
            break

        # Expand
        c_prev, f_prev, d_prev = c, fc, dc
        c = rho * c
        if max_step is not None:
            c = min(c, max_step)
            if c <= c_prev:
                # Hit max_step — bracket with what we have
                a, fa, da = c_prev, f_prev, d_prev
                b, fb, db = c, _safe_phi(c), 0.0
                _update_best(c, fb)
                break
    else:
        # Exhausted iterations in bracket phase
        return best_alpha, best_phi, False

    # ==================================================================
    # Phase 2: Zoom — secant-based bisection within the bracket
    # ==================================================================
    for it in range(max_iter):
        if abs(b - a) < 1e-15:
            break

        # Secant step using derivatives at bracket endpoints
        da_val = da if _is_finite(da) else 0.0
        db_val = db if _is_finite(db) else 0.0
        c = _secant_step(a, da_val, b, db_val)

        # Safeguard: ensure c is inside (a, b)
        lo, hi = min(a, b), max(a, b)
        margin = 0.01 * (hi - lo)
        if not (lo + margin < c < hi - margin):
            c = (a + b) / 2.0

        fc = _safe_phi(c)
        _update_best(c, fc)

        if not _is_finite(fc):
            b, fb, db = c, fc, 0.0
            continue

        dc = _safe_dphi(c)

        # Check Wolfe
        if _wolfe_ok(c, fc, dc):
            return c, fc, True

        # Update bracket
        a, fa, da, b, fb, db = _update(a, fa, da, b, fb, db, c)

        # Check if bracket is shrinking enough; if not, bisect
        old_width = hi - lo
        new_width = abs(b - a)
        if new_width > gamma * old_width:
            a, fa, da, b, fb, db = _bisect(a, fa, da, b, fb, db)

    # Return best point found even if Wolfe not satisfied
    return best_alpha, best_phi, False
