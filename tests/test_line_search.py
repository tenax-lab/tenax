"""Tests for Hager-Zhang line search."""

import pytest


class TestHagerZhangLineSearch:
    def test_quadratic_exact_minimum(self):
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(alpha):
            return float((alpha - 3.0) ** 2)

        def dphi(alpha):
            return float(2.0 * (alpha - 3.0))

        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi(0.0), dphi(0.0), alpha_init=1.0
        )
        assert converged
        assert f_alpha < phi(0.0)
        # Wolfe conditions don't require finding the exact minimum,
        # just a point with sufficient decrease and curvature.
        assert alpha > 0

    def test_returns_decrease(self):
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(alpha):
            return float(alpha**2 - 2 * alpha + 3)

        def dphi(alpha):
            return float(2 * alpha - 2)

        phi0, dphi0 = phi(0.0), dphi(0.0)
        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi0, dphi0, alpha_init=1.0
        )
        assert f_alpha < phi0

    def test_not_descent_returns_zero(self):
        from tenax.algorithms._line_search import hager_zhang_line_search

        alpha, f_alpha, converged = hager_zhang_line_search(
            lambda a: a**2,
            lambda a: 2 * a,
            0.0,
            1.0,  # dphi0 > 0
        )
        assert alpha == 0.0
        assert not converged

    def test_wolfe_conditions_satisfied(self):
        from tenax.algorithms._line_search import hager_zhang_line_search

        delta, sigma = 0.1, 0.9

        def phi(a):
            return float((a - 2.0) ** 2 + 1.0)

        def dphi(a):
            return float(2 * (a - 2.0))

        phi0, dphi0 = phi(0.0), dphi(0.0)
        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi0, dphi0, delta=delta, sigma=sigma
        )
        if converged:
            # Check standard Wolfe
            assert (
                f_alpha <= phi0 + delta * alpha * dphi0
                or f_alpha <= phi0 + 1e-6 * abs(phi0)
            )
            assert dphi(alpha) >= sigma * dphi0

    def test_no_jax_dependency(self):
        """Line search should work with plain Python floats."""
        from tenax.algorithms._line_search import hager_zhang_line_search

        call_count = [0]

        def phi(a):
            call_count[0] += 1
            return float((a - 1.5) ** 2 + 1.0)

        def dphi(a):
            return float(2 * (a - 1.5))

        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi(0.0), dphi(0.0)
        )
        assert call_count[0] > 0
        assert f_alpha < phi(0.0)


class TestLineSearchSafety:
    def test_respects_max_step(self):
        """Line search should not exceed max_step."""
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(a):
            return float((a - 100.0) ** 2)  # minimum far away

        def dphi(a):
            return float(2 * (a - 100.0))

        alpha, _, _ = hager_zhang_line_search(
            phi, dphi, phi(0.0), dphi(0.0), alpha_init=1.0, max_step=2.0
        )
        assert alpha <= 2.0, f"alpha={alpha} exceeded max_step=2.0"

    def test_rejects_unphysical_energy(self):
        """Line search should reject points with |E| > energy_bound."""
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(a):
            val = -0.5 - 10 * a  # energy goes unphysically low
            return float(val)

        def dphi(a):
            return -10.0

        alpha, f_alpha, _ = hager_zhang_line_search(
            phi,
            dphi,
            phi(0.0),
            dphi(0.0),
            alpha_init=1.0,
            energy_bound=5.0,
        )
        # Should not accept any point with |E| > 5
        assert abs(f_alpha) <= 5.0 or alpha == 0.0

    def test_max_step_preserves_convergence(self):
        """With a reasonable max_step, should still find a good point."""
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(a):
            return float((a - 1.5) ** 2 + 1.0)

        def dphi(a):
            return float(2 * (a - 1.5))

        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi(0.0), dphi(0.0), alpha_init=1.0, max_step=5.0
        )
        assert converged
        assert f_alpha < phi(0.0)


class TestHagerZhangBracketSkipDphi:
    """Issue #504: ``bracket_only_phi`` defers dphi to the zoom phase."""

    def _quadratic(self, *, minimum: float = 3.0):
        """Convex quadratic with min at ``minimum``.  Returns (phi, dphi)
        plus call-count dicts for each so tests can introspect them."""
        phi_n = {"n": 0}
        dphi_n = {"n": 0}

        def phi(a):
            phi_n["n"] += 1
            return float((a - minimum) ** 2)

        def dphi(a):
            dphi_n["n"] += 1
            return float(2.0 * (a - minimum))

        return phi, dphi, phi_n, dphi_n

    def test_bracket_only_phi_skips_dphi_during_expansion(self):
        """In a pure bracket-expansion scenario, flag=True calls 0 dphi.

        ``phi(α) = -α`` is monotonically decreasing, so the bracket loop
        never finds an energy-excess; ``dphi(α) = -1`` is constant and
        Wolfe never fires.  The bracket phase fills the entire
        ``max_iter`` budget — flag-off calls dphi once per probe;
        flag-on calls dphi zero times in this regime.
        """
        from tenax.algorithms._line_search import hager_zhang_line_search

        def make_probes():
            n = {"phi": 0, "dphi": 0}

            def phi(a):
                n["phi"] += 1
                return float(-a)

            def dphi(_a):
                n["dphi"] += 1
                return -1.0

            return phi, dphi, n

        phi, dphi, n_off = make_probes()
        phi0_off, dphi0_off = phi(0.0), dphi(0.0)
        n_off["phi"] = 0
        n_off["dphi"] = 0
        hager_zhang_line_search(
            phi,
            dphi,
            phi0_off,
            dphi0_off,
            alpha_init=1.0,
            max_step=1e6,
            max_iter=8,
            bracket_only_phi=False,
        )
        dphi_off = n_off["dphi"]

        phi, dphi, n_on = make_probes()
        phi0_on, dphi0_on = phi(0.0), dphi(0.0)
        n_on["phi"] = 0
        n_on["dphi"] = 0
        hager_zhang_line_search(
            phi,
            dphi,
            phi0_on,
            dphi0_on,
            alpha_init=1.0,
            max_step=1e6,
            max_iter=8,
            bracket_only_phi=True,
        )
        dphi_on = n_on["dphi"]

        # Flag-off must call dphi at least once per bracket probe.
        assert dphi_off >= 4, (
            f"flag-off must call dphi every probe in this regime; got {dphi_off}"
        )
        # Flag-on must call dphi zero times during the entire run because
        # the bracket loop never finds an excess and there is no zoom.
        assert dphi_on == 0, f"flag-on must skip dphi in bracket phase; got {dphi_on}"
        # Phi-only mode also runs the same probe count overall (bracket
        # expansion doesn't depend on dphi).
        assert n_on["phi"] >= 4, (
            f"flag-on should still run multiple phi probes; got {n_on['phi']}"
        )

    def test_bracket_only_phi_off_matches_legacy_behavior(self):
        """``bracket_only_phi=False`` must yield identical alpha/f to the
        pre-issue #504 implementation.  Smooth quadratic where the
        slope-sign-change shortcut at ``dc >= 0`` would have fired."""
        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(a):
            return float((a - 2.0) ** 2 + 1.0)

        def dphi(a):
            return float(2 * (a - 2.0))

        # alpha_init=1.5 puts the first probe near the minimum; with
        # dphi available, Wolfe-OK or dc>=0 detection should fire fast.
        alpha, f_alpha, converged = hager_zhang_line_search(
            phi,
            dphi,
            phi(0.0),
            dphi(0.0),
            alpha_init=1.5,
            bracket_only_phi=False,
        )
        assert converged
        # Wolfe with delta=0.1, sigma=0.9 on (a-2)²+1 — accepted alpha
        # should be close to the minimum at a=2.
        assert abs(alpha - 2.0) < 1.0

    def test_default_converges_on_monotone_decreasing_phi(self):
        """Codex P1 regression on #539: ``phi(a) = exp(-a) - 1`` is
        monotonically decreasing, so ``phi(c) > phi0 + eps`` never fires.
        Under the new ``bracket_only_phi=False`` default the Wolfe-OK
        shortcut at ``α=1`` runs as before and the line search returns
        ``converged=True``.  If the default ever flipped back to ``True``,
        this test would fail (bracket exhausts max_iter)."""
        import math

        from tenax.algorithms._line_search import hager_zhang_line_search

        def phi(a):
            return float(math.exp(-a) - 1.0)

        def dphi(a):
            return float(-math.exp(-a))

        alpha, f_alpha, converged = hager_zhang_line_search(
            phi, dphi, phi(0.0), dphi(0.0), alpha_init=1.0
        )
        assert converged, (
            "default bracket_only_phi must converge on monotone-decreasing "
            f"phi; got alpha={alpha} f_alpha={f_alpha}"
        )
        assert f_alpha < 0.0


def test_aborted_probe_returns_best_phi_point_immediately():
    """A probe raising LineSearchAborted ends the search at once (#1059: an
    unconverged dphi forward), returning the best phi point seen so far."""
    from tenax.algorithms._line_search import (
        LineSearchAborted,
        hager_zhang_line_search,
    )

    calls = {"phi": 0, "dphi": 0}

    def phi(a):
        calls["phi"] += 1
        return (a - 1.0) ** 2 - 1.0

    def dphi(a):
        calls["dphi"] += 1
        raise LineSearchAborted

    alpha, f, converged = hager_zhang_line_search(
        phi, dphi, phi(0.0), -2.0, alpha_init=0.5, bracket_only_phi=True
    )
    assert calls["dphi"] == 1
    assert not converged
    assert alpha > 0 and f < 0.0  # the best phi point, not alpha=0


class TestPhiOnlyBracketSeesSubEpsOvershoot:
    """The phi-only bracket must see an overshoot smaller than eps.

    Near convergence the energy rise at an overshooting probe is
    ~|g|^2, far below ``eps = eps_factor*|phi0|`` (6.6e-7 at E=-0.66).
    Judged by ``phi > phi0 + eps`` alone, every probe passed, the bracket
    grew to ``max_step`` and collapsed, and the search returned alpha=0
    (the log signature ``HZ probes phi=4 dphi=0 alpha=0``).  The iPEPS
    optimizer then stalled at |grad| ~ 3e-4 and could never meet the
    default grad_norm < 1e-5 test.  Numbers below are the iPEPS call
    site's: alpha_init=1, rho=1.5, max_step=2, |g| from the stalled
    D=2 chi=16 Heisenberg run.
    """

    PHI0 = -0.6625142352
    G2 = 3.372e-4**2  # steepest descent: dphi0 = -|g|^2

    def _search(self, argmin):
        from tenax.algorithms._line_search import hager_zhang_line_search

        k = self.G2 / argmin  # quadratic with its minimum at alpha=argmin

        def phi(a):
            return self.PHI0 - self.G2 * a + 0.5 * k * a * a

        def dphi(a):
            return -self.G2 + k * a

        # The regime this test exists for: the whole rise over the probed
        # range is below eps, so the eps band alone cannot see it.
        eps = 1e-6 * abs(self.PHI0)
        assert phi(2.0) - self.PHI0 < eps
        return hager_zhang_line_search(
            phi,
            dphi,
            self.PHI0,
            -self.G2,
            alpha_init=1.0,
            rho=1.5,
            max_step=2.0,
            bracket_only_phi=True,
        )

    @pytest.mark.parametrize("argmin", [0.3, 0.6])
    def test_overshoot_below_eps_still_decreases(self, argmin):
        alpha, f_alpha, converged = self._search(argmin)
        assert alpha > 0.0
        assert f_alpha < self.PHI0 - 1e-12  # a real decrease, as the stall test asks
        assert converged


class TestBisectAcceptsWolfe:
    """Codex P1 on #1078: an accepted point must end the search.

    ``phi(a) = -0.66 - 0.1*(1 - exp(-10a))`` flattens monotonically.  At
    alpha=1 it fails sufficient decrease yet satisfies approximate Wolfe;
    bisection that stops only on ``dphi >= 0`` never stops on it and pays
    one ``dphi`` -- an implicit-AD backward on iPEPS -- per pass, up to 50.
    """

    @staticmethod
    def _run(phi_fn, dphi_fn):
        from tenax.algorithms._line_search import hager_zhang_line_search

        n = {"phi": 0, "dphi": 0}

        def phi(a):
            n["phi"] += 1
            return phi_fn(a)

        def dphi(a):
            n["dphi"] += 1
            return dphi_fn(a)

        out = hager_zhang_line_search(
            phi,
            dphi,
            phi_fn(0.0),
            dphi_fn(0.0),
            alpha_init=1.0,
            rho=1.5,
            max_step=2.0,
            bracket_only_phi=True,
        )
        return out, n

    def test_sufficient_decrease_failure_accepts_a_wolfe_probe(self):
        import math

        (alpha, _, converged), n = self._run(
            lambda a: -0.66 - 0.1 * (1 - math.exp(-10 * a)),
            lambda a: -math.exp(-10 * a),
        )
        assert converged and alpha == 1.0
        assert n["dphi"] == 1

    def test_bisection_accepts_a_wolfe_midpoint(self):
        """Same flattening phi with a steep wall past alpha=0.9: the probe
        at alpha=1 rises far above eps, so the search bisects [0, 1].  The
        slope stays negative up to the wall, so without a Wolfe check the
        bisection walks to the wall (7 dphi here, against 1)."""
        import math

        def phi(a):
            return -0.66 - 0.1 * (1 - math.exp(-10 * a)) + 1e4 * max(0.0, a - 0.9) ** 2

        def dphi(a):
            return -math.exp(-10 * a) + 2e4 * max(0.0, a - 0.9)

        (_, _, converged), n = self._run(phi, dphi)
        assert phi(1.0) > phi(0.0)  # the regime: alpha=1 is past the wall
        assert converged
        assert n["dphi"] == 1


class TestApproxWolfeRequiresDecrease:
    """An approximate-Wolfe point must lower phi.

    The probes are the ones traced at the first stall of the 1x1 D=3
    Heisenberg run (chi=16, main c4bc0f8): alpha_init=0.1908 rises 7.3e-6
    (above eps), the bisection midpoint 0.0954 rises 3.2e-7 (below eps)
    with slope +3.9e-5.  That midpoint met the relaxed Wolfe test, so the
    search returned a rise; the optimizer, which needs a decrease, counted
    a stall and repeated the identical search until its budget ran out.
    phi is the cubic Hermite through those values on [0, h] (it dips
    below phi0 inside) and a quadratic through the alpha_init probe past h.
    """

    PHI0 = -0.668165747702
    S = -1.787e-4  # dphi0
    H, FH, GH = 0.09541529, 3.194e-7, 3.910e-5  # traced midpoint
    A0, FA0 = 0.19083058, 7.287e-6  # traced alpha_init probe

    def _phi_dphi(self):
        h, s, F, G = self.H, self.S, self.FH, self.GH
        K = (self.FA0 - F - G * (self.A0 - h)) / (self.A0 - h) ** 2

        def phi(a):
            if a <= h:
                t = a / h
                h10 = t**3 - 2 * t**2 + t
                h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
                return self.PHI0 + h10 * h * s + h01 * F + h11 * h * G
            x = a - h
            return self.PHI0 + F + G * x + K * x * x

        def dphi(a):
            if a <= h:
                t = a / h
                d10, d01, d11 = (
                    3 * t**2 - 4 * t + 1,
                    (-6 * t**2 + 6 * t) / h,
                    3 * t**2 - 2 * t,
                )
                return d10 * s + d01 * F + d11 * G
            return G + 2 * K * (a - h)

        return phi, dphi

    def test_a_sub_eps_rise_is_not_accepted(self):
        from tenax.algorithms._line_search import hager_zhang_line_search

        phi, dphi = self._phi_dphi()
        eps = 1e-6 * abs(self.PHI0)
        # The regime: the midpoint rises by less than eps, and its slope is
        # inside the relaxed curvature band -- the old test accepted it.
        assert 0.0 < phi(self.H) - self.PHI0 < eps
        assert phi(self.A0) - self.PHI0 > eps
        assert 0.9 * self.S <= dphi(self.H) <= -0.8 * self.S
        alpha, f_alpha, converged = hager_zhang_line_search(
            phi,
            dphi,
            self.PHI0,
            self.S,
            alpha_init=self.A0,
            rho=1.5,
            max_step=2 * self.A0,
            bracket_only_phi=True,
        )
        assert f_alpha < self.PHI0
        assert 0.0 < alpha < self.H


class TestBracketEndsNeedADecrease:
    """A point that does not lower phi must not become the left bracket end.

    The probes are the ones traced at the stall state of the 1x1 D=3
    Heisenberg run (chi=16, branch 1af82da, |grad| = 1.0e-3).  phi dips by
    8.8e-8 near alpha = 0.025, rises over a hump, and has a second local
    minimum at alpha = 0.261 that sits 1.8e-7 *above* phi0 -- below
    eps = 6.7e-7.  The bisection made alpha = 0.2276 (rise 1.85e-7, slope
    -2.1e-7) the left end because its rise was inside the eps band; the zoom
    then converged on the high minimum, where no Wolfe test that requires a
    decrease can pass, and returned alpha = 0 after 40 iterations.  phi is
    the piecewise cubic Hermite through the traced values and slopes (the
    two extrema between the probes are estimated from the scan).
    """

    PHI0 = -0.668182976262
    S = -7.1505e-6  # dphi0
    A0 = 0.9103308  # traced alpha_init
    # (alpha, phi - phi0, dphi): low minimum, hump, high minimum, probes.
    KNOTS = (
        (0.0, 0.0, S),
        (0.0247, -8.8e-8, 0.0),
        (0.15, 2.0e-7, 0.0),
        (0.2611, 1.813e-7, 0.0),
        (0.4552, 3.607e-7, 1.9924e-6),
        (0.9103, 2.580e-6, 8.0e-6),
    )

    def _phi_dphi(self):
        knots = self.KNOTS
        a_end, f_end, g_end = knots[-1]

        def _seg(a):
            for (x0, f0, g0), (x1, f1, g1) in zip(knots, knots[1:]):
                if a <= x1:
                    return x0, f0, g0, x1, f1, g1
            return None

        def phi(a):
            seg = _seg(a)
            if seg is None:  # quadratic continuation past the last probe
                x = a - a_end
                return self.PHI0 + f_end + g_end * x + 1e-5 * x * x
            x0, f0, g0, x1, f1, g1 = seg
            h = x1 - x0
            t = (a - x0) / h
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            return self.PHI0 + h00 * f0 + h10 * h * g0 + h01 * f1 + h11 * h * g1

        def dphi(a):
            seg = _seg(a)
            if seg is None:
                return g_end + 2e-5 * (a - a_end)
            x0, f0, g0, x1, f1, g1 = seg
            h = x1 - x0
            t = (a - x0) / h
            d00, d10 = (6 * t**2 - 6 * t) / h, 3 * t**2 - 4 * t + 1
            d01, d11 = (-6 * t**2 + 6 * t) / h, 3 * t**2 - 2 * t
            return d00 * f0 + d10 * g0 + d01 * f1 + d11 * g1

        return phi, dphi

    def _search(self, alpha_init, eps_factor=1e-6, bracket_only_phi=True):
        from tenax.algorithms._line_search import hager_zhang_line_search

        phi, dphi = self._phi_dphi()
        eps = eps_factor * abs(self.PHI0)
        # The regime: a decrease exists near 0; the traced left end and the
        # high minimum rise by less than eps; alpha_init rises by more than
        # phi0 and fails sufficient decrease.
        assert phi(0.0247) < self.PHI0
        assert 0.0 < phi(0.2276) - self.PHI0 < eps and dphi(0.2276) < 0.0
        assert 0.0 < phi(0.2611) - self.PHI0 < eps
        assert phi(alpha_init) > self.PHI0
        alpha, f_alpha, converged = hager_zhang_line_search(
            phi,
            dphi,
            self.PHI0,
            self.S,
            alpha_init=alpha_init,
            eps_factor=eps_factor,
            rho=1.5,
            max_step=2 * alpha_init,
            max_iter=40,
            bracket_only_phi=bracket_only_phi,
        )
        assert converged
        assert f_alpha < self.PHI0
        assert 0.0 < alpha < 0.15

    def test_the_traced_search_finds_the_low_minimum(self):
        # alpha_init rises above eps; either bracket-end test below
        # catches 0.2276, so this pins the pair, not each one.
        self._search(self.A0)

    def test_zoom_update_rejects_a_sub_eps_rise(self):
        # The derivative bracket: phi'(0.4552) > 0 brackets [0, 0.4552]
        # without a bisection, the zoom's first point is the midpoint
        # 0.2276, and only _update decides which end it becomes.
        phi, dphi = self._phi_dphi()
        assert dphi(0.4552) > 0.0
        self._search(0.4552, bracket_only_phi=False)

    def test_bisection_rejects_a_sub_eps_rise(self):
        # eps = 2.7e-7 puts phi(0.4552) = 3.6e-7 above the band, so the
        # phi-only bracket bisects [0, 0.4552]; its first midpoint 0.2276
        # (rise 1.85e-7, slope < 0) is judged inside _bisect.
        phi, _ = self._phi_dphi()
        eps = 4e-7 * abs(self.PHI0)
        assert phi(0.4552) - self.PHI0 > eps
        self._search(0.4552, eps_factor=4e-7)

    def test_derivative_bracket_rejects_a_sub_eps_rise(self):
        # Codex P2 on #1079: with dphi at every bracket probe, 0.2276
        # (rise 1.85e-7, slope < 0) fails Wolfe but was kept as c_prev; the
        # next probe's slope is >= 0, so the bracket [0.2276, 0.3414] lay
        # to the right of every decrease.
        _, dphi = self._phi_dphi()
        assert dphi(0.2276 * 1.5) >= 0.0
        self._search(0.2276, bracket_only_phi=False)
