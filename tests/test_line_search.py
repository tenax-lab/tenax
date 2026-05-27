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
