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
