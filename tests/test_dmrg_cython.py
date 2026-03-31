"""Tests for hard-coded DMRG contraction plans.

Verifies that ``build_two_site_plan`` and ``build_one_site_plan`` produce
results identical to ``np.einsum`` for various block shapes.
"""

import numpy as np
import pytest

from tenax.contraction._dmrg_plans import (
    build_one_site_plan,
    build_two_site_plan,
    get_dmrg_plan,
)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

TWO_SITE_SUBSCRIPTS = "abc,apqd,bpse,eqtf,dfg->cstg"
ONE_SITE_SUBSCRIPTS = "abc,apd,bpxe,def->cxf"


def _random_arrays(shapes, rng):
    """Return a list of random float64 arrays with given shapes."""
    return [rng.standard_normal(s) for s in shapes]


def _execute_plan(plan, arrays):
    """Execute a BlasExecPlan using the numpy fallback."""
    return plan.execute_numpy(arrays)


# ------------------------------------------------------------------ #
# Two-site plan tests                                                  #
# ------------------------------------------------------------------ #


class TestTwoSitePlan:
    """Tests for ``build_two_site_plan``."""

    @pytest.mark.parametrize(
        "shapes",
        [
            # (L, theta, W1, W2, R) — typical DMRG shapes
            # chi_l=4, chi_r=4, d=2 (spin-1/2)
            ((4, 4, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 4, 4)),
            # chi_l=8, chi_r=8, d=2
            ((8, 5, 8), (8, 2, 2, 8), (5, 2, 2, 5), (5, 2, 2, 5), (8, 5, 8)),
            # Asymmetric: chi_l != chi_r
            ((3, 5, 6), (3, 2, 2, 7), (5, 2, 3, 4), (4, 2, 3, 8), (7, 8, 6)),
            # Dimension-1 blocks (edge of chain)
            ((1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
            # Mixed dimension-1 and larger
            ((1, 3, 1), (1, 2, 2, 4), (3, 2, 2, 3), (3, 2, 2, 5), (4, 5, 1)),
            # Spin-1 (d=3) model
            ((4, 5, 4), (4, 3, 3, 4), (5, 3, 5, 5), (5, 3, 5, 5), (4, 5, 4)),
        ],
        ids=[
            "typical_d2",
            "chi8_d2",
            "asymmetric",
            "all_dim1",
            "mixed_dim1",
            "spin1_d3",
        ],
    )
    def test_matches_einsum(self, shapes):
        rng = np.random.default_rng(42)
        arrays = _random_arrays(shapes, rng)
        plan = build_two_site_plan(shapes)
        result = _execute_plan(plan, arrays)
        expected = np.einsum(TWO_SITE_SUBSCRIPTS, *arrays)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_plan_structure(self):
        shapes = ((4, 4, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = build_two_site_plan(shapes)
        assert plan.n_inputs == 5
        assert plan.n_buffers == 9
        assert len(plan.steps) == 4
        assert plan.output_perm == ()

    def test_buffer_indices(self):
        shapes = ((4, 4, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = build_two_site_plan(shapes)
        # Step 0: L(0) @ theta(1) -> I0(5)
        assert plan.steps[0].left_idx == 0
        assert plan.steps[0].right_idx == 1
        assert plan.steps[0].out_idx == 5
        # Step 1: I0(5) @ W1(2) -> I1(6)
        assert plan.steps[1].left_idx == 5
        assert plan.steps[1].right_idx == 2
        assert plan.steps[1].out_idx == 6
        # Step 2: I1(6) @ W2(3) -> I2(7)
        assert plan.steps[2].left_idx == 6
        assert plan.steps[2].right_idx == 3
        assert plan.steps[2].out_idx == 7
        # Step 3: I2(7) @ R(4) -> out(8)
        assert plan.steps[3].left_idx == 7
        assert plan.steps[3].right_idx == 4
        assert plan.steps[3].out_idx == 8


# ------------------------------------------------------------------ #
# One-site plan tests                                                  #
# ------------------------------------------------------------------ #


class TestOneSitePlan:
    """Tests for ``build_one_site_plan``."""

    @pytest.mark.parametrize(
        "shapes",
        [
            # (L, site, W, R) — typical DMRG shapes
            # chi_l=4, chi_r=4, d=2
            ((4, 4, 4), (4, 2, 4), (4, 2, 2, 4), (4, 4, 4)),
            # chi_l=8, chi_r=8, d=2
            ((8, 5, 8), (8, 2, 8), (5, 2, 2, 5), (8, 5, 8)),
            # Asymmetric: chi_l != chi_r
            ((3, 5, 6), (3, 2, 7), (5, 2, 3, 4), (7, 4, 6)),
            # Dimension-1 blocks (edge of chain)
            ((1, 1, 1), (1, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
            # Mixed dimension-1 and larger
            ((1, 3, 1), (1, 2, 4), (3, 2, 2, 3), (4, 3, 1)),
            # Spin-1 (d=3) model
            ((4, 5, 4), (4, 3, 4), (5, 3, 5, 5), (4, 5, 4)),
        ],
        ids=[
            "typical_d2",
            "chi8_d2",
            "asymmetric",
            "all_dim1",
            "mixed_dim1",
            "spin1_d3",
        ],
    )
    def test_matches_einsum(self, shapes):
        rng = np.random.default_rng(42)
        arrays = _random_arrays(shapes, rng)
        plan = build_one_site_plan(shapes)
        result = _execute_plan(plan, arrays)
        expected = np.einsum(ONE_SITE_SUBSCRIPTS, *arrays)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_plan_structure(self):
        shapes = ((4, 4, 4), (4, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = build_one_site_plan(shapes)
        assert plan.n_inputs == 4
        assert plan.n_buffers == 7
        assert len(plan.steps) == 3
        assert plan.output_perm == ()

    def test_buffer_indices(self):
        shapes = ((4, 4, 4), (4, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = build_one_site_plan(shapes)
        # Step 0: L(0) @ site(1) -> I0(4)
        assert plan.steps[0].left_idx == 0
        assert plan.steps[0].right_idx == 1
        assert plan.steps[0].out_idx == 4
        # Step 1: I0(4) @ W(2) -> I1(5)
        assert plan.steps[1].left_idx == 4
        assert plan.steps[1].right_idx == 2
        assert plan.steps[1].out_idx == 5
        # Step 2: I1(5) @ R(3) -> out(6)
        assert plan.steps[2].left_idx == 5
        assert plan.steps[2].right_idx == 3
        assert plan.steps[2].out_idx == 6


# ------------------------------------------------------------------ #
# Dispatcher tests                                                     #
# ------------------------------------------------------------------ #


class TestGetDmrgPlan:
    """Tests for the cached dispatcher ``get_dmrg_plan``."""

    def test_two_site_dispatch(self):
        shapes = ((4, 4, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = get_dmrg_plan(TWO_SITE_SUBSCRIPTS, shapes)
        assert plan.n_inputs == 5
        assert len(plan.steps) == 4

    def test_one_site_dispatch(self):
        shapes = ((4, 4, 4), (4, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan = get_dmrg_plan(ONE_SITE_SUBSCRIPTS, shapes)
        assert plan.n_inputs == 4
        assert len(plan.steps) == 3

    def test_fallback_to_generic(self):
        """Unknown subscripts should fall back to the generic planner."""
        subscripts = "ij,jk->ik"
        shapes = ((3, 4), (4, 5))
        plan = get_dmrg_plan(subscripts, shapes)
        # Should still produce a valid plan
        rng = np.random.default_rng(42)
        arrays = _random_arrays(shapes, rng)
        result = plan.execute_numpy(arrays)
        expected = np.einsum(subscripts, *arrays)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_caching(self):
        """Same inputs should return the same plan object (cached)."""
        shapes = ((4, 4, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 2, 2, 4), (4, 4, 4))
        plan1 = get_dmrg_plan(TWO_SITE_SUBSCRIPTS, shapes)
        plan2 = get_dmrg_plan(TWO_SITE_SUBSCRIPTS, shapes)
        assert plan1 is plan2


# ------------------------------------------------------------------ #
# Cross-validation with generic build_blas_plan                        #
# ------------------------------------------------------------------ #


class TestCrossValidation:
    """Verify hard-coded plans produce identical results to the generic planner."""

    def test_two_site_vs_generic(self):
        from tenax.contraction._blas_plan import build_blas_plan

        shapes = [(3, 5, 6), (3, 2, 2, 7), (5, 2, 3, 4), (4, 2, 3, 8), (7, 8, 6)]
        rng = np.random.default_rng(123)
        arrays = _random_arrays(shapes, rng)

        plan_hc = build_two_site_plan(shapes)
        plan_gen = build_blas_plan(TWO_SITE_SUBSCRIPTS, shapes)

        result_hc = plan_hc.execute_numpy(arrays)
        result_gen = plan_gen.execute_numpy(arrays)
        expected = np.einsum(TWO_SITE_SUBSCRIPTS, *arrays)

        np.testing.assert_allclose(result_hc, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result_gen, expected, rtol=1e-12, atol=1e-14)

    def test_one_site_vs_generic(self):
        from tenax.contraction._blas_plan import build_blas_plan

        shapes = [(3, 5, 6), (3, 2, 7), (5, 2, 3, 4), (7, 4, 6)]
        rng = np.random.default_rng(123)
        arrays = _random_arrays(shapes, rng)

        plan_hc = build_one_site_plan(shapes)
        plan_gen = build_blas_plan(ONE_SITE_SUBSCRIPTS, shapes)

        result_hc = plan_hc.execute_numpy(arrays)
        result_gen = plan_gen.execute_numpy(arrays)
        expected = np.einsum(ONE_SITE_SUBSCRIPTS, *arrays)

        np.testing.assert_allclose(result_hc, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result_gen, expected, rtol=1e-12, atol=1e-14)
