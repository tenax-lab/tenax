"""Tests for the BLAS execution plan builder."""

import numpy as np
import pytest

from tenax.contraction._blas_plan import (
    BlasExecPlan,
    GemmStep,
    build_blas_plan,
    get_cached_blas_plan,
)


class TestPlanStructure:
    """Tests for plan structure (steps, buffer counts, GEMM dimensions)."""

    def test_pairwise_matmul(self):
        """'ij,jk->ik' produces 1 step with correct M, N, K."""
        plan = build_blas_plan("ij,jk->ik", [(3, 4), (4, 5)])
        assert isinstance(plan, BlasExecPlan)
        assert len(plan.steps) == 1
        assert plan.n_inputs == 2

        step = plan.steps[0]
        assert step.m == 3
        assert step.n == 5
        assert step.k == 4
        # Left is already (i, j) = (free, contracted) => no permutation needed
        assert step.left_perm == ()
        # Right is already (j, k) = (contracted, free) => no permutation needed
        assert step.right_perm == ()

    def test_pairwise_with_transpose(self):
        """'ji,jk->ik' requires a left permutation since left is (j, i)."""
        plan = build_blas_plan("ji,jk->ik", [(4, 3), (4, 5)])
        assert len(plan.steps) == 1

        step = plan.steps[0]
        assert step.m == 3
        assert step.n == 5
        assert step.k == 4
        # Left operand has legs (j, i); contracted=j, free=i
        # Need to transpose to (i, j) = (free, contracted),
        # OR use trans_a=True. Either way the plan must produce correct results.
        # We just verify correctness via execute_numpy below.

    def test_three_tensor_chain(self):
        """'ij,jk,kl->il' produces 2 GEMM steps."""
        plan = build_blas_plan("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)])
        assert len(plan.steps) == 2
        assert plan.n_inputs == 3
        # Total buffers = 3 inputs + 1 intermediate + 1 final (or fewer if reused)
        assert plan.n_buffers >= 4

    def test_dmrg_one_site_matvec(self):
        """1-site DMRG: 4 tensors, 3 GEMM steps."""
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        plan = build_blas_plan("abc,apd,bpxe,def->cxf", shapes)
        assert len(plan.steps) == 3
        assert plan.n_inputs == 4

    def test_dmrg_two_site_matvec(self):
        """2-site DMRG: 5 tensors, 4 GEMM steps."""
        shapes = [(2, 3, 4), (2, 5, 6, 7), (3, 5, 8, 9), (9, 6, 10, 11), (7, 11, 4)]
        plan = build_blas_plan("abc,apqd,bpse,eqtf,dfg->cstg", shapes)
        assert len(plan.steps) == 4
        assert plan.n_inputs == 5


class TestExecuteNumpy:
    """Numerical correctness: execute_numpy must match np.einsum."""

    def test_execute_matches_einsum(self):
        """1-site matvec: plan.execute_numpy() matches np.einsum within rtol=1e-10."""
        rng = np.random.default_rng(42)
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        arrays = [rng.standard_normal(s) for s in shapes]

        subscripts = "abc,apd,bpxe,def->cxf"
        expected = np.einsum(subscripts, *arrays)

        plan = build_blas_plan(subscripts, shapes)
        result = plan.execute_numpy(arrays)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_execute_two_site_matches_einsum(self):
        """2-site matvec: plan.execute_numpy() matches np.einsum within rtol=1e-10."""
        rng = np.random.default_rng(123)
        shapes = [(2, 3, 4), (2, 5, 6, 7), (3, 5, 8, 9), (9, 6, 10, 11), (7, 11, 4)]
        arrays = [rng.standard_normal(s) for s in shapes]

        subscripts = "abc,apqd,bpse,eqtf,dfg->cstg"
        expected = np.einsum(subscripts, *arrays)

        plan = build_blas_plan(subscripts, shapes)
        result = plan.execute_numpy(arrays)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_execute_pairwise_matmul(self):
        """Simple matmul correctness."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal((3, 4))
        b = rng.standard_normal((4, 5))

        plan = build_blas_plan("ij,jk->ik", [(3, 4), (4, 5)])
        result = plan.execute_numpy([a, b])
        expected = a @ b

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_execute_with_transpose(self):
        """Matmul with transposed left operand."""
        rng = np.random.default_rng(11)
        a = rng.standard_normal((4, 3))  # (j, i)
        b = rng.standard_normal((4, 5))  # (j, k)

        plan = build_blas_plan("ji,jk->ik", [(4, 3), (4, 5)])
        result = plan.execute_numpy([a, b])
        expected = np.einsum("ji,jk->ik", a, b)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_execute_three_tensor_chain(self):
        """Three-tensor chain correctness."""
        rng = np.random.default_rng(99)
        shapes = [(3, 4), (4, 5), (5, 6)]
        arrays = [rng.standard_normal(s) for s in shapes]

        plan = build_blas_plan("ij,jk,kl->il", shapes)
        result = plan.execute_numpy(arrays)
        expected = np.einsum("ij,jk,kl->il", *arrays)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestCaching:
    """Tests for plan caching."""

    def test_plan_deterministic(self):
        """Same inputs produce equal plans."""
        shapes = [(2, 3, 4), (2, 5, 7), (3, 5, 6, 9), (7, 9, 4)]
        subscripts = "abc,apd,bpxe,def->cxf"
        plan1 = build_blas_plan(subscripts, shapes)
        plan2 = build_blas_plan(subscripts, shapes)
        assert plan1 == plan2

    def test_cached_plan(self):
        """get_cached_blas_plan returns the same object for same key."""
        shapes = ((3, 4), (4, 5))
        subscripts = "ij,jk->ik"
        plan1 = get_cached_blas_plan(subscripts, shapes)
        plan2 = get_cached_blas_plan(subscripts, shapes)
        assert plan1 is plan2


class TestCythonAvailability:
    def test_cython_blas_flag_exists(self):
        """CYTHON_BLAS_AVAILABLE flag should be importable."""
        from tenax.contraction import CYTHON_BLAS_AVAILABLE

        assert isinstance(CYTHON_BLAS_AVAILABLE, bool)

    def test_disable_env_var(self):
        """TENAX_DISABLE_CYTHON_BLAS=1 should force CYTHON_BLAS_AVAILABLE=False."""
        import importlib
        import os

        os.environ["TENAX_DISABLE_CYTHON_BLAS"] = "1"
        try:
            import tenax.contraction as mod

            importlib.reload(mod)
            assert not mod.CYTHON_BLAS_AVAILABLE
        finally:
            del os.environ["TENAX_DISABLE_CYTHON_BLAS"]
            importlib.reload(mod)
