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


# ------------------------------------------------------------------ #
# Fused Lanczos reorthogonalization tests                              #
# ------------------------------------------------------------------ #


class TestCythonLanczosReorth:
    """Verify fused reorthogonalization matches sequential approach."""

    @staticmethod
    def _sequential_reorth(basis_blocks_list, w_blocks):
        """Reference implementation: sequential inner + axpy per basis vector."""
        for q_blocks in basis_blocks_list:
            coeff = 0.0
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is not None:
                    coeff += np.vdot(q_blocks[k], wk).real
            if coeff == 0.0:
                continue
            for k in q_blocks:
                wk = w_blocks.get(k)
                if wk is not None:
                    w_blocks[k] = wk - coeff * q_blocks[k]

    def _make_random_blocks(self, rng, keys, shapes, dtype=np.float64):
        """Create a dict of random blocks with given keys and shapes."""
        return {k: rng.standard_normal(s).astype(dtype) for k, s in zip(keys, shapes)}

    def test_basic_float64(self):
        """Fused reorth matches sequential for float64 blocks."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(42)
        keys = [(0,), (1,), (2,)]
        shapes = [(4, 3), (5, 2), (3, 4)]

        # Build 5 basis vectors and a w vector
        basis = [self._make_random_blocks(rng, keys, shapes) for _ in range(5)]
        w_fused = self._make_random_blocks(rng, keys, shapes)
        w_seq = {k: v.copy() for k, v in w_fused.items()}

        # Run both
        cython_lanczos_reorth(basis, w_fused)
        self._sequential_reorth(basis, w_seq)

        for k in keys:
            np.testing.assert_allclose(w_fused[k], w_seq[k], rtol=1e-12, atol=1e-14)

    def test_partial_key_overlap(self):
        """Handles basis vectors with keys not present in w."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(123)
        # w has keys (0,) and (1,); basis[0] has (0,) and (2,)
        w_fused = {
            (0,): rng.standard_normal((3, 4)),
            (1,): rng.standard_normal((2, 5)),
        }
        basis = [
            {(0,): rng.standard_normal((3, 4)), (2,): rng.standard_normal((4, 3))},
            {(1,): rng.standard_normal((2, 5)), (3,): rng.standard_normal((6, 2))},
        ]
        w_seq = {k: v.copy() for k, v in w_fused.items()}

        cython_lanczos_reorth(basis, w_fused)
        self._sequential_reorth(basis, w_seq)

        for k in w_fused:
            np.testing.assert_allclose(w_fused[k], w_seq[k], rtol=1e-12, atol=1e-14)

    def test_empty_basis(self):
        """Empty basis list is a no-op."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(7)
        w = {(0,): rng.standard_normal((3, 4))}
        w_orig = {k: v.copy() for k, v in w.items()}
        cython_lanczos_reorth([], w)
        np.testing.assert_array_equal(w[(0,)], w_orig[(0,)])

    def test_single_basis_vector(self):
        """Single basis vector produces correct subtraction."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(99)
        keys = [(0,), (1,)]
        shapes = [(10,), (8,)]
        q = self._make_random_blocks(rng, keys, shapes)
        w_fused = self._make_random_blocks(rng, keys, shapes)
        w_seq = {k: v.copy() for k, v in w_fused.items()}

        cython_lanczos_reorth([q], w_fused)
        self._sequential_reorth([q], w_seq)

        for k in keys:
            np.testing.assert_allclose(w_fused[k], w_seq[k], rtol=1e-12, atol=1e-14)

    def test_orthogonality_achieved(self):
        """After reorth, w should be orthogonal to all basis vectors.

        Uses an orthonormal basis (Gram-Schmidt) so one reorth pass suffices.
        """
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(314)
        keys = [(0,), (1,)]
        shapes = [(20,), (15,)]

        # Build orthonormal basis via Gram-Schmidt
        basis = []
        for _ in range(3):
            q = self._make_random_blocks(rng, keys, shapes)
            # Subtract projections onto previous basis vectors
            for prev in basis:
                c = sum(np.vdot(prev[k], q[k]).real for k in keys)
                q = {k: q[k] - c * prev[k] for k in keys}
            norm = sum(np.vdot(q[k], q[k]).real for k in keys) ** 0.5
            q = {k: v / norm for k, v in q.items()}
            basis.append(q)

        w = self._make_random_blocks(rng, keys, shapes)
        cython_lanczos_reorth(basis, w)

        # Check orthogonality
        for q in basis:
            overlap = sum(np.vdot(q[k], w[k]).real for k in keys)
            assert abs(overlap) < 1e-10, f"overlap={overlap}"

    def test_many_basis_vectors(self):
        """Stress test with many basis vectors."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(555)
        keys = [(i,) for i in range(4)]
        shapes = [(8, 6), (5, 4), (3, 7), (6, 3)]

        basis = [self._make_random_blocks(rng, keys, shapes) for _ in range(20)]
        w_fused = self._make_random_blocks(rng, keys, shapes)
        w_seq = {k: v.copy() for k, v in w_fused.items()}

        cython_lanczos_reorth(basis, w_fused)
        self._sequential_reorth(basis, w_seq)

        for k in keys:
            np.testing.assert_allclose(w_fused[k], w_seq[k], rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ #
# Pre-transposed matvec combo tests                                    #
# ------------------------------------------------------------------ #


class TestPreTransposedMatvecCombos:
    """Verify pre-transposed env blocks produce identical matvec results."""

    @staticmethod
    def _build_block_plan_and_blocks(subscripts, shapes_list, theta_buf_idx, rng):
        """Build fake block plan and numpy block dicts for testing.

        ``shapes_list`` is a list of tuples, one per combo.
        Each tuple contains shapes for all input tensors.
        Returns (block_plan, np_blocks_list).
        """
        n_tensors = len(shapes_list[0])
        np_blocks_list = [{} for _ in range(n_tensors)]
        block_plan = []

        for combo_idx, shapes in enumerate(shapes_list):
            combo_keys = []
            for t_idx, shape in enumerate(shapes):
                key = (combo_idx, t_idx)
                np_blocks_list[t_idx][key] = rng.standard_normal(shape)
                combo_keys.append(key)
            output_key = (combo_idx,)
            block_plan.append((combo_keys, output_key))

        return block_plan, np_blocks_list

    @staticmethod
    def _einsum_reference(subscripts, block_plan, np_blocks_list):
        """Compute reference results using np.einsum for each combo."""
        results = {}
        for combo_keys, output_key in block_plan:
            arrays = [
                np_blocks_list[i][combo_keys[i]] for i in range(len(np_blocks_list))
            ]
            result = np.einsum(subscripts, *arrays)
            if output_key in results:
                results[output_key] = results[output_key] + result
            else:
                results[output_key] = result
        return results

    def test_two_site_single_combo(self):
        """Single combo, two-site DMRG pattern."""
        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(42)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
        ]
        block_plan, np_blocks_list = self._build_block_plan_and_blocks(
            subs, shapes, theta_buf_idx, rng
        )
        expected = self._einsum_reference(subs, block_plan, np_blocks_list)

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        # Extract theta blocks
        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_ba = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        for key, exp_arr in expected.items():
            np.testing.assert_allclose(
                result_ba.blocks[key], exp_arr, rtol=1e-12, atol=1e-14
            )

    def test_two_site_multiple_combos(self):
        """Multiple combos with different shapes, two-site pattern."""
        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(123)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 5, 6), (3, 2, 2, 7), (5, 2, 3, 4), (4, 2, 3, 8), (7, 8, 6)),
            ((1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
        ]
        block_plan, np_blocks_list = self._build_block_plan_and_blocks(
            subs, shapes, theta_buf_idx, rng
        )
        expected = self._einsum_reference(subs, block_plan, np_blocks_list)

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_ba = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        for key, exp_arr in expected.items():
            np.testing.assert_allclose(
                result_ba.blocks[key], exp_arr, rtol=1e-12, atol=1e-14
            )

    def test_one_site_single_combo(self):
        """Single combo, one-site DMRG pattern."""
        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(42)
        subs = ONE_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 4), (5, 2, 2, 5), (4, 5, 4)),
        ]
        block_plan, np_blocks_list = self._build_block_plan_and_blocks(
            subs, shapes, theta_buf_idx, rng
        )
        expected = self._einsum_reference(subs, block_plan, np_blocks_list)

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_ba = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        for key, exp_arr in expected.items():
            np.testing.assert_allclose(
                result_ba.blocks[key], exp_arr, rtol=1e-12, atol=1e-14
            )

    def test_accumulation_same_output_key(self):
        """Multiple combos accumulating into the same output slot."""
        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(777)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        # Two combos with different input blocks but same output key
        shapes = ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4))

        n_tensors = len(shapes)
        np_blocks_list = [{} for _ in range(n_tensors)]
        block_plan = []
        shared_output_key = (99,)

        for combo_idx in range(3):
            combo_keys = []
            for t_idx, shape in enumerate(shapes):
                key = (combo_idx, t_idx)
                np_blocks_list[t_idx][key] = rng.standard_normal(shape)
                combo_keys.append(key)
            block_plan.append((combo_keys, shared_output_key))

        expected = self._einsum_reference(subs, block_plan, np_blocks_list)

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_ba = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        np.testing.assert_allclose(
            result_ba.blocks[shared_output_key],
            expected[shared_output_key],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_changed_theta_gives_different_result(self):
        """Re-running with different theta blocks gives different output."""
        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(42)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
        ]
        block_plan, np_blocks_list = self._build_block_plan_and_blocks(
            subs, shapes, theta_buf_idx, rng
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        # First run
        theta_blocks_1 = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks_1[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_1 = _execute_matvec_combos(
            combos, theta_blocks_1, theta_buf_idx, out_keys, out_shapes, ()
        )

        # Second run with new random theta blocks (same keys, different values)
        theta_blocks_2 = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            shape = np_blocks_list[theta_buf_idx][theta_key].shape
            theta_blocks_2[theta_key] = rng.standard_normal(shape)

        result_2 = _execute_matvec_combos(
            combos, theta_blocks_2, theta_buf_idx, out_keys, out_shapes, ()
        )

        # Results should differ
        for key in result_1.blocks:
            assert not np.allclose(result_1.blocks[key], result_2.blocks[key])
