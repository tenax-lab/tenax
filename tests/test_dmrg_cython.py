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

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import (
                cython_lanczos_reorth,  # noqa: F401
            )
        except ImportError:
            pytest.skip("Cython BA extension not available")

    @staticmethod
    def _sequential_reorth(basis_blocks_list, w_blocks):
        """Reference implementation: sequential inner + axpy per basis vector.

        Union of keys, matching ``core._block_array.ba_sub_scaled`` -- which is
        what ``dmrg.py`` runs when the Cython path is disabled, and therefore
        the only thing this reference may encode.

        It previously skipped keys absent from ``w`` (``if wk is not None``),
        reproducing the very defect the Cython side had, so
        ``test_partial_key_overlap`` asserted agreement between two
        implementations that were wrong in the same way and passed while #829
        was live.  A hand-rolled reference is only worth having if it is
        derived from the real one.
        """
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
                else:
                    w_blocks[k] = -(coeff * q_blocks[k])

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

        # Key sets too, not just the shared values: the sectors a basis vector
        # has and ``w`` lacks are precisely the ones #829 dropped.
        assert sorted(w_fused) == sorted(w_seq), (sorted(w_fused), sorted(w_seq))
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

    def test_cython_matvec_combos_wrapper(self):
        """Verify refactored cython_matvec_combos wrapper still works."""
        try:
            from tenax.contraction._cython_blas import cython_matvec_combos
        except ModuleNotFoundError:
            pytest.skip("Cython BA extension not available")

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


# ------------------------------------------------------------------ #
# DMRGMatvec2Site cdef class tests                                     #
# ------------------------------------------------------------------ #


class TestCythonMatvecOp:
    """Verify DMRGMatvec2Site.apply() matches _execute_matvec_combos."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import DMRGMatvec2Site  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("Cython DMRGMatvec2Site not available")

    def test_matches_execute_matvec_combos_realistic_shapes(self):
        """Multiple combos with realistic DMRG-like block shapes."""
        from tenax.contraction._cython_blas import DMRGMatvec2Site

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(123)
        _subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1

        # Multiple combos mimicking a U(1)-symmetric Heisenberg block structure
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 3, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 3, 3)),
            ((2, 3, 4), (2, 2, 2, 5), (3, 2, 2, 3), (3, 2, 2, 6), (5, 6, 4)),
            ((1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
        ]
        block_plan, np_blocks_list = (
            TestPreTransposedMatvecCombos._build_block_plan_and_blocks(
                _subs, shapes, theta_buf_idx, rng
            )
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, _subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        # Python/Cython reference
        result_python = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        # DMRGMatvec2Site path
        op = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        result_cdef = op.py_apply(theta_blocks)

        assert set(result_cdef.keys()) == set(result_python.blocks.keys())
        for key in result_python.blocks:
            np.testing.assert_allclose(
                result_cdef[key],
                result_python.blocks[key],
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Mismatch at key {key}",
            )

    def test_synthetic_two_site(self):
        """Synthetic test with random blocks (no DMRG needed)."""
        from tenax.contraction._cython_blas import DMRGMatvec2Site

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(42)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 5, 6), (3, 2, 2, 7), (5, 2, 3, 4), (4, 2, 3, 8), (7, 8, 6)),
            ((1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
        ]

        block_plan, np_blocks_list = (
            TestPreTransposedMatvecCombos._build_block_plan_and_blocks(
                subs, shapes, theta_buf_idx, rng
            )
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        # Python path
        result_python = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        # DMRGMatvec2Site path
        op = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        result_cdef = op.py_apply(theta_blocks)

        assert set(result_cdef.keys()) == set(result_python.blocks.keys())
        for key in result_python.blocks:
            np.testing.assert_allclose(
                result_cdef[key],
                result_python.blocks[key],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_accumulation_same_output_key(self):
        """Multiple combos accumulating into same output slot."""
        from tenax.contraction._cython_blas import DMRGMatvec2Site

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(777)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
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

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_python = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        op = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        result_cdef = op.py_apply(theta_blocks)

        np.testing.assert_allclose(
            result_cdef[shared_output_key],
            result_python.blocks[shared_output_key],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_different_theta_gives_different_result(self):
        """Re-running with different theta gives different output."""
        from tenax.contraction._cython_blas import DMRGMatvec2Site

        from tenax.algorithms.dmrg import _precompute_matvec_combos

        rng = np.random.default_rng(42)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
        ]

        block_plan, np_blocks_list = (
            TestPreTransposedMatvecCombos._build_block_plan_and_blocks(
                subs, shapes, theta_buf_idx, rng
            )
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        op = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)

        theta_blocks_1 = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks_1[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_1 = op.py_apply(theta_blocks_1)

        theta_blocks_2 = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            shape = np_blocks_list[theta_buf_idx][theta_key].shape
            theta_blocks_2[theta_key] = rng.standard_normal(shape)

        result_2 = op.py_apply(theta_blocks_2)

        for key in result_1:
            assert not np.allclose(result_1[key], result_2[key])


# ------------------------------------------------------------------ #
# DMRGMatvec1Site cdef class tests                                     #
# ------------------------------------------------------------------ #


class TestCythonMatvecOp1Site:
    """Test DMRGMatvec1Site matches _execute_matvec_combos for 1-site."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import DMRGMatvec1Site  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Cython DMRGMatvec1Site not available")

    def test_1site_matches_reference(self):
        """Multiple combos with realistic 1-site DMRG block shapes."""
        from tenax.contraction._cython_blas import DMRGMatvec1Site

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _precompute_matvec_combos,
        )

        rng = np.random.default_rng(123)
        subs = ONE_SITE_SUBSCRIPTS
        theta_buf_idx = 1

        # 1-site shapes: (left_env=abc, site=apd, mpo=bpxe, right_env=def)
        shapes = [
            ((4, 5, 4), (4, 2, 4), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 3, 3), (3, 2, 3), (3, 2, 2, 3), (3, 3, 3)),
            ((2, 3, 4), (2, 2, 5), (3, 2, 2, 6), (5, 6, 4)),
        ]

        block_plan, np_blocks_list = (
            TestPreTransposedMatvecCombos._build_block_plan_and_blocks(
                subs, shapes, theta_buf_idx, rng
            )
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        # Python reference
        result_python = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        # DMRGMatvec1Site path
        op = DMRGMatvec1Site(combos, out_keys, out_shapes, theta_buf_idx)
        result_cdef = op.py_apply(theta_blocks)

        assert set(result_cdef.keys()) == set(result_python.blocks.keys())
        for key in result_python.blocks:
            np.testing.assert_allclose(
                result_cdef[key],
                result_python.blocks[key],
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Mismatch at key {key}",
            )

    def test_1site_single_combo(self):
        """Single combo, 1-site DMRG pattern."""
        from tenax.contraction._cython_blas import DMRGMatvec1Site

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

        block_plan, np_blocks_list = (
            TestPreTransposedMatvecCombos._build_block_plan_and_blocks(
                subs, shapes, theta_buf_idx, rng
            )
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        result_python = _execute_matvec_combos(
            combos, theta_blocks, theta_buf_idx, out_keys, out_shapes, ()
        )

        op = DMRGMatvec1Site(combos, out_keys, out_shapes, theta_buf_idx)
        result_cdef = op.py_apply(theta_blocks)

        assert set(result_cdef.keys()) == set(result_python.blocks.keys())
        for key in result_python.blocks:
            np.testing.assert_allclose(
                result_cdef[key],
                result_python.blocks[key],
                rtol=1e-12,
                atol=1e-14,
            )


# ------------------------------------------------------------------ #
# TestCythonLanczosGround                                              #
# ------------------------------------------------------------------ #


class TestCythonLanczosGround:
    """Test cython_lanczos_ground matches _lanczos_solve_np."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import (
                DMRGMatvec2Site,  # noqa: F401
                cython_lanczos_ground,  # noqa: F401
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Cython Lanczos not available")

    @staticmethod
    def _build_lanczos_block_plan(subscripts, shapes_list, theta_buf_idx, rng):
        """Build block plan where theta keys == output keys (Lanczos-safe).

        In a real DMRG block plan, each combo maps theta_key -> same output_key.
        Multiple combos may share the same theta (and output) key.
        This builder ensures that property holds, which is necessary for
        Lanczos iteration (the output of matvec has the same block structure
        as the input).
        """
        n_tensors = len(shapes_list[0])
        np_blocks_list = [{} for _ in range(n_tensors)]
        block_plan = []

        for combo_idx, shapes in enumerate(shapes_list):
            combo_keys = []
            for t_idx, shape in enumerate(shapes):
                if t_idx == theta_buf_idx:
                    # Use combo_idx as the charge key for theta AND output
                    key = (combo_idx,)
                else:
                    key = (combo_idx, t_idx)
                np_blocks_list[t_idx][key] = rng.standard_normal(shape)
                combo_keys.append(key)
            output_key = (combo_idx,)  # Same as theta key
            block_plan.append((combo_keys, output_key))

        return block_plan, np_blocks_list

    def test_eigenvalue_matches_python_lanczos(self):
        """Fused Cython Lanczos eigenvalue must match Python _lanczos_solve_np."""
        from tenax.contraction._cython_blas import (
            DMRGMatvec2Site,
            cython_lanczos_ground,
        )

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _lanczos_solve_np,
            _precompute_matvec_combos,
        )
        from tenax.core._block_array import BlockArray

        rng = np.random.default_rng(42)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1

        # Build synthetic multi-combo block plan (theta keys == output keys)
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 3, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 3, 3)),
        ]
        block_plan, np_blocks_list = self._build_lanczos_block_plan(
            subs, shapes, theta_buf_idx, rng
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        # Extract theta blocks
        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        # Build BlockArray for the Python path
        theta_ba = BlockArray(theta_blocks, ())

        # Python reference path
        def matvec_py(v_ba):
            return _execute_matvec_combos(
                combos,
                v_ba.blocks,
                theta_buf_idx,
                out_keys,
                out_shapes,
                (),
            )

        energy_py, vec_py = _lanczos_solve_np(matvec_py, theta_ba, 30, 1e-14)

        # Cython fused path
        mv = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        energy_cy, vec_cy_blocks = cython_lanczos_ground(mv, theta_blocks, 30, 1e-14)

        # Eigenvalue must match
        np.testing.assert_allclose(energy_cy, energy_py, atol=1e-10)

        # Eigenvector must preserve the full block support.
        assert set(vec_cy_blocks.keys()) == set(vec_py.blocks.keys())

        # Eigenvector overlap (up to global phase)
        overlap = 0.0
        for k in vec_py.blocks:
            if k in vec_cy_blocks:
                overlap += float(np.vdot(vec_py.blocks[k], vec_cy_blocks[k]).real)
        assert abs(abs(overlap) - 1.0) < 1e-8, f"Overlap: {overlap}"

    def test_single_combo(self):
        """Single-combo Lanczos should produce finite eigenvalue."""
        from tenax.contraction._cython_blas import (
            DMRGMatvec2Site,
            cython_lanczos_ground,
        )

        from tenax.algorithms.dmrg import _precompute_matvec_combos

        rng = np.random.default_rng(123)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1

        shapes = [
            ((3, 3, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 3, 3)),
        ]
        block_plan, np_blocks_list = self._build_lanczos_block_plan(
            subs, shapes, theta_buf_idx, rng
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        mv = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        energy, vec_blocks = cython_lanczos_ground(mv, theta_blocks, 20, 1e-12)

        assert np.isfinite(energy)
        # Eigenvector should be normalized
        norm_sq = sum(float(np.vdot(v, v).real) for v in vec_blocks.values())
        np.testing.assert_allclose(norm_sq, 1.0, atol=1e-10)

    def test_multiple_combos_consistent_shapes(self):
        """Multiple combos with consistent shapes, verifies Cython vs Python."""
        from tenax.contraction._cython_blas import (
            DMRGMatvec2Site,
            cython_lanczos_ground,
        )

        from tenax.algorithms.dmrg import (
            _execute_matvec_combos,
            _lanczos_solve_np,
            _precompute_matvec_combos,
        )
        from tenax.core._block_array import BlockArray

        rng = np.random.default_rng(999)
        subs = TWO_SITE_SUBSCRIPTS
        theta_buf_idx = 1

        # For Lanczos to iterate, output shape must match theta shape.
        # With subs "abc,apqd,bpse,eqtf,dfg->cstg", output is (c,s,t,g)
        # and theta is (a,p,q,d). For them to match we need c=a, s=p, t=q, g=d
        # which means L=(c,b,c), theta=(c,s,t,g), W1=(b,s,s,b), W2=(b,t,t,b), R=(g,b,g)
        # i.e. L is (chi,w,chi), theta is (chi,d,d,chi), W is (w,d,d,w), R=(chi,w,chi)
        shapes = [
            ((4, 5, 4), (4, 2, 2, 4), (5, 2, 2, 5), (5, 2, 2, 5), (4, 5, 4)),
            ((3, 3, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 3), (3, 3, 3)),
            ((1, 1, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1), (1, 1, 1)),
        ]
        block_plan, np_blocks_list = self._build_lanczos_block_plan(
            subs, shapes, theta_buf_idx, rng
        )

        combos, out_keys, out_shapes = _precompute_matvec_combos(
            block_plan, subs, np_blocks_list, theta_buf_idx
        )

        theta_blocks = {}
        for combo_keys, _ in block_plan:
            theta_key = combo_keys[theta_buf_idx]
            theta_blocks[theta_key] = np_blocks_list[theta_buf_idx][theta_key]

        theta_ba = BlockArray(theta_blocks, ())

        def matvec_py(v_ba):
            return _execute_matvec_combos(
                combos,
                v_ba.blocks,
                theta_buf_idx,
                out_keys,
                out_shapes,
                (),
            )

        energy_py, _ = _lanczos_solve_np(matvec_py, theta_ba, 30, 1e-14)

        mv = DMRGMatvec2Site(combos, out_keys, out_shapes, theta_buf_idx)
        energy_cy, _ = cython_lanczos_ground(mv, theta_blocks, 30, 1e-14)

        np.testing.assert_allclose(energy_cy, energy_py, atol=1e-10)
