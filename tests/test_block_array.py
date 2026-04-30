"""Tests for BlockArray: lightweight numpy-backed block-sparse array."""

import numpy as np
import pytest

from tenax import U1Symmetry
from tenax.core._block_array import (
    BlockArray,
    ba_add,
    ba_axpy,
    ba_conj,
    ba_inner,
    ba_norm,
    ba_scale,
    ba_scale_inplace,
    ba_sub,
    ba_to_symmetric,
    symmetric_to_ba,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor


@pytest.fixture
def simple_ba():
    """A BlockArray with two blocks for basic arithmetic tests."""
    sym = U1Symmetry()
    # charges: [0, 0, 1, 1, 1] -> 2 states charge 0, 3 states charge 1
    idx_a = TensorIndex.from_charges(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    # charges: [0, 0, 0, 0, 1, 1] -> 4 states charge 0, 2 states charge 1
    idx_b = TensorIndex.from_charges(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        (1, 1): np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


@pytest.fixture
def another_ba():
    """A BlockArray with same structure but different values."""
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.ones((2, 4)),
        (1, 1): np.ones((3, 2)) * 2.0,
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


class TestBaScale:
    def test_scale_multiplies_all_blocks(self, simple_ba):
        result = ba_scale(simple_ba, 3.0)
        for key in simple_ba.blocks:
            np.testing.assert_allclose(result.blocks[key], simple_ba.blocks[key] * 3.0)

    def test_scale_zero(self, simple_ba):
        result = ba_scale(simple_ba, 0.0)
        for key in result.blocks:
            np.testing.assert_allclose(result.blocks[key], 0.0)

    def test_scale_preserves_indices(self, simple_ba):
        result = ba_scale(simple_ba, 2.0)
        assert result.indices == simple_ba.indices


class TestBaAdd:
    def test_add_same_keys(self, simple_ba, another_ba):
        result = ba_add(simple_ba, another_ba)
        for key in simple_ba.blocks:
            expected = simple_ba.blocks[key] + another_ba.blocks[key]
            np.testing.assert_allclose(result.blocks[key], expected)

    def test_add_different_keys_union(self):
        """Adding with disjoint keys should produce union."""
        sym = U1Symmetry()
        idx = TensorIndex.from_charges(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="x"
        )
        idx_out = TensorIndex.from_charges(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="y"
        )
        ba1 = BlockArray(blocks={(0, 0): np.array([[1.0]])}, indices=(idx, idx_out))
        ba2 = BlockArray(blocks={(1, 1): np.array([[2.0]])}, indices=(idx, idx_out))
        result = ba_add(ba1, ba2)
        assert (0, 0) in result.blocks
        assert (1, 1) in result.blocks
        np.testing.assert_allclose(result.blocks[(0, 0)], [[1.0]])
        np.testing.assert_allclose(result.blocks[(1, 1)], [[2.0]])

    def test_add_preserves_indices(self, simple_ba, another_ba):
        result = ba_add(simple_ba, another_ba)
        assert result.indices == simple_ba.indices


class TestBaSub:
    def test_sub_same_keys(self, simple_ba, another_ba):
        result = ba_sub(simple_ba, another_ba)
        for key in simple_ba.blocks:
            expected = simple_ba.blocks[key] - another_ba.blocks[key]
            np.testing.assert_allclose(result.blocks[key], expected)


class TestBaInner:
    def test_inner_product(self, simple_ba):
        # Frobenius inner product = sum of element-wise products
        expected = sum(np.sum(b * b) for b in simple_ba.blocks.values())
        result = ba_inner(simple_ba, simple_ba)
        np.testing.assert_allclose(result, expected)

    def test_inner_different_arrays(self, simple_ba, another_ba):
        expected = sum(
            np.sum(simple_ba.blocks[k] * another_ba.blocks[k]) for k in simple_ba.blocks
        )
        result = ba_inner(simple_ba, another_ba)
        np.testing.assert_allclose(result, expected)


class TestBaNorm:
    def test_norm_matches_sqrt_inner(self, simple_ba):
        expected = np.sqrt(ba_inner(simple_ba, simple_ba))
        result = ba_norm(simple_ba)
        np.testing.assert_allclose(result, expected)


class TestBaConj:
    def test_conj_real_is_identity(self, simple_ba):
        result = ba_conj(simple_ba)
        for key in simple_ba.blocks:
            np.testing.assert_allclose(result.blocks[key], simple_ba.blocks[key])

    def test_conj_preserves_indices(self, simple_ba):
        result = ba_conj(simple_ba)
        assert result.indices == simple_ba.indices


class TestBaAxpy:
    def test_axpy_in_place(self, simple_ba, another_ba):
        """y += alpha * x should modify y in place."""
        alpha = -2.0
        # Save originals for expected values
        expected = {
            k: simple_ba.blocks[k] + alpha * another_ba.blocks[k]
            for k in simple_ba.blocks
        }
        ba_axpy(another_ba, simple_ba, alpha)
        for key in expected:
            np.testing.assert_allclose(simple_ba.blocks[key], expected[key])

    def test_axpy_skips_missing_keys(self):
        """Keys in x but not y should be ignored."""
        sym = U1Symmetry()
        idx = TensorIndex.from_charges(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="x"
        )
        idx_out = TensorIndex.from_charges(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="y"
        )
        x = BlockArray(
            blocks={(0, 0): np.array([[1.0]]), (1, 1): np.array([[2.0]])},
            indices=(idx, idx_out),
        )
        y = BlockArray(
            blocks={(0, 0): np.array([[10.0]])},
            indices=(idx, idx_out),
        )
        ba_axpy(x, y, 3.0)
        np.testing.assert_allclose(y.blocks[(0, 0)], [[13.0]])
        assert (1, 1) not in y.blocks


class TestBaScaleInplace:
    def test_scale_inplace(self, simple_ba):
        """In-place scaling should modify blocks without creating new dict."""
        original_blocks = simple_ba.blocks
        expected = {k: v * 3.0 for k, v in simple_ba.blocks.items()}
        ba_scale_inplace(simple_ba, 3.0)
        assert simple_ba.blocks is original_blocks  # same dict object
        for key in expected:
            np.testing.assert_allclose(simple_ba.blocks[key], expected[key])


class TestCythonBaInner:
    """Test that cython_ba_inner gives same result as pure-Python ba_inner."""

    def test_cython_inner_matches_python(self, simple_ba, another_ba):
        """Cython inner product must match Python fallback."""
        try:
            from tenax.contraction._cython_blas import cython_ba_inner
        except ImportError:
            pytest.skip("Cython BA extension not available")
        # Compute via Python fallback
        python_result = sum(
            float(np.sum(simple_ba.blocks[k] * another_ba.blocks[k]))
            for k in simple_ba.blocks
        )
        cython_result = cython_ba_inner(simple_ba.blocks, another_ba.blocks)
        np.testing.assert_allclose(cython_result, python_result, rtol=1e-14)

    def test_cython_inner_self(self, simple_ba):
        """Inner product with self should equal sum of squared elements."""
        try:
            from tenax.contraction._cython_blas import cython_ba_inner
        except ImportError:
            pytest.skip("Cython BA extension not available")
        expected = sum(np.sum(b**2) for b in simple_ba.blocks.values())
        result = cython_ba_inner(simple_ba.blocks, simple_ba.blocks)
        np.testing.assert_allclose(result, expected, rtol=1e-14)


@pytest.fixture
def complex_ba():
    """A BlockArray with complex128 blocks for complex arithmetic tests."""
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.array(
            [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], [9 + 10j, 11 + 12j, 13 + 14j, 15 + 16j]]
        ),
        (1, 1): np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]]),
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


@pytest.fixture
def another_complex_ba():
    """A second BlockArray with complex128 blocks and same structure."""
    sym = U1Symmetry()
    idx_a = TensorIndex.from_charges(
        sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    idx_b = TensorIndex.from_charges(
        sym, np.array([0, 0, 0, 0, 1, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    blocks = {
        (0, 0): np.array(
            [[2 - 1j, 4 - 3j, 6 - 5j, 8 - 7j], [10 - 9j, 12 - 11j, 14 - 13j, 16 - 15j]]
        ),
        (1, 1): np.array([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j], [5 - 5j, 6 - 6j]]),
    }
    return BlockArray(blocks=blocks, indices=(idx_a, idx_b))


class TestComplexBlockArray:
    """Tests for BlockArray operations with complex128 data."""

    def test_inner_hermitian(self, complex_ba, another_complex_ba):
        """ba_inner must compute full Hermitian inner product: sum conj(a)*b."""
        result = ba_inner(complex_ba, another_complex_ba)
        # Reference: np.vdot flattens and computes sum(conj(a)*b)
        expected = sum(
            np.vdot(complex_ba.blocks[k], another_complex_ba.blocks[k])
            for k in complex_ba.blocks
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_inner_self_is_real(self, complex_ba):
        """<v|v> must be real and positive for complex vectors."""
        result = ba_inner(complex_ba, complex_ba)
        # result may be complex type but imaginary part must be ~0
        assert abs(complex(result).imag) < 1e-12
        assert float(complex(result).real) > 0

    def test_norm_complex(self, complex_ba):
        """Norm of complex BlockArray = sqrt(<v|v>)."""
        expected = np.sqrt(ba_inner(complex_ba, complex_ba))
        result = ba_norm(complex_ba)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_axpy_complex(self, complex_ba, another_complex_ba):
        """ba_axpy works with complex arrays and real scalar."""
        alpha = 2.0
        expected = {
            k: another_complex_ba.blocks[k] + alpha * complex_ba.blocks[k]
            for k in another_complex_ba.blocks
        }
        ba_axpy(complex_ba, another_complex_ba, alpha)
        for key in expected:
            np.testing.assert_allclose(another_complex_ba.blocks[key], expected[key])

    def test_scale_inplace_complex(self, complex_ba):
        """ba_scale_inplace works with complex arrays."""
        expected = {k: v * 3.0 for k, v in complex_ba.blocks.items()}
        ba_scale_inplace(complex_ba, 3.0)
        for key in expected:
            np.testing.assert_allclose(complex_ba.blocks[key], expected[key])

    def test_roundtrip_complex(self):
        """ba_to_symmetric preserves complex dtype (t._data.dtype == np.complex128)."""
        sym = U1Symmetry()
        idx_a = TensorIndex.from_charges(
            sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
        )
        idx_b = TensorIndex.from_charges(
            sym,
            np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
            FlowDirection.OUT,
            label="b",
        )
        blocks = {
            (0, 0): np.array(
                [
                    [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                    [9 + 10j, 11 + 12j, 13 + 14j, 15 + 16j],
                ]
            ),
            (1, 1): np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]]),
        }
        ba = BlockArray(blocks=blocks, indices=(idx_a, idx_b))
        t = ba_to_symmetric(ba)
        assert t._data.dtype == np.complex128
        for key in blocks:
            np.testing.assert_allclose(
                np.asarray(t.blocks[key]), blocks[key], rtol=1e-12
            )


class TestRoundtrip:
    def test_symmetric_to_ba_and_back(self):
        """SymmetricTensor -> BlockArray -> SymmetricTensor preserves data."""
        sym = U1Symmetry()
        # charges for idx_a: [0, 0, 1, 1, 1]  (2 with q=0, 3 with q=1)
        idx_a = TensorIndex.from_charges(
            sym, np.array([0, 0, 1, 1, 1], dtype=np.int32), FlowDirection.IN, label="a"
        )
        # charges for idx_b: [0, 0, 0, 0, 1, 1]  (4 with q=0, 2 with q=1)
        idx_b = TensorIndex.from_charges(
            sym,
            np.array([0, 0, 0, 0, 1, 1], dtype=np.int32),
            FlowDirection.OUT,
            label="b",
        )
        rng = np.random.default_rng(42)
        blocks = {
            (0, 0): rng.standard_normal((2, 4)),
            (1, 1): rng.standard_normal((3, 2)),
        }
        import jax.numpy as jnp

        jax_blocks = {k: jnp.array(v) for k, v in blocks.items()}
        t = SymmetricTensor(jax_blocks, (idx_a, idx_b))

        ba = symmetric_to_ba(t)
        assert isinstance(ba, BlockArray)
        assert ba.indices == t.indices

        t2 = ba_to_symmetric(ba)
        assert isinstance(t2, SymmetricTensor)
        for key in t.blocks:
            np.testing.assert_allclose(
                np.asarray(t2.blocks[key]),
                np.asarray(t.blocks[key]),
                rtol=1e-12,
            )


class TestCythonComplexBa:
    """Test Cython BA functions with complex dtypes."""

    @pytest.fixture
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import cython_ba_inner
        except ImportError:
            pytest.skip("Cython BA extension not available")

    @pytest.fixture
    def complex_blocks(self):
        rng = np.random.default_rng(42)
        return {
            (0,): rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)),
            (1,): rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5)),
        }

    @pytest.fixture
    def complex_blocks_b(self):
        rng = np.random.default_rng(99)
        return {
            (0,): rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)),
            (1,): rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5)),
        }

    def test_cython_inner_complex128(
        self, require_cython, complex_blocks, complex_blocks_b
    ):
        from tenax.contraction._cython_blas import cython_ba_inner

        expected = sum(
            np.vdot(complex_blocks[k], complex_blocks_b[k]) for k in complex_blocks
        )
        result = cython_ba_inner(complex_blocks, complex_blocks_b)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_cython_inner_self_real(self, require_cython, complex_blocks):
        from tenax.contraction._cython_blas import cython_ba_inner

        result = cython_ba_inner(complex_blocks, complex_blocks)
        # <v|v> is always real; imaginary part must be ~0
        assert abs(complex(result).imag) < 1e-12
        assert float(complex(result).real) > 0
        expected = sum(np.vdot(v, v).real for v in complex_blocks.values())
        np.testing.assert_allclose(complex(result).real, expected, rtol=1e-12)

    def test_cython_axpy_complex128(
        self, require_cython, complex_blocks, complex_blocks_b
    ):
        from tenax.contraction._cython_blas import cython_ba_axpy

        alpha = 2.5
        expected = {
            k: complex_blocks_b[k] + alpha * complex_blocks[k] for k in complex_blocks
        }
        y = {k: v.copy() for k, v in complex_blocks_b.items()}
        cython_ba_axpy(complex_blocks, y, alpha)
        for k in expected:
            np.testing.assert_allclose(y[k], expected[k], rtol=1e-12)

    def test_cython_scale_complex128(self, require_cython, complex_blocks):
        from tenax.contraction._cython_blas import cython_ba_scale_inplace

        blocks = {k: v.copy() for k, v in complex_blocks.items()}
        expected = {k: v * 3.0 for k, v in complex_blocks.items()}
        cython_ba_scale_inplace(blocks, 3.0)
        for k in expected:
            np.testing.assert_allclose(blocks[k], expected[k])

    def test_cython_sub_scaled_complex128(
        self, require_cython, complex_blocks, complex_blocks_b
    ):
        from tenax.contraction._cython_blas import cython_ba_sub_scaled_inplace

        scalar = 1.5
        w = {k: v.copy() for k, v in complex_blocks.items()}
        expected = {
            k: complex_blocks[k] - scalar * complex_blocks_b[k] for k in complex_blocks
        }
        cython_ba_sub_scaled_inplace(w, complex_blocks_b, scalar)
        for k in expected:
            np.testing.assert_allclose(w[k], expected[k], rtol=1e-12)


class TestComplexLanczosReorth:
    """Regression tests for complex reorthogonalization (issue #216 bug 2)."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import cython_ba_inner  # noqa: F401
        except ImportError:
            pytest.skip("Cython BA extension not available")

    def test_complex_inner_preserves_imaginary(self):
        """ba_inner must return complex for complex inputs, not just real part."""
        from tenax.contraction._cython_blas import cython_ba_inner

        rng = np.random.default_rng(0)
        a = {(0,): rng.standard_normal(8) + 1j * rng.standard_normal(8)}
        b = {(0,): rng.standard_normal(8) + 1j * rng.standard_normal(8)}
        result = cython_ba_inner(a, b)
        expected = np.vdot(a[(0,)], b[(0,)])
        # Must match full complex value, not just real part
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_complex_reorth_removes_full_overlap(self):
        """After reorth, overlap <q|w> must be ~0 including imaginary part."""
        from tenax.contraction._cython_blas import cython_lanczos_reorth

        rng = np.random.default_rng(0)
        q0 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        q0 = q0 / np.linalg.norm(q0)
        q = {(0,): q0}
        w = {(0,): rng.standard_normal(8) + 1j * rng.standard_normal(8)}

        wc = {k: v.copy() for k, v in w.items()}
        cython_lanczos_reorth([q], wc)
        overlap = np.vdot(q0, wc[(0,)])
        assert abs(overlap) < 1e-14, f"Overlap after reorth: {overlap}"


class TestNonContiguousBLAS:
    """Regression tests for in-place BLAS on non-C-contiguous arrays (issue #216 bug 3)."""

    @pytest.fixture(autouse=True)
    def require_cython(self):
        try:
            from tenax.contraction._cython_blas import (
                cython_ba_scale_inplace,  # noqa: F401
            )
        except ImportError:
            pytest.skip("Cython BA extension not available")

    def test_scale_inplace_fortran_order(self):
        from tenax.contraction._cython_blas import cython_ba_scale_inplace

        A = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        blocks = {(0,): A.copy(order="F")}
        cython_ba_scale_inplace(blocks, 2.0)
        np.testing.assert_allclose(blocks[(0,)], A * 2)

    def test_axpy_fortran_order(self):
        from tenax.contraction._cython_blas import cython_ba_axpy

        A = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        y = {(0,): A.copy(order="F")}
        x = {(0,): np.ones((2, 3))}
        cython_ba_axpy(x, y, 3.0)
        np.testing.assert_allclose(y[(0,)], A + 3)

    def test_sub_scaled_fortran_order(self):
        from tenax.contraction._cython_blas import cython_ba_sub_scaled_inplace

        A = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        w = {(0,): A.copy(order="F")}
        q = {(0,): np.ones((2, 3))}
        cython_ba_sub_scaled_inplace(w, q, 1.0)
        np.testing.assert_allclose(w[(0,)], A - 1)
