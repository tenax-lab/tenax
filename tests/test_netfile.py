"""Tests for cytnx-style .net file parser and NetworkBlueprint."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor
from tenax.network.netfile import (
    NetworkBlueprint,
    _labels_to_subscripts_from_names,
    _parse_order,
    from_netfile,
    parse_netfile,
)

# ===================================================================
# Helper: make a simple DenseTensor with trivial symmetry
# ===================================================================


def _make_dense(
    shape: tuple[int, ...], labels: tuple[str, ...], seed: int = 0
) -> DenseTensor:
    """Create a DenseTensor with the given shape, labels, and deterministic data."""
    u1 = U1Symmetry()
    rng = np.random.RandomState(seed)
    data = jnp.array(rng.randn(*shape), dtype=jnp.float32)
    indices = tuple(
        TensorIndex.from_charges(
            u1, np.zeros(dim, dtype=np.int32), FlowDirection.IN, label=lbl
        )
        for dim, lbl in zip(shape, labels)
    )
    return DenseTensor(data, indices)


# ===================================================================
# Parser tests
# ===================================================================


class TestParseNetfile:
    def test_two_tensors(self):
        spec = parse_netfile("A: i, j\nB: j, k")
        assert spec["tensors"] == {"A": ["i", "j"], "B": ["j", "k"]}
        assert spec["tout"] is None
        assert spec["order"] is None

    def test_three_tensors_with_tout(self):
        spec = parse_netfile("A: i, j\nB: j, k\nC: k, l\nTOUT: i, l")
        assert spec["tensors"]["C"] == ["k", "l"]
        assert spec["tout"] == ["i", "l"]

    def test_comments_and_blank_lines(self):
        source = """
        # This is a comment
        A: i, j

        # Another comment
        B: j, k
        TOUT: i, k
        """
        spec = parse_netfile(source)
        assert set(spec["tensors"]) == {"A", "B"}
        assert spec["tout"] == ["i", "k"]

    def test_empty_tout(self):
        spec = parse_netfile("A: i, j\nB: j, i\nTOUT:")
        assert spec["tout"] == []

    def test_missing_tout(self):
        spec = parse_netfile("A: i, j\nB: j, k")
        assert spec["tout"] is None

    def test_order_line(self):
        spec = parse_netfile("A: i, j\nB: j, k\nORDER: (A,B)")
        assert spec["order"] == "(A,B)"

    def test_duplicate_name_error(self):
        with pytest.raises(ValueError, match="duplicate tensor name"):
            parse_netfile("A: i, j\nA: j, k")

    def test_label_on_three_tensors_error(self):
        with pytest.raises(ValueError, match="appears on 3 tensors"):
            parse_netfile("A: i, j\nB: j, k\nC: j, l")

    def test_no_tensors_error(self):
        with pytest.raises(ValueError, match="No tensor declarations"):
            parse_netfile("# only comments\n")

    def test_no_labels_error(self):
        with pytest.raises(ValueError, match="has no labels"):
            parse_netfile("A:")

    def test_list_input(self):
        spec = parse_netfile(["A: i, j", "B: j, k"])
        assert spec["tensors"] == {"A": ["i", "j"], "B": ["j", "k"]}

    def test_string_path_probe_errors_fall_back_to_inline(self, monkeypatch):
        def _boom(self):
            raise OSError("invalid path")

        monkeypatch.setattr("pathlib.Path.is_file", _boom)
        spec = parse_netfile("A: i, j")

        assert spec["tensors"] == {"A": ["i", "j"]}

    def test_inline_source_not_treated_as_path_when_too_long(self):
        # Regression test: very long inline content should not be passed through
        # Path.is_file(), which can raise OSError([Errno 36] File name too long).
        source = "\n".join(f"T{i}: a{i}, b{i}" for i in range(40))
        spec = parse_netfile(source)
        assert len(spec["tensors"]) == 40
        assert spec["tout"] is None
        assert spec["order"] is None


# ===================================================================
# ORDER parser tests
# ===================================================================


class TestParseOrder:
    def test_simple_pair(self):
        steps = _parse_order("(A,B)", {"A", "B"})
        assert steps == [("A", "B")]

    def test_left_nested(self):
        steps = _parse_order("((A,B),C)", {"A", "B", "C"})
        assert steps == [("A", "B"), ("_0", "C")]

    def test_right_nested(self):
        steps = _parse_order("(A,(B,C))", {"A", "B", "C"})
        assert steps == [("B", "C"), ("A", "_0")]

    def test_deeply_nested(self):
        steps = _parse_order("(((A,B),C),D)", {"A", "B", "C", "D"})
        assert steps == [("A", "B"), ("_0", "C"), ("_1", "D")]

    def test_unknown_tensor_error(self):
        with pytest.raises(ValueError, match="unknown tensor name"):
            _parse_order("(A,X)", {"A", "B"})

    def test_balanced_tree(self):
        steps = _parse_order("((A,B),(C,D))", {"A", "B", "C", "D"})
        assert steps == [("A", "B"), ("C", "D"), ("_0", "_1")]


# ===================================================================
# Subscript pre-computation tests
# ===================================================================


class TestLabelsToSubscripts:
    def test_matrix_multiply(self):
        subs = _labels_to_subscripts_from_names(
            {"A": ["i", "j"], "B": ["j", "k"]},
            ["A", "B"],
            None,
        )
        # i, j, k → a, b, c; subs = "ab,bc->ac"
        assert subs == "ab,bc->ac"

    def test_trace(self):
        subs = _labels_to_subscripts_from_names(
            {"A": ["i", "j"], "B": ["j", "i"]},
            ["A", "B"],
            [],
        )
        # i, j → a, b; both contracted; subs = "ab,ba->"
        assert subs == "ab,ba->"

    def test_three_tensor_chain(self):
        subs = _labels_to_subscripts_from_names(
            {"A": ["i", "j"], "B": ["j", "k"], "C": ["k", "l"]},
            ["A", "B", "C"],
            None,
        )
        # i,j,k,l → a,b,c,d; subs = "ab,bc,cd->ad"
        assert subs == "ab,bc,cd->ad"

    def test_explicit_output_order(self):
        subs = _labels_to_subscripts_from_names(
            {"A": ["i", "j"], "B": ["j", "k"]},
            ["A", "B"],
            ["k", "i"],
        )
        assert subs == "ab,bc->ca"

    def test_invalid_output_label(self):
        with pytest.raises(ValueError, match="not a free label"):
            _labels_to_subscripts_from_names(
                {"A": ["i", "j"], "B": ["j", "k"]},
                ["A", "B"],
                ["j"],  # j is contracted, not free
            )


# ===================================================================
# Blueprint lifecycle tests
# ===================================================================


class TestNetworkBlueprint:
    def test_basic_lifecycle(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        assert bp.tensor_names == ["A", "B"]
        assert not bp.is_ready()

        A = _make_dense((3, 4), ("i", "j"), seed=0)
        B = _make_dense((4, 5), ("j", "k"), seed=1)
        bp.put_tensor("A", A)
        bp.put_tensor("B", B)
        assert bp.is_ready()

        result = bp.launch()
        assert result.labels() == ("i", "k")

        # Verify numerically
        expected = jnp.einsum("ij,jk->ik", A.todense(), B.todense())
        np.testing.assert_allclose(result.todense(), expected, atol=1e-5)

    def test_wrong_name_error(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k")
        A = _make_dense((3, 4), ("i", "j"))
        with pytest.raises(KeyError, match="Unknown tensor name"):
            bp.put_tensor("C", A)

    def test_launch_before_ready(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k")
        A = _make_dense((3, 4), ("i", "j"))
        bp.put_tensor("A", A)
        with pytest.raises(RuntimeError, match="missing tensors"):
            bp.launch()

    def test_reuse_with_different_tensors(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k")

        # First run
        A1 = _make_dense((3, 4), ("i", "j"), seed=0)
        B1 = _make_dense((4, 5), ("j", "k"), seed=1)
        bp.put_tensors({"A": A1, "B": B1})
        r1 = bp.launch()

        # Second run with different tensors
        A2 = _make_dense((3, 4), ("i", "j"), seed=2)
        B2 = _make_dense((4, 5), ("j", "k"), seed=3)
        bp.put_tensors({"A": A2, "B": B2})
        r2 = bp.launch()

        # Results should differ
        assert not jnp.allclose(r1.todense(), r2.todense())

    def test_clear_tensors(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k")
        A = _make_dense((3, 4), ("i", "j"))
        B = _make_dense((4, 5), ("j", "k"))
        bp.put_tensors({"A": A, "B": B})
        assert bp.is_ready()

        bp.clear_tensors()
        assert not bp.is_ready()

    def test_rank_mismatch_error(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k")
        bad = _make_dense((3,), ("i",))
        with pytest.raises(ValueError, match="rank"):
            bp.put_tensor("A", bad)

    def test_label_order_remapping(self):
        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        # Tensor has labels (x, y) instead of (i, j)
        A = _make_dense((3, 4), ("x", "y"), seed=0)
        B = _make_dense((4, 5), ("j", "k"), seed=1)

        bp.put_tensor("A", A, label_order=["x", "y"])
        bp.put_tensor("B", B)
        result = bp.launch()

        expected = jnp.einsum("ij,jk->ik", A.todense(), B.todense())
        np.testing.assert_allclose(result.todense(), expected, atol=1e-5)

    def test_label_order_must_match_tensor_labels(self):
        bp = NetworkBlueprint("A: i, j")
        A = _make_dense((3, 4), ("x", "y"), seed=0)

        with pytest.raises(ValueError, match="label_order .* doesn't match"):
            bp.put_tensor("A", A, label_order=["x", "z"])


# ===================================================================
# Numerical correctness
# ===================================================================


class TestNumericalCorrectness:
    def test_matrix_multiply_matches_einsum(self):
        A = _make_dense((3, 4), ("i", "j"), seed=10)
        B = _make_dense((4, 5), ("j", "k"), seed=11)

        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        bp.put_tensors({"A": A, "B": B})
        result = bp.launch()

        expected = jnp.einsum("ij,jk->ik", A.todense(), B.todense())
        np.testing.assert_allclose(result.todense(), expected, atol=1e-5)

    def test_trace_is_scalar(self):
        A = _make_dense((3, 4), ("i", "j"), seed=20)
        B = _make_dense((4, 3), ("j", "i"), seed=21)

        bp = NetworkBlueprint("A: i, j\nB: j, i\nTOUT:")
        bp.put_tensors({"A": A, "B": B})
        result = bp.launch()

        expected = jnp.einsum("ij,ji->", A.todense(), B.todense())
        np.testing.assert_allclose(result.todense(), expected, atol=1e-5)

    def test_three_tensor_chain(self):
        A = _make_dense((2, 3), ("i", "j"), seed=30)
        B = _make_dense((3, 4), ("j", "k"), seed=31)
        C = _make_dense((4, 5), ("k", "l"), seed=32)

        bp = NetworkBlueprint("A: i, j\nB: j, k\nC: k, l\nTOUT: i, l")
        bp.put_tensors({"A": A, "B": B, "C": C})
        result = bp.launch()

        expected = jnp.einsum("ij,jk,kl->il", A.todense(), B.todense(), C.todense())
        np.testing.assert_allclose(result.todense(), expected, atol=1e-5)

    def test_dmrg_style_contraction(self):
        """DMRG effective Hamiltonian: L[a,b,c] M[a,p,q,d] A[b,p,s,e] M[e,q,t,f] R[d,f,g]."""
        L = _make_dense((2, 2, 2), ("a", "b", "c"), seed=40)
        M1 = _make_dense((2, 2, 2, 2), ("a", "p", "q", "d"), seed=41)
        A = _make_dense((2, 2, 2, 2), ("b", "p", "s", "e"), seed=42)
        M2 = _make_dense((2, 2, 2, 2), ("e", "q", "t", "f"), seed=43)
        R = _make_dense((2, 2, 2), ("d", "f", "g"), seed=44)

        bp = NetworkBlueprint("""
        L: a, b, c
        M1: a, p, q, d
        A: b, p, s, e
        M2: e, q, t, f
        R: d, f, g
        TOUT: c, s, t, g
        """)
        bp.put_tensors({"L": L, "M1": M1, "A": A, "M2": M2, "R": R})
        result = bp.launch()

        expected = jnp.einsum(
            "abc,apqd,bpse,eqtf,dfg->cstg",
            L.todense(),
            M1.todense(),
            A.todense(),
            M2.todense(),
            R.todense(),
        )
        np.testing.assert_allclose(result.todense(), expected, atol=1e-4)


# ===================================================================
# ORDER vs no-ORDER consistency
# ===================================================================


class TestOrderConsistency:
    def test_two_tensors_with_and_without_order(self):
        A = _make_dense((3, 4), ("i", "j"), seed=50)
        B = _make_dense((4, 5), ("j", "k"), seed=51)

        bp_no = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        bp_no.put_tensors({"A": A, "B": B})
        r_no = bp_no.launch()

        bp_ord = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k\nORDER: (A,B)")
        bp_ord.put_tensors({"A": A, "B": B})
        r_ord = bp_ord.launch()

        np.testing.assert_allclose(r_no.todense(), r_ord.todense(), atol=1e-5)

    def test_three_tensors_with_and_without_order(self):
        A = _make_dense((2, 3), ("i", "j"), seed=60)
        B = _make_dense((3, 4), ("j", "k"), seed=61)
        C = _make_dense((4, 5), ("k", "l"), seed=62)

        bp_no = NetworkBlueprint("A: i, j\nB: j, k\nC: k, l\nTOUT: i, l")
        bp_no.put_tensors({"A": A, "B": B, "C": C})
        r_no = bp_no.launch()

        bp_ord = NetworkBlueprint(
            "A: i, j\nB: j, k\nC: k, l\nTOUT: i, l\nORDER: ((A,B),C)"
        )
        bp_ord.put_tensors({"A": A, "B": B, "C": C})
        r_ord = bp_ord.launch()

        np.testing.assert_allclose(r_no.todense(), r_ord.todense(), atol=1e-5)

    def test_right_nested_order(self):
        A = _make_dense((2, 3), ("i", "j"), seed=70)
        B = _make_dense((3, 4), ("j", "k"), seed=71)
        C = _make_dense((4, 5), ("k", "l"), seed=72)

        bp_no = NetworkBlueprint("A: i, j\nB: j, k\nC: k, l\nTOUT: i, l")
        bp_no.put_tensors({"A": A, "B": B, "C": C})
        r_no = bp_no.launch()

        bp_ord = NetworkBlueprint(
            "A: i, j\nB: j, k\nC: k, l\nTOUT: i, l\nORDER: (A,(B,C))"
        )
        bp_ord.put_tensors({"A": A, "B": B, "C": C})
        r_ord = bp_ord.launch()

        np.testing.assert_allclose(r_no.todense(), r_ord.todense(), atol=1e-5)


# ===================================================================
# launch_raw tests
# ===================================================================


class TestLaunchRaw:
    def test_matrix_multiply(self):
        """launch_raw should produce same result as launch on raw arrays."""
        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        A = _make_dense((3, 4), ("i", "j"), seed=0)
        B = _make_dense((4, 5), ("j", "k"), seed=1)
        bp.put_tensors({"A": A, "B": B})

        result_tensor = bp.launch()
        result_raw = bp.launch_raw(A.todense(), B.todense())

        np.testing.assert_allclose(result_tensor.todense(), result_raw, atol=1e-6)

    def test_trace_scalar(self):
        """launch_raw should handle scalar output (empty TOUT)."""
        bp = NetworkBlueprint("A: i, j\nB: j, i\nTOUT:")
        A = _make_dense((3, 4), ("i", "j"), seed=20)
        B = _make_dense((4, 3), ("j", "i"), seed=21)
        bp.put_tensors({"A": A, "B": B})

        result_tensor = bp.launch()
        result_raw = bp.launch_raw(A.todense(), B.todense())

        np.testing.assert_allclose(result_tensor.todense(), result_raw, atol=1e-5)

    def test_three_tensor_chain(self):
        """launch_raw works with 3+ tensors."""
        bp = NetworkBlueprint("A: i, j\nB: j, k\nC: k, l\nTOUT: i, l")
        A = _make_dense((2, 3), ("i", "j"), seed=30)
        B = _make_dense((3, 4), ("j", "k"), seed=31)
        C = _make_dense((4, 5), ("k", "l"), seed=32)

        result_raw = bp.launch_raw(A.todense(), B.todense(), C.todense())
        expected = jnp.einsum("ij,jk,kl->il", A.todense(), B.todense(), C.todense())

        np.testing.assert_allclose(result_raw, expected, atol=1e-5)

    def test_wrong_number_of_arrays(self):
        """launch_raw should raise on wrong number of arrays."""
        bp = NetworkBlueprint("A: i, j\nB: j, k")
        A_arr = jnp.ones((3, 4))
        with pytest.raises(ValueError, match="Expected 2 arrays"):
            bp.launch_raw(A_arr)

    def test_dmrg_style(self):
        """launch_raw on a 5-tensor DMRG-style contraction."""
        bp = NetworkBlueprint("""
        L: a, b, c
        M1: a, p, q, d
        A: b, p, s, e
        M2: e, q, t, f
        R: d, f, g
        TOUT: c, s, t, g
        """)
        L = _make_dense((2, 2, 2), ("a", "b", "c"), seed=40)
        M1 = _make_dense((2, 2, 2, 2), ("a", "p", "q", "d"), seed=41)
        A = _make_dense((2, 2, 2, 2), ("b", "p", "s", "e"), seed=42)
        M2 = _make_dense((2, 2, 2, 2), ("e", "q", "t", "f"), seed=43)
        R = _make_dense((2, 2, 2), ("d", "f", "g"), seed=44)

        result_raw = bp.launch_raw(
            L.todense(), M1.todense(), A.todense(), M2.todense(), R.todense()
        )
        expected = jnp.einsum(
            "abc,apqd,bpse,eqtf,dfg->cstg",
            L.todense(),
            M1.todense(),
            A.todense(),
            M2.todense(),
            R.todense(),
        )
        np.testing.assert_allclose(result_raw, expected, atol=1e-4)

    def test_jit_compatible(self):
        """launch_raw should be JIT-traceable."""
        import jax

        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        A_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B_arr = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        @jax.jit
        def f(a, b):
            return bp.launch_raw(a, b)

        result = f(A_arr, B_arr)
        expected = A_arr @ B_arr
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ===================================================================
# Convenience function
# ===================================================================


class TestFromNetfile:
    def test_from_netfile(self):
        bp = from_netfile("A: i, j\nB: j, k")
        assert isinstance(bp, NetworkBlueprint)
        assert bp.tensor_names == ["A", "B"]


# ===================================================================
# to_tensor_network interop
# ===================================================================


class TestToTensorNetwork:
    def test_to_tensor_network_contracts_same(self):
        # Bond leg "j" must have compatible flows (IN on A, OUT on B)
        # so that to_tensor_network() can actually connect them.
        u1 = U1Symmetry()
        rng_a = np.random.RandomState(80)
        rng_b = np.random.RandomState(81)
        data_a = jnp.array(rng_a.randn(3, 4), dtype=jnp.float32)
        data_b = jnp.array(rng_b.randn(4, 5), dtype=jnp.float32)
        A = DenseTensor(
            data_a,
            (
                TensorIndex.from_charges(
                    u1, np.zeros(3, dtype=np.int32), FlowDirection.IN, label="i"
                ),
                TensorIndex.from_charges(
                    u1, np.zeros(4, dtype=np.int32), FlowDirection.OUT, label="j"
                ),
            ),
        )
        B = DenseTensor(
            data_b,
            (
                TensorIndex.from_charges(
                    u1, np.zeros(4, dtype=np.int32), FlowDirection.IN, label="j"
                ),
                TensorIndex.from_charges(
                    u1, np.zeros(5, dtype=np.int32), FlowDirection.IN, label="k"
                ),
            ),
        )

        bp = NetworkBlueprint("A: i, j\nB: j, k\nTOUT: i, k")
        bp.put_tensors({"A": A, "B": B})
        r_bp = bp.launch()

        tn = bp.to_tensor_network()
        r_tn = tn.contract(output_labels=["i", "k"])

        np.testing.assert_allclose(r_bp.todense(), r_tn.todense(), atol=1e-5)
