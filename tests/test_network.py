"""Tests for TensorNetwork and MPS/PEPS builders."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.network.network import TensorNetwork, build_mps, build_peps


def make_tensor(u1, shape, labels, flows=None, seed=0):
    """Helper: create a DenseTensor with zero charges (no symmetry constraint)."""
    if flows is None:
        flows = [FlowDirection.IN] * len(shape)
    charges = [np.zeros(s, dtype=np.int32) for s in shape]
    indices = tuple(
        TensorIndex.from_charges(u1, charges[i], flows[i], label=labels[i])
        for i in range(len(shape))
    )
    data = jax.random.normal(jax.random.PRNGKey(seed), shape)
    return DenseTensor(data, indices)


@pytest.fixture
def u1():
    return U1Symmetry()


class TestTensorNetworkNodes:
    def test_add_and_retrieve_node(self, u1):
        tn = TensorNetwork()
        t = make_tensor(u1, (3,), ["a"])
        tn.add_node("A", t)
        retrieved = tn.get_tensor("A")
        assert retrieved is t

    def test_add_duplicate_raises(self, u1):
        tn = TensorNetwork()
        t = make_tensor(u1, (3,), ["a"])
        tn.add_node("A", t)
        with pytest.raises(ValueError, match="already exists"):
            tn.add_node("A", t)

    def test_add_duplicate_label_raises(self, u1):
        tn = TensorNetwork()
        t = make_tensor(u1, (3, 3), ["a", "a"])  # duplicate label
        with pytest.raises(ValueError, match="duplicate"):
            tn.add_node("bad", t)

    def test_remove_node(self, u1):
        tn = TensorNetwork()
        t = make_tensor(u1, (3,), ["a"])
        tn.add_node("A", t)
        returned = tn.remove_node("A")
        assert returned is t
        assert "A" not in tn.node_ids()

    def test_remove_nonexistent_raises(self, u1):
        tn = TensorNetwork()
        with pytest.raises(KeyError):
            tn.remove_node("X")

    def test_replace_tensor(self, u1):
        tn = TensorNetwork()
        t1 = make_tensor(u1, (3,), ["a"], seed=1)
        t2 = make_tensor(u1, (3,), ["a"], seed=2)
        tn.add_node("A", t1)
        tn.replace_tensor("A", t2)
        assert tn.get_tensor("A") is t2

    def test_replace_different_labels_raises(self, u1):
        tn = TensorNetwork()
        t1 = make_tensor(u1, (3,), ["a"])
        t2 = make_tensor(u1, (3,), ["b"])  # different label
        tn.add_node("A", t1)
        with pytest.raises(ValueError, match="labels"):
            tn.replace_tensor("A", t2)

    def test_n_nodes(self, u1):
        tn = TensorNetwork()
        for i in range(4):
            tn.add_node(i, make_tensor(u1, (2,), [f"x{i}"]))
        assert tn.n_nodes() == 4


class TestTensorNetworkEdges:
    def test_connect_and_disconnect(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="bond"),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_a"),
            ),
        )
        B = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="bond"),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_b"),
            ),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "bond", "B", "bond")
        assert tn.n_edges() == 1

        tn.disconnect("A", "bond", "B", "bond")
        assert tn.n_edges() == 0

    def test_connect_incompatible_raises(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="leg"),),
        )
        charges2 = np.zeros(4, dtype=np.int32)  # different dimension
        B = DenseTensor(
            jnp.ones((4,)),
            (TensorIndex.from_charges(u1, charges2, FlowDirection.OUT, label="leg"),),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        with pytest.raises(ValueError, match="Incompatible"):
            tn.connect("A", "leg", "B", "leg")

    def test_connect_missing_label_raises(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),),
        )
        B = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="b"),),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        with pytest.raises(KeyError):
            tn.connect("A", "nonexistent", "B", "b")

    def test_connect_by_shared_label(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="shared"),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_a"),
            ),
        )
        B = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(
                    u1, charges, FlowDirection.OUT, label="shared"
                ),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_b"),
            ),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        count = tn.connect_by_shared_label("A", "B")
        assert count == 1
        assert tn.n_edges() == 1

    def test_connect_by_shared_label_no_shared_raises(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),),
        )
        B = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="b"),),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        with pytest.raises(ValueError, match="No shared"):
            tn.connect_by_shared_label("A", "B")

    def test_open_legs(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="shared"),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys"),
            ),
        )
        B = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(
                    u1, charges, FlowDirection.OUT, label="shared"
                ),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys2"),
            ),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect_by_shared_label("A", "B")

        open_A = tn.open_legs("A")
        assert "phys" in open_A
        assert "shared" not in open_A

    def test_relabel_bond(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="old"),),
        )
        tn.add_node("A", A)
        tn.relabel_bond("A", "old", "new")
        assert "new" in tn.get_tensor("A").labels()

    def test_cache_invalidated_on_connect(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),),
        )
        B = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="a"),),
        )
        tn.add_node("A", A)
        tn.add_node("B", B)

        # Contract once (populates cache)
        r1 = tn.contract()

        # Connect changes the graph → cache should be cleared
        tn.connect("A", "a", "B", "a")
        r2 = tn.contract()

        # Results should differ (connected vs disconnected)
        # Just check it doesn't error
        assert r1 is not r2


class TestTensorNetworkContraction:
    def test_single_node_contract(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(3, dtype=np.int32)
        data = jnp.ones((3,))
        A = DenseTensor(
            data, (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="a"),)
        )
        tn.add_node("A", A)
        result = tn.contract(["A"])
        np.testing.assert_allclose(result.todense(), data, rtol=1e-5)

    def test_two_node_contraction(self, u1):
        """Contract two 3-vectors sharing label 'i' → scalar (dot product)."""
        charges = np.zeros(3, dtype=np.int32)
        data = jnp.array([1.0, 2.0, 3.0])

        A = DenseTensor(
            data, (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="i"),)
        )
        B = DenseTensor(
            data, (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="i"),)
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "i", "B", "i")

        result = tn.contract()
        expected = jnp.dot(data, data)
        np.testing.assert_allclose(float(result.todense()), float(expected), rtol=1e-5)

    def test_cache_hit(self, u1):
        tn = TensorNetwork()
        charges = np.zeros(4, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((4,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="x"),),
        )
        tn.add_node("A", A)

        r1 = tn.contract()
        r2 = tn.contract()  # should come from cache
        assert r1 is r2

    def test_cache_key_respects_node_order(self, u1):
        """Cache must distinguish node orderings when output_labels=None.

        When output_labels is not specified, the free-label order depends on
        the iteration order of the node list.  Using frozenset(nodes) as the
        cache key collapsed ['A','B'] and ['B','A'] into the same entry,
        returning the wrong leg ordering for the second call.

        Regression test for the frozenset -> tuple fix in contract().
        """
        charges_a = np.zeros(2, dtype=np.int32)
        charges_b = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((2,)),
            (TensorIndex.from_charges(u1, charges_a, FlowDirection.IN, label="leg_a"),),
        )
        B = DenseTensor(
            jnp.ones((3,)),
            (TensorIndex.from_charges(u1, charges_b, FlowDirection.IN, label="leg_b"),),
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)

        # Contract in A,B order — free labels should be [leg_a, leg_b]
        r_ab = tn.contract(nodes=["A", "B"], output_labels=None)
        # Contract in B,A order — free labels should be [leg_b, leg_a]
        r_ba = tn.contract(nodes=["B", "A"], output_labels=None)

        # The label orders must reflect the node ordering
        assert r_ab.labels() == ("leg_a", "leg_b")
        assert r_ba.labels() == ("leg_b", "leg_a")

        # The underlying data should be transposed relative to each other
        np.testing.assert_allclose(r_ab.todense(), r_ba.todense().T, rtol=1e-7)

    def test_disconnected_same_label_not_contracted(self, u1):
        """Two unconnected tensors sharing a label should NOT be contracted."""
        charges = np.zeros(3, dtype=np.int32)
        A = DenseTensor(
            jnp.array([1.0, 2.0, 3.0]),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="p"),),
        )
        B = DenseTensor(
            jnp.array([4.0, 5.0, 6.0]),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="p"),),
        )
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        # No tn.connect() — the two "p" legs are NOT connected
        result = tn.contract()
        # Should be an outer product (3, 3), not a scalar dot product
        assert result.todense().shape == (3, 3)

    def test_repr(self, u1):
        tn = TensorNetwork(name="test")
        r = repr(tn)
        assert "test" in r
        assert "nodes=0" in r


class TestBuildMPS:
    def test_chain_of_3(self, u1):
        """Build a 3-site MPS and check structure."""
        charges_phys = np.zeros(2, dtype=np.int32)
        charges_bond = np.zeros(4, dtype=np.int32)

        tensors = []
        for i in range(3):
            if i == 0:
                indices = (
                    TensorIndex.from_charges(
                        u1, charges_phys, FlowDirection.IN, label=f"p{i}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                    ),
                )
                data = jax.random.normal(jax.random.PRNGKey(i), (2, 4))
            elif i == 2:
                indices = (
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.IN, label=f"v{i - 1}_{i}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_phys, FlowDirection.IN, label=f"p{i}"
                    ),
                )
                data = jax.random.normal(jax.random.PRNGKey(i), (4, 2))
            else:
                indices = (
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.IN, label=f"v{i - 1}_{i}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_phys, FlowDirection.IN, label=f"p{i}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                    ),
                )
                data = jax.random.normal(jax.random.PRNGKey(i), (4, 2, 4))
            tensors.append(DenseTensor(data, indices))

        mps = build_mps(tensors)
        assert mps.n_nodes() == 3
        # Virtual bonds should be connected
        assert mps.n_edges() >= 2

    def test_incompatible_shared_label_raises(self, u1):
        charges_phys = np.zeros(2, dtype=np.int32)
        charges_bond = np.zeros(3, dtype=np.int32)
        tensors = [
            DenseTensor(
                jnp.ones((2, 3)),
                (
                    TensorIndex.from_charges(
                        u1, charges_phys, FlowDirection.IN, label="p0"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.OUT, label="v0_1"
                    ),
                ),
            ),
            DenseTensor(
                jnp.ones((3, 2)),
                (
                    TensorIndex.from_charges(
                        u1, charges_bond, FlowDirection.OUT, label="v0_1"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_phys, FlowDirection.IN, label="p1"
                    ),
                ),
            ),
        ]

        with pytest.raises(ValueError, match="Incompatible indices"):
            build_mps(tensors)


class TestCacheMutationFuzz:
    """Cache must be invalidated after every structural mutation.

    Each test verifies that contract(cache=True) and contract(cache=False)
    agree immediately after a mutation, proving no stale entry was returned.
    """

    def _make_vec(self, u1, label, scale=1.0, seed=0, dim=3):
        charges = np.zeros(dim, dtype=np.int32)
        data = jax.random.normal(jax.random.PRNGKey(seed), (dim,)) * scale
        return DenseTensor(
            data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label=label),),
        )

    def test_replace_tensor_invalidates_cache(self, u1):
        """After replace_tensor the cache must not return the old result."""
        tn = TensorNetwork()
        t1 = self._make_vec(u1, "a", scale=1.0, seed=1)
        t2 = self._make_vec(u1, "a", scale=2.0, seed=2)
        tn.add_node("A", t1)

        r_before = tn.contract()
        tn.replace_tensor("A", t2)
        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        # Sanity: replacing with a different tensor changed the result
        assert not jnp.allclose(r_before.todense(), r_cached.todense())

    def test_add_node_invalidates_cache(self, u1):
        """After add_node the cached all-node contraction must include the new node."""
        tn = TensorNetwork()
        t_a = self._make_vec(u1, "x", seed=10)
        t_b = self._make_vec(u1, "y", seed=11)
        tn.add_node("A", t_a)

        r1 = tn.contract()  # only A, shape (3,)
        tn.add_node("B", t_b)

        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        # Adding B changes the result shape (outer product with B)
        assert r_cached.ndim == r1.ndim + t_b.ndim

    def test_remove_node_invalidates_cache(self, u1):
        """After remove_node the cache must not include the removed tensor."""
        tn = TensorNetwork()
        t_a = self._make_vec(u1, "x", seed=20)
        t_b = self._make_vec(u1, "y", seed=21)
        tn.add_node("A", t_a)
        tn.add_node("B", t_b)

        r_both = tn.contract()  # A ⊗ B, shape (3, 3)
        tn.remove_node("B")

        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        assert r_cached.ndim < r_both.ndim

    def test_connect_invalidates_cache(self, u1):
        """After connect, contraction changes from outer product to dot product.

        A has label "la", B has label "lb" — different labels means outer product
        before connecting. After connecting "la"-"lb", the contraction relabels
        B's leg to "la" so both share a label → dot product (scalar).
        """
        charges = np.zeros(3, dtype=np.int32)
        data = jnp.array([1.0, 2.0, 3.0])
        A = DenseTensor(
            data, (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="la"),)
        )
        B = DenseTensor(
            data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="lb"),),
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)

        r_outer = tn.contract()  # outer product, shape (3, 3)
        tn.connect("A", "la", "B", "lb")

        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        # After connecting, the bond is contracted → scalar
        assert r_cached.ndim < r_outer.ndim

    def test_disconnect_invalidates_cache(self, u1):
        """After disconnect, contraction expands back to outer product."""
        charges = np.zeros(3, dtype=np.int32)
        data = jnp.array([1.0, 2.0, 3.0])
        A = DenseTensor(
            data, (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="la"),)
        )
        B = DenseTensor(
            data,
            (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="lb"),),
        )

        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "la", "B", "lb")

        r_contracted = tn.contract()  # scalar
        tn.disconnect("A", "la", "B", "lb")

        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        assert r_cached.ndim > r_contracted.ndim

    def test_relabel_bond_invalidates_cache(self, u1):
        """After relabel_bond the cache is cleared and the new label is used."""
        tn = TensorNetwork()
        t = self._make_vec(u1, "old", seed=30)
        tn.add_node("A", t)

        tn.contract()
        tn.relabel_bond("A", "old", "new")

        r_cached = tn.contract(cache=True)
        r_fresh = tn.contract(cache=False)

        np.testing.assert_allclose(r_cached.todense(), r_fresh.todense(), rtol=1e-5)
        # The new label is on the result tensor
        assert "new" in tn.get_tensor("A").labels()

    def test_sequential_mutations_cache_consistency(self, u1):
        """Multiple mutations in sequence: cache always agrees with fresh result."""
        charges = np.zeros(3, dtype=np.int32)
        tn = TensorNetwork()

        for seed, scale in enumerate([1.0, 2.0, 3.0], start=40):
            data = jax.random.normal(jax.random.PRNGKey(seed), (3,)) * scale
            t = DenseTensor(
                data,
                (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="v"),),
            )
            if "A" not in tn.node_ids():
                tn.add_node("A", t)
            else:
                tn.replace_tensor("A", t)

            r_cached = tn.contract(cache=True)
            r_fresh = tn.contract(cache=False)
            np.testing.assert_allclose(
                r_cached.todense(),
                r_fresh.todense(),
                rtol=1e-5,
                err_msg=f"Cache inconsistency after replace with scale={scale}",
            )


class TestBuildPEPS:
    def test_2x2_peps(self, u1):
        """Build a 2x2 PEPS grid."""
        d_phys = 2
        D_bond = 3
        charges_p = np.zeros(d_phys, dtype=np.int32)
        charges_b = np.zeros(D_bond, dtype=np.int32)

        tensors = []
        for i in range(2):
            row = []
            for j in range(2):
                indices = (
                    TensorIndex.from_charges(
                        u1, charges_p, FlowDirection.IN, label=f"p{i}{j}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_b, FlowDirection.OUT, label=f"h{i}_{j}"
                    ),
                    TensorIndex.from_charges(
                        u1, charges_b, FlowDirection.OUT, label=f"v{i}_{j}"
                    ),
                )
                data = jax.random.normal(
                    jax.random.PRNGKey(i * 2 + j), (d_phys, D_bond, D_bond)
                )
                row.append(DenseTensor(data, indices))
            tensors.append(row)

        peps = build_peps(tensors, 2, 2)
        assert peps.n_nodes() == 4

    def test_invalid_row_count_raises(self, u1):
        charges = np.zeros(2, dtype=np.int32)
        tensor = DenseTensor(
            jnp.ones((2,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="p"),),
        )

        with pytest.raises(ValueError, match="rows"):
            build_peps([[tensor]], 2, 1)

    def test_invalid_column_count_raises(self, u1):
        charges = np.zeros(2, dtype=np.int32)
        tensor = DenseTensor(
            jnp.ones((2,)),
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="p"),),
        )

        with pytest.raises(ValueError, match="columns"):
            build_peps([[tensor, tensor]], 1, 1)

    def test_incompatible_shared_label_raises(self, u1):
        charges_phys = np.zeros(2, dtype=np.int32)
        charges_bond = np.zeros(3, dtype=np.int32)
        tensors = [
            [
                DenseTensor(
                    jnp.ones((2, 3)),
                    (
                        TensorIndex.from_charges(
                            u1, charges_phys, FlowDirection.IN, label="p00"
                        ),
                        TensorIndex.from_charges(
                            u1, charges_bond, FlowDirection.OUT, label="h0_0_1"
                        ),
                    ),
                ),
                DenseTensor(
                    jnp.ones((3, 2)),
                    (
                        TensorIndex.from_charges(
                            u1, charges_bond, FlowDirection.OUT, label="h0_0_1"
                        ),
                        TensorIndex.from_charges(
                            u1, charges_phys, FlowDirection.IN, label="p01"
                        ),
                    ),
                ),
            ]
        ]

        with pytest.raises(ValueError, match="Incompatible indices"):
            build_peps(tensors, 1, 2)


class TestEdgeLabelOwnership:
    """Regression tests for edge-label ownership tracking.

    networkx.MultiGraph.edges(node) always yields u=node as the first
    element, so checking `u == node_id` to decide between label_a and
    label_b is always True. The fix stores owner_a/owner_b in edge data.

    These tests use *different* label names on each side of a bond
    (e.g. "left" on A, "right" on B) to distinguish ownership.
    """

    def _make_pair(self, u1, dim=3):
        """Return (A, B) with different bond labels and compatible flows."""
        charges = np.zeros(dim, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((dim, dim)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_a"),
                TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="left"),
            ),
        )
        B = DenseTensor(
            jnp.ones((dim, dim)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="right"),
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys_b"),
            ),
        )
        return A, B

    def test_open_legs_with_different_bond_labels(self, u1):
        """open_legs must report the correct label for each node."""
        A, B = self._make_pair(u1)
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "left", "B", "right")

        open_a = tn.open_legs("A")
        open_b = tn.open_legs("B")

        # "left" is connected on A → should NOT appear as open
        assert "left" not in open_a
        assert "phys_a" in open_a

        # "right" is connected on B → should NOT appear as open
        assert "right" not in open_b
        assert "phys_b" in open_b

    def test_relabel_bond_with_different_labels(self, u1):
        """relabel_bond must update the correct side of an asymmetric bond."""
        A, B = self._make_pair(u1)
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "left", "B", "right")

        # Relabel B's side of the bond
        tn.relabel_bond("B", "right", "new_right")

        # B's tensor should have the new label
        assert "new_right" in tn.get_tensor("B").labels()
        assert "right" not in tn.get_tensor("B").labels()

        # The bond should still work — open_legs should not include new_right
        assert "new_right" not in tn.open_legs("B")

    def test_disconnect_reversed_argument_order(self, u1):
        """disconnect(B, lb, A, la) must work even though connect(A, la, B, lb)."""
        A, B = self._make_pair(u1)
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "left", "B", "right")
        assert tn.n_edges() == 1

        # Disconnect with reversed node order
        tn.disconnect("B", "right", "A", "left")
        assert tn.n_edges() == 0

    def test_contraction_with_different_bond_labels(self, u1):
        """Contraction must correctly pair legs even with asymmetric labels."""
        charges = np.zeros(3, dtype=np.int32)
        data_a = jnp.array([1.0, 2.0, 3.0])
        data_b = jnp.array([4.0, 5.0, 6.0])
        A = DenseTensor(
            data_a,
            (TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="x"),),
        )
        B = DenseTensor(
            data_b,
            (TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="y"),),
        )
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "x", "B", "y")

        result = tn.contract()
        expected = float(jnp.dot(data_a, data_b))
        np.testing.assert_allclose(float(result.todense()), expected, rtol=1e-5)

    def test_replace_tensor_validates_connected_leg(self, u1):
        """replace_tensor must validate the leg that is actually connected."""
        charges3 = np.zeros(3, dtype=np.int32)
        charges4 = np.zeros(4, dtype=np.int32)
        A = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges3, FlowDirection.IN, label="phys"),
                TensorIndex.from_charges(u1, charges3, FlowDirection.OUT, label="left"),
            ),
        )
        B = DenseTensor(
            jnp.ones((3, 3)),
            (
                TensorIndex.from_charges(u1, charges3, FlowDirection.IN, label="right"),
                TensorIndex.from_charges(
                    u1, charges3, FlowDirection.IN, label="phys_b"
                ),
            ),
        )
        tn = TensorNetwork()
        tn.add_node("A", A)
        tn.add_node("B", B)
        tn.connect("A", "left", "B", "right")

        # Replacing B with a tensor that changes dim on "right" should raise
        B_bad = DenseTensor(
            jnp.ones((4, 3)),
            (
                TensorIndex.from_charges(u1, charges4, FlowDirection.IN, label="right"),
                TensorIndex.from_charges(
                    u1, charges3, FlowDirection.IN, label="phys_b"
                ),
            ),
        )
        with pytest.raises(ValueError, match="dimension"):
            tn.replace_tensor("B", B_bad)

    def test_self_loop_both_labels_connected(self, u1):
        """Self-loop: both labels on the same node must be marked connected."""
        charges = np.zeros(3, dtype=np.int32)
        T = DenseTensor(
            jnp.ones((3, 3, 3)),
            (
                TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="phys"),
                TensorIndex.from_charges(
                    u1, charges, FlowDirection.OUT, label="trace_a"
                ),
                TensorIndex.from_charges(
                    u1, charges, FlowDirection.IN, label="trace_b"
                ),
            ),
        )
        tn = TensorNetwork()
        tn.add_node("T", T)
        tn.connect("T", "trace_a", "T", "trace_b")

        open_t = tn.open_legs("T")
        assert "trace_a" not in open_t
        assert "trace_b" not in open_t
        assert "phys" in open_t
