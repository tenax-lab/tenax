"""Shared fixtures for the Tenax test suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Auto-apply markers based on test file name                           #
# ------------------------------------------------------------------ #

_FILE_MARKERS = {
    "test_tensor.py": "core",
    "test_index.py": "core",
    "test_symmetry.py": "core",
    "test_contraction.py": "core",
    "test_network.py": "core",
    "test_netfile.py": "core",
    "test_fermionic.py": "core",
    "test_dmrg.py": "algorithm",
    "test_idmrg.py": "algorithm",
    "test_trg.py": "algorithm",
    "test_hotrg.py": "algorithm",
    "test_ipeps.py": "algorithm",
    "test_auto_mpo.py": "algorithm",
    "test_ad_utils.py": "algorithm",
    "test_fermionic_ipeps.py": "algorithm",
    "test_ipeps_excitations.py": "slow",
}


def pytest_collection_modifyitems(items):
    for item in items:
        filename = item.path.name
        if filename in _FILE_MARKERS:
            item.add_marker(getattr(pytest.mark, _FILE_MARKERS[filename]))


# ------------------------------------------------------------------ #
# Symmetry fixtures                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1():
    return U1Symmetry()


@pytest.fixture
def z2():
    return ZnSymmetry(2)


@pytest.fixture
def z3():
    return ZnSymmetry(3)


# ------------------------------------------------------------------ #
# Random key fixture                                                   #
# ------------------------------------------------------------------ #


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng2():
    return jax.random.PRNGKey(99)


# ------------------------------------------------------------------ #
# TensorIndex fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_charges_3(u1):
    """U(1) charges [-1, 0, 1] — typical for spin-1 or bond dim 3."""
    return np.array([-1, 0, 1], dtype=np.int32)


@pytest.fixture
def u1_charges_2(u1):
    """U(1) charges [-1, 1] — typical for spin-1/2."""
    return np.array([-1, 1], dtype=np.int32)


@pytest.fixture
def idx_in_3(u1, u1_charges_3):
    """U(1) IN index with charges [-1, 0, 1], label='left'."""
    return TensorIndex(u1, u1_charges_3, FlowDirection.IN, label="left")


@pytest.fixture
def idx_out_3(u1, u1_charges_3):
    """U(1) OUT index with dual charges [1, 0, -1], label='right'.
    This is the proper dual of idx_in_3.
    """
    return TensorIndex(u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="right")


@pytest.fixture
def u1_index_pair(idx_in_3, idx_out_3):
    """A compatible pair of U(1) indices (IN and its dual OUT)."""
    return idx_in_3, idx_out_3


# ------------------------------------------------------------------ #
# DenseTensor fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_dense_matrix(u1, rng):
    """A 3x3 DenseTensor (matrix) with U(1) indices."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3, 3))
    indices = (
        TensorIndex(u1, charges, FlowDirection.IN, label="row"),
        TensorIndex(u1, charges, FlowDirection.OUT, label="col"),
    )
    return DenseTensor(data, indices)


@pytest.fixture
def dense_vector(u1, rng):
    """A 3-element DenseTensor (vector) with U(1) index."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3,))
    idx = TensorIndex(u1, charges, FlowDirection.IN, label="vec")
    return DenseTensor(data, (idx,))


# ------------------------------------------------------------------ #
# SymmetricTensor fixtures                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_sym_tensor_2leg(u1, rng):
    """2-leg U(1)-symmetric tensor: IN x OUT, charges [-1, 0, 1]."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex(u1, charges, FlowDirection.IN, label="in"),
        TensorIndex(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_3leg(u1, rng):
    """3-leg U(1)-symmetric tensor: phys x left x right."""
    phys_c = np.array([-1, 1], dtype=np.int32)
    virt_c = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex(u1, phys_c, FlowDirection.IN, label="phys"),
        TensorIndex(u1, virt_c, FlowDirection.IN, label="left"),
        TensorIndex(u1, u1.dual(virt_c), FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_pair(u1, rng, rng2):
    """A pair of 3-leg U(1)-symmetric tensors that can be contracted on 'bond'.

    Both tensors use the SAME charge array for the shared bond leg with
    opposite flow directions (OUT for A, IN for B). Same charge array means
    position i in A's bond and position i in B's bond store the same charge
    value, so dense einsum contraction works correctly (no ordering mismatch).
    """
    phys_c = np.array([-1, 1], dtype=np.int32)
    bond_c = np.array(
        [-1, 0, 1], dtype=np.int32
    )  # same charges for both ends of shared bond

    indices_A = (
        TensorIndex(u1, phys_c, FlowDirection.IN, label="p0"),
        TensorIndex(u1, bond_c, FlowDirection.IN, label="bond_left"),
        TensorIndex(
            u1, bond_c, FlowDirection.OUT, label="bond"
        ),  # OUT end of shared bond
    )
    indices_B = (
        TensorIndex(u1, phys_c, FlowDirection.IN, label="p1"),
        TensorIndex(
            u1, bond_c, FlowDirection.IN, label="bond"
        ),  # IN end of shared bond (same charges)
        TensorIndex(u1, bond_c, FlowDirection.OUT, label="bond_right"),
    )
    A = SymmetricTensor.random_normal(indices_A, rng)
    B = SymmetricTensor.random_normal(indices_B, rng2)
    return A, B
