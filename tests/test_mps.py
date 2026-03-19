"""Tests for FiniteMPS class."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _make_dense_mps(L=4, d=2, chi=3, key=None):
    """Helper: build a random dense MPS as list[DenseTensor]."""
    if key is None:
        key = jax.random.PRNGKey(0)
    u1 = U1Symmetry()
    tensors = []
    for i in range(L):
        if i == 0:
            shape = (d, chi)
            indices = (
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            shape = (chi, d)
            indices = (
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
            )
        else:
            shape = (chi, d, chi)
            indices = (
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(u1, np.zeros(d, dtype=np.int32), IN, label=f"p{i}"),
                TensorIndex(
                    u1, np.zeros(chi, dtype=np.int32), OUT, label=f"v{i}_{i + 1}"
                ),
            )
        key, subkey = jax.random.split(key)
        data = jax.random.normal(subkey, shape)
        tensors.append(DenseTensor(data, indices))
    return tensors


def _make_symmetric_mps(L=4, d=2, chi=4, key=None):
    """Helper: build a random U(1)-symmetric MPS as list[SymmetricTensor]."""
    if key is None:
        key = jax.random.PRNGKey(0)
    sym = U1Symmetry()
    phys_charges = np.array([1, -1], dtype=np.int32)
    # Virtual charges: sectors -1, 0, +1 distributed across chi states
    # (mirrors build_random_symmetric_mps logic for target_charge=0)
    required_charges = [-1, 0, 1]
    n_sectors = len(required_charges)
    per_sector = max(1, chi // n_sectors)
    arrays = [np.full(per_sector, q, dtype=np.int32) for q in required_charges]
    virt_charges = np.concatenate(arrays)[:chi]
    if len(virt_charges) < chi:
        pad = np.full(chi - len(virt_charges), 0, dtype=np.int32)
        virt_charges = np.concatenate([virt_charges, pad])

    tensors = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        if i == 0:
            indices = (
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
                TensorIndex(sym, virt_charges, OUT, label=f"v{i}_{i + 1}"),
            )
        elif i == L - 1:
            indices = (
                TensorIndex(sym, virt_charges, IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
            )
        else:
            indices = (
                TensorIndex(sym, virt_charges, IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
                TensorIndex(sym, virt_charges, OUT, label=f"v{i}_{i + 1}"),
            )
        tensors.append(SymmetricTensor.random_normal(indices, subkey))
    return tensors


class TestFiniteMPSConstruction:
    def test_from_tensors_basic(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4, d=2, chi=3)
        mps = FiniteMPS.from_tensors(tensors)
        assert len(mps) == 4
        assert mps.orth_center is None
        assert mps.singular_values == [None, None, None]

    def test_from_tensors_with_orth_center(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors, orth_center=2)
        assert mps.orth_center == 2

    def test_len(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=6))
        assert mps.L == 6
        assert len(mps) == 6

    def test_getitem(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors)
        assert mps[0] is tensors[0]
        assert mps[3] is tensors[3]

    def test_setitem_invalidates_orth_center(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors, orth_center=2)
        assert mps.orth_center == 2
        mps[1] = tensors[1]  # replace with same tensor
        assert mps.orth_center is None

    def test_iter(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4)
        mps = FiniteMPS.from_tensors(tensors)
        assert list(mps) == tensors


class TestFiniteMPSProperties:
    def test_bond_dims(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.bond_dims == [3, 3, 3]

    def test_phys_dims(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.phys_dims == [2, 2, 2, 2]

    def test_max_bond_dim(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        assert mps.max_bond_dim == 3

    def test_is_symmetric_dense(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4))
        assert mps.is_symmetric is False


class TestFiniteMPSCanonicalize:
    def _check_left_canonical(self, tensor):
        """Check that tensor is left-isometric: A^dag A = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            mat = d.reshape(-1, d.shape[-1])
        else:
            mat = d.reshape(-1, d.shape[-1])
        eye = mat.conj().T @ mat
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def _check_right_canonical(self, tensor):
        """Check that tensor is right-isometric: A A^dag = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            mat = d.reshape(d.shape[0], -1)
        else:
            mat = d.reshape(d.shape[0], -1)
        eye = mat @ mat.conj().T
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def test_right_canonicalize_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_r = mps.right_canonicalize()
        assert mps_r.orth_center == 0
        for i in range(1, 6):
            self._check_right_canonical(mps_r[i])

    def test_left_canonicalize_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_l = mps.left_canonicalize()
        assert mps_l.orth_center == 5
        for i in range(5):
            self._check_left_canonical(mps_l[i])

    def test_canonicalize_center_dense(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=3)
        assert mps_c.orth_center == 3
        for i in range(3):
            self._check_left_canonical(mps_c[i])
        for i in range(4, 6):
            self._check_right_canonical(mps_c[i])

    def test_canonicalize_preserves_state(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4, d=2, chi=3)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=2)

        def _to_statevector(mps_tensors):
            v = mps_tensors[0].todense()
            for t in mps_tensors[1:]:
                v = jnp.tensordot(v, t.todense(), axes=([-1], [0]))
            return v.ravel()

        psi_orig = _to_statevector(mps.tensors)
        psi_canon = _to_statevector(mps_c.tensors)
        psi_orig = psi_orig / jnp.linalg.norm(psi_orig)
        psi_canon = psi_canon / jnp.linalg.norm(psi_canon)
        overlap = jnp.abs(jnp.dot(psi_orig.conj(), psi_canon))
        np.testing.assert_allclose(float(overlap), 1.0, atol=1e-12)

    def test_canonicalize_singular_values(self):
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=3)
        assert mps.singular_values[3] is not None
        assert len(mps.singular_values[3]) > 0
        sv = np.array(mps.singular_values[3])
        assert np.all(sv >= -1e-15)
        np.testing.assert_allclose(sv, np.sort(sv)[::-1], atol=1e-15)


class TestFiniteMPSNormOverlap:
    def test_norm_positive(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        n = mps.norm()
        assert n > 0
        assert isinstance(float(n), float)

    def test_norm_consistent_with_canonicalize(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        mps_c = mps.canonicalize(center=2)
        n_full = mps.norm()
        n_canon = mps_c.norm()
        np.testing.assert_allclose(n_full, n_canon, rtol=1e-10)

    def test_overlap_self(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        ov = mps.overlap(mps)
        n2 = mps.norm() ** 2
        np.testing.assert_allclose(float(jnp.abs(ov)), float(n2), rtol=1e-10)

    def test_overlap_different(self):
        from tenax.core.mps import FiniteMPS

        mps1 = FiniteMPS.from_tensors(
            _make_dense_mps(L=4, d=2, chi=3, key=jax.random.PRNGKey(0))
        )
        mps2 = FiniteMPS.from_tensors(
            _make_dense_mps(L=4, d=2, chi=3, key=jax.random.PRNGKey(99))
        )
        ov = mps1.overlap(mps2)
        normalized = jnp.abs(ov) / (mps1.norm() * mps2.norm())
        assert float(normalized) < 1.0


class TestFiniteMPSEntanglement:
    def test_entanglement_entropy_product_state(self):
        """Product state has zero entanglement entropy."""
        from tenax.core.mps import FiniteMPS

        tensors = []
        for i in range(4):
            data = np.array([1.0, 0.0])  # |up>
            if i == 0:
                indices = (
                    TensorIndex(
                        U1Symmetry(), np.zeros(2, dtype=np.int32), IN, label=f"p{i}"
                    ),
                    TensorIndex(
                        U1Symmetry(),
                        np.zeros(1, dtype=np.int32),
                        OUT,
                        label=f"v{i}_{i + 1}",
                    ),
                )
                data = data.reshape(2, 1)
            elif i == 3:
                indices = (
                    TensorIndex(
                        U1Symmetry(),
                        np.zeros(1, dtype=np.int32),
                        IN,
                        label=f"v{i - 1}_{i}",
                    ),
                    TensorIndex(
                        U1Symmetry(), np.zeros(2, dtype=np.int32), IN, label=f"p{i}"
                    ),
                )
                data = data.reshape(1, 2)
            else:
                indices = (
                    TensorIndex(
                        U1Symmetry(),
                        np.zeros(1, dtype=np.int32),
                        IN,
                        label=f"v{i - 1}_{i}",
                    ),
                    TensorIndex(
                        U1Symmetry(), np.zeros(2, dtype=np.int32), IN, label=f"p{i}"
                    ),
                    TensorIndex(
                        U1Symmetry(),
                        np.zeros(1, dtype=np.int32),
                        OUT,
                        label=f"v{i}_{i + 1}",
                    ),
                )
                data = data.reshape(1, 2, 1)
            tensors.append(DenseTensor(jnp.array(data), indices))

        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=2)
        S = mps.entanglement_entropy(bond=2)
        np.testing.assert_allclose(S, 0.0, atol=1e-12)

    def test_entanglement_entropy_bell_state(self):
        """Bell state |00>+|11> has entropy ln(2)."""
        from tenax.core.mps import FiniteMPS

        A0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (d=2, chi=2)
        A1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (chi=2, d=2)
        sym = U1Symmetry()
        idx0 = (
            TensorIndex(sym, np.zeros(2, dtype=np.int32), IN, label="p0"),
            TensorIndex(sym, np.zeros(2, dtype=np.int32), OUT, label="v0_1"),
        )
        idx1 = (
            TensorIndex(sym, np.zeros(2, dtype=np.int32), IN, label="v0_1"),
            TensorIndex(sym, np.zeros(2, dtype=np.int32), IN, label="p1"),
        )
        tensors = [DenseTensor(A0, idx0), DenseTensor(A1, idx1)]
        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=0)
        S = mps.entanglement_entropy(bond=0)
        np.testing.assert_allclose(S, np.log(2), atol=1e-12)

    def test_entanglement_entropy_uses_cached_svs(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        mps_c = mps.canonicalize(center=2)
        S = mps_c.entanglement_entropy(bond=2)
        assert S >= 0.0


class TestFiniteMPSCanonicalizeSymmetric:
    def test_right_canonicalize_symmetric(self):
        """Right-canonicalize works for SymmetricTensor MPS."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_symmetric_mps(L=4, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_r = mps.right_canonicalize()

        assert mps_r.orth_center == 0
        assert mps_r.is_symmetric  # all tensors remain SymmetricTensor
        for i in range(1, 4):
            d = mps_r[i].todense()
            mat = d.reshape(d.shape[0], -1)
            eye = mat @ mat.conj().T
            np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-10)

    def test_canonicalize_preserves_state_symmetric(self):
        """Canonicalization preserves the physical state for SymmetricTensor."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_symmetric_mps(L=4, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=2)

        def _to_statevector(ts):
            v = ts[0].todense()
            for t in ts[1:]:
                v = jnp.tensordot(v, t.todense(), axes=([-1], [0]))
            return v.ravel()

        psi_orig = _to_statevector(mps.tensors)
        psi_canon = _to_statevector(mps_c.tensors)
        psi_orig = psi_orig / jnp.linalg.norm(psi_orig)
        psi_canon = psi_canon / jnp.linalg.norm(psi_canon)
        overlap = jnp.abs(jnp.dot(psi_orig.conj(), psi_canon))
        np.testing.assert_allclose(float(overlap), 1.0, atol=1e-10)

    def test_no_todense_in_canonicalize(self):
        """Verify that canonicalize does NOT call todense() internally."""
        from unittest.mock import patch

        from tenax.core.mps import FiniteMPS

        tensors = _make_symmetric_mps(L=4, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)

        call_count = [0]
        orig_todense = SymmetricTensor.todense

        def counting_todense(self):
            call_count[0] += 1
            return orig_todense(self)

        with patch.object(SymmetricTensor, "todense", counting_todense):
            mps.canonicalize(center=2)

        assert call_count[0] == 0, (
            f"canonicalize() called todense() {call_count[0]} times; "
            "should use block-sparse operations only"
        )


class TestFiniteMPSRandom:
    def test_random_dense(self):
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(L=6, d=2, chi=4, key=key)
        assert len(mps) == 6
        assert mps.orth_center == 0
        # Boundary bonds are truncated to min(d, chi) during canonicalization
        assert mps.bond_dims == [2, 4, 4, 4, 2]
        assert mps.phys_dims == [2, 2, 2, 2, 2, 2]
        assert mps.is_symmetric is False

    def test_random_symmetric(self):
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(
            L=6,
            d=2,
            chi=4,
            key=key,
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        assert len(mps) == 6
        assert mps.orth_center == 0
        assert mps.is_symmetric is True

    def test_random_reproducible(self):
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(42)
        mps1 = FiniteMPS.random(L=4, d=2, chi=3, key=key)
        mps2 = FiniteMPS.random(L=4, d=2, chi=3, key=key)
        for t1, t2 in zip(mps1, mps2):
            np.testing.assert_allclose(t1.todense(), t2.todense())
