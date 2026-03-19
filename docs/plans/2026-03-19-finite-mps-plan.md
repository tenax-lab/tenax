# FiniteMPS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `FiniteMPS` class with canonical form tracking and block-sparse canonicalization, replacing `list[Tensor]` / `TensorNetwork` for finite MPS.

**Architecture:** `FiniteMPS` is a dataclass in `src/tenax/core/mps.py` wrapping `list[Tensor]` with `orth_center: int | None` and `singular_values: list[jax.Array | None]`. Canonicalization uses `tenax.linalg.qr` (block-sparse for SymmetricTensor) with QR sweeps + SVD at center. Algorithms updated to accept/return `FiniteMPS`.

**Tech Stack:** JAX, tenax.linalg (qr, svd), tenax.core.tensor (DenseTensor, SymmetricTensor, TensorIndex), pytest

**Label conventions:** Physical index = `f"p{i}"`, virtual bonds = `f"v{i-1}_{i}"` (left) and `f"v{i}_{i+1}"` (right). Boundary tensors are 2D (site 0: `(p0, v0_1)`, site L-1: `(v{L-2}_{L-1}, p{L-1})`), middle tensors are 3D `(v{i-1}_{i}, p{i}, v{i}_{i+1})`.

---

### Task 1: Core FiniteMPS dataclass + from_tensors

**Files:**
- Create: `src/tenax/core/mps.py`
- Create: `tests/test_mps.py`

**Step 1: Write failing tests**

```python
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
    tensors = []
    for i in range(L):
        if i == 0:
            shape = (d, chi)
            indices = (
                TensorIndex.dense(d, IN, label=f"p{i}"),
                TensorIndex.dense(chi, OUT, label=f"v{i}_{i+1}"),
            )
        elif i == L - 1:
            shape = (chi, d)
            indices = (
                TensorIndex.dense(chi, IN, label=f"v{i-1}_{i}"),
                TensorIndex.dense(d, IN, label=f"p{i}"),
            )
        else:
            shape = (chi, d, chi)
            indices = (
                TensorIndex.dense(chi, IN, label=f"v{i-1}_{i}"),
                TensorIndex.dense(d, IN, label=f"p{i}"),
                TensorIndex.dense(chi, OUT, label=f"v{i}_{i+1}"),
            )
        key, subkey = jax.random.split(key)
        data = jax.random.normal(subkey, shape)
        tensors.append(DenseTensor(data, indices))
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mps.py -v -x --no-header -m ""`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.core.mps'`

**Step 3: Write minimal implementation**

```python
"""Finite and infinite MPS containers with canonical form tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import jax.numpy as jnp

from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


@dataclass
class FiniteMPS:
    """Finite matrix product state with canonical form tracking.

    Attributes:
        tensors: Site tensors, length L.  Boundary sites are 2-leg
            (site 0: physical x right-bond, site L-1: left-bond x physical),
            bulk sites are 3-leg (left-bond x physical x right-bond).
            Labels follow the convention p{i} for physical, v{i}_{i+1} for bonds.
        orth_center: Position of the orthogonality center, or None if the
            canonical form is unknown.  When set, sites 0..orth_center-1 are
            left-canonical and sites orth_center+1..L-1 are right-canonical.
        singular_values: Singular values at each bond (length L-1).
            Entry i holds the singular values between sites i and i+1,
            or None if not yet computed.
    """

    tensors: list[Tensor]
    orth_center: int | None = None
    singular_values: list[jnp.ndarray | None] = field(default_factory=list)

    def __post_init__(self):
        if not self.singular_values:
            self.singular_values = [None] * max(len(self.tensors) - 1, 0)

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        orth_center: int | None = None,
        singular_values: list[jnp.ndarray | None] | None = None,
    ) -> FiniteMPS:
        """Wrap existing site tensors into a FiniteMPS."""
        L = len(tensors)
        if singular_values is None:
            singular_values = [None] * max(L - 1, 0)
        return FiniteMPS(
            tensors=list(tensors),
            orth_center=orth_center,
            singular_values=singular_values,
        )

    # -- Sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, i: int) -> Tensor:
        return self.tensors[i]

    def __setitem__(self, i: int, tensor: Tensor) -> None:
        self.tensors[i] = tensor
        self.orth_center = None  # invalidate

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.tensors)

    # -- Properties ---------------------------------------------------------

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions between sites (length L-1).

        bond_dims[i] = dimension of the virtual bond between sites i and i+1.
        """
        dims = []
        for i in range(self.L - 1):
            t = self.tensors[i]
            # The rightmost index of site i is the bond to site i+1.
            # For boundary site 0 (2-leg): index 1 is the bond.
            # For bulk sites (3-leg): index 2 is the bond.
            bond_label = f"v{i}_{i+1}"
            for idx in t.indices:
                if idx.label == bond_label:
                    dims.append(idx.dim)
                    break
            else:
                raise ValueError(
                    f"Site {i} has no index with label '{bond_label}'. "
                    f"Labels: {t.labels()}"
                )
        return dims

    @property
    def phys_dims(self) -> list[int]:
        """Physical dimensions at each site (length L)."""
        dims = []
        for i in range(self.L):
            phys_label = f"p{i}"
            for idx in self.tensors[i].indices:
                if idx.label == phys_label:
                    dims.append(idx.dim)
                    break
            else:
                raise ValueError(
                    f"Site {i} has no index with label '{phys_label}'. "
                    f"Labels: {self.tensors[i].labels()}"
                )
        return dims

    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension across all bonds."""
        dims = self.bond_dims
        return max(dims) if dims else 0

    @property
    def is_symmetric(self) -> bool:
        """True if all site tensors are SymmetricTensor."""
        return all(isinstance(t, SymmetricTensor) for t in self.tensors)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mps.py -v --no-header -m ""`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat: add FiniteMPS dataclass with from_tensors and properties"
```

---

### Task 2: FiniteMPS.canonicalize() — QR sweeps + SVD at center

**Files:**
- Modify: `src/tenax/core/mps.py`
- Modify: `tests/test_mps.py`

**Step 1: Write failing tests**

Add to `tests/test_mps.py`:

```python
class TestFiniteMPSCanonicalize:
    def _check_left_canonical(self, tensor):
        """Check that tensor is left-isometric: A^dag A = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            # boundary site 0: (d, chi) -> reshape to (d, chi)
            mat = d.reshape(-1, d.shape[-1])
        else:
            # bulk: (chi_l, d, chi_r) -> reshape to (chi_l * d, chi_r)
            mat = d.reshape(-1, d.shape[-1])
        eye = mat.conj().T @ mat
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def _check_right_canonical(self, tensor):
        """Check that tensor is right-isometric: A A^dag = I on the bond index."""
        d = tensor.todense()
        if d.ndim == 2:
            # boundary site L-1: (chi, d) -> reshape to (chi, d)
            mat = d.reshape(d.shape[0], -1)
        else:
            # bulk: (chi_l, d, chi_r) -> reshape to (chi_l, d * chi_r)
            mat = d.reshape(d.shape[0], -1)
        eye = mat @ mat.conj().T
        np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-12)

    def test_right_canonicalize_dense(self):
        """Right-canonicalize: orth_center=0, sites 1..L-1 are right-canonical."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_r = mps.right_canonicalize()

        assert mps_r.orth_center == 0
        for i in range(1, 6):
            self._check_right_canonical(mps_r[i])

    def test_left_canonicalize_dense(self):
        """Left-canonicalize: orth_center=L-1, sites 0..L-2 are left-canonical."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_l = mps.left_canonicalize()

        assert mps_l.orth_center == 5
        for i in range(5):
            self._check_left_canonical(mps_l[i])

    def test_canonicalize_center_dense(self):
        """Mixed canonical: sites left of center are A-form, right are B-form."""
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
        """Canonicalization preserves the physical state (overlap = 1)."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=4, d=2, chi=3)
        mps = FiniteMPS.from_tensors(tensors)
        mps_c = mps.canonicalize(center=2)

        # Contract full MPS to state vector and compare
        def _to_statevector(mps_tensors):
            v = mps_tensors[0].todense()
            for t in mps_tensors[1:]:
                td = t.todense()
                if td.ndim == 2:
                    # (chi, d) -> contract last of v with first of td
                    v = jnp.tensordot(v, td, axes=([-1], [0]))
                else:
                    v = jnp.tensordot(v, td, axes=([-1], [0]))
            return v.ravel()

        psi_orig = _to_statevector(mps.tensors)
        psi_canon = _to_statevector(mps_c.tensors)

        # Normalize and check overlap
        psi_orig = psi_orig / jnp.linalg.norm(psi_orig)
        psi_canon = psi_canon / jnp.linalg.norm(psi_canon)
        overlap = jnp.abs(jnp.dot(psi_orig.conj(), psi_canon))
        np.testing.assert_allclose(float(overlap), 1.0, atol=1e-12)

    def test_canonicalize_singular_values(self):
        """SVD at center bond populates singular_values."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_dense_mps(L=6, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=3)

        # singular_values at center bond (bond 3, between sites 3 and 4)
        # should be populated; others may or may not be
        assert mps.singular_values[3] is not None
        assert len(mps.singular_values[3]) > 0
        # Should be non-negative and sorted descending
        sv = np.array(mps.singular_values[3])
        assert np.all(sv >= -1e-15)
        np.testing.assert_allclose(sv, np.sort(sv)[::-1], atol=1e-15)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSCanonicalize -v -x --no-header -m ""`
Expected: FAIL — `AttributeError: 'FiniteMPS' has no attribute 'canonicalize'`

**Step 3: Write implementation**

Add to `FiniteMPS` in `src/tenax/core/mps.py`:

```python
from tenax.linalg import qr as _qr, svd as _svd

    # -- Canonicalization ---------------------------------------------------

    def canonicalize(self, center: int) -> FiniteMPS:
        """Return new MPS in mixed canonical form with orthogonality center at *center*.

        QR sweep left-to-right for sites 0..center-1 (making them left-canonical),
        QR sweep right-to-left for sites L-1..center+1 (making them right-canonical),
        then SVD at the center bond to obtain singular values.

        Uses tenax.linalg.qr (block-sparse for SymmetricTensor).
        """
        L = self.L
        if center < 0 or center >= L:
            raise ValueError(f"center={center} out of range [0, {L})")

        new_tensors = [t for t in self.tensors]  # shallow copy
        svs: list[jnp.ndarray | None] = [None] * max(L - 1, 0)

        # Left-to-right QR sweep: sites 0 .. center-1
        for i in range(center):
            site = new_tensors[i]
            bond_label = f"v{i}_{i+1}"
            phys_label = f"p{i}"

            # left_labels = everything except the right bond
            left_labels = [lb for lb in site.labels() if lb != bond_label]
            right_labels = [bond_label]

            q, r = _qr(site, left_labels, right_labels, new_bond_label=bond_label)
            new_tensors[i] = q

            # Absorb R into next site
            from tenax.contraction.contractor import contract
            next_site = new_tensors[i + 1]
            next_bond_label = f"v{i}_{i+1}"
            # R has labels (..., bond_label) and next_site has (bond_label, ...)
            # contract on the shared bond label
            merged = contract(r, next_site)
            new_tensors[i + 1] = merged

        # Right-to-left QR sweep: sites L-1 .. center+1
        for i in range(L - 1, center, -1):
            site = new_tensors[i]
            bond_label = f"v{i-1}_{i}"
            phys_label = f"p{i}"

            # For RQ: "left" = bond to previous site, "right" = everything else
            # We do QR with swapped groups to get RQ effect
            right_labels = [lb for lb in site.labels() if lb != bond_label]
            left_labels = [bond_label]

            q, r = _qr(site, right_labels, left_labels, new_bond_label=bond_label)
            # q has (right_labels..., bond_label) — this is the right-canonical tensor
            # r has (bond_label, left_labels...) = (bond_label, bond_label) — wait, that's wrong
            # We need to think about this differently.
            #
            # For RQ: we want site = L @ Q where Q is right-canonical.
            # tenax.linalg.qr(tensor, left_labels, right_labels) does:
            #   reshape to (left, right), QR -> Q(left, bond) R(bond, right)
            # For RQ, pass left_labels=right_group, right_labels=left_group:
            #   reshape to (right_group, left_group), QR -> Q(right_group, bond) R(bond, left_group)
            #   Q is our right-canonical tensor, R is absorbed left.
            new_tensors[i] = q

            # Absorb R into previous site
            prev_site = new_tensors[i - 1]
            merged = contract(prev_site, r)
            new_tensors[i - 1] = merged

        # SVD at center to get singular values
        if center < L - 1:
            site = new_tensors[center]
            bond_label = f"v{center}_{center+1}"
            left_labels = [lb for lb in site.labels() if lb != bond_label]
            right_labels = [bond_label]

            u, s, vh, s_full = _svd(site, left_labels, right_labels,
                                     new_bond_label=bond_label)
            svs[center] = s_full

            # Recombine: center site = U @ diag(s), next site absorbs Vh
            # Actually, keep center site = U @ diag(s) to preserve norm
            from tenax.contraction.contractor import contract
            # Build diagonal s tensor and contract with U
            # Simpler: absorb s into U
            s_diag = _make_diag_tensor(s, bond_label, u, vh)
            new_tensors[center] = contract(u, s_diag)
            # Absorb Vh into next site
            next_site = new_tensors[center + 1]
            new_tensors[center + 1] = contract(vh, next_site)

        return FiniteMPS(
            tensors=new_tensors,
            orth_center=center,
            singular_values=svs,
        )

    def left_canonicalize(self) -> FiniteMPS:
        """Return new MPS in left-canonical form (orth_center = L-1)."""
        return self.canonicalize(center=self.L - 1)

    def right_canonicalize(self) -> FiniteMPS:
        """Return new MPS in right-canonical form (orth_center = 0)."""
        return self.canonicalize(center=0)
```

Note: The SVD recombination needs a helper to build the diagonal s tensor. The exact implementation will depend on testing — the key contract is:
1. QR sweeps use `tenax.linalg.qr` (block-sparse for SymmetricTensor)
2. SVD at center uses `tenax.linalg.svd`
3. Absorption uses `tenax.contraction.contractor.contract`
4. No `todense()` calls anywhere

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSCanonicalize -v --no-header -m ""`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat: add FiniteMPS.canonicalize() with QR sweeps + SVD"
```

---

### Task 3: FiniteMPS.canonicalize() for SymmetricTensor

**Files:**
- Modify: `tests/test_mps.py`

**Step 1: Write failing tests**

Add to `tests/test_mps.py`:

```python
def _make_symmetric_mps(L=4, d=2, chi=4, key=None, target_charge=0):
    """Helper: build a random U(1)-symmetric MPS as list[SymmetricTensor]."""
    if key is None:
        key = jax.random.PRNGKey(0)
    sym = U1Symmetry()
    phys_charges = np.array([1, -1], dtype=np.int32)  # spin-1/2

    # Virtual charges: distribute across sectors
    virt_charges = np.array(
        [q for q in range(-chi // 2, chi // 2 + 1) for _ in range(1)][:chi],
        dtype=np.int32,
    )

    tensors = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        if i == 0:
            indices = (
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
                TensorIndex(sym, virt_charges, OUT, label=f"v{i}_{i+1}"),
            )
        elif i == L - 1:
            indices = (
                TensorIndex(sym, virt_charges, IN, label=f"v{i-1}_{i}"),
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
            )
        else:
            indices = (
                TensorIndex(sym, virt_charges, IN, label=f"v{i-1}_{i}"),
                TensorIndex(sym, phys_charges, IN, label=f"p{i}"),
                TensorIndex(sym, virt_charges, OUT, label=f"v{i}_{i+1}"),
            )
        tensors.append(SymmetricTensor.random_normal(indices, subkey))
    return tensors


class TestFiniteMPSCanonicalizeSymmetric:
    def test_right_canonicalize_symmetric(self):
        """Right-canonicalize works for SymmetricTensor MPS."""
        from tenax.core.mps import FiniteMPS

        tensors = _make_symmetric_mps(L=4, d=2, chi=4)
        mps = FiniteMPS.from_tensors(tensors)
        mps_r = mps.right_canonicalize()

        assert mps_r.orth_center == 0
        assert mps_r.is_symmetric  # all tensors remain SymmetricTensor
        # Check right-canonical form via dense
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSCanonicalizeSymmetric -v -x --no-header -m ""`
Expected: Should PASS if Task 2 implementation correctly uses `tenax.linalg.qr` (which dispatches to block-sparse). If any test fails, fix the implementation.

**Step 3: Commit**

```bash
git add tests/test_mps.py
git commit -m "test: add SymmetricTensor canonicalization tests for FiniteMPS"
```

---

### Task 4: FiniteMPS.norm() and FiniteMPS.overlap()

**Files:**
- Modify: `src/tenax/core/mps.py`
- Modify: `tests/test_mps.py`

**Step 1: Write failing tests**

Add to `tests/test_mps.py`:

```python
class TestFiniteMPSNormOverlap:
    def test_norm_uncanonicalized(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        n = mps.norm()
        assert n > 0
        assert isinstance(float(n), float)

    def test_norm_after_canonicalize(self):
        """After canonicalization, norm should be computable from center site."""
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        mps_c = mps.canonicalize(center=2)
        # Norm from full contraction vs. from center site only
        n_full = mps.norm()
        n_canon = mps_c.norm()
        np.testing.assert_allclose(n_full, n_canon, rtol=1e-10)

    def test_overlap_self(self):
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        ov = mps.overlap(mps)
        n2 = mps.norm() ** 2
        np.testing.assert_allclose(float(jnp.abs(ov)), float(n2), rtol=1e-10)

    def test_overlap_orthogonal(self):
        """Two random MPS should have small overlap relative to their norms."""
        from tenax.core.mps import FiniteMPS

        mps1 = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3, key=jax.random.PRNGKey(0)))
        mps2 = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3, key=jax.random.PRNGKey(99)))
        ov = mps1.overlap(mps2)
        # Not exactly zero, but normalized overlap should be < 1
        normalized = jnp.abs(ov) / (mps1.norm() * mps2.norm())
        assert float(normalized) < 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSNormOverlap -v -x --no-header -m ""`
Expected: FAIL — `AttributeError: 'FiniteMPS' has no attribute 'norm'`

**Step 3: Write implementation**

Add to `FiniteMPS` in `src/tenax/core/mps.py`:

```python
    def norm(self) -> float:
        """Frobenius norm ||psi|| via transfer matrix contraction."""
        from tenax.contraction.contractor import contract

        # Build transfer matrix left to right: env = <psi|psi>
        env = None
        for i, t in enumerate(self.tensors):
            t_conj = t.conj()
            # Relabel conj to avoid label clash
            relabeled = t_conj
            for lb in t_conj.labels():
                if lb.startswith("p"):
                    pass  # physical labels contract with each other
                else:
                    relabeled = relabeled.relabel(lb, lb + "*")

            if env is None:
                env = contract(t, relabeled)
            else:
                env = contract(env, t)
                env = contract(env, relabeled)

        # env should be a scalar
        val = env.todense()
        return float(jnp.sqrt(jnp.abs(val.ravel()[0])))

    def overlap(self, other: FiniteMPS) -> complex:
        """Compute <self|other> via transfer matrix contraction."""
        ...  # similar to norm but with two different MPS
```

Note: The exact implementation of norm/overlap depends on the label contraction mechanics. The core idea is transfer-matrix contraction from left to right. The implementation may need to relabel one MPS to avoid label collisions with the other during contraction. Detailed implementation to be refined during TDD.

**Step 4: Run tests**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSNormOverlap -v --no-header -m ""`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat: add FiniteMPS.norm() and overlap() via transfer matrix"
```

---

### Task 5: FiniteMPS.entanglement_entropy()

**Files:**
- Modify: `src/tenax/core/mps.py`
- Modify: `tests/test_mps.py`

**Step 1: Write failing tests**

```python
class TestFiniteMPSEntanglement:
    def test_entanglement_entropy_product_state(self):
        """Product state has zero entanglement entropy."""
        from tenax.core.mps import FiniteMPS

        # Build a product state: |up up up up>
        tensors = []
        for i in range(4):
            data = np.array([1.0, 0.0])  # |up>
            if i == 0:
                indices = (
                    TensorIndex.dense(2, IN, label=f"p{i}"),
                    TensorIndex.dense(1, OUT, label=f"v{i}_{i+1}"),
                )
                data = data.reshape(2, 1)
            elif i == 3:
                indices = (
                    TensorIndex.dense(1, IN, label=f"v{i-1}_{i}"),
                    TensorIndex.dense(2, IN, label=f"p{i}"),
                )
                data = data.reshape(1, 2)
            else:
                indices = (
                    TensorIndex.dense(1, IN, label=f"v{i-1}_{i}"),
                    TensorIndex.dense(2, IN, label=f"p{i}"),
                    TensorIndex.dense(1, OUT, label=f"v{i}_{i+1}"),
                )
                data = data.reshape(1, 2, 1)
            tensors.append(DenseTensor(jnp.array(data), indices))

        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=2)
        S = mps.entanglement_entropy(bond=2)
        np.testing.assert_allclose(S, 0.0, atol=1e-12)

    def test_entanglement_entropy_bell_state(self):
        """Bell state |00>+|11> has entropy ln(2)."""
        from tenax.core.mps import FiniteMPS

        # |00> + |11> as MPS: A[0] = [[1,0]], A[1] = [[1],[0]] for |0>
        #                      A[0] = [[0,1]], A[1] = [[0],[1]] for |1>
        # Site 0: (d=2, chi=2), Site 1: (chi=2, d=2)
        A0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (d, chi)
        A1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (chi, d)
        idx0 = (
            TensorIndex.dense(2, IN, label="p0"),
            TensorIndex.dense(2, OUT, label="v0_1"),
        )
        idx1 = (
            TensorIndex.dense(2, IN, label="v0_1"),
            TensorIndex.dense(2, IN, label="p1"),
        )
        tensors = [DenseTensor(A0, idx0), DenseTensor(A1, idx1)]
        mps = FiniteMPS.from_tensors(tensors).canonicalize(center=0)
        S = mps.entanglement_entropy(bond=0)
        np.testing.assert_allclose(S, np.log(2), atol=1e-12)

    def test_entanglement_entropy_uses_cached_svs(self):
        """If singular_values are already cached, don't recompute."""
        from tenax.core.mps import FiniteMPS

        mps = FiniteMPS.from_tensors(_make_dense_mps(L=4, d=2, chi=3))
        mps_c = mps.canonicalize(center=2)
        # Bond 2 has cached SVs from canonicalize
        S = mps_c.entanglement_entropy(bond=2)
        assert S >= 0.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSEntanglement -v -x --no-header -m ""`
Expected: FAIL

**Step 3: Write implementation**

```python
    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entanglement entropy at the given bond.

        Uses cached singular_values if available at this bond.
        Otherwise, canonicalizes to this bond and computes SVD.
        """
        if bond < 0 or bond >= self.L - 1:
            raise ValueError(f"bond={bond} out of range [0, {self.L - 1})")

        sv = self.singular_values[bond]
        if sv is None:
            # Need to canonicalize to get SVs at this bond
            mps_c = self.canonicalize(center=bond)
            sv = mps_c.singular_values[bond]

        # Von Neumann entropy: S = -sum(p * log(p)) where p = sv^2 / sum(sv^2)
        sv = jnp.array(sv)
        sv = sv[sv > 1e-15]  # filter near-zero
        p = sv ** 2
        p = p / jnp.sum(p)
        return float(-jnp.sum(p * jnp.log(p)))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSEntanglement -v --no-header -m ""`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat: add FiniteMPS.entanglement_entropy()"
```

---

### Task 6: FiniteMPS.random() — replace build_random_*_mps

**Files:**
- Modify: `src/tenax/core/mps.py`
- Modify: `tests/test_mps.py`

**Step 1: Write failing tests**

```python
class TestFiniteMPSRandom:
    def test_random_dense(self):
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(L=6, d=2, chi=4, key=key)
        assert len(mps) == 6
        assert mps.orth_center == 0  # right-canonicalized
        assert mps.bond_dims == [4, 4, 4, 4, 4]
        assert mps.phys_dims == [2, 2, 2, 2, 2, 2]
        assert mps.is_symmetric is False

    def test_random_symmetric(self):
        from tenax.core.mps import FiniteMPS

        key = jax.random.PRNGKey(42)
        mps = FiniteMPS.random(
            L=6, d=2, chi=4, key=key,
            symmetric=True, symmetry=U1Symmetry(), target_charge=0,
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSRandom -v -x --no-header -m ""`
Expected: FAIL — `AttributeError: type object 'FiniteMPS' has no attribute 'random'`

**Step 3: Write implementation**

The implementation should mirror the existing `build_random_symmetric_mps()` and `build_random_mps()` logic, but return a `FiniteMPS` that is right-canonicalized. Reuse the tensor construction from the existing functions, then call `self.right_canonicalize()`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_mps.py::TestFiniteMPSRandom -v --no-header -m ""`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tenax/core/mps.py tests/test_mps.py
git commit -m "feat: add FiniteMPS.random() for dense and symmetric MPS"
```

---

### Task 7: Export FiniteMPS and add to conftest markers

**Files:**
- Modify: `src/tenax/__init__.py` — add `FiniteMPS` to imports and `__all__`
- Modify: `tests/conftest.py` — add `"test_mps.py": "core"` to `_FILE_MARKERS`

**Step 1: Update exports**

Add to `src/tenax/__init__.py`:
```python
from tenax.core.mps import FiniteMPS
```

Add `"FiniteMPS"` to `__all__` under a new `# MPS` section.

**Step 2: Update test markers**

Add to `_FILE_MARKERS` in `tests/conftest.py`:
```python
"test_mps.py": "core",
```

**Step 3: Verify**

Run: `uv run pytest tests/test_mps.py -v --no-header -m core`
Expected: All tests collected and passing under core marker.

Run: `uv run python -c "from tenax import FiniteMPS; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/tenax/__init__.py tests/conftest.py
git commit -m "feat: export FiniteMPS from tenax and wire test marker"
```

---

### Task 8: Run full core test suite

**Step 1: Run core tests**

Run: `uv run pytest -m core -v --no-header`
Expected: All PASS — no regressions.

**Step 2: Run algorithm tests**

Run: `uv run pytest -m algorithm -v --no-header`
Expected: All PASS — nothing changed in algorithm code yet.

**Step 3: Commit (if any fixes needed)**

---

Plan complete and saved to `docs/plans/2026-03-19-finite-mps-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?