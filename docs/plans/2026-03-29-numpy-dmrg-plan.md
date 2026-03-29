# NumPy-Only DMRG/iDMRG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pure-numpy fast path for symmetric DMRG/iDMRG that eliminates all JAX overhead (JIT compilation, dispatch, pytree management), reducing 38.9s → ~8-12s at chi=32.

**Architecture:** A `BlockArray` dataclass holds numpy block dicts + index metadata. Parallel `_np` versions of SVD, QR, Lanczos, and DMRG update functions operate on BlockArray. Existing JAX functions stay as reference. A `numpy_blockwise` flag on DMRGConfig/iDMRGConfig selects the path. `_blockwise_contract` gains a mode that returns BlockArray directly (skipping `_init_flat_buffer`).

**Tech Stack:** numpy, scipy (np.linalg.svd/qr/eigh), existing opt_einsum + BLAS plan infrastructure

**Design doc:** `docs/plans/2026-03-29-numpy-dmrg-design.md`

---

### Task 1: BlockArray Data Type and Arithmetic

Add the lightweight numpy-backed block-sparse container and free functions for Lanczos arithmetic.

**Files:**
- Create: `src/tenax/algorithms/_block_array.py`
- Test: `tests/test_block_array.py`

**Step 1: Write the failing test**

Create `tests/test_block_array.py`:

```python
"""Tests for BlockArray numpy block-sparse arithmetic."""

import numpy as np
import pytest

from tenax.algorithms._block_array import (
    BlockArray,
    ba_add,
    ba_conj,
    ba_inner,
    ba_norm,
    ba_scale,
    ba_to_symmetric,
    symmetric_to_ba,
)


class TestBlockArrayArithmetic:
    def _make_ba(self, rng, keys=((0,), (1,), (2,)), shape=(4, 4)):
        blocks = {k: rng.standard_normal(shape) for k in keys}
        return BlockArray(blocks=blocks, indices=())

    def test_scale(self):
        rng = np.random.default_rng(42)
        ba = self._make_ba(rng)
        result = ba_scale(ba, 2.5)
        for k in ba.blocks:
            np.testing.assert_allclose(result.blocks[k], ba.blocks[k] * 2.5)

    def test_add(self):
        rng = np.random.default_rng(42)
        a = self._make_ba(rng)
        b = self._make_ba(rng)
        result = ba_add(a, b)
        for k in a.blocks:
            np.testing.assert_allclose(result.blocks[k], a.blocks[k] + b.blocks[k])

    def test_add_different_keys(self):
        """Add with non-overlapping keys produces union."""
        a = BlockArray({(0,): np.ones((2, 2))}, indices=())
        b = BlockArray({(1,): np.ones((2, 2)) * 2}, indices=())
        result = ba_add(a, b)
        assert (0,) in result.blocks and (1,) in result.blocks
        np.testing.assert_allclose(result.blocks[(0,)], 1.0)
        np.testing.assert_allclose(result.blocks[(1,)], 2.0)

    def test_inner(self):
        rng = np.random.default_rng(7)
        ba = self._make_ba(rng, shape=(3, 5))
        # inner(a, a) = sum of squares
        expected = sum(np.sum(v ** 2) for v in ba.blocks.values())
        np.testing.assert_allclose(ba_inner(ba, ba), expected)

    def test_norm(self):
        rng = np.random.default_rng(7)
        ba = self._make_ba(rng)
        expected = np.sqrt(ba_inner(ba, ba))
        np.testing.assert_allclose(ba_norm(ba), expected)

    def test_conj_real(self):
        rng = np.random.default_rng(42)
        ba = self._make_ba(rng)
        result = ba_conj(ba)
        for k in ba.blocks:
            np.testing.assert_allclose(result.blocks[k], ba.blocks[k])


class TestBlockArrayConversion:
    def test_roundtrip(self):
        """SymmetricTensor -> BlockArray -> SymmetricTensor preserves data."""
        from tenax import U1Symmetry
        from tenax.core.index import TensorIndex
        from tenax.core.tensor import SymmetricTensor

        sym = U1Symmetry()
        idx_a = TensorIndex(charges=np.array([0, 1]), dim_per_charge=np.array([2, 3]), flow_direction=1, label="a", symmetry=sym)
        idx_b = TensorIndex(charges=np.array([0, 1]), dim_per_charge=np.array([4, 2]), flow_direction=-1, label="b", symmetry=sym)

        blocks = {
            (0, 0): np.random.randn(2, 4),
            (1, 1): np.random.randn(3, 2),
        }
        t = SymmetricTensor(blocks, (idx_a, idx_b))

        ba = symmetric_to_ba(t)
        assert isinstance(ba, BlockArray)
        assert len(ba.blocks) == 2

        t2 = ba_to_symmetric(ba)
        assert isinstance(t2, SymmetricTensor)

        for k in blocks:
            np.testing.assert_allclose(
                np.asarray(t2.blocks[k]), np.asarray(t.blocks[k]), rtol=1e-14
            )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_block_array.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `_block_array.py`**

Create `src/tenax/algorithms/_block_array.py`:

```python
"""Lightweight numpy-backed block-sparse array for DMRG hot loops.

Avoids SymmetricTensor / JAX overhead. Used only inside the DMRG
sweep; converted to/from SymmetricTensor at sweep boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tenax.core.index import TensorIndex


@dataclass
class BlockArray:
    """Numpy block-sparse array with index metadata."""

    blocks: dict[tuple[int, ...], np.ndarray]
    indices: tuple[TensorIndex, ...]


def ba_scale(ba: BlockArray, scalar: float) -> BlockArray:
    """Multiply all blocks by a scalar."""
    return BlockArray(
        blocks={k: v * scalar for k, v in ba.blocks.items()},
        indices=ba.indices,
    )


def ba_add(a: BlockArray, b: BlockArray) -> BlockArray:
    """Add two BlockArrays (union of keys)."""
    result = dict(a.blocks)
    for k, v in b.blocks.items():
        if k in result:
            result[k] = result[k] + v
        else:
            result[k] = v.copy()
    return BlockArray(blocks=result, indices=a.indices)


def ba_sub(a: BlockArray, b: BlockArray) -> BlockArray:
    """Subtract b from a."""
    result = dict(a.blocks)
    for k, v in b.blocks.items():
        if k in result:
            result[k] = result[k] - v
        else:
            result[k] = -v
    return BlockArray(blocks=result, indices=a.indices)


def ba_inner(a: BlockArray, b: BlockArray) -> float:
    """Frobenius inner product: sum_k tr(a_k^H b_k)."""
    total = 0.0
    for k in a.blocks:
        if k in b.blocks:
            total += np.sum(np.conj(a.blocks[k]) * b.blocks[k]).real
    return float(total)


def ba_norm(ba: BlockArray) -> float:
    """Frobenius norm."""
    return float(np.sqrt(ba_inner(ba, ba)))


def ba_conj(ba: BlockArray) -> BlockArray:
    """Element-wise conjugation."""
    return BlockArray(
        blocks={k: np.conj(v) for k, v in ba.blocks.items()},
        indices=ba.indices,
    )


def symmetric_to_ba(t) -> BlockArray:
    """Convert a SymmetricTensor to BlockArray (numpy blocks)."""
    blocks = {k: np.asarray(v) for k, v in t.blocks.items()}
    return BlockArray(blocks=blocks, indices=t.indices)


def ba_to_symmetric(ba: BlockArray):
    """Convert a BlockArray back to SymmetricTensor."""
    from tenax.core.tensor import SymmetricTensor

    obj = object.__new__(SymmetricTensor)
    obj._indices = ba.indices
    obj._init_flat_buffer(ba.blocks)
    return obj
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_block_array.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_block_array.py tests/test_block_array.py
git commit -m "feat: add BlockArray numpy container for JAX-free DMRG"
```

---

### Task 2: _blockwise_contract returning BlockArray

Modify `_blockwise_contract` to optionally skip `_init_flat_buffer` and return a `BlockArray` directly. This avoids the JAX array creation in the Lanczos inner loop.

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py` (lines 1277-1417)
- Test: existing symmetric DMRG tests (regression)

**Step 1: Add `return_ba` parameter**

In `_blockwise_contract`, add parameter `return_ba: bool = False`. When `True`, return a `BlockArray` instead of `SymmetricTensor`. The change is at the end of the function (after the accumulation loop).

Replace the output construction block (lines ~1404-1417):

```python
    # Sum accumulated contributions per output key
    output_blocks: dict[tuple[int, ...], np.ndarray] = {}
    for key, arrays in output_accum.items():
        total = arrays[0]
        for a in arrays[1:]:
            total = total + a
        output_blocks[key] = total

    if return_ba:
        from tenax.algorithms._block_array import BlockArray
        return BlockArray(blocks=output_blocks, indices=output_indices)

    # Build SymmetricTensor result (original path)
    obj = object.__new__(SymmetricTensor)
    obj._indices = output_indices
    obj._init_flat_buffer(output_blocks)
    return obj
```

Also update the function signature and type hint to `-> SymmetricTensor | BlockArray`.

**Step 2: Run existing tests to verify no regression**

Run: `uv run pytest tests/test_dmrg.py -k symmetric -q`
Expected: All pass (default `return_ba=False` is unchanged behavior).

**Step 3: Commit**

```bash
git add src/tenax/algorithms/dmrg.py
git commit -m "feat: add return_ba option to _blockwise_contract"
```

---

### Task 3: NumPy Symmetric SVD

Add `_truncated_svd_symmetric_np` to `linalg.py`. Same algorithm as
`_truncated_svd_symmetric` (lines 91-358) but with `np.linalg.svd` and
numpy indexing instead of JAX.

**Files:**
- Modify: `src/tenax/linalg.py` (add function after line 358)
- Test: `tests/test_linalg_np.py`

**Step 1: Write the failing test**

Create `tests/test_linalg_np.py`:

```python
"""Tests for numpy symmetric linalg functions."""

import numpy as np
import pytest

from tenax import U1Symmetry
from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor


def _make_symmetric_matrix(sym, chi=8):
    """Create a symmetric 2-leg SymmetricTensor for SVD testing."""
    idx_l = TensorIndex(
        charges=np.array([0, 1, -1]),
        dim_per_charge=np.array([3, 3, 2]),
        flow_direction=1, label="l", symmetry=sym,
    )
    idx_r = TensorIndex(
        charges=np.array([0, 1, -1]),
        dim_per_charge=np.array([3, 3, 2]),
        flow_direction=-1, label="r", symmetry=sym,
    )
    rng = np.random.default_rng(42)
    blocks = {
        (0, 0): rng.standard_normal((3, 3)),
        (1, 1): rng.standard_normal((3, 3)),
        (-1, -1): rng.standard_normal((2, 2)),
    }
    return SymmetricTensor(blocks, (idx_l, idx_r))


class TestTruncatedSvdSymmetricNp:
    def test_matches_jax_version(self):
        """NumPy SVD gives same singular values as JAX version."""
        from tenax.linalg import _truncated_svd_symmetric, _truncated_svd_symmetric_np

        sym = U1Symmetry()
        t = _make_symmetric_matrix(sym)

        # JAX version
        U_jax, s_jax, Vh_jax, sf_jax = _truncated_svd_symmetric(
            t, ["l"], ["r"], None, None, "bond", False
        )

        # NumPy version
        U_np, s_np, Vh_np, sf_np = _truncated_svd_symmetric_np(
            t, ["l"], ["r"], None, None, "bond", False
        )

        np.testing.assert_allclose(np.sort(np.asarray(s_jax))[::-1],
                                   np.sort(s_np)[::-1], rtol=1e-12)
        np.testing.assert_allclose(np.sort(np.asarray(sf_jax))[::-1],
                                   np.sort(sf_np)[::-1], rtol=1e-12)

    def test_truncation(self):
        """NumPy SVD respects max_singular_values."""
        from tenax.linalg import _truncated_svd_symmetric_np

        sym = U1Symmetry()
        t = _make_symmetric_matrix(sym)

        U, s, Vh, s_full = _truncated_svd_symmetric_np(
            t, ["l"], ["r"], max_singular_values=4,
            max_truncation_err=None, new_bond_label="bond", normalize=False,
        )

        assert len(s) == 4
        assert len(s_full) == 8  # 3+3+2

    def test_reconstruction(self):
        """U @ diag(s) @ Vh ≈ original tensor."""
        from tenax.algorithms._block_array import ba_to_symmetric
        from tenax.linalg import _truncated_svd_symmetric_np

        sym = U1Symmetry()
        t = _make_symmetric_matrix(sym)

        U_ba, s, Vh_ba, _ = _truncated_svd_symmetric_np(
            t, ["l"], ["r"], None, None, "bond", False,
        )

        # Reconstruct: contract U * diag(s) * Vh
        U_t = ba_to_symmetric(U_ba)
        Vh_t = ba_to_symmetric(Vh_ba)
        from tenax.contraction import contract
        from tenax.linalg import scale_bond_axis
        Us = scale_bond_axis(U_t, "bond", s)
        recon = contract(Us, Vh_t)

        np.testing.assert_allclose(
            np.asarray(recon.todense()), np.asarray(t.todense()), atol=1e-12
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_linalg_np.py -v`
Expected: FAIL with `ImportError: cannot import name '_truncated_svd_symmetric_np'`

**Step 3: Implement `_truncated_svd_symmetric_np`**

Add to `src/tenax/linalg.py` after `_truncated_svd_symmetric` (after line 358).

The function follows the same algorithm as `_truncated_svd_symmetric` but replaces:
- `jnp.zeros(...)` → `np.zeros(...)`
- `matrix = matrix.at[r0:r1, c0:c1].set(block)` → `matrix[r0:r1, c0:c1] = block`
- `jnp.linalg.svd(...)` → `np.linalg.svd(...)`
- `jnp.array(...)` → `np.array(...)`
- Returns `BlockArray` instead of `SymmetricTensor` for U and Vh
- Returns `np.ndarray` instead of `jax.Array` for s and s_full

```python
def _truncated_svd_symmetric_np(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
) -> tuple:
    """NumPy version of _truncated_svd_symmetric.

    Same algorithm, same output semantics, but uses np.linalg.svd
    and returns (BlockArray, np.ndarray, BlockArray, np.ndarray)
    instead of (SymmetricTensor, jax.Array, SymmetricTensor, jax.Array).
    """
    from tenax.algorithms._block_array import BlockArray
    import numpy as np

    # --- Copy the logic from _truncated_svd_symmetric (lines 108-358),
    # replacing every jnp.* call with np.* and .at[].set() with direct
    # indexing. Return BlockArray for U/Vh and np.ndarray for s/s_full.
    # The template is _truncated_svd_symmetric — follow it line-by-line.
    ...
```

**Implementation note to engineer:** Copy `_truncated_svd_symmetric` (lines 91-358), then:
1. Replace `jnp.zeros` → `np.zeros` (lines 179, and similar)
2. Replace `matrix.at[r0:r1, c0:c1].set(flat_block)` → `matrix[r0:r1, c0:c1] = flat_block`
3. Replace `jnp.linalg.svd` → `np.linalg.svd` (line 206)
4. Replace `jnp.array` → `np.array` for s_full (line 239) and s_final (line 268)
5. At the end, instead of creating SymmetricTensor via `_init_flat_buffer`, create `BlockArray(blocks=U_blocks, indices=U_indices)` and `BlockArray(blocks=Vh_blocks, indices=Vh_indices)`
6. Return `(U_ba, s_final, Vh_ba, s_full)` where s_final and s_full are `np.ndarray`

**Step 4: Run tests**

Run: `uv run pytest tests/test_linalg_np.py -v`
Expected: All 3 tests pass.

**Step 5: Commit**

```bash
git add src/tenax/linalg.py tests/test_linalg_np.py
git commit -m "feat: add numpy symmetric SVD (_truncated_svd_symmetric_np)"
```

---

### Task 4: NumPy Symmetric QR

Add `_qr_symmetric_np` to `linalg.py`. Same pattern as Task 3.

**Files:**
- Modify: `src/tenax/linalg.py`
- Test: `tests/test_linalg_np.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_linalg_np.py`:

```python
class TestQrSymmetricNp:
    def test_matches_jax_version(self):
        """NumPy QR gives same R diagonal signs as JAX version."""
        from tenax.linalg import _qr_symmetric, _qr_symmetric_np

        sym = U1Symmetry()
        t = _make_symmetric_matrix(sym)

        Q_jax, R_jax = _qr_symmetric(t, ["l"], ["r"], "bond")
        Q_np, R_np = _qr_symmetric_np(t, ["l"], ["r"], "bond")

        # Q @ R should reconstruct the same tensor
        from tenax.algorithms._block_array import ba_to_symmetric
        from tenax.contraction import contract

        recon_jax = contract(Q_jax, R_jax)
        recon_np = contract(ba_to_symmetric(Q_np), ba_to_symmetric(R_np))

        np.testing.assert_allclose(
            np.asarray(recon_np.todense()),
            np.asarray(recon_jax.todense()),
            atol=1e-12,
        )
```

**Step 2: Implement `_qr_symmetric_np`**

Copy `_qr_symmetric` (lines 364-548), apply same jnp→np substitutions as Task 3.
Return `(Q_ba: BlockArray, R_ba: BlockArray)`.

**Step 3: Run tests, commit**

```bash
git commit -m "feat: add numpy symmetric QR (_qr_symmetric_np)"
```

---

### Task 5: NumPy Lanczos Solver

Add `_lanczos_solve_np` that operates on `BlockArray` using `ba_*` arithmetic and `np.linalg.eigh` for the tridiagonal matrix.

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
- Test: `tests/test_lanczos_np.py`

**Step 1: Write the failing test**

Create `tests/test_lanczos_np.py`:

```python
"""Test numpy Lanczos solver against JAX version."""

import numpy as np
import pytest

from tenax import U1Symmetry, build_mpo_heisenberg
from tenax.algorithms._block_array import symmetric_to_ba, ba_to_symmetric


class TestLanczosNp:
    def test_matches_jax_lanczos(self):
        """NumPy Lanczos finds same ground state energy as JAX version."""
        from tenax.algorithms.dmrg import (
            _lanczos_solve_np,
            _lanczos_solve_tensor,
            _precompute_block_plan,
            _to_np_blocks,
            _blockwise_contract,
        )
        from tenax.contraction import contract

        sym = U1Symmetry()
        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0)

        # Build a simple 2-site theta and environments for testing
        from tenax import FiniteMPS

        mps = FiniteMPS.random(L, d=2, chi=4, symmetry=sym, total_charge=0, seed=0)
        mps_tensors = list(mps.tensors)
        mpo_tensors = list(mpo.tensors)

        # Build trivial left/right envs
        from tenax.algorithms.dmrg import (
            _build_trivial_left_env_symmetric,
            _build_trivial_right_env_symmetric,
            _update_left_env_symmetric,
            _update_right_env_symmetric,
        )

        left_env = _build_trivial_left_env_symmetric()
        right_env = _build_trivial_right_env_symmetric()

        # Build right envs from right
        right_envs = [None] * L
        right_envs[L - 1] = right_env
        for i in range(L - 2, 0, -1):
            right_envs[i] = _update_right_env_symmetric(
                right_envs[i + 1], mps_tensors[i + 1], mpo_tensors[i + 1]
            )

        # 2-site theta
        theta = contract(mps_tensors[0], mps_tensors[1])
        left = left_env
        right = right_envs[1]
        mpo_l, mpo_r = mpo_tensors[0], mpo_tensors[1]

        # JAX Lanczos
        subs = "abc,apqd,bpse,eqtf,dfg->cstg"
        plan = _precompute_block_plan(
            [left, theta, mpo_l, mpo_r, right], subs
        )
        env_np = [_to_np_blocks(left), None, _to_np_blocks(mpo_l),
                  _to_np_blocks(mpo_r), _to_np_blocks(right)]
        cache_jax = {}

        def matvec_jax(v):
            return _blockwise_contract(
                [left, v, mpo_l, mpo_r, right], subs,
                output_indices=v.indices, expr_cache=cache_jax,
                block_plan=plan, np_blocks_cache=env_np,
            )

        e_jax, _ = _lanczos_solve_tensor(matvec_jax, theta, 20, 1e-12)

        # NumPy Lanczos
        theta_ba = symmetric_to_ba(theta)
        cache_np = {}

        def matvec_np(v_ba):
            v_sym = ba_to_symmetric(v_ba)
            result = _blockwise_contract(
                [left, v_sym, mpo_l, mpo_r, right], subs,
                output_indices=v_sym.indices, expr_cache=cache_np,
                block_plan=plan, np_blocks_cache=env_np,
                return_ba=True,
            )
            return result

        e_np, _ = _lanczos_solve_np(matvec_np, theta_ba, 20, 1e-12)

        np.testing.assert_allclose(e_np, e_jax, rtol=1e-10)
```

**Step 2: Implement `_lanczos_solve_np`**

Add to `dmrg.py` (after `_lanczos_solve_tensor`):

```python
def _lanczos_solve_np(
    matvec: Callable,
    initial: BlockArray,
    num_steps: int,
    tol: float,
) -> tuple[float, BlockArray]:
    """Lanczos eigensolver on BlockArray — pure numpy, no JAX."""
    from tenax.algorithms._block_array import (
        ba_add, ba_inner, ba_norm, ba_scale, ba_sub,
    )

    v_nrm = ba_norm(initial)
    v = ba_scale(initial, 1.0 / (v_nrm + 1e-15))

    basis: list[BlockArray] = [v]
    alphas: list[float] = []
    betas: list[float] = [0.0]

    for step in range(num_steps):
        w = matvec(basis[-1])
        alpha_val = ba_inner(basis[-1], w)
        alphas.append(alpha_val)

        w = ba_sub(w, ba_scale(basis[-1], alpha_val))
        if step > 0:
            w = ba_sub(w, ba_scale(basis[-2], betas[-1]))

        # Full reorthogonalization
        for q in basis:
            w = ba_sub(w, ba_scale(q, ba_inner(q, w)))

        beta_val = ba_norm(w)
        betas.append(beta_val)

        if beta_val < tol:
            break

        basis.append(ba_scale(w, 1.0 / beta_val))

    n = len(alphas)
    if n == 0:
        return 0.0, v
    if n == 1:
        return alphas[0], basis[0]

    # Tridiagonal eigendecomposition — pure numpy
    T = np.diag(alphas) + np.diag(betas[1:n], k=1) + np.diag(betas[1:n], k=-1)
    eigvals, eigvecs = np.linalg.eigh(T)
    idx = int(np.argmin(eigvals))
    eigenvalue = float(eigvals[idx])
    krylov_coefs = eigvecs[:, idx]

    # Reconstruct eigenvector
    eigenvector = ba_scale(basis[0], float(krylov_coefs[0]))
    for k in range(1, n):
        eigenvector = ba_add(eigenvector, ba_scale(basis[k], float(krylov_coefs[k])))

    ev_norm = ba_norm(eigenvector)
    eigenvector = ba_scale(eigenvector, 1.0 / (ev_norm + 1e-15))

    return eigenvalue, eigenvector
```

**Step 3: Run tests, commit**

```bash
git commit -m "feat: add numpy Lanczos solver (_lanczos_solve_np)"
```

---

### Task 6: NumPy DMRG Update Functions

Add `_two_site_update_symmetric_np`, `_one_site_update_symmetric_np`, and
`_svd_and_truncate_site_np`. These wire together the numpy Lanczos + numpy SVD.

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
- Test: existing `tests/test_dmrg.py` (parametrize in Task 8)

**Step 1: Add `_svd_and_truncate_site_np`**

Add after `_svd_and_truncate_site`:

```python
def _svd_and_truncate_site_np(
    theta_ba: BlockArray,
    site: int,
    config: DMRGConfig,
    sweep_right: bool = True,
) -> tuple[BlockArray, np.ndarray, BlockArray, float]:
    """NumPy version of _svd_and_truncate_site operating on BlockArray."""
    from tenax.algorithms._block_array import ba_to_symmetric
    from tenax.linalg import _truncated_svd_symmetric_np

    labels = [idx.label for idx in theta_ba.indices]

    # Determine left/right label split (same logic as _svd_and_truncate_site)
    if site > 0:
        left_virt = f"v{site - 1}_{site}"
    else:
        left_virt = "v_-1_0"
    right_virt = f"v{site + 1}_{site + 2}"
    left_phys = f"p{site}"
    right_phys = f"p{site + 1}"

    left_candidates = {left_virt, left_phys}
    right_candidates = {right_virt, right_phys}
    left_labels = [lbl for lbl in labels if lbl in left_candidates]
    right_labels = [lbl for lbl in labels if lbl in right_candidates]

    if not left_labels or not right_labels:
        n = len(labels)
        left_labels = list(labels[: n // 2])
        right_labels = list(labels[n // 2 :])

    bond_label = f"v{site}_{site + 1}"

    # Convert to SymmetricTensor for SVD (SVD needs full index metadata)
    theta_sym = ba_to_symmetric(theta_ba)

    A_ba, s, B_ba, s_full = _truncated_svd_symmetric_np(
        theta_sym,
        left_labels=left_labels,
        right_labels=right_labels,
        max_singular_values=config.max_bond_dim,
        max_truncation_err=config.svd_trunc_err,
        new_bond_label=bond_label,
        normalize=False,
    )

    # Truncation error
    n_keep = len(s)
    if len(s_full) > n_keep:
        total_sq = np.sum(s_full ** 2)
        trunc_sq = np.sum(s_full[n_keep:] ** 2)
        trunc_err = float(np.sqrt(trunc_sq / (total_sq + 1e-15)))
    else:
        trunc_err = 0.0

    # Absorb singular values (use numpy version of scale_bond_axis)
    if sweep_right:
        B_ba = _scale_bond_axis_ba(B_ba, bond_label, s)
    else:
        A_ba = _scale_bond_axis_ba(A_ba, bond_label, s)

    return A_ba, s, B_ba, trunc_err
```

Add helper `_scale_bond_axis_ba`:

```python
def _scale_bond_axis_ba(ba: BlockArray, bond_label: str, s: np.ndarray) -> BlockArray:
    """Scale BlockArray along bond axis by singular values."""
    bond_axis = None
    for i, idx in enumerate(ba.indices):
        if idx.label == bond_label:
            bond_axis = i
            break
    if bond_axis is None:
        return ba

    new_blocks = {}
    for key, block in ba.blocks.items():
        # Build shape for broadcasting: s along bond_axis
        shape = [1] * block.ndim
        shape[bond_axis] = len(s)
        new_blocks[key] = block * s.reshape(shape)
    return BlockArray(blocks=new_blocks, indices=ba.indices)
```

**Step 2: Add `_two_site_update_symmetric_np`**

```python
def _two_site_update_symmetric_np(
    site_l: Tensor,
    site_r: Tensor,
    left_env: Tensor,
    mpo_l: Tensor,
    mpo_r: Tensor,
    right_env: Tensor,
    config: DMRGConfig,
) -> tuple[Tensor, float]:
    """2-site DMRG update — pure numpy path."""
    from tenax.algorithms._block_array import (
        BlockArray, ba_to_symmetric, symmetric_to_ba,
    )

    _assert_symmetric(
        site_l, site_r, left_env, mpo_l, mpo_r, right_env,
        context="_two_site_update_symmetric_np",
    )

    # Contract theta
    shared = set(site_l.labels()) & set(site_r.labels())
    theta = contract(site_l, site_r) if shared else site_l
    theta_ba = symmetric_to_ba(theta)

    # Precompute block plan + env numpy blocks (reused across Lanczos)
    _subs = "abc,apqd,bpse,eqtf,dfg->cstg"
    _plan = _precompute_block_plan(
        [left_env, theta, mpo_l, mpo_r, right_env], _subs
    )
    _env_np = [
        _to_np_blocks(left_env), None,
        _to_np_blocks(mpo_l), _to_np_blocks(mpo_r),
        _to_np_blocks(right_env),
    ]
    _cache: dict = {}

    def matvec(v_ba: BlockArray) -> BlockArray:
        v_sym = ba_to_symmetric(v_ba)
        return _blockwise_contract(
            [left_env, v_sym, mpo_l, mpo_r, right_env],
            _subs,
            output_indices=v_sym.indices,
            expr_cache=_cache,
            block_plan=_plan,
            np_blocks_cache=_env_np,
            return_ba=True,
        )

    energy, theta_opt_ba = _lanczos_solve_np(
        matvec, theta_ba, config.lanczos_max_iter, config.lanczos_tol
    )

    return ba_to_symmetric(theta_opt_ba), energy
```

**Step 3: Add `_one_site_update_symmetric_np`**

Same pattern but with 1-site subscripts `"abc,apd,bpxe,def->cxf"`.

**Step 4: Run symmetric DMRG tests to verify**

Run: `uv run pytest tests/test_dmrg.py -k symmetric -v`
Expected: All pass (not yet wired in — Task 7 does dispatch).

**Step 5: Commit**

```bash
git commit -m "feat: add numpy DMRG update functions"
```

---

### Task 7: Config Flag and Dispatch

Add `numpy_blockwise` to `DMRGConfig` and `iDMRGConfig`. Wire dispatch.

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py` (DMRGConfig, _symmetric_ops, dmrg dispatch)
- Modify: `src/tenax/algorithms/idmrg.py` (iDMRGConfig, idmrg dispatch)

**Step 1: Add config field**

In `DMRGConfig` (line ~80), add:

```python
    numpy_blockwise: bool = True  # Use numpy-only path for symmetric DMRG
```

In `iDMRGConfig` (line ~68), add:

```python
    numpy_blockwise: bool = True  # Use numpy-only path for symmetric iDMRG
```

**Step 2: Modify `_symmetric_ops` to accept config**

Change signature to `_symmetric_ops(config: DMRGConfig) -> SweepOps` and dispatch:

```python
def _symmetric_ops(config: DMRGConfig) -> SweepOps:
    """Return the block-sparse symmetric backend callbacks."""
    if config.numpy_blockwise:
        return SweepOps(
            build_trivial_left_env=_build_trivial_left_env_symmetric,
            build_trivial_right_env=_build_trivial_right_env_symmetric,
            update_left_env=_update_left_env_symmetric,
            update_right_env=_update_right_env_symmetric,
            two_site_update=_two_site_update_symmetric_np,
            one_site_update=_one_site_update_symmetric_np,
        )
    return SweepOps(
        build_trivial_left_env=_build_trivial_left_env_symmetric,
        build_trivial_right_env=_build_trivial_right_env_symmetric,
        update_left_env=_update_left_env_symmetric,
        update_right_env=_update_right_env_symmetric,
        two_site_update=_two_site_update_symmetric,
        one_site_update=_one_site_update_symmetric,
    )
```

**Step 3: Update dmrg() dispatch**

At the call site (line ~189), pass config:

```python
    ops = _symmetric_ops(config)
```

**Step 4: Run full test suite**

Run: `uv run pytest -m core -x -q`
Expected: All pass. Symmetric DMRG now uses numpy path by default.

**Step 5: Commit**

```bash
git commit -m "feat: add numpy_blockwise config flag and dispatch"
```

---

### Task 8: iDMRG NumPy Path

Add numpy path to `_idmrg_sweep_symmetric` in `idmrg.py`.

**Files:**
- Modify: `src/tenax/algorithms/idmrg.py`

**Step 1: Add numpy Lanczos and SVD calls**

In `_idmrg_sweep_symmetric` (lines 1201-1351), add a branch on `config.numpy_blockwise`:
- Use `_lanczos_solve_np` instead of `_lanczos_solve_tensor`
- Use `_truncated_svd_symmetric_np` instead of `truncated_svd`
- Convert theta to/from `BlockArray` at the Lanczos boundary

**Step 2: Run iDMRG tests**

Run: `uv run pytest tests/test_idmrg.py -k symmetric -v`
Expected: All pass.

**Step 3: Commit**

```bash
git commit -m "feat: add numpy path to symmetric iDMRG"
```

---

### Task 9: Parametrized Testing and Benchmark

Ensure both numpy and JAX paths produce identical results.

**Files:**
- Modify: `tests/test_dmrg.py` (parametrize symmetric tests)
- Modify: `tests/test_idmrg.py` (parametrize symmetric tests)

**Step 1: Parametrize symmetric DMRG tests**

Add fixture and parametrize all symmetric tests:

```python
@pytest.fixture(params=[True, False], ids=["numpy", "jax"])
def numpy_blockwise(request):
    return request.param
```

Then in each symmetric test, pass `numpy_blockwise` to `DMRGConfig`:

```python
def test_symmetric_block_sparse_energy_matches_exact(self, numpy_blockwise):
    config = DMRGConfig(..., numpy_blockwise=numpy_blockwise)
    result = dmrg(mpo, mps, config)
    ...
```

**Step 2: Run all parametrized tests**

Run: `uv run pytest tests/test_dmrg.py tests/test_idmrg.py -k symmetric -v`
Expected: All pass for both `numpy=True` and `numpy=False`.

**Step 3: Benchmark**

```python
# Quick benchmark script
import time, jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

from tenax import DMRGConfig, build_mpo_heisenberg, build_random_symmetric_mps, dmrg

L, chi = 20, 32
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0)

for use_np in [True, False]:
    mps = build_random_symmetric_mps(L, bond_dim=4, seed=0, target_charge=0)
    config = DMRGConfig(max_bond_dim=chi, num_sweeps=10, convergence_tol=1e-12,
        lanczos_max_iter=30, two_site=True, target_charge=0,
        numpy_blockwise=use_np, verbose=False)
    t0 = time.perf_counter()
    result = dmrg(mpo, mps, config)
    elapsed = time.perf_counter() - t0
    label = "numpy" if use_np else "JAX"
    print(f'{label}: {elapsed:.1f}s  E={result.energy:.10f}')
```

Expected: numpy path 3-5x faster than JAX path.

**Step 4: Commit**

```bash
git commit -m "test: parametrize symmetric DMRG/iDMRG tests for numpy/JAX paths"
```
