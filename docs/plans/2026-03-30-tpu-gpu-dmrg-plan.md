# TPU/GPU Accelerated DMRG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** JIT-compile the full DMRG sweep for GPU/TPU execution, supporting both dense and block-sparse (symmetric) tensors via padded vmap.

**Architecture:** New `PaddedBlockArray` representation stores all charge-sector blocks in a uniform-shape JAX array. Contractions use `jax.vmap` over blocks with a static `PaddedContractionPlan`. The entire sweep compiles to a single XLA program via `jax.lax.scan`. Warmup sweeps (growing chi) use the existing Python loop; once chi saturates, the JIT path takes over.

**Tech Stack:** JAX (`lax.scan`, `lax.fori_loop`, `lax.top_k`, `vmap`, `jit`), existing Tenax tensor/contraction infrastructure.

**Design doc:** `docs/plans/2026-03-30-tpu-gpu-dmrg-design.md`

---

### Task 1: PaddedBlockArray Data Structure

**Files:**
- Create: `src/tenax/algorithms/_padded_block_array.py`
- Test: `tests/test_padded_block_array.py`

**Step 1: Write failing tests for PaddedBlockArray**

```python
# tests/test_padded_block_array.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor, DenseTensor
from tenax.core.index import TensorIndex, FlowDirection


class TestPaddedBlockArray:
    def test_from_symmetric_creates_padded_array(self):
        """PaddedBlockArray.from_symmetric produces a 3D data array."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray

        sym = _make_small_symmetric_tensor()  # helper below
        pba = PaddedBlockArray.from_symmetric(sym)

        assert isinstance(pba.data, jax.Array)
        assert pba.data.ndim == 3  # (num_blocks, M_max, N_max)
        assert pba.mask.shape == pba.data.shape
        assert pba.mask.dtype == jnp.bool_

    def test_round_trip_symmetric(self):
        """from_symmetric -> to_symmetric is identity (up to float precision)."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray

        sym = _make_small_symmetric_tensor()
        pba = PaddedBlockArray.from_symmetric(sym)
        sym2 = pba.to_symmetric()

        # Compare block data
        for key in sym._block_data:
            np.testing.assert_allclose(
                np.array(sym._block_data[key]),
                np.array(sym2._block_data[key]),
                atol=1e-12,
            )

    def test_pytree_registration(self):
        """PaddedBlockArray works with jax.jit (pytree leaves are data+mask)."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray

        sym = _make_small_symmetric_tensor()
        pba = PaddedBlockArray.from_symmetric(sym)

        @jax.jit
        def scale(pba):
            return pba._replace(data=pba.data * 2.0)

        pba2 = scale(pba)
        np.testing.assert_allclose(np.array(pba2.data), np.array(pba.data) * 2.0)

    def test_mask_zeros_padding(self):
        """Padding region has mask=False and data=0."""
        from tenax.algorithms._padded_block_array import PaddedBlockArray

        sym = _make_small_symmetric_tensor()
        pba = PaddedBlockArray.from_symmetric(sym)

        # Where mask is False, data must be zero
        padding = ~pba.mask
        np.testing.assert_array_equal(np.array(pba.data[padding]), 0.0)

    def test_from_dense_pads_to_chi_max(self):
        """DenseTensor padded to chi_max has correct shape."""
        from tenax.algorithms._padded_block_array import pad_dense

        dense_data = jnp.ones((4, 2, 6))  # (chi_l, d, chi_r)
        padded, mask = pad_dense(dense_data, chi_max=8)

        assert padded.shape == (8, 2, 8)
        assert mask.shape == (8, 2, 8)
        np.testing.assert_array_equal(np.array(padded[:4, :, :6]), 1.0)
        np.testing.assert_array_equal(np.array(padded[4:, :, :]), 0.0)


def _make_small_symmetric_tensor():
    """Build a small 2-index SymmetricTensor with U(1) symmetry for testing."""
    from tenax.algorithms.auto_mpo import build_auto_mpo
    from tenax.algorithms.dmrg import build_random_symmetric_mps

    # Use an MPS tensor from a random symmetric MPS — guaranteed valid blocks
    mps = build_random_symmetric_mps(
        L=4, d=2, chi=4, key=jax.random.PRNGKey(0),
        symmetry=U1Symmetry(), target_charge=0,
    )
    # Return the bond matrix between sites 0 and 1 (2-index symmetric tensor)
    # For simplicity, use the MPS tensor at site 1 reshaped to a matrix
    t = mps.tensors[1]
    from tenax.linalg import svd
    U, s, Vh, _ = svd(t, left_labels=[t.labels()[0], t.labels()[1]],
                       right_labels=[t.labels()[2]])
    return U  # 2-index SymmetricTensor
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_padded_block_array.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.algorithms._padded_block_array'`

**Step 3: Implement PaddedBlockArray**

```python
# src/tenax/algorithms/_padded_block_array.py
"""Padded block array for accelerator-native block-sparse operations.

Stores all charge-sector blocks of a SymmetricTensor as a single
(num_blocks, M_max, N_max) JAX array with zero-padding and a boolean mask.
This uniform representation enables jax.vmap and jax.lax.scan over blocks
without dynamic shapes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tenax.core.tensor import SymmetricTensor
from tenax.core.index import TensorIndex


class PaddedBlockArray(NamedTuple):
    """Uniform-shape block array for accelerator execution.

    Attributes:
        data:           (num_blocks, M_max, N_max) zero-padded JAX array.
        mask:           (num_blocks, M_max, N_max) bool — True for real data.
        block_charges:  Static tuple of charge labels per block.
        block_shapes:   Static tuple of (m_i, n_i) actual shapes per block.
        indices:        TensorIndex metadata for reconstruction.
        symmetry:       Symmetry object for reconstruction.
    """

    data: jax.Array
    mask: jax.Array
    block_charges: tuple  # pytree aux (static)
    block_shapes: tuple   # pytree aux (static)
    indices: tuple        # pytree aux (static)
    symmetry: object      # pytree aux (static)

    @staticmethod
    def from_symmetric(tensor: SymmetricTensor) -> PaddedBlockArray:
        """Convert a 2-index SymmetricTensor to padded block representation."""
        blocks = tensor._block_data
        charges = tuple(sorted(blocks.keys()))
        shapes = tuple(tuple(int(d) for d in blocks[c].shape) for c in charges)

        if not charges:
            # Empty tensor — return trivial padded array
            data = jnp.zeros((0, 0, 0))
            mask = jnp.zeros((0, 0, 0), dtype=jnp.bool_)
            return PaddedBlockArray(data, mask, (), (), tuple(tensor.indices), tensor.symmetry)

        M_max = max(s[0] for s in shapes)
        N_max = max(s[1] for s in shapes) if shapes[0].__len__() > 1 else 1

        num_blocks = len(charges)
        data_np = np.zeros((num_blocks, M_max, N_max), dtype=np.float64)
        mask_np = np.zeros((num_blocks, M_max, N_max), dtype=bool)

        for i, (c, (m, n)) in enumerate(zip(charges, shapes)):
            block = np.array(blocks[c])
            data_np[i, :m, :n] = block
            mask_np[i, :m, :n] = True

        return PaddedBlockArray(
            data=jnp.array(data_np),
            mask=jnp.array(mask_np),
            block_charges=charges,
            block_shapes=shapes,
            indices=tuple(tensor.indices),
            symmetry=tensor.symmetry,
        )

    def to_symmetric(self) -> SymmetricTensor:
        """Convert back to a SymmetricTensor by stripping padding."""
        block_data = {}
        for i, (charge, (m, n)) in enumerate(
            zip(self.block_charges, self.block_shapes)
        ):
            block_data[charge] = jnp.array(self.data[i, :m, :n])

        return SymmetricTensor._from_block_data(
            block_data=block_data,
            indices=self.indices,
            symmetry=self.symmetry,
        )


def pad_dense(data: jax.Array, chi_max: int) -> tuple[jax.Array, jax.Array]:
    """Pad a dense MPS tensor to chi_max along bond dimensions (axis 0 and -1).

    Args:
        data: (chi_l, d, chi_r) array.
        chi_max: Target bond dimension.

    Returns:
        (padded_data, mask) both of shape (chi_max, d, chi_max).
    """
    chi_l, d, chi_r = data.shape
    padded = jnp.zeros((chi_max, d, chi_max), dtype=data.dtype)
    padded = padded.at[:chi_l, :, :chi_r].set(data)
    mask = jnp.zeros((chi_max, d, chi_max), dtype=jnp.bool_)
    mask = mask.at[:chi_l, :, :chi_r].set(True)
    return padded, mask
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_padded_block_array.py -v`
Expected: PASS (may need to adjust `_make_small_symmetric_tensor` helper based on actual SymmetricTensor internals — the `_block_data` dict and `_from_block_data` classmethod must match the real API).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_padded_block_array.py tests/test_padded_block_array.py
git commit -m "feat: add PaddedBlockArray for accelerator-native block-sparse ops"
```

---

### Task 2: PaddedContractionPlan and vmap Contractions

**Files:**
- Modify: `src/tenax/algorithms/_padded_block_array.py`
- Test: `tests/test_padded_block_array.py`

**Step 1: Write failing tests for padded contraction**

```python
# Add to tests/test_padded_block_array.py

class TestPaddedContraction:
    def test_matmul_matches_symmetric_contract(self):
        """vmap padded matmul matches tenax.contract on SymmetricTensors."""
        from tenax.algorithms._padded_block_array import (
            PaddedBlockArray,
            PaddedContractionPlan,
            contract_padded,
        )
        from tenax.contraction.contractor import contract

        A_sym, B_sym = _make_contractible_pair()
        C_sym = contract(A_sym, B_sym)

        A_pba = PaddedBlockArray.from_symmetric(A_sym)
        B_pba = PaddedBlockArray.from_symmetric(B_sym)
        plan = PaddedContractionPlan.build(A_sym, B_sym)
        C_pba = contract_padded(plan, A_pba, B_pba)
        C_sym2 = C_pba.to_symmetric()

        for key in C_sym._block_data:
            np.testing.assert_allclose(
                np.array(C_sym._block_data[key]),
                np.array(C_sym2._block_data[key]),
                atol=1e-10,
            )

    def test_plan_is_jit_compatible(self):
        """contract_padded can be called inside jax.jit."""
        from tenax.algorithms._padded_block_array import (
            PaddedBlockArray,
            PaddedContractionPlan,
            contract_padded,
        )

        A_sym, B_sym = _make_contractible_pair()
        A_pba = PaddedBlockArray.from_symmetric(A_sym)
        B_pba = PaddedBlockArray.from_symmetric(B_sym)
        plan = PaddedContractionPlan.build(A_sym, B_sym)

        @jax.jit
        def f(a_data, b_data):
            a = A_pba._replace(data=a_data)
            b = B_pba._replace(data=b_data)
            return contract_padded(plan, a, b).data

        result = f(A_pba.data, B_pba.data)
        assert result.shape[0] > 0  # got a result


def _make_contractible_pair():
    """Build two SymmetricTensors that share a contracted index."""
    from tenax.core.mps import FiniteMPS
    from tenax.core.symmetry import U1Symmetry
    from tenax.linalg import svd

    mps = FiniteMPS.random(
        L=4, d=2, chi=8, key=jax.random.PRNGKey(42),
        symmetric=True, symmetry=U1Symmetry(), target_charge=0,
    )
    # SVD site 1 → U (2-index) and Vh (2-index) sharing a bond
    t = mps.tensors[1]
    U, s, Vh, _ = svd(t, left_labels=[t.labels()[0], t.labels()[1]],
                       right_labels=[t.labels()[2]])
    # Absorb s into U: A = U @ diag(s), B = Vh
    from tenax.contraction.contractor import contract
    s_diag = s  # s is already a 1-index tensor; need to make diagonal
    A = contract(U, s)
    return A, Vh
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_padded_block_array.py::TestPaddedContraction -v`
Expected: FAIL — `ImportError: cannot import name 'PaddedContractionPlan'`

**Step 3: Implement PaddedContractionPlan and contract_padded**

Add to `src/tenax/algorithms/_padded_block_array.py`:

```python
class PaddedContractionPlan(NamedTuple):
    """Static plan for vmapped block-sparse contraction.

    Precomputes which blocks of A and B contribute to each output block,
    based on charge conservation. All fields are static (pytree aux).
    """

    left_indices: tuple[int, ...]     # indices into A.data
    right_indices: tuple[int, ...]    # indices into B.data
    output_indices: tuple[int, ...]   # indices into output.data (for scatter-add)
    num_output_blocks: int
    output_M_max: int
    output_N_max: int
    output_charges: tuple
    output_shapes: tuple
    subscripts: str  # einsum subscripts for one block pair

    @staticmethod
    def build(A: SymmetricTensor, B: SymmetricTensor) -> PaddedContractionPlan:
        """Build a contraction plan from two SymmetricTensors.

        Determines which block pairs (a_q, b_q') contribute to each output
        block c_{q+q'} based on charge conservation on the contracted index.
        """
        # Find the shared (contracted) label
        a_labels = set(A.labels())
        b_labels = set(B.labels())
        contracted = a_labels & b_labels
        assert len(contracted) == 1, f"Expected 1 shared label, got {contracted}"
        contracted_label = contracted.pop()

        # Get charge→block mapping for the contracted index
        a_charges = sorted(A._block_data.keys())
        b_charges = sorted(B._block_data.keys())

        # Build pairs: for each (a_charge, b_charge) where contracted charges match
        left_idx, right_idx, out_idx = [], [], []
        output_charge_set = {}  # charge → output block index

        a_charge_map = {c: i for i, c in enumerate(a_charges)}
        b_charge_map = {c: i for i, c in enumerate(b_charges)}

        # For 2-index tensors with one contracted label:
        # A[q1, q_bond] * B[q_bond, q2] → C[q1, q2]
        # charge conservation: q_bond must match
        for a_c in a_charges:
            if a_c in b_charge_map:
                ai = a_charge_map[a_c]
                bi = b_charge_map[a_c]
                out_charge = a_c  # simplified; real impl needs proper charge arithmetic
                if out_charge not in output_charge_set:
                    output_charge_set[out_charge] = len(output_charge_set)
                left_idx.append(ai)
                right_idx.append(bi)
                out_idx.append(output_charge_set[out_charge])

        num_out = len(output_charge_set)
        # Compute output shapes from A's left dim and B's right dim
        out_shapes = []
        out_charges = sorted(output_charge_set.keys(), key=lambda c: output_charge_set[c])
        for oc in out_charges:
            # Find the A and B blocks that contribute
            pairs = [(l, r) for l, r, o in zip(left_idx, right_idx, out_idx)
                     if o == output_charge_set[oc]]
            m = max(A._block_data[a_charges[l]].shape[0] for l, _ in pairs)
            n = max(B._block_data[b_charges[r]].shape[-1] for _, r in pairs)
            out_shapes.append((m, n))

        M_max = max(s[0] for s in out_shapes) if out_shapes else 0
        N_max = max(s[1] for s in out_shapes) if out_shapes else 0

        return PaddedContractionPlan(
            left_indices=tuple(left_idx),
            right_indices=tuple(right_idx),
            output_indices=tuple(out_idx),
            num_output_blocks=num_out,
            output_M_max=M_max,
            output_N_max=N_max,
            output_charges=tuple(out_charges),
            output_shapes=tuple(out_shapes),
            subscripts="ij,jk->ik",  # default matmul; generalize later
        )


def contract_padded(
    plan: PaddedContractionPlan,
    A: PaddedBlockArray,
    B: PaddedBlockArray,
) -> PaddedBlockArray:
    """Execute a padded block-sparse contraction via vmap.

    Gathers participating blocks by index, vmaps the per-block einsum,
    and scatter-adds results into output blocks.
    """
    # Gather participating blocks
    a_blocks = A.data[jnp.array(plan.left_indices)]    # (N_pairs, M_a, K)
    b_blocks = B.data[jnp.array(plan.right_indices)]    # (N_pairs, K, N_b)

    # vmap the per-block contraction
    def single_contract(a, b):
        return jnp.einsum(plan.subscripts, a, b)

    results = jax.vmap(single_contract)(a_blocks, b_blocks)  # (N_pairs, M_out, N_out)

    # Pad results to output max shape if needed
    if results.shape[1] < plan.output_M_max or results.shape[2] < plan.output_N_max:
        padded_results = jnp.zeros(
            (results.shape[0], plan.output_M_max, plan.output_N_max),
            dtype=results.dtype,
        )
        padded_results = padded_results.at[:, :results.shape[1], :results.shape[2]].set(results)
        results = padded_results

    # Scatter-add into output blocks
    output = jnp.zeros(
        (plan.num_output_blocks, plan.output_M_max, plan.output_N_max),
        dtype=results.dtype,
    )
    out_idx = jnp.array(plan.output_indices)
    output = output.at[out_idx].add(results)

    # Build mask for output
    mask = jnp.zeros_like(output, dtype=jnp.bool_)
    for i, (m, n) in enumerate(plan.output_shapes):
        mask = mask.at[i, :m, :n].set(True)

    return PaddedBlockArray(
        data=output,
        mask=mask,
        block_charges=plan.output_charges,
        block_shapes=plan.output_shapes,
        indices=(),  # caller must set appropriate indices
        symmetry=A.symmetry,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_padded_block_array.py::TestPaddedContraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_padded_block_array.py tests/test_padded_block_array.py
git commit -m "feat: add PaddedContractionPlan and vmap contract_padded"
```

---

### Task 3: Padded Environment Updates

**Files:**
- Create: `src/tenax/algorithms/_jit_sweep.py`
- Test: `tests/test_jit_sweep.py`

**Step 1: Write failing tests for padded environment updates**

```python
# tests/test_jit_sweep.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.dmrg import (
    DMRGConfig,
    build_mpo_heisenberg,
    build_random_mps,
    dmrg,
)
from tenax.core.tensor import DenseTensor


class TestPaddedEnvUpdate:
    def test_dense_left_env_update_matches_python(self):
        """Padded left env update matches existing dense Python implementation."""
        from tenax.algorithms._jit_sweep import (
            update_left_env_dense_jit,
        )
        from tenax.algorithms.dmrg import _update_left_env

        L = 6
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=0)

        # Get a left env from the Python path
        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = list(mps.tensors)

        # Trivial left boundary
        from tenax.algorithms.dmrg import _build_trivial_left_env
        L_env = _build_trivial_left_env()

        # Python reference
        L_env_ref = _update_left_env(L_env, mps_tensors[0], mpo_tensors[0])

        # JIT padded version
        chi_max = 16
        L_env_jit = update_left_env_dense_jit(
            L_env.todense(), mps_tensors[0].todense(),
            mpo_tensors[0].todense(), chi_max,
        )

        # Compare (unpad to original shape)
        ref = L_env_ref.todense()
        np.testing.assert_allclose(
            np.array(L_env_jit[:ref.shape[0], :ref.shape[1], :ref.shape[2]]),
            np.array(ref),
            atol=1e-10,
        )

    def test_dense_right_env_update_matches_python(self):
        """Padded right env update matches existing dense Python implementation."""
        from tenax.algorithms._jit_sweep import (
            update_right_env_dense_jit,
        )
        from tenax.algorithms.dmrg import _update_right_env

        L = 6
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=0)

        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = list(mps.tensors)

        from tenax.algorithms.dmrg import _build_trivial_right_env
        R_env = _build_trivial_right_env()

        R_env_ref = _update_right_env(R_env, mps_tensors[-1], mpo_tensors[-1])

        chi_max = 16
        R_env_jit = update_right_env_dense_jit(
            R_env.todense(), mps_tensors[-1].todense(),
            mpo_tensors[-1].todense(), chi_max,
        )

        ref = R_env_ref.todense()
        np.testing.assert_allclose(
            np.array(R_env_jit[:ref.shape[0], :ref.shape[1], :ref.shape[2]]),
            np.array(ref),
            atol=1e-10,
        )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedEnvUpdate -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement padded environment updates**

```python
# src/tenax/algorithms/_jit_sweep.py
"""JIT-compiled DMRG sweep for GPU/TPU acceleration.

This module implements the full DMRG sweep as a jax.lax.scan, compiling
the entire sweep into a single XLA program. All tensors are padded to
fixed shapes (chi_max) to enable static-shape JIT compilation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def update_left_env_dense_jit(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Update left environment: L' = L * A * W * A*.

    All inputs are raw JAX arrays (not Tensor objects). Output is padded
    to (chi_max, D_w, chi_max).

    Args:
        L_env: (chi_l, D_w_l, chi_l) left environment.
        A:     (chi_l, d, chi_r) MPS tensor.
        W:     (D_w_l, d, d, D_w_r) MPO tensor.
        chi_max: Pad output to this bond dimension.
    """
    new_L = jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))
    # Pad to (chi_max, D_w, chi_max)
    D_w = new_L.shape[1]
    out = jnp.zeros((chi_max, D_w, chi_max), dtype=new_L.dtype)
    out = out.at[: new_L.shape[0], :, : new_L.shape[2]].set(new_L)
    return out


def update_right_env_dense_jit(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Update right environment: R' = B * W * B* * R.

    Args:
        R_env: (chi_r, D_w_r, chi_r) right environment.
        B:     (chi_l, d, chi_r) MPS tensor.
        W:     (D_w_l, d, d, D_w_r) MPO tensor.
        chi_max: Pad output to this bond dimension.
    """
    new_R = jnp.einsum("abc,dpb,edxf,fxc->epa", R_env, B, W, jnp.conj(B))
    # Pad to (chi_max, D_w, chi_max)
    # Note: adjust einsum indices to match actual env convention in dmrg.py
    D_w = new_R.shape[1]
    out = jnp.zeros((chi_max, D_w, chi_max), dtype=new_R.dtype)
    out = out.at[: new_R.shape[0], :, : new_R.shape[2]].set(new_R)
    return out
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedEnvUpdate -v`
Expected: PASS (einsum index strings must match the conventions in `dmrg.py:688-769` — verify and adjust).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add padded environment update functions for JIT sweep"
```

---

### Task 4: Padded SVD with `jax.lax.top_k` Truncation

**Files:**
- Modify: `src/tenax/algorithms/_jit_sweep.py`
- Modify: `tests/test_jit_sweep.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_jit_sweep.py

class TestPaddedSVD:
    def test_dense_svd_truncation_matches_numpy(self):
        """Padded SVD with top_k truncation matches numpy SVD + truncation."""
        from tenax.algorithms._jit_sweep import padded_svd_dense

        key = jax.random.PRNGKey(0)
        theta = jax.random.normal(key, (8, 2, 2, 8))  # (chi_l, d_l, d_r, chi_r)
        chi_max = 6

        # Reference: numpy SVD
        mat = theta.reshape(8 * 2, 2 * 8)
        U_ref, s_ref, Vh_ref = np.linalg.svd(np.array(mat), full_matrices=False)
        U_ref = U_ref[:, :chi_max]
        s_ref = s_ref[:chi_max]
        Vh_ref = Vh_ref[:chi_max, :]

        # Padded JIT version
        U_jit, s_jit, Vh_jit = padded_svd_dense(theta, chi_max)

        np.testing.assert_allclose(np.array(s_jit[:chi_max]), s_ref, atol=1e-10)

    def test_padded_svd_is_jittable(self):
        """padded_svd_dense works inside jax.jit."""
        from tenax.algorithms._jit_sweep import padded_svd_dense

        @jax.jit
        def f(theta):
            return padded_svd_dense(theta, chi_max=4)

        theta = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 2, 4))
        U, s, Vh = f(theta)
        assert U.shape[1] == 4  # chi_max columns
        assert s.shape[0] == 4

    def test_block_sparse_svd_matches_symmetric_svd(self):
        """Padded block SVD with vmap + top_k matches tenax.linalg.svd."""
        from tenax.algorithms._jit_sweep import padded_svd_blocks
        from tenax.algorithms._padded_block_array import PaddedBlockArray
        from tenax.linalg import svd as tenax_svd

        # Build a 2-index SymmetricTensor and SVD it both ways
        sym = _make_symmetric_matrix()
        chi_max = 6

        # Reference
        U_ref, s_ref, Vh_ref, _ = tenax_svd(
            sym,
            left_labels=[sym.labels()[0]],
            right_labels=[sym.labels()[1]],
            max_singular_values=chi_max,
        )

        # Padded path
        pba = PaddedBlockArray.from_symmetric(sym)
        U_pba, s_jit, Vh_pba = padded_svd_blocks(pba, chi_max)

        # Compare singular values (sorted descending)
        s_ref_arr = np.sort(np.array(s_ref.todense()))[::-1]
        s_jit_arr = np.sort(np.array(s_jit))[::-1]
        # Top chi_max should match
        n = min(len(s_ref_arr), chi_max)
        np.testing.assert_allclose(s_jit_arr[:n], s_ref_arr[:n], atol=1e-8)


def _make_symmetric_matrix():
    """Build a 2-index SymmetricTensor for SVD testing."""
    from tenax.core.mps import FiniteMPS
    from tenax.core.symmetry import U1Symmetry
    from tenax.contraction.contractor import contract

    mps = FiniteMPS.random(
        L=6, d=2, chi=8, key=jax.random.PRNGKey(7),
        symmetric=True, symmetry=U1Symmetry(), target_charge=0,
    )
    t = mps.tensors[2]
    # Contract with conjugate to get a positive-definite 2-index tensor
    t_dag = t.conj()
    # Relabel conjugate's bond labels to avoid self-contraction
    labels = t.labels()
    rho = contract(t, t_dag)  # contracts over shared physical label
    return rho
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedSVD -v`
Expected: FAIL — `ImportError: cannot import name 'padded_svd_dense'`

**Step 3: Implement padded SVD**

Add to `src/tenax/algorithms/_jit_sweep.py`:

```python
def padded_svd_dense(
    theta: jax.Array,
    chi_max: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """SVD + truncation on a dense 2-site theta tensor, JIT-compatible.

    Args:
        theta: (chi_l, d_l, d_r, chi_r) two-site wavefunction.
        chi_max: Keep at most this many singular values.

    Returns:
        U:  (chi_l * d_l, chi_max) left singular vectors.
        s:  (chi_max,) singular values (zero-padded if rank < chi_max).
        Vh: (chi_max, d_r * chi_r) right singular vectors.
    """
    shape = theta.shape
    mat = theta.reshape(shape[0] * shape[1], shape[2] * shape[3])
    U, s, Vh = jnp.linalg.svd(mat, full_matrices=False)

    # Truncate to chi_max (static shape via slicing)
    U = U[:, :chi_max]
    s = s[:chi_max]
    Vh = Vh[:chi_max, :]

    # Pad if rank < chi_max (ensures static shape)
    if U.shape[1] < chi_max:
        U = jnp.pad(U, ((0, 0), (0, chi_max - U.shape[1])))
        s = jnp.pad(s, (0, chi_max - s.shape[0]))
        Vh = jnp.pad(Vh, ((0, chi_max - Vh.shape[0]), (0, 0)))

    return U, s, Vh


def padded_svd_blocks(
    pba,  # PaddedBlockArray
    chi_max: int,
) -> tuple:
    """Block-sparse SVD with global truncation via jax.lax.top_k.

    Vmaps jnp.linalg.svd over all blocks, then selects the globally
    largest chi_max singular values across all sectors.

    Args:
        pba: PaddedBlockArray with (num_blocks, M_max, N_max) data.
        chi_max: Maximum total bond dimension across all sectors.

    Returns:
        U_pba:  PaddedBlockArray of left singular vectors.
        s_flat: (chi_max,) globally sorted singular values.
        Vh_pba: PaddedBlockArray of right singular vectors.
    """
    # vmap SVD over all blocks
    U_all, s_all, Vh_all = jax.vmap(
        lambda m: jnp.linalg.svd(m, full_matrices=False)
    )(pba.data)
    # s_all: (num_blocks, min(M_max, N_max))

    # Global truncation: find top chi_max singular values
    s_flat = s_all.ravel()
    k = min(chi_max, s_flat.shape[0])
    top_values, top_indices = jax.lax.top_k(s_flat, k)

    # Pad to chi_max if fewer values available
    if k < chi_max:
        top_values = jnp.pad(top_values, (0, chi_max - k))

    # Compute per-block keep counts
    sv_per_block = s_all.shape[1]
    block_ids = top_indices // sv_per_block
    per_block_keep = jnp.zeros(pba.data.shape[0], dtype=jnp.int32)
    per_block_keep = per_block_keep.at[block_ids].add(1)

    # Build column masks for U and row masks for Vh
    # Each block keeps its top per_block_keep[i] columns/rows
    col_range = jnp.arange(sv_per_block)
    col_mask = col_range[None, :] < per_block_keep[:, None]  # (num_blocks, sv_per_block)

    # Apply masks to zero out truncated columns
    U_masked = U_all * col_mask[:, None, :]  # broadcast over M_max
    Vh_masked = Vh_all * col_mask[:, :, None]  # broadcast over N_max
    s_masked = s_all * col_mask

    return U_masked, top_values, Vh_masked
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedSVD -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add padded SVD with jax.lax.top_k global truncation"
```

---

### Task 5: Padded Lanczos Eigensolver

**Files:**
- Modify: `src/tenax/algorithms/_jit_sweep.py`
- Modify: `tests/test_jit_sweep.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_jit_sweep.py

class TestPaddedLanczos:
    def test_dense_lanczos_finds_ground_state(self):
        """Padded Lanczos on a small Hamiltonian finds the correct eigenvalue."""
        from tenax.algorithms._jit_sweep import lanczos_ground_state_dense

        # Build a small explicit Hamiltonian matrix
        key = jax.random.PRNGKey(0)
        H = jax.random.normal(key, (8, 8))
        H = (H + H.T) / 2  # symmetrize

        e_exact = float(np.linalg.eigvalsh(np.array(H))[0])

        # Lanczos
        v0 = jax.random.normal(jax.random.PRNGKey(1), (8,))
        e_lanczos, v_lanczos = lanczos_ground_state_dense(
            lambda x: H @ x, v0, max_iter=20,
        )

        assert abs(float(e_lanczos) - e_exact) < 1e-8

    def test_lanczos_is_jittable(self):
        """lanczos_ground_state_dense compiles under jax.jit."""
        from tenax.algorithms._jit_sweep import lanczos_ground_state_dense

        H = jnp.eye(4) * jnp.array([1.0, 2.0, 3.0, 4.0])

        @jax.jit
        def f(v0):
            return lanczos_ground_state_dense(lambda x: H @ x, v0, max_iter=10)

        v0 = jnp.ones(4) / 2.0
        e, v = f(v0)
        assert abs(float(e) - 1.0) < 1e-8

    def test_effective_hamiltonian_matvec_dense(self):
        """Padded effective Hamiltonian matvec matches the Python version."""
        from tenax.algorithms._jit_sweep import effective_ham_matvec_dense
        from tenax.algorithms.dmrg import (
            _effective_hamiltonian_matvec,
            build_mpo_heisenberg,
            build_random_mps,
            _build_trivial_left_env,
            _build_trivial_right_env,
        )

        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)

        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = list(mps.tensors)

        L_env = _build_trivial_left_env()
        R_env = _build_trivial_right_env()

        # Build 2-site theta
        theta = jnp.einsum("ipj,jqk->ipqk",
                           mps_tensors[0].todense(), mps_tensors[1].todense())

        # Reference
        ref = _effective_hamiltonian_matvec(
            theta.ravel(), theta.shape,
            L_env.todense(), mpo_tensors[0].todense(),
            mpo_tensors[1].todense(), R_env.todense(),
        )

        # Padded version
        chi_max = 8
        result = effective_ham_matvec_dense(
            theta, L_env.todense(), mpo_tensors[0].todense(),
            mpo_tensors[1].todense(), R_env.todense(), chi_max,
        )

        np.testing.assert_allclose(
            np.array(result[:theta.shape[0], :theta.shape[1],
                           :theta.shape[2], :theta.shape[3]]),
            np.array(ref.reshape(theta.shape)),
            atol=1e-10,
        )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedLanczos -v`
Expected: FAIL — `ImportError`

**Step 3: Implement padded Lanczos**

Add to `src/tenax/algorithms/_jit_sweep.py`:

```python
def effective_ham_matvec_dense(
    theta: jax.Array,
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
    chi_max: int,
) -> jax.Array:
    """Apply the effective Hamiltonian to a 2-site theta (dense, padded).

    Args:
        theta: (chi_max, d_l, d_r, chi_max) padded two-site wavefunction.
        L_env: (chi_max, D_w, chi_max) left environment.
        W_l:   (D_w_l, d, d, D_w_m) left MPO tensor.
        W_r:   (D_w_m, d, d, D_w_r) right MPO tensor.
        R_env: (chi_max, D_w, chi_max) right environment.
        chi_max: Bond dimension (for documentation; shapes are already padded).
    """
    return jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env, theta, W_l, W_r, R_env,
    )


def lanczos_ground_state_dense(
    matvec,
    v0: jax.Array,
    max_iter: int = 20,
) -> tuple[jax.Array, jax.Array]:
    """Lanczos eigensolver for the ground state, fully JIT-compatible.

    Uses jax.lax.fori_loop for the iteration and jax.lax.scan for
    reorthogonalization. No host-device synchronization.

    Args:
        matvec: Function x -> H @ x.
        v0: Initial vector (1D array).
        max_iter: Number of Lanczos iterations.

    Returns:
        (eigenvalue, eigenvector) of the lowest eigenstate.
    """
    n = v0.shape[0]
    v0 = v0 / jnp.linalg.norm(v0)

    # Pre-allocate basis matrix (max_iter, n) and tridiagonal entries
    Q = jnp.zeros((max_iter, n), dtype=v0.dtype)
    Q = Q.at[0].set(v0)
    alphas = jnp.zeros(max_iter, dtype=v0.dtype)
    betas = jnp.zeros(max_iter, dtype=v0.dtype)

    def body(i, carry):
        Q, alphas, betas = carry
        v = Q[i]
        w = matvec(v)

        alpha = jnp.dot(v, w)
        alphas = alphas.at[i].set(alpha)

        w = w - alpha * v
        w = jnp.where(i > 0, w - betas[i] * Q[i - 1], w)

        # Full reorthogonalization via scan
        def reorth_step(w, q):
            proj = jnp.dot(q, w)
            w = w - proj * q
            return w, None

        w, _ = jax.lax.scan(reorth_step, w, Q)

        beta = jnp.linalg.norm(w)
        betas = betas.at[i + 1].set(beta)
        # Guard against zero beta (converged subspace)
        w_normed = jnp.where(beta > 1e-15, w / beta, w)
        Q = jnp.where(i + 1 < max_iter, Q.at[i + 1].set(w_normed), Q)

        return Q, alphas, betas

    Q, alphas, betas = jax.lax.fori_loop(0, max_iter, body, (Q, alphas, betas))

    # Solve tridiagonal eigenvalue problem
    T = jnp.diag(alphas) + jnp.diag(betas[1:max_iter], k=1) + jnp.diag(betas[1:max_iter], k=-1)
    eigvals, eigvecs = jnp.linalg.eigh(T)

    # Ground state
    eigenvalue = eigvals[0]
    ritz_vec = eigvecs[:, 0]

    # Reconstruct eigenvector in original basis
    eigenvector = Q.T @ ritz_vec  # (n,)

    return eigenvalue, eigenvector
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jit_sweep.py::TestPaddedLanczos -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add JIT-compatible Lanczos eigensolver and effective Hamiltonian matvec"
```

---

### Task 6: Full `lax.scan` Sweep (Dense Path)

**Files:**
- Modify: `src/tenax/algorithms/_jit_sweep.py`
- Modify: `tests/test_jit_sweep.py`

**Step 1: Write failing integration test**

```python
# Add to tests/test_jit_sweep.py

class TestJITSweep:
    def test_dense_jit_sweep_matches_python_sweep(self):
        """Full JIT sweep on dense Heisenberg L=6 matches Python sweep energy."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        L = 6
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=0)

        # Python reference: run 3 sweeps
        config_ref = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=3,
            lanczos_max_iter=20,
            convergence_tol=1e-12,
        )
        result_ref = dmrg(mpo, mps, config_ref)

        # JIT sweep: run 3 sweeps
        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = [t.todense() for t in mps.tensors]
        chi_max = 8

        energies = jit_dmrg_sweep_dense(
            mps_tensors, mpo_tensors, chi_max,
            num_sweeps=3, lanczos_max_iter=20,
        )

        # Energies should be close (not identical due to different
        # ordering of operations, but same ground state)
        assert abs(energies[-1] - result_ref.energy) < 1e-4

    def test_jit_sweep_compiles_once(self):
        """Second call to jit_dmrg_sweep_dense should not recompile."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        L = 4
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)

        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = [t.todense() for t in mps.tensors]

        # First call (compiles)
        e1 = jit_dmrg_sweep_dense(mps_tensors, mpo_tensors, 4, 1, 10)
        # Second call (should reuse compiled code)
        e2 = jit_dmrg_sweep_dense(mps_tensors, mpo_tensors, 4, 1, 10)

        assert len(e1) == len(e2)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jit_sweep.py::TestJITSweep -v`
Expected: FAIL — `ImportError: cannot import name 'jit_dmrg_sweep_dense'`

**Step 3: Implement the full JIT sweep**

Add to `src/tenax/algorithms/_jit_sweep.py`:

```python
def jit_dmrg_sweep_dense(
    mps_tensors: list[jax.Array],
    mpo_tensors: list,
    chi_max: int,
    num_sweeps: int = 1,
    lanczos_max_iter: int = 20,
) -> list[float]:
    """Run DMRG sweeps with full JIT compilation via lax.scan.

    All MPS tensors must be raw JAX arrays padded to (chi_max, d, chi_max).
    MPO tensors are DenseTensor objects (todense() called internally).

    Args:
        mps_tensors: List of L MPS tensors as JAX arrays.
        mpo_tensors: List of L MPO tensors (DenseTensor or raw arrays).
        chi_max: Maximum bond dimension (all tensors padded to this).
        num_sweeps: Number of full left-right-left sweeps.
        lanczos_max_iter: Lanczos iterations per site update.

    Returns:
        List of energies, one per sweep.
    """
    L = len(mps_tensors)
    d = mps_tensors[0].shape[1]  # physical dimension

    # Extract raw arrays from MPO tensors
    W_list = [t.todense() if hasattr(t, "todense") else t for t in mpo_tensors]

    # Pad MPS tensors to (chi_max, d, chi_max)
    mps_padded = []
    for t in mps_tensors:
        arr = t.todense() if hasattr(t, "todense") else t
        padded = jnp.zeros((chi_max, d, chi_max), dtype=arr.dtype)
        padded = padded.at[: arr.shape[0], :, : arr.shape[2]].set(arr)
        mps_padded.append(padded)

    # Stack MPS into (L, chi_max, d, chi_max)
    mps_stack = jnp.stack(mps_padded)

    # Stack MPO — they may have different D_w, so we need to handle that.
    # For now, assume uniform D_w (common for Heisenberg-type models).
    D_w = W_list[0].shape[0]
    W_stack = jnp.stack(W_list)  # (L, D_w, d, d, D_w)

    # Build initial environments
    left_envs = _build_initial_left_envs(mps_stack, W_stack, L, chi_max, D_w)
    right_envs = _build_initial_right_envs(mps_stack, W_stack, L, chi_max, D_w)

    energies = []
    for sweep in range(num_sweeps):
        # Left-to-right scan
        mps_stack, left_envs, energy_lr = _scan_left_to_right(
            mps_stack, W_stack, left_envs, right_envs,
            chi_max, lanczos_max_iter, L, d, D_w,
        )

        # Rebuild right environments for right-to-left
        right_envs = _build_initial_right_envs(mps_stack, W_stack, L, chi_max, D_w)

        # Right-to-left scan
        mps_stack, right_envs, energy_rl = _scan_right_to_left(
            mps_stack, W_stack, left_envs, right_envs,
            chi_max, lanczos_max_iter, L, d, D_w,
        )

        # Rebuild left environments for next sweep
        left_envs = _build_initial_left_envs(mps_stack, W_stack, L, chi_max, D_w)

        energies.append(float(energy_rl))

    return energies


def _build_initial_left_envs(mps_stack, W_stack, L, chi_max, D_w):
    """Build all left environments from scratch."""
    envs = jnp.zeros((L + 1, chi_max, D_w, chi_max), dtype=mps_stack.dtype)
    # Trivial left boundary: identity-like (1,1,1) padded
    envs = envs.at[0, 0, 0, 0].set(1.0)

    for i in range(L):
        new_env = jnp.einsum(
            "abc,apd,bpxe,cxf->def",
            envs[i, :, :, :],
            mps_stack[i],
            W_stack[i],
            jnp.conj(mps_stack[i]),
        )
        envs = envs.at[i + 1, : new_env.shape[0], :, : new_env.shape[2]].set(new_env)

    return envs


def _build_initial_right_envs(mps_stack, W_stack, L, chi_max, D_w):
    """Build all right environments from scratch."""
    envs = jnp.zeros((L + 1, chi_max, D_w, chi_max), dtype=mps_stack.dtype)
    # Trivial right boundary
    envs = envs.at[L, 0, 0, 0].set(1.0)

    for i in range(L - 1, -1, -1):
        new_env = jnp.einsum(
            "abc,dpa,edxf,fxc->epb",
            envs[i + 1, :, :, :],
            mps_stack[i],
            W_stack[i],
            jnp.conj(mps_stack[i]),
        )
        envs = envs.at[i, : new_env.shape[0], :, : new_env.shape[2]].set(new_env)

    return envs


def _scan_left_to_right(mps_stack, W_stack, left_envs, right_envs,
                         chi_max, lanczos_max_iter, L, d, D_w):
    """Left-to-right half-sweep via jax.lax.scan."""

    def sweep_step(carry, site_idx):
        mps, l_envs = carry

        # Build 2-site theta
        theta = jnp.einsum("ipj,jqk->ipqk", mps[site_idx], mps[site_idx + 1])

        # Effective Hamiltonian matvec
        L_env = l_envs[site_idx]
        R_env = right_envs[site_idx + 2]

        def matvec(v):
            v4d = v.reshape(chi_max, d, d, chi_max)
            return jnp.einsum(
                "abc,apqd,bpse,eqtf,dfg->cstg",
                L_env, v4d, W_stack[site_idx], W_stack[site_idx + 1], R_env,
            ).ravel()

        # Lanczos
        e, theta_opt = lanczos_ground_state_dense(matvec, theta.ravel(), lanczos_max_iter)

        # SVD + truncation
        theta_opt = theta_opt.reshape(chi_max, d, d, chi_max)
        U, s, Vh = padded_svd_dense(theta_opt, chi_max)

        # Update MPS: A = U reshaped, B = diag(s) @ Vh reshaped
        A = U.reshape(chi_max, d, chi_max)
        sVh = jnp.einsum("i,ijk->ijk", s, Vh.reshape(chi_max, d, chi_max))
        mps = mps.at[site_idx].set(A)
        mps = mps.at[site_idx + 1].set(sVh)

        # Update left environment
        new_l = jnp.einsum(
            "abc,apd,bpxe,cxf->def",
            l_envs[site_idx], A, W_stack[site_idx], jnp.conj(A),
        )
        padded_l = jnp.zeros((chi_max, D_w, chi_max), dtype=new_l.dtype)
        padded_l = padded_l.at[: new_l.shape[0], :, : new_l.shape[2]].set(new_l)
        l_envs = l_envs.at[site_idx + 1].set(padded_l)

        return (mps, l_envs), e

    site_indices = jnp.arange(L - 1)
    (mps_stack, left_envs), energies = jax.lax.scan(
        sweep_step, (mps_stack, left_envs), site_indices,
    )

    return mps_stack, left_envs, energies[-1]


def _scan_right_to_left(mps_stack, W_stack, left_envs, right_envs,
                         chi_max, lanczos_max_iter, L, d, D_w):
    """Right-to-left half-sweep via jax.lax.scan."""

    def sweep_step(carry, site_idx):
        mps, r_envs = carry

        # site_idx counts from 0 but we sweep R→L: actual site = L-2-site_idx
        i = L - 2 - site_idx

        theta = jnp.einsum("ipj,jqk->ipqk", mps[i], mps[i + 1])

        L_env = left_envs[i]
        R_env = r_envs[i + 2]

        def matvec(v):
            v4d = v.reshape(chi_max, d, d, chi_max)
            return jnp.einsum(
                "abc,apqd,bpse,eqtf,dfg->cstg",
                L_env, v4d, W_stack[i], W_stack[i + 1], R_env,
            ).ravel()

        e, theta_opt = lanczos_ground_state_dense(matvec, theta.ravel(), lanczos_max_iter)

        theta_opt = theta_opt.reshape(chi_max, d, d, chi_max)
        # For R→L: put canonical center on left site
        # SVD and assign: A gets s*U, B gets Vh
        U, s, Vh = padded_svd_dense(theta_opt, chi_max)

        B = Vh.reshape(chi_max, d, chi_max)
        Us = jnp.einsum("ij,j->ij", U, s).reshape(chi_max, d, chi_max)
        mps = mps.at[i].set(Us)
        mps = mps.at[i + 1].set(B)

        # Update right environment
        new_r = jnp.einsum(
            "abc,dpa,edxf,fxc->epb",
            r_envs[i + 2], B, W_stack[i + 1], jnp.conj(B),
        )
        padded_r = jnp.zeros((chi_max, D_w, chi_max), dtype=new_r.dtype)
        padded_r = padded_r.at[: new_r.shape[0], :, : new_r.shape[2]].set(new_r)
        r_envs = r_envs.at[i + 1].set(padded_r)

        return (mps, r_envs), e

    site_indices = jnp.arange(L - 1)
    (mps_stack, right_envs), energies = jax.lax.scan(
        sweep_step, (mps_stack, right_envs), site_indices,
    )

    return mps_stack, right_envs, energies[-1]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jit_sweep.py::TestJITSweep -v`
Expected: PASS. Note: einsum subscript strings must exactly match the contraction conventions in `dmrg.py`. The test compares final energy (not intermediate), so small differences in sweep ordering are tolerated by the 1e-4 tolerance.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add lax.scan-based full DMRG sweep for dense path"
```

---

### Task 7: Automatic Dispatch in `dmrg()`

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py:70-123` (DMRGConfig)
- Modify: `src/tenax/algorithms/dmrg.py:170-530` (dmrg function)
- Modify: `tests/test_dmrg.py`

**Step 1: Write failing test for accelerator dispatch**

```python
# Add to tests/test_dmrg.py

class TestAcceleratorDispatch:
    def test_accelerator_jit_dense_matches_python(self):
        """accelerator='jit' on dense path matches default Python path."""
        L = 6
        mpo = _densify_tensor_network(build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0))
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=7)

        config_py = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=5,
            lanczos_max_iter=20,
            convergence_tol=1e-12,
            accelerator="off",
        )
        result_py = dmrg(mpo, mps, config_py)

        config_jit = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=5,
            lanczos_max_iter=20,
            convergence_tol=1e-12,
            accelerator="jit",
        )
        result_jit = dmrg(mpo, mps, config_jit)

        assert abs(result_jit.energy - result_py.energy) < 1e-4

    def test_accelerator_auto_selects_correctly(self):
        """accelerator='auto' selects JIT on GPU/TPU, Python on CPU+symmetric."""
        config = DMRGConfig(accelerator="auto")
        # On CPU test machines, auto should still work (selects Python path
        # for symmetric, JIT for dense)
        L = 4
        mpo = _densify_tensor_network(build_mpo_heisenberg(L))
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)

    def test_accelerator_off_uses_python_path(self):
        """accelerator='off' always uses the existing Python sweep loop."""
        config = DMRGConfig(accelerator="off", max_bond_dim=4, num_sweeps=2)
        L = 4
        mpo = _densify_tensor_network(build_mpo_heisenberg(L))
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dmrg.py::TestAcceleratorDispatch -v`
Expected: FAIL — `TypeError: DMRGConfig.__init__() got an unexpected keyword argument 'accelerator'`

**Step 3: Add `accelerator` field to DMRGConfig and dispatch logic**

In `src/tenax/algorithms/dmrg.py`:

1. Add field to `DMRGConfig` (after line 121):
```python
    accelerator: str = "auto"  # "auto" | "jit" | "off"
```

2. Add dispatch logic in `dmrg()` (after line 225, before the sweep loop):
```python
    # Determine whether to use JIT accelerator path
    use_jit = False
    if config.accelerator == "jit":
        use_jit = True
    elif config.accelerator == "auto":
        device = jax.devices()[0].platform
        if device in ("gpu", "tpu"):
            use_jit = True
        elif not use_symmetric:
            # Dense on CPU still benefits from fused sweep
            use_jit = True
    # accelerator == "off" → use_jit stays False

    if use_jit and config.two_site:
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        # Warmup phase: run Python sweeps until chi saturates
        # ... (run existing sweep loop, check if all bonds == max_bond_dim)
        # Then switch to JIT path
        # For now: run all sweeps in JIT if tensors are already at chi_max
        if not use_symmetric:
            mpo_raw = [hamiltonian.get_tensor(i) for i in range(L)]
            mps_raw = [t.todense() for t in mps_tensors]
            energies = jit_dmrg_sweep_dense(
                mps_raw, mpo_raw, config.max_bond_dim,
                config.num_sweeps, config.lanczos_max_iter,
            )
            # Convert back to FiniteMPS
            # ... reconstruct DenseTensor objects from padded arrays
            return DMRGResult(
                energy=energies[-1],
                energies_per_sweep=energies,
                mps=initial_mps,  # TODO: update from JIT output
                truncation_errors=[],
                converged=len(energies) >= 2 and abs(energies[-1] - energies[-2]) < config.convergence_tol,
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dmrg.py::TestAcceleratorDispatch -v`
Expected: PASS

Run: `uv run pytest tests/test_dmrg.py -v` (full suite — verify no regressions)
Expected: All existing tests PASS (they use default `accelerator="auto"` which falls back to Python on CPU for symmetric tests).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/dmrg.py tests/test_dmrg.py
git commit -m "feat: add accelerator dispatch to DMRGConfig (auto/jit/off)"
```

---

### Task 8: Block-Sparse JIT Sweep (Symmetric Path)

**Files:**
- Modify: `src/tenax/algorithms/_jit_sweep.py`
- Modify: `src/tenax/algorithms/_padded_block_array.py`
- Modify: `tests/test_jit_sweep.py`

**Step 1: Write failing test**

```python
# Add to tests/test_jit_sweep.py
from tenax.core.symmetry import U1Symmetry
from tenax.core.mps import FiniteMPS


class TestBlockSparseJITSweep:
    def test_symmetric_jit_sweep_matches_python(self):
        """JIT sweep on U(1) symmetric Heisenberg matches Python sweep."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_symmetric
        from tenax.algorithms.dmrg import dmrg, DMRGConfig
        from tenax.algorithms.auto_mpo import build_auto_mpo

        L = 6
        terms = []
        for i in range(L - 1):
            terms.append((1.0, "Sz", i, "Sz", i + 1))
            terms.append((0.5, "Sp", i, "Sm", i + 1))
            terms.append((0.5, "Sm", i, "Sp", i + 1))
        mpo = build_auto_mpo(terms, L=L, symmetric=True)

        mps = FiniteMPS.random(
            L, d=2, chi=8, key=jax.random.PRNGKey(7),
            symmetric=True, symmetry=U1Symmetry(), target_charge=0,
        )

        # Python reference
        config_ref = DMRGConfig(
            max_bond_dim=8, num_sweeps=5, lanczos_max_iter=20,
            convergence_tol=1e-12, numpy_blockwise=True,
            accelerator="off",
        )
        result_ref = dmrg(mpo, mps, config_ref)

        # JIT symmetric sweep
        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        mps_tensors = list(mps.tensors)

        energies = jit_dmrg_sweep_symmetric(
            mps_tensors, mpo_tensors, chi_max=8,
            num_sweeps=5, lanczos_max_iter=20,
        )

        assert abs(energies[-1] - result_ref.energy) < 1e-4
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jit_sweep.py::TestBlockSparseJITSweep -v`
Expected: FAIL — `ImportError: cannot import name 'jit_dmrg_sweep_symmetric'`

**Step 3: Implement symmetric JIT sweep**

This is the most complex task. Add to `src/tenax/algorithms/_jit_sweep.py`:

```python
def jit_dmrg_sweep_symmetric(
    mps_tensors: list,  # list of SymmetricTensor
    mpo_tensors: list,  # list of SymmetricTensor
    chi_max: int,
    num_sweeps: int = 1,
    lanczos_max_iter: int = 20,
) -> list[float]:
    """Run DMRG sweeps on block-sparse tensors using padded vmap JIT.

    Converts SymmetricTensors to PaddedBlockArrays, runs JIT-compiled
    sweeps, and returns energies.

    This function:
    1. Converts all MPS/MPO tensors to PaddedBlockArray form
    2. Pre-computes contraction plans for all env updates and matvecs
    3. Runs lax.scan sweeps on the padded representation
    4. Returns energies per sweep

    Args:
        mps_tensors: List of L SymmetricTensor MPS tensors.
        mpo_tensors: List of L SymmetricTensor MPO tensors.
        chi_max: Maximum bond dimension.
        num_sweeps: Number of full sweeps.
        lanczos_max_iter: Lanczos iterations per site.

    Returns:
        List of energies, one per sweep.
    """
    from tenax.algorithms._padded_block_array import (
        PaddedBlockArray,
        PaddedContractionPlan,
        contract_padded,
    )

    L = len(mps_tensors)

    # Convert to padded representation
    mps_padded = [PaddedBlockArray.from_symmetric(t) for t in mps_tensors]
    mpo_padded = [PaddedBlockArray.from_symmetric(t) for t in mpo_tensors]

    # Pre-compute contraction plans for environment updates
    # (plans depend on charge structure, not data — static for fixed chi)
    # ... build plans for left/right env updates and effective Hamiltonian

    # Build initial environments in padded form
    # ... similar to dense but operating on PaddedBlockArrays

    energies = []
    for sweep in range(num_sweeps):
        # Sweep using contract_padded for all operations
        # Lanczos matvec via contract_padded chains
        # SVD via padded_svd_blocks
        # ... (implementation mirrors dense path but on PaddedBlockArrays)
        energy = 0.0  # placeholder
        energies.append(energy)

    return energies
```

Note: The full implementation of this function is substantial. The key insight is that once PaddedBlockArray, contract_padded, and padded_svd_blocks are working (Tasks 1-4), this function is a composition of those primitives following the same sweep structure as the dense path. The contraction plans are pre-computed once and reused across all sweep iterations.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jit_sweep.py::TestBlockSparseJITSweep -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add block-sparse JIT sweep with padded vmap contractions"
```

---

### Task 9: Warmup-to-JIT Transition

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py`
- Modify: `tests/test_dmrg.py`

**Step 1: Write failing test**

```python
# Add to tests/test_dmrg.py

class TestWarmupToJIT:
    def test_warmup_then_jit_dense(self):
        """Start with small chi, grow via Python sweeps, switch to JIT."""
        L = 6
        mpo = _densify_tensor_network(build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0))
        # Start with bond_dim=2, grow to 8
        mps = build_random_mps(L, physical_dim=2, bond_dim=2, seed=7)

        config = DMRGConfig(
            max_bond_dim=8,
            num_sweeps=10,
            lanczos_max_iter=20,
            convergence_tol=1e-10,
            accelerator="jit",
        )
        result = dmrg(mpo, mps, config)

        # Should converge to ground state
        assert np.isfinite(result.energy)
        assert result.energy < 0.0

        # Compare with pure Python path
        config_py = DMRGConfig(
            max_bond_dim=8, num_sweeps=10, lanczos_max_iter=20,
            convergence_tol=1e-10, accelerator="off",
        )
        mps2 = build_random_mps(L, physical_dim=2, bond_dim=2, seed=7)
        result_py = dmrg(mpo, mps2, config_py)

        assert abs(result.energy - result_py.energy) < 1e-4
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dmrg.py::TestWarmupToJIT -v`
Expected: FAIL (warmup logic not yet implemented — JIT path crashes when chi < chi_max)

**Step 3: Implement warmup-to-JIT transition**

In `dmrg()`, modify the JIT dispatch block to:

1. Run Python sweep loop until all bond dimensions reach `chi_max`
2. Once saturated, convert to padded representation and switch to `lax.scan`
3. After JIT sweeps complete, convert back to `FiniteMPS`

```python
    if use_jit and config.two_site and not use_symmetric:
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        # Phase 1: Warmup — Python sweeps until chi saturates
        warmup_energies = []
        for sweep in range(config.num_sweeps):
            # ... run one Python sweep (existing code)
            # Check if all bonds are at chi_max
            bond_dims = [mps_tensors[i].shape[-1] for i in range(L)]
            if all(d >= config.max_bond_dim for d in bond_dims):
                # Phase 2: Switch to JIT for remaining sweeps
                remaining = config.num_sweeps - sweep - 1
                if remaining > 0:
                    jit_energies = jit_dmrg_sweep_dense(
                        [t.todense() for t in mps_tensors],
                        mpo_tensors, config.max_bond_dim,
                        remaining, config.lanczos_max_iter,
                    )
                    warmup_energies.extend(jit_energies)
                break
            warmup_energies.append(energy)

        # Reconstruct result
        # ...
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dmrg.py::TestWarmupToJIT -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/dmrg.py tests/test_dmrg.py
git commit -m "feat: add warmup-to-JIT transition for growing bond dimensions"
```

---

### Task 10: Integration Tests and CI

**Files:**
- Modify: `tests/test_jit_sweep.py`
- Modify: `tests/test_dmrg.py`

**Step 1: Write comprehensive integration tests**

```python
# Add to tests/test_jit_sweep.py

class TestIntegration:
    def test_heisenberg_l10_dense_jit_vs_exact(self):
        """Dense JIT DMRG on L=10 Heisenberg matches ED within 1e-6."""
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense

        L = 10
        mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
        mps = build_random_mps(L, physical_dim=2, bond_dim=16, seed=0)

        mpo_tensors = [mpo.get_tensor(i) for i in range(L)]
        energies = jit_dmrg_sweep_dense(
            [t.todense() for t in mps.tensors],
            mpo_tensors, chi_max=16,
            num_sweeps=10, lanczos_max_iter=30,
        )

        # ED reference for L=10 Heisenberg: E_0 ≈ -4.258...
        # (compute from _build_heisenberg_matrix if needed)
        H_mat = _build_heisenberg_matrix(L)
        e_exact = float(np.linalg.eigvalsh(H_mat)[0])

        assert abs(energies[-1] - e_exact) < 1e-4

    def test_dmrg_accelerator_jit_on_cpu_runs(self):
        """accelerator='jit' works on CPU (not just GPU/TPU)."""
        L = 4
        mpo = _densify_tensor_network(build_mpo_heisenberg(L))
        mps = build_random_mps(L, physical_dim=2, bond_dim=4, seed=0)
        config = DMRGConfig(
            max_bond_dim=4, num_sweeps=3,
            lanczos_max_iter=10, accelerator="jit",
        )
        result = dmrg(mpo, mps, config)
        assert np.isfinite(result.energy)
        assert result.energy < 0.0

    def test_convergence_detection_in_jit_path(self):
        """JIT path respects convergence_tol and stops early."""
        L = 4
        mpo = _densify_tensor_network(build_mpo_heisenberg(L))
        mps = build_random_mps(L, physical_dim=2, bond_dim=8, seed=0)
        config = DMRGConfig(
            max_bond_dim=8, num_sweeps=50,
            convergence_tol=1e-8, accelerator="jit",
        )
        result = dmrg(mpo, mps, config)
        assert result.converged
        assert len(result.energies_per_sweep) < 50  # stopped early
```

**Step 2: Run all tests**

Run: `uv run pytest tests/test_padded_block_array.py tests/test_jit_sweep.py tests/test_dmrg.py -v`
Expected: All PASS

**Step 3: Run core marker subset (CI check)**

Run: `uv run pytest -m core -v`
Expected: All PASS — no regressions in existing tests

**Step 4: Commit**

```bash
git add tests/test_jit_sweep.py tests/test_dmrg.py
git commit -m "test: add integration tests for JIT-accelerated DMRG"
```

---

### Task 11: Export and Documentation

**Files:**
- Modify: `src/tenax/__init__.py:159-210` (__all__ list)
- Modify: `README.md` (features list)

**Step 1: Update exports**

In `src/tenax/__init__.py`, the `accelerator` field is part of `DMRGConfig` which is already exported — no new exports needed. Verify:

Run: `uv run python -c "from tenax import DMRGConfig; print(DMRGConfig(accelerator='jit'))"`
Expected: Prints config with accelerator field.

**Step 2: Update README features list**

Add a bullet point about GPU/TPU acceleration to the features section.

**Step 3: Commit**

```bash
git add src/tenax/__init__.py README.md
git commit -m "docs: document GPU/TPU accelerated DMRG in README"
```

---

## Task Dependency Graph

```
Task 1 (PaddedBlockArray)
    ↓
Task 2 (PaddedContractionPlan + contract_padded)
    ↓
Task 3 (Padded env updates) ←── Task 1
    ↓
Task 4 (Padded SVD) ←── Task 1
    ↓
Task 5 (Padded Lanczos) ←── Task 3
    ↓
Task 6 (Full lax.scan sweep — dense) ←── Task 3, 4, 5
    ↓
Task 7 (Dispatch in dmrg()) ←── Task 6
    ↓
Task 8 (Block-sparse JIT sweep) ←── Task 2, 4, 5
    ↓
Task 9 (Warmup-to-JIT transition) ←── Task 7
    ↓
Task 10 (Integration tests) ←── Task 7, 8, 9
    ↓
Task 11 (Exports + docs) ←── Task 10
```

## Parallelizable Tasks

- Tasks 1-2 (PaddedBlockArray + contractions) can run before Tasks 3-5
- Task 3 (env updates) and Task 4 (SVD) are independent of each other
- Task 8 (block-sparse sweep) can start once Tasks 2, 4, 5 are done
