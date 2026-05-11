# 2x2 Plaquette Projector — SymmetricTensor Support (#416) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `_compute_2x2_projector` (`src/tenax/algorithms/_ctm_tensor_projector_2x2.py`) to accept `SymmetricTensor` inputs with non-trivial charges. Block-sparse SVDs via `tenax.linalg.svd` with a per-sector gauge-fix helper; dense pipeline retained as the AD-tracer fallback. Unblocks 7 xfailed tests (`TestADSymmetric` × 4, `TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs`, `TestTodenseGradientFlow` × 2).

**Architecture:** Add `_compute_2x2_projector_symmetric` next to the existing dense function. Dispatch at the top of `_compute_2x2_projector`: SymmetricTensor inputs without JAX tracers → new symmetric helper; everything else (DenseTensor, or SymmetricTensor with tracers during AD backward) → existing dense pipeline. New helpers `_gauge_fix_symmetric_svd` and `_scale_bond_by_diag` keep the SVD-per-sector and `sqrt(S)` weighting block-sparse.

**Tech Stack:** `tenax.linalg.svd` (block-sparse, returns `(U_T, s, Vh_T, s_full)`), `SymmetricTensor`, `contract` (label-paired), `_derive_charges` (sector-allocation policy already used by `_svd_projector_symmetric`).

**Spec:** [docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md](../specs/2026-05-11-2x2-projector-symmetric-design.md).

**Predecessor design:** [docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md](../../plans/2026-05-07-ctm-multisite-2x2-projector-design.md) (dense path, PR #406).

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` | **modify** | Add 3 private helpers (`_gauge_fix_symmetric_svd`, `_scale_bond_by_diag`, `_compute_2x2_projector_symmetric`); dispatch at the top of `_compute_2x2_projector`; remove trivial-charge guard; add optional `base_charges` parameter |
| `tests/test_ctm_2x2_projector_symmetric.py` | **create** | New tests for the symmetric branch (gauge-fix helper, closure on non-trivial U(1), base_charges, AD fallback) |
| `tests/test_ipeps.py` | **modify** | Drop 5 `@pytest.mark.xfail(... #416 ...)` decorators (`TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs`, `TestADSymmetric` × 4) |
| `tests/test_fpeps_ad.py` | **modify** | Drop 2 `@pytest.mark.xfail(... #416 ...)` decorators (`TestTodenseGradientFlow::test_symmetric_nontrivial_energy_finite`, `_gradient_finite`) |
| `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md` | **modify** | Update the "Symmetric tensor follow-up" callout (line 48 and §"Open questions") to mark this work shipped |

Total addition: ~280 LOC source + ~250 LOC tests; net deletion of guard (~15 LOC) and 7 xfail markers (~50 LOC).

---

## Task 1: `_gauge_fix_symmetric_svd` helper

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (add helper near `_gauge_fixed_svd` at line 26)
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (create)

**Why this helper:** The dense `_gauge_fixed_svd` (lines 26-50) puts `conj(phase)` on `U` and bare `phase` on `Vh` so that `U @ diag(s) @ Vh == M` exactly (load-bearing for the 2x2 closure `P_bot · P_top = I` per the existing module comment). `tenax.linalg.svd` calls raw `jnp.linalg.svd` per sector without sign-fixing (verified at `src/tenax/linalg.py:206`). We need to apply the same convention per sector.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ctm_2x2_projector_symmetric.py`:

```python
"""Tests for SymmetricTensor support in the 2x2 plaquette projector (#416)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _compute_2x2_projector,
    _gauge_fix_symmetric_svd,
    _scale_bond_by_diag,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _make_test_matrix_tensor(seed: int = 0) -> SymmetricTensor:
    """Build a small 2-leg SymmetricTensor matrix with two U(1) charge sectors."""
    sym = U1Symmetry()
    left_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    right_charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(
        sym, left_charges, FlowDirection.IN, label="left"
    )
    right_idx = TensorIndex.from_charges(
        sym, right_charges, FlowDirection.OUT, label="right"
    )
    return SymmetricTensor.random_normal(
        (left_idx, right_idx), jax.random.PRNGKey(seed)
    )


def test_gauge_fix_symmetric_svd_preserves_reconstruction():
    """After gauge fix, U_T @ diag(S) @ Vh_T == original matrix (per sector)."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=0)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, Vh_fixed = _gauge_fix_symmetric_svd(U_T, Vh_T)

    # Reconstruct: contract U_fixed @ diag(s) @ Vh_fixed; compare to M_T.
    # diag(s) lives on the shared bond — scale U_fixed by s on the bond axis.
    from tenax.contraction.contractor import contract

    U_scaled = _scale_bond_by_diag(U_fixed, s, bond_label="bond")
    M_reconstructed = contract(U_scaled, Vh_fixed)
    np.testing.assert_allclose(
        np.asarray(M_reconstructed.todense()),
        np.asarray(M_T.todense()),
        atol=1e-10,
        err_msg="gauge-fixed SVD must preserve reconstruction U·diag(s)·Vh == M",
    )


def test_gauge_fix_symmetric_svd_real_positive_max_row():
    """After gauge fix, the entry of largest |U[:, j]| is real-positive for every j."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=1)
    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    U_fixed, _ = _gauge_fix_symmetric_svd(U_T, Vh_T)

    # Densify U_fixed (shape: (left_dim, bond_dim)).
    U_dense = np.asarray(U_fixed.todense())
    for j in range(U_dense.shape[1]):
        col = U_dense[:, j]
        if np.max(np.abs(col)) == 0.0:
            continue
        max_row = int(np.argmax(np.abs(col)))
        entry = col[max_row]
        assert entry.imag == pytest.approx(0.0, abs=1e-10), (
            f"column {j}: max-abs entry should be real, got {entry}"
        )
        assert entry.real >= 0.0, (
            f"column {j}: max-abs entry should be non-negative, got {entry}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_preserves_reconstruction -v --no-cov
```

Expected: FAIL with `ImportError: cannot import name '_gauge_fix_symmetric_svd' from 'tenax.algorithms._ctm_tensor_projector_2x2'`.

- [ ] **Step 3: Add the helper to `_ctm_tensor_projector_2x2.py`**

Insert just after `_gauge_fixed_svd` (around line 51):

```python
def _gauge_fix_symmetric_svd(
    U_T: SymmetricTensor, Vh_T: SymmetricTensor
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Per-sector gauge fix for SymmetricTensor SVD outputs.

    Mirrors :func:`_gauge_fixed_svd` (the dense 2x2 gauge convention) at the
    block level: for each kept singular vector j, finds the entry of largest
    ``|U[:, j]|`` across all U-blocks that share its bond charge, rotates U's
    column and Vh's row by ``conj(phase)`` / ``phase`` so that
    ``U @ diag(s) @ Vh == M`` is preserved.  Critical for the 2x2 closure
    ``P_bot · P_top = I`` (no intervening matrix to absorb a ``conj(phase)**2``
    factor — see the doctstring of :func:`_gauge_fixed_svd`).
    """
    bond_idx = U_T.indices[-1]  # last leg of U is the SVD bond
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    # Per global column j, find its (charge, in-sector local index).
    # The sector ordering matches _truncated_svd_symmetric (bond charges
    # are listed in descending-SV global order); within a sector, the
    # local indices are 0..n_q-1 in the order they appear in bond_charges.
    local_index_of: dict[int, dict[int, int]] = {}  # q -> {global_j: local_idx}
    counter: dict[int, int] = {}
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local_index_of.setdefault(q_int, {})[j] = counter.get(q_int, 0)
        counter[q_int] = counter.get(q_int, 0) + 1

    # Collect U-blocks indexed by bond charge (last key entry).
    u_blocks_by_q: dict[int, list[tuple[tuple[int, ...], jax.Array]]] = {}
    for key, block in U_T.blocks.items():
        q = int(key[-1])
        u_blocks_by_q.setdefault(q, []).append((key, block))

    # Collect Vh-blocks indexed by bond charge (FIRST key entry).
    vh_blocks_by_q: dict[int, list[tuple[tuple[int, ...], jax.Array]]] = {}
    for key, block in Vh_T.blocks.items():
        q = int(key[0])
        vh_blocks_by_q.setdefault(q, []).append((key, block))

    new_u_blocks: dict[tuple[int, ...], jax.Array] = {
        key: block for key, block in U_T.blocks.items()
    }
    new_vh_blocks: dict[tuple[int, ...], jax.Array] = {
        key: block for key, block in Vh_T.blocks.items()
    }

    # For each global column j, compute its phase and write it back.
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local = local_index_of[q_int][j]
        u_entries = u_blocks_by_q.get(q_int, [])

        # Find max-abs entry across all U-blocks' local-column `local`.
        best_abs = -1.0
        best_value: complex | float = 1.0
        for _key, block in u_entries:
            # block shape: (left_dims..., n_q); take the slice block[..., local]
            col_slice = block[..., local]
            col_flat = jnp.reshape(col_slice, (-1,))
            local_max_idx = int(jnp.argmax(jnp.abs(col_flat)))
            local_max_val = complex(col_flat[local_max_idx])
            local_max_abs = abs(local_max_val)
            if local_max_abs > best_abs:
                best_abs = local_max_abs
                best_value = local_max_val

        if best_abs <= 0.0:
            phase = 1.0 + 0.0j
        else:
            phase = best_value / abs(best_value)

        conj_phase = jnp.asarray(complex(phase).conjugate())
        bare_phase = jnp.asarray(complex(phase))

        # Apply conj(phase) to column `local` of every matching U-block.
        for key, block in u_entries:
            new_block = new_u_blocks[key]
            new_block = new_block.at[..., local].multiply(conj_phase)
            new_u_blocks[key] = new_block

        # Apply phase to row `local` of every matching Vh-block.
        for key, _block in vh_blocks_by_q.get(q_int, []):
            new_block = new_vh_blocks[key]
            new_block = new_block.at[local, ...].multiply(bare_phase)
            new_vh_blocks[key] = new_block

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out
```

Note on dtype: if the input tensors are real (`float64`), the phase is real (±1), and `at[...].multiply` keeps the dtype real. If complex, the phase is complex and `new_block.at[...].multiply(complex_scalar)` promotes the block to complex. This matches the dense `_gauge_fixed_svd` behaviour.

- [ ] **Step 4: Add `_scale_bond_by_diag` helper**

This helper is needed by Step 1's test (and Step 5/6 of the symmetric pipeline). Insert below `_gauge_fix_symmetric_svd`:

```python
def _scale_bond_by_diag(
    T: SymmetricTensor, diag: jax.Array, bond_label: str
) -> SymmetricTensor:
    """Multiply each block of ``T`` along its ``bond_label`` axis by ``diag``.

    The bond's TensorIndex charges encode each slot's sector; ``diag[j]`` is
    applied to slot ``j`` of every block whose bond key matches sector
    ``bond_charges[j]``.

    Used to express ``T @ diag(s)`` (or ``diag(s) @ T``) in the symmetric
    pipeline without densifying.
    """
    bond_axis = T.labels().index(bond_label)
    bond_idx = T.indices[bond_axis]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    # For each block: identify the sector slice along the bond axis and
    # multiply by the corresponding entries of `diag`.
    new_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in T.blocks.items():
        q = int(key[bond_axis])
        # Slots in the global bond ordering with charge q, in order:
        positions = [j for j, cq in enumerate(bond_charges) if int(cq) == q]
        # The block's bond axis has length len(positions) by construction.
        diag_slice = jnp.asarray(diag)[jnp.array(positions, dtype=np.int32)]
        # Broadcast diag_slice across all non-bond axes.
        shape = [1] * block.ndim
        shape[bond_axis] = len(positions)
        new_blocks[key] = block * diag_slice.reshape(shape)
    return SymmetricTensor._from_blocks_unchecked(new_blocks, T.indices)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_preserves_reconstruction tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_real_positive_max_row -v --no-cov
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "feat(ctm-2x2): add per-sector gauge-fix and bond-scaling helpers for #416"
```

---

## Task 2: `_compute_2x2_projector_symmetric` — direction="left"

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (add new function after `_compute_2x2_projector`)
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append)

**Key derivation (read before implementing):** the dense path collapses leg structure via `.todense().reshape()`, so the matrix-multiply `M_prime = second_half · first_half` works on opaque rows/cols. In the symmetric path, the cut seam (which connects two enlarged corners across M1/M2) carries different labels on the M1 side (`m1_left_labels` from TL's bottom seam) and the M2 side (`m2_right_labels` from BL's top seam) even though they're the same physical leg. So before the M_prime contract, we **rename `m1_left_labels` → `m2_right_labels`** on a copy of `first_half`. After the M_prime SVD, we **never re-rename** because the projectors get their outer-seam labels from the half that retains them (the un-renamed `first_half` for `P_first`, the un-renamed `second_half` for `P_second`).

Flow check (verified from `src/tenax/linalg.py:281-286`): `tensor_svd` returns U with bond flow OUT and Vh with bond flow IN. So:
- `U_M1_T` has `m1_bond` with flow OUT, `Vh_M1_T` has `m1_bond` with flow IN.
- `U_M2_T` has `m2_bond` with flow OUT, `Vh_M2_T` has `m2_bond` with flow IN.
- Inside M_prime construction (prime_order="second_first"): `M_prime = second_half · first_half` where `second_half = sqrt(M2_S) · Vh_M2` (has m2_bond IN) and `first_half = U_M1 · sqrt(M1_S)` (has m1_bond OUT). The cut seam after relabel pairs OUT↔IN automatically because the original Q_TR/Q_BL relabel maps in Step 1 set up matching flows.

- [ ] **Step 1: Write the failing closure test**

Append to `tests/test_ctm_2x2_projector_symmetric.py`:

```python
def _make_symmetric_enlarged_corner(
    chi: int,
    D: int,
    chi_label_a: str,
    chi_label_b: str,
    D2_label_a: str,
    D2_label_b: str,
    flow_chi_a: FlowDirection,
    flow_chi_b: FlowDirection,
    flow_D2_a: FlowDirection,
    flow_D2_b: FlowDirection,
    seed: int,
) -> SymmetricTensor:
    """4-leg enlarged-corner SymmetricTensor with non-trivial U(1) charges.

    Bond charges alternate [0, 1, 0, 1, ...] truncated to each leg's dimension.
    """
    sym = U1Symmetry()
    chi_charges = (np.arange(chi, dtype=np.int32) % 2)
    D2_charges = (np.arange(D**2, dtype=np.int32) % 2)
    indices = (
        TensorIndex.from_charges(sym, chi_charges, flow_chi_a, label=chi_label_a),
        TensorIndex.from_charges(sym, D2_charges, flow_D2_a, label=D2_label_a),
        TensorIndex.from_charges(sym, chi_charges, flow_chi_b, label=chi_label_b),
        TensorIndex.from_charges(sym, D2_charges, flow_D2_b, label=D2_label_b),
    )
    return SymmetricTensor.random_normal(indices, jax.random.PRNGKey(seed))


@pytest.fixture
def symmetric_corners():
    """Return (Q_TL, Q_TR, Q_BL, Q_BR) — 4-leg SymmetricTensors with non-trivial U(1)."""
    chi, D = 4, 2
    Q_TL = _make_symmetric_enlarged_corner(
        chi, D,
        chi_label_a="chi_R", chi_label_b="chi_B",
        D2_label_a="r2", D2_label_b="d2",
        flow_chi_a=FlowDirection.OUT, flow_chi_b=FlowDirection.OUT,
        flow_D2_a=FlowDirection.OUT, flow_D2_b=FlowDirection.OUT,
        seed=0,
    )
    Q_TR = _make_symmetric_enlarged_corner(
        chi, D,
        chi_label_a="chi_L", chi_label_b="chi_B",
        D2_label_a="l2", D2_label_b="d2",
        flow_chi_a=FlowDirection.IN, flow_chi_b=FlowDirection.OUT,
        flow_D2_a=FlowDirection.IN, flow_D2_b=FlowDirection.OUT,
        seed=1,
    )
    Q_BL = _make_symmetric_enlarged_corner(
        chi, D,
        chi_label_a="chi_R", chi_label_b="chi_T",
        D2_label_a="r2", D2_label_b="u2",
        flow_chi_a=FlowDirection.OUT, flow_chi_b=FlowDirection.IN,
        flow_D2_a=FlowDirection.OUT, flow_D2_b=FlowDirection.IN,
        seed=2,
    )
    Q_BR = _make_symmetric_enlarged_corner(
        chi, D,
        chi_label_a="chi_L", chi_label_b="chi_T",
        D2_label_a="l2", D2_label_b="u2",
        flow_chi_a=FlowDirection.IN, flow_chi_b=FlowDirection.IN,
        flow_D2_a=FlowDirection.IN, flow_D2_b=FlowDirection.IN,
        seed=3,
    )
    return Q_TL, Q_TR, Q_BL, Q_BR


def test_compute_2x2_projector_symmetric_closure_left(symmetric_corners):
    """Symmetric path: `P_bot · P_top = I_chi_new` (closure check)."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
    )
    # closure: contract P_bot with P_top via shared (chi_outer, fused_D2) labels.
    I_tensor = contract(P_bot, P_top)
    I_dense = np.asarray(I_tensor.todense())
    chi_new = P_top.indices[2].dim
    assert I_dense.shape == (chi_new, chi_new)
    np.testing.assert_allclose(
        I_dense, np.eye(chi_new), atol=1e-9,
        err_msg="P_bot · P_top must be identity on the truncated chi_new bond"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_symmetric_closure_left -v --no-cov
```

Expected: FAIL with the existing `NotImplementedError: _compute_2x2_projector currently supports trivial-charge tensors only;…` (until Task 4's dispatch is in place, the function still raises). To make this test runnable in isolation, **temporarily call `_compute_2x2_projector_symmetric` directly from the test** during development:

```python
P_top, P_bot = _compute_2x2_projector_symmetric(
    Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
)
```

Switch back to the public `_compute_2x2_projector` API once Task 4's dispatch lands.

- [ ] **Step 3: Add `_compute_2x2_projector_symmetric` (all four directions)**

Insert after `_compute_2x2_projector` (end of file). The implementation has six pipeline stages that mirror the dense path:

1. **Form M1, M2** — relabel inputs per direction; contract enlarged corner pairs.
2. **SVDs of M1, M2** — `tensor_svd` + per-sector `_gauge_fix_symmetric_svd`.
3. **Form halves** — `U·sqrt(S)` / `sqrt(S)·Vh` via `_scale_bond_by_diag`; normalize.
4. **Form M_prime** — rename cut seam labels so `contract` pairs them; SVD with truncation.
5. **Cross-projectors** — `contract(first_half, Vh_Mp.bar())` and `contract(U_Mp.bar(), second_half)`, then scale by `S^{-1/2}`.
6. **Rename to public labels** — chi_outer / fused_D2 / chi_new_top / chi_new_bot.

**Flow analysis (load-bearing, verify before coding):**
- `tensor_svd` returns U with bond flow OUT, Vh with bond flow IN (`src/tenax/linalg.py:281-286`).
- `Vh_T.bar()` flips all flows AND conjugates data — exactly the linear-algebra `Vh^†` for matrix multiplication when expressed as a tensor contract.
- For prime_order="second_first": M_prime_T = contract(second_half[m2_bond=IN, m2_right=IN], first_half_for_mp[m1_left_renamed=OUT, m1_bond=OUT]). Cut seam pairs m1_left_renamed(OUT) ↔ m2_right(IN) ✅. M_prime_T labels (m2_bond[IN], m1_bond[OUT]).
- After SVD of M_prime: U_Mp_T (m2_bond[IN], chi_new[OUT]); Vh_Mp_T (chi_new[IN], m1_bond[OUT]).
- contract(first_half[..., m1_bond=OUT], Vh_Mp_T.bar()[chi_new=OUT, m1_bond=IN]) → pairs m1_bond OUT↔IN ✅. Result has chi_new[OUT] free — matches the desired `chi_new_top[OUT]` output flow.
- contract(U_Mp_T.bar()[m2_bond=OUT, chi_new=IN], second_half[m2_bond=IN, ...]) → pairs m2_bond OUT↔IN ✅. Result has chi_new[IN] free — matches `chi_new_bot[IN]` output flow.

```python
def _compute_2x2_projector_symmetric(
    Q_TL: SymmetricTensor,
    Q_TR: SymmetricTensor,
    Q_BL: SymmetricTensor,
    Q_BR: SymmetricTensor,
    chi: int,
    *,
    direction: str,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Block-sparse 2x2 Fishman projector for SymmetricTensor inputs.

    Mirrors the dense pipeline in :func:`_compute_2x2_projector` stage-for-stage
    via :func:`tenax.linalg.svd` and the per-sector gauge-fix helper
    :func:`_gauge_fix_symmetric_svd`.

    See ``docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md``
    for the cut-seam relabel rationale and flow conventions.

    Returns:
        ``(P_top, P_bot)`` SymmetricTensor projectors with the same label /
        flow conventions as the dense path's output.
    """
    from tenax.contraction.contractor import contract
    from tenax.linalg import svd as tensor_svd

    if direction not in ("left", "right", "top", "bottom"):
        raise ValueError(f"unsupported direction={direction!r}")

    # ---- Stage 1: form M1, M2 as 4-leg SymmetricTensors. ----
    if direction in ("left", "right"):
        Q_TL_relab = Q_TL.relabels({"chi_B": "chi_B_TL", "d2": "d2_TL"})
        Q_TR_relab = Q_TR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_B": "chi_B_TR", "d2": "d2_TR"}
        )
        M1_T = contract(Q_TL_relab, Q_TR_relab)
        m1_left_labels = ("chi_B_TL", "d2_TL")
        m1_right_labels = ("chi_B_TR", "d2_TR")

        Q_BR_relab = Q_BR.relabels(
            {"chi_L": "chi_R", "l2": "r2", "chi_T": "chi_T_BR", "u2": "u2_BR"}
        )
        Q_BL_relab = Q_BL.relabels({"chi_T": "chi_T_BL", "u2": "u2_BL"})
        M2_T = contract(Q_BR_relab, Q_BL_relab)
        m2_left_labels = ("chi_T_BR", "u2_BR")
        m2_right_labels = ("chi_T_BL", "u2_BL")
    else:  # "top", "bottom"
        Q_BL_relab = Q_BL.relabels(
            {"chi_T": "chi_B", "u2": "d2", "chi_R": "chi_R_BL", "r2": "r2_BL"}
        )
        Q_TL_relab = Q_TL.relabels({"chi_R": "chi_R_TL", "r2": "r2_TL"})
        M1_T = contract(Q_BL_relab, Q_TL_relab)
        m1_left_labels = ("chi_R_BL", "r2_BL")
        m1_right_labels = ("chi_R_TL", "r2_TL")

        Q_TR_relab = Q_TR.relabels(
            {"chi_B": "chi_T", "d2": "u2", "chi_L": "chi_L_TR", "l2": "l2_TR"}
        )
        Q_BR_relab = Q_BR.relabels({"chi_L": "chi_L_BR", "l2": "l2_BR"})
        M2_T = contract(Q_TR_relab, Q_BR_relab)
        m2_left_labels = ("chi_L_TR", "l2_TR")
        m2_right_labels = ("chi_L_BR", "l2_BR")

    # ---- Stage 2: SVDs of M1, M2 with per-sector gauge fix. ----
    U_M1_T, M1_S, Vh_M1_T, _ = tensor_svd(
        M1_T,
        left_labels=m1_left_labels,
        right_labels=m1_right_labels,
        new_bond_label="m1_bond",
        max_singular_values=None,
    )
    U_M1_T, Vh_M1_T = _gauge_fix_symmetric_svd(U_M1_T, Vh_M1_T)
    M1_S = _fishman_truncate_S(M1_S, eps=1e-12)

    U_M2_T, M2_S, Vh_M2_T, _ = tensor_svd(
        M2_T,
        left_labels=m2_left_labels,
        right_labels=m2_right_labels,
        new_bond_label="m2_bond",
        max_singular_values=None,
    )
    U_M2_T, Vh_M2_T = _gauge_fix_symmetric_svd(U_M2_T, Vh_M2_T)
    M2_S = _fishman_truncate_S(M2_S, eps=1e-12)

    M1_sqrtS = jnp.sqrt(M1_S)
    M2_sqrtS = jnp.sqrt(M2_S)

    # ---- Stage 3: form halves, normalize. ----
    if direction in ("left", "bottom"):
        # M_prime = second_half · first_half.
        # first_half  = U_M1 · sqrt(M1_S)   labels: (m1_left_labels..., m1_bond=OUT)
        # second_half = sqrt(M2_S) · Vh_M2  labels: (m2_bond=IN, m2_right_labels...)
        first_half = _scale_bond_by_diag(U_M1_T, M1_sqrtS, bond_label="m1_bond")
        second_half = _scale_bond_by_diag(Vh_M2_T, M2_sqrtS, bond_label="m2_bond")
        prime_order = "second_first"
        first_outer_labels = m1_left_labels
        second_outer_labels = m2_right_labels
    else:  # "right", "top"
        # M_prime = first_half · second_half.
        # first_half  = sqrt(M1_S) · Vh_M1  labels: (m1_bond=IN, m1_right_labels...)
        # second_half = U_M2 · sqrt(M2_S)   labels: (m2_left_labels..., m2_bond=OUT)
        first_half = _scale_bond_by_diag(Vh_M1_T, M1_sqrtS, bond_label="m1_bond")
        second_half = _scale_bond_by_diag(U_M2_T, M2_sqrtS, bond_label="m2_bond")
        prime_order = "first_second"
        first_outer_labels = m1_right_labels
        second_outer_labels = m2_left_labels

    first_norm = jnp.sqrt(jnp.sum(M1_S) + 1e-30)
    second_norm = jnp.sqrt(jnp.sum(M2_S) + 1e-30)
    first_half = _scale_bond_by_diag(
        first_half, jnp.ones_like(M1_S) / first_norm, bond_label="m1_bond"
    )
    second_half = _scale_bond_by_diag(
        second_half, jnp.ones_like(M2_S) / second_norm, bond_label="m2_bond"
    )

    # ---- Stage 4: form M_prime by renaming cut seam + contract; SVD M_prime. ----
    if prime_order == "second_first":
        # Cut seam: first_half's m1_left_labels ↔ second_half's m2_right_labels.
        cut_relabel = dict(zip(m1_left_labels, m2_right_labels))
        first_half_for_mp = first_half.relabels(cut_relabel)
        M_prime_T = contract(second_half, first_half_for_mp)
        mp_left_labels = ("m2_bond",)
        mp_right_labels = ("m1_bond",)
    else:
        cut_relabel = dict(zip(m1_right_labels, m2_left_labels))
        first_half_for_mp = first_half.relabels(cut_relabel)
        M_prime_T = contract(first_half_for_mp, second_half)
        mp_left_labels = ("m1_bond",)
        mp_right_labels = ("m2_bond",)

    U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
        M_prime_T,
        left_labels=mp_left_labels,
        right_labels=mp_right_labels,
        new_bond_label="chi_new",
        max_singular_values=chi,
    )
    U_Mp_T, Vh_Mp_T = _gauge_fix_symmetric_svd(U_Mp_T, Vh_Mp_T)

    # ---- Stage 5: cross-projectors via bar() for the SVD adjoint. ----
    # bar() = element-wise conjugate + flip ALL flows.  For the matrix-multiply
    # contract first_half · Vh_Mp^†, bar(Vh_Mp_T) gives us conjugation AND the
    # right flow on the contracted m1_bond (OUT after flip; pairs with
    # first_half's m1_bond[OUT]?  NO, first_half is m1_bond[OUT] still — but bar
    # also flips IN to OUT, so Vh_Mp_T (m1_bond originally OUT from M_prime_T)
    # → bar gives m1_bond IN.  OUT(first_half) ↔ IN(bar(Vh_Mp_T)) pairs ✅.
    #
    # For "first_second" path: first_half has m1_bond[IN] (from Vh_M1_T).
    # Vh_Mp_T's m1_bond was originally IN (left_labels in SVD → bond IN; wait
    # no — left_labels of M_prime SVD = mp_left_labels = ("m1_bond",) for
    # "first_second", so U_Mp_T has m1_bond[IN] and Vh_Mp_T has m2_bond[OUT].
    # Stage 5 contracts first_half (m1_bond IN) with U_Mp_T.bar() (m1_bond
    # OUT after flip) → pairs IN↔OUT ✅.
    #
    # Conclusion: in both prime orders, bar() on the half OPPOSITE the
    # contracted SVD bond is the correct operation.
    s_max = float(jnp.max(S_Mp))
    cutoff = 1e-12 * (s_max + 1e-30)
    mask = S_Mp > cutoff
    S_safe = jnp.where(mask, S_Mp, 1.0)
    S_inv_sqrt = jnp.where(mask, 1.0 / jnp.sqrt(S_safe), 0.0)

    if prime_order == "second_first":
        # P_first  = first_half · V_Mp · S^{-1/2}   = contract(first_half, Vh_Mp.bar())
        #            on m1_bond → free axes (m1_left_labels..., chi_new[OUT])
        # P_second = S^{-1/2} · U_Mp^† · second_half = contract(U_Mp.bar(), second_half)
        #            on m2_bond → free axes (chi_new[IN], m2_right_labels...)
        P_first_unscaled = contract(first_half, Vh_Mp_T.bar())
        P_second_unscaled = contract(U_Mp_T.bar(), second_half)
    else:  # "first_second"
        # P_first  = U_Mp^† · first_half · S^{-1/2}  pattern reverses.
        # Symmetric of above:
        P_first_unscaled = contract(U_Mp_T.bar(), first_half)
        P_second_unscaled = contract(second_half, Vh_Mp_T.bar())

    P_first = _scale_bond_by_diag(P_first_unscaled, S_inv_sqrt, bond_label="chi_new")
    P_second = _scale_bond_by_diag(P_second_unscaled, S_inv_sqrt, bond_label="chi_new")

    # ---- Stage 6: relabel and reorder axes to match the dense path's output. ----
    # Dense conventions:
    #   P_top labels: (chi_outer[IN], fused_D2[IN], chi_new_top[OUT])
    #   P_bot labels: (chi_new_bot[IN], chi_outer[OUT], fused_D2[OUT])
    #
    # In our symmetric path, P_first carries one outer seam, P_second the other.
    # The mapping P_first/P_second → P_top/P_bot follows prime_order.
    if prime_order == "second_first":
        # P_first owns the M1 side (first_outer_labels) — that's the "top" seam.
        # P_second owns the M2 side (second_outer_labels) — the "bottom" seam.
        P_top_unwrapped, top_outer = P_first, first_outer_labels
        P_bot_unwrapped, bot_outer = P_second, second_outer_labels
    else:
        # prime_order="first_second": P_second owns the "top" seam.
        P_top_unwrapped, top_outer = P_second, second_outer_labels
        P_bot_unwrapped, bot_outer = P_first, first_outer_labels

    # Rename outer-seam labels → (chi_outer, fused_D2). Per the corner-spec
    # convention, the first of each `*_outer_labels` pair is the chi seam and
    # the second is the D² seam.
    chi_lbl_top, D2_lbl_top = top_outer
    chi_lbl_bot, D2_lbl_bot = bot_outer

    P_top = P_top_unwrapped.relabels({
        chi_lbl_top: "chi_outer",
        D2_lbl_top: "fused_D2",
        "chi_new": "chi_new_top",
    })
    P_bot = P_bot_unwrapped.relabels({
        chi_lbl_bot: "chi_outer",
        D2_lbl_bot: "fused_D2",
        "chi_new": "chi_new_bot",
    })

    # Final axis order: P_top=(chi_outer, fused_D2, chi_new_top);
    #                   P_bot=(chi_new_bot, chi_outer, fused_D2).
    P_top = P_top.transpose(
        tuple(P_top.labels().index(lbl) for lbl in ("chi_outer", "fused_D2", "chi_new_top"))
    )
    P_bot = P_bot.transpose(
        tuple(P_bot.labels().index(lbl) for lbl in ("chi_new_bot", "chi_outer", "fused_D2"))
    )
    return P_top, P_bot
```

- [ ] **Step 4: Run the closure test**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_symmetric_closure_left -v --no-cov
```

Expected: PASS (closure error < 1e-9).

**Debugging guide if closure fails:**
- **Closure error >> 1e-3** — wrong contract orientation (Stage 5 picks the wrong half to `bar()`). Print `P_first.labels()` and `P_second.labels()` after Stage 5; they should be (m1_left..., chi_new) and (chi_new, m2_right...) respectively for prime_order="second_first".
- **Closure error ~1e-3 to 1e-6** — possible flow mismatch on the cut seam. Print `M_prime_T.indices` and verify the flows pair OUT↔IN.
- **`contract` raises a flow-mismatch error** — fix the bar() placement (use bar on the OTHER half).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "feat(ctm-2x2): symmetric branch — direction='left' closure (#416)"
```

---

## Task 3: Extend to all four directions

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (the four-way branch inside `_compute_2x2_projector_symmetric` should already be in place from Task 2 if implemented correctly; verify with tests for the other three directions)
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append)

- [ ] **Step 1: Write parametrized closure tests for the three remaining directions**

The `symmetric_corners` fixture from Task 2 is reused via pytest injection. Append to `tests/test_ctm_2x2_projector_symmetric.py`:

```python
@pytest.mark.parametrize("direction", ["right", "top", "bottom"])
def test_compute_2x2_projector_symmetric_closure_other_directions(
    symmetric_corners, direction
):
    """Closure test for direction in {right, top, bottom}."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4
    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction=direction
    )
    I_tensor = contract(P_bot, P_top)
    I_dense = np.asarray(I_tensor.todense())
    chi_new = P_top.indices[2].dim
    np.testing.assert_allclose(
        I_dense, np.eye(chi_new), atol=1e-9,
        err_msg=f"P_bot · P_top must be identity for direction={direction!r}"
    )
```

- [ ] **Step 2: Run the three new tests**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py -v --no-cov -k "closure"
```

Expected: 4 passed (left + right + top + bottom).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_2x2_projector_symmetric.py src/tenax/algorithms/_ctm_tensor_projector_2x2.py
git commit -m "test(ctm-2x2): closure tests for all four symmetric directions (#416)"
```

---

## Task 4: Dispatch + drop trivial-charge guard

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:240-260` (replace guard with dispatch)

- [ ] **Step 1: Replace the trivial-charge guard with dispatch logic**

In `_compute_2x2_projector`, replace lines 240-260 (`Trivial-charge guard:` block) with:

```python
    # Dispatch: SymmetricTensor inputs (without JAX tracers in any block) go
    # through the block-sparse path.  Tracer-bearing symmetric blocks (AD
    # backward) and DenseTensor inputs fall through to the dense pipeline.
    if any(isinstance(q, SymmetricTensor) for q in (Q_TL, Q_TR, Q_BL, Q_BR)):
        def _has_tracer(t: Tensor) -> bool:
            if isinstance(t, SymmetricTensor):
                return any(
                    isinstance(b, jax.core.Tracer) for b in t.blocks.values()
                )
            return isinstance(getattr(t, "_data", None), jax.core.Tracer)

        if not any(_has_tracer(q) for q in (Q_TL, Q_TR, Q_BL, Q_BR)):
            return _compute_2x2_projector_symmetric(
                Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction
            )
        # Tracer-bearing symmetric path → fall through to dense (densify).
```

Add the `SymmetricTensor` and `Tensor` imports if not already present at the top (they should be — line 21).

- [ ] **Step 2: Run the full 2x2 test suite**

```bash
uv run pytest tests/test_ctm_2x2_projector.py tests/test_ctm_2x2_projector_symmetric.py -v --no-cov
```

Expected: all dense tests pass (no regression), all symmetric closure tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm-2x2): dispatch SymmetricTensor inputs to symmetric branch (#416)"
```

---

## Task 5: Add `base_charges` parameter and per-sector allocation

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (extend `_compute_2x2_projector` signature; thread through to symmetric helper)
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_compute_2x2_projector_symmetric_base_charges_drive_chi_new(symmetric_corners):
    """When base_charges is supplied, chi_new charges match _derive_charges(base_charges, chi)."""
    from tenax.algorithms._ctm_utils import _derive_charges

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4
    base_charges = np.array([0, 1, 0, 1], dtype=np.int32)

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=chi, direction="left", base_charges=base_charges
    )
    expected_chi_new = _derive_charges(base_charges, P_top.indices[2].dim)
    actual_chi_new = np.asarray(P_top.indices[2].charges, dtype=np.int32)
    np.testing.assert_array_equal(actual_chi_new, expected_chi_new)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_symmetric_base_charges_drive_chi_new -v --no-cov
```

Expected: FAIL with `TypeError: _compute_2x2_projector() got an unexpected keyword argument 'base_charges'`.

- [ ] **Step 3: Add `base_charges` to both function signatures**

Modify `_compute_2x2_projector` signature (line 164):

```python
def _compute_2x2_projector(
    Q_TL: Tensor,
    Q_TR: Tensor,
    Q_BL: Tensor,
    Q_BR: Tensor,
    chi: int,
    *,
    direction: str = "left",
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """... (existing docstring; add a line under Args:)
        base_charges: Optional charge tile (typically A.indices[0].charges).
            When supplied, the SymmetricTensor branch allocates chi_new per
            sector via _derive_charges; ignored on the dense path.
    """
```

Thread it through to `_compute_2x2_projector_symmetric` in the dispatch block from Task 4.

Inside `_compute_2x2_projector_symmetric`, change the M_prime SVD call (Task 2 Step 3) from:

```python
U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
    M_prime_T,
    left_labels=mp_left_labels,
    right_labels=mp_right_labels,
    new_bond_label="chi_new",
    max_singular_values=chi,
)
```

to a custom truncation that honors `base_charges` (since `tenax.linalg.svd` has only global top-k). Easiest: do the SVD without truncation, then re-truncate per `_derive_charges` mapping.

```python
if base_charges is None:
    U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
        M_prime_T,
        left_labels=mp_left_labels,
        right_labels=mp_right_labels,
        new_bond_label="chi_new",
        max_singular_values=chi,
    )
else:
    # Full-spectrum SVD, then per-sector re-truncation.
    U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
        M_prime_T,
        left_labels=mp_left_labels,
        right_labels=mp_right_labels,
        new_bond_label="chi_new",
        max_singular_values=None,
    )
    U_Mp_T, S_Mp, Vh_Mp_T = _retruncate_by_base_charges(
        U_Mp_T, S_Mp, Vh_Mp_T, base_charges=base_charges, chi=chi
    )
```

Add the `_retruncate_by_base_charges` helper. It does what `_svd_projector_symmetric` does at lines 683-710 (per-sector allocation):

```python
def _retruncate_by_base_charges(
    U_T: SymmetricTensor,
    S: jax.Array,
    Vh_T: SymmetricTensor,
    *,
    base_charges: np.ndarray,
    chi: int,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor]:
    """Re-truncate a full SymmetricTensor SVD to chi entries with per-sector allocation.

    Allocates target counts via `_derive_charges(base_charges, chi)`; greedy
    top-k fills any remaining budget across sectors.  Mirrors the per-sector
    allocation logic in `_svd_projector_symmetric` (`_ctm_projector.py:683-710`).
    """
    from tenax.algorithms._ctm_utils import _derive_charges

    bond_charges_full = np.asarray(U_T.indices[-1].charges, dtype=np.int32)
    target_charges = _derive_charges(base_charges, chi)
    target_count: dict[int, int] = {}
    for q in target_charges:
        target_count[int(q)] = target_count.get(int(q), 0) + 1

    # Within each sector, the SVD bond is already in descending-SV global order.
    # _truncated_svd_symmetric sorts per-sector singular values within the global
    # array, so we need the (sector, in-sector index) for each global j.
    # Build it here:
    in_sector_idx_of: dict[int, list[int]] = {}
    for j, q in enumerate(bond_charges_full):
        q_int = int(q)
        in_sector_idx_of.setdefault(q_int, []).append(j)

    keep_global: list[int] = []
    used: dict[int, int] = {}
    # First pass: honor target_count per sector (top entries within each sector).
    for q, want in sorted(target_count.items()):
        slots = in_sector_idx_of.get(q, [])
        take = min(want, len(slots))
        keep_global.extend(slots[:take])
        used[q] = take

    # Second pass: fill remaining budget greedily from any unused entry.
    remaining = chi - len(keep_global)
    if remaining > 0:
        used_set = set(keep_global)
        for j in range(len(bond_charges_full)):
            if remaining <= 0:
                break
            if j not in used_set:
                keep_global.append(j)
                used_set.add(j)
                remaining -= 1

    keep_global.sort()  # preserve descending-SV order broken? — see note
    # NOTE: keeping descending-SV order is not strictly required (the projector
    # is invariant under permutations of the chi_new bond as long as U/Vh slots
    # match).  But sorting ascending makes downstream rebuilds simpler.

    # Rebuild U_T, S, Vh_T over the kept indices.
    # Build new bond_charges:
    new_bond_charges = bond_charges_full[np.asarray(keep_global, dtype=np.int32)]
    S_new = jnp.asarray(S)[jnp.asarray(keep_global)]

    # Rebuild blocks by extracting kept-column slices of each block.
    # For each U-block (lk, q): its bond axis runs over the entries of
    # `in_sector_idx_of[q]` in order.  The kept entries for that block are
    # those keep_global slots whose charge is q; their in-block position
    # is their index within `in_sector_idx_of[q]`.
    sym = U_T.indices[0].symmetry
    new_bond_out = TensorIndex.from_charges(
        sym, new_bond_charges, FlowDirection.OUT, label=U_T.indices[-1].label
    )
    new_bond_in = TensorIndex.from_charges(
        sym, new_bond_charges, FlowDirection.IN, label=Vh_T.indices[0].label
    )
    new_U_indices = U_T.indices[:-1] + (new_bond_out,)
    new_Vh_indices = (new_bond_in,) + Vh_T.indices[1:]

    new_U_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in U_T.blocks.items():
        q = int(key[-1])
        in_sector_positions = in_sector_idx_of.get(q, [])
        kept_within_sector = [
            pos for pos, j in enumerate(in_sector_positions) if j in set(keep_global)
        ]
        if not kept_within_sector:
            continue
        idx_arr = jnp.array(kept_within_sector, dtype=np.int32)
        new_U_blocks[key] = jnp.take(block, idx_arr, axis=-1)

    new_Vh_blocks: dict[tuple[int, ...], jax.Array] = {}
    for key, block in Vh_T.blocks.items():
        q = int(key[0])
        in_sector_positions = in_sector_idx_of.get(q, [])
        kept_within_sector = [
            pos for pos, j in enumerate(in_sector_positions) if j in set(keep_global)
        ]
        if not kept_within_sector:
            continue
        idx_arr = jnp.array(kept_within_sector, dtype=np.int32)
        new_Vh_blocks[key] = jnp.take(block, idx_arr, axis=0)

    return (
        SymmetricTensor._from_blocks_unchecked(new_U_blocks, new_U_indices),
        S_new,
        SymmetricTensor._from_blocks_unchecked(new_Vh_blocks, new_Vh_indices),
    )
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_symmetric_base_charges_drive_chi_new -v --no-cov
```

Expected: PASS.

- [ ] **Step 5: Closure tests still pass with the re-truncation path**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py -v --no-cov
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "feat(ctm-2x2): per-sector chi_new allocation via base_charges (#416)"
```

---

## Task 6: AD fallback (tracer dispatch)

**Files:**
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append)

The dispatch in Task 4 already routes tracer-bearing SymmetricTensor inputs to the dense path. Verify with a `jax.grad`-style test.

- [ ] **Step 1: Write the failing test**

```python
def test_compute_2x2_projector_symmetric_ad_fallback_passes_tracer(symmetric_corners):
    """SymmetricTensor inputs under jax.grad must dispatch to the dense fallback."""
    import jax

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    chi = 4

    def scalar_of(seed_offset):
        # Trivial parameterization: scale one block of Q_TL by `t`, then
        # call the projector and return a scalar function of the output.
        t = seed_offset
        keys = sorted(Q_TL.blocks.keys())
        first_key = keys[0]
        new_block = Q_TL.blocks[first_key] * t
        new_blocks = {**Q_TL.blocks, first_key: new_block}
        Q_TL_perturbed = SymmetricTensor._from_blocks_unchecked(
            new_blocks, Q_TL.indices
        )
        P_top, P_bot = _compute_2x2_projector(
            Q_TL_perturbed, Q_TR, Q_BL, Q_BR, chi=chi, direction="left"
        )
        return jnp.sum(P_top.todense()**2) + jnp.sum(P_bot.todense()**2)

    # jax.grad should not raise.  The value of the gradient is not asserted;
    # this test is for "the AD path doesn't crash".
    grad = jax.grad(scalar_of)(1.0)
    assert jnp.isfinite(grad), f"grad must be finite, got {grad}"
```

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_symmetric_ad_fallback_passes_tracer -v --no-cov
```

Expected: PASS (the dispatch from Task 4 already does the right thing — densifies under tracing).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_2x2_projector_symmetric.py
git commit -m "test(ctm-2x2): AD fallback dispatch on tracer-bearing symmetric input (#416)"
```

---

## Task 7: Drop xfail decorators on the 7 blocked tests

**Files:**
- Modify: `tests/test_ipeps.py` (5 decorators)
- Modify: `tests/test_fpeps_ad.py` (2 decorators)

- [ ] **Step 1: Find the xfail decorators**

```bash
grep -nB1 "Issue #416\|#416" tests/test_ipeps.py tests/test_fpeps_ad.py
```

For each decorator block matching the form:
```python
@pytest.mark.xfail(
    strict=False,
    reason=(
        "_compute_2x2_projector lacks symmetric-tensor support (PR #406, "
        "design doc docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md); "
        "tracked as Issue #416."
    ),
)
```

remove the entire 6-7-line decorator (and its trailing blank line if there is one).

- [ ] **Step 2: Run the 7 previously-xfailed tests**

```bash
uv run pytest -v --no-cov \
  tests/test_ipeps.py::TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_runs \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_energy_decreases \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_matches_dense \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type \
  tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_energy_finite \
  tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite
```

Expected: all 7 pass.

**Contingency:** if any test fails for a reason unrelated to #416 (e.g. a numeric tolerance regression on the optimizer path), file a fresh issue with a specific failure note and re-xfail just that test, citing the new issue.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ipeps.py tests/test_fpeps_ad.py
git commit -m "test: unxfail 7 tests previously blocked on #416"
```

---

## Task 8: Full core test suite

**Files:** (none — verification only)

- [ ] **Step 1: Run full core suite**

```bash
uv run pytest -m core --no-cov -q
```

Expected: all green. The dense 2x2 path tests (`test_ctm_2x2_projector.py`) must pass without change (no regression on the dispatch + guard removal). All new symmetric tests must pass. The 7 unxfailed tests must pass.

- [ ] **Step 2: Run the algorithm suite (slow, ~5-10 min)**

```bash
uv run pytest -m algorithm --no-cov -q
```

Expected: all green (or no new failures vs `main`). The 7 unxfailed tests live in the algorithm suite.

---

## Task 9: Docs + memory update

**Files:**
- Modify: `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md` (mark §"Symmetric tensor follow-up" as shipped)
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_2x2_projector_handoff.md` (update with shipped status)
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/MEMORY.md` (one-line update)

- [ ] **Step 1: Update the predecessor design doc**

In `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md:48` (the `* SymmetricTensor support for the 2×2 path...` bullet) and `:254` (the "Symmetric tensor follow-up" bullet), change the text from "follow-up" to "shipped — see `docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md` and PR #XXX" (fill in the PR number after pushing).

- [ ] **Step 2: Update the memory file**

Edit `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_2x2_projector_handoff.md`:

Add a line at the top:
```markdown
**Update (2026-05-11):** Symmetric-tensor support for `_compute_2x2_projector` shipped in PR #XXX (#416 closed).  variPEPS-parity gap (-0.255) still deferred to M2b honeycomb-native CTM.
```

Update `MEMORY.md` line for `project_2x2_projector_handoff.md`: replace the existing one-line summary with a version that mentions #416 closure.

- [ ] **Step 3: Commit**

```bash
git add docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md
git commit -m "docs(ctm-2x2): mark symmetric tensor follow-up shipped (#416)"
```

Memory files do not go through git (they live outside the repo).

---

## Task 10: Open PR closing #416

**Files:** (none — git/gh only)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/2x2-projector-symmetric
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat(ctm-2x2): SymmetricTensor support for _compute_2x2_projector (#416)" --body "$(cat <<'EOF'
## Summary

- Extends `_compute_2x2_projector` to accept `SymmetricTensor` inputs with non-trivial charges (Issue #416). The dense pipeline is retained as the AD-tracer fallback; non-tracer SymmetricTensor inputs dispatch to a new `_compute_2x2_projector_symmetric` helper.
- Block-sparse SVDs via `tenax.linalg.svd` + per-sector gauge-fix helper `_gauge_fix_symmetric_svd` preserves the 2x2 closure `P_bot · P_top = I` per sector.
- `chi_new` allocation accepts optional `base_charges` (mirrors `_svd_projector_symmetric`); falls back to global top-k when not supplied.

## Tests

- New unit tests in `tests/test_ctm_2x2_projector_symmetric.py`: gauge-fix preservation (× 2), closure on non-trivial U(1) for all 4 directions, `base_charges` → `_derive_charges` allocation, AD-tracer fallback.
- Unxfail 7 tests previously blocked on #416 (`TestADSymmetric` × 4, `TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs`, `TestTodenseGradientFlow` × 2).

## Spec / Plan

- Design: `docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md`
- Plan: `docs/superpowers/plans/2026-05-11-2x2-projector-symmetric.md`

Closes #416.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Update memory + design doc with PR number**

After the PR URL is returned, run the find-and-replace in Task 9 Step 1 / 2 (which left `#XXX` as a placeholder) with the real PR number. Commit:

```bash
git add docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md
git commit -m "docs: link PR #<N> in 2x2 projector design doc"
git push
```

---

## Self-review checklist (run before opening PR)

- [ ] Every task lists exact files with line ranges where modifications happen.
- [ ] Every code step contains the actual code (no "TBD", no "similar to above"). Task 2 Step 3's design-discovery note **must** be replaced with the actual implementation before the closure test passes.
- [ ] No method-signature change goes silent — Task 5 adds `base_charges` as optional, keyword-only, default `None`, so all existing callers in `_ctm_tensor_moves.py` continue to work unchanged.
- [ ] Helper names and signatures are consistent across tasks (`_gauge_fix_symmetric_svd`, `_scale_bond_by_diag`, `_retruncate_by_base_charges`, `_compute_2x2_projector_symmetric`).
- [ ] `_compute_2x2_projector` retains its existing dense behavior for `DenseTensor` inputs (Task 4's dispatch only fires when `any(isinstance(_, SymmetricTensor) for _ in inputs)`).
- [ ] The 7 xfailed tests are individually verified to pass after the symmetric branch lands (Task 7 Step 2).
- [ ] No use of `_data` / `_blocks` / `_block_keys` outside SymmetricTensor itself except through `_from_blocks_unchecked` (which is the documented private constructor).

---

## Open design risks

1. **The contract orientation in Step 5 (cross-projector formation)** is the most error-prone part of `_compute_2x2_projector_symmetric`. The dense path's matrix view collapses leg structure; the symmetric path must thread labels carefully so `contract` auto-pairs the right axes. Task 2's closure test is the smoking gun — if Step 5's orientation is wrong, closure error will be large (not just numerical noise). Spend implementation effort here; do not skip to Task 3 until Task 2's closure test passes cleanly.

2. **Phase rotations in `_gauge_fix_symmetric_svd`** must apply to U **AND** Vh atomically (a phase on column j of U requires the conjugate phase on row j of Vh to preserve U·diag(s)·Vh). The Task 1 `test_gauge_fix_symmetric_svd_preserves_reconstruction` test catches this.

3. **Re-truncation in `_retruncate_by_base_charges` (Task 5)** is *not* a true SVD — it just keeps a per-sector subset of the existing SVD's columns. The resulting projector still satisfies closure because closure is `Vh^† · U^†` which is invariant under column-subset selection on both factors. But the projector may no longer be a "best rank-chi approximation" in the Frobenius-norm sense (the global top-k SVD is). This is consistent with `_svd_projector_symmetric`'s per-sector policy and is the right trade-off for sector-structure preservation.

4. **Complex dtype handling.** All helpers must work for complex-valued tensors (variPEPS-style implicit-AD uses complex128 for the χ=16 path per memory `feedback_phase_gauge_default.md`). The `_gauge_fix_symmetric_svd` test uses real-valued random inputs; add a complex-input variant to catch dtype-promotion bugs:

```python
def test_gauge_fix_symmetric_svd_preserves_reconstruction_complex():
    """Same as the real-valued test but with complex128 inputs."""
    # ... build M_T with complex random data ...
    # ... same assertions ...
```

Add this to Task 1's test list before commit.
