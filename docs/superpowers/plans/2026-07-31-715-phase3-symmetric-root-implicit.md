# #715 Phase 3 slice 1 — Root Implicit AD on `SymmetricTensor` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Phase 1 asymmetric 1x1 root-implicit CTMRG gradient to block-sparse `SymmetricTensor`, so the converged environment's gradient is computed with no SVD or eigh backward anywhere in the graph.

**Architecture:** Two new modules. A *sector layer* that knows nothing about CTM — charge layouts, per-sector full SVD with global truncation, per-sector dense matrix functions, and block reassembly. A *CTM layer* that knows nothing about charge arithmetic — environment, quadrants, projectors, sweep, characteristic equations, adjoint — which calls the sector layer at the cut. Environment tensors and all contractions stay `SymmetricTensor`; the projector core drops to dense per charge sector and reuses Phase 1's helpers unchanged.

**Tech Stack:** JAX (x64), `tenax.SymmetricTensor`, `tenax.contract`, `tenax.fuse_indices` / `split_index`, `tenax.linalg._group_blocks_by_bond_charge`, `gmres_pytree`, pytest.

**Spec:** `docs/plans/2026-07-31-715-phase3-symmetric-design.md`

---

## Ground truth established before this plan was written

These were run against the working tree, not assumed. Do not re-litigate them:

- **`SymmetricTensor` is a single-leaf pytree** and the L2 norm of that leaf *is* the Frobenius norm (3.700592080134 both ways on a 3-block U(1) tensor). `gmres_pytree` therefore carries the correct inner product with no adapter.
- **fuse -> split round-trips bit-exactly** (`max|diff| = 0.0`) and restores original label order.
- **Per-sector full SVD + global top-chi truncation works**: on a 4-leg enlarged corner with sector sizes `n_q = (4, 6, 4)`, chi=6 gives layout `{-1: 2, 0: 2, 1: 2}` summing exactly to 6, with `|U*^dag U_perp| ~ 1e-16` per sector.
- **Reassembly works**: build the truncated bond `TensorIndex` from the layout and call `SymmetricTensor._from_blocks_unchecked(blocks, indices)`; `_validate()` passes. Block keys are *charge values* per axis, and the constraint is that flow-weighted charges fuse to zero.
- **The contractor applies no Koszul signs** (`contractor.py:691-699`) because planar diagrams have no physical line crossings. CTM networks are planar.
- **`initialize_ctm_tensor_env` works for U(1) at D=2 and fails at D=3.** At D=3 it raises `ValueError: data.shape (4, 4, 4) does not match index dims (4, 9, 4)` — #667's one-`ref_axis`-per-corner bug. Z2 works at both. So the test site tensor is U(1) at D=2, which still fragments (fused-leg multiplicities `[1, 2, 1]`). Do not "fix" this inside Phase 3.

## File Structure

| File | Responsibility |
|---|---|
| `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py` (create) | Sector layer. `BondLayout`, per-sector full SVD + global truncation, per-sector matrix functions, block<->sector conversion, truncated-bond index construction. No CTM knowledge. |
| `src/tenax/algorithms/_ctm_root_implicit_symmetric.py` (create) | CTM layer. `SymEnv`, convention swap, init, quadrants, projectors, sweep, converge, characteristic residual, root parametrisation, adjoint. No charge arithmetic. |
| `tests/test_ctm_root_implicit_sym_sectors.py` (create) | Sector-layer unit tests. Cheap, all `core`. |
| `tests/test_ctm_root_implicit_symmetric.py` (create) | CTM-layer tests incl. the gradient gate and the trap test. |

The design doc names one module; this plan splits the sector layer out because it is independently testable and keeps each file focused. That is a refinement, not a change of approach.

Reference module to mirror function-for-function: `src/tenax/algorithms/_ctm_root_implicit_asym.py`.

---

## Task 1: Truncated-bond index from a charge layout

**Files:**
- Create: `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py`
- Test: `tests/test_ctm_root_implicit_sym_sectors.py`

- [ ] **Step 1: Write the failing test**

```python
"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3)."""

import numpy as np
import pytest

from tenax import FlowDirection, TensorIndex, U1Symmetry, ZnSymmetry
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    bond_index_from_layout,
)


def test_bond_index_from_layout_has_one_sector_per_retained_charge():
    layout = BondLayout(dims={-1: 2, 0: 3, 1: 2})
    idx = bond_index_from_layout(
        layout, U1Symmetry(), FlowDirection.OUT, "chi_new"
    )
    assert list(idx.sectors) == [-1, 0, 1]
    assert list(idx.multiplicities) == [2, 3, 2]
    assert idx.flow is FlowDirection.OUT
    assert idx.label == "chi_new"
    assert int(np.sum(idx.multiplicities)) == layout.total == 7


def test_bond_index_from_layout_drops_empty_sectors():
    # A sector that retained nothing must not appear as a zero-width sector:
    # a zero multiplicity is a shape of 0 that propagates into every
    # downstream contraction.
    layout = BondLayout(dims={-1: 0, 0: 4, 1: 0})
    idx = bond_index_from_layout(layout, U1Symmetry(), FlowDirection.IN, "b")
    assert list(idx.sectors) == [0]
    assert list(idx.multiplicities) == [4]
    assert layout.total == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_root_implicit_sym_sectors'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Sector layer for the symmetric root-implicit CTMRG gradient (#715 Phase 3).

Nothing here knows about CTM.  It knows about charge sectors: how a chi bond
is split across them, how to decompose a block-diagonal matrix sector by
sector, and how to get back to a :class:`SymmetricTensor`.  The CTM layer in
``_ctm_root_implicit_symmetric`` calls into this at the cut and nowhere else.

The split exists because the two halves fail differently.  A bug here is a
wrong *shape* or a wrong charge and shows up as an exception; a bug in the CTM
layer is a mis-glued network and shows up as a wrong number.  Keeping them
apart keeps the tests that catch them apart too.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tenax import FlowDirection, Label, TensorIndex
from tenax.symmetry import BaseSymmetry


class BondLayout(NamedTuple):
    """How a truncated chi bond is distributed over charge sectors.

    This is the symmetric analogue of "chi is an int".  Every downstream shape
    — ``u``, ``v``, ``S``, ``U*``, ``U_perp`` — is read off it, and it is
    *frozen* at the converged point: if it moved under AD the adjoint would be
    solving a different-sized system than the forward built.
    """

    dims: dict[int, int]

    @property
    def total(self) -> int:
        return int(sum(self.dims.values()))

    @property
    def charges(self) -> list[int]:
        """Retained charges, sorted, excluding sectors that kept nothing."""
        return sorted(q for q, d in self.dims.items() if d > 0)


def bond_index_from_layout(
    layout: BondLayout,
    symmetry: BaseSymmetry,
    flow: FlowDirection,
    label: Label,
) -> TensorIndex:
    """Build the truncated chi bond described by ``layout``.

    Sectors that retained nothing are dropped rather than kept at width zero.
    A zero multiplicity is a legal but poisonous index: it survives every
    charge check and then produces zero-size blocks that contract to zero.
    """
    charges = layout.charges
    if not charges:
        raise ValueError("BondLayout retained no charges; the cut is empty.")
    return TensorIndex(
        symmetry=symmetry,
        sectors=np.asarray(charges, dtype=np.int32),
        multiplicities=np.asarray(
            [layout.dims[q] for q in charges], dtype=np.int32
        ),
        flow=flow,
        label=label,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v`
Expected: PASS, 2 passed

If `from tenax.symmetry import BaseSymmetry` fails, find the real import path with
`python -c "import tenax; print(type(tenax.U1Symmetry()).__mro__)"` and fix the import.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py tests/test_ctm_root_implicit_sym_sectors.py
git commit -m "feat(#715): Phase 3 sector layer — bond layout and its index"
```

---

## Task 2: Per-sector full SVD with global truncation

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py`
- Test: `tests/test_ctm_root_implicit_sym_sectors.py`

This is the task the whole slice turns on. Truncation is **global across sectors** — the top chi singular values of the whole cut, wherever they live — which is what makes the layout data-dependent and therefore something that must be computed once and frozen.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_sym_sectors.py`:

```python
import jax.numpy as jnp

from tenax import SymmetricTensor, fuse_indices
from tenax.algorithms._ctm_root_implicit_sym_sectors import sector_svd


def _matrix_tensor(seed=0, sectors=(-1, 0, 1), mults=(1, 2, 1)):
    """A fused 2-leg tensor shaped like a half-infinite environment cut."""
    sym = U1Symmetry()

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.asarray(sectors),
            multiplicities=np.asarray(mults),
            flow=flow,
            label=lbl,
        )

    ec = SymmetricTensor.random_normal_np(
        (
            leg(FlowDirection.OUT, "chi_r"),
            leg(FlowDirection.OUT, "a_r"),
            leg(FlowDirection.IN, "chi_d"),
            leg(FlowDirection.IN, "a_d"),
        ),
        np.random.RandomState(seed),
    )
    fused = fuse_indices(ec, 2, 3, "row", FlowDirection.IN)
    return fuse_indices(fused, 0, 1, "col", FlowDirection.OUT)


def test_sector_svd_truncates_globally_not_per_sector():
    m = _matrix_tensor()
    chi = 6
    sectors, layout = sector_svd(m, chi, row_axis=1, col_axis=0)

    assert layout.total == chi
    # Global truncation: the retained values are exactly the top chi of the
    # union over sectors.  A per-sector rule would keep chi/n_sectors each.
    kept = sorted(
        (float(s) for q in layout.charges for s in sectors[q].s[: layout.dims[q]]),
        reverse=True,
    )
    every = sorted(
        (float(s) for q in sectors for s in sectors[q].s), reverse=True
    )
    assert kept == pytest.approx(every[:chi], rel=1e-12)


def test_sector_svd_null_space_is_the_exact_complement():
    m = _matrix_tensor()
    sectors, layout = sector_svd(m, 6, row_axis=1, col_axis=0)
    for q in layout.charges:
        blk = sectors[q]
        k = layout.dims[q]
        u_star, u_perp = blk.U[:, :k], blk.U[:, k:]
        assert float(jnp.max(jnp.abs(u_star.conj().T @ u_perp))) < 1e-12
        # U_perp must actually span the rest, not be empty by accident.
        assert u_perp.shape[1] == blk.U.shape[0] - k


def test_sector_svd_floors_against_the_global_maximum():
    # A sector whose own singular values are all tiny must not have its noise
    # promoted: the floor is relative to the largest SV of the whole cut, not
    # of the sector.  Built by hand so one sector is 1e-20 times the other.
    m = _matrix_tensor()
    sectors, layout = sector_svd(m, 6, row_axis=1, col_axis=0)
    biggest = max(float(sectors[q].s[0]) for q in sectors)
    for q in layout.charges:
        k = layout.dims[q]
        assert float(jnp.min(sectors[q].S_keep_diag[:k])) >= 1e-12 * biggest * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v -k sector_svd`
Expected: FAIL — `ImportError: cannot import name 'sector_svd'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py`:

```python
import jax
import jax.numpy as jnp

from tenax.core import BlockKey, SymmetricTensor
from tenax.linalg import _group_blocks_by_bond_charge


class SectorSVD(NamedTuple):
    """A full dense SVD of one charge sector of the cut, plus its keys.

    ``U`` and ``Vh`` are *full* (``full_matrices=True``): the null-space
    columns ``U[:, k:]`` and rows ``Vh[k:]`` are the ``U_perp`` / ``Vh_perp``
    of paper Eq. 71-72, restricted to this sector.  #715 planned to avoid
    materialising them; per sector these matrices are small, and the
    complementary-projector form would add a redundant null direction to
    ``d_yF`` for nothing.
    """

    U: jax.Array
    s: jax.Array
    Vh: jax.Array
    S_keep_diag: jax.Array
    row_key: BlockKey
    col_key: BlockKey


def sector_svd(
    matrix: SymmetricTensor,
    chi: int,
    *,
    row_axis: int,
    col_axis: int,
    floor_rtol: float = 1e-12,
) -> tuple[dict[int, SectorSVD], BondLayout]:
    """Full SVD per charge sector, then one global top-``chi`` truncation.

    The matrix is block diagonal in the bond charge, so each sector decomposes
    independently — but the *truncation* is global, over the union of every
    sector's spectrum.  That is what makes the retained charge distribution
    data-dependent, and hence what :class:`BondLayout` has to record.

    The floor is taken against the largest singular value of the **whole cut**,
    not of each sector.  Flooring per sector would rescale a sector whose
    values are all numerically zero up to the retained range and promote its
    noise to a kept direction.  Phase 1 floors for the same reason — an early
    environment is rank deficient and a singular ``S`` makes the matrix inverse
    square root produce NaNs (``_ctm_root_implicit_asym.all_projectors``).
    """
    grouped = _group_blocks_by_bond_charge(matrix, [row_axis], [col_axis])

    raw: dict[int, tuple] = {}
    for q, entries in grouped.items():
        if len(entries) != 1:  # pragma: no cover - a fused matrix has one per q
            raise ValueError(
                f"sector {q} has {len(entries)} blocks; expected a fused matrix"
            )
        (row_key, col_key, block) = entries[0]
        U, s, Vh = jnp.linalg.svd(block, full_matrices=True)
        raw[q] = (U, s, Vh, row_key, col_key)

    # Global truncation across sectors.
    ranked = sorted(
        ((float(sv), q, i) for q, (_U, s, *_r) in raw.items()
         for i, sv in enumerate(s)),
        key=lambda t: -t[0],
    )
    dims: dict[int, int] = {q: 0 for q in raw}
    for _sv, q, _i in ranked[: int(chi)]:
        dims[q] += 1
    layout = BondLayout(dims=dims)

    biggest = max((float(raw[q][1][0]) for q in raw), default=0.0)
    floor = floor_rtol * biggest

    sectors: dict[int, SectorSVD] = {}
    for q, (U, s, Vh, row_key, col_key) in raw.items():
        sectors[q] = SectorSVD(
            U=U,
            s=s,
            Vh=Vh,
            S_keep_diag=jnp.maximum(s, floor),
            row_key=row_key,
            col_key=col_key,
        )
    return sectors, layout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py tests/test_ctm_root_implicit_sym_sectors.py
git commit -m "feat(#715): Phase 3 per-sector full SVD with global truncation"
```

---

## Task 3: Per-sector matrix functions and reassembly

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py`
- Test: `tests/test_ctm_root_implicit_sym_sectors.py`

Phase 1's `_denman_beavers`, `_inv_sqrt`, `_quartic_root` are `jnp.linalg.inv` plus matmuls. They are reused **unchanged**, applied per sector. This is what makes #715's "block-diagonal quartic roots" a non-problem.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_sym_sectors.py`:

```python
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    sector_map,
    tensor_from_sector_matrices,
)


def test_sector_map_applies_the_dense_function_blockwise():
    from tenax.algorithms._ctm_root_implicit_asym import _inv_sqrt

    mats = {
        -1: jnp.eye(2) * 4.0,
        0: jnp.eye(3) * 9.0,
    }
    out = sector_map(_inv_sqrt, mats)
    assert set(out) == {-1, 0}
    assert float(jnp.max(jnp.abs(out[-1] - jnp.eye(2) * 0.5))) < 1e-10
    assert float(jnp.max(jnp.abs(out[0] - jnp.eye(3) / 3.0))) < 1e-10


def test_tensor_from_sector_matrices_round_trips_through_todense():
    m = _matrix_tensor()
    sectors, layout = sector_svd(m, 6, row_axis=1, col_axis=0)

    # Rebuild the *untruncated* matrix from its per-sector SVDs; it must equal
    # the original.  This is the reassembly path every projector uses.
    rebuilt_mats = {
        q: sectors[q].U @ jnp.diag(sectors[q].s.astype(sectors[q].U.dtype))
        @ sectors[q].Vh
        for q in sectors
    }
    rebuilt = tensor_from_sector_matrices(
        rebuilt_mats,
        row_index=m.indices[1],
        col_index=m.indices[0],
        row_axis=1,
        col_axis=0,
    )
    assert float(jnp.max(jnp.abs(rebuilt.todense() - m.todense()))) < 1e-10
    assert rebuilt.labels() == m.labels()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v -k "sector_map or round_trips"`
Expected: FAIL — `ImportError: cannot import name 'sector_map'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py`:

```python
from collections.abc import Callable


def sector_map(
    fn: Callable[[jax.Array], jax.Array], mats: dict[int, jax.Array]
) -> dict[int, jax.Array]:
    """Apply a dense matrix function to every charge sector.

    Phase 1's ``_denman_beavers`` / ``_inv_sqrt`` / ``_quartic_root`` go
    through here untouched.  They are ``jnp.linalg.inv`` plus matmuls, so they
    put no decomposition back into ``F`` — which is the whole point of the
    method — and they are already correct on a general (non-diagonal, complex)
    matrix, which is what ``S`` becomes in the reverse pass.

    Charge conservation forbids entries between sectors, so a block-diagonal
    matrix function *is* the per-sector function.  There is nothing to
    generalise.
    """
    return {q: fn(m) for q, m in mats.items()}


def tensor_from_sector_matrices(
    mats: dict[int, jax.Array],
    *,
    row_index: TensorIndex,
    col_index: TensorIndex,
    row_axis: int,
    col_axis: int,
) -> SymmetricTensor:
    """Rebuild a 2-leg :class:`SymmetricTensor` from per-sector matrices.

    Block keys are *charge values* per axis, and the charge constraint is that
    the flow-weighted charges fuse to zero.  For the usual pairing of one IN
    and one OUT leg that means both keys carry the same charge ``q``.
    """
    if {row_axis, col_axis} != {0, 1}:
        raise ValueError("row_axis and col_axis must be 0 and 1 in some order")

    indices: list[TensorIndex] = [None, None]  # type: ignore[list-item]
    indices[row_axis] = row_index
    indices[col_axis] = col_index

    blocks: dict[BlockKey, jax.Array] = {}
    for q, mat in mats.items():
        if mat.size == 0:
            continue
        key = [0, 0]
        key[row_axis] = int(q)
        key[col_axis] = int(q)
        blocks[tuple(key)] = mat if row_axis == 0 else mat.T
    return SymmetricTensor._from_blocks_unchecked(blocks, tuple(indices))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -v`
Expected: PASS, 7 passed

If the round-trip test fails on transposition, the `mat.T` branch is the suspect: check
whether the grouped block for sector `q` was stored as `(row, col)` or `(col, row)` by
printing `sectors[q].U.shape` against `m.blocks` keys, and fix the single branch.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 per-sector matrix functions and reassembly"
```

---

## Task 4: Symmetric environment and the convention swap

**Files:**
- Create: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

`_ctm_root_implicit_asym.py:516-535` builds symmetric tensors via `initialize_ctm_tensor_env` and then calls `.todense()` on all nine. Not densifying is most of the symmetric forward.

`swap_env_convention` is load-bearing and is where #718 lived: this module stores every tensor in the frame of its own direction, while `CTMTensorEnv` closes the ring with `C4` transposed and `T3`, `T4` reversed. It is an involution and a *no-op on symmetric initialisers*, which is exactly why the mismatch stayed invisible for months.

- [ ] **Step 1: Write the failing test**

```python
"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry
from tenax.algorithms._ctm_root_implicit_symmetric import (
    SymEnv,
    init_env_sym,
    swap_env_convention_sym,
)


def _site_tensor(seed: int = 0) -> SymmetricTensor:
    """A U(1) iPEPS site tensor with non-trivial charges on every leg.

    Non-trivial deliberately: a trivial-charge tensor has one block, so every
    layout bug is invisible.  The fused ``D**2`` leg comes out with sector
    multiplicities ``[1, 2, 1]`` — *unequal*, which is the fragmenting case
    (#566's D-parity finding) and the only one that exercises the layout
    arithmetic.

    **D=2 with virtual sectors [0, 1], not D=3 with [-1, 0, 1].** Verified
    2026-07-31: at D=3 ``initialize_ctm_tensor_env`` raises
    ``ValueError: data.shape (4, 4, 4) does not match index dims (4, 9, 4)``.
    That is #667 — ``_CORNER_SPECS`` gives one ``ref_axis`` per corner, so an
    env leg's charges are derived from a direction it does not physically
    touch.  It is a real production coverage gap, not a defect in this port,
    and the design doc defers it to follow-up slice 3.  D=2 initialises
    cleanly and still fragments, so the coverage goal survives.
    """
    sym = U1Symmetry()
    phys = TensorIndex(
        symmetry=sym,
        sectors=np.array([-1, 1]),
        multiplicities=np.array([1, 1]),
        flow=FlowDirection.OUT,
        label="p",
    )

    def virt(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([0, 1]),
            multiplicities=np.array([1, 1]),
            flow=flow,
            label=lbl,
        )

    return SymmetricTensor.random_normal_np(
        (
            phys,
            virt(FlowDirection.IN, "u"),
            virt(FlowDirection.OUT, "d"),
            virt(FlowDirection.IN, "l"),
            virt(FlowDirection.OUT, "r"),
        ),
        np.random.RandomState(seed),
    )


def test_init_env_sym_keeps_every_tensor_symmetric():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    assert isinstance(env, SymEnv)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        t = getattr(env, name)
        assert isinstance(t, SymmetricTensor), f"{name} was densified"
        assert t.n_blocks > 0
    assert isinstance(a, SymmetricTensor)


def test_swap_env_convention_sym_is_an_involution():
    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    twice = swap_env_convention_sym(swap_env_convention_sym(env))
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        lhs = getattr(twice, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_root_implicit_symmetric'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian).

Structurally this mirrors ``_ctm_root_implicit_asym`` function for function.
The difference is that every environment tensor stays a ``SymmetricTensor``
and every contraction goes through ``contract``, so charge bookkeeping is the
library's job and a wrong flow raises instead of silently mis-gluing the
network — which is the #718 failure mode.

Charge arithmetic at the cut lives in ``_ctm_root_implicit_sym_sectors``; this
file never touches a charge directly.
"""

from __future__ import annotations

from typing import NamedTuple

from tenax import SymmetricTensor, Tensor
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)


class SymEnv(NamedTuple):
    """Eight environment tensors in this module's rotation-uniform frame."""

    C1: SymmetricTensor
    C2: SymmetricTensor
    C3: SymmetricTensor
    C4: SymmetricTensor
    T1: SymmetricTensor
    T2: SymmetricTensor
    T3: SymmetricTensor
    T4: SymmetricTensor


def swap_env_convention_sym(env: SymEnv) -> SymEnv:
    """Between this module's uniform frame and ``CTMTensorEnv``'s.

    Same map as ``_ctm_root_implicit_asym.swap_env_convention``, and the same
    reason it exists: this module closes the ring uniformly (corner ``k`` is
    always ``(leg towards k-1, leg towards k)``), while ``CTMTensorEnv`` closes
    it with ``C4`` transposed and ``T3``, ``T4`` reversed.  Reinterpreting one
    as the other glues the network wrongly — 1.5% on the energy at D=2 chi=4,
    and the +-2.121e-3 per-bond antisymmetry that was #718.

    An involution, so one function converts either way.  It is a *no-op* on a
    symmetric initialiser, which is why the mismatch stayed invisible until the
    environment became genuinely asymmetric.
    """
    return env._replace(
        C4=env.C4.transpose((1, 0)),
        T3=env.T3.transpose((2, 1, 0)),
        T4=env.T4.transpose((2, 1, 0)),
    )


def init_env_sym(A: Tensor, chi: int) -> tuple[SymEnv, SymmetricTensor]:
    """Seed the environment and build the double layer, without densifying.

    ``_ctm_root_implicit_asym._init_env`` calls ``.todense()`` on all nine
    tensors here.  Keeping them symmetric is most of the symmetric forward.
    """
    env_t = initialize_ctm_tensor_env(A, chi)
    a_t = _build_double_layer_tensor(A)
    labels = list(a_t.labels())
    perm = tuple(labels.index(lbl) for lbl in ("u2", "d2", "l2", "r2"))
    a = a_t.transpose(perm)
    env = SymEnv(
        C1=env_t.C1, C2=env_t.C2, C3=env_t.C3, C4=env_t.C4,
        T1=env_t.T1, T2=env_t.T2, T3=env_t.T3, T4=env_t.T4,
    )
    return swap_env_convention_sym(env), a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 2 passed

`initialize_ctm_tensor_env` is already known to work on `_site_tensor()` as defined above
(verified 2026-07-31). If you change the site tensor's virtual dimension to D=3 it will
raise `ValueError: data.shape (4, 4, 4) does not match index dims (4, 9, 4)` — that is
#667, a production gap, not something to work around here. Keep D=2.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 symmetric environment, no densify"
```

---

## Task 5: Quadrants and the half-infinite cut as a fused matrix

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

The gate is agreement with the dense module's quadrant after densifying — a real oracle, since Phase 1 is on `main` and FD-validated.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_symmetric.py`:

```python
from tenax.algorithms._ctm_root_implicit_symmetric import (
    half_infinite_sym,
    upper_left_quadrant_sym,
)


def test_upper_left_quadrant_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import (
        AsymEnv,
        _upper_left_quadrant,
    )

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)

    dense_env = AsymEnv(
        *[jnp.asarray(getattr(env, n).todense())
          for n in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")]
    )
    expected = _upper_left_quadrant(dense_env, jnp.asarray(a.todense()))
    got = upper_left_quadrant_sym(env, a)
    assert got.labels() == ("chi_r", "a_r", "chi_d", "a_d")
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-12


def test_half_infinite_sym_is_block_diagonal_and_square():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    m = half_infinite_sym(env, a)
    assert m.ndim == 2
    dense = m.todense()
    assert dense.shape[0] == dense.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v -k quadrant`
Expected: FAIL — `ImportError: cannot import name 'upper_left_quadrant_sym'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`:

```python
from tenax import FlowDirection, contract, fuse_indices


def upper_left_quadrant_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """``C1 T1 T4 a`` with legs ``(chi_r, a_r, chi_d, a_d)``.

    Same network as ``_ctm_root_implicit_asym._upper_left_quadrant``; the
    einsum's index letters become labels.  ``(chi_d, a_d)`` is the vertical
    bond to be truncated, ``(chi_r, a_r)`` is what stays open to the right.
    """
    c1 = env.C1.relabels({env.C1.labels()[0]: "c", env.C1.labels()[1]: "e"})
    t1 = env.T1.relabels(
        dict(zip(env.T1.labels(), ("e", "f", "chi_r"), strict=True))
    )
    t4 = env.T4.relabels(
        dict(zip(env.T4.labels(), ("h", "i", "c"), strict=True))
    )
    a4 = a.relabels(dict(zip(a.labels(), ("f", "a_d", "i", "a_r"), strict=True)))
    q = contract(c1, t1, t4, a4, output_labels=("chi_r", "a_r", "h", "a_d"))
    return q.relabels({"h": "chi_d"}).transpose(
        tuple(q.relabels({"h": "chi_d"}).labels().index(lbl)
              for lbl in ("chi_r", "a_r", "chi_d", "a_d"))
    )


def _as_matrix_sym(quadrant: SymmetricTensor) -> SymmetricTensor:
    """Fuse ``(chi_d, a_d)`` to rows and ``(chi_r, a_r)`` to columns.

    Fusion happens *only* here, to feed the SVD.  Everywhere the singular
    values have to act, the cut leg stays split — see ``apply_bond_matrix``.
    """
    fused = fuse_indices(quadrant, 2, 3, "row", FlowDirection.IN)
    return fuse_indices(fused, 0, 1, "col", FlowDirection.OUT)


def half_infinite_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """Paper Eq. 65: upper-left quadrant glued to lower-left along the cut."""
    top = _as_matrix_sym(upper_left_quadrant_sym(env, a))
    bot = _as_matrix_sym(lower_left_quadrant_sym(env, a))
    return contract(
        top.relabels({"row": "k"}),
        bot.relabels({"col": "k"}),
        output_labels=("col", "row"),
    )
```

Also add `lower_left_quadrant_sym`, mirroring
`_ctm_root_implicit_asym._lower_left_quadrant` (`C4 T3 T4 a` -> `(chi_u, a_u, chi_r, a_r)`)
with the same relabel-then-`contract` pattern:

```python
def lower_left_quadrant_sym(env: SymEnv, a: SymmetricTensor) -> SymmetricTensor:
    """``C4 T3 T4 a`` with legs ``(chi_u, a_u, chi_r, a_r)``."""
    c4 = env.C4.relabels(dict(zip(env.C4.labels(), ("m", "n"), strict=True)))
    t3 = env.T3.relabels(dict(zip(env.T3.labels(), ("p", "q", "m"), strict=True)))
    t4 = env.T4.relabels(dict(zip(env.T4.labels(), ("n", "i", "t"), strict=True)))
    a4 = a.relabels(dict(zip(a.labels(), ("a_u", "q", "i", "a_r"), strict=True)))
    q_ = contract(c4, t3, t4, a4, output_labels=("t", "a_u", "p", "a_r"))
    return q_.relabels({"t": "chi_u", "p": "chi_r"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 4 passed

The relabel-to-`contract` translation of an einsum is the most error-prone step in this
plan. If the dense comparison fails, do **not** start permuting axes hopefully. Print
`got.labels()` and `expected.shape`, and check one leg at a time against the einsum
string in `_ctm_root_implicit_asym.py:151-174`, which documents what every letter is.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 symmetric quadrants and half-infinite cut"
```

---

## Task 6: Symmetric Fishman projectors with a per-sector gauge pin

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

The closure `P_right @ P_left = 1` holds **for any `S`, diagonal or not** — that is the property that makes the pair gauge-covariant, and it is why `S` must be a genuine matrix with a genuine matrix inverse square root rather than a vector of singular values.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_symmetric.py`:

```python
from tenax.algorithms._ctm_root_implicit_symmetric import all_projectors_sym


def test_projector_closure_is_the_identity_per_sector():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    projs = all_projectors_sym(env, a, chi=4)
    assert len(projs) == 4
    for k in range(4):
        p = projs[k]
        for q, k_q in p.layout.dims.items():
            if k_q == 0:
                continue
            closure = p.P_right[q] @ p.P_left[q]
            eye = jnp.eye(k_q, dtype=closure.dtype)
            assert float(jnp.max(jnp.abs(closure - eye))) < 1e-9, (k, q)


def test_retained_dimension_totals_chi():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    for p in all_projectors_sym(env, a, chi=4):
        assert p.layout.total == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v -k projector`
Expected: FAIL — `ImportError: cannot import name 'all_projectors_sym'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`:

```python
import jax.numpy as jnp

from tenax.algorithms._ctm_root_implicit_asym import _inv_sqrt
from tenax.algorithms._ctm_root_implicit_sym_sectors import (
    BondLayout,
    SectorSVD,
    sector_map,
    sector_svd,
)


class SymProjectors(NamedTuple):
    """One direction's Fishman pair, per charge sector, plus the frozen data."""

    P_left: dict[int, jnp.ndarray]
    P_right: dict[int, jnp.ndarray]
    S: dict[int, jnp.ndarray]
    sectors: dict[int, SectorSVD]
    layout: BondLayout


def _pin_bond_gauge_sector(P_left, P_right, U, Vh, k_q, prev_P_left=None):
    """Phase 1's ``_pin_bond_gauge``, restricted to one charge sector.

    An SVD fixes the singular subspaces and leaves one phase per retained
    index free.  That phase is a gauge of the CTM bond, so the environment
    converges element-wise in magnitude while individual signs keep flipping —
    and a characteristic equation cannot have a root under those conditions,
    because ``F`` compares tensors, not their magnitudes.

    Warm alignment to the previous sweep rather than ``argmax``: ``argmax`` is
    discontinuous, and when two retained singular values are close its row
    index hops between sweeps and the pinned phase oscillates with period two.
    """
    if prev_P_left is None:
        idx = jnp.argmax(jnp.abs(P_left), axis=0)
        ref = P_left[idx, jnp.arange(P_left.shape[1])]
    else:
        ref = jnp.sum(jnp.conj(prev_P_left) * P_left, axis=0)
    psi = jnp.where(jnp.abs(ref) > 0, jnp.conj(ref) / jnp.abs(ref), 1.0)
    P_left = P_left * psi[None, :]
    P_right = jnp.conj(psi)[:, None] * P_right
    U = U.at[:, :k_q].multiply(psi[None, :])
    Vh = Vh.at[:k_q, :].multiply(jnp.conj(psi)[:, None])
    return P_left, P_right, U, Vh


def all_projectors_sym(env: SymEnv, a: SymmetricTensor, chi: int, prev=None):
    """Decompose the cut in all four directions from the *same* environment.

    Simultaneous, not Gauss-Seidel.  A sequential sweep, where move ``k+1``
    sees the output of move ``k``, has a fixed point that does not satisfy
    Eqs. 76-77, because those evaluate all four moves at the same ``y``.
    """
    out = []
    env_k, a_k = env, a
    for k in range(4):
        top = _as_matrix_sym(upper_left_quadrant_sym(env_k, a_k))
        bot = _as_matrix_sym(lower_left_quadrant_sym(env_k, a_k))
        m = contract(
            top.relabels({"row": "kk"}),
            bot.relabels({"col": "kk"}),
            output_labels=("col", "row"),
        )
        sectors, layout = sector_svd(m, chi, row_axis=1, col_axis=0)

        top_blocks = _sector_blocks(top)
        bot_blocks = _sector_blocks(bot)

        S = {}
        for q in layout.charges:
            k_q = layout.dims[q]
            s_k = sectors[q].S_keep_diag[:k_q]
            S[q] = jnp.diag(
                s_k / (jnp.linalg.norm(s_k) + 1e-300)
            ).astype(m.dtype)
        inv = sector_map(_inv_sqrt, S)

        P_left, P_right, U_out, Vh_out = {}, {}, {}, {}
        for q in layout.charges:
            k_q = layout.dims[q]
            blk = sectors[q]
            pl = bot_blocks[q] @ blk.Vh[:k_q].conj().T @ inv[q]
            pr = inv[q] @ (blk.U[:, :k_q].conj().T @ top_blocks[q])
            prev_pl = None if prev is None else prev[k].P_left.get(q)
            pl, pr, U_q, Vh_q = _pin_bond_gauge_sector(
                pl, pr, blk.U, blk.Vh, k_q, prev_pl
            )
            P_left[q], P_right[q] = pl, pr
            U_out[q], Vh_out[q] = U_q, Vh_q

        sectors = {
            q: sectors[q]._replace(U=U_out[q], Vh=Vh_out[q])
            if q in U_out else sectors[q]
            for q in sectors
        }
        out.append(
            SymProjectors(
                P_left=P_left, P_right=P_right, S=S,
                sectors=sectors, layout=layout,
            )
        )
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    return out
```

Also add `_sector_blocks` (extract `{q: dense}` from a fused 2-leg tensor via
`_group_blocks_by_bond_charge`), and `rotate_env_sym` / `rotate_a_sym` mirroring
`_ctm_root_implicit_asym.rotate_env` (`_ctm_root_implicit_asym.py:109-135`) — a pure
relabel-and-reorder of the eight tensors plus a cyclic transpose of `a`:

```python
def _sector_blocks(matrix: SymmetricTensor) -> dict[int, jnp.ndarray]:
    """``{bond charge: dense block}`` of a fused 2-leg tensor."""
    from tenax.linalg import _group_blocks_by_bond_charge

    grouped = _group_blocks_by_bond_charge(matrix, [1], [0])
    return {q: entries[0][2] for q, entries in grouped.items()}


def rotate_env_sym(env: SymEnv) -> SymEnv:
    """Rotate the frame by 90 degrees: a pure relabel, no data movement.

    This is what the rotation-uniform convention buys — see
    ``_ctm_root_implicit_asym.rotate_env``, which this mirrors exactly.
    """
    return SymEnv(
        C1=env.C2, C2=env.C3, C3=env.C4, C4=env.C1,
        T1=env.T2, T2=env.T3, T3=env.T4, T4=env.T1,
    )


def rotate_a_sym(a: SymmetricTensor) -> SymmetricTensor:
    """``a`` legs ``(u, d, l, r)`` rotated to match ``rotate_env_sym``."""
    labels = a.labels()
    perm = (labels.index(labels[3]), labels.index(labels[2]),
            labels.index(labels[0]), labels.index(labels[1]))
    return a.transpose(perm)
```

Verify `rotate_env_sym` / `rotate_a_sym` against `_ctm_root_implicit_asym.rotate_env` /
`rotate_a` by densifying — the dense functions are the specification.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 symmetric Fishman projectors, per-sector gauge pin"
```

---

## Task 7: Sweep, convergence, and forward-energy parity

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

Convergence is **element-wise, not spectral**. Corner singular values are invariant under independent rotations of each bond, so a spectral criterion calls convergence while the tensors are still moving — and the characteristic equations compare tensors.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_symmetric.py`:

```python
from tenax.algorithms._ctm_root_implicit_symmetric import converge_sym, sym_energy


def test_symmetric_forward_energy_matches_the_dense_module():
    """The gate for the whole forward: same number, block-sparse or not."""
    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_energy,
        converge,
    )
    from tenax import heisenberg_gate

    A = _site_tensor()
    gate = heisenberg_gate()

    env_s, a_s, meta_s = converge_sym(A, chi=4, max_iter=60)
    e_sym = float(sym_energy(A, env_s, gate))

    env_d, a_d, meta_d = converge(A, chi=4, max_iter=60)
    e_dense = float(asym_energy(A, env_d, _template_env(A, 4), gate))

    assert meta_s["converged"], meta_s
    assert abs(e_sym - e_dense) < 1e-10, (e_sym, e_dense)
```

`_template_env(A, chi)` is `initialize_ctm_tensor_env(A, chi)`; add it as a helper in
the test file. `heisenberg_gate()` may need arguments — check
`python -c "import inspect, tenax; print(inspect.signature(tenax.heisenberg_gate))"`
and pass whatever the existing Phase 1 tests pass in
`tests/test_ctm_root_implicit_asym.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v -k forward_energy`
Expected: FAIL — `ImportError: cannot import name 'converge_sym'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`:

```python
from tenax import CTMTensorEnv
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor


def _normalize_sym(t: SymmetricTensor) -> SymmetricTensor:
    return t / (t.max_abs() + 1e-300)


def sweep_sym(env: SymEnv, a: SymmetricTensor, chi: int, prev=None):
    """One simultaneous CTMRG sweep: all four directions from one environment."""
    projs = all_projectors_sym(env, a, chi, prev)
    corners: list = [None] * 4
    edges: list = [None] * 4
    env_k, a_k = env, a
    for k in range(4):
        corners[_unrotate_index_sym(1, k) - 1] = _normalize_sym(
            renormalised_corner_sym(env_k, a_k, projs[k], projs[(k + 1) % 4])
        )
        edges[_unrotate_index_sym(4, k) - 1] = _normalize_sym(
            renormalised_edge_sym(env_k, a_k, projs[k])
        )
        env_k, a_k = rotate_env_sym(env_k), rotate_a_sym(a_k)
    return SymEnv(*corners, *edges), projs


def converge_sym(
    A: Tensor,
    chi: int,
    *,
    max_iter: int = 200,
    conv_tol: float = 1e-12,
    min_iter: int = 4,
    return_projectors: bool = False,
):
    """Sweep until every corner and edge stops moving element-wise.

    Element-wise, not spectral: corner singular values are invariant under
    independent rotations of each bond, so a spectral criterion calls
    convergence while the tensors are still moving — and the characteristic
    equations compare tensors.

    ``return_projectors`` is not a diagnostic.  The converged environment sits
    in the bond gauge of the chain that built it, and ``root_parametrize_sym``
    needs that same chain: a cold re-pin fixes a *different* gauge, leaving
    ``y*`` describing an environment it was not extracted from (#721).
    """
    env, a = init_env_sym(A, chi)
    prev_state, prev_projs = None, None
    residual, converged, iters = float("inf"), False, 0
    for it in range(int(max_iter)):
        env, prev_projs = sweep_sym(env, a, chi, prev_projs)
        iters = it + 1
        state = {
            name: getattr(env, name) / (getattr(env, name).norm() + 1e-300)
            for name in env._fields
        }
        if prev_state is not None:
            residual = max(
                float((state[n] - prev_state[n]).max_abs())
                for n in state
                if state[n].todense().shape == prev_state[n].todense().shape
            )
            if iters >= min_iter and residual < conv_tol:
                converged = True
                break
        prev_state = state

    meta = {"iters": iters, "residual": residual, "converged": converged}
    if return_projectors:
        return env, a, meta, prev_projs
    return env, a, meta


def _to_ctm_env_sym(env: SymEnv) -> CTMTensorEnv:
    """Hand the environment to the production RDM in *its* convention.

    This boundary is exactly where #718 lived.  Do not remove the swap.
    """
    swapped = swap_env_convention_sym(env)
    return CTMTensorEnv(**{n: getattr(swapped, n) for n in swapped._fields})


def sym_energy(A: Tensor, env: SymEnv, gate):
    return compute_energy_ctm_tensor(A, _to_ctm_env_sym(env), gate)
```

Also add `renormalised_corner_sym`, `renormalised_edge_sym` and `_unrotate_index_sym`,
mirroring `_ctm_root_implicit_asym._renormalised_corner` /`_renormalised_edge` /
`_unrotate_index` (`:472-491`, `:137-149`). The corner is the quadrant projected on
**both** open legs — `P_right` of move `k+1` on one, `P_left` of move `k` on the other —
and the projectors are applied per sector then reassembled with
`tensor_from_sector_matrices` against the bond index from
`bond_index_from_layout(projs[k].layout, ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 7 passed

If the energies differ by ~1.5%, suspect `_to_ctm_env_sym` first — that magnitude is the
#718 signature. Confirm with the gauge probe in Task 10's note before touching anything
else.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 symmetric sweep and forward-energy parity"
```

---

## Task 8: Characteristic equations on the symmetric environment

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

**This is where the `kron` is deleted rather than ported.** Phase 1 attaches the quartic
roots with `jnp.kron(root, eye_d2)` — the roots act on the `chi` factor of the cut leg
`n = chi * d2`. Under charge fusion that identity breaks: sector `q` of the fused leg is a
direct sum over all `(q_chi, q_d2)` with `q_chi + q_d2 = q`, so a matrix acting only on
the `chi` factor is *not* a per-sector `kron`. Keep the cut leg **split** and apply the
root by contraction on the `chi` leg.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_symmetric.py`:

```python
from tenax.algorithms._ctm_root_implicit_symmetric import (
    characteristic_residual_sym,
    root_parametrize_sym,
)


def test_the_converged_environment_is_a_root():
    A = _site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=60, return_projectors=True)
    root, residual = root_parametrize_sym(env, a, chi=4, prev_projs=projs)
    assert residual < 1e-10, residual


def test_apply_bond_matrix_acts_only_on_the_chi_factor():
    """The deleted kron, tested directly.

    Applying the identity on chi must be a no-op, and applying a scalar
    multiple must scale — on the *split* leg.  If this were done by fusing and
    kron-ing, the charge mixing would make even the identity wrong.
    """
    from tenax.algorithms._ctm_root_implicit_symmetric import apply_bond_matrix

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    quad = upper_left_quadrant_sym(env, a)
    layout = all_projectors_sym(env, a, chi=4)[0].layout
    eye = {q: jnp.eye(layout.dims[q]) for q in layout.charges}
    same = apply_bond_matrix(quad, eye, axis=2)
    assert float(jnp.max(jnp.abs(same.todense() - quad.todense()))) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v -k "root or bond_matrix"`
Expected: FAIL — `ImportError: cannot import name 'characteristic_residual_sym'`

- [ ] **Step 3: Write minimal implementation**

Port `_ctm_root_implicit_asym.asym_characteristic_residual_covariant` (`:780-878`),
`_covariant_pieces` (`:713-761`) and `_modified_env` (`:762-779`) with these
substitutions, and nothing else changed:

| Phase 1 | Phase 3 |
|---|---|
| `jnp.kron(_quartic_root(s), eye_d2)` | `apply_bond_matrix(t, sector_map(_quartic_root, s), axis=...)` on the **split** leg |
| `env.C1` etc. as arrays | `SymmetricTensor`, contracted by label |
| `S` a chi x chi array | `{q: (k_q x k_q) array}` |
| `U[:, :chi]`, `U[:, chi:]` | `{q: U_q[:, :k_q]}`, `{q: U_q[:, k_q:]}` |
| `jnp.vdot(X, Y)` for `lambda` | same, summed over sectors |

Add the helper the kron becomes:

```python
def apply_bond_matrix(
    t: SymmetricTensor, mats: dict[int, jnp.ndarray], *, axis: int
) -> SymmetricTensor:
    """Apply a per-sector matrix to the ``chi`` leg at ``axis``, leg unfused.

    This replaces Phase 1's ``jnp.kron(root, eye_d2)``.  That kron is only
    correct because the dense cut leg is ``chi`` outer ``d2`` with ``chi``
    slow; under charge fusion sector ``q`` mixes every ``(q_chi, q_d2)`` with
    ``q_chi + q_d2 = q``, so no per-sector kron reproduces it.  Contracting on
    the split leg is the same operation with nothing to get wrong.
    """
    label = t.labels()[axis]
    bond = t.indices[axis]
    op = tensor_from_sector_matrices(
        mats,
        row_index=bond.flip_flow().relabel("__in"),
        col_index=bond.relabel("__out"),
        row_axis=0,
        col_axis=1,
    )
    out = contract(
        t.relabels({label: "__in"}),
        op,
        output_labels=tuple(
            "__out" if lbl == label else lbl for lbl in t.labels()
        ),
    )
    return out.relabels({"__out": label})
```

If `TensorIndex` has no `flip_flow` / `relabel`, find the equivalents with
`python -c "import tenax, inspect; print([m for m in dir(tenax.TensorIndex) if not m.startswith('_')])"`
— the seam only needs a same-charges index with the opposite flow.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 9 passed. The root residual should land near 1e-14; anything above
1e-10 means the equations and the sweep disagree, which is how #718 started.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 characteristic equations, kron deleted not ported"
```

---

## Task 9: The adjoint, and the gradient gate

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_symmetric.py`
- Test: `tests/test_ctm_root_implicit_symmetric.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_root_implicit_symmetric.py`:

```python
from tenax.algorithms._ctm_root_implicit_symmetric import (
    sym_root_implicit_energy_and_grad,
)


@pytest.mark.slow
def test_symmetric_gradient_matches_the_dense_root_implicit_gradient():
    """The real gate for this slice."""
    from tenax.algorithms._ctm_root_implicit_asym import (
        asym_root_implicit_energy_and_grad,
    )
    from tenax import heisenberg_gate

    A = _site_tensor()
    gate = heisenberg_gate()

    e_sym, g_sym, diag = sym_root_implicit_energy_and_grad(A, gate, chi=4)
    e_den, g_den, _ = asym_root_implicit_energy_and_grad(A, gate, chi=4)

    assert abs(float(e_sym) - float(e_den)) < 1e-10
    num = float(jnp.linalg.norm(g_sym.todense() - jnp.asarray(g_den)))
    den = float(jnp.linalg.norm(jnp.asarray(g_den)))
    assert num / den < 1e-9, num / den


def test_no_svd_or_eigh_primitive_in_the_backward():
    """The claim the whole method rests on, asserted on the jaxpr."""
    import jax
    from tenax import heisenberg_gate

    A = _site_tensor()
    gate = heisenberg_gate()
    _e, _g, diag = sym_root_implicit_energy_and_grad(A, gate, chi=4)
    text = diag["backward_jaxpr"]
    assert "svd" not in text, "an SVD survived into the backward"
    assert "eigh" not in text, "an eigh survived into the backward"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v -k gradient`
Expected: FAIL — `ImportError: cannot import name 'sym_root_implicit_energy_and_grad'`

- [ ] **Step 3: Write minimal implementation**

Port `_ctm_root_implicit_asym.asym_root_implicit_energy_and_grad` (`:1013-1205`)
unchanged in structure. `gmres_pytree` needs no adapter — `SymmetricTensor` is a
single-leaf pytree whose leaf L2 norm is the Frobenius norm (verified). Add
`diag["backward_jaxpr"] = str(jax.make_jaxpr(...)(...))` for the assertion above.

Do **not** add a phase-fixing condition or a gauge quotient. The environment phase gauge
is real but benign: all eight phases are null directions of `d_yF`, `y_bar` is orthogonal
to every one of them (E is invariant along each), and differentiating `F(y*(p),p) = 0`
puts `d_pF` in range(`d_yF`), so the cokernel cannot reach the gradient. GMRES converges
on the singular system. #721 proposed gauge-fixing; it is unnecessary complexity.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 11 passed

**If the gradient disagrees, work in this order — the order is the finding of #718/#721:**
1. **Rule out sweep count.** The dense reference is itself truncated. Re-run it at 20 and
   30 sweeps; if the disagreement shrinks geometrically, the reference was the problem.
2. **Gauge-probe `F` and `E` separately.** Apply a finite per-bond gauge `W_k`, which
   cannot change anything a CTM environment computes. Ask (1) is `F` still a root, and
   (2) is `E` invariant. That splits "equations wrong" from "objective wrong" in one shot,
   with no reference value.
3. An **exact antisymmetry** between two directions means a transpose at a boundary, not
   a broken theory. Look at `_to_ctm_env_sym`.
4. The cotangent pairing is `Re sum(g * dz)`, **unconjugated**. The conjugated form
   manufactures violations that are not there.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(#715): Phase 3 symmetric adjoint — gradient parity vs dense"
```

---

## Task 10: The trap test and symmetry coverage

**Files:**
- Modify: `tests/test_ctm_root_implicit_symmetric.py`

`norm(F(y*))` being small does **not** validate the charge bookkeeping. The sector layout
has to be load-bearing, not incidental — this is the symmetric analogue of Phase 2's
cell-shift trap.

- [ ] **Step 1: Write the test**

```python
def test_a_wrong_bond_layout_breaks_the_root():
    """The layout must be load-bearing.

    Take the converged root and move one retained dimension from one charge
    sector to another, keeping the total at chi.  Every shape still fits and
    every charge still conserves — only the *physics* is wrong.  If F is still
    a root after that, the sector bookkeeping is not doing anything and none
    of the other tests mean what they claim.
    """
    A = _site_tensor()
    env, a, meta, projs = converge_sym(A, chi=4, max_iter=60, return_projectors=True)
    root, residual = root_parametrize_sym(env, a, chi=4, prev_projs=projs)
    assert residual < 1e-10

    layout = projs[0].layout
    charges = layout.charges
    assert len(charges) >= 2, "need >=2 populated sectors for this test"
    bad = dict(layout.dims)
    bad[charges[0]] -= 1
    bad[charges[1]] += 1
    assert sum(bad.values()) == layout.total

    with pytest.raises(Exception):
        _root, bad_residual = root_parametrize_sym(
            env, a, chi=4, prev_projs=projs, layout_override=BondLayout(dims=bad)
        )
        assert bad_residual > 1e-6, (
            "a wrong layout produced a valid root; the bookkeeping is inert"
        )


@pytest.mark.parametrize("symmetry_name", ["z2", "u1"])
def test_forward_energy_parity_across_symmetries(symmetry_name):
    """Both equal-sector (Z2) and fragmenting (U(1)) layouts.

    Testing only one hides layout bugs: per #566's D-parity finding, equal
    sector sizes collapse to a single block shape, and only the fragmenting
    case exercises the layout arithmetic.
    """
    from tenax.algorithms._ctm_root_implicit_asym import asym_energy, converge
    from tenax import heisenberg_gate

    A = _site_tensor_z2() if symmetry_name == "z2" else _site_tensor()
    gate = heisenberg_gate()
    env_s, _a, meta = converge_sym(A, chi=4, max_iter=60)
    env_d, _ad, _md = converge(A, chi=4, max_iter=60)
    e_sym = float(sym_energy(A, env_s, gate))
    e_den = float(asym_energy(A, env_d, _template_env(A, 4), gate))
    assert abs(e_sym - e_den) < 1e-10
```

Add `_site_tensor_z2()` alongside `_site_tensor()`, identical but with
`ZnSymmetry(2)` and sectors `[0, 1]` on every leg, so sector sizes come out equal.

`root_parametrize_sym` needs a `layout_override` keyword for the trap test — add it as an
optional argument that bypasses the computed layout. It exists only for this test and
should be documented as such.

- [ ] **Step 2: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_symmetric.py -v`
Expected: PASS, 14 passed

If `test_a_wrong_bond_layout_breaks_the_root` *passes silently* (no exception, small
residual), that is a **finding, not a green test**: the layout is not load-bearing.
Stop and investigate before proceeding.

- [ ] **Step 3: Run the full core suite for regressions**

Run: `JAX_PLATFORMS=cpu uv run pytest -m core -q`
Expected: no new failures versus `main`. Record the count.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test(#715): Phase 3 layout trap test and Z2/U(1) coverage"
```

---

## Task 11: Docs and PR

**Files:**
- Modify: `docs/plans/2026-07-31-715-phase3-symmetric-design.md`

- [ ] **Step 1: Record what was actually found**

Update the design doc's §1 caveat with the measured root residual, gradient parity, and
any convention surprise found in Task 5 or 7. If the port turned up a fact that would
have saved hours, write it down there — that is what made the Phase 1 and 2 docs worth
having.

- [ ] **Step 2: Commit and open the PR**

```bash
git add -A && git commit -m "docs(#715): Phase 3 findings from the symmetric port"
git push -u origin feat/715-phase3-symmetric-root-implicit
gh pr create --title "feat(#715): Phase 3 slice 1 — root implicit AD on SymmetricTensor" --body "$(cat <<'BODY'
> 🤖 **AI-generated PR** — written by Claude Code, posted by @yingjerkao.

Ports the Phase 1 asymmetric 1x1 root-implicit gradient to block-sparse
`SymmetricTensor`. Bosonic abelian, 1x1, D=2, chi=4-8.

Design: `docs/plans/2026-07-31-715-phase3-symmetric-design.md`

Gates: forward-energy parity vs the dense module; `norm(F(y*))`; gradient
parity vs the dense root-implicit gradient; a jaxpr assertion that no SVD or
eigh backward survives; and a trap test that a wrong charge layout breaks the
root.

Not in this slice, with blockers named in the design doc §8: fermionic,
multisite symmetric, production wiring, and the #566/#687 measurement.
BODY
)"
```

Per `CLAUDE.md`: merge with `gh pr merge <n> --squash --auto`, and **never** pass
`--delete-branch` — `main` uses a merge queue that deletes the head branch itself, and
the flag closes the PR the moment it enters the queue.

---

## Self-Review

**Spec coverage.** Design §2 representation -> Tasks 1-3, 5-6. §2 kron deleted -> Task 8.
§2 `U_perp` materialised -> Task 2. §3(a) `BondLayout` -> Task 1. §3(b) forward without
densify -> Tasks 4, 7. §3(c) sector projector core -> Task 6. §3(d) root variables ->
Task 8. §3(e) adjoint -> Task 9. §4 projector chain threading -> Task 7 `converge_sym`.
§5 empty/saturated sectors -> Task 1; global floor -> Task 2. §6 all five test tiers ->
Tasks 5 (dense agreement), 8 (root residual), 9 (gradient, jaxpr), 10 (trap, symmetry
coverage). §8 follow-ups -> out of scope by decision, restated in the PR body.

**Placeholder scan.** No TBD/TODO. Three places delegate to a named existing function
rather than repeating ~100 lines (`renormalised_corner_sym` in Task 7,
`characteristic_residual_sym` in Task 8, the adjoint in Task 9); each names the exact
source function and line range and lists the substitutions, because these are *ports of
reviewed code*, and retyping them here would invite drift from the reference that is the
actual specification.

**Type consistency.** `BondLayout(dims=...)` with `.total` / `.charges` throughout.
`sector_svd(matrix, chi, *, row_axis, col_axis)` returns `(dict[int, SectorSVD],
BondLayout)` — used consistently in Tasks 2, 3, 6. `SectorSVD` fields `U, s, Vh,
S_keep_diag, row_key, col_key` — `S_keep_diag` used in Tasks 2 and 6. `SymProjectors`
fields `P_left, P_right, S, sectors, layout` — used in Tasks 6, 7, 8, 10.
`tensor_from_sector_matrices(mats, *, row_index, col_index, row_axis, col_axis)` — used
in Tasks 3, 7, 8. `SymEnv` field names match `CTMTensorEnv`'s, which `_to_ctm_env_sym`
relies on.

**Known soft spots**, flagged so the implementer expects them rather than treating them
as failures: the einsum-to-`contract` relabel translation in Task 5 is the most
error-prone step and has a dense oracle; `TensorIndex.flip_flow` / `relabel` in Task 8
are assumed and given a discovery command; `heisenberg_gate()`'s signature is looked up
rather than guessed.
