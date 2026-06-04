# Stacked-Block Symmetric CTM AD (even-D first) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the symmetric iPEPS CTM-AD compile wall (and per-step runtime) by giving `SymmetricTensor` a *stacked working representation* over the unchanged flat `_data` buffer, so per-block structural op emission collapses to O(n_shape_groups) batched ops — validated even-D-first and gated on a hard ≥10× compile reduction at D=4.

**Architecture:** Add `stacked_blocks()` / `from_stacked_blocks()` to `SymmetricTensor` (one gather+reshape of `_data` per distinct block-shape group; one scatter back). Add a stacked contraction path that sources its batch axis directly from those contiguous views (not `jnp.stack` of per-block slices, which is the existing #571 `_contract_symmetric_batched` source). Route decomposition/fuse through the stacked views, then thread the stacked rep across the CTM sweep so it persists across the ~387 calls. Every phase is asserted against a frozen three-tier accuracy contract built first.

**Tech Stack:** Python, JAX (x64), `jnp.einsum`/`jax.ops.segment_sum`/`jax.vmap`, pytest (`-m core`), the existing `examples/profile_ctm_ad_wall_566.py` profiler for the A100 gate.

**Spec:** `docs/superpowers/specs/2026-06-04-symmetric-ctm-ad-stacked-blocks-design.md`. Scope = P0→P1d. P1e (ragged odd-D/U(1)) and P2 (cuTensorNet) are NOT in this plan — they are funded only if P1d clears ≥10× at D=4.

**Branch:** continue on `docs/symmetric-ctm-ad-stacked-design-566` (or branch a fresh `perf/stacked-blocksparse-566` off it). Open PRs per phase; never push to `main`. Merge via `gh pr merge <n> --auto --delete-branch`.

**Key facts grounded in the code (do not re-derive):**
- `SymmetricTensor` pytree = single leaf `_data` + static aux `(_block_keys, _block_shapes, _block_offsets, _indices)` (`tensor.py:757-794`). `_data` = `concatenate([blocks[k].ravel() for k in sorted(keys)])` (`tensor.py:675-694`). `_get_block(i)` = `_data[off:off+size].reshape(shape)` (`tensor.py:746-753`). `.blocks` unrolls one `_get_block` per block (`tensor.py:987-999`).
- Even-D `FermionParity` (alternating virtual charges 0,1): every block of a tensor shares ONE shape → 1 shape-group → all blocks contiguous in `_data` → `_data.reshape(n_blocks, *shape)` is O(1). Odd-D/U(1) fragment (out of scope here).
- `_contract_symmetric` (`contractor.py:358`) per-combo loop; `_contract_symmetric_batched` (`contractor.py:225-355`) groups survivors by input shape-sig → `jnp.stack` + one `jnp.einsum` + `jax.ops.segment_sum`, gated by `TENAX_BATCH_BLOCKSPARSE` (`contractor.py:540-549`). The grouping/segment logic is correct and reusable; only its **array source** changes (stack-of-slices → contiguous view).
- `_grouped_decomp_by_shape` (`linalg.py:108`), `svd` (`linalg.py:1675`), `qr` (`linalg.py:2228`), `eigh` (`linalg.py:2304`).
- Fermionic even-D site builder: `_build_initial_fpeps_tensor(FPEPSConfig(D=...), key)` (`fermionic_ipeps.py:140`).
- Profiler gate driver: `examples/profile_ctm_ad_wall_566.py` — `make_site_and_gate(sym, D, seed)` (line 161), `build_loss(gate, chi, depth, *, explicit, warmup)` (line 176), CLI `--D --sym --depth --json` (line 295+). A100 baseline fermionic `vg_cmp` at D=4 ≈ 2379 s.

---

## Phase P0 — Accuracy spine (build the guarantee FIRST)

### Task 1: Three-tier comparator + canonical-tensor factory

**Files:**
- Create: `tests/stacked/__init__.py` (empty)
- Create: `tests/stacked/_harness.py`
- Test: `tests/stacked/test_harness.py`

- [ ] **Step 1: Write the failing test for the comparator tiers**

```python
# tests/stacked/test_harness.py
import jax.numpy as jnp
import pytest
from tests.stacked._harness import assert_tiered, canonical_tensors

def test_bit_identical_tier_passes_on_equal():
    a = jnp.array([1.0, 2.0, 3.0])
    assert_tiered(a, a, tier="bit")  # atol=0

def test_bit_identical_tier_fails_on_tiny_diff():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a + 1e-15
    with pytest.raises(AssertionError):
        assert_tiered(a, b, tier="bit")

def test_bounded_fp_tier_passes_within_rtol():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a * (1 + 1e-13)
    assert_tiered(a, b, tier="fp")  # rtol ~1e-12

def test_bounded_fp_tier_fails_outside_rtol():
    a = jnp.array([1.0, 2.0, 3.0])
    b = a * (1 + 1e-6)
    with pytest.raises(AssertionError):
        assert_tiered(a, b, tier="fp")

def test_canonical_tensors_cover_required_cases():
    cases = dict(canonical_tensors())
    # even-D fermionic at D=2 and D=4, a dense U(1)-trivial tensor,
    # a degenerate-SV matrix, and a rank-deficient sector.
    assert {"ferm_D2", "ferm_D4", "dense_D2", "degenerate_sv", "rank_deficient"} <= set(cases)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/stacked/test_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.stacked._harness`.

- [ ] **Step 3: Implement the harness**

```python
# tests/stacked/_harness.py
"""Frozen accuracy spine for the stacked-block migration (#566, P0).

The CURRENT per-block path is the golden reference. Every later phase asserts
against it via assert_tiered with one of three tiers:
  - "bit": bit-identical (atol=0)        — data movement / order-preserved reductions
  - "fp" : bounded-fp (rtol=1e-12, f64)  — reductions XLA may reassociate (segment_sum)
  - gauge-invariants only for SVD/eigh   — never compare raw U/Vh (vmap LAPACK sign flips)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

FP_RTOL = 1e-12
FP_ATOL = 1e-12

def assert_tiered(ref, got, *, tier: str) -> None:
    ref = jnp.asarray(ref)
    got = jnp.asarray(got)
    assert ref.shape == got.shape, f"shape {ref.shape} != {got.shape}"
    if tier == "bit":
        assert bool(jnp.array_equal(ref, got)), (
            f"not bit-identical: max|d|={float(jnp.max(jnp.abs(ref - got))):.3e}"
        )
    elif tier == "fp":
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref), rtol=FP_RTOL, atol=FP_ATOL
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")

def assert_svd_invariants(A_ref, U, S, Vh, *, tier: str = "fp") -> None:
    """Compare gauge-INVARIANT SVD outputs only: singular values + reconstruction."""
    recon = (U * S[..., None, :]) @ Vh
    assert_tiered(jnp.sort(S)[::-1], jnp.sort(S)[::-1], tier="fp")  # S is gauge-invariant
    assert_tiered(A_ref, recon, tier=tier)

def canonical_tensors():
    """Yield (name, SymmetricTensor-or-array) covering the required golden cases."""
    from tenax.algorithms.fermionic_ipeps import _build_initial_fpeps_tensor
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig

    key = jax.random.PRNGKey(0)
    yield "ferm_D2", _build_initial_fpeps_tensor(FPEPSConfig(D=2), key)
    yield "ferm_D4", _build_initial_fpeps_tensor(FPEPSConfig(D=4), key)
    yield "dense_D2", _dense_u1_trivial(D=2, key=key)
    yield "degenerate_sv", _degenerate_sv_matrix()
    yield "rank_deficient", _rank_deficient_matrix()

def _dense_u1_trivial(D, key):
    # A 1-block (trivial-charge) SymmetricTensor == the dense baseline used by the profiler.
    from examples.profile_ctm_ad_wall_566 import make_site_and_gate
    site, _gate = make_site_and_gate("dense", D, seed=0)
    return site

def _degenerate_sv_matrix():
    # 4x4 with a repeated singular value (degenerate subspace).
    return jnp.diag(jnp.array([2.0, 2.0, 1.0, 0.5]))

def _rank_deficient_matrix():
    v = jnp.array([1.0, 2.0, 3.0, 4.0])
    return jnp.outer(v, v)  # rank 1
```

(If `FPEPSConfig` is not importable from `fermionic_ipeps`, import it from its actual module — grep `class FPEPSConfig` and adjust the import; this is the only allowed adaptation.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/stacked/test_harness.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Validate the comparator against EXISTING paths (self-check the spine)**

Add to `tests/stacked/test_harness.py`:

```python
def test_perblock_vs_batched_contract_is_tiered():
    """Sanity: existing batched path matches per-block within the fp tier.
    Proves the comparator is correctly calibrated before any new code exists."""
    import os
    from tenax.contraction.contractor import contract
    from tests.stacked._harness import canonical_tensors
    A = dict(canonical_tensors())["ferm_D2"]
    Abar = A.bar().relabels({"u": "U", "d": "Dn", "l": "L", "r": "R"})
    os.environ["TENAX_BATCH_BLOCKSPARSE"] = "0"
    ref = contract(A, Abar)  # per-block
    os.environ["TENAX_BATCH_BLOCKSPARSE"] = "1"
    got = contract(A, Abar)  # existing batched
    os.environ["TENAX_BATCH_BLOCKSPARSE"] = "0"
    assert_tiered(ref._data, got._data, tier="fp")
```

(Adjust `.bar()`/relabel call to the real API — grep `def bar` and `def relabels` in `tensor.py`; the double-layer must not collapse to a scalar, so phys is the only shared label.)

Run: `uv run pytest tests/stacked/test_harness.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/stacked/
git commit -m "test(#566): P0 three-tier accuracy comparator + canonical-tensor factory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase P1a — `StackedView` + bit-exact round-trip

### Task 2: `stacked_blocks()` / `from_stacked_blocks()` on `SymmetricTensor`

**Files:**
- Create: `src/tenax/core/stacked_view.py`
- Modify: `src/tenax/core/tensor.py` (add two methods near `_get_block`, ~line 753)
- Test: `tests/stacked/test_stacked_view.py`

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/stacked/test_stacked_view.py
import jax.numpy as jnp
import pytest
from tests.stacked._harness import canonical_tensors

@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4", "dense_D2"])
def test_round_trip_bit_exact(name):
    A = dict(canonical_tensors())[name]
    view = A.stacked_blocks()
    B = A.from_stacked_blocks(view)
    assert bool(jnp.array_equal(A._data, B._data)), "round-trip not bit-exact"
    assert A._block_keys == B._block_keys
    assert A._block_shapes == B._block_shapes
    assert A._block_offsets == B._block_offsets

def test_even_D_is_single_shape_group():
    A = dict(canonical_tensors())["ferm_D4"]
    view = A.stacked_blocks()
    # even-D FermionParity: all blocks one shape -> exactly one group
    assert len(view.groups) == 1
    (shape, grp), = view.groups.items()
    assert grp.array.shape[0] == len(A._block_keys)   # leading block axis
    assert grp.array.shape[1:] == shape

def test_stacked_block_values_match_get_block():
    A = dict(canonical_tensors())["ferm_D2"]
    view = A.stacked_blocks()
    for shape, grp in view.groups.items():
        for j, key in enumerate(grp.keys):
            idx = A._block_keys.index(key)
            assert bool(jnp.array_equal(grp.array[j], A._get_block(idx)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/stacked/test_stacked_view.py -v`
Expected: FAIL — `AttributeError: 'SymmetricTensor' object has no attribute 'stacked_blocks'`.

- [ ] **Step 3: Implement `StackedView`**

```python
# src/tenax/core/stacked_view.py
"""Stacked working representation over the flat _data buffer (#566 P1a).

A SymmetricTensor's blocks are grouped by identical shape; each group is one
array (n_blocks_in_group, *block_shape) obtained by gathering the group's flat
ranges out of _data and reshaping. Grouping + gather indices are STATIC
(computed from block metadata), so only the gather/reshape touches the tracer.
For even-D FermionParity (one shape, contiguous) the gather is a plain slice.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

BlockKey = tuple

@dataclass(frozen=True)
class StackGroup:
    keys: tuple                  # block keys in this group, in sorted-key order
    array: jax.Array             # (n_blocks, *block_shape)

@dataclass(frozen=True)
class StackedView:
    groups: dict                 # block_shape -> StackGroup
    indices: tuple               # the source tensor's TensorIndex tuple

def _group_layout(block_keys, block_shapes, block_offsets):
    """STATIC: shape -> (keys, flat_gather_index_array). No tracer ops."""
    by_shape: dict = {}
    for i, shape in enumerate(block_shapes):
        size = int(np.prod(shape)) if shape else 1
        off = block_offsets[i]
        flat_idx = np.arange(off, off + size, dtype=np.int64)
        rec = by_shape.setdefault(shape, ([], []))
        rec[0].append(block_keys[i])
        rec[1].append(flat_idx)
    layout = {}
    for shape, (keys, idx_lists) in by_shape.items():
        layout[shape] = (tuple(keys), np.stack(idx_lists, axis=0))  # (n, size)
    return layout

def build_stacked(data, block_keys, block_shapes, block_offsets, indices) -> StackedView:
    layout = _group_layout(block_keys, block_shapes, block_offsets)
    groups = {}
    for shape, (keys, gather) in layout.items():
        n = gather.shape[0]
        # one gather + reshape per group; contiguous even-D case = a slice
        flat = data[gather.reshape(-1)]            # (n*size,)
        groups[shape] = StackGroup(keys=keys, array=flat.reshape((n, *shape)))
    return StackedView(groups=groups, indices=indices)

def scatter_stacked(view, block_keys, block_shapes, block_offsets, total_size, dtype):
    """Inverse: write groups back into one flat buffer in canonical layout."""
    layout = _group_layout(block_keys, block_shapes, block_offsets)
    data = jnp.zeros(total_size, dtype=dtype)
    for shape, (keys, gather) in layout.items():
        grp = view.groups[shape]
        # reorder grp.array rows to canonical key order, then scatter (one op/group)
        order = [grp.keys.index(k) for k in keys]
        rows = grp.array[jnp.asarray(order)]
        data = data.at[jnp.asarray(gather.reshape(-1))].set(rows.reshape(-1))
    return data
```

- [ ] **Step 4: Wire the two methods onto `SymmetricTensor`**

Add after `_get_block` (`tensor.py:753`):

```python
    def stacked_blocks(self):
        """Return a StackedView: blocks grouped by shape, one array per group."""
        from tenax.core.stacked_view import build_stacked
        return build_stacked(
            self._data, self._block_keys, self._block_shapes,
            self._block_offsets, self._indices,
        )

    def from_stacked_blocks(self, view):
        """Rebuild a SymmetricTensor from a StackedView (canonical layout)."""
        from tenax.core.stacked_view import scatter_stacked
        total = self._block_offsets[-1] + (
            int(np.prod(self._block_shapes[-1])) if self._block_shapes[-1] else 1
        ) if self._block_keys else 0
        data = scatter_stacked(
            view, self._block_keys, self._block_shapes, self._block_offsets,
            total, self._data.dtype,
        )
        return SymmetricTensor._raw(
            indices=self._indices, data=data, block_keys=self._block_keys,
            block_shapes=self._block_shapes, block_offsets=self._block_offsets,
        )
```

(`np` is already imported in `tensor.py` — confirm via grep `import numpy`.)

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/stacked/test_stacked_view.py -v`
Expected: PASS (5 cases).

- [ ] **Step 6: Run the broader core suite for regressions**

Run: `uv run pytest -m core tests/test_block_array.py tests/test_contraction.py -q`
Expected: PASS (no regressions — we only ADDED methods).

- [ ] **Step 7: Commit**

```bash
git add src/tenax/core/stacked_view.py src/tenax/core/tensor.py tests/stacked/test_stacked_view.py
git commit -m "feat(#566): StackedView + stacked_blocks/from_stacked_blocks round-trip (P1a)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase P1b — Stacked contraction path (even-D), gated

### Task 3: `_contract_symmetric_stacked` sourcing from contiguous views

**Files:**
- Modify: `src/tenax/contraction/contractor.py` (new fn + dispatch near line 540)
- Test: `tests/stacked/test_stacked_contract.py`

**Design note:** reuse the survivor-grouping + `segment_sum` logic of `_contract_symmetric_batched` (correct, shipped). The ONLY change: when every input tensor has a single shape-group (even-D), take the batch axis straight from `tensor.stacked_blocks()` (a contiguous `_data.reshape`) instead of `jnp.stack` of per-block slices. Otherwise return `None` → caller falls back to the per-block path. New gate `TENAX_STACK_BLOCKSPARSE` (independent of `TENAX_BATCH_BLOCKSPARSE`).

- [ ] **Step 1: Write the failing equivalence test (golden tiers)**

```python
# tests/stacked/test_stacked_contract.py
import os
import jax.numpy as jnp
import pytest
from tenax.contraction.contractor import contract
from tests.stacked._harness import assert_tiered, canonical_tensors

def _double_layer(A):
    # contract physical leg only -> double-layer; relabel virtuals so nothing else matches
    Abar = A.bar().relabels({"u": "U", "d": "Dn", "l": "L", "r": "R"})
    return A, Abar

@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_stacked_contract_matches_perblock(name):
    A = dict(canonical_tensors())[name]
    A, Abar = _double_layer(A)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    ref = contract(A, Abar)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "1"
    got = contract(A, Abar)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    assert got._block_keys == ref._block_keys
    assert_tiered(ref._data, got._data, tier="fp")

def test_stacked_contract_grad_matches_perblock(name="ferm_D2"):
    import jax
    A0 = dict(canonical_tensors())[name]
    def loss(data, flag):
        os.environ["TENAX_STACK_BLOCKSPARSE"] = flag
        A = A0.from_stacked_blocks(A0.stacked_blocks())  # identity, keeps it a tensor
        A = type(A0)._raw(indices=A0._indices, data=data, block_keys=A0._block_keys,
                          block_shapes=A0._block_shapes, block_offsets=A0._block_offsets)
        Ab = A.bar().relabels({"u": "U", "d": "Dn", "l": "L", "r": "R"})
        out = contract(A, Ab)
        return jnp.sum(out._data ** 2)
    g_ref = jax.grad(loss)(A0._data, "0")
    g_got = jax.grad(loss)(A0._data, "1")
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    assert_tiered(g_ref, g_got, tier="fp")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/stacked/test_stacked_contract.py -v`
Expected: FAIL — `got._data` differs OR stacked path absent (flag is a no-op → values equal but test for the NEW path is meaningless). Confirm failure is real by asserting the stacked branch executed (Step 3 adds a counter).

- [ ] **Step 3: Implement `_contract_symmetric_stacked` + dispatch**

In `contractor.py`, add a module counter and the function:

```python
_STACK_FIRED = {"n": 0}  # test observability

def _all_single_shape_group(tensors) -> bool:
    return all(len(set(t._block_shapes)) <= 1 for t in tensors)

def _contract_symmetric_stacked(tensors, subscripts, output_indices, optimize):
    """Even-D fast path: batch axis sourced from contiguous _data views.
    Returns None to fall back when inputs are not single-shape-group."""
    if len(tensors) != 2 or not _all_single_shape_group(tensors):
        return None
    _STACK_FIRED["n"] += 1
    # Build survivor groups exactly as _contract_symmetric_batched does, but
    # read each operand's blocks from tensor.stacked_blocks() (one reshape).
    # ... reuse the survivor-collection + segment_sum from the batched path,
    #     substituting `stacked_view.groups[shape].array` for jnp.stack(...).
    # (Implementation mirrors contractor.py:225-355; only the array source and
    #  the per-tensor key->row index map change.)
    ...
    return SymmetricTensor._from_blocks_unchecked(output_blocks, output_indices)
```

Wire dispatch at the top of `_contract_symmetric` (after the cuTENSOR block, ~line 410):

```python
    if os.environ.get("TENAX_STACK_BLOCKSPARSE", "0") == "1":
        stacked = _contract_symmetric_stacked(
            list(tensors), subscripts, output_indices, optimize
        )
        if stacked is not None:
            return stacked
```

**The `...` is filled by porting `_contract_symmetric_batched`'s body** (lines 249-355): collect `survivors` with their `output_key`, group by input shape-sig, but for each operand build its stacked array once via `tensors[p].stacked_blocks().groups[shape].array` and index rows by the operand's `key -> row` map instead of `jnp.stack([slice...])`. Keep the `segment_sum` accumulation verbatim (this is what makes accumulating combos correct — the earlier prototype's bug was skipping it).

- [ ] **Step 4: Make the test assert the stacked branch fired**

Append to both tests: `from tenax.contraction.contractor import _STACK_FIRED` and assert `_STACK_FIRED["n"] > 0` after the flag=1 contraction.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/stacked/test_stacked_contract.py -v`
Expected: PASS — values match within `fp` tier, grad matches, stacked branch fired.

- [ ] **Step 6: Run contraction + fermionic suites for regressions**

Run: `uv run pytest -m core tests/test_contraction.py tests/test_fermionic.py tests/test_symmetric_custom_vjp.py -q`
Expected: PASS (flag default-off → byte-identical to today).

- [ ] **Step 7: Commit**

```bash
git add src/tenax/contraction/contractor.py tests/stacked/test_stacked_contract.py
git commit -m "feat(#566): even-D stacked contraction path (TENAX_STACK_BLOCKSPARSE) (P1b)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase P1c — Stacked decomposition + fuse/split (even-D)

### Task 4: Route svd/qr/eigh through stacked views (gauge-invariant assertions)

**Files:**
- Modify: `src/tenax/linalg.py` (svd `:1675`, qr `:2228`, eigh `:2304` — gate the stacked source under `TENAX_STACK_BLOCKSPARSE`, reuse `_grouped_decomp_by_shape` `:108`)
- Test: `tests/stacked/test_stacked_decomp.py`

- [ ] **Step 1: Write the failing test (invariants only — never raw U/Vh)**

```python
# tests/stacked/test_stacked_decomp.py
import os
import jax.numpy as jnp
import pytest
from tenax import linalg
from tests.stacked._harness import canonical_tensors, assert_tiered

@pytest.mark.parametrize("name", ["ferm_D2", "ferm_D4"])
def test_stacked_svd_invariants_match(name):
    A = dict(canonical_tensors())[name]
    M = A.fuse(("u", "d", "l"), "row").fuse(("r", "phys"), "col")  # to a matrix; adjust to real fuse API
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    U0, S0, Vh0 = linalg.svd(M, "row", "col")
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "1"
    U1, S1, Vh1 = linalg.svd(M, "row", "col")
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    # singular values are gauge-invariant -> fp tier
    assert_tiered(jnp.sort(S0._data), jnp.sort(S1._data), tier="fp")
    # reconstruction is gauge-invariant -> fp tier; raw U/Vh are NOT compared
    from tenax.contraction.contractor import contract
    rec0 = contract(contract(U0, S0), Vh0)
    rec1 = contract(contract(U1, S1), Vh1)
    assert_tiered(rec0._data, rec1._data, tier="fp")
```

(Adjust `fuse`/`svd` calls to the real signatures — grep `def fuse`, `def svd`; the point is: assert SVs + reconstruction, never raw factors.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/stacked/test_stacked_decomp.py -v`
Expected: FAIL — stacked decomp source not yet gated / no branch executed.

- [ ] **Step 3: Implement the stacked decomposition source**

In `linalg.svd`/`qr`/`eigh`, when `TENAX_STACK_BLOCKSPARSE=1` AND the matrix is single-shape-group, build the batched `(n, m, k)` array via `M.stacked_blocks()` and call the existing `_grouped_decomp_by_shape` machinery (already does `vmap(..., in_axes=(0, None))` over same-shape sectors). Reconstruct factor tensors via `from_stacked_blocks`. For non-uniform → existing per-block path. Gauge-fix consistently (same convention as the per-block path).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/stacked/test_stacked_decomp.py -v`
Expected: PASS (SV + reconstruction within `fp`).

- [ ] **Step 5: Degenerate + rank-deficient guard test**

```python
def test_degenerate_and_rank_deficient_invariants():
    import numpy as np
    for nm in ("degenerate_sv", "rank_deficient"):
        M = dict(canonical_tensors())[nm]  # plain jnp matrix
        U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
        recon = (U * S[None, :]) @ Vh
        np.testing.assert_allclose(np.asarray(recon), np.asarray(M), rtol=1e-12, atol=1e-12)
```

Run: `uv run pytest tests/stacked/test_stacked_decomp.py -v` → PASS.

- [ ] **Step 6: Run linalg + padded-linalg suites**

Run: `uv run pytest -m core tests/test_linalg.py tests/test_padded_linalg.py tests/test_symmetric_custom_vjp.py -q`
Expected: PASS.

- [ ] **Step 7: fuse/split — correctness is already guaranteed; optimize only if hot**

`fuse`/`split` round-trip correctness is covered by P1a's bit-exact round-trip test (Task 2). Do NOT add a stacked fuse/split path speculatively (YAGNI). Only route fuse/split through stacked views if the P1d op-histogram (`examples/probe_backward_jaxpr_566.py`) shows `_fuse_indices_symmetric` is still a material structural-op source on the sweep after Tasks 3–4. If so, add it here behind the same flag with a bit-exact round-trip test; otherwise note "fuse/split left per-block — not hot" and move on.

- [ ] **Step 8: Commit**

```bash
git add src/tenax/linalg.py tests/stacked/test_stacked_decomp.py
git commit -m "feat(#566): even-D stacked svd/qr/eigh via grouped decomp (P1c)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase P1d — Thread through the CTM sweep + the hard A100 gate

### Task 5: Persist the stacked rep across the sweep + end-to-end golden

**Files:**
- Modify: the CTM sweep / absorb / RDM call sites (discover via `grep -rn "stacked_blocks\|\.blocks\b\|_get_block" src/tenax/algorithms/` and the CTM-AD modules) so env tensors stay stacked across calls and pack to `_data` only at sweep boundaries.
- Test: `tests/stacked/test_ctm_ad_end_to_end.py`

- [ ] **Step 1: Write the failing end-to-end golden test**

```python
# tests/stacked/test_ctm_ad_end_to_end.py
import os
import jax
import jax.numpy as jnp
import pytest
from tests.stacked._harness import assert_tiered
from examples.profile_ctm_ad_wall_566 import make_site_and_gate, build_loss

@pytest.mark.parametrize("D", [2, 4])
def test_ctm_ad_energy_and_grad_match_golden(D):
    site, gate = make_site_and_gate("fermionic", D, seed=0)
    loss = build_loss(gate, chi=3 * D, depth=8, explicit=False, warmup=0)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    e0, g0 = jax.value_and_grad(loss)(site._data)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "1"
    e1, g1 = jax.value_and_grad(loss)(site._data)
    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    assert_tiered(jnp.asarray(e0), jnp.asarray(e1), tier="fp")  # energy: bounded-fp
    assert_tiered(g0, g1, tier="fp")                            # gradient: bounded-fp
```

(Adjust `build_loss` call to its real return contract — read `examples/profile_ctm_ad_wall_566.py:176`; it may take/return the site tensor rather than raw `_data`. Keep the assertion: stacked vs per-block energy+grad within `fp`.)

- [ ] **Step 2: Run to verify it fails (or reveals the real integration gap)**

Run: `uv run pytest tests/stacked/test_ctm_ad_end_to_end.py -v`
Expected: FAIL or mismatch — the sweep still round-trips through `.blocks`/`_get_block` per call, so flag=1 either changes nothing or diverges where stacked decomp/contract aren't yet on the sweep path. Use the failure to locate the un-migrated call sites.

- [ ] **Step 3: Migrate the sweep to carry stacked tensors across calls**

Replace per-call `.blocks`/`_get_block` round-trips in the absorb/projector/RDM steps so a tensor produced stacked stays stacked into the next contraction; pack to `_data` only at the sweep boundary (where the jit/scan carry needs the single leaf). One call site per commit where feasible; re-run Step 1 after each.

- [ ] **Step 4: Run to verify the end-to-end test passes**

Run: `uv run pytest tests/stacked/test_ctm_ad_end_to_end.py -v`
Expected: PASS (energy + grad within `fp` for D=2 and D=4).

- [ ] **Step 5: Full core + algorithm regression**

Run: `uv run pytest -m core -q` then `uv run pytest -m "not slow" tests/test_fpeps_ad.py tests/test_block_sparse_ctm_ad.py -q`
Expected: PASS.

- [ ] **Step 6: Commit the migration**

```bash
git add -A && git commit -m "feat(#566): thread stacked rep across CTM sweep; end-to-end golden (P1d)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 6: The hard A100 compile gate (≥10× at D=4)

**Files:**
- Use: `examples/profile_ctm_ad_wall_566.py` (already supports `--D --sym --json` and honest cold compiles via fresh `jax_compilation_cache_dir`).
- Create: `examples/stacked_gate_566_summary.md` (the recorded verdict).

- [ ] **Step 1: Baseline (per-block) compile at D=4 on A100**

Run on the A100 host (f64, implicit/fixed_point):
```bash
TENAX_STACK_BLOCKSPARSE=0 uv run python examples/profile_ctm_ad_wall_566.py \
  --D 4 --sym fermionic --depth 8 --reps 3 --json examples/stacked_gate_D4_off.json
```
Record `vg_cmp` (expect ≈ 2379 s, matching the prior baseline).

- [ ] **Step 2: Stacked compile at D=4 on A100**

```bash
TENAX_STACK_BLOCKSPARSE=1 uv run python examples/profile_ctm_ad_wall_566.py \
  --D 4 --sym fermionic --depth 8 --reps 3 --json examples/stacked_gate_D4_on.json
```

- [ ] **Step 3: Evaluate the gate**

Compute `ratio = vg_cmp(off) / vg_cmp(on)`.
- **PASS if `ratio >= 10`** (i.e. on ≤ ~238 s). Also record the warm-step ratio (secondary, non-gating).
- **FAIL otherwise → STOP.** Do not start P1e or P2. Write the verdict + numbers to `examples/stacked_gate_566_summary.md`, post to #566, and bring the result back for an approach review.

- [ ] **Step 4: Commit the gate artifacts + verdict**

```bash
git add examples/stacked_gate_D4_*.json examples/stacked_gate_566_summary.md
git commit -m "bench(#566): P1d A100 compile gate result at D=4 (PASS/FAIL: <ratio>x)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Update the #566 memory + issue with the measured ratio and the P1e/P2 go/no-go.**

---

## Final integration

- [ ] Open the phase PRs (or one stacked PR) targeting `main`; ensure `Tests (Python 3.11/3.12)` + `Tests (macOS, Python 3.12)` pass.
- [ ] Default remains `TENAX_STACK_BLOCKSPARSE=0` until the gate PASSES; flip to default-on (or device-aware) only as a follow-up PR justified by the A100 number.
- [ ] If the gate PASSES, file the P1e (ragged) and P2 (cuTensorNet) follow-up issues with the measured even-D win as justification.

## Notes for the implementer

- **Never compare raw `U`/`Vh`** from SVD/eigh — only singular values, eigenvalues, and reconstruction (`assert_svd_invariants`). `vmap`-ed LAPACK sign-/basis-flips degenerate subspaces (the #572 trap).
- **Keep `segment_sum`** in the stacked contraction — it is what makes accumulating block-combos correct; the earlier dead-end prototype's correctness bug came from dropping it.
- **Do not revert the flat buffer or add pytree leaves** — `_data`-as-single-leaf is #87's deliberate JIT/grad substrate and the #200/#568/#569 accelerator substrate.
- **Even-D only** in this plan. If a stacked path hits a non-single-shape-group tensor, it must return `None`/fall back, never pad — padding is P1e, behind its own flag.
- Run `uv run pytest -m core` (the CI-required marker) before each commit; `-m "not slow"` for the AD integration tests.
