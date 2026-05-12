# Tracer-Safe Symmetric 2x2 Projector (Issue #435) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SymmetricTensor 2x2 projector path tracer-safe so AD-traced symmetric inputs with non-trivial U(1) charges flow through the block-sparse pipeline instead of the dense fallback. Unblocks 4 `xfail` tests pinned to #435.

**Architecture:** Three coordinated edits. (1) New `_truncated_svd_symmetric_traced` in `linalg.py` composes the existing dense Lorentzian-regularized AD primitive (`truncated_svd_ad`) per sector, allocating per-sector keep counts statically via `_derive_charges(base_charges, chi)`. (2) Pure-JAX rewrite of `_gauge_fix_symmetric_svd` removes `int(jnp.argmax(...))` / `complex(col_flat[idx])` casts. (3) `base_charges` plumbed through `_ctm_tensor_moves.py` 2plaq path, mirroring PR #437. Drop the `_has_tracer` filter so tracer-bearing symmetric inputs use the symmetric path.

**Tech Stack:** `jnp.linalg.svd`, `truncated_svd_ad` (Lorentzian-regularized custom_vjp), `SymmetricTensor`, `_derive_charges`, `_get_base_charges`, `tenax.linalg.svd`.

**Spec:** [`docs/superpowers/specs/2026-05-12-issue-435-tracer-safe-2x2-projector-design.md`](../specs/2026-05-12-issue-435-tracer-safe-2x2-projector-design.md).

**Predecessor:** [`docs/superpowers/plans/2026-05-11-2x2-projector-symmetric.md`](2026-05-11-2x2-projector-symmetric.md) (PR #434 — eager symmetric path that introduced the dense fallback).

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/tenax/linalg.py` | **modify** | Add `_truncated_svd_symmetric_traced` (lines added after existing `_truncated_svd_symmetric`); add tracer-aware dispatch at entry of `_truncated_svd_symmetric`; add `base_charges: np.ndarray \| None = None` kwarg to `svd()` and forward to `_truncated_svd_symmetric` |
| `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` | **modify** | Pure-JAX rewrite of `_gauge_fix_symmetric_svd` (lines 53-148); replace `S_Mp[0]` with `jnp.max(S_Mp)` at line 960; drop `_has_tracer` filter at lines 411-426 in `_compute_2x2_projector`; thread `base_charges` from `_compute_2x2_projector_symmetric` into `tensor_svd` calls |
| `src/tenax/algorithms/_ctm_tensor_moves.py` | **modify** | Add `base_charges` kwarg to `_compute_2x2_projector_2plaq` and forward to `_compute_2x2_projector`; update the four multisite move helpers that call it; derive `base_charges = _get_base_charges(a)` at the multisite-sweep entry |
| `tests/test_ctm_2x2_projector_symmetric.py` | **modify** | Add 5 new tests (tracer-safety, sector-preservation, eager-vs-traced equivalence, gradient FD, closure under tracing) |
| `tests/test_ipeps.py` | **modify** | Drop 1 `@pytest.mark.xfail(reason="Issue #435 ...")` decorator |
| `tests/test_fpeps_ad.py` | **modify** | Drop 3 `@pytest.mark.xfail(reason="Issue #435 ...")` decorators |

Estimated diff: ~250 LOC added in source, ~200 LOC added in tests, ~40 LOC removed (xfail decorators + `_has_tracer` filter).

---

## Task 1: Rewrite `_gauge_fix_symmetric_svd` in pure JAX

**Why:** The current implementation uses `int(jnp.argmax(jnp.abs(col_flat)))` (line 112) and `complex(col_flat[local_max_idx])` (line 113), which raise `TracerArrayConversionError` under `jax.grad`. The outer Python loop over `j` is over a static numpy `bond_charges` array — that's fine — but the per-iteration body must be JAX-traceable.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:53-148`
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append new test)

- [ ] **Step 1: Write the failing tracer-safety test**

Append to `tests/test_ctm_2x2_projector_symmetric.py`:

```python
def test_gauge_fix_symmetric_svd_tracer_safe():
    """`_gauge_fix_symmetric_svd` does not raise TracerArrayConversionError under jax.grad."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=42)

    def loss(alpha: jax.Array) -> jax.Array:
        # Scale blocks by alpha to inject a tracer into block contents
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        U_T, _, Vh_T, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=None,
        )
        U_fixed, _ = _gauge_fix_symmetric_svd(U_T, Vh_T)
        # Return a scalar derived from the fixed U
        return jnp.sum(jnp.abs(jnp.concatenate([b.flatten() for b in U_fixed.blocks.values()])))

    grad_fn = jax.grad(loss)
    g = grad_fn(jnp.asarray(1.0))
    assert jnp.isfinite(g), f"gradient through gauge-fixed SVD must be finite, got {g}"
```

- [ ] **Step 2: Run the new test, expect failure**

```bash
cd /home/yjkao/tenax/.worktrees/issue-435-tracer-safe-2x2-design
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_tracer_safe -xvs
```

Expected: `TracerArrayConversionError` or `ConcretizationTypeError` raised inside `_gauge_fix_symmetric_svd` at the `int(...)` or `complex(...)` cast. (Note: this test will also exercise `_truncated_svd_symmetric` — if it fails earlier on the SVD truncation step, that confirms Task 2 is needed as well; the test will move from "fail at line ~232 of linalg.py" to "fail at line 112 of `_ctm_tensor_projector_2x2.py`" after Task 2.)

- [ ] **Step 3: Replace the inner loop body with pure-JAX equivalents**

In `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`, replace lines 99-148 (the `for j, q in enumerate(bond_charges)` loop body and the wrap-up) with:

```python
    # Detect dtype statically so we don't promote real blocks to complex.
    sample_block = next(iter(U_T.blocks.values()))
    is_complex = jnp.issubdtype(sample_block.dtype, jnp.complexfloating)

    # For each global column j, compute its phase and write it back.
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local = local_index_of[q_int][j]
        u_entries = u_blocks_by_q.get(q_int, [])
        vh_entries = vh_blocks_by_q.get(q_int, [])

        if not u_entries:
            # No U-blocks for this charge — should not occur in practice; skip.
            continue

        # Stack matching block column slices into a single 1-D array (static structure;
        # the values inside are traced).
        candidates = jnp.concatenate(
            [
                jnp.reshape(new_u_blocks[key][..., local], (-1,))
                for key, _ in u_entries
            ]
        )
        max_idx = jnp.argmax(jnp.abs(candidates))
        best_value = candidates[max_idx]
        abs_best = jnp.abs(best_value)
        phase = jnp.where(
            abs_best > 0,
            best_value / jnp.maximum(abs_best, jnp.asarray(1e-30, dtype=abs_best.dtype)),
            jnp.ones_like(best_value),
        )

        if is_complex:
            conj_phase = jnp.conj(phase)
            bare_phase = phase
        else:
            # For real-input SVD, phase is ±1; keep dtype matched so we don't trigger
            # the JAX complex-to-real cast warning.
            conj_phase = jnp.real(phase)
            bare_phase = jnp.real(phase)

        # Apply conj(phase) to column `local` of every matching U-block.
        for key, _block in u_entries:
            new_block = new_u_blocks[key]
            new_block = new_block.at[..., local].multiply(conj_phase)
            new_u_blocks[key] = new_block

        # Apply phase to row `local` of every matching Vh-block.
        for key, _block in vh_entries:
            new_block = new_vh_blocks[key]
            new_block = new_block.at[local, ...].multiply(bare_phase)
            new_vh_blocks[key] = new_block

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out
```

Keep lines 53-98 unchanged (the static index-builder portion using numpy is already traceable).

- [ ] **Step 4: Run the tracer-safety test and the existing reconstruction test**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_tracer_safe \
  tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_preserves_reconstruction \
  tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_real_positive_max_row \
  -xvs
```

Expected: the tracer-safety test may still fail if `_truncated_svd_symmetric` is the upstream blocker (proceed to Task 2 and re-run). The reconstruction and real-positive-max-row tests must pass — they prove the rewrite did not change behavior in the eager case.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "$(cat <<'EOF'
refactor(ctm): rewrite _gauge_fix_symmetric_svd in pure JAX (#435)

Replace `int(jnp.argmax(...))` / `complex(col_flat[idx])` casts with
jnp.argmax + dynamic-index over concatenated block slices. Per-column
phase is now a JAX scalar; real-vs-complex dispatch happens statically
based on input dtype.

The outer Python loop is unchanged (bounded by static bond_charges
length); only the per-iteration body becomes traceable. Reconstruction
behavior is preserved (existing reconstruction test still passes).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `_truncated_svd_symmetric_traced` and `base_charges` kwarg to `linalg.svd`

**Why:** `_truncated_svd_symmetric` does `np.array(s_q)` and `float(val)` at lines 231-233, then sorts SVs across sectors in Python — none of which is traceable. Add a parallel function that takes per-sector AD-primitive SVDs and allocates keep counts statically. Plumb `base_charges` through `linalg.svd` so the traced branch can use it.

**Files:**
- Modify: `src/tenax/linalg.py` (extend `_truncated_svd_symmetric`, add new helper, extend `svd` signature)
- Test: `tests/test_ctm_2x2_projector_symmetric.py` (append new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_2x2_projector_symmetric.py`:

```python
def test_truncated_svd_symmetric_traced_preserves_charges():
    """Tracer-bearing symmetric SVD produces non-trivial bond charges, runs under jax.grad."""
    from tenax.linalg import svd as tensor_svd

    M_T = _make_test_matrix_tensor(seed=7)
    base_charges = np.array([0, 1, 0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        new_blocks = {k: alpha * b for k, b in M_T.blocks.items()}
        M_scaled = SymmetricTensor._from_blocks_unchecked(new_blocks, M_T.indices)
        U_T, s, _, _ = tensor_svd(
            M_scaled,
            left_labels=("left",),
            right_labels=("right",),
            new_bond_label="bond",
            max_singular_values=3,
            base_charges=base_charges,
        )
        # Sanity: bond charges should NOT all be zero
        bond_charges = np.asarray(U_T.indices[-1].charges)
        # Capture the check inside an assertion that survives tracing
        assert not np.all(bond_charges == 0), (
            f"traced symmetric SVD must preserve non-trivial U(1) charges, got {bond_charges}"
        )
        return jnp.sum(s)

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g), f"gradient must be finite, got {g}"
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_truncated_svd_symmetric_traced_preserves_charges -xvs
```

Expected: `TypeError: svd() got an unexpected keyword argument 'base_charges'` OR `TracerArrayConversionError` inside `_truncated_svd_symmetric`.

- [ ] **Step 3: Add `base_charges` kwarg to `svd()` and `_truncated_svd_symmetric`**

In `src/tenax/linalg.py`, update the `svd` function signature (line 1128) to add the new kwarg before `normalize`:

```python
def svd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, jax.Array, Tensor, jax.Array]:
```

Add to the docstring under Args:

```
        base_charges:         Optional per-sector charge vector consumed by the
                              symmetric block-sparse path under JAX tracing.  When
                              supplied, traced inputs use
                              ``_derive_charges(base_charges, max_singular_values)``
                              for static per-sector keep allocation.  Ignored on
                              the dense path.
```

Update the dispatch call (line 1194-1203) to forward:

```python
    if isinstance(tensor, SymmetricTensor):
        return _truncated_svd_symmetric(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            max_truncation_err,
            new_bond_label,
            normalize,
            base_charges=base_charges,
        )
```

Update `_truncated_svd_symmetric` (line 91) signature:

```python
def _truncated_svd_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
```

- [ ] **Step 4: Add tracer dispatch at the entry of `_truncated_svd_symmetric`**

Insert at line 108 of `linalg.py` (immediately after the docstring), before the existing `all_labels = ...` line:

```python
    # Tracer-aware dispatch: if any block carries a JAX tracer (AD backward),
    # the Python-level global SV sort at lines 230-243 cannot run.  Route to
    # the traced variant that does per-sector static allocation.
    is_traced = any(
        isinstance(b, jax.core.Tracer) for b in tensor.blocks.values()
    )
    if is_traced and tensor.blocks:
        return _truncated_svd_symmetric_traced(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            new_bond_label,
            normalize,
            base_charges=base_charges,
        )
```

- [ ] **Step 5: Implement `_truncated_svd_symmetric_traced`**

Add immediately after `_truncated_svd_symmetric` ends (find the closing `return` of that function and add the new function below it). Per the spec §"Component 1 — `_truncated_svd_symmetric_traced`":

```python
def _truncated_svd_symmetric_traced(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    new_bond_label: Label,
    normalize: bool,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
    """Tracer-safe block-diagonal SVD for SymmetricTensor.

    Used under JAX tracing (e.g. AD backward through implicit-FP GMRES).  Each
    charge sector is SVD'd independently via :func:`truncated_svd_ad`, which
    applies Francuz et al. Lorentzian regularization per block.

    Allocation rule (static, no global SV sort):
      * If both ``base_charges`` and ``max_singular_values`` are provided:
        ``k_q = min(target_count[q], available_q)`` where
        ``target_count[q]`` is the count of ``q`` in
        ``_derive_charges(base_charges, max_singular_values)``.
      * If ``max_singular_values`` is None: ``k_q = min(rows_q, cols_q)``
        (full spectrum per sector).
      * Else (defensive fallback, base_charges=None and truncating):
        ``k_q = max(1, round(max_singular_values * available_q / total_available))``,
        adjusted so totals do not exceed ``max_singular_values``.

    The bond axis emerges in sector-block order, NOT global SV-descending
    order.  This differs from the eager path's output ordering; tensor
    contractions match by charge identity per block, so the permutation is
    safe.  Positional reads of S (e.g. ``S[0]``) should use ``jnp.max(S)``
    instead — see ``_compute_2x2_projector_symmetric`` line 960.

    Returns ``(U, s_truncated, Vh, s_full)``; under tracing ``s_full = s_truncated``
    (no pre-truncation spectrum tracked separately).
    """
    from tenax.algorithms._ad_primitives import truncated_svd_ad
    from tenax.algorithms._ctm_utils import _derive_charges

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector results: q -> (U_q matrix, s_q, Vh_q matrix, left_subkeys,
    # right_subkeys, left_row_sizes, right_col_sizes, available_q)
    sector_results: dict[int, tuple] = {}

    for q, entries in grouped.items():
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= idx.multiplicity(charge_val)
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        for rk in right_subkeys:
            size = 1
            for leg_pos, charge_val in zip(right_axes, rk):
                idx = tensor.indices[leg_pos]
                size *= idx.multiplicity(charge_val)
            right_col_sizes.append(size)

        total_rows = sum(left_row_sizes)
        total_cols = sum(right_col_sizes)
        if total_rows == 0 or total_cols == 0:
            continue

        # Assemble the per-sector block matrix (traceable)
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            if is_fermionic:
                full_key = [0] * len(tensor.indices)
                for ax, ch in zip(left_axes, lk):
                    full_key[ax] = ch
                for ax, ch in zip(right_axes, rk):
                    full_key[ax] = ch
                parities = tuple(
                    int(sym.parity(np.array([full_key[i]]))[0])
                    for i in range(len(full_key))
                )
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        available_q = min(total_rows, total_cols)
        sector_results[q] = (
            matrix,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
            available_q,
        )

    # --- Static per-sector keep allocation ---
    if max_singular_values is None:
        k_per_sector: dict[int, int] = {
            q: r[5] for q, r in sector_results.items()
        }
    elif base_charges is not None:
        target_charges = _derive_charges(base_charges, max_singular_values)
        target_count: dict[int, int] = {}
        for tq in target_charges:
            target_count[int(tq)] = target_count.get(int(tq), 0) + 1
        k_per_sector = {
            q: min(target_count.get(q, 0), r[5])
            for q, r in sector_results.items()
        }
    else:
        # Defensive fallback: proportional to per-sector available capacity.
        total_avail = sum(r[5] for r in sector_results.values()) or 1
        k_per_sector = {
            q: max(1, round(max_singular_values * r[5] / total_avail))
            for q, r in sector_results.items()
        }
        # Trim if rounding overshoots
        excess = sum(k_per_sector.values()) - max_singular_values
        for q in sorted(k_per_sector.keys()):
            if excess <= 0:
                break
            take = min(excess, k_per_sector[q] - 1)
            if take > 0:
                k_per_sector[q] -= take
                excess -= take

    # Floor at >=1 total (mirrors eager n_keep = max(1, n_keep) at line 1255)
    total_keep = sum(k_per_sector.values())
    if total_keep == 0 and sector_results:
        best_q = max(
            sector_results.keys(),
            key=lambda q: (sector_results[q][5], -q),
        )
        k_per_sector[best_q] = 1

    # --- Per-sector AD-primitive SVD ---
    sector_svd: dict[int, tuple[jax.Array, jax.Array, jax.Array]] = {}
    for q, (matrix, _, _, _, _, avail) in sector_results.items():
        k_q = k_per_sector.get(q, 0)
        if k_q <= 0:
            continue
        # truncated_svd_ad takes a dense matrix and returns U (m, k), s (k,), Vh (k, n)
        U_q, s_q, Vh_q = truncated_svd_ad(matrix, k_q)
        sector_svd[q] = (U_q, s_q, Vh_q)

    # --- Concatenate output in canonical sector-ascending order ---
    ordered_qs = sorted(sector_svd.keys())
    bond_charges = np.repeat(
        np.array(ordered_qs, dtype=np.int32),
        np.array([sector_svd[q][1].shape[0] for q in ordered_qs], dtype=np.int32),
    )
    s_final = jnp.concatenate([sector_svd[q][1] for q in ordered_qs])

    if normalize and s_final.shape[0] > 0:
        s_final = s_final / jnp.sum(s_final)

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    # --- Reconstruct U / Vh block dicts ---
    # U has indices: (left_indices..., bond_index_out)
    # Vh has indices: (bond_index_in, right_indices...)
    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}
    for q in ordered_qs:
        matrix, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes, _ = sector_results[q]
        U_q, _, Vh_q = sector_svd[q]
        # Split U_q's rows back into per-left-subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            block_rows = U_q[row_offset : row_offset + n_rows, :]
            row_offset += n_rows
            # Reshape (n_rows, k_q) back to per-leg multiplicities
            shape = tuple(
                tensor.indices[ax].multiplicity(ch)
                for ax, ch in zip(left_axes, lk)
            ) + (U_q.shape[1],)
            U_blocks[lk + (q,)] = block_rows.reshape(shape)
        # Split Vh_q's columns back into per-right-subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            block_cols = Vh_q[:, col_offset : col_offset + n_cols]
            col_offset += n_cols
            shape = (Vh_q.shape[0],) + tuple(
                tensor.indices[ax].multiplicity(ch)
                for ax, ch in zip(right_axes, rk)
            )
            Vh_blocks[(q,) + rk] = block_cols.reshape(shape)

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices
    U_T = SymmetricTensor._from_blocks_unchecked(U_blocks, U_indices)
    Vh_T = SymmetricTensor._from_blocks_unchecked(Vh_blocks, Vh_indices)

    # Under tracing, no separate pre-truncation spectrum is tracked.
    return U_T, s_final, Vh_T, s_final
```

- [ ] **Step 6: Run the test, expect pass**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_truncated_svd_symmetric_traced_preserves_charges \
  tests/test_ctm_2x2_projector_symmetric.py::test_gauge_fix_symmetric_svd_tracer_safe \
  -xvs
```

Expected: both pass. The eager tests in the same file must also still pass — run the file end-to-end:

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py -xvs
```

- [ ] **Step 7: Commit**

```bash
git add src/tenax/linalg.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "$(cat <<'EOF'
feat(linalg): tracer-safe per-sector symmetric SVD (#435)

Adds _truncated_svd_symmetric_traced — block-sparse SVD that composes
truncated_svd_ad (Francuz et al. Lorentzian) per sector. Allocates
per-sector keep counts statically via _derive_charges(base_charges, k)
or proportional-to-sector-dim fallback. Bond axis emerges in
sector-block order (not global SV order); permutation is safe because
tensor contractions match by charge identity per block.

linalg.svd gains a base_charges kwarg that forwards through to the
symmetric path. Eager (non-traced) symmetric SVD is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Thread `base_charges` through `_ctm_tensor_moves.py` 2plaq path

**Why:** The 2plaq projector calls in `_ctm_tensor_moves.py` currently pass no `base_charges`, so under tracing the symmetric path would hit the proportional fallback. Mirror PR #437's pattern for the 1x1 path: derive `base_charges = _get_base_charges(a)` once at the multisite-sweep entry and forward.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py` (signatures of `_compute_2x2_projector_2plaq` and the four multisite move helpers; multisite-sweep entry derivation)

- [ ] **Step 1: Locate the multisite-sweep entry point**

```bash
grep -n "def _ctm_tensor_sweep_multisite\|_compute_2x2_projector_2plaq\|_get_base_charges" /home/yjkao/tenax/.worktrees/issue-435-tracer-safe-2x2-design/src/tenax/algorithms/_ctm_tensor_moves.py | head -20
```

Identify the parent multisite sweep that calls each `_compute_2x2_projector_2plaq` chain. (Likely lives in `_ctm_tensor_convergence.py` or a multisite-specific moves file — confirm before editing.)

- [ ] **Step 2: Add `base_charges` kwarg to `_compute_2x2_projector_2plaq`**

In `src/tenax/algorithms/_ctm_tensor_moves.py`, find `_compute_2x2_projector_2plaq` (around line 343). Update signature:

```python
def _compute_2x2_projector_2plaq(
    env_TL,
    env_TR,
    env_BL,
    env_BR,
    a_TL,
    a_TR,
    a_BL,
    a_BR,
    chi: int,
    direction: str,
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
```

Forward to the `_compute_2x2_projector` call (line 355-357):

```python
    P_top_raw, P_bot_raw = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi,
        direction=direction,
        base_charges=base_charges,
    )
```

- [ ] **Step 3: Update the four multisite-move helpers that call `_compute_2x2_projector_2plaq`**

Locate calls at lines 925, 1029, 1117, 1203 (and also at the older direct-projector sites if they exist). Each helper's signature gets a new `base_charges` kwarg; each call to `_compute_2x2_projector_2plaq` forwards it.

Example for the helper at line ~925:

```python
def _ctm_tensor_move_left_2plaq(
    ...,
    base_charges: np.ndarray | None = None,
) -> ...:
    ...
    P_top, P_bot = _compute_2x2_projector_2plaq(
        env_TL=env_TL, env_TR=env_TR, env_BL=env_BL, env_BR=env_BR,
        a_TL=a_TL, a_TR=a_TR, a_BL=a_BL, a_BR=a_BR,
        chi=chi, direction="left",
        base_charges=base_charges,
    )
```

Repeat for right/top/bottom variants.

- [ ] **Step 4: Derive `base_charges` once at the multisite-sweep entry**

Find the multisite-sweep entry (likely `_ctm_tensor_sweep_multisite` in `_ctm_tensor_convergence.py`; mirror PR #437's pattern at `_ctm_tensor_sweep`). Add near the start:

```python
    from tenax.algorithms._ctm_tensor_moves import _get_base_charges
    # Derive base_charges from the (representative) ket tensor so the 2plaq
    # path's symmetric SVD can do per-sector allocation under AD tracing
    # (Issue #435; mirrors PR #437 for the 1x1 path).
    base_charges = _get_base_charges(a) if a is not None else None
```

Then thread `base_charges=base_charges` to each directional-move helper.

- [ ] **Step 5: Write a smoke test that confirms `base_charges` is forwarded**

Append to `tests/test_ctm_2x2_projector_symmetric.py`:

```python
def test_compute_2x2_projector_2plaq_forwards_base_charges(monkeypatch):
    """`_compute_2x2_projector_2plaq` forwards base_charges to `_compute_2x2_projector`."""
    from tenax.algorithms import _ctm_tensor_moves
    seen: dict[str, object] = {}
    real_fn = _ctm_tensor_moves._compute_2x2_projector

    def spy(*args, **kwargs):
        seen["base_charges"] = kwargs.get("base_charges")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(_ctm_tensor_moves, "_compute_2x2_projector", spy)

    # Use the existing symmetric_corners fixture by importing it
    # ... (build minimal corners + ket; call _compute_2x2_projector_2plaq
    #     with base_charges=np.array([0, 1], dtype=np.int32) and chi=4)

    expected = np.array([0, 1], dtype=np.int32)
    # ... after the call ...
    assert seen["base_charges"] is not None
    np.testing.assert_array_equal(seen["base_charges"], expected)
```

Fill in the corner/ket construction by reusing the existing `_make_symmetric_enlarged_corner` and `symmetric_corners` fixture patterns (line 83+ of the same test file).

- [ ] **Step 6: Run the smoke test**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py::test_compute_2x2_projector_2plaq_forwards_base_charges -xvs
```

Expected: passes after Steps 2-4. Also verify nothing else broke:

```bash
uv run pytest -m core tests/test_ctm_2x2_projector_symmetric.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_moves.py src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_ctm_2x2_projector_symmetric.py
git commit -m "$(cat <<'EOF'
feat(ctm): thread base_charges through 2plaq projector path (#435)

Mirrors PR #437's plumbing for the 1x1 path. _compute_2x2_projector_2plaq
and the four multisite directional-move helpers gain a base_charges
kwarg; the multisite-sweep entry derives it via _get_base_charges(a)
and threads through. Under AD tracing, this lets the symmetric SVD
allocate per-sector keep counts from base_charges instead of the
proportional-to-sector-dim defensive fallback.

For DenseTensor inputs _get_base_charges returns None — no behavior
change on the dense path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update `_compute_2x2_projector_symmetric` — fix `S_Mp[0]` and route base_charges to traced SVD

**Why:** With the bond axis emerging in sector-block order under tracing, `S_Mp[0]` is the largest SV of the *first sector*, not the global max. Use `jnp.max(S_Mp)` instead. Also forward `base_charges` from `_compute_2x2_projector_symmetric` into the M_prime `tensor_svd` call so the traced path receives it.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (lines 935-953 area, line 960)

- [ ] **Step 1: Replace `S_Mp[0]` at line 960**

In `_compute_2x2_projector_symmetric` (line 803-1016), find line 960:

```python
    s_max = S_Mp[0]
```

Replace with:

```python
    s_max = jnp.max(S_Mp)
```

- [ ] **Step 2: Forward `base_charges` into the M_prime SVD call**

Currently lines 934-953 branch on `base_charges is None`. Under tracing we want the traced path to handle the allocation. Simplify:

```python
    # Pass base_charges through; the linalg.svd dispatcher will route to the
    # traced path if blocks carry tracers, or the eager+retruncate path
    # otherwise.
    if base_charges is None:
        U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
            M_prime_T,
            left_labels=mp_left_labels,
            right_labels=mp_right_labels,
            new_bond_label="chi_new",
            max_singular_values=chi,
        )
    else:
        # Eager path: full-spectrum SVD then per-sector re-truncation honoring
        # base_charges (mirrors _retruncate_by_base_charges).  Under tracing,
        # the dispatcher in _truncated_svd_symmetric routes to the traced
        # variant which consumes base_charges directly.
        is_traced_inputs = any(
            isinstance(b, jax.core.Tracer)
            for q in (Q_TL, Q_TR, Q_BL, Q_BR)
            if isinstance(q, SymmetricTensor)
            for b in q.blocks.values()
        )
        if is_traced_inputs:
            U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
                M_prime_T,
                left_labels=mp_left_labels,
                right_labels=mp_right_labels,
                new_bond_label="chi_new",
                max_singular_values=chi,
                base_charges=base_charges,
            )
        else:
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

The tracer-detection here is on the inputs to `_compute_2x2_projector_symmetric`, not on `M_prime_T`'s blocks (which are derived). The `tensor_svd` dispatcher will also re-check; the early branch here just selects the right code path.

- [ ] **Step 3: Drop the `_has_tracer` filter at `_compute_2x2_projector` lines 411-426**

Replace lines 409-426:

```python
    # Dispatch: SymmetricTensor inputs (without JAX tracers in any block) go
    # through the block-sparse path (Issue #416).  Tracer-bearing symmetric
    # blocks (AD backward) and pure DenseTensor inputs fall through to the
    # dense pipeline below.  Densifying tracer-bearing symmetric inputs is
    # safe because the AD backward only runs the projector once per GMRES
    # matvec; eager-mode symmetric inputs (forward CTM) take the block-sparse
    # path for performance and charge-structure preservation.
    if inputs_are_symmetric:

        def _has_tracer(t: Tensor) -> bool:
            if isinstance(t, SymmetricTensor):
                return any(isinstance(b, jax.core.Tracer) for b in t.blocks.values())
            return isinstance(getattr(t, "_data", None), jax.core.Tracer)

        if not any(_has_tracer(q) for q in (Q_TL, Q_TR, Q_BL, Q_BR)):
            return _compute_2x2_projector_symmetric(
                Q_TL,
                Q_TR,
                Q_BL,
                Q_BR,
                chi,
                direction=direction,
                base_charges=base_charges,
            )
        # Tracer-bearing symmetric path falls through to densify-and-dense-pipeline below.
```

with the simpler:

```python
    # Dispatch: any SymmetricTensor input routes to the block-sparse path,
    # whether tracer-bearing (AD backward) or eager (forward CTM).  The
    # symmetric pipeline is tracer-safe end-to-end (Issue #435).  The dense
    # fallback below is reached only for all-DenseTensor inputs.
    if inputs_are_symmetric:
        return _compute_2x2_projector_symmetric(
            Q_TL,
            Q_TR,
            Q_BL,
            Q_BR,
            chi,
            direction=direction,
            base_charges=base_charges,
        )
```

The dense fallback (lines 428-704) survives unchanged — it's the path for all-DenseTensor inputs.

- [ ] **Step 4: Run the eager-path regression tests + the tracer-safety tests from Tasks 1-2**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py -xvs
```

Expected: all pass. If the eager `_retruncate_by_base_charges` path's previously-passing tests now fail, inspect — the dispatch logic should NOT change eager behavior.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py
git commit -m "$(cat <<'EOF'
feat(ctm): symmetric 2x2 projector goes live under tracing (#435)

* Drop the _has_tracer filter at _compute_2x2_projector — tracer-bearing
  symmetric inputs now route to _compute_2x2_projector_symmetric (which
  is tracer-safe end-to-end after the linalg + gauge-fix edits).
* Forward base_charges into the M_prime tensor_svd call; under tracing
  this lets _truncated_svd_symmetric_traced do per-sector allocation.
* Replace S_Mp[0] with jnp.max(S_Mp) so positional reads work under
  the sector-block bond ordering of the traced SVD output.

Dense fallback survives unchanged for all-DenseTensor inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Un-xfail the 4 acceptance tests

**Why:** PR #437 marked these xfail with `Issue #435 ...` as reason. After Tasks 1-4 land, the tracer error is gone and the tests should pass.

**Files:**
- Modify: `tests/test_ipeps.py` (drop 1 `@pytest.mark.xfail`)
- Modify: `tests/test_fpeps_ad.py` (drop 3 `@pytest.mark.xfail`)

- [ ] **Step 1: Locate the xfail decorators**

```bash
cd /home/yjkao/tenax/.worktrees/issue-435-tracer-safe-2x2-design
grep -n "Issue #435\|#435 " tests/test_ipeps.py tests/test_fpeps_ad.py 2>&1
```

Confirm 4 hits across the two files (decorators on the tests named in the spec).

- [ ] **Step 2: Run the 4 tests with the xfail still in place to confirm they're now xpass**

```bash
uv run pytest -xvs \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type \
  tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite \
  tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init \
  tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases
```

Expected: all four report `XPASS` (or `PASSED` if `strict=False`). If any still fails, investigate — there's a missing fix somewhere in Tasks 1-4.

- [ ] **Step 3: Remove the 4 xfail decorators**

For each of the four tests, remove the `@pytest.mark.xfail(strict=False, reason="Issue #435 ...")` decorator line. Use `Edit` with the exact decorator text plus one line of surrounding context to ensure unique replacement.

- [ ] **Step 4: Re-run the 4 tests, expect plain PASS**

```bash
uv run pytest -xvs \
  tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type \
  tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite \
  tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init \
  tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases
```

Expected: all four `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ipeps.py tests/test_fpeps_ad.py
git commit -m "$(cat <<'EOF'
test(ipeps): un-xfail 4 non-trivial-U(1) AD tests (closes #435)

Drop @pytest.mark.xfail decorators pinned to Issue #435. The tracer-safe
symmetric 2x2 projector (Tasks 1-4) makes these tests pass on their own:

- test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type
- test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite
- test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init
- test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add 5 new explicit acceptance tests

**Why:** The 4 un-xfailed tests are end-to-end AD-iPEPS optimizations — slow, broad coverage. Add fast unit tests that pin specific properties of the new tracer-safe path so future regressions surface quickly.

**Files:**
- Modify: `tests/test_ctm_2x2_projector_symmetric.py` (append 5 tests)

- [ ] **Step 1: Test 1 — tracer-safety end-to-end**

```python
def test_compute_2x2_projector_tracer_safe_under_grad(symmetric_corners):
    """`_compute_2x2_projector` runs inside jax.grad on non-trivial U(1) inputs."""
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        # Inject a tracer by scaling Q_TL's blocks
        scaled_blocks = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q_TL_traced = SymmetricTensor._from_blocks_unchecked(scaled_blocks, Q_TL.indices)
        P_top, _ = _compute_2x2_projector(
            Q_TL_traced, Q_TR, Q_BL, Q_BR,
            chi=4, direction="left", base_charges=base_charges,
        )
        return jnp.sum(jnp.abs(jnp.concatenate([b.flatten() for b in P_top.blocks.values()])))

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(g), f"AD through symmetric 2x2 projector must produce finite grad, got {g}"
```

- [ ] **Step 2: Test 2 — sector preservation**

```python
def test_compute_2x2_projector_preserves_non_trivial_charges(symmetric_corners):
    """P_top's chi_new_top leg carries non-trivial charges (not the trivial-zero wrap)."""
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR,
        chi=4, direction="left", base_charges=base_charges,
    )
    chi_new_charges = np.asarray(P_top.indices[-1].charges)
    assert not np.all(chi_new_charges == 0), (
        f"chi_new_top must carry non-trivial U(1) charges from base_charges, "
        f"got all-zeros: {chi_new_charges}"
    )
```

- [ ] **Step 3: Test 3 — eager vs traced numerical equivalence (trivial-charge)**

```python
def test_compute_2x2_projector_eager_vs_traced_trivial(symmetric_corners):
    """For trivial-charge symmetric inputs, eager and traced paths produce the same
    projector up to bond permutation (verified via P_top @ P_top.conj().T)."""
    # Build trivial-charge corners by zeroing all input charges
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners

    def to_trivial(Q):
        new_indices = tuple(
            TensorIndex.from_charges(
                idx.symmetry,
                np.zeros_like(np.asarray(idx.charges)),
                idx.flow,
                label=idx.label,
            )
            for idx in Q.indices
        )
        return SymmetricTensor._from_blocks_unchecked(Q.blocks, new_indices)

    Qs_trivial = [to_trivial(Q) for Q in (Q_TL, Q_TR, Q_BL, Q_BR)]

    # Eager path
    P_top_eager, _ = _compute_2x2_projector(*Qs_trivial, chi=4, direction="left")

    # Traced path (inject a no-op tracer via jax.lax.stop_gradient on a constant alpha)
    def get_p_top(alpha):
        scaled = {k: alpha * b for k, b in Qs_trivial[0].blocks.items()}
        Q0 = SymmetricTensor._from_blocks_unchecked(scaled, Qs_trivial[0].indices)
        P_top, _ = _compute_2x2_projector(
            Q0, *Qs_trivial[1:], chi=4, direction="left",
            base_charges=np.zeros(4, dtype=np.int32),
        )
        return jnp.asarray(P_top.todense())

    P_top_traced = jax.jit(get_p_top)(jnp.asarray(1.0))
    P_top_eager_dense = jnp.asarray(P_top_eager.todense())

    # Permutation-invariant comparison: P @ P.conj().T on the bond axis
    chi_outer = P_top_eager_dense.shape[0]
    fused_D2 = P_top_eager_dense.shape[1]
    P_eager_mat = P_top_eager_dense.reshape(chi_outer * fused_D2, -1)
    P_traced_mat = P_top_traced.reshape(chi_outer * fused_D2, -1)
    inner_eager = P_eager_mat @ P_eager_mat.conj().T
    inner_traced = P_traced_mat @ P_traced_mat.conj().T
    np.testing.assert_allclose(
        np.asarray(inner_eager), np.asarray(inner_traced),
        atol=1e-8, rtol=1e-6,
        err_msg="trivial-charge: P @ P.conj().T must match between eager and traced",
    )
```

- [ ] **Step 4: Test 4 — gradient finite-difference cross-check**

```python
def test_compute_2x2_projector_grad_matches_finite_difference(symmetric_corners):
    """jax.grad through `_compute_2x2_projector` matches central-difference."""
    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    def loss(alpha: jax.Array) -> jax.Array:
        scaled_blocks = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q_TL_traced = SymmetricTensor._from_blocks_unchecked(scaled_blocks, Q_TL.indices)
        P_top, _ = _compute_2x2_projector(
            Q_TL_traced, Q_TR, Q_BL, Q_BR,
            chi=4, direction="left", base_charges=base_charges,
        )
        return jnp.sum(jnp.real(jnp.asarray(P_top.todense())) ** 2)

    alpha0 = jnp.asarray(1.0)
    g_ad = jax.grad(loss)(alpha0)

    eps = 1e-4
    g_fd = (loss(alpha0 + eps) - loss(alpha0 - eps)) / (2 * eps)

    rel_err = jnp.abs(g_ad - g_fd) / (jnp.abs(g_fd) + 1e-10)
    assert rel_err < 1e-3, f"AD vs FD relative error too large: {rel_err} (g_ad={g_ad}, g_fd={g_fd})"
```

- [ ] **Step 5: Test 5 — closure under tracing**

```python
def test_compute_2x2_projector_closure_under_tracing(symmetric_corners):
    """P_bot · P_top = I on (chi_new_bot, chi_new_top) under tracing."""
    from tenax.contraction.contractor import contract

    Q_TL, Q_TR, Q_BL, Q_BR = symmetric_corners
    base_charges = np.array([0, 1], dtype=np.int32)

    @jax.jit
    def get_closure(alpha: jax.Array) -> jax.Array:
        scaled = {k: alpha * b for k, b in Q_TL.blocks.items()}
        Q0 = SymmetricTensor._from_blocks_unchecked(scaled, Q_TL.indices)
        P_top, P_bot = _compute_2x2_projector(
            Q0, Q_TR, Q_BL, Q_BR,
            chi=4, direction="left", base_charges=base_charges,
        )
        closure = contract(P_bot, P_top)
        return jnp.asarray(closure.todense())

    closure = get_closure(jnp.asarray(1.0))
    chi_new = closure.shape[0]
    np.testing.assert_allclose(
        np.asarray(closure), np.eye(chi_new),
        atol=1e-6,
        err_msg="P_bot · P_top must be identity on chi_new × chi_new (Fishman closure)",
    )
```

- [ ] **Step 6: Run all five new tests**

```bash
uv run pytest tests/test_ctm_2x2_projector_symmetric.py \
  -k "tracer_safe_under_grad or preserves_non_trivial_charges or eager_vs_traced_trivial or grad_matches_finite_difference or closure_under_tracing" \
  -xvs
```

Expected: all five `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_ctm_2x2_projector_symmetric.py
git commit -m "$(cat <<'EOF'
test(ctm): explicit acceptance tests for tracer-safe 2x2 projector (#435)

Five new tests pin specific properties of the symmetric tracer-safe path:

1. tracer-safety under jax.grad with non-trivial U(1)
2. chi_new_top charges are non-trivial after the projector wrap
3. eager vs traced numerical equivalence on trivial-charge inputs
   (via permutation-invariant P @ P.conj().T comparison)
4. AD gradient matches central finite-difference to 1e-3 relative
5. Fishman closure P_bot · P_top = I under jax.jit

Complements the 4 un-xfailed end-to-end AD-iPEPS optimization tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full CI verification and PR

- [ ] **Step 1: Run `pytest -m core` end-to-end**

```bash
cd /home/yjkao/tenax/.worktrees/issue-435-tracer-safe-2x2-design
uv run pytest -m core -q
```

Expected: all tests pass. If any regression appears in an unrelated test, debug before opening the PR.

- [ ] **Step 2: Run `pytest -m "not slow"` to catch algorithm-marked regressions locally**

```bash
uv run pytest -m "not slow" -q --timeout 600
```

Expected: passes. Slow tests are deferred to CI's `run-full-tests` label.

- [ ] **Step 3: Push and open PR with `run-full-tests` label**

```bash
git push -u origin docs/issue-435-tracer-safe-2x2-design
gh pr create \
  --title "fix(ctm): tracer-safe symmetric 2x2 projector (closes #435)" \
  --label "run-full-tests" \
  --body "$(cat <<'EOF'
## Summary

- Rewrite `_gauge_fix_symmetric_svd` in pure JAX (removes `int(jnp.argmax(...))` / `complex(col_flat[idx])` casts).
- Add `_truncated_svd_symmetric_traced` in `linalg.py` — per-sector AD-primitive SVD with static keep allocation via `_derive_charges(base_charges, chi)`.
- Thread `base_charges` through the 2plaq path in `_ctm_tensor_moves.py` (mirrors PR #437 for the 1x1 path).
- Drop the `_has_tracer` filter at `_compute_2x2_projector`; replace `S_Mp[0]` with `jnp.max(S_Mp)` to handle the sector-block bond ordering of the traced SVD.
- Un-xfail 4 non-trivial-U(1) AD tests; add 5 explicit acceptance tests.

Closes #435.

## Test plan

- [ ] `tests/test_ctm_2x2_projector_symmetric.py` passes locally (eager regression + 5 new tracer tests)
- [ ] The 4 un-xfailed tests pass:
  - `test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`
  - `test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite`
  - `test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init`
  - `test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases`
- [ ] Existing 5 `#416`-unxfailed tests (trivial-charge symmetric) stay green
- [ ] Existing eager-mode 2x2 projector tests stay green (global-sort path untouched)
- [ ] CI passes with `run-full-tests` label

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Queue auto-merge**

```bash
gh pr merge --squash --delete-branch --auto
```

Auto-merge fires once branch protection's required checks pass.

---

## Self-Review (writing-plans checklist)

**Spec coverage:** every spec section maps to a task —
- Approach decided (3 edits) → Tasks 1-4 (Task 1 = Component 2, Task 2 = Component 1, Task 3 = Component 3, Task 4 = dispatch + S_Mp fix)
- SVD backward strategy (truncated_svd_ad per block) → Task 2 Step 5 (imports it inside the new function)
- Bond-axis ordering concession → Task 4 Step 1 (S_Mp[0] fix)
- Truncation rule + drop-leftover → Task 2 Step 5 (the `k_per_sector` allocation block)
- Error-handling cases → Task 2 Step 5 (defensive guards in the new function)
- Testing primary acceptance → Task 5; new explicit tests → Task 6

**Placeholder scan:** searched for "TBD", "TODO", "..." in steps; the only "..." is in Task 3 Step 5's test sketch (the corner-construction reuses an existing fixture and is reasonable to leave as "fill in via existing pattern"). All other steps have complete code.

**Type consistency:** `_truncated_svd_symmetric_traced` signature in Task 2 Step 5 matches the call in Task 2 Step 4. `base_charges` kwarg name is consistent across all modules. `_compute_2x2_projector_2plaq` signature in Task 3 Step 2 matches the calls in Task 3 Step 3.

**One known partial:** Task 3 Step 5's test body sketches the corner construction but defers to the existing `symmetric_corners` fixture. That fixture lives in the same file (line 110); the engineer will read it and complete the test.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-12-issue-435-tracer-safe-2x2-projector.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
