# Vectorize `_gauge_fix_symmetric_svd` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-bond-column Python loop in `_gauge_fix_symmetric_svd` with a per-bond-charge-sector batched computation, cutting the χ-scaling op emission (~25% gauge-fix slice of the #566 CTM-AD compile wall) while producing **bit-identical** output.

**Architecture:** Behavior-preserving optimization. The current loop runs over ≈χ bond columns, each emitting `argmax`/`concatenate`/per-block `.at[col].multiply`. The rewrite groups columns by their (static) bond charge — ~2 sectors for Z₂ — stacks each sector's U-blocks into one `(R, n_q)` matrix, computes all `n_q` column phases in one vectorized `argmax`, and applies them with one broadcast-multiply per block. Concatenation order is preserved so `argmax` (incl. ties) is identical. Guarded by a frozen-reference parity test (this is a refactor: the parity test is the safety net, and goes red if the vectorization is wrong).

**Tech Stack:** JAX (jnp, jax.grad), NumPy (static charge arrays), tenax `SymmetricTensor` / `tenax.linalg.svd`, pytest.

---

## File Structure

- **Modify:** `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` — rewrite `_gauge_fix_symmetric_svd` (lines 53–145). Same name, signature, and output semantics; internal only.
- **Modify (tests):** `tests/test_ctm_2x2_projector_symmetric.py` — add a frozen reference copy of the current loop plus parity/grad/edge-case tests. Reuses the existing `_make_test_matrix_tensor` helper.

No other files change. No call sites change (the function is internal to the 2×2 projector). Always-on; no gate.

---

### Task 1: Lock current behavior with a frozen-reference parity test

**Files:**
- Modify: `tests/test_ctm_2x2_projector_symmetric.py`

- [ ] **Step 1: Add the frozen reference + parity/grad/edge tests**

Append to `tests/test_ctm_2x2_projector_symmetric.py` (the file already imports `jax`, `jnp`, `np`, `pytest`, `_gauge_fix_symmetric_svd`, `TensorIndex`, `FlowDirection`, `U1Symmetry`, `SymmetricTensor`):

```python
# --------------------------------------------------------------------------- #
# Vectorization parity (#566): the per-sector rewrite of _gauge_fix_symmetric_svd
# must be byte-identical to the original per-column loop, frozen here.
# --------------------------------------------------------------------------- #
def _reference_gauge_fix_loop(U_T, Vh_T):
    """Frozen copy of the original per-column _gauge_fix_symmetric_svd loop.

    Kept in the test as the behavioral oracle for the vectorized rewrite.
    """
    bond_idx = U_T.indices[-1]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    local_index_of: dict[int, dict[int, int]] = {}
    counter: dict[int, int] = {}
    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local_index_of.setdefault(q_int, {})[j] = counter.get(q_int, 0)
        counter[q_int] = counter.get(q_int, 0) + 1

    u_blocks_by_q: dict[int, list] = {}
    for key, block in U_T.blocks.items():
        u_blocks_by_q.setdefault(int(key[-1]), []).append((key, block))
    vh_blocks_by_q: dict[int, list] = {}
    for key, block in Vh_T.blocks.items():
        vh_blocks_by_q.setdefault(int(key[0]), []).append((key, block))

    new_u_blocks = {key: block for key, block in U_T.blocks.items()}
    new_vh_blocks = {key: block for key, block in Vh_T.blocks.items()}

    sample_block = next(iter(U_T.blocks.values()))
    is_complex = jnp.issubdtype(sample_block.dtype, jnp.complexfloating)

    for j, q in enumerate(bond_charges):
        q_int = int(q)
        local = local_index_of[q_int][j]
        u_entries = u_blocks_by_q.get(q_int, [])
        vh_entries = vh_blocks_by_q.get(q_int, [])
        if not u_entries:
            continue
        candidates = jnp.concatenate(
            [jnp.reshape(new_u_blocks[key][..., local], (-1,)) for key, _ in u_entries]
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
            conj_phase = jnp.real(phase)
            bare_phase = jnp.real(phase)
        for key, _block in u_entries:
            new_u_blocks[key] = new_u_blocks[key].at[..., local].multiply(conj_phase)
        for key, _block in vh_entries:
            new_vh_blocks[key] = new_vh_blocks[key].at[local, ...].multiply(bare_phase)

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out


def _svd_of(M_T):
    from tenax.linalg import svd as tensor_svd

    U_T, s, Vh_T, _ = tensor_svd(
        M_T, left_labels=("left",), right_labels=("right",), new_bond_label="bond"
    )
    return U_T, s, Vh_T


def _complex_matrix_tensor(seed: int = 7) -> SymmetricTensor:
    """Two-sector U(1) complex128 matrix tensor."""
    sym = U1Symmetry()
    charges = np.array([0, 0, 1, 1], dtype=np.int32)
    left_idx = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="left")
    right_idx = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="right")
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    re = SymmetricTensor.random_normal((left_idx, right_idx), k1)
    im = SymmetricTensor.random_normal((left_idx, right_idx), k2)
    blocks = {
        key: (re.blocks[key] + 1j * im.blocks[key]).astype(jnp.complex128)
        for key in re.blocks
    }
    return SymmetricTensor._from_blocks_unchecked(blocks, re.indices)


def _degenerate_matrix_tensor(seed: int = 3) -> SymmetricTensor:
    """Scaled-identity-per-sector matrix → fully degenerate singular values
    (exercises argmax ties, where concat order must match the reference)."""
    M_T = _make_test_matrix_tensor(seed=seed)
    blocks = {}
    for key, block in M_T.blocks.items():
        n = block.shape[0]
        blocks[key] = jnp.eye(n, block.shape[1], dtype=block.dtype) * (1.0 + key[0])
    return SymmetricTensor._from_blocks_unchecked(blocks, M_T.indices)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _make_test_matrix_tensor(seed=0),
        lambda: _make_test_matrix_tensor(seed=5),
        _complex_matrix_tensor,
        _degenerate_matrix_tensor,
    ],
    ids=["u1_real_a", "u1_real_b", "u1_complex", "degenerate_ties"],
)
def test_gauge_fix_vectorized_matches_reference_forward(factory):
    """Vectorized gauge fix is byte-identical to the frozen per-column loop."""
    M_T = factory()
    U_T, _s, Vh_T = _svd_of(M_T)

    U_ref, Vh_ref = _reference_gauge_fix_loop(U_T, Vh_T)
    U_new, Vh_new = _gauge_fix_symmetric_svd(U_T, Vh_T)

    for key in U_ref.blocks:
        assert jnp.array_equal(U_new.blocks[key], U_ref.blocks[key]), f"U block {key}"
    for key in Vh_ref.blocks:
        assert jnp.array_equal(Vh_new.blocks[key], Vh_ref.blocks[key]), f"Vh block {key}"


def test_gauge_fix_vectorized_matches_reference_grad():
    """Gradient through the vectorized gauge fix matches the reference (fp tier)."""
    M_T = _make_test_matrix_tensor(seed=2)
    U_T, _s, Vh_T = _svd_of(M_T)
    bond_label = "bond"

    def _loss(U_T_in, Vh_T_in, gauge_fn):
        U_fixed, Vh_fixed = gauge_fn(U_T_in, Vh_T_in)
        # Gauge-sensitive scalar over both factors.
        u = U_fixed.todense()
        v = Vh_fixed.todense()
        return jnp.real(jnp.sum(u * jnp.conj(u))) + jnp.real(jnp.sum(v))

    # Differentiate w.r.t. the U block buffers (the AD-relevant inputs).
    leaves, treedef = jax.tree.flatten(U_T)

    def _from_leaves(ls, fn):
        U_in = jax.tree.unflatten(treedef, ls)
        return _loss(U_in, Vh_T, fn)

    g_ref = jax.grad(lambda ls: _from_leaves(ls, _reference_gauge_fix_loop))(leaves)
    g_new = jax.grad(lambda ls: _from_leaves(ls, _gauge_fix_symmetric_svd))(leaves)
    for a, b in zip(jax.tree.leaves(g_new), jax.tree.leaves(g_ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run the new tests against the CURRENT implementation**

Run: `uv run pytest tests/test_ctm_2x2_projector_symmetric.py -q`
Expected: PASS — the current `_gauge_fix_symmetric_svd` equals the frozen reference (establishes the oracle is faithful and the harness is correct, before any change).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_2x2_projector_symmetric.py
git commit -m "test(#566): frozen-reference parity tests for _gauge_fix_symmetric_svd"
```

---

### Task 2: Rewrite `_gauge_fix_symmetric_svd` to per-sector batched

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:53-145`

- [ ] **Step 1: Replace the function body**

Replace the entire `_gauge_fix_symmetric_svd` function (lines 53–145) with:

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
    ``P_bot · P_top = I``.

    Vectorized over bond-charge sectors (#566): instead of looping over every
    bond column, we process each (static) bond charge once — stacking that
    sector's U-blocks into one matrix, computing all column phases in a single
    ``argmax``, and applying them with one broadcast-multiply per block.  The
    block concatenation order matches the column-order oracle, so ``argmax``
    (including ties) and the output are byte-identical to the per-column loop.
    """
    bond_idx = U_T.indices[-1]  # last leg of U is the SVD bond
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    # Group U-blocks by bond charge (last key entry) and Vh-blocks by bond
    # charge (first key entry), preserving block-dict order so the stacked
    # argmax matches the per-column reference's concatenation order.
    u_keys_by_q: dict[int, list] = {}
    for key in U_T.blocks:
        u_keys_by_q.setdefault(int(key[-1]), []).append(key)
    vh_keys_by_q: dict[int, list] = {}
    for key in Vh_T.blocks:
        vh_keys_by_q.setdefault(int(key[0]), []).append(key)

    new_u_blocks: dict = dict(U_T.blocks)
    new_vh_blocks: dict = dict(Vh_T.blocks)

    # Detect dtype statically so we don't promote real blocks to complex.
    sample_block = next(iter(U_T.blocks.values()))
    is_complex = jnp.issubdtype(sample_block.dtype, jnp.complexfloating)

    for q in np.unique(bond_charges):
        q_int = int(q)
        u_keys = u_keys_by_q.get(q_int, [])
        if not u_keys:
            continue

        # All U-blocks of this sector share the bond multiplicity n_q (last axis).
        n_q = new_u_blocks[u_keys[0]].shape[-1]

        # Stack each U-block's (rows_i, n_q) view → (R, n_q), same order as the
        # per-column reference's `candidates` concatenation.
        M_q = jnp.concatenate(
            [jnp.reshape(new_u_blocks[key], (-1, n_q)) for key in u_keys], axis=0
        )
        idx = jnp.argmax(jnp.abs(M_q), axis=0)  # (n_q,)
        best = M_q[idx, jnp.arange(n_q)]  # (n_q,)
        abs_best = jnp.abs(best)
        phase = jnp.where(
            abs_best > 0,
            best / jnp.maximum(abs_best, jnp.asarray(1e-30, dtype=abs_best.dtype)),
            jnp.ones_like(best),
        )
        if is_complex:
            conj_phase = jnp.conj(phase)
            bare_phase = phase
        else:
            conj_phase = jnp.real(phase)
            bare_phase = jnp.real(phase)

        # U columns (last axis) by conj(phase); Vh rows (first axis) by phase.
        for key in u_keys:
            new_u_blocks[key] = new_u_blocks[key] * conj_phase
        for key in vh_keys_by_q.get(q_int, []):
            blk = new_vh_blocks[key]
            bcast = jnp.reshape(bare_phase, (n_q,) + (1,) * (blk.ndim - 1))
            new_vh_blocks[key] = blk * bcast

    U_out = SymmetricTensor._from_blocks_unchecked(new_u_blocks, U_T.indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_vh_blocks, Vh_T.indices)
    return U_out, Vh_out
```

- [ ] **Step 2: Run the parity tests (the meaningful red/green checkpoint)**

Run: `uv run pytest tests/test_ctm_2x2_projector_symmetric.py -q`
Expected: PASS — vectorized output is byte-identical to the frozen reference, and grad matches. (A wrong concat order or broadcast would fail `test_gauge_fix_vectorized_matches_reference_forward`.)

- [ ] **Step 3: Run the broader symmetric-CTM AD suite for no regressions**

Run: `uv run pytest tests/test_ctm_2x2_projector_symmetric.py tests/stacked/ tests/test_block_sparse_ctm_ad.py -q`
Expected: PASS (no behavior change downstream).

- [ ] **Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py
git commit -m "perf(#566): vectorize _gauge_fix_symmetric_svd per bond-charge sector"
```

---

### Task 3: Measure the compile reduction and open the PR

**Files:**
- None modified (measurement + PR only).

- [ ] **Step 1: Measure HLO/compile before vs after (GPU rig)**

On the A100 (per `cutensornet-200-env` setup), capture the after-numbers and compare to the committed baseline `docs/superpowers/handoffs/570_results/profile_570_d4_batchoff.json` (D=4 OFF: χ=8/12/16 → env_hlo_instr 326547/400940/471073):

Run:
```bash
CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda,cpu uv run python -u \
  examples/profile_570_sweepvjp_compile.py --D 4 --chi-list 8 12 16 \
  --methods svd --full --reps 1 --json /tmp/gaugefix_after.json
```
Expected: `env_hlo_instr` and `total_compile_s` **lower** than the baseline, with the gap **growing with χ** (more bond columns collapsed). Record the deltas.

- [ ] **Step 2: Run the core suite once locally**

Run: `uv run pytest -m core -q`
Expected: PASS.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin perf/gauge-fix-symmetric-svd-vectorize-566
gh pr create --base main --title "perf(#566): vectorize _gauge_fix_symmetric_svd (per-column → per-sector)" \
  --body "Hoists the gauge-fix loop from ≈χ bond columns to ~n_sectors bond-charge sectors (batched argmax/phase + broadcast multiply). Byte-identical output (frozen-reference parity + grad tests); existing 2×2 + AD suites green. Compile measurement: <paste χ=8/12/16 before→after env_hlo_instr + total_compile_s>. Always-on, no interface change. Issue #566 (contained sub-lever from the #570 compile-wall localization)."
```

- [ ] **Step 4: Enable auto-merge**

```bash
gh pr merge <PR#> --auto
```

---

## Self-Review

**Spec coverage:**
- Per-sector vectorization of `_gauge_fix_symmetric_svd` → Task 2. ✓
- Bit-identical + grad parity bar → Task 1 tests (`array_equal` forward, `allclose` grad). ✓
- Existing invariants (reconstruction, real-positive-max-row, 2×2 closure) preserved → Task 2 Step 3 runs `test_ctm_2x2_projector_symmetric.py` (which holds those) + AD suites. ✓
- Edge cases (real/complex128, degenerate ties, single/empty sector) → Task 1 parametrization (`u1_complex`, `degenerate_ties`; empty-sector handled by the `if not u_keys: continue`). ✓
- Always-on, no gate, no interface change → Task 2 keeps name/signature; no call-site edits. ✓
- Measurement on the rig → Task 3. ✓

**Placeholder scan:** PR body has one intentional `<paste …>` for measured numbers (filled at Task 3 Step 1) and `<PR#>` (from Step 3 output) — both resolved during execution, not plan gaps. No TODO/TBD in code.

**Type/name consistency:** `_gauge_fix_symmetric_svd(U_T, Vh_T) -> (U_out, Vh_out)` unchanged across tasks; `_reference_gauge_fix_loop` used only in tests; helper names (`_svd_of`, `_complex_matrix_tensor`, `_degenerate_matrix_tensor`) defined in Task 1 and used only there; `SymmetricTensor._from_blocks_unchecked` and `_make_test_matrix_tensor` match existing code.
