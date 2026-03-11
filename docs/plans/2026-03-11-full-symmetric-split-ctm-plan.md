# Full Symmetric Split CTM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all todense() calls from split CTM sweeps, energy, and convergence so SymmetricTensor iPEPS runs fully block-sparse.

**Architecture:** Replace dense projector computation, interlayer contractions, edge projection, and wrapping functions with Tensor-protocol equivalents (contract, fuse_indices, linalg.svd, _compute_projector_tensor). Apply per-layer projectors sequentially to the grown edge instead of building a combined P_full. Charges are self-consistent by construction — no wrap functions needed.

**Tech Stack:** JAX, tenax.core.tensor (DenseTensor/SymmetricTensor), tenax.contraction.contractor (contract), tenax.algorithms._tensor_utils (fuse_indices), tenax.algorithms._ctm_projector (_compute_projector_tensor), tenax.linalg (svd)

**Design doc:** `docs/plans/2026-03-11-full-symmetric-split-ctm.md`

---

### Task 1: Make _grow_edge_no_double_layer return Tensor

The grown edge is already a Tensor at line 467 of `_split_ctm_tensor.py` — it just todenses and reshapes before returning. Change it to return the Tensor directly.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py:429-469`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Modify _grow_edge_no_double_layer to return Tensor**

Change the return type from `jax.Array` to `Tensor`. Remove the `todense().reshape()` at line 469. Return `grown` directly.

```python
def _grow_edge_no_double_layer(
    T_ket: Tensor,
    T_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    contracted_leg: str,
    ket_I_label: str,
    bra_I_label: str,
    output_labels: tuple[str, ...],
) -> Tensor:
    """Grow a T-edge by contracting ket/bra layers separately.

    Returns an 8-leg Tensor with labels matching output_labels.
    """
    ket_D_label = f"{contracted_leg}_ket"
    bra_D_label = f"{contracted_leg}_bra"

    A_ket = A.relabel(contracted_leg, ket_D_label)
    ket_half = contract(T_ket, A_ket)

    bra_mapping: dict[str, str] = {contracted_leg: bra_D_label}
    for v in _VIRTUAL_LEGS:
        if v != contracted_leg:
            bra_mapping[v] = v.upper()
    A_bra = A_bar.relabels(bra_mapping)
    bra_half = contract(T_bra, A_bra)

    ket_half = ket_half.relabel(ket_I_label, "_I")
    bra_half = bra_half.relabel(bra_I_label, "_I")
    return contract(ket_half, bra_half, output_labels=output_labels)
```

**Step 2: Update test_grow_edge_matches_double_layer**

The test at line 261 currently compares the grown edge as a dense array. Update to call `grown.todense().reshape(...)` on the Tensor return value.

**Step 3: Run test**

```bash
uv run pytest tests/test_split_ctm_tensor.py::TestSplitCTMTensorEnergy::test_grow_edge_matches_double_layer -xvs
```

Expected: PASS (identical behavior, just moved the todense to caller).

**Step 4: Commit**

```bash
git commit -m "refactor: _grow_edge_no_double_layer returns Tensor instead of dense array"
```

---

### Task 2: Rewrite _split_ctm_move_left in Tensor protocol

This is the core task. Rewrite the left move to use Tensor protocol throughout.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py:477-610` (left move)
- Depends on: `_compute_projector_tensor` from `_ctm_projector.py`

**Step 1: Replace ket projector computation (steps 1-4)**

Current code (lines 488-509):
```python
C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)
C1g_ket_dense = C1g_ket_fused.todense()
# ... same for C4g
P_ket = _compute_projector_dense(C1g_ket_dense, C4g_ket_dense, chi)
C1_mid_dense = P_ket.conj().T @ C1g_ket_dense
```

Replace with:
```python
C1g_ket_fused = fuse_indices(C1g_ket, 0, 1, "fused", FlowDirection.IN)
C4g_ket_fused = fuse_indices(C4g_ket, 0, 1, "fused", FlowDirection.IN)

P_ket = _compute_projector_tensor(C1g_ket_fused, C4g_ket_fused, chi)
# P_ket: (fused, chi_new) with flows (IN, OUT)

# Mid-corners: P_ket† @ C1g_ket = contract(P_ket.bar(), C1g_ket_fused)
# P_ket.bar(): (fused, chi_new) with flows (OUT, IN) — contracts "fused" with C1g
C1_mid = contract(P_ket.bar(), C1g_ket_fused)  # (chi_new, t1k_I_col)
C4_mid = contract(P_ket.bar(), C4g_ket_fused)  # (chi_new, t3k_I_col)
```

Note: `C1g_ket_fused` has labels `("fused", "t1k_I")` and `P_ket.bar()` has labels `("fused", "chi_new")`. The `"fused"` labels match and contract automatically.

**Step 2: Replace bra growth and projection (steps 5-6)**

Current code (lines 515-525): dense einsum + projector.

Replace with:
```python
# C1_mid: (chi_new, t1k_I) — contract t1k_I with T1_bra's t1b_I
C1_mid_r = C1_mid.relabel("t1k_I", "t1b_I")  # match T1_bra interlayer label
C1g_bra = contract(C1_mid_r, env.T1_bra)  # (chi_new, u_bra, t1b_r)
C1g_bra_fused = fuse_indices(C1g_bra, 0, 1, "fused", FlowDirection.IN)

C4_mid_r = C4_mid.relabel("t3k_I", "t3b_I")
C4g_bra = contract(C4_mid_r, env.T3_bra)  # (chi_new, d_bra, t3b_l)
C4g_bra_fused = fuse_indices(C4g_bra, 0, 1, "fused", FlowDirection.IN)

P_bra = _compute_projector_tensor(C1g_bra_fused, C4g_bra_fused, chi)

# New corners
C1_new = contract(P_bra.bar(), C1g_bra_fused)  # (chi_new_bra, t1b_r)
C4_new = contract(P_bra.bar(), C4g_bra_fused)  # (chi_new_bra, t3b_l)
```

Relabel corner outputs to match environment conventions:
```python
C1_new = C1_new.relabels({"chi_new": "c1_d", "t1b_r": "c1_r"})
C4_new = C4_new.relabels({"chi_new": "c4_r", "t3b_l": "c4_u"})
```

Normalize:
```python
C1_new = max_abs_normalize(C1_new)
C4_new = max_abs_normalize(C4_new)
```

**Step 3: Replace edge projection (steps 7-10) with sequential projector application**

The grown edge from Task 1 is an 8-leg Tensor:
`T4g: (t4k_d, u, U, r, R, t4b_u, d, D)`

Apply P_ket and P_bra sequentially to left and right sides:

```python
T4g = _grow_edge_no_double_layer(
    env.T4_ket, env.T4_bra, A, A_bar, "l",
    "t4k_I", "t4b_I",
    ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"),
)

# --- Left side: fuse ket pair, project, fuse bra pair, project ---
# Fuse (t4k_d, u) → matches P_ket's fused index structure
T4g = fuse_indices(T4g, T4g.labels().index("t4k_d"),
                   T4g.labels().index("u"), "fused", FlowDirection.IN)
# Contract P_ket.bar() over "fused" → produces "chi_new" leg
T4g = contract(P_ket.bar(), T4g)
# Fuse (chi_new, U) → matches P_bra's fused index structure
T4g = fuse_indices(T4g, T4g.labels().index("chi_new"),
                   T4g.labels().index("U"), "fused", FlowDirection.IN)
T4g = contract(P_bra.bar(), T4g)
# Left side now has "chi_new" leg (chi-dimensional) from P_bra
T4g = T4g.relabel("chi_new", "left_chi")

# --- Right side: fuse ket pair, project, fuse bra pair, project ---
T4g = fuse_indices(T4g, T4g.labels().index("d"),
                   T4g.labels().index("t4b_u"), "fused", FlowDirection.IN)
T4g = contract(P_ket.bar(), T4g)
T4g = fuse_indices(T4g, T4g.labels().index("chi_new"),
                   T4g.labels().index("D"), "fused", FlowDirection.IN)
T4g = contract(P_bra.bar(), T4g)
T4g = T4g.relabel("chi_new", "right_chi")

# Fuse perpendicular virtual legs: (r, R) → D² bond
T4g = fuse_indices(T4g, T4g.labels().index("r"),
                   T4g.labels().index("R"), "D2", FlowDirection.IN)
# T4g now: (left_chi, D2, right_chi) — projected double-layer edge
```

**Step 4: SVD split into ket/bra**

Use block-sparse `linalg.svd` instead of `_svd_split_edge_dense`:

```python
from tenax.linalg import svd as tensor_svd

# SVD: split (left_chi, D2) vs (right_chi) → ket (left_chi, D2, bond) + bra (bond, right_chi)
# Actually we want: (left_chi, D_ket) | (D_bra, right_chi) with bond = chi_I
# Need to unfuse D2 first... but we don't have unfuse_indices.

# Alternative: don't fuse (r, R) into D2. Keep as 4-leg tensor:
# (left_chi, r, R, right_chi) and SVD split as
# left_labels=["left_chi", "r"], right_labels=["R", "right_chi"]
U_t, s, Vh_t, s_full = tensor_svd(
    T4g,  # (left_chi, r, R, right_chi) — DON'T fuse r,R
    left_labels=["left_chi", "r"],
    right_labels=["R", "right_chi"],
    new_bond_label="_svd_bond",
    max_singular_values=chi_I,
)
# U_t: (left_chi, r, _svd_bond) — this is T4_ket with chi=left_chi, D=r, chi_I=_svd_bond
# Absorb sqrt(s) into both sides
sqrt_s = jnp.sqrt(s)
# ... apply sqrt_s to U_t and Vh_t bond legs

# Relabel to match edge conventions
T4_ket_new = U_t.relabels({"left_chi": "t4k_d", "r": "l_ket", "_svd_bond": "t4k_I"})
T4_bra_new = Vh_t.relabels({"_svd_bond": "t4b_I", "R": "l_bra", "right_chi": "t4b_u"})
```

Note: The SVD split naturally produces ket (chi, D, chi_I) and bra (chi_I, D, chi) Tensors with correct charges from the block-sparse SVD. No wrapping needed.

**Step 5: Remove dead code**

Delete from this move: all `todense()` calls, `_compute_projector_dense` usage, `_wrap_corner_dense`/`_wrap_edge_*_dense` calls, and the `_sym`/`_c1_q`/`_c4_q` charge variable bookkeeping.

**Step 6: Run tests**

```bash
uv run pytest tests/test_split_ctm_tensor.py -x -q
```

Expected: All 15 pass + 1 xfail (the todense guard test should now xfail differently or pass depending on whether all 4 moves are converted).

**Step 7: Commit**

```bash
git commit -m "feat: rewrite left CTM move to use Tensor protocol (no todense)"
```

---

### Task 3: Rewrite remaining three moves (right, top, bottom)

Each follows the same pattern as the left move with different label mappings.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py` (right: ~620-730, top: ~740-850, bottom: ~860-980)

**Key differences per move:**

| Move   | Ket-first corners | Bra-first corners | Edge | Contracted leg |
|--------|-------------------|-------------------|------|----------------|
| Left   | C1+T1_ket, C4+T3_ket | C1+T1_bra, C4+T3_bra | T4 | "l" |
| Right  | C2+T1_bra, C3+T3_bra | C2+T1_ket, C3+T3_ket | T2 | "r" |
| Top    | C1+T4_ket, C2+T2_ket | C1+T4_bra, C2+T2_bra | T1 | "u" |
| Bottom | C4+T4_bra, C3+T2_bra | C4+T4_ket, C3+T2_ket | T3 | "d" |

Note: Right and Bottom moves start with bra layer first (P_bra computed before P_ket). The sequential projector application order reverses accordingly.

**Step 1: Convert right move** following left move pattern
**Step 2: Convert top move**
**Step 3: Convert bottom move**
**Step 4: Run full test suite**

```bash
uv run pytest tests/test_split_ctm_tensor.py -x -q
```

**Step 5: Commit**

```bash
git commit -m "feat: rewrite right/top/bottom CTM moves to Tensor protocol"
```

---

### Task 4: Convert convergence check to block-sparse SVD

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py:~1230-1240`

**Step 1: Replace dense SVD**

Current:
```python
current_sv = jnp.linalg.svd(env.C1.todense(), compute_uv=False)
```

Replace with:
```python
_, current_sv, _, _ = tensor_svd(
    env.C1, left_labels=[env.C1.labels()[0]],
    right_labels=[env.C1.labels()[1]], new_bond_label="_sv",
)
```

**Step 2: Run tests, commit**

```bash
git commit -m "feat: use block-sparse SVD for split CTM convergence check"
```

---

### Task 5: Rewrite energy computation

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py:~1250-1320`
- Depends on: `compute_energy_ctm_tensor` from `_ctm_tensor.py`

**Step 1: Write _split_env_to_tensor_standard**

Merge split edges into standard double-layer edges via Tensor protocol:

```python
def _split_env_to_tensor_standard(
    env: SplitCTMTensorEnv,
) -> "CTMTensorEnv":
    """Convert SplitCTMTensorEnv to CTMTensorEnv via Tensor protocol."""
    from tenax.algorithms._ctm_tensor import CTMTensorEnv

    def merge_edge(T_ket: Tensor, T_bra: Tensor, interlayer_ket: str,
                   interlayer_bra: str, D_ket: str, D_bra: str,
                   out_labels: tuple[str, str, str]) -> Tensor:
        """Merge ket/bra edge into double-layer edge."""
        T_ket_r = T_ket.relabel(interlayer_ket, "_I")
        T_bra_r = T_bra.relabel(interlayer_bra, "_I")
        merged = contract(T_ket_r, T_bra_r)  # contracts over _I
        # Fuse ket and bra virtual legs → D² bond
        merged = fuse_indices(
            merged, merged.labels().index(D_ket),
            merged.labels().index(D_bra), "D2", FlowDirection.IN,
        )
        # Relabel to standard conventions
        labels = merged.labels()
        label_map = {}
        for lbl in labels:
            if lbl == "D2":
                label_map[lbl] = out_labels[1]
            elif lbl != out_labels[1]:
                # Map chi labels to standard names
                ...  # exact mapping depends on edge direction
        return merged.relabels(label_map)

    return CTMTensorEnv(
        C1=env.C1,  # already (chi, chi) Tensor, just relabel
        C2=env.C2,
        C3=env.C3,
        C4=env.C4,
        T1=merge_edge(env.T1_ket, env.T1_bra, "t1k_I", "t1b_I", "u_ket", "u_bra", ...),
        T2=merge_edge(env.T2_ket, env.T2_bra, "t2k_I", "t2b_I", "r_ket", "r_bra", ...),
        T3=merge_edge(env.T3_ket, env.T3_bra, "t3k_I", "t3b_I", "d_ket", "d_bra", ...),
        T4=merge_edge(env.T4_ket, env.T4_bra, "t4k_I", "t4b_I", "l_ket", "l_bra", ...),
    )
```

**Step 2: Rewrite compute_energy_split_ctm_tensor**

```python
def compute_energy_split_ctm_tensor(
    A: Tensor, env: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array, d: int | None = None,
) -> jax.Array:
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
    std_env = _split_env_to_tensor_standard(env)
    return compute_energy_ctm_tensor(A, std_env, hamiltonian_gate, d)
```

**Step 3: Run energy tests**

```bash
uv run pytest tests/test_split_ctm_tensor.py::TestSplitCTMTensorEnergy -xvs
```

**Step 4: Commit**

```bash
git commit -m "feat: tensor-protocol energy computation for split CTM"
```

---

### Task 6: Delete dead code

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py`

**Step 1: Remove unused functions**

Delete:
- `_compute_projector_dense`
- `_svd_split_edge_dense`
- `_wrap_corner_dense`
- `_wrap_edge_ket_dense`
- `_wrap_edge_bra_dense`
- `_split_env_to_dense_standard`

Also remove now-unused imports: `_derive_charges`, `_trivial_symmetry` from `_ctm_utils`.

**Step 2: Run full test suite + linter**

```bash
uv run pytest tests/test_split_ctm_tensor.py tests/test_ctm_tensor.py tests/test_ipeps.py -x -q
uv run ruff check src/tenax/algorithms/_split_ctm_tensor.py
```

**Step 3: Commit**

```bash
git commit -m "refactor: remove dead dense code from split CTM"
```

---

### Task 7: Flip xfail guard test

**Files:**
- Modify: `tests/test_split_ctm_tensor.py`

**Step 1: Remove xfail marker from test_symmetric_sweep_no_todense**

Remove the `@pytest.mark.xfail(...)` decorator.

**Step 2: Run the guard test**

```bash
uv run pytest tests/test_split_ctm_tensor.py::TestSplitCTMSymmetric::test_symmetric_sweep_no_todense -xvs
```

Expected: PASS with 0 todense calls.

**Step 3: Run full test suite**

```bash
uv run pytest tests/test_split_ctm_tensor.py tests/test_ctm_tensor.py tests/test_ipeps.py -x -q
```

**Step 4: Commit**

```bash
git commit -m "test: flip todense guard test to passing (all 20 todense calls eliminated)"
```

---

### Task 8: Final verification and PR

**Step 1: Run full algorithm test suite**

```bash
uv run pytest -m "not slow" -x -q
```

**Step 2: Run linter**

```bash
uv run ruff check src/tenax/algorithms/_split_ctm_tensor.py src/tenax/algorithms/_ctm_projector.py
```

**Step 3: Create PR**

```bash
gh pr create --title "feat: full tensor-protocol split CTM (no todense in sweeps)" --body "..."
gh pr merge <number> --squash --delete-branch --auto
```
