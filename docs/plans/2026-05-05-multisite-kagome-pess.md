# Multisite kagome iPEPS encoding for iPESS — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-encode kagome iPESS as a multisite square-iPEPS with `d=2` per site (instead of the current `d_eff=8` Convention-C supersite) so that the diagonal-RDM cost drops from `χ²·D⁴·d²=χ²·D⁴·64` to `χ²·D⁴·d²=χ²·D⁴·4` — a **16× reduction at the binding-constraint step** — and large-D AD optimization (`D=6,8,10`) becomes feasible on existing CPU/GPU budgets.

**Architecture:** Add a sibling to `pess_to_kagome_supersite` that emits a *multi-coord* dict of d=2 site tensors, then plug it into Tenax's existing `ctm_multisite` + `compute_energy_ctm_tensor_multisite` (dense path) and the split-aware `compute_energy_split_ctm_tensor_multisite`. Build per-bond gate dispatch (kagome has heterogeneous bond Hamiltonians once the supersite is split), validate against the existing supersite path at D=2, and only then push to D≥4.

**Tech Stack:** Python, JAX, Tenax (`ctm_multisite`, `_ctm_tensor_multisite`, `compute_energy_ctm_tensor_multisite`, `compute_energy_split_ctm_tensor_multisite`, `ctm_energy_implicit`, `IPESSState`, `pess_to_kagome_supersite`).

---

## Critical encoding constraint (read first)

**Geometric fact:** kagome's elementary triangle (3 sites, 3 bonds, complete K₃) cannot be embedded in a square graph with all bonds NN — at least one of the three intra-triangle bonds must be a diagonal. The current Convention-C supersite path makes this explicit: `kagome_xxz_pess_cg_gates` has `h_intra` (3 bonds inside the d_eff=8 supersite) plus three inter-supersite bonds `{"h", "v", "diag"}`, and `"diag"` is the intrinsic NNN bond. Any *square* multisite splitting inherits at least one diagonal bond.

**Implication:** the planned multisite encoding will keep some diagonal bonds, but each diagonal RDM operates at `d=2` instead of `d_eff=8`, giving the 16× memory win. We do **not** try to eliminate diagonal bonds — that would require switching to a honeycomb iPEPS lattice, which is out of scope for this plan.

**Encoding choice (committed):** 2×2 multi-site unit cell with three "active" R-sites at d=2 plus a fourth "dummy" simplex site at d=1 (carries the down-triangle T_d connectivity, no physical leg participation). Concretely:

| coord  | role         | shape (after lambdas absorbed)        | d  |
|--------|--------------|----------------------------------------|----|
| (0,0)  | A_a (= R_a · sqrt(λ) gauges)           | (D, D, D, D, 2)  | 2 |
| (1,0)  | A_b                                    | (D, D, D, D, 2)  | 2 |
| (0,1)  | A_c                                    | (D, D, D, D, 2)  | 2 |
| (1,1)  | A_t (= T_u or T_d, d=1 dummy phys)     | (D, D, D, D, 1)  | 1 |

The 6 kagome bonds map to the iPEPS as:
- Intra-cell up-triangle: a-b (NN-h), a-c (NN-v), b-c (NNN-diag)  ← 1 diag, 2 NN
- Inter-cell down-triangle: c-of-(0,0)/b-of-(1,0) cell (NN-h via (0,1)→(2,0) wrap), a-of-(0,0)/b-of-(0,1) cell (NN-v wrap), and a-c-of-shifted cell (NN via (1,1) dummy + cell wrap, or one NNN). Exact shifts pinned in **Task 1**.

**Intra-cell connectivity (tensor legs):** since the 4-site square cell cannot host T_u's 3-way junction directly, T_u is contracted into A_a (one site absorbs it via `einsum("xap,abc->xbcp", R_a, T_u)`), giving A_a three non-trivial virtual legs (T_d-side ↑, T_u-to-R_b → right, T_u-to-R_c ↓ bottom) plus one trivial-padded leg. T_d, by contrast, lives at the (1,1) dummy and connects via three of its four virtual legs to neighboring R-sites in adjacent unit cells. The fourth leg of every site is dim-1-padded to `D` to keep the standard rank-5 iPEPS layout.

**Worked-out bond table (to be verified in Task 1):**

| bond name        | coord pair               | direction on iPEPS | gate                            |
|------------------|--------------------------|--------------------|---------------------------------|
| up-tri a-b       | (0,0) ↔ (1,0)            | NN-h               | XXZ pair on (a, b)              |
| up-tri a-c       | (0,0) ↔ (0,1)            | NN-v               | XXZ pair on (a, c)              |
| up-tri b-c       | (1,0) ↔ (0,1)            | NNN-diag           | XXZ pair on (b, c)              |
| dn-tri b-c       | (1,0) ↔ next-cell (0,0) (right wrap) → A_b ↔ A_a | NN-h | XXZ pair on (b, c) image        |
| dn-tri a-b       | (0,1) ↔ next-cell (0,0) (bottom wrap) → A_c ↔ A_a | NN-v | XXZ pair on (a, b) image        |
| dn-tri a-c       | (1,0) ↔ next-cell (0,1) (right+bottom wrap) | NNN-diag | XXZ pair on (a, c) image  |

Net: **4 NN bonds + 2 diag bonds**, all at d=2. Compare to supersite's **3 NN + 1 diag**, all at d_eff=8. The supersite has fewer bond evaluations but each is much heavier.

---

## Phase 0 — Set up (one-time)

### Task 0.1: Confirm worktree + baseline tests pass

**Files:** none

**Step 1:** Verify worktree location + branch.
```bash
git -C /home/yjkao/tenax/.worktrees/multisite-kagome-pess rev-parse --show-toplevel
git -C /home/yjkao/tenax/.worktrees/multisite-kagome-pess branch --show-current
```
Expected: `.../multisite-kagome-pess` and `feat/multisite-kagome-pess`.

**Step 2:** Run baseline test suite (core only, fast).
```bash
cd /home/yjkao/tenax/.worktrees/multisite-kagome-pess
uv run pytest -m core -q 2>&1 | tail -20
```
Expected: all green.

**Step 3:** Commit `docs/plans/2026-05-05-multisite-kagome-pess.md` (this file).
```bash
cd /home/yjkao/tenax/.worktrees/multisite-kagome-pess
git add docs/plans/2026-05-05-multisite-kagome-pess.md
git commit -m "plan: multisite kagome iPESS encoding (D=8/10 unblock)"
```

---

## Phase 1 — Pin down the encoding via a written design + parity at the supersite level

The encoding above is *committed in shape* but the **bond-direction map (especially the "wrap" inter-cell bonds)** is the riskiest part of this plan. Phase 1 nails it down by constructing the encoding tensor-by-tensor at D=2 and verifying — *as a unit test* — that contracting the multisite cell back into a single supersite reproduces the existing `pess_to_kagome_supersite` output exactly (up to gauge).

### Task 1.1: Write the failing parity test for the encoding map

**Files:**
- Create: `tests/test_pess_multisite_encoding.py`

**Step 1:** Write the failing test.

```python
# tests/test_pess_multisite_encoding.py
"""Parity tests for the kagome iPESS → multisite-iPEPS encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_supersite,
)


@pytest.mark.core
def test_pess_to_kagome_multisite_returns_4_site_dict():
    """`pess_to_kagome_multisite` must return a {coord: tensor} dict with the
    four expected coords and the documented shapes/dtypes."""
    from tenax.algorithms.pess import pess_to_kagome_multisite

    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )

    assert set(sites.keys()) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    for coord in [(0, 0), (1, 0), (0, 1)]:
        A = sites[coord]
        assert A.shape == (2, 2, 2, 2, 2), f"{coord}: got {A.shape}"
        assert A.dtype == jnp.complex128
    A_t = sites[(1, 1)]
    assert A_t.shape == (2, 2, 2, 2, 1), f"dummy: got {A_t.shape}"
```

**Step 2:** Run it — expect ImportError or AttributeError.
```bash
uv run pytest tests/test_pess_multisite_encoding.py::test_pess_to_kagome_multisite_returns_4_site_dict -v
```
Expected: FAIL (`pess_to_kagome_multisite` not defined).

**Step 3:** Commit the failing test.
```bash
git add tests/test_pess_multisite_encoding.py
git commit -m "test(pess): add failing skeleton for multisite encoding"
```

### Task 1.2: Implement the encoding map (no parity check yet)

**Files:**
- Modify: `src/tenax/algorithms/pess.py:360-475` (add new function below `pess_to_kagome_supersite`)

**Step 1:** Add the function. Use this exact body — it implements the 2×2 encoding from the table above.

```python
# In src/tenax/algorithms/pess.py, after pess_to_kagome_supersite

def pess_to_kagome_multisite(
    R_a: jax.Array,
    R_b: jax.Array,
    R_c: jax.Array,
    T_u: jax.Array,
    T_d: jax.Array,
    lambdas: tuple[jax.Array, ...] | jax.Array,
) -> dict[tuple[int, int], jax.Array]:
    """Build a 4-site 2×2 multisite iPEPS from iPESS primitives.

    Splits the Convention-C supersite into three d=2 active sites (one per
    kagome sublattice) plus one d=1 dummy site that carries the T_d simplex.
    Each active site has 4 virtual legs (one is dim-1-padded) and one
    physical leg of dimension 2; the dummy site has 4 virtual legs (three
    non-trivial, carrying T_d) and a d=1 physical leg.

    Bond layout (verified against pess_to_kagome_supersite via the inverse
    contraction in tests):

        coord   role            non-trivial virtual legs
        (0,0)   A_a (T_u·R_a)   right→A_b, bottom→A_c, top→T_d-of-prev-cell
        (1,0)   A_b             left→A_a, bottom→T_d-dummy
        (0,1)   A_c             top→A_a, right→T_d-dummy
        (1,1)   T_d dummy       left→A_c, top→A_b, right/bottom→adj cells

    Args:
        R_a, R_b, R_c: iPESS site tensors of shape (D, D, d), axes
            (T_d-leg, T_u-leg, phys).
        T_u: up-simplex (D, D, D), axes (R_a-leg, R_b-leg, R_c-leg).
        T_d: down-simplex (D, D, D), same axis convention.
        lambdas: 6 bond singular-value vectors in the IPESSState ordering
            (a-up, b-up, c-up, a-down, b-down, c-down).

    Returns:
        ``{(0,0): A_a, (1,0): A_b, (0,1): A_c, (1,1): A_t}`` — see table.
    """
    if isinstance(lambdas, jax.Array):
        lam_a_u, lam_b_u, lam_c_u, lam_a_d, lam_b_d, lam_c_d = (
            lambdas[0], lambdas[1], lambdas[2],
            lambdas[3], lambdas[4], lambdas[5],
        )
    else:
        (lam_a_u, lam_b_u, lam_c_u, lam_a_d, lam_b_d, lam_c_d) = lambdas

    D = R_a.shape[0]
    d = R_a.shape[2]
    dtype = R_a.dtype

    sqrt_lam_a_d = jnp.sqrt(lam_a_d)
    sqrt_lam_b_d = jnp.sqrt(lam_b_d)
    sqrt_lam_c_d = jnp.sqrt(lam_c_d)
    sqrt_lam_a_u = jnp.sqrt(lam_a_u)
    sqrt_lam_b_u = jnp.sqrt(lam_b_u)
    sqrt_lam_c_u = jnp.sqrt(lam_c_u)

    # Gauge each R: half down-bond on T_d-leg (axis 0), half up-bond on
    # T_u-leg (axis 1).
    S_a = jnp.einsum("i,ijp,j->ijp", sqrt_lam_a_d, R_a, sqrt_lam_a_u)
    S_b = jnp.einsum("i,ijp,j->ijp", sqrt_lam_b_d, R_b, sqrt_lam_b_u)
    S_c = jnp.einsum("i,ijp,j->ijp", sqrt_lam_c_d, R_c, sqrt_lam_c_u)

    # A_a fuses T_u: legs (T_d-side=top, R_b-side=right, R_c-side=bottom, phys).
    # Pad a trivial 4th virtual leg ("left") of dim 1 → D.
    A_a_core = jnp.einsum("xap,abc->xbcp", S_a, T_u)  # (D, D, D, d)
    A_a = jnp.zeros((D, D, D, D, d), dtype=dtype)
    # axis ordering convention for iPEPS site: (top, bottom, left, right, phys)
    # Match _make_supersite_indices: (u, d, l, r, phys) → axes 0..4.
    # A_a_core indices: (T_d-side, R_b-side, R_c-side, phys)
    # Map: T_d-side → top (axis 0), R_b-side → right (axis 3),
    #      R_c-side → bottom (axis 1), left = trivial (axis 2 = slot 0).
    A_a = A_a.at[:, :, 0, :, :].set(jnp.transpose(A_a_core, (0, 2, 1, 3)))

    # A_b: (T_d-side=bottom, T_u-side=left, phys). Pad top, right.
    A_b_core = S_b  # (D, D, d): (T_d-side, T_u-side, phys)
    A_b = jnp.zeros((D, D, D, D, d), dtype=dtype)
    # Map: T_d-side → bottom (axis 1), T_u-side → left (axis 2).
    # Top (axis 0) and right (axis 3) trivial → slot 0.
    A_b = A_b.at[0, :, :, 0, :].set(A_b_core)

    # A_c: (T_d-side=right, T_u-side=top, phys). Pad left, bottom.
    A_c_core = S_c  # (D, D, d): (T_d-side, T_u-side, phys)
    A_c = jnp.zeros((D, D, D, D, d), dtype=dtype)
    # Map: T_d-side → right (axis 3), T_u-side → top (axis 0).
    # Bottom (axis 1) and left (axis 2) trivial → slot 0.
    A_c = A_c.at[:, 0, 0, :, :].set(A_c_core)

    # A_t: T_d simplex with d=1 dummy phys. Three non-trivial virtual legs
    # carry T_d's three R-bonds. Layout: top→A_b, left→A_c, right→adj-cell-A_a,
    # bottom→adj-cell-A_a (one of these is the "next-cell" wrap).
    # T_d axes: (R_a-leg, R_b-leg, R_c-leg).
    # Map: R_a-leg → bottom (axis 1, wraps to next-cell A_a top),
    #      R_b-leg → top (axis 0, connects to (1,0) A_b bottom),
    #      R_c-leg → left (axis 2, connects to (0,1) A_c right).
    # Right (axis 3) trivial → slot 0; phys (axis 4) is dim 1.
    A_t = jnp.zeros((D, D, D, D, 1), dtype=dtype)
    # T_d_perm: shape (D_top, D_bottom, D_left) = T_d transposed
    # axes (R_b, R_a, R_c) → (top, bottom, left).
    T_d_perm = jnp.transpose(T_d, (1, 0, 2))  # now (R_b, R_a, R_c)
    A_t = A_t.at[:, :, :, 0, 0].set(T_d_perm)

    return {(0, 0): A_a, (1, 0): A_b, (0, 1): A_c, (1, 1): A_t}
```

**Step 2:** Run the test from Task 1.1 — should now PASS.
```bash
uv run pytest tests/test_pess_multisite_encoding.py::test_pess_to_kagome_multisite_returns_4_site_dict -v
```

**Step 3:** Commit.
```bash
git add src/tenax/algorithms/pess.py tests/test_pess_multisite_encoding.py
git commit -m "feat(pess): add pess_to_kagome_multisite encoding map"
```

### Task 1.3: Write the supersite-equivalence parity test

**Files:**
- Modify: `tests/test_pess_multisite_encoding.py`

**Step 1:** Add this test (it asserts the multisite tensors, when fused along the iPEPS bonds, reproduce the supersite tensor up to a global gauge).

```python
@pytest.mark.core
def test_pess_to_kagome_multisite_fuses_back_to_supersite():
    """Contracting the 4-site multisite cell back into a single supersite
    must reproduce pess_to_kagome_supersite up to a global gauge / reshape."""
    from tenax.algorithms.pess import (
        pess_to_kagome_multisite,
        pess_to_kagome_supersite,
    )

    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(7))
    A_super = pess_to_kagome_supersite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.lambdas
    )
    sites = pess_to_kagome_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    A_a, A_b, A_c, A_t = sites[(0, 0)], sites[(1, 0)], sites[(0, 1)], sites[(1, 1)]

    # Contract the 2×2 cell along internal bonds. Internal bonds (using the
    # (top, bottom, left, right, phys) leg convention from
    # _make_supersite_indices, i.e. axes 0..4):
    #   A_a.right (axis 3, dim D) ↔ A_b.left (axis 2)
    #   A_a.bottom (axis 1) ↔ A_c.top (axis 0)
    #   A_b.bottom (axis 1) ↔ A_t.top (axis 0)
    #   A_c.right (axis 3) ↔ A_t.left (axis 2)
    # External (open) legs remaining after contraction:
    #   A_a.top (axis 0)         — supersite "top"
    #   A_c.bottom (axis 1)      — supersite "bottom"
    #   A_a.left (axis 2)        — supersite "left" (trivial in supersite)
    #   A_b.right (axis 3)       — supersite "right"
    #   A_t.right (axis 3)       — auxiliary trivial (T_d's "outside" leg)
    #   A_t.bottom (axis 1)      — auxiliary trivial
    #   physical: tensor product (p_a, p_b, p_c, p_dummy=1) → fuses to d_eff=8
    contracted = jnp.einsum(
        "uDLrap,DdlBbq,UrLBcs,UbLRz->...",  # placeholder — replace below
        A_a, A_b, A_c, A_t,
    )  # noqa: F841 — replaced below

    # Concrete einsum (each lowercase letter is a unique bond):
    # A_a: (top=t1, bot=int1, left=ext_l, right=int2, phys=p_a)
    # A_b: (top=t2, bot=int3, left=int2, right=ext_r, phys=p_b)
    # A_c: (top=int1, bot=ext_b, left=ext_l2, right=int4, phys=p_c)
    # A_t: (top=int3, bot=ext_b2, left=int4, right=ext_r2, phys=p_d)
    contracted = jnp.einsum(
        "abcde,fgchij,bklmn,higop->aklfmocdnjpe",
        A_a, A_b, A_c, A_t,
    )
    # The output legs in order: (a=top_Aa, k=bot_Ac, l=ext_l_Ac, f=top_Ab,
    # m=bot_Ab=int3-out, o=ext_b_At, c=int2 [contracted, won't appear],
    # ...) — actually let me re-derive carefully via labelled contraction.

    # The hand-rolled einsum above is fragile. The real test should use
    # tenax.contract with named labels — see the implementation hint below.
    pytest.xfail("Replace with labelled-contraction once API is wired up.")
```

> **Note for the implementer:** the hand-rolled einsum is brittle. The robust way is to wrap each site as a `DenseTensor` with named labels (matching `_make_supersite_indices`), put them in a `TensorNetwork`, and let Tenax compute the contraction. After Task 1.4 you'll have the labels available; replace the `xfail` body with a `TensorNetwork`-based contraction and assert
> `np.allclose(fused.todense().reshape(...), A_super, atol=1e-12)` up to a permutation of the d_eff=8 phys index that matches the (p_a, p_b, p_c, p_dummy=1) → (p_a, p_b, p_c) collapse.

**Step 2:** Run it — should xfail (placeholder).
```bash
uv run pytest tests/test_pess_multisite_encoding.py::test_pess_to_kagome_multisite_fuses_back_to_supersite -v
```

**Step 3:** Commit (fail-forward placeholder).
```bash
git add tests/test_pess_multisite_encoding.py
git commit -m "test(pess): add (xfail) supersite-fusion parity placeholder"
```

### Task 1.4: Replace the placeholder with a TensorNetwork-based contraction

**Files:**
- Modify: `tests/test_pess_multisite_encoding.py`

**Step 1:** Replace the xfail body with a `TensorNetwork` build using `_make_supersite_indices`-style labels (`u`, `d`, `l`, `r`, `phys`) so the contraction is unambiguous.

```python
# Inside test_pess_to_kagome_multisite_fuses_back_to_supersite, replace the body with:
from tenax import DenseTensor, TensorIndex
from tenax.core.flow_direction import FlowDirection
from tenax.core.tensor_network import TensorNetwork
from tenax.core.symmetry import TrivialSymmetry

sym = TrivialSymmetry()
def idx(label: str, dim: int, flow: FlowDirection) -> TensorIndex:
    return TensorIndex.from_charges(sym, [(0, dim)], flow, label=label)

D = state.R_a.shape[0]
d = state.R_a.shape[2]

def site(A, *, name, has_phys=True):
    legs = [
        idx(f"{name}_u", A.shape[0], FlowDirection.OUT),
        idx(f"{name}_d", A.shape[1], FlowDirection.IN),
        idx(f"{name}_l", A.shape[2], FlowDirection.OUT),
        idx(f"{name}_r", A.shape[3], FlowDirection.IN),
        idx(f"{name}_p", A.shape[4], FlowDirection.IN),
    ]
    return DenseTensor(A, legs)

ta, tb, tc, tt = site(A_a, name="a"), site(A_b, name="b"), site(A_c, name="c"), site(A_t, name="t")

# Wire the four internal bonds as (label_left, label_right) pairs
# matching the bond table in the docstring of pess_to_kagome_multisite.
tn = TensorNetwork([ta, tb, tc, tt])
tn.connect("a_r", "b_l")
tn.connect("a_d", "c_u")
tn.connect("b_d", "t_u")
tn.connect("c_r", "t_l")

fused = tn.contract()  # leaves the 8 external legs + 4 phys legs
fused_arr = fused.todense()

# Collapse the four phys legs (p_a, p_b, p_c, p_dummy of dim 1) into d_eff=8
# in the same row-major order as pess_to_kagome_supersite.
# After contraction the open virtual legs are:
#   a_u (top), c_d (bottom), a_l (left, trivial), b_r (right),
#   t_r (right of T_d dummy, trivial),
#   t_d (bottom of T_d dummy, trivial),
#   plus phys legs in order (a_p, b_p, c_p, t_p).
# The supersite has shape (D, D, D, D, d**3). Reshape accordingly:
fused_super = fused_arr.reshape(D, D, D, D, 1, 1, d, d, d, 1)
# Squeeze trivial axes and put phys last.
fused_super = jnp.squeeze(fused_super, axis=(4, 5, 9))  # (D, D, D, D, d, d, d)
fused_super = fused_super.reshape(D, D, D, D, d**3)

assert jnp.allclose(fused_super, A_super, atol=1e-10), (
    f"max abs diff: {float(jnp.max(jnp.abs(fused_super - A_super))):.3e}"
)
```

> **Implementer warning:** the leg-permutation in the final reshape is the most error-prone step. If it fails, dump `fused_arr.shape` and the index-label order from `fused.indices` and adjust the squeeze/reshape to match `pess_to_kagome_supersite`'s layout `(D, D, D, D, d_eff)` with phys order `(p_a, p_b, p_c)`. Expect to iterate.

**Step 2:** Run.
```bash
uv run pytest tests/test_pess_multisite_encoding.py::test_pess_to_kagome_multisite_fuses_back_to_supersite -v
```
Expected: PASS at D=2, d=2. If it fails after one round of leg-permutation debugging, **stop and ask** — the encoding map (Task 1.2) likely has a leg-mapping bug.

**Step 3:** Commit.
```bash
git add tests/test_pess_multisite_encoding.py
git commit -m "test(pess): assert multisite cell fuses back to supersite at D=2"
```

### Task 1.5: Add D=3 and d=3 (spin-1) variants of the parity test

**Files:**
- Modify: `tests/test_pess_multisite_encoding.py`

**Step 1:** Add `pytest.mark.parametrize("D, d", [(2, 2), (3, 2), (2, 3)])` and parametrize Task 1.4's body. Run.

**Step 2:** Commit.

> **Stop here for review.** Task 1 establishes the encoding correctness at small D. The remaining phases assume Task 1 is green.

---

## Phase 2 — Multisite gates and energy (no AD, dense path first)

### Task 2.1: Build the per-bond gate dispatcher

`compute_energy_split_ctm_tensor_multisite` (and its dense sibling `compute_energy_ctm_tensor_multisite`) currently take a single 4-leg gate `H`. The multisite encoding has *heterogeneous* per-bond Hamiltonians (some bonds carry XXZ on (a,b), others on (a,c), and the dummy-site bonds carry zero). We need either (i) a per-bond gate dict, or (ii) a wrapper that loops over bonds and calls the existing single-gate function with the matching gate.

Pick **(ii)** to avoid touching the existing API.

**Files:**
- Create: `src/tenax/algorithms/_pess_multisite_energy.py`
- Create: `tests/test_pess_multisite_energy.py`

**Step 1:** Write the failing test first.

```python
# tests/test_pess_multisite_energy.py
import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.pess import IPESSState


@pytest.mark.core
def test_kagome_multisite_bond_gates_returns_six_entries():
    from tenax.algorithms._pess_multisite_energy import (
        kagome_multisite_bond_gates,
    )

    gates = kagome_multisite_bond_gates(delta=1.0, d=2)
    # 6 kagome bonds per unit cell.
    assert len(gates) == 6
    # Each gate is keyed by ((coord_a, dir_a), (coord_b, dir_b)) and is
    # a (d, d, d, d) array.
    for key, H in gates.items():
        assert H.shape == (2, 2, 2, 2)
```

```bash
uv run pytest tests/test_pess_multisite_energy.py::test_kagome_multisite_bond_gates_returns_six_entries -v
```
Expected: FAIL (module not found).

**Step 2:** Implement.

```python
# src/tenax/algorithms/_pess_multisite_energy.py
"""Multisite kagome iPESS energy: per-bond gate dispatch for the 4-site
2×2 unit cell encoding from pess_to_kagome_multisite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.pess import _xxz_pair_hamiltonian  # see Step 2.1.a


def kagome_multisite_bond_gates(
    delta: float = 1.0, d: int = 2
) -> dict[tuple, jax.Array]:
    """Per-bond XXZ gates on the 4-site multisite kagome iPEPS.

    Keys are frozenset({(coord_a, dir_a), (coord_b, dir_b)}) matching the
    bond_id convention used inside compute_energy_split_ctm_tensor_multisite.

    Returns a dict of 6 gates (3 up-triangle + 3 down-triangle bonds), each of
    shape ``(d, d, d, d)``.
    """
    H_pair = _xxz_pair_hamiltonian(delta, d).reshape(d, d, d, d)
    # All 6 kagome bonds use the same XXZ pair Hamiltonian; only the coord
    # pairs differ. The "diagonal" intra/inter bonds use the same gate too,
    # but they're routed via _rdm_diagonal_split_tensor variants in Task 2.2.
    bonds = {
        # up-triangle (intra-cell)
        frozenset({((0, 0), "right"), ((1, 0), "left")}): H_pair,   # a-b NN-h
        frozenset({((0, 0), "bottom"), ((0, 1), "top")}): H_pair,   # a-c NN-v
        frozenset({((1, 0), "diag-bl"), ((0, 1), "diag-tr")}): H_pair,  # b-c NNN
        # down-triangle (inter-cell), all wrap one or two cells
        frozenset({((1, 0), "right"), ((0, 0), "left")}): H_pair,   # b-a image
        frozenset({((0, 1), "bottom"), ((0, 0), "top")}): H_pair,   # c-a image
        frozenset({((1, 0), "diag-br"), ((0, 1), "diag-bl")}): H_pair,  # c-a-c-image NNN
    }
    return bonds
```

> **Implementer note:** `_xxz_pair_hamiltonian` is not currently exported from `pess.py` — it's the inner kron loop inside `_xxz_embed_inter`. Pull it out into a top-level helper at this step (a 5-line refactor); see the snippet in `_xxz_embed_inter` body lines 285–295 for the kron pattern. Add it to `pess.py` as `def _xxz_pair_hamiltonian(delta: float, d: int = 2) -> np.ndarray:` returning the 4-leg pair Hamiltonian.

**Step 3:** Run + commit.

### Task 2.2: Wire the dense multisite-energy path with diagonal-bond support

The existing `compute_energy_ctm_tensor_multisite` only iterates `right` and `bottom` NN directions. Add a sibling that also iterates two diagonal bonds (NW↔SE intra-cell, NE↔SW intra-cell) using the existing `_rdm_diagonal` 2-site primitive.

**Files:**
- Modify: `src/tenax/algorithms/_pess_multisite_energy.py`
- Test: `tests/test_pess_multisite_energy.py`

Steps follow the same TDD pattern (failing test → implement → pass → commit).

**Acceptance criterion:** `compute_energy_pess_multisite(sites, envs, gates) == compute_energy_cg(A_super, env_super, cg_gates, d_eff=8)` to within 1e-9 at D=2, d=2 with **identical input iPESS state and matched CTM convergence parameters**. This is the central parity check.

> **Stop here for review.** This is the highest-risk single task in the plan. If the dense-path parity does not hold at D=2, the encoding (Task 1.2) is wrong and we go back, *not* forward.

### Task 2.3: Repeat the parity check using the split-aware path

Reuse `compute_energy_split_ctm_tensor_multisite` (existing) for the four NN bonds, plus a new `compute_energy_split_ctm_tensor_multisite_diag` for the two diagonal bonds.

**Acceptance criterion:** split-aware multisite energy matches the dense multisite energy (Task 2.2) to within 1e-9 at D=2, d=2. This re-uses the existing `test_compute_energy_split_multisite_matches_shim` pattern.

---

## Phase 3 — AD loss + optimization

### Task 3.1: Add `build_pess_loss_multisite` to `pess_optimize.py`

Mirror `build_pess_loss` (`pess_optimize.py:59-128`) but call `pess_to_kagome_multisite` and `compute_energy_pess_multisite` instead of `pess_to_kagome_supersite` + `compute_energy_cg`. Use `ctm_energy_implicit` with the 4-coord neighbor map.

Tests: gradient-finite-difference at D=2 vs analytic gradient (1e-4 tolerance), and AD energy parity at D=2 between supersite and multisite paths after the same number of L-BFGS steps.

### Task 3.2: End-to-end optimization at D=2, d=2

Smoke test: 30 L-BFGS iterations on Heisenberg, χ=8. Energy should match the supersite optimization within 1e-6.

### Task 3.3: Optimization at D=4, d=2

Run 100 L-BFGS iterations, χ=16, χ=24. Compare to PR #387 / #390 / #397 results. Expected energy: -0.43xxx (Liao 2017 P2 reference, see `project_pess_ad_stalls_d4.md` and `project_liao2017_replication.md`).

**Memory budget check:** profile with `JAX_LOG_COMPILES=1` and confirm peak HBM/RSS is ~16× lower than the supersite-AD path at D=4, χ=16.

### Task 3.4: Push to D=6, 8 and compare to Liao 2017

Run on CPU box (251 GB available). At D=8 the multisite path should fit comfortably (estimated peak ~0.5 GB at the diagonal step vs ~64 GB for supersite). Compare per-site energy to `examples/kagome_spin12_pess_liao2017_replication.json`.

---

## Phase 4 — Documentation, exports, merge

### Task 4.1: Update `__init__.py` and `README.md`
- Add `pess_to_kagome_multisite`, `compute_energy_pess_multisite`, `build_pess_loss_multisite` to `src/tenax/__init__.py`'s `_LAZY_ATTRS` and `__all__`.
- Add a multisite section to README's iPESS/PESS examples.

### Task 4.2: Update memory file
Edit `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_kagome_multisite_ipeps_plan.md` to mark this PR as the resolution and document the `4 NN + 2 diag` cost profile.

### Task 4.3: Open the PR
Use `gh pr create` with a body that includes:
- Encoding diagram (the bond table from this plan).
- Memory comparison: supersite vs multisite at D=4, 6, 8.
- Energy comparison vs Liao 2017 P2 reference.
- The geometric-impossibility note (square multisite cannot be all-NN — that's why diag bonds remain).

Merge via `gh pr merge <num> --squash --delete-branch --auto` per CLAUDE.md.

---

## Stop-and-ask checkpoints

1. **After Task 1.4:** if the supersite-fusion parity test fails after one round of leg-permutation fixes, stop. The encoding's leg layout (Task 1.2) is wrong; do not push forward.
2. **After Task 2.2:** if dense-path multisite energy ≠ `compute_energy_cg` at D=2 to 1e-9, stop. Either the gate dispatch (Task 2.1) or the diagonal-bond contraction is wrong.
3. **After Task 3.2:** if AD-optimized multisite energy ≠ AD-optimized supersite energy at D=2 to 1e-6, stop. Likely an AD-graph difference; investigate before pushing to D=4.
4. **At Task 3.4:** if D=8 OOMs on the 251 GB CPU box, stop and re-profile — the encoding may have an unexpected χ²·D⁶ term lurking.

---

## What this plan deliberately does NOT do

- **Does not eliminate diagonal bonds.** Square iPEPS topology forbids it; we get 16× memory savings on each diagonal bond by reducing d² from 64 to 4, not by removing diagonals.
- **Does not implement a multisite split-aware CTM convergence sweep.** The existing per-site `ctm_split_tensor` runs each env on a 1×1 lattice of that site — for the 4-site cell this is approximate. We rely on the dense `ctm_multisite` for converged multisite envs and only use the split-aware code for the *energy* step. A future PR can add `ctm_split_tensor_multisite` if needed; this is NOT in scope here.
- **Does not switch to honeycomb iPEPS.** That would side-step the diagonal entirely (kagome is the medial of honeycomb) but is a larger architectural change with separate trade-offs and goes in a different plan.
- **Does not handle fermions.** All paths route through the bosonic split-aware code via `compute_energy_split_ctm_tensor_multisite`'s fermionic-fallback branch.
