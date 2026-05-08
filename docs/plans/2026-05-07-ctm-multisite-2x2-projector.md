# 2×2 plaquette CTM projector for multisite path — implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `_ctm_tensor_sweep_multisite`'s 1×1 grown-corner projector with the standard 2×2 enlarged-corner (Corboz–Penc–Mila–Lauchli, PRB 84, 041108(R) (2011)) projector, so the kagome 3-site multisite encoding converges to the same physical fixed point as variPEPS.

**Architecture:** New file `_ctm_tensor_projector_2x2.py` for `_build_enlarged_corner` and `_compute_2x2_projector`. New `_ctm_tensor_move_*_2x2` functions in `_ctm_tensor_moves.py` taking 4 sites + 4 envs. `_ctm_tensor_sweep_multisite` defaults to `recipe="2x2"`; the `recipe` kwarg is private and exists only so legacy 1×1 callers can pin during transition. DenseTensor only; SymmetricTensor is a follow-up.

**Tech Stack:** JAX, Tenax (`Tensor` protocol, `contract`, `fuse_indices`, existing `_truncated_SVD` and `_ctm_projector` internals).

**Background (read first):**
- `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md` (this PR's design doc).
- `src/tenax/algorithms/_ctm_tensor_moves.py:241-453` (existing 1×1 moves — pattern to mirror).
- `src/tenax/algorithms/_ctm_projector.py:758-` (`_compute_projector_tensor` and the Fishman dense fallback — `_truncated_SVD` is reused).
- variPEPS reference (read-only, GPL-3.0; do **not** copy code): `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/ctmrg/projectors.py:228-322` (`_fishman_horizontal_cut`) and `:569-632` (`_left_projectors_workhorse`).
- `tests/test_pess_3site_multisite_rdm_invariants.py` (existing green multisite witnesses — must stay green).
- Saved AD-optimum: `logs/d4_ad_optimum.npz` (D=4, χ=16, e_opt=-0.852811). Tier-1 contract test loads from here.

---

## Task 0: Setup verification

**Step 1: Confirm worktree is on `worktree-fix-multisite-ctm-rdm-helpers`.**

Run: `git rev-parse --abbrev-ref HEAD`
Expected: `worktree-fix-multisite-ctm-rdm-helpers`.

**Step 2: Confirm the saved AD-optimum is present.**

Run: `ls -lh logs/d4_ad_optimum.npz`
Expected: file present, ~few hundred KB.

If not present, the smoking-gun probe in `examples/dev/save_d4_ad_optimum.py` regenerates it (~12 min on GPU, ~30+ min on CPU). Don't run unless missing.

**Step 3: Confirm pre-commit hooks installed.**

Run: `uv run pre-commit install --install-hooks`
Expected: hooks installed (per `feedback_precommit.md` — always before any commit).

---

## Task 1: Reference snapshot — variPEPS numerics on a fixed random tensor

**Files:**
- Create: `tests/_ctm_2x2_reference_data.npz` (gitignored — generated locally on demand by Task 1's helper)
- Create: `examples/dev/gen_2x2_projector_reference.py`

**Why:** We can't copy variPEPS code (GPL-3.0). We can compare numerics on a fixed-seed random tensor to verify the new Tenax 2×2 projector matches variPEPS's Fishman output up to gauge.

**Step 1: Write the reference generator.**

```python
# examples/dev/gen_2x2_projector_reference.py
"""Generate variPEPS 2x2 projector reference on a fixed-seed random tensor.

NOT FOR COMMIT. Outputs tests/_ctm_2x2_reference_data.npz which is gitignored
and used by Task 11's tier-2 sanity check.

Run on GPU 1 with miniforge Python (variPEPS dep):
  CUDA_VISIBLE_DEVICES=1 /home/yjkao/miniforge3/bin/python \
      examples/dev/gen_2x2_projector_reference.py
"""
import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, jax.numpy as jnp, jax
from pathlib import Path
import varipeps
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.ctmrg import calc_ctmrg_env

def main():
    D, chi, d = 2, 8, 2
    rng = np.random.default_rng(42)
    A = rng.standard_normal((D, D, d, D, D)) + 1j * rng.standard_normal((D, D, d, D, D))
    A = jnp.asarray(A / np.linalg.norm(A))

    pt = PEPS_Tensor.from_tensor(A, d=d, D=(D,)*4, chi=chi, ctm_tensors_are_identities=True)
    uc = PEPS_Unit_Cell.from_tensor_list([pt], structure=((0,),))
    varipeps.varipeps_config.ctmrg_max_steps = 80
    varipeps.varipeps_config.ctmrg_convergence_eps = 1e-9
    arrs = [pt.tensor for pt in uc.get_unique_tensors()]
    res = calc_ctmrg_env(arrs, uc, eps=1e-9, enforce_elementwise_convergence=True)
    new_uc = res[0] if isinstance(res, tuple) else res
    pt_out = new_uc.get_unique_tensors()[0]

    out = Path("tests/_ctm_2x2_reference_data.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(
        out,
        A=np.asarray(A),
        C1=np.asarray(pt_out.C1), C2=np.asarray(pt_out.C2),
        C3=np.asarray(pt_out.C3), C4=np.asarray(pt_out.C4),
        T1=np.asarray(pt_out.T1), T2=np.asarray(pt_out.T2),
        T3=np.asarray(pt_out.T3), T4=np.asarray(pt_out.T4),
        D=D, chi=chi, d=d,
    )
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
```

**Step 2: Run it.**

```bash
CUDA_VISIBLE_DEVICES=1 /home/yjkao/miniforge3/bin/python \
    examples/dev/gen_2x2_projector_reference.py
```
Expected: `saved -> tests/_ctm_2x2_reference_data.npz` and CTM finishes in 5–20 s.

**Step 3: Add `.npz` reference to `.gitignore`.**

Append to `.gitignore`:
```
tests/_ctm_2x2_reference_data.npz
```

**Step 4: Commit.**

```bash
git add .gitignore examples/dev/gen_2x2_projector_reference.py
git commit -m "test: variPEPS reference generator for 2x2 projector cross-check"
```

---

## Task 2: New file `_ctm_tensor_projector_2x2.py` — `_build_enlarged_corner` for top-left only

**Files:**
- Create: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`
- Test: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the failing test for top-left enlarged corner.**

```python
# tests/test_ctm_tensor_projector_2x2.py
"""Tests for the 2x2 plaquette CTM projector (multisite path)."""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.core.tensor import DenseTensor
from tenax.core.tensor_index import TensorIndex
from tenax.core.flow import FlowDirection


def _dense_tensor_5leg(D: int, d: int, seed: int = 0) -> DenseTensor:
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((D, D, D, D, d))
           + 1j * rng.standard_normal((D, D, D, D, d)))
    arr = arr / np.linalg.norm(arr)
    indices = [
        TensorIndex(label="u", dim=D, flow=FlowDirection.IN),
        TensorIndex(label="d", dim=D, flow=FlowDirection.OUT),
        TensorIndex(label="l", dim=D, flow=FlowDirection.IN),
        TensorIndex(label="r", dim=D, flow=FlowDirection.OUT),
        TensorIndex(label="phys", dim=d, flow=FlowDirection.OUT),
    ]
    return DenseTensor(jnp.asarray(arr), indices)


def test_build_enlarged_corner_top_left_shape():
    """Q_TL = C1 · T1 · T4 · a should produce a rank-4 (chi, D2, chi, D2) tensor."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)              # (u2, d2, l2, r2)
    env = initialize_ctm_tensor_env(A, chi)        # identity-init env

    Q_TL = _build_enlarged_corner(
        env.C1, env.T1, env.T4, a, position="top_left"
    )

    # Output rank-4 with axes (chi_T1_right, D2_a_right, chi_T4_bottom, D2_a_bottom)
    # — the "right" + "bottom" are the seam legs that connect to Q_TR / Q_BL.
    assert Q_TL.rank() == 4
    shapes = {ind.label: ind.dim for ind in Q_TL.indices}
    assert shapes == {"chi_R": chi, "r2": D * D, "chi_B": chi, "d2": D * D}
```

**Step 2: Run to verify it fails.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_build_enlarged_corner_top_left_shape -v
```
Expected: FAIL with `ModuleNotFoundError: tenax.algorithms._ctm_tensor_projector_2x2`.

**Step 3: Implement `_build_enlarged_corner` for `position="top_left"`.**

```python
# src/tenax/algorithms/_ctm_tensor_projector_2x2.py
"""2x2 plaquette enlarged-corner builder for the multisite CTM projector.

Implements the standard CTMRG enlarged-corner construction (Corboz, Penc,
Mila, Lauchli, PRB 84, 041108(R) (2011)). For each plaquette quarter,
contracts one corner C, two adjacent edges T_h and T_v, and the double-
layer site tensor `a` into a rank-4 tensor with two seam legs (the chi
and D² legs that connect to the adjacent quarter in the 2x2).

Used by ``_ctm_tensor_move_*_2x2`` in ``_ctm_tensor_moves.py``.
"""
from __future__ import annotations

from tenax.contraction.contractor import contract
from tenax.core.tensor import Tensor

__all__ = ["_build_enlarged_corner"]


def _build_enlarged_corner(
    C: Tensor,
    T_h: Tensor,
    T_v: Tensor,
    a: Tensor,
    *,
    position: str,
) -> Tensor:
    """Enlarged corner Q = C · T_h · T_v · a for one plaquette quarter.

    For position='top_left':
      C  = C1  (labels: c1_d, c1_r)
      T_h = T1  (labels: t1_l, u2, t1_r)
      T_v = T4  (labels: t4_d, l2, t4_u)
      a  = double-layer site tensor (labels: u2, d2, l2, r2)

    Contraction:
      C1.c1_r  ↔ T1.t1_l    (top-left corner connects to T1 left)
      C1.c1_d  ↔ T4.t4_d    (top-left corner connects to T4 top)
      T1.u2    ↔ a.u2       (T1 absorbs a's top virtual)
      T4.l2    ↔ a.l2       (T4 absorbs a's left virtual)

    Output legs (free):
      t1_r  → relabel chi_R (right seam, connects to Q_TR)
      r2    → relabel r2    (right D² seam; original label kept)
      t4_u  → relabel chi_B (bottom seam, connects to Q_BL)
      d2    → relabel d2    (bottom D² seam; original label kept)
    """
    if position == "top_left":
        C_r = C.relabel("c1_r", "t1_l")
        CT_h = contract(C_r, T_h)                          # → (c1_d, u2, t1_r)
        T_v_r = T_v.relabel("t4_d", "c1_d")
        CTT = contract(CT_h, T_v_r)                        # → (u2, t1_r, l2, t4_u)
        Q = contract(CTT, a)                               # contracts u2, l2
        # Q free axes: (t1_r, t4_u, r2, d2). Relabel seams.
        Q = Q.relabels({"t1_r": "chi_R", "t4_u": "chi_B"})
        return Q
    raise NotImplementedError(f"position={position!r} not implemented yet")
```

**Step 4: Run the test.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_build_enlarged_corner_top_left_shape -v
```
Expected: PASS.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): _build_enlarged_corner for 2x2 plaquette top-left quarter"
```

---

## Task 3: `_build_enlarged_corner` for top-right, bottom-left, bottom-right

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the failing tests for the other 3 positions.**

Add to `tests/test_ctm_tensor_projector_2x2.py`:

```python
@pytest.mark.parametrize("position", ["top_right", "bottom_left", "bottom_right"])
def test_build_enlarged_corner_other_positions_shape(position):
    """Q_TR / Q_BL / Q_BR all rank-4 (chi, D2, chi, D2) with appropriate seam labels."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    if position == "top_right":
        Q = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position=position)
        expected_seams = {"chi_L": chi, "l2": D * D, "chi_B": chi, "d2": D * D}
    elif position == "bottom_left":
        Q = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position=position)
        expected_seams = {"chi_R": chi, "r2": D * D, "chi_T": chi, "u2": D * D}
    elif position == "bottom_right":
        Q = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position=position)
        expected_seams = {"chi_L": chi, "l2": D * D, "chi_T": chi, "u2": D * D}

    assert Q.rank() == 4
    shapes = {ind.label: ind.dim for ind in Q.indices}
    assert shapes == expected_seams
```

**Step 2: Run, verify the 3 fail.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_build_enlarged_corner_other_positions_shape -v
```
Expected: 3 FAILs (`NotImplementedError`).

**Step 3: Implement the 3 other positions.**

Replace the `if position == "top_left":` block with:

```python
    if position == "top_left":
        # C1: (c1_d, c1_r), T1: (t1_l, u2, t1_r), T4: (t4_d, l2, t4_u)
        C_r = C.relabel("c1_r", "t1_l")
        CT_h = contract(C_r, T_h)
        T_v_r = T_v.relabel("t4_d", "c1_d")
        CTT = contract(CT_h, T_v_r)
        Q = contract(CTT, a)
        return Q.relabels({"t1_r": "chi_R", "t4_u": "chi_B"})

    if position == "top_right":
        # C2: (c2_l, c2_d), T1: (t1_l, u2, t1_r), T2: (t2_u, r2, t2_d)
        C_r = C.relabel("c2_l", "t1_r")
        CT_h = contract(C_r, T_h)                          # → (c2_d, t1_l, u2)
        T_v_r = T_v.relabel("t2_u", "c2_d")
        CTT = contract(CT_h, T_v_r)                        # → (t1_l, u2, r2, t2_d)
        Q = contract(CTT, a)                               # contracts u2, r2
        # Q free: (t1_l, t2_d, l2, d2). Relabel seams.
        return Q.relabels({"t1_l": "chi_L", "t2_d": "chi_B"})

    if position == "bottom_left":
        # C4: (c4_r, c4_u), T3: (t3_r, d2, t3_l), T4: (t4_d, l2, t4_u)
        C_r = C.relabel("c4_u", "t4_u")
        CT_v = contract(C_r, T_v)                          # → (c4_r, t4_d, l2)
        T_h_r = T_h.relabel("t3_r", "c4_r")
        CTT = contract(CT_v, T_h_r)                        # → (t4_d, l2, d2, t3_l)
        Q = contract(CTT, a)                               # contracts l2, d2
        # Q free: (t4_d, t3_l, u2, r2). Relabel seams.
        return Q.relabels({"t4_d": "chi_T", "t3_l": "chi_R"})

    if position == "bottom_right":
        # C3: (c3_u, c3_l), T3: (t3_r, d2, t3_l), T2: (t2_u, r2, t2_d)
        C_r = C.relabel("c3_l", "t3_l")
        CT_h = contract(C_r, T_h)                          # → (c3_u, t3_r, d2)
        T_v_r = T_v.relabel("t2_d", "c3_u")
        CTT = contract(CT_h, T_v_r)                        # → (t3_r, d2, t2_u, r2)
        Q = contract(CTT, a)                               # contracts r2, d2
        # Q free: (t3_r, t2_u, l2, u2). Relabel seams.
        return Q.relabels({"t3_r": "chi_L", "t2_u": "chi_T"})

    raise ValueError(f"unsupported position={position!r}")
```

**Note on relabel correctness:** The seam-leg names (`chi_L`, `chi_R`, `chi_T`, `chi_B`, plus the D² labels `l2`, `r2`, `u2`, `d2`) are chosen so adjacent quarters auto-pair under `contract()`. Q_TL's `chi_R` matches Q_TR's `chi_L`; Q_TL's `r2` matches Q_TR's `l2`; Q_TL's `chi_B` matches Q_BL's `chi_T`; Q_TL's `d2` matches Q_BL's `u2`. Verified by Task 4's seam-contraction test.

**Step 4: Run all 4 position tests.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py -v
```
Expected: 4 PASS.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): _build_enlarged_corner for all 4 plaquette quarters"
```

---

## Task 4: Seam contraction sanity test

**Files:**
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the seam test.**

```python
def test_2x2_quarters_seam_contraction_for_uniform_state():
    """Q_TL · Q_TR (top row) and Q_BR · Q_BL (bottom row) auto-pair via the
    shared seam labels, producing rank-4 row matrices that can be
    matrix-multiplied via the chi seam."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    Q_TL = _build_enlarged_corner(env.C1, env.T1, env.T4, a, position="top_left")
    Q_TR = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position="top_right")
    Q_BL = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position="bottom_left")
    Q_BR = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position="bottom_right")

    from tenax.contraction.contractor import contract

    # Q_TL.chi_R + Q_TL.r2 contract with Q_TR.chi_L + Q_TR.l2 → 4 free legs
    top_row = contract(Q_TL, Q_TR)
    assert top_row.rank() == 4
    top_shapes = {ind.label: ind.dim for ind in top_row.indices}
    # Free legs: Q_TL's left/bottom seam-pair (chi_B, d2) and Q_TR's right/bottom (chi_B, d2 of Q_TR)
    # — Q_TL has no chi_L (it uses C1, no left chi), Q_TR has no chi_R; both have chi_B and d2.
    # The two chi_Bs / d2s come from different quarters and stay distinct via fully-qualified
    # contract; we expect 4 free legs total: (Q_TL.chi_B, Q_TL.d2, Q_TR.chi_B, Q_TR.d2)
    # but they share NO label after the seam contract — assertion is just rank.
    assert top_row.rank() == 4
```

(The `top_row` rank-4 result has the 2×2 plaquette's left-half-bottom-half seam plus right-half-bottom-half seam — 2 chi + 2 D² = 4 axes. That's the input shape `(chi, D², chi, D²)` to the next step's matrix product.)

**Step 2: Run.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_2x2_quarters_seam_contraction_for_uniform_state -v
```
Expected: PASS.

**Step 3: Commit.**

```bash
git add tests/test_ctm_tensor_projector_2x2.py
git commit -m "test(ctm): seam-contraction sanity for 2x2 plaquette quarters"
```

---

## Task 5: `_compute_2x2_projector` — left direction only

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the failing test.**

```python
def test_compute_2x2_projector_left_shape_and_isometry():
    """For a converged-style env, _compute_2x2_projector returns a (P_top, P_bot)
    pair that approximately satisfies P_top^† · P_bot ≈ I (Fishman cross-projector)
    on the chi_new = chi truncated space."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=0)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    Q_TL = _build_enlarged_corner(env.C1, env.T1, env.T4, a, position="top_left")
    Q_TR = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position="top_right")
    Q_BL = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position="bottom_left")
    Q_BR = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position="bottom_right")

    from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction="left"
    )

    # Shapes
    assert P_top.rank() == 4   # (chi_T1, D², chi_new, ...) — exact layout from impl
    assert P_bot.rank() == 4

    # Fishman cross projector: contract( P_top, P_bot ) over the (chi_outer, D²)
    # legs gives an identity on chi_new (up to truncation eps).
    from tenax.contraction.contractor import contract
    closure = contract(P_top, P_bot)  # should reduce to (chi_new, chi_new) ≈ I
    assert closure.rank() == 2
    closure_dense = closure.todense()
    eye = jnp.eye(closure_dense.shape[0], dtype=closure_dense.dtype)
    err = float(jnp.linalg.norm(closure_dense - eye))
    assert err < 1e-6, f"P_top^T · P_bot = I has Frobenius error {err:.2e}"
```

**Step 2: Run, verify failure.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_compute_2x2_projector_left_shape_and_isometry -v
```
Expected: FAIL — function not exported.

**Step 3: Implement `_compute_2x2_projector` for direction='left'.**

Add to `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`:

```python
import jax.numpy as jnp

from tenax.algorithms._ctm_projector import _truncated_SVD  # reuse existing
from tenax.contraction.contractor import contract
from tenax.core.flow import FlowDirection
from tenax.core.tensor import DenseTensor
from tenax.core.tensor_index import TensorIndex


__all__ = ["_build_enlarged_corner", "_compute_2x2_projector"]


# ------------------------------------------------------------------ #
# Internal: dense Fishman cross-projector on a 2x2 row product       #
# ------------------------------------------------------------------ #


def _fishman_row_projector_dense(
    top_M: jnp.ndarray,
    bottom_M: jnp.ndarray,
    chi: int,
    truncation_eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Standard Fishman 2x2 cross-projector (clean-room re-implementation of
    the procedure in Corboz–Penc–Mila–Lauchli PRB 84, 041108(R) (2011)).

    Returns (P_top, P_bot) as dense matrices:
      P_top : (chi_outer, chi_new)
      P_bot : (chi_new, chi_outer)
    where chi_outer = top_M.shape[0] = bottom_M.shape[1].
    """
    top_U, top_S, _ = jnp.linalg.svd(top_M, full_matrices=False)
    top_S = jnp.where(top_S / top_S[0] >= truncation_eps, top_S, 0.0)

    _, bot_S, bot_Vh = jnp.linalg.svd(bottom_M, full_matrices=False)
    bot_S = jnp.where(bot_S / bot_S[0] >= truncation_eps, bot_S, 0.0)

    top_half = top_U * jnp.sqrt(top_S)[None, :]              # (chi_outer, kept)
    bot_half = jnp.sqrt(bot_S)[:, None] * bot_Vh              # (kept, chi_outer)

    M_prime = bot_half @ top_half                             # small (kept, kept)

    # Reuse the existing truncated SVD (handles multiplet, S_inv_sqrt threshold).
    S_inv_sqrt, U_M, V_M_h, _ = _truncated_SVD(
        M_prime, chi, truncation_eps
    )

    P_top = top_half @ V_M_h.conj().T * S_inv_sqrt[None, :]
    P_bot = S_inv_sqrt[:, None] * U_M.conj().T @ bot_half
    return P_top, P_bot


def _compute_2x2_projector(
    Q_TL: Tensor,
    Q_TR: Tensor,
    Q_BL: Tensor,
    Q_BR: Tensor,
    chi: int,
    *,
    direction: str,
    truncation_eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Compute (P_top, P_bot) projector pair from the 2x2 plaquette quarters.

    For direction='left':
      top_M = (Q_TL · Q_TR) reshaped to (chi_T4, D²_T4, chi_T2, D²_T2)
              then 2D-flattened with (chi_T4, D²_T4) as rows.
      bottom_M = (Q_BR · Q_BL) reshaped analogously, with the bottom row
              of the plaquette and rows on the chi_T2 side.
      Fishman cross-projector → (P_top_dense, P_bot_dense).
      Reshape to rank-4 Tensors with seam labels matching the absorption
      contractions in `_ctm_tensor_move_left_2x2`.

    Tensor labels:
      P_top : ("chi_outer", "fused_D2", "chi_new", <stub>)
              with axes (chi_T4, D², chi_new) — the 4th axis is a degenerate
              slot retained for symmetric-tensor extension; for dense it is
              dim 1 and we squeeze before returning.
    """
    if direction != "left":
        raise NotImplementedError(f"direction={direction!r} not implemented yet")

    # 1) Form top_row = Q_TL · Q_TR, contracting (chi_R, r2) seam.
    top_row = contract(Q_TL, Q_TR)
    # top_row free legs after auto-pair on (chi_R↔chi_L, r2↔l2):
    #   from Q_TL: (chi_B, d2)
    #   from Q_TR: (chi_B, d2)
    # We need a 2D matrix: (Q_TL.chi_B · Q_TL.d2) × (Q_TR.chi_B · Q_TR.d2).
    # Tenax's `Tensor.todense()` + reshape is the simplest path for this scalar
    # SVD step.
    # NOTE: the labels on top_row are ambiguous because both quarters carry
    # chi_B + d2. Disambiguate by relabeling before the contract.
    Q_TL_r = Q_TL.relabels({"chi_B": "chi_B_L", "d2": "d2_L"})
    Q_TR_r = Q_TR.relabels({"chi_B": "chi_B_R", "d2": "d2_R"})
    top_row = contract(Q_TL_r, Q_TR_r)
    # top_row labels: (chi_B_L, d2_L, chi_B_R, d2_R), all dim chi or D²
    top_dense = top_row.todense()
    chi_B_L_pos = top_row.indices_index_by_label("chi_B_L")
    d2_L_pos = top_row.indices_index_by_label("d2_L")
    chi_B_R_pos = top_row.indices_index_by_label("chi_B_R")
    d2_R_pos = top_row.indices_index_by_label("d2_R")
    # Permute to (chi_B_L, d2_L, chi_B_R, d2_R) order then reshape to 2D.
    perm = [chi_B_L_pos, d2_L_pos, chi_B_R_pos, d2_R_pos]
    top_dense = jnp.transpose(top_dense, perm)
    chi_outer_top = top_dense.shape[0] * top_dense.shape[1]
    top_M = top_dense.reshape(chi_outer_top, -1)

    # 2) Form bottom_row = Q_BR · Q_BL, with rows on the right-side chi seam.
    Q_BR_r = Q_BR.relabels({"chi_T": "chi_T_R", "u2": "u2_R"})
    Q_BL_r = Q_BL.relabels({"chi_T": "chi_T_L", "u2": "u2_L"})
    # Q_BR contracts with Q_BL via shared (chi_R↔chi_L, r2↔l2) — automatic
    # because we did NOT relabel those.
    bottom_row = contract(Q_BR_r, Q_BL_r)
    # bottom_row labels: (chi_T_R, u2_R, chi_T_L, u2_L)
    bottom_dense = bottom_row.todense()
    chi_T_R_pos = bottom_row.indices_index_by_label("chi_T_R")
    u2_R_pos = bottom_row.indices_index_by_label("u2_R")
    chi_T_L_pos = bottom_row.indices_index_by_label("chi_T_L")
    u2_L_pos = bottom_row.indices_index_by_label("u2_L")
    perm = [chi_T_R_pos, u2_R_pos, chi_T_L_pos, u2_L_pos]
    bottom_dense = jnp.transpose(bottom_dense, perm)
    bottom_M = bottom_dense.reshape(chi_outer_top, -1)

    # 3) Fishman cross-projector on the dense matrices.
    P_top_dense, P_bot_dense = _fishman_row_projector_dense(
        top_M, bottom_M, chi, truncation_eps
    )
    # P_top_dense: (chi_outer_top, chi_new)
    # P_bot_dense: (chi_new, chi_outer_top)
    chi_new = P_top_dense.shape[1]

    # 4) Reshape (chi_outer_top = chi · D²) → (chi, D, D) and wrap as Tensor.
    chi_outer = top_dense.shape[0]            # = chi (T4-side chi)
    D_outer = int(round((chi_outer_top // chi_outer) ** 0.5))
    P_top_4d = P_top_dense.reshape(chi_outer, D_outer * D_outer, chi_new)
    P_bot_4d = P_bot_dense.reshape(chi_new, chi_outer, D_outer * D_outer)

    # Return as DenseTensors with explicit labels.
    P_top_t = DenseTensor(
        P_top_4d,
        [
            TensorIndex(label="chi_outer", dim=chi_outer, flow=FlowDirection.IN),
            TensorIndex(label="d2", dim=D_outer * D_outer, flow=FlowDirection.IN),
            TensorIndex(label="chi_new", dim=chi_new, flow=FlowDirection.OUT),
        ],
    )
    P_bot_t = DenseTensor(
        P_bot_4d,
        [
            TensorIndex(label="chi_new", dim=chi_new, flow=FlowDirection.IN),
            TensorIndex(label="chi_outer", dim=chi_outer, flow=FlowDirection.OUT),
            TensorIndex(label="d2", dim=D_outer * D_outer, flow=FlowDirection.OUT),
        ],
    )
    return P_top_t, P_bot_t
```

**Note:** The shape/label conventions in this stub are draft. Task 6 will exercise them with real absorption code; if anything mismatches we update both ends.

**Step 4: Run the test.**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_compute_2x2_projector_left_shape_and_isometry -v
```
Expected: may FAIL on the rank-4 assertion or isometry — adjust the implementation until both PASS. The Fishman post-processing alone should produce P_top^† · P_bot = I; if not, debug `_fishman_row_projector_dense` against scratch numpy.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): _compute_2x2_projector for left direction with Fishman cross-projector"
```

---

## Task 6: variPEPS numerics cross-check on a fixed random tensor

**Files:**
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the cross-check test.**

```python
@pytest.mark.slow
def test_2x2_projector_matches_varipeps_on_fixed_seed(tmp_path):
    """Numerical cross-check: build the variPEPS converged env (loaded from
    tests/_ctm_2x2_reference_data.npz, generated by Task 1's helper), compute
    the 2x2 left projector via Tenax, and verify that the Fishman cross-
    projector identity P_top^† · P_bot ≈ I holds to 1e-6.

    This is an indirect cross-check (we don't compare projector tensors
    directly because of gauge differences). It verifies our Fishman
    implementation produces a valid projector pair on a non-trivial state.
    """
    npz_path = pytest.importorskip("pathlib").Path("tests/_ctm_2x2_reference_data.npz")
    if not npz_path.exists():
        pytest.skip("Reference data missing; run examples/dev/gen_2x2_projector_reference.py")

    data = np.load(npz_path)
    # ... build Tenax CTMTensorEnv from variPEPS-converged C/T tensors via reshape
    # ... run _build_enlarged_corner + _compute_2x2_projector
    # ... assert closure error < 1e-6
```

(Only the structure shown; the env-injection mapping needs the leg-permutation table from `examples/dev/d2_varipeps_rdm_compare.py:_tenax_to_varipeps_tensor` adapted in reverse. Defer the full body to implementation time — placeholder is enough to gate the task.)

**Step 2: Skip-by-default. Marked slow.**

The reference data file is gitignored; the test should `pytest.skip` cleanly when missing. Verify with `uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_2x2_projector_matches_varipeps_on_fixed_seed -v` → SKIP.

**Step 3: Commit.**

```bash
git add tests/test_ctm_tensor_projector_2x2.py
git commit -m "test(ctm): variPEPS numerics cross-check stub (slow, gated on ref data)"
```

---

## Task 7: `_compute_2x2_projector` — right, top, bottom directions

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write parametrized failing test.**

```python
@pytest.mark.parametrize("direction", ["right", "top", "bottom"])
def test_compute_2x2_projector_other_directions_isometry(direction):
    """The Fishman cross-projector identity holds in all 4 directions."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=1)  # different seed for variety
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    Q_TL = _build_enlarged_corner(env.C1, env.T1, env.T4, a, position="top_left")
    Q_TR = _build_enlarged_corner(env.C2, env.T1, env.T2, a, position="top_right")
    Q_BL = _build_enlarged_corner(env.C4, env.T3, env.T4, a, position="bottom_left")
    Q_BR = _build_enlarged_corner(env.C3, env.T3, env.T2, a, position="bottom_right")

    from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector
    P_top, P_bot = _compute_2x2_projector(Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction)

    closure = contract(P_top, P_bot)
    closure_dense = closure.todense()
    eye = jnp.eye(closure_dense.shape[0], dtype=closure_dense.dtype)
    err = float(jnp.linalg.norm(closure_dense - eye))
    assert err < 1e-6, f"direction={direction}: err {err:.2e}"
```

**Step 2: Run, verify 3 fail (NotImplementedError).**

```bash
uv run pytest tests/test_ctm_tensor_projector_2x2.py::test_compute_2x2_projector_other_directions_isometry -v
```

**Step 3: Implement the 3 missing branches.**

For each direction, the algebra is the same Fishman cross-projector but on a different cut of the 2x2:
- right: top_M = Q_TR · Q_TL (reverse), bottom_M = Q_BL · Q_BR (truncates right-side chi)
- top: left_M = Q_BL · Q_TL, right_M = Q_TR · Q_BR (truncates top-side chi)
- bottom: left_M = Q_TL · Q_BL, right_M = Q_BR · Q_TR (truncates bottom-side chi)

**Step 4: Run all 4 directions.**

Expected: 4 PASS.

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): _compute_2x2_projector for all 4 plaquette directions"
```

---

## Task 8: `_ctm_tensor_move_left_2x2` — full move with absorption

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py`
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write a failing integration test.**

```python
def test_ctm_tensor_move_left_2x2_uniform_state_one_step():
    """One step of _ctm_tensor_move_left_2x2 on a uniform-state plaquette
    (all 4 sites same A) reproduces the same chi-truncated env as
    _ctm_tensor_move_left up to gauge — checked via |C1_2x2|_F = |C1_1x1|_F."""
    D, chi, d = 2, 4, 2
    A = _dense_tensor_5leg(D, d, seed=2)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)

    from tenax.algorithms._ctm_tensor_moves import (
        _ctm_tensor_move_left,
        _ctm_tensor_move_left_2x2,
    )
    env_1x1 = _ctm_tensor_move_left(env, env, a, chi, "svd")
    env_2x2 = _ctm_tensor_move_left_2x2(env, env, env, env, a, a, a, a, chi, "svd")

    # Frobenius norms of corner singular values should agree (gauge-invariant).
    sv_1x1 = jnp.linalg.svd(env_1x1.C1.todense(), compute_uv=False)
    sv_2x2 = jnp.linalg.svd(env_2x2.C1.todense(), compute_uv=False)
    # Sort descending, normalize sum to 1.
    sv_1x1 = jnp.sort(sv_1x1)[::-1]; sv_1x1 = sv_1x1 / jnp.sum(sv_1x1)
    sv_2x2 = jnp.sort(sv_2x2)[::-1]; sv_2x2 = sv_2x2 / jnp.sum(sv_2x2)
    assert float(jnp.max(jnp.abs(sv_1x1 - sv_2x2))) < 5e-2  # loose; recipes differ
```

**Step 2: Run, verify failure.**

Expected: FAIL — `_ctm_tensor_move_left_2x2` doesn't exist.

**Step 3: Implement `_ctm_tensor_move_left_2x2`.**

Add to `src/tenax/algorithms/_ctm_tensor_moves.py`:

```python
def _ctm_tensor_move_left_2x2(
    env_TL: CTMTensorEnv,
    env_TR: CTMTensorEnv,
    env_BL: CTMTensorEnv,
    env_BR: CTMTensorEnv,
    a_TL: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
) -> CTMTensorEnv:
    """Left CTM move using 2×2 plaquette projectors (Corboz-Penc-Mila-Lauchli).

    Updates env_TL.{C1, C4, T4}.

    Plaquette layout (s = self at top-left):
        s_TL ── s_TR
         │       │
        s_BL ── s_BR

    All env_*.{C1, T1, T4, ...} legs labelled per CTMTensorEnv convention.
    """
    from tenax.algorithms._ctm_tensor_projector_2x2 import (
        _build_enlarged_corner,
        _compute_2x2_projector,
    )

    Q_TL = _build_enlarged_corner(env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left")
    Q_TR = _build_enlarged_corner(env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right")
    Q_BL = _build_enlarged_corner(env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left")
    Q_BR = _build_enlarged_corner(env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right")

    P_top, P_bot = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi,
        direction="left",
        projector_method=projector_method,
    )

    # Absorption (mirrors _ctm_tensor_move_left's absorption pattern but with
    # the new (P_top, P_bot) pair).
    # new_C1 = (C1 · T1, projected on the bottom side by P_top)
    # new_C4 = (C4 · T3, projected on the top side by P_bot)
    # new_T4 = (T4 · a, projected by both P_top and P_bot pair on top/bottom)
    # Exact label-relabel sequence: TBD during implementation; mirror
    # _apply_projector_with_reembed's pattern.

    # Phase-fix + normalize per the existing single-step convention.
    C1_new = _phase_fix_normalize_tensor(C1_new)
    C4_new = _phase_fix_normalize_tensor(C4_new)
    T4_new = _phase_fix_normalize_tensor(T4_new)
    return env_TL._replace(C1=C1_new, C4=C4_new, T4=T4_new)
```

**Note:** The absorption contraction details are sketched; the implementation must match the specific seam labels emitted by `_compute_2x2_projector` in Task 5. Iterate on contraction ordering until the test in Step 1 passes. The integration test itself is the ground truth.

**Step 4: Run.**

Expected: PASS (or close — the 5e-2 tolerance is loose because the two recipes converge to slightly different fixed points even on a uniform state).

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_moves.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): _ctm_tensor_move_left_2x2 with 2x2 plaquette absorption"
```

---

## Task 9: `_ctm_tensor_move_{right,top,bottom}_2x2`

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py`
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Parametrized integration test for the other 3 directions.**

```python
@pytest.mark.parametrize("direction", ["right", "top", "bottom"])
def test_ctm_tensor_move_2x2_one_step_other_directions(direction):
    """One step of _ctm_tensor_move_<direction>_2x2 produces a sensible env."""
    # ... build uniform A, a, env
    # ... call the appropriate move function with 4 envs, 4 a's
    # ... assert env corners are well-formed (no NaN, sv > 0, etc.)
```

**Step 2: Run, verify 3 fails.**

**Step 3: Implement the 3 missing functions.** Cyclic permutation of Task 8.

**Step 4: Run, expect PASS for all 3.**

**Step 5: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_moves.py tests/test_ctm_tensor_projector_2x2.py
git commit -m "feat(ctm): 2x2 plaquette move functions for all 4 directions"
```

---

## Task 10: Wire 2×2 moves into `_ctm_tensor_sweep_multisite`

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py`

**Step 1: Add `recipe` kwarg to `_ctm_tensor_sweep_multisite` (default `"2x2"`).**

```python
def _ctm_tensor_sweep_multisite(
    envs, double_layers, neighbors, chi, renormalize,
    projector_method="svd", projector_backward="auto",
    *, recipe: str = "2x2",
) -> dict[Coord, CTMTensorEnv]:
    """One full multisite CTM sweep.

    recipe: '2x2' uses the standard plaquette projector (Corboz-Penc-Mila-Lauchli);
            '1x1' uses the legacy 2-tensor projector (kept for transitional pinning).
    """
    if recipe == "2x2":
        return _ctm_tensor_sweep_multisite_2x2(envs, double_layers, neighbors, chi, renormalize, projector_method, projector_backward)
    elif recipe == "1x1":
        return _ctm_tensor_sweep_multisite_1x1(envs, double_layers, neighbors, chi, renormalize, projector_method, projector_backward)
    raise ValueError(f"unknown recipe={recipe!r}; expected '1x1' or '2x2'")


def _ctm_tensor_sweep_multisite_1x1(...):  # the original code
    # body unchanged

def _ctm_tensor_sweep_multisite_2x2(...):
    # iterate over coords + directions; for each coord, gather 4 plaquette sites
    # via neighbors[coord]['right'], ['bottom'], and the diagonal; call the
    # appropriate _ctm_tensor_move_*_2x2.
    ...
```

**Step 2: Propagate `recipe` through `_ctm_tensor_multisite` and `ctm_multisite`.**

Add `recipe="2x2"` kwarg to both, threaded down to `_ctm_tensor_sweep_multisite`.

**Step 3: Existing tests should still pass at recipe="2x2"** (or verifiably fail with diagnostics). Run:

```bash
uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py -v
```
Expected: tests still pass — the new recipe is more accurate, not less.

If any tests fail with the new recipe (e.g. because they pinned to 1x1's specific energy), bisect with `recipe="1x1"` to confirm they're 1x1-pinned, then either:
(a) re-pin to 2x2 (preferred — these tests measure physical energies that should match the more-accurate recipe),
(b) explicitly mark them as 1x1-only with `recipe="1x1"` (with a TODO to remove).

**Step 4: Commit.**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py
git commit -m "feat(ctm): _ctm_tensor_sweep_multisite defaults to 2x2 plaquette recipe"
```

---

## Task 11: Tier-1 contract test — saved AD-optimum matches variPEPS

**Files:**
- Create: `tests/test_ctm_multisite_2x2_contract.py`

**Step 1: Write the contract test.**

```python
"""Contract test: 2x2 multisite-CTM at the saved D=4 chi=16 AD-optimum
gives E/site ≈ -0.255 (matching variPEPS gate API, see project memory
project_c3_floor_breach_smoking_gun.md)."""
import pytest
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from tenax.algorithms.pess import IPESSState
from tenax.algorithms._pess_multisite_energy import kagome_xxz_pair_hamiltonian


@pytest.mark.slow
def test_kagome_3site_multisite_2x2_at_d4_ad_optimum_matches_varipeps():
    npz_path = Path("logs/d4_ad_optimum.npz")
    if not npz_path.exists():
        pytest.skip(f"saved AD-optimum {npz_path} missing; regenerate via examples/dev/save_d4_ad_optimum.py")

    npz = np.load(npz_path)
    state = IPESSState(
        R_a=jnp.asarray(npz["R_a"]),
        R_b=jnp.asarray(npz["R_b"]),
        R_c=jnp.asarray(npz["R_c"]),
        T_u=jnp.asarray(npz["T_u"]),
        T_d=jnp.asarray(npz["T_d"]),
        lambdas=tuple(jnp.asarray(npz[f"lambda_{i}"]) for i in range(6)),
    )
    H = jnp.asarray(kagome_xxz_pair_hamiltonian(delta=1.0, d=2))

    import sys; sys.path.insert(0, "examples")
    from kagome_pess_multisite_phase_c3_rdm_brute_force_diag import _collect_ctm_rdms

    rdms = _collect_ctm_rdms(state, chi=16, max_iter=120, conv_tol=1e-9)
    bonds = ("uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col")
    E = sum(float(complex(jnp.einsum("ijkl,ijkl->", rdms[b], H)).real) for b in bonds) / 3.0

    # variPEPS gate API target (per p2_varipeps_chi_scan.json):
    target = -0.255359
    assert abs(E - target) < 1e-3, (
        f"2x2 multisite-CTM E/site = {E:.6f}, target {target:.6f} (diff {E - target:+.4f})"
    )
```

**Step 2: Run.**

```bash
uv run pytest tests/test_ctm_multisite_2x2_contract.py -v -m slow
```
Expected: PASS at E/site ≈ -0.2554 ± 1e-3.

If PASS — the bug is fixed.
If FAIL — debug. Likely places: incorrect projector reshape (Task 5), wrong absorption seam (Task 8), or a subtle leg-permutation in `_compute_2x2_projector`. The Tier-2 vanilla regression (Task 12) should catch most of these without needing the saved AD-optimum.

**Step 3: Commit.**

```bash
git add tests/test_ctm_multisite_2x2_contract.py
git commit -m "test(ctm): tier-1 contract — 2x2 multisite at AD-optimum matches variPEPS"
```

---

## Task 12: Tier-2 vanilla regression test

**Files:**
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the test.**

```python
def test_2x2_multisite_matches_1x1_for_uniform_dense_state():
    """For a translation-invariant single-site state, the 2x2 multisite recipe
    should agree with the 1x1 single-site CTM on a 1-site observable (e.g.
    Sz expectation) to chi-truncation tolerance at D=2 chi=16."""
    D, chi, d = 2, 16, 2
    A = _dense_tensor_5leg(D, d, seed=3)
    A_jax = A._data  # raw jnp array

    from tenax.algorithms._ctm_tensor_convergence import (
        ctm_multisite,
        ctm_tensor,
        SINGLE_SITE_NEIGHBORS,
    )
    from tenax.core.lattice import square

    # 1x1 single-site CTM
    env_1x1 = ctm_tensor(A, chi=chi, conv_tol=1e-8)
    # 2x2 multisite CTM on the trivial "single-site" lattice
    env_2x2 = ctm_multisite({"a": A}, square(), chi=chi, conv_tol=1e-8, recipe="2x2")["a"]

    # Compare 1-site Sz expectations (gauge-invariant).
    Sz = jnp.diag(jnp.array([0.5, -0.5]))
    e_1x1 = _one_site_expectation(A, env_1x1, Sz)
    e_2x2 = _one_site_expectation(A, env_2x2, Sz)
    assert abs(e_1x1 - e_2x2) < 1e-3
```

**Step 2: Run, debug if disagreement is bigger than 1e-3.**

Expected: PASS with discrepancy ≤ 1e-3 (chi truncation has different effects but should agree at the converged-fixed-point level).

**Step 3: Commit.**

```bash
git add tests/test_ctm_tensor_projector_2x2.py
git commit -m "test(ctm): tier-2 vanilla regression for 2x2 multisite path"
```

---

## Task 13: Tier-3 AD-FD agreement

**Files:**
- Modify: `tests/test_ctm_tensor_projector_2x2.py`

**Step 1: Write the AD-FD test.**

```python
def test_2x2_multisite_ctm_ad_matches_fd_at_d2():
    """jax.grad of (CTM energy via 2x2) agrees with finite-difference at D=2
    random state, chi=8. Wirtinger convention matched. This verifies the new
    projector composes cleanly with the existing implicit-AD GMRES path."""
    # ... build a small d=2 D=2 chi=8 single-site iPEPS
    # ... define energy_fn(A) = trace(rho(A, ctm_multisite(A, chi=8, recipe="2x2")) @ H)
    # ... compute jax.grad(energy_fn)(A) and compare to finite-difference
    # ... assert max diff < 1e-5
```

**Step 2: Run. Debug projector primitive ops if AD ≠ FD.**

The Fishman SVD path should be JAX-traceable; if not, trace through `_compute_2x2_projector` to find a non-traceable op.

**Step 3: Commit.**

```bash
git add tests/test_ctm_tensor_projector_2x2.py
git commit -m "test(ctm): tier-3 AD-FD agreement for 2x2 multisite path"
```

---

## Task 14: Run full regression and benchmark

**Step 1: Run all multisite tests.**

```bash
uv run pytest tests/ -k "multisite" -v
```
Expected: all pass; investigate any failures (re-pin or revert per Task 10's note).

**Step 2: Run core test suite.**

```bash
uv run pytest -m core -v
```
Expected: green.

**Step 3: Sanity benchmark.**

Run `examples/dev/p1_tenax_chi_scan.py` after the change (no script edit needed — it uses the default recipe). Expected: `chi=16  E/site = -0.255 ± 1e-3` (matching variPEPS).

**Step 4: Compare wall time.**

Compare to the old log (`logs/p1_tenax_chi_scan.log` shows ~3-50 s for chi=8-48 on CPU). New should be < 10× slower at chi=16 on CPU. If > 30×, raise concern in the PR.

**Step 5: Update memory + close out.**

Update `~/.claude/projects/-home-yjkao-tenax/memory/project_c3_floor_breach_smoking_gun.md` with the resolution status (fix shipped, contract met).

**Step 6: Commit (if any final tweaks).**

---

## Task 15: Open PR

**Step 1: Run pre-commit on all files.**

```bash
uv run pre-commit run --all-files
```
Expected: PASS or auto-fixes.

**Step 2: Push branch.**

```bash
git push -u origin worktree-fix-multisite-ctm-rdm-helpers
```

**Step 3: Open PR.**

```bash
gh pr create --title "feat(ctm): 2x2 plaquette projector for multisite path" --body "$(cat <<'EOF'
## Summary
- Replace `_ctm_tensor_sweep_multisite`'s 1×1 grown-corner projector with the standard 2×2 enlarged-corner (Corboz-Penc-Mila-Lauchli, PRB 84, 041108(R) (2011)) recipe
- New file `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` containing `_build_enlarged_corner` and `_compute_2x2_projector`
- 4 new `_ctm_tensor_move_*_2x2` move functions in `_ctm_tensor_moves.py`
- DenseTensor only; SymmetricTensor follow-up
- Single-site `_ctm_tensor_sweep` and paired moves untouched
- Closes the C.3 floor-breach diagnosed at saved D=4 chi=16 AD-optimum: E/site goes from -0.913 (1×1 recipe, non-physical) to -0.255 (matches variPEPS gate API)

## Test plan
- [ ] tier-1 contract: `pytest tests/test_ctm_multisite_2x2_contract.py -m slow` passes (E/site = -0.2554 ± 1e-3 at saved AD-optimum)
- [ ] tier-2 vanilla: `pytest tests/test_ctm_tensor_projector_2x2.py::test_2x2_multisite_matches_1x1_for_uniform_dense_state` passes
- [ ] tier-3 AD-FD: `pytest tests/test_ctm_tensor_projector_2x2.py::test_2x2_multisite_ctm_ad_matches_fd_at_d2` passes
- [ ] existing multisite witnesses still green: `pytest tests/test_pess_3site_multisite_rdm_invariants.py`
- [ ] core: `pytest -m core` green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 4: Return PR URL.**

---

## Reference cross-links

- Design: `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md`
- Diagnosis memory: `~/.claude/projects/-home-yjkao-tenax/memory/project_c3_floor_breach_smoking_gun.md`
- Probes: `examples/dev/{p1_tenax_chi_scan, p2_varipeps_chi_scan, p3_tenax_projector_scan}.py`; logs in `logs/`
- variPEPS reference (read-only, GPL-3.0; do **not** copy): `varipeps/ctmrg/{absorption,projectors}.py`
- Saved AD-optimum: `logs/d4_ad_optimum.npz`
