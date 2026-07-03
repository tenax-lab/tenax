# 2-site Joint Split-CTM — Dense Forward (Phases 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a genuine joint 2-site (AB checkerboard) split-CTM *forward* on the DenseTensor path, so `(env_A, env_B)` are genuinely coupled (A's env absorbs B's double-layer and vice versa), producing 2-site energies that match the fused `ctm_tensor_2site` oracle to ~1e-10.

**Architecture:** Approach A from the design doc — a parallel split multisite driver (`dict[Coord, SplitCTMTensorEnv]`) that structurally mirrors the fused `_ctm_tensor_multisite` / `_ctm_tensor_sweep_multisite` (recipe `"2x2"`), but over split (ket/bra-separated) edges. The split *enlarged corner* is built to return the SAME rank-4 object as the fused `_build_enlarged_corner`, so the Fishman projector `_compute_2x2_projector` is reused verbatim and `(P_top, P_bot)` are parity-testable. Four split absorb functions grow ket/bra edges, apply the projector halves leg-by-leg (`_apply_proj_unfused`), and SVD-split the result back into ket/bra.

**Tech Stack:** Python, JAX, Tenax tensor protocol (`DenseTensor`, `contract`, `tenax.linalg.svd`). Tests use `pytest -m core`.

**Scope:** DenseTensor forward + 2-site energy parity ONLY. AD (Phase 2), SymmetricTensor (Phase 3), and fermionic (Phase 4) are separate plans per the design doc. Chi-bump/schedule stay guarded off.

**Design doc:** `docs/superpowers/specs/2026-07-02-2site-split-ctm-joint-forward-design.md`

---

## Orientation: reference code to mirror (read before starting)

These are the existing functions the new code twins. All under `src/tenax/algorithms/`.

### Fused multisite driver — the exact structural template

`_ctm_tensor_convergence.py:711` `_ctm_tensor_multisite(site_tensors, neighbors, chi, ...)`:
```python
double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
# ... optional QR warmup ...
prev_svs = {}
for _ in range(max_iter):
    envs, _, _ = _ctm_tensor_sweep_multisite(envs, double_layers, neighbors, chi,
                                             renormalize, projector_method,
                                             projector_backward=..., recipe=recipe)
    converged = True
    for c in sorted(envs):
        sv = _corner_singular_values(envs[c].C1)
        if c in prev_svs:
            if float(_ctm_sv_diff(sv, prev_svs[c])) >= conv_tol:
                converged = False
        else:
            converged = False
        prev_svs[c] = sv
    if converged:
        break
return envs
```

`_ctm_tensor_convergence.py:787` `ctm_tensor_2site(A, B, chi, ...)` builds `{(0,0):A,(1,0):B}` with `CHECKERBOARD_NEIGHBORS` and delegates to `_ctm_tensor_multisite`.

`_ctm_tensor_convergence.py:274` `_ctm_tensor_sweep_multisite` — the `recipe == "2x2"` branch is the template for `_split_ctm_sweep_multisite`. Its shape:
```python
for direction in ("left", "top", "right", "bottom"):
    envs_old = dict(envs)
    projectors = {}
    for s_anchor in all_coords:
        s_TR = neighbors[s_anchor]["right"]; s_BL = neighbors[s_anchor]["bottom"]
        s_BR = neighbors[s_TR]["bottom"]
        P_top, P_bot, eps_T, smallest_S = _compute_plaquette_projector_pair(
            envs_old[s_anchor], envs_old[s_TR], envs_old[s_BL], envs_old[s_BR],
            double_layers[s_anchor], double_layers[s_TR],
            double_layers[s_BL], double_layers[s_BR],
            chi, direction, base_charges=base_charges)
        projectors[s_anchor] = (P_top, P_bot)
    new_envs = {}
    for s_dst in _sort_coords_for_direction(all_coords, direction):
        if direction == "left":
            s_src = neighbors[s_dst]["left"]
            s_above_anchor = neighbors[s_src]["top"]
            P_top_above, P_bot_above = projectors[s_above_anchor]
            P_top_curr, P_bot_curr = projectors[s_src]
            C1_new, T4_new, C4_new = _ctm_tensor_absorb_left_2plaq(
                envs_old[s_src], double_layers[s_src],
                P_top_above, P_bot_above, P_top_curr, P_bot_curr)
            new_envs[s_dst] = envs_old[s_dst]._replace(C1=C1_new, T4=T4_new, C4=C4_new)
        elif direction == "right":
            s_src = neighbors[s_dst]["right"]; s_above_anchor = neighbors[s_dst]["top"]
            P_top_above, P_bot_above = projectors[s_above_anchor]
            P_top_curr, P_bot_curr = projectors[s_dst]
            C2_new, T2_new, C3_new = _ctm_tensor_absorb_right_2plaq(...)
            new_envs[s_dst] = envs_old[s_dst]._replace(C2=C2_new, T2=T2_new, C3=C3_new)
        elif direction == "top":
            s_src = neighbors[s_dst]["top"]; s_left_anchor = neighbors[s_src]["left"]
            P_top_left, P_bot_left = projectors[s_left_anchor]
            P_top_curr, P_bot_curr = projectors[s_src]
            C1_new, T1_new, C2_new = _ctm_tensor_absorb_top_2plaq(...)
            new_envs[s_dst] = envs_old[s_dst]._replace(C1=C1_new, T1=T1_new, C2=C2_new)
        else:  # bottom
            s_src = neighbors[s_dst]["bottom"]; s_left_anchor = neighbors[s_dst]["left"]
            P_top_left, P_bot_left = projectors[s_left_anchor]
            P_top_curr, P_bot_curr = projectors[s_dst]
            C4_new, T3_new, C3_new = _ctm_tensor_absorb_bottom_2plaq(...)
            new_envs[s_dst] = envs_old[s_dst]._replace(C4=C4_new, T3=T3_new, C3=C3_new)
    envs = new_envs
if renormalize:
    envs = {c: _renormalize_tensor_env(e) for c, e in envs.items()}
```

`CHECKERBOARD_NEIGHBORS` (`_ctm_tensor_convergence.py:191`), `_sort_coords_for_direction` (`:250`), `_get_base_charges` (`:237`) are reused as-is (import them).

### Fused enlarged corner + projector (reused verbatim by the split path)

`_ctm_tensor_projector_2x2.py:164` `_build_enlarged_corner(C, T_h, T_v, a, *, position)` — returns rank-4:
- `top_left`  → `(chi_R, r2, chi_B, d2)` (relabels `t1_r→chi_R`, `t4_u→chi_B`)
- `top_right` → `(chi_L, l2, chi_B, d2)`
- `bottom_left` → `(chi_T, u2, chi_R, r2)` (relabels `t4_d→chi_T`, `t3_l→chi_R`)
- `bottom_right` → `(chi_L, l2, chi_T, u2)` (relabels `t3_r→chi_L`, `t2_u→chi_T`; uses `C3.c3_u <-> T2.t2_d`, `C3.c3_l <-> T3.t3_l`)

`_ctm_tensor_projector_2x2.py:302` `_compute_2x2_projector(Q_TL, Q_TR, Q_BL, Q_BR, chi, *, direction, base_charges)` → `(P_top, P_bot, eps_T, smallest_S)`. **Reused verbatim** — the split enlarged corners feed it directly.

`_ctm_tensor_moves.py:80` `_apply_proj_unfused(P, env_T, chi_label, d2_label, *, chi_new="chi_new", env_first=False)` — applies a fused `(chi_outer, fused_D2)` projector leg-by-leg. **Reused verbatim.**

`_ctm_tensor_moves.py:395` `_compute_plaquette_projector_pair` — the fused projector-pair (calls `_build_enlarged_corner` ×4 then `_compute_2x2_projector`, then `_half_to_chi_new_top/_bot`). The split twin mirrors it with `_build_split_enlarged_corner`.

`_ctm_tensor_moves.py:689` `_ctm_tensor_absorb_bottom_2plaq(env_src, a_src, P_top_left, P_bot_left, P_top_curr, P_bot_curr)` — the fused bottom absorb, DenseTensor branch (quoted in full in the Appendix). This is the exact twin target for `_split_ctm_absorb_bottom_2plaq`, but producing ket/bra edges.

### Split-side helpers to reuse

`_split_ctm_tensor_init.py:448` `initialize_split_ctm_tensor_env(A, chi, chi_I)` — per-site env builder (reused per coord).

`_split_ctm_tensor_moves.py:63` `_doublelayer_grown_corner(C, T_ket, T_bra, c_relabel, ket_I, bra_I, fuse_labels)` — grows a corner from ket+bra edges into a fused `(env, u_ket, u_bra)` leg. Used to build the split enlarged corner's corner-part.

`_split_ctm_tensor_moves.py:233` `_svd_split_edge_tensor(Tg, left_labels, right_labels, chi_I, ket_relabels, bra_relabels, base_charges)` — SVD-splits a grown edge back into `(ket, bra)` across the interlayer bond. Reused in the absorb functions.

`_split_ctm_tensor_moves.py:1025` `_split_ctm_move_bottom(env, A, A_bar, chi, chi_I)` — the single-site split bottom move (quoted in the Appendix). Shows the corner-grow → project → edge-grow → SVD-split flow that the 2x2 absorb rearranges (projector precomputed, applied leg-by-leg).

`_split_ctm_tensor_energy.py:1133` `compute_energy_split_ctm_tensor_2site(env_A, env_B, A, B, h)` and `:1177` `compute_energy_split_ctm_tensor_multisite(...)` — energy on coupled envs (input change only; see Task 8).

---

## Phase 0 — multisite plumbing (smoke-testable)

### Task 0.1: Multisite env + per-coord (A, A_bar) map

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py`
- Test: `tests/test_split_ctm_2site.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split_ctm_2site.py
import numpy as np
import pytest
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
)
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv


def _random_dense_A(D=2, d=2, seed=0):
    """5-leg (u,d,l,r,phys) DenseTensor iPEPS site tensor."""
    from tenax.core.tensor import DenseTensor
    from tenax.core.index import TensorIndex, FlowDirection
    from tenax.core.symmetry import U1Symmetry  # trivial use; charges all zero
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((D, D, D, D, d)) + 0j
    # Use the same trivial-symmetry dense constructor the split path uses.
    from tenax.algorithms._split_ctm_tensor_init import _trivial_symmetry
    sym = _trivial_symmetry()
    def idx(n, flow, label):
        return TensorIndex.from_charges(sym, np.zeros(n, dtype=np.int32), flow, label=label)
    return DenseTensor(data, (
        idx(D, FlowDirection.OUT, "u"), idx(D, FlowDirection.OUT, "d"),
        idx(D, FlowDirection.OUT, "l"), idx(D, FlowDirection.OUT, "r"),
        idx(d, FlowDirection.OUT, "phys"),
    ))


def test_initialize_split_multisite_env_keys_and_type():
    A = _random_dense_A(seed=1)
    B = _random_dense_A(seed=2)
    envs = _initialize_split_multisite_env(
        {(0, 0): A, (1, 0): B}, chi=6, chi_I=6
    )
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    assert isinstance(envs[(0, 0)], SplitCTMTensorEnv)
    assert isinstance(envs[(1, 0)], SplitCTMTensorEnv)
    # A-env and B-env must be distinct objects built from distinct tensors.
    assert envs[(0, 0)].C1 is not envs[(1, 0)].C1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_initialize_split_multisite_env_keys_and_type -v`
Expected: FAIL with `ImportError: cannot import name '_initialize_split_multisite_env'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/_split_ctm_tensor_convergence.py` (after the imports; import `Coord` from `_ctm_tensor_convergence`):

```python
from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _corner_singular_values,
    _ctm_sv_diff,
)
from tenax.algorithms._split_ctm_tensor_init import (
    initialize_split_ctm_tensor_env,
)


def _initialize_split_multisite_env(
    site_tensors: dict[Coord, Tensor],
    chi: int,
    chi_I: int,
) -> dict[Coord, SplitCTMTensorEnv]:
    """Per-coord split env init: reuse the single-site builder per site."""
    return {
        c: initialize_split_ctm_tensor_env(A, chi, chi_I)
        for c, A in site_tensors.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_initialize_split_multisite_env_keys_and_type -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_convergence.py
git commit -m "feat(#463): split-CTM multisite env init (per-coord)"
```

---

### Task 0.2: Multisite driver + sweep skeleton reusing single-site moves (1x1 recipe smoke)

This lands the `dict[Coord, ...]` driver plumbing before the 2x2 absorb functions exist, using the existing single-site moves as a smoke test. For a UNIFORM cell (A == B) the 1x1-per-site sweep must reproduce the single-site `ctm_split_tensor` energy.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py`
- Test: `tests/test_split_ctm_2site.py`

- [ ] **Step 1: Write the failing test**

```python
def test_split_multisite_uniform_matches_single_site():
    """recipe='1x1' multisite sweep on a uniform cell == single-site forward."""
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_multisite,
        ctm_split_tensor,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm_1site_split_tensor,
    )
    A = _random_dense_A(seed=3)
    chi = 6
    single = ctm_split_tensor(A, chi, max_iter=20, conv_tol=0.0)
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): A}, CHECKERBOARD_NEIGHBORS, chi,
        max_iter=20, conv_tol=0.0, recipe="1x1",
    )
    rho_single = _rdm_1site_split_tensor(single, A)
    rho_multi = _rdm_1site_split_tensor(envs[(0, 0)], A)
    assert np.allclose(
        rho_single.todense(), rho_multi.todense(), atol=1e-8
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_multisite_uniform_matches_single_site -v`
Expected: FAIL with `ImportError: cannot import name '_split_ctm_multisite'`.

- [ ] **Step 3: Write minimal implementation**

Add to `_split_ctm_tensor_convergence.py`. The `recipe="1x1"` branch reuses the existing single-site moves (`_split_ctm_move_left`, etc.) per coord, passing the *neighbor's* `(A, A_bar)` as the absorbed tensor:

```python
from tenax.algorithms._ctm_tensor_convergence import _sort_coords_for_direction
from tenax.algorithms._split_ctm_tensor_moves import (
    _split_ctm_move_left,
    _split_ctm_move_top,
    _split_ctm_move_right,
    _split_ctm_move_bottom,
)

_SPLIT_DIRECTION_MOVES = {
    "left": _split_ctm_move_left,
    "top": _split_ctm_move_top,
    "right": _split_ctm_move_right,
    "bottom": _split_ctm_move_bottom,
}


def _split_ctm_sweep_multisite(
    envs: dict[Coord, SplitCTMTensorEnv],
    site_tensors: dict[Coord, Tensor],
    bars: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    chi_I: int,
    renormalize: bool,
    recipe: str = "2x2",
) -> dict[Coord, SplitCTMTensorEnv]:
    """One full split multisite CTM sweep. recipe='1x1' | '2x2'."""
    envs = dict(envs)
    all_coords = list(envs.keys())
    if recipe == "1x1":
        for direction in ("left", "top", "right", "bottom"):
            move_fn = _SPLIT_DIRECTION_MOVES[direction]
            for coord in _sort_coords_for_direction(all_coords, direction):
                nb = neighbors[coord][direction]
                # The single-site move rebuilds the absorbed edge from the
                # neighbor's (A, A_bar) but reads/writes ``coord``'s env.
                merged = move_fn(envs[coord], site_tensors[nb], bars[nb], chi, chi_I)
                envs[coord] = merged
    elif recipe == "2x2":
        envs = _split_ctm_sweep_multisite_2x2(
            envs, site_tensors, bars, neighbors, chi, chi_I
        )
    else:
        raise ValueError(f"Unknown split CTM recipe {recipe!r}: expected '1x1' or '2x2'.")
    if renormalize:
        envs = {c: _renormalize_split_env(e) for c, e in envs.items()}
    return envs


def _split_ctm_multisite(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    recipe: str = "2x2",
) -> dict[Coord, SplitCTMTensorEnv]:
    """Run split multisite CTM to convergence (mirror of _ctm_tensor_multisite)."""
    if chi_I is None:
        chi_I = chi
    bars = {c: A.bar() for c, A in site_tensors.items()}
    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    prev_svs: dict[Coord, "jax.Array"] = {}
    for _ in range(max_iter):
        envs = _split_ctm_sweep_multisite(
            envs, site_tensors, bars, neighbors, chi, chi_I, renormalize, recipe
        )
        converged = True
        for c in sorted(envs):
            sv = _corner_singular_values(envs[c].C1)
            if c in prev_svs:
                if float(_ctm_sv_diff(sv, prev_svs[c])) >= conv_tol:
                    converged = False
            else:
                converged = False
            prev_svs[c] = sv
        if converged:
            break
    return envs
```

Add a temporary stub so the `2x2` branch imports resolve (replaced in Task 1.3):

```python
def _split_ctm_sweep_multisite_2x2(envs, site_tensors, bars, neighbors, chi, chi_I):
    raise NotImplementedError("2x2 split sweep lands in Task 1.3")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_multisite_uniform_matches_single_site -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_convergence.py
git commit -m "feat(#463): split-CTM multisite driver + 1x1 sweep (smoke)"
```

---

## Phase 1 — dense 2x2 joint forward

### Task 1.1: Split enlarged corner (parity with fused `_build_enlarged_corner`)

The linchpin: the split enlarged corner, built from ket+bra split edges and separate `(A, A_bar)`, must return the SAME rank-4 tensor as the fused `_build_enlarged_corner` (so `_compute_2x2_projector` is reused). We prove it by parity.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py`
- Test: `tests/test_split_ctm_2site.py`

- [ ] **Step 1: Write the failing test**

```python
def _split_env_and_fused_env(A, chi):
    """Build a matched pair: a split env and the equivalent fused env from
    the same converged single-site fused CTM, for enlarged-corner parity."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
    fused = ctm_tensor(A, chi, max_iter=30, conv_tol=0.0)
    split = ctm_split_tensor(A, chi, chi_I=chi, max_iter=30, conv_tol=0.0)
    return split, fused


@pytest.mark.parametrize("position", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_split_enlarged_corner_matches_fused(position):
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._split_ctm_tensor_moves import _build_split_enlarged_corner
    A = _random_dense_A(seed=5)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    # Fused reference (uses the fused env's fused corners/edges).
    if position == "top_left":
        Q_ref = _build_enlarged_corner(fused.C1, fused.T1, fused.T4, a, position=position)
        Q_split = _build_split_enlarged_corner(
            split.C1, split.T1_ket, split.T1_bra, split.T4_ket, split.T4_bra,
            A, A_bar, position=position)
    elif position == "top_right":
        Q_ref = _build_enlarged_corner(fused.C2, fused.T1, fused.T2, a, position=position)
        Q_split = _build_split_enlarged_corner(
            split.C2, split.T1_ket, split.T1_bra, split.T2_ket, split.T2_bra,
            A, A_bar, position=position)
    elif position == "bottom_left":
        Q_ref = _build_enlarged_corner(fused.C4, fused.T3, fused.T4, a, position=position)
        Q_split = _build_split_enlarged_corner(
            split.C4, split.T3_ket, split.T3_bra, split.T4_ket, split.T4_bra,
            A, A_bar, position=position)
    else:  # bottom_right
        Q_ref = _build_enlarged_corner(fused.C3, fused.T3, fused.T2, a, position=position)
        Q_split = _build_split_enlarged_corner(
            split.C3, split.T3_ket, split.T3_bra, split.T2_ket, split.T2_bra,
            A, A_bar, position=position)
    # Compare after aligning label order (contraction invariant up to the
    # env-bond gauge; the single-site uniform env has identical corners so
    # raw parity holds up to normalization).
    Qr = Q_ref.transpose(tuple(sorted(range(Q_ref.rank))))  # canonical order
    Qs = Q_split.transpose(tuple(Qs_i for Qs_i in _match_axes(Q_split, Q_ref)))
    dr = Qr.todense(); ds = Qs.todense()
    dr = dr / np.max(np.abs(dr)); ds = ds / np.max(np.abs(ds))
    assert np.allclose(np.abs(dr), np.abs(ds), atol=1e-7)
```

Add the small axis-matching helper at the top of the test file:

```python
def _match_axes(src, ref):
    """Return a permutation of src's axes so its labels line up with ref's."""
    ref_labels = list(ref.labels())
    src_labels = list(src.labels())
    return tuple(src_labels.index(lbl) for lbl in ref_labels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py -k test_split_enlarged_corner_matches_fused -v`
Expected: FAIL with `ImportError: cannot import name '_build_split_enlarged_corner'`.

- [ ] **Step 3: Write minimal implementation**

Add `_build_split_enlarged_corner` to `_split_ctm_tensor_moves.py`. It mirrors `_build_enlarged_corner` (Appendix A) but: (a) the corner+edges are grown with ket AND bra halves joined over the interlayer bond, and (b) the physical double-layer is formed from `A` (ket) and `A_bar` (bra) with the physical index traced. Build it so the four OPEN legs carry the fused-path labels (`chi_R`/`chi_B`/`r2`/`d2`, etc.).

Concretely, for `top_left` (the others follow `_build_enlarged_corner`'s recipe with ket/bra edges):

```python
from tenax.contraction.contractor import contract


def _build_split_enlarged_corner(
    C: Tensor,
    T_h_ket: Tensor,
    T_h_bra: Tensor,
    T_v_ket: Tensor,
    T_v_bra: Tensor,
    A: Tensor,
    A_bar: Tensor,
    *,
    position: str,
) -> Tensor:
    """Split enlarged corner. Returns the SAME rank-4 object as the fused
    :func:`_build_enlarged_corner`, assembled from ket/bra split edges and
    a physical double layer (A ket, A_bar bra) with the phys index traced.

    Output free legs match the fused recipe exactly:
      top_left     -> (chi_R, r2, chi_B, d2)
      top_right    -> (chi_L, l2, chi_B, d2)
      bottom_left  -> (chi_T, u2, chi_R, r2)
      bottom_right -> (chi_L, l2, chi_T, u2)
    """
    if position == "top_left":
        # Corner joins T1 (horizontal) and T4 (vertical) over the env bonds;
        # each edge is a ket/bra pair joined over its interlayer bond.
        # 1) grow C with the T1 ket/bra pair (interlayer t1k_I<->t1b_I):
        C_r = C.relabel("c1_r", "t1k_l")
        CTh = contract(C_r, T1_ket_labeled(T_h_ket))       # (c1_d, u_ket, t1k_I)
        CTh = contract(CTh.relabel("t1k_I", "t1b_I"), T_h_bra)  # (c1_d, u_ket, u_bra, t1b_r)
        # 2) join T4 ket/bra pair over the vertical env bond (c1_d<->t4k_d):
        Tv = contract(T_v_ket.relabel("t4k_d", "c1_d"), CTh)   # brings (l_ket, t4k_I, ...)
        Tv = contract(Tv.relabel("t4k_I", "t4b_I"), T_v_bra)   # adds (l_bra, t4b_u)
        # 3) contract physical double layer: ket legs u_ket,l_ket with A;
        #    bra legs u_bra,l_bra with A_bar; trace phys.
        Q = _absorb_phys_double_layer(
            Tv, A, A_bar,
            ket_map={"u_ket": "u", "l_ket": "l"},
            bra_map={"u_bra": "u", "l_bra": "l"},
        )  # open ket virtual d,r -> d2/r2 fused with bra; env bonds t1b_r,t4b_u
        # 4) relabel open env seams and D^2 seams to the fused convention.
        return Q.relabels({"t1b_r": "chi_R", "t4b_u": "chi_B"})
    ...  # top_right / bottom_left / bottom_right by the same recipe
    raise ValueError(f"unsupported position={position!r}")
```

Where `_absorb_phys_double_layer(env_grown, A, A_bar, ket_map, bra_map)` contracts the two open ket virtual legs against `A`, the two open bra virtual legs against `A_bar`, traces the shared `phys`, and fuses each remaining ket/bra virtual-leg pair into the `d2`/`r2`-style D²-seam using `fuse_indices` (the same fusion the fused double layer carries). Implement it with `contract` + `fuse_indices` following the ket/bra leg names in `SplitCTMTensorEnv` (`_split_ctm_tensor_init.py:45-56`). The exact per-position recipes come straight from `_build_enlarged_corner` (Appendix A): use the same `C.relabel`/seam-relabel pairs, only splitting each edge contraction into a ket step then a bra step joined over the interlayer bond.

> Implementation note: `_build_split_enlarged_corner` is a mechanical merge of two quoted references — the fused `_build_enlarged_corner` (Appendix A, for the contraction/relabel recipe per position) and the single-site split `_doublelayer_grown_corner` (which shows the ket→bra interlayer-join pattern). Keep the D²-seam fusion order (ket-first, then bra) identical to `_doublelayer_grown_corner` so charges line up with the fused `a`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py -k test_split_enlarged_corner_matches_fused -v`
Expected: PASS for all four positions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_moves.py
git commit -m "feat(#463): split enlarged corner (parity with fused _build_enlarged_corner)"
```

---

### Task 1.2: Split plaquette projector pair (parity with fused)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py`
- Test: `tests/test_split_ctm_2site.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_split_plaquette_projector_matches_fused(direction):
    from tenax.algorithms._ctm_tensor_moves import _compute_plaquette_projector_pair
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
    )
    A = _random_dense_A(seed=7)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    Pt_ref, Pb_ref, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, direction)
    Pt_s, Pb_s, _, _ = _compute_split_plaquette_projector_pair(
        split, split, split, split, A, A_bar, A, A_bar, A, A_bar, A, A_bar,
        chi, direction)
    # Projectors match up to a per-column sign/gauge; compare the closure
    # invariant P_bot . P_top (== identity on chi_new) and |P| magnitudes.
    assert np.allclose(
        np.sort(np.abs(Pt_ref.todense()).ravel()),
        np.sort(np.abs(Pt_s.todense()).ravel()),
        atol=1e-6,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py -k test_split_plaquette_projector_matches_fused -v`
Expected: FAIL with `ImportError: cannot import name '_compute_split_plaquette_projector_pair'`.

- [ ] **Step 3: Write minimal implementation**

Add to `_split_ctm_tensor_moves.py`. Direct twin of `_compute_plaquette_projector_pair` (Appendix B), swapping `_build_enlarged_corner` for `_build_split_enlarged_corner` and importing the reused `_compute_2x2_projector`, `_half_to_chi_new_top`, `_half_to_chi_new_bot`:

```python
from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector
from tenax.algorithms._ctm_tensor_moves import (
    _half_to_chi_new_top,
    _half_to_chi_new_bot,
)


def _compute_split_plaquette_projector_pair(
    env_TL, env_TR, env_BL, env_BR,
    A_TL, Abar_TL, A_TR, Abar_TR, A_BL, Abar_BL, A_BR, Abar_BR,
    chi, direction, base_charges=None,
):
    """Split twin of _compute_plaquette_projector_pair.

    Builds the four split enlarged corners (identical rank-4 objects to the
    fused path) and feeds the reused Fishman cross-projector verbatim.
    """
    Q_TL = _build_split_enlarged_corner(
        env_TL.C1, env_TL.T1_ket, env_TL.T1_bra, env_TL.T4_ket, env_TL.T4_bra,
        A_TL, Abar_TL, position="top_left")
    Q_TR = _build_split_enlarged_corner(
        env_TR.C2, env_TR.T1_ket, env_TR.T1_bra, env_TR.T2_ket, env_TR.T2_bra,
        A_TR, Abar_TR, position="top_right")
    Q_BL = _build_split_enlarged_corner(
        env_BL.C4, env_BL.T3_ket, env_BL.T3_bra, env_BL.T4_ket, env_BL.T4_bra,
        A_BL, Abar_BL, position="bottom_left")
    Q_BR = _build_split_enlarged_corner(
        env_BR.C3, env_BR.T3_ket, env_BR.T3_bra, env_BR.T2_ket, env_BR.T2_bra,
        A_BR, Abar_BR, position="bottom_right")
    P_top_raw, P_bot_raw, eps_T, smallest_S = _compute_2x2_projector(
        Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction, base_charges=base_charges)
    return (
        _half_to_chi_new_top(P_top_raw),
        _half_to_chi_new_bot(P_bot_raw),
        eps_T,
        smallest_S,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py -k test_split_plaquette_projector_matches_fused -v`
Expected: PASS for all four directions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_moves.py
git commit -m "feat(#463): split plaquette projector pair (parity with fused)"
```

---

### Task 1.3: The four split absorb functions + 2x2 sweep

Each `_split_ctm_absorb_{left,right,top,bottom}_2plaq` is the twin of the fused `_ctm_tensor_absorb_*_2plaq` (Appendix C shows `bottom`), but grows ket/bra edges, applies the passed-in projector halves via `_apply_proj_unfused`, and SVD-splits the new edge into ket/bra via `_svd_split_edge_tensor`.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py`
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py` (`_split_ctm_sweep_multisite_2x2`)
- Test: `tests/test_split_ctm_2site.py`

- [ ] **Step 1: Write the failing test (per-move env parity, bottom first)**

```python
def test_split_absorb_bottom_matches_fused():
    """Bottom absorb: split (C4,T3,C3) == fused after ket/bra contraction."""
    from tenax.algorithms._ctm_tensor_moves import (
        _ctm_tensor_absorb_bottom_2plaq,
    )
    from tenax.algorithms._ctm_tensor_moves import _compute_plaquette_projector_pair
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._split_ctm_tensor_moves import (
        _split_ctm_absorb_bottom_2plaq,
        _compute_split_plaquette_projector_pair,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_edge_to_fused,  # ket/bra -> single fused edge helper (Task 8 dep)
    )
    A = _random_dense_A(seed=11)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A); A_bar = A.bar()
    Ptl, Pbl, _, _ = _compute_plaquette_projector_pair(
        fused, fused, fused, fused, a, a, a, a, chi, "bottom")
    Ptc, Pbc = Ptl, Pbl  # uniform env: left-anchor and curr plaquettes coincide
    C4f, T3f, C3f = _ctm_tensor_absorb_bottom_2plaq(fused, a, Ptl, Pbl, Ptc, Pbc)
    sPtl, sPbl, _, _ = _compute_split_plaquette_projector_pair(
        split, split, split, split, A, A_bar, A, A_bar, A, A_bar, A, A_bar,
        chi, "bottom")
    C4s, T3k, T3b, C3s = _split_ctm_absorb_bottom_2plaq(
        split, A, A_bar, sPtl, sPbl, sPtl, sPbl, chi_I=chi)
    T3s = _split_edge_to_fused(T3k, T3b, "T3")  # -> fused (t3_r, d2, t3_l)
    def norm(x): d = x.todense(); return np.sort(np.abs(d).ravel())
    assert np.allclose(norm(C4s), norm(C4f), atol=1e-6)
    assert np.allclose(norm(C3s), norm(C3f), atol=1e-6)
    assert np.allclose(norm(T3s), norm(T3f), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_absorb_bottom_matches_fused -v`
Expected: FAIL with `ImportError: cannot import name '_split_ctm_absorb_bottom_2plaq'`.

- [ ] **Step 3: Write minimal implementation**

Add the four absorb functions. `_split_ctm_absorb_bottom_2plaq` mirrors the fused `_ctm_tensor_absorb_bottom_2plaq` (Appendix C) with the `c3_u <-> t2_d` convention, but on ket/bra edges. Structure (bottom):

```python
def _split_ctm_absorb_bottom_2plaq(
    env_src: SplitCTMTensorEnv,
    A: Tensor,
    A_bar: Tensor,
    P_top_left: Tensor,
    P_bot_left: Tensor,
    P_top_curr: Tensor,
    P_bot_curr: Tensor,
    chi_I: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split BOTTOM absorption -> (C4_new, T3_ket_new, T3_bra_new, C3_new).

    Twin of the fused ``_ctm_tensor_absorb_bottom_2plaq`` (Appendix C):
    same C3.c3_u<->T2.t2_d convention (#674/#670), same projector-half
    application via ``_apply_proj_unfused``, but grows ket/bra edge halves
    and SVD-splits the new T3 into ket/bra over the interlayer bond.
    """
    # ---- C4·T4 (grow both ket and bra of the left edge, join interlayer) ----
    C4_r = env_src.C4.relabel("c4_r", "t4k_u")
    C4g = contract(C4_r, env_src.T4_ket)               # (c4_u, l_ket, t4k_I)
    C4g = contract(C4g.relabel("t4k_I", "t4b_I"), env_src.T4_bra)  # +(l_bra, t4b_u)
    # ---- C3·T2 with the c3_u<->t2_d convention (#670) ----
    C3_u = env_src.C3.relabel("c3_u", "t2b_d")
    C3g = contract(C3_u, env_src.T2_bra)               # (c3_l, r_bra, t2b_I)
    C3g = contract(C3g.relabel("t2b_I", "t2k_I"), env_src.T2_ket)  # +(r_ket, t2k_u)
    # ---- T3·A·A_bar (grow the edge double layer) ----
    T3g = contract(env_src.T3_ket, A)                  # ket layer
    T3g = contract(T3g, env_src.T3_bra)                # join bra edge
    T3g = contract(T3g, A_bar)                         # bra layer
    # ---- apply projector halves leg-by-leg (mirror Appendix C relabels) ----
    C4_new = _apply_proj_unfused(P_bot_left, C4g, "c4_u", "l2")
    C4_new = C4_new.relabels({"chi_new": "c4_r", "t4b_u": "c4_u"})
    C3_new = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")
    C3_new = C3_new.relabels({"chi_new": "c3_u", "t2k_u": "c3_l"})
    step = _apply_proj_unfused(P_top_left, T3g, "t3_r", "l2")
    T3g = _apply_proj_unfused(P_bot_curr, step, "t3_l", "r2",
                              chi_new="chi_new_r", env_first=True)
    # ---- SVD-split the projected edge into ket/bra over chi_I ----
    T3_ket_new, T3_bra_new = _svd_split_edge_tensor(
        T3g,
        left_labels=["chi_new", "d_ket"],
        right_labels=["d_bra", "chi_new_r"],
        chi_I=chi_I,
        ket_relabels={"chi_new": "t3k_r", "d_ket": "d_ket", "_svd_bond": "t3k_I"},
        bra_relabels={"_svd_bond": "t3b_I", "d_bra": "d_bra", "chi_new_r": "t3b_l"},
        base_charges=None,
    )
    C4_new = _ensure_corner_flows(C4_new, "C4")
    C3_new = _ensure_corner_flows(C3_new, "C3")
    T3_ket_new, T3_bra_new = _ensure_edge_flows(T3_ket_new, T3_bra_new, "T3")
    return C4_new, T3_ket_new, T3_bra_new, C3_new
```

> Note on `d_ket`/`d_bra` and `l2`/`r2` labels: the growth step must relabel the ket/bra physical-virtual legs to the `d2`-seam names the projector's `fused_D2` split expects. Follow the exact seam labels the fused Appendix-C code uses (`l2`, `r2`, `u2`→`d2`); on the split side the D²-seam is a ket/bra pair, so fuse them the same ket-first order as `_doublelayer_grown_corner` before applying `_apply_proj_unfused`, and carry the ket/bra split through to `_svd_split_edge_tensor`. The `left`/`right`/`top` twins follow the fused `_ctm_tensor_absorb_{left,right,top}_2plaq` recipes identically (grow the corresponding corner/edge triple, same projector-half order).

Import the helpers already in the module: `_apply_proj_unfused` (from `_ctm_tensor_moves`), `_ensure_corner_flows`, `_ensure_edge_flows`, `_svd_split_edge_tensor` (local).

Then implement `_split_ctm_sweep_multisite_2x2` in `_split_ctm_tensor_convergence.py` as the exact twin of the fused 2x2 branch (Orientation section), calling the four split absorb functions and `_replace`-ing ket/bra edge fields:

```python
def _split_ctm_sweep_multisite_2x2(envs, site_tensors, bars, neighbors, chi, chi_I):
    from tenax.algorithms._split_ctm_tensor_moves import (
        _compute_split_plaquette_projector_pair,
        _split_ctm_absorb_left_2plaq,
        _split_ctm_absorb_right_2plaq,
        _split_ctm_absorb_top_2plaq,
        _split_ctm_absorb_bottom_2plaq,
    )
    from tenax.algorithms._ctm_tensor_convergence import (
        _get_base_charges, _sort_coords_for_direction,
    )
    all_coords = list(envs.keys())
    base_charges = None  # dense path
    for direction in ("left", "top", "right", "bottom"):
        envs_old = dict(envs)
        projectors = {}
        for s in all_coords:
            s_TR = neighbors[s]["right"]; s_BL = neighbors[s]["bottom"]
            s_BR = neighbors[s_TR]["bottom"]
            Pt, Pb, _, _ = _compute_split_plaquette_projector_pair(
                envs_old[s], envs_old[s_TR], envs_old[s_BL], envs_old[s_BR],
                site_tensors[s], bars[s], site_tensors[s_TR], bars[s_TR],
                site_tensors[s_BL], bars[s_BL], site_tensors[s_BR], bars[s_BR],
                chi, direction, base_charges=base_charges)
            projectors[s] = (Pt, Pb)
        new_envs = {}
        for s_dst in _sort_coords_for_direction(all_coords, direction):
            if direction == "left":
                s_src = neighbors[s_dst]["left"]; s_a = neighbors[s_src]["top"]
                Pta, Pba = projectors[s_a]; Ptc, Pbc = projectors[s_src]
                C1n, T4k, T4b, C4n = _split_ctm_absorb_left_2plaq(
                    envs_old[s_src], site_tensors[s_src], bars[s_src],
                    Pta, Pba, Ptc, Pbc, chi_I)
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C1=C1n, T4_ket=T4k, T4_bra=T4b, C4=C4n)
            elif direction == "right":
                s_src = neighbors[s_dst]["right"]; s_a = neighbors[s_dst]["top"]
                Pta, Pba = projectors[s_a]; Ptc, Pbc = projectors[s_dst]
                C2n, T2k, T2b, C3n = _split_ctm_absorb_right_2plaq(
                    envs_old[s_src], site_tensors[s_src], bars[s_src],
                    Pta, Pba, Ptc, Pbc, chi_I)
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C2=C2n, T2_ket=T2k, T2_bra=T2b, C3=C3n)
            elif direction == "top":
                s_src = neighbors[s_dst]["top"]; s_a = neighbors[s_src]["left"]
                Ptl, Pbl = projectors[s_a]; Ptc, Pbc = projectors[s_src]
                C1n, T1k, T1b, C2n = _split_ctm_absorb_top_2plaq(
                    envs_old[s_src], site_tensors[s_src], bars[s_src],
                    Ptl, Pbl, Ptc, Pbc, chi_I)
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C1=C1n, T1_ket=T1k, T1_bra=T1b, C2=C2n)
            else:  # bottom
                s_src = neighbors[s_dst]["bottom"]; s_a = neighbors[s_dst]["left"]
                Ptl, Pbl = projectors[s_a]; Ptc, Pbc = projectors[s_dst]
                C4n, T3k, T3b, C3n = _split_ctm_absorb_bottom_2plaq(
                    envs_old[s_src], site_tensors[s_src], bars[s_src],
                    Ptl, Pbl, Ptc, Pbc, chi_I)
                new_envs[s_dst] = envs_old[s_dst]._replace(
                    C4=C4n, T3_ket=T3k, T3_bra=T3b, C3=C3n)
        envs = new_envs
    return envs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_absorb_bottom_matches_fused -v`
Expected: PASS. (Add analogous `test_split_absorb_{left,right,top}_matches_fused` mirroring the bottom test with the fused twin + corresponding `_replace` fields, and run them too.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_moves.py src/tenax/algorithms/_split_ctm_tensor_convergence.py
git commit -m "feat(#463): four split 2x2 absorb functions + split 2x2 sweep"
```

---

### Task 1.4: Public `ctm_split_tensor_2site` entry

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py`
- Test: `tests/test_split_ctm_2site.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ctm_split_tensor_2site_returns_two_envs():
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv
    A = _random_dense_A(seed=13); B = _random_dense_A(seed=14)
    env_A, env_B = ctm_split_tensor_2site(A, B, chi=6, max_iter=10, conv_tol=0.0)
    assert isinstance(env_A, SplitCTMTensorEnv)
    assert isinstance(env_B, SplitCTMTensorEnv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_ctm_split_tensor_2site_returns_two_envs -v`
Expected: FAIL with `ImportError: cannot import name 'ctm_split_tensor_2site'`.

- [ ] **Step 3: Write minimal implementation**

```python
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS


def ctm_split_tensor_2site(
    A: Tensor,
    B: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    chi_I: int | None = None,
    renormalize: bool = True,
    recipe: str = "2x2",
) -> tuple[SplitCTMTensorEnv, SplitCTMTensorEnv]:
    """Run 2-site checkerboard split-CTM to convergence.

    Twin of :func:`ctm_tensor_2site`: builds ``{(0,0): A, (1,0): B}`` with
    ``CHECKERBOARD_NEIGHBORS`` and delegates to :func:`_split_ctm_multisite`.
    Returns ``(env_A, env_B)`` genuinely coupled (A's env absorbs B and
    vice versa).
    """
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, chi,
        max_iter=max_iter, conv_tol=conv_tol, chi_I=chi_I,
        renormalize=renormalize, recipe=recipe,
    )
    return envs[(0, 0)], envs[(1, 0)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_ctm_split_tensor_2site_returns_two_envs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_convergence.py
git commit -m "feat(#463): public ctm_split_tensor_2site entry"
```

---

### Task 1.5: Fixed-point energy parity vs fused `ctm_tensor_2site` (Tier-2)

The acceptance test: converge both paths on a DIRECTION-DEPENDENT pair (A ≠ B) and compare 2-site energies.

**Files:**
- Test: `tests/test_split_ctm_2site.py`
- (No source change expected; if energy mismatches, debug per Risk notes.)

- [ ] **Step 1: Write the failing test**

```python
def _heisenberg_2site_h():
    """NN Heisenberg two-site gate as a (phys,phys,phys',phys') operator."""
    from tenax.operators.spin import heisenberg_two_site  # existing helper
    return heisenberg_two_site()


@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 8)])
def test_split_2site_energy_matches_fused(D, chi):
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import (
        compute_energy_ctm_tensor_2site,  # fused 2-site energy
    )
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )
    A = _random_dense_A(D=D, seed=21); B = _random_dense_A(D=D, seed=22)
    h = _heisenberg_2site_h()
    envA_f, envB_f = ctm_tensor_2site(A, B, chi, max_iter=40, conv_tol=1e-10)
    E_f = compute_energy_ctm_tensor_2site(envA_f, envB_f, A, B, h)
    envA_s, envB_s = ctm_split_tensor_2site(A, B, chi, chi_I=chi,
                                            max_iter=40, conv_tol=1e-10)
    E_s = compute_energy_split_ctm_tensor_2site(envA_s, envB_s, A, B, h)
    assert abs(float(E_s) - float(E_f)) < 1e-8
```

Confirm the exact names/signatures of `compute_energy_ctm_tensor_2site` and `heisenberg_two_site` before writing (grep `tests/test_split_ctm_tensor.py` for the shim comparison it already does — reuse that harness's operator + energy calls verbatim).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_2site_energy_matches_fused -v`
Expected: initially may FAIL on the tightest tolerance if a convention slipped; debug via the per-move parity tests (Tier-1) which localize the offending direction/edge.

- [ ] **Step 3: Fix any mismatch**

If Tier-1 per-move tests pass but Tier-2 fails, the issue is in convergence coupling or renormalization order — check that `_split_ctm_sweep_multisite_2x2` uses `envs_old` (snapshot) for BOTH projector build and absorption, exactly as the fused sweep does. If a specific direction's per-move test fails, the C↔T pairing relabel in that absorb function is wrong; compare against the fused Appendix-C twin line-by-line.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_2site_energy_matches_fused -v`
Expected: PASS for all (D, chi) params.

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py
git commit -m "test(#463): 2-site split-CTM energy parity vs fused oracle (Tier-2)"
```

---

### Task 1.6: Wire `compute_energy_split_ctm_tensor_2site` to route through multisite

Per design §6, route the 2-site energy through the N=2 case of `compute_energy_split_ctm_tensor_multisite` to avoid a divergent code path, and add the `_split_edge_to_fused` helper used by the Tier-1 tests if it doesn't already exist.

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_2site.py` (reuse Task 1.5 test as regression)

- [ ] **Step 1: Confirm current behavior**

Run: `grep -n "_split_edge_to_fused\|_split_env_to_tensor_standard\|def compute_energy_split_ctm_tensor_2site\|def compute_energy_split_ctm_tensor_multisite" src/tenax/algorithms/_split_ctm_tensor_energy.py`
If `_split_edge_to_fused` is absent, extract the ket⊗bra→fused-edge contraction already inside `_split_env_to_tensor_standard` into a small reusable helper (the Tier-1 tests import it).

- [ ] **Step 2: Write the failing test**

```python
def test_split_2site_energy_routes_through_multisite():
    """The _2site energy must equal the multisite N=2 energy for the same envs."""
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
        compute_energy_split_ctm_tensor_multisite,
    )
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    A = _random_dense_A(seed=31); B = _random_dense_A(seed=32)
    h = _heisenberg_2site_h()
    envA, envB = ctm_split_tensor_2site(A, B, chi=6, chi_I=6, max_iter=20, conv_tol=0.0)
    E2 = compute_energy_split_ctm_tensor_2site(envA, envB, A, B, h)
    Em = compute_energy_split_ctm_tensor_multisite(
        {(0, 0): envA, (1, 0): envB}, {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS, h)
    assert abs(float(E2) - float(Em)) < 1e-10
```

- [ ] **Step 3: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_split_ctm_2site.py::test_split_2site_energy_routes_through_multisite -v`
If it passes already, the two paths agree — proceed to make `_2site` delegate to multisite internally (keeps one path). If it fails, align `compute_energy_split_ctm_tensor_2site` to call `compute_energy_split_ctm_tensor_multisite` with the checkerboard neighbor map.

- [ ] **Step 4: Implement delegation & rerun**

Make `compute_energy_split_ctm_tensor_2site` delegate:

```python
def compute_energy_split_ctm_tensor_2site(env_A, env_B, A, B, h):
    """2-site checkerboard energy = N=2 case of the multisite energy."""
    from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
    return compute_energy_split_ctm_tensor_multisite(
        {(0, 0): env_A, (1, 0): env_B},
        {(0, 0): A, (1, 0): B},
        CHECKERBOARD_NEIGHBORS,
        h,
    )
```

Run: `uv run pytest tests/test_split_ctm_2site.py -v`
Expected: all PASS (including the Task 1.5 parity test as regression).

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_ctm_2site.py src/tenax/algorithms/_split_ctm_tensor_energy.py
git commit -m "refactor(#463): route split 2-site energy through multisite (single path)"
```

---

### Task 1.7: Full suite + marker check

- [ ] **Step 1: Run the split + fused CTM suites**

Run: `uv run pytest tests/test_split_ctm_2site.py tests/test_split_ctm_tensor.py -v`
Expected: all PASS.

- [ ] **Step 2: Run core marker suite (CI gate)**

Run: `uv run pytest -m core -q`
Expected: all PASS (no regressions).

- [ ] **Step 3: Commit any test-marker or import fixups**

```bash
git commit -am "test(#463): split 2-site suite green under -m core" --allow-empty
```

---

## Appendix — reference code (quoted verbatim for the executor)

### Appendix A — `_build_enlarged_corner` (`_ctm_tensor_projector_2x2.py:164`)

(See Orientation for the per-position seam recipes; the four branches use:
`top_left`: `C1.c1_r<->T1.t1_l`, `C1.c1_d<->T4.t4_d`, then absorb `a`, relabel `t1_r→chi_R`, `t4_u→chi_B`.
`top_right`: `C2.c2_l<->T1.t1_r`, `C2.c2_d<->T2.t2_u`, relabel `t1_l→chi_L`, `t2_d→chi_B`.
`bottom_left`: `C4.c4_r<->T4.t4_u`, `C4.c4_u<->T3.t3_r`, relabel `t4_d→chi_T`, `t3_l→chi_R`.
`bottom_right`: `C3.c3_l<->T3.t3_l`, `C3.c3_u<->T2.t2_d`, relabel `t3_r→chi_L`, `t2_u→chi_T`.)

### Appendix B — `_compute_plaquette_projector_pair` (`_ctm_tensor_moves.py:395`)

```python
Q_TL = _build_enlarged_corner(env_TL.C1, env_TL.T1, env_TL.T4, a_TL, position="top_left")
Q_TR = _build_enlarged_corner(env_TR.C2, env_TR.T1, env_TR.T2, a_TR, position="top_right")
Q_BL = _build_enlarged_corner(env_BL.C4, env_BL.T3, env_BL.T4, a_BL, position="bottom_left")
Q_BR = _build_enlarged_corner(env_BR.C3, env_BR.T3, env_BR.T2, a_BR, position="bottom_right")
P_top_raw, P_bot_raw, eps_T, smallest_S = _compute_2x2_projector(
    Q_TL, Q_TR, Q_BL, Q_BR, chi, direction=direction, base_charges=base_charges)
return (_half_to_chi_new_top(P_top_raw), _half_to_chi_new_bot(P_bot_raw), eps_T, smallest_S)
```

### Appendix C — `_ctm_tensor_absorb_bottom_2plaq` DenseTensor branch (`_ctm_tensor_moves.py:689`)

```python
# C4·T4
C4_r = env_src.C4.relabel("c4_r", "t4_u"); C4g = contract(C4_r, env_src.T4)
# C3·T2 with C3.c3_u <-> T2.t2_d (#670)
C3_u = env_src.C3.relabel("c3_u", "t2_d"); C3g = contract(C3_u, env_src.T2)
# T3·ket
T3_with_a = contract(env_src.T3, a_src)
# C4 project with P_bot_left
C4_new = _apply_proj_unfused(P_bot_left, C4g, "c4_u", "l2")
C4_new = C4_new.relabels({"chi_new": "c4_r", "t4_d": "c4_u"})
# C3 project with P_top_curr
C3_new = _apply_proj_unfused(P_top_curr, C3g, "c3_l", "r2")
C3_new = C3_new.relabels({"chi_new": "c3_u", "t2_u": "c3_l"})
# T3 sandwiched: P_top_left (t3_r,l2) then P_bot_curr (t3_l,r2), env_first=True
step = _apply_proj_unfused(P_top_left, T3_with_a, "t3_r", "l2")
T3_new = _apply_proj_unfused(P_bot_curr, step, "t3_l", "r2", chi_new="chi_new_r", env_first=True)
T3_new = T3_new.relabels({"chi_new": "t3_r", "chi_new_r": "t3_l", "u2": "d2"})
T3_new = _flip_leg_flow(T3_new, "d2")
# phase-fix normalize all three
```

### Appendix D — single-site split bottom move (`_split_ctm_tensor_moves.py:1025`)

Shows the corner-grow (`_doublelayer_grown_corner`) → project (`_apply_projector`) → edge-grow (`_grow_and_project_edge_lr`) → SVD-split (`_svd_split_edge_tensor`) flow. The 2x2 absorb rearranges this: projectors are precomputed (Task 1.2) and applied leg-by-leg (`_apply_proj_unfused`) instead of the 1x1 biorthogonal pair. Reuse `_svd_split_edge_tensor`, `_ensure_corner_flows`, `_ensure_edge_flows` from here.

---

## Notes for the executor

- **Verify signatures before each task.** A few helper names/signatures (`compute_energy_ctm_tensor_2site`, `heisenberg_two_site`, `_flip_leg_flow`, `_half_to_chi_new_top/_bot`, `_split_edge_to_fused`) are cited from context and MUST be confirmed with a quick `grep` before use; if a name differs, use the actual one (do not invent).
- **Parity comparisons use magnitude/sorted-SV invariants**, not raw tensors, because the split env carries an extra interlayer bond with residual gauge freedom in degenerate subspaces (design §10, #425 class). Where a raw comparison is possible (uniform env), prefer it; otherwise compare `np.sort(np.abs(...))` or energies.
- **Debugging order is Tier-1 → Tier-2.** If the energy parity (1.5) fails, the per-move parity tests (1.1–1.3) localize which direction/edge/convention slipped. Never chase Tier-2 without green Tier-1.
- **Do not touch the fused path.** All new code lives in `_split_ctm_tensor_*`. The fused functions are imported and reused read-only.
