# Split-CTM double-layer corner-pair projector (DenseTensor) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the split-CTM forward so `ctm_split_tensor` converges to the correct fixed point by building renormalization projectors from **double-layer corner pairs** (not single-layer corners), keeping envs split (χ²·D⁴ memory).

**Architecture:** In each of the four split moves, replace the per-layer projector construction with a double-layer corner-pair projector via the existing `_compute_projector_tensor` (Fishman biorthogonal `(P_1, P_2)`). Apply `P_1` to the C1-side and `P_2` to the C4-side of both corners and the edge. Implement the edge application with the **closed** path first (correctness), then switch to the **factorized bounded** path (memory). Validate every step against an exact-parity oracle: `split(χ, χ_I=χ·D) == fused(χ)`.

**Tech Stack:** Python, JAX (float64), Tenax Tensor protocol, pytest.

**Spec:** `docs/superpowers/specs/2026-06-27-split-ctmrg-doublelayer-projector-design.md`
**Reference:** [[reference_split_ctmrg_paper]] (Naumann et al. PRB 111 235116); the corner-pair variant here is the spike-proven, figure-free construction. Half-system projectors are a deferred accuracy upgrade.

---

## Background the implementer needs

- **The bug:** `_split_ctm_move_left` (and the other 3 in `src/tenax/algorithms/_split_ctm_tensor_moves.py`) call `_compute_projector_tensor(C1g_ket_fused, C4g_ket_fused, chi)` where `C1g_ket = contract(C1, T1_ket)` is **ket-only**. The projector then truncates the environment bond using only ket information → wrong fixed point (~40% energy error vs the fused path; verified).
- **The fix:** build the grown corner as a **double layer** — `C1` contracted with **both** `T1_ket` and `T1_bra` (joined over the interlayer `_I` bond) — so the fused leg is `(env, u_ket, u_bra)` (dim χ·D²), exactly like the fused move's `C1g` (`_ctm_tensor_moves.py:1018`, fused leg `(c1_d, u2)` with `u2`=D²). Then `_compute_projector_tensor` selects the correct χ directions.
- **`_compute_projector_tensor(C1g, C4g, chi, ...)`** (`src/tenax/algorithms/_ctm_projector.py:830+`) returns `(P_1, P_2, eps_T)`. `P_1` is for the C1g side, `P_2` for the C4g side; `P_1†P_2 = I`. Each has labels `(fused, chi_new)`. **The current split code discards `P_2` and applies `P_1` to both sides — that is wrong for the biorthogonal pair and must be fixed.**
- **Split env labels** (from `SplitCTMTensorEnv` in `_split_ctm_tensor_init.py`): corners `C1..C4` with bond labels like `c1_d,c1_r`; edges `T1_ket(t1k_l,u_ket,t1k_I)`, `T1_bra(t1b_I,u_bra,t1b_r)`, etc. (ket/bra share the interlayer `_I`). These are the labels the proven spike code uses.
- **Oracle:** the trusted fused path is `ctm_tensor(A, chi, ...)` (`_ctm_tensor_convergence.py`) + `compute_energy_ctm_tensor(A, env, gate)` (`_ctm_tensor_energy.py`). At lossless `χ_I = χ·D`, the split path must reproduce it exactly.

Always enable float64 in tests: `jax.config.update("jax_enable_x64", True)` at import time.

---

## File Structure

- **Create** `tests/_split_ctm_oracle.py` — test helper: `make_site`, `heisenberg_gate`, `fused_env_to_split` (proven spike code) + `evolve_physical_site`. One responsibility: build oracle inputs/conversions for tests.
- **Create** `tests/test_split_ctm_doublelayer_projector.py` — all new tests.
- **Modify** `src/tenax/algorithms/_split_ctm_tensor_moves.py` — projector construction + factorization helpers; rewrite the four moves' Phase A; edge application with separate left/right projectors.
- **Modify** `src/tenax/algorithms/_split_ctm_tensor_convergence.py` — `min_iter` convergence guard in `ctm_split_tensor`.

---

## Task 1: Oracle test helper (`fused_env_to_split`) + round-trip proof

**Files:**
- Create: `tests/_split_ctm_oracle.py`
- Test: `tests/test_split_ctm_doublelayer_projector.py`

- [ ] **Step 1: Create the oracle helper module.** Copy the **proven** spike code verbatim into `tests/_split_ctm_oracle.py`: the functions `_make_site`→`make_site`, `_heisenberg_gate`→`heisenberg_gate`, `_unfuse_d2_leg`, `_svd_split`, `_fused_env_to_split`→`fused_env_to_split`, `_evolve_physical_site`→`evolve_physical_site`, `_norm_spectrum`, `_eff_rank`. The exact bodies are in `docs/superpowers/specs/`-referenced spike (`/tmp/dl_spike.py` if present); they use:
  - `from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv`
  - `from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard, compute_energy_split_ctm_tensor`
  - `from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor`
  - `from tenax.linalg import svd as tensor_svd`
  - `from tenax.algorithms._tensor_utils import absorb_sqrt_singular_values, max_abs_normalize`
  - `from tenax.core.index import FlowDirection, TensorIndex`; `from tenax.core.symmetry import U1Symmetry`; `from tenax.core.tensor import DenseTensor`

  `fused_env_to_split(fused_env, D, chi_I)` unfuses each fused edge's `D²` leg into `(ket, bra)` (ket slow-varying), SVD-splits across `(env_left, D_ket)|(D_bra, env_right)` to `chi_I`, and assembles a `SplitCTMTensorEnv` with the exact labels listed in Background.

- [ ] **Step 2: Write the round-trip test** in `tests/test_split_ctm_doublelayer_projector.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard
from tests._split_ctm_oracle import make_site, heisenberg_gate, fused_env_to_split

pytestmark = pytest.mark.core


@pytest.mark.parametrize("D", [2, 3])
def test_fused_to_split_roundtrip(D):
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=8, max_iter=200, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = fused_env_to_split(fused_env, D, chi_I=8 * D)  # lossless
    rt_env = _split_env_to_tensor_standard(split_env)
    E_rt = float(compute_energy_ctm_tensor(A, rt_env, gate))
    np.testing.assert_allclose(E_rt, E_fused, atol=1e-8)
```

- [ ] **Step 3: Run it.** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k roundtrip -v` → PASS (this is proven spike code; if `tests` isn't importable as a package, add `tests/__init__.py` or use a `conftest.py` sys.path shim consistent with the repo).

- [ ] **Step 4: Commit.**
```bash
git add tests/_split_ctm_oracle.py tests/test_split_ctm_doublelayer_projector.py
git commit -m "test(#463): split-CTM oracle helper + fused->split round-trip proof"
```

---

## Task 2: Load-bearing parity test (drives the whole fix)

**Files:**
- Test: `tests/test_split_ctm_doublelayer_projector.py`

- [ ] **Step 1: Write the parity test** (it will FAIL until the moves are fixed):

```python
from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor


@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 6)])
def test_split_matches_fused_lossless_chi_I(D, chi):
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=chi, max_iter=300, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = ctm_split_tensor(A, chi=chi, chi_I=chi * D, max_iter=300, conv_tol=1e-12)
    E_split = float(compute_energy_split_ctm_tensor(A, split_env, gate))
    np.testing.assert_allclose(E_split, E_fused, atol=1e-8)
```

- [ ] **Step 2: Run it; confirm it FAILS now.** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k matches_fused -v` → FAIL (split is ~40% off). This documents the bug and is the gate for Tasks 3–5.

- [ ] **Step 3: Mark xfail temporarily** so the suite is green between tasks (remove in Task 5):
```python
@pytest.mark.xfail(reason="#463: split moves use per-layer projector; fixed in Task 5", strict=True)
@pytest.mark.parametrize(...)  # keep params
def test_split_matches_fused_lossless_chi_I(D, chi): ...
```

- [ ] **Step 4: Commit.**
```bash
git add tests/test_split_ctm_doublelayer_projector.py
git commit -m "test(#463): xfail parity test split==fused at lossless chi_I (drives fix)"
```

---

## Task 3: Double-layer corner-pair projector + factorization helpers

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py` (add helpers; add to `__all__`)
- Test: `tests/test_split_ctm_doublelayer_projector.py`

- [ ] **Step 1: Write the factorization unit test:**

```python
def test_factorize_projector_reconstructs():
    # P over (env, ketD, braD) -> chi factorizes exactly into P_first . P_second
    import jax.numpy as jnp
    from tenax.algorithms._split_ctm_tensor_moves import _factorize_projector
    from tenax.core.tensor import DenseTensor
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    sym = U1Symmetry()
    env, Dk, Db, chi = 4, 2, 2, 5
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (env, Dk, Db, chi))
    z = lambda n: __import__("numpy").zeros(n, dtype="int32")
    idx = [TensorIndex.from_charges(sym, z(env), FlowDirection.IN, label="env"),
           TensorIndex.from_charges(sym, z(Dk), FlowDirection.IN, label="ketD"),
           TensorIndex.from_charges(sym, z(Db), FlowDirection.IN, label="braD"),
           TensorIndex.from_charges(sym, z(chi), FlowDirection.OUT, label="chi_new")]
    P = DenseTensor(data, idx)
    P_first, P_second, m = _factorize_projector(P, "env", "ketD", "braD", "chi_new")
    # contract P_first . P_second over the factorization bond -> reconstruct P
    from tenax.contraction.contractor import contract
    P_rec = contract(P_first, P_second)
    # compare dense values up to leg order
    a = np.asarray(P.todense()); b = np.asarray(P_rec.transpose(tuple(P_rec.labels().index(l) for l in ["env","ketD","braD","chi_new"])).todense())
    np.testing.assert_allclose(b, a, atol=1e-10)
```

- [ ] **Step 2: Run → FAIL** (`_factorize_projector` undefined). `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k factorize -v`.

- [ ] **Step 3: Implement the helpers** in `_split_ctm_tensor_moves.py`:

```python
def _doublelayer_grown_corner(C, T_ket, T_bra, c_relabel, ket_I, bra_I, fuse_labels):
    """Grow a corner with BOTH ket and bra edges, joined over the interlayer.

    Mirrors the fused move's grown corner but keeps it as a double layer:
    fused leg = (env, u_ket, u_bra) of dim chi*D^2; the remaining leg is the
    next-corner env bond. Returns (C_grown_fused, remaining_label).
    """
    C_r = C.relabel(*c_relabel)               # align bond label to the ket edge
    Cg = contract(C_r, T_ket)                 # (env, u_ket, ket_I)
    Cg = contract(Cg.relabel(ket_I, bra_I), T_bra)  # (env, u_ket, u_bra, bra_r)
    labels = Cg.labels()
    # fuse the three to-truncate legs into 'fused' (env first, then u_ket, u_bra)
    Cg = fuse_indices(Cg, labels.index(fuse_labels[0]), labels.index(fuse_labels[1]),
                      "fused", FlowDirection.IN)
    labels = Cg.labels()
    Cg = fuse_indices(Cg, labels.index("fused"), labels.index(fuse_labels[2]),
                      "fused", FlowDirection.IN)
    remaining = [l for l in Cg.labels() if l != "fused"][0]
    return Cg, remaining


def _factorize_projector(P, env_label, ketD_label, braD_label, chi_label):
    """Factorize a projector P[(env,ketD),(braD,chi)] -> P_first . P_second.

    SVD across (env, ketD) | (braD, chi); factorization bond m <= env*ketD.
    No truncation (exact rewrite). Returns (P_first, P_second, m).
    P_first: (env, ketD, _fac), P_second: (_fac, braD, chi).
    """
    U, s, Vh, _ = tensor_svd(
        P,
        left_labels=[env_label, ketD_label],
        right_labels=[braD_label, chi_label],
        new_bond_label="_fac",
        max_singular_values=None,
    )
    P_first, P_second = absorb_sqrt_singular_values(U, s, Vh, "_fac")
    m = s.shape[0]
    return P_first, P_second, m
```

> `fuse_indices`, `absorb_sqrt_singular_values`, `tensor_svd`, `contract` are already imported at the top of `_split_ctm_tensor_moves.py`. Add `_doublelayer_grown_corner` and `_factorize_projector` to `__all__`.

- [ ] **Step 4: Run → PASS.** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k factorize -v`.

- [ ] **Step 5: Commit.**
```bash
git add src/tenax/algorithms/_split_ctm_tensor_moves.py tests/test_split_ctm_doublelayer_projector.py
git commit -m "feat(#463): double-layer grown-corner + projector factorization helpers"
```

---

## Task 4: Rewrite the four moves — double-layer projector + corners (closed edge)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py` (the four `_split_ctm_move_*` functions)
- Test: drives Task 2's parity test (keep xfail until Task 5 if edge not yet correct; but corners+projector should already make energy correct if the closed edge path is used)

This task replaces **Phase A** of each move and uses the **closed** edge path (correctness-first). The four moves are rotations; do `left` first, validate, then mirror.

- [ ] **Step 1: Rewrite `_split_ctm_move_left` Phase A + corners.** Replace the per-layer projector block (lines ~630–672) with:

```python
    base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None
    # Double-layer grown corners (C1 with T1_ket+T1_bra; C4 with T3_ket+T3_bra)
    C1g, c1_rem = _doublelayer_grown_corner(
        env.C1, env.T1_ket, env.T1_bra, ("c1_r", "t1k_l"), "t1k_I", "t1b_I",
        ("c1_d", "u_ket", "u_bra"))
    C4g, c4_rem = _doublelayer_grown_corner(
        env.C4, env.T3_ket, env.T3_bra, ("c4_r", "t3k_r"), "t3k_I", "t3b_I",
        ("c4_u", "d_ket", "d_bra"))
    # Correct double-layer Fishman pair (P_1 for C1 side, P_2 for C4 side)
    P_1, P_2, _eps = _compute_projector_tensor(C1g, C4g, chi, base_charges=base_charges)
    # New corners: apply P_1 to C1g, P_2 to C4g (cheap; corner is chi^2 D^2)
    C1_new = _apply_projector(P_1, C1g).relabels({"chi_new": "c1_d", c1_rem: "c1_r"})
    C4_new = _apply_projector(P_2, C4g).relabels({"chi_new": "c4_r", c4_rem: "c4_u"})
    C1_new = _ensure_corner_flows(C1_new, "C1"); C4_new = _ensure_corner_flows(C4_new, "C4")
    C1_new, _ = max_abs_normalize(C1_new); C4_new, _ = max_abs_normalize(C4_new)
```

> Verify the exact `relabel` source labels against the actual grown-corner remaining labels (`c1_rem`, `c4_rem`) by printing `.labels()` once during implementation; the spike's label conventions (`t1b_r`, `t3b_l`) are the expected remaining bonds.

- [ ] **Step 2: Replace the edge step (Phase B/C) with the CLOSED path using P_1 (left end) and P_2 (right end).** Factorize each projector and apply to the respective edge ends. Use the existing closed builder + a projector application that takes **separate** left/right pairs:

```python
    P1f, P1s, _ = _factorize_projector(P_1, "c1_d", "u_ket", "u_bra", "chi_new")  # adapt env/ketD labels per fused parents
    P2f, P2s, _ = _factorize_projector(P_2, "c4_u", "d_ket", "d_bra", "chi_new")
    Tg = _grow_edge_no_double_layer(env.T4_ket, env.T4_bra, A, A_bar, "l",
            "t4k_I", "t4b_I",
            ("t4k_d", "u", "U", "r", "R", "t4b_u", "d", "D"))
    # Apply P1 (factorized) to the C1/left end, P2 (factorized) to the C4/right end:
    T4g = _project_grown_edge_tensor_lr(Tg, P1f, P1s, P2f, P2s,
            left_fuse=("t4k_d", "u", "U"), right_fuse=("d", "t4b_u", "D"))
    T4_ket_new, T4_bra_new = _svd_split_edge_tensor(T4g,
            left_labels=["left_chi", "r"], right_labels=["R", "right_chi"], chi_I=chi_I,
            ket_relabels={"left_chi": "t4k_d", "r": "l_ket", "_svd_bond": "t4k_I"},
            bra_relabels={"_svd_bond": "t4b_I", "R": "l_bra", "right_chi": "t4b_u"},
            base_charges=base_charges)
    T4_ket_new, T4_bra_new = _ensure_edge_flows(T4_ket_new, T4_bra_new, "T4")
    return env._replace(C1=C1_new, C4=C4_new, T4_ket=T4_ket_new, T4_bra=T4_bra_new)
```

- [ ] **Step 3: Add `_project_grown_edge_tensor_lr`** — a variant of the existing `_project_grown_edge_tensor` that applies a **left** sequential pair `(P_left_first, P_left_second)` to the left side and a **right** pair `(P_right_first, P_right_second)` to the right side (the current function uses one pair for both). Copy `_project_grown_edge_tensor` (lines 384–442) and parameterize the right side to use the right pair. Concretely: left side keeps `P_first.bar()`/`P_second.bar()` = `P1f.bar()`/`P1s.bar()`; right side uses `P2f.bar()`/`P2s.bar()`.

- [ ] **Step 4: Run the parity test (still xfail-marked).** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k matches_fused -v --runxfail`. Iterate the leg labels/flows until energy matches fused to 1e-8 for `(D=2, chi=4)` at least. The oracle (Task 2) is the gate — adjust relabels/fuse order, not the math.

- [ ] **Step 5: Mirror to `_split_ctm_move_right/top/bottom`.** Apply the same transformation (double-layer grown corners, `(P_1,P_2)` to the two sides, factorized closed-edge `_lr` application). Reuse the per-move `left_fuse/right_fuse`/relabel tuples already present in each existing move (they encode the rotation).

- [ ] **Step 6: Run all parity params with `--runxfail`** → all should now pass. Commit.
```bash
git add src/tenax/algorithms/_split_ctm_tensor_moves.py
git commit -m "feat(#463): double-layer corner-pair projector in split moves (closed edge)"
```

---

## Task 5: Switch edge to the bounded path; un-xfail parity

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_moves.py`
- Test: `tests/test_split_ctm_doublelayer_projector.py`

- [ ] **Step 1: Add a `bounded==closed` test:**

```python
@pytest.mark.parametrize("D,chi", [(2, 4), (3, 6)])
def test_split_bounded_equals_closed(D, chi):
    # one sweep via bounded vs closed edge path must agree to 1e-10
    from tenax.algorithms._split_ctm_tensor_init import initialize_split_ctm_tensor_env
    from tenax.algorithms._split_ctm_tensor_convergence import _split_ctm_tensor_sweep
    A = make_site(D, 2, seed=7)
    env0 = initialize_split_ctm_tensor_env(A, chi, chi * D)
    # closed (env flag off) vs bounded (default) — see Step 2 for the flag
    import tenax.algorithms._split_ctm_tensor_moves as M
    M._FORCE_CLOSED_EDGE = True
    e_closed = _split_ctm_tensor_sweep(env0, A, chi, chi * D, True)
    M._FORCE_CLOSED_EDGE = False
    e_bounded = _split_ctm_tensor_sweep(env0, A, chi, chi * D, True)
    import numpy as np
    for c in ("C1", "T4_ket", "T4_bra"):
        a = np.asarray(getattr(e_closed, c).todense()); b = np.asarray(getattr(e_bounded, c).todense())
        np.testing.assert_allclose(b, a, atol=1e-10)
```

- [ ] **Step 2: Implement the bounded edge path** behind a module flag `_FORCE_CLOSED_EDGE = False`. In each move, route the edge through `_grow_and_project_bounded` (already χ²·D⁴-bounded) using the factorized `(P1f,P1s)` / `(P2f,P2s)` for left/right ends. `_grow_and_project_bounded` currently takes a single `(P_first, P_second)`; add a `_grow_and_project_bounded_lr(..., P_left_first, P_left_second, P_right_first, P_right_second, ...)` that precombines the left pair for the left side and the right pair for the right side (mirror `_precombine_projector_pair` usage with the respective pairs). When `_FORCE_CLOSED_EDGE`, fall back to the Task-4 closed `_lr` path.

- [ ] **Step 3: Run** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -k "bounded_equals_closed" -v` → PASS (1e-10).

- [ ] **Step 4: Remove the `xfail` from `test_split_matches_fused_lossless_chi_I`** and run the full file: `uv run pytest tests/test_split_ctm_doublelayer_projector.py -v` → all PASS (bounded path now default).

- [ ] **Step 5: Commit.**
```bash
git add src/tenax/algorithms/_split_ctm_tensor_moves.py tests/test_split_ctm_doublelayer_projector.py
git commit -m "feat(#463): memory-bounded chi^2*D^4 edge application + un-xfail parity"
```

---

## Task 6: Convergence-criterion guard

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_convergence.py` (`ctm_split_tensor`, ~line 87)
- Test: `tests/test_split_ctm_doublelayer_projector.py`

- [ ] **Step 1: Write the convergence-honesty test:**

```python
@pytest.mark.parametrize("D,chi", [(2, 4)])
def test_split_energy_sweepcount_stable(D, chi):
    A = make_site(D, 2, seed=7); gate = heisenberg_gate()
    e1 = compute_energy_split_ctm_tensor(A, ctm_split_tensor(A, chi=chi, chi_I=chi*D, max_iter=60, conv_tol=1e-12), gate)
    e2 = compute_energy_split_ctm_tensor(A, ctm_split_tensor(A, chi=chi, chi_I=chi*D, max_iter=200, conv_tol=1e-12), gate)
    np.testing.assert_allclose(float(e1), float(e2), atol=1e-8)
```

- [ ] **Step 2: Run → it should already PASS** if Task 5 fixed the fixed point; if it FAILS (transient-plateau early break), add a `min_iter` floor: in `ctm_split_tensor`'s loop, only allow the `conv_tol` early break once `iteration >= min_iter` (add `min_iter: int = 4` parameter, mirroring the fused loop). Re-run → PASS.

- [ ] **Step 3: Commit.**
```bash
git add src/tenax/algorithms/_split_ctm_tensor_convergence.py tests/test_split_ctm_doublelayer_projector.py
git commit -m "fix(#463): min_iter guard so split CTM convergence tracks the true fixed point"
```

---

## Task 7: Production-χ_I convergence + regression

**Files:**
- Test: `tests/test_split_ctm_doublelayer_projector.py`
- Verification only otherwise.

- [ ] **Step 1: Add a production-χ_I convergence test** (χ_I=χ; not exact parity, just physical + converging):

```python
def test_split_chi_I_equals_chi_physical():
    D = 2; A = evolve_physical_site(make_site(D, 2, seed=7), heisenberg_gate(), D)
    gate = heisenberg_gate()
    energies = []
    for chi in (4, 8, 12):
        env = ctm_split_tensor(A, chi=chi, chi_I=chi, max_iter=300, conv_tol=1e-12)
        e = float(compute_energy_split_ctm_tensor(A, env, gate))
        assert abs(e) <= 0.75 + 1e-6, f"unphysical bond energy {e}"
        energies.append(e)
    # converging: successive differences shrink
    d1, d2 = abs(energies[1]-energies[0]), abs(energies[2]-energies[1])
    assert d2 <= d1 + 1e-9
```

- [ ] **Step 2: Run** `uv run pytest tests/test_split_ctm_doublelayer_projector.py -v` → all PASS.

- [ ] **Step 3: Regression — existing split suite + core marker.**
Run: `uv run pytest tests/test_split_ctm_tensor.py -q` then `uv run pytest -m core -q`.
Expected: all pass. If `tests/test_split_ctm_tensor.py::*matches_shim` tests assert a now-changed (previously-wrong) split energy, update those expected values to the corrected (fused-matching) numbers and note why in the commit (they were self-consistency checks on a wrong env).

- [ ] **Step 4: Commit.**
```bash
git add tests/test_split_ctm_doublelayer_projector.py tests/test_split_ctm_tensor.py
git commit -m "test(#463): production chi_I convergence + regression for corrected split CTM"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** Component 1 (double-layer corners)→Task 3/4; Component 2 (reuse `_compute_projector_tensor`, biorthogonal P_1/P_2)→Task 4; Component 3 (factorize + bounded)→Tasks 3/5; Component 4 (min_iter guard)→Task 6; oracle §1 (exact parity)→Task 2/5; oracle §2 (production χ_I)→Task 7; oracle §3 (sweep-count stable)→Task 6; Test 4 (bounded==closed)→Task 5; no-regression→Task 7.
- **Name consistency:** `_doublelayer_grown_corner`, `_factorize_projector`, `_project_grown_edge_tensor_lr`, `_grow_and_project_bounded_lr`, `_FORCE_CLOSED_EDGE`, `fused_env_to_split` used consistently across tasks.
- **Altitude / honesty:** Tasks 4–5 carry the leg-wiring risk; they are gated by the exact-parity oracle (Task 2) and `bounded==closed` (Task 5) — the implementer iterates relabels/flows against these gates, not the math. The closed-edge-first sequencing isolates "projector correct" (Task 4) from "memory-bounded application correct" (Task 5). The corner application and helpers are fully concrete; the edge `_lr` variants are precise adaptations of named existing functions.
- **Deferred (not in plan):** half-system projectors, SymmetricTensor/fermionic, split AD-stability, `fuse_virtual_legs` wiring (resumes after Task 5 parity passes).
