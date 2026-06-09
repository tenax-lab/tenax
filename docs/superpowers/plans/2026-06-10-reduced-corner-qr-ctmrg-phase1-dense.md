# Reduced-corner QR-CTMRG — Phase 1 (dense 1×1 forward) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dense, forward-only reduced-corner QR projector on the single-site (`recipe="1x1"`) CTM path that drives the CTM to the same ground-state energy as the existing eigh/SVD projector on 2D Heisenberg D=2 — de-risking the reduced-corner *reconstruction* (the paper gives no indices) before any AD, symmetry, or multisite work.

**Architecture:** The reduced-corner reconstruction is unknown, so Task 1 is a **reconstruction spike** that discovers and energy-validates the construction; Tasks 2–6 productionize exactly what the spike validated, behind `projector_method="qr"` on the dense 1×1 path (replacing the current eigh-alias). No CTM move is rewritten — the new projector slots into the existing `_compute_projector_tensor` dispatcher, returning a single isometry `P` (with `P_1 = P_2 = P`, as the eigh/qr contract already specifies).

**Tech Stack:** JAX (`jnp.linalg.qr`), Tenax `DenseTensor` / `tenax.linalg`, the 1×1 CTM move (`_ctm_tensor_moves.py`), `_compute_projector_tensor` dispatch (`_ctm_projector.py:830`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md`

---

## Key existing facts (verified)

- The 1×1 left move builds enlarged corners `C1g = (fused, t1_r)` and `C4g = (fused, t3_l)` and calls `_compute_projector_tensor(C1g, C4g, chi, projector_method, base_charges, projector_backward)` (`_ctm_tensor_moves.py:694`). `fused` is the χ·D² grown bond to be projected to χ; `t1_r`/`t3_l` are the **already-χ** cut bonds.
- `_compute_projector_tensor` (`_ctm_projector.py:830`) returns `(P_1, P_2, eps_T)`; for `"eigh"`/`"qr"`, `P_1 = P_2 = P` (a single isometry with labels `(fused, chi_new)`, flows `(IN, OUT)`). The `"qr"` branch is at line 1026 and is currently an eigh-alias.
- **Reconstruction hypothesis (Task 1 starting point):** because `C1g`'s cut leg `t1_r` is already dimension χ, `QR(C1g)` over `(fused | t1_r)` yields `Q1 (fused, χ)` — a χ-isometry on the fused leg with no truncation. Same for `C4g → Q4`. The half-system projector then combines `Q1, Q4`; the genuinely-reduced variant needs **no large SVD** (at most a χ×χ overlap decomposition). Task 1 determines the exact combination by energy match.
- Oracle/fixtures: dense 2D Heisenberg D=2 CTM lives in `tests/test_ctm_tensor.py` / `tests/test_ctm_python_loop.py`; projector unit tests in `tests/test_ctm_projector.py`. Reuse these — do not hand-roll a model.

Run fast tests: `uv run pytest -m core`. Targeted: `uv run pytest tests/test_reduced_corner_qr.py -v`.

---

## File structure

- **Create** `examples/probe_reduced_corner_qr_reconstruction_570.py` — Task 1 spike.
- **Modify** `src/tenax/algorithms/_ctm_projector.py` — add `_reduced_qr_projector` (+ `_gauge_fix_qr_dense` helper if not reusing the inline one at 1064–1071); route the `"qr"` dense branch (line ~1026) to it.
- **Create** `tests/test_reduced_corner_qr.py` — isometry, gauge, energy, convergence tests.
- **Modify** `ipeps_config.py` (`projector_method` docstring at line 76), `README.md` / `docs/guide/algorithms/ctm.md` — Task 7.

---

## Task 1: SPIKE — reduced-corner reconstruction (gates everything)

Discover the dense reduced-corner QR construction that reproduces the eigh-CTM energy. No production code yet; the deliverable is the *validated construction*, recorded.

**Files:** Create `examples/probe_reduced_corner_qr_reconstruction_570.py`

- [ ] **Step 1: Build the probe harness around the existing 1×1 dense CTM**

Locate the dense 2D Heisenberg D=2 CTM driver used by `tests/test_ctm_tensor.py` / `tests/test_ctm_python_loop.py` (the `recipe="1x1"` path through `python_loop_ctm_converge` or equivalent) and the energy evaluation it uses. The probe must: build the D=2 Heisenberg `a` tensor + initial env, run the existing **eigh** 1×1 CTM to convergence at χ ∈ {8,16,24}, and record the reference energies `E_eigh(χ)`.

```python
"""SPIKE (#570 Phase 1): find the dense reduced-corner QR projector construction
that reproduces the eigh-CTM ground-state energy on 2D Heisenberg D=2.

Run: JAX_PLATFORMS=cpu uv run python examples/probe_reduced_corner_qr_reconstruction_570.py
"""
import jax
jax.config.update("jax_enable_x64", True)
# import the existing dense 1x1 Heisenberg-D2 CTM driver + energy fn from the
# test module / tenax public API (locate during implementation).
```

- [ ] **Step 2: Implement candidate reduced-corner QR projectors**

Each candidate is a function `proj(C1g, C4g, chi) -> P` returning a dense isometry with labels `(fused, chi_new)`. Implement, in order of fidelity to the paper:

- **Candidate A (primary — pure reduced corner):** `Q1, _ = qr(C1g over (fused | t1_r))`, `Q4, _ = qr(C4g over (fused | t3_l))`. Form the χ×χ overlap `O = Q1† Q4`; take its QR/polar factor to align the two halves; `P = Q1` (or the aligned combination) — the construction with **no large SVD and no truncation** (the cut legs are already χ). Apply a `diag(R)≥0` gauge fix to each `Q`.
- **Candidate B (reduced + small overlap SVD):** as A but truncate via a **χ×χ** SVD of `O = Q1† Q4` (small, not the χD² SVD) → `U_o, s_o, V_o`; `P = Q1 U_o`. This is the half-system QR projector; still no large SVD.
- **Candidate C (diagnostic only — drop-in style):** QR each corner, then SVD the R-overlap. Recorded only to confirm the energy target is reachable; **not** the faithful target (keeps a larger SVD).

- [ ] **Step 3: Run the energy sweep for each candidate**

For each candidate, run the 1×1 dense CTM with that projector (eigh warm-up via `qr_warmup_steps`, then the candidate) at χ ∈ {8,16,24}; record `E_cand(χ)` and `|E_cand − E_eigh|`.

Run: `JAX_PLATFORMS=cpu uv run python examples/probe_reduced_corner_qr_reconstruction_570.py`
Expected: a table `candidate | χ=8 | χ=16 | χ=24 | |ΔE|` and a printed VERDICT naming the candidate whose energy matches eigh within ~1e-3 at χ=8 and **shrinks** as χ grows.

- [ ] **Step 4: Record the validated construction**

Append a `## Phase 1 Task 1 result` section to `docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md`: the table, the chosen candidate (A preferred; B acceptable; if only C matches, **STOP and report** — that means the pure reduced corner doesn't reproduce the energy and the reconstruction needs rethinking before productionizing), and the exact contraction/QR steps of the winner.

- [ ] **Step 5: Commit**

```bash
git add examples/probe_reduced_corner_qr_reconstruction_570.py docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md
git commit -m "spike(#570): reduced-corner QR reconstruction — validated construction on Heisenberg D=2"
```

- [ ] **Step 6: Gate** — if the verdict is "only Candidate C (large SVD) matches" or "none match", do **not** proceed to Task 2. Report to the controller; the reconstruction (or the spec's Phase-1 premise) needs revisiting.

---

## Task 2: `_reduced_qr_projector` + isometry unit test

Productionize the Task-1 winner as a library function. (Code below shows Candidate A/B shape; replace the marked block with the *exact* validated construction from Task 1.)

**Files:** Modify `src/tenax/algorithms/_ctm_projector.py`; Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the failing isometry test**

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_projector import _reduced_qr_projector
# reuse the dense enlarged-corner fixture used by tests/test_ctm_projector.py
from tests.test_ctm_projector import _random_dense_enlarged_corners  # locate/adjust name


@pytest.mark.parametrize("chi", [4, 8])
def test_reduced_qr_projector_is_isometry(chi):
    C1g, C4g = _random_dense_enlarged_corners(seed=0, chi=chi, D=2)
    P = _reduced_qr_projector(C1g, C4g, chi)
    # P has labels (fused, chi_new); P† P = I_chi over the fused leg.
    from tenax.contraction.contractor import contract
    PtP = contract(P.bar(), P)  # contracts fused, leaves (chi_new, chi_new')
    dense = PtP.todense()
    np.testing.assert_allclose(dense, np.eye(dense.shape[0]), atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reduced_corner_qr.py::test_reduced_qr_projector_is_isometry -v`
Expected: FAIL — `ImportError: cannot import name '_reduced_qr_projector'`.

- [ ] **Step 3: Implement `_reduced_qr_projector`**

In `_ctm_projector.py`, add (using the validated construction from Task 1 in the marked block):

```python
def _reduced_qr_projector(C1g, C4g, chi):
    """Dense reduced-corner QR projector (single isometry) for the 1x1 CTM.

    QR-CTMRG (Yang/Zhang/Corboz, arXiv:2505.00494): the cut legs of the enlarged
    corners are already dimension chi, so an unpivoted QR yields a chi-isometry on
    the fused leg with no truncation SVD.  Returns P with labels (fused, chi_new),
    flows (IN, OUT); the caller uses P_1 = P_2 = P (single isometry, like eigh).

    The exact construction is the one validated in Task 1 of
    docs/superpowers/plans/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense.md.
    """
    fused_dim = ...  # C1g fused-axis dim
    # === BEGIN validated construction (from Task 1 verdict) ===
    M1 = C1g._data  # (fused, t1_r) dense matrix
    M4 = C4g._data  # (fused, t3_l) dense matrix
    Q1, R1 = jnp.linalg.qr(M1)            # Q1: (fused, chi)
    Q1 = _gauge_fix_qr_dense(Q1, R1)
    Q4, R4 = jnp.linalg.qr(M4)
    Q4 = _gauge_fix_qr_dense(Q4, R4)
    # Candidate B (half-system overlap) — replace per Task 1 verdict:
    O = Q1.conj().T @ Q4                  # (chi, chi)
    U_o, _s_o, _Vh_o = jnp.linalg.svd(O)  # small chi x chi
    k = min(chi, U_o.shape[1])
    P_dense = Q1 @ U_o[:, :k]             # (fused, k)
    # === END validated construction ===
    fused_idx = C1g.indices[C1g.labels().index("fused")]
    chi_new_idx = _make_chi_new_index(fused_idx, k)  # reuse helper used by eigh path
    return _wrap_dense_projector(P_dense, fused_idx, chi_new_idx)
```

Reuse the dense-wrap helpers the eigh/qr branch already uses (`_make_chi_new_index`, `_wrap_dense_projector` — grep them in `_ctm_projector.py`). If `_make_chi_new_index` needs `base_charges`, pass `None` (dense Phase 1).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_reduced_corner_qr.py::test_reduced_qr_projector_is_isometry -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_projector.py tests/test_reduced_corner_qr.py
git commit -m "feat(#570): dense _reduced_qr_projector (reduced-corner QR isometry) + isometry test"
```

---

## Task 3: `diag(R)≥0` gauge fix + smoothness test

The dense QR sign convention must be fixed so the forward is continuous across CTM iterations (matters for convergence even pre-AD).

**Files:** Modify `src/tenax/algorithms/_ctm_projector.py`; Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the failing gauge test**

```python
from tenax.algorithms._ctm_projector import _gauge_fix_qr_dense

def test_gauge_fix_qr_dense_makes_diag_R_nonneg_and_preserves_QR():
    key = jax.random.PRNGKey(3)
    M = jax.random.normal(key, (12, 6))
    Q, R = jnp.linalg.qr(M)
    Qf = _gauge_fix_qr_dense(Q, R)
    # The phase applied to Q is the conjugate of the phase applied to R's diagonal;
    # Q_fixed still spans the same column space and Q_fixed has orthonormal columns.
    np.testing.assert_allclose(Qf.conj().T @ Qf, np.eye(Qf.shape[1]), atol=1e-10)
    # Re-deriving R' = Q_fixed† M has real, non-negative diagonal.
    Rp = Qf.conj().T @ M
    d = jnp.diag(Rp)
    assert jnp.all(jnp.real(d) >= -1e-10)
    assert jnp.allclose(jnp.imag(d), 0.0, atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k gauge_fix_qr_dense -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `_gauge_fix_qr_dense`** (mirror `_ctm_projector.py:1064–1071`)

```python
def _gauge_fix_qr_dense(Q, R):
    """Rephase Q's columns so diag(R) is real-nonnegative (zero-diagonal -> phase 1).
    Q @ R is invariant under the simultaneous (Q, R) rephasing, so the forward output
    is unchanged but the gauge is continuous across CTM iterations."""
    diag_R = jnp.diag(R)
    abs_diag = jnp.abs(diag_R)
    phase = jnp.where(
        abs_diag > 0, diag_R / jnp.where(abs_diag > 0, abs_diag, 1.0), jnp.ones_like(diag_R)
    ).astype(R.dtype)
    return Q * phase[None, :]
```

If Task 2 already inlined this, extract it here and have `_reduced_qr_projector` call it.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k gauge_fix_qr_dense -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_projector.py tests/test_reduced_corner_qr.py
git commit -m "feat(#570): diag(R)>=0 gauge fix for dense reduced-corner QR + test"
```

---

## Task 4: Dispatch `projector_method="qr"` → reduced-corner QR (dense path)

Replace the eigh-alias `"qr"` branch (dense, non-tracer) with `_reduced_qr_projector`.

**Files:** Modify `src/tenax/algorithms/_ctm_projector.py:1026` (qr branch); Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the failing dispatch test**

```python
def test_compute_projector_tensor_qr_routes_to_reduced_qr():
    from tenax.algorithms._ctm_projector import _compute_projector_tensor
    C1g, C4g = _random_dense_enlarged_corners(seed=1, chi=6, D=2)
    P1, P2, _eps = _compute_projector_tensor(C1g, C4g, 6, "qr", None, "auto")
    # eigh/qr contract: P_1 is P_2 (single isometry) and P_1† P_1 = I.
    assert P1 is P2 or np.allclose(P1.todense(), P2.todense())
    from tenax.contraction.contractor import contract
    PtP = contract(P1.bar(), P1).todense()
    np.testing.assert_allclose(PtP, np.eye(PtP.shape[0]), atol=1e-9)
    # And it must NOT equal the eigh projector byte-for-byte (different scheme).
    P1_eigh, _, _ = _compute_projector_tensor(C1g, C4g, 6, "eigh", None, "auto")
    assert not np.allclose(P1.todense(), P1_eigh.todense(), atol=1e-7)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k routes_to_reduced_qr -v`
Expected: FAIL — current `"qr"` returns the eigh projector, so the `not allclose(..eigh..)` assertion fails.

- [ ] **Step 3: Implement the dispatch**

In the `if projector_method == "qr":` block (`_ctm_projector.py:1026`), for the **dense** path (DenseTensor inputs, or the SymmetricTensor path is out of Phase-1 scope), call `_reduced_qr_projector(C1g, C4g, chi)` and `return P, P, jnp.asarray(0.0)`. Leave the SymmetricTensor non-tracer branch (`_qr_projector_symmetric`, line 1033) untouched for Phase 1 — add a comment that Phase 3 replaces it with the block-sparse reduced-corner QR. Update the `_compute_projector_tensor` docstring (lines 859/876) to note `"qr"` is now the reduced-corner QR isometry on the dense path.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k routes_to_reduced_qr -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_projector.py tests/test_reduced_corner_qr.py
git commit -m "feat(#570): route projector_method='qr' to dense reduced-corner QR (1x1 path)"
```

---

## Task 5: Energy agreement on 2D Heisenberg D=2 (the physics gate, T1)

**Files:** Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the energy-agreement test**

```python
@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [8, 16])
def test_reduced_qr_energy_matches_eigh_heisenberg_D2(chi):
    """Dense 1x1 CTM ground-state energy agrees between projector_method eigh and qr,
    with the gap shrinking as chi grows (different scheme, same physics)."""
    e_eigh = _heisenberg_D2_ctm_energy_1x1(chi=chi, projector_method="eigh")
    e_qr = _heisenberg_D2_ctm_energy_1x1(chi=chi, projector_method="qr")
    assert abs(e_qr - e_eigh) < 1e-3  # loosened vs eps; tighten at larger chi
```

Implement `_heisenberg_D2_ctm_energy_1x1` by reusing the existing dense 1×1 Heisenberg-D2 CTM driver + energy evaluation from `tests/test_ctm_tensor.py` / `tests/test_ctm_python_loop.py` (import the fixture; do not hand-roll the model or energy). Add an assertion that the χ=16 gap ≤ the χ=8 gap.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k energy_matches_eigh -v`
Expected: PASS. (If it fails, the productionized construction diverges from the spike — re-check Task 2's construction against the Task 1 verdict; do not loosen the tolerance to force a pass.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reduced_corner_qr.py
git commit -m "test(#570): reduced-corner QR vs eigh energy agreement (Heisenberg D2, 1x1)"
```

---

## Task 6: Convergence + warm-up (T3)

**Files:** Test `tests/test_reduced_corner_qr.py`

- [ ] **Step 1: Write the convergence test**

```python
@pytest.mark.algorithm
def test_reduced_qr_ctm_converges_with_warmup():
    """The 1x1 CTM with projector_method='qr' + qr_warmup_steps converges:
    no NaNs, and the energy stabilizes between successive sweeps."""
    energies = _heisenberg_D2_ctm_energy_trace_1x1(
        chi=8, projector_method="qr", qr_warmup_steps=3, n_sweeps=40
    )
    assert np.all(np.isfinite(energies))
    # Stabilizes: last-5 spread is small.
    assert np.ptp(energies[-5:]) < 1e-6
```

`_heisenberg_D2_ctm_energy_trace_1x1` returns the per-sweep energy list (reuse the driver, expose its per-sweep energies).

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_reduced_corner_qr.py -k converges_with_warmup -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reduced_corner_qr.py
git commit -m "test(#570): reduced-corner QR 1x1 CTM convergence with warm-up"
```

---

## Task 7: Docs + PR

- [ ] **Step 1: Update docs**

In `ipeps_config.py:76` (`projector_method` comment) and `docs/guide/algorithms/ctm.md`, document that `projector_method="qr"` on the dense 1×1 path now runs the reduced-corner QR-CTMRG isometry (Phase 1: dense, forward-only; SymmetricTensor/AD/multisite are later phases). Note it's opt-in, default stays `"svd"`.

- [ ] **Step 2: Run the core suite + the new file**

Run: `uv run pytest -m core tests/test_reduced_corner_qr.py tests/test_ctm_projector.py -v`
Expected: PASS.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin feat/qr-projector-2x2-570
gh pr create --title "feat(#570): reduced-corner QR-CTMRG Phase 1 — dense 1x1 projector" --body "$(cat <<'EOF'
Phase 1 of the faithful reduced-corner QR-CTMRG (pivot after the drop-in NO-GO):
a dense, forward-only reduced-corner QR projector on the single-site (recipe="1x1")
CTM path, opt-in via projector_method="qr" (previously an eigh-alias). Reduced-corner
construction validated by energy match vs eigh-CTM on 2D Heisenberg D=2 (reconstruction
spike). No AD / SymmetricTensor / multisite yet (later phases — see roadmap in the spec).

Spec: docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md
Plan: docs/superpowers/plans/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Merge after CI**

```bash
gh pr merge --squash --delete-branch --auto
```

---

## Self-review notes

- **Spec coverage:** C1 (`_reduced_qr_projector`)→Task 2; gauge fix→Task 3; C2 dispatch→Task 4; C3 warm-up→Task 6 (reuses existing knob, no new code); reconstruction→Task 1; T1 energy→Task 5; T2 isometry→Task 2; T3 convergence→Task 6; T4 gauge→Task 3; out-of-scope (AD/symmetric/multisite) explicitly deferred in Task 4's comment + docs. Covered.
- **The research-risk step (Task 1) is gated:** Steps 4/6 require the spike to name a faithful (no-large-SVD) winner, with an explicit STOP if only the diagnostic Candidate C matches — we don't productionize a construction that isn't actually the reduced-corner scheme.
- **Fixture reuse:** Tasks 1/2/5/6 reuse existing dense Heisenberg-D2 CTM + enlarged-corner fixtures from `tests/test_ctm_tensor.py` / `tests/test_ctm_python_loop.py` / `tests/test_ctm_projector.py`; the implementer locates exact names during execution (helper names in the test code above are placeholders to adjust to the real fixtures).
- **Consistency:** `_reduced_qr_projector(C1g, C4g, chi)` and `_gauge_fix_qr_dense(Q, R)` signatures are used identically in Tasks 2–5.
