# Multisite-CTM RDM Helpers Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Localise and fix the multisite-CTM RDM bug that causes `c5_smoking_gun_bf_vs_ctm_at_optimum.py` to report a 0.87/site bias gap (E_BF=−0.044, E_CTM=−0.913) at D=4 χ=16 AD-optimum.

**Architecture:** Approach C with 2-hour budget on parallel surgical (A) and variPEPS cross-check (D) tracks. Reuse the existing smoking-gun probe as the end-to-end gate; supplement with per-RDM diagnostic and variPEPS reference run. Pivot to honeycomb-native CTM (M2b, PR #347) if budget elapses without localisation.

**Tech Stack:** JAX, Tenax (`_ctm_tensor_energy.py`, `_pess_multisite_energy.py`, `_ctm_tensor_convergence.py`), variPEPS 1.4.2 (`/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps`).

**Background (read first):** `docs/plans/2026-05-07-multisite-ctm-rdm-helpers-fix-design.md` and the smoking-gun probe at `.worktrees/multisite-kagome-pess/examples/c5_smoking_gun_bf_vs_ctm_at_optimum.py`. The smoking-gun probe is the only known reproducer; existing witnesses in `tests/test_pess_3site_multisite_rdm_invariants.py` PASS at D=2 χ=16 SU and do not gate the bug.

---

## Task 0: Shared scaffolding — save the AD-optimised state

**Files:**
- Create: `examples/dev/save_d4_ad_optimum.py` (uncommitted dev probe)

**Step 1: Write the script.**

```python
"""Run c5 to AD-optimum at D=4 chi=16, persist the IPESSState to disk
for reuse by Track A and Track D probes. Saves to logs/d4_ad_optimum.npz."""
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState, kagome_triangle_xxz_hamiltonian, pess_simple_update,
)
from tenax.algorithms.pess_optimize import optimize_pess_3site_multisite_ad

def main():
    D, chi, delta = 4, 16, 1.0
    H = kagome_triangle_xxz_hamiltonian(delta=delta, d=2)
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    state = pess_simple_update(state, H, dt_schedule=[(0.1, 100), (0.01, 100)], D_max=D)
    bond_gates = kagome_3site_bond_gates(delta=delta, d=2)
    cfg = CTMConfig(chi=chi, max_iter=80, min_iter=10, conv_tol=1e-7,
                    projector_method="svd", forward_gauge="phase",
                    ctm_conv_method="elementwise", gmres_tol=1e-4,
                    gmres_maxiter=80, gmres_restart=20, chi_ramp=None)
    state_opt, e_opt = optimize_pess_3site_multisite_ad(
        state, bond_gates, cfg, max_iter=15, verbose=True)
    Path("logs").mkdir(exist_ok=True)
    np.savez("logs/d4_ad_optimum.npz",
             R_a=state_opt.R_a, R_b=state_opt.R_b, R_c=state_opt.R_c,
             T_u=state_opt.T_u, T_d=state_opt.T_d,
             **{f"lambda_{i}": l for i, l in enumerate(state_opt.lambdas)},
             e_opt=e_opt)
    print(f"Saved AD-optimum: E_optimizer = {e_opt:.6f}")

if __name__ == "__main__":
    main()
```

**Step 2: Run on GPU.**

Run: `cd /home/yjkao/tenax && CUDA_VISIBLE_DEVICES=0 uv run python .claude/worktrees/fix-multisite-ctm-rdm-helpers/examples/dev/save_d4_ad_optimum.py`
Expected: prints `Saved AD-optimum: E_optimizer = -0.852811`. Wall ~12 min on RTX 4070.

**Step 3: Helper to reload state.** Add `_load_optimum()` to a shared probe utility for downstream tasks:

```python
def _load_d4_ad_optimum() -> "IPESSState":
    z = np.load("logs/d4_ad_optimum.npz")
    return IPESSState(
        R_a=jnp.asarray(z["R_a"]), R_b=jnp.asarray(z["R_b"]), R_c=jnp.asarray(z["R_c"]),
        T_u=jnp.asarray(z["T_u"]), T_d=jnp.asarray(z["T_d"]),
        lambdas=tuple(jnp.asarray(z[f"lambda_{i}"]) for i in range(6)),
    )
```

**Step 4: Commit.**

Don't commit the .npz (large) or the helper script. Add to `.gitignore` if needed.

---

## Track A — Surgical localisation (GPU 0)

### Task A1: Per-RDM diagnostic probe at the AD-optimum

**Files:**
- Create: `examples/dev/a1_per_rdm_diag.py`

**Step 1: Write the probe.**

For each of the 6 multisite-CTM RDMs (uv_h, uv_v, wu_h, wu_v, vw_row, vw_col), report:
- ‖ρ − ρ†‖_F (Hermiticity error)
- λ_min of the Hermitised RDM (PSD violation)
- |Tr(ρ) − 1| (trace violation)
- ‖ρ_CTM − ρ_BF‖_F (BF disagreement; ρ_BF from 3×3 torus contraction at the loaded state)
- ⟨ρ, H_pair⟩ (per-bond energy; spin-½ XXZ allowed range [−0.75, 0.25])

Reuse helpers from `examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py` (`_brute_force_rdms`, `_collect_ctm_rdms`).

**Step 2: Run.**

Run: `CUDA_VISIBLE_DEVICES=0 uv run python examples/dev/a1_per_rdm_diag.py | tee logs/a1_per_rdm_diag.log`
Expected: a 6-row table; the worst BF-disagreement bond is the prime suspect.

**Step 3: Marginalisation-consistency cross-check.**

For the v-w bonds, also compare `_rdm_3site_marginal_vw_row` vs `_rdm_3site_marginal_vw_col` for two different bond positions that physically share the same v-w pair. If they disagree on shared envs, at least one helper has a leg-mapping or partial-trace bug.

**Step 4: Record findings as tasks.**

Update `TaskList` with the prime suspect (e.g. "Investigate `_rdm_3site_marginal_vw_row`"). Do NOT proceed to A2 until A1's output is interpreted.

### Task A2: Investigate the worst helper

**Files:**
- Modify: probably one of `src/tenax/algorithms/_pess_multisite_energy.py:96` (vw_row), `:191` (vw_col), or `src/tenax/algorithms/_ctm_tensor_energy.py:531` (2x1), `:601` (1x2)

**Step 1: Read the suspect helper end-to-end.** Trace: input env legs → contraction order → output RDM legs. Verify against the docstring's claim about leg ordering.

**Step 2: Element-wise compare.** For the worst bond, print full 4×4 matrices of ρ_CTM and ρ_BF side-by-side. Look for: transposed indices, conjugation missing, off-by-one trace direction.

**Step 3: Hypothesise + write a failing test.**

For the suspected bug, write a focused test in `tests/test_pess_3site_multisite_rdm_helpers.py` that fails on the current code at D=4 χ=16 AD-optimum. The test loads the saved state and asserts the BF-CTM Frobenius gap on the suspected bond is below `1e-2`.

```python
def test_<bond>_matches_bf_at_d4_ad_optimum():
    state = _load_d4_ad_optimum()
    rho_bf = _brute_force_rdms(state)["<bond>"]
    rho_ctm = _collect_ctm_rdms(state, chi=16, max_iter=100, conv_tol=1e-9)["<bond>"]
    diff = float(jnp.linalg.norm(rho_bf - rho_ctm))
    assert diff < 1e-2, f"|ρ_CTM - ρ_BF|_F = {diff:.4e}"
```

**Step 4: Run the test to verify it fails.**

Run: `uv run pytest tests/test_pess_3site_multisite_rdm_helpers.py::test_<bond>_matches_bf -v`
Expected: FAIL with `|ρ_CTM - ρ_BF|_F` ≫ 1e-2.

**Step 5: Apply the minimal patch.** Change ONE thing at a time (axis swap, conjugation, contraction order). Re-run the focused test.

**Step 6: Commit per attempt.**

```bash
git add src/... tests/...
git commit -m "fix(multisite-ctm): <specific change> in <helper>"
```

If the test still fails, revert and try the next hypothesis. Don't pile patches.

### Task A3: Verify across all bonds + smoking-gun gate

**Step 1: Re-run A1 probe.**

Confirm all 6 RDMs now agree with BF to within 1e-2 at the loaded AD-optimum.

**Step 2: Re-run smoking-gun probe.**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python .worktrees/multisite-kagome-pess/examples/c5_smoking_gun_bf_vs_ctm_at_optimum.py | tee logs/smoking_gun_after_fix.log
```

Expected: per-site `|E_BF − E_CTM| < 0.05` at the new optimum.

**Step 3: Run the existing witness tests** to confirm no regression.

```bash
uv run pytest tests/test_pess_3site_multisite_rdm_invariants.py -v
```

Expected: still 3 passed (no regression at D=2 SU).

**Step 4: Commit + open PR.**

```bash
git push -u origin worktree-fix-multisite-ctm-rdm-helpers
gh pr create --title "fix(multisite-ctm): <specific helper> RDM at D≥2" --body "..."
```

---

## Track D — variPEPS cross-check (GPU 1)

### Task D1: Map IPESSState → variPEPS PEPS_Unit_Cell

**Files:**
- Create: `examples/dev/d1_varipeps_state_map.py`

**Step 1: Read variPEPS unit-cell construction.** Skim:
- `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/peps/unit_cell.py` — `PEPS_Unit_Cell.from_*` factory methods
- `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/peps/tensor.py` — leg ordering convention

**Step 2: Build a 1×3 unit cell from the kagome 3-site multisite encoding.**

`pess_to_kagome_3site_multisite` returns a dict `{"u", "v", "w"}` of rank-5 tensors with axes `(left, up, right, down, physical)` (verify this from `_make_multisite_indices`!). Map to variPEPS's leg convention.

```python
state = _load_d4_ad_optimum()
sites = pess_to_kagome_3site_multisite(state.R_a, state.R_b, state.R_c,
                                        state.T_u, state.T_d, state.lambdas)
# Build PEPS_Unit_Cell with sites["u"], sites["v"], sites["w"] in a 1×3 unit cell
```

**Step 3: Smoke-test by running variPEPS CTM at χ=16.**

If variPEPS rejects the state (leg-dim mismatch, neighbour map mismatch), the encoding maps badly and Track D is a dead end. Document and exit.

### Task D2: Extract variPEPS RDMs and compare to BF

**Files:**
- Create: `examples/dev/d2_varipeps_rdm_compare.py`

**Step 1: Use variPEPS expectation helpers.**

For 2-site bonds: `varipeps.expectation.two_sites.calc_two_sites_horizontal_single_gate` etc.
For 3-site triangle (kagome up-triangle): `calc_three_sites_triangle_without_top_left_single_gate` (or matching corner).

**Step 2: Compare to BF ground truth.**

For each of the 6 bonds: BF (3×3 torus exact), Tenax CTM (broken), variPEPS CTM (cross-check).

**Step 3: Verdict.**

- If `‖ρ_variPEPS − ρ_BF‖_F < 1e-2` for all 6 bonds: variPEPS is a working reference. Use as gold standard for Track A's investigation. Document.
- If variPEPS *also* disagrees with BF substantially: the multisite encoding *itself* may be wrong. Escalate (out of scope).

### Task D3: Decide on porting

**Step 1: If D2 passes, write a one-pager** in `docs/plans/2026-05-07-varipeps-port-feasibility.md` describing:
- Which variPEPS file/function gives the correct contraction
- Cost of porting to Tenax data structures
- License (variPEPS shows blank License; needs check before any code copy — runtime cross-check only is fine)

**Step 2: Don't port without explicit approval.** This is a cross-check, not a rewrite.

---

## Exit and Pivot

**At any point if 2 hours elapse:**

1. Save findings as a memory entry (per-RDM diagnostic table, variPEPS verdict).
2. Open an issue describing the localised bug or escalation reason.
3. Pivot to honeycomb-native CTM (M2b, PR #347 scaffolding); the smoking-gun probe is the same end-to-end gate there.

**On success (smoking-gun gap < 0.05 at D=4):**

1. Re-run all `pytest -m core` to confirm no regression.
2. Open PR with linked issue references (#391, #401, #402 + the C.3 sequence).
3. Update memory `project_c3_floor_breach_smoking_gun.md` with the fix.

---

## Reference cross-links

- Design: `docs/plans/2026-05-07-multisite-ctm-rdm-helpers-fix-design.md`
- Smoking gun probe: `.worktrees/multisite-kagome-pess/examples/c5_smoking_gun_bf_vs_ctm_at_optimum.py`
- Existing audit: `examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.{py,json}`
- Existing witnesses (currently green at D=2): `tests/test_pess_3site_multisite_rdm_invariants.py`
- variPEPS: `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/expectation/{two_sites,three_sites,helpers}.py`
- Memory: `project_c3_floor_breach_smoking_gun.md`, `project_kagome_3site_multisite_pivot.md`, `project_varipeps_2site_honeycomb_works.md`, `reference_varipeps_multisite_ad.md`
