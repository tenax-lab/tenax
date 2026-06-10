# Reduced-corner QR-CTMRG — Phase 2: dense multisite + implicit-diff AD

**Date:** 2026-06-10
**Issue:** #570 (CTM-AD compile wall) · **Umbrella:** #566
**Status:** DESIGN — awaiting build
**Builds on:** Phase 1 (`2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md`, merged in #595/#596)

---

## Goal

Make `projector_method="qr"` (the reduced-corner QR-CTMRG isometry from Phase 1) work
**end-to-end under the production implicit-diff AD** (`optimize_gs_ad`) on **dense multisite**
unit cells — i.e. a usable QR-CTMRG ground-state optimizer. Still dense; the block-sparse
compile win is Phase 3.

**AD-path decision:** target the **implicit-diff fixed-point adjoint** (the production default for
`optimize_gs_ad`), *not* the QR-CTMRG paper's explicit/truncated-backprop unroll. This makes QR
usable in the standard production AD optimizer. (Divergence from the paper's scheme is deliberate.)

## What Phase 1 already gives us (verified 2026-06-10)

- `recipe="1x1"` **already supports multisite unit cells**: `_ctm_tensor_sweep_multisite`
  (`_ctm_tensor_convergence.py:324`) iterates per site and calls the 1×1 moves, which call
  `_compute_projector_tensor` per bond. So the Phase 1 QR projector already runs multisite via
  `recipe="1x1"` + `projector_method="qr"`. Passing 2-site `recipe="1x1"` tests exist.
- **Single isometry suffices off C₄ᵥ / for asymmetric cuts.** The 1×1 path returns `P_1=P_2=P` for
  eigh/qr; bi-orthogonality `P_1† P_2 = I` holds by construction, and Candidate C reproduces the eigh
  density-matrix subspace (which already works multisite). So **no projector-pair code is needed** —
  Phase 2 settles this empirically (T3), it is not a new build.
- The implicit-diff path **already threads `projector_method`** through forward *and* the fixed-point
  backward (`_jit_apply_Jt` / `_jit_dE_denv`: `_ctm_energy_ad.py:608,914,1032,1068,1211,1244`) and
  **already has QR warm-up** (`projector_method=="qr" and qr_warmup_steps>0`, line 571).

## The gap

The implicit path's CTM sweep (`_make_jit_ctm_step` → `_ctm_tensor_sweep_multisite`) uses a hardcoded
**`recipe="2x2"`** (Fishman plaquette via `_compute_plaquette_projector_pair`), which **ignores
`projector_method`**. So `projector_method="qr"` is silently a no-op under implicit AD today. Phase 2
adds a `recipe` knob so `recipe="1x1"` routes the forward and adjoint sweeps through the QR-capable
moves. And the QR projector's backward (raw `jnp.linalg.qr`, unstable near rank-deficiency — Phase 1
spike) needs a `regularized_qr` custom-VJP.

## Scope

- **Dense only** (`DenseTensor`) — no `SymmetricTensor` (Phase 3, the compile-win phase).
- **Implicit-diff AD** via `ctm_energy_implicit` / `optimize_gs_ad`.
- **Multisite** dense unit cells (validate on a 2-site A≠B cell).
- Default behavior unchanged: `recipe="2x2"`, `projector_method="svd"`. QR+1×1 AD is opt-in.

## Architecture / components (smallest blast radius first)

### C1. Multisite forward validation (small — mostly tests)
`recipe="1x1"`+`"qr"` already runs multisite. Validate converged energy vs eigh on a 2-site (A≠B)
dense Heisenberg cell. Settles the single-isometry-vs-pair question. No new projector code.

### C2. `regularized_qr` custom-VJP — `src/tenax/algorithms/_ad_primitives.py`
A `jax.custom_vjp` mirroring `regularized_svd` (line 331). Forward = plain QR. Backward = the standard
thin-QR adjoint `M̄ = (Q̄ + Q·copyltu(Q̄ᴴQ − R̄Rᴴ))·R⁻ᴴ` with the **`R⁻¹` triangular solve
regularized** (floor / pseudo-inverse on `diag(R)`), so gradients stay finite through near-rank-
deficient bonds — analogous to `regularized_svd`'s `1/(sᵢ²−sⱼ²)` Lorentzian floor. The exact
regularization form is resolved by the C2 spike (T1). `_reduced_qr_projector` (`_ctm_projector.py`)
calls `regularized_qr` **only under tracing**; eager forward stays plain `jnp.linalg.qr`
(byte-identical to Phase 1).

### C3. `recipe` knob into the implicit-diff path
- Add `recipe` parameter to `_make_jit_ctm_step` (`_ctm_python_loop.py`), passed to
  `_ctm_tensor_sweep_multisite(recipe=...)`.
- Add `recipe` to `ctm_energy_implicit` (`_ctm_energy_ad.py:337`); thread to both the forward
  `jit_step` (line 558) and backward `jit_step_bwd` (line 974).
- Add a `gs_recipe: str = "2x2"` field to `iPEPSConfig` (CTMConfig has no recipe field), wired through
  `ipeps_ad_policy` like `gs_projector_method`. Default `"2x2"` preserves current behavior.
- `projector_method` / `qr_warmup_steps` already flow through — unchanged.

### C4. Default & API
Defaults unchanged. Opt-in: `gs_recipe="1x1"` + `gs_projector_method="qr"`.

## Testing gate (cheapest-first, TDD)

- **T1 (gates AD). `regularized_qr` spike.** `jax.test_util.check_grads` (reverse) on `regularized_qr`
  for well-conditioned, tall, and **near-rank-deficient** matrices — the last must pass where raw
  `jnp.linalg.qr` failed in Phase 1's spike (`examples/probe_qr_vjp_stability_570.py`). Red→green
  before wiring.
- **T2. `regularized_qr` unit test.** Backward matches plain QR's VJP on well-conditioned inputs (no
  regression); finite/correct on rank-deficient.
- **T3. Multisite forward energy (physics gate).** `recipe="1x1"`+`"qr"` converged energy matches
  eigh on a genuine **2-site (A≠B)** dense Heisenberg cell; `|ΔE| < tol` (≈1e-3, loosened vs eps);
  gap shrinks with χ. Reuse `ctm_tensor_2site` / `compute_energy_ctm_tensor_2site` and the
  `test_ctm_tensor.py` 2-site fixtures.
- **T4. Gradient parity (AD gate).** (a) finite-difference vs implicit-AD gradient on small dense
  Heisenberg D=2, `recipe="1x1"`+qr; (b) QR-AD gradient ≈ eigh-AD gradient within tol on the same
  state.
- **T5. Optimization + adjoint convergence.** A short `optimize_gs_ad` run with `recipe="1x1"`+qr
  decreases the energy, no NaNs, tracks the eigh/svd-AD result; assert the implicit adjoint either
  converges (ρ(Jᵀ)<1) or falls back to GMRES cleanly (`_ctm_energy_ad.py:1323`) — no divergence/NaN.
- **T6. Regression.** Existing implicit-AD tests (`"svd"`/`"eigh"`, `recipe="2x2"`) stay green; the new
  `recipe`/`gs_recipe` defaults preserve behavior.

**Acceptance:** T1 green + T2–T6 green. Defaults unchanged; QR+1×1 AD opt-in.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `regularized_qr` backward math wrong/unstable | High | T1 `check_grads` spike gates everything; mirror `regularized_svd` |
| Implicit adjoint doesn't contract with QR projector | Med | GMRES fallback already exists (`:1323`); T5 asserts converge-or-fallback |
| `recipe="1x1"` under implicit AD less-tested than 2×2 | Med | T4 gradient parity + T6 regression are the guards |
| Multisite single-isometry inaccurate off-C₄ᵥ | Low | T3 settles it empirically vs eigh; Phase-1 evidence says it holds |

## Out of scope (later phases)

- `SymmetricTensor` / block-sparse reduced-corner QR (Phase 3 — the compile-win phase, the real
  research risk; the per-sector reduced-corner rank property is unvalidated).
- Explicit-unroll / truncated-backprop AD (the paper's scheme; deliberately not chosen here).
- Larger D, fermionic, multisite > 2-site, GPU tuning (Phase 4).

## References

- Phase 1 spec/plan: `2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md` (+ `-phase1-dense.md`).
- `examples/probe_qr_vjp_stability_570.py` — Phase 1 spike showing raw QR VJP fails near rank-
  deficiency (the C2/T1 motivation).
- `regularized_svd` (`_ad_primitives.py:331`) — the custom-VJP template `regularized_qr` mirrors.
- Implicit-diff AD: `ctm_energy_implicit` (`_ctm_energy_ad.py:337`), `_jit_fused_fixed_point_bwd`
  (`:1082`), GMRES fallback (`:1323`).
- Multisite 1×1 dispatch: `_ctm_tensor_convergence.py:324`; 2-site fixtures `tests/test_ctm_tensor.py`.
