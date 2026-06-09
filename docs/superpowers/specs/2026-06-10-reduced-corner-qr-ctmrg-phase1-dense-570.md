# Reduced-corner QR-CTMRG — Phase 1: dense single-site forward (reconstruction de-risk)

**Date:** 2026-06-10
**Issue:** #570 (CTM-AD compile wall) · **Umbrella:** #566
**Branch:** `feat/qr-projector-2x2-570` (continues; despite the name, this spec is the 1×1 reduced-corner work)
**Status:** DESIGN — awaiting build
**Supersedes the drop-in approach:** see `2026-06-09-570-qr-dropin-nogo-pivot-faithful.md` (drop-in NO-GO).

---

## Why this exists

The drop-in QR projector was NO-GO: PR #593 already banked the decomposition win, leaving the
drop-in at ~4.4% backward ops / ~1.4× capped per-projector compile (it kept the M′ truncation SVD).
The compile micro-spike showed a single symmetric SVD backward lowers to **2.2–2.5× the HLO** of a
QR backward — but that full win is only reachable by removing **all** decomposition SVDs, which only
the **faithful reduced-corner QR-CTMRG** (Yang/Zhang/Corboz, arXiv 2505.00494) does: a rank-χ
*reduced* corner + unpivoted QR makes the isometry `Q` the projector with **no truncation SVD**.

This is a multi-phase, research-grade build with a dominant **reconstruction risk** (the paper gives
no reduced-corner tensor indices). **Phase 1 de-risks the reconstruction in the cheapest, paper-
closest setting** — dense, single-site, forward-only — before any AD, symmetry, or multisite work.

## Phased roadmap (each phase = its own spec → plan)

| Phase | Scope | Why / status |
|---|---|---|
| **1 (this spec)** | Dense, single-site (`recipe="1x1"`), **forward-only** reduced-corner QR projector + warm-up. Validate energy vs eigh/SVD CTM on 2D Heisenberg D=2. | De-risk the reconstruction; resolve single-isometry-vs-pair as far as the 1×1 case needs. |
| 2 | Multisite reduced-corner QR **recipe** (single-isometry-per-cut, Corboz-style, arbitrary unit cells) **+ AD** via the existing explicit unroll (`gs_explicit_ad_steps` / `gs_explicit_ad_backward_steps`). Dense. | The first genuinely useful deliverable; multisite is what production uses. |
| 3 | **Block-sparse** (`SymmetricTensor`) reduced-corner QR. | **The real research risk AND where the compile win lives** — see below. |
| 4 | Scale: larger D, fermionic, multisite cells; the D=6–7 aspiration. | — |

## Critical strategic context — the original is DENSE-ONLY

Verified against arXiv 2505.00494v2: the paper uses **only C₄ᵥ spatial point-group symmetry on a
single real tensor** — **no U(1)/Z₂/SU(2)/charge-sector/block-sparse tensors anywhere**. The
unpivoted-QR rank-revealing property and the GPU speedup **rely on density**. Consequences:

1. **Phase 3 (block-sparse) leaves the validated literature.** The "reduced corner is already rank χ
   → unpivoted QR, no truncation" is a *dense* property. Block-sparse requires each **charge sector's**
   reduced corner to be already-rank-χ_sector, with the global χ distributed across sectors — a new
   question the paper never addresses.
2. **Phase 3 is exactly where Tenax's compile win must come from** (the wall is the *block-sparse*
   SVD-VJP). So the payoff lives in the novel, unvalidated part. The per-sector QR still emits the
   #566 per-sector structural ops; the bet is the measured 2.2–2.5×-per-decomposition × removing
   *all* SVDs — real but **contingent on the per-sector reduced-corner rank property holding**.
3. **Single-isometry vs projector-pair is tied to symmetry.** C₄ᵥ on one tensor makes the corner
   symmetric/Hermitian, plausibly *why* a single isometry suffices in the paper. General multisite /
   lower-symmetry cases may need a projector pair. Phase 1 resolves this only as far as the 1×1 case
   needs; Phase 2 confronts it for general cells.

**The compile win is a HYPOTHESIS to be tested at Phase 3, not a guarantee inherited from the paper.**
This spec deliberately de-risks the *correctness* foundation (reconstruction) first; the *payoff* is
gated separately at Phase 3.

## Scope (Phase 1, deliberately tight)

- **Dense only** (`DenseTensor`) — no `SymmetricTensor`.
- **Forward only** — no autodiff/gradients; projectors computed eagerly.
- **Single-site `recipe="1x1"`** — the single-isometry directional move the scheme belongs to.
- **Goal:** a dense reduced-corner QR projector that drives `recipe="1x1"` CTM to the **same energy**
  as the existing eigh/SVD `recipe="1x1"` CTM on 2D Heisenberg D=2.

## Existing code this slots into (verified 2026-06-09/10)

- 1×1 directional moves: `_ctm_tensor_move_left/right/top/bottom` (`_ctm_tensor_moves.py:659–873`)
  build enlarged corners (`C1g`, `C4g`) and call `_compute_projector_tensor` with `projector_method`.
  **No move rewrite needed** — the new projector returns an isometry used where eigh/SVD's output is.
- Projector dispatch: `_compute_projector_tensor` (`_ctm_projector.py`, ~line 600 / the method branch
  near 1026). `projector_method="qr"` there is currently an **eigh alias** (docstring lines 18–19).
- Warm-up: `qr_warmup_steps` (config `ipeps_config.py:78`, default 3) runs eigh sweeps then switches
  to `"qr"` (`_ctm_tensor_convergence.py:641`). Already wired; reuse as-is.
- Dense QR sign-fix pattern already exists at `_ctm_projector.py:1064–1071` (mirror it).

## Architecture / components

### C1. `_reduced_qr_projector(C_up, C_dn, chi, ...)` — dense, in `_ctm_projector.py`
- Consumes the **same enlarged corners** the existing 1×1 move builds; returns a dense isometry `P`
  (cut-bond → χ) used exactly where the eigh/SVD projector output is used today.
- Internals are the **reconstruction** (below). Includes a `diag(R)≥0` gauge fix on `Q`
  (mirror `_ctm_projector.py:1064–1071`) for iteration-to-iteration forward stability.

### C2. Dispatch
- `projector_method="qr"` → `_reduced_qr_projector` on the **dense 1×1 path** (replacing the eigh
  alias). Symmetric and 2×2 routes untouched in Phase 1. Update the docstring/changelog: `"qr"` on
  the 1×1 path now runs the real reduced-corner QR (behavior change for configs relying on
  `"qr"`==eigh).

### C3. Warm-up
- Reuse the existing `qr_warmup_steps` machinery unchanged (eigh grows χ, then QR takes over).

## The reconstruction (Phase 1's core, spike-driven)

The paper gives no indices, so **Step 1 of the build is a reconstruction spike**, not an assumption.

**Best-hypothesis starting point** (dense left move; others are mirror images):
- The standard 1×1 projector decomposes the *enlarged* corner (corner + edge + bulk-row growth),
  whose cut-bond is rank ~χD, and SVD/eigh-truncates it to χ.
- **Reduced-corner hypothesis:** build the corner with **one fewer bulk row** (corner + edges
  *without* the final D²-growth absorption on the cut side) so its cut-leg is already ≤ χ; take an
  **unpivoted QR** → `Q` (m×χ) is the isometry directly, no truncation. Renormalize with `Q`,
  folding `R` into the corner update (the paper's "T′, R, Q" contraction, cost χ³D²).

**The spike implements this and validates energy.** If the converged energy does not match the
eigh/SVD 1×1 CTM (and approach it as χ grows), the spike reports which variant *does*, or that the
reconstruction is ambiguous. **We do not ship a projector that fails to reproduce the energy.** The
single-isometry-vs-pair question is resolved here only to the extent the 1×1 case requires.

## Testing gate (inverted-parity — QR is a different scheme, so not byte-parity vs eigh/SVD)

Cheapest-first (TDD):

- **T1 (gates all). Reconstruction + energy.** Dense reduced-corner QR drives `recipe="1x1"` CTM on
  2D Heisenberg D=2 to within tol of the existing eigh/SVD 1×1 CTM energy, with the gap shrinking as
  χ grows (e.g. χ ∈ {8,16,24}). Tolerance loosened vs machine-eps (different fixed point); concrete
  bar set from the eigh/SVD reference spread (target ≲1e-3 at χ=8, tighter as χ grows). Reuse the
  existing dense 2D Heisenberg D=2 fixture from the iPEPS/CTM tests as the oracle.
- **T2. Isometry.** `Q† Q = I_χ` per cut on random enlarged corners.
- **T3. Convergence.** CTM converges (corner-spectrum stabilizes, no NaNs) with `qr_warmup_steps`
  warm-up across a χ sweep.
- **T4. Gauge stability.** `diag(R)≥0` fix keeps `Q` continuous across iterations (forward
  smoothness — needed for clean convergence even pre-AD).

**Acceptance:** T1 green (reconstruction reproduces the energy) + T2/T3/T4 green. Default stays
eigh/SVD; QR is opt-in via `projector_method="qr"` on the 1×1 path.

## Out of scope (Phase 1)

- AD / gradients (Phase 2).
- `SymmetricTensor` / block-sparse (Phase 3 — the compile-win phase, the real research risk).
- Multisite recipe / arbitrary unit cells (Phase 2).
- The general projector-pair-vs-single-isometry treatment (Phase 2; Phase 1 resolves only the 1×1 case).
- Larger D, fermionic, GPU tuning (Phase 4).

## References

- Yang, Zhang, Corboz — QR-CTMRG, arXiv:2505.00494v2 (dense, C₄ᵥ; reduced corner + unpivoted QR).
- `2026-06-09-570-qr-dropin-nogo-pivot-faithful.md` — drop-in NO-GO + the SVD/QR compile ratios.
- `examples/probe_svd_vs_qr_compile_570.py` — the 2.2–2.5×-per-decomposition compile ratio that
  motivates removing *all* SVDs (Phase 3).
- Codebase map: `_ctm_tensor_moves.py` (1×1 moves), `_ctm_projector.py` (`_compute_projector_tensor`
  dispatch + dense QR sign-fix), `_ctm_tensor_convergence.py:641` (warm-up), `ipeps_config.py:76,78`.
