# 2-site split-CTM — SymmetricTensor support (Phase 3) — design

> Issue #463, Phase 3. Extends the coupled 2-site checkerboard split-CTM path
> (forward + AD) from DenseTensor to **bosonic SymmetricTensor** (trivial and
> nontrivial U(1)/Zₙ). Fermionic (FermionParity / FermionicU1 with Koszul signs)
> is **Phase 4** and explicitly out of scope here.

## Problem

Phases 0–2 landed the dense joint 2-site split-CTM forward (`ctm_split_tensor_2site`,
PR #684) and its explicit + implicit AD (`_split_ctm_energy_ad.py`, PR #685 on
`design/463-2site-split-ctm-ad`). Both are exercised **only** by DenseTensor tests.
The single-site split path is already fully SymmetricTensor-capable and tested
(`tests/test_split_ctm_tensor.py::TestSplitCTMSymmetric` — trivial-U(1)
energy-matches-dense to 1e-6, FermionicU1 charge preservation across sweeps), so a
symmetric 2-site path is the natural next increment and the design §9 Phase-3 item.

## Key finding (scope is narrower than design §9's wording)

Design §9 phrased Phase 3 as "per-block layout for split enlarged corners + the
four absorbs." A code audit shows the 2plaq path is **already substantially
polymorphic**, so the real work is one threaded argument plus test coverage, not a
per-block rewrite:

- The 2-site **enlarged-corner assembly** (`_build_split_enlarged_corner`,
  `_split_ctm_tensor_moves.py:103`) is pure label-based `contract` + invertible
  `_fuse_ket_bra` (which carries `FuseInfo` via `fuse_indices`). Feed SymmetricTensor
  `C`/`T`/`A` and you get a SymmetricTensor `Q` — no type branching needed.
- The **projector** already dispatches on type: `_compute_split_plaquette_projector_pair`
  → `_compute_2x2_projector` → `_compute_2x2_projector_symmetric` (block-sparse,
  tracer-safe, Issue #435). Comes for free once the split `Q`s are symmetric.
- **Projector application is already #605-safe.** The 2plaq absorbs use
  `_apply_proj_unfused` (splits the projector's fused leg before contracting — the
  YASTN-style fix for the D≥3 hard-fusion charge-conjugation clash). They do **not**
  touch the DenseTensor-only `_unfuse_projector_fused` reshape (that lives on the
  single-site move path only — a red herring for Phase 3).

**The actual gap:** `_split_ctm_sweep_multisite_2x2` hard-codes `base_charges=None`
(`_split_ctm_tensor_convergence.py:272`) and never forwards it into the four
`_split_ctm_absorb_*_2plaq` → their `_svd_split_edge_tensor` calls. Without it the
per-sector-truncation symmetric branch never engages. The single-site moves derive
it (`A.indices[0].charges if isinstance(A, SymmetricTensor) else None`, moves lines
1363/1451/1539/1627); the 2-site path must do the same.

## Requirements (pinned)

- Bosonic SymmetricTensor (trivial-U(1) and nontrivial-charge U(1)/Zₙ) works
  end-to-end on the 2-site checkerboard split forward **and** AD.
- No change to the DenseTensor path behaviour (byte-identical fast paths).
- Fermionic is **out of scope** (Phase 4): no `_env_is_fermionic` bifurcation is
  added here. Bosonic charges are reachable through the existing unfused-projector
  application without a Koszul-sign branch.
- `chi_I = chi` (lossless) as in the single-site and dense-2-site paths;
  smaller-`chi_I` truncation stays out of scope.

## Chosen approach

Thread `base_charges` through the 2-site sweep and the four absorbs so per-sector
SVD truncation engages, then mirror the single-site symmetric test methodology on
the 2-site path. The AD `custom_vjp` machinery is pure `jax.tree` over the env
pytree and symmetry-agnostic — it is **verified**, not rewritten.

## Design

### 1. Source changes

**`_split_ctm_tensor_convergence.py`** (`_split_ctm_sweep_multisite_2x2`, ~line 272):
- Derive per-sublattice `base_charges` from each site tensor:
  `base_charges = A.indices[0].charges if isinstance(A, SymmetricTensor) else None`
  (mirroring single-site moves). Pass it to
  `_compute_split_plaquette_projector_pair` (replacing the hard-coded `None`) **and**
  into each of the four `_split_ctm_absorb_*_2plaq` calls.

**`_split_ctm_tensor_moves.py`** (the four `_split_ctm_absorb_{bottom,left,right,top}_2plaq`):
- Add a `base_charges` parameter (default `None` to preserve dense call sites) and
  forward it to their internal `_svd_split_edge_tensor` calls so the symmetric
  per-sector-truncation branch (`_truncate_svd_per_sector` → `_select_bond_entries`)
  engages. The enlarged-corner assembly, projector dispatch, and
  `_apply_proj_unfused` application are unchanged — already polymorphic / #605-safe.

**`_split_ctm_energy_ad.py`** (2-site AD wrappers):
- **Verify only.** The multisite `custom_vjp` (`_split_ctm_converge_multisite` +
  fwd/bwd) and the Γ phase-fix (`_phase_fix_split_ctm_tensor`) are `jax.tree`-generic
  and should carry SymmetricTensor leaves unchanged. If a leaf-type assumption
  surfaces (e.g. a `.todense()` or dense-only reshape on the AD path), fix it
  minimally; do not restructure the VJP.

### 2. Testing strategy

New file `tests/test_split_ctm_2site_symmetric.py` (or an extension of
`tests/test_split_ctm_2site_ad.py`), mirroring `TestSplitCTMSymmetric`. All tight
parity tests use a **convergent, direction-dependent** state (A ≠ B), never random
tensors (the fused 2-site CTM oscillates on random input — see
`feedback_ctm_parity_needs_convergent_input`). Reuse the `_build_su_neel(D=...)` /
`_heisenberg_gate` helpers from `tests/test_split_ctm_2site.py`.

- **Tier 1/2 — energy parity (trivial-U(1) wrap of dense):** build the convergent
  dense Néel checkerboard `(A, B)`, wrap each as a trivial-U(1) SymmetricTensor with
  `SymmetricTensor.from_dense(A.todense(), A.indices)`, converge
  `ctm_split_tensor_2site` in both dense and symmetric form, assert
  `|E_sym − E_dense| < 1e-6`. Across D∈{2,3}, χ∈{4,8}. (Direct 2-site analogue of
  `test_symmetric_energy_matches_dense`; the D=3 case is also the §10 hard-fusion
  guard.)

- **Tier 3 — AD parity (trivial-U(1) wrap):** on the wrapped state,
  - symmetric `implicit == explicit` gradient: `cos > 1 − 1e-9`, `rel < 1e-6`;
  - symmetric-grad == dense-grad (same wrapped data) to `rel < 1e-6`.
  Use the **XXZ Δ=0.3 clean-regime** gate for the machine-exact assertion; keep a
  Heisenberg companion that only asserts the looser degenerate-SV floor
  (`rel ~ 5e-4`, see `feedback_ad_parity_degenerate_svd_floor`). Gradient taken
  w.r.t. sublattice A only for a clean scalar parity, mirroring the dense Task-2 test.

- **Structural smoke (nontrivial charges):** build a nontrivial-charge 2-site pair
  (FermionicU1 or Zₙ charges, `SymmetricTensor.random_normal`), run a few
  `_split_ctm_sweep_multisite_2x2` sweeps, assert all env tensors stay finite,
  remain SymmetricTensor, and keep ≥1 block (charge sectors survive). This exercises
  real block structure and per-sector truncation without needing a convergent charged
  oracle (mirrors `test_fermionic_u1_charges_preserved_across_sweeps`). No energy or
  variational assertion here — a genuinely-charged convergent state (symmetric SU)
  is out of scope; the trivial-U(1) parity carries the correctness weight.

### 3. Branch / PR

Stack on `design/463-2site-split-ctm-ad` (PR #685 still open; a fresh branch off
`main` would lack the Phase-2 AD code). Own commits, own green suite. Retarget to
`main` after #685 merges. Follows CLAUDE.md git workflow (PR, `--squash`,
`run-full-tests` label for the AD/slow tests).

## Risks & open questions

- **Hard fusion at D≥3 (design §10):** the split enlarged-corner assembly must stay
  clear of raw-reshape HARD fusion. Audit confirms it currently does (`_fuse_ket_bra`
  is invertible; projectors applied via `_apply_proj_unfused`). The Tier-1/2 D=3
  energy-parity test is the guard — if it fails at D=3 but passes at D=2, suspect a
  charge-conjugation clash in the corner seams.
- **Symmetric AD degenerate-SV floor:** the block-sparse SVD backward has a
  degenerate-singular-value floor at the Heisenberg point (same as dense/fused). The
  machine-exact gate is XXZ Δ=0.3; the Heisenberg companion asserts only the
  documented `~5e-4` floor. This is a known limitation, not a Phase-3 bug.
- **`base_charges` per-sublattice:** on the checkerboard A and B may carry different
  index charges; each absorb must receive the charges of the correct site tensor, not
  a single global value. Thread per-coord, matching how the single-site moves read
  `A.indices[0].charges` for the site being absorbed.
- **AD leaf-type surprise:** the expectation is zero AD-code change. If a
  dense-only assumption surfaces on the VJP path, fix it minimally and note it — do
  not restructure the Neumann backward.

## Acceptance

- [ ] Tier-1/2 energy parity: split-symmetric == split-dense to `< 1e-6` on a
  convergent Néel checkerboard (trivial-U(1) wrap), D∈{2,3}, χ∈{4,8}.
- [ ] Tier-3 AD parity: symmetric `implicit == explicit` and symmetric == dense
  gradient to `rel < 1e-6` (XXZ Δ=0.3 clean-regime gate; Heisenberg companion at the
  `~5e-4` floor).
- [ ] Structural smoke: nontrivial-charge 2-site sweep stays finite, SymmetricTensor,
  charge sectors preserved.
- [ ] DenseTensor 2-site path unchanged (existing dense suites green:
  `test_split_ctm_2site.py`, `test_split_ctm_2site_ad.py`).
- [ ] `-m core` green; slow AD tests green under the `run-full-tests` label.

## Out of scope

- Fermionic (FermionParity / FermionicU1 Koszul signs) — Phase 4.
- A convergent nontrivial-charge 2-site state (symmetric SU) — smoke only here.
- chi-auto-bump / chi-schedule / smaller-than-`chi` interlayer truncation on the
  split path (stays guarded off).
- General multisite (>2 sites).
