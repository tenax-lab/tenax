# CTM Bug 3a — chi_init=1 in fixed-shape container (design)

**Date:** 2026-05-11
**Status:** approved, pending implementation plan
**Branch:** `fix/ctm-bug-3a-chi-init`
**Predecessors:** PR #422 (bugs 1, 2, 3b), PR #423 (`_flow_flip_no_conj` SymmetricTensor follow-up)
**Related memory:** `project_ctm_two_init_bugs_found.md`

## Problem

Bug 3a, diagnosed alongside bugs 1/2/3b but deferred from PR #422, is the
remaining cause of CTM convergence regressions on random complex iPEPS:

- Tenax's `initialize_ctm_tensor_env` packs the chi-target shape with a
  rank-D corner (`eye(min(chi, D²))` zero-padded to `chi×chi`) and a rank-min(chi, D)
  edge (δ\_{ket=bra} replicated across the first D chi slots).
- That occupies a D-dimensional rank in chi-space, which the diagnostic in
  `project_ctm_two_init_bugs_found.md` showed traps CTM at a paired-degenerate
  fixed point. variPEPS reproduces the same trap when forced to start at
  `chi_init=D`.
- variPEPS's default is `chi_init=1` (rank-1 corners and edges), which breaks
  the Z₂ from the start and converges to the physical fixed point.

After PR #422 (which fixes the δ\_{ket=bra} pattern, bar-conjugation, and
double-renormalize bugs), Tenax still settles at the symmetric paired-degenerate
fixed point on cold init. Bug 3a is the last remaining standard-CTM
convergence regression on generic complex iPEPS.

## Decision

Replace the rank-D init with a rank-1 init *inside the existing chi-target
container*. variPEPS-style chi_init=1, but stored in chi-target shape so JIT
signatures, AD path, and the convergence loop are all unchanged.

### Why not variable shapes (Option B)?

A faithful chi_init=1 → chi_target growth phase would change shapes between
absorptions (1 → D → D² → … → chi_target), producing new JIT traces per
shape signature. After PR #421's fused-backward and the warm-start cache
work, Tenax's hot path is heavily tuned around fixed JIT shapes. Option A
gets variPEPS's chi_init=1 dynamics without disrupting that contract.

### Why not random orthogonal init (Option C)?

Breaks Z₂, but introduces seed dependence and deviates from variPEPS's
canonical init — which is what we want to *match* for parity reasons.

## Architecture

The fix lives entirely in the init path. Sweep loop, projector, absorption,
implicit-AD adjoint — all unchanged.

- **Corner**: write only `(0, 0) = 1`, rest 0. (Today: `eye(min(chi, D²))` zero-pad.)
- **Edge**: write the δ\_{ket=bra} pattern only on the `(0, ·, 0)` slice
  (i.e. `T[0, j*(D+1), 0] = 1` for `j ∈ 0..D-1`), rest 0. (Today: same δ
  pattern but replicated across `i ∈ 0..min(chi, D)-1`.)
- **Symmetric path**: identical structural change. Charge tilings on the
  chi-leg and D²-leg stay as today; only the data inside the leading block
  is non-zero. `SymmetricTensor.from_dense` slots the data into the
  corresponding charge sector and leaves all others empty.

## Components

| File | Function | Change |
|---|---|---|
| `src/tenax/algorithms/_ctm_utils.py` | `_make_dense_corner` | `jnp.eye(min(chi, D))` zero-pad → write `(0, 0) = 1` only |
| `src/tenax/algorithms/_ctm_tensor_init.py` | `_make_dense_standard_edge` | drop `for i in range(min(chi, D))` loop, write `T[0, diag_idx, 0]` only |
| `src/tenax/algorithms/_ctm_tensor_init.py` | `_init_symmetric_standard_corner` | `jnp.eye(chi)` → write `(0, 0) = 1` only |
| `src/tenax/algorithms/_ctm_tensor_init.py` | `_init_symmetric_standard_edge` | same loop drop as the dense edge |

Net delta: ~30 lines across 4 functions, mostly removing loops.

### Out of scope

- Split CTM (`_split_ctm_tensor_init.py`) — different chi-grow handling, separate
  diagnostic. Leave for a follow-up if/when split-CTM divergence is observed
  on the same workloads.
- C4v reference path (`_ctm_tensor_c4v.py`) — known-divergent on different grounds
  and not on the iPEPS-AD hot path.
- Any change to the convergence loop, projector, or absorption code.

## Data flow

With chi=chi_target shape but only the (0, ·, 0) slot nonzero on sweep 0:

1. **Sweep 0**: env is rank-1 in chi-space. Enlarged corners (`Q_TL`, `Q_TR`, …)
   built by contracting the env with iPEPS site `A` are rank-D in the fused
   `(chi·D)` leg. `_compute_2x2_projector` SVDs `M = Q_TL · Q_TR` of shape
   `(chi_target·D, chi_target·D²)`, but only the leading `(D, D²)` block is
   nonzero. SVD finds at most `min(D, D²) = D` nonzero singular values; the
   rank-aware F-mask (PR #400) zeros the `chi_target − D` zero modes.
   Projector has D effective columns.
2. **Sweep 1**: env rank ≈ D in chi. Enlarged corners rank ≈ D². SVD finds D²
   nonzero modes (or chi_target if D² ≥ chi_target).
3. **Saturates** at chi_target after roughly `⌈log_D(chi_target)⌉` sweeps.
   Then normal fixed-point convergence.

For D=2 chi=16 that's 4 growth sweeps; for D=3 chi=64 also 4. Negligible
compared to the typical 50–100 sweep convergence budget.

This matches variPEPS's chi_init=1 dynamics — same growth schedule, same
fixed point. Tenax just stores it in a chi_target-shaped buffer the whole
time so JIT signatures don't change.

## Symmetric path / charge derivation

Charge layouts on the chi-leg and D²-leg stay exactly as today
(`_fused_chi_charges` tiled to chi_target, D²-fused from A's virtual
indices). Only the data inside the `(0, ·, 0)` block of edges and `(0, 0)`
block of corners is non-zero.

The `(chi[0], D²[0], chi[0])` block must satisfy charge conservation —
it does, because `chi[0]` is *defined* as the leading entry of the
D²-fused tile and the standard edge specs use complementary flows that
make the conservation law trivially hold for that specific block. (This
is the same charge layout that ships today; we're only zeroing the
extra slots, not changing which sectors exist.)

## Edge cases & error handling

- **`chi_target < D`**: rank-1 init still works — only writes (0, 0). New
  code drops the conditional entirely.
- **Warm-start callers** (e.g. AD path warmed from a previous L-BFGS step)
  skip `initialize_ctm_tensor_env` entirely. No change for them.
- **Rank-aware SVD path**: if `_compute_2x2_projector` is called with the
  rank-aware path off (some non-default branch in `projector_method`),
  early sweeps will inject zero-mode noise and convergence may regress.
  Rank-aware truncation is the default; the convergence regression test
  pins `projector_method="svd"` (rank-aware as of PR #400) explicitly.

## Testing

Three layers:

1. **Init invariants** (unit tests, fast). For both dense and symmetric
   variants: assert corner has exactly one nonzero entry at `(0, 0)`;
   assert edge has D nonzero entries at `(0, j·(D+1), 0)` and zero
   everywhere else. Cover D=2, D=3 to confirm the diag-pattern formula.
2. **Convergence regression** (smoking gun from
   `project_ctm_two_init_bugs_found.md`). Random complex iPEPS at D=2,
   chi=4, seed 0. After ≤30 sweeps, expect leading C1 SV ≈ 0.948
   (variPEPS reference), `sv_diff < 1e-5`, **non-degenerate** SVs (no
   `[a, a, b, b]` pattern). Pre-fix this test fails (sv_diff plateaus
   at ~0.05 at the `[0.68, 0.68, 0.20, 0.20]` degenerate fp).
3. **Existing CTM suite**: must not regress beyond the 4 pre-existing
   failures already documented (`test_one_sweep_*_finite`,
   `test_qr_projector_symmetric_matches_eigh`).

## Acceptance

- All 3 test layers pass.
- chi=16 iPEPS-AD benchmark cold-from-random no longer times out at the
  L-BFGS line-search phase (root cause per the diagnostic memo was
  CTM not reaching tol within `max_iter=100`).
- No JIT recompiles introduced (init shape signature unchanged).

## Open questions / risk

- The symmetric path assumes the `(chi[0], D²[0], chi[0])` block is a
  valid sector. Confirmed by inspection for the U(1) trivial-charge case
  (which is the only path exercised today). Non-trivial U(1)/Zn paths are
  blocked at `_compute_2x2_projector` per PR #416 — re-validate when
  that path unblocks.
- For very small `chi` the growth phase eats up several sweeps before
  fixed-point convergence even starts. If users run with `max_iter`
  set tight, iteration budget may need to be increased. Mitigation:
  document the expected `⌈log_D(chi_target)⌉` growth-phase cost in the
  CTM convergence docstring.
