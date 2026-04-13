# Extending C4v-reference AD to multisite iPEPS — design

**Status:** design, not implemented
**Date:** 2026-04-13
**Author:** YJ Kao + Claude
**Related:** PR #304 (c4v_reference 1-site), PR #306 (rename), PR #310
(docs sync), PR #311 (Tikhonov damping), issue #292 (GMRES broken), issue
#299 (2-site χ=8 convergence gap)

## Problem

Tenax has a working *implicit*-AD path for 1-site C4v iPEPS
(`ctm_ad_mode="c4v_reference"`) that follows Francuz–Schuch–Vanhecke,
*PRR* **7**, 013237 (2025): stable Lorentzian-regularized truncated-eigh
backward plus an implicit linear solve `(I − Jᵀ)λ = g` at the CTM fixed
point via bicgstab / gmres. It is the most stable AD path in the repo
for AFM square-lattice models.

The 2-site (and larger) unit-cell paths do **not** have access to any of
this. Specifically:

- The `_use_reference_c4v_path` gate in `ipeps_optimize.py:366–373`
  requires `unit_cell == "1x1"`. The 2-site shared-tensor C4v forward
  added in PR #304 uses the generic multisite CTM and standard explicit
  AD — no Lorentzian backward, no implicit adjoint.
- The explicit 2-site path converges only to E ≈ −0.558 at χ=8 on the
  Heisenberg AFM (literature: −0.6548). Tracked as issue #299.
- All "c4v-then-release" and "independent c4v on (A, B)" experiments
  have failed (see `project_2site_c4v_benchmark.md`,
  `project_multisite_ad_instability.md`). Today's
  `bench_2site_c4v_release.py` run drove stage B to E = −0.7901,
  non-variational drift.

The goal is to give the 2-site path (and in principle any multisite C4v
unit cell) access to the same gradient-quality improvements that
reference mode gives the 1-site path.

## Scope

Two approaches, landed in order. Approach A first; B builds on it.

### Approach A — Lorentzian projector backward on the multisite path

**Cheap, high-leverage, addresses the χ=8 gap.** Borrow only the
**Lorentzian-regularized truncated-eigh backward** from reference mode
and drop it in as the projector VJP on the standard (non-reference)
CTM path. Keep the forward (explicit, unrolled + `jax.checkpoint`)
unchanged. This is orthogonal to the sigma-gauge dead end
(`project_2site_sigma_gauge_dead_end.md`) — sigma kills gradient flow
via `stop_gradient`; the Lorentzian backward is just a better local
projector derivative and is compatible with phase/QR gauge.

**Target paths:**

- 1-site explicit AD, `projector_method="eigh"`
- 2-site explicit AD, `projector_method="eigh"`
- Dense + U(1) `SymmetricTensor` (FermionParity deferred)

**Non-goals for A:**

- Implicit fixed-point adjoint (that is B)
- `projector_method="qr"` and `"svd"` projectors (only `"eigh"`)
- FermionParity / graded tensors
- 1-site reference mode is already covered via the shared kernel and
  stays unchanged

### Approach B — Full implicit adjoint on multisite CTM

**Larger lift, real structural extension.** Generalize the Krylov
adjoint `(I − Jᵀ)λ = g` from the reduced `(C, T)` pair used by 1-site
C4v to the full multisite environment (8 tensors per sublattice).
Drops the `unit_cell="1x1"` gate. Strictly implicit
(`gs_explicit_ad=False`). Reuses A's Lorentzian kernel pointwise per
sublattice projector. Unlocks the path for honeycomb C3v and any
multi-site C4v configuration. Full design deferred until A lands.

## Approach A — detailed design

### Architectural map (what changes, what doesn't)

```
Before (2-site explicit AD):
  params (A, B) → CTM forward (unrolled) → projector eigh [JAX default VJP] → loss → grad

After (2-site explicit AD, auto-promoted):
  params (A, B) → CTM forward (unrolled) → projector eigh [Lorentzian VJP] → loss → grad
                                                                  ▲
                                                       This is the only change.
```

The CTM forward, the explicit unrolling, the checkpointing, the phase
gauge, the metric preconditioner, and the L-BFGS / Hager-Zhang line
search all stay exactly as they are.

### Kernel extraction and dispatch

Move the Lorentzian kernel out of `_ctm_tensor_c4v_reference_ad.py`
into a new shared module `src/tenax/algorithms/_lorentzian_eigh.py`
with two layers:

- `_lorentzian_eigh_dense(matrix, …)` — the current kernel, operates
  on a raw JAX array. This is the existing
  `_truncated_eigh_lorentzian_backward` + `truncated_eigh_regularized`
  logic, moved verbatim.
- `lorentzian_eigh(tensor, …)` — dispatch layer:
  - `DenseTensor` input → call dense kernel directly.
  - `SymmetricTensor` input (U(1) or Z_n) → iterate over symmetry
    sectors, apply the dense kernel per block, reassemble. Per-block
    degenerate-eigenvalue regularization is handled locally; no
    cross-block coupling is introduced.
  - `SymmetricTensor` bearing `FermionParity` → raise
    `NotImplementedError("Lorentzian backward for fermionic tensors "
    "not yet supported; see design doc 2026-04-13")`.

Re-export both from `_ctm_tensor_c4v_reference_ad.py` so PR #304's
reference-mode path picks up the refactor transparently — zero behavior
change for existing `c4v_reference` users.

### Config surface

New field on `CTMConfig`:

```python
projector_backward: Literal["standard", "lorentzian"] = "standard"
```

Default `"standard"` preserves the current JAX-native eigh backward
for any caller who constructs `CTMConfig` directly (forward-only CTM,
notebooks, diagnostics).

### Auto-promotion

In `ipeps_optimize.py`, add an auto-promotion block that mirrors the
existing `forward_gauge="qr" → "phase"` promotion:

```
if gs_explicit_ad and projector_method == "eigh" and not user_set(projector_backward):
    projector_backward = "lorentzian"
    log.info("Auto-promoted projector_backward to 'lorentzian' for explicit AD.")
```

User-explicit values are never overridden, in either direction.

### Projector wiring

In `_ctm_projector.py`, locate the `projector_method="eigh"` branch
that calls eigh on `ρ = C·Cᵀ + …`. Add a thin dispatch:

- `projector_backward == "standard"` → existing call
- `projector_backward == "lorentzian"` → `lorentzian_eigh(ρ, …)`

`projector_backward` is plumbed through the same config-threading path
that `projector_method` already uses
(`CTMConfig → _ctm_tensor_*` → `_ctm_projector`).

**Risk:** the projector may not have a single clean eigh call-site.
If there are multiple, hoist them into a helper first as a separate
refactor commit before introducing the dispatch.

### Tests (TDD — write first)

**Core tier** (`tests/test_lorentzian_projector_backward.py`):

- `test_fd_ad_gradient_matches_dense` — FD–AD agreement on a toy 2-site
  checkerboard Heisenberg, D=2, χ=4 (dense). Mirrors the
  `test_c4v_reference_ad.py::test_ctm_tensor_c4v_reference_backward_matches_fd`
  structure.
- `test_fd_ad_gradient_matches_u1` — same but with a U(1)
  `SymmetricTensor` site tensor. Exercises the per-block dispatch.
- `test_degenerate_eigenvalue_regression` — repeat the reference-mode
  degenerate-eigenvalue case to confirm extraction preserved behavior.
- `test_explicit_opt_out_not_promoted` —
  `projector_backward="standard"` + `gs_explicit_ad=True` +
  `projector_method="eigh"` → not auto-promoted.
- `test_auto_promotion_is_logged` — capture log, assert promotion
  happens and user-explicit values are respected.
- `test_fermion_parity_raises` — `FermionParity` tensor →
  `NotImplementedError`.
- `test_cross_block_degeneracy` — two U(1) blocks with numerically
  equal eigenvalues, gradient is still finite and FD-matching.

**Slow tier** (`tests/test_ipeps_lorentzian_convergence.py`,
`@pytest.mark.slow`):

- `test_2site_chi8_closes_issue_299` — 2-site shared-tensor C4v
  Heisenberg at D=2, χ=8 with Lorentzian backward. Expect E < −0.62
  within 50 L-BFGS steps. Current path only reaches −0.558.
- `test_1site_parity_chi16` — 1-site C4v + Lorentzian at D=2, χ=16;
  no regression vs existing 1-site explicit-AD path.

### Docs

- Update `docs/ipeps-code-paths.md`:
  - Add `projector_backward` row to the Config Cheat Sheet.
  - Add Status Summary row "Lorentzian projector backward (explicit
    AD) — Working, auto-default when `projector_method='eigh'`".
- Update `docs/guide/algorithms/ipeps_ad_paths.md`: note the new
  auto-promotion on the recommended path.
- No change to `CLAUDE.md` — this is a config knob, not a workflow
  rule.

### Migration risk

- Any existing caller with `gs_explicit_ad=True` +
  `projector_method="eigh"` + no explicit `projector_backward` gets a
  different gradient silently. This is the whole point; flag it in the
  PR description and release notes.
- Reference-mode path unchanged via the re-export.
- `projector_method="qr"` and `"svd"` untouched.

## Approach B — outline

Deferred; sketched for forward compatibility only.

- **Goal:** implicit fixed-point AD for any multisite C4v unit cell.
  Drops the `unit_cell="1x1"` gate in `_use_reference_c4v_path`.
- **Core change:** lift the Krylov adjoint from the reduced `(C, T)`
  pair to the full multisite environment dict (8 tensors per
  sublattice). Residual
  `F(params, env_dict) = env_dict − CTM_step_multisite(params, env_dict)`,
  built via `jax.vjp` tracing, same pattern as the current reference
  mode.
- **Reuse from A:** the Lorentzian kernel applied pointwise per
  sublattice projector.
- **New pieces:** environment flatten / unflatten utilities for Krylov
  input; diagonal preconditioner scaled per sublattice; Tikhonov
  damping as in PR #311.
- **Gate:** `ctm_ad_mode="c4v_reference"` extended to accept
  `unit_cell in {"1x1", "2site"}`. `gs_explicit_ad=False` required.
- **Prerequisite validation:** check that the multisite CTM forward in
  a pure reference-mode config produces the same fixed point as the
  current 2-site shared-tensor forward before differentiating.
- **Risks:** larger null space for the residual (Krylov stalls,
  Tikhonov load-bearing); honeycomb needs a C3v parameterization that
  does not exist yet.
- **What A does for B:** validates the Lorentzian kernel on the
  non-reference path and forces the `_ctm_projector.py` hook surface
  into a clean shape that B can reuse.

## Success criteria

**Approach A:**

- All core-tier FD–AD tests pass.
- Issue #299 closed: 2-site shared-tensor C4v Heisenberg at D=2, χ=8
  converges to E < −0.62 with auto-promoted Lorentzian backward.
- 1-site C4v path shows no regression at D=2, χ=16.
- Reference mode (`c4v_reference`) behavior bit-identical via the
  re-export.

**Approach B:** out of scope for this design doc; will be captured in a
follow-up design once A lands.

## Out of scope

- Fermionic iPEPS (`FermionParity`) — separate PR
- SVD two-projector regularized backward
- `projector_method="qr"` backward improvements
- Honeycomb C3v parameterization
- Multi-state targeting
- Implicit adjoint (that's Approach B)
