# #463 Phase 2 — `fuse_virtual_legs` flag + AD-complete split-CTM path

**Date:** 2026-06-26
**Issue:** [#463](https://github.com/tenax-lab/tenax/issues/463) — *unify split-CTM as the canonical path; archive fused virtual-leg construction*
**Scope this PR:** Phase 2 only (the flag + caller wiring). Default behavior is unchanged. Phases 3 (default-flip + regression cycle) and 4 (delete fused path) are out of scope.

## Goal

Make the modern Tensor-protocol **split-CTM** path selectable as a drop-in
replacement for the **fused double-layer** path through a single configuration
flag, with *both* AD paths (explicit and implicit) producing gradients that match
the fused path to tolerance. After this PR a user can set
`CTMConfig.fuse_virtual_legs=False` and run `optimize_gs_ad` on the
**single-site (`recipe="1x1"`) path** exactly as before, but with the χ²·D⁴ split
memory profile instead of χ²·D⁶.

The flag defaults to `True` (fused), so nothing changes for existing callers.

### Scope restriction — single-site only

The split **forward** (`ctm_split_tensor`, `_split_ctm_tensor_sweep`,
`_split_ctm_tensor_step`) is **single-site only**: it converges one `A` as an
isolated 1×1 iPEPS. The `compute_energy_split_ctm_tensor_2site` / `_multisite`
energy functions exist, but they consume per-site `SplitCTMTensorEnv`s that the
current forward can only produce by converging each site *independently* as its
own 1×1 lattice (see `test_split_ctm_tensor.py:890,923`) — which is **not** a
physically-correct multisite CTM. There is no true multisite split sweep.

Therefore `fuse_virtual_legs=False` is honored **only for the single-site
(`len(site_tensors) == 1`) path**. The default `recipe="2x2"` multisite/
checkerboard path, the 2-site path, c4v, honeycomb, and PESS all raise
`NotImplementedError` when the flag is `False`. A genuine multisite split
forward is deferred (Phase-1-completion follow-up). This single-site path is
exactly the large-D / multi-GPU `recipe="1x1"` Heisenberg showcase where the
χ²·D⁴ memory win is most valuable.

## Background / current state

- The fused builder `_build_double_layer_tensor` (`_ctm_tensor_init.py:84`) fuses
  ket/bra virtual pairs into rank-4 double-layer tensors at D² per leg. Peak
  memory χ²·D⁴·d² (energy) / χ²·D⁶ (moves).
- The split path (`_split_ctm_tensor_{init,moves,convergence,energy}.py`) keeps
  ket and bra layers separate (`SplitCTMTensorEnv`, χ_I interlayer bond). It has
  a complete **forward** (`ctm_split_tensor`, `_split_ctm_tensor_sweep`),
  **energy** (`compute_energy_split_ctm_tensor` / `_2site` / `_multisite`, with a
  fermionic shim fallback below `_MIXED_ENV_RDM_TRACE_FLOOR`), and a flat
  per-sweep step (`_split_ctm_tensor_step` in `ad_utils.py:1216`).
- **What's missing:** the split path has *no* `custom_vjp` implicit backward
  (verified: zero `custom_vjp` in `_split_ctm_tensor*.py`). Today its only
  gradient source is explicit unrolled AD via
  `ctm_split_tensor_converge_explicit`. `ctm_split_tensor_fixed_point`'s
  docstring claims a GMRES backward but the body just calls the plain forward.
- The single production AD dispatcher is `make_ctm_energy_fn`
  (`ipeps_ad_policy.py:154`): it already centralizes the explicit-vs-implicit
  branch and reads every knob off the live `CTMConfig`. All three bipartite/
  multisite `optimize_gs_ad` dispatchers route through it.
- c4v, honeycomb, and PESS have **no** split energy companion.

## Design

### Component 1 — `CTMConfig.fuse_virtual_legs`

Add to `CTMConfig` (`ipeps_config.py`):

```python
fuse_virtual_legs: bool = True
```

- Default `True` ⇒ existing fused path, zero behavior change.
- Documented in the class docstring: when `False`, the **single-site**
  (`recipe="1x1"`) CTM-AD path uses the split (ket/bra-separate) double layer
  with χ²·D⁴ memory; multisite (`recipe="2x2"`), 2-site, c4v, honeycomb, and
  PESS do not yet support it (raise — see Component 3).
- No `__post_init__` validation needed beyond the dispatch guard in Component 3
  (it's a plain bool).

### Component 2 — split AD energy entry points

Add two functions mirroring the fused `ctm_energy_explicit` / `ctm_energy_implicit`
signatures, living in a new module `_split_ctm_energy_ad.py` (keeps the split AD
scaffolding out of the already-large `_ctm_energy_ad.py`; re-exported where the
fused ones are). Both take the **single-site** `site_tensors` dict (exactly one
coord); they assert `len(site_tensors) == 1` and operate on that one `A`:

- **`ctm_energy_split_explicit(...)`** — runs `ctm_split_tensor_converge_explicit`
  (warmup under `stop_gradient`, then differentiable sweeps) and evaluates energy
  with `compute_energy_split_ctm_tensor` (single-site, h+v bonds). Plain JAX
  autodiff through the unrolled split forward. Honors `chi`, `chi_I`,
  `renormalize`, warmup/backprop/backward step counts, and `energy_fn`.

- **`ctm_energy_split_implicit(...)`** — `jax.custom_vjp`:
  - **Forward:** `ctm_split_tensor` to convergence (reusing the existing
    `_split_ctm_tensor_sweep` + gauge fixing), then `compute_energy_split_ctm_tensor`.
  - **Backward:** the *same* fixed-point adjoint already used by the fused
    implicit path — solve `(I − Jᵀ_env) λ = dE/denv`, then chain to `dE/dA`.
    `Jᵀ_env` (`vjp_env_fn`) and `∂step/∂A` (`vjp_site_fn`) come from `jax.vjp`
    of `_split_ctm_tensor_step` (env-arg and A-arg respectively). The solve
    reuses the generic Neumann-series / GMRES selector keyed on
    `config.adjoint_method`. **No new SVD/projector backward is written** — the
    split moves already differentiate through the shared `tenax.linalg.svd`
    custom VJP.
  - **Implementation note:** factor the fused implicit backward's solver core
    (the `(I−Jᵀ)λ=g` Neumann/GMRES loop in `_make_implicit_vjp_fn`) into a
    step-function-agnostic helper if it isn't already, so both env types share
    it. If factoring proves invasive, duplicate the ~40-line solver loop in the
    split module rather than refactor the fused path under a Phase-2 PR (keeps
    the fused path's regression surface untouched).

### Component 3 — dispatch in `make_ctm_energy_fn`

Branch once, on `ctm_cfg.fuse_virtual_legs`, inside `_ctm_energy_fn`
(`ipeps_ad_policy.py`). The branch is gated on a **single-site** cell:

```python
if not ctm_cfg.fuse_virtual_legs:
    if len(site_tensors) != 1:
        raise NotImplementedError(
            "split-CTM (fuse_virtual_legs=False) supports only the single-site "
            "(recipe='1x1') path; got a {n}-site unit cell. Use "
            "fuse_virtual_legs=True for multisite/2-site.".format(n=len(site_tensors))
        )
    if use_explicit:
        return ctm_energy_split_explicit(site_tensors, neighbors, gate, ...)
    return ctm_energy_split_implicit(site_tensors, neighbors, gate, ...)
# existing fused branch unchanged
```

- Knobs forwarded from `ctm_cfg`: `chi`, `chi_I`, `renormalize`, `max_iter`,
  `conv_tol`, `min_iter`, warmup/backprop/backward step counts, `energy_fn`,
  `adjoint_method`, and the GMRES tolerances. Knobs that are fused-only or have
  no split analogue (e.g. sigma-gauge `forward_gauge="sigma"`) are **not**
  forwarded; if the user set an incompatible combination with
  `fuse_virtual_legs=False`, raise a clear `ValueError` listing the offending
  knob (consistent with the codebase's "no silent promotion" convention).

**Other-path guards (NotImplementedError):** c4v, honeycomb, and PESS run through
their own dispatchers (`_ctm_tensor_c4v.py`, `_ctm_honeycomb_*.py`,
`_pess_multisite_energy.py`), not `make_ctm_energy_fn`. Add a guard at each of
those entry points (and/or the `optimize_gs_ad` recipe selection that routes to
them): if `fuse_virtual_legs is False`, raise
`NotImplementedError("split-CTM (fuse_virtual_legs=False) is not yet supported "
"for the <c4v|honeycomb|PESS> path; use fuse_virtual_legs=True")`. This is a
documented Phase-2 limitation, not a regression. The 2-site dispatcher
(`_optimize_gs_ad_tensor_2site`) is covered by the `len != 1` guard above since
it passes a 2-coord `site_tensors`; add an early guard there too for a clearer
message before the energy fn is built.

## Data flow

```
optimize_gs_ad (single-site, recipe="1x1")
  → make_ctm_energy_fn(get_ctm_cfg=…)        # reads CTMConfig live
      → _ctm_energy_fn(site_tensors)         # len(site_tensors) == 1
          ├ fuse_virtual_legs=True  → ctm_energy_{explicit,implicit}       (fused, unchanged)
          └ fuse_virtual_legs=False → ctm_energy_split_{explicit,implicit} (split, χ²·D⁴)
                                          └ _split_ctm_tensor_step / ctm_split_tensor
                                          └ compute_energy_split_ctm_tensor
   (len != 1 with flag False → NotImplementedError; c4v/honeycomb/PESS → NotImplementedError)
```

## Error handling

- Multisite/2-site cell (`len(site_tensors) != 1`) + `fuse_virtual_legs=False`
  → `NotImplementedError`.
- c4v / honeycomb / PESS path + `fuse_virtual_legs=False` → `NotImplementedError`.
- Unsupported knob combination (fused-only knob set with split selected) →
  `ValueError` naming the knob.
- Fermionic tensors on the split path fall back to the existing shim inside the
  split RDM functions when the mixed-env RDM trace underflows — no new handling.

## Testing

All tests target `pytest -m core` runnability at small D/χ (CPU-deliverable).

All tests use a **single-site** (1-coord) iPEPS unit cell.

1. **Implicit gradient parity (new, the load-bearing test).**
   Single-site energy gradient `dE/dA` through the *full fixed point*:
   `ctm_energy_split_implicit` vs `ctm_energy_implicit`, D=2/3/4, χ=8/12/16,
   trivial + U(1) charges, agree to **1e-8**. (Extends the existing
   env-fixed `test_compute_energy_split_native_grad_matches_shim`, which did not
   exercise the fixed-point backward.)
2. **Explicit gradient parity.** `ctm_energy_split_explicit` vs
   `ctm_energy_explicit`, same grid, **1e-8**.
3. **Flag dispatch / end-to-end.** A few single-site `optimize_gs_ad` steps with
   `fuse_virtual_legs=False` vs `True` agree on energy + gradient to **1e-8**
   (mechanism test, not convergence — per `feedback_test_mechanism_not_convergence`).
4. **Fermionic.** `ctm_energy_split_implicit` keeps the existing
   `test_fermionic_ed_reference.py` variational-bound + ED checks green at D=2
   (shim fallback path), single-site.
5. **Guards.** `fuse_virtual_legs=False` raises `NotImplementedError` for a
   2-site/multisite cell and for c4v / honeycomb / PESS; a fused-only knob +
   split raises `ValueError`.

**Parity bar precedent:** `tests/test_ctm_env_pad_chi_schedule.py` (1e-8 grad)
and `test_split_ctm_tensor.py` (1e-10 energy).

## Out of scope (explicit)

- Flipping the default to `False` (Phase 3) and the production Heisenberg
  D=3/χ=24 regression cycle.
- Deleting `_build_double_layer_tensor` and removing the flag (Phase 4).
- A true **multisite split forward** (`recipe="2x2"`/2-site) — the per-site
  CTM convergence that absorbs real unit-cell neighbors. This is the gating
  Phase-1-completion follow-up that would let the flag cover the production
  default path.
- Split companions for c4v / honeycomb / PESS (future Phase-1-style extensions).
- The legacy array-based `ctm_split` API in `ipeps_ctm_convergence.py`.
- The dense `ctm_2site()` simple-update path.

## Risks

- **Solver factoring** (Component 2 note): if the fused `(I−Jᵀ)λ=g` loop is
  entangled with fused-specific flatten/unflatten, prefer duplication over
  refactoring the fused backward in this PR.
- **χ_I default**: split forward uses `chi_I = chi_I or chi`. Ensure
  `make_ctm_energy_fn` forwards `ctm_cfg.chi_I` so split convergence matches the
  fused χ resolution on the parity tests.
- **Gauge fixing**: fused implicit relies on sigma/phase gauge for a
  well-conditioned `(I−Jᵀ)`. Confirm the split sweep applies an equivalent gauge
  fix (phase, per `feedback_phase_gauge_default`) so the adjoint converges; if
  not, the explicit path is the fallback for the parity gate while the gauge is
  added.

## Acceptance

- [ ] `CTMConfig.fuse_virtual_legs: bool = True` added + documented.
- [ ] `ctm_energy_split_explicit` / `ctm_energy_split_implicit` implemented
      (single-site); implicit has a `custom_vjp` fixed-point backward.
- [ ] `make_ctm_energy_fn` dispatches on the flag for single-site; multisite/
      2-site + flag=False raises; c4v/honeycomb/PESS + flag=False raises.
- [ ] Tests 1–5 pass under `pytest -m core`; default-`True` runs bit-identical
      to current `main`.
