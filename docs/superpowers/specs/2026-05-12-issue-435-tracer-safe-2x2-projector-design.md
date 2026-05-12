# Tracer-Safe Symmetric 2x2 Projector — Eliminate Dense Fallback Under AD (Issue #435)

**Status:** Design — pending implementation.
**Tracking issue:** [#435](https://github.com/tenax-lab/tenax/issues/435).
**Predecessor:** [`docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md`](2026-05-11-2x2-projector-symmetric-design.md) — the eager-only symmetric pipeline (PR #434).
**Related:** PR #437 threaded `base_charges` through the 1x1 `_ctm_tensor_sweep` path; this design mirrors that threading on the 2plaq path.

## Goal

Make the SymmetricTensor 2x2 projector path tracer-safe so AD-traced symmetric inputs flow through the block-sparse pipeline instead of falling back to the dense path. The dense fallback's trivial-U(1)-charge wrap (`SymmetricTensor.from_dense(arr, idx, tol=float("inf"))` with all-zero chi_outer / fused_D2 / chi_new charges) produces shape mismatches downstream when inputs carry non-trivial sector structure.

Unblocks the 4 tests xfailed by PR #437 with Issue #435 as reason:

- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`
- `tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite`
- `tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init`
- `tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases`

## Approach (decided)

Three coordinated edits:

1. **`src/tenax/linalg.py`** — add tracer-aware dispatch inside `_truncated_svd_symmetric`. Traced inputs take a new `_truncated_svd_symmetric_traced` branch that does per-sector SVD via `truncated_svd_ad` (composing the existing Lorentzian-regularized dense AD primitive per block), allocates per-sector keep counts statically via `_derive_charges(base_charges, chi)`, and assembles output SymmetricTensors with sector-aware bond charges (not trivial zeros).
2. **`src/tenax/algorithms/_ctm_tensor_projector_2x2.py`** — rewrite `_gauge_fix_symmetric_svd` in pure JAX, removing `int(jnp.argmax(...))` and `complex(col_flat[idx])` casts. Drop the `_has_tracer` filter in `_compute_2x2_projector` so symmetric inputs always take the (now tracer-safe) symmetric path. Replace `S_Mp[0]` at line 960 with `jnp.max(S_Mp)` to handle the new sector-block bond order.
3. **`src/tenax/algorithms/_ctm_tensor_moves.py`** — thread `base_charges` through `_compute_2x2_projector_2plaq` to `_compute_2x2_projector`, mirroring the PR #437 pattern for the 1x1 path. Six call sites; `base_charges` is derived once at the multisite-sweep entry via `_get_base_charges(a)`.

## SVD backward strategy

Per-block reuse of `truncated_svd_ad` (the existing dense Lorentzian-regularized custom_vjp at `_ad_primitives.py:107`). Each sector's dense matrix block goes through `truncated_svd_ad(M_q, k_q)`. Smaller diff than writing a new per-sector custom_vjp; equivalent correctness for intra-sector degeneracies.

## Architecture

```
_compute_2x2_projector(Q_TL, Q_TR, Q_BL, Q_BR, chi, direction, base_charges=None)
├── if any input is SymmetricTensor
│   └── _compute_2x2_projector_symmetric(...)            ← unified path
│         └── tensor_svd(...) → linalg.svd → _truncated_svd_symmetric(...)
│               ├── if any block is a jax.core.Tracer
│               │   └── _truncated_svd_symmetric_traced(...)  ← new
│               └── else: existing global-sort path (unchanged)
└── else (all-DenseTensor inputs)
    └── existing dense pipeline (unchanged)
```

The `_has_tracer` filter at lines 411-426 of `_compute_2x2_projector` is removed; tracer-bearing symmetric inputs no longer route to dense. The dense path survives only for genuinely all-DenseTensor inputs and stays unchanged.

## Component 1 — `_truncated_svd_symmetric_traced` (`linalg.py`)

Signature:

```python
def _truncated_svd_symmetric_traced(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    new_bond_label: Label,
    normalize: bool,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
```

Pipeline:

1. **Per-sector matrix assembly** — identical to lines 141-204 of `_truncated_svd_symmetric` (build `matrix_q` for each q via `jnp.zeros + .at[...].set(...)`, applying Koszul signs for fermions). Already traceable.
2. **Static per-sector keep allocation** — `k_per_sector: dict[int, int]`:
   - If `base_charges is not None` and `max_singular_values is not None`: compute `target_charges = _derive_charges(base_charges, max_singular_values)`; `target_count[q] = count of q in target_charges`; `k_q = min(target_count[q], available_q)`.
   - Else if `max_singular_values is None`: `k_q = min(rows_q, cols_q)` (full spectrum per sector).
   - Else (truncating with `base_charges=None`, defensive fallback): `k_q = max(1, round(max_singular_values * available_q / total_available))`, adjusted so totals don't exceed `max_singular_values` (drop-leftover: prefer correctness over budget exhaustion).
3. **Per-sector AD-primitive SVD** — `U_q, s_q, Vh_q = truncated_svd_ad(matrix_q, k_q)` for each sector with `k_q > 0`.
4. **Concatenate** — `s_final = jnp.concatenate([s_q for q in sectors])`; `bond_charges = np.repeat([q for q in sectors], [k_q for q in sectors])` (numpy, static).
5. **No global SV re-sort** — bond axis ends up in sector-block order, not global SV-descending order. This is a deliberate concession; see the bond-ordering note below.
6. **Wrap** — assemble `U`, `Vh` SymmetricTensors with non-trivial `bond_charges`. `s_full = s_final` under tracing (no separate pre-truncation spectrum tracked).

**Defensive guards:**

- If `tensor.blocks` is empty, fall through to the eager path.
- If `sum_q k_q == 0`, force `k_q = 1` on the sector with the largest `available_q` (ties broken by smallest q) to mirror the eager `n_keep = max(1, n_keep)` floor at `linalg.py:1255`.

**Tracer detection at `_truncated_svd_symmetric` entry:**

```python
is_traced = any(isinstance(b, jax.core.Tracer) for b in tensor.blocks.values())
if is_traced:
    return _truncated_svd_symmetric_traced(
        tensor, left_labels, right_labels, max_singular_values,
        new_bond_label, normalize, base_charges=base_charges,
    )
```

The `base_charges` parameter must thread up through `linalg.svd` as an optional kwarg (mirroring how `_compute_2x2_projector_symmetric` already accepts it).

## Component 2 — Rewritten `_gauge_fix_symmetric_svd` (`_ctm_tensor_projector_2x2.py`)

Structure stays: outer Python loop over global column index `j` (static, bounded by `len(bond_charges)`). Per-iteration body becomes pure JAX:

```python
# Replace current lines 105-122 (the int(...)/complex(...) cast block):
if not u_entries:
    sample_block = next(iter(U_T.blocks.values()))
    phase = jnp.asarray(1.0, dtype=sample_block.dtype)
else:
    candidates = jnp.concatenate([
        jnp.reshape(new_u_blocks[key][..., local], (-1,))
        for key, _ in u_entries
    ])
    max_idx = jnp.argmax(jnp.abs(candidates))
    best_value = candidates[max_idx]
    abs_best = jnp.abs(best_value)
    phase = jnp.where(abs_best > 0, best_value / abs_best, jnp.ones_like(best_value))

# Replace current lines 127-132 (real/complex dispatch):
sample_block = next(iter(U_T.blocks.values()))
is_complex = jnp.issubdtype(sample_block.dtype, jnp.complexfloating)  # static
conj_phase = jnp.conj(phase) if is_complex else jnp.real(phase)
bare_phase = phase if is_complex else jnp.real(phase)
```

The `u_entries` list, `j`-to-`local` mapping, and `is_complex` dispatch are all built from static charge structure (`bond_idx.charges` is numpy, blocks dict keys are static). Only values inside blocks are traced. The `.at[..., local].multiply(conj_phase)` writes already work under tracing.

`_scale_bond_by_diag` is left as-is — inspection confirms it has no `int(...)`/`float(...)` casts on traced values; only static structure walks the blocks dict.

## Component 3 — `base_charges` plumbing (`_ctm_tensor_moves.py`)

Six call sites, mirroring the PR #437 shape:

- `_compute_2x2_projector_2plaq` (line ~343) — accept `base_charges`, forward to `_compute_2x2_projector` on line 355-357.
- Four `_compute_2x2_projector_2plaq` callers in the multisite move helpers (around lines 925, 1029, 1117, 1203) — each already has `base_charges` available from its parent multisite sweep.
- Multisite-sweep entry — derive `base_charges = _get_base_charges(a)` once and thread through.

For DenseTensor inputs `_get_base_charges` returns `None` (existing behavior preserved). For trivial-charge SymmetricTensor inputs it returns the trivial array (no change to allocation behavior). For non-trivial U(1), it returns the sector vector that `_derive_charges` consumes.

## Bond-axis ordering — deliberate concession

The traced `_truncated_svd_symmetric_traced` does **not** globally sort the bond axis by SV magnitude after assembly. Result: bond charges and `s_final` are in sector-block order (sector q1's k_q1 entries, then sector q2's k_q2, ...), not global SV-descending order.

This differs from the eager path's output, which preserves global SV-descending order. The difference is a permutation of the bond axis only — the per-sector SV content kept (which singular values survive truncation) is identical when `base_charges` is supplied, because both paths consume `_derive_charges(base_charges, chi)`.

**Why this is safe:** tensor contractions in Tenax match by charge identity per block, not by bond position. The chi_new TensorIndex emerges in sector-block order under tracing, and downstream `contract(...)` operates per-block. Block content per (charge tuple) is identical between eager and traced paths.

**One read site needs an update:** `S_Mp[0]` at `_compute_2x2_projector_symmetric` line 960 reads the largest SV; under traced sector-block ordering this would be the largest of the *first* sector, not the global max. Replace with `jnp.max(S_Mp)`.

No other position-dependent reads found in the projector or absorb code paths.

## Data flow / charge propagation

```
                       FORWARD (eager)                BACKWARD (traced)
ket tensor `a`
   │
   │ _get_base_charges(a)
   ▼
base_charges ────────────────────────────── (same numpy array, captured at trace)
   │                                                       │
   │ multisite sweep entry                                 │ implicit-GMRES matvec
   ▼                                                       ▼
_ctm_tensor_sweep_multisite(... , base_charges=bc)   <traced replay of same sweep>
   │                                                       │
   │ _compute_2x2_projector_2plaq(..., base_charges=bc)    │ tracer-bearing Q tensors
   ▼                                                       ▼
_compute_2x2_projector(Q_*, chi, base_charges=bc)
   │
   ├─ inputs_are_symmetric? yes (drop the _has_tracer filter)
   │
   ▼
_compute_2x2_projector_symmetric (now tracer-safe)
   │
   ├─ Stage 2: M1/M2 SVDs (max_singular_values=None) → full-spectrum per sector
   ├─ Stage 2: _gauge_fix_symmetric_svd (rewritten pure-JAX)
   ├─ Stage 4: M_prime SVD with chi truncation
   │     → _truncated_svd_symmetric → traced branch on detection
   │     → k_q from _derive_charges(base_charges, chi)
   │     → drop-leftover rule
   ├─ Stage 4: S_Mp[0] → jnp.max(S_Mp)
   └─ Stages 5-6: cross-projector contractions, relabel, transpose

returns (P_top, P_bot) SymmetricTensor with chi_new charges
derived from base_charges (NOT trivial zero)
```

Forward and backward see the same `base_charges` (captured at trace), so `_derive_charges(base_charges, chi)` produces identical `target_charges`. Truncation decision is consistent between forward and backward.

## Truncation rule under tracing

Adapts `_retruncate_by_base_charges` for the traced path. Three steps:

1. **target_count from base_charges (purely structural).** `target_count[q] = number of times q appears in _derive_charges(base_charges, chi)`. No SV magnitudes involved.
2. **per-sector top-k.** For each sector q: keep the first `min(target_count[q], available_q)` SVs. Per-sector SVD already returns descending; slicing `[:k_q]` keeps the top-k.
3. **Drop leftover.** If `sum_q min(target_count[q], available_q) < chi`, the output bond dim is less than chi. No greedy fill across sectors (which would require global SV order, not available under tracing). Downstream code tolerates variable bond dim.

The eager path with `base_charges` uses steps 1 and 2 identically, then back-fills leftover via global SV order (step 3 in eager). For typical CTM inputs, `target_count` matches `available` per sector and leftover-fill rarely triggers, so the forward/backward divergence in leftover behavior is bounded.

## Error handling / edge cases

**Inherited (no new code):**

- Zero-dimension sectors (`rows_q = 0` or `cols_q = 0`) — existing `continue` guard at line 175-176 applies in the traced variant.
- Sectors in target but absent from input blocks — `available_q = 0`, sector contributes 0 to output bond.
- Fermionic Koszul signs — assembled before SVD (lines 186-199), composes with `truncated_svd_ad` without modification.
- Mixed DenseTensor + SymmetricTensor inputs — handled at the outer `inputs_are_symmetric` dispatch (line 398).

**New, design-level:**

- Total output dim = 0 — force `k_q = 1` on the sector with the largest `available_q` (ties broken by smallest q). Mirrors eager `n_keep = max(1, n_keep)` floor.
- Tracer detection on empty `tensor.blocks` — fall through to eager path defensively.
- Dtype consistency — if mixed-dtype blocks ever appear (shouldn't, defensively), cast all blocks to the widest dtype before per-sector SVD.

**Documented but unfixed:**

- Non-U(1) symmetries (Z_n, FermionParity) inherit the traced path automatically; acceptance tests cover U(1) only.
- `base_charges` captured at trace time is a JAX-tracing user-error class; document in docstring.

## Testing

**Primary acceptance (un-xfail in the implementation PR):**

- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`
- `tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite`
- `tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init`
- `tests/test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_energy_decreases`

Remove `@pytest.mark.xfail(reason="Issue #435 ...")` decorators in the same PR.

**New explicit tests (`tests/test_ctm_2x2_projector.py` or equivalent):**

1. **Tracer-safety unit test.** Call `_compute_2x2_projector` inside a `jax.grad(...)` of a scalar functional, with symmetric non-trivial-U(1) inputs. Assert no `TracerArrayConversionError` / `ConcretizationTypeError`.
2. **Sector-preservation check.** After `_compute_2x2_projector` with symmetric inputs, assert `P_top.indices[-1].charges` (chi_new_top leg) is **not** all zeros.
3. **Eager vs traced numerical equivalence (trivial-charge).** Run both paths on trivial-U(1) inputs; assert `P_top @ P_top.conj().T` (permutation-invariant on bond) matches to 1e-8 relative.
4. **Gradient finite-difference cross-check.** Small non-trivial-U(1) iPEPS (D=2, chi=4): `jax.grad(|P_top|²)` vs central-difference, agreement to 1e-4 relative.
5. **Closure check under tracing.** `P_bot · P_top` on chi_outer / fused_D2 seam = identity to 1e-8.

**Regression guards:**

- Existing 5 `#416`-unxfailed tests (trivial-charge symmetric) stay green.
- Eager-mode 2x2 projector tests stay green (global-sort path untouched).
- Existing AD iPEPS optimization tests with trivial charges stay green.

**CI scope:** primary acceptance tests are slow AD-iPEPS optimization (`algorithm`-marked); run on push-to-main and via `run-full-tests` PR label. New unit tests tagged `core`, run on every CI invocation.

## Implementation order (for writing-plans)

1. Component 2 first — rewrite `_gauge_fix_symmetric_svd` in pure JAX. Verifiable in isolation with a small symmetric SVD test inside `jax.grad`.
2. Component 1 — add `_truncated_svd_symmetric_traced` and thread `base_charges` through `linalg.svd`. Verifiable in isolation with a per-sector SVD test inside `jax.grad`.
3. Update `S_Mp[0] → jnp.max(S_Mp)` at `_compute_2x2_projector_symmetric` line 960.
4. Component 3 — thread `base_charges` through `_ctm_tensor_moves.py` 2plaq path.
5. Drop the `_has_tracer` filter at `_compute_2x2_projector` lines 411-426.
6. Un-xfail the 4 acceptance tests; add the 5 new explicit tests.
7. Run full CI, confirm no regressions.

## Out of scope

- A native per-sector custom_vjp SVD primitive — Component 1 composes `truncated_svd_ad` per block instead.
- Non-U(1) acceptance coverage — inherits support automatically; explicit tests deferred.
- The remaining 4 open CTM issues (#411, #392, #336, #200) — orthogonal to this fix.

## References

- PR #437 — `base_charges` threading pattern for the 1x1 `_ctm_tensor_sweep` path.
- PR #434 — original symmetric path that introduced the dense fallback this design eliminates.
- `truncated_svd_ad` (Francuz et al. PRR 7, 013237) at `src/tenax/algorithms/_ad_primitives.py:107`.
- `_retruncate_by_base_charges` reference pattern at `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:708`.
- Predecessor design doc at `docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md`.
