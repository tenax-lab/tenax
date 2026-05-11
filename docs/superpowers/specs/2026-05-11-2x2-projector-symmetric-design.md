# 2x2 Plaquette Projector — SymmetricTensor Support (Issue #416)

**Status:** Design — pending implementation.
**Tracking issue:** [#416](https://github.com/tenax-lab/tenax/issues/416).
**Predecessor:** [`docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md`](../../plans/2026-05-07-ctm-multisite-2x2-projector-design.md) — the dense-only implementation (PR #406).

## Goal

Extend `_compute_2x2_projector` in `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` to accept `SymmetricTensor` inputs carrying non-trivial U(1) charges (and beyond — any `Symmetry` Tenax already supports). On forward CTM (non-tracer) the function must return block-sparse `SymmetricTensor` projectors whose charge structure is consistent with the input corners' fused legs. On JAX-traced backward (AD via implicit-FP GMRES), fall back to the existing dense pipeline.

Unblocks: 7 tests xfailed in PR #417 (TestADSymmetric × 4, TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs, TestTodenseGradientFlow × 2).

## Approach (decided)

- **Block-sparse via fused-SymmetricTensor SVD.** Replace the explicit `.todense().reshape()` matrix-view skeleton with label-based SVDs via `tenax.linalg.svd`. The four-leg enlarged-corner-product tensor (e.g. `top_T = Q_TL · Q_TR`) is passed directly to `svd(...)` with two of its labels grouped as `left_labels` and the other two as `right_labels`; the per-sector block-sparse SVD handler in `tenax/linalg.py::_truncated_svd_symmetric` does the work.
- **`chi_new` charge allocation:** mirror `_svd_projector_symmetric`. The function signature gains an optional `base_charges` argument; when supplied, the truncated-SV-to-`chi_new` allocation is done sector-by-sector via `_derive_charges(base_charges, chi)`. When not supplied, fall back to global top-k. Default `None` preserves current callers.

## Architecture

```
_compute_2x2_projector(Q_TL, Q_TR, Q_BL, Q_BR, chi, direction, base_charges=None)
├── if any input is SymmetricTensor + no JAX tracers in any block
│   └── _compute_2x2_projector_symmetric(...)            ← new
└── else (DenseTensor inputs, or symmetric tensors under AD tracing)
    └── existing dense pipeline (unchanged)              ← keep verbatim
```

Single dispatch point at the top of `_compute_2x2_projector`. The dense pipeline is left in place — it's the AD backward path and the trivial-charge fast path. The new helper is a parallel implementation that mirrors the dense flow step-for-step but on `SymmetricTensor` inputs.

## `_compute_2x2_projector_symmetric` — block-sparse pipeline

Steps mirror the dense path (`_ctm_tensor_projector_2x2.py:260-501`).

### Step 1 — Form M1 / M2 as 4-leg `SymmetricTensor`s

For `direction in ("left", "right")`:

```python
top_T = contract(Q_TL.relabels({...}), Q_TR.relabels({...}))
# Labels: ("chi_B_TL", "d2_TL", "chi_B_TR", "d2_TR")

bot_T = contract(Q_BR.relabels({...}), Q_BL.relabels({...}))
# Labels: ("chi_T_BR", "u2_BR", "chi_T_BL", "u2_BL")
```

No `.todense().reshape()`. The relabel maps are unchanged from the dense path.

### Step 2 — Label-based block-sparse SVD via `tenax.linalg.svd`

```python
from tenax.linalg import svd as tensor_svd

U_M1_T, M1_S, Vh_M1_T, M1_S_full = tensor_svd(
    top_T,
    left_labels=("chi_B_TL", "d2_TL"),
    right_labels=("chi_B_TR", "d2_TR"),
    new_bond_label="m1_new",
    max_singular_values=None,   # keep full spectrum, like the dense path
)

# Apply 2x2 gauge-fix per sector (preserves U @ diag(s) @ Vh == M)
U_M1_T, Vh_M1_T = _gauge_fix_symmetric_svd(U_M1_T, Vh_M1_T)
M1_S = _fishman_truncate_S(M1_S, eps=1e-12)

# Identical for M2
U_M2_T, M2_S, Vh_M2_T, M2_S_full = tensor_svd(bot_T, ...)
U_M2_T, Vh_M2_T = _gauge_fix_symmetric_svd(U_M2_T, Vh_M2_T)
M2_S = _fishman_truncate_S(M2_S, eps=1e-12)
```

### Step 3 — Half construction (`first_half`, `second_half`)

The dense code multiplies U / Vh by `sqrt(S)` broadcast along the bond axis. For SymmetricTensor: the new bond's `TensorIndex` charges identify each kept SV's sector, so the scaling is naturally per-block. Implementation via a small helper `_scale_bond_by_diag(T, diag, bond_label)`:

- For each block of `T`, find the bond axis and multiply by the corresponding slice of `diag` (the slice is determined by the bond TensorIndex slot range for that block).

`first_half`, `second_half` are 3-leg `SymmetricTensor`s with labels `(chi_outer, fused_D2, m1_new)` etc. (per `direction`).

### Step 4 — Form `M_prime`, SVD it

The dense path computes `M_prime = first @ second` (or `second @ first`) as a matrix-multiply. In SymmetricTensor land this is a `contract(first_half, second_half)` over the shared bond label:

```python
# first_half labels: (..., m1_new)
# second_half labels: (m2_new, ...)
# Want to contract m1_new with m2_new (or vice versa); use relabels to pair them.
M_prime_T = contract(first_half.relabel("m1_new", "shared_bond"),
                     second_half.relabel("m2_new", "shared_bond"))

U_Mp_T, S_Mp, Vh_Mp_T, _ = tensor_svd(
    M_prime_T,
    left_labels=first_outer_labels,   # first_half's labels minus shared_bond
    right_labels=second_outer_labels, # second_half's labels minus shared_bond
    new_bond_label="chi_new",
    max_singular_values=chi,
)
U_Mp_T, Vh_Mp_T = _gauge_fix_symmetric_svd(U_Mp_T, Vh_Mp_T)
```

### Step 5 — Build Fishman cross-projectors

Mirroring the dense Step 5:

- `P_first ~ first_half · V_Mp · S^{-1/2}`
- `P_second ~ S^{-1/2} · U_Mp^H · second_half`

In symmetric form: `S_inv_sqrt` is applied per-block along the `chi_new` bond via `_scale_bond_by_diag`. The contractions become `contract(first_half, V_Mp_T_scaled)` and `contract(U_Mp_dagger_scaled, second_half)`.

### Step 6 — Relabel to public output names

The dense path wraps with hard-coded zero-charge indices on `("chi_outer", "fused_D2", "chi_new_top")` / `("chi_new_bot", "chi_outer", "fused_D2")`. In the symmetric branch the indices come straight from the contracted SymmetricTensors — no need to construct them. Just relabel via `.relabels({"m_new": "chi_new_top", ...})`.

## `_gauge_fix_symmetric_svd` helper

The dense `_gauge_fixed_svd` puts `conj(phase)` on `U` and `phase` on `Vh` so that `U @ diag(s) @ Vh == M` exactly (critical for the 2x2 closure where there's no intervening matrix to absorb a `conj(phase)²` factor).

The symmetric variant iterates the `U`'s blocks. For each block with bond-axis sector `q`:

```python
# block shape: (... legs..., k_q)
# For each kept column j in 0..k_q:
#   - find max-abs row in that column of the block
#   - phase = block[max_row, j] / |block[max_row, j]| if nonzero else 1
#   - U_block[:, j] *= conj(phase)
#   - Vh_block[j, :] *= phase
```

Because each kept SV lives in exactly one sector (block-diagonal property of symmetric SVD), the phase application is per-sector with no cross-sector mixing. Output: a new pair `(U_T, Vh_T)` with the same charge / shape metadata but phase-rotated blocks.

This helper is also useful elsewhere — but for now it lives next to `_compute_2x2_projector_symmetric` and is private.

## AD fallback

```python
def _has_tracer(t: Tensor) -> bool:
    if isinstance(t, SymmetricTensor):
        return any(isinstance(b, jax.core.Tracer) for b in t.blocks.values())
    return isinstance(getattr(t, "_data", None), jax.core.Tracer)

if (any(isinstance(q, SymmetricTensor) for q in (Q_TL, Q_TR, Q_BL, Q_BR))
        and not any(_has_tracer(q) for q in (Q_TL, Q_TR, Q_BL, Q_BR))):
    return _compute_2x2_projector_symmetric(...)
# else: existing dense pipeline (densify if SymmetricTensor + tracer; pure dense passthrough otherwise)
```

This matches the dispatch pattern in `_compute_projector_tensor` for the standard 1x1 projector. The dense fallback handles AD backward (GMRES matvec evaluates projectors with tracers; symmetric per-sector loops are not JIT-able under tracing because `k_q` is a Python int).

## `chi_new` sector allocation

Add an optional `base_charges: np.ndarray | None = None` argument to `_compute_2x2_projector`. Plumb it through to the M_prime SVD via `_truncated_svd_symmetric`'s allocation logic. In the symmetric branch:

- If `base_charges is not None`: derive target sector counts via `_derive_charges(base_charges, chi)`, allocate per-sector accordingly, fill any remaining budget by global top-k.
- If `base_charges is None`: pure global top-k via the standard `tenax.linalg.svd` truncation path.

Default `None` preserves all existing callers in `_ctm_tensor_moves.py`. New callers (any that want sector-preservation across CTM sweeps) can supply `A.indices[0].charges`.

## Drop the guard

`_ctm_tensor_projector_2x2.py:247-260` raises `NotImplementedError` on any non-trivial-charge input. After the symmetric branch lands, replace that guard with the dispatch above.

## Tests

New tests in `tests/test_ctm_2x2_projector.py` (already exists for the dense path):

- `test_compute_2x2_projector_symmetric_trivial_charges_matches_dense` — symmetric inputs with all-zero charges should produce the same projectors as the dense path (within an SVD gauge equivalence, checked via the closure `P_bot @ P_top = I`).
- `test_compute_2x2_projector_symmetric_nontrivial_u1_closure` — non-trivial U(1) charges → projectors satisfy `P_bot · P_top = I_chi_new`.
- `test_compute_2x2_projector_symmetric_chi_new_charges_from_base_charges` — when `base_charges` is supplied, the resulting `chi_new` charges match `_derive_charges(base_charges, chi)`.
- `test_compute_2x2_projector_symmetric_ad_fallback_on_tracer` — wrap symmetric inputs in `jax.grad` of a downstream contraction; the function returns without raising (dense fallback path).

Unxfail the 7 tests blocked by #416 (memory `project_2x2_projector_handoff.md` lists them):

- `tests/test_ipeps.py::TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs`
- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_runs`
- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_energy_decreases`
- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_matches_dense`
- `tests/test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`
- `tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_energy_finite`
- `tests/test_fpeps_ad.py::TestTodenseGradientFlow::test_symmetric_nontrivial_gradient_finite`

If any of these still fails after the symmetric branch is in place, re-xfail it with a fresh issue and a specific failure-mode note.

## Out of scope

- **Symmetric AD-traced 2x2 projector.** Block-sparse SVD under JAX tracing has the same `k_q must be Python int` constraint as the 1x1 path — covered by the dense fallback. Future work if the gradient through dense becomes a bottleneck.
- **Eigh / QR projector methods for the 2x2 path.** The 2x2 projector is SVD/Fishman only; other methods aren't used by the multisite CTM moves.
- **Honeycomb-native 2x2 projector.** That's a different code path (`_ctm_honeycomb_projector.py`); symmetric support there is the M2b roadmap.

## File map

| Path | Change | Lines (approx.) |
|---|---|---|
| `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` | Add `_compute_2x2_projector_symmetric`, `_gauge_fix_symmetric_svd`, `_scale_bond_by_diag` helpers; dispatch from `_compute_2x2_projector`; remove the trivial-charge guard; add `base_charges` parameter (optional, defaults None) | +250 / −15 |
| `tests/test_ctm_2x2_projector.py` | Add 4 new tests (closure + base_charges + AD fallback + dense parity) | +120 |
| `tests/test_ipeps.py` | Remove 5 xfail decorators (lines per `project_2x2_projector_handoff.md`) | −30 |
| `tests/test_fpeps_ad.py` | Remove 2 xfail decorators | −12 |
| `docs/plans/2026-05-07-ctm-multisite-2x2-projector-design.md` | Update §"Symmetric tensor follow-up" to reference this spec | +5 |

## Risks & validations

- **Gauge convention.** The dense path's `_gauge_fixed_svd` is load-bearing for the 2x2 closure (the comment block in the existing code explains why). The symmetric variant must apply the same convention per-block. Closure test (`P_bot · P_top = I`) is the smoking gun — if it fails, the gauge is wrong.
- **`tenax.linalg.svd` truncation semantics.** With `max_singular_values=None` it returns the full spectrum; we need this for M1/M2 SVDs (no truncation there, only Fishman zero-out via `_fishman_truncate_S`). Confirm before implementation by reading `_truncated_svd_symmetric`.
- **Block-shape compatibility under `contract`.** When `first_half` and `second_half` share the labelled bond (`shared_bond`), `contract` must auto-pair them correctly. Tenax's contractor handles label matching by name + flow; `_truncated_svd_symmetric` (verified in `src/tenax/linalg.py:281-286`) returns `bond_index_out` (FlowDirection.OUT) on U and `bond_index_in` (FlowDirection.IN) on Vh of the new bond. So the shared-bond contraction pairs OUT↔IN cleanly by construction. Verify on a hand-built test before the full integration.
- **`tenax.linalg.svd` with `max_singular_values=None`.** Verified at `src/tenax/linalg.py:257-258`: when `None`, no truncation; full spectrum returned in descending order. So M1 / M2 keep the full spectrum and Fishman zero-out (`_fishman_truncate_S`) handles the sub-cutoff zeroing without changing the array length. M_prime truncation uses `max_singular_values=chi` to cap at the target bond.

## Open questions

None.
