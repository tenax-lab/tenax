# Split-CTM energy at large D — design

**Date:** 2026-05-04
**Author:** YJ Kao (with Claude)
**Status:** design approved; implementation plan to follow.

## Problem

`compute_energy_split_ctm_tensor` is currently a shim: `_split_env_to_tensor_standard`
merges each `(T_ket, T_bra)` over the interlayer `_I` bond and fuses the two
D-legs into D², then delegates to `compute_energy_ctm_tensor`, which further
fuses A and `A.bar_super()` into a 6-leg `ao = (u2, d2, l2, r2, phys, phys_bra)`
factor. Net result — split CTM saves nothing at energy time.

The dominant intermediate inside `_rdm1x2_tensor` and similar half-RDM
constructions sits at `χ²·D⁴·d²` (post-PR #389; was `χ²·D⁵·d²` before). At the
Liao 2017 kagome target (D=10, χ=200, d=2) that's ~410 GB and OOMs every box
we have. Issue #388 misattributed this to the SVD projector; PR #389 fixed the
contraction-order regression but couldn't break below the `χ²·D⁴·d²` floor
without redesigning how `ao` is built and consumed.

## Goal

Replace the shim with a split-aware energy path that bounds the peak
intermediate at **≤ χ²·D⁴ ≈ 6 GB** at D=10 χ=200, fitting a 10 GB box. Forward
energy only in this round; AD is a deliberate follow-up.

## Scope

**In scope (decisions Q1, Q2, Q3 from brainstorming):**

- All six RDM functions get split-aware variants:
  `_rdm_1site_split_tensor`, `_rdm2x1_split_tensor`, `_rdm1x2_split_tensor`,
  `_rdm_diagonal_split_tensor`, plus 2-site mixed-env variants
  `_rdm{2x1,1x2}_split_tensor_2site`.
- Three energy entry points:
  - `compute_energy_split_ctm_tensor` (rewritten in place; replaces the shim).
  - `compute_energy_split_ctm_tensor_2site` (new).
  - `compute_energy_split_ctm_tensor_multisite` (new; unblocks the Liao kagome
    audit which goes through the multisite driver).
- The existing shim `_split_env_to_tensor_standard` is **retained as a private
  test helper** — used inside Tier-1 parity tests at D≤4 to prove the new path
  matches the old one numerically. Not exported.
- Both `DenseTensor` and `SymmetricTensor` (U(1)) backends. Symmetry comes for
  free via the label-based `Tensor.contract` machinery.
- Fermionic correctness via `A.bar_super()` (not a separate code path).

**Out of scope:**

- AD support (`optimize_gs_ad`-compatible gradient through the new RDMs). Will
  be a follow-up PR with `jax.checkpoint` annotations and gradient parity tests.
- Split-CTM SVD projector (`project_ctm_tech_debt.md` item 4). Independent of
  this work; the new energy path works with the current `eigh` projector.

## Approach (chosen: Approach 1 — split-edge with interleaved absorption)

Pre-merge `T_ket·T_bra` over `_I` into a 4-leg "split edge"
`(chi, u_ket, u_bra, chi)` with the two D-legs left **unfused**. This eliminates
the `_I` bond at the boundary, dodging the `χ_I³` trap that Approach 2 would
hit. Then build half-RDMs in a fixed contraction order that:

1. Absorbs `A` and `A.bar_super()` as **separate** 5-leg tensors (no `ao`
   fusion).
2. Consumes one D-leg pair at a time, so no intermediate carries more than
   four D legs simultaneously.

### Approaches considered and rejected

- **Approach 2 — defer ket/bra fusion all the way; let opt_einsum schedule.**
  In principle reaches the lowest peak; in practice (a) opt_einsum's path
  search is exponential in 10-tensor inputs, and (b) χ_I in the current split
  CTM is left at ~2D² so a path that keeps three `_I` bonds open mid-computation
  hits `χ²·χ_I³·D·d` ≈ 50 TB at D=10. Only viable after a separate χ_I
  compression effort, which is unrelated work.
- **Approach 3 — per-site env, uniform across all four RDMs.** Same memory
  bounds as Approach 1 but a much bigger refactor: the existing `_rdm2x1` and
  `_rdm1x2` use left/right halves, not per-site envs. Held for a possible
  future cleanup PR if Approach 1's per-RDM hand-tuned ordering proves brittle.

## Architecture

All new code lives in `src/tenax/algorithms/_split_ctm_tensor_energy.py`.
No new files.

```
_split_ctm_tensor_energy
  ├── _ctm_tensor_init     (does NOT call _build_double_layer_open_tensor)
  ├── _split_ctm_tensor_init  (SplitCTMTensorEnv)
  ├── _ctm_tensor_init     (CTMTensorEnv — used inside the retained shim only)
  ├── tenax.contraction.contractor.contract
  └── tenax.core.tensor.Tensor / FlowDirection
```

The new path **never calls `_build_double_layer_open_tensor` or
`_build_double_layer_tensor`**. Each RDM uses A (5-leg) and `A.bar_super()`
(5-leg, suffix-relabeled) directly.

### File layout after this work

```
_split_ctm_tensor_energy.py
  # Helpers (new)
  _make_split_edge(...)
  _make_split_edges(env) -> {T1, T2, T3, T4}
  _A_bra(A) -> Tensor

  # Split-aware RDMs (new)
  _rdm_1site_split_tensor(A, A_bra, env, split_edges)
  _rdm2x1_split_tensor(A, A_bra, env, split_edges)
  _rdm1x2_split_tensor(A, A_bra, env, split_edges)
  _rdm_diagonal_split_tensor(A, A_bra, env, split_edges)
  _rdm2x1_split_tensor_2site(A, B, A_bra, B_bra, env_A, env_B, split_edges_A, split_edges_B)
  _rdm1x2_split_tensor_2site(...)

  # Entry points
  compute_energy_split_ctm_tensor(A, env, gate, d=None)            # rewritten
  compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=None)   # new
  compute_energy_split_ctm_tensor_multisite(site_tensors, envs, neighbors, gate, d=None)  # new

  # Demoted shim (parity tests only)
  _split_env_to_tensor_standard(env) -> CTMTensorEnv
```

`src/tenax/__init__.py` and `_split_ctm_tensor.py`'s `__all__` gain the two
new entry points; existing exports unchanged.

## Components and contraction orders

### Pattern A — two-half (2×1 horizontal, 1×2 vertical, plus 2-site variants)

Same left/right or top/bottom decomposition as the standard `_rdm2x1_tensor` /
`_rdm1x2_tensor`. The new contribution is the half-builder ordering.

Top half of `_rdm1x2_split_tensor`:

```
T1_split = _make_split_edge(env.T1_ket, env.T1_bra, ...)   # (chi, u_ket, u_bra, chi)
T4_T_split, T2_T_split similar (top-row copies of T4, T2)
A_bra = A.bar_super().relabels({u: u_bra, d: d_bra, l: l_bra, r: r_bra,
                                phys: phys_bra})

top_row     = C1 · T1_split · C2                # χ²·D²
top_T4      = top_row · T4_T_split              # χ²·D⁴   ← peak edge stage
top_T4_A    = top_T4 · A                        # χ²·D⁴·d ← peak overall
top_T4_A_T2 = top_T4_A · T2_T_split             # χ²·D³·d
top_half    = top_T4_A_T2 · A_bra               # χ²·D²·d²

# Bottom half: mirror; peak shapes identical.
# Combine: shared (t4_u↔t4_dB chi×2, d_ket↔u_ket_B, d_bra↔u_bra_B);
# output (phys, phys_B, phys_bra, phys_braB).
```

The non-obvious choice is at `top_T4_A`: contract A using only its `u` and `l`
legs (matched to `u_ket` and `l_ket`), leaving `r`, `d`, `phys` open. T2 is
absorbed *before* `A_bra` so the r-leg gets contracted while only one D layer
is open.

`_rdm2x1_split_tensor` is the same recipe rotated 90°. 2-site variants reuse
the body with `env_A` / `env_B`, `A` / `B` substitutions.

**Peak at D=10 χ=200 d=2 (complex128, 16 B/element):**
- `top_T4_A`: χ²·D⁴·d = 4×10⁴ · 10⁴ · 2 entries × 16 B ≈ **13 GB**.
- All other stages ≤ 6 GB.

This is tighter than the χ²·D⁴ ≈ 6 GB I quoted in the brainstorm; the d-factor
brings the worst stage to ~13 GB on complex128. At 8 GB float64 (real spin
models without complex SU sectors), the same stage is ~6 GB. The Liao audit is
real-arithmetic kagome, so we land near the 6 GB target.

### Pattern B — per-site env (1site, diagonal)

`_rdm_diagonal_split_tensor` keeps the existing 4-site (TL, TR, BL, BR)
decomposition. Each per-site env is built with split edges and separate
A / `A.bar_super()`, using the same interleaved D-consumption as Pattern A.

Closed sites (TR, BL) trace `phys` early so their `A·A_bra` contracts down to
an 8-leg block where the four D-leg pairs stay **unfused**:

```
ac_TR = (A · A_bra) traced over phys
       # 8-leg, dims (D,D,D,D,D,D,D,D), no D² fuse
       # peak during ac_TR build = D⁵·d (≈10 MB at D=10 d=2)
site_env_TR = TR_env · ac_TR                    # χ²·D⁴ peak
```

The wrinkle: the standard `_rdm_diagonal_tensor` uses
`ac = _build_double_layer_tensor(A)` which fuses to `(u2, d2, l2, r2)`. Here we
keep `ac` 8-leg per leg. That preserves the χ²·D⁴ peak in `TR_env · ac_TR` and
avoids the carried-D² leftover at the column / final-combine stage.

`_rdm_1site_split_tensor` is a degenerate per-site env (no glue).

### Helpers

```python
def _make_split_edge(T_ket, T_bra, ket_I_label, bra_I_label,
                     out_chi_l, out_d_ket, out_d_bra, out_chi_r) -> Tensor:
    """Contract T_ket and T_bra over the _I bond; do NOT fuse the two D-legs."""
    k = T_ket.relabel(ket_I_label, "_I")
    b = T_bra.relabel(bra_I_label, "_I")
    merged = contract(k, b)                             # (chi, u_ket, u_bra, chi)
    return merged.relabels({...: out_chi_l, ...: out_d_ket,
                            ...: out_d_bra, ...: out_chi_r})

def _make_split_edges(env: SplitCTMTensorEnv) -> dict[str, Tensor]:
    """Return {'T1': ..., 'T2': ..., 'T3': ..., 'T4': ...} with consistent labels."""
```

`_make_split_edge` is the only place the `_I` bond is consumed. After this,
no peak intermediate carries χ_I.

## Data flow

**Single-site `compute_energy_split_ctm_tensor(A, env, gate, d)`:**

```
1. resolve d (from A's phys index)
2. H = gate.reshape(d, d, d, d)
3. split_edges = _make_split_edges(env)               # 4× χ²·D² tensors, built once
4. A_bra = A.bar_super().relabels({u: u_bra, ...})    # built once
5. rdm_h = _rdm2x1_split_tensor(A, A_bra, env, split_edges)
6. rdm_v = _rdm1x2_split_tensor(A, A_bra, env, split_edges)
7. return (einsum(rdm_h, H) + einsum(rdm_v, H)).real
```

`split_edges` and `A_bra` are built **once** per energy call and passed into
both halves — no per-RDM re-merge. The current shim accidentally rebuilds the
merged env for each RDM internally; the new path reuses.

**2-site `compute_energy_split_ctm_tensor_2site`:** same caching for both
sublattices.

**Multisite `compute_energy_split_ctm_tensor_multisite`:** mechanical copy of
`compute_energy_ctm_tensor_multisite` (lines 603-675 of `_ctm_tensor_energy.py`)
with per-coord caching of `split_edges` and `A_bra`, dispatching to the
single-env or mixed-env split RDM by `coord == nb_coord`.

**Cache lifetime:** local to each energy call; out of scope after return. JAX
trace boundaries handle re-use within a single jitted wrapper if the user
provides one.

## Error handling

Forward-only path; reuses `Tensor.contract`, `Tensor.relabels`,
`Tensor.bar_super`, and the existing `(d²,d²)` reshape/symmetrize/normalize
tail. Most error surfaces are inherited.

Three real concerns:

1. **Label collisions across split edges.** Four `T_split` tensors are live
   simultaneously. `_make_split_edges` chooses globally unique names per edge
   (only ever appearing on one tensor). A cheap one-shot assertion runs in
   `_make_split_edges` to verify the four returned tensors share no leg labels
   except their chi-bonds.
2. **Fermionic Koszul correctness.** `_build_double_layer_open_tensor` does
   bar+fuse in one go, so any twist sign is baked in pre-fuse. Approach 1
   calls `bar_super()` then never fuses, so the twist is applied per leg by
   Tenax's contraction machinery itself. *Should* be equivalent — covered by
   the Tier-2 fermionic parity test.
3. **`todense()` on the final RDM.** Each split RDM ends with
   `rdm_t.todense()` returning `(d²,d²)`. Same pattern as the existing
   standard CTM RDMs and within the codebase rule (small dense at the tail).

**Non-concerns:** empty/zero-charge `SymmetricTensor` sectors (handled by
`contract`); D=1 / χ=1 degenerate cases (collapse cleanly); mismatched flow
on `_I` (caught by `contract` with a clear error).

## Testing

### Tier 1 — parity vs the demoted shim (small D)

In `tests/test_split_ctm_tensor.py`, for each new RDM and each new energy
entry point:

```
@pytest.mark.parametrize("backend", ["dense", "symmetric_u1"])
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
def test_split_energy_matches_shim(backend, D, chi):
    A = make_random_site(backend, D, d=2)
    env = run_split_ctm(A, chi=chi, ...)
    E_split = compute_energy_split_ctm_tensor(A, env, gate)         # new path
    std_env = _split_env_to_tensor_standard(env)                    # demoted shim
    E_shim  = compute_energy_ctm_tensor(A, std_env, gate)           # existing standard path
    assert jnp.allclose(E_split, E_shim, atol=1e-10)
```

Per-RDM parity (`_rdm{1site,2x1,1x2,diagonal}_split_tensor` vs the shim's
`_rdm*_tensor` over the merged env), checking the `(d,d,d,d)` arrays
element-wise. 2-site and multisite get analogous parity tests; multisite uses
a 3-site Y-shaped unit cell (smallest non-trivial kagome simplex).

### Tier 2 — fermionic parity (small D)

One additional parametrize: `FermionParity` site tensor, D=2/3, χ=8/12. Same
shim-vs-split parity at `1e-10`. Marked `@pytest.mark.slow` (pulls fermionic
infrastructure).

### Tier 3 — memory regression on the kagome harness

Use `examples/kagome_spin12_pess_liao2017_replication.py` (lands with PR #387)
as the harness. New `tests/test_split_ctm_large_d_memory.py` (marked `slow`):

1. Single `compute_energy_split_ctm_tensor_multisite` call at **D=8 χ=128** on
   the canonical kagome PESS site tensor, wrapped in `tracemalloc`. Assert
   peak < 8 GB.
2. Same harness at **D=4 χ=32** asserts the energy reproduces -0.347185 within
   1e-4 (the regression point validated by PR #389).

D=10 χ=200 is **not** in CI (would need a 16 GB box that the runners don't
have); documented in the PR description for local probes.

### Out of test scope

- AD correctness — out of scope per Q2 (forward-only). Follow-up PR adds
  `optimize_gs_ad` + `jax.checkpoint` and gradient-parity tests.
- Convergence / projector quality at large D — that's the still-open
  `project_ctm_tech_debt.md` item 4 (split-CTM SVD projector). Energy can
  still be evaluated with the current `eigh` projector; it just may converge
  worse.

## Memory accounting summary

At D=10 χ=200 d=2, complex128:

| Stage                        | Shape          | Bytes |
|------------------------------|----------------|-------|
| `top_T4_A` (peak)            | χ²·D⁴·d        | ~13 GB |
| `top_T4_A_T2`                | χ²·D³·d        | ~1.3 GB |
| `top_half` (final pre-glue)  | χ²·D²·d²       | ~256 MB |

For float64 (typical real-arithmetic spin model), halve. Under the kagome PESS
audit's working dtype (real for SU phase, complex for the small-D AD probes),
peak ~6–13 GB lands inside a 10–16 GB box.

## References

- PR #389 (rebalance vertical/diagonal RDM contractions to χ²·D⁴·d² floor)
- `project_rdm_oom_fix.md` — issue #388 misdiagnosis; current floor.
- `project_ctm_tech_debt.md` — split-CTM SVD projector (item 4, independent).
- `examples/kagome_spin12_pess_liao2017_replication.py` (in PR #387 worktree).
- `arXiv:2502.10298` — split-CTM with SVD projector reference (out of scope here).
