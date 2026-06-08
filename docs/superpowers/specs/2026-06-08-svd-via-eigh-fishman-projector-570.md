# SVD-via-eigh in the Fishman 2×2 projector — banking the #570 decomposition-VJP win

**Date:** 2026-06-08
**Status:** ❌ FALSIFIED 2026-06-08 (prototype measured, premise does not hold) — see below.
**Issue:** #570 (CTM-AD compile wall = block-sparse SVD VJP, confirmed in PR #589).
**Branch:** `feat/svd-via-eigh-projector-570` (off `main`).

---

## ❌ FALSIFICATION (2026-06-08) — do not pursue SVD-via-eigh as a compile lever

Prototyped `truncated_svd_via_eigh_ad` (eigh of the Gram matrix + reconstruct
`U = M·V/s`) and measured its backward against `truncated_svd_ad` on the SAME full
`(U, s, Vh)` contract the Fishman projector needs:

- **No op-count win:** eigh-route backward = **274 ops** vs `truncated_svd_ad` =
  **265** (break-even / slightly worse) on a 16×16 matrix, k=8.
- **Mechanistic root cause (decisive):** `eigh(MᴴM)` has eigenvalues `w = S²`, so
  its degeneracy F-matrix is `1/(wᵢ−wⱼ) = 1/(sᵢ²−sⱼ²)` — **identical** to the SVD's.
  There is no *fundamental* backward saving from SVD→eigh for a full decomposition.
  The earlier "~2.6×" (probe_decomp_vjp_cost_570: 261 vs 113) compared eigh's bare
  `(w, v)` loss against svd's `(U, s, Vh)` loss — **apples to oranges**. Once you
  reconstruct the `U = M·V/s` (and form `MᴴM` and differentiate through it) that
  the projector requires, the leaner eigh backward is fully offset.
- Secondary symptom: the naive reconstruction's **complex128 gradient was wrong**
  (max|Δ|≈32 vs `truncated_svd_ad`) — a Gram-matrix conjugation subtlety. Moot
  given the op-count result.

**Conclusion:** "cheaper per-sector decomposition" is NOT a real #570 lever; the SVD
backward's cost is intrinsic (the `1/(sᵢ²−sⱼ²)` F-matrix), shared by eigh-of-Gram.
The prototype + test were reverted (no broken/no-win primitive shipped). **Pivot to
(b): batch the per-sector decompositions** — that attacks the confirmed χ-driver
(per-SECTOR multiplicity: PR #589 showed per-decomp VJP is flat in size, so the
χ-scaling is *count of sectors*, not their size). Collapsing N per-sector SVD VJPs
into ONE batched SVD VJP removes the per-sector emission the loop creates — cf. the
gated `TENAX_BATCH_BLOCKSPARSE` batched-SVD (#572), now specifically justified.
Truncated backprop remains the orthogonal depth lever.

The original design follows as the record of the road not taken.

---

## Goal

Bank the measured **~2.3–2.6× cheaper decomposition VJP** on the compile-dominant
symmetric CTM-AD path, **without changing the CTM fixed point** (so energy/gradient
parity holds). PR #589 established: the fused backward's only χ-scaling term is the
block-sparse SVD VJP (61% at D=4/χ=12), and an isolated dense decomposition VJP is
261 ops (production SVD) vs 113 (eigh) — the SVD's Lorentzian/gauge F-matrix
machinery is the overhead, which eigh's backward avoids.

## Key idea (why parity is preserved)

For `M = U S Vᴴ`, the Gram matrix `Mᴴ M = V S² Vᴴ` is Hermitian. So:
`eigh(Mᴴ M)` → eigenvalues `= S²`, eigenvectors `= V`. Reconstruct
`S = sqrt(eigenvalues)`, `U = M V S⁻¹`. The singular **subspaces are identical** to
`svd(M)` (up to the usual sign/phase gauge, which the projector's
`_fix_svd_signs` / `_gauge_fix_symmetric_svd` already canonicalize). So the Fishman
two-projector construction is **mathematically unchanged** — same projectors, same
fixed point, same energy — but the differentiated decomposition is now `eigh`
(cheap Lorentzian VJP) instead of `svd` (expensive F-matrix VJP).

Numerical caveat (the thing the parity test gates): `Mᴴ M` squares the condition
number, so singular values below ~√(machine-eps)·σ_max lose precision. In f64 with
Fishman's existing `eps=1e-12` truncation this is expected to be benign in the
truncated regime; the energy/FD-gradient parity test is the gate. Compute the Gram
matrix on the **smaller** dimension (`MᴴM` if m≥n else `M Mᴴ`) for stability + cost.

## Architecture (4 pieces, smallest-blast-radius first)

### 1. `truncated_svd_via_eigh_ad(M, chi)` — `_ad_primitives.py`
A **drop-in for `truncated_svd_ad`**: same `(U[m,k], s[k], Vh[k,n])` contract and
the same `_fix_svd_signs` gauge. Computed via `regularized_eigh` on the Gram matrix
so the VJP is the eigh backward (no SVD F-matrix). It is **NOT** a new `custom_vjp`
— the cheap backward comes for free by routing the eigendecomposition through the
existing `regularized_eigh` custom_vjp and doing the `S=√w`, `U=M V/s`
reconstruction with plain differentiable ops.

```python
def truncated_svd_via_eigh_ad(M, chi):
    m, n = M.shape
    k = min(chi, min(m, n))
    if m >= n:
        w, V = regularized_eigh(M.conj().T @ M)       # w asc, = S²; V = right vecs
        w_desc = w[::-1]; V = V[:, ::-1]
        s = jnp.sqrt(jnp.clip(w_desc[:k], a_min=0.0))
        Vk = V[:, :k]
        s_inv = jnp.where(s > _SV_FLOOR, 1.0 / s, 0.0)
        U = (M @ Vk) * s_inv[None, :]                 # (m,k)
        Vh = Vk.conj().T
    else:
        w, U_ = regularized_eigh(M @ M.conj().T)      # left vecs
        ... symmetric reconstruction (Vh = U_ᴴ M / s) ...
    return _fix_svd_signs(U, s, Vh)
```
`_SV_FLOOR` mirrors the existing zero-SV handling; `_zero_subrank_singular_values`
parity is checked by the test, not assumed.

### 2. eigh variant of `_truncated_svd_symmetric_traced` — `linalg.py`
The per-sector loop already calls `truncated_svd_ad` (linalg.py:625/803). Add a
`decomp="svd"|"eigh"` parameter (default `"svd"`, byte-identical) that selects
`truncated_svd_via_eigh_ad` per sector instead. Thread it through
`_truncated_svd_symmetric` → the public `tenax.linalg.svd`'s symmetric path via a
keyword (default off).

### 3. wire into `_compute_2x2_projector_symmetric` — `_ctm_tensor_projector_2x2.py`
Add `decomp="svd"` param; pass it to the three `tensor_svd` calls. When `"eigh"`,
the Fishman construction is identical but each SVD uses the eigh route.

### 4. dispatch from the CTM step
`_make_jit_ctm_step` / the symmetric single-site step gains a way to select the
eigh decomposition. **Decision:** use a dedicated env/config knob
(`TENAX_CTM_DECOMP=eigh`, default `svd`) rather than overloading
`projector_method="qr"` (whose existing meaning is the isometric density-matrix
projector on the *other* path — overloading it would be confusing). The projector
*algorithm* is unchanged (still Fishman); only the decomposition backend changes,
so a separate knob is the honest name.

## Testing (the parity gate — TDD, each piece before wiring the next)

`tests/test_svd_via_eigh_570.py` (mark `core`):
1. **Value parity (gauge-invariant).** `truncated_svd_via_eigh_ad(M,k)` vs
   `truncated_svd_ad(M,k)`: reconstruction `U·diag(s)·Vh` and singular values `s`
   match at fp tier (1e-10), for well-conditioned + rank-deficient + degenerate-SV
   matrices, real **and** complex128. (Never compare raw U/Vh — gauge.)
2. **Gradient parity (the crux).** `jax.grad` of a gauge-invariant scalar loss
   (`sum(s²)` and `‖U diag(s) Vh‖²`) through the eigh route == through
   `truncated_svd_ad` == finite-difference, fp tier, real + complex128. Well-
   conditioned matrices first; then a moderate-conditioning case to bound the
   `MᴴM`-squaring error.
3. **VJP op-count win.** `len(make_jaxpr(grad(eigh_route)))` < `truncated_svd_ad`
   on the same matrix (the #570 lever, ~2.3×); assert strict reduction.
4. **Symmetric parity.** `_truncated_svd_symmetric_traced(decomp="eigh")` vs
   `decomp="svd"`: per-sector reconstruction + bond spectrum match, FermionParity
   ferm_D2/D4 sites.

`tests/test_ctm_decomp_eigh_parity_570.py` (mark `algorithm`):
5. **Energy parity.** A CTM-converged energy with `TENAX_CTM_DECOMP=eigh` matches
   `svd` to ≤1e-8 on a small fermionic/Heisenberg iPEPS — the fixed-point gate.
6. **End-to-end gradient parity.** `value_and_grad` of the CTM energy: eigh vs svd
   gradient agree at fp tier (and FD-consistent) — proves the AD path is correct.

Plus a **backward-attribution re-run** (not a unit test): with the eigh path wired,
`probe_bwd_subop_attribution_570.py` should show `svd_vjp` collapse and
`eigh_vjp` rise at D=4/χ=12 — the direct evidence the lever fired.

## Scope / non-goals
- **Default unchanged.** `decomp="svd"` everywhere by default; the eigh route is
  opt-in (env/kwarg) until the A100 compile-time benefit + parity are confirmed.
- **Does NOT change the χ-scaling** (per-sector emission persists — that's #572
  batched-decomp + truncated backprop, separate levers). This banks the constant.
- **No true QR-CTMRG** (separate, larger effort).
- **No GPU default-flip** (pending the A100 compile-time measurement).

## Risks
- **Conditioning** (`MᴴM` squares κ): the central risk; gated by tests 1/2/5. If a
  well-conditioned projector regime passes but degenerate cases fail, restrict the
  eigh route to where κ is safe, or fall back per-sector.
- **Gradient through `U = M V/s`** near small `s`: mirror the existing zero-SV floor;
  test 2's degenerate case is the gate.
