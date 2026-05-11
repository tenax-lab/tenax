# 2×2 plaquette CTM projector for the multisite path — design

Date: 2026-05-07
Branch: `worktree-fix-multisite-ctm-rdm-helpers`

## Background

Investigation of the C.3 floor breach (`project_c3_floor_breach_smoking_gun.md`) identified that Tenax's `_ctm_tensor_sweep_multisite` and variPEPS's `calc_ctmrg_env` converge to **different physical fixed points** on the kagome 3-site multisite-encoded state at the saved D=4 χ=16 AD-optimum:

| source | E/site | math |
|---|---|---|
| Tenax multisite-CTM, RDM route | -0.912858 | correct trace, 1×1 projector |
| variPEPS gate API / variPEPS RDM (correct trace) | -0.255359 | correct trace, 2×2 projector |

Per chi-scan probes `examples/dev/p1_tenax_chi_scan.py` (Tenax) and `p2_varipeps_chi_scan.py` (variPEPS), both implementations are bit-identical across χ ∈ {8, 16, 32, 48} — they're at fully-converged stationary points. Per `examples/dev/p3_tenax_projector_scan.py`, all three Tenax projector methods (svd/eigh/qr) produce different attractors, none matching variPEPS.

Reading both implementations end-to-end (`_ctm_tensor_moves.py:241-293` and `varipeps/ctmrg/projectors.py:569-632` + `definitions.py:645-656`) shows the structural difference:

* **Tenax**: 1×1 grown corners. `C1g = C1·T1` (2 tensors), `C4g = C4·T3` (2 tensors). Projector SVD is on `M = C1g^† · C4g`, a (chi, chi) cross product.
* **variPEPS**: 2×2 plaquette quarters. `Q_TL = C1·T1·T4·a` (5 tensors), similarly Q_TR/Q_BL/Q_BR. Projector SVD is on `top_M = Q_TL · Q_TR`, a (chi·D², chi·D²) matrix.

The 2×2 plaquette recipe is the modern CTMRG standard (Corboz, Penc, Mila, Lauchli, PRB 84, 041108(R) (2011)) and retains all D² intermediate degrees of freedom. The 1×1 recipe (Nishino-Okunishi 1996; YASTN's `proj_corners` for the simplest case) is cheaper and accurate for well-conditioned states, but on ill-conditioned states (such as the multisite encoding where S_v / S_w have two virtual legs trivial-padded to dim 1) it can lock onto a non-physical fixed point.

The dim-1 trivial padding is structural to the 3-site multisite encoding (`pess_to_kagome_3site_multisite`) and not a bug — it's the mechanism by which v-w correlations propagate through the absorbed T_u/T_d simplices on the u sublattice. Adopting the standard 2×2 plaquette projector for the multisite path resolves the conditioning issue without changing the encoding.

## Goal

Replace the projector recipe used by `_ctm_tensor_sweep_multisite` with the 2×2 plaquette / Fishman cross-projector formulation, preserving the existing API. Single-site `_ctm_tensor_sweep`, paired moves, C4v, and honeycomb-CTM are not touched.

## Contract

At the saved D=4 χ=16 AD-optimum (`logs/d4_ad_optimum.npz`), `_collect_ctm_rdms` (driven by the new sweep) gives a per-site energy

```
E/site = -0.255359 ± 1e-3
```

matching variPEPS's gate API to 3 decimals.

## Scope

**In:**
* New 2×2 plaquette projector recipe — DenseTensor only (every current multisite caller passes DenseTensors).
* All four directions (left, right, top, bottom).
* AD path inherits via JAX tracing — same forward primitive, gradient via the existing implicit-AD GMRES infrastructure.

**Out:**
* ~~SymmetricTensor support for the 2×2 path.~~ Shipped on `feat/2x2-projector-symmetric` (Issue #416). Block-sparse via `tenax.linalg.svd` + per-sector gauge-fix; AD-tracer fallback wraps dense output as SymmetricTensor. See `docs/superpowers/specs/2026-05-11-2x2-projector-symmetric-design.md`.
* Single-site `_ctm_tensor_sweep` (used by DMRG observables and vanilla iPEPS) keeps the 1×1 recipe.
* Paired moves, C4v, honeycomb CTM — untouched.
* Performance optimization beyond a sanity check.

## Implementation layout

### New file: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`

```python
def _build_enlarged_corner(
    C: Tensor,
    T_horizontal: Tensor,
    T_vertical: Tensor,
    a: Tensor,
    *,
    position: str,            # "top_left" | "top_right" | "bottom_left" | "bottom_right"
) -> Tensor:
    """Enlarged corner Q = C · T_h · T_v · a.

    Output is rank-4 with axes (chi_outer_h, D2_outer_h, chi_outer_v, D2_outer_v)
    where the "_outer_*" legs are the bonds that connect to the adjacent quarter
    in the 2×2 plaquette. The fuse convention follows Tenax's existing
    {C, T} edge convention (chi · D² fused with chi-slow, D²-fast)."""

def _compute_2x2_projector(
    Q_TL: Tensor, Q_TR: Tensor, Q_BL: Tensor, Q_BR: Tensor,
    chi: int,
    *,
    direction: str,           # "left" | "right" | "top" | "bottom"
    projector_method: str = "svd",
    truncation_eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Compute (P_top, P_bot) projector pair from the 2×2 quarters.

    For direction='left':
        top_M    = Q_TL · Q_TR     # contracts top-row chi+D² seam
        bottom_M = Q_BR · Q_BL     # contracts bottom-row chi+D² seam (reversed)
        Fishman cross-projector:
          (top_U, top_S, _) = SVD(top_M); top_S_trunc = drop(eps)
          (_, bot_S, bot_Vh) = SVD(bottom_M); bot_S_trunc = drop(eps)
          top_half = top_U @ sqrt(top_S_trunc)
          bot_half = sqrt(bot_S_trunc) @ bot_Vh
          M_prime  = bot_half @ top_half           # small (kept_bot, kept_top)
          (U_M, S_M, V_M_h) = truncated_SVD(M_prime, chi)
          S_inv_sqrt = 1 / sqrt(S_M)
          P_top = top_half @ V_M_h.T.conj() * S_inv_sqrt
          P_bot = S_inv_sqrt[:, None] * U_M.T.conj() @ bot_half
        Reshape P_top → (chi_outer, D, D, chi_new)
        Reshape P_bot → (chi_new, chi_outer, D, D)
    Other directions: cyclic permutation of the same template."""
```

Internals reuse `_truncated_SVD` from `_ctm_projector.py` (the truncation logic, multiplet handling, complex-128 stability all stay shared). The Fishman post-processing in `_ctm_projector.py:_svd_projector_dense` is structurally similar — the difference is only the input shape (small (chi, chi) vs large (chi·D², chi·D²)).

### Modified file: `src/tenax/algorithms/_ctm_tensor_moves.py`

Add four new functions parallel to the existing 1×1 moves:

```python
def _ctm_tensor_move_left_2x2(
    env_self: CTMTensorEnv,             # site at (x, y) ── top-left of plaquette
    env_TR: CTMTensorEnv,               # neighbors[(x,y)]["right"]
    env_BL: CTMTensorEnv,               # neighbors[(x,y)]["bottom"]
    env_BR: CTMTensorEnv,               # neighbors[(x,y)]["right"]["bottom"]
    a_self: Tensor,
    a_TR: Tensor,
    a_BL: Tensor,
    a_BR: Tensor,
    chi: int,
    projector_method: str = "svd",
) -> CTMTensorEnv:
    """Left-move using 2×2 plaquette projectors. Updates self.{C1, C4, T4}."""
```

The signature change (4 sites + 4 envs vs 2) is unavoidable — that's the structural distinction.

### Modified file: `src/tenax/algorithms/_ctm_tensor_convergence.py`

`_ctm_tensor_sweep_multisite` switches its default sweep to the `_2x2`-suffixed moves. A private `recipe: Literal["1x1", "2x2"] = "2x2"` kwarg lets the existing 1×1 path be reached for transitional pinning:

```python
def _ctm_tensor_sweep_multisite(
    envs, double_layers, neighbors, chi, renormalize,
    projector_method="svd", projector_backward="auto",
    *, recipe: Literal["1x1", "2x2"] = "2x2",
) -> dict[Coord, CTMTensorEnv]: ...
```

`ctm_multisite` and `_ctm_tensor_multisite` propagate the `recipe` kwarg.

### Untouched

* `_ctm_projector.py` — projector internals (SVD, eigh, qr) are shared.
* `_ctm_tensor_paired_moves.py` — paired moves stay 1×1.
* `_ctm_tensor.py` (single-site path) — stays 1×1.
* `_ctm_tensor_c4v*.py` — C4v stays as-is.
* `_ctm_honeycomb_*.py` — honeycomb CTM stays.

## Plaquette site geometry

For each direction's move, the 2×2 plaquette around the **edge** being projected uses 4 site coords from `neighbors[]`. For the **left** move at coord `s`:

```
s_TL = s
s_TR = neighbors[s]["right"]
s_BL = neighbors[s]["bottom"]
s_BR = neighbors[s_TR]["bottom"]      # = neighbors[s_BL]["right"], lattice consistency
```

The four enlarged corners are built from each site's local env:
* `Q_TL = C1(s_TL) · T1(s_TL) · T4(s_TL) · a(s_TL)`
* `Q_TR = C2(s_TR) · T1(s_TR) · T2(s_TR) · a(s_TR)`
* `Q_BL = C4(s_BL) · T3(s_BL) · T4(s_BL) · a(s_BL)`
* `Q_BR = C3(s_BR) · T3(s_BR) · T2(s_BR) · a(s_BR)`

Adjacent quarters share chi-D² seams; e.g., Q_TL's right-side (chi_T1_right + D²_a_right) seam contracts with Q_TR's left-side (chi_T1_left + D²_a_left) seam in `top_M = Q_TL · Q_TR`. Right/top/bottom moves: cyclic permutation.

For 1-site unit cells all 4 plaquette positions are the same site (single tensor used 4 times — standard square-iPEPS case). For checkerboard, positions alternate A-B-B-A. For kagome 3-site, the plaquette is `(u, v, v, w)` with `(env_u, env_v, env_v, env_w)` — v appears twice (since both `u.right` and `u.bottom` map to v in `kagome().neighbor_map`); reusing v's env at both positions is correct (Tenax's per-sublattice multisite env is one env per name, not per coord).

## Projector formula

Following variPEPS `_fishman_horizontal_cut` + `_left_projectors_workhorse`, reimplemented from the Corboz et al. PRB 84, 041108(R) (2011) paper (clean-room — variPEPS is GPL-3.0):

For **left** move:

```
1. top_M = Q_TL_2D @ Q_TR_2D                       # (chi·D², chi·D²)
   bot_M = Q_BR_2D @ Q_BL_2D

2. (top_U, top_S, top_Vh) = gauge_fixed_SVD(top_M)
   top_S = where(top_S/top_S[0] >= eps, top_S, 0)
   (bot_U, bot_S, bot_Vh) = gauge_fixed_SVD(bot_M)
   bot_S = where(bot_S/bot_S[0] >= eps, bot_S, 0)

3. top_half = top_U * sqrt(top_S)[None, :]         # (chi·D², kept_top)
   bot_half = sqrt(bot_S)[:, None] * bot_Vh        # (kept_bot, chi·D²)

4. M_prime = bot_half @ top_half                    # (kept_bot, kept_top)
   (U_M, S_M, V_M_h) = truncated_SVD(M_prime, chi)
   S_inv_sqrt = 1 / sqrt(S_M)

5. P_top = top_half @ V_M_h.conj().T * S_inv_sqrt[None, :]   # (chi·D², chi)
   P_bot = S_inv_sqrt[:, None] * U_M.conj().T @ bot_half     # (chi, chi·D²)

6. Reshape P_top → rank-4 (chi_outer, D, D, chi_new) Tensor
   Reshape P_bot → rank-4 (chi_new, chi_outer, D, D) Tensor
```

Absorption (uses both projectors as a P-pair satisfying P_top^† · P_bot ≈ I):

```
new_C1 = contract(C1_self · T1_self, P_top_for_C1)
new_T4 = contract(T4_self · a_self, P_top, P_bot_perp)
new_C4 = contract(C4_self · T3_self, P_bot_for_C4)
```

Per the Corboz paper, the `(P_top, P_bot)` pair from the **left** plaquette SVD truncates the `chi_outer` of T4. The right side of the same plaquette generates a different projector pair that truncates the right side of T2 in the right move. variPEPS reuses the SVD twice (one plaquette → 2 projector pairs); we can do the same — see optimization note below.

Right/top/bottom moves: cyclic permutation of the template.

## Validation

Three test tiers in `tests/test_ctm_multisite_2x2_projector.py`:

**Tier 1 — Contract:**
```python
@pytest.mark.slow
def test_kagome_3site_multisite_at_d4_ad_optimum_matches_varipeps():
    """At saved AD-optimum, 2×2 multisite-CTM gives E/site ≈ -0.2554
    (matches variPEPS gate API)."""
    state = _load_d4_ad_optimum()
    rdms = _collect_ctm_rdms(state, chi=16, recipe="2x2")
    E = sum(jnp.einsum("ijkl,ijkl", rdms[b], H_pair) for b in BONDS) / 3
    assert abs(E - (-0.255359)) < 1e-3
```
Marked `slow` (depends on `logs/d4_ad_optimum.npz`).

**Tier 2 — Vanilla regression:**
```python
def test_2x2_multisite_matches_1x1_for_uniform_dense_state():
    """For a translation-invariant single-site state (no ill-conditioning),
    2×2 multisite agrees with 1×1 single-site CTM to chi-truncation tolerance.
    """
```

**Tier 3 — AD-FD:**
```python
def test_2x2_multisite_ctm_ad_matches_fd_at_d2():
    """jax.grad of CTM-energy via 2×2 sweep agrees with finite-difference
    at D=2 random state, chi=8. Wirtinger convention matched."""
```

**Existing regression — confirm green:**
* `tests/test_pess_3site_multisite_rdm_invariants.py` (D=2 SU-warmstart witnesses).
* Any other `tests/test_*multisite*.py`.

**Performance budget:**
* D=2 χ=16 single-site multisite-CTM convergence: 1×1 baseline ~3 s on CPU. Target 2×2 < 30 s (10× regression budget).
* Multisite kagome 3-site D=4 χ=16: target < 5 min on GPU. Raise concern if > 30 min.

## Risks and open questions

* **Performance regression on existing multisite callers.** Some current multisite users (kagome/honeycomb iPEPS in `examples/`) may have their L-BFGS optimizer settle to a slightly different (more accurate) local minimum after the projector switch. They should be re-validated.
* **AD path stability at D=2 χ small.** The bigger M' SVD has more near-degenerate singular values, which can hurt complex-128 Wirtinger AD. The existing `_truncated_SVD` multiplet handling should help; needs verification at Tier 3.
* **Reuse of left-SVD for right moves.** Optimization (variPEPS's `partial_unitary_mode`) — explicitly out of scope; first revision computes a fresh SVD per move direction.
* ~~**Symmetric tensor follow-up.**~~ Shipped (Issue #416). Per-charge-sector SVDs of M' via `tenax.linalg.svd`; `_gauge_fix_symmetric_svd` preserves the 2x2 closure convention per sector; AD-tracer fallback wraps the dense projector output as SymmetricTensor.

## Implementation order (preview)

1. Write `_build_enlarged_corner` for one direction (left's Q_TL); unit test against a hand-built reference.
2. Extend to all 4 quarter positions; unit test the seam contractions.
3. Write `_compute_2x2_projector` for left direction; unit test against variPEPS's `_left_projectors_workhorse` numerics on a fixed random tensor (no GPL code copy — just numerical comparison).
4. Extend to all 4 directions.
5. Write `_ctm_tensor_move_*_2x2` move functions with absorption.
6. Wire into `_ctm_tensor_sweep_multisite` with `recipe="2x2"` default.
7. Run Tier 2 vanilla regression; debug any divergence.
8. Run Tier 3 AD-FD; debug if gradient mismatch.
9. Run Tier 1 contract test against the saved AD-optimum.
10. Run existing regression tests.

## References

* Corboz, Penc, Mila, Lauchli, PRB 84, 041108(R) (2011) — 2×2 enlarged-corner CTMRG.
* Fishman et al., PRB 98, 235148 (2018) — Fishman cross-projector.
* Nishino, Okunishi, J. Phys. Soc. Jpn. 65, 891 (1996) — original CTMRG.
* variPEPS source (read-only reference, GPL-3.0): `varipeps/ctmrg/{absorption,projectors}.py`, `varipeps/contractions/definitions.py:645-`.
* Memory: `project_c3_floor_breach_smoking_gun.md`.
* Probes: `examples/dev/{p1_tenax_chi_scan, p2_varipeps_chi_scan, p3_tenax_projector_scan, d2_varipeps_rdm_compare}.py`; logs in `logs/`.
