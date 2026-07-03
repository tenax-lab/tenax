# 2-site joint split-CTM forward — design

- **Date:** 2026-07-02
- **Status:** approved design, ready for implementation plan
- **Umbrella issue:** #463 (unify split-CTM as canonical path; archive fused virtual-leg construction)
- **Related:** #392 (fermionic Koszul cross-terms), #605 (D≥3 hard-fusion charge-conjugation bug), #670/#674/#676 (fused 2×2 direction-dependent bond bookkeeping), #479/#485 (2-site mixed-env RDM trace floor)

## Problem

The split-CTM **forward** is single-site only. `ctm_split_tensor(A, chi)` converges one `A` as an
isolated 1×1 iPEPS — i.e. as if the lattice were all-`A`. The existing 2-site energy/RDM helpers
(`compute_energy_split_ctm_tensor_2site`, `_rdm1x2_split_tensor_2site`, `_rdm2x1_split_tensor_2site`)
stitch together **two independently-converged single-site environments**, which is the physically
wrong all-`A` / all-`B` approximation for a genuine AB checkerboard state. Every AD/optimizer entry
point raises `NotImplementedError` for multisite.

This design adds a **genuine joint 2-site forward** where site `A`'s environment absorbs site `B`'s
double-layer and vice versa (true bipartite checkerboard), plus the energy and AD/optimizer wiring
on top of the genuinely-coupled environment.

## Requirements (pinned)

- **Target:** genuine joint 2-site AB-checkerboard forward — `(env_A, env_B)` genuinely coupled, not
  two stitched single-site envs.
- **Backend:** all the way to fermionic, phased dense → SymmetricTensor → fermionic. Fermionic is
  tractable *because* the split formulation keeps each layer's contraction intra-layer, so the
  #392 cross-layer Koszul terms never form.
- **Stack:** forward + energy + AD (explicit + implicit) + `optimize_gs_ad` wiring. **No** chi-bump /
  chi-schedule this round (stays guarded off).
- **Correctness bar:** mirror the fused 2×2 path's conventions exactly and **parity-test each move**
  at the environment level (per direction), plus fixed-point energy parity — validated on
  **direction-dependent** inputs (A ≠ B, A.l ≠ A.r), never a uniform oracle.

## Chosen approach (Approach A): parallel split multisite driver

Build a driver that structurally mirrors the fused multisite CTM 1:1, but over split
(ket/bra-separated) double-layer tensors. Rejected alternatives:

- **B — minimal 2-site loop reusing single-site split moves:** the single-site moves use the **1×1**
  projector recipe + fused pair helper, a *different* C3↔T2 convention and projector formula than the
  fused 2×2 path we must mirror, so per-move parity breaks; also re-enters the #605 D≥3 hard-fusion
  bug the 2×2 path was built to avoid, and sets up nothing for multisite.
- **C — split branch inside the fused `_2plaq` functions:** bloats the already-large
  `_ctm_tensor_moves.py`, entangles two representations (and two AD graphs) in one hot loop against
  the isolation principle; the split ket/bra edges have different arity so the branches diverge anyway.

Approach A is the only option that satisfies the per-move-parity bar (each split move has a fused
twin), inherits the #605-safe leg-by-leg projector, and lands multisite as a near-free byproduct.

## Reference: the fused path being mirrored

All under `src/tenax/algorithms/`.

- Env: `dict[Coord, CTMTensorEnv]`, `CTMTensorEnv` NamedTuple at `_ctm_tensor_init.py:39`.
- Entries: `ctm_tensor_2site(A,B,chi,...)` (`_ctm_tensor_convergence.py:787`) → `{(0,0):A,(1,0):B}` +
  `CHECKERBOARD_NEIGHBORS` → shared driver `_ctm_tensor_multisite` (`:711`).
- Sweep: `_ctm_tensor_sweep_multisite` (`:274`), `recipe="2x2"` default. Direction order
  `("left","top","right","bottom")`. Two phases per direction: (1) build plaquette projectors per
  anchor via `_compute_plaquette_projector_pair`; (2) absorb neighbor double-layers per `s_dst`.
- Neighbor absorption is **coord arithmetic through the `neighbors` map** — `s_src =
  neighbors[s_dst][dir]`, absorb `double_layers[s_src]` using `envs_old[s_src]`.
- C3↔T2 convention (`c3_u <-> t2_d`, #674/#670): bottom absorb `_ctm_tensor_absorb_bottom_2plaq`
  (`_ctm_tensor_moves.py:711`) and enlarged-corner `bottom_right` builder
  (`_ctm_tensor_projector_2x2.py:246`).
- 2×2 projector applied **leg-by-leg via `_apply_proj_unfused`**, never the fused pair helper —
  specifically to dodge the #605 D≥3 hard-fusion charge-conjugation bug.

## Design

### 1. Module layout & public surface

Isolated in the `_split_ctm_tensor_*` family (no changes to the fused hot loop):

| File | Addition |
|---|---|
| `_split_ctm_tensor_convergence.py` | `ctm_split_tensor_2site(A, B, chi, ...)`; shared driver `_split_ctm_multisite(site_tensors, neighbors, chi, ...)`; `_split_ctm_sweep_multisite(envs, dls, neighbors, chi, chi_I, recipe="2x2", ...)` |
| `_split_ctm_tensor_moves.py` | `_split_ctm_absorb_{left,right,top,bottom}_2plaq(...)`; `_compute_split_plaquette_projector_pair(...)` |
| `_split_ctm_tensor_init.py` | `initialize_split_ctm_multisite_env(site_tensors, neighbors, chi, chi_I)` (per-coord reuse of the single-site builder) |
| `_split_ctm_tensor_energy.py` | feed `compute_energy_split_ctm_tensor_2site` the genuinely-coupled `(env_A, env_B)`; route it through the N=2 case of `compute_energy_split_ctm_tensor_multisite` |
| `_split_ctm_energy_ad.py` | 2-site AD entries; lift single-site-only `NotImplementedError` for the 2-site recipe |
| `ipeps_ad_policy.py` | allow `recipe=2site` under `fuse_virtual_legs=False`; keep chi-bump/schedule guards |

Environment: `dict[Coord, SplitCTMTensorEnv]` (`SplitCTMTensorEnv` at `_split_ctm_tensor_init.py:37`,
ket/bra split edges `T{1..4}_{ket,bra}`). `ctm_split_tensor_2site` builds `{(0,0):A,(1,0):B}` with
`CHECKERBOARD_NEIGHBORS` and delegates to the shared driver. `ctm_split_multisite` falls out of the
same driver later at near-zero cost.

### 2. Split double-layer tensors & neighbor absorption

Per coord, precompute a split double-layer pair `dls[coord] = (A_ket_dl, A_bra_dl)` — ket and bra
layers kept apart, never fused into `(u_k,u_b)`. This is what makes fermionic tractable: each layer
contracts within itself, so no cross-layer Koszul terms (#392) form.

Which tensor to absorb is coord arithmetic through `neighbors` (not explicit A/B args):

- LEFT into `s_dst`: `s_src = neighbors[s_dst]["left"]`; grow `envs_old[s_src]` against `dls[s_src]`
  (ket + bra halves); write `C1, T4_ket, T4_bra, C4` of `s_dst`.
- RIGHT → `C2, T2_*, C3`; TOP → `C1, T1_*, C2`; BOTTOM → `C4, T3_*, C3`.

A corner is grown against different edges per direction (e.g. C3 meets T2 in bottom/right but T3
elsewhere), matching the fused path; ket/bra arity is threaded consistent with the
`SplitCTMTensorEnv` field conventions.

### 3. Joint forward: driver, sweep, two-phase move

`_split_ctm_multisite` mirrors `_ctm_tensor_multisite`:

1. Build `dls = {coord: (ket_dl, bra_dl)}` and `envs = initialize_split_ctm_multisite_env(...)`.
2. Loop `max_iter`: `envs = _split_ctm_sweep_multisite(...)`; check per-coord corner-SV convergence
   (reuse `_corner_singular_values` / `_ctm_sv_diff` — same criterion as single-site + fused, so all
   three track the same fixed point).

One sweep iterates `("left","top","right","bottom")`, each direction in two phases (parallel of
`_ctm_tensor_sweep_multisite`):

- **Phase 1 — projectors:** per anchor coord, build the 2×2 plaquette
  `(TL=s, TR=s.right, BL=s.bottom, BR=s.right.bottom)` and compute the split projector pair
  `(P_top, P_bot)` into `projectors[s_anchor]`.
- **Phase 2 — absorb:** per `s_dst` (ordered by the fused `_sort_coords_for_direction`), call the
  matching `_split_ctm_absorb_*_2plaq` with projector halves from the two neighboring plaquettes;
  `_replace` the destination env's corner + ket/bra edge fields.

Projectors are applied **leg-by-leg** (split analogue of `_apply_proj_unfused`), never a pre-fused
pair helper — inheriting the #605 D≥3 hard-fusion-safe behavior the single-site split path (1×1
fused-pair helper) lacks.

### 4. The four split absorb functions & the C3↔T2 convention

Each `_split_ctm_absorb_*_2plaq` is the split twin of a fused `_..._2plaq`. The fused bottom absorb
(`_ctm_tensor_moves.py:711`, #674/#670) encodes the load-bearing convention:

```
C3.c3_u <-> T2.t2_d      # relabel c3_u -> t2_d, contract C3·T2, then apply P
```

The split version reproduces this on **both layers**: grow `C3` against `T2_ket` and `T2_bra` with
the same `c3_u <-> t2_d` pairing, keeping the interlayer bond `chi_I` intact. This is the convention
flip flagged in the prior-session note: the split bottom move switches from the single-site
`c3_l <-> t2_d` to the 2×2 `c3_u <-> t2_d` recipe, validated against a **direction-dependent** pair
(A ≠ B, A.l ≠ A.r), never a uniform oracle.

Direction-dependent asymmetry lives implicitly in which corner/edge pair each function grows
(left→C1/C4+T4; right→C2/C3+T2; top→C1/C2+T1; bottom→C4/C3+T3), matching the fused path. The
fermionic reference is `_ctm_tensor_absorb_bottom_2plaq_fused` (#674) — but in the split formulation
no fused-leg variant is needed; the same ket/bra absorb serves bosonic and fermionic, with Koszul
signs staying intra-layer.

### 5. The split plaquette projector

`_compute_split_plaquette_projector_pair`:

1. Build the four enlarged corners from the split double-layer plaquette — each corner grown from
   ket + bra halves with the interlayer bond `chi_I` contracted *within* the corner (a genuine
   double-layer object, assembled without ever fusing virtual legs; reuse `_doublelayer_grown_corner`
   from the single-site split path).
2. Feed the same Fishman cross-projector math (`_compute_2x2_projector`) → `(P_top, P_bot, eps_T,
   smallest_S)`.

Once the enlarged corners are built, the projector is **identical to the fused one** (the corner is a
contracted double-layer object either way), so `(P_top, P_bot)` is parity-testable against
`_compute_plaquette_projector_pair` up to gauge. Truncation (the `chi` cut) happens on the fused-corner
spectrum exactly as in the fused path — so energy parity is structurally guaranteed if the corners match.

### 6. Energy / RDM on the coupled environment

The 2-site RDM/energy helpers already exist: `_rdm1x2_split_tensor_2site`
(`_split_ctm_tensor_energy.py:673`), `_rdm2x1_split_tensor_2site` (`:837`),
`compute_energy_split_ctm_tensor_2site` (`:1133`). This design changes only their **input**: they now
receive the genuinely coupled `(env_A, env_B)` from `ctm_split_tensor_2site`; the contraction code is
unchanged.

- The adversarial low-trace fallback delegating to `_split_env_to_tensor_standard` + the fused RDM
  (#479/#485, lines 696–700) stays as a numerical safety net but should rarely fire on a true fixed
  point. Add a test asserting the native split path (not the fallback) is exercised on
  well-conditioned inputs.
- Route `_2site` through `compute_energy_split_ctm_tensor_multisite` (`:1177`, already sums NN bonds
  over `{coord: env}`) as its N=2 instantiation, to avoid a divergent second code path.

### 7. AD wiring (explicit + implicit + optimizer policy)

Mirror `_split_ctm_energy_ad.py`, with the fixed-point map now the joint sweep over `(env_A, env_B)`:

- **Explicit AD:** differentiate through a fixed number of `_split_ctm_sweep_multisite` calls; energy
  closes over `(A, B)` and the coupled envs. Warm-start / correctness path.
- **Implicit AD:** treat `envs* = F(envs*; A, B)` as the joint fixed point; apply the
  implicit-function theorem with the same GMRES/adjoint machinery as single-site/fused, but the
  residual and adjoint solve are on the coupled `(env_A, env_B)` pytree. No new linear-algebra
  primitives.
- `_extract_single_site` and the `NotImplementedError("...single-site...")` guards
  (`_split_ctm_energy_ad.py:26–28, 51, 232–233`) gain a 2-site branch; single-site guards remain the
  fallthrough for N>2 (multisite still deferred).

`ipeps_ad_policy.py`: `validate_split_ctm_config` (lines 60–77) relaxes to **allow the 2-site
checkerboard recipe under `fuse_virtual_legs=False`**, keeping rejections for `chi_auto_bump`,
`chi_ramp`, `ctmrg_heuristic_increase_chi`, custom `energy_fn`, and `cg_gates`.
`optimize_gs_ad(..., fuse_virtual_legs=False, recipe=2site)` dispatches to the new 2-site split AD.

### 8. Testing strategy

Pinned against the fused 2×2 path as oracle, on **direction-dependent** inputs (A ≠ B, A.l ≠ A.r
random seeds), never uniform tensors.

- **Tier 1 — per-move env parity:** for each direction and each `s_dst`,
  `_split_ctm_absorb_*_2plaq` env == `_ctm_tensor_absorb_*_2plaq` env to ~1e-10, compared after
  `_split_env_to_tensor_standard` (contract ket/bra → fused edge) up to interlayer gauge. Plus a
  standalone projector-parity test: `_compute_split_plaquette_projector_pair` vs
  `_compute_plaquette_projector_pair`.
- **Tier 2 — fixed-point energy parity:** converge `ctm_split_tensor_2site(A,B)`; assert energy ==
  fused `ctm_tensor_2site` energy to ~1e-10 across D∈{2,3,4}, χ∈{4,8,16}.
- **Tier 3 — AD parity:** 2-site split energy gradient matches (a) fused-path gradient and (b) finite
  difference to ~1e-6–1e-8, for explicit and implicit AD (reuse the existing FD-parity harness).
- **Tier 4 — backend phasing:** the four-tier suite per backend (dense → symmetric → fermionic);
  fermionic adds an ED cross-check on a small t-V lattice (the #463 Phase-1 acceptance pattern),
  asserting the native split path (not the standard-RDM fallback) is exercised.
- **Guard-lift regression:** `optimize_gs_ad(fuse_virtual_legs=False, recipe=2site)` no longer raises;
  chi-bump/schedule still do.

### 9. Implementation phasing

Each phase is independently landable (own PR, own green suite), gated on the prior phase's parity:

- **Phase 0 — scaffolding:** multisite env init + `dls` builder + `_split_ctm_multisite` /
  `_split_ctm_sweep_multisite` skeletons wired to reuse single-site moves as a smoke test (no 2×2
  absorb yet). Lands the `dict[Coord, ...]` plumbing.
- **Phase 1 — dense forward (bosonic):** the four `_split_ctm_absorb_*_2plaq` +
  `_compute_split_plaquette_projector_pair`; Tier-1 + Tier-2 green on dense. Core of the work.
- **Phase 2 — dense AD:** explicit + implicit 2-site AD, policy guard-lift, `optimize_gs_ad`
  dispatch; Tier-3 green.
- **Phase 3 — SymmetricTensor:** per-block layout for split enlarged corners + absorbs; re-run
  Tiers 1–3 on U(1)/Zn.
- **Phase 4 — fermionic:** FermionParity/FermionicU1; intra-layer Koszul only; Tier-4 ED cross-check
  on t-V. Should be mechanically small if the split architecture holds — the thesis of #463.

### 10. Risks & open questions

- **Interlayer-gauge in per-move parity:** the split env carries a `chi_I` interlayer bond the fused
  env lacks; parity comparisons must contract it out first (`_split_env_to_tensor_standard`), and
  residual gauge freedom in degenerate SV subspaces may force comparing contraction invariants
  (energies, corner spectra) rather than raw tensors on some moves — same class as #425.
- **Coupled fixed-point convergence:** the joint `(env_A, env_B)` update may converge differently than
  two independent single-site loops; `min_iter` / `conv_tol` carry over but may need retuning. Define
  "converged" on the joint corner spectra.
- **Implicit-AD adjoint on the coupled system:** larger fixed-point pytree; GMRES conditioning on the
  joint operator is unproven at χ≥16 — the single-site adjoint is the closest precedent, not a
  guarantee.
- **Symmetric split enlarged-corner block layout (Phase 3):** #605 hard-fusion charge-conjugation bug
  motivated leg-by-leg application; the split enlarged-corner assembly must stay clear of hard fusion
  at D≥3.
- **`chi_I` default:** stays `chi_I = chi` (lossless) as in single-site; smaller-`chi_I` truncation is
  out of scope.

## Acceptance

- [ ] Tier-1 per-move env parity green (all 4 directions, both sublattices) on a direction-dependent
  pair, dense.
- [ ] Tier-2 fixed-point energy parity to ~1e-10 across D∈{2,3,4}, χ∈{4,8,16}, dense.
- [ ] Tier-3 explicit + implicit AD gradient parity (fused + FD) to ~1e-6–1e-8, dense.
- [ ] `optimize_gs_ad(fuse_virtual_legs=False, recipe=2site)` runs a bipartite Heisenberg optimization
  end-to-end.
- [ ] SymmetricTensor Tiers 1–3 green (U(1)/Zn).
- [ ] Fermionic Tier-4 ED cross-check on a small t-V lattice; native split path exercised.

## Out of scope

- chi-auto-bump / chi-schedule on the split path (stays guarded off; separate follow-up).
- General multisite (>2 sites) forward — the driver supports it, but this round validates only 2-site.
- Smaller-than-`chi` interlayer truncation.
- The dense `ctm_2site()` legacy SU path (separate retirement track).
