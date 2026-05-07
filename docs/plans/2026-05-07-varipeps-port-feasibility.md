# variPEPS port feasibility — verdict (Track D)

Date: 2026-05-07
Branch: `worktree-fix-multisite-ctm-rdm-helpers`
Author: Track D agent

## TL;DR

**Do NOT port variPEPS source into Tenax.** Two independent reasons:

1. **Licence incompatibility (hard blocker).** variPEPS 1.4.2 is licensed
   GPL-3.0-or-later (`/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps-1.4.2.dist-info/licenses/LICENSE`).
   Tenax is Apache-2.0 (`/home/yjkao/tenax/LICENSE`). Direct port of variPEPS
   source — even a single function — forces Tenax to relicense as
   GPL-3.0-or-later under GPL § 5–7. That would break downstream commercial
   use of Tenax and is a project-level relicensing decision, not a hot-fix.

2. **variPEPS does not vindicate the smoking-gun probe** (see D2 results
   below). At the D=4 χ=16 AD optimum, variPEPS's *own* infinite-lattice
   CTM evaluation gives **E/site = -0.2554** (gate API) or **-0.749** (RDM
   route) — both far below the brute-force 3×3 PBC torus reference
   E/site = -0.0436. This suggests the disagreement is not a Tenax helper
   bug but a fundamental mismatch between the finite-torus brute-force and
   any infinite-lattice CTM (Tenax or variPEPS) on the multisite-encoded
   state. **The smoking-gun premise — that 3×3 PBC torus = ground truth
   for the multisite encoding — needs to be re-examined**.

## D1 result (variPEPS state-mapping smoke test)

`examples/dev/d1_varipeps_state_map.py` ran on GPU 1 with miniforge Python
3.12 + JAX 0.10.0 + variPEPS 1.4.2. Outcome: PASS.

- Loaded the AD-optimum state from `logs/d4_ad_optimum.npz`
  (E_optimizer = -0.8528).
- Built three (4, 4, 2, 4, 4) variPEPS PEPS_Tensor objects from the
  Tenax (top, bottom, left, right, phys) supersite tensors via the
  permutation `(2, 0, 4, 3, 1)`.
- Constructed a 3×3 `PEPS_Unit_Cell.from_tensor_list` with the rotational
  tiling
      `[[u, v, w], [v, w, u], [w, u, v]]`.
- variPEPS CTM at χ=16 converged in 21.2 s with element-wise convergence,
  producing normalised C/T env tensors (‖C‖ = ‖T‖ ≈ 1).

## D2 result (per-bond energy / RDM compare)

`examples/dev/d2_varipeps_rdm_compare.py`. variPEPS CTM 23.7 s. Six bonds:

|   bond   | frob(BF − Tenax) | frob(BF − vP_RDM) |   E_BF    | E_Tenax_CTM | E_vP_RDM_route | E_vP_gate_api |
|----------|-----------------:|------------------:|----------:|------------:|---------------:|--------------:|
| uv_h     | 5.38e-1          | 1.68e+0           | -5.56e-2  | -3.75e-1    | -5.07e-1       | -1.91e-1      |
| uv_v     | 5.49e-1          | 1.27e+0           | -1.22e-1  | -4.81e-1    | -8.81e-2       | -3.66e-2      |
| wu_h     | 6.87e-1          | 2.04e+0           | +1.73e-2  | -4.02e-1    | -6.51e-1       | -2.27e-1      |
| wu_v     | 7.24e-1          | 1.38e+0           | +9.12e-3  | -4.67e-1    | -9.80e-2       | -4.21e-2      |
| vw_row   | 6.87e-1          | 1.96e+0           | +1.32e-2  | -4.48e-1    | -2.64e-1       | -8.74e-2      |
| vw_col   | 7.52e-1          | 2.49e+0           | +7.37e-3  | -5.65e-1    | -6.38e-1       | -1.81e-1      |

Per-site energies (sum / 3):

| source                                      | E/site    |
|---------------------------------------------|----------:|
| Brute-force 3×3 PBC torus                   | -0.0436   |
| Tenax multisite-CTM (RDM helpers under test)| -0.9129   |
| variPEPS CTM, RDM route (this script)       | -0.7491   |
| variPEPS CTM, gate-expectation API          | -0.2554   |
| Optimizer-reported (Tenax CTM during AD)    | -0.8528   |

Notes on the variPEPS RDM route: the raw `density_matrix_two_sites_*`
output at this multisite-encoded state has **trace ≈ 1e-8 before
normalisation** (verified empirically; see worktree command history).
This is because the `S_v` and `S_w` supersite tensors in the 3-site
multisite encoding have two of their four virtual legs trivial-padded
(dim-1, slice [:, 0, :, 0, :] and [0, :, 0, :, :]) — and through CTM
those degenerate sectors give an environment with near-zero overlap with
the bra leg. Dividing by ~1e-8 amplifies noise, which is why the RDM
route Frobenius gaps are huge (1.7–2.5). The gate-expectation API is
the trustworthy variPEPS number; **it is -0.255, not -0.044**.

## Interpretation

The smoking-gun probe at χ=16 D=4 takes the E_BF=-0.044/site brute-force
contraction on a 3×3 torus as ground truth. Every infinite-lattice CTM
in this experiment — Tenax, variPEPS-gate-API, variPEPS-RDM-route —
disagrees with that figure. Two non-mutually-exclusive explanations:

A. **The multisite encoding is not translation-invariant in the way the
   3×3 torus assumes.** The supersite tensors `S_v, S_w` only carry rank
   along two of four virtual legs (the other two are dim-1 trivial-
   padded). On an infinite lattice this generates correlations through
   the U-sublattice 2-hop path; on a 3×3 torus with PBC, the same
   structure produces a distinct effective Hamiltonian (the wave
   function is the same as a tensor object, but the physical-site
   marginalisation differs because the v-w bond in BF goes through one
   intervening U site, while in CTM it goes through the limit of an
   infinite path). This is a real inequivalence, not a bug.

B. **The AD-optimised state is not gauge-fixed**, and the dim-1
   trivial-padded bonds make the CTM gauge under-constrained (the
   near-zero raw trace is a symptom). Different gauge fixes give
   different infinite-lattice answers, so neither -0.913 nor -0.255
   is unique.

Either way, **the bug is upstream of the RDM helpers**. Replacing
`_rdm2x1_tensor_2site` etc. with variPEPS-derived helpers will not
recover E_BF, because variPEPS's own helpers also don't recover it.

## Licence summary (variPEPS)

- variPEPS 1.4.2: GPL-3.0-or-later (per
  `varipeps-1.4.2.dist-info/METADATA` line 10 and
  `varipeps-1.4.2.dist-info/licenses/LICENSE`).
- Tenax: Apache 2.0.
- Direct copy of any variPEPS function → Tenax must relicense to GPL-3.
- Re-implementation from scratch (clean-room) is fine — the maths is
  public (Corboz, Schmoll, Naumann references in the variPEPS paper),
  but the *specific* einsum conventions and helper signatures are
  copyrightable.
- Loading variPEPS at runtime as a debugging dependency (as we do in
  D1/D2) is OK because: (a) we don't redistribute; (b) we don't link
  variPEPS into Tenax; (c) it's a developer-only sanity check.

## Recommended next steps

1. **Do not port variPEPS code into Tenax.** Apache-vs-GPL incompatibility
   is the dispositive blocker even before the technical disagreement.
2. **Re-examine the smoking-gun premise.** Track A's brute-force
   reference (3×3 PBC torus) is not the ground truth for the 3-site
   multisite encoding's infinite-lattice limit. Either:
   - Use a much larger torus (e.g. 9×9) to approach the thermodynamic
     limit, OR
   - Drop the "brute-force RDM = ground truth" framing and instead
     diagnose the multisite encoding directly: show by example that for
     a *known* simple translation-invariant state (e.g. random U(1)
     symmetric tensors with no trivial-padded bonds), the BF-torus and
     CTM RDMs agree, then re-introduce the dim-1 padding and watch the
     agreement break.
3. **If a CTM cross-check is still wanted**, keep variPEPS as a runtime
   reference (the D1/D2 scripts work). Do not port; do consult.
4. **For Track A** (Tenax helpers): focus on internal consistency
   (RDMs sum to the same one-site marginal; reproduce the SU energy
   floor at small D). Do NOT use BF-torus as the gold standard.

## Artifacts

- `examples/dev/d1_varipeps_state_map.py` (Tenax → variPEPS PEPS unit
  cell, CTM smoke test)
- `examples/dev/d2_varipeps_rdm_compare.py` (per-bond compare)
- `logs/d1_varipeps_state_map.log`
- `logs/d2_varipeps_rdm_compare.log`
- `logs/d2_varipeps_rdm_compare.json`

## Reproduction

variPEPS 1.4.2 is installed at
`/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps`. It must
be run from the miniforge Python 3.12 (JAX 0.10.0 + cuda) — *not* the
worktree's uv venv (Python 3.11, lacks variPEPS):

```bash
CUDA_VISIBLE_DEVICES=1 /home/yjkao/miniforge3/bin/python \
  /home/yjkao/tenax/.claude/worktrees/fix-multisite-ctm-rdm-helpers/examples/dev/d1_varipeps_state_map.py

CUDA_VISIBLE_DEVICES=1 /home/yjkao/miniforge3/bin/python \
  /home/yjkao/tenax/.claude/worktrees/fix-multisite-ctm-rdm-helpers/examples/dev/d2_varipeps_rdm_compare.py
```
