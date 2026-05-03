# Liao 2017 PESS replication audit

**Date:** 2026-05-03
**Branch:** `feat/spin1-xxz-pess-ad`
**Status:** Design approved — ready for implementation plan.

## Why

Before committing to M2b (honeycomb-native CTM + honeycomb SU + inter-triangle
`energy_fn`), verify that we have not misinterpreted the kagome-PESS reference
we are matching. The "AD stalls at E/site = −0.343 vs Liao −0.4324" narrative
in `project_pess_ad_stalls_d4.md`, and the "SU now matches Liao 2019's
−0.420 at D=4" claim in `project_pess_su_collapse_bug.md`, both anchor on a
paper attribution and a target value that this audit found to be inaccurate.

## Audit findings

1. **Reference paper miscited.** Our existing benchmark
   (`examples/kagome_spin12_pess_ad_benchmark.py`) and the two project memory
   entries above cite *Liao et al., PRX 9, 031041 (2019)* ("Differentiable
   Programming Tensor Networks") as the source of the kagome target
   −0.4378 / −0.4324 / −0.420. That paper does **not** contain a kagome
   calculation; its Section IV.B is square-lattice S=½ Heisenberg iPEPS +
   AD with D ∈ {2, …, 7}, χ ∈ {30, 50, 80, 100, 144, 160}, target E/site ≈
   −0.66944.
2. **Correct reference is Liao 2017** — Liao et al., PRL 118, 137202 (2017),
   arXiv:1610.04727, "Gapless spin-liquid ground state in the S = 1/2
   kagome antiferromagnet." 3-PESS ansatz; SU for D ≤ 25; FU for D ≤ 13;
   no AD; HOSVD on the triangle, same kernel as PR #387.
3. **Liao 2017 measures energy by MPS projection, not CTM.** The optimized
   PESS is projected onto a 1D MPS basis with `D_mps ≈ 4·D²` and the
   expectation value is taken in MPS form. There is no Convention-C
   supersite anywhere in their pipeline. Our `build_pess_loss` energy
   measurement (PESS → square supersite → CTM) is a Tenax-specific
   shortcut, not a Liao replication.
4. **Recorded D=4 match is illusory.** Liao 2017 Fig 1(a) shows 3-PESS SU
   at D=4 between roughly −0.428 and −0.430. Our `−0.420` at D=4 is
   ~0.008–0.010 *above* Liao's value. The gap is almost certainly the
   Convention-C + CTM measurement, not the SU kernel — the kernel matches
   Liao to the line of code (HOSVD on the triangle).
5. **AD-stall narrative is incomplete.** Liao 2017 has no AD baseline. Our
   AD result at E/site = −0.343 is failing to recover even the SU energy
   (read through the same CTM probe), and the SU energy itself is biased
   ~0.010 above Liao via that same probe. So Convention-C is rate-limiting
   the *upper* end of every number we currently report, with AD adding its
   own ~0.077 gap on top of that. M2b's case sharpens accordingly: it is
   not just an AD path; it is the only path that delivers an
   energy-measurement bias below Liao's published numbers.

## Design

### Goal

Run a controlled SU-only D-sweep that **separates the SU error from the
CTM-via-Convention-C measurement error**, and compare both against Liao
2017 Fig 1(a). The outcome of this run determines whether M2b is on the
critical path or whether Convention-C is acceptable.

### Run config

- **Bond dimensions:** D ∈ {4, 6, 8, 10}.
- **CTM bond dimension (P2 only):** χ = 2·D².
- **dt schedule:** `(0.1, 200) → (0.01, 200) → (0.001, 100) → (0.0001, 100)`.
- **Init:** `IPESSState.random` with fixed seed 0 (init scheme is
  irrelevant per Liao 2017).
- **Hamiltonian:** `kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)` —
  isotropic spin-½ Heisenberg.

### Two energy probes per D

- **(P1) Local / Husimi-tree energy** — contract one up-triangle's 3-site
  RDM using only the bond-λ mean-field gauges, no environment.
  Implementation: ~30 lines, pure JAX, no CTM. This is the cheapest
  SU-consistent estimate; it sits slightly above Liao's MPS-projection
  number but tracks its D-dependence.
- **(P2) Convention-C + CTM** — our existing `build_pess_loss` path
  (PESS → square supersite → CTM). χ = 2·D².

### Diagnostic table — three possible outcomes

| Outcome | Implication | Next step |
|---|---|---|
| P1 ≈ Liao, P2 above Liao | Convention-C is the bias source | Proceed with M2b decisively |
| P1 above Liao | Residual SU-kernel bug | Re-open SU audit *before* M2b |
| Both ≈ Liao | Convention-C is fine | Deprioritize M2b; revisit AD-stall under different lens |

### Out of scope

- **Full update.** Liao 2017's FU column gives slightly lower E than SU at
  the same D, but it is not on the critical path for the M2b decision.
  Defer.
- **MPS-projection energy probe.** Faithful to Liao but ~1 week of
  work; P1 covers >90 % of the diagnostic value at <1 day.
- **AD experiments.** Separate issue. Settle the SU baseline first.

### Deliverables

1. New helper `pess_local_energy(state, h_intra)` in
   `src/tenax/algorithms/pess.py`. Pure einsum + bond-λ contraction.
2. New benchmark script
   `examples/kagome_spin12_pess_liao2017_replication.py`. Runs the D-sweep,
   writes JSON `{D, e_p1, e_p2, liao_target, wall_seconds}`. Liao targets
   read off Fig 1(a) of arXiv:1610.04727 and stored as a constant table.
3. Smoke test in `tests/test_pess_local_energy.py` confirming P1 and P2
   agree on a trivial state (D=2, identity gauges) and that P1 lies between
   Liao's value and our P2 number on a SU-converged state.
4. Memory/docstring corrections (Task #7):
   - `examples/kagome_spin12_pess_ad_benchmark.py` docstring → cite Liao
     2017, not Liao 2019.
   - `project_pess_su_collapse_bug.md` → "matches Liao 2017 Fig 1(a) at
     D=4" with the actual target (~−0.428).
   - `project_pess_ad_stalls_d4.md` → "AD stalls at −0.343; SU-via-CTM
     itself is ~0.010 above Liao 2017 due to Convention-C" framing.
   - Add new memory: `project_liao2017_replication.md` with the D-sweep
     results once the run lands.

### Acceptance criteria

- All 4 D values ran to completion in CI-compatible time (~30 min wall
  total on a CPU runner).
- The JSON output is committed under `examples/` (small, ~1 KB).
- The smoke test passes locally and is marked `@pytest.mark.algorithm` so
  it does not block the `core` CI bucket.
- The 3-row outcome table in this doc is annotated in the resulting PR
  body with which row we landed on.

## Non-goals

- This is **not** a "Liao replication paper" in the publication sense.
  It is a diagnostic to settle the M2b decision and clean up two
  inaccurate memory entries.
- We are not changing the SU kernel, the CTM, or any AD path in this
  work. Only adding the P1 helper, the benchmark script, and the smoke
  test.
