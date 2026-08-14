# Simple update without stored lambdas — design

**Status:** draft for review · **Date:** 2026-08-14 · **Refs:** #667, #851, #863, #865, #869, #870, #875, #877, #878

## 1. Why

Simple update is the simplest method in the library and has been the least
reliable. Four separate defects have been filed and fixed against it in as many
months, and they are the same defect wearing different clothes:

| issue | what it actually was |
|---|---|
| #667 | a stored lambda re-absorbed that was already inside `Gamma` — the bond carried `lambda**1.5`, and the fixed point *was* the product state |
| #851 | two stored slots for four inequivalent bonds, so `num_imaginary_steps % 4` selected which bond's gauge was stamped on the lattice |
| #865 | the truncation constrained by a stored charge layout, so the SVD discarded the *largest* singular value |
| #869 | a stored lambda (`ones`) that was never true to begin with |

Every one of them is a defect in **lambda carried as mutable state between
updates**. None is expressible without that storage.

### The rule

> *Do not store bond lambdas for imaginary-time evolution. It is fine to store
> them for real time.*

An imaginary-time gate is **non-unitary**. It contracts some Schmidt directions
and not others, so applying it on bond X invalidates the stored spectra on every
*other* bond immediately. Real-time gates are unitary, the canonical structure
survives, and caching is legitimate.

This is where the reference implementations draw the line:

- **TeNPy** keeps a separate `TEBDEngine.update_bond_imag` whose docstring says
  it preserves the canonical form so one can sweep "**without using old singular
  values**"; `update_imag` warns the saved values "are not exactly correct after
  the update, since the non-unitary update on other bonds can change them", and
  calls `psi.canonical_form()` to repair it.
- **YASTN** carries *no* cached per-bond quantity across gate applications in any
  PEPS environment (`EnvNTU`/`EnvCTM`/`EnvBP`). It recomputes a `bond_metric`
  from the current tensors every time.

tenax does precisely the invalid thing. And the damage is already measured: BP
run on `main`'s *converged* two-lambda D=3 state reports `[1, 0.14243, 0.01130]`
against the stored `[1, 0.16586, 0.01564]` — 15% off on `lambda_2`, ~35% on the
tail (#869/#870). The shipped weights were never the Schmidt values they were
being used as.

### Why this is simpler, not merely different

Removing the storage deletes `_inv_lambda`, `safe_inv_lambda`, and the entire
absorb → gate → SVD → **pseudo-invert → divide** path. That machinery exists
*only* to undo storage. What remains is: gauge-fix, apply gate, truncate.

## 2. Design

One step, for each bond:

```
1. gauge-fix the state          BP on the loopy lattice
                                (== canonicalisation on a tree)
2. apply the Trotter gate       to the two site tensors sharing the bond
3. truncate                     bare SVD, in the gauge from step 1
```

There is no lambda to initialise, no far-bond bookkeeping, no staleness, and no
`steps % 4` dependence, because nothing is carried between steps. The gauge is
re-derived from the tensors each time because the gate just invalidated it.

**Bare SVD is the right truncation here, not a simplification.** In the BP gauge
on a tree, BP *is* the canonical form, so a bare SVD is the provably optimal
truncation. On the loopy square lattice it degrades to the standard simple-update
approximation — but against an *honest* gauge rather than a stale cached one.
v1 is therefore exact where exactness is available and standard where it is not,
with no metric-optimisation machinery to get wrong. (YASTN's iterative
`truncate_optimize_` in a real NTU/CTM metric is the strictly better truncation
and is explicitly **out of scope for v1**; see §8.)

### Cadence: every step, by construction

Not a tunable. The gate invalidated the gauge, so the gauge is recomputed. A
"re-gauge every N steps" knob would reintroduce exactly the staleness this
design exists to remove, under a new name.

### BP tolerance: 1e-6

Measured this session, dense, gauging a 60-phase SU state from its stored
lambdas, with every JIT trace warmed before the clock starts:

| D | SU cycle | BP @1e-12 | @1e-8 | @1e-6 | @1e-4 | @1e-2 |
|---|---|---|---|---|---|---|
| 2 | 17 ms | 31.3× | 18.5× | 21.4× | 12.3× | 5.9× |
| 3 | 1.18 s | 1.35× | 1.10× | 0.67× | 0.27× | 0.05× |
| 4 | 5.93 s | 0.19× | 0.08× | **0.08×** | 0.02× | 0.00× |
| 6 | 19.2 s | 0.03× | 0.02× | **0.01×** | 0.01× | 0.00× |

Cross-check on the methodology: the D=2 SU cycle here (17.4 ms) matches an
independently measured `ipeps()` run (0.9 s / 200 steps = 18 ms per 4-phase
cycle). An earlier version of this benchmark cycled `t % 4` over too few trials
and let first-time compiles for phases 2 and 3 land inside the measured window;
its ratios were wrong and are not the ones above.

BP iteration count is **flat-to-decreasing in D** (22, 33, 23, 15 at 1e-12;
9, 13, 9, 6 at 1e-6), so BP is not the part that scales badly. At D≥3 — where SU
is actually broken and where anyone cares — gauging every step costs ≤8% of an
SU cycle. The cost objection does not survive the measurement.

These are cold gauges (each trial re-gauges the same state), which is
representative of the per-step cost precisely because warm ≈ cold — see below.

1e-2 is nearly free (1 iteration) and rejected: residual 1e-3..9e-3 is not an
honest gauge, and honesty is the entire point.

The D=2 ratios (6–31×) are an artefact — the D=2 SU cycle sits on the
host-dispatch floor at 17 ms, so the ratio measures Python overhead. Absolute
worst case there is 0.5 s, and D=2 was never the broken case.

**Warm ≈ cold.** A hypothesis worth recording as refuted: re-gauging after one
gate does *not* converge in fewer iterations than gauging from scratch (D=3:
cold 32, warm 33). The truncating SVD perturbs the state enough that BP
re-converges. Do not budget for a cheap incremental gauge.

## 3. Architecture

Three independent simple-update implementations exist today:

| module | ansatz | notes |
|---|---|---|
| `ipeps_simple_update.py` | 2-site checkerboard | bosonic dense + symmetric; consolidated in #877 |
| `fermionic_ipeps.py` | **1-site** | its own absorb/invert/truncate; `A.l` and `A.r` are the same bond |
| `pess.py` | kagome simplex | different lattice — **out of scope** |

The rewrite unifies the first two behind one engine.

```
ipeps_su.py                     NEW — the engine
  su_step(state, gate, max_D)     gauge -> gate -> truncate, one bond
  su_evolve(state, gate, ...)     drive it; no lambda in the signature
  SUState                         site tensors + lattice topology. NO lambdas.

ipeps_gauge.py                  BP gauge, generalised from #870
  gauge_fix(state, tol=1e-6)      -> gauged state + BondWeights + info
```

`SUState` deliberately has no lambda field. That is the design in one line: if
there is nowhere to put a stale spectrum, there is no stale spectrum.

`bp_gauge_checkerboard` (#870) is already verified exact to 1.3e-15 and hardened
against the norm runaway (max-abs rescale, applied per iteration, plus a guard
that refuses to certify a non-state). It is the starting point for
`ipeps_gauge.py`, generalised from the 2-site checkerboard to also cover the
1-site ansatz.

## 4. Coverage

| path | v1 | notes |
|---|---|---|
| bosonic dense | yes | the main path |
| symmetric U(1)/Z_n | yes | works only since #875 |
| fermionic (#878) | yes | currently returns **exactly zero**; see §5.4 |
| PESS | no | different lattice; separate work |

## 5. Risks — stated, not buried

### 5.1 The 1-site layout constraint may be structural, not storage

`base_charges` pins the new bond's per-sector keep counts to the old layout.
#865 established it is *wrong* on the 2-site checkerboard, where `A.l` and `A.r`
are different bonds. But on the **1-site fermionic ansatz they are the same
bond**, so the index produced by the truncation becomes both `A.l` and `A.r` on
the next step, and its charge layout has to be self-consistent (#558/#559/#563).

**That constraint does not obviously disappear when storage does.** It is a
property of the ansatz's topology, not of caching. Removing stored lambda does
not by itself make a 1-site bond free to relabel.

This is the single largest risk in the design. It is *not* resolved by this
spec. Options, to be decided by measurement in Phase 1:

- keep an explicit layout constraint on the 1-site path only, applied at the
  index level rather than as a truncation constraint (i.e. permute/relabel the
  post-truncation bond into the canonical layout instead of restricting which
  singular values may be kept);
- treat the 1-site ansatz as a 2-site checkerboard with `A == B` and inherit the
  unconstrained truncation;
- keep 1-site fermionic on the existing code until the above is settled.

A **go/no-go gate** at the end of Phase 1 decides which. If none works, fermionic
drops to v2 and the spec is amended rather than the schedule being met by
shipping something unproven.

### 5.2 Fermionic BP has never been run

The BP gauge was built and verified for the bosonic checkerboard. Messages on a
graded tensor network must respect Koszul signs, and nothing has exercised that.
Phase 1 must prove `gauge_fix` is an exact gauge transformation on a fermionic
state (same invariance test #870 used: re-contract and compare, expect ~1e-15)
*before* any evolution is built on it.

### 5.3 #869's divergence is seed-dependent at D=3 and universal from D=4

Measured this session and **not recorded in #869**. Fraction of random seeds on
which the unmodified four-lambda baseline diverges, `dt=0.05`, 1200 steps:

| D | seeds diverging | |
|---|---|---|
| 2 | 0 / 3 | never broken |
| 3 | **1 / 3** (seed 0) | seed-dependent |
| 4 | **3 / 3** | universal |
| 5 | 3 / 3, flat spectra | universal |

Every measurement in #869, and every one of mine before this grid, used a single
seed. That is why D=3 results looked contradictory: at D=3 whether it diverges
is luck of the draw, while at D≥4 it always does. The failure becomes *more*
universal with D, not less.

Consequence for this work: **acceptance tests must sweep seeds *and* D.** Either
axis alone would have missed this — a D=3 single-seed run can pass on a broken
implementation, and a D=4 single-seed run cannot distinguish "always broken"
from "unlucky".

### 5.4 The fermionic path currently returns exactly zero (#878)

Filed this session as **#878**, and larger than a stale epsilon.

`fpeps()` collapses to exactly zero by step 10 at D=2, `dt=0.05` — measured, not
inferred. `min(lam_h)` falls 2.9e-01 → 2.0e-02 → 2.9e-03 → 1.1e-07 → 0, and
`lam_h`/`lam_v` both end `[0, 0]`. The default is 200 steps, so every shipped
`fpeps()` run is far past it.

Root cause is #667 verbatim, in a module #844 never touched: `sigma` is stored
as the new lambda at `fermionic_ipeps.py:291` *and* its square root is absorbed
into `Gamma` at `:298`, so the next `_absorb_lambdas` scales that bond by lambda
again and it carries `lambda**1.5`. #667's own title reads "bosonic →
near-classical rank-3; **fermionic → zero norm**"; only the bosonic half was
fixed, and the issue is closed.

Three further defects in the same functions (see #878): the additive epsilon at
`:291`/`:373` plus a third inside `jnp.sqrt(sigma + EPS)`; the shared bond
receiving `lam_h` from both ends because `_absorb_lambdas` scales all four legs;
and `fpeps()` building its physical tensor with full `lambda` rather than
`sqrt(lambda)`, so each bond carries `lambda**2`.

**Consequence for this design.** Phase 0 is a real fix, not a one-line change,
and until it lands there is **no trustworthy fermionic baseline to validate the
rewrite against** — the current path produces zero, so "the rewrite differs from
the old path" carries no information there. Either Phase 0 lands first, or
fermionic coverage waits for it.

Note also `|A|` reads a healthy 1.0 the whole way down, because
`_normalize_tensor` runs last. Any guard for this must assert a *spectrum*
(`min(lam) > 0`) or an energy, never a norm.

## 6. Testing

The bar is set by how this subsystem has failed before, not by coverage
percentage.

1. **Gauge invariance (exactness).** `gauge_fix` must not change the physical
   state: re-contract and compare, expect ~1e-15. Per tensor type, including
   fermionic (§5.2). This is the foundation — if the gauge is not exact,
   nothing above it means anything.
2. **Tree exactness.** On a 1D/tree topology, the BP gauge must reproduce the
   MPS canonical form and the lambdas must equal the Schmidt values to machine
   precision. This is the one place we have ground truth, and tenax already has
   `FiniteMPS.canonicalize` to check against.
3. **Reference spectra, swept over seeds *and* D.** D=3 →
   `[1, 0.165865, 0.015641]`, D=4 → `[1, 0.168753, 0.017325, 0.012895]`,
   independently measured on the dense path and again by the BP gauge in #870.
   **≥3 seeds × D ∈ {2,3,4}**, because §5.3 shows neither axis alone
   discriminates. Note D≥4 needs enough imaginary time to converge: 1200 steps
   at dt=0.01 is only 12 time units and reads as a failure that is really
   under-convergence.
4. **No `steps % 4` dependence.** #851's actual symptom: stopping the sweep at
   different phases must not change the physical state. The `dt = 0` identity-gate
   construction from #863 makes this exact with no tolerance to argue about.
5. **Not the product state.** #667's guard, retained: D=2 must reach ~−0.659,
   not −0.5.
6. **Rank honesty.** #667's other guard: a nominally-D=3 state must actually use
   its third bond direction (`lam_3 > 1e-3`).
7. **Mutation testing.** Every guard above must be shown to kill a faithful
   re-introduction of the defect it targets. This repo has shipped guards
   calibrated *against* bugs (three tests encoded #667 as expected behaviour),
   so a guard that has not killed a mutant is not yet a guard.

Registered `core` where it runs without a CTM.

## 7. Migration

The engine is built alongside the existing code, not in place.

1. **Phase 0** — fix **#878**, so fermionic has a baseline that is not zero.
   Four defects, not a one-line epsilon change (§5.4). This **blocks fermionic
   coverage**: until it lands, "the rewrite differs from the old path" carries
   no information on that path, because the old path produces zero. It does not
   block Phases 1–2, which are bosonic.
2. **Phase 1** — `ipeps_gauge.py`: generalise the #870 gauge to 1-site and to
   fermionic. Prove exactness (§6.1, §6.2). **Go/no-go on §5.1.**
3. **Phase 2** — `ipeps_su.py`: the engine, bosonic dense first, against the
   reference spectra swept over seeds and D.
4. **Phase 3** — symmetric, then fermionic (gated on Phase 0 and §5.1).
5. **Phase 4** — migrate `ipeps()` and `fpeps()`. The seven call sites are now
   one (#877), so this is a single edit rather than seven.
6. **Phase 5** — delete the old path, `_inv_lambda`, `safe_inv_lambda`, and the
   absorb/divide machinery.

Each phase is its own PR. The old path stays until Phase 5, so nothing regresses
while the new one is unproven.

## 8. Non-goals for v1

- Metric-optimised truncation (YASTN `truncate_optimize_` in an NTU/CTM metric).
  Strictly better; explicitly deferred.
- PESS / kagome.
- Real-time evolution. The rule permits stored lambda there, and that path is
  untouched.
- Making SU variational. It is not, and this does not change that.

## 9. Open questions for review

1. **§5.1** is the real risk. Is dropping fermionic to v2 acceptable if the
   Phase 1 gate fails, or should the spec commit to one of the three options now?
2. Should `SUState` be a new type, or should this reuse the existing iPEPS state
   container with the lambda fields removed?
3. Phase 0 (§5.4 / #878) — fix on `main` as its own PR before any rewrite work,
   or run it in parallel with Phases 1–2 since those are bosonic and unblocked?
