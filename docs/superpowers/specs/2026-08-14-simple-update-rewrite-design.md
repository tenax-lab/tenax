# Simple update without stored lambdas — design

**Status:** accepted · **Date:** 2026-08-14, decisions 2026-08-15 · **Refs:** #667, #851, #863, #865, #869, #870, #875, #877, #878, #879, #881

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

This is the expensive choice at small D — four BP solves per cycle, ~3.7× at
D=3 — and it is made with that cost in view rather than in ignorance of it.
At the **default** D=2 the naive cost is ~85×, which is not acceptable and is
not accepted: it is host-dispatch overhead in an untraced Python loop, and
tracing it is a Phase 1 gate. See the measurements below.

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

**The table is per BP solve; the cadence needs four of them per cycle.** §2
re-gauges before *each bond*, and a checkerboard cycle has four, so the real
overhead is 4× the figures above:

| D | per solve @1e-6 | **per 4-bond cycle** | total slowdown |
|---|---|---|---|
| 2 | 21.4× | 85.6× | **default D** — see the gate below |
| 3 | 0.67× | **2.68×** | ~3.7× |
| 4 | 0.08× | **0.32×** | ~1.3× |
| 6 | 0.01× | **0.04×** | ~1.04× |

So the honest statement is *not* "≤8%". At **D=3 the gauge dominates the step**,
costing about 2.7 SU cycles per cycle of evolution. It becomes cheap from D=4
(32%) and negligible by D=6 (4%).

That is still the right trade at D≥4, and D≥4 is where large-D work happens. At
D=3 it is a real ~3.7× cost, accepted deliberately: a D=3 run is seconds either
way, and §5.3 shows D=3 is precisely where the stored-lambda scheme is least
trustworthy, so paying for an honest gauge there is the point rather than a
regrettable side effect.

BP iteration count is **flat-to-decreasing in D** (22, 33, 23, 15 at 1e-12;
9, 13, 9, 6 at 1e-6), so BP is not the part that scales badly — the overhead
shrinks with D because the SU cycle grows, not because BP gets cheaper.

These are cold gauges (each trial re-gauges the same state), which is
representative of the per-step cost precisely because warm ≈ cold — see below.

1e-2 is nearly free (1 iteration) and rejected: residual 1e-3..9e-3 is not an
honest gauge, and honesty is the entire point.

### D=2 is the default, and 85× is not an artefact anyone can ignore

An earlier draft called the D=2 ratio an artefact of the host-dispatch floor
and moved on. The diagnosis is right and the dismissal is not: **`D = 2` is the
default in both `iPEPSConfig` and `FPEPSConfig`**, so this is the path most
callers are on.

The arithmetic, from this document's own numbers: four mandatory solves at
371 ms is **1.48 s per cycle** against a 17 ms SU cycle, so a default 200-step
run goes from **~0.9 s to ~74 s**. (The "worst case 0.5 s" in the earlier draft
was a *single* solve, not a run.) Shipping that as the default would be a severe
regression regardless of what the ratio is *made of*.

**It is Python, and it is fixable.** BP's cost per *iteration* is flat in D —
41 ms at D=2, 61 at D=3, 52 at D=4, 32 at D=6 — which is the signature of host
dispatch rather than arithmetic, on tensors that at D=2 are a few dozen numbers.
`ipeps_bp_gauge.py` today is a pure Python loop over eager ops: no `jax.jit`, no
`lax.scan`, no `lax.while_loop`.

**Phase 1 gate, therefore:** the BP iteration must be traced (`lax.while_loop`
on the residual, or `scan` over a fixed iteration budget) before the engine is
built on it, and the D=2 per-solve cost must come down to the point where a
default 200-step run is within ~2× of today's. If it cannot, the cadence
decision in §2 has to be revisited for small D rather than the cost quietly
accepted — and that revisit is a design change requiring its own review, not an
implementation detail.

**Warm ≈ cold.** A hypothesis worth recording as refuted: re-gauging after one
gate does *not* converge in fewer iterations than gauging from scratch (D=3:
cold 32, warm 33). The truncating SVD perturbs the state enough that BP
re-converges. Do not budget for a cheap incremental gauge.

## 3. Architecture

Three independent simple-update implementations exist today:

| module | ansatz | notes |
|---|---|---|
| `ipeps_simple_update.py` | 2-site checkerboard | bosonic dense + symmetric; consolidated in #877 |
| `fermionic_ipeps.py` | 1-site on `main`; **2-site** in #881 | had its own absorb/invert/truncate; #881 routes it through the shared sweep |
| `pess.py` | kagome simplex | different lattice — **out of scope** |

The rewrite unifies the first two behind one engine.

```
ipeps_su.py                     NEW — the engine (internal, see below)
  _su_step(state, gate, max_D)    gauge -> gate -> truncate, one bond
  _su_evolve(state, gate, ...)    drive it; no lambda in the signature
  _SUState                        site tensors + lattice topology. NO lambdas.

ipeps_gauge.py                  BP gauge, generalised from #870
  gauge_fix(state, tol=1e-6)      PUBLIC -> gauged state + BondWeights + info
```

**API surface, decided:** `_su_step`, `_su_evolve` and `_SUState` are **internal**
for v1 — hence the leading underscores above — matching the private
`_simple_update_checkerboard_sweep` they replace — v1 is a drop-in behind
`ipeps()` and `fpeps()`, and committing public API to an unproven engine buys
nothing. `gauge_fix` is **public**, because it generalises the already-exported
`bp_gauge_checkerboard`; Phase 4 therefore carries an explicit task to add it to
`src/tenax/__init__.py` (`__all__`) and to `README.md`, per the repository rule
that new public API must be exported and documented. Promoting the engine later
is a deliberate decision with its own export task, not a side effect of naming.

`_SUState` is a **new type**, not the existing iPEPS container with the lambda
fields removed. Deliberately: the point of the design is that there is nowhere
to put a stale spectrum, and a type that *used* to have lambda fields invites
them back the first time something wants to cache one. A new type also lets the
old container keep its lambdas for the real-time path, where storing them is
legitimate, instead of forcing one container to mean two different things.

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
| fermionic (#878/#881) | yes | returned **exactly zero**; five of six defects fixed in #881, energy still unvalidated (#879). See §5.4, §5.5 |
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

A **go/no-go gate** at the end of Phase 1 decides which.

**Decided: if none works, fermionic drops to v2.** Authorised in advance rather
than reopened under schedule pressure — the failure mode to avoid is meeting a
date by shipping a fermionic path nobody has shown correct, which is how #878
came to exist. Dropping to v2 leaves the existing code in place (repaired by
#881) and costs nothing that works today.

**Scope of that gate, precisely.** The constraint above is *1-site-specific*:
it exists because `A.l` and `A.r` are the same bond there. #881 moves fPEPS to
a 2-site checkerboard, where they are different bonds, so a failure of the
1-site experiment does **not** by itself condemn the 2-site fermionic engine,
and the v2 drop must not be triggered by it alone.

There is, separately, a measured reason the 2-site fermionic path may still
need a layout constraint: removing the pin from `_truncation_base_charges`
collapses the 2-site sweep at **every** D tested, including the D=2 and D=4
that otherwise work (#881). Why a 2-site path needs it, when the stated
justification is 1-site-only, is **not understood** and is its own Phase 1
question. That — not the 1-site experiment — is what gates fermionic v1.

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

#### Independently confirmed on the fermionic path

The same behaviour appears in `fermionic_ipeps.py`, an implementation that
shares **nothing** with the bosonic one except the stored-lambda state model
(#878/#881). Seeds 0–4, 600 steps, dt=0.05, counting seeds whose bond spectrum
survives:

| D | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| seeds alive | 4/5 | **2/5** | 4/5 | 4/5 |

Every bond dimension has both surviving and dying seeds, so "D≥3 is broken" is
not the shape of the defect on either path. Seed 0 happens to die at D=3 and
D=6 and live at D=2 and D=4, which is what made it look like a bond-dimension
bug for as long as only seed 0 was measured.

**This is the strongest evidence in this document.** One implementation's basin
failure is a hypothesis about that implementation. The *same* failure arising
independently in two implementations whose only common element is lambda carried
across a non-unitary sweep is evidence about the state model itself — which is
what this design removes. Four hypotheses were refuted reaching that conclusion
on the fermionic side alone (misplaced layout pin, odd D, unbalanced virtual
charges, "D=3 is special"), every one of them by a control arm.

### 5.4 The fermionic path returned exactly zero (#878, fixed in #881)

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

**Status: five of the six are fixed in #881**, which also moves fPEPS to a
2-site checkerboard — the 1-site ansatz discarded `Vh` at both SVD sites, so `A`
received the left/top half of every gate and never the right/bottom, and no
1-site tensor can represent the t-V charge-density wave anyway (measured
sublattice gap 0.27 to 1.4). D=2 now gives `lam_h = [1, 0.678]` against `[1, 0]`.
What remains is the seed-dependent basin failure in §5.3.

**Consequence for this design.** Phase 0 shrinks: #881 supplies the fermionic
baseline this document said was missing, so Phase 3 no longer waits on a fix
that does not exist. It does *not* supply a trustworthy fermionic **energy** —
see §5.5 — so acceptance for the fermionic path must be stated on bond spectra
and the variational bound, not on an energy value.

Note also `|A|` reads a healthy 1.0 the whole way down, because
`_normalize_tensor` runs last. Any guard for this must assert a *spectrum*
(`min(lam) > 0`) or an energy, never a norm.

### 5.5 No fermionic CTM energy is asserted against a reference (#879)

The ED machinery exists — `_build_tv_hamiltonian_2x2_pbc` plus `eigvalsh` gives
a 4-site 2×2 PBC Jordan-Wigner ground state, **E_gs/site = −0.5** at t=1 — and
`test_tier7_variational_bound_check` uses it properly, on 7 seeds. But it guards
`reference_energy_2x2_pbc`, the direct contractor path. The only place the CTM
energies meet that reference, `test_tier5_compare_to_ipeps_ctm_energy`, prints
four numbers and asserts nothing.

So `compute_energy_ctm_tensor`, `compute_energy_split_ctm_tensor` and
`compute_energy_split_ctm_tensor_2site` can each return any finite real number
on a fermionic state unchallenged. #881 measures ≈ −5e−5 on a healthy spectrum
with a full-rank corner, and that **cannot currently be distinguished from a
pre-existing defect**.

**Consequence for this design.** A fermionic energy cannot be an acceptance
criterion until #879 lands.

**And tier 7's bound is not the fix**, though an earlier draft of this section
said it was. That bound is stated for a state on the *four-site 2x2 PBC torus*;
the CTM energies are infinite-lattice quantities, and
`test_tier5_compare_to_ipeps_ctm_energy` says so itself -- "NOT expected to
match exactly ... the CTM energy is for the infinite lattice while the
reference is for a 2x2 PBC torus". A valid infinite-lattice energy may lie
*below* a finite-cluster ground state, and a finite-chi CTM contraction is not
a strict variational bound in the first place. Applying it would reject correct
implementations.

What is applicable, in increasing strength:

- **Cross-path agreement, between paths of the same arity.** Needs no physics
  reference, and is exactly the check whose failure was #392 — but it only
  means anything when both sides compute the same observable:
  - *2-site* (the fPEPS target after #881): `compute_energy_ctm_tensor_2site`
    against `compute_energy_split_ctm_tensor_2site`, on the same split
    environment converted to standard form.
  - *1-site*: `compute_energy_ctm_tensor` against
    `compute_energy_split_ctm_tensor`, as a separate parity test.

  Do **not** require the 1-site and 2-site functions to agree. The 1-site APIs
  repeat a single tensor; on a CDW state, where `A != B` by construction and
  the measured sublattice gap reaches 1.4 (#881), they are not computing the
  same quantity and equality would be meaningless.
- **The finite-cluster bound on the finite-cluster path only** -- tier 7 keeps
  guarding `reference_energy_2x2_pbc`, which is the quantity it is valid for.
- **chi-convergence**: the CTM energy must settle as chi grows, and a value that
  moves with chi is not yet an answer.
- **A magnitude anchor on a state whose energy is hand-computable** (a product
  state), which is what catches "approximately 0" -- the actual symptom in
  #881.

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
3. **Acceptance is on energy and self-consistency, NOT on a reference spectrum.**

   An earlier draft of this section asked the rewrite to reproduce
   `[1, 0.165865, 0.015641]` at D=3 and `[1, 0.168753, 0.017325, 0.012895]` at
   D=4. **Those are the stale stored spectra**, and §1 of this same document
   says they were never the Schmidt values they were used as — BP on that state
   reports `[1, 0.14243, 0.01130]` and `[1, 0.14534, 0.01258, 0.01017]`. Making
   them acceptance targets would reject a correct BP-gauged result, or invite
   tuning the rewrite back toward the behaviour it exists to delete.

   Nor are the BP numbers the right target: they are the BP-consistent spectra
   *of `main`'s converged two-lambda state*, and the rewrite converges to its own
   fixed point. **We do not know the correct spectrum a priori, and the spec must
   not pretend otherwise.** What can be asserted without one:

   - **Energy.** A lower converged energy on the same model at the same D is
     better evidence. Note the ED bound available here is for a *finite* 2x2
     torus and does **not** bound an infinite-lattice CTM energy — see §5.5.
   - **Self-consistency.** The returned lambdas must equal the BP messages of
     the returned state to the gauge tolerance. This is near-tautological by
     construction, which is the point: it is exactly the property `main`'s
     stored weights fail by 15–35%.
   - **Tree ground truth** (§6.2), where the answer *is* known exactly.
   - **Independent agreement, within one model only**: dense vs symmetric on the
     *bosonic Heisenberg* problem, which they both encode, must land on the same
     physics at the same D. **Fermionic is excluded**: it evolves the spinless
     t-V model with graded exchange statistics, so equal D does not make its
     output comparable, and requiring agreement would reject a valid fermionic
     result. Fermionic needs its own anchors (§5.5), and this criterion cannot
     be evaluated in Phase 2 at all, which is dense-only.

   Swept **≥3 seeds × D ∈ {2,3,4}**, because §5.3 shows neither axis alone
   discriminates. Note D≥4 needs enough imaginary time: 1200 steps at dt=0.01 is
   only 12 time units and reads as a failure that is really under-convergence.
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

1. **Phase 0** — land **#881** (the fermionic baseline: five of six defects, and
   fPEPS on a 2-site checkerboard) and **#879**. #881 is open; #879 is not
   started.

   #879 is **not** "a variational bound on the CTM energies" — that phrasing
   survived here after §5.5 disclaimed it, and following it would restore the
   invalid check. It is: 2-site cross-path agreement, chi-convergence, a
   product-state magnitude anchor, and the finite-cluster bound left on
   `reference_energy_2x2_pbc` alone. See §5.5.

   **Decided: runs in parallel with Phases 1–2**, which are bosonic and blocked
   by neither. Phase 3 still needs both — without #881 the old path returns zero
   so comparisons carry no information, and without #879 no fermionic energy can
   serve as an acceptance criterion (§5.4, §5.5) — so Phase 0 is a prerequisite
   for Phase 3, not for starting.
2. **Phase 1** — `ipeps_gauge.py`: generalise the #870 gauge to 1-site and to
   fermionic. Prove exactness (§6.1, §6.2). **Go/no-go on §5.1.**
3. **Phase 2** — `ipeps_su.py`: the engine, bosonic dense first, against the
   §6.3 criteria *available on one representation*: energy, lambda/BP
   self-consistency, tree ground truth, chi-convergence, no `steps % 4`
   dependence, not-the-product-state and rank honesty — swept over seeds and D.

   **Not** cross-path agreement: that compares dense against symmetric and the
   second representation does not exist until Phase 3, so listing it here leaves
   Phase 2 with an unsatisfiable gate. **Not** a reference spectrum either;
   §6.3 establishes there is no valid one.
4. **Phase 3** — symmetric, then fermionic (gated on Phase 0, and on the
   *2-site* layout question in §5.1 — not on the 1-site experiment). **This is
   where dense-vs-symmetric cross-path agreement becomes evaluable**, once a
   second representation of the same bosonic model exists.
5. **Phase 4** — migrate `ipeps()` and `fpeps()`. The seven call sites are now
   one (#877), so this is a single edit rather than seven. **Also export
   `gauge_fix` in `src/tenax/__init__.py` (`__all__`) and document it in
   `README.md`** — it is the one public symbol this design adds (§3).
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

## 9. Decisions

Resolved 2026-08-15; the reasoning is folded into the sections named.

1. **Fermionic drops to v2 if the §5.1 gate fails** — authorised in advance, so
   it is not a question reopened under schedule pressure. Narrowed: the gate is
   the *2-site* layout question, since the 1-site constraint is superseded by
   #881. See §5.1.
2. **`_SUState` is a new type** (internal for v1), not the existing container
   with lambda removed.
   See §3.
3. **Phase 0 runs in parallel with Phases 1–2**, and gates Phase 3 only. See §7.

What remains genuinely open is empirical, not a choice: whether the 1-site layout
constraint survives removing the storage (§5.1), and whether BP is an exact gauge
on a graded tensor network (§5.2). Both are Phase 1 measurements.
