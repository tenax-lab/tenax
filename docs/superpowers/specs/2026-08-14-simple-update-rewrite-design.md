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
At the **default dense** D=2 the naive cost is 85–122×, which is not acceptable
and is not accepted: it is host-dispatch overhead in an untraced Python loop,
and tracing it is a Phase 1 gate. The fermionic default is a separate case with
a separate gate — its SU cycle is already 96× the dense one before any gauge is
added. See the measurements below.

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
independently measured `ipeps()` run (0.9 s / 200 phases = 18 ms per 4-phase
cycle). An earlier version of this benchmark cycled `t % 4` over too few trials
and let first-time compiles for phases 2 and 3 land inside the measured window;
its ratios were wrong and are not the ones above.

A third measurement, taken later against a *random* pair rather than a 60-phase
SU state, gives **12.2 ms** for the same D=2 cycle — a 1.4× spread on a quantity
that is almost entirely host dispatch at this size. Nothing here depends on
which end of that spread is used, but the regression claim below does, so it
uses the smaller baseline: it is the conservative choice, and it is the one
measured by the same method as the fermionic number it is compared against.

**The table is per BP solve; the cadence needs four of them per cycle.** §2
re-gauges before *each bond*, and a checkerboard cycle has four, so the real
overhead is 4× the figures above:

| D | per solve @1e-6 | **per 4-bond cycle** | total slowdown |
|---|---|---|---|
| 2 | 21.4× | 85.6× | **default D** — 122× on the conservative baseline; see the gate below |
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

### D=2 is the default — and the two defaults are not the same problem

An earlier draft called the D=2 ratio an artefact of the host-dispatch floor
and moved on. The diagnosis is right and the dismissal is not: **`D = 2` is the
default in both `iPEPSConfig` and `FPEPSConfig`**, so this is the path most
callers are on.

A second draft then projected *both* defaults from one number, and that was also
wrong, in three separate ways: the dense default is **100** steps, not 200
(`iPEPSConfig.num_imaginary_steps = 100`; only `FPEPSConfig` is 200); a "step"
is one *bond*, so the dense default is 25 four-bond cycles while `fpeps()` —
which calls the shared sweep with `4 * steps` — is **200** cycles, eight times
as many; and the fermionic projection borrowed dense timings for a block-sparse
path. Re-measured, all four numbers in one session by one method:

| path | default steps | 4-bond cycles | SU cycle @D=2 | baseline run | with mandatory gauge |
|---|---|---|---|---|---|
| dense (`iPEPSConfig`) | 100 | 25 | 12.2 ms | **0.30 s** | 25 × 1.49 s = **37 s** (~122×) |
| fermionic (`FPEPSConfig`) | 200 | 200 | **1167 ms** | **233 s** | not projectable — see below |

**The dense default is the regression, and it is worse than the 85× figure it
replaces** — not because the gauge got more expensive (four solves at 370 ms is
1.48 s per cycle, unchanged) but because the baseline it is measured against is
smaller than the earlier draft used.

**The fermionic default is a different problem that this design did not
create — and it is far worse.** A fermionic 4-bond cycle at D=2 costs 1167 ms
against dense's 12.2 ms — **96×** — because at D=2 the block-sparse blocks are
single numbers and the path is pure eager host dispatch, the wall documented in
#566/#618. So the fermionic default already takes ~233 s today.

An earlier draft stopped there, refusing to project the gauged cost because the
gauge had not been generalised to fermionic — and noted that *if* fermionic BP
matched dense timings the run would reach ~529 s, "a mere ~2.3×". **That
hypothetical was wrong by a factor of 17, and it has now been measured
directly.** `bp_gauge_checkerboard` in fact runs on a fermionic
`SymmetricTensor` as-is: at D=2 it converges in 29 iterations at tol=1e-6,
residual 6.9e-13. The cost:

| | per iteration | iterations @1e-6 | per solve | per 4-bond cycle |
|---|---|---|---|---|
| dense | 18.5 ms | 20 | 0.37 s | 1.48 s |
| **fermionic** | **392 ms** | 29 | **11.4 s** | **45.5 s** |

Per-iteration cost is flat in tolerance — 406 / 391 / 392 / 393 ms at
1e-2 / 1e-4 / 1e-6 / 1e-8 — so this is per-iteration dispatch, not setup, and
the total scales linearly with iterations. Against a 1.167 s SU cycle the gauge
is **39× the step it is gauging**, and the fermionic default goes from ~233 s to
**~9,300 s (~2.6 hours), a ~40× regression**.

**This is the second independent blocker on fermionic v1**, alongside the
unexplained 2-site layout pin in §5.1, and unlike that one it does not need an
experiment to discover. It is also the blocker least likely to yield: the
21× per-iteration gap over dense is the same block-sparse eager-dispatch wall
that #566, #618 and #630 each attacked and each closed NO-GO, concluding it is a
JAX compile-model limit rather than an algorithmic one. A tracing pass that
rescues the dense path (which is plain array ops) has no reason to rescue this
one.

Consequence: the §9.1 decision to drop fermionic to v2 if its gate fails now
rests on a measurement rather than on a risk. Phase 1 should confirm this number
on a second seed and D, and then take the decision rather than attempt the
rescue.

**It is Python, and on the dense path it is fixable.** BP's cost per *iteration*
is flat in D and enormous relative to the work: 18.5 ms per iteration at D=2, on
tensors of 32 numbers. `ipeps_bp_gauge.py` today is a pure Python loop over
eager ops — no `jax.jit`, no `lax.scan`, no `lax.while_loop`.

**Phase 1 gate, therefore:** the BP iteration must be traced (`lax.while_loop`
on the residual, or `scan` over a fixed iteration budget) before the engine is
built on it, and the D=2 per-solve cost must come down to the point where the
**default dense** run is within ~2× of its 0.30 s baseline. If it cannot, the
cadence decision in §2 has to be revisited for small D rather than the cost
quietly accepted — and that revisit is a design change requiring its own review,
not an implementation detail. The fermionic default gets its own gate once the
fermionic gauge exists to measure; it is not covered by this one, and the
block-sparse dispatch wall behind its 1167 ms cycle is a pre-existing problem
that the rewrite neither causes nor fixes.

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

### The state is the absorbed form, and that is what makes `_SUState` complete

A state with no lambdas is only a state if the site tensors already carry them.
This has to be said explicitly, because the routine being generalised assumes
the opposite: `bp_gauge_checkerboard` works in **Vidal form**, *requires* its
incoming `weights`, and its docstring warns that passing `BondWeights.ones()`
instead "discards that pair's `lambda` and re-gauges a different state". Taken
literally, a `_SUState` holding bare `Gamma` and a `gauge_fix` handing the
weights back *beside* it would throw away half the state on every solve.

**The convention, therefore:** every tensor in `_SUState` is in **absorbed
form** — each bond's weight split symmetrically, `sqrt(lambda)` into each of its
two ends, exactly as `_to_physical_pair` already does for the CTM. A site tensor
is then the wavefunction, not one factor of it. Vidal form exists only
*transiently*: inside `gauge_fix` between its SVDs, and inside `_su_step`
between the gate and the truncation. It is never a field, never returned as
state, and never persists across a call boundary.

Three consequences, and they are the whole point:

- The `BondWeights` that `gauge_fix` returns are a **diagnostic output** — the
  honest Schmidt spectrum of the state at the BP fixed point, used for
  truncation and for reporting. Dropping them loses no physics, only a report.
- `gauge_fix` needs **no incoming weights**. It gauges what it is given, and
  what it is given is already the whole state.
- **Truncation must restore the convention**: after the gate and its SVD, the
  new singular values are split `sqrt(sigma)` into the two ends. A truncation
  that left `sigma` on one side would silently reintroduce a Vidal-form state
  under a non-Vidal type.

**Measured, not assumed.** Absorbing `sqrt(lambda)` and then gauging from
`ones()` reaches the same physical state *and* the same BP fixed point as the
Vidal call, on both tensor types, using the `_torus_2x2` invariance probe from
`tests/test_ipeps_bp_gauge.py`:

| pair | D | absorbed input vs Vidal input | gauged output, route-to-route | fixed-point weights | BP iterations |
|---|---|---|---|---|---|
| dense | 2 | 2.8e-16 | 2.7e-15 | 1.2e-14 | 26 vs 26 |
| dense | 3 | 5.4e-16 | 3.5e-14 | 1.6e-12 | 52 vs 51 |
| symmetric | 2 | 9.3e-18 | 1.0e-15 | 3.3e-13 | 39 vs 40 |
| symmetric | 3 | 1.2e-16 | 8.1e-16 | 4.8e-13 | 152 vs 153 |

The `ones()` warning in that docstring is about passing `ones()` alongside
*bare* `Gamma`, which drops `lambda`. Absorbing first and then passing `ones()`
is the same state written differently — `A sqrt(lam) · sqrt(lam) B = A lam B` —
and the table is the check that it is, on the symmetric path too, where a flow
mistake would collapse charge sectors rather than raise.

The iteration counts also say the convention is **free**: starting from `ones()`
on absorbed tensors costs at most one extra sweep. That is the same measurement
as "warm ≈ cold" below, seen from the other side — BP's fixed point belongs to
the physical state, so the parameterisation it starts from moves the cost by
about one iteration and the answer not at all.

**What that table could not see, and what does now (added 2026-08-17).** Every
column above passes the weights back into the probe, so all of them test the
*Vidal* reading. The claim this section actually rests on — that the returned
pair alone is the state, and dropping the weights loses only a report — was
asserted nowhere, in the spec or in the tests. The first implementation of
`gauge_fix` consequently returned `bp_gauge_checkerboard`'s Vidal pair unchanged
while its docstring promised the absorbed one, and no test could tell:

| reading of `gauge_fix`'s output | dense D=2 | dense D=3 |
|---|---|---|
| as Vidal — weights passed back in | 3.7e-15 | 1.7e-14 |
| as absorbed — **weights dropped**, as documented | **1.25e+00** | **8.9e-01** |

Comparisons here must normalise: BP deliberately rescales `Gamma` and
max-normalises `lambda`, and the torus is degree 4 in each site tensor, so a raw
relative norm reports 6.5e-01 for the reading that is exact to 3.7e-15. Getting
that wrong makes the probe fail and pass for the wrong reasons.

`gauge_fix` now absorbs before returning, so the convention holds at the call
boundary as written above, and the guard that was missing asserts **both**
directions — dropping the weights must leave the state alone, *and* feeding them
back must move it, since a one-sided assertion would still pass on the defect it
is meant to catch. Re-absorbing is not a harmless redundancy: it puts
`lambda**1.5` on every bond, which is #667's mechanism verbatim.

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

### 5.2 Fermionic BP converges to a gauge that is **not exact** — measured, and localised

**ANSWERED, 2026-08-16.** This section originally recorded an open question. It
now records a result, and the answer is the unwelcome one. The evidence below
was produced by an experiment that was stopped part-way when the bosonic path
took priority; its measurements and its unlanded test code are preserved in
`.superpowers/sdd/2026-08-16-simple-update-rewrite/` (`task-5-report.md`,
`task-5-wip.patch`). **Nothing here is on `main`.**

The BP gauge was built and verified for the bosonic checkerboard. Messages on a
graded tensor network must respect Koszul signs, and nothing had exercised that.

**It runs.** `bp_gauge_checkerboard` accepts a fermionic `SymmetricTensor` from
`_initialize_fpeps` without modification and converges — D=2 in 65 iterations to
residual 6.9e-13, D=3 in 28 iterations to 6.2e-13 — returning a plausible
four-bond spectrum. It is also, at 392 ms per iteration, catastrophically slow
(§2).

**Converging is not being exact, and here it is not exact.** Against the planar
witness of §5.2a, at D=3 and χ=20:

| | displacement |
|---|---|
| BP gauge | **7.28e-02** |
| a gauge that is exact by construction, same state, same χ | 6.25e-04 |
| ratio | **116×** |

The floor is set by a non-diagonal, parity-block-diagonal gauge whose inverse is
analytic and whose round-trip was verified to ~1e-16 with no environment at all
(gauge one bond, contract the two sites over exactly that bond, compare
elementwise — exact for any statistics). BP's displacement is also **flat in χ**
— 6.68e-02 at χ=8 against 7.28e-02 at χ=20 — while the floor falls 3.6× over the
same range. Flat in χ is this project's defect signature; falling with χ is
truncation.

**Localised to a single sign.** `_reorder` (`ipeps_bp_gauge.py:255`) routes
through `SymmetricTensor.transpose`, which applies a Koszul sign. Suppressing
that one sign:

- drops the displacement to 8.93e-04 — 1.43× the floor, i.e. into the noise;
- makes the D=2 gauge exactly per-leg (sign mismatches 6/16 → 0/16, rectangle
  identity 2.000e+00 → 5e-15);
- leaves the BP fixed point itself untouched.

That is a diagnosis, not a fix: the right correction may be to apply a
*compensating* sign elsewhere rather than to delete this one, and deciding that
needs the graded-network derivation, not a bisection. **Deferred with the rest
of the fermionic line** (§9.1) — it changes nothing bosonic, since
`SymmetricTensor.transpose` returns the identity sign on a non-graded symmetry.

**D=2 cannot express this question, and nearly hid it.** `fermionic_ipeps.py:167`
sets `virt_charges = [i % 2 for i in range(D)]`, so at D=2 the parity sectors
have sizes {0: 1, 1: 1} and *every* parity-preserving matrix on a virtual leg is
1×1. Every parity-preserving gauge at D=2 is therefore diagonal, and the
sign-carrying decompositions have no non-trivial block to act on. The first run
of this experiment was specified at D=2 and would have reported "BP is exact on a
graded network" — an artefact of sector structure, not a measurement. The same
trap appears in the mutation suite: "inverse not transposed on the far end" is a
**no-op at D=2** (1.06e-16) and 2.86–3.37 at D=3. Any fermionic gauge work must
run at D≥3; D=4 gives 2×2 blocks in both sectors. U(1)-Sz is not a substitute —
its D=3 charges `[0, 1, -1]` also give all sectors size 1.

**Cost, for whoever resumes.** One D=3/χ=20 witness solve is 46 s warm, ~390 s
cold (a fresh process pays 250–300 s of import and compile). `max_iter=20` agrees
with `max_iter=40` to 1.96e-13 — four thousand times below the effect being
measured — and halves the cost; the CTM does not exit early on `conv_tol`, so
this is a straight 2× saving. `gauge_fix` itself is 29 s at D=3. D=4 was never
run.

**The obstacle is sharper than an earlier draft of this section understood, and
it rules out the probe that draft prescribed.** `_torus_2x2` — the probe #870
relies on — densifies and contracts with `np.einsum`, which has no notion of
exchange signs, so on a fermionic state it computes the wrong scalar and would
certify a broken gauge. That much is right. The prescribed remedy, "build the
probe out of the graded `contract()` instead", is **not available**:

- `contract` **does not apply Koszul signs**, by an explicit design decision
  (`contraction/contractor.py:949-957`, #555): *"the contractor does NOT
  auto-apply Koszul signs from leg permutations… For planar networks — the only
  kind Tenax's CTM/RDM/energy code uses — no signs are needed… For future
  non-planar applications an explicit `twist` primitive can be added."* No
  `twist` primitive exists.
- Signs enter **only** through `SymmetricTensor.transpose` and the `linalg`
  decompositions. `_koszul_sign` is defined at `core/tensor.py:104` and called
  from exactly eight places: `linalg.py:374, 764, 1042, 1271, 1450, 1650, 2046`
  and `core/tensor.py:1167` (`transpose`). `bar()` applies none
  (`tensor.py:302-312`). *(Corrected 2026-08-17: an earlier version of this list
  also named `fuse`. It carries no sign — the `fuse` methods in `core/symmetry.py`
  are charge-fusion rules that combine charge arrays, and
  `_fuse_indices_symmetric` reorders with a bare `jnp.transpose`. The conclusion
  below is unchanged, but this list is what a reader uses to decide whether some
  other contraction is sign-safe, so a wrong member makes that decision wrong in
  both directions.)*
- A closed 2×2 torus is precisely the **non-planar** case that carve-out
  excludes: its wrap-around edges cross the interior.

So a torus built on `contract` performs no transpose, fuse or decomposition and
is categorically sign-free. Measured on a real `FermionParity` pair, it is
value-identical to the `np.einsum` probe it was meant to replace — 2.15e-15
relative. Routing through `contract` buys nothing.

Note the asymmetry this creates: `bp_gauge_checkerboard` **is** sign-aware,
because it goes through `eigh`. The gauge machinery and any torus probe
therefore use different sign conventions, and a sign-level defect in the gauge
is exactly what such a probe cannot see.

That was written as a hazard. It turned out to be a description of the actual
defect: the fault localised above sits in `SymmetricTensor.transpose`'s sign,
inside the gauge — precisely the class the torus is blind to. Had the re-scope
not happened, the torus would have certified this gauge as exact.

**Decided: use a planar witness instead of a closed amplitude.** Gauge
invariance is tested by a quantity the library can already contract correctly —
the fermionic CTM energy, or a reduced density matrix — asserted **before and
after** the gauge. Those paths are planar, which is the regime #555 states is
sign-correct, and they reuse machinery that is already exercised. Two properties
make this sufficient rather than a compromise:

- it is a **relative** check, so it does not depend on the absolute energy being
  certified — which matters, because §5.5 establishes it is not (#879);
- it is the quantity that actually matters downstream; a gauge that preserved a
  closed amplitude but moved the energy would be useless regardless.

It is weaker than a full amplitude comparison, and that is recorded here rather
than glossed: it cannot detect a defect that leaves every planar observable
invariant. Closing that gap needs the `twist` primitive, which is out of scope
for v1 (§8). Any witness built this way must be validated by **mutation** — a
deliberately mispaired gauge must make it fail — because an invariance assertion
that has never seen a violation is not evidence of anything.

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
and on the checks §5.5 finds applicable, not on an energy value.

**Not on "the variational bound".** An earlier version of this sentence said
that, and §5.5 then proved it wrong: the tier-7 bound is stated for the
four-site 2x2 PBC torus, while the CTM energies are infinite-lattice
quantities, a valid infinite-lattice energy may lie *below* a finite-cluster
ground state, and a finite-χ CTM contraction is not a strict variational bound
in the first place. Applying it would reject correct implementations. The bound
stays where it is valid — guarding `reference_energy_2x2_pbc`. What acceptance
uses instead, from §5.5: same-arity cross-path agreement (2-site against
2-site, 1-site against 1-site, never across arities), χ-convergence, and a
magnitude anchor on a product state whose energy is hand-computable.

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
   state: re-contract and compare. This is the foundation — if the gauge is not
   exact, nothing above it means anything.

   **The tolerance is per witness, not one number.** ~1e-15 belongs only to
   contractions that can be evaluated *exactly*: the bosonic torus, and the
   one-bond elementwise round trip of §5.2 (gauge one bond, contract the two
   sites over exactly that bond, compare elementwise — exact for any
   statistics, verified to ~1e-16 with no environment at all).

   On the fermionic path §5.2 rejects the exact torus contraction and
   prescribes a finite-χ planar CTM/RDM witness, and that witness has a
   **measured floor of 6.25e-04** at D=3, χ=20 — under a gauge that is exact
   *by construction*. Requiring ~1e-15 there would reject a corrected
   fermionic gauge on the strength of the witness's own resolution. State the
   fermionic tolerance relative to that floor, and pair it with a mutation
   requirement: the check must be shown to kill a gauge that is wrong by more
   than the floor. BP's own displacement is 7.28e-02, **116×** the floor and
   flat in χ, so the separation being asked for is two orders wide — the
   tolerance does not need to be tight to be decisive.
2. **Tree exactness.** On a 1D/tree topology, the BP gauge must reproduce the
   MPS canonical form and the lambdas must equal the Schmidt values to machine
   precision. This is the one place we have ground truth, and tenax already has
   `FiniteMPS.canonicalize` to check against.

   **2a. The absorbed-form convention holds end to end (§3).** Two assertions,
   both cheap and both guarding a way the state can silently become half a
   state: gauging an absorbed pair from `ones()` reproduces the Vidal call's
   physical state *and* fixed point (the table in §3, as a test); and a full
   `_su_step` — gate, SVD, truncate — splits `sqrt(sigma)` into **both** ends.

   **The second one must not be phrased as "its next `gauge_fix` is exact".**
   An earlier version was, and that assertion cannot fail: `gauge_fix` takes
   the tensors it is handed as the complete state, so a pair with all of
   `sigma` on one side is simply a *different gauge* of a different state, and
   it is preserved exactly. Invariance is silent on the thing being guarded.

   What does catch it is a comparison between the two ends rather than a
   round trip through the gauge. Implemented as
   `test_su_step_splits_sqrt_sigma_into_both_ends`, which reads each end's Gram
   matrix in the Vidal metric and requires them to agree; measured, the
   all-`sigma`-on-one-factor mutation of §6.2a is killed at
   `|G_i - G_j| = 3.450e+00` against a gate at 1e-11, and the same mutation
   survives a gauge-invariance assertion untouched. Run on the symmetric path
   too: a flow mistake there collapses charge sectors rather than raising, and
   dense cannot see it.
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
   fermionic. Prove exactness (§6.1, §6.2). Take the absorbed-form entry point
   (§3) — no incoming `BondWeights`, weights out as diagnostics only. **Go/no-go
   on §5.1**, and two performance gates: trace the BP iteration until the default
   dense run is within ~2× of its 0.30 s baseline, and **measure** the fermionic
   gauge cost rather than projecting it from dense timings (§2).
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

1. **Fermionic drops to v2 if its gates fail** — authorised in advance, so it is
   not a question reopened under schedule pressure. There are now **two**
   independent gates, and the second has already been measured:
   - the *2-site* layout question of §5.1, still unexplained and still needing
     an experiment (the 1-site constraint is superseded by #881);
   - **cost**, §2: the fermionic gauge is 39× the SU step it gauges, taking the
     default run from ~233 s to ~2.6 hours. This is the same block-sparse
     eager-dispatch wall that #566, #618 and #630 each closed NO-GO, so it is
     the gate least likely to yield to effort.

   Phase 1 should confirm the cost on a second seed and D and then **take the
   decision**, rather than spend the phase attempting a rescue that three prior
   issues concluded is a JAX compile-model limit.
2. **`_SUState` is a new type** (internal for v1), not the existing container
   with lambda removed.
   See §3.
3. **Phase 0 runs in parallel with Phases 1–2**, and gates Phase 3 only. See §7.

What remains genuinely open is empirical, not a choice: whether the 1-site layout
constraint survives removing the storage (§5.1), and whether BP is an exact gauge
on a graded tensor network (§5.2). Both are Phase 1 measurements.
