# Enforcing the fermionic sign convention (#1035) — design

Date: 2026-09-23 · Status: **proposed, for review — revision 2** · Tracks: #1035, #1037, #995, #1023, #1034

Every file:line below is on `main` at `ad9c213`. Every numerical claim comes
from a probe run against that commit. The probe is described next to its result.

> **Revision 2 (same day).** Revision 1 assumed that a planar network needs no
> fermionic sign, so that only reorders could be wrong. **That premise is
> false** (#1037, §3.3): tenax's double layer with no gates computes the
> *hard-core-boson* energy. This revision adds the evidence (§3.3), a prototype
> of the alternative revision 1 did not consider, a **graded contractor**
> (§3.4), and why it is not a revert of #555 (§3.5). **The recommendation
> changes** (§4, D0): build the graded contractor. Revision 1's explicit-crossing
> plan (`braid` + a raising graded `transpose`) is kept as alternative A.
> Unchanged: §2, §3.1, §3.2.

---

## 1. The problem in one paragraph

#555 made `contract` sign-free. Under that choice a leg's **storage order carries
no physics**, and every fermionic sign must be placed on purpose. One operation,
the graded `SymmetricTensor.transpose`, still follows the other convention, in
which a reorder **is** a fermion exchange. The library has 84 graded `transpose`
call sites, and nothing at any of them records which convention the author
meant. Picking wrong raises nothing and breaks no norm. The result is still a
plausible, PSD, trace-1 density matrix, just of the wrong state. #994, #997 and
#995 are three instances of this. #1035 has the full history.

Revision 2 widens the problem. The reorders are not the only place a sign can go
missing: tenax's double layer has none at all, and without them the functional
is bosonic (§3.3). So the question is no longer only "which convention does each
reorder follow?" but "where does the library put fermionic signs at all?".
§4 D0 answers it.

## 2. What the code actually does today

### 2.1 Where signs come from

| Operation | Sign applied | Storage-order dependent? | Use in `src/` |
|---|---|---|---|
| `contract` | none (`contractor.py:949`, #555) | — | everywhere |
| `permute_legs` | none (`tensor.py:319`, `:1198`) | — | 10 sites |
| `bar()` | none: conjugate + flip flows (`tensor.py:1159`). **Revision 2:** a graded bra must also reverse the leg order (§3.4, rule 3) | — | CTM bra layer |
| fuse / split, linalg matricization | none, since #998 (`f8fe036`) | — | every decomposition |
| **graded `transpose`** | **(−1)^(# inverted odd–odd pairs)** via `_koszul_sign` (`tensor.py:105`, one caller at `:1192`) | **yes** | **84 graded sites** |
| `dagger()` | (−1)^C(k,2), where k = # odd legs (`tensor.py:1149`) | no, verified (§3.1) | 4, all HOTRG |
| `swap_gate(i, j)` | (−1)^(pᵢ pⱼ) (`tensor.py:1213`) | no | **0** |
| `twist(axes)` | (−1)^(parity) per leg | no | not on `main` (#1034) |
| `BaseSymmetry.twist_phase` | declared (`symmetry.py:235`) | — | **0 callers** |
| `trg.py:537, 569` | hand-built Grassmann tensor entries | model data | fermionic TRG only |

**`transpose` is the only source of a storage-order-dependent sign.** #1035
counts eight sign-bearing places. Five of them apply no order-dependent sign,
and two (`swap_gate`, `twist_phase`) have no callers. The design problem
therefore comes down to one method and its call sites.

### 2.2 Corrections this establishes

- `ipeps_gauge.py:316-320` says Koszul signs enter through two groups, with linalg's
  decompositions making up "the other seven". Linalg has not called
  `_koszul_sign` since #998, so that sentence is stale. It is the fifth dead
  citation found in this subsystem this month, after #1023's test, `bar()`'s
  twist (#1033), #898's `_contributors` marker, and `_retruncate_by_base_charges`.
- `permute_legs`' docstring says `contract` returns legs in "an internal order
  that carries no meaning". That order is not private. It is either
  `output_labels`, or the first-seen order of free labels scanned across the
  operands (`contractor.py:140-160`). It is deterministic, but the caller is not
  choosing it on purpose. The docstring's *rule* is still correct.

### 2.3 The explicit-crossing primitive exists and is unused

`swap_gate` is exactly the primitive the #555 convention calls for: a sign placed as
network data at a named crossing of two lines. Its docstring cites the Corboz
convention. **Nothing in `src/` calls it.** Every fermionic sign that actually acts
in the CTM stack today comes from a graded `transpose`.

## 3. Measurements

### 3.1 Graded `transpose` is exactly `permute_legs` plus swap gates

**Probe:** a random rank-5 tensor on `FermionParity` (16 blocks) and on
`FermionicU1` (155 blocks), with mixed flows, under 60 random permutations each.
For each permutation, compare `T.transpose(p)` against
`swap_gate(a, b)` applied to every pair of original legs `a > b` that the
permutation inverts, followed by `permute_legs(p)`.

| Symmetry | max \|transpose − factored\| | transpose ≠ permute_legs (regime) | drop-one-gate mutant caught | max \|dagger∘permute − permute∘dagger\| |
|---|---|---|---|---|
| FermionParity | **0.0** | 59/60 | 59/59 | **0.0** |
| FermionicU1 | **0.0** | 60/60 | 60/60 | **0.0** |

So `transpose(p) ≡ permute_legs(p) ∘ ∏_{inverted pairs} swap_gate`, exactly.
This extends an existing test. `test_swap_gate.py:96`
(`test_adjacent_transpose_cross_check`, from #988) already pins the identity for
a single **adjacent** swap on `FermionParity`. The probe covers arbitrary
permutations, `FermionicU1`, and the `dagger` commutation. Two consequences
follow:

1. **Removing the signed form costs nothing.** Anyone who really wants the braid
   can write it out as swap gates plus a sign-free permutation, and get the same
   value to the last bit.
2. **Every existing graded `transpose` is really a set of swap gates.** A signed
   reorder places a crossing sign between each pair of odd lines whose storage
   order it inverts. Whether that is right is a question about the diagram: *do
   those two lines cross in the planar picture?* Nobody can answer that while the
   crossings are implicit in a permutation tuple.

`dagger`'s sign commutes with any reorder, so `dagger` is **not** part of this
problem.

### 3.2 A storage-order test does not settle #995

**Probe:** a random fermionic checkerboard pair (D=2, d=2, `FermionParity`), run
through the production fermionic CTM (`ctm_split_tensor_2site`, χ=6, 4 sweeps,
`conv_tol=0`) and then `_rdm2x1_split_tensor_2site`. The same computation is
repeated with both site tensors' legs reordered by `permute_legs` under four
permutations σ. Each arm runs in its own process on CPU. Convergence is
deliberately *not* required: this is a deterministic map, and its answer must not
depend on the representation of its input.

| Arm | ⟨n_A⟩ | ⟨n_B⟩ | ⟨hop⟩ | worst max\|ΔRDM\| over σ |
|---|---|---|---|---|
| `main` | 0.6714716492 | 0.5879767945 | 0.0908055327 | **1.7e−16** |
| `transpose := permute_legs` | 0.6572464858 | 0.5507763377 | 0.0962584896 | **3.9e−16** |

1. **Neither arm depends on storage order.** The fermionic CTM on `main` is not
   representation-dependent the way #994 was. The simplest reading, inferred
   rather than measured, is that its signed transposes act on internal layouts
   that are fixed before they are reached, so each one applies a **constant**
   sign pattern, i.e. a fixed set of swap gates on the environment tensors'
   legs.
2. **The arms disagree with each other by 2–7% on every observable.** The signs
   are physical, as #995 said (−0.042753 vs −0.018505). They are not gauge.
3. **Storage-order tests are blind to constant sign patterns by construction.**
   They are still the right guard against the #994 class (see §5), but they
   **cannot** decide #995. This overturns the suggestion, made while this
   design was being drafted, that a storage-order test could adjudicate #995
   without an oracle.

### 3.3 The double layer with no gates is the hard-core-boson functional (#1037)

`_build_double_layer_tensor` / `_build_double_layer_open_tensor`
(`_ctm_tensor_init.py:90, 128`) are `contract(A, A.bar())` followed by a
sign-free fuse, with no swap gate and no twist. The comment that justified this
(`fermionic_ipeps.py:413-416`, "Koszul signs in transpose, contraction, and SVD
... no explicit swap gates are needed") has been void since #555 made `contract`
sign-free and #998 made SVD sign-free.

**Oracle:** PR #1038 adds `tests/_fermionic_fock_oracle.py`. It builds a small
open-boundary fermionic PEPS in Fock space, where every sign comes from the
operator algebra: site operators in `(u, d, l, r, p)` order, bond projectors
`(1 + a_t a_s)`, then a projection onto the virtual vacuum. A closed-form sign
formula checks it exhaustively on every configuration.

**Certificate (convention-free, a variational bound):** D=2, OBC, t=1, V=0.
Minimising the current double-layer functional lands on the hard-core-boson
ground state, which is below the fermionic bound.

| Cluster | min of tenax's functional | E_HCB (ED) | E_F (ED) |
|---|---|---|---|
| 2×2 | −2.828427 | −2.828427 | −2.0 |
| 2×3 | −3.9235 | −3.9247 | −3.4142 |

A functional that goes below E_F cannot be fermionic under any convention.
#1038 also pins it directly: tenax's own `_build_double_layer_open_tensor` plus
`contract` return exactly the HCB energy of the same tensors (`abs=1e-10`).

**A local rule exists but has to be derived.** A GF(2) fit gives a 24-term sign
rule: a per-site rule over ket and bra bits, plus a per-operator rule on the two
hop sites. It was checked on held-out clusters (0 violations over 6100
configurations on 3×4, 4×4, 5×4, 3×5 and 4×3) and it matches Fock to 3e−16.
**A drawing did not produce it.** The first derivation for this design, with two ket–bra gates
per site read off the diagram, gave negative norms. It misses the ket/bra
reorder signs that arise because `(u, d, l, r)` is not a planar cyclic order.
This is the cost of alternative A in concrete form (§4 D0): a sign rule has to
be derived per diagram and per operator, and a plausible drawing gets it wrong.

**Consequences:**
- Every fermionic SU/CTM energy computed without the 9 signed CTM transposes is
  a hard-core-boson energy.
- With them (`main`), it is a third functional that has not been identified.
  #995's "which arm is right?" is probably answered by *neither*.

### 3.4 A graded contractor reproduces the oracle with three rules and nothing fitted

**Probe:** `docs/plans/2026-09-23-1035-graded-contractor-prototype.py`, which is
plain numpy, dense, and runs on tiny clusters against the #1038 oracle. A
tensor's storage order **is** its Grassmann generator order, and the index
parity is the value mod 2. The whole convention is three rules:

1. **Reorder** with the Koszul sign: (−1)^(Σ over inverted odd–odd pairs). This
   is today's graded `transpose`.
2. **Contract a pair** by bringing the two legs next to each other at the end
   (rule 1), then weight the diagonal +1 if the OUT leg comes first and
   (−1)^p if the IN leg comes first.
3. **`bar`** conjugates, flips the flows **and reverses the leg order**.
   Equivalently, it keeps the order and multiplies by (−1)^(Σ_{i<j} pᵢpⱼ), the
   Koszul sign of a full reversal. That is exactly the phase of the `bar_super`
   that #557 deleted.

The bond pairing convention is a gauge. Rule 2 with "OUT first" equals the
oracle's `(1 + a_t a_s)` up to a Z on each bond's t-side leg. **That prediction
was written down before the first run**, and it is the only choice made.

| Check (2×2, 2×3, 3×2; 3 random states each) | Result |
|---|---|
| graded vs Fock energy | **≤ 6.7e−16** on 9/9; norm > 0 |
| graded vs #1037's fitted rule | ≤ 6.7e−16. Rules 1–3 generate it. |
| graded vs HCB | 0.05–2.2: the check can fail |
| contraction order reversed | ≤ 4.4e−16 |
| every site's legs reversed with a **signed** reorder | **0.0** |
| every site's legs reversed with a **sign-free** reorder | 0.7–3.7: under this scheme, storage order is physics |

| Mutant (worst \|E − E_Fock\|, 6 states) | |
|---|---|
| none | 2.2e−16 |
| rule 3 dropped: `bar` keeps the leg order (today's `bar()`) | **1.8** |
| rule 2 dropped: no pair-orientation sign | **1.8** |
| rule 1 dropped: sign-free reorder | **1.8** |

**Odd tensors, which is the excitation shape.** Put an odd-parity B at site x in
the ket and at site y in the bra, on top of a shared even background. Contract
every tensor in the true global order `bar(K_N)…bar(K_1) [O] K_1…K_N`.
Norm and hopping matrix elements match Fock for **all** (x, y): worst 8.9e−16
over 16 pairs (2×2) and 36 pairs (2×3). Regroup the same contraction into
per-site double layers `bar(K_s) K_s`, which is what a CTM builds, and the
answer flips sign **exactly when y comes after x** in the row-major order
(2×2: rows `+--- ++-- +++- ++++`). That is the non-local string. A graded
contractor produces it from the declared order. A swap-gate scheme has to
thread it through the environment by hand.

**What this does not show:**
- Only finite OBC clusters were tested, not an infinite-lattice CTM.
- Nothing was truncated, so no linalg under graded semantics was exercised.
- An even ground-state network does not depend on the order in which its
  tensors are multiplied, since even tensors commute. So the CTM for ground
  states needs only the local rules. Odd objects need a declared global order.
  A graded contractor turns that order into data, but it does not remove the
  choice.

### 3.5 Why this is not a revert of #555

The contractor before #557 (`ad56b6a^:src/tenax/contraction/contractor.py:224-300`,
`_contraction_inversion_pairs`) applied rule 1, bringing contracted legs
together with Koszul signs. It did **not** apply rule 2: the pairs were summed
with no flow-dependent sign. It applied rule 3 only where callers used
`bar_super` rather than `bar`. This comes from reading the code. I have not
rerun the old contractor. The §3.4 mutants show that dropping either rule 2 or
rule 3 alone moves the energy by O(1).

#557's acceptance evidence could not have caught this:
- The 2×2 PBC variational bound is a non-planar network with no wrap-bond
  signs, which #995 later found inadmissible.
- The OBC tier contracted with the same sign-free `contract` it was testing.

So #557 removed a *partial* graded contractor and validated the removal against
references that could not tell the difference. What §3.4 proposes is the
complete one, gated on an oracle that can.

## 4. Decisions

### D0. Where fermionic signs live. Recommended: **B, a graded contractor**

| | A: explicit crossings (revision 1) | **B: graded contractor** |
|---|---|---|
| sign source | `contract` sign-free; signs placed as `swap_gate` / `braid` / `twist` | `contract`, `transpose` and `bar` apply rules 1–3 of §3.4 |
| storage order | carries no physics | **is** physics: the Grassmann order |
| double layer | needs a derived gate rule (24 terms for the NN hop, §3.3); a drawing got it wrong | falls out of rules 1–3 (§3.4, matches Fock to 1e−16) |
| each new operator / RDM shape | new rule to derive and certify | nothing new |
| odd tensors (excitations, c† correlators) | non-local string through the environment, by hand | follows from the declared global order (§3.4) |
| the #994 class | a signed reorder of a sign-free result | a **sign-free** reorder (`permute_legs`) of a graded result |
| compile / AD cost | ~0 | a per-block constant sign; #986 measured graded overhead at ~2% of compile |
| migration risk | every missed gate is silent | every sign-free reorder is silent; linalg and the 9 CTM sites must be re-derived |

**Why B:**
- **Correctness by construction.** Three local, mechanical rules reproduce the
  oracle exactly, including odd tensors. A needs a certified rule for every
  diagram shape, and §3.3 shows the obvious rule is wrong.
- **Excitations** (a stated goal) need odd B tensors, which is where A is
  weakest.
- **Enforcement stays possible.** B also has one silent failure: a sign-free
  reorder of fermionic data. A strict mode can catch it, just as revision 1's
  plan catches a graded `transpose`: `permute_legs` raises on fermionic
  `SymmetricTensor`s. The direction of enforcement flips, but the mechanism is
  the same.

**What would make A the better choice:** if linalg under graded semantics
(truncated SVD / QR / eigh with a new bond whose orientation must obey rule 2)
turned out not to have a clean definition. That is plan step 2's gate, and it
comes before any production path changes.

### D1. `contract`'s output order: no API change

`contract(..., output_labels=...)` already declares the result's order, and
because the contractor is sign-free, a declared order is the same as a
`permute_legs` afterwards. Call sites that reorder straight after a `contract`
should prefer `output_labels`, since that removes a reorder rather than choosing
one. This is guidance and a lint target, not a mechanism. **Priority: low.** Fix
the docstring wording from §2.2. Under B the guidance is stronger:
`output_labels` is how a caller gets a *correctly signed* order, and a reorder
afterwards has to be graded.

### D2. Enforcement under alternative A (revision 1's recommendation, kept for comparison)

Under B the analogue is a strict mode that makes `permute_legs` raise on fermionic tensors, plus a census of the sites that reach it (§5 step 5). The rest of this subsection applies only if A is chosen.


| | Option | Effect | Risk |
|---|---|---|---|
| **A** | **graded `transpose` raises**; add `braid(axes)` implemented *as* swap gates + `permute_legs` | every call site must choose `permute_legs` (bookkeeping) or `braid` (crossing) | 84 sites plus tests must migrate; bosonic/dense `transpose` unchanged |
| B | make graded `transpose` sign-free; add `braid` | no call site has to change | **silent** change of semantics at every signed site, which is the failure mode this issue is about |
| C | required keyword: `transpose(axes, *, braid: bool)` | same as A with one verb | a bare call fails only with a `TypeError` on *all* tensors, so bosonic callers churn too |

Why A:

- **`braid` built from `swap_gate` makes `swap_gate` the only sign source** for
  crossings, with `twist` added once #1034 lands. The table in §2.1 then has one
  row per concept, and a grep for `braid(` / `swap_gate(` lists every crossing in
  the library.
- **Enforcement is decoupled from adjudication.** Per §3.1, replacing a graded
  `transpose(p)` with `braid(p)` is **bit-identical**, so the whole migration can
  land with **zero behaviour change** and without solving #995 first. After that,
  each `braid` → `permute_legs` switch is its own small, reviewable, testable
  change. The failure #1035 names ("doing (4) alone is what produced #994 → #997
  → #995") came from adjudicating before enforcing.
- **Dispatch is by runtime type, which is the only accurate census.** Whether a
  site is graded is decided by the tensor it receives, not by the code. An AST
  audit can only estimate, and #995's 15 → 9 correction shows how far off an
  estimate can be. A strict mode enumerates exactly the sites the suite reaches.

`braid(axes)` semantics: equal to the current graded `transpose(axes)`, pinned by
the §3.1 identity as a test. On bosonic symmetries and `DenseTensor` it equals
`permute_legs`. On `ANYONIC` it raises `NotImplementedError`, matching #1034's
`twist`.

### D3. Acceptance gates: two layers, because one test class cannot do both jobs

| Gate | Catches | Misses | Cost |
|---|---|---|---|
| **G1. Storage-order metamorphic test** over fermionic CTM → RDM → energy (pattern already used at `test_ipeps_bp_gauge.py:1095` and `test_fermionic.py:1308`) | the #994 class: signs that depend on input or contractor layout | constant sign patterns (§3.2) | seconds; no oracle, no convergence |
| **G2. Physical adjudication**, one per `braid` site: *does the planar diagram contain this crossing?* confirmed by an exchange-sensitive reference | constant sign patterns: the #995 class | — | needs a fixture that has so far defeated three attempts |
| **G2′ (revision 2). The Fock oracle (#1038)** on every contraction *primitive*: double-layer build, operator insertion, RDM, odd strings, a truncated update | any wrong local rule, exactly, in seconds | infinite-lattice / truncation effects | exact, 2×2 in the core gate |

G1 is cheap and permanent. It ships in step 1 and goes into the core suite.

Revision 2: G2′ now exists for finite clusters and makes the local rules
exactly checkable. Under B, G1 becomes "a **signed** reorder of inputs leaves
every observable invariant" (§3.4 measures 0.0), and a sign-free reorder is the
mutant it must catch. What remains of G2 is the infinite-lattice CTM, where the
only error left once G2′ holds should be ordinary truncation.

G2 is still the hard part, and this design does not claim to have solved it.
What D2 changes is the *question*. Today #995 asks "which of two conventions is
right at 9 opaque permutation tuples?". After migration each site asks "does
line X cross line Y here?", which can be read off the CTM diagram. The four
`_ctm_tensor_moves.py` sites (838/920/990/1041) are the clearest case. Each
reorders one renormalised edge tensor into its canonical `(chi, D², chi)`
layout, which is one diagram element with no line crossing another.
~~The diagrammatic answer there is `permute_legs`.~~ **Withdrawn in revision
2:** "no crossing, so no sign" is exactly the premise §3.3 refutes. Tenax's
`(u, d, l, r)` storage order is not a planar cyclic order, so signs are needed
even where no two lines cross. The measured disagreement in §3.2
says that choice matters. It does **not** say which arm is right, and no site
should switch without G2 confirming it.

Constraints on the G2 fixture, learned from #995's three failures:
**partially filled** (⟨c†c⟩ is Pauli-blocked at full filling), an **off-diagonal
observable** (⟨n⟩ is diagonal and barely feels the exchange sign), and a **short
correlation length**, so that both the CTM χ-scan and the reference's size scan
converge well inside the arm separation (§3.2 gives 2–7% here). Candidate still
to be evaluated: a lattice-symmetry covariance check. The four move directions
use four different canonical layouts, so a constant sign that is wrong in one
direction breaks the relation between horizontal and vertical bonds. **This
needs the fermionic sign rule for the lattice map itself to be derived first**,
since reflection is orientation-reversing. It is listed as a candidate, not a
result.

## 5. Plan (PR-sized), for B

0. **Land the oracle** (#1038). Every later step is gated on it.
1. **A reference `graded_contract` on `SymmetricTensor`** (rules 1–2) and a graded
   `bar` (rule 3), added **beside** the current path with no behaviour change.
   Port §3.4's checks (even states, the three mutants, odd-B) to it, against the
   oracle.
2. **Linalg under graded semantics.** Define the orientation of the new bond so
   that `U·S·V` reconstructs the input under rule 2, and check a truncated
   2-site update on a 2×2 cluster against Fock. **This is the go/no-go for B**
   (D0).
3. **Double layer, RDM and energy on the graded path.** #1038's strict xfail
   flips, and its #1037 characterization test is removed.
4. **Re-derive the CTM reorders.** Each of the 9 sites is either a correct
   graded reorder already or has to change. Check them with the signed-reorder
   G1 and, where a finite patch allows, with G2′. #995 closes and #1023 is
   superseded.
5. **Strict mode.** `permute_legs` on fermionic tensors raises when a flag is
   set. Run the full suite under it for the census, then migrate.
6. **Flip the default** and delete the sign-free fermionic contraction path.
   Re-check the fermionic SU against G2′; it has not been verified yet.
7. **Excitations** on the graded path, with odd B checked against G2′ (§3.4
   shape) before any momentum-space code.

**If A is chosen instead**, revision 1's plan applies unchanged: add `braid` and
G1, add a strict graded `transpose`, migrate mechanically, flip the default,
then adjudicate. It needs one extra item. The #1037 rule, and a rule for every
further operator shape, has to be implemented as gates and certified against
G2′.

## 6. Out of scope

- `dagger`'s sign convention (order-independent, §3.1; used only by HOTRG).
- The general anyonic phase (`twist_phase`), which #1034 declines deliberately.
- `trg.py`'s Grassmann signs (model data, not a reorder).
- `DenseTensor.twist` being a no-op on fermionic indices (pre-existing,
  documented in #1034).

## 7. Questions for reviewers

1. **A vs B** (§4 D0). Is there a reason to keep `contract` sign-free that
   outweighs §3.3–3.4?
2. **Migration shape for B.** A separate `graded_contract` during steps 1–4, or
   a keyword on `contract`? A separate function keeps the two semantics apart,
   and the default flips in one place (step 6).
3. **Pair orientation (rule 2).** "OUT first" matches tenax's physical flows
   (ket `p` IN, bra `p~` OUT) with the natural `⟨0|c c†|0⟩ = +1`. Virtual bonds
   then differ from the oracle's projector by a bond Z, which is a gauge. Is
   there a convention elsewhere in tenax (for example `twist_phase`) that this
   must agree with?
4. **Non-planar networks under B.** A wrap bond is just another pair, and the
   twist becomes an explicit choice of boundary condition (periodic or
   antiperiodic) rather than a correction for a missing crossing sign. This is
   untested, because the oracle is OBC. Should #1034's `twist` wait for this?
