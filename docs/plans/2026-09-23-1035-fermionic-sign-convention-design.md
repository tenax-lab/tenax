# Enforcing the fermionic sign convention (#1035) — design

Date: 2026-09-23 · Status: **proposed, for review** · Tracks: #1035, #995, #1023, #1034

Every file:line below is on `main` at `ad9c213`. Every numerical claim comes
from a probe run against that commit. The probe is described next to its result.

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

## 2. What the code actually does today

### 2.1 Where signs come from

| Operation | Sign applied | Storage-order dependent? | Use in `src/` |
|---|---|---|---|
| `contract` | none (`contractor.py:949`, #555) | — | everywhere |
| `permute_legs` | none (`tensor.py:319`, `:1198`) | — | 10 sites |
| `bar()` | none: conjugate + flip flows (`tensor.py:1159`) | — | CTM bra layer |
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

## 3. Two measurements

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

## 4. Decisions

### D1. `contract`'s output order: no API change

`contract(..., output_labels=...)` already declares the result's order, and
because the contractor is sign-free, a declared order is the same as a
`permute_legs` afterwards. Call sites that reorder straight after a `contract`
should prefer `output_labels`, since that removes a reorder rather than choosing
one. This is guidance and a lint target, not a mechanism. **Priority: low.** Fix
the docstring wording from §2.2.

### D2. Make the choice explicit at every call site. Recommended: **option A**

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

G1 is cheap and permanent. It ships in step 1 and goes into the core suite.

G2 is still the hard part, and this design does not claim to have solved it.
What D2 changes is the *question*. Today #995 asks "which of two conventions is
right at 9 opaque permutation tuples?". After migration each site asks "does
line X cross line Y here?", which can be read off the CTM diagram. The four
`_ctm_tensor_moves.py` sites (838/920/990/1041) are the clearest case. Each
reorders one renormalised edge tensor into its canonical `(chi, D², chi)`
layout, which is one diagram element with no line crossing another. The
diagrammatic answer there is `permute_legs`. The measured disagreement in §3.2
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

## 5. Plan (PR-sized)

1. **Add `braid` and G1, with no behaviour change.**
   `SymmetricTensor.braid(axes)` as swap gates + `permute_legs`, a test extending
   `test_adjacent_transpose_cross_check` to the full §3.1 identity (with the
   drop-one-gate mutant), and a
   storage-order metamorphic test over the fermionic CTM → RDM path. Fix the
   stale citations from §2.2.
2. **Add a strict mode.** `TENAX_STRICT_GRADED_TRANSPOSE=1` makes graded
   `transpose` raise. Run the full suite under it, including the
   `run-full-tests` matrix, to produce the **runtime** census of graded sites the
   suite reaches.
3. **Mechanical migration**, `transpose` → `braid` on every site in the census,
   in modules of about 10 sites each. Bit-identical by §3.1, so each PR's
   evidence is "the suite and G1 are unchanged". No adjudication happens here.
4. **Flip the default.** Graded `transpose` raises without the flag, and
   `transpose` keeps its current meaning only for bosonic and dense tensors.
5. **Adjudicate, one site per PR:** `braid` → `permute_legs` wherever the planar
   diagram has no crossing, each confirmed by G2. #995 closes here, and #1023 is
   superseded by the per-site PRs.
6. **Declare planarity.** Once #1034 lands, record which diagrams are non-planar
   (the PBC references) and require `twist` on their wrap crossings.

Steps 1–4 do not depend on G2 and can land while the fixture question stays
open. Step 5 cannot.

## 6. Out of scope

- `dagger`'s sign convention (order-independent, §3.1; used only by HOTRG).
- The general anyonic phase (`twist_phase`), which #1034 declines deliberately.
- `trg.py`'s Grassmann signs (model data, not a reorder).
- `DenseTensor.twist` being a no-op on fermionic indices (pre-existing,
  documented in #1034).

## 7. Questions for reviewers

1. **A vs C** (§D2): is the churn on bosonic callers under C worth having one
   verb instead of two?
2. Is **`braid`** the right name? The operation is "reorder, with a crossing sign
   at every inverted odd pair". For Z2 grading over- and under-crossings agree,
   so "braid" is accurate here, but it would need a direction argument if
   `ANYONIC` is ever implemented.
3. G2: is there an **exactly solvable, partially filled fermionic PEPS** (for
   example a Gaussian fPEPS with a known correlation matrix) small enough to
   serve as the oracle? That would remove the fixture-convergence problem
   altogether.
