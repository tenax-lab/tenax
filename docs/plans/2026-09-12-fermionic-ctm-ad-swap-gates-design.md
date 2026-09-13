# Fermionic CTM AD via swap gates + fixed-point adjoint — design

**Status:** proposed · **Date:** 2026-09-12 · **Refs:** #565, #566, #618,
#630, #557, #555, #562, #881, #882 (§9.1 v2 track), #297 ·
**Reference implementation:** arXiv:2512.20697 (Chen, Neupert, Hasik — YASTN) ·
**Also:** #391/#393 (canonical I-charges precedent) ·
**Supersedes:** `docs/plans/2026-04-01-symmetric-fermionic-ad-design.md`
(whose "no swap gates" assumption this design reverses; the
`fermionic_ipeps.py:416` comment stating that assumption goes stale with
Phase 5 and gets updated there)

## 1. Why

Fermionic CTM AD is unusable at production scale for a *compile-time* reason,
not a numerical one. Locating the cost precisely matters, because the naive
story ("Koszul signs inside every contraction") is wrong for this codebase:
planar contraction applies **no Koszul signs at all**
(`contraction/contractor.py:495` — the #557 auto-Koszul removal, exposed by #556's JW-ED test scaffolding), and
#566 attributes the large traced graph to symmetry-generic per-sector
packing shared with the *bosonic* symmetric path. What keeps the fermionic
graph strictly bigger than the bosonic-symmetric one is **metadata-driven
dispatch onto graded branches**, and the executing delta on the CTM path
is exactly three mechanisms: per-block transpose signs
(`core/tensor.py:1166` via `_koszul_sign`, :104); the CTM moves'
Koszul-correct *fused* paths selected by `_env_is_fermionic` at **four**
gates — all four absorption directions, `_ctm_tensor_moves.py:494/:588/
:664/:720` (the cheaper unfused #605 path is bosonic-only by design; the
split path has its own separately-named `_split_env_is_fermionic` gate,
out of scope here); and the per-block decomposition negations in
`tenax.linalg` (`if is_fermionic: flat_block = -flat_block`,
`linalg.py:297/:384–386` and :848 — the two CTM-reachable sites, run
three times per projector by `_compute_2x2_projector_symmetric`; the
:1126/:1373 twins are not on the CTM path). Nothing else contributes:
the super-algebra `dagger()` (:1100) carries twist signs but is unused
on the CTM/iPEPS path (HOTRG only), and `bar()` (:1134) is a plain
conjugate-plus-flow-flip with no per-block phase (Codex round 2 — an
earlier draft miscounted bar/super signs in this delta, and its "exactly
two mechanisms" phrasing then contradicted §3.3's own decomposition-sign
finding; self-review). The signs
themselves are
trace-time constants, but the branches they live on trace per-block
operations the bosonic path never sees, and AD re-traces all of it in the
VJP. Measured consequences:

- #565: the fermionic AD "hang" was a slow one-time block-sparse compile, not
  a deadlock.
- #566: block-sparse VJP compile cost scales with charge-block count, is
  backend-invariant (XLA graph size, not kernels; cuTensorNet NO-GO), and
  closed NO-GO twice more as #618/#630.
- #882 §2: the fermionic BP gauge measured ~39× the SU step it gauges — the
  same eager-dispatch wall from the other side.

Meanwhile the *bosonic* symmetric CTM compiles and runs. The design goal is
therefore: **make the fermionic CTM's traced graph identical in shape to the
bosonic-symmetric one**, and make the backward **one traced matvec** instead
of unrolled sweeps. Stated as sharply as possible: the only cost this
design can remove is the fermionic-vs-bosonic-symmetric *delta* — the
graded branches above. The block-sparse VJP wall #566 measured is shared
with bosonic-symmetric and stays (risk 3); Phase 0 therefore measures the
delta first, and a small delta kills the reform before any code is
written.

## 2. Reference point: arXiv:2512.20697

Chen/Neupert/Hasik simulate fermionic fractional Chern insulators with iPEPS
in YASTN using exactly this pair of choices:

- **Swap gates for fermionic statistics**: rank-4 parity tensors at diagram
  crossings ("a minus sign only when both incoming legs are fermionic"), each
  leg decomposed into even/odd sectors. Signs are *network data placed at
  build time*, not logic inside contraction.
- **Fixed-point adjoint**: their Eq. 13 is the (seedless, untransposed)
  Neumann series for the *environment derivative* at the converged fixed
  point `env = f(env, A)`; the energy gradient is then assembled in the
  standard adjoint form — seed with the objective cotangent
  `b = ∂E/∂env`, solve `(I − J_envᵀ) λ = b` as `λ = Σₙ (J_envᵀ)ⁿ b`, and
  add the direct term: `dE/dA = ∂E/∂A + J_Aᵀ λ`. That assembled form is
  what tenax's fused Neumann VJP already implements
  (`_ctm_energy_ad.py:1441–1515`), so the paper and the in-tree backward
  compute the same object by the same series. The payoff, paraphrasing
  the paper: the backward differentiates one fixed-point condition
  instead of storing and backpropagating through every CTMRG iteration.
  (Two earlier drafts each got one half wrong — first presenting the
  seedless series as the gradient, then attributing the seeded form to
  the paper's Eq. 13 and decorating it with a spliced pseudo-quote;
  self-review. Check the paper's Eq. 13 expecting the derivative series,
  not the assembled gradient.) Gauge is fixed *after*
  convergence-up-to-gauge by an invertible G per boundary tensor so the
  output is a strict elementwise fixed point of G∘f.

Caveat when citing it here: YASTN runs PyTorch **eager** — no tracing, no
compile wall; their cost is per-op dispatch (the same wall #882 §2 measured
at ~39× on the fermionic BP gauge in our stack). The paper proves the formulation at physics scale (U(1)-charged
fermions, ~10³ CTM iterations). In our stack the swap-gate reform does double
duty: it is also what shrinks the *traced* graph to bosonic size, so the case
for it is stronger in JAX than in their stack.

## 3. Design

### 3.1 Swap-gate primitive

`SymmetricTensor.swap_gate(axes=(i, j), grading=None)`: multiply each
block by `(-1)^{p_i(block) * p_j(block)}` where `p` is the Z2 parity
grading of the block's charge on that leg. On graded tensors the default
grading comes from `BaseSymmetry.parity` (`symmetry.py:192` — all charge
arithmetic stays behind the symmetry object, per #734). The explicit
`grading` override (per-leg charge→parity maps) exists because the
retyped-bosonic pipeline **cannot** read parity through the symmetry
object: bosonic `parity()` returns all-even *by definition*
(`symmetry.py:203`), so a post-retyping insertion relying on it would be
identically +1 exactly on the odd sectors that need the sign (Codex
round 9). Every insertion made after retyping consumes the grading map
the builder captured from the fermionic symmetry at retyping time —
never `symmetry.parity` on the retyped index. Properties:

- Metadata-driven ±1 per block, computed on host, applied as one fused
  multiply — a *constant* under tracing, so its VJP is the same multiply.
- Involution: `swap_gate(axes) ∘ swap_gate(axes) = id` (test anchor).
- Additive: no existing `SymmetricTensor` operation changes.

### 3.2 Swap-gated fermionic network builder

A new module builds the objects the CTM consumes, resolving the *fixed*
square-lattice diagram's crossings. "At build time" means **per gradient
evaluation, inside the traced loss** — not once outside the optimizer.
The CTM step rebuilds the double layer from the site tensor on every
sweep (`_make_jit_ctm_step` calls `_build_double_layer_tensor(A)`
unconditionally, `_ctm_python_loop.py:135`, and ~15 modules share the
pattern), and that is not incidental: dE/dA *needs* the A→layer map in
the trace. So the deliverable is a **differentiable builder function**
(graded A → swap-absorbed, retyped bosonic layer) that the fermionic
path threads into the CTM step and energy helpers as a double-layer
builder hook, replacing `_build_double_layer_tensor` there (Codex round
2 P1 — an earlier draft said "prebuilt, configuration only", which the
existing interfaces cannot consume). Retyping is metadata-only, so the
builder stays differentiable w.r.t. the block data. The retyping map
**retains an explicit per-leg grading map** (charge→parity, captured
from the fermionic symmetry before the type is dropped) as builder
metadata: the retyped indices report all-even parity by definition, so
this map is the only place the fermionic grading survives, and every
downstream swap insertion consumes it via §3.1's `grading` parameter,
never `symmetry.parity` on a retyped index (Codex round 9). What it
builds:

- **Double layer**: bra-ket contraction of A with its conjugate has a known,
  layout-determined crossing pattern; apply `swap_gate` at each crossing
  while building the double-layer tensor, then **retype the result as a
  plain bosonic `SymmetricTensor`** (Z₂ parity charges — Z₂×U(1) for t-V —
  on a non-graded symmetry object, via a small graded→bosonic index map).
  Retyping is the load-bearing step, not a convenience (Codex P1 on this
  PR): signs enter the CTM only through metadata-driven *dispatch* — the
  planar contraction path applies none (`contractor.py:495`) — so a
  swap-absorbed tensor still typed FermionParity would still take both
  graded branches from §1: per-block transpose signs and the
  `_env_is_fermionic` fused move paths. With retyping those branches are
  unreachable by construction, and the forward traces the same ops as
  bosonic-symmetric.
- **RDM / gate insertion**: the 2-site RDMs and the Hamiltonian-gate
  insertion have finitely many crossings, all diagram-determined and
  absorbed the same way, into the standard 2-site RDM contraction
  (`_ctm_tensor_energy` helpers). **No parity string enters the
  nearest-neighbor energy path** (Codex round 12): the correct reason is
  the planar-diagram argument on PR #557's record — planar NN bond
  diagrams (horizontal *and* vertical) contain no genuine line
  crossings, and `spinless_fermion_gate` already encodes the local
  hopping matrix elements, so a mandated inter-site string would
  double-count statistics and break the graded-oracle match. (The
  earlier citation of `test_jw_insertion_nn_hopping` was wrong evidence:
  that is a 1D AutoMPO MPO-builder test that never touches iPEPS/CTM
  code, and "no intermediate sites" is a 1D-ordering argument;
  self-review. The 2D regression coverage for this claim is Phase 2's
  oracle, not any existing test.) Whatever crossing signs a
  vertical-bond RDM insertion does need are exactly the
  diagram-determined swaps of the absorption sentence above.
  Jordan–Wigner strings appear only for genuinely nonlocal operators
  (NNN and beyond), outside the NN energy path this design ships.

The graded formalism is **not removed**: it keeps serving the existing SU /
energy paths and is the oracle this design validates against (the #562
pattern — native vs shim to 1e-10).

### 3.3 Forward CTM

Run the existing symmetric 2x2 tensor-CTM with the §3.2 builder threaded
in as the double-layer hook. The convergence and adjoint entry points
must grow the hook parameter (or dispatch on a marker the builder
attaches) — this is wiring in existing files, not configuration, and the
blast-radius table counts it.

**In-sweep statistics** (Codex rounds 8–9 — the sweep machinery is
*not* automatically sign-complete for this path). The graded formalism
injects signs at two places *inside* the sweep, and neither can be
absorbed into the fixed site double layer:

- **Edge absorption**: the graded fused path carries Koszul signs from
  grouping the χ⊕D² edge legs, and the unfused path accumulates a
  *different* sign on the renormalized edge (`_env_is_fermionic` and
  `_apply_proj_unfused` docstrings, `_ctm_tensor_moves.py:121–160`;
  corners are unaffected). These involve the *environment* legs.
- **Projector decompositions**: `tenax.linalg` applies `_koszul_sign`
  whenever a decomposition's left/right grouping is a nonidentity
  permutation (`linalg.py:384/:848/:1126/:1373`), and
  `_compute_2x2_projector_symmetric` runs three such SVDs per
  projector. Retyping silences these signs too — the swap-gated sweep
  can diverge from the graded oracle *before* absorption even begins.

Both remain **fixed-diagram, per-block parity signs**: the move and
decomposition topologies are static per call site, and the grading
survives in the builder's explicit grading map (§3.2) — so whatever
statistics the graded sweep encodes reappears as §3.1 insertions (with
explicit `grading`) in a sweep variant selected by the builder marker.
The design's hypothesis — that a globally consistent, build-time
assignment of such insertions exists for this planar network, per the
Corboz swap-gate construction — is exactly what the **Phase 3 graded
oracle decides, on states with odd-parity boundary sectors populated**;
it is not asserted here. This is the design's highest-risk seam after
the §3.2 retyping map (risk 4).

The hook must also own **environment initialization** (Codex round 3 P1):
on the default `env_init is None` path the loop builds every env from the
*site tensor* (`initialize_ctm_tensor_env(A, chi)`,
`_ctm_python_loop.py:330`), and `_env_is_fermionic` dispatches off
`env_src.C1`'s symmetry — so with a graded `A` the corners and edges are
born graded, the first sweep takes the fused graded path regardless of
the layer's type, and its metadata clashes with the retyped layer. On the
builder path the initial env is therefore constructed from the
**bosonicized layer's indices**, making `_env_is_fermionic` False from
sweep 0. A caller-supplied `env_init` carrying graded tensors on this
path is rejected with a clear error, not silently retyped — retyping
someone else's environment is exactly the class of silent relabeling
#938 taught us to refuse.

The hook must also enter **both compile-cache keys** (Codex round 4 P1):
`_JIT_STEP_CACHE` is keyed only by
`(id(neighbors), recipe, device_mesh, ctm_chunk_size)`
(`_ctm_python_loop.py:110`) and `_VJP_CACHE`'s static-config key carries
no builder field either — so a legacy-vs-swap-gated comparison in one
process (the Phase 5 flag A/B, or any test doing both) would silently
reuse whichever closure compiled first, executing the wrong forward and
adjoint. Both keys gain a static builder-identity marker. This is the
#938/#973 silent-wrong-result class in cache form, and the Phase 3 tests
must include an in-process legacy↔swap-gated alternation that fails if
either cache conflates the paths.

The **χ-bump / χ-schedule machinery is a fifth seam** (self-review —
none of the thirteen review rounds reached it): the bump path builds
legacy graded double layers from the *site tensor* at five sites outside
the hook/init/keying wiring — `_ctm_loop_core.py:165` computes bump base
charges via `_get_base_charges(_build_double_layer_tensor(A))` inside
`python_loop_ctm_converge` itself and pads the env with them (:211), and
the three optimizer dispatchers repeat the base-charge computation on
their SymmetricTensor branches (`ipeps_optimize.py:1285/:1469/:3061/
:4486`). A `swap_gates` run with `chi_auto_bump`,
`ctmrg_heuristic_increase_chi`, or `gs_chi_schedule_steps` set would
inject graded metadata into the retyped run mid-optimization — the exact
clash this section refuses. **v1 rejects `swap_gates` combined with any
of those three fields at config validation**; threading the builder
through the bump machinery is explicit follow-up, not Phase 3.

**Returned-environment type contract** (self-review): under
`swap_gates` the optimizer returns `(graded A, retyped-bosonic env)`,
and every checked-in measurement consumer feeds that pair to graded
energy helpers (`compute_energy_fermionic_ctm`,
`fermionic_ipeps.py:459`; `sublattice_gap`, :316;
`tests/test_fpeps_ad.py:211`). The #834 leg-pairing check compares
flows and charge *values* only (`contractor.py:719–728`), and
FermionParity charges {0,1} coincide with retyped Z₂ charges — so the
mixed pair contracts **silently wrong** (graded-side Koszul signs
against all-even env parities) whenever odd boundary sectors are
populated. Phase 5 therefore (a) states the return contract in the
`gs_fermion_backend` docs, (b) adds a symmetry-object mismatch guard at
the graded measurement entry points (mixed graded/retyped pairs are
rejected, not contracted), and (c) routes post-optimization measurement
for `swap_gates` results through the builder path itself.

Once hook + init + keying + bump-rejection are in, the FermionParity
special cases in
`_ctm_tensor_init` / `_ctm_tensor_moves` are dead on this path **by
type**: the retyped layers and envs make `_env_is_fermionic` False, so
the moves take the same unfused #605 path the bosonic tensors take. They
stay for the legacy path and get an audit note, not a deletion.

### 3.4 Fixed-point adjoint

v1 wires **one** engine onto the new forward: the fused Neumann VJP
(`_ctm_energy_ad`, which already carries partial `SymmetricTensor`
handling around :382–:449). The root-implicit symmetric engine
(`_ctm_root_implicit_symmetric`) is *not* a v1 candidate — Phase 5's
flag rejects every `ctm_ad_mode` (see scope), those engines build their
own graded double layers internally (`_ctm_root_implicit_symmetric.py`
:122/:1850), and wiring them is follow-up work (an earlier draft listed
it as a live candidate while simultaneously rejecting the only config
spelling that dispatches it; self-review).

The strict-fixed-point requirement is **already satisfied** on the
block-sparse forward — an earlier draft treated this as the design's
open question, but the tree answers it (self-review): the phase gauge is
type-generic and #362-hardened for `SymmetricTensor`
(`_ctm_energy_ad.py:801–804` installs `_phase_fix_ctm_tensor` on the
implicit forward and backward with no type test; `ad_utils.py:419–453`
handles the block-sparse case via the deliberately-blessed small-env
todense round-trip), and `validate_ctm_for_implicit_ad` makes
`forward_gauge="phase"` a hard precondition of the implicit path
(`ipeps_ad_policy.py:30–31`). The #565/#566 fermionic CTM-AD runs cited
in §1 already executed exactly this machinery. Phase 4's first task is
therefore *re-validation on the swap-gated layer*, not discovery; the
paper's post-hoc G∘f construction remains available as a footnote-level
contingency only.

Standing constraint from the library's own history: the adjoint solve is
meaningless on an unconverged forward — starving `max_iter` shows up as
gradient garbage, not an error. Validation must gate on forward convergence
before judging the backward.

## 4. Phases

Each phase is its own PR; every phase gates on the graded-formalism oracle.

- **Phase 0a — measure before building (parity-only).** Record current
  fermionic CTM-AD compile and wall time at D=2 and D=3, two seeds each
  (#882 §9.1's own instruction: confirm cost on a second seed and D
  before deciding anything), **and the bosonic-symmetric CTM-AD control
  at matched D, χ and block structure**. Pinned parameters (self-review
  — every axis below can flip the verdict if left floating): unit cells
  = the 2-site checkerboard (Phase 5's primary workload) *and* 1-site;
  engine = the fused Neumann VJP in **both** arms (same-engine parity);
  χ = 16; `JAX_PLATFORMS=cpu` pinned (#813-class backend drift is in
  this repo's record). The bosonic control's construction is named, not
  left to improvisation: **bosonic `ZnSymmetry(2)` at the fermionic
  tensors' exact block structure** (in-tree precedent:
  `test_ctm_root_implicit_symmetric.py`); no in-tree bosonic Z₂ gate
  exists, so the control state/gate is hand-built and budgeted. Harness
  seed: `examples/profile_ctm_ad_wall_566.py` already measures this
  wall — extend it rather than rewriting it. **Numeric NO-GO**: if the
  fermionic-minus-bosonic compile-time delta is below ~30% of the
  fermionic compile wall, the premise fails and the reform stops here —
  the parity-only arm costs zero code.
- **Phase 0b — the U(1)×P (t-V) arm.** This arm is **not free**
  (self-review): no FermionicU1 t-V pipeline exists in-tree —
  `spinless_fermion_gate` and `_build_initial_fpeps_tensor` hardcode
  FermionParity (`fermionic_ipeps.py:130/:196`), `FPEPSConfig` has no
  symmetry knob, and FermionicU1 appears in `src/tenax/algorithms/`
  nowhere at all. Running the U(1)×P cells means first building a
  charged gate, a charged-sector state init, and an energy/AD harness —
  real pipeline code, now budgeted in §6, and a prerequisite for Phase
  4's FermionicU1 gradient gate and §3.2's "Z₂×U(1) for t-V" retype
  target. 0b lands after 0a's GO and before Phase 4. Measurement hygiene, since this is the GO/NO-GO gate
  (Codex round 6): **one fresh process per cell** (no `_JIT_STEP_CACHE` /
  `_VJP_CACHE` reuse across cells), the persistent JAX compilation cache
  pointed at an empty per-cell directory (tenax enables it globally, and
  it silently converts recompiles into cache hits), `block_until_ready`
  around every timed region, and **compile time and post-warm runtime
  reported as separate numbers** — run ordering must not be able to move
  the delta. Record **iteration counts** (forward sweeps, Neumann/adjoint
  iterations) with every cell and compare fixed-count or per-iteration
  timings alongside total wall (Codex round 7): if the fermionic and
  bosonic controls converge in different numbers of sweeps, the raw
  wall-time delta measures different amounts of work, not the
  graded-branch overhead. Pin graded-path energies as oracles. Every
  later claim is judged against these numbers; nothing is frozen from a
  single run. (Hygiene and iteration-count rules apply to both 0a and
  0b.)
- **Phase 1 — `swap_gate` primitive.** ~100 lines + tests (involution;
  parity bookkeeping against a hand-computed 2-leg case; a graded-transpose
  cross-check on a random small tensor). Mutation: dropping the sign must
  fail the cross-check. Public-API contract: `swap_gate` is a method on
  the already-exported `SymmetricTensor`, and `__all__` holds only
  module-level symbols (Codex round 3) — so the contract is satisfied by
  documenting the method on the class plus a README example using the
  actual signature, kept aligned with the tests; no `__all__` entry, and
  no second top-level function invented just to have one.
- **Phase 2 — network builder.** Double layer + RDM/gate insertion with
  swaps absorbed. Gate: a **builder-level oracle** — the retyped layer
  and RDM network reproduce the graded formalism on small
  exactly-contractible cases (brute-force / small-χ contraction), to
  ~1e-10, **on fixtures asserted to populate odd-parity sectors on the
  crossing legs** — the same #884 vacuous-regime rule Phase 3 already
  carries, which an earlier draft imposed there but not here: an
  even-sector-only Phase 2 fixture passes at 1e-10 with every swap gate
  dropped, and this is the phase the doc calls highest-risk
  (self-review). The end-to-end energy-vs-graded-forward gate lives in
  **Phase 3**, not here: Phase 2 has no compatible bosonic environment
  yet (env initialization is Phase 3 work), and Phase 0's oracle
  environments are graded — mixing them with the retyped network is
  exactly the metadata clash §3.3 rejects (Codex round 13). This phase
  owns the highest risk (see §5) and its oracle test runs per commit.
- **Phase 3 — forward CTM on the bosonicized layer.** Thread the builder
  hook through the convergence entry points, including env initialization
  from the bosonicized indices, the graded-`env_init` rejection, and the
  builder marker in `_JIT_STEP_CACHE` / `_VJP_CACHE` keys (§3.3); settle
  the **move-level swap insertions** against the graded oracle, on
  fixtures verified to populate odd-parity boundary sectors (a
  vacuous-regime assertion in the test, per the #884 lesson — an
  even-sector-only fixture would pass with the signs entirely wrong);
  fixed point + energy vs the graded forward on the same states; an
  in-process legacy↔swap-gated alternation test that fails on cache
  conflation; audit (not delete) the FermionParity special cases the new
  path makes dead.
- **Phase 4 — adjoint.** First task: *re-validate* the phase gauge on
  the swap-gated layer (it is already wired and type-generic — §3.4;
  the earlier "verify availability / choose G∘f" framing is retired).
  Gates: gradient vs FD at D=2 parity-only; **a directional
  finite-difference gradient check on a nontrivial FermionicU1 (t-V)
  state** (requires Phase 0b's pipeline) — the parity-only FD case
  cannot catch charge-dual or parity-mapping errors specific to the
  U(1)×P adjoint, and every other gate in this plan exercises forward
  values only (Codex P2); and compile time within ~2× of the
  *bosonic-symmetric* CTM AD at the same block structure — that ratio,
  not an absolute number, is what option 1 buys.
- **Phase 5 — switch the fermionic AD paths** behind a flag, gated on
  both Phase 4 gradient checks (parity-only *and* FermionicU1). The flag
  is a named config field — **`iPEPSConfig.gs_fermion_backend`, values
  `"graded"` (default) and `"swap_gates"`** — documented in the README
  with a usage example (public-API contract; Codex round 7). The
  dispatch mechanism is structural, not nine hand-edits (self-review):
  **fold the backend into the effective `CTMConfig` in
  `build_ad_ctm_config` and emit it from `ctm_converge_kwargs`** — the
  exact `gs_projector_method` precedent (`ipeps_ad_policy.py:143–154`).
  Every optimizer-side forward (the one-/two-/multisite
  `_update_env_cache`, line-search, and `_eval_fresh` paths) already
  calls `python_loop_ctm_converge(**ctm_converge_kwargs(...))`, and all
  three dispatchers call `build_ad_ctm_config`, so two functions close
  the seam and a forgotten tenth call site becomes structurally
  impossible. This seam is the #938 mechanism verbatim — nine optimizer
  forwards silently dropping `gs_recipe`, five review rounds to close
  (and the base rev still exhibits the live precedent:
  `python_loop_ctm_converge` accepts `recipe`, `_ctm_python_loop.py:176`,
  but `ctm_converge_kwargs` never emits it). `make_ctm_energy_fn` alone
  is not the dispatch surface (Codex round 13). Not on the `optimize_fpeps_ad` wrapper either: that
  wrapper calls only the single-site `_optimize_gs_ad_tensor`, while the
  primary two-site fermionic workload — the finite-V t-V checkerboard
  implicated in #565 — routes through `optimize_gs_ad(unit_cell="2site")`
  to `_optimize_gs_ad_2site`, and a wrapper-level flag would leave it on
  the legacy builder with the two-site FermionParity tests never
  exercising the new path (Codex round 5). And because `"graded"` is the
  default, *unchanged* test configs keep running the legacy path — so
  the deprecation gate is not "the 26 files pass as they are" but
  **every backend-eligible FermionParity test parameterized over
  `gs_fermion_backend`** (Codex round 8), with the graded run staying
  the oracle and the `"swap_gates"` run the candidate; tests for
  configurations the flag rejects (split/explicit) assert the rejection
  instead. **Checkpoint resume**: `gs_fermion_backend` joins
  `_FATAL_CONFIG_FIELDS` (Codex round 10) — the current fatal list is
  only `(max_bond_dim, unit_cell, gs_c4v, gs_implicit_ad)`, so a
  backend flip across resume would be a *soft* diff and the restored
  `env_cache` would feed graded environments to the bosonic path or
  vice versa; a resume test covers both flip directions. Checkpoints
  written before the field existed carry no saved value, and
  `validate_config` compares the key union via `.get()` — so a missing
  saved value is **normalized to `"graded"`** before the fatal
  comparison (legacy graded checkpoints stay resumable), with an
  upgrade-path test alongside the two flip tests (Codex round 11).

  **Scope of the flag**: four builder-blind routes exist, not three
  (Codex rounds 7+9; the fourth from self-review). Two live in the
  policy — `fuse_virtual_legs=False` routes to the split engines
  (`ipeps_ad_policy.py:416`), and `gs_implicit_ad=False` to
  `ctm_energy_explicit` (:425). Two dispatch **above** it, from
  `optimize_gs_ad` before `make_ctm_energy_fn` ever runs: the
  root-implicit engines (`ipeps_optimize.py:752–757`) and
  `ctm_ad_mode="c4v_reference"` (:764) — the latter requires
  `gs_implicit_ad=True`, so it *passes* a rejection trio keyed on the
  first three and today dies on an off-topic TypeError
  (`ipeps_optimize.py:885–888`) instead of a clear refusal. So the rule
  is: `gs_fermion_backend="swap_gates"` combined with
  `fuse_virtual_legs=False`, `gs_implicit_ad=False`, **or any non-None
  `ctm_ad_mode`** is rejected with a clear error. And the rejection
  **cannot live in `ipeps_ad_policy`** — the above-policy routes never
  reach it; its home is the shared validation preamble of
  `optimize_gs_ad` / `optimize_fpeps_ad`, the `_reject_mislabelled_1x1`
  precedent (#938/#972; an earlier draft budgeted the validation at a
  surface two of the four routes bypass — self-review). `swap_gates` combined with the χ-bump/schedule fields
  (`chi_auto_bump`, `ctmrg_heuristic_increase_chi`,
  `gs_chi_schedule_steps`) is rejected at the same surface (§3.3).
  Wiring split/explicit/root-implicit/c4v — or the bump machinery —
  onto the builder is follow-up work, not Phase 5. The graded path stays
  until the parameterized suite plus a t-V energy replication pass on
  the new path. Deprecation is then a decision, not a side effect.

## 5. Risk register

1. **Bra-conjugation / dual parity convention** (Phase 2): the #555 failure
   class — #557's fix was the auto-Koszul + bar_super removal itself; the
   *shared canonical I-charges* repair belongs to #391/PR #393, the
   split-CTM bond-charge canonicalisation, and that is the actual
   precedent for the block-mispairing class the retyping map risks (an
   earlier draft crossed the two attributions; self-review). The
   graded→bosonic retyping map (§3.2) belongs to that class: a wrong
   dual/flow mapping mis-pairs blocks rather than mis-signing them. The
   graded-oracle equality test at 1e-10 is the guard for both; it runs per
   commit in Phase 2, not per PR.
2. **Phase gauge block-sparse** (Phase 4): RETIRED as an open question —
   already wired, type-generic, #362-hardened, and a hard precondition
   of the implicit path (§3.4; self-review). Residual exposure is only
   the small-env todense round-trip cost inside
   `_phase_fix_ctm_tensor` at large χ, re-measured in Phase 4's
   validation; G∘f stays a footnote-level contingency.
3. **#566 residual at U(1)×P block counts**: the block-sparse VJP wall
   belongs to block-sparse AD *generally* and is shared with the bosonic
   symmetric path. Parity-only fPEPS has ≲16 blocks per tensor, so exposure
   there is small; U(1)×P at scale eventually wants sweep-level batching
   (#566's recommendation; `core/stacked_tensor.py` is the seed). That is a
   separate track — this design's success gate is parity with
   bosonic-symmetric, deliberately not an absolute compile time.
4. **In-sweep statistics** (Phase 3, §3.3): the graded sweep injects
   signs the site double layer cannot carry — the fused edge
   absorption's χ⊕D² grouping signs AND the `_koszul_sign` reordering
   signs inside the projector SVDs (`linalg.py:384/:848`, the two
   CTM-reachable sites);
   getting either wrong is invisible on even-sector fixtures. Guard:
   the Phase 3 graded-oracle gate runs on states asserted to populate
   odd-parity boundary sectors and covers the full sweep including
   projector computation; if no fixed-diagram placement of insertions
   reproduces the oracle, the premise fails and the reform stops at
   Phase 3.

## 6. Blast radius

| Surface | Change |
|---|---|
| `core/tensor.py` graded machinery | untouched — legacy path keeps working; `swap_gate` is additive |
| New code | ~1.5–2k lines: one core method, one network-builder module (incl. the graded→bosonic retyping map), adjoint wiring, config validation in the `optimize_gs_ad`/`optimize_fpeps_ad` preamble (NOT `ipeps_ad_policy` — two of the four rejected routes bypass it), **and the Phase 0b FermionicU1 t-V pipeline** (charged gate + state init + energy/AD harness — nothing in-tree provides it) |
| Existing algorithm files | convergence + adjoint entry points gain the double-layer builder hook, **builder-owned env initialization**, and a **builder marker in the `_JIT_STEP_CACHE` / `_VJP_CACHE` keys** (Codex rounds 2–4); the **`_ctm_tensor_energy` helpers** thread the builder too (internal layer builds at :209/:325 plus the eight `_build_double_layer_open_tensor` sites — the seam a legacy-builder leftover would turn into a #938-style silent wrong forward); `_ctm_tensor_init/_moves` special cases audit-only, but the edge absorption gains a swap-insertion variant behind the builder marker if the Phase 3 oracle demands it (§3.3, risk 4; unsized until that decision); backend dispatch = `build_ad_ctm_config` + `ctm_converge_kwargs` (two functions, all entries); `swap_gates` + χ-bump/schedule fields rejected (§3.3), threading them is follow-up |
| Public API / docs | `swap_gate` documented on the exported `SymmetricTensor` class + README example (no `__all__` entry — methods are not module symbols); `iPEPSConfig.gs_fermion_backend` field + README example in Phase 5 |
| Tests | additive (~500 lines) through Phase 4; in Phase 5 every backend-eligible FermionParity test is parameterized over `gs_fermion_backend` (graded run = oracle, swap_gates run = candidate; ~26 files touched mechanically) — "unchanged files" would silently keep testing only the legacy default (Codex round 8); plus the measurement mixed-pair guard tests (§3.3) |
| PRs | ~7 — one per phase with 0a/0b separate, each independently green (an earlier draft said "~5" against its own six phases; self-review) |

## 7. Non-goals

- The fermionic **SU/BP-gauge** Koszul sign (#882 §5.2a) — a different sign
  in a different algorithm; separate gate on the fermionic-SU track.
- Sweep-level batching for U(1)×P block counts (#566) — tracked separately;
  see risk 3.
- Any change to the bosonic paths, dense or symmetric.
- Removing the graded formalism — it is the oracle, and SU evolution keeps
  using it until this design's Phase 5 has held for a while.
