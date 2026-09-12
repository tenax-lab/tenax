# Fermionic CTM AD via swap gates + fixed-point adjoint — design

**Status:** proposed · **Date:** 2026-09-12 · **Refs:** #565, #566, #618,
#630, #557, #555, #562, #881, #882 (§9.1 v2 track), #297 ·
**Reference implementation:** arXiv:2512.20697 (Chen, Neupert, Hasik — YASTN)

## 1. Why

Fermionic CTM AD is unusable at production scale for a *compile-time* reason,
not a numerical one. The graded formalism handles Koszul signs inside every
`SymmetricTensor` operation (`fermionic_ipeps.py:5` states this as a feature):
`core/tensor.py` computes per-block permutation signs (`_koszul_sign`, :104)
and bar/super-algebra signs (:1105) and applies them block-by-block in
transpose/contract. The signs themselves are trace-time *constants* — they
depend only on charge metadata — but the formalism forces every fermionic
contraction to trace per-block operations, and AD re-traces all of it in the
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
of unrolled sweeps.

## 2. Reference point: arXiv:2512.20697

Chen/Neupert/Hasik simulate fermionic fractional Chern insulators with iPEPS
in YASTN using exactly this pair of choices:

- **Swap gates for fermionic statistics**: rank-4 parity tensors at diagram
  crossings ("a minus sign only when both incoming legs are fermionic"), each
  leg decomposed into even/odd sectors. Signs are *network data placed at
  build time*, not logic inside contraction.
- **Fixed-point adjoint**: dE/dA = Σₙ (∂f/∂E)ⁿ ∂f/∂A at the converged CTMRG
  fixed point (their Eq. 13 — the Neumann form of the adjoint solve),
  "eliminating [the] memory bottleneck in plain backpropagation". Gauge is
  fixed *after* convergence-up-to-gauge by an invertible G per boundary
  tensor so the output is a strict elementwise fixed point of G∘f.

Caveat when citing it here: YASTN runs PyTorch **eager** — no tracing, no
compile wall; their cost is per-op dispatch (the wall we measured at ~40× in
JAX). The paper proves the formulation at physics scale (U(1)-charged
fermions, ~10³ CTM iterations). In our stack the swap-gate reform does double
duty: it is also what shrinks the *traced* graph to bosonic size, so the case
for it is stronger in JAX than in their stack.

## 3. Design

### 3.1 Swap-gate primitive

`SymmetricTensor.swap_gate(axes=(i, j))`: multiply each block by
`(-1)^{p_i(block) * p_j(block)}` where `p` is the Z2 parity grading of the
block's charge on that leg (`BaseSymmetry.parity`, `symmetry.py:192` — all
charge arithmetic stays behind the symmetry object, per #734). Properties:

- Metadata-driven ±1 per block, computed on host, applied as one fused
  multiply — a *constant* under tracing, so its VJP is the same multiply.
- Involution: `swap_gate(axes) ∘ swap_gate(axes) = id` (test anchor).
- Additive: no existing `SymmetricTensor` operation changes.

### 3.2 Swap-gated fermionic network builder

A new module builds the objects the CTM consumes, resolving the *fixed*
square-lattice diagram's crossings once, at construction:

- **Double layer**: bra-ket contraction of A with its conjugate has a known,
  layout-determined crossing pattern; apply `swap_gate` at each crossing
  while building the double-layer tensor. The output is a `SymmetricTensor`
  whose downstream contractions need no graded logic — the CTM treats it as
  bosonic.
- **RDM / gate insertion**: the 2-site RDMs and the Hamiltonian-gate
  insertion have finitely many crossings plus the parity string between the
  two physical sites (for hopping terms); both are diagram-determined and
  absorbed the same way, into the standard 2-site RDM contraction
  (`_ctm_tensor_energy` helpers).

The graded formalism is **not removed**: it keeps serving the existing SU /
energy paths and is the oracle this design validates against (the #562
pattern — native vs shim to 1e-10).

### 3.3 Forward CTM

Run the existing symmetric 2x2 tensor-CTM on the swap-absorbed double layer.
This should be configuration, not new code. The FermionParity special cases
in `_ctm_tensor_init` / `_ctm_tensor_moves` should be dead on this path;
they stay for the legacy path and get an audit note, not a deletion.

### 3.4 Fixed-point adjoint

Wire the implicit engines onto the new forward. Two candidates already in
tree: the fused Neumann VJP (`_ctm_energy_ad`, which already carries partial
`SymmetricTensor` handling around :382–:449) and the root-implicit symmetric
engine (`_ctm_root_implicit_symmetric`, `_ctm_root_implicit_sym_sectors`).

**Open design point (the one place new algorithmic code may be needed):**
the strict-fixed-point requirement. The dense path satisfies it with
`forward_gauge="phase"` enforced during iteration; whether phase gauge is
wired for the block-sparse forward must be *verified first* in Phase 4. If
it is not, the fallback is the paper's construction: converge up to gauge,
then solve for invertible per-boundary-tensor G so the output is a strict
fixed point of G∘f, and differentiate that map.

Standing constraint from the library's own history: the adjoint solve is
meaningless on an unconverged forward — starving `max_iter` shows up as
gradient garbage, not an error. Validation must gate on forward convergence
before judging the backward.

## 4. Phases

Each phase is its own PR; every phase gates on the graded-formalism oracle.

- **Phase 0 — measure before building.** Record current fermionic CTM-AD
  compile and wall time at D=2 and D=3, parity-only *and* U(1)×P (t-V), two
  seeds each (#882 §9.1's own instruction: confirm cost on a second seed and
  D before deciding anything). Pin graded-path energies as oracles. Every
  later claim is judged against these numbers; nothing is frozen from a
  single run.
- **Phase 1 — `swap_gate` primitive.** ~100 lines + tests (involution;
  parity bookkeeping against a hand-computed 2-leg case; a graded-transpose
  cross-check on a random small tensor). Mutation: dropping the sign must
  fail the cross-check.
- **Phase 2 — network builder.** Double layer + RDM/gate insertion with
  swaps absorbed. Gate: end-to-end energy equals the graded forward to
  ~1e-10 on Phase 0's oracle states. This phase owns the highest risk
  (see §5) and its oracle test runs per commit.
- **Phase 3 — forward CTM on the bosonicized layer.** Fixed point + energy
  vs the graded forward on the same states; audit (not delete) the
  FermionParity special cases the new path makes dead.
- **Phase 4 — adjoint.** First task: verify phase-gauge availability on the
  block-sparse forward; choose in-iteration phase gauge or post-hoc G∘f
  accordingly. Gates: gradient vs FD at D=2 parity-only; compile time within
  ~2× of the *bosonic-symmetric* CTM AD at the same block structure — that
  ratio, not an absolute number, is what option 1 buys.
- **Phase 5 — switch `optimize_fpeps_ad`** behind a flag. The graded path
  stays until the 26 FermionParity test files plus a t-V energy replication
  pass on the new path. Deprecation is then a decision, not a side effect.

## 5. Risk register

1. **Bra-conjugation / dual parity convention** (Phase 2): the #555 failure
   class — bar_super and auto-Koszul were removed for exactly this kind of
   sign ambiguity, and #557 fixed it with shared canonical I-charges. The
   graded-oracle equality test at 1e-10 is the guard; it runs per commit in
   Phase 2, not per PR.
2. **Phase gauge block-sparse** (Phase 4): may not exist; the G∘f fallback
   is more code and its gauge solve is itself a linear solve per boundary
   tensor. Verified first, decided second.
3. **#566 residual at U(1)×P block counts**: the block-sparse VJP wall
   belongs to block-sparse AD *generally* and is shared with the bosonic
   symmetric path. Parity-only fPEPS has ≲16 blocks per tensor, so exposure
   there is small; U(1)×P at scale eventually wants sweep-level batching
   (#566's recommendation; `core/stacked_tensor.py` is the seed). That is a
   separate track — this design's success gate is parity with
   bosonic-symmetric, deliberately not an absolute compile time.

## 6. Blast radius

| Surface | Change |
|---|---|
| `core/tensor.py` graded machinery | untouched — legacy path keeps working; `swap_gate` is additive |
| New code | ~1–1.5k lines: one core method, one network-builder module, adjoint wiring, `ipeps_ad_policy` validation |
| Existing algorithm files | audit-only (`_ctm_tensor_init/_moves` special cases); `optimize_fpeps_ad` gains a dispatch flag |
| Tests | additive (~500 lines); all 26 FermionParity test files run unchanged as oracles |
| PRs | ~5, one per phase, each independently green |

## 7. Non-goals

- The fermionic **SU/BP-gauge** Koszul sign (#882 §5.2a) — a different sign
  in a different algorithm; separate gate on the fermionic-SU track.
- Sweep-level batching for U(1)×P block counts (#566) — tracked separately;
  see risk 3.
- Any change to the bosonic paths, dense or symmetric.
- Removing the graded formalism — it is the oracle, and SU evolution keeps
  using it until this design's Phase 5 has held for a while.
