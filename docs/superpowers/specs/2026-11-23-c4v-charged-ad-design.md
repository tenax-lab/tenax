# Charged C4v iPEPS AD: the optimiser on the 2-sublattice sweep

**Status:** design, 2026-11-23. Spec 2 of 2. Spec 1 (the forward sweep,
`docs/superpowers/specs/2026-11-22-c4v-charged-u1-design.md`) merged as
#1211 (`2dcbab92`).
**Branch:** `feat/c4v-charged-ad`. **Follows:** #762, #763 (closed by #1211).

## Problem

A charged C4v `SymmetricTensor` has four faces with equal flow-weighted
charges and a virtual charge q with q != dual(q). It is the 2-sublattice
state (A, negate(A)). Since #1211, `ctm_tensor_c4v` runs a 2-sublattice
sweep on it and `compute_energy_ctm_tensor` measures it. No optimiser uses
that sweep:

- `jax.grad` through `ctm_tensor_c4v` on charged input raises. On
  `exact_u1_d2` (wrapped as below) at chi=10 the energy is −0.6602280177,
  and `jax.grad` raises `TracerArrayConversionError` (`__array__()` called
  on a traced array). The sweep ranks its candidates with `np.asarray`,
  `float()` and Python sorting, and its loop tests `float(diff)`.
- `optimize_gs_ad` with `unit_cell="1x1"` and `gs_c4v=True` densifies a
  `SymmetricTensor` `A_init` with no warning (`ipeps_optimize.py:2141-2155`
  at `2dcbab92`). It projects `A.todense()` onto the dense C4v basis and
  wraps the result as a `DenseTensor`. The 2-site C4v path does the same
  (`ipeps_optimize.py:4212-4233`). The run then optimises the dense C4v
  ansatz, not the U(1)/Z_n one, and keeps no block sparsity.
- `ctm_ad_mode="c4v_reference"` refuses a `SymmetricTensor` (`TypeError`),
  and the root-implicit modes refuse `gs_c4v=True`.

## What the user asked for

From the brainstorm on 2026-11-23:

- **Deliverable:** the optimiser end to end. `optimize_gs_ad` with
  `gs_c4v=True` accepts a charged U(1)/Z_n C4v `SymmetricTensor` and
  optimises it.
- **Success:** the gradient matches converged FD with an h scan, and runs
  reach the U(1)-C4v energies of Hasik et al.
- **Validation:** Heisenberg J1 (J2=0).
  - D=2 in CI: reach the U(1)-C4v optimum near peps-torch's −0.66023109348,
    and agree with the dense C4v optimiser on todense(A).
  - D=3 as a benchmark outside CI, against Hasik's U(1)-C4v J2=0 state.
  - FD-check the gradient at both D.

Assumptions the user accepted:

1. The entry is `optimize_gs_ad(H_rot, A_init, config)` with
   `unit_cell="1x1"` and `gs_c4v=True`. `A_init` is a charged C4v
   `SymmetricTensor`, and its charges fix the ansatz. The gate is the
   sublattice-rotated one, as on the dense 1x1 C4v path.
2. AD differentiates the fixed point of the #1211 sweep, not that of
   another CTM.
3. The layout (the kept states per charge) is chosen once and frozen for
   the run. A step where the free ranking would choose another layout is
   recorded; the run does not follow it.
4. Real, bosonic data only. Z_n works by construction and gets an FD smoke
   test, with no physics target.

## Scope

In scope: implicit AD through the 2-sublattice sweep, and an
`optimize_gs_ad` branch for charged C4v input on `unit_cell="1x1"`, for
U(1) and Z_n.

Out of scope:

- J2 or other NNN terms. `optimize_gs_ad` has none.
- `unit_cell="2site"` with charged input. It is refused (section 5).
- `envs_init`. It is already refused with `gs_c4v=True`.
- Complex and fermionic input. Complex is refused; fermionic input is not
  charged C4v (`_is_charged_c4v` returns False).
- The general `ctm_tensor` on charged input (#1210), and the pair splitting
  of the general symmetric CTM (#1101).
- A chi that changes during a run. It is refused (section 5).
- The #1211 review minors (`_negate_charges` drops `fuse_info`; the name
  "staggered U(1)" for Z_n) and the #833 docs sentence.
- The C4v-A1 ansatz gap of #1072 (D=2: −0.66023 against −0.66251
  unconstrained). It bounds the dense and the charged C4v paths equally.

## Approach

Fixed-point implicit AD of the 2-sublattice sweep, modelled on the dense
`c4v_reference` engine (`_ctm_tensor_c4v_reference_ad.py`). A
`custom_vjp`:

- the forward iterates the sweep to an element-wise fixed point at a frozen
  layout;
- the backward solves (I − Jᵀ)λ = g.

Why this one:

- It differentiates exactly the map that #1211 ships.
- It reuses the adjoint machinery of the dense reference: the Tikhonov
  shift and the residual threshold.
- It follows the certified recipe: an element-wise forward, the flow
  adjoint (the VJP includes the projector), and converged FD with an h
  scan.
- There is one decomposition per block, so values of different charges
  never meet in one backward. An SU(2) multiplet spans several charges, so
  this matters.

Rejected:

| Approach | Reason |
|---|---|
| C4v root-implicit engine | New characteristic equations for the pair blocks. The symmetric root engine costs 4.78 GB and about 180 s per gradient at D=2, chi=4, and the C4v root engine is validation-only. It is the fallback if the sign rule fails. |
| Dense engine on todense(A) | The dense eigh couples all charges, and a dense CTM does not keep the mirror pairs exact on a hard state. On the spec 1 test fixture the dense `ctm_tensor` and the block-sparse 2-site CTM differ by 5.1e-2 at chi=8 (2.4e-6 at chi=16), and the dense C4v corner has no exact (q, −q) pairs there. No U(1) saving. |
| Unrolled explicit AD | A truncated gradient, memory that grows with the sweep count, and 100% NaN at D=3 from the SVD backward on the degenerate discarded spectrum (#566, #687). Useful only as a test reference. |

## Measurements behind the design

Probes on `2dcbab92`, CPU, float64. The scripts are scratch files; the plan
re-runs what it needs.

**The fixture.** `exact_u1_d2()` (`tests/_ipeps_spectrum_fixtures.py`)
wraps exactly as a charged C4v tensor with `vc = pc = [0, 1]`: u and l OUT
with `vc`, d and r IN with `dual(vc)`, phys IN with `pc`, as
`c4v_fixture` in `tests/_c4v_staggered_fixtures.py`. Every nonzero entry
has u+d+l+r = s. C4v residual 1.1e-16; residual outside the blocks 0.

**Element-wise fixed point and decomposition hazards.** The #1211 eager
sweep ran to an sv-diff below 1e-13, then 30-40 more sweeps. Each sweep was
compared with the one before it, raw and after a sign alignment with
`_bond_component_phases`.

| Fixture, chi | Sweeps to sv-convergence | E | Raw element-wise residual over the extra sweeps | Sign flips | Argmax tie gap, block q=0 | Argmax tie gap, pair blocks | Smallest gap of kept values in a block |
|---|---|---|---|---|---|---|---|
| exact_u1_d2, 10 | 12 | −0.6602280177 | 6.0e-14, then about 2e-16 | 0 | 2.2e-14 | 0.81, 0.84 | 6.2e-3 |
| `c4v_fixture(U1Symmetry(), [0, 1], [0, 1])` (random, key 42), 12 | 228 | −0.5673694868 | 3.6e-13 → 5.9e-15 over 40 sweeps | 0 | 2.3e-15 | 6.5e-5, 0.35 | 7.5e-2 |

Three facts follow:

1. The raw sweep reaches an element-wise fixed point on both fixtures, with
   no sign flips.
2. The self-partner block (q=0) has an exact argmax tie in a kept column.
   The pair blocks have none. This fits the ket-bra swap, which maps sector
   q to dual(q) and sector 0 to itself.
3. No block has near-degenerate kept values.

**The energy gradient at a fixed env.** On exact_u1_d2 at chi=10, `jax.grad`
of `compute_energy_ctm_tensor`, with the env built from (C, T) by
`_c4v_staggered_to_full_env`, against a central FD along one random
direction. Every error falls as h² down to the noise floor.

| Derivative | Best relative error | h |
|---|---|---|
| ∂E/∂A | 2.5e-10 | 1e-6 |
| ∂E/∂C | 5.7e-12 | 1e-5 |
| ∂E/∂T | 7.5e-11 | 1e-6 |

**The C4v projection.** P = `from_dense ∘ symmetrize_c4v ∘ todense` on the
`vc = pc = [0, 1]` block data is traceable (`jax.jacrev`), with
|P − Pᵀ| = 0 and |P² − P| = 0. It has rank 2 of 5 block entries. So the D=2
ansatz of this charge layout has one free parameter after normalisation.

**The D=3 reference.**
`single-site_pg-C4v-A1_internal-U1/j20.0/state_1s_A1_U1B_j20.0_D3_chi_opt72.json`
in github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), optimised at
chi=72. Its `.dat` file:

| chi_obs | E | m |
|---|---|---|
| 36 | −0.6646014335383251 | 0.3713617774647949 |
| 72 | −0.6646014335619033 | 0.3713617774646437 |
| 180 | −0.6646014335619134 | 0.3713617774646436 |

The `.dat` energy evaluates the NN term on a 2x2 window; tenax uses a 2x1
window. On Hasik's D=4 J2=0.3 state, the tenax 2x1 error at chi=32 was 2.08
times the 2x2 error.

**The D=2 references.**

- peps-torch's D=2 U(1)-C4v energy: −0.66023109348 (`E_D2` in
  `tests/_ipeps_spectrum_fixtures.py`).
- tenax measures exact_u1_d2 at −0.6602280177 at chi=10.
- #1072: the dense 1x1 C4v optimiser reaches −0.66023109 at D=2, chi=16,
  from a random complex start (74 steps at tol 1e-9, |g| = 2.2e-5).

## Design

### 1. Architecture and data flow

Components (the plan may rename them):

1. `_ctm_tensor_c4v_staggered.py` (#1211). The projector splits in two:
   - `_c4v_staggered_layout(Qf, chi) -> BondLayout`: the global ranking,
     eager, as today;
   - `_c4v_staggered_projectors_at(Qf, layout)`: the traced projector of
     section 2.

   One helper builds P (keys, flows, the two faces) for both the eager
   `_c4v_staggered_projectors` and the traced builder. The eager path keeps
   its `_dense_svd` and eigh calls and its `max_abs()` normalisation, so
   the #1211 outputs stay bit-identical.
2. A new `_ctm_tensor_c4v_charged_ad.py`: the differentiable engine
   (sections 2 and 3).
3. A new `ipeps_optimize_c4v_charged.py`: the optimiser branch
   (section 5).

One loss evaluation:

- **Forward:** params → C4v projection → A →
  `charged_c4v_converge(A, warm, layout, config.ctm)` → (C*, T*) →
  `_c4v_staggered_to_full_env` → `compute_energy_ctm_tensor` → E.
- **Gradient:** ∂E/∂A directly, plus (∂F̃/∂A)ᵀλ through the `custom_vjp`,
  then back through the projection.

Unchanged:

- `ctm_tensor_c4v` stays eager-only, so `jax.grad` through it still
  raises.
- The dense C4v paths.
- Self-dual symmetric input (Z2, all-zero U(1)). The default path still
  densifies it.

### 2. The frozen sweep and its gauge

**Layout.** L gives the kept states per chi charge, with
count(q) = count(dual q). Its type is `BondLayout`
(`_ctm_root_implicit_sym_sectors.py`): frozen, hashable and not a pytree,
so it is a static argument.

**The traced projector** `_c4v_staggered_projectors_at(Qf, L)`:

- **Pair block** (q, dual q), with q the first of the two in sorted order,
  as today: `truncated_svd_ad(M_q, k_q)`. The U columns go to sector q, the
  V columns to sector dual(q). Both copies of a pair value come from one
  SVD, so no backward term divides by their zero gap (the #1178
  mechanism).
- **Self block** (q = dual q): `regularized_eigh` of 0.5(M + Mᵀ). Keep the
  k_q largest |λ| (an argsort with a static k).
- A sector with k_q = 0 gets no decomposition.

**The map the AD differentiates:** F̃(e; A, ref) = align(F_L(e, a(A)), ref).

- **F_L** is the #1211 sweep with the traced projector: C′ = PᵀQP and
  T′ = negate(Pᵀ(T a)P). Each is divided by `_max_abs_scale`
  (`_ctm_tensor_moves.py`), which gives max|·| as its value and the
  gradient of a smooth norm.
  - Why not the plain `max_abs()`: the ket-bra swap ties the largest
    entry of a double-layer env exactly. With a plain max there, a 1e-15
    change to a D=3 U(1) chi=9 env moved Jᵀv by 15%
    (`tests/test_ctm_normalize_maxabs_kink.py`).
  - Why not a frozen scale: the derivative of the normalisation removes
    the scale direction from J. Without it, the adjoint solve stops
    converging (same file).
- **align** puts one sign g_i on each chi index:
  - X_ik = Σ_d T′[i,d,k]·T_ref[i,d,k] = g_i g_k W_ik, with W ≥ 0;
  - `_bond_component_phases` (`ad_utils.py`, the #841 `bond_phase`
    solver) gives g for each static connected component of the block
    pattern of W;
  - apply T → gTg and C → gCg;
  - compute g from `stop_gradient` inputs. The signs are piecewise
    constant, so their derivative is zero.

  C is diagonal and carries no sign information; T carries all of it.

**Why alignment, not a sign rule inside the decomposition:**

- The q=0 block has an exact argmax tie (Measurements), so an argmax rule
  is a rounding coin flip there.
- A covariant rule copies the input sign of another chi index. That is a
  map on signs, and it can cycle: on the 2x2 path a whole-column overlap
  rule gave a period-2 chi-sign cycle, and #1055 rejected it.
- Alignment recovers a pure sign gauge exactly, whatever sign each
  decomposition returns. Where the raw signs were stable, it changed
  nothing (0 flips).

**The forward** is an eager loop over a jitted F̃, compiled once per
layout:

1. **Cold start** (the first evaluation of a run): the #1211 eager loop runs
   to sv-convergence (`config.ctm.conv_tol`, `max_iter`). Its last layout is
   L. If it does not converge within `max_iter`, raise
   `CTMNotConvergedError`: a layout from a transient env is not frozen.
   `_ctm_tensor_c4v_staggered` returns no convergence flag today, so the
   cold start adds one.
2. **Element-wise phase:** iterate e ← F̃(e; A, ref=e) until
   max(|ΔC|/max|C|, |ΔT|/max|T|) < `config.ctm.conv_tol`. At `max_iter`,
   raise `CTMNotConvergedError`; there is no adjoint without a fixed
   point.
3. **Warm start** (later evaluations): step 2 only, from the warm env of
   section 5. The layout and the shapes are the same, so nothing recompiles.

**Not handled:** a continuous gauge, i.e. degenerate kept values inside one
block. It makes step 2 fail and raise. The measurements found no case (the
smallest kept gap was 6.2e-3).

### 3. The fixed-point VJP

`charged_c4v_converge(A, warm, layout, config) -> (C*, T*)` is a
`jax.custom_vjp` with `layout` and `config` in `nondiff_argnums`. `warm` is
a normal argument with a zero cotangent: the fixed point does not depend on
the start.

**Forward:** the element-wise phase of section 2, from `warm`. It saves
(A, C*, T*).

**Backward**, as `_ctm_tensor_c4v_reference_implicit_backward`:

1. **Jacobians.** J is the VJP of e ↦ F̃(e; A, ref=e*) at e*. ∂F/∂A is the
   VJP of A′ ↦ F̃(e*; A′, ref=e*). The reference is a constant here, so J
   is the Jacobian of the forward's own iteration map, and the forward's
   measured contraction rate estimates ρ(J).
2. **Solve** (I − Jᵀ + τI)λ = (g_C, g_T) with `gmres_eager`
   (`_gmres_eager.py`): τ = `adjoint_tikhonov` (default 1e-6, applied as
   in the reference), tolerance `adjoint_tol`, `adjoint_maxiter` restarts,
   Krylov dimension 30 (the C4v root engine's default). The (C, T) pytrees
   are flattened with `jax.flatten_util.ravel_pytree`.
3. **Result:** dA = (∂F/∂A)ᵀλ. Return (dA, a zero cotangent for `warm`).

**Why `gmres_eager` and not the reference's `jax.scipy` solvers**
(BiCGSTAB by default, or GMRES): JAX's GMRES traces the operator about nine
times. On the root engine the jaxpr went from 76,884 equations for the
matvec to 700,013 for the solve. For a block-sparse sweep VJP (the #566
wall) that is the whole compile cost. The C4v root engine already uses
`gmres_eager`.

**What the gradient includes:**

- the full projector response through `truncated_svd_ad` and
  `regularized_eigh` (the flow adjoint; there is no frozen-projector
  option);
- nothing from the signs or the layout, which are piecewise constant.

**Fail closed:**

- If the relative residual ‖b − Ax‖/‖b‖ that `gmres_eager` measures is
  above max(10·`adjoint_tol`, 1e-12), raise `RuntimeError`. The threshold
  is the reference's. The reference applies it to the absolute residual,
  which is loose when the cotangent is small near an optimum.
- A non-finite λ or dA raises `RuntimeError`.
- There is no Arnoldi precheck. Element-wise convergence of the forward
  already indicates ρ(J) < 1, and the residual gate catches a hard solve.
- Metadata records the forward sweeps, the contraction rate, and the GMRES
  restarts and residual.

**Compile:** three jitted programs per layout: F̃, the matvec
v ↦ v − Jᵀv + τv, and λ ↦ (∂F/∂A)ᵀλ. The shapes do not change during a
run, so every optimiser step reuses them.

### 4. The energy

- **Loss.** The loss is
  `compute_energy_ctm_tensor(A, _c4v_staggered_to_full_env(C*, T*), gate, d)`.
  This is the #1211 charged branch: a metadata guard, then the 2-site energy
  of (A, negate(A)) with (env, negate(env)). No code change.
- **Gradient.** JAX supplies ∂E/∂A and the env cotangent (g_C, g_T) that
  section 3 receives. The Measurements show both are correct.
- **Cost.** The 2-site formula measures four bonds, and C4v makes them
  equal. A one-bond formula would cost less and give the same C4v-projected
  gradient. Keep the tested formula, unless the energy VJP is a measured
  cost at D=3.
- **Gate.** The Heisenberg AFM uses
  `sublattice_rotate_gate(heisenberg_gate())`.

### 5. Optimiser wiring

**Dispatch in `optimize_gs_ad`:**

- A new gate `use_charged_c4v_path(config, A_init)`, in `ipeps_ad_policy.py`
  next to `use_reference_c4v_path`. It is true for `unit_cell="1x1"`,
  `gs_c4v=True`, `ctm_ad_mode=None` and `_is_charged_c4v(A_init)`. It reads
  index metadata only. The branch goes after the `c4v_reference` branch and
  before the default 1x1 branch, which densifies today.
- `unit_cell="2site"` with a charged `A_init` is refused, because the 1x1
  charged ansatz is already the 2-sublattice state.
- Unchanged:
  - `c4v_reference` still refuses a `SymmetricTensor`;
  - the root-implicit modes still refuse `gs_c4v`;
  - `envs_init` is still refused with `gs_c4v`;
  - with `A_init=None`, SU builds a dense state and the dense path runs as
    today.

**The loop**, modelled on `_optimize_gs_ad_tensor_reference_c4v`:

1. **Start.** Project `A_init` to C4v and normalise it. The cold start gives
   env₀ and the layout L, which stays frozen for the run.
2. **Loss.** The parameter is the block pytree of A. The loss computes
   A = normalise(P(params)), runs `charged_c4v_converge` from the warm env,
   and returns the section 4 energy.
3. **Optimiser.** L-BFGS with the configured line search
   (`_build_optimizer`, with `value_fn` as in the reference loop).
   Best-point tracking, the `grad_norm` exit test (`gs_grad_norm_tol`) and
   the count of non-finite gradients work as in the reference loop.
4. **Warm start.** The warm env is the converged env at the last accepted
   point. Every evaluation starts from it, line-search trials included. Only
   the evaluation at an accepted point replaces it, so a failed trial cannot
   move the next start.
5. **Layout check.** After each accepted step, one free-ranking evaluation
   (`_c4v_staggered_layout`) runs at e*. A mismatch with L is recorded and
   logged once.
6. **Return.** The loop returns `(A, env, E)` with
   env = `_c4v_staggered_to_full_env(C*, T*)`, the env that
   `ctm_tensor_c4v` returns for charged input. `return_history=True` adds,
   per step: the energy, |g|, the step time, the forward sweeps and the
   GMRES restarts. It also adds `converged`, the layout and the steps with a
   layout mismatch. With `gs_num_steps=0` the loop returns the projected
   `A_init`, its converged env (the cold start, then the element-wise phase)
   and its energy, as the reference loop does.

**Failures:**

- **Forward** not element-wise converged: `CTMNotConvergedError`. The loop
  resets to the best params and the best env, with a fresh L-BFGS state,
  within `gs_stall_recovery_retries`. When no retries are left, it
  re-raises. This is the 1-site rule (#1059).
- **Adjoint solve** fails: `RuntimeError`, with no recovery. This is the
  reference rule.

**Config.** Each refusal states its reason.

| Setting | Behaviour |
|---|---|
| `gs_implicit_ad=False` | Refused: there is no explicit path. |
| Complex data | Refused: the mirror pairing holds for real data only (#1211). |
| `projector_method` | `"svd"` (the default) and `"eigh"` are accepted. Both select the #1211 block projector: one SVD per pair block, one eigh per self block. `"qr"` is refused. `gs_projector_method` overrides the setting first, as on `c4v_reference`. (`ctm_tensor_c4v` itself still accepts only `"eigh"` on charged input.) |
| `forward_gauge` | `"auto"` and `"bond_phase"` are accepted. Any other value is refused. |
| `ctm_conv_method="sv"` | Refused: the adjoint needs an element-wise fixed point. `"elementwise"` (the default) is accepted. |
| `projector_backward` | `"auto"` and `"flow"` are accepted. `"standard"` and `"lorentzian"` are refused. |
| `on_unconverged="warn"` | Refused: without a fixed point there is no gradient. |
| `chi_ramp`, `chi_auto_bump`, `ctmrg_heuristic_increase_chi` | Refused: the layout is frozen. |
| `gs_stall_recovery="noise"` | Refused: the noise kick assumes dense params, as on the 2-site C4v path. |
| `gs_checkpoint_path` | Refused: not wired, as on `c4v_reference` (#497). The existing check gets the charged gate. |
| `cg_gates` | Refused: the charged branch has no coarse-grained energy. |
| `fuse_virtual_legs=False` | Refused: the split CTM is dense-only. |
| `gs_recipe="1x1"` | Refused already: the #938 check refuses it on every fused path. |
| `gs_metric_precond` | Not applied, as on the dense C4v path. |
| `adjoint_solver`, `adjoint_method` | Not read: the solve is `gmres_eager` only (section 3). The reference reads `adjoint_solver`, whose default is `"bicgstab"`. |
| `adjoint_arnoldi_precheck` | Not read: there is no precheck (section 3). |

The `optimize_gs_ad` docstring lists the settings that are not applied or
not read.

### 6. Validation

**The FD rule**, for every gradient check below:

- **Sample points.** A central FD of the converged energy. At each sample
  point the forward is re-converged element-wise to 1e-13 from e*(A), so
  every point uses the same layout and the same gauge branch. Without this
  an FD of a re-converged energy can diverge as h shrinks: on
  `ctm_energy_implicit` it went from −0.105 to −1.63 as h fell from 1e-3 to
  1e-6.
- **h scan.** h runs from 1e-2 to 1e-6. Only a plateau counts. h is chosen
  by the agreement of neighbouring h values, never by the agreement with
  AD.
- **Report.** |g·v|, fd_unc and grad_err for each check.
- **Regime asserts.**
  - |g·v| must be well above the FD floor. A check at an optimum is
    vacuous.
  - At the test chi, |g_flow − g_frozen| must be above 10·fd_unc, or the
    projector term goes untested. So the FD checks use a small chi. The
    test gets g_frozen by a monkeypatch that puts `stop_gradient` on P; the
    engine has no frozen option.

**CI tests, D=2.** The plan sets the markers (core or algorithm) from
measured run times.

| Test | What it checks | Named mutation it must catch |
|---|---|---|
| FD gradient | Two fixtures at chi of about 6: exact_u1_d2 moved off its optimum (one C4v direction), and the #762 charges `vc=[-1, 1]`, `pc=[-2, 0]` (3 C4v invariants; 2 random directions) | P frozen (`stop_gradient`); λ = g (no solve) |
| One-sweep VJP | FD of ⟨w, F̃(e₀ + hv)⟩ from a fixed start e₀ | the same two |
| Max-abs kink | Jᵀv changes by less than 1e-10 relative under a 1e-15 relative change of e*. Regime assert: the largest \|entry\| of T is tied at e*. | the plain `max_abs()` in F_L |
| D=2 optimum | Two starts: exact_u1_d2 moved off its optimum, and `c4v_fixture(U1Symmetry(), [0, 1], [0, 1])`. At chi=16 each run ends with E ≤ E_tenax(exact_u1_d2; 16) + 1e-9, \|E − (−0.66023109348)\| ≤ 1e-6 and \|g\| < 1e-5 | a wrong B-site in the energy |
| Dense parity | The dense `ctm_tensor` energy of todense(A*) equals E_charged(A*) at chi=16 within 1e-10. The dense 1x1 C4v optimiser (`gs_c4v=True` on `DenseTensor(todense(A_init))`) reaches the same energy within 1e-8 at chi=16; #1072 measured −0.66023109 there. | — |
| Z4 smoke | FD in one direction on the #1211 Z4 fixture `c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1])` | — |

**Mechanism tests** (mocked signals, no convergence claims):

- **Dispatch:** charged input reaches the new branch and is not densified;
  `unit_cell="2site"` with charged input is refused; each refusal row of
  section 5 raises.
- **Alignment:**
  - align(G·e·G, e) returns e exactly for random signs G;
  - with a mock F_L that flips one chi sign on every second call, the
    element-wise phase still converges. Without alignment, it does not.
- **Reset:** a mock `CTMNotConvergedError` at step k gives a reset to the
  best point. When no retries are left, the loop re-raises.
- **Layout mismatch:** a mock free ranking that returns another layout at
  step k is recorded, and the run keeps L.
- **Solve gate:** a mock residual above the gate raises `RuntimeError`.
- **#1211:** the existing tests stay green, with bit-identical outputs.

**Benchmark, D=3, outside CI** (`benchmarks/ipeps_c4v_charged_ad.py`):

1. **Load.** Read Hasik's D=3 state from a local copy of the repository
   (a path argument). `load_state` and `load_charges` in
   `benchmarks/ipeps_spectrum_hasik.py` read the tensor and its charges.
   `SymmetricTensor.from_dense` builds the charged tensor, and the benchmark
   asserts `_is_charged_c4v` on it.
2. **Energy check.** Measure the tenax energy at chi = 36, 54 and 72 and
   compare it with the `.dat` values. Hasik's energy changes by about 3e-11
   over chi = 36-180, so the 2x1-against-2x2 window gap should be small. The
   benchmark measures it and does not assume a value.
3. **FD check.** Apply the FD rule at a perturbed state, in 2-3 C4v
   directions.
4. **Optimise.** Start from a perturbed Hasik state at chi=36, and from a
   random charged start. The target is E ≤ E_tenax(Hasik; 36) + 1e-9 with
   |g| < 1e-5.
5. **Report.** Steps, wall time, the first-gradient compile time and the
   peak RSS.
6. **Hardware.** CPU first (cores 0-11, load-gated). Use the A100 if the
   compile cost (#566) is too high.

**Docs:**

- the `optimize_gs_ad` docstring: the branch, the refused and unread
  settings;
- `docs/guide/algorithms/ipeps_ad_paths.md`: the charged C4v path, its
  options and caveats, and the measured D=2 and D=3 numbers;
- one feature line in `README.md` that links to the guide page.

There is no new public symbol.

## Points the plan settles

Each point has a method. None is open in the design.

- **Test markers:** measured run times decide core or algorithm, and the
  `conftest.py` map gets the entries.
- **The FD chi and fixtures:** the regime asserts (|g·v| above the floor;
  flow against frozen above 10·fd_unc) are measured before the tests are
  written.
- **The matvec program:** it either re-linearises per call or receives the
  linearised residuals as a `jax.tree_util.Partial`. Measured compile and
  run times decide.
- **The D=2 tolerance against −0.66023109348:** measured at chi=16 first.
  A gap above 1e-6 goes to the user.
- **The max-abs kink fixture:** the plan measures whether the largest
  |entry| of T* is tied on the D=2 fixtures. If it is tied on neither, the
  test uses a D=3 charged fixture.
- **The chi of the dense parity check:** a dense CTM is a reference only
  where its corner keeps the (q, −q) pairs exact (spec 1). The plan checks
  the pairs of the dense corner of exact_u1_d2 at chi=16 before it writes
  the test, and picks another chi if they are not exact.
- **The GMRES restart:** 30 to start with. If a solve stalls, raise the
  restart, not the tolerance (cf. #1153 on macOS).
- **The `return_history` keys:** aligned with the 1x1 path where they
  apply.
- **The D=3 hardware:** CPU first; the A100 if the first-gradient compile is
  too slow.

## References

- Spec 1 and #1211 (`2dcbab92`): the 2-sublattice sweep.
- `_ctm_tensor_c4v_reference_ad.py`: the template engine.
- `_gmres_eager.py`, `ad_utils._bond_component_phases`,
  `_ctm_tensor_moves._max_abs_scale`, `_ad_primitives.truncated_svd_ad`,
  `_ad_primitives.regularized_eigh`,
  `_ctm_root_implicit_sym_sectors.BondLayout`,
  `_ctm_convergence_policy.CTMNotConvergedError`.
- #841 (the `bond_phase` gauge), #1055 (the rejected sign rule), #1059 (the
  reset on `CTMNotConvergedError`), #566 and #687 (the block-sparse compile
  wall and the explicit-AD NaN), #497 (checkpoint wiring), #938 (the
  `gs_recipe` check), #1072 (the C4v ansatz gap), #1101 (pair splitting and
  the eager layout cycle in the general symmetric CTM), #1153, #1178, #1210.
- Francuz et al., Phys. Rev. Research 7, 013237: implicit AD of CTM.
- The states of Hasik et al.: github.com/jurajHasik/j1j2_ipeps_states.
