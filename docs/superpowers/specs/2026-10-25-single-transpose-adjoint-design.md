# One transpose: the joint adjoint solve as one program

**Status:** approved in chat 2026-10-25, written for review · **Author:** Claude Code (for @yingjerkao)
**Branch:** `perf/linearize-once-adjoint` (same concern: the adjoint compile cost)
**Amends:** `2026-10-25-linearize-once-adjoint-design.md` §4 (its §9 has the A1 no-go that motivates this)
**Related:** #566, #858/#1087 (GMRES seeds), #1123 (joint chain rule), #1111 (flow slope check)

## 1. Goal

Compile the transpose J^T v once per backward, not twice. Today the head of
this branch compiles it in `_jit_neumann_bwd` and again in
`_jit_gmres_joint_solve`. Gate A1 of the parent spec (backward compile at
D=3 chi=16 <= 0.7x main) must pass. The gradient stays equal to today's to
round-off.

## 2. Why

Parent spec §9. At D=2 chi=8 (CPU) the head's step-1 solver programs hold
773,305 optimised HLO instructions against main's 830,453 (0.931); at D=3
chi=16 (GPU) the backward compile ratio is 0.945. The two solver programs
are about the same size (79.5k lowered ops each), and nearly all of each is
J^T v. One copy fewer removes 264k-307k of the 773k optimised instructions.
Estimate (not measured): about 0.61 of main at D=2 chi=8.

## 3. Decisions taken in chat

- λ must equal today's Neumann -> GMRES result to round-off. A change of
  solver strategy (for example GMRES only) is out.
- Approach A: Neumann becomes one phase of the GMRES phase machine, in one
  compiled program with one matvec call site. Host-driven loops (approach B)
  and host callbacks (approach C) are rejected: a host sync every iteration
  puts gate A2 at risk, and B needs a second GMRES implementation.

## 4. Design

### 4.1 Programs and data flow (default joint path)

```
vjp_joint = _jit_linearize_joint(params, env, chi, pb)          # unchanged
dE_denv   = _jit_dE_denv(params, env)                           # unchanged
out       = _jit_joint_adjoint_solve(vjp_joint, dE_denv, init_lam, use_init,
                                     tol=, maxiter=, restart=)  # NEW
direct    = _jit_direct(params, env)                            # unchanged
```

- `_jit_joint_adjoint_solve` replaces `_jit_neumann_bwd` and
  `_jit_gmres_joint_solve`. It returns one tuple:
  `(diverged, converged, n_iter, lam_neumann, abs_resid_neumann, lam,
  abs_resid, b_norm, indirect)`.
- `_bwd_core` does the same host work after the one call:
  `_F3_LAST_DIAGNOSTICS`, the amplification diagnostic,
  `warm_start_invalidated`, `_report_adjoint`, `prev_lam`, the #1111 slope
  check.
- The "Eager GMRES" debug line moves from before the GMRES call to after the
  joint solve, still written only when Neumann did not solve the system.
- Fusion: the program holds the params VJP, so it is called through
  `_fusion_variant(..., by_bond_dim=True)`. The Neumann iterations move from
  the `by_bond_dim=False` rule to `True`.
- `adjoint_method="gmres"` with `"joint"` calls the same program with the
  Neumann phase off.

**Not changed:** `adjoint_chain_rule="separate"`,
`_jit_gmres_one_matvec_solve`, `_jit_fused_fixed_point_bwd`, the split CTM,
the forward CTM, `_jit_linearize_joint`, `_jit_dE_denv`, `_jit_direct`.

### 4.2 The phase machine (`src/tenax/algorithms/_gmres_lax.py`)

- The phase functions of `gmres_single_matvec` (`_SEED`, `_RESID`,
  `_ARNOLDI`) move into one shared builder. `gmres_single_matvec` keeps its
  signature and behaviour; with the Neumann phase off it traces no Neumann
  code, so its lowered program stays identical to main's.
- New entry point:
  `neumann_then_gmres(jt_aux, b, init_lam, use_init, *, tol, maxiter, restart)`.
  `jt_aux(v)` returns `(J_env^T v, J_params^T v)`. The `apply` step forms
  `A v = v - J_env^T v` for the GMRES phases only.
- New phase `_NEUMANN`, the first phase. Its arithmetic is that of
  `_neumann_loop` today:
  - λ_0 = `where(use_init, init_lam, b)`;
  - `new = b + J_env^T λ`; `delta = new - λ`;
  - `diff` = ℓ1-of-ℓ2 over the leaves (per-leaf norms from static leaf
    offsets in the flat vector); `resid` = L2 of `delta`;
  - divergence guard `diff > prev_diff & k > 5`;
  - stop on `k >= maxiter`, divergence, or `diff <= tol`.
- Handoff to `_SEED`: `solved = converged & ~diverged`; the candidates are
  `[λ_neumann if solved else 0, b]`, the list the host passes today. The
  candidates become loop state, not a static stack. The seed rules (#858),
  the measured-residual exits and the `aux` carry stay as they are.
- Cost: each Neumann iteration also computes `J_params^T v` (it shares the
  call site) and drops it. At D=2 chi=8 that half is ~250 lowered ops. Gate
  A2 measures the runtime effect.

### 4.3 Errors and diagnostics

Unchanged in behaviour:

- Neumann divergence or non-convergence -> GMRES from zero and `b`;
  `warm_start_invalidated=True`.
- A non-finite Neumann `diff` is not converged (`inf > tol`; NaN fails
  `<=`). The `_RESID` NaN rules are unchanged.
- `maxiter=0`: zero Neumann steps, `inf` residual (fails closed), then
  `_SEED`.
- `n_iter`, `converged`, `diverged`, the Neumann `adjoint_residual`, then the
  GMRES `adjoint_residual` written over it reach `_F3_LAST_DIAGNOSTICS` with
  today's values and in today's order.

## 5. Testing

All tests of the parent spec stay, with the program names updated. New or
changed:

1. **One transpose copy.** The lowered `_jit_joint_adjoint_solve` has a
   `dot_general` count <= today's `_jit_gmres_joint_solve` count + 16, and 0
   SVDs.
2. **λ equals today's to round-off.** On the `any_size` U(1) and fermionic
   fixtures, `(lam_neumann, n_iter, diverged, converged, lam, indirect)`
   matches today's `_jit_neumann_bwd` then `_jit_gmres_joint_solve` at rel
   1e-12: cold start, warm start, and a forced-divergence case (small
   `maxiter` or the existing mock signal; assert the regime). The two old
   programs stay as test-only oracles until the last task deletes them.
3. **Gradient equals `"separate"`** to rel 1e-10 (existing tests).
4. **Compile once:** 0 new jaxpr traces over backwards 2 and 3
   (`/jax/core/compile/jaxpr_trace_duration`).
5. **Fusion:** a large U(1) state takes the `by_bond_dim=True` variant.
6. **`gmres_single_matvec` unchanged:** with the Neumann phase off, its
   lowered text equals main's on a fixed operator.

**Mutants** (commit first, unique anchors, `PYTHONDONTWRITEBYTECODE=1`,
delete `__pycache__`):

| Mutant | Change | Killed by |
|---|---|---|
| N1 | the Neumann phase calls the pullback at a second site | test 1 |
| N2 | the Neumann update uses `v - J^T v` in place of `J^T v` | test 2 |
| N3 | the handoff seeds λ even after divergence | test 2 (divergence case) |
| N4 | the divergence guard is dropped | test 2 |
| N5 | fusion uses `by_bond_dim=False` | test 5 |

## 6. Gates

| Gate | Criterion |
|---|---|
| G0 | CPU pre-gate, before any GPU time: D=2 chi=8 step-1 optimised HLO of linearise + dE_denv + joint solve <= 0.7 x 830,453 (main). Above 0.8: stop and report. |
| A1-A4 | As the parent spec §6, at D=3 chi=16 on GPU 1, base = main. Grade step 1; record step 2 (both arms recompile there). |

The GPU A/B runs with no other heavy job on the host, under the
host-memory guard (kill the benchmark below 12 GB MemAvailable). GPU 0 runs
another job; never touch it.

## 7. Rollout

Replace in place, no flag. Update `docs/guide/algorithms/ipeps_ad_paths.md`
for the new program list. PR only on the user's approval; never armed
unasked.

## 8. Out of scope

- Folding `_jit_dE_denv` into the linearise program (fewer programs, same
  total ops).
- The step-2 recompile on a new chi layout (both arms; a separate concern).
- `"separate"`, the split CTM, the C4v path, the forward CTM.

## 9. Results (2026-10-27)

Run conditions: GPU 1 (CUDA_VISIBLE_DEVICES=1), D=3 chi=16, `--steps 4`,
compile cache off, cores 32-47, host-memory guard at 12 GB. Base `b71f5a5`
(the parent spec's base arm), head `49661c2` (`57d5deb` changes tests only).
G0 ran on CPU at D=2 chi=8.

**G0 = 0.614: PASS** (bar 0.7). Step-1 programs, lowered StableHLO ops /
optimised HLO instructions:

| Program | Base | Head |
|---|---|---|
| `_jit_fused_fixed_point_bwd` | 160,289 / 435,955 | — |
| `_jit_gmres_joint_solve` | 143,315 / 394,498 | — |
| `_jit_linearize_joint` | — | 62,571 / 128,771 |
| `_jit_dE_denv` | — | 23,349 / 73,723 |
| `_jit_joint_adjoint_solve` | — | 79,869 / 307,395 |
| **Total** | **303,604 / 830,453** | **165,789 / 509,889** |

The joint solve (Neumann phase and GMRES in one program) is the size of the
parent head's GMRES program alone (79,696 / 306,806): the second copy of
J^T v is gone. The solve holds 0 SVDs after optimisation.

**A1 = 0.685: PASS** (bar 0.7). Step-1 backward compile, seconds:

| Program | Base | Head |
|---|---|---|
| `_jit_fused_fixed_point_bwd` | 790.2 | — |
| `_jit_gmres_joint_solve` | 1116.4 | — |
| `_jit_linearize_joint` | — | 163.5 |
| `_jit_dE_denv` | — | 106.4 |
| `_jit_joint_adjoint_solve` | — | 917.0 |
| `_jit_direct` | 496.9 | 458.4 |
| **Total** | **2403.5** | **1645.3** |

Step 2 recompiles the backward on both arms (a new chi layout): base
5127.2 s, head 3145.1 s (linearize 339.9 + dE_denv 212.5 + joint 2030.6 +
direct 562.1), ratio 0.613.

**A2 = 1.084: MISS** (bar 1.05). Median `step_times`: head 35.943 s (steps
`[4.911, 48.272, 35.943]`), base 33.145 s (`[6.446, 49.911, 33.145]`). This
is the clean rerun. The first head run (1.097) had a CPU job beside it, so it
was not graded; the clean run repeats it to ~1%. The sum of `step_times` is
equal (89.13 s head, 89.50 s base, 0.996): the first two entries are faster
on head (-24%, -3%), the third is slower (+8.4%). The cause is not measured.
Candidates: each Neumann iteration also computes `J_params^T v` and drops it
(§4.2); the Neumann loop reads the saved pullback residuals instead of
recomputing them in one fused program; each iteration passes a 4-way
`lax.switch` whose state carries the GMRES basis.

**A3 = 8.0e-15: PASS** (bar 1e-8). Relative L2 distance of the SU-state
gradients. Step-1 E = -0.5529995394 and GMRES resid = 1.329e-06, as base.
Energies at steps 1-4 equal base.

**A4 = 1.024: PASS** (bar 1.2). `device_peak_bytes` 1,597,896,448 head /
1,561,049,088 base. The value is process-lifetime: SU and the forward CTM
are included.

Optimize wall time: 12,723 s head, 16,103 s base (0.790). Most of the gain
is compile time.
