# Joint-VJP chain rule for the implicit-AD CTM backward

Status: design, approved in conversation 2026-10-18.  Opt-in; a separate PR
flips the default only if the D=3 GPU measurement justifies it.

## Problem

After #1112 (single-matvec GMRES) the D=3 chi=16 GPU profile on `ac712ac`
spends step 1 at 7,909 s, and the largest single compile is now
`_jit_chain_rule` at 2,604 s.  The chain rule is

    grad = g * (dE/dparams |_env  +  J_params^T lambda)

and ~97% of its compile is the indirect term, the params-VJP of the
gauge-fixed CTM sweep.  That VJP is the same sweep, linearised at the same
point, that the adjoint solve already differentiates with respect to the
environment at its matvec call site.  Compiling it a second time, in its own
program, is the cost.

## Measured (throwaway spike, D=2 chi=8, CPU)

Scratch probe `inv/chainrule/probe.py` (not in the repo).

| program                         | dense compile | Z2 compile | warm      |
|---------------------------------|--------------:|-----------:|----------:|
| GMRES (`(I-J_env^T)` matvec)    | 480 s         | 16.9 s     | 118 ms    |
| chain rule (direct + indirect)  | 494 s         | 13.9 s     | 18 ms     |
| indirect term alone             | 410 s         | 13.8 s     | 19 ms     |
| direct term alone               | 15.8 s        | 0.49 s     | 0.9 ms    |
| GMRES, joint matvec             | 594 s         | 17.5 s     | 137 ms    |

* GMRES-path compile, GMRES + chain rule -> joint GMRES + direct:
  974 s -> 610 s (-37%); Z2 30.8 s -> 18.0 s (-42%).
* Gradient identical to the current path to 2e-16; same 11 matvecs, same
  residual 3.83e-11.
* Warm cost: the joint matvec is +16% per matvec (19.5 vs 18.5 ms standalone;
  137 vs 118 ms per solve), against one 18 ms chain-rule call saved.
  Break-even is about 11 matvecs per backward; the fused loop typically runs
  ~19 iterations, so warm cost rises by a few percent there.
* Estimated D=3 payoff (unmeasured): GMRES-only path 4,421 s -> ~2,335 s;
  fused + fallback 6,012 s -> ~4,310 s.

## Design

### Knob

`CTMConfig.adjoint_chain_rule: Literal["separate", "joint"] = "separate"`,
validated in `CTMConfig.__post_init__` (`ValueError` otherwise), threaded
through `ipeps_ad_policy` -> `ctm_energy_implicit(adjoint_chain_rule=...)` ->
the dispatch-cache key -> `_make_implicit_vjp_fn`.  It is part of the cache
key, so the two modes never share a compiled closure.  PESS callers keep the
default (they do not pass it).  `"separate"` runs exactly today's programs.

### `gmres_single_matvec(..., aux=False)`

With `aux=True` the matvec returns `(A v, a(v))` where `a` is linear in `v`
(here `a(v) = J_params^T v`).  The solver adds `best_aux` to its state,
initialised to zeros of `a`'s output shape (from `jax.eval_shape`), and
returns `(x, abs_resid, b_norm, best_aux)`.

The only places `best_x` changes are `_SEED` (to candidate `c_k`) and
`_RESID` (to the current `x`), and in both phases the matvec input `v_in` is
exactly the vector being adopted.  So `a(best_x)` is available from the same
call: `best_aux` takes the matvec's aux under the same strict-`<` rule as
`best_x`.  The Krylov basis needs no aux copy, and no extra matvec is spent.
`(best_x, best_aux)` is therefore always a consistent pair; when
`best_x` stays zero, `best_aux` stays zero, which is `a(0)`.

With `aux=False` the traced program is unchanged (pinned by a jaxpr
equality test).

### `_jit_apply_joint(params, env, v, *, chi, pb)`

One `jax.vjp` of the gauge-fixed sweep over `(params, env)`, returning
`((I - J_env^T) v, J_params^T v)`.  The gauge-fix reference is
`stop_gradient(env)`: this is what `_jit_apply_Jt` uses, and the chain rule's
reference is the constant `envs`, which has no params dependence, so both
cotangents match today's programs.

### Fused Neumann loop, `joint=True` (static)

* The energy VJP is taken over `(params, env)`: one call gives `dE_denv` (the
  RHS, as today) and `direct`.
* The loop body calls the joint VJP at `lambda_k`, giving `J_env^T lambda_k`
  (for the step) and `J_params^T lambda_k`, carried as `pc`.
* On exit the carry holds `lambda_K` and `pc = J_params^T lambda_{K-1}`.
  The gradient uses `lambda_{K-1}`: `diff` and `abs_resid` are measured on the
  step `lambda_K - lambda_{K-1} = b - (I - J^T) lambda_{K-1}`, so
  `lambda_{K-1}` is the iterate whose residual the loop certifies.  No extra
  iteration.  `lambda_K` is still returned and cached as the next warm start.
* Returns `(diverged, converged, n_iter, lam_final, abs_resid, b_norm,
  dE_denv, direct, indirect)`; `joint=False` keeps today's return tuple.
* The joint gradient differs from the separate one by
  `J_params^T (lambda_K - lambda_{K-1})`, i.e. O(`gmres_tol` * ||J_params||),
  inside the solve tolerance.

### `_bwd_core` with `adjoint_chain_rule="joint"`

* Fused, converged: `grads = g * (direct + indirect)` from the fused program.
* Fused, diverged/unconverged -> fallback: `_jit_gmres_one_matvec_solve` with
  `apply_Jt=_jit_apply_joint, aux=True`; `grads = g * (direct_from_fused +
  indirect)`.
* `adjoint_method="gmres"`: the same joint solve; `direct` from a new small
  `_jit_direct(params, env)` (energy VJP over params only).
* `_jit_chain_rule` is never called.
* Unchanged: `_report_adjoint` gating and warnings, `adjoint_residual` and the
  other `_F3_LAST_DIAGNOSTICS`, warm-start validation (#1113), the #1028
  flow -> frozen retry in `f_bwd` (each `pb` compiles its own joint programs,
  as the chain rule does today), and the #1111 slope check (reads only the
  final gradient).

### CPU fusion gating

`block_sparse_jit` gating applies only to block-sparse CPU calls.  Today the
chain rule uses `by_bond_dim=True` and the GMRES/J^T programs `False`.  The
joint programs mix both, so the plan measures both settings at symmetric D=3
on CPU and ships the faster one.  Until measured they use `by_bond_dim=True`
(unfused at virtual D>=3), the conservative side of the 8,569 s fusion-pass
wall (#1102).  Dense and GPU calls are unaffected.

### Error handling

* `gmres_maxiter=0` in the fused loop: the body never runs, `pc` stays zero,
  `converged=False` forces the fallback, so the zero is never consumed.
* Non-finite GMRES: `abs_resid` NaN fail-closed path unchanged; `best_aux`
  follows `best_x`.
* dtype: `aux` keeps the params dtype; the real-`b`/complex-data `TypeError`
  is unchanged and checks only `A v`.

## Testing

1. `gmres_single_matvec(aux=True)` on a linear toy `a(v) = P v`:
   `best_aux == P @ best_x` when a seed wins, on convergence, at the
   `maxiter` cap, with `maxiter=0`, and for an operator that returns NaN.
   `aux=False` jaxpr equals today's.
2. Gradient parity, joint vs separate, on a D=2 dense and a D=2 Z2 fixture:
   `adjoint_method="gmres"` to 1e-12 relative; fused converged to a bound
   scaled by `gmres_tol`; fused-with-forced-fallback (small `gmres_maxiter`)
   to 1e-12; `projector_backward="flow"` path.
3. Spy: under `"joint"` `_jit_chain_rule` is never called; under
   `"separate"` the joint programs are never called.
4. Config: invalid value raises; the two values give different cache keys.

## Measurement (gate for the default-flip PR)

* `benchmarks/` driver: per-program compile and warm time for both modes at
  D=2 chi=8 CPU, dense and Z2.
* CPU fusion-gating A/B for the joint programs at symmetric D=3.
* Peer session (4070 Ti): D=3 chi=16 GPU, both modes, step-1 compile and
  warm step.
* Flip the default only if D=3 compile drops and the warm backward rises by
  no more than ~20%.

## Out of scope

* Flipping the default (separate PR).
* PESS callers.
* Folding the direct term into `_jit_dE_denv` (the Arnoldi precheck consumes
  it).
* `_jit_gmres_solve` (legacy JAX-GMRES program).

## Revision 2026-10-19: fused path = env-only loop + joint GMRES polish

The fused design above (carry `J_params^T λ` in the Neumann loop, energy VJP
over `(params, env)` in the fused program) was measured and replaced.

* GPU D=3 chi=16 U(1), step 1: the joint fused program compiled 4,211 s
  against 2,244 s separate, cancelling the chain-rule saving (step-1 compile
  −3.2%, inside noise).  It also tripped an XLA GPU `HoistFusedBitcasts`
  CHECK-abort that needed a pass-disabling workaround.
* CPU D=3 chi=8 spike: the energy VJP over `(params, env)` inside the fused
  program cost ~+1,700 s of compile; the carry ~+46 s.  On GPU the carry
  alone (variant "jointC": joint loop + standalone direct term) still cost
  +1,355 s, so the CPU ratio did not transfer.
* Adopted ("polishA"): the fused program is the separate one, unchanged; a
  joint GMRES solve always follows, seeded with the loop's `λ` when the loop
  solved the system, else cold; `_jit_direct` gives the direct term.  GPU
  step 1: compile 7,795 → 5,977 s (−23%), wall 8,231 → 6,388 s, RSS
  51.1 → 39.0 GB, warm 4.7/3.8 → 4.6/3.9 s, gradient equal to separate to
  ≤4e-15 on cold and warm calls.  The pass workaround is no longer needed and
  was removed.
