# Linearise once, jit only the transpose: the implicit-AD adjoint

**Status:** approved in chat 2026-10-25, written for review · **Author:** Claude Code (for @yingjerkao)
**Branch:** `perf/linearize-once-adjoint` (from `main` @ `a0778e3`)
**Related:** #566 (symmetric AD compile wall), #1010 §2.1 (torch-eager spike, "cheaper lead"), #1123 (joint chain rule), #858/#1087 (GMRES seeds), #1111 (flow slope check)

## 1. Goal

Cut the compile time of the implicit-AD backward on the symmetric path. Today
each backward program compiles the forward CTM sweep with its residuals
inside itself. After this change, one program linearises the sweep and the
adjoint solvers compile only the transpose. The gradient stays the same to
round-off.

## 2. Why

The 2026-10-22 spike (#1010 §2.1) replayed tenax's backward programs at D=3
U(1). The J^T v that reuses saved residuals is 39,348 ops and compiled in
62 s (CPU, chi=9). The full re-linearising J^T v is 220,244 ops. On GPU at
chi=16 the joint GMRES program compiled in 2,856 s. These numbers come from
different hosts and chi; they show the size of the gap, not its exact ratio.

On 2026-10-24 the symmetric D=4 chi=16 run on this host (RTX 4070 Ti SUPER)
compiled `_jit_fused_fixed_point_bwd` in 913.3 s and
`_jit_gmres_joint_solve` in 1,284.5 s.

## 3. Current state (`src/tenax/algorithms/_ctm_energy_ad.py` @ `a0778e3`)

The default backward is `adjoint_method="fixed_point"` with
`adjoint_chain_rule="auto"` (which runs `"joint"` on the fused CTM). It calls
three programs on every backward:

| Program | What it compiles |
|---|---|
| `_jit_fused_fixed_point_bwd(params, env, init_lam, use_init)` | energy VJP w.r.t. env; `jax.vjp` of the gauge-fixed sweep w.r.t. env (once, outside the `while_loop`); the Neumann loop |
| `_jit_gmres_joint_solve(params, env, rhs, candidates, apply_joint=...)` | `gmres_single_matvec` whose matvec calls `jax.vjp(_gauge_fixed_sweep_joint, params, env)` **inside the loop body**: forward + backward of the sweep |
| `_jit_direct(params, env)` | energy VJP w.r.t. params |

So the forward sweep with residuals compiles in two programs, and the GMRES
program probably re-runs the forward on each iteration unless XLA hoists it.

## 4. Design

### 4.1 Data flow (default path)

```
vjp_joint     = _jit_linearize_joint(params, env, chi, pb)     # forward once; jax.vjp Partial
dE_denv       = _jit_dE_denv(params, env)                      # energy VJP w.r.t. env
lam, ...      = _jit_neumann_bwd(vjp_joint, dE_denv, init_lam, use_init)   # transpose only
lam, indirect = _jit_gmres_joint_solve(vjp_joint, rhs, candidates, ...)    # transpose only
direct        = _jit_direct(params, env)                       # unchanged
```

- `_jit_linearize_joint` runs the same gauge-fixed sweep as
  `_gauge_fixed_sweep_joint` today, over `(params, env)`, with the gauge
  reference `stop_gradient(env)`. It returns `jax.vjp`'s pullback, a
  `jax.tree_util.Partial` whose leaves are the residuals.
- `_jit_neumann_bwd` is today's Neumann loop (divergence guard, warm start,
  diagnostics, return tuple) with `J_env^T v = vjp_joint(v)[1]`. XLA
  dead-code elimination drops the unused params half.
- `_jit_gmres_joint_solve` keeps its matvec
  `v -> ((v - J_env^T v), J_params^T v)` and `gmres_single_matvec(...,
  aux=True)`, but takes `vjp_joint` in place of `(params, env, apply_joint)`.
- One backward builds `vjp_joint` once and passes the same object to both
  consumers.
- `adjoint_method="gmres"` with `"joint"` uses the new
  `_jit_gmres_joint_solve` too.

**Not changed:** `adjoint_chain_rule="separate"`, the eager fallback that only
`"separate"` reaches (`_jit_gmres_one_matvec_solve` + `_jit_chain_rule`), the
split CTM (`fuse_virtual_legs=False`), the forward CTM, `_jit_direct`.

### 4.2 Feasibility (measured, throwaway probe, JAX 0.10.2, CPU)

A jitted `lin(p, e)` returning `jax.vjp(f, p, e)[1]`, consumed by a jitted
`use(vjp, v)` with a `fori_loop` over `vjp`: over two calls with new values,
`lin` traced once and `use` traced once; the two Partials had equal treedefs;
the cotangent matched a direct `jax.vjp` to 0.0; the lowered `use` program had
0 forward (`sine`) ops.

### 4.3 Cache stability

The Partial's treedef carries the pullback function object. A cached
`_jit_linearize_joint` call returns the same object, so the consumers hit
their caches. When the linearise program recompiles (new chi, chi layout,
`pb`, dtype), the consumers recompile too; they do so today as well, being
keyed on the same shapes, so the compile count per layout does not rise.

### 4.4 Fusion dispatch

`block_sparse_jit` picks its no-fusion CPU variant from large
`SymmetricTensor` arguments. A Partial has only array leaves, so the consumers
would silently take the fused variant on CPU. The backward therefore computes
the choice once from `params_data_tuple` with today's rules (`by_bond_dim=True`
for the joint programs, `False` where the program is the sweep VJP only) and
calls each consumer's `.block_sparse` or `.dense` variant directly.

### 4.5 Memory

The residuals live from linearise until the GMRES solve ends; the backward
then drops its reference. The joint residuals also hold the params-path terms
that today's env-only Neumann program never keeps, so the peak may rise. The
benchmark measures it (gate A4).

### 4.6 Errors and fallbacks

Unchanged in behaviour: Neumann divergence; a non-convergent Neumann run
followed by GMRES from `rhs`; `_report_adjoint` warnings; `prev_lam`
invalidation (#1113); the #1111 flow slope check. They receive `vjp_joint` in
place of `(params, env)`. A `"separate"` or split-CTM call never builds
`vjp_joint`.

## 5. Testing

The oracle is `adjoint_chain_rule="separate"`: its code stays, and it agrees
with `"joint"` to round-off (#1123).

1. **Gradient parity.** The new default backward matches `"separate"` to
   rel 1e-10 on dense 2-site D=2 chi=8 and U(1) 2-site D=2 chi=8 (XXZ
   Delta=0.3), and fermionic U(1) 2-site D=2 (t-V), all from an SU state (SU
   input; the degenerate-SVD floor). The reported adjoint residual matches main's to
   1e-12.
2. **Second backward, new values.** On the same layout the gradient still
   matches `"separate"` (catches a stale `vjp_joint`).
3. **Consumers hold no forward.** The traced jaxprs of `_jit_neumann_bwd` and
   `_jit_gmres_joint_solve` contain 0 `svd` primitives. The forward projector
   SVDs live only in the linearise program. On main the GMRES program fails
   this, so the test is RED first.
4. **Compile once per layout.** 0 new jaxpr traces of the linearise, Neumann
   and GMRES programs over backwards 2 and 3 (counted with
   `/jax/core/compile/jaxpr_trace_duration` events; `_cache_size()` reads 0
   under pytest-cov).
5. **Fusion dispatch.** For a large U(1) state on CPU each consumer takes the
   same `.block_sparse` / `.dense` variant as main.
6. **Existing suites** pass unchanged: implicit-AD FD parity, #1111 slope
   check, #858/#1087 seeds. Core runs serially and load-gated.

**Mutants** (commit first, unique anchors, `PYTHONDONTWRITEBYTECODE=1`,
delete `__pycache__`):

| Mutant | Change | Killed by |
|---|---|---|
| M1 | GMRES matvec re-linearises inside the loop | test 3 |
| M2 | Neumann takes the params cotangent half | test 1 |
| M3 | `vjp_joint` cached across backwards | test 2 |
| M4 | fusion chosen from the `vjp_joint` args only | test 5 |
| M5 | gauge reference not under `stop_gradient` | test 1 |

## 6. Benchmark and acceptance

GPU 1 only (GPU 0 runs S5). Fixture: U(1) 2-site, XXZ Delta=0.3, D=3 chi=16,
from an SU state, same seed on base (`main`) and branch. Record per-program
compile time on step 1 (`JAX_LOG_COMPILES`), warm backward time on steps 2-3,
device peak, host RSS, and the gradient. Optional: D=4 chi=16, 1 step, 4 h
cap; baseline 913.3 s + 1,284.5 s (2026-10-24, this host).

| Gate | Criterion |
|---|---|
| A1 | total backward compile at D=3 <= 0.7x main; > 0.9x is a no-go (stop and report before any PR) |
| A2 | warm backward time <= 1.05x main |
| A3 | step-1 gradient equals main's to rel 1e-8 |
| A4 | device peak <= 1.2x main; above is a finding to report, not to tune away |

The 0.7x bar, not the spike's 5.6x op ratio: the linearise program still
compiles the forward with residuals, about one sweep's size. The expectation
is about 2x; it is not measured.

## 7. Rollout

Replace in place, no opt-in flag: the gradient is the same to round-off, and
legacy AD paths are a fallback, not a constraint. Update
`docs/guide/algorithms/ipeps_ad_paths.md` for the new program list. One
branch, one concern. PR only on the user's approval; never armed unasked.

## 8. Out of scope

- `"separate"` and its fallback programs; `_jit_chain_rule`.
- A joint energy VJP that merges `_jit_dE_denv` and `_jit_direct`.
- The forward CTM, the split CTM, the C4v path.
- The dense masked-D=4 CTM non-convergence (separate investigation).

## 9. Results (2026-10-25): A1 NO-GO

GPU 1 (RTX 4070 Ti SUPER), U(1) 2-site XXZ Delta=0.3, D=3 chi=16, SU state,
`benchmarks/bench_linearize_once.py --steps 4`, compile cache off. Base
`b71f5a5` (main), head `a97d5f8`.

Step-1 backward compile (s):

| Program | Base | Head |
|---|---|---|
| `_jit_fused_fixed_point_bwd` | 790.2 | — |
| `_jit_linearize_joint` | — | 164.5 |
| `_jit_dE_denv` | — | 106.3 |
| `_jit_neumann_bwd` | — | 578.2 |
| `_jit_gmres_joint_solve` | 1116.4 | 960.5 |
| `_jit_direct` | 496.9 | 462.4 |
| **Total** | **2403.5** | **2271.9** |

- **A1 = 0.945** (0.949 without `_jit_direct`). The bar is 0.7, and above 0.9
  is a no-go.
- Step 2 recompiles the backward on both arms (a new chi layout): 5127.2 s
  (base) and 4497.0 s (head), ratio 0.877.
- E and |g| are equal to 10 digits at steps 1-4 on both arms.
- The head arm was killed during its post-optimize SU gradient, by the
  controller's host-memory guard (MemAvailable 11 GB; a CPU job ran beside it).
  So A2, A4 and the full-precision A3 were not graded. Base: optimize wall
  16,102.9 s, max RSS 108.6 GB, device peak 1.56 GB.

Why: op counts per step-1 program (CPU, D=2 chi=8; lowered StableHLO ops /
optimised HLO instructions):

| Program | Base | Head |
|---|---|---|
| `_jit_fused_fixed_point_bwd` | 160,289 / 435,955 | — |
| `_jit_linearize_joint` | — | 62,571 / 128,771 |
| `_jit_dE_denv` | — | 23,349 / 73,723 |
| `_jit_neumann_bwd` | — | 79,451 / 264,005 |
| `_jit_gmres_joint_solve` | 143,315 / 394,498 | 79,696 / 306,806 |
| **Solver total** | **303,604 / 830,453** | **245,067 / 773,305** |

- The design did what it said: the head solvers hold 0 SVDs after
  optimisation (base: 104 each).
- But the transpose J^T v alone (~79.5k lowered ops) is larger than the
  forward linearisation (62.6k), and XLA grows it 3.3-3.9x (the forward
  2.1x). The head still compiles J^T v twice, once in Neumann and once in
  GMRES. So the optimised total falls only to 0.931 of base, which matches
  the compile-time ratio.
- The params half of the transpose is small: the env-only Neumann program and
  the joint GMRES program differ by ~250 lowered ops.

Next: compile J^T v once — spec
`2026-10-25-single-transpose-adjoint-design.md`.
