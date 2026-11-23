# Charged C4v iPEPS AD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `optimize_gs_ad` with `unit_cell="1x1"` and `gs_c4v=True` optimises a charged U(1)/Z_n C4v `SymmetricTensor` through implicit AD of the #1211 2-sublattice sweep at a frozen layout.

**Architecture:** The #1211 module `_ctm_tensor_c4v_staggered.py` splits its projector into an eager ranking (`_c4v_staggered_layout`) and a traced builder at a frozen `BondLayout` (`_c4v_staggered_projectors_at`). Its eager outputs stay bit-identical. A new engine `_ctm_tensor_c4v_charged_ad.py` holds the aligned sweep F̃, the two forward phases and the `charged_c4v_converge` custom_vjp. Its backward solves (I − Jᵀ + τI)λ = g with `gmres_eager`. A new loop `ipeps_optimize_c4v_charged.py` runs L-BFGS with the Hager-Zhang or Armijo line search on the block data of A. `optimize_gs_ad` routes charged input to it through `use_charged_c4v_path` (`ipeps_ad_policy.py`).

**Tech Stack:** Python 3.11, JAX 0.10.2 (CPU, x64), tenax `SymmetricTensor` / `BondLayout` / `contract`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md` (commit `08b22a32`, branch `feat/c4v-charged-ad`). The spec is the binding authority; this plan argues from it. Where the plan departs from the spec, a ruling below says why.

## Global Constraints

- No new public symbol (spec section 6: "There is no new public symbol"). `src/tenax/__init__.py` does not change. `README.md` gets one feature line that links to the guide. `CHANGELOG.md` does not change (it is written at release time).
- `ctm_tensor_c4v` stays eager-only, so `jax.grad` through it still raises. The #1211 eager outputs (`_c4v_staggered_projectors`, `_c4v_staggered_sweep`, `_ctm_tensor_c4v_staggered`) stay bit-identical.
- Unchanged: the dense C4v paths, `ctm_ad_mode="c4v_reference"` (it still refuses a `SymmetricTensor`), the root-implicit modes (they still refuse `gs_c4v`), `envs_init` (still refused with `gs_c4v`), and self-dual symmetric input (still densified on the default path).
- All charge arithmetic goes through `sym.dual` and `_dual_charge`. No code writes `-q`.
- No `todense()` on the block-sparse env path. Allowed: the site tensor A (D⁴·d entries) for the C4v projection, and the chi × chi `C.todense()` of the #1211 sv test.
- Real, bosonic data only. Complex input is refused; fermionic input is not charged C4v.
- Tests:
  - `tests/test_c4v_charged_ad.py` (new, `core`): mechanism tests, at chi ≤ 6.
  - `tests/test_c4v_charged_ad_physics.py` (new, `algorithm`): converged FD with an h scan, the D=2 optimum, dense parity.
  - Additions to `tests/test_c4v_staggered.py` (`core`) and `tests/_c4v_staggered_fixtures.py`.
  - Register both new files in `tests/conftest.py` `_FILE_MARKERS`.
- RED first. Commit before mutating. Mutate in place with `PYTHONDONTWRITEBYTECODE=1`, unique anchors, and `git checkout --` in a `finally` block. Run no other tenax process while a mutant is in place.
- Run the suites one at a time. Before a heavy run, read `uptime`; start it only if the load is below 30.
- Scratch directory: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan`. Shell state does not persist, so each command below sets `S` itself.
- Test command: `cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider ...`. Light probes use `taskset -c 12-23`.
- Redirect every long run to a file under `$S` and read the file. Never pipe a long run to `tail`. The foreground limit is 600 s; run longer jobs in the background.
- Use `command grep` (`grep` here is a ugrep wrapper). `grep -c` exits 1 on a zero count.
- Commits: stage files by name (never `out/` or `out2/`). pre-commit (ruff-check `--fix`, ruff-format) must pass. Never use `--no-verify`, `--amend` or `--reuse-message=HEAD`. If a hook rewrites a file, `git add` the same files and run the same `git commit -F` again. Every message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`
- Push, PR, labels and GitHub comments only on the user's word. The PR body starts with `> 🤖 **AI-generated comment** — written by Claude Code, posted by @yingjerkao.` and ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`, a blank line, and `https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`. Never arm `--auto`. Never pass `--delete-branch`.
- Read-only: the Hasik state files under `/home/yjkao/tenax-sma/hasik_states_u1/` (never write in `~/tenax-sma`). Never run `fu3_archive/plan/spin1/probe_spin1.py` or `probe_spin1_flip.py`.

## Review Focus

These are the input classes that the spec implies but does not test by name, most likely first. Each line names the task whose test pins it.

1. **A charged `A_init` that is not C4v** (raw random blocks in the C4v charge layout). `_is_charged_c4v` reads metadata only, so the branch takes it. The loop must project it to C4v at the start; the returned A is C4v (|symmetrize_c4v(todense(A)) − todense(A)| < 1e-12). Pinned by Task 6, `test_a_non_c4v_start_is_projected`.
2. **`gs_projector_method` against `ctm.projector_method`.** The override comes first (spec, config table). `ctm.projector_method="qr"` with `gs_projector_method="svd"` must run; `ctm.projector_method="svd"` with `gs_projector_method="qr"` must be refused. Pinned by Task 6, `test_gs_projector_method_overrides_before_the_check`.
3. **An unconverged forward at a line-search trial.** φ must return +inf and the step must go on, not raise. Pinned by Task 6, `test_an_unconverged_trial_is_a_bad_point`.
4. **An `A_init` with a missing block** (its stored blocks omit a valid key). Without canonicalisation the C4v projection matrix is not square. The loop rebuilds A through `from_dense`, so the parameter has every valid block. Pinned by Task 6, `test_a_missing_block_is_canonicalised`.
5. **Two runs at two chi in one process.** The jit caches are keyed by layout, so a stale program must not serve the second run: each `gs_num_steps=0` energy must equal the eager `ctm_tensor_c4v` energy at its chi. Pinned by Task 6, `test_two_chi_in_one_process`.

## Rulings made while planning

Each ruling records a decision, why, and its cost if wrong.

1. **The loop runs a real line search.** The spec (section 5, step 3) says "L-BFGS with the configured line search (`_build_optimizer`, with `value_fn` as in the reference loop)". But `_build_optimizer`'s L-BFGS chain (`scale_by_lbfgs`, `clip_by_global_norm`, `scale(-1)`) runs no line search, and optax ignores `value_fn` (#792; see the warning at `ipeps_optimize_root_implicit.py:383-403`). The charged loop uses the machinery of the 1x1 C4v path (`ipeps_optimize.py:3296-3470`): `lbfgs_two_loop` with an identity H0, memory 10, a curvature pair only when sᵀy > 1e-10; then `hager_zhang_line_search`, or `_backtracking_line_search` when `_resolve_line_search_method` gives `"armijo"`; then `_is_real_decrease` and the stall reset. *If wrong:* a loop that differs from the reference loop's unsearched steps. The D=2 optimum test pins the outcome.
2. **The one-sweep VJP test pins "P frozen" only.** It has no solve, so it cannot catch "λ = g". The converged FD test catches both. *If wrong:* none.
3. **The D=2 optimum test pins "the loss skips the C4v projection", not "a wrong B-site".** The energy is the #1211 charged branch, and the spec (section 4) changes no code there. A wrong B-site either raises (the env guard checks the D² pairing) or is a #1211 defect. The test asserts that the returned A is C4v; a loss that drops the projection moves A off C4v. *If wrong:* a B-site defect that the #1211 tests already miss.
4. **No fixture ties the largest |entry|, so the kink test pins the same mutation deterministically.** The spec: "If it is tied on neither [D=2 fixture], the test uses a D=3 charged fixture." Probe 2 (2026-11-23, 15 fixture-chi pairs: D=2 and D=3 U(1), Z3, Z4, chi 6-12) found the top-2 relative gap of |T*| at 9.5e-3 to 0.70 and of |C*| at 0.16 to 0.98; D=2 at chi 6-24 has no tie in T or C. So no D=3 fixture has the tie either. The mutation (the plain `max_abs()` in F_L) is pinned by a radial-cotangent test: a cotangent (C′, 0) or (0, T′) along the output has a zero VJP with `_normalize_by_max_abs` (measured 8.6e-16 relative) and a VJP of relative size 2.8 with the plain `max_abs()`. *If wrong:* a kink at a real tie goes untested here; `tests/test_ctm_normalize_maxabs_kink.py` still pins `_normalize_by_max_abs` at a constructed tie.
5. **The one-sweep FD cotangent has no radial part.** `_max_abs_scale` gives max|·| as its value and the gradient of c‖·‖_F. So the VJP of F̃ differs from the derivative of F̃'s value by a rank-1 term along the output. The test uses w with w_C ⊥ C′(e₀) and w_T ⊥ T′(e₀). The converged gradient does not change: E is scale invariant in C and in T, so g ⊥ e* per tensor; F(αC, βT) = F(C, T), so J e* = 0 per tensor; hence λ ⊥ e* per tensor. *If wrong:* none; the converged FD test checks the full gradient.
6. **The record counts adjoint matvecs, not GMRES restarts.** `gmres_eager` returns `(x, resid)` only. The history key is `adjoint_matvecs`. One restart costs 31 matvecs (30 Krylov steps and one residual). *If wrong:* a reader divides by 31.
7. **The matvec re-linearises per call.** One jitted program per layout runs one F̃ forward and one VJP per call. The spec leaves the choice to measured times. Task 4 measures the compile and run time of the three programs at D=2, chi=16 and writes them to the ledger. A `jax.tree_util.Partial` of the linearised residuals is tried only if the matvec dominates the D=3 benchmark. *If wrong:* a slower solve at D=3; the benchmark reports it.
8. **Extra refusals, beyond the spec table:** `gs_optimizer != "lbfgs"`, `gs_line_search=False`, `ctm.ctm_mixing > 0`, `ctm.device_mesh`, `gs_chi_schedule_steps`, `gs_ctm_conv_tol_schedule`, `gs_ctm_max_iter_schedule` and `gs_plateau_patience_schedule`. This loop would ignore each of them silently. *If wrong:* a user who sets one gets an error instead of a run.
9. **The parameter is the flat block buffer of A.** The spec says "the block pytree of A"; a `SymmetricTensor` has one pytree leaf, its flat buffer, so the two are the same. The C4v projection is a dense matrix P, built once by `jax.jacfwd` of x ↦ buffer(from_dense(symmetrize_c4v(todense(with_data(A_c, x))))). P is symmetric and idempotent (spec, Measurements), n × n with n the number of block entries (5 at D=2). *If wrong:* none at D ≤ 4.
10. **The D=2 assert is tighter than the spec's.** At chi=16 the golden-section minimum over the one free parameter is E* = −0.660231093480 at t* = 0.4210174, 3.986e-14 above E_D2 (3.963e-14 at chi 24 and 32). The test asserts |E − E_D2| ≤ 1e-8, not 1e-6. At |g| < 1e-5 on a 1-D quadratic with E'' ≈ 4.3, |E − E*| ≈ 1e-11. *If wrong:* a flaky 1e-8 assert; the executor reports the gap.
11. **FD at chi=6.** exact_u1_d2 at t = 1.15·t0 keeps 6 states, charges [−1, −1, 0, 0, 1, 1], cut gap 0.75, 13 sweeps. The #762 fixture keeps 6, [−2, 0, 0, 0, 0, 2], cut gap 0.227, 67 sweeps. The Z4 fixture keeps 6, cut gap 0.129, 17 sweeps. If the flow-against-frozen regime assert fails at chi=6, the executor tries chi=4 (cut gaps 0.80 and 0.535) and ledgers the ruling.
12. **The 2-site refusal needs `gs_c4v=True`.** The spec refuses "`unit_cell="2site"` with a charged `A_init`" in the C4v dispatch. With `gs_c4v=False` the general symmetric 2-site path runs today, and this spec does not change it. *If wrong:* a `gs_c4v=False` 2-site charged run proceeds as before.
13. **An adjoint failure at a line-search trial is a bad point.** `hager_zhang_line_search` treats any probe exception except `LineSearchAborted` as a bad point (`_line_search.py:103-118`). So a `RuntimeError` from the solve at a dφ trial rejects that α and logs a warning. At the step's own evaluation it raises, with no recovery (spec). *If wrong:* a trial-point solve failure is not fatal; the next evaluation at an accepted point still is.
14. **The layout check runs once per evaluation of the main loop,** right after `value_and_grad`, on the enlarged corner of that point's e*. Step 0 checks the start; step k checks the point accepted at step k − 1. A mismatch is recorded in `layout_mismatch_steps` and logged once. *If wrong:* one extra eager projector per step.
15. **History keys:** `energies`, `grad_norms`, `step_times`, `num_steps`, `converged`, `ctm_converged`, `forward_sweeps`, `adjoint_matvecs`, `adjoint_residual`, `layout`, `layout_mismatch_steps`. The 1x1 keys `jit_compile_time`, `ctm_sv_diff`, `ctm_step_multiplier`, `ctm_stationarity`, `final_env_source` and `probes` do not apply. *If wrong:* a missing key that a caller expects.
16. **`_resolve_projector_backward` does not run for the charged branch.** It validates the fused implicit path, which rejects `projector_method="eigh"`; the charged branch accepts it and checks its own settings. *If wrong:* none; the dispatch comment says why.
17. **`warm` gets zeros of (C*, T*) as its cotangent,** and the forward asserts that `warm` has the treedef of (C*, T*). *If wrong:* none.
18. **FD settings.** The converged FD tests run τ = 0 (`adjoint_tikhonov=0.0`), `adjoint_tol=1e-12`, `adjoint_maxiter=100`, and a forward re-converged element-wise to 1e-13 (`CTM_FD`, Task 5). Probe 7 put the element-wise floor at 4.2e-16 to 8.9e-16. The default τ = 1e-6 biases the gradient by about τ/(1 − ρ) relative, with ρ the contraction rate of the sweep. Probe 5 measured ρ up to 0.895, so the bias reaches 1e-5, above the FD resolution. *If wrong:* an FD test that measures the Tikhonov bias, not the gradient.
19. **The loop reaches the engine through its module object** (`_engine.charged_c4v_converge`, `_engine._elementwise_phase`), and it reaches trials through the module function `_trial_energy`. So `monkeypatch.setattr` on `ad` or on `opt` reaches the loop. Each test that mocks asserts a call count or an effect of the mock. *If wrong:* a mock that does not apply, and a test that passes vacuously.
20. **The step's evaluation is not reused from a dφ call.** The 1x1 loop carries the gradient of an accepted HZ probe into the next step (`_AcceptedProbeEval`, `ipeps_optimize.py:582`). The charged loop does not port it, so each HZ step costs one extra gradient. *If wrong:* one more gradient per step at D=3; Task 9 reports the step times.
21. **A D=2 random start that ends on another fixed point is a finding, not a tolerance.** Task 8 marks that case `xfail(strict=True)` with the measured energy, ledgers it and reports it to the user. The assert is not loosened. *If wrong:* an optimiser defect behind an xfail; the ledger line and the final message carry the number.
22. **`_chi_components` returns numpy int64 index arrays.** `_bond_component_phases` indexes with them; tuples of Python ints raised `IndexError` in probe 5. *If wrong:* none; the align tests run it.
23. **The reset test patches `ad.charged_c4v_converge` and runs Armijo.** Armijo trials call `_elementwise_phase`, not `charged_c4v_converge`, so call 2 is the evaluation of step 2. Under HZ, dφ calls it too, and the count would depend on the search. *If wrong:* none; the test checks the reset by its energies.
24. **The loop calls `_normalize_stall_recovery(config, unit_cell="1x1")` before its checks.** The default `gs_stall_recovery=None` becomes `"reset"`; an explicit value stays, so `"noise"` still reaches the refusal. *If wrong:* the reset rule runs under another name; the reset test pins its behaviour.
25. **`CTMNotConvergedError` is a subclass of `RuntimeError`.** So dφ catches it first and raises `LineSearchAborted`: HZ ends the search at its best φ point, as on the 1x1 path. Any other `RuntimeError` (a failed adjoint solve) is logged and re-raised, and HZ makes that α a bad point (ruling 13). The main loop catches only `CTMNotConvergedError`, so a failed solve at an evaluated point raises (spec). *If wrong:* an unconverged dφ trial ends the search early: a shorter step, not a wrong one.
26. **The start projection is not mutated.** `site(x)` applies P at every evaluation, so a start that skips P changes only |x| and the scale of the first step, never an evaluated state. Review Focus 1's test pins "`site` projects" instead: two steps from a raw start, one of them accepted. *If wrong:* a start-projection defect that changes only the step scale.
27. **The checkpoint refusal keeps its message and appends one sentence** ("The charged C4v path (spec 2) is not wired either."). So an existing `match=` on the old text still holds. *If wrong:* none; Task 7 Step 5 runs the existing tests that name `gs_checkpoint_path`.

## Settled points (the spec's "Points the plan settles")

| Point | How it is settled |
|---|---|
| Test markers | `test_c4v_charged_ad.py` is `core` if it runs in under 120 s on cores 0-11; else its slowest tests move to the `algorithm` file (ledger ruling). The physics file is `algorithm`. |
| FD chi and fixtures | chi=6 for both D=2 fixtures and the Z4 smoke (ruling 11). The regime asserts run inside each test. |
| The matvec program | Re-linearise per call (ruling 7); Task 4 measures it. |
| D=2 tolerance | Measured 3.986e-14 at chi=16; the test asserts 1e-8 (ruling 10). |
| Max-abs kink fixture | No tie on any fixture (ruling 4); the radial-cotangent test pins the mutation. |
| Dense parity chi | chi=16: |E_charged − E_dense ctm_tensor| = 3.3e-16 at t0 and 1.1e-16 at t*; the dense C4v corner agrees with the charged one to 1.2e-9 (sv gap), cut gap 9.0e-4. |
| GMRES restart | Krylov dimension 30 (`_KRYLOV_DIM`). If a solve stalls, raise the dimension, not the tolerance. |
| History keys | Ruling 15. |
| D=3 hardware | CPU first, cores 0-11, load-gated. If the first-gradient compile takes more than 2 h, stop and ask the user before any A100 run. |

Probe numbers (2026-11-23, CPU, float64, scripts in `$S`):

- **The chi=16 layout does not move along the D=2 path.** At t = 0.4210, 0.42, 0.44, 0.46, 0.4828, 0.52, 0.60 and 1.5491, and for `c4v_fixture(U1Symmetry(), [0, 1], [0, 1])`, the layout is {−2: 1, −1: 4, 0: 6, 1: 4, 2: 1}. At chi=6 it is {−1: 2, 0: 2, 1: 2} on the same points.
- **The random D=2 fixture is in the one-parameter family,** at t = 1.5491 (A[1000,1]/A[0000,0]; the C4v projection has rank 2). At chi=16 its cold start lands on another fixed point than `charged_d2(1.5491)`: E = −0.567526272345 against −0.567573615736 (cut gap 0.215 against 0.041). The D=2 optimum test therefore starts from two different fixed points.
- **jit:** the enlarged corner and the T update with a fixed P compile in 0.35 s and run in 0.12 ms at chi=16 (eager full sweep 37.8 ms); |dC| = 4.1e-18, |dT| = 1.1e-16 against eager.
- **Traced blocks:** `SymmetricTensor(blocks, indices)` with traced blocks, `truncated_svd_ad` and `regularized_eigh` build and differentiate under `jax.jit` and `jax.grad`.
- **jit cache:** `f._cache_size()` counts compiled programs per static argument; `jax.clear_caches()` resets it to 0.
- **Cold start and element-wise phase (probes 4, 5, 7):** at sv tol 1e-8 the kept charges are fixed after 3 sweeps on every FD fixture except z4 at chi 6. After the cold start, the aligned sweep reaches 1e-12 element-wise in 3 to 58 sweeps, at rates 0.048 to 0.895 (7 cases). The element-wise floor is 4.2e-16 to 8.9e-16 (3 fixtures).
- **Differentiable energy (probe 6):** at chi=6, the energy through 20 sweeps at a fixed layout has a finite gradient on the three FD fixtures (|∂E/∂A| = 0.186 to 0.510). The C4v projection P has rank 2, 3 and 3, with |P − Pᵀ| = 0 and |P² − P| = 0.
- **custom_vjp (probe 8, `$S/probe_cvjp.py`):** on jax 0.10.2, `jax.custom_vjp` with `nondiff_argnums` for the layout, the `CTMConfig` (not hashable) and the record dict works. Under `jax.grad` the fwd and bwd rules run eagerly (`float()` works in them). A primal-only call runs the primal body. A jitted step at one layout has `_cache_size() == 1`.
- **Config (probe 9, `$S/probe_cfg.py`):** every refused setting of Task 6 constructs, except that `ctmrg_heuristic_increase_chi=True` needs `chi_max` (the test passes 8). The defaults are `gs_stall_recovery=None`, `projector_method="svd"` and `ctm_ad_mode=None`.

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/tenax/algorithms/_ctm_tensor_c4v_staggered.py` | modified (Tasks 1, 2) | Split projector: `_c4v_staggered_blocks`, `_c4v_staggered_rank`, `_c4v_staggered_assemble`, `_c4v_staggered_layout`, `_c4v_staggered_projectors_at`; `_c4v_staggered_project`; `_c4v_staggered_loop` |
| `src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py` | new (Tasks 3, 4) | Alignment, F_L and F̃, the jitted step, cold start, element-wise phase, `charged_c4v_converge` and its backward |
| `src/tenax/algorithms/ipeps_optimize_c4v_charged.py` | new (Task 6) | Config refusals, the C4v projection matrix, the optimiser loop |
| `src/tenax/algorithms/ipeps_ad_policy.py` | modified (Task 7) | `use_charged_c4v_path` |
| `src/tenax/algorithms/ipeps_optimize.py` | modified (Task 7) | Dispatch, the 2-site refusal, the checkpoint refusal, the docstring |
| `tests/_c4v_staggered_fixtures.py` | modified (Tasks 3, 5) | `c4v_indices`, `charged_d2`, `T_EXACT` (Task 3); `with_data`, `c4v_direction`, `FD_STEPS`, `fd_plateau`, `frozen_projectors` (Task 5) |
| `tests/test_c4v_staggered.py` | modified (Tasks 1, 2) | Tests of the split projector and the loop flag |
| `tests/test_c4v_charged_ad.py` | new (Tasks 3, 4, 5, 6, 7) | Core mechanism tests |
| `tests/test_c4v_charged_ad_physics.py` | new (Tasks 5, 8) | Converged FD, D=2 optimum, dense parity |
| `tests/conftest.py` | modified (Task 3) | Registers the two new test files |
| `benchmarks/ipeps_c4v_charged_ad.py` | new (Task 9) | D=3 benchmark against Hasik's U(1)-C4v state |
| `docs/guide/algorithms/ipeps_ad_paths.md` | modified (Task 10) | Path 4b: the charged C4v path |
| `README.md` | modified (Task 10) | One feature line |

---

### Task 1: Split the eager projector; flag the loop's convergence

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_c4v_staggered.py` (imports and `__all__` at lines 25-61; `_c4v_staggered_projectors` at 68-201; `_c4v_staggered_sweep` at 315-341; `_ctm_tensor_c4v_staggered` at 374-398)
- Test: `tests/test_c4v_staggered.py`

**Interfaces:**
- Consumes: `BondLayout.from_dims(Mapping[int, int])` from `tenax.algorithms._ctm_root_implicit_sym_sectors`; `CTMConvergeInfo(converged, iterations, sv_diff, ..., best_iteration, step_multiplier)` from `tenax.algorithms._ctm_convergence_policy`.
- Produces, in `tenax.algorithms._ctm_tensor_c4v_staggered`:
  - `_c4v_staggered_blocks(Qf) -> (sym, fused, col, mats)`; `mats: dict[int, jax.Array]`, block (q, q) with the fused states of charge q as rows. Raises `RuntimeError` on unequal faces.
  - `_c4v_staggered_rank(sym, mats, chi) -> (decomp, keep)`; `decomp[q] = ("self", W, None) | ("pair", U, V)`, `keep: dict[int, list[int]]`. Eager. Raises the two `ValueError`s of today.
  - `_c4v_staggered_assemble(sym, fused, col, cols) -> (P_left, P_right)`; `cols: dict[int, jax.Array]`, one column block per kept charge.
  - `_c4v_staggered_layout(Qf, chi) -> BondLayout`; eager.
  - `_c4v_staggered_project(Qf, T, a, P_left, P_right) -> (C_new, T_new)`; unnormalised, traceable.
  - `_c4v_staggered_loop(A, chi, max_iter, conv_tol) -> (C, T, a, CTMConvergeInfo)`.
  - `_c4v_staggered_projectors`, `_c4v_staggered_sweep` and `_ctm_tensor_c4v_staggered` keep their signatures and their outputs, bit for bit.

- [ ] **Step 1: Snapshot the #1211 outputs before any edit**

Create `$S/snap_1211.py`:

```python
"""Snapshot the #1211 eager env (save), or compare with the snapshot (check)."""

import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, "/home/yjkao/tenax/tests")
from _c4v_staggered_fixtures import c4v_fixture

from tenax.algorithms._ctm_tensor_c4v_staggered import _ctm_tensor_c4v_staggered
from tenax.core.symmetry import U1Symmetry, ZnSymmetry

S = "/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan"
FIELDS = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
cases = {
    "762": (c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0]), 8),
    "test": (c4v_fixture(U1Symmetry(), [0, 1], [0, 1]), 7),
    "z4": (c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1]), 7),
}
out = {}
for name, (A, chi) in cases.items():
    env = _ctm_tensor_c4v_staggered(A, chi, 40, 1e-10)
    for field in FIELDS:
        X = getattr(env, field)
        out[f"{name}_{field}"] = np.asarray(jax.tree_util.tree_leaves(X)[0])
        meta = [(i.label, int(i.flow), np.asarray(i.charges).tolist()) for i in X.indices]
        out[f"{name}_{field}_meta"] = np.asarray(repr(meta))
if sys.argv[1] == "save":
    np.savez(f"{S}/snap_1211.npz", **out)
    print("saved", len(out))
else:
    ref = np.load(f"{S}/snap_1211.npz")
    bad = [k for k in out if not np.array_equal(out[k], ref[k])]
    print("identical" if not bad else f"DIFFER {bad}")
```

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u $S/snap_1211.py save > $S/t1-snap.log 2>&1; cat $S/t1-snap.log`
Expected: `saved 48`

- [ ] **Step 2: Write the failing tests**

In `tests/test_c4v_staggered.py`, add `from collections import Counter` to the standard-library imports, add `from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout` to the tenax imports, and add `_c4v_staggered_layout` and `_c4v_staggered_loop` to the `from tenax.algorithms._ctm_tensor_c4v_staggered import (...)` block (alphabetical order). Append at the end of the file:

```python
# ------------------------------------------------------------------ #
# Spec 2: the split projector and the loop flag                       #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("chi", [5, 4, 3, 2])
def test_layout_counts_the_projector_states(chi):
    Qf = _synthetic_q(U1Symmetry(), **_U1_Q)
    P_left, _ = _c4v_staggered_projectors(Qf, chi)
    charges = np.asarray(leg(P_left, "chi_new").charges).tolist()
    assert _c4v_staggered_layout(Qf, chi) == BondLayout.from_dims(Counter(charges))


def test_layout_counts_a_self_partner_block():
    # Z4: the pair (1, 3) has value 4; the self-partner charge 2 has value -5.
    blocks = {1: [[4.0]], 3: [[4.0]], 2: [[-5.0]]}
    Qf = _synthetic_q(ZnSymmetry(4), [1, 2, 3], blocks)
    assert _c4v_staggered_layout(Qf, 3) == BondLayout.from_dims({1: 1, 2: 1, 3: 1})
    assert _c4v_staggered_layout(Qf, 1) == BondLayout.from_dims({2: 1})


def test_layout_refuses_a_chi_that_splits_the_top_pair():
    with pytest.raises(ValueError, match="chi=1"):
        _c4v_staggered_layout(_synthetic_q(U1Symmetry(), **_U1_Q), 1)


def test_loop_reports_convergence():
    A = _site("762")
    _C, _T, _a, info = _c4v_staggered_loop(A, 4, 2, 1e-300)
    assert (info.converged, info.iterations) == (False, 2)
    C, T, _a, info = _c4v_staggered_loop(A, 4, 200, 1e-8)
    assert info.converged and 2 <= info.iterations < 200
    assert info.sv_diff < 1e-8
    full = _c4v_staggered_to_full_env(C, T)
    env = _ctm_tensor_c4v_staggered(A, 4, 200, 1e-8)
    for field in ("C1", "T1", "T2"):
        np.testing.assert_array_equal(
            np.asarray(jax.tree_util.tree_leaves(getattr(full, field))[0]),
            np.asarray(jax.tree_util.tree_leaves(getattr(env, field))[0]),
        )
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_staggered.py -k "layout_counts or layout_refuses or loop_reports" -q > $S/t1-red.log 2>&1; tail -n 5 $S/t1-red.log`
Expected: `ERROR tests/test_c4v_staggered.py` with `ImportError: cannot import name '_c4v_staggered_layout'`.

- [ ] **Step 4: Split the projector, the sweep and the loop**

In `src/tenax/algorithms/_ctm_tensor_c4v_staggered.py`:

(a) Append to the module docstring, before its closing `"""`:

```python
Spec 2 (``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``)
differentiates the sweep at a frozen layout.  So the projector is split
into the eager ranking (``_c4v_staggered_rank``, ``_c4v_staggered_layout``)
and the assembly of P from its columns (``_c4v_staggered_assemble``), which
the traced builder ``_c4v_staggered_projectors_at`` shares.  The eager
outputs do not change.
```

(b) Replace `__all__` with:

```python
__all__ = [
    "_c4v_staggered_assemble",
    "_c4v_staggered_blocks",
    "_c4v_staggered_enlarged_corner",
    "_c4v_staggered_initial_state",
    "_c4v_staggered_layout",
    "_c4v_staggered_loop",
    "_c4v_staggered_project",
    "_c4v_staggered_projectors",
    "_c4v_staggered_rank",
    "_c4v_staggered_sweep",
    "_c4v_staggered_to_full_env",
    "_check_c4v_staggered_invariants",
    "_ctm_tensor_c4v_staggered",
]
```

(c) Add two imports, in ruff's order, to the tenax import block:

```python
from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
```

(d) Replace the body of `_c4v_staggered_projectors` (lines 104-201, from `labels = Qf.labels()` to `return P_left, P_right`; keep its docstring) with:

```python
    sym, fused, col, mats = _c4v_staggered_blocks(Qf)
    decomp, keep = _c4v_staggered_rank(sym, mats, chi)
    cols = {}
    for q in sorted(keep):
        kind, U, V = decomp[q]
        ns = sorted(keep[q])
        cols[q] = U[:, ns]
        if kind == "pair":
            cols[_dual_charge(sym, q)] = V[:, ns]
    return _c4v_staggered_assemble(sym, fused, col, cols)
```

(e) Insert these four functions after `_c4v_staggered_projectors`. Their bodies are the old lines 104-119, 121-166 and 176-201, unchanged:

```python
def _c4v_staggered_blocks(Qf: SymmetricTensor):
    """The faces of the enlarged corner and its blocks: ``(sym, fused, col, mats)``.

    ``mats[q]`` is block (q, q): the fused states of charge q are its rows,
    the col states of charge q (the fused states of charge ``dual(q)``) its
    columns.  Metadata decides everything here, so it is safe under tracing.

    Raises:
        RuntimeError: the two faces of ``Qf`` do not carry the same
            flow-weighted charges (not a 2-sublattice corner).
    """
    labels = Qf.labels()
    fpos, cpos = labels.index("fused"), labels.index("col")
    fused, col = Qf.indices[fpos], Qf.indices[cpos]
    sym = fused.symmetry
    if not np.array_equal(col.charges, _dual(sym, fused.charges)):
        raise RuntimeError(
            "charged C4v projector: the two faces of the enlarged corner do not "
            "carry the same flow-weighted charges, so the env is not a "
            "2-sublattice env"
        )
    # Block (q, q): rows are the fused states of charge q, columns the col
    # states of charge q, which are the fused states of charge dual(q).
    mats = {
        int(key[fpos]): (block if fpos < cpos else block.T)
        for key, block in Qf.blocks.items()
    }
    return sym, fused, col, mats


def _c4v_staggered_rank(sym, mats, chi: int):
    """Decompose every block and rank all values together: ``(decomp, keep)``.

    Eager: the ranking reads the values on the host.  ``decomp[q]`` is
    ``("self", W, None)`` for a self-partner block and ``("pair", U, V)``
    for a mirror pair, keyed by the first charge of the pair in sorted
    order.  ``keep[q]`` lists the kept columns.  The ranking is the one
    described in ``_c4v_staggered_projectors``.

    Raises:
        ValueError: nothing survives the cutoff (a zero corner), or chi = 1
            and the largest value belongs to a mirror pair.
    """
    decomp, cands, done = {}, [], set()
    for q in sorted(mats):
        if q in done:
            continue
        qd = _dual_charge(sym, q)
        M = mats[q]
        if qd == q:
            # Self-partner block: symmetric by the mirror; rank by |lambda|.
            lam, W = jnp.linalg.eigh(0.5 * (M + M.T))
            decomp[q] = ("self", W, None)
            for n, v in enumerate(np.abs(np.asarray(lam))):
                cands.append((float(v), q, n, 0))
            done.add(q)
        else:
            # Mirror pair (q, dual(q)): one SVD.  Each value has two copies,
            # one state in sector q and one in sector dual(q).
            U, S, Vh = _dense_svd(M, full_matrices=False)
            decomp[q] = ("pair", U, Vh.T)
            for n, v in enumerate(np.asarray(S)):
                cands += [(float(v), q, n, 0), (float(v), q, n, 1)]
            done.update((q, qd))

    vmax = max((c[0] for c in cands), default=0.0)
    alive = sorted(
        (c for c in cands if c[0] > 0.0 and c[0] >= _REL_CUTOFF * vmax),
        key=lambda c: (-c[0], c[1], c[2], c[3]),
    )
    if not alive:
        raise ValueError("charged C4v projector: the enlarged corner is zero")

    copies: dict[tuple[int, int], int] = {}
    for _v, q, n, _copy in alive[:chi]:
        copies[(q, n)] = copies.get((q, n), 0) + 1
    keep: dict[int, list[int]] = {}
    for (q, n), count in copies.items():
        # A cut between the two copies of a pair value would keep one state
        # of a mirror pair without its partner: drop both.
        if decomp[q][0] == "pair" and count == 1:
            continue
        keep.setdefault(q, []).append(n)
    if not keep:
        raise ValueError(
            f"ctm_tensor_c4v: chi={chi} keeps no state of the charged C4v "
            "corner. Its largest value belongs to a mirror pair (q, dual(q)), "
            "which needs two states. Use chi >= 2."
        )
    return decomp, keep


def _c4v_staggered_assemble(sym, fused, col, cols):
    """``(P_left, P_right)`` from one column block per kept charge.

    One array per sector on both faces.  On the col face, the states of
    fused charge x have col charge dual(x), and so does their chi_new_r leg.
    The chi index lists the kept charges in sorted order, each with
    ``cols[x].shape[1]`` states.
    """
    left, right, chi_charges = {}, {}, []
    for x in sorted(cols):
        xd = _dual_charge(sym, x)
        left[(x, x)] = cols[x]
        right[(xd, xd)] = cols[x]
        chi_charges += [x] * cols[x].shape[1]
    chi_charges = np.asarray(chi_charges, dtype=np.int32)
    P_left = SymmetricTensor(
        left,
        (
            fused.flip_flow(),
            TensorIndex.from_charges(sym, chi_charges, IN, label="chi_new"),
        ),
    )
    P_right = SymmetricTensor(
        right,
        (
            col.flip_flow(),
            TensorIndex.from_charges(
                sym, _dual(sym, chi_charges), OUT, label="chi_new_r"
            ),
        ),
    )
    return P_left, P_right


def _c4v_staggered_layout(Qf: SymmetricTensor, chi: int) -> BondLayout:
    """The kept states per chi charge under the free ranking (eager).

    The layout that ``_c4v_staggered_projectors(Qf, chi)`` builds: a mirror
    pair keeps the same count in q and in ``dual(q)``.  Spec 2 freezes it
    for an AD run (``_c4v_staggered_projectors_at``) and re-runs it at each
    step to check the frozen layout.
    """
    sym, _fused, _col, mats = _c4v_staggered_blocks(Qf)
    decomp, keep = _c4v_staggered_rank(sym, mats, chi)
    dims = {}
    for q, ns in keep.items():
        dims[q] = len(ns)
        if decomp[q][0] == "pair":
            dims[_dual_charge(sym, q)] = len(ns)
    return BondLayout.from_dims(dims)
```

(f) Replace `_c4v_staggered_sweep` (lines 315-341) with:

```python
def _c4v_staggered_project(
    Qf: SymmetricTensor,
    T: SymmetricTensor,
    a: SymmetricTensor,
    P_left: SymmetricTensor,
    P_right: SymmetricTensor,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """``(C_new, T_new)`` from a projector, before the normalisation.

    C_new = P^T Q P is the enlarged corner projected on both faces.  T·a is
    the edge above a B-site; projected on both faces and negated, it is the
    edge above an A-site again.  Traceable: the AD sweep calls it with the
    traced projector.
    """
    C_new = contract(contract(P_left, Qf), P_right)
    C_new = C_new.relabels({"chi_new": "c_a", "chi_new_r": "c_b"})

    T_with_a = contract(T.relabel("D2", "u2"), a)  # (t_l, t_r, d2, l2, r2)
    Tg = _fuse_pair_by_label(T_with_a, "t_l", "l2", "fl", IN)
    Tg = _fuse_pair_by_label(Tg, "t_r", "r2", "fr", OUT)  # (fl, d2, fr)
    T_new = contract(P_left.relabel("fused", "fl"), Tg)  # (chi_new, d2, fr)
    T_new = contract(T_new, P_right.relabel("col", "fr"))  # (chi_new, d2, chi_new_r)
    T_new = _negate_charges(T_new).relabels(
        {"chi_new": "t_l", "d2": "D2", "chi_new_r": "t_r"}
    )
    return C_new, T_new


def _c4v_staggered_sweep(
    C: SymmetricTensor, T: SymmetricTensor, a: SymmetricTensor, chi: int
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """One step of the 2-sublattice C4v sweep; returns ``(C_new, T_new)``.

    ``_c4v_staggered_project`` with the eager projector; both outputs are
    normalised by ``max_abs``, as in ``_c4v_sweep``.
    """
    Qf = _c4v_staggered_enlarged_corner(C, T, a)
    P_left, P_right = _c4v_staggered_projectors(Qf, chi)
    C_new, T_new = _c4v_staggered_project(Qf, T, a, P_left, P_right)
    C_new = C_new * (1.0 / (C_new.max_abs() + 1e-15))
    T_new = T_new * (1.0 / (T_new.max_abs() + 1e-15))
    return C_new, T_new
```

(g) Replace `_ctm_tensor_c4v_staggered` (lines 374-398) with:

```python
def _c4v_staggered_loop(
    A: SymmetricTensor, chi: int, max_iter: int, conv_tol: float
) -> tuple[SymmetricTensor, SymmetricTensor, SymmetricTensor, CTMConvergeInfo]:
    """The loop of ``ctm_tensor_c4v``: ``(C, T, a, info)``.

    Start from the trace boundary, check I1-I3 on it, and sweep until the
    corner singular values change by less than ``conv_tol``
    (``_ctm_sv_diff`` with the forced-rank floor of #903), or for
    ``max_iter`` sweeps.  ``info.converged`` says which; ``info.sv_diff`` is
    the last sv change (inf before the second sweep).
    """
    a = _build_double_layer_tensor(A)
    a_B = _build_double_layer_tensor(_negate_charges(A))
    C, T = _c4v_staggered_initial_state(a)
    _check_c4v_staggered_invariants(C, T, a, a_B)
    max_rank = _forced_corner_rank(_max_virtual_bond_dim(a))
    prev_sv, diff, converged, iterations = None, float("inf"), False, 0
    for iterations in range(1, max_iter + 1):
        C, T = _c4v_staggered_sweep(C, T, a, chi)
        current_sv = _dense_svd(C.todense(), compute_uv=False)  # chi x chi
        if prev_sv is not None:
            diff = float(_ctm_sv_diff(current_sv, prev_sv, max_rank=max_rank))
            if diff < conv_tol:
                converged = True
                break
        prev_sv = current_sv
    info = CTMConvergeInfo(
        converged=converged,
        iterations=iterations,
        sv_diff=diff,
        best_iteration=iterations,
        step_multiplier=float("nan"),
    )
    return C, T, a, info


def _ctm_tensor_c4v_staggered(
    A: SymmetricTensor, chi: int, max_iter: int, conv_tol: float
) -> CTMTensorEnv:
    """Run the 2-sublattice C4v CTM; return the env of an A-site.

    ``_c4v_staggered_loop``, expanded to the eight env tensors.
    ``ctm_tensor_c4v`` validates ``A`` first.
    """
    C, T, _a, _info = _c4v_staggered_loop(A, chi, max_iter, conv_tol)
    return _c4v_staggered_to_full_env(C, T)
```

- [ ] **Step 5: Run the new tests to see them pass**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_staggered.py -k "layout_counts or layout_refuses or loop_reports" -q > $S/t1-green.log 2>&1; tail -n 3 $S/t1-green.log`
Expected: `7 passed`

- [ ] **Step 6: Check that the #1211 outputs are bit-identical**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u $S/snap_1211.py check > $S/t1-check.log 2>&1; cat $S/t1-check.log`
Expected: `identical`

- [ ] **Step 7: Run the #1211 suites**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_staggered.py tests/test_ctm_tensor_c4v.py -q > $S/t1-suite.log 2>&1`
Expected: the last line of `$S/t1-suite.log` reports `passed` with 0 failed and 0 errors.

- [ ] **Step 8: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t1.txt <<'EOF'
refactor(ctm): split the charged C4v projector into ranking and assembly

Spec 2 differentiates the 2-sublattice sweep at a frozen layout, so the
eager projector of #1211 splits into its block view, its host-side
ranking and the assembly of P from its columns.  _c4v_staggered_layout
returns the layout the ranking chooses, _c4v_staggered_project the
unnormalised step, and _c4v_staggered_loop the loop with a convergence
flag.  The #1211 outputs are bit-identical (snapshot of three fixtures).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/_ctm_tensor_c4v_staggered.py tests/test_c4v_staggered.py
git commit -F $S/msg-ad-t1.txt > $S/commit-ad-t1.log 2>&1; tail -n 3 $S/commit-ad-t1.log
```

Expected: one commit; ruff and ruff-format `Passed`.

---

### Task 2: The traced projector at a frozen layout

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_c4v_staggered.py` (imports, `__all__`, a new function after `_c4v_staggered_layout`)
- Test: `tests/test_c4v_staggered.py`

**Interfaces:**
- Consumes: `truncated_svd_ad(M, k) -> (U[:, :k], s, Vh[:k])` (custom_vjp, `k` static) and `regularized_eigh(M) -> (w ascending, v)` from `tenax.algorithms._ad_primitives`; `_c4v_staggered_blocks` and `_c4v_staggered_assemble` (Task 1); `BondLayout.dims`, `BondLayout.dim_of(q)`.
- Produces: `_c4v_staggered_projectors_at(Qf, layout: BondLayout) -> (P_left, P_right)`. The same keys, flows and column order as `_c4v_staggered_projectors(Qf, chi)` when `layout == _c4v_staggered_layout(Qf, chi)`; the column signs may differ. Traceable under `jax.jit` (with `layout` static) and `jax.grad`. Raises `ValueError` when the layout keeps a charge with no block, or more states than its block has.

- [ ] **Step 1: Write the failing tests**

In `tests/test_c4v_staggered.py`, add `_c4v_staggered_projectors_at` to the staggered import block, then append:

```python
def _swept_corner(name, chi):
    """The enlarged corner after the three ``_swept`` steps at this chi, and its layout."""
    _A, a, _a_B, states = _swept(name, chi=chi)
    C, T = states[-1]
    Qf = _c4v_staggered_enlarged_corner(C, T, a)
    return Qf, _c4v_staggered_layout(Qf, chi)


@pytest.mark.parametrize(("name", "chi"), [("762", 4), ("z4", 4), ("test", 6)])
def test_projectors_at_match_the_eager_projector_up_to_signs(name, chi):
    Qf, layout = _swept_corner(name, chi)
    sym = leg(Qf, "fused").symmetry
    kinds = {int(sym.dual(np.asarray([q]))[0]) == q for q in layout.sectors}
    assert kinds == {True, False}, f"vacuous: layout {layout.dims}"
    eager = _c4v_staggered_projectors(Qf, chi)
    traced = _c4v_staggered_projectors_at(Qf, layout)
    for P_e, P_t in zip(eager, traced):
        assert P_t.indices == P_e.indices
        assert set(P_t.blocks) == set(P_e.blocks)
        for key, block in P_e.blocks.items():
            overlap = np.sum(np.asarray(block) * np.asarray(P_t.blocks[key]), axis=0)
            np.testing.assert_allclose(np.abs(overlap), 1.0, atol=1e-10)


def test_projectors_at_keep_the_layout_values():
    Qf = _synthetic_q(U1Symmetry(), **_U1_Q)
    expected = {5: [3.0, 3.0, 2.0, 1.0, 1.0], 3: [3.0, 3.0, 2.0], 2: [3.0, 3.0]}
    for chi, values in expected.items():
        layout = _c4v_staggered_layout(Qf, chi)
        P_left, P_right = _c4v_staggered_projectors_at(Qf, layout)
        kept = Counter(np.asarray(leg(P_left, "chi_new").charges).tolist())
        assert BondLayout.from_dims(kept) == layout
        C = contract(contract(P_left, Qf), P_right)
        diag = [abs(float(v)) for b in C.blocks.values() for v in np.diag(np.asarray(b))]
        assert sorted(diag, reverse=True) == pytest.approx(values)


def test_projectors_at_trace_under_jit_and_grad():
    Qf, layout = _swept_corner("762", 4)
    (data,), treedef = jax.tree_util.tree_flatten(Qf)

    def leaves(x):
        Q = jax.tree_util.tree_unflatten(treedef, [x])
        P_left, P_right = _c4v_staggered_projectors_at(Q, layout)
        return P_left._data, P_right._data

    eager = leaves(data)
    for e, j in zip(eager, jax.jit(leaves)(data)):
        np.testing.assert_allclose(np.abs(np.asarray(j)), np.abs(np.asarray(e)), atol=1e-12)
    weights = jax.random.normal(jax.random.PRNGKey(1), eager[0].shape)
    grad = jax.grad(lambda x: jnp.sum(weights * leaves(x)[0]))(data)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_projectors_at_refuse_a_layout_the_corner_cannot_hold():
    Qf = _synthetic_q(U1Symmetry(), **_U1_Q)
    with pytest.raises(ValueError, match="of charge 7"):
        _c4v_staggered_projectors_at(Qf, BondLayout.from_dims({7: 1}))
    with pytest.raises(ValueError, match="keeps 3 states of charge 0"):
        _c4v_staggered_projectors_at(Qf, BondLayout.from_dims({0: 3}))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_staggered.py -k projectors_at -q > $S/t2-red.log 2>&1; tail -n 5 $S/t2-red.log`
Expected: `ERROR tests/test_c4v_staggered.py` with `ImportError: cannot import name '_c4v_staggered_projectors_at'`.

- [ ] **Step 3: Write the traced projector**

In `src/tenax/algorithms/_ctm_tensor_c4v_staggered.py`, add `from tenax.algorithms._ad_primitives import regularized_eigh, truncated_svd_ad` to the tenax imports, add `"_c4v_staggered_projectors_at",` to `__all__` after `"_c4v_staggered_projectors",`, and insert after `_c4v_staggered_layout`:

```python
def _c4v_staggered_projectors_at(
    Qf: SymmetricTensor, layout: BondLayout
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The block projector at a frozen layout, traceable (spec 2, section 2).

    The keys, flows and column order of ``_c4v_staggered_projectors`` for
    the layout that the free ranking chose, with differentiable
    decompositions:

    - A mirror pair (q, dual(q)), q the first of the two in sorted order:
      ``truncated_svd_ad(M_q, k_q)``.  U goes to sector q, V to sector
      dual(q).  Both copies of a pair value come from one SVD, so no
      backward term divides by their zero gap (#1178).
    - A self-partner block: ``regularized_eigh`` of 0.5 (M + M^T), and the
      k_q largest |lambda| in ascending index order (a stable argsort with
      a static k: the eager tie-break).
    - A charge with k_q = 0 gets no decomposition.

    The column signs may differ from the eager projector's
    (``truncated_svd_ad`` fixes them with ``_fix_svd_signs``); the AD sweep
    aligns them.

    Raises:
        RuntimeError: as ``_c4v_staggered_blocks``.
        ValueError: the layout keeps a charge that has no block, or more
            states than its block has.
    """
    sym, fused, col, mats = _c4v_staggered_blocks(Qf)
    for q, k in layout.dims:
        if q not in mats or k > min(mats[q].shape):
            shape = tuple(mats[q].shape) if q in mats else None
            raise ValueError(
                f"charged C4v projector: the layout keeps {k} states of charge "
                f"{q}, but the enlarged corner's block of that charge has shape "
                f"{shape}"
            )
    cols = {}
    for q in sorted(mats):
        k = layout.dim_of(q)
        if k == 0 or q in cols:
            continue
        qd = _dual_charge(sym, q)
        M = mats[q]
        if qd == q:
            lam, W = regularized_eigh(0.5 * (M + M.T))
            cols[q] = W[:, jnp.sort(jnp.argsort(-jnp.abs(lam))[:k])]
        else:
            U, _s, Vh = truncated_svd_ad(M, k)
            cols[q] = U
            cols[qd] = Vh.T
    return _c4v_staggered_assemble(sym, fused, col, cols)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_staggered.py -q > $S/t2-green.log 2>&1; tail -n 3 $S/t2-green.log`
Expected: every test passes, the 6 new ones included. If the regime assert of `test_projectors_at_match_the_eager_projector_up_to_signs` fails for one fixture (its layout has one kind of block only), ledger the layout it found and use the next chi up for that fixture.

- [ ] **Step 5: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t2.txt <<'EOF'
feat(ctm): traced charged C4v projector at a frozen layout

_c4v_staggered_projectors_at builds the #1211 block projector for a given
BondLayout with differentiable decompositions: truncated_svd_ad for each
mirror pair, regularized_eigh and a static top-k for each self-partner
block.  It matches the eager projector up to column signs, and it traces
under jit and grad.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/_ctm_tensor_c4v_staggered.py tests/test_c4v_staggered.py
git commit -F $S/msg-ad-t2.txt > $S/commit-ad-t2.log 2>&1; tail -n 3 $S/commit-ad-t2.log
```

Expected: one commit; the hooks pass.

---

### Task 3: The aligned sweep F̃ and the jitted step

**Files:**
- Create: `src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py`
- Modify: `tests/_c4v_staggered_fixtures.py` (new `T_EXACT`, `c4v_indices`, `charged_d2`; `c4v_fixture` builds its indices with `c4v_indices`)
- Modify: `tests/conftest.py:554` (register the two new test files)
- Test: `tests/test_c4v_charged_ad.py` (new)

**Interfaces:**
- Consumes: `_c4v_staggered_projectors_at(Qf, layout)` (Task 2); `_c4v_staggered_project(Qf, T, a, P_left, P_right)` and `_c4v_staggered_loop(A, chi, max_iter, conv_tol) -> (C, T, a, info)` (Task 1); `_c4v_staggered_enlarged_corner(C, T, a)`, `_leg(T, label)` (#1211); `_bond_blocks(tensor) -> [(key, block, pos)]`, `_bond_ref_blocks(t_new, t_ref) -> {key: stop_gradient block}`, `_bond_rebuild(tensor, blocks)`, `_bond_component_phases(G, components) -> w` (`tenax.algorithms.ad_utils`; `components` must be a tuple of numpy index arrays, not tuples, since `w.at[comp]` reads a tuple as a multi-axis index); `_normalize_by_max_abs` (`_ctm_tensor_moves`).
- Produces, in `tenax.algorithms._ctm_tensor_c4v_charged_ad` (module `ad` in the tests):
  - `_layout_of(C) -> BondLayout` (the counts of the `c_a` charges)
  - `_chi_components(T) -> tuple[np.ndarray, ...]`
  - `_scale_chi(X, g, chi_labels) -> SymmetricTensor`
  - `_align(C_new, T_new, T_ref) -> (C, T)`
  - `_sweep_at(A, C, T, layout) -> (C_new, T_new)`: F_L, normalised by `_normalize_by_max_abs`
  - `_aligned_sweep(A, C, T, layout) -> (C, T)`: F̃ with ref = T
  - `_rel_change(X_new, X_old) -> jax.Array`
  - `_step_impl(A, C, T, layout) -> (C1, T1, residual)`; `_step = jax.jit(_step_impl, static_argnames=("layout",))`
- Produces, in `tests/_c4v_staggered_fixtures.py`: `T_EXACT = 0.41986`; `c4v_indices(sym, vc, pc)`; `charged_d2(t) -> SymmetricTensor` (A[0000,0] = 1, A[e_k,1] = t, normalised; vc = pc = [0, 1]).

- [ ] **Step 1: Add the fixtures and register the test files**

In `tests/_c4v_staggered_fixtures.py`, replace the `indices = (...)` tuple inside `c4v_fixture` (and the `vd`/`pc` lines before it) with `indices = c4v_indices(sym, vc, pc)`, keep `vc = np.asarray(vc, dtype=np.int32)` only if a later line uses it (it does not; delete it), and insert after `c4v_fixture`:

```python
# The free parameter of the D=2 exact U(1) state (peps-torch; ``exact_u1_d2``
# in ``_ipeps_spectrum_fixtures.py``).  E(T_EXACT) = -0.660228017729 at chi=16;
# the chi=16 minimum over t is -0.660231093480 at t = 0.4210174.
T_EXACT = 0.41986


def c4v_indices(sym, vc, pc):
    """The five legs of a charged C4v site: u, l OUT with ``vc``; d, r IN
    with ``sym.dual(vc)``; phys IN with ``pc``."""
    vc = np.asarray(vc, dtype=np.int32)
    vd = sym.dual(vc)
    return (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, np.asarray(pc, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )


def charged_d2(t):
    """The D=2 U(1) C4v family: A[0000, 0] = 1 and A[e_k, 1] = t, normalised.

    One virtual leg at charge 1 with the spin flipped; ``T_EXACT`` is the
    peps-torch state.  The C4v projection has rank 2 on this layout, so any
    real C4v tensor of it is ``charged_d2(t)`` up to sign and norm.
    """
    A = np.zeros((2,) * 5)
    A[0, 0, 0, 0, 0] = 1.0
    for k in range(4):
        index = [0, 0, 0, 0]
        index[k] = 1
        A[(*index, 1)] = t
    A /= np.linalg.norm(A)
    return SymmetricTensor.from_dense(
        jnp.asarray(A), c4v_indices(U1Symmetry(), [0, 1], [0, 1])
    )
```

and add `from tenax.core.symmetry import U1Symmetry` to its imports. The new `c4v_fixture` body is:

```python
    indices = c4v_indices(sym, vc, pc)
    raw = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(key))
    data = symmetrize_c4v(jnp.asarray(raw.todense()))
    return SymmetricTensor.from_dense(data, indices)
```

In `tests/conftest.py`, after the line `"test_c4v_staggered.py": "core",` (line 554), insert:

```python
    # Charged C4v AD (spec 2): mechanism tests of the AD path at chi <= 6 --
    # the alignment, the element-wise phase, the custom_vjp gates, the
    # one-sweep VJP, the config refusals, the loop mocks and the dispatch.
    "test_c4v_charged_ad.py": "core",
    # Charged C4v AD (spec 2): converged FD with an h scan at chi=6, the D=2
    # optimum and the dense parity at chi=16.
    "test_c4v_charged_ad_physics.py": "algorithm",
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_c4v_charged_ad.py`:

```python
"""Mechanism tests of the charged C4v AD path (spec 2), at chi <= 6.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.
The converged FD checks, the D=2 optimum and the dense parity are in
``test_c4v_charged_ad_physics.py`` (algorithm bucket).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import T_EXACT, c4v_fixture, charged_d2, leg

from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_loop,
    _c4v_staggered_sweep,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.core.symmetry import U1Symmetry, ZnSymmetry

_NAMES = ("d2_off", "762", "z4")


@functools.cache
def _site(name):
    """The site tensors of these tests, each built once."""
    if name == "d2_off":  # the D=2 exact U(1) state, moved off its optimum
        return charged_d2(1.15 * T_EXACT)
    if name == "762":  # the #762 fixture: charged U(1), 10 block entries
        return c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0])
    if name == "z4":  # Z4: the pair (1, 3) and the self-partner charge 2
        return c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1])
    raise KeyError(name)


@functools.cache
def _cold(name):
    """``(A, C, T, layout)``: the #1211 loop at chi=6 to sv_diff < 1e-10."""
    A = _site(name)
    C, T, _a, info = _c4v_staggered_loop(A, 6, 200, 1e-10)
    assert info.converged, name
    return A, C, T, ad._layout_of(C)


# ------------------------------------------------------------------ #
# The aligned sweep (spec 2, section 1)                               #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("name", "dims"),
    [
        ("d2_off", {-1: 2, 0: 2, 1: 2}),
        ("762", {-2: 1, 0: 4, 2: 1}),
        ("z4", {0: 2, 1: 2, 3: 2}),
    ],
)
def test_layout_of_reads_the_corner_charges(name, dims):
    assert _cold(name)[3] == BondLayout.from_dims(dims)


@pytest.mark.parametrize("name", _NAMES)
def test_align_recovers_a_pure_sign_gauge(name):
    _A, C, T, _layout = _cold(name)
    n = leg(T, "t_l").dim
    g = np.where(np.random.default_rng(7).random(n) < 0.5, -1.0, 1.0)
    g[0] = -g[-1]  # at least one relative sign
    g = jnp.asarray(g)
    C_g = ad._scale_chi(C, g, ("c_a", "c_b"))
    T_g = ad._scale_chi(T, g, ("t_l", "t_r"))
    assert float(ad._rel_change(T_g, T)) > 0.1, "vacuous: the gauge leaves T as it is"
    C_a, T_a = ad._align(C_g, T_g, T)
    np.testing.assert_array_equal(np.asarray(C_a._data), np.asarray(C._data))
    np.testing.assert_array_equal(np.asarray(T_a._data), np.asarray(T._data))


@pytest.mark.parametrize("name", _NAMES)
def test_sweep_at_is_the_eager_sweep_up_to_signs(name):
    A, C, T, layout = _cold(name)
    C_e, T_e = _c4v_staggered_sweep(C, T, _build_double_layer_tensor(A), 6)
    assert ad._layout_of(C_e) == layout
    C_t, T_t = ad._align(*ad._sweep_at(A, C, T, layout), T_e)
    np.testing.assert_allclose(np.asarray(C_t._data), np.asarray(C_e._data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(T_t._data), np.asarray(T_e._data), atol=1e-12)


def test_step_compiles_once_and_matches_eager():
    A, C, T, layout = _cold("d2_off")
    jax.clear_caches()
    C1, T1, r1 = ad._step(A, C, T, layout)
    ad._step(A, C1, T1, layout)
    assert ad._step._cache_size() == 1
    C_e, T_e, r_e = ad._step_impl(A, C, T, layout)
    np.testing.assert_allclose(np.asarray(C1._data), np.asarray(C_e._data), atol=1e-13)
    np.testing.assert_allclose(np.asarray(T1._data), np.asarray(T_e._data), atol=1e-13)
    assert abs(float(r1) - float(r_e)) < 1e-13
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q > $S/t3-red.log 2>&1; tail -n 5 $S/t3-red.log`
Expected: `ERROR tests/test_c4v_charged_ad.py` with `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_tensor_c4v_charged_ad'`.

- [ ] **Step 4: Write the engine module (sweep and alignment)**

Create `src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py`:

```python
"""Implicit AD through the charged C4v (2-sublattice) CTM sweep (spec 2).

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.

``ctm_tensor_c4v`` runs the #1211 sweep on a charged C4v ``SymmetricTensor``
A, as the 2-sublattice state (A, negate(A)).  It ranks its projector on the
host, so it is eager-only.  This module differentiates the sweep at a
frozen layout L:

- ``_sweep_at`` is F_L: the sweep with the traced projector
  ``_c4v_staggered_projectors_at(Qf, L)``, both outputs normalised by
  ``_normalize_by_max_abs``.
- ``_align`` fixes the one free sign per chi index against a reference T:
  F~(e; A, ref) = align(F_L(e, a(A)), ref).
- ``charged_c4v_cold_start`` runs the #1211 loop to sv-convergence and
  freezes the layout of its last sweep.  ``_elementwise_phase`` iterates
  e <- F~(e; A, ref=e) to an element-wise fixed point.
- ``charged_c4v_converge`` is that phase as a ``jax.custom_vjp``.  Its
  backward solves (I - J^T + tau I) lambda = g with ``gmres_eager`` and
  returns dA = (dF~/dA)^T lambda.

The step, the adjoint matvec and the site VJP compile once per layout.
"""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_root_implicit_sym_sectors import BondLayout
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_project,
    _c4v_staggered_projectors_at,
    _leg,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ctm_tensor_moves import _normalize_by_max_abs
from tenax.algorithms.ad_utils import (
    _bond_blocks,
    _bond_component_phases,
    _bond_rebuild,
    _bond_ref_blocks,
)

__all__ = [
    "_align",
    "_aligned_sweep",
    "_chi_components",
    "_layout_of",
    "_rel_change",
    "_scale_chi",
    "_step",
    "_step_impl",
    "_sweep_at",
]


def _layout_of(C) -> BondLayout:
    """The layout of a 2-sublattice corner: the kept states per ``c_a`` charge."""
    return BondLayout.from_dims(Counter(np.asarray(_leg(C, "c_a").charges).tolist()))


def _chi_components(T) -> tuple[np.ndarray, ...]:
    """The chi positions that the blocks of T connect, as static index arrays.

    Union-find over the positions of t_l and t_r: a block joins every
    position it touches.  ``_bond_component_phases`` fixes the signs of a
    component up to one sign, which cancels on T (both chi legs carry the
    same sign).
    """
    labels = tuple(i.label for i in T.indices)
    a, b = labels.index("t_l"), labels.index("t_r")
    parent = list(range(T.indices[a].dim))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _key, blk, pos in _bond_blocks(T):
        if blk.size == 0:
            continue
        members = [int(p) for p in pos[a]] + [int(p) for p in pos[b]]
        root = find(members[0])
        for m in members[1:]:
            parent[find(m)] = root
    groups: dict[int, list[int]] = {}
    for i in range(len(parent)):
        groups.setdefault(find(i), []).append(i)
    return tuple(np.asarray(g, dtype=np.int64) for g in sorted(groups.values()))


def _scale_chi(X, g, chi_labels):
    """``X`` with the factor ``g[i]`` on chi position i of each leg in ``chi_labels``."""
    labels = tuple(i.label for i in X.indices)
    axes = [labels.index(lab) for lab in chi_labels]
    out = []
    for _key, blk, pos in _bond_blocks(X):
        for ax in axes:
            shape = [1] * blk.ndim
            shape[ax] = -1
            blk = blk * g[pos[ax]].reshape(shape)
        out.append(blk)
    return _bond_rebuild(X, out)


def _align(C_new, T_new, T_ref):
    """Fix the free sign of each chi index against ``T_ref`` (spec 2, section 1).

    Each projector column has a free sign.  Column i of P_right is column i
    of P_left, so a sign g_i acts on chi position i of both legs:
    T_new = g T g and C_new = g C g.  Then
    X[i, k] = sum_d T_new[i, d, k] T_ref[i, d, k] = g_i g_k W[i, k] with
    W >= 0 at the fixed point, the form that ``_bond_component_phases``
    reads.  The signs are piecewise constant, so they come from
    ``stop_gradient`` inputs, and ``_bond_ref_blocks`` stops the reference.
    C is diagonal at a fresh projection, so the signs leave it unchanged;
    they are applied anyway, to keep the map a gauge for any C.
    """
    labels = tuple(i.label for i in T_new.indices)
    a, b = labels.index("t_l"), labels.index("t_r")
    n = T_new.indices[a].dim
    refs = _bond_ref_blocks(T_new, T_ref)
    X = jnp.zeros((n, n), dtype=T_new._data.dtype)
    for key, blk, pos in _bond_blocks(T_new):
        rblk = refs.get(key)
        if rblk is None or blk.size == 0:
            continue
        M = jax.lax.stop_gradient(blk) * rblk
        rest = tuple(k for k in range(M.ndim) if k not in (a, b))
        M = jnp.sum(M, axis=rest)
        if a > b:
            M = M.T
        X = X.at[np.ix_(pos[a], pos[b])].add(M)
    g = jax.lax.stop_gradient(
        _bond_component_phases(X + X.T, _chi_components(T_new))
    )
    return _scale_chi(C_new, g, ("c_a", "c_b")), _scale_chi(T_new, g, ("t_l", "t_r"))


def _sweep_at(A, C, T, layout: BondLayout):
    """F_L: one sweep at the frozen layout, both outputs normalised by max |.|.

    ``_normalize_by_max_abs`` differentiates the scale as c |X|_F, so a
    cotangent along the output has a zero VJP; the plain ``max_abs()`` of
    the eager sweep does not (spec 2, section 2).
    """
    a = _build_double_layer_tensor(A)
    Qf = _c4v_staggered_enlarged_corner(C, T, a)
    P_left, P_right = _c4v_staggered_projectors_at(Qf, layout)
    C_new, T_new = _c4v_staggered_project(Qf, T, a, P_left, P_right)
    return _normalize_by_max_abs(C_new), _normalize_by_max_abs(T_new)


def _aligned_sweep(A, C, T, layout: BondLayout):
    """F~(e; A, ref=T) = align(F_L(e, a(A)), T)."""
    C_new, T_new = _sweep_at(A, C, T, layout)
    return _align(C_new, T_new, T)


def _rel_change(X_new, X_old) -> jax.Array:
    """max |X_new - X_old| / max |X_old| over the block data (one layout)."""
    return jnp.max(jnp.abs(X_new._data - X_old._data)) / jnp.max(jnp.abs(X_old._data))


def _step_impl(A, C, T, layout: BondLayout):
    """One element-wise step: ``(C1, T1, residual)``."""
    C1, T1 = _aligned_sweep(A, C, T, layout)
    return C1, T1, jnp.maximum(_rel_change(C1, C), _rel_change(T1, T))


_step = jax.jit(_step_impl, static_argnames=("layout",))
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py tests/test_c4v_staggered.py -q > $S/t3-green.log 2>&1; tail -n 3 $S/t3-green.log`
Expected: all pass; the new file contributes `10 passed`. Probe 5 (`$S/probe_align.py`, the same `_align` code with the eager projector) measured the aligned sweep converging element-wise on 7 cases, and the raw (unaligned) residual staying at 0.05-0.5 on 5 of them.

- [ ] **Step 6: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t3.txt <<'EOF'
feat(ctm): sign-aligned charged C4v sweep at a frozen layout

F~(e; A, ref) = align(F_L(e, a(A)), ref): the #1211 sweep with the traced
projector at a frozen BondLayout, normalised by _normalize_by_max_abs,
with one sign per chi index aligned to the reference through
_bond_component_phases.  _step jits one element-wise step per layout.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py tests/_c4v_staggered_fixtures.py tests/conftest.py tests/test_c4v_charged_ad.py
git commit -F $S/msg-ad-t3.txt > $S/commit-ad-t3.log 2>&1; tail -n 3 $S/commit-ad-t3.log
```

Expected: one commit; the hooks pass.

---

### Task 4: The forward phases, the custom_vjp and its gates

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py`
- Test: `tests/test_c4v_charged_ad.py`

**Interfaces:**
- Consumes: Task 3's `_step`, `_sweep_at`, `_align`, `_layout_of`; `_c4v_staggered_loop` (Task 1); `gmres_eager(matvec, b, *, tol, maxiter, restart) -> (x, relative residual)` (`maxiter` counts restarts; `x0` defaults to b); `CTMNotConvergedError(info, site, step=None, *, conv_tol, chi, detail)`; `CTMConvergeInfo`.
- Produces:
  - `charged_c4v_cold_start(A, ctm: CTMConfig) -> (C, T, layout)`; raises `CTMNotConvergedError` (message contains "cold start").
  - `_elementwise_phase(A, warm, layout, ctm, record) -> (C, T)`; writes `forward_sweeps`, `forward_residual`, `forward_rate` into `record`; raises `CTMNotConvergedError` (message contains "element-wise").
  - `charged_c4v_converge(A, warm, layout, ctm, record) -> (C, T)`: `jax.custom_vjp`, `nondiff_argnums=(2, 3, 4)`. Its backward writes `adjoint_matvecs` and `adjoint_residual` into `record` and raises `RuntimeError` ("relative residual" or "not finite").
  - `_adjoint_matvec(v, A, C, T, layout, tau)` and `_site_vjp(lam, A, C, T, layout)`: jitted, `layout` static.
  - `_KRYLOV_DIM = 30`.
  - The engine reads `ctm.chi`, `ctm.max_iter`, `ctm.conv_tol`, `ctm.adjoint_tol`, `ctm.adjoint_maxiter`, `ctm.adjoint_tikhonov`, and nothing else.

- [ ] **Step 1: Write the failing tests**

In `tests/test_c4v_charged_ad.py`, add `import dataclasses` to the standard-library imports, and add to the tenax imports:

```python
from tenax import CTMConfig
from tenax.algorithms._ctm_convergence_policy import CTMNotConvergedError
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_layout,
)
```

(merge the second import into the existing `_ctm_tensor_c4v_staggered` block). Append:

```python
# ------------------------------------------------------------------ #
# The forward phases and the custom_vjp (spec 2, sections 2-3)        #
# ------------------------------------------------------------------ #

CTM6 = CTMConfig(chi=6, max_iter=200, conv_tol=1e-10)


def _loss_through_converge(C, T, layout, record):
    """A_ -> <w, T*(A_)> through ``charged_c4v_converge``, w fixed and random."""
    w = jax.random.normal(jax.random.PRNGKey(3), T._data.shape)

    def loss(A_):
        _C1, T1 = ad.charged_c4v_converge(A_, (C, T), layout, CTM6, record)
        return jnp.sum(w * T1._data)

    return loss


def test_cold_start_freezes_the_last_layout():
    A, C, T, layout = _cold("d2_off")
    C1, T1, layout1 = ad.charged_c4v_cold_start(A, CTM6)
    assert layout1 == layout
    np.testing.assert_array_equal(np.asarray(C1._data), np.asarray(C._data))
    a = _build_double_layer_tensor(A)
    assert _c4v_staggered_layout(_c4v_staggered_enlarged_corner(C1, T1, a), 6) == layout


def test_cold_start_raises_when_the_sweep_does_not_converge():
    ctm = dataclasses.replace(CTM6, max_iter=3, conv_tol=1e-300)
    with pytest.raises(CTMNotConvergedError, match="cold start"):
        ad.charged_c4v_cold_start(_site("d2_off"), ctm)


def test_elementwise_phase_raises_at_max_iter():
    A, C, T, layout = _cold("d2_off")
    ctm = dataclasses.replace(CTM6, max_iter=2, conv_tol=1e-300)
    with pytest.raises(CTMNotConvergedError, match="element-wise"):
        ad._elementwise_phase(A, (C, T), layout, ctm, {})


def test_converge_returns_a_fixed_point():
    A, C, T, layout = _cold("d2_off")
    record = {}
    C1, T1 = ad.charged_c4v_converge(A, (C, T), layout, CTM6, record)
    _C2, _T2, res = ad._step(A, C1, T1, layout)
    assert float(res) < CTM6.conv_tol
    assert record["forward_sweeps"] >= 1
    assert record["forward_residual"] < CTM6.conv_tol


def test_elementwise_phase_needs_the_alignment(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    ctm = dataclasses.replace(CTM6, conv_tol=1e-12, max_iter=60)
    flip = jnp.asarray(np.r_[-1.0, np.ones(leg(T, "t_l").dim - 1)])
    assert float(ad._rel_change(ad._scale_chi(T, flip, ("t_l", "t_r")), T)) > 0.1, (
        "vacuous: the flip leaves T as it is"
    )
    sweep, calls = ad._sweep_at, [0]

    def flipping_sweep(A_, C_, T_, layout_):
        C1, T1 = sweep(A_, C_, T_, layout_)
        calls[0] += 1
        if calls[0] % 2 == 0:  # a pure gauge: chi index 0 changes sign
            C1 = ad._scale_chi(C1, flip, ("c_a", "c_b"))
            T1 = ad._scale_chi(T1, flip, ("t_l", "t_r"))
        return C1, T1

    monkeypatch.setattr(ad, "_step", ad._step_impl)  # eager, so the patches act
    monkeypatch.setattr(ad, "_sweep_at", flipping_sweep)
    ad._elementwise_phase(A, (C, T), layout, ctm, {})
    assert calls[0] >= 2, "vacuous: no sweep was flipped"
    monkeypatch.setattr(ad, "_align", lambda C_new, T_new, T_ref: (C_new, T_new))
    with pytest.raises(CTMNotConvergedError, match="element-wise"):
        ad._elementwise_phase(A, (C, T), layout, ctm, {})


def test_backward_refuses_an_unsolved_adjoint(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    monkeypatch.setattr(ad, "gmres_eager", lambda matvec, b, **kw: (b, 1.0))
    with pytest.raises(RuntimeError, match="relative residual"):
        jax.grad(_loss_through_converge(C, T, layout, {}))(A)


def test_backward_refuses_a_non_finite_adjoint(monkeypatch):
    A, C, T, layout = _cold("d2_off")
    monkeypatch.setattr(ad, "gmres_eager", lambda matvec, b, **kw: (b * jnp.nan, 0.0))
    with pytest.raises(RuntimeError, match="not finite"):
        jax.grad(_loss_through_converge(C, T, layout, {}))(A)


def test_programs_compile_once_per_layout():
    _A, C, T, layout = _cold("d2_off")
    jax.clear_caches()
    record = {}
    for t in (1.15 * T_EXACT, 1.1 * T_EXACT):
        A_t = charged_d2(t)
        g = jax.grad(_loss_through_converge(C, T, layout, record))(A_t)
        assert bool(jnp.all(jnp.isfinite(g._data)))
    assert ad._step._cache_size() == 1
    assert ad._adjoint_matvec._cache_size() == 1
    assert ad._site_vjp._cache_size() == 1
    assert set(record) == {
        "forward_sweeps",
        "forward_residual",
        "forward_rate",
        "adjoint_matvecs",
        "adjoint_residual",
    }
    assert record["adjoint_residual"] <= max(10 * CTM6.adjoint_tol, 1e-12)
    assert record["adjoint_matvecs"] >= 1
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q > $S/t4-red.log 2>&1; tail -n 12 $S/t4-red.log`
Expected: the 10 Task 3 tests pass; the 8 new ones fail with `AttributeError: module 'tenax.algorithms._ctm_tensor_c4v_charged_ad' has no attribute 'charged_c4v_cold_start'` (or `'charged_c4v_converge'`, `'_elementwise_phase'`, `'gmres_eager'`).

- [ ] **Step 3: Write the forward phases and the custom_vjp**

In `src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py`:

(a) Add to the imports (ruff order):

```python
import math
from functools import partial

from jax.flatten_util import ravel_pytree

from tenax.algorithms._ctm_convergence_policy import (
    CTMConvergeInfo,
    CTMNotConvergedError,
)
from tenax.algorithms._gmres_eager import gmres_eager
```

and add `_c4v_staggered_loop` to the `_ctm_tensor_c4v_staggered` import block.

(b) Extend `__all__` with `"_KRYLOV_DIM"`, `"_adjoint_matvec"`, `"_elementwise_phase"`, `"_site_vjp"`, `"charged_c4v_cold_start"`, `"charged_c4v_converge"` (keep the list sorted).

(c) Append:

```python
_KRYLOV_DIM = 30  # GMRES restart length, the C4v root engine's default


def charged_c4v_cold_start(A, ctm):
    """The #1211 loop to sv-convergence: ``(C, T, layout)``.

    The layout of the last sweep is the frozen layout L of an AD run.

    Raises:
        CTMNotConvergedError: no ``sv_diff < ctm.conv_tol`` in ``ctm.max_iter``
            sweeps.
    """
    C, T, _a, info = _c4v_staggered_loop(A, ctm.chi, ctm.max_iter, ctm.conv_tol)
    if not info.converged:
        raise CTMNotConvergedError(
            info,
            site="gradient",
            conv_tol=ctm.conv_tol,
            chi=ctm.chi,
            detail="charged C4v AD: the cold start (the #1211 sweep) did not converge",
        )
    return C, T, _layout_of(C)


def _elementwise_phase(A, warm, layout, ctm, record):
    """Iterate e <- F~(e; A, ref=e) from ``warm`` to an element-wise fixed point.

    Stops when max(|dC| / max|C|, |dT| / max|T|) < ``ctm.conv_tol``, and
    writes ``forward_sweeps``, ``forward_residual`` and ``forward_rate``
    (the last residual ratio, an estimate of rho(J)) into ``record``.

    Raises:
        CTMNotConvergedError: no fixed point in ``ctm.max_iter`` sweeps, or a
            non-finite residual.
    """
    C, T = warm
    res, prev, sweep = float("inf"), float("inf"), 0
    for sweep in range(1, ctm.max_iter + 1):
        C, T, r = _step(A, C, T, layout)
        prev, res = res, float(r)
        if not math.isfinite(res):
            break
        if res < ctm.conv_tol:
            rate = res / prev if math.isfinite(prev) and prev > 0 else float("nan")
            record.update(forward_sweeps=sweep, forward_residual=res, forward_rate=rate)
            return C, T
    info = CTMConvergeInfo(
        converged=False,
        iterations=sweep,
        sv_diff=res,
        best_iteration=sweep,
        step_multiplier=float("nan"),
    )
    raise CTMNotConvergedError(
        info,
        site="gradient",
        conv_tol=ctm.conv_tol,
        chi=ctm.chi,
        detail=(
            "charged C4v AD: no element-wise fixed point at the frozen layout "
            f"{layout.dims} (residual max(|dC|/max|C|, |dT|/max|T|) = {res:.3e})"
        ),
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def charged_c4v_converge(A, warm, layout, ctm, record):
    """The element-wise fixed point e* = F~(e*; A, ref=e*), with an implicit VJP.

    Args:
        A: the charged C4v site tensor.
        warm: ``(C, T)`` at ``layout`` to start from.  Its cotangent is zero.
        layout: the frozen ``BondLayout`` (``charged_c4v_cold_start``).
        ctm: the ``CTMConfig``.  The forward reads ``max_iter`` and
            ``conv_tol``; the backward reads ``adjoint_tol``,
            ``adjoint_maxiter`` and ``adjoint_tikhonov``.
        record: a dict that receives the forward and backward metadata.
    """
    return _elementwise_phase(A, warm, layout, ctm, record)


def _converge_fwd(A, warm, layout, ctm, record):
    C, T = _elementwise_phase(A, warm, layout, ctm, record)
    if jax.tree_util.tree_structure(warm) != jax.tree_util.tree_structure((C, T)):
        raise ValueError(
            "charged C4v AD: the warm env does not have the layout of the fixed point"
        )
    return (C, T), (A, C, T)


def _converge_bwd(layout, ctm, record, res, cot):
    """Solve (I - J^T + tau I) lambda = g; return (dF~/dA)^T lambda (spec 2, section 3)."""
    A, C, T = res
    b, _unravel = ravel_pytree(cot)
    tau = float(ctm.adjoint_tikhonov or 0.0)
    matvecs = [0]

    def matvec(v):
        matvecs[0] += 1
        return _adjoint_matvec(v, A, C, T, layout, tau)

    lam, resid = gmres_eager(
        matvec, b, tol=ctm.adjoint_tol, maxiter=ctm.adjoint_maxiter, restart=_KRYLOV_DIM
    )
    gate = max(10.0 * ctm.adjoint_tol, 1e-12)
    record.update(adjoint_matvecs=matvecs[0], adjoint_residual=float(resid))
    if not (math.isfinite(resid) and resid <= gate):
        raise RuntimeError(
            "charged C4v AD: the adjoint solve stopped at relative residual "
            f"{resid:.3e} > {gate:.1e} after {matvecs[0]} matvecs "
            f"(adjoint_tol={ctm.adjoint_tol}, adjoint_maxiter={ctm.adjoint_maxiter})"
        )
    if not bool(jnp.all(jnp.isfinite(lam))):
        raise RuntimeError("charged C4v AD: the adjoint solution is not finite")
    dA = _site_vjp(lam, A, C, T, layout)
    if not bool(jnp.all(jnp.isfinite(dA._data))):
        raise RuntimeError("charged C4v AD: the site cotangent dA is not finite")
    return dA, jax.tree_util.tree_map(jnp.zeros_like, (C, T))


charged_c4v_converge.defvjp(_converge_fwd, _converge_bwd)


def _adjoint_matvec_impl(v, A, C, T, layout, tau):
    """v - J^T v + tau v, J the Jacobian of e -> F~(e; A, ref=T) at e = (C, T)."""
    _flat, unravel = ravel_pytree((C, T))
    _out, vjp = jax.vjp(lambda e: _align(*_sweep_at(A, *e, layout), T), (C, T))
    (jt_v,) = vjp(unravel(v))
    return v - ravel_pytree(jt_v)[0] + tau * v


_adjoint_matvec = jax.jit(_adjoint_matvec_impl, static_argnames=("layout",))


def _site_vjp_impl(lam, A, C, T, layout):
    """(dF~/dA)^T lambda at the fixed point (C, T)."""
    _flat, unravel = ravel_pytree((C, T))
    _out, vjp = jax.vjp(lambda A_: _align(*_sweep_at(A_, C, T, layout), T), A)
    (dA,) = vjp(unravel(lam))
    return dA


_site_vjp = jax.jit(_site_vjp_impl, static_argnames=("layout",))
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q --durations=5 > $S/t4-green.log 2>&1; tail -n 12 $S/t4-green.log`
Expected: `18 passed`. Read the five slowest durations; ledger the file's total time (the core budget is 120 s for the whole file, ruling "Test markers").

- [ ] **Step 5: Measure the three programs at D=2, chi=16 (ruling 7)**

Create `$S/time_programs.py`:

```python
"""Compile and run times of the three charged C4v AD programs (D=2, chi=16)."""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree

sys.path.insert(0, "/home/yjkao/tenax/tests")
from _c4v_staggered_fixtures import T_EXACT, charged_d2

from tenax import CTMConfig
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad

A = charged_d2(T_EXACT)
ctm = CTMConfig(chi=16, max_iter=500, conv_tol=1e-10)
C, T, layout = ad.charged_c4v_cold_start(A, ctm)
v = jax.random.normal(jax.random.PRNGKey(0), ravel_pytree((C, T))[0].shape)
for name, fn, args in (
    ("step", ad._step, (A, C, T, layout)),
    ("matvec", ad._adjoint_matvec, (v, A, C, T, layout, 1e-6)),
    ("site_vjp", ad._site_vjp, (v, A, C, T, layout)),
):
    t = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(10):
        out = fn(*args)
    jax.block_until_ready(out)
    print(f"{name}: first call {first:.2f} s, run {(time.perf_counter() - t) * 100:.2f} ms", flush=True)
record = {}
t = time.perf_counter()
ad._elementwise_phase(A, (C, T), layout, ctm, record)
print(f"element-wise phase: {time.perf_counter() - t:.2f} s, {record}", flush=True)
```

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u $S/time_programs.py > $S/t4-time.log 2>&1; cat $S/t4-time.log`
Expected: three `first call` / `run` lines and one phase line. Ledger them as `Task 4: Ruling: matvec re-linearises per call — compile <x> s, run <y> ms at D=2 chi=16 — revisit at D=3 only if the matvec dominates (Task 9)`.

- [ ] **Step 6: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t4.txt <<'EOF'
feat(ctm): implicit VJP of the charged C4v fixed point

charged_c4v_cold_start runs the #1211 loop to sv-convergence and freezes
its last layout; _elementwise_phase iterates the aligned sweep to an
element-wise fixed point.  charged_c4v_converge is that phase as a
custom_vjp: the backward solves (I - J^T + tau I) lambda = g with
gmres_eager (restart 30) and fails closed on a relative residual above
max(10 adjoint_tol, 1e-12) or a non-finite result.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py tests/test_c4v_charged_ad.py
git commit -F $S/msg-ad-t4.txt > $S/commit-ad-t4.log 2>&1; tail -n 3 $S/commit-ad-t4.log
```

Expected: one commit; the hooks pass.

---

### Task 5: Gradient checks

**Files:**
- Modify: `tests/_c4v_staggered_fixtures.py` (new `with_data`, `c4v_direction`, `fd_plateau`, `frozen_projectors`)
- Test: `tests/test_c4v_charged_ad.py` (one-sweep VJP, radial cotangent)
- Test: `tests/test_c4v_charged_ad_physics.py` (new: converged FD)

**Interfaces:**
- Consumes: Task 4's `charged_c4v_cold_start`, `_elementwise_phase`, `charged_c4v_converge`; Task 3's `_sweep_at`, `_align`; `compute_energy_ctm_tensor(A, env, gate, d=2)` and `_c4v_staggered_to_full_env(C, T)`; `symmetrize_c4v(dense)`.
- Produces, in `tests/_c4v_staggered_fixtures.py`:
  - `with_data(X, data)`: X's pytree with a new flat buffer;
  - `c4v_direction(A, key) -> jax.Array`: a unit C4v direction in A's buffer;
  - `fd_plateau(phi, hs=...) -> (fd, fd_unc)`: the central difference on its h plateau;
  - `frozen_projectors()`: a context manager, the "P frozen" mutation (stop_gradient on `_c4v_staggered_projectors_at` in the engine), with `jax.clear_caches()` on entry and exit.

These tests pass when written: they check Tasks 3-4. So the RED step here is the mutation step: each test is run once against its named mutant before the commit.

- [ ] **Step 1: Add the shared gradient helpers**

Append to `tests/_c4v_staggered_fixtures.py` (add `import contextlib` to its standard-library imports):

```python
def with_data(X, data):
    """``X`` with a new flat block buffer (same pytree structure)."""
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(X), [data])


def c4v_direction(A, key):
    """A random unit direction in the C4v subspace of A's block buffer."""
    r = jax.random.normal(key, A._data.shape)
    v = SymmetricTensor.from_dense(
        symmetrize_c4v(with_data(A, r).todense()), A.indices
    )._data
    return v / jnp.linalg.norm(v)


FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def fd_plateau(phi, hs=FD_STEPS):
    """The central difference of ``phi`` at 0 on its h plateau: ``(fd, fd_unc)``.

    ``fd_unc`` is the smallest difference between neighbouring h; ``fd`` is
    the mean of that pair.  The pair is chosen by the agreement of
    neighbouring h, never by the agreement with AD (spec 2, the FD rule).
    """
    fds = [(phi(h) - phi(-h)) / (2.0 * h) for h in hs]
    diffs = [abs(x - y) for x, y in zip(fds, fds[1:])]
    k = int(np.argmin(diffs))
    return 0.5 * (fds[k] + fds[k + 1]), diffs[k]


@contextlib.contextmanager
def frozen_projectors():
    """The "P frozen" mutation: stop_gradient on the traced projector.

    The engine has no frozen option (spec 2, section 3); the FD regime
    asserts compare against this.  The caches are cleared on entry and exit,
    so no jitted program keeps the wrong projector.
    """
    from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad

    real = ad._c4v_staggered_projectors_at

    def frozen(Qf, layout):
        return jax.tree_util.tree_map(jax.lax.stop_gradient, real(Qf, layout))

    jax.clear_caches()
    ad._c4v_staggered_projectors_at = frozen
    try:
        yield
    finally:
        ad._c4v_staggered_projectors_at = real
        jax.clear_caches()
```

- [ ] **Step 2: Write the core gradient tests**

In `tests/test_c4v_charged_ad.py`, extend the fixture import to `from _c4v_staggered_fixtures import (T_EXACT, c4v_fixture, charged_d2, fd_plateau, frozen_projectors, leg, with_data)`, then append:

```python
# ------------------------------------------------------------------ #
# One-sweep gradient checks (spec 2, section 6)                       #
# ------------------------------------------------------------------ #


def _one_sweep(C, T, layout):
    """A_ -> F~((C, T); A_, ref=T): the map whose VJP is (dF~/dA)^T."""
    return lambda A_: ad._align(*ad._sweep_at(A_, C, T, layout), T)


def _orthogonal_to(x, key):
    """A random vector with no component along x."""
    w = jax.random.normal(key, x.shape)
    return w - jnp.vdot(x, w) / jnp.vdot(x, x) * x


def test_one_sweep_vjp_matches_fd():
    """Catches a frozen projector; the cotangent has no radial part (ruling 5)."""
    A, C, T, layout = _cold("d2_off")
    f = _one_sweep(C, T, layout)
    C1, T1 = f(A)
    w = (
        with_data(C1, _orthogonal_to(C1._data, jax.random.PRNGKey(1))),
        with_data(T1, _orthogonal_to(T1._data, jax.random.PRNGKey(2))),
    )
    v = jax.random.normal(jax.random.PRNGKey(3), A._data.shape)
    v = v / jnp.linalg.norm(v)

    def phi(eps):
        Ce, Te = f(with_data(A, A._data + eps * v))
        return float(jnp.vdot(w[0]._data, Ce._data) + jnp.vdot(w[1]._data, Te._data))

    (dA,) = jax.vjp(f, A)[1](w)
    slope = float(jnp.vdot(dA._data, v))
    fd, fd_unc = fd_plateau(phi)
    with frozen_projectors():
        (dA_frozen,) = jax.vjp(f, A)[1](w)
    frozen = float(jnp.vdot(dA_frozen._data, v))
    # grad_err first: a mutant must fail here, not on a regime assert.
    assert abs(slope - fd) <= 3 * fd_unc + 1e-12, (slope, fd, fd_unc)
    assert abs(slope) > 100 * fd_unc, f"vacuous: |g.v|={abs(slope):.1e}, fd_unc={fd_unc:.1e}"
    assert abs(slope - frozen) > 10 * fd_unc, "vacuous: dP/dA is in the FD noise"


@pytest.mark.parametrize("which", ["C", "T"])
def test_a_radial_cotangent_has_zero_vjp(which):
    """Catches the plain max_abs() in F_L (ruling 4): only the smooth scale
    gives a cotangent along the output a zero VJP."""
    A, C, T, layout = _cold("d2_off")
    f = _one_sweep(C, T, layout)
    C1, T1 = f(A)
    X = C1 if which == "C" else T1
    r = jax.random.normal(jax.random.PRNGKey(4), X._data.shape)
    r = r * (jnp.linalg.norm(X._data) / jnp.linalg.norm(r))
    zero_C = jax.tree_util.tree_map(jnp.zeros_like, C1)
    zero_T = jax.tree_util.tree_map(jnp.zeros_like, T1)
    radial = (C1, zero_T) if which == "C" else (zero_C, T1)
    rnd = (with_data(C1, r), zero_T) if which == "C" else (zero_C, with_data(T1, r))
    vjp = jax.vjp(f, A)[1]
    (d_radial,) = vjp(radial)
    (d_random,) = vjp(rnd)
    n_random = float(jnp.linalg.norm(d_random._data))
    assert n_random > 1e-3, "vacuous: the random cotangent has no VJP"
    assert float(jnp.linalg.norm(d_radial._data)) <= 1e-10 * n_random
```

- [ ] **Step 3: Write the converged FD tests**

Create `tests/test_c4v_charged_ad_physics.py`:

```python
"""Physics tests of the charged C4v AD path (spec 2): converged FD with an h
scan, the D=2 optimum and the dense parity.  ``algorithm`` bucket.

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``,
section 6.  Mechanism tests are in ``test_c4v_charged_ad.py`` (core).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import (
    T_EXACT,
    c4v_direction,
    c4v_fixture,
    charged_d2,
    fd_plateau,
    frozen_projectors,
    with_data,
)

from tenax import CTMConfig, heisenberg_gate, sublattice_rotate_gate
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_tensor_c4v_staggered import _c4v_staggered_to_full_env
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.core.symmetry import U1Symmetry, ZnSymmetry

GATE = sublattice_rotate_gate(heisenberg_gate())

# The FD rule: an exact adjoint (tau = 0) solved to 1e-12, and every forward
# re-converged element-wise to 1e-13 from the base fixed point (the floor is
# about 1e-15: probe 7).
CTM_FD = CTMConfig(
    chi=6,
    max_iter=500,
    conv_tol=1e-13,
    adjoint_tol=1e-12,
    adjoint_maxiter=100,
    adjoint_tikhonov=0.0,
)

_SITES = {
    "d2_off": lambda: charged_d2(1.15 * T_EXACT),
    "762": lambda: c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0]),
    "z4": lambda: c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1]),
}


def _energy_of(A, C, T):
    return compute_energy_ctm_tensor(A, _c4v_staggered_to_full_env(C, T), GATE, d=2)


@functools.cache
def _fixed_point(name):
    """``(A, (C*, T*), layout)`` at chi=6, element-wise to 1e-13."""
    A = _SITES[name]()
    C, T, layout = ad.charged_c4v_cold_start(A, dataclasses.replace(CTM_FD, conv_tol=1e-10))
    return A, ad._elementwise_phase(A, (C, T), layout, CTM_FD, {}), layout


def _grad(A, warm, layout, record):
    def energy(A_):
        C, T = ad.charged_c4v_converge(A_, warm, layout, CTM_FD, record)
        return _energy_of(A_, C, T)

    return jax.grad(energy)(A)


@pytest.mark.parametrize(
    ("name", "key"), [("d2_off", 0), ("762", 0), ("762", 1), ("z4", 0)]
)
def test_converged_gradient_matches_fd(name, key):
    """Catches a frozen projector and lambda = g (spec 2, section 6)."""
    A, warm, layout = _fixed_point(name)
    v = c4v_direction(A, jax.random.PRNGKey(key))
    record = {}
    slope = float(jnp.vdot(_grad(A, warm, layout, record)._data, v))

    def phi(eps):
        A_e = with_data(A, A._data + eps * v)
        C, T = ad._elementwise_phase(A_e, warm, layout, CTM_FD, {})
        return float(_energy_of(A_e, C, T))

    fd, fd_unc = fd_plateau(phi)
    with frozen_projectors():
        frozen = float(jnp.vdot(_grad(A, warm, layout, {})._data, v))
    print(
        f"{name} dir {key}: |g.v|={abs(slope):.3e} fd_unc={fd_unc:.1e} "
        f"grad_err={abs(slope - fd):.1e} flow-frozen={abs(slope - frozen):.1e}"
    )
    # grad_err first: a mutant must fail here, not on a regime assert.
    assert abs(slope - fd) <= 3 * fd_unc + 1e-12, (slope, fd, fd_unc)
    assert abs(slope) > 100 * fd_unc, "vacuous: |g.v| is at the FD floor"
    assert abs(slope - frozen) > 10 * fd_unc, "vacuous: dP/dA is in the FD noise"
    assert record["adjoint_residual"] <= 1e-11
```

- [ ] **Step 4: Run the gradient tests**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py tests/test_c4v_charged_ad_physics.py -q -s --durations=8 > $S/t5-green.log 2>&1`
Expected: `25 passed`: 21 in the core file (18 + 1 + 2) and 4 in the physics file. Copy the four printed `|g.v| fd_unc grad_err flow-frozen` lines into the ledger. If a regime assert fails at chi=6 (ruling 11), rerun that fixture at chi=4 (`dataclasses.replace(CTM_FD, chi=4)` in `_fixed_point`), and ledger the ruling with both measurements.

- [ ] **Step 5: Watch each gradient test fail on its mutant**

Each mutant is applied in place, run, and reverted; nothing else runs while a mutant is in place. Create `$S/mutate.py`:

```python
"""Apply one named mutant, run its tests, revert (git checkout --) in finally."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/home/yjkao/tenax")
MUTANTS = {
    "P_frozen": (
        "src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py",
        "    P_left, P_right = _c4v_staggered_projectors_at(Qf, layout)\n",
        "    P_left, P_right = jax.tree_util.tree_map(\n"
        "        jax.lax.stop_gradient, _c4v_staggered_projectors_at(Qf, layout)\n"
        "    )\n",
    ),
    "lambda_is_g": (
        "src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py",
        "    lam, resid = gmres_eager(\n",
        "    lam, resid = b, 0.0\n    _unused = lambda: gmres_eager(\n",
    ),
    "plain_max_abs": (
        "src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py",
        "    return _normalize_by_max_abs(C_new), _normalize_by_max_abs(T_new)\n",
        "    return C_new * (1.0 / (C_new.max_abs() + 1e-15)), "
        "T_new * (1.0 / (T_new.max_abs() + 1e-15))\n",
    ),
}


def main(name, *pytest_args):
    rel, old, new = MUTANTS[name]
    path = ROOT / rel
    text = path.read_text()
    assert text.count(old) == 1, f"anchor of {name} is not unique"
    for pyc in ROOT.glob("src/tenax/algorithms/__pycache__/*.pyc"):
        pyc.unlink()
    path.write_text(text.replace(old, new))
    try:
        cmd = ["uv", "run", "--no-sync", "pytest", "--no-cov", "-p", "no:cacheprovider", "-q", *pytest_args]
        rc = subprocess.run(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu"}).returncode
    finally:
        subprocess.run(["git", "checkout", "--", rel], cwd=ROOT, check=True)
    print(f"MUTANT {name}: pytest exit {rc} ({'KILLED' if rc != 0 else 'SURVIVED'})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
```

The `lambda_is_g` mutant keeps the call as a dead lambda, so the file still parses: `lam, resid = b, 0.0`, then `_unused = lambda: gmres_eager(` with the original argument lines and `)`. A dead tuple would not parse, because the call has keyword arguments. The gate passes (`resid = 0.0`), so the backward returns (∂F̃/∂A)ᵀ g: the one-sweep term only.

Run, one at a time (each in the background; read each log):

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax
taskset -c 0-11 uv run --no-sync python -u $S/mutate.py P_frozen tests/test_c4v_charged_ad.py -k one_sweep > $S/t5-mut1.log 2>&1
taskset -c 0-11 uv run --no-sync python -u $S/mutate.py P_frozen tests/test_c4v_charged_ad_physics.py > $S/t5-mut2.log 2>&1
taskset -c 0-11 uv run --no-sync python -u $S/mutate.py lambda_is_g tests/test_c4v_charged_ad_physics.py > $S/t5-mut3.log 2>&1
taskset -c 0-11 uv run --no-sync python -u $S/mutate.py plain_max_abs tests/test_c4v_charged_ad.py -k radial > $S/t5-mut4.log 2>&1
```

Expected: each log ends with `KILLED`, and `git status --short src/` is empty after each run. The `P_frozen` physics run must fail on the `grad_err` assert, not on a regime assert (read the assertion line); the regime assert compares against the same frozen gradient, so a mutant that fails only there is not a kill. Ledger: `Task 5: mutants P_frozen (2 files), lambda_is_g, plain_max_abs: KILLED`.

- [ ] **Step 6: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t5.txt <<'EOF'
test(ctm): gradient checks of the charged C4v AD path

One-sweep VJP against FD (radial-free cotangent), a zero VJP for a
radial cotangent, and the converged energy gradient against FD with an h
scan on three fixtures at chi=6, with regime asserts on |g.v| and on the
projector term.  Mutants killed: P frozen, lambda = g, plain max_abs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add tests/_c4v_staggered_fixtures.py tests/test_c4v_charged_ad.py tests/test_c4v_charged_ad_physics.py
git commit -F $S/msg-ad-t5.txt > $S/commit-ad-t5.log 2>&1; tail -n 3 $S/commit-ad-t5.log
```

Expected: one commit; the hooks pass.

---

### Task 6: The optimiser loop

**Files:**
- Create: `src/tenax/algorithms/ipeps_optimize_c4v_charged.py`
- Test: `tests/test_c4v_charged_ad.py`

**Interfaces:**
- Consumes: the engine module object `tenax.algorithms._ctm_tensor_c4v_charged_ad` (imported as `_engine`, so a test's `monkeypatch.setattr(ad, ...)` reaches the loop): `charged_c4v_cold_start(A, ctm) -> (C, T, layout)`, `_elementwise_phase(A, warm, layout, ctm, record) -> (C, T)`, `charged_c4v_converge(A, warm, layout, ctm, record) -> (C, T)` (Task 4). `_c4v_staggered_enlarged_corner(C, T, a)`, `_c4v_staggered_layout(Qf, chi) -> BondLayout` (a cut through a mirror pair drops both copies; it raises `ValueError` only when it keeps no state), `_c4v_staggered_to_full_env(C, T)` (#1211, Task 1). From `ipeps_optimize` (a local import, as `ipeps_optimize` imports this module lazily): `_backtracking_line_search(params, direction, grad, energy, loss_fn_fwd, ..., max_steps) -> (params, energy, alpha)` (it returns the input params when no trial is lower), `_is_real_decrease(new, old) -> bool`, `_log_ad_step(backend, step, num_steps, energy, delta_energy, best_energy, *, grad_norm)`, `_log_ctm_not_converged_reset(exc, *, conv_tol, n_reset, retries, verbose)` (it prints `exc.step`), `_normalize_stall_recovery(config, *, unit_cell) -> iPEPSConfig` (`gs_stall_recovery=None` becomes `"reset"`), `_resolve_line_search_method(config, ctm) -> "armijo" | "hager_zhang"`. From `_ipeps_optimize_shared`: `_converged_outer(config, dE, grad_norm) -> bool`, `_euclidean_grads(g)`, `_grad_l2_norm(g) -> float`, `_log_ad_converged(...)`, `_normalize_params(x) = x / (|x| + 1e-10)`, `_should_accept_best(*, current_best, candidate, floor) -> bool`. `hager_zhang_line_search(phi, dphi, phi0, dphi0, *, alpha_init, rho, max_step, energy_bound, max_iter, bracket_only_phi) -> (alpha, f_alpha, wolfe)`: a probe exception becomes a bad point (phi: +inf; dphi: NaN), except `LineSearchAborted`, which ends the search at the best phi point. `lbfgs_two_loop(g, history, h0_matvec)`. `CTMNotConvergedError` is a subclass of `RuntimeError`.
- Produces, in `tenax.algorithms.ipeps_optimize_c4v_charged` (module `opt` in the tests):
  - `optimize_gs_ad_c4v_charged(hamiltonian_gate, A_init, config) -> (A, env, E)` or `(A, env, E, history)` with the ruling 15 keys.
  - `_check_charged_c4v_config(config, ctm, A_init) -> None`; raises `ValueError("charged C4v AD: <setting> is not supported: <reason>")`.
  - `_c4v_projector_matrix(A) -> jax.Array` (n × n).
  - `_trial_energy(A, warm, layout, ctm, gate, d) -> float` (+inf without a fixed point).
  - `_free_layout(A, C, T, chi) -> BondLayout | None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_c4v_charged_ad.py`, extend the imports (ruff sorts them on commit):

```python
from tenax import CTMConfig, heisenberg_gate, iPEPSConfig, sublattice_rotate_gate
from tenax.algorithms import ipeps_optimize_c4v_charged as opt
from tenax.algorithms._ctm_convergence_policy import (
    CTMConvergeInfo,
    CTMNotConvergedError,
)
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.tensor import SymmetricTensor
```

and add `_c4v_staggered_to_full_env` to the `_ctm_tensor_c4v_staggered` import block. Append:

```python
# ------------------------------------------------------------------ #
# The optimiser loop (spec 2, section 5)                              #
# ------------------------------------------------------------------ #

GATE = sublattice_rotate_gate(heisenberg_gate())
_MESH = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))


def _config(ctm=None, **kw):
    """A charged C4v config at chi=6: three Armijo steps, then ``kw``."""
    ctm_fields = {"chi": 6, "max_iter": 200, "conv_tol": 1e-10, **(ctm or {})}
    fields = {
        "max_bond_dim": 2,
        "gs_c4v": True,
        "su_init": False,
        "gs_num_steps": 3,
        "gs_line_search_method": "armijo",
        "ctm": CTMConfig(**ctm_fields),
    }
    return iPEPSConfig(**{**fields, **kw})


def _c4v_error(A):
    """max |symmetrize_c4v(todense(A)) - todense(A)|."""
    dense = np.asarray(A.todense())
    return float(np.max(np.abs(np.asarray(symmetrize_c4v(jnp.asarray(dense))) - dense)))


_REFUSED = [
    ({"gs_implicit_ad": False}, None, "gs_implicit_ad"),
    ({}, {"projector_method": "qr"}, "projector_method"),
    ({}, {"forward_gauge": "phase"}, "forward_gauge"),
    ({}, {"ctm_conv_method": "sv"}, "ctm_conv_method"),
    ({}, {"projector_backward": "standard"}, "projector_backward"),
    ({}, {"projector_backward": "lorentzian"}, "projector_backward"),
    ({}, {"on_unconverged": "warn"}, "on_unconverged"),
    ({}, {"chi_ramp": [(4, 2)]}, "chi_ramp"),
    ({}, {"chi_auto_bump": True}, "chi_auto_bump"),
    (
        {},
        {"ctmrg_heuristic_increase_chi": True, "chi_max": 8},
        "ctmrg_heuristic_increase_chi",
    ),
    ({}, {"fuse_virtual_legs": False}, "fuse_virtual_legs"),
    ({}, {"ctm_mixing": 0.5}, "ctm_mixing"),
    ({}, {"device_mesh": _MESH}, "device_mesh"),
    ({"gs_stall_recovery": "noise"}, None, "gs_stall_recovery"),
    ({"cg_gates": object()}, None, "cg_gates"),
    ({"gs_optimizer": "adam"}, None, "gs_optimizer"),
    ({"gs_line_search": False}, None, "gs_line_search"),
    ({"gs_chi_schedule_steps": [(4, 2)]}, None, "gs_chi_schedule_steps"),
    ({"gs_ctm_conv_tol_schedule": [(0.5, 1e-8)]}, None, "gs_ctm_conv_tol_schedule"),
    ({"gs_ctm_max_iter_schedule": [(0.5, 50)]}, None, "gs_ctm_max_iter_schedule"),
    (
        {"gs_plateau_patience_schedule": [(0.5, 10)]},
        None,
        "gs_plateau_patience_schedule",
    ),
]


@pytest.mark.parametrize(
    ("kw", "ctm", "match"),
    [
        pytest.param(kw, ctm, match, id=f"{match}-{i}")
        for i, (kw, ctm, match) in enumerate(_REFUSED)
    ],
)
def test_refusals(kw, ctm, match):
    """Each refused setting of spec 2, section 5 and ruling 8 raises first."""
    with pytest.raises(ValueError, match=f"charged C4v AD: {match}"):
        opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), _config(ctm, **kw))


def test_complex_data_is_refused():
    A = _site("d2_off")
    A_complex = with_data(A, A._data.astype(jnp.complex128))
    with pytest.raises(ValueError, match="charged C4v AD: complex data"):
        opt.optimize_gs_ad_c4v_charged(GATE, A_complex, _config())


def test_zero_steps_returns_the_converged_start():
    """gs_num_steps=0: the projected start, its converged env, its energy."""
    A = _site("d2_off")
    A_out, env, E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, A, _config(gs_num_steps=0, return_history=True)
    )
    np.testing.assert_allclose(np.asarray(A_out._data), np.asarray(A._data), atol=1e-12)
    assert abs(E - float(compute_energy_ctm_tensor(A_out, env, GATE, d=2))) < 1e-12
    _A, C, T, _layout = _cold("d2_off")
    env_cold = _c4v_staggered_to_full_env(C, T)
    assert abs(E - float(compute_energy_ctm_tensor(A, env_cold, GATE, d=2))) < 1e-9
    assert hist["num_steps"] == 0 and hist["energies"] == []
    assert hist["layout"] == BondLayout.from_dims({-1: 2, 0: 2, 1: 2}).dims


def test_a_non_c4v_start_is_projected():
    """Review Focus 1: a charged start with raw (non-C4v) blocks is projected,
    and the run stays C4v after an accepted step."""
    raw = SymmetricTensor.random_normal(_site("d2_off").indices, jax.random.PRNGKey(5))
    assert _c4v_error(raw) > 0.1, "vacuous: the raw start is C4v"
    A, _env, E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, raw, _config(gs_num_steps=2, return_history=True)
    )
    assert hist["energies"][1] < hist["energies"][0] - 1e-12, "vacuous: no step"
    assert _c4v_error(A) < 1e-12
    assert np.isfinite(E)


def test_gs_projector_method_overrides_before_the_check():
    """Review Focus 2: gs_projector_method replaces ctm.projector_method first."""
    A = _site("d2_off")
    run = _config({"projector_method": "qr"}, gs_projector_method="svd", gs_num_steps=0)
    assert np.isfinite(opt.optimize_gs_ad_c4v_charged(GATE, A, run)[2])
    refused = _config(
        {"projector_method": "svd"}, gs_projector_method="qr", gs_num_steps=0
    )
    with pytest.raises(ValueError, match="charged C4v AD: projector_method='qr'"):
        opt.optimize_gs_ad_c4v_charged(GATE, A, refused)


def test_a_missing_block_is_canonicalised():
    """Review Focus 4: an A_init that stores fewer blocks than its charges allow."""
    A = _site("d2_off")
    blocks = dict(A.blocks)
    del blocks[next(iter(blocks))]
    A_missing = SymmetricTensor(blocks, A.indices)
    assert A_missing._data.size < A._data.size, "vacuous: no block is missing"
    A_out, _env, E = opt.optimize_gs_ad_c4v_charged(
        GATE, A_missing, _config(gs_num_steps=0)
    )
    assert A_out._data.size == A._data.size
    assert _c4v_error(A_out) < 1e-12 and np.isfinite(E)


def test_two_chi_in_one_process():
    """Review Focus 5: each chi gets its own programs; no stale one serves a run."""
    A = _site("d2_off")
    for chi in (4, 6, 4):
        _A, _env, E = opt.optimize_gs_ad_c4v_charged(
            GATE, A, _config({"chi": chi}, gs_num_steps=0)
        )
        env = ctm_tensor_c4v(A, chi, max_iter=200, conv_tol=1e-10)
        E_eager = float(compute_energy_ctm_tensor(A, env, GATE, d=2))
        assert abs(E - E_eager) < 1e-9, (chi, E, E_eager)


def test_trial_energy_is_inf_without_a_fixed_point():
    A, C, T, layout = _cold("d2_off")
    gate = GATE.todense()
    no_budget = dataclasses.replace(CTM6, max_iter=1, conv_tol=1e-300)
    assert opt._trial_energy(A, (C, T), layout, no_budget, gate, 2) == float("inf")
    assert np.isfinite(opt._trial_energy(A, (C, T), layout, CTM6, gate, 2))


@pytest.mark.parametrize("method", ["armijo", "hager_zhang"])
def test_an_unconverged_trial_is_a_bad_point(monkeypatch, method):
    """Review Focus 3: a trial with no fixed point is +inf; the run spends its
    stall budget instead of raising."""
    monkeypatch.setattr(opt, "_trial_energy", lambda *args: float("inf"))
    config = _config(
        gs_line_search_method=method, gs_stall_recovery_retries=1, return_history=True
    )
    _A, _env, E, hist = opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
    assert hist["num_steps"] == 2  # two failed searches spend one reset
    assert abs(hist["energies"][1] - hist["energies"][0]) < 1e-9
    assert E == min(hist["energies"])


@pytest.mark.parametrize(("retries", "raises"), [(1, False), (0, True)])
def test_reset_on_an_unconverged_forward(monkeypatch, retries, raises):
    """A forward with no fixed point at step 2 resets to the best point; with
    no retry left it re-raises (the 1-site rule, #1059).  Armijo trials do not
    call charged_c4v_converge, so call 2 is the second step's evaluation."""
    real, calls = ad.charged_c4v_converge, [0]

    def flaky(*args):
        calls[0] += 1
        if calls[0] == 2:
            info = CTMConvergeInfo(
                converged=False,
                iterations=1,
                sv_diff=1.0,
                best_iteration=1,
                step_multiplier=float("nan"),
            )
            raise CTMNotConvergedError(
                info, site="gradient", conv_tol=1e-10, chi=6, detail="mock"
            )
        return real(*args)

    monkeypatch.setattr(ad, "charged_c4v_converge", flaky)
    config = _config(gs_stall_recovery_retries=retries, return_history=True)
    if raises:
        with pytest.raises(CTMNotConvergedError):
            opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
        return
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(GATE, _site("d2_off"), config)
    assert calls[0] >= 3
    # Step 3 evaluates the best point again: the step-1 point.
    assert abs(hist["energies"][1] - hist["energies"][0]) < 1e-9


def test_layout_mismatch_is_recorded(monkeypatch):
    """A free ranking that keeps another layout at step 2 is recorded once;
    the run keeps the frozen layout."""
    real, calls = opt._c4v_staggered_layout, [0]
    other = BondLayout.from_dims({0: 6})

    def ranking(Qf, chi):
        calls[0] += 1
        return other if calls[0] == 2 else real(Qf, chi)

    monkeypatch.setattr(opt, "_c4v_staggered_layout", ranking)
    _A, _env, _E, hist = opt.optimize_gs_ad_c4v_charged(
        GATE, _site("d2_off"), _config(return_history=True)
    )
    assert calls[0] >= 2, "vacuous: the second check did not run"
    assert hist["layout_mismatch_steps"] == [2]
    assert hist["layout"] == BondLayout.from_dims({-1: 2, 0: 2, 1: 2}).dims
    assert set(hist) == {
        "energies",
        "grad_norms",
        "step_times",
        "num_steps",
        "converged",
        "ctm_converged",
        "forward_sweeps",
        "adjoint_matvecs",
        "adjoint_residual",
        "layout",
        "layout_mismatch_steps",
    }
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q > $S/t6-red.log 2>&1; tail -n 5 $S/t6-red.log`
Expected: `ERROR tests/test_c4v_charged_ad.py` with `ModuleNotFoundError: No module named 'tenax.algorithms.ipeps_optimize_c4v_charged'`.

- [ ] **Step 3: Write the loop module**

Create `src/tenax/algorithms/ipeps_optimize_c4v_charged.py`:

```python
"""``optimize_gs_ad`` for a charged C4v ``SymmetricTensor`` (spec 2, section 5).

Design: ``docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md``.

A charged C4v site tensor A (a virtual charge q with q != dual(q)) is the
2-sublattice state (A, negate(A)) (#1211).  This loop optimises it with
L-BFGS and the line searches of the 1x1 loop.  The gradient comes from
``charged_c4v_converge``: implicit AD through the #1211 sweep at the layout
of the cold start, which stays frozen for the run.  The parameter x is A's
flat block buffer; the site is A(x) = P x / |P x|, with P the C4v
projection of the buffer.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time

import jax
import jax.numpy as jnp

from tenax.algorithms import _ctm_tensor_c4v_charged_ad as _engine
from tenax.algorithms._ctm_convergence_policy import CTMNotConvergedError
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_layout,
    _c4v_staggered_to_full_env,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ipeps_optimize_shared import (
    _converged_outer,
    _euclidean_grads,
    _grad_l2_norm,
    _log_ad_converged,
    _normalize_params,
    _should_accept_best,
)
from tenax.algorithms._line_search import LineSearchAborted, hager_zhang_line_search
from tenax.algorithms._metric_precond import lbfgs_two_loop
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.tensor import SymmetricTensor, Tensor

__all__ = [
    "_c4v_projector_matrix",
    "_check_charged_c4v_config",
    "_free_layout",
    "_trial_energy",
    "optimize_gs_ad_c4v_charged",
]

_logger = logging.getLogger(__name__)
_BACKEND = "1site-c4v-charged"
_LBFGS_MEMORY = 10  # curvature pairs kept, as in the 1x1 loop


def _check_charged_c4v_config(config, ctm, A_init) -> None:
    """Refuse each setting this loop cannot honour or would ignore.

    The table of spec 2, section 5, and ruling 8 of the plan.  ``ctm`` is
    ``config.ctm`` after the ``gs_projector_method`` override.

    Raises:
        ValueError: naming the first refused setting and the reason.
    """

    def refuse(setting: str, reason: str) -> None:
        raise ValueError(f"charged C4v AD: {setting} is not supported: {reason}")

    if not config.gs_implicit_ad:
        refuse("gs_implicit_ad=False", "there is no explicit-AD path for it")
    if jnp.iscomplexobj(A_init._data):
        refuse("complex data", "the mirror pairing holds for real data only (#1211)")
    if ctm.projector_method not in ("svd", "eigh"):
        refuse(
            f"projector_method={ctm.projector_method!r}",
            "'svd' and 'eigh' both select the #1211 block projector",
        )
    if ctm.forward_gauge not in ("auto", "bond_phase"):
        refuse(
            f"forward_gauge={ctm.forward_gauge!r}",
            "the loop aligns one sign per chi index itself",
        )
    if ctm.ctm_conv_method != "elementwise":
        refuse(
            f"ctm_conv_method={ctm.ctm_conv_method!r}",
            "the adjoint needs an element-wise fixed point",
        )
    if ctm.projector_backward not in ("auto", "flow"):
        refuse(
            f"projector_backward={ctm.projector_backward!r}",
            "the gradient always includes the projector response",
        )
    if ctm.on_unconverged != "raise":
        refuse(
            f"on_unconverged={ctm.on_unconverged!r}",
            "without a fixed point there is no gradient",
        )
    for name in ("chi_ramp", "chi_auto_bump", "ctmrg_heuristic_increase_chi"):
        if getattr(ctm, name):
            refuse(name, "the layout is frozen for the run")
    if not ctm.fuse_virtual_legs:
        refuse("fuse_virtual_legs=False", "the split CTM is dense-only")
    if ctm.ctm_mixing > 0:
        refuse(
            f"ctm_mixing={ctm.ctm_mixing}",
            "a mixed sweep is not the map whose fixed point the adjoint solves",
        )
    if ctm.device_mesh is not None:
        refuse("device_mesh", "the charged sweep runs on one device")
    if config.gs_stall_recovery == "noise":
        refuse("gs_stall_recovery='noise'", "the noise kick assumes dense params")
    if config.cg_gates is not None:
        refuse("cg_gates", "the charged branch has no coarse-grained energy")
    if config.gs_optimizer.lower() != "lbfgs":
        refuse(f"gs_optimizer={config.gs_optimizer!r}", "the loop runs L-BFGS only")
    if config.gs_line_search is False:
        refuse("gs_line_search=False", "the L-BFGS step needs a line search")
    for name in (
        "gs_chi_schedule_steps",
        "gs_ctm_conv_tol_schedule",
        "gs_ctm_max_iter_schedule",
        "gs_plateau_patience_schedule",
    ):
        if getattr(config, name):
            refuse(name, "the loop runs one CTMConfig for the whole run")


def _c4v_projector_matrix(A) -> jax.Array:
    """The C4v projection of A's flat block buffer, as an n x n matrix P.

    P is the Jacobian of the linear map
    x -> buffer(from_dense(symmetrize_c4v(todense(A(x))))).  It is symmetric
    and idempotent.  A must hold every block that its charges allow, or P
    is not square.
    """
    treedef = jax.tree_util.tree_structure(A)

    def project(x):
        dense = jax.tree_util.tree_unflatten(treedef, [x]).todense()
        return SymmetricTensor.from_dense(symmetrize_c4v(dense), A.indices)._data

    return jax.jacfwd(project)(A._data)


def _trial_energy(A, warm, layout, ctm, gate, d) -> float:
    """The energy at a line-search trial: the element-wise phase from ``warm``.

    Returns +inf when the phase finds no fixed point or the energy is not
    finite.  A trial is a bad point, never an error (Review Focus 3).
    """
    try:
        C, T = _engine._elementwise_phase(A, warm, layout, ctm, {})
    except CTMNotConvergedError:
        return float("inf")
    env = _c4v_staggered_to_full_env(C, T)
    E = float(compute_energy_ctm_tensor(A, env, gate, d=d))
    return E if math.isfinite(E) else float("inf")


def _free_layout(A, C, T, chi):
    """The layout that a free ranking keeps at e* = (C, T); None if it refuses."""
    Qf = _c4v_staggered_enlarged_corner(C, T, _build_double_layer_tensor(A))
    try:
        return _c4v_staggered_layout(Qf, chi)
    except ValueError:
        return None


def optimize_gs_ad_c4v_charged(hamiltonian_gate, A_init, config):
    """``optimize_gs_ad`` on a charged C4v ``SymmetricTensor``.

    Args:
        hamiltonian_gate: the 2-site gate, shape ``(d, d, d, d)``.
        A_init: a charged C4v ``SymmetricTensor``.  It need not be C4v or
            hold every block: the loop rebuilds it with every valid block
            and projects it to C4v.
        config: an ``iPEPSConfig`` that ``_check_charged_c4v_config`` accepts.

    Returns:
        ``(A, env, E)``, and a ``history`` dict last when
        ``config.return_history``.  ``env`` is
        ``_c4v_staggered_to_full_env(C*, T*)``, the env that
        ``ctm_tensor_c4v`` returns for charged input.

    Raises:
        ValueError: a refused setting, or an ``A_init`` with no C4v part.
        CTMNotConvergedError: a forward with no fixed point and no reset left.
        RuntimeError: a failed adjoint solve at an evaluated point.
    """
    from tenax.algorithms.ipeps_optimize import (
        _backtracking_line_search,
        _is_real_decrease,
        _log_ad_step,
        _log_ctm_not_converged_reset,
        _normalize_stall_recovery,
        _resolve_line_search_method,
    )

    ctm = config.ctm
    if config.gs_projector_method is not None:
        ctm = dataclasses.replace(ctm, projector_method=config.gs_projector_method)
    config = _normalize_stall_recovery(config, unit_cell="1x1")
    _check_charged_c4v_config(config, ctm, A_init)
    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.asarray(hamiltonian_gate)
    )
    d = gate.shape[0]

    # Rebuild with every valid block, so P is square (Review Focus 4).
    A0 = SymmetricTensor.from_dense(A_init.todense(), A_init.indices)
    treedef = jax.tree_util.tree_structure(A0)
    P = _c4v_projector_matrix(A0)
    x = P @ A0._data
    x_norm = float(jnp.linalg.norm(x))
    if x_norm == 0.0:
        raise ValueError("charged C4v AD: A_init has no C4v component")
    x = x / x_norm

    def site(x_):
        y = P @ x_
        return jax.tree_util.tree_unflatten(treedef, [y / jnp.linalg.norm(y)])

    record: dict = {}
    C, T, layout = _engine.charged_c4v_cold_start(site(x), ctm)
    warm = _engine._elementwise_phase(site(x), (C, T), layout, ctm, record)

    def loss(x_, warm_):
        A_ = site(x_)
        C_, T_ = _engine.charged_c4v_converge(A_, warm_, layout, ctm, record)
        env_ = _c4v_staggered_to_full_env(C_, T_)
        return compute_energy_ctm_tensor(A_, env_, gate, d=d), (C_, T_)

    value_and_grad = jax.value_and_grad(loss, has_aux=True)

    per_step = (
        "energies",
        "grad_norms",
        "step_times",
        "forward_sweeps",
        "adjoint_matvecs",
        "adjoint_residual",
    )
    hist: dict = {key: [] for key in per_step}
    mismatch_steps: list[int] = []
    state = {"num_steps": 0, "converged": False}

    def finish(x_, warm_, E_):
        A_ = site(x_)
        env_ = _c4v_staggered_to_full_env(*warm_)
        if not config.return_history:
            return A_, env_, E_
        history = {
            **hist,
            **state,
            "ctm_converged": True,  # every unconverged forward raised
            "layout": layout.dims,
            "layout_mismatch_steps": mismatch_steps,
        }
        return A_, env_, E_, history

    if config.gs_num_steps == 0:
        env0 = _c4v_staggered_to_full_env(*warm)
        E0 = float(compute_energy_ctm_tensor(site(x), env0, gate, d=d))
        return finish(x, warm, E0)

    line_search = _resolve_line_search_method(config, ctm)
    retries = config.gs_stall_recovery_retries
    best_E, best_x, best_warm = float("inf"), x, warm
    prev_E = float("inf")
    lbfgs_history: list = []
    prev_x = prev_g = None
    stall = 0

    for step in range(config.gs_num_steps):
        t0 = time.perf_counter()
        state["num_steps"] = step + 1
        try:
            (E_val, e_star), g = value_and_grad(x, warm)
        except CTMNotConvergedError as exc:
            # The 1-site rule (#1059): reset to the best point within the
            # retry budget; re-raise when the failing point is the best one,
            # no best exists yet, or no retry is left.
            exc.step = step + 1
            if x is best_x or best_E == float("inf") or stall + 1 > retries:
                raise
            _log_ctm_not_converged_reset(
                exc,
                conv_tol=ctm.conv_tol,
                n_reset=stall + 1,
                retries=retries,
                verbose=config.gs_verbose,
            )
            stall += 1
            x, warm = best_x, best_warm
            lbfgs_history.clear()
            prev_x = prev_g = None
            prev_E = float("inf")
            continue
        warm = e_star

        # One free ranking per evaluation (spec 2, section 5, step 5).
        free = _free_layout(site(x), *warm, ctm.chi)
        if free != layout:
            if not mismatch_steps:
                _logger.warning(
                    "[iPEPS-AD:%s] step %d: a free ranking at e* keeps %s, not "
                    "the frozen layout %s; the run keeps the frozen layout "
                    "(logged once)",
                    _BACKEND,
                    step + 1,
                    None if free is None else free.dims,
                    layout.dims,
                )
            mismatch_steps.append(step + 1)

        n_nonfinite = int(jnp.sum(~jnp.isfinite(g)))
        g = _euclidean_grads(jnp.where(jnp.isfinite(g), g, 0.0))
        E = float(E_val)
        grad_norm = _grad_l2_norm(g)
        hist["energies"].append(E)
        hist["grad_norms"].append(grad_norm)
        for key in ("forward_sweeps", "adjoint_matvecs", "adjoint_residual"):
            hist[key].append(record.get(key))
        if _should_accept_best(
            current_best=best_E, candidate=E, floor=config.gs_energy_floor
        ):
            best_E, best_x, best_warm = E, x, warm
        delta = abs(E - prev_E)
        prev_E = E
        if config.gs_verbose:
            _log_ad_step(
                _BACKEND,
                step,
                config.gs_num_steps,
                E,
                delta,
                best_E,
                grad_norm=grad_norm,
            )
        if n_nonfinite:
            prev_E = float("inf")  # a masked gradient must not pass the exit test
        elif _converged_outer(config, delta, grad_norm):
            state["converged"] = True
            if config.gs_verbose:
                _log_ad_converged(
                    _BACKEND,
                    step,
                    delta,
                    config.gs_conv_tol,
                    grad_norm=grad_norm,
                    grad_norm_tol=config.gs_grad_norm_tol,
                    criterion=config.gs_conv_criterion,
                )
            hist["step_times"].append(time.perf_counter() - t0)
            break

        # The L-BFGS direction, identity H0, as on the 1x1 C4v path.
        if prev_x is not None:
            s, y = x - prev_x, g - prev_g
            sy = float(jnp.vdot(s, y))
            if sy > 1e-10:
                lbfgs_history.append((s, y, 1.0 / sy))
                if len(lbfgs_history) > _LBFGS_MEMORY:
                    lbfgs_history.pop(0)
        prev_x, prev_g = x, g
        direction = -lbfgs_two_loop(g, lbfgs_history, lambda v: v)
        slope = float(jnp.vdot(g, direction))
        if slope >= 0:
            direction, slope = -g, -float(jnp.vdot(g, g))
            lbfgs_history.clear()

        def trial_energy(x_trial):
            return _trial_energy(site(x_trial), warm, layout, ctm, gate, d)

        if line_search == "hager_zhang":

            def phi(alpha):
                return trial_energy(_normalize_params(x + alpha * direction))

            def dphi(alpha):
                trial = _normalize_params(x + alpha * direction)
                try:
                    _value, g_trial = value_and_grad(trial, warm)
                except CTMNotConvergedError as exc:  # first: a RuntimeError
                    raise LineSearchAborted from exc
                except RuntimeError as exc:  # HZ makes it a bad point
                    _logger.warning(
                        "[iPEPS-AD:%s] the adjoint solve failed at a "
                        "line-search trial (alpha=%.3e): %s",
                        _BACKEND,
                        alpha,
                        exc,
                    )
                    raise
                return float(jnp.vdot(_euclidean_grads(g_trial), direction))

            x_scale = float(jnp.linalg.norm(x))
            d_scale = max(float(jnp.linalg.norm(direction)), 1e-15)
            alpha0 = min(1.0, 0.1 * x_scale / d_scale)
            alpha, E_new, _wolfe = hager_zhang_line_search(
                phi,
                dphi,
                E,
                slope,
                alpha_init=alpha0,
                rho=1.5,
                max_step=2.0 * alpha0,
                energy_bound=max(2.0, 2.0 * abs(best_E)),
                max_iter=config.gs_hz_max_iter,
                bracket_only_phi=True,
            )
            x_new = _normalize_params(x + alpha * direction)
        else:
            x_new, E_new, _alpha = _backtracking_line_search(
                x,
                direction,
                g,
                E,
                trial_energy,
                max_steps=config.gs_line_search_max_steps,
            )

        if _is_real_decrease(E_new, E):
            x, stall = x_new, 0
        else:
            stall += 1
            if stall > retries:
                if config.gs_verbose:
                    print(
                        f"[iPEPS-AD:{_BACKEND}] stall budget exhausted after "
                        f"{stall - 1} resets, returning best E={best_E:.10f}",
                        flush=True,
                    )
                hist["step_times"].append(time.perf_counter() - t0)
                break
            x, warm = best_x, best_warm
            lbfgs_history.clear()
            prev_x = prev_g = None
            prev_E = float("inf")
        hist["step_times"].append(time.perf_counter() - t0)

    if best_E == float("inf"):  # gs_energy_floor refused every energy
        A_last = site(x)
        e_last = _engine._elementwise_phase(A_last, warm, layout, ctm, {})
        env_last = _c4v_staggered_to_full_env(*e_last)
        E_last = float(compute_energy_ctm_tensor(A_last, env_last, gate, d=d))
        return finish(x, e_last, E_last)
    return finish(best_x, best_warm, best_E)
```

- [ ] **Step 4: Run the tests to see them pass, and time the core file**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q --durations=10 > $S/t6-green.log 2>&1; tail -n 16 $S/t6-green.log`
Expected: `54 passed` (21 from Tasks 3-5, 33 new). If `_config(...)` itself raises for a refusal case (an `iPEPSConfig` cross-check), use a value that constructs, and ledger the change. Read the total time and the ten slowest tests. If the file takes 120 s or more (the "Test markers" point), move the slowest loop tests to `tests/test_c4v_charged_ad_physics.py` until the core file is under 120 s, and ledger `Task 6: Ruling: moved <tests> to the algorithm file — the core file took <t> s — they run only with run-full-tests`.

- [ ] **Step 5: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t6.txt <<'EOF'
feat(ipeps): optimiser loop for a charged C4v SymmetricTensor

optimize_gs_ad_c4v_charged projects the start to C4v (every valid block,
P by jacfwd), freezes the layout of the cold start, and runs L-BFGS with
the 1x1 Hager-Zhang or Armijo line search on the flat block buffer.  The
gradient comes from charged_c4v_converge.  An unconverged forward resets
to the best point within gs_stall_recovery_retries; a trial without a
fixed point is a bad point; a free ranking after each evaluation records
any layout mismatch.  Every setting the loop cannot honour is refused.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/ipeps_optimize_c4v_charged.py tests/test_c4v_charged_ad.py
git commit -F $S/msg-ad-t6.txt > $S/commit-ad-t6.log 2>&1; tail -n 3 $S/commit-ad-t6.log
```

Expected: one commit; the hooks pass.

---

### Task 7: Dispatch from `optimize_gs_ad`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py` (a new gate after `use_root_implicit_path`, line ~176)
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (the import block at lines 41-46; the docstring at ~1438-1440; the projector-backward skip at ~1573; the checkpoint refusal at ~1583-1592; a new helper above `def _reject_mislabelled_1x1(`; the branches at ~1631-1636)
- Test: `tests/test_c4v_charged_ad.py`

**Interfaces:**
- Consumes: `_is_charged_c4v(A) -> bool` (`tenax.algorithms._ctm_c4v_charges`; False for anything that is not a `SymmetricTensor`, and for self-dual or fermionic input); `optimize_gs_ad_c4v_charged` (Task 6).
- Produces: `use_charged_c4v_path(config, A_init) -> bool` in `tenax.algorithms.ipeps_ad_policy`; `_refuse_charged_c4v_2site(A_init, config) -> None` in `tenax.algorithms.ipeps_optimize`. The charged branch imports `optimize_gs_ad_c4v_charged` lazily, so a test's monkeypatch of `opt.optimize_gs_ad_c4v_charged` reaches it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_c4v_charged_ad.py`, add to the imports:

```python
from tenax import optimize_gs_ad
from tenax.algorithms.ipeps_ad_policy import use_charged_c4v_path
from tenax.core.tensor import DenseTensor
```

(merge `optimize_gs_ad` into the `from tenax import` line and `DenseTensor` into the `tenax.core.tensor` line). Append:

```python
# ------------------------------------------------------------------ #
# Dispatch (spec 2, section 5)                                        #
# ------------------------------------------------------------------ #


def test_dispatch_reaches_the_charged_loop(monkeypatch):
    """Charged input reaches the new branch and is not densified."""
    seen = {}

    def fake(gate, A_init, config):
        seen["A"] = A_init
        return "charged"

    monkeypatch.setattr(opt, "optimize_gs_ad_c4v_charged", fake)
    A = _site("d2_off")
    assert optimize_gs_ad(GATE, A, _config()) == "charged"
    assert seen["A"] is A


@pytest.mark.parametrize(
    ("kw", "ctm", "expected"),
    [
        ({}, None, True),
        ({"unit_cell": "2site"}, None, False),
        ({"gs_c4v": False}, None, False),
        ({}, {"ctm_ad_mode": "c4v_reference"}, False),
        ({}, {"ctm_ad_mode": "root_implicit_symmetric"}, False),
    ],
)
def test_charged_gate_reads_the_config(kw, ctm, expected):
    assert use_charged_c4v_path(_config(ctm, **kw), _site("d2_off")) is expected


def test_charged_gate_reads_the_input():
    """Only a charged C4v SymmetricTensor takes the branch."""
    config, A = _config(), _site("d2_off")
    assert not use_charged_c4v_path(config, DenseTensor(A.todense(), A.indices))
    assert not use_charged_c4v_path(config, None)
    self_dual = c4v_fixture(ZnSymmetry(2), [0, 1], [0, 1])
    assert not use_charged_c4v_path(config, self_dual)


@pytest.mark.parametrize("pair", [False, True])
def test_two_site_charged_c4v_is_refused(pair):
    A = _site("d2_off")
    with pytest.raises(ValueError, match="charged C4v input with unit_cell='2site'"):
        optimize_gs_ad(GATE, (A, A) if pair else A, _config(unit_cell="2site"))


def test_checkpoint_is_refused_on_the_charged_path(tmp_path):
    config = _config(gs_checkpoint_path=str(tmp_path / "run.npz"))
    with pytest.raises(NotImplementedError, match="charged C4v path"):
        optimize_gs_ad(GATE, _site("d2_off"), config)


def test_dispatch_skips_the_fused_projector_check():
    """Ruling 16: the fused-path check refuses "eigh"; the charged loop takes it."""
    config = _config({"projector_method": "eigh"}, gs_num_steps=0)
    assert np.isfinite(optimize_gs_ad(GATE, _site("d2_off"), config)[2])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q > $S/t7-red.log 2>&1; tail -n 5 $S/t7-red.log`
Expected: `ERROR tests/test_c4v_charged_ad.py` with `ImportError: cannot import name 'use_charged_c4v_path' from 'tenax.algorithms.ipeps_ad_policy'`.

- [ ] **Step 3: Add the gate**

In `src/tenax/algorithms/ipeps_ad_policy.py`, insert after the `use_root_implicit_path` function:

```python
def use_charged_c4v_path(config: iPEPSConfig, A_init) -> bool:
    """Return ``True`` when ``A_init`` takes the charged C4v loop (spec 2).

    The gate: ``unit_cell="1x1"``, ``gs_c4v=True``, no ``ctm_ad_mode``, and a
    charged C4v ``SymmetricTensor`` (a virtual charge q with
    ``q != dual(q)``, see ``_is_charged_c4v``).  It reads index metadata
    only.
    """
    from tenax.algorithms._ctm_c4v_charges import _is_charged_c4v

    return (
        config.unit_cell == "1x1"
        and config.gs_c4v
        and getattr(config.ctm, "ctm_ad_mode", None) is None
        and _is_charged_c4v(A_init)
    )
```

- [ ] **Step 4: Wire the dispatch**

In `src/tenax/algorithms/ipeps_optimize.py`:

(a) The import block (lines 41-46) becomes:

```python
from tenax.algorithms.ipeps_ad_policy import (
    build_ad_ctm_config,
    resolve_projector_backward,
    use_charged_c4v_path,
    use_reference_c4v_path,
    use_root_implicit_path,
)
```

(b) Replace

```python
    if not (use_root_implicit_path(config) or _use_reference_c4v_path(config)):
        config = _resolve_projector_backward(config)
```

with

```python
    # The charged C4v loop checks its own settings (spec 2) and accepts the
    # "eigh" projector that this check refuses.
    if not (
        use_root_implicit_path(config)
        or _use_reference_c4v_path(config)
        or use_charged_c4v_path(config, A_init)
    ):
        config = _resolve_projector_backward(config)
```

(c) Replace

```python
    if config.gs_checkpoint_path is not None and (
        isinstance(config.unit_cell, Lattice) or _use_reference_c4v_path(config)
    ):
        raise NotImplementedError(
            "iPEPSConfig.gs_checkpoint_path is not yet supported for generic "
            "Lattice multisite unit cells or the dense C4v-reference path; use "
            "unit_cell='2site' or '1x1' (non-C4v-reference, incl. cg_gates). "
            "Multisite / C4v-reference checkpoint wiring is a follow-up (#497)."
        )
```

with

```python
    if config.gs_checkpoint_path is not None and (
        isinstance(config.unit_cell, Lattice)
        or _use_reference_c4v_path(config)
        or use_charged_c4v_path(config, A_init)
    ):
        raise NotImplementedError(
            "iPEPSConfig.gs_checkpoint_path is not yet supported for generic "
            "Lattice multisite unit cells or the dense C4v-reference path; use "
            "unit_cell='2site' or '1x1' (non-C4v-reference, incl. cg_gates). "
            "Multisite / C4v-reference checkpoint wiring is a follow-up (#497). "
            "The charged C4v path (spec 2) is not wired either."
        )
```

The old text stays as it is, so an existing `match=` on it still holds (ruling 27).

(d) Replace

```python
    if config.unit_cell == "2site":
        return _optimize_gs_ad_2site(
            hamiltonian_gate, A_init, config, envs_init=envs_init
        )
    if _use_reference_c4v_path(config):
        return _optimize_gs_ad_tensor_reference_c4v(hamiltonian_gate, A_init, config)
```

with

```python
    if config.unit_cell == "2site":
        _refuse_charged_c4v_2site(A_init, config)
        return _optimize_gs_ad_2site(
            hamiltonian_gate, A_init, config, envs_init=envs_init
        )
    if _use_reference_c4v_path(config):
        return _optimize_gs_ad_tensor_reference_c4v(hamiltonian_gate, A_init, config)
    if use_charged_c4v_path(config, A_init):
        # A charged C4v SymmetricTensor (spec 2): implicit AD through the #1211
        # 2-sublattice sweep on the block data, with no densification.
        from tenax.algorithms.ipeps_optimize_c4v_charged import (
            optimize_gs_ad_c4v_charged,
        )

        return optimize_gs_ad_c4v_charged(hamiltonian_gate, A_init, config)
```

(e) Insert directly above the line `def _reject_mislabelled_1x1(`:

```python
def _refuse_charged_c4v_2site(A_init, config) -> None:
    """Refuse a charged C4v site on the 2-site C4v path (spec 2, section 5).

    The 1x1 charged C4v ansatz is already the 2-sublattice state
    (A, negate(A)) (#1211).  With ``gs_c4v=False`` the general symmetric
    2-site path runs as before.
    """
    if not config.gs_c4v:
        return
    from tenax.algorithms._ctm_c4v_charges import _is_charged_c4v

    sites = A_init if isinstance(A_init, tuple) else (A_init,)
    if any(_is_charged_c4v(site) for site in sites):
        raise ValueError(
            "charged C4v input with unit_cell='2site' and gs_c4v=True is not "
            "supported: the 1x1 charged C4v ansatz is already the 2-sublattice "
            "state (A, negate(A)); use unit_cell='1x1'"
        )


```

(f) In the `optimize_gs_ad` docstring, replace

```python
    Accepts dense
    ``jax.Array`` or Tensor-protocol objects (``DenseTensor``,
    ``SymmetricTensor``).

    Args:
```

with

```python
    Accepts dense
    ``jax.Array`` or Tensor-protocol objects (``DenseTensor``,
    ``SymmetricTensor``).

    A charged C4v ``SymmetricTensor`` (a virtual charge q with
    ``q != dual(q)``, which ``ctm_tensor_c4v`` runs as the 2-sublattice
    state (A, negate(A)), #1211) with ``unit_cell="1x1"``, ``gs_c4v=True``
    and no ``ctm_ad_mode`` takes its own loop: implicit AD through the #1211
    sweep at the layout of its cold start, on the block data, with L-BFGS
    and the configured line search.  That loop refuses
    ``gs_implicit_ad=False``, complex data, ``projector_method="qr"``, a
    ``forward_gauge`` other than ``"auto"`` or ``"bond_phase"``,
    ``ctm_conv_method="sv"``, ``projector_backward`` ``"standard"`` or
    ``"lorentzian"``, ``on_unconverged="warn"``, ``chi_ramp``,
    ``chi_auto_bump``, ``ctmrg_heuristic_increase_chi``, the
    ``gs_chi_schedule_steps`` and ``gs_ctm_*`` / ``gs_plateau_*``
    schedules, ``gs_stall_recovery="noise"``, ``gs_checkpoint_path``,
    ``cg_gates``, ``fuse_virtual_legs=False``, a ``gs_optimizer`` other
    than ``"lbfgs"``, ``gs_line_search=False``, ``ctm_mixing`` and
    ``device_mesh``.  It does not apply ``gs_metric_precond``, and it does
    not read ``adjoint_solver``, ``adjoint_method`` or
    ``adjoint_arnoldi_precheck``.  A charged C4v site with
    ``unit_cell="2site"`` and ``gs_c4v=True`` is refused.  See Path 4b in
    ``docs/guide/algorithms/ipeps_ad_paths.md``.

    Args:
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad.py -q > $S/t7-green.log 2>&1; tail -n 3 $S/t7-green.log`
Expected: `65 passed` (54 + 11).

Then run the existing tests that touch the edited dispatch lines: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && F=$(command grep -l "use_reference_c4v_path\|gs_checkpoint_path\|_resolve_projector_backward\|ipeps_ad_policy" tests/*.py | tr '\n' ' '); echo $F; uptime; JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider -m core $F -q > $S/t7-dispatch.log 2>&1; tail -n 3 $S/t7-dispatch.log`
Expected: the file list prints; `passed` with no failure (if the list is empty, ledger that and go on).

- [ ] **Step 6: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t7.txt <<'EOF'
feat(ipeps): route charged C4v input to its own AD loop

use_charged_c4v_path (unit_cell="1x1", gs_c4v=True, no ctm_ad_mode, a
charged C4v SymmetricTensor) sends optimize_gs_ad to the charged loop
without densifying.  The fused projector check is skipped for it, the
checkpoint refusal covers it, and a charged site on the 2-site C4v path
is refused: the 1x1 charged ansatz is already the 2-sublattice state.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add src/tenax/algorithms/ipeps_ad_policy.py src/tenax/algorithms/ipeps_optimize.py tests/test_c4v_charged_ad.py
git commit -F $S/msg-ad-t7.txt > $S/commit-ad-t7.log 2>&1; tail -n 3 $S/commit-ad-t7.log
```

Expected: one commit; the hooks pass.

---

### Task 8: The D=2 optimum and the dense parity

**Files:**
- Test: `tests/test_c4v_charged_ad_physics.py`

**Interfaces:**
- Consumes: `optimize_gs_ad` with the Task 7 dispatch; `charged_d2`, `T_EXACT`, `c4v_fixture` (Task 3); `E_D2 = -0.66023109348` (`tests/_ipeps_spectrum_fixtures.py`); `ctm_tensor_c4v(A, chi, max_iter, conv_tol)`; `ctm_tensor(A, chi=, max_iter=, conv_tol=) -> (env, info)` (`tenax.algorithms._ctm_tensor_convergence`).
- Produces: `test_d2_optimum[d2_off]`, `test_d2_optimum[random]`, `test_dense_parity`; the printed lines that Task 10 copies into the guide.

These tests pass when written. Their RED step is Step 3: the "loss skips the C4v projection" mutant (ruling 3).

- [ ] **Step 1: Write the tests**

In `tests/test_c4v_charged_ad_physics.py`, add `import numpy as np` and:

```python
from _ipeps_spectrum_fixtures import E_D2

from tenax import iPEPSConfig, optimize_gs_ad
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.tensor import DenseTensor
```

(merge into the existing import lines). Append:

```python
# ------------------------------------------------------------------ #
# The D=2 optimum and the dense parity (spec 2, section 6)            #
# ------------------------------------------------------------------ #

CHI = 16


def _eager_energy(A, chi=CHI):
    """The #1211 eager energy of A: ctm_tensor_c4v to sv_diff < 1e-12."""
    env = ctm_tensor_c4v(A, chi, max_iter=500, conv_tol=1e-12)
    return float(compute_energy_ctm_tensor(A, env, GATE, d=2))


def _run_config(conv_tol=1e-10, **kw):
    fields = {
        "max_bond_dim": 2,
        "gs_c4v": True,
        "su_init": False,
        "gs_num_steps": 40,
        "return_history": True,
        "ctm": CTMConfig(chi=CHI, max_iter=500, conv_tol=conv_tol),
    }
    return iPEPSConfig(**{**fields, **kw})


_STARTS = {
    "d2_off": lambda: charged_d2(1.15 * T_EXACT),
    # In the same one-parameter family (t = 1.5491), but its cold start at
    # chi=16 lands on another fixed point than charged_d2(1.5491) does.
    "random": lambda: c4v_fixture(U1Symmetry(), [0, 1], [0, 1]),
}


@pytest.mark.parametrize("start", sorted(_STARTS))
def test_d2_optimum(start):
    """Catches a loss that skips the C4v projection (ruling 3).  The C4v
    assert comes first, so that mutant fails there."""
    A, _env, E, hist = optimize_gs_ad(GATE, _STARTS[start](), _run_config())
    dense = np.asarray(A.todense())
    sym = np.asarray(symmetrize_c4v(jnp.asarray(dense)))
    c4v_error = float(np.max(np.abs(sym - dense)))
    print(
        f"{start}: E={E:.12f} E-E_D2={E - E_D2:+.2e} "
        f"|g|={hist['grad_norms'][-1]:.2e} steps={hist['num_steps']} "
        f"c4v_error={c4v_error:.1e} mismatch={hist['layout_mismatch_steps']}"
    )
    assert c4v_error < 1e-12
    assert hist["converged"], hist["grad_norms"][-3:]
    assert hist["grad_norms"][-1] < 1e-5
    assert E <= _eager_energy(charged_d2(T_EXACT)) + 1e-9
    assert abs(E - E_D2) <= 1e-8


def test_dense_parity():
    """The dense CTM energy of todense(A*), and the dense C4v optimiser, agree
    with the charged path at chi=16 (spec 2, section 6)."""
    A0 = charged_d2(1.15 * T_EXACT)
    A, _env, E, _hist = optimize_gs_ad(GATE, A0, _run_config(conv_tol=1e-12))
    A_dense = DenseTensor(A.todense(), A.indices)
    env_dense, _info = ctm_tensor(A_dense, chi=CHI, max_iter=500, conv_tol=1e-12)
    E_dense = float(compute_energy_ctm_tensor(A_dense, env_dense, GATE, d=2))
    print(f"dense parity: |E - E_dense(ctm_tensor)| = {abs(E - E_dense):.2e}")
    assert abs(E - E_dense) <= 1e-10
    dense_run = _run_config(gs_num_steps=60, return_history=False)
    # Spec 2, section 6: the dense 1x1 C4v optimiser on DenseTensor(todense(A_init)).
    A0_dense = DenseTensor(A0.todense(), A0.indices)
    _A, _env_d, E_opt = optimize_gs_ad(GATE, A0_dense, dense_run)
    print(f"dense parity: E_dense_opt={E_opt:.12f} |E - E_dense_opt| = {abs(E - E_opt):.2e}")
    assert abs(E - E_opt) <= 1e-8
```

- [ ] **Step 2: Run the tests**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad_physics.py -q -s --durations=8 > $S/t8-green.log 2>&1`
Expected: `7 passed` (4 FD + 3 new), and the printed `d2_off:`, `random:` and `dense parity:` lines. Copy them into the ledger. If `test_d2_optimum[random]` ends on another fixed point (|E − E_D2| > 1e-8 with |g| < 1e-5), do not loosen the assert: ledger `Task 8: Ruling: the random start converges to E=<E> — another fixed point — reported to the user`, mark the test `pytest.mark.xfail(strict=True, reason="<the ledger line>")`, and report it in the final message.

- [ ] **Step 3: Watch the D=2 optimum test fail on its mutant**

Add to `MUTANTS` in `$S/mutate.py`:

```python
    "site_skips_P": (
        "src/tenax/algorithms/ipeps_optimize_c4v_charged.py",
        "        y = P @ x_\n",
        "        y = x_\n",
    ),
```

Run (background; read the logs): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax; taskset -c 0-11 uv run --no-sync python -u $S/mutate.py site_skips_P tests/test_c4v_charged_ad_physics.py -k d2_optimum > $S/t8-mut1.log 2>&1; taskset -c 0-11 uv run --no-sync python -u $S/mutate.py site_skips_P tests/test_c4v_charged_ad.py -k non_c4v_start > $S/t8-mut2.log 2>&1`
Expected: both logs end with `KILLED`, and the failing assertion is the C4v one (`c4v_error < 1e-12` in the physics file, `_c4v_error(A) < 1e-12` in the core file). If a log ends with `SURVIVED`, the mutant may be equivalent: measure |g − P g| / |g| at the d2_off start with a probe (`$S/probe_pg.py`: the raw gradient of E(A(x)) with `site` = x/|x|, against its C4v projection). Ledger `Task 8: Ruling: site_skips_P <KILLED|equivalent, |g − Pg|/|g| = <r>>`.

- [ ] **Step 4: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t8.txt <<'EOF'
test(ipeps): charged C4v AD reaches the D=2 U(1) optimum

From exact_u1_d2 moved off its optimum and from the random U(1) C4v
fixture, optimize_gs_ad at chi=16 ends with |g| < 1e-5 and
|E - (-0.66023109348)| <= 1e-8, on a C4v tensor.  The dense ctm_tensor
energy of the result agrees within 1e-10, and the dense C4v optimiser
reaches the same energy within 1e-8.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add tests/test_c4v_charged_ad_physics.py
git commit -F $S/msg-ad-t8.txt > $S/commit-ad-t8.log 2>&1; tail -n 3 $S/commit-ad-t8.log
```

Expected: one commit; the hooks pass.

---

### Task 9: The D=3 benchmark against Hasik's U(1)-C4v state

**Files:**
- Create: `benchmarks/ipeps_c4v_charged_ad.py`

**Interfaces:**
- Consumes: `load_state(path) -> np.ndarray` (A[u, d, l, r, s], normalised) and `load_charges(path) -> (phys, virt, total)` from `benchmarks/ipeps_spectrum_hasik.py` (`charges[:2]`, `charges[2:]`, `data["total_u1_charge"]`); the engine (Tasks 3-4); `optimize_gs_ad` (Task 7). The state file is `/home/yjkao/tenax-sma/hasik_states_u1/state_1s_A1_U1B_j20.0_D3_chi_opt72.json` (read only), u1_charges [1, −1, 0, 2, 0], total 1; its `.dat` has columns `j2 D chi_obs energy m l0..l5`, rows at chi_obs 36, 37, 45, 54, …
- Produces: the report lines and JSON files that Task 10 copies into the guide.

This task has no pytest. Its checks are the script's exit status and the measured numbers, read from the logs.

- [ ] **Step 1: Write the script**

Create `benchmarks/ipeps_c4v_charged_ad.py`:

```python
"""Charged C4v iPEPS AD at D=3 against Hasik's U(1)-C4v state (spec 2).

Outside CI (spec 2, section 6, "Benchmark").  ``--state`` is a local copy of
Hasik's D=3 state file; the ``.dat`` with the same stem holds his energies
by chi.

Modes:
    load    convert to a charged C4v SymmetricTensor and check it
    energy  the tenax energy at each --chi against the .dat value
    fd      the FD rule in three C4v directions at a perturbed state
    opt     optimise from a perturbed Hasik start and from a random start
    all     energy, fd and opt in turn

The exit status is 1 when the conversion loses weight or an FD check fails.
The optimisation targets are reported, not enforced.

    JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync python -u \\
        -m benchmarks.ipeps_c4v_charged_ad --state <path> --mode load
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ipeps_spectrum_hasik import load_charges, load_state
from tenax import (
    CTMConfig,
    heisenberg_gate,
    iPEPSConfig,
    optimize_gs_ad,
    sublattice_rotate_gate,
)
from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad
from tenax.algorithms._ctm_c4v_charges import _is_charged_c4v
from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_c4v_staggered import _c4v_staggered_to_full_env
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

GATE = sublattice_rotate_gate(heisenberg_gate())
FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
PERTURBATION = 0.05  # relative size of the C4v kick away from Hasik's optimum


def charged_state(path):
    """Hasik's state as a charged C4v SymmetricTensor, and the weight it loses.

    Hasik's tensor is nonzero where phys + sum(virtual) = total.  The tenax
    legs u, l (OUT, vc), d, r (IN, dual(vc)) and phys (IN, pc) conserve
    sum(virtual) = pc, so pc = total - phys.
    """
    A = np.asarray(load_state(path))
    if np.iscomplexobj(A):
        assert float(np.max(np.abs(A.imag))) < 1e-14, "the state is complex"
        A = A.real
    phys, virt, total = load_charges(path)
    sym = U1Symmetry()
    vc = np.asarray(virt, dtype=np.int32)
    pc = np.asarray([total - p for p in phys], dtype=np.int32)
    vd = sym.dual(vc)
    indices = (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc, FlowDirection.IN, label="phys"),
    )
    A_sym = SymmetricTensor.from_dense(jnp.asarray(A), indices)
    lost = float(np.max(np.abs(np.asarray(A_sym.todense()) - A)))
    return A_sym, lost, vc.tolist(), pc.tolist()


def eager_energy(A, chi):
    """The #1211 eager energy: ctm_tensor_c4v to sv_diff < 1e-12."""
    env = ctm_tensor_c4v(A, chi, max_iter=2000, conv_tol=1e-12)
    return float(compute_energy_ctm_tensor(A, env, GATE, d=2))


def dat_energies(path):
    """Hasik's energy by chi_obs, from the .dat next to the state file."""
    rows = {}
    for line in pathlib.Path(path).with_suffix(".dat").read_text().splitlines():
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            continue  # a header or a comment
        if len(values) >= 4:
            rows[int(round(values[2]))] = values[3]
    return rows


def with_data(X, data):
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(X), [data])


def c4v_direction(A, key):
    """A random unit direction in the C4v subspace of A's block buffer."""
    r = jax.random.normal(key, A._data.shape)
    dense = symmetrize_c4v(with_data(A, r).todense())
    v = SymmetricTensor.from_dense(dense, A.indices)._data
    return v / jnp.linalg.norm(v)


def perturbed(A, key, size=PERTURBATION):
    x = A._data + size * c4v_direction(A, key)
    return with_data(A, x / jnp.linalg.norm(x))


@contextlib.contextmanager
def frozen_projectors():
    """The "P frozen" gradient: stop_gradient on the traced projector."""
    real = ad._c4v_staggered_projectors_at

    def frozen(Qf, layout):
        return jax.tree_util.tree_map(jax.lax.stop_gradient, real(Qf, layout))

    jax.clear_caches()
    ad._c4v_staggered_projectors_at = frozen
    try:
        yield
    finally:
        ad._c4v_staggered_projectors_at = real
        jax.clear_caches()


def fd_check(A, chi, n_dirs=3):
    """The FD rule of spec 2, section 6, at A."""
    ctm = CTMConfig(
        chi=chi,
        max_iter=2000,
        conv_tol=1e-13,
        adjoint_tol=1e-12,
        adjoint_maxiter=200,
        adjoint_tikhonov=0.0,
    )
    C, T, layout = ad.charged_c4v_cold_start(A, dataclasses.replace(ctm, conv_tol=1e-10))
    warm = ad._elementwise_phase(A, (C, T), layout, ctm, {})
    record = {}

    def energy(A_):
        C_, T_ = ad.charged_c4v_converge(A_, warm, layout, ctm, record)
        return compute_energy_ctm_tensor(A_, _c4v_staggered_to_full_env(C_, T_), GATE, d=2)

    def phi(v, eps):
        A_e = with_data(A, A._data + eps * v)
        C_, T_ = ad._elementwise_phase(A_e, warm, layout, ctm, {})
        return float(
            compute_energy_ctm_tensor(A_e, _c4v_staggered_to_full_env(C_, T_), GATE, d=2)
        )

    t = time.perf_counter()
    g = jax.grad(energy)(A)
    first_grad_s = time.perf_counter() - t
    adjoint = dict(record)
    print(f"fd: first gradient {first_grad_s:.1f} s, layout {layout.dims}, {adjoint}", flush=True)
    with frozen_projectors():
        g_frozen = jax.grad(energy)(A)
    dirs = []
    for k in range(n_dirs):
        v = c4v_direction(A, jax.random.PRNGKey(k))
        slope = float(jnp.vdot(g._data, v))
        frozen = float(jnp.vdot(g_frozen._data, v))
        fds = [(phi(v, h) - phi(v, -h)) / (2.0 * h) for h in FD_STEPS]
        diffs = [abs(a - b) for a, b in zip(fds, fds[1:])]
        i = int(np.argmin(diffs))
        fd, fd_unc = 0.5 * (fds[i] + fds[i + 1]), diffs[i]
        row = {
            "dir": k,
            "g_dot_v": slope,
            "fd": fd,
            "fd_unc": fd_unc,
            "grad_err": abs(slope - fd),
            "flow_minus_frozen": abs(slope - frozen),
            "ok": abs(slope - fd) <= 3.0 * fd_unc + 1e-12,
            "regime": abs(slope) > 100.0 * fd_unc
            and abs(slope - frozen) > 10.0 * fd_unc,
        }
        print(
            f"fd dir {k}: |g.v|={abs(slope):.3e} fd_unc={fd_unc:.1e} "
            f"grad_err={row['grad_err']:.1e} "
            f"flow-frozen={row['flow_minus_frozen']:.1e} "
            f"ok={row['ok']} regime={row['regime']}",
            flush=True,
        )
        dirs.append(row)
    return {
        "chi": chi,
        "layout": layout.dims,
        "first_grad_s": first_grad_s,
        "adjoint": adjoint,
        "dirs": dirs,
    }


def optimise(A_start, chi, steps, label):
    config = iPEPSConfig(
        max_bond_dim=3,
        gs_c4v=True,
        su_init=False,
        gs_num_steps=steps,
        return_history=True,
        gs_verbose=True,
        ctm=CTMConfig(chi=chi, max_iter=2000, conv_tol=1e-10),
    )
    t = time.perf_counter()
    _A, _env, E, hist = optimize_gs_ad(GATE, A_start, config)
    wall_s = time.perf_counter() - t
    times = hist["step_times"]
    row = {
        "label": label,
        "E": E,
        "grad_norm": hist["grad_norms"][-1] if hist["grad_norms"] else None,
        "steps": hist["num_steps"],
        "converged": hist["converged"],
        "wall_s": wall_s,
        "first_step_s": times[0] if times else None,
        "median_later_step_s": float(np.median(times[1:])) if len(times) > 1 else None,
        "layout": hist["layout"],
        "layout_mismatch_steps": hist["layout_mismatch_steps"],
    }
    print(f"opt {label}: {row}", flush=True)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Charged C4v iPEPS AD at D=3")
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--chi", type=int, nargs="+", default=[36, 54, 72])
    parser.add_argument(
        "--mode", choices=("load", "energy", "fd", "opt", "all"), default="all"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)

    A, lost, vc, pc = charged_state(args.state)
    report = {
        "state": str(args.state),
        "vc": vc,
        "pc": pc,
        "entries": int(A._data.size),
        "lost": lost,
        "charged": _is_charged_c4v(A),
    }
    print(
        f"load: vc={vc} pc={pc} charged={report['charged']} "
        f"entries={report['entries']} max|dense - A|={lost:.1e}",
        flush=True,
    )
    ok = report["charged"] and lost < 1e-12
    modes = ("energy", "fd", "opt") if args.mode == "all" else (args.mode,)
    if ok and "energy" in modes:
        dat = dat_energies(args.state)
        report["energy"] = []
        for chi in args.chi:
            E = eager_energy(A, chi)
            ref = dat.get(chi)
            gap = None if ref is None else E - ref
            report["energy"].append(
                {"chi": chi, "E_tenax": E, "E_hasik": ref, "gap": gap}
            )
            print(f"energy chi={chi}: tenax {E:.13f} hasik {ref} gap {gap}", flush=True)
    if ok and "fd" in modes:
        report["fd"] = fd_check(perturbed(A, jax.random.PRNGKey(100)), args.chi[0])
        ok = all(row["ok"] for row in report["fd"]["dirs"])
    if ok and "opt" in modes:
        chi = args.chi[0]
        E_ref = eager_energy(A, chi)
        start = perturbed(A, jax.random.PRNGKey(101))
        random_start = SymmetricTensor.random_normal(A.indices, jax.random.PRNGKey(7))
        runs = [
            optimise(start, chi, args.steps, "perturbed Hasik"),
            optimise(random_start, chi, args.steps, "random"),
        ]
        for row in runs:
            row["target_met"] = (
                row["E"] <= E_ref + 1e-9
                and row["grad_norm"] is not None
                and row["grad_norm"] < 1e-5
            )
        report["opt"] = {"chi": chi, "E_ref": E_ref, "runs": runs}
        print(
            f"opt targets (E <= {E_ref:.12f} + 1e-9, |g| < 1e-5): "
            f"{[row['target_met'] for row in runs]}",
            flush=True,
        )
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS {report['peak_rss_mb']:.0f} MB; ok={ok}", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Load and check the state**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; H=/home/yjkao/tenax-sma/hasik_states_u1/state_1s_A1_U1B_j20.0_D3_chi_opt72.json; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u -m benchmarks.ipeps_c4v_charged_ad --state $H --mode load --out $S/t9-load.json > $S/t9-load.log 2>&1; echo "exit $?"; cat $S/t9-load.log`
Expected: `load: vc=[0, 2, 0] pc=[0, 2] charged=True entries=<n> max|dense - A|=<x>` with x < 1e-12, then `peak RSS <m> MB; ok=True` and `exit 0`.

- [ ] **Step 3: The energy check**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; H=/home/yjkao/tenax-sma/hasik_states_u1/state_1s_A1_U1B_j20.0_D3_chi_opt72.json; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync python -u -m benchmarks.ipeps_c4v_charged_ad --state $H --mode energy --out $S/t9-energy.json > $S/t9-energy.log 2>&1`
Expected: three lines `energy chi=36|54|72: tenax <E> hasik <E_H> gap <gap>`. The script asserts nothing here (spec: "measures it and does not assume a value"). Ledger the three gaps. The reference [[reference_hasik_dat_nn_2x2_window]] expects a 2x1-against-2x2 window gap at finite chi.

- [ ] **Step 4: The FD check at chi=36**

Run (background; read the log when it ends): the same command with `--mode fd --out $S/t9-fd.json > $S/t9-fd.log 2>&1`.
Expected: `fd: first gradient <t> s, ...`, then three `fd dir k:` lines with `ok=True regime=True`, then `ok=True`. If the line `fd: first gradient` has not appeared 2 h after the start, stop the run (find its PID with `pgrep -x python` and read `/proc/<pid>/cmdline` before you kill it; never `pkill -f`) and ask the user before any A100 run (settled point "D=3 hardware"). If a direction has `ok=False`, stop and debug (superpowers:systematic-debugging): it is a gradient defect at D=3. If a direction has `regime=False` only, ledger `Task 9: Ruling: FD regime not met at D=3 chi=36 in dir <k> (<numbers>) — the D=2 FD tests carry the gradient claim`.

- [ ] **Step 5: The optimisation runs**

Run (background; read the log when it ends): the same command with `--mode opt --steps 100 --out $S/t9-opt.json > $S/t9-opt.log 2>&1`.
Expected: two `opt <label>: {...}` lines and `opt targets (...): [True, True]`. Ledger E, |g|, steps, wall_s, first_step_s, median_later_step_s and the peak RSS. If a target is missed, ledger it and report it to the user; do not change the loop or the targets.

- [ ] **Step 6: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t9.txt <<'EOF'
bench(ipeps): charged C4v AD at D=3 against Hasik's U(1)-C4v state

benchmarks/ipeps_c4v_charged_ad.py converts Hasik's D=3 U(1)-C4v state
to a charged C4v SymmetricTensor, compares the tenax energy with his .dat
values, runs the FD rule in three C4v directions at a perturbed state,
and optimises from a perturbed and a random start.  It reports steps,
wall time, the first-step time and the peak RSS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add benchmarks/ipeps_c4v_charged_ad.py
git commit -F $S/msg-ad-t9.txt > $S/commit-ad-t9.log 2>&1; tail -n 3 $S/commit-ad-t9.log
```

Expected: one commit.

---

### Task 10: Docs

**Files:**
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md` (a new section between Path 4, which ends at the bullet "- supports `gs_num_steps>0` optimization with implicit gradients.", and the line `(root-implicit-ad)=`)
- Modify: `README.md` (one line after the "**AD ground-state optimization**" bullet, line 33)

**Interfaces:**
- Consumes: the measured numbers in the ledger from Tasks 4, 5, 8 and 9.
- Produces: the guide anchor `#path-4b-charged-c4v-symmetric-input` (the GitHub and MyST slug of the heading), which the README line and the `optimize_gs_ad` docstring (Task 7) name.

- [ ] **Step 1: Write the guide section**

Insert, after the Path 4 bullet list and one blank line:

````markdown
### Path 4b: Charged C4v Symmetric Input

A charged C4v `SymmetricTensor` has a virtual charge q with `q != dual(q)`:
for U(1) any q ≠ 0, for Z_n any q ≠ n/2. `ctm_tensor_c4v` runs it as the
2-sublattice state (A, negate(A)) (#1211). With `unit_cell="1x1"` and
`gs_c4v=True`, `optimize_gs_ad` optimises it on the block data, with no
densification:

```
Cold start:  the #1211 sweep to sv-convergence; its layout L is frozen
Forward:     e <- align(F_L(e; A), e) to an element-wise fixed point
Backward:    (I - J^T + tau I) lambda = g by gmres_eager (restart 30),
             then dA = (dF/dA)^T lambda
Optimiser:   L-BFGS (memory 10) with the Hager-Zhang or Armijo line search
```

F_L is the sweep with its projector built at the frozen layout L (the
number of kept states per charge). `align` fixes the one free sign of each
chi index against the previous iterate, so the iteration converges
element-wise and the implicit function theorem applies.

**Enable it with:**

```python
from tenax import (CTMConfig, iPEPSConfig, heisenberg_gate, optimize_gs_ad,
                   sublattice_rotate_gate)

config = iPEPSConfig(
    unit_cell="1x1",
    gs_c4v=True,
    su_init=False,
    ctm=CTMConfig(chi=16, max_iter=500, conv_tol=1e-10),
)
# A_charged: a SymmetricTensor with labels u, d, l, r, phys whose four
# virtual faces carry equal flow-weighted charges, at least one of them
# not self-dual.
A, env, E = optimize_gs_ad(sublattice_rotate_gate(heisenberg_gate()),
                           A_charged, config)
```

**Options and caveats:**
- The start is projected to C4v and rebuilt with every block its charges
  allow; it need not be C4v.
- The layout L stays frozen for the run. After each evaluation a free
  ranking runs at the fixed point; a different layout is logged once and
  listed in `history["layout_mismatch_steps"]` (with `return_history=True`),
  and the run keeps L.
- The solve reads `adjoint_tol` (default 1e-8), `adjoint_maxiter` (default
  50 restarts of 30 matvecs) and `adjoint_tikhonov` (default 1e-6). It
  fails closed: a relative residual above max(10·`adjoint_tol`, 1e-12)
  raises `RuntimeError`. The Tikhonov term biases the gradient by about
  τ/(1 − ρ) relative, with ρ the spectral radius of the sweep Jacobian; set
  `adjoint_tikhonov=0.0` for an exact gradient (the FD tests do).
- A forward with no element-wise fixed point raises `CTMNotConvergedError`;
  the loop resets to the best point within `gs_stall_recovery_retries`. A
  line-search trial without a fixed point is a bad point.
- Refused: `gs_implicit_ad=False`, complex data, `projector_method="qr"`
  (`"svd"` and `"eigh"` both select the #1211 block projector), a
  `forward_gauge` other than `"auto"`/`"bond_phase"`,
  `ctm_conv_method="sv"`, `projector_backward` `"standard"` or
  `"lorentzian"`, `on_unconverged="warn"`, the χ-changing knobs and
  schedules, `gs_stall_recovery="noise"`, `gs_checkpoint_path`, `cg_gates`,
  `fuse_virtual_legs=False`, `gs_optimizer` other than `"lbfgs"`,
  `gs_line_search=False`, `ctm_mixing` and `device_mesh`.
- Not applied: `gs_metric_precond`. Not read: `adjoint_solver`,
  `adjoint_method`, `adjoint_arnoldi_precheck`.
- A charged C4v site with `unit_cell="2site"` and `gs_c4v=True` is refused:
  the 1x1 charged ansatz is already the 2-sublattice state.
- `jax.grad` through `ctm_tensor_c4v` itself still raises on charged
  input; the AD runs through this path only.

**Measured** (CPU, float64, Heisenberg J1, sublattice-rotated gate):

| D | χ | Start | E | \|g\| | Steps | Reference |
|---|---|---|---|---|---|---|
| 2 | 16 | exact_u1_d2 at 1.15·t | | | | −0.66023109348 (peps-torch) |
| 2 | 16 | random U(1) C4v | | | | −0.66023109348 |
| 3 | 36 | Hasik, perturbed | | | | E_tenax(Hasik; 36) |
| 3 | 36 | random U(1) C4v | | | | E_tenax(Hasik; 36) |

Converged FD at χ=6 (4 checks): grad_err ≤ … · fd_unc. Dense parity at
χ=16: |E − E_dense| = …. D=3, χ=36: first gradient … s, later steps … s,
peak RSS … MB; the tenax energy of Hasik's state differs from his `.dat`
value by … at χ=36, … at χ=54 and … at χ=72.
````

- [ ] **Step 2: Fill the measured numbers**

Fill every empty cell and every `…` in the section above from the ledger, digit for digit as the logs printed them:
- the D=2 rows: the `d2_off:` and `random:` lines of `$S/t8-green.log` (E, |g|, steps);
- the D=3 rows: the two `opt <label>:` lines of `$S/t9-opt.log` (E, grad_norm, steps), and `E_ref` from its `opt targets` line;
- the FD sentence: the largest `grad_err / fd_unc` over the four lines printed in `$S/t5-green.log`;
- the dense parity: the first `dense parity:` line of `$S/t8-green.log`;
- the D=3 sentence: `first_grad_s` from `$S/t9-fd.log`, `median_later_step_s` and `peak RSS` from `$S/t9-opt.log`, and the three gaps from `$S/t9-energy.log`.

A number that a task did not produce (for example a D=3 run that was stopped) is written as "not measured (see #<issue>)" only after the user names the issue; until then, ask. Then check that no empty cell or `…` is left:

Run: `cd /home/yjkao/tenax; awk '/^### Path 4b/,/^\(root-implicit-ad\)=/' docs/guide/algorithms/ipeps_ad_paths.md | command grep -n "…\|| |" || echo "complete"`
Expected: `complete`.

- [ ] **Step 3: The README line**

In `README.md`, insert after the line that starts with `- **AD ground-state optimization**`:

```markdown
- **Charged C4v AD** — `optimize_gs_ad` on a charged U(1)/Z_n C4v `SymmetricTensor`, block-sparse, through the 2-sublattice CTM sweep ([AD paths](docs/guide/algorithms/ipeps_ad_paths.md#path-4b-charged-c4v-symmetric-input))
```

- [ ] **Step 4: Check the example against the API**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && cat > $S/doc_example.py <<'EOF'
import sys
sys.path.insert(0, "/home/yjkao/tenax/tests")
from _c4v_staggered_fixtures import T_EXACT, charged_d2
from tenax import (CTMConfig, iPEPSConfig, heisenberg_gate, optimize_gs_ad,
                   sublattice_rotate_gate)

A_charged = charged_d2(1.15 * T_EXACT)
config = iPEPSConfig(
    unit_cell="1x1",
    gs_c4v=True,
    su_init=False,
    ctm=CTMConfig(chi=16, max_iter=500, conv_tol=1e-10),
)
A, env, E = optimize_gs_ad(sublattice_rotate_gate(heisenberg_gate()),
                           A_charged, config)
print(f"guide example: E={E:.12f}")
EOF
JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u $S/doc_example.py > $S/t10-example.log 2>&1; tail -n 2 $S/t10-example.log`
Expected: `guide example: E=-0.6602310…` (within 1e-8 of −0.66023109348).

- [ ] **Step 5: Commit**

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan
cat > $S/msg-ad-t10.txt <<'EOF'
docs(ipeps): the charged C4v AD path (Path 4b)

The AD-paths guide gets Path 4b: what charged C4v input is, how the
frozen-layout implicit AD works, how to enable it, its options, refusals
and caveats, and the measured D=2 and D=3 numbers.  README gets one
feature line that links to it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
cd /home/yjkao/tenax
git add docs/guide/algorithms/ipeps_ad_paths.md README.md
git commit -F $S/msg-ad-t10.txt > $S/commit-ad-t10.log 2>&1; tail -n 3 $S/commit-ad-t10.log
```

Expected: one commit.

---

### Task 11: The mutation sweep

**Files:**
- Scratch only: `$S/mutate.py` (rewritten with every mutant and a `--cmd` mode)

**Interfaces:**
- Consumes: the code of Tasks 3-9 as committed (commit before mutating; every mutant is reverted by `git checkout --` in `finally`).
- Produces: one ledger line per mutant, `Task 11: mutant <name>: KILLED by <test>` or a ruling.

Each mechanism test was RED only through a missing module or name. This task names the defect that each one catches ([[feedback_red_then_green_certifies_one_defect]]).

- [ ] **Step 1: Rewrite the mutation script**

Replace `$S/mutate.py` with:

```python
"""Apply one named mutant, run its check, revert (git checkout --) in finally.

    mutate.py <name> <pytest args...>      run pytest on the mutant
    mutate.py <name> --cmd <command...>    run a command on the mutant
"""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/home/yjkao/tenax")
ENGINE = "src/tenax/algorithms/_ctm_tensor_c4v_charged_ad.py"
LOOP = "src/tenax/algorithms/ipeps_optimize_c4v_charged.py"
POLICY = "src/tenax/algorithms/ipeps_ad_policy.py"
DISPATCH = "src/tenax/algorithms/ipeps_optimize.py"
BENCH = "benchmarks/ipeps_c4v_charged_ad.py"
MUTANTS = {
    # Task 5
    "P_frozen": (
        ENGINE,
        "    P_left, P_right = _c4v_staggered_projectors_at(Qf, layout)\n",
        "    P_left, P_right = jax.tree_util.tree_map(\n"
        "        jax.lax.stop_gradient, _c4v_staggered_projectors_at(Qf, layout)\n"
        "    )\n",
    ),
    "lambda_is_g": (
        ENGINE,
        "    lam, resid = gmres_eager(\n",
        "    lam, resid = b, 0.0\n    _unused = lambda: gmres_eager(\n",
    ),
    "plain_max_abs": (
        ENGINE,
        "    return _normalize_by_max_abs(C_new), _normalize_by_max_abs(T_new)\n",
        "    return C_new * (1.0 / (C_new.max_abs() + 1e-15)), "
        "T_new * (1.0 / (T_new.max_abs() + 1e-15))\n",
    ),
    # Task 8
    "site_skips_P": (LOOP, "        y = P @ x_\n", "        y = x_\n"),
    # Task 11
    "no_alignment": (
        ENGINE,
        "    return _align(C_new, T_new, T)\n",
        "    return C_new, T_new\n",
    ),
    "gate_removed": (
        ENGINE,
        "    if not (math.isfinite(resid) and resid <= gate):\n",
        "    if False:\n",
    ),
    "layout_check_skipped": (
        LOOP,
        "            mismatch_steps.append(step + 1)\n",
        "            pass\n",
    ),
    "reset_reraises": (
        LOOP,
        '            if x is best_x or best_E == float("inf") or stall + 1 > retries:\n',
        "            if True:\n",
    ),
    "refusal_dropped": (
        LOOP,
        '    if ctm.ctm_conv_method != "elementwise":\n',
        "    if False:\n",
    ),
    "trial_reraises": (
        LOOP,
        '    except CTMNotConvergedError:\n        return float("inf")\n',
        '    except ValueError:\n        return float("inf")\n',
    ),
    "no_canonicalise": (
        LOOP,
        "    A0 = SymmetricTensor.from_dense(A_init.todense(), A_init.indices)\n",
        "    A0 = A_init\n",
    ),
    "override_after_check": (
        LOOP,
        "    _check_charged_c4v_config(config, ctm, A_init)\n",
        "    _check_charged_c4v_config(config, config.ctm, A_init)\n",
    ),
    "dispatch_off": (
        POLICY,
        "        and _is_charged_c4v(A_init)\n",
        "        and False\n",
    ),
    "two_site_refusal_dropped": (
        DISPATCH,
        "        _refuse_charged_c4v_2site(A_init, config)\n",
        "",
    ),
    "projector_check_not_skipped": (
        DISPATCH,
        "        or use_charged_c4v_path(config, A_init)\n"
        "    ):\n"
        "        config = _resolve_projector_backward(config)\n",
        "    ):\n        config = _resolve_projector_backward(config)\n",
    ),
    "wrong_k": (
        BENCH,
        "    pc = np.asarray([total - p for p in phys], dtype=np.int32)\n",
        "    pc = np.asarray([p - total for p in phys], dtype=np.int32)\n",
    ),
}


def main(name, *args):
    rel, old, new = MUTANTS[name]
    path = ROOT / rel
    text = path.read_text()
    assert text.count(old) == 1, f"the anchor of {name} is not unique"
    for pyc in path.parent.glob("__pycache__/*.pyc"):
        pyc.unlink()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu"}
    if args and args[0] == "--cmd":
        cmd = list(args[1:])
    else:
        cmd = [
            "uv", "run", "--no-sync", "pytest", "--no-cov",
            "-p", "no:cacheprovider", "-q", *args,
        ]
    path.write_text(text.replace(old, new))
    try:
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    finally:
        subprocess.run(["git", "checkout", "--", rel], cwd=ROOT, check=True)
    print(f"MUTANT {name}: exit {rc} ({'KILLED' if rc != 0 else 'SURVIVED'})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
```

Then dry-run the anchors with no mutation applied:

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && uv run --no-sync python - <<'EOF'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("m", "/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan/mutate.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for name, (rel, old, new) in m.MUTANTS.items():
    n = (m.ROOT / rel).read_text().count(old)
    print(f"{name}: {n}")
EOF`
Expected: every line ends with `: 1`. A count of 0 means the code differs from the plan: read the file, fix the anchor to the committed text, and ledger the change.

- [ ] **Step 2: Run each mutant against its test**

`git status --short src/ benchmarks/ tests/` must be empty first (everything committed). Then run each line in the background, one at a time, and read its log before the next:

```bash
S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax
M="taskset -c 0-11 uv run --no-sync python -u $S/mutate.py"
T=tests/test_c4v_charged_ad.py
H=/home/yjkao/tenax-sma/hasik_states_u1/state_1s_A1_U1B_j20.0_D3_chi_opt72.json
$M no_alignment $T -k "fixed_point or needs_the_alignment" > $S/t11-no_alignment.log 2>&1
$M gate_removed $T -k unsolved_adjoint > $S/t11-gate_removed.log 2>&1
$M layout_check_skipped $T -k layout_mismatch > $S/t11-layout_check_skipped.log 2>&1
$M reset_reraises $T -k reset_on_an_unconverged > $S/t11-reset_reraises.log 2>&1
$M refusal_dropped $T -k "refusals and ctm_conv_method" > $S/t11-refusal_dropped.log 2>&1
$M trial_reraises $T -k trial_energy_is_inf > $S/t11-trial_reraises.log 2>&1
$M no_canonicalise $T -k missing_block > $S/t11-no_canonicalise.log 2>&1
$M override_after_check $T -k overrides_before_the_check > $S/t11-override_after_check.log 2>&1
$M dispatch_off $T -k dispatch_reaches > $S/t11-dispatch_off.log 2>&1
$M two_site_refusal_dropped $T -k two_site_charged > $S/t11-two_site_refusal_dropped.log 2>&1
$M projector_check_not_skipped $T -k skips_the_fused_projector > $S/t11-projector_check_not_skipped.log 2>&1
$M wrong_k --cmd uv run --no-sync python -u -m benchmarks.ipeps_c4v_charged_ad --state $H --mode load > $S/t11-wrong_k.log 2>&1
```

Expected: each log ends with `MUTANT <name>: exit <non-zero> (KILLED)`, and `git status --short src/ benchmarks/` is empty after each run. Read the failing assertion in each log: it must be the assertion the test exists for, not an import error or a regime assert. A mutant that SURVIVES is a finding: either the test does not pin its defect (fix the test, rerun the mutant, ledger both) or the mutant is equivalent (show why with a measurement, and ledger `Task 11: Ruling: <name> is equivalent — <evidence>`). Ledger one line per mutant.

Not mutated, by ruling: the start projection (`x = P @ A0._data`). `site` projects every evaluation, so a start that skips P changes only |x| and the step scale, not any evaluated state.

---

### Task 12: Verification

**Files:** none changed (fixes found here go into the task that owns the code, with a ledger ruling).

- [ ] **Step 1: The #1211 outputs are unchanged**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 12-23 uv run --no-sync python -u $S/snap_1211.py check > $S/t12-snap.log 2>&1; tail -n 2 $S/t12-snap.log`
Expected: `identical`.

- [ ] **Step 2: The core suite**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider -m core -q > $S/t12-core.log 2>&1`
Expected: the last line reports `passed` with `0 failed` and no `error`. A failure outside the new files is checked against `main` first ([[reference_host_only_warnings]], [[feedback_verify_against_the_right_ref]]): run the same test on a `git worktree` of `origin/main` only if the user allows a worktree; otherwise report it by name.

- [ ] **Step 3: The algorithm files of this work**

Run (background; read the log when it ends): `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; uptime; cd /home/yjkao/tenax && JAX_PLATFORMS=cpu taskset -c 0-11 uv run --no-sync pytest --no-cov -p no:cacheprovider tests/test_c4v_charged_ad_physics.py tests/test_ctm_tensor_c4v.py -q > $S/t12-algo.log 2>&1`
Expected: all passed (an `xfail` from Task 8 counts only with its ledger line).

- [ ] **Step 4: Hooks and tree**

Run: `S=/home/yjkao/.claude/jobs/1ad6c4e6/tmp/c4v-plan; cd /home/yjkao/tenax && uv run --no-sync pre-commit run --files $(git diff --name-only main...HEAD) > $S/t12-hooks.log 2>&1; tail -n 6 $S/t12-hooks.log; git status --short; git log --oneline main..HEAD`
Expected: every hook `Passed` or `Skipped`; `git status --short` shows only `?? out` and `?? out2`; the log lists the spec, the plan and the task commits.

---

## Finish

1. Run the final whole-branch review that the chosen execution skill prescribes (superpowers:subagent-driven-development or superpowers:executing-plans), with the Review Focus section above passed verbatim and a pointer to the ledger's `Ruling:` lines.
2. Use superpowers:finishing-a-development-branch. The base branch is `main`. Pushing, the PR, labels and every GitHub post wait for the user's word. Suggest the `run-full-tests` label (the physics file is `algorithm`, so required CI skips it; [[reference_run_full_tests_label_needs_push]]: the label takes effect on the next push).
3. If the user asks for the PR: push with `git push origin feat/c4v-charged-ad:refs/heads/feat/c4v-charged-ad`, write the body to `$S/pr-body.md` from the template below, and run `gh pr create -R tenax-lab/tenax --base main --head feat/c4v-charged-ad --title "feat(ipeps): charged C4v iPEPS AD through the 2-sublattice sweep" --body-file $S/pr-body.md`. Never arm `--auto`; never pass `--delete-branch`. Before any merge, read both review endpoints and the issue comments with `--paginate` (CLAUDE.md).

PR body template (fill each `<…>` from the ledger, digit for digit):

```markdown
> 🤖 **AI-generated comment** — written by Claude Code, posted by @yingjerkao.

## Summary

- `optimize_gs_ad` with `unit_cell="1x1"` and `gs_c4v=True` optimises a charged U(1)/Z_n C4v `SymmetricTensor` on its block data (spec 2): implicit AD through the #1211 2-sublattice sweep at the frozen layout of its cold start.
- New modules: `_ctm_tensor_c4v_charged_ad.py` (the sign-aligned sweep, the element-wise phase, the `charged_c4v_converge` custom_vjp with a fail-closed GMRES solve) and `ipeps_optimize_c4v_charged.py` (L-BFGS with the 1x1 line searches, the refusals, the reset rule, the layout check). `use_charged_c4v_path` routes to it.
- The #1211 projector splits into an eager ranking and a traced builder at a frozen layout; the eager outputs are bit-identical (snapshot check).

Spec: `docs/superpowers/specs/2026-11-23-c4v-charged-ad-design.md`. Plan: `docs/superpowers/plans/2026-11-23-c4v-charged-ad.md`.

## Validation (CPU, float64, J=1)

| Check | Result |
|---|---|
| Converged FD at χ=6, h scan, 4 checks | max grad_err / fd_unc = <…>; flow − frozen ≥ <…>·fd_unc |
| D=2 optimum, χ=16, two starts | E = <…>, <…>; \|E − (−0.66023109348)\| ≤ <…>; \|g\| < 1e-5 |
| Dense parity, χ=16 | \|E − E_dense\| = <…>; dense C4v optimiser within <…> |
| D=3 vs Hasik, χ=36/54/72 | energy gap <…>, <…>, <…>; FD grad_err / fd_unc ≤ <…> |
| D=3 optimisation, χ=36 | perturbed: E = <…>, \|g\| = <…>, <…> steps; random: <…> |
| Mutants | <n> killed: <names>; rulings: <…> |

## Test plan

- [x] `pytest -m core` (the new core file: <n> tests, <t> s)
- [x] `tests/test_c4v_charged_ad_physics.py`, `tests/test_ctm_tensor_c4v.py`
- [ ] `run-full-tests` label

Follows #762 and #763 (closed by #1211).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
```
