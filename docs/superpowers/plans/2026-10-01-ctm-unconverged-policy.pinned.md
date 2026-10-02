# Regression inventory: tests the `on_unconverged="raise"` default broke (#1059)

Baseline: `c606103` (this branch's merge-base), same machine, same commands,
CPU (`JAX_PLATFORMS=cpu`), per-file pytest processes. Only failures that are
new on this branch are listed. Failures that also occur on the baseline
(`test_adjoint_seed_858`, `test_ctm_hold::test_a_converged_dense_d2_environment_holds`,
the `test_ctm_sharding_parity` / `test_ctm_energy_implicit` / `test_fpeps_ad`
slow chronic set, the `test_ipeps_gauge_perf` timing flake) are not.

Buckets:

- **(a) pinned** to `CTMConfig(on_unconverged="warn")`, with a
  `# needs an unconverged forward: <why>; see #1059` comment.
- **(b) encoded the bug**: the test passed on a forward that never converged.
  The fix raises `max_iter` until every checked forward converges. Each value
  below was verified under the default `"raise"`, so the test passing proves
  convergence at every checked site. When a test asserted a number, that number
  was an unconverged one before.
- **(c) genuine regression**: the implementation was fixed.

"Flat in max_iter" means the forward's residual did not move when the budget
grew. Such a forward cycles rather than converging slowly, so raising
`max_iter` cannot fix it.

## (c) Genuine regression

| Test | Reason / fix |
|---|---|
| `test_architecture_imports.py::test_algorithm_import_cycles_are_allowlisted_only` | Task 6's own `strict=` imported `CTMConvergeInfo` from `_ctm_python_loop` inside `_ctm_tensor_convergence`, which `_ctm_python_loop` imports, creating a new import cycle. Fixed by moving `CTMConvergeInfo` to the leaf `_ctm_convergence_policy` module, re-exported from `_ctm_python_loop` (identity and import paths unchanged). |

## (a) Pinned to `on_unconverged="warn"`

| Test | Why it needs an unconverged forward |
|---|---|
| `test_ipeps_final_energy_is_fresh_899.py` (module fixture, pinned in Task 5) | `max_iter=12` is deliberately too few to converge (cheap fixture). The test is about which env seeds the final evaluation, not about convergence. |
| `test_ipeps.py::test_gs_ctm_max_iter_schedule_caps_late_step_ctm` | The schedule deliberately caps late steps at 5 CTM sweeps. Truncation is the feature under test. |
| `test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_runs` | **Cannot converge, not deliberate.** This init's 1-site CTM cycles: stationarity residual 0.238 at step 1, identical at max_iter 10, 100 and 300, with dense and symmetric agreeing. The test checks the SymmetricTensor round-trip. |
| `test_ipeps.py::TestADSymmetric::test_optimize_gs_ad_symmetric_matches_dense` | Same init as the previous row: cannot converge. Type round-trip plus finiteness only. |
| `test_ipeps_excitations.py::TestOptimizeGsAd::test_heisenberg_negative_energy` | **Cannot converge.** 20 steps from a random init reach a state whose 1-site CTM cycles: the step-13 residual is 1.2e-4 even at max_iter=300. The assertion is a loose `E < 1.0`. |
| `test_ipeps_excitations.py::TestOptimizeGsAd::test_su_init_ignored_when_A_init_provided` | **Cannot converge.** This `A_init`'s step-1 forward residual is 7.6e-6 at max_iter 50, 100 and 300 alike. The test checks that `su_init` is ignored. |
| `test_ipeps_u1sz.py::TestU1SzSymmetricMatchesDense::test_one_step_symmetric_charged_ctm_no_collapse` | **Encoded the bug, and the budget cannot fix it.** Its docstring already says the CTM "is not fully converged". Neither forward converges: symmetric residual 7.3e-3 at max_iter 100 and 300; dense 0.64 at 20 and 0.78 at 100. The `E_sym ≈ E_dense (2e-2)` check therefore compares two unconverged energies. Pinned to keep the #602 no-collapse guard. **Follow-up:** find a converging init and drop the pin. |

## (b) Encoded the bug: budget raised until the forward converges

| Test | Before → after | Evidence |
|---|---|---|
| `test_ctm_convergence_meta.py::test_su_warm_start_never_runs_the_measurement_ctm[1x1]`, `[2site]` | max_iter 5 → 30 | 5 < `min_iter`=10, so the final CTM was never measured (`sv_diff inf`). Passes at 20, 30 and 50. |
| `test_ipeps_ad_conv_criterion.py::test_grad_norm_criterion_exits_when_grad_is_tiny`, `::test_dE_default_unchanged_when_dE_tol_loose`, `::test_both_criterion_requires_both_to_pass` | 5 → 30 | Same never-measured final CTM. Whole file passes at 30 and 60. |
| `test_ipeps_ad_history.py::test_optimize_gs_ad_returns_history_1site`, `::test_optimize_gs_ad_default_no_history` | 5 → 30 (1-site helper only) | Same. Passes at 30 and 60. |
| `test_ipeps_checkpoint_resume.py::test_first_evaluation_convergence_still_writes_checkpoints`, `::test_normal_completion_flushes_last_checkpoint_despite_cadence`, `::test_final_step_state_is_serialized_once` | chi=1 helper: 5 → 20 | Never measured (5 < `min_iter`). Passes at 20 and 50. |
| `test_ipeps_checkpoint_resume.py::test_resume_rejects_plain_to_cg` (slow) | 20 → 100 | Gradient forward residual 1.2e-7 vs tol 1e-8 at step 1. Passes at 100 and 300. |
| `test_ipeps_checkpoint_resume.py::test_resume_1site_continues_from_saved_step` | none | Listed by the Task 2/3 reviews. It **passes on this branch head unchanged**: Task 5's final-energy handling fixed it. A standalone run converges at max_iter 100, 300 and 1000 alike. |
| `test_coarse_grain.py::TestCGOptimizerGuards::test_user_supplied_raw_params_are_respected` | 20 → 100 | It compared two final energies, each unconverged (`sv_diff` 3.7e-5). Passes at 100 and 300. |
| `test_ipeps_chi_bump_integration.py` (all 4) | 10 → 60 | Final and gradient forwards were short of `conv_tol`=1e-3. Passes at 60 and 200. |
| `test_ipeps_chi_schedule_wiring.py::test_chi_schedule_bumps_between_stages`, `::test_reactive_plus_scheduled_compose_2site_smoke` | 10 → 60 | Step-1 gradient forward unconverged. Passes at 60 and 200. |
| `test_ipeps_excitations.py::TestOptimizeGsAd::test_runs_without_error` | 5 → 300 | Step-3 forward unconverged at 50 and 100 (residual ~4e-8 vs 1e-8). At 300, every step converges. The run's energy was 0.295 at 100 (unconverged, warn mode) against 0.4547 converged. |
| `test_ipeps_excitations.py::TestOptimizeGsAd::test_su_init_runs_without_error` | 10 → 50 | Passes at 50, 100 and 300. |
| `test_ipeps.py`: 15 tests (`TestOptimizeGsAd2Site::test_2site_ad_runs`, `::test_2site_ad_zero_steps_returns_energy`, `::test_2site_ad_mixed_init_types_work`, `::test_2site_noc4v_ad_stays_variational_issue_328`, `::test_2site_noc4v_ad_norms_stay_unit`; `TestOptimizeGsAdLogging::test_verbose_prints_progress`; `TestOptimizeGsAdDenseOnly::test_symmetric_tensor_2site_runs`; `TestOptimizeGsAdOptimizers::test_lbfgs_optimizer_runs`, `::test_cg_optimizer_runs`; `TestADSymmetric::test_optimize_gs_ad_nontrivial_u1_preserves_symmetric_type`; `test_noise_floor_does_not_gate_healthy_optimization`; `test_loss_fn_fwd_updates_env_cache_for_hz_dphi_reuse`; `test_loss_fn_fwd_probe_envs_dont_leak_past_line_search`; `test_2site_implicit_ad_ctmrg_heuristic_increase_chi_grows_env`; `test_gs_ctm_max_iter_schedule_default_none_unchanged`) | 3, 5, 8, 10 or 12 → 100 | All were unconverged at the old budget, and all pass at 100. `default_none_unchanged` asserts that the unscheduled `max_iter` passes through unchanged, so its expected value moved from `{12}` to `{100}` with the config. |
| `test_ipeps.py::test_stall_reset_reinits_optax_lbfgs_state` | 5 → 50 | The step-1 forward was unconverged, so the run raised before any stall could happen (`contextlib.suppress` hid it). Passes at 50. |
| `test_split_ctm_fuse_flag.py::test_optimize_gs_ad_fused_still_returns_fused_env` | 40 → 200 (`ctm={"max_iter": 200}`) | Fused 2x2 gradient forward residual 1.4e-6 vs tol 1e-10. Passes at 200 and 500. |
| `test_ctm_sharding_backward.py::test_sharded_optimize_gs_ad_matches_single_device` (`tests/_rung2_optimize_parity_subproc.py`) | 60 → 500 | The subprocess's own comment claims "converge to a true fixed point", but 60 sweeps left the gradient forward at 1.1e-6 vs tol 1e-10. At 200, the final CTM still stops at 4.7e-10. At 500 it converges: `\|dE\|`=1.1e-16, `max\|dA\|`=1.1e-16. |
| `test_varipeps_compare.py::test_smoke_run_tenax_single_site_d2_chi4_test_fast` (`benchmarks/varipeps_compare/run_tenax.py`, `test_fast` only) | 5 → 30 | Same never-measured final CTM. Passes at 30. Production runs are unchanged. |
| `test_fpeps_ad.py::TestOptimizeFpepsAd::test_optimize_fpeps_ad_with_explicit_init` (slow) | fixture's 10 → 100 (this test only) | Step-1 gradient forward was at 1.94 vs tol 1e-4. At 100 sweeps, all three steps converge. |
| `test_frozen_layout_ad.py::test_frozen_layout_ad_lowers_the_energy_and_keeps_the_layouts` (slow) | 50 → 400 (this test only; `_ad_config` now takes a `max_iter` override) | At 50, a forward was unconverged, the site-1 reset fired, and E1 −0.53714 > E0 −0.53767. At 150, E1 −0.537567 was still above E0 −0.537672. At 400 it passes, and all 7 slow tests in the file pass at HEAD (837 s against 686 s on the baseline for this test). This test is the weakest (b): passing under `"raise"` rules out an uncaught unconverged forward, but not a reset that recovered. |

## Not caused by this branch

| Test | Status |
|---|---|
| `test_ctm_sharding_backward.py::test_sharded_backward_grad_matches_single_device` | `subprocess.TimeoutExpired` (900 s) at machine load ~600 to 1200, twice. This is load, not the branch. Its subprocess (`ctm_energy_implicit` directly) touches no code this branch changed. Run side by side under the same load, the two clones print identical output (`\|dE\|`=1.39e-17, `grad_max\|delta\|`=1.90e-15) in 2337 s (branch) and 2197 s (baseline). |
