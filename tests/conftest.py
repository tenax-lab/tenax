"""Shared fixtures for the Tenax test suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Auto-apply markers based on test file name                           #
# ------------------------------------------------------------------ #

_FILE_MARKERS = {
    "test_tensor.py": "core",
    "test_index.py": "core",
    "test_symmetry.py": "core",
    "test_contraction.py": "core",
    "test_network.py": "core",
    "test_netfile.py": "core",
    "test_fermionic.py": "core",
    "test_dmrg.py": "algorithm",
    "test_idmrg.py": "algorithm",
    "test_itebd.py": "algorithm",
    "test_trg.py": "algorithm",
    "test_hotrg.py": "algorithm",
    "test_ipeps.py": "algorithm",
    "test_ipeps_bp_gauge.py": "algorithm",
    "test_ipeps_gauge.py": "algorithm",
    # #882 Tasks 7 and 7b: the dense ``gauge_fix`` is one traced call -- a
    # ``lax.while_loop`` solve with ``absorb_weights`` compiled around it --
    # which is what makes re-gauging every simple-update step affordable.
    # Guards traced-vs-eager parity at both boundaries, flow preservation, the
    # compile-cache key (a second compilation costs more than the whole
    # re-gauging budget and is invisible to wall-clock), and the budget itself.
    # The two wall-clock assertions carry ``@pytest.mark.timing``; one is a
    # machine-independent ratio and the other withdraws under coverage or on a
    # slow box rather than reporting a number it cannot measure.
    "test_ipeps_gauge_perf.py": "algorithm",
    # #882 Phase 2: the simple-update engine with no stored bond spectrum.
    # Every test runs at least one BP gauge solve (and the split guard runs
    # three), which puts it well past the `core` budget.
    "test_ipeps_su.py": "algorithm",
    # #882 Task 13: every guard in test_ipeps_su.py killed by a faithful
    # re-introduction of the defect it is named for.  Each cell runs its
    # guard twice -- unmutated (which must pass) and mutated (which must
    # fail, on the named assertion, at a number in range) -- so it costs
    # about twice the guard.  Eight cells; exactly one carries `slow`, and
    # it is a `pytest.param` mark rather than a function mark because the
    # kills are one parametrised test.  Measured on CPU with --no-cov, box
    # load 4-7: -m "not slow" is 45.6 s over seven cells (the two dearest
    # being the D=2 imaginary-time run at 17.7 s -- which is NOT the slow
    # one -- and the symmetric chain anchor at 22.6 s), and -m slow is
    # 234 s in that single cell, the symmetric D=3 truncation guard.
    #
    # **Three of its rows carry an explicit `core` mark** (851, 869, 6.2a --
    # 2.35 s, 0.46 s and 0.03 s both sides), so the file contributes to the
    # required gate.  It has to: this is the file that certifies the guards in
    # `test_ipeps_su.py` can fail, and with the whole file in `algorithm` it
    # collected zero tests under `-m core` -- every mutation cell ran only in
    # `fast-other`, which is not required and is chronically red on `main`.
    # That is the state the `test_arnoldi.py` note below calls "a guard that
    # runs in no required job is not one", and it was the state of the one
    # file whose subject is exactly that.  See `_MUTANTS` for the measurement.
    "test_ipeps_su_mutations.py": "algorithm",
    "test_ipeps_core.py": "core",
    "test_auto_mpo.py": "algorithm",
    "test_ad_utils.py": "algorithm",
    "test_fermionic_ipeps.py": "algorithm",
    "test_fpeps_ad.py": "algorithm",
    "test_fermionic_ed_reference.py": "algorithm",
    "test_ipeps_excitations.py": "algorithm",
    "test_code_review_regressions.py": "core",
    "test_tensor_utils.py": "core",
    "test_split_ctm_tensor.py": "algorithm",
    "test_split_ctm_2site.py": "algorithm",
    # 2-site split-CTM SymmetricTensor (#463 Phase 3): forward/energy-parity
    # tests run in the algorithm bucket; the AD parity tests in this file carry
    # their own explicit @pytest.mark.slow so they stay out of -m "not slow".
    "test_split_ctm_2site_symmetric.py": "algorithm",
    # 2-site split-CTM fermionic (#463 Phase 4): block-sparse forward parity vs
    # the fused sweep; algorithm bucket like its symmetric/split siblings.
    "test_split_ctm_2site_fermionic.py": "algorithm",
    # Dense 2-site split-CTM AD (#463 Phase 2): each parity test converges a
    # coupled fixed point + a full AD gradient (~5 min); slow-only so they stay
    # out of the -m core CI gate and the -m "not slow" bucket.
    "test_split_ctm_2site_ad.py": "slow",
    # The rank-1 environment-collapse guards (#723 fused / #746 split).  These
    # gate a defect that silently produced chi-independent mean-field energies
    # on the default path for every result predating #747, so they belong in
    # the *required* check rather than the unwatched full suite -- both files
    # were previously absent from this table entirely, which meant `-m core`
    # deselected them and they never ran in a required job.  Cheap: D=2,
    # chi<=16 dense, ~50s each.
    "test_ctm_723_single_site_collapse.py": "core",
    "test_split_ctm_746_single_site_collapse.py": "core",
    # The #747 collapse detectors themselves. Cheap (D=2, chi=8) and they guard
    # the guard: if these rot, nothing else notices a collapsed environment.
    "test_ctm_collapse_detector.py": "core",
    # ``ctm_tensor(return_meta=True)``: the only way a caller can find out
    # whether the environment it is about to read an energy from is a fixed
    # point.  Same defect class as the two files above -- a silent wrong answer
    # on the default path -- so it belongs in the required gate for the same
    # reason.  Shares one D=2 chi=8 module fixture across every case (~9s).
    "test_ctm_tensor_return_info.py": "core",
    # The #911 ``recipe="1x1"`` deprecation contract.  ``core`` rather than the
    # ``_UNBUCKETED_LEGACY`` set the older deprecation files sit in: a warning
    # that silently stops firing is indistinguishable from one nobody hit, and
    # the suite-wide ``ignore:`` filter in pyproject means nothing else would
    # notice.  No CTM convergence anywhere -- ``max_iter=2`` on a random D=2
    # site, since the warning does not depend on physics (~4.5s).
    "test_recipe_1x1_deprecation.py": "core",
    # #785: the only thing that says whether a root-implicit gradient is
    # accurate.  The contract tests run on a closed-form quartic with no CTM
    # anywhere (~0.2s), so the measurement semantics -- a wrong gradient
    # reported wrong by the right amount, an unresolvable scan reported as
    # unresolvable -- are cheap enough for the required gate.  The rank-matched
    # pair that pins the residual's anti-correlation is @slow (~6 min).
    "test_root_implicit_gradient_check.py": "core",
    # #667: simple update converged to the product state, and survived because
    # *no test asserted a simple-update energy* -- one test even documented the
    # bug ("small dt causes the bond lambdas to converge to a product-like fixed
    # point") as intended behaviour.  A guard against a silent-wrong-answer
    # defect of that shape belongs in the gate that blocks a merge.  The D=2 run
    # is shared across the cases via a module fixture; the D=4 case is @slow.
    "test_su_865_symmetric_collapse.py": "core",
    "test_su_667_product_state.py": "core",
    # #851: the four-phase checkerboard sweep stored two Schmidt spectra for
    # its four bonds, so ``num_imaginary_steps % 4`` selected which bond's
    # gauge was stamped onto the lattice -- a silent-wrong-answer defect
    # steered by an ordinary convergence knob, and one that only shows up away
    # from the symmetric fixed point, i.e. exactly where nobody is looking.
    # Cheap: D=2/3 simple update, no CTM anywhere (~30s total).
    "test_su_851_four_bond_lambdas.py": "core",
    # The four-phase sweep had seven byte-identical copies, and #667 had to be
    # applied to each by hand.  These are exact-equality checks on a handful of
    # short sweeps (~18s, no CTM), and they are the only thing standing between
    # the consolidation and a silent behaviour change, so they belong in the
    # gate rather than in a bucket nothing runs on a PR.  They now also pin the
    # #851 default: with shared spectra the sweep must still reproduce the
    # pre-#851 loop element for element.
    "test_su_sweep_consolidation.py": "core",
    # #878's collapse guards (fpeps() returned exactly 0.0 by step 10).  These
    # assert on the bond SPECTRUM, never the norm -- `_normalize_tensor` runs
    # last in the update, so |A| reads a healthy 1.0 right up to the step where
    # it is exactly 0, and `isfinite` passes on the corpse.  They belong in
    # `algorithm` rather than the merge gate purely on cost: a fermionic 4-bond
    # cycle is ~1.2 s at D=2 (block-sparse eager dispatch, #566/#618), so the
    # 10/20/40-step sweeps here run 280 s locally -- two orders of magnitude
    # above anything else in `core`.
    "test_fpeps_878_su_collapse.py": "algorithm",
    # The fPEPS sublattice diagnostic (#881 review).  ``fpeps()`` returns two
    # tensors only because the t-V ground state is a checkerboard CDW, and this
    # is the probe that says whether a given run produced one -- a wrong answer
    # from it is a wrong answer about the whole reason the return type changed.
    # The first version compared leg Gram spectra, which are not gauge
    # invariant, so it reported a difference between representations; these pin
    # the replacement against an explicit non-unitary bond gauge.  Same cost
    # bracket as its `algorithm` neighbours (short D=2 sweeps + chi=8 CTM).
    "test_fpeps_881_sublattice_gap.py": "algorithm",
    # Root-implicit AD wiring (#715): dispatch + guard surface only, no
    # CTM convergence, so it is milliseconds.  The production-run case it
    # also carries is explicitly @slow (#772).
    "test_root_implicit_wiring.py": "core",
    # #858: the adjoint solver was seeded above ``||b||``, so it could return a
    # ``lambda`` that solves ``(I - J^T)lambda = dE/denv`` *worse than
    # ``lambda = 0``* -- measured relative residuals 1.581 and 2.103.  A wrong
    # gradient that still optimizes is the least visible failure this library
    # has, so it belongs in the gate that blocks a merge.  Mostly synthetic
    # (milliseconds); the two D=2 chi<=8 backward cases that cover the call
    # sites are ~25s each.
    "test_adjoint_seed_858.py": "core",
    # MPS/MPO physical-basis agreement (#816).  A mismatched charge ORDER
    # still converges and reports the correct energy while returning a
    # permuted state, so the energy check that normally catches everything
    # passes.  Small L=6 DMRG runs, a few seconds.
    "test_dmrg_phys_basis.py": "core",
    # Knobs the root-implicit path accepted and then ignored (#792).  Mostly
    # config validation and one warning: milliseconds, no CTM.  The masked-
    # gradient convergence-flag test does converge one final D=2/chi=4 env
    # (~1s) and is intentionally left in ``core`` -- it guards a flag a
    # benchmark consumer reads (#812).  Only the end-to-end return_history
    # shape test is expensive; it carries its own ``@pytest.mark.slow``, which
    # the rule below honours by *withholding* this ``core``.
    "test_root_implicit_ignored_knobs.py": "core",
    # A masked gradient must not end the optimization (#812).  Scripted
    # (energy, grad) pairs, so the only real work is one final D=2/chi=4 env
    # per test (~1s).  Guards control flow a benchmark consumer cannot see.
    "test_root_implicit_masked_convergence.py": "core",
    # Environment-phase (gauge) invariance of the RDM builders (#748, follow-up
    # to #725/#742).  Cheap -- one module-scoped D=2 simple-update state per
    # file, then pure phase reruns of the contraction (~5s each).  These guard
    # a defect that silently rescaled the physical part of every RDM under a
    # CTM gauge the sweep does not fix, so they belong in the *required* gate
    # rather than the unwatched full suite; unregistered, all 80 of them were
    # deselected by ``-m core`` and ran only on push to main.
    "test_ipeps_rdm_gauge.py": "core",
    "test_ipeps_excitations_gauge.py": "core",
    "test_ctm_honeycomb_energy_gauge.py": "core",
    # Multi-operand mixed-dtype einsum must not emit a real-by-real GEMM with
    # a complex output (#813).  Jaxpr inspection + an AST scan, so it is
    # microseconds and -- critically -- backend-independent: the crash it
    # guards only reproduces on CUDA, and the required gate is CPU-only.  The
    # two gauge files above are what it actually protects: they are already
    # ``core``, CI ran them green, and they still died on GPU.
    "test_einsum_mixed_dtype.py": "core",
    # ``_normalise_rdm``'s zero-matrix branch must stay differentiable: the
    # excitation H_eff/N are built by differentiating the norm at ``B = 0``, so
    # a NaN cotangent there is as fatal as a NaN value.  Milliseconds for the
    # unit half; the reachability test converges one D=2 chi=8 CTM.
    "test_normalise_rdm_zero_grad.py": "core",
    # The bucket guard itself (#805). Pure filesystem inspection, microseconds,
    # and it must run in the gate it protects or it protects nothing.
    "test_bucket_registry.py": "core",
    # The dense CTM entry points must report whether they converged (#839).
    # D=2, chi=6, starved budgets -- a few seconds. It belongs in the required
    # gate because the defect it guards is *silence*: nothing fails when the
    # flag goes missing again, which is exactly how it went unnoticed until a
    # scan showed conv_tol having no effect on the answer.
    "test_ctm_complex_site_tensors.py": "core",
    "test_rdm_validity_guard.py": "core",
    # #799: the symmetry core disagreeing with itself about charges. ``core``
    # because both defects are silent and fail *open* -- one admits a
    # nonconserving block, the other drops a block's data on ``todense()``
    # -- so nothing else in the suite would notice a regression. Runs in <1s.
    "test_symmetry_charge_arithmetic_799.py": "core",
    # #789: the phase-fix VJP guard. ``core`` because the defect is a NaN
    # gradient that only appears in the zero column -- everything else in a
    # cotangent looks healthy, so nothing else in the suite would notice a
    # regression, and the production default phase gauge is one of the four
    # sites. The whole file runs in ~6s.
    "test_phase_fix_nan_vjp_789.py": "core",
    # #834: block-sparse contract() silently disagreeing with the densified
    # contraction on legs whose charges do not line up. ``core`` because the
    # defect is representation-dependent arithmetic in the contraction engine
    # -- every algorithm above it inherits the wrong answer, and nothing warns.
    # It also pins the two facts that shaped the design: that the default CTM
    # path must keep contracting, and that one site on it is still 8.3e-01
    # wrong. ~90s, dominated by densifying both operands across the grid so that
    # dense, not the checks' own predicate, is the reference.
    "test_contract_leg_pairing_834.py": "core",
    "test_ctm_convergence_meta.py": "core",
    # Scope of the implicit-AD CTM guard (#802 rows for #349/#350/#343). The
    # dispatch half is mocked at the engine boundary (microseconds); the one
    # numeric test converges three D=2 chi=8 C4v CTMs, ~4.5s of a ~5s file. It
    # gates a merge because the failure it prevents -- the guard regrowing to
    # cover paths it does not describe -- is a silent config-time refusal.
    "test_implicit_ad_guard_scope.py": "core",
    # Arnoldi spectral-radius estimator: the divergence precheck for the
    # explicit-AD CTM backward (``if rho >= threshold: raise``).  Pure linear
    # algebra on 3x3 matrices, ~1s for the file.  Bucketed out of
    # _UNBUCKETED_LEGACY because #828 was a live defect here -- a stale
    # duplicate under-reported rho on complex input, so the guard passed a
    # divergent adjoint -- and a guard that runs in no required job is not one.
    "test_arnoldi.py": "core",
    # #898: the CTM convergence criterion reads identically zero on a rank-1
    # corner, so a collapsed environment was certified as converged at ANY
    # max_iter, conv_tol and chi.  The unit half is pure array algebra --
    # milliseconds -- and belongs in the required gate because the defect is
    # *silence*: nothing failed, the loop just exited early and returned a
    # mean-field number.  The three end-to-end cases converge 2-site CTMs and
    # carry their own explicit ``@pytest.mark.slow``, which the rule below
    # honours by *withholding* this ``core``.
    "test_ctm_criterion_rank_blind_898.py": "core",
    # The eager-loop GMRES every root-implicit adjoint runs through (#731).
    # ~20s on dense matrices, no CTM.  ``core`` because the property it guards
    # is *structural* and invisible to every numerical test: put the loop back
    # under ``jit`` and the operator is compiled into a ``while_loop`` body
    # again, which took the symmetric adjoint to 8.63 GB against ~7 GB runners.
    # Every tolerance assertion in the suite stays green through that.
    "test_gmres_eager.py": "core",
    # The iPEPS AD guide against the code it documents (#808).  Pure markdown
    # parsing plus config validation, microseconds.  ``core`` because both
    # defects it pins were *silent*: the guide specified a Path 2 config that
    # raises ValueError before the first CTM sweep, and recommended against the
    # path the code runs by default.  Nothing else in the suite reads the docs.
    "test_docs_ad_paths_808.py": "core",
    # #800 item 2: the root-implicit quadrants against the production enlarged
    # corner.  Milliseconds, no CTM convergence.  ``core`` because every
    # endpoint here has dimension chi, so an endpoint swap is invisible to
    # shape and to a fixed-point test -- the #700/#702 shape -- and because the
    # file also pins that its own fixture is asymmetric enough to see one.
    "test_quadrant_endpoint_pairing_800.py": "core",
    # The adjoint-convergence gate on the DEFAULT iPEPS AD gradient path
    # (#801, first raised on #341).  An unconverged adjoint yields a gradient
    # that is wrong, finite, and indistinguishable downstream -- the exact
    # class of defect the required gate exists for.  Cheap: D=2, chi=4, and
    # the starved-budget cases converge in one iteration by construction.
    "test_adjoint_convergence_gate.py": "core",
    # The root-implicit gates' fail-closed comparison (#796 / #787).  Pure
    # unit tests on the shared predicate and the gauge_consistency reduction:
    # no CTM, milliseconds.  They guard a defect class that has now recurred
    # on four engines, so they belong in the required gate.
    "test_root_implicit_nan_gates.py": "core",
    # Wiring for ctm_ad_mode="root_implicit_symmetric" (#715 Phase 3).  Every
    # gap it pins is a *type* gap -- a SymmetricTensor parameter, a
    # SymmetricTensor gradient, a three-value return -- so the checks run
    # against a stub engine in milliseconds.  ``core`` because the failure
    # shape is a silent densification: the run would still converge to
    # something, just not to the symmetric problem the caller posed.  The one
    # end-to-end descent test carries its own ``slow``.
    "test_root_implicit_symmetric_wiring.py": "core",
    # The multisite clamped-residual gate (#784).  The rank-report half is one
    # SVD per coordinate and runs in the gate; the two end-to-end cases each
    # converge a 300-sweep CTM plus an adjoint solve and carry their own
    # explicit ``@pytest.mark.slow``, which the rule below honours by
    # *withholding* this ``core``; see ``pytest_collection_modifyitems``.
    "test_multisite_clamped_gate.py": "core",
    "test_ctm_tensor.py": "algorithm",
    # The invariant that keeps the 2x2 enlarged corner contraction-correct on
    # non-dual bonds (#834/#762).  It is a property of the *sweep*, not of the
    # flow convention, so nothing in the type system enforces it -- required
    # gate.  The D=3 case carries its own ``@pytest.mark.slow``.
    "test_enlarged_corner_flow_invariant_834.py": "core",
    "test_integration_regression.py": "algorithm",
    "test_krylov.py": "core",
    "test_tdvp.py": "algorithm",
    "test_ctm_tensor_c4v.py": "algorithm",
    # Root implicit AD for C4v CTMRG (#715 Phase 0): each parity test runs a
    # full CTM convergence plus finite-difference reruns.
    "test_ctm_c4v_root_implicit.py": "algorithm",
    "test_ctm_root_implicit_asym.py": "algorithm",
    # Multisite root-implicit AD (#715 Phase 2): the Appendix-F cell-shift
    # index tables are the whole risk of this phase -- a wrong shift still
    # yields a well-conditioned Jacobian and a plausible root, then a silently
    # wrong gradient (#700 / #702 failure shape).  Pinning them costs 0.14s for
    # 13 tests, so they belong in the required gate.  The root/gradient tests
    # in the same file each converge a 2x2 CTM plus finite-difference reruns
    # and carry their own explicit ``@pytest.mark.slow``, which the rule below
    # honours by *withholding* this ``core``; see ``pytest_collection_modifyitems``.
    # Previously absent from this table entirely, so all 33 tests were
    # deselected by ``-m core`` and never ran in a required check (#730).
    "test_ctm_root_implicit_multisite.py": "core",
    "test_ctm_root_implicit_sym_sectors.py": "core",
    # Symmetric root-implicit AD (#715 Phase 3): the structural half of this
    # file is cheap and belongs in the required gate, but the gradient tests
    # peak at 4.78 GB RSS and take ~180 s for the fixture alone.  That used to
    # be 8.63 GB -- XLA compiling the ~15k-equation block-sparse VJP inside
    # GMRES's ``lax.while_loop`` -- until #731 moved the loop out of the jit;
    # it now fits a ~7 GB runner, but it is still the largest single fixture in
    # the suite and the wall time alone keeps it out of a required gate.  They
    # carry their own explicit ``@pytest.mark.slow``, which the rule below
    # honours by *withholding* this ``core``; see ``pytest_collection_modifyitems``.
    "test_ctm_root_implicit_symmetric.py": "core",
    "test_ctm_truncation_error.py": "core",
    "test_ctm_paired.py": "algorithm",
    "test_ctm_python_loop.py": "algorithm",
    "test_rsvd.py": "core",
    "test_cbe.py": "algorithm",
    "test_lattice.py": "algorithm",
    "test_linalg.py": "core",
    "test_linalg_np.py": "core",
    "test_observables.py": "algorithm",
    "test_cbe_validation.py": "algorithm",
    "test_mps.py": "core",
    # FiniteMPS.norm() vs an explicit contraction (#819).  Neither backend had
    # such a test, which is how an exactly-zero symmetric norm survived.
    "test_mps_norm.py": "core",
    "test_blas_plan.py": "core",
    "test_padded_block_array.py": "core",
    "test_jit_sweep.py": "core",
    "test_padded_linalg.py": "core",
    "test_lanczos_np.py": "algorithm",
    "test_block_array.py": "core",
    # Cython/Python parity for the Lanczos block-array ops (#829).  dmrg.py
    # picks between the two at import time, so a semantic gap makes the same
    # input converge differently depending on whether the extension loaded --
    # and the extension is the default.  Pure numpy on 2-element blocks,
    # microseconds, so it belongs in the required gate.
    "test_cython_lanczos_block_keys.py": "core",
    "test_dmrg3s.py": "algorithm",
    "test_dmrg_cython.py": "algorithm",
    "test_ipeps_config.py": "core",
    "test_pess.py": "algorithm",
    "test_pess_ad.py": "algorithm",
    "test_pess_validation.py": "slow",
    "test_chi_auto_bump.py": "core",
    "test_ctm_env_pad.py": "core",
    "test_ipeps_chi_bump_integration.py": "algorithm",
    "test_ctm_convergence_random_iPEPS.py": "algorithm",
    "test_ipeps_grad_spike_guard.py": "slow",
    # Block-sparse stacked-contraction backend seam (#200 / #566): fast,
    # mechanism-level (small even-D fermionic tensors, no CTM convergence).
    "test_harness.py": "core",
    "test_stacked_view.py": "core",
    "test_stacked_contract.py": "core",
    "test_stacked_decomp.py": "core",
    "test_backend_dispatch.py": "core",
    "test_vjp_seam.py": "core",
    "test_vjp_seam_cutensor.py": "core",
    "test_stacked_tensor_dtype.py": "core",
    # GSPMD-sharded dense CTM (large-D rung 1): fast mechanism unit tests
    # (mesh specs, commit helpers, config default) gate every PR; the heavy
    # subprocess parity sweep (N=2,4 fake CPU devices) is slow-only.
    "test_ctm_sharding.py": "core",
    # These three test the orchestration helpers of driver scripts under
    # examples/, not the library: they path-load the example and exercise its
    # argument/schedule plumbing, and are jax-free by design.  The required
    # gate is for library behaviour, so a broken example should not block a
    # merge -- they still run in the full suite and on push to main.
    "test_showcase_scaling.py": "algorithm",
    "test_heisenberg_d4_chi_scaling.py": "algorithm",
    "test_heisenberg_d8_chi_scaling.py": "algorithm",
    "test_ctm_sharding_parity.py": "slow",
    # Chunked dense-CTM edge absorption (chunk x shard #632 Increment 1): fast
    # mechanism unit tests; collected as core so CI required checks run them.
    "test_ctm_chunked_absorb.py": "core",
    # gs_recipe="2x2" (the iPEPSConfig default) production guard (#676 / #702).
    # Only the forward-only cell-size consistency half is core; the end-to-end
    # physical anchor lives in test_ctm_recipe_2x2_production_correctness.py so
    # it does not inherit this marker and run in required CI (it is @slow).
    "test_ctm_recipe_2x2_consistency.py": "core",
    # #632 Increment 2 gate byproduct: forward-chunk grads through the (monolith)
    # implicit-AD backward stay exact. Tiny D=2 CPU value_and_grad parity.
    "test_ctm_chunk_backward_grad.py": "core",
    # #632 frontier benchmark (phase 1): tiny D=2 CPU value_and_grad finite +
    # path-guard checks. Fast; core so CI required checks run them.
    "test_frontier_probe.py": "core",
    # Stays out of the gate for the same reason as the scaling drivers above:
    # it only checks the `skip_reason` arithmetic of an examples/ benchmark
    # script.  (test_frontier_probe.py above is NOT in this group -- it runs a
    # real D=2 chi=6 CTM energy+gradient, so it is library behaviour.)
    "test_frontier_bench_guard.py": "algorithm",
    # Same group, same reason: bookkeeping helpers of the #672 re-derivation
    # driver under scripts/ (gate classification, cache identity, wall set,
    # device labelling), path-loaded and jax-free.  Not library behaviour, so
    # it does not gate a merge -- but it is what keeps a crashed gate cell from
    # being recorded as a confirmed rank-1 collapse (#747).
    "test_bench_672_driver_guard.py": "algorithm",
}


# ------------------------------------------------------------------ #
# Frozen debt: test files that predate the bucket guard.              #
# See tests/test_bucket_registry.py and issue #805.                    #
# This list may SHRINK, never grow.                                    #
# ------------------------------------------------------------------ #

_UNBUCKETED_LEGACY = {
    "test_ad_primitives_rank_aware.py",
    "test_apply_chi_bump.py",
    "test_architecture_imports.py",
    "test_block_sparse_ctm_ad.py",
    "test_c4v_reference_ad.py",
    "test_chi_ramp_chi_auto_bump_deprecation.py",
    "test_coarse_grain.py",
    "test_complex128_ad.py",
    "test_ctm_2x2_projector_symmetric.py",
    "test_ctm_670_symmetric_2x2.py",
    "test_ctm_674_fermionic_fused.py",
    "test_ctm_700_env_collapse.py",
    "test_ctm_chi_ramp.py",
    "test_ctm_compiled.py",
    "test_ctm_direction_dependent_bonds.py",
    "test_ctm_energy_implicit.py",
    "test_ctm_energy_implicit_chi_bump.py",
    "test_ctm_env_pad_chi_schedule.py",
    "test_ctm_explicit_tbptt.py",
    "test_ctm_honeycomb_ad.py",
    "test_ctm_honeycomb_convergence.py",
    "test_ctm_honeycomb_cross_path.py",
    "test_ctm_honeycomb_energy.py",
    "test_ctm_honeycomb_env.py",
    "test_ctm_honeycomb_forward.py",
    "test_ctm_honeycomb_init.py",
    "test_ctm_honeycomb_lukin_sotnikov.py",
    "test_ctm_honeycomb_moves.py",
    "test_ctm_honeycomb_projector.py",
    "test_ctm_honeycomb_safeguards.py",
    "test_ctm_implicit_warm_start_adjoint.py",
    "test_ctm_in_loop_bump_ad_paths.py",
    "test_ctm_in_loop_chi_bump.py",
    "test_ctm_loop_core.py",
    "test_ctm_multisite_2x2_contract.py",
    "test_ctm_projector.py",
    "test_ctm_recipe_2x2_production_correctness.py",
    "test_ctm_sharding_backward.py",
    "test_ctm_tensor_flow_flip.py",
    "test_ctm_tensor_init_rank1.py",
    "test_ctm_tensor_projector_2x2.py",
    "test_ctm_tensor_tiling.py",
    "test_gmres_lax.py",
    "test_hotrg_sharding.py",
    "test_ipeps_ad_adjoint_methods.py",
    "test_ipeps_ad_conv_criterion.py",
    "test_ipeps_ad_f3_fused_bwd.py",
    "test_ipeps_ad_history.py",
    "test_ipeps_ad_policy.py",
    "test_ipeps_checkpoint.py",
    "test_ipeps_checkpoint_resume.py",
    "test_ipeps_chi_adaptive_bump_unit.py",
    "test_ipeps_chi_bump_rollback_desync.py",
    "test_ipeps_chi_schedule_wiring.py",
    "test_ipeps_config_chi_ceiling_bailout.py",
    "test_ipeps_config_grad_spike.py",
    "test_ipeps_config_hz_max_iter.py",
    "test_ipeps_config_stall_recovery_retries.py",
    "test_ipeps_ctm_stall_recovery_cap.py",
    "test_ipeps_stall_recovery_cap.py",
    "test_ipeps_tree_dot.py",
    "test_ipeps_u1sz.py",
    "test_line_search.py",
    "test_line_search_auto.py",
    "test_lorentzian_eigh_kernel.py",
    "test_make_neighbors.py",
    "test_metric_precond.py",
    "test_optimize_gs_ad_chi_schedule_shim.py",
    "test_optimize_gs_ad_chi_schedule_unified.py",
    "test_pess_3site_multisite_encoding.py",
    "test_pess_3site_multisite_rdm_invariants.py",
    "test_pess_3site_multisite_wavefunction.py",
    "test_pess_ad_honeycomb.py",
    "test_pess_local_energy.py",
    "test_profiler_u1sz_arm.py",
    "test_projector_backward_dispatch.py",
    "test_reduced_corner_qr.py",
    "test_regularized_qr.py",
    "test_regularized_svd.py",
    "test_split_ctm_chi_frozen_726.py",
    "test_split_ctm_doublelayer_projector.py",
    "test_split_ctm_energy_gauge.py",
    "test_split_ctm_fuse_flag.py",
    "test_split_ctm_large_d_memory.py",
    "test_split_ctm_production_correctness.py",
    "test_sublattice_rotation.py",
    "test_svd_adjoint_fd_750.py",
    "test_symmetric_custom_vjp.py",
    "test_truncated_lowrank_svd.py",
    "test_tuning_registry.py",
    "test_u1sz_defrag_prototype_610.py",
    "test_varipeps_compare.py",
    "test_varipeps_compare_payload.py",
    "test_varipeps_compare_su.py",
}


def pytest_collection_modifyitems(items):
    """Apply the file's bucket marker, except where an explicit ``slow`` wins.

    ``core`` is the *required* CI gate (``pytest -m core``, see
    ``.github/workflows/ci.yml``), and ``-m core`` is a positive selector: a
    test carrying both ``core`` and ``slow`` is selected by it.  So for a file
    mapped to ``core``, an explicit ``@pytest.mark.slow`` on a test has to
    *withhold* the file marker or it means nothing — the test would run in the
    required gate anyway, which is how an 8.63 GB block-sparse AD test came to
    be pointed at GitHub's ~7 GB Linux runners (#731; that fixture is 4.78 GB
    since the adjoint loop came out of the jit, but it is still the largest
    here and still marked).

    The ``algorithm`` files are deliberately left alone.  Their coexistence of
    ``algorithm`` + explicit ``slow`` is already correct and already relied on
    (see the ``test_split_ctm_2site_symmetric.py`` note above): the two
    non-core buckets select ``not core and not slow``, so a slow test in an
    ``algorithm`` file is excluded from them by the ``slow`` half regardless.
    Measured, this rule moves exactly the two tests it is meant to move —
    ``-m "core and slow"`` collected 2 items before it, both in
    ``test_ctm_root_implicit_symmetric.py``, against 17 for
    ``-m "algorithm and slow"``.
    """
    for item in items:
        marker = _FILE_MARKERS.get(item.path.name)
        if marker is None:
            continue
        if marker == "core" and item.get_closest_marker("slow") is not None:
            continue
        item.add_marker(getattr(pytest.mark, marker))


# ------------------------------------------------------------------ #
# Cap memory growth across tests by clearing the JAX in-memory       #
# compile cache once peak RSS crosses a threshold.                    #
#                                                                    #
# Why this matters: each AD/CTM test compiles a distinct JAX/XLA     #
# variant (different chi, charges, optimizer config, ...). The JIT   #
# cache retains compiled artifacts in memory across tests. With the  #
# fast-ipeps bucket's ~30+ AD tests this snowballs past 18 GB locally,#
# OOM-killing GH's 7 GB Linux runners (manifesting as "the runner    #
# lost communication with the server" mid-bucket).                   #
#                                                                    #
# ``jax.clear_caches()`` releases the Python-side cache references   #
# so XLA can reuse buffer slots; combined with ``gc.collect()`` it   #
# bounds peak RSS to a single test's working set (~5 GB) instead of  #
# accumulating. The persistent on-disk cache (``~/.cache/jax``, see  #
# ``tenax/__init__.py``) is preserved, so cross-session reuse still  #
# works.                                                             #
#                                                                    #
# Threshold gating: ``ru_maxrss`` is monotonic, so once a bucket's   #
# peak crosses the threshold we clear from then on, but cheap        #
# buckets that never cross stay fast forever.                        #
#                                                                    #
# Threshold tuning (measured locally, 735-test ``-m core`` run):     #
#                                                                    #
#   threshold     core peak       core time   fast-ipeps?            #
#   2 GB          3.0 GB          12m51s      OK (cleared)           #
#   4 GB          4.8 GB           9m15s      OK (cleared)           #
#   6 GB          5.6 GB           2m53s      OK (cleared @ t#44)    #
#   ∞ (no clear)  5.6 GB           2m54s      18 GB → OOM            #
#                                                                    #
# 6 GB is the sweet spot: above ``-m core`` natural peak (5.6 GB) so #
# the cheap bucket stays at baseline speed, but below the 7 GB GH    #
# Linux runner limit so fast-ipeps still engages clearing in time    #
# (its first AD-heavy test crosses 6 GB at iteration ~44, well       #
# before the without-clearing snowball reaches 7 GB OOM).            #
# ------------------------------------------------------------------ #

_RSS_CLEAR_THRESHOLD_MB = 6000


def _peak_rss_mb() -> float:
    """Peak RSS used by this process, in MB.

    ``ru_maxrss`` units differ by platform (BSD convention): macOS reports
    bytes, Linux reports KB.
    """
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    if _peak_rss_mb() < _RSS_CLEAR_THRESHOLD_MB:
        return

    import gc

    jax.clear_caches()
    gc.collect()


# ------------------------------------------------------------------ #
# Symmetry fixtures                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1():
    return U1Symmetry()


@pytest.fixture
def z2():
    return ZnSymmetry(2)


@pytest.fixture
def z3():
    return ZnSymmetry(3)


# ------------------------------------------------------------------ #
# Random key fixture                                                   #
# ------------------------------------------------------------------ #


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng2():
    return jax.random.PRNGKey(99)


# ------------------------------------------------------------------ #
# TensorIndex fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_charges_3(u1):
    """U(1) charges [-1, 0, 1] — typical for spin-1 or bond dim 3."""
    return np.array([-1, 0, 1], dtype=np.int32)


@pytest.fixture
def u1_charges_2(u1):
    """U(1) charges [-1, 1] — typical for spin-1/2."""
    return np.array([-1, 1], dtype=np.int32)


@pytest.fixture
def idx_in_3(u1, u1_charges_3):
    """U(1) IN index with charges [-1, 0, 1], label='left'."""
    return TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="left")


@pytest.fixture
def idx_out_3(u1, u1_charges_3):
    """U(1) OUT index with dual charges [1, 0, -1], label='right'.
    This is the proper dual of idx_in_3.
    """
    return TensorIndex.from_charges(
        u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="right"
    )


@pytest.fixture
def u1_index_pair(idx_in_3, idx_out_3):
    """A compatible pair of U(1) indices (IN and its dual OUT)."""
    return idx_in_3, idx_out_3


# ------------------------------------------------------------------ #
# DenseTensor fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_dense_matrix(u1, rng):
    """A 3x3 DenseTensor (matrix) with U(1) indices."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3, 3))
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
        TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
    )
    return DenseTensor(data, indices)


@pytest.fixture
def dense_vector(u1, rng):
    """A 3-element DenseTensor (vector) with U(1) index."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3,))
    idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="vec")
    return DenseTensor(data, (idx,))


# ------------------------------------------------------------------ #
# SymmetricTensor fixtures                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_sym_tensor_2leg(u1, rng):
    """2-leg U(1)-symmetric tensor: IN x OUT, charges [-1, 0, 1]."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
        TensorIndex.from_charges(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_3leg(u1, rng):
    """3-leg U(1)-symmetric tensor: phys x left x right."""
    phys_c = np.array([-1, 1], dtype=np.int32)
    virt_c = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(u1, virt_c, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(u1, u1.dual(virt_c), FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_pair(u1, rng, rng2):
    """A pair of 3-leg U(1)-symmetric tensors that can be contracted on 'bond'.

    Both tensors use the SAME charge array for the shared bond leg with
    opposite flow directions (OUT for A, IN for B). Same charge array means
    position i in A's bond and position i in B's bond store the same charge
    value, so dense einsum contraction works correctly (no ordering mismatch).
    """
    phys_c = np.array([-1, 1], dtype=np.int32)
    bond_c = np.array(
        [-1, 0, 1], dtype=np.int32
    )  # same charges for both ends of shared bond

    indices_A = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="p0"),
        TensorIndex.from_charges(u1, bond_c, FlowDirection.IN, label="bond_left"),
        TensorIndex.from_charges(
            u1, bond_c, FlowDirection.OUT, label="bond"
        ),  # OUT end of shared bond
    )
    indices_B = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="p1"),
        TensorIndex.from_charges(
            u1, bond_c, FlowDirection.IN, label="bond"
        ),  # IN end of shared bond (same charges)
        TensorIndex.from_charges(u1, bond_c, FlowDirection.OUT, label="bond_right"),
    )
    A = SymmetricTensor.random_normal(indices_A, rng)
    B = SymmetricTensor.random_normal(indices_B, rng2)
    return A, B
