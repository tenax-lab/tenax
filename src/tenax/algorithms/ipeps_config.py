"""iPEPS configuration dataclasses and environment NamedTuples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import jax

from tenax.core.lattice import Lattice


@dataclass
class CTMConfig:
    """Configuration for CTM environment computation.

    Attributes:
        chi:                Bond dimension of CTM environment tensors.
        max_iter:           Maximum CTM iterations before declaring convergence.
        conv_tol:           Convergence tolerance (based on singular value change
                            between CTM iterations).
        renormalize:        Whether to renormalize environment tensors at each step
                            to prevent exponential growth (always recommended).
        ad_regularize_svd:  Use Lorentzian-regularized SVD backward pass in AD
                            optimization.  When True, the SVD custom VJP uses
                            broadening to prevent NaN from degenerate singular
                            values (Francuz et al., PRR 7, 013237).
        forward_gauge:      Gauge fix applied after each CTM sweep.  One of
                            ``"phase"`` (default), ``"qr"``, ``"sigma"``, or
                            ``"none"``.  Explicit user choice is preserved —
                            no silent promotion in AD paths.  See
                            ``docs/guide/algorithms/ipeps_ad_paths.md`` for
                            the full mode matrix and benchmarks.
        ad_backward_method: Backward method for the implicit-diff path.
                            ``"vjp"`` (default) is the regression-covered
                            Neumann-series backward.  ``"gmres"`` is
                            currently documented unstable (spectral radius
                            > 1 without tight sigma-gauge alignment) and
                            its regression test is marked ``xfail`` —
                            tracked by issue #292.  Prefer
                            ``ad_backward_method="vjp"`` (the default)
                            until GMRES is stabilized.
        chi_auto_bump:      Reactive auto-χ_E bump (variPEPS §2.8.2).  When
                            ``True``, the optimizer raises ``chi`` by
                            ``chi_auto_bump_step`` between L-BFGS steps
                            whenever the CTM truncation error exceeds
                            ``chi_auto_bump_eps``.  Mutually exclusive with
                            ``chi_ramp`` (a deterministic schedule).
                            Off by default.

                            For new code, prefer ``chi_schedule`` +
                            ``optimize_gs_ad_chi_schedule`` with
                            convergence-triggered ramping (#455);
                            ``chi_auto_bump`` is retained as an
                            orthogonal CTM-truncation sentinel for the
                            case where the optimizer is making progress
                            but ε_T indicates CTM under-resolution.
                            These two mechanisms compose (reactive fires
                            first, scheduled second).
        chi_auto_bump_eps:  Truncation-error threshold that triggers an
                            auto-χ bump.  Default ``1e-5`` follows
                            variPEPS §2.8.2.
        chi_auto_bump_step: Additive increment applied to ``chi`` on each
                            bump event.  Must be a positive integer.
                            Default ``2``.
        chi_max:            Hard ceiling on ``chi`` for the auto-bump
                            mechanism.  ``None`` means unbounded.  When
                            set, must satisfy ``chi_max >= chi``.
        fuse_virtual_legs:  When ``True`` (default) the CTM uses the fused
                            double-layer tensor.  When ``False`` the single-site
                            (``recipe="1x1"``) AD path uses the split ket/bra
                            double layer (χ²·D⁴ memory).  Only the single-site
                            path is supported; other lattices raise
                            ``NotImplementedError`` (#463 Phase 2).
    """

    chi: int = 20
    chi_ramp: list[tuple[int, int | None]] | None = None
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True
    projector_method: str = "svd"  # "svd" (Fishman, default), "eigh", or "qr".
    # "qr" on the dense single-site (recipe="1x1") path runs the reduced-corner
    # QR-CTMRG isometry (arXiv:2505.00494, Zhang/Yang/Corboz); forward-only.
    # SymmetricTensor/block-sparse "qr" and the AD backward are later phases.
    min_iter: int = 10  # minimum CTM sweeps before checking convergence
    qr_warmup_steps: int = 3  # eigh warm-up iterations before QR kicks in
    chi_I: int | None = None  # interlayer bond dim for split-CTMRG; None => chi_I = chi
    fuse_virtual_legs: bool = True  # True: fused double-layer (χ²·D⁶, default).
    # False: single-site (recipe="1x1") split ket/bra double layer (χ²·D⁴).
    # #463 Phase 2 — multisite/2-site/c4v/honeycomb/PESS raise NotImplementedError
    # when False (no multisite split forward yet).
    ad_regularize_svd: bool = True  # use Lorentzian-regularized SVD backward in AD
    gmres_precondition: bool = (
        False  # diagonal scaling preconditioner for GMRES backward (experimental)
    )
    ad_backward_method: str = "vjp"  # "vjp" (iterative VJP) or "gmres" (xfail — #292)
    gmres_tol: float = 1e-6  # tolerance for GMRES backward solve
    gmres_restart: int = 20  # Krylov dimension for GMRES (no outer restarts)
    gmres_maxiter: int = 200  # max total GMRES iterations (outer budget)
    # Adjoint solver for the implicit-AD CTM backward.
    #   "fixed_point" (default) — Python-loop ``λ_{k+1} = b + J^T λ_k``.
    #     Mirrors variPEPS's ``_ctmrg_rev_workhorse``.  Converges
    #     geometrically when ρ(J^T) < 1, which the Arnoldi precheck
    #     guarantees.  Each iteration is one cached ``_jit_apply_Jt``
    #     matvec — no Krylov subspace, no per-iter compile cost.
    #   "gmres" — eager Krylov solve via ``gmres_pytree_jax``.  Use as a
    #     safety opt-out when the fixed-point loop diverges.
    # Both paths reuse ``gmres_tol`` (residual / step tolerance) and
    # ``gmres_maxiter`` (step cap).  The fixed-point loop additionally
    # falls back to GMRES in-loop when ``diff > prev_diff`` after step 5.
    adjoint_method: Literal["fixed_point", "gmres"] = "fixed_point"
    ctm_conv_method: str = "elementwise"  # "elementwise" or "sv" (singular value)
    # forward_gauge: "phase" (default — Frobenius-norm phase fix per CTM
    # absorption; works for both implicit and explicit AD, 1-site and
    # 2-site).  "sigma" (transfer-matrix eigenvector alignment, 1-site
    # only), "qr" (legacy), or "none" (diagnostic).  No silent promotion
    # — explicit user choice is preserved.  See ipeps_ad_paths.md.
    forward_gauge: str = "phase"
    # Optional reference-mode implicit AD mode (App. C-F) for dense 1-site C4v.
    ctm_ad_mode: str | None = None  # None or "c4v_reference"
    adjoint_solver: str = "bicgstab"  # "bicgstab" or "gmres"
    adjoint_maxiter: int = 50
    adjoint_tol: float = 1e-8
    adjoint_degen_tol: float = 1e-10
    adjoint_diag_shift: float = 1e-12
    # Tikhonov damping applied to the linear adjoint system:
    #   ((I - J^T) + tau I) lambda = g
    # Near a CTM fixed point, J has eigenvalues approaching 1 along the
    # slowest modes, making (I - J^T) near-singular; a small positive tau
    # biases gradients slightly but keeps the Krylov solve stable.
    #
    # Default 1e-6 is a numerical-robustness floor: smaller than the
    # target residual implied by ``adjoint_tol`` (×10 fudge), so it can't
    # bias gradients beyond the tolerance we already accept in the solve,
    # but large enough to prevent Krylov stalls near the physical GS.
    # Set to 0.0 for a strictly exact adjoint; increase to 1e-4…1e-3 when
    # the outer optimizer approaches a well-converged ground state and the
    # solve otherwise fails to reach ``adjoint_tol``.
    #
    # The Krylov-breakdown-near-GS phenomenon and the use of regularization
    # to stabilize implicit-diff CTM are discussed in:
    #   - Liao, Liu, Wang, Xiang, "Differentiable Programming Tensor Networks",
    #     Phys. Rev. X 9, 031041 (2019). (arXiv:1903.09650)
    #   - Francuz, Schmoll, Rizzi, Eisert, Naumann, "Stable and efficient
    #     differentiation of tensor network algorithms",
    #     Phys. Rev. Research 7, 013237 (2025). (arXiv:2311.11894)
    #   - Naumann, Weerda, Rizzi, Eisert, Schmoll, "variPEPS — a versatile
    #     tensor network library for variational ground state simulations in
    #     two spatial dimensions", SciPost Phys. Codebases (arXiv:2308.12358).
    adjoint_tikhonov: float = 1e-6
    # Backward pass used for the eigh projector inside AD.
    #   "auto"       -> default; ``optimize_gs_ad`` promotes to "lorentzian"
    #                   when ``gs_implicit_ad=False`` and the effective
    #                   projector is "eigh"; otherwise the effective value is
    #                   "standard".  Mirrors the ``forward_gauge`` pattern.
    #   "standard"   -> force the existing ``regularized_eigh`` backward.
    #                   Never auto-promoted away from this value.
    #   "lorentzian" -> force the truncated-eigh Lorentzian backward from
    #                   ``_lorentzian_eigh.py`` (plan PR #315, Task 2).
    #                   Honored even when ``projector_method != "eigh"``, in
    #                   which case the forward doesn't use eigh at all and
    #                   the flag is a no-op.
    # Kept at the end of the dataclass to preserve positional-argument
    # compatibility for callers that construct CTMConfig positionally.
    projector_backward: Literal["auto", "standard", "lorentzian"] = "auto"
    adjoint_arnoldi_precheck: bool = True
    adjoint_arnoldi_threshold: float = 5.0
    # variPEPS §2.8.2 reactive auto-χ_E bump.  When enabled, the optimizer
    # raises ``chi`` between L-BFGS steps if the CTM truncation error exceeds
    # ``chi_auto_bump_eps``.  Mutually exclusive with ``chi_ramp`` (which is
    # a deterministic schedule).  Off by default.
    chi_auto_bump: bool = False
    chi_auto_bump_eps: float = 1e-5
    chi_auto_bump_step: int = 2
    # variPEPS §2.8.2 indicator choice (Tenax extension).  ``"eps_T"``
    # (default, backward-compatible) reads ``info.max_truncation_error``
    # (discarded SV tail mass).  ``"norm_smallest_S"`` reads
    # ``info.max_smallest_S`` (smallest *retained* SV / largest), matching
    # variPEPS's literal trigger.  The two metrics produce different bump
    # cadences for the same nominal threshold.  Currently honoured on the
    # 2-site bipartite path; other paths fall back to ``eps_T``.
    chi_auto_bump_metric: str = "eps_T"
    chi_max: int | None = None
    # variPEPS-style in-CTM χ-bump (Issue #492).  When enabled,
    # ``python_loop_ctm_converge`` grows ``chi`` *during* CTM convergence
    # by ``ctmrg_heuristic_increase_chi_step_size`` whenever the running
    # max ``norm_smallest_S`` (smallest kept SV / largest, per projector
    # SVD) exceeds ``ctmrg_heuristic_increase_chi_threshold``.  Capped
    # at ``chi_max``.  Unlike ``chi_auto_bump`` (end-of-outer-step), this
    # never returns a half-formed env to the AD optimizer — every gradient
    # is computed at a converged CTM fixed point.  Off by default; mutually
    # exclusive with ``chi_ramp`` (deterministic schedule).
    ctmrg_heuristic_increase_chi: bool = False
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6
    ctmrg_heuristic_increase_chi_step_size: int = 2
    # Early-bail when the running minimum of the CTM convergence metric
    # has not improved for ``plateau_patience`` consecutive iterations.
    # Default ``20`` is a stop-loss against the known SU/random-init CTM
    # plateau (issue #425/#426, memory
    # ``project_tenax_ctm_doesnt_converge_random_init``).  A healthy
    # converging run never accumulates 20 non-improving iters because
    # each better ``best_diff`` resets the counter, so the default does
    # not interfere with normal convergence.  Set to ``None`` to restore
    # the pre-2026-05-11 "run to ``max_iter``" behavior.
    #
    # **AD-path note**: the implicit-AD backward solves the fixed-point
    # adjoint ``(I - J^T) λ = ∂L/∂env`` around the returned env, which is
    # only well-defined when env is a true fixed point.  In early AD the
    # CTM plateaus regardless (the env is *not* a fixed point even after
    # ``max_iter`` sweeps — see #425/#426), so the implicit-function
    # premise is already broken and gradients are approximate either way.
    # variPEPS exploits the same trade-off via
    # ``optimizer_ctmrg_preconverged_eps=1e-5`` (loose CTM during AD)
    # which is why its AD inner loop is much faster than a tight
    # fixed-point CTM.  Keeping ``plateau_patience`` finite during AD
    # matches that pragmatism.  Drop to ``None`` at the final chi stage
    # (or when the CTM actually converges) for strict variational
    # gradients.
    plateau_patience: int | None = 20
    # Issue #503: line-search φ probes don't need a fully converged env.
    # When set, these override ``max_iter`` / ``conv_tol`` only on the
    # forward-only ``loss_fn_fwd`` path used by ``hager_zhang_line_search``
    # and ``_backtracking_line_search``.  variPEPS caps probe sweeps at
    # 10-15 to bound the worst-case HZ probe cost.  ``None`` (default)
    # preserves prior behavior — probes use the same ``max_iter`` /
    # ``conv_tol`` as the accepted-step CTM converge.
    #
    # **Bias note:** an aggressively-capped probe can produce an
    # under-converged energy estimate at the trial α.  The optimizer's
    # ``best_energy`` and ``best_params`` are still maintained from the
    # full-CTM ``loss_fn`` (the value returned by ``jax.value_and_grad``
    # on the next iteration), so probe bias never corrupts the rolling
    # best.  But the per-step accept gate ``_is_real_decrease(f_alpha,
    # energy_float)`` uses the probe-capped ``f_alpha`` — keep
    # ``probe_max_iter >= max_iter // 2`` (or set ``probe_conv_tol`` no
    # looser than ~10 × ``conv_tol``) to keep the bias below the noise
    # floor.  Appended at the end of the dataclass to preserve positional
    # CTMConfig ABI.
    probe_max_iter: int | None = None
    probe_conv_tol: float | None = None
    # Optional ``jax.sharding.Mesh`` for the GSPMD-sharded dense CTM-AD path
    # (large-D, single-node multi-GPU; #632 rung 1 forward + rung 2 backward).
    # When set, the implicit-AD forward commits env tensors to a D²-sharded
    # layout and GSPMD partitions both the forward CTM and the implicit-AD
    # backward across the mesh; per-device peak memory drops ~N^(1/6) in D
    # (dense CTM memory ~D⁶). ``None`` (default) → single-device, bit-for-bit
    # unchanged. Build a mesh via ``tenax.algorithms.ctm_sharding.build_ctm_mesh``.
    # Appended at the end of the dataclass to preserve positional CTMConfig ABI.
    device_mesh: jax.sharding.Mesh | None = None
    # Chunk the χ²·D⁶ edge absorption over the boundary-χ axis via
    # ``lax.map(batch_size=ctm_chunk_size)`` (1×1 recipe, dense envs only).
    # Lowers per-device peak memory ≈÷K at large D and composes with
    # ``device_mesh`` (≈÷(N·K)). ``None`` (default) → single monolithic
    # contraction (byte-for-byte unchanged). See the #632 chunk×shard gate.
    # Appended at the end of the dataclass to preserve positional CTMConfig ABI.
    ctm_chunk_size: int | None = None

    def __post_init__(self):
        valid_modes = {None, "c4v_reference"}
        if self.ctm_ad_mode not in valid_modes:
            raise ValueError(
                f"ctm_ad_mode must be one of {valid_modes}, got {self.ctm_ad_mode!r}"
            )
        valid_solvers = {"bicgstab", "gmres"}
        if self.adjoint_solver not in valid_solvers:
            raise ValueError(
                f"adjoint_solver must be one of {valid_solvers}, "
                f"got {self.adjoint_solver!r}"
            )
        valid_projector_backward = {"auto", "standard", "lorentzian"}
        if self.projector_backward not in valid_projector_backward:
            raise ValueError(
                f"projector_backward must be one of {valid_projector_backward}, "
                f"got {self.projector_backward!r}"
            )
        valid_adjoint_methods = {"fixed_point", "gmres"}
        if self.adjoint_method not in valid_adjoint_methods:
            raise ValueError(
                f"adjoint_method must be one of {valid_adjoint_methods}, "
                f"got {self.adjoint_method!r}"
            )
        # Phase-1 deprecation (#512): both the scheduled ``chi_ramp`` and
        # the end-of-outer-step ``chi_auto_bump`` zero-pad the env between
        # L-BFGS steps and hand a non-physical partial env back to the
        # optimizer, which can descend to a ghost minimum below the
        # variational floor.  ``ctmrg_heuristic_increase_chi`` (variPEPS
        # §2.8.2; PR #514) grows chi inside CTM convergence so the env is
        # always converged at the new chi before the optimizer sees it.
        import warnings

        if self.chi_ramp is not None:
            warnings.warn(
                "chi_ramp is deprecated and will be removed in a future "
                "release. Use ctmrg_heuristic_increase_chi=True with "
                "chi_max set instead — it grows chi inside CTM "
                "convergence, avoiding the zero-padded-env cliff-edge "
                "artifact that chi_ramp introduces between L-BFGS steps. "
                "Migration: replace chi_ramp=[(9, 20), (12, 20), (16, 20)] "
                "with chi=9, chi_max=16, ctmrg_heuristic_increase_chi=True. "
                "See issue #512.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.chi_auto_bump:
            warnings.warn(
                "chi_auto_bump (end-of-outer-step reactive bump) is "
                "deprecated and will be removed in a future release. Use "
                "ctmrg_heuristic_increase_chi=True with chi_max set "
                "instead — it grows chi inside CTM convergence, avoiding "
                "the zero-padded-env cliff-edge artifact that the "
                "end-of-outer-step bump introduces between L-BFGS steps. "
                "See issue #512.",
                DeprecationWarning,
                stacklevel=2,
            )

        # variPEPS §2.8.2 auto-bump validation.
        if self.chi_auto_bump and self.chi_ramp is not None:
            raise ValueError(
                "chi_auto_bump and chi_ramp are mutually exclusive: "
                "chi_ramp is a deterministic schedule, chi_auto_bump is reactive"
            )
        if self.chi_auto_bump and self.chi_auto_bump_step <= 0:
            raise ValueError(
                f"chi_auto_bump_step must be a positive integer, got {self.chi_auto_bump_step}"
            )
        if self.chi_auto_bump_metric not in ("eps_T", "norm_smallest_S"):
            raise ValueError(
                "chi_auto_bump_metric must be 'eps_T' or 'norm_smallest_S', "
                f"got {self.chi_auto_bump_metric!r}"
            )
        if self.chi_max is not None and self.chi_max < self.chi:
            raise ValueError(f"chi_max ({self.chi_max}) must be >= chi ({self.chi})")
        # Issue #492 in-CTM χ-bump validation.
        if self.ctmrg_heuristic_increase_chi and self.chi_ramp is not None:
            raise ValueError(
                "ctmrg_heuristic_increase_chi and chi_ramp are mutually exclusive: "
                "chi_ramp is a deterministic optimizer-side schedule, "
                "ctmrg_heuristic_increase_chi is reactive inside CTM convergence"
            )
        # In-CTM bump and end-of-outer-step chi_auto_bump cannot coexist:
        # the in-CTM bump can raise chi (e.g. 2→8) inside a single CTM
        # call, but ``_maybe_bump_chi`` still reads stale ``ctm_cfg.chi``
        # (still 2) and tries to pad to chi+step (4), which is smaller
        # than the already-grown 8 → ``pad_dense_env_chi`` raises
        # ``chi_new < chi_old`` at runtime.  Reject the combination up
        # front so users get a clear error at config construction.
        if self.ctmrg_heuristic_increase_chi and self.chi_auto_bump:
            raise ValueError(
                "ctmrg_heuristic_increase_chi and chi_auto_bump are mutually "
                "exclusive: both grow chi, but the in-CTM bump operates "
                "inside CTM convergence while chi_auto_bump operates between "
                "L-BFGS steps.  Pick one — typically "
                "ctmrg_heuristic_increase_chi=True is the recommended path "
                "(see Issue #492)."
            )
        if (
            self.ctmrg_heuristic_increase_chi
            and self.ctmrg_heuristic_increase_chi_step_size <= 0
        ):
            raise ValueError(
                "ctmrg_heuristic_increase_chi_step_size must be a positive integer, "
                f"got {self.ctmrg_heuristic_increase_chi_step_size}"
            )
        # ctmrg_heuristic_increase_chi without an explicit chi_max ceiling
        # would be a silent no-op (chi_max_eff defaults to chi → growth
        # guard ``chi_current < chi_max_eff`` is always False).  Require
        # an explicit ceiling so users get a clear error instead.
        if self.ctmrg_heuristic_increase_chi and self.chi_max is None:
            raise ValueError(
                "ctmrg_heuristic_increase_chi=True requires chi_max to be set; "
                "without an explicit ceiling the in-CTM bump would silently "
                "no-op (chi can never grow above its initial value). "
                "Set chi_max to the largest chi you want CTM to grow into."
            )


@dataclass
class iPEPSConfig:
    """Configuration for iPEPS simple update optimization.

    Attributes:
        max_bond_dim:          PEPS virtual bond dimension D.
        num_imaginary_steps:   Number of imaginary time evolution steps.
        dt:                    Imaginary time step size.
        ctm:                   CTM configuration for environment computation.
        svd_trunc_err:         SVD truncation error for simple update.
        gate_order:            Order of bond updates: "sequential" or "random".
        su_init:               If True, ``optimize_gs_ad`` initializes the site
                               tensor via simple update (``ipeps()``) instead of
                               random initialization.  Ignored when ``A_init``
                               is provided explicitly.
        gs_verbose:            If True, print AD optimization progress.
        gs_log_interval:       Print every N AD steps when ``gs_verbose`` is
                               enabled. The first and final steps are always
                               logged.
        gs_implicit_ad:        Use implicit differentiation through the
                               CTM fixed-point equation instead of
                               differentiating through unrolled CTM sweeps.
                               ``True`` by default: the implicit-diff path
                               (iterative VJP backward) is the recommended
                               AD path.  ``False`` opts into explicit AD
                               (backprop through unrolled CTM sweeps).
                               ``ctm.forward_gauge`` defaults to ``"phase"``
                               for both paths — there is no silent gauge
                               promotion, and the implicit path validates
                               ``forward_gauge="phase"``.  See
                               ``docs/guide/algorithms/ipeps_ad_paths.md``.
        gs_ctm_conv_tol_schedule:
                               Optional ramp for the CTM convergence
                               tolerance across AD optimization steps. A list
                               of ``(step_fraction, conv_tol)`` pairs: at each
                               AD step the optimizer looks up the tolerance
                               corresponding to ``step_idx / gs_num_steps``
                               and rebuilds the CTM config accordingly.
                               ``None`` (default) uses ``ctm.conv_tol``
                               throughout.  This is an advanced tuning knob
                               for large-chi runs where an initially loose
                               CTM is enough to start moving.
        gs_ctm_max_iter_schedule:
                               Optional ramp for ``ctm.max_iter`` across
                               AD optimization steps (issue #507).  Same
                               ``(step_fraction, value)`` schema as
                               ``gs_ctm_conv_tol_schedule``.  Caps the
                               accepted-step CTM converge — late steps
                               where the env is barely changing finish
                               in ~10-15 sweeps instead of the
                               early-descent budget.  Independent of
                               ``CTMConfig.probe_max_iter`` (issue #503),
                               which caps HZ probes.  ``None`` (default)
                               uses ``ctm.max_iter`` throughout.
        gs_chi_schedule_steps: Outer-loop χ schedule for
                               ``optimize_gs_ad_chi_schedule`` (#453 / #455).
                               List of ``(target_chi, max_steps_in_stage)``
                               pairs; ``None`` disables the schedule.
                               Each pair is one stage: "run up to
                               max_steps_in_stage optimizer iterations
                               at logical chi = target_chi, then advance".
                               Normally set by
                               ``optimize_gs_ad_chi_schedule`` internally;
                               users should not set it directly.
        gs_stall_recovery:     Stall-recovery mode for line-search failures
                               (issue #298).  ``"noise"`` injects a Frobenius
                               perturbation (legacy 1-site C4v path);
                               ``"reset"`` clears L-BFGS ``(s, y)`` history
                               and rolls back to ``best_params`` (variPEPS
                               style).  ``None`` (default) lets
                               ``optimize_gs_ad`` pick per dispatcher:
                               ``"noise"`` for 1-site, ``"reset"`` for 2-site.
        gs_stall_recovery_retries:
                               Maximum consecutive resets allowed on the
                               ``"reset"`` recovery path before the
                               optimizer exits with ``best_params``.
                               Analogous to ``gs_noise_recovery_retries``
                               for the ``"noise"`` path.  Default ``5``
                               matches variPEPS's
                               ``optimizer_random_noise_max_retries``.
                               (issue #454)
        gs_energy_floor:       Optional variational sanity floor on in-loop
                               best-state tracking (issue #298).  Candidate
                               energies strictly below this value are
                               rejected as non-variational CTM artifacts.
                               ``None`` (default) disables the check.
    """

    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = field(default_factory=CTMConfig)
    svd_trunc_err: float | None = None
    gate_order: str = "sequential"
    unit_cell: str | Lattice = "1x1"  # "1x1", "2site", or Lattice(...)
    # AD ground-state optimization settings
    gs_optimizer: str = "lbfgs"  # "lbfgs", "cg", or "adam"
    gs_learning_rate: float = 1e-3
    gs_num_steps: int = 200
    gs_conv_tol: float = 1e-8
    # Outer convergence criterion for ``optimize_gs_ad``. ``"dE"`` (current
    # legacy default, **deprecated** — see ``__post_init__``) exits when
    # ``|E_step - E_step-1| < gs_conv_tol`` — this is variationally fragile
    # near flat minima or right after a stall recovery, where ``|dE|`` can
    # underflow ``gs_conv_tol`` while the gradient is still large.
    # ``"grad_norm"`` matches variPEPS by exiting when
    # ``||grad E||_2 < gs_grad_norm_tol`` (a true stationarity test); it
    # will become the default in a future release. ``"both"`` requires both
    # to hold simultaneously (most conservative). See issue #448.
    gs_conv_criterion: Literal["dE", "grad_norm", "both"] = "dE"
    # Gradient-norm tolerance used by the ``"grad_norm"`` and ``"both"``
    # criteria. The default ``1e-5`` matches variPEPS
    # ``optimizer_convergence_eps``.
    gs_grad_norm_tol: float = 1e-5
    # Outer-loop χ schedule for ``optimize_gs_ad_chi_schedule``
    # (#453 / #455).  List of ``(target_chi, max_steps_in_stage)``
    # pairs.  Each pair specifies one stage: at most
    # ``max_steps_in_stage`` optimizer iterations at logical
    # chi = ``target_chi``, then advance to the next stage.  Envs
    # are padded to ``ctm.chi_max`` from step 1, so the JIT-compiled
    # kernels never see a shape change.
    #
    # ``None`` (default) means the schedule mechanism is disabled and
    # the inner optimizer uses ``ctm.chi`` throughout.
    # ``optimize_gs_ad_chi_schedule`` sets this field internally
    # (#455 PR 1: passes ``chi_schedule`` through directly without
    # cumulative conversion).
    gs_chi_schedule_steps: list[tuple[int, int]] | None = None
    gs_verbose: bool = False
    gs_log_interval: int = 10
    gs_max_grad_norm: float = 1.0  # gradient clipping (max global norm)
    # Reject a step whose new ``||grad||_2`` exceeds
    # ``gs_grad_spike_ratio * max(median(recent), 1.0)``, where
    # ``recent`` is the last ``gs_grad_spike_window`` accepted gradient
    # norms.  A 10x+ jump in gradient magnitude is the cleanest
    # signature of a non-variational collapse in 2-site implicit-AD
    # (see project_v9_collapse_findings.md).  On rejection the
    # optimizer rolls back to ``best_params``, clears L-BFGS history,
    # and reinits the optimizer state.  ``None`` (default) disables
    # the guard.  Currently honoured only on the 2-site Tensor AD
    # path.
    gs_grad_spike_ratio: float | None = None
    gs_grad_spike_window: int = 5
    gs_line_search: bool | None = None  # None = auto (True for lbfgs/cg)
    gs_line_search_max_steps: int = 8
    # Line search method: "armijo", "hager_zhang", or "auto".  "auto" (#509)
    # picks Armijo backtracking when ``ctm.chi < gs_line_search_hz_chi_threshold``
    # and Hager-Zhang otherwise.  Rationale: iPEPS-AD's gradient at small chi
    # carries CTM truncation noise of order ε_T that frequently exceeds HZ's
    # strong-Wolfe curvature tolerance (σ ≈ 0.9), so HZ wastes
    # ``value_and_grad`` calls hunting an unachievable Wolfe point and falls
    # back to a poor α.  Armijo only checks sufficient decrease, dodging the
    # noise entirely at ~5-10× lower per-probe cost.  Resolved once at
    # optimizer entry from ``ctm.chi``, not adaptively rebound mid-run.
    gs_line_search_method: str = "hager_zhang"  # "armijo", "hager_zhang", "auto"
    # χ threshold under which "auto" picks Armijo; at or above, HZ.
    # variPEPS notes the 2-site bipartite implicit-AD path is variational only
    # at χ ≥ 16 (warning at ``ipeps_optimize.py``); below that the gradient is
    # too noisy for strong Wolfe.  16 mirrors that threshold.
    gs_line_search_hz_chi_threshold: int = 16
    # Hager-Zhang line search inner iteration cap (bracket + secant loops in
    # ``_line_search.hager_zhang_line_search``).  Each iteration triggers
    # one ``value_and_grad`` call, so on chi-saturated 2-site implicit-AD
    # one runaway probe at the default 40 burns ~10 minutes.  Lowering to
    # 15-20 caps the worst case at the cost of accepting more
    # ``converged=False`` line searches.  Honoured on the iPEPS 1-site /
    # 2-site / multisite Tensor AD paths.
    gs_hz_max_iter: int = 40
    # Chi-ceiling bail-out (variPEPS §2.8.2 mechanism).  When > 0 and the
    # CTM is at ``ctm.chi_max`` AND the bump indicator (``eps_T`` or
    # ``norm_smallest_S`` per ``chi_auto_bump_metric``) exceeds
    # ``chi_auto_bump_eps`` for this many consecutive optimizer steps,
    # the optimizer exits early and returns ``best_params``.  Counters
    # reset on a chi bump, a step that brings the indicator below
    # threshold, or stall recovery.  Default 0 disables the check
    # (existing behaviour).  Currently honoured only on the 2-site
    # Tensor AD path.
    gs_chi_ceiling_bailout: int = 0
    gs_noise_recovery_retries: int = 3  # max retries with noise injection on stall
    gs_stall_recovery_retries: int = 5  # max consecutive resets before giving up (#454)
    gs_noise_amplitude: float = 0.1  # relative noise amplitude for recovery
    # Stall recovery mode for L-BFGS / CG line search failures.
    #   "noise"  -> inject gs_noise_amplitude Frobenius perturbation (legacy,
    #               required for 1-site C4v production path to break out of the
    #               SU-init plateau at step 0).
    #   "reset"  -> clear L-BFGS (s, y) history, roll back params to best_params,
    #               force steepest descent on next step.  Matches variPEPS.
    #   None     -> auto-default per dispatcher: "noise" for 1-site, "reset" for
    #               2-site.  Set by optimize_gs_ad at entry.
    gs_stall_recovery: Literal["noise", "reset"] | None = None
    # Optional variational sanity floor on in-loop best-state tracking.  Any
    # candidate energy strictly below this value is rejected as a non-
    # variational CTM artifact (see issue #298).  None disables the check.
    gs_energy_floor: float | None = None
    gs_implicit_ad: bool = True  # implicit diff (VJP + sigma gauge)
    # When True, optimize_gs_ad appends a trajectory dict
    # ``{energies, step_times, jit_compile_time, num_steps, converged}``
    # to its return tuple.  Default is False so existing callers see the
    # same return shape.  Currently supported only on the 1-site
    # (non-C4v-reference) and 2-site Tensor-protocol AD paths.
    return_history: bool = False
    # Deprecated alias — use gs_implicit_ad instead.  Accepted for backwards
    # compatibility; mapped to gs_implicit_ad in __post_init__.
    gs_explicit_ad: bool | None = None
    gs_explicit_ad_steps: int = 20  # CTM steps for explicit AD backprop phase
    gs_explicit_ad_warmup: int = 3  # warmup CTM steps (no gradient tracking)
    # TBPTT cutoff (#506): when set, only the last K of ``gs_explicit_ad_steps``
    # sweeps are differentiated; the leading ``gs_explicit_ad_steps - K`` sweeps
    # use ``jax.lax.stop_gradient`` so JAX skips their backward.  CTM converges
    # geometrically, so the early-iteration backward contribution is bounded by
    # ``ρ^K × (full contribution)`` — negligible for ρ≲0.5 and K≳10.  ``None``
    # (default) keeps the full backward; an int 1..gs_explicit_ad_steps enables
    # truncation.  Composes with the existing per-step ``jax.checkpoint`` wrap
    # in ``ctm_energy_explicit`` to combine memory and compute savings.
    gs_explicit_ad_backward_steps: int | None = None
    su_init: bool = True  # initialize via simple update before AD
    gs_c4v: bool = False  # enforce C4v symmetry on site tensor during AD
    gs_projector_method: str | None = (
        None  # projector override for AD; None => use ctm.projector_method
    )
    # CTM recipe for the GS-AD path: "2x2" (Fishman, default) or "1x1"
    # (1-site moves that enable projector_method, incl. reduced-corner "qr").
    gs_recipe: str = "2x2"
    # CTM convergence tolerance schedule: list of (step_fraction, conv_tol) pairs.
    # Ramps conv_tol from loose (early) to tight (late) during AD optimization.
    # Example: [(0.0, 1e-5), (0.5, 1e-6), (0.8, 1e-7)]
    # None = use config.ctm.conv_tol throughout.
    gs_ctm_conv_tol_schedule: list[tuple[float, float]] | None = None
    # CTM max_iter schedule: list of (step_fraction, max_iter) pairs (#507).
    # Ramps the accepted-step CTM iteration cap across AD progress so late
    # steps — where the env is barely changing — bail in 10-15 sweeps
    # instead of the full early-descent budget (typically 75).  Mirrors
    # the ``gs_ctm_conv_tol_schedule`` lookup contract (largest threshold
    # ≤ ``step_idx / gs_num_steps`` wins).  ``None`` (default) uses
    # ``config.ctm.max_iter`` throughout.  Example late-stage cap:
    #
    #     gs_ctm_max_iter_schedule=[(0.0, 75), (0.5, 30), (0.8, 15)]
    #
    # Independent of ``CTMConfig.probe_max_iter`` (issue #503): that
    # caps HZ line-search probes; this caps the accepted-step CTM.
    gs_ctm_max_iter_schedule: list[tuple[float, int]] | None = None
    # CTM plateau-patience schedule: list of (step_fraction, plateau_patience)
    # pairs.  Ramps the early-bail patience across AD steps so callers can
    # keep a finite stop-loss while the CTM is plateauing (fast, approximate
    # gradients à la variPEPS ``optimizer_ctmrg_preconverged_eps``) and drop
    # to ``None`` at the final stage for strict variational gradients.
    # Example: ``[(0.0, 20), (0.7, None)]``.  ``None`` (default) uses
    # ``config.ctm.plateau_patience`` throughout.
    #
    # **Design note for auto-tuning frameworks**: kept parallel to
    # ``gs_ctm_conv_tol_schedule`` rather than bundled into a single rich
    # schedule so each schedule can be registered, sampled, and ablated
    # independently in a tuning registry.  Use
    # ``aligned_ctm_schedules(...)`` (helper in ``ipeps_config``) when
    # hand-writing configs that want the two ramps to share stage
    # fractions.
    gs_plateau_patience_schedule: list[tuple[float, int | None]] | None = None
    # Metric preconditioning (natural gradient, Rader et al. arXiv:2511.09546)
    gs_metric_precond: bool = True  # metric preconditioning for CG/L-BFGS
    metric_gmres_maxiter: int = 30  # Krylov dimension for metric inversion
    metric_gmres_tol: float = 1e-2  # GMRES tolerance (loose is fine)
    # Coarse-grained iPEPS: when set, optimize_gs_ad uses compute_energy_cg
    # instead of compute_energy_ctm_tensor.  Only valid with unit_cell="1x1".
    cg_gates: object | None = None
    # Checkpointing for long AD runs (see tenax.algorithms._checkpoint).
    # When ``gs_checkpoint_path`` is set, the optimizer writes
    # ``ckpt.last.pkl`` every ``gs_checkpoint_every`` steps and at every
    # chi-schedule stage boundary, plus a ``ckpt.best.pkl`` snapshot
    # each time the best-seen energy improves.  ``gs_resume=True``
    # loads ``ckpt.last.pkl`` at startup, validates config compat, and
    # resumes from the saved step.  Currently wired for the 2-site
    # path only (``unit_cell="2site"``); other paths raise
    # ``NotImplementedError`` if a path is set.
    gs_checkpoint_path: str | None = None
    gs_checkpoint_every: int = 10
    gs_resume: bool = False

    def __post_init__(self):
        import warnings

        if self.gs_explicit_ad is not None:
            warnings.warn(
                "gs_explicit_ad is deprecated; use gs_implicit_ad instead "
                "(gs_implicit_ad=True is the new default, equivalent to "
                "gs_explicit_ad=False).",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "gs_implicit_ad", not self.gs_explicit_ad)
        object.__setattr__(self, "gs_explicit_ad", None)

        valid_unit_cells = {"1x1", "2site"}
        if (
            not isinstance(self.unit_cell, Lattice)
            and self.unit_cell not in valid_unit_cells
        ):
            raise ValueError(
                f"unit_cell must be one of {valid_unit_cells} or a Lattice, "
                f"got {self.unit_cell!r}"
            )
        if self.cg_gates is not None and self.unit_cell != "1x1":
            raise ValueError(
                f"cg_gates requires unit_cell='1x1', got {self.unit_cell!r}"
            )
        if self.cg_gates is not None and self.su_init:
            raise ValueError(
                "cg_gates is incompatible with su_init=True "
                "(simple update uses the microscopic gate, not the CG gates)"
            )
        valid_stall_recovery = {None, "noise", "reset"}
        if self.gs_stall_recovery not in valid_stall_recovery:
            raise ValueError(
                f"gs_stall_recovery must be one of {valid_stall_recovery}, "
                f"got {self.gs_stall_recovery!r}"
            )
        valid_conv_criteria = {"dE", "grad_norm", "both"}
        if self.gs_conv_criterion not in valid_conv_criteria:
            raise ValueError(
                f"gs_conv_criterion must be one of {valid_conv_criteria}, "
                f"got {self.gs_conv_criterion!r}"
            )
        if self.gs_grad_norm_tol <= 0:
            raise ValueError(
                f"gs_grad_norm_tol must be positive, got {self.gs_grad_norm_tol}"
            )
        if self.gs_stall_recovery_retries < 0:
            raise ValueError(
                f"gs_stall_recovery_retries must be non-negative, "
                f"got {self.gs_stall_recovery_retries}"
            )
        if self.gs_grad_spike_ratio is not None and self.gs_grad_spike_ratio <= 1.0:
            raise ValueError(
                "gs_grad_spike_ratio must be > 1.0 (ratio above recent median), "
                f"got {self.gs_grad_spike_ratio}"
            )
        if self.gs_grad_spike_window <= 0:
            raise ValueError(
                "gs_grad_spike_window must be positive, "
                f"got {self.gs_grad_spike_window}"
            )
        if self.gs_hz_max_iter <= 0:
            raise ValueError(
                f"gs_hz_max_iter must be positive, got {self.gs_hz_max_iter}"
            )
        valid_line_search_methods = {"armijo", "hager_zhang", "auto"}
        if self.gs_line_search_method not in valid_line_search_methods:
            raise ValueError(
                f"gs_line_search_method must be one of {valid_line_search_methods}, "
                f"got {self.gs_line_search_method!r}"
            )
        if self.gs_line_search_hz_chi_threshold <= 0:
            raise ValueError(
                "gs_line_search_hz_chi_threshold must be positive, "
                f"got {self.gs_line_search_hz_chi_threshold}"
            )
        if self.gs_explicit_ad_backward_steps is not None:
            if self.gs_explicit_ad_backward_steps < 1:
                raise ValueError(
                    "gs_explicit_ad_backward_steps must be >= 1 (or None to "
                    "disable truncation), "
                    f"got {self.gs_explicit_ad_backward_steps}"
                )
            if self.gs_explicit_ad_backward_steps > self.gs_explicit_ad_steps:
                raise ValueError(
                    "gs_explicit_ad_backward_steps "
                    f"({self.gs_explicit_ad_backward_steps}) cannot exceed "
                    f"gs_explicit_ad_steps ({self.gs_explicit_ad_steps}); "
                    "set to None to backprop through all sweeps."
                )
        if self.gs_chi_ceiling_bailout < 0:
            raise ValueError(
                "gs_chi_ceiling_bailout must be >= 0 (0 disables), "
                f"got {self.gs_chi_ceiling_bailout}"
            )
        if self.gs_checkpoint_every <= 0:
            raise ValueError(
                f"gs_checkpoint_every must be positive, got {self.gs_checkpoint_every}"
            )
        if self.gs_resume and self.gs_checkpoint_path is None:
            raise ValueError("gs_resume=True requires gs_checkpoint_path to be set")
        if self.gs_conv_criterion == "dE":
            warnings.warn(
                "gs_conv_criterion='dE' is deprecated and will be replaced by "
                "'grad_norm' as the default in a future release (issue #448). "
                "The dE criterion underflows near flat minima and after stall "
                "recoveries, causing premature exit. "
                "Set gs_conv_criterion='grad_norm' to opt in now, or 'both' "
                "for the most conservative criterion.",
                DeprecationWarning,
                stacklevel=2,
            )


def aligned_ctm_schedules(
    stages: list[tuple[float, float, int | None]],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, int | None]],
]:
    """Build aligned ``conv_tol`` / ``plateau_patience`` schedules.

    Convenience for the common manual case where both ramps share stage
    fractions.  Each ``stages`` entry is ``(step_fraction, conv_tol,
    plateau_patience)``; returns ``(conv_tol_schedule, patience_schedule)``
    suitable for ``iPEPSConfig.gs_ctm_conv_tol_schedule`` and
    ``iPEPSConfig.gs_plateau_patience_schedule`` respectively.

    For auto-tuning frameworks: leave the two schedules unbundled at the
    config layer (so they can be tuned independently) and use this helper
    only when the user wants a single hand-written spec.

    Example::

        conv, patience = aligned_ctm_schedules(
            [(0.0, 1e-5, 20), (0.7, 1e-7, None)]
        )
        cfg = iPEPSConfig(
            ...,
            gs_ctm_conv_tol_schedule=conv,
            gs_plateau_patience_schedule=patience,
        )
    """
    conv_tol_schedule: list[tuple[float, float]] = []
    patience_schedule: list[tuple[float, int | None]] = []
    for frac, conv_tol, patience in stages:
        conv_tol_schedule.append((float(frac), float(conv_tol)))
        patience_schedule.append((float(frac), patience))
    return conv_tol_schedule, patience_schedule


class CTMEnvironment(NamedTuple):
    """The 8 CTM environment tensors (4 corners + 4 edge tensors).

    Corner convention (looking at a single site):
        C1 --- T1 --- C2
        |             |
        T4    [A]    T2
        |             |
        C4 --- T3 --- C3

    Corners (chi x chi tensors):
        C1: top-left     C2: top-right
        C3: bottom-right C4: bottom-left

    Edges (chi x D^2 x chi tensors, where D = PEPS bond dim):
        T1: top    T2: right
        T3: bottom T4: left

    All shapes use chi for environment bonds, D^2 for the PEPS bond (physical
    space of the doubled layer A * A^* is D^2 = D*D).
    """

    C1: jax.Array  # shape (chi, chi)
    C2: jax.Array  # shape (chi, chi)
    C3: jax.Array  # shape (chi, chi)
    C4: jax.Array  # shape (chi, chi)
    T1: jax.Array  # shape (chi, D2, chi) — top edge
    T2: jax.Array  # shape (chi, D2, chi) — right edge
    T3: jax.Array  # shape (chi, D2, chi) — bottom edge
    T4: jax.Array  # shape (chi, D2, chi) — left edge


class SplitCTMEnvironment(NamedTuple):
    """Split CTM environment keeping ket and bra layers separate.

    Corners are shared between ket and bra (shape ``(chi, chi)``).
    Each edge is split into a ket half ``(chi, D, chi_I)`` and a bra
    half ``(chi_I, D, chi)`` connected by an interlayer bond ``chi_I``.

    This reduces the projector cost from O(chi^3 * D^6) to
    O(chi^3 * D^3) (arXiv:2502.10298).
    """

    C1: jax.Array  # (chi, chi)
    C2: jax.Array  # (chi, chi)
    C3: jax.Array  # (chi, chi)
    C4: jax.Array  # (chi, chi)
    T1_ket: jax.Array  # (chi, D, chi_I)
    T1_bra: jax.Array  # (chi_I, D, chi)
    T2_ket: jax.Array  # (chi, D, chi_I)
    T2_bra: jax.Array  # (chi_I, D, chi)
    T3_ket: jax.Array  # (chi, D, chi_I)
    T3_bra: jax.Array  # (chi_I, D, chi)
    T4_ket: jax.Array  # (chi, D, chi_I)
    T4_bra: jax.Array  # (chi_I, D, chi)
