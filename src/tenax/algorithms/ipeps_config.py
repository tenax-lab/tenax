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
    """

    chi: int = 20
    chi_ramp: list[tuple[int, int | None]] | None = None
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True
    projector_method: str = "svd"  # "svd" (Fishman, default), "eigh", or "qr"
    min_iter: int = 10  # minimum CTM sweeps before checking convergence
    qr_warmup_steps: int = 3  # eigh warm-up iterations before QR kicks in
    chi_I: int | None = None  # interlayer bond dim for split-CTMRG; None => chi_I = chi
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
        if self.chi_max is not None and self.chi_max < self.chi:
            raise ValueError(f"chi_max ({self.chi_max}) must be >= chi ({self.chi})")
        # Issue #492 in-CTM χ-bump validation.
        if self.ctmrg_heuristic_increase_chi and self.chi_ramp is not None:
            raise ValueError(
                "ctmrg_heuristic_increase_chi and chi_ramp are mutually exclusive: "
                "chi_ramp is a deterministic optimizer-side schedule, "
                "ctmrg_heuristic_increase_chi is reactive inside CTM convergence"
            )
        if (
            self.ctmrg_heuristic_increase_chi
            and self.ctmrg_heuristic_increase_chi_step_size <= 0
        ):
            raise ValueError(
                "ctmrg_heuristic_increase_chi_step_size must be a positive integer, "
                f"got {self.ctmrg_heuristic_increase_chi_step_size}"
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
                               (VJP backward with sigma gauge) is the
                               recommended AD path.  When ``True`` and
                               ``ctm.forward_gauge == "qr"`` (the static
                               default), ``optimize_gs_ad`` transparently
                               promotes the forward gauge to ``"sigma"``
                               for stable element-wise CTM convergence.
                               When ``False``, it promotes to ``"phase"``
                               instead — see
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
    gs_line_search: bool | None = None  # None = auto (True for lbfgs/cg)
    gs_line_search_max_steps: int = 8
    gs_line_search_method: str = "hager_zhang"  # "armijo" or "hager_zhang"
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
    su_init: bool = True  # initialize via simple update before AD
    gs_c4v: bool = False  # enforce C4v symmetry on site tensor during AD
    gs_projector_method: str | None = (
        None  # projector override for AD; None => use ctm.projector_method
    )
    # CTM convergence tolerance schedule: list of (step_fraction, conv_tol) pairs.
    # Ramps conv_tol from loose (early) to tight (late) during AD optimization.
    # Example: [(0.0, 1e-5), (0.5, 1e-6), (0.8, 1e-7)]
    # None = use config.ctm.conv_tol throughout.
    gs_ctm_conv_tol_schedule: list[tuple[float, float]] | None = None
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
