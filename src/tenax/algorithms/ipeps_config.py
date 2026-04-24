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
                            ``"qr"`` (default), ``"phase"``, ``"sigma"``, or
                            ``"none"``.  The static default stays ``"qr"``
                            for forward-only CTM, diagnostics, and notebooks.
                            ``optimize_gs_ad`` auto-promotes ``"qr"`` to
                            ``"phase"`` when ``gs_implicit_ad=False`` and the
                            user has not opted into a different gauge — see
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
        gs_stall_recovery:     Stall-recovery mode for line-search failures
                               (issue #298).  ``"noise"`` injects a Frobenius
                               perturbation (legacy 1-site C4v path);
                               ``"reset"`` clears L-BFGS ``(s, y)`` history
                               and rolls back to ``best_params`` (variPEPS
                               style).  ``None`` (default) lets
                               ``optimize_gs_ad`` pick per dispatcher:
                               ``"noise"`` for 1-site, ``"reset"`` for 2-site.
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
    gs_verbose: bool = False
    gs_log_interval: int = 10
    gs_max_grad_norm: float = 1.0  # gradient clipping (max global norm)
    gs_line_search: bool | None = None  # None = auto (True for lbfgs/cg)
    gs_line_search_max_steps: int = 8
    gs_line_search_method: str = "hager_zhang"  # "armijo" or "hager_zhang"
    gs_noise_recovery_retries: int = 3  # max retries with noise injection on stall
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
    # Metric preconditioning (natural gradient, Rader et al. arXiv:2511.09546)
    gs_metric_precond: bool = True  # metric preconditioning for CG/L-BFGS
    metric_gmres_maxiter: int = 30  # Krylov dimension for metric inversion
    metric_gmres_tol: float = 1e-2  # GMRES tolerance (loose is fine)

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
        valid_stall_recovery = {None, "noise", "reset"}
        if self.gs_stall_recovery not in valid_stall_recovery:
            raise ValueError(
                f"gs_stall_recovery must be one of {valid_stall_recovery}, "
                f"got {self.gs_stall_recovery!r}"
            )


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
