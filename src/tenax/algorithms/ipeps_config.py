"""iPEPS configuration dataclasses and environment NamedTuples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax


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
                            ``"phase"`` when ``gs_explicit_ad=True`` and the
                            user has not opted into a different gauge — see
                            ``docs/guide/algorithms/ipeps_ad_paths.md`` for
                            the full mode matrix and benchmarks.
        ad_backward_method: Backward method for the implicit-diff path.
                            ``"vjp"`` (default) is the regression-covered
                            Neumann-series backward.  ``"gmres"`` is
                            currently documented unstable (spectral radius
                            > 1 without tight sigma-gauge alignment) and
                            its regression test is marked ``xfail`` —
                            tracked by issue #292.  Prefer the explicit-AD
                            path (``iPEPSConfig.gs_explicit_ad=True``)
                            until GMRES is stabilized.
    """

    chi: int = 20
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True
    projector_method: str = "eigh"  # "eigh", "qr", or "svd" (Fishman)
    min_iter: int = 10  # minimum CTM sweeps before checking convergence
    qr_warmup_steps: int = 3  # eigh warm-up iterations before QR kicks in
    chi_I: int | None = None  # interlayer bond dim for split-CTMRG; None => chi_I = chi
    ad_regularize_svd: bool = True  # use Lorentzian-regularized SVD backward in AD
    gmres_precondition: bool = (
        False  # diagonal scaling preconditioner for GMRES backward (experimental)
    )
    ad_backward_method: str = "vjp"  # "vjp" (iterative VJP) or "gmres" (xfail — #292)
    ctm_conv_method: str = "sv"  # "sv" (singular value) or "elementwise"
    # forward_gauge: "qr" (default, auto-promoted to "phase" by optimize_gs_ad
    # when gs_explicit_ad=True), "phase" (explicit-AD default after promotion),
    # "sigma" (implicit-diff path), or "none" (diagnostic).  See ipeps_ad_paths.md.
    forward_gauge: str = "qr"
    jit_ctm: bool = False  # use jax.lax.while_loop for GPU kernel fusion


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
        gs_explicit_ad:        Differentiate through unrolled CTM sweeps
                               instead of using implicit differentiation.
                               ``True`` by default and the recommended AD
                               path post-PR-#291.  When ``True`` and
                               ``ctm.forward_gauge == "qr"`` (the static
                               default), ``optimize_gs_ad`` transparently
                               promotes the forward gauge to ``"phase"`` for
                               the unrolled CTM sweeps — see
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
    """

    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = field(default_factory=CTMConfig)
    svd_trunc_err: float | None = None
    gate_order: str = "sequential"
    unit_cell: str = "1x1"  # "1x1" or "2site"
    # AD ground-state optimization settings
    gs_optimizer: str = "cg"  # "cg", "adam", or "lbfgs"
    gs_learning_rate: float = 1e-3
    gs_num_steps: int = 200
    gs_conv_tol: float = 1e-8
    gs_verbose: bool = False
    gs_log_interval: int = 10
    gs_max_grad_norm: float = 1.0  # gradient clipping (max global norm)
    gs_line_search: bool | None = None  # None = auto (True for lbfgs/cg)
    gs_line_search_max_steps: int = 8
    gs_line_search_method: str = "armijo"  # "armijo" or "hager_zhang"
    gs_noise_recovery_retries: int = 3  # max retries with noise injection on stall
    gs_noise_amplitude: float = 0.1  # relative noise amplitude for recovery
    gs_explicit_ad: bool = True  # explicit diff through unrolled CTM
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
        valid_unit_cells = {"1x1", "2site"}
        if self.unit_cell not in valid_unit_cells:
            raise ValueError(
                f"unit_cell must be one of {valid_unit_cells}, got {self.unit_cell!r}"
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
