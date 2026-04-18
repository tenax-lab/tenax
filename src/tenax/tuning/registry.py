"""Tuning parameter registry.

Defines the schema and catalog of tunable parameters across Tenax's
algorithm config dataclasses. Future autotuners can import this module,
enumerate ``ParamSpec`` entries, and use the metadata to drive search.

Schema design notes
-------------------
* **Authoritative defaults live in the dataclasses**, not here. The
  registry stores semantic metadata (tuning range, cost model,
  dependencies) that dataclasses can't carry.
* Entries are declarative — no runtime logic, no imports of algorithm
  internals. This keeps the registry cheap to import and safe to use
  from tooling / CI checks.
* The ``all_params()`` function returns a copy so callers can mutate
  without polluting the module-level catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Scale(Enum):
    """Parameter search scale — hint for autotuners."""

    LINEAR = "linear"
    LOG10 = "log10"
    LOG2 = "log2"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Sensitivity(Enum):
    """How strongly tuning this parameter affects results.

    ``HIGH`` = primary accuracy / convergence knob; small changes matter.
    ``MEDIUM`` = visible effect, often method-dependent.
    ``LOW`` = safe default usually fine; touch only in edge cases.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TuningCategory(Enum):
    """What kind of knob is this?

    Used by docs generation and by autotuners that want to search only
    a subset of the parameter space (e.g. "tune accuracy only, leave
    method selection fixed").
    """

    # Primary accuracy / convergence knobs (chi, max_iter, tolerances).
    ACCURACY = "accuracy"
    # Performance-oriented (jit, preconditioning, warmup steps).
    PERFORMANCE = "performance"
    # Numerical stability / regularization (Tikhonov, SVD broadening).
    STABILITY = "stability"
    # Selects *which* algorithm/variant is used (projector_method, solver).
    METHOD_SELECTION = "method_selection"
    # Optimizer behavior for ground-state search (lr, num_steps, line search).
    OPTIMIZER = "optimizer"
    # Initialization strategy (su_init, random seed handling).
    INITIALIZATION = "initialization"
    # Diagnostic / logging only — no effect on numerics.
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class CostModel:
    """How tuning this parameter changes resource usage.

    All fields are strings because the cost axes are often symbolic
    (``"O(chi^6)"``) or qualitative (``"monotone_improving"``). A future
    autotuner can parse these if it wants to, but the primary consumer
    is the human reference page.
    """

    runtime: str | None = None  # e.g. "O(chi^6)" or "neutral"
    memory: str | None = None  # e.g. "O(chi^4)" or "none"
    accuracy: str | None = None  # e.g. "monotone_improving" or "biases_O(tau)"


@dataclass(frozen=True)
class Dependency:
    """Cross-parameter relationship (for validators / coupled search)."""

    name: str  # dotted name of the related parameter
    relation: str  # human-readable description of the coupling


@dataclass(frozen=True)
class TuningHint:
    """Metadata for automated or manual tuning of a parameter."""

    scale: Scale
    sensitivity: Sensitivity
    # For numeric params: inclusive search range as (lo, hi). None if N/A.
    range: tuple[float, float] | None = None
    # For categorical / enum params: allowed values. None if N/A.
    values: tuple[Any, ...] | None = None
    # Cost model for this knob.
    cost: CostModel = field(default_factory=CostModel)
    # When is this parameter even *applicable*? E.g. a CTM-AD knob is
    # only meaningful when ``ctm_ad_mode="c4v_reference"``. Expressed as
    # ``{other_param_name: required_value_or_values}``. Empty = always.
    applies_when: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParamSpec:
    """A single tunable parameter."""

    name: str  # field name on the dataclass
    dataclass_name: str  # e.g. "CTMConfig" or "iPEPSConfig"
    type_str: str  # e.g. "int", "float", "str", "bool"
    default: Any  # current dataclass default (mirrored — keep in sync!)
    category: TuningCategory
    description: str
    hint: TuningHint
    dependencies: tuple[Dependency, ...] = ()
    when_to_tune: str = ""  # "Increase when energy doesn't converge"
    when_not_to_tune: str = ""  # "Usually the default is fine"
    references: tuple[str, ...] = ()  # paper DOIs / arXiv IDs

    def fully_qualified_name(self) -> str:
        return f"{self.dataclass_name}.{self.name}"


# ---------------------------------------------------------------------------
# Parameter catalog
# ---------------------------------------------------------------------------

_PARAMETERS: list[ParamSpec] = []


def _register(spec: ParamSpec) -> None:
    _PARAMETERS.append(spec)


# --- CTMConfig: primary accuracy knobs --------------------------------------

_register(
    ParamSpec(
        name="chi",
        dataclass_name="CTMConfig",
        type_str="int",
        default=20,
        category=TuningCategory.ACCURACY,
        description=(
            "Bond dimension of the CTM environment tensors. The primary "
            "accuracy knob: higher chi = more environment fidelity."
        ),
        hint=TuningHint(
            scale=Scale.LOG2,
            sensitivity=Sensitivity.HIGH,
            range=(4, 128),
            cost=CostModel(
                runtime="O(chi^6) per sweep for full CTM",
                memory="O(chi^4)",
                accuracy="monotone_improving",
            ),
        ),
        when_to_tune=(
            "Increase when the ground-state energy has not converged as "
            "a function of chi. For D<=3 square iPEPS, chi in [16, 32] is "
            "usually enough; kagome / D>3 may need chi>=64."
        ),
        when_not_to_tune=(
            "Doubling chi typically costs ~64x more runtime. Pin chi and "
            "tune optimizer first."
        ),
    )
)

_register(
    ParamSpec(
        name="max_iter",
        dataclass_name="CTMConfig",
        type_str="int",
        default=100,
        category=TuningCategory.ACCURACY,
        description="Maximum CTM sweeps before declaring convergence.",
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.MEDIUM,
            range=(20, 500),
            cost=CostModel(runtime="linear", memory="none"),
        ),
        when_to_tune=(
            "Raise if CTM hits the iteration cap before reaching conv_tol. "
            "For tight conv_tol (1e-10) near criticality, 200–500 is common."
        ),
    )
)

_register(
    ParamSpec(
        name="conv_tol",
        dataclass_name="CTMConfig",
        type_str="float",
        default=1e-8,
        category=TuningCategory.ACCURACY,
        description=(
            "CTM convergence tolerance on the singular-value difference "
            "between successive sweeps."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.HIGH,
            range=(1e-10, 1e-4),
            cost=CostModel(
                runtime="inversely coupled to max_iter",
                memory="none",
                accuracy="tighter = better observables",
            ),
        ),
        dependencies=(
            Dependency(
                name="CTMConfig.max_iter",
                relation="tol<<1 requires max_iter high enough to reach it",
            ),
        ),
        when_to_tune=(
            "Loosen to 1e-5 for rapid prototyping or early-training schedule; "
            "tighten to 1e-10 for final observables."
        ),
    )
)

_register(
    ParamSpec(
        name="projector_method",
        dataclass_name="CTMConfig",
        type_str="str",
        default="eigh",
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Algorithm used to compute CTM projectors: 'eigh' (symmetric "
            "eigendecomposition), 'qr' (QR gauge), or 'svd' (Fishman SVD)."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.MEDIUM,
            values=("eigh", "qr", "svd"),
            cost=CostModel(
                runtime="eigh ~= svd < qr for small chi; qr wins at large chi",
                memory="neutral",
                accuracy="'svd' is the only fully AD-stable choice post PR #291",
            ),
        ),
        when_to_tune=(
            "Use 'svd' when differentiating through the CTM sweep "
            "(Fishman projector); try 'qr' for forward-only runs with "
            "chi > 32 where eigh becomes a bottleneck."
        ),
        references=("arXiv:1711.01288",),  # Fishman projectors
    )
)

_register(
    ParamSpec(
        name="forward_gauge",
        dataclass_name="CTMConfig",
        type_str="str",
        default="qr",
        category=TuningCategory.STABILITY,
        description=(
            "Gauge fix applied to the environment tensors after each CTM "
            "sweep. Determines AD stability for implicit- and explicit-diff "
            "paths."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=("qr", "phase", "sigma", "none"),
            cost=CostModel(runtime="neutral", memory="none"),
        ),
        when_to_tune=(
            "'phase' is the default for explicit-AD (auto-promoted from "
            "'qr'); 'sigma' is required for implicit-diff stability at "
            "D>=2. 'none' is diagnostic only."
        ),
        references=("arXiv:2311.11894",),  # Francuz et al.
    )
)


# --- CTMConfig: implicit-diff adjoint knobs ---------------------------------

_register(
    ParamSpec(
        name="ctm_ad_mode",
        dataclass_name="CTMConfig",
        type_str="str | None",
        default=None,
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Opt-in reference-mode implicit AD path for dense 1-site C4v "
            "(Francuz et al. App. C–F). None disables; 'c4v_reference' "
            "enables the dense C4v fixed-point backward."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=(None, "c4v_reference"),
        ),
        dependencies=(
            Dependency(
                name="iPEPSConfig.gs_implicit_ad",
                relation="'c4v_reference' requires gs_implicit_ad=True",
            ),
            Dependency(
                name="iPEPSConfig.gs_c4v",
                relation="'c4v_reference' requires gs_c4v=True",
            ),
            Dependency(
                name="iPEPSConfig.unit_cell",
                relation="'c4v_reference' requires unit_cell='1x1'",
            ),
        ),
        references=("arXiv:2311.11894",),
    )
)

_register(
    ParamSpec(
        name="adjoint_solver",
        dataclass_name="CTMConfig",
        type_str="str",
        default="bicgstab",
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Krylov solver used for the linear adjoint system "
            "(I - J^T) lambda = g in reference-mode implicit AD."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.LOW,
            values=("bicgstab", "gmres"),
            applies_when={"CTMConfig.ctm_ad_mode": "c4v_reference"},
        ),
        when_to_tune=(
            "BiCGStab is cheaper per iteration; GMRES is the automatic "
            "fallback when BiCGStab stalls."
        ),
    )
)

_register(
    ParamSpec(
        name="adjoint_tol",
        dataclass_name="CTMConfig",
        type_str="float",
        default=1e-8,
        category=TuningCategory.ACCURACY,
        description=(
            "Target residual tolerance for the Krylov adjoint solve. The "
            "effective residual target is max(10*tol, 1e-12)."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.MEDIUM,
            range=(1e-10, 1e-4),
            applies_when={"CTMConfig.ctm_ad_mode": "c4v_reference"},
            cost=CostModel(runtime="linear in solver iterations"),
        ),
        when_to_tune=(
            "Loosen to 1e-5 when the outer optimizer is near the GS and "
            "the adjoint solver fails to converge; tighten to 1e-10 for "
            "publication-quality gradients."
        ),
    )
)

_register(
    ParamSpec(
        name="adjoint_maxiter",
        dataclass_name="CTMConfig",
        type_str="int",
        default=50,
        category=TuningCategory.ACCURACY,
        description="Maximum Krylov iterations in the adjoint solve.",
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.LOW,
            range=(20, 500),
            applies_when={"CTMConfig.ctm_ad_mode": "c4v_reference"},
        ),
        when_to_tune=(
            "Raise to 200+ when adjoint_tol is tight or the system is "
            "ill-conditioned near the fixed point."
        ),
    )
)

_register(
    ParamSpec(
        name="adjoint_tikhonov",
        dataclass_name="CTMConfig",
        type_str="float",
        default=1e-6,
        category=TuningCategory.STABILITY,
        description=(
            "Tikhonov damping on the linear adjoint system: solve "
            "((I - J^T) + tau I) lambda = g instead of (I - J^T) lambda = g. "
            "Stabilizes the Krylov solve when (I - J^T) becomes near-singular."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.MEDIUM,
            # Strictly positive lower bound so LOG10 sampling is well-defined;
            # pass 0.0 explicitly for a strictly exact adjoint outside tuning.
            range=(1e-10, 1e-2),
            applies_when={"CTMConfig.ctm_ad_mode": "c4v_reference"},
            cost=CostModel(
                runtime="neutral (improves convergence = fewer iters)",
                memory="none",
                accuracy="biases gradient by O(tau) along slow modes",
            ),
        ),
        dependencies=(
            Dependency(
                name="CTMConfig.adjoint_tol",
                relation="tau <= 10*adjoint_tol has no visible effect",
            ),
        ),
        when_to_tune=(
            "Raise to 1e-4..1e-3 if the Krylov solver throws "
            "'failed to converge'. Set to 0 for strictly exact gradients."
        ),
        references=(
            "arXiv:1903.09650",  # Liao et al.
            "arXiv:2311.11894",  # Francuz et al.
            "arXiv:2308.12358",  # variPEPS
        ),
    )
)

# --- iPEPSConfig: optimizer selection & step budget -------------------------

_register(
    ParamSpec(
        name="gs_optimizer",
        dataclass_name="iPEPSConfig",
        type_str="str",
        default="cg",
        category=TuningCategory.METHOD_SELECTION,
        description="Outer-loop optimizer for ground-state AD.",
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=("cg", "adam", "lbfgs"),
        ),
        when_to_tune=(
            "CG is the default and works best with metric preconditioning. "
            "L-BFGS often reaches lower energies but can stall in bad "
            "basins. Adam is more robust to noisy gradients but slower to "
            "converge."
        ),
        dependencies=(
            Dependency(
                name="iPEPSConfig.gs_metric_precond",
                relation="preconditioning especially helps CG and L-BFGS",
            ),
            Dependency(
                name="iPEPSConfig.gs_line_search",
                relation="CG/L-BFGS default to line search on",
            ),
        ),
    )
)

_register(
    ParamSpec(
        name="gs_num_steps",
        dataclass_name="iPEPSConfig",
        type_str="int",
        default=200,
        category=TuningCategory.OPTIMIZER,
        description="Number of AD ground-state optimization steps.",
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.HIGH,
            range=(10, 5000),
            cost=CostModel(runtime="linear", memory="none"),
        ),
        when_to_tune=(
            "variPEPS typically uses 2000+ steps for publication runs. "
            "Smoke tests: 10–50. Benchmarks: 100–500."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_learning_rate",
        dataclass_name="iPEPSConfig",
        type_str="float",
        default=1e-3,
        category=TuningCategory.OPTIMIZER,
        description="Base learning rate / trust-region scale for the optimizer.",
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.HIGH,
            range=(1e-5, 1e-1),
        ),
        when_to_tune=(
            "Adam wants 1e-3..1e-2; CG/L-BFGS typically ignore this in "
            "favor of line search."
        ),
        dependencies=(
            Dependency(
                name="iPEPSConfig.gs_optimizer",
                relation="Effect depends on optimizer; line-search paths "
                "use it only for initial step size",
            ),
        ),
    )
)

_register(
    ParamSpec(
        name="gs_line_search_method",
        dataclass_name="iPEPSConfig",
        type_str="str",
        default="armijo",
        category=TuningCategory.METHOD_SELECTION,
        description="Line-search algorithm used by CG/L-BFGS.",
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.MEDIUM,
            values=("armijo", "hager_zhang"),
        ),
        when_to_tune=(
            "Hager-Zhang is stronger and matches variPEPS; Armijo is "
            "cheaper per step and the current default."
        ),
        references=("doi:10.1137/030601880",),  # Hager-Zhang 2005
    )
)

_register(
    ParamSpec(
        name="gs_stall_recovery",
        dataclass_name="iPEPSConfig",
        type_str="str | None",
        default=None,
        category=TuningCategory.OPTIMIZER,
        description=(
            "Strategy when the line search stalls: 'noise' injects a "
            "Frobenius perturbation (legacy 1-site C4v); 'reset' clears "
            "L-BFGS (s,y) history and rolls back to best_params "
            "(variPEPS). None = auto-per-dispatcher."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.MEDIUM,
            values=(None, "noise", "reset"),
        ),
        when_to_tune=(
            "Set to 'reset' for 2-site runs that get stuck; 'noise' is "
            "required for 1-site C4v to break out of SU-init plateaus."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_metric_precond",
        dataclass_name="iPEPSConfig",
        type_str="bool",
        default=True,
        category=TuningCategory.PERFORMANCE,
        description=(
            "Metric preconditioning (natural gradient) using the local "
            "tangent-space metric N_ij = <d_i psi | d_j psi>."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.HIGH,
            cost=CostModel(
                runtime="~1.5–3x per step (GMRES solves)",
                accuracy="dramatically faster outer-loop convergence",
            ),
        ),
        references=("arXiv:2511.09546",),  # Rader et al.
    )
)

_register(
    ParamSpec(
        name="gs_implicit_ad",
        dataclass_name="iPEPSConfig",
        type_str="bool",
        default=True,
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Use implicit differentiation through the CTM fixed-point "
            "equation (True) instead of differentiating through unrolled "
            "CTM sweeps (False). Implicit is the recommended path."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.HIGH,
            cost=CostModel(
                runtime="explicit (implicit_ad=False) = ~gs_explicit_ad_steps x forward cost",
                memory="explicit (implicit_ad=False) = higher (retained activations)",
            ),
        ),
        dependencies=(
            Dependency(
                name="CTMConfig.forward_gauge",
                relation="auto-promotes qr -> phase when implicit_ad=False",
            ),
            Dependency(
                name="CTMConfig.ctm_ad_mode",
                relation="c4v_reference path requires gs_implicit_ad=True",
            ),
        ),
    )
)

_register(
    ParamSpec(
        name="su_init",
        dataclass_name="iPEPSConfig",
        type_str="bool",
        default=True,
        category=TuningCategory.INITIALIZATION,
        description=(
            "Initialize the site tensor via simple update before AD "
            "(True) or from a random tensor (False)."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.HIGH,
        ),
        when_to_tune=(
            "SU init usually helps. EXCEPTION: for the sublattice-rotated "
            "Heisenberg Hamiltonian, SU converges to the classical product "
            "state, which is a gradient-zero extremum at E=-0.5/site — "
            "L-BFGS cannot escape. Use su_init=False there."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_c4v",
        dataclass_name="iPEPSConfig",
        type_str="bool",
        default=False,
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Enforce C4v symmetry on the site tensor during AD by "
            "parameterizing in a C4v basis."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.HIGH,
        ),
        when_to_tune=(
            "Turn on for C4v-symmetric models (sublattice-rotated square "
            "Heisenberg, XXZ, TFIM). For D=2, this reduces the parameter "
            "count from 16 to ~4 and is the only stable path for 1-site "
            "iPEPS AD in Tenax today."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_ctm_conv_tol_schedule",
        dataclass_name="iPEPSConfig",
        type_str="list[tuple[float, float]] | None",
        default=None,
        category=TuningCategory.PERFORMANCE,
        description=(
            "Ramp CTM conv_tol during optimization: list of (step_fraction, "
            "conv_tol) pairs. Early steps can use a looser CTM tolerance "
            "to save time, tightening as the optimizer converges."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,  # arbitrary schedule structure
            sensitivity=Sensitivity.MEDIUM,
            cost=CostModel(
                runtime="10–30% speedup for long runs",
                accuracy="neutral if schedule ends tight",
            ),
        ),
        when_to_tune=(
            "Large chi + many steps: e.g. [(0.0, 1e-5), (0.5, 1e-7), (0.8, 1e-9)]."
        ),
    )
)


# --- CTMConfig: AD backward & stability knobs -------------------------------

_register(
    ParamSpec(
        name="min_iter",
        dataclass_name="CTMConfig",
        type_str="int",
        default=10,
        category=TuningCategory.STABILITY,
        description=(
            "Minimum CTM sweeps before the convergence check kicks in. "
            "Prevents early exits when the initial SV spectrum happens "
            "to be near-stationary for the first few iterates."
        ),
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.LOW,
            range=(1, 100),
            cost=CostModel(runtime="linear", memory="none"),
        ),
        when_to_tune=(
            "Raise to 30–50 if CTM appears to converge prematurely on "
            "the first few sweeps, especially near criticality."
        ),
    )
)

_register(
    ParamSpec(
        name="projector_backward",
        dataclass_name="CTMConfig",
        type_str="str",
        default="auto",
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Backward pass used for the eigh projector inside explicit AD. "
            "'auto' promotes to 'lorentzian' when gs_implicit_ad=False and "
            "the effective projector is 'eigh'. 'standard' forces the "
            "legacy regularized_eigh backward; 'lorentzian' forces the "
            "Francuz-Schmoll truncated-eigh Lorentzian backward."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=("auto", "standard", "lorentzian"),
            cost=CostModel(
                runtime="lorentzian slightly slower per backward",
                accuracy="lorentzian stabilizes chi>=16",
            ),
        ),
        when_to_tune=(
            "Force 'lorentzian' if the default 'auto' promotion is bypassed "
            "by a non-eigh projector_method. Leave as 'auto' otherwise."
        ),
        references=("arXiv:2311.11894",),
    )
)

_register(
    ParamSpec(
        name="ad_regularize_svd",
        dataclass_name="CTMConfig",
        type_str="bool",
        default=True,
        category=TuningCategory.STABILITY,
        description=(
            "Use Lorentzian-regularized SVD backward in AD. Stabilizes "
            "gradients through SVD projectors at near-degenerate spectra."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.HIGH,
            cost=CostModel(runtime="neutral", accuracy="stabilizes backward"),
        ),
        when_to_tune=(
            "Disable only for diagnostic comparisons against unregularized "
            "SVD backward."
        ),
    )
)

_register(
    ParamSpec(
        name="ad_backward_method",
        dataclass_name="CTMConfig",
        type_str="str",
        default="vjp",
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Backward solver for the multisite implicit-diff path "
            "(ctm_tensor_converge custom_vjp). 'vjp' runs the iterative "
            "Neumann-series backward (YASTN-style); 'gmres' solves "
            "(I - J^T) lambda = g directly. GMRES is currently xfail "
            "(issue #292) without tight sigma-gauge alignment."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=("vjp", "gmres"),
            cost=CostModel(
                runtime="gmres fewer backward iters when converged",
                accuracy="vjp more robust today",
            ),
        ),
        when_to_tune=(
            "Prefer 'vjp' until issue #292 lands and GMRES is stabilized "
            "with forward_gauge='sigma' and adjoint Tikhonov."
        ),
        references=("arXiv:2311.11894",),
    )
)

_register(
    ParamSpec(
        name="gmres_precondition",
        dataclass_name="CTMConfig",
        type_str="bool",
        default=False,
        category=TuningCategory.PERFORMANCE,
        description=(
            "Enable a diagonal scaling preconditioner for the GMRES "
            "backward solver. Only meaningful when "
            "ad_backward_method='gmres'."
        ),
        hint=TuningHint(
            scale=Scale.BOOLEAN,
            sensitivity=Sensitivity.MEDIUM,
            applies_when={"CTMConfig.ad_backward_method": "gmres"},
            cost=CostModel(runtime="fewer GMRES iters", memory="O(chi^2)"),
        ),
        when_to_tune=(
            "Enable once the GMRES backward is stabilized (#292); "
            "experimental until then."
        ),
    )
)

_register(
    ParamSpec(
        name="ctm_conv_method",
        dataclass_name="CTMConfig",
        type_str="str",
        default="sv",
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Convergence check for the CTM fixed-point loop. 'sv' compares "
            "corner singular values across sweeps (fast, gauge-insensitive); "
            "'elementwise' compares raw environment tensors."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.LOW,
            values=("sv", "elementwise"),
        ),
        when_to_tune=(
            "'elementwise' only for diagnostic comparisons; 'sv' is almost "
            "always the right choice."
        ),
    )
)


# --- iPEPSConfig: explicit-AD budget ----------------------------------------

_register(
    ParamSpec(
        name="gs_explicit_ad_steps",
        dataclass_name="iPEPSConfig",
        type_str="int",
        default=20,
        category=TuningCategory.ACCURACY,
        description=(
            "Number of CTM sweeps differentiated through when "
            "gs_implicit_ad=False. The final K sweeps carry gradients; "
            "earlier sweeps run under stop_gradient as warmup."
        ),
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.HIGH,
            range=(5, 50),
            applies_when={"iPEPSConfig.gs_implicit_ad": False},
            cost=CostModel(
                runtime="linear in K", memory="linear in K (saved activations)"
            ),
        ),
        when_to_tune=(
            "Increase when gradients look truncated (energy plateau with "
            "small residual); decrease to save memory at large chi."
        ),
        references=("arXiv:2511.09546",),  # Rader et al. use 3+20
    )
)

_register(
    ParamSpec(
        name="gs_explicit_ad_warmup",
        dataclass_name="iPEPSConfig",
        type_str="int",
        default=3,
        category=TuningCategory.PERFORMANCE,
        description=(
            "Non-differentiated CTM warmup sweeps before the "
            "gs_explicit_ad_steps differentiated sweeps (when gs_implicit_ad=False). Lets the "
            "environment pre-converge cheaply."
        ),
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.MEDIUM,
            range=(0, 30),
            applies_when={"iPEPSConfig.gs_implicit_ad": False},
            cost=CostModel(runtime="linear", memory="constant"),
        ),
        when_to_tune=(
            "Raise when the warm-started env is far from the fixed point "
            "(e.g. after a large L-BFGS step)."
        ),
        references=("arXiv:2511.09546",),
    )
)

_register(
    ParamSpec(
        name="metric_gmres_maxiter",
        dataclass_name="iPEPSConfig",
        type_str="int",
        default=30,
        category=TuningCategory.PERFORMANCE,
        description=(
            "Maximum GMRES iterations when solving the metric "
            "(quantum-geometric-tensor) preconditioner system "
            "N g' = g. Only active when gs_metric_precond=True."
        ),
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.MEDIUM,
            range=(5, 100),
            applies_when={"iPEPSConfig.gs_metric_precond": True},
            cost=CostModel(runtime="linear per precondition call"),
        ),
        when_to_tune=(
            "Raise when metric_gmres_tol is not reached; lower to save "
            "time when the outer optimizer is still far from GS."
        ),
        references=("arXiv:2511.09546",),
    )
)

_register(
    ParamSpec(
        name="metric_gmres_tol",
        dataclass_name="iPEPSConfig",
        type_str="float",
        default=1e-2,
        category=TuningCategory.ACCURACY,
        description=(
            "Relative residual tolerance for the metric preconditioner GMRES solve."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.MEDIUM,
            range=(1e-6, 1e-1),
            applies_when={"iPEPSConfig.gs_metric_precond": True},
            cost=CostModel(runtime="tighter = more GMRES iters"),
        ),
        when_to_tune=(
            "Tighten to 1e-3–1e-4 near the GS for accurate natural "
            "gradient; 1e-2 is fine early."
        ),
        references=("arXiv:2511.09546",),
    )
)

_register(
    ParamSpec(
        name="gs_conv_tol",
        dataclass_name="iPEPSConfig",
        type_str="float",
        default=1e-8,
        category=TuningCategory.ACCURACY,
        description=(
            "Energy-change convergence tolerance for the outer L-BFGS/CG "
            "ground-state optimizer. Loop exits early when "
            "|E_k - E_{k-1}| < gs_conv_tol."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.MEDIUM,
            range=(1e-12, 1e-4),
            cost=CostModel(accuracy="lower = tighter ground state"),
        ),
        when_to_tune=(
            "Loosen to 1e-6 for fast exploratory runs; tighten to 1e-10 for production."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_max_grad_norm",
        dataclass_name="iPEPSConfig",
        type_str="float",
        default=1.0,
        category=TuningCategory.STABILITY,
        description=(
            "Clip the L-BFGS/CG gradient norm before the search direction "
            "is built. Protects against spurious large gradients from "
            "CTM near-degeneracies."
        ),
        hint=TuningHint(
            scale=Scale.LOG10,
            sensitivity=Sensitivity.MEDIUM,
            range=(1e-2, 1e2),
            cost=CostModel(accuracy="clip biases update direction"),
        ),
        when_to_tune=(
            "Lower to 0.1 when the optimizer takes large rejected steps; "
            "raise to 10 if the clip cuts healthy gradients."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_line_search",
        dataclass_name="iPEPSConfig",
        type_str="bool | None",
        default=None,
        category=TuningCategory.OPTIMIZER,
        description=(
            "Enable line search for the outer optimizer. None = auto "
            "(on for L-BFGS, off for plain gradient descent)."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.HIGH,
            values=(None, True, False),
            cost=CostModel(runtime="2–5x more energy evals per step"),
        ),
        when_to_tune=(
            "Disable only for diagnostic comparisons; line search is "
            "almost always worth its cost on noisy gradients. ``None`` "
            "(auto) is the baseline; override only to force a specific "
            "behavior."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_line_search_max_steps",
        dataclass_name="iPEPSConfig",
        type_str="int",
        default=8,
        category=TuningCategory.OPTIMIZER,
        description=(
            "Maximum backtracking steps for the line search (both Armijo "
            "and Hager-Zhang)."
        ),
        hint=TuningHint(
            scale=Scale.LINEAR,
            sensitivity=Sensitivity.LOW,
            range=(3, 20),
            cost=CostModel(runtime="linear in budget"),
        ),
        when_to_tune=(
            "Raise to 16 when line search frequently hits the cap without "
            "accepting; lower to save time on cheap-to-eval loss."
        ),
    )
)

_register(
    ParamSpec(
        name="gs_projector_method",
        dataclass_name="iPEPSConfig",
        type_str="str | None",
        default=None,
        category=TuningCategory.METHOD_SELECTION,
        description=(
            "Override CTMConfig.projector_method for the AD path without "
            "affecting CTM calls outside the optimizer. None inherits "
            "from ctm.projector_method."
        ),
        hint=TuningHint(
            scale=Scale.CATEGORICAL,
            sensitivity=Sensitivity.MEDIUM,
            values=(None, "eigh", "qr", "svd"),
        ),
        when_to_tune=(
            "Set to 'eigh' to force the AD-friendly projector while "
            "letting the rest of the pipeline use 'qr' or 'svd'."
        ),
    )
)


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


def all_params() -> list[ParamSpec]:
    """Return all registered parameters as a fresh list."""
    return list(_PARAMETERS)


def get_param(fully_qualified_name: str) -> ParamSpec | None:
    """Look up a parameter by its ``Dataclass.field`` name."""
    for p in _PARAMETERS:
        if p.fully_qualified_name() == fully_qualified_name:
            return p
    return None


def params_by_category(category: TuningCategory) -> list[ParamSpec]:
    """Return all parameters in a given semantic category."""
    return [p for p in _PARAMETERS if p.category == category]


def params_by_dataclass(dataclass_name: str) -> list[ParamSpec]:
    """Return all parameters declared on a given dataclass."""
    return [p for p in _PARAMETERS if p.dataclass_name == dataclass_name]
