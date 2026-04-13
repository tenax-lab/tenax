"""Machine-readable tuning registry for Tenax algorithm parameters.

This subpackage declares a structured catalog of tunable parameters
across Tenax's config dataclasses. It is consumed by:

  - ``docs/guide/tuning.md`` (human-facing reference, hand-written)
  - future autotuning / hyperparameter-search tooling that needs
    machine-readable metadata (ranges, scales, dependencies, costs)

The registry does *not* replace the dataclass defaults — those remain
authoritative. The registry adds semantic metadata that pure Python
dataclasses can't carry (tuning range, cost model, dependencies,
categorical groupings).
"""

from tenax.tuning.registry import (
    CostModel,
    Dependency,
    ParamSpec,
    Scale,
    Sensitivity,
    TuningCategory,
    TuningHint,
    all_params,
    get_param,
    params_by_category,
    params_by_dataclass,
)

__all__ = [
    "CostModel",
    "Dependency",
    "ParamSpec",
    "Scale",
    "Sensitivity",
    "TuningCategory",
    "TuningHint",
    "all_params",
    "get_param",
    "params_by_category",
    "params_by_dataclass",
]
