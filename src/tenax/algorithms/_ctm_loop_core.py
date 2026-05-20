"""Shared bump-aware CTM convergence loop.

Consumed by python_loop_ctm_converge, _sigma_gauged_ctm_converge (implicit-AD
forward), and ctm_energy_explicit warmup.  Centralizing the bump pad+resweep
sequence keeps the variPEPS-style growth contract (#492) in one place across
all three forward CTM paths (#514).
"""

from __future__ import annotations

__all__ = ["CTMLoopResult", "_run_ctm_loop_with_bump"]

from typing import NamedTuple

from tenax.algorithms._ctm_tensor_convergence import Coord
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv


class CTMLoopResult(NamedTuple):
    """Outcome of one bump-aware CTM convergence loop run."""

    envs: dict[Coord, CTMTensorEnv]
    converged: bool
    iterations: int
    sv_diff: float
    max_truncation_error: float
    max_smallest_S: float
    final_chi: int
    bump_extra_sweeps: int


def _run_ctm_loop_with_bump(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError  # filled in Task 2
