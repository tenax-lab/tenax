"""Native honeycomb CTM environment data structure.

Per sublattice, the env consists of 3 corner tensors (one per honeycomb
edge direction α ∈ {0, 1, 2}) and 3 left + 3 right column tensors.

Shapes (chi = boundary dim, D = bond dim):
    C_α: (chi, chi)         [labels: (chi_in_α, chi_out_α)]
    L_α: (chi, D**2, chi)   [labels: (chi_in_α, e_α_d2, chi_out_α)]
    R_α: (chi, D**2, chi)   [labels: (chi_in_α, e_α_d2, chi_out_α)]
"""

from __future__ import annotations

from typing import NamedTuple

from tenax.core.tensor import Tensor

__all__ = ["HoneycombCTMEnv"]


class HoneycombCTMEnv(NamedTuple):
    """Per-sublattice honeycomb CTM environment.

    Index ``α ∈ {0, 1, 2}`` selects the honeycomb edge direction (NOT
    a corner number). Three corners and three left/right column tensors
    per sublattice.

    Corner ``C_α`` is a 2-leg tensor ``(chi, chi)``; column tensors
    ``L_α`` and ``R_α`` are 3-leg tensors ``(chi, D**2, chi)`` carrying
    the fused double-layer bond.
    """

    C0: Tensor  # (chi_in_0, chi_out_0)         flows: (IN, OUT)
    C1: Tensor  # (chi_in_1, chi_out_1)         flows: (IN, OUT)
    C2: Tensor  # (chi_in_2, chi_out_2)         flows: (IN, OUT)
    L0: Tensor  # (chi_in_0, e_0_d2, chi_out_0)  flows: (IN, ?, OUT)
    L1: Tensor  # (chi_in_1, e_1_d2, chi_out_1)  flows: (IN, ?, OUT)
    L2: Tensor  # (chi_in_2, e_2_d2, chi_out_2)  flows: (IN, ?, OUT)
    R0: Tensor  # (chi_in_0, e_0_d2, chi_out_0)  flows: (IN, ?, OUT)
    R1: Tensor  # (chi_in_1, e_1_d2, chi_out_1)  flows: (IN, ?, OUT)
    R2: Tensor  # (chi_in_2, e_2_d2, chi_out_2)  flows: (IN, ?, OUT)
