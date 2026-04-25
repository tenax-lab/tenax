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
    C0: Tensor
    C1: Tensor
    C2: Tensor
    L0: Tensor
    L1: Tensor
    L2: Tensor
    R0: Tensor
    R1: Tensor
    R2: Tensor
