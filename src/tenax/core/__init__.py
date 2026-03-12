"""Core tensor and symmetry classes."""

from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)
from tenax.core.symmetry import (
    BaseNonAbelianSymmetry,
    BaseSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tenax.core.tensor import BlockKey, DenseTensor, SymmetricTensor, Tensor

# Shared epsilon constants used across algorithms to prevent underflow.
# EPS: general-purpose zero-guard for division and normalization (1e-15).
# LOG_EPS: safe addend before jnp.log to avoid log(0) = -inf (1e-250).
EPS = 1e-15
LOG_EPS = 1e-250

__all__ = [
    "BaseSymmetry",
    "U1Symmetry",
    "ZnSymmetry",
    "BaseNonAbelianSymmetry",
    "FlowDirection",
    "Label",
    "TensorIndex",
    "Tensor",
    "DenseTensor",
    "SymmetricTensor",
    "BlockKey",
    "EPS",
    "LOG_EPS",
    # Lattice
    "Bond",
    "Lattice",
    "square",
    "checkerboard",
    "honeycomb",
    "triangular",
    "kagome",
]
