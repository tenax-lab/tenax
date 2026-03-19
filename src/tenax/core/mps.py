"""Finite and infinite MPS containers with canonical form tracking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import jax.numpy as jnp

from tenax.core.tensor import SymmetricTensor, Tensor


@dataclass
class FiniteMPS:
    """Finite matrix product state with canonical form tracking.

    Attributes:
        tensors: Site tensors, length L.  Boundary sites are 2-leg
            (site 0: physical x right-bond, site L-1: left-bond x physical),
            bulk sites are 3-leg (left-bond x physical x right-bond).
            Labels follow the convention p{i} for physical, v{i}_{i+1} for bonds.
        orth_center: Position of the orthogonality center, or None if the
            canonical form is unknown.  When set, sites 0..orth_center-1 are
            left-canonical and sites orth_center+1..L-1 are right-canonical.
        singular_values: Singular values at each bond (length L-1).
            Entry i holds the singular values between sites i and i+1,
            or None if not yet computed.
    """

    tensors: list[Tensor]
    orth_center: int | None = None
    singular_values: list[jnp.ndarray | None] = field(default_factory=list)

    def __post_init__(self):
        if not self.singular_values:
            self.singular_values = [None] * max(len(self.tensors) - 1, 0)

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        orth_center: int | None = None,
        singular_values: list[jnp.ndarray | None] | None = None,
    ) -> FiniteMPS:
        """Wrap existing site tensors into a FiniteMPS."""
        L = len(tensors)
        if singular_values is None:
            singular_values = [None] * max(L - 1, 0)
        return FiniteMPS(
            tensors=list(tensors),
            orth_center=orth_center,
            singular_values=singular_values,
        )

    # -- Sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, i: int) -> Tensor:
        return self.tensors[i]

    def __setitem__(self, i: int, tensor: Tensor) -> None:
        self.tensors[i] = tensor
        self.orth_center = None  # invalidate

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.tensors)

    # -- Properties ---------------------------------------------------------

    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions between sites (length L-1).

        bond_dims[i] = dimension of the virtual bond between sites i and i+1.
        """
        dims = []
        for i in range(self.L - 1):
            t = self.tensors[i]
            bond_label = f"v{i}_{i + 1}"
            for idx in t.indices:
                if idx.label == bond_label:
                    dims.append(idx.dim)
                    break
            else:
                raise ValueError(
                    f"Site {i} has no index with label '{bond_label}'. "
                    f"Labels: {t.labels()}"
                )
        return dims

    @property
    def phys_dims(self) -> list[int]:
        """Physical dimensions at each site (length L)."""
        dims = []
        for i in range(self.L):
            phys_label = f"p{i}"
            for idx in self.tensors[i].indices:
                if idx.label == phys_label:
                    dims.append(idx.dim)
                    break
            else:
                raise ValueError(
                    f"Site {i} has no index with label '{phys_label}'. "
                    f"Labels: {self.tensors[i].labels()}"
                )
        return dims

    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension across all bonds."""
        dims = self.bond_dims
        return max(dims) if dims else 0

    @property
    def is_symmetric(self) -> bool:
        """True if all site tensors are SymmetricTensor."""
        return all(isinstance(t, SymmetricTensor) for t in self.tensors)
