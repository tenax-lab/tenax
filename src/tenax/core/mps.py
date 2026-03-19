"""Finite and infinite MPS containers with canonical form tracking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import jax.numpy as jnp

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract
from tenax.core.tensor import SymmetricTensor, Tensor
from tenax.linalg import qr, svd


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

    # -- Canonicalization ---------------------------------------------------

    def canonicalize(self, center: int) -> FiniteMPS:
        """Return a new MPS in mixed-canonical form with orthogonality center.

        Left-to-right QR sweep for sites 0..center-1, then right-to-left QR
        sweep for sites L-1..center+1, and finally SVD at center to extract
        singular values.

        Args:
            center: Site index of the orthogonality center.

        Returns:
            A new FiniteMPS with ``orth_center=center`` and
            ``singular_values[center]`` populated (when center < L-1).
        """
        L = self.L
        if center < 0 or center >= L:
            raise ValueError(f"center={center} out of range [0, {L})")

        tensors = [t for t in self.tensors]  # shallow copy
        sv = [None] * max(L - 1, 0)

        # --- Left-to-right QR sweep: sites 0 .. center-1 ---
        for i in range(center):
            site = tensors[i]
            right_bond = f"v{i}_{i + 1}"
            left_labels = [lb for lb in site.labels() if lb != right_bond]
            # Use a temp bond label to avoid duplicate labels in R
            tmp_bond = f"_qr_{right_bond}"
            Q, R = qr(site, left_labels, [right_bond], new_bond_label=tmp_bond)
            # Q has (..., tmp_bond), rename to right_bond
            tensors[i] = Q.relabel(tmp_bond, right_bond)
            # R has (tmp_bond, right_bond) — contract on right_bond with
            # next site, then rename tmp_bond -> right_bond
            absorbed = contract(R, tensors[i + 1])
            tensors[i + 1] = absorbed.relabel(tmp_bond, right_bond)

        # --- Right-to-left QR sweep: sites L-1 .. center+1 ---
        for i in range(L - 1, center, -1):
            site = tensors[i]
            left_bond = f"v{i - 1}_{i}"
            other_labels = [lb for lb in site.labels() if lb != left_bond]
            # Use a temp bond label to avoid duplicate labels in R
            tmp_bond = f"_qr_{left_bond}"
            Q, R = qr(site, other_labels, [left_bond], new_bond_label=tmp_bond)
            # Q has (other_labels..., tmp_bond), rename tmp_bond -> left_bond
            Q = Q.relabel(tmp_bond, left_bond)
            # Reorder legs so left_bond is first (MPS convention)
            labels = Q.labels()
            bond_pos = labels.index(left_bond)
            if bond_pos != 0:
                axes = (bond_pos,) + tuple(
                    j for j in range(len(labels)) if j != bond_pos
                )
                Q = Q.transpose(axes)
            tensors[i] = Q
            # R has (tmp_bond, left_bond) — contract on left_bond with
            # prev site, then rename tmp_bond -> left_bond
            absorbed = contract(tensors[i - 1], R)
            tensors[i - 1] = absorbed.relabel(tmp_bond, left_bond)

        # --- SVD at center to extract singular values ---
        if center < L - 1:
            site = tensors[center]
            right_bond = f"v{center}_{center + 1}"
            left_labels = [lb for lb in site.labels() if lb != right_bond]
            tmp_bond = f"_svd_{right_bond}"
            U, s, Vh, s_full = svd(
                site, left_labels, [right_bond], new_bond_label=tmp_bond
            )
            sv[center] = s_full

            # Absorb singular values into U along the bond axis
            US = scale_bond_axis(U, tmp_bond, s)
            # Rename tmp_bond -> right_bond
            tensors[center] = US.relabel(tmp_bond, right_bond)

            # Vh has (tmp_bond, right_bond) — contract with next site on
            # right_bond, then rename tmp_bond -> right_bond
            tensors[center + 1] = contract(Vh, tensors[center + 1]).relabel(
                tmp_bond, right_bond
            )

        return FiniteMPS(
            tensors=tensors,
            orth_center=center,
            singular_values=sv,
        )

    # -- Norm / Overlap -----------------------------------------------------

    def overlap(self, other: FiniteMPS) -> complex:
        """Compute <self|other> via left-to-right transfer matrix contraction.

        Args:
            other: Another FiniteMPS with the same length and physical dims.

        Returns:
            The scalar overlap <self|other>.
        """
        if len(self) != len(other):
            raise ValueError("MPS lengths must match for overlap")

        env = None
        for i in range(self.L):
            ket = other[i]
            bra = self[i].conj()

            # Relabel bra's virtual bond labels to avoid collision with ket.
            # Physical labels (p{i}) stay the same so they contract.
            for label in bra.labels():
                if label.startswith("v"):
                    bra = bra.relabel(label, label + "*")

            # Contract bra and ket on physical index
            site_transfer = contract(bra, ket)

            if env is None:
                env = site_transfer
            else:
                env = contract(env, site_transfer)

        # env should be a scalar (or 0-dim); extract value
        return complex(env.todense())

    def norm(self) -> float:
        """Compute the norm ||psi|| = sqrt(<psi|psi>).

        Returns:
            The norm as a non-negative real number.
        """
        return float(jnp.sqrt(jnp.abs(self.overlap(self))))

    def entanglement_entropy(self, bond: int) -> float:
        """Compute the Von Neumann entanglement entropy at a bond.

        Args:
            bond: Bond index (between sites ``bond`` and ``bond+1``).

        Returns:
            The entanglement entropy S = -sum(p * log(p)) where p = sv^2 / sum(sv^2).
        """
        L = self.L
        if bond < 0 or bond >= L - 1:
            raise ValueError(f"bond={bond} out of range [0, {L - 1})")

        sv = self.singular_values[bond]
        if sv is None:
            mps_c = self.canonicalize(center=bond)
            sv = mps_c.singular_values[bond]

        sv = jnp.asarray(sv)
        # Filter near-zero singular values
        sv = sv[sv > 1e-15]
        p = sv**2
        p = p / jnp.sum(p)
        S = -jnp.sum(p * jnp.log(p))
        return float(S)

    def left_canonicalize(self) -> FiniteMPS:
        """Return a new MPS in left-canonical form (orth_center = L-1)."""
        return self.canonicalize(center=self.L - 1)

    def right_canonicalize(self) -> FiniteMPS:
        """Return a new MPS in right-canonical form (orth_center = 0)."""
        return self.canonicalize(center=0)
