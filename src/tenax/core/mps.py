"""Finite and infinite MPS containers with canonical form tracking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
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

    @staticmethod
    def random(
        L: int,
        d: int,
        chi: int,
        key: jax.Array,
        *,
        symmetric: bool = False,
        symmetry: Any = None,
        target_charge: int | None = None,
        dtype: Any = jnp.float64,
    ) -> FiniteMPS:
        """Random MPS, right-canonicalized (orth_center=0).

        For symmetric=True, requires symmetry and target_charge.
        """
        if symmetric:
            tensors = _build_random_symmetric_tensors(
                L, d, chi, key, symmetry, target_charge, dtype
            )
        else:
            tensors = _build_random_dense_tensors(L, d, chi, key, dtype)
        mps = FiniteMPS.from_tensors(tensors)
        return mps.right_canonicalize()

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

    # -- TensorNetwork compatibility ----------------------------------------

    def n_nodes(self) -> int:
        """Return the number of sites (TensorNetwork compatibility)."""
        return len(self.tensors)

    def get_tensor(self, i: int) -> Tensor:
        """Return the tensor at site *i* (TensorNetwork compatibility)."""
        return self.tensors[i]

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


@dataclass
class InfiniteMPS:
    """Infinite matrix product state with a finite unit cell.

    Attributes:
        tensors: Unit cell site tensors (length = unit_cell_size).
            Labels: v_l/p_l/v_c for left site, v_c/p_r/v_r for right site (2-site cell).
            For general N-site cells, labels follow the iDMRG convention.
        singular_values: Singular values at each bond in the unit cell.
            For a 2-site cell, singular_values[0] is the bond between sites 0 and 1.
    """

    tensors: list[Tensor]
    singular_values: list[jax.Array]

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        singular_values: list[jax.Array],
    ) -> InfiniteMPS:
        """Wrap existing tensors into an InfiniteMPS."""
        return InfiniteMPS(
            tensors=list(tensors),
            singular_values=list(singular_values),
        )

    # -- Sequence protocol --------------------------------------------------

    @property
    def unit_cell_size(self) -> int:
        """Number of sites in the unit cell."""
        return len(self.tensors)

    def __len__(self) -> int:
        return self.unit_cell_size

    def __getitem__(self, i: int) -> Tensor:
        return self.tensors[i % self.unit_cell_size]

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.tensors)

    # -- Properties ---------------------------------------------------------

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions at each internal bond in the unit cell.

        For each bond between site i and site (i+1) % N, finds the shared
        label and returns its dimension.
        """
        N = self.unit_cell_size
        dims = []
        for i in range(N - 1):
            labels_i = set(self.tensors[i].labels())
            labels_next = set(self.tensors[(i + 1) % N].labels())
            shared = labels_i & labels_next
            if not shared:
                raise ValueError(
                    f"No shared label between site {i} and site {(i + 1) % N}. "
                    f"Labels: {self.tensors[i].labels()}, "
                    f"{self.tensors[(i + 1) % N].labels()}"
                )
            bond_label = shared.pop()
            for idx in self.tensors[i].indices:
                if idx.label == bond_label:
                    dims.append(idx.dim)
                    break
        return dims

    @property
    def is_symmetric(self) -> bool:
        """True if all site tensors are SymmetricTensor."""
        return all(isinstance(t, SymmetricTensor) for t in self.tensors)

    # -- Entanglement -------------------------------------------------------

    def entanglement_entropy(self, bond: int = 0) -> float:
        """Compute the Von Neumann entanglement entropy at a bond.

        Args:
            bond: Bond index within the unit cell.

        Returns:
            The entanglement entropy S = -sum(p * log(p)) where p = sv^2 / sum(sv^2).
        """
        if bond < 0 or bond >= len(self.singular_values):
            raise ValueError(
                f"bond={bond} out of range [0, {len(self.singular_values)})"
            )
        sv = jnp.asarray(self.singular_values[bond])
        sv = sv[sv > 1e-15]
        p = sv**2
        p = p / jnp.sum(p)
        S = -jnp.sum(p * jnp.log(p))
        return float(S)


# ---------------------------------------------------------------------------
# Helpers for FiniteMPS.random()
# ---------------------------------------------------------------------------


def _build_random_dense_tensors(
    L: int, d: int, chi: int, key: jax.Array, dtype: Any
) -> list[Tensor]:
    """Build a list of random dense MPS tensors."""
    from tenax.core.symmetry import U1Symmetry

    sym = U1Symmetry()
    phys_charges = np.zeros(d, dtype=np.int32)
    virt_charges = np.zeros(chi, dtype=np.int32)

    tensors: list[Tensor] = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        if L == 1:
            shape = (d,)
            indices: tuple[TensorIndex, ...] = (
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
            )
        elif i == 0:
            shape = (d, chi)
            indices = (
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            shape = (chi, d)
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
            )
        else:
            shape = (chi, d, chi)
            indices = (
                TensorIndex(sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"),
                TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        data = jax.random.normal(subkey, shape, dtype=dtype)
        data = data / jnp.linalg.norm(data)
        tensors.append(DenseTensor(data, indices))
    return tensors


def _build_random_symmetric_tensors(
    L: int,
    d: int,
    chi: int,
    key: jax.Array,
    symmetry: Any,
    target_charge: int | None,
    dtype: Any,
) -> list[Tensor]:
    """Build a list of random symmetric MPS tensors."""
    if symmetry is None:
        raise ValueError("symmetry must be provided for symmetric MPS")
    if target_charge is None:
        raise ValueError("target_charge must be provided for symmetric MPS")

    # Physical: spin up = +1, spin down = -1 (for d=2)
    phys_charges = np.array([1, -1], dtype=np.int32)[:d]

    # Virtual bond charges
    if target_charge == 0:
        required_charges = [-1, 0, 1]
    else:
        lo = min(-1, target_charge - 1)
        hi = max(1, target_charge + 1)
        required_charges = list(range(lo, hi + 1))

    n_sectors = len(required_charges)
    per_sector = max(1, chi // n_sectors)
    arrays = [np.full(per_sector, q, dtype=np.int32) for q in required_charges]
    virt_charges = np.concatenate(arrays)[:chi]
    if len(virt_charges) < chi:
        mid_q = required_charges[n_sectors // 2]
        pad = np.full(chi - len(virt_charges), mid_q, dtype=np.int32)
        virt_charges = np.concatenate([virt_charges, pad])

    tensors: list[Tensor] = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        site_target = target_charge if i == L - 1 else None

        if i == 0:
            indices: tuple[TensorIndex, ...] = (
                TensorIndex(symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    symmetry, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            indices = (
                TensorIndex(
                    symmetry, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"),
            )
        else:
            indices = (
                TensorIndex(
                    symmetry, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex(symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"),
                TensorIndex(
                    symmetry, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )

        tensor = SymmetricTensor.random_normal(
            indices, key=subkey, dtype=dtype, target=site_target
        )
        tensors.append(tensor)
    return tensors
