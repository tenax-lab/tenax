"""Finite and infinite MPS containers with canonical form tracking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import qr, svd


@dataclass
class FiniteMPS:
    """Finite matrix product state with canonical form tracking.

    The state represented is |psi> = exp(log_norm) * |psi_tensors>, where
    |psi_tensors> is the state encoded by the site tensors alone.

    Attributes:
        tensors: Site tensors, length L.  All sites are 3-leg
            (left-bond x physical x right-bond), including boundaries which
            have a trivial (dimension-1) bond on the open end.
            Labels follow the convention p{i} for physical, v{i}_{i+1} for bonds.
        orth_center: Position of the orthogonality center, or None if the
            canonical form is unknown.  When set, sites 0..orth_center-1 are
            left-canonical and sites orth_center+1..L-1 are right-canonical.
        singular_values: Singular values at each bond (length L-1).
            Entry i holds the singular values between sites i and i+1,
            or None if not yet computed.
        log_norm: Logarithm of the norm factor.  After canonicalization the
            singular values are normalized (sum(sv**2) == 1) and the original
            norm information is stored here so that
            ``norm() == exp(log_norm) * sqrt(<tensors|tensors>)``.
        target_charge: Total quantum number sector for symmetric MPS, or None
            for dense MPS.  Set on construction and propagated through all
            methods that return new FiniteMPS instances.
    """

    tensors: list[Tensor]
    orth_center: int | None = None
    singular_values: list[jnp.ndarray | None] = field(default_factory=list)
    log_norm: float = 0.0
    target_charge: int | None = None

    def __post_init__(self):
        if not self.singular_values:
            self.singular_values = [None] * max(len(self.tensors) - 1, 0)

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        orth_center: int | None = None,
        singular_values: list[jnp.ndarray | None] | None = None,
        log_norm: float = 0.0,
        target_charge: int | None = None,
        verify: bool = False,
    ) -> FiniteMPS:
        """Wrap existing site tensors into a FiniteMPS.

        Args:
            tensors: Site tensors.
            orth_center: Position of the orthogonality center, or None.
            singular_values: Singular values at each bond.
            log_norm: Logarithm of the norm factor.
            verify: If True and orth_center is not None, verify that sites
                left of orth_center are left-canonical and sites right of
                orth_center are right-canonical.  Raises ValueError if not.
        """
        if verify and orth_center is not None:
            _verify_orthogonality(tensors, orth_center)
        L = len(tensors)
        if singular_values is None:
            singular_values = [None] * max(L - 1, 0)
        return FiniteMPS(
            tensors=list(tensors),
            orth_center=orth_center,
            singular_values=singular_values,
            log_norm=log_norm,
            target_charge=target_charge,
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
        mps = FiniteMPS.from_tensors(tensors, target_charge=target_charge)
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
        new_log_norm = self.log_norm
        if center < L - 1:
            site = tensors[center]
            right_bond = f"v{center}_{center + 1}"
            left_labels = [lb for lb in site.labels() if lb != right_bond]
            tmp_bond = f"_svd_{right_bond}"
            U, s, Vh, s_full = svd(
                site, left_labels, [right_bond], new_bond_label=tmp_bond
            )

            # Normalize singular values and accumulate norm into log_norm
            sv_norm_sq = jnp.sum(s_full**2)
            sv_norm_sq = jnp.where(sv_norm_sq > 0, sv_norm_sq, 1.0)
            inv_sv_norm = 1.0 / jnp.sqrt(sv_norm_sq)
            s = s * inv_sv_norm
            s_full = s_full * inv_sv_norm
            new_log_norm = self.log_norm + 0.5 * float(jnp.log(sv_norm_sq))

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
            log_norm=new_log_norm,
            target_charge=self.target_charge,
        )

    # -- Norm / Overlap -----------------------------------------------------

    def _raw_overlap(self, other: FiniteMPS) -> complex:
        """Compute <self_tensors|other_tensors> ignoring log_norm."""
        if len(self) != len(other):
            raise ValueError("MPS lengths must match for overlap")

        env = None
        for i in range(self.L):
            ket = other[i]
            # ``bar()``, not ``conj()``: the bra must have *opposite flows* so
            # its legs can contract, while keeping identical charges so blocks
            # still match by equality.  ``conj()`` only conjugates the data and
            # leaves the flows alone, so on a SymmetricTensor every physical
            # pairing is OUT-against-OUT, no block matches, and the whole
            # overlap collapses to exactly 0j -- see #819.  ``dagger()`` flips
            # the flows but also duals the charges, which reintroduces the
            # charge mismatch ``bar()`` was written to avoid.
            bra = self[i].bar()

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

        # env should be a scalar (possibly with trivial dim-1 boundary legs);
        # squeeze to extract value
        result = env.todense()
        return complex(result.reshape(()))

    def overlap(self, other: FiniteMPS) -> complex:
        """Compute <self|other> via left-to-right transfer matrix contraction.

        Accounts for log_norm of both MPS:
        <self|other> = exp(self.log_norm + other.log_norm) * <self_tensors|other_tensors>.

        Args:
            other: Another FiniteMPS with the same length and physical dims.

        Returns:
            The scalar overlap <self|other>.
        """
        raw = self._raw_overlap(other)
        return complex(jnp.exp(self.log_norm + other.log_norm) * raw)

    def norm(self) -> float:
        """Compute the norm ||psi|| = exp(log_norm) * sqrt(<tensors|tensors>).

        Returns:
            The norm as a non-negative real number.
        """
        # Fast path, following ITensor: with a known orthogonality center every
        # other site is an isometry, so <t|t> is just the center tensor's
        # Frobenius norm squared.  O(chi^2 d) instead of O(L chi^3) -- and it
        # cannot be corrupted by a bug in the transfer-matrix contraction,
        # which is exactly how #819 went unnoticed.
        if self.orth_center is not None:
            centre = float(self.tensors[self.orth_center].norm())
            return float(jnp.exp(self.log_norm) * centre)

        raw = self._raw_overlap(self)
        # <t|t> is a positive real by construction.  Anything else means the
        # contraction is broken, and the old ``jnp.abs(raw)`` laundered exactly
        # that into a plausible non-negative float: #819 returned 0.0 for a
        # perfectly good state and nothing complained.  Fail instead.
        raw_c = complex(raw)
        scale = abs(raw_c)
        if scale > 0.0 and abs(raw_c.imag) > 1e-8 * scale:
            raise ValueError(
                f"<psi|psi> came back complex ({raw_c!r}); the norm of a state "
                "is a positive real, so the overlap contraction is wrong. This "
                "is a bug in tenax, not in your state (#819)."
            )
        if raw_c.real < 0.0:
            raise ValueError(
                f"<psi|psi> came back negative ({raw_c.real!r}); the norm of a "
                "state is a positive real, so the overlap contraction is wrong. "
                "This is a bug in tenax, not in your state (#819)."
            )
        if raw_c.real == 0.0 and any(float(t.norm()) > 0.0 for t in self.tensors):
            raise ValueError(
                "<psi|psi> came back exactly 0 for an MPS whose site tensors "
                "are not all zero, so the overlap contraction is wrong rather "
                "than the state being null (#819). A genuinely zero state "
                "would have zero tensors."
            )
        return float(jnp.exp(self.log_norm) * jnp.sqrt(raw_c.real))

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

    def compute_singular_values(self) -> FiniteMPS:
        """Compute singular values at ALL bonds via a single SVD sweep.

        Returns a new FiniteMPS in right-canonical form (orth_center=0) with
        all ``singular_values`` entries populated and normalized (sum(sv²)=1
        at each bond).

        Algorithm:
            1. Start from left-canonical form (orth_center = L-1).
            2. Sweep right-to-left with SVD (no truncation) at each bond:
               for i = L-1 down to 1, SVD site i, store normalized SVs,
               absorb U·s into site i-1.
            3. Result: right-canonical MPS with all SVs populated.
        """
        L = self.L
        if L <= 1:
            return FiniteMPS(
                tensors=list(self.tensors),
                orth_center=0,
                singular_values=[],
                log_norm=self.log_norm,
                target_charge=self.target_charge,
            )

        # Step 1: ensure left-canonical form
        if self.orth_center != L - 1:
            mps = self.left_canonicalize()
        else:
            mps = FiniteMPS(
                tensors=list(self.tensors),
                orth_center=self.orth_center,
                singular_values=list(self.singular_values),
                log_norm=self.log_norm,
                target_charge=self.target_charge,
            )

        tensors = list(mps.tensors)
        sv_list: list[jnp.ndarray | None] = [None] * (L - 1)
        new_log_norm = mps.log_norm

        # Step 2: sweep right-to-left with SVD
        for i in range(L - 1, 0, -1):
            site = tensors[i]
            left_bond = f"v{i - 1}_{i}"
            right_labels = [lb for lb in site.labels() if lb != left_bond]
            tmp_bond = f"_svd_{left_bond}"

            U, s, Vh, s_full = svd(
                site, [left_bond], right_labels, new_bond_label=tmp_bond
            )

            # Normalize singular values
            sv_norm_sq = jnp.sum(s_full**2)
            sv_norm_sq = jnp.where(sv_norm_sq > 0, sv_norm_sq, 1.0)
            inv_sv_norm = 1.0 / jnp.sqrt(sv_norm_sq)
            s = s * inv_sv_norm
            s_full = s_full * inv_sv_norm
            new_log_norm = new_log_norm + 0.5 * float(jnp.log(sv_norm_sq))

            sv_list[i - 1] = s_full

            # Vh has (tmp_bond, right_labels...) — rename tmp_bond -> left_bond
            # Reorder so left_bond is first
            Vh_relabeled = Vh.relabel(tmp_bond, left_bond)
            labels = Vh_relabeled.labels()
            bond_pos = labels.index(left_bond)
            if bond_pos != 0:
                axes = (bond_pos,) + tuple(
                    j for j in range(len(labels)) if j != bond_pos
                )
                Vh_relabeled = Vh_relabeled.transpose(axes)
            tensors[i] = Vh_relabeled

            # Absorb U·diag(s) into site i-1
            US = scale_bond_axis(U, tmp_bond, s)
            absorbed = contract(tensors[i - 1], US)
            tensors[i - 1] = absorbed.relabel(tmp_bond, left_bond)

        return FiniteMPS(
            tensors=tensors,
            orth_center=0,
            singular_values=sv_list,
            log_norm=new_log_norm,
            target_charge=self.target_charge,
        )


@dataclass
class InfiniteMPS:
    """Infinite matrix product state with a finite unit cell.

    Attributes:
        tensors: Unit cell site tensors (length L = unit_cell_size).
            Labels: v_l/p_l/v_c for left site, v_c/p_r/v_r for right site (2-site cell).
            For general N-site cells, labels follow the iDMRG convention.
        singular_values: Singular values at each bond, length L+1.
            For a 2-site cell (L=2):
              - singular_values[0]: left boundary bond (between adjacent unit cells)
              - singular_values[1]: centre bond (between sites 0 and 1)
              - singular_values[2]: right boundary bond (same as [0], shifted by qshift)
        qshift: Charge per unit cell for U(1)-symmetric iMPS, or None for dense.
        log_norm: Logarithm of the norm factor (consistent with FiniteMPS).
    """

    tensors: list[Tensor]
    singular_values: list[jax.Array]
    qshift: int | None = None
    log_norm: float = 0.0

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        singular_values: list[jax.Array],
        qshift: int | None = None,
        log_norm: float = 0.0,
    ) -> InfiniteMPS:
        """Wrap existing tensors into an InfiniteMPS."""
        return InfiniteMPS(
            tensors=list(tensors),
            singular_values=list(singular_values),
            qshift=qshift,
            log_norm=log_norm,
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
# Orthogonality verification helper
# ---------------------------------------------------------------------------


def _verify_orthogonality(
    tensors: list[Tensor], orth_center: int, atol: float = 1e-10
) -> None:
    """Verify canonical form: left sites are left-canonical, right are right-canonical.

    Raises:
        ValueError: If any site fails the isometry check.
    """
    for i in range(orth_center):
        d = tensors[i].todense()
        mat = d.reshape(-1, d.shape[-1])
        eye = mat.conj().T @ mat
        if not jnp.allclose(eye, jnp.eye(eye.shape[0]), atol=atol):
            raise ValueError(f"Site {i} is not left-canonical (A†A ≠ I)")
    for i in range(orth_center + 1, len(tensors)):
        d = tensors[i].todense()
        mat = d.reshape(d.shape[0], -1)
        eye = mat @ mat.conj().T
        if not jnp.allclose(eye, jnp.eye(eye.shape[0]), atol=atol):
            raise ValueError(f"Site {i} is not right-canonical (AA† ≠ I)")


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

    trivial_charges = np.zeros(1, dtype=np.int32)

    tensors: list[Tensor] = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        if L == 1:
            shape = (1, d, 1)
            indices: tuple[TensorIndex, ...] = (
                TensorIndex.from_charges(
                    sym, trivial_charges, FlowDirection.IN, label="v_-1_0"
                ),
                TensorIndex.from_charges(
                    sym, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    sym, trivial_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == 0:
            shape = (1, d, chi)
            indices = (
                TensorIndex.from_charges(
                    sym, trivial_charges, FlowDirection.IN, label="v_-1_0"
                ),
                TensorIndex.from_charges(
                    sym, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    sym, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            shape = (chi, d, 1)
            indices = (
                TensorIndex.from_charges(
                    sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex.from_charges(
                    sym, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    sym, trivial_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        else:
            shape = (chi, d, chi)
            indices = (
                TensorIndex.from_charges(
                    sym, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex.from_charges(
                    sym, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
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

    trivial_zero = np.array([0], dtype=np.int32)

    tensors: list[Tensor] = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        site_target = target_charge if i == L - 1 else None

        if L == 1:
            trivial_target = np.array([0], dtype=np.int32)
            indices: tuple[TensorIndex, ...] = (
                TensorIndex.from_charges(
                    symmetry, trivial_zero, FlowDirection.IN, label="v_-1_0"
                ),
                TensorIndex.from_charges(
                    symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, trivial_target, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == 0:
            indices = (
                TensorIndex.from_charges(
                    symmetry, trivial_zero, FlowDirection.IN, label="v_-1_0"
                ),
                TensorIndex.from_charges(
                    symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        elif i == L - 1:
            trivial_target = np.array([0], dtype=np.int32)
            indices = (
                TensorIndex.from_charges(
                    symmetry, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, trivial_target, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )
        else:
            indices = (
                TensorIndex.from_charges(
                    symmetry, virt_charges, FlowDirection.IN, label=f"v{i - 1}_{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, phys_charges, FlowDirection.IN, label=f"p{i}"
                ),
                TensorIndex.from_charges(
                    symmetry, virt_charges, FlowDirection.OUT, label=f"v{i}_{i + 1}"
                ),
            )

        tensor = SymmetricTensor.random_normal(
            indices, key=subkey, dtype=dtype, target=site_target
        )
        tensors.append(tensor)
    return tensors
