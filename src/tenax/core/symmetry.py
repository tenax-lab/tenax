"""Symmetry group definitions for symmetric tensor networks.

This module provides the mathematical foundation for symmetric tensors.
All symmetry classes operate on numpy integer arrays of charge values.
No JAX dependency — pure Python/numpy arithmetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class BraidingStyle(Enum):
    """Classification of particle exchange statistics.

    BOSONIC:   Trivial exchange (sign = +1 always).
    FERMIONIC: Anti-commuting exchange (sign = (-1)^{p_a * p_b}).
    ANYONIC:   General phase on exchange (reserved for future use).
    """

    BOSONIC = "bosonic"
    FERMIONIC = "fermionic"
    ANYONIC = "anyonic"


class BaseSymmetry(ABC):
    """Abstract base for symmetry groups governing tensor index charges.

    A symmetry group defines how charges combine (fuse) when tensor legs
    are merged, and how charges transform when a leg's flow is reversed (dual).

    Concrete subclasses must implement fuse, dual, identity, and n_values.
    All concrete classes should implement __eq__ and __hash__ so they can
    serve as dict keys and be compared for compatibility checks.
    """

    @abstractmethod
    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        """Fuse two charge arrays element-wise under group multiplication.

        Args:
            charges_a: Integer charge array of shape (D,).
            charges_b: Integer charge array of shape (D,).

        Returns:
            Fused charge array of shape (D,).
        """

    @abstractmethod
    def dual(self, charges: np.ndarray) -> np.ndarray:
        """Return the group inverse (dual) of each charge.

        Used to flip a leg's flow direction while preserving compatibility.

        Args:
            charges: Integer charge array of shape (D,).

        Returns:
            Dual charge array of shape (D,).
        """

    @abstractmethod
    def identity(self) -> int:
        """Return the identity element (neutral charge, typically 0)."""

    @abstractmethod
    def n_values(self) -> int | None:
        """Return the number of distinct charge values, or None if infinite (U(1))."""

    def fuse_many(self, charge_list: list[np.ndarray]) -> np.ndarray:
        """Fuse a list of charge arrays left-to-right via repeated fuse().

        Args:
            charge_list: Non-empty list of integer charge arrays, all shape (D,).

        Returns:
            Fully fused charge array of shape (D,).
        """
        if not charge_list:
            raise ValueError("charge_list must be non-empty")
        result = charge_list[0]
        for c in charge_list[1:]:
            result = self.fuse(result, c)
        return result

    @property
    def braiding_style(self) -> BraidingStyle:
        """Exchange statistics of this symmetry (bosonic by default)."""
        return BraidingStyle.BOSONIC

    @property
    def is_fermionic(self) -> bool:
        """True if this symmetry has fermionic exchange statistics."""
        return self.braiding_style == BraidingStyle.FERMIONIC

    def parity(self, charges: np.ndarray) -> np.ndarray:
        """Return the Z2 parity grading of each charge (0=even, 1=odd).

        For bosonic symmetries, all charges have even parity.

        Args:
            charges: Integer charge array of shape (D,).

        Returns:
            Integer array of shape (D,) with values in {0, 1}.
        """
        return np.zeros_like(charges)

    def exchange_sign(self, charge_a: int, charge_b: int) -> int:
        """Sign from exchanging two particles with given charges.

        Returns (-1)^{p_a * p_b} where p is the parity grading.
        For bosonic symmetries, always returns +1.

        Args:
            charge_a: Charge of first particle.
            charge_b: Charge of second particle.

        Returns:
            +1 or -1.
        """
        return 1

    def exchange_phase(self, charge_a: int, charge_b: int) -> complex:
        """Phase from exchanging two particles (generalizes exchange_sign).

        For fermionic symmetries this equals exchange_sign. For anyonic
        symmetries this can be a general complex phase.

        Args:
            charge_a: Charge of first particle.
            charge_b: Charge of second particle.

        Returns:
            Complex phase factor.
        """
        return complex(self.exchange_sign(charge_a, charge_b))

    def twist_phase(self, charge: int) -> complex:
        """Topological twist (ribbon element) for a charge sector.

        For fermions: (-1)^p where p is the parity of the charge.
        For bosons: always 1.

        Args:
            charge: Charge value.

        Returns:
            Complex phase factor.
        """
        return 1.0

    def is_conserved(
        self,
        charges_per_leg: list[np.ndarray],
        flows: list[int],
        target: int | None = None,
    ) -> bool:
        """Check if a single charge combination satisfies conservation.

        Args:
            charges_per_leg: List of scalar-or-array charge values per leg.
            flows: List of +1 (IN) or -1 (OUT) per leg.
            target: Required net charge; defaults to identity().

        Returns:
            True if the net charge equals target.
        """
        if target is None:
            target = self.identity()
        net = sum(int(f) * int(q) for f, q in zip(flows, charges_per_leg))
        # For modular groups we need to reduce the net charge
        n = self.n_values()
        if n is not None:
            net = net % n
        return net == target


class U1Symmetry(BaseSymmetry):
    """U(1) symmetry: integer charges, fusion by addition.

    Represents the continuous U(1) group (particle number conservation,
    total Sz conservation, etc.). Charges are unbounded integers.

    Example:
        >>> sym = U1Symmetry()
        >>> sym.fuse(np.array([0, 1, -1]), np.array([1, -1, 0]))
        array([1, 0, -1])
    """

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return charges_a + charges_b

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return -charges

    def identity(self) -> int:
        return 0

    def n_values(self) -> None:
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, U1Symmetry)

    def __hash__(self) -> int:
        return hash("U1Symmetry")

    def __repr__(self) -> str:
        return "U1Symmetry()"


class ZnSymmetry(BaseSymmetry):
    """Z_n symmetry: integer charges mod n, fusion by addition mod n.

    Represents the discrete cyclic group Z_n (e.g., Z_2 for fermion parity,
    Z_3 for three-state Potts model, etc.).

    Args:
        n: The order of the group. Must be >= 2.

    Example:
        >>> sym = ZnSymmetry(3)
        >>> sym.fuse(np.array([1, 2]), np.array([2, 2]))
        array([0, 1])
    """

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = n

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return (charges_a + charges_b) % self.n

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return (-charges) % self.n

    def identity(self) -> int:
        return 0

    def n_values(self) -> int:
        return self.n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZnSymmetry) and self.n == other.n

    def __hash__(self) -> int:
        return hash(("ZnSymmetry", self.n))

    def __repr__(self) -> str:
        return f"ZnSymmetry({self.n})"


class BaseNonAbelianSymmetry(BaseSymmetry):
    """Stub base class for non-Abelian symmetries (e.g., SU(2)).

    Non-Abelian symmetries require Clebsch-Gordan / recoupling coefficients
    for tensor contractions, making them significantly more complex than
    Abelian symmetries. This base provides the interface contract for future
    concrete implementations.

    Note:
        Concrete subclasses must implement recoupling_coefficients and
        allowed_fusions in addition to the BaseSymmetry abstract methods.
        The fuse/dual/identity/n_values methods for non-Abelian groups
        operate on irrep labels (e.g., spin quantum numbers).
    """

    @abstractmethod
    def recoupling_coefficients(
        self,
        j1: int,
        j2: int,
        j3: int,
    ) -> np.ndarray:
        """Return Clebsch-Gordan or 6j-symbol coefficients for recoupling.

        These coefficients are needed when contracting non-Abelian symmetric
        tensors. The specific form depends on the group (CG coefficients for
        SU(2), etc.).

        Args:
            j1: First irrep label.
            j2: Second irrep label.
            j3: Output irrep label (must be in allowed_fusions(j1, j2)).

        Returns:
            Coefficient array for the (j1, j2) -> j3 channel.
        """

    @abstractmethod
    def allowed_fusions(self, j1: int, j2: int) -> list[int]:
        """Return list of irrep labels appearing in tensor product j1 x j2.

        For SU(2) with spin quantum numbers j1, j2, this returns
        ``[abs(j1 - j2), ..., j1 + j2]`` (triangular rule).

        Args:
            j1: First irrep label.
            j2: Second irrep label.

        Returns:
            Sorted list of irrep labels in the decomposition.
        """


class FermionParity(BaseSymmetry):
    """Z2 symmetry with fermionic braiding statistics.

    Charges are 0 (even/bosonic) or 1 (odd/fermionic). Fusion is
    addition mod 2. Exchanging two odd-parity objects yields a minus sign.

    Example:
        >>> sym = FermionParity()
        >>> sym.exchange_sign(1, 1)
        -1
        >>> sym.exchange_sign(0, 1)
        1
    """

    @property
    def braiding_style(self) -> BraidingStyle:
        return BraidingStyle.FERMIONIC

    @property
    def is_fermionic(self) -> bool:
        return True

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return (charges_a + charges_b) % 2

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return charges  # Z2 is self-dual

    def identity(self) -> int:
        return 0

    def n_values(self) -> int:
        return 2

    def parity(self, charges: np.ndarray) -> np.ndarray:
        return np.asarray(charges % 2, dtype=np.int32)

    def exchange_sign(self, charge_a: int, charge_b: int) -> int:
        return 1 - 2 * ((charge_a % 2) * (charge_b % 2))

    def exchange_phase(self, charge_a: int, charge_b: int) -> complex:
        return complex(self.exchange_sign(charge_a, charge_b))

    def twist_phase(self, charge: int) -> complex:
        return (-1.0) ** (charge % 2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FermionParity)

    def __hash__(self) -> int:
        return hash("FermionParity")

    def __repr__(self) -> str:
        return "FermionParity()"


# Built-in grading functions for FermionicU1
_GRADING_REGISTRY: dict[str, object] = {
    "abs_mod_2": lambda q: abs(int(q)) % 2,
    "mod_2": lambda q: int(q) % 2,
}


class FermionicU1(BaseSymmetry):
    """U(1) symmetry with fermionic exchange statistics.

    Charges are unbounded integers (like U1Symmetry), but a grading
    function maps each charge to a Z2 parity (0=even, 1=odd).

    Args:
        grading: Callable mapping int -> {0, 1}. If None, uses the
            default ``|q| % 2`` (odd particle number = fermionic).
        grading_key: String key identifying the grading function.
            Used for equality/hash so that two FermionicU1 instances
            with the same grading are considered equal.

    Example:
        >>> sym = FermionicU1()
        >>> sym.parity(np.array([0, 1, 2, 3]))
        array([0, 1, 0, 1])
    """

    def __init__(
        self,
        grading: object = None,
        grading_key: str = "abs_mod_2",
    ) -> None:
        if grading is not None:
            self._grading = grading
        elif grading_key in _GRADING_REGISTRY:
            self._grading = _GRADING_REGISTRY[grading_key]
        else:
            raise ValueError(f"Unknown grading_key: {grading_key!r}")
        self._grading_key = grading_key

    @property
    def braiding_style(self) -> BraidingStyle:
        return BraidingStyle.FERMIONIC

    @property
    def is_fermionic(self) -> bool:
        return True

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return charges_a + charges_b

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return -charges

    def identity(self) -> int:
        return 0

    def n_values(self) -> None:
        return None

    def parity(self, charges: np.ndarray) -> np.ndarray:
        return np.array([self._grading(q) for q in charges], dtype=np.int32)

    def exchange_sign(self, charge_a: int, charge_b: int) -> int:
        pa = self._grading(charge_a)
        pb = self._grading(charge_b)
        return 1 - 2 * (pa * pb)

    def exchange_phase(self, charge_a: int, charge_b: int) -> complex:
        return complex(self.exchange_sign(charge_a, charge_b))

    def twist_phase(self, charge: int) -> complex:
        return (-1.0) ** self._grading(charge)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FermionicU1) and self._grading_key == other._grading_key
        )

    def __hash__(self) -> int:
        return hash(("FermionicU1", self._grading_key))

    def __repr__(self) -> str:
        return f"FermionicU1(grading_key={self._grading_key!r})"


class ProductSymmetry(BaseSymmetry):
    """Direct product of two symmetries with bit-packed charges.

    Charges from the two factor symmetries are encoded into a single int32
    via bit-packing: ``encoded = (q2 << 16) | (q1 & 0xFFFF)``.
    Component charges are limited to the int16 range [-32768, 32767].

    Args:
        sym1: First factor symmetry.
        sym2: Second factor symmetry.

    Raises:
        TypeError: If either factor is itself a ProductSymmetry (no nesting).

    Limitations:
        Only two factors are supported.  Three-factor groups
        (e.g., U(1)×U(1)×Z₂) would require a ``MultiProductSymmetry``
        with wider bit-packing or variable-length charge vectors.

    Example:
        >>> sym = ProductSymmetry(FermionParity(), U1Symmetry())
        >>> encoded = ProductSymmetry.encode(1, 3)
        >>> ProductSymmetry.decode(encoded)
        (1, 3)
    """

    def __init__(self, sym1: BaseSymmetry, sym2: BaseSymmetry) -> None:
        if isinstance(sym1, ProductSymmetry) or isinstance(sym2, ProductSymmetry):
            raise TypeError("Nested ProductSymmetry is not supported")
        self.sym1 = sym1
        self.sym2 = sym2

    @staticmethod
    def encode(q1: int, q2: int) -> int:
        """Pack two int16 charges into one int32."""
        return (int(np.int16(q2)) << 16) | (int(np.int16(q1)) & 0xFFFF)

    @staticmethod
    def decode(encoded: int) -> tuple[int, int]:
        """Unpack one int32 into two int16 charges."""
        # Use numpy array to handle signed conversion correctly
        packed = np.array([encoded], dtype=np.int32)
        q1 = int((packed & 0xFFFF).astype(np.uint16).view(np.int16)[0])
        q2 = int(((packed >> 16) & 0xFFFF).astype(np.uint16).view(np.int16)[0])
        return q1, q2

    @staticmethod
    def encode_charges(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Vectorized encoding of two charge arrays."""
        a1 = arr1.astype(np.int16).astype(np.int32)
        a2 = arr2.astype(np.int16).astype(np.int32)
        return ((a2 << 16) | (a1 & 0xFFFF)).astype(np.int32)

    @staticmethod
    def decode_charges(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized decoding of a packed charge array."""
        arr = np.asarray(arr, dtype=np.int32)
        q1 = (arr & 0xFFFF).astype(np.int16).astype(np.int32)
        q2 = ((arr >> 16) & 0xFFFF).astype(np.int16).astype(np.int32)
        return q1, q2

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        a1, a2 = self.decode_charges(charges_a)
        b1, b2 = self.decode_charges(charges_b)
        return self.encode_charges(self.sym1.fuse(a1, b1), self.sym2.fuse(a2, b2))

    def dual(self, charges: np.ndarray) -> np.ndarray:
        q1, q2 = self.decode_charges(charges)
        return self.encode_charges(self.sym1.dual(q1), self.sym2.dual(q2))

    def identity(self) -> int:
        return self.encode(self.sym1.identity(), self.sym2.identity())

    def n_values(self) -> int | None:
        n1 = self.sym1.n_values()
        n2 = self.sym2.n_values()
        if n1 is not None and n2 is not None:
            return n1 * n2
        return None

    @property
    def braiding_style(self) -> BraidingStyle:
        styles = {self.sym1.braiding_style, self.sym2.braiding_style}
        if BraidingStyle.ANYONIC in styles:
            return BraidingStyle.ANYONIC
        if BraidingStyle.FERMIONIC in styles:
            return BraidingStyle.FERMIONIC
        return BraidingStyle.BOSONIC

    @property
    def is_fermionic(self) -> bool:
        return self.braiding_style == BraidingStyle.FERMIONIC

    def parity(self, charges: np.ndarray) -> np.ndarray:
        q1, q2 = self.decode_charges(charges)
        return (self.sym1.parity(q1) + self.sym2.parity(q2)) % 2

    def exchange_sign(self, charge_a: int, charge_b: int) -> int:
        a1, a2 = self.decode(charge_a)
        b1, b2 = self.decode(charge_b)
        pa = (
            self.sym1.parity(np.array([a1]))[0] + self.sym2.parity(np.array([a2]))[0]
        ) % 2
        pb = (
            self.sym1.parity(np.array([b1]))[0] + self.sym2.parity(np.array([b2]))[0]
        ) % 2
        return 1 - 2 * (pa * pb)

    def exchange_phase(self, charge_a: int, charge_b: int) -> complex:
        return complex(self.exchange_sign(charge_a, charge_b))

    def twist_phase(self, charge: int) -> complex:
        q1, q2 = self.decode(charge)
        p = (
            self.sym1.parity(np.array([q1]))[0] + self.sym2.parity(np.array([q2]))[0]
        ) % 2
        return (-1.0) ** p

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ProductSymmetry)
            and self.sym1 == other.sym1
            and self.sym2 == other.sym2
        )

    def __hash__(self) -> int:
        return hash(("ProductSymmetry", self.sym1, self.sym2))

    def __repr__(self) -> str:
        return f"ProductSymmetry({self.sym1!r}, {self.sym2!r})"
