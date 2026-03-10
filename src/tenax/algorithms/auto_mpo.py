"""AutoMPO: Automatic MPO construction for arbitrary Hamiltonians.

Builds Matrix Product Operators from symbolic Hamiltonian descriptions using
a finite-automaton (left-partial-state) approach.  Each Hamiltonian term is
a product of local operators at specified sites:

    H = sum_k  c_k * O_{s0}^(k) ⊗ O_{s1}^(k) ⊗ ... ⊗ O_{sm}^(k)

At each internal bond j, a term is *in-flight* when min_site <= j < max_site.
The MPO bond dimension = (# in-flight terms) + 2 (done + vacuum states), giving
a correct—if uncompressed—MPO.  An optional SVD compression pass reduces bond
dimension toward the optimal.

State index convention (matches build_mpo_heisenberg):
  0           = "done"   (accumulated result, passes through with identity)
  1 .. n      = one per in-flight term (ordered by term enumeration)
  n + 1 = D-1 = "vacuum" (not yet started, passes through with identity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tenax.network.network import TensorNetwork

# ---------------------------------------------------------------------------
# Built-in operator sets
# ---------------------------------------------------------------------------


def spin_half_ops() -> dict[str, np.ndarray]:
    """Standard spin-1/2 single-site operators (d=2).

    Returns a dict with keys "Sz", "Sp", "Sm", "Id".
    """
    return {
        "Sz": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64),
        "Sp": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        "Sm": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "Id": np.eye(2, dtype=np.float64),
    }


def spin_one_ops() -> dict[str, np.ndarray]:
    """Standard spin-1 single-site operators (d=3).

    Basis ordering: ``|m=+1>``, ``|m=0>``, ``|m=-1>`` mapping to indices 0, 1, 2.
    Returns a dict with keys "Sz", "Sp", "Sm", "Id".
    """
    sq2 = np.sqrt(2.0)
    return {
        "Sz": np.diag([1.0, 0.0, -1.0]).astype(np.float64),
        "Sp": np.array(
            [[0.0, sq2, 0.0], [0.0, 0.0, sq2], [0.0, 0.0, 0.0]], dtype=np.float64
        ),
        "Sm": np.array(
            [[0.0, 0.0, 0.0], [sq2, 0.0, 0.0], [0.0, sq2, 0.0]], dtype=np.float64
        ),
        "Id": np.eye(3, dtype=np.float64),
    }


def fermion_site_ops() -> dict[str, np.ndarray]:
    """Spinless fermion single-site operators (d=2).

    Basis ordering: ``|0>`` (empty), ``|1>`` (occupied).
    Returns a dict with keys "C", "Cd", "N", "F", "Id".

    - C  = annihilation operator
    - Cd = creation operator (C†)
    - N  = number operator (Cd @ C)
    - F  = Jordan-Wigner string operator (-1)^N = diag(1, -1)
    - Id = identity
    """
    return {
        "C": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        "Cd": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        "N": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "F": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
        "Id": np.eye(2, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianTerm:
    """One term in the Hamiltonian: coefficient * product of local operators.

    Attributes:
        coefficient: Scalar (complex) prefactor.
        ops: Tuple of (site, operator_matrix) pairs, sorted by site.
    """

    coefficient: complex
    ops: tuple[tuple[int, np.ndarray], ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_bond_states(terms: list[HamiltonianTerm], L: int) -> list[dict[int, int]]:
    """Assign MPO bond-state indices to in-flight terms at each internal bond.

    Returns ``bond_states`` where ``bond_states[j][t_id]`` is the state index
    for term ``t_id`` at bond ``j`` (between sites ``j`` and ``j+1``).

    Done = 0 and vacuum = n_active+1 are *not* stored in this dict; callers
    derive them from ``len(bond_states[j]) + 1``.
    """
    bond_states: list[dict[int, int]] = []
    for j in range(L - 1):
        states: dict[int, int] = {}
        idx = 1
        for t_id, term in enumerate(terms):
            sites = [s for s, _ in term.ops]
            if min(sites) <= j < max(sites):
                states[t_id] = idx
                idx += 1
        bond_states.append(states)
    return bond_states


def _build_w_matrices(
    terms: list[HamiltonianTerm],
    bond_states: list[dict[int, int]],
    L: int,
    d: int,
    identity: np.ndarray,
) -> list[np.ndarray]:
    """Build a W-matrix for every site.

    Shape conventions:
      * Left boundary  (i=0):   (1,   d, d, D_r) — left dim = 1 (vacuum only)
      * Right boundary (i=L-1): (D_l, d, d, 1  ) — right dim = 1 (done only)
      * Bulk sites:             (D_l, d, d, D_r)

    State index encoding:
      done = 0, in-flight 1..n, vacuum = D-1 (last index).
    """
    # NumPy >=2.0 forbids adding complex128 into a float64 array.  Use
    # complex128 only when at least one coefficient has a nonzero imaginary part.
    needs_complex = any(t.coefficient.imag != 0.0 for t in terms)
    w_dtype = np.complex128 if needs_complex else np.float64

    w_mats: list[np.ndarray] = []

    for i in range(L):
        # ----- bond dimensions -----
        D_l = 1 if i == 0 else len(bond_states[i - 1]) + 2
        D_r = 1 if i == L - 1 else len(bond_states[i]) + 2

        W = np.zeros((D_l, d, d, D_r), dtype=w_dtype)

        # State indices on each side
        # Left boundary: single state acts as vacuum (index 0).
        # Right boundary: single state acts as done (index 0).
        vac_l = 0 if i == 0 else D_l - 1
        done_l = 0
        done_r = 0
        vac_r = D_r - 1  # for right boundary D_r=1 → vac_r=0, but never written there

        # ----- pass-through identities -----
        if L == 1:
            pass  # No bonds → no pass-throughs; W is purely the operator sum
        elif i == 0:
            # Left boundary: vacuum propagates to vacuum on the right
            W[0, :, :, vac_r] = identity
        elif i == L - 1:
            # Right boundary: done propagates from done on the left
            W[done_l, :, :, 0] = identity
        else:
            # Bulk site: both done→done and vac→vac pass-throughs
            W[done_l, :, :, done_r] = identity
            W[vac_l, :, :, vac_r] = identity

        # ----- fill each Hamiltonian term -----
        for t_id, term in enumerate(terms):
            op_dict = dict(term.ops)
            sites = sorted(op_dict.keys())
            min_s, max_s = sites[0], sites[-1]

            if i < min_s or i > max_s:
                continue  # term not yet started or already finished

            op = op_dict.get(i, identity)
            # Use real scalar when imaginary part is zero to stay within the W dtype.
            coeff = term.coefficient if needs_complex else term.coefficient.real

            if min_s == max_s:
                # Single-site term: vacuum → done in one step
                W[vac_l, :, :, done_r] += coeff * op

            elif i == min_s:
                # First site of multi-site term: absorb coefficient here
                state_r = bond_states[i][t_id]
                W[vac_l, :, :, state_r] += coeff * op

            elif i == max_s:
                # Last site: close into done (coefficient already absorbed)
                state_l = bond_states[i - 1][t_id]
                W[state_l, :, :, done_r] += op

            else:
                # Intermediate site: propagate in-flight state
                # Use SET (not +=) — each term owns unique state indices,
                # so no two terms write the same (state_l, state_r) pair.
                state_l = bond_states[i - 1][t_id]
                state_r = bond_states[i][t_id]
                W[state_l, :, :, state_r] = op

        w_mats.append(W)

    return w_mats


def _compress_mpo_bond(
    w_left: np.ndarray,
    w_right: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD-compress one MPO bond between two adjacent W-matrices.

    Performs a single left-to-right compression step: the left singular
    vectors are absorbed into ``w_left``; the right factor is applied to
    ``w_right``.  Singular values below ``tol * sigma_max`` are discarded.

    Args:
        w_left:  Shape (D_l, d, d, D_mid).
        w_right: Shape (D_mid, d, d, D_r).
        tol:     Relative truncation threshold.

    Returns:
        ``(w_left_new, w_right_new)`` with compressed shared bond.
    """
    D_l, d, _, D_mid = w_left.shape
    _, _, _, D_r = w_right.shape

    # Reshape to (D_l*d*d, D_mid) for SVD
    flat = w_left.reshape(D_l * d * d, D_mid)
    U, s, Vt = np.linalg.svd(flat, full_matrices=False)

    threshold = tol * s[0] if s[0] > 0.0 else tol
    rank = max(int(np.sum(s > threshold)), 1)

    # Left factor absorbs singular values
    w_left_new = (U[:, :rank] * s[:rank]).reshape(D_l, d, d, rank)
    # Right factor: new_left_bond = rank
    w_right_new = np.einsum("km,mabn->kabn", Vt[:rank, :], w_right)

    return w_left_new, w_right_new


def _compute_operator_charge(op: np.ndarray, phys_charges: np.ndarray) -> int:
    """Determine the U(1) charge transfer of a single-site operator.

    For a nonzero element ``op[i, j]``, the charge transfer is
    ``phys_charges[i] - phys_charges[j]``.  All nonzero elements must
    agree; otherwise the operator mixes charge sectors.

    Args:
        op:            Square (d, d) operator matrix.
        phys_charges:  Physical-leg charge array of length d.

    Returns:
        Integer charge transfer.

    Raises:
        ValueError: If the operator has nonzero entries with inconsistent
            charge transfers (i.e., it is not a well-defined irrep operator).
    """
    charge: int | None = None
    for i in range(op.shape[0]):
        for j in range(op.shape[1]):
            if abs(op[i, j]) > 1e-14:
                q = int(phys_charges[i]) - int(phys_charges[j])
                if charge is None:
                    charge = q
                elif q != charge:
                    raise ValueError(
                        f"Operator mixes charge sectors: found transfers "
                        f"{charge} and {q}."
                    )
    if charge is None:
        return 0  # zero operator → neutral
    return charge


def _compute_bond_charges(
    terms: list[HamiltonianTerm],
    bond_states: list[dict[int, int]],
    L: int,
    phys_charges: np.ndarray,
) -> list[np.ndarray]:
    """Compute the U(1) charge array for each internal MPO bond.

    At bond ``j`` (between sites ``j`` and ``j + 1``), an in-flight term
    carries the accumulated charge transfer from the operators applied on
    sites ``0 .. j``.  The "done" state (index 0) and "vacuum" state
    (last index) both carry charge 0.

    Args:
        terms:         List of Hamiltonian terms.
        bond_states:   Output of ``_assign_bond_states``.
        L:             Chain length.
        phys_charges:  Physical-leg charge array (length d).

    Returns:
        List of length ``L - 1``, each entry an ``np.ndarray`` of int32
        charges with size ``len(bond_states[j]) + 2``.
    """
    bond_charges: list[np.ndarray] = []
    for j in range(L - 1):
        D = len(bond_states[j]) + 2
        charges = np.zeros(D, dtype=np.int32)
        # charges[0] = 0  (done state)
        # charges[D-1] = 0  (vacuum state)
        for t_id, state_idx in bond_states[j].items():
            term = terms[t_id]
            # Accumulate charge transfer from ops on sites <= j
            acc = 0
            for site, op in term.ops:
                if site <= j:
                    acc += _compute_operator_charge(op, phys_charges)
            charges[state_idx] = acc
        bond_charges.append(charges)
    return bond_charges


def _w_matrices_to_symmetric_mpo(
    w_matrices: list[np.ndarray],
    d: int,
    phys_charges: np.ndarray,
    bond_charges: list[np.ndarray],
    dtype: Any = jnp.float64,
    name: str = "AutoMPO_sym",
) -> TensorNetwork:
    """Wrap W-matrices into a TensorNetwork of SymmetricTensor nodes.

    Same structure as ``_w_matrices_to_mpo`` but uses physical charges
    and per-bond charges to build proper ``SymmetricTensor`` objects.

    Args:
        w_matrices:   One ndarray per site, shape (D_l, d, d, D_r).
        d:            Local Hilbert-space dimension.
        phys_charges: Physical-leg charge array (length d).
        bond_charges: Per-bond charge arrays from ``_compute_bond_charges``.
        dtype:        JAX dtype for on-device tensors.
        name:         TensorNetwork name.

    Returns:
        TensorNetwork MPO with SymmetricTensor nodes.
    """
    L = len(w_matrices)
    sym = U1Symmetry()

    mpo = TensorNetwork(name=name)

    for i, W_np in enumerate(w_matrices):
        D_l, _, _, D_r = W_np.shape
        W = jnp.array(W_np, dtype=dtype)

        # Bond charges: left boundary = [0], right boundary = [0]
        if i == 0:
            left_charges = np.zeros(D_l, dtype=np.int32)
        else:
            left_charges = bond_charges[i - 1]

        if i == L - 1:
            right_charges = np.zeros(D_r, dtype=np.int32)
        else:
            right_charges = bond_charges[i]

        left_label = "w_left_0" if i == 0 else f"w{i - 1}_{i}"
        right_label = "w_right" if i == L - 1 else f"w{i}_{i + 1}"

        indices = (
            TensorIndex(sym, left_charges, FlowDirection.IN, label=left_label),
            TensorIndex(sym, phys_charges, FlowDirection.IN, label=f"mpo_top_{i}"),
            TensorIndex(sym, phys_charges, FlowDirection.OUT, label=f"mpo_bot_{i}"),
            TensorIndex(sym, right_charges, FlowDirection.OUT, label=right_label),
        )
        tensor = SymmetricTensor.from_dense(W, indices)
        mpo.add_node(i, tensor)

    # Connect virtual MPO bonds by shared label
    for i in range(L - 1):
        shared = set(mpo.get_tensor(i).labels()) & set(mpo.get_tensor(i + 1).labels())
        for label in sorted(shared, key=str):
            try:
                mpo.connect(i, label, i + 1, label)
            except (ValueError, KeyError):
                pass

    return mpo


def _w_matrices_to_mpo(
    w_matrices: list[np.ndarray],
    d: int,
    dtype: Any = jnp.float64,
    name: str = "AutoMPO",
) -> TensorNetwork:
    """Wrap a list of W-matrices into a TensorNetwork of DenseTensor nodes.

    Follows the same label/index conventions as ``build_mpo_heisenberg``:
      - Left bond labels:  ``"w_left_0"`` (i=0), ``"w{i-1}_{i}"`` (i>0)
      - Right bond labels: ``"w_right"`` (i=L-1), ``"w{i}_{i+1}"`` (i<L-1)
      - Physical labels:   ``"mpo_top_{i}"``, ``"mpo_bot_{i}"``
      - All MPO bond charges are zero (U1Symmetry, no conservation enforced).

    Args:
        w_matrices: One ndarray per site, shape (D_l, d, d, D_r).
        d:          Local Hilbert-space dimension.
        dtype:      JAX dtype for on-device tensors.
        name:       TensorNetwork name.

    Returns:
        TensorNetwork MPO compatible with ``dmrg()``.
    """
    L = len(w_matrices)
    sym = U1Symmetry()
    bond_d = np.zeros(d, dtype=np.int32)

    mpo = TensorNetwork(name=name)

    for i, W_np in enumerate(w_matrices):
        D_l, _, _, D_r = W_np.shape
        W = jnp.array(W_np, dtype=dtype)

        bond_l = np.zeros(D_l, dtype=np.int32)
        bond_r = np.zeros(D_r, dtype=np.int32)

        left_label = "w_left_0" if i == 0 else f"w{i - 1}_{i}"
        right_label = "w_right" if i == L - 1 else f"w{i}_{i + 1}"

        indices = (
            TensorIndex(sym, bond_l, FlowDirection.IN, label=left_label),
            TensorIndex(sym, bond_d, FlowDirection.IN, label=f"mpo_top_{i}"),
            TensorIndex(sym, bond_d, FlowDirection.OUT, label=f"mpo_bot_{i}"),
            TensorIndex(sym, bond_r, FlowDirection.OUT, label=right_label),
        )
        mpo.add_node(i, DenseTensor(W, indices))

    # Connect virtual MPO bonds by shared label
    for i in range(L - 1):
        shared = set(mpo.get_tensor(i).labels()) & set(mpo.get_tensor(i + 1).labels())
        for label in sorted(shared, key=str):
            try:
                mpo.connect(i, label, i + 1, label)
            except (ValueError, KeyError):
                pass

    return mpo


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class AutoMPO:
    """Symbolic Hamiltonian builder that produces an MPO TensorNetwork.

    Operator names are resolved against a ``site_ops`` dictionary; the
    defaults are ``spin_half_ops()`` for ``d=2`` and ``spin_one_ops()`` for
    ``d=3``.

    Example::

        auto = AutoMPO(L=6)
        for i in range(5):
            auto += (1.0, "Sz", i, "Sz", i + 1)
            auto += (0.5, "Sp", i, "Sm", i + 1)
            auto += (0.5, "Sm", i, "Sp", i + 1)
        mpo = auto.to_mpo()
        result = dmrg(mpo, mps, config)

    The resulting MPO is directly compatible with ``dmrg()``.
    """

    def __init__(
        self,
        L: int,
        d: int = 2,
        site_ops: dict[str, np.ndarray] | None = None,
    ) -> None:
        if L < 1:
            raise ValueError(f"L must be >= 1, got {L}")
        self.L = L
        self.d = d
        if site_ops is not None:
            self._site_ops: dict[str, np.ndarray] = site_ops
        elif d == 2:
            self._site_ops = spin_half_ops()
        elif d == 3:
            self._site_ops = spin_one_ops()
        else:
            raise ValueError(
                f"No default site_ops for d={d}; provide site_ops explicitly."
            )
        self._terms: list[HamiltonianTerm] = []

    # ------------------------------------------------------------------
    # Term addition
    # ------------------------------------------------------------------

    def add_term(self, coeff: complex, *args: Any) -> None:
        """Add one term to the Hamiltonian.

        Args:
            coeff: Scalar coefficient.
            *args: Alternating ``(op_name, site)`` pairs, at least one pair.

        Example::

            auto.add_term(0.5, "Sp", 0, "Sm", 1)
            auto.add_term(1.0, "Sz", 0, "Sz", 1, "Sz", 2)  # 3-body term
        """
        if len(args) == 0 or len(args) % 2 != 0:
            raise ValueError(
                "args must be alternating (op_name, site) pairs; "
                f"got {len(args)} argument(s)."
            )

        pairs: list[tuple[int, np.ndarray]] = []
        for k in range(0, len(args), 2):
            op_name = args[k]
            site = args[k + 1]
            if not isinstance(op_name, str):
                raise TypeError(
                    f"Expected operator name (str) at position {k}, got {type(op_name).__name__}."
                )
            if op_name not in self._site_ops:
                raise KeyError(
                    f"Operator '{op_name}' not in site_ops. "
                    f"Available: {sorted(self._site_ops)}"
                )
            if not isinstance(site, int) or not (0 <= site < self.L):
                raise ValueError(f"Site {site!r} out of range [0, {self.L}).")
            pairs.append((site, self._site_ops[op_name]))

        pairs.sort(key=lambda x: x[0])
        sites = [s for s, _ in pairs]
        if len(sites) != len(set(sites)):
            raise ValueError(f"Duplicate sites in term: {sites}")

        self._terms.append(
            HamiltonianTerm(coefficient=complex(coeff), ops=tuple(pairs))
        )

    def __iadd__(self, args: tuple) -> AutoMPO:
        """Convenience operator: ``auto_mpo += (coeff, op1, site1, ...)``.

        The first element of the tuple is the coefficient; the remaining
        elements are alternating (op_name, site) pairs.
        """
        self.add_term(args[0], *args[1:])
        return self

    # ------------------------------------------------------------------
    # MPO construction
    # ------------------------------------------------------------------

    def to_mpo(
        self,
        compress: bool = False,
        compress_tol: float = 1e-12,
        dtype: Any = jnp.float64,
        symmetric: bool = False,
        phys_charges: np.ndarray | None = None,
    ) -> TensorNetwork:
        """Build and return the MPO as a TensorNetwork.

        Args:
            compress:      Apply a left-to-right SVD compression pass to
                           reduce bond dimension (approximate; not globally
                           optimal but useful for long-range interactions).
            compress_tol:  Relative singular-value threshold for compression.
            dtype:         JAX dtype for on-device MPO tensors.
            symmetric:     If True, build a ``SymmetricTensor``-based MPO
                           with proper U(1) charge assignments on every leg.
            phys_charges:  Physical-leg charge array (length d).  If *None*,
                           defaults to ``[1, -1]`` for d=2 and
                           ``[2, 0, -2]`` for d=3.

        Returns:
            TensorNetwork with L site tensors in the standard MPO format,
            compatible with ``dmrg()``.

        Raises:
            ValueError: If no terms have been added.
        """
        if not self._terms:
            raise ValueError("No terms; call add_term() before to_mpo().")

        identity = np.eye(self.d, dtype=np.float64)
        bond_states = _assign_bond_states(self._terms, self.L)
        w_mats = _build_w_matrices(self._terms, bond_states, self.L, self.d, identity)

        if compress and self.L > 1:
            for j in range(self.L - 1):
                w_mats[j], w_mats[j + 1] = _compress_mpo_bond(
                    w_mats[j], w_mats[j + 1], tol=compress_tol
                )

        if symmetric:
            if phys_charges is None:
                if self.d == 2:
                    phys_charges = np.array([1, -1], dtype=np.int32)
                elif self.d == 3:
                    phys_charges = np.array([2, 0, -2], dtype=np.int32)
                else:
                    raise ValueError(
                        f"No default phys_charges for d={self.d}; "
                        "provide phys_charges explicitly."
                    )
            else:
                phys_charges = np.asarray(phys_charges, dtype=np.int32)
            bond_charges = _compute_bond_charges(
                self._terms, bond_states, self.L, phys_charges
            )
            return _w_matrices_to_symmetric_mpo(
                w_mats,
                self.d,
                phys_charges,
                bond_charges,
                dtype=dtype,
            )

        return _w_matrices_to_mpo(w_mats, self.d, dtype=dtype)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def bond_dims(self) -> list[int]:
        """Return the (uncompressed) MPO bond dimensions at each internal bond."""
        bond_states = _assign_bond_states(self._terms, self.L)
        return [len(bs) + 2 for bs in bond_states]

    def n_terms(self) -> int:
        """Number of Hamiltonian terms added so far."""
        return len(self._terms)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def build_auto_mpo(
    terms_spec: list[tuple],
    L: int,
    d: int = 2,
    site_ops: dict[str, np.ndarray] | None = None,
    compress: bool = False,
    compress_tol: float = 1e-12,
    dtype: Any = jnp.float64,
    symmetric: bool = False,
    phys_charges: np.ndarray | None = None,
) -> TensorNetwork:
    """Build an MPO from a list of term specifications.

    Args:
        terms_spec:    List of tuples ``(coeff, op1, site1, op2, site2, ...)``.
        L:             Chain length (number of sites).
        d:             Local Hilbert-space dimension (2 for spin-1/2).
        site_ops:      Operator name → matrix dict; defaults to spin_half_ops()
                       for d=2 and spin_one_ops() for d=3.
        compress:      Apply left-to-right SVD compression.
        compress_tol:  Relative singular-value threshold for compression.
        dtype:         JAX dtype for MPO tensors.
        symmetric:     If True, build a ``SymmetricTensor``-based MPO.
        phys_charges:  Physical-leg charge array (length d).

    Returns:
        TensorNetwork MPO compatible with ``dmrg()``.

    Example::

        mpo = build_auto_mpo(
            [(1.0, "Sz", i, "Sz", i + 1) for i in range(L - 1)]
            + [(0.5, "Sp", i, "Sm", i + 1) for i in range(L - 1)]
            + [(0.5, "Sm", i, "Sp", i + 1) for i in range(L - 1)],
            L=L,
        )
    """
    auto = AutoMPO(L=L, d=d, site_ops=site_ops)
    for term in terms_spec:
        auto.add_term(term[0], *term[1:])
    return auto.to_mpo(
        compress=compress,
        compress_tol=compress_tol,
        dtype=dtype,
        symmetric=symmetric,
        phys_charges=phys_charges,
    )
