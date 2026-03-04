"""Observable computation from MPS ground states.

Provides functions to compute local expectation values, two-point correlation
functions, and detect operator charge quantum numbers from MPS wavefunctions
produced by DMRG.

All functions work polymorphically on both DenseTensor and SymmetricTensor MPS.
For block-sparse MPS, charge compatibility is enforced naturally: contracting a
charge-changing operator with a pure-sector MPS sandwich finds no matching
blocks and returns 0.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.core.tensor import Tensor
from tenax.network.network import TensorNetwork


def _mps_tensors_from(mps: TensorNetwork | list[Tensor]) -> list[Tensor]:
    """Extract ordered list of MPS site tensors."""
    if isinstance(mps, TensorNetwork):
        L = mps.n_nodes()
        return [mps.get_tensor(i) for i in range(L)]
    return list(mps)


def expectation_value(
    mps: TensorNetwork | list[Tensor],
    operator: np.ndarray,
    site: int,
) -> float:
    """Compute <psi|O_i|psi> for a local operator at a given site.

    Contracts the MPS transfer matrices from left to right, inserting the
    operator at the specified site.

    Args:
        mps:      MPS as TensorNetwork or list of site tensors.
        operator: d×d numpy array representing the local operator.
        site:     Site index where the operator acts.

    Returns:
        Real part of the expectation value (float).
    """
    tensors = _mps_tensors_from(mps)
    L = len(tensors)

    if site < 0 or site >= L:
        raise ValueError(f"site={site} out of range for L={L} MPS")

    # Normalize the MPS: compute <psi|psi> and <psi|O|psi> simultaneously
    # by doing a single left-to-right sweep with identity everywhere except site

    # We need to build the full contraction. For simplicity, use the approach of:
    # 1. Contract all sites left-to-right with identity transfer matrices
    # 2. At the operator site, insert the operator
    # Strategy: accumulate transfer matrix from left

    norm_sq = _contract_sandwich(tensors, op=None, op_site=None)
    ov = _contract_sandwich(tensors, op=operator, op_site=site)

    result = ov / norm_sq
    return float(jnp.real(result))


def correlation(
    mps: TensorNetwork | list[Tensor],
    op_i: np.ndarray,
    site_i: int,
    op_j: np.ndarray,
    site_j: int,
) -> float:
    """Compute <psi|O_i O_j|psi> for two-point correlation.

    Args:
        mps:    MPS as TensorNetwork or list of site tensors.
        op_i:   d×d operator at site_i.
        site_i: First site index.
        op_j:   d×d operator at site_j.
        site_j: Second site index.

    Returns:
        Real part of the correlation function (float).
    """
    tensors = _mps_tensors_from(mps)
    L = len(tensors)

    if site_i > site_j:
        site_i, site_j = site_j, site_i
        op_i, op_j = op_j, op_i

    if site_i < 0 or site_j >= L:
        raise ValueError(f"Sites ({site_i}, {site_j}) out of range for L={L}")

    norm_sq = _contract_sandwich(tensors, op=None, op_site=None)
    if site_i == site_j:
        # Same-site: compose operators O_i @ O_j
        combined = np.asarray(op_i) @ np.asarray(op_j)
        ov = _contract_sandwich(tensors, op=combined, op_site=site_i)
    else:
        ov = _contract_sandwich(
            tensors, op=op_i, op_site=site_i, op2=op_j, op_site2=site_j
        )

    result = ov / norm_sq
    return float(jnp.real(result))


def _contract_sandwich(
    tensors: list[Tensor],
    op: np.ndarray | None,
    op_site: int | None,
    op2: np.ndarray | None = None,
    op_site2: int | None = None,
) -> jnp.ndarray:
    """Contract <psi|O1(site1) O2(site2)|psi> via transfer matrices.

    If op is None and op_site is None, computes <psi|psi>.
    Works by converting everything to dense arrays and using einsum
    for reliability across both DenseTensor and SymmetricTensor MPS.

    Einsum index convention:
        a = chi_ket (left bond of transfer matrix / ket MPS)
        b = chi_bra (left bond of transfer matrix / bra MPS)
        p = physical ket
        q = physical bra (or operator output)
        r = chi_ket right
        s = chi_bra right
    """
    L = len(tensors)

    # Get dense arrays for all sites
    dense_sites = []
    for t in tensors:
        arr = t.todense() if hasattr(t, "todense") else t
        dense_sites.append(arr)

    # Build operators at specific sites
    ops_at_site: dict[int, np.ndarray] = {}
    if op is not None and op_site is not None:
        ops_at_site[op_site] = jnp.array(op)
    if op2 is not None and op_site2 is not None:
        ops_at_site[op_site2] = jnp.array(op2)

    # Left-to-right transfer matrix contraction
    # Transfer matrix T has shape (chi_ket, chi_bra) for middle sites
    tm = None  # will be (chi_ket, chi_bra) or scalar

    for i in range(L):
        A = dense_sites[i]
        A_conj = jnp.conj(A)

        if i in ops_at_site:
            op = ops_at_site[i]
            if A.ndim == 2:
                labels = tensors[i].labels()
                is_left = isinstance(labels[0], str) and labels[0].startswith("p")
                if is_left:
                    # A: (p, r), op: (p, q) → sum_pq op[p,q] A[p,r] A*[q,s] → (r, s)
                    contracted = jnp.einsum("pq,pr,qs->rs", op, A, A_conj)
                else:
                    # A: (a, p), right boundary
                    if tm is None:
                        contracted = jnp.einsum("pq,ap,aq->", op, A, A_conj)
                    else:
                        contracted = jnp.einsum("ab,pq,ap,bq->", tm, op, A, A_conj)
            else:
                # A: (a, p, r), middle site
                if tm is None:
                    contracted = jnp.einsum("pq,apr,aqs->rs", op, A, A_conj)
                else:
                    contracted = jnp.einsum("ab,pq,apr,bqs->rs", tm, op, A, A_conj)
        else:
            # Identity: contract physical indices directly
            if A.ndim == 2:
                labels = tensors[i].labels()
                is_left = isinstance(labels[0], str) and labels[0].startswith("p")
                if is_left:
                    # A: (p, r) → sum_p A[p,r] A*[p,s] → (r, s)
                    contracted = jnp.einsum("pr,ps->rs", A, A_conj)
                else:
                    # A: (a, p), right boundary
                    if tm is None:
                        contracted = jnp.einsum("ap,ap->", A, A_conj)
                    else:
                        contracted = jnp.einsum("ab,ap,bp->", tm, A, A_conj)
            else:
                # A: (a, p, r), middle site
                if tm is None:
                    contracted = jnp.einsum("apr,aps->rs", A, A_conj)
                else:
                    contracted = jnp.einsum("ab,apr,bps->rs", tm, A, A_conj)

        tm = contracted

    # tm should be a scalar at the end
    return tm


def operator_charge(op: np.ndarray, phys_charges: np.ndarray | None = None) -> int:
    """Detect the U(1) charge of a local operator.

    For each nonzero element op[i,j], the charge transferred is
    phys_charges[i] - phys_charges[j] (row charge minus column charge).
    All nonzero elements must carry the same charge.

    Args:
        op:            d×d numpy array representing the operator.
        phys_charges:  Charge array for the physical index. Defaults to
                       spin-1/2 convention: [+1, -1].

    Returns:
        Integer charge of the operator.

    Raises:
        ValueError: If the operator has mixed charges (not a well-defined
                    irreducible tensor operator).
    """
    if phys_charges is None:
        phys_charges = np.array([1, -1], dtype=np.int32)

    op_arr = np.asarray(op)
    charges: set[int] = set()
    for i in range(op_arr.shape[0]):
        for j in range(op_arr.shape[1]):
            if abs(op_arr[i, j]) > 1e-15:
                charges.add(int(phys_charges[i] - phys_charges[j]))

    if len(charges) == 0:
        return 0
    if len(charges) == 1:
        return charges.pop()
    raise ValueError(
        f"Operator has mixed charges {charges}; not a well-defined "
        f"irreducible tensor operator."
    )
