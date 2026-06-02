"""iPEPS simple update functions for imaginary time evolution.

Contains 2-site Tensor-protocol simple update routines and the Trotter gate
builder, extracted from the monolithic ``ipeps.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import safe_inv_lambda, scale_bond_axis
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor

# ------------------------------------------------------------------ #
# 2-site Tensor-protocol simple update                                #
# ------------------------------------------------------------------ #


def _simple_update_2site_horizontal_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the horizontal bond (A.r <-> B.l).

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     Left site tensor with labels (u, d, l, r, phys).
        B:     Right site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_h_new).
    """
    # 1. Absorb outer lambdas onto A: u<-lam_v, d<-lam_v, l<-lam_h
    #    + shared lambda on A.r<-lam_h
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)

    # 2. Absorb outer lambdas onto B: u<-lam_v, d<-lam_v, r<-lam_h (NOT l)
    B_abs = scale_bond_axis(B, "u", lam_v)
    B_abs = scale_bond_axis(B_abs, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.r with B.l
    A_left = A_abs.relabel("r", "shared")
    B_right = B_abs.relabels(
        {
            "u": "u_B",
            "d": "d_B",
            "l": "shared",
            "r": "r_B",
            "phys": "phys_B",
        }
    )
    theta = contract(A_left, B_right)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. Truncated SVD — preserve the canonical layout of the old horizontal
    #    bond (A.r) in the new bond. For fermionic SymmetricTensor inputs this
    #    keeps the per-parity-sector keep COUNTS canonical at D>2 (#558/#559
    #    in the single-site path; same root cause here per #563). For
    #    DenseTensor (trivial-charge) inputs base_charges is a no-op.
    base_charges = np.asarray(A.indices[A.labels().index("r")].charges)
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "d", "l", "si_out"],
        right_labels=["u_B", "d_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
        base_charges=base_charges,
    )

    # 6. New lambda (normalized)
    lam_h_new = sigma / (jnp.max(sigma) + EPS)
    sqrt_sig = jnp.sqrt(sigma + EPS)

    # 7. Reconstruct A_new from U: labels are (u, d, l, si_out, bond_new)
    #    Transpose so bond_new is in the r position: (u, d, l, bond_new, si_out)
    A_new = U.transpose((0, 1, 2, 4, 3))
    A_new = A_new.relabels({"bond_new": "r", "si_out": "phys"})
    A_new = scale_bond_axis(A_new, "r", sqrt_sig)

    # 8. Reconstruct B_new from Vh: labels are (bond_new, u_B, d_B, r_B, sj_out)
    #    Transpose so bond_new is in the l position: (u_B, d_B, bond_new, r_B, sj_out)
    B_new = Vh.transpose((1, 2, 0, 3, 4))
    B_new = B_new.relabels(
        {"bond_new": "l", "u_B": "u", "d_B": "d", "r_B": "r", "sj_out": "phys"}
    )
    B_new = scale_bond_axis(B_new, "l", sqrt_sig)

    # 9. Remove outer lambdas (pseudo-inverse: drop dead sectors, no 1/eps blow-up)
    inv_lam_v = safe_inv_lambda(lam_v)
    inv_lam_h = safe_inv_lambda(lam_h)
    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "d", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)

    B_new = scale_bond_axis(B_new, "u", inv_lam_v)
    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize each tensor
    norm_A = float(A_new.norm())
    if norm_A > EPS:
        A_new = A_new * (1.0 / norm_A)
    norm_B = float(B_new.norm())
    if norm_B > EPS:
        B_new = B_new * (1.0 / norm_B)

    return A_new, B_new, lam_h_new


def _simple_update_2site_vertical_tensor(
    A: Tensor,
    B: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
    """2-site simple update on the vertical bond (A.d <-> B.u).

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     Top site tensor with labels (u, d, l, r, phys).
        B:     Bottom site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, B_new, lam_v_new).
    """
    # 1. Absorb outer lambdas onto A: u<-lam_v, l<-lam_h, r<-lam_h
    #    + shared lambda on A.d<-lam_v
    A_abs = scale_bond_axis(A, "u", lam_v)
    A_abs = scale_bond_axis(A_abs, "d", lam_v)
    A_abs = scale_bond_axis(A_abs, "l", lam_h)
    A_abs = scale_bond_axis(A_abs, "r", lam_h)

    # 2. Absorb outer lambdas onto B: d<-lam_v, l<-lam_h, r<-lam_h (NOT u)
    B_abs = scale_bond_axis(B, "d", lam_v)
    B_abs = scale_bond_axis(B_abs, "l", lam_h)
    B_abs = scale_bond_axis(B_abs, "r", lam_h)

    # 3. Contract A.d with B.u
    A_top = A_abs.relabel("d", "shared")
    B_bottom = B_abs.relabels(
        {
            "u": "shared",
            "d": "d_B",
            "l": "l_B",
            "r": "r_B",
            "phys": "phys_B",
        }
    )
    theta = contract(A_top, B_bottom)

    # 4. Apply gate
    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    # 5. Truncated SVD — preserve the canonical layout of the old vertical
    #    bond (A.d) in the new bond. See horizontal counterpart above for the
    #    rationale (#563).
    base_charges = np.asarray(A.indices[A.labels().index("d")].charges)
    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "l", "r", "si_out"],
        right_labels=["d_B", "l_B", "r_B", "sj_out"],
        new_bond_label="bond_new",
        max_singular_values=max_D,
        base_charges=base_charges,
    )

    # 6. New lambda (normalized)
    lam_v_new = sigma / (jnp.max(sigma) + EPS)
    sqrt_sig = jnp.sqrt(sigma + EPS)

    # 7. Reconstruct A_new from U: labels are (u, l, r, si_out, bond_new)
    #    Transpose so bond_new is in the d position: (u, bond_new, l, r, si_out)
    A_new = U.transpose((0, 4, 1, 2, 3))
    A_new = A_new.relabels({"bond_new": "d", "si_out": "phys"})
    A_new = scale_bond_axis(A_new, "d", sqrt_sig)

    # 8. Reconstruct B_new from Vh: labels are (bond_new, d_B, l_B, r_B, sj_out)
    #    Transpose so bond_new is in the u position: (bond_new, d_B, l_B, r_B, sj_out)
    #    Already in the right order for (u, d, l, r, phys)
    B_new = Vh.relabels(
        {"bond_new": "u", "d_B": "d", "l_B": "l", "r_B": "r", "sj_out": "phys"}
    )
    B_new = scale_bond_axis(B_new, "u", sqrt_sig)

    # 9. Remove outer lambdas (pseudo-inverse: drop dead sectors, no 1/eps blow-up)
    inv_lam_v = safe_inv_lambda(lam_v)
    inv_lam_h = safe_inv_lambda(lam_h)
    A_new = scale_bond_axis(A_new, "u", inv_lam_v)
    A_new = scale_bond_axis(A_new, "l", inv_lam_h)
    A_new = scale_bond_axis(A_new, "r", inv_lam_h)

    B_new = scale_bond_axis(B_new, "d", inv_lam_v)
    B_new = scale_bond_axis(B_new, "l", inv_lam_h)
    B_new = scale_bond_axis(B_new, "r", inv_lam_h)

    # 10. Normalize each tensor
    norm_A = float(A_new.norm())
    if norm_A > EPS:
        A_new = A_new * (1.0 / norm_A)
    norm_B = float(B_new.norm())
    if norm_B > EPS:
        B_new = B_new * (1.0 / norm_B)

    return A_new, B_new, lam_v_new


def _make_trotter_gate_tensor(
    hamiltonian_gate: jax.Array | Tensor,
    dt: float,
    site_tensor: Tensor | None = None,
) -> Tensor:
    """Build the Trotter gate exp(-dt * H) as a 4-leg Tensor.

    When *site_tensor* is provided, the gate indices use the same symmetry
    and physical charges as the site tensor's physical leg.  If the site
    tensor is a ``SymmetricTensor``, the gate is built via ``from_dense``
    so that it matches the type and can be contracted directly.

    Args:
        hamiltonian_gate: 2-site Hamiltonian as (d,d,d,d) array or Tensor.
        dt: Imaginary time step.
        site_tensor: Optional reference tensor whose physical index provides
                     symmetry and charge information for the gate.

    Returns:
        Tensor with labels (si, sj, si_out, sj_out).
    """
    if isinstance(hamiltonian_gate, Tensor):
        H_dense = hamiltonian_gate.todense()
    else:
        H_dense = jnp.asarray(hamiltonian_gate)

    d = H_dense.shape[0]
    H_mat = H_dense.reshape(d * d, d * d)
    H_mat = 0.5 * (H_mat + H_mat.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(H_mat)
    gate_mat = eigvecs @ jnp.diag(jnp.exp(-dt * eigvals)) @ eigvecs.conj().T
    gate_4leg = gate_mat.reshape(d, d, d, d)

    # Derive index metadata from site tensor's physical leg if available
    if site_tensor is not None:
        phys_idx = site_tensor.indices[-1]  # last leg = phys
        sym = phys_idx.symmetry
        charges = phys_idx.charges
    else:
        sym = U1Symmetry()
        charges = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )

    if isinstance(site_tensor, SymmetricTensor):
        return SymmetricTensor.from_dense(gate_4leg, indices)

    return DenseTensor(gate_4leg, indices)
