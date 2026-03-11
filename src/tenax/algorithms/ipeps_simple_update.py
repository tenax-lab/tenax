"""iPEPS simple update functions for imaginary time evolution.

Contains dense-array, Tensor-protocol, and 2-site simple update routines
extracted from the monolithic ``ipeps.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.contraction.contractor import contract, truncated_svd
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def _simple_update_1x1(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
    *,
    bond: str = "horizontal",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update step for a 1x1 unit cell PEPS.

    Applies the gate on the bond between A and B (treating B = A by translational
    invariance). Updates lambda matrices for environment approximation.

    Args:
        A:            Left/top site tensor of shape (D, D, D, D, d) or (D, D, d).
                      Convention: A[u, d, l, r, s] for full 5-leg PEPS.
        B:            Right/bottom site tensor (same as A for 1x1 unit cell).
        lambdas:      Dict of lambda vectors for each bond direction.
        gate:         2-site gate of shape (d, d, d, d).
        max_bond_dim: Maximum D after truncation.
        bond:         Which bond to update: ``"horizontal"`` (A.r ↔ B.l) or
                      ``"vertical"`` (A.d ↔ B.u).

    Returns:
        (A_new, lambdas_new)
    """
    d = gate.shape[0]

    if A.ndim == 3:
        return _simple_update_3leg(A, B, lambdas, gate, max_bond_dim, d)

    # --- Full 5-leg tensors: A[u, d, l, r, s] ---
    D_u, D_d, D_l, D_r, phys = A.shape
    lam_h = lambdas.get("horizontal", jnp.ones(D_r))
    lam_v = lambdas.get("vertical", jnp.ones(D_d))

    if bond == "horizontal":
        return _simple_update_horizontal(A, lam_h, lam_v, gate, max_bond_dim, lambdas)
    else:
        return _simple_update_vertical(A, lam_h, lam_v, gate, max_bond_dim, lambdas)


def _simple_update_3leg(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
    d: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update for the legacy 3-leg (D_l, D_r, d) tensor path."""
    D_l, D_r, phys = A.shape
    lam_r = lambdas.get(
        "horizontal", lambdas.get("right", jnp.ones(min(D_r, max_bond_dim)))
    )

    A_abs = A * lam_r[None, : min(D_r, len(lam_r)), None]
    theta = jnp.einsum("lrs,Lrs,sstT->lLtT", A_abs, A, gate.reshape(phys, phys, d, d))

    theta_mat = theta.reshape(D_l * D_l, d * d)
    U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

    n_keep = min(max_bond_dim, len(s))
    U = U[:, :n_keep]
    s_new = s[:n_keep]

    s_norm = s_new / (jnp.max(s_new) + 1e-15)
    lam_inv = 1.0 / (lam_r[: min(D_r, len(lam_r))] + 1e-15)

    A_new_mat = U.reshape(D_l, D_l, n_keep)[:, 0, :]
    A_new = (A_new_mat * lam_inv[None, : min(D_l, len(lam_inv))]).reshape(
        D_l, n_keep, d
    )

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = s_norm
    lambdas_new.pop("right", None)
    return A_new, lambdas_new


def _simple_update_bond(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond, parameterized by axis.

    A[u, d, l, r, s]:
      axis="horizontal": shared bond = r (horizontal lambda)
      axis="vertical":   shared bond = d (vertical lambda)

    Args:
        A:             iPEPS site tensor, shape (D_u, D_d, D_l, D_r, d).
        lam_h:         Horizontal bond lambdas.
        lam_v:         Vertical bond lambdas.
        gate:          Trotter gate, shape (d, d, d, d).
        max_bond_dim:  Maximum bond dimension after SVD.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = EPS

    if axis == "horizontal":
        # Absorb outer lambdas onto A (all except shared bond r)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        # Absorb shared-bond lambda onto A.r
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B = A, absorb outer lambdas (all except B.l = shared)
        B_abs = A * lam_v[:D_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.r with B_abs.l → theta
        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = D_u * D_d * D_r * d
        left_shape = (D_u, D_d, D_l, d)
        # new bond goes into r slot: transpose to (D_u, D_d, D_l, keep, d)
        left_perm = (0, 1, 2, 4, 3)

        # Outer lambda removal: u←lam_v, d←lam_v, l←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_v + eps), 1, D_d),  # axis 1
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
        ]
    else:  # vertical
        # Absorb outer lambdas onto A (all except shared bond d)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        # Absorb shared-bond lambda onto A.d
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B = A, absorb outer lambdas (all except B.u = shared)
        B_abs = A * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :D_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.d with B_abs.u → theta
        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = D_d * D_l * D_r * d
        left_shape = (D_u, D_l, D_r, d)
        # new bond goes into d slot: transpose to (D_u, keep, D_l, D_r, d)
        left_perm = (0, 4, 1, 2, 3)

        # Outer lambda removal: u←lam_v, l←lam_h, r←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
            (1.0 / (lam_h + eps), 3, D_r),  # axis 3 (after transpose)
        ]

    # SVD split
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]

    # New lambda (normalized)
    lam_new = sigma / (jnp.max(sigma) + eps)

    # Reconstruct A_new from U_mat
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*left_shape, keep)
    A_new = A_left.transpose(left_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in outer_inv_slices:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)

    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, lambdas_new


def _simple_update_horizontal(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond (A.r ↔ B.l, B=A by periodicity)."""
    return _simple_update_bond(
        A, lam_h, lam_v, gate, max_bond_dim, lambdas, "horizontal"
    )


def _simple_update_vertical(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond (A.d ↔ B.u, B=A by periodicity)."""
    return _simple_update_bond(A, lam_h, lam_v, gate, max_bond_dim, lambdas, "vertical")


# ------------------------------------------------------------------ #
# Tensor-protocol simple update (polymorphic DenseTensor/SymmetricTensor) #
# ------------------------------------------------------------------ #


def _absorb_lambdas_tensor(A: Tensor, lam_h: jax.Array, lam_v: jax.Array) -> Tensor:
    """Absorb lambda vectors into all virtual legs of a 5-leg site tensor.

    Args:
        A:     Site tensor with labels (u, d, l, r, phys).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.

    Returns:
        Tensor with lambdas absorbed on u(lam_v), d(lam_v), l(lam_h), r(lam_h).
    """
    result = scale_bond_axis(A, "u", lam_v)
    result = scale_bond_axis(result, "d", lam_v)
    result = scale_bond_axis(result, "l", lam_h)
    result = scale_bond_axis(result, "r", lam_h)
    return result


def _simple_update_horizontal_tensor(
    A: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, jax.Array]:
    """Simple update on the horizontal bond using label-based contraction.

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     iPEPS site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, lam_h_new) where A_new has labels (u, d, l, r, phys).
    """
    A_abs = _absorb_lambdas_tensor(A, lam_h, lam_v)

    A_left = A_abs.relabel("r", "shared")
    B_right = A_abs.relabels(
        {
            "u": "u_B",
            "d": "d_B",
            "l": "shared",
            "r": "r_B",
            "phys": "phys_B",
        }
    )

    theta = contract(A_left, B_right)

    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "d", "l", "si_out"],
        right_labels=["u_B", "d_B", "r_B", "sj_out"],
        new_bond_label="r_new",
        max_singular_values=max_D,
    )

    lam_h_new = sigma / (jnp.max(sigma) + EPS)

    U_reordered = U.transpose((0, 1, 2, 4, 3))
    U_final = U_reordered.relabels({"r_new": "r", "si_out": "phys"})

    sqrt_sig = jnp.sqrt(sigma + EPS)
    U_final = scale_bond_axis(U_final, "r", sqrt_sig)

    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)
    U_final = scale_bond_axis(U_final, "u", inv_lam_v)
    U_final = scale_bond_axis(U_final, "d", inv_lam_v)
    U_final = scale_bond_axis(U_final, "l", inv_lam_h)

    norm_val = float(U_final.norm())
    if norm_val > EPS:
        U_final = U_final * (1.0 / norm_val)

    return U_final, lam_h_new


def _simple_update_vertical_tensor(
    A: Tensor,
    gate: Tensor,
    lam_h: jax.Array,
    lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, jax.Array]:
    """Simple update on the vertical bond using label-based contraction.

    Works polymorphically with DenseTensor and SymmetricTensor.

    Args:
        A:     iPEPS site tensor with labels (u, d, l, r, phys).
        gate:  Trotter gate with labels (si, sj, si_out, sj_out).
        lam_h: Horizontal bond lambda vector.
        lam_v: Vertical bond lambda vector.
        max_D: Maximum bond dimension after SVD.

    Returns:
        (A_new, lam_v_new) where A_new has labels (u, d, l, r, phys).
    """
    A_abs = _absorb_lambdas_tensor(A, lam_h, lam_v)

    A_top = A_abs.relabel("d", "shared")
    B_bottom = A_abs.relabels(
        {
            "u": "shared",
            "d": "d_B",
            "l": "l_B",
            "r": "r_B",
            "phys": "phys_B",
        }
    )

    theta = contract(A_top, B_bottom)

    theta = theta.relabel("phys", "si")
    theta = theta.relabel("phys_B", "sj")
    theta = contract(theta, gate)

    U, sigma, Vh, s_full = truncated_svd(
        theta,
        left_labels=["u", "l", "r", "si_out"],
        right_labels=["d_B", "l_B", "r_B", "sj_out"],
        new_bond_label="d_new",
        max_singular_values=max_D,
    )

    lam_v_new = sigma / (jnp.max(sigma) + EPS)

    U_reordered = U.transpose((0, 4, 1, 2, 3))
    U_final = U_reordered.relabels({"d_new": "d", "si_out": "phys"})

    sqrt_sig = jnp.sqrt(sigma + EPS)
    U_final = scale_bond_axis(U_final, "d", sqrt_sig)

    inv_lam_v = 1.0 / (lam_v + EPS)
    inv_lam_h = 1.0 / (lam_h + EPS)
    U_final = scale_bond_axis(U_final, "u", inv_lam_v)
    U_final = scale_bond_axis(U_final, "l", inv_lam_h)
    U_final = scale_bond_axis(U_final, "r", inv_lam_h)

    norm_val = float(U_final.norm())
    if norm_val > EPS:
        U_final = U_final * (1.0 / norm_val)

    return U_final, lam_v_new


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
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="si_out"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="sj_out"),
    )

    if isinstance(site_tensor, SymmetricTensor):
        return SymmetricTensor.from_dense(gate_4leg, indices)

    return DenseTensor(gate_4leg, indices)


def _simple_update_2site_bond(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond for a 2-site unit cell.

    Args:
        A, B:          iPEPS site tensors, shape (D_u, D_d, D_l, D_r, d).
        lam_h, lam_v:  Bond lambdas.
        gate:          Trotter gate.
        max_bond_dim:  Maximum bond dimension.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    B_u, B_d, B_l, B_r, _ = B.shape
    eps = EPS

    if axis == "horizontal":
        # A: outer = u(v), d(v), l(h); shared = r(h)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B: outer = u(v), d(v), r(h); shared = l (from contraction)
        B_abs = B * lam_v[:B_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = B_u * B_d * B_r * d
        a_left_shape = (D_u, D_d, D_l, d)
        b_right_shape = (B_u, B_d, B_r, d)
        a_perm = (0, 1, 2, 4, 3)  # new bond → r slot
        b_perm = (1, 2, 0, 3, 4)  # new bond → l slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_v + eps), 1, D_d),
            (1.0 / (lam_h + eps), 2, D_l),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 0, B_u),
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 3, B_r),
        ]
    else:  # vertical
        # A: outer = u(v), l(h), r(h); shared = d(v)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B: outer = d(v), l(h), r(h); shared = u (from contraction)
        B_abs = B * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :B_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = B_d * B_l * B_r * d
        a_left_shape = (D_u, D_l, D_r, d)
        b_right_shape = (B_d, B_l, B_r, d)
        a_perm = (0, 4, 1, 2, 3)  # new bond → d slot
        b_perm = (0, 1, 2, 3, 4)  # new bond → u slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_h + eps), 2, D_l),
            (1.0 / (lam_h + eps), 3, D_r),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 2, B_l),
            (1.0 / (lam_h + eps), 3, B_r),
        ]

    # SVD
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    lam_new = sigma / (jnp.max(sigma) + eps)
    sqrt_sig = jnp.sqrt(sigma + eps)

    # Reconstruct A_new
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*a_left_shape, keep)
    A_new = A_left.transpose(a_perm)

    # Reconstruct B_new
    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, *b_right_shape)
    B_new = B_right.transpose(b_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in a_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    for inv_lam, ax, dim in b_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        B_new = B_new * inv_lam[:dim].reshape(shape)
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_horizontal(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond A.r ↔ B.l for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, d←lam_v, l←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    # 2. Absorb shared-bond lambda onto A.r
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]

    # 3. Absorb outer lambdas onto B: u←lam_v, d←lam_v, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[:B_u, None, None, None, None]
    B_abs = B_abs * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.r with B_abs.l
    theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

    # 6. SVD: group (u,d,l,S) vs (U,D,R,T)
    left_size = D_u * D_d * D_l * d
    right_size = B_u * B_d * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new with sqrt(sigma) absorbed
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_d, D_l, d, keep)
    A_new = A_left.transpose(0, 1, 2, 4, 3)  # (D_u, D_d, D_l, keep, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_u, B_d, B_r, d)
    B_new = B_right.transpose(1, 2, 0, 3, 4)  # (B_u, B_d, keep, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_v_inv[None, :D_d, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[:B_u, None, None, None, None]
    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_vertical(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond A.d ↔ B.u for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, l←lam_h, r←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]
    # 2. Absorb shared-bond lambda onto A.d
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]

    # 3. Absorb outer lambdas onto B: d←lam_v, l←lam_h, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, :B_l, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.d with B_abs.u
    theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

    # 6. SVD: group (u,l,r,S) vs (D,L,R,T)
    left_size = D_u * D_l * D_r * d
    right_size = B_d * B_l * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_l, D_r, d, keep)
    A_new = A_left.transpose(0, 4, 1, 2, 3)  # (D_u, keep, D_l, D_r, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_d, B_l, B_r, d)
    B_new = B_right.transpose(0, 1, 2, 3, 4)  # (keep, B_d, B_l, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new * lam_h_inv[None, None, None, :D_r, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, :B_l, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["vertical"] = lam_new
    return A_new, B_new, lambdas_new
