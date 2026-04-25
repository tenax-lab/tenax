"""iPESS (infinite Projected Entangled Simplex State) on the kagome lattice."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from tenax.algorithms.auto_mpo import spin_half_ops, spin_one_ops

D_PHYS_DEFAULT = 3  # spin-1


def _site_ops(d: int) -> dict[str, np.ndarray]:
    """Return ``{"Sz", "Sp", "Sm", "Id"}`` for physical dimension ``d``."""
    if d == 2:
        return spin_half_ops()
    if d == 3:
        return spin_one_ops()
    raise ValueError(f"Unsupported physical dimension d={d}; expected 2 or 3.")


def kagome_triangle_xxz_hamiltonian(
    delta: float, d: int = D_PHYS_DEFAULT
) -> np.ndarray:
    """Build the 3-site XXZ Hamiltonian on a kagome triangle.

    H_tri = H_12 + H_23 + H_31 with each pair term

        H_ij = delta * Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j).

    Args:
        delta: Ising anisotropy on the SzSz channel.
        d: Physical dimension per site (2 for spin-1/2, 3 for spin-1).

    Returns:
        Hermitian ``(d**3, d**3)`` numpy array.
    """
    ops = _site_ops(d)
    Sz = ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    Id = ops["Id"]

    # H_12: acts on sites 1,2; identity on site 3.
    h12 = delta * np.kron(np.kron(Sz, Sz), Id) + 0.5 * (
        np.kron(np.kron(Sp, Sm), Id) + np.kron(np.kron(Sm, Sp), Id)
    )
    # H_23: identity on site 1; acts on sites 2,3.
    h23 = delta * np.kron(Id, np.kron(Sz, Sz)) + 0.5 * (
        np.kron(Id, np.kron(Sp, Sm)) + np.kron(Id, np.kron(Sm, Sp))
    )
    # H_31: acts on sites 1,3; identity on site 2.
    h31 = delta * np.kron(Sz, np.kron(Id, Sz)) + 0.5 * (
        np.kron(Sp, np.kron(Id, Sm)) + np.kron(Sm, np.kron(Id, Sp))
    )

    return h12 + h23 + h31


def make_triangle_gate(
    H: np.ndarray, dt: complex, d: int = D_PHYS_DEFAULT
) -> np.ndarray:
    """Compute ``exp(-dt * H)`` and reshape to a 3-site Trotter gate.

    Args:
        H: ``(d**3, d**3)`` Hamiltonian matrix.
        dt: Time step. Real positive ``dt`` gives imaginary-time evolution;
            imaginary ``dt`` (e.g. ``1j * t``) gives real-time evolution.
        d: Physical dimension per site.

    Returns:
        ``(d, d, d, d, d, d)`` complex128 array — the 3-site Trotter gate.
    """
    gate = expm(-dt * H)
    return gate.reshape(d, d, d, d, d, d).astype(np.complex128)


def hosvd_truncate(
    theta: jnp.ndarray, D_max: int, d: int = D_PHYS_DEFAULT
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
    """Truncate a contracted 3-site tensor back into PESS form via HOSVD.

    Args:
        theta: Rank-6 tensor of shape ``(D_a, D_b, D_c, d, d, d)`` — the
            external bonds for sites a, b, c followed by their three physical
            indices, in matching order.
        D_max: Maximum internal bond dimension to keep on each leg of the
            truncated core. The actual kept dimension on a leg is
            ``min(D_max, D_x * d)`` where ``D_x`` is the external bond size.
        d: Physical dimension per site.

    Returns:
        A tuple ``(S_a, S_b, S_c, core, [lam_a, lam_b, lam_c])`` where
        ``S_x`` has shape ``(D_x_ext, D_x_int, d)``, ``core`` has shape
        ``(D_a_int, D_b_int, D_c_int)``, and each ``lam_x`` is a unit-norm
        singular-value vector along the corresponding internal bond. The
        reconstruction ``einsum("aip,bjq,ckr,ijk->abcpqr", S_a, S_b, S_c, core)``
        equals ``theta`` exactly when ``D_max >= D_x * d`` for every leg.
    """
    D_ext_a, D_ext_b, D_ext_c = theta.shape[0], theta.shape[1], theta.shape[2]

    theta_reordered = theta.transpose(0, 3, 1, 4, 2, 5)

    # Site a
    mat_a = theta_reordered.reshape(D_ext_a * d, D_ext_b * d * D_ext_c * d)
    U_a, _, _ = jnp.linalg.svd(mat_a, full_matrices=False)
    D_int_a = min(D_max, U_a.shape[1])
    U_a = U_a[:, :D_int_a]

    # Site b
    mat_b = theta_reordered.transpose(2, 3, 0, 1, 4, 5).reshape(
        D_ext_b * d, D_ext_a * d * D_ext_c * d
    )
    U_b, _, _ = jnp.linalg.svd(mat_b, full_matrices=False)
    D_int_b = min(D_max, U_b.shape[1])
    U_b = U_b[:, :D_int_b]

    # Site c
    mat_c = theta_reordered.transpose(4, 5, 0, 1, 2, 3).reshape(
        D_ext_c * d, D_ext_a * d * D_ext_b * d
    )
    U_c, _, _ = jnp.linalg.svd(mat_c, full_matrices=False)
    D_int_c = min(D_max, U_c.shape[1])
    U_c = U_c[:, :D_int_c]

    # Project onto truncated basis (complex-safe: use conj().T, not .T).
    theta_3mode = theta_reordered.reshape(D_ext_a * d, D_ext_b * d, D_ext_c * d)
    core = jnp.tensordot(U_a.conj().T, theta_3mode, axes=([1], [0]))
    core = jnp.tensordot(U_b.conj().T, core, axes=([1], [1]))
    core = core.transpose(1, 0, 2)
    core = jnp.tensordot(U_c.conj().T, core, axes=([1], [2]))
    core = core.transpose(1, 2, 0)

    # Extract per-bond singular value vectors from the core. We only need the
    # singular values, so skip the U/V allocations via compute_uv=False.
    lam_a = jnp.linalg.svd(core.reshape(D_int_a, D_int_b * D_int_c), compute_uv=False)
    lam_b = jnp.linalg.svd(
        core.transpose(1, 0, 2).reshape(D_int_b, D_int_a * D_int_c),
        compute_uv=False,
    )
    lam_c = jnp.linalg.svd(
        core.transpose(2, 0, 1).reshape(D_int_c, D_int_a * D_int_b),
        compute_uv=False,
    )

    lam_a = lam_a / jnp.linalg.norm(lam_a)
    lam_b = lam_b / jnp.linalg.norm(lam_b)
    lam_c = lam_c / jnp.linalg.norm(lam_c)

    S_a = U_a.reshape(D_ext_a, d, D_int_a).transpose(0, 2, 1)
    S_b = U_b.reshape(D_ext_b, d, D_int_b).transpose(0, 2, 1)
    S_c = U_c.reshape(D_ext_c, d, D_int_c).transpose(0, 2, 1)

    return S_a, S_b, S_c, core, [lam_a, lam_b, lam_c]


@dataclass(frozen=True)
class IPESSState:
    """Kagome iPESS parameters.

    R_a, R_b, R_c: rank-3 site tensors, shape (D, D, d). Index order is
        (leg-to-T_u, leg-to-T_d, physical).
    T_u, T_d: rank-3 simplex tensors, shape (D, D, D). Index order is
        (leg-to-R_a, leg-to-R_b, leg-to-R_c).
    lambdas: 6 bond singular-value vectors of length D, ordered
        (a-up, b-up, c-up, a-down, b-down, c-down).
    """

    R_a: jax.Array
    R_b: jax.Array
    R_c: jax.Array
    T_u: jax.Array
    T_d: jax.Array
    lambdas: tuple[jax.Array, ...]

    @classmethod
    def random(
        cls,
        D: int,
        d: int = D_PHYS_DEFAULT,
        key: jax.Array | None = None,
        scale: float = 0.1,
    ) -> IPESSState:
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        def cmplx(k, shape):
            re = jax.random.normal(k, shape) * scale
            im = jax.random.normal(jax.random.fold_in(k, 1), shape) * scale
            return (re + 1j * im).astype(jnp.complex128)

        return cls(
            R_a=cmplx(keys[0], (D, D, d)),
            R_b=cmplx(keys[1], (D, D, d)),
            R_c=cmplx(keys[2], (D, D, d)),
            T_u=cmplx(keys[3], (D, D, D)),
            T_d=cmplx(keys[4], (D, D, D)),
            lambdas=tuple(jnp.ones(D) for _ in range(6)),
        )
