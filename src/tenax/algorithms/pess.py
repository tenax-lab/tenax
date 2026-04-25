"""iPESS (infinite Projected Entangled Simplex State) on the kagome lattice."""

from __future__ import annotations

from dataclasses import dataclass, replace

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


def pess_simple_update_triangle(
    state: IPESSState,
    gate: jax.Array,
    triangle: str,
    D_max: int,
) -> IPESSState:
    """One simple-update step on a single kagome triangle.

    Absorbs the triangle's external and internal bond weights into the three
    site tensors, contracts them with the chosen simplex tensor and the 3-site
    gate, then HOSVD-truncates the result back into iPESS form.

    Args:
        state: Input iPESS state.
        gate: 3-site gate of shape ``(d, d, d, d, d, d)``. Index order matches
            ``einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate)`` from the example
            (last 3 axes are ``ket``, first 3 are ``bra``).
        triangle: ``"up"`` updates ``T_u`` and the up-bond lambdas
            (indices 0, 1, 2); ``"down"`` updates ``T_d`` and the down-bond
            lambdas (indices 3, 4, 5).
        D_max: Maximum internal bond dimension kept by the HOSVD truncation.

    Returns:
        New :class:`IPESSState` with the updated site tensors, the chosen
        simplex tensor, and the matching internal lambda triplet replaced;
        the other simplex and the external lambdas are unchanged.

    Raises:
        ValueError: If ``triangle`` is not ``"up"`` or ``"down"``.
    """
    if triangle == "up":
        T = state.T_u
        ext_idx = (3, 4, 5)
        int_idx = (0, 1, 2)
    elif triangle == "down":
        T = state.T_d
        ext_idx = (0, 1, 2)
        int_idx = (3, 4, 5)
    else:
        raise ValueError(f"triangle must be 'up' or 'down'; got {triangle!r}.")

    lam_ext = (
        state.lambdas[ext_idx[0]],
        state.lambdas[ext_idx[1]],
        state.lambdas[ext_idx[2]],
    )
    lam_int = (
        state.lambdas[int_idx[0]],
        state.lambdas[int_idx[1]],
        state.lambdas[int_idx[2]],
    )

    R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    d = R_a.shape[2]

    # Weight site tensors with ext (axis 0) and int (axis 1) lambdas.
    S_a_w = jnp.einsum("i,ijd,j->ijd", lam_ext[0], R_a, lam_int[0])
    S_b_w = jnp.einsum("i,ijd,j->ijd", lam_ext[1], R_b, lam_int[1])
    S_c_w = jnp.einsum("i,ijd,j->ijd", lam_ext[2], R_c, lam_int[2])

    # Contract weighted site tensors with the simplex into theta, apply gate.
    theta = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", S_a_w, S_b_w, S_c_w, T)
    theta = jnp.einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate)

    # HOSVD-truncate back to iPESS form.
    S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new = hosvd_truncate(theta, D_max, d)

    # Strip the external lambdas back off so the returned R tensors are
    # "ungauged" again. Safe-divide to avoid division by zero.
    def _safe_inv(lam):
        return jnp.where(lam > 1e-12, 1.0 / lam, 0.0)

    inv_a = _safe_inv(lam_ext[0])
    inv_b = _safe_inv(lam_ext[1])
    inv_c = _safe_inv(lam_ext[2])
    R_a_new = jnp.einsum("i,ijd->ijd", inv_a, S_a_new)
    R_b_new = jnp.einsum("i,ijd->ijd", inv_b, S_b_new)
    R_c_new = jnp.einsum("i,ijd->ijd", inv_c, S_c_new)

    # Build new lambdas tuple with the int triplet replaced.
    new_lambdas = list(state.lambdas)
    new_lambdas[int_idx[0]] = lambdas_int_new[0]
    new_lambdas[int_idx[1]] = lambdas_int_new[1]
    new_lambdas[int_idx[2]] = lambdas_int_new[2]

    if triangle == "up":
        return replace(
            state,
            R_a=R_a_new,
            R_b=R_b_new,
            R_c=R_c_new,
            T_u=T_new,
            lambdas=tuple(new_lambdas),
        )
    return replace(
        state,
        R_a=R_a_new,
        R_b=R_b_new,
        R_c=R_c_new,
        T_d=T_new,
        lambdas=tuple(new_lambdas),
    )


def pess_simple_update(
    state: IPESSState,
    hamiltonian: np.ndarray,
    dt_schedule: list[tuple[float, int]],
    D_max: int,
) -> IPESSState:
    """Run alternating up/down triangle simple update on a kagome iPESS state.

    For each ``(dt, num_steps)`` segment in ``dt_schedule``, build the 3-site
    Trotter gate ``exp(-dt * H)`` once and run ``num_steps`` SU iterations.
    Each iteration performs one up-triangle update followed by one
    down-triangle update.

    Args:
        state: Input iPESS state.
        hamiltonian: ``(d**3, d**3)`` Hermitian Hamiltonian on a triangle
            (e.g. from :func:`kagome_triangle_xxz_hamiltonian`).
        dt_schedule: List of ``(dt, num_steps)`` tuples, e.g.
            ``[(0.1, 200), (0.01, 200), (0.001, 100)]``. Real positive ``dt``
            performs imaginary-time evolution.
        D_max: Maximum internal bond dimension preserved by the per-step
            HOSVD truncation.

    Returns:
        New :class:`IPESSState` after the full schedule.
    """
    d = state.R_a.shape[2]
    for dt, num_steps in dt_schedule:
        gate = jnp.asarray(make_triangle_gate(hamiltonian, dt, d))
        for _ in range(num_steps):
            state = pess_simple_update_triangle(state, gate, "up", D_max)
            state = pess_simple_update_triangle(state, gate, "down", D_max)
    return state
