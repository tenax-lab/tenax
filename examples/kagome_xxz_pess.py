#!/usr/bin/env python3
"""Kagome XXZ ground state via PESS simple update.

Implements Projected Entangled Simplex States (PESS) for the spin-1/2 XXZ
model on the Kagome lattice.  The algorithm performs imaginary-time simple
update on triangular simplices, then maps the PESS to an effective
square-lattice iPEPS for energy measurement via the Corner Transfer Matrix
(CTM) method.

Reference: Xie et al., PRL 112, 147203 (2014).

Usage::

    python examples/kagome_xxz_pess.py --D 2 --chi 20 --delta 1.0 --steps 200 --dt 0.01

"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)

from tenax import (  # noqa: E402
    CTMConfig,
    iPEPSConfig,
    optimize_gs_ad,
)
from tenax.algorithms.ipeps_ctm import ctm  # noqa: E402
from tenax.algorithms.ipeps_rdm import compute_energy_ctm  # noqa: E402

# ---------------------------------------------------------------------------
# Kagome triangle Hamiltonian
# ---------------------------------------------------------------------------


def kagome_triangle_hamiltonian(delta: float = 1.0) -> np.ndarray:
    """Build the 3-site XXZ Hamiltonian on a Kagome triangle.

    H_tri = H_12 + H_23 + H_31 where each pair term is
    H_ij = delta * Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j).

    Returns:
        Real (8, 8) numpy array.
    """
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def two_site_term(A, B):
        return delta * np.kron(A, B) + 0.5 * (
            np.kron(Sp if A is Sz else A, Sm if B is Sz else B)
            + np.kron(Sm if A is Sz else A, Sp if B is Sz else B)
        )

    # Build pair Hamiltonians in the 3-site Hilbert space (dim 8)
    # H_12: acts on sites 1,2; identity on site 3
    h12 = delta * np.kron(np.kron(Sz, Sz), I2) + 0.5 * (
        np.kron(np.kron(Sp, Sm), I2) + np.kron(np.kron(Sm, Sp), I2)
    )
    # H_23: identity on site 1; acts on sites 2,3
    h23 = delta * np.kron(I2, np.kron(Sz, Sz)) + 0.5 * (
        np.kron(I2, np.kron(Sp, Sm)) + np.kron(I2, np.kron(Sm, Sp))
    )
    # H_31: acts on sites 1,3; identity on site 2
    h31 = delta * np.kron(Sz, np.kron(I2, Sz)) + 0.5 * (
        np.kron(Sp, np.kron(I2, Sm)) + np.kron(Sm, np.kron(I2, Sp))
    )

    return h12 + h23 + h31


def make_trotter_gate_3site(H: np.ndarray, dt: float) -> np.ndarray:
    """Compute exp(-dt * H) and reshape to (2,2,2, 2,2,2).

    Args:
        H: (8, 8) Hamiltonian matrix.
        dt: Imaginary time step.

    Returns:
        (2, 2, 2, 2, 2, 2) numpy array — the 3-site Trotter gate.
    """
    gate = expm(-dt * H)
    return gate.reshape(2, 2, 2, 2, 2, 2)


# ---------------------------------------------------------------------------
# PESS initialization
# ---------------------------------------------------------------------------


def init_pess(
    D: int, d: int = 2, key: jax.Array | None = None
) -> tuple[list, list, list]:
    """Create random PESS tensors for the Kagome lattice.

    The unit cell has 3 site tensors and 2 simplex tensors.
    Each site tensor S has shape (D_ext, D_int, d):
      - D_ext connects to the *other* simplex
      - D_int connects to *this* simplex
      - d is the physical dimension

    Each simplex tensor T has shape (D_int, D_int, D_int).

    Lambda vectors live on each of the 6 bonds (3 internal per simplex,
    but bonds are shared between up and down triangles — in the infinite
    lattice, each site has one leg to the up simplex and one to the down
    simplex).

    Args:
        D: Bond dimension.
        d: Physical dimension (default 2 for spin-1/2).
        key: JAX PRNG key.

    Returns:
        (site_tensors, simplex_tensors, lambdas) where
        site_tensors = [S_a, S_b, S_c]  shape (D, D, d)
        simplex_tensors = [T_up, T_down]  shape (D, D, D)
        lambdas = [lam_a_up, lam_b_up, lam_c_up,
                   lam_a_down, lam_b_down, lam_c_down]
                  each of shape (D,)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    keys = jax.random.split(key, 5)

    # Site tensors: (D_ext, D_int, d)
    S_a = jax.random.normal(keys[0], (D, D, d)) * 0.1
    S_b = jax.random.normal(keys[1], (D, D, d)) * 0.1
    S_c = jax.random.normal(keys[2], (D, D, d)) * 0.1

    # Simplex tensors: (D, D, D)
    T_up = jax.random.normal(keys[3], (D, D, D)) * 0.1
    T_down = jax.random.normal(keys[4], (D, D, D)) * 0.1

    # Lambda vectors — initialized to ones (identity mean-field)
    lambdas = [jnp.ones(D) for _ in range(6)]

    return [S_a, S_b, S_c], [T_up, T_down], lambdas


# ---------------------------------------------------------------------------
# HOSVD truncation
# ---------------------------------------------------------------------------


def hosvd_truncate(
    theta: jnp.ndarray, D_max: int, d: int = 2
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list]:
    """Truncate a contracted 3-site tensor back into PESS form via HOSVD.

    Given theta of shape (D_ext_a, D_ext_b, D_ext_c, d, d, d) — the
    full contraction of 3 site tensors + simplex + gate — decompose it
    into S_a, S_b, S_c (site tensors) and T (simplex core) plus updated
    lambda vectors on the 3 internal bonds.

    The HOSVD proceeds by:
    1. For each site leg, form the mode-unfolding, do SVD, and keep the
       top D_max left singular vectors as the new site isometry.
    2. Project theta onto the truncated basis to get the core (new T).
    3. Extract lambda vectors from the core via SVD along each mode.

    Args:
        theta: Shape (D_ext_a, D_ext_b, D_ext_c, d_a, d_b, d_c).
        D_max: Maximum internal bond dimension to keep.
        d: Physical dimension per site.

    Returns:
        (S_a, S_b, S_c, T_new, [lam_a, lam_b, lam_c])
        S_x shape: (D_ext, D_int, d)
        T_new shape: (D_int, D_int, D_int)
    """
    D_ext_a, D_ext_b, D_ext_c = theta.shape[0], theta.shape[1], theta.shape[2]

    # Mode unfoldings and truncated SVD for each site
    # Site a: group (D_ext_a, d_a) vs rest
    mat_a = theta.reshape(
        D_ext_a * d, -1
    )  # (D_ext_a * d) x (D_ext_b * D_ext_c * d * d)
    # Reorder: (D_ext_a, D_ext_b, D_ext_c, d_a, d_b, d_c) -> (D_ext_a, d_a, D_ext_b, d_b, D_ext_c, d_c)
    theta_reordered = theta.transpose(0, 3, 1, 4, 2, 5)
    # Now reshape for each mode

    # Site a: mode = (D_ext_a, d_a), remainder = (D_ext_b, d_b, D_ext_c, d_c)
    mat_a = theta_reordered.reshape(D_ext_a * d, D_ext_b * d * D_ext_c * d)
    U_a, s_a, _ = jnp.linalg.svd(mat_a, full_matrices=False)
    D_int_a = min(D_max, U_a.shape[1])
    U_a = U_a[:, :D_int_a]  # (D_ext_a * d, D_int_a)

    # Site b: mode = (D_ext_b, d_b), remainder = (D_ext_a, d_a, D_ext_c, d_c)
    mat_b = theta_reordered.transpose(1, 0, 2, 3, 4, 5).reshape(
        D_ext_b * d, D_ext_a * d * D_ext_c * d
    )  # wrong — need to rearrange properly
    # Actually: theta_reordered is (D_ext_a, d_a, D_ext_b, d_b, D_ext_c, d_c)
    # For site b mode unfolding: group (D_ext_b, d_b) vs (D_ext_a, d_a, D_ext_c, d_c)
    mat_b = theta_reordered.transpose(2, 3, 0, 1, 4, 5).reshape(
        D_ext_b * d, D_ext_a * d * D_ext_c * d
    )
    U_b, s_b, _ = jnp.linalg.svd(mat_b, full_matrices=False)
    D_int_b = min(D_max, U_b.shape[1])
    U_b = U_b[:, :D_int_b]

    # Site c: mode = (D_ext_c, d_c), remainder = (D_ext_a, d_a, D_ext_b, d_b)
    mat_c = theta_reordered.transpose(4, 5, 0, 1, 2, 3).reshape(
        D_ext_c * d, D_ext_a * d * D_ext_b * d
    )
    U_c, s_c, _ = jnp.linalg.svd(mat_c, full_matrices=False)
    D_int_c = min(D_max, U_c.shape[1])
    U_c = U_c[:, :D_int_c]

    # Project theta onto truncated basis to get the core tensor
    # core = U_a^T . theta_reordered . U_b^T . U_c^T  (mode products)
    # theta_reordered: (D_ext_a*d_a, D_ext_b*d_b, D_ext_c*d_c)
    theta_3mode = theta_reordered.reshape(D_ext_a * d, D_ext_b * d, D_ext_c * d)
    # Mode-0 projection
    core = jnp.tensordot(U_a.T, theta_3mode, axes=([1], [0]))
    # core: (D_int_a, D_ext_b*d_b, D_ext_c*d_c)
    core = jnp.tensordot(U_b.T, core, axes=([1], [1]))
    # core: (D_int_b, D_int_a, D_ext_c*d_c) -> transpose to (D_int_a, D_int_b, D_ext_c*d_c)
    core = core.transpose(1, 0, 2)
    core = jnp.tensordot(U_c.T, core, axes=([1], [2]))
    # core: (D_int_c, D_int_a, D_int_b) -> transpose to (D_int_a, D_int_b, D_int_c)
    core = core.transpose(1, 2, 0)

    # Extract per-bond singular value vectors from the core via SVD along each mode.
    sig_a = jnp.linalg.svd(core.reshape(D_int_a, D_int_b * D_int_c), compute_uv=False)
    sig_b = jnp.linalg.svd(
        core.transpose(1, 0, 2).reshape(D_int_b, D_int_a * D_int_c),
        compute_uv=False,
    )
    sig_c = jnp.linalg.svd(
        core.transpose(2, 0, 1).reshape(D_int_c, D_int_a * D_int_b),
        compute_uv=False,
    )

    # Factor the bond spectra OUT of ``core`` so they live exclusively in the
    # explicit lambdas. Without this step ``core`` carries the spectra baked
    # into its matricizations AND the same spectra are re-applied as gauges
    # next SU step (via ``S_a_w = lam_ext * R * lam_int``), squaring the
    # bond spectrum every iteration → rank-1 collapse → energy stuck at the
    # classical 120° value. Lorentzian-regularized inverse for stability;
    # scalar prefactor ``K = ||σ_a||·||σ_b||·||σ_c||`` keeps reconstruction
    # with NORMALIZED lambdas exact.
    eps = 1e-12
    inv_a = (sig_a / (sig_a**2 + eps**2)).astype(core.dtype)
    inv_b = (sig_b / (sig_b**2 + eps**2)).astype(core.dtype)
    inv_c = (sig_c / (sig_c**2 + eps**2)).astype(core.dtype)
    norm_a = jnp.linalg.norm(sig_a)
    norm_b = jnp.linalg.norm(sig_b)
    norm_c = jnp.linalg.norm(sig_c)
    K = (norm_a * norm_b * norm_c).astype(core.dtype)
    core = core * inv_a[:, None, None] * inv_b[None, :, None] * inv_c[None, None, :] * K

    lam_a = sig_a / norm_a
    lam_b = sig_b / norm_b
    lam_c = sig_c / norm_c

    # Reshape U matrices into site tensors: (D_ext, D_int, d)
    S_a = U_a.reshape(D_ext_a, d, D_int_a).transpose(0, 2, 1)
    S_b = U_b.reshape(D_ext_b, d, D_int_b).transpose(0, 2, 1)
    S_c = U_c.reshape(D_ext_c, d, D_int_c).transpose(0, 2, 1)

    return S_a, S_b, S_c, core, [lam_a, lam_b, lam_c]


# ---------------------------------------------------------------------------
# PESS simple update
# ---------------------------------------------------------------------------


def pess_simple_update_triangle(
    S_a: jnp.ndarray,
    S_b: jnp.ndarray,
    S_c: jnp.ndarray,
    T: jnp.ndarray,
    lambdas_ext: list[jnp.ndarray],
    lambdas_int: list[jnp.ndarray],
    gate: np.ndarray,
    D_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list]:
    """One simple-update step on a single triangle.

    Steps:
    1. Absorb external lambda weights onto site tensors.
    2. Absorb internal lambda weights onto site tensors.
    3. Contract S_a, S_b, S_c with T into theta.
    4. Apply the 3-site Trotter gate.
    5. HOSVD decompose back into S_a', S_b', S_c', T', lambdas_int'.
    6. Remove external lambda weights from site tensors.

    Args:
        S_a, S_b, S_c: Site tensors, shape (D_ext, D_int, d).
        T: Simplex tensor, shape (D_int, D_int, D_int).
        lambdas_ext: [lam_a_ext, lam_b_ext, lam_c_ext] — weights on
                     external bonds (connecting to the other simplex).
        lambdas_int: [lam_a_int, lam_b_int, lam_c_int] — weights on
                     internal bonds (connecting to this simplex).
        gate: 3-site Trotter gate, shape (2, 2, 2, 2, 2, 2).
        D_max: Maximum bond dimension.

    Returns:
        (S_a', S_b', S_c', T', lambdas_int')
    """
    d = S_a.shape[2]

    # 1. Absorb external lambdas: multiply onto ext leg (axis 0)
    S_a_w = jnp.einsum("i,ijd->ijd", lambdas_ext[0], S_a)
    S_b_w = jnp.einsum("i,ijd->ijd", lambdas_ext[1], S_b)
    S_c_w = jnp.einsum("i,ijd->ijd", lambdas_ext[2], S_c)

    # 2. Absorb internal lambdas: multiply onto int leg (axis 1)
    S_a_w = jnp.einsum("ijd,j->ijd", S_a_w, lambdas_int[0])
    S_b_w = jnp.einsum("ijd,j->ijd", S_b_w, lambdas_int[1])
    S_c_w = jnp.einsum("ijd,j->ijd", S_c_w, lambdas_int[2])

    # 3. Contract site tensors with simplex tensor into theta
    # S_a_w: (D_ext_a, D_int_a, d_a), T: (D_int_a, D_int_b, D_int_c)
    # theta = S_a[ea, ia, da] * S_b[eb, ib, db] * S_c[ec, ic, dc] * T[ia, ib, ic]
    # Result: theta(ea, eb, ec, da, db, dc)
    theta = jnp.einsum(
        "xad,ybf,zcg,abc->xyzdfg",
        S_a_w,
        S_b_w,
        S_c_w,
        T,
    )
    # theta shape: (D_ext_a, D_ext_b, D_ext_c, d_a, d_b, d_c)

    # 4. Apply the 3-site gate
    # gate: (d_a', d_b', d_c', d_a, d_b, d_c)
    gate_jnp = jnp.array(gate)
    theta = jnp.einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate_jnp)

    # 5. HOSVD decompose
    S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new = hosvd_truncate(theta, D_max, d)

    # 6. Remove external lambdas from site tensors
    lam_a_inv = jnp.where(lambdas_ext[0] > 1e-12, 1.0 / lambdas_ext[0], 0.0)
    lam_b_inv = jnp.where(lambdas_ext[1] > 1e-12, 1.0 / lambdas_ext[1], 0.0)
    lam_c_inv = jnp.where(lambdas_ext[2] > 1e-12, 1.0 / lambdas_ext[2], 0.0)
    S_a_new = jnp.einsum("i,ijd->ijd", lam_a_inv, S_a_new)
    S_b_new = jnp.einsum("i,ijd->ijd", lam_b_inv, S_b_new)
    S_c_new = jnp.einsum("i,ijd->ijd", lam_c_inv, S_c_new)

    return S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new


def pess_simple_update(
    site_tensors: list,
    simplex_tensors: list,
    lambdas: list,
    H_tri: np.ndarray,
    dt: float,
    D_max: int,
    num_steps: int,
) -> tuple[list, list, list]:
    """Full PESS simple update loop alternating up and down triangles.

    Lambda indexing:
        [0..2] = internal bonds of up triangle   (lam_a_up, lam_b_up, lam_c_up)
        [3..5] = internal bonds of down triangle  (lam_a_down, lam_b_down, lam_c_down)

    The external lambdas for the up triangle are the down-triangle internal
    lambdas (and vice versa), since each site connects to both simplices.

    Args:
        site_tensors: [S_a, S_b, S_c].
        simplex_tensors: [T_up, T_down].
        lambdas: 6 lambda vectors.
        H_tri: (8, 8) triangle Hamiltonian.
        dt: Imaginary time step.
        D_max: Bond dimension cutoff.
        num_steps: Number of simple-update sweeps.

    Returns:
        Updated (site_tensors, simplex_tensors, lambdas).
    """
    gate = make_trotter_gate_3site(H_tri, dt)

    S_a, S_b, S_c = site_tensors
    T_up, T_down = simplex_tensors

    for step in range(num_steps):
        # --- Update up triangle ---
        # Internal bonds: lambdas[0:3], external bonds: lambdas[3:6]
        S_a, S_b, S_c, T_up, lam_int_up = pess_simple_update_triangle(
            S_a,
            S_b,
            S_c,
            T_up,
            lambdas_ext=[lambdas[3], lambdas[4], lambdas[5]],
            lambdas_int=[lambdas[0], lambdas[1], lambdas[2]],
            gate=gate,
            D_max=D_max,
        )
        lambdas[0], lambdas[1], lambdas[2] = lam_int_up

        # --- Update down triangle ---
        # Internal bonds: lambdas[3:6], external bonds: lambdas[0:3].
        # ``pess_simple_update_triangle`` always applies ext on axis 0 of S
        # and contracts the simplex via axis 1 of S, but per the iPESS
        # convention used everywhere else (axis 0 = T_d-leg, axis 1 =
        # T_u-leg) the down-triangle's ext side (T_u-leg) is axis 1.
        # Transpose S_x around the call so axis 0 sees the ext side, then
        # transpose back. Without this, T_d would contract the T_u-leg of S
        # and the down-triangle would not actually evolve under SU.
        S_a_T = S_a.transpose(1, 0, 2)
        S_b_T = S_b.transpose(1, 0, 2)
        S_c_T = S_c.transpose(1, 0, 2)
        S_a_T, S_b_T, S_c_T, T_down, lam_int_down = pess_simple_update_triangle(
            S_a_T,
            S_b_T,
            S_c_T,
            T_down,
            lambdas_ext=[lambdas[0], lambdas[1], lambdas[2]],
            lambdas_int=[lambdas[3], lambdas[4], lambdas[5]],
            gate=gate,
            D_max=D_max,
        )
        S_a = S_a_T.transpose(1, 0, 2)
        S_b = S_b_T.transpose(1, 0, 2)
        S_c = S_c_T.transpose(1, 0, 2)
        lambdas[3], lambdas[4], lambdas[5] = lam_int_down

        if (step + 1) % max(1, num_steps // 10) == 0 or step == 0:
            # Print a progress indicator
            norm = sum(float(jnp.linalg.norm(lv)) for lv in lambdas)
            print(f"  SU step {step + 1:4d}/{num_steps}  lambda_norm={norm:.6f}")

    return [S_a, S_b, S_c], [T_up, T_down], lambdas


# ---------------------------------------------------------------------------
# PESS -> iPEPS coarse-graining
# ---------------------------------------------------------------------------


def pess_to_ipeps(
    site_tensors: list,
    simplex_tensors: list,
    lambdas: list,
) -> jnp.ndarray:
    """Contract PESS into an effective square-lattice iPEPS super-site tensor.

    Coarse-grains one up-triangle (S_a, S_b, S_c, T_up) into a single
    tensor with:
      - 4 virtual legs of dimension D^2 (for the square lattice)
      - 1 physical leg of dimension d^3 = 8

    The mapping:
    1. Absorb lambda weights onto site tensors.
    2. Contract S_a, S_b, S_c with T_up.
    3. Reshape/group virtual legs for a square lattice layout.

    The square lattice virtual legs are formed by pairing the external
    bonds of site tensors:
      - up:    (ext_a, ext_b) -> D^2
      - right: (ext_b, ext_c) -> D^2
      - down:  (ext_c, ext_a) -> D^2
      - left:  (ext_a, ext_c) -> D^2  [or similar pairing]

    For a first implementation we use a simpler scheme: contract the full
    triangle into a tensor of shape (D_ext^3, d^3) then pad/reshape into
    (D_eff, D_eff, D_eff, D_eff, d_eff) with D_eff = D^2.

    Returns:
        A: shape (D_eff, D_eff, D_eff, D_eff, d_eff) iPEPS site tensor.
    """
    S_a, S_b, S_c = site_tensors
    T_up = simplex_tensors[0]
    D = S_a.shape[0]
    d = S_a.shape[2]

    # Absorb lambda weights
    # Up-triangle internal lambdas on int legs (axis 1)
    S_a_w = jnp.einsum("ijd,j->ijd", S_a, lambdas[0])
    S_b_w = jnp.einsum("ijd,j->ijd", S_b, lambdas[1])
    S_c_w = jnp.einsum("ijd,j->ijd", S_c, lambdas[2])

    # Also absorb sqrt of external (down) lambdas on ext legs (axis 0)
    # so the effective iPEPS includes the mean-field environment
    sqrt_lam_a = jnp.sqrt(jnp.maximum(lambdas[3], 1e-14))
    sqrt_lam_b = jnp.sqrt(jnp.maximum(lambdas[4], 1e-14))
    sqrt_lam_c = jnp.sqrt(jnp.maximum(lambdas[5], 1e-14))
    S_a_w = jnp.einsum("i,ijd->ijd", sqrt_lam_a, S_a_w)
    S_b_w = jnp.einsum("i,ijd->ijd", sqrt_lam_b, S_b_w)
    S_c_w = jnp.einsum("i,ijd->ijd", sqrt_lam_c, S_c_w)

    # Contract with T_up: T_up[ia, ib, ic] * S_a[ea, ia, da] * S_b[eb, ib, db] * S_c[ec, ic, dc]
    # Result: (ea, eb, ec, da, db, dc)
    theta = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", S_a_w, S_b_w, S_c_w, T_up)

    # Reshape into iPEPS site tensor (D_eff, D_eff, D_eff, D_eff, d_eff)
    # Physical dimension: d^3 = 8
    d_eff = d**3

    # Virtual legs: we have 3 external legs of dimension D each.
    # For a square-lattice iPEPS we need 4 virtual legs.
    # Strategy: pair external legs and pad the 4th with a trivial dim.
    # A[up, right, down, left, phys]
    # up = ea, right = eb, down = ec, left = 1 (trivial)
    # This gives D_eff = D for 3 legs, 1 for the 4th.
    # For better compatibility, reshape so all virtual dims are D:
    # (D, D, D, 1, d^3) -> pad left dim to D by adding zeros
    theta_phys = theta.reshape(D, D, D, d_eff)

    # Pad to 5-leg tensor with 4th virtual leg of dim D (pad with zeros)
    A = jnp.zeros((D, D, D, D, d_eff))
    # Embed: A[ea, eb, ec, 0, :] = theta_phys[ea, eb, ec, :]
    A = A.at[:, :, :, 0, :].set(theta_phys)

    return A


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_pess(
    filename: str,
    site_tensors: list,
    simplex_tensors: list,
    lambdas: list,
) -> None:
    """Save PESS tensors to a .npz file."""
    data = {}
    for i, name in enumerate(["S_a", "S_b", "S_c"]):
        data[name] = np.array(site_tensors[i])
    for i, name in enumerate(["T_up", "T_down"]):
        data[name] = np.array(simplex_tensors[i])
    for i in range(6):
        data[f"lambda_{i}"] = np.array(lambdas[i])
    np.savez(filename, **data)
    print(f"Saved PESS state to {filename}")


def load_pess(filename: str) -> tuple[list, list, list]:
    """Load PESS tensors from a .npz file."""
    data = np.load(filename)
    site_tensors = [
        jnp.array(data["S_a"]),
        jnp.array(data["S_b"]),
        jnp.array(data["S_c"]),
    ]
    simplex_tensors = [
        jnp.array(data["T_up"]),
        jnp.array(data["T_down"]),
    ]
    lambdas = [jnp.array(data[f"lambda_{i}"]) for i in range(6)]
    print(f"Loaded PESS state from {filename}")
    return site_tensors, simplex_tensors, lambdas


# ---------------------------------------------------------------------------
# Energy measurement
# ---------------------------------------------------------------------------


def compute_pess_energy(
    site_tensors: list,
    simplex_tensors: list,
    lambdas: list,
    H_tri: np.ndarray,
    chi: int,
) -> float:
    """Compute energy per site via PESS -> iPEPS -> CTM.

    1. Map PESS to effective square-lattice iPEPS.
    2. Run CTM to get the environment.
    3. Compute energy using 2-site RDM with an effective 2-site gate.

    For the Kagome lattice with 3 sites per triangle, the energy per site
    is E_triangle / 3.

    Args:
        site_tensors: [S_a, S_b, S_c].
        simplex_tensors: [T_up, T_down].
        lambdas: 6 lambda vectors.
        H_tri: (8, 8) triangle Hamiltonian.
        chi: CTM bond dimension.

    Returns:
        Energy per site (float).
    """
    A = pess_to_ipeps(site_tensors, simplex_tensors, lambdas)
    d_eff = A.shape[4]

    # Build an effective 2-site gate: H_tri (x) I measures the triangle
    # Hamiltonian on the first super-site of each bond pair only.
    # compute_energy_ctm returns E_h + E_v = 2*<H_tri>, correctly
    # accounting for the 2 triangles (up + down) per unit cell.
    # Dividing by 3 gives E/site = 2<H_tri>/3.
    H_eff = np.kron(H_tri, np.eye(d_eff))
    H_eff_4leg = H_eff.reshape(d_eff, d_eff, d_eff, d_eff)

    config = CTMConfig(chi=chi, max_iter=100)
    env = ctm(A, config)

    E = compute_energy_ctm(A, env, jnp.array(H_eff_4leg), d_eff)
    E_per_site = float(E) / 3.0

    return E_per_site


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Kagome XXZ ground state via PESS simple update"
    )
    parser.add_argument("--D", type=int, default=2, help="Bond dimension")
    parser.add_argument("--chi", type=int, default=20, help="CTM bond dimension")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="XXZ anisotropy parameter"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simple-update steps"
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Imaginary time step")
    parser.add_argument("--save", type=str, default=None, help="Save PESS to file")
    parser.add_argument("--load", type=str, default=None, help="Load PESS from file")
    parser.add_argument(
        "--ad", action="store_true", help="Run AD optimization after SU"
    )
    parser.add_argument(
        "--ad-steps", type=int, default=100, help="Number of AD optimization steps"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="AD learning rate")
    args = parser.parse_args()

    print("Kagome XXZ via PESS simple update")
    print(f"  delta={args.delta}, D={args.D}, chi={args.chi}")
    print(f"  SU steps={args.steps}, dt={args.dt}")

    # Build Hamiltonian
    H_tri = kagome_triangle_hamiltonian(args.delta)

    # Initialize or load tensors
    if args.load:
        site_tensors, simplex_tensors, lambdas = load_pess(args.load)
    else:
        site_tensors, simplex_tensors, lambdas = init_pess(args.D)

    # Run simple update
    if args.steps > 0:
        print("\n--- Simple Update ---")
        t0 = time.perf_counter()
        site_tensors, simplex_tensors, lambdas = pess_simple_update(
            site_tensors,
            simplex_tensors,
            lambdas,
            H_tri,
            args.dt,
            args.D,
            args.steps,
        )
        su_time = time.perf_counter() - t0
        print(f"  SU completed in {su_time:.1f}s")

    # Optionally save
    if args.save:
        save_pess(args.save, site_tensors, simplex_tensors, lambdas)

    # Compute energy via CTM
    print("\n--- CTM Energy ---")
    t0 = time.perf_counter()
    energy = compute_pess_energy(
        site_tensors, simplex_tensors, lambdas, H_tri, args.chi
    )
    ctm_time = time.perf_counter() - t0
    print(f"  E/site = {energy:.6f}")
    print(f"  CTM time = {ctm_time:.1f}s")

    # Optionally run AD optimization on the mapped iPEPS
    if args.ad:
        print("\n--- AD Optimization ---")
        A_init = pess_to_ipeps(site_tensors, simplex_tensors, lambdas)
        d_eff = 8
        # Build effective 2-site gate for AD
        H_eff = np.kron(H_tri, np.eye(d_eff)) + np.kron(np.eye(d_eff), H_tri)
        H_eff_gate = jnp.array(H_eff.reshape(d_eff, d_eff, d_eff, d_eff))

        config = iPEPSConfig(
            max_bond_dim=args.D,
            ctm=CTMConfig(chi=args.chi, max_iter=100),
            gs_optimizer="adam",
            gs_learning_rate=args.lr,
            gs_num_steps=args.ad_steps,
            su_init=False,
        )

        t0 = time.perf_counter()
        A_opt, env, E_gs = optimize_gs_ad(H_eff_gate, A_init, config)
        ad_time = time.perf_counter() - t0
        E_per_site = float(E_gs) / 3.0
        print(f"  AD E/site = {E_per_site:.6f}")
        print(f"  AD time = {ad_time:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
