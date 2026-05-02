#!/usr/bin/env python3
"""Kagome XXZ ground state for spin-1 via PESS simple update.

Adapts the spin-1/2 PESS example to spin-1 (d=3) using spin_one_ops().
The PESS structure and update algorithm are identical; only the
Hamiltonian and physical dimension change.

Reference: Xie et al., PRL 112, 147203 (2014).

Usage::

    python examples/kagome_xxz_spin1_pess.py --D 2 --chi 20 --delta 1.0 --steps 200 --dt 0.01

"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, spin_one_ops  # noqa: E402
from tenax.algorithms.ipeps_ctm import ctm  # noqa: E402
from tenax.algorithms.ipeps_rdm import compute_energy_ctm  # noqa: E402

# Physical dimension for spin-1
D_PHYS = 3

# ---------------------------------------------------------------------------
# Kagome triangle Hamiltonian (spin-1)
# ---------------------------------------------------------------------------


def kagome_triangle_hamiltonian_spin1(delta: float = 1.0) -> np.ndarray:
    """Build the 3-site XXZ Hamiltonian on a Kagome triangle for spin-1.

    H_tri = H_12 + H_23 + H_31 where each pair term is
    H_ij = delta * Sz_i Sz_j + 0.5 * (S+_i S-_j + S-_i S+_j).

    Returns:
        Real (27, 27) numpy array.
    """
    ops = spin_one_ops()
    Sz = ops["Sz"]  # (3, 3)
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    I3 = ops["Id"]

    # H_12: acts on sites 1,2; identity on site 3
    h12 = delta * np.kron(np.kron(Sz, Sz), I3) + 0.5 * (
        np.kron(np.kron(Sp, Sm), I3) + np.kron(np.kron(Sm, Sp), I3)
    )
    # H_23: identity on site 1; acts on sites 2,3
    h23 = delta * np.kron(I3, np.kron(Sz, Sz)) + 0.5 * (
        np.kron(I3, np.kron(Sp, Sm)) + np.kron(I3, np.kron(Sm, Sp))
    )
    # H_31: acts on sites 1,3; identity on site 2
    h31 = delta * np.kron(Sz, np.kron(I3, Sz)) + 0.5 * (
        np.kron(Sp, np.kron(I3, Sm)) + np.kron(Sm, np.kron(I3, Sp))
    )

    return h12 + h23 + h31


def make_trotter_gate_3site(H: np.ndarray, dt: float, d: int = D_PHYS) -> np.ndarray:
    """Compute exp(-dt * H) and reshape to (d,d,d, d,d,d).

    Args:
        H: (d^3, d^3) Hamiltonian matrix.
        dt: Imaginary time step.
        d: Physical dimension per site.

    Returns:
        (d, d, d, d, d, d) numpy array — the 3-site Trotter gate.
    """
    gate = expm(-dt * H)
    return gate.reshape(d, d, d, d, d, d)


# ---------------------------------------------------------------------------
# PESS initialization (unchanged, just d=3)
# ---------------------------------------------------------------------------


def init_pess(
    D: int, d: int = D_PHYS, key: jax.Array | None = None
) -> tuple[list, list, list]:
    """Create random PESS tensors for the Kagome lattice.

    Args:
        D: Bond dimension.
        d: Physical dimension (3 for spin-1).
        key: JAX PRNG key.

    Returns:
        (site_tensors, simplex_tensors, lambdas)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    keys = jax.random.split(key, 5)

    S_a = jax.random.normal(keys[0], (D, D, d)) * 0.1
    S_b = jax.random.normal(keys[1], (D, D, d)) * 0.1
    S_c = jax.random.normal(keys[2], (D, D, d)) * 0.1

    T_up = jax.random.normal(keys[3], (D, D, D)) * 0.1
    T_down = jax.random.normal(keys[4], (D, D, D)) * 0.1

    lambdas = [jnp.ones(D) for _ in range(6)]

    return [S_a, S_b, S_c], [T_up, T_down], lambdas


# ---------------------------------------------------------------------------
# HOSVD truncation (dimension-agnostic)
# ---------------------------------------------------------------------------


def hosvd_truncate(
    theta: jnp.ndarray, D_max: int, d: int = D_PHYS
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list]:
    """Truncate a contracted 3-site tensor back into PESS form via HOSVD."""
    D_ext_a, D_ext_b, D_ext_c = theta.shape[0], theta.shape[1], theta.shape[2]

    theta_reordered = theta.transpose(0, 3, 1, 4, 2, 5)

    # Site a
    mat_a = theta_reordered.reshape(D_ext_a * d, D_ext_b * d * D_ext_c * d)
    U_a, s_a, _ = jnp.linalg.svd(mat_a, full_matrices=False)
    D_int_a = min(D_max, U_a.shape[1])
    U_a = U_a[:, :D_int_a]

    # Site b
    mat_b = theta_reordered.transpose(2, 3, 0, 1, 4, 5).reshape(
        D_ext_b * d, D_ext_a * d * D_ext_c * d
    )
    U_b, s_b, _ = jnp.linalg.svd(mat_b, full_matrices=False)
    D_int_b = min(D_max, U_b.shape[1])
    U_b = U_b[:, :D_int_b]

    # Site c
    mat_c = theta_reordered.transpose(4, 5, 0, 1, 2, 3).reshape(
        D_ext_c * d, D_ext_a * d * D_ext_b * d
    )
    U_c, s_c, _ = jnp.linalg.svd(mat_c, full_matrices=False)
    D_int_c = min(D_max, U_c.shape[1])
    U_c = U_c[:, :D_int_c]

    # Project onto truncated basis
    theta_3mode = theta_reordered.reshape(D_ext_a * d, D_ext_b * d, D_ext_c * d)
    core = jnp.tensordot(U_a.T, theta_3mode, axes=([1], [0]))
    core = jnp.tensordot(U_b.T, core, axes=([1], [1]))
    core = core.transpose(1, 0, 2)
    core = jnp.tensordot(U_c.T, core, axes=([1], [2]))
    core = core.transpose(1, 2, 0)

    # Extract per-bond singular values, then factor them OUT of ``core`` so
    # the bond spectra live exclusively in the explicit lambdas (avoids
    # double-counting when ``S_a_w = lam_ext * R * lam_int`` re-applies them
    # next SU step). See ``src/tenax/algorithms/pess.py:hosvd_truncate``.
    sig_a = jnp.linalg.svd(core.reshape(D_int_a, D_int_b * D_int_c), compute_uv=False)
    sig_b = jnp.linalg.svd(
        core.transpose(1, 0, 2).reshape(D_int_b, D_int_a * D_int_c),
        compute_uv=False,
    )
    sig_c = jnp.linalg.svd(
        core.transpose(2, 0, 1).reshape(D_int_c, D_int_a * D_int_b),
        compute_uv=False,
    )
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

    S_a = U_a.reshape(D_ext_a, d, D_int_a).transpose(0, 2, 1)
    S_b = U_b.reshape(D_ext_b, d, D_int_b).transpose(0, 2, 1)
    S_c = U_c.reshape(D_ext_c, d, D_int_c).transpose(0, 2, 1)

    return S_a, S_b, S_c, core, [lam_a, lam_b, lam_c]


# ---------------------------------------------------------------------------
# PESS simple update (dimension-agnostic)
# ---------------------------------------------------------------------------


def pess_simple_update_triangle(
    S_a, S_b, S_c, T, lambdas_ext, lambdas_int, gate, D_max
):
    """One simple-update step on a single triangle."""
    d = S_a.shape[2]

    S_a_w = jnp.einsum("i,ijd->ijd", lambdas_ext[0], S_a)
    S_b_w = jnp.einsum("i,ijd->ijd", lambdas_ext[1], S_b)
    S_c_w = jnp.einsum("i,ijd->ijd", lambdas_ext[2], S_c)

    S_a_w = jnp.einsum("ijd,j->ijd", S_a_w, lambdas_int[0])
    S_b_w = jnp.einsum("ijd,j->ijd", S_b_w, lambdas_int[1])
    S_c_w = jnp.einsum("ijd,j->ijd", S_c_w, lambdas_int[2])

    theta = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", S_a_w, S_b_w, S_c_w, T)

    gate_jnp = jnp.array(gate)
    theta = jnp.einsum("xyzdfg,DFGdfg->xyzDFG", theta, gate_jnp)

    S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new = hosvd_truncate(theta, D_max, d)

    lam_a_inv = jnp.where(lambdas_ext[0] > 1e-12, 1.0 / lambdas_ext[0], 0.0)
    lam_b_inv = jnp.where(lambdas_ext[1] > 1e-12, 1.0 / lambdas_ext[1], 0.0)
    lam_c_inv = jnp.where(lambdas_ext[2] > 1e-12, 1.0 / lambdas_ext[2], 0.0)
    S_a_new = jnp.einsum("i,ijd->ijd", lam_a_inv, S_a_new)
    S_b_new = jnp.einsum("i,ijd->ijd", lam_b_inv, S_b_new)
    S_c_new = jnp.einsum("i,ijd->ijd", lam_c_inv, S_c_new)

    return S_a_new, S_b_new, S_c_new, T_new, lambdas_int_new


def pess_simple_update(
    site_tensors, simplex_tensors, lambdas, H_tri, dt, D_max, num_steps, d=D_PHYS
):
    """Full PESS simple update loop alternating up and down triangles."""
    gate = make_trotter_gate_3site(H_tri, dt, d)

    S_a, S_b, S_c = site_tensors
    T_up, T_down = simplex_tensors

    for step in range(num_steps):
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

        # Down-triangle: transpose S_x around the call so axis 0 of the
        # passed-in tensor is the T_u-leg (ext side for the down SU).
        # Without this, ``pess_simple_update_triangle`` would tie T_d to
        # the T_u-leg of S and the down triangle would not actually evolve
        # under SU. See library: ``pess_simple_update_triangle`` in
        # ``src/tenax/algorithms/pess.py`` for the same fix.
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
            norm = sum(float(jnp.linalg.norm(lv)) for lv in lambdas)
            print(f"  SU step {step + 1:4d}/{num_steps}  lambda_norm={norm:.6f}")

    return [S_a, S_b, S_c], [T_up, T_down], lambdas


# ---------------------------------------------------------------------------
# PESS -> iPEPS coarse-graining
# ---------------------------------------------------------------------------


def pess_to_ipeps(site_tensors, simplex_tensors, lambdas, d=D_PHYS):
    """Contract PESS into an effective square-lattice iPEPS super-site tensor."""
    S_a, S_b, S_c = site_tensors
    T_up = simplex_tensors[0]
    D = S_a.shape[0]

    S_a_w = jnp.einsum("ijd,j->ijd", S_a, lambdas[0])
    S_b_w = jnp.einsum("ijd,j->ijd", S_b, lambdas[1])
    S_c_w = jnp.einsum("ijd,j->ijd", S_c, lambdas[2])

    sqrt_lam_a = jnp.sqrt(jnp.maximum(lambdas[3], 1e-14))
    sqrt_lam_b = jnp.sqrt(jnp.maximum(lambdas[4], 1e-14))
    sqrt_lam_c = jnp.sqrt(jnp.maximum(lambdas[5], 1e-14))
    S_a_w = jnp.einsum("i,ijd->ijd", sqrt_lam_a, S_a_w)
    S_b_w = jnp.einsum("i,ijd->ijd", sqrt_lam_b, S_b_w)
    S_c_w = jnp.einsum("i,ijd->ijd", sqrt_lam_c, S_c_w)

    theta = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", S_a_w, S_b_w, S_c_w, T_up)

    d_eff = d**3  # 27 for spin-1
    theta_phys = theta.reshape(D, D, D, d_eff)

    A = jnp.zeros((D, D, D, D, d_eff))
    A = A.at[:, :, :, 0, :].set(theta_phys)

    return A


# ---------------------------------------------------------------------------
# Energy measurement
# ---------------------------------------------------------------------------


def compute_pess_energy(site_tensors, simplex_tensors, lambdas, H_tri, chi, d=D_PHYS):
    """Compute energy per site via PESS -> iPEPS -> CTM.

    Each super-site encodes one up-triangle (3 Kagome sites).
    ``compute_energy_ctm`` returns E_h + E_v from horizontal and vertical
    2-site RDMs.  Using ``H_eff = H_tri ⊗ I`` measures the triangle
    Hamiltonian only on the first super-site of each bond pair, so
    E_h + E_v = 2 * <H_tri>.  This correctly accounts for the 2 triangles
    (up + down) per unit cell.  Dividing by 3 gives the energy per Kagome
    site: E/site = 2<H_tri>/3.
    """
    A = pess_to_ipeps(site_tensors, simplex_tensors, lambdas, d)
    d_eff = A.shape[4]

    # Measure H_tri only on the first super-site of each bond pair
    H_eff = np.kron(H_tri, np.eye(d_eff))
    H_eff_4leg = H_eff.reshape(d_eff, d_eff, d_eff, d_eff)

    config = CTMConfig(chi=chi, max_iter=100)
    env = ctm(A, config)

    E = compute_energy_ctm(A, env, jnp.array(H_eff_4leg), d_eff)
    E_per_site = float(E) / 3.0

    return E_per_site


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_pess(filename, site_tensors, simplex_tensors, lambdas):
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


def load_pess(filename):
    """Load PESS tensors from a .npz file."""
    data = np.load(filename)
    site_tensors = [jnp.array(data[n]) for n in ["S_a", "S_b", "S_c"]]
    simplex_tensors = [jnp.array(data[n]) for n in ["T_up", "T_down"]]
    lambdas = [jnp.array(data[f"lambda_{i}"]) for i in range(6)]
    print(f"Loaded PESS state from {filename}")
    return site_tensors, simplex_tensors, lambdas


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Kagome XXZ (spin-1) ground state via PESS simple update"
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
    args = parser.parse_args()

    print("Kagome XXZ (spin-1) via PESS simple update")
    print(f"  delta={args.delta}, D={args.D}, chi={args.chi}")
    print(f"  SU steps={args.steps}, dt={args.dt}")
    print(f"  Physical dimension d={D_PHYS}")

    # Build Hamiltonian
    H_tri = kagome_triangle_hamiltonian_spin1(args.delta)

    # Verify Hermiticity
    assert np.allclose(H_tri, H_tri.T.conj()), "Hamiltonian is not Hermitian!"
    evals = np.linalg.eigvalsh(H_tri)
    print(f"  H_tri spectrum: min={evals[0]:.4f}, max={evals[-1]:.4f}")

    # Initialize or load tensors
    if args.load:
        site_tensors, simplex_tensors, lambdas = load_pess(args.load)
    else:
        site_tensors, simplex_tensors, lambdas = init_pess(args.D, d=D_PHYS)

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
            d=D_PHYS,
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
        site_tensors, simplex_tensors, lambdas, H_tri, args.chi, d=D_PHYS
    )
    ctm_time = time.perf_counter() - t0
    print(f"  E/site = {energy:.6f}")
    print(f"  CTM time = {ctm_time:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
