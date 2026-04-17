"""Coarse-grained iPEPS gate construction for non-square lattices.

Maps non-square lattices (honeycomb, kagome, ...) onto the 1-site
square-lattice iPEPS pipeline by grouping physical sites into a single
coarse-grained tensor with effective physical dimension d_eff.
"""

from __future__ import annotations

__all__ = ["CGGates", "honeycomb_cg_gates", "kagome_cg_gates", "compute_energy_cg"]

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor,
    _rdm2x1_tensor,
    _rdm_1site_tensor,
)
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.core.tensor import Tensor


@dataclass(frozen=True)
class CGGates:
    """Coarse-grained Hamiltonian gates and parameterization.

    Attributes:
        h_intra:  Intra-cell interaction, shape ``(d_eff, d_eff)``
                  where ``d_eff = d ** n_sites``.
        h_inter:  Inter-cell interactions, a dict mapping direction labels
                  (e.g. ``"h"``, ``"v"``) to rank-4 tensors of shape
                  ``(d_eff, d_eff, d_eff, d_eff)``.
        n_sites:  Number of physical sites per coarse-grained tensor.
        map_fn:   Callable that contracts raw site tensors into a CG tensor.
                  Signature: ``map_fn(*params) -> jnp.ndarray`` of shape
                  ``(D, D, D, D, d_eff)``.  When ``None``, the optimizer
                  treats the CG tensor itself as the variational parameter.
        init_fn:  Callable that creates initial site tensors.
                  Signature: ``init_fn(D, key) -> tuple[jnp.ndarray, ...]``.
                  When ``None``, a random CG tensor is used directly.
    """

    h_intra: jnp.ndarray
    h_inter: dict[str, jnp.ndarray]
    n_sites: int
    map_fn: Callable | None = None
    init_fn: Callable | None = None


def _ss_2site(dtype=jnp.float64) -> jnp.ndarray:
    """Return the (4,4) Heisenberg S*S matrix for two spin-1/2 sites.

    Uses the identity  S_1 . S_2 = Sz Sz + 0.5 (S+ S- + S- S+).
    Basis ordering: |00>, |01>, |10>, |11>  (computational / product basis).
    """
    h = jnp.zeros((4, 4), dtype=dtype)
    # Sz Sz: diagonal in product basis
    # |00>: +1/4,  |01>: -1/4,  |10>: -1/4,  |11>: +1/4
    h = h.at[0, 0].set(0.25)
    h = h.at[1, 1].set(-0.25)
    h = h.at[2, 2].set(-0.25)
    h = h.at[3, 3].set(0.25)
    # 0.5 * (S+ S- + S- S+): off-diagonal swap terms
    # S+_1 S-_2: |10> -> |01>, so <01|S+S-|10> = 1  -> factor 0.5
    # S-_1 S+_2: |01> -> |10>, so <10|S-S+|01> = 1  -> factor 0.5
    h = h.at[1, 2].set(0.5)
    h = h.at[2, 1].set(0.5)
    return h


def honeycomb_cg_gates(J: float = 1.0, dtype=jnp.float64) -> CGGates:
    """Build coarse-grained Hamiltonian gates for the honeycomb lattice.

    The honeycomb lattice has 2 sub-sites (a, b) per coarse-grained tensor,
    connected by the x-link.  Physical index ordering: |phys_a, phys_b>.
    d_eff = 4 (two spin-1/2 sites).

    Inter-cell links:
        - y-link (vertical, ``"v"``): connects phys_b of CG tensor 1 to
          phys_a of CG tensor 2.  Operator: I_a1 x H_{b1,a2} x I_b2.
        - z-link (horizontal, ``"h"``): connects phys_a of CG tensor 1 to
          phys_b of CG tensor 2.  Operator acts on indices (0, 3).

    Args:
        J:      Coupling constant (default 1.0).
        dtype:  Data type for the arrays.

    Returns:
        A :class:`CGGates` instance.
    """
    ss = _ss_2site(dtype)  # (4, 4) = (d_a d_b, d_a d_b)

    # --- intra-cell: x-link, S_a . S_b on the product space ---
    h_intra = J * ss

    # --- inter-cell gates as rank-4 tensors (d_eff, d_eff, d_eff, d_eff) ---
    # Each CG tensor has basis |a, b> with d=2 each, so d_eff=4.
    # Gate legs: (site1_out, site2_out, site1_in, site2_in) where
    # site_k = (a_k, b_k), then reshaped to d_eff.

    d = 2  # single-site dimension
    eye = jnp.eye(d, dtype=dtype)
    ss_2x2 = ss.reshape(d, d, d, d)  # H[i,j,k,l] acting on two d=2 sites

    # y-link ("v"): S_{b1} . S_{a2}  --  I_{a1} x H_{b1,a2} x I_{b2}
    # 8-leg tensor: h_v[a1, b1, a2, b2, a1', b1', a2', b2']
    #   = eye[a1,a1'] * ss_2x2[b1,a2,b1',a2'] * eye[b2,b2']
    h_v_8leg = jnp.einsum("ae,bcfg,dh->abcdefgh", eye, ss_2x2, eye)
    h_v_gate = J * h_v_8leg.reshape(4, 4, 4, 4)

    # z-link ("h"): S_{a1} . S_{b2}  --  H_{a1,b2} x I_{b1} x I_{a2}
    # 8-leg tensor: h_h[a1, b1, a2, b2, a1', b1', a2', b2']
    #   = ss_2x2[a1,b2,a1',b2'] * eye[b1,b1'] * eye[a2,a2']
    h_h_8leg = jnp.einsum("adeh,bf,cg->abcdefgh", ss_2x2, eye, eye)
    h_h_gate = J * h_h_8leg.reshape(4, 4, 4, 4)

    h_inter = {"v": h_v_gate, "h": h_h_gate}

    def _honeycomb_map(t1: jnp.ndarray, t2: jnp.ndarray) -> jnp.ndarray:
        """Contract two honeycomb site tensors into a CG square-lattice tensor.

        t1: (D, D, d, D_x) — legs [left, down, phys_a, x_right]
        t2: (D_x, d, D, D) — legs [x_left, phys_b, right, up]
        Result: (D, D, d*d, D, D) reshaped to (left, down, d_eff, right, up)
                then transposed to (up, down, left, right, d_eff).
        """
        # Contract over x-link: t1 axis 3 with t2 axis 0
        cg = jnp.tensordot(t1, t2, ((3,), (0,)))
        # cg shape: (D_l, D_d, d_a, d_b, D_r, D_u) = (l, d, sa, sb, r, u)
        D_l, D_d, d_a, d_b, D_r, D_u = cg.shape
        # Fuse physical dims, reorder to (u, d, l, r, phys)
        cg = cg.reshape(D_l, D_d, d_a * d_b, D_r, D_u)
        return cg.transpose(4, 1, 0, 3, 2)  # (u, d, l, r, phys)

    def _honeycomb_init(D: int, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Random initialization for two honeycomb site tensors."""
        k1, k2 = jax.random.split(key)
        t1 = jax.random.normal(k1, (D, D, d, D))  # (l, d, phys_a, x)
        t2 = jax.random.normal(k2, (D, d, D, D))  # (x, phys_b, r, u)
        return (t1, t2)

    return CGGates(
        h_intra=h_intra,
        h_inter=h_inter,
        n_sites=2,
        map_fn=_honeycomb_map,
        init_fn=_honeycomb_init,
    )


def _embed_inter(site_1: int, site_2: int, d: int = 2, n_sites: int = 3):
    """Embed S*S between sub-site *site_1* of CG tensor 1 and *site_2* of CG tensor 2.

    Returns a rank-4 tensor of shape ``(d_eff, d_eff, d_eff, d_eff)`` where
    ``d_eff = d ** n_sites``.  Legs are ``(cg1_out, cg2_out, cg1_in, cg2_in)``.
    """
    import numpy as np

    d_eff = d**n_sites
    Sz = np.array([[0.5, 0], [0, -0.5]])
    Sp = np.array([[0, 1.0], [0, 0]])
    Sm = np.array([[0, 0], [1.0, 0]])
    eye = np.eye(d)
    result = np.zeros((d_eff, d_eff, d_eff, d_eff))
    for Sa, Sb in [(Sz, Sz), (0.5 * Sp, Sm), (0.5 * Sm, Sp)]:
        # Build operator for CG tensor 1: identity on all sub-sites except site_1
        op1 = [eye.copy() for _ in range(n_sites)]
        op1[site_1] = Sa
        kron1 = op1[0]
        for o in op1[1:]:
            kron1 = np.kron(kron1, o)
        # Build operator for CG tensor 2: identity on all sub-sites except site_2
        op2 = [eye.copy() for _ in range(n_sites)]
        op2[site_2] = Sb
        kron2 = op2[0]
        for o in op2[1:]:
            kron2 = np.kron(kron2, o)
        # Outer product: result[i,j,k,l] += kron1[i,k] * kron2[j,l]
        result += np.einsum("ik,jl->ijkl", kron1, kron2)
    return result


def kagome_cg_gates(J: float = 1.0, dtype=jnp.float64) -> CGGates:
    """Build coarse-grained Hamiltonian gates for the kagome lattice.

    Three kagome sub-sites (u, v, w) in an up-triangle fuse into one
    coarse-grained tensor with ``d_eff = 2**3 = 8``.  Physical index
    ordering: ``|phys_u, phys_v, phys_w>``.

    Intra-cell interaction:
        ``H_tri = S_u . S_v + S_u . S_w + S_v . S_w``

    Inter-cell links (between neighbouring CG tensors):
        - horizontal (``"h"``): w of left CG <-> v of right CG
        - vertical (``"v"``): v of top CG <-> u of bottom CG
        - diagonal (``"diag"``): w of top CG <-> u of bottom-right CG

    Args:
        J:      Coupling constant (default 1.0).
        dtype:  Data type for the arrays.

    Returns:
        A :class:`CGGates` instance with ``n_sites=3``.
    """
    import numpy as np

    d = 2
    eye = np.eye(d)
    ss = np.array(_ss_2site(dtype))  # (4, 4)
    ss4 = ss.reshape(d, d, d, d)  # [s1_ket, s2_ket, s1_bra, s2_bra]

    # --- intra-cell: 3 pair interactions on the up-triangle ---
    # 6-index tensor layout: [u_ket, v_ket, w_ket, u_bra, v_bra, w_bra]

    # H_uv: ss4 acts on (u, v), identity on w
    # Result[u, v, w, u', v', w'] = ss4[u, v, u', v'] * eye[w, w']
    h_uv = np.einsum("ijkl,mn->ijmkln", ss4, eye).reshape(8, 8)

    # H_uw: ss4 acts on (u, w), identity on v
    # Result[u, v, w, u', v', w'] = ss4[u, w, u', w'] * eye[v, v']
    h_uw = np.einsum("ijkl,mn->imjknl", ss4, eye).reshape(8, 8)

    # H_vw: ss4 acts on (v, w), identity on u
    # Result[u, v, w, u', v', w'] = eye[u, u'] * ss4[v, w, v', w']
    h_vw = np.einsum("mn,ijkl->mijnkl", eye, ss4).reshape(8, 8)

    h_intra = jnp.array(J * (h_uv + h_uw + h_vw), dtype=dtype)

    # --- inter-cell gates ---
    h_inter = {
        "h": jnp.array(J * _embed_inter(2, 1), dtype=dtype),  # w1 <-> v2
        "v": jnp.array(J * _embed_inter(1, 0), dtype=dtype),  # v1 <-> u2
        "diag": jnp.array(J * _embed_inter(2, 0), dtype=dtype),  # w1 <-> u2
    }
    return CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=3)


# Maps direction keys to the RDM function that produces the corresponding
# 2-site reduced density matrix.
_RDM_DISPATCH: dict[str, object] = {
    "h": _rdm2x1_tensor,
    "v": _rdm1x2_tensor,
}


def compute_energy_cg(
    A: Tensor,
    env: CTMTensorEnv,
    gates: CGGates,
    d_eff: int,
) -> jax.Array:
    """Energy per microscopic site for a coarse-grained 1-site iPEPS.

    Combines the intra-cell (1-site RDM) and inter-cell (2-site RDM) energy
    contributions and divides by the number of physical sites per
    coarse-grained tensor.

    Args:
        A:      The coarse-grained iPEPS tensor (physical dim = *d_eff*).
        env:    Converged CTM environment for *A*.
        gates:  Coarse-grained Hamiltonian gates (:class:`CGGates`).
        d_eff:  Effective physical dimension (``d ** n_sites``).

    Returns:
        Scalar real energy per microscopic site.
    """
    # --- intra-cell energy from 1-site RDM ---
    rdm_1 = _rdm_1site_tensor(A, env)  # (d_eff, d_eff)
    e_intra = jnp.einsum("ij,ij->", rdm_1, gates.h_intra)

    # --- inter-cell energies from 2-site RDMs ---
    e_inter = jnp.zeros((), dtype=rdm_1.dtype)
    for direction, gate in gates.h_inter.items():
        if direction in _RDM_DISPATCH:
            rdm_fn = _RDM_DISPATCH[direction]
        elif direction == "diag":
            # Diagonal RDM will be implemented in Task 6; lazy import to
            # avoid a hard dependency until then.
            from tenax.algorithms._ctm_tensor_energy import _rdm_diagonal_tensor

            rdm_fn = _rdm_diagonal_tensor
        else:
            raise ValueError(f"Unknown inter-cell direction: {direction!r}")
        rdm = rdm_fn(A, env)  # (d_eff, d_eff, d_eff, d_eff)
        e_inter = e_inter + jnp.einsum("ijkl,ijkl->", rdm, gate)

    return ((e_intra + e_inter) / gates.n_sites).real
