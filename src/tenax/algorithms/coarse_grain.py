"""Coarse-grained iPEPS gate construction for non-square lattices.

Maps non-square lattices (honeycomb, kagome, ...) onto the 1-site
square-lattice iPEPS pipeline by grouping physical sites into a single
coarse-grained tensor with effective physical dimension d_eff.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class CGGates:
    """Coarse-grained Hamiltonian gates.

    Attributes:
        h_intra:  Intra-cell interaction, shape ``(d_eff, d_eff)``
                  where ``d_eff = d ** n_sites``.
        h_inter:  Inter-cell interactions, a dict mapping direction labels
                  (e.g. ``"h"``, ``"v"``) to rank-4 tensors of shape
                  ``(d_eff, d_eff, d_eff, d_eff)``.
        n_sites:  Number of physical sites per coarse-grained tensor.
    """

    h_intra: jnp.ndarray
    h_inter: dict[str, jnp.ndarray]
    n_sites: int


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
    h_v_8leg = jnp.einsum("ae,bfcg,dh->abcdefgh", eye, ss_2x2, eye)
    h_v_gate = J * h_v_8leg.reshape(4, 4, 4, 4)

    # z-link ("h"): S_{a1} . S_{b2}  --  H_{a1,b2} x I_{b1} x I_{a2}
    # 8-leg tensor: h_h[a1, b1, a2, b2, a1', b1', a2', b2']
    #   = ss_2x2[a1,b2,a1',b2'] * eye[b1,b1'] * eye[a2,a2']
    h_h_8leg = jnp.einsum("adeh,bf,cg->abcdefgh", ss_2x2, eye, eye)
    h_h_gate = J * h_h_8leg.reshape(4, 4, 4, 4)

    h_inter = {"v": h_v_gate, "h": h_h_gate}
    return CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=2)
