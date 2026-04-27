"""Honeycomb CTM observables — 2-vertex bond RDM.

Topology for the bond along honeycomb-edge direction ``α`` (Lukin-Sotnikov
PRB 107, 054424 (2023), Fig. 2(b)). The bond connects sublattice ``A`` and
``B`` via their shared ``e_α`` legs, which are left **open** in the RDM.
The surrounding 6-corner ring closes around the dimer using:

* All 6 corners (``C^A_{0,1,2}`` + ``C^B_{0,1,2}``).
* 4 of 12 column tensors: from each sublattice, **one R column** from the
  ``β = (α+1) % 3`` perpendicular direction, and **one L column** from the
  ``γ = (α+2) % 3`` perpendicular direction.

Going clockwise around the ring: ``C R C C L C R C C L`` — 10 env tensors.
The two ``C C`` direct closures are inter-sublattice contractions parallel
to the bond axis, between corners at the same perpendicular direction:

* ``C^A_β`` ↔ ``C^B_β`` along their α-direction chi legs (top of ring).
* ``C^B_γ`` ↔ ``C^A_γ`` along their α-direction chi legs (bottom of ring).

Used = 10 of 18 env tensors per bond. Symmetric under A↔B + ring flip.
"""

from __future__ import annotations

import jax.numpy as jnp

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_init import _double_layer_honeycomb_open
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.contraction.contractor import contract
from tenax.core import EPS
from tenax.core.tensor import Tensor

__all__ = [
    "_rdm1",
    "_rdm2_bond",
    "compute_honeycomb_energy",
    "compute_honeycomb_triangle_energy",
]


def _rdm2_bond(
    sites: dict[Coord, Tensor],
    envs: dict[Coord, HoneycombCTMEnv],
    *,
    alpha: int,
) -> jnp.ndarray:
    """2-vertex reduced density matrix for the honeycomb bond along direction ``α``.

    Args:
        sites: ``{(0, 0): A, (1, 0): B}`` rank-4 site tensors with labels
            ``(e0, e1, e2, phys)``.
        envs: per-sublattice :class:`HoneycombCTMEnv` dict (converged env).
        alpha: bond direction ``∈ {0, 1, 2}``.

    Returns:
        ``(d², d²)`` complex matrix, Hermitian, trace 1.

    Topology (10 env + 2 site tensors). For α=0::

        C^A_0 - R^A_1 - C^A_1 == C^B_1 - L^B_1 - C^B_0
                                                  |
        L^A_2 - C^A_2 == C^B_2 - R^B_2 ----------+

    where ``==`` is a direct chi-leg contraction parallel to the bond,
    and the bond ``A.e_α ↔ B.e_α`` is left open in both ket and bra.
    """
    beta = (alpha + 1) % 3
    gamma = (alpha + 2) % 3
    e_alpha = f"e{alpha}_d2"

    # 1. Open double-layer for both impurity sites (phys + phys_bra free,
    #    the e_α leg is left open, e_β / e_γ get absorbed by the ring).
    T_A_open = _double_layer_honeycomb_open(sites[(0, 0)])
    T_B_open = _double_layer_honeycomb_open(sites[(1, 0)])

    # 2. Pick the participating env tensors per the topology.
    env_A = envs[(0, 0)]
    env_B = envs[(1, 0)]
    C_A_alpha = getattr(env_A, f"C{alpha}")
    C_A_beta = getattr(env_A, f"C{beta}")
    C_A_gamma = getattr(env_A, f"C{gamma}")
    C_B_alpha = getattr(env_B, f"C{alpha}")
    C_B_beta = getattr(env_B, f"C{beta}")
    C_B_gamma = getattr(env_B, f"C{gamma}")
    R_A_beta = getattr(env_A, f"R{beta}")
    L_A_gamma = getattr(env_A, f"L{gamma}")
    L_B_beta = getattr(env_B, f"L{beta}")
    R_B_gamma = getattr(env_B, f"R{gamma}")

    # 3. Distinguish each tensor's chi/D² labels with sublattice + position
    #    suffixes so contract() picks up only the intended pairings.
    #    Naming: every leg that should be contracted gets a label of the
    #    form ``_<bond>_X`` shared between exactly two tensors.

    # Direct corner-to-corner closures (parallel to α):
    #   pos 3-4: C^A_β.chi_in_β  ↔  C^B_β.chi_in_β  (top)
    #   pos 8-9: C^A_γ.chi_in_γ  ↔  C^B_γ.chi_in_γ  (bottom)
    # Both legs get the same sentinel label so contract() auto-pairs them.
    # The OTHER chi leg of each top/bottom corner connects via a transfer.
    C_A_beta_r = C_A_beta.relabels(
        {f"chi_in_{beta}": "_top_close", f"chi_out_{beta}": "_top_A_chi"}
    )
    C_B_beta_r = C_B_beta.relabels(
        {f"chi_in_{beta}": "_top_close", f"chi_out_{beta}": "_top_B_chi"}
    )

    C_A_gamma_r = C_A_gamma.relabels(
        {f"chi_in_{gamma}": "_bot_close", f"chi_out_{gamma}": "_bot_A_chi"}
    )
    C_B_gamma_r = C_B_gamma.relabels(
        {f"chi_in_{gamma}": "_bot_close", f"chi_out_{gamma}": "_bot_B_chi"}
    )

    # Back corners: C^A_α and C^B_α each connect to two transfers via their
    # chi legs (no involvement in the direct closures since those are at β/γ).
    C_A_alpha_r = C_A_alpha.relabels(
        {f"chi_in_{alpha}": "_back_A_in", f"chi_out_{alpha}": "_back_A_out"}
    )
    C_B_alpha_r = C_B_alpha.relabels(
        {f"chi_in_{alpha}": "_back_B_in", f"chi_out_{alpha}": "_back_B_out"}
    )

    # R^A_β: connects C^A_α (back, "_back_A_out") to C^A_β (top, "_top_A_chi"),
    # absorbing T_A_open's e_β leg via the ring's bulk side.
    R_A_beta_r = R_A_beta.relabels(
        {
            f"chi_in_{beta}": "_back_A_out",
            f"e{beta}_d2": "_RA_bulk",
            f"chi_out_{beta}": "_top_A_chi",
        }
    )
    # L^A_γ: connects C^A_γ (bot, "_bot_A_chi") to C^A_α (back, "_back_A_in"),
    # absorbing T_A_open's e_γ leg.
    L_A_gamma_r = L_A_gamma.relabels(
        {
            f"chi_in_{gamma}": "_bot_A_chi",
            f"e{gamma}_d2": "_LA_bulk",
            f"chi_out_{gamma}": "_back_A_in",
        }
    )
    # L^B_β: connects C^B_β (top, "_top_B_chi") to C^B_α (back, "_back_B_in"),
    # absorbing T_B_open's e_β leg.
    L_B_beta_r = L_B_beta.relabels(
        {
            f"chi_in_{beta}": "_top_B_chi",
            f"e{beta}_d2": "_LB_bulk",
            f"chi_out_{beta}": "_back_B_in",
        }
    )
    # R^B_γ: connects C^B_α (back, "_back_B_out") to C^B_γ (bot, "_bot_B_chi"),
    # absorbing T_B_open's e_γ leg.
    R_B_gamma_r = R_B_gamma.relabels(
        {
            f"chi_in_{gamma}": "_back_B_out",
            f"e{gamma}_d2": "_RB_bulk",
            f"chi_out_{gamma}": "_bot_B_chi",
        }
    )

    # T_A_open: rank-5 (e0_d2, e1_d2, e2_d2, phys, phys_bra). Relabel to
    # match the ring: e_β (R_A side), e_γ (L_A side); leave e_α as the
    # OPEN bond (renamed so it auto-contracts with T_B's e_α).
    T_A_r = T_A_open.relabels(
        {
            f"e{beta}_d2": "_RA_bulk",
            f"e{gamma}_d2": "_LA_bulk",
            e_alpha: "_bond",
            "phys": "_phys_A",
            "phys_bra": "_phys_bra_A",
        }
    )
    T_B_r = T_B_open.relabels(
        {
            f"e{beta}_d2": "_LB_bulk",
            f"e{gamma}_d2": "_RB_bulk",
            e_alpha: "_bond",
            "phys": "_phys_B",
            "phys_bra": "_phys_bra_B",
        }
    )

    # 4. Single big contract — all shared sentinel labels auto-pair.
    rdm_t = contract(
        C_A_alpha_r,
        C_A_beta_r,
        C_A_gamma_r,
        C_B_alpha_r,
        C_B_beta_r,
        C_B_gamma_r,
        R_A_beta_r,
        L_A_gamma_r,
        L_B_beta_r,
        R_B_gamma_r,
        T_A_r,
        T_B_r,
        output_labels=("_phys_A", "_phys_B", "_phys_bra_A", "_phys_bra_B"),
    )

    rdm = rdm_t.todense()  # (d, d, d, d) — ket_A, ket_B, bra_A, bra_B
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat


# ------------------------------------------------------------------ #
# 1-site RDM and triangle-energy helper (Task 10)                      #
# ------------------------------------------------------------------ #


def _rdm1(
    sites: dict[Coord, Tensor],
    envs: dict[Coord, HoneycombCTMEnv],
    *,
    sublattice: Coord,
) -> jnp.ndarray:
    """1-site RDM at the given sublattice.

    Topology (Lukin-Sotnikov PRB 107, 054424 Fig. 2(a)): 3 corners +
    3 column tensors form a closed triangular ring around 1 open
    impurity site::

            C^s_0 -- L^s_0 -- C^s_1 -- L^s_1 -- C^s_2 -- L^s_2 -- (back to C^s_0)
                       \\         |          /
                        +-------- T_open(s) ----------+
                                  (e0, e1, e2 absorbed; phys + phys_bra free)

    The choice of L over R is arbitrary; for the converged env, both
    give the same result (Paper 1 Fig 2(c) consistency check).

    Args:
        sites: ``{(0, 0): A, (1, 0): B}`` rank-4 site tensors.
        envs: per-sublattice :class:`HoneycombCTMEnv` dict.
        sublattice: which sublattice to compute the RDM at.

    Returns:
        ``(d, d)`` complex matrix, Hermitian, trace 1.
    """
    A = sites[sublattice]
    env = envs[sublattice]
    T_open = _double_layer_honeycomb_open(A)

    # Triangle ring bond labels — one per edge of the closed ring.
    # Going clockwise: C0 - L0 - C1 - L1 - C2 - L2 - (back to C0).
    bonds = ["_b_0", "_b_1", "_b_2", "_b_3", "_b_4", "_b_5"]

    C0 = env.C0.relabels({"chi_in_0": bonds[5], "chi_out_0": bonds[0]})
    L0 = env.L0.relabels({"chi_in_0": bonds[0], "e0_d2": "_e0", "chi_out_0": bonds[1]})
    C1 = env.C1.relabels({"chi_in_1": bonds[1], "chi_out_1": bonds[2]})
    L1 = env.L1.relabels({"chi_in_1": bonds[2], "e1_d2": "_e1", "chi_out_1": bonds[3]})
    C2 = env.C2.relabels({"chi_in_2": bonds[3], "chi_out_2": bonds[4]})
    L2 = env.L2.relabels({"chi_in_2": bonds[4], "e2_d2": "_e2", "chi_out_2": bonds[5]})

    T_r = T_open.relabels(
        {
            "e0_d2": "_e0",
            "e1_d2": "_e1",
            "e2_d2": "_e2",
            "phys": "_phys",
            "phys_bra": "_phys_bra",
        }
    )

    rdm_t = contract(C0, L0, C1, L1, C2, L2, T_r, output_labels=("_phys", "_phys_bra"))
    rdm = rdm_t.todense()  # (d, d) — ket, bra
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm


def compute_honeycomb_triangle_energy(
    sites: dict[Coord, Tensor],
    envs: dict[Coord, HoneycombCTMEnv],
    hamiltonian: jnp.ndarray,
) -> jnp.ndarray:
    """Energy = ``Tr(ρ_A · H) + Tr(ρ_B · H)`` where each ρ is a 1-site RDM.

    Designed as the ``energy_fn`` override the kagome iPESS path passes in:
    each "site" of the honeycomb env is a kagome triangle (up or down),
    and ``hamiltonian`` is the intra-triangle 3-spin operator of shape
    ``(d^3, d^3)``. The total kagome energy per unit cell sums the
    up-triangle and down-triangle contributions.

    For non-kagome callers (e.g. a per-site magnetic-field test) the
    function still works as-is — ``hamiltonian`` must just match the
    site's physical dimension ``d``.
    """
    rho_A = _rdm1(sites, envs, sublattice=(0, 0))
    rho_B = _rdm1(sites, envs, sublattice=(1, 0))
    return jnp.trace(rho_A @ hamiltonian) + jnp.trace(rho_B @ hamiltonian)


# ------------------------------------------------------------------ #
# Default energy: 3-edge NN bond sum (Task 11)                         #
# ------------------------------------------------------------------ #


def compute_honeycomb_energy(
    sites: dict[Coord, Tensor],
    envs: dict[Coord, HoneycombCTMEnv],
    hamiltonian: jnp.ndarray,
) -> jnp.ndarray:
    """Energy = ``Σ_{α∈{0,1,2}} Tr(ρ_α · H_bond)`` for the 3 NN edges.

    Default ``energy_fn`` for vanilla honeycomb iPEPS use cases (e.g. the
    spin-1/2 Heisenberg model). ``hamiltonian`` is the same 2-site bond
    operator on every α-direction and must have shape ``(d², d²)``, where
    ``d`` is the site physical dimension. The result is the per-unit-cell
    energy, summing the contributions of all three NN bonds emanating from
    sublattice A.
    """
    return sum(
        jnp.trace(_rdm2_bond(sites, envs, alpha=alpha) @ hamiltonian)
        for alpha in (0, 1, 2)
    )
