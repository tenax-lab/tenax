"""Per-bond gates and energy wrapper for the kagome 3-site multisite iPESS.

Companion to :func:`tenax.algorithms.pess.pess_to_kagome_3site_multisite`.
The 3-site multisite encoding maps each kagome sublattice onto one ``d=2``
(or ``d=3``) site of a 3-site multisite iPEPS keyed by ``{"u", "v", "w"}``;
all 6 kagome bonds become NN bonds on the kagome
:func:`tenax.core.lattice.kagome` Lattice (3 NN-h up-triangle bonds plus 3
NN-v down-triangle bonds, no diagonal RDM). See
``docs/plans/2026-05-05-multisite-kagome-pess.md`` "ACTIVE PLAN" Phase B.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_tensor_energy import (
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
)
from tenax.algorithms.pess import _site_ops
from tenax.core.tensor import Tensor

__all__ = [
    "kagome_xxz_pair_hamiltonian",
    "kagome_3site_bond_gates",
    "compute_energy_pess_3site_multisite",
]


def kagome_xxz_pair_hamiltonian(delta: float = 1.0, d: int = 2) -> np.ndarray:
    """Single XXZ pair Hamiltonian on two ``d``-dimensional sites.

    ``H = δ Sz Sz + 0.5 (S+ S- + S- S+)``. Returned as ``(d, d, d, d)`` array
    with index layout ``(site1_out, site2_out, site1_in, site2_in)`` matching
    the gate convention consumed by ``_rdm{2x1,1x2}_tensor_2site``'s
    ``einsum("ijkl,ijkl->", rdm, H)`` contraction.
    """
    ops = _site_ops(d)
    Sz = ops["Sz"]
    Sp = ops["Sp"]
    Sm = ops["Sm"]
    H = delta * np.einsum("ik,jl->ijkl", Sz, Sz)
    H = H + 0.5 * np.einsum("ik,jl->ijkl", Sp, Sm)
    H = H + 0.5 * np.einsum("ik,jl->ijkl", Sm, Sp)
    return H.astype(np.complex128)


def kagome_3site_bond_gates(
    delta: float = 1.0, d: int = 2
) -> dict[frozenset, jax.Array]:
    """Per-bond XXZ gates for the 3-site multisite kagome iPESS.

    Returns 6 entries — one per kagome bond — keyed by
    ``frozenset({(name_a, dir_a), (name_b, dir_b)})`` so the order in which
    the energy iterator emits a bond does not matter (both
    ``(coord, direction)`` and ``(neighbor, reverse_direction)`` hash to the
    same bond ID).  Direction labels match
    :func:`tenax.core.lattice.kagome` 's neighbor map:

    Up-triangle (NN-h, "right"/"left"):
      - ``u-v``: ``frozenset({("u","right"), ("v","left")})``
      - ``v-w``: ``frozenset({("v","right"), ("w","left")})``
      - ``w-u``: ``frozenset({("w","right"), ("u","left")})``

    Down-triangle (NN-v, "bottom"/"top"):
      - ``u-v``: ``frozenset({("u","bottom"), ("v","top")})``
      - ``v-w``: ``frozenset({("v","bottom"), ("w","top")})``
      - ``w-u``: ``frozenset({("w","bottom"), ("u","top")})``

    All 6 gates are the same XXZ pair Hamiltonian (uniform kagome).
    """
    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta, d))
    return {
        frozenset({("u", "right"), ("v", "left")}): H_pair,
        frozenset({("v", "right"), ("w", "left")}): H_pair,
        frozenset({("w", "right"), ("u", "left")}): H_pair,
        frozenset({("u", "bottom"), ("v", "top")}): H_pair,
        frozenset({("v", "bottom"), ("w", "top")}): H_pair,
        frozenset({("w", "bottom"), ("u", "top")}): H_pair,
    }


def compute_energy_pess_3site_multisite(
    site_tensors: dict,
    envs: dict,
    neighbors: dict,
    bond_gates: dict[frozenset, jax.Array | Tensor],
    d: int | None = None,
) -> jax.Array:
    """Per-microscopic-site energy of the 3-site multisite kagome iPESS.

    Iterates the 6 NN bonds of the kagome unit cell using ``right``/``bottom``
    traversal (each bond visited once via canonical ``frozenset`` bond ID),
    dispatches the appropriate 2-site RDM primitive, contracts against the
    matching per-bond gate from ``bond_gates``, and divides the total by the
    number of sites (3) to return energy per microscopic site.

    Mirrors :func:`tenax.algorithms._ctm_tensor_energy.compute_energy_ctm_tensor_multisite`
    structurally but uses **per-bond** gates instead of one shared ``gate``,
    and the dispatcher only looks up NN bonds (no diagonal RDM at all — the
    3-site multisite encoding routes every kagome bond through NN).

    Args:
        site_tensors: ``{name: Tensor}`` mapping sublattice names
            ``{"u", "v", "w"}`` to iPEPS site tensors (typically obtained
            from :func:`tenax.algorithms.pess.pess_to_kagome_3site_multisite`
            wrapped as :class:`tenax.core.tensor.DenseTensor`).
        envs:         ``{name: CTMTensorEnv}`` converged environments per
            sublattice (from :func:`tenax.algorithms._ctm_tensor_convergence._ctm_tensor_multisite`).
        neighbors:    ``{name: {"left": name, "right": name, "top": name,
                      "bottom": name}}`` — typically
                      ``tenax.core.lattice.kagome().neighbor_map``.
        bond_gates:   Per-bond gate dict from
                      :func:`kagome_3site_bond_gates`.
        d:            Physical dimension (inferred from the first site
                      tensor if ``None``).

    Returns:
        Real scalar — total NN-bond energy divided by ``len(site_tensors)``.

    Raises:
        KeyError: when a NN bond visited via the right/bottom sweep has no
            matching entry in ``bond_gates``.
    """
    if d is None:
        first_A = next(iter(site_tensors.values()))
        phys_idx = [i for i in first_A.indices if i.label == "phys"]
        d = phys_idx[0].dim if phys_idx else first_A.indices[-1].dim

    n_sites = len(site_tensors)
    total_energy = jnp.array(0.0)
    counted_bonds: set = set()

    for name, A in site_tensors.items():
        env_A = envs[name]
        for direction in ("right", "bottom"):
            nb_name = neighbors[name][direction]
            reverse_dir = "left" if direction == "right" else "top"
            bond_id = frozenset([(name, direction), (nb_name, reverse_dir)])
            if bond_id in counted_bonds:
                continue
            counted_bonds.add(bond_id)

            gate = bond_gates[bond_id]
            if isinstance(gate, Tensor):
                H = gate.todense().reshape(d, d, d, d)
            else:
                H = jnp.asarray(gate).reshape(d, d, d, d)

            B = site_tensors[nb_name]
            env_B = envs[nb_name]
            if direction == "right":
                rdm = _rdm2x1_tensor_2site(A, B, env_A, env_B)
            else:
                rdm = _rdm1x2_tensor_2site(A, B, env_A, env_B)

            total_energy = total_energy + jnp.einsum("ijkl,ijkl->", rdm, H)

    return total_energy.real / n_sites
