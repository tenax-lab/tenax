"""Double-layer construction and env initialization for honeycomb CTM."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv
from tenax.algorithms._ctm_honeycomb_topology import Coord
from tenax.algorithms._ctm_tensor_init import _fuse_pair_by_label
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

__all__ = [
    "_double_layer_honeycomb",
    "_double_layer_honeycomb_open",
    "initialize_honeycomb_env",
]

IN = FlowDirection.IN
OUT = FlowDirection.OUT


def _double_layer_honeycomb(A: Tensor) -> Tensor:
    """Build the rank-3 double-layer tensor for a honeycomb site.

    Input:  A with labels ``(e0, e1, e2, phys)``, 4 legs.
    Output: 3-leg tensor with labels ``(e0_d2, e1_d2, e2_d2)``, dimensions D².

    Mirrors ``_ctm_tensor_init._build_double_layer_tensor`` (rank-5 square
    case) but with 3 virtual legs instead of 4.

    Flow convention: each fused leg ``e{α}_d2`` inherits the bra (post-``bar()``)
    direction. Site virtual flows are all ``OUT`` (per Tenax honeycomb convention),
    so all three fused legs are ``IN``. Mirrors the square pattern in
    ``_ctm_tensor_init._build_double_layer_tensor`` where fused-flow = bra-flow.
    """
    A_bra = A.bar().relabels({"e0": "E0", "e1": "E1", "e2": "E2"})
    a6 = contract(A, A_bra)
    result = _fuse_pair_by_label(a6, "e0", "E0", "e0_d2", IN)
    result = _fuse_pair_by_label(result, "e1", "E1", "e1_d2", IN)
    result = _fuse_pair_by_label(result, "e2", "E2", "e2_d2", IN)
    return result


def _double_layer_honeycomb_open(A: Tensor) -> Tensor:
    """Open-physical-leg double-layer for the 2-vertex bond RDM.

    Same as :func:`_double_layer_honeycomb` but with the physical leg of
    the bra layer relabeled to ``phys_bra`` (instead of being contracted
    against the ket). The result is a rank-5 tensor with labels
    ``(e0_d2, e1_d2, e2_d2, phys, phys_bra)``.

    Mirrors :func:`tenax.algorithms._ctm_tensor_init._build_double_layer_open_tensor`
    (rank-6 square case) but with 3 virtual legs instead of 4.
    """
    A_bra = A.bar().relabels({"e0": "E0", "e1": "E1", "e2": "E2", "phys": "phys_bra"})
    a_open = contract(A, A_bra)
    result = _fuse_pair_by_label(a_open, "e0", "E0", "e0_d2", IN)
    result = _fuse_pair_by_label(result, "e1", "E1", "e1_d2", IN)
    result = _fuse_pair_by_label(result, "e2", "E2", "e2_d2", IN)
    return result


def initialize_honeycomb_env(
    sites: dict[Coord, Tensor],
    chi_init: int,
    *,
    seed: int = 0,
) -> dict[Coord, HoneycombCTMEnv]:
    """Random complex128 init at ``chi_init`` for each sublattice's env.

    Each corner is rank-2 ``(chi_init, chi_init)``, label flows ``(IN, OUT)``.
    Each column is rank-3 ``(chi_init, D**2, chi_init)``, label flows
    ``(IN, IN, OUT)`` — the middle ``IN`` matches the double-layer fused-leg
    flow established in :func:`_double_layer_honeycomb`.
    """
    sym = U1Symmetry()
    envs: dict[Coord, HoneycombCTMEnv] = {}
    key = jax.random.PRNGKey(seed)
    for coord, A in sites.items():
        D = A.indices[A.labels().index("e0")].dim
        d2 = D * D
        chi_charges = np.zeros(chi_init, dtype=np.int32)
        d2_charges = np.zeros(d2, dtype=np.int32)

        def chi_idx(flow: FlowDirection, lbl: str) -> TensorIndex:
            return TensorIndex.from_charges(sym, chi_charges.copy(), flow, label=lbl)

        def d2_idx(lbl: str) -> TensorIndex:
            return TensorIndex.from_charges(sym, d2_charges.copy(), IN, label=lbl)

        corners: list[DenseTensor] = []
        lefts: list[DenseTensor] = []
        rights: list[DenseTensor] = []
        for alpha in range(3):
            k_c, k_l, k_r, key = jax.random.split(key, 4)
            c_data = (
                jax.random.normal(k_c, (chi_init, chi_init))
                + 1j
                * jax.random.normal(jax.random.fold_in(k_c, 1), (chi_init, chi_init))
            ).astype(jnp.complex128)
            l_data = (
                jax.random.normal(k_l, (chi_init, d2, chi_init))
                + 1j
                * jax.random.normal(
                    jax.random.fold_in(k_l, 1), (chi_init, d2, chi_init)
                )
            ).astype(jnp.complex128)
            r_data = (
                jax.random.normal(k_r, (chi_init, d2, chi_init))
                + 1j
                * jax.random.normal(
                    jax.random.fold_in(k_r, 1), (chi_init, d2, chi_init)
                )
            ).astype(jnp.complex128)
            corners.append(
                DenseTensor(
                    c_data,
                    (
                        chi_idx(IN, f"chi_in_{alpha}"),
                        chi_idx(OUT, f"chi_out_{alpha}"),
                    ),
                )
            )
            lefts.append(
                DenseTensor(
                    l_data,
                    (
                        chi_idx(IN, f"chi_in_{alpha}"),
                        d2_idx(f"e{alpha}_d2"),
                        chi_idx(OUT, f"chi_out_{alpha}"),
                    ),
                )
            )
            rights.append(
                DenseTensor(
                    r_data,
                    (
                        chi_idx(IN, f"chi_in_{alpha}"),
                        d2_idx(f"e{alpha}_d2"),
                        chi_idx(OUT, f"chi_out_{alpha}"),
                    ),
                )
            )
        envs[coord] = HoneycombCTMEnv(
            C0=corners[0],
            C1=corners[1],
            C2=corners[2],
            L0=lefts[0],
            L1=lefts[1],
            L2=lefts[2],
            R0=rights[0],
            R1=rights[1],
            R2=rights[2],
        )
    return envs
