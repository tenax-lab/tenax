"""Test-only dense iPEPS Heisenberg CTM probe for GSPMD sharding parity.

A small dense iPEPS Heisenberg energy probe used solely by the sharding tests.
Composed from the same internal CTM builders the production large-D dense path
uses; imported by ``tests/_ctm_sharding_parity_subproc.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def heisenberg_dense_probe_energy(
    *,
    D: int,
    chi: int,
    device_mesh=None,
    seed: int = 0,
    max_iter: int = 20,
    well_conditioned: bool = False,
) -> float:
    """Tiny dense iPEPS Heisenberg energy via one CTM convergence (test probe).

    Builds a deterministic D-bond 1-site iPEPS ``A``, converges the dense CTM
    (optionally on ``device_mesh``), returns the NN Heisenberg energy. Used to
    assert the GSPMD-sharded forward CTM matches the single-device result.

    Runs the 1-site cell through the default ``recipe="2x2"`` dispatch (the same
    path the production large-D CTM uses), with a fixed seed and small
    ``max_iter`` for a fast, deterministic probe.

    ``well_conditioned``: when True, build a near-product-state tensor (one
    dominant bond component) so the CTM fixed point has a well-separated leading
    eigenvalue (low condition number). Sharding the contraction reassociates the
    D²-axis sum, so sharded-vs-single agree only to O(eps·κ); on a random
    (ill-conditioned) state at larger χ that gap reaches ~1e-4, but on a
    well-conditioned state it stays at machine precision (~1e-17). The flag lets
    a parity test assert tight agreement in the physically-relevant regime.
    """
    d = 2
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    if well_conditioned:
        # Dominant single-bond component → near product state → well-separated
        # leading CTM eigenvalue (low κ), so reassociation noise stays ~1e-17.
        data = 0.02 * data
        data = data.at[0, 0, 0, 0, :].add(1.0)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    A = DenseTensor(data, indices)

    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

    envs, _ = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        chi=chi,
        max_iter=max_iter,
        conv_tol=1e-12,
        plateau_patience=None,
        device_mesh=device_mesh,
    )
    energy = compute_energy_ctm_tensor(A, envs[(0, 0)], gate, d)
    return float(energy)
