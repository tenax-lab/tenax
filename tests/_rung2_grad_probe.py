"""Rung-2 gate probe: value_and_grad of the dense CTM-AD energy, single vs mesh.

Self-contained (does NOT depend on the rung-1 probe helpers): builds a 1-site
dense iPEPS, computes ``jax.value_and_grad`` of the implicit-AD CTM energy
(``ctm_energy_implicit``) optionally on a ``device_mesh``, and returns the energy
plus the flat site-tensor gradient. Used by the rung-2 backward-sharding gate to
(a) assert gradient parity single-vs-sharded and (b) measure backward per-device
memory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

_D_PHYS = 2


def _indices(D: int):
    sym = U1Symmetry()
    bc = np.zeros(D, dtype=np.int32)
    pc = np.zeros(_D_PHYS, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )


def _init_data(D: int, seed: int, well_conditioned: bool):
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, _D_PHYS))
    if well_conditioned:
        # Dominant single-bond component → near product state → well-separated
        # leading CTM eigenvalue (low κ) so the sharded-vs-single gradient gap
        # stays at machine precision (only FP reassociation differs).
        data = 0.02 * data
        data = data.at[0, 0, 0, 0, :].add(1.0)
    return data / (jnp.linalg.norm(data) + 1e-10)


def implicit_energy_and_grad(
    *,
    D: int,
    chi: int,
    device_mesh=None,
    seed: int = 0,
    well_conditioned: bool = True,
    max_iter: int = 50,
):
    """Return (energy: float, grad: (D,D,D,D,2) array) of the implicit-AD CTM
    energy w.r.t. the 1-site iPEPS tensor, optionally sharded on ``device_mesh``.
    """
    idx = _indices(D)
    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
    data0 = _init_data(D, seed, well_conditioned)

    def loss(data):
        A = DenseTensor(data, idx)
        return ctm_energy_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-10,
            forward_gauge="phase",
            adjoint_method="fixed_point",
            recipe="2x2",
            device_mesh=device_mesh,
        )

    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)
