"""Frontier benchmark probe: value_and_grad of the CTM-AD energy across paths.

Dispatches the dense (``ctm_energy_implicit``, recipe="1x1", optionally sharded
+ chunked) and split (``ctm_energy_split_implicit``, single-GPU, chi^2 * D^4)
paths to a common ``jax.value_and_grad`` for the large-D x large-chi multi-GPU
frontier study (phase 1, #632). Reuses the rung-2 probe's 1-site dense iPEPS +
gate + well-conditioned init so per-device peaks are directly comparable to the
shard-reach benchmark.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _rung2_grad_probe import _indices, _init_data  # pristine helpers (do not edit)

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit
from tenax.core.tensor import DenseTensor

_HEISENBERG = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)


def frontier_energy_and_grad(
    *,
    path,
    D,
    chi,
    chi_I=None,
    device_mesh=None,
    ctm_chunk_size=None,
    seed=0,
    well_conditioned=True,
    max_iter=30,
):
    """Return (energy: float, grad: (D,D,D,D,2) array) of one value_and_grad step.

    path="dense": ctm_energy_implicit(recipe="1x1", device_mesh=..., ctm_chunk_size=...)
    path="split": ctm_energy_split_implicit(chi_I=chi_I or chi)  # single-GPU only
    """
    idx = _indices(D)
    data0 = _init_data(D, seed, well_conditioned)

    if path == "dense":

        def loss(data):
            A = DenseTensor(data, idx)
            return ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _HEISENBERG,
                chi=chi,
                max_iter=max_iter,
                conv_tol=1e-10,
                forward_gauge="phase",
                adjoint_method="fixed_point",
                recipe="1x1",
                device_mesh=device_mesh,
                ctm_chunk_size=ctm_chunk_size,
            )

    elif path == "split":
        if device_mesh is not None:
            raise ValueError("split path is single-GPU only (no device_mesh)")
        if ctm_chunk_size is not None:
            raise ValueError("split path does not support ctm_chunk_size")

        def loss(data):
            A = DenseTensor(data, idx)
            return ctm_energy_split_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _HEISENBERG,
                chi=chi,
                chi_I=chi_I or chi,
                max_iter=max_iter,
                conv_tol=1e-10,
            )

    else:
        raise ValueError(f"unknown path {path!r} (expected 'dense' or 'split')")

    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)
