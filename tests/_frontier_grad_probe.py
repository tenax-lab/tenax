"""Frontier benchmark probe: value_and_grad of the CTM-AD energy across paths.

Dispatches the dense (``ctm_energy_implicit``, optionally sharded + chunked) and
split (``ctm_energy_split_implicit``, single-GPU) paths to a common
``jax.value_and_grad`` for the large-D x large-chi multi-GPU frontier study
(phase 1, #632). Reuses the rung-2 probe's 1-site dense iPEPS + gate +
well-conditioned init so per-device peaks are directly comparable to the
shard-reach benchmark.

*recipe* selects the CTM projector on **both** paths and defaults to ``"2x2"``,
matching the library default since #746. The original #672 frontier run pinned
``"1x1"`` on the dense arm, whose corner-pair projector collapses the
environment to a rank-1 corner (#723/#726) -- so its (D, chi) reach numbers
describe how cheaply a chi_eff=1 mean-field boundary scales, not the reach of a
converged CTM. Pass ``recipe="1x1"`` to reproduce that historical run; see #747.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _rung2_grad_probe import _indices, _init_data  # pristine helpers (do not edit)

from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit
from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
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
    recipe="2x2",
):
    """Return (energy: float, grad: (D,D,D,D,2) array) of one value_and_grad step.

    path="dense": ctm_energy_implicit(recipe=..., device_mesh=..., ctm_chunk_size=...)
    path="split": ctm_energy_split_implicit(chi_I=chi_I or chi, recipe=...)  # 1-GPU
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
                recipe=recipe,
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
                recipe=recipe,
            )

    else:
        raise ValueError(f"unknown path {path!r} (expected 'dense' or 'split')")

    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)


def frontier_corner_rank(
    *,
    path,
    D,
    chi,
    chi_I=None,
    seed=0,
    well_conditioned=True,
    max_iter=30,
    recipe="2x2",
):
    """Return (energy_free_rank: int, chi: int) for the *forward* CTM env.

    The collapse gate for the frontier grid. ``value_and_grad`` returns only a
    scalar, so the environment is unreachable from :func:`frontier_energy_and_grad`;
    this runs the matching forward convergence on the same tensor and reports
    ``rank(C1)``. Per #747 comment 3, ``corner_rank`` is the only sound collapse
    detector -- chi-flatness is unsound in both directions. A ``2x2`` run should
    give ``rank == chi``; ``1x1`` gives ``rank == 1``.

    Costs one forward convergence, so gate a few representative cells rather
    than every cell of the grid -- the recipe is a global switch, not per-cell.
    """
    idx = _indices(D)
    A = DenseTensor(_init_data(D, seed, well_conditioned), idx)

    if path == "split":
        env = ctm_split_tensor(
            A,
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-10,
            chi_I=chi_I or chi,
            recipe=recipe,
        )
    elif path == "dense":
        envs, _info = python_loop_ctm_converge(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-10,
            conv_method="sv",
            recipe=recipe,
        )
        env = envs[(0, 0)]
    else:
        raise ValueError(f"unknown path {path!r} (expected 'dense' or 'split')")

    return ctm_corner_rank(env), chi
