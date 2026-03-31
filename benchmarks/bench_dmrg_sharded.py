"""Multi-GPU sharded DMRG benchmark cases.

Compares single-device JIT vs multi-device sharded DMRG at large bond
dimensions (chi=1000-2000).  Each size is benchmarked with both
``accelerator="jit"`` and ``accelerator="sharded"`` so the results can
be compared side-by-side.

Usage::

    python -m benchmarks.run --algorithm dmrg_sharded --size small --trials 1
    python -m benchmarks.run --algorithm dmrg_sharded --size all -o results.json
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import DMRGConfig, dmrg
from tenax.core.mps import FiniteMPS

_SIZES = {
    "small": {"L": 20, "chi": 1000, "sweeps": 2, "lanczos": 20},
    "medium": {"L": 20, "chi": 1500, "sweeps": 2, "lanczos": 20},
    "large": {"L": 40, "chi": 1000, "sweeps": 2, "lanczos": 20},
}


def _build_dense_heisenberg(L):
    """Build a dense Heisenberg MPO via AutoMPO."""
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=False)


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():
        for accel in ("jit", "sharded"):

            def _make_fns(p=p, accel=accel, dtype=dtype):
                def setup():
                    mpo = _build_dense_heisenberg(p["L"])
                    mps = FiniteMPS.random(
                        p["L"],
                        d=2,
                        chi=p["chi"],
                        key=jax.random.PRNGKey(42),
                    )
                    config = DMRGConfig(
                        max_bond_dim=p["chi"],
                        num_sweeps=p["sweeps"],
                        lanczos_max_iter=p["lanczos"],
                        convergence_tol=0.0,
                        accelerator=accel,
                        verbose=False,
                    )
                    return (mpo, mps, config)

                def run(mpo, mps, config):
                    return dmrg(mpo, mps, config)

                return setup, run

            setup_fn, run_fn = _make_fns()
            benchmarks.append(
                {
                    "name": "dmrg_sharded",
                    "size_label": f"{size_label}_{accel}",
                    "params": {
                        **p,
                        "accelerator": accel,
                        "num_devices": len(jax.devices()),
                    },
                    "setup_fn": setup_fn,
                    "run_fn": run_fn,
                }
            )
    return benchmarks
