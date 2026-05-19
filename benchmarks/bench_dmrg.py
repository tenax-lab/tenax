"""DMRG benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import (
    DMRGConfig,
    build_mpo_heisenberg,
    build_random_symmetric_mps,
    dmrg,
)

_SIZES = {
    "small": {"L": 20, "chi": 32, "sweeps": 5, "init_bond_dim": 8},
    "medium": {"L": 40, "chi": 64, "sweeps": 5, "init_bond_dim": 16},
    "large": {"L": 80, "chi": 128, "sweeps": 3, "init_bond_dim": 16},
}


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p, dtype=dtype):
            def setup():
                mpo = build_mpo_heisenberg(p["L"], dtype=dtype)
                # build_mpo_heisenberg returns a symmetric (U(1)) MPO, so the
                # initial MPS must also be symmetric — dmrg() rejects mixed types.
                mps = build_random_symmetric_mps(
                    p["L"], bond_dim=p["init_bond_dim"], dtype=dtype
                )
                config = DMRGConfig(
                    max_bond_dim=p["chi"],
                    num_sweeps=p["sweeps"],
                    convergence_tol=1e-10,
                    two_site=True,
                    verbose=False,
                )
                return (mpo, mps, config)

            def run(mpo, mps, config):
                return dmrg(mpo, mps, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "dmrg",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
