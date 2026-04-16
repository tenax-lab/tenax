"""Benchmark CG iPEPS against variPEPS lecture notes (SciPost 86).

Usage:
    python -m benchmarks.bench_cg_varipeps --lattice honeycomb --D 2 --chi 16
    python -m benchmarks.bench_cg_varipeps --lattice kagome --D 2 --chi 20
"""

import argparse
import time

import jax.numpy as jnp

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
from tenax.algorithms.coarse_grain import honeycomb_cg_gates, kagome_cg_gates

VARIPEPS_HONEYCOMB_VU = {
    2: -0.5376,
    3: -0.5411,
    4: -0.5442,
    5: -0.5445,
    6: -0.5445,
    7: -0.5445,
}
VARIPEPS_KAGOME_VU = {
    2: -0.4045,
    3: -0.4269,
    4: -0.4304,
    5: -0.4329,
    6: -0.4345,
    7: -0.4353,
    8: -0.4355,
}


def run(lattice: str, D: int, chi: int, steps: int) -> None:
    if lattice == "honeycomb":
        gates = honeycomb_cg_gates()
        ref = VARIPEPS_HONEYCOMB_VU
    elif lattice == "kagome":
        gates = kagome_cg_gates()
        ref = VARIPEPS_KAGOME_VU
    else:
        raise ValueError(f"Unknown lattice: {lattice}")

    d_eff = 2**gates.n_sites
    dummy_gate = jnp.zeros((d_eff,) * 4)

    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(chi=chi, max_iter=50, min_iter=10),
        gs_num_steps=steps,
        gs_learning_rate=1e-2,
        gs_optimizer="lbfgs",
        gs_line_search=True,
        gs_line_search_method="armijo",
        gs_c4v=True,
        gs_explicit_ad=True,
        gs_explicit_ad_steps=15,
        gs_explicit_ad_warmup=3,
        gs_metric_precond=True,
        gs_verbose=True,
        su_init=False,
        cg_gates=gates,
    )

    t0 = time.time()
    _, _, E_gs = optimize_gs_ad(dummy_gate, None, config)
    elapsed = time.time() - t0

    ref_E = ref.get(D)
    print(f"\n{'=' * 50}")
    print(f"Lattice: {lattice}, D={D}, chi={chi}")
    print(f"Tenax E/site:    {E_gs:.6f}")
    if ref_E is not None:
        print(f"variPEPS E/site: {ref_E:.6f}")
        print(f"Difference:      {E_gs - ref_E:+.6f}")
    print(f"Wall time:       {elapsed:.1f}s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark CG iPEPS against variPEPS (SciPost 86)"
    )
    parser.add_argument(
        "--lattice", default="honeycomb", choices=["honeycomb", "kagome"]
    )
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run(args.lattice, args.D, args.chi, args.steps)
