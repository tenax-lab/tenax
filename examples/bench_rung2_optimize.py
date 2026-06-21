"""Rung-2 deliverable: end-to-end sharded optimize_gs_ad at large D on N GPUs.

Runs a few AD ground-state steps via CTMConfig.device_mesh and reports the energy
trajectory + per-device peak — demonstrating multi-GPU optimization at a D the
single-GPU value_and_grad cannot fit.

    CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_rung2_optimize.py --D 10 --chi 24 --steps 3 --shard
"""

import argparse
import time

import jax
import jax.numpy as jnp

from tenax.algorithms.ctm_sharding import build_ctm_mesh
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def _heisenberg():
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--shard", action="store_true")
    args = ap.parse_args()

    mesh = build_ctm_mesh() if args.shard else None
    n = jax.device_count() if args.shard else 1
    D = args.D
    # well-conditioned (near-product) init → CTM converges fast, valid gradient.
    key = jax.random.PRNGKey(0)
    A = 0.02 * jax.random.normal(key, (D, D, D, D, 2))
    A = A.at[0, 0, 0, 0, :].add(1.0)
    A = A / (jnp.linalg.norm(A) + 1e-10)
    H = _heisenberg()
    print(
        f"# optimize_gs_ad devices={jax.device_count()} shard={args.shard} "
        f"mesh_n={n} D={D} chi={args.chi} steps={args.steps}"
    )

    def cfg(nsteps):
        return iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(
                chi=args.chi, max_iter=40, conv_tol=1e-8,
                plateau_patience=None, device_mesh=mesh,
            ),
            gs_num_steps=nsteps,
            gs_learning_rate=1e-2,
            su_init=False,
            gs_metric_precond=False,
            gs_line_search=False,
        )

    t0 = time.perf_counter()
    try:
        _, _, E0 = optimize_gs_ad(H, A, cfg(0))
        _, _, Ef = optimize_gs_ad(H, A, cfg(args.steps))
        dt = time.perf_counter() - t0
        print(
            f"D={D}: OK  E_init={float(E0):.6f} -> E_final={float(Ef):.6f}  "
            f"dE={float(Ef) - float(E0):+.3e}  per_device_peak={_peak_gb():.2f} GB  "
            f"wall={dt:.1f}s"
        )
    except Exception as ex:  # noqa: BLE001
        dt = time.perf_counter() - t0
        print(f"D={D}: FAILED ({type(ex).__name__}: {str(ex)[:100]})  wall={dt:.1f}s")


if __name__ == "__main__":
    main()
