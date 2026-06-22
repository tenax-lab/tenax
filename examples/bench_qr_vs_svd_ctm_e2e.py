"""End-to-end: full forward CTM with SVD projector (recipe=2x2) vs reduced-corner
QR projector (recipe=1x1) at large χ. Sizes the REAL speedup (projector is only a
fraction of the step), not just the isolated decomposition.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_qr_vs_svd_ctm_e2e.py --D 4 --chi 96 --method svd
    method svd → (recipe=2x2, projector_method=svd); qr → (recipe=1x1, qr).
One method×config per process (clean peak); reports warm per-sweep time.
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402


def _make_A(D, seed=0):
    d = 2
    data = 0.05 * jax.random.normal(jax.random.PRNGKey(seed), (D, D, D, D, d))
    data = data.at[0, 0, 0, 0, :].add(1.0)  # near-product → stable CTM
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    bc = np.zeros(D, dtype=np.int32)
    pc = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


def _peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def _run(A, chi, recipe, proj, max_iter):
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A}, SINGLE_SITE_NEIGHBORS, chi=chi, max_iter=max_iter,
        conv_tol=1e-14, plateau_patience=None, recipe=recipe,
        projector_method=proj, qr_warmup_steps=0,
    )
    return envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--method", choices=["svd2x2", "svd1x1", "qr1x1"], required=True)
    ap.add_argument("--max-iter", type=int, default=8)
    args = ap.parse_args()
    # svd2x2 = current default; svd1x1 vs qr1x1 = clean projector isolation
    # (same 1-site scheme, only the projector differs).
    recipe, proj = {
        "svd2x2": ("2x2", "svd"),
        "svd1x1": ("1x1", "svd"),
        "qr1x1": ("1x1", "qr"),
    }[args.method]
    A = _make_A(args.D)
    print(
        f"# e2e CTM  D={args.D} chi={args.chi} method={args.method} "
        f"(recipe={recipe} proj={proj}) max_iter={args.max_iter} x64={jax.config.jax_enable_x64}"
    )
    try:
        jax.block_until_ready(_run(A, args.chi, recipe, proj, args.max_iter)[(0, 0)])  # compile
        t0 = time.perf_counter()
        envs = _run(A, args.chi, recipe, proj, args.max_iter)
        jax.block_until_ready(envs[(0, 0)])
        dt = time.perf_counter() - t0
        print(
            f"D={args.D} chi={args.chi} method={args.method}  "
            f"per_sweep={dt / args.max_iter * 1e3:.1f} ms  warm_total={dt:.3f}s  "
            f"peak={_peak_gb():.2f} GB"
        )
    except Exception as ex:  # noqa: BLE001
        print(f"D={args.D} chi={args.chi} method={args.method}  FAILED({type(ex).__name__}: {str(ex)[:70]})")


if __name__ == "__main__":
    main()
