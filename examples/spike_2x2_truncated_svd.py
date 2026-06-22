"""Option-2 prototype: the 2x2 half-system SVD matrices are rank-χ (measured), so a
randomized top-χ SVD is EXACT and ~100× cheaper than the full χD²×χD² cuSOLVER SVD.

Patches `_gauge_fixed_svd` with a randomized top-k SVD (+gauge fix), runs the
end-to-end 2x2 forward CTM, and checks energy parity vs the full-SVD 2x2 + speed.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/spike_2x2_truncated_svd.py --D 8 --chi 48
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import tenax.algorithms._ctm_python_loop as PL  # noqa: E402
import tenax.algorithms._ctm_tensor_projector_2x2 as M2x2  # noqa: E402
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

_ORIG_SVD = M2x2._gauge_fixed_svd
_K = {"k": 64, "key": jax.random.PRNGKey(0)}


def _gauge_fix(U, s, Vh):
    # Mirror _gauge_fixed_svd's argmax phase fix exactly.
    max_idx = jnp.argmax(jnp.abs(U), axis=0)
    diag = U[max_idx, jnp.arange(U.shape[1])]
    phases = jnp.where(jnp.abs(diag) > 0, diag / jnp.abs(diag), 1.0)
    return U * jnp.conj(phases)[None, :], s, Vh * phases[:, None]


def randomized_topk_svd(M):
    # M is rank ≤ χ; top-k (k=χ+oversample) is exact. Randomized range finder.
    _K["used"] = _K.get("used", 0) + 1  # trace-time marker
    m, n = M.shape
    k = min(_K["k"], n)
    Omega = jax.random.normal(_K["key"], (n, k), dtype=M.dtype)
    Y = M @ Omega
    Q, _ = jnp.linalg.qr(Y)  # (m, k) tall-skinny QR — cheap
    B = Q.conj().T @ M  # (k, n)
    Ub, s, Vh = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return _gauge_fix(U, s, Vh)


def _make_A(D, seed=0):
    data = 0.05 * jax.random.normal(jax.random.PRNGKey(seed), (D, D, D, D, 2))
    data = data.at[0, 0, 0, 0, :].add(1.0)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    bc = np.zeros(D, dtype=np.int32)
    pc = np.zeros(2, dtype=np.int32)
    idx = tuple(
        TensorIndex.from_charges(sym, (bc if lbl != "phys" else pc).copy(), f, label=lbl)
        for lbl, f in [("u", FlowDirection.OUT), ("d", FlowDirection.IN),
                       ("l", FlowDirection.OUT), ("r", FlowDirection.IN),
                       ("phys", FlowDirection.IN)]
    )
    return DenseTensor(data, idx)


def _converge(A, chi, mi):
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A}, SINGLE_SITE_NEIGHBORS, chi=chi, max_iter=mi, conv_tol=0.0,
        plateau_patience=None, recipe="2x2", projector_method="svd", qr_warmup_steps=0,
    )
    return envs


def _energy_time(A, chi, mi):
    jax.block_until_ready(_converge(A, chi, mi)[(0, 0)])  # compile
    t0 = time.perf_counter()
    envs = _converge(A, chi, mi)
    jax.block_until_ready(envs[(0, 0)])
    dt = time.perf_counter() - t0
    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
    E = float(compute_energy_ctm_tensor(A, envs[(0, 0)], gate, 2))
    return E, dt / mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--mi", type=int, default=4)
    args = ap.parse_args()
    _K["k"] = args.chi + 16
    A = _make_A(args.D)
    print(f"# 2x2 truncated-SVD prototype  D={args.D} chi={args.chi} k={_K['k']}")

    # NB: _make_jit_ctm_step is module-cached → must clear it AND patch BEFORE
    # the step is traced, else the cached step keeps the original SVD.
    M2x2._gauge_fixed_svd = _ORIG_SVD
    PL._JIT_STEP_CACHE.clear()
    _K["used"] = 0
    e_full, t_full = _energy_time(A, args.chi, args.mi)
    print(f"  full-SVD 2x2 : E={e_full:.8f}  per_sweep={t_full*1e3:.1f} ms  (patched_calls={_K['used']})")

    M2x2._gauge_fixed_svd = randomized_topk_svd
    PL._JIT_STEP_CACHE.clear()
    _K["used"] = 0
    e_tr, t_tr = _energy_time(A, args.chi, args.mi)
    print(f"  trunc-SVD 2x2: E={e_tr:.8f}  per_sweep={t_tr*1e3:.1f} ms  (patched_calls={_K['used']})")
    print(f"  => |dE|={abs(e_full-e_tr):.2e}  speedup={t_full/t_tr:.1f}x")


if __name__ == "__main__":
    main()
