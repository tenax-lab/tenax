"""F3 microbench: cold compile + 1 backward, warm 2 backwards.

Run twice — once on F2 (baseline), once on F3 (this branch). Report
JSON to stdout for easy diff.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp

# Force CPU complex128 for apples-to-apples with variPEPS reference.
os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_enable_x64", True)

# The tenax imports below MUST stay below the JAX env-var setup — moving
# them to the top of the file would let JAX initialize before
# ``JAX_PLATFORMS`` takes effect, silently flipping the bench to GPU.
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor  # noqa: E402


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def main() -> None:
    H = _heisenberg_gate()
    key = jax.random.PRNGKey(2026)
    A = jax.random.normal(key, (2, 2, 2, 2, 2), dtype=jnp.complex128)
    A = _wrap_as_dense_tensor(A / jnp.linalg.norm(A))

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=8,
            max_iter=40,
            conv_tol=1e-6,
            forward_gauge="phase",
            gmres_tol=1e-6,
            gmres_maxiter=200,
            arnoldi_precheck=False,
            adjoint_method="fixed_point",
        )

    grad_fn = jax.grad(loss)

    t0 = time.perf_counter()
    g0 = grad_fn(A)
    jax.block_until_ready(g0)
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(2):
        t0 = time.perf_counter()
        g = grad_fn(A)
        jax.block_until_ready(g)
        warm.append(time.perf_counter() - t0)

    print(json.dumps({"cold_s": cold, "warm_s": warm}, indent=2))


if __name__ == "__main__":
    main()
