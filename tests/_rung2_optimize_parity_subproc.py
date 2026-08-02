"""Rung-2 config surface: optimize_gs_ad single-device vs GSPMD-sharded parity.

Runs a few steps of AD ground-state optimization with ``CTMConfig.device_mesh``
unset (single) vs set (sharded), from the same seed, and asserts the final
energy + tensor match. Validates that the config-level mesh surface routes the
whole implicit-AD optimization through sharding with no change in result.

Invoked under XLA_FLAGS=--xla_force_host_platform_device_count=N (fake CPU
devices). Exits 0 on parity. Usage: _rung2_optimize_parity_subproc.py [D] [chi]
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig  # noqa: E402
from tenax.algorithms.ipeps_optimize import optimize_gs_ad  # noqa: E402


def _heisenberg():
    d = 2
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def _cfg(mesh, chi):
    return iPEPSConfig(
        max_bond_dim=2,
        # plateau_patience=None → converge to a true fixed point so the
        # implicit-AD gradient is exact (not the approximate early-bail
        # gradient); gs_line_search=False so the only CTM path is the sharded
        # value_and_grad.
        ctm=CTMConfig(
            chi=chi,
            max_iter=60,
            conv_tol=1e-10,
            plateau_patience=None,
            device_mesh=mesh,
        ),
        gs_num_steps=4,
        gs_learning_rate=1e-2,
        su_init=False,
        gs_metric_precond=False,
        gs_line_search=False,
    )


def main() -> int:
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    chi = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # WELL-CONDITIONED (near-product) init: sharding is exact up to FP
    # reassociation in the warm-started backward, which stays at machine
    # precision on a well-separated fixed point. A *random* init makes the CTM
    # fixed point ill-conditioned, so reassociation + the chaotic optimization
    # trajectory amplify the difference to ~1e-2 (benign — both converge to the
    # same minimum, but not testable at a tight tolerance). Same lesson as the
    # rung-1 forward parity.
    key = jax.random.PRNGKey(7)
    A = 0.02 * jax.random.normal(key, (D, D, D, D, 2))
    A = A.at[0, 0, 0, 0, :].add(1.0)
    A = A / (jnp.linalg.norm(A) + 1e-10)
    H = _heisenberg()

    A1, _, E1 = optimize_gs_ad(H, A, _cfg(None, chi))
    mesh = build_ctm_mesh()
    A4, _, E4 = optimize_gs_ad(H, A, _cfg(mesh, chi))

    a1 = np.asarray(A1.todense() if hasattr(A1, "todense") else A1)
    a4 = np.asarray(A4.todense() if hasattr(A4, "todense") else A4)
    de = abs(float(E1) - float(E4))
    da = float(np.max(np.abs(a1 - a4)))
    print(
        f"devices={jax.device_count()} D={D} chi={chi} "
        f"E_single={float(E1):.10f} E_sharded={float(E4):.10f} "
        f"|dE|={de:.2e} max|dA|={da:.2e}"
    )
    # Energy is the physical observable and must match tightly (both trajectories
    # converge to the same minimum). The tensor can drift along a flat/gauge
    # direction by the accumulated per-step reassociation (~1e-6, stable across
    # steps), so it gets only a loose sanity bound.
    return 0 if (de < 1e-8 and da < 1e-3) else 1


if __name__ == "__main__":
    sys.exit(main())
