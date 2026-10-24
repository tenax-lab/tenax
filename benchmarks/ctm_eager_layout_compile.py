"""Compile count and wall time of the #1098 eager chi-layout sweeps.

Fixture: U(1) XXZ Delta=0.3 2-site iPEPS from a short simple update, then a
cold ``python_loop_ctm_converge``.  Each eager sweep is timed and the XLA
backend compiles inside it are counted (``jax.monitoring``).  After each
eager sweep the chi-leg layout ((charge, multiplicity) per chi leg) is printed;
at the end, the energy per site of the converged envs and, where the device
reports it, the device peak memory.

Run on one GPU (``CUDA_VISIBLE_DEVICES=1``) or with ``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import time

import jax

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._ctm_python_loop as cpl  # noqa: E402
from tenax import CTMConfig, ipeps, iPEPSConfig  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS  # noqa: E402
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    compute_energy_ctm_tensor_2site,
)
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402

_COMPILES = [0, 0.0]


def _listener(name, dur, **_):
    if name == "/jax/core/compile/backend_compile_duration":
        _COMPILES[0] += 1
        _COMPILES[1] += dur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--chi", type=int, default=8)
    ap.add_argument("--su-steps", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=40)
    a = ap.parse_args()
    jax.monitoring.register_event_duration_secs_listener(_listener)
    print("devices", jax.devices(), flush=True)

    A0, B0 = heisenberg_u1sz_init_pair(a.D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=a.D, num_imaginary_steps=a.su_steps, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))

    n_sweeps = [0]
    raw = cpl._eager_ctm_sweep

    def timed(*args, **kwargs):
        c0, s0, t0 = _COMPILES[0], _COMPILES[1], time.time()
        out = raw(*args, **kwargs)
        jax.block_until_ready(jax.tree.leaves(out[0]))
        n_sweeps[0] += 1
        print(
            f"eager sweep {n_sweeps[0]}: wall {time.time() - t0:.1f} s, "
            f"compiles {_COMPILES[0] - c0}, compile_s {_COMPILES[1] - s0:.1f}",
            flush=True,
        )
        layout = repr(cpl._chi_layout(out[0]))
        digest = hashlib.sha256(layout.encode()).hexdigest()[:16]
        print(f"eager sweep {n_sweeps[0]} layout {digest}: {layout}", flush=True)
        return out

    cpl._eager_ctm_sweep = timed
    cfg = dataclasses.replace(
        CTMConfig(chi=a.chi),
        conv_tol=1e-10,
        max_iter=a.max_iter,
        min_iter=5,
        plateau_patience=None,
        on_unconverged="warn",
    )
    c0, t0 = _COMPILES[0], time.time()
    envs, _ = cpl.python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
    )
    print(
        f"converge: wall {time.time() - t0:.1f} s, compiles {_COMPILES[0] - c0}, "
        f"eager_sweeps {n_sweeps[0]}",
        flush=True,
    )
    g = xxz_gate(0.3).todense()
    e = compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], g, 2)
    print(f"energy per site {float(e):.15f}", flush=True)
    stats = jax.devices()[0].memory_stats()
    if stats and "peak_bytes_in_use" in stats:
        print(f"device peak_bytes_in_use {stats['peak_bytes_in_use']}", flush=True)
    else:
        print("device peak_bytes_in_use n/a", flush=True)


if __name__ == "__main__":
    main()
