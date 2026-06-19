#!/usr/bin/env python3
"""THROWAWAY probe (not committed): isolate the ~7e-2 callback energy discrepancy.

For a FIXED normalized fermionic D=2 site, compute the CTM energy four ways, each
with a FRESH env_cache (kills the warm-start confound), at a loose and a tight
conv_tol:
  (a) direct call (jit)
  (b) via jax.pure_callback (jit)
  (c) via jax.pure_callback + jax.disable_jit() inside the host
  (d) direct call + jax.disable_jit()

If a==b==c==d at tight tol but they spread at loose tol -> the discrepancy is
convergence-looseness / max_iter, not a pure_callback/disable_jit artifact, and
the spike architecture (disable_jit in callback) is sound.
"""
from __future__ import annotations

import contextlib
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_spec = importlib.util.spec_from_file_location(
    "spike", pathlib.Path("examples/spike_ctm_cadjoint_566.py")
)
spike = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spike)

from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn
from tenax.algorithms.ipeps_config import CTMConfig


def fresh_energy_fn(gate, chi, depth, conv_tol, max_iter):
    cfg = CTMConfig(chi=chi, max_iter=max_iter, conv_tol=conv_tol)
    return make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: cfg,
        env_cache={},  # FRESH per build -> no warm-start carryover
        use_explicit=False,
        explicit_warmup=0,
        explicit_steps=depth,
    )


def main():
    A, gate = spike._PROF.make_site_and_gate("fermionic", 2, seed=42)
    reconstruct = spike.make_reconstructor(A)
    data = spike.leaf_of(A)
    A_norm = reconstruct(data) * (1.0 / (reconstruct(data).norm() + spike._EPS))
    data_norm = np.asarray(spike.leaf_of(A_norm))
    chi, depth = 8, 8

    def direct(conv_tol, max_iter, disable):
        efn = fresh_energy_fn(gate, chi, depth, conv_tol, max_iter)
        ctx = jax.disable_jit() if disable else contextlib.nullcontext()
        with ctx:
            return float(efn({(0, 0): reconstruct(jnp.asarray(data_norm))}))

    def callback(conv_tol, max_iter, disable):
        efn = fresh_energy_fn(gate, chi, depth, conv_tol, max_iter)

        def host(d):
            ctx = jax.disable_jit() if disable else contextlib.nullcontext()
            with ctx:
                return np.asarray(
                    efn({(0, 0): reconstruct(jnp.asarray(d))}), dtype=np.float64
                )

        out = jax.pure_callback(
            host, jax.ShapeDtypeStruct((), data_norm.dtype), jnp.asarray(data_norm)
        )
        return float(out)

    for conv_tol, max_iter, label in [
        (1e-4, 8, "loose conv_tol=1e-4 max_iter=8 (spike default)"),
        (1e-10, 200, "tight conv_tol=1e-10 max_iter=200"),
    ]:
        a = direct(conv_tol, max_iter, disable=False)
        b = callback(conv_tol, max_iter, disable=False)
        c = callback(conv_tol, max_iter, disable=True)
        d = direct(conv_tol, max_iter, disable=True)
        print(f"\n=== {label} ===")
        print(f"  (a) direct jit         : {a:.12f}")
        print(f"  (b) callback jit       : {b:.12f}   |b-a|={abs(b-a):.2e}")
        print(f"  (c) callback disable   : {c:.12f}   |c-a|={abs(c-a):.2e}")
        print(f"  (d) direct disable     : {d:.12f}   |d-a|={abs(d-a):.2e}")


if __name__ == "__main__":
    main()
