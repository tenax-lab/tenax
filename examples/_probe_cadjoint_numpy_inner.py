#!/usr/bin/env python3
"""THROWAWAY probe: does pure_callback collapse the OUTER compile to O(1)?

Disambiguates the Gate-1 NO-GO. The real-CTM spike scaled (166s at D3) because
``disable_jit`` runs the inner CTM as eager JAX -> per-op XLA compiles. This probe
replaces the inner CTM with a pure-NUMPY 'fake energy' (zero JAX ops inside the
callback), so the only thing XLA compiles is the trivial OUTER graph (normalize +
two pure_callback ops). If compile is tiny AND flat across fermionic D=2 vs D=3
(both opaque), the architecture (pure_callback) DOES collapse the outer compile,
and the real-CTM scaling is purely the JAX-eager-inner tax a C kernel would remove.
"""
from __future__ import annotations

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


def make_numpy_loss(reconstruct):
    @jax.custom_vjp
    def fake(data):
        return jax.pure_callback(
            lambda d: np.float64(np.sum(np.asarray(d) ** 2)),
            jax.ShapeDtypeStruct((), data.dtype),
            data,
        )

    def fwd(data):
        return fake(data), data

    def bwd(res, ct):
        g = jax.pure_callback(
            lambda d, c: np.asarray(2.0 * np.asarray(d) * np.asarray(c), d.dtype),
            jax.ShapeDtypeStruct(res.shape, res.dtype),
            res, ct,
        )
        return (g,)

    fake.defvjp(fwd, bwd)

    def loss(data):
        A = reconstruct(data)
        A_norm = A * (1.0 / (A.norm() + spike._EPS))
        return fake(spike.leaf_of(A_norm))

    return loss


def main():
    cap = spike._PROF._install_compile_capture()
    print(f"# numpy-inner outer-compile probe  [{jax.devices()[0].device_kind}]")
    for sym, D in [("fermionic", 2), ("fermionic", 3)]:
        A, _gate = spike._PROF.make_site_and_gate(sym, D, seed=42)
        reconstruct = spike.make_reconstructor(A)
        data = spike.leaf_of(A)
        loss = make_numpy_loss(reconstruct)
        vg = jax.value_and_grad(loss)
        wall, events, _out = spike._PROF._cold(vg, data, cap)
        comp = sum(t for _, t in events)
        print(f"  {sym} D={D} blk={A.n_blocks}: outer_vg_compile={comp:.3f}s "
              f"n_compiles={len(events)} wall={wall:.3f}s")


if __name__ == "__main__":
    main()
