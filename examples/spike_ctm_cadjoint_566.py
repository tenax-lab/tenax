#!/usr/bin/env python3
"""#566 C-adjoint feasibility spike — architectural GO/NO-GO (numpy callbacks, no C).

Wraps the production ``ctm_energy_implicit`` (via ``make_ctm_energy_fn``) in a
parallel ``jax.custom_vjp`` whose forward/backward are ``jax.pure_callback``s.
The host callbacks run the production energy and its ``jax.vjp`` under
``jax.disable_jit()`` so XLA never emits per-block ops in the outer graph
(the callback is opaque) and no fused per-block jaxpr is built inside the
callback either (every internal jit is eager).

Two staged gates (see the design spec):
  Gate 1 (compile collapse): spike vg_compile ~flat in block count, seconds not minutes.
  Gate 2 (AD-correctness):   spike grad vs production grad < 1e-6 at fermionic D=2.

Usage::
    uv run python examples/spike_ctm_cadjoint_566.py --self-check
    uv run python examples/spike_ctm_cadjoint_566.py --gate1 --json spike_gate1.json
    uv run python examples/spike_ctm_cadjoint_566.py --gate2 --json spike_gate2.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reuse the production loss dispatcher + site/gate builders + compile capture.
_SPEC = importlib.util.spec_from_file_location(
    "profile_ctm_ad_wall_566",
    pathlib.Path(__file__).parent / "profile_ctm_ad_wall_566.py",
)
_PROF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROF)

from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import make_ctm_energy_fn  # noqa: E402
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402

_EPS = 1e-10


def make_reconstructor(template):
    """Return ``reconstruct(data) -> Tensor`` reusing template's static pytree aux."""
    treedef = jax.tree_util.tree_structure(template)

    def reconstruct(data):
        return jax.tree_util.tree_unflatten(treedef, [data])

    return reconstruct


def leaf_of(tensor):
    """Return the single flat-buffer leaf of a SymmetricTensor/DenseTensor."""
    leaves, _ = jax.tree_util.tree_flatten(tensor)
    assert len(leaves) == 1, f"expected single leaf, got {len(leaves)}"
    return leaves[0]


def build_energy_fn(gate, chi, depth):
    """Production 1x1 implicit-AD energy_fn: site_tensors_dict -> energy."""
    ctm_cfg = CTMConfig(chi=chi, max_iter=depth, conv_tol=1e-4)
    return make_ctm_energy_fn(
        neighbors=SINGLE_SITE_NEIGHBORS,
        gate=gate,
        get_ctm_cfg=lambda: ctm_cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=0,
        explicit_steps=depth,
    )


def make_ctm_energy_cb(energy_fn, reconstruct, *, stub_backward):
    """custom_vjp over the flat data buffer; fwd/bwd run via pure_callback.

    Host functions run the production energy (and its vjp) under
    ``jax.disable_jit()`` so every internal ``@jax.jit`` becomes eager
    op-by-op dispatch: no fused per-block jaxpr is built anywhere, and XLA
    sees one opaque ``pure_callback`` op each direction in the outer graph.
    ``stub_backward=True`` returns a zero cotangent (Gate-1 compile test only).

    Verified (examples/_probe_cadjoint_discrepancy.py, fermionic D=2, FRESH
    env_cache): direct-jit, callback-jit, callback-disable_jit and
    direct-disable_jit agree on the energy to machine precision (6.7e-16 at
    conv_tol=1e-4; 1.2e-11 at conv_tol=1e-10). ``disable_jit`` does NOT change
    the numerics; an earlier ~7e-2 spread was an env-warm-start confound from a
    SHARED env_cache + max_iter=8, not a pure_callback/disable_jit artifact
    (make_losses now uses separate fresh caches).
    """

    def host_energy(data_np):
        with jax.disable_jit():
            A = reconstruct(jnp.asarray(data_np))
            return np.asarray(energy_fn({(0, 0): A}), dtype=np.float64)

    def host_grad(data_np, ct_np):
        with jax.disable_jit():
            data = jnp.asarray(data_np)

            def e_of_data(d):
                return energy_fn({(0, 0): reconstruct(d)})

            _, vjp = jax.vjp(e_of_data, data)
            (g,) = vjp(jnp.asarray(ct_np))
            return np.asarray(g, dtype=data_np.dtype)

    @jax.custom_vjp
    def ctm_energy_cb(data):
        return jax.pure_callback(
            host_energy, jax.ShapeDtypeStruct((), data.dtype), data
        )

    def _fwd(data):
        return ctm_energy_cb(data), data

    def _bwd(res, ct):
        data = res
        if stub_backward:
            return (jnp.zeros_like(data),)
        g = jax.pure_callback(
            host_grad, jax.ShapeDtypeStruct(data.shape, data.dtype), data, ct
        )
        return (g,)

    ctm_energy_cb.defvjp(_fwd, _bwd)
    return ctm_energy_cb


def make_losses(gate, chi, depth, reconstruct, *, stub_backward):
    """Return (loss_spike, loss_prod), both flat-array -> scalar, with the
    SAME normalization the production loss uses.

    loss_spike and loss_prod use SEPARATE fresh env_caches so a single-shot
    parity/gradient comparison is not contaminated by env warm-start (the
    confound that earlier made a shared-cache check look like a ~7e-2 callback
    discrepancy). loss_spike runs through the pure_callback path; loss_prod is
    the production (jitted) reference.
    """
    efn_spike = build_energy_fn(gate, chi, depth)
    efn_prod = build_energy_fn(gate, chi, depth)
    cb = make_ctm_energy_cb(efn_spike, reconstruct, stub_backward=stub_backward)

    def _normalized(data):
        A = reconstruct(data)
        return A * (1.0 / (A.norm() + _EPS))

    def loss_spike(data):
        return cb(leaf_of(_normalized(data)))

    def loss_prod(data):
        return efn_prod({(0, 0): _normalized(data)})

    return loss_spike, loss_prod


def _self_check():
    A, _gate = _PROF.make_site_and_gate("fermionic", 2, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    A2 = reconstruct(data)
    d = float(jnp.max(jnp.abs(leaf_of(A2) - data)))
    assert d == 0.0, f"round-trip mismatch {d}"
    print(f"[self-check] reconstruct round-trip exact (n_blocks={A.n_blocks}, "
          f"leaf={data.shape}) OK")


def _fwd_check(sym="fermionic", D=2, chi=8, depth=8):
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    loss_spike, loss_prod = make_losses(
        gate, chi, depth, reconstruct, stub_backward=True
    )
    # Forward parity THROUGH the pure_callback path (loss_spike) vs the
    # production reference (loss_prod). Separate fresh env_caches (see
    # make_losses) remove the warm-start confound, so these agree to machine
    # precision.
    e_spike = float(loss_spike(data))
    e_prod = float(loss_prod(data))
    print(f"[fwd-check] {sym} D={D} chi={chi}: spike={e_spike:.10f} "
          f"prod={e_prod:.10f} |Δ|={abs(e_spike - e_prod):.2e}")
    assert abs(e_spike - e_prod) < 1e-8, "forward energy mismatch through callback"
    # value_and_grad must run without error even with the stub backward.
    val, grad = jax.value_and_grad(loss_spike)(data)
    assert bool(jnp.all(jnp.isfinite(grad))), "non-finite stub grad"
    print(f"[fwd-check] value_and_grad ran (stub grad finite, ||g||="
          f"{float(jnp.linalg.norm(grad)):.2e}) OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--fwd-check", action="store_true")
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.self_check:
        _self_check()
    if args.fwd_check:
        _fwd_check()


if __name__ == "__main__":
    main()
