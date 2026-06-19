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


# Gate 1: spike compile collapse.  (sym, D, chi, depth)
_GATE1_SPIKE_GRID = [
    ("fermionic", 2, 8, 8),
    ("fermionic", 3, 12, 8),
    ("dense", 3, 12, 8),
]


def _measure_compile(loss, data, cap):
    """Cold value_and_grad compile: (wall_s, compile_s, n_compiles)."""
    vg = jax.value_and_grad(loss)
    wall, events, _out = _PROF._cold(vg, data, cap)
    return wall, sum(t for _, t in events), len(events)


def _gate1_row(arm, sym, D, chi, depth, cap):
    A, gate = _PROF.make_site_and_gate(sym, D, seed=42)
    reconstruct = make_reconstructor(A)
    data = leaf_of(A)
    loss_spike, loss_prod = make_losses(
        gate, chi, depth, reconstruct, stub_backward=True
    )
    loss = loss_spike if arm == "spike" else loss_prod
    wall, comp, ncomp = _measure_compile(loss, data, cap)
    return {
        "arm": arm, "sym": sym, "D": D, "chi": chi,
        "n_blocks": int(getattr(A, "n_blocks", 1)),
        "vg_wall_s": wall, "vg_compile_s": comp, "n_compiles": ncomp,
    }


def run_gate1(json_path=None):
    cap = _PROF._install_compile_capture()
    dev = jax.devices()[0]
    print("=" * 78)
    print(f"# Gate 1: spike compile collapse  [{dev.platform} {dev.device_kind}]")
    print("=" * 78)
    rows = []
    # production anchor (fermionic D=2) — same machine/code contrast (~minutes).
    rows.append(_gate1_row("baseline", "fermionic", 2, 8, 8, cap))
    for sym, D, chi, depth in _GATE1_SPIKE_GRID:
        rows.append(_gate1_row("spike", sym, D, chi, depth, cap))
        if json_path:
            with open(json_path, "w") as fh:
                json.dump({"platform": dev.platform, "rows": rows}, fh, indent=2)
    for r in rows:
        print(f"  {r['arm']:>8} {r['sym']:>9} D={r['D']} chi={r['chi']:>2} "
              f"blk={r['n_blocks']:>2}: vg_compile={r['vg_compile_s']:8.2f}s "
              f"wall={r['vg_wall_s']:8.2f}s n_compiles={r['n_compiles']}")
    if json_path:
        with open(json_path, "w") as fh:
            json.dump({"platform": dev.platform, "rows": rows}, fh, indent=2)
    sp = {(r["sym"], r["D"]): r for r in rows if r["arm"] == "spike"}
    bl = {(r["sym"], r["D"]): r for r in rows if r["arm"] == "baseline"}
    fD2, fD3, dD3 = sp[("fermionic", 2)], sp[("fermionic", 3)], sp[("dense", 3)]
    ratio = fD3["vg_compile_s"] / max(fD2["vg_compile_s"], 1e-9)
    go = (fD3["vg_compile_s"] < 30.0) and (ratio < 2.0)
    print("-" * 78)
    print(f"  spike fermionic D2->D3 compile ratio = {ratio:.2f}  (GO if < 2.0)")
    print(f"  spike fermionic D3 compile = {fD3['vg_compile_s']:.2f}s  (GO if < 30s)")
    print(f"  spike fermionic D3 vs dense D3 = {fD3['vg_compile_s']:.2f}s vs "
          f"{dD3['vg_compile_s']:.2f}s  (want ~equal)")
    if ("fermionic", 2) in bl:
        b = bl[("fermionic", 2)]
        sx = b["vg_compile_s"] / max(fD2["vg_compile_s"], 1e-9)
        print(f"  production baseline fermionic D2 = {b['vg_compile_s']:.2f}s  "
              f"(spike D2 = {fD2['vg_compile_s']:.2f}s -> {sx:.1f}x faster)")
    print("  recorded baseline fermionic vg_cmp: 206s -> 2111s D2->D3 (~10x)")
    print(f"\n  GATE 1: {'GO' if go else 'NO-GO'}")
    return go


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
    if args.gate1:
        run_gate1(args.json)


if __name__ == "__main__":
    main()
