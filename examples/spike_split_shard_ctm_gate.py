"""THROWAWAY feasibility gate (#632 frontier): does GSPMD multi-GPU sharding
give real per-device memory relief on the split-CTM (chi^2*D^4)
``value_and_grad``, or does the projector SVD force replication?

Spike only. NO src/ edits — the sharding hook is installed by monkeypatching
``tenax.algorithms._split_ctm_tensor_moves._doublelayer_grown_corner`` from
this standalone script. Do not commit; this lives in scratchpad/.

Background (see handoff prompt for full file:line citations):
  - Split value_and_grad entry: ctm_energy_split_implicit(...) — a custom_vjp
    implicit-AD fixed point; the backward reuses the same move functions via
    jax.vjp, so a with_sharding_constraint planted in a forward move
    propagates to the backward automatically.
  - Dominant chi^2*D^4 intermediate: _doublelayer_grown_corner grows Cg with
    legs (env=chi, u_ket=D, u_bra=D, remaining=chi) BEFORE it is fused to
    (chi*D^2, chi) and fed to the projector SVD (_compute_projector_tensor,
    M = C1g^H @ C4g, jnp.linalg.svd(M)).
  - The split path has ZERO existing device_mesh/sharding hooks.

IMPORTANT CAVEAT discovered while building this: the split-CTM forward
convergence loop (_converge_split_gauge_fixed) and the implicit backward's
Neumann loop (_split_ctm_converge_bwd) both use Python-level data-dependent
`if <concrete float> < conv_tol: break` control flow. This means the FULL
value_and_grad pipeline is NOT jittable end-to-end (confirmed empirically
below --hlo raises ConcretizationTypeError if you try). This mirrors the
dense path's own design (_make_jit_ctm_step jits ONE sweep step, not the
whole convergence loop) so for --hlo we jit exactly one gauge-fixed sweep
step (_split_step), which is the natural jittable unit and contains the
corner-growth + projector-SVD region we care about. --peak and --verify run
the real eager pipeline exactly as production code would call it (no
extra jit wrapper), so those numbers reflect genuine end-to-end behavior.
"""

from __future__ import annotations

import argparse
import os
import sys

# Must happen before `import jax`: give CPU-only runs a 2-device mesh so
# --verify can exercise real multi-device sharding without a GPU.
if os.environ.get("JAX_PLATFORMS") == "cpu" and "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.lax import with_sharding_constraint  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "tests"))
from _rung2_grad_probe import _indices, _init_data  # noqa: E402

from tenax.algorithms import _split_ctm_tensor_moves as _moves  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._split_ctm_energy_ad import (  # noqa: E402
    _split_step,
    ctm_energy_split_implicit,
)
from tenax.algorithms._split_ctm_tensor_init import (  # noqa: E402
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms._tensor_utils import fuse_indices  # noqa: E402
from tenax.core.index import FlowDirection  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

_AXIS = "d"
GATE = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

# ---------------------------------------------------------------------- #
# Monkeypatch: shard Cg's u_ket (dim-D) leg before the (env,u_ket,u_bra)   #
# fuse that feeds the projector SVD.                                      #
# ---------------------------------------------------------------------- #

_ORIGINAL_DOUBLELAYER_GROWN_CORNER = _moves._doublelayer_grown_corner

# Module-global mesh switch. None => patched fn behaves identically to the
# original (verified below). Set => inserts a with_sharding_constraint.
_MESH: Mesh | None = None

# Positive-control evidence collected by the patch (populated only when
# arrays are concrete, i.e. in the eager forward loop, not inside a vjp
# linearization trace).
_SHARD_EVIDENCE: list[dict] = []


def _patched_doublelayer_grown_corner(C, T_ket, T_bra, c_relabel, ket_I, bra_I, fuse_labels):
    """Re-implementation of _moves._doublelayer_grown_corner with an inserted
    with_sharding_constraint on the u_ket (fuse_labels[1]) axis, applied to
    the grown-but-not-yet-fused Cg, when a module mesh is active. Mirrors the
    original exactly when _MESH is None."""
    from tenax.contraction.contractor import contract

    C_r = C.relabel(*c_relabel)
    Cg = contract(C_r, T_ket)
    Cg = contract(Cg.relabel(ket_I, bra_I), T_bra)

    if _MESH is not None:
        labels = Cg.labels()
        axis = labels.index(fuse_labels[1])  # the ket-dim D leg, e.g. "u_ket"/"d_ket"
        ndim = Cg.ndim
        spec = [None] * ndim
        spec[axis] = _AXIS
        sh = NamedSharding(_MESH, PartitionSpec(*spec))
        leaves, treedef = jax.tree_util.tree_flatten(Cg)
        new_leaves = []
        for leaf in leaves:
            constrained = with_sharding_constraint(leaf, sh)
            new_leaves.append(constrained)
            if not isinstance(constrained, jax.core.Tracer):
                _SHARD_EVIDENCE.append(
                    {
                        "fuse_labels": fuse_labels,
                        "axis": axis,
                        "shape": tuple(constrained.shape),
                        "sharding": str(constrained.sharding),
                        "is_fully_replicated": bool(
                            constrained.sharding.is_fully_replicated
                        ),
                        "num_devices": len(constrained.sharding.device_set),
                    }
                )
        Cg = jax.tree_util.tree_unflatten(treedef, new_leaves)

    labels = Cg.labels()
    Cg = fuse_indices(
        Cg,
        labels.index(fuse_labels[0]),
        labels.index(fuse_labels[1]),
        "fused",
        FlowDirection.IN,
    )
    labels = Cg.labels()
    Cg = fuse_indices(
        Cg,
        labels.index("fused"),
        labels.index(fuse_labels[2]),
        "fused",
        FlowDirection.IN,
    )
    remaining = [lbl for lbl in Cg.labels() if lbl != "fused"][0]
    return Cg, remaining


def install_patch():
    _moves._doublelayer_grown_corner = _patched_doublelayer_grown_corner


def set_mesh(mesh: Mesh | None):
    global _MESH
    _MESH = mesh


# ---------------------------------------------------------------------- #
# State builder (shared, matches tests/_rung2_grad_probe.py conventions)  #
# ---------------------------------------------------------------------- #


def build_A(D: int, seed: int = 0, well_conditioned: bool = True, mesh: Mesh | None = None):
    idx = _indices(D)
    data = _init_data(D, seed, well_conditioned)
    if mesh is not None:
        n = mesh.devices.size
        if D % n != 0:
            raise ValueError(f"D={D} not divisible by mesh size {n}")
        spec = [None] * data.ndim
        spec[2] = _AXIS  # shard the "l" leg (axis 2), a D-dim virtual leg
        sh = NamedSharding(mesh, PartitionSpec(*spec))
        data = jax.device_put(data, sh)
    return DenseTensor(data, idx)


def energy_and_grad(D, chi, max_iter, mesh):
    idx = _indices(D)

    def loss(data):
        A = DenseTensor(data, idx)
        return ctm_energy_split_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            GATE,
            chi=chi,
            chi_I=chi,
            max_iter=max_iter,
            conv_tol=1e-10,
        )

    data0 = _init_data(D, 0, well_conditioned=True)
    if mesh is not None:
        # Pre-shard the (concrete, not-yet-differentiated) input on a D leg
        # ("l", axis 2) so the in-move with_sharding_constraint has a
        # non-replicated starting point to propagate from.
        n = mesh.devices.size
        if D % n != 0:
            raise ValueError(f"D={D} not divisible by mesh size {n}")
        spec = [None] * data0.ndim
        spec[2] = _AXIS
        data0 = jax.device_put(data0, NamedSharding(mesh, PartitionSpec(*spec)))
    set_mesh(mesh)
    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)


# ---------------------------------------------------------------------- #
# Modes                                                                  #
# ---------------------------------------------------------------------- #


def mode_verify(D, chi, max_iter):
    install_patch()
    devs = jax.devices()
    print(f"# devices={devs}")
    if len(devs) < 2:
        print("BLOCKED: --verify needs >=2 devices for a real mesh; "
              f"only {len(devs)} visible. Set XLA_FLAGS host device count "
              "or run with 2 GPUs visible.")
        return

    mesh = Mesh(np.asarray(devs[:2]), axis_names=(_AXIS,))

    # --- Positive control: run ONE eager forward move directly (not under
    # jax.grad) so the arrays touched by the patch are concrete, and we can
    # read back real sharding metadata. ---
    _SHARD_EVIDENCE.clear()
    set_mesh(mesh)
    A = build_A(D, seed=0, well_conditioned=True, mesh=mesh)
    env = initialize_split_ctm_tensor_env(A, chi, chi)
    _ = _split_step(A, env, chi, chi, renormalize=True)
    set_mesh(None)

    print(f"# positive-control evidence entries: {len(_SHARD_EVIDENCE)}")
    for ev in _SHARD_EVIDENCE[:8]:
        print(f"#   {ev}")
    if not _SHARD_EVIDENCE:
        print("POSITIVE CONTROL: FAILED (no evidence captured — patch never "
              "saw a concrete array; likely all calls were traced tracers)")
    else:
        any_sharded = any(not ev["is_fully_replicated"] for ev in _SHARD_EVIDENCE)
        print(f"POSITIVE CONTROL: {'PASS' if any_sharded else 'FAIL (fully replicated)'}"
              f" — any_non_replicated={any_sharded}")

    # --- Grad parity: shard vs no-shard ---
    e0, g0 = energy_and_grad(D, chi, max_iter, mesh=None)
    e1, g1 = energy_and_grad(D, chi, max_iter, mesh=mesh)
    set_mesh(None)

    diff = np.max(np.abs(g1 - g0))
    denom = np.max(np.abs(g0)) + 1e-300
    rel = diff / denom
    print(f"E (no-shard) = {e0!r}")
    print(f"E (shard)    = {e1!r}")
    print(f"max|g_shard - g_noshard| = {diff:.3e}")
    print(f"max|g_noshard| = {np.max(np.abs(g0)):.3e}")
    print(f"relative parity = {rel:.3e}")
    print(f"PARITY_{'PASS' if rel <= 1e-10 else 'FAIL'}: rel={rel:.3e} (threshold 1e-10)")


def _peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def mode_peak(D, chi, max_iter, shard):
    install_patch()
    devs = jax.devices()
    mesh = None
    if shard:
        mesh = Mesh(np.asarray(devs), axis_names=(_AXIS,))
    n = len(devs) if shard else 1
    print(f"# devices={devs} shard={shard} mesh_n={n} D={D} chi={chi} max_iter={max_iter}")
    import time

    t0 = time.perf_counter()
    try:
        e, g = energy_and_grad(D, chi, max_iter, mesh=mesh)
        dt = time.perf_counter() - t0
        print(
            f"D={D} chi={chi}: OK  E={e:.6f}  per_device_peak={_peak_gb():.3f} GB  "
            f"wall={dt:.1f}s  grad_norm={np.linalg.norm(g):.4e}"
        )
    except Exception as ex:  # noqa: BLE001 - benchmark reports failure and stops
        dt = time.perf_counter() - t0
        print(
            f"D={D} chi={chi}: FAILED ({type(ex).__name__}: {str(ex)[:200]})  "
            f"wall={dt:.1f}s"
        )
        raise
    finally:
        set_mesh(None)


def mode_hlo(D, chi):
    """Jit exactly one gauge-fixed sweep step (the natural jittable unit;
    the full convergence loop has Python-level data-dependent breaks and is
    NOT jittable — see module docstring) and inspect its compiled HLO for
    all-gather ops, especially near the projector SVD's fused (chi*D^2)
    leg."""
    install_patch()
    devs = jax.devices()
    if len(devs) < 2:
        print(f"BLOCKED: --hlo needs >=2 devices, only {len(devs)} visible.")
        return
    mesh = Mesh(np.asarray(devs[:2]), axis_names=(_AXIS,))
    set_mesh(mesh)

    A = build_A(D, seed=0, well_conditioned=True, mesh=mesh)
    env = initialize_split_ctm_tensor_env(A, chi, chi)

    def sweep_fn(A_data, env):
        A_ = DenseTensor(A_data, _indices(D))
        return _split_step(A_, env, chi, chi, renormalize=True)

    jitted = jax.jit(sweep_fn)
    try:
        lowered = jitted.lower(A.todense(), env)
        compiled = lowered.compile()
    except Exception as ex:  # noqa: BLE001
        print(f"BLOCKED compiling single sweep step: {type(ex).__name__}: {ex}")
        set_mesh(None)
        return
    hlo = compiled.as_text()
    set_mesh(None)

    out_path = os.path.join(_HERE, f"split_shard_gate_hlo_D{D}_chi{chi}.txt")
    with open(out_path, "w") as f:
        f.write(hlo)

    lines = hlo.splitlines()
    ag_lines = [l for l in lines if "all-gather" in l]
    svd_lines_idx = [i for i, l in enumerate(lines) if "custom-call" in l and (
        "cusolver" in l.lower() or "Svd" in l or "svd" in l.lower()
    )]
    print(f"# HLO written to {out_path} ({len(lines)} lines)")
    print(f"all-gather count: {len(ag_lines)}")
    for l in ag_lines[:20]:
        print(f"  AG: {l.strip()[:160]}")
    print(f"custom-call (svd-like) count: {len(svd_lines_idx)}")
    for i in svd_lines_idx[:10]:
        lo, hi = max(0, i - 3), min(len(lines), i + 4)
        print(f"  --- context around SVD call at line {i} ---")
        for j in range(lo, hi):
            marker = ">>" if j == i else "  "
            print(f"  {marker} {lines[j].strip()[:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--peak", action="store_true")
    ap.add_argument("--hlo", action="store_true")
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--shard", action="store_true")
    args = ap.parse_args()

    if args.verify:
        mode_verify(args.D, args.chi, args.max_iter)
    elif args.peak:
        mode_peak(args.D, args.chi, args.max_iter, args.shard)
    elif args.hlo:
        mode_hlo(args.D, args.chi)
    else:
        print("Specify one of --verify / --peak / --hlo")


if __name__ == "__main__":
    main()
