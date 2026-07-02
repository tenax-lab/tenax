r"""Probe (a) for #570 QR-CTMRG "re-open multi-GPU" hypothesis.

Question
--------
The multi-GPU CTM-AD NO-GO (#632/#663/#672/#673) is *path-agnostic*: XLA lowers the
CTM projector SVD as **unshardable**, so it all-gathers whatever GSPMD sharded ->
the dominant intermediate is replicated -> ~no per-device relief.

But that verdict was measured on the **dense 2x2** (full ``chiD^2 x chiD^2`` SVD) and
the **split** ``(chiD)^2`` paths, where the decomposition consumes a *large* operand.
The reduced-corner ``recipe="1x1"`` scheme (#570) decomposes a *small* corner:
``chiD^2 x chi`` (SVD) or ``chiD^2 x 2chi`` -> tiny ``2chi x 2chi`` (QR then SVD of R).

So the structural question this probe settles:

    When the decomposition consumes the *reduced* corner, does XLA still all-gather
    the BIG absorption intermediate (chi^2 D^4) back to replicated (same NO-GO),
    or does it keep the big contraction SHARDED and only touch the SMALL corner?

If the big contraction stays sharded (per-device peak ~ 1/N), payoff-2 is
structurally alive and worth a real build. If the reduced path replicates the big
intermediate just like the dense path, it is the same path-agnostic NO-GO.

Method
------
Three CTM-move-shaped patterns, identical producer contraction, differing only in
the decomposition operand size:

  * ``full``        : reshape X -> M_full (chiD^2 x chiD^2) -> svd          (dense 2x2 analog)
  * ``reduced-svd`` : reduce  X -> M_red  (chiD^2 x chi)    -> svd          (recipe=1x1, method=svd)
  * ``reduced-qr``  : reduce  X -> M_red  (chiD^2 x 2chi)   -> qr -> svd(R) (recipe=1x1, method=qr)

For each: shard a *surviving* D^2 leg of the double-layer ``a`` (as the production
sharding does), build the SPMD executable, and report
  (1) parity of the gauge-invariant subspace projector P P^H vs single-device,
  (2) every collective op XLA inserted + its largest operand (bytes),
  (3) per-device peak temp memory (memory_analysis), 1-device vs N-device.

Run
---
    # GSPMD partitioning decision (platform-agnostic), faked CPU devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python \
        examples/probe_qr_shard_reopen_570.py --chi 16 --d2 16

    # confirm on the real GPU mesh:
    uv run python examples/probe_qr_shard_reopen_570.py --chi 16 --d2 64
"""

from __future__ import annotations

import argparse
import re

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import numpy as np

_AXIS = "d"


# --------------------------------------------------------------------------- #
# Producer: the "big" absorption contraction (carries the sharded D^2 leg).
# Mirrors examples/spike_ctm_sharding.enlarged_corner: X = (chi, D2, D2, chi).
# --------------------------------------------------------------------------- #
def _producer(C, Th, Tv, a):
    x = jnp.einsum("ij,jkl->ikl", C, Th)  # (chi, D2, chi)
    x = jnp.einsum("ikl,lmn->ikmn", x, Tv)  # (chi, D2, D2, chi)
    x = jnp.einsum("ikmn,kmpq->ipqn", x, a)  # (chi, D2, D2, chi); axis p is a's survivor
    return x  # (chi, D2_i-side, D2_p=SHARDED, chi)


def _truncate_svd(M, chi):
    U, _S, _Vh = jnp.linalg.svd(M, full_matrices=False)
    return U[:, :chi]


def _reduced_qr_proj(M, chi):
    # mirrors _ctm_projector._reduced_qr_projector / the tracer qr branch:
    # QR of the (chiD^2 x 2chi) reduced corner, then eigh/svd of the tiny R.
    Q, R = jnp.linalg.qr(M)  # M: (chiD^2, 2chi) -> Q (chiD^2, 2chi), R (2chi, 2chi)
    U_R, _S, _Vh = jnp.linalg.svd(R, full_matrices=False)  # tiny 2chi x 2chi
    k = min(chi, U_R.shape[1])
    return Q @ U_R[:, :k]  # (chiD^2, chi)


def make_fns(chi, d2):
    """Return {name: fn(C, Th, Tv, a, R0, R0b) -> P (chiD^2, chi)}."""

    def full(C, Th, Tv, a, R0, R0b):
        X = _producer(C, Th, Tv, a)  # (chi, D2, D2, chi)
        M = X.reshape(chi * d2, d2 * chi)  # (chiD^2, chiD^2); sharded axis on COLS
        return _truncate_svd(M, chi)  # svd of the FULL corner (the wall)

    def reduced_svd(C, Th, Tv, a, R0, R0b):
        X = _producer(C, Th, Tv, a)  # (chi, D2_k, D2_m=SHARDED, chi_n)
        # reduce: contract the sharded (m) and chi (n) legs -> M_red (chi, D2_k, chi_j)
        M = jnp.einsum("ikmn,mnj->ikj", X, R0)  # (chi, D2, chi); contracts sharded m
        M = M.reshape(chi * d2, chi)  # (chiD^2, chi)
        return _truncate_svd(M, chi)

    def reduced_qr(C, Th, Tv, a, R0, R0b):
        X = _producer(C, Th, Tv, a)
        M1 = jnp.einsum("ikmn,mnj->ikj", X, R0).reshape(chi * d2, chi)
        M2 = jnp.einsum("ikmn,mnj->ikj", X, R0b).reshape(chi * d2, chi)
        M = jnp.concatenate([M1, M2], axis=1)  # (chiD^2, 2chi)  (both reduced corners)
        return _reduced_qr_proj(M, chi)

    return {"full": full, "reduced-svd": reduced_svd, "reduced-qr": reduced_qr}


def make_inputs(chi, d2, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 6)
    C = jax.random.normal(k[0], (chi, chi))
    Th = jax.random.normal(k[1], (chi, d2, chi))
    Tv = jax.random.normal(k[2], (chi, d2, chi))
    a = jax.random.normal(k[3], (d2, d2, d2, d2))
    R0 = jax.random.normal(k[4], (d2, chi, chi))  # reducer (m, n) -> j
    R0b = jax.random.normal(k[5], (d2, chi, chi))
    return C, Th, Tv, a, R0, R0b


# --------------------------------------------------------------------------- #
# HLO collective inspection
# --------------------------------------------------------------------------- #
_COLLECTIVES = (
    "all-gather",
    "all-reduce",
    "all-to-all",
    "reduce-scatter",
    "collective-permute",
    "dynamic-slice",  # GSPMD sometimes materializes gathers as ds loops (rare)
)
_SHAPE_RE = re.compile(r"(?:f64|f32|c128|c64)\[[\d,]+\]")


def _bytes_of(shape_tok):
    dtype, dims = shape_tok.split("[")
    dims = dims.rstrip("]")
    n = 1
    for d in dims.split(","):
        n *= int(d)
    width = {"f64": 8, "f32": 4, "c128": 16, "c64": 8}[dtype]
    return n * width


def scan_collectives(hlo_text):
    """Return list of (op, biggest_operand_shape, bytes) for real cross-device ops."""
    found = []
    for line in hlo_text.splitlines():
        for op in _COLLECTIVES:
            if f" {op}(" in line or line.strip().startswith(op + "(") or f"= {op}" in line:
                if op == "dynamic-slice":  # only count if it looks like a partition gather
                    if "replica" not in line and "partition" not in line:
                        continue
                shapes = _SHAPE_RE.findall(line)
                if not shapes:
                    continue
                biggest = max(shapes, key=_bytes_of)
                found.append((op, biggest, _bytes_of(biggest)))
                break
    return found


def subspace(P):
    return P @ P.conj().T  # gauge-invariant projector onto column space


def run_pattern(name, fn, args, chi, mesh, n):
    # single-device reference
    single = jax.jit(fn, static_argnums=())(*args)
    G_single = subspace(single)

    # sharded: shard a's surviving leg (axis 2 = the producer's p axis)
    a_spec = PartitionSpec(None, None, _AXIS, None)
    in_shardings = (
        NamedSharding(mesh, PartitionSpec(None, None)),  # C
        NamedSharding(mesh, PartitionSpec(None, None, None)),  # Th
        NamedSharding(mesh, PartitionSpec(None, None, None)),  # Tv
        NamedSharding(mesh, a_spec),  # a  (surviving D2 leg sharded)
        NamedSharding(mesh, PartitionSpec(None, None, None)),  # R0
        NamedSharding(mesh, PartitionSpec(None, None, None)),  # R0b
    )
    jf = jax.jit(fn, in_shardings=in_shardings)
    args_sh = [jax.device_put(x, s) for x, s in zip(args, in_shardings)]
    lowered = jf.lower(*args_sh)
    compiled = lowered.compile()
    sharded = compiled(*args_sh)
    G_sh = subspace(jax.device_get(sharded))

    err = float(jnp.max(jnp.abs(G_single - G_sh)))
    hlo = compiled.as_text()
    colls = scan_collectives(hlo)

    # per-device peak temp memory
    def peak(exe):
        try:
            ma = exe.memory_analysis()
            return getattr(ma, "temp_size_in_bytes", 0) + getattr(
                ma, "output_size_in_bytes", 0
            )
        except Exception:
            return -1

    single_exe = jax.jit(fn).lower(*args).compile()
    peak_1 = peak(single_exe)
    peak_n = peak(compiled)

    return {
        "name": name,
        "err": err,
        "collectives": colls,
        "peak_1dev": peak_1,
        "peak_ndev": peak_n,
        "max_coll_bytes": max((b for _, _, b in colls), default=0),
    }


def isolated(chi, d2, mesh, n):
    """Purest form of probe (a): a matrix already sharded on its tall (chiD^2) axis,
    decomposed. Square (dense corner) vs tall-skinny (reduced corner).

    A tall-skinny SVD is *mathematically* shardable (M^H M is chi x chi -> small
    all-reduce; U = M V S^-1 stays sharded on the tall axis). A square SVD is not.
    This checks whether XLA's lowering exploits that.
    """
    m = chi * d2
    row_sh = NamedSharding(mesh, PartitionSpec(_AXIS, None))  # shard tall axis

    cases = {
        "square-svd  (chiD^2 x chiD^2)": (m, lambda M: _truncate_svd(M, chi)),
        "tall-svd    (chiD^2 x chi)": (chi, lambda M: _truncate_svd(M, chi)),
        "tall-qr     (chiD^2 x 2chi)": (2 * chi, lambda M: _reduced_qr_proj(M, chi)),
    }
    print(f"## isolated decomposition (input sharded on tall axis)  m={m}")
    for label, (ncol, fn) in cases.items():
        M = jax.random.normal(jax.random.PRNGKey(1), (m, ncol))
        single = jax.jit(fn)(M)
        G1 = subspace(single)
        jf = jax.jit(fn, in_shardings=(row_sh,))
        Ms = jax.device_put(M, row_sh)
        compiled = jf.lower(Ms).compile()
        out = compiled(Ms)
        Gs = subspace(jax.device_get(out))
        err = float(jnp.max(jnp.abs(G1 - Gs)))
        out_shard = out.sharding.shard_shape(out.shape)
        out_sharded = out_shard != out.shape
        colls = scan_collectives(compiled.as_text())
        maxb = max((b for _, _, b in colls), default=0)
        print(f"  {label}")
        print(f"     parity={err:.1e}  out_shape={out.shape} out_shard={out_shard} "
              f"sharded={out_sharded}")
        print(f"     collectives={len(colls)} biggest_operand={maxb/1e6:.3f}MB "
              f"input={m*ncol*8/1e6:.3f}MB")
        seen = set()
        for op, shp, b in colls:
            if (op, shp) in seen:
                continue
            seen.add((op, shp))
            print(f"        {op:20s} {shp:24s} {b/1e6:.3f}MB")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--d2", type=int, default=16, help="D^2 (e.g. 16 for D=4, 64 for D=8)")
    ap.add_argument("--only-isolated", action="store_true")
    args_ns = ap.parse_args()
    chi, d2 = args_ns.chi, args_ns.d2

    jax.config.update("jax_enable_x64", True)
    n = jax.device_count()
    mesh = Mesh(np.asarray(jax.devices()), axis_names=(_AXIS,))

    print(f"# probe_qr_shard_reopen_570  devices={n}  chi={chi}  D2={d2}  chiD2={chi*d2}")
    print(f"#   X (big intermediate)  = chi^2 D^4 = {chi*chi*d2*d2} elems "
          f"= {chi*chi*d2*d2*8/1e6:.2f} MB (f64)")
    print(f"#   M_full (dense corner) = chiD^2 x chiD^2 = {(chi*d2)**2} elems "
          f"= {(chi*d2)**2*8/1e6:.2f} MB")
    print(f"#   M_red  (reduced)      = chiD^2 x chi    = {chi*d2*chi} elems "
          f"= {chi*d2*chi*8/1e6:.2f} MB")
    print()

    isolated(chi, d2, mesh, n)
    if args_ns.only_isolated:
        return

    fns = make_fns(chi, d2)
    inp = make_inputs(chi, d2)
    for name, fn in fns.items():
        r = run_pattern(name, fn, inp, chi, mesh, n)
        relief = (r["peak_1dev"] / r["peak_ndev"]) if r["peak_ndev"] > 0 else float("nan")
        print(f"== {name} ==")
        print(f"   parity |G_single - G_sharded|max = {r['err']:.2e}")
        print(f"   per-device peak: 1dev={r['peak_1dev']/1e6:.2f}MB  "
              f"{n}dev={r['peak_ndev']/1e6:.2f}MB  relief={relief:.2f}x  (ideal={n}x)")
        if r["collectives"]:
            print(f"   collectives ({len(r['collectives'])}); biggest operand "
                  f"{r['max_coll_bytes']/1e6:.3f}MB:")
            # show up to 6 distinct
            seen = set()
            for op, shp, b in r["collectives"]:
                key = (op, shp)
                if key in seen:
                    continue
                seen.add(key)
                print(f"       {op:20s} {shp:24s} {b/1e6:.3f}MB")
                if len(seen) >= 8:
                    break
        else:
            print("   collectives: NONE")
        print()


if __name__ == "__main__":
    main()
