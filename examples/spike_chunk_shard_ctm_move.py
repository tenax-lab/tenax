"""Gate: chunk x shard the REAL dense-CTM edge move.

Extends spike_chunked_ctm_move.py with a GSPMD mesh. Shards a's r2=n (a free D2
output leg = ctm_sharding's left-move surviving axis), orthogonal to the chunk
axis (boundary chi). Four variants, ONE per process (peak_bytes is cumulative):
  full       : replicated monolith            (peak ~ chi^2 D^6)
  chunked    : 1-device, chunked over i        (peak / K)
  sharded    : monolith, a sharded on n        (peak / N)
  chunkshard : chunked over i, a sharded on n   (peak / (N*K))

  CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
      --D 10 --chi 64 --batch 16 --mesh-n 2 --variant chunkshard
"""
import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from tenax.algorithms._ctm_compiled_moves import _apply_projector_raw  # noqa: E402
from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402


def _inputs(D, chi, seed):
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    T4 = jax.random.normal(k[0], (chi, D2, chi)) / D2  # (t4_d=i, l2=j, t4_u=k)
    a = jax.random.normal(k[1], (D2, D2, D2, D2)) / D2  # (u2=l, d2=m, l2=j, r2=n)
    P1 = jax.random.normal(k[2], (chi * D2, chi)) / (chi * D2)
    P2 = jax.random.normal(k[3], (chi * D2, chi)) / (chi * D2)
    C1g = jax.random.normal(k[4], (chi * D2, chi)) / (chi * D2)
    C4g = jnp.zeros((chi * D2, chi))
    return T4, a, P1, P2, C1g, C4g


def full_move(T4, a, P1, P2, C1g, C4g, chi, D2):
    T4_a = jnp.einsum("ijk,lmjn->iklmn", T4, a)  # (chi,chi,D2,D2,D2)  PEAK
    T4_a = T4_a.transpose(0, 2, 1, 3, 4)
    T4g = T4_a.reshape(chi * D2, chi * D2, D2)
    _c1, _c4, T_new = _apply_projector_raw(P1, P2, C1g, C4g, T4g)
    return T_new  # (k_out, D2, k_out)


def chunked_move(T4, a, P1, P2, C1g, C4g, chi, D2, batch):
    P1r = P1.reshape(chi, D2, -1)  # (i, l, k_out)

    def per_i(args):
        T4_i, P1_i = args  # (D2[j], chi[k]) , (D2[l], k_out)
        T4a_i = jnp.einsum("jk,lmjn->klmn", T4_i, a)  # (k, l, m, n)=(chi,D2,D2,D2)
        T4g_i = T4a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (l, fr, n)
        step = jnp.tensordot(P1_i.conj(), T4g_i, axes=([0], [0]))  # (k_out, fr, n)
        return jnp.tensordot(step, P2, axes=([1], [0]))  # (k_out, D2, k_out)

    contribs = lax.map(per_i, (T4, P1r), batch_size=batch)  # (chi, k_out, D2, k_out)
    return contribs.sum(0)


def _commit(mesh, tensors, shard_a):
    rep = NamedSharding(mesh, P())
    a_sh = NamedSharding(mesh, P(None, None, None, "d")) if shard_a else rep
    T4, a, P1, P2, C1g, C4g = tensors
    return (
        jax.device_put(T4, rep),
        jax.device_put(a, a_sh),
        jax.device_put(P1, rep),
        jax.device_put(P2, rep),
        jax.device_put(C1g, rep),
        jax.device_put(C4g, rep),
    )


def _peak_gb(devs):
    vals = []
    for d in devs:
        try:
            vals.append(d.memory_stats()["peak_bytes_in_use"] / 1e9)
        except Exception:  # noqa: BLE001
            pass
    return max(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mesh-n", type=int, default=1)
    ap.add_argument(
        "--variant",
        default="chunkshard",
        choices=["full", "chunked", "sharded", "chunkshard", "parity"],
    )
    ap.add_argument("--hlo", action="store_true", help="print HLO collective/temp summary, no run")
    args = ap.parse_args()

    D2 = args.D * args.D
    devs = jax.devices()[: args.mesh_n]
    mesh = build_ctm_mesh(devs)
    shard = args.variant in ("sharded", "chunkshard")
    if shard and D2 % args.mesh_n != 0:
        raise SystemExit(f"D2={D2} not divisible by mesh_n={args.mesh_n}")
    tens = _inputs(args.D, args.chi, 0)
    print(
        f"# chunk-shard  D={args.D} chi={args.chi} D2={D2} batch={args.batch} "
        f"({args.chi // args.batch} chunks) mesh_n={args.mesh_n} variant={args.variant}"
    )

    # Arrays are passed as jit ARGUMENTS (not closed-over constants) so XLA does
    # not constant-fold the contraction at compile time — essential for a faithful
    # runtime peak. Committed (sharded) args carry their layout into the jit.
    jfull = jax.jit(lambda T4, a, P1, P2, C1g, C4g: full_move(T4, a, P1, P2, C1g, C4g, args.chi, D2))
    jchunk = jax.jit(
        lambda T4, a, P1, P2, C1g, C4g: chunked_move(T4, a, P1, P2, C1g, C4g, args.chi, D2, args.batch)
    )

    if args.variant == "parity":
        full = jax.block_until_ready(jfull(*_commit(mesh, tens, shard_a=False)))
        cs = jax.block_until_ready(jchunk(*_commit(mesh, tens, shard_a=True)))
        d = float(jnp.max(jnp.abs(full - cs)))
        rel = d / (float(jnp.max(jnp.abs(full))) + 1e-30)
        print(f"PARITY chunkshard vs full: max|d|={d:.2e} rel={rel:.2e}")
        return

    committed = _commit(mesh, tens, shard_a=shard)
    jfn = jchunk if args.variant in ("chunked", "chunkshard") else jfull

    if args.hlo:
        try:
            compiled = jfn.lower(*committed).compile()
            txt = compiled.as_text()
            ag = txt.count("all-gather")
            full_tok = f"[{D2}]"  # the sharded n=D2 axis, gathered would show full D2
            ag_fulln = sum(
                1 for ln in txt.splitlines() if "all-gather" in ln and full_tok in ln
            )
            temp = compiled.memory_analysis().temp_size_in_bytes / 1e9
            print(f"HLO all-gather={ag} (touching full n={D2}: {ag_fulln}) temp={temp:.3f} GB")
        except Exception as ex:  # noqa: BLE001
            print(f"HLO FAILED({type(ex).__name__}: {str(ex)[:90]})")
        return

    try:
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*committed))
        dt = time.perf_counter() - t0
        print(f"RUN variant={args.variant} peak={_peak_gb(devs):.2f} GB wall={dt:.3f}s")
    except Exception as ex:  # noqa: BLE001
        print(f"RUN variant={args.variant} FAILED({type(ex).__name__}: {str(ex)[:90]})")


if __name__ == "__main__":
    main()
