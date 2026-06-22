"""Spike: is chunked einsum (jax.lax.map batch_size + remat) a single-GPU peak-memory
lever for the dense-CTM χ²·D⁶ intermediate?

Mimics the CTM peak: a contraction whose intermediate scales as B·D⁶ (B = χ²) with a
free batch axis that chunks cleanly. Compares full materialization vs chunked
(lax.map batch_size) vs chunked+remat — measuring parity, per-device peak, and wall.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/spike_chunked_einsum.py --D 10 12 14 --chi 48 --chunks 4
    add --grad to measure the backward (where remat matters).

Gate: chunked peak ≈ full_peak / chunks (XLA respects the chunk) AND chunked runs
where full OOMs AND runtime tax ≈ chunks× (not catastrophic). Else NO-GO.
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)  # match the f64 CTM peak sizes

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402


def _make(D, chi, seed):
    D2 = D * D
    B = chi * chi
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    X = jax.random.normal(k[0], (B, D2, D2)) / D2
    Y = jax.random.normal(k[1], (D2, D2, D2)) / D2
    return X, Y


def full(X, Y):
    # M = (B, D², D², D²) = B·D⁶ — the materialized peak (χ²·D⁶ shape).
    M = jnp.einsum("bik,kjl->bijl", X, Y)
    return M.sum((2, 3))  # (B, D²)


def _per_row(Xb, Y):
    Mb = jnp.einsum("ik,kjl->ijl", Xb, Y)  # (D², D², D²) = D⁶
    return Mb.sum((1, 2))  # (D²,)


def chunked(X, Y, batch, remat=False):
    def f(Xb):
        return _per_row(Xb, Y)

    if remat:
        f = jax.checkpoint(f)
    return lax.map(f, X, batch_size=batch)  # peak ≈ batch·D⁶


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def _time(fn, *args):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - t0


def run_one(D, chi, chunks, grad):
    X, Y = _make(D, chi, 0)
    B = chi * chi
    batch = max(1, B // chunks)
    jfull = jax.jit(full)
    jchunk = jax.jit(lambda X, Y: chunked(X, Y, batch))
    jchunk_r = jax.jit(lambda X, Y: chunked(X, Y, batch, remat=True))
    if grad:
        # backward: d sum(out) / dX, where remat should cut activation memory.
        gfull = jax.jit(jax.grad(lambda X, Y: full(X, Y).sum()))
        gchunk = jax.jit(jax.grad(lambda X, Y: chunked(X, Y, batch).sum()))
        gchunk_r = jax.jit(jax.grad(lambda X, Y: chunked(X, Y, batch, remat=True).sum()))

    res = {"D": D, "B": B, "batch": batch, "chunks": B // batch}
    try:
        of, tf = _time(jfull, X, Y)
        res["full"] = (peak_gb(), tf)
    except Exception as ex:  # noqa: BLE001
        res["full"] = (f"OOM:{type(ex).__name__}", float("nan"))
        of = None
    try:
        oc, tc = _time(jchunk, X, Y)
        err = float(jnp.max(jnp.abs(oc - of))) if of is not None else float("nan")
        res["chunked"] = (peak_gb(), tc, err)
    except Exception as ex:  # noqa: BLE001
        res["chunked"] = (f"OOM:{type(ex).__name__}", float("nan"), float("nan"))
    try:
        or_, tr = _time(jchunk_r, X, Y)
        res["chunked_remat"] = (peak_gb(), tr)
    except Exception as ex:  # noqa: BLE001
        res["chunked_remat"] = (f"OOM:{type(ex).__name__}", float("nan"))
    if grad:
        for name, g in (("g_full", gfull), ("g_chunk", gchunk), ("g_chunk_remat", gchunk_r)):
            try:
                _o, _t = _time(g, X, Y)
                res[name] = (peak_gb(), _t)
            except Exception as ex:  # noqa: BLE001
                res[name] = (f"OOM:{type(ex).__name__}", float("nan"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--grad", action="store_true")
    args = ap.parse_args()
    print(
        f"# chunked-einsum spike  chi={args.chi} (B={args.chi**2})  "
        f"chunks={args.chunks}  grad={args.grad}  x64={jax.config.jax_enable_x64}"
    )
    for D in args.D:
        r = run_one(D, args.chi, args.chunks, args.grad)
        print(f"D={D} B={r['B']} batch={r['batch']} ({r['chunks']} chunks):")
        for kk in ("full", "chunked", "chunked_remat", "g_full", "g_chunk", "g_chunk_remat"):
            if kk in r:
                print(f"    {kk:14s} {r[kk]}")


if __name__ == "__main__":
    main()
