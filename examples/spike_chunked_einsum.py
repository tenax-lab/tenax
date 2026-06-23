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


def _build(variant, batch):
    """Return the jitted fn for one variant. ``peak_bytes_in_use`` is a cumulative
    high-water mark JAX never resets, so each variant×D MUST run in its own process
    (this script measures exactly ONE variant per invocation)."""
    if variant == "full":
        return jax.jit(full)
    if variant == "chunked":
        return jax.jit(lambda X, Y: chunked(X, Y, batch))
    if variant == "remat":
        return jax.jit(lambda X, Y: chunked(X, Y, batch, remat=True))
    if variant == "g_full":
        return jax.jit(jax.grad(lambda X, Y: full(X, Y).sum()))
    if variant == "g_chunk":
        return jax.jit(jax.grad(lambda X, Y: chunked(X, Y, batch).sum()))
    if variant == "g_remat":
        return jax.jit(jax.grad(lambda X, Y: chunked(X, Y, batch, remat=True).sum()))
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument(
        "--variant",
        default="full",
        choices=["full", "chunked", "remat", "g_full", "g_chunk", "g_remat", "parity"],
    )
    args = ap.parse_args()
    B = args.chi * args.chi
    batch = max(1, B // args.chunks)
    X, Y = _make(args.D, args.chi, 0)
    hdr = (
        f"# chunked-einsum spike  D={args.D} chi={args.chi} (B={B}) "
        f"batch={batch} ({B // batch} chunks) x64={jax.config.jax_enable_x64}"
    )
    print(hdr)
    if args.variant == "parity":  # full vs chunked in one process — peak ignored
        of = jax.block_until_ready(jax.jit(full)(X, Y))
        oc = jax.block_until_ready(jax.jit(lambda X, Y: chunked(X, Y, batch))(X, Y))
        print(f"D={args.D} variant=parity  max|chunked-full|={float(jnp.max(jnp.abs(oc - of))):.2e}")
        return
    fn = _build(args.variant, batch)
    try:
        _o, dt = _time(fn, X, Y)
        print(
            f"D={args.D} variant={args.variant}  per_device_peak={peak_gb():.2f} GB  wall={dt:.3f}s"
        )
    except Exception as ex:  # noqa: BLE001
        print(f"D={args.D} variant={args.variant}  FAILED({type(ex).__name__}: {str(ex)[:80]})")


if __name__ == "__main__":
    main()
