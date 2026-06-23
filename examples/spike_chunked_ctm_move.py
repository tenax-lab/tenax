"""Production follow-up spike: can the REAL dense-CTM move expose a chunkable axis
on its χ²·D⁶ peak intermediate, and does chunking it cut the move's peak?

Mirrors the left-move edge path of `_ctm_compiled_moves._compiled_move_left`:
the peak is T4_a = einsum('ijk,lmjn->iklmn', T4, a) = (χ,χ,D²,D²,D²), reshaped to
the grown edge T4g and projected to T_new = P1^H @ T4g @ P2. The leading χ axis
(t4_d=i) is free in T4_a but becomes a *contracted* axis through the projector
(fl=(i,l)), so chunking it = stream over i + accumulate the projector contraction.

This script implements that chunked-fused move, checks parity vs the real
`_apply_projector_raw`, and measures per-device peak full vs chunked. ONE variant
per process (peak_bytes_in_use is cumulative).

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/spike_chunked_ctm_move.py --D 10 --chi 48 --chunks 4 --variant full
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from tenax.algorithms._ctm_compiled_moves import _apply_projector_raw  # noqa: E402


def _inputs(D, chi, seed):
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    T4 = jax.random.normal(k[0], (chi, D2, chi)) / D2  # (t4_d=i, l2=j, t4_u=k)
    a = jax.random.normal(k[1], (D2, D2, D2, D2)) / D2  # (u2=l, d2=m, l2=j, r2=n)
    P1 = jax.random.normal(k[2], (chi * D2, chi)) / (chi * D2)  # (fl, k_out)
    P2 = jax.random.normal(k[3], (chi * D2, chi)) / (chi * D2)  # (fr, k_out)
    C1g = jax.random.normal(k[4], (chi * D2, chi)) / (chi * D2)
    C4g = jnp.zeros((chi * D2, chi))
    return T4, a, P1, P2, C1g, C4g


def full_move(T4, a, P1, P2, C1g, C4g, chi, D2):
    # The real left-move edge path (peak = T4_a, χ²·D⁶).
    T4_a = jnp.einsum("ijk,lmjn->iklmn", T4, a)  # (chi, chi, D2, D2, D2)
    T4_a = T4_a.transpose(0, 2, 1, 3, 4)  # (i, l, k, m, n)
    T4g = T4_a.reshape(chi * D2, chi * D2, D2)  # (fl=(i,l), fr=(k,m), n)
    _c1, _c4, T_new = _apply_projector_raw(P1, P2, C1g, C4g, T4g)
    return T_new  # (k_out, D2, k_out)


def chunked_move(T4, a, P1, P2, C1g, C4g, chi, D2, batch):
    # Stream over t4_d=i; each i contributes to the projector contraction; sum.
    P1r = P1.reshape(chi, D2, -1)  # (i, l, k_out)

    def per_i(args):
        T4_i, P1_i = args  # T4_i: (D2, chi) [l2=j, t4_u=k]; P1_i: (D2[l], k_out)
        T4a_i = jnp.einsum("jk,lmjn->klmn", T4_i, a)  # (k, l, m, n) = (chi, D2, D2, D2)
        T4g_i = T4a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (l, fr=(k,m), n)
        step = jnp.tensordot(P1_i.conj(), T4g_i, axes=([0], [0]))  # (k_out, fr, n)
        return jnp.tensordot(step, P2, axes=([1], [0]))  # (k_out, D2, k_out)

    contribs = lax.map(per_i, (T4, P1r), batch_size=batch)  # (chi, k_out, D2, k_out)
    return contribs.sum(0)


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=0, help="fixed chunk size (0 → chi//chunks)")
    ap.add_argument(
        "--variant", default="full", choices=["full", "chunked", "parity"]
    )
    args = ap.parse_args()
    D2 = args.D * args.D
    batch = args.batch if args.batch > 0 else max(1, args.chi // args.chunks)
    T4, a, P1, P2, C1g, C4g = _inputs(args.D, args.chi, 0)
    print(
        f"# ctm-move chunk  D={args.D} chi={args.chi} D2={D2} "
        f"batch={batch} ({args.chi // batch} chunks) x64={jax.config.jax_enable_x64}"
    )

    jfull = jax.jit(lambda: full_move(T4, a, P1, P2, C1g, C4g, args.chi, D2))
    jchunk = jax.jit(lambda: chunked_move(T4, a, P1, P2, C1g, C4g, args.chi, D2, batch))

    if args.variant == "parity":
        a_ = jax.block_until_ready(jfull())
        b_ = jax.block_until_ready(jchunk())
        rel = float(jnp.max(jnp.abs(a_ - b_)) / (jnp.max(jnp.abs(a_)) + 1e-30))
        print(f"D={args.D} parity max|d|={float(jnp.max(jnp.abs(a_ - b_))):.2e} rel={rel:.2e}")
        return
    fn = jfull if args.variant == "full" else jchunk
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        dt = time.perf_counter() - t0
        print(f"D={args.D} variant={args.variant} peak={peak_gb():.2f} GB wall={dt:.3f}s")
    except Exception as ex:  # noqa: BLE001
        print(f"D={args.D} variant={args.variant} FAILED({type(ex).__name__}: {str(ex)[:80]})")


if __name__ == "__main__":
    main()
