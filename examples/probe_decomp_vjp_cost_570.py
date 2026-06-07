#!/usr/bin/env python3
"""Isolated decomposition-VJP compile-cost: SVD vs QR vs eigh (the #570 premise).

`probe_bwd_subop_attribution_570.py` showed the CTM-AD compile wall is the
block-sparse **SVD VJP** (61% of the fused backward at D=4/χ=12, the only
χ-scaling term). And `_ctm_tensor_projector_2x2.py` (the compile-dominant
symmetric single-site projector) hardcodes SVD — so #570's "QR projector" lever
is NOT a config flip; it must be *implemented* there. Before paying that, test
the PREMISE in isolation:

    Is a QR / eigh decomposition VJP actually cheaper to compile than the SVD VJP,
    and how does each scale with matrix size n (= per-sector χ)?

This traces the **backward** (`jax.vjp`) jaxpr of each decomposition on a dense
n×n matrix and counts ops + decomposition kernels, sweeping n. Two things matter:
  (1) the constant factor (ops at fixed n) — the per-sector compile cost;
  (2) the n-scaling — whether per-sector matrix SIZE drives op count (vs the
      separate per-SECTOR multiplicity that drives the real backward's χ-scaling).

Decompositions compared (all reduced to a comparable "keep top-k subspace" use,
loss = ‖reconstruct/subspace‖² so the VJP is exercised):
  * svd_prod : `truncated_svd_ad` (the PRODUCTION SVD VJP, gauge/Lorentzian);
  * svd_plain: `jnp.linalg.svd` (baseline);
  * qr       : `jnp.linalg.qr` (the Yang/Corboz-style lever);
  * eigh_prod: `regularized_eigh` on MᴴM (the PRODUCTION eigh VJP).

Usage::

    JAX_PLATFORMS=cpu uv run python examples/probe_decomp_vjp_cost_570.py \
        --n 8 16 32 64
"""

from __future__ import annotations

import argparse
import collections

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ad_primitives import regularized_eigh  # noqa: E402
from tenax.algorithms.ad_utils import truncated_svd_ad  # noqa: E402

_KERNEL_PRIMS = (
    "svd",
    "eigh",
    "qr",
    "geqrf",
    "householder_product",
    "triangular_solve",
    "lu",
    "eig",
    "custom_linear_solve",
)


def _count(jaxpr) -> tuple[int, collections.Counter]:
    """(total ops, kernel-primitive counts), recursing into sub-jaxprs."""
    total = 0
    kernels: collections.Counter = collections.Counter()
    jpr = getattr(jaxpr, "jaxpr", jaxpr)

    def rec(jx):
        nonlocal total
        for eqn in jx.eqns:
            total += 1
            if eqn.primitive.name in _KERNEL_PRIMS:
                kernels[eqn.primitive.name] += 1
            for vparam in eqn.params.values():
                items = vparam if isinstance(vparam, (list, tuple)) else (vparam,)
                for it in items:
                    sub = getattr(it, "jaxpr", it)
                    if hasattr(sub, "eqns"):
                        rec(sub)

    rec(jpr)
    return total, kernels


def _vjp_ops(loss_fn, M) -> tuple[int, collections.Counter]:
    """Count the backward (cotangent) jaxpr ops of ``loss_fn`` at ``M``."""
    jx = jax.make_jaxpr(lambda m: jax.vjp(loss_fn, m)[1](1.0))(M)
    return _count(jx)


def make_losses(k: int):
    """Real-valued losses that exercise each decomposition's top-k subspace VJP."""

    def svd_prod(M):
        U, S, Vh = truncated_svd_ad(M, k)
        return jnp.sum(jnp.abs(S) ** 2) + jnp.real(jnp.sum(jnp.abs(U) ** 2))

    def svd_plain(M):
        U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
        return jnp.sum(S[:k] ** 2) + jnp.real(jnp.sum(jnp.abs(U[:, :k]) ** 2))

    def qr(M):
        Q, R = jnp.linalg.qr(M, mode="reduced")
        # Top-k "subspace" use: diag of R (analogue of singular values) + Q cols.
        return jnp.sum(jnp.abs(jnp.diagonal(R)[:k]) ** 2) + jnp.real(
            jnp.sum(jnp.abs(Q[:, :k]) ** 2)
        )

    def eigh_prod(M):
        # Density matrix MᴴM -> Hermitian eigh (the eigh-projector construction).
        w, v = regularized_eigh(M.conj().T @ M)
        return jnp.sum(jnp.abs(w[-k:]) ** 2) + jnp.real(
            jnp.sum(jnp.abs(v[:, -k:]) ** 2)
        )

    return {
        "svd_prod": svd_prod,
        "svd_plain": svd_plain,
        "qr": qr,
        "eigh_prod": eigh_prod,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument(
        "--k-frac",
        type=float,
        default=0.5,
        help="keep k = round(k_frac * n) (truncation ratio).",
    )
    args = ap.parse_args()

    print(
        f"# #570 isolated decomposition-VJP op cost | x64={jax.config.read('jax_enable_x64')}"
    )
    print(
        f"# backward (jax.vjp) jaxpr ops on a dense n×n matrix | k = {args.k_frac}·n\n"
    )

    methods = ["svd_prod", "svd_plain", "qr", "eigh_prod"]
    print(f"{'n':>5} {'k':>4} " + " ".join(f"{m:>22}" for m in methods))
    key = jax.random.PRNGKey(0)
    for n in args.n:
        k = max(1, round(args.k_frac * n))
        M = jax.random.normal(key, (n, n))
        losses = make_losses(k)
        cells = []
        for m in methods:
            try:
                total, kern = _vjp_ops(losses[m], M)
                ksum = sum(kern.values())
                cells.append(f"{total:>6} ops/{ksum}k")
            except Exception as e:  # pragma: no cover - diagnostic
                cells.append(f"ERR:{type(e).__name__}")
        print(f"{n:>5} {k:>4} " + " ".join(f"{c:>22}" for c in cells))

    print("\n# READOUT:")
    print(
        "#  - Compare svd_prod vs qr / eigh_prod at fixed n: the per-sector "
        "compile-cost ratio of the #570 lever."
    )
    print(
        "#  - Compare the n-scaling: if svd_prod ops are ~FLAT in n, the real "
        "backward's χ-scaling is per-SECTOR multiplicity (a #566 representation"
    )
    print(
        "#    issue), and a cheaper per-sector decomp (QR) lowers the CONSTANT "
        "but not the scaling. If svd_prod GROWS with n, size matters too."
    )


if __name__ == "__main__":
    main()
