"""Spike: is the reduced-corner QR projector more GPU-friendly than the SVD
projector at LARGE χ? (revisiting #570 in the regime #570 didn't test.)

Isolates the projector decomposition on a tall-skinny reduced-corner matrix
M:(χD², χ): SVD path (top-χ left singular vectors) vs reduced-corner QR path
(concat both corners → thin QR → tiny 2χ×2χ eigh → P=Q@V). Times forward AND the
VJP (the #570 wall) at increasing χ on GPU. Same O(χ³D²) FLOPs; QR is direct
Householder vs SVD's iterative bidiagonalization → expect QR more GPU-friendly.

    CUDA_VISIBLE_DEVICES=0 uv run python examples/spike_qr_vs_svd_projector.py --chi 64 128 256 --D 8
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from tenax.algorithms._ad_primitives import regularized_qr  # noqa: E402


def svd_projector(M, chi):  # M: (fused=χD², χ)
    U, _s, _Vh = jnp.linalg.svd(M, full_matrices=False)
    return U[:, :chi]  # (fused, χ) isometry


def qr_projector(M1, M2, chi):  # reduced-corner QR (both corners)
    M = jnp.concatenate([M1, M2], axis=1)  # (fused, 2χ)
    Q, R = regularized_qr(M)  # thin QR; AD-safe
    rho = R @ R.conj().T  # (2χ, 2χ)
    rho = 0.5 * (rho + rho.conj().T)
    _e, V = jnp.linalg.eigh(rho)
    return Q @ V[:, -chi:][:, ::-1]  # (fused, χ)


def _warm(fn, *a, reps=8):
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--D", type=int, default=8)
    args = ap.parse_args()
    D2 = args.D * args.D
    print(f"# qr-vs-svd projector  D={args.D} D2={D2} x64={jax.config.jax_enable_x64}")
    print(f"# {'chi':>5} {'fused=χD²':>10} | {'svd_fwd':>9} {'qr_fwd':>9} {'fwd↑':>5} | "
          f"{'svd_bwd':>9} {'qr_bwd':>9} {'bwd↑':>5}  (ms; ↑ = svd/qr speedup)")
    for chi in args.chi:
        fused = chi * D2
        k = jax.random.split(jax.random.PRNGKey(chi), 2)
        M1 = jax.random.normal(k[0], (fused, chi)) / fused
        M2 = jax.random.normal(k[1], (fused, chi)) / fused
        f_svd = jax.jit(lambda M: svd_projector(M, chi))
        f_qr = jax.jit(lambda M1, M2: qr_projector(M1, M2, chi))
        g_svd = jax.jit(jax.grad(lambda M: svd_projector(M, chi).sum()))
        g_qr = jax.jit(jax.grad(lambda M1, M2: qr_projector(M1, M2, chi).sum()))
        try:
            sf, qf = _warm(f_svd, M1), _warm(f_qr, M1, M2)
            sb, qb = _warm(g_svd, M1), _warm(g_qr, M1, M2)
            print(f"  {chi:>5} {fused:>10} | {sf:>9.2f} {qf:>9.2f} {sf/qf:>5.2f} | "
                  f"{sb:>9.2f} {qb:>9.2f} {sb/qb:>5.2f}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {chi:>5} {fused:>10} | FAILED({type(ex).__name__}: {str(ex)[:50]})")


if __name__ == "__main__":
    main()
