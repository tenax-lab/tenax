"""SPIKE (#570): is the block-sparse QR backward stable enough to drop into the
2x2 projector? Checks grads on well-conditioned AND near-rank-deficient sectors.

Run: JAX_PLATFORMS=cpu uv run python examples/probe_qr_vjp_stability_570.py
"""
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)


def _qr_reduce(M):
    Q, R = jnp.linalg.qr(M)
    d = jnp.diag(R)
    phase = jnp.where(jnp.abs(d) > 0, d / jnp.where(jnp.abs(d) > 0, jnp.abs(d), 1.0), 1.0)
    Q = Q * phase[None, :]
    return Q


def _make_rank_deficient(key, n, drop):
    A = jax.random.normal(key, (n, n))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[n - drop:].set(0.0)
    return (U * s) @ Vh


def main():
    key = jax.random.PRNGKey(0)
    for label, M in [
        ("well-conditioned 12x12", jax.random.normal(key, (12, 12))),
        ("tall 16x8", jax.random.normal(key, (16, 8))),
        ("near-rank-deficient 12x12", _make_rank_deficient(key, 12, drop=4)),
    ]:
        try:
            check_grads(_qr_reduce, (M,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
            print(f"PASS  {label}")
        except Exception as e:
            print(f"FAIL  {label}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
