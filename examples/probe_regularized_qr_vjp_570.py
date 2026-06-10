"""SPIKE (#570 Phase 2): a QR custom-VJP stable near rank-deficiency.
Run: JAX_PLATFORMS=cpu uv run python examples/probe_regularized_qr_vjp_570.py
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)
_R_FLOOR = 1e-12


def _H(X):
    return X.conj().T


@partial(jax.custom_vjp)
def regularized_qr(M):
    return jnp.linalg.qr(M)


def _fwd(M):
    Q, R = jnp.linalg.qr(M)
    return (Q, R), (Q, R)


def _bwd(residuals, g):
    # Reverse-mode VJP of the thin QR M = Q R (m >= n), obtained by transposing
    # JAX's own thin-QR JVP rule (real branch). Verified to machine precision
    # against jax.vjp(jnp.linalg.qr, .) for square AND tall real matrices.
    #
    #   P      = R̄ Rᴴ
    #   S      = Qᴴ Q̄
    #   under  = S − P                       (= Qᴴ Q̄ − R̄ Rᴴ)
    #   B̄      = (P − S) + tril(under − underᴴ, −1)
    #   Ā      = Q̄ + Q B̄
    #   M̄      = Ā R⁻ᴴ          ← regularized triangular solve (floor diag R)
    #
    # The R⁻ᴴ solve is the only place diag(R)→0 bites; flooring diag(R) there
    # keeps M̄ finite near rank-deficiency without changing the well-conditioned
    # answer (floor only activates when |diag(R)| < _R_FLOOR).
    Q, R = residuals
    dQ, dR = g
    P = dR @ _H(R)
    S = _H(Q) @ dQ
    under = S - P
    Bbar = (P - S) + jnp.tril(under - _H(under), -1)
    Abar = dQ + Q @ Bbar
    d = jnp.diag(R)
    safe = jnp.where(jnp.abs(d) > _R_FLOOR, d, _R_FLOOR)
    R_reg = R - jnp.diag(d) + jnp.diag(safe)
    # M̄ = Ā R⁻ᴴ  ⟺  M̄ Rᴴ = Ā  ⟺  R M̄ᴴ = Āᴴ  (upper-tri solve in R).
    dM = _H(jax.scipy.linalg.solve_triangular(R_reg, _H(Abar), lower=False))
    return (dM,)


regularized_qr.defvjp(_fwd, _bwd)


def _rank_deficient(key, n, drop, sv=1e-6):
    """n×n matrix with `drop` tiny (≈sv) singular values.

    NOTE: sv must be > _R_FLOOR and large enough that finite differences can
    still resolve the (large but finite) gradient. At EXACTLY zero singular
    values the QR derivative does not exist (a kink: the sign of the null
    columns of Q is undetermined), so finite-difference cannot validate it and
    check_grads is meaningless there — see the exact-singular stability check
    below, which instead asserts the regularized backward stays finite.
    """
    A = jax.random.normal(key, (n, n))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[n - drop:].set(sv)
    return (U * s) @ Vh


def _scalar(M):
    Q, R = regularized_qr(M)
    return jnp.real(jnp.sum(Q) + jnp.sum(R))


def _ref_qr(M):
    Q, R = jnp.linalg.qr(M)
    return (Q, R)


def _vjp_matches_analytic(M, rtol=1e-10):
    """RIGOROUS validation: our custom VJP must equal JAX's own analytic QR VJP.

    This is a stronger and more meaningful check than check_grads-vs-finite-
    differences in the near-rank-deficient regime: there the true gradient is
    large-but-finite (norm ~1/sv), and central finite differences cannot
    resolve it (FD truncation error ∝ gradient curvature), so check_grads
    spuriously 'fails' even when the backward is exact. Comparing against the
    analytic VJP sidesteps the FD-resolution wall."""
    Q, R = jnp.linalg.qr(M)
    ct = (jnp.ones_like(Q), jnp.ones_like(R))
    (g_ref,) = jax.vjp(_ref_qr, M)[1](ct)
    (g_ours,) = jax.vjp(regularized_qr, M)[1](ct)
    rel = jnp.linalg.norm(g_ours - g_ref) / (jnp.linalg.norm(g_ref) + 1e-30)
    return bool(jnp.isfinite(g_ours).all()) and float(rel) < rtol


def _exact_singular_is_finite(key):
    """The whole point of the floor: at EXACT rank-deficiency the backward must
    return FINITE (non-NaN/Inf) gradients rather than blow up. The QR
    derivative does not exist here (the null columns of Q have undetermined
    sign), so there is no 'correct' value to match — only finiteness."""
    A = jax.random.normal(key, (12, 12))
    U, s, Vh = jnp.linalg.svd(A)
    M = (U * s.at[8:].set(0.0)) @ Vh  # 4 exact-zero singular values
    g = jax.grad(_scalar)(M)
    return bool(jnp.all(jnp.isfinite(g)))


def main():
    key = jax.random.PRNGKey(0)

    # (1) Well-conditioned + tall: validate against finite differences (FD is
    #     reliable here because the gradient is O(1)).
    for label, M in [
        ("well-conditioned 12x12", jax.random.normal(key, (12, 12))),
        ("tall 16x8", jax.random.normal(key, (16, 8))),
    ]:
        try:
            check_grads(_scalar, (M,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
            print(f"PASS  {label}  [check_grads vs FD]")
        except Exception as e:
            print(f"FAIL  {label}: {type(e).__name__}: {str(e)[:140]}")

    # (2) Near-rank-deficient: validate against JAX's analytic VJP (FD cannot
    #     resolve the ~1/sv gradient). Sweep sv down to just above the floor.
    for sv in (1e-2, 1e-4, 1e-6, 1e-9):
        M = _rank_deficient(key, 12, 4, sv=sv)
        ok = _vjp_matches_analytic(M)
        print(f"{'PASS' if ok else 'FAIL'}  near-rank-deficient 12x12 (sv={sv:.0e})"
              f"  [VJP == JAX analytic VJP, finite]")

    # (3) Exact rank-deficiency: only finiteness is well-defined.
    ok = _exact_singular_is_finite(key)
    print(f"{'PASS' if ok else 'FAIL'}  exact-singular backward is finite "
          f"(floor prevents NaN/Inf)")


if __name__ == "__main__":
    main()
