"""Tests for ``regularized_qr`` — a thin-QR custom-VJP whose backward stays
finite through near/exactly rank-deficient matrices (#570 Phase 2 Task 2).

Validation methodology (established by the Task 1 spike): finite differences
CANNOT validate the near-deficient regime (the true gradient norm ~1/sigma is
too steep for central FD, and at exact rank-deficiency the QR derivative does
not exist). So correctness is checked against JAX's own analytic QR VJP where
the floor does not engage, FD is used only on a well-conditioned input, and the
rank-deficient case asserts only finiteness (the floor's actual job).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ad_primitives import regularized_qr


def _scalar(M):
    Q, R = regularized_qr(M)
    return jnp.real(jnp.sum(Q) + jnp.sum(R))


@pytest.mark.core
def test_regularized_qr_forward_matches_plain_qr():
    M = jax.random.normal(jax.random.PRNGKey(1), (10, 6))
    Q, R = regularized_qr(M)
    Q0, R0 = jnp.linalg.qr(M)
    np.testing.assert_allclose(Q, Q0, atol=1e-12)
    np.testing.assert_allclose(R, R0, atol=1e-12)


@pytest.mark.core
def test_regularized_qr_grads_well_conditioned_vs_fd():
    # Well-conditioned: FD is valid here.
    M = jax.random.normal(jax.random.PRNGKey(0), (12, 8))
    check_grads(_scalar, (M,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


@pytest.mark.core
def test_regularized_qr_backward_matches_analytic_vjp_well_conditioned():
    # The regularized backward must EQUAL JAX's analytic QR VJP where the floor
    # does not engage (well-conditioned). This is the precise correctness check.
    M = jax.random.normal(jax.random.PRNGKey(2), (12, 8))
    Q, R = jnp.linalg.qr(M)
    cot = (jnp.ones_like(Q), jnp.ones_like(R))
    # ``jnp.linalg.qr`` returns a ``QRResult`` namedtuple; wrap it so its
    # output tree matches the plain-tuple cotangent (mirrors the spike's
    # ``_ref_qr``). This still uses JAX's own analytic QR VJP as the ground
    # truth — only the container type is normalized.
    def _ref_qr(A):
        q, r = jnp.linalg.qr(A)
        return (q, r)

    _, vjp_reg = jax.vjp(regularized_qr, M)
    _, vjp_raw = jax.vjp(_ref_qr, M)
    np.testing.assert_allclose(vjp_reg(cot)[0], vjp_raw(cot)[0], atol=1e-10)


@pytest.mark.core
def test_regularized_qr_backward_finite_at_rank_deficiency():
    # Exactly rank-deficient: raw QR VJP would NaN; the floor keeps ours finite.
    A = jax.random.normal(jax.random.PRNGKey(3), (12, 12))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[8:].set(0.0)          # 4 exactly-zero singular values
    M = (U * s) @ Vh
    g = jax.grad(_scalar)(M)
    assert jnp.all(jnp.isfinite(g))
