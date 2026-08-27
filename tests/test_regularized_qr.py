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
    s = s.at[8:].set(0.0)  # 4 exactly-zero singular values
    M = (U * s) @ Vh
    g = jax.grad(_scalar)(M)
    assert jnp.all(jnp.isfinite(g))


# --------------------------------------------------------------------------- #
# Wide matrices (m < n) -- #912.                                               #
#                                                                              #
# The VJP above was written for thin QR with ``m >= n`` and its comment said   #
# so ("verified ... for square AND tall real matrices", #570).  Nothing        #
# enforced that, and the QR projector feeds it a wide matrix: it concatenates  #
# the two enlarged corners side by side, and on the C4v path both blocks are   #
# the same square ``(chi*D^2, chi*D^2)`` corner, so ``M`` is wide by exactly   #
# 2x at every chi.  ``R`` is then ``(m, n)`` while ``jnp.diag(jnp.diag(R))``   #
# is ``(m, m)`` and the backward died with a broadcasting ``TypeError``.       #
# --------------------------------------------------------------------------- #


def _ref_qr_tuple(A):
    q, r = jnp.linalg.qr(A)
    return (q, r)


@pytest.mark.core
@pytest.mark.parametrize(
    ("m", "n"),
    [
        (4, 8),
        (4, 12),
        (6, 18),
        # The shapes the C4v projector actually produces, at chi=4/8/12 with
        # D=2: m = chi*D^2, n = 2*chi*D^2.  Pinned concretely so a future
        # change to the projector's concat layout shows up here.
        (16, 32),
        (32, 64),
        (48, 96),
    ],
)
def test_regularized_qr_backward_matches_jax_on_wide_matrices(m, n):
    """Wide-``M`` VJP parity against JAX's own QR VJP.

    JAX's ``jnp.linalg.qr`` differentiates wide matrices perfectly well, so it
    is a valid oracle here -- the gap was ours alone.
    """
    rng = np.random.default_rng(m * 1000 + n)
    M = jnp.asarray(rng.standard_normal((m, n)))
    cot = (
        jnp.asarray(rng.standard_normal((m, min(m, n)))),
        jnp.asarray(rng.standard_normal((min(m, n), n))),
    )
    _, vjp_reg = jax.vjp(regularized_qr, M)
    _, vjp_raw = jax.vjp(_ref_qr_tuple, M)
    got, want = vjp_reg(cot)[0], vjp_raw(cot)[0]
    scale = float(jnp.max(jnp.abs(want))) + 1e-300
    assert float(jnp.max(jnp.abs(got - want))) / scale < 1e-10, (
        f"wide {m}x{n}: relative deviation "
        f"{float(jnp.max(jnp.abs(got - want))) / scale:.3e}"
    )


@pytest.mark.core
def test_regularized_qr_wide_backward_needs_the_R2_cotangent_path():
    """The ``Q``-bar correction is load-bearing, and its absence is *silent*.

    For wide ``M = [M1 | M2]``, ``Q`` feeds ``R2 = Q^H M2`` as well as ``R1``,
    so ``Q``-bar must pick up ``M2 R2-bar^H``.  A wide rule that only routed
    ``R2``-bar into ``M2``-bar returns finite, plausible numbers that are
    ~75% wrong -- no crash, no NaN, nothing downstream can tell.  This test
    fails (relative deviation 0.66-1.23) if that term is dropped, which is
    what makes the parity test above worth having rather than decorative.
    """
    rng = np.random.default_rng(912)
    m, n = 16, 32
    M = jnp.asarray(rng.standard_normal((m, n)))
    cot = (
        jnp.asarray(rng.standard_normal((m, m))),
        jnp.asarray(rng.standard_normal((m, n))),
    )
    _, vjp_reg = jax.vjp(regularized_qr, M)
    got = vjp_reg(cot)[0]

    # Independent reconstruction of the wide rule, spelled out rather than
    # imported, so this does not merely restate the implementation.
    Q, R = jnp.linalg.qr(M)
    R1, R2 = R[:, :m], R[:, m:]
    dR1, dR2 = cot[1][:, :m], cot[1][:, m:]
    dQ_eff = cot[0] + Q @ (R2 @ dR2.conj().T)
    P = dR1 @ R1.conj().T
    S = Q.conj().T @ dQ_eff
    under = S - P
    Bbar = (P - S) + jnp.tril(under - under.conj().T, -1)
    Abar = dQ_eff + Q @ Bbar
    dM1 = jax.scipy.linalg.solve_triangular(R1, Abar.conj().T, lower=False).conj().T
    want = jnp.concatenate([dM1, Q @ dR2], axis=1)

    scale = float(jnp.max(jnp.abs(want))) + 1e-300
    assert float(jnp.max(jnp.abs(got - want))) / scale < 1e-10


@pytest.mark.core
def test_regularized_qr_wide_stays_finite_at_rank_deficiency():
    """The regularization must survive into the wide branch, not just square.

    This is ``regularized_qr``'s entire reason to exist over ``jnp.linalg.qr``,
    so a wide branch that quietly lost the ``diag(R)`` floor would be a
    regression the parity tests above cannot see (they use well-conditioned
    inputs, where the floor never activates).
    """
    rng = np.random.default_rng(5)
    M = np.asarray(rng.standard_normal((16, 32)))
    M[8:, :] = 0.0  # kill half the row space -> exactly singular R1
    M = jnp.asarray(M)
    cot = (
        jnp.asarray(rng.standard_normal((16, 16))),
        jnp.asarray(rng.standard_normal((16, 32))),
    )
    _, vjp_reg = jax.vjp(regularized_qr, M)
    assert jnp.all(jnp.isfinite(vjp_reg(cot)[0])), "wide backward went non-finite"

    # And confirm the guard is not vacuous: raw JAX really does NaN here.
    _, vjp_raw = jax.vjp(_ref_qr_tuple, M)
    assert not jnp.all(jnp.isfinite(vjp_raw(cot)[0])), (
        "raw jnp.linalg.qr is finite on this input, so it does not exercise "
        "the floor and this test proves nothing -- pick a more singular M"
    )
