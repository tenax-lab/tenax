"""AD-vs-finite-difference gates on the decomposition adjoints (#750, #751, #753).

Primarily the SVD kernel ``_svd_sector_backward``; the sibling
``regularized_eigh`` is covered at the end, because it shared the same complex
cotangent-convention defect (#753).

This is the test that was missing.  Before it, 107 core tests passed
*identically* with and without a bug that put the off-diagonal (F-matrix)
contribution of ``_svd_sector_backward`` at exactly -0.5x its correct value:
nothing compared an SVD-path gradient to a trustworthy independent reference.
The existing gates assert finiteness, direction, determinism, or
``implicit == explicit`` -- and that last one *cancels* the bug, because both
sides share the same primitive.

Finite differences are taken on exactly the gauge-fixed forward being
differentiated, so every sign/phase convention cancels: the comparison is
gauge-free by construction and needs no analytic reference.

Three defects are pinned here:

* **#750** ``F_ij`` index order and a spurious ``1/2`` on the antisymmetric
  projections, together giving ``-0.5x`` on terms 2 and 3.
* **#751a** ``_fix_svd_signs`` scaled ``U[:, j]`` and ``Vh[j, :]`` by the *same*
  ``conj(phase_j)``, multiplying term ``j`` of ``U diag(s) Vh`` by
  ``conj(phase_j)**2``.  That is 1 only for real ``+-1``; for a genuine complex
  phase it destroyed the reconstruction (measured 100% error).
* **#751b** the kernel was stated in the textbook cotangent convention while
  JAX's cotangents for a real-valued loss of a complex input are the conjugate
  of that, and the complex-only Wan & Zhang phase term was missing.

Scope note -- what is deliberately *not* asserted to machine precision:

* **Truncation.**  Terms 4/5 are the ``s_discarded -> 0`` limit of the
  kept/discarded coupling.  The error vanishes with the discarded weight (see
  ``test_truncation_error_vanishes_with_discarded_weight``), so it is a
  controlled approximation, not a defect.  Tracked in #752.
* **Phase-dependent complex losses.**  A loss that reads ``U`` or ``Vh``
  entrywise depends on the gauge that ``_fix_svd_signs`` pins, and the adjoint
  does not differentiate through that gauge fixing.  Every physical loss
  (energies, observables, CTM projectors composed into a fixed point) is
  gauge-invariant, which is what the complex gates below use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ad_primitives import (
    _fix_svd_signs,
    regularized_eigh,
    regularized_svd,
    truncated_svd_ad,
)
from tenax.linalg import _dense_svd


def _fd_grad(fn, M, h=1e-6):
    """Central-difference gradient in JAX's cotangent convention.

    For a real-valued ``fn`` of a complex input JAX's ``grad`` returns
    ``dL/dRe - i dL/dIm``; for a real input it is just ``dL/dRe``.  Calibrated
    against closed-form gradients in ``test_the_fd_harness_is_itself_correct``.
    """
    M = np.asarray(M)
    out = np.zeros_like(M)
    shifts = ((h + 0j, 1.0), (1j * h, -1j)) if np.iscomplexobj(M) else ((h, 1.0),)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for step, mul in shifts:
                Mp, Mm = M.copy(), M.copy()
                Mp[i, j] += step
                Mm[i, j] -= step
                out[i, j] += (
                    mul
                    * (float(fn(jnp.asarray(Mp))) - float(fn(jnp.asarray(Mm))))
                    / (2 * h)
                )
    return out


def _agreement(fn, M):
    """Return ``(relative_error, cosine)`` between AD and FD gradients."""
    g_ad = np.asarray(jax.grad(fn)(jnp.asarray(M)))
    g_fd = _fd_grad(fn, M)
    cos = float(np.vdot(g_ad, g_fd).real) / (
        np.linalg.norm(g_ad) * np.linalg.norm(g_fd)
    )
    rel = float(np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd))
    return rel, cos


def _rng(seed=7):
    return np.random.default_rng(seed)


def _mat(rng, shape, complex_):
    if complex_:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return rng.standard_normal(shape)


# ---------------------------------------------------------------------------
# 0. The harness must be trustworthy before it can judge the kernel.
# ---------------------------------------------------------------------------


def test_the_fd_harness_is_itself_correct():
    """Calibrate ``_fd_grad`` on functions with closed-form gradients.

    Without this, a wrong FD convention reads as a kernel defect.  That trap
    fired during the #750 investigation: an FD built as ``d/dRe + i d/dIm``
    made *JAX's own* SVD VJP look broken (cos = -0.03).
    """
    rng = _rng()
    n = 5
    for complex_ in (False, True):
        M = _mat(rng, (n, n), complex_)
        W = _mat(rng, (n, n), complex_)
        for tag, fn in (
            ("linear", lambda A, W=W: jnp.real(jnp.sum(jnp.conj(W) * A))),
            ("bilinear", lambda A, W=W: jnp.real(jnp.sum(jnp.conj(W) * (A @ A)))),
            ("norm^2", lambda A: jnp.sum(jnp.abs(A) ** 2)),
        ):
            rel, cos = _agreement(fn, M)
            assert rel < 1e-6, (
                f"FD harness wrong on {tag} (complex={complex_}): {rel:.2e}"
            )
            assert cos > 1 - 1e-9, f"FD harness direction wrong on {tag}: {cos}"

    # ...and against a reference SVD VJP that is known-good.
    M = _mat(rng, (n, n), True)
    W = _mat(rng, (n, n), True)

    def ref(A):
        U, s, Vh = jnp.linalg.svd(A, full_matrices=False)
        return jnp.real(jnp.sum(jnp.conj(W) * (U @ jnp.diag(s**2) @ Vh)))

    rel, cos = _agreement(ref, M)
    assert rel < 1e-6 and cos > 1 - 1e-9, (
        f"harness disagrees with jnp SVD VJP: {rel:.2e}"
    )


# ---------------------------------------------------------------------------
# 1. #751a -- the forward gauge fix must preserve the factorisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_fix_svd_signs_preserves_the_reconstruction(complex_):
    """``U diag(s) Vh == M`` must survive gauge fixing.

    The shipped version scaled both ``U[:, j]`` and ``Vh[j, :]`` by
    ``conj(phase_j)``, so the reconstruction picked up ``conj(phase_j)**2``.
    Harmless for real ``+-1`` signs, catastrophic for complex: this assertion
    failed at ``1.1e+01`` on a norm-10 matrix.
    """
    rng = _rng(3)
    n = 6
    M = jnp.asarray(_mat(rng, (n, n), complex_))

    U0, s0, Vh0 = _dense_svd(M, full_matrices=False)
    U, s, Vh = _fix_svd_signs(U0, s0, Vh0)

    err = float(jnp.linalg.norm(U @ jnp.diag(s) @ Vh - M))
    scale = float(jnp.linalg.norm(M))
    assert err / scale < 1e-12, (
        f"gauge fixing broke the SVD: ||U s Vh - M||/||M|| = {err / scale:.3e}"
    )

    # the gauge fix must still be a gauge fix: U, Vh stay isometric
    assert float(jnp.linalg.norm(U.conj().T @ U - jnp.eye(n))) < 1e-10
    assert float(jnp.linalg.norm(Vh @ Vh.conj().T - jnp.eye(n))) < 1e-10


def test_fix_svd_signs_actually_fixes_the_gauge():
    """The point of the helper: max-|U| entry of each column is real-positive."""
    rng = _rng(4)
    n = 6
    M = jnp.asarray(_mat(rng, (n, n), True))
    U, _s, _Vh = _fix_svd_signs(*_dense_svd(M, full_matrices=False))
    lead = U[jnp.argmax(jnp.abs(U), axis=0), jnp.arange(n)]
    assert np.allclose(np.asarray(jnp.imag(lead)), 0.0, atol=1e-12)
    assert np.all(np.asarray(jnp.real(lead)) > 0)


# ---------------------------------------------------------------------------
# 2. #750/#751b -- the adjoint, full rank (no truncation terms in play).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_full_rank_svd_gradient_matches_finite_differences(complex_):
    """The headline gate.  With the -0.5x bug this gave cos ~ -0.57."""
    rng = _rng(11)
    n = 6
    M = _mat(rng, (n, n), complex_)
    W = _mat(rng, (n, n), complex_)

    def loss(A):
        U, s, Vh = regularized_svd(A)
        return jnp.real(jnp.sum(jnp.conj(W) * (U @ jnp.diag(s**2) @ Vh)))

    rel, cos = _agreement(loss, M)
    assert cos > 1 - 1e-9, f"gradient DIRECTION wrong: cos={cos:.9f}"
    assert rel < 1e-6, f"gradient MAGNITUDE wrong: rel={rel:.3e}"


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_singular_value_gradient_matches_finite_differences(complex_):
    """``sum(s)`` exercises term 1 alone -- isolates the conjugation convention."""
    rng = _rng(12)
    n = 6
    M = _mat(rng, (n, n), complex_)
    rel, cos = _agreement(lambda A: jnp.sum(regularized_svd(A)[1]), M)
    assert cos > 1 - 1e-9 and rel < 1e-6, f"rel={rel:.3e} cos={cos:.9f}"


def test_real_gradient_is_exact_for_a_phase_dependent_loss():
    """Reading ``U``/``Vh`` entrywise is fine in the real case.

    The real gauge fix is a discrete ``+-1``, locally constant in ``M``, so it
    contributes nothing to the derivative.  (Its complex counterpart is a
    continuous phase and does contribute -- see the module docstring.)  This is
    the exact configuration reported in #750: ``cos = -0.570746546``.
    """
    rng = _rng(13)
    n = 6
    M = rng.standard_normal((n, n))
    W = rng.standard_normal((n, n))
    Z = rng.standard_normal((n, n))

    def loss(A):
        U, s, Vh = regularized_svd(A)
        return jnp.sum(W * U) + jnp.sum(Z * Vh) + jnp.sum(s)

    rel, cos = _agreement(loss, M)
    assert cos > 1 - 1e-9 and rel < 1e-6, f"rel={rel:.3e} cos={cos:.9f}"


def test_off_diagonal_contribution_is_not_halved():
    """Scale-sensitive gate on terms 2/3 in isolation.

    The ``s`` term is omitted so the loss depends *only* on the off-diagonal
    (``dU``/``dVh``) contribution -- the part #750 put at ``-0.5x``.  A cosine
    gate cannot see a constant factor, so this asserts the magnitude against
    FD, and reports the measured ratio on failure.

    The reference is finite differences, deliberately not ``jnp.linalg.svd``:
    JAX does not gauge-fix, so its ``U``/``Vh`` differ from the gauge-fixed
    ones by a per-column sign and the two gradients answer different questions.
    """
    rng = _rng(14)
    n = 6
    M = rng.standard_normal((n, n))
    W = rng.standard_normal((n, n))
    Z = rng.standard_normal((n, n))

    def loss(A):
        U, _s, Vh = regularized_svd(A)
        return jnp.sum(W * U) + jnp.sum(Z * Vh)

    g_ad = np.asarray(jax.grad(loss)(jnp.asarray(M)))
    g_fd = _fd_grad(loss, M)
    ratio = float(np.vdot(g_fd, g_ad).real / np.vdot(g_fd, g_fd).real)
    rel = float(np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd))
    assert abs(ratio - 1.0) < 1e-5 and rel < 1e-6, (
        f"off-diagonal contribution is {ratio:.9f}x finite differences "
        f"(rel={rel:.3e}); -0.5 is the #750 signature"
    )


# ---------------------------------------------------------------------------
# 3. #752 -- truncation is exact, at any discarded weight.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_truncated_gradient_is_exact_when_nothing_is_discarded(complex_):
    """Baseline: negligible discarded weight, where even the old limit was exact."""
    rng = _rng(15)
    n, k = 6, 3
    Ug, _ = np.linalg.qr(_mat(rng, (n, n), complex_))
    Vg, _ = np.linalg.qr(_mat(rng, (n, n), complex_))
    M = Ug @ np.diag([4.0, 3.0, 2.0, 1e-7, 5e-8, 2e-8]) @ Vg.conj().T
    W = _mat(rng, (n, n), complex_)

    def loss(A):
        U, s, Vh = truncated_svd_ad(A, k)
        return jnp.real(jnp.sum(jnp.conj(W) * (U @ jnp.diag(s**2) @ Vh)))

    rel, cos = _agreement(loss, M)
    assert cos > 1 - 1e-9 and rel < 1e-6, f"rel={rel:.3e} cos={cos:.9f}"


@pytest.mark.parametrize("tail", [2e-1, 2e-2, 2e-3])
def test_truncated_gradient_is_exact_at_any_discarded_weight(tail):
    """#752: the truncated adjoint is exact, not a small-discarded-weight limit.

    Terms 4/5 used to handle the kept/discarded coupling with weight
    ``1/s_kept``, the ``s_discarded -> 0`` limit of the exact
    ``s_j / (s_j^2 - s_r^2)``.  Its error tracked the discarded weight -- 2.2e-02
    at ``s_disc/s_kept = 0.1``, falling ~10x per decade -- so it was a genuine
    approximation, and a test could only assert that it *shrank*.

    Zero-padding the cotangents onto the full spectrum makes the adjoint exact
    instead, because the truncated factors are a slice of the full SVD.  The
    error is now flat at the finite-difference noise floor (~2.4e-09) across
    this whole sweep, so the assertion is exactness at every point.

    The largest ``tail`` here discards 10% of the leading singular value -- well
    past where the old limit was defensible.
    """
    rng = _rng(16)
    n, k = 6, 3
    Ug, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Vg, _ = np.linalg.qr(rng.standard_normal((n, n)))
    W = rng.standard_normal((n, n))

    def loss(A):
        U, s, Vh = truncated_svd_ad(A, k)
        return jnp.sum(W * (U @ jnp.diag(s**2) @ Vh))

    M = Ug @ np.diag([4.0, 3.0, 2.0, tail, tail / 2, tail / 4]) @ Vg.T
    rel, cos = _agreement(loss, M)
    assert cos > 1 - 1e-9, f"direction lost at discarded={tail}: cos={cos:.9f}"
    assert rel < 1e-6, (
        f"truncated adjoint is not exact at discarded={tail}: rel={rel:.3e} "
        "-- the s_discarded -> 0 approximation has come back (#752)"
    )


def test_truncated_gradient_is_exact_for_non_square_matrices():
    """Padding must not disturb the genuine null space (``m > p``).

    After padding, terms 4/5 no longer carry the kept/discarded coupling -- they
    carry only the true null space, which exists solely for tall inputs.  Both
    orientations are checked so a null-space regression cannot hide.
    """
    rng = _rng(17)
    for m, n in ((8, 5), (5, 8)):
        k = 3
        M = rng.standard_normal((m, n))
        W = rng.standard_normal((m, n))

        def loss(A, W=W, k=k):
            U, s, Vh = truncated_svd_ad(A, k)
            return jnp.sum(W * (U @ jnp.diag(s**2) @ Vh))

        rel, cos = _agreement(loss, M)
        assert cos > 1 - 1e-9 and rel < 1e-6, f"({m}x{n}) rel={rel:.3e} cos={cos:.9f}"


def test_truncated_gradient_is_finite_on_rank_deficient_input():
    """Exact zeros in the spectrum must stay finite, not divide by sigma=0.

    The rank mask covers kept-but-zero columns, whose cotangents are gauge-
    arbitrary.  Discarded indices are deliberately left unmasked -- their padded
    cotangents are zero, so they inject nothing -- and this pins that the choice
    does not produce NaN/inf through the 1/(s_j^2 - s_r^2) entries.
    """
    rng = _rng(18)
    n = 6
    Ug, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Vg, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = Ug @ np.diag([4.0, 3.0, 0.0, 0.0, 0.0, 0.0]) @ Vg.T
    W = rng.standard_normal((n, n))

    def loss(A):
        U, s, Vh = truncated_svd_ad(A, 4)
        return jnp.sum(W * (U @ jnp.diag(s**2) @ Vh))

    g = np.asarray(jax.grad(loss)(jnp.asarray(M)))
    assert np.all(np.isfinite(g)), "rank-deficient input produced a non-finite gradient"


# ---------------------------------------------------------------------------
# 4. #753 -- the sibling eigh adjoint shares the complex convention defect.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_eigh_eigenvector_gradient_matches_finite_differences(complex_):
    """``regularized_eigh`` on an eigen*vector*-dependent loss.

    The F-matrix orientation here was already corrected in #316, so the real
    path was fine.  Complex Hermitian input was not: it needed the same
    conj-in/conj-out bridge as the SVD kernel, and gave cos = 0.122 without it.
    """
    rng = _rng(21)
    n = 6
    A = _mat(rng, (n, n), complex_)
    H = A + A.conj().T
    W = rng.standard_normal((n, n))
    Wh = jnp.asarray(W + W.T)

    def loss(M):
        w, v = regularized_eigh(M)
        return jnp.real(jnp.sum(Wh * (v @ jnp.diag(w**2) @ v.conj().T)))

    rel, cos = _agreement(loss, H)
    assert cos > 1 - 1e-9 and rel < 1e-6, f"rel={rel:.3e} cos={cos:.9f}"


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_eigh_eigenvalue_gradient_matches_finite_differences(complex_):
    """``sum(w)`` -- passed even *with* the #753 bug, so it cannot stand alone.

    Its gradient is the identity, which is real, so a missing conjugation is
    invisible.  Kept as the companion that shows why the eigenvector test above
    is the one that actually gates the convention.
    """
    rng = _rng(22)
    n = 6
    A = _mat(rng, (n, n), complex_)
    H = A + A.conj().T
    rel, cos = _agreement(lambda M: jnp.sum(regularized_eigh(M)[0]), H)
    assert cos > 1 - 1e-9 and rel < 1e-6, f"rel={rel:.3e} cos={cos:.9f}"
