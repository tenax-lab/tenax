"""The 2x2 projector's SVD gauge must not depend on how an exact tie rounds (#841).

The double-corner matrices ``M1``/``M2`` of the 2x2 CTM projector carry an
EXACT ket<->bra swap symmetry on their fused ``D^2`` leg, with a +-1 sign on
the chi leg: ``S M T = M`` for signed permutations ``S`` (rows) and ``T``
(columns), measured to 3e-16 on the production fixtures.  Every left singular
vector is then an eigenvector of ``S`` (``S u = +-u``), so ``|u_i| = |u_{Pi}|``
EXACTLY for every row pair ``(i, Pi)`` the swap exchanges -- and, for half the
vectors, those two entries have opposite sign.

The old gauge rule made "the row of largest ``|U|``" real-positive.  On such a
tie the argmax is decided by the last bit of the SVD, so the column sign was a
coin flip per call: the CTM forward never reached an element-wise fixed point,
the #841 stationarity guard fired, and the flowing implicit adjoint had no
solution.

These tests build a matrix with exactly that symmetry and require the gauge-
fixed ``U`` of ``M`` and of a 1e-15, symmetry-preserving perturbation of ``M``
to agree column for column.  They also assert the regime -- that the fixture
really does contain an opposite-sign exact tie at a column's largest entry --
so they cannot go vacuous if the construction drifts.

A second property pins WHERE the fix may read the phase: the input
environment's chi legs carry a +-1 gauge per chi index, and the output column
sign must copy exactly one of those signs (the chi group of the largest
entry), as the old rule did.  A whole-column overlap satisfies the tie test but
not this one, and locked the dense D=2 chi=6 forward into a period-2 cycle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_projector_2x2 import (  # noqa: E402
    _gauge_fix_symmetric_svd,
    _gauge_fixed_svd,
)
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import SymmetricTensor  # noqa: E402

N_CHI, D = 5, 3  # rows = (chi a, ket k, bra b): 45; ket<->bra swap k <-> b
N_COL = 45
PERTURBATION_SEEDS = range(8)


def _row_swap(n_chi: int, d: int, tau: np.ndarray):
    """Signed permutation ``S`` on rows ``(a, k, b)``: ``(a, k, b) -> tau_a (a, b, k)``."""
    perm = np.array(
        [
            a * d * d + b * d + k
            for a in range(n_chi)
            for k in range(d)
            for b in range(d)
        ]
    )
    sign = np.repeat(tau, d * d).astype(float)
    return perm, sign


def _apply_S(X, perm, sign):
    """``(S X)_i = sign_i X_{perm_i}`` -- an involution (``perm`` swaps, ``tau^2=1``)."""
    return sign[:, None] * X[perm]


def _swap_symmetric(M0, perm, sign, col_sign):
    """``M = M0 + S M0 T``, so ``S M T = M`` (``S`` and ``T`` are involutions)."""
    return M0 + _apply_S(M0, perm, sign) * col_sign[None, :]


def _assert_opposite_sign_tie(U, perm, sign):
    """Guard the regime: some column's max-|U| row ties EXACTLY with its swap
    partner at the opposite sign -- the configuration the argmax rule cannot
    resolve.  Without it the test below proves nothing about #841."""
    U = np.asarray(U)
    hits = 0
    for j in range(U.shape[1]):
        col = U[:, j]
        i = int(np.argmax(np.abs(col)))
        p = int(perm[i])
        if p == i:
            continue
        if (
            abs(abs(col[p]) - abs(col[i])) <= 1e-12 * abs(col[i])
            and np.real(col[p] * np.conj(col[i])) < 0
        ):
            hits += 1
    assert hits >= 3, f"fixture lost its opposite-sign exact ties ({hits} columns)"


def _dense_fixture(complex_: bool):
    rng = np.random.default_rng(841)
    tau = rng.choice([-1.0, 1.0], size=N_CHI)
    col_sign = rng.choice([-1.0, 1.0], size=N_COL)
    perm, sign = _row_swap(N_CHI, D, tau)
    M0 = rng.standard_normal((N_CHI * D * D, N_COL))
    if complex_:
        M0 = M0 + 1j * rng.standard_normal(M0.shape)
    M = _swap_symmetric(M0, perm, sign, col_sign)
    assert np.max(np.abs(_apply_S(M, perm, sign) * col_sign[None, :] - M)) == 0.0
    return M, perm, sign, col_sign


def _perturbed(M, perm, sign, col_sign, seed):
    rng = np.random.default_rng(10_000 + seed)
    E = rng.standard_normal(M.shape)
    if np.iscomplexobj(M):
        E = E + 1j * rng.standard_normal(M.shape)
    E = 0.5 * _swap_symmetric(E, perm, sign, col_sign)
    return M + 1e-15 * np.max(np.abs(M)) * E


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_dense_gauge_is_stable_under_swap_symmetric_rounding(complex_):
    M, perm, sign, col_sign = _dense_fixture(complex_)
    U, s, Vh = _gauge_fixed_svd(jnp.asarray(M), swap_block=D * D)
    s = np.asarray(s)
    # Non-degenerate spectrum, so the gauge is a per-column phase only.
    assert np.min(np.abs(np.diff(s))) > 1e-6 * s[0]
    _assert_opposite_sign_tie(U, perm, sign)
    np.testing.assert_allclose(
        np.asarray(U) * s[None, :] @ np.asarray(Vh), M, atol=1e-10 * np.abs(M).max()
    )
    for seed in PERTURBATION_SEEDS:
        Mp = _perturbed(M, perm, sign, col_sign, seed)
        Up, _, Vhp = _gauge_fixed_svd(jnp.asarray(Mp), swap_block=D * D)
        np.testing.assert_allclose(
            np.asarray(Up),
            np.asarray(U),
            atol=1e-10,
            err_msg=f"gauge-fixed U moved under a 1e-15 perturbation (seed {seed})",
        )
        np.testing.assert_allclose(np.asarray(Vhp), np.asarray(Vh), atol=1e-10)


def test_dense_gauge_copies_one_chi_sign_of_the_input_gauge():
    """Rows ``(a, d)`` rescaled by a chi-leg gauge ``sigma_a``: U must follow
    covariantly, each column's sign flipping by the ``sigma`` of ONE chi group
    -- the group of its largest entry.  (Columns by ``col_sign`` too: that
    gauge lives on V and must not move U at all.)"""
    M, perm, sign, col_sign = _dense_fixture(False)
    U, _s, _Vh = (
        np.asarray(x) for x in _gauge_fixed_svd(jnp.asarray(M), swap_block=D * D)
    )
    rng = np.random.default_rng(2)
    mixed = 0
    for trial in range(6):
        sigma = np.repeat(rng.choice([-1.0, 1.0], size=N_CHI), D * D)
        mixed += int(np.any(sigma != sigma[0]))
        tau = rng.choice([-1.0, 1.0], size=N_COL)
        Mg = sigma[:, None] * M * tau[None, :]
        Ug = np.asarray(_gauge_fixed_svd(jnp.asarray(Mg), swap_block=D * D)[0])
        group_sign = sigma[np.argmax(np.abs(U), axis=0)]  # one sigma per column
        np.testing.assert_allclose(
            Ug,
            sigma[:, None] * U * group_sign[None, :],
            atol=1e-10,
            err_msg=f"column sign is not a copy of one chi-group gauge (trial {trial})",
        )
    assert mixed >= 3  # the gauge actually mixed signs across chi groups


# --------------------------------------------------------------------------- #
# Block-sparse helper: the same symmetry spread over several U-blocks.
# --------------------------------------------------------------------------- #
def _symmetric_fixture():
    """Rows (a, k, b) with U(1) charges; the swap k <-> b maps block
    ``(qa, qk, qb)`` to ``(qa, qb, qk)`` -- a DIFFERENT U-block of the SAME bond
    sector, so the tie is across the blocks the helper stacks."""
    sym = U1Symmetry()
    rng = np.random.default_rng(1841)
    qa = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    qd = np.array([0, 1, 1], dtype=np.int32)
    qc = np.array([0, 1, 2, 3] * 12, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, qa, FlowDirection.IN, label="a"),
        TensorIndex.from_charges(sym, qd, FlowDirection.IN, label="k"),
        TensorIndex.from_charges(sym, qd, FlowDirection.IN, label="b"),
        TensorIndex.from_charges(sym, qc, FlowDirection.OUT, label="c"),
    )
    tau = rng.choice([-1.0, 1.0], size=len(qa))
    col_sign = rng.choice([-1.0, 1.0], size=len(qc))
    perm, sign = _row_swap(len(qa), len(qd), tau)
    M0 = np.asarray(
        SymmetricTensor.random_normal(idx, jax.random.PRNGKey(7)).todense()
    ).reshape(-1, len(qc))
    M = _swap_symmetric(M0, perm, sign, col_sign)
    return M, idx, perm, sign, col_sign


def _sym_gauge_fixed_U(M, idx):
    from tenax.linalg import svd as tensor_svd

    shape = tuple(ix.dim for ix in idx)
    T = SymmetricTensor.from_dense(jnp.asarray(M.reshape(shape)), idx)
    U_T, s, Vh_T, _ = tensor_svd(
        T, left_labels=("a", "k", "b"), right_labels=("c",), new_bond_label="bond"
    )
    U_T, Vh_T = _gauge_fix_symmetric_svd(U_T, Vh_T, swap_outer_legs=1)
    U = np.asarray(U_T.todense()).reshape(M.shape[0], -1)
    s, q = np.asarray(s), np.asarray(U_T.indices[-1].charges)
    # The swap splits each sector into even/odd halves of unequal size, so some
    # sectors are rank deficient: their null vectors are an arbitrary basis
    # (any gauge rule is free there), carry zero weight in U diag(s) Vh, and
    # reorder freely across sectors.  Keep the range only, ordered by
    # (sector, singular value) so the comparison is by column identity.
    live = np.where(s > 1e-8 * s.max())[0]
    order = live[np.lexsort((-s[live], q[live]))]
    return U[:, order], s[order], q[order]


def test_symmetric_gauge_is_stable_under_swap_symmetric_rounding():
    M, idx, perm, sign, col_sign = _symmetric_fixture()
    U, s, q = _sym_gauge_fixed_U(M, idx)
    for qq in np.unique(q):  # the kept range is non-degenerate per sector
        sq = s[q == qq]
        assert sq.size < 2 or np.min(np.abs(np.diff(sq))) > 1e-6 * s.max()
    assert U.shape[1] >= 20, U.shape
    _assert_opposite_sign_tie(U, perm, sign)
    for seed in PERTURBATION_SEEDS:
        Mp = _perturbed(M, perm, sign, col_sign, seed)
        Up, _sp, qp = _sym_gauge_fixed_U(Mp, idx)
        np.testing.assert_array_equal(qp, q)
        np.testing.assert_allclose(
            Up,
            U,
            atol=1e-10,
            err_msg=f"gauge-fixed U moved under a 1e-15 perturbation (seed {seed})",
        )


# --------------------------------------------------------------------------- #
# The rule stays differentiable and NaN-free.
# --------------------------------------------------------------------------- #
def test_gauge_fixed_svd_grad_is_finite_on_the_symmetric_fixture():
    M, *_ = _dense_fixture(False)

    def loss(m):
        U, s, Vh = _gauge_fixed_svd(m, regularized=True, swap_block=D * D)
        return jnp.sum(U[:, :6] ** 3) + jnp.sum(Vh[:6] ** 3)

    g = jax.grad(loss)(jnp.asarray(M))
    assert bool(jnp.all(jnp.isfinite(g)))
    # A zero chi group under the argmax, and a zero matrix: fallback branches.
    M0 = np.array(M)
    M0[: D * D] = 0.0
    for m in (jnp.asarray(M0), jnp.zeros((N_CHI * D * D, 7))):
        g = jax.grad(loss)(m)
        assert bool(jnp.all(jnp.isfinite(g)))
