"""#858: an adjoint solve must never come back worse than ``lambda = 0``.

GMRES minimises ``||b - A x||`` over ``x0 + Krylov``, so its final residual is
bounded by its **starting** residual ``||b - A x0||`` -- not by ``||b||``.  The
two coincide only at ``x0 = 0``.

``gmres_pytree_jax`` substituted ``b`` for a missing ``x0``, and all three
adjoint call sites passed ``x0 = rhs`` explicitly: the first Neumann term of
``(I - J^T)^-1 b``, which is a good seed while ``||J^T b|| < ||b||`` and a bad
one otherwise.  On the C4v D=2 state with ``recipe="1x1"`` -- rank-1 corners at
every chi (#723/#726) -- it was the latter, and the measured relative residuals
were 9.0e-02, **1.581** and **2.103**: a ``lambda`` solving the adjoint equation
worse than ``lambda = 0`` would.  It read as GMRES diverging, which GMRES cannot
do.

These tests use a synthetic non-contracting operator rather than a CTM, because
the property is about the solver contract, holds for every operator, and costs
milliseconds to check.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import (
    _best_adjoint_seed,
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._gmres_lax import gmres_pytree, gmres_pytree_jax
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate


def _non_contracting_system(n=24, scale=3.0, seed=0):
    """``A = I - J^T`` with ``||J^T b|| > ||b||`` -- so ``x0 = b`` is worse than 0.

    ``scale > 1`` puts the spectral radius of ``J`` above 1, which is exactly
    the regime where the fixed point is weakly attracting and the Neumann seed
    stops being a good guess.  Returned as single-leaf pytrees, matching how
    the adjoint sites call the solver.
    """
    rng = np.random.default_rng(seed)
    J = scale * rng.standard_normal((n, n)) / np.sqrt(n)
    b = jnp.asarray(rng.standard_normal(n))

    def matvec(v):
        return (v[0] - jnp.asarray(J).T @ v[0],)

    return matvec, (b,)


def _l2(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(tree))))


def _residual(matvec, x, b):
    return _l2(jax.tree.map(lambda p, q: p - q, matvec(x), b))


def test_the_neumann_seed_really_is_worse_than_zero_here():
    """The premise. Without it the tests below would prove nothing.

    ``x0 = 0`` starts GMRES at residual ``||b||``; ``x0 = b`` starts it at
    ``||b - A b|| = ||J^T b||``.  This fixture is built so the second is
    larger, which is the only situation in which the seed can hurt.
    """
    matvec, b = _non_contracting_system()
    start_zero = _residual(matvec, jax.tree.map(jnp.zeros_like, b), b)
    start_b = _residual(matvec, b, b)

    assert start_b > start_zero, (
        f"x0=b starts at residual {start_b:.4e}, no worse than x0=0's "
        f"{start_zero:.4e}, so this operator cannot exhibit #858. Raise "
        f"`scale` until the Neumann seed is genuinely bad."
    )


@pytest.mark.parametrize("maxiter", [1, 2, 5])
def test_the_solver_never_returns_worse_than_lambda_zero(maxiter):
    """The property #858 is about, at solver budgets small enough to stagnate.

    ``lambda = 0`` gives residual ``||b||`` for nothing.  Any solve that hands
    back more than that has actively made the gradient worse, and the caller
    cannot tell -- the returned ``lambda`` is used either way.

    Parametrized over tiny ``maxiter`` because a solve given enough iterations
    converges from any seed; the defect only surfaces when it stops early,
    which on a real CTM backward is the normal case (the budget is finite and
    the operator is ill-conditioned).
    """
    matvec, b = _non_contracting_system()
    b_norm = _l2(b)

    for solver in (gmres_pytree, gmres_pytree_jax):
        lam, _info = solver(matvec, b, tol=1e-12, maxiter=maxiter, restart=maxiter)
        resid = _residual(matvec, lam, b)
        assert resid <= b_norm * (1 + 1e-9), (
            f"{solver.__name__} returned a lambda with residual {resid:.4e} "
            f"against ||b|| = {b_norm:.4e} -- worse than lambda = 0, which is "
            f"free. GMRES cannot diverge; a residual above ||b|| means it was "
            f"seeded above ||b||. See #858."
        )


def test_a_missing_x0_means_zeros_not_b():
    """The wrapper's default must match JAX and SciPy.

    ``gmres_pytree`` passed ``None`` through (JAX then uses zeros) while
    ``gmres_pytree_jax`` substituted ``b`` -- two implementations of one
    interface disagreeing on the default, which is how the deviation survived.
    Pinning it as equality against an explicit zero seed rather than as
    "not equal to the b-seeded answer", so the test states the contract.
    """
    matvec, b = _non_contracting_system()
    zero = jax.tree.map(jnp.zeros_like, b)

    implicit, _ = gmres_pytree_jax(matvec, b, tol=1e-12, maxiter=3, restart=3)
    explicit_zero, _ = gmres_pytree_jax(
        matvec, b, zero, tol=1e-12, maxiter=3, restart=3
    )
    np.testing.assert_allclose(
        np.asarray(implicit[0]),
        np.asarray(explicit_zero[0]),
        rtol=1e-12,
        atol=1e-12,
        err_msg=(
            "gmres_pytree_jax(matvec, b) with no x0 did not match an explicit "
            "zero seed. A non-zero default removes the guarantee that the "
            "answer beats x = 0 (#858)."
        ),
    )


def test_the_seed_picker_never_loses_to_zero():
    """``_best_adjoint_seed`` keeps zero unless a candidate strictly beats it."""
    matvec, b = _non_contracting_system()
    b_norm = _l2(b)

    seed = _best_adjoint_seed(matvec, b, [b])
    assert _residual(matvec, seed, b) <= b_norm * (1 + 1e-12)
    # On this operator b is the bad seed, so zero must win outright.
    np.testing.assert_allclose(np.asarray(seed[0]), 0.0, atol=0.0)


def test_the_seed_picker_takes_a_candidate_that_helps():
    """...and it is not merely "always return zero", which would also pass above.

    Hands it the exact solution.  A picker that ignored its candidates would
    return zero at residual ``||b||``; the right answer is residual ~0.
    """
    matvec, b = _non_contracting_system()
    exact, _ = gmres_pytree_jax(matvec, b, tol=1e-14, maxiter=200, restart=50)
    assert _residual(matvec, exact, b) < 1e-8, "fixture: the exact solve failed"

    seed = _best_adjoint_seed(matvec, b, [exact])
    np.testing.assert_allclose(
        np.asarray(seed[0]), np.asarray(exact[0]), rtol=1e-12, atol=1e-12
    )


def test_a_non_finite_candidate_never_wins():
    """Fails closed: a blown-up seed must not be selected.

    The comparison is a strict ``resid < best``, and every comparison against
    NaN is False, so the incumbent survives. Written as a test because the
    ``>=`` spelling inverts it silently -- the #796/#848 shape, where a guard
    took the quiet branch exactly when the input was garbage.
    """
    matvec, b = _non_contracting_system()
    poisoned = (b[0].at[0].set(jnp.nan),)

    seed = _best_adjoint_seed(matvec, b, [poisoned])
    assert np.all(np.isfinite(np.asarray(seed[0]))), (
        "a NaN-bearing candidate was selected as the adjoint seed"
    )
    np.testing.assert_allclose(np.asarray(seed[0]), 0.0, atol=0.0)


def _implicit_energy(
    A_arr,
    *,
    chi,
    recipe="1x1",
    projector_method="eigh",
    max_iter=80,
    adjoint_method="gmres",
):
    """Implicit-AD energy of a raw ``(D,D,D,D,d)`` array, single-site.

    Two non-default settings, both load-bearing:

    * ``adjoint_method="gmres"`` rather than the ``"fixed_point"`` default:
      this file is about the GMRES seed, and the default routes to the fused
      Neumann branch, which never calls the solver at all.  That branch has
      its own seeding (``init_lam``) but not this failure mode -- it is not
      minimising over a Krylov space, and it carries its own divergence guard.
    * ``projector_method="eigh"`` rather than the ``"svd"`` default, because
      the ``svd`` projector leaves the Neumann seed *harmless* on this fixture
      (measured ``||J^T b|| / ||b|| = 0.29`` at chi=16, against 5.14 under
      ``eigh``), and a fixture where the seed cannot hurt cannot distinguish a
      measured seed from an unconditional one.
    """
    A = _wrap_as_dense_tensor(A_arr)
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            heisenberg_gate(),
            recipe=recipe,
            projector_method=projector_method,
            chi=chi,
            max_iter=max_iter,
            min_iter=8,
            conv_tol=1e-12,
            adjoint_method=adjoint_method,
        )
    )


#: chi at which the Neumann seed is measurably harmful on ``_random_D2_site``.
#:
#: The seed hurts exactly when ``||J^T b|| > ||b||``, and that is a property of
#: the fixture, not of the code under test -- so it has to be *found*, not
#: assumed.  Scanned over chi with ``recipe="1x1"`` and the eigh projector:
#:
#:     chi   ||J^T b|| / ||b||
#:       4   2.8e-01 (30 sweeps) .. 3.9e-01 (80 sweeps)
#:      16   **5.14**
#:      24   **3.44**
#:
#: At chi=4 the seed is harmless, so the same test there proves nothing: an
#: earlier version parametrized over chi in {4, 8} and the mutant that reverts
#: all three call sites to a bare ``rhs`` **passed it**.  The test asserts the
#: premise per solve rather than trusting this table, so a fixture drift that
#: made the seed benign again fails loudly instead of going quiet.
_HARMFUL_SEED_CHI = 16


def _random_D2_site(seed=0):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


def test_the_real_adjoint_seeds_the_solver_no_worse_than_zero(monkeypatch):
    """Covers the *wiring*, which the synthetic tests above do not.

    Reverting the three ``_best_adjoint_seed`` calls to a bare ``rhs`` leaves
    every other test in this file passing -- they exercise the wrapper and the
    picker, not the call sites. So this one runs a real implicit-AD backward
    and inspects the ``x0`` the library actually hands the solver.

    Asserting on the seed rather than on the final residual is deliberate.
    Whether a bad seed *survives* to the answer depends on how much progress
    GMRES then makes -- measured on the C4v D=2 state, ``x0 = b`` starts at a
    relative residual of 2.786 at chi=16 and GMRES still claws it back to
    0.104. A test gated on the final number passes there while the defect is
    fully present; an earlier version of this test did exactly that and the
    call-site mutant survived it. The seed is what this fix controls, and
    ``<= ||b||`` is precisely the property that makes "worse than lambda = 0"
    unreachable.

    ``recipe="1x1"`` is deliberate: it collapses the corners to rank 1 at every
    chi (#723/#726), which is what makes ``(I - J^T)`` badly enough conditioned
    for the seed to matter. This asserts the *solver contract*, not that the
    gradient is accurate -- on a collapsed environment it is not, and that is
    #795's problem rather than this one.
    """
    import tenax.algorithms._ctm_energy_ad as ad

    seen = []
    real = ad.gmres_pytree_jax

    def spy(matvec, b_tree, x0_tree=None, **kw):
        seen.append((matvec, b_tree, x0_tree))
        return real(matvec, b_tree, x0_tree, **kw)

    monkeypatch.setattr(ad, "gmres_pytree_jax", spy)

    A0 = _random_D2_site()
    jax.grad(lambda a: _implicit_energy(a, chi=_HARMFUL_SEED_CHI, max_iter=80))(A0)

    assert seen, (
        "no eager GMRES solve ran, so this test observed nothing. The "
        "backward took the fused fixed-point branch instead; re-point the "
        "fixture at the branch that calls gmres_pytree_jax."
    )
    for i, (matvec, b_tree, x0_tree) in enumerate(seen):
        b_norm = _l2(b_tree)
        # The premise, per solve: without it a passing assertion below would
        # only mean the fixture is benign at this chi.
        neumann = _residual(matvec, b_tree, b_tree)
        assert neumann > b_norm, (
            f"solve {i}: the Neumann seed starts at {neumann:.4e} against "
            f"||b|| = {b_norm:.4e}, i.e. it is *not* harmful here, so this "
            f"test cannot tell a measured seed from an unconditional one. "
            f"Re-scan chi (measured 5.14x at chi={_HARMFUL_SEED_CHI})."
        )
        seed_resid = _residual(matvec, x0_tree, b_tree)
        assert seed_resid <= b_norm * (1 + 1e-9), (
            f"adjoint solve {i} was seeded at residual {seed_resid:.4e} "
            f"against ||b|| = {b_norm:.4e}. GMRES's residual bound is against "
            f"its *starting* point, so seeding above ||b|| lets it return a "
            f"lambda worse than lambda = 0 -- and it cannot recover if it "
            f"stagnates. See #858."
        )


def test_the_real_adjoint_never_reports_a_relative_residual_above_one():
    """The end-to-end contract, in the library's own reported number.

    ``adjoint_residual`` is documented as ``||(I - J^T) lam - b|| / ||b||``, so
    ``1.0`` is exactly the residual of ``lam = 0``. Anything above it means the
    solve handed back a gradient direction worse than contracting nothing, and
    #858 measured 1.581 and 2.103 on the C4v D=2 state.

    Weaker than the seed assertion above and kept anyway: it is stated in the
    units a caller reads, so it stays meaningful if the seeding is ever
    reworked.
    """
    A0 = _random_D2_site()
    jax.grad(lambda a: _implicit_energy(a, chi=_HARMFUL_SEED_CHI, max_iter=80))(A0)

    diag = get_last_implicit_ad_diagnostics()
    assert "adjoint_residual" in diag, (
        "the backward published no adjoint residual, so this test cannot see "
        "what the solve returned"
    )
    rel = float(diag["adjoint_residual"])
    assert rel <= 1.0 + 1e-9, (
        f"the adjoint solve reported a relative residual of {rel:.4e}. Above "
        f"1.0 means the returned lambda solves (I - J^T)lambda = b worse than "
        f"lambda = 0 does, which GMRES cannot produce unless it was seeded "
        f"above ||b||. See #858."
    )


def test_the_seed_picker_works_under_jit():
    """It has to: one of the two call sites is inside ``_jit_gmres_solve``.

    A host-side variant (pulling residuals back with ``device_get`` and
    comparing with Python ``<``) raises ``ConcretizationTypeError`` here, and
    would have left the compiled adjoint path still carrying #858 while the
    eager one was fixed.
    """
    matvec, b = _non_contracting_system()

    @jax.jit
    def pick(rhs):
        return _best_adjoint_seed(matvec, rhs, [rhs])

    seed = pick(b)
    assert np.all(np.isfinite(np.asarray(seed[0])))
    assert _residual(matvec, seed, b) <= _l2(b) * (1 + 1e-12)
