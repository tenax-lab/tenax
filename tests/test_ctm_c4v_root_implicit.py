"""Root implicit differentiation for C4v CTMRG (#715 Phase 0, arXiv:2607.15030).

The point of these tests is the paper's central claim: the gradient is
obtained from an algebraic characteristic equation, so no ``eigh``/``svd``
backward appears in the gradient path, and degenerate spectra — which make
the truncated-decomposition VJP diverge — are harmless.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_c4v_root_implicit import (
    c4v_characteristic_residual,
    c4v_root_implicit_energy,
    c4v_root_implicit_energy_and_grad,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _site_tensor(D: int = 2, d: int = 2, seed: int = 42, eps: float = 0.5):
    """Entangled, genuinely C4v-symmetric iPEPS tensor with trivial U(1) charges.

    ``eps`` controls the deviation from a product state.  It must be large
    enough that the kept corner spectrum sits above numerical noise: at
    ``eps = 0.1`` and ``chi = 8`` the truncation cuts through eigenvalues of
    order 1e-12 and the environment is no longer a root of the
    characteristic equations.

    The projection onto the C4v-invariant subspace is load-bearing, not
    cosmetic.  The enlarged corner ``C·E_top·E_left·a`` is Hermitian *only*
    for a C4v-symmetric state — on the raw random tensor this fixture used to
    return, ``‖M - M†‖/‖M‖`` is 0.97 and no root exists to find.  The old
    ``M = 2 Cg Cg†`` was Hermitian PSD for any input whatsoever, so it
    accepted states the scheme does not apply to (#760).
    """
    from tenax.algorithms.ipeps import symmetrize_c4v

    rng = np.random.RandomState(seed)
    data = eps * jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = symmetrize_c4v(data)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(data, indices)


def _xxz_gate(delta: float = 1.0):
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


# ``conv_tol`` is load-bearing for ``test_gradient_matches_finite_difference``
# and must not be loosened.  The finite difference divides by ``2h = 2e-5``, so
# it amplifies any residual error in the converged energy by 5e4.  At
# ``conv_tol=1e-12`` the environment still carries a ~1.4e-13 energy error,
# which lands as ~9e-9 in ``fd`` — the same order as the 1e-8 tolerance the
# test asserts.  The test then measures the FD noise floor rather than the
# gradient: it passes at ``h=1e-5`` and fails at ``h=1e-6`` (9.0e-8) on code
# whose gradient is fine.  At 1e-14 the parity is 1.2e-10..2.2e-09 across both
# h and every parametrization, and the assertion means what it says.
# Costs ~11 extra CTM iterations (58 -> 69 at chi=6).
_CTM_KW = dict(max_iter=300, conv_tol=1e-14, projector_method="eigh")


def _directional_fd(A, gate, direction, h, **kw):
    base = A.todense()

    def at(t):
        e, _ = c4v_root_implicit_energy_and_grad(
            DenseTensor(base + t * direction, A.indices), gate, **kw
        )
        return float(e)

    return (at(h) - at(-h)) / (2.0 * h)


# ------------------------------------------------------------------ #
# The root                                                            #
# ------------------------------------------------------------------ #


def test_characteristic_residual_vanishes_at_root():
    """``F(y*, p) = 0`` — the environment really is a root, not just a fixed point.

    This is the precondition for the implicit function theorem, and per the
    paper's Fig. 1 it is what bounds the gradient accuracy.
    """
    A = _site_tensor(eps=1.0)
    _e, _g, diag = c4v_root_implicit_energy_and_grad(
        A, _xxz_gate(1.0), chi=6, return_diagnostics=True, **_CTM_KW
    )
    assert diag["root_residual"] < 1e-9, diag
    assert diag["adjoint_residual"] < 1e-8, diag


def test_no_eigh_or_svd_in_the_differentiated_equations():
    """The characteristic equations are pure contractions.

    This is the structural claim that makes the method stable: whatever the
    forward contraction does, the backward differentiates only ``F``, and
    ``F`` contains no decomposition to differentiate.
    """
    rng = np.random.RandomState(0)
    chi, d2 = 4, 4
    n = chi * d2
    C = jnp.array(rng.standard_normal((chi, chi)))
    E = jnp.array(rng.standard_normal((chi, d2, chi)))
    a = jnp.array(rng.standard_normal((d2, d2, d2, d2)))
    Q, _ = jnp.linalg.qr(jnp.array(rng.standard_normal((n, n))))
    U_star, U_perp = Q[:, :chi], Q[:, chi:]
    C_inv = jnp.eye(chi)
    u = jnp.zeros((n - chi, chi))

    def F(y):
        return c4v_characteristic_residual(y, a, U_star, U_perp, C_inv)

    forward = str(jax.make_jaxpr(F)((C, E, u)))
    _, vjp = jax.vjp(F, (C, E, u))
    backward = str(jax.make_jaxpr(vjp)(F((C, E, u))))

    for banned in ("eigh", "svd", "eig "):
        assert banned not in forward, f"{banned!r} in forward jaxpr of F"
        assert banned not in backward, f"{banned!r} in backward jaxpr of F"


# ------------------------------------------------------------------ #
# Gradient parity                                                     #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("delta,chi", [(0.3, 6), (1.0, 6), (1.0, 4)])
def test_gradient_matches_finite_difference(delta, chi):
    """AD gradient reproduces a symmetric finite difference.

    ``delta = 1.0`` is the isotropic Heisenberg point, where the corner
    spectrum is SU(2)-degenerate — the case that floors the truncated-SVD
    backward on the production path at ~5e-4.

    The difference direction is projected onto the C4v-invariant subspace,
    because that is the manifold the method is defined on: off it the
    enlarged corner ``C·E_top·E_left·a`` is not Hermitian, so ``A ± h·v``
    along a generic ``v`` (76% off-manifold for this seed) is not a state the
    C4v CTM contracts consistently.  Measured at chi=4: generic ``v`` floors
    at 3.6e-5 while a symmetric one reaches 1.7e-9 — the parity is exact
    in-manifold, and it was only the old scheme's ``M = 2 Cg Cg†``, Hermitian
    for *any* input, that made an off-manifold difference look meaningful
    (#760).
    """
    from tenax.algorithms.ipeps import symmetrize_c4v

    A = _site_tensor(eps=1.0)
    gate = _xxz_gate(delta)
    kw = dict(chi=chi, **_CTM_KW)

    _e, grad, diag = c4v_root_implicit_energy_and_grad(
        A, gate, return_diagnostics=True, **kw
    )
    assert diag["root_residual"] < 1e-9, diag

    rng = np.random.RandomState(7)
    v = jnp.array(rng.standard_normal(A.todense().shape))
    v = symmetrize_c4v(v)
    v = v / jnp.linalg.norm(v)

    ad = float(jnp.sum(grad * v))
    fd = _directional_fd(A, gate, v, 1e-5, **kw)
    assert abs(fd - ad) / max(abs(fd), 1e-30) < 1e-8, (fd, ad)


def test_gradient_is_finite_on_a_degenerate_spectrum():
    """A deliberately degenerate corner spectrum yields a finite gradient.

    A near-product state at ``chi = 8`` retains eigenvalues that are equal
    to within numerical noise — the configuration that produces NaNs from a
    ``_gauge_fixed_svd`` backward on the production path.  Nothing here
    divides by an eigenvalue difference, so the gradient stays finite.
    """
    A = _site_tensor(eps=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _e, grad = c4v_root_implicit_energy_and_grad(
            A, _xxz_gate(1.0), chi=8, on_root_residual="warn", **_CTM_KW
        )
    assert jnp.all(jnp.isfinite(grad))


def test_raises_on_a_non_vanishing_residual_by_default():
    """A non-vanishing ``‖F(y*)‖`` is a hard failure, not a warning.

    The gradient solves the adjoint of equations ``y*`` does not satisfy, so
    it comes back finite and silently wrong.  An unattended optimizer cannot
    detect that, so the default has to stop rather than report.

    Driven by making the *tolerance* impossible rather than by engineering a
    degenerate spectrum, matching
    ``test_ctm_root_implicit_multisite.test_an_unconverged_root_raises_by_default``.
    This used to truncate a near-product state through numerical noise at
    ``eps=0.1, chi=8``, but the corrected enlarged corner (#760) no longer
    produces a collapsing corner spectrum — a sweep of
    ``eps ∈ {0.02..0.3} × chi ∈ {8,12,16}`` now stays under 1.9e-10
    throughout, so that trigger is gone and the policy needs a deterministic
    one.
    """
    from tenax.algorithms._ad_primitives import RootResidualError

    A = _site_tensor(eps=0.1)
    with pytest.raises(RootResidualError, match=r"‖F\(y\*\)‖") as excinfo:
        c4v_root_implicit_energy_and_grad(
            A, _xxz_gate(1.0), chi=8, root_residual_warn=0.0, **_CTM_KW
        )
    assert excinfo.value.residual >= 0.0
    assert excinfo.value.tolerance == 0.0


def test_a_non_vanishing_residual_still_warns_under_the_warn_policy():
    """The diagnostic itself is not lost -- ``on_root_residual='warn'`` keeps
    the old reporting behaviour for callers that want to inspect the bad
    gradient rather than abort."""
    A = _site_tensor(eps=0.1)
    with pytest.warns(RuntimeWarning, match=r"‖F\(y\*\)‖"):
        c4v_root_implicit_energy_and_grad(
            A,
            _xxz_gate(1.0),
            chi=8,
            root_residual_warn=0.0,
            on_root_residual="warn",
            **_CTM_KW,
        )


def test_warns_when_the_adjoint_solve_does_not_converge():
    """An unconverged solve is an invalid gradient, not an approximate one.

    ``F̆`` is defined as *the* solution of Eq. 17; stop the Krylov iteration
    early and what comes back is finite, plausible-looking and wrong.  The
    residual used to be visible only under ``return_diagnostics``.
    """
    A = _site_tensor(eps=1.0)
    # ``solve_maxiter`` counts GMRES *restarts*, so the Krylov dimension has
    # to be squeezed too -- one restart of a 30-dimensional space converges.
    with pytest.warns(RuntimeWarning, match="adjoint solve did not converge"):
        c4v_root_implicit_energy_and_grad(
            A,
            _xxz_gate(1.0),
            chi=6,
            solve_maxiter=1,
            solve_restart=1,
            solve_tol=1e-14,
            **_CTM_KW,
        )


def test_energy_only_helper_skips_the_adjoint_solve():
    """``c4v_root_implicit_energy`` must not pay for -- or fail in -- a solve.

    Sabotaging the adjoint solve leaves the energy untouched, which is the
    property that makes the helper usable in a line search.
    """
    import tenax.algorithms._ctm_c4v_root_implicit as mod

    A = _site_tensor(eps=1.0)
    gate = _xxz_gate(1.0)
    kw = dict(chi=6, **_CTM_KW)

    reference, _grad = c4v_root_implicit_energy_and_grad(A, gate, **kw)

    original = mod._solve_root_adjoint
    mod._solve_root_adjoint = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("energy-only path ran the adjoint solve")
    )
    try:
        energy = c4v_root_implicit_energy(A, gate, **kw)
    finally:
        mod._solve_root_adjoint = original

    assert abs(float(energy) - float(reference)) < 1e-12


# ------------------------------------------------------------------ #
# Guards                                                              #
# ------------------------------------------------------------------ #


def test_rejects_symmetric_tensor():
    """SymmetricTensor support is #715 Phase 3, not silently wrong here."""
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()
    idx = tuple(
        TensorIndex.from_charges(
            sym, np.zeros(2, dtype=np.int32), FlowDirection.OUT, label=lbl
        )
        for lbl in ("u", "d", "l", "r", "phys")
    )
    A = SymmetricTensor.from_dense(jnp.ones((2, 2, 2, 2, 2)), idx, tol=float("inf"))
    with pytest.raises(TypeError, match="Phase 3"):
        c4v_root_implicit_energy_and_grad(A, _xxz_gate(1.0), chi=4)
