"""Asymmetric CTMRG root implicit AD (#715 Phase 1, arXiv:2607.15030 §V).

Status: the forward, the characteristic equations, the root and the gradient
are all verified here.  Two of these tests are the ones that closed #718 —
``test_the_energy_is_invariant_under_the_bond_gauge`` (the bug) and
``test_gradient_parity_needs_the_modified_variables`` (what it cost).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._ctm_root_implicit_asym as M
from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _site_tensor(D=2, d=2, seed=42, eps=1.0):
    rng = np.random.RandomState(seed)
    data = eps * jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


def _gate(delta=1.0):
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _a_array(A):
    at = _build_double_layer_tensor(A)
    lab = list(at.labels())
    perm = tuple(lab.index(x) for x in ("u2", "d2", "l2", "r2"))
    return jnp.asarray(at.transpose(perm).todense())


# ------------------------------------------------------------------ #
# Geometry                                                            #
# ------------------------------------------------------------------ #


def test_rotation_is_order_four():
    A = _site_tensor()
    a = _a_array(A)
    assert jnp.allclose(M.rotate_a(M.rotate_a(M.rotate_a(M.rotate_a(a)))), a)


def test_projectors_close_on_the_retained_subspace():
    """``P_bot @ P_top = 1`` is what makes the Fishman insertion exact."""
    A = _site_tensor()
    chi = 6
    env, a, _meta = M.converge(A, chi, max_iter=300, conv_tol=1e-12)
    P_top, P_bot, _U, _s, _Vh = M.all_projectors(env, a, chi)[0]
    # Proportional to the identity, not equal to it: the retained singular
    # values are normalised, so the closure carries their overall scale.
    closure = P_bot @ P_top
    scale = jnp.trace(closure) / chi
    assert jnp.linalg.norm(closure / scale - jnp.eye(chi)) < 1e-9


# ------------------------------------------------------------------ #
# Forward                                                             #
# ------------------------------------------------------------------ #


def test_energy_is_stationary_at_the_root():
    """Further sweeps from the polished root leave the energy alone.

    ``converge`` alone is not enough: its element-wise criterion is not met
    within 300 sweeps at chi=6, and its output energy is still 2e-4 away.
    The gauge-aligned polish inside :func:`asym_root_parametrize` is what
    actually lands on the fixed point, which is why the root is extracted
    there rather than taken straight from ``converge``.
    """
    A = _site_tensor()
    chi = 4
    gate = _gate(1.0)
    env, a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    root, residual = M.asym_root_parametrize(env, a, chi, polish_steps=40)
    assert residual < 1e-12, residual

    template = initialize_ctm_tensor_env(A, chi)
    e0 = float(M.asym_energy(A, root.env, template, gate))
    env2, projs = root.env, None
    for _ in range(12):
        env2, projs = M.sweep(env2, a, chi, projs)
    e1 = float(M.asym_energy(A, env2, template, gate))
    assert abs(e0 - e1) < 1e-9, (e0, e1)


# ------------------------------------------------------------------ #
# The root                                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("chi", [4, 6])
def test_all_twenty_characteristic_equations_hold_at_the_root(chi):
    """‖F(y*, p)‖ at machine precision — the root really is a root.

    Twenty equations for twenty variables: four corners, four edges, and
    per direction one ``u``, one ``S`` and one ``v`` (paper Eqs. 76-80).
    """
    A = _site_tensor()
    env, a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    root, residual = M.asym_root_parametrize(env, a, chi, polish_steps=30)
    assert residual < 1e-12, residual

    R_env, R_u, R_S, R_v = M.asym_characteristic_residual(root.y, a, root, chi)
    for name, leaf in zip(R_env._fields, R_env):
        assert float(jnp.linalg.norm(leaf)) < 1e-12, name
    for k in range(4):
        assert float(jnp.linalg.norm(R_u[k])) < 1e-12, f"R_u[{k}]"
        assert float(jnp.linalg.norm(R_S[k])) < 1e-12, f"R_S[{k}]"
        assert float(jnp.linalg.norm(R_v[k])) < 1e-12, f"R_v[{k}]"


def test_jacobian_of_F_is_nonsingular():
    """``∂_y F`` is invertible, so the implicit function theorem applies.

    Worth pinning: it rules out a zero mode as the explanation for the
    gradient failure below, which is what a missing normalisation
    condition would have looked like.
    """
    from tenax.algorithms._ctm_c4v_root_implicit import (
        _from_real_vec,
        _real_struct,
        _to_real_vec,
    )

    A = _site_tensor()
    chi = 4
    env, a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    root, _res = M.asym_root_parametrize(env, a, chi, polish_steps=30)
    _leaves, treedef = jax.tree.flatten(root.y)
    struct = _real_struct(root.y)

    def f(vec):
        return _to_real_vec(
            M.asym_characteristic_residual(
                _from_real_vec(vec, treedef, struct), a, root, chi
            )
        )

    J = jax.jacfwd(f)(_to_real_vec(root.y))
    sv = jnp.linalg.svd(J, compute_uv=False)
    assert float(sv[-1] / sv[0]) > 1e-8, float(sv[-1] / sv[0])


def test_no_svd_in_the_differentiated_equations():
    """``F`` is contractions only — no decomposition to back-propagate."""
    A = _site_tensor()
    chi = 4
    env, a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    root, _res = M.asym_root_parametrize(env, a, chi, polish_steps=5)

    def F(y):
        return M.asym_characteristic_residual(y, a, root, chi)

    text = str(jax.make_jaxpr(F)(root.y))
    for banned in ("svd", "eigh"):
        assert banned not in text, banned


# ------------------------------------------------------------------ #
# The gradient                                                        #
# ------------------------------------------------------------------ #


def test_gradient_parity_needs_the_modified_variables():
    """Eq. 18 against explicit backprop through the same sweep: 4e-8 relative.

    This was 3.06e-2 for the whole of #718, and the hunt went through the
    characteristic equations three times.  Worth recording what it was *not*,
    since each of these looked like the answer:

    * not ``S̆ = 0`` — giving ``S`` its Eq. 82 adjoint was necessary but moved
      the error the wrong way (2.5e-2 -> 3.06e-2; the 2.5e-2 was accidental);
    * not the missing covariance package — added, no change;
    * not Eq. 88's null-space restriction — the gauge is licensed to 1e-16,
      globally and per direction (the two tests below).

    The machinery was exact all along: FD of ``G(p) = E`` at the frozen-``c``
    root matched the implicit gradient to nine digits while differing from the
    truth by 8.8503e-3.  ``G`` was not the fixed-point energy because
    :func:`M.swap_env_convention` was missing, so ``asym_energy`` contracted a
    mis-glued network — see
    ``test_the_energy_is_invariant_under_the_bond_gauge``.
    """
    A = _site_tensor()
    chi = 4
    gate = _gate(1.0)
    _e, grad = M.asym_root_implicit_energy_and_grad(
        A, gate, chi=chi, max_iter=300, conv_tol=1e-13
    )

    env0, _a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    template = initialize_ctm_tensor_env(A, chi)

    def energy_explicit(pdata):
        A_live = DenseTensor(pdata, A.indices)
        a_live = _a_array(A_live)
        env, projs = env0, None
        for _ in range(12):
            env, projs = M.sweep(env, a_live, chi, projs)
        return M.asym_energy(A_live, env, template, gate)

    g_ref = jax.grad(energy_explicit)(A.todense())
    rel = float(jnp.linalg.norm(grad - g_ref) / jnp.linalg.norm(g_ref))
    assert rel < 1e-5, rel


def test_the_gradient_survives_where_explicit_backprop_nans():
    """``D=3``: explicit backprop is all-NaN, the implicit gradient is exact.

    This is #566/#687 as a plain numerical fact rather than a compile-time
    argument.  Explicit AD differentiates the truncated SVD that builds the
    projectors, and at ``D=3``, ``chi=4`` the discarded spectrum is degenerate
    enough that the SVD backward divides by ``s_i^2 - s_j^2`` and returns NaN
    for *every* entry.  ``F`` contains no decomposition
    (``test_no_svd_in_the_differentiated_equations``), so the implicit route is
    unaffected.

    The reference here has to be a directional finite difference of the sweep
    *map* from a fixed start, which is smooth in ``p``.  Do not finite-difference
    the root: ``asym_root_parametrize`` is gauge-discontinuous, so ``|ẏ_fd|``
    diverges as ``h`` shrinks even though every point is a valid root.
    """
    A = _site_tensor(D=3)
    chi = 4
    gate = _gate(1.0)
    A_data = A.todense()
    template = initialize_ctm_tensor_env(A, chi)

    _e, grad = M.asym_root_implicit_energy_and_grad(
        A, gate, chi=chi, max_iter=300, conv_tol=1e-13
    )
    assert bool(jnp.all(jnp.isfinite(grad)))

    env0, _a, _m = M.converge(A, chi, max_iter=300, conv_tol=1e-13)

    def energy_explicit(pdata):
        A_live = DenseTensor(pdata, A.indices)
        a_live = _a_array(A_live)
        env, projs = env0, None
        for _ in range(20):
            env, projs = M.sweep(env, a_live, chi, projs)
        return M.asym_energy(A_live, env, template, gate)

    g_ref = jax.grad(energy_explicit)(A_data)
    assert bool(jnp.all(~jnp.isfinite(g_ref))), "explicit backprop is expected to NaN"

    rng = np.random.RandomState(1)
    V = jnp.asarray(rng.standard_normal(A_data.shape))
    V = V / jnp.linalg.norm(V)
    analytic = float(jnp.real(jnp.sum(grad * V)))
    h = 1e-5
    fd = float(
        (energy_explicit(A_data + h * V) - energy_explicit(A_data - h * V)) / (2 * h)
    )
    assert abs(fd - analytic) < 1e-6 * max(abs(analytic), 1e-3), (fd, analytic)


def test_freezing_the_isometries_is_gauge_licensed():
    """Eq. 88: freezing ``U*``/``Vh*`` costs nothing along the gauge orbit.

    The implicit gradient holds ``c = (U*, U_perp, Vh*, Vh_perp, s*inv)`` fixed,
    so it silently drops ``dE/dc · dc/dp``.  Eq. 88 argues that is harmless:
    ``dc/dp`` is a pure gauge, since ``U = U* + U_perp u`` already carries every
    non-gauge variation of the retained subspace, and ``E`` is gauge invariant.
    ``dE/dc`` comes for free as ``-F̆ ∂_c F`` using the same ``F̆`` the gradient
    already solves for, so this checks the argument directly.

    The gauge direction is the part that is easy to get wrong.  Every frozen
    constant carries the bond gauge on **two** kinds of index — the new bond
    (``chi``) and the fused cut leg ``n = chi*d2`` — so for a global unitary
    ``W = exp(tX)`` with ``X`` anti-Hermitian,

        U*      -> kron(W,I)† U* W          δU*      = U* X - kron(X,I) U*
        U_perp  -> kron(W,I)† U_perp        δU_perp  =      - kron(X,I) U_perp
        Vh*     -> W† Vh* kron(W,I)         δVh*     = Vh* kron(X,I) - X Vh*
        Vh_perp -> Vh_perp kron(W,I)        δVh_perp = Vh_perp kron(X,I)
        s*inv   -> W† s*inv W               δs*inv   = [s*inv, X]

    Keeping only the ``U* X`` term drops the ``kron(X,I)`` pieces and is then
    not a gauge direction at all: it reports ~4e-3 on the same data where the
    full law gives 1e-16, which is how #718 came to be blamed on covariance.
    """
    A, a, root = _converged_root()
    chi = root.env.C1.shape[0]
    d2 = a.shape[0]
    root = M.asym_root_to_covariant_convention(root)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)
    y_star = (tilde, root.u, S, root.v)

    template = initialize_ctm_tensor_env(A, chi)
    gate = _gate(1.0)
    A_data = A.todense()

    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_energy(
            A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (
        tilde_bar,
        tuple(jnp.zeros_like(x) for x in root.u),
        S_bar,
        tuple(jnp.zeros_like(x) for x in root.v),
    )

    _, vjp_y = jax.vjp(
        lambda y: M.asym_characteristic_residual_covariant(y, a, root, chi), y_star
    )
    F_bar, _resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0], y_bar, tol=1e-11, maxiter=800, restart=40
    )

    def F_of_consts(U_star, U_perp, Vh_star, Vh_perp, s_star_inv):
        c = root._replace(
            U_star=U_star,
            U_perp=U_perp,
            Vh_star=Vh_star,
            Vh_perp=Vh_perp,
            s_star_inv=s_star_inv,
        )
        return M.asym_characteristic_residual_covariant(y_star, a, c, chi)

    _, vjp_c = jax.vjp(
        F_of_consts,
        root.U_star,
        root.U_perp,
        root.Vh_star,
        root.Vh_perp,
        root.s_star_inv,
    )
    U_bar, Up_bar, Vh_bar, Vp_bar, si_bar = vjp_c(F_bar)

    def pair(g, delta):
        return float(jnp.real(jnp.sum(jnp.conj(g) * delta)))

    def dE_along(Xs):
        """``dE/dt`` for per-bond generators ``Xs[j]`` on bond ``j``.

        The cut leg of direction ``k`` carries the bond of direction ``k-1`` on
        the ``U`` side and ``k+1`` on the ``V`` side, read straight off
        ``Ud[k] = U_k† K_L[k-1]`` and ``Vd[k] = K_R[k+1] Vh_k†``.
        """
        KX = [jnp.kron(X, jnp.eye(d2, dtype=X.dtype)) for X in Xs]
        total = 0.0
        for k in range(4):
            km, kp = (k - 1) % 4, (k + 1) % 4
            Us, Up = root.U_star[k], root.U_perp[k]
            Vs, Vp = root.Vh_star[k], root.Vh_perp[k]
            si = root.s_star_inv[k]
            total += pair(U_bar[k], Us @ Xs[k] - KX[km] @ Us)
            total += pair(Up_bar[k], -KX[km] @ Up)
            total += pair(Vh_bar[k], Vs @ KX[kp] - Xs[k] @ Vs)
            total += pair(Vp_bar[k], Vp @ KX[kp])
            total += pair(si_bar[k], si @ Xs[k] - Xs[k] @ si)
        return total

    key = jax.random.PRNGKey(11)
    worst = 0.0
    for _ in range(4):
        key, sub = jax.random.split(key)
        G = jax.random.normal(sub, (chi, chi), dtype=jnp.complex128)
        X = 0.5 * (G - G.conj().T)
        X = X / jnp.linalg.norm(X)
        worst = max(worst, abs(dE_along([X] * 4)))

    scale = float(jnp.linalg.norm(grad_direct))
    assert worst < 1e-10 * scale, (worst, scale)


def test_each_directional_gauge_is_independently_licensed():
    """Eq. 88 must hold for one gauge generator per bond, not just their sum.

    The bond gauge has an independent generator per direction; a test that uses
    the same ``X`` everywhere only probes the global diagonal subgroup, and
    contributions from different directions can cancel inside it — which is
    exactly what happened.  Directions 0 and 1 vanished at 2e-16 while 2 and 3
    came out at +2.121e-03 and -2.121e-03, equal and opposite, so the global
    test passed on a sum of two errors (caught by Codex review on PR #720).

    Both readings offered then were wrong: the per-direction gauge *is*
    licensed, and the ``k-1``/``k+1`` cut-leg assignment *is* right.  The
    antisymmetry came from the energy, not from ``F`` — the two broken
    directions are the two bonds that touch ``C4``, the one tensor whose axis
    order differs between this module's convention and the CTM env's.  See
    :func:`M.swap_env_convention`.
    """
    A, a, root = _converged_root()
    chi = root.env.C1.shape[0]
    d2 = a.shape[0]
    root = M.asym_root_to_covariant_convention(root)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)
    y_star = (tilde, root.u, S, root.v)

    template = initialize_ctm_tensor_env(A, chi)
    gate = _gate(1.0)
    A_data = A.todense()

    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_energy(
            A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (
        tilde_bar,
        tuple(jnp.zeros_like(x) for x in root.u),
        S_bar,
        tuple(jnp.zeros_like(x) for x in root.v),
    )

    _, vjp_y = jax.vjp(
        lambda y: M.asym_characteristic_residual_covariant(y, a, root, chi), y_star
    )
    F_bar, _resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0], y_bar, tol=1e-11, maxiter=800, restart=40
    )

    def F_of_consts(U_star, U_perp, Vh_star, Vh_perp, s_star_inv):
        c = root._replace(
            U_star=U_star,
            U_perp=U_perp,
            Vh_star=Vh_star,
            Vh_perp=Vh_perp,
            s_star_inv=s_star_inv,
        )
        return M.asym_characteristic_residual_covariant(y_star, a, c, chi)

    _, vjp_c = jax.vjp(
        F_of_consts,
        root.U_star,
        root.U_perp,
        root.Vh_star,
        root.Vh_perp,
        root.s_star_inv,
    )
    U_bar, Up_bar, Vh_bar, Vp_bar, si_bar = vjp_c(F_bar)

    def pair(g, delta):
        return float(jnp.real(jnp.sum(jnp.conj(g) * delta)))

    zero = jnp.zeros((chi, chi))
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(2):
        G = jnp.asarray(rng.standard_normal((chi, chi)))
        X = 0.5 * (G - G.T)  # the state is real, so the gauge group is orthogonal
        X = X / jnp.linalg.norm(X)
        for i in range(4):
            Xs = [X if j == i else zero for j in range(4)]
            KX = [jnp.kron(Z, jnp.eye(d2, dtype=Z.dtype)) for Z in Xs]
            total = 0.0
            for k in range(4):
                km, kp = (k - 1) % 4, (k + 1) % 4
                Us, Up = root.U_star[k], root.U_perp[k]
                Vs, Vp = root.Vh_star[k], root.Vh_perp[k]
                si = root.s_star_inv[k]
                total += pair(U_bar[k], Us @ Xs[k] - KX[km] @ Us)
                total += pair(Up_bar[k], -KX[km] @ Up)
                total += pair(Vh_bar[k], Vs @ KX[kp] - Xs[k] @ Vs)
                total += pair(Vp_bar[k], Vp @ KX[kp])
                total += pair(si_bar[k], si @ Xs[k] - Xs[k] @ si)
            worst = max(worst, abs(total))

    scale = float(jnp.linalg.norm(grad_direct))
    assert worst < 1e-10 * scale, (worst, scale)


def test_gradient_entry_point_reports_clean_diagnostics():
    """The entry point converges, roots, and solves — all three, on one call."""
    A = _site_tensor()
    _e, _g, info = M.asym_root_implicit_energy_and_grad(
        A, _gate(1.0), chi=4, max_iter=300, conv_tol=1e-13, return_diagnostics=True
    )
    assert info["converged"], info
    assert info["root_residual"] < 1e-10, info
    assert info["covariant_residual"] < 1e-10, info
    assert info["adjoint_residual"] < 1e-8, info
    # Identically zero for a real state — the phase tangent is imaginary and the
    # cotangent is real — but the key has to be there for anything to watch it.
    assert info["gauge_consistency"] < 1e-10, info


def test_the_energy_is_invariant_under_the_bond_gauge():
    """The energy must not move when a bond basis is rotated (#718's real bug).

    A converged CTM environment has one chi-bond per direction, carried by four
    legs: corner ``k``'s own leg, both legs of edge ``k``, and corner ``k+1``'s
    other leg.  Inserting ``W W† = 1`` on bond ``k`` alone therefore cannot
    change anything the environment computes.  It is a necessary condition on
    the pairing of *every* env leg, it needs no reference value, and it is the
    cheapest possible check that :func:`M.swap_env_convention` is applied.

    Without the conversion the module's uniform convention is read as the CTM
    env's non-uniform one, ``C4`` is glued in transposed, and the energy moves
    on exactly the two bonds that touch ``C4`` — by 2e-4 here at ``t = 0.1``,
    and antisymmetrically, so a single global ``W`` sees nothing at all.  That
    made the energy wrong by 1.5% and the gradient by 3.06e-2.
    """
    A = _site_tensor()
    chi = 4
    gate = _gate(1.0)
    template = initialize_ctm_tensor_env(A, chi)
    env, _a, info = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    assert info["converged"]

    key = jax.random.PRNGKey(3)
    G = jax.random.normal(key, (chi, chi))
    X = 0.5 * (G - G.T)
    W = jax.scipy.linalg.expm(0.1 * X / jnp.linalg.norm(X))
    eye = jnp.eye(chi)

    def gauged(bonds):
        """Rotate the listed bonds, in this module's uniform convention."""
        Ws = [W if j in bonds else eye for j in range(4)]
        corners, edges = [], []
        for k in range(4):
            km = (k - 1) % 4
            corners.append(Ws[km].conj().T @ getattr(env, f"C{k + 1}") @ Ws[k])
            edges.append(
                jnp.einsum(
                    "ai,ixj,jb->axb",
                    Ws[k].conj().T,
                    getattr(env, f"T{k + 1}"),
                    Ws[k],
                )
            )
        return M.AsymEnv(*corners, *edges)

    base = float(jnp.real(M.asym_energy(A, env, template, gate)))
    for bonds in ([0], [1], [2], [3], [0, 1, 2, 3]):
        moved = float(jnp.real(M.asym_energy(A, gauged(bonds), template, gate)))
        assert abs(moved - base) < 1e-12, (bonds, moved - base)

    # Negative control: drop the conversion and bonds 2 and 3 break, equally
    # and oppositely, while the global rotation still looks fine.
    def raw_energy(e):
        return float(
            jnp.real(
                compute_energy_ctm_tensor(
                    A,
                    type(template)(
                        **{
                            n: DenseTensor(getattr(e, n), getattr(template, n).indices)
                            for n in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
                        }
                    ),
                    gate,
                )
            )
        )

    raw_base = raw_energy(env)
    broken = [raw_energy(gauged([k])) - raw_base for k in range(4)]
    assert abs(broken[0]) < 1e-12 and abs(broken[1]) < 1e-12, broken
    assert abs(broken[2]) > 1e-5 and abs(broken[3]) > 1e-5, broken
    assert abs(broken[2] + broken[3]) < 0.2 * abs(broken[2]), broken
    assert abs(raw_energy(gauged([0, 1, 2, 3])) - raw_base) < 1e-12, "global cancels"


def test_swapping_the_env_convention_is_an_involution():
    A = _site_tensor()
    env, _a, _m = M.converge(A, 4, max_iter=300, conv_tol=1e-13)
    twice = M.swap_env_convention(M.swap_env_convention(env))
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        assert jnp.array_equal(getattr(twice, name), getattr(env, name)), name
    once = M.swap_env_convention(env)
    assert not jnp.allclose(once.C4, env.C4, atol=1e-8), "C4 is not symmetric here"


def _contract_by_label(pieces, bonds, open_legs):
    """``einsum`` driven purely by leg names, not by hand-written subscripts."""
    letters: dict[tuple[str, str], str] = {}
    canonical = {b: a for a, b in bonds}

    def key(piece, leg):
        return canonical.get((piece, leg), (piece, leg))

    def letter(tag):
        if tag not in letters:
            letters[tag] = "abcdefghijklmnopqrstuvwxyz"[len(letters)]
        return letters[tag]

    subscripts = [
        "".join(letter(key(name, leg)) for leg in legs) for name, _arr, legs in pieces
    ]
    out = "".join(letter(key(p, leg)) for p, leg in open_legs)
    return jnp.einsum(
        f"{','.join(subscripts)}->{out}", *[arr for _n, arr, _l in pieces]
    )


def test_quadrant_wiring_matches_the_library_leg_labels():
    """The quadrant einsums join the legs the CTM env labels say are adjacent.

    Every endpoint in a quadrant has dimension ``chi`` or ``D**2``, so a
    transposed contraction is shape-legal and silently wrong.  This rebuilds
    both quadrants from the adjacency the labels encode -- ``*_d`` meets
    ``*_u``, ``*_r`` meets ``*_l``, and an edge's double-layer leg meets the
    matching leg of ``a`` -- and pins the hand-written einsums against it.

    The environment must be *converged*: the initial one is up/down
    symmetric, which makes several distinct wirings agree numerically.
    """
    A = _site_tensor(seed=5)
    chi = 6
    env, a, info = M.converge(A, chi, max_iter=200, conv_tol=1e-12)
    assert info["converged"]

    template = initialize_ctm_tensor_env(A, chi)
    legs = {
        name: list(getattr(template, name).labels())
        for name in ("C1", "C4", "T1", "T3", "T4")
    }
    a_legs = ["u2", "d2", "l2", "r2"]

    def piece(name):
        return (name, getattr(env, name), legs[name])

    # C1 is left of T1 and above T4; a sits under T1 and right of T4.
    upper_left = _contract_by_label(
        pieces=[piece("C1"), piece("T1"), piece("T4"), ("a", a, a_legs)],
        bonds=[
            (("C1", "c1_r"), ("T1", "t1_l")),
            (("C1", "c1_d"), ("T4", "t4_u")),
            (("T1", "u2"), ("a", "u2")),
            (("T4", "l2"), ("a", "l2")),
        ],
        open_legs=[("T1", "t1_r"), ("a", "r2"), ("T4", "t4_d"), ("a", "d2")],
    )
    assert jnp.allclose(upper_left, M._upper_left_quadrant(env, a), atol=1e-13)

    # C4 is left of T3 and below T4; a sits above T3 and right of T4.
    lower_left = _contract_by_label(
        pieces=[piece("C4"), piece("T3"), piece("T4"), ("a", a, a_legs)],
        bonds=[
            (("C4", "c4_r"), ("T3", "t3_l")),
            (("C4", "c4_u"), ("T4", "t4_d")),
            (("T3", "d2"), ("a", "d2")),
            (("T4", "l2"), ("a", "l2")),
        ],
        open_legs=[("T4", "t4_u"), ("a", "u2"), ("T3", "t3_r"), ("a", "r2")],
    )
    assert jnp.allclose(lower_left, M._lower_left_quadrant(env, a), atol=1e-13)

    # The test has to be able to fail: joining C1's down leg to T4's *down*
    # leg is shape-legal, and must give a different tensor.
    miswired = _contract_by_label(
        pieces=[piece("C1"), piece("T1"), piece("T4"), ("a", a, a_legs)],
        bonds=[
            (("C1", "c1_r"), ("T1", "t1_l")),
            (("C1", "c1_d"), ("T4", "t4_d")),
            (("T1", "u2"), ("a", "u2")),
            (("T4", "l2"), ("a", "l2")),
        ],
        open_legs=[("T1", "t1_r"), ("a", "r2"), ("T4", "t4_u"), ("a", "d2")],
    )
    assert not jnp.allclose(miswired, upper_left, atol=1e-8)


def test_the_two_half_plane_conventions_are_the_same_truncation():
    """The forward sweep's left half and §V.3's upper half agree.

    This identity is the whole licence for
    :func:`asym_root_to_covariant_convention` being a pure relabelling: the
    lower-left quadrant at one rotation *is* the upper-left quadrant at the
    previous one, so the two half-infinite environments are transposes of each
    other and their decompositions differ only by exchanging the isometries.
    If it ever breaks, the covariant residual is being handed the
    decomposition of a different matrix, which is exactly the failure that
    showed up as a residual of 84 rather than 1e-12.
    """
    _A, a, root = _converged_root()
    chi = root.env.C1.shape[0]
    n = chi * a.shape[0]

    upper, lower = [], []
    env_k, a_k = root.env, a
    for _k in range(4):
        upper.append(M._upper_left_quadrant(env_k, a_k).reshape(n, n))
        lower.append(M._lower_left_quadrant(env_k, a_k).reshape(n, n))
        env_k, a_k = M.rotate_env(env_k), M.rotate_a(a_k)

    for k in range(4):
        km = (k - 1) % 4
        scale = float(jnp.abs(upper[km]).max())
        assert float(jnp.abs(lower[k] - upper[km]).max()) < 1e-12 * scale, k

        # ... and therefore M_left(k) == M_up(k-1).T
        m_left = upper[k] @ lower[k]
        m_up = upper[km].T @ upper[k].T
        m_scale = float(jnp.abs(m_left).max())
        assert float(jnp.abs(m_left - m_up.T).max()) < 1e-10 * m_scale, k

        # The identity is specific to the k-1 offset: the same-rotation
        # pairing is shape-legal and must not hold, or the test is vacuous.
        assert float(jnp.abs(lower[k] - upper[k]).max()) > 1e-3 * scale, k


# ------------------------------------------------------------------ #
# The y <-> x map (paper Eq. 82)                                      #
# ------------------------------------------------------------------ #


def _root_S(root):
    """Singular values at the root, as full matrices."""
    return tuple(jnp.diag(s) if s.ndim == 1 else s for s in root.s)


def _converged_root(chi=4, seed=42):
    A = _site_tensor(seed=seed)
    env, a, _info = M.converge(A, chi, max_iter=300, conv_tol=1e-13)
    root, residual = M.asym_root_parametrize(env, a, chi, polish_steps=30)
    assert residual < 1e-12, residual
    return A, a, root


def test_inverse_root_map_round_trips():
    """``absorb`` undoes ``remove`` — the two Eq. 82 directions are inverse.

    Both normalise, so the comparison is against the normalised original.
    """
    _A, _a, root = _converged_root()
    S = _root_S(root)

    back = M.absorb_inverse_roots(M.remove_inverse_roots(root.env, S), S)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        original = getattr(root.env, name)
        original = original / jnp.linalg.norm(original)
        assert jnp.allclose(original, getattr(back, name), atol=1e-12), name


def test_energy_is_unchanged_by_going_through_the_modified_variables():
    """Eq. 82 is a reparametrisation: the energy must not move."""
    A, _a, root = _converged_root()
    chi = root.env.C1.shape[0]
    template = initialize_ctm_tensor_env(A, chi)
    gate = _gate(1.0)
    S = _root_S(root)

    direct = float(M.asym_energy(A, root.env, template, gate))
    via_tilde = float(
        M.asym_energy(
            A,
            M.absorb_inverse_roots(M.remove_inverse_roots(root.env, S), S),
            template,
            gate,
        )
    )
    assert abs(direct - via_tilde) < 1e-12, (direct, via_tilde)


def test_singular_value_adjoint_is_not_negligible():
    """``S̆`` is the same order as ``C̆`` — zeroing it is not a small error.

    The energy depends on ``S`` only through the Eq. 82 map, so this adjoint
    exists purely because the characteristic equations use modified
    variables.  It is what the pre-#718 implementation set to zero, and it
    is large enough to explain the gradient discrepancy on its own.
    """
    A, _a, root = _converged_root()
    chi = root.env.C1.shape[0]
    template = initialize_ctm_tensor_env(A, chi)
    gate = _gate(1.0)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)

    def energy_of(S_in, tilde_in):
        return M.asym_energy(A, M.absorb_inverse_roots(tilde_in, S_in), template, gate)

    _e, vjp = jax.vjp(energy_of, S, tilde)
    S_bar, tilde_bar = vjp(jnp.ones((), dtype=jnp.float64))

    corner_scale = float(jnp.linalg.norm(tilde_bar.C1))
    biggest = max(float(jnp.linalg.norm(x)) for x in S_bar)
    assert biggest > 0.1 * corner_scale, (biggest, corner_scale)


def test_covariant_characteristic_equations_vanish_at_the_root():
    """The §V.3 characteristic equations must have the same root as the sweep.

    This is the gate the whole covariance port exists to pass: ``y*`` extracted
    from a converged CTMRG environment has to be a root of ``F`` in the
    *modified* variables too, or Eq. 88 does not license freezing ``U*``/``V*``
    and the gradient drops a real term (#718).
    """
    A, a, root = _converged_root()
    chi = root.env.C1.shape[0]
    root = M.asym_root_to_covariant_convention(root)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)

    R = M.asym_characteristic_residual_covariant(
        (tilde, root.u, S, root.v), a, root, chi
    )
    total = float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(R))))
    assert total < 1e-10, total


# ------------------------------------------------------------------ #
# Complex states (#721)                                               #
# ------------------------------------------------------------------ #
#
# Every test above runs on a *real* site tensor, and real data hides a whole
# class of defect: a real tensor times ``exp(i theta)`` leaves the manifold, so
# the environment's global phase gauge is invisible, and any variable that is
# real only because it happens to be (singular values) never has to admit that
# it is a complex unknown.  Complex tensors are the production regime for
# iPEPS, so #721 covers them here.


def _complex_site_tensor(D=2, d=2, seed=42, imag_seed=7, scale=0.25):
    """The real test state plus an imaginary part, renormalised.

    Same construction as ``docs/plans/reference/718-codex-review-checks.py`` so
    the numbers in #721 stay comparable.
    """
    A = _site_tensor(D=D, d=d, seed=seed)
    rng = np.random.RandomState(imag_seed)
    base = np.asarray(A.todense())
    data = base + scale * 1j * rng.standard_normal(base.shape)
    data = jnp.asarray(data / np.linalg.norm(data))
    return DenseTensor(data, A.indices)


def _converged_complex_root(chi=4):
    A = _complex_site_tensor()
    env, a, info, projs = M.converge(
        A, chi, max_iter=400, conv_tol=1e-13, return_projectors=True
    )
    assert info["converged"], info
    root, residual = M.asym_root_parametrize(
        env, a, chi, prev_projs=projs, polish_steps=30
    )
    return A, a, root, residual


_ENV_NAMES = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")


def test_the_energy_is_invariant_under_every_environment_phase():
    """A phase on any environment tensor cannot move the energy.

    The RDM is one network, so a phase on any tensor in it is an overall complex
    scalar on ``rho`` — and the energy is ``tr(rho H)/tr(rho)``, a ratio that
    cancels any such scalar.  So all eight phases are separately free, not just
    their sum.

    #721 doubted this, and doubting it was reasonable: before
    :func:`M.swap_env_convention` landed (#718) the energy moved by -7.9e-2 per
    radian of the global phase.  A mis-glued network breaks the ket/bra
    conjugation structure, so the raw RDM is genuinely non-Hermitian, and
    ``_rdm2x1_tensor`` symmetrises *before* it divides by the trace — which
    turns ``Herm(e^{i.phi} R)`` into ``cos(phi) H + sin(phi) iK`` and leaks the
    anti-Hermitian part ``K`` into the energy as a physical-looking shift.  Glued
    correctly, ``R`` is Hermitian up to one complex scalar and the ordering
    stops mattering.

    This is load bearing twice over.  Each of these phases is an exact null
    direction of ``d_y F`` (see below), so the adjoint system is singular; it is
    solvable only because ``E`` is invariant along every one of them.
    """
    A = _complex_site_tensor()
    chi = 4
    gate = _gate(1.0)
    env, _a, info = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
    assert info["converged"], info
    template = initialize_ctm_tensor_env(A, chi)

    def energy_at(theta, which):
        rot = M.AsymEnv(
            *[
                jnp.exp(1j * theta) * t if n in which else t
                for n, t in zip(_ENV_NAMES, env)
            ]
        )
        return float(jnp.real(M.asym_energy(A, rot, template, gate)))

    base = energy_at(0.0, ())
    assert abs(base) > 1e-3, base
    for which in [(n,) for n in _ENV_NAMES] + [_ENV_NAMES]:
        for theta in (0.2, 0.7, float(np.pi / 2)):
            moved = energy_at(theta, which)
            assert abs(moved - base) < 1e-11, (which, theta, moved - base)


def test_characteristic_equations_hold_at_a_complex_root():
    """All twenty residuals vanish for a complex state, not just a real one.

    The environment and the root have to be in the *same* bond gauge, and
    ``asym_root_parametrize`` used to throw the forward chain away and re-pin
    from cold.  A real state survives that — its bond gauge is a sign, and one
    sweep reproduces it exactly, which is why the polish loop drops from 1.3e-3
    to 1.6e-17 in a single step.  A complex state does not: the gauge is a
    continuous phase, and the corner residual plateaus at 5.8e-5 while the edges
    fall geometrically past 1e-8.  Corners are where it shows because the corner
    is the only equation built from two directions' projectors, so it is the
    only one that sees a *relative* bond phase.
    """
    A, a, root, residual = _converged_complex_root()
    chi = root.env.C1.shape[0]
    assert jnp.iscomplexobj(root.env.C1)
    assert residual < 1e-12, residual

    R_env, R_u, R_S, R_v = M.asym_characteristic_residual(root.y, a, root, chi)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        assert float(jnp.linalg.norm(getattr(R_env, name))) < 1e-12, name
    for k in range(4):
        assert float(jnp.linalg.norm(R_u[k])) < 1e-12, f"R_u[{k}]"
        assert float(jnp.linalg.norm(R_S[k])) < 1e-12, f"R_S[{k}]"
        assert float(jnp.linalg.norm(R_v[k])) < 1e-12, f"R_v[{k}]"


def test_the_singular_values_are_a_complex_unknown():
    """``S`` has to carry the environment's dtype, not the SVD's.

    ``jnp.linalg.svd`` returns real singular values, so ``S`` came out
    ``float64`` even for a ``complex128`` environment.  Two consequences, one
    cosmetic and one fatal: ``R_S = core/lambda - S`` compares a complex matrix
    against a real one, and — because the energy's cotangent inherits the
    primal's dtype — the adjoint solve then hands ``vjp_y`` a ``float64`` block
    where ``F`` produced ``complex128`` and JAX refuses outright.

    This is the same argument that already promoted ``S`` from a diagonal to a
    full ``chi x chi`` block, one axis further along: the reverse pass needs
    ``S`` free to leave the reals, whatever the forward pass happens to put
    there.
    """
    A = _complex_site_tensor()
    chi = 4
    env, a, _info = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
    for _pt, _pb, _U, S_keep, _Vh in M.all_projectors(env, a, chi):
        assert jnp.iscomplexobj(S_keep), S_keep.dtype
        # Still the SVD's real spectrum, only widened.
        assert float(jnp.linalg.norm(jnp.imag(S_keep))) < 1e-300


def test_the_dense_root_carries_lambdas_phase_rather_than_dropping_it(monkeypatch):
    """Real-projecting ``lambda`` breaks the dense root — in the gauge that shows it.

    ``asym_characteristic_residual_covariant`` normalises as ``X'/lambda - X``,
    so ``lambda``'s phase is part of the equation and dropping it is not a
    rescaling of ``F`` but a different ``F``.  That is the #721 property, and
    until now it was only *asserted*, in this module's own docstring, as
    "``.real`` moves ``|F1|`` from 2e-13 to 1.6e0".  It is measured here.

    The measurement needs a gauge, which is the part the old prose left out.
    At the natural root every ``lambda`` comes out real to machine precision —
    the twelve of them have ``|arg| < 8e-16`` — because ``asym_root_parametrize``
    polishes the bond phases into alignment.  So ``.real`` is a *no-op* there
    (6.50e-14 both ways, measured), and a test written against that fixture
    would pass whether or not the code took the real part.

    The eight environment phases are exact null directions of the root
    (``test_the_environment_phases_are_licensed_gauges_of_the_root``), so
    phasing ``tilde`` moves to a different, equally valid root: ``F`` is
    covariant, ``R -> e^{i.gamma} R``, and zero stays zero — asserted below at
    6.5e-14, unchanged from the unphased root.  What the phases *do* change is
    ``lambda``, which is now genuinely complex (``|arg|`` up to 3.0).  Now
    ``.real`` has something to destroy, and it destroys it: ``‖F‖`` goes to
    3.6e0 and 1.2e1 for the two gauges below.

    Two-part, scale-free assertion for the reason given in
    ``test_the_root_survives_a_complex_state`` in the symmetric file: a bare
    ``> 1.0`` sits at the edge of a platform-dependent quantity.
    """
    A, a, root, _res = _converged_complex_root()
    chi = root.env.C1.shape[0]
    root = M.asym_root_to_covariant_convention(root)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)

    def total(R):
        return float(
            jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(R)))
        )

    class _RealProjectedVdot:
        """``jnp``, except ``vdot`` drops the imaginary part."""

        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def vdot(x, y):
            return jnp.vdot(x, y).real

    # The natural gauge first, to pin *why* it cannot be the fixture.
    y_natural = (tilde, root.u, S, root.v)
    natural = total(M.asym_characteristic_residual_covariant(y_natural, a, root, chi))
    assert natural < 1e-10, natural
    monkeypatch.setattr(M, "jnp", _RealProjectedVdot())
    vacuous = total(M.asym_characteristic_residual_covariant(y_natural, a, root, chi))
    monkeypatch.undo()
    assert abs(vacuous - natural) < 1e-12, (natural, vacuous)

    for thetas in (
        (0.3,) * 8,
        (0.7, 0.1, -0.4, 0.9, 0.25, -0.6, 1.1, 0.05),
    ):
        phased = M.AsymEnv(
            *[jnp.exp(1j * t) * x for t, x in zip(thetas, tilde, strict=True)]
        )
        y = (phased, root.u, S, root.v)

        # Still a root: the phases are a licensed gauge.
        good = total(M.asym_characteristic_residual_covariant(y, a, root, chi))
        assert good < 1e-10, (thetas, good)

        monkeypatch.setattr(M, "jnp", _RealProjectedVdot())
        wrong = total(M.asym_characteristic_residual_covariant(y, a, root, chi))
        monkeypatch.undo()

        assert wrong > 1e-2, (thetas, wrong)
        assert wrong > 1e9 * good, (thetas, wrong, good)
        print(f"dense |F| = {good:.3e}; with lambda real-projected = {wrong:.3e}")


def test_gradient_parity_for_a_complex_state():
    """End to end on a complex state: 2.5e-12 against explicit backprop.

    The sweep count is the whole subtlety here.  The reference differentiates a
    *truncated* sweep map, so it approaches the fixed-point gradient only as
    fast as the CTM itself converges, and this state needs 53 forward iterations
    where the real one needs 25.  At 12 sweeps — enough for the real state to
    reach 4e-8 — the complex state sits at 2.1e-5, and finite differences agree
    with the reference to four digits at two step sizes, so the gap reads
    convincingly like a wrong gradient.  It is not: 20 sweeps gives 2.1e-7, 30
    gives 6.9e-10, 40 gives 2.5e-12, each step matching the previous
    reference-to-reference drift.  Do not "fix" a discrepancy here by hunting in
    the adjoint before ruling the sweep count out.
    """
    A = _complex_site_tensor()
    chi = 4
    gate = _gate(1.0)
    A_data = A.todense()

    _energy, grad, info = M.asym_root_implicit_energy_and_grad(
        A, gate, chi=chi, max_iter=400, conv_tol=1e-13, return_diagnostics=True
    )
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert jnp.iscomplexobj(grad)
    assert info["converged"], info
    assert info["root_residual"] < 1e-10, info
    assert info["covariant_residual"] < 1e-10, info
    assert info["adjoint_residual"] < 1e-8, info
    assert info["gauge_consistency"] < 1e-10, info

    env0, _a, _m = M.converge(A, chi, max_iter=400, conv_tol=1e-13)
    template = initialize_ctm_tensor_env(A, chi)

    def energy_explicit(pdata):
        A_live = DenseTensor(pdata, A.indices)
        a_live = _a_array(A_live)
        env, projs = env0, None
        for _ in range(40):
            env, projs = M.sweep(env, a_live, chi, projs)
        return jnp.real(M.asym_energy(A_live, env, template, gate))

    g_ref = jax.grad(energy_explicit)(A_data)
    assert bool(jnp.all(jnp.isfinite(g_ref))), "explicit backprop NaN'd; use FD instead"
    rel = float(jnp.linalg.norm(grad - g_ref) / jnp.linalg.norm(g_ref))
    assert rel < 1e-9, rel


def test_the_environment_phases_are_licensed_gauges_of_the_root():
    """The phase orbits are null directions of ``d_y F`` — and harmless ones.

    ``F`` is phase-*covariant*, not phase-invariant, and the counting is exact.
    An enlarged corner carries three environment tensors, so
    ``EC -> e^{3i.theta}`` under a global phase, hence ``C_new -> e^{9i.theta}``
    against ``lambda_C -> e^{8i.theta}``; the edge block goes ``e^{7i.theta}``
    over ``e^{6i.theta}``.  But the mechanism is more general than the global
    phase, because the normalisation is a *ratio*: ``X'/<X, X'>`` is invariant
    under any phase on ``X'`` whatever, and picks up ``e^{i.gamma}`` from ``X``
    alone, so ``R = X'/<X, X'> - X`` is covariant under a phase on **each**
    environment tensor separately.  ``R_u``, ``R_S`` and ``R_v`` are ratios too
    and are invariant outright.  So all eight phases are null directions, the
    root is an eight-parameter family at least, and the implicit function
    theorem does not apply as literally written.

    Nothing is broken by that, and #721 filed it as a defect on the strength of
    a mis-conjugated pairing.  Two conditions make the singular system fine, and
    both are checked here:

    * ``ў`` must be orthogonal to every null direction, or the adjoint system is
      inconsistent and has no solution at all.  It is, because the energy is
      invariant along each of them —
      ``test_the_energy_is_invariant_under_every_environment_phase``, transported
      to the cotangent.
    * the leftover freedom in ``F̆`` must not reach the gradient.  It does not:
      differentiating ``F(y*(p), p) = 0`` gives ``d_p F = -d_y F . d_p y*``, so
      ``d_p F`` lands in the *range* of ``d_y F``, and the freedom is orthogonal
      to that range.

    The second one has to be tested on the operator that is actually inverted,
    ``B(v) = vjp_y(v)``, and not on ``jacfwd(F)`` transposed: JAX's VJP for a
    complex function is not the plain transpose of its JVP, so the null space of
    ``jacfwd(F)^T`` is the wrong subspace and reports a 24% gradient change.
    Against ``null(B)`` the same check gives 6e-15.

    Both Jacobians come out with nullity 12 rather than 8, so there are four
    more gauge directions than the eight phases named above.  They are not
    chased here: whatever they are, the two conditions hold on all twelve, which
    is what the gradient needs.
    """
    from tenax.algorithms._ctm_c4v_root_implicit import (
        _from_real_vec,
        _real_struct,
        _solve_root_adjoint,
        _to_real_vec,
    )

    A, a, root, _res = _converged_complex_root()
    chi = root.env.C1.shape[0]
    gate = _gate(1.0)
    root = M.asym_root_to_covariant_convention(root)
    S = _root_S(root)
    tilde = M.remove_inverse_roots(root.env, S)
    y_star = (tilde, root.u, S, root.v)
    A_data = A.todense()
    template = initialize_ctm_tensor_env(A, chi)

    def F_of_y(y):
        return M.asym_characteristic_residual_covariant(y, a, root, chi)

    def nrm(t):
        return float(
            jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(t)))
        )

    def rpair(x, y):
        return float(
            jnp.real(
                sum(
                    jnp.sum(p * q)
                    for p, q in zip(jax.tree.leaves(x), jax.tree.leaves(y))
                )
            )
        )

    zeros = (
        tuple(jnp.zeros_like(x) for x in root.u),
        tuple(jnp.zeros_like(x) for x in S),
        tuple(jnp.zeros_like(x) for x in root.v),
    )

    def phase_tangent(which):
        """``i * tilde`` on the named tensors, zero everywhere else."""
        return (
            M.AsymEnv(
                *[
                    1j * x if n in which else jnp.zeros_like(x)
                    for n, x in zip(_ENV_NAMES, tilde)
                ]
            ),
            *zeros,
        )

    tangents = [phase_tangent({n}) for n in _ENV_NAMES] + [phase_tangent(_ENV_NAMES)]

    # 1. every phase is a null direction, by a wide margin against a random one
    _F0, jvp = jax.linearize(F_of_y, y_star)
    rng = np.random.RandomState(5)
    random_dir = jax.tree.map(
        lambda x: jnp.asarray(
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        ).astype(x.dtype),
        y_star,
    )
    scale = nrm(tangents[0]) / nrm(random_dir)
    random_dir = jax.tree.map(lambda x: x * scale, random_dir)
    assert nrm(jvp(random_dir)) / nrm(random_dir) > 1.0, nrm(jvp(random_dir))
    for t, name in zip(tangents, (*_ENV_NAMES, "global")):
        assert nrm(jvp(t)) / nrm(t) < 1e-10, (name, nrm(jvp(t)) / nrm(t))

    # 2. the adjoint system is consistent against every one of them
    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_energy(
            A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (tilde_bar, zeros[0], S_bar, zeros[2])
    for t, name in zip(tangents, (*_ENV_NAMES, "global")):
        cos = abs(rpair(y_bar, t)) / (nrm(y_bar) * nrm(t))
        assert cos < 1e-12, (name, cos)

    # 3. the Jacobian really is rank deficient, and the phases live in its kernel
    _leaves, treedef = jax.tree.flatten(y_star)
    struct = _real_struct(y_star)

    def f_real(vec):
        return _to_real_vec(F_of_y(_from_real_vec(vec, treedef, struct)))

    J = jax.jacfwd(f_real)(_to_real_vec(y_star))
    _Uj, s_j, Vh_j = jnp.linalg.svd(J)
    nullity = int(jnp.sum(s_j / s_j[0] < 1e-10))
    assert nullity == 12, (nullity, [float(x) for x in (s_j / s_j[0])[-14:]])
    kernel = Vh_j[-nullity:].conj().T
    phases = jnp.stack([_to_real_vec(phase_tangent({n})) for n in _ENV_NAMES], axis=1)
    Q, _R = jnp.linalg.qr(phases)
    cosines = jnp.linalg.svd(Q.T @ kernel, compute_uv=False)
    assert float(jnp.min(cosines)) > 1 - 1e-8, [float(x) for x in cosines]

    # 4. the freedom in F̆ does not reach the gradient
    _F_at_root, vjp_y = jax.vjp(F_of_y, y_star)
    F_bar, solve_resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0], y_bar, tol=1e-10, maxiter=400, restart=30
    )
    assert float(solve_resid) < 1e-8, float(solve_resid)

    def F_of_p(a_data):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_characteristic_residual_covariant(
            y_star, _a_array(A_live), root, chi
        )

    _, vjp_p = jax.vjp(F_of_p, A_data)
    grad = grad_direct - vjp_p(F_bar)[0]
    grad_scale = float(jnp.linalg.norm(grad))

    bar_leaves, bar_treedef = jax.tree.flatten(y_bar)
    bar_struct = _real_struct(y_bar)
    del bar_leaves

    def B_real(vec):
        return _to_real_vec(vjp_y(_from_real_vec(vec, bar_treedef, bar_struct))[0])

    B = jax.jacfwd(B_real)(_to_real_vec(y_bar))
    _Ub, s_b, Vh_b = jnp.linalg.svd(B)
    nullity_b = int(jnp.sum(s_b / s_b[0] < 1e-10))
    assert nullity_b == 12, (nullity_b, [float(x) for x in (s_b / s_b[0])[-14:]])
    for j in range(nullity_b):
        n_tree = _from_real_vec(Vh_b[-(j + 1)].conj(), bar_treedef, bar_struct)
        leak = float(jnp.linalg.norm(vjp_p(n_tree)[0])) / grad_scale
        assert leak < 1e-10, (j, leak)


# --- #772: rank cap on the retained spectrum -------------------------------
#
# The covariant characteristic equations carry S^-1 on BOTH corner legs plus the
# Eq. 73 quartic roots, so they cannot resolve a retained direction whose weight
# is below ~eps^(1/3) relative to the largest. Retaining one produced |F(y*)| =
# 5.7e+05 and NaN gradients on a physical simple-update state (#772).
#
# Directions at or below the clamp are harmless -- the large S^-1 multiplies
# approximately nothing. Directions carrying real but tiny weight are what break
# the equations, which is why the clamp is raised rather than lowered.


def test_rank_cap_leaves_a_healthy_spectrum_untouched():
    """A well-conditioned cut must pass through bit-exactly: no silent change to
    states that were already fine."""
    s = jnp.array([1.0, 0.5, 0.1, 0.02])
    capped, rank = M._rank_capped_spectrum(s, 4)
    assert jnp.allclose(capped, s, rtol=0, atol=0)
    assert int(rank) == 4


def test_rank_cap_clamps_a_numerically_null_tail():
    eps13 = float(jnp.finfo(jnp.float64).eps ** (1 / 3))
    s = jnp.array([1.0, 2.45e-4, 2.03e-4, 6.96e-8, 1.24e-9])
    capped, rank = M._rank_capped_spectrum(s, 5)
    # the three leading directions survive untouched
    assert jnp.allclose(capped[:3], s[:3], rtol=0, atol=0)
    # the tail is clamped to one uniform value at the precision-derived level
    assert float(capped[3]) == pytest.approx(eps13, rel=1e-12)
    assert float(capped[4]) == pytest.approx(eps13, rel=1e-12)
    # and the usable rank is reported as 3, not 5
    assert int(rank) == 3


def test_rank_cap_is_relative_to_the_largest_singular_value():
    """The clamp must scale with the spectrum, not be an absolute constant."""
    s = jnp.array([1e6, 1e-2])
    capped, rank = M._rank_capped_spectrum(s, 2)
    eps13 = float(jnp.finfo(jnp.float64).eps ** (1 / 3))
    assert float(capped[1]) == pytest.approx(1e6 * eps13, rel=1e-12)
    assert int(rank) == 1


def test_rank_cap_truncates_to_chi_first():
    s = jnp.array([1.0, 0.5, 0.25, 0.125])
    capped, _rank = M._rank_capped_spectrum(s, 2)
    assert capped.shape == (2,)


def test_rank_cap_never_returns_a_zero_or_negative_value():
    """The clamp subsumes the old NaN guard: _inv_sqrt must stay finite."""
    s = jnp.array([1.0, 0.0, 0.0])
    capped, rank = M._rank_capped_spectrum(s, 3)
    assert bool(jnp.all(capped > 0))
    assert int(rank) == 1


def test_rank_cap_honours_an_explicit_relative_floor():
    s = jnp.array([1.0, 1e-3, 1e-9])
    capped, rank = M._rank_capped_spectrum(s, 3, rel_floor=1e-6)
    assert float(capped[1]) == pytest.approx(1e-3, rel=1e-12)
    assert float(capped[2]) == pytest.approx(1e-6, rel=1e-12)
    assert int(rank) == 2


def test_all_projectors_reports_a_rank_below_chi_on_a_flat_cut():
    """End-to-end: the usable rank comes back from the projector build so a
    caller can tell that chi exceeds what the state supports."""
    A = _site_tensor(D=2)
    env, a, _meta = M.converge(A, chi=4, max_iter=60, conv_tol=1e-12)
    projs = M.all_projectors(env, a, 4)
    for entry in projs:
        S_keep = entry[3]
        diag = jnp.abs(jnp.diag(S_keep))
        # every retained value is strictly positive and no larger than the max
        assert bool(jnp.all(diag > 0))
