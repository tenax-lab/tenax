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
