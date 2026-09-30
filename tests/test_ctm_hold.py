"""CTM hold test (#1035): successive-sweep agreement does not certify an attractor.

The eager 2-site CTM certified a SADDLE S of the fermionic t-V D=3 chi=12 step
(successive sweeps agree to 1e-10, yet a displacement grows x1.041/sweep and
the loop, given ~650 more sweeps, lands on a stable fixed point B 1.4e-2
away).  These tests pin the mechanism with maps whose stability is known by
construction -- not by running a real CTM to convergence -- plus one real
dense CTM that must still pass.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_tensor_convergence as conv
from tenax.algorithms._ctm_hold import (
    env_invariant_distance,
    env_spectral_invariants,
    hold_test,
    perturb_env,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    _ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# hold_test on linear maps of known spectrum                                   #
# --------------------------------------------------------------------------- #


class _Vec(NamedTuple):
    x: jax.Array


def _vec_env(v):
    return {(0, 0): _Vec(jnp.asarray(v, dtype=jnp.float64))}


def _identity_invariants(envs):
    return {"x": np.asarray(envs[(0, 0)].x)}


def _linear_step(J, xstar):
    J = jnp.asarray(J)

    def step(envs):
        x = envs[(0, 0)].x
        return _vec_env(xstar + J @ (x - xstar))

    return step


XSTAR = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])


def test_a_saddle_fails_the_hold():
    """Five stable directions at 0.87 and one unstable at 1.041 -- the #1035
    measured rates.  The unstable direction is 1/6 of a random perturbation."""
    J = np.diag([0.87, 0.87, 0.87, 0.87, 0.87, 1.041])
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert not held.passed
    assert held.rate > 1.0
    # A saddle never passes a re-test either: it runs the full 3*K extension,
    # three CTM steps per sweep (the point and two perturbed directions).
    assert held.sweeps == 3 * 3 * 40
    assert all(r > 1.0 for r in held.rates)


def test_an_attractor_with_a_transient_passes_where_d_k_below_d_0_would_not():
    """Non-normal but stable (eigenvalues 0.95, 0.9): the displacement first
    grows ~20x -- the attractor B of #1035 grows ~25x -- then decays.  The
    fitted tail rate is < 1 and passes; the naive ``d_K < d_0`` would still
    read "unstable" at K=40, which is why the hold does not use it."""
    J = np.eye(6) * 0.9
    J[0, 0] = 0.95
    J[0, 1:] = 3.0  # strong non-normal coupling into the slow direction
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert held.rate < 1.0
    # Regime, on a plain (un-renormalised) perturbed trajectory: the
    # transient is real, and d_K < d_0 would have failed this attractor.
    step = _linear_step(J, XSTAR)
    x, y = _vec_env(XSTAR), perturb_env(_vec_env(XSTAR), 1e-6, jax.random.PRNGKey(0))
    d = [env_invariant_distance(_identity_invariants(y), _identity_invariants(x))]
    for _ in range(40):
        x, y = step(x), step(y)
        d.append(
            env_invariant_distance(_identity_invariants(y), _identity_invariants(x))
        )
    assert max(d) > 10 * d[0]
    assert d[40] > d[0]


def test_a_claimed_point_off_by_its_residual_still_holds():
    """The loop certifies a point only to ``conv_tol``: here the claimed
    point sits 1e-7 from the true attractor, so the perturbed trajectory's
    distance to the CLAIMED point plateaus at 1e-7.  The verdict compares
    against the co-evolved unperturbed trajectory, which has no such floor,
    and recovers the true rate.
    (The discriminating case -- a plateau that fits to a rate >= 1 -- is the
    mocked loop below, whose attractor the distance rule rejected once.)"""
    J = np.eye(6) * 0.9
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR + 1e-7),
        sweeps=40,
        perturbation=1e-6,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert abs(held.rate - 0.9) < 1e-6
    # Regime: the perturbed trajectory ends 1e-7 from the claimed point --
    # the plateau a distance-to-claimed-point statistic would sit on.
    assert float(jnp.max(jnp.abs(held.envs[(0, 0)].x - (XSTAR + 1e-7)))) > 0.5e-7


def test_a_transient_that_outlasts_the_first_window_is_not_a_saddle():
    """A Jordan block at 0.975: ``d_k ~ k 0.975^k`` peaks at k~40, so the fit
    over [20, 40] still grows (0.975 * 2**(1/20) = 1.009).  Measured on the
    real #1035 attractor too (1.019 at K=40, 0.959 at K=60).  The re-test on
    the next window passes it; with no extension it would be rejected."""
    J = np.eye(6) * 0.5
    J[0, 0] = J[1, 1] = 0.975
    J[0, 1] = 5.0
    kw = dict(sweeps=40, invariants=_identity_invariants)
    rejected = hold_test(_linear_step(J, XSTAR), _vec_env(XSTAR), max_sweeps=40, **kw)
    assert not rejected.passed  # regime: the first window alone gets it wrong
    held = hold_test(_linear_step(J, XSTAR), _vec_env(XSTAR), **kw)
    assert held.passed
    assert 2 * 40 < held.sweeps < 2 * 3 * 40


class _Two(NamedTuple):
    a: jax.Array
    b: jax.Array


def test_a_weakly_excited_saddle_that_contracts_first_still_fails():
    """Codex P1 on #1058.  The unstable direction lives in a leaf whose scale
    is 1e-6, so a relative perturbation excites it at ~1e-12 while the stable
    leaf moves at ~1e-6.  The stable part decays at 0.3/sweep: the
    displacement contracts by 1e-3 within ~6 sweeps -- and then the unstable
    part, growing at 1.05/sweep, takes over.  Accepting on early contraction
    passed this saddle."""
    astar = jnp.linspace(1.0, 0.5, 5)
    bstar = jnp.full(3, 1e-6)

    def step(envs):
        e = envs[(0, 0)]
        return {(0, 0): _Two(astar + 0.3 * (e.a - astar), bstar + 1.05 * (e.b - bstar))}

    def inv(envs):
        e = envs[(0, 0)]
        return {"a": np.asarray(e.a), "b": np.asarray(e.b)}

    start = {(0, 0): _Two(astar, bstar)}
    # Regime: without renormalisation the displacement really does contract
    # by 1e-3 of its peak before the unstable part shows (the old early pass).
    y = perturb_env(start, 1e-6, jax.random.PRNGKey(0))
    x, d = start, []
    for _ in range(12):
        x, y = step(x), step(y)
        d.append(env_invariant_distance(inv(y), inv(x)))
    assert min(d) < 1e-3 * max(d)
    held = hold_test(step, start, sweeps=40, invariants=inv)
    assert not held.passed
    assert held.rate > 1.0


def test_every_direction_must_contract_not_just_one():
    """Codex P1 on #1058 asked for >= 2 independent directions, accepted only
    if ALL tails contract.  A one-sided instability makes that observable:
    the ``b[0]`` coordinate escapes at 1.05/sweep when displaced upward and
    contracts at 0.5 when displaced downward, so a perturbation that happens
    to land on the stable side sees an attractor.  The key is chosen so the
    two split directions land on opposite sides and the unsplit key on the
    stable side -- one direction, or ``any`` instead of ``all``, passes it."""
    astar = jnp.linspace(1.0, 0.5, 5)
    bstar = jnp.full(3, 0.5)

    def step(envs):
        e = envs[(0, 0)]
        db = e.b - bstar
        g = jnp.where(db[0] > 0, 1.05, 0.5)
        return {(0, 0): _Two(astar + 0.3 * (e.a - astar), bstar + g * db)}

    def inv(envs):
        e = envs[(0, 0)]
        return {"a": np.asarray(e.a), "b": np.asarray(e.b)}

    start = {(0, 0): _Two(astar, bstar)}

    def side(k):
        return float(perturb_env(start, 1e-6, k)[(0, 0)].b[0] - bstar[0]) > 0

    for n in range(100):
        key = jax.random.PRNGKey(n)
        k0, k1 = jax.random.split(key, 2)
        if side(k0) != side(k1) and not side(key):
            break
    else:
        pytest.fail("no key splits the two directions across the fold")

    held = hold_test(step, start, sweeps=40, invariants=inv, key=key)
    assert not held.passed
    assert sorted(r > 1.0 for r in held.rates) == [False, True]


def test_a_fast_attractor_passes_at_the_first_window():
    """No early exit any more (Codex P1): even a fast attractor is judged on
    its tail.  Renormalisation keeps the displacement off the float floor, so
    the fit reads the true rate."""
    J = np.eye(6) * 0.1
    held = hold_test(
        _linear_step(J, XSTAR),
        _vec_env(XSTAR),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert held.passed
    assert held.sweeps == 3 * 40
    assert all(abs(r - 0.1) < 1e-3 for r in held.rates)


def test_a_locally_constant_step_is_an_attractor():
    """Codex P2 on #1058 (_ctm_hold.py:472): a step that maps every nearby
    point exactly onto the fixed point collapses the displacement to 0.  The
    rescale cannot restore ``y - x = 0``; the direction must be re-seeded and
    the collapse counted as a large contraction, not as a flat log (rate 1,
    a false rejection of the most attracting map there is)."""

    def step(envs):
        return _vec_env(XSTAR)

    held = hold_test(step, _vec_env(XSTAR), sweeps=40, invariants=_identity_invariants)
    assert held.passed
    assert all(r < 1e-3 for r in held.rates)


def test_the_default_metric_is_blind_to_a_period_two_sign_cycle():
    """An attractor whose step flips the sign of alternate rows every sweep --
    the gauge-covariant env's +-1 period-2 cycle.  Element-wise, the
    trajectory never approaches the claimed point; in the spectral invariants
    it contracts, and the hold passes."""
    rng = np.random.default_rng(0)
    Cstar = jnp.asarray(rng.standard_normal((5, 5)))
    sigma = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0])[:, None]

    def step(envs):
        # Stateless period-2 gauge cycle: find which of {C*, sigma C*} the
        # input sits near (g), contract the displacement, and emit it next to
        # the OTHER one.  The sign pattern never settles; the spectrum does.
        C = envs[(0, 0)].x
        g = (
            1.0
            if jnp.sum((C - Cstar) ** 2) <= jnp.sum((C - sigma * Cstar) ** 2)
            else sigma
        )
        delta = C - g * Cstar
        return {(0, 0): _Vec(sigma * g * Cstar + 0.5 * sigma * delta)}

    start = {(0, 0): _Vec(Cstar)}
    # Regime: element-wise, one sweep moves the claimed point by O(1) -- the
    # element-wise metric would never see this attractor converge.
    assert float(jnp.max(jnp.abs(step(start)[(0, 0)].x - Cstar))) > 0.1
    held = hold_test(step, start, sweeps=20)
    assert held.passed


class _Edge(NamedTuple):
    T: jax.Array


def _rotation_saddle():
    """An edge tensor ``T[a, m, b]`` whose unstable mode is a pure rotation of
    the middle (D^2) leg: ``T -> T* x_2 R(theta)`` with ``theta <- 1.05
    theta``, while every other displacement decays at 0.8/sweep.  A rotation
    of the D^2 leg leaves every per-leg singular-value spectrum unchanged,
    but it is NOT a gauge -- that leg contracts against the fixed double
    layer -- so this is a saddle (Codex P1 on #1058, _ctm_hold.py:92)."""
    import jax.scipy.linalg as jsl

    rng = np.random.default_rng(7)
    Tstar = jnp.asarray(rng.standard_normal((4, 4, 4)))
    K = np.zeros((4, 4))
    K[0, 1], K[1, 0] = 1.0, -1.0
    K = jnp.asarray(K)
    u = jnp.einsum("amb,mn->anb", Tstar, K)  # tangent of the rotation at T*

    def rot(theta):
        return jnp.einsum("amb,mn->anb", Tstar, jsl.expm(theta * K))

    def step(envs):
        T = envs[(0, 0)].T
        theta = jnp.vdot(u, T - Tstar) / jnp.vdot(u, u)
        rest = T - rot(theta)
        return {(0, 0): _Edge(rot(1.05 * theta) + 0.8 * rest)}

    return step, {(0, 0): _Edge(Tstar)}, rot


def test_a_rotation_of_the_d2_leg_is_seen():
    """The per-leg spectra alone are blind to the rotation (regime, asserted
    first); the hold must still reject the saddle."""
    step, start, rot = _rotation_saddle()
    blind = env_spectral_invariants(start)
    moved = {(0, 0): _Edge(rot(0.3))}
    per_leg = {
        k: v
        for k, v in env_spectral_invariants(moved).items()
        if not isinstance(k[2], str)
    }
    per_leg0 = {k: v for k, v in blind.items() if not isinstance(k[2], str)}
    assert env_invariant_distance(per_leg, per_leg0) < 1e-12  # regime: spectra blind
    assert env_invariant_distance(env_spectral_invariants(moved), blind) > 1e-2
    held = hold_test(step, start, sweeps=40)
    assert not held.passed
    assert all(r > 1.0 for r in held.rates)


def test_the_block_sparse_d2_gram_sees_a_rotation_within_a_sector():
    """Same blind spot on the ``SymmetricTensor`` path: rotate two D^2 slots
    of equal charge (a unitary within the sector -- all a symmetric map can
    do).  Per-leg spectra do not move; the block-sparse D^2 Gram does."""
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()
    chi = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    d2 = np.array([0, 0, 0, 1, -1], dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, chi, FlowDirection.IN, label="a"),
        TensorIndex.from_charges(sym, d2, FlowDirection.IN, label="m"),
        TensorIndex.from_charges(sym, chi, FlowDirection.OUT, label="b"),
    )
    T = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(3))
    c, sn = np.cos(0.3), np.sin(0.3)
    R = np.eye(5)
    R[:2, :2] = [[c, -sn], [sn, c]]  # slots 0 and 1 are both charge 0
    Td = np.einsum("amb,mn->anb", np.asarray(T.todense()), R)
    T2 = SymmetricTensor.from_dense(jnp.asarray(Td), idx)
    a = env_spectral_invariants({(0, 0): _Edge(T)})
    b = env_spectral_invariants({(0, 0): _Edge(T2)})
    legs_a = {k: v for k, v in a.items() if not isinstance(k[2], str)}
    legs_b = {k: v for k, v in b.items() if not isinstance(k[2], str)}
    assert env_invariant_distance(legs_a, legs_b) < 1e-12  # regime: spectra blind
    assert env_invariant_distance(a, b) > 1e-2


def _isotropic_slices(chi=4, n=4, seed=11):
    """``T[a, m, b]`` whose D^2 slices are Frobenius-orthonormal: ``G = I``,
    so every D^2 rotation leaves ``G`` -- and every per-leg spectrum --
    unchanged.  The slices are generic (non-commuting), so the rotation still
    changes the state: it is not a gauge."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, chi * chi))
    Q, _ = np.linalg.qr(V.T)  # columns: orthonormal vectorised slices
    return np.transpose(Q.T.reshape(n, chi, chi), (1, 0, 2))


def _rotation_saddle_in(Tstar, plane=(0, 1)):
    """``theta <- 1.05 theta`` along a D^2 rotation in ``plane``; the rest of
    the displacement decays at 0.8/sweep (dense arrays)."""
    import jax.scipy.linalg as jsl

    Tstar = jnp.asarray(Tstar)
    n = Tstar.shape[1]
    K = np.zeros((n, n))
    K[plane[0], plane[1]], K[plane[1], plane[0]] = 1.0, -1.0
    K = jnp.asarray(K)
    u = jnp.einsum("amb,mn->anb", Tstar, K)

    def rot(theta):
        return jnp.einsum("amb,mn->anb", Tstar, jsl.expm(theta * K))

    def advance(T):
        theta = jnp.vdot(u, T - Tstar) / jnp.vdot(u, u)
        return rot(1.05 * theta) + 0.8 * (T - rot(theta))

    return advance, rot


def _without(inv, tag):
    return {k: v for k, v in inv.items() if tag not in k}


def test_a_rotation_inside_a_degenerate_d2_gram_is_seen():
    """Codex P1 on #1058 (_ctm_hold.py:165): with ``G = I`` a D^2 rotation
    moves neither ``G`` nor any spectrum.  The fourth-order slice invariant
    ``Tr(T_m T_m'^+ T_n T_n'^+)`` sees it; the hold must reject the saddle."""
    Tstar = _isotropic_slices()
    advance, rot = _rotation_saddle_in(Tstar)
    start = {(0, 0): _Edge(jnp.asarray(Tstar))}
    moved = {(0, 0): _Edge(rot(0.3))}
    a, b = env_spectral_invariants(start), env_spectral_invariants(moved)
    # Regime: spectra and the D^2 Gram are blind to this rotation.
    assert (
        env_invariant_distance(_without(a, "d2sketch"), _without(b, "d2sketch")) < 1e-12
    )
    assert env_invariant_distance(a, b) > 1e-2

    def step(envs):
        return {(0, 0): _Edge(advance(envs[(0, 0)].T))}

    held = hold_test(step, start, sweeps=40)
    assert not held.passed
    assert all(r > 1.0 for r in held.rates)


def _symmetric_isotropic():
    """U(1) edge tensor; its two charge-0 D^2 slots carry orthonormal
    equal-norm slices, so the charge-0 block of ``G`` is proportional to I
    and a rotation of those two slots is invisible to ``G``."""
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import SymmetricTensor

    sym = U1Symmetry()
    chi = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    d2 = np.array([0, 0, 1, -1], dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, chi, FlowDirection.IN, label="a"),
        TensorIndex.from_charges(sym, d2, FlowDirection.IN, label="m"),
        TensorIndex.from_charges(sym, chi, FlowDirection.OUT, label="b"),
    )
    T = np.array(SymmetricTensor.random_normal(idx, jax.random.PRNGKey(5)).todense())
    s0, s1 = T[:, 0, :].copy(), T[:, 1, :].copy()
    s1 -= np.sum(s0 * s1) / np.sum(s0 * s0) * s0
    T[:, 0, :] = s0 / np.linalg.norm(s0)
    T[:, 1, :] = s1 / np.linalg.norm(s1)
    return idx, T


def test_a_rotation_inside_a_degenerate_block_sparse_d2_gram_is_seen():
    from tenax.core.tensor import SymmetricTensor

    idx, Tstar = _symmetric_isotropic()
    advance, rot = _rotation_saddle_in(Tstar, plane=(0, 1))

    def sym(x):
        return SymmetricTensor.from_dense(jnp.asarray(x), idx)

    start = {(0, 0): _Edge(sym(Tstar))}
    moved = {(0, 0): _Edge(sym(rot(0.3)))}
    a, b = env_spectral_invariants(start), env_spectral_invariants(moved)
    assert (
        env_invariant_distance(_without(a, "d2sketch"), _without(b, "d2sketch")) < 1e-12
    )
    assert env_invariant_distance(a, b) > 1e-3

    def step(envs):
        return {(0, 0): _Edge(sym(advance(envs[(0, 0)].T.todense())))}

    held = hold_test(step, start, sweeps=40)
    assert not held.passed
    assert all(r > 1.0 for r in held.rates)


def test_the_block_sparse_sketch_matches_the_dense_trace():
    """The sector bookkeeping (``A_r B_r^+`` summed over ``q_b``, the trace
    closing ``q_a -> q_a' -> q_a``) must reproduce the dense
    ``Tr(A_r B_r^+ C_r E_r^+)`` built from the same coefficients, embedded in
    their D^2 sectors, for every sector pair."""
    from tenax.algorithms._ctm_hold import _SKETCH_R, _d2_sketch, _sketch_coeffs
    from tenax.core.tensor import SymmetricTensor

    idx, Td = _symmetric_isotropic()
    T = SymmetricTensor.from_dense(jnp.asarray(Td), idx)
    q = np.asarray(idx[1].charges)

    def dense_combo(role, sector):
        c = np.zeros((_SKETCH_R, q.size))
        sl = np.flatnonzero(q == sector)
        c[:, sl] = _sketch_coeffs(role, sector, sl.size)
        return np.einsum("amb,rm->rab", Td, c)

    got = _d2_sketch(T)
    assert set(got) == {(a, b) for a in np.unique(q) for b in np.unique(q)}
    for (qs, ps), s in got.items():
        A, B = dense_combo(0, qs), dense_combo(1, qs)
        C, E = dense_combo(2, ps), dense_combo(3, ps)
        ref = np.einsum("rab,rcb,rcd,rad->r", A, B.conj(), C, E.conj())
        np.testing.assert_allclose(s, ref, atol=1e-12)
        assert np.max(np.abs(ref)) > 1e-3  # regime: not trivially zero


def test_the_block_sparse_invariants_are_blind_to_symmetric_chi_gauges():
    """Per-sector orthogonal maps and signs on each chi leg (all a symmetric
    gauge can do), plus a global scale: no invariant may move."""
    from tenax.core.tensor import SymmetricTensor

    idx, Td = _symmetric_isotropic()
    chi = np.asarray(idx[0].charges)
    rng = np.random.default_rng(9)

    def sector_orthogonal():
        Q = np.zeros((chi.size, chi.size))
        for c in np.unique(chi):
            sl = np.flatnonzero(chi == c)
            orth, _ = np.linalg.qr(rng.standard_normal((sl.size, sl.size)))
            Q[np.ix_(sl, sl)] = orth
        return Q

    Qa, Qb = sector_orthogonal(), sector_orthogonal()
    T2 = -2.5 * np.einsum("xa,amb,yb->xmy", Qa, Td, Qb)
    a = env_spectral_invariants(
        {(0, 0): _Edge(SymmetricTensor.from_dense(jnp.asarray(Td), idx))}
    )
    b = env_spectral_invariants(
        {(0, 0): _Edge(SymmetricTensor.from_dense(jnp.asarray(T2), idx))}
    )
    assert any("d2sketch" in k for k in a)  # regime: the quartic is present
    assert env_invariant_distance(a, b) < 1e-12


@pytest.mark.parametrize("D", [2, 6, 8])
def test_the_fingerprint_beyond_the_gram_does_not_grow_with_d(D):
    """Codex P2 on #1058 (_ctm_hold.py:240): the full fourth-order tensor is
    D^8 per edge (256 MiB at D=8).  Beyond the D^2 x D^2 Gram, an edge's
    fingerprint must be bounded independently of D."""
    rng = np.random.default_rng(D)
    chi = 3
    T = jnp.asarray(rng.standard_normal((chi, D * D, chi)))
    inv = env_spectral_invariants({(0, 0): _Edge(T)})
    beyond = sum(
        v.size for k, v in inv.items() if isinstance(k[2], str) and k[2] != "d2gram"
    )
    assert beyond <= 2 * 64  # one sector combination: R <= 64 complex numbers
    spectra = sum(v.size for k, v in inv.items() if not isinstance(k[2], str))
    assert spectra == chi + min(D * D, chi * chi) + chi  # D^2 leg rank <= chi^2
    gram = sum(v.size for k, v in inv.items() if k[2] == "d2gram")
    assert gram == 2 * D**4


def test_the_d2_gram_is_blind_to_chi_gauges():
    """Separate orthogonal maps on the two chi legs of an edge tensor, and
    the global scale, are gauges: the D^2 Gram must not move."""
    rng = np.random.default_rng(8)
    T = rng.standard_normal((5, 4, 6))
    Qa, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    Qb, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    T2 = -3.0 * np.einsum("xa,amb,yb->xmy", Qa, T, Qb)
    a = env_spectral_invariants({(0, 0): _Edge(jnp.asarray(T))})
    b = env_spectral_invariants({(0, 0): _Edge(jnp.asarray(T2))})
    assert env_invariant_distance(a, b) < 1e-12


def test_the_invariant_distance_is_gauge_blind_and_continuous():
    rng = np.random.default_rng(1)
    C = jnp.asarray(rng.standard_normal((4, 4)))
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    a = env_spectral_invariants({(0, 0): _Vec(C)})
    b = env_spectral_invariants({(0, 0): _Vec(jnp.asarray(Q) @ C[:, ::-1])})
    assert env_invariant_distance(a, b) < 1e-12
    c = env_spectral_invariants({(0, 0): _Vec(C + 1e-7 * jnp.eye(4))})
    assert 0 < env_invariant_distance(a, c) < 1e-5
    nan = env_spectral_invariants({(0, 0): _Vec(C * jnp.nan)})
    assert math.isinf(env_invariant_distance(a, nan))


# --------------------------------------------------------------------------- #
# The eager loop: a mocked sweep with a saddle and an attractor                #
# --------------------------------------------------------------------------- #


def _site(seed=0, D=2, d=2):
    from tests._split_ctm_oracle import make_site

    return make_site(D, d, seed)


CHI = 4


def _saddle_sweep(env0, a, contract=0.5):
    """A fake sweep with a known fixed point ``ref`` and known stability.

    ``ref`` is the loop's initial environment plus a random full-rank corner
    offset (the initial corner is rank 1, which the criterion is blind to).
    Every direction contracts onto ``ref`` at ``contract``, except one scalar
    ``s`` along a corner direction ``u``: ``s <- s + a s (1 - s^2 / s_B^2)``.
    For ``a > 0`` s=0 is a saddle (rate 1 + a) and s=+-s_B are attractors
    (rate 1 - 2a); for ``a < 0`` s=0 is the attractor.  ``u`` is orthogonal
    to the initial displacement, so the loop converges exactly onto s=0.
    """
    rng = np.random.default_rng(3)
    ref, u = {}, {}
    for c, e in env0.items():
        C1 = e.C1.todense()
        R = rng.standard_normal(C1.shape)
        ref[c] = e._replace(C1=type(e.C1)(C1 + jnp.asarray(R), e.C1.indices))
        v = rng.standard_normal(R.shape)
        v -= np.vdot(R, v) / np.vdot(R, R) * R
        u[c] = jnp.asarray(v / np.linalg.norm(v))
    s_B = 0.1 * float(jnp.linalg.norm(ref[(0, 0)].C1.todense()))
    ref_leaves = {c: jax.tree.leaves(e) for c, e in ref.items()}

    def fake(envs, *args, **kwargs):
        out = {}
        for c, e in envs.items():
            leaves, tdef = jax.tree.flatten(e)
            new = []
            for i, (x, r) in enumerate(zip(leaves, ref_leaves[c])):
                dx = x - r
                if i == 0:  # C1
                    s = jnp.vdot(u[c], dx).real
                    perp = dx - s * u[c]
                    s = s + a * s * (1 - s**2 / s_B**2)
                    new.append(r + contract * perp + s * u[c])
                else:
                    new.append(r + contract * dx)
            out[c] = jax.tree.unflatten(tdef, new)
        return out, None, None

    def s_of(envs):
        d = envs[(0, 0)].C1.todense() - ref[(0, 0)].C1.todense()
        return float(jnp.vdot(u[(0, 0)], d).real)

    return fake, s_B, s_of


def _run_mocked(monkeypatch, a, verdicts=None, **kw):
    kw.setdefault("hold_sweeps", 40)  # the hold is opt-in
    A = _site()
    sites = {(0, 0): A, (1, 0): A}
    env0 = {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}
    fake, s_B, s_of = _saddle_sweep(env0, a)
    monkeypatch.setattr(conv, "_ctm_tensor_sweep_multisite", fake)
    if verdicts is not None:
        real = conv.hold_test

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            verdicts.append(out.passed)
            return out

        monkeypatch.setattr(conv, "hold_test", spy)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        envs = _ctm_tensor_multisite(sites, CHECKERBOARD_NEIGHBORS, CHI, **kw)
    not_conv = [
        w
        for w in rec
        if "did not converge" in str(w.message) or "not verified" in str(w.message)
    ]
    blind = [w for w in rec if "could not be certified" in str(w.message)]
    assert not blind  # regime: the criterion can see these corners
    return envs, s_B, s_of, not_conv


def test_regime_without_the_hold_the_loop_certifies_the_saddle(monkeypatch):
    """The defect this file exists for: the loop starts ON the saddle, so
    successive sweeps agree exactly and the successive criterion certifies it."""
    envs, s_B, s_of, not_conv = _run_mocked(
        monkeypatch, 0.1, max_iter=2000, hold_sweeps=0
    )
    assert abs(s_of(envs)) < 1e-12
    assert not not_conv


def test_the_loop_walks_off_a_saddle_to_the_attractor(monkeypatch):
    verdicts = []
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, 0.1, verdicts, max_iter=2000)
    assert abs(abs(s_of(envs)) - s_B) < 1e-5 * s_B
    assert not not_conv
    # One hold rejects the saddle, the next accepts the attractor -- the
    # attractor is not rejected for sitting conv_tol off its own fixed point
    # (a rate fitted to the distance-to-claimed-point did: [False, False, True]).
    assert verdicts == [False, True]


def test_a_saddle_at_max_iter_is_reported_not_converged(monkeypatch):
    # Criterion passes at ~sweep 22; the hold, capped by the budget at
    # (150 - 22) // 2 sweeps, rejects the saddle and exhausts max_iter.
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, 0.1, max_iter=150)
    assert not_conv
    msg = str(not_conv[0].message)
    assert "saddle" in msg
    # Codex P2 on #1058: the criterion DID pass here; the warning must blame
    # the hold, not claim conv_tol was never reached.
    assert "without reaching conv_tol" not in msg
    assert "the hold rejected the point where it was met" in msg


def test_an_attractor_is_returned_as_is(monkeypatch):
    envs, s_B, s_of, not_conv = _run_mocked(monkeypatch, -0.1, max_iter=2000)
    assert abs(s_of(envs)) < 1e-12  # the claimed point, not the perturbed one
    assert not not_conv


def test_no_budget_left_for_the_hold_is_reported_honestly(monkeypatch):
    """The criterion passes at sweep ~22; 22 + 3*40 > max_iter=45, so nothing
    can certify the point.  Ruling (Codex P2 on #1058): stop there and say so
    -- the sweeps that actually ran, and UNVERIFIED -- rather than claim the
    loop "ran the full max_iter", or run a hold cut below its measured
    window."""
    _, _, _, not_conv = _run_mocked(monkeypatch, -0.1, max_iter=45)
    assert len(not_conv) == 1
    msg = str(not_conv[0].message)
    assert "ran the full" not in msg
    import re

    m = re.search(r"stopped after (\d+) of max_iter=45 sweeps", msg)
    assert m and int(m.group(1)) < 45
    assert "UNVERIFIED" in msg
    assert "120 more CTM steps" in msg


def test_hold_sweeps_below_four_is_refused():
    A = _site()
    with pytest.raises(ValueError, match="hold_sweeps"):
        _ctm_tensor_multisite(
            {(0, 0): A, (1, 0): A}, CHECKERBOARD_NEIGHBORS, CHI, hold_sweeps=2
        )


# --------------------------------------------------------------------------- #
# A real, converged dense environment must pass                                #
# --------------------------------------------------------------------------- #


def test_a_converged_dense_d2_environment_holds():
    """Dense D=2 Heisenberg SU state, chi=12: the hold passes early (the
    displacement collapses ~1e-3 in a few sweeps) and hands back exactly the
    point the hold-free loop returns."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.core.tensor import DenseTensor
    from tests.test_ctm_chi_truncation_policy_922 import _su_pair

    A, B = _su_pair(D=2)
    A = DenseTensor(A.todense(), A.indices)
    B = DenseTensor(B.todense(), B.indices)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        held = ctm_tensor_2site(A, B, 12, max_iter=200, conv_tol=1e-10, hold_sweeps=40)
    plain = ctm_tensor_2site(A, B, 12, max_iter=200, conv_tol=1e-10, hold_sweeps=0)
    for x, y in zip(jax.tree.leaves(held), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_hold_is_off_by_default(monkeypatch):
    """Opt-in: without ``hold_sweeps`` the loop never calls the hold and
    certifies the saddle on successive-sweep agreement, as before #1058."""
    calls = []
    monkeypatch.setattr(
        conv, "hold_test", lambda *a, **k: calls.append(1) or pytest.fail("ran")
    )
    A = _site()
    sites = {(0, 0): A, (1, 0): A}
    env0 = {c: initialize_ctm_tensor_env(t, CHI) for c, t in sites.items()}
    fake, _s_B, s_of = _saddle_sweep(env0, 0.1)
    monkeypatch.setattr(conv, "_ctm_tensor_sweep_multisite", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        envs = _ctm_tensor_multisite(sites, CHECKERBOARD_NEIGHBORS, CHI, max_iter=2000)
    assert not calls
    # Regime: the run did converge (on the saddle), so the hold was reachable.
    assert abs(s_of(envs)) < 1e-12


def test_ctm_hold_test_infers_single_site_neighbors(monkeypatch):
    """Codex P2 on #1058: a one-site env with ``neighbors`` omitted used the
    checkerboard map and raised KeyError on (1, 0) at the first sweep."""
    A = _site()
    envs = {(0, 0): initialize_ctm_tensor_env(A, CHI)}
    stepped = []

    def fake_hold(step, e, **kw):
        stepped.append(step(e))  # one real sweep with the inferred map
        return "held"

    monkeypatch.setattr(conv, "hold_test", fake_hold)
    assert conv.ctm_hold_test({(0, 0): A}, envs, CHI) == "held"
    assert set(stepped[0]) == {(0, 0)}
    # Regime: a cell with no default topology must pass neighbors.
    with pytest.raises(ValueError, match="neighbors"):
        conv.ctm_hold_test(
            {(0, 0): A, (0, 1): A}, {(0, 0): envs[(0, 0)], (0, 1): envs[(0, 0)]}, CHI
        )


def test_a_violently_unstable_saddle_fails_closed_not_overflow():
    """Codex P2 on #1058: growth x1000/sweep keeps the renormalised
    trajectories finite, but the unwrapped log distance passes exp's range;
    the result must be a rejection, not an OverflowError."""
    J = np.diag([0.5, 0.5, 1000.0])
    held = hold_test(
        _linear_step(J, np.ones(3)),
        _vec_env(np.ones(3)),
        sweeps=40,
        invariants=_identity_invariants,
    )
    assert not held.passed
    # Regime: the unwrapped growth really did leave float range.
    assert math.isinf(max(held.distances))


def test_a_stable_two_cycle_is_not_certified_as_a_fixed_point():
    """Codex P1 on #1058: perturbed copies contract onto a stable MOVING
    orbit too, so every rate is < 1; the reference's own motion must veto."""
    p, q = np.zeros(4), np.zeros(4)
    p[0], q[1] = 1.0, 1.0

    def step(envs):
        v = np.asarray(envs[(0, 0)].x)
        near, other = (
            (p, q) if np.linalg.norm(v - p) <= np.linalg.norm(v - q) else (q, p)
        )
        return _vec_env(other + 0.5 * (v - near))

    held = hold_test(step, _vec_env(p), sweeps=40, invariants=_identity_invariants)
    assert not held.passed
    assert held.drift >= 0.5  # the reference visited q
    # Regime: the copies did contract -- the rates alone would have passed.
    assert all(r < 1.0 for r in held.rates)


def test_a_drift_rejection_is_not_reported_as_a_saddle():
    """Codex P2 on #1058: a contracting verdict vetoed by the reference's
    drift used to be logged as a generic failure, and the warning then said
    the perturbation "did not contract (rate 0.5000 >= 1)"."""
    note = conv._hold_failure_note([("drift", 200, 1.0, False)])
    assert "not fixed" in note and "drift 1" in note
    assert ">= 1" not in note and "saddle" not in note
    # Regime: a genuine unstable-rate rejection keeps the saddle wording.
    assert ">= 1" in conv._hold_failure_note([("fail", 200, 1.05, False)])


def test_a_saddle_hiding_in_a_zero_leaf_is_excited():
    """Codex P1 on #1058: an exactly-zero leaf got perturbation scale 0, so
    an unstable direction living there was never excited and the stable
    leaf alone passed the hold."""

    class _Two(NamedTuple):
        a: jax.Array
        b: jax.Array

    def step(envs):
        e = envs[(0, 0)]
        return {(0, 0): _Two(0.5 * e.a + 0.5 * jnp.ones(3), 1.1 * e.b)}

    def inv(envs):
        e = envs[(0, 0)]
        return {"a": np.asarray(e.a), "b": np.asarray(e.b)}

    point = {(0, 0): _Two(jnp.ones(3), jnp.zeros(3))}  # b = 0 is a saddle
    held = hold_test(step, point, sweeps=40, invariants=inv)
    assert not held.passed
    assert held.rate > 1.0


def test_a_nonfinite_hold_is_not_resumed_from(monkeypatch):
    """Codex P2 on #1058: a perturbed copy that blew up was installed as the
    loop's next environment and would crash the next projector SVD."""
    from tenax.algorithms._ctm_hold import HoldResult

    def blown(step, envs, **kw):
        bad = {c: jax.tree.map(lambda a: a * jnp.nan, e) for c, e in envs.items()}
        return HoldResult(False, math.inf, (math.inf, math.inf), (1.0,), bad, 3)

    monkeypatch.setattr(conv, "hold_test", blown)
    envs, _s_B, _s_of, not_conv = _run_mocked(monkeypatch, -0.1, max_iter=2000)
    leaves = [np.asarray(x) for e in envs.values() for x in jax.tree.leaves(e)]
    assert all(np.isfinite(x).all() for x in leaves)
    assert not_conv and "non-finite" in str(not_conv[0].message)
