"""The reference graded contractor against #1038's exact Fock oracle (#1035).

Design §3.4 showed, in plain numpy, that three rules reproduce Fock with
nothing fitted.  These tests make the same claim about tenax's own
``SymmetricTensor`` + ``tenax.core._graded`` -- including the per-site order
a CTM builds, odd tensors, and an SVD regauge (the design's go/no-go).
"""

from __future__ import annotations

import itertools

import _graded_cluster
import jax.numpy as jnp
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    _annihilate,
    _create,
    bonds_of,
    fock_psi,
    fock_psi_ordered,
    hop_energy,
    hop_energy_matvec,
    leg_dims,
    plain_amplitudes,
    random_even_tensors,
    sites_of,
    z_gauge,
)
from _graded_cluster import (
    cluster_energy,
    cluster_value,
    double_layer_energy,
    double_layer_value,
    fock_of_kets,
    gate_operator,
    ket_site,
)

import tenax.algorithms._graded_double_layer as _gdl
from tenax.algorithms._ctm_tensor_projector_2x2 import _scale_bond_by_diag
from tenax.core._graded import graded_contract, graded_reorder, graded_svd

CLUSTERS = [(2, 2), pytest.param(2, 3, marks=pytest.mark.slow)]


def test_the_ordered_fock_state_is_the_oracle_state_for_even_tensors():
    As = random_even_tensors(2, 2, np.random.default_rng(0))
    np.testing.assert_allclose(
        fock_psi_ordered(2, 2, As), fock_psi(2, 2, As), atol=1e-14
    )


@pytest.mark.parametrize("order", ["global", "per_site"])
@pytest.mark.parametrize("R,C", CLUSTERS)
def test_graded_energy_matches_fock(R, C, order):
    rng = np.random.default_rng(7)
    for _ in range(2):
        As = random_even_tensors(R, C, rng)
        E, norm = cluster_energy(R, C, As, order=order)
        E_fock = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert norm > 0
        assert E == pytest.approx(E_fock, abs=1e-12)
        assert abs(E - E_hcb) > 1e-3  # regime: the bosonic answer is different


def test_graded_energy_matches_fock_for_complex_tensors():
    rng = np.random.default_rng(4)
    re, im = random_even_tensors(2, 2, rng), random_even_tensors(2, 2, rng)
    As = {s: re[s] + 1j * im[s] for s in re}
    E, norm = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert norm > 0
    assert E == pytest.approx(E_fock, abs=1e-12)


def test_regime_an_ungraded_bar_misses_the_oracle(monkeypatch):
    """Rule 3 is load-bearing: today's ``bar()`` in place of ``graded_bar``."""
    monkeypatch.setattr(_graded_cluster, "graded_bar", lambda t: t.bar())
    As = random_even_tensors(2, 2, np.random.default_rng(7))
    E, _ = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert abs(E - E_fock) > 1e-3


def _odd(shape, rng):
    B = rng.standard_normal(shape)
    for k in itertools.product(*[range(n) for n in shape]):
        if sum(k) % 2 == 0:
            B[k] = 0.0
    return B


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_one_odd_tensor_per_side_on_an_auxiliary_leg_matches_fock_in_both_orders(R, C):
    """Design §5 step 7's representation: an odd site tensor carries its
    parity on a dimension-1 odd leg, contracted bra-to-ket.  Every tensor is
    then even, so the per-site (CTM) order is as good as the global one.
    Each ket and each bra has exactly one odd site, so this covers only the
    bra-versus-ket crossing (the sign when y is applied after x); with a
    single odd operator per side the Fock state does not depend on
    application order.  Two odd tensors in the same ket need an ordering
    convention for several auxiliary legs, and are left to the next phase."""
    rng = np.random.default_rng(11)
    sites = sites_of(R, C)
    bg = random_even_tensors(R, C, rng)
    Bs = {s: _odd([leg_dims(R, C, *s)[x] for x in "udlr"] + [2], rng) for s in sites}
    bonds = bonds_of(R, C)
    for x in sites:
        for y in sites:
            kx, by = dict(bg), dict(bg)
            kx[x], by[y] = Bs[x], Bs[y]
            px = fock_psi_ordered(R, C, z_gauge(R, C, kx))
            py = fock_psi_ordered(R, C, z_gauge(R, C, by))
            N_f = py @ px
            assert abs(N_f) > 1e-3, (x, y)
            H_f = py @ hop_energy_matvec(R, C, px, fermion=True)
            for order in ("global", "per_site"):
                kw = dict(ket_aux=x, bra_aux=y, order=order)
                N = cluster_value(R, C, by, kx, **kw).real
                H = sum(
                    cluster_value(R, C, by, kx, op_bond=(s, t), **kw).real
                    for s, _, t, _ in bonds
                )
                assert N == pytest.approx(N_f, abs=1e-12), (x, y, order)
                assert H == pytest.approx(H_f, abs=1e-12), (x, y, order)


def _regauge_bond0(As, *, reorder):
    """Merge sites (0,0)-(0,1) over bond b0, SVD back with sqrt(S) on each
    side (exact rank), and restore each site's leg order with ``reorder``."""
    s, t = (0, 0), (0, 1)
    Ks, Kt = ket_site(2, 2, s, As[s]), ket_site(2, 2, t, As[t])
    M = graded_contract(Ks, Kt)
    left = [lab for lab in Ks.labels() if lab != "b0"]
    right = [lab for lab in Kt.labels() if lab != "b0"]
    _, S, _, _ = graded_svd(M, left, right, "b0")
    keep = int(np.sum(np.asarray(S) > 1e-12 * float(np.max(S))))
    U, S, Vh, _ = graded_svd(M, left, right, "b0", max_singular_values=keep)
    r = jnp.sqrt(S)
    return {
        s: reorder(_scale_bond_by_diag(U, r, "b0"), list(Ks.labels())),
        t: reorder(_scale_bond_by_diag(Vh, r, "b0"), list(Kt.labels())),
    }


def _sign_free(t, labels):
    return t.permute_legs(tuple(t.labels().index(lab) for lab in labels))


def test_go_no_go_an_svd_regauge_leaves_the_state_unchanged():
    """Design §5 step 2: linalg under graded semantics.  Splitting a bond and
    re-absorbing sqrt(S) is a gauge move; the energy must not change."""
    As = random_even_tensors(2, 2, np.random.default_rng(3))
    E0, n0 = cluster_energy(2, 2, As)
    E1, n1 = cluster_energy(
        2, 2, As, override=_regauge_bond0(As, reorder=graded_reorder)
    )
    assert E1 == pytest.approx(E0, abs=1e-12)
    assert n1 == pytest.approx(n0, rel=1e-12)
    E_bad, _ = cluster_energy(2, 2, As, override=_regauge_bond0(As, reorder=_sign_free))
    assert abs(E_bad - E0) > 1e-3  # regime: a sign-free reorder around the SVD is wrong


TAU = 0.4


def _fock_gate(psi, a, b, tau):
    """``exp(-tau h_ab) psi`` in Fock space, ``h_ab = -(c_a^+ c_b + h.c.)``."""
    n = psi.size
    h = np.zeros((n, n))
    for col in range(n):
        e = np.zeros(n)
        e[col] = 1.0
        h[:, col] = -(_create(_annihilate(e, b), a) + _create(_annihilate(e, a), b))
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-tau * w)) @ v.T @ psi


def _update_bond0(kets, *, keep, reorder):
    """Apply the imaginary-time gate to bond b0 = (0,0)-(0,1), split it with
    ``graded_svd`` and re-absorb sqrt(S) on each side.

    Returns the kept sites (a real truncation to ``keep`` singular values),
    the discarded remainder (the full-rank split with the kept values
    masked out), and the kept and full bond parities."""
    s, t = (0, 0), (0, 1)
    Ks, Kt = kets[s], kets[t]
    M = graded_contract(gate_operator(0, 1, TAU), graded_contract(Ks, Kt))
    M = M.relabels({"P0": "p0", "P1": "p1"})
    left = [lab for lab in Ks.labels() if lab != "b0"]
    right = [lab for lab in Kt.labels() if lab != "b0"]

    def sites(U, r, Vh):
        return {
            s: reorder(_scale_bond_by_diag(U, r, "b0"), list(Ks.labels())),
            t: reorder(_scale_bond_by_diag(Vh, r, "b0"), list(Kt.labels())),
        }

    U, S, Vh, _ = graded_svd(M, left, right, "b0")
    Uk, Sk, Vhk, _ = graded_svd(M, left, right, "b0", max_singular_values=keep)
    S, Sk = np.asarray(S), np.asarray(Sk)
    np.testing.assert_allclose(np.sort(Sk)[::-1], np.sort(S)[::-1][:keep], rtol=1e-12)
    drop = np.ones_like(S)
    drop[np.argsort(-S)[:keep]] = 0.0
    parity = lambda T: sorted(T.indices[T.labels().index("b0")].charges)  # noqa: E731
    return (
        sites(Uk, jnp.sqrt(Sk), Vhk),
        sites(U, jnp.sqrt(S * drop), Vh),
        parity(Uk),
        parity(U),
    )


@pytest.mark.parametrize("keep", [1, 2, 3])
def test_go_no_go_a_truncated_two_site_update_matches_fock(keep):
    """Design §5 step 2's gate: a truncated two-site update on a 2x2 cluster
    against Fock.  The gate acts through the graded contractor (its physical
    legs are reordered onto the merged pair), ``graded_svd`` splits the bond
    and the split is truncated, discarding whole singular vectors -- at
    ``keep=1`` a whole parity sector of the bond.

    Fock has no bond to truncate on a cluster with a loop, so the oracle
    check is linearity: the kept state plus the discarded remainder must be
    exactly the Fock-space gate applied to the Fock state, and the graded
    contractor must measure the kept state as Fock does."""
    R, C = 2, 2
    As = random_even_tensors(R, C, np.random.default_rng(3))
    kets = {q: ket_site(R, C, q, As[q]) for q in sites_of(R, C)}
    psi0 = fock_psi(R, C, z_gauge(R, C, As))
    np.testing.assert_allclose(fock_of_kets(R, C, kets), psi0, atol=1e-13)

    kept, dropped, kept_par, full_par = _update_bond0(
        kets, keep=keep, reorder=graded_reorder
    )
    psi_full = _fock_gate(psi0, 0, 1, TAU)
    psi_keep = fock_of_kets(R, C, {**kets, **kept})
    psi_drop = fock_of_kets(R, C, {**kets, **dropped})

    # regime: the gate moves the state, and the truncation discards weight
    cos = abs(np.vdot(psi0, psi_full)) / np.linalg.norm(psi0) / np.linalg.norm(psi_full)
    assert cos < 1 - 1e-3
    assert np.linalg.norm(psi_drop) > 1e-3 * np.linalg.norm(psi_full)
    assert full_par == [0, 0, 1, 1]
    if keep == 1:
        assert len(set(kept_par)) == 1  # a whole parity sector of the bond is gone

    np.testing.assert_allclose(psi_keep + psi_drop, psi_full, atol=1e-12)
    E, norm = cluster_energy(R, C, As, override=kept)
    assert norm == pytest.approx(np.vdot(psi_keep, psi_keep).real, rel=1e-12)
    assert E == pytest.approx(hop_energy(R, C, psi_keep, fermion=True), abs=1e-12)

    # regime: a sign-free reorder around the split breaks the update
    k_bad, d_bad, _, _ = _update_bond0(kets, keep=keep, reorder=_sign_free)
    bad = fock_of_kets(R, C, {**kets, **k_bad}) + fock_of_kets(R, C, {**kets, **d_bad})
    assert np.linalg.norm(bad - psi_full) > 1e-3 * np.linalg.norm(psi_full)


# ------------------------------------------------------------------ #
# Phase 2: the graded double layer (design §5 step 3)                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_the_graded_double_layer_energy_matches_fock(R, C):
    """Production-shaped double layers (``build_graded_double_layer``: same
    labels, flows and fused legs as ``_build_double_layer_tensor``),
    contracted site by site, give the energy of the ket-level state that
    ``graded_contract`` defines -- the state gates and SVDs act on."""
    rng = np.random.default_rng(2)
    for _ in range(2):
        As = random_even_tensors(R, C, rng)
        psi = fock_psi(R, C, z_gauge(R, C, As))
        E, norm = double_layer_energy(R, C, As)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert norm == pytest.approx(np.vdot(psi, psi).real, rel=1e-12)
        assert E == pytest.approx(hop_energy(R, C, psi, fermion=True), abs=1e-12)
        assert abs(E - E_hcb) > 1e-3  # regime: the bosonic answer is different


def test_the_graded_double_layer_energy_matches_fock_for_complex_tensors():
    rng = np.random.default_rng(4)
    re, im = random_even_tensors(2, 2, rng), random_even_tensors(2, 2, rng)
    As = {s: re[s] + 1j * im[s] for s in re}
    psi = fock_psi(2, 2, z_gauge(2, 2, As))
    E, norm = double_layer_energy(2, 2, As)
    assert norm == pytest.approx(np.vdot(psi, psi).real, rel=1e-12)
    assert E == pytest.approx(hop_energy(2, 2, psi, fermion=True), abs=1e-12)


def _fock_bond_element(psi, a, b, P_a, P_b, p_a, p_b):
    """``<psi| (c_a^+)^P_a (c_b^+)^P_b |0><0|_ab (c_b)^p_b (c_a)^p_a |psi>``:
    the operator ``|P_a P_b><p_a p_b|`` in the local ``(a, b)`` basis."""
    v = psi
    if p_a:
        v = _annihilate(v, a)
    if p_b:
        v = _annihilate(v, b)
    idx = np.arange(v.size)
    v = np.where(((idx >> a) & 1) | ((idx >> b) & 1), 0, v)  # |0><0| on a, b
    if P_b:
        v = _create(v, b)
    if P_a:
        v = _create(v, a)
    return np.vdot(psi, v)


def test_every_two_site_rdm_element_matches_fock():
    """The two-site reduced density matrix of every bond, element by
    element: all eight parity-even operators ``|P_s P_t><p_s p_t|``,
    including the pairing ones (the random tensors conserve parity, not
    number) and the bond whose sites are not neighbours in Jordan-Wigner
    order, (0,0)-(1,0)."""
    R, C = 2, 2
    As = random_even_tensors(R, C, np.random.default_rng(6))
    psi = fock_psi(R, C, z_gauge(R, C, As))
    n_of = {s: n for n, s in enumerate(sites_of(R, C))}
    checked = 0
    for s, _, t, _ in bonds_of(R, C):
        for k in itertools.product((0, 1), repeat=4):
            if sum(k) % 2:
                continue
            h2 = np.zeros((2, 2, 2, 2))
            h2[k] = 1.0
            got = double_layer_value(R, C, As, op=((s, t), h2))
            want = _fock_bond_element(psi, n_of[s], n_of[t], *k)
            assert got == pytest.approx(want, abs=1e-12), (s, t, k)
            checked += abs(want) > 1e-3
    assert checked >= 16  # regime: most elements are not trivially zero


def test_regime_the_fuse_needs_its_pair_sign(monkeypatch):
    """Without the pair sign, the fused double layer is a different state."""
    monkeypatch.setattr(_gdl, "_pair_sign", lambda t, k: t)
    As = random_even_tensors(2, 2, np.random.default_rng(2))
    psi = fock_psi(2, 2, z_gauge(2, 2, As))
    E, _ = double_layer_energy(2, 2, As)
    assert abs(E - hop_energy(2, 2, psi, fermion=True)) > 1e-3
