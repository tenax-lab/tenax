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
from _graded_cluster import cluster_energy, cluster_value, ket_site

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
