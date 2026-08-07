"""MPS and MPO must agree on the physical basis, not just on charge values (#816).

Symmetric contraction pairs sectors by *charge value*, so an MPS and MPO that
list the same physical charges in a different **order** still converge to the
right state and report the right energy -- and then hand back an MPS whose dense
physical index is permuted relative to the caller's convention.  Every quantity
read off that state (occupations, correlators, entanglement across an orbital
cut) is silently in the wrong basis, while the energy check that would normally
catch it passes.

The reporter isolated this to the fermionic / Jordan-Wigner path.  It is not
fermionic: a plain XXZ chain hides it only because its Sz=0 ground state is
spin-flip symmetric, so the relabelling maps that state to itself.  Break that
symmetry with one on-site field and a pure spin model fails identically --
``test_the_failure_is_not_fermion_specific`` pins exactly that, so the fix is
never re-scoped to the fermionic path.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenax.algorithms import DMRGConfig, build_auto_mpo
from tenax.algorithms.auto_mpo import fermion_site_ops, spin_half_ops
from tenax.algorithms.dmrg import build_random_symmetric_mps
from tenax.algorithms.dmrg import dmrg as run_dmrg

L = 6


def _mpo(terms, site_ops, fermionic_ops, phys_charges):
    return build_auto_mpo(
        terms,
        L=L,
        d=2,
        site_ops=site_ops,
        fermionic_ops=fermionic_ops,
        symmetric=True,
        phys_charges=np.array(phys_charges),
        compress=True,
        compress_tol=1e-10,
    )


def _cfg():
    return DMRGConfig(
        max_bond_dim=16,
        num_sweeps=4,
        target_charge=0,
        numpy_blockwise=True,
        svd_trunc_err=1e-9,
        convergence_tol=1e-14,
    )


def _rayleigh(mps, mpo):
    """<psi|H|psi>/<psi|psi> read off the returned MPS in the MPO's basis."""
    w = [np.asarray(mpo.get_tensor(i).todense()) for i in range(L)]
    env = np.zeros((1, 1, 1), dtype=complex)
    env[0, 0, 0] = 1.0
    for i, x in enumerate(w):
        a = np.asarray(mps.get_tensor(i).todense())
        env = np.einsum("awb,apc,wpqx,bqd->cxd", env, a.conj(), x, a, optimize=True)
    n = np.ones((1, 1), dtype=complex)
    for i in range(L):
        a = np.asarray(mps.get_tensor(i).todense())
        n = np.einsum("ab,apc,bpd->cd", n, a.conj(), a, optimize=True)
    return float(env.ravel()[0].real) / float(n.ravel()[0].real)


_XXZ = [
    t
    for j in range(L - 1)
    for t in (
        (0.5, "Sp", j, "Sm", j + 1),
        (0.5, "Sm", j, "Sp", j + 1),
        (1.0, "Sz", j, "Sz", j + 1),
    )
]
_TV = [
    t
    for j in range(L - 1)
    for t in (
        (-1.0, "Cd", j, "C", j + 1),
        (-1.0, "Cd", j + 1, "C", j),
        (2.0, "N", j, "N", j + 1),
    )
]


def test_a_mismatched_physical_basis_is_refused():
    """The reporter's exact configuration: MPO [-1, 1] vs the builder's [1, -1].

    Previously this ran to completion and returned a particle-hole-flipped
    state alongside a correct energy.
    """
    mpo = _mpo(_TV, fermion_site_ops(), {"C", "Cd"}, [-1, 1])
    mps0 = build_random_symmetric_mps(L=L, bond_dim=8, seed=1, target_charge=0)
    with pytest.raises(ValueError, match="physical basis"):
        run_dmrg(mpo, mps0, _cfg())


def test_matching_the_basis_makes_the_returned_state_the_reported_one():
    """With the bases aligned, the returned MPS *is* the state whose energy
    was reported -- the property #816 found violated."""
    mpo = _mpo(_TV, fermion_site_ops(), {"C", "Cd"}, [-1, 1])
    mps0 = build_random_symmetric_mps(
        L=L, bond_dim=8, seed=1, target_charge=0, phys_charges=np.array([-1, 1])
    )
    res = run_dmrg(mpo, mps0, _cfg())
    reported = float(res.energies_per_sweep[-1])
    assert abs(reported - _rayleigh(res.mps, mpo)) < 1e-8


def test_the_default_pairing_is_not_refused():
    """Both builders default to [1, -1], so the default path must stay legal.

    This is the regression the guard is most likely to introduce.
    """
    mpo = _mpo(_TV, fermion_site_ops(), {"C", "Cd"}, [1, -1])
    mps0 = build_random_symmetric_mps(L=L, bond_dim=8, seed=1, target_charge=0)
    res = run_dmrg(mpo, mps0, _cfg())
    assert abs(float(res.energies_per_sweep[-1]) - _rayleigh(res.mps, mpo)) < 1e-8


def test_the_failure_is_not_fermion_specific():
    """A pure spin model with the flip symmetry broken fails identically.

    XXZ alone cannot show this: its Sz=0 ground state is invariant under
    Sz -> -Sz, so the permuted state equals the original. One on-site field
    removes that accident. Without this test the fix invites re-scoping to
    the Jordan-Wigner path, where the bug does not live.
    """
    terms = [*_XXZ, (0.7, "Sz", 0)]
    mpo = _mpo(terms, spin_half_ops(), None, [-1, 1])
    mps0 = build_random_symmetric_mps(L=L, bond_dim=8, seed=1, target_charge=0)
    with pytest.raises(ValueError, match="physical basis"):
        run_dmrg(mpo, mps0, _cfg())

    aligned = build_random_symmetric_mps(
        L=L, bond_dim=8, seed=1, target_charge=0, phys_charges=np.array([-1, 1])
    )
    res = run_dmrg(mpo, aligned, _cfg())
    assert abs(float(res.energies_per_sweep[-1]) - _rayleigh(res.mps, mpo)) < 1e-8


def test_builder_rejects_charges_it_cannot_honour():
    """The virtual-sector construction assumes +-1 per site."""
    with pytest.raises(ValueError, match="reordering of"):
        build_random_symmetric_mps(
            L=L, bond_dim=8, seed=1, target_charge=0, phys_charges=np.array([2, -2])
        )
