"""The exact fermionic reference (#1037), and what tenax's double layer is.

``_fermionic_fock_oracle.py`` builds a small open-boundary fermionic PEPS in
Fock space, where every sign comes from the operator algebra.  These tests

1. check the oracle against an independent closed form, exhaustively;
2. check that the clusters used can tell fermions from hard-core bosons at all
   (otherwise nothing below could fail);
3. pin #1037's derived local double-layer rule against the oracle; and
4. record the defect: tenax's own double layer + ``contract`` computes the
   HARD-CORE-BOSON energy of the sign-free amplitudes, and does not match
   the oracle.

Everything is exact (no CTM, no optimisation): D=2 bonds, t=1, V=0.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    bonds_of,
    double_layer_energy,
    fock_psi,
    ground_energy,
    hop_energy,
    hop_energy_matvec,
    leg_dims,
    plain_amplitudes,
    plain_double_layer,
    random_even_tensors,
    real_scalar,
    sign_formula,
    site_bits,
    sites_of,
)

from tenax.algorithms._ctm_tensor_init import _build_double_layer_open_tensor
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

jax.config.update("jax_enable_x64", True)

# 2x2 is the required-gate case (~1 s each); 2x3 adds a second plaquette and a
# 2^20-dim Fock space (~20 s per test), so it runs in the slow bucket.
CLUSTERS = [(2, 2), pytest.param(2, 3, marks=pytest.mark.slow)]


# ------------------------------------------------------------------ #
# 1. The oracle agrees with its closed form                          #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_the_operator_algebra_matches_the_closed_form_on_every_configuration(R, C):
    """Two independent derivations of every sign, compared exhaustively."""
    sites, bonds = sites_of(R, C), bonds_of(R, C)
    checked = 0
    for occ in itertools.product((0, 1), repeat=len(bonds)):
        As, phys = {}, {}
        for s in sites:
            dm = leg_dims(R, C, *s)
            bits, p = site_bits(R, C, occ, s)
            A = np.zeros([dm["u"], dm["d"], dm["l"], dm["r"], 2])
            A[bits["u"], bits["d"], bits["l"], bits["r"], p] = 1.0
            As[s], phys[s] = A, p
        psi = fock_psi(R, C, As)
        idx = sum(phys[s] << n for n, s in enumerate(sites))
        assert np.count_nonzero(psi) == 1 and abs(abs(psi[idx]) - 1.0) < 1e-12
        expected = -1.0 if sign_formula(R, C, occ, phys) else 1.0
        assert psi[idx] == expected, f"occ={occ}"
        checked += 1
    assert checked == 2 ** len(bonds)


# ------------------------------------------------------------------ #
# 2. Regime: these clusters separate fermions from hard-core bosons  #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "R,C,E_F,E_B",
    [
        (2, 2, -2.0, -2.0 * np.sqrt(2.0)),
        pytest.param(2, 3, -3.414213562373095, -3.924681, marks=pytest.mark.slow),
    ],
)
def test_the_clusters_separate_fermions_from_hard_core_bosons(R, C, E_F, E_B):
    """A 2-site gate is the same 4x4 matrix for both statistics; only loops
    tell them apart.  Both clusters contain one, and the gap is large."""
    assert ground_energy(R, C, fermion=True) == pytest.approx(E_F, abs=1e-9)
    assert ground_energy(R, C, fermion=False) == pytest.approx(E_B, abs=1e-6)
    assert E_F - E_B > 0.5


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_an_oracle_state_never_goes_below_the_fermionic_ground_state(R, C):
    E_F = ground_energy(R, C, fermion=True)
    rng = np.random.default_rng(0)
    for _ in range(4):
        psi = fock_psi(R, C, random_even_tensors(R, C, rng))
        assert hop_energy(R, C, psi, fermion=True) >= E_F - 1e-12


def test_the_oracle_energy_is_a_bra_ket_for_complex_states():
    """``hop_energy`` must conjugate the bra: a bilinear ``psi @ H psi`` is not
    an expectation value once the tensors are complex."""
    R, C = 2, 2
    rng = np.random.default_rng(3)
    re, im = random_even_tensors(R, C, rng), random_even_tensors(R, C, rng)
    psi = fock_psi(R, C, {s: re[s] + 1j * im[s] for s in re})
    assert np.abs(psi.imag).max() > 1e-3  # regime: genuinely complex
    N = psi.size
    H = np.stack([hop_energy_matvec(R, C, e, fermion=True) for e in np.eye(N)], axis=1)
    expected = (np.conj(psi) @ H @ psi / (np.conj(psi) @ psi)).real
    assert hop_energy(R, C, psi, fermion=True) == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------------------------ #
# 3. #1037's derived local rule reproduces the oracle                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_the_derived_local_rule_reproduces_the_oracle(R, C):
    rng = np.random.default_rng(1)
    for _ in range(3):
        As = random_even_tensors(R, C, rng)
        E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
        Es = {s: plain_double_layer(A) for s, A in As.items()}
        E_rule, norm = double_layer_energy(R, C, Es, rule=True)
        E_plain, _ = double_layer_energy(R, C, Es, rule=False)
        assert norm > 0
        assert E_rule == pytest.approx(E_fock, abs=1e-12)
        # regime: without the rule the same tensors give a different answer
        assert abs(E_plain - E_fock) > 1e-3


def _complex_even_tensors(R, C, seed):
    rng = np.random.default_rng(seed)
    re, im = random_even_tensors(R, C, rng), random_even_tensors(R, C, rng)
    return {s: re[s] + 1j * im[s] for s in re}


def test_the_derived_local_rule_reproduces_the_oracle_for_complex_tensors():
    R, C = 2, 2
    As = _complex_even_tensors(R, C, 4)
    E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
    Es = {s: plain_double_layer(A) for s, A in As.items()}
    E_rule, norm = double_layer_energy(R, C, Es, rule=True)
    assert norm > 0
    assert E_rule == pytest.approx(E_fock, abs=1e-12)


# ------------------------------------------------------------------ #
# 4. tenax's double layer (the #1037 defect)                          #
# ------------------------------------------------------------------ #


def _tenax_site(A: np.ndarray) -> SymmetricTensor:
    sym = FermionParity()
    flows = (FlowDirection.OUT, FlowDirection.IN, FlowDirection.OUT, FlowDirection.IN)
    idx = tuple(
        TensorIndex.from_charges(sym, np.arange(n, dtype=np.int32) % 2, f, label=lbl)
        for n, f, lbl in zip(A.shape[:4], flows, "udlr")
    ) + (
        TensorIndex.from_charges(
            sym, np.array([0, 1], dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    return SymmetricTensor.from_dense(jnp.asarray(A), idx)


def _tenax_energy(R: int, C: int, As: dict) -> float:
    """Norm and hopping energy from ``_build_double_layer_open_tensor`` +
    ``contract`` alone -- the double layer every fermionic CTM/RDM path uses."""
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    rename = {s: {} for s in sites}
    for b, (s, x, t, y) in enumerate(bonds_of(R, C)):
        rename[s][f"{x}2"] = rename[t][f"{y}2"] = f"b{b}"
    ao = {}
    for s in sites:
        T = _build_double_layer_open_tensor(_tenax_site(As[s]))
        m = {f"{x}2": f"open_{n_of[s]}_{x}" for x in "udlr"} | rename[s]
        m |= {"phys": f"p{n_of[s]}", "phys_bra": f"P{n_of[s]}"}
        ao[s] = T.relabels(m)
    sym = FermionParity()
    ch = np.array([0, 1], dtype=np.int32)

    def ident(n):
        return SymmetricTensor.from_dense(
            jnp.eye(2),
            (
                TensorIndex.from_charges(sym, ch, FlowDirection.OUT, label=f"p{n}"),
                TensorIndex.from_charges(sym, ch, FlowDirection.IN, label=f"P{n}"),
            ),
        )

    h = np.zeros((2, 2, 2, 2))
    h[1, 0, 0, 1] = h[0, 1, 1, 0] = -1.0

    def hop(ns, nt):  # h[P_s, P_t, p_s, p_t]
        return SymmetricTensor.from_dense(
            jnp.asarray(h),
            (
                TensorIndex.from_charges(sym, ch, FlowDirection.IN, label=f"P{ns}"),
                TensorIndex.from_charges(sym, ch, FlowDirection.IN, label=f"P{nt}"),
                TensorIndex.from_charges(sym, ch, FlowDirection.OUT, label=f"p{ns}"),
                TensorIndex.from_charges(sym, ch, FlowDirection.OUT, label=f"p{nt}"),
            ),
        )

    def value(bond):
        ops = [ao[s] for s in sites]
        on = () if bond is None else (n_of[bond[0]], n_of[bond[1]])
        ops += [ident(n_of[s]) for s in sites if n_of[s] not in on]
        if bond is not None:
            ops.append(hop(*on))
        return real_scalar(contract(*ops).todense())

    norm = value(None)
    return sum(value((s, t)) for s, _, t, _ in bonds_of(R, C)) / norm


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_1037_tenax_double_layer_is_the_hard_core_boson_functional(R, C):
    """Pins the DEFECT: remove together with the fix for #1037.

    tenax's double layer, contracted with tenax's ``contract``, returns exactly
    the hard-core-boson energy of the sign-free ket amplitudes -- and not the
    fermionic energy of the same tensors.
    """
    rng = np.random.default_rng(2)
    for _ in range(3):
        As = random_even_tensors(R, C, rng)
        E_tenax = _tenax_energy(R, C, As)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
        assert E_tenax == pytest.approx(E_hcb, abs=1e-10)
        assert abs(E_tenax - E_fock) > 1e-3


def test_1037_characterization_holds_for_complex_tensors():
    """Same defect pin as above, with complex site tensors (2x2)."""
    R, C = 2, 2
    As = _complex_even_tensors(R, C, 5)
    E_tenax = _tenax_energy(R, C, As)
    E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
    E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
    assert E_tenax == pytest.approx(E_hcb, abs=1e-10)
    assert abs(E_tenax - E_fock) > 1e-3


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1037: the fermionic double layer applies no fermionic sign",
)
@pytest.mark.parametrize("R,C", CLUSTERS)
def test_tenax_double_layer_matches_the_fermionic_oracle(R, C):
    rng = np.random.default_rng(2)
    As = random_even_tensors(R, C, rng)
    E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
    assert _tenax_energy(R, C, As) == pytest.approx(E_fock, abs=1e-10)


# ------------------------------------------------------------------ #
# 5. Where the defect starts: the first closed loop                  #
# ------------------------------------------------------------------ #
#
# Every cluster above has a plaquette, so all of them separate fermions
# from hard-core bosons.  A 1xC cluster is a TREE, and there the two
# functionals are not merely close -- they are identically equal, because
# the fermionic sign of a hopping term can only differ from the bosonic
# one around a cycle.  tenax's double layer is therefore already EXACT on
# a tree, and #1037 switches on with the first loop.
#
# This is the boundary of the defect, so it is the regression guard for
# the fix: whatever #1035/#1036 change, tree clusters must not move.

TREE_CLUSTERS = [(1, 3), (1, 4)]


def _is_acyclic(R: int, C: int) -> bool:
    """A connected graph is acyclic iff |edges| = |vertices| - 1."""
    return len(bonds_of(R, C)) == len(sites_of(R, C)) - 1


@pytest.mark.parametrize("R,C", TREE_CLUSTERS)
def test_a_tree_cluster_cannot_separate_fermions_from_hard_core_bosons(R, C):
    """The mirror of ``test_the_clusters_separate_...``: on a tree it can't.

    Not an approximation -- the two functionals agree to the last bit, so
    there is no fermionic sign to get wrong in the first place.
    """
    assert _is_acyclic(R, C), f"{R}x{C} is not a tree"
    rng = np.random.default_rng(2)
    As = random_even_tensors(R, C, rng)
    E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
    E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
    # Guard against a vacuous pass: 1x2 has E == 0 for both, which would
    # satisfy any equality assertion without exercising anything.
    assert abs(E_fock) > 1e-3, f"{R}x{C} energy is trivial ({E_fock})"
    assert E_fock == pytest.approx(E_hcb, abs=1e-12)


@pytest.mark.parametrize("R,C", TREE_CLUSTERS)
def test_tenax_double_layer_is_already_exact_on_a_tree(R, C):
    """#1037 is a LOOP defect: on an acyclic cluster tenax is exact.

    Keep this passing through the #1037 fix.  A fix that corrects the
    plaquette clusters by changing tree results has introduced a second
    defect where there was none.
    """
    assert _is_acyclic(R, C), f"{R}x{C} is not a tree"
    rng = np.random.default_rng(2)
    As = random_even_tensors(R, C, rng)
    E_fock = hop_energy(R, C, fock_psi(R, C, As), fermion=True)
    assert abs(E_fock) > 1e-3, f"{R}x{C} energy is trivial ({E_fock})"
    assert _tenax_energy(R, C, As) == pytest.approx(E_fock, abs=1e-10)
