"""The exact fermionic reference (#1037), and what tenax's double layer is.

``_fermionic_fock_oracle.py`` builds a small open-boundary fermionic PEPS in
Fock space, where every sign comes from the operator algebra.  These tests

1. check the oracle against an independent closed form, exhaustively;
2. check that the clusters used can tell fermions from hard-core bosons at all
   (otherwise nothing below could fail);
3. pin #1037's derived local double-layer rule against the oracle; and
4. check that the Tensor CTM's double layer, contracted as the CTM contracts
   it (graded, #1035 step 4), gives the oracle's fermionic energy.  Before
   that step it gave the HARD-CORE-BOSON energy of the sign-free amplitudes
   (#1037).

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
    z_gauge,
)

from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
)
from tenax.core._graded import graded_contract
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
    """Norm and hopping energy from the Tensor CTM's own builders, contracted
    site by site with ``graded_contract`` as the CTM contracts them (#1035
    step 4): ``_build_double_layer_open_tensor`` on the operator's two sites,
    ``_build_double_layer_tensor`` (physical leg traced inside, graded)
    everywhere else.  Closing a site by pairing its open legs with an
    identity tensor instead is a sign-free trace -- a supertrace under
    rule 2 -- and flips the energy."""
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    rename = {s: {} for s in sites}
    for b, (s, x, t, y) in enumerate(bonds_of(R, C)):
        rename[s][f"{x}2"] = rename[t][f"{y}2"] = f"b{b}"
    ao, a = {}, {}
    for s in sites:
        m = {f"{x}2": f"open_{n_of[s]}_{x}" for x in "udlr"} | rename[s]
        a[s] = _build_double_layer_tensor(_tenax_site(As[s])).relabels(m)
        T = _build_double_layer_open_tensor(_tenax_site(As[s]))
        m |= {"phys": f"p{n_of[s]}", "phys_bra": f"P{n_of[s]}"}
        ao[s] = T.relabels(m)
    sym = FermionParity()
    ch = np.array([0, 1], dtype=np.int32)

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
        on = () if bond is None else (n_of[bond[0]], n_of[bond[1]])
        ops = [ao[s] if n_of[s] in on else a[s] for s in sites]
        if bond is not None:
            ops.append(hop(*on))
        out = ops[0]
        for op in ops[1:]:
            out = graded_contract(out, op)
        return real_scalar(out.todense())

    norm = value(None)
    return sum(value((s, t)) for s, _, t, _ in bonds_of(R, C)) / norm


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_tenax_double_layer_matches_the_fermionic_oracle(R, C):
    """#1037 fixed on the Tensor CTM's path: its double layer, contracted as
    the CTM contracts it, gives the fermionic energy of the state graded
    contraction defines, ``Fock(z_gauge(As))`` (design §9) -- not the
    hard-core-boson energy of the plain amplitudes."""
    rng = np.random.default_rng(2)
    for _ in range(3):
        As = random_even_tensors(R, C, rng)
        E = _tenax_energy(R, C, As)
        E_fock = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert E == pytest.approx(E_fock, abs=1e-10)
        assert abs(E - E_hcb) > 1e-3  # regime: the two functionals differ here


def test_tenax_double_layer_matches_the_oracle_for_complex_tensors():
    R, C = 2, 2
    As = _complex_even_tensors(R, C, 5)
    E_fock = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
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
