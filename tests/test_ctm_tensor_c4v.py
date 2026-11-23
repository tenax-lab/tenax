"""Tests for C4v-symmetric CTM with Tensor protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from _c4v_staggered_fixtures import (
    c4v_fixture,
    charged_block_ratio,
    corner_edge,
    cut_gap,
    norm_sv,
    sv_gap,
)

from tenax.algorithms._ctm_c4v_charges import _negate_charges
from tenax.algorithms._ctm_tensor_c4v import _c4v_sweep, ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_c4v_staggered import (
    _c4v_staggered_enlarged_corner,
    _c4v_staggered_sweep,
    _check_c4v_staggered_invariants,
)
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor, ctm_tensor_2site
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
)
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv, _build_double_layer_tensor
from tenax.algorithms.ipeps import sublattice_rotate_gate
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_dense():
    """Near-product-state DenseTensor iPEPS site tensor, D=2, d=2, trivial U1.

    A near-product-state tensor has a unique CTM fixed point that is
    naturally C4v-symmetric, so both general and C4v CTMs converge to
    the same environment.
    """
    D, d = 2, 2
    rng = np.random.RandomState(42)
    data = 0.01 * jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
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


@pytest.fixture
def entangled_peps_c4v():
    """*Entangled* C4v-symmetric iPEPS site tensor, D=2, d=2, trivial U1.

    ``small_peps_dense`` perturbs a product state by 1%, which leaves the
    corner spectrum so close to rank-1 that a wrong enlarged corner barely
    moves the energy — that fixture passed throughout #760.  This one is a
    fully random tensor projected onto the C4v-invariant subspace, so the
    environment carries real weight beyond the leading eigenvalue and the
    two CTM schemes have something to disagree about.
    """
    from tenax.algorithms.ipeps import symmetrize_c4v

    D, d = 2, 2
    rng = np.random.RandomState(7)
    data = jnp.array(rng.standard_normal((D, D, D, D, d)))
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


@pytest.fixture
def heisenberg_gate():
    """Heisenberg 2-site Hamiltonian gate as dense array."""
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestC4vCTM:
    def test_returns_ctm_tensor_env(self, small_peps_dense):
        """ctm_tensor_c4v returns a CTMTensorEnv."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_dense, chi=4, max_iter=5)
        assert isinstance(env, CTMTensorEnv)

    def test_all_tensors_finite(self, small_peps_dense):
        """All environment tensors are finite after 20 sweeps."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_dense, chi=8, max_iter=20)
        for field in env:
            dense = field.todense()
            assert jnp.all(jnp.isfinite(dense)), (
                f"Non-finite values in tensor with labels {field.labels()}"
            )

    def test_energy_matches_general_ctm(self, small_peps_dense, heisenberg_gate):
        """C4v energy matches general CTM energy (atol=1e-4)."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8

        # General CTM
        env_gen, _ = ctm_tensor(small_peps_dense, chi=chi, max_iter=60, conv_tol=1e-10)
        E_gen = float(
            compute_energy_ctm_tensor(small_peps_dense, env_gen, heisenberg_gate, d=2)
        )

        # C4v CTM
        env_c4v = ctm_tensor_c4v(small_peps_dense, chi=chi, max_iter=60, conv_tol=1e-10)
        E_c4v = float(
            compute_energy_ctm_tensor(small_peps_dense, env_c4v, heisenberg_gate, d=2)
        )

        np.testing.assert_allclose(E_c4v, E_gen, atol=1e-4)

    def test_energy_matches_general_ctm_on_an_entangled_state(
        self, entangled_peps_c4v, heisenberg_gate
    ):
        """#760: two converged CTMs must contract the same state identically.

        The C4v sweep grew its corner as ``C·T`` — one edge, and the
        double-layer tensor ``a`` absorbed only into the *edge*, never into
        the corner.  Standard CTMRG enlarges the corner as ``C·T_h·T_v·a``
        (:func:`_build_enlarged_corner`).  The omission converges cleanly to
        the *wrong* fixed point, so nothing upstream reports a failure; the
        energy is simply wrong, by 4.6e-3 at D=2 and 1.8e-2 at D=4 on
        optimised Heisenberg states, and the gap does **not** close as chi
        grows.

        Both schemes are run to 1e-12 here, so a converged-vs-converged
        comparison is exact up to CTM tolerance — no ``atol=1e-4`` slack that
        a near-product fixture can hide inside.
        """
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8
        env_gen, _ = ctm_tensor(
            entangled_peps_c4v,
            chi=chi,
            max_iter=300,
            conv_tol=1e-12,
            projector_method="eigh",
        )
        E_gen = float(
            compute_energy_ctm_tensor(entangled_peps_c4v, env_gen, heisenberg_gate, d=2)
        )

        env_c4v = ctm_tensor_c4v(
            entangled_peps_c4v, chi=chi, max_iter=300, conv_tol=1e-12
        )
        E_c4v = float(
            compute_energy_ctm_tensor(entangled_peps_c4v, env_c4v, heisenberg_gate, d=2)
        )

        assert abs(E_c4v - E_gen) / max(abs(E_gen), 1e-30) < 1e-8, (
            f"C4v CTM energy {E_c4v!r} != general CTM energy {E_gen!r} "
            f"(rel {abs(E_c4v - E_gen) / abs(E_gen):.3e}); the two schemes "
            "converged to different environments (#760)."
        )


# ------------------------------------------------------------------ #
# U(1) SymmetricTensor tests                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_peps_u1():
    """C4v-symmetric charged U(1) SymmetricTensor iPEPS with D=2, d=2.

    The test fixture of the charged-C4v design (#762): its virtual charges
    ``[0, 1]`` are not self-dual, so it describes the 2-sublattice state
    ``(A, negate(A))``, and ``ctm_tensor_c4v`` runs the 2-sublattice sweep on
    it.  It is a monomer-dimer state that converges slowly in chi.

    Three constraints make this fixture fiddly; earlier versions violated
    them, and each violation was silent.

    **1. It must be non-zero.**  It used ``vc = [-1, 1]`` with ``pc = [-1, 1]``,
    which admits no charge-conserving block at all: the four virtual legs
    contribute an even total (four odd charges, signed by flow) and the single
    odd physical charge cannot cancel it.  ``random_normal`` therefore returned
    0 blocks and ``‖A‖ = 0``, so the energy test of that time compared
    ``0`` against ``0`` at ``atol=1e-4`` and passed no matter what the code did.
    ``vc`` must span BOTH parities; ``[0, 1]`` does, and keeps the physical
    spin-1/2 leg.

    **2. It must actually be C4v.**  These tests exercise the C4v CTM, which is
    only defined on the C4v-invariant manifold — the same trap as the 75%
    non-C4v ``_site_tensor`` that #760 had to fix.  But ``symmetrize_c4v`` is
    not applicable to an arbitrary charged tensor: the C4 rotation reinterprets
    IN legs as OUT, which for U(1) needs ``q -> -q``, so with ``vc`` on all four
    virtual legs only trivial charges survive the projection.  Assigning
    ``u, l = OUT/vc`` and ``d, r = IN/(-vc)`` makes every C4v element replace a
    leg only by one of matching flow-signed charge, so the projection preserves
    the block structure exactly.

    **3. It must not be physically trivial.**  ``pc = [-1, 1]`` alongside
    ``vc = [0, 1]`` projects onto a *polarised product state*: only one physical
    basis state carries weight, the physical index factorises out of the tensor,
    and the Heisenberg energy is exactly ``2 x 0.25 = 0.5`` at every chi.  Such a
    fixture is non-zero and genuinely C4v yet still cannot discriminate — the
    symmetric and dense paths agree on it because there is nothing to get wrong.
    ``pc = [0, 1]`` gives 5 blocks and a rank-2 physical-leg Gram matrix.

    All three properties are asserted below rather than assumed.  Each one was
    violated in turn while this fixture was being repaired, and every violation
    was silent, so the assertions are the point of the fixture as much as the
    tensor is.
    """
    from tenax.algorithms.ipeps import symmetrize_c4v

    key = jax.random.PRNGKey(42)
    sym = U1Symmetry()
    vc = np.array([0, 1], dtype=np.int32)
    pc = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, (-vc).copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, (-vc).copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    raw = SymmetricTensor.random_normal(indices, key)
    projected = symmetrize_c4v(jnp.asarray(raw.todense()))
    A = SymmetricTensor.from_dense(projected, indices)

    assert len(A.blocks) > 1 and float(A.norm()) > 0.0, (
        f"vacuous U(1) fixture: {len(A.blocks)} blocks, norm {float(A.norm())}"
    )
    kept = float(A.norm()) / float(jnp.linalg.norm(projected))
    assert abs(kept - 1.0) < 1e-12, (
        f"symmetrize_c4v broke charge conservation: kept {kept}; the index "
        "charges do not admit the C4v action"
    )
    dense = np.asarray(A.todense())
    # C4v invariance: the projector is idempotent, so a symmetric tensor is a
    # fixed point of it.  This covers all 8 group elements, not just one.
    residual = float(
        jnp.linalg.norm(symmetrize_c4v(jnp.asarray(dense)) - dense)
    ) / np.linalg.norm(dense)
    assert residual < 1e-12, f"fixture is not C4v-symmetric: residual {residual}"

    # Non-triviality: Gram matrix of the physical leg after tracing the four
    # virtual legs.  Rank 1 means A[u,d,l,r,s] = B[u,d,l,r] * v[s], i.e. the
    # physical index factorises and the state is a product state.  (This is a
    # property of the tensor, not an entropy of the state — there is no
    # environment in it.)
    gram = np.einsum("udlrs,udlrt->st", dense, dense.conj())
    gram_rank = int(np.linalg.matrix_rank(gram, tol=1e-10 * np.trace(gram).real))
    assert gram_rank > 1, (
        f"fixture is a product state: physical-leg Gram rank {gram_rank}; "
        "symmetric and dense agree on it trivially"
    )
    return A


@pytest.fixture
def small_peps_fermionic():
    """Random FermionParity SymmetricTensor iPEPS with D=2, d=2."""
    key = jax.random.PRNGKey(7)
    sym = FermionParity()
    vc = np.array([0, 1], dtype=np.int32)
    pc = np.array([0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    return SymmetricTensor.random_normal(indices, key)


class TestC4vCTMSymmetric:
    def test_u1_converges(self, small_peps_u1):
        """C4v CTM converges with U(1) SymmetricTensor."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_u1, chi=6, max_iter=30, conv_tol=1e-8)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))


class TestC4vCTMFermionic:
    def test_fermionic_converges(self, small_peps_fermionic):
        """C4v CTM converges with FermionParity SymmetricTensor."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_fermionic, chi=4, max_iter=30, conv_tol=1e-8)
        assert isinstance(env, CTMTensorEnv)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))

    def test_fermionic_energy_finite(self, small_peps_fermionic, heisenberg_gate):
        """FermionParity C4v CTM produces a finite energy.

        C4v CTM internally densifies fermionic tensors (Koszul signs from
        the C4v flow-flip expansion cause cancellation for SymmetricTensor).
        Energy must be computed with the matching DenseTensor A.
        """
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        chi = 8
        A_dense = DenseTensor(
            small_peps_fermionic.todense(), small_peps_fermionic.indices
        )
        env = ctm_tensor_c4v(A_dense, chi=chi, max_iter=80, conv_tol=1e-10)
        E = float(compute_energy_ctm_tensor(A_dense, env, heisenberg_gate, d=2))
        assert jnp.isfinite(E), f"Energy not finite: {E}"

    def test_fermionic_many_sweeps_stable(self, small_peps_fermionic):
        """FermionParity C4v CTM runs 50 sweeps without crashing."""
        from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v

        env = ctm_tensor_c4v(small_peps_fermionic, chi=4, max_iter=50, conv_tol=1e-14)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))


# ------------------------------------------------------------------ #
# Charged C4v input (#762, #763)                                       #
# ------------------------------------------------------------------ #


class TestChargedC4v:
    """Charged C4v input: the 2-sublattice sweep and its energy (#762, #763)."""

    def test_guard_refuses_a_general_ctm_env(self, heisenberg_gate):
        """An env from the general ctm_tensor was not built for charged C4v A (#1210)."""
        A = c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0])
        env, _ = ctm_tensor(A, chi=4, max_iter=1)
        with pytest.raises(ValueError, match="not built for this A"):
            compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2)

    @pytest.mark.parametrize("chi", [8, 16])
    def test_762_energy_matches_dense_c4v(self, chi, heisenberg_gate):
        """#762: the energy and corner spectrum of dense C4v CTM on todense(A).

        The same algorithm on the same state.  Where the cut splits no mirror
        pair, the two sweeps keep the same space.  The regime is asserted
        before the comparison.
        """
        A = c4v_fixture(U1Symmetry(), [-1, 1], [-2, 0])
        Ad = DenseTensor(A.todense(), A.indices)
        env = ctm_tensor_c4v(A, chi=chi, max_iter=200, conv_tol=1e-10)
        env_d = ctm_tensor_c4v(Ad, chi=chi, max_iter=200, conv_tol=1e-10)

        a, ad = _build_double_layer_tensor(A), _build_double_layer_tensor(Ad)
        (C, T), (Cd, Td) = corner_edge(env), corner_edge(env_d)
        sv, sv_d = norm_sv(C), norm_sv(Cd)
        drift = sv_gap(norm_sv(_c4v_staggered_sweep(C, T, a, chi)[0]), sv)
        drift_d = sv_gap(norm_sv(_c4v_sweep(Cd, Td, ad, chi, "eigh")[0]), sv_d)
        assert drift < 1e-9 and drift_d < 1e-9, (drift, drift_d)  # converged
        Q = _c4v_staggered_enlarged_corner(C, T, a)
        Qd = _c4v_staggered_enlarged_corner(Cd, Td, ad)
        gaps = (cut_gap(Q, chi), cut_gap(Qd, chi))
        assert min(gaps) >= 0.05, gaps  # the cut splits no pair
        assert charged_block_ratio(Q) >= 1e-3  # charged blocks carry weight
        assert sv[1] / sv[0] > 0.1, sv[:2]  # not a product state

        for gate in (heisenberg_gate, sublattice_rotate_gate(heisenberg_gate)):
            E = float(compute_energy_ctm_tensor(A, env, gate, d=2))
            E_d = float(compute_energy_ctm_tensor(Ad, env_d, gate, d=2))
            assert abs(E - E_d) <= 1e-8, (E, E_d)
        assert sv_gap(sv, sv_d) <= 1e-8

        if chi == 16:  # at chi=8 the C4v and 2x2 truncations differ by 1.7e-5
            env_g, _ = ctm_tensor(Ad, chi=16, max_iter=200, conv_tol=1e-10)
            E_g = float(compute_energy_ctm_tensor(Ad, env_g, heisenberg_gate, d=2))
            E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
            assert abs(E - E_g) <= 1e-8, (E, E_g)

    @pytest.mark.parametrize(
        ("vc", "pc", "chi"),
        [
            pytest.param([-1, 1], [-2, 0], 16, id="762-chi16"),
            pytest.param([-1, 1], [-2, 0], 32, id="762-chi32"),
            pytest.param([0, 1], [-1, 1], 16, id="polarised-chi16"),
        ],
    )
    def test_763_charged_input_runs_at_large_chi(self, vc, pc, chi, heisenberg_gate):
        """#763: these raised a size-mismatch ValueError in the energy call."""
        A = c4v_fixture(U1Symmetry(), vc, pc)
        env = ctm_tensor_c4v(A, chi=chi, max_iter=50, conv_tol=1e-10)
        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
        assert np.isfinite(E)
        if pc == [-1, 1]:  # a polarised product state: exactly 2 x 0.25
            assert abs(E - 0.5) < 1e-10, E

    def test_monomer_dimer_fixture_runs_at_chi16(self, small_peps_u1):
        """The test fixture at chi=16: a finite env, I1-I3, and whole pairs."""
        A = small_peps_u1
        env = ctm_tensor_c4v(A, chi=16, max_iter=200, conv_tol=1e-10)
        for field in env:
            assert jnp.all(jnp.isfinite(field.todense()))
        C, T = corner_edge(env)
        a = _build_double_layer_tensor(A)
        a_B = _build_double_layer_tensor(_negate_charges(A))
        _check_c4v_staggered_invariants(C, T, a, a_B)  # I2 counts the pairs

    @pytest.mark.slow
    def test_monomer_dimer_energy_matches_the_2site_ctm(
        self, small_peps_u1, heisenberg_gate
    ):
        """Against the block-sparse 2-site CTM on (A, negate(A)); about 8 min.

        chi=12, because there the sweep has one fixed point.  At chi=16 it
        has two, E = 0.37474583 and 0.37478857: in the early sweeps a value
        of the q=0 block equals the (-2, 2) pair value at the cut (relative
        gap 2e-16), and roundoff decides which one stays.  macOS CI reached
        the second, Linux the first, and 1e-12 noise on A reaches either.
        Both fixed points converge, with gaps of 22% and 4% at the final cut,
        so the final state does not flag this; chi was picked on the
        trajectory.  At chi=12 no decision on the way comes closer than a
        54% gap, the fixture and four noise seeds agree to 9e-13, and the
        2-site CTM converges.  The two energies, and the dense ctm_tensor on
        todense(A), then agree to 1e-12.
        """
        chi = 12
        A = small_peps_u1
        env = ctm_tensor_c4v(A, chi=chi, max_iter=200, conv_tol=1e-10)
        a = _build_double_layer_tensor(A)
        C, T = corner_edge(env)
        sv = norm_sv(C)
        drift = sv_gap(norm_sv(_c4v_staggered_sweep(C, T, a, chi)[0]), sv)
        assert drift < 1e-9, drift  # converged
        Q = _c4v_staggered_enlarged_corner(C, T, a)
        assert cut_gap(Q, chi) >= 0.05, cut_gap(Q, chi)  # the cut splits no pair
        assert charged_block_ratio(Q) >= 1e-3  # charged blocks carry weight
        assert sv[1] / sv[0] > 0.1, sv[:2]  # not a product state

        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
        B = _negate_charges(A)
        env_A, env_B = ctm_tensor_2site(
            A, B, chi=chi, max_iter=200, conv_tol=1e-10, strict=True
        )
        E_2 = float(
            compute_energy_ctm_tensor_2site(A, B, env_A, env_B, heisenberg_gate, d=2)
        )
        assert abs(E - E_2) <= 1e-8, (E, E_2)

    def test_z2_self_dual_path_is_unchanged(self, heisenberg_gate):
        """Z2 control: self-dual input keeps the old sweep, block-sparse = dense."""
        A = c4v_fixture(ZnSymmetry(2), [0, 1], [0, 1])
        Ad = DenseTensor(A.todense(), A.indices)
        env = ctm_tensor_c4v(A, chi=4, max_iter=200, conv_tol=1e-10)
        env_d = ctm_tensor_c4v(Ad, chi=4, max_iter=200, conv_tol=1e-10)
        sv = norm_sv(corner_edge(env)[0])
        assert sv[1] / sv[0] > 0.1, sv[:2]  # not a product state
        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
        E_d = float(compute_energy_ctm_tensor(Ad, env_d, heisenberg_gate, d=2))
        assert abs(E - E_d) <= 1e-12, (E, E_d)

    @pytest.mark.parametrize("chi", [7, 10])
    def test_z4_self_partner_charge_next_to_pairs(self, chi, heisenberg_gate):
        """Z4: the self-partner charge 2 is kept next to the pairs (1, 3)."""
        A = c4v_fixture(ZnSymmetry(4), [0, 1], [0, 1])
        Ad = DenseTensor(A.todense(), A.indices)
        env = ctm_tensor_c4v(A, chi=chi, max_iter=200, conv_tol=1e-10)
        env_d = ctm_tensor_c4v(Ad, chi=chi, max_iter=200, conv_tol=1e-10)
        a = _build_double_layer_tensor(A)
        C, T = corner_edge(env)
        charges = np.asarray(C.indices[C.labels().index("c_a")].charges).tolist()
        assert 2 in charges, f"vacuous: no state of charge 2 in {charges}"
        sv = norm_sv(C)
        assert sv_gap(norm_sv(_c4v_staggered_sweep(C, T, a, chi)[0]), sv) < 1e-9
        assert cut_gap(_c4v_staggered_enlarged_corner(C, T, a), chi) >= 0.05
        assert sv[1] / sv[0] > 0.1, sv[:2]
        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
        E_d = float(compute_energy_ctm_tensor(Ad, env_d, heisenberg_gate, d=2))
        assert abs(E - E_d) <= 1e-8, (E, E_d)

    def test_d3_chi_below_d_squared(self, heisenberg_gate):
        """D=3 at chi=6 < D^2 = 9: against dense C4v."""
        A = c4v_fixture(U1Symmetry(), [-1, 0, 1], [0, 1])
        Ad = DenseTensor(A.todense(), A.indices)
        chi = 6
        env = ctm_tensor_c4v(A, chi=chi, max_iter=200, conv_tol=1e-10)
        env_d = ctm_tensor_c4v(Ad, chi=chi, max_iter=200, conv_tol=1e-10)
        a = _build_double_layer_tensor(A)
        C, T = corner_edge(env)
        sv = norm_sv(C)
        assert sv_gap(norm_sv(_c4v_staggered_sweep(C, T, a, chi)[0]), sv) < 1e-9
        Q = _c4v_staggered_enlarged_corner(C, T, a)
        assert cut_gap(Q, chi) >= 0.05 and charged_block_ratio(Q) >= 1e-3
        assert sv[1] / sv[0] > 0.1, sv[:2]
        E = float(compute_energy_ctm_tensor(A, env, heisenberg_gate, d=2))
        E_d = float(compute_energy_ctm_tensor(Ad, env_d, heisenberg_gate, d=2))
        assert abs(E - E_d) <= 1e-8, (E, E_d)
