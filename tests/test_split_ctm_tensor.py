"""Tests for split CTM with Tensor protocol (polymorphic dense/symmetric)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._split_ctm_tensor import (
    SplitCTMTensorEnv,
    _split_ctm_tensor_sweep,
    compute_energy_split_ctm_tensor,
    ctm_split_tensor,
    initialize_split_ctm_tensor_env,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.ipeps_ctm import ctm, ctm_split
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


def make_random_dense_site(D: int, d: int, seed: int) -> DenseTensor:
    """Build a random U(1)-trivial DenseTensor iPEPS site for parity tests."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    z_D = np.zeros(D, dtype=np.int32)
    z_d = np.zeros(d, dtype=np.int32)
    return DenseTensor(
        data,
        (
            TensorIndex.from_charges(sym, z_D, FlowDirection.OUT, label="u"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label="d"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.OUT, label="l"),
            TensorIndex.from_charges(sym, z_D, FlowDirection.IN, label="r"),
            TensorIndex.from_charges(sym, z_d, FlowDirection.IN, label="phys"),
        ),
    )


def make_random_fermionic_site(D: int, d: int, seed: int) -> SymmetricTensor:
    """Build a random FermionParity-symmetric iPEPS site for parity tests.

    Mirrors :func:`tenax.algorithms.fermionic_ipeps._build_initial_fpeps_tensor`:
    virtual charges alternate 0,1,0,1,... and physical charges are [0, 1]
    (empty, occupied).  Uses ``random_normal`` to populate every allowed
    block — strictly safer than ``from_dense`` which would silently drop
    charge-violating entries.
    """
    from tenax.core.symmetry import FermionParity

    sym = FermionParity()
    virt_charges = np.array([i % 2 for i in range(D)], dtype=np.int32)
    phys_charges = np.array([0, 1], dtype=np.int32)[:d]
    indices = (
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    key = jax.random.PRNGKey(seed)
    A = SymmetricTensor.random_normal(indices, key)
    norm_val = float(A.norm())
    if norm_val > 0:
        A = A * (1.0 / norm_val)
    return A


@pytest.fixture
def small_peps_dense():
    """Random DenseTensor iPEPS site tensor, D=2, d=2."""
    key = jax.random.PRNGKey(42)
    D, d = 2, 2
    data = jax.random.normal(key, (D, D, D, D, d))
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
def small_peps_symmetric():
    """Random SymmetricTensor iPEPS site tensor with trivial U(1) charges."""
    key = jax.random.PRNGKey(99)
    D, d = 2, 2
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
    data = jax.random.normal(key, (D, D, D, D, d))
    return SymmetricTensor.from_dense(data, indices)


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
# Public API smoke test                                                #
# ------------------------------------------------------------------ #


def test_public_exports_resolve():
    """Top-level tenax package exposes the new split-CTM energy entry points."""
    import tenax

    assert hasattr(tenax, "compute_energy_split_ctm_tensor_2site")
    assert hasattr(tenax, "compute_energy_split_ctm_tensor_multisite")


# ------------------------------------------------------------------ #
# Phase 1: Initialization tests                                        #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorInit:
    """Tests for SplitCTMTensorEnv initialization."""

    def test_dense_init_shapes(self, small_peps_dense):
        """All 12 tensors should have correct shapes and labels."""
        chi, chi_I, D = 8, 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        assert isinstance(env, SplitCTMTensorEnv)

        # Corners: (chi, chi)
        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert C.todense().shape == (chi, chi)
            assert C.ndim == 2

        # Ket edges: (chi, D, chi_I)
        for T_ket in [env.T1_ket, env.T2_ket, env.T3_ket, env.T4_ket]:
            assert T_ket.todense().shape == (chi, D, chi_I)

        # Bra edges: (chi_I, D, chi)
        for T_bra in [env.T1_bra, env.T2_bra, env.T3_bra, env.T4_bra]:
            assert T_bra.todense().shape == (chi_I, D, chi)

    def test_dense_init_labels(self, small_peps_dense):
        """Check that labels are assigned correctly."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)

        assert env.C1.labels() == ("c1_d", "c1_r")
        assert env.C2.labels() == ("c2_l", "c2_d")
        assert env.C3.labels() == ("c3_u", "c3_l")
        assert env.C4.labels() == ("c4_r", "c4_u")

        assert env.T1_ket.labels() == ("t1k_l", "u_ket", "t1k_I")
        assert env.T1_bra.labels() == ("t1b_I", "u_bra", "t1b_r")

    def test_dense_init_finite(self, small_peps_dense):
        """All initialized tensors should be finite."""
        chi, chi_I = 8, 4
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))

    def test_symmetric_init_shapes(self, small_peps_symmetric):
        """SymmetricTensor initialization should produce correct shapes."""
        chi, chi_I = 4, 2
        D = 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)
        assert isinstance(env, SplitCTMTensorEnv)

        for C in [env.C1, env.C2, env.C3, env.C4]:
            assert isinstance(C, SymmetricTensor)
            assert C.todense().shape == (chi, chi)

        for T_ket in [env.T1_ket, env.T2_ket, env.T3_ket, env.T4_ket]:
            assert isinstance(T_ket, SymmetricTensor)
            assert T_ket.todense().shape == (chi, D, chi_I)

        for T_bra in [env.T1_bra, env.T2_bra, env.T3_bra, env.T4_bra]:
            assert isinstance(T_bra, SymmetricTensor)
            assert T_bra.todense().shape == (chi_I, D, chi)

    @pytest.mark.parametrize("D, chi", [(2, 4), (2, 6), (2, 8), (3, 8), (3, 12)])
    def test_fermionic_interlayer_bonds_match(self, D, chi):
        """Issue #391: ket/bra interlayer (`*_I`) bonds are the SAME SVD bond
        and must carry identical raw charges.

        Pre-fix the ket derived ``qI = qc + qd`` while the bra derived
        ``qI = qd − qc``; for non-trivial charges these gave different raw
        sequences AND different per-parity dimensions, so the relabel-and-
        contract step in ``_split_ctm_move_left`` either silently broadcast
        size-1 dims or crashed.
        """
        A = make_random_fermionic_site(D, d=2, seed=70)
        env = initialize_split_ctm_tensor_env(A, chi=chi, chi_I=chi)
        for ket, bra in [
            (env.T1_ket, env.T1_bra),
            (env.T2_ket, env.T2_bra),
            (env.T3_ket, env.T3_bra),
            (env.T4_ket, env.T4_bra),
        ]:
            ket_I = ket.indices[
                ket.labels().index(
                    next(lbl for lbl in ket.labels() if lbl.endswith("_I"))
                )
            ]
            bra_I = bra.indices[
                bra.labels().index(
                    next(lbl for lbl in bra.labels() if lbl.endswith("_I"))
                )
            ]
            assert np.array_equal(
                np.asarray(ket_I.charges), np.asarray(bra_I.charges)
            ), (
                f"ket I-charges {list(ket_I.charges)} != bra I-charges {list(bra_I.charges)}"
            )
            # Flows on the I-bond must be opposite (one IN, one OUT) so the
            # interlayer contraction is flow-compatible.
            assert int(ket_I.flow) == -int(bra_I.flow)

    @pytest.mark.parametrize("chi", [6, 8])
    def test_fermionic_ctm_runs_at_chi_8(self, chi):
        """Issue #391: ``ctm_split_tensor`` must not crash inside the projector
        path on a FermionParity site at chi=8/D=2.  Pre-fix this raised
        ``ValueError: Size of label 'b' for operand 1 (3) does not match
        previous terms (2)`` from inside ``_split_ctm_move_left``.
        """
        A = make_random_fermionic_site(D=2, d=2, seed=70)
        env = ctm_split_tensor(A, chi=chi, max_iter=2, chi_I=chi)
        for t in env:
            arr = t.todense() if hasattr(t, "todense") else t
            assert jnp.all(jnp.isfinite(arr))


# ------------------------------------------------------------------ #
# Phase 2: Single-move tests                                           #
# ------------------------------------------------------------------ #


class TestSplitCTMMoves:
    """Tests for individual CTM moves."""

    def test_one_sweep_produces_finite(self, small_peps_dense):
        """One full sweep should produce finite tensors."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_dense, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, small_peps_dense, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "Sweep produced non-finite tensors"
            )


class TestSplitCTMBoundedEdge:
    """Issue #463/#641: the dense forward move must avoid the chi^2*D^6 edge.

    Each split move builds the double-layer biorthogonal corner-pair projector
    ``(P_1, P_2)`` and applies it to the grown edge.  The default
    (memory-bounded) path factorizes ``P_1``/``P_2`` and routes through
    :func:`_grow_and_project_bounded_lr`, which absorbs the factors into the
    open half-edges before the interlayer join — peak intermediate
    ``chi^2*D^3*d`` instead of ``chi^2*D^6``.  It must reproduce the closed
    ``_lr`` path (grow the full ``chi^2*D^6`` edge, then
    :func:`_project_grown_edge_tensor_lr`) to machine precision.

    The closed path is reachable via the ``moves._FORCE_CLOSED_EDGE`` flag.
    """

    @pytest.mark.parametrize("D, seed", [(2, 1), (3, 7), (4, 11)])
    def test_bounded_matches_closed_path_all_moves(self, D, seed):
        """Bounded projected edge == closed-path projected edge, every move.

        Spies on :func:`_svd_split_edge_tensor` (the single consumer of the
        4-leg projected edge in every move) to capture that edge, runs each
        move once with the bounded path (default) and once with the closed
        path (``_FORCE_CLOSED_EDGE``), and compares the captured edges.
        """
        import tenax.algorithms._split_ctm_tensor_moves as moves

        site = make_random_dense_site(D, 2, seed)
        chi = 8
        env = initialize_split_ctm_tensor_env(site, chi, chi)
        for _ in range(6):
            env = _split_ctm_tensor_sweep(env, site, chi, chi, True)
        A_bar = site.bar()

        move_fns = [
            moves._split_ctm_move_left,
            moves._split_ctm_move_right,
            moves._split_ctm_move_top,
            moves._split_ctm_move_bottom,
        ]

        def _capture_projected_edge(move_fn, force_closed):
            captured = {}
            orig = moves._svd_split_edge_tensor

            def spy(T, *a, **k):
                captured["edge"] = T
                return orig(T, *a, **k)

            saved = moves._FORCE_CLOSED_EDGE
            moves._svd_split_edge_tensor = spy
            moves._FORCE_CLOSED_EDGE = force_closed
            try:
                move_fn(env, site, A_bar, chi, chi)
            finally:
                moves._svd_split_edge_tensor = orig
                moves._FORCE_CLOSED_EDGE = saved
            return captured["edge"]

        for move_fn in move_fns:
            bounded = _capture_projected_edge(move_fn, force_closed=False)
            closed = _capture_projected_edge(move_fn, force_closed=True)

            ref = closed.todense()
            perm = tuple(bounded.labels().index(lbl) for lbl in closed.labels())
            got = bounded.transpose(perm).todense()
            scale = float(jnp.max(jnp.abs(ref))) + 1e-30
            relerr = float(jnp.max(jnp.abs(ref - got))) / scale
            assert relerr < 1e-10, f"{move_fn.__name__}: relerr={relerr:.2e}"

    def test_bounded_peak_is_chi2_d4_bounded(self):
        """The dense move must not allocate a chi^2*D^6 intermediate.

        Tracks the largest dense array materialised during one move with the
        bounded path (default) and asserts it stays at/under chi^2*D^4 and far
        below the closed chi^2*D^6 grown edge (issue #463/#641).  D=4, chi=16:
        chi^2*D^6 = 1.05e9 elems; the bounded path's peak is ~chi^2*D^3*d.
        """
        import tenax.algorithms._split_ctm_tensor_moves as moves

        D, chi, d = 4, 16, 2
        site = make_random_dense_site(D, d, seed=3)
        env = initialize_split_ctm_tensor_env(site, chi, chi)
        for _ in range(3):
            env = _split_ctm_tensor_sweep(env, site, chi, chi, True)
        A_bar = site.bar()

        # Instrument DenseTensor construction to record the largest array size
        # allocated during the bounded move's edge application.
        peak = {"n": 0}
        real_init = DenseTensor.__init__

        def tracking_init(self, data, indices, *a, **k):
            try:
                peak["n"] = max(peak["n"], int(np.prod(data.shape)))
            except Exception:
                pass
            return real_init(self, data, indices, *a, **k)

        saved = moves._FORCE_CLOSED_EDGE
        moves._FORCE_CLOSED_EDGE = False  # bounded path (default)
        DenseTensor.__init__ = tracking_init
        try:
            moves._split_ctm_move_left(env, site, A_bar, chi, chi)
        finally:
            DenseTensor.__init__ = real_init
            moves._FORCE_CLOSED_EDGE = saved

        closed_peak = chi**2 * D**6  # the chi^2*D^6 edge we must avoid
        target = chi**2 * D**4  # the issue #641 chi^2*D^4 bound
        assert peak["n"] > 0
        assert peak["n"] <= target, (
            f"bounded peak {peak['n']:.2e} exceeds chi^2*D^4={target:.2e}"
        )
        assert peak["n"] < closed_peak / 10, (
            f"bounded peak {peak['n']:.2e} not far below chi^2*D^6={closed_peak:.2e}"
        )


# ------------------------------------------------------------------ #
# Phase 3: Convergence tests                                           #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorConvergence:
    """Tests for full CTM convergence."""

    def test_converges(self, small_peps_dense):
        """Split-CTM should produce finite environment after convergence."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=4)
        assert isinstance(env, SplitCTMTensorEnv)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))

    def test_chi_I_equals_chi(self, small_peps_dense):
        """chi_I=chi should also work (no interlayer compression)."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=8)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense()))

    def test_return_info_reports_iterations_and_converged(self, small_peps_dense):
        """return_info=True returns (env, info) with a sweep count and a bool
        converged flag; the default call still returns a bare env."""
        env_only = ctm_split_tensor(small_peps_dense, chi=8, max_iter=30, chi_I=4)
        assert isinstance(env_only, SplitCTMTensorEnv)  # default unchanged

        env, info = ctm_split_tensor(
            small_peps_dense, chi=8, max_iter=30, chi_I=4, return_info=True
        )
        assert isinstance(env, SplitCTMTensorEnv)
        assert isinstance(info.iterations, int) and info.iterations >= 1
        assert isinstance(info.converged, bool)


# ------------------------------------------------------------------ #
# Correctness tests                                                    #
# ------------------------------------------------------------------ #


class TestSplitCTMTensorEnergy:
    """Energy correctness tests."""

    def test_energy_is_finite(self, small_peps_dense, heisenberg_gate):
        """Split-CTM should produce finite energy."""
        env = ctm_split_tensor(small_peps_dense, chi=8, max_iter=50)
        E = compute_energy_split_ctm_tensor(small_peps_dense, env, heisenberg_gate, d=2)
        assert jnp.isfinite(E)

    def test_energy_roundtrip_via_standard(self, small_peps_dense, heisenberg_gate):
        """Energy via split env should match energy via standard-converted env.

        This mirrors the existing ``test_split_ctm_energy_matches_standard``
        for the dense split-CTM: convert the split environment to a standard
        CTMTensorEnv and verify the energy is identical.
        """
        from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor import _split_env_to_tensor_standard

        d = 2
        chi = 8
        chi_I = chi * 2  # lossless

        env = ctm_split_tensor(small_peps_dense, chi=chi, max_iter=50, chi_I=chi_I)
        E_split = compute_energy_split_ctm_tensor(
            small_peps_dense, env, heisenberg_gate, d
        )

        # Manually convert and compute
        std_env = _split_env_to_tensor_standard(env)
        E_from_std = compute_energy_ctm_tensor(
            small_peps_dense, std_env, heisenberg_gate, d
        )

        assert jnp.abs(E_split - E_from_std) < 1e-12, (
            f"Roundtrip mismatch: split={float(E_split)}, from_std={float(E_from_std)}"
        )

    # test_grow_edge_matches_double_layer removed: 4× bit-parity checks
    # (atol=1e-12) of `_grow_edge_no_double_layer` against the old einsum
    # path on random tensors.  The user-visible contract (energy via the
    # no-double-layer path matches the standard path) is already covered
    # by `test_energy_roundtrip_via_standard` above on the same fixture.


# ------------------------------------------------------------------ #
# SymmetricTensor tests                                                #
# ------------------------------------------------------------------ #


class TestSplitCTMSymmetric:
    """Tests for SymmetricTensor iPEPS with trivial and nontrivial charges."""

    def test_symmetric_one_sweep_finite(self, small_peps_symmetric):
        """One CTM sweep with trivial-charge SymmetricTensor A produces finite tensors."""
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, small_peps_symmetric, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "SymmetricTensor sweep produced non-finite tensors"
            )

    def test_fermionic_u1_one_sweep_finite(self):
        """One CTM sweep with FermionicU1 charges produces finite tensors.

        This is the key regression test: with dagger(), the physical trace
        loses blocks because charge 1 is dualled to -1 and mismatches.
        With bar(), charges stay identical so all blocks are preserved.
        """
        from tenax.core.symmetry import FermionicU1

        key = jax.random.PRNGKey(77)
        sym = FermionicU1()
        virt_charges = np.array([0, 1], dtype=np.int32)
        phys_charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="u"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="d"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="l"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="r"
            ),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        A = SymmetricTensor.random_normal(indices, key)
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "FermionicU1 sweep produced non-finite tensors"
            )

    def test_symmetric_multi_sweep_converges(self, small_peps_symmetric):
        """Multiple CTM sweeps with SymmetricTensor A don't crash (regression for type mixing)."""
        env = ctm_split_tensor(small_peps_symmetric, chi=4, max_iter=5, chi_I=2)
        for t in env:
            assert jnp.all(jnp.isfinite(t.todense())), (
                "Multi-sweep symmetric CTM produced non-finite tensors"
            )

    def test_symmetric_energy_matches_dense(self, small_peps_dense, heisenberg_gate):
        """SymmetricTensor split-CTM energy should match DenseTensor result.

        Uses the same tensor data wrapped as both DenseTensor and SymmetricTensor
        (with trivial U(1) charges) to verify the symmetric path is correct.
        """
        # Build SymmetricTensor with same data as small_peps_dense
        A_sym = SymmetricTensor.from_dense(
            small_peps_dense.todense(), small_peps_dense.indices
        )

        chi, chi_I = 8, 4
        d = 2

        env_d = ctm_split_tensor(small_peps_dense, chi=chi, max_iter=50, chi_I=chi_I)
        E_dense = compute_energy_split_ctm_tensor(
            small_peps_dense, env_d, heisenberg_gate, d
        )

        env_s = ctm_split_tensor(A_sym, chi=chi, max_iter=50, chi_I=chi_I)
        E_sym = compute_energy_split_ctm_tensor(A_sym, env_s, heisenberg_gate, d)

        assert jnp.isfinite(E_sym), f"Symmetric energy is not finite: {float(E_sym)}"
        assert jnp.abs(E_dense - E_sym) < 1e-6, (
            f"Energy mismatch: dense={float(E_dense)}, sym={float(E_sym)}"
        )

    def test_fermionic_u1_charges_preserved_across_sweeps(self):
        """Nontrivial charge sectors survive multiple CTM sweeps (regression for charge collapse)."""
        from tenax.core.symmetry import FermionicU1

        key = jax.random.PRNGKey(77)
        sym = FermionicU1()
        virt_charges = np.array([0, 1], dtype=np.int32)
        phys_charges = np.array([0, 1], dtype=np.int32)
        indices = (
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="u"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="d"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.OUT, label="l"
            ),
            TensorIndex.from_charges(
                sym, virt_charges.copy(), FlowDirection.IN, label="r"
            ),
            TensorIndex.from_charges(
                sym, phys_charges.copy(), FlowDirection.IN, label="phys"
            ),
        )
        A = SymmetricTensor.random_normal(indices, key)
        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(A, chi, chi_I)
        # Run 3 sweeps
        for _ in range(3):
            env = _split_ctm_tensor_sweep(env, A, chi, chi_I, True)
        # All tensors must remain SymmetricTensors with at least 1 block.
        # Block count may decrease from initial because the block-sparse
        # projector keeps the dominant charge sectors (unlike the old
        # dense code which forced charge distribution via _derive_charges).
        for t in env:
            assert isinstance(t, SymmetricTensor), (
                f"Expected SymmetricTensor, got {type(t)}"
            )
            assert t.n_blocks >= 1, (
                "All blocks collapsed to 0 — environment degenerated"
            )

    def test_symmetric_sweep_no_todense(self, small_peps_symmetric):
        """CTM sweeps on SymmetricTensor must not call todense() or from_dense().

        This guards against regressions that silently densify during the
        symmetric path.  The energy measurement is excluded because it
        currently bridges to dense (known limitation).
        """
        from unittest.mock import patch

        chi, chi_I = 4, 2
        env = initialize_split_ctm_tensor_env(small_peps_symmetric, chi, chi_I)

        todense_calls = []
        from_dense_calls = []

        orig_todense = SymmetricTensor.todense
        orig_from_dense = SymmetricTensor.from_dense

        def tracking_todense(self):
            todense_calls.append(True)
            return orig_todense(self)

        @classmethod
        def tracking_from_dense(cls, *args, **kwargs):
            from_dense_calls.append(True)
            return orig_from_dense(*args, **kwargs)

        with (
            patch.object(SymmetricTensor, "todense", tracking_todense),
            patch.object(SymmetricTensor, "from_dense", tracking_from_dense),
        ):
            _split_ctm_tensor_sweep(env, small_peps_symmetric, chi, chi_I, True)

        assert len(todense_calls) == 0, (
            f"todense() called {len(todense_calls)} times during symmetric sweep"
        )
        assert len(from_dense_calls) == 0, (
            f"from_dense() called {len(from_dense_calls)} times during symmetric sweep"
        )


class TestSplitEdgeHelper:
    """Tests for the new _make_split_edge / _make_split_edges helpers."""

    def test_make_split_edge_shape_and_labels(self, small_peps_dense):
        """_make_split_edge contracts T_ket·T_bra on _I, leaves D-legs unfused."""
        from tenax.algorithms._split_ctm_tensor_energy import _make_split_edge

        env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
        T1 = _make_split_edge(
            env.T1_ket,
            env.T1_bra,
            ket_I="t1k_I",
            bra_I="t1b_I",
            ket_chi="t1k_l",
            bra_chi="t1b_r",
            out_chi_l="t1_l",
            out_chi_r="t1_r",
        )
        labels = T1.labels()
        # Four legs: chi_l, u_ket, u_bra, chi_r — D-legs unchanged from inputs.
        assert set(labels) == {"t1_l", "u_ket", "u_bra", "t1_r"}
        # Dimensions: (chi, D, D, chi). With D=2, chi=4 → (4, 2, 2, 4).
        dim_by_label = {idx.label: idx.dim for idx in T1.indices}
        assert dim_by_label["t1_l"] == 4
        assert dim_by_label["t1_r"] == 4
        assert dim_by_label["u_ket"] == 2
        assert dim_by_label["u_bra"] == 2

    def test_make_split_edges_no_label_collisions(self, small_peps_dense):
        """The four T_split tensors share no D-leg labels; chi labels follow the
        standard CTM convention (t1_l/t1_r, t2_u/t2_d, t3_l/t3_r, t4_d/t4_u)."""
        from tenax.algorithms._split_ctm_tensor_energy import _make_split_edges

        env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
        splits = _make_split_edges(env)

        assert set(splits.keys()) == {"T1", "T2", "T3", "T4"}
        # D-leg label sets per edge are disjoint:
        d_labels = {
            "T1": {"u_ket", "u_bra"},
            "T2": {"r_ket", "r_bra"},
            "T3": {"d_ket", "d_bra"},
            "T4": {"l_ket", "l_bra"},
        }
        for name, want in d_labels.items():
            labs = set(splits[name].labels())
            assert want.issubset(labs), f"{name} missing {want - labs}"
        # No D-leg label collides across edges:
        all_d = set().union(*d_labels.values())
        assert len(all_d) == 8  # all distinct


class TestSplitRDMs:
    """RDM tests for the split-CTM path.

    The single-env per-RDM bit-parity vs shim checks were removed
    (issue #487): they exercised random-tensor inputs whose RDMs are not
    physically valid (eigenvalues can be negative), so neither bit-parity
    against the shim nor physics-property smoke (Hermitian + PSD + trace-1)
    is a meaningful contract on this input distribution.  The
    contraction-sequence drift they detected at small χ is not a
    production-level failure mode — the production contract is checked
    end-to-end by ``test_compute_energy_split_*_matches_shim`` below,
    where the inputs are physical bond gates and the parity scalar (the
    energy) has well-defined sign and magnitude.

    The two ``*_2site_matches_shim`` tests below are retained as
    regression guards for the delegation introduced by PRs #479 and #486:
    the split-aware ``*_2site`` functions now route through the shim, so
    bit-parity holds by construction; the test catches accidental
    reverts of the delegation.
    """

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm1x2_2site_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm1x2_split_tensor_2site,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=10)
        B = make_random_dense_site(D, d=2, seed=11)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm1x2_split_tensor_2site(A, B, env_A, env_B)
        rdm_shim = _rdm1x2_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

        # Distinctness check: mixed-env result must differ from single-env result with A on both sites.
        from tenax.algorithms._split_ctm_tensor_energy import _rdm1x2_split_tensor

        rdm_single = _rdm1x2_split_tensor(A, env_A)
        assert not jnp.allclose(rdm_split, rdm_single, atol=1e-6)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm2x1_2site_matches_shim(self, D, chi):
        from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm2x1_split_tensor_2site,
            _split_env_to_tensor_standard,
        )

        A = make_random_dense_site(D, d=2, seed=20)
        B = make_random_dense_site(D, d=2, seed=21)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm2x1_split_tensor_2site(A, B, env_A, env_B)
        rdm_shim = _rdm2x1_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
        )
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)

        # Distinctness check: mixed-env result must differ from single-env result with A on both sites.
        from tenax.algorithms._split_ctm_tensor_energy import _rdm2x1_split_tensor

        rdm_single = _rdm2x1_split_tensor(A, env_A)
        assert not jnp.allclose(rdm_split, rdm_single, atol=1e-6)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
    def test_compute_energy_split_native_matches_shim(self, D, chi, heisenberg_gate):
        """Split-aware energy must match shim energy at small D."""
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor,
        )

        A = make_random_dense_site(D, d=2, seed=30)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor(A, env, heisenberg_gate, d=2)
        E_shim = compute_energy_ctm_tensor(
            A, _split_env_to_tensor_standard(env), heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
    def test_compute_energy_split_native_grad_matches_shim(
        self, D, chi, heisenberg_gate
    ):
        """Split-aware energy GRADIENT must match the shim gradient.

        Phase-1 acceptance for #463 ("1e-8 gradient" parity). Holds the
        converged split env fixed (a constant w.r.t. the differentiated A) and
        differentiates the per-site energy w.r.t. the site tensor through both
        the split-aware RDM path and the shim path built from the *same* env.
        Equal energies (1e-10, asserted above) imply equal A-gradients up to
        the backward's accumulation error. Complements
        ``test_compute_energy_split_native_matches_shim`` which only pins the
        forward value.
        """
        import jax

        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor,
        )

        A = make_random_dense_site(D, d=2, seed=30)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_std = _split_env_to_tensor_standard(env)

        def loss_split(a):
            return compute_energy_split_ctm_tensor(a, env, heisenberg_gate, d=2).real

        def loss_shim(a):
            return compute_energy_ctm_tensor(a, env_std, heisenberg_gate, d=2).real

        g_split = jax.tree_util.tree_leaves(jax.grad(loss_split)(A))
        g_shim = jax.tree_util.tree_leaves(jax.grad(loss_shim)(A))
        assert len(g_split) == len(g_shim) and g_split, "gradient pytree mismatch"
        for ls, lh in zip(g_split, g_shim):
            np.testing.assert_allclose(
                np.asarray(ls),
                np.asarray(lh),
                atol=1e-8,
                rtol=1e-8,
                err_msg="split-aware energy gradient diverges from shim gradient",
            )

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_2site_matches_shim(self, D, chi, heisenberg_gate):
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_2site,
        )

        A = make_random_dense_site(D, d=2, seed=40)
        B = make_random_dense_site(D, d=2, seed=41)
        env_A = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        env_B = ctm_split_tensor(B, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor_2site(
            A, B, env_A, env_B, heisenberg_gate, d=2
        )
        E_shim = compute_energy_ctm_tensor_2site(
            A,
            B,
            _split_env_to_tensor_standard(env_A),
            _split_env_to_tensor_standard(env_B),
            heisenberg_gate,
            d=2,
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_multisite_matches_shim(self, D, chi, heisenberg_gate):
        """Split-aware multisite energy must match shim at small D on a Y-shaped 3-site cell."""
        from tenax.algorithms._ctm_tensor_energy import (
            compute_energy_ctm_tensor_multisite,
        )
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_multisite,
        )

        coords = [(0, 0), (1, 0), (0, 1)]
        site_tensors = {
            (0, 0): make_random_dense_site(D, d=2, seed=50),
            (1, 0): make_random_dense_site(D, d=2, seed=51),
            (0, 1): make_random_dense_site(D, d=2, seed=52),
        }
        envs_split = {
            c: ctm_split_tensor(site_tensors[c], chi=chi, max_iter=20, chi_I=chi)
            for c in coords
        }
        # Y-shaped neighbors: (0,0) connects right→(1,0) and bottom→(0,1).
        # Other coords loop back to themselves on directions that don't exit the cell.
        neighbors = {
            (0, 0): {
                "right": (1, 0),
                "bottom": (0, 1),
                "left": (1, 0),
                "top": (0, 1),
            },
            (1, 0): {
                "left": (0, 0),
                "top": (0, 1),
                "right": (0, 0),
                "bottom": (0, 1),
            },
            (0, 1): {
                "top": (0, 0),
                "right": (1, 0),
                "bottom": (0, 0),
                "left": (1, 0),
            },
        }

        E_split = compute_energy_split_ctm_tensor_multisite(
            site_tensors, envs_split, neighbors, heisenberg_gate, d=2
        )
        envs_std = {c: _split_env_to_tensor_standard(envs_split[c]) for c in coords}
        E_shim = compute_energy_ctm_tensor_multisite(
            site_tensors, envs_std, neighbors, heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_compute_energy_split_multisite_1site_cell(self, D, chi, heisenberg_gate):
        """Single-site cell with self-loops exercises coord == nb_coord branch."""
        from tenax.algorithms._ctm_tensor_energy import (
            compute_energy_ctm_tensor_multisite,
        )
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor_multisite,
        )

        coord = (0, 0)
        A = make_random_dense_site(D, d=2, seed=60)
        site_tensors = {coord: A}
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)
        envs_split = {coord: env}
        # All directions self-loop — every bond is single-env (coord == nb_coord).
        neighbors = {
            coord: {"right": coord, "bottom": coord, "left": coord, "top": coord}
        }

        E_split = compute_energy_split_ctm_tensor_multisite(
            site_tensors, envs_split, neighbors, heisenberg_gate, d=2
        )
        envs_std = {coord: _split_env_to_tensor_standard(env)}
        E_shim = compute_energy_ctm_tensor_multisite(
            site_tensors, envs_std, neighbors, heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)


@pytest.mark.slow
class TestSplitRDMsFermionic:
    """Native split-aware energy matches shim on FermionParity sites (#392).

    Pre-#555 the split-aware path absorbed an extra Koszul phase via
    ``A.bar_super()`` that ``_build_double_layer_open_tensor``'s raw
    ``fuse_indices`` (used by the standard path) couldn't cancel; the
    energies disagreed once the bra was non-trivial.  PR #557 removed both
    the contractor's auto-Koszul and ``bar_super()`` (only ``A.bar()``
    remains), eliminating the convention mismatch.  The fermionic shim
    fallback that ``compute_energy_split_ctm_tensor`` used to apply is now
    gone, so this parity test exercises the native ``chi²·D⁴`` split-aware
    contraction directly.

    Note on chi choice: with ``FermionParity`` virtual charges (alternating
    0/1), ``ctm_split_tensor`` converges cleanly at every chi after the
    fix for issue #391 (canonical SVD-bond charges shared between ket and
    bra).
    """

    @pytest.mark.parametrize("D, chi", [(2, 6), (2, 8), (3, 12)])
    def test_fermionic_energy_matches_shim(self, D, chi, heisenberg_gate):
        from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
        from tenax.algorithms._split_ctm_tensor_energy import (
            _split_env_to_tensor_standard,
            compute_energy_split_ctm_tensor,
        )

        A = make_random_fermionic_site(D, d=2, seed=70)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor(A, env, heisenberg_gate, d=2)
        E_shim = compute_energy_ctm_tensor(
            A, _split_env_to_tensor_standard(env), heisenberg_gate, d=2
        )
        assert jnp.allclose(E_split, E_shim, atol=1e-10)
