"""Tests for fermionic iPEPS (fPEPS) algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    _trotter_gate,
    fpeps,
    spinless_fermion_gate,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def fp():
    return FermionParity()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_config():
    return FPEPSConfig()


# ------------------------------------------------------------------ #
# Task 1: spinless_fermion_gate                                        #
# ------------------------------------------------------------------ #


class TestSpinlessFermionGate:
    """Tests for the 2-site Hamiltonian gate H = -t(c†c + h.c.) + V(nn)."""

    def test_returns_symmetric_tensor(self, default_config):
        H = spinless_fermion_gate(default_config)
        assert isinstance(H, SymmetricTensor)

    def test_shape_is_2x2x2x2(self, default_config):
        H = spinless_fermion_gate(default_config)
        dense = H.todense()
        assert dense.shape == (2, 2, 2, 2)

    def test_labels(self, default_config):
        H = spinless_fermion_gate(default_config)
        assert H.labels() == ("si", "sj", "si_out", "sj_out")

    def test_flows(self, default_config):
        H = spinless_fermion_gate(default_config)
        flows = [idx.flow for idx in H.indices]
        assert flows == [
            FlowDirection.IN,
            FlowDirection.IN,
            FlowDirection.OUT,
            FlowDirection.OUT,
        ]

    def test_hermitian(self, default_config):
        """H reshaped as 4x4 matrix should be Hermitian."""
        H = spinless_fermion_gate(default_config)
        dense = H.todense().reshape(4, 4)
        np.testing.assert_allclose(dense, dense.T.conj(), atol=1e-14)

    def test_free_fermion_spectrum(self):
        """For V=0 (free fermion), eigenvalues of -t(c†c + h.c.) are known."""
        cfg = FPEPSConfig(t=1.0, V=0.0)
        H = spinless_fermion_gate(cfg)
        dense = H.todense().reshape(4, 4)
        eigvals = np.sort(np.linalg.eigvalsh(np.array(dense)))
        # Basis: |00>, |01>, |10>, |11>
        # H = -t(c1†c2 + c2†c1)
        # In the 1-particle sector: eigenvalues -t, +t
        # 0-particle and 2-particle sectors: eigenvalue 0
        expected = np.array([-1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(eigvals, expected, atol=1e-14)

    def test_interaction_diagonal(self):
        """V*n_i*n_j should only contribute to the |11> state."""
        cfg = FPEPSConfig(t=0.0, V=2.5)
        H = spinless_fermion_gate(cfg)
        dense = H.todense().reshape(4, 4)
        # Only (|11>-><11>) = V
        expected = np.diag([0.0, 0.0, 0.0, 2.5])
        np.testing.assert_allclose(dense, expected, atol=1e-14)


# ------------------------------------------------------------------ #
# Task 2: _trotter_gate                                                #
# ------------------------------------------------------------------ #


class TestTrotterGate:
    """Tests for the Trotter gate exp(-dt * H)."""

    def test_returns_symmetric_tensor(self, default_config):
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        assert isinstance(G, SymmetricTensor)

    def test_shape_preserved(self, default_config):
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        assert G.todense().shape == (2, 2, 2, 2)

    def test_dt_zero_is_identity(self, default_config):
        """exp(0 * H) = identity operator reshaped to (2,2,2,2)."""
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, dt=0.0)
        dense = G.todense().reshape(4, 4)
        np.testing.assert_allclose(dense, np.eye(4), atol=1e-12)

    def test_symmetric_positive_definite(self, default_config):
        """exp(-dt*H) for real dt and Hermitian H is symmetric positive-definite."""
        H = spinless_fermion_gate(default_config)
        G = _trotter_gate(H, default_config.dt)
        dense = np.array(G.todense().reshape(4, 4))
        # Symmetric
        np.testing.assert_allclose(dense, dense.T, atol=1e-14)
        # Positive-definite: all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(dense)
        assert np.all(eigvals > 0)


# ------------------------------------------------------------------ #
# Task 3: _initialize_fpeps                                            #
# ------------------------------------------------------------------ #


class TestFPEPSInit:
    """Tests for fPEPS site tensor initialization."""

    def test_returns_symmetric_tensor(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert isinstance(A, SymmetricTensor)

    def test_ndim_is_5(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert A.ndim == 5

    def test_labels(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        assert A.labels() == ("u", "d", "l", "r", "phys")

    def test_physical_dim_is_2(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        phys_idx = A.indices[4]
        assert phys_idx.dim == 2

    def test_virtual_dim_matches_D(self, rng):
        for D in [2, 3, 4]:
            cfg = FPEPSConfig(D=D)
            A = _initialize_fpeps(cfg, rng)
            for i in range(4):  # u, d, l, r
                assert A.indices[i].dim == D

    def test_flows(self, default_config, rng):
        A = _initialize_fpeps(default_config, rng)
        flows = [idx.flow for idx in A.indices]
        assert flows == [
            FlowDirection.OUT,  # u
            FlowDirection.IN,  # d
            FlowDirection.OUT,  # l
            FlowDirection.IN,  # r
            FlowDirection.IN,  # phys
        ]

    def test_different_keys_give_different_tensors(self, default_config):
        A1 = _initialize_fpeps(default_config, jax.random.PRNGKey(0))
        A2 = _initialize_fpeps(default_config, jax.random.PRNGKey(1))
        d1 = A1.todense()
        d2 = A2.todense()
        assert not jnp.allclose(d1, d2)


# ------------------------------------------------------------------ #
# Task 6: _fpeps_simple_update                                         #
# ------------------------------------------------------------------ #


class TestFPEPSSimpleUpdate:
    """Tests for the full simple update loop."""

    def test_simple_update_runs(self):
        """5 steps should complete without error."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(0)
        A = _initialize_fpeps(cfg, key)
        H = spinless_fermion_gate(cfg)
        # #878: 2-site checkerboard, so both sublattices are returned, and
        # #851: all four bond spectra, as a BondWeights.
        A_opt, B_opt, lambdas = _fpeps_simple_update(
            A, H, max_D=cfg.D, dt=cfg.dt, steps=5
        )
        assert isinstance(A_opt, SymmetricTensor)
        assert isinstance(B_opt, SymmetricTensor)
        assert jnp.all(jnp.isfinite(A_opt.todense()))
        assert set(lambdas._fields) == {"h_AB", "h_BA", "v_AB", "v_BA"}

    def test_simple_update_changes_tensor(self):
        """20 steps of imaginary time evolution should change A."""
        cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.01)
        key = jax.random.PRNGKey(0)
        A = _initialize_fpeps(cfg, key)
        A_before = A.todense()
        H = spinless_fermion_gate(cfg)
        # #878: 2-site checkerboard, so both sublattices are returned.
        A_opt, B_opt, _lambdas = _fpeps_simple_update(
            A, H, max_D=cfg.D, dt=cfg.dt, steps=20
        )
        A_after = A_opt.todense()
        assert not jnp.allclose(A_before, A_after)

    @pytest.mark.parametrize("D", [3, 4])
    def test_simple_update_preserves_bond_layout_at_Dgt2(self, D):
        """Regression for #558.

        At D>2 the SU's truncated_svd used global democratic truncation, which
        let the new `r` bond drift from the canonical {even: D/2, odd: D/2}
        block layout. On the next step the contract(A_left, B_right) on the
        unchanged `l` axis vs the new `r` axis triggered an opt_einsum size
        mismatch. Per-sector keep allocation (via base_charges) preserves the
        canonical layout and lets SU continue across many steps.
        """
        cfg = FPEPSConfig(D=D, t=1.0, V=0.0, dt=0.01)
        A = _initialize_fpeps(cfg, jax.random.PRNGKey(0))
        H = spinless_fermion_gate(cfg)
        A_opt, B_opt, _ = _fpeps_simple_update(A, H, max_D=cfg.D, dt=cfg.dt, steps=5)
        # #558's contract holds for BOTH sublattices now, not just one.
        for name, site in (("A", A_opt), ("B", B_opt)):
            for axis_label in ("u", "d", "l", "r"):
                ax = site.labels().index(axis_label)
                idx = site.indices[ax]
                assert idx.dim == D, f"{name}.{axis_label} dim {idx.dim} != D={D}"
                charges = sorted(int(c) for c in idx.charges)
                expected = sorted([i % 2 for i in range(D)])
                assert charges == expected, (
                    f"{name}.{axis_label} charges {charges} != canonical {expected}"
                )
            assert jnp.all(jnp.isfinite(site.todense()))


class TestFPEPS2SiteSimpleUpdate:
    """Tests for the 2-site bipartite simple update on FermionParity sites.

    The 2-site SU (``_simple_update_2site_horizontal_tensor`` /
    ``_simple_update_2site_vertical_tensor``) is polymorphic over the Tensor
    protocol, so it accepts FermionParity inputs. At D>2 the truncated SVD
    inside must preserve the canonical ``[i % 2 for i in range(D)]`` bond
    layout (#563); otherwise the new bond drifts and the next SU step's
    contractions trip on per-axis charge-order mismatches (same root cause
    as #558/#559 in the 1-site path).
    """

    @staticmethod
    def _make_fermionic_AB(D, key, d=2):
        sym = FermionParity()
        virt = np.array([i % 2 for i in range(D)], dtype=np.int32)
        phys = np.array([0, 1], dtype=np.int32)

        def _indices():
            return (
                TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="u"),
                TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="d"),
                TensorIndex.from_charges(sym, virt, FlowDirection.OUT, label="l"),
                TensorIndex.from_charges(sym, virt, FlowDirection.IN, label="r"),
                TensorIndex.from_charges(sym, phys, FlowDirection.IN, label="phys"),
            )

        kA, kB = jax.random.split(key)
        A = SymmetricTensor.random_normal(_indices(), kA)
        B = SymmetricTensor.random_normal(_indices(), kB)
        return A, B

    @pytest.mark.parametrize("D", [3, 4])
    def test_2site_simple_update_preserves_bond_layout_at_Dgt2(self, D):
        """20 alternating H/V 2-site SU steps at V=1 must keep the exact
        canonical position-order ``[i % 2 for i in range(D)]`` on every
        virtual axis of both A and B (#563).

        The pre-fix code emits new-bond charges in global SV-magnitude
        order, which (a) can keep different sector counts than canonical
        and (b) even when counts match, scrambles position-order so
        downstream ``scale_bond_axis(lam, axis)`` slices the wrong sectors.
        Exact position-order matching is the discriminating check (sorted
        counts alone can coincide pre-fix at low step counts).
        """
        from tenax.algorithms.ipeps_simple_update import (
            _simple_update_2site_horizontal_tensor,
            _simple_update_2site_vertical_tensor,
        )

        cfg = FPEPSConfig(D=D, t=1.0, V=1.0, dt=0.01)
        A, B = self._make_fermionic_AB(D=D, key=jax.random.PRNGKey(0))
        H = spinless_fermion_gate(cfg)
        trotter = _trotter_gate(H, dt=cfg.dt)
        lam_h = jnp.ones(D)
        lam_v = jnp.ones(D)
        for step in range(20):
            if step % 2 == 0:
                A, B, lam_h = _simple_update_2site_horizontal_tensor(
                    A, B, trotter, lam_h, lam_v, D
                )
            else:
                A, B, lam_v = _simple_update_2site_vertical_tensor(
                    A, B, trotter, lam_h, lam_v, D
                )

        canonical = [i % 2 for i in range(D)]
        for name, T in (("A", A), ("B", B)):
            for axis_label in ("u", "d", "l", "r"):
                ax = T.labels().index(axis_label)
                idx = T.indices[ax]
                assert idx.dim == D, f"{name}.{axis_label} dim {idx.dim} != D={D}"
                charges = [int(c) for c in idx.charges]
                assert charges == canonical, (
                    f"{name}.{axis_label} position-order {charges} "
                    f"!= canonical {canonical}"
                )
            assert jnp.all(jnp.isfinite(T.todense()))


# ------------------------------------------------------------------ #
# Task 7: fpeps (entry point)                                          #
# ------------------------------------------------------------------ #


class TestFPEPS:
    """Tests for the fpeps entry point with CTM evaluation."""

    def test_fpeps_runs(self):
        """fpeps should return a finite energy."""
        cfg = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.01,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(cfg)
        key = jax.random.PRNGKey(99)
        # #878: 2-site checkerboard, so state and env are pairs.
        energy, (A_opt, B_opt), (env_A, env_B) = fpeps(H, cfg, key=key)
        assert jnp.isfinite(energy)

    def test_fpeps_returns_symmetric_tensor(self):
        """A_opt should be a SymmetricTensor."""
        cfg = FPEPSConfig(
            D=2,
            t=1.0,
            V=0.0,
            dt=0.01,
            num_imaginary_steps=5,
            ctm_chi=4,
            ctm_max_iter=10,
            ctm_conv_tol=1e-4,
        )
        H = spinless_fermion_gate(cfg)
        key = jax.random.PRNGKey(99)
        # #878: 2-site checkerboard, so state and env are pairs.
        energy, (A_opt, B_opt), (env_A, env_B) = fpeps(H, cfg, key=key)
        assert isinstance(A_opt, SymmetricTensor)
        assert isinstance(B_opt, SymmetricTensor)


# ------------------------------------------------------------------ #
# Task 8: Exports                                                      #
# ------------------------------------------------------------------ #


def test_fpeps_importable_from_tenax():
    """Public API should be importable from top-level tenax package."""
    from tenax import FPEPSConfig, fpeps, spinless_fermion_gate

    assert FPEPSConfig is not None
    assert fpeps is not None
    assert spinless_fermion_gate is not None


# --------------------------------------------------------------------------- #
# #997: a single sweep phase is exact at full rank, fermionic == bosonic       #
# --------------------------------------------------------------------------- #


def test_single_phase_full_rank_identity_matches_bosonic_control():
    """One bond update at max_D = D*d must reconstruct gate*theta, and the
    fermionic arm must do so exactly as well as the block-identical bosonic
    Z2 control (#997).

    Pre-fix, linalg's matricization Koszul signs (640 of 2560 fired in a
    5-step sweep) plus the factor transposes made a single production phase
    destroy the state to fidelity 0.634 (H) / 0.437 (V) while the Z2 retype
    of the same blocks scored 0.9999 -- the shared gate-rank truncation
    floor.  Asserting FP == Z2 to 1e-9 makes the test self-calibrating: the
    floor moves with the fixture, the equality does not.

    The mean-field symptom this pins down was an unphysical vacuum drain
    (<n>: 0.44 -> 1e-4 in 40 steps against a gate that is exact to 2e-16
    and amplifies occupation), which had been misread as seed-dependent
    state-model fragility (#878/#881, #882 SS5.3).
    """
    import tenax.algorithms.ipeps_simple_update as isu
    from tenax.algorithms.ipeps_simple_update import (
        _simple_update_2site_horizontal_tensor,
        _simple_update_2site_vertical_tensor,
    )
    from tenax.contraction.contractor import contract
    from tenax.core._tensor_utils import scale_bond_axis
    from tenax.core.symmetry import ZnSymmetry
    from tenax.core.tensor import _koszul_sign

    D = 3
    cfg = FPEPSConfig(D=D, dt=0.05, V=1.0)
    gate = _trotter_gate(spinless_fermion_gate(cfg), cfg.dt)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(1))
    B0 = _initialize_fpeps(cfg, jax.random.PRNGKey(5))
    rng = np.random.default_rng(3)
    lh, lv, lhf, lvo = (jnp.array(rng.uniform(0.3, 1, D)) for _ in range(4))

    def retype_z2(T):
        z2 = ZnSymmetry(2)
        idx = tuple(
            TensorIndex.from_charges(z2, i.charges, i.flow, label=i.label)
            for i in T.indices
        )
        return SymmetricTensor(dict(T.blocks), idx)

    def theta_h(A, B):
        Aa = scale_bond_axis(A, "u", lvo)
        Aa = scale_bond_axis(Aa, "d", lv)
        Aa = scale_bond_axis(Aa, "l", lhf)
        Aa = scale_bond_axis(Aa, "r", lh)
        Ba = scale_bond_axis(B, "u", lv)
        Ba = scale_bond_axis(Ba, "d", lvo)
        Ba = scale_bond_axis(Ba, "r", lhf)
        return contract(
            Aa.relabel("r", "shared"),
            Ba.relabels(
                {"u": "u_B", "d": "d_B", "l": "shared", "r": "r_B", "phys": "phys_B"}
            ),
        )

    def rebuild_h(A, B, ln):
        Aa = scale_bond_axis(A, "u", lvo)
        Aa = scale_bond_axis(Aa, "d", lv)
        Aa = scale_bond_axis(Aa, "l", lhf)
        Aa = scale_bond_axis(Aa, "r", ln)
        Ba = scale_bond_axis(B, "u", lv)
        Ba = scale_bond_axis(Ba, "d", lvo)
        Ba = scale_bond_axis(Ba, "r", lhf)
        return contract(
            Aa.relabel("r", "shared"),
            Ba.relabels(
                {"u": "u_B", "d": "d_B", "l": "shared", "r": "r_B", "phys": "phys_B"}
            ),
        )

    def fid(X, Y):
        ylab = [
            "si_out" if lab == "phys" else "sj_out" if lab == "phys_B" else lab
            for lab in Y.labels()
        ]
        xd = np.asarray(X.todense())
        yd = np.transpose(
            np.asarray(Y.todense()), tuple(ylab.index(lab) for lab in X.labels())
        )
        x, y = xd.ravel(), yd.ravel()
        return abs(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

    # The base_charges pin is fermionic-only, so with it active the two
    # arms run DIFFERENT truncations (per-sector keep counts vs global
    # top-k) and their fidelities differ at the pin's expense, not the
    # signs'.  Disable it so the arms are code-identical.  That is the
    # whole reason, and it is local to this comparison: this test drives
    # one bond update per arm and never reaches a CTM or an AD trace, so
    # nothing here depends on the bond layout holding still.
    #
    # It previously also cited "SS5.1 already established the pin is a
    # regularizer, not a structural need".  **That claim is retracted.**
    # Measured on the shipped ``fpeps()`` path (dt=0.05, 100 steps, 5
    # seeds), the pin is not a weak regulariser that helps a little -- it
    # is what drives the collapse:
    #
    #     V=0, pin ON    D=3 3/5 survive   D=4 3/5
    #     V=0, pin OFF   D=3 5/5           D=4 5/5
    #     V=2, pin ON    D=3 2/5           D=4 4/5
    #
    # Nor is "not a structural need" right in general: under a tracer the
    # global SV sort cannot run at all (``linalg.py`` dispatches to a
    # static per-sector allocation), and unpinning currently breaks the
    # 2x2 split-CTM, which cannot contract a corner against an edge once
    # the layout goes direction-dependent (#1024).  See #878.
    orig_pin = isu._truncation_base_charges
    isu._truncation_base_charges = lambda A, leg: None
    try:
        results = {}
        for name, A, B, G in (
            ("FP", A0, B0, gate),
            ("Z2", retype_z2(A0), retype_z2(B0), retype_z2(gate)),
        ):
            theta = theta_h(A, B)
            gated = contract(theta.relabel("phys", "si").relabel("phys_B", "sj"), G)
            if name == "FP":
                # Regime: the SVD split of THIS theta must braid odd past
                # odd, or the decomposition applies no sign and the
                # arm-equality asserts nothing (#997's first reproducer
                # passed that way).
                split = ("u", "d", "l", "si_out", "u_B", "d_B", "r_B", "sj_out")
                perm = tuple(gated.labels().index(lab) for lab in split)
                fp = gated.indices[0].symmetry
                signs = [
                    _koszul_sign(
                        tuple(int(fp.parity(np.array([q]))[0]) for q in k), perm
                    )
                    for k in gated.blocks
                ]
                assert -1 in signs, "fixture out of regime: never braids"
            An, Bn, ln = _simple_update_2site_horizontal_tensor(
                A, B, G, lh, lv, 6, lam_h_far=lhf, lam_v_other=lvo
            )
            results[name] = fid(gated, rebuild_h(An, Bn, ln))

        assert results["FP"] > 0.999, f"fermionic phase not exact: {results['FP']}"
        np.testing.assert_allclose(
            results["FP"],
            results["Z2"],
            atol=1e-9,
            err_msg="fermionic single-phase fidelity differs from the "
            "block-identical bosonic control (#997)",
        )

        # Vertical phase, same contract: kills a factor-transpose regression
        # on the path the horizontal test does not touch (U at (0,4,1,2,3)).
        # Full reconstruction fidelity, NOT a spectra comparison -- Koszul
        # signs cancel in every spectrum (SS5.2a), so spectra are blind to
        # exactly the defect class this test exists for.
        def theta_v(A, B, lv_shared):
            Aa = scale_bond_axis(A, "u", lvo)
            Aa = scale_bond_axis(Aa, "l", lhf)
            Aa = scale_bond_axis(Aa, "r", lh)
            Aa = scale_bond_axis(Aa, "d", lv_shared)
            Ba = scale_bond_axis(B, "d", lvo)
            Ba = scale_bond_axis(Ba, "l", lh)
            Ba = scale_bond_axis(Ba, "r", lhf)
            return contract(
                Aa.relabel("d", "shared"),
                Ba.relabels(
                    {
                        "u": "shared",
                        "d": "d_B",
                        "l": "l_B",
                        "r": "r_B",
                        "phys": "phys_B",
                    }
                ),
            )

        vres = {}
        for name, A, B, G in (
            ("FP", A0, B0, gate),
            ("Z2", retype_z2(A0), retype_z2(B0), retype_z2(gate)),
        ):
            gated = contract(
                theta_v(A, B, lv).relabel("phys", "si").relabel("phys_B", "sj"), G
            )
            An, Bn, lnv = _simple_update_2site_vertical_tensor(
                A, B, G, lh, lv, 6, lam_v_far=lvo, lam_h_other=lhf
            )
            vres[name] = fid(gated, theta_v(An, Bn, lnv))
        assert vres["FP"] > 0.999, f"vertical phase not exact: {vres['FP']}"
        np.testing.assert_allclose(
            vres["FP"],
            vres["Z2"],
            atol=1e-9,
            err_msg="vertical single-phase fidelity differs from the "
            "block-identical bosonic control (#997)",
        )
    finally:
        isu._truncation_base_charges = orig_pin
