"""Tests for the GILT (Graph-Independent Local Truncation) module."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.gilt import (
    GiltConfig,
    GiltTNRConfig,
    gilt_plaquette,
    gilt_tnr,
    gilt_tnr_step,
)
from tenax.algorithms.trg import (
    TRGConfig,
    compute_free_wilson_fermion_tensor,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

BETA_C = 0.44068679350977147  # Onsager critical point


def _make_dense_tensor(arr: np.ndarray) -> DenseTensor:
    """Wrap a raw (d,d,d,d) array as a DenseTensor with TRG labels."""
    sym = U1Symmetry()
    d = arr.shape[0]
    charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="up"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="down"),
        TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="right"),
    )
    return DenseTensor(jnp.array(arr), indices)


def _cdl_tensor(crossed: bool = False) -> DenseTensor:
    """A pure corner-double-line tensor with chi = 4 (2 x 2 sublegs).

    Sub-leg convention: up/down legs carry (left, right) sublegs, left/right
    legs carry (up, down) sublegs. With ``crossed=True`` the corner matrices
    are wired diagonally across the site instead — a genuinely non-local
    (line-threading) correlation that GILT must NOT truncate.
    """
    rng = np.random.default_rng(0)
    m = rng.normal(size=(2, 2))
    m = m @ m.T + 2 * np.eye(2)  # positive definite corner matrix
    T = np.zeros((4, 4, 4, 4))
    for u1 in range(2):
        for u2 in range(2):
            for d1 in range(2):
                for d2 in range(2):
                    for l1 in range(2):
                        for l2 in range(2):
                            for r1 in range(2):
                                for r2 in range(2):
                                    if crossed:
                                        val = (
                                            m[l2, u1]
                                            * m[u2, r1]
                                            * m[r2, d1]
                                            * m[d2, l1]
                                        )
                                    else:
                                        val = (
                                            m[l1, u1]
                                            * m[u2, r1]
                                            * m[r2, d2]
                                            * m[l2, d1]
                                        )
                                    T[
                                        u1 * 2 + u2,
                                        d1 * 2 + d2,
                                        l1 * 2 + l2,
                                        r1 * 2 + r2,
                                    ] = val
    return _make_dense_tensor(T / np.abs(T).max())


class TestGiltConfig:
    def test_default_values(self):
        cfg = GiltConfig()
        assert cfg.gilt_eps == 1e-6
        assert cfg.convergence_eps == 1e-2
        assert cfg.max_laps == 4
        assert cfg.split_factor == 1e-3

    def test_tnr_config_defaults(self):
        cfg = GiltTNRConfig()
        assert cfg.max_bond_dim == 16
        assert cfg.num_steps == 10
        assert isinstance(cfg.gilt, GiltConfig)


class TestGiltPlaquette:
    def test_eps_zero_is_exact_noop(self):
        T = compute_ising_tensor(beta=0.4)
        B1, B2, info = gilt_plaquette(T, GiltConfig(gilt_eps=0.0))
        assert B1 is T
        assert B2 is T
        assert info["laps"] == []

    def test_cdl_collapse(self):
        """GILT must collapse a pure CDL tensor's bonds from 4 to 2."""
        T = _cdl_tensor()
        _, _, info = gilt_plaquette(T, GiltConfig(gilt_eps=1e-4))
        assert all(d == 2 for d in info["bond_dims"].values()), info

    def test_crossed_lines_not_truncated(self):
        """Diagonally-crossed line correlations are non-local: GILT must
        leave them alone (truncating them would change the network)."""
        T = _cdl_tensor(crossed=True)
        _, _, info = gilt_plaquette(T, GiltConfig(gilt_eps=1e-4))
        assert all(d == 4 for d in info["bond_dims"].values()), info

    def test_symmetric_stays_symmetric(self):
        T = compute_ising_tensor(beta=BETA_C, symmetric=True)
        B1, B2, _ = gilt_plaquette(T, GiltConfig(gilt_eps=1e-4))
        assert isinstance(B1, SymmetricTensor)
        assert isinstance(B2, SymmetricTensor)

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError):
            gilt_plaquette(np.zeros((2, 2, 2, 2)), GiltConfig())

    def test_rejects_fermionic(self):
        """The gram densify-and-transpose and the bar() double layer are
        bosonic-only; fermionic input must be rejected, not corrupted."""
        T = compute_free_wilson_fermion_tensor(mass=1.0)
        with pytest.raises(NotImplementedError):
            gilt_plaquette(T, GiltConfig(gilt_eps=1e-6))

    def test_symmetric_gram_never_densified(self):
        """The environment gram of a SymmetricTensor must be kept in the
        charge-difference-block representation (the full chi^4 dense array is
        never materialized); a DenseTensor keeps the whole matrix."""
        from tenax.algorithms.gilt import _bond_gram

        Ts = compute_ising_tensor(beta=BETA_C, symmetric=True)
        gram_sym = _bond_gram((Ts, Ts, Ts, Ts), "top")
        assert gram_sym.dense_full is None  # no chi^4 array formed
        assert gram_sym.blocks  # block-diagonal sectors present

        Td = compute_ising_tensor(beta=BETA_C, symmetric=False)
        gram_dense = _bond_gram((Td, Td, Td, Td), "top")
        assert gram_dense.dense_full is not None  # dense path keeps the matrix


class TestGiltTNRStep:
    def test_step_returns_normalized_tensor(self):
        T = compute_ising_tensor(beta=0.4)
        T_new, log_norm, info = gilt_tnr_step(T, GiltTNRConfig(max_bond_dim=8))
        assert set(T_new.labels()) == {"up", "down", "left", "right"}
        assert float(T_new.max_abs()) == pytest.approx(1.0)
        assert float(log_norm) > 0
        assert "laps" in info

    def test_flows_allow_iteration(self):
        """The output legs must come out in the library flow convention so
        the step composes with itself."""
        T = compute_ising_tensor(beta=0.4)
        cfg = GiltTNRConfig(max_bond_dim=6)
        for _ in range(3):
            T, _, _ = gilt_tnr_step(T, cfg)
        flows = {idx.label: idx.flow for idx in T.indices}
        assert flows["up"] == FlowDirection.IN
        assert flows["down"] == FlowDirection.OUT
        assert flows["left"] == FlowDirection.IN
        assert flows["right"] == FlowDirection.OUT


class TestGiltTNRFreeEnergy:
    def test_critical_ising_beats_plain_trg(self):
        """At the critical point Gilt-TNR must clearly beat plain TRG at
        equal bond dimension (the CDL plateau is what GILT removes)."""
        T = compute_ising_tensor(beta=BETA_C)
        lz_exact = float(-ising_free_energy_exact(BETA_C) * BETA_C)
        cfg = GiltTNRConfig(
            max_bond_dim=8, num_steps=20, gilt=GiltConfig(gilt_eps=1e-6)
        )
        err_gilt = abs(float(gilt_tnr(T, cfg)) - lz_exact)
        err_trg = abs(float(trg(T, TRGConfig(max_bond_dim=8, num_steps=20))) - lz_exact)
        # measured: gilt ~5.5e-5, plain ~2.1e-3 at chi=8
        assert err_gilt < 2e-4
        assert err_gilt < err_trg / 5

    def test_off_critical_accuracy(self):
        beta = 0.35
        T = compute_ising_tensor(beta=beta)
        lz_exact = float(-ising_free_energy_exact(beta) * beta)
        cfg = GiltTNRConfig(
            max_bond_dim=8, num_steps=20, gilt=GiltConfig(gilt_eps=1e-6)
        )
        # measured: 1.0e-5 at chi=8 (chi-truncation limited)
        assert abs(float(gilt_tnr(T, cfg)) - lz_exact) < 5e-5

    def test_symmetric_matches_dense(self):
        """The Z2 block-sparse path must reproduce the dense path."""
        Td = compute_ising_tensor(beta=BETA_C)
        Ts = compute_ising_tensor(beta=BETA_C, symmetric=True)
        cfg = GiltTNRConfig(
            max_bond_dim=8, num_steps=12, gilt=GiltConfig(gilt_eps=1e-6)
        )
        fd = float(gilt_tnr(Td, cfg))
        fs = float(gilt_tnr(Ts, cfg))
        assert abs(fd - fs) < 1e-10

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError):
            gilt_tnr(np.zeros((2, 2, 2, 2)), GiltTNRConfig())
