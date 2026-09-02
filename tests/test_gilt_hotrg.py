"""Tests for the Gilt-HOTRG module (GILT bond filtering + HOTRG)."""

import numpy as np
import pytest

from tenax.algorithms.gilt import GiltConfig
from tenax.algorithms.gilt_hotrg import (
    GiltHOTRGConfig,
    gilt_hotrg,
    gilt_hotrg_step,
)
from tenax.algorithms.hotrg import HOTRGConfig, hotrg
from tenax.algorithms.trg import (
    compute_ising_tensor,
    ising_free_energy_exact,
)
from tenax.core.index import FlowDirection
from tenax.core.tensor import SymmetricTensor

BETA_C = 0.44068679350977147  # Onsager critical point


class TestGiltHOTRGConfig:
    def test_default_values(self):
        cfg = GiltHOTRGConfig()
        assert cfg.max_bond_dim == 16
        assert cfg.num_steps == 10
        assert cfg.direction_order == "alternating"
        assert isinstance(cfg.gilt, GiltConfig)

    def test_invalid_direction_rejected(self):
        T = compute_ising_tensor(beta=0.4, symmetric=True)
        with pytest.raises(ValueError):
            gilt_hotrg(T, GiltHOTRGConfig(direction_order="diagonal"))

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError):
            gilt_hotrg(np.zeros((2, 2, 2, 2)), GiltHOTRGConfig())


class TestGiltHOTRGStep:
    def test_step_returns_labeled_tensor(self):
        T = compute_ising_tensor(beta=0.4, symmetric=True)
        cfg = GiltHOTRGConfig(max_bond_dim=8, gilt=GiltConfig(gilt_eps=1e-3))
        T_new, log_norm, info = gilt_hotrg_step(T, cfg, horizontal=True)
        assert set(T_new.labels()) == {"up", "down", "left", "right"}
        assert np.isfinite(float(log_norm))
        assert "filtered_bond_dims" in info

    def test_step_preserves_flows(self):
        T = compute_ising_tensor(beta=0.4, symmetric=True)
        cfg = GiltHOTRGConfig(max_bond_dim=8, gilt=GiltConfig(gilt_eps=1e-3))
        T_new, _, _ = gilt_hotrg_step(T, cfg, horizontal=True)
        flows = {idx.label: idx.flow for idx in T_new.indices}
        assert flows["up"] == FlowDirection.IN
        assert flows["down"] == FlowDirection.OUT
        assert flows["left"] == FlowDirection.IN
        assert flows["right"] == FlowDirection.OUT

    def test_symmetric_stays_symmetric(self):
        T = compute_ising_tensor(beta=BETA_C, symmetric=True)
        cfg = GiltHOTRGConfig(max_bond_dim=8, gilt=GiltConfig(gilt_eps=1e-3))
        T_new, _, _ = gilt_hotrg_step(T, cfg, horizontal=True)
        assert isinstance(T_new, SymmetricTensor)


class TestGiltHOTRGDropIn:
    """gilt_eps == 0.0 must recover plain HOTRG bit-for-bit: the GILT
    filter is a no-op, so gilt_hotrg is an exact drop-in replacement."""

    def test_eps_zero_equals_plain_hotrg(self):
        beta, chi, ns = BETA_C, 16, 12
        T1 = compute_ising_tensor(beta, symmetric=True)
        f_gilt = float(
            gilt_hotrg(
                T1,
                GiltHOTRGConfig(
                    max_bond_dim=chi, num_steps=ns, gilt=GiltConfig(gilt_eps=0.0)
                ),
            )
        )
        T2 = compute_ising_tensor(beta, symmetric=True)
        f_hotrg = float(hotrg(T2, HOTRGConfig(max_bond_dim=chi, num_steps=ns)))
        assert f_gilt == pytest.approx(f_hotrg, abs=1e-12)


class TestGiltHOTRGFreeEnergy:
    """The free energy is correct (both HOTRG and Gilt-HOTRG match Onsager
    to ~1e-3). GILT is NOT expected to beat plain HOTRG on the bulk free
    energy -- HOTRG's HOSVD already suppresses the corner-double-line
    entanglement that GILT targets, so on this smooth observable the
    filter's bond-gauge perturbation dominates. GILT's HOTRG payoff shows
    in the critical data (see the example)."""

    def test_off_critical_accuracy(self):
        beta, chi, ns = 0.5, 16, 18  # deep ordered phase
        lz_exact = float(-ising_free_energy_exact(beta) * beta)
        T = compute_ising_tensor(beta, symmetric=True)
        cfg = GiltHOTRGConfig(
            max_bond_dim=chi, num_steps=ns, gilt=GiltConfig(gilt_eps=1e-3)
        )
        assert abs(float(gilt_hotrg(T, cfg)) - lz_exact) < 3e-3

    def test_filter_perturbation_shrinks_with_eps(self):
        beta, chi, ns = BETA_C, 16, 16
        lz_exact = float(-ising_free_energy_exact(beta) * beta)

        def err(eps):
            T = compute_ising_tensor(beta, symmetric=True)
            f = float(
                gilt_hotrg(
                    T,
                    GiltHOTRGConfig(
                        max_bond_dim=chi,
                        num_steps=ns,
                        gilt=GiltConfig(gilt_eps=eps),
                    ),
                )
            )
            return abs(f - lz_exact)

        # smaller eps -> weaker filter -> closer to plain HOTRG's error
        assert err(1e-4) <= err(1e-3) + 1e-9
