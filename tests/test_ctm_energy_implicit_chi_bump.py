"""Implicit-AD + in-CTM chi-bump correctness tests (#516 chi-lock).

These tests verify that ctm_energy_implicit produces correct gradients
when ctmrg_heuristic_increase_chi=True forces the forward CTM to grow
chi mid-convergence.  See docs/plans/2026-05-20-chi-lock-design.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.core import DenseTensor, FlowDirection, TensorIndex, U1Symmetry


def _build_site_tensor(D: int = 2, d: int = 2, seed: int = 0) -> DenseTensor:
    """Build a small trivial-U(1) (D, d) single-site tensor for D=2 probes.

    Uses a trivial U(1) symmetry (all charges zero) wrapped in DenseTensor,
    matching the validation-shim pattern used by ``test_ctm_energy_implicit``.
    """
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, phys_charges.copy(), FlowDirection.IN, label="p"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
    )
    data = jnp.asarray(
        rng.standard_normal((d, D, D, D, D)).astype(np.float64), dtype=jnp.float64
    )
    return DenseTensor(data, indices)


def test_implicit_ad_no_longer_raises_with_bump():
    """ctm_energy_implicit(..., ctmrg_heuristic_increase_chi=True) returns a value.

    Was a NotImplementedError before chi-lock (#516); now should run.
    """
    site_tensors = {(0, 0): _build_site_tensor()}
    gate = heisenberg_gate()

    energy = ctm_energy_implicit(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        gate,
        chi=4,
        max_iter=4,
        ctmrg_heuristic_increase_chi=True,
        chi_max=8,
    )
    assert jnp.isfinite(energy)
