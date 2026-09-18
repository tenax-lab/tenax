"""#879: a magnitude anchor on the fermionic CTM energy path.

Before this, every fermionic energy assertion in the tree was a finiteness /
positivity / shape check -- ``compute_energy_split_ctm_tensor_2site`` (the path
``fpeps()`` actually returns) could return *any* finite real number and no test
objected.  That is exactly how #878 hid for months: ``fpeps()`` returned exactly
zero while every ``isfinite`` check passed on the corpse.

This pins the energy to a **known analytic value** on a **known state**, which
catches the "≈ 0" / "≈ −5e-5" failure mode a bound alone would miss.

The state is the fully-polarised checkerboard CDW (one sublattice occupied, the
other empty).  With the particle-hole chemical potential ``mu = 2V`` grafted on
(``-(mu/4)(n_i + n_j)`` per bond, the /4 splitting each site's ``-mu n`` across
its four NN bonds), the CDW's per-site energy is analytically **``-V``**: the
hopping term vanishes on a product state, the ``V n_i n_j`` interaction vanishes
because every NN bond joins an occupied to an empty site, and only the chemical
potential survives at ``-mu/2 = -V`` per site.

Two things are asserted, and they are complementary:

* the RDMs are **PSD** -- the #854 validity gate.  A fermionic energy is an
  expectation value ``tr(rho H)`` only when ``rho`` is a density matrix; on a
  non-PSD environment neither the manual trace nor the function is bounded by
  physics, so the anchor below would be meaningless.  Asserting PSD first turns
  the "trustworthy iff no #854 warning" rule (established across #996/#997/#998)
  into a standing check.
* the energy equals ``-V`` to a tight tolerance -- the magnitude anchor proper.

The fixture is a **seeded** CDW (#999): ``_fpeps_simple_update`` defaults ``B``
to ``A``, so from a symmetric start the CDW appears only by spontaneous symmetry
breaking, whose basin is a floating-point/JAX-context lottery -- the pair
collapses to a ~symmetric state on some builds and reads ``E ≈ 0``.  Biasing the
two sublattices' occupation oppositely seeds the breaking deterministically, so
the fixture is a CDW on every build.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
from tenax.algorithms._split_ctm_tensor_energy import (
    _rdm1x2_split_tensor_2site,
    _rdm2x1_split_tensor_2site,
    compute_energy_split_ctm_tensor_2site,
)
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_simple_update import _to_physical_pair
from tenax.core.tensor import SymmetricTensor

#: min-eigenvalue / spectral-radius floor for "this RDM is a density matrix".
#: Matches ``tenax.algorithms._ctm_diagnostics.RDM_PSD_TOL`` (the #854 gate).
_PSD_TOL = 1e-8


def _gate_with_mu(cfg: FPEPSConfig, mu: float) -> SymmetricTensor:
    """``spinless_fermion_gate`` plus a diagonal ``-(mu/4)(n_i + n_j)`` per bond.

    The added term is diagonal in the occupation basis, so it preserves the
    ``FermionParity`` block structure -- it is grafted straight onto the dense
    4x4 gate and re-wrapped with the same indices.
    """
    g = spinless_fermion_gate(cfg)
    h = np.array(g.todense()).reshape(4, 4)
    # basis order |00>, |01>, |10>, |11>  ->  n_i + n_j = 0, 1, 1, 2
    h = h + np.diag(-(mu / 4.0) * np.array([0.0, 1.0, 1.0, 2.0]))
    return SymmetricTensor.from_dense(jnp.array(h.reshape(2, 2, 2, 2)), g.indices)


def _seeded_cdw_pair(cfg: FPEPSConfig, gate: SymmetricTensor, steps: int):
    """A deterministically CDW-seeded, SU-evolved ``(A, B)`` checkerboard pair."""
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    # Strong (16x) opposite occupation bias on the two sublattices -- seeds the
    # symmetry breaking so the CDW is reproduced on every build (#999).
    a = scale_bond_axis(A0, "phys", jnp.array([4.0, 0.25]))
    b = scale_bond_axis(A0, "phys", jnp.array([0.25, 4.0]))
    A, B, lam = _fpeps_simple_update(a, gate, max_D=cfg.D, dt=cfg.dt, steps=steps, B=b)
    return _to_physical_pair(A, B, lam)


def _psd_margin(rdm) -> float:
    """Smallest eigenvalue over spectral radius; ``>= -_PSD_TOL`` means PSD."""
    m = np.array(rdm).reshape(4, 4)
    m = 0.5 * (m + m.conj().T)
    ev = np.linalg.eigvalsh(m)
    return float(ev.min() / max(abs(ev).max(), 1e-300))


def test_fermionic_ctm_energy_equals_the_cdw_analytic_value():
    """``compute_energy_split_ctm_tensor_2site`` on the fully-polarised CDW must
    read ``-V`` per site (mu = 2V), on a PSD environment.

    Mutation coverage: an energy contraction that dropped a bond, mis-normalised,
    or returned the ~0 of #878 fails ``E ≈ -V``; a sign error fails it at ``+V``;
    a collapsed / non-PSD environment fails the PSD assertion first (and would
    also miss ``-V``).
    """
    V = 4.0
    cfg = FPEPSConfig(D=2, t=1.0, V=V, dt=0.05)
    gate = _gate_with_mu(cfg, mu=2.0 * V)

    A, B = _seeded_cdw_pair(cfg, gate, steps=20)
    env_A, env_B = ctm_split_tensor_2site(A, B, chi=4, max_iter=30, conv_tol=1e-10)

    rdm_h = _rdm2x1_split_tensor_2site(A, B, env_A, env_B)
    rdm_v = _rdm1x2_split_tensor_2site(A, B, env_A, env_B)
    m_h, m_v = _psd_margin(rdm_h), _psd_margin(rdm_v)
    assert m_h >= -_PSD_TOL and m_v >= -_PSD_TOL, (
        f"the CDW environment is not PSD (rdm_h {m_h:.2e}, rdm_v {m_v:.2e} of "
        f"spectral radius below zero) -- the #854 gate; the energy below is not "
        f"an expectation value on a non-density-matrix, so the anchor is void"
    )

    E = float(compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2))
    assert E == pytest.approx(-V, abs=1e-3), (
        f"fermionic CTM energy of the fully-polarised CDW is {E:.8f}, not the "
        f"analytic -V = {-V} (mu = 2V).  A ~0 here is the #878 failure mode; a "
        f"+V is a sign error; anything else is a broken contraction/normalisation"
    )
