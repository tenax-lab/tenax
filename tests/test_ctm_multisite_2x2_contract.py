"""Tier-1 contract tests for the 2x2 plaquette CTM projector.

The legacy 1x1 multisite-CTM recipe lands on a fixed point that breaches the
spin-1/2 AFH per-bond floor (-0.439/site for kagome AFH at delta=1): on the
saved D=4 chi=16 AD-optimum, 1x1 reports E/site = -0.913 — far below any
physical state of the model.  See `project_c3_floor_breach_smoking_gun.md`
for the full diagnosis.

The 2x2 plaquette projector (variPEPS-style enlarged-corner truncation,
this branch's contribution) lifts the fixed point above the floor.  At the
saved AD-optimum, E/site = -0.092, well above -0.439.  However, this still
does not match variPEPS's gate-API reference of -0.255.  The residual gap
is documented as M2b honeycomb-native CTM follow-up — the saved AD-optimum
itself was tuned under the broken legacy energy_fn, so reaching variPEPS
parity also requires re-running AD optimisation under the new 2x2
energy_fn.

Reference numbers reproduced by:
    examples/dev/p1_tenax_chi_scan.py        (1x1 -0.913)
    examples/dev/p2_varipeps_chi_scan.py     (variPEPS -0.255)
    examples/dev/diag_gauge_2plaq_vs_1plaq.py (1x1 vs 2x2 single-site
                                               RDM diff = 0.40)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._pess_multisite_energy import kagome_xxz_pair_hamiltonian
from tenax.algorithms.pess import IPESSState

# Spin-1/2 AFH per-bond floor for the kagome 3-site multisite encoding
# (-0.5 * 2 * 6 / 3 + per-site corrections).  Conservative bound that the
# 1x1 recipe explicitly breaches at -0.913.
_SPIN_HALF_FLOOR = -0.439

# Energy reported by variPEPS's gate API at chi=16 on the same saved state.
# Target for the M2b honeycomb-native CTM follow-up.
_VARIPEPS_TARGET = -0.255359


def _energy_per_site_at_d4_ad_optimum() -> tuple[float, dict[str, float]]:
    """Run multisite-CTM at the saved D=4 chi=16 AD optimum, return E/site
    and per-bond energies."""
    npz_path = Path("logs/d4_ad_optimum.npz")
    if not npz_path.exists():
        pytest.skip(
            f"saved AD-optimum {npz_path} missing; "
            "regenerate via examples/dev/save_d4_ad_optimum.py"
        )

    npz = np.load(npz_path)
    state = IPESSState(
        R_a=jnp.asarray(npz["R_a"]),
        R_b=jnp.asarray(npz["R_b"]),
        R_c=jnp.asarray(npz["R_c"]),
        T_u=jnp.asarray(npz["T_u"]),
        T_d=jnp.asarray(npz["T_d"]),
        lambdas=tuple(jnp.asarray(npz[f"lambda_{i}"]) for i in range(6)),
    )
    H_pair = jnp.asarray(kagome_xxz_pair_hamiltonian(delta=1.0, d=2))

    # _collect_ctm_rdms lives in the existing diagnostic probe.
    sys.path.insert(0, str(Path("examples").resolve()))
    from kagome_pess_multisite_phase_c3_rdm_brute_force_diag import (  # type: ignore
        _collect_ctm_rdms,
    )

    rdms = _collect_ctm_rdms(state, chi=16, max_iter=120, conv_tol=1e-9)
    bonds = ("uv_h", "uv_v", "wu_h", "wu_v", "vw_row", "vw_col")
    bond_E = {
        b: float(complex(jnp.einsum("ijkl,ijkl->", rdms[b], H_pair)).real)
        for b in bonds
    }
    return sum(bond_E.values()) / 3.0, bond_E


@pytest.mark.slow
def test_kagome_3site_multisite_2x2_lifts_above_spin_half_floor():
    """At the saved AD-optimum, 2x2 multisite-CTM E/site is above the
    spin-1/2 AFH per-bond floor of -0.439.

    With the 1x1 recipe (the C.3 bug), this same state gives E/site = -0.913,
    breaching the floor.  Switching to the 2x2 plaquette projector lifts E/site
    to ~-0.092, well above -0.439."""
    e_per_site, bond_E = _energy_per_site_at_d4_ad_optimum()
    msg = f"E/site = {e_per_site:.6f}, floor {_SPIN_HALF_FLOOR}, bond Es: {bond_E}"
    assert e_per_site > _SPIN_HALF_FLOOR, msg
    # Also assert E is qualitatively negative (not zero / not the trivial
    # all-corners-truncated state).
    assert e_per_site < -0.05, msg


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason=(
        "2x2 multisite-CTM at the AD-optimum gives E/site ~ -0.092, "
        "vs variPEPS's -0.255.  The residual gap requires M2b honeycomb-"
        "native CTM + AD re-optimisation under the new 2x2 energy_fn."
    ),
)
def test_kagome_3site_multisite_2x2_at_d4_ad_optimum_matches_varipeps():
    """Aspirational: 2x2 multisite-CTM matches variPEPS's gate API
    (-0.2554/site) at the saved AD-optimum.

    Currently fails by ~0.16/site.  See xfail reason for the follow-up."""
    e_per_site, bond_E = _energy_per_site_at_d4_ad_optimum()
    diff = e_per_site - _VARIPEPS_TARGET
    msg = (
        f"E/site = {e_per_site:.6f}, target {_VARIPEPS_TARGET:.6f}, "
        f"diff {diff:+.4f}\nbond Es: {bond_E}"
    )
    assert abs(diff) < 1e-3, msg
