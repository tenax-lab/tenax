"""Tier-1 contract test for the 2x2 plaquette CTM projector — verifies that
at the saved D=4 chi=16 AD-optimum, the multisite-CTM energy E/site
matches variPEPS's gate API (-0.2554) instead of the 1x1 recipe's -0.913.

Diagnosis: see ~/.claude/projects/-home-yjkao-tenax/memory/project_c3_floor_breach_smoking_gun.md.
Probe scripts that produced the reference numbers:
    examples/dev/p1_tenax_chi_scan.py (1x1 -0.913)
    examples/dev/p2_varipeps_chi_scan.py (variPEPS -0.255)
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._pess_multisite_energy import kagome_xxz_pair_hamiltonian
from tenax.algorithms.pess import IPESSState


@pytest.mark.slow
def test_kagome_3site_multisite_2x2_at_d4_ad_optimum_matches_varipeps():
    """At the saved AD-optimum, 2x2 multisite-CTM energy ≈ variPEPS -0.2554/site.

    With the 1x1 recipe (the C.3 bug), this same state gives -0.913/site —
    a 0.66/site gap. Switching to the 2x2 plaquette projector should recover
    the physical (above-floor) energy variPEPS reports."""
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
    E_per_site = sum(bond_E.values()) / 3.0

    target = -0.255359  # variPEPS gate API at chi=16
    diff = E_per_site - target
    msg = (
        f"E/site = {E_per_site:.6f}, target {target:.6f}, "
        f"diff {diff:+.4f}\nbond Es: {bond_E}"
    )
    assert abs(diff) < 1e-3, msg
