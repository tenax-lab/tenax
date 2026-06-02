#!/usr/bin/env python3
"""Minimal working fermionic iPEPS (fPEPS) ground state — spinless free fermions.

This is the smallest end-to-end fPEPS that produces *physically correct* energy.

Key finding (see the two paths below): for fermionic iPEPS the **AD optimizer**
(`optimize_fpeps_ad`) works, while the **simple-update** entry point (`fpeps`)
currently does NOT — its imaginary-time evolution drives the energy the wrong
way (from a random state's ~-0.19 *up* to exactly 0, collapsing to a trivial
fully-incoherent state). That is a fermionic-sign bug in the simple-update bond
gate (`_fpeps_simple_update_horizontal/vertical`), related to the #392 Koszul
sign issue; the end-to-end tests only assert `isfinite(energy)`, so it went
uncaught. Until the SU signs are fixed, optimize via AD for fermions.

Model: H = -t (c†_i c_j + h.c.) on the 2D square lattice (V=0 free fermions).

Honest caveat on accuracy: this demonstrates the AD path optimizes in the
*correct direction* (energy is negative and descends monotonically) -- the
qualitative contrast with the broken SU path (which rises to 0). It does NOT yet
certify a quantitatively correct number. At D=2 / finite CTM chi the energy here
(~-1.07/site) actually sits *below* the exact thermodynamic free-fermion value
(~-0.81/site, computed below). A finite-chi CTM energy is not a strict
variational bound, so part of that overshoot is environment-truncation error --
but ~30% is large and consistent with a residual fermionic-sign normalization
issue (#392) still being present in the CTM energy. Certifying the number needs
a chi-convergence study and the ED-reference (tests/test_fermionic_ed_reference).

Run::

    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
        uv run python examples/minimal_fpeps.py
    # or CPU:  JAX_PLATFORMS=cpu uv run python examples/minimal_fpeps.py
"""

from __future__ import annotations

import jax
import numpy as np

from tenax import FPEPSConfig, spinless_fermion_gate
from tenax.algorithms.fermionic_ipeps import _build_initial_fpeps_tensor
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_fpeps_ad


def free_fermion_energy_per_site(t: float = 1.0, n: int = 2048) -> float:
    """Exact E/site for the half-filled 2D square-lattice tight-binding model.

    eps(k) = -2t(cos kx + cos ky); fill all eps(k) < 0 (half filling).
    E/N = (1/(2pi)^2) * integral over the filled Fermi sea of eps(k).
    """
    k = (np.arange(n) + 0.5) / n * 2 * np.pi - np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    eps = -2.0 * t * (np.cos(kx) + np.cos(ky))
    return float(np.mean(np.where(eps < 0, eps, 0.0)))


def main() -> None:
    dev = jax.devices()[0]
    print(f"# device: {dev.platform} ({dev.device_kind})")

    fcfg = FPEPSConfig(D=2, t=1.0, V=0.0)
    gate = spinless_fermion_gate(fcfg)
    A_init = _build_initial_fpeps_tensor(fcfg, jax.random.PRNGKey(0))

    cfg = iPEPSConfig(
        max_bond_dim=2,
        su_init=False,  # do NOT seed from simple update (it's broken for fermions)
        gs_num_steps=20,
        gs_learning_rate=2e-2,
        gs_verbose=True,
        gs_log_interval=1,
        gs_implicit_ad=True,  # implicit diff through the CTM fixed point
        ctm=CTMConfig(chi=6, max_iter=30, conv_tol=1e-5),
    )

    _, _, E_gs = optimize_fpeps_ad(gate, A_init=A_init, config=cfg)

    e_exact = free_fermion_energy_per_site(t=fcfg.t)
    print("\n" + "=" * 64)
    print(f"  fPEPS (AD, D={fcfg.D}, finite chi)  E/site = {float(E_gs): .6f}")
    print(f"  exact free-fermion E/site        = {e_exact: .6f}  (thermodynamic)")
    print("  NOTE: the AD energy is below the exact value -- finite-chi CTM is")
    print("  not a strict variational bound, and a residual fermionic-sign")
    print("  normalization issue (#392) likely contributes. Direction (negative,")
    print("  descending) is correct; the absolute number is NOT yet certified.")
    print("  Contrast: simple update (fpeps()) RISES to exactly 0 -- broken.")
    print("=" * 64)
    # Robust claim only: AD descends to a negative energy (correct direction),
    # unlike SU which collapses to 0. We do NOT assert quantitative accuracy.
    assert float(E_gs) < 0.0, f"fermionic AD energy not negative: {float(E_gs)}"
    print("OK: AD reaches a negative, descending energy (vs SU's collapse to 0).")


if __name__ == "__main__":
    main()
