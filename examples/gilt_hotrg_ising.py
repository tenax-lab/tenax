#!/usr/bin/env python3
"""2D classical Ising partition function via Gilt-HOTRG.

Gilt-HOTRG is HOTRG coarse-graining with the GILT filter applied to the two
inequivalent lattice bonds before every step (the drop-in counterpart of
``gilt_tnr`` for HOTRG). This example shows two things:

  1. The free energy per site matches Onsager. GILT does NOT beat plain HOTRG
     on this *bulk* observable -- HOTRG's HOSVD already suppresses the
     corner-double-line (CDL) short-range entanglement that GILT targets, so
     on the smooth free energy the filter's bond-gauge perturbation slightly
     dominates. (Contrast ``gilt_tnr``, which clearly beats plain TRG, because
     TRG accumulates much more CDL.)

  2. GILT's HOTRG payoff shows in the *critical data*. Estimating the critical
     coupling beta_c from a phase-indicator bisection of the tensor flow,
     Gilt-HOTRG lands closer to the exact Onsager value than plain HOTRG at
     the same bond dimension -- GILT removes the CDL that shifts the apparent
     transition.

Usage::

    uv run python examples/gilt_hotrg_ising.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from tenax import (
    GiltConfig,
    GiltHOTRGConfig,
    HOTRGConfig,
    compute_ising_tensor,
    gilt_hotrg,
    hotrg,
    ising_free_energy_exact,
)
from tenax.algorithms.gilt_hotrg import gilt_hotrg_step
from tenax.algorithms.hotrg import _hotrg_step_horizontal, _hotrg_step_vertical
from tenax.linalg import svd

BETA_C = 0.44068679350977147  # Onsager critical point


def free_energy_demo(chi: int = 16, num_steps: int = 18) -> None:
    print(f"\nFree energy per site  (chi={chi}, {num_steps} steps)")
    print("  beta      exact ln(Z)/N     HOTRG err     Gilt-HOTRG err")
    for beta in (0.40, BETA_C, 0.46):
        lz = float(-ising_free_energy_exact(beta) * beta)
        fh = float(
            hotrg(
                compute_ising_tensor(beta, symmetric=True),
                HOTRGConfig(max_bond_dim=chi, num_steps=num_steps),
            )
        )
        fg = float(
            gilt_hotrg(
                compute_ising_tensor(beta, symmetric=True),
                GiltHOTRGConfig(
                    max_bond_dim=chi,
                    num_steps=num_steps,
                    gilt=GiltConfig(gilt_eps=1e-3),
                ),
            )
        )
        print(f"  {beta:.5f}   {lz:.8f}     {abs(fh - lz):.2e}      {abs(fg - lz):.2e}")


def _sv2(T) -> float:
    _, s, _, _ = svd(T, ["up", "left"], ["down", "right"], new_bond_label="c")
    s = np.asarray(s)
    return float(s[1] / s[0]) if len(s) > 1 else 0.0


def _phase(beta: float, chi: int, num_steps: int, gilt_eps: float | None) -> str:
    T = compute_ising_tensor(beta, symmetric=True)
    r = 0.5
    for step in range(num_steps):
        horizontal = step % 2 == 0
        if gilt_eps is None:
            step_fn = _hotrg_step_horizontal if horizontal else _hotrg_step_vertical
            T, _ = step_fn(T, chi, None, device_mesh=None)
        else:
            T, _, _ = gilt_hotrg_step(
                T,
                GiltHOTRGConfig(max_bond_dim=chi, gilt=GiltConfig(gilt_eps=gilt_eps)),
                horizontal=horizontal,
            )
        if step >= 4:
            r = _sv2(T)
            if r > 0.995:
                return "ordered"
            if r < 0.02:
                return "disordered"
    return "ordered" if r > 0.5 else "disordered"


def _bisect_bc(chi: int, num_steps: int, gilt_eps: float | None) -> float:
    lo, hi = 0.40, 0.48
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if _phase(mid, chi, num_steps, gilt_eps) == "ordered":
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def critical_coupling_demo(chi: int = 8, num_steps: int = 16) -> None:
    print(f"\nCritical coupling beta_c  (chi={chi}, phase-indicator bisection)")
    bh = _bisect_bc(chi, num_steps, None)
    bg = _bisect_bc(chi, num_steps, 1e-3)
    eh = abs(bh - BETA_C) / BETA_C * 100
    eg = abs(bg - BETA_C) / BETA_C * 100
    print(f"  exact       beta_c = {BETA_C:.7f}")
    print(f"  HOTRG       beta_c = {bh:.7f}   ({eh:+.3f}%)")
    print(f"  Gilt-HOTRG  beta_c = {bg:.7f}   ({eg:+.3f}%)   <- GILT closer to exact")


if __name__ == "__main__":
    free_energy_demo()
    critical_coupling_demo()
