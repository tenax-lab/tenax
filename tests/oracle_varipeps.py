"""variPEPS 2-site CTMRG oracle (test/diagnostic only). GPL reference — NEVER
import this from src/. Skips cleanly if variPEPS is not installed.

variPEPS is GPL-licensed. It is used here purely as a read-only numerical
oracle for cross-checking Tenax's dense 2-site CTMRG environment axis
conventions. It must never be imported from anything under ``src/``.
"""

from __future__ import annotations

import numpy as np


def varipeps_available() -> bool:
    """Return True iff variPEPS can be imported in this environment."""
    try:
        import varipeps  # noqa: F401

        return True
    except Exception:
        return False


def run_varipeps_2site_ctmrg(
    D: int = 2, d: int = 2, chi: int = 8, seed: int = 0
) -> dict:
    """Run dense 2-site checkerboard CTMRG and report converged env shapes.

    Returns a dict with the shapes of the eight environment tensors
    (``C1``..``C4`` corners, ``T1``..``T4`` edges) for unit-cell site 0,
    plus an ``_axis_note`` string documenting variPEPS' axis convention.

    Raises if variPEPS is unavailable (callers should guard with
    ``varipeps_available()``).
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    from varipeps.ctmrg.routine import calc_ctmrg_env
    from varipeps.peps import PEPS_Unit_Cell

    uc = PEPS_Unit_Cell.random(
        structure=[[0, 1], [1, 0]],
        d=d,
        D=D,
        chi=chi,
        max_chi=chi,
        seed=seed,
        dtype=np.complex128,
    )
    peps_arrays = [t.tensor for t in uc.get_unique_tensors()]
    # calc_ctmrg_env returns (converged_unitcell, norm_smallest_S)
    conv_uc = calc_ctmrg_env(peps_arrays, uc)[0]
    site0 = conv_uc[0, 0][0][0]

    shapes = {
        name: tuple(getattr(site0, name).shape)
        for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
    }
    shapes["_axis_note"] = (
        "variPEPS square-PEPS env convention (varipeps.peps.tensor.PEPS_Tensor): "
        "corners C1..C4 are rank-2 (chi, chi); edges are rank-4. "
        "T1 (top): (chi_left, D_ket_up, D_bra_up, chi_right); "
        "T2 (right): (D_ket_right, D_bra_right, chi_up, chi_down); "
        "T3 (bottom): (chi_left, chi_right, D_ket_down, D_bra_down); "
        "T4 (left): (chi_up, D_ket_left, D_bra_left, chi_down). "
        "PEPS tensor axes: (D_left, D_up, phys, D_right, D_down)."
    )
    return shapes


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    if not varipeps_available():
        print("variPEPS not available")
    else:
        for k, v in run_varipeps_2site_ctmrg().items():
            print(k, v)
