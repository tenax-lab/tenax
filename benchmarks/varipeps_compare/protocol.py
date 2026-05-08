"""Single source of truth for the Tenax↔variPEPS benchmark protocol.

Both runners (``run_tenax.py``, ``run_varipeps.py``) and the orchestrator
(``compare.py``) import constants from this module.  Do not redefine knobs
locally — change them here.
"""

from __future__ import annotations

PATHS = ("single_site", "bipartite_2site")
D_VALUES = (2, 3)
CHI_VALUES = (16, 24)

GRID = tuple(
    {"path": p, "D": D, "chi": chi}
    for p in PATHS
    for D in D_VALUES
    for chi in CHI_VALUES
)  # 8 points

TOL = 1e-3
MAX_STEPS = 30
SEED = 0
DTYPE = "complex128"
# Note: TOL/MAX_STEPS lowered from 1e-6 / 100 after the first dry run showed
# Tenax's implicit-AD path on CPU could not complete a single point in the
# 30-min subprocess budget at the original protocol.  variPEPS converged in
# ~23 steps to E=-0.6625 on single_site D=2 chi=16 (E_ref=-0.6614), so the
# looser TOL still yields a meaningful comparison.  Tighten back to 1e-6
# when running on GPU.

LBFGS_HISTORY = 10  # both libs use L-BFGS with history depth 10
CTM_TOL = 1e-8
CTM_MAX_ITER = 100

SUBPROCESS_TIMEOUT_SEC = 30 * 60  # 30 min per (path, D, chi, lib)


def grid_key(path: str, D: int, chi: int) -> str:
    """Canonical filesystem key for a grid point."""
    return f"{path}_D{D}_chi{chi}"
