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

# Outer-loop stop for BOTH libraries: ||grad E||_2 < TOL.  Tenax runs
# gs_conv_criterion="grad_norm" with gs_grad_norm_tol=TOL; variPEPS sets
# optimizer_convergence_eps=TOL (it compares the L2 norm of the raw
# gradient, the same quantity).  1e-5 is variPEPS's own default, so the
# published variPEPS reference (run at that default) stays comparable.
TOL = 1e-5
MAX_STEPS = 100
SEED = 0
DTYPE = "complex128"

LBFGS_HISTORY = 10  # both libs use L-BFGS with history depth 10
CTM_TOL = 1e-8
CTM_MAX_ITER = 100

SUBPROCESS_TIMEOUT_SEC = 30 * 60  # 30 min per (path, D, chi, lib)


def grid_key(path: str, D: int, chi: int) -> str:
    """Canonical filesystem key for a grid point."""
    return f"{path}_D{D}_chi{chi}"
