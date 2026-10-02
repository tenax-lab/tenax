"""What to do when a CTM forward reports ``converged=False`` (#1059, #1060).

The implicit-AD gradient is only valid at an element-wise fixed point, and an
unconverged environment can report an energy below the variational bound
(measured: spinless t-V V=2, E+V = -0.3024 vs ED -0.2918).  Two distinct
non-convergence modes exist (a period-2 cycle with step multiplier ~ -1 and a
slow wander), and no cheap pre-check predicts either, so every consumer of a
forward checks the flag.  ``CTMConfig.on_unconverged`` picks the response.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

_VALID_POLICIES = ("raise", "warn")


class CTMConvergeInfo(NamedTuple):
    """Convergence information from python_loop_ctm_converge.

    Lives here, in a module with no tenax imports, so every CTM driver can
    build one without an import cycle; ``_ctm_python_loop`` re-exports it.
    """

    converged: bool
    iterations: int  # CTM sweeps actually performed (#781)
    sv_diff: float
    max_truncation_error: float = 0.0  # variPEPS §2.8.2 indicator (last sweep)
    max_smallest_S: float = 0.0  # variPEPS norm_smallest_S indicator (#492)
    final_chi: int = 0  # final chi after any in-CTM bumps (#492); 0 ⇒ unchanged
    # Sweep index whose environment is returned.  Equals ``iterations``
    # except on the ``plateau_patience`` bail, where the best-metric env is
    # handed back and this trails ``iterations`` by ``plateau_patience``.
    # ``sv_diff`` is the metric of *this* sweep, not of ``iterations``.
    best_iteration: int = 0


def format_step_multiplier(info) -> str:
    """Signed slow-mode multiplier from #1061, or ``"n/a"``.

    ~ -1: flip (period-2) cycle, damping helps; ~ +1: slow monotone
    convergence, raise max_iter; otherwise a wander.
    """
    m = getattr(info, "step_multiplier", None)
    if m is None:
        return "n/a"
    try:
        m = float(m)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(m) else f"{m:.3g}"


def _fmt(x, spec: str) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(xf) or (isinstance(x, int) and x < 0):
        return "n/a"
    return format(x, spec)


def _describe(info, site, step, conv_tol, chi) -> str:
    where = f"site={site}" + (f", step {step}" if step is not None else "")
    return (
        f"CTM forward did not converge ({where}): sweeps "
        f"{_fmt(getattr(info, 'iterations', None), 'd')}, sv_diff "
        f"{_fmt(getattr(info, 'sv_diff', None), '.3g')} vs conv_tol "
        f"{_fmt(conv_tol, 'g')}, chi={_fmt(chi, 'd')}, step multiplier "
        f"{format_step_multiplier(info)}. An unconverged environment invalidates "
        f"the implicit-AD gradient and can report a sub-variational energy. Set "
        f"CTMConfig(on_unconverged='warn') to continue anyway."
    )


class CTMNotConvergedError(RuntimeError):
    """A CTM forward feeding a gradient or a reported energy did not converge."""

    def __init__(
        self,
        info,
        site: str,
        step: int | None = None,
        *,
        conv_tol: float | None = None,
        chi: int | None = None,
    ):
        self.info = info
        self.site = site
        self.step = step
        super().__init__(_describe(info, site, step, conv_tol, chi))


class CTMNotConvergedWarning(UserWarning):
    """Emitted instead of CTMNotConvergedError under on_unconverged='warn'."""


def check_ctm_converged(
    info,
    *,
    site: str,
    policy: str,
    step: int | None = None,
    conv_tol: float | None = None,
    chi: int | None = None,
) -> bool:
    """Return ``info.converged``; raise or warn per ``policy`` when False."""
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"on_unconverged must be one of {_VALID_POLICIES}, got {policy!r}"
        )
    if bool(getattr(info, "converged", False)):
        return True
    if policy == "raise":
        raise CTMNotConvergedError(info, site, step, conv_tol=conv_tol, chi=chi)
    warnings.warn(
        _describe(info, site, step, conv_tol, chi), CTMNotConvergedWarning, stacklevel=2
    )
    return False
