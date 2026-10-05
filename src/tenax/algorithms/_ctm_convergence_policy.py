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
    # Signed multiplier of the gauged step on the last two sweeps (#1060):
    # near -1 flags a two-state cycle (retry with ``mixing > 0``), 0 < rho < 1
    # a slow contraction.  NaN when not measurable (``conv_method="sv"``).
    step_multiplier: float = float("nan")


class GradientForwardInfo(NamedTuple):
    """What site 1 (the implicit-AD gradient forward) knows about its forward.

    ``iterations`` / ``sv_diff`` / ``best_iteration`` are the CTM loop's own
    (``sv_diff`` is the number it compared with ``conv_tol``).  The #841
    one-sweep stationarity residual is a different quantity with a different
    threshold, so it travels in its own fields and is printed under its own
    label.
    """

    converged: bool
    iterations: int
    sv_diff: float
    best_iteration: int = 0
    stationarity_residual: float | None = None
    stationarity_threshold: float | None = None
    # Signed step multiplier of the forward loop (#1060); NaN when unknown.
    step_multiplier: float = float("nan")


def _plateau_bailed(info) -> bool:
    """True when the loop stopped on the ``plateau_patience`` bail: it hands
    back the best-metric env, whose sweep index trails the sweep count.  On
    budget exhaustion the two are equal; 0 means the caller did not say."""
    if bool(getattr(info, "converged", False)):
        return False
    try:
        best = int(getattr(info, "best_iteration", 0) or 0)
        total = int(getattr(info, "iterations", 0) or 0)
    except (TypeError, ValueError):
        return False
    return 0 < best < total


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
    msg = (
        f"CTM forward did not converge ({where}): sweeps "
        f"{_fmt(getattr(info, 'iterations', None), 'd')}, sv_diff "
        f"{_fmt(getattr(info, 'sv_diff', None), '.3g')} vs conv_tol "
        f"{_fmt(conv_tol, 'g')}, chi={_fmt(chi, 'd')}, step multiplier "
        f"{format_step_multiplier(info)}"
    )
    residual = getattr(info, "stationarity_residual", None)
    if residual is not None:
        msg += (
            f", stationarity residual {_fmt(residual, '.3g')} (#841 threshold "
            f"{_fmt(getattr(info, 'stationarity_threshold', None), '.3g')})"
        )
    msg += (
        ". An unconverged environment invalidates the implicit-AD gradient and "
        "can report a sub-variational energy."
    )
    if _plateau_bailed(info):
        msg += (
            f" The loop stopped on the plateau bail (no sv_diff improvement for "
            f"plateau_patience sweeps after sweep {info.best_iteration}), not "
            f"on max_iter, so raising max_iter alone will not help: try a "
            f"larger CTMConfig(plateau_patience=...) (None disables the bail) "
            f"or a different chi."
        )
    return msg + " Set CTMConfig(on_unconverged='warn') to continue anyway."


class CTMNotConvergedError(RuntimeError):
    """A CTM forward feeding a gradient or a reported energy did not converge.

    Raised under ``CTMConfig(on_unconverged="raise")`` (the default) by
    ``optimize_gs_ad`` on the fused-CTM implicit-AD 1x1 and 2-site paths, and
    by ``ctm_tensor_2site(..., strict=True)`` (#1059).  The optimizer first
    tries to recover by resetting to the best params; it re-raises (after
    writing ``ckpt.last.pkl`` when ``gs_checkpoint_path`` is set) when it
    cannot.

    Attributes:
        info: The forward's convergence info (``converged``, ``iterations``,
            ``sv_diff`` and, when measured, ``step_multiplier`` and the #841
            stationarity residual).
        site: Where the forward was used: ``"gradient"``, ``"final_energy"``
            or ``"ctm_tensor_2site"``.
        step: The optimizer step, or None outside the optimizer loop.

    The message gives the sweep count, ``sv_diff`` against ``conv_tol``, the
    step multiplier (near -1: a two-state cycle that ``ctm_mixing`` cures;
    near +1: slow convergence) and, after a plateau bail, that raising
    ``max_iter`` alone will not help.
    """

    def __init__(
        self,
        info,
        site: str,
        step: int | None = None,
        *,
        conv_tol: float | None = None,
        chi: int | None = None,
        detail: str | None = None,
    ):
        self.info = info
        self.site = site
        self.step = step
        msg = _describe(info, site, step, conv_tol, chi)
        # ``detail``: the diagnostic a strict caller would otherwise have been
        # warned with (blind corners, hold-test verdicts), so raising loses
        # none of it.
        super().__init__(f"{msg}\n{detail}" if detail else msg)


class CTMNotConvergedWarning(UserWarning):
    """Emitted instead of ``CTMNotConvergedError`` under
    ``CTMConfig(on_unconverged="warn")``, which keeps the pre-#1059 control
    flow: the unconverged forward is used anyway, except that a warm-start
    env refresh that did not converge still never replaces a cached env.
    """


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
