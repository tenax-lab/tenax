"""CTM environment health diagnostics (#723, #726, #746, #747).

A CTM environment can be finite, Hermitian, PSD and *completely wrong*.  The
``1x1`` corner-pair projector collapses the environment to rank-1 corners: its
``M = C1g^H C4g`` is ``chi x chi``, so the ``chi * D**2`` seam is summed away
and ``rank(P) <= rank(C1g)``, which is 1 at the cold seed.  Rank-1 is therefore
an absorbing state and the boundary is a ``chi_eff = 1`` mean-field object
rather than a corner transfer matrix.

Nothing raises when this happens, which is why it survived across four
benchmark campaigns (#747).  The two cheap detectors are here:

* :func:`ctm_corner_rank` / :func:`env_is_collapsed` -- the *direct* check, one
  SVD of a ``chi x chi`` corner.
* :func:`frozen_chi_pairs` -- the *indirect* check, for a chi scan: a genuine
  corner transfer matrix improves as the boundary grows, so an energy that is
  **bit-identical** across a change in chi is a broken environment, not a
  converged one.

Both are cheap enough to run unconditionally in benchmark drivers.
"""

from __future__ import annotations

__all__ = [
    "CollapsedEnvironmentError",
    "check_ctm_env",
    "ctm_corner_rank",
    "env_is_collapsed",
    "frozen_chi_pairs",
]

import warnings

import numpy as np


class CollapsedEnvironmentError(RuntimeError):
    """Raised by :func:`check_ctm_env` in ``strict`` mode on a rank-1 corner."""


def _corner_spectrum(env) -> np.ndarray:
    """Normalized singular values of the environment's ``C1`` corner.

    Works for both ``CTMTensorEnv`` and ``SplitCTMTensorEnv`` (both expose
    ``C1``), dense or block-sparse.  The corner is ``chi x chi``, so densifying
    it is cheap even when the rest of the environment is large -- this is not
    the ``todense()`` the symmetric-tensor guidance warns about.
    """
    C1 = getattr(env, "C1", None)
    if C1 is None:
        raise TypeError(
            f"expected a CTM environment exposing 'C1', got {type(env).__name__}"
        )
    M = np.asarray(C1.todense() if hasattr(C1, "todense") else C1)
    if M.ndim != 2:
        M = M.reshape(M.shape[0], -1)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.abs(s)
    top = s[0] if s.size and s[0] > 0 else 1.0
    return s / top


def ctm_corner_rank(env, tol: float = 1e-10) -> int:
    """Numerical rank of the environment's ``C1`` corner.

    Args:
        env: A ``CTMTensorEnv`` or ``SplitCTMTensorEnv``.
        tol: Relative singular-value cutoff, against the largest value.

    Returns:
        The number of singular values above ``tol`` (at least 1 for a
        non-zero corner).
    """
    return int((_corner_spectrum(env) > tol).sum())


def env_is_collapsed(env, tol: float = 1e-10) -> bool:
    """True when ``C1`` is rank-1, i.e. a ``chi_eff = 1`` product boundary.

    This is the #726 signature.  A rank-1 corner is not merely inaccurate --
    it carries no boundary entanglement at all, so the energy it produces is
    a mean-field number that does not respond to ``chi``.
    """
    return ctm_corner_rank(env, tol) <= 1


def check_ctm_env(
    env,
    *,
    context: str = "",
    tol: float = 1e-10,
    strict: bool = False,
) -> int:
    """Warn (or raise) when the environment has collapsed; return its rank.

    Intended to be called once after each CTM convergence in benchmark and
    production drivers, so a mean-field environment cannot be reported as a
    convergence success (#747).

    Args:
        env:     A ``CTMTensorEnv`` or ``SplitCTMTensorEnv``.
        context: Free-text label included in the message (e.g. ``"D=8 chi=384
                 split"``), so a sweep says *which* cell collapsed.
        tol:     Relative singular-value cutoff.
        strict:  Raise :class:`CollapsedEnvironmentError` instead of warning.

    Returns:
        The corner rank, so callers can record it alongside the energy.
    """
    rank = ctm_corner_rank(env, tol)
    if rank <= 1:
        where = f" [{context}]" if context else ""
        msg = (
            f"CTM environment has collapsed to a rank-{rank} corner{where}: "
            f"this is a chi_eff=1 mean-field boundary, not a corner transfer "
            f"matrix, and its energy will not respond to chi. The '1x1' "
            f"recipe does this by construction (#723, #726, #746); use "
            f"recipe='2x2' / gs_recipe='2x2'. See #747."
        )
        if strict:
            raise CollapsedEnvironmentError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return rank


def frozen_chi_pairs(chi_to_energy) -> list[tuple[int, int]]:
    """Find chi pairs whose energies are **bit-identical**.

    A converged CTM environment improves as ``chi`` grows, so two different
    ``chi`` returning the exact same float is the collapse signature rather
    than a sign of convergence -- on the D=8 split scan of #747 the energy was
    identical to 13 digits across ``chi`` 48 -> 384, which was read as clean
    convergence and was in fact a rank-1 boundary.

    Exact equality is deliberate: this is a *detector*, not a convergence
    criterion.

    .. warning::
        This check alone **can** false-positive.  A fully converged environment
        is also flat in chi -- that is what convergence means -- so a scan whose
        energy has saturated to the last bit will be reported here even though
        nothing is wrong.  The collapse signature is the *conjunction*: frozen
        in chi **and** :func:`env_is_collapsed`.  Use :func:`ctm_corner_rank`
        to disambiguate, which is why the drivers record ``corner_rank``
        alongside this rather than relying on either signal by itself.

        This is not hypothetical: a #723 regression test asserted the chi
        response on its own and was a platform coin flip, failing on macOS
        while reporting the correct converged energy.

    Args:
        chi_to_energy: Mapping ``{chi: energy}``, or an iterable of
                       ``(chi, energy)`` pairs.  ``None`` energies are skipped.

    Returns:
        Sorted ``(chi_lo, chi_hi)`` pairs that returned identical energies.
    """
    items = (
        list(chi_to_energy.items())
        if hasattr(chi_to_energy, "items")
        else list(chi_to_energy)
    )
    pts = sorted((int(c), float(e)) for c, e in items if e is not None)
    out = []
    for i, (c_lo, e_lo) in enumerate(pts):
        for c_hi, e_hi in pts[i + 1 :]:
            if e_lo == e_hi:
                out.append((c_lo, c_hi))
    return out
