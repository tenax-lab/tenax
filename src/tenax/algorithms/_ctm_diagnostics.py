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
    "INVALID_RDM_DEFECT",
    "RDM_PSD_TOL",
    "RDM_TRACE_TOL",
    "CollapsedEnvironmentError",
    "CollapsedRDMError",
    "check_ctm_env",
    "check_rdm",
    "ctm_corner_rank",
    "env_is_collapsed",
    "frozen_chi_pairs",
    "rdm_trace_defect",
]

import warnings

import numpy as np


class CollapsedEnvironmentError(RuntimeError):
    """Raised by :func:`check_ctm_env` in ``strict`` mode on a rank-1 corner."""


class CollapsedRDMError(RuntimeError):
    """Raised by :func:`check_rdm` in ``strict`` mode on an invalid RDM.

    Covers all three failures it detects -- non-finite entries, a non-unit
    trace, and loss of positive semi-definiteness -- not the trace alone.
    """


#: Tolerance on ``|tr(rdm) - 1|``.  A trace-normalised RDM that carried any
#: physical content has a trace of *exactly* 1 -- ``_normalise_rdm`` divides by
#: the trace, and Hermitian symmetrisation preserves it -- so this only has to
#: absorb the rounding of that division, never a physical spread.
RDM_TRACE_TOL = 1e-8

#: Tolerance on how far below zero the smallest eigenvalue may sit, **relative
#: to the spectral radius**.
#:
#: Relative, and not an exact ``min_eig < 0``, for one reason: a converged RDM
#: is routinely rank-deficient -- a product state's two-site RDM has a single
#: non-zero eigenvalue -- and the zero eigenvalues land a few ulp either side
#: of zero.  An exact test would warn on every healthy run, and a guard that
#: cries wolf gets filtered out, which is how the two earlier holes in this
#: function survived.  Roundoff here is ~1e-16 of the radius, so 1e-8 leaves
#: eight orders of headroom while still catching the #853 case, whose smallest
#: eigenvalue was **0.8 of the spectral radius** below zero.
RDM_PSD_TOL = 1e-8

#: Returned by :func:`check_rdm` in place of the trace defect when the RDM
#: carries non-finite entries, or is not positive semi-definite.
#:
#: ``inf`` rather than ``nan``, deliberately.  The entire failure class in #848
#: is a tolerance comparison that fails open: ``nan > tol`` is ``False``, so
#: handing back ``nan`` invites a caller who gates on
#: ``check_rdm(...) > tol`` to repeat the bug one level up.  ``inf > tol`` is
#: ``True``, so that caller is fail-closed instead.
#:
#: :func:`rdm_trace_defect` is unaffected and keeps returning the honest
#: ``|tr - 1|``; this substitution belongs to the *validity check*, which is
#: reporting "not a usable measurement", not to the measurement itself.
INVALID_RDM_DEFECT = float("inf")


def _as_rdm_matrix(rdm) -> np.ndarray:
    """Densify an RDM to a NumPy array, accepting 2- or 4-leg forms.

    The RDM is ``d x d`` or ``d**2 x d**2`` -- a local operator, not a boundary
    object -- so ``todense()`` here is the sanctioned small-result case, not the
    one the symmetric-tensor guidance warns about.
    """
    M = np.asarray(rdm.todense() if hasattr(rdm, "todense") else rdm)
    if M.ndim not in (2, 4):
        raise ValueError(f"expected a 2- or 4-leg RDM, got ndim={M.ndim}")
    return M


def _rdm_negativity(M: np.ndarray) -> tuple[float, float]:
    """How far below zero the spectrum reaches, relative to its radius.

    Returns ``(negativity, min_eigenvalue)``.  ``negativity`` is
    ``max(0, -min_eig) / max|eig|``, so it is dimensionless and comparable
    against :data:`RDM_PSD_TOL` regardless of the RDM's scale.

    The matrix is symmetrised first.  That is not a repair, it is the correct
    quantity: for Hermitian ``H``, ``tr(rho H) = tr(Herm(rho) H)``, so the
    anti-Hermitian part contributes nothing to any energy and the bound
    ``<H> in [min eig H, max eig H]`` is controlled entirely by the spectrum of
    ``Herm(rho)``.  It also makes the test well defined on an input that is not
    Hermitian, where ``eigvalsh`` would otherwise silently report the spectrum
    of whichever triangle it happened to read.  On the energy path this is a
    no-op: ``_normalise_rdm`` returns ``0.5 * (mat + mat^H)``.

    A zero radius (the collapsed all-zero RDM) returns ``0.0`` -- vacuously
    PSD, which is why :func:`check_rdm` tests the trace first.
    """
    if M.ndim == 4:
        M = M.reshape(M.shape[0] * M.shape[1], M.shape[2] * M.shape[3])
    eig = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    if eig.size == 0:
        return 0.0, 0.0
    radius = float(np.max(np.abs(eig)))
    min_eig = float(eig.min())
    if radius == 0.0:
        return 0.0, min_eig
    return max(0.0, -min_eig) / radius, min_eig


def rdm_trace_defect(rdm) -> float:
    """``|tr(rdm) - 1|`` for a trace-normalised reduced density matrix.

    Accepts the ``(d, d)`` / ``(d*d, d*d)`` matrix form or the ``(d, d, d, d)``
    form the RDM builders return.

    The value is a clean two-state discriminator rather than a quality score.
    ``_normalise_rdm`` divides by the trace, so **any** RDM with physical
    content comes back with trace exactly 1 and defect 0.  A defect of 1 means
    the raw network contracted to zero and the normaliser's zero-matrix guard
    returned zeros -- correctly, since ``0/0`` is not a density matrix.

    .. warning::
        **A small defect does not certify the RDM.**  The trace is one number
        summed over the diagonal, so it says nothing about the rest of the
        array, and it fails in two independent ways:

        * An RDM carrying ``NaN`` or ``inf`` *off* the diagonal has a trace of
          exactly 1 and a defect of exactly 0, while the energy contracted
          from it is ``NaN`` (#848).
        * Being a *sum*, it lets negative eigenvalues cancel against positive
          ones inside itself.  A spectrum of ``[-1.2997, -0.0736, 0.7446,
          1.6287]`` traces to exactly 1 and reports defect 0, and the energy
          contracted from it is not bounded by the Hamiltonian (#854).

        Use :func:`check_rdm`, which tests finiteness over the whole array and
        positivity of the spectrum as well.
    """
    M = _as_rdm_matrix(rdm)
    tr = np.einsum("ijij->", M) if M.ndim == 4 else np.trace(M)
    return float(abs(complex(tr) - 1.0))


def check_rdm(
    rdm,
    *,
    context: str = "",
    tol: float = RDM_TRACE_TOL,
    psd_tol: float = RDM_PSD_TOL,
    strict: bool = False,
) -> float:
    """Warn (or raise) when an RDM is not a density matrix; return the defect.

    The failure this exists for is #845: on a degenerate corner one bond's RDM
    network contracts to **exactly zero**, and the energy sum then adds ``0``
    for that bond instead of its real value.  Measured at D=2, chi=4 (where
    ``chi = D**2`` makes the corner spectrum exactly flat, ``s0/s_last = 1.00``)
    the horizontal RDM came back with ``||rdm||_F = 0`` and ``tr = 0`` while the
    vertical one was untouched, so ``E`` was ``-0.2750`` against ``-0.5486``
    from every other chi -- a factor of 1.995, reported as a converged energy.

    Nothing upstream catches it: the CTM convergence criterion compares corner
    *singular values*, which are identical between the two bases, so it reports
    ``converged=True`` with ``diff ~ 1e-17``.  See also :func:`check_ctm_env`,
    which detects a different collapse (rank-1 corner) at the environment
    level.

    Three distinct failures are reported, because they need different
    responses -- and because **no single scalar certifies a matrix**.  Each of
    the three was found only after the previous one had shipped as "the"
    check:

    * **Non-finite entries** (``NaN``/``inf``) -- the contraction was
      numerically unstable upstream.  Checked over the whole array, not via the
      trace: a ``NaN`` off the diagonal leaves the trace at exactly 1, so a
      trace-only test hands back a defect of ``0.0`` -- an affirmative clean
      bill of health -- for an RDM whose energy is ``NaN`` (#848).  The
      all-``NaN`` case is missed for a second reason: the defect is then
      ``NaN``, and ``NaN > tol`` is ``False``, so the comparison below fails
      open in ``strict`` mode too.
    * **Trace defect** -- the #845 collapse described above.
    * **Loss of positive semi-definiteness** -- a negative eigenvalue (#854).
      The trace cannot see this either, for a reason of its own: it sums over
      the diagonal, so negative eigenvalues cancel against positive ones
      *inside the quantity being tested*.  Measured on a live U(1)-Sz D=3
      chi=12 run at 50 sweeps (#853), the spectrum was
      ``[-1.2997, -0.0736, 0.7446, 1.6287]`` at a trace of exactly 1 -- a
      negative eigenvalue larger in magnitude than the whole trace, passing
      silently.  It matters because ``<H> = tr(rho H)`` lies within the
      spectrum of a Hermitian ``H`` **only** for PSD ``rho``: #853 reported
      ``E = +0.759`` for two ``S.S`` bonds whose attainable maximum is
      ``+0.5``, a number no state can have.  Unlike the first two, this one
      was not constructed -- it fell out of an ordinary CTM run.

    The order is load-bearing.  Finiteness is tested first, because on a
    poisoned array the trace defect is not a meaningful number to report.  The
    trace is tested before positivity, because the collapsed all-zero RDM has
    an all-zero spectrum and is PSD *vacuously* -- checking positivity first
    would return it clean and lose #845.

    Args:
        rdm:     A trace-normalised RDM, 2- or 4-leg.
        context: Free-text label naming the bond (e.g. ``"2x1 horizontal"``),
                 so the message says *which* contribution died.
        tol:     Tolerance on ``|tr - 1|``.
        psd_tol: Tolerance on how far below zero the smallest eigenvalue may
                 sit, relative to the spectral radius.  See
                 :data:`RDM_PSD_TOL` for why it is relative rather than exact.
        strict:  Raise :class:`CollapsedRDMError` instead of warning.

    Returns:
        The trace defect, so callers can record it alongside the energy --
        except on a non-finite or non-PSD RDM, where
        :data:`INVALID_RDM_DEFECT` (``inf``) is returned instead.  The
        substitution is the point, and it applies to both: ``|tr - 1|`` is
        exactly ``0.0`` for an RDM whose only ``NaN`` sits off the diagonal,
        and exactly ``0.0`` again for one with a negative eigenvalue.  That is
        the same value a healthy RDM reports, so returning it would let a
        caller record an unusable RDM as "checked, healthy" -- the defect this
        function exists to prevent, one level up (#848, #854).
    """
    M = _as_rdm_matrix(rdm)
    defect = rdm_trace_defect(M)
    where = f" [{context}]" if context else ""

    n_bad = int((~np.isfinite(M)).sum())
    if n_bad:
        defect = INVALID_RDM_DEFECT
        msg = (
            f"reduced density matrix is not finite{where}: "
            f"{n_bad} of {M.size} entries are NaN or inf. The energy "
            f"contracted from it is NaN, which propagates into the total and "
            f"into every gradient taken through it. Note that the trace alone "
            f"does not catch this -- a non-finite entry off the diagonal "
            f"leaves |tr - 1| at 0. Look upstream at the contraction that "
            f"built this RDM, not at the normalisation. See #848."
        )
    elif defect > tol:
        msg = (
            f"reduced density matrix is not a density matrix{where}: "
            f"|tr - 1| = {defect:.3g} (a valid RDM has trace exactly 1). "
            f"The raw network contracted to zero or to a vanishing trace, so "
            f"this bond contributes 0 to the energy instead of its real "
            f"value, and the total is silently too small. This happens on a "
            f"degenerate corner -- check the corner spectrum, and note that "
            f"the CTM convergence flag cannot see it. See #845."
        )
    else:
        negativity, min_eig = _rdm_negativity(M)
        if negativity <= psd_tol:
            return defect
        defect = INVALID_RDM_DEFECT
        msg = (
            f"reduced density matrix is not positive semi-definite{where}: "
            f"smallest eigenvalue {min_eig:.3g}, which is {negativity:.3g} of "
            f"the spectral radius below zero. An expectation value "
            f"<H> = tr(rho H) lies within the spectrum of H only when rho is "
            f"PSD, so the energy contracted from this is unbounded by physics "
            f"and can fall outside the range attainable by any state. The "
            f"trace cannot see this -- it sums over the diagonal, so negative "
            f"eigenvalues cancel against positive ones inside it, and "
            f"|tr - 1| is still ~0 here. Look at whether the CTM actually "
            f"converged. See #854."
        )

    if strict:
        raise CollapsedRDMError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return defect


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
        **This is a weak signal. It errs in both directions, and
        :func:`ctm_corner_rank` is the only sound detector.**

        *False positives.* A fully converged environment is flat in chi too --
        that is what convergence means -- so a scan whose energy has saturated
        to the last bit is reported here with nothing wrong.

        *False negatives.* A collapsed environment is not reliably
        **bit**-identical either.  The rank-1 corner is the same at every chi,
        but the energy contracted from it need not be: the split ``1x1`` path
        returns ``0.49620072949960814`` at chi=4 and ``...803`` at chi=16 on
        macOS while agreeing exactly on Linux.  So exact equality can miss a
        genuine collapse that differs in the last ULP.

        Both directions were found the hard way, by required CI, on successive
        attempts to write a #723 regression test around the chi response.

        Use this only to *audit recorded data* where the corner rank was never
        stored -- bit-identity across a wide chi range (the D=8 scan of #747
        was identical to 13 digits across an 8x range) is a strong smell worth
        chasing.  It is not a test, and the drivers record ``corner_rank``
        rather than relying on it.

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
