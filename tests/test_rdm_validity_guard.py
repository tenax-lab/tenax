"""An RDM that contracts to zero must not be summed into the energy (#845).

``_normalise_rdm`` divides by the trace, so a valid RDM comes back with trace
exactly 1.  When the raw network contracts to zero its zero-matrix guard
returns zeros -- correctly, since ``0/0`` is not a density matrix -- and the
energy sum then added that bond's ``0`` as though it were a measurement.

Measured on a D=2 state at chi=4 (where ``chi = D**2`` makes the corner
spectrum exactly flat): the horizontal RDM came back with ``||rdm||_F = 0``
while the vertical one was untouched, so ``E = -0.2750`` against ``-0.5486``
from every other chi -- a factor of 1.995, with ``converged=True`` and
``diff ~ 1e-17`` reported throughout.  Nothing upstream could catch it: the CTM
criterion compares corner *singular values*, which are identical between the
two bases.

The trigger here is a zeroed environment tensor rather than that degeneracy.
The degenerate case is real but not a stable CI hook -- which corner basis the
sweep lands on shifted with the JAX/LAPACK build -- whereas zeroing a corner
reproduces the same *mechanism* (one bond's network contracts to zero, the
other survives) deterministically.
"""

from __future__ import annotations

import math
import pathlib
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_diagnostics import (
    INVALID_RDM_DEFECT,
    RDM_PSD_TOL,
    RDM_TRACE_TOL,
    CollapsedRDMError,
    _psd_tol_for,
    check_rdm,
    rdm_trace_defect,
)
from tenax.algorithms._ctm_tensor_energy import (
    _normalise_rdm,
    _normalise_rdm_for_energy,
)
from tenax.algorithms.ipeps import heisenberg_gate, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.ipeps_ctm_convergence import ctm_2site
from tenax.algorithms.ipeps_rdm import (
    _rdm1x2_2site,
    _rdm2x1_2site,
    compute_energy_ctm_2site,
)

D = 2
DPHYS = 2
CFG = CTMConfig(chi=4, chi_I=4, max_iter=5, conv_tol=1e-10)
MATCH = "not a density matrix"

#: Common prefix of all three guard messages -- "reduced density matrix is not
#: finite / not a density matrix / not positive semi-definite".
#:
#: The "must stay silent" tests below filter on *this*, not on ``MATCH``.  They
#: used to filter on ``MATCH`` alone, which made them blind to any check added
#: later: when the PSD test landed (#854) they would have passed a healthy
#: environment that warned about positivity, because that message does not
#: contain "not a density matrix".  A silence assertion has to be able to see
#: every warning the guard can raise, or it silently narrows as the guard grows.
ANY_RDM_MATCH = "reduced density matrix"


def _site(seed: int) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (D, D, D, D, DPHYS))


@pytest.fixture(scope="module")
def env():
    """A healthy 2-site environment plus its state and gate."""
    A, B = _site(0), _site(7)
    env_A, env_B = ctm_2site(A, B, CFG)
    gate = sublattice_rotate_gate(heisenberg_gate()).todense()
    return A, B, env_A, env_B, gate


def _kill(env_B, name):
    """Zero one environment tensor, killing the bonds that contract through it."""
    return env_B._replace(**{name: jnp.zeros_like(getattr(env_B, name))})


def _rdm_warnings(fn):
    """Run ``fn`` and return every RDM-validity warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [w for w in caught if ANY_RDM_MATCH in str(w.message)]


# ---------------------------------------------------------------------------
# The guard fires on a dead bond, and says which one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corner", "expected"),
    [("C2", "_rdm2x1_2site"), ("C4", "_rdm1x2_2site")],
)
def test_a_dead_bond_warns_and_names_itself(env, corner, expected):
    """One bond dies, the other survives -- the #845 shape exactly.

    The context label matters: with two bonds in the sum, "an RDM collapsed"
    does not tell you whether the horizontal or the vertical contribution went
    missing, which is the first thing you need in order to chase it.
    """
    A, B, env_A, env_B, _gate = env
    dead = _kill(env_B, corner)

    _r, caught = _rdm_warnings(
        lambda: (
            _rdm2x1_2site(A, B, env_A, dead, DPHYS),
            _rdm1x2_2site(A, B, env_A, dead, DPHYS),
        )
    )

    assert len(caught) == 1, f"expected exactly one dead bond, got {len(caught)}"
    assert expected in str(caught[0].message)
    assert issubclass(caught[0].category, RuntimeWarning)


def test_both_bonds_dead_warns_for_each(env):
    """Zeroing ``C3`` takes out both RDMs; neither may pass quietly."""
    A, B, env_A, env_B, _gate = env
    dead = _kill(env_B, "C3")

    _r, caught = _rdm_warnings(
        lambda: (
            _rdm2x1_2site(A, B, env_A, dead, DPHYS),
            _rdm1x2_2site(A, B, env_A, dead, DPHYS),
        )
    )

    assert len(caught) == 2


# ---------------------------------------------------------------------------
# ...and stays quiet otherwise. A guard that cries wolf gets filtered out.
# ---------------------------------------------------------------------------


def test_a_healthy_environment_is_silent(env):
    """The ordinary path must not warn, or the warning is worthless."""
    A, B, env_A, env_B, gate = env

    _E, caught = _rdm_warnings(
        lambda: compute_energy_ctm_2site(A, B, env_A, env_B, gate, DPHYS)
    )

    assert not caught, f"warned on a healthy environment: {caught}"


def test_a_mutilated_environment_with_valid_rdms_is_silent(env):
    """Zeroing ``C1`` leaves **both** RDMs with trace 1 -- so: no warning.

    This is the sharp negative control.  The guard keys on the RDM actually
    being a density matrix, not on the environment looking damaged, and this
    distinguishes those two.  A check that fired here would fire on healthy
    physics too.
    """
    A, B, env_A, env_B, gate = env
    dead = _kill(env_B, "C1")

    E, caught = _rdm_warnings(
        lambda: compute_energy_ctm_2site(A, B, env_A, dead, gate, DPHYS)
    )

    assert not caught, f"warned although both RDMs were valid: {caught}"
    assert jnp.isfinite(E)


# ---------------------------------------------------------------------------
# The harm the warning is about.
# ---------------------------------------------------------------------------


def test_the_energy_is_silently_short_by_exactly_the_dead_bond(env):
    """Pins the defect: the total is the surviving bond alone, not an error.

    Without the guard this number is indistinguishable from a real energy --
    which is how it survived as a plausible-looking value ~2x off.
    """
    A, B, env_A, env_B, gate = env
    dead = _kill(env_B, "C2")  # horizontal only
    H = gate.reshape(DPHYS, DPHYS, DPHYS, DPHYS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_full = float(compute_energy_ctm_2site(A, B, env_A, env_B, gate, DPHYS))
        E_dead = float(compute_energy_ctm_2site(A, B, env_A, dead, gate, DPHYS))
        rdm_v = _rdm1x2_2site(A, B, env_A, dead, DPHYS)
    E_v_only = float(jnp.real(jnp.einsum("ijkl,ijkl->", rdm_v, H)))

    # The horizontal contribution is not wrong, it is *absent*.
    assert E_dead == pytest.approx(E_v_only, abs=1e-12)
    assert E_dead != pytest.approx(E_full, abs=1e-6)


# ---------------------------------------------------------------------------
# Scope: the check must not leak into tracing or into the excitation path.
# ---------------------------------------------------------------------------


def test_the_guard_is_inert_under_jit(env):
    """Under ``jit`` there is no concrete value; it must skip, not crash.

    The AD path traces these functions on every optimiser step, so a guard
    that raised on a tracer -- or that inserted a callback per bond per sweep
    -- would be worse than the bug.
    """
    A, B, env_A, env_B, gate = env
    dead = _kill(env_B, "C2")

    jitted = jax.jit(
        lambda a, b, ea, eb: compute_energy_ctm_2site(a, b, ea, eb, gate, DPHYS)
    )
    E, caught = _rdm_warnings(lambda: float(jitted(A, B, env_A, dead)))

    assert not caught
    assert jnp.isfinite(E)


def test_the_plain_normaliser_still_accepts_an_all_zero_matrix():
    """The excitation path passes ``B = 0`` legitimately and must stay quiet.

    A zero excitation vector has zero norm; that is a real input, not a
    collapse.  This is why the checked variant is a separate function rather
    than a change to ``_normalise_rdm`` -- putting the check in the shared
    helper would break the excitation solver.
    """
    zeros = jnp.zeros((4, 4))

    out, caught = _rdm_warnings(lambda: _normalise_rdm(zeros))

    assert not caught
    assert jnp.all(out == 0)


def test_the_checked_normaliser_warns_on_the_same_input():
    """Same matrix, opposite verdict -- the two functions must differ here."""
    zeros = jnp.zeros((4, 4))

    with pytest.warns(RuntimeWarning, match=MATCH):
        _normalise_rdm_for_energy(zeros, "unit-test")


# ---------------------------------------------------------------------------
# The detector itself.
# ---------------------------------------------------------------------------


def test_trace_defect_is_a_two_state_discriminator():
    """0 for anything with physical content, 1 for the collapsed case."""
    valid = jnp.eye(4) / 4.0
    assert rdm_trace_defect(valid) == pytest.approx(0.0, abs=1e-12)
    assert rdm_trace_defect(jnp.zeros((4, 4))) == pytest.approx(1.0, abs=1e-12)
    # ...and it accepts the 4-leg form the builders return.
    assert rdm_trace_defect(valid.reshape(2, 2, 2, 2)) == pytest.approx(0.0, abs=1e-12)


def test_check_rdm_strict_raises_instead_of_warning():
    """Drivers that would rather stop than record a wrong number can."""
    with pytest.raises(CollapsedRDMError, match=MATCH):
        check_rdm(jnp.zeros((4, 4)), context="unit-test", strict=True)

    assert check_rdm(jnp.eye(4) / 4.0) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Non-finite RDMs (#848).
#
# The guard shipped in #847 tested only the trace, which fails open on NaN
# twice over: ``NaN > tol`` is False, and a non-finite entry off the diagonal
# does not move the trace at all.  Both cases returned "healthy" for an RDM
# whose energy is NaN.
# ---------------------------------------------------------------------------

NONFINITE_MATCH = "not finite"


def _poisoned(where: tuple[int, int], value: float) -> jnp.ndarray:
    """A trace-1 RDM with a single non-finite entry at ``where``."""
    M = jnp.eye(4) / 4.0
    return M.at[where].set(value)


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
@pytest.mark.parametrize("where", [(0, 1), (2, 3), (1, 0)])
def test_a_nonfinite_entry_off_the_diagonal_is_caught(where, value):
    """The trace is exactly 1 here -- only an array-wide test sees these.

    This is the case the #847 review did *not* find, and the more dangerous of
    the two: ``rdm_trace_defect`` returns ``0.0``, an affirmative clean bill of
    health, for an RDM that contracts to a NaN energy.
    """
    rdm = _poisoned(where, value)
    assert rdm_trace_defect(rdm) == pytest.approx(0.0, abs=1e-12), (
        "precondition: this RDM must have a clean trace, or the test is not "
        "exercising the off-diagonal blind spot"
    )

    with pytest.warns(RuntimeWarning, match=NONFINITE_MATCH):
        check_rdm(rdm, context="unit-test")

    with pytest.raises(CollapsedRDMError, match=NONFINITE_MATCH):
        check_rdm(rdm, context="unit-test", strict=True)


def test_an_all_nan_rdm_is_caught():
    """The review's case: ``NaN > tol`` is False, so the comparison failed open.

    Note this is the *less* severe of the two -- the defect at least came back
    ``NaN`` rather than ``0.0``.
    """
    rdm = jnp.full((4, 4), jnp.nan)

    with pytest.warns(RuntimeWarning, match=NONFINITE_MATCH):
        check_rdm(rdm, context="unit-test")

    with pytest.raises(CollapsedRDMError, match=NONFINITE_MATCH):
        check_rdm(rdm, context="unit-test", strict=True)


def test_the_nonfinite_message_is_not_the_collapse_message():
    """The two failures need different responses, so they must read differently.

    A collapsed RDM points at the corner spectrum (#845); a poisoned one points
    upstream at the contraction (#848).  Reporting a NaN as "not a density
    matrix ... contributes 0 to the energy" would send a reader to the wrong
    place -- it contributes NaN, not 0.
    """
    with pytest.warns(RuntimeWarning) as poisoned:
        check_rdm(_poisoned((0, 1), jnp.nan), context="unit-test")
    with pytest.warns(RuntimeWarning) as collapsed:
        check_rdm(jnp.zeros((4, 4)), context="unit-test")

    poisoned_msg = str(poisoned[0].message)
    collapsed_msg = str(collapsed[0].message)

    assert "#848" in poisoned_msg and "#845" not in poisoned_msg
    assert "#845" in collapsed_msg and "#848" not in collapsed_msg
    assert "contributes 0 to the energy" not in poisoned_msg


@pytest.mark.parametrize(
    "rdm",
    [
        _poisoned((0, 1), jnp.nan),
        _poisoned((2, 3), jnp.inf),
        jnp.full((4, 4), jnp.nan),
    ],
    ids=["nan-off-diagonal", "inf-off-diagonal", "all-nan"],
)
def test_a_nonfinite_rdm_never_reports_a_healthy_defect(rdm):
    """The returned defect is what drivers record next to the energy.

    A poisoned RDM must not be recordable as ``0.0`` -- that is precisely the
    number that means "checked, healthy" -- and must not read as healthy under
    the tolerance comparison a caller is most likely to write.

    An earlier version of this test asserted
    ``defect != 0.0 or not isfinite(rdm).all()``, which is **vacuous**: the
    right operand is ``True`` for every poisoned input by construction, so the
    whole disjunction is ``True`` regardless of what ``check_rdm`` returns. It
    passed against an implementation that returned ``0.0`` here. Assert the
    property directly, with no escape clause.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        defect = check_rdm(rdm, context="unit-test")

    assert defect != 0.0, "a poisoned RDM reported the healthy sentinel 0.0"
    assert defect > RDM_TRACE_TOL, (
        f"defect {defect!r} does not exceed the tolerance, so a caller gating "
        f"on `check_rdm(...) > tol` reads this poisoned RDM as healthy"
    )
    assert defect == INVALID_RDM_DEFECT


def test_the_invalid_sentinel_is_not_nan():
    """``nan`` would re-create #848 one level up.

    The bug being fixed is a tolerance comparison failing open. Returning
    ``nan`` hands that same trap to every caller, since ``nan > tol`` is
    ``False``; the sentinel must compare *greater* than any tolerance.
    """
    assert not math.isnan(INVALID_RDM_DEFECT)
    assert INVALID_RDM_DEFECT > RDM_TRACE_TOL
    assert INVALID_RDM_DEFECT > 1e300


def test_a_healthy_rdm_still_returns_its_real_defect():
    """The sentinel must not leak into the valid path."""
    assert check_rdm(jnp.eye(4) / 4.0) == pytest.approx(0.0, abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert check_rdm(jnp.zeros((4, 4))) == pytest.approx(1.0, abs=1e-12)


def test_a_finite_rdm_with_a_real_trace_defect_still_reports_the_collapse():
    """Adding the finiteness test must not shadow the #845 path it precedes."""
    with pytest.warns(RuntimeWarning, match=MATCH):
        check_rdm(jnp.zeros((4, 4)), context="unit-test")


def test_the_finiteness_check_covers_the_4_leg_form():
    """The builders return ``(d, d, d, d)``; the guard must not only see 2-leg."""
    rdm = _poisoned((0, 1), jnp.nan).reshape(2, 2, 2, 2)
    with pytest.warns(RuntimeWarning, match=NONFINITE_MATCH):
        check_rdm(rdm, context="unit-test")


# ---------------------------------------------------------------------------
# Non-PSD RDMs (#854).
#
# The third hole in the same guard, and the first that was not constructed --
# it fell out of an ordinary CTM run (#853).  The trace is a *sum* over the
# diagonal, so negative eigenvalues cancel against positive ones inside the
# very quantity being tested.  No single scalar certifies a matrix.
# ---------------------------------------------------------------------------

PSD_MATCH = "not positive semi-definite"

#: Eigenvalues measured on a live U(1)-Sz D=3 chi=12 CTM run at 50 sweeps
#: (#853).  They sum to exactly 1, and the most negative one is larger in
#: magnitude than the whole trace.
LIVE_NONPSD_EIGS = (-1.299674, -0.073617, 0.744597, 1.628694)


def _hermitian_with_spectrum(eigs, seed: int | None = None) -> jnp.ndarray:
    """A Hermitian matrix whose eigenvalues are exactly ``eigs``.

    ``seed=None`` gives the diagonal form, where the negative eigenvalue is
    plainly visible.  A seed conjugates it by a fixed orthogonal ``Q``, so no
    entry is individually suspicious and the check has to actually diagonalise.
    """
    L = jnp.diag(jnp.asarray(eigs, dtype=float))
    if seed is None:
        return L
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (len(eigs),) * 2))
    return Q @ L @ Q.conj().T


def _any_warnings(fn):
    """Run ``fn`` and return every warning it raised, filtered by nothing.

    Stronger than :func:`_rdm_warnings` and used for the "a valid RDM must be
    silent" controls, which call ``check_rdm`` directly: there is no other
    source of warnings on that path, so anything at all -- including a NumPy
    warning out of ``eigvalsh`` -- is a result worth failing on.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, list(caught)


def test_the_trace_cannot_see_a_negative_eigenvalue():
    """The claim #854 rests on, stated as an assertion rather than prose.

    Every existing check passes this matrix: it is finite, and its trace is
    exactly 1.  It is still not a density matrix, and the energy contracted
    from it is not bounded by the Hamiltonian's spectrum.
    """
    rdm = _hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=3)

    assert bool(jnp.isfinite(rdm).all()), "precondition: finiteness must pass"
    assert rdm_trace_defect(rdm) == pytest.approx(0.0, abs=1e-12), (
        "precondition: the trace must be clean, or this is not exercising the "
        "blind spot"
    )
    assert float(jnp.min(jnp.linalg.eigvalsh(rdm))) < -1.0


@pytest.mark.parametrize("seed", [None, 3], ids=["diagonal", "rotated"])
def test_a_non_psd_rdm_is_caught(seed):
    """Both spellings of the same spectrum, since one is diagonal by luck."""
    rdm = _hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=seed)

    with pytest.warns(RuntimeWarning, match=PSD_MATCH):
        check_rdm(rdm, context="unit-test")

    with pytest.raises(CollapsedRDMError, match=PSD_MATCH):
        check_rdm(rdm, context="unit-test", strict=True)


def test_a_non_psd_rdm_never_reports_a_healthy_defect():
    """Same recording trap as #848: ``0.0`` is the value meaning "healthy".

    The trace defect of this RDM is exactly ``0.0``, so returning it would let
    a driver record a physically impossible measurement as checked and clean.
    """
    rdm = _hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        defect = check_rdm(rdm, context="unit-test")

    assert defect != 0.0, "a non-PSD RDM reported the healthy sentinel 0.0"
    assert defect > RDM_TRACE_TOL, (
        f"defect {defect!r} does not exceed the tolerance, so a caller gating "
        f"on `check_rdm(...) > tol` reads this non-PSD RDM as healthy"
    )
    assert defect == INVALID_RDM_DEFECT


@pytest.mark.parametrize(
    "eigs",
    [
        (0.25, 0.25, 0.25, 0.25),
        (0.7, 0.2, 0.1, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, -1e-14),
    ],
    ids=["flat", "generic", "pure", "roundoff-negative"],
)
@pytest.mark.parametrize("seed", [None, 5], ids=["diagonal", "rotated"])
def test_a_psd_rdm_is_silent(eigs, seed):
    """A guard that cries wolf gets filtered out, so the controls come first.

    The last spectrum is the one that sets the tolerance. A converged RDM's
    smallest eigenvalues land a few ulp either side of zero -- a rank-deficient
    one sits exactly there -- so an exact ``min_eig < 0`` test would warn on
    every healthy run and the warning would be suppressed within a week.
    """
    rdm = _hermitian_with_spectrum(eigs, seed=seed)

    _defect, caught = _any_warnings(lambda: check_rdm(rdm, context="unit-test"))

    assert not caught, f"warned on a valid density matrix: {caught}"


def test_the_collapsed_rdm_still_reports_the_trace_defect():
    """Ordering: an all-zero RDM is PSD *vacuously*, so the trace check wins.

    Its spectrum is all zeros, which passes any PSD test. Were the PSD check to
    run first and return early, #845 would stop being reported -- so this pins
    the order rather than merely the presence of the new check.
    """
    with pytest.warns(RuntimeWarning, match=MATCH) as caught:
        defect = check_rdm(jnp.zeros((4, 4)), context="unit-test")

    assert PSD_MATCH not in str(caught[0].message)
    assert defect == pytest.approx(1.0, abs=1e-12)


def test_the_psd_message_is_not_either_of_the_other_two():
    """Three failures, three responses, so the message has to say which.

    Non-finite points upstream at the contraction (#848); a dead trace points
    at the corner spectrum (#845); a negative eigenvalue means the environment
    is not a positive map and the energy is unbounded by physics (#854).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_rdm(_hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=3), context="t")
    psd_msg = str(caught[0].message)

    assert "#854" in psd_msg
    assert "#845" not in psd_msg and "#848" not in psd_msg
    assert "contributes 0 to the energy" not in psd_msg


def test_the_psd_check_covers_the_4_leg_form():
    """The builders return ``(d, d, d, d)``; the guard must not only see 2-leg."""
    rdm = _hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=3).reshape(2, 2, 2, 2)

    with pytest.warns(RuntimeWarning, match=PSD_MATCH):
        check_rdm(rdm, context="unit-test")


def test_a_non_psd_rdm_can_exceed_the_hamiltonian_spectrum():
    """The harm: ``<H> = tr(rho H)`` is bounded by H only when ``rho`` is PSD.

    Built by aligning the RDM's negative eigenvalue with the Heisenberg gate's
    ground state -- the configuration that produced #853's ``E = +0.759``
    against an attainable maximum of ``+0.5`` over two bonds.  The resulting
    number is not merely inaccurate; no state can have it.
    """
    H = jnp.asarray(heisenberg_gate().todense()).reshape(4, 4)
    h, U = jnp.linalg.eigh(H)  # ascending, matching LIVE_NONPSD_EIGS
    rdm = U @ jnp.diag(jnp.asarray(LIVE_NONPSD_EIGS)) @ U.conj().T

    E = float(
        jnp.real(
            jnp.einsum("ijkl,ijkl->", rdm.reshape(2, 2, 2, 2), H.reshape(2, 2, 2, 2))
        )
    )

    assert rdm_trace_defect(rdm) == pytest.approx(0.0, abs=1e-12)
    assert E > float(h.max()), (
        f"E = {E:.6f} must exceed max eig(H) = {float(h.max()):.6f} for this "
        f"test to be demonstrating impossible physics"
    )

    with pytest.warns(RuntimeWarning, match=PSD_MATCH):
        check_rdm(rdm, context="unit-test")


# ---------------------------------------------------------------------------
# Every energy path, not just the one #845 was filed against.
# ---------------------------------------------------------------------------


def test_no_energy_path_bypasses_the_checked_normaliser():
    """#845 reached the energy through one of 21 shared call sites.

    Fixing only the two functions named in the issue would have left the other
    19 silently summing dead bonds -- the same "fix landed on one of N copies"
    pattern as #828, #829 and #842.  This asserts the sweep stayed complete, so
    a new RDM builder cannot quietly reintroduce it.

    ``ipeps_excitations`` is the one legitimate exception: ``B = 0`` is a valid
    excitation vector there and must normalise to zero without complaint.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "tenax"
    allowed = {
        # the checked wrapper delegates to the plain one by construction
        ("_ctm_tensor_energy.py", "_normalise_rdm_for_energy"),
        ("ipeps_excitations.py", "_rdm2x1_with_open_tensors"),
        ("ipeps_excitations.py", "_rdm1x2_with_open_tensors"),
    }

    offenders = []
    for path in sorted(root.rglob("*.py")):
        func = "?"
        for line in path.read_text().splitlines():
            match = re.match(r"def (\w+)", line)
            if match:
                func = match.group(1)
                continue  # the definition's own signature is not a call site
            if re.search(r"(?<!_for_energy)\b_normalise_rdm\(", line):
                if (path.name, func) not in allowed:
                    offenders.append(f"{path.name}:{func}")

    assert not offenders, (
        f"these build an RDM for an energy contraction but skip the validity "
        f"check: {offenders}. Use _normalise_rdm_for_energy(mat, '<name>')."
    )


# ---------------------------------------------------------------------------
# The PSD tolerance is dimensionless but not dtype-free (#873).
#
# RDM_PSD_TOL's 1e-8 was chosen against float64's ~1e-16 roundoff.  float32
# eps is 1.19e-7 -- an order of magnitude *above* that tolerance -- so on a
# float32 input the guard fires on its own arithmetic.  Latent rather than
# live: ``import tenax`` sets jax_enable_x64, so this is a caller who
# deliberately works in single precision.
# ---------------------------------------------------------------------------


def _psd_rank_deficient(n: int, rank: int, seed: int, dtype) -> np.ndarray:
    """PSD by construction: ``Q diag(s) Q^T`` with ``Q`` orthonormal.

    Rank-deficient on purpose -- that is the ordinary case (a product state's
    two-site RDM has one non-zero eigenvalue), and it is what puts eigenvalues
    at exactly zero where roundoff decides their sign.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, rank)))
    rho = Q @ np.diag(np.linspace(1.0, 0.2, rank)) @ Q.T
    return (rho / np.trace(rho)).astype(dtype)


@pytest.mark.parametrize("n,rank", [(4, 2), (16, 8), (64, 32)])
@pytest.mark.parametrize("seed", range(4))
def test_a_valid_float32_rdm_is_not_reported_non_psd(n, rank, seed):
    """The regression: these are PSD by construction and must stay silent.

    Measured before the fix, worst negativity over 20 trials per size: 2.7e-08
    at n=4, 2.6e-08 at n=16, 3.1e-08 at n=64 -- all above RDM_PSD_TOL's 1e-8,
    firing in 7/20, 19/20 and 20/20 trials.  The identical matrices in float64
    give ~1e-17.
    """
    rdm = _psd_rank_deficient(n, rank, seed, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        defect = check_rdm(rdm, context="f32")
    assert defect != INVALID_RDM_DEFECT


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_float32_floor_does_not_blind_the_check(dtype):
    """The fix must not be "loosen until nothing fires".

    #853's real spectrum sits 0.8 of the spectral radius below zero, which is
    seven orders above even the float32 floor, so raising the tolerance to
    cover f32 roundoff cannot cost the case the check was built for.
    """
    rdm = np.asarray(_hermitian_with_spectrum(LIVE_NONPSD_EIGS, seed=3), dtype=dtype)
    with pytest.warns(RuntimeWarning, match=PSD_MATCH):
        defect = check_rdm(rdm, context="t")
    assert defect == INVALID_RDM_DEFECT


def test_the_floor_is_inert_in_float64():
    """It must not quietly relax the default path.

    ``32 * eps(float64)`` is 7.1e-15, four orders below RDM_PSD_TOL, so the
    tolerance every existing test was calibrated against is unchanged.
    """
    assert _psd_tol_for(np.zeros(1, dtype=np.float64), RDM_PSD_TOL) == RDM_PSD_TOL
    assert _psd_tol_for(np.zeros(1, dtype=np.complex128), RDM_PSD_TOL) == RDM_PSD_TOL
    f32 = _psd_tol_for(np.zeros(1, dtype=np.float32), RDM_PSD_TOL)
    assert f32 > RDM_PSD_TOL
    assert f32 == pytest.approx(32 * np.finfo(np.float32).eps)
