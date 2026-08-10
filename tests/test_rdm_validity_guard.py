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

import pathlib
import re
import warnings

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_diagnostics import (
    CollapsedRDMError,
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
    """Run ``fn`` and return the RDM-validity warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [w for w in caught if MATCH in str(w.message)]


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
