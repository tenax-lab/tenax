"""The dense CTM entry points must report whether they converged (#839).

``ctm``, ``ctm_2site`` and ``ctm_split`` each computed a ``converged`` flag and
an iteration count inside their loop and then discarded both, so a caller could
not tell a converged environment from one that silently exhausted ``max_iter``.
``ipeps()`` -- public API -- returned an energy from such an environment with
no channel to say so.  That is the forward-side twin of #801/#824.

The tests are written so that a regression is *visible* rather than merely
possible: each one starves a real sweep and checks the reported status against
what the loop actually did, rather than trusting the flag against itself.
"""

from __future__ import annotations

import warnings

import jax
import pytest

from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import (
    CTMConfig,
    CTMEnvironment,
    SplitCTMEnvironment,
    iPEPSConfig,
)
from tenax.algorithms.ipeps_ctm_convergence import (
    CTMConvergenceInfo,
    ctm,
    ctm_2site,
    ctm_split,
)


@pytest.fixture(scope="module")
def su_state():
    """A real D=2 state and its gate.

    Deliberately the *unconverged* tau=2 simple-update state (40 steps at
    dt=0.05): its CTM provably does not converge at chi=6 (#838), which is what
    makes it a usable negative case here.  Do not "improve" it to a longer
    imaginary time -- a state whose CTM converges cannot exercise the
    ``converged=False`` branch.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _E, (A, B), _ = ipeps(
            gate,
            None,
            iPEPSConfig(
                max_bond_dim=2,
                num_imaginary_steps=40,
                dt=0.05,
                unit_cell="1x1",
                # This CTM is dead weight: ``ipeps()`` runs simple update first and the
                # tensors are fixed before it starts, so ``config.ctm`` cannot affect
                # them -- and this fixture discards both the energy and the env.  It
                # was spending its whole budget without converging (the
                # "CTM did not converge in ipeps()" warning), then throwing the
                # result away.  chi is unchanged; only the sweep count is cut (#933).
                ctm=CTMConfig(chi=6, max_iter=2, conv_tol=1e-10),
            ),
        )
    return A.todense(), B.todense(), gate.todense()


# ---------------------------------------------------------------------------
# The flag must distinguish the two exits, in both directions.
# ---------------------------------------------------------------------------


def test_ctm_2site_reports_exhausted_max_iter(su_state):
    """A starved sweep must say so, and say how far off it was."""
    A, B, _ = su_state
    max_iter = 100
    _eA, _eB, info = ctm_2site(
        A, B, CTMConfig(chi=6, max_iter=max_iter, conv_tol=1e-10), return_meta=True
    )

    assert not bool(info.converged)
    # Ran the whole budget -- this is what "exhausted" means, and it is the
    # claim a bare `converged=False` could satisfy vacuously.
    assert int(info.n_iter) == max_iter
    # ...and was nowhere near the tolerance it was given, so the flag is not
    # just reporting a marginal miss.
    assert float(info.diff) > 1e-10


def test_ctm_2site_reports_genuine_convergence(su_state):
    """The same state with a reachable tolerance must converge and stop early.

    Without this the ``converged`` field could be hardwired False and every
    other test here would still pass.
    """
    A, B, _ = su_state
    max_iter = 100
    _eA, _eB, info = ctm_2site(
        A, B, CTMConfig(chi=6, max_iter=max_iter, conv_tol=1e-2), return_meta=True
    )

    assert bool(info.converged)
    assert int(info.n_iter) < max_iter, "converged but still ran the full budget"
    assert float(info.diff) < 1e-2


def test_ctm_reports_status(su_state):
    """Same contract on the 1x1 entry point."""
    A, _B, _ = su_state
    starved = ctm(A, CTMConfig(chi=6, max_iter=20, conv_tol=1e-12), return_meta=True)[1]
    assert not bool(starved.converged)
    assert int(starved.n_iter) == 20

    loose = ctm(A, CTMConfig(chi=6, max_iter=100, conv_tol=1e-2), return_meta=True)[1]
    assert bool(loose.converged)
    assert int(loose.n_iter) < 100


def test_ctm_split_reports_status(su_state):
    """``ctm_split`` is a Python loop with a ``break``; same contract."""
    A, _B, _ = su_state
    _env, starved = ctm_split(
        A, CTMConfig(chi=6, chi_I=4, max_iter=5, conv_tol=1e-14), return_meta=True
    )
    assert not bool(starved.converged)
    assert int(starved.n_iter) == 5

    _env, loose = ctm_split(
        A, CTMConfig(chi=6, chi_I=4, max_iter=60, conv_tol=1e-2), return_meta=True
    )
    assert bool(loose.converged)
    assert int(loose.n_iter) < 60


# ---------------------------------------------------------------------------
# The fix must not change what these functions already returned.
# ---------------------------------------------------------------------------


def test_default_return_arity_is_unchanged(su_state):
    """All three are public API; ``return_meta`` is opt-in for that reason.

    Asserted by type, not by ``isinstance(x, tuple)``: the environments are
    themselves NamedTuples, so a "not a tuple" check passes vacuously for the
    wrong reason and would keep passing if these started returning pairs.
    """
    A, B, _ = su_state
    cfg = CTMConfig(chi=6, chi_I=4, max_iter=10)

    pair = ctm_2site(A, B, cfg)
    assert len(pair) == 2
    assert all(isinstance(e, CTMEnvironment) for e in pair)

    env = ctm(A, cfg)
    assert isinstance(env, CTMEnvironment)

    split_env = ctm_split(A, cfg)
    assert isinstance(split_env, SplitCTMEnvironment)

    # ...and the opt-in really does change the shape, so the check above is
    # discriminating rather than trivially true.
    assert isinstance(ctm(A, cfg, return_meta=True)[1], CTMConvergenceInfo)


def test_environments_are_bit_identical_with_and_without_meta(su_state):
    """Reporting must be free: the carry change must not perturb the sweep.

    ``diff`` was added to the ``lax.while_loop`` carry to be reported.  If that
    altered the iteration in any way, every energy in the suite would shift.
    """
    A, B, _ = su_state
    cfg = CTMConfig(chi=6, max_iter=30, conv_tol=1e-10)

    eA_plain, eB_plain = ctm_2site(A, B, cfg)
    eA_meta, eB_meta, _info = ctm_2site(A, B, cfg, return_meta=True)

    for plain, meta, side in ((eA_plain, eA_meta, "A"), (eB_plain, eB_meta, "B")):
        for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
            p, m = getattr(plain, name), getattr(meta, name)
            assert (p == m).all(), f"env_{side}.{name} differs with return_meta"


# ---------------------------------------------------------------------------
# The user-facing half.
# ---------------------------------------------------------------------------


def test_ipeps_warns_when_its_ctm_did_not_converge():
    """``ipeps()`` has no return slot for status, so it must warn (#747 policy).

    This is the actual user-facing harm in #839: a public entry point handing
    back an energy derived from a non-converged environment, silently.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=6, max_iter=100, conv_tol=1e-10),
    )
    with pytest.warns(UserWarning, match="CTM did not converge"):
        ipeps(gate, None, config)


def test_ipeps_is_silent_when_the_ctm_converges():
    """The warning must be informative, not ambient.

    A warning that fires on every call is one users filter out, which would
    put us back where #839 started.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=40,
        dt=0.05,
        unit_cell="1x1",
        ctm=CTMConfig(chi=6, max_iter=100, conv_tol=1e-2),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ipeps(gate, None, config)
    ctm_warnings = [w for w in caught if "CTM did not converge" in str(w.message)]
    assert not ctm_warnings, f"warned on a converged CTM: {ctm_warnings}"


def test_ipeps_energy_is_unchanged_by_the_warning_path(su_state):
    """The reported energy must be exactly what it was before #839.

    ``ipeps()`` now calls ``ctm_2site(..., return_meta=True)``; that must be a
    pure addition, since every benchmark number in the repo comes through here.
    """
    A, B, gate_dense = su_state
    from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site

    cfg = CTMConfig(chi=6, max_iter=100, conv_tol=1e-10)
    eA, eB = ctm_2site(A, B, cfg)
    expected = float(compute_energy_ctm_2site(A, B, eA, eB, gate_dense, 2))

    gate = sublattice_rotate_gate(heisenberg_gate())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E, _t, _e = ipeps(
            gate,
            None,
            iPEPSConfig(
                max_bond_dim=2,
                num_imaginary_steps=40,
                dt=0.05,
                unit_cell="1x1",
                ctm=cfg,
            ),
        )
    assert float(E) == expected


def test_info_fields_survive_jit(su_state):
    """``ctm`` / ``ctm_2site`` keep the flag as JAX arrays to stay jittable.

    Pins the documented contract: the entry points must remain usable under
    ``jit``, which forbids converting the carry to Python scalars internally.
    """
    A, B, _ = su_state
    cfg = CTMConfig(chi=6, max_iter=10, conv_tol=1e-12)

    @jax.jit
    def run(a, b):
        _eA, _eB, info = ctm_2site(a, b, cfg, return_meta=True)
        return info.converged, info.n_iter, info.diff

    conv, n_iter, diff = run(A, B)
    assert not bool(conv)
    assert int(n_iter) == 10
    assert float(diff) > 0.0
