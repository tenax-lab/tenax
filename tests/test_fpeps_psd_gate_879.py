"""#879: ``fpeps()`` must refuse (NaN) an energy built from an invalid RDM.

The fermionic energy ``fpeps()`` returns is ``tr(rho H)`` summed over bonds.
That is an expectation value -- bounded by the spectrum of ``H`` -- **only when
every** ``rho`` **is a density matrix** (finite, unit-trace, PSD).  On a
degenerate or collapsed environment an RDM stops being one, and
:func:`~tenax.algorithms._ctm_diagnostics.check_rdm` warns (#845 trace-collapse,
#848 non-finite, #854 non-PSD) -- but the energy path *warned and returned the
number anyway*.  A caller who did not read stderr got a finite, wrong energy
reported as if it were valid; this is the mechanism behind the #854 ``E = +0.759``
for two ``S.S`` bonds whose attainable maximum is ``+0.5``.

The fix is an opt-in ``nan_on_invalid_rdm`` gate on
:func:`compute_energy_split_ctm_tensor_2site` /
:func:`compute_energy_split_ctm_tensor_multisite`, which ``fpeps()`` sets: when a
bond's concrete RDM fails ``check_rdm``, the returned energy is ``NaN`` rather
than a plausible-looking lie.  It is **default-off** and only inspected on the
concrete (non-tracer) path, so the AD/optimizer path and every other caller are
bit-for-bit unchanged.

The PSD arm has a tolerance, ``psd_tol``.  ``fpeps()`` loosens it
(``_FPEPS_RDM_PSD_TOL = 1e-2``): a low-chi CTM leaves a small negativity that is
convergence noise, not a collapse (~1e-3 of the spectral radius at chi=8, V=1/V=2),
so those runs still return a number and only *gross* non-PSD is refused.  The
non-finite and trace-collapse arms keep their own tolerances, so a collapsed
environment is refused at *any* ``psd_tol`` -- loosening the PSD arm never weakens
them.

**Why the fixture zeroes a corner rather than finding a non-PSD run.**  The
headline case in production is #854 (a negative eigenvalue), but a *deterministic*
non-PSD RDM is not a stable CI hook -- which corner basis a live CTM lands on
shifts with the JAX/LAPACK build (see ``test_ctm_symmetric_rdm_positive_853``).
Zeroing ``env_A.C1`` reproduces the same *observable* the gate keys on -- a bond
whose RDM fails ``check_rdm`` while the energy stays finite -- deterministically
on every build (the #845 trace-collapse mechanism, as in
``test_rdm_validity_guard``).  The gate treats all three ``check_rdm`` failure
classes identically, so exercising it through the reproducible one is faithful.

Reproduced at D=2, V=4, chi=4 (the ``test_fermionic_ctm_energy_anchor_879``
fixture): the valid cell reads ``E = -V`` with both RDMs valid; zeroing ``C1``
leaves *both* RDMs invalid yet the ungated energy returns a finite ``-2.0``.

**Mutation anchors.**  Deleting the gate (returning the number regardless) makes
``test_the_gate_nans_an_invalid_rdm_energy`` read that finite ``-2.0`` and fail
``isnan``; a gate that always NaNs fails ``test_the_gate_is_finite_on_a_valid_env``
at ``E = -V``; dropping ``nan_on_invalid_rdm=True`` from ``fpeps()``'s call site
makes ``test_fpeps_opts_into_the_gate`` see the default ``False`` and fail.
Collapsing ``psd_tol`` back to the strict default in ``fpeps()`` fails that same
test's tolerance assertion; ignoring ``psd_tol`` inside the gate (always strict)
fails ``test_the_gate_tolerance_separates_mild_from_gross_non_psd`` (the loose
call would NaN), and routing the loose tolerance into the trace/finite arms fails
that test's strict-refuses half and ``test_the_gate_nans_an_invalid_rdm_energy``'s
loose-still-NaN assertion.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import fermionic_ipeps
from tenax.algorithms._ctm_diagnostics import CollapsedRDMError, check_rdm
from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
from tenax.algorithms._split_ctm_tensor_energy import (
    _rdm1x2_split_tensor_2site,
    _rdm2x1_split_tensor_2site,
    compute_energy_split_ctm_tensor_2site,
)
from tenax.algorithms._tensor_utils import scale_bond_axis
from tenax.algorithms.fermionic_ipeps import (
    FPEPSConfig,
    _fpeps_simple_update,
    _initialize_fpeps,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_simple_update import _to_physical_pair
from tenax.core.tensor import SymmetricTensor

_V = 4.0


def _gate_with_mu(cfg: FPEPSConfig, mu: float) -> SymmetricTensor:
    """``spinless_fermion_gate`` + a diagonal ``-(mu/4)(n_i + n_j)`` per bond."""
    g = spinless_fermion_gate(cfg)
    h = np.array(g.todense()).reshape(4, 4)
    h = h + np.diag(-(mu / 4.0) * np.array([0.0, 1.0, 1.0, 2.0]))
    return SymmetricTensor.from_dense(jnp.array(h.reshape(2, 2, 2, 2)), g.indices)


@pytest.fixture(scope="module")
def valid_cell():
    """A converged, PSD fully-polarised CDW cell: ``(A, B, env_A, env_B, gate)``.

    The same seeded-CDW fixture as ``test_fermionic_ctm_energy_anchor_879`` -- its
    ungated energy is ``-V`` and both RDMs are PSD, so it is the baseline the gate
    must leave untouched and the source of the env the invalid case corrupts.
    """
    cfg = FPEPSConfig(D=2, t=1.0, V=_V, dt=0.05)
    gate = _gate_with_mu(cfg, mu=2.0 * _V)
    A0 = _initialize_fpeps(cfg, jax.random.PRNGKey(3))
    a = scale_bond_axis(A0, "phys", jnp.array([4.0, 0.25]))
    b = scale_bond_axis(A0, "phys", jnp.array([0.25, 4.0]))
    A, B, lam = _fpeps_simple_update(a, gate, max_D=cfg.D, dt=cfg.dt, steps=20, B=b)
    A, B = _to_physical_pair(A, B, lam)
    env_A, env_B = ctm_split_tensor_2site(A, B, chi=4, max_iter=30, conv_tol=1e-10)
    return A, B, env_A, env_B, gate


def _rdm_is_invalid(rdm) -> bool:
    try:
        check_rdm(rdm, context="test", strict=True)
        return False
    except CollapsedRDMError:
        return True


def test_the_gate_nans_an_invalid_rdm_energy(valid_cell):
    """Zeroing ``C1`` invalidates the bond RDMs; the gate must NaN the energy,
    while the default (ungated) path still returns the finite lie it always has.
    """
    A, B, env_A, env_B, gate = valid_cell
    dead = env_A._replace(C1=env_A.C1 * 0.0)

    # Regime pin: the corruption must genuinely invalidate an RDM, or a NaN
    # below would prove nothing about the gate.
    rdm_h = _rdm2x1_split_tensor_2site(A, B, dead, env_B)
    rdm_v = _rdm1x2_split_tensor_2site(A, B, dead, env_B)
    assert _rdm_is_invalid(rdm_h) and _rdm_is_invalid(rdm_v), (
        "zeroing C1 did not invalidate the bond RDMs -- the fixture drifted out "
        "of the regime this test needs (check_rdm no longer fires)"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the check_rdm warnings
        e_off = float(
            compute_energy_split_ctm_tensor_2site(A, B, dead, env_B, gate, d=2)
        )
        e_on = float(
            compute_energy_split_ctm_tensor_2site(
                A, B, dead, env_B, gate, d=2, nan_on_invalid_rdm=True
            )
        )
        # A collapse is *gross*: it must be refused even at fpeps()'s loose
        # PSD tolerance, because C1=0 kills the trace (|tr - 1| = 1) and the
        # trace-collapse arm has its own tolerance, independent of psd_tol.
        e_on_loose = float(
            compute_energy_split_ctm_tensor_2site(
                A, B, dead, env_B, gate, d=2, nan_on_invalid_rdm=True, psd_tol=1e-2
            )
        )

    # The ungated default is unchanged: it still hands back a finite, wrong
    # number (this is the #879 gap the gate closes, and the regression guard
    # that the gate is genuinely opt-in).
    assert jnp.isfinite(e_off), (
        f"the ungated energy changed: expected the finite pre-#879 value, got "
        f"{e_off} -- the gate must not fire unless nan_on_invalid_rdm=True"
    )
    # With the gate on, the invalid RDM must poison the energy to NaN -- at the
    # strict default and, since this is a collapse not a mild negativity, at the
    # loose tolerance too.
    assert jnp.isnan(e_on), (
        f"nan_on_invalid_rdm=True returned {e_on} on an invalid-RDM environment; "
        f"an energy built from a non-density-matrix is not bounded by physics and "
        f"must be refused, not reported (#879)"
    )
    assert jnp.isnan(e_on_loose), (
        f"a trace-collapse leaked through the loose psd_tol=1e-2 gate ({e_on_loose}); "
        f"loosening the PSD arm must not weaken the trace-collapse arm (#845)"
    )


def test_the_gate_is_finite_on_a_valid_env(valid_cell):
    """The gate must not false-positive: on the PSD baseline it returns ``-V``."""
    A, B, env_A, env_B, gate = valid_cell
    e = float(
        compute_energy_split_ctm_tensor_2site(
            A, B, env_A, env_B, gate, d=2, nan_on_invalid_rdm=True
        )
    )
    assert jnp.isfinite(e) and e == pytest.approx(-_V, abs=1e-3), (
        f"the gate NaN'd (or moved) a valid CDW energy: {e} vs -V = {-_V}; "
        f"check_rdm must pass on a PSD environment"
    )


def test_the_gate_tolerance_separates_mild_from_gross_non_psd(valid_cell, monkeypatch):
    """The ``psd_tol`` knob draws the mild/gross line: a small negativity is
    refused at the strict default but tolerated at fpeps()'s loose tolerance.

    This is the behaviour ``fpeps()`` relies on -- a low-chi CTM leaves a small
    (~1e-3) negativity that is convergence noise, so fpeps() loosens ``psd_tol``
    to keep returning a number there while still refusing a gross collapse.  A
    synthetic RDM with a controlled negativity of 2e-3 (min eig -0.001 over a
    spectral radius 0.5, trace exactly 1) isolates the PSD arm deterministically,
    without depending on which basin a live CTM lands on.
    """
    import numpy as _np

    from tenax.algorithms import _split_ctm_tensor_energy as energy_mod
    from tenax.algorithms._ctm_diagnostics import _as_rdm_matrix, _rdm_negativity

    A, B, env_A, env_B, gate = valid_cell
    mild = jnp.asarray(_np.diag([0.5, 0.3, 0.201, -0.001]).reshape(2, 2, 2, 2))

    # Regime pin: the synthetic negativity must sit strictly between the strict
    # default (1e-8) and fpeps()'s loose tolerance (1e-2), or the test proves
    # nothing about where the line falls.
    neg, _ = _rdm_negativity(_as_rdm_matrix(mild))
    assert 1e-8 < neg < 1e-2, (
        f"synthetic negativity {neg:.3g} is not in the (strict 1e-8, loose 1e-2) "
        f"band -- the mild/gross separation this test asserts is vacuous"
    )

    monkeypatch.setattr(energy_mod, "_rdm2x1_split_tensor_2site", lambda *a, **k: mild)
    monkeypatch.setattr(energy_mod, "_rdm1x2_split_tensor_2site", lambda *a, **k: mild)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e_strict = float(
            energy_mod.compute_energy_split_ctm_tensor_2site(
                A, B, env_A, env_B, gate, d=2, nan_on_invalid_rdm=True
            )
        )  # psd_tol=None -> strict RDM_PSD_TOL (1e-8)
        e_loose = float(
            energy_mod.compute_energy_split_ctm_tensor_2site(
                A, B, env_A, env_B, gate, d=2, nan_on_invalid_rdm=True, psd_tol=1e-2
            )
        )

    assert jnp.isnan(e_strict), (
        f"the strict 1e-8 gate must refuse a {neg:.3g} negativity, got {e_strict}"
    )
    assert jnp.isfinite(e_loose), (
        f"the loose 1e-2 gate (fpeps()'s default) must tolerate a {neg:.3g} "
        f"negativity, got {e_loose}"
    )


def test_fpeps_opts_into_the_gate(monkeypatch):
    """``fpeps()`` must pass ``nan_on_invalid_rdm=True`` *and* the loose
    ``_FPEPS_RDM_PSD_TOL`` to the energy fn.

    Spies on the energy call and runs a minimal fpeps; the return value is
    irrelevant (a tiny env may itself be invalid), only the kwargs are asserted.
    ``fpeps()`` imports the energy fn lazily (``from ... import ...`` inside the
    function body), so the spy patches the *source* module it resolves against,
    not the ``fermionic_ipeps`` namespace.
    """
    from tenax.algorithms import _split_ctm_tensor_energy as energy_mod
    from tenax.algorithms._ctm_diagnostics import RDM_PSD_TOL

    recorded: dict = {}
    real = energy_mod.compute_energy_split_ctm_tensor_2site

    def spy(*args, **kwargs):
        recorded["nan_on_invalid_rdm"] = kwargs.get("nan_on_invalid_rdm", False)
        recorded["psd_tol"] = kwargs.get("psd_tol", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(energy_mod, "compute_energy_split_ctm_tensor_2site", spy)
    cfg = FPEPSConfig(
        D=2, t=1.0, V=1.0, dt=0.05, num_imaginary_steps=1, ctm_chi=2, ctm_max_iter=3
    )
    gate = spinless_fermion_gate(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fermionic_ipeps.fpeps(gate, cfg)

    assert recorded.get("nan_on_invalid_rdm") is True, (
        "fpeps() did not opt into the RDM validity gate -- it must call the "
        "energy fn with nan_on_invalid_rdm=True so a non-PSD/collapsed "
        "environment yields NaN, not a plausible-looking wrong number (#879)"
    )
    # And it must loosen the PSD arm, or low-chi runs (with ~1e-3 convergence
    # negativity) would NaN instead of returning a number.
    assert recorded.get("psd_tol") == fermionic_ipeps._FPEPS_RDM_PSD_TOL, (
        f"fpeps() passed psd_tol={recorded.get('psd_tol')}, not the loose "
        f"_FPEPS_RDM_PSD_TOL={fermionic_ipeps._FPEPS_RDM_PSD_TOL}; a low-chi CTM's "
        f"small negativity is convergence noise, not a collapse"
    )
    assert fermionic_ipeps._FPEPS_RDM_PSD_TOL > RDM_PSD_TOL, (
        "the fpeps() tolerance must be looser than the strict #854 gate, or the "
        "loosening is a no-op"
    )
