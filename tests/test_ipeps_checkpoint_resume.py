"""End-to-end resume tests for the 2-site iPEPS AD checkpoint hook.

Runs the optimizer, persists a mid-point, then loads the checkpoint
back via ``gs_resume=True`` and checks that the run continues from
the saved step with the saved best-seen energy intact.

Marked ``algorithm`` (not ``core``) because it exercises a real
optimizer step at ``D=2 / chi=4``, which costs a few seconds per
step.  Unit-scoped primitive tests live in
``test_ipeps_checkpoint.py``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tenax.algorithms._checkpoint import (
    checkpoint_exists,
    load_checkpoint,
)
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad


def _heisenberg_gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _base_cfg(ckpt_path, *, gs_num_steps, gs_resume):
    return iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=gs_num_steps,
        gs_checkpoint_path=ckpt_path,
        gs_checkpoint_every=1,
        gs_resume=gs_resume,
        gs_c4v=True,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )


@pytest.mark.algorithm
def test_resume_2site_continues_from_saved_step(tmp_path):
    """End-to-end: run 2 steps, then resume to 4 steps total.

    Verifies:
      * ckpt.last.pkl is written and records the last completed step
      * a resume run with the same ``gs_checkpoint_path`` advances the
        step counter rather than restarting at 0
      * the final energy from the resumed run is no worse than the
        first run (monotone within tolerance)
    """
    gate = _heisenberg_gate()
    ckpt_path = str(tmp_path / "ckpt")

    cfg_phase_a = _base_cfg(ckpt_path, gs_num_steps=2, gs_resume=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_a = optimize_gs_ad(gate, None, cfg_phase_a)
    _, _, E_a = result_a[:3]

    assert checkpoint_exists(ckpt_path), "phase A did not write a checkpoint"
    bundle = load_checkpoint(ckpt_path)
    assert bundle is not None
    assert bundle["step"] == 1, (
        f"ckpt.last should record the last completed step (1); got {bundle['step']!r}"
    )

    cfg_phase_b = _base_cfg(ckpt_path, gs_num_steps=4, gs_resume=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_b = optimize_gs_ad(gate, None, cfg_phase_b)
    _, _, E_b = result_b[:3]

    bundle_b = load_checkpoint(ckpt_path)
    assert bundle_b["step"] == 3, (
        f"resumed run should advance ckpt.last.step to 3; got {bundle_b['step']!r}"
    )

    # Optimizer is monotone in best-seen energy across resume; allow a
    # tiny tolerance for pickle round-trip float drift.
    assert E_b <= E_a + 1e-6, f"resumed run regressed: E_a={E_a!r}, E_b={E_b!r}"


@pytest.mark.algorithm
def test_resume_no_op_when_already_completed(tmp_path):
    """Resume from a fully-completed run does nothing (loop range empty).

    After running ``gs_num_steps=2``, ``ckpt.last`` has ``step=1``.
    Resume with ``gs_num_steps=2`` again gives ``start_step=2`` and
    ``range(2, 2)`` runs zero iterations — the finalize block still
    fires so a valid result is returned.
    """
    gate = _heisenberg_gate()
    ckpt_path = str(tmp_path / "ckpt")

    cfg = _base_cfg(ckpt_path, gs_num_steps=2, gs_resume=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_a = optimize_gs_ad(gate, None, cfg)
    _, _, E_a = result_a[:3]

    cfg_resume = _base_cfg(ckpt_path, gs_num_steps=2, gs_resume=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_b = optimize_gs_ad(gate, None, cfg_resume)
    _, _, E_b = result_b[:3]

    # No optimizer steps ran on the second call, so the final
    # re-evaluation simply re-uses the saved best/last params; the
    # returned energy should match the first run within fresh-CTM
    # convergence drift.
    assert abs(E_b - E_a) < 1e-3, f"E_a={E_a!r}, E_b={E_b!r}"


@pytest.mark.algorithm
def test_resume_rejects_different_hamiltonian_gate(tmp_path):
    """A silent gate swap on resume must be a fatal error.

    Phase A writes a checkpoint optimized against Heisenberg; phase B
    tries to resume against a different (Ising-like) gate with the
    same shape.  ``_optimize_gs_ad_tensor_2site`` must raise rather
    than continue with weights tuned for the wrong Hamiltonian.
    (Codex P2 review on PR #497.)
    """
    heisenberg = _heisenberg_gate()
    ckpt_path = str(tmp_path / "ckpt")

    cfg_a = _base_cfg(ckpt_path, gs_num_steps=2, gs_resume=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize_gs_ad(heisenberg, None, cfg_a)

    # Same shape, different bytes: transverse-field Ising-like (Sx⊗Sx
    # only).  Same (d,d,d,d) layout, completely different physics.
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    ising = jnp.einsum("ij,kl->ikjl", sx, sx).real
    assert ising.shape == heisenberg.shape

    cfg_b = _base_cfg(ckpt_path, gs_num_steps=4, gs_resume=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="hamiltonian gate has changed"):
            optimize_gs_ad(ising, None, cfg_b)


@pytest.mark.algorithm
def test_resume_discards_optimizer_state_on_optimizer_change(tmp_path):
    """Changing ``gs_optimizer`` across resume warns and starts the
    new optimizer's history fresh, rather than installing a saved
    L-BFGS state into Adam (or vice versa) — which would crash on the
    next ``optimizer.update``.  (Codex P2 review on PR #497.)

    Tests the L-BFGS → Adam transition: a saved Adam ``opt_state``
    cannot be a valid metric-L-BFGS state, so we must discard it on
    resume.  The optimizer must complete the resumed step without
    raising.
    """
    gate = _heisenberg_gate()
    ckpt_path = str(tmp_path / "ckpt")

    # Phase A: write checkpoint with optax-lbfgs (gs_metric_precond=False)
    cfg_a = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=2,
        gs_checkpoint_path=ckpt_path,
        gs_checkpoint_every=1,
        gs_optimizer="lbfgs",
        gs_metric_precond=False,  # optax-backed LBFGS, opt_state populated
        gs_c4v=True,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimize_gs_ad(gate, None, cfg_a)

    saved = load_checkpoint(ckpt_path)
    assert saved is not None
    assert saved["opt_state"] is not None, "phase A should have written opt_state"

    # Phase B: resume with metric-LBFGS — different optimizer state layout.
    cfg_b = iPEPSConfig(
        unit_cell="2site",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=4,
        gs_checkpoint_path=ckpt_path,
        gs_checkpoint_every=1,
        gs_resume=True,
        gs_optimizer="lbfgs",
        gs_metric_precond=True,  # changed — opt_state becomes None
        gs_c4v=True,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )

    # Must warn (about discarded optimizer history) but not raise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize_gs_ad(gate, None, cfg_b)
    discard_warnings = [
        w for w in caught if "discarding saved optimizer history" in str(w.message)
    ]
    assert discard_warnings, (
        "expected a warning about discarded optimizer history; "
        f"got: {[str(w.message) for w in caught]}"
    )


@pytest.mark.core
def test_resume_with_missing_checkpoint_raises(tmp_path):
    """``gs_resume=True`` with no checkpoint file must raise fast.

    Silent fresh-init on a typo'd / deleted / never-written
    checkpoint path would discard the user's intended long-run state
    and start overwriting the wrong directory.  (Codex P2 review on
    PR #497.)
    """
    gate = _heisenberg_gate()
    missing_path = str(tmp_path / "does_not_exist")
    cfg = _base_cfg(missing_path, gs_num_steps=2, gs_resume=True)
    with pytest.raises(FileNotFoundError, match="no checkpoint found"):
        optimize_gs_ad(gate, None, cfg)


@pytest.mark.core
def test_checkpoint_path_allows_1site_rejects_lattice():
    """1-site (incl. CG) checkpointing is now wired; generic Lattice
    multisite still raises NotImplementedError."""
    from tenax.core.lattice import Lattice, checkerboard

    gate = _heisenberg_gate()
    # 1-site no longer raises NotImplementedError on the guard. 0 steps -> returns fast.
    cfg_1site = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=0,
        gs_checkpoint_path="/tmp/ckpt_guard_1site",
        gs_c4v=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    optimize_gs_ad(gate, None, cfg_1site)  # must NOT raise NotImplementedError

    # generic Lattice multisite is still guarded
    lat = checkerboard()  # minimal valid 2-tensor lattice
    assert isinstance(lat, Lattice)
    cfg_multi = iPEPSConfig(
        unit_cell=lat,
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=1,
        gs_checkpoint_path="/tmp/ckpt_guard_lattice",
        su_init=False,
    )
    with pytest.raises(NotImplementedError, match="Lattice|multisite"):
        optimize_gs_ad(gate, None, cfg_multi)


@pytest.mark.core
def test_checkpoint_path_rejects_c4v_reference():
    """The dense C4v-reference path has no checkpoint wiring, so a checkpoint
    path on it must RAISE (not silently no-op): gs_resume would otherwise start
    fresh and discard the intended long run."""
    gate = _heisenberg_gate()
    cfg_ref = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, ctm_ad_mode="c4v_reference"),
        gs_num_steps=0,
        gs_c4v=True,
        gs_implicit_ad=True,
        gs_checkpoint_path="/tmp/ckpt_guard_c4v_ref",
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    with pytest.raises(NotImplementedError, match="C4v|Lattice"):
        optimize_gs_ad(gate, None, cfg_ref)


def test_1site_writes_checkpoint(tmp_path):
    """A plain 1-site run with gs_checkpoint_path writes ckpt.last.pkl whose
    recorded step matches the last completed optimizer step."""
    from tenax.algorithms._checkpoint import checkpoint_exists, load_checkpoint

    gate = _heisenberg_gate()
    cfg = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=2,
        gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1,
        gs_c4v=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    optimize_gs_ad(gate, None, cfg)
    assert checkpoint_exists(str(tmp_path))
    bundle = load_checkpoint(str(tmp_path))
    assert bundle["step"] == 1  # 0-indexed last of 2 steps
    assert "params" in bundle and "opt_state" in bundle
    assert bundle["cg_gates_fingerprint"] is None  # plain 1-site, no cg_gates


def test_resume_1site_continues_from_saved_step(tmp_path):
    """Run 2 steps, checkpoint; resume to 8 total; the resumed run picks up at
    step 2 and finishes with the saved step recorded as 7.

    The resume mechanism is verified by the checkpoint step counter; the
    energy is a sanity guard that the resumed run continued the saved
    trajectory rather than diverging.
    """
    gate = _heisenberg_gate()

    def cfg(nsteps, resume):
        return iPEPSConfig(
            unit_cell="1x1",
            max_bond_dim=2,
            ctm=CTMConfig(chi=4),
            gs_num_steps=nsteps,
            gs_checkpoint_path=str(tmp_path),
            gs_checkpoint_every=1,
            gs_resume=resume,
            gs_c4v=False,
            su_init=False,
            gs_conv_criterion="grad_norm",
        )

    _, _, E_phaseA = optimize_gs_ad(gate, None, cfg(2, False))  # phase A: 2 steps
    _, _, E_resumed = optimize_gs_ad(gate, None, cfg(8, True))  # resume -> 8

    from tenax.algorithms._checkpoint import load_checkpoint

    assert load_checkpoint(str(tmp_path))["step"] == 7  # 0-indexed last of 8

    # Energy sanity: the resumed best-seen energy is finite and no worse than
    # the phase-A energy (mirrors test_resume_2site_continues_from_saved_step).
    # The old `E_resumed < 0` gate flaked in CI (#692): the chi=4 random-init
    # CTM is noise-dominated, so the absolute energy plateaus near the untrained
    # value and its SIGN is BLAS/XLA-sensitive (it does not reliably cross zero
    # in 8 steps). A relative check cancels that context-dependent CTM offset;
    # the 1e-3 tolerance covers chi=4 re-evaluation drift (observed ~1e-5).
    assert jnp.isfinite(E_resumed)
    assert E_resumed <= E_phaseA + 1e-3, f"resumed regressed: {E_phaseA=} {E_resumed=}"


# ---------------------------------------------------------------------------
# Coarse-grained (cg_gates) 1-site resume tests
# ---------------------------------------------------------------------------


def _honeycomb_cg_cfg(tmp_path, *, nsteps, resume, cg_gates=None):
    """Verified-working CG 1-site checkpoint config (implicit-AD default).

    Routes to ``_optimize_gs_ad_tensor`` (the checkpoint-wired non-c4v
    path, log tag ``[iPEPS-AD:1site-tensor]``) — NOT the c4v reference
    path, which has no checkpoint wiring.  ``su_init=False`` is required
    with ``cg_gates``; the Hamiltonian placeholder is a (4,4,4,4) dummy
    (the real interaction lives in ``cg_gates``, d_eff=4 for honeycomb).

    ``cg_gates`` overrides the default honeycomb gates so the reject test
    differs from the baseline in EXACTLY that one field (every other knob
    stays in lockstep via this helper).
    """
    from tenax.algorithms.coarse_grain import honeycomb_cg_gates

    return iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=8, max_iter=20, min_iter=5),
        gs_num_steps=nsteps,
        gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1,
        gs_resume=resume,
        gs_c4v=False,
        su_init=False,
        cg_gates=honeycomb_cg_gates() if cg_gates is None else cg_gates,
        gs_conv_criterion="grad_norm",
    )


@pytest.mark.slow
def test_resume_cg_1site_continues_from_saved_step(tmp_path):
    """Coarse-grained (cg_gates) 1-site run checkpoints and resumes; the saved
    bundle records a non-None cg_gates fingerprint and resume continues."""
    from tenax.algorithms._checkpoint import load_checkpoint

    dummy = jnp.zeros((4, 4, 4, 4))
    optimize_gs_ad(dummy, None, _honeycomb_cg_cfg(tmp_path, nsteps=2, resume=False))
    b = load_checkpoint(str(tmp_path))
    assert b["step"] == 1
    assert b["cg_gates_fingerprint"] is not None

    _, _, E = optimize_gs_ad(
        dummy, None, _honeycomb_cg_cfg(tmp_path, nsteps=4, resume=True)
    )
    assert load_checkpoint(str(tmp_path))["step"] == 3
    assert E is not None  # finished without error (don't over-assert convergence)


@pytest.mark.slow
def test_resume_rejects_different_cg_gates(tmp_path):
    """Resuming a CG run against perturbed cg_gates is a fatal mismatch
    (the dummy hamiltonian gate is unchanged, isolating the cg_gates check)."""
    from tenax.algorithms.coarse_grain import honeycomb_cg_gates

    dummy = jnp.zeros((4, 4, 4, 4))
    optimize_gs_ad(dummy, None, _honeycomb_cg_cfg(tmp_path, nsteps=2, resume=False))

    # Identical config EXCEPT the coarse-grained gates (J=2.0 vs default J=1.0),
    # so the only thing that can trigger the mismatch is the cg_gates check.
    cfg_other = _honeycomb_cg_cfg(
        tmp_path, nsteps=4, resume=True, cg_gates=honeycomb_cg_gates(J=2.0)
    )
    with pytest.raises(ValueError, match="cg_gates"):
        optimize_gs_ad(dummy, None, cfg_other)


@pytest.mark.slow
def test_resume_rejects_plain_to_cg(tmp_path):
    """A PLAIN 1-site checkpoint resumed with cg_gates set is a fatal mismatch.

    The saved bundle has ``cg_gates_fingerprint=None`` while the live config has
    a CG fingerprint; the full-inequality resume check must reject this (else
    plain-tensor params would be evaluated through the CG path and crash). A
    matched (4,4,4,4) dummy gate is shared so the GATE fingerprint agrees and the
    CG check (not the gate check) is what fires.
    """
    import jax

    g = jax.random.normal(jax.random.PRNGKey(0), (16, 16))
    shared = jnp.asarray(0.5 * (g + g.T)).reshape(4, 4, 4, 4)  # d_phys=4 == d_eff

    plain_cfg = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=20, min_iter=5),
        gs_num_steps=2,
        gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1,
        gs_c4v=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    optimize_gs_ad(shared, None, plain_cfg)

    from tenax.algorithms._checkpoint import load_checkpoint

    assert load_checkpoint(str(tmp_path))["cg_gates_fingerprint"] is None
    # resume the SAME gate but now with cg_gates -> fatal CG-mismatch
    cg_cfg = _honeycomb_cg_cfg(tmp_path, nsteps=4, resume=True)
    with pytest.raises(ValueError, match="cg_gates|parameterization"):
        optimize_gs_ad(shared, None, cg_cfg)
