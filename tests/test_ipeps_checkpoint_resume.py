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
def test_checkpoint_path_rejects_non_2site_paths():
    """1-site and multisite paths must raise NotImplementedError when a
    checkpoint path is requested (only 2-site is wired in this PR)."""
    gate = _heisenberg_gate()
    cfg_1site = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=1,
        gs_checkpoint_path="/tmp/should_not_be_used",
        gs_c4v=True,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    with pytest.raises(NotImplementedError, match="2site"):
        optimize_gs_ad(gate, None, cfg_1site)
