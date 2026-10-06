"""Hager-Zhang phi probes warm-start CTM from the step's env, not the last probe's.

``loss_fn_fwd`` writes each probe's converged env to the env cache so that the
``dphi`` probe at the same alpha starts from it (#502).  The next ``phi`` probe
read that cache too, so the warm start followed HZ's probe sequence, not the
line through alpha = 0.  On 2-site D=3 Heisenberg (chi=16) at a stall state,
HZ bisected down from alpha = 1 and every probe at alpha <= 2.4e-4 started
from an env on another CTM fixed point: it did not converge in 100 sweeps and
was rejected as +inf (#1059), while a warm start from the alpha = 0 env
converged in 10 sweeps to the decrease that dphi0 predicts.

Mechanism tests: a scripted line search calls phi at several alphas and the
spied CTM records the ``env_init`` each forward receives.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_energy_ad as _cea
import tenax.algorithms._ctm_python_loop as _cpl
import tenax.algorithms._line_search as _ls
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


def _heisenberg_gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _rand(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))
    return jnp.asarray(a / np.linalg.norm(a))


def _setup(unit_cell):
    if unit_cell == "lattice":
        from tenax.core.lattice import checkerboard

        return checkerboard(), None
    if unit_cell == "2site":
        return "2site", (_rand(0), _rand(1))
    return "1x1", _rand(0)


def _run(monkeypatch, unit_cell, script):
    """One optimizer step whose line search is ``script(phi, dphi)``.

    Returns ``(tag, env_init, envs)`` for every CTM forward run inside the
    script, in call order; ``tag`` is ``"phi"`` or ``"dphi"``.
    """
    real_cpl = _cpl.python_loop_ctm_converge
    real_sigma = _cea._sigma_gauged_ctm_converge
    current = {"tag": None}
    calls = []

    def spy_cpl(*a, **k):
        envs, info = real_cpl(*a, **k)
        if current["tag"] is not None:
            calls.append((current["tag"], k.get("env_init"), envs))
        return envs, info

    def spy_sigma(*a, **k):  # the implicit-AD (gradient) forward
        out = real_sigma(*a, **k)
        if current["tag"] is not None:
            calls.append((current["tag"], k.get("env_init"), out[0]))
        return out

    def spy_hz(phi, dphi, phi0, slope, **kw):
        def tagged(tag, fn):
            def run(alpha):
                current["tag"] = tag
                try:
                    return fn(alpha)
                finally:
                    current["tag"] = None

            return run

        script(tagged("phi", phi), tagged("dphi", dphi))
        return 0.0, phi0, False

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", spy_cpl)
    monkeypatch.setattr(_cea, "_sigma_gauged_ctm_converge", spy_sigma)
    monkeypatch.setattr(_ls, "hager_zhang_line_search", spy_hz)
    cell, init = _setup(unit_cell)
    cfg = iPEPSConfig(
        unit_cell=cell,
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=300, min_iter=1, conv_tol=1e-8, on_unconverged="warn"
        ),
        gs_num_steps=1,
        gs_line_search_method="hager_zhang",
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    _opt.optimize_gs_ad(_heisenberg_gate(), init, cfg)
    return calls


UNIT_CELLS = ["1x1", "2site", "lattice"]


@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_phi_probes_start_from_the_step_env(monkeypatch, unit_cell):
    def script(phi, dphi):
        for alpha in (1e-2, 5e-3, 2.5e-3):  # a bisection toward 0
            phi(alpha)

    calls = _run(monkeypatch, unit_cell, script)
    phis = [c for c in calls if c[0] == "phi"]
    assert len(phis) == 3, calls
    seeds = [env_init for _, env_init, _ in phis]
    returned = [envs for _, _, envs in phis]
    # The regime: the step has an env to start from.
    assert seeds[0] is not None
    # Every probe starts from that one env, never from an earlier probe's.
    assert all(s is seeds[0] for s in seeds)
    assert all(s is not r for s in seeds for r in returned)


@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_dphi_still_reuses_phi_env_at_the_same_alpha(monkeypatch, unit_cell):
    # #502: the dphi forward at alpha warm-starts from the phi env at alpha.
    def script(phi, dphi):
        phi(5e-3)
        dphi(5e-3)

    calls = _run(monkeypatch, unit_cell, script)
    phi_envs = [envs for tag, _, envs in calls if tag == "phi"]
    dphi_seeds = [env_init for tag, env_init, _ in calls if tag == "dphi"]
    assert len(phi_envs) == 1 and dphi_seeds, calls
    assert dphi_seeds[0] is phi_envs[0]
