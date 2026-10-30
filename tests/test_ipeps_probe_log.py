"""``gs_log_probes``: one log line and one history record per line-search probe.

A Hager-Zhang line search once accepted a step whose phi probe reported an
energy below phi0 while the converged energy there was higher.  The probe's
seed env and sweep count could not be replayed.  ``gs_log_probes`` prints
every phi / dphi probe so the next false accept can be traced.

Mechanism tests: a scripted line search calls phi / dphi at a known alpha and
the log is read back (stdout and ``history["probes"]``).  D=2, chi=4, one
optimizer step; convergence is not tested.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_energy_ad as _cea
import tenax.algorithms._line_search as _ls
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

LINE_RE = re.compile(
    r"^\[iPEPS-AD:(?P<tag>[\w-]+)\] step (?P<step>\d+) probe (?P<kind>phi|dphi) "
    r"alpha=(?P<alpha>nan|[-+]?\d\.\d{3}e[-+]\d{2}) "
    r"E=(?P<E>inf|-?\d+\.\d{10}) "
    r"sweeps=(?P<sweeps>\d+|n/a) "
    r"sv_diff=(?P<sv>n/a|[-+]?\d\.\d{3}e[-+]\d{2}|nan|inf) "
    r"conv=(?P<conv>True|False|n/a) "
    r"seed=(?P<seed>[0-9a-f]{8}|cold)$"
)
TAGS = {"1x1": "1site-tensor", "2site": "2site-tensor", "lattice": "multisite-tensor"}
UNIT_CELLS = list(TAGS)
KEYS = {"step", "kind", "alpha", "energy", "sweeps", "sv_diff", "converged", "seed"}


def _gate():
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


def _run(monkeypatch, unit_cell, script=None, **cfg_kw):
    """One optimizer step.  ``script(phi, dphi)`` replaces the HZ line search."""
    if script is not None:

        def spy_hz(phi, dphi, phi0, slope, **kw):
            script(phi, dphi)
            return 0.0, phi0, False

        monkeypatch.setattr(_ls, "hager_zhang_line_search", spy_hz)
    cell, init = _setup(unit_cell)
    kw = dict(
        unit_cell=cell,
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=4, max_iter=300, min_iter=1, conv_tol=1e-8, on_unconverged="warn"
        ),
        gs_num_steps=1,
        gs_line_search_method="hager_zhang",
        su_init=False,
        gs_conv_criterion="grad_norm",
        return_history=True,
    )
    kw.update(cfg_kw)
    out = _opt.optimize_gs_ad(_gate(), init, iPEPSConfig(**kw))
    return out[-1]


def _probe_lines(text):
    return [ln for ln in text.splitlines() if "probe " in ln and "iPEPS-AD:" in ln]


def _phi_dphi(phi, dphi):
    phi(5e-3)
    dphi(5e-3)


# --- the helpers ------------------------------------------------------------


def test_seed_hash_is_stable_and_content_sensitive():
    env = {"a": jnp.arange(6.0), "b": [jnp.ones((2, 2))]}
    same = {"a": jnp.arange(6.0), "b": [jnp.ones((2, 2))]}
    other = {"a": jnp.arange(6.0) + 1e-9, "b": [jnp.ones((2, 2))]}
    h = _opt._env_seed_hash(env)
    assert re.fullmatch(r"[0-9a-f]{8}", h)
    assert h == _opt._env_seed_hash(same)
    assert h != _opt._env_seed_hash(other)


def test_seed_hash_of_no_env_is_cold():
    assert _opt._env_seed_hash(None) == "cold"


def test_format_matches_the_spec_and_prints_na_without_diagnostics():
    full = _opt._format_probe_line(
        "2site-tensor", 3, "phi", 5e-3, -0.5, 12, 1e-9, True, "ab12cd34"
    )
    assert full == (
        "[iPEPS-AD:2site-tensor] step 3 probe phi alpha=5.000e-03 E=-0.5000000000 "
        "sweeps=12 sv_diff=1.000e-09 conv=True seed=ab12cd34"
    )
    assert LINE_RE.match(full)
    na = _opt._format_probe_line(
        "1site-tensor", 1, "dphi", float("nan"), float("inf"), None, None, None, "cold"
    )
    assert na == (
        "[iPEPS-AD:1site-tensor] step 1 probe dphi alpha=nan E=inf "
        "sweeps=n/a sv_diff=n/a conv=n/a seed=cold"
    )
    assert LINE_RE.match(na)


# --- the optimizers -----------------------------------------------------------


@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_flag_on_logs_every_probe_and_history(monkeypatch, capsys, unit_cell):
    hist = _run(monkeypatch, unit_cell, _phi_dphi, gs_log_probes=True)
    lines = _probe_lines(capsys.readouterr().out)
    ms = [LINE_RE.match(ln) for ln in lines]
    assert all(ms), lines
    kinds = [m["kind"] for m in ms]
    assert "phi" in kinds and "dphi" in kinds, lines
    assert all(m["tag"] == TAGS[unit_cell] for m in ms)
    for m in ms:
        if m["kind"] == "phi" or m["alpha"] != "nan":
            # The scripted probes carry their alpha; HZ's own bookkeeping
            # (top-of-step value_and_grad) logs nothing.
            assert m["alpha"] == "5.000e-03", m.string
    phi = next(m for m in ms if m["kind"] == "phi")
    dphi = next(m for m in ms if m["kind"] == "dphi")
    # phi reports the CTM loop's own sweep count; dphi reads the implicit-AD
    # forward diagnostics: both are real numbers here.
    assert phi["sweeps"] != "n/a" and dphi["sweeps"] != "n/a"
    assert phi["conv"] in ("True", "False") and dphi["conv"] in ("True", "False")
    assert phi["seed"] != "cold"  # the step has an env to start from
    # dphi starts from the env phi just converged (#502), not the step env.
    assert dphi["seed"] not in ("cold", phi["seed"])
    probes = hist["probes"]
    assert len(probes) == len(ms)
    assert all(set(p) == KEYS for p in probes)
    assert [p["kind"] for p in probes] == kinds
    assert all(p["alpha"] == pytest.approx(5e-3) for p in probes)
    assert all(p["step"] == 1 for p in probes)
    assert float(phi["E"]) == pytest.approx(
        probes[kinds.index("phi")]["energy"], abs=1e-9
    )


@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_flag_off_prints_nothing_and_adds_no_key(monkeypatch, capsys, unit_cell):
    hist = _run(monkeypatch, unit_cell, _phi_dphi)
    assert _probe_lines(capsys.readouterr().out) == []
    assert "probes" not in hist


def test_flag_does_not_change_the_result(monkeypatch):
    off = _run(monkeypatch, "1x1")
    on = _run(monkeypatch, "1x1", gs_log_probes=True)
    assert on["energies"] == off["energies"]
    assert len(on["probes"]) > 0  # the flag did something


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_rejected_phi_probe_is_logged_as_inf(monkeypatch, capsys, unit_cell):
    monkeypatch.setattr(_opt, "_site3_reject", lambda info, cfg: True)
    hist = _run(monkeypatch, unit_cell, _phi_dphi, gs_log_probes=True)
    phis = [p for p in hist["probes"] if p["kind"] == "phi"]
    assert phis and all(p["energy"] == float("inf") for p in phis)
    lines = [ln for ln in _probe_lines(capsys.readouterr().out) if "probe phi" in ln]
    assert lines and all(" E=inf " in ln for ln in lines)
    assert all(LINE_RE.match(ln) for ln in lines)


@pytest.mark.parametrize("unit_cell", UNIT_CELLS)
def test_missing_dphi_diagnostics_print_na(monkeypatch, capsys, unit_cell):
    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", lambda: {})
    hist = _run(monkeypatch, unit_cell, _phi_dphi, gs_log_probes=True)
    dphis = [p for p in hist["probes"] if p["kind"] == "dphi"]
    assert dphis and all(p["sweeps"] is None for p in dphis)
    lines = [ln for ln in _probe_lines(capsys.readouterr().out) if "probe dphi" in ln]
    assert lines and all("sweeps=n/a sv_diff=n/a conv=n/a" in ln for ln in lines)
    assert all(LINE_RE.match(ln) for ln in lines)


@pytest.mark.parametrize("unit_cell", ["1x1", "2site"])
def test_backtracking_probes_have_nan_alpha(monkeypatch, capsys, unit_cell):
    hist = _run(
        monkeypatch,
        unit_cell,
        None,
        gs_line_search_method="armijo",
        gs_log_probes=True,
    )
    probes = hist["probes"]
    assert probes and all(p["kind"] == "phi" for p in probes)
    assert all(np.isnan(p["alpha"]) for p in probes)
    lines = _probe_lines(capsys.readouterr().out)
    assert lines and all(" alpha=nan " in ln for ln in lines)


def test_config_default_is_off():
    assert iPEPSConfig().gs_log_probes is False
