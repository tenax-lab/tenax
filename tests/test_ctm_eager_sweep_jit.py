"""Eager #1098 layout sweeps jit their static-shape pieces; the result is unchanged."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate

jax.config.update("jax_enable_x64", True)

D = 2
CHI = 6  # U(1) chi=6 cuts the eager C1 spectrum in a clean gap (#1098 test)


def _sweep_kwargs(chi):
    return dict(
        neighbors=CHECKERBOARD_NEIGHBORS,
        recipe="2x2",
        device_mesh=None,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        chunk_size=None,
        chi=chi,
    )


@pytest.fixture(scope="module")
def sites():
    A0, B0 = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


def _sweeps(sites, chi, n, enabled, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", enabled)
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in sites.items()}
    for _ in range(n):
        envs, _, _ = cpl._eager_ctm_sweep(sites, envs, **_sweep_kwargs(chi))
        envs = cpl._charge_sorted_envs(envs)
    return envs


def _assert_same_envs(a, b, atol=1e-10):
    assert cpl._chi_layout(a) == cpl._chi_layout(b)
    for c in a:
        for f in a[c]._fields:
            x = np.asarray(getattr(a[c], f).todense())  # chi=6, D^2=4: small
            y = np.asarray(getattr(b[c], f).todense())
            np.testing.assert_allclose(x, y, atol=atol, err_msg=f"{c}.{f}")


def test_jitted_eager_sweeps_match_plain_eager_sweeps(sites, monkeypatch):
    plain = _sweeps(sites, CHI, 3, False, monkeypatch)
    jitted = _sweeps(sites, CHI, 3, True, monkeypatch)
    _assert_same_envs(plain, jitted)


def test_eager_sweep_runs_the_absorb_moves_under_jit(sites, monkeypatch):
    seen = []
    raw_jitted = ej._jitted

    def spy(fn, static_argnames):
        seen.append(fn.__name__)
        return raw_jitted(fn, static_argnames)

    monkeypatch.setattr(ej, "_jitted", spy)
    _sweeps(sites, CHI, 1, True, monkeypatch)
    for d in ("left", "right", "top", "bottom"):
        assert f"_ctm_tensor_absorb_{d}_2plaq" in seen
    assert "_build_double_layer_tensor" in seen
