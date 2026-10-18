"""Opt-in joint-VJP chain rule (``CTMConfig.adjoint_chain_rule="joint"``).

The implicit-AD backward's chain rule needs ``J_params^T λ``, the params-VJP
of the CTM sweep, which ``_jit_chain_rule`` compiles as its own program -- the
largest single compile at D=3 (2,604 s on GPU).  Under ``"joint"`` the adjoint
solve's matvec is one VJP over ``(params, env)`` and yields that term itself.
These tests pin gradient parity with ``"separate"`` on every backward path and
that the separate chain rule is never compiled.  Spec:
docs/superpowers/specs/2026-10-18-joint-vjp-chain-rule-design.md
"""

from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms import _ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

#: One gate object: the gate fingerprint is part of the dispatch cache key.
_GATE = heisenberg_gate()

#: Same fixture as tests/test_shared_chain_rule_bwd.py: at chi=4 the fused
#: loop solves seed 0 and its divergence guard fires on seed 2.  Asserted per
#: call, not assumed.
_FUSED_SEED, _FALLBACK_SEED = 0, 2


@pytest.fixture
def fresh_vjp_cache():
    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


def _site(seed=0, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((2, 2, 2, 2, 2))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A0 = A0 + 0.1j * rng.standard_normal((2, 2, 2, 2, 2))
    A0 = jnp.asarray(A0, dtype=dtype)
    return A0 / jnp.linalg.norm(A0)


def _energy(arrs, mode, *, method="fixed_point", tol=1e-11, neighbors=None, **kw):
    if not isinstance(arrs, (tuple, list)):
        arrs = (arrs,)
    neighbors = neighbors or SINGLE_SITE_NEIGHBORS
    coords = sorted(neighbors)
    return jnp.real(
        ctm_energy_implicit(
            {c: _wrap_as_dense_tensor(a) for c, a in zip(coords, arrs)},
            neighbors,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=method,
            gmres_tol=tol,
            adjoint_chain_rule=mode,
            **kw,
        )
    )


def _grad(arrs, mode, **kw):
    return jax.grad(lambda a: _energy(a, mode, **kw))(arrs)


# ----------------------------------------------------------------- config


def test_default_is_separate():
    assert CTMConfig().adjoint_chain_rule == "separate"


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="adjoint_chain_rule"):
        CTMConfig(adjoint_chain_rule="fused")


@pytest.mark.usefixtures("fresh_vjp_cache")
def test_the_modes_get_separate_cache_entries():
    A0 = _site(_FUSED_SEED)
    _energy(A0, "separate")
    _energy(A0, "joint")
    keys = list(cea._VJP_CACHE)
    assert len(keys) == 2
    assert {"separate", "joint"} <= {p for k in keys for p in k if isinstance(p, str)}
