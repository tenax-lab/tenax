"""The fused fixed-point backward and its GMRES fallback share one chain rule.

The chain rule -- the params-VJP of the CTM sweep -- is most of the implicit
backward's compile.  It used to be compiled inside ``_jit_fused_fixed_point_bwd``
and again as ``_jit_chain_rule`` for the eager-GMRES fallback, so the first
fallback paid the compile a second time: 2 h 56 min at D=3 chi=12 on CPU.  Now
the fused program returns ``lam`` only and both paths call ``_jit_chain_rule``.

Pinned here by JAX's own compile events, on one cache entry driven down both
paths: the fused path must compile ``_jit_chain_rule``, and a later fallback
must not compile another.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps import _wrap_as_dense_tensor, heisenberg_gate

_TRACED: list[str] = []
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _TRACED.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/jaxpr_trace_duration"
        else None
    )
)

#: One gate object: ``id(gate)`` is part of the dispatch cache key.
_GATE = heisenberg_gate()


def _energy(A_arr, adjoint_method):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)},
            SINGLE_SITE_NEIGHBORS,
            _GATE,
            recipe="2x2",
            chi=4,
            max_iter=60,
            min_iter=8,
            conv_tol=1e-10,
            adjoint_method=adjoint_method,
        )
    )


def _site(seed=0):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


#: On this configuration the fused Neumann loop solves seed 0 (19 iterations)
#: and its divergence guard fires on seed 2 (7 iterations), so the two seeds
#: drive the two paths of ONE ``_VJP_CACHE`` entry.  Measured, not assumed:
#: both premises are asserted per call.
_FUSED_SEED, _FALLBACK_SEED = 0, 2


@pytest.fixture
def _fresh_vjp_cache():
    """An empty ``_VJP_CACHE`` for this test, restored afterwards.

    The compile-event assertions need this test's own cache entry: run after
    a sibling that built the same configuration, nothing would be traced and
    the first assertion would fail on a correct implementation (Codex review
    of #1089).
    """
    import collections

    from tenax.algorithms import _ctm_energy_ad as cea

    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


@pytest.mark.usefixtures("_fresh_vjp_cache")
def test_fallback_in_the_same_entry_reuses_the_chain_rule():
    """The fused path compiles ``_jit_chain_rule``; a later fallback in the
    same cache entry must not compile another (Codex review of #1089).  When
    the chain rule lived inside the fused program, this fallback traced
    ``_jit_chain_rule`` for the first time -- the D=3 multi-hour compile."""
    grad = jax.grad(lambda a: _energy(a, "fixed_point"))

    mark = len(_TRACED)
    grad(_site(_FUSED_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("converged") and not diag.get("diverged"), (
        f"premise: seed {_FUSED_SEED} must be solved by the fused loop; got "
        f"converged={diag.get('converged')} diverged={diag.get('diverged')}"
    )
    first = _TRACED[mark:]
    assert "_jit_chain_rule" in first, (
        "a backward solved by the fused loop did not trace _jit_chain_rule, "
        "so the chain rule is still inside _jit_fused_fixed_point_bwd. "
        f"Traced: {sorted(set(first))}"
    )

    mark = len(_TRACED)
    g = grad(_site(_FALLBACK_SEED))
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("diverged") or not diag.get("converged"), (
        f"premise: seed {_FALLBACK_SEED} must make the fused loop give up "
        f"(n_iter={diag.get('n_iter')}), else no fallback ran"
    )
    second = _TRACED[mark:]
    assert "_jit_chain_rule" not in second, (
        "the GMRES fallback traced its own _jit_chain_rule instead of reusing "
        f"the fused path's. Traced: {sorted(set(second))}"
    )
    assert np.all(np.isfinite(np.asarray(g)))


def test_fused_and_gmres_adjoints_give_the_same_gradient():
    """Numerical check only: ``adjoint_method`` is part of the cache key, so
    this compares two separate entries and says nothing about sharing."""
    A0 = _site(_FUSED_SEED)
    g_fused = jax.grad(lambda a: _energy(a, "fixed_point"))(A0)
    g_gmres = jax.grad(lambda a: _energy(a, "gmres"))(A0)
    np.testing.assert_allclose(
        np.asarray(g_fused),
        np.asarray(g_gmres),
        rtol=1e-5,
        atol=1e-8,
        err_msg="fused and GMRES adjoints disagree on the same gradient",
    )
