"""The fused fixed-point backward and its GMRES fallback share one chain rule.

The chain rule -- the params-VJP of the CTM sweep -- is most of the implicit
backward's compile.  It used to be compiled inside ``_jit_fused_fixed_point_bwd``
and again as ``_jit_chain_rule`` for the eager-GMRES fallback, so the first
fallback paid the compile a second time: 2 h 56 min at D=3 chi=12 on CPU.  Now
the fused program returns ``lam`` only and both paths call ``_jit_chain_rule``.

Pinned here by JAX's own compile events: a backward that the fused loop solves
must compile ``_jit_chain_rule`` (when the chain rule lived inside the fused
program it never did), and the gradient must match the eager-GMRES path's.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np

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


def _site():
    rng = np.random.default_rng(0)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


def test_fused_backward_uses_the_shared_chain_rule():
    A0 = _site()
    mark = len(_TRACED)
    g_fused = jax.grad(lambda a: _energy(a, "fixed_point"))(A0)
    diag = get_last_implicit_ad_diagnostics()
    assert diag.get("converged") and not diag.get("diverged"), (
        f"premise: the fused loop must solve this backward itself (no "
        f"fallback), else this test observes the fallback, not the fused "
        f"path; got converged={diag.get('converged')} "
        f"diverged={diag.get('diverged')} n_iter={diag.get('n_iter')}"
    )
    traced = _TRACED[mark:]
    assert "_jit_chain_rule" in traced, (
        "a backward solved by the fused loop did not trace _jit_chain_rule, "
        "so the chain rule is still inside _jit_fused_fixed_point_bwd and "
        "the GMRES fallback would compile a second copy. Traced: "
        f"{sorted(set(traced))}"
    )

    g_gmres = jax.grad(lambda a: _energy(a, "gmres"))(A0)
    np.testing.assert_allclose(
        np.asarray(g_fused),
        np.asarray(g_gmres),
        rtol=1e-5,
        atol=1e-8,
        err_msg="fused and GMRES adjoints disagree on the same gradient",
    )
