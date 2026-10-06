"""#1087: the eager-GMRES adjoint must compile once per configuration.

``adjoint_method="gmres"`` (and the fused fixed-point branch's fallback, which
shares the code) handed ``gmres_pytree_jax`` a matvec closure built fresh on
every backward, capturing that call's params and environment.  JAX's GMRES
runs a ``lax.while_loop``, so every backward re-traced and re-compiled the
whole Krylov loop: measured ~170 s per call at D=2 chi=8 on CPU, against a
0.6 s solve, and 3-5 min per call at D=3.

The contract tested here is implementation-agnostic: a second backward at the
same static configuration -- new parameter *values*, same shapes -- compiles
nothing.  Counted with JAX's own compile events, so it does not depend on
which helper does the solve.
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

_EVENTS: list[tuple[str, str]] = []
_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
)
jm.register_event_duration_secs_listener(
    lambda event, duration, **kw: (
        _EVENTS.append((event, kw.get("fun_name", "?")))
        if event in _COMPILE_EVENTS
        else None
    )
)


#: One gate object for every call: ``id(gate)`` is part of the implicit-AD
#: dispatch cache key, so a fresh gate per call would rebuild (and recompile)
#: every backward helper and hide the defect behind a legitimate cache miss.
_GATE = heisenberg_gate()


def _energy(A_arr):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)},
            SINGLE_SITE_NEIGHBORS,
            _GATE,
            recipe="1x1",
            projector_method="eigh",
            chi=4,
            max_iter=30,
            min_iter=8,
            conv_tol=1e-12,
            adjoint_method="gmres",
        )
    )


def _site(seed):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


def test_second_gmres_backward_compiles_nothing():
    grad = jax.grad(_energy)
    g1 = grad(_site(0))
    assert "adjoint_residual" in get_last_implicit_ad_diagnostics(), (
        "no adjoint solve reported a residual, so the gmres branch may not "
        "have run and this test would observe nothing"
    )
    mark = len(_EVENTS)
    g2 = grad(_site(1))
    new = _EVENTS[mark:]
    assert not new, (
        f"the second gmres backward at the same configuration traced or "
        f"compiled {len(new)} function(s): {sorted({fn for _, fn in new})}. "
        f"Each backward must reuse the first one's compiled solve (#1087)."
    )
    assert np.all(np.isfinite(np.asarray(g1))) and np.all(np.isfinite(np.asarray(g2)))
