"""The metric preconditioner's GMRES solve compiles once per configuration.

``precondition_gradient`` handed JAX's GMRES a ``matvec`` closure built
fresh on every call over that call's metric matrix and ``delta``.  GMRES runs
a ``lax.while_loop``, so every call re-traced and re-compiled the Krylov
loop: measured 2 compiles per optimizer step, 1.38 s per step on GPU (about
6% of a 2-site D=3 chi=16 step) and 0.23 s on CPU.  Same pattern as the
eager-GMRES adjoint (#1087).

Contract, implementation-agnostic: a second call at the same shapes and
GMRES settings -- new tensor, environment-scale, gradient and ``delta``
*values* -- compiles nothing.  Counted with JAX's own compile events.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax import CTMConfig, iPEPSConfig  # noqa: E402
from tenax.algorithms import _metric_precond as mp  # noqa: E402
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import _wrap_as_dense_tensor  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402

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

CFG = iPEPSConfig()  # the default GMRES settings


def _site(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2, 2, 2, 2, 2))
    return _wrap_as_dense_tensor(jnp.asarray(A / np.linalg.norm(A)))


def _env(A):
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        **ctm_converge_kwargs(CTMConfig(chi=4, max_iter=30, min_iter=8)),
    )
    return envs[(0, 0)]


def _scale_env(env, c):
    return env._replace(**{f: getattr(env, f) * c for f in env._fields})


def test_second_preconditioner_call_compiles_nothing():
    A1, A2 = _site(0), _site(1)
    env = _env(A1)
    g1 = mp.precondition_gradient(A1, env, _site(2), 0.3, CFG)
    # The regime: the metric was usable, so GMRES ran (not the degenerate-norm
    # early return, which compiles nothing either way).
    assert mp._normalized_metric_matrix(A1, env) is not None
    # Build the second call's inputs first: only the call itself is counted.
    env2, grad2 = _scale_env(env, 1.7), _site(3)
    mark = len(_EVENTS)
    g2 = mp.precondition_gradient(A2, env2, grad2, 0.05, CFG)
    new = _EVENTS[mark:]
    assert not new, (
        f"the second preconditioner call at the same configuration traced or "
        f"compiled {len(new)} function(s): {sorted({fn for _, fn in new})}. "
        f"Each call must reuse the first one's compiled GMRES solve."
    )
    assert np.all(np.isfinite(np.asarray(g1))) and np.all(np.isfinite(np.asarray(g2)))
    assert not np.allclose(np.asarray(g1), np.asarray(g2))  # it did new work


def test_delta_as_a_jax_scalar_compiles_nothing_new():
    # The optimizer passes delta as a Python float on later steps and as a
    # JAX scalar on step 0; both must hit one compiled solve.
    A = _site(0)
    env = _env(A)
    mp.precondition_gradient(A, env, _site(2), 0.3, CFG)
    # Strongly typed, as _tree_dot returns it (jnp.asarray(0.05) would be weak,
    # like a Python float, and test nothing).
    delta = jnp.asarray(0.05, dtype=jnp.float64)
    assert not delta.weak_type
    mark = len(_EVENTS)
    mp.precondition_gradient(A, env, _site(2), delta, CFG)
    new = _EVENTS[mark:]
    assert not new, sorted({fn for _, fn in new})
