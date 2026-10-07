"""The implicit-AD forward evaluates its energy under jit, and keeps its checks.

Eager, the forward's energy contraction ran op by op and was most of a warm
gradient (7.1 s of 8.3 s at D=2 chi=8 on the fermionic 2-site benchmark).  It
now runs as ``_jit_forward_energy``.  A traced RDM has no value for
``check_rdm`` (#845/#854), so the jitted energy records its RDMs and the
forward replays the checks after the call; these tests pin that the replay
happens, and that a warnings-as-errors caller still gets the warning itself.
"""

from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.monitoring as jm
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
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

_GATE = heisenberg_gate()


def _site(seed=0):
    rng = np.random.default_rng(seed)
    A0 = jnp.asarray(rng.standard_normal((2, 2, 2, 2, 2)))
    return A0 / jnp.linalg.norm(A0)


def _energy(A_arr, **kw):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(A_arr)}, SINGLE_SITE_NEIGHBORS, _GATE, **kw
        )
    )


#: Converges cleanly; the energy is a plain number to compare.
_CLEAN = dict(recipe="2x2", chi=4, max_iter=60, min_iter=8, conv_tol=1e-10)
#: Its RDMs are not positive semi-definite (smallest eigenvalue ~ -2.7, see
#: #854), so the forward's energy check must warn.
_NON_PSD = dict(
    recipe="1x1",
    projector_method="eigh",
    chi=4,
    max_iter=30,
    min_iter=8,
    conv_tol=1e-12,
    adjoint_method="gmres",
)


def test_forward_energy_is_jitted_and_matches_eager():
    from tenax.algorithms import _ctm_energy_ad as cea
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor

    A0 = _site()
    mark = len(_TRACED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = float(_energy(A0, **_CLEAN))
    assert "_jit_forward_energy" in _TRACED[mark:], (
        "the implicit-AD forward did not trace _jit_forward_energy, so its "
        "energy still runs op by op"
    )
    # Eager evaluation at the same converged environment.
    envs, _chi, _info = cea._sigma_gauged_ctm_converge(
        {(0, 0): _wrap_as_dense_tensor(A0)},
        SINGLE_SITE_NEIGHBORS,
        chi=4,
        max_iter=60,
        conv_tol=1e-10,
        projector_method="svd",
        renormalize=True,
        projector_backward="auto",
        qr_warmup_steps=3,
        env_init=None,
        forward_gauge="bond_phase",
        conv_method="elementwise",
        min_iter=8,
        return_info=True,
    )
    e_eager = float(
        jnp.real(
            compute_energy_ctm_tensor(_wrap_as_dense_tensor(A0), envs[(0, 0)], _GATE)
        )
    )
    np.testing.assert_allclose(e, e_eager, rtol=1e-10, atol=1e-12)


def test_rdm_check_still_warns_from_the_jitted_forward():
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        jax.grad(lambda a: _energy(a, **_NON_PSD))(_site())


def test_warnings_as_errors_get_the_warning_not_an_xla_error():
    """The check is replayed in Python after the call, so under
    warnings-as-errors the caller sees the RuntimeWarning itself -- not a
    callback failure wrapped in an XLA runtime error."""
    # Compile first without the filter, so the raise below comes from the
    # replay on a cached program, the steady state of an optimizer loop.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _energy(_site(), **_NON_PSD)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="not positive semi-definite"):
            _energy(_site(), **_NON_PSD)
