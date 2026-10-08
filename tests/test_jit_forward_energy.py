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


@pytest.fixture
def _non_psd_rdms(monkeypatch):
    """Every energy RDM gets a negative eigenvalue, by construction.

    A CTM fixture whose RDMs happen to come out non-PSD depends on numerics
    that move with the JAX version and BLAS (Codex review of #1092), so the
    defect is injected instead: ``_normalise_rdm`` returns ``rho + P`` with
    ``P = diag(2, -2, 0, ...)``.  ``P`` is traceless and Hermitian, so the
    trace and Hermiticity checks still pass, and the smallest eigenvalue is
    at most ``rho_11 - 2 <= -1``.  The implicit-AD entries are cleared on
    both sides, so the patch is traced in and does not leak out.
    """
    import collections

    from tenax.algorithms import _ctm_energy_ad as cea
    from tenax.algorithms import _ctm_tensor_energy as te

    orig = te._normalise_rdm

    def poisoned(mat):
        out = orig(mat)
        if out.ndim != 2 or out.shape[0] < 2:
            return out
        p = jnp.zeros(out.shape[0], dtype=out.dtype).at[0].set(2.0).at[1].set(-2.0)
        return out + jnp.diag(p)

    saved = collections.OrderedDict(cea._VJP_CACHE)
    cea._VJP_CACHE.clear()
    monkeypatch.setattr(te, "_normalise_rdm", poisoned)
    yield
    cea._VJP_CACHE.clear()
    cea._VJP_CACHE.update(saved)


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


@pytest.mark.usefixtures("_non_psd_rdms")
def test_rdm_check_still_warns_from_the_jitted_forward():
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        jax.grad(lambda a: _energy(a, **_CLEAN))(_site())


@pytest.mark.usefixtures("_non_psd_rdms")
def test_warnings_as_errors_get_the_warning_not_an_xla_error():
    """The check is replayed in Python after the call, so under
    warnings-as-errors the caller sees the RuntimeWarning itself -- not a
    callback failure wrapped in an XLA runtime error."""
    # Compile first without the filter, so the raise below comes from the
    # replay on a cached program, the steady state of an optimizer loop.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _energy(_site(), **_CLEAN)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="not positive semi-definite"):
            _energy(_site(), **_CLEAN)


# --------------------------------------------------------------------------
# The replay queue (Codex review of #1092): per call, ordered, thread-local.
# --------------------------------------------------------------------------


def _spy_checks(monkeypatch):
    from tenax.algorithms import _ctm_diagnostics

    seen = []
    monkeypatch.setattr(
        _ctm_diagnostics,
        "check_rdm",
        lambda rdm, *, context="", **kw: seen.append((context, float(rdm[0, 0]))),
    )
    return seen


def test_replay_follows_trace_order_not_callback_order(monkeypatch):
    from tenax.algorithms import _ctm_tensor_energy as te

    seen = _spy_checks(monkeypatch)
    for pos in (2, 0, 1):  # callbacks may arrive in any order
        te._queue_traced_rdm(np.full((2, 2), pos), 7, position=pos, context=f"b{pos}")
    te.replay_traced_rdm_checks(7)
    assert [c for c, _ in seen] == ["b0", "b1", "b2"]


def test_a_call_replays_only_its_own_rdms(monkeypatch):
    from tenax.algorithms import _ctm_tensor_energy as te

    seen = _spy_checks(monkeypatch)
    te._queue_traced_rdm(np.ones((2, 2)), 11, position=0, context="call11")
    te._queue_traced_rdm(np.ones((2, 2)), 12, position=0, context="call12")
    te.replay_traced_rdm_checks(11)
    assert [c for c, _ in seen] == ["call11"]
    assert 12 in te._TRACED_RDMS and 11 not in te._TRACED_RDMS
    te.discard_traced_rdm_checks(12)
    assert 12 not in te._TRACED_RDMS


def test_recording_is_invisible_to_other_threads():
    import threading

    from tenax.algorithms import _ctm_tensor_energy as te

    seen_elsewhere = []
    with te.recording_traced_rdms(5):
        assert te._RDM_RECORDING.get() is not None
        t = threading.Thread(
            target=lambda: seen_elsewhere.append(te._RDM_RECORDING.get())
        )
        t.start()
        t.join()
    assert seen_elsewhere == [None]
    assert te._RDM_RECORDING.get() is None


def test_a_raising_check_leaves_nothing_queued():
    from tenax.algorithms import _ctm_tensor_energy as te

    bad = np.array([[1.0, 0.0], [0.0, -5.0]])  # not positive semi-definite
    te._queue_traced_rdm(bad, 21, position=0, context="bad")
    te._queue_traced_rdm(bad, 21, position=1, context="bad2")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            te.replay_traced_rdm_checks(21)
    assert 21 not in te._TRACED_RDMS
