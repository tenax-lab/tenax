"""#1028: ``projector_backward="auto"`` lets the 2x2 projectors flow on implicit AD.

The fixed-point adjoint used to run with the plaquette projectors frozen
(``stop_gradient``), which drops ``dP/dA`` from the gradient.  Measured on
the 1x1 Heisenberg implicit-AD energy, relative AD-vs-FD error frozen was
0.37-0.66% at D=2 chi=8 and 6.8-74% at D=3 chi=16/32; flowing with the
``"bond_phase"`` gauge it was 1e-6 with the adjoint residual at ~3e-11.  So
``"auto"`` now resolves to ``"flow"`` exactly where ``forward_gauge`` resolves
to ``"bond_phase"`` on the implicit path, and stays frozen everywhere else.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_ad_policy import build_ad_ctm_config, make_ctm_energy_fn
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig


@pytest.mark.parametrize(
    ("ctm_kw", "implicit", "expected"),
    [
        ({}, True, "flow"),  # the default implicit path
        ({"forward_gauge": "bond_phase"}, True, "flow"),
        ({}, False, "auto"),  # explicit AD: no adjoint solve, stays frozen
        ({"forward_gauge": "phase"}, True, "auto"),  # not the measured gauge
        ({"fuse_virtual_legs": False}, True, "auto"),  # split CTM -> "phase"
        ({"ctm_ad_mode": "root_implicit"}, True, "auto"),
        ({"projector_backward": "standard"}, True, "standard"),  # explicit wins
        ({"projector_backward": "flow"}, False, "flow"),
    ],
)
def test_auto_resolves_to_flow_only_with_bond_phase_on_implicit(
    ctm_kw, implicit, expected
):
    config = iPEPSConfig(
        max_bond_dim=2, ctm=CTMConfig(chi=4, **ctm_kw), gs_implicit_ad=implicit
    )
    assert build_ad_ctm_config(config).projector_backward == expected
    assert config.ctm.projector_backward == ctm_kw.get("projector_backward", "auto")


@pytest.mark.parametrize(
    ("forward_gauge", "expected"), [("auto", "flow"), ("phase", "auto")]
)
def test_the_implicit_dispatcher_resolves_auto(monkeypatch, forward_gauge, expected):
    """The closure resolves ``"auto"`` itself, like ``forward_gauge``.

    It is handed an unresolved ``CTMConfig`` here, so this fails if the call
    site forwards the raw field and a caller skips ``build_ad_ctm_config``.
    """
    from tenax.algorithms import _ctm_energy_ad
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

    seen = {}

    def _stub(*args, **kwargs):
        seen.update(kwargs)
        return jnp.asarray(0.0)

    monkeypatch.setattr(_ctm_energy_ad, "ctm_energy_implicit", _stub)
    cfg = CTMConfig(chi=4, forward_gauge=forward_gauge)
    energy_fn = make_ctm_energy_fn(
        neighbors={(0, 0): {d: (0, 0) for d in ("left", "right", "top", "bottom")}},
        gate=jnp.zeros((2, 2, 2, 2)),
        get_ctm_cfg=lambda: cfg,
        env_cache={},
        use_explicit=False,
        explicit_warmup=1,
        explicit_steps=1,
    )
    energy_fn({(0, 0): _wrap_as_dense_tensor(jnp.ones((2, 2, 2, 2, 2)))})
    assert seen["projector_backward"] == expected


def _su_state(D):
    from tenax.algorithms.ipeps import heisenberg_gate, ipeps

    _, (A_su, _), _ = ipeps(
        heisenberg_gate(), None, iPEPSConfig(max_bond_dim=D, num_imaginary_steps=100)
    )
    data = jnp.asarray(A_su.todense())
    return data / jnp.linalg.norm(data)


@pytest.mark.slow
def test_default_implicit_gradient_matches_finite_differences():
    """The default implicit-AD closure must differentiate its own energy.

    D=2 chi=6 SU state through ``build_ad_ctm_config`` + ``make_ctm_energy_fn``
    with nothing but CTM convergence knobs set, so it runs whatever "auto"
    resolves to.  Control: the same closure with the projectors frozen must
    miss by far more than the tolerance, or the fixture cannot see #1028.

    Measured on two directions (h=1e-4): flowing 1.6e-7 / 2.4e-7, scaling
    as h^2 (FD truncation); frozen 3.8e-3 / 9.1e-3, flat in h (a real
    bias).  At chi=8 the frozen bias on this state is only 2.8e-4.
    """
    from tenax import heisenberg_gate, sublattice_rotate_gate
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

    gate = sublattice_rotate_gate(heisenberg_gate())
    A0 = _su_state(D=2)
    direction = jax.random.normal(jax.random.PRNGKey(0), A0.shape, dtype=A0.dtype)
    direction = direction / jnp.linalg.norm(direction)

    def closure(projector_backward):
        ctm = CTMConfig(
            chi=6,
            max_iter=400,
            conv_tol=1e-10,
            gmres_tol=1e-10,
            projector_backward=projector_backward,
        )
        cfg = build_ad_ctm_config(
            iPEPSConfig(max_bond_dim=2, ctm=ctm, gs_implicit_ad=True)
        )
        fn = make_ctm_energy_fn(
            neighbors={(0, 0): {d: (0, 0) for d in ("left", "right", "top", "bottom")}},
            gate=gate,
            get_ctm_cfg=lambda: cfg,
            env_cache={},
            use_explicit=False,
            explicit_warmup=1,
            explicit_steps=1,
        )

        def energy(data):
            return fn({(0, 0): _wrap_as_dense_tensor(data / jnp.linalg.norm(data))})

        return cfg, energy

    def rel_error(energy):
        g = jax.grad(energy)(A0)
        ad = float(jnp.vdot(g, direction).real)
        h = 1e-4
        fd = (float(energy(A0 + h * direction)) - float(energy(A0 - h * direction))) / (
            2 * h
        )
        return abs(ad - fd) / abs(fd)

    cfg, energy = closure("auto")
    assert cfg.projector_backward == "flow"
    err_default = rel_error(energy)

    _, energy_frozen = closure("standard")
    err_frozen = rel_error(energy_frozen)

    assert err_frozen > 1e-3, f"fixture blind to #1028: frozen error {err_frozen:.2e}"
    assert err_default < 1e-5, (
        f"default implicit gradient off FD by {err_default:.2e} "
        f"(frozen control {err_frozen:.2e})"
    )
    assert np.isfinite(err_default)
