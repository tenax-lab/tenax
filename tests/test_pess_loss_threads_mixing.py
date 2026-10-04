"""The PESS AD losses must hand ``CTMConfig.ctm_mixing`` to the CTM (#1060).

``build_pess_loss``, ``build_pess_loss_exact`` and
``build_pess_loss_3site_multisite`` call ``ctm_energy_implicit`` themselves.
They passed ``forward_gauge`` but not ``mixing``, so a ``ctm_mixing`` config
ran unmixed -- and the 3-site optimizer's env refresh (``ctm_converge_kwargs``)
did mix, so it differentiated a loss on a different iteration than it warmed.
Mechanism test: spy on ``ctm_energy_implicit`` and read the kwarg.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import pess_optimize  # noqa: E402
from tenax.algorithms._pess_multisite_energy import (
    kagome_3site_bond_gates,  # noqa: E402
)
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402
from tenax.algorithms.pess import (  # noqa: E402
    IPESSState,
    kagome_xxz_pess_cg_gates,
    kagome_xxz_pess_cg_gates_exact,
)

_BUILDERS = {
    "supersite": (pess_optimize.build_pess_loss, kagome_xxz_pess_cg_gates),
    "exact": (pess_optimize.build_pess_loss_exact, kagome_xxz_pess_cg_gates_exact),
    "3site": (pess_optimize.build_pess_loss_3site_multisite, kagome_3site_bond_gates),
}


@pytest.mark.parametrize("which", sorted(_BUILDERS))
@pytest.mark.parametrize("mixing", [0.0, 0.3])
def test_pess_loss_passes_ctm_mixing(monkeypatch, which, mixing):
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return jnp.zeros(())

    monkeypatch.setattr(pess_optimize, "ctm_energy_implicit", spy)
    build, gates_fn = _BUILDERS[which]
    cfg = CTMConfig(chi=4, ctm_mixing=mixing)
    loss_fn = build(gates_fn(delta=1.0, d=3), cfg)
    loss_fn(IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0)))

    assert seen, "the loss never reached ctm_energy_implicit"
    assert seen[-1].get("mixing", 0.0) == mixing
