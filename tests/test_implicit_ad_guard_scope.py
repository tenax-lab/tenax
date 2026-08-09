"""The implicit-AD CTM guard must only judge the path it describes.

``validate_ctm_for_implicit_ad`` encodes the requirements of the **fused
fixed-point** implicit backward: ``projector_method`` in ``("svd", "qr")``,
``forward_gauge="phase"``, ``ctm_conv_method="elementwise"``.  Those are real
-- phase gauge in particular is the verified AD default and sigma is known to
blow up on 2-site -- so this file does *not* relax them.

It pins their **scope**.  ``optimize_gs_ad`` used to run the check on every
``gs_implicit_ad`` config before choosing an engine, so the C4v-reference and
root-implicit paths were judged against a path they never reach.  Both own
their validation (``validate_root_implicit_config``; the C4v sweep's own
supported set), and the root path reads none of the three knobs at all.  The
result was four unverified-P1 rows in #802 (#349, #350 x2, #343), all of the
same shape: a valid configuration refused at config time.

That the rejection was false, and not a guard doing its job, is measured by
``test_c4v_projector_methods_agree`` below.
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import numpy as np
import pytest

from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad

GUARD_MSG = "Implicit AD requires CTM settings"

C4V_TARGET = "tenax.algorithms.ipeps_optimize._optimize_gs_ad_tensor_reference_c4v"
ROOT_TARGET = (
    "tenax.algorithms.ipeps_optimize_root_implicit.optimize_gs_ad_root_implicit"
)


def _config(*, ctm_kwargs, **cfg_kwargs) -> iPEPSConfig:
    return iPEPSConfig(
        ctm=CTMConfig(chi=8, **ctm_kwargs),
        unit_cell="1x1",
        gs_implicit_ad=True,
        max_bond_dim=2,
        **cfg_kwargs,
    )


# ---------------------------------------------------------------------------
# The paths the guard has no claim over must reach their own dispatcher.
#
# Mocked at the dispatcher boundary on purpose: the assertion is about *which
# engine is selected*, and running either one would pay for a CTM convergence
# to observe a routing decision already made.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ctm_kwargs"),
    [
        # #349 / #350: eigh is the C4v sweep's own default projector, and the
        # isometric one C4v needs -- the guard rejected it outright.
        ("eigh", {"projector_method": "eigh"}),
        # #343 / #350: "qr" is documented valid in CTMConfig's own docstring.
        ("qr_gauge", {"forward_gauge": "qr"}),
    ],
)
def test_c4v_reference_reaches_its_dispatcher(name, ctm_kwargs):
    """A C4v-reference config must not be judged by the fused path's rules."""
    config = _config(
        ctm_kwargs={"ctm_ad_mode": "c4v_reference", **ctm_kwargs},
        gs_c4v=True,
    )
    sentinel = object()
    with patch(C4V_TARGET, return_value=sentinel) as dispatcher:
        assert optimize_gs_ad(heisenberg_gate(), None, config) is sentinel
    dispatcher.assert_called_once()


def test_root_implicit_reaches_its_dispatcher():
    """The root-implicit path reads none of the three knobs the guard checks.

    ``gs_c4v`` stays False here because ``validate_root_implicit_config``
    rejects it -- that is the root path validating itself, which is the point.
    """
    config = _config(
        ctm_kwargs={"ctm_ad_mode": "root_implicit", "projector_method": "eigh"},
    )
    sentinel = object()
    with patch(ROOT_TARGET, return_value=sentinel) as dispatcher:
        assert optimize_gs_ad(heisenberg_gate(), None, config) is sentinel
    dispatcher.assert_called_once()


# ---------------------------------------------------------------------------
# ...and the fused path must still be judged by them.
#
# Without these, "scope the guard" is indistinguishable from "delete the
# guard", which would trade a loud false rejection for a quiet wrong gradient.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ctm_kwargs"),
    [
        ("eigh", {"projector_method": "eigh"}),
        ("sigma_gauge", {"forward_gauge": "sigma"}),
        ("qr_gauge", {"forward_gauge": "qr"}),
        ("sv_conv", {"ctm_conv_method": "sv"}),
    ],
)
def test_fused_implicit_ad_still_rejects(name, ctm_kwargs):
    """The default (fused fixed-point) implicit-AD path keeps every rule."""
    config = _config(ctm_kwargs=ctm_kwargs)
    with pytest.raises(ValueError, match=GUARD_MSG):
        optimize_gs_ad(heisenberg_gate(), None, config)


def test_guard_is_scoped_by_mode_not_disabled():
    """Flipping only ``ctm_ad_mode`` must flip the verdict.

    Pins the two branches against the *same* otherwise-identical config, so a
    future change that removes the check entirely fails here even if every
    test above is rewritten around it.
    """
    ctm_kwargs = {"projector_method": "eigh"}
    with pytest.raises(ValueError, match=GUARD_MSG):
        optimize_gs_ad(heisenberg_gate(), None, _config(ctm_kwargs=ctm_kwargs))

    routed = _config(
        ctm_kwargs={"ctm_ad_mode": "c4v_reference", **ctm_kwargs}, gs_c4v=True
    )
    with patch(C4V_TARGET, return_value=None):
        optimize_gs_ad(heisenberg_gate(), None, routed)


# ---------------------------------------------------------------------------
# Why the rejection was false rather than protective.
# ---------------------------------------------------------------------------


def test_c4v_projector_methods_agree():
    """eigh / qr / svd give the same C4v energy, so rejecting eigh protected nothing.

    The C4v sweep calls ``_compute_projector_tensor(Qf, Qf, chi, method)`` and
    keeps only ``P_1``.  For eigh/qr that is exact by construction
    (``P_1 is P_2``).  For svd (Fishman) the two projectors are distinct in
    general -- but a C4v enlarged corner is *symmetric*, so ``M = Qf^H Qf`` is
    Hermitian, ``U = V``, and they coincide anyway.

    Hence all three are interchangeable here, and the guard was neither
    protecting the svd it allowed nor justified in refusing the eigh it did
    not.  If that ever stops holding, this test fails and the exemption above
    needs re-deriving -- which is the outcome to want.
    """
    from tenax.algorithms._ctm_tensor import compute_energy_ctm_tensor
    from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
    from tenax.algorithms.ipeps import (
        _wrap_as_dense_tensor,
        build_c4v_basis,
        c4v_coeffs_from_tensor,
        c4v_tensor_from_coeffs,
    )

    D, d = 2, 2
    shape = (D, D, D, D, d)
    A_raw = jax.random.normal(jax.random.PRNGKey(0), shape)
    basis = jax.numpy.array(build_c4v_basis(D, d))
    A_c4v = c4v_tensor_from_coeffs(c4v_coeffs_from_tensor(A_raw, basis), basis, shape)
    A = _wrap_as_dense_tensor(A_c4v / jax.numpy.linalg.norm(A_c4v))

    gate = heisenberg_gate().todense()
    energies = {}
    for method in ("eigh", "qr", "svd"):
        env = ctm_tensor_c4v(
            A, chi=8, max_iter=60, conv_tol=1e-10, projector_method=method
        )
        energies[method] = complex(compute_energy_ctm_tensor(A, env, gate, d)).real

    np.testing.assert_allclose(
        energies["svd"],
        energies["eigh"],
        rtol=1e-9,
        atol=1e-11,
        err_msg=f"C4v energy depends on projector_method: {energies}",
    )
    np.testing.assert_allclose(
        energies["qr"],
        energies["eigh"],
        rtol=1e-9,
        atol=1e-11,
        err_msg=f"C4v energy depends on projector_method: {energies}",
    )
