"""ε_T is structurally blind on some (recipe, projector_method) pairs (#727).

``max_truncation_error`` drives the variPEPS §2.8.2 auto-χ bump.  On some
configurations it is a genuine measurement of discarded weight; on others it
is ``0.0`` for reasons that have nothing to do with how much was discarded,
and a ``0.0`` of the second kind is indistinguishable from a lossless
truncation.  The bump then silently never fires however saturated χ is.

#727 claimed this was a property of the ``"1x1"`` *recipe*.  Measured, that
is too broad by one row: ``"1x1"``/``"eigh"`` reports a genuine ε_T.  The
blindness belongs to the **projector kernel's matrix shape**, not the recipe:

    "1x1" + "svd"   M = C1g^H C4g is chi x chi   -> S_full[chi:] empty -> 0
    "1x1" + "eigh"  rho on the fused index       -> (chi*D^2)-long     -> genuine
    "1x1" + "qr"    returns a literal 0.0        -> never computed
    "2x2" + any     retains chi*D^2 on both sides -> genuine

These tests pin that table.  They exist because the ``ctm_tensor`` docstring
asserted the opposite of two of these rows for months -- it said ε_T was
meaningful "only on the dense, non-tracer SVD path" and ``0.0`` for
``"eigh"``/``"qr"`` -- and nothing contradicted it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_projector import _compute_projector_tensor
from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

CHI = 6
D = 2


def _enlarged_corners(chi=CHI, bond=D, seed=0):
    """A ``(fused | cut)`` enlarged-corner pair, ``dim(fused) = chi * D**2``.

    The shapes are the whole point: ``fused`` is ``chi * D**2`` and the cut
    leg is ``chi``, so a kernel that contracts ``fused`` away lands on a
    ``chi x chi`` matrix with no room for a discarded tail, while one that
    keeps it has ``chi * D**2`` singular values and ``chi * D**2 - chi``
    genuinely discarded.
    """
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    fused_dim = chi * bond * bond
    fused = TensorIndex.from_charges(
        sym, np.zeros(fused_dim, dtype=np.int32), FlowDirection.IN, label="fused"
    )
    t1_r = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t1_r"
    )
    t3_l = TensorIndex.from_charges(
        sym, np.zeros(chi, dtype=np.int32), FlowDirection.OUT, label="t3_l"
    )
    base = rng.standard_normal((fused_dim, chi))
    pert = 0.05 * rng.standard_normal((fused_dim, chi))
    return (
        DenseTensor(jnp.asarray(base), (fused, t1_r)),
        DenseTensor(jnp.asarray(base + pert), (fused, t3_l)),
    )


def _site(bond=D, phys=2, seed=0):
    rng = np.random.default_rng(seed)
    sym = U1Symmetry()
    idx = tuple(
        TensorIndex.from_charges(sym, np.zeros(bond, dtype=np.int32), flow, label=lbl)
        for lbl, flow in [
            ("u", FlowDirection.OUT),
            ("d", FlowDirection.IN),
            ("l", FlowDirection.OUT),
            ("r", FlowDirection.IN),
        ]
    ) + (
        TensorIndex.from_charges(
            sym, np.zeros(phys, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    A = DenseTensor(
        jnp.asarray(rng.standard_normal((bond, bond, bond, bond, phys))), idx
    )
    return A * (1.0 / float(A.norm()))


# --------------------------------------------------------------------------- #
# The mechanism, at the projector kernel — no CTM convergence involved.        #
# --------------------------------------------------------------------------- #


@pytest.mark.core
def test_svd_projector_reports_zero_eps_t_because_its_matrix_is_chi_by_chi():
    """The blind row. ``M = C1g^H C4g`` is chi x chi, so nothing is at index >= chi.

    This is *not* "the truncation was lossless": the corners carry a
    ``chi * D**2`` fused index that the projector cuts down to ``chi``, so
    real weight is discarded on every call.  It is invisible because the
    cross-product summed the seam away before anyone looked.
    """
    C1g, C4g = _enlarged_corners()
    _P1, _P2, eps = _compute_projector_tensor(C1g, C4g, CHI, "svd", None, "auto")

    assert float(eps) == 0.0

    # ...and the discarded weight it failed to see is not small: the same
    # corners, measured on the spectrum that *does* retain the fused index,
    # report a substantial tail.
    _P1e, _P2e, eps_eigh = _compute_projector_tensor(
        C1g, C4g, CHI, "eigh", None, "auto"
    )
    assert float(eps_eigh) > 1e-6, (
        "the eigh kernel should see the discarded fused weight that svd "
        "structurally cannot; if this is ~0 the corners are too well "
        "conditioned to make the point"
    )


@pytest.mark.core
def test_eigh_projector_reports_a_genuine_eps_t():
    """The row #727 got wrong, and the docstring got backwards.

    ``rho = C1g C1g^H + C4g C4g^H`` lives on the ``fused`` index, so its
    spectrum is ``chi * D**2`` long and ``eigvals[chi:]`` is a real discarded
    tail.
    """
    C1g, C4g = _enlarged_corners()
    _P1, _P2, eps = _compute_projector_tensor(C1g, C4g, CHI, "eigh", None, "auto")
    assert np.isfinite(float(eps))
    assert float(eps) > 0.0


@pytest.mark.core
def test_qr_projector_returns_a_hardcoded_zero():
    """``"qr"`` does not compute an ε_T at all — it returns the literal 0.0.

    Distinct from the svd row: that one runs a truncation-error computation
    and gets 0 from an empty tail; this one never runs the computation.
    Same consequence for the auto-bump, different cause, so both are pinned.
    """
    C1g, C4g = _enlarged_corners()
    _P1, _P2, eps = _compute_projector_tensor(C1g, C4g, CHI, "qr", None, "auto")
    assert float(eps) == 0.0


# --------------------------------------------------------------------------- #
# End to end, through the public entry point.                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.core
@pytest.mark.parametrize("method", ["svd", "eigh", "qr"])
def test_2x2_reports_a_genuine_eps_t_for_every_projector_method(method):
    """``"2x2"`` keeps chi*D^2 on both sides, so ε_T is genuine.

    It also ignores ``projector_method`` (it hardcodes Fishman SVD), which is
    why all three parametrizations give the *same* ε_T rather than three
    different ones — asserted below so that fact stays visible.
    """
    A = _site()
    _env, eps = ctm_tensor(
        A, chi=CHI, max_iter=8, conv_tol=1e-10, projector_method=method, recipe="2x2"
    )
    assert float(eps) > 0.0


@pytest.mark.core
def test_2x2_eps_t_is_identical_across_projector_methods():
    """Because ``"2x2"`` ignores the selector, not because they agree."""
    A = _site()
    vals = [
        float(
            ctm_tensor(
                A,
                chi=CHI,
                max_iter=8,
                conv_tol=1e-10,
                projector_method=m,
                recipe="2x2",
            )[1]
        )
        for m in ("svd", "eigh", "qr")
    ]
    assert vals[0] == vals[1] == vals[2], (
        f"2x2 should hardcode Fishman SVD and ignore projector_method, got {vals}"
    )


@pytest.mark.core
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_1x1_eps_t_follows_the_projector_not_the_recipe():
    """The table's whole point: ``"1x1"`` is blind on svd/qr, genuine on eigh.

    #727 says ε_T is "identically zero for every input and every environment"
    on this recipe.  That is one row too broad, and the row it gets wrong is
    the one a caller could actually use to escape the blindness.
    """
    A = _site()
    eps = {}
    for m in ("svd", "eigh", "qr"):
        _env, e = ctm_tensor(
            A, chi=CHI, max_iter=8, conv_tol=1e-10, projector_method=m, recipe="1x1"
        )
        eps[m] = float(e)

    assert eps["svd"] == 0.0, f"svd should be structurally blind, got {eps['svd']}"
    assert eps["qr"] == 0.0, f"qr never computes eps_T, got {eps['qr']}"
    assert eps["eigh"] > 0.0, (
        f"eigh diagonalises rho on the fused index and should see a genuine "
        f"discarded tail, got {eps['eigh']} — if this is 0 then #727's "
        f"'identically zero on 1x1' claim became true and the docstring table "
        f"needs updating back"
    )


# --------------------------------------------------------------------------- #
# The consequence: a dead auto-bump is now loud.                               #
# --------------------------------------------------------------------------- #


@pytest.mark.core
@pytest.mark.parametrize("method", ["svd", "qr"])
def test_auto_bump_on_a_blind_projector_warns(method):
    """Asking for a reactive χ bump that provably cannot fire should say so."""
    from tenax import CTMConfig, iPEPSConfig

    with pytest.warns(UserWarning, match="chi_auto_bump cannot fire"):
        iPEPSConfig(
            max_bond_dim=2,
            gs_recipe="1x1",
            ctm=CTMConfig(
                chi=4, chi_auto_bump=True, chi_max=8, projector_method=method
            ),
        )


@pytest.mark.core
def test_auto_bump_does_not_warn_where_eps_t_is_genuine():
    """The negative half — otherwise the guard could be an unconditional warn.

    ``"1x1"``/``"eigh"`` measures a real ε_T, and ``"2x2"`` measures one for
    any projector method, so neither should be flagged.
    """
    import warnings

    from tenax import CTMConfig, iPEPSConfig

    for recipe, method in (("1x1", "eigh"), ("2x2", "svd"), ("2x2", "qr")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            iPEPSConfig(
                max_bond_dim=2,
                gs_recipe=recipe,
                ctm=CTMConfig(
                    chi=4, chi_auto_bump=True, chi_max=8, projector_method=method
                ),
            )
