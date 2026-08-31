"""``recipe="1x1"`` deprecation (#911), phase 1: warn, do not yet reject.

#911 measured that this recipe reaches **no fixed point in any configuration
reachable from the public API** -- three separate mechanisms, none of which
crossed ``conv_tol=1e-10`` in 240 sweeps:

* ``projector_method='svd'`` (the default) collapses the corner to rank 1, so
  the energy is bit-identical across a 4x change in chi;
* ``'eigh'``/``'qr'`` hold full rank but limit-cycle, energy ranging over
  3.4e-3 to 4.9e-3 across the last 40 sweeps;
* on a non-uniform cell one bond is truncated by two inequivalent projectors on
  alternating sweeps, so nothing is stationary under both.

These tests do not re-measure any of that -- #911 and the tests it cites already
do. What is pinned here is the *deprecation contract*: that every entry point
which accepts the recipe warns, that the healthy engines do not, and that the
message names a migration target the user can actually act on.

``pyproject.toml`` suppresses this warning suite-wide, because a dozen tests pass
``recipe="1x1"`` deliberately -- they exist to assert the collapse. Every test
here therefore un-suppresses explicitly, following the pattern already
established for the ``chi_ramp`` and ``gs_conv_criterion`` deprecations.
"""

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_c4v import ctm_tensor_c4v
from tenax.algorithms._ctm_tensor_convergence import (
    SINGLE_SITE_NEIGHBORS,
    _ctm_tensor_multisite,
    ctm_tensor,
    ctm_tensor_2site,
)
from tenax.algorithms.ipeps_config import iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

MATCH = r"recipe='1x1' is deprecated"


def _site(D=2, d=2, seed=0):
    """Minimal PEPS site tensor -- the warning does not depend on physics."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(D, D, D, D, d))
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(jnp.array(data), indices)


# ------------------------------------------------------------------ #
# Every entry point that accepts the recipe                           #
# ------------------------------------------------------------------ #


def test_ctm_tensor_warns():
    with pytest.warns(DeprecationWarning, match=MATCH):
        ctm_tensor(_site(), chi=4, max_iter=2, recipe="1x1")


def test_ctm_tensor_2site_warns():
    with pytest.warns(DeprecationWarning, match=MATCH):
        ctm_tensor_2site(_site(seed=0), _site(seed=1), chi=4, max_iter=2, recipe="1x1")


def test_ctm_tensor_multisite_warns():
    with pytest.warns(DeprecationWarning, match=MATCH):
        _ctm_tensor_multisite(
            {(0, 0): _site()}, SINGLE_SITE_NEIGHBORS, chi=4, max_iter=2, recipe="1x1"
        )


def test_ipeps_config_warns_on_gs_recipe():
    """Raised at config construction, so an optimizer warns once, not per step."""
    with pytest.warns(DeprecationWarning, match=MATCH):
        iPEPSConfig(max_bond_dim=2, gs_recipe="1x1")


# ------------------------------------------------------------------ #
# ...and nothing else. These are the false positives that matter.     #
# ------------------------------------------------------------------ #


def test_the_default_recipe_does_not_warn():
    """Without this the suite passes if the warning is unconditional."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ctm_tensor(_site(), chi=4, max_iter=2, recipe="2x2")
        iPEPSConfig(max_bond_dim=2, gs_conv_criterion="grad_norm")


def test_ctm_tensor_c4v_does_not_warn():
    """The *other* thing called "1x1" is a different function, and it is healthy.

    ``ctm_tensor_c4v`` runs one 'down' move per sweep and passes ``(Qf, Qf)`` --
    a Gram matrix of a corner that has already absorbed the double layer -- into
    the same ``_compute_projector_tensor``.  #911 measured it converging at full
    rank under all three projector methods, agreeing with ``recipe="2x2"`` to
    1e-12.  Deprecating it too would be the name collision doing damage.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ctm_tensor_c4v(_site(), chi=4, max_iter=2)


# ------------------------------------------------------------------ #
# Properties of the message and of where it is raised                 #
# ------------------------------------------------------------------ #


def test_warns_once_per_call_not_once_per_sweep():
    """The reason it is raised at the entry point rather than inside the loop.

    A per-sweep warning would emit 50 times here and drown the run.  ``always``
    is required: the default filter dedups by location and would make this pass
    whichever way the code is written.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ctm_tensor(_site(), chi=4, max_iter=50, conv_tol=0.0, recipe="1x1")
    assert sum(1 for w in rec if issubclass(w.category, DeprecationWarning)) == 1


def test_message_names_both_migration_targets():
    """A deprecation a user cannot act on is just noise.

    The migration is state-dependent -- ``2x2`` in general, but that recipe
    ignores ``projector_method``, so a caller who needs ``qr``/``eigh`` has to
    be sent to ``ctm_tensor_c4v`` instead.  Both must be named.
    """
    with pytest.warns(DeprecationWarning) as rec:
        ctm_tensor(_site(), chi=4, max_iter=2, recipe="1x1")
    msg = str(rec[0].message)
    assert "recipe='2x2'" in msg
    assert "ctm_tensor_c4v" in msg
    assert "#911" in msg
    # The trap this closes: symmetrizing the *state* looks like a fix and is
    # not.  #911 measured the C4v-symmetrized state cycling as hard or harder
    # (energy range 4.47e-3 vs 4.86e-3 raw), contradicting a load-bearing
    # comment in test_reduced_corner_qr.py.
    assert "C4v symmetrization of the *state* does not rescue" in msg


def test_the_suite_wide_suppression_still_lets_it_fire():
    """``pyproject.toml`` ignores this warning; that filter must not over-match.

    If the ``ignore:`` pattern were broadened to swallow the warning even under
    ``pytest.warns``, every test above would pass vacuously.  This asserts the
    suppression is active by default and that un-suppressing genuinely restores
    it -- the same check the chi_ramp deprecation tests make.
    """
    with warnings.catch_warnings(record=True) as rec:
        # No simplefilter: inherit pytest's configured filters, which include
        # the pyproject ignore.
        ctm_tensor(_site(), chi=4, max_iter=2, recipe="1x1")
    assert not [w for w in rec if issubclass(w.category, DeprecationWarning)]
