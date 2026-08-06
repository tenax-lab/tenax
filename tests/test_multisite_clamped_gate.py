"""The multisite root-implicit gate must follow the clamp that sets its floor.

#784.  The multisite engine imports ``_rank_capped_spectrum`` and clamps the
numerically-null tail exactly as the asymmetric engine does (#772/#778), but
kept the pre-#778 flat ``1e-6`` gate.  Clamping leaves an O(``rel_floor``)
inconsistency in the characteristic equations *by construction*, so a fixed
tolerance rejects precisely the states the clamp has just made solvable.

Measured on the frozen physical D=2 simple-update state (issue body):

    chi=4   root_residual 8.493e-06   8.5x over the 1e-6 default -> raises
    chi=6   root_residual 1.911e-05   19x over                   -> raises

and the gradients are *finite* in both cases -- multisite never had the NaN
half of #772, only the gate.  Those numbers are identical to the asymmetric
engine's covariant residual at the same chi, which is why the same
``_root_residual_tolerance`` law applies rather than a new constant.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from _su_fixtures import physical_su_d2

from tenax.algorithms._ad_primitives import RootResidualError
from tenax.algorithms._ctm_root_implicit_multisite import (
    cell_root_implicit_energy_and_grad,
)


def _sz():
    """A ONE-site operator.

    ``cell_root_implicit_energy_and_grad`` takes a single-site ``(d, d)``
    observable, not a two-site ``(d, d, d, d)`` gate -- see the ``_sz()`` used
    throughout ``test_ctm_root_implicit_multisite.py``.  Passing a 2-site gate
    reaches the adjoint solve and dies there on a cotangent shape mismatch,
    which looks like an engine bug and is not one.
    """
    import jax.numpy as jnp

    return jnp.array([[0.5, 0.0], [0.0, -0.5]])


def _random_site(D=2, d=2, seed=42):
    """Well-conditioned random state: full-rank cut, so the clamp stays inert."""
    import numpy as np

    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    rng = np.random.RandomState(seed)
    data = jax.numpy.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = data / (jax.numpy.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


@pytest.mark.slow
@pytest.mark.parametrize("chi", [4, 6])
def test_a_clamped_physical_state_is_not_rejected(chi):
    """The headline of #784: this raised ``RootResidualError`` before the fix.

    ``on_root_residual="raise"`` is the default and is left in place on
    purpose -- the whole failure is that the default policy aborts on a state
    the engine handles correctly.
    """
    A = physical_su_d2()
    out = cell_root_implicit_energy_and_grad(
        {(0, 0): A},
        _sz(),
        chi=chi,
        nrows=1,
        ncols=1,
        max_iter=300,
        conv_tol=1e-13,
        return_diagnostics=True,
    )
    E, grads = out[0], out[1]
    assert bool(jax.numpy.isfinite(E)), E
    leaves = jax.tree.leaves(grads)
    assert leaves, "no gradient returned"
    assert all(bool(jax.numpy.all(jax.numpy.isfinite(g))) for g in leaves)


@pytest.mark.slow
def test_a_full_rank_state_keeps_the_strict_gate():
    """The relaxation must be *conditional on the clamp*, not unconditional.

    This is the property that stops the fix becoming "accept anything": when
    the clamp is inert (``usable_rank == chi``) there is no intrinsic floor,
    roundoff is reachable -- a well-conditioned random state reads 2.4e-14 per
    the issue -- so the strict tolerance still applies and still fires.

    Note what this test canNOT be: asking a *clamped* state to honour a
    stricter ``root_residual_warn``.  ``_root_residual_tolerance`` returns
    ``max(base_tol, floor)``, so with the clamp active a caller cannot tighten
    below the floor at all.  That is deliberate -- demanding a residual below
    the level the equations can even represent is unsatisfiable, not strict --
    but it means a clamped state can never be driven to raise this way, and a
    test written that way is testing nothing.
    """
    A = _random_site()
    with pytest.raises(RootResidualError):
        cell_root_implicit_energy_and_grad(
            {(0, 0): A},
            _sz(),
            chi=4,
            nrows=1,
            ncols=1,
            max_iter=200,
            conv_tol=1e-13,
            root_residual_warn=1e-30,
        )


def test_the_rank_report_sees_the_clamp_on_a_physical_state():
    """``usable_rank < chi`` is what makes the tolerance relax at all.

    Cheap (one SVD per coordinate, no adjoint solve), so this half runs in the
    required gate while the two end-to-end cases above are slow-only.
    """
    from tenax.algorithms._ctm_root_implicit_multisite import (
        converge_multisite,
        retained_rank_report_multisite,
    )

    A = physical_su_d2()
    corners, edges, _meta, _projs, a_by_cell = converge_multisite(
        {(0, 0): A}, 6, 1, 1, max_iter=60, conv_tol=1e-12, return_projectors=True
    )
    rep = retained_rank_report_multisite(corners, edges, a_by_cell, 6, 1, 1)
    assert rep["usable_rank"] < 6, rep
    assert rep["retained_smin_rtol"] < 1.0, rep
