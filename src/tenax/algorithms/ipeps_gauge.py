"""Belief-propagation gauge on an absorbed-form iPEPS state.

The state here is the site tensors alone: every bond weight is split
``sqrt(lambda)`` into each of its two ends, so a site tensor is the
wavefunction rather than one factor of it.  Vidal form exists only
transiently, inside this module and inside one simple-update step.

That convention is what lets the simple-update engine hold no lambdas at
all (#882 §3).  The weights this module returns are a **diagnostic** -- the
honest Schmidt spectrum at the BP fixed point -- not part of the state.
Dropping them loses a report, not physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_bp_gauge import (
    _BOND_OF,
    _LEGS,
    BondWeights,
    BPGaugeInfo,
    bp_gauge_checkerboard,
)
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.tensor import Tensor

__all__ = [
    "absorb_weights",
    "ctm_rdm2x1_planar",
    "gauge_fix",
    "torus_2x2_sign_free",
]


def absorb_weights(A: Tensor, B: Tensor, weights: BondWeights) -> tuple[Tensor, Tensor]:
    """Split every bond weight symmetrically into its two ends.

    Vidal ``Gamma_A lambda Gamma_B`` becomes absorbed
    ``(Gamma_A sqrt(lambda)) (sqrt(lambda) Gamma_B)`` -- the same physical
    object, since ``sqrt(lam) * sqrt(lam) == lam`` on every bond exactly once.

    Precondition: every entry of ``weights`` must be finite and non-negative.
    This takes ``sqrt()`` with no validation of its own, unlike
    :func:`bp_gauge_checkerboard`, which raises on a non-finite or
    non-positive weight vector -- a negative entry here would silently
    produce ``nan``.  Every real call site hands over SVD singular values,
    which already satisfy this.
    """
    out = {"A": A, "B": B}
    for site in ("A", "B"):
        for leg in _LEGS:
            lam = getattr(weights, _BOND_OF[(site, leg)])
            out[site] = scale_bond_axis(out[site], leg, jnp.sqrt(lam))
    return out["A"], out["B"]


def gauge_fix(
    A: Tensor,
    B: Tensor,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[Tensor, Tensor, BondWeights, BPGaugeInfo]:
    """Re-derive the BP gauge of an **absorbed-form** pair.

    Takes no incoming weights, because there are none to hand over: the pair
    already carries them.  Internally this is ``bp_gauge_checkerboard`` with
    ``BondWeights.ones`` -- correct precisely because the absorbed tensors
    already *are* ``Gamma_A lambda Gamma_B``.

    Args:
        A, B:     Absorbed-form site tensors, labels ``(u,d,l,r,phys)``.
        tol:      Stop once the largest relative weight change falls below this.
        max_iter: Maximum BP sweeps.

    Returns:
        ``(A, B, weights, info)``.  ``weights`` is a diagnostic: the Schmidt
        spectrum of the returned state at the BP fixed point.
    """
    labels = A.labels()
    D_h = A.indices[labels.index("r")].dim
    D_v = A.indices[labels.index("d")].dim
    return bp_gauge_checkerboard(
        A, B, BondWeights.ones(D_h, D_v), tol=tol, max_iter=max_iter
    )


# The 2x2 torus, one entry per site copy: which edge each leg sits on.  Sites
# are ``A00 B10 / B01 A11``, and the eight edges are the ones
# ``tests/_ipeps_gauge_helpers.py:_torus_2x2`` writes out::
#
#     i = A00.r-B10.l  h_AB     m = A00.d-B01.u  v_AB
#     j = B10.r-A00.l  h_BA *   n = B01.d-A00.u  v_BA *
#     k = B01.r-A11.l  h_BA     o = B10.d-A11.u  v_BA
#     l = A11.r-B01.l  h_AB *   p = A11.d-B10.u  v_AB *
#
# (``*`` = wraps around the torus; those four are closed by nothing more than
# a shared label, which is why this needs no boundary special-casing.)  Every
# label below appears on exactly two sites, and the pairs are always a
# ``r``-with-``l`` or a ``d``-with-``u``, i.e. exactly the leg pairings the
# checkerboard already contracts -- so the flows line up on the symmetric path
# without a single flip.  The ``__`` prefix keeps an edge label named after
# the bond it carries (``__l``) from colliding with the leg named ``l``.
_TORUS_EDGE_OF: dict[str, dict[str, str]] = {
    "A00": {"u": "__n", "d": "__m", "l": "__j", "r": "__i"},
    "B10": {"u": "__p", "d": "__o", "l": "__i", "r": "__j"},
    "B01": {"u": "__m", "d": "__n", "l": "__l", "r": "__k"},
    "A11": {"u": "__o", "d": "__p", "l": "__k", "r": "__l"},
}
#: Which sublattice tensor each copy is, and the open physical label it keeps.
#: The order is the one ``_torus_2x2``'s ``->wxyz`` uses, so the two probes'
#: outputs are directly comparable without a transpose.
_TORUS_COPIES: tuple[tuple[str, str, str], ...] = (
    ("A00", "A", "__w"),
    ("B10", "B", "__x"),
    ("B01", "B", "__y"),
    ("A11", "A", "__z"),
)


def torus_2x2_sign_free(A: Tensor, B: Tensor, weights: BondWeights) -> Tensor:
    """The closed 2x2 checkerboard torus, physical legs open.  **Bosonic only.**

    Same network as ``tests/_ipeps_gauge_helpers.py:_torus_2x2`` -- sites
    ``A(0,0) B(1,0) / B(0,1) A(1,1)``, each carrying the weight of the bond
    leaving its ``r`` and ``d`` legs, which places each of the four weights
    exactly twice and each edge's weight exactly once -- but routed through
    :func:`~tenax.contraction.contractor.contract` instead of ``np.einsum`` on
    densified blocks.

    .. warning::
        This is **not** a graded contraction and must not be used to certify a
        fermionic gauge.  ``contract`` does not apply Koszul signs -- that is a
        deliberate choice (#555, ``contraction/contractor.py``), justified for
        *planar* networks, "the only kind Tenax's CTM/RDM/energy code uses".
        Signs enter Tenax only through ``SymmetricTensor.transpose``, ``fuse``
        and the ``linalg`` decompositions; ``bar()`` applies none, and this
        function calls none of them.  So it is categorically sign-free, and a
        closed 2x2 torus is exactly the non-planar case that carve-out
        excludes -- its wrap-around edges cross the interior.  Measured on a
        ``FermionParity`` pair it is value-identical to the ``np.einsum``
        probe it was once meant to replace (2.15e-15 relative), i.e. it adds
        no fermionic information whatsoever.  Use
        :func:`ctm_rdm2x1_planar` for the fermionic case.

    What it *is* good for: ``contract`` is multilinear and every bond here is
    contracted directly, so an ``X``/``X^-1`` insertion still cancels
    algebraically.  On a bosonic pair -- where sign-free *is* correct -- this
    is an exact state-equality witness covering all four bonds at once, and it
    catches a "gauge" whose two halves fail to cancel.

    Closed on both axes for the reason ``_torus_2x2`` documents: a two-site
    probe leaves three legs open per site, so a gauge on any of them survives
    uncancelled and proves nothing.  Here each of the four bonds appears twice,
    each time with the gauge on one end and its inverse on the other.

    Args:
        A, B:    Site tensors of the two sublattices, labels ``(u,d,l,r,phys)``.
        weights: Bond weights the pair currently carries.  Pass
                 :meth:`BondWeights.ones` for an absorbed-form pair.

    Returns:
        A 4-leg tensor of the four open physical legs, in the order
        ``A(0,0), B(1,0), B(0,1), A(1,1)`` -- the same axis order
        ``_torus_2x2`` returns, so the two are comparable elementwise.
    """
    dressed = {}
    for site, t in (("A", A), ("B", B)):
        for leg in ("d", "r"):
            t = scale_bond_axis(t, leg, getattr(weights, _BOND_OF[(site, leg)]))
        dressed[site] = t

    copies = [
        dressed[site].relabels({**_TORUS_EDGE_OF[name], "phys": phys})
        for name, site, phys in _TORUS_COPIES
    ]
    return contract(*copies, output_labels=[phys for _, _, phys in _TORUS_COPIES])


def ctm_rdm2x1_planar(
    A: Tensor,
    B: Tensor,
    *,
    chi: int = 12,
    max_iter: int = 60,
    conv_tol: float = 1e-9,
) -> jax.Array:
    """The horizontal 2x1 reduced density matrix, via converged CTM.

    A gauge-invariance witness that is valid **for fermions**, which
    :func:`torus_2x2_sign_free` is not.  The whole point is that this stays
    inside the regime the contractor's sign convention is documented for:
    #555 declines to auto-apply Koszul signs and justifies that for *planar*
    networks, "the only kind Tenax's CTM/RDM/energy code uses".  This is that
    code -- ``ctm_tensor_2site`` then ``_rdm2x1_tensor_2site`` -- so the
    contraction is sign-correct by the same argument that makes the shipped
    fermionic energies sign-correct.  A closed torus is not.

    Gauge-invariant because it is a *physical* observable of the state, not a
    function of the tensors' parametrisation: the environment is re-derived
    from whatever ``A``/``B`` it is handed, so a genuine gauge -- which leaves
    the state alone -- leaves this alone too.  That also makes it a *relative*
    check, which is the only kind available here: tenax's absolute fermionic
    energy is not certified (#392, #879), but "unchanged across a gauge" needs
    no certified zero point.

    The returned matrix is trace-normalised by ``_rdm2x1_tensor_2site``, so an
    overall rescale of ``A`` or ``B`` -- which ``bp_gauge_checkerboard`` does
    every bond, deliberately -- cannot show up as a false difference.

    Args:
        A, B:      Checkerboard site tensors, labels ``(u,d,l,r,phys)``.
        chi:       Environment bond dimension.
        max_iter:  Maximum CTM sweeps.
        conv_tol:  CTM convergence tolerance on the corner spectrum.

    Returns:
        The ``(d, d, d, d)`` horizontal A-B reduced density matrix.

    Raises:
        ValueError: if the converged environment has a rank-1 corner.  That is
            the ``recipe="1x1"`` collapse signature (#723/#726/#747), and a
            collapsed environment produces an energy that is bit-identical
            across a 4x change in ``chi`` -- i.e. a witness that cannot see
            anything, which would silently certify any gauge as exact.  The
            2x2 recipe is used here precisely to avoid it, so hitting this
            means something else went wrong and the result must not be
            believed.
    """
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
    from tenax.algorithms._ctm_tensor_energy import _rdm2x1_tensor_2site

    env_A, env_B = ctm_tensor_2site(
        A, B, chi=chi, max_iter=max_iter, conv_tol=conv_tol, recipe="2x2"
    )
    for name, env in (("A", env_A), ("B", env_B)):
        # chi x chi, always small -- todense() is fine here.
        c1 = np.asarray(env.C1.todense())
        rank = int(np.linalg.matrix_rank(c1, tol=1e-10 * max(np.abs(c1).max(), 1e-300)))
        if rank <= 1:
            raise ValueError(
                f"CTM environment for sublattice {name} converged to a rank-"
                f"{rank} corner.  A collapsed environment makes this witness "
                f"blind, so it must not be used to certify a gauge."
            )
    return _rdm2x1_tensor_2site(A, B, env_A, env_B)
