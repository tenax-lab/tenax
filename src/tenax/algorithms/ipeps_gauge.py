"""Belief-propagation gauge on an absorbed-form iPEPS state.

The state here is the site tensors alone: every bond weight is split
``sqrt(lambda)`` into each of its two ends, so a site tensor is the
wavefunction rather than one factor of it.  Vidal form exists only
transiently, inside this module and inside one simple-update step.

That convention is what lets the simple-update engine hold no lambdas at
all (#882 §3).  The weights :func:`gauge_fix` returns are a **diagnostic** --
the honest Schmidt spectrum at the BP fixed point, *already absorbed into the
pair returned with them*.  Dropping them loses a report, not physics; absorbing
them a second time loses the state, by order unity.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# The module, not just its names: ``gauge_fix`` reaches through it for the
# driver dispatch and the solve body, so that a test which patches something in
# there -- ``_use_traced_loop`` to force the eager driver onto a dense pair,
# ``_sweep_is_healthy`` to inject a rollback -- is honoured here too.  Bound by
# value they would not be, and the two drivers could then only be compared
# through ``bp_gauge_checkerboard``, i.e. never through this function.
import tenax.algorithms.ipeps_bp_gauge as _bp
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

# No ``__all__`` until Phase 4.
#
# This module deliberately declares no exports.  ``gauge_fix``,
# ``absorb_weights``, ``ctm_rdm2x1_planar`` and ``torus_2x2_sign_free`` are
# public-*named* because Phase 4 (#882) makes ``gauge_fix`` part of the
# supported top-level API once the simple-update engine that consumes it
# exists.  Until that lands, none of them is re-exported from
# ``tenax/__init__.py`` or documented in ``README.md``, so an ``__all__``
# listing them advertised a public surface that does not resolve as
# ``tenax.X``.
#
# An earlier revision kept the list, defended by "39 of the 46 modules under
# ``src/tenax`` that define ``__all__`` name at least one symbol absent from
# ``tenax.__all__``".  That count is real but does not support the claim, and
# it was used to rebut a review finding, so the record is corrected rather than
# quietly dropped: 37 of those 46 modules are ``_``-prefixed, where divergence
# is definitional, and a symbol is also reachable when ``tenax/__init__.py``
# registers it in ``_LAZY_IMPORTS`` instead of listing it in ``__all__``.
# Counting public-named symbols of public-named modules against
# ``tenax.__all__ | _LAZY_IMPORTS``, exactly **two** diverge -- this module and
# ``ipeps_optimize_root_implicit`` -- while the nearest sibling goes the other
# way, all three of ``ipeps_bp_gauge.__all__`` resolving as ``tenax.X``.  One
# precedent is not a norm, so the finding was right and the list is gone.
#
# Add ``__all__`` back in Phase 4, together with the ``tenax/__init__.py``
# export and the ``README.md`` entry, as one change.


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


def _identity_weights(A: Tensor) -> BondWeights:
    """``BondWeights.ones`` at this pair's two bond dimensions.

    Correct as the *incoming* weights precisely because the pair is in absorbed
    form: it already carries its own ``lambda``, so the bonds between the
    tensors handed to the solve really are unweighted.  Read off ``A``'s ``r``
    and ``d`` legs rather than assuming a square ``D``: the chain anchor's PEPS
    embedding has dimension-1 vertical legs.
    """
    labels = A.labels()
    return BondWeights.ones(
        A.indices[labels.index("r")].dim, A.indices[labels.index("d")].dim
    )


@partial(jax.jit, static_argnums=(2, 3))
def _gauge_fix_traced(A: Tensor, B: Tensor, max_iter: int, tol: float):
    """:func:`gauge_fix`'s dense route, whole, behind **one** jit boundary.

    Everything :func:`gauge_fix` does is in here except the three casts that
    rebuild :class:`BPGaugeInfo` at its documented ``(int, float, bool)`` type.
    That is the point.  Tracing the solve alone (Task 7) left the boundary
    around it eager, and on a warm D=2 solve the boundary was the *majority* of
    the cost: 2.47 ms total, of which the ``lax.while_loop`` was 0.89 ms and the
    rest was ``BondWeights.ones``, the entry point's validation and casts, and
    -- the bulk -- :func:`absorb_weights`' eight ``scale_bond_axis`` calls, each
    an eager dispatch with an eager ``jnp.sqrt`` in front of it.  Eight tiny
    dispatches cost far more than the arithmetic they carry, which is the same
    99.75%-host shape the sweep itself had.  Measured here: **0.91-0.94 ms**,
    from 2.47.  Re-gauging every simple-update step is on the *step* budget, so
    that 1.5 ms is multiplied by ``num_imaginary_steps``, and it is what took
    the #882 gate from 528 ms to **379-391 ms** against a 450 ms budget (and
    the cold-process accounting from 702 ms to 402-422), over six fresh
    processes on a quiet 128-core box.

    ``_validate_weights`` is deliberately not called: the weights here are
    :func:`_identity_weights`' own ``ones``, so there is no caller input to
    reject and the documented ``ValueError`` is unreachable.  The pair itself is
    not validated by that function on any path.

    **Cache key.**  ``(A, B)`` are pytrees whose ``TensorIndex`` metadata is
    *aux* data, so the key is shape, dtype **and flow convention**, plus the
    static ``max_iter``/``tol``.  The flow half is not a curiosity: a
    simple-update-evolved pair has its four virtual flows inverted relative to
    ``_wrap_as_dense_tensor``, so mixing two conventions inside one run costs a
    second ~285 ms compile -- most of the budget.  What keeps a run to one
    convention is ``_restore_caller_structure`` handing each caller its own
    metadata back instead of stamping this module's onto it;
    ``test_every_solve_in_a_run_hits_one_compiled_entry`` pins all three splits.

    Calls :func:`~tenax.algorithms.ipeps_bp_gauge._bp_solve`, the unjitted
    driver, rather than its jitted alias: a nested ``jit`` traces and inlines
    correctly but leaves the inner executable cache empty, so the counter that
    gate reads would sit at zero while the real key lives out here.
    """
    weights = _identity_weights(A)
    gam, w, residual, done, converged, _ = _bp._bp_solve(A, B, weights, max_iter, tol)
    A_out, B_out = absorb_weights(gam["A"], gam["B"], w)
    # Already true by construction -- the driver rebuilds every iterate on the
    # caller's own indices -- and applied anyway, because it is free under trace
    # and it is what makes "absorbed form in, absorbed form out" a property of
    # this function rather than of something it calls.
    return (
        _bp._restore_caller_structure(A_out, A),
        _bp._restore_caller_structure(B_out, B),
        w,
        residual,
        done,
        converged,
    )


def gauge_fix(
    A: Tensor,
    B: Tensor,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[Tensor, Tensor, BondWeights, BPGaugeInfo]:
    """Re-derive the BP gauge of an **absorbed-form** pair.

    Absorbed form in, absorbed form out.  Takes no incoming weights, because
    there are none to hand over: the pair already carries them.  Internally
    this is the BP solve started from ``BondWeights.ones`` -- correct precisely
    because the absorbed tensors already *are* ``Gamma_A lambda Gamma_B`` --
    followed by :func:`absorb_weights` on the Vidal pair that comes back, which
    is what keeps the boundary convention intact.  Vidal form therefore exists
    only *between* those two steps and is never returned (#882 §3).  On a dense
    pair the two steps are one compiled call and Vidal form never reaches the
    host at all; see *Performance* below.

    Args:
        A, B:     Absorbed-form site tensors, labels ``(u,d,l,r,phys)``.
        tol:      Stop once the largest relative weight change falls below this.
        max_iter: Maximum BP sweeps.

    Returns:
        ``(A, B, weights, info)``.  The pair alone **is** the state: read it
        with :meth:`BondWeights.ones`, or with nothing at all.

        ``weights`` is the Schmidt spectrum of that state at the BP fixed
        point, and it is **already absorbed into the returned pair**.  It is a
        diagnostic and a truncation input -- dropping it loses a report, not
        physics.  Do **not** absorb it again, and do not pass it back into a
        contraction alongside the pair: either double-counts every bond, which
        is the ``lambda**1.5`` mechanism of #667 verbatim.  Measured, feeding
        it back moves the state by 9.0e-02 to 8.4e-01 depending on the pair
        (``test_gauge_fix_returns_an_absorbed_pair_not_a_vidal_one``).

    Performance:
        A **dense** pair takes :func:`_gauge_fix_traced` -- the solve *and* the
        conversion back to absorbed form as one compiled call, so a warm solve
        is ~0.92 ms rather than 2.47 and the only host syncs left are the three
        casts below.  A ``SymmetricTensor`` pair takes the eager route
        (``bp_gauge_checkerboard`` then :func:`absorb_weights`), which is the
        original code path, unchanged and bit-identical; the two blockers on
        tracing it live outside both modules and are named in
        ``ipeps_bp_gauge``'s docstring.
    """
    if _bp._use_traced_loop(A, B):
        A_out, B_out, weights, residual, done, converged = _gauge_fix_traced(
            A, B, max_iter, tol
        )
        # The whole traced route's host syncs, all three of them, and they are
        # not optional: ``BPGaugeInfo`` is documented as ``(int, float, bool)``,
        # and a 0-d array would satisfy ``assert info.converged`` *silently*
        # while failing ``info.residual == float("inf")`` loudly.
        return (
            A_out,
            B_out,
            weights,
            BPGaugeInfo(int(done), float(residual), bool(converged)),
        )

    A_v, B_v, weights, info = bp_gauge_checkerboard(
        A, B, _identity_weights(A), tol=tol, max_iter=max_iter
    )
    # bp_gauge_checkerboard returns Vidal ``Gamma lambda Gamma``, which is its
    # documented and correct contract.  Converting here rather than there is
    # deliberate: this is the boundary the spec draws.  (It is not that other
    # callers need the Vidal form -- ``src/`` has none; this function is
    # ``bp_gauge_checkerboard``'s only in-repo caller.  The spec is the whole
    # reason.)  Returning the raw pair moved the state by up to 1.25 (dense
    # D=2) when read as the docstring above promises.
    A_out, B_out = absorb_weights(A_v, B_v, weights)
    return A_out, B_out, weights, info


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
        Koszul signs enter Tenax through exactly two groups of call sites, and
        the enumeration is checkable by grepping ``_koszul_sign`` (defined in
        ``core/tensor.py``) rather than by memory: ``SymmetricTensor.transpose``
        is one of them, and the ``linalg`` decompositions -- svd (three
        variants), rsvd, qr (two), eigh -- are the other seven.  ``fuse`` is
        **not** among them, despite an earlier version of this warning saying
        so: ``_fuse_indices_symmetric`` permutes each block with a bare
        ``jnp.transpose`` and never asks the symmetry for a parity, and the
        ``Symmetry.fuse`` methods are charge-addition rules with nothing to do
        with statistics.  ``bar()`` applies no sign either.  This function
        calls nothing in either group, so it is categorically sign-free, and a
        closed 2x2 torus is exactly the non-planar case that carve-out
        excludes -- its wrap-around edges cross the interior.  Measured on
        ``FermionParity`` pairs it is value-identical to the ``np.einsum``
        probe it was once meant to replace (2.7e-16 at D=2, 3.0e-16 at D=3),
        i.e. it adds no fermionic information whatsoever.  Use
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

    The gauge-invariance witness that is **usable on fermions**, which
    :func:`torus_2x2_sign_free` is not.  The whole point is that this stays
    inside the regime the contractor's sign convention is documented for:
    #555 declines to auto-apply Koszul signs and justifies that for *planar*
    networks, "the only kind Tenax's CTM/RDM/energy code uses".  This is that
    code -- ``ctm_tensor_2site`` then ``_rdm2x1_tensor_2site`` -- so the
    contraction is sign-correct by the same argument that makes the shipped
    fermionic energies sign-correct.  A closed torus is not.

    That validity is *inherited*, not independently established: it is exactly
    as strong as #555's planar carve-out and no stronger.  Task 5 measured a
    state on which that convention and the Koszul sign
    ``SymmetricTensor.transpose`` applies disagree, and did **not** settle
    which of the two is wrong (#882 §5.2).  So read this as "the same sign
    convention the shipped fermionic energies use", not as "certified
    graded-correct".

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

    **Resolution differs by arm, and the good number is the dense one.**  CTM
    keeps only ``chi`` environment states and a gauge changes the virtual basis
    that truncation is taken in, so an exactly-gauged pair does not reproduce
    the ungauged one to machine precision; the residual is the witness's noise
    floor.  On a *dense* pair that floor falls steeply with ``chi`` (D=2:
    5.5e-04 at ``chi=4`` down to 2.5e-09 at ``chi=32``).  On a *fermionic* pair
    at D=2 it does not: 3.13e-03 (``chi=6``), 1.06e-03 (``chi=8``), 1.24e-03
    (``chi=12``) -- a plateau around 1e-3 over a 2x change in ``chi``.  All five
    numbers are from ``task-4-report.md``; none was re-measured here.

    Two consequences, and no third.  (1) A fermionic violation smaller than
    ~1e-3 at small ``chi`` is invisible to this witness, so compare against a
    known-exact gauge on the same state at the same ``chi``, never against
    zero.  (2) Flat-in-``chi`` is this project's own defect signature, so that
    plateau is **not** established as truncation -- it is unexplained.  A
    gauge-strength scan was once offered as independent evidence that it is
    truncation; it is not independent, because as the gauge tends to the
    identity the gauged and reference runs become the same computation and the
    displacement must fall whatever the cause.  The question is deferred, not
    answered: see #882 §5.2.  (Related, and also deferred: at D=3 the fermionic
    floor *does* fall, 2.24e-03 at ``chi=8`` to 6.25e-04 at ``chi=20``, per
    ``task-5-report.md`` §1 -- so the plateau may be specific to D=2, where
    both parity sectors have size 1.  Read those two rows side by side with
    care: **they are different gauges**, not one gauge at two ``D``.  The D=2
    plateau is a single-bond *diagonal* gauge, ``g = [1.7, 0.55]`` on ``A.r``
    with ``1/g`` on ``B.l`` (``task-4-report.md``); the D=3 numbers are a
    four-bond *non-diagonal* parity-block gauge at spread 1.5
    (``task-5-report.md``'s ``nd15``).  So "falls at D=3, flat at D=2" is not
    yet a statement about ``D``.)

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
