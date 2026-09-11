"""Belief-propagation gauge for the 2-site checkerboard iPEPS.

Simple update stores each bond's Schmidt spectrum straight from the SVD that
produced it and never recomputes it.  That is only valid while the state does
not move: a *non-unitary* gate applied to a neighbouring bond changes the
Schmidt values on this one, so the stored weights drift away from the spectra
they are taken to be.  Both reference implementations avoid this rather than
tolerate it --- TeNPy carries a separate ``update_bond_imag`` for imaginary time
whose whole point is to "sweep left or right *without using old singular
values*", and YASTN's ``EnvBP.post_truncation_`` recomputes the messages on a
bond, in both directions, after every truncation.

This module supplies the missing step.  Bond weights on a PEPS are belief-
propagation messages, so re-deriving them is one BP fixed-point solve
(Tindall & Fishman, *Gauging tensor networks with belief propagation*, SciPost
Phys. 15, 222 (2023)), specialised here to the 2-site checkerboard.

Measured on the shipped simple update at its own converged state, the stored
weights are **not** BP-self-consistent --- 15% off on the second Schmidt value
and ~35% on the tail::

    D=3   stored [1, 0.16586, 0.01564]   BP [1, 0.14243, 0.01130]
    D=4   stored [1, 0.16875, 0.01732, 0.01289]
                                         BP [1, 0.14534, 0.01258, 0.01017]

so anything that reads ``lambda`` as a Schmidt spectrum --- entanglement
entropy, a truncation-error estimate, the symmetric gauge handed to a CTM ---
is reading a drifted number without this.

.. note::
    This fixes the *weights*, not simple update's dynamics.  It does not
    rescue the four-independent-spectra sweep of #851 at ``D >= 3``: the
    diverged state's BP-consistent weights are the diverged ones, so the
    bond weights were never that defect.  See #869.

What one sweep does
-------------------
1. Recompute all eight messages from the current ``Gamma`` tensors.  In the
   gauge this function leaves behind, messages are diagonal (``= lambda**2``),
   so an incoming message is applied by scaling ``Gamma`` on that leg --- ket
   and bra pick up one factor each.
2. Re-gauge each bond.  Writing the two messages as ``mL = X^dag X`` and
   ``mR = Y^dag Y``, and SVD-ing the object the bond actually carries in Vidal
   form, ``X lambda_old Y^T = U S V^dag``, the insertion

   .. math::

       I = X^{-1} (X \\lambda_{old} Y^T) Y^{-T}

   leaves the state untouched and makes the bond weight ``S``::

       Gamma_L <- Gamma_L (X^-1 U)     Gamma_R <- (V^dag Y^-T) Gamma_R
       lambda  <- S

   Dropping ``lambda_old`` from that SVD makes this not a gauge transformation
   at all --- it moved the energy by 1.2e-01 in testing.

3. Rescale each site.  This is bookkeeping, not physics --- an overall scale on
   ``Gamma`` is not observable and ``lambda`` is separately max-normalised ---
   but it has to happen after every *bond*.  ``X^-1`` is a pseudo-inverse
   square root, so a bond whose message has small eigenvalues multiplies
   ``||Gamma||`` by a large factor, and the factor grows as the spectrum
   decays.  Rescaling once per sweep lets four of them compound: on an
   SU-evolved U(1)-Sz ``D=3`` state that reached ``||Gamma|| ~ 1e112`` by sweep
   59 and then ``inf``, after which normalising returns exactly **zero** ---
   and zero is an absorbing fixed point that the residual test certifies as
   converged, since ``norm(0 - 0) / max(0, 1e-300)`` is 0.  The residual fell
   monotonically to 9.8e-12 the whole way down, so nothing in the convergence
   report hinted at it.  Hence :func:`_is_representable`, checked every sweep:
   an iterate that has left f64 is not a solution, and must not be reported as
   one.

   The rescale is by **max-abs**, never the Frobenius norm, and the input is
   rescaled before the first message rather than only between sweeps.  Both for
   the same reason: ``_message`` and ``||Gamma||`` each square before they sum,
   so a caller handing over the same state with an overall prefactor of 1e-200
   or 1e200 -- unobservable, by definition -- underflows or overflows the first
   message, while a norm-based rescale is itself 0 or ``inf`` there and so
   silently does nothing about it.

The four bonds touch a different leg of each tensor, so their gauge
transformations commute and are applied together.

Why the dense path is traced
----------------------------
Simple update re-gauges after *every* step, so a solve is on the step budget
rather than the run budget.  Run eagerly one sweep costs **18.9 ms**, of which
almost nothing is arithmetic: the identical arithmetic under ``jit`` is
**0.034 ms**, so the eager sweep is ~99.8% host overhead.  (Measured on a quiet
128-core machine at ``JAX_PLATFORMS=cpu``: a D=2 simple-update-evolved pair
solves in 26 sweeps, 490.6 ms eager against 2.44 ms traced.)  The carrier is
~300 eager dispatches per sweep plus tenax-level Python -- label bookkeeping,
``TensorIndex`` construction, ``opt_einsum`` expression lookup.  The host-side
``float()``/``bool()`` syncs are a *minority* contributor, and the count is
**history, not a description of this file**: on ``main`` a sweep did 18 of them,
and removing all 18 without a traced loop was worth perhaps 1.2-1.4x, not 200x.
Tracing the solve took the count to **2** on the way past --
``_is_representable`` and ``_residual`` return 0-d arrays now, so the eager
driver syncs only on the health gate and the residual (see
:func:`_bp_solve_eager`, which states the current number).  Do not read the 18
as current; the point it carries is that the syncs were never the carrier, so
the fix is to trace the whole solve.

:func:`_bp_solve` is that: one ``lax.while_loop`` over a six-slot carry,
compiled once per ``(D, dtype, carry treedef, max_iter, tol)`` and reused by
every subsequent solve.  ``lax.while_loop`` rather than a Python loop over a
jitted body is not a preference: at 0.143 ms per dispatch a jitted *body* costs
3.7 ms for a 26-sweep solve, which is 1.5x the whole traced solve and eats the
entire warm allowance on its own.

It is kept **jittable rather than jitted**, with :data:`_bp_solve_traced` as the
boundary this module's own entry point uses, because the solve is not the whole
of what a caller wants compiled: ``ipeps_gauge.gauge_fix`` wraps it together
with ``absorb_weights`` in one jit, and *that* boundary is what a
simple-update step pays.  Left as two nested jits the outer one still works, but
the inner executable cache stays empty and the one-compile gate would be
counting a cache nothing reaches.

The remaining cost is **compile**, and it is the binding one: 84 ms to trace and
lower plus 214 ms of XLA, on a 324-equation sweep body, *flat in D* (214/216/224
ms at D=2/3/4 -- structure-bound, not array-size-bound, unlike #633's CTM
finding).  Against the rewrite's 450 ms budget for a 100-step run that is 66%
before a single solve runs; see ``tests/test_ipeps_gauge_perf.py``, whose gate
*asserts* that budget (it recorded a shortfall while only the solve was traced,
and stopped once ``gauge_fix``'s own boundary went inside the jit too).

``SymmetricTensor`` took the eager loop until both of the things that stopped
it being traced were fixed, the same way:

1. ``_eigh_symmetric`` laid its bond out by ranking the whole spectrum, reading
   the eigenvalues through ``np.array(...)``, which raises on a tracer.  Fixed
   in #939: :func:`_sqrt_pinv` never truncates, so the ranking decides nothing
   there, and it asks for ``eigh(..., bond_order="sector")`` -- the
   charge-grouped layout, which needs no host read.
2. The SVD in :func:`_gauge_bond` read the spectrum the same way, and its
   tracer reroute was worse than a crash: the rerouted path's per-sector floor
   zeroes any singular value below ``1e-12 * (s_max + 1e-30)``, and on the 1x1
   sectors these bonds carry the ``+1e-30`` arm is an *absolute* ~1e-42
   cutoff.  A measured ``4.6e-43`` singular value came back exactly 0.0, the
   zeroed direction carried 13.6% of the 2-site norm at the collapsed state
   that reaches it, and iterating past the health gate reproduced the
   historical 3.0e-01 "stopped being a gauge" drift bit-for-bit.  The bond
   *order* -- the other suspect -- contributes exactly zero: every ``lam`` is
   consumed positionally against the very bond its own decomposition emitted.
   Fixed the same way as (1): ``svd(..., bond_order="sector")`` runs the one
   eager code path under the tracer, floor-free.  The floor is not a
   tightenable knob, it is semantically wrong here -- legitimate f64-walk
   trajectories dip to ~1e-27 relative and recover, so *any* relative floor
   breaks states this module handles fine.

With both fixed the sweep body compiles and is exact -- measured at ``D=3``,
921 ms eager against 0.200 ms traced, one compile (4.6 s) paying for itself
after 4.9 sweeps, block structure preserved and the charge layout stable
across sweeps -- so a ``SymmetricTensor`` pair now takes the same traced
driver a dense pair does (see :func:`_bp_solve` for how its carry holds
block buffers).  The sweep body is shared verbatim between the two loops, so
there is one implementation of the physics and only the driver differs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tenax.algorithms._ctm_tensor_moves import _flow_flip_no_conj

# Re-exported so ``from tenax.algorithms.ipeps_bp_gauge import BondWeights``
# keeps working.  It was defined here first (#870), but the simple update needs
# the same four bonds and is the lower layer, so it owns the type -- one class,
# not two structurally identical ``NamedTuple``s that would silently
# type-check against each other (#851).
from tenax.algorithms.ipeps_simple_update import BondWeights
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
from tenax.linalg import eigh, svd

__all__ = ["BPGaugeInfo", "BondWeights", "bp_gauge_checkerboard"]

# Relative cutoff for the 1/sqrt(w) pseudo-inverse of a message.  A message
# eigenvalue at zero is a bond direction the state does not use; inverting it
# would turn 0 into inf rather than project it out.
#
# Kept as a ``jnp.where`` mask rather than a rank-truncating slice: the mask
# leaves the ``__k`` bond dimension static, which is what makes the traced
# carry shape-stable.  A dynamic slice here would break the while_loop.
_PINV_CUTOFF = 1e-12

#: A weight this small, relative to its bond's largest, is on its way to zero
#: rather than describing the state.  Used only by
#: :func:`_a_weight_underflowed`, and **bracketed by measurement on both
#: sides** rather than chosen:
#:
#: * a direction the state genuinely does not use dies from a relative
#:   ``1.0`` -- it is full-sized on the sweep before it goes, and goes in one
#:   step (measured on the starved pair of
#:   ``test_su_step_survives_a_bond_direction_the_state_does_not_use``);
#: * a weight that underflows dies from ``1.1e-08`` at the very worst, and
#:   typically ``3e-10``, after decaying geometrically for tens of sweeps.
#:
#: So anything from ~1e-7 to ~1e-2 separates them.  This sits two orders above
#: every fatal observation and six below the legitimate one.
_UNDERFLOW_EPS = 1e-6


_BRA = "__bra"
_K = "__k"
_K2 = "__k2"
_S = "__s"
_B = "__b"

#: Guard against ``0/0`` in the relative residual.  Reachable only through a
#: weight vector the entry point's validation would have rejected, so it is a
#: floor rather than a policy.
_RESIDUAL_FLOOR = 1e-300


class BPGaugeInfo(NamedTuple):
    """Convergence report for :func:`bp_gauge_checkerboard`."""

    iterations: int
    residual: float
    converged: bool


# Which bond sits on each leg of each site.  This is the same map
# ``_to_physical_pair`` uses, written once.
_BOND_OF: dict[tuple[str, str], str] = {
    ("A", "u"): "v_BA",
    ("A", "d"): "v_AB",
    ("A", "l"): "h_BA",
    ("A", "r"): "h_AB",
    ("B", "u"): "v_AB",
    ("B", "d"): "v_BA",
    ("B", "l"): "h_AB",
    ("B", "r"): "h_BA",
}
# Each bond as (name, left/upper end, right/lower end).
_BONDS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("h_AB", ("A", "r"), ("B", "l")),
    ("h_BA", ("B", "r"), ("A", "l")),
    ("v_AB", ("A", "d"), ("B", "u")),
    ("v_BA", ("B", "d"), ("A", "u")),
)
_LEGS = ("u", "d", "l", "r")


def _message(gamma: Tensor, site: str, out_leg: str, weights: BondWeights) -> Tensor:
    """Outgoing BP message of ``site`` along ``out_leg``.

    Incoming messages on the other three legs are diagonal in this gauge, so
    scaling ``Gamma`` by ``lambda`` there gives the ket one factor and the bra
    the other, i.e. ``lambda**2`` across the pair.
    """
    g = gamma
    for leg in _LEGS:
        if leg != out_leg:
            g = scale_bond_axis(g, leg, getattr(weights, _BOND_OF[(site, leg)]))
    # ``bar`` is the bra operation: conjugate with flows flipped, so the shared
    # legs contract.  ``conj`` alone leaves the flows unflipped and silently
    # collapses the message to a single charge sector.
    return contract(g, g.bar().relabel(out_leg, _BRA))


def _sqrt_pinv(m: Tensor, out_leg: str) -> tuple[Tensor, Tensor]:
    """Factor a PSD message ``m = X^dag X``; return ``X`` and ``X^-1``."""
    # ``bond_order="sector"`` rather than the default magnitude ranking: this
    # call never truncates, so the ranking decides nothing, and asking for it
    # is what pinned the SymmetricTensor pair to the eager driver -- the rank
    # is read through ``np.array`` on the eigenvalues, which raises on a tracer.
    # ``s`` is used through ``jnp.max`` and paired with ``V`` column by column,
    # so nothing here depends on which order the bond comes back in.
    V, w = eigh(
        m,
        left_labels=[out_leg],
        right_labels=[_BRA],
        new_bond_label=_K,
        bond_order="sector",
    )
    w = jnp.clip(w, 0.0, None)
    s = jnp.sqrt(w)
    keep = s > _PINV_CUTOFF * jnp.max(s)
    s_inv = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
    # m == V diag(w) V.bar(), so X = diag(sqrt w) V.bar() and X^-1 = V diag(1/sqrt w).
    return scale_bond_axis(V.bar(), _K, s), scale_bond_axis(V, _K, s_inv)


def _gauge_bond(
    gam_L: Tensor,
    gam_R: Tensor,
    leg_L: str,
    leg_R: str,
    msg_L: Tensor,
    msg_R: Tensor,
    lam: jax.Array,
) -> tuple[Tensor, Tensor, jax.Array]:
    """Re-gauge one bond from its two messages; return both ends and the weight.

    Exact: ``gam_L lam gam_R`` is unchanged, to machine precision.  That is the
    property :func:`bp_gauge_checkerboard` rests on and the one worth testing,
    since every error mode here (a dropped ``lam``, a same-flow contraction)
    breaks it while leaving a plausible-looking spectrum behind.
    """
    X, X_inv = _sqrt_pinv(msg_L, leg_L)
    Y, Y_inv = _sqrt_pinv(msg_R, leg_R)
    # The bond carries ``lam`` in Vidal form, so the object being re-gauged is
    # X lam Y^T, not X Y^T.  Dropping it is not a gauge transformation.
    XL = scale_bond_axis(X.relabel(leg_L, _B), _B, lam)
    M = contract(XL, Y.relabel(leg_R, _B).relabel(_K, _K2))
    # ``bond_order="sector"`` for the same reason :func:`_sqrt_pinv` asks
    # eigh for it: this call never truncates, so the ranking decides
    # nothing, and sector mode is the one that runs under a tracer as the
    # same code path it runs eagerly.  The default's tracer reroute is not
    # merely a permutation -- its per-sector floor zeroes real ~1e-43
    # singular values (absolute, via the ``+1e-30`` arm on 1x1 sectors),
    # and a gauge built from a floored SVD is not a gauge: measured
    # 3.0e-01 state drift on the whole solve.  ``lam_new``/``U``/``Vh``
    # pair with the ``_S`` bond positionally through ``scale_bond_axis``,
    # so any single consistent layout is exact; dense pairs ignore the
    # argument entirely.
    U, s, Vh, _ = svd(
        M,
        left_labels=[_K],
        right_labels=[_K2],
        new_bond_label=_S,
        bond_order="sector",
    )
    smax = jnp.max(s)
    lam_new = s / jnp.where(smax > 0, smax, 1.0)

    # ``X_inv``'s bond leg carries the SAME flow as Gamma's, so contracting them
    # directly is a same-flow contraction: on a SymmetricTensor that silently
    # collapses charge sectors rather than raising (measured: the gauge stopped
    # being a gauge, 2.7e-01 on the 2-site object, while DenseTensor stayed
    # exact at 5e-16).  Flip the flows without touching the data -- charges, and
    # hence block keys, are unchanged.
    G_L = _flow_flip_no_conj(contract(X_inv, U))
    G_R = _flow_flip_no_conj(contract(Vh, Y_inv.relabel(_K, _K2)))
    return (
        contract(gam_L, G_L).relabel(_S, leg_L),
        contract(gam_R, G_R).relabel(_S, leg_R),
        lam_new,
    )


def _rescale(t: Tensor) -> Tensor:
    """Scale ``t`` to unit max-abs.

    Max-abs, never the Frobenius norm, because the norm squares before it
    sums: ``||Gamma||`` of a state whose entries are ~1e-200 underflows to
    exactly 0 and ~1e200 overflows to ``inf``, so a norm-based rescale silently
    does nothing on precisely the inputs that need it most.  An overall scale
    on ``Gamma`` is not observable, so this is free.

    No additive epsilon in the denominator: that would make the result depend
    on the input's scale, which is the bug this exists to prevent.
    """
    m = t.max_abs()
    return t * (1.0 / jnp.where(m > 0, m, 1.0))


def _is_representable(gam: dict[str, Tensor], new_weights) -> jax.Array:
    """Is the iterate still a state, rather than an overflow artefact?

    Checked every sweep because the convergence test cannot see this: an
    all-zero iterate is an absorbing fixed point that reports ``residual = 0``.

    Both clauses matter, and the ``> 0`` one is not implied by finiteness: zero
    is finite, and ``norm(0 - 0) / max(0, 1e-300)`` is 0, which passes any
    tolerance.  Losing that clause does not crash -- it produces a confidently
    "converged" corpse.

    Returns a **0-d ``jnp`` bool**, not a Python ``bool``, so the traced driver
    can use it inside a ``while_loop``; the eager driver wraps it in ``bool()``.
    Scale-invariant by construction (max-abs, no norm), so an unobservable
    prefactor of 1e-200 or 1e200 is healthy.

    Args:
        gam:         ``{"A": ..., "B": ...}`` site tensors.
        new_weights: The sweep's candidate bond weights.  Any pytree of weight
                     vectors -- a :class:`BondWeights` from :func:`_sweep`, or
                     a plain ``dict`` from a unit test.
    """
    ok = jnp.asarray(True)
    for t in gam.values():
        n = t.max_abs()
        ok = ok & jnp.isfinite(n) & (n > 0.0)
    for w in jax.tree_util.tree_leaves(new_weights):
        ok = ok & jnp.all(jnp.isfinite(w)) & (jnp.max(w) > 0.0)
    return ok


def _a_weight_underflowed(new_weights, old_weights) -> jax.Array:
    """Did a weight that was already collapsing reach exactly zero?

    :func:`_is_representable` cannot see this.  It is scale-invariant by design,
    and a bond that loses its smallest weight keeps ``max(lambda) = 1`` -- so a
    *partial* collapse passes every clause of it while a *total* one does not.
    Measured on a D=3 U(1)-Sz pair: the smallest weight on each bond decayed
    geometrically (1.1e-08, 9.1e-10, 3.3e-10 ...) and reached exactly 0.0 at
    sweep 109.  ``_sqrt_pinv`` then had no direction left to invert, so the
    transformation stopped being a gauge, and by sweep 114 the solve reported
    ``residual = 1.19e-16`` and *converged* on a state that had moved by
    3.0e-01, with the health gate returning True throughout.  #870 is the same
    failure with the sign flipped -- growth to ``inf`` there -- and in both the
    residual certifies the corpse.

    **A weight reaching zero is not by itself wrong**, which is why this asks
    where it came *from*.  A direction the state genuinely does not use dies
    from a relative 1.0 in a single sweep, and refusing that breaks a real
    solve: an earlier version of this check counted rank instead, and rejected
    ``test_su_step_survives_a_bond_direction_the_state_does_not_use`` on its
    very first sweep.  Only a weight that was already collapsing --
    below :data:`_UNDERFLOW_EPS` of its bond's largest -- and then hit zero is
    the failure this describes.

    **The first sweep is exempt, because the caller's stored weights are not a
    trajectory.**  They are exactly the drifted numbers this module exists to
    discard, so "was collapsing" cannot be judged against them: a state whose
    unused direction carries a stored weight of 1e-8 is the same state as one
    carrying 1.0 there, and only the caller's arbitrary number would separate
    "accepted" from "rejected at sweep 0" (Codex P2 on #940 -- measured: the
    starved pair converges in 52 sweeps from a tail of 1.0 and was refused with
    0 iterations from a tail of 1e-8).  Both call sites therefore consult this
    only from the second sweep on, where both operands are the solve's own
    iterates.  The fatal trajectory dies at sweep 109; nothing is lost.

    **Shape changes are not inspected.**  A bond weight may legitimately change
    length between sweeps when a charge sector empties (#904/#906), and the two
    vectors then cannot be aligned entry by entry.  Such a sweep is accepted;
    the failure this exists for does not change any length (all four bonds stay
    at their width throughout the trajectory above).  The shape test is a
    Python-level branch on static shapes, so it costs nothing under trace.

    Returns a 0-d ``jnp`` bool, like :func:`_is_representable`, so both drivers
    can use it -- the traced one inside its ``while_loop``.
    """
    bad = jnp.asarray(False)
    for new, old in zip(
        jax.tree_util.tree_leaves(new_weights),
        jax.tree_util.tree_leaves(old_weights),
        strict=True,
    ):
        if new.shape != old.shape:
            continue
        m = jnp.max(old)
        was_collapsing = (old > 0) & (old < _UNDERFLOW_EPS * jnp.where(m > 0, m, 1.0))
        bad = bad | jnp.any((new == 0) & was_collapsing)
    return bad


def _sweep_is_healthy(gam: dict[str, Tensor], new_weights, sweep) -> jax.Array:
    """Health gate on one completed sweep; ``sweep`` is its 0-based index.

    A thin seam over :func:`_is_representable`, which does the actual work and
    ignores ``sweep``.  It exists because the two drivers must be testable the
    same way: inside ``lax.while_loop`` there is no host-side call count, so a
    test cannot place an injected failure at a chosen sweep by counting calls
    the way it can on the eager loop.  Passing the index -- which both drivers
    have anyway, as the Python loop variable and as the carry's ``done`` slot --
    makes ``lambda gam, w, sweep: sweep < 2`` a valid injection on both.

    ``done`` equals the sweep index at this point in either driver, because the
    counter is incremented *after* this check and the loop stops the first time
    it fails.
    """
    return _is_representable(gam, new_weights)


def _residual(new: BondWeights, old: BondWeights) -> jax.Array:
    """Largest relative change across the four bond weights, as a 0-d array.

    ``jnp.max`` over a stacked vector rather than Python ``max`` over four
    ``float()``s: the point is that this is called from inside a traced loop,
    where a host sync is not available.  Two consequences on the eager path
    too, since both drivers share it -- residuals move at the 1e-15 level
    (measured 7.0562e-13 vs 7.0430e-13 on the same state), and NaN now
    *propagates* instead of depending on Python's comparison order.
    """

    def _one(n: jax.Array, o: jax.Array) -> jax.Array:
        # Zero-pad to the common length before subtracting (#904).  A bond's
        # weight vector carries one entry per *non-empty* block, not per slot of
        # the leg, so it changes length whenever a charge sector dies or
        # revives: ``_gauge_bond`` takes ``lam_new`` from a block-sparse ``svd``
        # and hands back tensors whose bond leg is that same new bond, so the
        # leg and the weights shrink *together* and the state stays
        # self-consistent -- measured on a U(1)-Sz D=3 pair, sweep 56, where
        # ``v_BA`` went 3 -> 2 and ``Gamma_A.u``/``Gamma_B.d`` both came back at
        # dim 2 on charges ``[-1, 1]``.  What does not survive is the comparison
        # *across* sweeps, which is the only place the two lengths meet.
        #
        # Padding is right here for the same reason it is right in
        # ``_ctm_tensor_convergence._ctm_sv_diff`` (#670): this is a convergence
        # *indicator*, not a per-sector difference, and a vector that changed
        # shape must register as "still moving" rather than raise.  It is not
        # the #834 hazard -- nothing downstream indexes ``new`` against ``old``;
        # each is used only with the leg it came back with.
        k = max(n.shape[0], o.shape[0])  # Python max: jnp.pad needs a STATIC width
        n = jnp.pad(n, (0, k - n.shape[0]))
        o = jnp.pad(o, (0, k - o.shape[0]))
        return jnp.linalg.norm(n - o) / jnp.maximum(jnp.linalg.norm(o), _RESIDUAL_FLOOR)

    return jnp.max(jnp.stack([_one(n, o) for n, o in zip(new, old, strict=True)]))


def _sweep(
    gam: dict[str, Tensor], weights: BondWeights
) -> tuple[dict[str, Tensor], BondWeights]:
    """One BP sweep: recompute all eight messages, then re-gauge all four bonds.

    The physics, shared **verbatim** by the eager and the traced driver so
    there is one implementation of it and only the loop differs.  Pure: the
    input ``gam`` is not mutated, which is what lets the rollback be a
    rejection of the candidate rather than a restore from a saved copy.

    Every step is a gauge transformation, so the returned pair represents the
    same physical state as the input pair -- to machine precision, on all four
    bonds at once.
    """
    msg = {
        (site, leg): _message(gam[site], site, leg, weights)
        for site in ("A", "B")
        for leg in _LEGS
    }

    gam = dict(gam)
    new_weights: dict[str, jax.Array] = {}
    for bond, (site_L, leg_L), (site_R, leg_R) in _BONDS:
        gam[site_L], gam[site_R], new_weights[bond] = _gauge_bond(
            gam[site_L],
            gam[site_R],
            leg_L,
            leg_R,
            msg[(site_L, leg_L)],
            msg[(site_R, leg_R)],
            getattr(weights, bond),
        )
        # Rescale after every *bond*, not once per sweep.  ``X_inv`` is a
        # pseudo-inverse square root, so a bond whose message has small
        # eigenvalues multiplies ``||Gamma||`` by a large factor; letting four
        # of them compound before any rescale walks the iterate out of f64.
        # Measured on an SU-evolved U(1)-Sz D=3 state, per-sweep rescaling
        # reached ||Gamma|| ~ 1e112 by sweep 59 and then inf, after which
        # normalising returns exactly zero.  An overall scale on Gamma is not
        # physical and lambda is separately max-normalised, so doing it more
        # often cannot move the fixed point -- and ``_rescale`` performs no
        # sync and is free under trace, so there is no performance reason to
        # hoist it out of this loop either.
        for site in (site_L, site_R):
            gam[site] = _rescale(gam[site])
    return gam, BondWeights(**new_weights)


def _reorder(t: Tensor, labels: tuple[str, ...]) -> Tensor:
    """Restore ``labels`` as the axis order.

    ``contract`` returns legs in its own order, so a gauged tensor comes back
    as e.g. ``('phys','r','l','d','u')``.  Everything here is label-driven and
    does not care, but a caller indexing by position would, so the input's
    order is handed back.
    """
    current = t.labels()
    if current == labels:
        return t
    return t.transpose(tuple(current.index(lab) for lab in labels))


def _restore_caller_structure(t: Tensor, like: Tensor) -> Tensor:
    """Hand back the caller's axis order -- and, on a dense pair, its metadata.

    A swept tensor differs from its input in two ways.  ``contract`` returns
    legs in its own order, which :func:`_reorder` has always undone.  And
    :func:`_gauge_bond` stamps the flow its own SVD produced on every virtual
    leg -- ``IN`` on the left/upper end, ``OUT`` on the right/lower one -- which
    nothing undid: handed a pair using the opposite convention, the solve
    silently returned all four *virtual* flows inverted.  ``phys`` is not among
    them; ``_gauge_bond`` never touches it, so it always came back as handed
    over, which is what made the other four easy to miss.  That is not
    hypothetical.
    ``_simple_update_checkerboard_sweep``'s own output is such a pair, so the
    caller this module exists for was hitting it, and
    ``test_the_solve_converges_and_hands_back_the_same_tensor_structure``
    ("flows must survive, or callers break silently") passed only because every
    fixture it runs on already used this module's convention.

    On a ``DenseTensor`` the flows are inert -- ``contract`` pairs legs by
    label, and the #834 flow check is opt-in *and* satisfied under either
    convention, since each message is built from the very ``Gamma`` it will be
    contracted against -- so restoring them is metadata-only and makes that
    promise true for every input.

    A ``SymmetricTensor``'s indices are **not** restored.  There the charges are
    load-bearing, rewriting them would be exactly #834's silent mis-pairing, and
    the symmetric path keeps the stamped metadata it has always returned.
    """
    t = _reorder(t, like.labels())
    if isinstance(t, DenseTensor) and isinstance(like, DenseTensor):
        return DenseTensor(t.todense(), like.indices)
    return t


class _StructureNotTraceable(Exception):
    """This pair's block structure cannot ride the traced carry.

    Raised at **trace time** -- from :func:`_canonical_symmetric_layout` when a
    bond's two ends disagree about their own layout, or from
    :func:`_bp_solve`'s fixed-point probe when one sweep does not return to
    the canonical metadata (a sweep that structurally empties or grows a
    sector would).  Both drivers represent the same physics, so the caller's
    remedy is the eager loop, and :func:`bp_gauge_checkerboard` and
    ``ipeps_gauge.gauge_fix`` fall back to it on this exception.  Note the
    cost profile of that fallback: the failed trace is *not* cached, so a
    caller looping over such a pair pays a fresh trace attempt per solve --
    correct, loud in the report, and slow, in that order of importance.
    """


def _canonical_symmetric_layout(
    gam: dict[str, Tensor], weights: BondWeights
) -> tuple[dict[str, Tensor], BondWeights]:
    """Relayout a ``SymmetricTensor`` pair into the metadata the sweep emits.

    The traced driver's carry holds bare buffers with the index metadata
    closed over, so the loop needs input whose metadata is already the sweep's
    fixed point: each virtual leg charge-grouped ascending, flows stamped
    ``IN`` on the left/upper end of every bond and ``OUT`` on the right/lower
    one (the convention :func:`_gauge_bond` produces).  Measured, one sweep
    maps any accepted layout onto exactly that and a second sweep leaves it
    there.

    Every step is a *relabel*, not a transformation, so the represented state
    is bit-for-bit the input:

    * a leg whose flow already matches keeps its charges; one whose flow must
      flip takes their duals -- charge ``q`` at one flow and ``dual(q)`` at
      the other are the same conservation constraint, which is also why the
      sweep itself emits the dual multiset for an opposite-convention caller
      (measured: ``[-2, 1, 2]``/OUT in, ``[-2, -1, 2]``/IN out);
    * the charge list is then stably sorted ascending, which moves **no block
      data** -- a block's rows are its charge's slots in order of appearance,
      and a stable sort preserves that order -- only the list itself and the
      bond's weight vector, which is permuted identically;
    * both ends of a bond carry the same list, so one permutation serves the
      leg on each site and the ``lambda`` between them, and the pairing
      :func:`scale_bond_axis` does by position is preserved exactly.

    A slot whose charge holds no block on **either** end is dropped along the
    way, with its weight entry.  A ``D >= 3`` simple-update evolution
    produces exactly that pair -- a leg counting three charges with one dead
    (#906) -- and the sweep's SVD, which only sees occupied sectors, would
    shrink the bond on its first pass; a static carry cannot follow a
    shrink, but it does not have to, because dropping a chargeless slot is
    as much a relabel as the rest: no block references it, so no data moves,
    and the eager sweep's math annihilates the slot on contact anyway.

    Raises:
        _StructureNotTraceable: if a bond's two ends do not carry the same
            charge list with opposite flows, if their ends disagree about
            which charges are occupied (a one-sided zombie the drop rule
            cannot relabel away), or if a bond has no occupied charge at all
            -- input the positional weight convention cannot describe, so no
            layout fixes it and the eager driver owns the case.
    """
    plans: dict[str, dict[str, tuple[TensorIndex, bool]]] = {"A": {}, "B": {}}
    new_w: dict[str, jax.Array] = {}
    for bond, (site_L, leg_L), (site_R, leg_R) in _BONDS:
        ax_L = gam[site_L].labels().index(leg_L)
        ax_R = gam[site_R].labels().index(leg_R)
        idx_L = gam[site_L].indices[ax_L]
        idx_R = gam[site_R].indices[ax_R]
        if not np.array_equal(idx_L.charges, idx_R.charges) or (
            idx_L.flow == idx_R.flow
        ):
            raise _StructureNotTraceable(
                f"bond {bond}: its ends carry charges "
                f"{idx_L.charges.tolist()}/{idx_L.flow.name} and "
                f"{idx_R.charges.tolist()}/{idx_R.flow.name}; the positional "
                f"weight convention needs one list with opposite flows"
            )
        occ_L = {key[ax_L] for key in gam[site_L].blocks}
        occ_R = {key[ax_R] for key in gam[site_R].blocks}
        if occ_L != occ_R:
            raise _StructureNotTraceable(
                f"bond {bond}: its ends disagree about occupied charges "
                f"({sorted(occ_L)} vs {sorted(occ_R)})"
            )
        alive = np.array([c in occ_L for c in idx_L.charges.tolist()])
        if not alive.any():
            raise _StructureNotTraceable(f"bond {bond} has no occupied charge")
        kept = np.where(alive)[0]
        sym = idx_L.symmetry
        flip = idx_L.flow != FlowDirection.IN
        canon = np.asarray(sym.dual(idx_L.charges) if flip else idx_L.charges)[kept]
        perm = kept[np.argsort(canon, kind="stable")]
        sorted_charges = np.sort(canon, kind="stable")
        plans[site_L][leg_L] = (
            TensorIndex.from_charges(
                sym, sorted_charges, FlowDirection.IN, label=leg_L
            ),
            flip,
        )
        plans[site_R][leg_R] = (
            TensorIndex.from_charges(
                sym, sorted_charges, FlowDirection.OUT, label=leg_R
            ),
            flip,
        )
        new_w[bond] = getattr(weights, bond)[perm]

    out = {}
    for site, t in gam.items():
        labels = t.labels()
        new_indices = tuple(
            plans[site][lab][0] if lab in plans[site] else t.indices[ax]
            for ax, lab in enumerate(labels)
        )
        flip_axes = [
            ax
            for ax, lab in enumerate(labels)
            if plans[site].get(lab, (None, False))[1]
        ]
        if flip_axes:
            sym = t.indices[0].symmetry
            blocks = {}
            for key, blk in t.blocks.items():
                k = list(key)
                for ax in flip_axes:
                    k[ax] = int(np.asarray(sym.dual(np.array([key[ax]])))[0])
                blocks[tuple(k)] = blk
        else:
            blocks = dict(t.blocks)
        out[site] = SymmetricTensor._from_blocks_unchecked(blocks, new_indices)
    return out, BondWeights(**new_w)


def _embed_zero_blocks(t: SymmetricTensor, target: SymmetricTensor) -> SymmetricTensor:
    """Rebuild ``t`` on ``target``'s block set, zero-filling what it lacks.

    A sweep can *create* blocks its input did not carry structurally -- the
    message and gauge contractions populate every conservation-allowed
    product -- so an input's block set may be a strict subset of the sweep's
    fixed point.  Adding an explicit zero block is value-identical to leaving
    it out, so this is a relabel like the rest of the canonicalization: the
    represented state does not move.

    Raises:
        _StructureNotTraceable: if ``t`` carries a block ``target`` lacks, or
            their indices differ -- then ``target`` is not a structural
            superset and the embedding would drop data.
    """
    if t.indices != target.indices:
        raise _StructureNotTraceable(
            "the sweep moved a leg's index metadata rather than only growing "
            "the block set; the traced carry cannot follow"
        )
    blocks = dict(t.blocks)
    missing = [k for k in blocks if k not in set(target._block_keys)]
    if missing:
        raise _StructureNotTraceable(
            f"the sweep structurally dropped blocks {missing}; embedding "
            f"into its layout would lose them"
        )
    out = {
        key: blocks.get(key, jnp.zeros(shape, t.dtype))
        for key, shape in zip(target._block_keys, target._block_shapes)
    }
    return SymmetricTensor._from_blocks_unchecked(out, t.indices)


def _use_traced_loop(A: Tensor, B: Tensor) -> bool:
    """Does this pair take the traced driver?

    ``DenseTensor`` and ``SymmetricTensor`` pairs both do, since #939 and the
    sector-mode SVD removed the two symmetric blockers the module docstring
    records.  The dispatch is a named function rather than an inline
    ``isinstance`` so a test can pin the two drivers against each other on the
    *same* input, which is the only way to check that tracing did not move the
    answer -- and so the drivers' callers can force the eager reference.  A
    symmetric pair can still *end up* on the eager loop at runtime: the traced
    driver rejects, via :class:`_StructureNotTraceable`, any pair whose
    structure cannot hold its fixed carry, and the callers fall back.
    """
    if isinstance(A, DenseTensor) and isinstance(B, DenseTensor):
        return True
    return isinstance(A, SymmetricTensor) and isinstance(B, SymmetricTensor)


def _validate_weights(weights: BondWeights) -> None:
    """Reject a weight vector that is not a state, before anything else runs.

    Eager and outside the traced region on purpose: it is the documented
    ``ValueError`` and a traced solve cannot raise one.

    **One** host sync on the happy path, not eight.  The per-bond loop it
    replaced synced twice per bond, and at ~2.7 ms per warm solve that is not
    noise -- the whole traced solve is 0.8 ms.  The failing path re-reads the
    offending bond, which costs a second pass only when it is about to raise.
    """
    ok = jnp.stack([jnp.all(jnp.isfinite(w)) & (jnp.max(w) > 0.0) for w in weights])
    if bool(jnp.all(ok)):
        return
    bond = BondWeights._fields[int(jnp.argmin(ok))]
    raise ValueError(
        f"weights.{bond} is not a usable bond weight (max "
        f"{float(jnp.max(getattr(weights, bond))):.3g}).  Every weight vector "
        f"must be finite with at least one positive entry: it is half the "
        f"state being gauged, and the convergence test divides by its norm."
    )


def _prepare(
    A: Tensor, B: Tensor, weights: BondWeights
) -> tuple[dict[str, Tensor], BondWeights]:
    """Put the pair and its weights at unit scale.  Shared by both drivers.

    Before the first message, not just between sweeps: ``_message`` squares
    ``Gamma`` *and* multiplies three incoming weights into it, so a caller's
    overall scale of 1e-200 or 1e200 -- physically the same state -- would
    under/overflow the very first message and the solve would fail on a state
    it handles fine at unit scale.

    Normalising ``lambda`` is free, and not merely harmless: ``_gauge_bond`` is
    already exactly scale-invariant in ``lam`` -- scaling it scales the SVD's
    ``s`` by the same factor and ``lam_new = s / max(s)`` is unchanged -- so
    this cannot move the answer.  It also makes the first sweep's residual
    meaningful, which otherwise compares a max-1 output against whatever
    convention the caller's input used.

    The traced driver calls this *inside* its jit, so on the dense path it
    costs no eager dispatches at all.

    **The weights come back at the pair's own real precision.**  Weights are
    singular values, so the target is the real dtype behind a complex pair --
    ``float32`` for ``complex64``, ``float64`` for ``complex128``.  Without the
    cast the carry is heterogeneous: tenax enables x64 globally, so
    ``BondWeights.ones`` and :func:`~tenax.algorithms.ipeps_gauge.gauge_fix`'s
    identity weights are ``float64``, and against a ``float32`` pair the first
    sweep promotes the candidate tensors to ``float64`` while the carry's tensor
    slot is still ``float32``.  ``lax.while_loop`` then rejects the body outright
    with *"carry input and carry output must have equal types"*, so **every**
    dense ``float32`` pair failed -- and it failed only on the traced driver,
    which the eager loop's Python rebinding tolerated, making it a regression
    introduced with the tracing rather than a standing limitation.

    Casting here rather than at the loop boundary keeps both drivers on one
    convention, so the eager reference cannot silently accept a mix the traced
    driver rejects.

    **The two sites are brought to one dtype as well**, not just the weights:
    ``A`` at ``float32`` beside ``B`` at ``float64`` promotes ``A``'s candidate
    inside the sweep while its carry slot stays ``float32``, which is the same
    ``while_loop`` carry rejection one level in.  Weights alone are not enough.

    The dtypes are read from ``Tensor.dtype`` and the cast is a scalar multiply,
    so **nothing is densified**: ``todense()`` here would materialise a full
    ``D**4 * d`` array per site purely to inspect a dtype, which on the
    ``SymmetricTensor`` path defeats block sparsity before the solve even
    starts.  The multiply is exact -- the common dtype is by construction the
    *promoted* one, so this only ever widens, never rounds -- and it preserves
    the tensor class.
    """
    gam = {"A": _rescale(A), "B": _rescale(B)}
    dtype = jnp.result_type(*(t.dtype for t in gam.values()))
    if any(t.dtype != dtype for t in gam.values()):
        one = jnp.array(1, dtype)
        gam = {s: (t if t.dtype == dtype else t * one) for s, t in gam.items()}
    real = jnp.finfo(dtype).dtype if jnp.issubdtype(dtype, jnp.inexact) else dtype
    return (
        gam,
        BondWeights(*((w / jnp.max(w)).astype(real) for w in weights)),
    )


def _bp_solve_eager(
    gam: dict[str, Tensor],
    weights: BondWeights,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, Tensor], BondWeights, BPGaugeInfo]:
    """The Python-loop driver: the reference the traced driver is checked
    against, and the fallback for a pair whose block structure the traced
    carry cannot hold (:class:`_StructureNotTraceable`).

    Two host syncs per sweep (the health gate and the residual) and ~300 eager
    dispatches, which is what makes it 18.9 ms/sweep -- 555x the traced path's
    0.034 ms -- and why a dense pair does not come here.
    """
    residual = float("inf")
    done = 0

    for sweep in range(max_iter):
        cand_gam, cand_weights = _sweep(gam, weights)

        healthy = _sweep_is_healthy(cand_gam, cand_weights, sweep)
        if sweep >= 1:
            # From the second sweep on, ``weights`` is the solve's own iterate;
            # at sweep 0 it is the caller's stored numbers, which are not a
            # trajectory -- see :func:`_a_weight_underflowed`.
            healthy = healthy & ~_a_weight_underflowed(cand_weights, weights)
        if not bool(healthy):
            # Reject the candidate; do not call it converged.  ``_sweep`` does
            # not mutate its input, so ``gam``/``weights`` still hold the last
            # healthy iterate -- which is an exact gauge of the caller's state,
            # merely unconverged.  No saved copy is involved, so the two cannot
            # drift apart.
            residual = float("inf")
            break

        gam, done = cand_gam, sweep + 1
        residual = float(_residual(cand_weights, weights))
        weights = cand_weights
        if residual < tol:
            return gam, weights, BPGaugeInfo(done, residual, True)

    # ``done``, not ``max_iter``: a health rollback stops early, and reporting
    # the cap would claim sweeps that never ran.
    return gam, weights, BPGaugeInfo(done, residual, False)


def _bp_solve(
    A: Tensor,
    B: Tensor,
    weights: BondWeights,
    max_iter: int,
    tol: float,
):
    """The whole solve as one ``lax.while_loop``.

    Traceable, and deliberately **not** jitted itself -- :data:`_bp_solve_traced`
    is the jitted entry point, and a caller that wants a *wider* boundary calls
    this instead so the solve is inlined into its own trace.
    ``ipeps_gauge.gauge_fix`` does exactly that: nesting ``jit`` inside ``jit``
    would work, but it would leave ``_bp_solve_traced``'s executable cache empty
    while the real key lives on the outer entry, which is the counter the
    one-compile gate reads.  One function, two boundaries, no second copy of the
    driver.

    Compiled once per ``(D, dtype, carry treedef, max_iter, tol)`` and reused,
    which is the property the cost of re-gauging every simple-update step rests
    on: compile is the binding term, not steady state.  ``max_iter`` and ``tol``
    are static so the loop's exit conditions are literals; varying either costs
    a recompile, so a caller sweeping tolerances should expect one compile per
    distinct value.

    Carry, six slots::

        gam        {"A", "B"} site *leaves*     (dict of list[jax.Array])
        weights    the four bond weights        (BondWeights)
        residual   last healthy sweep's residual, or inf
        done       completed *healthy* sweeps
        converged  reached ``tol`` on a healthy sweep
        dead       the last sweep left f64

    **Leaves, not ``Tensor``s, and the tree metadata is a closure constant.**
    ``TensorIndex`` (and, for ``SymmetricTensor``, the block table) is pytree
    *aux* data, and a swept tensor is not metadata-identical to its input:
    ``contract`` permutes the legs and ``_gauge_bond`` stamps its own flows
    (see :func:`_restore_caller_structure`).  With ``Tensor``s in the carry
    that changes the treedef between iterations and ``while_loop`` fails with
    ``Mismatch custom node data`` -- which is precisely what a
    simple-update-evolved pair triggers, since its virtual flows are the
    opposite of this module's.  Carrying bare leaves removes the aux data
    entirely (a ``DenseTensor`` is one array; a ``SymmetricTensor`` is one
    flat block buffer), so the carry is stable by construction rather than by
    a coincidence of conventions, and its shapes are visibly fixed -- the
    property the ``_PINV_CUTOFF`` mask exists to preserve.

    A dense pair's metadata is already loop-stable because
    :func:`_restore_caller_structure` rebuilds its indices verbatim each
    sweep.  A symmetric pair's has to be *made* stable, in two steps at trace
    time: :func:`_canonical_symmetric_layout` relabels the caller's pair into
    the layout the sweep emits, and the fixed-point probe above the loop
    closes the block set with :func:`_embed_zero_blocks`.  Both are relabels
    -- the state handed to sweep 0 is exactly the caller's, so the health
    gate's semantics, including rejecting an unhealthy *first* sweep back to
    the caller's own state, match the eager driver's.  A pair whose structure
    defeats this raises :class:`_StructureNotTraceable` at trace time and the
    entry points fall back to the eager loop.

    ``converged`` and ``dead`` are separate slots rather than one ``stop`` flag
    because the two exits carry different payloads: the tolerance exit reports
    the residual it reached, the health rollback reports ``inf`` and
    ``converged=False``.

    No ``last_good`` slot is needed.  At every loop boundary the last good
    iterate *is* the carry, so the rollback is not a restore but a **rejection
    of the candidate**: ``carry_out = where(healthy, candidate, carry_in)``.
    That halves the carry and removes the failure mode where two copies of
    ``gam`` drift apart.

    .. warning::
        Not reverse-mode differentiable.  ``lax.while_loop`` has no reverse
        rule, and the ``where``-select would leak NaN under ``grad`` even where
        the primal is clean.  A differentiable gauge would need ``scan`` with a
        fixed trip count and a sticky ``frozen`` flag, which pays for every
        unused sweep; nothing needs it today.
    """
    gam, weights = _prepare(A, B, weights)
    if isinstance(gam["A"], SymmetricTensor):
        # The carry needs input whose metadata already is the sweep's fixed
        # point: relabel into it (charge-grouped legs, this module's flows,
        # weights permuted along) and then let the probe below close the
        # block set.  Every step is a relabel, so the state does not move.
        gam, weights = _canonical_symmetric_layout(gam, weights)
    # Traced once per compile, then constant for every call that reuses it.
    like = dict(gam)

    if isinstance(like["A"], SymmetricTensor):
        # Fixed-point probe, trace time only.  One sweep from the canonical
        # layout must land back on it for the carry to be stable; the one
        # legitimate way it cannot is by *growing* the block set (message and
        # gauge contractions populate every conservation-allowed product), so
        # embed zero blocks and try again until the structure closes.  The
        # probes' arrays are never used, so XLA's DCE removes the runtime
        # cost; anything the embedding cannot absorb raises
        # :class:`_StructureNotTraceable` and the caller falls back to the
        # eager driver.  Two rounds close every structure seen in practice;
        # four bounds the trace-time cost before declaring the pair untraceable.
        for _ in range(4):
            probe, _probe_w = _sweep(dict(like), weights)
            probe = {s: _restore_caller_structure(t, like[s]) for s, t in probe.items()}
            if all(
                jax.tree_util.tree_flatten(probe[s])[1]
                == jax.tree_util.tree_flatten(like[s])[1]
                for s in like
            ):
                break
            like = {s: _embed_zero_blocks(like[s], probe[s]) for s in like}
        else:
            raise _StructureNotTraceable(
                "the sweep's block structure did not close after 4 rounds of "
                "zero-block embedding"
            )
        gam = like

    treedef = {s: jax.tree_util.tree_flatten(t)[1] for s, t in like.items()}

    def as_tensors(leaves: dict[str, list[jax.Array]]) -> dict[str, Tensor]:
        return {
            s: jax.tree_util.tree_unflatten(treedef[s], ls) for s, ls in leaves.items()
        }

    def as_leaves(tensors: dict[str, Tensor]) -> dict[str, list[jax.Array]]:
        out = {}
        for s, t in tensors.items():
            ls, td = jax.tree_util.tree_flatten(_restore_caller_structure(t, like[s]))
            if td != treedef[s]:
                raise _StructureNotTraceable(
                    f"sweep output for site {s} left the carry's structure"
                )
            out[s] = ls
        return out

    init = (
        {s: jax.tree_util.tree_flatten(t)[0] for s, t in gam.items()},
        weights,
        # ``inf`` in the residual slot has to carry exactly the dtype the body
        # will write there or the carry is inconsistent, so take it from the
        # residual itself rather than guessing.  The dead arithmetic is DCE'd.
        jnp.full_like(_residual(weights, weights), jnp.inf),
        jnp.zeros((), jnp.int32),
        jnp.zeros((), bool),
        jnp.zeros((), bool),
    )

    def cond(carry):
        _, _, _, done, converged, dead = carry
        return ~(converged | dead) & (done < max_iter)

    def body(carry):
        arr_in, w_in, _, done, _, _ = carry
        cand_gam, cand_weights = _sweep(as_tensors(arr_in), w_in)
        # ``done >= 1`` for the same reason the eager driver gates on
        # ``sweep >= 1``: at ``done == 0`` the carry still holds the caller's
        # stored weights, which are not a trajectory.
        ok = _sweep_is_healthy(cand_gam, cand_weights, done) & ~(
            _a_weight_underflowed(cand_weights, w_in) & (done >= 1)
        )
        res = _residual(cand_weights, w_in)
        accept = lambda cand, prev: jnp.where(ok, cand, prev)  # noqa: E731
        return (
            jax.tree_util.tree_map(accept, as_leaves(cand_gam), arr_in),
            jax.tree_util.tree_map(accept, cand_weights, w_in),
            jnp.where(ok, res, jnp.inf),
            # After the health gate, never before: ``done`` counts completed
            # *healthy* sweeps.
            done + ok.astype(jnp.int32),
            ok & (res < tol),
            ~ok,
        )

    arr, weights, residual, done, converged, dead = lax.while_loop(cond, body, init)
    return as_tensors(arr), weights, residual, done, converged, dead


#: :func:`_bp_solve` as a compiled entry point -- the boundary
#: :func:`bp_gauge_checkerboard` uses.  ``max_iter`` and ``tol`` are static, so
#: the loop's exit conditions are literals and a caller sweeping tolerances pays
#: one compile per distinct value.
_bp_solve_traced = jax.jit(_bp_solve, static_argnums=(3, 4))


def bp_gauge_checkerboard(
    A: Tensor,
    B: Tensor,
    weights: BondWeights,
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[Tensor, Tensor, BondWeights, BPGaugeInfo]:
    """Re-derive the four bond weights from the current tensors, and re-gauge.

    The returned state is the *same physical state* --- every step is a gauge
    transformation, exact to machine precision --- but its bond weights are now
    the self-consistent BP messages rather than whatever the last SVD left
    behind.

    Dense **and** symmetric pairs take the traced driver.  Only the input
    validation stays on the host, because it raises the documented
    ``ValueError`` and a traced solve cannot; the initial rescale and the
    weight normalisation run *inside* the jit (:func:`_prepare`), and three
    casts rebuild :class:`BPGaugeInfo` on the way out.  So **four** host syncs
    for the whole solve -- one in :func:`_validate_weights` plus those three
    casts -- down from 18 per sweep.  A ``SymmetricTensor`` pair whose block
    structure cannot hold the traced carry falls back to the eager Python
    loop at trace time (see :class:`_StructureNotTraceable`); the answer is
    the same either way, since both drivers share the sweep body verbatim.

    .. warning::
        The dense path is **not reverse-mode differentiable** -- see
        :func:`_bp_solve`.  Nothing in ``src`` differentiates through
        this today.

    ``weights`` is **required**, and is not an initial guess: in Vidal form the
    state is ``... Gamma_A lambda Gamma_B ...``, so the incoming ``lambda`` is
    half of what the caller is handing over, and it is what each bond's
    re-gauging SVD factors.  Defaulting it to one would silently re-gauge
    ``Gamma_A I Gamma_B`` --- a *different* state --- while still reporting the
    invariance guarantee above, so a simple-update pair must pass its own
    ``lambda`` and a fresh random pair must pass :meth:`BondWeights.ones`
    explicitly.

    Args:
        A:        Bare Vidal ``Gamma`` for sublattice A, labels ``(u,d,l,r,phys)``.
        B:        Bare Vidal ``Gamma`` for sublattice B, same labels.
        weights:  The bond weights ``A`` and ``B`` currently carry.
        max_iter: Maximum BP sweeps.
        tol:      Stop once the largest relative change in any weight vector
                  falls below this.

    Returns:
        ``(A, B, weights, info)``.  If a sweep walks the iterate out of f64,
        the last healthy sweep is returned instead --- still an exact gauge of
        the input, just not converged --- with ``info.converged`` false and
        ``info.residual`` infinite.  It is never the overflow artefact.

    Raises:
        ValueError: if any weight vector is non-finite or has no positive
            entry.  That is not a state, and the residual divides by its norm.

    Example:
        >>> w = BondWeights(lam_h, lam_h, lam_v, lam_v)     # doctest: +SKIP
        >>> A, B, w, info = bp_gauge_checkerboard(A, B, w)  # doctest: +SKIP
        >>> info.converged                                  # doctest: +SKIP
        True
    """
    _validate_weights(weights)

    traced = _use_traced_loop(A, B)
    if traced:
        try:
            gam, weights_out, residual, done, converged, _ = _bp_solve_traced(
                A, B, weights, max_iter, tol
            )
        except _StructureNotTraceable:
            # Raised at trace time: this pair's block structure cannot hold
            # the traced carry.  The eager loop represents the same physics,
            # so fall back rather than fail -- correct and slow, and the
            # failed trace repeats per call (see the exception's docstring).
            traced = False
        else:
            weights = weights_out
            # The only host syncs in the whole traced solve: three, to
            # rebuild ``BPGaugeInfo`` at its documented ``(int, float,
            # bool)`` type.  A 0-d array would satisfy ``assert
            # info.converged`` *silently*, and would fail ``info.residual ==
            # float("inf")`` loudly.
            info = BPGaugeInfo(int(done), float(residual), bool(converged))
    if not traced:
        gam, weights, info = _bp_solve_eager(*_prepare(A, B, weights), max_iter, tol)

    # The caller's structure is handed back.  On the traced path this is
    # already a no-op -- that driver restores it every sweep, because it has to
    # -- and it is kept because it is what makes the two drivers agree on what
    # they return, and what documents the contract.
    return (
        _restore_caller_structure(gam["A"], A),
        _restore_caller_structure(gam["B"], B),
        weights,
        info,
    )
