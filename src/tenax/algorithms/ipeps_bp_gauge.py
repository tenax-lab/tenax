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
``float()``/``bool()`` syncs are a *minority* contributor: there are exactly 18
per sweep, and removing all of them without a traced loop is worth perhaps
1.2-1.4x, not 200x.  So the fix is to trace the whole solve, which removes all
three at once.

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
before a single solve runs; see ``tests/test_ipeps_gauge_perf.py``, which
records the shortfall rather than hiding it.

``SymmetricTensor`` stays on the eager loop, because it cannot be traced today:
``_eigh_symmetric`` derives the output bond's charges from eigenvalue
*magnitudes* via ``np.array(...)``, and rerouting through the traceable
symmetric SVD instead hits ``_zero_subrank_singular_values``, whose relative
floor snaps small bond weights to exactly zero and broke the gauge by 4.9e-01
on 2 of 8 fixtures.  Both live outside this module.  The sweep body is shared
verbatim between the two loops, so there is one implementation of the physics
and only the driver differs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
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
from tenax.core.tensor import DenseTensor, Tensor
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
    V, w = eigh(m, left_labels=[out_leg], right_labels=[_BRA], new_bond_label=_K)
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
    U, s, Vh, _ = svd(M, left_labels=[_K], right_labels=[_K2], new_bond_label=_S)
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
    return jnp.max(
        jnp.stack(
            [
                jnp.linalg.norm(n - o)
                / jnp.maximum(jnp.linalg.norm(o), _RESIDUAL_FLOOR)
                for n, o in zip(new, old, strict=True)
            ]
        )
    )


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


def _use_traced_loop(A: Tensor, B: Tensor) -> bool:
    """Does this pair take the traced driver?

    ``DenseTensor`` yes, ``SymmetricTensor`` no -- see the module docstring for
    the two blockers on the symmetric path, both of which live in files this
    module does not own.  The dispatch is a named function rather than an
    inline ``isinstance`` so a test can pin the two drivers against each other
    on the *same* dense input, which is the only way to check that tracing did
    not move the answer.
    """
    return isinstance(A, DenseTensor) and isinstance(B, DenseTensor)


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
    """
    return (
        {"A": _rescale(A), "B": _rescale(B)},
        BondWeights(*(w / jnp.max(w) for w in weights)),
    )


def _bp_solve_eager(
    gam: dict[str, Tensor],
    weights: BondWeights,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, Tensor], BondWeights, BPGaugeInfo]:
    """The Python-loop driver.  Used for ``SymmetricTensor``, and as the
    reference the traced driver is checked against.

    Two host syncs per sweep (the health gate and the residual) and ~300 eager
    dispatches, which is what makes it 18.9 ms/sweep -- 555x the traced path's
    0.034 ms -- and why a dense pair does not come here.
    """
    residual = float("inf")
    done = 0

    for sweep in range(max_iter):
        cand_gam, cand_weights = _sweep(gam, weights)

        if not bool(_sweep_is_healthy(cand_gam, cand_weights, sweep)):
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
    """The whole solve as one ``lax.while_loop``.  Dense only.

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

        gam        {"A", "B"} site *arrays*     (dict of jax.Array)
        weights    the four bond weights        (BondWeights)
        residual   last healthy sweep's residual, or inf
        done       completed *healthy* sweeps
        converged  reached ``tol`` on a healthy sweep
        dead       the last sweep left f64

    **Arrays, not ``Tensor``s, and the index metadata is a closure constant.**
    ``TensorIndex`` is pytree *aux* data, and a swept tensor is not
    metadata-identical to its input: ``contract`` permutes the legs and
    ``_gauge_bond`` stamps its own flows (see
    :func:`_restore_caller_structure`).  With ``Tensor``s in the carry that
    changes the treedef between iterations and ``while_loop`` fails with
    ``Mismatch custom node data`` -- which is precisely what a
    simple-update-evolved pair triggers, since its virtual flows are the
    opposite of this module's.  Carrying bare arrays removes the aux data
    entirely, so the carry is stable by construction rather than by a
    coincidence of conventions, and its shapes are visibly fixed -- the property
    the ``_PINV_CUTOFF`` mask exists to preserve.

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
    # Traced once per compile, then constant for every call that reuses it.
    like = dict(gam)

    def as_tensors(arrays: dict[str, jax.Array]) -> dict[str, Tensor]:
        return {s: DenseTensor(a, like[s].indices) for s, a in arrays.items()}

    def as_arrays(tensors: dict[str, Tensor]) -> dict[str, jax.Array]:
        return {
            s: _restore_caller_structure(t, like[s]).todense()
            for s, t in tensors.items()
        }

    init = (
        {s: t.todense() for s, t in gam.items()},
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
        ok = _sweep_is_healthy(cand_gam, cand_weights, done)
        res = _residual(cand_weights, w_in)
        accept = lambda cand, prev: jnp.where(ok, cand, prev)  # noqa: E731
        return (
            jax.tree_util.tree_map(accept, as_arrays(cand_gam), arr_in),
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

    A **dense** pair takes the traced driver.  Only the input validation stays
    on the host, because it raises the documented ``ValueError`` and a traced
    solve cannot; the initial rescale and the weight normalisation run *inside*
    the jit (:func:`_prepare`), and three casts rebuild :class:`BPGaugeInfo` on
    the way out.  So **four** host syncs for the whole solve -- one in
    :func:`_validate_weights` plus those three casts -- down from 18 per sweep.
    A ``SymmetricTensor`` pair takes the eager Python loop, which is unchanged;
    see the module docstring for why it cannot be traced yet.  Both share the
    sweep body verbatim.

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

    if _use_traced_loop(A, B):
        gam, weights, residual, done, converged, _ = _bp_solve_traced(
            A, B, weights, max_iter, tol
        )
        # The only host syncs in the whole traced solve: three, to rebuild
        # ``BPGaugeInfo`` at its documented ``(int, float, bool)`` type.  A 0-d
        # array would satisfy ``assert info.converged`` *silently*, and would
        # fail ``info.residual == float("inf")`` loudly.
        info = BPGaugeInfo(int(done), float(residual), bool(converged))
    else:
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
