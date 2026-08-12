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

The four bonds touch a different leg of each tensor, so their gauge
transformations commute and are applied together.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_moves import _flow_flip_no_conj
from tenax.contraction.contractor import contract
from tenax.core._tensor_utils import scale_bond_axis
from tenax.core.tensor import Tensor
from tenax.linalg import eigh, svd

__all__ = ["BondWeights", "BPGaugeInfo", "bp_gauge_checkerboard"]

# Relative cutoff for the 1/sqrt(w) pseudo-inverse of a message.  A message
# eigenvalue at zero is a bond direction the state does not use; inverting it
# would turn 0 into inf rather than project it out.
_PINV_CUTOFF = 1e-12

_BRA = "__bra"
_K = "__k"
_K2 = "__k2"
_S = "__s"
_B = "__b"


class BondWeights(NamedTuple):
    """The Schmidt spectra of the **four** bonds of a checkerboard unit cell.

    A two-site checkerboard has two inequivalent sites and therefore four
    inequivalent nearest-neighbour bonds, not two::

        h_AB : A.r <-> B.l          v_AB : A.d <-> B.u
        h_BA : B.r <-> A.l          v_BA : B.d <-> A.u

    On a translation-invariant Hamiltonian the AB and BA bonds are related by
    the translation that swaps A and B, so they coincide at the physical fixed
    point --- but not away from it, and BP resolves them separately (#851).
    """

    h_AB: jax.Array
    h_BA: jax.Array
    v_AB: jax.Array
    v_BA: jax.Array

    @classmethod
    def ones(cls, D_h: int, D_v: int) -> BondWeights:
        """Unweighted bonds, the natural starting point for a BP solve."""
        return cls(
            h_AB=jnp.ones(D_h),
            h_BA=jnp.ones(D_h),
            v_AB=jnp.ones(D_v),
            v_BA=jnp.ones(D_v),
        )


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


def _leg_dim(t: Tensor, leg: str) -> int:
    return t.indices[t.labels().index(leg)].dim


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


def bp_gauge_checkerboard(
    A: Tensor,
    B: Tensor,
    weights: BondWeights | None = None,
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[Tensor, Tensor, BondWeights, BPGaugeInfo]:
    """Re-derive the four bond weights from the current tensors, and re-gauge.

    The returned state is the *same physical state* --- every step is a gauge
    transformation, exact to machine precision --- but its bond weights are now
    the self-consistent BP messages rather than whatever the last SVD left
    behind.

    Args:
        A:        Bare Vidal ``Gamma`` for sublattice A, labels ``(u,d,l,r,phys)``.
        B:        Bare Vidal ``Gamma`` for sublattice B, same labels.
        weights:  Starting bond weights.  ``None`` starts from unweighted bonds.
        max_iter: Maximum BP sweeps.
        tol:      Stop once the largest relative change in any weight vector
                  falls below this.

    Returns:
        ``(A, B, weights, info)``.

    Example:
        >>> A, B, w, info = bp_gauge_checkerboard(A, B)   # doctest: +SKIP
        >>> info.converged                                # doctest: +SKIP
        True
    """
    if weights is None:
        weights = BondWeights.ones(_leg_dim(A, "r"), _leg_dim(A, "d"))

    order = {"A": A.labels(), "B": B.labels()}
    gam = {"A": A, "B": B}
    residual = float("inf")
    for sweep in range(max_iter):
        msg = {
            (site, leg): _message(gam[site], site, leg, weights)
            for site in ("A", "B")
            for leg in _LEGS
        }

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

        for site in ("A", "B"):
            n = gam[site].norm()
            gam[site] = gam[site] * (1.0 / jnp.where(n > 0, n, 1.0))

        residual = max(
            float(
                jnp.linalg.norm(new_weights[b] - getattr(weights, b))
                / jnp.maximum(jnp.linalg.norm(getattr(weights, b)), 1e-300)
            )
            for b in new_weights
        )
        weights = BondWeights(**new_weights)
        if residual < tol:
            return (
                _reorder(gam["A"], order["A"]),
                _reorder(gam["B"], order["B"]),
                weights,
                BPGaugeInfo(sweep + 1, residual, True),
            )

    return (
        _reorder(gam["A"], order["A"]),
        _reorder(gam["B"], order["B"]),
        weights,
        BPGaugeInfo(max_iter, residual, False),
    )
