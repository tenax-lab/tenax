"""Simple update with no stored bond spectrum (#882 Phase 2).

The defect class this module replaces is one defect wearing four issue
numbers.  Simple update stores each bond's Schmidt spectrum straight from the
SVD that produced it; an imaginary-time gate is **non-unitary**, so a gate on
any *other* bond invalidates that spectrum, and every subsequent step then
divides out and re-absorbs a number that is no longer the state's.  #667 (the
bond carried ``lambda**1.5``), #851 (two stored spectra for four inequivalent
bonds, so ``steps % 4`` chose the answer), #865 (a pinned per-sector layout
made the SVD discard the *largest* singular value) and #869 (a non-monotonic
D=3 energy) are all instances of it.

The rewrite's premise is that the storage goes away rather than the bugs being
fixed one at a time: :class:`_SUState` holds two site tensors in **absorbed
form** and has nowhere to put a spectrum, and each step re-derives the gauge
from the tensors themselves with
:func:`~tenax.algorithms.ipeps_gauge.gauge_fix`.  Vidal form exists only
transiently, inside one step, between the SVD and the ``sqrt(sigma)`` split
that immediately consumes it.

Everything here is underscore-private for v1.  Nothing is exported from
``tenax/__init__.py`` and nothing is documented in ``README.md``: the public
entry point arrives in #882's Phase 4, once ``ipeps()`` is wired to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenax.core.tensor import Tensor

#: Canonical leg order of a checkerboard site tensor, shared with
#: ``ipeps._wrap_as_dense_tensor`` and ``ipeps.heisenberg_u1sz_init_pair``.
_SITE_LABELS: tuple[str, ...] = ("u", "d", "l", "r", "phys")

#: The four virtual legs, in the same order.
_VIRTUAL_LEGS: tuple[str, ...] = ("u", "d", "l", "r")


@dataclass(frozen=True)
class _SUState:
    """An iPEPS checkerboard pair in absorbed form.

    Absorbed form means each bond's weight is already split ``sqrt(lambda)``
    into both of its ends, so ``A`` and ``B`` together **are** the
    wavefunction -- not one factor of it.  That is the same convention
    :func:`~tenax.algorithms.ipeps_gauge.gauge_fix` takes and returns, which is
    what lets a step hand its output straight to the next step's gauge.

    There is deliberately nowhere to store a bond spectrum.  An imaginary-time
    gate is non-unitary, so a spectrum cached on one bond is invalidated by a
    gate on any other, and every defect this module replaces was a defect in
    exactly that storage (#667, #851, #865, #869).  This is a *new* type rather
    than the existing iPEPS container with its lambda fields removed (#882
    §9.2): a type that used to have them invites them back the first time
    something wants to cache a spectrum, and the old container keeps its
    lambdas legitimately for the real-time path, where a unitary gate leaves
    the other bonds' spectra alone.

    ``test_su_state_has_no_lambda_fields`` is that premise as an executable
    assertion.  Do not add a field to satisfy a caller; re-derive the spectrum
    with ``gauge_fix``, which returns it as a diagnostic.
    """

    A: Tensor
    B: Tensor

    @classmethod
    def from_pair(cls, A: Tensor, B: Tensor) -> _SUState:
        """Build a state from an **absorbed-form** pair, checking its labels.

        The label check is the only validation available: absorbed form is a
        claim about how the bond weights are distributed between two tensors,
        and no property of the two tensors alone can distinguish it from Vidal
        form.  What this does catch is a pair with a leg named ``left`` or
        ``phys_B``, or a 4-leg tensor, i.e. the mistakes that would otherwise
        surface several frames deep inside a ``contract``.

        Args:
            A, B: Absorbed-form site tensors, labels ``(u, d, l, r, phys)`` in
                  any order.

        Returns:
            The state.

        Raises:
            ValueError: if either tensor's label set is not
                ``{u, d, l, r, phys}``.
        """
        for name, t in (("A", A), ("B", B)):
            labels = set(t.labels())
            if labels != set(_SITE_LABELS):
                raise ValueError(
                    f"{name} must be a checkerboard site tensor with labels "
                    f"{set(_SITE_LABELS)}; got {sorted(labels)}.  This type "
                    f"holds the pair in absorbed form -- the site tensors "
                    f"alone, with every bond weight already split into both "
                    f"of its ends."
                )
        return cls(A=A, B=B)

    @property
    def max_D(self) -> int:
        """The largest virtual bond dimension in the pair.

        Read off both tensors and all four virtual legs rather than off ``A.r``
        alone, because the four bonds are **not** constrained to share a
        dimension: a truncation to ``max_D`` on one bond leaves the other three
        as they were, and on the block-sparse path a bond's per-sector layout
        moves under truncation too.  ``phys`` is excluded -- it is not a bond.
        """
        return max(
            t.indices[t.labels().index(leg)].dim
            for t in (self.A, self.B)
            for leg in _VIRTUAL_LEGS
        )
