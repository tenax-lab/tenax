"""The frozen physical state the root-implicit engines are tested against (#772).

A D=2 Heisenberg simple-update ground state, frozen rather than regenerated so
the guard costs seconds instead of the ~2 minutes the imaginary-time evolution
takes, and so it cannot drift when the simple-update implementation changes.

Regenerate with ``scripts/gen_su_fixture.py``.  ``test_the_frozen_fixture_still
_matches_simple_update`` (slow) checks it against a live run.

This state matters because it *violates* the retained-spectrum precondition of
the covariant characteristic equations: at chi=4 its half-infinite environment
retains a direction at 3e-8 relative, and at chi>=6 it reaches the 1e-12 floor.
That is what NaN'd the gradient in #772.  The random ``_site_tensor`` fixture
never gets near it -- its smallest retained value at chi=8 is 4e-4.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# labels = ['u', 'd', 'l', 'r', 'phys']
PHYSICAL_SU_D2_E_SU = -0.5517095758652025
PHYSICAL_SU_D2_DATA = np.array(
    [
        [
            [
                [
                    [-0.6379969547839984, 0.14955752476730827],
                    [-0.022694759049087777, -0.09913501081523961],
                ],
                [
                    [0.20102630271814131, -0.018186111002694454],
                    [0.0066570650289896035, 0.02932241358528334],
                ],
            ],
            [
                [
                    [-0.02380904501065569, -0.10155708079629704],
                    [0.0009507624897592651, 0.0032988066061362437],
                ],
                [
                    [0.007003736711182391, 0.03008425530708316],
                    [-0.0002830675286400421, -0.0009837142663797524],
                ],
            ],
        ],
        [
            [
                [
                    [-0.3952301358851809, 0.13594046501958498],
                    [-0.014797849223720391, -0.0642761423016416],
                ],
                [
                    [0.5399822331591551, -0.12777193997695901],
                    [0.019228510065822237, 0.08398377508174674],
                ],
            ],
            [
                [
                    [-0.015494758674418345, -0.06577856685078609],
                    [0.0006136802412025929, 0.002126901318404443],
                ],
                [
                    [0.02017179180963998, 0.08603379993461818],
                    [-0.0008053772512215553, -0.0027943070596935637],
                ],
            ],
        ],
    ]
)
PHYSICAL_SU_D2_FLOWS = (-1, -1, -1, -1, -1)
PHYSICAL_SU_D2_LABELS = ("u", "d", "l", "r", "phys")


def physical_su_d2() -> DenseTensor:
    """The frozen state as a ``DenseTensor``, axes ``(u, d, l, r, phys)``.

    Charges are trivial (all zero) on every leg, so this is dense data wearing
    the symmetric-tensor interface the engine expects.

    Flows come from ``PHYSICAL_SU_D2_FLOWS`` rather than being written out
    here.  Simple update does **not** return the ``(OUT, IN, OUT, IN, IN)``
    layout ``ipeps._wrap_as_dense_tensor`` starts from -- the state it hands
    back is all-OUT -- so this fixture differs from
    ``test_ctm_root_implicit_asym._site_tensor`` in exactly that respect.  With
    trivial charges the distinction is numerically inert on this dense path,
    but #718 is a standing reminder that leg conventions are where this engine
    goes wrong, so the fixture records what the generator saw instead of
    asserting what it ought to be.
    """
    sym = U1Symmetry()
    data = np.asarray(PHYSICAL_SU_D2_DATA)
    idx = tuple(
        TensorIndex.from_charges(
            sym,
            np.zeros(data.shape[axis], dtype=np.int32),
            FlowDirection(flow),
            label=label,
        )
        for axis, (flow, label) in enumerate(
            zip(PHYSICAL_SU_D2_FLOWS, PHYSICAL_SU_D2_LABELS)
        )
    )
    # Already unit-norm -- gen_su_fixture.py normalises before emitting the
    # literal -- so no renormalisation here.
    return DenseTensor(jnp.asarray(data), idx)
