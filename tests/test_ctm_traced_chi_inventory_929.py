"""Under tracing the chi bond inherits the environment's inventory (#929).

#922 gave the *eager* CTM a chi cut that follows the singular values.  The AD
path cannot have that cut: ``jax.jit`` bakes the per-sector block shapes at
trace time, so which sector owns each chi slot has to be fixed before the SVD
runs.  Its fallback was the double-layer ``u2`` charge list tiled to ``chi`` — a
guess about the environment made from the *state* — and it is measurably wrong:

```
 D  chi   eager (#922)   traced (tiled guess)
 2    8   1.540e-04      9.924e-04
 3   16   4.441e-16      1.839e-07     (gap vs the same-state dense reference)
```

The environment's own chi leg is a far better guess, and it is static metadata
even under tracing.  At a fixed point it *is* the bond ``chi_new`` replaces, so
inheriting its multiset reproduces whatever inventory the environment carries.
Seeding the static rule with the eager inventory and re-converging reproduces
the eager environment **exactly** — ``|E_eager - E_seeded| = 0.0`` at both D=2
chi=8 and D=3 chi=16 — and does so faster than the eager cut, because a fixed
inventory does not churn block shapes between sweeps.

**This does not by itself fix a cold AD run.** A cold environment starts from
``initialize_ctm_tensor_env``, whose chi inventory is the same tiled guess, and
inheritance then perpetuates it. What changes is that a *good* inventory now
survives: before this, handing the AD path a converged environment did not
help, because the first traced sweep re-imposed the tiled guess and destroyed
it. Seeding ``env_init`` from an eager pre-pass is the remaining half, tracked
in #929.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import (
    _compute_2x2_projector_symmetric,
    _incoming_chi_charges,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _corner(chi_charges, chi_label_a, chi_label_b, fa, fb, fda, fdb, la, lb, seed, D=2):
    sym = U1Symmetry()
    d2 = np.arange(D**2, dtype=np.int32) % 2
    idx = (
        TensorIndex.from_charges(
            sym, np.asarray(chi_charges, np.int32), fa, label=chi_label_a
        ),
        TensorIndex.from_charges(sym, d2, fda, label=la),
        TensorIndex.from_charges(
            sym, np.asarray(chi_charges, np.int32), fb, label=chi_label_b
        ),
        TensorIndex.from_charges(sym, d2, fdb, label=lb),
    )
    return SymmetricTensor.random_normal(idx, jax.random.PRNGKey(seed))


IN, OUT = FlowDirection.IN, FlowDirection.OUT


def _corners(chi_charges):
    """Four enlarged corners whose chi legs carry ``chi_charges``."""
    return (
        _corner(chi_charges, "chi_R", "chi_B", OUT, OUT, IN, IN, "r2", "d2", 0),
        _corner(chi_charges, "chi_L", "chi_B", IN, OUT, OUT, IN, "l2", "d2", 1),
        _corner(chi_charges, "chi_R", "chi_T", OUT, IN, IN, OUT, "r2", "u2", 2),
        _corner(chi_charges, "chi_L", "chi_T", IN, IN, OUT, OUT, "l2", "u2", 3),
    )


@pytest.mark.parametrize(
    ("direction", "corner_ix", "label"),
    [
        ("left", 0, "chi_B"),
        ("right", 1, "chi_B"),
        ("top", 0, "chi_R"),
        ("bottom", 2, "chi_R"),
    ],
)
def test_the_named_leg_is_the_seam_the_projector_compresses(
    direction, corner_ix, label
):
    """Each direction reads the chi leg whose bond ``chi_new`` replaces."""
    chi_charges = [0, 0, 0, 1]
    corners = _corners(chi_charges)

    got = _incoming_chi_charges(*corners, direction, chi=4)

    expected = np.asarray(
        corners[corner_ix].indices[corners[corner_ix].labels().index(label)].charges,
        dtype=np.int32,
    )
    assert got is not None
    assert got.tolist() == expected.tolist()


def test_a_leg_of_the_wrong_width_is_refused():
    """Mid chi-ramp the environment is still at the old width.

    A stale inventory is worse than the tiled guess — it would size the new
    bond by the *previous* chi — so the leg is only used when it is exactly
    ``chi`` wide.
    """
    corners = _corners([0, 0, 0, 1])

    assert _incoming_chi_charges(*corners, "left", chi=8) is None
    assert _incoming_chi_charges(*corners, "left", chi=4) is not None


def test_the_traced_projector_inherits_the_environment_inventory():
    """chi_new follows the incoming chi leg, not ``base_charges`` tiled to chi.

    The corners' chi legs carry ``{0: 3, 1: 1}`` while ``base_charges=[0, 1]``
    tiles to ``{0: 2, 1: 2}``, so the two rules give different answers and the
    test can tell which one ran.
    """
    from collections import Counter

    corners = _corners([0, 0, 0, 1])
    base_charges = np.array([0, 1], dtype=np.int32)
    inventory: Counter = Counter()

    def _probe(alpha):
        scaled = {k: alpha * b for k, b in corners[0].blocks.items()}
        Q_TL = SymmetricTensor._from_blocks_unchecked(scaled, corners[0].indices)
        P_top = _compute_2x2_projector_symmetric(
            Q_TL, *corners[1:], chi=4, direction="left", base_charges=base_charges
        )[0]
        inventory.update(int(q) for q in np.asarray(P_top.indices[2].charges))
        return jnp.asarray(0.0)

    jax.eval_shape(_probe, jnp.asarray(1.0))

    assert sum(inventory.values()) == 4
    assert inventory == Counter({0: 3, 1: 1}), (
        f"traced chi_new is {dict(inventory)}; the tiled guess would be "
        "{0: 2, 1: 2} — the incoming leg was not inherited"
    )


def test_the_eager_cut_is_untouched():
    """Eagerly the spectrum still chooses, so the two paths may disagree.

    #922's value-aware cut is only reachable eagerly; this pins that #929 did
    not quietly route the eager path through the static rule as well.
    """
    from collections import Counter

    corners = _corners([0, 0, 0, 1])
    base_charges = np.array([0, 1], dtype=np.int32)

    P_top = _compute_2x2_projector_symmetric(
        *corners, chi=4, direction="left", base_charges=base_charges
    )[0]
    eager = Counter(int(q) for q in np.asarray(P_top.indices[2].charges))

    assert sum(eager.values()) == 4
    # The floor still applies: every named charge keeps a slot.
    for q in (0, 1):
        assert eager[q] >= 1


# --------------------------------------------------------------------------- #
# The root-implicit symmetric path truncates globally already                  #
# --------------------------------------------------------------------------- #


def test_the_root_implicit_layout_is_the_global_top_chi():
    """``ctm_ad_mode="root_implicit_symmetric"`` never had the #922 defect.

    That path does not go through the 2x2 projector at all — it has its own
    sweep, and ``_ctm_root_implicit_sym_sectors.sector_svd`` decomposes each
    charge sector and then takes **one global top-chi** over the union of the
    spectra, recording the result in a ``BondLayout`` that is frozen for the
    adjoint.  So the retained charge distribution is data-dependent there, and
    the quota that #922 removed from the eager cut was never imposed on it.

    Pinned here because the other path shows how this regresses: a per-sector
    quota looks like a stabilisation and silently caps the bond.  The fixture
    is deliberately lopsided — sector ``q = 0`` holds every large singular
    value — so an even or capacity-shaped split would fail.
    """
    from tenax.algorithms._ctm_root_implicit_sym_sectors import sector_svd

    sym = U1Symmetry()
    charges = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    row = TensorIndex.from_charges(sym, charges, IN, label="row")
    col = TensorIndex.from_charges(sym, charges, OUT, label="col")

    blocks = {
        (0, 0): jnp.diag(jnp.array([9.0, 8.0, 7.0])),
        (1, 1): jnp.diag(jnp.array([0.3, 0.2, 0.1])),
    }
    M = SymmetricTensor._from_blocks_unchecked(blocks, (row, col))

    _sectors, layout = sector_svd(M, chi=4, row_axis=0, col_axis=1)

    assert layout.total == 4
    assert layout.dim_of(0) == 3, (
        f"q=0 holds the three largest singular values but kept "
        f"{layout.dim_of(0)} — the truncation is not global"
    )
    assert layout.dim_of(1) == 1

    # ... and the split follows the values, not the sector sizes: swap which
    # sector is dominant and the layout swaps with it.
    swapped = SymmetricTensor._from_blocks_unchecked(
        {
            (0, 0): jnp.diag(jnp.array([0.3, 0.2, 0.1])),
            (1, 1): jnp.diag(jnp.array([9.0, 8.0, 7.0])),
        },
        (row, col),
    )
    _s2, layout2 = sector_svd(swapped, chi=4, row_axis=0, col_axis=1)

    assert (layout2.dim_of(0), layout2.dim_of(1)) == (1, 3)
