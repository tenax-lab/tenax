"""#1098: unit tests for ``_chi_layout``, the eager phase's stop criterion.

The layout the jitted projector freezes (#929) is the per-sector
``(charge, size)`` split of each chi leg -- not its total size, and not the
order of a tensor's legs.  Hand-built environments, no CTM: core.
"""

from __future__ import annotations

import jax
import numpy as np

from tenax.algorithms._ctm_python_loop import _chi_layout
from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT
SYM = U1Symmetry()
D2 = np.array([0, 1, -1, 0], dtype=np.int32)

# (field, (label, flow) per leg); chi legs are every leg but u2/r2/d2/l2.
_SPECS = {
    "C1": (("c1_d", IN), ("c1_r", OUT)),
    "C2": (("c2_l", IN), ("c2_d", OUT)),
    "C3": (("c3_u", IN), ("c3_l", OUT)),
    "C4": (("c4_r", OUT), ("c4_u", IN)),
    "T1": (("t1_l", IN), ("u2", OUT), ("t1_r", OUT)),
    "T2": (("t2_u", IN), ("r2", IN), ("t2_d", OUT)),
    "T3": (("t3_r", OUT), ("d2", IN), ("t3_l", IN)),
    "T4": (("t4_d", OUT), ("l2", OUT), ("t4_u", IN)),
}


def _env(chi_charges, c1_charges=None):
    """An env whose every chi leg carries ``chi_charges`` (C1: ``c1_charges``)."""
    key = jax.random.PRNGKey(0)
    fields = []
    for field, legs in _SPECS.items():
        idx = []
        for label, flow in legs:
            if label.endswith("2"):
                q = D2
            elif field == "C1" and c1_charges is not None:
                q = c1_charges
            else:
                q = chi_charges
            idx.append(TensorIndex.from_charges(SYM, q.copy(), flow, label=label))
        key, sub = jax.random.split(key)
        fields.append(SymmetricTensor.random_normal(tuple(idx), sub))
    return CTMTensorEnv(*fields)


BASE = np.array([-2, 0, 0, 0, 0, 0, 0, 2], dtype=np.int32)  # (-2:1, 0:6, 2:1)
MOVED = np.array([-4, -2, -2, 0, 0, 0, 2, 2], dtype=np.int32)  # same chi = 8


def test_an_equal_total_chi_with_a_different_sector_split_is_a_change():
    """Kills "compare by total chi": both legs are 8 wide."""
    a = {(0, 0): _env(BASE)}
    b = {(0, 0): _env(BASE, c1_charges=MOVED)}
    assert a[(0, 0)].C1.indices[0].dim == b[(0, 0)].C1.indices[0].dim == 8
    assert _chi_layout(a) != _chi_layout(b)


def test_swapping_a_corners_legs_is_not_a_change():
    """The cold-init corners come back from a sweep with their two legs in
    the other order; that alone must not count as a layout change."""
    env = _env(BASE, c1_charges=MOVED)
    swapped = env._replace(C1=env.C1.transpose((1, 0)))
    assert swapped.C1.labels() == ("c1_r", "c1_d")
    assert _chi_layout({(0, 0): env}) == _chi_layout({(0, 0): swapped})


def test_the_layout_ignores_the_d2_legs_and_dense_tensors():
    env = _env(BASE)
    layout = _chi_layout({(0, 0): env})
    labels = {entry[2] for entry in layout}
    assert labels.isdisjoint({"u2", "r2", "d2", "l2"})
    assert len(layout) == 4 * 2 + 4 * 2  # two chi legs per corner and edge
    assert _chi_layout({(0, 0): CTMTensorEnv(*[None] * 8)}) == ()


def test_charge_sorting_a_chi_leg_is_a_faithful_basis_permutation():
    """The eager outputs' chi legs are put in ascending-charge order so that
    equal layouts give equal pytree structures (no per-call recompile).  The
    relabelling must be the dense permutation ``argsort(charges, stable)`` on
    every chi leg, leave the D^2 leg alone, and reuse the block data."""
    from tenax.algorithms._ctm_python_loop import _charge_sorted_chi_legs

    shuffled = np.array([0, 2, -2, 0, -2, 2, -4, 4], dtype=np.int32)
    idx = (
        TensorIndex.from_charges(SYM, shuffled, IN, label="t1_l"),
        TensorIndex.from_charges(SYM, D2[::-1].copy(), OUT, label="u2"),
        TensorIndex.from_charges(SYM, shuffled[::-1].copy(), OUT, label="t1_r"),
    )
    t = SymmetricTensor.random_normal(idx, jax.random.PRNGKey(3))
    s = _charge_sorted_chi_legs(t)
    p0 = np.argsort(shuffled, kind="stable")
    p2 = np.argsort(shuffled[::-1], kind="stable")
    np.testing.assert_array_equal(
        np.asarray(s.todense()), np.asarray(t.todense())[p0][:, :, p2]
    )
    assert np.array_equal(np.asarray(s.indices[1].charges), D2[::-1])
    assert s._data is t._data


def _rec(alloc, spectra, floor=None):
    return {
        "q": np.arange(len(alloc), dtype=np.int32),
        "alloc": np.asarray(alloc, dtype=np.int32),
        "floor": np.asarray(floor or [0] * len(alloc), dtype=np.int32),
        "s": tuple(np.asarray(s, dtype=float) for s in spectra),
    }


def test_the_probe_accepts_a_top_chi_cut_and_rejects_a_frozen_one():
    """The host side of the #1098 probe: a traced allocation is accepted iff
    it is a global top-chi cut, up to ties within 1e-10 * s_max."""
    from tenax.algorithms._ctm_python_loop import _probe_says_layout_is_eager as ok

    spectra = ([1.0, 0.5, 0.01], [0.3, 0.2])  # top-3 = 1.0, 0.5, 0.3
    assert ok([_rec([2, 1], spectra)])
    assert not ok([_rec([3, 0], spectra)])  # keeps 0.01, drops 0.3
    assert not ok([_rec([1, 2], spectra)])  # keeps 0.2, drops 0.5
    # A tie at the cut: either side may be kept, within the tolerance only.
    tied = ([1.0, 0.25], [0.25 * (1 + 1e-12), 0.1])
    assert ok([_rec([2, 0], tied)]) and ok([_rec([1, 1], tied)])
    split = ([1.0, 0.25], [0.25 * (1 + 1e-6), 0.1])
    assert not ok([_rec([2, 0], split)]) and ok([_rec([1, 1], split)])
    # The eager floor: a base charge with a vector must keep one slot.
    assert not ok([_rec([3, 0], ([1.0, 0.9, 0.8], [0.01]), floor=[1, 1])])
    assert ok([_rec([2, 1], ([1.0, 0.9, 0.8], [0.01]), floor=[1, 1])])
    # Every record must pass, and no record is no certificate.
    assert not ok([_rec([2, 1], spectra), _rec([3, 0], spectra)])
    assert not ok([])
    assert not ok([_rec([1, 1], ([0.0, 0.0], [0.0]))])  # blind: all zero
