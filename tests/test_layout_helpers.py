"""layout_of / same_layout compare per-sector sizes on every leg."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_init import CTMTensorEnv
from tenax.algorithms.ctm_layout import layout_of, same_layout
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor


def _env(chi_charges):
    sym = U1Symmetry()

    def leg(lbl, flow):
        return TensorIndex.from_charges(
            sym, np.array(chi_charges, np.int32), flow, label=lbl
        )

    def corner(a, b):
        return SymmetricTensor.random_normal(
            (leg(a, FlowDirection.IN), leg(b, FlowDirection.OUT)), jax.random.PRNGKey(0)
        )

    # Only the corners matter for these tests; reuse a corner for the edges.
    c = corner("x", "y")
    return CTMTensorEnv(c, c, c, c, c, c, c, c)


def test_equal_sector_sizes_are_the_same_layout():
    a = {(0, 0): _env([0, 1, -1, 0])}
    b = {(0, 0): _env([0, -1, 1, 0])}
    assert same_layout(layout_of(a), layout_of(b))


def test_one_sector_moved_at_equal_total_is_a_different_layout():
    a = {(0, 0): _env([0, 1, -1, 0])}
    b = {(0, 0): _env([0, 1, 1, -1])}  # chi 4 both; (0:2,1:1,-1:1) vs (0:1,1:2,-1:1)
    assert not same_layout(layout_of(a), layout_of(b))
