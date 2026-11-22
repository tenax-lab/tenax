"""Shared fixtures for the charged C4v CTM tests (#762, #763).

``test_c4v_staggered.py`` (core) and ``test_ctm_tensor_c4v.py`` (algorithm)
import from here.  The probes here do not use the code under test as their
oracle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor


def c4v_fixture(sym, vc, pc, key=42):
    """A real C4v site tensor in the charged C4v layout; D = len(vc), d = len(pc).

    u, l are OUT with charges ``vc``; d, r are IN with ``sym.dual(vc)``; phys is
    IN with ``pc``.  The four faces then carry the same flow-weighted charge at
    each position, so ``symmetrize_c4v`` keeps the block structure.  The data
    is random (``key``), projected onto the C4v-invariant subspace.
    """
    vc = np.asarray(vc, dtype=np.int32)
    vd = sym.dual(vc)
    pc = np.asarray(pc, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )
    raw = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(key))
    data = symmetrize_c4v(jnp.asarray(raw.todense()))
    return SymmetricTensor.from_dense(data, indices)


def leg(T, label):
    """The index of ``T`` with this label."""
    return T.indices[T.labels().index(label)]


def pairs(x, y):
    """True when the legs ``x`` and ``y`` contract: opposite flows, same charges."""
    return x.flow != y.flow and np.array_equal(x.charges, y.charges)
