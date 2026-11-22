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
from tenax.linalg import _dense_svd


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


def corner_edge(env):
    """The corner and the edge of a C4v env, with the sweep's labels.

    Inverts the C1/T1 relabel of ``_c4v_to_full_env`` and of
    ``_c4v_staggered_to_full_env``.
    """
    C = env.C1.relabels({"c1_d": "c_a", "c1_r": "c_b"})
    T = env.T1.relabels({"t1_l": "t_l", "u2": "D2", "t1_r": "t_r"})
    return C, T


def norm_sv(C):
    """The singular values of the chi x chi corner, normalised to sum 1."""
    s = np.asarray(_dense_svd(C.todense(), compute_uv=False))
    return s / s.sum()


def sv_gap(s1, s2):
    """The largest difference of two spectra; the shorter is padded with zeros."""
    n = max(s1.size, s2.size)
    return float(
        np.max(np.abs(np.pad(s1, (0, n - s1.size)) - np.pad(s2, (0, n - s2.size))))
    )


def cut_gap(Qf, chi):
    """The relative gap at the cut, (s[chi-1] - s[chi]) / s[chi-1].

    s holds the |eigenvalues| of the symmetrised dense enlarged corner, sorted
    down: a mirror-pair value s appears as +s and -s, a self-partner value
    once.  That is the ranking of both C4v sweeps.  The dense Q is
    (chi D^2) x (chi D^2): small at the D and chi of the tests.
    """
    M = np.asarray(Qf.todense())
    s = np.sort(np.abs(np.linalg.eigvalsh(0.5 * (M + M.T))))[::-1]
    return float((s[chi - 1] - s[chi]) / s[chi - 1])


def charged_block_ratio(Qf):
    """The largest singular value of a charged block over that of all blocks."""
    sym = Qf.indices[0].symmetry
    top = charged = 0.0
    for key, block in Qf.blocks.items():
        q = np.asarray([key[0]], dtype=np.int32)
        value = float(np.linalg.norm(np.asarray(block), 2))
        top = max(top, value)
        if sym.dual(q)[0] != q[0]:
            charged = max(charged, value)
    return charged / top
