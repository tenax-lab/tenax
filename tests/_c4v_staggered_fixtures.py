"""Shared fixtures for the charged C4v CTM tests (#762, #763).

``test_c4v_staggered.py`` (core) and ``test_ctm_tensor_c4v.py`` (algorithm)
import from here.  The probes here do not use the code under test as their
oracle.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps import symmetrize_c4v
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import SymmetricTensor
from tenax.linalg import _dense_svd


def c4v_fixture(sym, vc, pc, key=42):
    """A real C4v site tensor in the charged C4v layout; D = len(vc), d = len(pc).

    u, l are OUT with charges ``vc``; d, r are IN with ``sym.dual(vc)``; phys is
    IN with ``pc``.  The four faces then carry the same flow-weighted charge at
    each position, so ``symmetrize_c4v`` keeps the block structure.  The data
    is random (``key``), projected onto the C4v-invariant subspace.
    """
    indices = c4v_indices(sym, vc, pc)
    raw = SymmetricTensor.random_normal(indices, jax.random.PRNGKey(key))
    data = symmetrize_c4v(jnp.asarray(raw.todense()))
    return SymmetricTensor.from_dense(data, indices)


# The free parameter of the D=2 exact U(1) state (peps-torch; ``exact_u1_d2``
# in ``_ipeps_spectrum_fixtures.py``).  E(T_EXACT) = -0.660228017729 at chi=16;
# the chi=16 minimum over t is -0.660231093480 at t = 0.4210174.
T_EXACT = 0.41986


def c4v_indices(sym, vc, pc):
    """The five legs of a charged C4v site: u, l OUT with ``vc``; d, r IN
    with ``sym.dual(vc)``; phys IN with ``pc``."""
    vc = np.asarray(vc, dtype=np.int32)
    vd = sym.dual(vc)
    return (
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, vc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, vd.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, np.asarray(pc, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )


def charged_d2(t):
    """The D=2 U(1) C4v family: A[0000, 0] = 1 and A[e_k, 1] = t, normalised.

    One virtual leg at charge 1 with the spin flipped; ``T_EXACT`` is the
    peps-torch state.  The C4v projection has rank 2 on this layout, so any
    real C4v tensor of it is ``charged_d2(t)`` up to sign and norm.
    """
    A = np.zeros((2,) * 5)
    A[0, 0, 0, 0, 0] = 1.0
    for k in range(4):
        index = [0, 0, 0, 0]
        index[k] = 1
        A[(*index, 1)] = t
    A /= np.linalg.norm(A)
    return SymmetricTensor.from_dense(
        jnp.asarray(A), c4v_indices(U1Symmetry(), [0, 1], [0, 1])
    )


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


def with_data(X, data):
    """``X`` with a new flat block buffer (same pytree structure)."""
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(X), [data])


def c4v_direction(A, key):
    """A random unit direction in the C4v subspace of A's block buffer."""
    r = jax.random.normal(key, A._data.shape)
    v = SymmetricTensor.from_dense(
        symmetrize_c4v(with_data(A, r).todense()), A.indices
    )._data
    return v / jnp.linalg.norm(v)


FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def fd_plateau(phi, hs=FD_STEPS):
    """The central difference of ``phi`` at 0 on its h plateau: ``(fd, fd_unc)``.

    ``fd`` is the mean of the neighbouring pair of h that agree best.  The
    pair is chosen by the agreement of neighbouring h, never by the
    agreement with AD (spec 2, the FD rule).  ``fd_unc`` is the largest of
    that pair's difference and the differences of its two adjacent pairs
    (clipped at the ends of ``hs``).  The pair's own difference is not
    enough: on the plateau the truncation error (~h^2) and the forward-lag
    error (~1/h, a forward stopped at its element-wise tolerance) can have
    the same sign, and then that difference understates the FD error (about
    3x on the #762 fixture, direction 0).
    """
    fds = [(phi(h) - phi(-h)) / (2.0 * h) for h in hs]
    diffs = [abs(x - y) for x, y in zip(fds, fds[1:])]
    k = int(np.argmin(diffs))
    return 0.5 * (fds[k] + fds[k + 1]), max(diffs[max(k - 1, 0) : k + 2])


@contextlib.contextmanager
def frozen_projectors():
    """The "P frozen" mutation: stop_gradient on the traced projector.

    The engine has no frozen option (spec 2, section 3); the FD regime
    asserts compare against this.  The caches are cleared on entry and exit,
    so no jitted program keeps the wrong projector.
    """
    from tenax.algorithms import _ctm_tensor_c4v_charged_ad as ad

    real = ad._c4v_staggered_projectors_at

    def frozen(Qf, layout):
        return jax.tree_util.tree_map(jax.lax.stop_gradient, real(Qf, layout))

    jax.clear_caches()
    ad._c4v_staggered_projectors_at = frozen
    try:
        yield
    finally:
        ad._c4v_staggered_projectors_at = real
        jax.clear_caches()
