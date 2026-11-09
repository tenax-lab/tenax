"""Hasik's D=2 J2=0 Heisenberg state, for the compute_spectrum tests.

Source: github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), j2=0.0, D=2,
chi_opt48, in the sublattice-rotated frame. peps-torch lists E=-0.66023109348.
"""

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ipeps_excitation_ctm import CELL, NEIGHBORS

_D2_ENTRIES = [
    1.0, -1.6194835639214339e-02, 1.7282585382645214e-02, 4.1985986298862715e-01,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4700030141459453e-05, 3.0093367495500688e-02,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4720562436946177e-05, 3.0093429238755503e-02,
    -4.4720562436946177e-05, 3.0093429238755503e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4720562436946177e-05, 3.0093429238755503e-02,
    -4.4720562436946177e-05, 3.0093429238755503e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    -4.4700030141459453e-05, 3.0093367495500688e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    -2.5360940365643496e-05, 1.6170125837876856e-03, -1.9191912448893797e-06, 7.7277947587816155e-05,
]  # fmt: skip

SY_REAL = np.array([[0.0, 0.5], [-0.5, 0.0]])  # i S^y
SY_COMPLEX = np.array([[0.0, -0.5j], [0.5j, 0.0]])
SX = np.array([[0.0, 0.5], [0.5, 0.0]])
SZ = np.diag([0.5, -0.5])
E_D2 = -0.66023109348


def hasik_d2() -> np.ndarray:
    """The state, normalised to unit Frobenius norm, in ``(u, d, l, r, s)``."""
    A = np.array(_D2_ENTRIES).reshape(2, 2, 2, 2, 2)
    return A / np.linalg.norm(A)


def exact_u1_d2() -> np.ndarray:
    """``hasik_d2`` cut to its two leading numbers: U(1) and C4v hold exactly.

    Every kept entry has charge ``(1 - 2 s) + 2 (u + d + l + r) = 1``, so the
    environment has exact pairs. E = -0.6602280177 at chi=10.
    """
    A = np.zeros((2,) * 5)
    A[0, 0, 0, 0, 0] = 1.0
    for leg in range(4):
        index = [0, 0, 0, 0]
        index[leg] = 1
        A[(*index, 1)] = 0.41986
    return A / np.linalg.norm(A)


def heisenberg_gate() -> np.ndarray:
    """``S.S`` as ``(s1, s2, s1', s2')``: next-nearest neighbours, same sublattice."""
    return np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    ).reshape(2, 2, 2, 2)


def rotated_gate() -> np.ndarray:
    """``S.S`` in the sublattice-rotated frame: nearest neighbours."""
    H = heisenberg_gate().reshape(4, 4)
    W = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return (W @ H @ W.T).reshape(2, 2, 2, 2)


def apply_op(A: np.ndarray, op: np.ndarray) -> np.ndarray:
    return np.tensordot(A, op, axes=([4], [1]))


def regauge_cell(envs, rng):
    """One invertible matrix per link of the cell, so shared bonds stay closed.

    A link joins ``c`` to its right or bottom neighbour in one of the four edge
    rows. A contraction that pairs environments of the wrong cell sites sees two
    unrelated matrices and changes value.
    """
    chi = envs[(0, 0)]["C1"].shape[0]

    def gauge():
        g = rng.normal(size=(chi, chi)) + 3 * np.eye(chi)
        return g, np.linalg.inv(g)

    top = {c: gauge() for c in CELL}
    right = {c: gauge() for c in CELL}
    bottom = {c: gauge() for c in CELL}
    left = {c: gauge() for c in CELL}
    out = {}
    for c in CELL:
        E = {k: np.asarray(v) for k, v in envs[c].items()}
        west, north = NEIGHBORS[c]["left"], NEIGHBORS[c]["top"]
        out[c] = {
            "C1": left[north][0].T @ E["C1"] @ top[west][0],
            "T1": np.einsum("ab,bxyd,dc->axyc", top[west][1], E["T1"], top[c][0]),
            "C2": top[c][1] @ E["C2"] @ right[north][0],
            "T2": np.einsum("ab,bxyd,dc->axyc", right[north][1], E["T2"], right[c][0]),
            "C3": right[c][1] @ E["C3"] @ bottom[c][1].T,
            "T3": np.einsum("ab,bxyd,dc->axyc", bottom[west][1], E["T3"], bottom[c][0]),
            "C4": left[c][1] @ E["C4"] @ bottom[west][0],
            "T4": np.einsum("ab,bxyd,dc->axyc", left[north][1], E["T4"], left[c][0]),
        }
    return {c: {k: jnp.asarray(v) for k, v in out[c].items()} for c in CELL}
