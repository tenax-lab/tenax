"""Heisenberg gate constructors + path-dependent init dispatcher.

For ``single_site`` (1×1 + sublattice-rotated gate, unconstrained tensor):
    SU on the rotated gate converges to the |↑↑⟩ saddle (E=−0.5/site) which
    L-BFGS cannot escape (see ``ipeps_optimize.py:1389`` reference-mode
    comment).  Use random init instead.
For ``bipartite_2site`` (2-tensor checkerboard + bare gate):
    Tenax SU on the bare gate finds a Néel-like state.  Use it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax import CTMConfig, ipeps, iPEPSConfig, sublattice_rotate_gate

DTYPE_NP = np.complex128


def build_heisenberg_gate(dtype=jnp.complex128) -> np.ndarray:
    """H = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+) for spin-½, returned as (2,2,2,2).

    Equivalent to Sx⊗Sx + Sy⊗Sy + Sz⊗Sz but written in raising/lowering
    form so the matrix is real-valued in the standard basis (the imaginary
    parts of Sy⊗Sy cancel exactly).  This lets us build the gate in real
    dtype when needed — e.g. ``dtype=jnp.float64`` for SU bootstrap, where
    Tenax's legacy ``ctm_2site`` has a complex-dtype carry bug
    (``ipeps_ctm_convergence.py:268``).
    """
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return np.asarray(H.reshape(2, 2, 2, 2))


def build_sublattice_rotated_gate(dtype=jnp.complex128) -> np.ndarray:
    """``single_site`` path gate: bare H rotated by Y on B sublattice.

    Lets a 1×1 unit cell encode the AFM ground state in the rotated frame.
    """
    return np.asarray(sublattice_rotate_gate(jnp.asarray(build_heisenberg_gate(dtype))))


def _random_complex(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    return (re + 1j * im).astype(DTYPE_NP)


def make_init(
    *,
    path: str,
    D: int,
    seed: int = 0,
    su_num_steps: int = 100,
    su_dt: float = 0.01,
) -> np.ndarray:
    """Build the init tensor for the given path.

    Args:
        path: ``"single_site"`` (random) or ``"bipartite_2site"`` (Tenax SU).
        D:    iPEPS bond dimension.
        seed: numpy seed for random init (only used for ``single_site``).
        su_num_steps / su_dt: SU schedule (only used for ``bipartite_2site``).

    Returns:
        ``single_site``      → ``(D, D, D, D, d)`` complex128 array.
        ``bipartite_2site``  → ``(2, D, D, D, D, d)`` stacked (A, B) complex128.
    """
    d = 2  # spin-½
    if path == "single_site":
        return _random_complex((D, D, D, D, d), seed=seed)
    elif path == "bipartite_2site":
        # SU runs in float64 to dodge the legacy ctm_2site complex-carry bug
        # (ipeps_ctm_convergence.py:268).  Init is cast to complex128 below
        # so the AD optimizers in both libs see complex tensors.
        gate = jnp.asarray(build_heisenberg_gate(dtype=jnp.float64))
        config = iPEPSConfig(
            max_bond_dim=D,
            num_imaginary_steps=su_num_steps,
            dt=su_dt,
            ctm=CTMConfig(chi=4 * D),
            unit_cell="2site",
        )
        _, (A, B), _ = ipeps(gate, None, config)
        # ipeps() returns DenseTensor; extract underlying jax.Array.
        A_arr = A.todense() if hasattr(A, "todense") else A
        B_arr = B.todense() if hasattr(B, "todense") else B
        return np.stack([np.asarray(A_arr), np.asarray(B_arr)], axis=0).astype(DTYPE_NP)
    else:
        raise ValueError(f"unknown path: {path}")
