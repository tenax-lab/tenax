"""#890: ``ipeps()`` accepted an ``initial_peps`` whose bond dimension disagreed
with ``config.max_bond_dim`` and never validated it -- the mismatch surfaced
several frames into the simple-update sweep as an opaque ``TypeError: cannot
reshape array of shape (D0,) into shape [...]``.  Warm-starting a larger-D run
from a converged smaller-D state is the natural way to hit it, and neither the
signature nor the docstring hinted the two had to agree.

The fix validates the supplied pair up front on the default SU path and raises a
clear ``ValueError``.  ``su_independent_bond_lambdas=True`` re-dimensions over a
full four-bond cycle and is left exempt.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import (
    _validate_initial_bond_dim,
    _wrap_as_dense_tensor,
    ipeps,
    iPEPSConfig,
)

# Bucket is assigned by ``tests/conftest.py`` (registered ``core``): the guard
# is a clear-error gate on the default public path, and every case here is
# cheap (validation fires before SU; the two SU-only cases pass
# ``compute_energy=False``).


def _heisenberg_gate():
    d = 2
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = Sp.T
    H = np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp))
    return jnp.array(H.reshape(d, d, d, d))


def _pair(D: int):
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.normal(size=(D, D, D, D, 2)))
    B = jnp.asarray(rng.normal(size=(D, D, D, D, 2)))
    return A, B


# -- unit anchors on the validator -------------------------------------------


def test_validator_raises_on_bond_dim_mismatch():
    A = _wrap_as_dense_tensor(_pair(2)[0])
    with pytest.raises(ValueError) as exc:
        _validate_initial_bond_dim(A, "A", 4)
    msg = str(exc.value)
    assert "#890" in msg
    assert "max_bond_dim=4" in msg
    assert "{'u': 2, 'd': 2, 'l': 2, 'r': 2}" in msg
    # It suggests the matching max_bond_dim for a uniform site.
    assert "set config.max_bond_dim=2 to match" in msg


def test_validator_is_a_noop_when_bonds_match():
    A = _wrap_as_dense_tensor(_pair(3)[0])
    assert _validate_initial_bond_dim(A, "A", 3) is None  # no raise


# -- integration through ipeps() ---------------------------------------------


def test_ipeps_rejects_mismatched_initial_peps_on_default_path():
    """The reported repro: D=2 state, max_bond_dim=4 -> clear ValueError, not a
    reshape TypeError from deep in the sweep.  Fires before SU, so it is cheap.
    """
    gate = _heisenberg_gate()
    A, B = _pair(2)
    with pytest.raises(ValueError, match=r"#890"):
        ipeps(gate, (A, B), iPEPSConfig(max_bond_dim=4, num_imaginary_steps=4))

    # No TypeError leaks: the failure must be the validated ValueError.
    with pytest.raises(ValueError):
        ipeps(gate, (A, B), iPEPSConfig(max_bond_dim=4, num_imaginary_steps=4))


def test_ipeps_accepts_matching_initial_peps():
    gate = _heisenberg_gate()
    A, B = _pair(2)
    # compute_energy=False keeps this to the SU sweep (no CTM): D=2, ~fast.
    _env, (A_out, B_out), _e = ipeps(
        gate,
        (A, B),
        iPEPSConfig(max_bond_dim=2, num_imaginary_steps=4),
        compute_energy=False,
    )
    assert A_out is not None and B_out is not None


def test_ipeps_allows_redimension_with_independent_bond_lambdas():
    """The exempt path: ``su_independent_bond_lambdas=True`` re-dimensions a
    supplied D=2 state up to max_bond_dim=4 over a full bond cycle.  The guard
    must not block it -- else it would remove a capability the library has.
    """
    gate = _heisenberg_gate()
    A, B = _pair(2)
    _env, (A_out, B_out), _e = ipeps(
        gate,
        (A, B),
        iPEPSConfig(
            max_bond_dim=4,
            num_imaginary_steps=4,
            su_independent_bond_lambdas=True,
        ),
        compute_energy=False,
    )
    # The state came back re-dimensioned to D=4 on every virtual leg.
    labels = A_out.labels()
    for lbl in ("u", "d", "l", "r"):
        assert int(A_out.indices[labels.index(lbl)].dim) == 4
