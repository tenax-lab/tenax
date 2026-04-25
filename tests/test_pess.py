import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.pess import IPESSState, kagome_triangle_xxz_hamiltonian


def test_ipess_state_shapes():
    D, d = 4, 3
    key = jax.random.PRNGKey(0)
    state = IPESSState.random(D=D, d=d, key=key)
    assert state.R_a.shape == (D, D, d)
    assert state.R_b.shape == (D, D, d)
    assert state.R_c.shape == (D, D, d)
    assert state.T_u.shape == (D, D, D)
    assert state.T_d.shape == (D, D, D)
    assert all(lam.shape == (D,) for lam in state.lambdas)
    assert state.R_a.dtype == jnp.complex128
    assert state.T_u.dtype == jnp.complex128


def test_triangle_hamiltonian_hermitian_spin1():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    assert H.shape == (27, 27)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_triangle_hamiltonian_hermitian_spin_half():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    assert H.shape == (8, 8)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_triangle_hamiltonian_xy_isotropic():
    H1 = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    H0 = kagome_triangle_xxz_hamiltonian(delta=0.0, d=3)
    diff = H1 - H0
    # Difference should be only the Sz Sz couplings
    assert np.linalg.norm(diff) > 0
