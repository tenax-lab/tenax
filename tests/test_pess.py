import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_trotter_gate_unitarity_real_time():
    from tenax.algorithms.pess import make_triangle_gate

    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    gate = make_triangle_gate(H, dt=1j * 0.05, d=3)  # real time
    G = np.asarray(gate).reshape(27, 27)
    np.testing.assert_allclose(G @ G.conj().T, np.eye(27), atol=1e-10)


def test_trotter_gate_imag_time_decreases_norm_on_excited_state():
    from tenax.algorithms.pess import make_triangle_gate

    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    gate = make_triangle_gate(H, dt=0.1, d=3)
    G = np.asarray(gate).reshape(27, 27)
    eigvals = np.linalg.eigvalsh(H)
    # Largest singular value of e^{-dt H} = e^{-dt * lambda_min}
    assert np.max(np.linalg.svd(G, compute_uv=False)) == pytest.approx(
        np.exp(-0.1 * eigvals[0]), rel=1e-8
    )


def test_hosvd_truncate_idempotent_no_truncation():
    """If D_max >= input dim, theta should round-trip."""
    from tenax.algorithms.pess import hosvd_truncate

    D, d = 3, 3
    key = jax.random.PRNGKey(7)
    theta = (
        jax.random.normal(key, (D, D, D, d, d, d))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d, d, d))
    ).astype(jnp.complex128)
    S_a, S_b, S_c, core, lams = hosvd_truncate(theta, D_max=D * d, d=d)
    # Reconstruct theta:
    # theta[a,b,c,p_a,p_b,p_c] = sum_{i,j,k} S_a[a,i,p_a] * S_b[b,j,p_b] * S_c[c,k,p_c] * core[i,j,k]
    theta_reco = jnp.einsum("aip,bjq,ckr,ijk->abcpqr", S_a, S_b, S_c, core)
    np.testing.assert_allclose(theta, theta_reco, atol=1e-10)
