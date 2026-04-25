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


def test_hosvd_truncate_under_truncation():
    """When D_max < D * d, S_x must be clipped and isometric, and the
    reconstruction must be a non-trivial approximation (closer to theta
    than zero)."""
    from tenax.algorithms.pess import hosvd_truncate

    D, d = 3, 3
    D_max = 4  # D_int = 4 < D * d = 9 -> truncation actually happens
    assert D_max < D * d
    key = jax.random.PRNGKey(11)
    theta = (
        jax.random.normal(key, (D, D, D, d, d, d))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d, d, d))
    ).astype(jnp.complex128)

    S_a, S_b, S_c, core, lams = hosvd_truncate(theta, D_max=D_max, d=d)

    # 1. Output dimensions are correctly clipped on the internal bond.
    assert S_a.shape == (D, D_max, d)
    assert S_b.shape == (D, D_max, d)
    assert S_c.shape == (D, D_max, d)
    assert core.shape == (D_max, D_max, D_max)
    for lam in lams:
        assert lam.shape == (D_max,)

    # 2. Each S_x is an isometry along the internal bond. S_x has layout
    # (D_ext, D_int, d); unfolding to (D_ext * d, D_int) and contracting on
    # the external+physical legs should give the (D_int, D_int) identity.
    eye = jnp.eye(D_max, dtype=jnp.complex128)
    for S in (S_a, S_b, S_c):
        S_unf = S.transpose(0, 2, 1).reshape(D * d, D_max)
        gram = S_unf.conj().T @ S_unf
        np.testing.assert_allclose(gram, eye, atol=1e-10)

    # 3. The truncated reconstruction is a non-trivial approximation: it
    # is closer to theta than the zero tensor would be.
    theta_reco = jnp.einsum("aip,bjq,ckr,ijk->abcpqr", S_a, S_b, S_c, core)
    err = jnp.linalg.norm(theta - theta_reco)
    norm_theta = jnp.linalg.norm(theta)
    assert float(err) < float(norm_theta)
