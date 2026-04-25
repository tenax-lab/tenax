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


def _build_theta_full(state, triangle):
    """Reference: build the gauged theta tensor from an IPESSState.

    Uses the same gauging as ``pess_simple_update_triangle``: external lambdas
    on axis 0 of each R, internal lambdas on axis 1, contracted with the
    chosen simplex tensor.
    """
    if triangle == "up":
        T = state.T_u
        ext = (state.lambdas[3], state.lambdas[4], state.lambdas[5])
        int_ = (state.lambdas[0], state.lambdas[1], state.lambdas[2])
    else:
        T = state.T_d
        ext = (state.lambdas[0], state.lambdas[1], state.lambdas[2])
        int_ = (state.lambdas[3], state.lambdas[4], state.lambdas[5])
    Sa = jnp.einsum("i,ijd,j->ijd", ext[0], state.R_a, int_[0])
    Sb = jnp.einsum("i,ijd,j->ijd", ext[1], state.R_b, int_[1])
    Sc = jnp.einsum("i,ijd,j->ijd", ext[2], state.R_c, int_[2])
    return jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, T)


def test_su_step_identity_gate_no_truncation_is_identity_up():
    """With identity gate and D_max >= D*d, the HOSVD-preserved quantity matches.

    The gauged ``theta`` (ext * R_old * int_old contracted with T_old) is
    bitwise identical (up to numerical tolerance) to the same quantity
    constructed from the new state with the OLD external lambdas and the OLD
    internal lambdas absorbed via the HOSVD identity ``S * core == theta``.
    Concretely: with R_new = lam_ext_inv * S, T_new = core, and lambdas[ext]
    untouched, we have ``(lam_ext * R_new) * T_new == S * core == theta``.
    """
    from tenax.algorithms.pess import pess_simple_update_triangle

    D, d = 2, 3
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
    gate = jnp.eye(d**3, dtype=jnp.complex128).reshape(d, d, d, d, d, d)
    new_state = pess_simple_update_triangle(state, gate, triangle="up", D_max=D * d)

    theta_orig = _build_theta_full(state, "up")

    # Contract the new state with ONLY external lambdas (no internal) to
    # recover S * core, which equals the original gauged theta.
    Sa = jnp.einsum("i,ijd->ijd", new_state.lambdas[3], new_state.R_a)
    Sb = jnp.einsum("i,ijd->ijd", new_state.lambdas[4], new_state.R_b)
    Sc = jnp.einsum("i,ijd->ijd", new_state.lambdas[5], new_state.R_c)
    theta_new = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, new_state.T_u)
    np.testing.assert_allclose(theta_orig, theta_new, atol=1e-10)

    # Other simplex (T_d) and external lambdas (3,4,5) must be untouched.
    np.testing.assert_array_equal(state.T_d, new_state.T_d)
    for i in (3, 4, 5):
        np.testing.assert_array_equal(state.lambdas[i], new_state.lambdas[i])


def test_su_step_identity_gate_no_truncation_is_identity_down():
    """Same as the 'up' check, but for the down triangle."""
    from tenax.algorithms.pess import pess_simple_update_triangle

    D, d = 2, 3
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(1))
    gate = jnp.eye(d**3, dtype=jnp.complex128).reshape(d, d, d, d, d, d)
    new_state = pess_simple_update_triangle(state, gate, triangle="down", D_max=D * d)

    theta_orig = _build_theta_full(state, "down")

    # For "down", external lambdas live at indices (0, 1, 2).
    Sa = jnp.einsum("i,ijd->ijd", new_state.lambdas[0], new_state.R_a)
    Sb = jnp.einsum("i,ijd->ijd", new_state.lambdas[1], new_state.R_b)
    Sc = jnp.einsum("i,ijd->ijd", new_state.lambdas[2], new_state.R_c)
    theta_new = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, new_state.T_d)
    np.testing.assert_allclose(theta_orig, theta_new, atol=1e-10)

    # Other simplex (T_u) and external lambdas (0,1,2) must be untouched.
    np.testing.assert_array_equal(state.T_u, new_state.T_u)
    for i in (0, 1, 2):
        np.testing.assert_array_equal(state.lambdas[i], new_state.lambdas[i])


def test_su_step_invalid_triangle_raises():
    from tenax.algorithms.pess import pess_simple_update_triangle

    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    gate = jnp.eye(27, dtype=jnp.complex128).reshape(3, 3, 3, 3, 3, 3)
    with pytest.raises(ValueError):
        pess_simple_update_triangle(state, gate, triangle="sideways", D_max=2)


# ---------------------------------------------------------------------------
# Test helpers for the full SU loop
# ---------------------------------------------------------------------------


def _local_triangle_energy(state, H_tri, triangle="up"):
    """Environment-free local triangle energy: <psi_tri | H_tri | psi_tri> / <psi_tri | psi_tri>.

    Used as a quick monotonicity check for the SU algorithm — it is NOT the
    true infinite-system energy, but is monotonically decreasing under SU on
    the same triangle.
    """
    if triangle == "up":
        T = state.T_u
        ext = (state.lambdas[3], state.lambdas[4], state.lambdas[5])
        int_ = (state.lambdas[0], state.lambdas[1], state.lambdas[2])
    else:
        T = state.T_d
        ext = (state.lambdas[0], state.lambdas[1], state.lambdas[2])
        int_ = (state.lambdas[3], state.lambdas[4], state.lambdas[5])
    Sa = jnp.einsum("i,ijd,j->ijd", ext[0], state.R_a, int_[0])
    Sb = jnp.einsum("i,ijd,j->ijd", ext[1], state.R_b, int_[1])
    Sc = jnp.einsum("i,ijd,j->ijd", ext[2], state.R_c, int_[2])
    psi_tri = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, T)

    d = state.R_a.shape[2]
    H_resh = jnp.asarray(H_tri).reshape(d, d, d, d, d, d).astype(jnp.complex128)
    psi_H_psi = jnp.einsum("xyzdfg,DFGdfg,xyzDFG->", psi_tri.conj(), H_resh, psi_tri)
    psi_psi = jnp.einsum("xyzdfg,xyzdfg->", psi_tri.conj(), psi_tri)
    return float(jnp.real(psi_H_psi / psi_psi))


def test_su_decreases_energy_d2():
    from tenax.algorithms.pess import pess_simple_update

    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(1))
    state1 = pess_simple_update(state0, H, dt_schedule=[(0.05, 100)], D_max=2)
    e0 = _local_triangle_energy(state0, H)
    e1 = _local_triangle_energy(state1, H)
    assert e1 < e0


# ---------------------------------------------------------------------------
# Honeycomb supersite coarse-graining
# ---------------------------------------------------------------------------


def test_supersite_shapes():
    from tenax.algorithms.pess import pess_to_honeycomb_supersites

    state = IPESSState.random(D=4, d=3, key=jax.random.PRNGKey(0))
    A_u, A_d = pess_to_honeycomb_supersites(state)
    assert A_u.shape == (3, 3, 3, 4, 4, 4)
    assert A_d.shape == (3, 3, 3, 4, 4, 4)


def test_supersite_grad_flows():
    from tenax.algorithms.pess import pess_to_honeycomb_supersites

    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))

    def loss(s):
        A_u, A_d = pess_to_honeycomb_supersites(s)
        return jnp.real(jnp.vdot(A_u.ravel(), A_u.ravel())) + jnp.real(
            jnp.vdot(A_d.ravel(), A_d.ravel())
        )

    g = jax.grad(loss)(state)
    # All five primitive tensors should receive a finite gradient.
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u, g.T_d):
        assert jnp.all(jnp.isfinite(arr))
        assert jnp.linalg.norm(arr) > 0
        assert arr.dtype == jnp.complex128
