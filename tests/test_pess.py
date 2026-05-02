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


def test_triangle_hamiltonian_ed_spectrum_spin_half():
    """ED spectrum of the spin-1/2 isotropic triangle.

    For SU(2) Heisenberg on a triangle ``H = J Σ_<ij> S_i·S_j``, the
    operator identity ``S_tot² = Σ S_i² + 2 Σ S_i·S_j`` gives
    ``H = (S_tot² - 9/4) / 2``. The spin-1/2 triangle decomposes into
    two ``S=1/2`` doublets (4 states at ``E=-3/4``) and one ``S=3/2``
    quartet (4 states at ``E=+3/4``). Verifying the full spectrum
    catches any sign or normalization regression in
    :func:`kagome_triangle_xxz_hamiltonian`.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    evals = np.linalg.eigvalsh(H)
    expected = np.array([-0.75] * 4 + [0.75] * 4)
    np.testing.assert_allclose(evals, expected, atol=1e-12)


def test_triangle_hamiltonian_ed_spectrum_spin_one():
    """ED spectrum of the spin-1 isotropic triangle.

    Spin-1 Heisenberg triangle: ``H = (S_tot² - 6) / 2``. Total spin
    decomposes as ``1 ⊗ 1 ⊗ 1 = 0 ⊕ 1⊕3 ⊕ 2⊕2 ⊕ 3``, giving
    ``S_tot ∈ {0 (×1), 1 (×3), 2 (×2), 3 (×1)}`` with multiplicity
    ``(2S+1)`` each — total ``1 + 3·3 + 2·5 + 1·7 = 27 = 3³``. The
    eigenvalues ``(S_tot² - 6) / 2 = (S(S+1) - 6) / 2`` are
    ``{-3, -2, 0, 3}`` with degeneracies ``{1, 9, 10, 7}``.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    evals = np.linalg.eigvalsh(H)
    expected = np.concatenate(
        [
            np.full(1, -3.0),
            np.full(9, -2.0),
            np.full(10, 0.0),
            np.full(7, 3.0),
        ]
    )
    np.testing.assert_allclose(evals, expected, atol=1e-10)


def test_triangle_hamiltonian_ed_xxz_easy_axis_spin_half():
    """ED ground state in the Ising-anisotropic limit (XXZ Δ=2, spin-1/2).

    At Δ=2 the SzSz term dominates the in-plane coupling. The
    ground states are still in the ``S^z_total = ±1/2`` sector
    (frustrated Néel-like configurations like ``|↑↑↓⟩`` with a
    ferromagnetic-bond penalty). Easy-axis ED gives
    ``E_gs = (1/4)·Δ·(-1) + 0.5·(spin-flip term...)`` which evaluates
    numerically to the value below; matching it to 1e-12 is a strong
    catch for any wiring regression on the XXZ anisotropy parameter.
    """
    H = kagome_triangle_xxz_hamiltonian(delta=2.0, d=2)
    e_gs = float(np.linalg.eigvalsh(H)[0])
    # Closed form: at Δ=2, H_tri eigenstates split into the same Stot
    # sectors as the isotropic case but with shifted SzSz weight.
    # Numerical reference recomputed from the same kron construction.
    Sz = np.array([[0.5, 0], [0, -0.5]])
    Sp = np.array([[0, 1.0], [0, 0]])
    Sm = np.array([[0, 0], [1.0, 0]])
    I2 = np.eye(2)

    def _bond(A, B, C):
        return 2.0 * np.kron(np.kron(A, B), C) + 0.5 * (
            np.kron(np.kron(Sp, Sm), C) + np.kron(np.kron(Sm, Sp), C)
        )

    # H_12, H_23, H_31 with delta=2 each.
    h12 = 2.0 * np.kron(np.kron(Sz, Sz), I2) + 0.5 * (
        np.kron(np.kron(Sp, Sm), I2) + np.kron(np.kron(Sm, Sp), I2)
    )
    h23 = 2.0 * np.kron(I2, np.kron(Sz, Sz)) + 0.5 * (
        np.kron(I2, np.kron(Sp, Sm)) + np.kron(I2, np.kron(Sm, Sp))
    )
    h31 = 2.0 * np.kron(Sz, np.kron(I2, Sz)) + 0.5 * (
        np.kron(Sp, np.kron(I2, Sm)) + np.kron(Sm, np.kron(I2, Sp))
    )
    H_ref = h12 + h23 + h31
    e_ref = float(np.linalg.eigvalsh(H_ref)[0])
    assert abs(e_gs - e_ref) < 1e-12, (
        f"ED ground state mismatch vs explicit kron reference: "
        f"{e_gs:.10f} vs {e_ref:.10f}"
    )


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
    """If D_max >= input dim, theta should round-trip when the explicit
    lambda gauges are folded back in. The bond spectra are factored out
    of ``core`` and live in the returned ``lams``; recombining them
    along each mode reproduces ``theta`` exactly.
    """
    from tenax.algorithms.pess import hosvd_truncate

    D, d = 3, 3
    key = jax.random.PRNGKey(7)
    theta = (
        jax.random.normal(key, (D, D, D, d, d, d))
        + 1j * jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d, d, d))
    ).astype(jnp.complex128)
    S_a, S_b, S_c, core, lams = hosvd_truncate(theta, D_max=D * d, d=d)
    # Reconstruct theta with the explicit bond-gauge lambdas inserted on
    # each internal bond:
    # theta[a,b,c,p_a,p_b,p_c]
    #   = sum_{i,j,k} S_a[a,i,p_a] lam_a[i] S_b[b,j,p_b] lam_b[j] S_c[c,k,p_c] lam_c[k] core[i,j,k]
    theta_reco = jnp.einsum(
        "aip,i,bjq,j,ckr,k,ijk->abcpqr",
        S_a,
        lams[0].astype(theta.dtype),
        S_b,
        lams[1].astype(theta.dtype),
        S_c,
        lams[2].astype(theta.dtype),
        core,
    )
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
    # is closer to theta than the zero tensor would be. Lambdas live in
    # the explicit ``lams`` tuple, so they must be folded back in for the
    # reconstruction to match the un-truncated theta.
    theta_reco = jnp.einsum(
        "aip,i,bjq,j,ckr,k,ijk->abcpqr",
        S_a,
        lams[0].astype(theta.dtype),
        S_b,
        lams[1].astype(theta.dtype),
        S_c,
        lams[2].astype(theta.dtype),
        core,
    )
    err = jnp.linalg.norm(theta - theta_reco)
    norm_theta = jnp.linalg.norm(theta)
    assert float(err) < float(norm_theta)


def _build_theta_full(state, triangle):
    """Reference: build the gauged theta tensor from an IPESSState.

    Geometrically correct gauging per ``IPESSState`` convention
    (axis 0 of R = T_d-leg, axis 1 = T_u-leg):

    - ``"up"``: ext (down-bonds) live on axis 0 of R, int (up-bonds) on
      axis 1. T_u contracts axis 1 of R.
    - ``"down"``: ext (up-bonds) live on axis 1 of R, int (down-bonds)
      on axis 0. T_d contracts axis 0 of R.

    The output theta has shape ``(D_ext_a, D_ext_b, D_ext_c, d, d, d)``
    where ``D_ext`` is the size of the leg the ext lambdas live on.
    """
    if triangle == "up":
        T = state.T_u
        ext = (state.lambdas[3], state.lambdas[4], state.lambdas[5])  # T_d-leg = axis 0
        int_ = (
            state.lambdas[0],
            state.lambdas[1],
            state.lambdas[2],
        )  # T_u-leg = axis 1
        # ext on axis 0 (i), int on axis 1 (j); T_u contracts axis 1.
        Sa = jnp.einsum("i,ijd,j->ijd", ext[0], state.R_a, int_[0])
        Sb = jnp.einsum("i,ijd,j->ijd", ext[1], state.R_b, int_[1])
        Sc = jnp.einsum("i,ijd,j->ijd", ext[2], state.R_c, int_[2])
        return jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, T)
    T = state.T_d
    ext = (state.lambdas[0], state.lambdas[1], state.lambdas[2])  # T_u-leg = axis 1
    int_ = (state.lambdas[3], state.lambdas[4], state.lambdas[5])  # T_d-leg = axis 0
    # int on axis 0 (i), ext on axis 1 (j); T_d contracts axis 0.
    Sa = jnp.einsum("i,ijd,j->ijd", int_[0], state.R_a, ext[0])
    Sb = jnp.einsum("i,ijd,j->ijd", int_[1], state.R_b, ext[1])
    Sc = jnp.einsum("i,ijd,j->ijd", int_[2], state.R_c, ext[2])
    return jnp.einsum("axd,byf,czg,abc->xyzdfg", Sa, Sb, Sc, T)


def test_su_step_identity_gate_no_truncation_is_identity_up():
    """With identity gate and D_max >= D*d, the gauged ``theta`` is preserved
    exactly when both external AND internal lambdas are folded back into the
    R sites.

    The bond spectra are factored OUT of ``T_new`` (= core) and live in the
    explicit internal lambdas, so the reconstruction
    ``(lam_ext * R_new * lam_int) * T_new`` exactly equals
    ``(lam_ext * R_old * lam_int_old) * T_old``.
    """
    from tenax.algorithms.pess import pess_simple_update_triangle

    D, d = 2, 3
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
    gate = jnp.eye(d**3, dtype=jnp.complex128).reshape(d, d, d, d, d, d)
    new_state = pess_simple_update_triangle(state, gate, triangle="up", D_max=D * d)

    theta_orig = _build_theta_full(state, "up")

    Sa = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[3], new_state.R_a, new_state.lambdas[0]
    )
    Sb = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[4], new_state.R_b, new_state.lambdas[1]
    )
    Sc = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[5], new_state.R_c, new_state.lambdas[2]
    )
    theta_new = jnp.einsum("xad,ybf,zcg,abc->xyzdfg", Sa, Sb, Sc, new_state.T_u)
    np.testing.assert_allclose(theta_orig, theta_new, atol=1e-10)

    # Other simplex (T_d) and external lambdas (3,4,5) must be untouched.
    np.testing.assert_array_equal(state.T_d, new_state.T_d)
    for i in (3, 4, 5):
        np.testing.assert_array_equal(state.lambdas[i], new_state.lambdas[i])


def test_su_step_identity_gate_no_truncation_is_identity_down():
    """Same as the 'up' check, but for the down triangle.

    Geometric reconstruction: int (down-bonds, indices 3-5) on axis 0
    (T_d-leg), ext (up-bonds, indices 0-2) on axis 1 (T_u-leg); T_d
    contracts axis 0 of R.
    """
    from tenax.algorithms.pess import pess_simple_update_triangle

    D, d = 2, 3
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(1))
    gate = jnp.eye(d**3, dtype=jnp.complex128).reshape(d, d, d, d, d, d)
    new_state = pess_simple_update_triangle(state, gate, triangle="down", D_max=D * d)

    theta_orig = _build_theta_full(state, "down")

    Sa = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[3], new_state.R_a, new_state.lambdas[0]
    )
    Sb = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[4], new_state.R_b, new_state.lambdas[1]
    )
    Sc = jnp.einsum(
        "i,ijd,j->ijd", new_state.lambdas[5], new_state.R_c, new_state.lambdas[2]
    )
    theta_new = jnp.einsum("axd,byf,czg,abc->xyzdfg", Sa, Sb, Sc, new_state.T_d)
    np.testing.assert_allclose(theta_orig, theta_new, atol=1e-10)

    # Other simplex (T_u) and external lambdas (0,1,2) must be untouched.
    np.testing.assert_array_equal(state.T_u, new_state.T_u)
    for i in (0, 1, 2):
        np.testing.assert_array_equal(state.lambdas[i], new_state.lambdas[i])


def test_su_step_identity_gate_is_stationary_after_one_step():
    """Identity-gate SU must be a fixed point AFTER one step.

    With the bond spectra factored out of ``core``, applying SU again with
    an identity gate must reproduce the same lambdas (steady state).

    Regression test: the original ``hosvd_truncate`` left the bond spectra
    inside ``core`` while ALSO returning explicit normalized lambdas. The
    next SU step then re-applied those lambdas as gauges, squaring the
    relative bond spectrum every iteration. After ~10 steps every lambda
    collapsed to a rank-1 distribution, pinning SU at the classical 120°
    energy on spin-½ kagome (the bug behind PR #387's stalled E = -0.25).
    """
    from tenax.algorithms.pess import pess_simple_update_triangle

    D, d = 4, 2
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
    gate_id = jnp.eye(d**3, dtype=jnp.complex128).reshape(d, d, d, d, d, d)

    state1 = pess_simple_update_triangle(state, gate_id, "up", D_max=D)
    state2 = pess_simple_update_triangle(state1, gate_id, "up", D_max=D)
    state3 = pess_simple_update_triangle(state2, gate_id, "up", D_max=D)

    # After the first identity step the lambdas reflect the random
    # initial state's HOSVD spectrum. Subsequent identity steps must NOT
    # change them (no spectrum squaring).
    for i in range(3):
        np.testing.assert_allclose(state2.lambdas[i], state1.lambdas[i], atol=1e-10)
        np.testing.assert_allclose(state3.lambdas[i], state1.lambdas[i], atol=1e-10)

    # Same stability test on the down triangle.
    state1d = pess_simple_update_triangle(state, gate_id, "down", D_max=D)
    state2d = pess_simple_update_triangle(state1d, gate_id, "down", D_max=D)
    for i in range(3, 6):
        np.testing.assert_allclose(state2d.lambdas[i], state1d.lambdas[i], atol=1e-10)


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

    Quick monotonicity check for SU. Reuses :func:`_build_theta_full` for
    the geometric gauging convention (ext on the leg away from the
    simplex being considered, int on the leg toward it; T contracted on
    the int-side leg of R).
    """
    psi_tri = _build_theta_full(state, triangle)
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


def test_su_isotropic_heisenberg_evolves_both_triangles():
    """For isotropic Heisenberg on kagome (where H_up = H_dn), SU must
    evolve up and down triangles symmetrically. After a moderate
    schedule the two local triangle energies must agree to a few
    percent.

    Regression: the original ``pess_simple_update_triangle`` always
    contracted the simplex via R's axis 1 regardless of triangle, so
    the down-triangle gate was tied to the T_u-leg of R rather than
    the T_d-leg. Result: only the up-triangle actually converged
    under SU; the down-triangle stayed near its random initial energy
    (E_dn ≈ 0 while E_up → -0.75 at D=4 spin-½). The fix transposes R
    around the SU body so T_d contracts axis 0.
    """
    from tenax.algorithms.pess import pess_simple_update

    D, d = 4, 2
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=d)
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(1))
    state = pess_simple_update(state, H, dt_schedule=[(0.1, 100), (0.01, 200)], D_max=D)
    e_up = _local_triangle_energy(state, H, "up")
    e_dn = _local_triangle_energy(state, H, "down")
    assert abs(e_up - e_dn) < 0.01, (
        f"Up/down asymmetry: E_up={e_up:.4f}, E_dn={e_dn:.4f} "
        f"(diff={abs(e_up - e_dn):.4f}). Pre-fix this gap was ~0.7."
    )
    # Also: per-site env-free energy must be well below classical 120° (-0.25).
    e_per_site = (e_up + e_dn) / 3.0
    assert e_per_site < -0.30, (
        f"Per-site E={e_per_site:.4f} not below classical -0.25 "
        "— SU may have collapsed to product state again."
    )


# ---------------------------------------------------------------------------
# Square coarse-grained supersite (Convention C)
# ---------------------------------------------------------------------------


def test_supersite_shape_and_dummy_bond():
    from tenax.algorithms.pess import pess_to_kagome_supersite

    state = IPESSState.random(D=4, d=3, key=jax.random.PRNGKey(0))
    A = pess_to_kagome_supersite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.lambdas
    )
    assert A.shape == (4, 4, 4, 4, 27)
    # Convention C dummy bond: only the [:, :, :, 0, :] slice is non-zero.
    assert jnp.all(A[:, :, :, 1:, :] == 0)


def test_supersite_grad_flows():
    from tenax.algorithms.pess import pess_to_kagome_supersite

    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))

    def loss(s):
        A = pess_to_kagome_supersite(s.R_a, s.R_b, s.R_c, s.T_u, s.lambdas)
        return jnp.real(jnp.vdot(A.ravel(), A.ravel()))

    g = jax.grad(loss)(state)
    # Optimization variables must receive a finite, non-zero gradient.
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u):
        assert jnp.all(jnp.isfinite(arr))
        assert jnp.linalg.norm(arr) > 0
        assert arr.dtype == jnp.complex128
    for lam_g in g.lambdas:
        assert jnp.all(jnp.isfinite(lam_g))
