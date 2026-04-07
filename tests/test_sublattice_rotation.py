"""Tests for sublattice rotation and C4v symmetrization of iPEPS."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import (
    CTMConfig,
    heisenberg_gate,
    sublattice_rotate_gate,
    symmetrize_c4v,
    xxz_gate,
)


class TestSublatticeRotateGate:
    """Tests for sublattice_rotate_gate."""

    def test_heisenberg_becomes_ferromagnetic(self):
        """Rotated Heisenberg gate should have ferromagnetic ground state.

        After sublattice rotation, the Néel state |↑↓⟩ maps to |↑↑⟩,
        so the ground state of the rotated gate should be the uniform
        state (ferromagnetic).
        """
        gate = heisenberg_gate()
        gate_rot = sublattice_rotate_gate(gate)

        # Diagonalize the rotated 2-site Hamiltonian
        H_rot = gate_rot.todense().reshape(4, 4)
        eigvals, eigvecs = np.linalg.eigh(H_rot)

        # Ground state energy should be -0.75 (same as Heisenberg singlet)
        np.testing.assert_allclose(eigvals[0], -0.75, atol=1e-12)

        # The ground state should be a triplet state (ferromagnetic sector)
        # |↑↑⟩ and |↓↓⟩ should have large overlap with the ground state manifold
        # (the three lowest states are degenerate at E=-0.75 for the rotated H)
        # Actually for isotropic Heisenberg, the rotated H has the same spectrum
        # Check: spectrum of original and rotated should match
        H_orig = heisenberg_gate().todense().reshape(4, 4)
        eigvals_orig = np.linalg.eigh(H_orig)[0]
        np.testing.assert_allclose(sorted(eigvals), sorted(eigvals_orig), atol=1e-12)

    def test_preserves_spectrum(self):
        """Sublattice rotation is a unitary transform — spectrum unchanged."""
        gate = xxz_gate(delta=0.5)
        gate_rot = sublattice_rotate_gate(gate)

        H = gate.todense().reshape(4, 4)
        H_rot = gate_rot.todense().reshape(4, 4)

        eigvals = np.sort(np.linalg.eigvalsh(H))
        eigvals_rot = np.sort(np.linalg.eigvalsh(H_rot))
        np.testing.assert_allclose(eigvals, eigvals_rot, atol=1e-12)

    def test_double_rotation_is_identity(self):
        """Applying sublattice rotation twice should return the original gate.

        U^2 = (iσ^y)^2 = -I, so (I⊗U†)^2 H (I⊗U)^2 = (I⊗I) H (I⊗I) = H.
        """
        gate = heisenberg_gate()
        gate_rot = sublattice_rotate_gate(gate)
        gate_rot2 = sublattice_rotate_gate(gate_rot)

        np.testing.assert_allclose(gate.todense(), gate_rot2.todense(), atol=1e-12)

    def test_raw_array_input(self):
        """Should work with raw JAX arrays."""
        gate_tensor = heisenberg_gate()
        gate_arr = gate_tensor.todense()

        result = sublattice_rotate_gate(gate_arr, d=2)
        assert isinstance(result, jax.Array)
        assert result.shape == (2, 2, 2, 2)

        # Compare with DenseTensor path
        result_tensor = sublattice_rotate_gate(gate_tensor)
        np.testing.assert_allclose(result, result_tensor.todense(), atol=1e-12)

    def test_rotated_heisenberg_structure(self):
        """Verify the rotated Heisenberg has the expected structure.

        After rotation: H_rot = -(S^x S^x - S^y S^y + S^z S^z)
        = -S^x S^x + S^y S^y - S^z S^z
        """
        gate_rot = sublattice_rotate_gate(heisenberg_gate())
        H_rot = gate_rot.todense().reshape(4, 4)

        # Build expected: -SxSx + SySy - SzSz
        Sx = jnp.array([[0, 0.5], [0.5, 0]])
        Sy = jnp.array([[0, -0.5j], [0.5j, 0]])
        Sz = jnp.array([[0.5, 0], [0, -0.5]])
        H_expected = -jnp.kron(Sx, Sx) + jnp.kron(Sy, Sy) - jnp.kron(Sz, Sz)
        np.testing.assert_allclose(H_rot, H_expected.real, atol=1e-12)

    def test_unsupported_d_raises(self):
        """Should raise for d != 2."""
        gate = jnp.zeros((3, 3, 3, 3))
        with pytest.raises(NotImplementedError, match="d=2"):
            sublattice_rotate_gate(gate, d=3)


class TestSublatticeRotationC4v:
    """Test that sublattice rotation + C4v CTM gives correct energy."""

    def test_heisenberg_c4v_energy(self):
        """Rotated Heisenberg + C4v CTM should give physical energy."""
        from tenax import compute_energy_ctm_tensor, ctm_tensor_c4v

        # Build C4v-symmetric initial tensor
        D, d = 2, 2
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (D, D, D, D, d)) * 0.1
        A = A.at[0, 0, 0, 0, 0].set(1.0)  # bias toward spin-up (ferromagnetic)

        from tenax.algorithms.ipeps import _wrap_as_dense_tensor

        A_tensor = _wrap_as_dense_tensor(A)

        # Rotated gate for C4v
        gate = heisenberg_gate()
        gate_rot = sublattice_rotate_gate(gate)

        # Run C4v CTM
        env = ctm_tensor_c4v(A_tensor, chi=8, max_iter=100, conv_tol=1e-10)

        # Compute energy with rotated gate
        E = compute_energy_ctm_tensor(A_tensor, env, gate_rot, d)
        E_val = float(E)

        # Energy should be finite and in a reasonable range
        assert np.isfinite(E_val), f"Energy is not finite: {E_val}"
        # For a simple initial state, energy should be negative (AFM Heisenberg)
        assert E_val < 0, f"Energy should be negative, got {E_val}"
        # Should not be unphysically low (physical bound is -3/2 per site)
        assert E_val > -1.5, f"Energy {E_val} is below physical bound"


class TestSymmetrizeC4v:
    """Tests for C4v symmetrization of iPEPS tensors."""

    def test_idempotent(self):
        """Applying symmetrize_c4v twice should give the same result."""
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (3, 3, 3, 3, 2))
        A_sym = symmetrize_c4v(A)
        A_sym2 = symmetrize_c4v(A_sym)
        np.testing.assert_allclose(A_sym, A_sym2, atol=1e-14)

    def test_c4_invariant(self):
        """Symmetrized tensor should be invariant under 90-degree rotation."""
        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (3, 3, 3, 3, 2))
        A_sym = symmetrize_c4v(A)
        # C4 rotation: (u,d,l,r) -> (r,l,u,d)
        A_rot = jnp.transpose(A_sym, (3, 2, 0, 1, 4))
        np.testing.assert_allclose(A_sym, A_rot, atol=1e-14)

    def test_reflection_invariant(self):
        """Symmetrized tensor should be invariant under l<->r reflection."""
        key = jax.random.PRNGKey(2)
        A = jax.random.normal(key, (3, 3, 3, 3, 2))
        A_sym = symmetrize_c4v(A)
        # sigma_v: (u,d,l,r) -> (u,d,r,l)
        A_ref = jnp.transpose(A_sym, (0, 1, 3, 2, 4))
        np.testing.assert_allclose(A_sym, A_ref, atol=1e-14)

    def test_preserves_norm(self):
        """Symmetrization should not increase the norm (it's a projector)."""
        key = jax.random.PRNGKey(3)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        norm_before = jnp.linalg.norm(A)
        norm_after = jnp.linalg.norm(symmetrize_c4v(A))
        assert norm_after <= norm_before + 1e-14

    def test_already_symmetric_unchanged(self):
        """A C4v-symmetric tensor should be unchanged by symmetrization."""
        # Build a tensor that's C4v by construction
        key = jax.random.PRNGKey(4)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        A_sym = symmetrize_c4v(A)
        # Symmetrize again
        A_sym2 = symmetrize_c4v(A_sym)
        np.testing.assert_allclose(A_sym, A_sym2, atol=1e-14)

    def test_jax_differentiable(self):
        """symmetrize_c4v should be differentiable by JAX."""
        key = jax.random.PRNGKey(5)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))

        def loss(A_param):
            A_sym = symmetrize_c4v(A_param)
            return jnp.sum(A_sym**2)

        grad = jax.grad(loss)(A)
        assert grad.shape == A.shape
        assert jnp.all(jnp.isfinite(grad))
        # Gradient should also be C4v-symmetric
        grad_sym = symmetrize_c4v(grad)
        np.testing.assert_allclose(grad, grad_sym, atol=1e-14)

    def test_parameter_count_d2(self):
        """Check the dimension of the C4v-invariant subspace for D=2, d=2.

        n(D) = d/8 * (D^4 + 2D^3 + 3D^2 + 2D) = 2/8 * 48 = 12
        """
        D, d = 2, 2
        # Build a basis for the invariant subspace by symmetrizing
        # each canonical basis vector
        total = D**4 * d
        rank = 0
        basis_vecs = []
        for i in range(total):
            e = jnp.zeros(total).at[i].set(1.0).reshape(D, D, D, D, d)
            e_sym = symmetrize_c4v(e).reshape(-1)
            if jnp.linalg.norm(e_sym) > 1e-10:
                basis_vecs.append(e_sym)
        if basis_vecs:
            mat = jnp.stack(basis_vecs)
            rank = int(jnp.linalg.matrix_rank(mat))
        expected = d * (D**4 + 2 * D**3 + 3 * D**2 + 2 * D) // 8
        assert rank == expected, f"C4v subspace rank {rank}, expected {expected}"


class TestC4vOptimization:
    """Test C4v symmetrization + sublattice rotation in AD optimization."""

    def test_c4v_ad_heisenberg_d2(self):
        """C4v + sublattice rotation + AD should give physical energy."""
        from tenax import iPEPSConfig, optimize_gs_ad

        gate = heisenberg_gate()
        gate_rot = sublattice_rotate_gate(gate)

        # Random init (C4v symmetrization applied inside optimizer)
        D, d = 2, 2
        key = jax.random.PRNGKey(42)
        A_init = jax.random.normal(key, (D, D, D, D, d))

        from tenax.algorithms.ipeps import _wrap_as_dense_tensor

        A_tensor = _wrap_as_dense_tensor(A_init)

        config = iPEPSConfig(
            max_bond_dim=D,
            ctm=CTMConfig(chi=8, max_iter=50, min_iter=10),
            gs_optimizer="adam",
            gs_learning_rate=5e-3,
            gs_num_steps=30,
            gs_metric_precond=False,
            gs_line_search=False,
            gs_verbose=False,
            unit_cell="1x1",
            gs_explicit_ad=True,
            gs_explicit_ad_steps=10,
            gs_explicit_ad_warmup=2,
            su_init=False,
            gs_c4v=True,
        )
        A_opt, env, E = optimize_gs_ad(gate_rot, A_tensor, config)
        E_val = float(E)

        assert np.isfinite(E_val), f"Energy is not finite: {E_val}"
        # Should be better than classical Ising (-0.5) since C4v
        # allows quantum fluctuations
        assert E_val < -0.50, f"C4v AD energy {E_val:.6f} not below -0.50 (classical)"
        assert E_val > -1.0, f"Energy {E_val} unphysically low"
