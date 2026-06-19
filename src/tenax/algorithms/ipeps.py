"""Infinite Projected Entangled Pair States (iPEPS) algorithm.

iPEPS is a variational ansatz for 2D quantum lattice models. The state is
represented as a PEPS (Projected Entangled Pair States) tensor network where
each site has a local tensor A[u,d,l,r,s] (up,down,left,right,physical).

For infinite systems, we use a unit cell (typically 1x1 for translationally
invariant states) and compute observables using the Corner Transfer Matrix (CTM)
method to approximate the infinite environment.

This module implements:
1. Simple update: fast imaginary time evolution optimization
2. CTM algorithm: environment computation for expectation values
3. Energy evaluation using CTM environment

Reference:
- Corboz et al., PRB 81, 165104 (2010) (CTM)
- Jiang et al., PRB 78, 134432 (2008) (simple update)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms.ipeps_config import CTMEnvironment, iPEPSConfig
from tenax.algorithms.ipeps_ctm_convergence import ctm_2site
from tenax.algorithms.ipeps_rdm import compute_energy_ctm_2site
from tenax.algorithms.ipeps_simple_update import (
    _make_trotter_gate_tensor,
    _simple_update_2site_horizontal_tensor,
    _simple_update_2site_vertical_tensor,
)
from tenax.core import EPS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor


def heisenberg_gate(dtype=jnp.float64) -> DenseTensor:
    """Build the 2-site Heisenberg Hamiltonian as a DenseTensor.

    ``H = Sz Sz + 0.5 (S+ S- + S- S+)`` on two spin-1/2 sites, returned
    as a 4-leg tensor with labels ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)


def heisenberg_gate_u1sz(dtype=jnp.float64) -> SymmetricTensor:
    """Build the 2-site Heisenberg Hamiltonian as a U(1)-Sz SymmetricTensor.

    Identical numerics to :func:`heisenberg_gate`
    (``H = Sz Sz + 0.5 (S+ S- + S- S+)``) but the physical legs carry
    U(1) charges ``[+1, -1]`` for ``{up, down}`` (units of ``2*Sz``,
    matching the ``S+``/``S-`` charge-(+/-)2 convention in
    ``tests/test_observables.py``). Sz conservation makes the gate
    block-sparse. Returned as a 4-leg ``SymmetricTensor`` with labels
    ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.array([1, -1], dtype=np.int32)  # Sz = +1/2, -1/2 -> 2*Sz
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return SymmetricTensor.from_dense(H.reshape(2, 2, 2, 2), indices)


def heisenberg_u1sz_init_pair(
    D: int, key: jax.Array
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Build a random U(1)-Sz-symmetric 2-site iPEPS pair ``(A, B)``.

    Each site tensor has 5 legs ``(u, d, l, r, phys)`` with flows
    ``u=OUT, d=IN, l=OUT, r=IN, phys=IN`` (matching
    ``_build_initial_fpeps_tensor``). Physical charges are ``[+1, -1]``
    (2*Sz, where Sz=+1/2 maps to +1 and Sz=-1/2 maps to -1); virtual
    charges cycle through ``[0, +1, -1, 0, +1, -1, ...]`` over bond
    dimension ``D`` so that the Sz-conservation law
    ``-q_u + q_d - q_l + q_r + q_phys = 0`` has non-trivial solutions
    (pure ``±1`` virtual charges produce a parity obstruction: all
    virtual contributions are even while physical charges are odd,
    giving zero valid sectors). Both tensors are Sz-conserving; the AFM
    correlations emerge from optimization within the Sz=0 sector.

    Args:
        D:   Virtual bond dimension. Must be ``>= 2``: at D=1 the only
             virtual charge is 0 and no Sz-conserving sector with a charged
             physical leg exists (the tensor would be all-zeros).
        key: JAX random key (split internally for A and B).

    Returns:
        Tuple ``(A, B)`` of SymmetricTensors.

    Raises:
        ValueError: if ``D < 2``.
    """
    if D < 2:
        raise ValueError(f"D must be >= 2 for non-trivial Sz blocks; got D={D}")
    sym = U1Symmetry()
    pattern = [0, 1, -1]
    virt_charges = np.array([pattern[i % 3] for i in range(D)], dtype=np.int32)
    phys_charges = np.array([1, -1], dtype=np.int32)

    indices = (
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, virt_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )

    kA, kB = jax.random.split(key)
    A = SymmetricTensor.random_normal(indices, kA)
    B = SymmetricTensor.random_normal(indices, kB)
    return A, B


def xxz_gate(delta: float = 1.0, dtype=jnp.float64) -> DenseTensor:
    """Build the 2-site XXZ Hamiltonian as a DenseTensor.

    ``H = delta * Sz Sz + 0.5 (S+ S- + S- S+)`` on two spin-1/2 sites.

    Args:
        delta: Anisotropy parameter. delta=1 is isotropic Heisenberg,
               delta=0 is XX model, delta->inf is Ising limit.
        dtype: Array dtype.

    Returns:
        4-leg DenseTensor with labels ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = delta * jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.zeros(2, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return DenseTensor(H.reshape(2, 2, 2, 2), indices)


def symmetrize_c4v(A: jax.Array) -> jax.Array:
    """Project an iPEPS tensor onto the C₄ᵥ-invariant subspace.

    Averages over the 8 elements of the C₄ᵥ point group (4 rotations
    + 4 reflections) acting on the virtual indices ``(u, d, l, r)``.
    The physical index is left unchanged.

    This is the standard symmetrization projector::

        A_sym = (1/8) Σ_{g ∈ C₄ᵥ} g(A)

    When used inside an AD loss function, JAX differentiates through
    the linear projection, effectively constraining the gradient to
    the symmetric subspace.

    Args:
        A: iPEPS site tensor of shape ``(D, D, D, D, d)`` with
           legs ``(u, d, l, r, phys)``.

    Returns:
        Symmetrized tensor of the same shape.

    Reference: Corboz, PRB 94, 035133 (2016), Sec. II.
    """
    # C₄ rotations: E, C₄, C₂, C₄⁻¹
    a0 = A  # E:    (u, d, l, r, s)
    a1 = jnp.transpose(A, (3, 2, 0, 1, 4))  # C₄:   (r, l, u, d, s)
    a2 = jnp.transpose(A, (1, 0, 3, 2, 4))  # C₂:   (d, u, r, l, s)
    a3 = jnp.transpose(A, (2, 3, 1, 0, 4))  # C₄⁻¹: (l, r, d, u, s)
    # Reflections: σ_v, σ_h, σ_d, σ_d'
    a4 = jnp.transpose(A, (0, 1, 3, 2, 4))  # σ_v:  (u, d, r, l, s)
    a5 = jnp.transpose(A, (1, 0, 2, 3, 4))  # σ_h:  (d, u, l, r, s)
    a6 = jnp.transpose(A, (3, 2, 1, 0, 4))  # σ_d:  (r, l, d, u, s)
    a7 = jnp.transpose(A, (2, 3, 0, 1, 4))  # σ_d': (l, r, u, d, s)
    return (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7) / 8.0


def build_c4v_basis(D: int, d: int = 2) -> np.ndarray:
    """Build orthonormal basis for the C₄ᵥ-invariant subspace.

    Constructs basis by symmetrizing each canonical basis vector of
    ℝ^{dD⁴} and orthonormalizing via QR decomposition.

    Called once at optimizer initialization (not JIT-traced).

    Args:
        D: Bond dimension.
        d: Physical dimension (default 2).

    Returns:
        NumPy array of shape ``(n, D⁴d)`` where
        ``n = d(D⁴ + 2D³ + 3D² + 2D) / 8``.
    """
    total = D**4 * d
    shape = (D, D, D, D, d)
    sym_vecs = []
    for i in range(total):
        e = np.zeros(total)
        e[i] = 1.0
        e_sym = np.asarray(symmetrize_c4v(jnp.array(e.reshape(shape)))).reshape(-1)
        if np.linalg.norm(e_sym) > 1e-14:
            sym_vecs.append(e_sym)
    if not sym_vecs:
        return np.zeros((0, total))
    mat = np.stack(sym_vecs, axis=0)  # (k, total)
    # SVD to get orthonormal basis for the column space (= C4v subspace)
    U, S, _ = np.linalg.svd(mat.T, full_matrices=False)
    rank = np.sum(S > 1e-12)
    basis = U[:, :rank].T  # (rank, total)
    return basis


def c4v_coeffs_from_tensor(A: jax.Array, basis: jax.Array | np.ndarray) -> jax.Array:
    """Project an iPEPS tensor onto the C₄ᵥ basis.

    Args:
        A: Tensor of shape ``(D, D, D, D, d)``.
        basis: Orthonormal basis of shape ``(n, D⁴d)`` from
            :func:`build_c4v_basis`.

    Returns:
        Coefficient vector of shape ``(n,)``.
    """
    return jnp.array(basis) @ A.reshape(-1)


def c4v_tensor_from_coeffs(
    coeffs: jax.Array,
    basis: jax.Array | np.ndarray,
    shape: tuple[int, ...],
) -> jax.Array:
    """Reconstruct a C₄ᵥ-symmetric tensor from coefficients.

    Args:
        coeffs: Coefficient vector of shape ``(n,)``.
        basis: Orthonormal basis of shape ``(n, D⁴d)``.
        shape: Target tensor shape, e.g. ``(D, D, D, D, d)``.

    Returns:
        Tensor of shape *shape*, guaranteed C₄ᵥ-symmetric.
    """
    return (jnp.array(basis).T @ coeffs).reshape(shape)


def sublattice_rotate_gate(gate: Tensor | jax.Array, d: int = 2) -> Tensor | jax.Array:
    """Apply sublattice rotation to a 2-site Hamiltonian gate.

    Performs ``H → (I ⊗ U†) H (I ⊗ U)`` where ``U = e^{iπσ^y/2}``
    is the spin-π rotation around the y-axis on the second site.
    This maps an antiferromagnetic Hamiltonian to a ferromagnetic one,
    so the Néel ground state becomes a uniform state representable by
    a single C₄ᵥ-symmetric iPEPS tensor.

    For the spin-1/2 Heisenberg model:
        ``S·S → -(S^x S^x - S^y S^y + S^z S^z)``

    Reference: Corboz, PRB 94, 035133 (2016).

    Args:
        gate: 2-site gate with shape ``(d, d, d, d)`` as a
              :class:`DenseTensor` (labels ``si, sj, si_out, sj_out``)
              or raw JAX array.
        d: Physical dimension (default 2 for spin-1/2).

    Returns:
        Rotated gate in the same format as input.

    Example::

        gate = heisenberg_gate()
        gate_rot = sublattice_rotate_gate(gate)
        # Now use single-site C4v CTM:
        env = ctm_tensor_c4v(A, chi=20)
        E = compute_energy_ctm_tensor(A, env, gate_rot)
    """
    if d != 2:
        raise NotImplementedError(
            f"sublattice rotation currently supports d=2 only, got d={d}"
        )

    is_tensor = isinstance(gate, Tensor)
    gate_arr = gate.todense() if is_tensor else jnp.asarray(gate)
    gate_arr = gate_arr.reshape(d, d, d, d)

    # U = e^{iπ σ^y / 2} = iσ^y = [[0, 1], [-1, 0]]
    U = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=gate_arr.dtype)
    U_dag = jnp.conj(U).T

    # H_rot[i,j,k,s] = U†[j,a] H[i,a,k,b] U[b,s]
    gate_rot = jnp.einsum("ja,iakb,bs->ijks", U_dag, gate_arr, U)

    if is_tensor:
        return DenseTensor(gate_rot, gate.indices)
    return gate_rot


def _wrap_as_dense_tensor(arr: jax.Array) -> DenseTensor:
    """Wrap a raw ``jax.Array`` iPEPS site tensor as a ``DenseTensor``.

    Assumes shape ``(D, D, D, D, d)`` with trivial U(1) charges
    (all zeros) and labels ``(u, d, l, r, phys)``.
    """
    arr = jnp.asarray(arr)
    D = arr.shape[0]
    d = arr.shape[4]
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    return DenseTensor(arr, indices)


def ipeps(
    hamiltonian_gate: Tensor | jax.Array,
    initial_peps: tuple[Tensor, Tensor] | tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, tuple[Tensor, Tensor], tuple[CTMEnvironment, CTMEnvironment]]:
    """Run 2-site iPEPS simple update + CTM for a 2D quantum lattice model.

    Always uses the Tensor-protocol 2-site simple update path.  Raw JAX
    arrays are automatically wrapped as ``DenseTensor`` with trivial U(1)
    charges.

    Algorithm overview:

    1. Simple update (imaginary time evolution) -- apply ``exp(-dt * H_bond)``
       on each bond, SVD-truncate to D, update lambda matrices.
    2. CTM environment computation -- initialise and iteratively absorb
       rows/columns until convergence.
    3. Compute energy per site using the CTM environment.

    Args:
        hamiltonian_gate: The 2-site Hamiltonian as a 4-leg tensor of shape
                          ``(d, d, d, d)`` representing H on a bond.
        initial_peps:     Tuple ``(A, B)`` of site tensors (Tensor or raw
                          JAX arrays), or ``None`` for random initialization.
        config:           iPEPSConfig.

    Returns:
        ``(energy_per_site, (A, B), (env_A, env_B))``
    """
    if initial_peps is not None and not isinstance(initial_peps, tuple):
        raise TypeError(
            f"ipeps() requires initial_peps to be a tuple of two tensors "
            f"or None, got {type(initial_peps).__name__}"
        )

    # Convert gate to dense JAX array for CTM energy evaluation
    gate_dense = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d = gate_dense.shape[0]
    D = config.max_bond_dim

    # Initialize A and B tensors — default is random
    if initial_peps is not None:
        A_raw, B_raw = initial_peps
        A = A_raw if isinstance(A_raw, Tensor) else _wrap_as_dense_tensor(A_raw)
        B = B_raw if isinstance(B_raw, Tensor) else _wrap_as_dense_tensor(B_raw)
    else:
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A_data = jax.random.normal(key_A, (D, D, D, D, d))
        B_data = jax.random.normal(key_B, (D, D, D, D, d))
        A = _wrap_as_dense_tensor(A_data)
        B = _wrap_as_dense_tensor(B_data)

    # Normalize
    norm_A = float(A.norm())
    if norm_A > EPS:
        A = A * (1.0 / norm_A)
    norm_B = float(B.norm())
    if norm_B > EPS:
        B = B * (1.0 / norm_B)

    # Build Trotter gate via Tensor protocol
    gate = _make_trotter_gate_tensor(hamiltonian_gate, config.dt, site_tensor=A)

    # Initialize lambdas from actual tensor bond dimensions
    _labels = A.labels()
    D_h = A.indices[_labels.index("r")].dim
    D_v = A.indices[_labels.index("d")].dim
    lam_h = jnp.ones(D_h)
    lam_v = jnp.ones(D_v)

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        if step % 2 == 0:
            A, B, lam_h = _simple_update_2site_horizontal_tensor(
                A, B, gate, lam_h, lam_v, D
            )
        else:
            A, B, lam_v = _simple_update_2site_vertical_tensor(
                A, B, gate, lam_h, lam_v, D
            )

    # CTM environment (uses dense arrays)
    A_dense = A.todense()
    B_dense = B.todense()
    env_A, env_B = ctm_2site(A_dense, B_dense, config.ctm)

    # Compute energy
    energy = compute_energy_ctm_2site(A_dense, B_dense, env_A, env_B, gate_dense, d)

    return float(energy), (A, B), (env_A, env_B)


def _heisenberg_dense_probe_energy(
    *, D: int, chi: int, device_mesh=None, seed: int = 0
) -> float:
    """Tiny dense iPEPS Heisenberg energy via one CTM convergence (test probe).

    Builds a deterministic random D-bond 1-site iPEPS ``A``, converges the dense
    CTM (optionally on ``device_mesh``), returns the NN Heisenberg energy. Tests
    only — exercised by ``tests/_ctm_sharding_parity_subproc.py`` to assert that
    the GSPMD-sharded forward CTM matches the single-device result bit-for-bit.

    Runs the 1-site cell through the default ``recipe="2x2"`` dispatch (the same
    path the production large-D CTM uses), with a fixed seed and small
    ``max_iter`` for a fast, deterministic probe.
    """
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor

    d = 2
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    bond_charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(
            sym, bond_charges.copy(), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(sym, bond_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(
            sym, phys_charges.copy(), FlowDirection.IN, label="phys"
        ),
    )
    A = DenseTensor(data, indices)

    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

    envs, _ = python_loop_ctm_converge(
        {(0, 0): A},
        SINGLE_SITE_NEIGHBORS,
        chi=chi,
        max_iter=20,
        conv_tol=1e-12,
        plateau_patience=None,
        device_mesh=device_mesh,
    )
    energy = compute_energy_ctm_tensor(A, envs[(0, 0)], gate, d)
    return float(energy)
