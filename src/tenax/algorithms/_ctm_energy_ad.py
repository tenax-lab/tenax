"""CTM-to-energy AD wrappers: Python-loop forward with configurable backward."""

from __future__ import annotations

__all__ = ["ctm_energy_explicit", "ctm_energy_implicit"]

import jax
import jax.numpy as jnp

from tenax.algorithms._arnoldi import arnoldi_spectral_radius_pytree
from tenax.algorithms._ctm_python_loop import (
    Coord,
    _make_jit_ctm_step,
    python_loop_ctm_converge,
)
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._gmres_lax import gmres_pytree, gmres_pytree_jax
from tenax.algorithms.ad_utils import CTMRGGradientError, _phase_fix_ctm_tensor
from tenax.contraction.contractor import contract


def _default_energy(site_tensors, envs, gate, coords, neighbors):
    """Compute default energy: single-site or multisite depending on unit cell."""
    if len(coords) == 1:
        coord0 = coords[0]
        return compute_energy_ctm_tensor(site_tensors[coord0], envs[coord0], gate)
    else:
        return compute_energy_ctm_tensor_multisite(site_tensors, envs, neighbors, gate)


def ctm_energy_explicit(
    site_tensors: dict[Coord, object],
    neighbors: dict[Coord, dict[str, Coord]],
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    projector_method: str = "svd",
    renormalize: bool = True,
    projector_backward: str = "auto",
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    energy_fn=None,
) -> jnp.ndarray:
    """Compute iPEPS energy with explicit-differentiation backward.

    Forward: warmup (no grad) + checkpointed CTM sweeps.
    Backward: standard JAX autodiff through checkpointed sweeps.
    """
    jit_step = _make_jit_ctm_step(neighbors)

    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
    )

    # Warmup: no gradient tracking
    for _ in range(warmup_steps):
        envs, _eps = jax.lax.stop_gradient(
            jit_step(
                site_tensors,
                envs,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )

    # Backprop phase: checkpointed sweeps
    def _step_envs_only(st, e):
        envs_out, _eps = jit_step(
            st,
            e,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        return envs_out

    for _ in range(backprop_steps):
        envs = jax.checkpoint(_step_envs_only)(site_tensors, envs)

    if energy_fn is not None:
        return energy_fn(site_tensors, envs, gate)
    coords = sorted(site_tensors.keys())
    return _default_energy(site_tensors, envs, gate, coords, neighbors)


# ---------------------------------------------------------------------------
# Implicit differentiation path: Python-loop forward + GMRES backward
# ---------------------------------------------------------------------------


def _transfer_matrix_leading_eigvec(T_dense, n_iter=30):
    """Compute leading right eigenvector of the double-layer transfer matrix.

    T_dense has shape (chi, D2, chi).  The transfer matrix is
    T_{(a,c),(b,d)} = T_{a,D2,b} * conj(T_{c,D2,d}) summed over D2.
    """
    chi = T_dense.shape[0]
    rho = jnp.eye(chi, dtype=T_dense.dtype)
    for _ in range(n_iter):
        # rho_new = sum_D2 T^* . rho . T^T
        rho = jnp.einsum("aib,cd,cid->ab", T_dense.conj(), rho, T_dense)
        rho = rho / (jnp.linalg.norm(rho) + 1e-30)
    return rho


def _wrap_sigma(sigma_data, contract_idx, output_idx, env_tensor):
    """Wrap a dense sigma matrix as a Tensor matching *env_tensor*'s type.

    The sigma is a chi x chi gauge transform.  ``contract_idx`` is the
    TensorIndex of the env leg that sigma contracts with; ``output_idx`` is
    the TensorIndex for the resulting (free) leg.

    For DenseTensor envs this creates a DenseTensor.
    For SymmetricTensor envs this creates a SymmetricTensor via from_dense,
    preserving block-sparse structure (sigma is block-diagonal when the
    transfer matrix respects the symmetry, which it does by construction).
    """
    from tenax.core.tensor import DenseTensor, SymmetricTensor

    indices = (output_idx, contract_idx)
    if isinstance(env_tensor, SymmetricTensor):
        return SymmetricTensor.from_dense(sigma_data, indices, tol=float("inf"))
    return DenseTensor(sigma_data, indices)


def _apply_sigma_to_corner(corner, s_left_data, s_right_data):
    """Apply sigma gauge to a corner: s_left^H @ corner @ s_right.

    corner is a 2-leg Tensor with indices (idx0, idx1).
    s_left^H contracts with idx0, s_right contracts with idx1.
    """
    idx0, idx1 = corner.indices
    # Temporary output labels — must not collide with existing labels
    tmp0 = ("_sigma_out", idx0.label)
    tmp1 = ("_sigma_out", idx1.label)

    # s_left^H: conjugate transpose. Row index = output (tmp0), col = idx0 (contracts).
    s_left_dag = _wrap_sigma(s_left_data.conj().T, idx0, idx0.relabel(tmp0), corner)
    # s_right: row index = idx1 (contracts), col = output (tmp1).
    # Sigma has shape (chi, chi) with layout (output, contract) — but here
    # we need (contract, output) so transpose the data.
    s_right = _wrap_sigma(s_right_data.T, idx1, idx1.relabel(tmp1), corner)
    # Contract: s_left_dag @ corner @ s_right
    # s_left_dag has labels (tmp0, idx0.label), corner has (idx0.label, idx1.label)
    # -> intermediate has (tmp0, idx1.label)
    # s_right has labels (tmp1, idx1.label) -> result has (tmp0, tmp1)
    result = contract(s_left_dag, corner, s_right)
    return result.relabel(tmp0, idx0.label).relabel(tmp1, idx1.label)


def _apply_sigma_to_edge(edge, s_data):
    """Apply sigma gauge to an edge: s^H @ edge @ s.

    edge is a 3-leg Tensor with indices (chi_left, phys, chi_right).
    The same sigma acts on both chi legs.
    """
    idx_l, idx_p, idx_r = edge.indices
    tmp_l = ("_sigma_out", idx_l.label)
    tmp_r = ("_sigma_out", idx_r.label)

    # s^H on the left chi leg: (tmp_l, idx_l.label)
    s_dag = _wrap_sigma(s_data.conj().T, idx_l, idx_l.relabel(tmp_l), edge)
    # s on the right chi leg: (tmp_r, idx_r.label) with transposed data
    s_right = _wrap_sigma(s_data.T, idx_r, idx_r.relabel(tmp_r), edge)
    result = contract(s_dag, edge, s_right)
    return result.relabel(tmp_l, idx_l.label).relabel(tmp_r, idx_r.label)


def _sigma_gauge_fix_env(env_new, env_old):
    """Fix gauge via transfer-matrix eigenvector alignment (sigma gauge).

    Aligns env_new to env_old so that the environment converges element-wise
    (not just spectrally). Based on arxiv:2311.11894.

    Sigma computation densifies edge tensors (chi x D^2 x chi) for the
    power-method transfer-matrix eigenvector — this is acceptable because
    the edge tensor size is at most chi x D^2 x chi (e.g. 16 x 9 x 16 =
    2304 elements at chi=16, D=3), which is always small.

    Sigma application uses label-based Tensor contractions, preserving
    SymmetricTensor type when the environment carries one.  The sigma
    matrix itself is chi x chi (small), wrapped as the same Tensor type
    as the environment.
    """
    # Densify edge tensors for sigma computation (small: chi x D^2 x chi).
    T1_n_d = env_new.T1.todense()
    T2_n_d = env_new.T2.todense()
    T3_n_d = env_new.T3.todense()
    T4_n_d = env_new.T4.todense()
    T1_o_d = env_old.T1.todense()
    T2_o_d = env_old.T2.todense()
    T3_o_d = env_old.T3.todense()
    T4_o_d = env_old.T4.todense()

    def _compute_sigma(T_new, T_old):
        """Compute sigma = Q_new @ Q_old^H from transfer matrix eigenvectors."""
        rho_new = _transfer_matrix_leading_eigvec(T_new)
        rho_old = _transfer_matrix_leading_eigvec(T_old)
        Q_new, R_new = jnp.linalg.qr(rho_new)
        Q_old, R_old = jnp.linalg.qr(rho_old)
        signs_new = jnp.sign(jnp.diag(R_new))
        signs_old = jnp.sign(jnp.diag(R_old))
        signs_new = jnp.where(signs_new == 0, 1.0, signs_new)
        signs_old = jnp.where(signs_old == 0, 1.0, signs_old)
        Q_new = Q_new * signs_new[None, :]
        Q_old = Q_old * signs_old[None, :]
        return Q_new @ Q_old.conj().T

    # stop_gradient on the sigmas: at the converged fixed point sigma = I,
    # so its derivative w.r.t. the environment is not needed for implicit
    # differentiation — only the CTM step Jacobian matters.  Without this,
    # the QR inside _compute_sigma produces NaN VJPs when the transfer-matrix
    # eigenvector density matrix is rank-deficient (chi > D^2).
    s1 = jax.lax.stop_gradient(_compute_sigma(T1_n_d, T1_o_d))
    s2 = jax.lax.stop_gradient(_compute_sigma(T2_n_d, T2_o_d))
    s3 = jax.lax.stop_gradient(_compute_sigma(T3_n_d, T3_o_d))
    s4 = jax.lax.stop_gradient(_compute_sigma(T4_n_d, T4_o_d))

    # Apply sigma to corners: s_row^H @ C @ s_col
    # Preserves SymmetricTensor type via label-based contraction.
    C1_f = _apply_sigma_to_corner(env_new.C1, s4, s1)
    C2_f = _apply_sigma_to_corner(env_new.C2, s1, s2)
    C3_f = _apply_sigma_to_corner(env_new.C3, s2, s3)
    C4_f = _apply_sigma_to_corner(env_new.C4, s3, s4)

    # Apply sigma to edges: s^H @ T @ s
    T1_f = _apply_sigma_to_edge(env_new.T1, s1)
    T2_f = _apply_sigma_to_edge(env_new.T2, s2)
    T3_f = _apply_sigma_to_edge(env_new.T3, s3)
    T4_f = _apply_sigma_to_edge(env_new.T4, s4)

    return CTMTensorEnv(
        C1=C1_f,
        C2=C2_f,
        C3=C3_f,
        C4=C4_f,
        T1=T1_f,
        T2=T2_f,
        T3=T3_f,
        T4=T4_f,
    )


def ctm_energy_implicit(
    site_tensors: dict[Coord, object],
    neighbors: dict[Coord, dict[str, Coord]],
    gate,
    *,
    chi: int = 20,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    projector_method: str = "svd",
    renormalize: bool = True,
    projector_backward: str = "lorentzian",
    qr_warmup_steps: int = 3,
    chi_ramp=None,
    env_init=None,
    forward_gauge: str = "phase",
    conv_method: str = "sv",
    min_iter: int = 4,
    gmres_tol: float = 1e-6,
    gmres_maxiter: int = 200,
    gmres_restart: int = 30,
    energy_fn=None,
    arnoldi_precheck: bool = False,
    adjoint_method: str = "fixed_point",
) -> jnp.ndarray:
    """Compute iPEPS energy with implicit-differentiation backward (GMRES).

    Forward: Python-loop CTM convergence with sigma gauge fixing.
    Backward: JIT-fused GMRES solve of ``(I - J_env^T) lam = dE/denv``,
    then chain rule to site tensor gradients.

    The forward applies sigma gauge fixing (transfer-matrix eigenvector
    alignment) at each CTM step, ensuring the converged environment is an
    element-wise fixed point.  The backward VJP is taken through the sigma-
    gauged step function, so ``(I - J_env^T)`` is well-conditioned.

    Uses SVD projectors with Lorentzian-regularized backward by default.
    SVD projectors achieve element-wise CTM convergence (no eigh sign
    ambiguity), and Lorentzian regularization provides correct truncation
    gradients through the fixed point (Francuz et al., PRR 7, 013237).

    Args:
        site_tensors:      Map from coordinate to iPEPS site tensor (Tensor).
        neighbors:         Map from coordinate to direction->neighbor coordinate.
        gate:              2-site Hamiltonian gate.
        chi:               Environment bond dimension.
        max_iter:          Maximum CTM iterations.
        conv_tol:          Convergence tolerance on corner singular values.
        projector_method:  ``"svd"`` (default), ``"eigh"``, or ``"qr"``.
        renormalize:       Renormalize environments after each sweep.
        projector_backward: ``"lorentzian"`` (default) or ``"standard"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        chi_ramp:          Optional chi-ramp schedule.
        env_init:          Optional initial environments.
        gmres_tol:         GMRES relative tolerance.
        gmres_maxiter:     GMRES maximum iterations.
        gmres_restart:     GMRES restart parameter.
        forward_gauge:     Gauge fixing in forward/backward: ``"phase"`` (default),
                           ``"sigma"`` (transfer-matrix eigenvector alignment), or
                           ``"none"`` (no gauge fixing).
        conv_method:       Convergence criterion: ``"sv"`` (corner singular
                           values, default) or ``"elementwise"`` (max element-wise
                           difference across all env tensors).
        min_iter:          Minimum iterations before checking convergence.
        energy_fn:         Optional custom energy function ``(A, env, gate) -> scalar``.
        arnoldi_precheck:  If True, run 20-step Arnoldi to estimate rho(J^T)
                           before GMRES. Raises ``CTMRGGradientError`` if
                           rho >= 1. Enable when the caller has recovery logic.
        adjoint_method:    ``"fixed_point"`` (default) runs a Python-loop
                           Neumann iteration ``λ_{k+1} = b + J^T λ_k``
                           that mirrors variPEPS's ``_ctmrg_rev_workhorse``
                           and converges geometrically when ρ(J^T) < 1.
                           ``"gmres"`` retains the eager Krylov solve as
                           an opt-out for divergent edge cases.  Both
                           paths reuse ``gmres_tol`` / ``gmres_maxiter``.

    Returns:
        Scalar energy per site.
    """
    coords = sorted(site_tensors.keys())
    # Pass Tensor objects directly through custom_vjp boundary.
    # Both DenseTensor and SymmetricTensor are registered JAX pytrees,
    # so JAX can differentiate through them without densifying.
    params_data = tuple(site_tensors[c] for c in coords)

    return _ctm_energy_implicit_dispatch(
        params_data,
        coords,
        neighbors,
        gate,
        chi,
        max_iter,
        conv_tol,
        projector_method,
        renormalize,
        projector_backward,
        qr_warmup_steps,
        chi_ramp,
        env_init,
        forward_gauge,
        conv_method,
        min_iter,
        gmres_tol,
        gmres_maxiter,
        gmres_restart,
        energy_fn,
        arnoldi_precheck,
        adjoint_method,
    )


def _sigma_gauged_ctm_converge(
    site_tensors,
    neighbors,
    *,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    renormalize,
    projector_backward,
    qr_warmup_steps,
    env_init,
    forward_gauge="phase",
    conv_method="sv",
    min_iter=4,
):
    """CTM convergence with sigma gauge fixing for element-wise fixed point.

    Runs CTM sweeps with sigma gauge alignment at each step, ensuring
    the converged environment is a literal fixed point of the gauged step.
    """
    from tenax.algorithms._ctm_tensor_convergence import (
        _corner_singular_values,
        _ctm_sv_diff,
        _max_env_leaf_diff,
    )

    jit_step = _make_jit_ctm_step(neighbors)
    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
    )

    # QR warm-up: only needed when projector_method == "qr"
    warmup = (
        min(qr_warmup_steps, max_iter)
        if projector_method == "qr" and qr_warmup_steps > 0
        else 0
    )
    for _ in range(warmup):
        envs, _eps = jit_step(
            site_tensors,
            envs,
            chi=chi,
            projector_method="eigh",
            renormalize=renormalize,
            projector_backward=projector_backward,
        )

    prev_svs: dict = {}
    prev_envs: dict | None = None
    for i in range(max_iter - warmup):
        envs_new, _eps = jit_step(
            site_tensors,
            envs,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        # Gauge fix: apply selected convention for element-wise convergence
        if forward_gauge == "phase":
            envs = {c: _phase_fix_ctm_tensor(envs_new[c]) for c in envs_new}
        elif forward_gauge == "sigma":
            envs = {c: _sigma_gauge_fix_env(envs_new[c], envs[c]) for c in envs_new}
        else:
            # forward_gauge == "none": no gauge fixing
            envs = envs_new

        # Skip convergence check until min_iter
        total_iter = warmup + i + 1
        if total_iter < min_iter:
            if conv_method == "sv":
                for c in sorted(envs):
                    prev_svs[c] = _corner_singular_values(envs[c].C1)
            else:
                prev_envs = {c: envs[c] for c in envs}
            continue

        if conv_method == "elementwise":
            # Element-wise: max absolute difference across all env tensor leaves
            if prev_envs is None:
                prev_envs = {c: envs[c] for c in envs}
                continue
            max_diff = 0.0
            for c in sorted(envs):
                max_diff = max(max_diff, _max_env_leaf_diff(prev_envs[c], envs[c]))
            converged = max_diff < conv_tol
            prev_envs = {c: envs[c] for c in envs}
        else:
            # SV convergence (default): corner singular value difference
            converged = True
            for c in sorted(envs):
                sv = _corner_singular_values(envs[c].C1)
                if c in prev_svs:
                    diff = float(_ctm_sv_diff(sv, prev_svs[c]))
                    if diff >= conv_tol:
                        converged = False
                else:
                    converged = False
                prev_svs[c] = sv

        if converged:
            break

    return envs


_VJP_CACHE: dict = {}


def _ctm_energy_implicit_dispatch(
    params_data_tuple,
    coords,
    neighbors,
    gate,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    renormalize,
    projector_backward,
    qr_warmup_steps,
    chi_ramp,
    env_init,
    forward_gauge,
    conv_method,
    min_iter,
    gmres_tol,
    gmres_maxiter,
    gmres_restart,
    energy_fn,
    arnoldi_precheck,
    adjoint_method,
):
    """Dispatch to custom_vjp-decorated function with caching.

    The ``custom_vjp`` + ``@jax.jit`` backward is expensive to rebuild.
    We cache it on a key derived from the static configuration so that
    optimizer loops reuse the compiled backward across steps.
    """
    # Build a hashable key from the static configuration.
    # Gate and energy_fn must be in the key because the JIT backward
    # captures them at trace time as compile-time constants.
    cache_key = (
        tuple(coords),
        chi,
        max_iter,
        conv_tol,
        projector_method,
        renormalize,
        projector_backward,
        qr_warmup_steps,
        forward_gauge,
        conv_method,
        min_iter,
        gmres_tol,
        gmres_maxiter,
        gmres_restart,
        id(neighbors),  # same dict object across optimizer steps
        id(gate),  # different Hamiltonian → different backward
        id(energy_fn),  # different energy callback → different backward
        arnoldi_precheck,
        adjoint_method,
    )

    entry = _VJP_CACHE.get(cache_key)
    if entry is not None:
        f, mutables = entry
        # Update per-call mutable state
        mutables["gate"] = gate
        mutables["chi_ramp"] = chi_ramp
        mutables["env_init"] = env_init
        mutables["energy_fn"] = energy_fn
        return f(params_data_tuple)

    # First call with this config — build and cache
    mutables = {
        "gate": gate,
        "chi_ramp": chi_ramp,
        "env_init": env_init,
        "energy_fn": energy_fn,
    }
    f = _make_implicit_vjp_fn(
        coords=coords,
        mutables=mutables,
        neighbors=neighbors,
        chi=chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        qr_warmup_steps=qr_warmup_steps,
        forward_gauge=forward_gauge,
        conv_method=conv_method,
        min_iter=min_iter,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_restart=gmres_restart,
        arnoldi_precheck=arnoldi_precheck,
        adjoint_method=adjoint_method,
    )
    _VJP_CACHE[cache_key] = (f, mutables)
    return f(params_data_tuple)


def _make_implicit_vjp_fn(
    coords,
    mutables,
    neighbors,
    chi,
    max_iter,
    conv_tol,
    projector_method,
    renormalize,
    projector_backward,
    qr_warmup_steps,
    forward_gauge,
    conv_method,
    min_iter,
    gmres_tol,
    gmres_maxiter,
    gmres_restart,
    arnoldi_precheck=False,
    adjoint_method="fixed_point",
):
    """Build a custom_vjp-decorated function closed over static config.

    Per-call mutable state (gate, chi_ramp, env_init, energy_fn)
    is read from the ``mutables`` dict, which is updated by the dispatch
    function before each call. This allows the compiled ``@jax.jit``
    backward to be reused across optimizer steps.
    """

    # Select gauge-fix function based on forward_gauge parameter.
    if forward_gauge == "phase":
        _gauge_fix_fn = _phase_fix_ctm_tensor
    elif forward_gauge == "sigma":
        _gauge_fix_fn = None  # sigma gauge handled by _sigma_gauge_fix_env (pair)
    elif forward_gauge == "none":
        _gauge_fix_fn = None
    else:
        raise ValueError(
            f"Unknown forward_gauge={forward_gauge!r}; use 'phase', 'sigma', or 'none'"
        )

    # Mutable cache for treedef from forward (needed in backward).
    _cached = {}

    def _run_forward(site_tensors):
        """Run CTM convergence (shared by f and f_fwd)."""
        chi_ramp = mutables["chi_ramp"]
        env_init = mutables["env_init"]
        if chi_ramp is not None:
            envs, _ = python_loop_ctm_converge(
                site_tensors,
                neighbors,
                chi=chi,
                max_iter=max_iter,
                min_iter=min_iter,
                conv_tol=conv_tol,
                conv_method=conv_method,
                renormalize=renormalize,
                projector_method=projector_method,
                qr_warmup_steps=qr_warmup_steps,
                projector_backward=projector_backward,
                chi_ramp=chi_ramp,
                env_init=env_init,
                gauge_fix_fn=_gauge_fix_fn,
            )
        else:
            envs = _sigma_gauged_ctm_converge(
                site_tensors,
                neighbors,
                chi=chi,
                max_iter=max_iter,
                conv_tol=conv_tol,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
                qr_warmup_steps=qr_warmup_steps,
                env_init=env_init,
                forward_gauge=forward_gauge,
                conv_method=conv_method,
                min_iter=min_iter,
            )
        return envs

    def _compute_energy(site_tensors, envs):
        """Compute energy using energy_fn or default."""
        gate = mutables["gate"]
        energy_fn = mutables["energy_fn"]
        if energy_fn is not None:
            return energy_fn(site_tensors, envs, gate)
        return _default_energy(site_tensors, envs, gate, coords, neighbors)

    @jax.custom_vjp
    def f(params_data_tuple):
        site_tensors = dict(zip(coords, params_data_tuple))
        envs = _run_forward(site_tensors)
        return _compute_energy(site_tensors, envs)

    def f_fwd(params_data_tuple):
        site_tensors = dict(zip(coords, params_data_tuple))
        envs = _run_forward(site_tensors)
        energy = _compute_energy(site_tensors, envs)

        _cached["env_treedef"] = jax.tree.structure(envs)
        env_leaves = tuple(jax.tree.leaves(envs))
        residuals = (params_data_tuple, env_leaves)
        return energy, residuals

    # Build JIT'd sweep step for backward (same as forward).
    jit_step_bwd = _make_jit_ctm_step(neighbors)

    # --- JIT'd building blocks for the backward ---
    # dE/denv and chain_rule are small JIT programs. The GMRES solve is
    # fully JIT-fused: the while_loop + all matvec VJPs compile into one
    # XLA program, eliminating Python-loop overhead.

    def _apply_gauge_fix(e_out, e_in):
        """Apply the same gauge fix used in the forward pass."""
        if forward_gauge == "phase":
            return {c: _phase_fix_ctm_tensor(e_out[c]) for c in coords}
        elif forward_gauge == "sigma":
            return {c: _sigma_gauge_fix_env(e_out[c], e_in[c]) for c in coords}
        else:
            return e_out

    @jax.jit
    def _jit_dE_denv(params_data_tuple, env_leaves):
        """Step 1: compute dE/denv (GMRES RHS)."""
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]

        def energy_from_env(env_leaves_flat):
            e = jax.tree.unflatten(env_treedef, env_leaves_flat)
            if energy_fn_ is not None:
                return energy_fn_(site_tensors, e, gate_)
            return _default_energy(site_tensors, e, gate_, coords, neighbors)

        _, vjp_fn = jax.vjp(energy_from_env, env_leaves)
        return vjp_fn(jnp.ones(()))[0]

    @jax.jit
    def _jit_apply_Jt(params_data_tuple, env_leaves, v):
        """Step 2: one J^T application (GMRES matvec)."""
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]

        def gauge_fixed_sweep_from_env(env_leaves_flat):
            e = jax.tree.unflatten(env_treedef, env_leaves_flat)
            # stop_gradient on the reference ensures gradients flow only
            # through the forward map, not through the alignment target.
            # Without this, the sigma gauge Jacobian includes reference-path
            # derivatives that push eigenvalues toward 1, making (I - J^T)
            # near-singular and GMRES returns near-zero lambda.
            # Cf. ad_utils.py step_fn_sigma (line ~1232).
            e_ref = jax.tree.map(jax.lax.stop_gradient, e)
            e_out, _eps = jit_step_bwd(
                site_tensors,
                e,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            e_fixed = _apply_gauge_fix(e_out, e_ref)
            return tuple(jax.tree.leaves(e_fixed))

        _, vjp_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)
        jt_v = vjp_fn(v)[0]
        return tuple(vi - ji for vi, ji in zip(v, jt_v))

    @jax.jit
    def _jit_chain_rule(params_data_tuple, env_leaves, lam, g_scalar):
        """Steps 3-4: direct gradient + indirect (J_params^T @ lam)."""
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        env_treedef = _cached["env_treedef"]
        envs = jax.tree.unflatten(env_treedef, env_leaves)

        # Direct: dE/dparams at fixed env
        def energy_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            if energy_fn_ is not None:
                return energy_fn_(st, envs, gate_)
            return _default_energy(st, envs, gate_, coords, neighbors)

        _, vjp_energy_params = jax.vjp(energy_from_params, params_data_tuple)
        direct = vjp_energy_params(jnp.ones(()))[0]

        # Indirect: J_params^T @ lam
        def gauge_fixed_sweep_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            e_out, _eps = jit_step_bwd(
                st,
                envs,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            e_fixed = _apply_gauge_fix(e_out, envs)
            return tuple(jax.tree.leaves(e_fixed))

        _, vjp_sweep_params = jax.vjp(gauge_fixed_sweep_from_params, params_data_tuple)
        indirect = vjp_sweep_params(lam)[0]

        total = jax.tree.map(lambda d, ind: g_scalar * (d + ind), direct, indirect)
        return (total,)

    @jax.jit
    def _jit_gmres_solve(params_data_tuple, env_leaves, rhs):
        """GMRES solve compiled into a single XLA program.

        Fuses the entire (I - J^T) solve into one XLA computation.
        All GMRES iterations run on-device with zero Python overhead.
        """
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]

        def apply_I_minus_Jt(v):
            """(I - J^T) matvec — inlined for JIT fusion."""

            def gauge_fixed_sweep_from_env(env_leaves_flat):
                e = jax.tree.unflatten(env_treedef, env_leaves_flat)
                e_ref = jax.tree.map(jax.lax.stop_gradient, e)
                e_out, _eps = jit_step_bwd(
                    site_tensors,
                    e,
                    chi=chi,
                    projector_method=projector_method,
                    renormalize=renormalize,
                    projector_backward=projector_backward,
                )
                e_fixed = _apply_gauge_fix(e_out, e_ref)
                return tuple(jax.tree.leaves(e_fixed))

            _, vjp_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)
            jt_v = vjp_fn(v)[0]
            return tuple(vi - ji for vi, ji in zip(v, jt_v))

        lam, info = gmres_pytree(
            apply_I_minus_Jt,
            rhs,
            rhs,
            tol=gmres_tol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        return tuple(jax.tree.leaves(lam)), info

    def f_bwd(residuals, g):
        """Backward: adjoint solve (fixed-point or GMRES) + JIT'd chain rule."""
        params_data_tuple, env_leaves = residuals

        # Step 1: dE/denv
        dE_denv = _jit_dE_denv(params_data_tuple, env_leaves)

        # Step 1.5: Arnoldi precheck (optional, uses eager matvec).
        # Load-bearing for the fixed-point path: ρ(J^T) < 1 is what makes
        # the Neumann iteration converge.
        if arnoldi_precheck:

            def apply_Jt_only(v):
                """Apply J^T (not I - J^T)."""
                i_minus_jt_v = _jit_apply_Jt(params_data_tuple, env_leaves, v)
                return tuple(vi - ri for vi, ri in zip(v, i_minus_jt_v))

            rho = arnoldi_spectral_radius_pytree(apply_Jt_only, dE_denv, n_iter=20)
            if rho >= 1.0:
                raise CTMRGGradientError(rho)

        # Step 2: solve (I - J^T) λ = dE/denv.  Two paths share the same
        # cached ``_jit_apply_Jt`` matvec; the choice is a config knob.
        def _eager_apply_I_minus_Jt(v):
            return _jit_apply_Jt(params_data_tuple, env_leaves, v)

        if adjoint_method == "fixed_point":
            # variPEPS-style Neumann iteration: λ_{k+1} = b + J^T λ_k.
            # Equivalent to solving (I - J^T) λ = b iff ρ(J^T) < 1, which
            # the Arnoldi precheck above guarantees when enabled.  Each
            # iter is one cached matvec — no Krylov subspace, no per-iter
            # compile cost after the first.
            def _apply_Jt(v):
                i_minus_jt_v = _jit_apply_Jt(params_data_tuple, env_leaves, v)
                return tuple(vi - im for vi, im in zip(v, i_minus_jt_v))

            lam = dE_denv  # initialize at b (one matvec saved)
            prev_diff = float("inf")
            for k in range(gmres_maxiter):
                jt_lam = _apply_Jt(lam)
                new_lam = tuple(b + j for b, j in zip(dE_denv, jt_lam))
                diff = sum(float(jnp.linalg.norm(n - p)) for n, p in zip(new_lam, lam))
                lam = new_lam
                if diff < gmres_tol:
                    break
                if diff > prev_diff and k > 5:
                    # Diverging — fall back to GMRES for safety.
                    lam, _info = gmres_pytree_jax(
                        _eager_apply_I_minus_Jt,
                        dE_denv,
                        dE_denv,
                        tol=gmres_tol,
                        maxiter=gmres_maxiter,
                        restart=gmres_restart,
                    )
                    break
                prev_diff = diff
        else:
            # adjoint_method == "gmres": eager Krylov via JAX's built-in
            # solver.  Retained as an opt-out for divergent edge cases.
            lam, _info = gmres_pytree_jax(
                _eager_apply_I_minus_Jt,
                dE_denv,
                dE_denv,
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
            )
        lam_leaves = tuple(jax.tree.leaves(lam))

        # Steps 3-4: chain rule
        return _jit_chain_rule(params_data_tuple, env_leaves, lam_leaves, g)

    f.defvjp(f_fwd, f_bwd)
    return f
