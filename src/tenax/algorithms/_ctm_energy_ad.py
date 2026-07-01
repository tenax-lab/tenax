"""CTM-to-energy AD wrappers: Python-loop forward with configurable backward."""

from __future__ import annotations

__all__ = [
    "ctm_energy_explicit",
    "ctm_energy_implicit",
    "get_last_implicit_ad_diagnostics",
    "set_implicit_ad_norm_diagnostics",
]

import logging
from functools import partial

import jax
import jax.numpy as jnp

from tenax.algorithms._arnoldi import arnoldi_spectral_radius_pytree
from tenax.algorithms._ctm_loop_core import (
    _run_ctm_loop_with_bump,
    _validate_chi_bump_args,
)
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

_GMRES_LOGGER = logging.getLogger("tenax.ctm.gmres")


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
    backward_steps: int | None = None,
    projector_method: str = "svd",
    renormalize: bool = True,
    projector_backward: str = "auto",
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    energy_fn=None,
    # NEW (#514) — in-CTM χ-bump knobs.  Defaults preserve existing
    # bump-off contract.  When ``ctmrg_heuristic_increase_chi=True``,
    # the warmup phase honors the bump; the backprop phase always uses
    # a fixed chi (post-warmup value) for tape integrity.
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
) -> jnp.ndarray:
    """Compute iPEPS energy with explicit-differentiation backward.

    Forward: warmup (no grad, optionally bump-aware) + checkpointed CTM
    sweeps at fixed chi.
    Backward: standard JAX autodiff through checkpointed sweeps.

    When ``ctmrg_heuristic_increase_chi=True`` is passed, the warmup phase
    grows chi reactively (variPEPS-style; Issue #492) up to ``chi_max``.
    The backprop phase uses the final post-warmup chi as a Python int so
    JAX traces the checkpointed sweep once and reuses across all backprop
    iterations — bump cannot fire during backprop.

    When ``backward_steps=K`` is set (TBPTT; #506), only the last ``K`` of
    the ``backprop_steps`` sweeps are differentiated; the leading
    ``backprop_steps - K`` sweeps run under ``jax.lax.stop_gradient`` so
    JAX skips their backward.  CTM converges geometrically, so the early
    backward contribution is bounded by ``ρ^K × (full contribution)`` —
    negligible for ρ≲0.5 and K≳10.  ``None`` (default) preserves the
    full backward.
    """
    if backward_steps is not None:
        if backward_steps < 1:
            raise ValueError(
                "backward_steps must be >= 1 (or None to disable TBPTT), "
                f"got {backward_steps}"
            )
        if backward_steps > backprop_steps:
            raise ValueError(
                f"backward_steps ({backward_steps}) cannot exceed "
                f"backprop_steps ({backprop_steps}); set to None to "
                "backprop through all sweeps."
            )
    # ---- validation (mirror ctm_energy_implicit / _sigma_gauged_ctm_converge) ----
    chi_current = _validate_chi_bump_args(
        chi=chi,
        chi_max=chi_max,
        env_init=env_init,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    )

    jit_step = _make_jit_ctm_step(neighbors)
    envs = (
        env_init
        if env_init is not None
        else {
            c: initialize_ctm_tensor_env(A, chi_current)
            for c, A in site_tensors.items()
        }
    )

    # ---- WARMUP — bump-aware, no-grad ----
    if warmup_steps > 0:
        warmup_result = _run_ctm_loop_with_bump(
            jit_step,
            site_tensors,
            envs,
            chi_current=chi_current,
            chi_max=chi_max,
            bump_enabled=ctmrg_heuristic_increase_chi,
            bump_threshold=ctmrg_heuristic_increase_chi_threshold,
            bump_step_size=ctmrg_heuristic_increase_chi_step_size,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
            gauge_fix_fn=None,  # warmup historically no gauge-fix
            max_iter=warmup_steps,
            min_iter=warmup_steps + 1,  # disables early-exit
            conv_tol=1e30,  # never satisfies
            conv_method="sv",
            plateau_patience=None,
        )
        envs = jax.tree.map(jax.lax.stop_gradient, warmup_result.envs)
        chi_post_warmup = warmup_result.final_chi
    else:
        chi_post_warmup = chi_current

    # ---- BACKPROP — fixed chi (tape integrity) ----
    def _step_envs_only(st, e):
        envs_out, _eps, _smin = jit_step(
            st,
            e,
            chi=chi_post_warmup,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        return envs_out

    # TBPTT split (#506): leading ``truncated`` sweeps stop-gradient, last
    # ``backward_steps`` sweeps remain differentiated.  When backward_steps
    # is None we backprop through everything (preserves pre-#506 contract).
    truncated = 0 if backward_steps is None else max(0, backprop_steps - backward_steps)
    for _ in range(truncated):
        envs = jax.lax.stop_gradient(_step_envs_only(site_tensors, envs))
    for _ in range(backprop_steps - truncated):
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
    plateau_patience: int | None = None,
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
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
        recipe:            CTM sweep recipe for BOTH the forward and the
                           fixed-point adjoint: ``"2x2"`` (Fishman plaquette,
                           default) or ``"1x1"`` (1-site moves that honor
                           ``projector_method``, incl. reduced-corner "qr").
                           Both passes use the same recipe so the adjoint
                           differentiates the sweep the forward converged.
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
        ctmrg_heuristic_increase_chi:
                           If True, run variPEPS-style in-CTM chi bumping
                           toward ``chi_max`` when the smallest retained
                           singular value drops below
                           ``ctmrg_heuristic_increase_chi_threshold``.
                           Requires ``chi_max`` to be set and is mutex
                           with ``chi_ramp``.  Defaults to False
                           (existing behaviour).

                           The backward pass is chi-locked to the
                           post-bump chi via JAX JIT static_argnames
                           (see docs/plans/2026-05-20-chi-lock-design.md):
                           the four backward JIT helpers re-trace one
                           compiled binary per distinct chi value
                           visited.  At defaults (chi 9 -> chi_max 24,
                           step 2) this is ~8 traces per helper,
                           amortised across the optimizer run.
        ctmrg_heuristic_increase_chi_threshold:
                           Smallest-S threshold that triggers an in-CTM
                           chi bump.  Default ``1e-6``.
        ctmrg_heuristic_increase_chi_step_size:
                           Number of additional chi units added per bump.
                           Default ``2``.
        chi_max:           Optional ceiling for in-CTM chi bumping.
                           Required when ``ctmrg_heuristic_increase_chi``
                           is True.

    Returns:
        Scalar energy per site.
    """
    # Reject ``ctmrg_heuristic_increase_chi`` + ``chi_ramp`` at the
    # ctm_energy_implicit boundary.  Without this guard, the chi_ramp
    # branch inside ``_run_forward`` silently drops the bump kwargs
    # (``_python_loop_chi_ramp`` doesn't accept them), so a caller
    # passing both would get a bump-off run with no error.  Mirrors
    # ``python_loop_ctm_converge``'s identical check (#514 Task 5
    # review follow-up).
    if ctmrg_heuristic_increase_chi and chi_ramp is not None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi and chi_ramp are mutually "
            "exclusive: chi_ramp is a deterministic schedule applied "
            "across stages, while ctmrg_heuristic_increase_chi is reactive "
            "inside a single CTM convergence call."
        )

    # Mirror the bump-kwarg validation from ``_sigma_gauged_ctm_converge`` at
    # the public boundary so kwarg-validation tests (chi_max=None,
    # step_size<=0, env_init above chi_max) surface their specific
    # ValueError at the entry point, before dispatch hands off to the
    # custom_vjp factory.  Chi-lock (#516) lifted the prior
    # NotImplementedError raise that this block historically preceded.
    _validate_chi_bump_args(
        chi=chi,
        chi_max=chi_max,
        env_init=env_init,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    )

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
        plateau_patience,
        ctmrg_heuristic_increase_chi,
        ctmrg_heuristic_increase_chi_threshold,
        ctmrg_heuristic_increase_chi_step_size,
        chi_max,
        recipe,
        device_mesh,
        ctm_chunk_size,
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
    plateau_patience: int | None = None,
    # NEW kwargs (#514) — defaults preserve existing callsite contract
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
):
    """CTM convergence with sigma gauge fixing for element-wise fixed point.

    Runs CTM sweeps with sigma gauge alignment at each step, ensuring
    the converged environment is a literal fixed point of the gauged step.

    The bump-aware loop body is shared with python_loop_ctm_converge via
    _run_ctm_loop_with_bump.  Validation mirrors that function so direct
    callers get consistent error messages.  When
    ``ctmrg_heuristic_increase_chi=True`` the loop performs variPEPS-style
    in-CTM chi bumping toward ``chi_max`` (see #492/#514); defaults keep
    bump-off behaviour for existing implicit-AD callers.

    Returns ``(envs, final_chi)`` so callers can route the post-bump chi
    into custom_vjp residuals for the chi-lock backward (#516).
    """
    # ---- validation (mirror python_loop_ctm_converge) ----
    chi_current = _validate_chi_bump_args(
        chi=chi,
        chi_max=chi_max,
        env_init=env_init,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    )

    jit_step_raw = _make_jit_ctm_step(
        neighbors, recipe, device_mesh=device_mesh, ctm_chunk_size=ctm_chunk_size
    )
    # Bind chunk_size so the bump helper and warmup loop pass it transparently.
    jit_step = partial(jit_step_raw, chunk_size=ctm_chunk_size)
    envs = (
        env_init
        if env_init is not None
        else {
            c: initialize_ctm_tensor_env(A, chi_current)
            for c, A in site_tensors.items()
        }
    )
    # Rung-2 gate: commit the initial envs to their D²-sharded layout so the
    # forward (and the residuals it saves for the implicit-AD backward) stay
    # sharded; GSPMD then propagates through the jitted step.  No-op when
    # ``device_mesh is None`` (default → today's single-device path).
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import commit_env

        envs = {c: commit_env(e, device_mesh) for c, e in envs.items()}

    # ---- QR warmup (unchanged) ----
    warmup = (
        min(qr_warmup_steps, max_iter)
        if projector_method == "qr" and qr_warmup_steps > 0
        else 0
    )
    for _ in range(warmup):
        envs, _eps, _smin = jit_step(
            site_tensors,
            envs,
            chi=chi_current,
            projector_method="eigh",
            renormalize=renormalize,
            projector_backward=projector_backward,
        )

    # ---- gauge_fix_fn pair adapter ----
    if forward_gauge == "phase":

        def _gauge_pair(envs_new, _envs_old):
            return {c: _phase_fix_ctm_tensor(envs_new[c]) for c in envs_new}
    elif forward_gauge == "sigma":

        def _gauge_pair(envs_new, envs_old):
            return {c: _sigma_gauge_fix_env(envs_new[c], envs_old[c]) for c in envs_new}
    elif forward_gauge == "none":
        _gauge_pair = None
    else:
        raise ValueError(f"Unknown forward_gauge={forward_gauge!r}")

    # ---- delegate to shared helper ----
    result = _run_ctm_loop_with_bump(
        jit_step,
        site_tensors,
        envs,
        chi_current=chi_current,
        chi_max=chi_max,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_threshold=ctmrg_heuristic_increase_chi_threshold,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        gauge_fix_fn=_gauge_pair,
        max_iter=max_iter - warmup,
        min_iter=max(0, min_iter - warmup),
        conv_tol=conv_tol,
        conv_method=conv_method,
        plateau_patience=plateau_patience,
    )
    return result.envs, result.final_chi


_VJP_CACHE: dict = {}


def invalidate_implicit_ad_warm_start() -> int:
    """Clear cached Neumann warm-start seeds across the implicit-AD VJP cache.

    The implicit-AD backward (``adjoint_method="fixed_point"`` and the
    eager-GMRES fallback) caches the final adjoint vector ``λ`` and feeds
    it back as the initial guess for the next gradient evaluation, which
    converges in ~3-10 iterations instead of ~20-40 in smooth basins
    (issue #501).  After events that decouple consecutive iterates --
    stall recovery rolling ``params`` back to ``best_params``, a scheduled
    χ-stage advance, or a reactive ``chi_auto_bump`` -- the cached ``λ``
    is no longer a good seed and the optimizer should drop it.

    Shape mismatches between ``prev_lam`` and the current ``env_leaves``
    are already auto-invalidated inside ``f_bwd``; this hook covers the
    parameter-jump case where shapes still line up but the iterate has
    moved far enough that the cached seed costs more than it saves.

    Returns
    -------
    int
        Number of cache entries whose seed was cleared (one per distinct
        static-config key currently held in ``_VJP_CACHE``).
    """
    n = 0
    for _f, mutables in _VJP_CACHE.values():
        cb = mutables.get("_invalidate_warm_start")
        if cb is not None:
            cb()
            n += 1
    return n


# F3 diagnostic: written by f_bwd after each fused-JIT call. Tests can
# read this to assert the fused fixed-point iteration converged without
# falling back to eager GMRES. Not part of the public API.
_F3_LAST_DIAGNOSTICS: dict = {}

# Toggle for the per-leaf ``device_get`` norm extraction in ``f_bwd``.
# Default False so a production implicit-AD run pays zero extra
# host-sync cost per backward (codex PR #524 P2).  The 2-site Tensor AD
# path flips this on at loop entry when both ``gs_implicit_ad`` and
# ``gs_verbose`` are set, then restores the previous value on exit.
_F3_DIAG_COMPUTE_NORMS: bool = False


def set_implicit_ad_norm_diagnostics(enabled: bool) -> bool:
    """Enable or disable per-backward ``||lam||`` / ``||init_lam||``
    extraction in the implicit-AD fused fixed-point backward.

    When disabled (default), ``_F3_LAST_DIAGNOSTICS`` still receives
    ``converged`` / ``diverged`` / ``n_iter`` (single scalars, one
    ``device_get`` each), but the per-env-leaf norm reductions are
    skipped.  Enabling adds one ``device_get`` per env leaf per
    backward (~ms-scale on GPU at chi~24).

    Returns the previous toggle value so callers can restore it.
    """
    global _F3_DIAG_COMPUTE_NORMS
    prev = _F3_DIAG_COMPUTE_NORMS
    _F3_DIAG_COMPUTE_NORMS = bool(enabled)
    return prev


def get_last_implicit_ad_diagnostics() -> dict:
    """Snapshot of the most recent implicit-AD adjoint solve diagnostics.

    Populated by the fused fixed-point backward (``adjoint_method="fixed_point"``)
    after every gradient call.  Keys:

    * ``converged`` -- True if the fixed-point iteration crossed
      ``gmres_tol`` within ``gmres_maxiter``.
    * ``diverged`` -- True if the in-loop divergence guard fired
      (``||lam_{k+1} - lam_k||`` grew past k>5).
    * ``n_iter`` -- Number of fixed-point iterations consumed.
    * ``warm_start_invalidated`` -- True if the cached ``prev_lam_leaves``
      was dropped due to a shape mismatch.
    * ``lam_norm`` -- ``||lam_final||`` in the env-leaf L2 norm.
    * ``init_lam_norm`` -- ``||init_lam||`` (warm-start seed or
      ``dE/dEnv`` on cold start).
    * ``amplification`` -- ``||lam_final|| / ||init_lam||``.  A large
      ratio indicates the linear adjoint operator ``(I - J^T)`` is
      near-singular (e.g. chi-ceiling with degenerate retained SVs in
      the frozen projectors).

    Returns a shallow copy so the caller cannot mutate internal state.
    Empty dict if no backward has run yet.
    """
    return dict(_F3_LAST_DIAGNOSTICS)


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
    plateau_patience,
    ctmrg_heuristic_increase_chi,
    ctmrg_heuristic_increase_chi_threshold,
    ctmrg_heuristic_increase_chi_step_size,
    chi_max,
    recipe="2x2",
    device_mesh=None,
    ctm_chunk_size=None,
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
        plateau_patience,
        ctmrg_heuristic_increase_chi,
        ctmrg_heuristic_increase_chi_threshold,
        ctmrg_heuristic_increase_chi_step_size,
        chi_max,
        recipe,  # distinct sweep recipe → distinct cached forward+backward
        device_mesh,  # sharded vs single → distinct cached forward+backward
        ctm_chunk_size,  # distinct chunk size → distinct forward lax.map shape
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
        plateau_patience=plateau_patience,
        ctmrg_heuristic_increase_chi=ctmrg_heuristic_increase_chi,
        ctmrg_heuristic_increase_chi_threshold=ctmrg_heuristic_increase_chi_threshold,
        ctmrg_heuristic_increase_chi_step_size=ctmrg_heuristic_increase_chi_step_size,
        chi_max=chi_max,
        recipe=recipe,
        device_mesh=device_mesh,
        ctm_chunk_size=ctm_chunk_size,
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
    plateau_patience: int | None = None,
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
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
    # ``prev_lam_leaves`` caches the implicit-AD adjoint solution across
    # consecutive ``f_bwd`` calls.  Consecutive L-BFGS steps usually take small
    # site-tensor steps, so the adjoint problem ``(I - J^T) λ = b`` is "close"
    # to the previous one — using the previous ``λ`` as a warm seed for the
    # Neumann iteration converges in fewer iterations (#501).  Cleared on
    # divergence/non-convergence so the next call gets a fresh start.
    _cached = {"prev_lam_leaves": None}

    def _invalidate_warm_start() -> None:
        """Drop the cached ``prev_lam_leaves`` warm-start seed.

        Called by ``invalidate_implicit_ad_warm_start`` when the optimizer
        triggers a stall reset (params roll back to ``best_params``), a
        scheduled chi advance, or a reactive ``chi_auto_bump``.  Without
        this hook the next backward would seed the Neumann iteration from
        a λ computed at a now-stale parameter point; correctness is still
        preserved by the in-loop divergence guard, but convergence costs
        extra iterations (see issue #501).
        """
        _cached["prev_lam_leaves"] = None

    mutables["_invalidate_warm_start"] = _invalidate_warm_start

    def _run_forward(site_tensors):
        """Run CTM convergence (shared by f and f_fwd).

        Returns ``(envs, chi_post)`` where ``chi_post`` is the post-bump
        chi (== ``chi`` when no bump fires).  ``chi_post`` threads into
        the custom_vjp residuals so the backward (#516 chi-lock) can
        dispatch JIT helpers at the actual post-forward chi.
        """
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
                plateau_patience=plateau_patience,
                recipe=recipe,
                device_mesh=device_mesh,
                ctm_chunk_size=ctm_chunk_size,
            )
            # chi_ramp doesn't trigger in-CTM bump (mutex enforced in dispatch);
            # chi_post is the final ramp stage's chi, which equals ``chi`` for
            # the implicit-AD entry point that uses ramp only.
            chi_post = chi
        else:
            envs, chi_post = _sigma_gauged_ctm_converge(
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
                plateau_patience=plateau_patience,
                ctmrg_heuristic_increase_chi=ctmrg_heuristic_increase_chi,
                ctmrg_heuristic_increase_chi_threshold=ctmrg_heuristic_increase_chi_threshold,
                ctmrg_heuristic_increase_chi_step_size=ctmrg_heuristic_increase_chi_step_size,
                chi_max=chi_max,
                recipe=recipe,
                device_mesh=device_mesh,
                ctm_chunk_size=ctm_chunk_size,
            )
        return envs, chi_post

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
        envs, _chi_post = _run_forward(site_tensors)  # chi_post unused outside grad
        return _compute_energy(site_tensors, envs)

    def f_fwd(params_data_tuple):
        site_tensors = dict(zip(coords, params_data_tuple))
        envs, chi_post = _run_forward(site_tensors)
        energy = _compute_energy(site_tensors, envs)

        _cached["env_treedef"] = jax.tree.structure(envs)
        env_leaves = tuple(jax.tree.leaves(envs))
        residuals = (params_data_tuple, env_leaves, chi_post)
        return energy, residuals

    # Build JIT'd sweep step for backward (same recipe as forward, so the
    # fixed-point adjoint differentiates the sweep the forward converged).
    # ``ctm_chunk_size`` is deliberately NOT threaded here: the #632 Increment-2
    # gate measured that chunking the adjoint's J^T matvec (VJP through the
    # chunked lax.map absorb) *increases* peak memory — it defeats XLA's existing
    # rematerialization of the chi^2*D^6 intermediate and adds stacked-residual /
    # VJP-transpose overhead (D=10 chi=16: monolith 28 GB fits, chunked OOMs).
    # The backward stays monolith (XLA remat handles it better); large-D backward
    # memory relief comes from GSPMD sharding (rung-2), not chunking. See
    # docs/superpowers/handoffs/2026-07-01-chunk-ctm-absorb-increment2-backward-gate.md.
    jit_step_bwd = _make_jit_ctm_step(neighbors, recipe)

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

    @partial(jax.jit, static_argnames=("chi",))
    def _jit_apply_Jt(params_data_tuple, env_leaves, v, *, chi):
        """Apply the GMRES matvec ``(I - J^T) v`` (NOT just ``J^T v``).

        The wrapping makes this the matvec for the linear system
        ``(I - J^T) λ = b`` that the eager-GMRES adjoint solves.
        Callers wanting the raw ``J^T v`` should subtract from ``v``,
        or use ``vjp_env_fn`` directly inside a JIT'd context.
        """
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
            e_out, _eps, _smin = jit_step_bwd(
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

    @partial(jax.jit, static_argnames=("chi",))
    def _jit_chain_rule(params_data_tuple, env_leaves, lam, g_scalar, *, chi):
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
            e_out, _eps, _smin = jit_step_bwd(
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

    @partial(jax.jit, static_argnames=("chi",))
    def _jit_fused_fixed_point_bwd(
        params_data_tuple,
        env_leaves,
        g_scalar,
        init_lam,
        *,
        chi,
    ):
        """F3: fused dE/denv + adjoint fixed-point + chain rule.

        One @jax.jit boundary in place of the F2 trio
        (_jit_dE_denv, _jit_apply_Jt, _jit_chain_rule) + Python adjoint
        loop. The adjoint runs as lax.while_loop so the loop body and
        the J^T VJP closure trace once and reuse on every iter.

        ``init_lam`` is the warm-start seed for the Neumann iteration
        (shape-matched to ``dE_denv``).  Cold start passes ``dE_denv``
        itself; subsequent calls pass the previous ``lam_final`` so the
        loop converges in fewer iterations (#501).

        Returns
        -------
        grads : tuple of pytrees
            Same shape as ``params_data_tuple``.
        diverged : bool scalar
            Set when the in-loop check (diff > prev_diff after step 5)
            fires. Caller falls back to eager GMRES when True.
        converged : bool scalar
            Set when ``diff < gmres_tol``. Diagnostic only.
        n_iter : int32 scalar
            Diagnostic only.
        lam_final : tuple of leaves
            Final adjoint vector, shaped like ``dE_denv``.  Caller caches
            this and passes it back as ``init_lam`` on the next call to
            warm-start the Neumann iteration.
        """
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        site_tensors = dict(zip(coords, params_data_tuple))
        env_treedef = _cached["env_treedef"]
        envs = jax.tree.unflatten(env_treedef, env_leaves)

        # --- 1. dE/denv via VJP through the energy function ---
        def energy_from_env(env_leaves_flat):
            e = jax.tree.unflatten(env_treedef, env_leaves_flat)
            if energy_fn_ is not None:
                return energy_fn_(site_tensors, e, gate_)
            return _default_energy(site_tensors, e, gate_, coords, neighbors)

        _, vjp_energy_env = jax.vjp(energy_from_env, env_leaves)
        dE_denv = vjp_energy_env(jnp.ones(()))[0]

        # --- 2. Build the J^T matvec closure ONCE inside this JIT ---
        def gauge_fixed_sweep_from_env(env_leaves_flat):
            e = jax.tree.unflatten(env_treedef, env_leaves_flat)
            e_ref = jax.tree.map(jax.lax.stop_gradient, e)
            e_out, _eps, _smin = jit_step_bwd(
                site_tensors,
                e,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            e_fixed = _apply_gauge_fix(e_out, e_ref)
            return tuple(jax.tree.leaves(e_fixed))

        _, vjp_env_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)

        def apply_Jt(v):
            # vjp_env_fn(v)[0] is the raw VJP through gauge_fixed_sweep_from_env,
            # which equals J^T v. Note this differs from the surviving F2
            # helper _jit_apply_Jt — that one wraps the VJP as (I - J^T) v
            # because GMRES needs the matvec for (I - J^T)·λ = b. The Neumann
            # iteration here uses J^T directly: λ_{k+1} = b + J^T λ_k.
            return vjp_env_fn(v)[0]

        # --- 3. Adjoint fixed-point inside lax.while_loop ---
        # ``init_lam`` is the caller-supplied warm-start seed (cold-start
        # uses ``dE_denv``; warm-start uses the previous ``lam_final``).
        # The RHS ``b = dE_denv`` of the Neumann step ``λ_{k+1} = b + J^T λ_k``
        # is unchanged — only the initial guess is seeded from cache.
        real_dtype = jnp.real(dE_denv[0]).dtype if dE_denv else jnp.float64
        init_lam_local = init_lam  # lambda_0 (warm or cold seed)
        init_diff = jnp.array(jnp.inf, dtype=real_dtype)
        init_k = jnp.array(0, dtype=jnp.int32)
        init_diverged = jnp.array(False)

        tol_arr = jnp.array(gmres_tol, dtype=real_dtype)
        maxiter_arr = jnp.array(gmres_maxiter, dtype=jnp.int32)

        def cond_fn(carry):
            _lam, prev_diff, k, diverged = carry
            return (k < maxiter_arr) & (~diverged) & (prev_diff > tol_arr)

        def body_fn(carry):
            lam, prev_diff, k, _diverged = carry
            jt_lam = apply_Jt(lam)
            new_lam = tuple(b + j for b, j in zip(dE_denv, jt_lam))
            diff = sum(jnp.linalg.norm(n - p) for n, p in zip(new_lam, lam)).astype(
                real_dtype
            )
            # Divergence guard: same trigger as F2 Python loop.
            new_diverged = (diff > prev_diff) & (k > jnp.array(5, jnp.int32))
            return (new_lam, diff, k + jnp.array(1, jnp.int32), new_diverged)

        lam_final, final_diff, n_iter, diverged = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (init_lam_local, init_diff, init_k, init_diverged),
        )
        converged = final_diff <= tol_arr

        # --- 4. Chain rule: direct dE/dparams + indirect J_p^T @ lam ---
        def energy_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            if energy_fn_ is not None:
                return energy_fn_(st, envs, gate_)
            return _default_energy(st, envs, gate_, coords, neighbors)

        _, vjp_energy_params = jax.vjp(energy_from_params, params_data_tuple)
        direct = vjp_energy_params(jnp.ones(()))[0]

        def gauge_fixed_sweep_from_params(p_tuple):
            st = dict(zip(coords, p_tuple))
            e_out, _eps, _smin = jit_step_bwd(
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
        indirect = vjp_sweep_params(lam_final)[0]

        total = jax.tree.map(lambda d, ind: g_scalar * (d + ind), direct, indirect)
        return (total,), diverged, converged, n_iter, lam_final

    @partial(jax.jit, static_argnames=("chi",))
    def _jit_gmres_solve(params_data_tuple, env_leaves, rhs, *, chi):
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
                e_out, _eps, _smin = jit_step_bwd(
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
        """Backward: adjoint solve (fixed-point or GMRES) + JIT'd chain rule.

        ``chi_post`` (from residuals) is the post-bump chi reported by the
        forward CTM.  It is passed to the four JIT'd backward helpers as a
        ``static_argnames`` parameter so JAX traces one compiled binary
        per distinct chi value visited (#516 chi-lock).
        """
        params_data_tuple, env_leaves, chi_post = residuals

        # Defense-in-depth: a malformed final_chi from a future
        # _run_ctm_loop_with_bump regression would silently corrupt the
        # backward.  Bounds-check at the boundary.
        chi_max_eff = chi_max if chi_max is not None else chi
        assert chi <= chi_post <= chi_max_eff, (
            f"chi_post={chi_post} out of bounds "
            f"[chi_initial={chi}, chi_max={chi_max_eff}]"
        )
        if chi_post != chi:
            _GMRES_LOGGER.debug(
                "chi-lock: backward at chi_post=%d (chi_initial=%d)",
                chi_post,
                chi,
            )

        # dE/denv is the RHS for the (I - J^T)·λ = b solve. F3's fused JIT
        # recomputes it internally, so the eager call is only needed by:
        # (a) the Arnoldi precheck (uses it as the Krylov seed), or
        # (b) the eager-GMRES path — either the "gmres" branch below or
        # the divergence-fallback path inside the "fixed_point" branch.
        # Compute lazily so the normal F3 success path doesn't pay an
        # extra _jit_dE_denv trace/dispatch on every backward call.
        _dE_denv_cached: tuple | None = None

        def _eager_dE_denv() -> tuple:
            nonlocal _dE_denv_cached
            if _dE_denv_cached is None:
                _dE_denv_cached = _jit_dE_denv(params_data_tuple, env_leaves)
            return _dE_denv_cached

        # Arnoldi precheck (optional, uses eager matvec). Load-bearing for
        # the fixed-point path: ρ(J^T) < 1 is what makes the Neumann
        # iteration converge.
        if arnoldi_precheck:

            def apply_Jt_only(v):
                """Apply J^T (not I - J^T)."""
                i_minus_jt_v = _jit_apply_Jt(
                    params_data_tuple, env_leaves, v, chi=chi_post
                )
                return tuple(vi - ri for vi, ri in zip(v, i_minus_jt_v))

            rho = arnoldi_spectral_radius_pytree(
                apply_Jt_only, _eager_dE_denv(), n_iter=20
            )
            if rho >= 1.0:
                raise CTMRGGradientError(rho)

        # Both eager-GMRES paths (fixed_point divergence fallback + the
        # "gmres" branch) share the same cached _jit_apply_Jt matvec.
        def _eager_apply_I_minus_Jt(v):
            return _jit_apply_Jt(params_data_tuple, env_leaves, v, chi=chi_post)

        if adjoint_method == "fixed_point":
            # F3 fused JIT: dE/denv + adjoint fixed-point + chain rule
            # in one trace. Returns final grads directly; if the fused
            # loop did not solve the system (diverged or hit
            # gmres_maxiter without reaching gmres_tol) we fall back
            # to eager GMRES below (same path as adjoint_method=="gmres").
            #
            # Warm-start (#501): the cached ``prev_lam_leaves`` from the
            # previous successful backward is fed back as the Neumann
            # initial guess.  Cold start (first call, or after a
            # divergence reset) seeds from ``dE_denv`` itself —
            # equivalent to the previous behaviour.
            prev_lam = _cached.get("prev_lam_leaves")
            invalidated_by_shape = False
            if prev_lam is not None:
                # Validate shapes match current env_leaves.  Without this, a D-sweep
                # or tensor-representation switch (with cache hit on coords/gate/
                # neighbors) would feed a stale-shape init_lam into the JIT and
                # raise a VJP tree/shape mismatch.  Cache structure: prev_lam is a
                # flat tuple of leaves shaped like env_leaves.
                if len(prev_lam) != len(env_leaves) or any(
                    a.shape != b.shape for a, b in zip(prev_lam, env_leaves)
                ):
                    _cached["prev_lam_leaves"] = None
                    invalidated_by_shape = True
                    prev_lam = None
            if prev_lam is None:
                init_lam = _eager_dE_denv()
            else:
                init_lam = prev_lam

            (
                grads_tuple,
                diverged,
                _converged,
                _n_iter,
                lam_final,
            ) = _jit_fused_fixed_point_bwd(
                params_data_tuple, env_leaves, g, init_lam, chi=chi_post
            )
            _F3_LAST_DIAGNOSTICS["diverged"] = bool(jax.device_get(diverged))
            _F3_LAST_DIAGNOSTICS["converged"] = bool(jax.device_get(_converged))
            _F3_LAST_DIAGNOSTICS["n_iter"] = int(jax.device_get(_n_iter))
            _F3_LAST_DIAGNOSTICS["warm_start_invalidated"] = invalidated_by_shape
            # Adjoint-amplification diagnostic: ||lam_final|| vs ||init_lam||.
            # Large ratios point to (I - J^T) being near-singular (typical at
            # chi-ceiling with degenerate retained SVs in the frozen
            # projectors).  Gated on ``_F3_DIAG_COMPUTE_NORMS`` so production
            # implicit-AD runs pay no per-backward host-sync cost when the
            # diagnostic isn't consumed (codex PR #524 P2).
            if _F3_DIAG_COMPUTE_NORMS:
                _lam_norm_sq = sum(
                    float(jax.device_get(jnp.sum(jnp.abs(_l) ** 2))) for _l in lam_final
                )
                _init_lam_norm_sq = sum(
                    float(jax.device_get(jnp.sum(jnp.abs(_l) ** 2))) for _l in init_lam
                )
                _F3_LAST_DIAGNOSTICS["lam_norm"] = _lam_norm_sq**0.5
                _F3_LAST_DIAGNOSTICS["init_lam_norm"] = _init_lam_norm_sq**0.5
                _F3_LAST_DIAGNOSTICS["amplification"] = (
                    (_lam_norm_sq / _init_lam_norm_sq) ** 0.5
                    if _init_lam_norm_sq > 0
                    else float("nan")
                )
                _GMRES_LOGGER.debug(
                    "F3 adjoint: n_iter=%d converged=%s diverged=%s tol=%g "
                    "||lam||=%.3e amp=%.3e",
                    _F3_LAST_DIAGNOSTICS["n_iter"],
                    _F3_LAST_DIAGNOSTICS["converged"],
                    _F3_LAST_DIAGNOSTICS["diverged"],
                    gmres_tol,
                    _F3_LAST_DIAGNOSTICS["lam_norm"],
                    _F3_LAST_DIAGNOSTICS["amplification"],
                )
            else:
                # Clear stale values so the consumer can detect that
                # the diagnostic wasn't computed on this backward.
                _F3_LAST_DIAGNOSTICS.pop("lam_norm", None)
                _F3_LAST_DIAGNOSTICS.pop("init_lam_norm", None)
                _F3_LAST_DIAGNOSTICS.pop("amplification", None)
                _GMRES_LOGGER.debug(
                    "F3 adjoint: n_iter=%d converged=%s diverged=%s tol=%g",
                    _F3_LAST_DIAGNOSTICS["n_iter"],
                    _F3_LAST_DIAGNOSTICS["converged"],
                    _F3_LAST_DIAGNOSTICS["diverged"],
                    gmres_tol,
                )
            if (
                _F3_LAST_DIAGNOSTICS["diverged"]
                or not _F3_LAST_DIAGNOSTICS["converged"]
            ):
                # Either the in-loop divergence guard fired (diff
                # increasing past k>5) or the loop ran out of
                # gmres_maxiter without diff crossing gmres_tol. The
                # latter is the contractive-but-slow regime — the
                # in-loop guard cannot detect it because diff stays
                # monotonically decreasing. Without this branch the
                # truncated Neumann iterate would be silently consumed
                # by _jit_chain_rule as if (I - J^T) λ = b had been
                # solved, biasing the gradient (issue #420).
                #
                # Invalidate the stale warm-start cache before the
                # eager fallback fires; the eager GMRES result then
                # becomes the new seed for the next call.
                _cached["prev_lam_leaves"] = None
                _F3_LAST_DIAGNOSTICS["warm_start_invalidated"] = True
                rhs = _eager_dE_denv()
                _GMRES_LOGGER.debug(
                    "Eager GMRES: maxiter=%d tol=%g restart=%d",
                    gmres_maxiter,
                    gmres_tol,
                    gmres_restart,
                )
                lam, _info = gmres_pytree_jax(
                    _eager_apply_I_minus_Jt,
                    rhs,
                    rhs,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
                lam_leaves = tuple(jax.tree.leaves(lam))
                _cached["prev_lam_leaves"] = lam_leaves
                return _jit_chain_rule(
                    params_data_tuple, env_leaves, lam_leaves, g, chi=chi_post
                )
            _cached["prev_lam_leaves"] = tuple(jax.tree.leaves(lam_final))
            return grads_tuple
        else:
            # adjoint_method == "gmres": eager Krylov via JAX's built-in
            # solver.  Retained as an opt-out for divergent edge cases.
            # Warm-start from the cached ``prev_lam_leaves`` when shape-
            # compatible (issue #501); fall back to ``rhs`` as the cold
            # x0 otherwise.  The shape check matches the fixed-point
            # branch above.
            rhs = _eager_dE_denv()
            prev_lam = _cached.get("prev_lam_leaves")
            if prev_lam is not None and (
                len(prev_lam) != len(env_leaves)
                or any(a.shape != b.shape for a, b in zip(prev_lam, env_leaves))
            ):
                _cached["prev_lam_leaves"] = None
                prev_lam = None
            if prev_lam is not None:
                # Unflatten back into the same pytree structure as rhs.
                rhs_treedef = jax.tree.structure(rhs)
                x0 = jax.tree.unflatten(rhs_treedef, prev_lam)
            else:
                x0 = rhs
            _GMRES_LOGGER.debug(
                "Eager GMRES: maxiter=%d tol=%g restart=%d warm=%s",
                gmres_maxiter,
                gmres_tol,
                gmres_restart,
                prev_lam is not None,
            )
            lam, _info = gmres_pytree_jax(
                _eager_apply_I_minus_Jt,
                rhs,
                x0,
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
            )
            lam_leaves = tuple(jax.tree.leaves(lam))
            _cached["prev_lam_leaves"] = lam_leaves
            return _jit_chain_rule(
                params_data_tuple, env_leaves, lam_leaves, g, chi=chi_post
            )

    f.defvjp(f_fwd, f_bwd)
    return f
