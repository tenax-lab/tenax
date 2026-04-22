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
    projector_method: str = "eigh",
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
        envs = jax.lax.stop_gradient(
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
    for _ in range(backprop_steps):
        envs = jax.checkpoint(
            lambda st, e: jit_step(
                st,
                e,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )(site_tensors, envs)

    if energy_fn is not None:
        return energy_fn(site_tensors, envs, gate)
    coords = sorted(site_tensors.keys())
    return _default_energy(site_tensors, envs, gate, coords, neighbors)


# ---------------------------------------------------------------------------
# Implicit differentiation path: Python-loop forward + GMRES backward
# ---------------------------------------------------------------------------


def _wrap_tensor(data, original):
    """Wrap dense data back into a Tensor preserving original index structure."""
    from tenax.core.tensor import SymmetricTensor

    if isinstance(original, SymmetricTensor):
        return SymmetricTensor.from_dense(data, original.indices, tol=float("inf"))
    return type(original)(data, original.indices)


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


def _sigma_gauge_fix_env(env_new, env_old):
    """Fix gauge via transfer-matrix eigenvector alignment (sigma gauge).

    Aligns env_new to env_old so that the environment converges element-wise
    (not just spectrally). Based on arxiv:2311.11894.
    """
    C1_n, C2_n, C3_n, C4_n = (
        c.todense() for c in (env_new.C1, env_new.C2, env_new.C3, env_new.C4)
    )
    T1_n, T2_n, T3_n, T4_n = (
        t.todense() for t in (env_new.T1, env_new.T2, env_new.T3, env_new.T4)
    )
    T1_o, T2_o, T3_o, T4_o = (
        t.todense() for t in (env_old.T1, env_old.T2, env_old.T3, env_old.T4)
    )

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
    s1 = jax.lax.stop_gradient(_compute_sigma(T1_n, T1_o))
    s2 = jax.lax.stop_gradient(_compute_sigma(T2_n, T2_o))
    s3 = jax.lax.stop_gradient(_compute_sigma(T3_n, T3_o))
    s4 = jax.lax.stop_gradient(_compute_sigma(T4_n, T4_o))

    # Apply sigma to corners: s_row^H @ C @ s_col
    C1_f = s4.conj().T @ C1_n @ s1
    C2_f = s1.conj().T @ C2_n @ s2
    C3_f = s2.conj().T @ C3_n @ s3
    C4_f = s3.conj().T @ C4_n @ s4

    # Apply sigma to edges: s^H @ T @ s
    T1_f = jnp.einsum("ab,bdc,ce->ade", s1.conj().T, T1_n, s1)
    T2_f = jnp.einsum("ab,bdc,ce->ade", s2.conj().T, T2_n, s2)
    T3_f = jnp.einsum("ab,bdc,ce->ade", s3.conj().T, T3_n, s3)
    T4_f = jnp.einsum("ab,bdc,ce->ade", s4.conj().T, T4_n, s4)

    return CTMTensorEnv(
        C1=_wrap_tensor(C1_f, env_new.C1),
        C2=_wrap_tensor(C2_f, env_new.C2),
        C3=_wrap_tensor(C3_f, env_new.C3),
        C4=_wrap_tensor(C4_f, env_new.C4),
        T1=_wrap_tensor(T1_f, env_new.T1),
        T2=_wrap_tensor(T2_f, env_new.T2),
        T3=_wrap_tensor(T3_f, env_new.T3),
        T4=_wrap_tensor(T4_f, env_new.T4),
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

    Returns:
        Scalar energy per site.
    """
    coords = sorted(site_tensors.keys())
    # Extract dense data for custom_vjp (JAX requires array leaves).
    # For SymmetricTensor inputs this densifies, which is necessary
    # because jax.custom_vjp cannot differentiate through block-sparse ops.
    params_data = [site_tensors[c].todense() for c in coords]
    templates = {c: site_tensors[c] for c in coords}

    return _ctm_energy_implicit_dispatch(
        tuple(params_data),
        coords,
        templates,
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
    )


def _reconstruct_site_tensors(params_data, coords, templates):
    """Reconstruct site_tensors dict from raw data arrays + templates."""
    from tenax.core.tensor import SymmetricTensor

    result = {}
    for i, c in enumerate(coords):
        tmpl = templates[c]
        if isinstance(tmpl, SymmetricTensor):
            result[c] = SymmetricTensor.from_dense(
                params_data[i], tmpl.indices, tol=float("inf")
            )
        else:
            result[c] = tmpl.__class__(params_data[i], tmpl.indices)
    return result


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
    from tenax.algorithms._ctm_tensor_convergence import _ctm_sv_diff

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
        envs = jit_step(
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
        envs_new = jit_step(
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
                    prev_svs[c] = jnp.linalg.svd(envs[c].C1.todense(), compute_uv=False)
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
                for told, tnew in zip(
                    jax.tree.leaves(prev_envs[c]),
                    jax.tree.leaves(envs[c]),
                ):
                    a = told.todense() if hasattr(told, "todense") else told
                    b = tnew.todense() if hasattr(tnew, "todense") else tnew
                    diff = float(jnp.max(jnp.abs(b - a)))
                    max_diff = max(max_diff, diff)
            converged = max_diff < conv_tol
            prev_envs = {c: envs[c] for c in envs}
        else:
            # SV convergence (default): corner singular value difference
            converged = True
            for c in sorted(envs):
                sv = jnp.linalg.svd(envs[c].C1.todense(), compute_uv=False)
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
    templates,
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
):
    """Dispatch to custom_vjp-decorated function with caching.

    The ``custom_vjp`` + ``@jax.jit`` backward is expensive to rebuild.
    We cache it on a key derived from the static configuration so that
    optimizer loops reuse the compiled backward across steps.
    """
    # Build a hashable key from the static configuration.
    # Templates and env_init change per call (updated tensors/envs) but
    # have the same structure, so the JIT backward compiles once and
    # reuses. Gate and energy_fn must be in the key because the JIT
    # backward captures them at trace time as compile-time constants.
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
    )

    entry = _VJP_CACHE.get(cache_key)
    if entry is not None:
        f, mutables = entry
        # Update per-call mutable state
        mutables["templates"] = templates
        mutables["gate"] = gate
        mutables["chi_ramp"] = chi_ramp
        mutables["env_init"] = env_init
        mutables["energy_fn"] = energy_fn
        return f(params_data_tuple)

    # First call with this config — build and cache
    mutables = {
        "templates": templates,
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
):
    """Build a custom_vjp-decorated function closed over static config.

    Per-call mutable state (templates, gate, chi_ramp, env_init, energy_fn)
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

    def _run_forward(params_data, site_tensors):
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
        params_data = list(params_data_tuple)
        templates = mutables["templates"]
        site_tensors = _reconstruct_site_tensors(params_data, coords, templates)
        envs = _run_forward(params_data, site_tensors)
        return _compute_energy(site_tensors, envs)

    def f_fwd(params_data_tuple):
        params_data = list(params_data_tuple)
        templates = mutables["templates"]
        site_tensors = _reconstruct_site_tensors(params_data, coords, templates)
        envs = _run_forward(params_data, site_tensors)
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
        params_data = list(params_data_tuple)
        templates_ = mutables["templates"]
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        site_tensors = _reconstruct_site_tensors(params_data, coords, templates_)
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
        params_data = list(params_data_tuple)
        templates_ = mutables["templates"]
        site_tensors = _reconstruct_site_tensors(params_data, coords, templates_)
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
            e_out = jit_step_bwd(
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
        templates_ = mutables["templates"]
        gate_ = mutables["gate"]
        energy_fn_ = mutables["energy_fn"]
        env_treedef = _cached["env_treedef"]
        envs = jax.tree.unflatten(env_treedef, env_leaves)

        # Direct: dE/dparams at fixed env
        def energy_from_params(p_tuple):
            pd = list(p_tuple)
            st = _reconstruct_site_tensors(pd, coords, templates_)
            if energy_fn_ is not None:
                return energy_fn_(st, envs, gate_)
            return _default_energy(st, envs, gate_, coords, neighbors)

        _, vjp_energy_params = jax.vjp(energy_from_params, params_data_tuple)
        direct = vjp_energy_params(jnp.ones(()))[0]

        # Indirect: J_params^T @ lam
        def gauge_fixed_sweep_from_params(p_tuple):
            pd = list(p_tuple)
            st = _reconstruct_site_tensors(pd, coords, templates_)
            e_out = jit_step_bwd(
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

        total = tuple(g_scalar * (d + ind) for d, ind in zip(direct, indirect))
        return (total,)

    @jax.jit
    def _jit_gmres_solve(params_data_tuple, env_leaves, rhs):
        """GMRES solve compiled into a single XLA program.

        Fuses the entire (I - J^T) solve into one XLA computation.
        All GMRES iterations run on-device with zero Python overhead.
        """
        params_data = list(params_data_tuple)
        templates_ = mutables["templates"]
        site_tensors = _reconstruct_site_tensors(params_data, coords, templates_)
        env_treedef = _cached["env_treedef"]

        def apply_I_minus_Jt(v):
            """(I - J^T) matvec — inlined for JIT fusion."""

            def gauge_fixed_sweep_from_env(env_leaves_flat):
                e = jax.tree.unflatten(env_treedef, env_leaves_flat)
                e_ref = jax.tree.map(jax.lax.stop_gradient, e)
                e_out = jit_step_bwd(
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
        """Backward: JIT-fused GMRES solve + JIT'd chain rule."""
        params_data_tuple, env_leaves = residuals

        # Step 1: dE/denv
        dE_denv = _jit_dE_denv(params_data_tuple, env_leaves)

        # Step 1.5: Arnoldi precheck (optional, uses eager matvec)
        if arnoldi_precheck:

            def apply_Jt_only(v):
                """Apply J^T (not I - J^T)."""
                i_minus_jt_v = _jit_apply_Jt(params_data_tuple, env_leaves, v)
                return tuple(vi - ri for vi, ri in zip(v, i_minus_jt_v))

            rho = arnoldi_spectral_radius_pytree(apply_Jt_only, dE_denv, n_iter=20)
            if rho >= 1.0:
                raise CTMRGGradientError(rho)

        # Step 2: GMRES solve via JAX's built-in solver (eager, not JIT-fused).
        # Uses jax.scipy.sparse.linalg.gmres which converges reliably at
        # large chi where the custom gmres_lax can produce wrong gradients.
        def _eager_apply_I_minus_Jt(v):
            return _jit_apply_Jt(params_data_tuple, env_leaves, v)

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
