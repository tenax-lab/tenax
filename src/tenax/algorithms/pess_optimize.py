"""AD loss closure for kagome iPESS via the square coarse-grained iPEPS CTM.

The kagome iPESS state is coarse-grained to a 1-site square iPEPS supersite
(Convention C, Liao 2019): one supersite per up-triangle carrying ``T_u``,
``R_a``, ``R_b``, ``R_c`` and bond gauges that absorb ``T_d``. Phys leg
``d_eff = d**3`` (3 spins fused). The full kagome Hamiltonian is recovered
via :class:`tenax.algorithms.coarse_grain.CGGates` with ``h_intra``
(up-triangle) plus 3 ``h_inter`` bonds (down-triangle), the latter
evaluated through the existing horizontal / vertical / diagonal 2-site
RDM helpers in :mod:`tenax.algorithms._ctm_tensor_energy`.

This was a deliberate pivot away from a 2-sublattice honeycomb supersite
construction (a.k.a. "Convention A"). The latter required custom
``across-A_d`` 2-site RDM machinery that does not yet exist in the
honeycomb CTM (M2b follow-up to PR #347), so we reuse the CG-iPEPS path
that does ship with full kagome support on ``main``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.coarse_grain import CGGates, compute_energy_cg
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_supersite,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

__all__ = ["build_pess_loss", "optimize_pess_ad"]


def _make_supersite_indices(D: int, d_eff: int) -> tuple[TensorIndex, ...]:
    """Square-iPEPS rank-5 ``(u, d, l, r, phys)`` index tuple expected by
    the standard CTM. Trivial U(1) charges (single sector at charge 0).
    Flows match the convention used by ``_ctm_tensor_energy``."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d_eff, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )


def build_pess_loss(
    cg_gates: CGGates,
    config: CTMConfig,
) -> Callable[[IPESSState], jnp.ndarray]:
    """Build the AD loss closure for kagome iPESS optimization via the CG path.

    Args:
        cg_gates: :class:`CGGates` describing the kagome XXZ Hamiltonian
            on a 1-site square coarse-grained iPEPS, e.g. from
            :func:`tenax.algorithms.pess.kagome_xxz_pess_cg_gates`.
        config: CTM convergence settings (``chi``, ``max_iter``,
            ``conv_tol``, projector method, gauge, GMRES backward).

    Returns:
        ``loss_fn(state: IPESSState) -> jnp.ndarray`` returning the real
        scalar energy per kagome site. Differentiable via ``jax.grad``
        through the implicit-AD square CTM.
    """
    d_eff = int(cg_gates.h_intra.shape[0])

    def _energy_fn(site_tensors, envs, _gate):
        # Custom energy function for ctm_energy_implicit's energy_fn hook.
        # Drops the 2-site ``gate`` arg (unused) and routes through
        # compute_energy_cg, which handles intra (1-site RDM × h_intra)
        # plus 3 inter (h/v/diag 2-site RDM × h_inter) terms.
        A_norm = site_tensors[(0, 0)]
        return compute_energy_cg(A_norm, envs[(0, 0)], cg_gates, d_eff)

    def loss_fn(state: IPESSState) -> jnp.ndarray:
        A_super = pess_to_kagome_supersite(
            state.R_a, state.R_b, state.R_c, state.T_u, state.lambdas
        )
        # Projective gauge: only direction matters for the iPEPS state.
        # Normalize the supersite so the un-normalized RDM contraction
        # stays well above f64 zero across random / SU initial conditions.
        A_super = A_super / (jnp.linalg.norm(A_super) + 1e-12)
        D = A_super.shape[0]
        indices = _make_supersite_indices(D, d_eff)
        A_tensor = DenseTensor(A_super, indices)
        site_tensors = {(0, 0): A_tensor}

        return ctm_energy_implicit(
            site_tensors,
            SINGLE_SITE_NEIGHBORS,
            gate=None,  # ignored; energy_fn takes over
            chi=config.chi,
            max_iter=config.max_iter,
            conv_tol=config.conv_tol,
            projector_method=config.projector_method,
            renormalize=config.renormalize,
            forward_gauge=config.forward_gauge,
            conv_method=config.ctm_conv_method,
            min_iter=config.min_iter,
            chi_ramp=config.chi_ramp,
            energy_fn=_energy_fn,
            gmres_tol=config.gmres_tol,
            gmres_maxiter=config.gmres_maxiter,
            gmres_restart=config.gmres_restart,
            arnoldi_precheck=False,
        )

    return loss_fn


def _tree_real_dot(a, b) -> float:
    """Real part of the Hermitian inner product over a pytree of arrays."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return float(
        jnp.real(sum(jnp.sum(jnp.conj(la) * lb) for la, lb in zip(leaves_a, leaves_b)))
    )


def _backtracking_line_search(
    params: dict,
    direction: dict,
    grad: dict,
    energy: float,
    loss_fn: Callable[[dict], jnp.ndarray],
    c1: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 8,
) -> tuple[dict, float, float]:
    """Armijo backtracking on the L-BFGS descent direction."""
    slope = _tree_real_dot(grad, direction)
    if slope >= 0.0:
        direction = jax.tree.map(lambda g: -g, grad)
        slope = -_tree_real_dot(grad, grad)

    p_norm = math.sqrt(max(_tree_real_dot(params, params), 1e-30))
    d_norm = math.sqrt(max(_tree_real_dot(direction, direction), 1e-30))
    alpha = min(1.0, 0.1 * p_norm / d_norm)

    best_trial, best_f, best_alpha = params, energy, 0.0
    for _ in range(max_steps):
        trial = jax.tree.map(lambda p, d: p + alpha * d, params, direction)
        f_trial = float(loss_fn(trial))
        if f_trial < best_f:
            best_trial, best_f, best_alpha = trial, f_trial, alpha
        if f_trial <= energy + c1 * alpha * slope:
            return trial, f_trial, alpha
        alpha *= rho

    return best_trial, best_f, best_alpha


def optimize_pess_ad(
    initial_state: IPESSState,
    cg_gates: CGGates,
    config: CTMConfig,
    *,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[IPESSState, float]:
    """L-BFGS optimization of kagome iPESS via the CG-iPEPS square path.

    Variational parameters are the iPESS primitives ``(R_a, R_b, R_c,
    T_u, lambdas)``. ``T_d`` is held frozen at its input value: in the
    CG-iPEPS coarse-graining, ``T_d`` is absorbed into the supersite via
    the down-bond ``sqrt(λ)`` gauges, and the remaining gauge freedom is
    spanned by the down-bond ``lambdas[3:6]`` themselves.

    Inner step uses ``optax.scale_by_lbfgs`` (memory 10) for the
    quasi-Newton direction; line search is a Python-level Armijo
    backtracker since the square CTM forward pass uses Python control
    flow that can't be ``jit``-traced through ``optax.lbfgs``'s bundled
    zoom search.

    Args:
        initial_state: Starting :class:`IPESSState`. Typically the output
            of :func:`tenax.algorithms.pess.pess_simple_update`, but a
            freshly randomized state also works.
        cg_gates: kagome XXZ CG gates from
            :func:`kagome_xxz_pess_cg_gates`. Must encode the same
            ``delta`` and ``d`` as ``initial_state``.
        config: CTM settings for the inner forward+backward sweeps.
        max_iter: Maximum L-BFGS outer iterations.
        verbose: Print energy at each step.

    Returns:
        ``(optimized_state, final_energy_per_site)``.
    """
    import optax

    loss_fn_state = build_pess_loss(cg_gates, config)
    T_d_frozen = initial_state.T_d

    params = {
        "R_a": initial_state.R_a,
        "R_b": initial_state.R_b,
        "R_c": initial_state.R_c,
        "T_u": initial_state.T_u,
        "lambdas": tuple(initial_state.lambdas),
    }

    def _params_to_state(p: dict) -> IPESSState:
        return IPESSState(
            R_a=p["R_a"],
            R_b=p["R_b"],
            R_c=p["R_c"],
            T_u=p["T_u"],
            T_d=T_d_frozen,
            lambdas=tuple(p["lambdas"]),
        )

    def loss(p: dict) -> jnp.ndarray:
        return loss_fn_state(_params_to_state(p))

    optimizer = optax.chain(
        optax.scale_by_lbfgs(memory_size=10),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(loss)

    last_energy = float(loss(params))
    for step in range(max_iter):
        e_val, grads = grad_fn(params)
        last_energy = float(e_val)
        direction, opt_state = optimizer.update(grads, opt_state, params)
        params, last_energy, alpha = _backtracking_line_search(
            params, direction, grads, last_energy, loss
        )
        if verbose:
            print(
                f"[optimize_pess_ad] step {step + 1}/{max_iter}: "
                f"e = {last_energy:.10f}  alpha = {alpha:.3e}",
                flush=True,
            )
        if alpha == 0.0:
            break

    return _params_to_state(params), last_energy
