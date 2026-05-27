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

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._pess_multisite_energy import (
    compute_energy_pess_3site_multisite,
)
from tenax.algorithms.coarse_grain import CGGates, compute_energy_cg
from tenax.algorithms.ipeps_ad_policy import (
    ctm_converge_kwargs,
    validate_ctm_for_implicit_ad,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_3site_multisite,
    pess_to_kagome_supersite,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.lattice import kagome
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, Tensor

__all__ = [
    "build_pess_loss",
    "build_pess_loss_3site_multisite",
    "optimize_pess_3site_multisite_ad",
    "optimize_pess_ad",
]


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
            adjoint_method=config.adjoint_method,
            plateau_patience=config.plateau_patience,
        )

    return loss_fn


def _tree_real_dot(a, b) -> float:
    """Real part of the Hermitian inner product over a pytree of arrays."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return float(
        jnp.real(sum(jnp.sum(jnp.conj(la) * lb) for la, lb in zip(leaves_a, leaves_b)))
    )


def _initial_alpha(energy: float, slope: float) -> float:
    """Predict-to-zero initial step size for line search.

    ``alpha0 = min(1, |f0| / |slope|)`` — Nocedal & Wright's
    ``f_low = 0`` heuristic (Numerical Optimization, 2nd ed., §3.5).
    Linearly extrapolates the descent direction to predict the step
    that drives ``f`` to zero, then caps at 1.

    Replaces the previous ``min(1, 0.1·|p|/|d|)`` formula inherited from
    the iPEPS optimiser: that one assumes parameters are pre-normalised,
    which is true for iPEPS site tensors but NOT for iPESS primitives
    (the SU output's ``lambdas`` are unnormalised singular-value vectors
    that inflate ``|p|`` and pin alpha0 at 1.0).  At D=4 SU state, the
    old formula gave ``alpha0 = 1.0`` while the optimal step is
    ``~6e-6`` — HZ then needed ~17 internal bisections to recover.

    For typical iPESS energies ``|E| ~ 0.1`` and gradients ``|∇E| ~
    10²-10⁴``, this gives ``alpha0 ~ 1e-3 to 1e-5``, matching the
    Newton-predicted optimum within 1-2 bisections.

    A constant shift in the supplied bond gates can drive ``f(x0) → 0``
    while the gradient remains nonzero (the gradient is shift-invariant);
    the bare ratio then collapses to ``alpha0 = 0`` and both Armijo and
    Hager-Zhang accept it as a no-op step.  Floor the *numerator*
    ``|E|`` at ``1e-12`` so the heuristic still scales by ``|slope|`` in
    the low-energy regime — a fixed step would be far too coarse for
    stiff objectives where the optimal step is ``|E_floor| / |slope|``
    several orders of magnitude below it (see issue #401, PR #404 review).

    The floor is gated on a nonzero slope.  When the slope is also
    essentially zero we are at a true stationary point (zero gradient or
    a direction orthogonal to it); ``alpha0 = 0`` is the documented
    "no improvement" signal that ``_backtracking_line_search`` /
    ``_hager_zhang_line_search_step`` use to break the outer L-BFGS
    loop.  Returning a positive step there would trap the optimiser in
    ``max_iter`` no-op CTM evaluations at the same parameters.
    """
    if abs(slope) < 1e-30:
        return 0.0
    return min(1.0, max(abs(energy), 1e-12) / max(abs(slope), 1e-30))


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

    alpha = _initial_alpha(energy, slope)

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


def _hager_zhang_line_search_step(
    params: dict,
    direction: dict,
    grad: dict,
    energy: float,
    loss_fn: Callable[[dict], jnp.ndarray],
    grad_fn: Callable[[dict], tuple[jnp.ndarray, dict]],
) -> tuple[dict, float, float]:
    """Hager-Zhang line search on the L-BFGS descent direction.

    Mirrors the iPEPS optimizer's HZ wiring (see
    :mod:`tenax.algorithms.ipeps_optimize`).  Hager-Zhang enforces
    approximate Wolfe conditions (Hager & Zhang, SIAM J. Optim. 16(1),
    2005) — both sufficient decrease *and* the curvature condition —
    which keeps the L-BFGS Hessian approximation positive-definite on
    plateau regions where pure Armijo backtracking accumulates degenerate
    ``(s_k, y_k)`` pairs and eventually stalls.

    Returns ``(new_params, new_energy, alpha)``.  ``alpha == 0.0`` means
    the line search produced no improvement (caller should break).
    """
    from tenax.algorithms._line_search import hager_zhang_line_search

    slope = _tree_real_dot(grad, direction)
    if slope >= 0.0:
        direction = jax.tree.map(lambda g: -g, grad)
        slope = -_tree_real_dot(grad, grad)

    alpha0 = _initial_alpha(energy, slope)

    def _phi(a: float) -> float:
        trial = jax.tree.map(lambda p, d: p + a * d, params, direction)
        return float(loss_fn(trial))

    def _dphi(a: float) -> float:
        trial = jax.tree.map(lambda p, d: p + a * d, params, direction)
        _, g = grad_fn(trial)
        return _tree_real_dot(g, direction)

    alpha, f_alpha, _ = hager_zhang_line_search(
        _phi,
        _dphi,
        energy,
        slope,
        alpha_init=alpha0,
        rho=1.5,
        max_step=2.0 * alpha0,
        energy_bound=max(2.0, 2.0 * abs(energy)),
        bracket_only_phi=True,  # #504: skip dphi in bracket (implicit-AD path)
    )
    if alpha == 0.0 or f_alpha >= energy:
        return params, energy, 0.0
    new_params = jax.tree.map(lambda p, d: p + alpha * d, params, direction)
    return new_params, f_alpha, alpha


def _run_line_search(
    method: str,
    params: dict,
    direction: dict,
    grad: dict,
    energy: float,
    loss_fn: Callable[[dict], jnp.ndarray],
    grad_fn: Callable[[dict], tuple[jnp.ndarray, dict]],
) -> tuple[dict, float, float]:
    """Dispatch to the requested line search.

    Args:
        method: ``"hager_zhang"`` (default — enforces approximate Wolfe
            conditions, recommended for L-BFGS) or ``"armijo"`` (legacy
            backtracker, kept for opt-out and reproducibility).
    """
    if method == "hager_zhang":
        return _hager_zhang_line_search_step(
            params, direction, grad, energy, loss_fn, grad_fn
        )
    if method == "armijo":
        return _backtracking_line_search(params, direction, grad, energy, loss_fn)
    raise ValueError(
        f"Unknown line_search_method={method!r}. Expected 'hager_zhang' or 'armijo'."
    )


def optimize_pess_ad(
    initial_state: IPESSState,
    cg_gates: CGGates,
    config: CTMConfig,
    *,
    max_iter: int = 50,
    verbose: bool = False,
    line_search_method: str = "hager_zhang",
) -> tuple[IPESSState, float]:
    """L-BFGS optimization of kagome iPESS via the CG-iPEPS square path.

    Variational parameters are the iPESS primitives ``(R_a, R_b, R_c,
    T_u, lambdas)``. ``T_d`` is held frozen at its input value: in the
    CG-iPEPS coarse-graining, ``T_d`` is absorbed into the supersite via
    the down-bond ``sqrt(λ)`` gauges, and the remaining gauge freedom is
    spanned by the down-bond ``lambdas[3:6]`` themselves.

    Inner step uses ``optax.scale_by_lbfgs`` (memory 10) for the
    quasi-Newton direction; line search is a Python-level routine since
    the square CTM forward pass uses Python control flow that can't be
    ``jit``-traced through ``optax.lbfgs``'s bundled zoom search.

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
        line_search_method: ``"hager_zhang"`` (default — approximate
            Wolfe conditions, recommended for L-BFGS) or ``"armijo"``
            (legacy backtracker; kept for opt-out and reproducibility).

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
        params, last_energy, alpha = _run_line_search(
            line_search_method, params, direction, grads, last_energy, loss, grad_fn
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


# ---------------------------------------------------------------------------
# 3-site multisite AD path
# ---------------------------------------------------------------------------
#
# Mirrors the supersite path above but routes through the 3-site multisite
# encoding (:func:`tenax.algorithms.pess.pess_to_kagome_3site_multisite`) and
# the per-bond energy iterator
# (:func:`tenax.algorithms._pess_multisite_energy.compute_energy_pess_3site_multisite`,
# the corrected "4 NN + 2 marginalised-3-site" formula).
#
# Differences from the supersite path:
#   1. ``T_d`` is a real variational parameter (the multisite encoding uses
#      it explicitly, unlike the CG path which freezes it bit-exact).
#   2. Three site tensors per unit cell instead of one supersite — the CTM
#      runs as a 3-site multisite using the kagome neighbour map.
#   3. Energy peak intermediate is bounded by ``χ²·D⁶`` everywhere with no
#      diagonal-RDM term (vs supersite's ``χ²·D⁶`` plus the diagonal at
#      ``χ²·D⁴·d²·d_eff²``).

# Static layout used by the multisite path: 3 sublattices placed in a 3-cell
# strip so the multisite CTM's ``_sort_coords_for_direction`` (which indexes
# coords as ``c[0]``/``c[1]``) operates on tuples.  Topology is governed by
# the kagome neighbour map below, not by these labels.
_MULTISITE_NAME_TO_COORD: dict[str, tuple[int, int]] = {
    "u": (0, 0),
    "v": (1, 0),
    "w": (2, 0),
}
_MULTISITE_COORD_TO_NAME: dict[tuple[int, int], str] = {
    c: n for n, c in _MULTISITE_NAME_TO_COORD.items()
}
_MULTISITE_LATTICE = kagome()
_MULTISITE_COORD_NEIGHBORS: dict[tuple[int, int], dict[str, tuple[int, int]]] = {
    _MULTISITE_NAME_TO_COORD[n]: {
        direction: _MULTISITE_NAME_TO_COORD[
            _MULTISITE_LATTICE.neighbor_map[n][direction]
        ]
        for direction in ("left", "right", "top", "bottom")
    }
    for n in _MULTISITE_LATTICE.sites
}


def _make_multisite_indices(D: int, d: int) -> tuple[TensorIndex, ...]:
    """Square-iPEPS rank-5 ``(u, d, l, r, phys)`` index tuple for a multisite
    site tensor.  Trivial U(1) charges (single sector at charge 0); flows
    match the convention used by ``_ctm_tensor_energy``."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )


def build_pess_loss_3site_multisite(
    bond_gates: dict,
    config: CTMConfig,
    *,
    env_cache: dict | None = None,
) -> Callable[[IPESSState], jnp.ndarray]:
    """Build the AD loss closure for kagome iPESS via the 3-site multisite path.

    Args:
        bond_gates: Per-bond Hamiltonian dict from
            :func:`tenax.algorithms._pess_multisite_energy.kagome_3site_bond_gates`.
            6 entries keyed by ``frozenset({(name, dir), (nb_name, rev_dir)})``.
        config: CTM convergence settings (``chi``, ``max_iter``, ``conv_tol``,
            projector method, gauge, GMRES backward).
        env_cache: Optional mutable dict for CTM env warm-starting across
            gradient evaluations. When provided, each call reads
            ``env_cache.get("envs")`` and threads it as ``env_init=`` into
            the implicit-AD entry. The caller (typically
            :func:`optimize_pess_3site_multisite_ad`) is responsible for
            populating this dict after each accepted step via an eager
            ``python_loop_ctm_converge``. Mirrors the iPEPS AD policy
            plumbing in :func:`ipeps_ad_policy.make_ctm_energy_fn`.

    Returns:
        ``loss_fn(state: IPESSState) -> jnp.ndarray`` returning real scalar
        energy per microscopic kagome site.  Differentiable via ``jax.grad``
        through the implicit-AD multisite CTM; ``T_d`` participates in the
        gradient.
    """
    validate_ctm_for_implicit_ad(config)
    # Promote Tensor-valued gates to ndarray at entry so the rest of the
    # closure (and the .shape inference below) sees a uniform jax.Array
    # type — issue #402.  ``compute_energy_pess_3site_multisite`` accepts
    # both, but its ``.shape[0]`` access here would otherwise crash on
    # symmetry-aware / Tensor-protocol gates before reaching the supported
    # downstream code path.
    bond_gates_arr = {
        k: (g.todense() if isinstance(g, Tensor) else g) for k, g in bond_gates.items()
    }
    # Infer physical dimension from any bond gate (each is shape (d,d,d,d)).
    d = int(next(iter(bond_gates_arr.values())).shape[0])

    def _energy_fn(site_tensors_coord, envs_coord, _gate):
        site_tensors_name = {
            _MULTISITE_COORD_TO_NAME[c]: A for c, A in site_tensors_coord.items()
        }
        envs_name = {_MULTISITE_COORD_TO_NAME[c]: e for c, e in envs_coord.items()}
        return compute_energy_pess_3site_multisite(
            site_tensors_name,
            envs_name,
            _MULTISITE_LATTICE.neighbor_map,
            bond_gates_arr,
            d=d,
        )

    def loss_fn(state: IPESSState) -> jnp.ndarray:
        sites = pess_to_kagome_3site_multisite(
            state.R_a,
            state.R_b,
            state.R_c,
            state.T_u,
            state.T_d,
            state.lambdas,
        )
        D = sites["u"].shape[0]
        indices = _make_multisite_indices(D, d)
        # Per-site projective normalisation (same rationale as the supersite
        # path's single-tensor normalisation).  Each S_x scaling is absorbed
        # into a global wavefunction prefactor that drops out of the energy
        # ratio E = <ψ|H|ψ>/<ψ|ψ>; per-site bounding keeps CTM numerics
        # stable across random / SU initial conditions.
        site_tensors = {}
        for name, A in sites.items():
            A_norm = A / (jnp.linalg.norm(A) + 1e-12)
            site_tensors[_MULTISITE_NAME_TO_COORD[name]] = DenseTensor(A_norm, indices)

        env_init = env_cache.get("envs") if env_cache is not None else None
        return ctm_energy_implicit(
            site_tensors,
            _MULTISITE_COORD_NEIGHBORS,
            gate=None,  # ignored; energy_fn takes over
            chi=config.chi,
            max_iter=config.max_iter,
            conv_tol=config.conv_tol,
            projector_method=config.projector_method,
            renormalize=config.renormalize,
            projector_backward=config.projector_backward,
            forward_gauge=config.forward_gauge,
            conv_method=config.ctm_conv_method,
            min_iter=config.min_iter,
            chi_ramp=config.chi_ramp,
            env_init=env_init,
            energy_fn=_energy_fn,
            gmres_tol=config.gmres_tol,
            gmres_maxiter=config.gmres_maxiter,
            gmres_restart=config.gmres_restart,
            arnoldi_precheck=False,
            adjoint_method=config.adjoint_method,
            plateau_patience=config.plateau_patience,
        )

    return loss_fn


def optimize_pess_3site_multisite_ad(
    initial_state: IPESSState,
    bond_gates: dict,
    config: CTMConfig,
    *,
    max_iter: int = 50,
    verbose: bool = False,
    line_search_method: str = "hager_zhang",
) -> tuple[IPESSState, float]:
    """L-BFGS optimisation of kagome iPESS via the 3-site multisite path.

    Variational parameters: all 5 iPESS primitives plus the lambdas —
    ``(R_a, R_b, R_c, T_u, T_d, lambdas)``.  ``T_d`` is a real variable
    here (in contrast to :func:`optimize_pess_ad`, which freezes it).

    Args:
        initial_state: Starting :class:`IPESSState`.
        bond_gates: Per-bond gate dict from
            :func:`tenax.algorithms._pess_multisite_energy.kagome_3site_bond_gates`.
        config: CTM settings for the inner forward+backward sweeps.
        max_iter: Maximum L-BFGS outer iterations.
        verbose: Print energy at each step.
        line_search_method: ``"hager_zhang"`` (default — approximate
            Wolfe conditions, recommended for L-BFGS) or ``"armijo"``
            (legacy backtracker; kept for opt-out and reproducibility).

    Returns:
        ``(optimized_state, final_energy_per_site)``.
    """
    import optax

    # Promote Tensor-valued gates to ndarray once at the optimizer entry
    # so both the loss closure and the warm-start helper below see a
    # uniform jax.Array type — issue #402 / PR #405 review.  Without this
    # hoist, ``_build_site_tensors_for_warm_start`` would do
    # ``next(iter(bond_gates.values())).shape[0]`` on the un-promoted dict
    # and crash on Tensor inputs *after* the first loss eval succeeds,
    # leaving the advertised Tensor-valued multisite path unusable.
    bond_gates_arr = {
        k: (g.todense() if isinstance(g, Tensor) else g) for k, g in bond_gates.items()
    }
    d = int(next(iter(bond_gates_arr.values())).shape[0])

    # Env warm-start cache. Analogous to but simpler than
    # _optimize_gs_ad_multisite — no stall recovery, no best_env_cache
    # snapshot, no fresh-CTM final eval. The env never escapes back to
    # the caller, so trajectory-end env divergence from best_params is
    # currently harmless.
    _env_cache: dict[str, dict] = {}
    loss_fn_state = build_pess_loss_3site_multisite(
        bond_gates_arr, config, env_cache=_env_cache
    )

    params = {
        "R_a": initial_state.R_a,
        "R_b": initial_state.R_b,
        "R_c": initial_state.R_c,
        "T_u": initial_state.T_u,
        "T_d": initial_state.T_d,
        "lambdas": tuple(initial_state.lambdas),
    }

    def _params_to_state(p: dict) -> IPESSState:
        return IPESSState(
            R_a=p["R_a"],
            R_b=p["R_b"],
            R_c=p["R_c"],
            T_u=p["T_u"],
            T_d=p["T_d"],
            lambdas=tuple(p["lambdas"]),
        )

    def _build_site_tensors_for_warm_start(p: dict) -> dict:
        """Eager (no-AD) build of coord-keyed DenseTensors for the warm-start
        ``python_loop_ctm_converge`` call. Mirrors the loss-internal block."""
        sites = pess_to_kagome_3site_multisite(
            p["R_a"],
            p["R_b"],
            p["R_c"],
            p["T_u"],
            p["T_d"],
            p["lambdas"],
        )
        D = sites["u"].shape[0]
        indices = _make_multisite_indices(D, d)
        site_tensors = {}
        for name, A in sites.items():
            A_norm = A / (jnp.linalg.norm(A) + 1e-12)
            site_tensors[_MULTISITE_NAME_TO_COORD[name]] = DenseTensor(A_norm, indices)
        return site_tensors

    def _update_env_cache(p: dict) -> None:
        site_tensors = _build_site_tensors_for_warm_start(p)
        envs, _ = python_loop_ctm_converge(
            site_tensors,
            _MULTISITE_COORD_NEIGHBORS,
            **ctm_converge_kwargs(config, env_init=_env_cache.get("envs")),
        )
        _env_cache["envs"] = envs

    def loss(p: dict) -> jnp.ndarray:
        return loss_fn_state(_params_to_state(p))

    optimizer = optax.chain(
        optax.scale_by_lbfgs(memory_size=10),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(loss)

    last_energy = float(loss(params))
    _update_env_cache(params)
    best_energy = last_energy
    best_params = params
    for step in range(max_iter):
        e_val, grads = grad_fn(params)
        last_energy = float(e_val)
        direction, opt_state = optimizer.update(grads, opt_state, params)
        params, last_energy, alpha = _run_line_search(
            line_search_method, params, direction, grads, last_energy, loss, grad_fn
        )
        # Skip refresh on rejected steps (alpha == 0): params is unchanged
        # from the prior iterate so the cache is already current, and the
        # next line breaks the loop anyway.
        if alpha != 0.0:
            _update_env_cache(params)
        if last_energy < best_energy:
            best_energy = last_energy
            best_params = params
        if verbose:
            print(
                f"[optimize_pess_3site_multisite_ad] step {step + 1}/{max_iter}: "
                f"e = {last_energy:.10f}  alpha = {alpha:.3e}",
                flush=True,
            )
        if alpha == 0.0:
            break

    return _params_to_state(best_params), best_energy
