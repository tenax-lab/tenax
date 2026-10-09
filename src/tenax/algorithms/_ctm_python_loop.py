"""Python-loop CTM convergence — JIT single sweeps, Python convergence loop.

Instead of JIT-tracing the entire CTM convergence loop (which takes 20-90+
minutes to compile), we JIT only single CTM sweeps and run the convergence
loop in Python.  This gives the same converged environment with seconds of
compile time.
"""

from __future__ import annotations

__all__ = [
    "CTMConvergeInfo",
    "_make_jit_ctm_step",
    "python_loop_ctm_converge",
]

import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from jax.sharding import Mesh

# Defined in the leaf policy module so ``_ctm_tensor_convergence`` can build
# one without importing this module (which imports it: an import cycle).
from tenax.algorithms._ctm_convergence_policy import CTMConvergeInfo
from tenax.algorithms._ctm_loop_core import (
    _run_ctm_loop_with_bump,
    _validate_chi_bump_args,
)
from tenax.algorithms._ctm_tensor_convergence import (
    Coord,
    _ctm_tensor_sweep_multisite,
    _warn_recipe_1x1_deprecated,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor, Tensor

# Process-lifetime cache so repeat calls with the same neighbors dict reuse
# the same compiled @jit'd ``_step`` function.  Without this, every callsite
# (forward CTM convergence, implicit-AD f_bwd, line search, etc.) creates a
# fresh ``_step`` closure with its own JIT cache and pays redundant compile
# cost.  Diagnosed in docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md.
# Keyed by id(neighbors) — safe because neighbors dicts are constructed once
# per optimizer invocation and stay alive throughout.
# Key is (id(neighbors), recipe, device_mesh, ctm_chunk_size): a 4-tuple of an
# int, a str, a ``jax.sharding.Mesh | None`` and an ``int | None`` (Mesh
# imported only under TYPE_CHECKING; the annotation is a string here thanks to
# ``from __future__ import annotations``).
_JIT_STEP_CACHE: dict[tuple[int, str, Mesh | None, int | None], Callable] = {}


def _make_jit_ctm_step(
    neighbors: dict[Coord, dict[str, Coord]],
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
):
    """Create a JIT-compiled CTM step function for a given neighbor topology.

    The ``neighbors`` dict is captured in the closure so it is not traced
    by JAX (it contains only Python-level coordinate tuples, not arrays).

    Memoised by ``id(neighbors)`` so all callsites in a single optimizer
    invocation share one ``_step`` function and its JIT cache.

    Returns:
        A JIT-compiled function with signature::

            step(site_tensors, envs, *, chi, projector_method,
                 renormalize, projector_backward, chunk_size)
                 -> tuple[dict[Coord, CTMTensorEnv], jax.Array]

        The returned tuple is ``(new_envs, max_truncation_error)`` where
        ``max_truncation_error`` is the largest singular-value truncation
        error (ε_T) observed across all projector computations in the sweep.
    """
    # Include ``recipe`` in the cache key so the "1x1" and "2x2" sweeps get
    # distinct compiled ``_step`` closures even when they share the same
    # ``neighbors`` dict (which is reused across optimizer steps).
    # ``device_mesh`` is captured (static, non-traced) in the closure, so it
    # must participate in the cache key: the sharded and single-device steps
    # are different compiled functions.
    # ``ctm_chunk_size`` is static: different chunk sizes yield differently
    # shaped lax.map computations that must not share a JIT cache entry.
    # Asymmetric keying: ``neighbors`` is an unhashable dict so we fall back to
    # ``id()``; ``device_mesh`` is a hashable ``jax.sharding.Mesh`` (or None)
    # with a value ``__eq__``, so it is used directly — equal meshes share a
    # cache entry and we avoid id-reuse hazards after GC.
    cache_key = (id(neighbors), recipe, device_mesh, ctm_chunk_size)
    cached = _JIT_STEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    @partial(
        jax.jit,
        static_argnames=(
            "chi",
            "projector_method",
            "renormalize",
            "projector_backward",
            "chunk_size",
        ),
    )
    def _step(
        site_tensors: dict[Coord, Tensor],
        envs: dict[Coord, CTMTensorEnv],
        *,
        chi: int,
        projector_method: str = "svd",
        renormalize: bool = False,
        projector_backward: str = "auto",
        chunk_size: int | None = None,
    ) -> tuple[dict[Coord, CTMTensorEnv], jax.Array, jax.Array]:
        return _ctm_sweep(
            site_tensors,
            envs,
            neighbors=neighbors,
            recipe=recipe,
            device_mesh=device_mesh,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
            chunk_size=chunk_size,
        )

    _JIT_STEP_CACHE[cache_key] = _step
    return _step


def _ctm_sweep(
    site_tensors: dict[Coord, Tensor],
    envs: dict[Coord, CTMTensorEnv],
    *,
    neighbors: dict[Coord, dict[str, Coord]],
    recipe: str,
    device_mesh,
    chi: int,
    projector_method: str,
    renormalize: bool,
    projector_backward: str,
    chunk_size: int | None,
) -> tuple[dict[Coord, CTMTensorEnv], jax.Array, jax.Array]:
    """One CTM sweep: the body that ``_make_jit_ctm_step`` jits.

    Called directly (not jitted) by the #1098 eager layout phase.
    """
    double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    return _ctm_tensor_sweep_multisite(
        envs,
        double_layers,
        neighbors,
        chi,
        renormalize,
        projector_method,
        projector_backward=projector_backward,
        recipe=recipe,
        device_mesh=device_mesh,
        chunk_size=chunk_size,
    )


# Labels of the D² legs of the edge tensors (``CTMTensorEnv``).  Every other
# env leg is a chi leg.
_D2_LEG_LABELS = frozenset({"u2", "r2", "d2", "l2"})


def _chi_layout(envs: dict[Coord, CTMTensorEnv]) -> tuple:
    """Per-sector ``(charge, size)`` of every chi leg of every env tensor (#1098).

    The layout of a ``SymmetricTensor`` chi leg is what the traced projector
    can never change (it reuses the leg's sector sizes, #929).  Dense tensors
    have no layout and contribute nothing.  Keyed by leg label and sorted, so
    a sweep that only reorders a tensor's legs (the cold-init corners come
    back with their two legs swapped) is not a layout change.
    """
    out = []
    for c in sorted(envs):
        for field, t in zip(envs[c]._fields, envs[c]):
            if not isinstance(t, SymmetricTensor):
                continue
            for ix in t.indices:
                if ix.label in _D2_LEG_LABELS:
                    continue
                pairs = zip(
                    np.asarray(ix.sectors).tolist(),
                    np.asarray(ix.multiplicities).tolist(),
                )
                out.append((c, field, ix.label, tuple(sorted(pairs))))
    return tuple(sorted(out))


def _charge_sorted_chi_legs(t: SymmetricTensor) -> SymmetricTensor:
    """``t`` with every chi leg's basis in ascending-charge order (#1098).

    The eager projector orders a new chi bond by singular value, so the charge
    *sequence* along the leg depends on the values: two eager runs with the
    same layout give different ``TensorIndex`` aux data, and every jitted
    consumer (``_step``, the bond-phase gauge) recompiles.  The jitted
    projector and the cold init emit the bond in ascending-charge order.

    A stable sort by charge keeps the order inside each sector, and blocks
    are stored per sector, so only the index metadata changes: this is a
    faithful relabelling of the dense basis (the same one on both ends of a
    bond, which carry the same charge list, #905), and the block data is
    reused as is.
    """
    new_indices = []
    changed = False
    for ix in t.indices:
        if ix.label in _D2_LEG_LABELS or getattr(ix, "fuse_info", None) is not None:
            new_indices.append(ix)
            continue
        q = np.asarray(ix.charges)
        q_sorted = np.sort(q, kind="stable")
        if np.array_equal(q, q_sorted):
            new_indices.append(ix)
            continue
        changed = True
        new_indices.append(
            TensorIndex.from_charges(ix.symmetry, q_sorted, ix.flow, label=ix.label)
        )
    if not changed:
        return t
    return SymmetricTensor._raw(
        indices=tuple(new_indices),
        data=t._data,
        block_keys=t._block_keys,
        block_shapes=t._block_shapes,
        block_offsets=t._block_offsets,
    )


def _charge_sorted_envs(envs: dict[Coord, CTMTensorEnv]) -> dict[Coord, CTMTensorEnv]:
    """Apply :func:`_charge_sorted_chi_legs` to every symmetric env tensor."""
    return {
        c: type(env)(
            *(
                _charge_sorted_chi_legs(t) if isinstance(t, SymmetricTensor) else t
                for t in env
            )
        )
        for c, env in envs.items()
    }


# Layout of the cold init, per (site index metadata, chi); the init's chi
# charges are a function of the site legs and chi alone.
_COLD_LAYOUT_CACHE: dict = {}


def _cold_init_layout(site_tensors: dict[Coord, Tensor], chi: int) -> tuple:
    """``_chi_layout`` of ``initialize_ctm_tensor_env`` for these sites."""
    key = (
        chi,
        tuple((c, tuple(site_tensors[c].indices)) for c in sorted(site_tensors)),
    )
    cached = _COLD_LAYOUT_CACHE.get(key)
    if cached is None:
        cached = _chi_layout(
            {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
        )
        _COLD_LAYOUT_CACHE[key] = cached
    return cached


def _needs_eager_layout_phase(site_tensors, env_init) -> bool:
    """True for concrete symmetric input: the only case the traced cut freezes.

    Dense input has no chi-leg layout, so it never enters the eager phase and
    runs the same compiled step as before (#1098).  Under an enclosing trace
    the sweep cannot run eagerly, so the phase is skipped there too.
    """
    leaves = jax.tree.leaves(
        (site_tensors, env_init),
        is_leaf=lambda x: isinstance(x, Tensor),
    )
    if not any(isinstance(x, SymmetricTensor) for x in leaves):
        return False
    arrays = jax.tree.leaves((site_tensors, env_init))
    return not any(isinstance(a, jax.core.Tracer) for a in arrays)


def _eager_ctm_sweep(site_tensors, envs, **kwargs):
    """One un-jitted CTM sweep (#1098).

    Outside ``jax.jit`` the projector sees concrete blocks and takes its eager
    branch (global top-chi, may add or drop charge sectors).  Module-level so
    tests can count eager sweeps.
    """
    return _ctm_sweep(site_tensors, envs, **kwargs)


def _adapt_symmetric_layout(
    site_tensors,
    envs,
    *,
    max_sweeps: int,
    gauge_pair,
    compare_input: bool,
    **sweep_kwargs,
):
    """Eager sweeps until the chi-leg layout stops changing (#1098).

    Every sweep of the jitted ``step`` sizes each sector from the leg it
    overwrites (``_seam_chi_charges``, #929), so a jitted symmetric CTM keeps
    whatever layout it starts with -- for a cold start, the tiling of
    ``_tile_fused_to_chi``.  This phase lets the eager projector choose the
    layout; the jitted sweeps that follow inherit it.

    Stop rule:

    * ``compare_input=True`` (a warm start whose layout is not the cold-init
      tiling): stable as soon as an eager sweep returns the input's layout, so
      a settled warm start costs one eager sweep.
    * ``compare_input=False`` (a cold start, or any env with the cold-init
      layout): stable when two consecutive eager *outputs* agree; the input
      does not count.  From the rank-1 cold init the first sweep can return
      the initial tiling unchanged and only move it on the second (measured,
      Z2 D=2 chi=8: (0:6, 1:2) -> (0:6, 1:2) -> (0:4, 1:4)).

    Each eager output's chi legs are put in ascending-charge order
    (:func:`_charge_sorted_envs`), so equal layouts give equal pytree
    structures and the jitted consumers do not recompile per call.  The gauge
    is applied only on the stable sweep (it pairs legs index by index, which
    needs one layout); mixing is not applied here.

    Returns ``(envs, n_sweeps, stable)``.
    """
    prev = _chi_layout(envs) if compare_input else None
    for n in range(1, max_sweeps + 1):
        start = envs
        envs, _, _ = _eager_ctm_sweep(site_tensors, start, **sweep_kwargs)
        envs = _charge_sorted_envs(envs)
        layout = _chi_layout(envs)
        if layout == prev:
            if gauge_pair is not None:
                envs = gauge_pair(envs, start)
            return envs, n, True
        prev = layout
    return envs, max_sweeps, False


def python_loop_ctm_converge(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    *,
    chi: int,
    max_iter: int = 100,
    min_iter: int = 10,
    conv_tol: float = 1e-8,
    conv_method: str = "sv",
    renormalize: bool = False,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    chi_ramp: list[tuple[int, int | None]] | None = None,
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    gauge_fix_fn=None,
    plateau_patience: int | None = 20,
    ctmrg_heuristic_increase_chi: bool = False,
    ctmrg_heuristic_increase_chi_threshold: float = 1e-6,
    ctmrg_heuristic_increase_chi_step_size: int = 2,
    chi_max: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
    forward_gauge: str | None = None,
    mixing: float = 0.0,
    eager_layout_sweeps: int = 20,
    _recipe_warning_emitted: bool = False,
) -> tuple[dict[Coord, CTMTensorEnv], CTMConvergeInfo]:
    """Run CTM to convergence using a Python for-loop over JIT'd sweeps.

    Each sweep is JIT-compiled via ``_make_jit_ctm_step``; convergence
    checking happens in Python so no recompilation is needed when
    tolerances change.

    With ``SymmetricTensor`` input the first sweeps run eagerly (see
    ``eager_layout_sweeps``): a jitted sweep cannot change the charge-sector
    layout of the chi legs (#929), so a cold start would otherwise keep the
    initial tiling and converge to the energy of that restricted layout
    (#1098).

    Args:
        site_tensors:      Map from coordinate to iPEPS site tensor.
        neighbors:         Map from coordinate to direction->neighbor coordinate.
        chi:               Final environment bond dimension.
        max_iter:          Maximum CTM iterations (per chi-ramp stage).
        min_iter:          Minimum iterations before checking convergence.
        conv_tol:          Convergence tolerance on corner singular values.
        conv_method:       Convergence method: ``"sv"`` (corner singular
                           values) or ``"elementwise"`` (max element-wise
                           difference across all env tensors).  Prefer
                           ``"sv"``: the environment is defined only up to a
                           gauge on each chi-bond, so ``"elementwise"``
                           measures gauge motion as well as convergence and
                           plateaus far above any usable ``conv_tol`` on a
                           physical state (#780).  Note this default does NOT
                           reach config-driven callers -- ``ctm_converge_kwargs``
                           always emits ``CTMConfig.ctm_conv_method``, which
                           is ``"elementwise"`` for the AD path (#351).
        renormalize:       Renormalize environments after each sweep.
        projector_method:  ``"svd"`` (Fishman, default), ``"eigh"``, or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.
        projector_backward: Backward mode for projector.
        chi_ramp:          Optional chi-ramp schedule as list of
                           ``(stage_chi, max_sweeps)`` tuples.  The last
                           stage uses ``max_iter`` if ``max_sweeps is None``.
        env_init:          Optional initial environments.  If ``None``,
                           identity-initialized environments are created.
        plateau_patience:  Early-bail when the running minimum of the
                           convergence metric (``sv_diff`` or elementwise
                           ``max_diff``) has not improved over the last
                           ``plateau_patience`` iterations.  The loop
                           returns the env that achieved the best metric
                           and ``CTMConvergeInfo.converged=False`` — the
                           bail is a stop-loss, not a fixed point.  On that
                           path ``iterations`` counts the sweeps performed
                           (the bail sweep) while ``best_iteration`` locates
                           the returned env, ``plateau_patience`` sweeps
                           earlier; divide elapsed time by ``iterations``
                           for a per-sweep cost (#781).
                           Default ``20`` is a sane stop-loss against the
                           SU/random-init CTM plateau tracked in #425/#426
                           (memory ``project_tenax_ctm_doesnt_converge_random_init``):
                           a healthy converging run never accumulates 20
                           non-improving iterations because each better
                           ``best_diff`` resets the counter, while a true
                           plateau bails an order of magnitude faster than
                           burning the full ``max_iter`` budget.  Set to
                           ``None`` to restore the pre-2026-05-11 "run to
                           ``max_iter``" behavior.

        forward_gauge:     ``None`` (default): only ``gauge_fix_fn`` applies,
                           as before.  ``"bond_phase"``: the paired,
                           reference-aligned per-bond gauge of the
                           implicit-AD loss (``ad_utils._bond_phase_fix_envs``,
                           #841), so optimizer forwards (warm start,
                           line-search probe, final evaluation) converge to
                           the same element-wise fixed point the loss
                           linearizes.  Exclusive with ``gauge_fix_fn`` and
                           with ``chi_ramp``.
        mixing:            Linear mixing ``beta`` in ``[0, 1)`` (#1060).
                           ``0`` (default) is the plain iteration.  ``beta >
                           0`` iterates ``e <- (1-beta) gauge(step(e)) + beta
                           e``: same fixed points, but a two-state cycle around
                           an unstable fixed point can contract.  Convergence
                           is still tested on the undamped residual.  Needs
                           ``conv_method="elementwise"`` and a gauge
                           (``forward_gauge="bond_phase"`` or
                           ``gauge_fix_fn``).  See
                           ``CTMConvergeInfo.step_multiplier`` for when to
                           reach for it.
        eager_layout_sweeps: Maximum number of eager (un-jitted) sweeps run
                           before the jitted ones when any site or env tensor
                           is a ``SymmetricTensor`` (#1098).  The eager
                           projector keeps the global top-``chi`` singular
                           values and so can move chi slots between charge
                           sectors; the jitted projector keeps each sector's
                           size (#929).  Stop rule: a warm start whose
                           layout is off the cold-init tiling stops after
                           the first eager sweep that returns its layout
                           (one sweep once settled); a cold start, or an env
                           with the cold-init layout, stops when two
                           consecutive eager sweeps agree (so a value of 1
                           cannot certify it, and warns).  If the layout is
                           still changing when this cap or ``max_iter`` ends
                           the phase, a warning is issued and any jitted
                           sweeps continue from there.  Eager outputs are put
                           in ascending-charge order along each chi leg, so
                           the jitted step does not recompile per call.
                           Eager sweeps count toward ``max_iter``,
                           ``min_iter`` and the returned iteration counts.
                           The gauge is applied on the final (stable) eager
                           sweep only, and ``mixing`` only to jitted sweeps.
                           Dense input never enters this phase.  ``0``
                           disables it (the pre-#1098 behaviour).  Applied
                           per stage under ``chi_ramp``.  Not covered: chi
                           slots added by the in-loop chi bump
                           (``ctmrg_heuristic_increase_chi``) are padded
                           inside the jitted loop and keep a frozen layout
                           (follow-up issue to be filed).

    Returns:
        ``(envs, CTMConvergeInfo)`` — converged environments and info.
    """
    if forward_gauge not in (None, "bond_phase"):
        raise ValueError(
            f"python_loop_ctm_converge: forward_gauge={forward_gauge!r}; "
            "only None or 'bond_phase' (other gauges go through gauge_fix_fn)"
        )
    if forward_gauge == "bond_phase" and gauge_fix_fn is not None:
        raise ValueError(
            "python_loop_ctm_converge: forward_gauge='bond_phase' and "
            "gauge_fix_fn are exclusive"
        )
    if forward_gauge == "bond_phase" and chi_ramp is not None:
        raise NotImplementedError(
            "forward_gauge='bond_phase' is not supported with chi_ramp "
            "(the per-stage chi change has no index-by-index reference); "
            "use ctmrg_heuristic_increase_chi, or forward_gauge='phase'."
        )
    # #911: warned here rather than in ``_make_jit_ctm_step`` or
    # ``_python_loop_chi_ramp`` -- this is the once-per-convergence boundary,
    # those are once-per-sweep and once-per-ramp-stage.  Before the chi_ramp
    # early-return below, or a ramped call would never reach it.
    #
    # ``_recipe_warning_emitted`` is private and exists for exactly one
    # caller: ``_python_loop_chi_ramp`` re-enters this function once per ramp
    # stage, so without it a ramped ``recipe="1x1"`` call warns N+1 times --
    # once here and once per stage, with the stage warnings attributed to the
    # internal delegation rather than the user's line.  Placing the warning
    # before the early-return fixed "a ramped call never warns" and created
    # "a ramped call warns per stage"; this closes the second without
    # reopening the first.  (#921 review r4.)
    if recipe == "1x1" and not _recipe_warning_emitted:
        _warn_recipe_1x1_deprecated("python_loop_ctm_converge")
    if eager_layout_sweeps < 0:
        raise ValueError(
            f"eager_layout_sweeps must be >= 0, got {eager_layout_sweeps!r}"
        )

    # Reject ``ctmrg_heuristic_increase_chi`` + ``chi_ramp`` BEFORE the
    # chi_ramp early-return, otherwise the bump flag is silently ignored
    # by ``_python_loop_chi_ramp`` (which doesn't accept the new knobs).
    # CTMConfig also enforces this; defense-in-depth for direct callers
    # (codex review on PR #513).
    if ctmrg_heuristic_increase_chi and chi_ramp is not None:
        raise ValueError(
            "ctmrg_heuristic_increase_chi and chi_ramp are mutually "
            "exclusive: chi_ramp is a deterministic schedule applied "
            "across stages, while ctmrg_heuristic_increase_chi is reactive "
            "inside a single CTM convergence call."
        )
    if chi_ramp is not None:
        return _python_loop_chi_ramp(
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
            gauge_fix_fn=gauge_fix_fn,
            plateau_patience=plateau_patience,
            recipe=recipe,
            device_mesh=device_mesh,
            ctm_chunk_size=ctm_chunk_size,
            mixing=mixing,
            eager_layout_sweeps=eager_layout_sweeps,
        )

    # Build the JIT'd step function (captures neighbors + device_mesh in closure)
    jit_step = _make_jit_ctm_step(
        neighbors, recipe, device_mesh=device_mesh, ctm_chunk_size=ctm_chunk_size
    )
    # Bind chunk_size so callers that don't know about it (e.g.
    # _run_ctm_loop_with_bump) pass it transparently.
    jit_step = partial(jit_step, chunk_size=ctm_chunk_size)

    # chi may grow during the loop when ``ctmrg_heuristic_increase_chi``
    # is enabled (variPEPS-style in-CTM bump; Issue #492).  ``chi_current``
    # is the live value used by JIT'd sweeps; the new chi is static_argname,
    # so each distinct chi value will retrace once on first use.
    #
    # Validation (chi_max required, step_size > 0, env_init chi vs chi_max,
    # chi_max >= chi_current) and env_init-driven chi_current promotion are
    # centralised in ``_validate_chi_bump_args`` so all three forward-CTM
    # entry points (this function, _sigma_gauged_ctm_converge, and
    # ctm_energy_explicit) raise identical errors.
    chi_current = _validate_chi_bump_args(
        chi=chi,
        chi_max=chi_max,
        env_init=env_init,
        bump_enabled=ctmrg_heuristic_increase_chi,
        bump_step_size=ctmrg_heuristic_increase_chi_step_size,
    )

    # Build gauge_fix_fn pair adapter
    if forward_gauge == "bond_phase":
        from tenax.algorithms.ad_utils import (
            _bond_gauge_families,
            _bond_phase_fix_envs_jit,
        )

        _bond_families = _bond_gauge_families(tuple(site_tensors), neighbors)

        def _gauge_pair(envs_new, envs_old):
            return _bond_phase_fix_envs_jit(envs_new, envs_old, _bond_families)
    elif gauge_fix_fn is not None:
        _user_gauge = gauge_fix_fn

        def _gauge_pair(envs_new, envs_old):
            return {c: _user_gauge(envs_new[c]) for c in envs_new}
    else:
        _gauge_pair = None

    # Initialize envs
    envs = (
        env_init
        if env_init is not None
        else {
            c: initialize_ctm_tensor_env(A, chi_current)
            for c, A in site_tensors.items()
        }
    )

    # GSPMD: commit the initial envs onto the device mesh (edges D²-sharded,
    # corners replicated) so the JIT'd sweep's inputs already carry the
    # sharding the per-move constraints chain off.  No-op when device_mesh
    # is None (single-device path is byte-for-byte unchanged).
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import commit_env

        envs = {coord: commit_env(env, device_mesh) for coord, env in envs.items()}

    # QR warm-up: run a few eigh iterations before switching to QR
    warmup = 0
    if projector_method == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs, _, _ = jit_step(
                site_tensors,
                envs,
                chi=chi_current,
                projector_method="eigh",
                renormalize=renormalize,
                projector_backward=projector_backward,
            )

    # #1098: let eager sweeps choose the symmetric chi layout; the jitted
    # sweeps below keep the layout they are handed (#929).
    eager = 0
    eager_budget = min(eager_layout_sweeps, max_iter - warmup)
    if eager_budget > 0 and _needs_eager_layout_phase(site_tensors, envs):
        # A warm start whose layout is already off the cold-init tiling needs
        # one eager sweep to confirm it; a cold(-shaped) env needs two.
        compare_input = env_init is not None and _chi_layout(envs) != _cold_init_layout(
            site_tensors, chi_current
        )
        envs, eager, stable = _adapt_symmetric_layout(
            site_tensors,
            envs,
            max_sweeps=eager_budget,
            gauge_pair=_gauge_pair,
            compare_input=compare_input,
            neighbors=neighbors,
            recipe=recipe,
            device_mesh=device_mesh,
            chi=chi_current,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
            chunk_size=ctm_chunk_size,
        )
        if not stable:
            cause = (
                f"eager_layout_sweeps={eager_layout_sweeps}"
                if eager == eager_layout_sweeps
                else f"the iteration budget (max_iter={max_iter})"
            )
            warnings.warn(
                "python_loop_ctm_converge: the symmetric chi-leg layout was "
                f"still changing when {cause} ended the eager phase after "
                f"{eager} sweeps (#1098).  Any jitted sweeps continue from the "
                "last eager environment and keep its layout, so the result "
                "may be the energy of a restricted layout; raise "
                "eager_layout_sweeps or max_iter.",
                stacklevel=2,
            )
    pre = warmup + eager

    # Run the bump-aware loop via shared helper.
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
        max_iter=max_iter - pre,
        min_iter=max(0, min_iter - pre),
        conv_tol=conv_tol,
        conv_method=conv_method,
        plateau_patience=plateau_patience,
        mixing=mixing,
    )

    return result.envs, CTMConvergeInfo(
        converged=result.converged,
        iterations=pre + result.iterations,
        sv_diff=result.sv_diff,
        max_truncation_error=result.max_truncation_error,
        max_smallest_S=result.max_smallest_S,
        final_chi=result.final_chi,
        best_iteration=pre + result.best_iteration,
        step_multiplier=result.step_multiplier,
    )


def _python_loop_chi_ramp(
    site_tensors: dict[Coord, Tensor],
    neighbors: dict[Coord, dict[str, Coord]],
    *,
    chi: int,
    max_iter: int,
    min_iter: int,
    conv_tol: float,
    conv_method: str,
    renormalize: bool,
    projector_method: str,
    qr_warmup_steps: int,
    projector_backward: str,
    chi_ramp: list[tuple[int, int | None]],
    env_init: dict[Coord, CTMTensorEnv] | None,
    gauge_fix_fn=None,
    plateau_patience: int | None = None,
    recipe: str = "2x2",
    device_mesh=None,
    ctm_chunk_size: int | None = None,
    mixing: float = 0.0,
    eager_layout_sweeps: int = 20,
) -> tuple[dict[Coord, CTMTensorEnv], CTMConvergeInfo]:
    """Run CTM with chi-ramp schedule."""
    envs = env_init
    prev_chi: int | None = None
    info = CTMConvergeInfo(
        converged=False, iterations=0, sv_diff=float("inf"), max_truncation_error=0.0
    )

    for stage_idx, (stage_chi, stage_sweeps) in enumerate(chi_ramp):
        is_last = stage_idx == len(chi_ramp) - 1

        # Determine iteration budget for this stage
        if stage_sweeps is not None:
            stage_max = stage_sweeps
        else:
            stage_max = max_iter

        # Determine convergence tolerance: only converge on last stage
        stage_tol = conv_tol if is_last else 0.0

        # Re-initialize when chi changes
        if prev_chi is not None and stage_chi != prev_chi:
            envs = None

        # Disable plateau early-bail when the user pinned an explicit
        # warm-up budget for a non-final stage: a ramp like
        # ``[(8, 100), (16, None)]`` is a contract to spend exactly 100
        # sweeps at chi=8 before moving on, so the next stage sees a
        # consistent warm-start env regardless of the plateau metric
        # (codex review on PR #439).  The final stage and any stage with
        # ``stage_sweeps=None`` still honor ``plateau_patience``.
        stage_patience = (
            None if (not is_last and stage_sweeps is not None) else plateau_patience
        )

        envs, info = python_loop_ctm_converge(
            site_tensors,
            neighbors,
            chi=stage_chi,
            max_iter=stage_max,
            min_iter=min_iter,
            conv_tol=stage_tol,
            conv_method=conv_method,
            renormalize=renormalize,
            projector_method=projector_method,
            qr_warmup_steps=qr_warmup_steps if stage_idx == 0 else 0,
            projector_backward=projector_backward,
            chi_ramp=None,
            env_init=envs,
            gauge_fix_fn=gauge_fix_fn,
            plateau_patience=stage_patience,
            recipe=recipe,
            # Stage calls re-enter the function that already warned for
            # this convergence; without this a ramped 1x1 run emits N+1
            # deprecations, N of them attributed here (#921 review r4).
            _recipe_warning_emitted=True,
            device_mesh=device_mesh,
            ctm_chunk_size=ctm_chunk_size,
            mixing=mixing,
            eager_layout_sweeps=eager_layout_sweeps,
        )
        prev_chi = stage_chi

    return envs, info
