"""#983: the 2x2 CTM projector's response to A must reach the gradient.

``_compute_2x2_projector`` returned ``stop_gradient(P_top), ``
``stop_gradient(P_bot)`` unconditionally, so ``dP/dA`` was dropped from
every gradient taken through the default ``recipe="2x2"`` path and the
``projector_backward`` knob had no effect there at all.  It is now honoured:
``"flow"`` lets the response through.  That is opt-in, not the default --
see ``test_everything_but_flow_keeps_the_stop_gradient`` for why.

These tests measure ONE CTM sweep, not a converged fixed point.  That is
deliberate: at a fixed point the comparison is confounded by #841 (the 2x2
forward is not an element-wise fixed point at D=3, so an FD-vs-AD ratio
there measures #841's non-stationary forward, not this defect).  A single
sweep has no fixed-point premise, so the FD reference is unambiguous.

The functional is the singular spectrum of ``C1``, which is invariant under
the chi-bond gauge ``C1 -> U C1 V`` that the projector SVD leaves free.  A
gauge-*dependent* functional is genuinely discontinuous in A and its finite
difference measures gauge jumps rather than a derivative -- an earlier draft
of this probe used a random linear functional of the env tensors and read
FD ~ 1e6 on a function of value ~5 for exactly that reason.

``H_FD = 1e-6`` sits on a plateau: the central difference is stable to 6+
significant digits across ``h`` in ``[1e-7, 1e-4]`` at these points, which
is what certifies the reference as a derivative rather than a jump.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import (
    SINGLE_SITE_NEIGHBORS,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor

H_FD = 1e-6
WARM_SWEEPS = 6


def _warm_env(A_data, chi):
    """Converge an env a few sweeps off the rank-1 cold init, then freeze it."""
    A0 = _wrap_as_dense_tensor(A_data)
    envs = {(0, 0): initialize_ctm_tensor_env(A0, chi)}
    dl = {(0, 0): _build_double_layer_tensor(A0)}
    for _ in range(WARM_SWEEPS):
        envs, _, _ = _ctm_tensor_sweep_multisite(
            envs, dl, SINGLE_SITE_NEIGHBORS, chi, True, "svd"
        )
    return jax.tree.map(jax.lax.stop_gradient, envs[(0, 0)])


def _random_site(D, seed, d=2):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((D, D, D, D, d)))
    return A / jnp.linalg.norm(A)


def _sweep_functional(A_data, env_warm, chi, projector_backward):
    """Gauge-invariant scalar of ONE sweep: weighted singular spectrum of C1."""
    out, _, _ = _ctm_tensor_sweep_multisite(
        {(0, 0): env_warm},
        {(0, 0): _build_double_layer_tensor(_wrap_as_dense_tensor(A_data))},
        SINGLE_SITE_NEIGHBORS,
        chi,
        True,
        "svd",
        projector_backward=projector_backward,
    )
    s = jnp.linalg.svd(out[(0, 0)].C1.todense(), compute_uv=False)
    w = jnp.arange(1, s.shape[0] + 1, dtype=s.dtype)
    return jnp.sum(w * s / (s[0] + 1e-30))


def _ad_fd_ratios(A_data, chi, projector_backward, n_dirs=2):
    env_warm = _warm_env(A_data, chi)

    def f(p):
        return _sweep_functional(p, env_warm, chi, projector_backward)

    g = jax.grad(f)(A_data)
    ratios = []
    for i in range(n_dirs):
        v = np.random.default_rng(100 + i).standard_normal(A_data.shape)
        v = jnp.asarray(v / np.linalg.norm(v))
        fd = (float(f(A_data + H_FD * v)) - float(f(A_data - H_FD * v))) / (2 * H_FD)
        ad = float(jnp.sum(g * v))
        ratios.append(ad / fd)
    return g, ratios


@pytest.mark.parametrize("seed", [2, 7])
def test_2x2_sweep_gradient_matches_finite_difference(seed):
    """One 2x2 sweep's AD gradient must agree with its finite difference.

    Measured before the fix (stop_gradient in place): seed 7 gave ratios
    0.032 / 0.654, seed 2 gave -0.261 / -0.250 -- wrong by up to 31x and,
    at seed 2, wrong in SIGN.  With ``dP/dA`` restored both land on 1.000.
    """
    A_data = _random_site(D=2, seed=seed)
    g, ratios = _ad_fd_ratios(A_data, chi=4, projector_backward="flow")

    assert jnp.all(jnp.isfinite(g)), "gradient is not finite"
    for i, r in enumerate(ratios):
        assert abs(r - 1.0) < 0.05, (
            f"seed={seed} dir={i}: AD/FD ratio {r:.6f} is not 1 -- the 2x2 "
            f"projector response dP/dA is missing from the gradient (#983)"
        )


def test_2x2_sweep_gradient_finite_on_rank_deficient_corner():
    """A rank-deficient half must not produce a NaN gradient.

    seed=3 at D=2, chi=4 drives ``_fishman_truncate_S`` to zero 2 of the 16
    singular values of a half, so the SVD backward's
    ``F_ij = 1/(s_i^2 - s_j^2)`` hits ``1/(0-0)`` on the zero multiplet.
    Unregularized, that yields 32 NaN entries.  This pins the regularized
    backward, not merely the removal of the stop_gradient.

    Note the trigger is EXACT rank deficiency, not near-degeneracy: seed=2
    above carries a 2.2e-06 adjacent gap and is perfectly finite raw.
    """
    A_data = _random_site(D=2, seed=3)
    g, ratios = _ad_fd_ratios(A_data, chi=4, projector_backward="flow")

    assert jnp.all(jnp.isfinite(g)), (
        f"{int(jnp.sum(jnp.isnan(g)))} NaN entries -- the zero multiplet of a "
        f"rank-deficient Fishman half reached an unregularized SVD backward"
    )
    for i, r in enumerate(ratios):
        assert abs(r - 1.0) < 0.10, f"dir={i}: AD/FD ratio {r:.6f}"


@pytest.mark.parametrize("projector_backward", ["auto", "none", "lorentzian"])
def test_everything_but_flow_keeps_the_stop_gradient(projector_backward):
    """Only ``"flow"`` unfreezes the projectors; every other value must not.

    ``"flow"`` is opt-in rather than default because restoring ``dP/denv``
    puts the CTM gauge mode back into ``J``, and the implicit-AD adjoint
    ``(I - J^T) λ = dE/denv`` then has no reliable solution -- residual
    4.5e-15 at ctm_max_iter=40 but 7.9e-01 at 300, flat in both Krylov
    restart and maxiter.  See ``_PROJECTOR_BACKWARD_FLOW``.

    This pins that the default path is unchanged, and that the value
    ``"lorentzian"`` -- which ``ctm_energy_implicit`` passes by default, and
    which selects a regularized *eigh* backward on the 1x1 recipe -- does
    NOT accidentally unfreeze the 2x2 projectors.
    """
    A_data = _random_site(D=2, seed=7)
    _, ratios = _ad_fd_ratios(A_data, chi=4, projector_backward=projector_backward)

    assert max(abs(r - 1.0) for r in ratios) > 0.1, (
        f"projector_backward={projector_backward!r} produced FD-consistent "
        f"ratios {ratios}; only 'flow' is supposed to unfreeze the projectors"
    )


# --------------------------------------------------------------------- #
# The option has to survive the trip from the public API to the          #
# projector.  Both gaps below were found by review on the first revision #
# of this PR, and both would have left the fix above unreachable.        #
# --------------------------------------------------------------------- #


def test_flow_survives_the_config_and_the_jit_boundary():
    """``CTMConfig`` must accept ``"flow"`` and round-trip it.

    ``optimize_gs_ad`` reaches the CTM through ``CTMConfig``, and the config
    is packed into a hashable tuple at the JIT boundary by
    ``_config_to_tuple``.  A value missing from either the validation set or
    the ``_PB_*`` maps is rejected outright or -- worse -- silently encoded
    as ``0 == "auto"`` by the map's ``.get`` default, which re-freezes the
    projectors exactly where the fix is supposed to apply.
    """
    from tenax.algorithms.ad_utils import _config_from_tuple, _config_to_tuple
    from tenax.algorithms.ipeps_config import CTMConfig

    for value in ("auto", "standard", "lorentzian", "flow"):
        cfg = CTMConfig(projector_backward=value)
        assert cfg.projector_backward == value
        round_tripped = _config_from_tuple(_config_to_tuple(cfg))
        assert round_tripped.projector_backward == value, (
            f"projector_backward={value!r} degraded to "
            f"{round_tripped.projector_backward!r} at the JIT boundary"
        )

    with pytest.raises(ValueError, match="projector_backward must be one of"):
        CTMConfig(projector_backward="not-a-mode")


def test_flow_reaches_the_split_2x2_projector_too():
    """The split recipe calls the same projector and must honour the option.

    ``_compute_split_plaquette_projector_pair`` used to call
    ``_compute_2x2_projector`` without forwarding ``projector_backward``, so
    the split 2x2 path silently kept the frozen default no matter what the
    caller asked for.  Asserts the forward is untouched and the gradient
    genuinely moves.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_sweep_multisite,
    )
    from tenax.algorithms._split_ctm_tensor_init import (
        initialize_split_ctm_tensor_env,
    )

    chi = 4
    A_data = _random_site(D=2, seed=5)
    env0 = jax.tree.map(
        jax.lax.stop_gradient,
        initialize_split_ctm_tensor_env(_wrap_as_dense_tensor(A_data), chi, chi),
    )

    def f(p, projector_backward):
        A = _wrap_as_dense_tensor(p)
        out = _split_ctm_sweep_multisite(
            {(0, 0): env0},
            {(0, 0): A},
            {(0, 0): A.bar()},
            SINGLE_SITE_NEIGHBORS,
            chi,
            chi,
            True,
            "2x2",
            projector_backward=projector_backward,
        )[(0, 0)]
        s = jnp.linalg.svd(out.C1.todense(), compute_uv=False)
        w = jnp.arange(1, s.shape[0] + 1, dtype=s.dtype)
        return jnp.sum(w * s / (s[0] + 1e-30))

    assert float(f(A_data, "auto")) == float(f(A_data, "flow")), (
        "the forward moved -- only the VJP may change"
    )

    g_auto = jax.grad(lambda p: f(p, "auto"))(A_data)
    g_flow = jax.grad(lambda p: f(p, "flow"))(A_data)
    assert jnp.all(jnp.isfinite(g_flow))
    assert float(jnp.linalg.norm(g_flow - g_auto)) > 1e-10, (
        "'flow' changed nothing on the split recipe -- projector_backward is "
        "not reaching _compute_2x2_projector through the split call chain"
    )
