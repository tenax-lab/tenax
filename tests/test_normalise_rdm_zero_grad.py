"""``_normalise_rdm`` must stay differentiable at an all-zero RDM.

The zero matrix is a *supported* input, not a corner case: a zero excitation
vector ``B = 0`` must have zero norm, and ``_normalise_rdm``'s floor is written
to keep that ``0/0`` from becoming ``NaN`` (see the docstring on the helper).
Guarding only the forward value is not enough, because the excitation
``H_eff``/``N`` matrices are built by *differentiating* those norms —
``jax.vmap(jax.grad(norm_fn))`` in ``_build_effective_matrices`` — so a NaN
cotangent poisons the quasiparticle spectrum just as effectively as a NaN
value would.

The trap is a JAX one: ``jnp.linalg.norm`` has a NaN derivative at the
all-zero matrix (``x/||x||`` is ``0/0``), and a surrounding ``jnp.where`` does
**not** short-circuit that VJP — the unselected branch still contributes
``0 * NaN = NaN``.  The fix keeps ``sqrt`` away from zero via a second
``where`` on its argument.  These tests pin the gradient, since the value was
already correct while the gradient was not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_energy import _normalise_rdm

DTYPES = [jnp.float64, jnp.complex128]


def _sum_real(flat, dtype):
    return jnp.sum(_normalise_rdm(flat.reshape(2, 2).astype(dtype))).real


@pytest.mark.parametrize("dtype", DTYPES, ids=["real", "complex"])
def test_gradient_at_the_zero_matrix_is_finite(dtype):
    """The documented zero-RDM branch must have a usable cotangent."""
    zeros = jnp.zeros(4, dtype=dtype)
    g = jax.grad(_sum_real)(zeros, dtype)
    assert not bool(jnp.isnan(g).any()), (
        f"NaN cotangent at the all-zero RDM ({dtype.__name__}): {g}. "
        "jnp.where does not short-circuit the norm() VJP."
    )
    assert bool(jnp.isfinite(g).all()), f"non-finite cotangent: {g}"


@pytest.mark.parametrize("dtype", DTYPES, ids=["real", "complex"])
def test_value_at_the_zero_matrix_is_still_exactly_zero(dtype):
    """The forward contract the floor exists for is unchanged by the fix."""
    out = _normalise_rdm(jnp.zeros((2, 2), dtype=dtype))
    assert bool(jnp.all(out == 0)), f"zero RDM did not normalise to zero: {out}"


@pytest.mark.parametrize("dtype", DTYPES, ids=["real", "complex"])
def test_the_nonzero_path_is_untouched(dtype):
    """A generic RDM must keep the gradient it had before the zero guard.

    Pinned against finite differences rather than against a golden array so
    this stays a statement about the function, not about a captured run.
    """
    x = jnp.array([1.0, 0.2, 0.3, 0.9], dtype=dtype)
    g = jax.grad(_sum_real)(x, dtype)
    assert bool(jnp.isfinite(g).all()), f"non-finite cotangent on a generic RDM: {g}"

    h = 1e-6
    for i in range(4):
        step = jnp.zeros(4, dtype=dtype).at[i].set(h)
        fd = (_sum_real(x + step, dtype) - _sum_real(x - step, dtype)) / (2 * h)
        assert abs(float(g[i].real) - float(fd)) < 1e-6 * max(abs(float(fd)), 1.0), (
            f"component {i}: AD {g[i]} vs FD {fd}"
        )


def test_gradient_is_finite_just_above_and_below_the_floor():
    """The branch boundary itself must not produce a NaN either.

    ``|tr|`` crossing ``EPS * ||mat||`` is where the two branches meet; a guard
    that is only checked at exactly zero would miss a NaN introduced on the
    selected-but-adjacent side.
    """
    from tenax.algorithms._ctm_tensor_energy import EPS

    # Traceless up to a controlled amount, so |tr| straddles the floor.
    for factor in (0.1, 1.0, 10.0):
        tr_target = factor * EPS * 2.0
        x = jnp.array([tr_target, 1.0, 1.0, 0.0])
        g = jax.grad(_sum_real)(x, jnp.float64)
        assert bool(jnp.isfinite(g).all()), (
            f"non-finite cotangent at |tr| = {factor}x the floor: {g}"
        )


def test_excitation_norm_gradient_at_zero_B_is_finite():
    """Reachability: the production path that differentiates a zero-B norm.

    ``_build_effective_matrices`` runs ``jax.vmap(jax.grad(norm_fn))``; this is
    that call at the one input the guard was added for.

    Deliberately *not* marked ``slow``: it is the test that proves the unit
    tests above are about a reachable defect rather than a hypothetical one,
    and at D=2 / chi=8 it costs a few seconds.  Under the ``core`` mapping in
    ``conftest.py`` an explicit ``slow`` would withhold that marker and keep
    this out of the required CI gate.
    """
    from tenax.algorithms.ipeps_config import CTMConfig
    from tenax.algorithms.ipeps_ctm import ctm
    from tenax.algorithms.ipeps_excitations import _compute_norm

    d = 2
    A = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 2, 2, d))
    A = A / (jnp.linalg.norm(A) + 1e-10)
    env = ctm(A, CTMConfig(chi=8, max_iter=40))
    k = jnp.array([0.0, 0.0])

    g = jax.grad(lambda B: _compute_norm(A, B, env, k, d).real)(jnp.zeros_like(A))
    assert not bool(jnp.isnan(g).any()), (
        "grad of the excitation norm at B = 0 is NaN; _build_effective_matrices "
        "vmaps exactly this gradient"
    )
