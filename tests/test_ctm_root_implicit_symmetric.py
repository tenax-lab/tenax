"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry
from tenax.algorithms._ctm_root_implicit_symmetric import (
    SymEnv,
    half_infinite_sym,
    init_env_sym,
    lower_left_quadrant_sym,
    swap_env_convention_sym,
    upper_left_quadrant_sym,
)


def _site_tensor(seed: int = 0) -> SymmetricTensor:
    """A U(1) iPEPS site tensor with non-trivial charges on every leg.

    Non-trivial deliberately: a trivial-charge tensor has one block, so every
    layout bug is invisible.  The fused ``D**2`` leg comes out with sector
    multiplicities ``[1, 2, 1]`` — *unequal*, which is the fragmenting case
    and the only one that exercises the layout arithmetic.

    **Axis order is ``(u, d, l, r, phys)`` — physical leg last.**  That is the
    library-wide iPEPS convention (``_build_double_layer_tensor``'s docstring,
    ``_ctm_utils`` line 62, ``ipeps_optimize``, ``pess_optimize``, ...) and it
    is load-bearing here, not cosmetic.  ``_build_double_layer_tensor``
    contracts by *label* and so is order-blind, but ``_CORNER_SPECS`` and
    ``_STD_EDGE_SPECS`` name their ``ref_axis`` *positionally* (0 = u, 1 = d).
    With ``phys`` first, ``ref_axis=0`` lands on the physical leg, and every
    corner/edge chi leg that quotes it inherits ``phys (x) phys-bar`` charges
    instead of virtual ones.  Measured consequences of getting this wrong:

    - The environment stops being uniform.  With ``phys`` first, ``T4``'s two
      chi legs come out with charges ``[-2, 0, 0, 2]`` and ``[-1, 0, 0, 1]``,
      so the two half-infinite quadrants cannot be glued along the cut — the
      contraction that ``half_infinite_sym`` has to do is not merely
      awkward, it has no charge-consistent form.  With ``phys`` last every
      chi leg carries ``[-1, 0, 0, 1]`` and the cut closes.
    - At D=3 (virtual sectors ``[-1, 0, 1]``) ``phys`` first makes
      ``initialize_ctm_tensor_env`` raise ``ValueError: data.shape (4, 4, 4)
      does not match index dims (4, 9, 4)`` — ``ref_axis=0`` is the d=2
      physical leg while ``ref_axis=1`` is a D=3 virtual one.  That was read
      as issue #667; it is really just this axis order.

    **D=2 with virtual sectors [0, 1], and the physical leg labelled
    ``phys``.** Both are load-bearing and were measured, not guessed:

    - Non-trivial charges deliberately: a trivial-charge tensor has one
      block, so every layout bug is invisible.
    - With any physical label other than ``phys``,
      ``compute_energy_ctm_tensor`` raises ``ValueError: output_labels
      contains 'phys' which is not a free label``.
    """
    sym = U1Symmetry()
    phys = TensorIndex(
        symmetry=sym,
        sectors=np.array([-1, 1]),
        multiplicities=np.array([1, 1]),
        flow=FlowDirection.OUT,
        label="phys",
    )

    def virt(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([0, 1]),
            multiplicities=np.array([1, 1]),
            flow=flow,
            label=lbl,
        )

    return SymmetricTensor.random_normal_np(
        (
            virt(FlowDirection.IN, "u"),
            virt(FlowDirection.OUT, "d"),
            virt(FlowDirection.IN, "l"),
            virt(FlowDirection.OUT, "r"),
            phys,
        ),
        np.random.RandomState(seed),
    )


def test_init_env_sym_keeps_every_tensor_symmetric():
    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    assert isinstance(env, SymEnv)
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        t = getattr(env, name)
        assert isinstance(t, SymmetricTensor), f"{name} was densified"
        assert t.n_blocks > 0
    assert isinstance(a, SymmetricTensor)


def test_swap_env_convention_sym_is_an_involution():
    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    twice = swap_env_convention_sym(swap_env_convention_sym(env))
    for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
        lhs = getattr(twice, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name


def test_swap_env_convention_sym_is_not_identity_on_asymmetric_env():
    """``init_env_sym``'s C4 is symmetric and T3/T4 palindromic by
    construction, so the involution test above would pass even if
    ``swap_env_convention_sym`` returned its input unchanged.  Replace C4,
    T3, T4 with fresh random tensors on the *same* indices — still valid
    ``SymmetricTensor``s, but generically neither symmetric nor palindromic —
    and check the swap actually transposes them rather than no-op'ing.
    """
    A = _site_tensor()
    env, _a = init_env_sym(A, chi=4)
    env = env._replace(
        C4=SymmetricTensor.random_normal_np(env.C4.indices, np.random.RandomState(1)),
        T3=SymmetricTensor.random_normal_np(env.T3.indices, np.random.RandomState(2)),
        T4=SymmetricTensor.random_normal_np(env.T4.indices, np.random.RandomState(3)),
    )
    swapped = swap_env_convention_sym(env)

    # Not the identity: the swapped C4/T3/T4 differ from the originals.
    assert float(jnp.max(jnp.abs(swapped.C4.todense() - env.C4.todense()))) > 1e-6
    assert float(jnp.max(jnp.abs(swapped.T3.todense() - env.T3.todense()))) > 1e-6
    assert float(jnp.max(jnp.abs(swapped.T4.todense() - env.T4.todense()))) > 1e-6

    # And it is exactly the documented transpose, not some other change.
    assert (
        float(
            jnp.max(jnp.abs(swapped.C4.todense() - env.C4.transpose((1, 0)).todense()))
        )
        < 1e-14
    )
    assert (
        float(
            jnp.max(
                jnp.abs(swapped.T3.todense() - env.T3.transpose((2, 1, 0)).todense())
            )
        )
        < 1e-14
    )
    assert (
        float(
            jnp.max(
                jnp.abs(swapped.T4.todense() - env.T4.transpose((2, 1, 0)).todense())
            )
        )
        < 1e-14
    )

    # The untouched tensors are unaffected.
    for name in ("C1", "C2", "C3", "T1", "T2"):
        lhs = getattr(swapped, name).todense()
        rhs = getattr(env, name).todense()
        assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-14, name


def _dense_env_of(env):
    from tenax.algorithms._ctm_root_implicit_asym import AsymEnv

    return AsymEnv(
        *[
            jnp.asarray(getattr(env, n).todense())
            for n in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
        ]
    )


def test_upper_left_quadrant_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import _upper_left_quadrant

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = _upper_left_quadrant(_dense_env_of(env), jnp.asarray(a.todense()))
    got = upper_left_quadrant_sym(env, a)
    assert got.labels() == ("chi_r", "a_r", "chi_d", "a_d")
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-12


def test_lower_left_quadrant_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import _lower_left_quadrant

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = _lower_left_quadrant(_dense_env_of(env), jnp.asarray(a.todense()))
    got = lower_left_quadrant_sym(env, a)
    assert got.labels() == ("chi_u", "a_u", "chi_r", "a_r")
    assert float(jnp.max(jnp.abs(got.todense() - expected))) < 1e-12


def test_half_infinite_sym_matches_the_dense_module():
    from tenax.algorithms._ctm_root_implicit_asym import half_infinite_environment

    A = _site_tensor()
    env, a = init_env_sym(A, chi=4)
    expected = half_infinite_environment(_dense_env_of(env), jnp.asarray(a.todense()))
    got = half_infinite_sym(env, a)
    assert got.ndim == 2
    dense = got.todense()
    assert dense.shape == expected.shape
    assert float(jnp.max(jnp.abs(dense - expected))) < 1e-10
