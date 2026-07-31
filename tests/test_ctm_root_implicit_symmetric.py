"""Symmetric root-implicit CTMRG gradient (#715 Phase 3, 1x1 bosonic abelian)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry
from tenax.algorithms._ctm_root_implicit_symmetric import (
    SymEnv,
    init_env_sym,
    swap_env_convention_sym,
)


def _site_tensor(seed: int = 0) -> SymmetricTensor:
    """A U(1) iPEPS site tensor with non-trivial charges on every leg.

    Non-trivial deliberately: a trivial-charge tensor has one block, so every
    layout bug is invisible.  The fused ``D**2`` leg comes out with sector
    multiplicities ``[1, 2, 1]`` — *unequal*, which is the fragmenting case
    and the only one that exercises the layout arithmetic.

    **D=2 with virtual sectors [0, 1], and the physical leg labelled
    ``phys``.** Both are load-bearing and were measured, not guessed:

    - At D=3 (virtual sectors ``[-1, 0, 1]``) ``initialize_ctm_tensor_env``
      raises ``ValueError: data.shape (4, 4, 4) does not match index dims
      (4, 9, 4)``.  That is issue #667 — ``_CORNER_SPECS`` gives one
      ``ref_axis`` per corner, so an env leg derives its charges from a
      direction it does not physically touch.  A production gap, deferred;
      do not try to fix it here.
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
            phys,
            virt(FlowDirection.IN, "u"),
            virt(FlowDirection.OUT, "d"),
            virt(FlowDirection.IN, "l"),
            virt(FlowDirection.OUT, "r"),
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
