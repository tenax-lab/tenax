"""HoneycombCTMEnv shape and pytree behavior."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_honeycomb_env import HoneycombCTMEnv


def _dummy_tensor(shape):
    return jnp.zeros(shape, dtype=jnp.complex128)


def test_env_has_nine_fields():
    env = HoneycombCTMEnv(
        C0=_dummy_tensor((4, 4)),
        C1=_dummy_tensor((4, 4)),
        C2=_dummy_tensor((4, 4)),
        L0=_dummy_tensor((4, 9, 4)),
        L1=_dummy_tensor((4, 9, 4)),
        L2=_dummy_tensor((4, 9, 4)),
        R0=_dummy_tensor((4, 9, 4)),
        R1=_dummy_tensor((4, 9, 4)),
        R2=_dummy_tensor((4, 9, 4)),
    )
    assert env.C0.shape == (4, 4)
    assert env.L1.shape == (4, 9, 4)
    assert env.R2.shape == (4, 9, 4)


def test_env_is_pytree():
    """jax.tree_util.tree_map should iterate the 9 fields."""
    env = HoneycombCTMEnv(
        *[_dummy_tensor((4, 4)) for _ in range(3)],
        *[_dummy_tensor((4, 9, 4)) for _ in range(6)],
    )
    leaves = jax.tree_util.tree_leaves(env)
    assert len(leaves) == 9
    incremented = jax.tree_util.tree_map(lambda x: x + 1.0, env)
    assert jnp.all(incremented.C0 == 1.0)
