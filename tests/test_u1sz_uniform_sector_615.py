"""#615 uniform-sector env: flag-gated Gate-B measurement scaffold."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair


def _env_block_signature(env):
    """A structural fingerprint: per-tensor sorted (block-key, shape) list."""
    sig = {}
    for name in env._fields:
        t = getattr(env, name)
        sig[name] = sorted(
            (tuple(int(q) for q in k), tuple(int(s) for s in b.shape))
            for k, b in t.blocks.items()
        )
    return sig


def test_keep_sectors_none_is_identity():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env_a, e_a = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4)
    env_b, e_b = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4, keep_sectors=None)
    assert _env_block_signature(env_a) == _env_block_signature(env_b)
    assert float(e_a) == float(e_b)
