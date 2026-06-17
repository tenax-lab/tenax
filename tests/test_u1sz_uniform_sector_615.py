"""#615 uniform-sector env: flag-gated Gate-B measurement scaffold."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context
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


def _chi_sectors(t):
    """Set of charges appearing on chi-bond legs of tensor ``t``.

    Corners: all legs are chi-bond legs (labels like c1_d, c1_r).
    Edges: chi legs are the first and last; the D² middle leg has a label
    ending in '2' (u2, r2, d2, l2).  We identify chi legs as those whose
    label does NOT end with '2'.
    """
    out = set()
    for ix in t.indices:
        if not ix.label.lower().endswith("2"):
            out |= {int(q) for q in ix.charges}
    return out


def test_keep_sectors_none_is_identity():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env_a, e_a = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4)
    env_b, e_b = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4, keep_sectors=None)
    assert _env_block_signature(env_a) == _env_block_signature(env_b)
    assert float(e_a) == float(e_b)


def test_env_init_seeds_only_keep_sectors():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    # Baseline: chi legs carry the full {-2..+2}.
    env0 = initialize_ctm_tensor_env(A, chi=12)
    base = set().union(*(_chi_sectors(getattr(env0, n)) for n in env0._fields))
    assert 2 in base or -2 in base, "baseline should carry |Sz|=2 chi sectors"
    # Under keep={-1,0,1}, no chi leg may carry |Sz|=2.
    with keep_sectors_context({-1, 0, 1}):
        env = initialize_ctm_tensor_env(A, chi=12)
    for n in env._fields:
        assert _chi_sectors(getattr(env, n)) <= {-1, 0, 1}, f"{n} kept |Sz|=2"
