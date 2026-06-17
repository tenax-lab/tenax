"""#615 uniform-sector env: flag-gated Gate-B measurement scaffold."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from examples.probe_backward_jaxpr_566 import backward_vjp_jaxpr
from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair


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
    """Charges on the chi-bond legs of ``t``.

    D² legs are labelled with a trailing '2' (u2/d2/l2/r2); every other leg
    is a chi bond. So a chi leg is any leg whose label does not end in '2'.
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


def test_forward_env_block_counts_drop_under_keep():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env0, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6)
    n0 = {n: len(getattr(env0, n).blocks) for n in env0._fields}
    env1, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6, keep_sectors={-1, 0, 1})
    n1 = {n: len(getattr(env1, n).blocks) for n in env1._fields}
    for name in n0:
        assert n1[name] < n0[name], f"{name}: {n1[name]} !< {n0[name]}"
    for name in env1._fields:
        assert _chi_sectors(getattr(env1, name)) <= {-1, 0, 1}


def test_cold_backward_vjp_builds_under_keep():
    """#615 make-or-break: the 2x2 multisite backward VJP — which raised
    ValueError('Size of label d ...') in the #610 spike — now traces cold under
    the sector drop. This is the structural gate the whole issue exists to clear.
    """
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    jax.clear_caches()  # COLD: do not reuse a flag-off jaxpr (cold-trace caveat)
    with keep_sectors_context({-1, 0, 1}):
        jaxpr = backward_vjp_jaxpr(A, chi=12)
    assert jaxpr is not None
    assert len(jaxpr.eqns) > 0


def test_keep_env_is_faithful_at_d3_chi12():
    """#615 faithfulness guard: the keep-active CTM env must be VALID before the
    Gate-B op-count measurement is trustworthy — it must converge, be finite, and
    give a sane (random-init, not ground-state) Heisenberg energy."""
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7, keep_sectors={-1, 0, 1})
    for name in env._fields:
        assert np.all(np.isfinite(np.asarray(getattr(env, name)._data))), f"{name} non-finite"
    e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e), f"energy {e} non-finite"
    assert -2.0 < e < 0.0, f"energy {e} outside sane Heisenberg window (-2, 0)"
