import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_config import CTMConfig

pytestmark = pytest.mark.core


def test_fuse_virtual_legs_defaults_true():
    cfg = CTMConfig(chi=8)
    assert cfg.fuse_virtual_legs is True


def test_fuse_virtual_legs_can_disable():
    cfg = CTMConfig(chi=8, fuse_virtual_legs=False)
    assert cfg.fuse_virtual_legs is False


from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_site(D, d, seed):
    """Single-site U(1)-trivial DenseTensor with labels (u,d,l,r,phys)."""
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, d))
    data = data / jnp.linalg.norm(data)
    sym = U1Symmetry()
    zD = np.zeros(D, dtype=np.int32)
    zd = np.zeros(d, dtype=np.int32)
    idx = [
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, zD.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, zd.copy(), FlowDirection.IN, label="phys"),
    ]
    return DenseTensor(data, idx)


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


@pytest.mark.parametrize("D", [2, 3])
def test_split_explicit_energy_and_grad_finite(D):
    """ctm_energy_split_explicit returns finite energy and gradient.

    Uses chi = D² (lossless, avoids rank-deficient SVD backward NaN) with
    few backprop steps (no warm-start degeneracy).  This is a smoke/sanity
    test — fused vs split use different projectors and converge to different
    fixed points so gradient parity is not asserted.
    """
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_explicit

    chi = D * D  # lossless: rank of D×D corner ≤ D²
    A = _make_site(D, 2, seed=7)
    gate = _heisenberg_gate()

    def loss(a):
        return ctm_energy_split_explicit(
            {(0, 0): a},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            warmup_steps=2,
            backprop_steps=3,
            chi_I=chi,
        ).real

    e, g = jax.value_and_grad(loss)(A)

    assert jnp.isfinite(e), f"energy is not finite: {e}"
    gs = jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    assert jnp.all(jnp.isfinite(gs)), "gradient contains non-finite values"
    assert float(jnp.sum(jnp.abs(gs))) > 0, "gradient is identically zero"


def test_split_explicit_raises_for_multisite():
    """ctm_energy_split_explicit raises NotImplementedError for >1 site."""
    from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_explicit

    A = _make_site(2, 2, seed=0)
    gate = _heisenberg_gate()

    with pytest.raises(NotImplementedError, match="single-site"):
        ctm_energy_split_explicit(
            {(0, 0): A, (1, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=4,
        )
