from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_honeycomb_init import _double_layer_honeycomb
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Build a rank-4 honeycomb site tensor with labels (e0, e1, e2, phys)."""
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


def test_double_layer_shape_and_labels():
    A = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(0))
    T = _double_layer_honeycomb(A)
    assert set(T.labels()) == {"e0_d2", "e1_d2", "e2_d2"}
    for label in ("e0_d2", "e1_d2", "e2_d2"):
        ax = T.labels().index(label)
        assert T.indices[ax].dim == 9  # D**2 = 9


def test_double_layer_finite_under_random_contraction():
    """Sanity: contracting T against arbitrary vectors gives finite result.

    (Not testing hermiticity — for a generic A, T = Σ_s A^s ⊗ A̅^s is not
    Hermitian on the virtual indices; only positive in trace under closure.)
    """
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    T = _double_layer_honeycomb(A)
    key = jax.random.PRNGKey(2)
    e = []
    for i in range(3):
        v = jax.random.normal(
            jax.random.fold_in(key, i), (4,)
        ) + 1j * jax.random.normal(jax.random.fold_in(key, i + 10), (4,))
        e.append(v.astype(jnp.complex128))
    val = jnp.einsum("ijk,i,j,k->", T.todense(), e[0], e[1], e[2])
    assert jnp.isfinite(val.real) and jnp.isfinite(val.imag)
