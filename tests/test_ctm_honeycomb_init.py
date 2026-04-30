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


def test_initialize_env_returns_two_sublattices():
    from tenax.algorithms._ctm_honeycomb_init import initialize_honeycomb_env

    A = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(3))
    B = _make_random_honeycomb_site(D=3, d=2, key=jax.random.PRNGKey(4))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4)
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    for coord, env in envs.items():
        # Corners shape (chi, chi); columns shape (chi, D**2, chi); D=3 so D**2=9
        # Use TensorIndex.dim since the Tensor protocol has no .shape:
        for c_field in ("C0", "C1", "C2"):
            T = getattr(env, c_field)
            assert tuple(idx.dim for idx in T.indices) == (4, 4)
        for col_field in ("L0", "L1", "L2", "R0", "R1", "R2"):
            T = getattr(env, col_field)
            assert tuple(idx.dim for idx in T.indices) == (4, 9, 4)


def test_double_layer_trace_equals_frobenius_norm():
    """Tr_{e0,e1,e2}[T] = Σ_{e0,e1,e2,s} |A[e0,e1,e2,s]|² (iPEPS norm identity).

    For the rank-3 double-layer T = Σ_s A^s ⊗ Ā^s, summing the diagonal
    over each fused leg recovers the Frobenius norm of A. A wrong fuse
    pairing (e.g. fusing e0 with E1) would still produce a finite tensor
    but would fail this identity.
    """
    D, d = 3, 2
    A = _make_random_honeycomb_site(D=D, d=d, key=jax.random.PRNGKey(11))
    T = _double_layer_honeycomb(A)

    # T is rank-3 with each leg of dim D**2; the fused index is (e_ket, e_bra)
    # with axis_a (ket) slow-varying — diagonal entries i*D+i for i in 0..D-1.
    T_dense = T.todense()
    # Reshape (D², D², D²) → (D, D, D, D, D, D) with order (e0_ket, e0_bra, ...)
    T6 = T_dense.reshape(D, D, D, D, D, D)
    trace = jnp.einsum("iijjkk->", T6)

    A_dense = A.todense()
    expected = jnp.sum(jnp.abs(A_dense) ** 2)

    assert jnp.allclose(trace, expected, rtol=1e-10), (
        f"trace {trace} != Frobenius² {expected}"
    )
