"""Parity tests for chunked dense-CTM edge-absorption core.

Each test verifies that _chunked_T_new_{left,right,top,bottom} matches the
monolithic einsum reference at rel-error <= 1e-12 (x64), across three batch
sizes including a ragged one.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_chunked_absorb import (
    _chunked_T_new_bottom,
    _chunked_T_new_left,
    _chunked_T_new_right,
    _chunked_T_new_top,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _ctm_tensor_move_bottom,
    _ctm_tensor_move_left,
    _ctm_tensor_move_right,
    _ctm_tensor_move_top,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.fixture(autouse=True, scope="session")
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# Monolith references (ground truth)
# ---------------------------------------------------------------------------

def _monolith_left(T4, a, P1, P2, chi, D2):
    # Faithful reference: contract(T4, a) then conj(P1)^T . Tg . P2 (matches
    # _apply_projector_tensor for the LEFT move).
    T4a = jnp.einsum("ijk,lmjn->iklmn", T4, a)          # (t4_d, t4_u, u2, d2, r2)
    Tg = T4a.transpose(0, 2, 1, 3, 4).reshape(chi * D2, chi * D2, D2)  # (fl, fr, r2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))  # (chi_new, fr, r2)
    return jnp.tensordot(step, P2, axes=([1], [0]))       # (chi_new, r2, chi_new_r)


def _monolith_right(T2, a, P1, P2, chi, D2):
    T2a = jnp.einsum("ijk,lmnj->iklmn", T2, a)          # (t2_u, t2_d, u2, d2, l2)
    Tg = T2a.transpose(0, 2, 1, 3, 4).reshape(chi * D2, chi * D2, D2)  # (fl=(t2_u,u2), fr=(t2_d,d2), l2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))
    return jnp.tensordot(step, P2, axes=([1], [0]))     # (chi_new, l2, chi_new_r)


def _monolith_top(T1, a, P1, P2, chi, D2):
    T1a = jnp.einsum("ijk,jlmn->iklmn", T1, a)          # (t1_l, t1_r, d2, l2, r2)
    # fl=(t1_l,l2), fr=(t1_r,r2), surv=d2 -> order (t1_l, l2, t1_r, r2, d2)
    Tg = T1a.transpose(0, 3, 1, 4, 2).reshape(chi * D2, chi * D2, D2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))
    return jnp.tensordot(step, P2, axes=([1], [0]))     # (chi_new, d2, chi_new_r)


def _monolith_bottom(T3, a, P1, P2, chi, D2):
    T3a = jnp.einsum("ijk,ljmn->iklmn", T3, a)          # (t3_r, t3_l, u2, l2, r2)
    # fl=(t3_r,l2), fr=(t3_l,r2), surv=u2 -> order (t3_r, l2, t3_l, r2, u2)
    Tg = T3a.transpose(0, 3, 1, 4, 2).reshape(chi * D2, chi * D2, D2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))
    return jnp.tensordot(step, P2, axes=([1], [0]))     # (chi_new, u2, chi_new_r)


# ---------------------------------------------------------------------------
# LEFT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [16, 8, 5])  # K=1, K=2, ragged
def test_chunked_left_matches_monolith(batch):
    D, chi = 3, 16
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    T4 = jax.random.normal(k[0], (chi, D2, chi))
    a = jax.random.normal(k[1], (D2, D2, D2, D2))
    P1 = jax.random.normal(k[2], (chi * D2, chi))
    P2 = jax.random.normal(k[3], (chi * D2, chi))
    ref = _monolith_left(T4, a, P1, P2, chi, D2)
    got = _chunked_T_new_left(T4, a, P1, P2, chi, D2, batch)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, rel


# ---------------------------------------------------------------------------
# RIGHT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [16, 8, 5])
def test_chunked_right_matches_monolith(batch):
    D, chi = 3, 16
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    T2 = jax.random.normal(k[0], (chi, D2, chi))
    a = jax.random.normal(k[1], (D2, D2, D2, D2))
    P1 = jax.random.normal(k[2], (chi * D2, chi))
    P2 = jax.random.normal(k[3], (chi * D2, chi))
    ref = _monolith_right(T2, a, P1, P2, chi, D2)
    got = _chunked_T_new_right(T2, a, P1, P2, chi, D2, batch)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, rel


# ---------------------------------------------------------------------------
# TOP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [16, 8, 5])
def test_chunked_top_matches_monolith(batch):
    D, chi = 3, 16
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(2), 4)
    T1 = jax.random.normal(k[0], (chi, D2, chi))
    a = jax.random.normal(k[1], (D2, D2, D2, D2))
    P1 = jax.random.normal(k[2], (chi * D2, chi))
    P2 = jax.random.normal(k[3], (chi * D2, chi))
    ref = _monolith_top(T1, a, P1, P2, chi, D2)
    got = _chunked_T_new_top(T1, a, P1, P2, chi, D2, batch)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, rel


# ---------------------------------------------------------------------------
# BOTTOM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [16, 8, 5])
def test_chunked_bottom_matches_monolith(batch):
    D, chi = 3, 16
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(3), 4)
    T3 = jax.random.normal(k[0], (chi, D2, chi))
    a = jax.random.normal(k[1], (D2, D2, D2, D2))
    P1 = jax.random.normal(k[2], (chi * D2, chi))
    P2 = jax.random.normal(k[3], (chi * D2, chi))
    ref = _monolith_bottom(T3, a, P1, P2, chi, D2)
    got = _chunked_T_new_bottom(T3, a, P1, P2, chi, D2, batch)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, rel


# ---------------------------------------------------------------------------
# Complex-projector path (M-3): verifies P1.conj() is correct for complex P1
# ---------------------------------------------------------------------------

def test_chunked_left_complex_projector():
    """LEFT with complex P1/P2 and T4 — exercises P1.conj() on non-real inputs."""
    D, chi = 3, 16
    D2 = D * D
    keys = jax.random.split(jax.random.PRNGKey(42), 8)
    T4 = (jax.random.normal(keys[0], (chi, D2, chi))
          + 1j * jax.random.normal(keys[1], (chi, D2, chi)))
    a  = (jax.random.normal(keys[2], (D2, D2, D2, D2))
          + 1j * jax.random.normal(keys[3], (D2, D2, D2, D2)))
    P1 = (jax.random.normal(keys[4], (chi * D2, chi))
          + 1j * jax.random.normal(keys[5], (chi * D2, chi)))
    P2 = (jax.random.normal(keys[6], (chi * D2, chi))
          + 1j * jax.random.normal(keys[7], (chi * D2, chi)))
    ref = _monolith_left(T4, a, P1, P2, chi, D2)
    got = _chunked_T_new_left(T4, a, P1, P2, chi, D2, batch=8)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, f"complex-projector rel error {rel} > 1e-12"


# ---------------------------------------------------------------------------
# Full 1x1 LEFT move: chunked branch == default branch (Task 2)
# ---------------------------------------------------------------------------

def _random_dense_site_tensor(D: int, d: int, seed: int = 0) -> DenseTensor:
    """Random dense iPEPS site tensor with labels (u,d,l,r,phys), all-zero U(1) charges."""
    rng = np.random.RandomState(seed)
    data = jnp.array(rng.randn(D, D, D, D, d), dtype=jnp.float64)
    sym = U1Symmetry()
    indices = (
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="u"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="d"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="l"),
        TensorIndex.from_charges(sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="r"),
        TensorIndex.from_charges(sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, indices)


def test_full_left_move_chunked_matches_default():
    """Chunked branch of _ctm_tensor_move_left must be bit-identical to default (rel <= 1e-12)."""
    D, d, chi = 2, 2, 12
    A = _random_dense_site_tensor(D, d, seed=7)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    base_env, _ = _ctm_tensor_move_left(env, env, a, chi)
    chunked_env, _ = _ctm_tensor_move_left(env, env, a, chi, chunk_size=4)
    for field in ("C1", "C4", "T4"):
        b = getattr(base_env, field)
        c = getattr(chunked_env, field)
        b_arr = b.todense()
        c_arr = c.todense()
        # align leg order via labels before comparing
        b_labels = list(b.labels())
        c_labels = list(c.labels())
        perm = [c_labels.index(x) for x in b_labels]
        c_arr_aligned = jnp.transpose(c_arr, perm)
        rel = float(
            jnp.max(jnp.abs(c_arr_aligned - b_arr))
            / (jnp.max(jnp.abs(b_arr)) + 1e-30)
        )
        assert rel <= 1e-12, f"field={field!r} rel={rel}"


@pytest.mark.parametrize("move_fn,fields", [
    (_ctm_tensor_move_right, ("C2", "C3", "T2")),
    (_ctm_tensor_move_top, ("C1", "C2", "T1")),
    (_ctm_tensor_move_bottom, ("C4", "C3", "T3")),
])
def test_full_move_chunked_matches_default(move_fn, fields):
    """Chunked branch of right/top/bottom CTM moves must match default (rel <= 1e-12)."""
    D, d, chi = 2, 2, 12
    A = _random_dense_site_tensor(D, d, seed=7)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    base_env, _ = move_fn(env, env, a, chi)
    chunked_env, _ = move_fn(env, env, a, chi, chunk_size=4)
    for field in fields:
        b = getattr(base_env, field)
        c = getattr(chunked_env, field)
        b_arr = b.todense()
        c_arr = c.todense()
        b_labels = list(b.labels())
        c_labels = list(c.labels())
        perm = [c_labels.index(x) for x in b_labels]
        c_arr_aligned = jnp.transpose(c_arr, perm)
        rel = float(
            jnp.max(jnp.abs(c_arr_aligned - b_arr))
            / (jnp.max(jnp.abs(b_arr)) + 1e-30)
        )
        assert rel <= 1e-12, f"move={move_fn.__name__!r} field={field!r} rel={rel}"


def test_full_left_move_chunked_ragged_matches_default():
    """Ragged chunk_size=5 on chi=12 (5 does not divide 12) must match default (rel <= 1e-12)."""
    D, d, chi = 2, 2, 12
    A = _random_dense_site_tensor(D, d, seed=13)
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    base_env, _ = _ctm_tensor_move_left(env, env, a, chi)
    chunked_env, _ = _ctm_tensor_move_left(env, env, a, chi, chunk_size=5)
    for field in ("C1", "C4", "T4"):
        b = getattr(base_env, field)
        c = getattr(chunked_env, field)
        b_arr = b.todense()
        c_arr = c.todense()
        # align leg order via labels before comparing
        b_labels = list(b.labels())
        c_labels = list(c.labels())
        perm = [c_labels.index(x) for x in b_labels]
        c_arr_aligned = jnp.transpose(c_arr, perm)
        rel = float(
            jnp.max(jnp.abs(c_arr_aligned - b_arr))
            / (jnp.max(jnp.abs(b_arr)) + 1e-30)
        )
        assert rel <= 1e-12, f"ragged chunk: field={field!r} rel={rel}"
