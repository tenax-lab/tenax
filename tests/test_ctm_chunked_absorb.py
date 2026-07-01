"""Parity tests for chunked dense-CTM edge-absorption core.

Each test verifies that _chunked_T_new_{left,right,top,bottom} matches the
monolithic einsum reference at rel-error <= 1e-12 (x64), across three batch
sizes including a ragged one.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_chunked_absorb import (
    _chunked_T_new_left,
    _chunked_T_new_right,
    _chunked_T_new_top,
    _chunked_T_new_bottom,
)


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
