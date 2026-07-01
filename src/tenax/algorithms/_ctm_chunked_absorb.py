"""Chunked dense-CTM edge absorption (raw arrays).

Reproduces the expensive ``contract(edge, a)`` + projector sandwich of the
1x1 ``_ctm_tensor_move_*`` functions, but chunks the boundary-chi axis with
``lax.map`` so the chi^2 * D^6 intermediate is never materialized in full.
Numerically faithful to ``_apply_projector_tensor`` (T_new = conj(P1)^T . (edge.a) . P2;
P1 is barred -> conj data, P2 is not). Dense path only. See the gate findings
docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md (STRONG GO).
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _chunked_T_new_left(T4, a, P1, P2, chi, D2, batch):
    """LEFT: edge T4 (t4_d, l2, t4_u), a (u2, d2, l2, r2).

    P1 raw (fl=(t4_d, u2), chi_new); P2 raw (fr=(t4_u, d2), chi_new).
    Returns T_new (chi_new, r2, chi_new_r).
    """
    P1r = P1.reshape(chi, D2, -1)                       # (t4_d, u2, chi_new)

    def per_i(args):
        T4_i, P1_i = args                               # (l2, t4_u), (u2, chi_new)
        T4a_i = jnp.einsum("jk,lmjn->klmn", T4_i, a)    # (t4_u, u2, d2, r2)
        Tg_i = T4a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (u2, (t4_u,d2), r2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, r2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, r2, chi_new_r)

    return lax.map(per_i, (T4, P1r), batch_size=batch).sum(0)


def _chunked_T_new_right(T2, a, P1, P2, chi, D2, batch):
    """RIGHT: edge T2 (t2_u, r2, t2_d). P1 fl=(t2_u,u2), P2 fr=(t2_d,d2). T_new (chi_new, l2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t2_u, u2, chi_new)

    def per_i(args):
        T2_i, P1_i = args                               # (r2, t2_d), (u2, chi_new)
        T2a_i = jnp.einsum("jk,lmnj->klmn", T2_i, a)    # (t2_d, u2, d2, l2)
        Tg_i = T2a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (u2, (t2_d,d2), l2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))
        return jnp.tensordot(step, P2, axes=([1], [0]))

    return lax.map(per_i, (T2, P1r), batch_size=batch).sum(0)


def _chunked_T_new_top(T1, a, P1, P2, chi, D2, batch):
    """TOP: edge T1 (t1_l, u2, t1_r). P1 fl=(t1_l,l2), P2 fr=(t1_r,r2). T_new (chi_new, d2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t1_l, l2, chi_new)

    def per_i(args):
        T1_i, P1_i = args                               # (u2, t1_r), (l2, chi_new)
        T1a_i = jnp.einsum("jk,jlmn->klmn", T1_i, a)    # (t1_r, d2, l2, r2)
        # need (fl_D2=l2, fr=(t1_r,r2), surv=d2): order axes (l2, t1_r, r2, d2)
        Tg_i = T1a_i.transpose(2, 0, 3, 1).reshape(D2, chi * D2, D2)  # (l2, (t1_r,r2), d2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, d2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, d2, chi_new_r)

    return lax.map(per_i, (T1, P1r), batch_size=batch).sum(0)


def _chunked_T_new_bottom(T3, a, P1, P2, chi, D2, batch):
    """BOTTOM: edge T3 (t3_r, d2, t3_l). P1 fl=(t3_r,l2), P2 fr=(t3_l,r2). T_new (chi_new, u2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t3_r, l2, chi_new)

    def per_i(args):
        T3_i, P1_i = args                               # (d2, t3_l), (l2, chi_new)
        T3a_i = jnp.einsum("jk,ljmn->klmn", T3_i, a)    # (t3_l, u2, l2, r2)
        # need (fl_D2=l2, fr=(t3_l,r2), surv=u2): order axes (l2, t3_l, r2, u2)
        Tg_i = T3a_i.transpose(2, 0, 3, 1).reshape(D2, chi * D2, D2)  # (l2, (t3_l,r2), u2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, u2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, u2, chi_new_r)

    return lax.map(per_i, (T3, P1r), batch_size=batch).sum(0)
