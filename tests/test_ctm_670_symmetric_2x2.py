"""Symmetric 2x2 must stay bond-consistent after absorption (#670)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import tenax.algorithms.ipeps_simple_update as SU
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS as NB,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _sort_coords_for_direction,
)
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_moves import (
    _compute_plaquette_projector_pair,
    _ctm_tensor_absorb_left_2plaq,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

CHI = 12


def _uniform_multicharge_pair(D=3, steps=40, dt=0.1):
    """Normal U(1)-Sz SU (base_charges kept) -> direction-uniform multi-charge."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lh, lv = jnp.ones(D), jnp.ones(D)
    for s in range(steps):
        if s % 2 == 0:
            A, B, lh = SU._simple_update_2site_horizontal_tensor(A, B, gate, lh, lv, D)
        else:
            A, B, lv = SU._simple_update_2site_vertical_tensor(A, B, gate, lh, lv, D)
    return A, B


def _direction_dependent_pair(D=3, steps=40, dt=0.1):
    """base_charges-free U(1)-Sz SU -> A.l != A.r (direction-dependent bonds)."""
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    H = heisenberg_gate_u1sz()
    gate = SU._make_trotter_gate_tensor(H, dt, site_tensor=A)
    lh, lv = jnp.ones(D), jnp.ones(D)
    orig = SU.truncated_svd
    SU.truncated_svd = lambda *a, **k: orig(*a, **{**k, "base_charges": None})
    try:
        for s in range(steps):
            if s % 2 == 0:
                A, B, lh = SU._simple_update_2site_horizontal_tensor(
                    A, B, gate, lh, lv, D
                )
            else:
                A, B, lv = SU._simple_update_2site_vertical_tensor(
                    A, B, gate, lh, lv, D
                )
    finally:
        SU.truncated_svd = orig
    return A, B


def _one_left(A, B):
    site = {(0, 0): A, (1, 0): B}
    dl = {c: _build_double_layer_tensor(t) for c, t in site.items()}
    envs = {c: initialize_ctm_tensor_env(t, CHI) for c, t in site.items()}
    projectors = {}
    for s_anchor in envs:
        s_TR = NB[s_anchor]["right"]
        s_BL = NB[s_anchor]["bottom"]
        s_BR = NB[s_TR]["bottom"]
        Pt, Pb, _, _ = _compute_plaquette_projector_pair(
            envs[s_anchor],
            envs[s_TR],
            envs[s_BL],
            envs[s_BR],
            dl[s_anchor],
            dl[s_TR],
            dl[s_BL],
            dl[s_BR],
            CHI,
            "left",
        )
        projectors[s_anchor] = (Pt, Pb)
    new = {}
    for s_dst in _sort_coords_for_direction(list(envs), "left"):
        s_src = NB[s_dst]["left"]
        sa = NB[s_src]["top"]
        Pta, Pba = projectors[sa]
        Ptc, Pbc = projectors[s_src]
        C1, T4, C4 = _ctm_tensor_absorb_left_2plaq(
            envs[s_src], dl[s_src], Pta, Pba, Ptc, Pbc
        )
        new[s_dst] = envs[s_dst]._replace(C1=C1, T4=T4, C4=C4)
    return new, dl


def test_enlarged_corners_build_after_left_absorption_multicharge():
    A, B = _direction_dependent_pair()
    new, dl = _one_left(A, B)
    for s_dst, env in new.items():
        for pos, (C, Th, Tv) in {
            "top_left": (env.C1, env.T1, env.T4),
            "top_right": (env.C2, env.T1, env.T2),
            "bottom_left": (env.C4, env.T3, env.T4),
            "bottom_right": (env.C3, env.T3, env.T2),
        }.items():
            Q = _build_enlarged_corner(C, Th, Tv, dl[s_dst], position=pos)
            assert Q is not None, f"{pos} failed to build at {s_dst}"
