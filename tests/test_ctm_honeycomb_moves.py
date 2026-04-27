"""Honeycomb CTM single-direction move — built sub-step by sub-step.

Sub-step 1: ``_enlarged_left_column(L, T_self, alpha)`` — rank-3, chi → chi·D².
Sub-step 2: ``_enlarged_right_column(R, T_other, alpha)`` — rank-3, chi → chi·D².
Sub-step 3: ``_enlarged_corner_matrix(L, C, R, T_self, T_other, alpha)`` — (χ·D², χ·D²).
Sub-step 4: ``_corner_biorthogonal_projector_from_envs(...)`` — (P_L, P_R) with P_L·P_R=I.
Sub-step 5: ``_apply_projector_to_envs(...)`` — truncate everything back to χ.
Sub-step 6: ``move_direction_alpha(...)`` — full move composition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_honeycomb_init import (
    _double_layer_honeycomb,
    initialize_honeycomb_env,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Inline copy of the rank-4 site fixture (matches test_ctm_honeycomb_init.py)."""
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


# ------------------------------------------------------------------ #
# Sub-step 1: enlarged L column                                        #
# ------------------------------------------------------------------ #


def test_enlarged_left_column_shape_and_finite():
    """L^s_α^new = L^s_α · T_self → rank-3 with chi grown to chi·D² on chi_out."""
    from tenax.algorithms._ctm_honeycomb_moves import _enlarged_left_column

    D, chi = 2, 4
    d2 = D * D
    A = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = {(0, 0): A, (1, 0): A}  # use same site for A,B → T_self is the same
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)
    T_A = _double_layer_honeycomb(A)

    for alpha in range(3):
        L_alpha = getattr(envs[(0, 0)], f"L{alpha}")
        L_new_unproj = _enlarged_left_column(L_alpha, T_A, alpha=alpha)

        dims = tuple(idx.dim for idx in L_new_unproj.indices)
        assert dims == (chi * d2, d2, chi), (
            f"alpha={alpha}: got dims {dims}, want ({chi * d2}, {d2}, {chi})"
        )
        arr = L_new_unproj.todense()
        assert jnp.all(jnp.isfinite(arr)), f"alpha={alpha}: non-finite entries"


# ------------------------------------------------------------------ #
# Sub-step 2: enlarged R column                                        #
# ------------------------------------------------------------------ #


def test_enlarged_right_column_shape_and_finite():
    """R^s_α^new = R^s_α · T_(other s) → rank-3 with chi grown to chi·D²."""
    from tenax.algorithms._ctm_honeycomb_moves import _enlarged_right_column

    D, chi = 2, 4
    d2 = D * D
    A = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(10))
    B = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(11))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)
    T_B = _double_layer_honeycomb(B)

    for alpha in range(3):
        R_alpha = getattr(envs[(0, 0)], f"R{alpha}")
        # For sublattice A's R column, the "other" sublattice is B.
        R_new_unproj = _enlarged_right_column(R_alpha, T_B, alpha=alpha)
        dims = tuple(idx.dim for idx in R_new_unproj.indices)
        assert dims == (chi, d2, chi * d2), (
            f"alpha={alpha}: got dims {dims}, want ({chi}, {d2}, {chi * d2})"
        )
        assert jnp.all(jnp.isfinite(R_new_unproj.todense()))


# ------------------------------------------------------------------ #
# Sub-step 3: enlarged corner matrix (chi·D², chi·D²)                  #
# ------------------------------------------------------------------ #


def test_enlarged_corner_matrix_shape_and_finite():
    """``M = L_new · C · R_new`` reshaped to (chi·D², chi·D²) for biorthogonalization."""
    from tenax.algorithms._ctm_honeycomb_moves import _enlarged_corner_matrix

    D, chi = 2, 4
    d2 = D * D
    A = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(20))
    B = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(21))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi, seed=42)
    T_A = _double_layer_honeycomb(A)
    T_B = _double_layer_honeycomb(B)

    # Sublattice A's enlarged corner: T_self=T_A, T_other=T_B.
    for alpha in range(3):
        env_A = envs[(0, 0)]
        L = getattr(env_A, f"L{alpha}")
        C = getattr(env_A, f"C{alpha}")
        R = getattr(env_A, f"R{alpha}")
        M_A = _enlarged_corner_matrix(L, C, R, T_A, T_B, alpha=alpha)
        assert M_A.shape == (chi * d2, chi * d2), (
            f"alpha={alpha}: M_A shape {M_A.shape}, want ({chi * d2}, {chi * d2})"
        )
        assert jnp.all(jnp.isfinite(M_A)), f"alpha={alpha}: non-finite M_A"

        # Sublattice B's enlarged corner: T_self=T_B, T_other=T_A.
        env_B = envs[(1, 0)]
        L_b = getattr(env_B, f"L{alpha}")
        C_b = getattr(env_B, f"C{alpha}")
        R_b = getattr(env_B, f"R{alpha}")
        M_B = _enlarged_corner_matrix(L_b, C_b, R_b, T_B, T_A, alpha=alpha)
        assert M_B.shape == (chi * d2, chi * d2)
        assert jnp.all(jnp.isfinite(M_B))


# ------------------------------------------------------------------ #
# Sub-step 4: end-to-end biorthogonal projector from real env         #
# ------------------------------------------------------------------ #


def test_biorthogonal_projector_from_real_env_PL_PR_identity():
    """Feed real env-derived (M_A, M_B) into Corboz; ``P_L · P_R = I_chi``."""
    from tenax.algorithms._ctm_honeycomb_moves import _enlarged_corner_matrix
    from tenax.algorithms._ctm_honeycomb_projector import (
        compute_honeycomb_corner_biorthogonal_projector,
    )

    D, chi_init = 2, 4
    chi_target = 4  # truncate the chi·D² = 16 → chi = 4
    A = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(30))
    B = _make_random_honeycomb_site(D=D, d=2, key=jax.random.PRNGKey(31))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=chi_init, seed=42)
    T_A = _double_layer_honeycomb(A)
    T_B = _double_layer_honeycomb(B)

    for alpha in range(3):
        env_A = envs[(0, 0)]
        env_B = envs[(1, 0)]
        M_A = _enlarged_corner_matrix(
            getattr(env_A, f"L{alpha}"),
            getattr(env_A, f"C{alpha}"),
            getattr(env_A, f"R{alpha}"),
            T_A,
            T_B,
            alpha=alpha,
        )
        M_B = _enlarged_corner_matrix(
            getattr(env_B, f"L{alpha}"),
            getattr(env_B, f"C{alpha}"),
            getattr(env_B, f"R{alpha}"),
            T_B,
            T_A,
            alpha=alpha,
        )
        P_L, P_R = compute_honeycomb_corner_biorthogonal_projector(
            M_A, M_B, chi=chi_target
        )
        prod = P_L @ P_R
        eye = jnp.eye(chi_target, dtype=prod.dtype)
        assert jnp.allclose(prod, eye, atol=1e-6), (
            f"alpha={alpha}: max|P_L·P_R - I| = "
            f"{float(jnp.max(jnp.abs(prod - eye))):.3e}"
        )


# ------------------------------------------------------------------ #
# Sub-step 6: full move_direction_alpha                                #
# ------------------------------------------------------------------ #


def test_move_preserves_env_shapes_eigh():
    """``move_direction_alpha`` (eigh path) returns env with same dims."""
    from tenax.algorithms._ctm_honeycomb_moves import move_direction_alpha

    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(0))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(1))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    new_envs = move_direction_alpha(
        envs, sites, alpha=0, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for coord in envs:
        env_old = envs[coord]
        env_new = new_envs[coord]
        for field in env_old._fields:
            old_dims = tuple(idx.dim for idx in getattr(env_old, field).indices)
            new_dims = tuple(idx.dim for idx in getattr(env_new, field).indices)
            assert old_dims == new_dims, (
                f"{coord} {field}: shape changed {old_dims} → {new_dims}"
            )


def test_move_returns_finite_eigh():
    """``move_direction_alpha`` returns finite values across all 9 fields x 2 sites."""
    from tenax.algorithms._ctm_honeycomb_moves import move_direction_alpha

    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(2))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(3))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    new_envs = move_direction_alpha(
        envs, sites, alpha=1, chi=4, projector_method="eigh", forward_gauge="phase"
    )
    for env in new_envs.values():
        for field in env._fields:
            arr = getattr(env, field).todense()
            assert jnp.all(jnp.isfinite(arr)), f"{field}: non-finite entries"


def test_move_biorthogonal_preserves_shapes_and_finite():
    """Biorthogonal default: shapes preserved + finite values across all 6 updated fields."""
    from tenax.algorithms._ctm_honeycomb_moves import move_direction_alpha

    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(40))
    B = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(41))
    sites = {(0, 0): A, (1, 0): B}
    envs = initialize_honeycomb_env(sites, chi_init=4, seed=42)
    for alpha in range(3):
        new_envs = move_direction_alpha(
            envs,
            sites,
            alpha=alpha,
            chi=4,
            projector_method="biorthogonal",
            forward_gauge="phase",
        )
        for coord in envs:
            env_old = envs[coord]
            env_new = new_envs[coord]
            for field in env_old._fields:
                old_dims = tuple(idx.dim for idx in getattr(env_old, field).indices)
                new_dims = tuple(idx.dim for idx in getattr(env_new, field).indices)
                assert old_dims == new_dims, (
                    f"alpha={alpha} {coord} {field}: {old_dims} → {new_dims}"
                )
                arr = getattr(env_new, field).todense()
                assert jnp.all(jnp.isfinite(arr)), (
                    f"alpha={alpha} {coord} {field}: non-finite"
                )
