"""Structural-correctness gate for the 3-site multisite kagome encoding.

Task B.3a of plan ``docs/plans/2026-05-05-multisite-kagome-pess.md`` — replaces
the original B.3 (supersite-vs-multisite energy parity).  That parity test is
structurally invalid: ``pess_to_kagome_supersite`` does not take ``T_d`` as an
argument (the supersite/CG path freezes ``T_d`` bit-exact, see
``tests/test_pess_ad.py:116``).  The supersite and 3-site multisite encodings
live on different variational manifolds and cannot agree on an arbitrary
``IPESSState``.

This test instead asserts wavefunction-level equality between

  (a) the iPESS state contracted directly on a 1-unit-cell PBC kagome torus,
  (b) the 3-site multisite tensors contracted on a 1-cell 3-cycle PBC torus
      (closing the kagome neighbour map cyclically among the 3 sublattices).

Both produce a vector in the same ``(d, d, d)`` Hilbert space and should be
colinear (fidelity = 1) for any iPESS state.  CTM-free, AD-free, χ-free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    pess_to_kagome_3site_multisite,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _contract_ipess_one_cell_pbc(state: IPESSState) -> jnp.ndarray:
    """Contract the iPESS state on a 1-unit-cell PBC kagome torus.

    All 6 bonds (3 R-T_u up-triangle + 3 R-T_d down-triangle) close inside
    the cell.  Gauge convention matches :func:`pess_to_kagome_3site_multisite`:
    sqrt(λ) on each R's T_d-side (axis 0) and full λ on each R's T_u-side
    (axis 1).  T_u and T_d carry no extra gauges.

    Returns:
        Rank-3 array ``(d, d, d)`` indexed by ``(p_u, p_v, p_w)`` (sublattice
        ``a/b/c`` mapped to multisite names ``u/v/w``).
    """
    R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    T_u, T_d = state.T_u, state.T_d
    lam_au, lam_bu, lam_cu = state.lambdas[0:3]
    lam_ad, lam_bd, lam_cd = state.lambdas[3:6]

    dtype = R_a.dtype

    def sqrt_lam(x):
        # Smooth ``sqrt`` mirrors :func:`pess_to_kagome_3site_multisite`.
        return jnp.power(jnp.real(x) ** 2 + 1e-28, 0.25).astype(dtype)

    sda = sqrt_lam(lam_ad)
    sdb = sqrt_lam(lam_bd)
    sdc = sqrt_lam(lam_cd)
    lau = lam_au.astype(dtype)
    lbu = lam_bu.astype(dtype)
    lcu = lam_cu.astype(dtype)

    # iPESS axis convention on R: (T_d-side, T_u-side, phys).
    Ra = jnp.einsum("i,ijp,j->ijp", sda, R_a, lau)
    Rb = jnp.einsum("i,ijp,j->ijp", sdb, R_b, lbu)
    Rc = jnp.einsum("i,ijp,j->ijp", sdc, R_c, lcu)

    # Closure: T_u[ja, jb, jc] couples the three T_u-sides.  T_d[ia, ib, ic]
    # couples the three T_d-sides.  Output is the rank-3 cell wavefunction
    # in ``(p_a, p_b, p_c)`` order = ``(p_u, p_v, p_w)``.
    return jnp.einsum(
        "ABu,CDv,EFw,BDF,ACE->uvw",
        Ra,
        Rb,
        Rc,
        T_u.astype(dtype),
        T_d.astype(dtype),
    )


def _contract_multisite_3cycle_pbc(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Contract the 3-site multisite tensors on a 1-cell 3-cycle PBC torus.

    Tensor leg layout (matches :func:`pess_to_kagome_3site_multisite` output)::

        axes = (top, bottom, left, right, phys)

    Bond closures from :func:`tenax.core.lattice.kagome` 's neighbour map
    (each pair of sublattices is connected by 2 bonds — one up-triangle, one
    down-triangle)::

        u.top   ↔ w.bottom   (down-tri u-w bond)            label "T"
        u.bottom↔ v.top      (down-tri u-v bond)            label "B"
        u.left  ↔ w.right    (up-tri  u-w bond)             label "L"
        u.right ↔ v.left     (up-tri  u-v bond)             label "R"
        v.bottom↔ w.top      (v-w bond, both sides dim 1)   label "X"
        v.right ↔ w.left     (v-w bond, both sides dim 1)   label "Y"

    Returns:
        Rank-3 array ``(d, d, d)`` indexed by ``(p_u, p_v, p_w)``.
    """
    S_u = sites["u"]  # axes (top=T, bot=B, lft=L, rgt=R, phys=u)
    S_v = sites["v"]  # axes (top=B, bot=X, lft=R, rgt=Y, phys=v)
    S_w = sites["w"]  # axes (top=X, bot=T, lft=Y, rgt=L, phys=w)
    return jnp.einsum(
        "TBLRu,BXRYv,XTYLw->uvw",
        S_u,
        S_v,
        S_w,
    )


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_3site_multisite_wavefunction_matches_ipess_on_1cell_torus(D):
    """Fidelity == 1 on the smallest valid kagome PBC torus.

    A failure at D≥2 with success at D=1 would localise a leg-axis bug to
    the encoding of the non-trivial bond legs (which are dim 1 at D=1).
    """
    d = 2
    state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))

    psi_ipess = _contract_ipess_one_cell_pbc(state)
    sites = pess_to_kagome_3site_multisite(
        state.R_a, state.R_b, state.R_c, state.T_u, state.T_d, state.lambdas
    )
    psi_ms = _contract_multisite_3cycle_pbc(sites)

    assert psi_ipess.shape == (d, d, d)
    assert psi_ms.shape == (d, d, d)

    n_ipess = float(jnp.linalg.norm(psi_ipess))
    n_ms = float(jnp.linalg.norm(psi_ms))
    assert n_ipess > 0.0, "iPESS wavefunction is zero — random IPESSState gave 0"
    assert n_ms > 0.0, "multisite wavefunction is zero"
    overlap = jnp.vdot(psi_ipess.reshape(-1), psi_ms.reshape(-1))
    fidelity = float(jnp.abs(overlap) ** 2 / (n_ipess**2 * n_ms**2))

    np.testing.assert_allclose(
        fidelity,
        1.0,
        atol=1e-12,
        err_msg=(
            f"Encoding fidelity FAILED at D={D}: |<ψ_iPESS|ψ_ms>|² / "
            f"(||ψ_iPESS||²·||ψ_ms||²) = {fidelity:.12f} != 1. "
            f"Multisite encoding does not faithfully represent the iPESS "
            f"state (plan stop-and-ask checkpoint #2)."
        ),
    )
