"""Brute-force vs multisite-CTM RDM diagnostic + structural-invariants gates.

See docs/plans/2026-05-06-multisite-ctm-rdm-diagnostic.md for design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import IPESSState, pess_to_kagome_3site_multisite

# 3×3 PBC torus position → sublattice. See the plan's "Lattice constants".
_POS_TO_NAME: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "v",
    "w",
    "u",
    "w",
    "u",
    "v",
)


def _contract_multisite_3x3_torus(sites: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Exact contraction of the 3-site multisite encoding on a 3×3 PBC torus.

    Sublattice = (x + y) mod 3 ↦ {0:u, 1:v, 2:w}. 9 sites at row-major
    positions 0..8. Each site has 4 virtual legs (top, bot, lft, rgt) and
    1 physical leg. Output: rank-9 array indexed by physical legs in
    row-major position order.

    Bond labels (1 letter per bond):
      Horizontal (right→left): a..i for the 9 H_BONDS pairs.
      Vertical   (bot→top):    j..r for the 9 V_BONDS pairs.
      Physical:                A..I (for positions 0..8).

    See the plan for derivation.
    """
    # Per-site einsum strings: each is "top, bot, lft, rgt, phys".
    strings = (
        "pjcaA",  # pos 0 (u): top=p (V6 wrap), bot=j (V0), lft=c (H2 wrap), rgt=a (H0)
        "qkabB",  # pos 1 (v): top=q (V7 wrap), bot=k (V1), lft=a (H0),     rgt=b (H1)
        "rlbcC",  # pos 2 (w): top=r (V8 wrap), bot=l (V2), lft=b (H1),     rgt=c (H2)
        "jmfdD",  # pos 3 (v): top=j (V0),      bot=m (V3), lft=f (H5 wrap),rgt=d (H3)
        "kndeE",  # pos 4 (w): top=k (V1),      bot=n (V4), lft=d (H3),     rgt=e (H4)
        "loefF",  # pos 5 (u): top=l (V2),      bot=o (V5), lft=e (H4),     rgt=f (H5)
        "mpigG",  # pos 6 (w): top=m (V3),      bot=p (V6), lft=i (H8 wrap),rgt=g (H6)
        "nqghH",  # pos 7 (u): top=n (V4),      bot=q (V7), lft=g (H6),     rgt=h (H7)
        "orhiI",  # pos 8 (v): top=o (V5),      bot=r (V8), lft=h (H7),     rgt=i (H8)
    )
    spec = ",".join(strings) + "->ABCDEFGHI"
    args = [sites[_POS_TO_NAME[p]] for p in range(9)]
    return jnp.einsum(spec, *args, optimize="optimal")


@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_multisite_3x3_torus_translation_invariant_diagonal(D):
    """ψ on the 3×3 PBC torus must be invariant under the (1,-1) diagonal
    shift: this preserves the (x+y) mod 3 sublattice assignment, so a
    correct contraction reproduces ψ exactly under the induced 9-position
    permutation.

    Permutation π: (x,y) → ((x+1) mod 3, (y-1) mod 3). For row-major
    pos = 3y+x, the new pos for old pos p with (x = p%3, y = p//3) is
    new_pos = 3*((y-1) % 3) + ((x+1) % 3).
    """
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    sites = pess_to_kagome_3site_multisite(
        state.R_a,
        state.R_b,
        state.R_c,
        state.T_u,
        state.T_d,
        state.lambdas,
    )
    psi = _contract_multisite_3x3_torus(sites)

    perm = tuple(3 * ((p // 3 - 1) % 3) + ((p % 3 + 1) % 3) for p in range(9))
    # Permute the AXES of psi by perm (axis 0 should now hold what was at axis perm[0]).
    psi_shifted = jnp.transpose(psi, perm)

    np.testing.assert_allclose(
        np.asarray(psi),
        np.asarray(psi_shifted),
        rtol=1e-12,
        atol=1e-12,
        err_msg=(
            f"Multisite 3×3 torus wavefunction not invariant under (1,-1) "
            f"diagonal shift at D={D}. Bond labelling in "
            f"_contract_multisite_3x3_torus is wrong."
        ),
    )
