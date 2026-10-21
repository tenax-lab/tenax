"""``_fishman_truncate_S`` must measure against the GLOBAL maximum.

The symmetric 2x2 projector passes the block-sparse SVD's singular values
concatenated sector by sector, so ``S[0]`` is the lowest-charge sector's
largest value, not the spectrum's.  The truncation floor must not depend on
the order the values arrive in.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_projector_2x2 import _fishman_truncate_S


def test_floor_is_relative_to_the_global_max_not_the_first_entry():
    # Sector-concatenated order: a weak leading sector, then the dominant one.
    S = jnp.array([1e-13, 5e-14, 1.0, 0.5, 2e-13])
    out = np.asarray(_fishman_truncate_S(S, eps=1e-12))
    # Everything below 1e-12 * max(S) = 1e-12 is dropped, wherever it sits.
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 0.5, 0.0])


@pytest.mark.parametrize("seed", range(5))
def test_truncation_commutes_with_any_ordering_of_the_spectrum(seed):
    rng = np.random.default_rng(seed)
    S = np.concatenate([10.0 ** rng.uniform(-16, 0, size=12), [1.0]])
    perm = rng.permutation(S.size)
    sorted_out = np.asarray(_fishman_truncate_S(jnp.asarray(np.sort(S)[::-1])))
    perm_out = np.asarray(_fishman_truncate_S(jnp.asarray(S[perm])))
    np.testing.assert_array_equal(np.sort(perm_out)[::-1], sorted_out)
