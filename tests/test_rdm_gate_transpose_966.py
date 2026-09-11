"""Complex-Hermitian gates must contract against the RDM's bra axes (#966).

Every energy path pairs a grouped ``(s1_ket, s2_ket, s1_bra, s2_bra)`` RDM
with a ``(out1, out2, in1, in2)`` gate.  Pairing the axes elementwise
(``ijkl,ijkl``) computes ``Tr(rho H^T)``: identical for real-symmetric
gates — the entire pre-existing suite — and sign-flipped on the coupling
between the state's and the gate's imaginary parts.  On the exact product
state ``|+y>`` at every site, ``H = Sy (x) I`` has bond expectation +0.5
and per-site 2-bond sum +1.0; every path returned −1.0 before the fix.

D = 1 product fixtures make each oracle exact (atol 1e-12): the CTM has a
one-dimensional environment and converges to the product environment, so
there is no truncation caveat anywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_energy import (
    compute_energy_ctm_tensor,
    compute_energy_ctm_tensor_2site,
    compute_energy_ctm_tensor_multisite,
)
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.ipeps_ctm import ctm
from tenax.algorithms.ipeps_rdm import compute_energy_ctm

pytestmark = pytest.mark.core

_SY = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])

# <+y|Sy|+y> = +1/2; the (x)I partner contributes a factor 1.
_TRUE_PER_SITE = 1.0  # two bonds per site, +0.5 each


def _plus_y_array():
    v = jnp.array([1.0, 1j]) / np.sqrt(2)
    return jnp.zeros((1, 1, 1, 1, 2), dtype=jnp.complex128).at[0, 0, 0, 0].set(v)


def _gate():
    return jnp.einsum("ab,cd->acbd", _SY, jnp.eye(2))


def _kw():
    return ctm_converge_kwargs(CTMConfig(chi=4, max_iter=50, conv_tol=1e-12))


def test_tensor_single_site_energy():
    A = _wrap_as_dense_tensor(_plus_y_array())
    envs, _ = python_loop_ctm_converge({(0, 0): A}, SINGLE_SITE_NEIGHBORS, **_kw())
    e = complex(compute_energy_ctm_tensor(A, envs[(0, 0)], _gate(), 2))
    np.testing.assert_allclose(e, _TRUE_PER_SITE, atol=1e-12)


def test_rdm_single_site_energy():
    A = _plus_y_array()
    env = ctm(A, CTMConfig(chi=4, max_iter=50, conv_tol=1e-12))
    e = complex(compute_energy_ctm(A, env, _gate(), 2))
    np.testing.assert_allclose(e, _TRUE_PER_SITE, atol=1e-12)


def test_tensor_2site_energy():
    A = _wrap_as_dense_tensor(_plus_y_array())
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A, (1, 0): A}, CHECKERBOARD_NEIGHBORS, **_kw()
    )
    e = complex(
        compute_energy_ctm_tensor_2site(A, A, envs[(0, 0)], envs[(1, 0)], _gate(), 2)
    )
    np.testing.assert_allclose(e, _TRUE_PER_SITE, atol=1e-12)


def test_tensor_multisite_energy():
    A = _wrap_as_dense_tensor(_plus_y_array())
    sites = {(0, 0): A, (1, 0): A}
    envs, _ = python_loop_ctm_converge(sites, CHECKERBOARD_NEIGHBORS, **_kw())
    e = complex(
        compute_energy_ctm_tensor_multisite(
            sites, envs, CHECKERBOARD_NEIGHBORS, _gate()
        )
    )
    np.testing.assert_allclose(e, _TRUE_PER_SITE, atol=1e-12)


def test_split_single_site_energy():
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor,
    )

    A = _wrap_as_dense_tensor(_plus_y_array())
    env = ctm_split_tensor(A, chi=4, chi_I=4, max_iter=50, conv_tol=1e-12)
    e = complex(compute_energy_split_ctm_tensor(A, env, _gate(), 2))
    np.testing.assert_allclose(e, _TRUE_PER_SITE, atol=1e-12)


def test_real_symmetric_gate_is_bit_identical_either_way():
    """Regression guard on the guard: for a real-symmetric gate the flip is
    the identity, so the whole pre-existing (real) suite is untouched.  A
    failure here means an RDM builder changed its grouping convention."""
    A = _wrap_as_dense_tensor(_plus_y_array())
    envs, _ = python_loop_ctm_converge({(0, 0): A}, SINGLE_SITE_NEIGHBORS, **_kw())
    sz = 0.5 * jnp.diag(jnp.array([1.0, -1.0]))
    gate = jnp.einsum("ab,cd->acbd", sz, sz)
    e = complex(compute_energy_ctm_tensor(A, envs[(0, 0)], gate, 2))
    # <+y|Sz|+y> = 0 exactly, on both bonds.
    np.testing.assert_allclose(e, 0.0, atol=1e-12)
