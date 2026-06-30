"""Production-correctness of the 1-site split-CTM path (#463 pre-flip check).

Validates that the split (``fuse_virtual_legs=False``) single-site path, run
WITH C4v on the sublattice-rotated square Heisenberg gate (under which a 1-site
iPEPS represents Neel order), converges to a *variational* energy: above the
QMC ground-state energy and below the disordered energy.

Empirical anchors (D=2, chi=10, gs_c4v=True, grad_norm |g|<1e-3):
  split+c4v = -0.6505/site (variational, +0.019 above QMC -0.6694)
  fused+c4v = -0.6601/site
WITHOUT C4v the unconstrained 1-site CTM is non-variational for BOTH paths
(split breaches to -0.714) -- so gs_c4v=True is mandatory here. The split/fused
~0.01/site gap is the bounded #425 fixed-point difference (both physical).
"""

import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tests.test_split_ctm_fuse_flag import _make_site

# Sandvik QMC square-lattice spin-1/2 Heisenberg, energy per site.
QMC_FLOOR = -0.6694
ORDERED_CEIL = -0.60  # below the disordered/product energy => genuine order


def _rotated_heisenberg():
    """H_rot = -SzSz - 0.5(S+S+ + S-S-): sublattice-rotated so a 1-site iPEPS
    represents Neel order (unitary image of the AFM Heisenberg gate)."""
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = -jnp.kron(Sz, Sz) - 0.5 * (jnp.kron(Sp, Sp) + jnp.kron(Sm, Sm))
    return H.reshape(2, 2, 2, 2)


def _config(fuse):
    return iPEPSConfig(
        ctm=CTMConfig(
            chi=10,
            chi_I=10,
            fuse_virtual_legs=fuse,
            max_iter=80,
            conv_tol=1e-10,
            min_iter=4,
        ),
        unit_cell="1x1",
        gs_recipe="1x1",
        gs_implicit_ad=True,
        gs_c4v=True,
        gs_metric_precond=False,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-3,
        gs_num_steps=100,
        gs_log_interval=10,
        su_init=False,
    )


@pytest.mark.slow
def test_split_1site_is_variational_with_c4v():
    """Split + C4v on rotated Heisenberg must land in [-0.6694, -0.60]/site.

    The lower bound is THE assertion: a correct variational 1-site path stays
    above the QMC ground state; the #425-spurious sub-QMC fixed point (which
    appears WITHOUT C4v, E=-0.714) would breach it.
    """
    A = _make_site(2, 2, seed=3)
    _, _, E = optimize_gs_ad(_rotated_heisenberg(), A, _config(fuse=False))
    E = float(E)  # per site = E_h + E_v
    assert E >= QMC_FLOOR - 1e-3, (
        f"split breaches QMC variational floor: E/site={E:.6f} < {QMC_FLOOR}"
    )
    assert E <= ORDERED_CEIL, f"split did not order: E/site={E:.6f} > {ORDERED_CEIL}"
