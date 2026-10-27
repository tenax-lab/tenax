"""Input guards of the symmetric AD validation helpers (fast)."""

import dataclasses

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._symmetric_ad_validation import make_2site_loss
from tenax.algorithms.ctm_layout import normalized_pair
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate, xxz_gate_u1sz
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs

CHI = 4


@pytest.fixture(scope="module")
def pair_and_envs():
    """A U(1) pair and a one-sweep chi=4 environment (no seed checks: the
    guards need only an environment of known chi, so the fixture stopped
    converging it in the 2026-10-26 test-trim pass)."""
    A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(1))
    A_n, B_n = normalized_pair(A, B)
    cfg = dataclasses.replace(
        CTMConfig(chi=CHI), on_unconverged="warn", max_iter=1, min_iter=1
    )
    envs, _ = python_loop_ctm_converge(
        {(0, 0): A_n, (1, 0): B_n},
        CHECKERBOARD_NEIGHBORS,
        **ctm_converge_kwargs(cfg),
    )
    assert envs[(0, 0)].C1.indices[0].dim == CHI  # regime: the chi under test
    return A, B, envs


def test_loss_refuses_envs_init_of_wrong_chi(pair_and_envs):
    _, _, envs = pair_and_envs
    with pytest.raises(ValueError, match="envs_init chi 4 .* ctm_cfg.chi 6"):
        make_2site_loss(xxz_gate_u1sz(0.3), CTMConfig(chi=6), envs)


def test_symmetric_gate_is_accepted(pair_and_envs):
    """The symmetric gate is densified: same loss as the dense gate (ruling R3)."""
    A, B, envs = pair_and_envs
    cfg = CTMConfig(chi=CHI)
    loss_sym, _ = make_2site_loss(xxz_gate_u1sz(0.3), cfg, envs)
    loss_dense, _ = make_2site_loss(xxz_gate(0.3), cfg, envs)
    e_sym, e_dense = float(loss_sym((A, B))), float(loss_dense((A, B)))
    assert abs(e_sym - e_dense) < 1e-12, (e_sym, e_dense)
