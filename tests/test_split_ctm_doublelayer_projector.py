import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard

pytestmark = pytest.mark.core


def _oracle():
    # import helper module robustly whether or not `tests` is a package
    try:
        from tests._split_ctm_oracle import (
            fused_env_to_split,
            heisenberg_gate,
            make_site,
        )
    except ModuleNotFoundError:
        from _split_ctm_oracle import fused_env_to_split, heisenberg_gate, make_site
    return make_site, heisenberg_gate, fused_env_to_split


@pytest.mark.parametrize("D", [2, 3])
def test_fused_to_split_roundtrip(D):
    make_site, heisenberg_gate, fused_env_to_split = _oracle()
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=8, max_iter=200, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = fused_env_to_split(fused_env, D, chi_I=8 * D)
    rt_env = _split_env_to_tensor_standard(split_env)
    E_rt = float(compute_energy_ctm_tensor(A, rt_env, gate))
    np.testing.assert_allclose(E_rt, E_fused, atol=1e-8)


from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor
from tenax.algorithms._split_ctm_tensor_energy import compute_energy_split_ctm_tensor


@pytest.mark.xfail(
    reason="#463: split moves use per-layer projector; fixed in DL-Task 5", strict=True
)
@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 6)])
def test_split_matches_fused_lossless_chi_I(D, chi):
    make_site, heisenberg_gate, fused_env_to_split = _oracle()
    A = make_site(D, 2, seed=7)
    gate = heisenberg_gate()
    fused_env, _ = ctm_tensor(A, chi=chi, max_iter=300, conv_tol=1e-12)
    E_fused = float(compute_energy_ctm_tensor(A, fused_env, gate))
    split_env = ctm_split_tensor(
        A, chi=chi, chi_I=chi * D, max_iter=300, conv_tol=1e-12
    )
    E_split = float(compute_energy_split_ctm_tensor(A, split_env, gate))
    np.testing.assert_allclose(E_split, E_fused, atol=1e-8)


def test_factorize_projector_reconstructs():
    # P over (env, ketD, braD) -> chi factorizes exactly into P_first . P_second
    import jax.numpy as jnp  # noqa: F401

    from tenax.algorithms._split_ctm_tensor_moves import _factorize_projector
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.symmetry import U1Symmetry
    from tenax.core.tensor import DenseTensor

    sym = U1Symmetry()
    env, Dk, Db, chi = 4, 2, 2, 5
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (env, Dk, Db, chi))
    z = lambda n: __import__("numpy").zeros(n, dtype="int32")  # noqa: E731
    idx = [
        TensorIndex.from_charges(sym, z(env), FlowDirection.IN, label="env"),
        TensorIndex.from_charges(sym, z(Dk), FlowDirection.IN, label="ketD"),
        TensorIndex.from_charges(sym, z(Db), FlowDirection.IN, label="braD"),
        TensorIndex.from_charges(sym, z(chi), FlowDirection.OUT, label="chi_new"),
    ]
    P = DenseTensor(data, idx)
    P_first, P_second, m = _factorize_projector(P, "env", "ketD", "braD", "chi_new")
    # contract P_first . P_second over the factorization bond -> reconstruct P
    from tenax.contraction.contractor import contract

    P_rec = contract(P_first, P_second)
    # compare dense values up to leg order
    a = np.asarray(P.todense())
    b = np.asarray(
        P_rec.transpose(
            tuple(
                P_rec.labels().index(lbl) for lbl in ["env", "ketD", "braD", "chi_new"]
            )
        ).todense()
    )
    np.testing.assert_allclose(b, a, atol=1e-10)
