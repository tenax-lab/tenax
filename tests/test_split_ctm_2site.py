import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
)
from tenax.algorithms._split_ctm_tensor_init import SplitCTMTensorEnv


def _match_axes(src, ref):
    """Return a permutation of src's axes so its labels line up with ref's."""
    ref_labels = list(ref.labels())
    src_labels = list(src.labels())
    return tuple(src_labels.index(lbl) for lbl in ref_labels)


def _random_dense_A(D=2, d=2, seed=0):
    """5-leg (u,d,l,r,phys) DenseTensor iPEPS site tensor, trivial symmetry."""
    from tenax.algorithms._ctm_utils import _trivial_symmetry
    from tenax.core.index import FlowDirection, TensorIndex
    from tenax.core.tensor import DenseTensor

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((D, D, D, D, d)) + 0j
    sym = _trivial_symmetry()

    def idx(n, flow, label):
        return TensorIndex.from_charges(
            sym, np.zeros(n, dtype=np.int32), flow, label=label
        )

    return DenseTensor(
        data,
        (
            idx(D, FlowDirection.OUT, "u"),
            idx(D, FlowDirection.OUT, "d"),
            idx(D, FlowDirection.OUT, "l"),
            idx(D, FlowDirection.OUT, "r"),
            idx(d, FlowDirection.OUT, "phys"),
        ),
    )


def test_initialize_split_multisite_env_keys_and_type():
    A = _random_dense_A(seed=1)
    B = _random_dense_A(seed=2)
    envs = _initialize_split_multisite_env({(0, 0): A, (1, 0): B}, chi=6, chi_I=6)
    assert set(envs.keys()) == {(0, 0), (1, 0)}
    assert isinstance(envs[(0, 0)], SplitCTMTensorEnv)
    assert isinstance(envs[(1, 0)], SplitCTMTensorEnv)
    assert envs[(0, 0)].C1 is not envs[(1, 0)].C1


def test_split_multisite_uniform_matches_single_site():
    """recipe='1x1' multisite sweep on a uniform cell == single-site forward."""
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_multisite,
        ctm_split_tensor,
    )
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm_1site_split_tensor,
    )

    A = _random_dense_A(seed=3)
    chi = 6
    single = ctm_split_tensor(A, chi, max_iter=20, conv_tol=0.0)
    envs = _split_ctm_multisite(
        {(0, 0): A, (1, 0): A},
        CHECKERBOARD_NEIGHBORS,
        chi,
        max_iter=20,
        conv_tol=0.0,
        recipe="1x1",
    )
    rho_single = _rdm_1site_split_tensor(A, single)
    rho_multi = _rdm_1site_split_tensor(A, envs[(0, 0)])
    assert np.allclose(rho_single, rho_multi, atol=1e-8)


def _split_env_and_fused_env(A, chi):
    """Matched pair: split env and fused env from the same converged single-site
    CTM, for enlarged-corner parity (uniform 1x1, so envs are directly comparable)."""
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor

    fused, _eps = ctm_tensor(A, chi, max_iter=30, conv_tol=0.0)
    split = ctm_split_tensor(A, chi, chi_I=chi, max_iter=30, conv_tol=0.0)
    return split, fused


@pytest.mark.parametrize(
    "position", ["top_left", "top_right", "bottom_left", "bottom_right"]
)
def test_split_enlarged_corner_matches_fused(position):
    from tenax.algorithms._ctm_tensor_convergence import _build_double_layer_tensor
    from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
    from tenax.algorithms._split_ctm_tensor_moves import _build_split_enlarged_corner

    A = _random_dense_A(seed=5)
    chi = 6
    split, fused = _split_env_and_fused_env(A, chi)
    a = _build_double_layer_tensor(A)
    A_bar = A.bar()
    if position == "top_left":
        Q_ref = _build_enlarged_corner(
            fused.C1, fused.T1, fused.T4, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C1,
            split.T1_ket,
            split.T1_bra,
            split.T4_ket,
            split.T4_bra,
            A,
            A_bar,
            position=position,
        )
    elif position == "top_right":
        Q_ref = _build_enlarged_corner(
            fused.C2, fused.T1, fused.T2, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C2,
            split.T1_ket,
            split.T1_bra,
            split.T2_ket,
            split.T2_bra,
            A,
            A_bar,
            position=position,
        )
    elif position == "bottom_left":
        Q_ref = _build_enlarged_corner(
            fused.C4, fused.T3, fused.T4, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C4,
            split.T3_ket,
            split.T3_bra,
            split.T4_ket,
            split.T4_bra,
            A,
            A_bar,
            position=position,
        )
    else:  # bottom_right
        Q_ref = _build_enlarged_corner(
            fused.C3, fused.T3, fused.T2, a, position=position
        )
        Q_split = _build_split_enlarged_corner(
            split.C3,
            split.T3_ket,
            split.T3_bra,
            split.T2_ket,
            split.T2_bra,
            A,
            A_bar,
            position=position,
        )
    # Align split axes to ref label order, compare magnitudes up to gauge/normalization.
    Qs = Q_split.transpose(_match_axes(Q_split, Q_ref))
    dr = Q_ref.todense()
    ds = Qs.todense()
    dr = dr / np.max(np.abs(dr))
    ds = ds / np.max(np.abs(ds))
    assert dr.shape == ds.shape
    assert np.allclose(np.abs(dr), np.abs(ds), atol=1e-6)
