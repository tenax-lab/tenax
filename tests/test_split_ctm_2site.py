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
