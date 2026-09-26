"""Fermionic 2-site split CTM: refused since #1035 step 4.

#463 Phase 4 routed fermionic split envs through ``merge → fused sweep →
resplit``, and this file pinned that route: one split sweep equal to one fused
sweep, and the edge resplit round trip.  Both conversions are sign-free
(``contract`` + ``fuse_indices`` on fermionic legs), so under the graded
contraction of #1035 they hand the fused sweep a hard-core-boson environment,
and the corners come out with inconsistent flows (``C1.c1_d`` vs ``T4.t4_d``).

Fermions now run on the fused graded Tensor CTM (``ctm_tensor_2site``), which
``tests/test_graded_ctm.py`` checks against an exact finite patch.  The split
CTM refuses fermionic input instead of returning a wrong environment; these
tests pin the refusal on both the public and the sweep entry points.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FermionParity = pytest.importorskip("tenax.core.symmetry").FermionParity

from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    CHECKERBOARD_NEIGHBORS as NB,
)
from tenax.algorithms._split_ctm_tensor_convergence import (  # noqa: E402
    _initialize_split_multisite_env,
    _split_ctm_sweep_multisite_2x2,
    ctm_split_tensor_2site,
)
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.tensor import SymmetricTensor  # noqa: E402


def _make_fp_site(D=2, seed=0):
    """Uniform-compatible FermionParity iPEPS site (``A.l`` charges == ``A.r``)."""
    sym = FermionParity()
    ch = np.array(([0, 1] * D)[:D], dtype=np.int32)
    phys = np.array([0, 1], dtype=np.int32)

    def idx(c, flow, label):
        return TensorIndex.from_charges(sym, c, flow, label=label)

    return SymmetricTensor.random_normal(
        (
            idx(ch, FlowDirection.OUT, "u"),
            idx(ch, FlowDirection.IN, "d"),
            idx(ch, FlowDirection.OUT, "l"),
            idx(ch, FlowDirection.IN, "r"),
            idx(phys, FlowDirection.IN, "phys"),
        ),
        jax.random.PRNGKey(seed),
    )


def test_the_fermionic_split_sweep_is_refused():
    A = _make_fp_site(seed=11)
    site_tensors = {(0, 0): A, (1, 0): A}
    envs = _initialize_split_multisite_env(site_tensors, 6, 6)
    bars = {c: T.bar() for c, T in site_tensors.items()}
    with pytest.raises(NotImplementedError, match="split CTM: fermionic"):
        _split_ctm_sweep_multisite_2x2(envs, site_tensors, bars, NB, 6, 6)


def test_the_fermionic_split_2site_public_path_is_refused():
    A, B = _make_fp_site(seed=1), _make_fp_site(seed=2)
    with pytest.raises(NotImplementedError, match="split CTM: fermionic"):
        ctm_split_tensor_2site(A, B, 6, max_iter=2)
