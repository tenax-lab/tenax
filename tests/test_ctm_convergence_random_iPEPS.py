"""Convergence regression for bug 3a (random complex iPEPS, cold init).

Pre-bug-3a-fix this test fails: Tenax converges to the paired-degenerate
fixed point (corner SVs ~ [0.68, 0.68, 0.20, 0.20]) where variPEPS
converges to the physical [0.95, 0.22, 0.19, 0.13] in ~10 iters.

The post-fix SVs are ``[0.937, 0.302, 0.143, 0.102]`` (non-degenerate,
hierarchical) — close to the variPEPS reference. At seed 0 the loop
converges to ``sv_diff ≈ 9e-4`` in ~13 iterations under
``conv_tol=1e-3``. The 1e-3 tolerance (vs the design-doc 1e-5) is
chosen because a residual oscillation noted in
``project_ctm_two_init_bugs_found.md`` is a separate downstream issue
not in scope for bug 3a; the smoking-gun goal here is to *distinguish*
the physical fp from the pre-fix paired-degenerate plateau
(``sv_diff ~ 5e-2``), which 1e-3 does cleanly.

See ``project_ctm_two_init_bugs_found.md`` and
``docs/plans/2026-05-11-ctm-bug-3a-design.md``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


@pytest.mark.algorithm
def test_random_complex_ipeps_converges_to_physical_fp():
    """D=2 chi=4 random complex iPEPS converges to non-degenerate physical fp."""
    D, d, chi = 2, 2, 4
    sym = U1Symmetry()
    rng = np.random.RandomState(0)
    raw = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal(
        (D, D, D, D, d)
    )
    raw = raw / np.linalg.norm(raw)
    indices = (
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="u"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="d"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.OUT, label="l"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(D, dtype=np.int32), FlowDirection.IN, label="r"
        ),
        TensorIndex.from_charges(
            sym, np.zeros(d, dtype=np.int32), FlowDirection.IN, label="phys"
        ),
    )
    A = DenseTensor(jnp.array(raw), indices)

    site_tensors = {(0, 0): A}
    envs_final, info = python_loop_ctm_converge(
        site_tensors,
        SINGLE_SITE_NEIGHBORS,
        chi=chi,
        max_iter=50,
        conv_tol=1e-3,
        projector_method="svd",
    )

    # Convergence reached. (1e-3 distinguishes physical fp from the
    # paired-degenerate fp, which plateaus at sv_diff ~ 5e-2.)
    assert info.converged, (
        f"CTM did not converge in 50 iters at tol=1e-3; "
        f"sv_diff={info.sv_diff}, iterations={info.iterations}"
    )

    # Inspect leading C1 corner SVs.
    C1 = envs_final[(0, 0)].C1
    C1_dense = C1.todense() if hasattr(C1, "todense") else C1._data
    sv = jnp.linalg.svd(C1_dense, compute_uv=False)
    sv = np.asarray(sv)
    sv_sorted = np.sort(sv)[::-1]

    # Physical fixed point: leading SV ~0.95, decaying. The pre-fix
    # paired-degenerate fp has SVs [0.68, 0.68, 0.20, 0.20].
    assert sv_sorted[0] > 0.85, (
        f"leading SV {sv_sorted[0]:.4f} is too small (paired-degenerate fp?); "
        f"sv_sorted={sv_sorted}"
    )
    # Reject paired degeneracy.
    assert abs(sv_sorted[0] - sv_sorted[1]) > 0.05, (
        f"top two SVs nearly degenerate: {sv_sorted[:2]}"
    )
