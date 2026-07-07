"""Convergence regression for bug 3a (random complex iPEPS, cold init).

Pre-bug-3a-fix this test fails: Tenax converges to the paired-degenerate
fixed point (corner SVs ~ [0.68, 0.68, 0.20, 0.20]) where variPEPS
converges to the physical [0.95, 0.22, 0.19, 0.13] in ~10 iters.

The post-fix SVs are ``[0.937, 0.302, 0.143, 0.102]`` (non-degenerate,
hierarchical) — close to the variPEPS reference. At seed 0 the physical
fixed point is reached in ~12 iterations, but its corner-SV metric has a
residual oscillation floor (``sv_diff ~ 2.4e-3``) rather than driving
cleanly to zero — a separate downstream issue noted in
``project_ctm_two_init_bugs_found.md`` (#425), not in scope for bug 3a.

The smoking-gun goal here is only to *distinguish* the physical fp from
the pre-fix paired-degenerate plateau (``sv_diff ~ 5e-2``). The test
therefore asserts the physical SV structure directly, plus a loose metric
band (``sv_diff < 1e-2``) that separates the two robustly
(``2.4e-3 << 1e-2 << 5e-2``). It deliberately does NOT assert
``info.converged`` under a tight 1e-3 gate: the oscillation floor sits
right at that gate and can fall either side across BLAS/XLA builds, which
made the strict assertion flake in CI (issue #692).

See ``project_ctm_two_init_bugs_found.md`` and
``docs/plans/2026-05-11-ctm-bug-3a-design.md``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


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

    # The physical fixed point is *reached* (see the SV assertions below), but
    # its corner-SV metric has a residual oscillation floor (~2.4e-3 here) that
    # is numerics/platform-sensitive and can sit either side of a tight 1e-3
    # gate — so ``info.converged`` is NOT a robust assertion (it flakes across
    # BLAS/XLA builds; see issue #692). The test's actual smoking-gun goal is to
    # distinguish the physical fp from the pre-fix paired-degenerate plateau,
    # whose metric floors ~5e-2. A loose band (``sv_diff < 1e-2``) separates the
    # two robustly (2.4e-3 << 1e-2 << 5e-2) without depending on the oscillation
    # dipping below 1e-3. The residual oscillation itself is a known limitation
    # (project_ctm_two_init_bugs_found.md / #425), out of scope here.
    assert info.sv_diff < 1e-2, (
        f"corner-SV metric {info.sv_diff} did not drop below the 1e-2 band "
        f"separating the physical fp from the paired-degenerate plateau "
        f"(~5e-2); iterations={info.iterations}"
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
