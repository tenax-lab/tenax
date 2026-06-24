"""Memory regression for the split-aware coarse-grained (CG) kagome energy.

The kagome lattice has three sites per unit cell, so to use the 1-site (C4v)
CTM algorithm the triangle is coarse-grained into a single supersite with
effective physical dimension ``d_eff = 2**3 = 8``. The split-aware CG energy
(:func:`compute_energy_cg_split`) routes every RDM through the split path.

Scaling reality (see ``project_kagome_multisite_ipeps_plan``): the CG diagonal
RDM intrinsically peaks at ``chi^2 * D^4 * d^2 * d_eff^2`` (``d = 2`` micro,
``d_eff = 8`` super). The ``d_eff^2 = 64`` factor means the split path only
undercuts the std ``chi^2 * D^6`` path for ``D >~ 16``; at the feasible sizes
probed here it ties the shim in *memory* but must agree with it to machine
precision. Consequently ``D = 4``, ``chi = 32`` is the feasible probe point
(measured ~0.25 GB); ``D = 6`` already exceeds ~25 GB and ``D = 8 chi = 128``
needs ~64 GB — the historical "fits an 8 GB box at D=8 chi=128" target dropped
the ``d_eff^2`` factor and was never attainable for the CG workload.

These probes guard:
  1. correctness — split CG energy == shim CG energy on the same env (exact);
  2. memory — the split CG energy peak stays under a realistic budget at a
     feasible ``(D, chi)``.

The harness ``examples.kagome_spin12_pess_liao2017_replication`` must expose
``build_canonical_pess`` / ``heisenberg_gate`` (PR #387 follow-up).
"""

import tracemalloc

import jax
import pytest

# Skip the entire module if the canonical Liao replication harness isn't here.
liao_harness = pytest.importorskip("examples.kagome_spin12_pess_liao2017_replication")

build_canonical_pess = getattr(liao_harness, "build_canonical_pess", None)
heisenberg_gate_fn = getattr(liao_harness, "heisenberg_gate", None)
if build_canonical_pess is None or heisenberg_gate_fn is None:
    pytest.skip(
        "harness lacks build_canonical_pess/heisenberg_gate helpers "
        "(PR #387 follow-up)",
        allow_module_level=True,
    )

from tenax.algorithms._split_ctm_tensor_energy import (  # noqa: E402
    _split_env_to_tensor_standard,
)
from tenax.algorithms.coarse_grain import (  # noqa: E402
    compute_energy_cg,
    compute_energy_cg_split,
)

# ``heisenberg_gate`` is part of the harness API (used by the skip guard and
# available for standalone probes); ``build_canonical_pess`` already returns the
# matching ``cg_gates`` bundle, so the tests consume that directly.
_ = heisenberg_gate_fn

# Feasible CG probe point. The split-aware CG diagonal RDM peaks at
# ``chi^2 * D^4 * d^2 * d_eff^2``; at (D=4, chi=32) that is ~0.25 GB measured.
_D, _CHI = 4, 32
_BUDGET_GB = 1.5  # ~6x headroom over the ~0.25 GB measured peak


@pytest.fixture(scope="module")
def canonical_d4():
    """Converged split-CTM env for the canonical kagome supersite at D=4, chi=32."""
    return build_canonical_pess(D=_D, chi=_CHI)


@pytest.mark.slow
def test_kagome_cg_split_energy_matches_shim(canonical_d4):
    """Split-aware CG energy must equal the std (shim) CG energy on the same env.

    Replaces the stale ``-0.347185`` magic number (which predates later CTM
    refactors and no longer reproduces on any current path) with the live
    invariant: the ``chi^2 * D^4 * d_eff^2`` split path agrees with the
    ``chi^2 * D^6`` shim path to machine precision for an identical
    environment.
    """
    A, env, cg_gates, d_eff = canonical_d4
    E_split = compute_energy_cg_split(A, env, cg_gates, d_eff)
    env_std = _split_env_to_tensor_standard(env)
    E_shim = compute_energy_cg(A, env_std, cg_gates, d_eff)
    assert abs(float(E_split) - float(E_shim)) < 1e-9


@pytest.mark.slow
def test_kagome_cg_split_energy_peak_under_budget(canonical_d4):
    """Split-aware CG energy peak stays under budget at the feasible CG size.

    ``tracemalloc`` tracks host allocations only, so this is a CPU-backend
    probe (skipped on GPU/TPU where device memory is untracked). Only the
    energy contraction is measured — the CTM convergence inside
    ``build_canonical_pess`` is excluded.

    Scope note: this guards the *energy evaluation* peak. The split-CTM forward
    *convergence* is NOT ``chi^2 * D^4``-bounded at large D (it peaks ~32 GB for
    a bare d=2 site at D=8 chi=128); that large-D convergence-memory limitation
    is tracked separately, not by this test.
    """
    if jax.default_backend() != "cpu":
        pytest.skip("tracemalloc tracks host memory only; CPU-backend probe")

    A, env, cg_gates, d_eff = canonical_d4

    tracemalloc.start()
    E = compute_energy_cg_split(A, env, cg_gates, d_eff)
    _ = jax.block_until_ready(E)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak / 1024**3
    assert peak_gb < _BUDGET_GB, f"peak {peak_gb:.3f} GB exceeds {_BUDGET_GB} GB budget"
