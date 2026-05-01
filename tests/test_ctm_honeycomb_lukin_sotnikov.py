"""M2a regression: Lukin-Sotnikov uniform-iPEPS Heisenberg honeycomb.

Reference: Lukin-Sotnikov *et al.*, **PRB 107, 054424 (2023)**, Table I —
ground-state energy per site of the spin-1/2 Heisenberg model on the
honeycomb lattice as a function of bond dimension ``D``.

**Status (2026-04-27): SKIPPED until the prerequisites land.** This test
is intentionally structural-only because it has two prerequisites that
are not part of PR #347:

1. **Honeycomb simple update / L-BFGS optimizer** to bring a random-init
   site close to the variational minimum. The bare CTM at random A=B does
   not converge (verified during Task 12 development: env element-wise
   diff plateaus at ~2.0 regardless of χ for biorthogonal projectors on
   random tensors). Without an SU-initialized state, the energy after
   200 CTM sweeps remains in the +0.1 to +0.4 range, nowhere near the
   published ``D=2`` energy.
2. **Lukin-Sotnikov Table I numerical entry** for ``D=2`` (and likely
   the corresponding χ ≥ 20 it was computed at). The plan
   (``docs/plans/2026-04-25-honeycomb-ctm-plan.md`` Task 16) flags this
   as a paper lookup before merge.

Adjacent memory data points (variPEPS reference, complex128, after AD
optimization):

* ``project_complex128_benchmark_2026_04_21.md`` — D=2 χ=16: E=-0.6406.
* ``project_honeycomb_cg_benchmark_2026_04_19.md`` — D=3 χ=16: E=-0.6521.

These set the ballpark of the literature value but do NOT replace it for
the regression-test gate (the literature value is the authoritative
target).

**To unskip:** (a) implement honeycomb SU + a thin L-BFGS wrapper around
``honeycomb_ctm_energy_implicit`` (or run an existing optimizer with the
new ``energy_fn`` injected); (b) replace ``LUKIN_SOTNIKOV_D2_ENERGY``
with the actual paper value; (c) remove the ``pytest.mark.skip``
decorator and verify the assertion passes at the prescribed χ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.honeycomb_ctm import honeycomb_ctm_energy_implicit
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

# Placeholder — replace with actual Lukin-Sotnikov 2023 Table I, D=2 entry
# before unskipping. The current value is a memory-derived ballpark, NOT
# the authoritative literature number.
LUKIN_SOTNIKOV_D2_ENERGY = -0.5443


def _heisenberg_bond_xxz(d: int = 2) -> jnp.ndarray:
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return jnp.asarray(
        np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz),
        dtype=jnp.complex128,
    )


def _make_random_honeycomb_site(D: int, d: int, key: jax.Array) -> DenseTensor:
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    data = (re + 1j * im).astype(jnp.complex128)
    return DenseTensor(data, indices)


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Requires honeycomb SU/L-BFGS to converge from random init AND a "
        "looked-up Lukin-Sotnikov Table I D=2 reference value. See module "
        "docstring for the unskip checklist."
    )
)
def test_lukin_sotnikov_d2_uniform_reproduction():
    """Heisenberg honeycomb ground-state energy at D=2 reproduces literature.

    With a short L-BFGS minimization from random init (or from an SU output),
    the converged energy at χ ≥ 20 should match Lukin-Sotnikov Table I to
    within 5%.
    """
    A = _make_random_honeycomb_site(D=2, d=2, key=jax.random.PRNGKey(42))
    sites = {(0, 0): A, (1, 0): A}  # uniform A=B per the published setup
    H = _heisenberg_bond_xxz()

    # NOTE: this is the test as written assuming honeycomb SU/L-BFGS is wired.
    # Until that lands, this test is skipped. The body remains as a contract
    # for what the unskipped run will do.
    energy = honeycomb_ctm_energy_implicit(
        sites,
        H,
        chi=20,
        max_iter=200,
        conv_tol=1e-10,
        projector_method="biorthogonal",
        forward_gauge="phase",
        chi_ramp=[(4, 20), (10, 20), (20, None)],
    )
    rel_err = abs(float(energy) - LUKIN_SOTNIKOV_D2_ENERGY) / abs(
        LUKIN_SOTNIKOV_D2_ENERGY
    )
    assert rel_err < 0.05, (
        f"E={float(energy):.4f}, expected≈{LUKIN_SOTNIKOV_D2_ENERGY:.4f} "
        f"(rel err {rel_err:.2%}, must be < 5%)"
    )
