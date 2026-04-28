"""M2b strict regression: kagome iPESS — CG-iPEPS reference vs rank-4 native path.

The two-path agreement check this file scaffolds compares converged kagome
iPESS energies between:

* the **CG-iPEPS reference path** — 1-site square iPEPS with
  ``d_eff = 2**3 = 8`` (one kagome triangle per supersite), driven by
  :func:`tenax.optimize_gs_ad` with ``cg_gates=tenax.kagome_cg_gates()``;
* the **rank-4 native honeycomb path** introduced in PR #347 — 2-sublattice
  honeycomb tensors of shape ``(D, D, D, d_eff=8)``, driven by
  :func:`honeycomb_ctm_energy_implicit`.

Both express the same physical state, so converged energies at fixed seed
must agree within ``1e-3``.

**Status (2026-04-28): SKIPPED — partial unblock only.** PR #352 (merged)
delivered :func:`tenax.kagome_cg_gates`, :func:`tenax.compute_energy_cg`,
and the ``cg_gates`` hookpoint in :func:`tenax.optimize_gs_ad`, so the
*reference* leg is now available on ``main``. The native leg, however,
is still incomplete: :func:`compute_honeycomb_triangle_energy` only sums
1-site (intra-triangle) RDMs and does **not** include the 3 inter-triangle
bond terms (kagome ``h_inter["h" | "v" | "diag"]``). Comparing ground-state
energies without the inter-triangle terms would only test 1/4 of the
kagome Hamiltonian.

PR #347's actual correctness contract is covered without M2b by:

* :mod:`tests.test_ctm_honeycomb_ad` — FD-vs-AD agreement at D=2 (M1 gate);
* :mod:`tests.test_ctm_honeycomb_cross_path` — rank-4 CTM topology / RDM
  agreement vs the standard square CTM on a hand-built supersite at
  ``d=2`` (no optimization, no kagome triangle).

So M2b stays skipped; the missing piece belongs in a follow-up PR.

**To unskip (follow-up PR):**

(a) Extend the rank-4-path energy contract: write a kagome-aware
    ``energy_fn`` that combines :func:`compute_honeycomb_triangle_energy`
    (intra) with a 3-bond inter-triangle sum that contracts each of the
    honeycomb NN edges (``e0``, ``e1``, ``e2``) against the corresponding
    ``kagome_cg_gates().h_inter`` entry via the appropriate 2-site RDM.
(b) Wrap ``honeycomb_ctm_energy_implicit`` in an L-BFGS loop (the rank-4
    path is not currently exposed through ``optimize_gs_ad``).
(c) Build matching kagome supersite tensors for both paths from a shared
    seed: random ``(D, D, D, D, 8)`` for the CG path, and the same
    physical state mapped onto two ``(D, D, D, 8)`` honeycomb sublattice
    tensors for the rank-4 path.
(d) Run both at fixed seed, compare converged energies within 1e-3.
(e) Remove the ``pytest.mark.skip`` decorator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.honeycomb_ctm import (
    compute_honeycomb_triangle_energy,
    honeycomb_ctm_energy_implicit,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _heisenberg_3spin_triangle_op(d: int = 3) -> jnp.ndarray:
    """Placeholder for the intra-triangle 3-spin Heisenberg operator.

    Replaced at unskip time by the actual operator used by
    ``tests/test_pess_ad.py`` so both paths see the same Hamiltonian.
    """
    return jnp.eye(d, dtype=jnp.complex128)


def _make_kagome_triangle_supersite(D: int, d: int, key: jax.Array) -> DenseTensor:
    """Placeholder for the kagome triangle supersite tensor.

    Replaced at unskip time by the construction from
    ``coarse_grain.py:kagome_cg_gates`` / the iPESS pipeline so both
    paths start from the same physical state.
    """
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
        "PR #352 unblocks the CG-iPEPS reference leg (kagome_cg_gates + "
        "optimize_gs_ad), but the rank-4 native leg is still missing the "
        "inter-triangle energy_fn (compute_honeycomb_triangle_energy "
        "covers intra only). Belongs in a follow-up PR; see module "
        "docstring for the unskip checklist. M1 (FD-vs-AD) and the "
        "cross-path RDM smoke already cover PR #347's correctness "
        "contract at d=2."
    )
)
def test_kagome_ipess_native_matches_dummy_bond_hack():
    """Converged kagome iPESS energy: rank-4 native path vs brick-wall hack.

    Plan target: D=2, d=3, χ=8, fixed seed; both paths converge under
    L-BFGS; assert ``|E_native - E_hack| < 1e-3``.
    """
    D, d, chi = 2, 3, 8
    seed = 42

    A = _make_kagome_triangle_supersite(D=D, d=d, key=jax.random.PRNGKey(seed))
    B = _make_kagome_triangle_supersite(D=D, d=d, key=jax.random.PRNGKey(seed + 1))
    sites = {(0, 0): A, (1, 0): B}
    H_triangle = _heisenberg_3spin_triangle_op(d=d)

    # Rank-4 native path. Both paths would be wrapped in a short L-BFGS
    # at unskip time; here we just call the energy forward to exercise
    # the energy_fn=triangle hook end-to-end.
    E_native = honeycomb_ctm_energy_implicit(
        sites,
        H_triangle,
        chi=chi,
        max_iter=80,
        conv_tol=1e-8,
        projector_method="biorthogonal",
        forward_gauge="phase",
        energy_fn=compute_honeycomb_triangle_energy,
    )

    # Dummy-bond brick-wall path. Replaced at unskip time with the actual
    # pess_optimize / coarse-grain entry from feat/coarse-grain-ipeps.
    E_hack = E_native  # placeholder until the hack path is wired

    rel = abs(float(E_native) - float(E_hack)) / max(abs(float(E_hack)), 1e-12)
    assert rel < 1e-3, f"E_native={E_native}, E_hack={E_hack}, rel={rel:.3e}"
