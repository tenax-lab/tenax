"""M2b strict regression: kagome iPESS dummy-bond hack vs rank-4 native path.

Mirrors the kagome iPESS smoke in ``tests/test_pess_ad.py`` (which lives on
``feat/coarse-grain-ipeps``) against the new
:func:`honeycomb_ctm_energy_implicit` entry point with
:func:`compute_honeycomb_triangle_energy` as the ``energy_fn`` override.
The two paths describe the same physical state, so converged energies at
fixed seed must agree within ``1e-3``.

**Status (2026-04-28): SKIPPED until the prerequisites land.** This test
is intentionally structural-only because PR #347 does not deliver the
kagome iPESS plumbing it depends on. PR #347 is the rank-4 honeycomb
CTM port; the kagome iPESS site construction, the dummy-bond brick-wall
``pess_optimize.py`` hack, and the kagome-specific gate constructor all
live on ``feat/coarse-grain-ipeps`` (commits ``f05166e``,
``32070f6``, ``a15c090``, ``6fa7039``).

The cross-path smoke in ``tests/test_ctm_honeycomb_cross_path.py``
(commit ``e879043``) covers the rank-4 CTM topology / RDM gate at the
honeycomb level (no optimization, ``d=2``, no kagome triangle), so the
*new* contract Task 17 strictly gates is:

1. The rank-4 path agrees with the *brick-wall dummy-bond hack* (not just
   with the standard square CTM on a hand-built supersite).
2. Agreement holds *after* both paths converge under L-BFGS (so any
   variational-vs-non-variational bias would surface).
3. Agreement holds at the *kagome iPESS shape* (``D=2, d=3, χ=8``,
   3-site triangle supersite + intra-triangle 3-spin energy_fn).

**To unskip:**

(a) Cherry-pick or merge the kagome iPESS site fixture from
    ``feat/coarse-grain-ipeps`` (the relevant pieces are
    ``coarse_grain.py:kagome_cg_gates`` and the supersite construction
    inside ``pess_optimize.py`` / ``tests/test_pess_ad.py``).
(b) Wire the dummy-bond path's ``optimize_gs_ad`` for the reference
    converged energy.
(c) Wire the rank-4 path with ``energy_fn=compute_honeycomb_triangle_energy``
    and the same kagome-supersite-on-honeycomb-cell ansatz.
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
        "Requires kagome iPESS site fixture + dummy-bond optimize_gs_ad "
        "reference, both on feat/coarse-grain-ipeps. See module docstring "
        "for the unskip checklist. The cross-path RDM smoke in "
        "test_ctm_honeycomb_cross_path.py covers the topology subset of "
        "this gate at the honeycomb (d=2) level."
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
