"""#463 Phase 3 — 2-site split-CTM SymmetricTensor support (bosonic U(1)/Zn).

Trivial-U(1) parity (Tasks 2/3) validates the polymorphic symmetric path is
correct (symmetric == dense). The nontrivial-charge sector-preservation smoke
(Task 1) is the red->green gate for per-sector interlayer-SVD truncation: without
base_charges the 2plaq path uses GLOBAL truncation, which starves a charge sector
across sweeps. Mirrors the single-site regression
test_fermionic_u1_charges_preserved_across_sweeps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
    _split_ctm_sweep_multisite_2x2,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor
from tests.test_split_ctm_2site import _build_su_neel, _heisenberg_gate


def _bond_sector_count(env_tensor, interlayer_label):
    """Number of distinct charge sectors on a tensor's interlayer bond leg."""
    pos = env_tensor.labels().index(interlayer_label)
    return len(np.unique(np.asarray(env_tensor.indices[pos].charges)))


def _min_sector_count(env_tensor, interlayer_label):
    """Minimum number of bond slots allocated to any charge sector."""
    pos = env_tensor.labels().index(interlayer_label)
    charges = np.asarray(env_tensor.indices[pos].charges)
    _, counts = np.unique(charges, return_counts=True)
    return int(np.min(counts)) if len(counts) > 0 else 0


def _build_nontrivial_u1_pair(D=2, d=2):
    """Direction-dependent (A != B) nontrivial bosonic-U(1) checkerboard pair.

    Virtual/phys charges [0, 1] give two competing sectors, so global vs
    per-sector interlayer truncation differ. Requires D == 2 and d == 2.
    """
    assert D == 2 and d == 2, "helper hard-codes the [0,1] two-sector layout"
    sym = U1Symmetry()
    vc = np.array([0, 1], dtype=np.int32)
    pc = np.array([0, 1], dtype=np.int32)
    flows = (
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.IN,
    )
    labels = ("u", "d", "l", "r", "phys")
    charge_sets = (vc, vc, vc, vc, pc)
    indices = tuple(
        TensorIndex.from_charges(sym, c.copy(), f, label=lbl)
        for c, f, lbl in zip(charge_sets, flows, labels)
    )
    kA, kB = jax.random.split(jax.random.PRNGKey(7))
    A = SymmetricTensor.random_normal(indices, kA)
    B = SymmetricTensor.random_normal(indices, kB)
    return A, B


def test_2site_symmetric_charge_sectors_preserved():
    """Nontrivial-charge 2-site split sweeps stay finite, remain SymmetricTensor,
    and preserve per-sector interlayer-SVD budget (no sector starvation).

    Global truncation (base_charges=None) allocates SVD slots to the sector
    with the largest singular values, leaving the weaker sector with fewer than
    chi_I // n_sectors slots.  Per-sector truncation (base_charges engaged)
    guarantees each sector gets at least floor(chi_I / n_sectors) slots.

    Red before the fix: global top-k gives 3 slots to sector 0 and 1 to
    sector 1 (min_count=1 < chi_I//2=2). Green after: per-sector gives 2+2.
    """
    A, B = _build_nontrivial_u1_pair(D=2, d=2)
    site_tensors = {(0, 0): A, (1, 0): B}
    bars = {c: t.bar() for c, t in site_tensors.items()}
    chi, chi_I = 6, 4

    envs = _initialize_split_multisite_env(site_tensors, chi, chi_I)
    for _ in range(3):
        envs = _split_ctm_sweep_multisite_2x2(
            envs, site_tensors, bars, CHECKERBOARD_NEIGHBORS, chi, chi_I
        )

    n_sectors = 2  # U(1) with charges [0, 1]
    fair_share = chi_I // n_sectors  # 2

    for coord, env in envs.items():
        for t in env:
            assert isinstance(t, SymmetricTensor), (
                f"{coord} env tensor collapsed to non-symmetric type"
            )
            assert jnp.all(jnp.isfinite(t.todense())), (
                f"{coord} env tensor non-finite after sweeps"
            )
        # The T1 ket interlayer bond must retain BOTH input sectors with a
        # fair share of slots: per-sector truncation guarantees >= fair_share
        # per sector; global truncation gives 3+1 (min=1 < fair_share=2).
        assert _bond_sector_count(env.T1_ket, "t1k_I") >= 2, (
            f"{coord}: interlayer bond collapsed to a single charge sector"
        )
        assert _min_sector_count(env.T1_ket, "t1k_I") >= fair_share, (
            f"{coord}: interlayer bond sector starvation detected — "
            f"min slots per sector = {_min_sector_count(env.T1_ket, 't1k_I')} "
            f"< fair_share = {fair_share}. "
            f"Global truncation dropped slots from the weaker charge sector "
            "(base_charges not engaged for per-sector interlayer-SVD truncation)"
        )


def _to_trivial_u1(A):
    """Wrap a dense iPEPS site (shape (D,D,D,D,d), labels u,d,l,r,phys) as a
    trivial-U(1) SymmetricTensor with the same data — robust to whatever indices
    the SU builder attached."""
    data = A.todense()
    sym = U1Symmetry()
    flows = (
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.OUT,
        FlowDirection.IN,
        FlowDirection.IN,
    )
    labels = ("u", "d", "l", "r", "phys")
    indices = tuple(
        TensorIndex.from_charges(sym, np.zeros(n, dtype=np.int32), f, label=lbl)
        for n, f, lbl in zip(data.shape, flows, labels)
    )
    return SymmetricTensor.from_dense(data, indices)


@pytest.mark.parametrize("D,chi", [(2, 4), (2, 8), (3, 8)])
def test_2site_symmetric_energy_matches_dense(D, chi):
    """Trivial-U(1) symmetric split energy == dense split energy on a convergent
    Neel checkerboard. The D=3 case is also the design-§10 hard-fusion guard."""
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    A, B = _build_su_neel(D=D)
    gate = _heisenberg_gate()

    envA_d, envB_d = ctm_split_tensor_2site(
        A, B, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_dense = float(
        compute_energy_split_ctm_tensor_2site(A, B, envA_d, envB_d, gate, d=2)
    )

    As, Bs = _to_trivial_u1(A), _to_trivial_u1(B)
    envA_s, envB_s = ctm_split_tensor_2site(
        As, Bs, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_sym = float(
        compute_energy_split_ctm_tensor_2site(As, Bs, envA_s, envB_s, gate, d=2)
    )

    assert np.isfinite(E_sym), f"symmetric energy not finite: {E_sym}"
    assert abs(E_sym - E_dense) < 1e-6, (
        f"sym={E_sym} dense={E_dense} (D={D}, chi={chi})"
    )


def test_2site_symmetric_forward_converges_d3():
    """#463 Phase 3 root-cause regression: the D=3 trivial-U(1) 2-site split-CTM
    forward must CONVERGE (finite, non-zero, == dense-split), not collapse.

    Two bugs (both fixed) made the D=3 symmetric forward degenerate:

    1. ``_init_symmetric_corner`` seeded ``eye(min(chi, D))`` instead of the
       dense path's rank-1 ``(0, 0) = 1`` corner — an artificially degenerate
       corner whose subspace rotates every sweep, so the env never converges
       element-wise.

    2. ``_build_split_enlarged_corner`` returned its legs in ``(d2, chi)`` order
       while ``_build_enlarged_corner`` (fused reference) uses ``(chi, d2)``.
       The block-sparse ``tensor_svd`` in ``_compute_2x2_projector`` reshapes
       each sector block *in native axis order* into ``(left_rows, right_cols)``
       without first transposing to ``left_labels=(chi, d2)`` order, so the
       wrong-order corner transposed the SVD row basis → the projector kept a
       different subspace → the env drifted and the energy collapsed to exactly
       0.0 from ~iter 8 at D=3.

    Together they collapsed the energy: ``-0.523 → -0.348 → … → 0.0``.  This
    test is RED before either fix (energy is 0.0 / not close to dense) and
    GREEN after both.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )

    D, chi = 3, 8
    A, B = _build_su_neel(D=D)
    gate = _heisenberg_gate()

    envA_d, envB_d = ctm_split_tensor_2site(
        A, B, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_dense = float(
        compute_energy_split_ctm_tensor_2site(A, B, envA_d, envB_d, gate, d=2)
    )

    As, Bs = _to_trivial_u1(A), _to_trivial_u1(B)
    envA_s, envB_s = ctm_split_tensor_2site(
        As, Bs, chi, max_iter=60, conv_tol=1e-10, chi_I=chi
    )
    E_sym = float(
        compute_energy_split_ctm_tensor_2site(As, Bs, envA_s, envB_s, gate, d=2)
    )

    # Finite, non-zero (the collapse drove it to exactly 0.0), and == dense.
    assert np.isfinite(E_sym), f"symmetric energy not finite: {E_sym}"
    assert abs(E_sym) > 1e-2, (
        f"symmetric energy collapsed toward zero: {E_sym} (D={D}, chi={chi})"
    )
    assert abs(E_sym - E_dense) < 1e-6, (
        f"sym={E_sym} dense={E_dense} (D={D}, chi={chi})"
    )
