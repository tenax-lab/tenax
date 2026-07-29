"""Shared fixtures for the Tenax test suite."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

# ------------------------------------------------------------------ #
# Auto-apply markers based on test file name                           #
# ------------------------------------------------------------------ #

_FILE_MARKERS = {
    "test_tensor.py": "core",
    "test_index.py": "core",
    "test_symmetry.py": "core",
    "test_contraction.py": "core",
    "test_network.py": "core",
    "test_netfile.py": "core",
    "test_fermionic.py": "core",
    "test_dmrg.py": "algorithm",
    "test_idmrg.py": "algorithm",
    "test_itebd.py": "algorithm",
    "test_trg.py": "algorithm",
    "test_hotrg.py": "algorithm",
    "test_ipeps.py": "algorithm",
    "test_ipeps_core.py": "core",
    "test_auto_mpo.py": "algorithm",
    "test_ad_utils.py": "algorithm",
    "test_fermionic_ipeps.py": "algorithm",
    "test_fpeps_ad.py": "algorithm",
    "test_fermionic_ed_reference.py": "algorithm",
    "test_ipeps_excitations.py": "algorithm",
    "test_code_review_regressions.py": "core",
    "test_tensor_utils.py": "core",
    "test_split_ctm_tensor.py": "algorithm",
    "test_split_ctm_2site.py": "algorithm",
    # 2-site split-CTM SymmetricTensor (#463 Phase 3): forward/energy-parity
    # tests run in the algorithm bucket; the AD parity tests in this file carry
    # their own explicit @pytest.mark.slow so they stay out of -m "not slow".
    "test_split_ctm_2site_symmetric.py": "algorithm",
    # 2-site split-CTM fermionic (#463 Phase 4): block-sparse forward parity vs
    # the fused sweep; algorithm bucket like its symmetric/split siblings.
    "test_split_ctm_2site_fermionic.py": "algorithm",
    # Dense 2-site split-CTM AD (#463 Phase 2): each parity test converges a
    # coupled fixed point + a full AD gradient (~5 min); slow-only so they stay
    # out of the -m core CI gate and the -m "not slow" bucket.
    "test_split_ctm_2site_ad.py": "slow",
    "test_ctm_tensor.py": "algorithm",
    "test_integration_regression.py": "algorithm",
    "test_krylov.py": "core",
    "test_tdvp.py": "algorithm",
    "test_ctm_tensor_c4v.py": "algorithm",
    # Root implicit AD for C4v CTMRG (#715 Phase 0): each parity test runs a
    # full CTM convergence plus finite-difference reruns.
    "test_ctm_c4v_root_implicit.py": "algorithm",
    "test_ctm_truncation_error.py": "core",
    "test_ctm_paired.py": "algorithm",
    "test_ctm_python_loop.py": "algorithm",
    "test_rsvd.py": "core",
    "test_cbe.py": "algorithm",
    "test_lattice.py": "algorithm",
    "test_linalg.py": "core",
    "test_linalg_np.py": "core",
    "test_observables.py": "algorithm",
    "test_cbe_validation.py": "algorithm",
    "test_mps.py": "core",
    "test_blas_plan.py": "core",
    "test_padded_block_array.py": "core",
    "test_jit_sweep.py": "core",
    "test_padded_linalg.py": "core",
    "test_lanczos_np.py": "algorithm",
    "test_block_array.py": "core",
    "test_dmrg3s.py": "algorithm",
    "test_dmrg_cython.py": "algorithm",
    "test_ipeps_config.py": "core",
    "test_pess.py": "algorithm",
    "test_pess_ad.py": "algorithm",
    "test_pess_validation.py": "slow",
    "test_chi_auto_bump.py": "core",
    "test_ctm_env_pad.py": "core",
    "test_ipeps_chi_bump_integration.py": "algorithm",
    "test_ctm_convergence_random_iPEPS.py": "algorithm",
    "test_ipeps_grad_spike_guard.py": "slow",
    # Block-sparse stacked-contraction backend seam (#200 / #566): fast,
    # mechanism-level (small even-D fermionic tensors, no CTM convergence).
    "test_harness.py": "core",
    "test_stacked_view.py": "core",
    "test_stacked_contract.py": "core",
    "test_stacked_decomp.py": "core",
    "test_backend_dispatch.py": "core",
    "test_vjp_seam.py": "core",
    "test_vjp_seam_cutensor.py": "core",
    "test_stacked_tensor_dtype.py": "core",
    # GSPMD-sharded dense CTM (large-D rung 1): fast mechanism unit tests
    # (mesh specs, commit helpers, config default) gate every PR; the heavy
    # subprocess parity sweep (N=2,4 fake CPU devices) is slow-only.
    "test_ctm_sharding.py": "core",
    "test_showcase_scaling.py": "core",
    "test_heisenberg_d4_chi_scaling.py": "core",
    "test_heisenberg_d8_chi_scaling.py": "core",
    "test_ctm_sharding_parity.py": "slow",
    # Chunked dense-CTM edge absorption (chunk x shard #632 Increment 1): fast
    # mechanism unit tests; collected as core so CI required checks run them.
    "test_ctm_chunked_absorb.py": "core",
    # gs_recipe="2x2" (the iPEPSConfig default) production guard (#676 / #702).
    # Only the forward-only cell-size consistency half is core; the end-to-end
    # physical anchor lives in test_ctm_recipe_2x2_production_correctness.py so
    # it does not inherit this marker and run in required CI (it is @slow).
    "test_ctm_recipe_2x2_consistency.py": "core",
    # #632 Increment 2 gate byproduct: forward-chunk grads through the (monolith)
    # implicit-AD backward stay exact. Tiny D=2 CPU value_and_grad parity.
    "test_ctm_chunk_backward_grad.py": "core",
    # #632 frontier benchmark (phase 1): tiny D=2 CPU value_and_grad finite +
    # path-guard checks. Fast; core so CI required checks run them.
    "test_frontier_probe.py": "core",
    "test_frontier_bench_guard.py": "core",
}


def pytest_collection_modifyitems(items):
    for item in items:
        filename = item.path.name
        if filename in _FILE_MARKERS:
            item.add_marker(getattr(pytest.mark, _FILE_MARKERS[filename]))


# ------------------------------------------------------------------ #
# Cap memory growth across tests by clearing the JAX in-memory       #
# compile cache once peak RSS crosses a threshold.                    #
#                                                                    #
# Why this matters: each AD/CTM test compiles a distinct JAX/XLA     #
# variant (different chi, charges, optimizer config, ...). The JIT   #
# cache retains compiled artifacts in memory across tests. With the  #
# fast-ipeps bucket's ~30+ AD tests this snowballs past 18 GB locally,#
# OOM-killing GH's 7 GB Linux runners (manifesting as "the runner    #
# lost communication with the server" mid-bucket).                   #
#                                                                    #
# ``jax.clear_caches()`` releases the Python-side cache references   #
# so XLA can reuse buffer slots; combined with ``gc.collect()`` it   #
# bounds peak RSS to a single test's working set (~5 GB) instead of  #
# accumulating. The persistent on-disk cache (``~/.cache/jax``, see  #
# ``tenax/__init__.py``) is preserved, so cross-session reuse still  #
# works.                                                             #
#                                                                    #
# Threshold gating: ``ru_maxrss`` is monotonic, so once a bucket's   #
# peak crosses the threshold we clear from then on, but cheap        #
# buckets that never cross stay fast forever.                        #
#                                                                    #
# Threshold tuning (measured locally, 735-test ``-m core`` run):     #
#                                                                    #
#   threshold     core peak       core time   fast-ipeps?            #
#   2 GB          3.0 GB          12m51s      OK (cleared)           #
#   4 GB          4.8 GB           9m15s      OK (cleared)           #
#   6 GB          5.6 GB           2m53s      OK (cleared @ t#44)    #
#   ∞ (no clear)  5.6 GB           2m54s      18 GB → OOM            #
#                                                                    #
# 6 GB is the sweet spot: above ``-m core`` natural peak (5.6 GB) so #
# the cheap bucket stays at baseline speed, but below the 7 GB GH    #
# Linux runner limit so fast-ipeps still engages clearing in time    #
# (its first AD-heavy test crosses 6 GB at iteration ~44, well       #
# before the without-clearing snowball reaches 7 GB OOM).            #
# ------------------------------------------------------------------ #

_RSS_CLEAR_THRESHOLD_MB = 6000


def _peak_rss_mb() -> float:
    """Peak RSS used by this process, in MB.

    ``ru_maxrss`` units differ by platform (BSD convention): macOS reports
    bytes, Linux reports KB.
    """
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    if _peak_rss_mb() < _RSS_CLEAR_THRESHOLD_MB:
        return

    import gc

    jax.clear_caches()
    gc.collect()


# ------------------------------------------------------------------ #
# Symmetry fixtures                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1():
    return U1Symmetry()


@pytest.fixture
def z2():
    return ZnSymmetry(2)


@pytest.fixture
def z3():
    return ZnSymmetry(3)


# ------------------------------------------------------------------ #
# Random key fixture                                                   #
# ------------------------------------------------------------------ #


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng2():
    return jax.random.PRNGKey(99)


# ------------------------------------------------------------------ #
# TensorIndex fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_charges_3(u1):
    """U(1) charges [-1, 0, 1] — typical for spin-1 or bond dim 3."""
    return np.array([-1, 0, 1], dtype=np.int32)


@pytest.fixture
def u1_charges_2(u1):
    """U(1) charges [-1, 1] — typical for spin-1/2."""
    return np.array([-1, 1], dtype=np.int32)


@pytest.fixture
def idx_in_3(u1, u1_charges_3):
    """U(1) IN index with charges [-1, 0, 1], label='left'."""
    return TensorIndex.from_charges(u1, u1_charges_3, FlowDirection.IN, label="left")


@pytest.fixture
def idx_out_3(u1, u1_charges_3):
    """U(1) OUT index with dual charges [1, 0, -1], label='right'.
    This is the proper dual of idx_in_3.
    """
    return TensorIndex.from_charges(
        u1, u1.dual(u1_charges_3), FlowDirection.OUT, label="right"
    )


@pytest.fixture
def u1_index_pair(idx_in_3, idx_out_3):
    """A compatible pair of U(1) indices (IN and its dual OUT)."""
    return idx_in_3, idx_out_3


# ------------------------------------------------------------------ #
# DenseTensor fixtures                                                 #
# ------------------------------------------------------------------ #


@pytest.fixture
def small_dense_matrix(u1, rng):
    """A 3x3 DenseTensor (matrix) with U(1) indices."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3, 3))
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="row"),
        TensorIndex.from_charges(u1, charges, FlowDirection.OUT, label="col"),
    )
    return DenseTensor(data, indices)


@pytest.fixture
def dense_vector(u1, rng):
    """A 3-element DenseTensor (vector) with U(1) index."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    data = jax.random.normal(rng, (3,))
    idx = TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="vec")
    return DenseTensor(data, (idx,))


# ------------------------------------------------------------------ #
# SymmetricTensor fixtures                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def u1_sym_tensor_2leg(u1, rng):
    """2-leg U(1)-symmetric tensor: IN x OUT, charges [-1, 0, 1]."""
    charges = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, charges, FlowDirection.IN, label="in"),
        TensorIndex.from_charges(u1, u1.dual(charges), FlowDirection.OUT, label="out"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_3leg(u1, rng):
    """3-leg U(1)-symmetric tensor: phys x left x right."""
    phys_c = np.array([-1, 1], dtype=np.int32)
    virt_c = np.array([-1, 0, 1], dtype=np.int32)
    indices = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="phys"),
        TensorIndex.from_charges(u1, virt_c, FlowDirection.IN, label="left"),
        TensorIndex.from_charges(u1, u1.dual(virt_c), FlowDirection.OUT, label="right"),
    )
    return SymmetricTensor.random_normal(indices, rng)


@pytest.fixture
def u1_sym_tensor_pair(u1, rng, rng2):
    """A pair of 3-leg U(1)-symmetric tensors that can be contracted on 'bond'.

    Both tensors use the SAME charge array for the shared bond leg with
    opposite flow directions (OUT for A, IN for B). Same charge array means
    position i in A's bond and position i in B's bond store the same charge
    value, so dense einsum contraction works correctly (no ordering mismatch).
    """
    phys_c = np.array([-1, 1], dtype=np.int32)
    bond_c = np.array(
        [-1, 0, 1], dtype=np.int32
    )  # same charges for both ends of shared bond

    indices_A = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="p0"),
        TensorIndex.from_charges(u1, bond_c, FlowDirection.IN, label="bond_left"),
        TensorIndex.from_charges(
            u1, bond_c, FlowDirection.OUT, label="bond"
        ),  # OUT end of shared bond
    )
    indices_B = (
        TensorIndex.from_charges(u1, phys_c, FlowDirection.IN, label="p1"),
        TensorIndex.from_charges(
            u1, bond_c, FlowDirection.IN, label="bond"
        ),  # IN end of shared bond (same charges)
        TensorIndex.from_charges(u1, bond_c, FlowDirection.OUT, label="bond_right"),
    )
    A = SymmetricTensor.random_normal(indices_A, rng)
    B = SymmetricTensor.random_normal(indices_B, rng2)
    return A, B
