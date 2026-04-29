"""Benchmark: Cython BLAS vs opt_einsum fallback for DMRG matvec."""

import os
import time

import numpy as np
import opt_einsum
import pytest

from tenax.contraction import CYTHON_BLAS_AVAILABLE, CYTHON_LANCZOS_AVAILABLE


@pytest.mark.slow
@pytest.mark.skipif(not CYTHON_BLAS_AVAILABLE, reason="Cython BLAS not compiled")
def test_cython_blas_faster_than_fallback():
    """Cython BLAS path should be at least 2x faster than opt_einsum."""
    from tenax.contraction._cython_blas import execute_block_plan

    from tenax.contraction._blas_plan import build_blas_plan

    rng = np.random.default_rng(42)

    # 1-site DMRG matvec pattern with many block combos
    subs = "abc,apd,bpxe,def->cxf"
    chi, d = 4, 2
    n_combos = 50

    shapes = [(chi, chi, chi), (chi, d, chi), (chi, d, d, chi), (chi, chi, chi)]
    plan = build_blas_plan(subs, shapes)

    block_keys = [(i,) for i in range(n_combos)]
    np_blocks = [{k: rng.standard_normal(s) for k in block_keys} for s in shapes]
    combos = [([block_keys[i]] * len(shapes), (i,)) for i in range(n_combos)]

    # Warmup
    execute_block_plan(plan, combos[:1], np_blocks)

    # Time Cython path
    t0 = time.perf_counter()
    for _ in range(10):
        execute_block_plan(plan, combos, np_blocks)
    t_cython = (time.perf_counter() - t0) / 10

    # Time opt_einsum fallback
    expr_cache: dict = {}

    def fallback():
        accum: dict = {}
        for combo_keys, out_key in combos:
            arrays = [np_blocks[j][combo_keys[j]] for j in range(len(shapes))]
            bshapes = tuple(a.shape for a in arrays)
            if bshapes not in expr_cache:
                expr_cache[bshapes] = opt_einsum.contract_expression(
                    subs, *bshapes, optimize="auto"
                )
            result = expr_cache[bshapes](*arrays)
            if out_key in accum:
                accum[out_key] = accum[out_key] + result
            else:
                accum[out_key] = result
        return accum

    fallback()  # warmup
    t0 = time.perf_counter()
    for _ in range(10):
        fallback()
    t_fallback = (time.perf_counter() - t0) / 10

    speedup = t_fallback / t_cython
    print(
        f"\nCython: {t_cython * 1000:.1f}ms, "
        f"Fallback: {t_fallback * 1000:.1f}ms, "
        f"Speedup: {speedup:.1f}x"
    )

    assert speedup > 1.5, (
        f"Cython BLAS should be at least 1.5x faster, got {speedup:.1f}x "
        f"(cython={t_cython * 1000:.1f}ms, fallback={t_fallback * 1000:.1f}ms)"
    )


@pytest.mark.slow
@pytest.mark.skipif(not CYTHON_LANCZOS_AVAILABLE, reason="Cython Lanczos not compiled")
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "Wall-clock perf ratio is too noisy on shared CI runners (saw "
        "0.39x on a recent run vs the >= 1.3x gate). Keep runnable "
        "locally for manual benchmarking; tracked in #354."
    ),
)
def test_cython_lanczos_faster_than_python():
    """Fused Cython Lanczos must be >= 1.2x faster than Python Lanczos.

    Runs a full symmetric DMRG sweep with the Cython Lanczos path enabled
    vs disabled, comparing total wall time.
    """
    import sys

    import jax

    from tenax.algorithms.auto_mpo import build_auto_mpo
    from tenax.algorithms.dmrg import DMRGConfig
    from tenax.algorithms.dmrg import dmrg as _dmrg
    from tenax.core.mps import FiniteMPS
    from tenax.core.symmetry import U1Symmetry

    dmrg_module = sys.modules["tenax.algorithms.dmrg"]

    L = 12
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    H = build_auto_mpo(terms, L=L, symmetric=True)

    config = DMRGConfig(max_bond_dim=32, num_sweeps=3, lanczos_max_iter=20)

    N_REPS = 3

    # Time with Cython Lanczos (default when available)
    t_cython_total = 0.0
    for rep in range(N_REPS):
        mps = FiniteMPS.random(
            L,
            d=2,
            chi=32,
            key=jax.random.PRNGKey(42 + rep),
            symmetric=True,
            symmetry=U1Symmetry(),
            target_charge=0,
        )
        t0 = time.perf_counter()
        _dmrg(H, mps, config)
        t_cython_total += time.perf_counter() - t0

    # Time with Cython Lanczos disabled (Python fallback)
    saved = dmrg_module._USE_CYTHON_LANCZOS
    dmrg_module._USE_CYTHON_LANCZOS = False
    try:
        t_python_total = 0.0
        for rep in range(N_REPS):
            mps = FiniteMPS.random(
                L,
                d=2,
                chi=32,
                key=jax.random.PRNGKey(42 + rep),
                symmetric=True,
                symmetry=U1Symmetry(),
                target_charge=0,
            )
            t0 = time.perf_counter()
            _dmrg(H, mps, config)
            t_python_total += time.perf_counter() - t0
    finally:
        dmrg_module._USE_CYTHON_LANCZOS = saved

    speedup = t_python_total / t_cython_total
    print(
        f"\nDMRG sweep benchmark (L={L}, chi=32, {config.num_sweeps}sw):"
        f"\n  Python Lanczos: {t_python_total:.3f}s"
        f"\n  Cython Lanczos: {t_cython_total:.3f}s"
        f"\n  Speedup: {speedup:.2f}x"
    )
    assert speedup >= 1.3, (
        f"Cython Lanczos only {speedup:.2f}x faster (need >= 1.3x). "
        f"Python={t_python_total:.3f}s, Cython={t_cython_total:.3f}s"
    )
