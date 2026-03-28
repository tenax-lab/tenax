"""Benchmark: Cython BLAS vs opt_einsum fallback for DMRG matvec."""

import time

import numpy as np
import opt_einsum
import pytest

from tenax.contraction import CYTHON_BLAS_AVAILABLE


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
