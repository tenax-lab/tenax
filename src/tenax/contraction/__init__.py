"""Tensor contraction engine with label-based API."""

import os

from tenax.contraction.contractor import (
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)

CYTHON_BLAS_AVAILABLE = False
if os.environ.get("TENAX_DISABLE_CYTHON_BLAS", "0") != "1":
    try:
        from tenax.contraction._cython_blas import execute_block_plan  # noqa: F401

        CYTHON_BLAS_AVAILABLE = True
    except ImportError:
        pass

CYTHON_BLAS_V2_AVAILABLE = False
if os.environ.get("TENAX_DISABLE_CYTHON_BLAS", "0") != "1":
    try:
        from tenax.contraction._cython_blas import execute_blas_kernel_v2  # noqa: F401

        CYTHON_BLAS_V2_AVAILABLE = True
    except ImportError:
        pass

__all__ = [
    "contract",
    "contract_with_subscripts",
    "truncated_svd",
    "qr_decompose",
    "CYTHON_BLAS_AVAILABLE",
    "CYTHON_BLAS_V2_AVAILABLE",
]
