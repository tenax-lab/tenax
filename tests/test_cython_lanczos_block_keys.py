"""The Cython Lanczos block-array ops must agree with the pure-Python ones.

``dmrg.py`` picks between them at import time (``_USE_CYTHON_REORTH`` /
``_USE_CYTHON_SUB``, gated on ``TENAX_DISABLE_CYTHON_BLAS``), so any semantic
difference makes the same DMRG input converge differently depending on whether
the extension loaded -- and the extension is the default.

That is what #829 was: the Cython side skipped keys present in ``q`` but absent
from ``w`` while ``ba_sub_scaled`` inserts them, so reorthogonalisation left
``w`` non-orthogonal to a basis vector in every sector ``w`` happened to lack,
and never created the sector.

These are written as **differential** tests rather than against hardcoded
vectors: the defect was a divergence between two implementations, and only a
differential check catches the next one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenax.core._block_array import BlockArray, ba_inner, ba_sub_scaled

cython_blas = pytest.importorskip(
    "tenax.contraction._cython_blas",
    reason="Cython BLAS extension not built",
)


def _q_two_sectors(dtype=np.float64):
    """Unit basis vector spanning sectors (0,) and (1,)."""
    q = {
        (0,): np.array([1.0, 0.0], dtype=dtype),
        (1,): np.array([1.0, 0.0], dtype=dtype),
    }
    nrm = np.sqrt(sum(float(np.vdot(v, v).real) for v in q.values()))
    return {k: v / nrm for k, v in q.items()}


def _w_one_sector(dtype=np.float64):
    """Running vector missing sector (1,) -- the case that used to be skipped."""
    return {(0,): np.array([1.0, 0.0], dtype=dtype)}


def _python_reorth(q_blocks, w_blocks):
    """What ``dmrg.py`` does when the Cython path is disabled."""
    q = BlockArray(blocks=dict(q_blocks), indices=None)
    w = BlockArray(blocks=dict(w_blocks), indices=None)
    return ba_sub_scaled(w, q, ba_inner(q, w)).blocks


def _inner(a_blocks, b_blocks):
    return sum(
        complex(np.vdot(a_blocks[k], b_blocks[k])) for k in a_blocks if k in b_blocks
    )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.complex64])
def test_reorth_matches_python_when_w_lacks_a_sector(dtype):
    """The #829 case, across every dtype branch of ``_lanczos_reorth_impl``.

    Each dtype takes a different Phase-2 path in the Cython source (BLAS axpy
    for float64, hand-rolled loops for the complex ones), so a fix applied to
    one of them is not a fix.
    """
    q = _q_two_sectors(dtype)

    w_cy = _w_one_sector(dtype)
    cython_blas.cython_lanczos_reorth([q], w_cy)
    w_py = _python_reorth(q, _w_one_sector(dtype))

    assert sorted(w_cy) == sorted(w_py), (
        f"key sets diverge: cython {sorted(w_cy)} vs python {sorted(w_py)}"
    )
    for k in w_py:
        np.testing.assert_allclose(
            w_cy[k], w_py[k], rtol=1e-5, atol=1e-6, err_msg=f"sector {k}"
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.complex64])
def test_reorth_actually_orthogonalises_the_missing_sector(dtype):
    """The property the divergence broke, stated directly.

    Without this a future implementation could match Python by both being
    wrong.  ``<q|w>`` must be ~0 afterwards; before the fix it was 0.354.
    """
    q = _q_two_sectors(dtype)
    w = _w_one_sector(dtype)
    cython_blas.cython_lanczos_reorth([q], w)

    tol = 1e-6 if dtype == np.complex64 else 1e-12
    assert abs(_inner(q, w)) < tol, f"w is not orthogonal to q: <q|w> = {_inner(q, w)}"
    assert (1,) in w, "the sector w lacked was never created"


def test_sub_scaled_inplace_matches_python_on_a_missing_key():
    """``_ba_sub_scaled_impl`` is the other half of the Lanczos hot loop.

    ``dmrg.py`` calls it directly for the ``beta`` subtraction, so it needs the
    same union semantics as the reorthogonalisation above.
    """
    q = _q_two_sectors()
    w_cy = _w_one_sector()
    scalar = 0.25

    cython_blas.cython_ba_sub_scaled_inplace(w_cy, q, scalar)
    w_py = ba_sub_scaled(
        BlockArray(blocks=_w_one_sector(), indices=None),
        BlockArray(blocks=dict(q), indices=None),
        scalar,
    ).blocks

    assert sorted(w_cy) == sorted(w_py), (sorted(w_cy), sorted(w_py))
    for k in w_py:
        np.testing.assert_allclose(w_cy[k], w_py[k], rtol=1e-12, atol=1e-14)


def test_shared_key_case_is_unchanged():
    """The fix must not perturb the path that was already correct.

    Union and shared-keys semantics coincide when the key sets match, which is
    the common case; this pins that the change is inert there.
    """
    q = _q_two_sectors()
    w = {
        (0,): np.array([0.3, 0.4]),
        (1,): np.array([-0.2, 0.7]),
    }

    w_cy = {k: v.copy() for k, v in w.items()}
    cython_blas.cython_lanczos_reorth([q], w_cy)
    w_py = _python_reorth(q, w)

    assert sorted(w_cy) == sorted(w_py)
    for k in w_py:
        np.testing.assert_allclose(w_cy[k], w_py[k], rtol=1e-12, atol=1e-14)
    assert abs(_inner(q, w_cy)) < 1e-12
