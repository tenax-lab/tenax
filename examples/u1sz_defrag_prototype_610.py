"""THROWAWAY #610 C-lever prototype: U(1)-Sz CTM env sector-dropping.

This is a branch-local, throwaway research prototype for spike #610. It is
NEVER imported by ``src/`` and makes NO committed change to library code. It
exists only so the spike's Gate-B / Gate-C measurements can re-profile the CTM
backward graph and end-to-end run under a "sector-dropped" environment.

What "sector dropping" means
----------------------------
The active CTM path for U(1)-Sz SymmetricTensors is the *paired-moves* path
(``tenax.algorithms._ctm_tensor_paired_moves``). Each paired move derives a
per-sector chi allocation from ``base_charges`` produced by
``_get_base_charges(a)`` — the D^2 (u2) leg charges of the double-layer site.
At D=3 these are ``{-2,-1,0,1,2}``. The C-lever's goal is to make the
truncation keep ONLY the small-|Sz| chi-bond sectors (default ``{-1,0,1}``),
dropping |Sz| > 1.

Why a single ``_get_base_charges`` filter is NOT enough
-------------------------------------------------------
``base_charges`` flows into both the eager forward truncation
(``_svd_projector_symmetric`` in ``_ctm_projector.py``) and the traced
backward truncation (``_truncated_svd_symmetric_traced`` in ``linalg.py``).
Filtering ``base_charges`` to ``{-1,0,1}`` correctly restricts the *initial*
per-sector allocation (``target_count`` only requests ``{-1,0,1}``). BUT both
truncation routines contain a **greedy backfill**: when the kept sectors do not
saturate ``chi`` (at D=3 χ=12 the ``{-1,0,1}`` sectors supply only ~10 of the
12 singular values), the remaining budget is filled from the *global*
singular-value list, which re-introduces the ``±2`` sectors. Measured: the
forward projector ``chi_new`` comes out ``{-2:1,-1:3,0:4,1:3,2:1}`` — the
backfill re-adds one ``±2`` each. So the env block counts never drop.

The fix: drop the sectors AND suppress the backfill
---------------------------------------------------
This prototype monkeypatches (restoring all on exit):

  1. ``_get_base_charges`` — filter to ``keep`` (sets the truncation intent).
  2. ``_svd_projector_symmetric`` (forward, eager block-sparse) — wrap the
     original and drop any ``chi_new`` columns whose charge is outside ``keep``.
     This post-filters the genuine projector output, removing whole orthonormal
     columns for the dropped charges so each projector stays an isometry on the
     retained sectors and the resulting env is a valid (narrower)
     charge-conserving environment.
  3. (OPT-IN, default OFF) ``_truncated_svd_symmetric_traced`` (the traced
     block-sparse SVD). See ``sector_dropping_truncation``'s ``patch_traced_svd``
     argument for why this is off by default.

Two-path obstruction (key #610 finding — see ``patch_traced_svd`` docstring)
----------------------------------------------------------------------------
The single-site forward (``ctm_tensor`` -> paired-moves) and the AD backward
that the Gate-B/Gate-C probes profile (``_make_jit_ctm_step`` ->
``_ctm_tensor_sweep_multisite`` -> 2x2 plaquette projector) are DIFFERENT
projector implementations. (2) faithfully drops the **forward** env (the
faithfulness guard passes; corners 5->3, edges 19->9 at D=3 χ=12, matching the
Gate-A prediction). But the backward 2x2 plaquette path truncates through
``_truncated_svd_symmetric_traced`` and feeds the chi bond into
``_build_enlarged_corner``; physically dropping that bond there breaks the
enlarged-corner contraction (the un-dropped neighbour legs no longer match), so
the single-sweep VJP cannot even be traced. The ``_get_base_charges`` filter
alone leaves the backward bond un-dropped (its greedy backfill re-adds ``±2``),
so the backward jaxpr is baseline-shaped. Net: a clean forward sector-drop does
NOT carry into the traced backward via post-hoc bond filtering. The default
config keeps Gate-B trace-able (backward not yet dropped) and leaves the traced
drop as an opt-in hook for the Gate-B task to resolve.

Usage
-----
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, ...)
"""

from __future__ import annotations

import contextlib

import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_projector as _proj
import tenax.algorithms._ctm_tensor_paired_moves as _pm
import tenax.linalg as _linalg
from tenax.core.index import TensorIndex
from tenax.core.tensor import SymmetricTensor

# Pristine originals captured once at import so repeated / nested use always
# restores the genuine implementations.
_ORIG_GET_BASE_CHARGES = _pm._get_base_charges
_ORIG_SVD_PROJECTOR = _proj._svd_projector_symmetric
_ORIG_TRACED_SVD = _linalg._truncated_svd_symmetric_traced


def _keep_columns_of_chi_new(P: SymmetricTensor, keep: set[int]) -> SymmetricTensor:
    """Drop ``chi_new``-axis columns whose charge is outside ``keep``.

    ``P`` is a projector with indices ``(fused, chi_new)`` and blocks keyed
    ``(fq, fq)``. Removing the whole ``(q, q)`` block for ``q not in keep``
    deletes those orthonormal columns; the result stays an isometry on the
    retained sectors. Rebuilds ``chi_new`` without the dropped charges.
    """
    if not isinstance(P, SymmetricTensor):
        return P
    chi_pos = P.labels().index("chi_new")
    fused_idx = P.indices[1 - chi_pos]  # the other (fused) index
    chi_idx = P.indices[chi_pos]
    chi_charges = np.asarray(chi_idx.charges, dtype=np.int32)
    if all(int(q) in keep for q in chi_charges):
        return P  # nothing to drop

    new_blocks: dict[tuple[int, ...], jnp.ndarray] = {}
    new_chi_charges: list[int] = []
    for key, block in P.blocks.items():
        # chi_new charge is the entry at the chi axis position of the key
        q_chi = int(key[chi_pos])
        if q_chi not in keep:
            continue
        new_blocks[key] = block
        new_chi_charges.extend([q_chi] * block.shape[chi_pos])

    if not new_chi_charges:
        # Degenerate: keep set removed everything — fall back to the original
        # so we never emit an empty bond.
        return P

    new_chi_idx = TensorIndex.from_charges(
        chi_idx.symmetry,
        np.array(new_chi_charges, dtype=np.int32),
        chi_idx.flow,
        label="chi_new",
    )
    new_indices = (
        (fused_idx, new_chi_idx) if chi_pos == 1 else (new_chi_idx, fused_idx)
    )
    return SymmetricTensor._from_blocks_unchecked(new_blocks, new_indices)


def _drop_bond_sectors_usv(U_T, s, Vh_T, keep: set[int]):
    """Drop new-bond sectors outside ``keep`` from a traced SVD ``(U, s, Vh)``.

    ``U_T`` has the new bond as its LAST index; ``Vh_T`` as its FIRST. ``s`` is
    a flat 1-D array in sector-ascending order matching the bond. Removing whole
    bond sectors keeps U/Vh sub-isometric on the retained charges.
    """
    bond_pos_U = len(U_T.indices) - 1
    bond_idx = U_T.indices[bond_pos_U]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)
    if all(int(q) in keep for q in bond_charges):
        return U_T, s, Vh_T

    keep_mask = np.array([int(q) in keep for q in bond_charges], dtype=bool)
    if not keep_mask.any():
        return U_T, s, Vh_T  # never emit an empty bond

    new_bond_charges = bond_charges[keep_mask]
    s_new = s[jnp.asarray(np.nonzero(keep_mask)[0])]

    label = bond_idx.label
    sym = bond_idx.symmetry
    U_bond_in = U_T.indices  # for index reconstruction below
    new_U_bond = TensorIndex.from_charges(
        sym, new_bond_charges, bond_idx.flow, label=label
    )
    vh_bond_idx = Vh_T.indices[0]
    new_Vh_bond = TensorIndex.from_charges(
        sym, new_bond_charges, vh_bond_idx.flow, label=vh_bond_idx.label
    )

    new_U_blocks = {
        key: blk for key, blk in U_T.blocks.items()
        if int(key[bond_pos_U]) in keep
    }
    new_Vh_blocks = {
        key: blk for key, blk in Vh_T.blocks.items()
        if int(key[0]) in keep
    }
    new_U_indices = U_bond_in[:bond_pos_U] + (new_U_bond,)
    new_Vh_indices = (new_Vh_bond,) + Vh_T.indices[1:]
    U_out = SymmetricTensor._from_blocks_unchecked(new_U_blocks, new_U_indices)
    Vh_out = SymmetricTensor._from_blocks_unchecked(new_Vh_blocks, new_Vh_indices)
    return U_out, s_new, Vh_out


def _patched_traced_svd(
    tensor,
    left_labels,
    right_labels,
    max_singular_values,
    new_bond_label,
    normalize,
    base_charges=None,
    *,
    _keep=None,
):
    """Traced (backward/AD) SVD with non-``keep`` new-bond sectors dropped."""
    U_T, s_trunc, Vh_T, s_full = _ORIG_TRACED_SVD(
        tensor,
        left_labels,
        right_labels,
        max_singular_values,
        new_bond_label,
        normalize,
        base_charges=base_charges,
    )
    U_T, s_trunc, Vh_T = _drop_bond_sectors_usv(U_T, s_trunc, Vh_T, _keep)
    return U_T, s_trunc, Vh_T, s_trunc


@contextlib.contextmanager
def sector_dropping_truncation(keep=frozenset({-1, 0, 1}), *, patch_traced_svd=False):
    """Monkeypatch the CTM truncation to drop chi-bond sectors outside ``keep``.

    Parameters
    ----------
    keep:
        Iterable of integer charges to retain on the environment (chi) bonds.
        Any chi-bond sector whose charge is not in ``keep`` is dropped from the
        eager forward (block-sparse paired-moves) projector.
    patch_traced_svd:
        Whether to ALSO patch the traced block-sparse SVD
        (``_truncated_svd_symmetric_traced``) to drop non-``keep`` new-bond
        sectors. Default ``False``.

        IMPORTANT — read before flipping this on. The two CTM entry points use
        different projector machinery:

          * ``ctm_tensor`` (single-site forward, used by the faithfulness
            guard) -> ``_ctm_tensor_sweep_paired`` -> ``_svd_projector_symmetric``
            (block-sparse). The default patches (``_get_base_charges`` +
            ``_svd_projector_symmetric``) drop sectors faithfully here.
          * ``_make_jit_ctm_step`` / ``ctm_multisite`` (the **AD backward** path
            that ``_optimize_gs_ad_tensor`` and the Gate-B/Gate-C probes
            actually use) -> ``_ctm_tensor_sweep_multisite`` -> the 2x2
            plaquette projector (``_ctm_tensor_projector_2x2``), which truncates
            via ``tensor_svd`` -> ``_truncated_svd_symmetric_traced``.

        Setting ``patch_traced_svd=True`` makes the traced SVD physically drop
        the chi bond, but in the 2x2 plaquette path that bond feeds
        ``_build_enlarged_corner``, whose other (un-dropped) env legs then
        mismatch the narrowed bond -> the single-sweep VJP fails to *trace*
        (``dot_general``/einsum size error). So the traced drop, as a post-hoc
        bond filter, is NOT consistent with the 2x2 enlarged-corner contraction.
        Left as an opt-in hook for the Gate-B task to investigate; the default
        keeps the prototype trace-able so Gate-B can at least build the
        backward jaxpr (baseline-shaped, i.e. backward NOT yet dropped).

    Guards
    ------
    * ``_get_base_charges`` returning ``None`` (dense/trivial-charge) is a
      no-op pass-through.
    * If filtering would remove *every* charge from ``base_charges`` or from a
      produced bond, the original (unfiltered) value is kept so the env can
      never collapse to an empty bond.
    """
    keep = {int(q) for q in keep}

    def _patched_get_base_charges(a):
        bc = _ORIG_GET_BASE_CHARGES(a)
        if bc is None:
            return None
        filt = np.array([q for q in bc if int(q) in keep], dtype=np.int32)
        return filt if filt.size else bc

    def _svd_proj(C1g, C4g, chi, base_charges=None, **kw):
        # Forward (eager) projector with non-keep chi_new columns dropped.
        P1, P2, eps = _ORIG_SVD_PROJECTOR(
            C1g, C4g, chi, base_charges=base_charges, **kw
        )
        P1 = _keep_columns_of_chi_new(P1, keep)
        P2 = _keep_columns_of_chi_new(P2, keep)
        return P1, P2, eps

    def _traced(
        tensor,
        left_labels,
        right_labels,
        max_singular_values,
        new_bond_label,
        normalize,
        base_charges=None,
    ):
        return _patched_traced_svd(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            new_bond_label,
            normalize,
            base_charges=base_charges,
            _keep=keep,
        )

    _pm._get_base_charges = _patched_get_base_charges
    _proj._svd_projector_symmetric = _svd_proj
    if patch_traced_svd:
        _linalg._truncated_svd_symmetric_traced = _traced
    try:
        yield
    finally:
        _pm._get_base_charges = _ORIG_GET_BASE_CHARGES
        _proj._svd_projector_symmetric = _ORIG_SVD_PROJECTOR
        _linalg._truncated_svd_symmetric_traced = _ORIG_TRACED_SVD
