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

What sector dropping costs: RDM positivity (#853)
--------------------------------------------------
**Energies read off a dropped environment are not bounded by the Hamiltonian.**
The prototype deletes whole *charge* sectors from the environment's chi bonds,
and each projector stays an isometry on the sectors it keeps -- which is why the
corner stays rank-4 and the environment stays well-formed by every structural
check. But a CTM reduced density matrix is positive because the environment
approximates ``<psi|...|psi>`` with the ket and bra layers truncated
*consistently*, and a hard projection selected by charge rather than by weight is
not that kind of truncation.

Measured at D=3, chi=12, ``keep={-1,0,1}``, 50 sweeps::

    dropping   min eig / spectral radius   E_h + E_v
    off        +6.65e-02                   +0.080313
    on         -7.98e-01                   +0.759466

``E_h + E_v`` for ``H = S_i . S_j`` lies in ``[-1.5, +0.5]`` for *any* state, so
``+0.759`` is not an inaccurate energy -- it is not an energy. The trace cannot
see this: ``rdm_h``'s trace is exactly 1 in both rows, because negative
eigenvalues cancel against positive ones inside the sum (#854).

So positivity is something the C-lever trades away, not a defect that leaked in,
and the plain symmetric CTM keeps it on the same input at every sweep budget
(pinned by ``test_the_plain_symmetric_ctm_keeps_the_rdm_positive``). Gate-B and
Gate-C should treat block counts and timings from this prototype as meaningful
and any *energy* as not.

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
import tenax.algorithms._ctm_tensor_convergence as _conv
import tenax.algorithms._ctm_tensor_paired_moves as _pm
import tenax.linalg as _linalg
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor

# Pristine originals captured once at import so repeated / nested use always
# restores the genuine implementations.
_ORIG_GET_BASE_CHARGES = _pm._get_base_charges
# The AD backward path (_make_jit_ctm_step -> _ctm_tensor_sweep_multisite, used by
# both the Gate-B probe and production _ctm_energy_ad) reads base_charges from a
# SECOND, distinct function in _ctm_tensor_convergence — NOT the paired-moves one.
# Both must be filtered for the drop to reach the traced backward (see Step 0/#610).
_ORIG_CONV_GET_BASE_CHARGES = _conv._get_base_charges
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


def _make_keep_filtered_traced_svd(keep: set[int]):
    """Build a drop-in for ``_truncated_svd_symmetric_traced`` that allocates
    ZERO singular values to any new-bond sector ``q`` with ``int(q) not in keep``.

    This is a faithful COPY of ``tenax.linalg._truncated_svd_symmetric_traced``
    (``src/tenax/linalg.py`` lines 583-864 at the time of writing) with EXACTLY
    two minimal edits, nothing else changed:

      1. In the Phase-1 ``base_charges is not None`` allocation and the defensive
         ``else`` fallback: a sector ``q`` outside ``keep`` is forced to
         ``k_per_sector[q] = 0`` (dropped — never requests SVs for it).
      2. In the Phase-2 greedy χ-backfill loop: ``if int(q) not in keep:
         continue`` so the leftover χ budget is NEVER refilled from a
         non-``keep`` sector (the bug that re-added ±2 at D=3 χ=12).

    Consequence: ``chi_new`` is the SUM of kept-sector allocations and may be
    < ``max_singular_values`` (e.g. ~10 at D=3 χ=12). That smaller bond is the
    HONEST sector-dropped bond dim, produced BY the truncation itself, so the
    downstream enlarged-corner contraction accepts it (unlike the post-hoc
    column drop in ``_drop_bond_sectors_usv``, which broke the trace).

    All library internals are referenced through the ``_linalg`` module so the
    copy stays pinned to the same helpers (`_group_blocks_by_bond_charge`,
    `_grouped_decomp_by_shape`, `_batch_blocksparse_enabled`, `_koszul_sign`).
    """
    keep = {int(q) for q in keep}

    def _keep_filtered_traced(
        tensor,
        left_labels,
        right_labels,
        max_singular_values,
        new_bond_label,
        normalize,
        base_charges=None,
    ):
        from tenax.algorithms._ad_primitives import truncated_svd_ad
        from tenax.algorithms._ctm_utils import _derive_charges

        all_labels = tensor.labels()
        label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
        left_axes = [label_to_axis[lbl] for lbl in left_labels]
        right_axes = [label_to_axis[lbl] for lbl in right_labels]
        left_indices = tuple(tensor.indices[i] for i in left_axes)
        right_indices = tuple(tensor.indices[i] for i in right_axes)

        grouped = _linalg._group_blocks_by_bond_charge(tensor, left_axes, right_axes)

        sym = tensor.indices[0].symmetry
        is_fermionic = sym.is_fermionic
        decomp_perm = tuple(left_axes + right_axes)

        sector_results: dict[int, tuple] = {}

        for q, entries in grouped.items():
            left_subkeys_seen: dict = {}
            right_subkeys_seen: dict = {}
            for lk, rk, _ in entries:
                if lk not in left_subkeys_seen:
                    left_subkeys_seen[lk] = len(left_subkeys_seen)
                if rk not in right_subkeys_seen:
                    right_subkeys_seen[rk] = len(right_subkeys_seen)

            left_subkeys = list(left_subkeys_seen.keys())
            right_subkeys = list(right_subkeys_seen.keys())

            left_row_sizes: list[int] = []
            for lk in left_subkeys:
                size = 1
                for leg_pos, charge_val in zip(left_axes, lk):
                    idx = tensor.indices[leg_pos]
                    size *= idx.multiplicity(charge_val)
                left_row_sizes.append(size)

            right_col_sizes: list[int] = []
            for rk in right_subkeys:
                size = 1
                for leg_pos, charge_val in zip(right_axes, rk):
                    idx = tensor.indices[leg_pos]
                    size *= idx.multiplicity(charge_val)
                right_col_sizes.append(size)

            total_rows = sum(left_row_sizes)
            total_cols = sum(right_col_sizes)
            if total_rows == 0 or total_cols == 0:
                continue

            matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
            for lk, rk, block in entries:
                li = left_subkeys_seen[lk]
                ri = right_subkeys_seen[rk]
                row_start = sum(left_row_sizes[:li])
                col_start = sum(right_col_sizes[:ri])
                flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
                if is_fermionic:
                    full_key = [0] * len(tensor.indices)
                    for ax, ch in zip(left_axes, lk):
                        full_key[ax] = ch
                    for ax, ch in zip(right_axes, rk):
                        full_key[ax] = ch
                    parities = tuple(
                        int(sym.parity(np.array([full_key[i]]))[0])
                        for i in range(len(full_key))
                    )
                    ksign = _linalg._koszul_sign(parities, decomp_perm)
                    if ksign < 0:
                        flat_block = -flat_block
                matrix = matrix.at[
                    row_start : row_start + left_row_sizes[li],
                    col_start : col_start + right_col_sizes[ri],
                ].set(flat_block)

            available_q = min(total_rows, total_cols)
            sector_results[q] = (
                matrix,
                left_subkeys,
                right_subkeys,
                left_row_sizes,
                right_col_sizes,
                available_q,
            )

        # --- Static per-sector keep allocation ---
        if max_singular_values is None:
            k_per_sector: dict[int, int] = {
                q: r[5] for q, r in sector_results.items()
            }
        elif base_charges is not None:
            target_charges = _derive_charges(base_charges, max_singular_values)
            target_count: dict[int, int] = {}
            for tq in target_charges:
                target_count[int(tq)] = target_count.get(int(tq), 0) + 1
            # EDIT 1: a sector q outside `keep` is dropped (k_per_sector[q] = 0).
            k_per_sector = {
                q: (
                    0
                    if int(q) not in keep
                    else min(target_count.get(q, 0), r[5])
                )
                for q, r in sector_results.items()
            }
            remaining = max_singular_values - sum(k_per_sector.values())
            if remaining > 0:
                for q in sorted(
                    sector_results.keys(),
                    key=lambda qq: (
                        -(sector_results[qq][5] - k_per_sector.get(qq, 0)),
                        qq,
                    ),
                ):
                    if remaining <= 0:
                        break
                    # EDIT 2: never backfill from a non-keep sector (the bug that
                    # re-added the ±2 sectors at D=3 χ=12).
                    if int(q) not in keep:
                        continue
                    capacity_left = sector_results[q][5] - k_per_sector.get(q, 0)
                    take = min(remaining, capacity_left)
                    if take > 0:
                        k_per_sector[q] = k_per_sector.get(q, 0) + take
                        remaining -= take
        else:
            total_avail = sum(r[5] for r in sector_results.values()) or 1
            # EDIT 1 (fallback): sectors outside `keep` get 0; others proportional.
            k_per_sector = {
                q: (
                    0
                    if int(q) not in keep
                    else max(1, round(max_singular_values * r[5] / total_avail))
                )
                for q, r in sector_results.items()
            }
            excess = sum(k_per_sector.values()) - max_singular_values
            for q in sorted(k_per_sector.keys()):
                if excess <= 0:
                    break
                take = min(excess, k_per_sector[q] - 1)
                if take > 0:
                    k_per_sector[q] -= take
                    excess -= take
            if excess > 0:
                for q in sorted(
                    k_per_sector.keys(),
                    key=lambda qq: (sector_results[qq][5], qq),
                ):
                    if excess <= 0:
                        break
                    take = min(excess, k_per_sector[q])
                    k_per_sector[q] -= take
                    excess -= take

        # Floor at >=1 total (mirrors eager n_keep = max(1, n_keep)).  Restricted
        # to keep sectors so the floor can never re-introduce a dropped sector.
        total_keep = sum(k_per_sector.values())
        if total_keep == 0 and sector_results:
            keep_candidates = [
                q for q in sector_results.keys() if int(q) in keep
            ] or list(sector_results.keys())
            best_q = max(
                keep_candidates,
                key=lambda q: (sector_results[q][5], -q),
            )
            k_per_sector[best_q] = 1

        # --- Per-sector AD-primitive SVD ---
        sector_svd: dict = {}
        _ad_mats_by_q: dict = {}
        for q, (matrix, _, _, _, _, _) in sector_results.items():
            k_q = k_per_sector.get(q, 0)
            if k_q <= 0:
                continue
            _ad_mats_by_q[q] = matrix

        if _linalg._batch_blocksparse_enabled():
            _ad_svd_by_q = _linalg._grouped_decomp_by_shape(
                _ad_mats_by_q,
                truncated_svd_ad,
                group_extra_key=lambda q: int(k_per_sector[q]),
            )
            for q, res in _ad_svd_by_q.items():
                sector_svd[q] = res
        else:
            for q, matrix in _ad_mats_by_q.items():
                U_q, s_q, Vh_q = truncated_svd_ad(matrix, int(k_per_sector[q]))
                sector_svd[q] = (U_q, s_q, Vh_q)

        # --- Concatenate output in canonical sector-ascending order ---
        ordered_qs = sorted(sector_svd.keys())
        bond_charges = np.repeat(
            np.array(ordered_qs, dtype=np.int32),
            np.array(
                [sector_svd[q][1].shape[0] for q in ordered_qs], dtype=np.int32
            ),
        )
        s_final = jnp.concatenate([sector_svd[q][1] for q in ordered_qs])

        if normalize and s_final.shape[0] > 0:
            s_final = s_final / jnp.sum(s_final)

        bond_index_out = TensorIndex.from_charges(
            sym, bond_charges, FlowDirection.OUT, label=new_bond_label
        )
        bond_index_in = TensorIndex.from_charges(
            sym, bond_charges, FlowDirection.IN, label=new_bond_label
        )

        U_blocks: dict = {}
        Vh_blocks: dict = {}
        for q in ordered_qs:
            (
                _matrix,
                left_subkeys,
                right_subkeys,
                left_row_sizes,
                right_col_sizes,
                _,
            ) = sector_results[q]
            U_q, _, Vh_q = sector_svd[q]
            row_offset = 0
            for li, lk in enumerate(left_subkeys):
                n_rows = left_row_sizes[li]
                block_rows = U_q[row_offset : row_offset + n_rows, :]
                row_offset += n_rows
                shape = tuple(
                    tensor.indices[ax].multiplicity(ch)
                    for ax, ch in zip(left_axes, lk)
                ) + (U_q.shape[1],)
                U_blocks[lk + (q,)] = block_rows.reshape(shape)
            col_offset = 0
            for ri, rk in enumerate(right_subkeys):
                n_cols = right_col_sizes[ri]
                block_cols = Vh_q[:, col_offset : col_offset + n_cols]
                col_offset += n_cols
                shape = (Vh_q.shape[0],) + tuple(
                    tensor.indices[ax].multiplicity(ch)
                    for ax, ch in zip(right_axes, rk)
                )
                Vh_blocks[(q,) + rk] = block_cols.reshape(shape)

        U_indices = left_indices + (bond_index_out,)
        Vh_indices = (bond_index_in,) + right_indices
        U_T = SymmetricTensor._from_blocks_unchecked(U_blocks, U_indices)
        Vh_T = SymmetricTensor._from_blocks_unchecked(Vh_blocks, Vh_indices)

        return U_T, s_final, Vh_T, s_final

    return _keep_filtered_traced


@contextlib.contextmanager
def sector_dropping_truncation(keep=frozenset({-1, 0, 1}), *, patch_traced_svd=True):
    """Monkeypatch the CTM truncation to drop chi-bond sectors outside ``keep``.

    A single ``with sector_dropping_truncation(keep={-1, 0, 1}):`` activates ALL
    of the following patches and restores every original on exit:

      1. ``_ctm_tensor_paired_moves._get_base_charges`` — keep-filter (sets the
         truncation intent on the FORWARD single-site paired-moves path).
      2. ``_ctm_projector._svd_projector_symmetric`` — drops non-``keep``
         ``chi_new`` columns from the eager forward block-sparse projector.
      3. ``_ctm_tensor_convergence._get_base_charges`` — the SAME keep-filter as
         (1) but on the SECOND, distinct ``_get_base_charges`` that the AD
         backward path (``_make_jit_ctm_step`` -> ``_ctm_tensor_sweep_multisite``,
         used by the Gate-B probe AND production ``_ctm_energy_ad``) reads from.
         Without this the traced truncation never sees the filter (Step 0(a)).
      4. ``tenax.linalg._truncated_svd_symmetric_traced`` — replaced by a faithful
         copy whose per-sector keep allocation (a) drops non-``keep`` sectors and
         (b) NEVER greedy-backfills the leftover χ budget from a non-``keep``
         sector.  Result: ``chi_new`` is the SUM of kept-sector allocations and
         may be < ``max_singular_values`` (~10 at D=3 χ=12).  This smaller bond
         is produced BY the truncation, so the enlarged-corner contraction
         accepts it (Step 0(b)).

    Why both (3) and (4) are needed for the BACKWARD drop
    -----------------------------------------------------
    The forward single-site (``ctm_tensor`` -> paired moves) and the AD backward
    that the Gate-B/Gate-C probes profile (``_make_jit_ctm_step`` ->
    ``_ctm_tensor_sweep_multisite`` -> 2x2 plaquette projector ->
    ``_truncated_svd_symmetric_traced``) are DIFFERENT projector paths that read
    ``base_charges`` from DIFFERENT ``_get_base_charges`` functions.  Filtering
    only the paired-moves one (1) leaves the backward truncation unfiltered, and
    even a filtered ``base_charges`` was defeated by the traced SVD's Phase-2
    greedy backfill re-adding ±2.  Patches (3)+(4) close both gaps so the drop
    reaches the TRACED backward as a self-consistent smaller bond (NOT a post-hoc
    column drop — that broke the trace; see ``_drop_bond_sectors_usv``, retained
    only for reference / the legacy forward hook).

    Parameters
    ----------
    keep:
        Iterable of integer charges to retain on the environment (chi) bonds.
    patch_traced_svd:
        Whether to apply patches (3)+(4) (the traced/backward drop).  Default
        ``True`` (the make-or-break Gate-B behaviour).  Set ``False`` to patch
        only the forward eager path (1)+(2), e.g. to inspect the forward env
        drop without touching the backward jaxpr.

    Guards
    ------
    * Either ``_get_base_charges`` returning ``None`` (dense/trivial-charge) is a
      no-op pass-through.
    * If filtering would remove *every* charge from ``base_charges``, the
      original (unfiltered) value is kept so the env can never collapse.
    * The traced allocator floors at >=1 SV restricted to ``keep`` sectors, so
      the bond can never be empty nor re-introduce a dropped sector.
    """
    keep = {int(q) for q in keep}

    def _patched_get_base_charges(a):
        bc = _ORIG_GET_BASE_CHARGES(a)
        if bc is None:
            return None
        filt = np.array([q for q in bc if int(q) in keep], dtype=np.int32)
        return filt if filt.size else bc

    def _patched_conv_get_base_charges(a):
        # Same keep-filter as the paired-moves one, but for the SECOND
        # _get_base_charges that the AD backward (_ctm_tensor_sweep_multisite)
        # reads from (Step 0(a)).
        bc = _ORIG_CONV_GET_BASE_CHARGES(a)
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

    _keep_filtered_traced = _make_keep_filtered_traced_svd(keep)

    _pm._get_base_charges = _patched_get_base_charges
    _proj._svd_projector_symmetric = _svd_proj
    if patch_traced_svd:
        _conv._get_base_charges = _patched_conv_get_base_charges
        _linalg._truncated_svd_symmetric_traced = _keep_filtered_traced
    try:
        yield
    finally:
        _pm._get_base_charges = _ORIG_GET_BASE_CHARGES
        _proj._svd_projector_symmetric = _ORIG_SVD_PROJECTOR
        _conv._get_base_charges = _ORIG_CONV_GET_BASE_CHARGES
        _linalg._truncated_svd_symmetric_traced = _ORIG_TRACED_SVD
