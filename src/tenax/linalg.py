r"""Linear algebra decompositions for Tenax tensors.

Public API::

    svd(tensor, left_labels, right_labels, ...) -> (U, s, Vh, s_full)
    rsvd(tensor, left_labels, right_labels, ...) -> (U, s, Vh)
    qr(tensor, left_labels, right_labels, ...) -> (Q, R)
    eigh(tensor, left_labels, right_labels, ...) -> (V, eigenvalues)

Each function dispatches to a block-sparse path for ``SymmetricTensor``
(decomposing each charge sector independently) or falls through to
dense ``jnp.linalg`` for ``DenseTensor``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tenax._rsvd_core import hmt_rsvd
from tenax.core.index import FlowDirection, Label, TensorIndex
from tenax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _koszul_sign,
)

# ---------- Shared helpers ----------


def _has_nonstandard_blocks(tensor: SymmetricTensor) -> bool:
    """Return True if any block violates standard conservation sum(flow*q)==0."""
    if not tensor.blocks:
        return False
    sym = tensor.indices[0].symmetry
    identity = sym.identity()
    for key in tensor.blocks:
        total = 0
        for idx, q in zip(tensor.indices, key):
            total += int(idx.flow) * q
        if total != identity:
            return True
    return False


def _group_blocks_by_bond_charge(
    tensor: SymmetricTensor,
    left_leg_positions: list[int],
    right_leg_positions: list[int],
) -> dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]]:
    """Group tensor blocks by their bond charge sector.

    For each block, the "bond charge" is determined by fusing the flow-weighted
    charges of the left legs.  Blocks sharing the same bond charge belong to
    the same diagonal block in the matrix representation.

    Args:
        tensor:              SymmetricTensor to decompose.
        left_leg_positions:  Axis positions belonging to the left (U / Q) factor.
        right_leg_positions: Axis positions belonging to the right (Vh / R) factor.

    Returns:
        Dict mapping bond charge ``q`` to a list of
        ``(left_subkey, right_subkey, block_array)`` tuples.
    """
    sym = tensor.indices[0].symmetry
    grouped: dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]] = {}

    for key, block in tensor.blocks.items():
        # Compute bond charge from left legs
        effective = [
            np.array([int(tensor.indices[i].flow) * int(key[i])], dtype=np.int32)
            for i in left_leg_positions
        ]
        q = int(sym.fuse_many(effective)[0])

        left_subkey = tuple(key[i] for i in left_leg_positions)
        right_subkey = tuple(key[i] for i in right_leg_positions)
        grouped.setdefault(q, []).append((left_subkey, right_subkey, block))

    return grouped


def _batch_blocksparse_enabled() -> bool:
    """Return True iff the ``TENAX_BATCH_BLOCKSPARSE`` umbrella gate is truthy.

    Uses the same allowlist parse as
    ``tenax.contraction.contractor`` (issue #568): only the explicit
    on-values ``"1"/"true"/"yes"/"on"`` (case-folded, stripped) enable the
    batched path, so non-canonical falsey strings such as ``"FALSE"``/``"no"``
    are never misread as enabled.  Default (unset/falsey) keeps the per-sector
    Python loop byte-identical to before.
    """
    import os

    return os.environ.get("TENAX_BATCH_BLOCKSPARSE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _grouped_decomp_by_shape(
    mats_by_q: dict[int, jax.Array],
    decomp_fn,
    group_extra_key=None,
):
    """Batch per-sector dense decompositions that share an assembled shape.

    ``mats_by_q`` maps a charge-sector key ``q`` to its assembled dense matrix.
    Sectors are partitioned into groups keyed by ``matrix.shape`` (optionally
    extended by ``group_extra_key(q)`` — e.g. the static ``chi`` for the
    truncated-SVD-AD path, which must be uniform within any ``vmap`` batch).

    For every group with more than one member the matrices are stacked and the
    decomposition is run through a single ``jax.vmap`` call; singletons call
    ``decomp_fn`` directly with no stacking.  ``decomp_fn`` must return a tuple
    of arrays.

    Returns ``results_by_q``: a dict mapping each ``q`` to ``decomp_fn``'s tuple
    output for that sector.  Iteration order of ``mats_by_q`` is preserved in
    the returned dict, so downstream block-key ordering is unchanged versus the
    sequential path; ``vmap(f) == [f(x_i)]`` keeps values (and gradients)
    identical as well.

    ``decomp_fn`` is either ``f(M) -> tuple`` (then vmapped with
    ``in_axes=0``) or, when ``group_extra_key`` is given, ``f(M, extra) ->
    tuple`` where ``extra`` is the static non-diff argument shared across the
    group (vmapped with ``in_axes=(0, None)``).
    """
    # Partition q's into shape[/extra] groups, preserving first-seen order.
    groups: dict[tuple, list[int]] = {}
    for q, mat in mats_by_q.items():
        if group_extra_key is None:
            gkey = (tuple(mat.shape),)
        else:
            gkey = (tuple(mat.shape), group_extra_key(q))
        groups.setdefault(gkey, []).append(q)

    raw_results: dict[int, tuple] = {}
    for gkey, qs in groups.items():
        if len(qs) == 1:
            q = qs[0]
            if group_extra_key is None:
                raw_results[q] = decomp_fn(mats_by_q[q])
            else:
                raw_results[q] = decomp_fn(mats_by_q[q], group_extra_key(q))
        else:
            stacked = jnp.stack([mats_by_q[q] for q in qs])
            if group_extra_key is None:
                batched = jax.vmap(decomp_fn, in_axes=0)(stacked)
            else:
                extra = group_extra_key(qs[0])
                batched = jax.vmap(decomp_fn, in_axes=(0, None))(stacked, extra)
            for i, q in enumerate(qs):
                raw_results[q] = tuple(arr[i] for arr in batched)

    # Restore original q iteration order.
    return {q: raw_results[q] for q in mats_by_q}


# ---------- Block-sparse SVD ----------


def _truncated_svd_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
    """Block-diagonal SVD for SymmetricTensor.

    Each charge sector is decomposed independently, then singular values
    are merged and truncated globally.

    Returns ``(U, s_truncated, Vh, s_full)`` where *s_full* contains all
    singular values (sorted descending) before truncation.
    """
    # Tracer-aware dispatch: if any block carries a JAX tracer (AD backward),
    # the Python-level global SV sort at lines 230-243 cannot run.  Route to
    # the traced variant that does per-sector static allocation.
    is_traced = any(isinstance(b, jax.core.Tracer) for b in tensor.blocks.values())
    if is_traced and tensor.blocks:
        return _truncated_svd_symmetric_traced(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            new_bond_label,
            normalize,
            base_charges=base_charges,
        )

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    # The permutation from original leg order to (left_axes, right_axes)
    decomp_perm = tuple(left_axes + right_axes)

    # For each charge sector, we need to know the row/col dimensions of the
    # block-diagonal matrix.  Rows are indexed by unique left_subkeys within
    # the sector; columns by unique right_subkeys.

    # Per-sector SVD results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
        ],
    ] = {}

    # Gated batched path (#569): collect assembled matrices + metadata in a
    # first pass, then dispatch the per-sector SVDs (grouped by matrix shape
    # under the gate, one-by-one otherwise). Sequential semantics unchanged.
    _batch = _batch_blocksparse_enabled()
    _mats_by_q: dict[int, jax.Array] = {}
    _meta_by_q: dict[int, tuple] = {}

    for q, entries in grouped.items():
        # Collect unique left / right subkeys (preserving order for determinism)
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        # Determine row size per left_subkey and col size per right_subkey
        # by computing the product of charge-multiplicities along each leg.
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

        # Assemble the block matrix for this charge sector
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # Defer the SVD: stash the assembled matrix + reconstruction metadata.
        _mats_by_q[q] = matrix
        _meta_by_q[q] = (
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    if _batch:
        _svd_by_q = _grouped_decomp_by_shape(
            _mats_by_q,
            lambda M: jnp.linalg.svd(M, full_matrices=False),
        )
    else:
        _svd_by_q = {
            q: jnp.linalg.svd(M, full_matrices=False) for q, M in _mats_by_q.items()
        }

    for q in _mats_by_q:
        U_q, s_q, Vh_q = _svd_by_q[q]
        left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = _meta_by_q[q]
        sector_results[q] = (
            U_q,
            s_q,
            Vh_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Global ("democratic") truncation: merge singular values from all charge
    # sectors and sort globally descending.  The largest singular values are
    # kept regardless of which sector they belong to.  This is the standard
    # choice for ground-state DMRG — it minimises the total truncation error
    # in the 2-norm (Frobenius norm of the discarded weight).
    #
    # Alternative: per-sector truncation preserves sector weight ratios but
    # can waste bond dimension on sectors with small total weight.  It may be
    # preferable for finite-temperature (purification) or time-evolution
    # applications where sector balance matters physically.
    all_sv_pairs: list[
        tuple[float, int, int]
    ] = []  # (value, sector_q, index_in_sector)
    for q, (_, s_q, _, _, _, _, _) in sector_results.items():
        s_np = np.array(s_q)
        for i, val in enumerate(s_np):
            all_sv_pairs.append((float(val), q, i))

    # Sort descending by singular value
    all_sv_pairs.sort(key=lambda x: -x[0])

    # Preserve the full singular-value spectrum before truncation
    s_full = jnp.array([v for v, _, _ in all_sv_pairs])

    # Determine global keep count
    n_total = len(all_sv_pairs)
    n_keep = n_total

    if max_truncation_err is not None and n_total > 0:
        total_sq = sum(x[0] ** 2 for x in all_sv_pairs)
        if total_sq > 0:
            trunc_sq = 0.0
            for i in range(n_total - 1, 0, -1):
                trunc_sq += all_sv_pairs[i][0] ** 2
                if trunc_sq / total_sq > max_truncation_err**2:
                    n_keep = i + 1
                    break
            else:
                n_keep = n_total

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, min(n_keep, n_total))

    # Select which singular values to keep.
    # When ``base_charges`` is supplied with ``max_singular_values``, allocate
    # keep counts per sector to match the canonical layout, mirroring the
    # traced-path behavior in ``_truncated_svd_symmetric_traced``. This is the
    # right policy whenever the caller needs a fixed bond charge structure
    # (e.g. fPEPS simple update — #558 — where ``A.l`` and ``A.r`` are the
    # same physical bond and the next step crashes if the SVD lets one drift).
    # Without ``base_charges`` the historical global "democratic" truncation is
    # retained — it minimises 2-norm truncation error and is the standard
    # choice for DMRG.
    if base_charges is not None and max_singular_values is not None:
        from tenax.algorithms._ctm_utils import _derive_charges

        available = {q: len(sector_results[q][1]) for q in sector_results}
        # Pre-build per-sector lists of (value, q, idx_in_sector). all_sv_pairs
        # is globally descending, so per-sector slices preserve within-sector
        # descending order.
        per_sector_pool: dict[int, list[tuple[float, int, int]]] = {}
        for p in all_sv_pairs:
            per_sector_pool.setdefault(p[1], []).append(p)

        def _canonical_select(target_n: int):
            """Allocate per-sector keep matching ``_derive_charges(base, n)``,
            with greedy fill for over-allocated sectors. Returns
            ``(k_per_sector, target_charges, kept_pairs_set)``.
            """
            t_charges = _derive_charges(base_charges, target_n)
            t_count: dict[int, int] = {}
            for tq in t_charges:
                t_count[int(tq)] = t_count.get(int(tq), 0) + 1
            k_per: dict[int, int] = {
                q: min(t_count.get(q, 0), available[q]) for q in available
            }
            remaining = target_n - sum(k_per.values())
            if remaining > 0:
                for q in sorted(
                    available.keys(),
                    key=lambda qq: (-(available[qq] - k_per.get(qq, 0)), qq),
                ):
                    if remaining <= 0:
                        break
                    capacity_left = available[q] - k_per.get(q, 0)
                    take = min(remaining, capacity_left)
                    if take > 0:
                        k_per[q] = k_per.get(q, 0) + take
                        remaining -= take
            pair_set = {(q, i) for q, k in k_per.items() if k > 0 for i in range(k)}
            return k_per, t_charges, pair_set

        # Iteratively expand ``n_keep`` whenever the canonical-prefix kept set
        # for the current ``n_keep`` discards weight in excess of
        # ``max_truncation_err``. The global-cumulative err computed earlier
        # assumed global top-n selection; under base_charges the canonical
        # prefix may keep weaker SVs from required sectors and discard larger
        # ones from over-represented sectors, so the actual err can exceed
        # the budget. Expand up to ``max_singular_values``; if the budget
        # still cannot be met we return what we have at the cap. (PR #561
        # codex P2 review.)
        if max_truncation_err is not None and n_total > 0:
            total_sq = sum(p[0] ** 2 for p in all_sv_pairs)
            err_sq_budget = max_truncation_err**2 * total_sq
            cap = max_singular_values
            while n_keep < cap:
                _, _, pair_set = _canonical_select(n_keep)
                discarded_sq = sum(
                    p[0] ** 2 for p in all_sv_pairs if (p[1], p[2]) not in pair_set
                )
                if discarded_sq <= err_sq_budget:
                    break
                n_keep += 1

        k_per_sector, target_charges, _ = _canonical_select(n_keep)

        # Emit ``kept`` (and therefore ``bond_charges``/``s_final``) in the
        # caller's canonical position order, *not* in global SV-magnitude
        # order. This matters because downstream code (fPEPS SU, traced path
        # consumers) applies the returned ``sigma``/``lam`` to the opposite
        # bond axis -- whose ``idx.charges`` is the canonical pattern -- via
        # ``scale_bond_axis``, which slices the scale vector by position
        # under ``np.where(idx.charges == q)``.  Mismatched ordering would
        # multiply the wrong charge sectors and silently corrupt the state
        # without crashing (PR #560 codex review).
        kept = []
        used = {q: 0 for q in k_per_sector}
        # Phase 1: fill in canonical-position order until target_charges is
        # exhausted *or* a sector runs out of its k_per_sector quota.
        for tq in target_charges:
            q = int(tq)
            if used.get(q, 0) < k_per_sector.get(q, 0):
                kept.append(per_sector_pool[q][used[q]])
                used[q] += 1
        # Phase 2: append any remaining quota for sectors that got more from
        # greedy fill than target_count requested. These tail entries don't
        # have a canonical position in ``base_charges`` -- placing them at
        # the end preserves sector-grouped contiguity for the overflow.
        for q in sorted(k_per_sector.keys()):
            while used[q] < k_per_sector[q]:
                kept.append(per_sector_pool[q][used[q]])
                used[q] += 1
    else:
        kept = all_sv_pairs[:n_keep]

    # Build bond charges and singular values in global descending order
    # so that s_final[k] pairs with U[:,k] and Vh[k,:].
    bond_charges = np.array([q for _, q, _ in kept], dtype=np.int32)
    s_final = jnp.array([v for v, _, _ in kept])

    # Per-sector: map each kept singular value to its global position
    # sector_cols[q] = list of (global_col, index_in_sector)
    sector_cols: dict[int, list[tuple[int, int]]] = {}
    for global_col, (_, q, idx_in_sector) in enumerate(kept):
        sector_cols.setdefault(q, []).append((global_col, idx_in_sector))

    if normalize and jnp.sum(s_final) > 0:
        s_final = s_final / jnp.sum(s_final)

    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    # Reconstruct U blocks: keys are (left_subkey..., bond_charge_q)
    # U has indices: (left_indices..., bond_index_out)
    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}

    for q, cols in sector_cols.items():
        U_q, _, Vh_q, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = (
            sector_results[q]
        )
        sv_indices = [idx for _, idx in cols]
        n_q = len(cols)

        # Select kept singular vectors in their global order
        U_q_trunc = U_q[:, sv_indices]
        Vh_q_trunc = Vh_q[sv_indices, :]

        # Split U_q rows back into individual left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            u_slice = U_q_trunc[row_offset : row_offset + n_rows, :]
            # Reshape: (prod(left_shape_for_lk), n_q) -> (left_shape_for_lk..., n_q)
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            u_block = u_slice.reshape(left_shape + (n_q,))
            block_key = lk + (q,)
            U_blocks[block_key] = u_block
            row_offset += n_rows

        # Split Vh_q cols back into individual right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            vh_slice = Vh_q_trunc[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            vh_block = vh_slice.reshape((n_q,) + right_shape)
            block_key = (q,) + rk
            Vh_blocks[block_key] = vh_block
            col_offset += n_cols

    # Check if input tensor has a non-identity target (e.g. boundary MPS
    # tensor targeting Sz != 0).  If so, the output tensors may also have
    # non-identity targets and need to bypass standard validation.
    input_target = 0
    if tensor.blocks:
        key0 = next(iter(tensor.blocks))
        input_target = sum(
            int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key0)
        )

    if input_target != 0:
        # Bypass validation for non-identity targets
        U_tensor = object.__new__(SymmetricTensor)
        U_tensor._indices = U_indices
        U_tensor._init_flat_buffer(U_blocks)
        Vh_tensor = object.__new__(SymmetricTensor)
        Vh_tensor._indices = Vh_indices
        Vh_tensor._init_flat_buffer(Vh_blocks)
    else:
        U_tensor = SymmetricTensor(U_blocks, U_indices)
        Vh_tensor = SymmetricTensor(Vh_blocks, Vh_indices)

    return U_tensor, s_final, Vh_tensor, s_full


# ---------- Tracer-safe block-sparse SVD ----------


def _truncated_svd_symmetric_traced(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    new_bond_label: Label,
    normalize: bool,
    base_charges: np.ndarray | None = None,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor, jax.Array]:
    """Tracer-safe block-diagonal SVD for SymmetricTensor.

    Used under JAX tracing (e.g. AD backward through implicit-FP GMRES).  Each
    charge sector is SVD'd independently via :func:`truncated_svd_ad`, which
    applies Francuz et al. Lorentzian regularization per block.

    Allocation rule (static, no global SV sort):
      * If both ``base_charges`` and ``max_singular_values`` are provided:
        ``k_q = min(target_count[q], available_q)`` where
        ``target_count[q]`` is the count of ``q`` in
        ``_derive_charges(base_charges, max_singular_values)``.  When
        ``target_count`` over-allocates a sector (target exceeds
        available), the unused budget is greedily redistributed to
        sectors with remaining capacity, ordered by largest unused
        capacity first, ties broken by smallest q for determinism.  This
        matches the eager-path :func:`_retruncate_by_base_charges` so
        traced bond dimension agrees with the forward (eager) path under
        AD (codex P2 review on PR #440).
      * If ``max_singular_values`` is None: ``k_q = min(rows_q, cols_q)``
        (full spectrum per sector).
      * Else (defensive fallback, base_charges=None and truncating):
        ``k_q = max(1, round(max_singular_values * available_q / total_available))``,
        adjusted so totals do not exceed ``max_singular_values``.

    The bond axis emerges in sector-block order, NOT global SV-descending
    order.  This differs from the eager path's output ordering; tensor
    contractions match by charge identity per block, so the permutation is
    safe.  Positional reads of S (e.g. ``S[0]``) should use ``jnp.max(S)``
    instead — see ``_compute_2x2_projector_symmetric`` line 960.

    Returns ``(U, s_truncated, Vh, s_full)``; under tracing ``s_full = s_truncated``
    (no pre-truncation spectrum tracked separately).
    """
    from tenax.algorithms._ad_primitives import truncated_svd_ad
    from tenax.algorithms._ctm_utils import _derive_charges

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector results: q -> (matrix, left_subkeys, right_subkeys,
    # left_row_sizes, right_col_sizes, available_q)
    sector_results: dict[int, tuple] = {}

    for q, entries in grouped.items():
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
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

        # Assemble the per-sector block matrix (traceable)
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
                ksign = _koszul_sign(parities, decomp_perm)
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
        k_per_sector: dict[int, int] = {q: r[5] for q, r in sector_results.items()}
    elif base_charges is not None:
        target_charges = _derive_charges(base_charges, max_singular_values)
        target_count: dict[int, int] = {}
        for tq in target_charges:
            target_count[int(tq)] = target_count.get(int(tq), 0) + 1
        k_per_sector = {
            q: min(target_count.get(q, 0), r[5]) for q, r in sector_results.items()
        }
        # Greedy fill: if base_charges over-allocates a sector (target_count[q]
        # > available_q), distribute the unused budget to sectors with
        # remaining capacity. Mirrors _retruncate_by_base_charges in
        # _ctm_tensor_projector_2x2.py:744-753, so traced bond dim matches the
        # eager path under AD (codex P2 review on PR #440 / comment 3223755136).
        # Order: largest unused capacity first, ties broken by smallest q for
        # determinism.
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
                capacity_left = sector_results[q][5] - k_per_sector.get(q, 0)
                take = min(remaining, capacity_left)
                if take > 0:
                    k_per_sector[q] = k_per_sector.get(q, 0) + take
                    remaining -= take
    else:
        # Defensive fallback: proportional to per-sector available capacity.
        total_avail = sum(r[5] for r in sector_results.values()) or 1
        k_per_sector = {
            q: max(1, round(max_singular_values * r[5] / total_avail))
            for q, r in sector_results.items()
        }
        excess = sum(k_per_sector.values()) - max_singular_values
        # First pass: drain but keep each sector >= 1
        for q in sorted(k_per_sector.keys()):
            if excess <= 0:
                break
            take = min(excess, k_per_sector[q] - 1)
            if take > 0:
                k_per_sector[q] -= take
                excess -= take
        # Second pass: if still over the cap, drain to zero
        # (least-capacity sector first, then smallest q for determinism)
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

    # Floor at >=1 total (mirrors eager n_keep = max(1, n_keep))
    total_keep = sum(k_per_sector.values())
    if total_keep == 0 and sector_results:
        best_q = max(
            sector_results.keys(),
            key=lambda q: (sector_results[q][5], -q),
        )
        k_per_sector[best_q] = 1

    # --- Per-sector AD-primitive SVD ---
    # truncated_svd_ad is a jax.custom_vjp with nondiff_argnums=(1,) (chi=k_q),
    # so the static chi must match across any vmapped batch. Under the gate we
    # group sectors by (matrix.shape, k_q) and vmap with in_axes=(0, None);
    # since vmap(f) == [f(x_i)] both values and gradients are identical to the
    # sequential loop (#569).
    sector_svd: dict[int, tuple[jax.Array, jax.Array, jax.Array]] = {}
    _ad_mats_by_q: dict[int, jax.Array] = {}
    for q, (matrix, _, _, _, _, _) in sector_results.items():
        k_q = k_per_sector.get(q, 0)
        if k_q <= 0:
            continue
        _ad_mats_by_q[q] = matrix

    if _batch_blocksparse_enabled():
        _ad_svd_by_q = _grouped_decomp_by_shape(
            _ad_mats_by_q,
            truncated_svd_ad,
            group_extra_key=lambda q: int(k_per_sector[q]),
        )
        for q, res in _ad_svd_by_q.items():
            sector_svd[q] = res
    else:
        for q, matrix in _ad_mats_by_q.items():
            # truncated_svd_ad takes a jax.Array matrix and chi
            U_q, s_q, Vh_q = truncated_svd_ad(matrix, int(k_per_sector[q]))
            sector_svd[q] = (U_q, s_q, Vh_q)

    # --- Concatenate output in canonical sector-ascending order ---
    ordered_qs = sorted(sector_svd.keys())
    bond_charges = np.repeat(
        np.array(ordered_qs, dtype=np.int32),
        np.array([sector_svd[q][1].shape[0] for q in ordered_qs], dtype=np.int32),
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

    # --- Reconstruct U / Vh block dicts ---
    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}
    for q in ordered_qs:
        _matrix, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes, _ = (
            sector_results[q]
        )
        U_q, _, Vh_q = sector_svd[q]
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            block_rows = U_q[row_offset : row_offset + n_rows, :]
            row_offset += n_rows
            shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            ) + (U_q.shape[1],)
            U_blocks[lk + (q,)] = block_rows.reshape(shape)
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            block_cols = Vh_q[:, col_offset : col_offset + n_cols]
            col_offset += n_cols
            shape = (Vh_q.shape[0],) + tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            Vh_blocks[(q,) + rk] = block_cols.reshape(shape)

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices
    U_T = SymmetricTensor._from_blocks_unchecked(U_blocks, U_indices)
    Vh_T = SymmetricTensor._from_blocks_unchecked(Vh_blocks, Vh_indices)

    return U_T, s_final, Vh_T, s_final


# ---------- Block-sparse SVD (numpy) ----------


def _truncated_svd_symmetric_np(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    max_singular_values: int | None,
    max_truncation_err: float | None,
    new_bond_label: Label,
    normalize: bool,
) -> tuple:
    """Block-diagonal SVD for SymmetricTensor using numpy (no JAX).

    Same algorithm as ``_truncated_svd_symmetric`` but returns
    ``(U_ba, s_final, Vh_ba, s_full)`` where U_ba and Vh_ba are
    :class:`~tenax.core._block_array.BlockArray` objects and
    s_final, s_full are ``np.ndarray``.
    """
    from tenax.core._block_array import BlockArray

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    # The permutation from original leg order to (left_axes, right_axes)
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector SVD results
    sector_results: dict[
        int,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
        ],
    ] = {}

    for q, entries in grouped.items():
        # Collect unique left / right subkeys (preserving order for determinism)
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())
        right_subkeys = list(right_subkeys_seen.keys())

        # Determine row size per left_subkey and col size per right_subkey
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

        # Assemble the block matrix for this charge sector
        matrix = np.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = np.asarray(block).reshape(
                left_row_sizes[li], right_col_sizes[ri]
            )
            # Apply Koszul sign for leg reordering (original -> left+right)
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ] = flat_block

        # SVD this sector
        U_q, s_q, Vh_q = np.linalg.svd(matrix, full_matrices=False)
        sector_results[q] = (
            U_q,
            s_q,
            Vh_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Global truncation: merge singular values from all charge sectors
    all_sv_pairs: list[
        tuple[float, int, int]
    ] = []  # (value, sector_q, index_in_sector)
    for q, (_, s_q, _, _, _, _, _) in sector_results.items():
        for i, val in enumerate(s_q):
            all_sv_pairs.append((float(val), q, i))

    # Sort descending by singular value
    all_sv_pairs.sort(key=lambda x: -x[0])

    # Preserve the full singular-value spectrum before truncation
    s_full = np.array([v for v, _, _ in all_sv_pairs])

    # Determine global keep count
    n_total = len(all_sv_pairs)
    n_keep = n_total

    if max_truncation_err is not None and n_total > 0:
        total_sq = sum(x[0] ** 2 for x in all_sv_pairs)
        if total_sq > 0:
            trunc_sq = 0.0
            for i in range(n_total - 1, 0, -1):
                trunc_sq += all_sv_pairs[i][0] ** 2
                if trunc_sq / total_sq > max_truncation_err**2:
                    n_keep = i + 1
                    break
            else:
                n_keep = n_total

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, min(n_keep, n_total))

    # Count per-sector keep
    kept = all_sv_pairs[:n_keep]

    # Build bond charges and singular values in global descending order
    bond_charges = np.array([q for _, q, _ in kept], dtype=np.int32)
    s_final = np.array([v for v, _, _ in kept])

    # Per-sector: map each kept singular value to its global position
    sector_cols: dict[int, list[tuple[int, int]]] = {}
    for global_col, (_, q, idx_in_sector) in enumerate(kept):
        sector_cols.setdefault(q, []).append((global_col, idx_in_sector))

    if normalize and np.sum(s_final) > 0:
        s_final = s_final / np.sum(s_final)

    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    # Reconstruct U blocks: keys are (left_subkey..., bond_charge_q)
    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_blocks: dict[BlockKey, np.ndarray] = {}
    Vh_blocks: dict[BlockKey, np.ndarray] = {}

    for q, cols in sector_cols.items():
        U_q, _, Vh_q, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = (
            sector_results[q]
        )
        sv_indices = [idx for _, idx in cols]
        n_q = len(cols)

        # Select kept singular vectors in their global order
        U_q_trunc = U_q[:, sv_indices]
        Vh_q_trunc = Vh_q[sv_indices, :]

        # Split U_q rows back into individual left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            u_slice = U_q_trunc[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            u_block = u_slice.reshape(left_shape + (n_q,))
            block_key = lk + (q,)
            U_blocks[block_key] = u_block
            row_offset += n_rows

        # Split Vh_q cols back into individual right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            vh_slice = Vh_q_trunc[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            vh_block = vh_slice.reshape((n_q,) + right_shape)
            block_key = (q,) + rk
            Vh_blocks[block_key] = vh_block
            col_offset += n_cols

    U_ba = BlockArray(blocks=U_blocks, indices=U_indices)
    Vh_ba = BlockArray(blocks=Vh_blocks, indices=Vh_indices)
    return (U_ba, s_final, Vh_ba, s_full)


def _qr_symmetric_np(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
) -> tuple:
    """Block-diagonal QR decomposition for SymmetricTensor using numpy (no JAX).

    Same algorithm as ``_qr_symmetric`` but returns
    ``(Q_ba, R_ba)`` where Q_ba and R_ba are
    :class:`~tenax.core._block_array.BlockArray` objects.
    """
    from tenax.core._block_array import BlockArray

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector QR results
    sector_results: dict[
        int,
        tuple[
            np.ndarray,
            np.ndarray,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
            int,
        ],
    ] = {}

    bond_charges_list: list[int] = []

    for q in sorted(grouped.keys()):
        entries = grouped[q]

        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
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

        # Assemble block matrix
        matrix = np.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = np.asarray(block).reshape(
                left_row_sizes[li], right_col_sizes[ri]
            )
            # Apply Koszul sign for leg reordering (original -> left+right)
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ] = flat_block

        Q_q, R_q = np.linalg.qr(matrix)
        bond_dim_q = Q_q.shape[1]

        bond_charges_list.extend([q] * bond_dim_q)
        sector_results[q] = (
            Q_q,
            R_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
            bond_dim_q,
        )

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_blocks: dict[BlockKey, np.ndarray] = {}
    R_blocks: dict[BlockKey, np.ndarray] = {}

    for q, (
        Q_q,
        R_q,
        left_subkeys,
        right_subkeys,
        left_row_sizes,
        right_col_sizes,
        bond_dim_q,
    ) in sector_results.items():
        # Split Q rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            q_slice = Q_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            q_block = q_slice.reshape(left_shape + (bond_dim_q,))
            Q_blocks[lk + (q,)] = q_block
            row_offset += n_rows

        # Split R cols back into right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            r_slice = R_q[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            r_block = r_slice.reshape((bond_dim_q,) + right_shape)
            R_blocks[(q,) + rk] = r_block
            col_offset += n_cols

    Q_ba = BlockArray(blocks=Q_blocks, indices=Q_indices)
    R_ba = BlockArray(blocks=R_blocks, indices=R_indices)
    return (Q_ba, R_ba)


# ---------- Block-sparse QR ----------


def _qr_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Block-diagonal QR decomposition for SymmetricTensor.

    Each charge sector is decomposed independently; the bond index carries
    the sector charge with multiplicity = min(left_dim, right_dim) per sector.
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector QR results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
            int,
        ],
    ] = {}

    bond_charges_list: list[int] = []
    _qr_mats_by_q: dict[int, jax.Array] = {}
    _qr_meta_by_q: dict[int, tuple] = {}

    for q in sorted(grouped.keys()):
        entries = grouped[q]

        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
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

        # Assemble block matrix
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # Defer the QR: stash the assembled matrix + reconstruction metadata
        # (in sorted-q order so bond_charges_list ordering is unchanged).
        _qr_mats_by_q[q] = matrix
        _qr_meta_by_q[q] = (
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Gated batched QR (#569): group sectors by assembled-matrix shape and
    # vmap jnp.linalg.qr; sequential one-by-one otherwise. Order preserved.
    if _batch_blocksparse_enabled():
        _qr_by_q = _grouped_decomp_by_shape(_qr_mats_by_q, lambda M: jnp.linalg.qr(M))
    else:
        _qr_by_q = {q: jnp.linalg.qr(M) for q, M in _qr_mats_by_q.items()}

    for q in _qr_mats_by_q:
        Q_q, R_q = _qr_by_q[q]
        left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = _qr_meta_by_q[q]
        bond_dim_q = Q_q.shape[1]

        bond_charges_list.extend([q] * bond_dim_q)
        sector_results[q] = (
            Q_q,
            R_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
            bond_dim_q,
        )

    bond_charges = np.array(bond_charges_list, dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_blocks: dict[BlockKey, jax.Array] = {}
    R_blocks: dict[BlockKey, jax.Array] = {}

    for q, (
        Q_q,
        R_q,
        left_subkeys,
        right_subkeys,
        left_row_sizes,
        right_col_sizes,
        bond_dim_q,
    ) in sector_results.items():
        # Split Q rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            q_slice = Q_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            q_block = q_slice.reshape(left_shape + (bond_dim_q,))
            Q_blocks[lk + (q,)] = q_block
            row_offset += n_rows

        # Split R cols back into right_subkey blocks
        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            r_slice = R_q[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            r_block = r_slice.reshape((bond_dim_q,) + right_shape)
            R_blocks[(q,) + rk] = r_block
            col_offset += n_cols

    # If the input tensor has non-standard conservation (e.g. non-zero target
    # charge at an MPS boundary), the factors may also violate sum(flow*q)==0.
    # Bypass validation in that case.
    _bypass = _has_nonstandard_blocks(tensor)
    if _bypass:
        Q_tensor = object.__new__(SymmetricTensor)
        Q_tensor._indices = tuple(Q_indices)
        Q_tensor._init_flat_buffer(Q_blocks)
        R_tensor = object.__new__(SymmetricTensor)
        R_tensor._indices = tuple(R_indices)
        R_tensor._init_flat_buffer(R_blocks)
    else:
        Q_tensor = SymmetricTensor(Q_blocks, Q_indices)
        R_tensor = SymmetricTensor(R_blocks, R_indices)

    return Q_tensor, R_tensor


# ---------- Block-sparse eigh ----------


def _eigh_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label,
    max_eigenvalues: int | None,
) -> tuple[SymmetricTensor, jax.Array]:
    """Block-diagonal Hermitian eigendecomposition for SymmetricTensor.

    Each charge sector is eigendecomposed independently, then eigenvalues
    are merged and truncated globally (keeping the largest).
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    # Check if fermionic signs are needed for leg reordering
    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector eigh results: (eigvecs, eigvals, left_subkeys, left_row_sizes)
    sector_results: dict[
        int,
        tuple[jax.Array, jax.Array, list[BlockKey], list[int]],
    ] = {}

    _eigh_mats_by_q: dict[int, jax.Array] = {}
    _eigh_meta_by_q: dict[int, tuple] = {}

    for q, entries in grouped.items():
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
        for lk, rk, _ in entries:
            if lk not in left_subkeys_seen:
                left_subkeys_seen[lk] = len(left_subkeys_seen)
            if rk not in right_subkeys_seen:
                right_subkeys_seen[rk] = len(right_subkeys_seen)

        left_subkeys = list(left_subkeys_seen.keys())

        left_row_sizes: list[int] = []
        for lk in left_subkeys:
            size = 1
            for leg_pos, charge_val in zip(left_axes, lk):
                idx = tensor.indices[leg_pos]
                size *= idx.multiplicity(charge_val)
            left_row_sizes.append(size)

        right_col_sizes: list[int] = []
        right_subkeys = list(right_subkeys_seen.keys())
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

        # Assemble the block matrix
        matrix = jnp.zeros((total_rows, total_cols), dtype=tensor.dtype)
        for lk, rk, block in entries:
            li = left_subkeys_seen[lk]
            ri = right_subkeys_seen[rk]
            row_start = sum(left_row_sizes[:li])
            col_start = sum(right_col_sizes[:ri])
            flat_block = block.reshape(left_row_sizes[li], right_col_sizes[ri])
            # Apply Koszul sign for leg reordering (original -> left+right)
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # Symmetrize for numerical stability (before the decomposition, exactly
        # as the sequential path); stash for the gated batched eigh.
        matrix = 0.5 * (matrix + matrix.conj().T)
        _eigh_mats_by_q[q] = matrix
        _eigh_meta_by_q[q] = (left_subkeys, left_row_sizes)

    # Gated batched eigh (#569): group already-Hermitian sectors by shape and
    # vmap jnp.linalg.eigh; sequential one-by-one otherwise. Order preserved.
    if _batch_blocksparse_enabled():
        _eigh_by_q = _grouped_decomp_by_shape(
            _eigh_mats_by_q, lambda M: jnp.linalg.eigh(M)
        )
    else:
        _eigh_by_q = {q: jnp.linalg.eigh(M) for q, M in _eigh_mats_by_q.items()}

    for q in _eigh_mats_by_q:
        eigvals_q, eigvecs_q = _eigh_by_q[q]
        left_subkeys, left_row_sizes = _eigh_meta_by_q[q]
        sector_results[q] = (eigvecs_q, eigvals_q, left_subkeys, left_row_sizes)

    # Global truncation: merge eigenvalues across sectors, keep top-k
    all_eig_pairs: list[tuple[float, int, int]] = []
    for q, (_, eigvals_q, _, _) in sector_results.items():
        ev_np = np.array(eigvals_q)
        for i, val in enumerate(ev_np):
            all_eig_pairs.append((float(val), q, i))

    # Sort descending by eigenvalue, then descending by index to match
    # the dense convention of taking eigvecs[:, -k:] for degenerate eigenvalues.
    all_eig_pairs.sort(key=lambda x: (-x[0], -x[2]))

    n_total = len(all_eig_pairs)
    n_keep = n_total
    if max_eigenvalues is not None:
        n_keep = min(n_keep, max_eigenvalues)
    n_keep = max(1, min(n_keep, n_total))

    kept = all_eig_pairs[:n_keep]

    # Eigenvalues in descending order
    eigenvalues = jnp.array([v for v, _, _ in kept])

    # Build bond charges in global descending eigenvalue order (matching
    # the eigenvalues array) so that V[:,k] pairs with eigenvalues[k].
    bond_charges = np.array([q for _, q, _ in kept], dtype=np.int32)
    sym = tensor.indices[0].symmetry

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )

    V_indices = left_indices + (bond_index_out,)

    # Per-sector: collect which eigenvector columns to keep and their
    # position in the global output bond dimension.
    # sector_cols[q] = list of (global_col, eigvec_index_in_sector)
    sector_cols: dict[int, list[tuple[int, int]]] = {}
    for global_col, (_, q, idx_in_sector) in enumerate(kept):
        sector_cols.setdefault(q, []).append((global_col, idx_in_sector))

    V_blocks: dict[BlockKey, jax.Array] = {}

    for q, cols in sector_cols.items():
        eigvecs_q, _, left_subkeys, left_row_sizes = sector_results[q]
        eigvec_indices = [idx for _, idx in cols]
        n_q = len(cols)

        # Select kept eigenvectors in their global order
        V_q = eigvecs_q[:, eigvec_indices]

        # Split rows back into left_subkey blocks
        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            v_slice = V_q[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            v_block = v_slice.reshape(left_shape + (n_q,))
            V_blocks[lk + (q,)] = v_block
            row_offset += n_rows

    V_tensor = SymmetricTensor(V_blocks, V_indices)
    return V_tensor, eigenvalues


# ---------- Public API ----------


def svd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
    base_charges: np.ndarray | None = None,
) -> tuple[Tensor, jax.Array, Tensor, jax.Array]:
    """Reshape tensor into matrix, compute SVD, truncate, reshape back.

    The tensor is first reshaped into a matrix by grouping left_labels as
    rows and right_labels as columns. After SVD and truncation, the result
    is reshaped back.

    The new bond leg (connecting U and Vh factors) is given label
    new_bond_label, making it immediately usable in label-based contractions.

    Output labels::

        U:  (left_labels..., new_bond_label)
        Vh: (new_bond_label, right_labels...)

    Note:
        This function is not JIT-able as a whole because the truncation
        cutoff is determined dynamically from singular values (dynamic shape).
        Apply ``@jax.jit`` to the inner SVD step only; call this at Python level.

    Args:
        tensor:               Tensor to decompose.
        left_labels:          Labels forming the "left" (U) factor.
        right_labels:         Labels forming the "right" (Vh) factor.
        new_bond_label:       Label for the new virtual bond.
        max_singular_values:  Hard cap on bond dimension after truncation.
        max_truncation_err:   Truncate until relative truncation error <= this.
        normalize:            Normalize singular values to sum to 1.
        base_charges:         Optional per-sector charge vector consumed by the
                              symmetric block-sparse path under JAX tracing.  When
                              supplied, traced inputs use
                              ``_derive_charges(base_charges, max_singular_values)``
                              for static per-sector keep allocation.  Ignored on
                              the dense path.

    Returns:
        ``(U_tensor, singular_values, Vh_tensor, singular_values_full)``
        -- U has labels ``(left_labels..., new_bond_label)``.
        Vh has labels ``(new_bond_label, right_labels...)``.
        singular_values is a 1-D JAX float array (truncated).
        singular_values_full is a 1-D JAX float array containing **all**
        singular values before truncation (length = min(left_dim, right_dim)),
        useful for computing truncation error without a second SVD.

    Raises:
        ValueError: If left_labels + right_labels don't cover all tensor labels.
    """
    all_labels = tensor.labels()
    all_labels_set = set(all_labels)
    left_set = set(left_labels)
    right_set = set(right_labels)

    if left_set | right_set != all_labels_set:
        raise ValueError(
            f"left_labels {list(left_labels)} + right_labels {list(right_labels)} "
            f"must cover all tensor labels {list(all_labels)}"
        )
    if left_set & right_set:
        raise ValueError(
            f"left_labels and right_labels must be disjoint, "
            f"got overlap: {left_set & right_set}"
        )

    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _truncated_svd_symmetric(
            tensor,
            left_labels,
            right_labels,
            max_singular_values,
            max_truncation_err,
            new_bond_label,
            normalize,
            base_charges=base_charges,
        )

    # Build axis ordering: left labels first, then right labels
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    # Get dense representation and reshape
    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)

    # SVD (not JIT-able at this level due to dynamic truncation)
    U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Preserve the full singular-value spectrum before truncation
    s_full = s

    # Determine truncation cutoff
    # Under JAX tracing (e.g. jax.grad through explicit AD), singular values
    # are abstract tracers and cannot be converted to numpy.  In that case
    # only static rank truncation (max_singular_values) is supported.
    _is_traced = isinstance(s, jax.core.Tracer)

    if _is_traced:
        n_keep = max_singular_values if max_singular_values is not None else s.shape[0]
    else:
        s_np = np.array(s)
        n_keep = len(s_np)

        if max_truncation_err is not None:
            # Keep singular values until truncation error <= max_truncation_err
            total_sq = float(np.sum(s_np**2))
            trunc_sq = 0.0
            for i in range(len(s_np) - 1, -1, -1):
                trunc_sq += float(s_np[i] ** 2)
                if trunc_sq / total_sq > max_truncation_err**2:
                    n_keep = i + 1
                    break
            else:
                n_keep = len(s_np)

        if max_singular_values is not None:
            n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, n_keep)  # always keep at least one

    # Truncate
    U = U[:, :n_keep]
    s = s[:n_keep]
    Vh = Vh[:n_keep, :]

    if normalize:
        s = s / jnp.sum(s)

    # Reshape back and build output tensors
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    U_dense = U.reshape(left_shape + (n_keep,))
    Vh_dense = Vh.reshape((n_keep,) + right_shape)

    # Build new bond index
    bond_charges_out = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    elif right_indices:
        sym = right_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges_out, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges_out, FlowDirection.IN, label=new_bond_label
    )

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_tensor = DenseTensor(U_dense, U_indices)
    Vh_tensor = DenseTensor(Vh_dense, Vh_indices)

    return U_tensor, s, Vh_tensor, s_full


# ---------- Randomized SVD helpers ----------


def _rsvd_matrix(
    matrix: jax.Array,
    rank: int,
    oversampling: int,
    n_power_iter: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Randomized SVD of a 2-D JAX array (Halko, Martinsson & Tropp 2011).

    Returns ``(U, s, Vh)`` with shapes ``(m, rank)``, ``(rank,)``,
    ``(rank, n)`` respectively.
    """
    # Shared HMT core (see tenax._rsvd_core); plain jnp decompositions here, the
    # AD-stable counterpart (truncated_lowrank_svd) injects regularized VJPs.
    return hmt_rsvd(
        matrix,
        rank,
        oversampling=oversampling,
        n_power_iter=n_power_iter,
        key=key,
        qr_fn=jnp.linalg.qr,
        svd_fn=lambda b: jnp.linalg.svd(b, full_matrices=False),
    )


def _rsvd_symmetric(
    tensor: SymmetricTensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    rank: int,
    oversampling: int,
    n_power_iter: int,
    key: jax.Array,
    new_bond_label: Label,
) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor]:
    """Block-diagonal randomized SVD for SymmetricTensor."""
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]
    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)

    grouped = _group_blocks_by_bond_charge(tensor, left_axes, right_axes)

    sym = tensor.indices[0].symmetry
    is_fermionic = sym.is_fermionic
    decomp_perm = tuple(left_axes + right_axes)

    # Per-sector RSVD results
    sector_results: dict[
        int,
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            list[BlockKey],
            list[BlockKey],
            list[int],
            list[int],
        ],
    ] = {}

    sector_idx = 0
    for q, entries in grouped.items():
        left_subkeys_seen: dict[BlockKey, int] = {}
        right_subkeys_seen: dict[BlockKey, int] = {}
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
                ksign = _koszul_sign(parities, decomp_perm)
                if ksign < 0:
                    flat_block = -flat_block
            matrix = matrix.at[
                row_start : row_start + left_row_sizes[li],
                col_start : col_start + right_col_sizes[ri],
            ].set(flat_block)

        # RSVD this sector
        sector_rank = min(rank, total_rows, total_cols)
        subkey = jax.random.fold_in(key, sector_idx)
        sector_idx += 1
        U_q, s_q, Vh_q = _rsvd_matrix(
            matrix, sector_rank, oversampling, n_power_iter, subkey
        )
        sector_results[q] = (
            U_q,
            s_q,
            Vh_q,
            left_subkeys,
            right_subkeys,
            left_row_sizes,
            right_col_sizes,
        )

    # Global truncation across sectors (keep top-`rank` singular values)
    all_sv_pairs: list[tuple[float, int, int]] = []
    for q, (_, s_q, _, _, _, _, _) in sector_results.items():
        s_np = np.array(s_q)
        for i, val in enumerate(s_np):
            all_sv_pairs.append((float(val), q, i))

    all_sv_pairs.sort(key=lambda x: -x[0])

    n_keep = min(rank, len(all_sv_pairs))
    n_keep = max(1, n_keep)
    kept = all_sv_pairs[:n_keep]

    bond_charges = np.array([q for _, q, _ in kept], dtype=np.int32)
    s_final = jnp.array([v for v, _, _ in kept])

    sector_cols: dict[int, list[tuple[int, int]]] = {}
    for global_col, (_, q, idx_in_sector) in enumerate(kept):
        sector_cols.setdefault(q, []).append((global_col, idx_in_sector))

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_blocks: dict[BlockKey, jax.Array] = {}
    Vh_blocks: dict[BlockKey, jax.Array] = {}

    for q, cols in sector_cols.items():
        U_q, _, Vh_q, left_subkeys, right_subkeys, left_row_sizes, right_col_sizes = (
            sector_results[q]
        )
        sv_indices = [idx for _, idx in cols]
        n_q = len(cols)

        U_q_trunc = U_q[:, sv_indices]
        Vh_q_trunc = Vh_q[sv_indices, :]

        row_offset = 0
        for li, lk in enumerate(left_subkeys):
            n_rows = left_row_sizes[li]
            u_slice = U_q_trunc[row_offset : row_offset + n_rows, :]
            left_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(left_axes, lk)
            )
            u_block = u_slice.reshape(left_shape + (n_q,))
            U_blocks[lk + (q,)] = u_block
            row_offset += n_rows

        col_offset = 0
        for ri, rk in enumerate(right_subkeys):
            n_cols = right_col_sizes[ri]
            vh_slice = Vh_q_trunc[:, col_offset : col_offset + n_cols]
            right_shape = tuple(
                tensor.indices[ax].multiplicity(ch) for ax, ch in zip(right_axes, rk)
            )
            vh_block = vh_slice.reshape((n_q,) + right_shape)
            Vh_blocks[(q,) + rk] = vh_block
            col_offset += n_cols

    input_target = 0
    if tensor.blocks:
        key0 = next(iter(tensor.blocks))
        input_target = sum(
            int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key0)
        )

    if input_target != 0:
        U_tensor = object.__new__(SymmetricTensor)
        U_tensor._indices = U_indices
        U_tensor._init_flat_buffer(U_blocks)
        Vh_tensor = object.__new__(SymmetricTensor)
        Vh_tensor._indices = Vh_indices
        Vh_tensor._init_flat_buffer(Vh_blocks)
    else:
        U_tensor = SymmetricTensor(U_blocks, U_indices)
        Vh_tensor = SymmetricTensor(Vh_blocks, Vh_indices)

    return U_tensor, s_final, Vh_tensor


def rsvd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    rank: int = 10,
    oversampling: int = 5,
    n_power_iter: int = 1,
    key: jax.Array | None = None,
) -> tuple[Tensor, jax.Array, Tensor]:
    """Randomized SVD of a tensor (Halko, Martinsson & Tropp 2011).

    Computes an approximate rank-*k* SVD using a randomized algorithm.
    This is much faster than a full SVD when only the top singular values
    are needed and the matrix is large.

    The tensor is first reshaped into a matrix by grouping *left_labels* as
    rows and *right_labels* as columns.  After the randomized SVD the result
    is reshaped back.

    Output labels::

        U:  (left_labels..., new_bond_label)
        Vh: (new_bond_label, right_labels...)

    Args:
        tensor:          Tensor to decompose.
        left_labels:     Labels forming the "left" (U) factor.
        right_labels:    Labels forming the "right" (Vh) factor.
        new_bond_label:  Label for the new virtual bond.
        rank:            Target rank (number of singular values to compute).
        oversampling:    Extra random vectors for accuracy (default 5).
        n_power_iter:    Number of power iterations (default 1).
        key:             JAX PRNG key.  If *None*, ``PRNGKey(0)`` is used.

    Returns:
        ``(U_tensor, singular_values, Vh_tensor)``
        -- U has labels ``(left_labels..., new_bond_label)``.
        Vh has labels ``(new_bond_label, right_labels...)``.
        singular_values is a 1-D JAX float array of length <= *rank*.

    Raises:
        ValueError: If left_labels + right_labels don't cover all tensor labels.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    all_labels = tensor.labels()
    all_labels_set = set(all_labels)
    left_set = set(left_labels)
    right_set = set(right_labels)

    if left_set | right_set != all_labels_set:
        raise ValueError(
            f"left_labels {list(left_labels)} + right_labels {list(right_labels)} "
            f"must cover all tensor labels {list(all_labels)}"
        )
    if left_set & right_set:
        raise ValueError(
            f"left_labels and right_labels must be disjoint, "
            f"got overlap: {left_set & right_set}"
        )

    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _rsvd_symmetric(
            tensor,
            left_labels,
            right_labels,
            rank,
            oversampling,
            n_power_iter,
            key,
            new_bond_label,
        )

    # Dense path
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)

    U, s, Vh = _rsvd_matrix(matrix, rank, oversampling, n_power_iter, key)
    n_keep = len(s)

    # Reshape back and build output tensors
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    U_dense = U.reshape(left_shape + (n_keep,))
    Vh_dense = Vh.reshape((n_keep,) + right_shape)

    # Build new bond index
    bond_charges_out = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    elif right_indices:
        sym = right_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges_out, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges_out, FlowDirection.IN, label=new_bond_label
    )

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_tensor = DenseTensor(U_dense, U_indices)
    Vh_tensor = DenseTensor(Vh_dense, Vh_indices)

    return U_tensor, s, Vh_tensor


def qr(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
) -> tuple[Tensor, Tensor]:
    """QR decomposition of a tensor for canonical form in DMRG.

    Reshapes tensor into a matrix, performs QR, then reshapes back.

    Output labels::

        Q: (left_labels..., new_bond_label)
        R: (new_bond_label, right_labels...)

    Args:
        tensor:          Tensor to decompose.
        left_labels:     Labels forming the Q (isometric) factor.
        right_labels:    Labels forming the R (upper triangular) factor.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        (Q_tensor, R_tensor) where Q is isometric (Q^dag Q = I).
    """
    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _qr_symmetric(tensor, left_labels, right_labels, new_bond_label)

    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    Q, R = jnp.linalg.qr(matrix)

    bond_dim = Q.shape[1]
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    Q_dense = Q.reshape(left_shape + (bond_dim,))
    R_dense = R.reshape((bond_dim,) + right_shape)

    bond_charges = np.zeros(bond_dim, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index_out = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    bond_index_in = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.IN, label=new_bond_label
    )

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_tensor = DenseTensor(Q_dense, Q_indices)
    R_tensor = DenseTensor(R_dense, R_indices)

    return Q_tensor, R_tensor


def eigh(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_eigenvalues: int | None = None,
) -> tuple[Tensor, jax.Array]:
    """Eigendecompose a Hermitian tensor.

    Reshapes the tensor into a square matrix (left_labels vs right_labels),
    computes the eigendecomposition, and returns eigenvectors as a Tensor.

    Eigenvalues are sorted in descending order. If ``max_eigenvalues`` is
    given, only the top-k eigenvalues (and corresponding eigenvectors) are
    kept.

    Output labels::

        V: (left_labels..., new_bond_label)

    Args:
        tensor:           Hermitian tensor to decompose.
        left_labels:      Labels forming the row side of the matrix.
        right_labels:     Labels forming the column side of the matrix.
        new_bond_label:   Label for the eigenvector bond index.
        max_eigenvalues:  Keep only the top-k eigenvalues.

    Returns:
        ``(V, eigenvalues)`` where V has labels ``(left_labels..., new_bond_label)``
        and eigenvalues is a 1-D JAX array sorted descending.
    """
    # Dispatch to block-sparse path for SymmetricTensor
    if isinstance(tensor, SymmetricTensor):
        return _eigh_symmetric(
            tensor, left_labels, right_labels, new_bond_label, max_eigenvalues
        )

    # Dense path
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    matrix = 0.5 * (matrix + matrix.conj().T)  # symmetrize
    eigvals, eigvecs = jnp.linalg.eigh(matrix)

    # eigh returns ascending; reverse for descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Truncate
    n_total = len(eigvals)
    n_keep = n_total
    if max_eigenvalues is not None:
        n_keep = min(n_keep, max_eigenvalues)
    n_keep = max(1, n_keep)

    eigvals = eigvals[:n_keep]
    eigvecs = eigvecs[:, :n_keep]

    # Reshape back
    left_shape = tuple(idx.dim for idx in left_indices)
    V_dense = eigvecs.reshape(left_shape + (n_keep,))

    bond_charges = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tenax.core.symmetry import U1Symmetry

        sym = U1Symmetry()

    bond_index = TensorIndex.from_charges(
        sym, bond_charges, FlowDirection.OUT, label=new_bond_label
    )
    V_indices = left_indices + (bond_index,)
    V_tensor = DenseTensor(V_dense, V_indices)

    return V_tensor, eigvals
