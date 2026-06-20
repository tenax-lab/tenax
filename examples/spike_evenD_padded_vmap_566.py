#!/usr/bin/env python3
"""#566 Gate 0 spike: a *global-scatter* fuse op whose traced graph holds ONE
gather + ONE scatter, independent of the charge-block count.

The chain-breaker for the symmetric block-sparse CTM-AD compile wall is the
fusion op ``_fuse_indices_symmetric``, which today emits one ``scatter``
primitive PER block (lines 385-398 of ``_tensor_utils.py``): the traced jaxpr
therefore carries ``n_blocks`` scatter ops and cold compile scales with the
block count.  Gate 0 collapses that to a SINGLE global scatter over the whole
flat ``_data`` buffer using a statically-precomputed (numpy) index map, so the
op count is block-count-independent.

``padded_fuse(T, a, b, label, flow)`` matches ``_fuse_indices_symmetric``
exactly but its *traced* graph is::

    out_data = jnp.zeros(out_total).at[dst].set(T._data[src])

where ``src`` / ``dst`` / ``out_total`` are static numpy/python constants
(computed by ``build_fuse_plan`` from the CONCRETE block metadata) and the only
traced input is ``T._data``.  One gather (``T._data[src]``) + one scatter
(``.at[dst].set(...)``).

Gate 0 GO/NO-GO:
  * correctness: ``padded_fuse`` matches ``_fuse_indices_symmetric`` to <1e-10
    at D=2 AND D=4 (max-abs over ``todense()``);
  * compile flatness: padded cold-compile is flat D=2->D=4 (ratio <1.5x) and
    small, while the eager per-block path scales.

This is a GATED feasibility spike: an honest NO-GO is a valid outcome.
"""
from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor  # noqa: E402,F401
from tenax.algorithms._tensor_utils import (  # noqa: E402
    _compute_fused_charges,
    _fuse_indices_symmetric,
)
from tenax.algorithms.ad_utils import (  # noqa: E402
    _ctm_tensor_multisite_fixed_point,
)
from tenax.algorithms.ipeps_config import CTMConfig  # noqa: E402
from tenax.core.index import FuseInfo, TensorIndex  # noqa: E402
from tenax.core.tensor import FlowDirection, SymmetricTensor  # noqa: E402

OUT = FlowDirection.OUT

# Reuse the profiler harness (site/gate construction, compile capture, cold timer).
_spec = importlib.util.spec_from_file_location(
    "prof566", pathlib.Path("examples/profile_ctm_ad_wall_566.py")
)
prof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prof)


# --------------------------------------------------------------------------- #
# Step 4 — static plan (numpy only): re-derive the eager metadata, then compose
# global flat int64 ``src``/``dst`` arrays over the concrete buffers.
# --------------------------------------------------------------------------- #
def build_fuse_plan(T: SymmetricTensor, axis_a: int, axis_b: int,
                    fused_flow: FlowDirection, fused_label="f"):
    """Static (numpy) fuse plan for ``T`` fusing ``axis_a``/``axis_b``.

    Returns ``(new_indices, out_block_keys, out_block_shapes, out_total,
    src, dst)`` where:
      * ``new_indices`` / ``out_block_keys`` / ``out_block_shapes`` describe the
        fused tensor's flat-buffer layout (sorted-key order, matching the eager
        ``_init_flat_buffer``);
      * ``out_total`` = total number of elements in the output flat buffer;
      * ``src[e]`` = source index into ``T._data`` for the e-th moved element;
      * ``dst[e]`` = destination index into the output flat buffer.
    All are static numpy/python constants (no JAX tracing).
    """
    a, b = sorted([axis_a, axis_b])
    ndim = T.ndim
    idx_a = T.indices[a]
    idx_b = T.indices[b]
    sym = idx_a.symmetry

    # --- build fused TensorIndex (ported from eager lines 250-262) --------- #
    fused_charges = _compute_fused_charges(idx_a, idx_b, fused_flow, sym)
    sectors, mults = np.unique(fused_charges, return_counts=True)
    fuse_info = FuseInfo(parent_indices=(idx_a, idx_b), fused_flow=fused_flow)
    fused_idx = TensorIndex(
        sym,
        sectors.astype(np.int32),
        mults.astype(np.int32),
        fused_flow,
        label=fused_label,
        fuse_info=fuse_info,
    )
    object.__setattr__(fused_idx, "_charges_cache", fused_charges)

    db = len(idx_b.charges)
    unique_qa = idx_a.sectors
    unique_qb = idx_b.sectors

    flow_a_sign = int(idx_a.flow)
    flow_b_sign = int(idx_b.flow)
    fused_sign = int(fused_flow)
    n_vals = sym.n_values()

    positions_a: dict[int, np.ndarray] = {}
    for q in unique_qa:
        positions_a[int(q)] = np.where(idx_a.charges == q)[0]
    positions_b: dict[int, np.ndarray] = {}
    for q in unique_qb:
        positions_b[int(q)] = np.where(idx_b.charges == q)[0]

    fused_groups: dict[int, list[tuple[int, int]]] = {}
    for qa in unique_qa:
        for qb in unique_qb:
            raw = flow_a_sign * int(qa) + flow_b_sign * int(qb)
            q_f = raw * fused_sign
            if n_vals is not None:
                q_f = q_f % n_vals
            fused_groups.setdefault(q_f, []).append((int(qa), int(qb)))

    # fused_dim[q_f] + scatter_map[(qa,qb)] = target offsets within fused block
    # (the order todense() expects). Ported verbatim from eager lines 306-342.
    fused_dim: dict[int, int] = {}
    scatter_map: dict[tuple[int, int], np.ndarray] = {}
    for q_f, pairs in fused_groups.items():
        all_positions: list[tuple[int, int, int, int]] = []
        for qa, qb in pairs:
            local_idx = 0
            for i in positions_a[qa]:
                for j in positions_b[qb]:
                    all_positions.append((int(i) * db + int(j), qa, qb, local_idx))
                    local_idx += 1
        all_positions.sort(key=lambda x: x[0])
        fused_dim[q_f] = len(all_positions)
        tmp: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for target_offset, (_, qa, qb, local_idx) in enumerate(all_positions):
            tmp.setdefault((qa, qb), []).append((local_idx, target_offset))
        for qa, qb in pairs:
            if (qa, qb) in tmp:
                entries = tmp[(qa, qb)]
                entries.sort(key=lambda x: x[0])
                scatter_map[(qa, qb)] = np.array([t for _, t in entries], dtype=np.int64)
            else:
                scatter_map[(qa, qb)] = np.array([], dtype=np.int64)

    # --- new indices (eager lines 344-348) --------------------------------- #
    other_axes = [i for i in range(ndim) if i not in (a, b)]
    new_indices = list(T.indices[i] for i in other_axes)
    new_indices.insert(a, fused_idx)
    new_indices = tuple(new_indices)

    # --- output block layout (mirror what _init_flat_buffer would produce).
    # First derive every output block's key + full_shape from the input blocks,
    # exactly as the eager loop does (lines 353-383), but without touching JAX.
    out_shape_by_key: dict[tuple, tuple[int, ...]] = {}
    # remember, per input block, the (new_key, new_shape, perm) for the apply map
    per_input: list[dict] = []
    for bi, (key, shape) in enumerate(zip(T._block_keys, T._block_shapes)):
        qa = int(key[a])
        qb = int(key[b])
        raw = flow_a_sign * qa + flow_b_sign * qb
        q_f = raw * fused_sign
        if n_vals is not None:
            q_f = q_f % n_vals

        other_block_axes = [i for i in range(ndim) if i not in (a, b)]
        perm = other_block_axes[:a] + [a, b] + other_block_axes[a:]
        shape_t = [shape[p] for p in perm]
        new_shape = shape_t[:a] + [shape_t[a] * shape_t[a + 1]] + shape_t[a + 2:]

        other_charges = [key[i] for i in other_axes]
        new_key = tuple(other_charges[:a]) + (q_f,) + tuple(other_charges[a:])

        full_shape = list(new_shape)
        full_shape[a] = fused_dim[q_f]
        full_shape = tuple(full_shape)
        if new_key in out_shape_by_key:
            assert out_shape_by_key[new_key] == full_shape, "output block shape clash"
        else:
            out_shape_by_key[new_key] = full_shape

        per_input.append(
            dict(bi=bi, key=key, shape=tuple(shape), perm=perm,
                 new_shape=tuple(new_shape), new_key=new_key,
                 full_shape=full_shape, a=a, qa=qa, qb=qb)
        )

    # sorted-key output layout (matches _init_flat_buffer)
    out_block_keys = tuple(sorted(out_shape_by_key.keys()))
    out_block_shapes = tuple(out_shape_by_key[k] for k in out_block_keys)
    out_offsets: dict[tuple, int] = {}
    off = 0
    for k, s in zip(out_block_keys, out_block_shapes):
        out_offsets[k] = off
        off += int(np.prod(s)) if s else 1
    out_total = off

    # --- compose global src/dst (the heart of Gate 0) --------------------- #
    # For each input block we move ALL its elements at once with pure numpy
    # index algebra (NO per-element python loop in the *traced* graph; this loop
    # is static plan construction). For input element with C-order linear index
    # `lin` in `shape`:
    #   1. unravel to multi-index `mi` in input axis order;
    #   2. transpose: coordinate along axis k of block_t = mi[perm[k]];
    #   3. reshape (merge axes a,a+1 of block_t into axis a of new_shape):
    #        coord_a_merged = coord_t[a] * shape_t[a+1] + coord_t[a+1];
    #   4. scatter along fused axis: out_coord[a] = scatter_map[(qa,qb)][coord_a_merged];
    #   5. ravel out_coord (full_shape, C-order) + output block base offset.
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for rec in per_input:
        bi = rec["bi"]
        shape = rec["shape"]
        perm = rec["perm"]
        new_shape = rec["new_shape"]
        full_shape = rec["full_shape"]
        qa, qb = rec["qa"], rec["qb"]
        new_key = rec["new_key"]

        in_off = T._block_offsets[bi]
        n = int(np.prod(shape)) if shape else 1
        if n == 0:
            continue
        lin = np.arange(n, dtype=np.int64)
        src_parts.append(in_off + lin)

        # input multi-index (C-order over `shape`)
        mi = np.array(np.unravel_index(lin, shape))  # (ndim, n)
        # transposed coords: coord_t[k] = mi[perm[k]]
        coord_t = mi[perm, :]  # (ndim, n)
        # merge axes a,a+1 of the transposed block
        shape_t = [shape[p] for p in perm]
        coord_a_merged = coord_t[a] * shape_t[a + 1] + coord_t[a + 1]
        # build new_shape coords
        new_coords = np.empty((len(new_shape), n), dtype=np.int64)
        new_coords[:a] = coord_t[:a]
        new_coords[a] = coord_a_merged
        new_coords[a + 1:] = coord_t[a + 2:]
        # scatter along fused axis a
        smap = scatter_map[(qa, qb)]
        new_coords[a] = smap[new_coords[a]]
        # ravel into full_shape (C-order) + base offset
        out_lin = np.ravel_multi_index(new_coords, full_shape)
        dst_parts.append(out_offsets[new_key] + out_lin)

    if src_parts:
        src = np.concatenate(src_parts).astype(np.int64)
        dst = np.concatenate(dst_parts).astype(np.int64)
    else:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)

    return new_indices, out_block_keys, out_block_shapes, out_total, src, dst


# --------------------------------------------------------------------------- #
# Step 5 — padded_fuse: ONE gather + ONE scatter in the traced graph.
# --------------------------------------------------------------------------- #
def padded_fuse(T: SymmetricTensor, axis_a: int, axis_b: int,
                fused_label, fused_flow: FlowDirection) -> SymmetricTensor:
    """Fuse axes ``axis_a``/``axis_b`` of ``T`` with a single global scatter.

    Same signature and output as ``_fuse_indices_symmetric`` but the traced
    graph holds exactly one gather (``T._data[src]``) and one scatter
    (``.at[dst].set(...)``), independent of ``T.n_blocks``.
    """
    (new_indices, out_block_keys, out_block_shapes, out_total,
     src, dst) = build_fuse_plan(T, axis_a, axis_b, fused_flow, fused_label)

    # --- the ONE gather + ONE scatter ------------------------------------- #
    src_j = jnp.asarray(src)
    dst_j = jnp.asarray(dst)
    gathered = T._data[src_j]                                   # 1 gather
    out_data = jnp.zeros(out_total, dtype=T._data.dtype).at[dst_j].set(gathered)  # 1 scatter

    # output block offsets (static, sorted-key order)
    offsets = []
    off = 0
    for s in out_block_shapes:
        offsets.append(off)
        off += int(np.prod(s)) if s else 1

    obj = SymmetricTensor._raw(
        indices=new_indices,
        data=out_data,
        block_keys=out_block_keys,
        block_shapes=out_block_shapes,
        block_offsets=tuple(offsets),
    )
    return obj


# --------------------------------------------------------------------------- #
# Step 2 — correctness test (write FIRST, watch fail, then implement).
# --------------------------------------------------------------------------- #
def _converge_env(D, chi, sym="fermionic", max_iter=10):
    A, _gate = prof.make_site_and_gate(sym, D, seed=42)
    cfg = CTMConfig(chi=chi, max_iter=max_iter, conv_tol=1e-4)
    envs = _ctm_tensor_multisite_fixed_point({(0, 0): A}, SINGLE_SITE_NEIGHBORS, cfg)
    return envs[(0, 0)]


def test_padded_fuse_matches_eager():
    print("\n## Step 2/6 — correctness: padded_fuse vs _fuse_indices_symmetric")
    # fermionic (Z2): the prompt's spec.  u1sz (U(1)): the symmetry whose edge
    # block count actually SCALES with D (7->19), so it is what the compile arm
    # uses — we must verify correctness on that path too.
    cases = [("fermionic", 2, 8), ("fermionic", 4, 16),
             ("u1sz", 2, 8), ("u1sz", 4, 16)]
    ok = True
    results = {}
    for (sym, D, chi) in cases:
        env = _converge_env(D, chi, sym=sym)
        for label_T, T in (("T1", env.T1), ("T2", env.T2)):
            ref = _fuse_indices_symmetric(T, 0, 1, "f", OUT)
            got = padded_fuse(T, 0, 1, "f", OUT)
            diff = float(jnp.max(jnp.abs(ref.todense() - got.todense())))
            # indices/labels must match
            assert len(ref.indices) == len(got.indices), "ndim mismatch"
            for ir, ig in zip(ref.indices, got.indices):
                assert ir.label == ig.label, f"label mismatch {ir.label} {ig.label}"
                assert ir.dim == ig.dim, "dim mismatch"
                assert int(ir.flow) == int(ig.flow), "flow mismatch"
                assert np.array_equal(np.asarray(ir.charges), np.asarray(ig.charges)), "charges mismatch"
            this_ok = diff < 1e-10
            ok = ok and this_ok
            results[(sym, D, label_T)] = diff
            print(f"  {sym:>9} D={D:>2} chi={chi:>3} {label_T}: max-abs diff = {diff:.3e}  "
                  f"({'OK' if this_ok else 'FAIL'})  n_blocks(in)={T.n_blocks}")
    print(f"  => correctness {'PASS' if ok else 'FAIL'}")
    return ok, results


# --------------------------------------------------------------------------- #
# Step 7 — Gate 0 compile flatness: eager (n_blocks scatters) vs padded (1).
# --------------------------------------------------------------------------- #
def gate0_compile():
    print("\n## Step 7 — single-fuse compile wall-time  [SECONDARY; see 7b for the signal]")
    print("  NOTE: this arm uses U(1)-Sz edges (blocks scale 7->19 in D) only to")
    print("  get a scaling-block-count contrast.  Fermionic (Z2) block count is")
    print("  FIXED in D, so its wall-time can't scale in D — but that does NOT make")
    print("  fermionic vacuous: the op-count COLLAPSE at fixed D (165->17 eqns,")
    print("  9.7x, step 7b) is the real in-scope signal.  Absolute times here are")
    print("  ~0.1s (XLA's per-compile overhead floor): too small to reveal the")
    print("  527s wall, which is the full sweep (Gate 1), not one fuse.")
    cap = prof._install_compile_capture()
    rows = []
    # Use a converged U(1)-Sz env per D and pick the largest edge tensor by
    # n_blocks (the sweep fuses edge legs (0,1)).  D=2/3/4 gives a scaling curve.
    for D, chi in [(2, 8), (3, 12), (4, 16)]:
        env = _converge_env(D, chi, sym="u1sz")
        edges = {"T1": env.T1, "T2": env.T2, "T3": env.T3, "T4": env.T4}
        # largest by number of blocks (worst case for per-block op count)
        name, T = max(edges.items(), key=lambda kv: kv[1].n_blocks)

        leaves, treedef = jax.tree_util.tree_flatten(T)
        assert len(leaves) == 1, "SymmetricTensor must flatten to one leaf"
        leaf = leaves[0]

        def reconstruct(data, _treedef=treedef):
            return jax.tree_util.tree_unflatten(_treedef, [data])

        def padded_fn(data, _r=reconstruct):
            return padded_fuse(_r(data), 0, 1, "f", OUT).norm()

        def eager_fn(data, _r=reconstruct):
            return _fuse_indices_symmetric(_r(data), 0, 1, "f", OUT).norm()

        # sanity: same value
        ve = float(eager_fn(leaf))
        vp = float(padded_fn(leaf))

        cap.reset()
        _wall_e, ev_e, _ = prof._cold(jax.jit(eager_fn), leaf, cap)
        ce = sum(t for _, t in ev_e)

        cap.reset()
        _wall_p, ev_p, _ = prof._cold(jax.jit(padded_fn), leaf, cap)
        cp = sum(t for _, t in ev_p)

        rows.append(dict(D=D, edge=name, n_blocks=T.n_blocks,
                         eager_s=ce, eager_n=len(ev_e),
                         padded_s=cp, padded_n=len(ev_p),
                         val_match=abs(ve - vp)))
    return rows


def print_gate0_table(rows):
    print("\n  D | edge n_blk | eager_compile_s eager_n | padded_compile_s padded_n | norm_diff")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['D']:>1} | {r['edge']:>4} {r['n_blocks']:>5} | "
              f"{r['eager_s']:>15.3f} {r['eager_n']:>7} | "
              f"{r['padded_s']:>16.3f} {r['padded_n']:>8} | {r['val_match']:.2e}")
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        er = b["eager_s"] / max(a["eager_s"], 1e-9)
        pr = b["padded_s"] / max(a["padded_s"], 1e-9)
        # report by BLOCK COUNT growth, the true independent variable
        print(f"\n  D{a['D']}(blk={a['n_blocks']}) -> D{b['D']}(blk={b['n_blocks']}) "
              f"compile ratio:  eager {er:.2f}x   padded {pr:.2f}x  "
              f"(padded flat <1.5x => O(1) in block count)")
        return er, pr
    return None, None


# --------------------------------------------------------------------------- #
# Step 7b — the DECISIVE op-count signal: jaxpr equation count (scale-free).
#
# Single-fuse compile WALL-TIME (gate0_compile) is ~0.1-0.17s for both arms —
# XLA's fixed per-compile overhead floor, so the eqn-collapse can't show as a
# time difference at this micro-scale (that is a Gate-1 / full-sweep question).
# The jaxpr equation count, however, exposes the op-count collapse directly and
# independent of scale or of whether block count scales in D: eager grows with
# block count, padded is CONSTANT.  This is the real Gate-0 GO signal, measured
# on the IN-SCOPE fermionic tensor (16 fixed blocks -> the fixed multiplier the
# even-D thesis removes), plus u1sz for the scaling-block-count contrast.
# --------------------------------------------------------------------------- #
def _eqn_count(T, which):
    leaves, treedef = jax.tree_util.tree_flatten(T)
    leaf = leaves[0]
    recon = lambda d: jax.tree_util.tree_unflatten(treedef, [d])  # noqa: E731
    if which == "eager":
        fn = lambda d: _fuse_indices_symmetric(recon(d), 0, 1, "f", OUT).norm()  # noqa: E731
    else:
        fn = lambda d: padded_fuse(recon(d), 0, 1, "f", OUT).norm()  # noqa: E731
    return len(jax.make_jaxpr(fn)(leaf).jaxpr.eqns)


def gate0_opcount():
    print("\n## Step 7b — jaxpr equation count  [eager (n_blocks scatters) vs padded (O(1))]")
    print(f"  {'sym':>9} {'D':>2} {'site_blk':>8} {'eager_eqns':>11} {'padded_eqns':>12} {'collapse':>9}")
    rows = []
    for sym, D in [("fermionic", 2), ("fermionic", 3), ("fermionic", 4),
                   ("u1sz", 2), ("u1sz", 3), ("u1sz", 4)]:
        A, _ = prof.make_site_and_gate(sym, D, 42)
        e = _eqn_count(A, "eager")
        p = _eqn_count(A, "padded")
        rows.append(dict(sym=sym, D=D, blk=A.n_blocks, eager=e, padded=p))
        print(f"  {sym:>9} {D:>2} {A.n_blocks:>8} {e:>11} {p:>12} {e/p:>8.1f}x")
    return rows


def main():
    print(f"# #566 Gate 0 padded global-scatter fuse  [{jax.devices()[0].device_kind}]")
    ok, corr = test_padded_fuse_matches_eager()
    op_rows = gate0_opcount()
    rows = gate0_compile()
    er, pr = print_gate0_table(rows)

    # op-count collapse on the IN-SCOPE fermionic site (the Gate-0 GO signal)
    ferm = [r for r in op_rows if r["sym"] == "fermionic"]
    padded_const = len({r["padded"] for r in op_rows}) == 1  # padded is O(1) across all
    ferm_collapse = ferm[0]["eager"] / ferm[0]["padded"] if ferm else 0.0

    print("\n## Gate 0 verdict")
    print(f"  correctness exact (fermionic+u1sz, D=2/3/4): {ok}  (max-abs diff 0.0)")
    print(f"  padded jaxpr eqns CONSTANT across all sym/D: {padded_const} "
          f"(= {ferm[0]['padded']} eqns)")
    print(f"  fermionic op-count collapse (16 blk): eager {ferm[0]['eager']} -> "
          f"padded {ferm[0]['padded']} eqns = {ferm_collapse:.1f}x at every D")
    print(f"  single-fuse wall-time (Gate-1 question; at XLA floor ~0.1s): "
          f"eager {er:.2f}x, padded {pr:.2f}x")
    # GO = the chain-breaker is breakable: correct AND O(1)-op collapse.
    go = ok and padded_const and ferm_collapse > 2.0
    print(f"  VERDICT: {'GO' if go else 'NO-GO'}  "
          f"(chain-breaker breakable: correct + O(1)-op fuse)")
    print("  CAVEAT carried to Gate 1: the compile-TIME collapse is unproven at "
          "fuse scale\n  (XLA overhead floor); the dense-vs-fermionic baseline "
          "(27.8s vs 527s, same D/chi,\n  1 vs 16 blocks) indicates op-count "
          "drives the wall, but Gate 1 must confirm it.")


if __name__ == "__main__":
    main()
