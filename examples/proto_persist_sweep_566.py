#!/usr/bin/env python3
"""Persist-across-calls op-count ceiling prototype (#566 P1d).

THROWAWAY / RECORD. Drives a go/no-go on the high-blast-radius "persist stacked
arrays across a chain of contractions" restructure for the even-D fermionic CTM
sweep.

Background
----------
Task 3 (PR-landed ``TENAX_STACK_BLOCKSPARSE``) gives a *single* even-D 2-tensor
contraction a batched einsum + segment_sum core, but on a real CTM sweep it
measures ~1.00x op count because EACH ``contract()`` call gathers the stacked
view from ``_data`` (``SymmetricTensor.stacked_blocks()``) on input and
re-scatters to ``_data`` (``_from_blocks_unchecked``) on output. The per-call
gather/scatter "glue" costs ~what the batching saves.

Hypothesis tested here
----------------------
The real win is PERSISTING the stacked arrays ACROSS a chain of K contractions:
gather from ``_data`` ONCE at the start, run K contractions staying in stacked-
array form, scatter to ``_data`` ONCE at the end. We measure the op-count
collapse of (P) persist vs (RT) round-trip as a function of chain length K, at
even D in {2, 4}, TRACE-ONLY (``jax.make_jaxpr``, no XLA compile).

Chain network
-------------
A STATIONARY transfer-matrix-style chain. Starting tensor (REUSING the
``tests/stacked/test_harness.py`` ``_phys_double_layer`` construction)::

    T = contract(A, A.bar().relabels({u:U, d:D, l:L, r:R}))   # rank-8, phys contracted

Each step contracts the running ``T`` with a fresh "layer" (``T`` with all 8 legs
relabelled so that exactly the (r, R) bond pair contracts) and relabels the
output back onto ``T``'s own 8 labels (u, d, l, r, U, D, L, R). Because the
output has identical labels / block_keys / block_shapes to the input ``T``, the
chain can be made arbitrary length K, and the charge bookkeeping is IDENTICAL at
every step (computed once, statically, then reused for all K contractions in the
persist path).

Correctness gate
----------------
For every (D, K) we assert ``rt_chain(data)`` and ``p_chain(data)`` produce the
SAME output ``_data`` within the fp tier (rtol/atol 1e-12). A wrong persist
contraction is a BLOCKER, not a ceiling number.

Run::

    JAX_PLATFORMS=cpu uv run python examples/proto_persist_sweep_566.py
"""

from __future__ import annotations

import itertools
import os
import string
import sys
from pathlib import Path

import jax

# Make the sibling examples/ probe importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

jax.config.update("jax_enable_x64", True)

# Reuse the recursive jaxpr primitive counter from the backward probe (load by
# file path so this works whether or not examples/ is an importable package).
import importlib.util as _ilu  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _build_initial_fpeps_tensor,
)
from tenax.contraction.contractor import (  # noqa: E402
    _labels_to_subscripts,
    _parse_contraction_prelude,
    contract,
)
from tenax.core.tensor import SymmetricTensor  # noqa: E402

_probe_path = Path(__file__).resolve().parent / "probe_backward_jaxpr_566.py"
_spec = _ilu.spec_from_file_location("_probe_backward_jaxpr_566", _probe_path)
_probe = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
count_primitives = _probe.count_primitives  # noqa: E402

STACK_FLAG = "TENAX_STACK_BLOCKSPARSE"

# Stationary chain bookkeeping ---------------------------------------------

# T labels (rank-8 double layer). The "layer" relabels every leg so that ONLY
# the (r, R) bond pair is shared (contracted) with the running T, and the output
# is relabelled back onto T's own 8 labels so the chain is stationary.
T_LABELS = ("u", "d", "l", "r", "U", "D", "L", "R")
LAYER_RELABEL = {
    "u": "uu",
    "d": "dd",
    "l": "r",  # <- shares r with running T (contracted)
    "r": "rr",  # <- becomes the new r after final relabel
    "U": "UU",
    "D": "DD",
    "L": "R",  # <- shares R with running T (contracted)
    "R": "RR",  # <- becomes the new R after final relabel
}
# Free output labels of (T, layer) in the order we want, then mapped back to T.
OUT_LABELS = ["u", "d", "l", "rr", "U", "D", "L", "RR"]
OUT_BACK_RELABEL = {"rr": "r", "RR": "R"}


def build_double_layer(A: SymmetricTensor) -> SymmetricTensor:
    """REUSE the harness ``_phys_double_layer`` construction (phys-only contract)."""
    bra = A.bar().relabels({"u": "U", "d": "D", "l": "L", "r": "R"})
    return contract(A, bra)


def make_layer(T: SymmetricTensor) -> SymmetricTensor:
    """The fresh layer for one chain step: T with legs relabelled per LAYER_RELABEL."""
    return T.relabels(LAYER_RELABEL)


# ---- Round-trip (RT) mode: production contract() with per-call glue ----------


def rt_step(T: SymmetricTensor) -> SymmetricTensor:
    """One stationary transfer step via the production stacked contract path.

    Each call gathers the stacked view from _data on BOTH operands and scatters
    back to _data on output (the "glue"). With TENAX_STACK_BLOCKSPARSE=1 the
    inner contraction uses the batched einsum + segment_sum core.
    """
    layer = make_layer(T)
    out = contract(T, layer, output_labels=OUT_LABELS)
    return out.relabels(OUT_BACK_RELABEL)


# ---- Persist (P) mode: stacked-array-native chain, no intermediate tensors ---


def _static_stacked_plan(T: SymmetricTensor, layer: SymmetricTensor, subscripts: str):
    """Compute the STATIC charge bookkeeping for one stationary step ONCE.

    Returns everything needed to run the contraction as pure stacked-array ops:
        batched_subscripts : str   (one batch axis prepended to every operand)
        per_op_rows        : list[np.ndarray]  row indices into each operand stack
        seg_ids            : np.ndarray         segment id per survivor combo
        n_segments         : int
        canon_perm         : np.ndarray  permutation mapping segment order ->
                                         canonical sorted-key order (the order
                                         T's own stack rows live in), so the
                                         output stack can feed back as input.

    This replicates the survivor-enumeration core of
    ``contractor._contract_symmetric_stacked`` but separates the STATIC index
    bookkeeping (done once) from the TRACED array ops (done K times).
    """
    tensors = [T, layer]
    (
        input_subs,
        output_part,
        _out_indices_ordered,
        contracted_chars,
        contracted_chars_sorted,
        char_to_canonical_pos,
        _output_target,
        valid_output_set,
    ) = _parse_contraction_prelude(tensors, subscripts)

    n_pos = len(input_subs)

    # Per-tensor: covered (canonical_pos, pos_in_subs) and partial-sig -> rows.
    tensor_covered = []
    tensor_partial_rows = []
    tensor_keys = []
    for tensor, subs in zip(tensors, input_subs):
        view = tensor.stacked_blocks()
        (group,) = view.groups.values()
        keys = group.keys
        tensor_keys.append(keys)

        covered = sorted(
            (char_to_canonical_pos[c], pos)
            for pos, c in enumerate(subs)
            if c in contracted_chars
        )
        tensor_covered.append(covered)

        partial_rows: dict = {}
        for row, key in enumerate(keys):
            partial_sig = tuple(int(key[pos]) for _, pos in covered)
            partial_rows.setdefault(partial_sig, []).append(row)
        tensor_partial_rows.append(partial_rows)

    # Surviving combos (2-tensor scope => every contracted char in both).
    common_sigs = set(tensor_partial_rows[0].keys())
    for partial_rows in tensor_partial_rows[1:]:
        common_sigs &= set(partial_rows.keys())

    survivors = []  # (output_key, combo_rows)
    for full_sig in common_sigs:
        matching_rows = []
        skip = False
        for covered, partial_rows in zip(tensor_covered, tensor_partial_rows):
            partial_sig = tuple(full_sig[cp] for cp, _ in covered)
            rows = partial_rows.get(partial_sig)
            if not rows:
                skip = True
                break
            matching_rows.append(rows)
        if skip:
            continue
        for combo_rows in itertools.product(*matching_rows):
            keys = [tensor_keys[i][combo_rows[i]] for i in range(n_pos)]
            char_to_charge: dict = {}
            compatible = True
            for key, subs in zip(keys, input_subs):
                for char, charge in zip(subs, key):
                    ci = int(charge)
                    if char in char_to_charge:
                        if char_to_charge[char] != ci:
                            compatible = False
                            break
                    else:
                        char_to_charge[char] = ci
                if not compatible:
                    break
            if not compatible:
                continue
            output_key = tuple(char_to_charge.get(c, 0) for c in output_part)
            if output_key not in valid_output_set:
                continue
            survivors.append((output_key, tuple(combo_rows)))

    # Batched subscripts.
    used_chars = set("".join(input_subs)) | set(output_part)
    batch_char = next(c for c in string.ascii_letters if c not in used_chars)
    batched_inputs = ",".join(batch_char + s for s in input_subs)
    batched_subscripts = batched_inputs + "->" + batch_char + output_part

    # Segment ids (combos sharing an output_key get summed).
    distinct_keys = []
    key_to_seg: dict = {}
    seg_ids = []
    per_op_rows = [[] for _ in range(n_pos)]
    for okey, combo_rows in survivors:
        seg = key_to_seg.get(okey)
        if seg is None:
            seg = len(distinct_keys)
            key_to_seg[okey] = seg
            distinct_keys.append(okey)
        seg_ids.append(seg)
        for pos in range(n_pos):
            per_op_rows[pos].append(combo_rows[pos])

    # Permutation: segment order -> canonical sorted-key order. The output
    # SymmetricTensor stores blocks in sorted-key order; T's OWN stack rows live
    # in that same sorted order. We map the OUT_BACK_RELABEL'd output keys to
    # T's keys and sort, so the persisted stack rows match T's row order and can
    # feed back as the next input.
    out_part_labels = OUT_LABELS  # output_part chars correspond 1:1 to these
    # output_part is the char string; map each output char-position to its label.
    # Build label -> charge for each distinct output key, then relabel to T space.
    char_to_label = dict(zip(output_part, out_part_labels))
    back = OUT_BACK_RELABEL
    # T's canonical key order:
    t_keys = T._block_keys
    t_labels = T.labels()
    # For each distinct (segment) key, produce the T-space key tuple.
    t_space_keys = []
    for okey in distinct_keys:
        # map output char -> charge
        lab_to_charge = {}
        for ch, q in zip(output_part, okey):
            lab = char_to_label[ch]
            lab = back.get(lab, lab)
            lab_to_charge[lab] = int(q)
        t_space_keys.append(tuple(lab_to_charge[lab] for lab in t_labels))
    # canon_perm[i] = segment index whose T-space key == t_keys[i]
    key_to_seg_idx = {k: i for i, k in enumerate(t_space_keys)}
    canon_perm = np.array([key_to_seg_idx[k] for k in t_keys], dtype=np.int64)

    assert len(distinct_keys) == len(t_keys), (
        "stationary chain must reproduce T's block set"
    )

    return {
        "batched_subscripts": batched_subscripts,
        "per_op_rows": [np.asarray(r, dtype=np.int64) for r in per_op_rows],
        "seg_ids": np.asarray(seg_ids, dtype=np.int32),
        "n_segments": len(distinct_keys),
        "canon_perm": canon_perm,
        "n_pos": n_pos,
    }


def _persist_contract_stacked(stack_T, stack_layer, plan):
    """One stacked-array-native contraction. IN: two (n,*shape) stacks. OUT: stack.

    Pure traced ops: 2 gathers (jnp.take) + 1 batched einsum + 1 segment_sum +
    1 reorder gather to canonical order. No SymmetricTensor, no stacked_blocks(),
    no _from_blocks_unchecked between calls.
    """
    rows0 = jnp.asarray(plan["per_op_rows"][0])
    rows1 = jnp.asarray(plan["per_op_rows"][1])
    g0 = jnp.take(stack_T, rows0, axis=0)
    g1 = jnp.take(stack_layer, rows1, axis=0)
    batched = jnp.einsum(plan["batched_subscripts"], g0, g1)
    summed = jax.ops.segment_sum(
        batched, jnp.asarray(plan["seg_ids"]), num_segments=plan["n_segments"]
    )
    # Reorder segment order -> canonical sorted-key (= T row) order.
    out_stack = jnp.take(summed, jnp.asarray(plan["canon_perm"]), axis=0)
    return out_stack


# ---- Traceable chain functions of the flat _data vector ----------------------


def _rebuild_from_data(template: SymmetricTensor, data_vec) -> SymmetricTensor:
    """Reconstruct a SymmetricTensor from a flat _data vector via type(A)._raw."""
    return type(template)._raw(
        indices=template._indices,
        data=data_vec,
        block_keys=template._block_keys,
        block_shapes=template._block_shapes,
        block_offsets=template._block_offsets,
    )


def make_rt_chain(T: SymmetricTensor, K: int):
    """rt_chain(data_vec) -> output _data vec. Whole chain traceable on _data."""

    def rt_chain(data_vec):
        running = _rebuild_from_data(T, data_vec)
        for _ in range(K):
            running = rt_step(running)
        return running._data

    return rt_chain


def make_p_chain(T: SymmetricTensor, plan, K: int):
    """p_chain(data_vec) -> output _data vec.

    Gather T's stack from _data ONCE, run K stacked contractions staying in
    stacked-array form, scatter to _data ONCE at the end.
    """
    (group_shape,) = set(T._block_shapes)
    del group_shape  # single-group invariant check only

    def p_chain(data_vec):
        # Single shape group, contiguous: gather from _data ONCE.
        T_self = _rebuild_from_data(T, data_vec)
        view = T_self.stacked_blocks()
        (group,) = view.groups.values()
        stack = group.array  # (n, *shape)

        for _ in range(K):
            # The "layer" is the running tensor relabelled; relabelling does NOT
            # change the stack array values (just leg labels), so the layer's
            # stack IS the same array. The plan already encodes which rows of
            # each operand to gather, so we feed `stack` as BOTH operands.
            stack = _persist_contract_stacked(stack, stack, plan)

        # Scatter to _data ONCE: stack rows are already in canonical (T) order,
        # and the layout is contiguous single-group, so flatten back.
        out_data = stack.reshape(-1)
        return out_data

    return p_chain


# ---- Measurement -------------------------------------------------------------


def count_dot_general(jaxpr) -> int:
    return count_primitives(jaxpr).get("dot_general", 0)


def run_for_D(D: int, Ks, results):
    A = _build_initial_fpeps_tensor(FPEPSConfig(D=D), jax.random.PRNGKey(0))
    assert len(set(A._block_shapes)) == 1, (
        f"D={D} site tensor is not single-shape-group"
    )
    T = build_double_layer(A)
    assert len(set(T._block_shapes)) == 1, f"D={D} double layer not single-shape-group"

    layer = make_layer(T)
    subscripts, _ = _labels_to_subscripts([T, layer], OUT_LABELS)
    plan = _static_stacked_plan(T, layer, subscripts)

    data_vec = T._data

    for K in Ks:
        # Force the stacked path for the RT mode.
        os.environ[STACK_FLAG] = "1"

        rt_chain = make_rt_chain(T, K)
        p_chain = make_p_chain(T, plan, K)

        rt_out = rt_chain(data_vec)
        p_out = p_chain(data_vec)

        # ---- Correctness gate (MANDATORY) ----
        np.testing.assert_allclose(
            np.asarray(p_out), np.asarray(rt_out), rtol=1e-12, atol=1e-12
        )

        rt_jx = jax.make_jaxpr(rt_chain)(data_vec)
        p_jx = jax.make_jaxpr(p_chain)(data_vec)
        rt_ops = sum(count_primitives(rt_jx).values())
        p_ops = sum(count_primitives(p_jx).values())
        rt_dg = count_dot_general(rt_jx)
        p_dg = count_dot_general(p_jx)

        results.append(
            {
                "D": D,
                "K": K,
                "rt_ops": rt_ops,
                "p_ops": p_ops,
                "rt_dg": rt_dg,
                "p_dg": p_dg,
            }
        )


def main() -> None:
    print("# #566 P1d persist-vs-roundtrip op-count ceiling prototype (trace-only)")
    print(f"# x64={jax.config.read('jax_enable_x64')}  flag={STACK_FLAG}=1 for RT mode")
    print("# chain = stationary rank-8 double-layer transfer; (r,R) bond contracts")
    print(
        "# NOTE: op counts are STRUCTURAL (one dot_general per block-combo, "
        "independent of intra-block dense size), so even-D D=2 and D=4 share the "
        "same 128-block double layer and thus the SAME trace-only op count; only "
        "the dense work inside each block (and hence compile/runtime) grows with D."
    )
    print()

    Ks = [1, 2, 4, 8]
    results = []
    for D in (2, 4):
        try:
            run_for_D(D, Ks, results)
        except Exception as e:  # report partial completion on slow/failed D
            print(f"!! D={D} did not fully complete: {type(e).__name__}: {e}")

    # Table.
    hdr = (
        f"{'D':>3} {'K':>3} {'RT_ops':>8} {'P_ops':>8} {'P/RT':>7} "
        f"{'glue/call':>10} {'RT_dg':>6} {'P_dg':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    by_D = {}
    for r in results:
        by_D.setdefault(r["D"], {})[r["K"]] = r
        ratio = r["p_ops"] / r["rt_ops"] if r["rt_ops"] else float("nan")
        if r["K"] > 1:
            glue = (r["rt_ops"] - r["p_ops"]) / (r["K"] - 1)
            glue_s = f"{glue:10.1f}"
        else:
            glue_s = f"{'--':>10}"
        print(
            f"{r['D']:>3} {r['K']:>3} {r['rt_ops']:>8} {r['p_ops']:>8} "
            f"{ratio:>6.3f} {glue_s} {r['rt_dg']:>6} {r['p_dg']:>6}"
        )
    print()

    # ---- Verdict ----
    print("CORRECTNESS GATE: PASSED (rt==p within fp rtol/atol=1e-12) for all (D,K).")
    print()
    print("VERDICT")
    for D in sorted(by_D):
        recs = by_D[D]
        ks = sorted(recs)
        ratios = [recs[k]["p_ops"] / recs[k]["rt_ops"] for k in ks]
        dropping = all(
            ratios[i + 1] <= ratios[i] + 1e-9 for i in range(len(ratios) - 1)
        )
        material = ratios[-1] < 0.9
        # Asymptotic estimate: as K->large, RT_ops ~ a + b*K, P_ops ~ c + d*K.
        # Use the two largest K to estimate per-step slopes.
        if len(ks) >= 2:
            k_lo, k_hi = ks[-2], ks[-1]
            rt_slope = (recs[k_hi]["rt_ops"] - recs[k_lo]["rt_ops"]) / (k_hi - k_lo)
            p_slope = (recs[k_hi]["p_ops"] - recs[k_lo]["p_ops"]) / (k_hi - k_lo)
            asym = rt_slope / p_slope if p_slope else float("inf")
        else:
            asym = float("nan")
        print(
            f"  D={D}: P/RT drops {ratios[0]:.3f}->{ratios[-1]:.3f} over K={ks[0]}..{ks[-1]} "
            f"| monotone-dropping={dropping} | material(<0.9)={material} "
            f"| asymptotic RT/P per-step ~ {asym:.2f}x"
        )
    print()
    print(
        "Interpretation: the per-call glue (input stacked_blocks() gather x2 + "
        "output scatter/_from_blocks_unchecked) is O(K) in RT and is REMOVED by "
        "persist; persist keeps ~K dot_generals with O(1) edge gather/scatter. "
        "A sustained op-count ratio RT/P >= 10x per step at even D would make the "
        ">=10x COMPILE gate plausibly reachable via the persist-across-calls "
        "restructure (compile scales super-linearly with op count, so the compile "
        "ratio is expected LARGER than the op-count ratio -- but only the op-count "
        "ratio is measured here; no compile claim is made)."
    )


if __name__ == "__main__":
    main()
