# U(1)-Sz Stacked Block-Sparse Feasibility Spike (#566) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide GO/NO-GO on improving end-to-end U(1)-Sz iPEPS-AD runtime by measuring where block-count cost concentrates, then prototyping the single highest-leverage lever (stacked-aware `fuse`) on the one chain the measurements identify.

**Architecture:** A measurement-first spike. Cheap static + compile measurements (S1 census, S2 profiler grid, S3 backward histogram) run before any code change and can each independently force a NO-GO. Only on GO do we build the contingent prototype: make `_fuse_indices_symmetric` stacked-aware so the persisting `StackedSymmetricTensor` chain survives through the CTM sweep, behind the existing `TENAX_STACK_BLOCKSPARSE` flag. Everything is gated against the stack-OFF U(1)-Sz golden for correctness.

**Tech Stack:** Python, JAX (x64), Tenax `SymmetricTensor`/`StackedView`/`StackedSymmetricTensor`, `ctm_tensor`, `make_ctm_energy_fn`, pytest. Reuses `examples/profile_ctm_ad_wall_566.py` and `examples/probe_backward_jaxpr_566.py`.

**Spec:** `docs/superpowers/specs/2026-06-15-u1sz-stacked-feasibility-spike-566-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `examples/census_u1sz_block_shapes_566.py` | Create | S1: static block-shape fragmentation census over CTM env tensors |
| `examples/profile_ctm_ad_wall_566.py` | Modify (`make_site_and_gate`, arg `--sym` choices) | S2: add `u1sz` arm so the off/on compile+runtime grid covers U(1)-Sz |
| `tests/test_profiler_u1sz_arm.py` | Create | Correctness gate for the `u1sz` profiler arm (site/gate are valid U(1)-Sz, energy finite) |
| `examples/probe_backward_jaxpr_566.py` | Modify (`--sym u1sz`) | S3: backward op-histogram for the U(1)-Sz fixed-point backward |
| `src/tenax/algorithms/_tensor_utils.py` | Modify (`_fuse_indices_symmetric`) | (GO only) stacked-aware fuse: consume `StackedView`, emit `StackedSymmetricTensor`, no `_data` round-trip |
| `tests/stacked/test_stacked_fuse.py` | Create (GO only) | round-trip bit-exactness + energy equality of stacked fuse vs per-block |
| `docs/superpowers/handoffs/2026-06-16-u1sz-stacked-spike-findings.md` | Create | The deliverable: census table, off/on grid, op histogram, GO/NO-GO verdict |

**Branch:** `spike/566-u1sz-stacked-feasibility` (already created, spec committed). Stay on it.

**Hardware note:** D=2 arms run on CPU for cheap iteration (`JAX_PLATFORMS=cpu`). D=3 compile arms are minutes-long — run on the A100 (`uv sync --extra cuda13`, see memory `a100-gpu-env`). Each task says which.

---

## Task 0: Prerequisites — working D=3 U(1)-Sz path + drift characterization

The spike depends on two facts being true *before* measuring: (1) #605/#608 actually makes a D=3 U(1)-Sz forward+grad run on this branch, and (2) the flagged ~4.6e-4 `TENAX_STACK_BLOCKSPARSE` energy drift is bounded so the GO gate's "energy unchanged" criterion is meaningful.

**Files:**
- Test: `tests/test_profiler_u1sz_arm.py` (created here, extended in Task 2)

- [ ] **Step 1: Write a smoke test that a D=3 U(1)-Sz CTM forward + one grad step runs**

```python
# tests/test_profiler_u1sz_arm.py
"""U(1)-Sz arm prerequisites for the #566 feasibility spike."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, heisenberg_gate_u1sz
from tenax.algorithms._ctm_tensor import ctm_tensor


@pytest.mark.parametrize("D,chi", [(2, 8), (3, 8)])
def test_u1sz_ctm_forward_runs(D, chi):
    """#605/#608: D>=3 U(1)-Sz CTM must not raise (the unfused-projector fix)."""
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _trunc = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    # Every env field is a finite SymmetricTensor.
    for name in env._fields:
        t = getattr(env, name)
        assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"
```

- [ ] **Step 2: Run the smoke test (D=2 must pass; D=3 confirms #605/#608)**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py::test_u1sz_ctm_forward_runs -v`
Expected: D=2 PASS. D=3 PASS confirms #608 fixed the T4 absorb. **If D=3 raises** (e.g. the #605 `ValueError`), record it, mark the D=3 case `xfail` with the exact error, and cap the spike at D=2 + D=3 χ=8 fallback per the spec's risk note — do not block the spike.

- [ ] **Step 3: Write a drift-characterization helper (energy stack-OFF vs stack-ON)**

```python
# tests/test_profiler_u1sz_arm.py  (append)
import os
from tenax import compute_energy_ctm_tensor  # re-exported from _ctm_tensor_energy


def _u1sz_energy(D, chi, stack: str):
    """CTM energy for a U(1)-Sz site with TENAX_STACK_BLOCKSPARSE = stack."""
    prev = os.environ.get("TENAX_STACK_BLOCKSPARSE")
    os.environ["TENAX_STACK_BLOCKSPARSE"] = stack
    try:
        jax.clear_caches()
        A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
        env, _ = ctm_tensor(A, chi=chi, max_iter=8, conv_tol=1e-6)
        gate = heisenberg_gate_u1sz()
        return float(compute_energy_ctm_tensor(A, env, gate))  # add d=2 if required

    finally:
        if prev is None:
            os.environ.pop("TENAX_STACK_BLOCKSPARSE", None)
        else:
            os.environ["TENAX_STACK_BLOCKSPARSE"] = prev


def test_stack_flag_energy_drift_is_bounded():
    """Quantify the flagged ~4.6e-4 stacked-core drift on the U(1)-Sz path.

    This is a CHARACTERIZATION, not a pass/fail spec assertion: it records the
    drift so the GO gate's 'energy unchanged' criterion is interpretable.
    """
    e_off = _u1sz_energy(D=2, chi=8, stack="0")
    e_on = _u1sz_energy(D=2, chi=8, stack="1")
    drift = abs(e_on - e_off)
    print(f"\nU1Sz D=2 chi=8 energy drift |on-off| = {drift:.3e} "
          f"(off={e_off:.8f}, on={e_on:.8f})")
    # Loose guard: if this blows up past 1e-2 the stacked core is broken for
    # U(1)-Sz and the off/on comparison is meaningless -> investigate before S2.
    assert drift < 1e-2, f"stacked drift {drift:.3e} too large to trust off/on grid"
```

- [ ] **Step 4: Run the drift characterization and record the number**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py::test_stack_flag_energy_drift_is_bounded -v -s`
Expected: prints a drift value. Record it in the handoff (Task 6). If drift ≳ 1e-3, the GO gate's energy-equality tolerance must be set to this drift (not tighter), and that is noted as a caveat — the stacked path is not bit-faithful for U(1)-Sz.

- [ ] **Step 5: Commit**

```bash
git add tests/test_profiler_u1sz_arm.py
git commit -m "test(#566): U(1)-Sz spike prereqs — D=3 CTM smoke + stack-flag drift characterization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: S1 — static block-shape fragmentation census (the early kill switch)

If U(1)-Sz CTM tensors fragment into all-distinct shapes, the stacked grouping collapse is ~1× and stacking cannot help — NO-GO before any compile is spent. This task answers that statically.

**Files:**
- Create: `examples/census_u1sz_block_shapes_566.py`

- [ ] **Step 1: Write the census script**

```python
# examples/census_u1sz_block_shapes_566.py
"""S1 of the #566 U(1)-Sz spike: block-shape fragmentation census.

Grouping collapse ceiling = n_blocks / n_distinct_shapes per tensor. Even-D
FermionParity ~ 16 (all blocks one shape); general U(1) is the open question.
Pure static metadata (block keys/shapes) over real CTM env tensors. No grad,
short CTM, cheap. Run on CPU.

    JAX_PLATFORMS=cpu uv run python examples/census_u1sz_block_shapes_566.py \\
        --D 2 3 --chi-factor 4 --json census_u1sz.json
"""
import argparse
import json

import jax

from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
from tenax.algorithms._ctm_tensor import ctm_tensor


def census_one(D: int, chi: int) -> dict:
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    rows = []
    targets = [("site", A)] + [(n, getattr(env, n)) for n in env._fields]
    for name, t in targets:
        shapes = list(getattr(t, "_block_shapes", ()))
        n_blocks = len(shapes)
        n_shapes = len(set(shapes))
        collapse = (n_blocks / n_shapes) if n_shapes else 0.0
        rows.append({
            "tensor": name, "n_blocks": n_blocks,
            "n_distinct_shapes": n_shapes, "collapse_ceiling": round(collapse, 3),
        })
    return {"D": D, "chi": chi, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--chi-factor", type=int, default=4)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    out = []
    for D in args.D:
        res = census_one(D, args.chi_factor * D)
        out.append(res)
        print(f"\n=== U(1)-Sz census D={D} chi={args.chi_factor * D} ===")
        print(f"{'tensor':>6} {'n_blocks':>9} {'n_shapes':>9} {'collapse':>9}")
        for r in res["rows"]:
            print(f"{r['tensor']:>6} {r['n_blocks']:>9} "
                  f"{r['n_distinct_shapes']:>9} {r['collapse_ceiling']:>9}")
        ceilings = [r["collapse_ceiling"] for r in res["rows"]]
        print(f"  median collapse ceiling = {sorted(ceilings)[len(ceilings)//2]:.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the census at D=2 and D=3**

Run: `JAX_PLATFORMS=cpu uv run python examples/census_u1sz_block_shapes_566.py --D 2 3 --chi-factor 4 --json census_u1sz.json`
Expected: a per-tensor table + median collapse ceiling for each D. **Interpretation gate:** median collapse ceiling ≥ ~3 across the hot tensors (corners/edges) → stacking has headroom, proceed. ≈ 1 → record NO-GO-leaning evidence; the prototype must become the pad-to-max fallback probe, or the spike stops (decided at Task 4).

- [ ] **Step 3: Commit (script + the JSON artifact)**

```bash
git add examples/census_u1sz_block_shapes_566.py census_u1sz.json
git commit -m "bench(#566): S1 U(1)-Sz block-shape fragmentation census

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: S2 — add the `u1sz` arm to the profiler and run the off/on grid

The #566 profiler has `fermionic`/`dense` arms only. Add `u1sz` so we get `{dense,u1sz}×{stack off,on}` for `fwd_cmp/vg_cmp/bwd_cmp` + warm `step_s` + persist hit-rate.

**Files:**
- Modify: `examples/profile_ctm_ad_wall_566.py:161` (`make_site_and_gate`) and the `--sym` arg (`:327`)
- Test: `tests/test_profiler_u1sz_arm.py` (append)

- [ ] **Step 1: Write a failing test that the profiler exposes a valid `u1sz` arm**

```python
# tests/test_profiler_u1sz_arm.py  (append)
def test_profiler_u1sz_arm_builds_symmetric_site_and_gate():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "profile_ctm_ad_wall_566",
        pathlib.Path(__file__).parent.parent / "examples" / "profile_ctm_ad_wall_566.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    site, gate = mod.make_site_and_gate("u1sz", D=2, seed=0)
    from tenax.core.tensor import SymmetricTensor
    assert isinstance(site, SymmetricTensor)          # block-sparse, not dense
    assert len(site._block_keys) > 1                  # genuinely multi-block
    assert isinstance(gate, SymmetricTensor)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py::test_profiler_u1sz_arm_builds_symmetric_site_and_gate -v`
Expected: FAIL with `ValueError: unknown sym 'u1sz'`.

- [ ] **Step 3: Add the `u1sz` arm to `make_site_and_gate`**

In `examples/profile_ctm_ad_wall_566.py`, inside `make_site_and_gate` (before the final `raise ValueError`), add:

```python
    if sym == "u1sz":
        from tenax.algorithms.ipeps import (
            heisenberg_u1sz_init_pair,
            heisenberg_gate_u1sz,
        )
        A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))
        return A, heisenberg_gate_u1sz()
```

And widen the CLI choices — change the `--sym` argument default/help (`:327`) so `u1sz` is accepted (it uses `nargs="+"` with no `choices=`, so no change is strictly needed, but update the help string):

```python
    ap.add_argument("--sym", nargs="+", default=["fermionic", "dense"],
                    help="arms: fermionic | dense | u1sz")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit the arm**

```bash
git add examples/profile_ctm_ad_wall_566.py tests/test_profiler_u1sz_arm.py
git commit -m "bench(#566): add u1sz arm to the CTM-AD compile-wall profiler

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Run the D=2 off/on grid on CPU (cheap, must complete)**

Run both, capturing JSON:
```bash
JAX_PLATFORMS=cpu TENAX_STACK_BLOCKSPARSE=0 uv run python examples/profile_ctm_ad_wall_566.py \
    --D 2 --chi-factor 4 --depth 8 --sym u1sz dense --reps 3 --json profile_u1sz_d2_stackoff.json
JAX_PLATFORMS=cpu TENAX_STACK_BLOCKSPARSE=1 uv run python examples/profile_ctm_ad_wall_566.py \
    --D 2 --chi-factor 4 --depth 8 --sym u1sz dense --reps 3 --json profile_u1sz_d2_stackon.json
```
Expected: four `(fwd_cmp, vg_cmp, bwd_cmp, step_s)` rows. Headline = does `u1sz` `vg_cmp` drop from stackoff→stackon, and how does `u1sz` compare to `dense` at matched D/χ.

- [ ] **Step 7: Run the D=3 grid on the A100 (the real target)**

On the A100 (per memory `a100-gpu-env`):
```bash
TENAX_STACK_BLOCKSPARSE=0 uv run python examples/profile_ctm_ad_wall_566.py \
    --D 3 --chi-factor 4 --depth 8 --sym u1sz dense --reps 3 --json profile_u1sz_d3_stackoff.json
TENAX_STACK_BLOCKSPARSE=1 uv run python examples/profile_ctm_ad_wall_566.py \
    --D 3 --chi-factor 4 --depth 8 --sym u1sz dense --reps 3 --json profile_u1sz_d3_stackon.json
```
Expected: minutes-long compiles. If D=3 χ=12 raises the #605 error despite #608, fall back to `--chi-factor 3` (χ=9) or the D=3 χ=8 case from Task 0 and label it. Record all four numbers.

- [ ] **Step 8: Capture the persist hit-rate (stackon, both D)**

The profiler imports `tenax.contraction.contractor`. After a stackon run, the `_STACK_PERSIST` dict holds `{calls, fully_persisted, persisted_inputs, gathered_inputs}`. Add a `--dump-persist` print at the end of the profiler's `main()` (one line), or run a short inline snippet:
```python
from tenax.contraction.contractor import _STACK_PERSIST
print("PERSIST", dict(_STACK_PERSIST))
```
Record the U(1)-Sz hit-rate (even-D fermionic baseline was 27%). Commit the JSON artifacts:
```bash
git add profile_u1sz_*.json
git commit -m "bench(#566): S2 U(1)-Sz off/on compile+runtime grid (D=2 CPU, D=3 A100)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: S3 — backward op-histogram localization for U(1)-Sz

Name the single dominant per-block-emitting chain in the U(1)-Sz fixed-point backward, and confirm whether `fuse`/`bar`/`_get_block` is what breaks the persist chain (the even-D finding).

**Files:**
- Modify: `examples/probe_backward_jaxpr_566.py` (add a `u1sz` site source)

- [ ] **Step 1: Point the probe at the `u1sz` arm**

`examples/probe_backward_jaxpr_566.py` builds a site/gate then traces the backward jaxpr and histograms primitives. Make it reuse the profiler's arm so there is one source of truth. Near its imports add:

```python
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "profile_ctm_ad_wall_566",
    pathlib.Path(__file__).parent / "profile_ctm_ad_wall_566.py")
_prof = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_prof)
```
and route its site/gate construction through `_prof.make_site_and_gate(args.sym, args.D, seed=0)`, adding a `--sym` arg (default `u1sz`) if the probe does not already have one.

- [ ] **Step 2: Run the probe (trace-only, no XLA — fast, CPU)**

Run: `JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py --sym u1sz --D 2 --chi 8 --depth 8`
Expected: an op histogram of the backward jaxpr. Identify (a) the top primitives by count, (b) the count of `dynamic_slice`/`reshape`/`gather` attributable to `_get_block`/`fuse` materialization, (c) whether the dominant chain passes through `_fuse_indices_symmetric`.

- [ ] **Step 3: Record the localization and commit**

Write the top-10 op table into the handoff (Task 6). Commit the probe change:
```bash
git add examples/probe_backward_jaxpr_566.py
git commit -m "diag(#566): S3 U(1)-Sz backward op-histogram via shared profiler arm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: GO/NO-GO decision gate (analysis checkpoint — no code)

- [ ] **Step 1: Apply the decision rule from S1–S3**

Decide, in writing, based on the three measurements:
- **S1** median collapse ceiling on hot tensors: ≥3 → stacking has headroom; ≈1 → pad-to-max fallback or stop.
- **S2** does `u1sz vg_cmp` already improve stackoff→stackon, and is the gap to `dense` large enough to be worth attacking? Persist hit-rate < ~50% confirms the chain is broken (expected).
- **S3** is `_fuse_indices_symmetric` the dominant chain-breaker (the even-D hypothesis), or is the cost diffuse?

**Branch:**
- **fuse is the dominant breaker AND S1 has headroom → GO with Task 5 (stacked-aware fuse).**
- **S1 ≈ 1× collapse → GO only with the pad-to-max probe variant of Task 5 (Step note below), else NO-GO.**
- **cost is diffuse across many ops → NO-GO; the lever is the broad sweep-restructure, out of spike scope.**

- [ ] **Step 2: Record the verdict in the handoff before proceeding.** If NO-GO, skip to Task 6.

---

## Task 5: (GO only) stacked-aware `fuse` prototype on the hottest chain

Make `_fuse_indices_symmetric` consume an already-stacked input and return a persisting `StackedSymmetricTensor` without a `_data` round-trip, so the chain S3 identified survives. Scope strictly to fuse (and `bar` only if S3 flagged it).

**Files:**
- Modify: `src/tenax/algorithms/_tensor_utils.py` (`_fuse_indices_symmetric`, the block-reassembly section after `:349`)
- Test: `tests/stacked/test_stacked_fuse.py` (create)

- [ ] **Step 1: Write the failing correctness test (bit-exact round-trip + energy equality)**

```python
# tests/stacked/test_stacked_fuse.py
"""Stacked-aware fuse must match the per-block fuse bit-for-bit (#566 Task 5b)."""
import os
import jax
import numpy as np
import pytest

from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
from tenax.algorithms._tensor_utils import fuse_indices
from tenax.core.tensor import FlowDirection


def _fuse_first_two(A):
    return fuse_indices(A, 0, 1, "f", FlowDirection.OUT)


def test_stacked_fuse_matches_per_block():
    A, _ = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))

    os.environ["TENAX_STACK_BLOCKSPARSE"] = "0"
    ref = _fuse_first_two(A)

    os.environ["TENAX_STACK_BLOCKSPARSE"] = "1"
    got = _fuse_first_two(A)
    os.environ.pop("TENAX_STACK_BLOCKSPARSE", None)

    # Materialized buffers must be bit-identical (canonical block order).
    assert np.array_equal(np.asarray(ref._data), np.asarray(got._data)), (
        f"max|d|={np.max(np.abs(np.asarray(ref._data)-np.asarray(got._data))):.2e}")
    assert ref._block_keys == got._block_keys
    assert ref._block_shapes == got._block_shapes
```

- [ ] **Step 2: Run it to verify it fails (or trivially passes by materializing)**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/stacked/test_stacked_fuse.py -v`
Expected: PASS today only because stack-on currently materializes `_data` (no stacked path in fuse yet). This test is the *correctness guard* the implementation must keep green; the *win* is measured in Step 5, not asserted here. (If it FAILS today, the stacked core already has a fuse drift — stop and fix that first.)

- [ ] **Step 3: Implement the stacked-aware path in `_fuse_indices_symmetric`**

Guard the new path on the flag and on the input already carrying a cached `StackedView`; otherwise fall through to the existing per-block reassembly unchanged. Mirror the stacked contractor (`src/tenax/contraction/contractor.py:498-588`): read `T.stacked_blocks()` groups, apply the static `scatter_map`/`fused_dim` (already computed at `:305-341`) as a batched gather/scatter over the leading block axis per shape-group, and return `StackedSymmetricTensor.from_stacked(view=..., indices=new_indices, block_keys=..., block_shapes=..., block_offsets=...)`. Sketch of the seam to add right before the existing per-block reassembly at `:349`:

```python
    from tenax.contraction.blocksparse_backend import _backend_opt_in
    from tenax.core.stacked_tensor import StackedSymmetricTensor
    if _backend_opt_in() and isinstance(T, StackedSymmetricTensor):
        out_view = _fuse_stacked(  # new local helper: batched scatter over groups
            T.stacked_blocks(), scatter_map, fused_dim, fused_groups,
            new_block_keys, new_block_shapes,
        )
        return StackedSymmetricTensor.from_stacked(
            view=out_view, indices=new_indices,
            block_keys=new_block_keys, block_shapes=new_block_shapes,
            block_offsets=new_block_offsets,
            total_size=new_total_size, dtype=T._data.dtype if hasattr(T, "_data") else out_view_dtype,
        )
    # ---- existing per-block reassembly (unchanged) ----
```

Implement `_fuse_stacked` and the `new_block_*` metadata: `new_block_keys`, `new_block_shapes`, `new_block_offsets`, and `new_total_size` (= last offset + last block size). The existing per-block path already computes the fused block keys/shapes/offsets further down — factor those static computations above the branch so both paths share them. `from_stacked` requires `total_size` and `dtype` (verified signature); pass the fused total size and the input dtype. Keep the scatter index math identical so the round-trip stays bit-exact. If the input is not a `StackedSymmetricTensor` (no cached view), the guard is false and behavior is byte-identical to today — note that `StackedSymmetricTensor` is only produced upstream when `_backend_opt_in()` is true, so the `isinstance` check both gates the flag and confirms a cache exists.

- [ ] **Step 4: Run the correctness test (must stay PASS, now exercising the stacked branch)**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/stacked/test_stacked_fuse.py tests/stacked/ -v`
Expected: PASS, and the whole `tests/stacked/` suite stays green. Then re-run the energy goldens: `JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py -v`. Energy drift must not exceed the Task 0 baseline.

- [ ] **Step 5: Re-measure the off/on grid with the stacked fuse in place**

Re-run Task 2 Step 6 (D=2 CPU) and Step 7 (D=3 A100) `TENAX_STACK_BLOCKSPARSE=1` arms into `*_stackon_fuse.json`, plus the persist hit-rate (Task 2 Step 8). **GO gate: `u1sz vg_cmp` at D=3 improves ≥1.5× vs the Task 2 stackoff baseline, energy within the Task 0 drift, warm `step_s` not regressed.** Record the before/after.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_tensor_utils.py tests/stacked/test_stacked_fuse.py profile_u1sz_*_fuse.json
git commit -m "perf(#566): stacked-aware fuse (Task 5b) for the U(1)-Sz CTM persist chain

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> **Pad-to-max variant (only if Task 4 routed here):** instead of grouping by exact shape, pad each tensor's blocks to the per-tensor max block shape so a single stacked group forms, and measure whether the collapse win exceeds the padding-compute waste on the hottest tensor only. Same correctness test (the materialized `_data` must still round-trip bit-exactly after un-padding). This is a scoped probe of P1e, not its delivery — record the padding overhead alongside the win.

---

## Task 6: Handoff writeup (the deliverable)

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-16-u1sz-stacked-spike-findings.md`

- [ ] **Step 1: Write the findings doc**

Include, with the actual recorded numbers: (1) Task 0 D=3-runs verdict + measured energy drift; (2) S1 census table + median collapse ceilings; (3) S2 off/on 2×2 grid (`fwd/vg/bwd_cmp`, `step_s`) at D=2 and D=3 + persist hit-rates; (4) S3 top-10 backward op histogram; (5) the Task 4 verdict; (6) if GO, the Task 5 before/after and whether the ≥1.5× gate cleared; (7) explicit next step (fund the broad sweep-restructure, or the documented wall that stops it). Label everything "post-#605 unfused-projector representation."

- [ ] **Step 2: Update memory**

Append a one-line pointer to `MEMORY.md` and write a `project`-type memory capturing the spike verdict (GO/NO-GO + the decisive number), linking `[[u1sz-perf-study-d3-findings]]` and `[[570-u1sz-blocked-core-bug]]`.

- [ ] **Step 3: Commit and open the PR**

```bash
git add docs/superpowers/handoffs/2026-06-16-u1sz-stacked-spike-findings.md
git commit -m "docs(#566): U(1)-Sz stacked feasibility spike — findings + GO/NO-GO verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -u origin spike/566-u1sz-stacked-feasibility
gh pr create --base main --title "study(#566): U(1)-Sz stacked block-sparse feasibility spike" --body "..."
```
Per CLAUDE.md: open a PR (do not push to `main`); merge with `gh pr merge --squash --delete-branch --auto`. Note the spike branch is based off `fix/605-u1sz-unfused-ctm` (#608) — if #608 has not merged, the PR base may need to wait for it or be rebased onto `main` once #608 lands.

---

## Self-Review Notes

- **Spec coverage:** S1 census → Task 1; S2 grid+hit-rate → Task 2; S3 localization → Task 3; drift characterization → Task 0; #605/#608 dependency → Task 0; default prototype (stacked fuse) → Task 5; pad-to-max fallback → Task 5 variant; GO gate (≥1.5×) → Task 4 + Task 5 Step 5; correctness contract (stack-OFF golden, tiered) → Task 0/Task 5; deliverable handoff → Task 6. All spec sections map to a task.
- **Contingency is explicit, not vague:** Tasks 0–4 are fully concrete and run regardless; Task 5's exact target is chosen at the Task 4 gate from a named menu (stacked fuse | pad-to-max | NO-GO), each with concrete entry points and the same correctness test — no "implement later."
- **Type/name consistency:** `make_site_and_gate(sym, D, seed)`, `heisenberg_u1sz_init_pair(D, key)`, `heisenberg_gate_u1sz()`, `ctm_tensor(A, chi, max_iter, conv_tol)`, `CTMTensorEnv._fields`, `_STACK_PERSIST`, `_fuse_indices_symmetric`, `StackedSymmetricTensor.from_stacked` used consistently across tasks.
