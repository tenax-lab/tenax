# U(1)-Sz CTM env de-fragmentation spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a representation-level change that reduces the distinct charge-sector count on the U(1)-Sz CTM environment tensors materially shrinks the backward charge-mask op cluster and buys D=3 χ=12 iPEPS-AD runtime to ≥ parity-with-dense within 1% energy — and emit a documented GO/NO-GO finding.

**Architecture:** A three-gate measurement staircase, cheapest-decisive-first (static sector census → trace-only backward re-profile → A100 end-to-end + energy). Each gate can kill the spike before the next is funded. The representation prototype is a **branch-local monkeypatch over `tenax.algorithms._ctm_utils._derive_charges`** (the per-sector χ-allocation knob, re-imported inside `_retruncate_by_base_charges` so the patch takes effect) — **no committed `src/` change**.

**Tech Stack:** Python, JAX (x64), pytest, the Tenax block-sparse `SymmetricTensor` CTM-AD path. Reuses the #609 spike assets (`examples/census_u1sz_block_shapes_566.py`, `examples/probe_backward_jaxpr_566.py`, `examples/profile_ctm_ad_wall_566.py`, `tests/test_profiler_u1sz_arm.py`).

**Research-spike note (read before executing):** This is a measurement spike, not a feature build. The deterministic tooling (static census predictor, faithfulness guard) gets full TDD. The prototype *mechanism* (the exact `_derive_charges` filter) may need one or two iterations to converge a valid env — the plan makes the **acceptance criteria** crisp (faithfulness guard passes; sector count actually drops) even where the mechanism is tuned. **Honor the gates: if a gate fails, stop at that task, record the finding, and skip the remaining gated tasks.** No `src/` file is committed at any point.

**Spec:** `docs/superpowers/specs/2026-06-16-u1sz-env-defrag-design.md`

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `examples/census_u1sz_block_shapes_566.py` | Modify | Add a **static candidate-evaluation** function: filter baseline env block-keys by a candidate keep-set and report predicted post-candidate sector count (Gate A). Keep the existing shape census. |
| `tests/test_u1sz_defrag_census_610.py` | Create | Unit test for the static candidate predictor (deterministic, synthetic block-keys). |
| `examples/u1sz_defrag_prototype_610.py` | Create | The throwaway C-lever prototype: a context manager that monkeypatches `_derive_charges` to drop \|Sz\|>1 / enforce uniform per-sector χ. Branch-local; never imported by `src/`. |
| `tests/test_u1sz_defrag_prototype_610.py` | Create | Faithfulness guard: under the prototype, CTM converges and energy is finite/sane at D=3 χ=12. |
| `examples/probe_backward_jaxpr_566.py` | Use (no edit if `--sym u1sz` already wired) | Gate B: backward op-histogram, baseline vs under-prototype. |
| `examples/profile_ctm_ad_wall_566.py` | Use | Gate C: off/on compile+runtime grid on A100. |
| `docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md` | Create (last task) | The GO/NO-GO finding writeup. |

---

## Task 1: Baseline lock (Stage 0)

Record the fragmented baseline at D=3 χ=12: env sector/shape census and the unconstrained-truncation energy `E_frag`. No code change — run existing tools and capture numbers.

**Files:**
- Use: `examples/census_u1sz_block_shapes_566.py`
- Use: `tests/test_profiler_u1sz_arm.py` (`_u1sz_energy` helper pattern)

- [ ] **Step 1: Confirm D=3 χ=12 U(1)-Sz CTM-AD runs on this branch**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py::test_u1sz_ctm_forward_runs -v
```
Expected: PASS for `[2-8]` and `[3-8]` (post-#605/#608 unfused-projector fix). If it fails, STOP — the prerequisite is broken; record and report.

- [ ] **Step 2: Capture the baseline shape/sector census at D=3 χ=12**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/census_u1sz_block_shapes_566.py \
    --D 3 --chi-factor 4 --json census_u1sz_baseline_610.json
```
Expected: prints per-tensor `n_blocks`, `n_distinct_shapes`, `collapse_ceiling`; writes JSON. Record the env (C1–C4, T1–T4) `n_blocks` — this is the baseline **sector count** the candidate must reduce.

- [ ] **Step 3: Capture `E_frag` (unconstrained-truncation energy) at D=3 χ=12**

Add a tiny throwaway script `examples/_e_frag_610.py` (kept on the branch, never committed to `src/`):
```python
# examples/_e_frag_610.py — record the unconstrained-truncation baseline energy.
import jax
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

jax.config.update("jax_enable_x64", True)
A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
gate = heisenberg_gate_u1sz()
print("E_frag(D=3,chi=12) =", float(compute_energy_ctm_tensor(A, env, gate)))
```
Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/_e_frag_610.py
```
Expected: a finite negative energy near the D=3 Heisenberg estimate. Record `E_frag`.

- [ ] **Step 4: Commit the baseline artifacts**

```bash
git add examples/_e_frag_610.py census_u1sz_baseline_610.json
git commit -m "study(#610): lock D=3 chi=12 fragmented baseline (census + E_frag)"
```

---

## Task 2: Gate A — static sector-count prediction under candidates (Stage 1)

Predict, **with no compile**, the post-candidate env sector count by filtering the baseline env tensors' block-keys. This is the cheapest decisive kill: if no candidate cuts the sector count ≥2× on paper, NO-GO before building anything.

**Files:**
- Modify: `examples/census_u1sz_block_shapes_566.py`
- Test: `tests/test_u1sz_defrag_census_610.py`

- [ ] **Step 1: Write the failing test for the static predictor**

Create `tests/test_u1sz_defrag_census_610.py`:
```python
"""Static candidate-evaluation predictor for #610 Gate A."""
from examples.census_u1sz_block_shapes_566 import predict_sectors_under_keep


def test_drop_high_sz_sectors_reduces_block_count():
    # block-keys: each key is a per-axis charge tuple; axes (chi_a, chi_b, d2).
    # Keep-set {-1,0,1} on the chi axes (axes 0 and 1) drops any block whose
    # chi charge has |q| > 1.
    block_keys = [
        (0, 0, 0), (1, -1, 0), (-1, 1, 0),   # all chi charges within {-1,0,1}
        (2, 0, -2), (-2, 0, 2), (0, 2, -2),  # contain a chi charge with |q|=2
    ]
    kept = predict_sectors_under_keep(block_keys, chi_axes=(0, 1), keep={-1, 0, 1})
    assert kept == 3  # only the first three survive


def test_keep_all_is_identity():
    block_keys = [(0, 0, 0), (2, 0, -2)]
    kept = predict_sectors_under_keep(block_keys, chi_axes=(0, 1), keep={-2, -1, 0, 1, 2})
    assert kept == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_defrag_census_610.py -v
```
Expected: FAIL with `ImportError: cannot import name 'predict_sectors_under_keep'`.

- [ ] **Step 3: Implement the predictor + candidate evaluation in the census**

Append to `examples/census_u1sz_block_shapes_566.py` (after `census_one`):
```python
def predict_sectors_under_keep(block_keys, chi_axes, keep) -> int:
    """Count blocks that survive restricting the chi-axis charges to ``keep``.

    Static Gate-A proxy for the post-candidate sector count: a block survives
    iff every chi-axis charge component lies in ``keep`` (the sector-dropping
    candidate, e.g. keep={-1,0,1} drops |Sz|=2 chi sectors).
    """
    keep = set(keep)
    return sum(
        all(k[a] in keep for a in chi_axes)
        for k in block_keys
    )


def _chi_axes_for(tensor) -> tuple:
    """Indices of legs whose label starts with 'chi' (the truncated bonds)."""
    labels = [ix.label for ix in tensor.indices]
    return tuple(i for i, lab in enumerate(labels) if lab.lower().startswith("chi"))


def candidate_report(D: int, chi: int, keep) -> dict:
    """For each env tensor, baseline n_blocks vs predicted-kept under ``keep``."""
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    rows = []
    for name in env._fields:
        t = getattr(env, name)
        keys = list(getattr(t, "_block_keys", ()))
        chi_axes = _chi_axes_for(t)
        n0 = len(keys)
        n1 = predict_sectors_under_keep(keys, chi_axes, keep) if chi_axes else n0
        rows.append({
            "tensor": name, "n_blocks": n0, "kept": n1,
            "reduction": round(n0 / n1, 3) if n1 else float("inf"),
        })
    return {"D": D, "chi": chi, "keep": sorted(keep), "rows": rows}
```
Add a `--candidate-keep` CLI branch in `main()` that, when passed (e.g. `-1 0 1`), prints `candidate_report` for each D and the **median reduction** across env tensors.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_defrag_census_610.py -v
```
Expected: PASS (both tests).

- [ ] **Step 5: Run the Gate-A measurement at D=3 χ=12**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/census_u1sz_block_shapes_566.py \
    --D 3 --chi-factor 4 --candidate-keep -1 0 1 \
    --json census_u1sz_candidateC_610.json
```
Record the median env-tensor `reduction`. Also note: the brief flags that **B (alt virtual-leg charge set) is cramped at D=3** — if you want B in the report, additionally evaluate `keep={-1,0,1}` is the same as C here; B's distinct lever (changing the *init* charge multiset) has no sub-`{0,+1,-1}` option at D=3, so record "B: no headroom at D=3" rather than running it.

- [ ] **Step 6: GATE A decision**

**Pass criterion:** median env-tensor sector-count `reduction` ≥ **2×** under candidate C.
- If **≥2×** → record PASS, proceed to Task 3.
- If **<2×** → record **NO-GO** in a note, jump to Task 6 (findings), and skip Tasks 3–5.

- [ ] **Step 7: Commit**

```bash
git add examples/census_u1sz_block_shapes_566.py tests/test_u1sz_defrag_census_610.py census_u1sz_candidateC_610.json
git commit -m "study(#610): Gate A — static sector-count predictor + D=3 candidate-C census"
```

---

## Task 3: Build the C-lever prototype + faithfulness guard (Stage 2 setup)

Only if Gate A passed. Build the throwaway prototype that produces a uniform/sector-dropped env, and prove it produces a *valid* env (converges, energy finite) before profiling.

**Files:**
- Create: `examples/u1sz_defrag_prototype_610.py`
- Test: `tests/test_u1sz_defrag_prototype_610.py`

- [ ] **Step 1: Write the failing faithfulness-guard test**

Create `tests/test_u1sz_defrag_prototype_610.py`:
```python
"""Faithfulness guard for the #610 C-lever prototype (Stage 2 prereq)."""
import jax
import numpy as np

from examples.u1sz_defrag_prototype_610 import sector_dropping_derive_charges
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair


def test_prototype_ctm_converges_and_energy_is_sane():
    jax.config.update("jax_enable_x64", True)
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_derive_charges(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
        for name in env._fields:
            t = getattr(env, name)
            assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"
        e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e), "prototype energy non-finite"
    assert -2.0 < e < 0.0, f"prototype energy {e} outside sane Heisenberg window"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_defrag_prototype_610.py -v
```
Expected: FAIL with `ImportError` (module/function not yet defined).

- [ ] **Step 3: Implement the prototype monkeypatch**

Create `examples/u1sz_defrag_prototype_610.py`:
```python
"""Throwaway #610 C-lever prototype: sector-dropping chi-bond truncation.

Monkeypatches the per-sector chi allocation knob
``tenax.algorithms._ctm_utils._derive_charges`` so chi_new charges with
|Sz| outside ``keep`` are removed before tiling. NEVER imported by src/.
This exists only to produce a uniform/sector-dropped env to profile.
"""
import contextlib

import numpy as np

import tenax.algorithms._ctm_utils as _cu

_orig_derive = _cu._derive_charges


@contextlib.contextmanager
def sector_dropping_derive_charges(keep=frozenset({-1, 0, 1})):
    keep = set(keep)

    def patched(base_charges, target_dim):
        base = np.asarray(base_charges, dtype=np.int32)
        filt = np.array([q for q in base if int(q) in keep], dtype=np.int32)
        if filt.size == 0:           # degenerate guard: never empty the bond
            filt = base
        return _orig_derive(filt, target_dim)

    _cu._derive_charges = patched
    try:
        yield
    finally:
        _cu._derive_charges = _orig_derive
```

- [ ] **Step 4: Run the faithfulness guard to verify it passes**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_defrag_prototype_610.py -v
```
Expected: PASS. If it FAILS (env non-finite, energy out of window, or `_derive_charges` not the binding knob), **iterate the mechanism**: confirm `_retruncate_by_base_charges` is the active truncation path at D=3 χ=12 (add a print in `patched` to confirm it is called), and adjust the filter (e.g. also enforce uniform per-sector counts) until the env is valid. Do not proceed to Task 4 until the guard passes — an invalid env makes the re-profile meaningless.

- [ ] **Step 5: Confirm the prototype actually drops sectors (sanity)**

Add a quick check (can be a `print` run, not committed): under the context manager, re-run the census `candidate_report`-style block-key count on the prototype env and confirm env `n_blocks` dropped ≈ the Gate-A prediction. This links the static prediction (Task 2) to the real prototype env.

Run:
```bash
JAX_PLATFORMS=cpu uv run python -c "
import jax; jax.config.update('jax_enable_x64', True)
from examples.u1sz_defrag_prototype_610 import sector_dropping_derive_charges
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
A,_=heisenberg_u1sz_init_pair(D=3,key=jax.random.PRNGKey(0))
with sector_dropping_derive_charges():
    env,_=ctm_tensor(A,chi=12,max_iter=8,conv_tol=1e-6)
    print({n: len(getattr(env,n)._block_keys) for n in env._fields})
"
```
Expected: env `n_blocks` materially below the Task-1 baseline. Record.

- [ ] **Step 6: Commit**

```bash
git add examples/u1sz_defrag_prototype_610.py tests/test_u1sz_defrag_prototype_610.py
git commit -m "study(#610): C-lever sector-dropping prototype + faithfulness guard"
```

---

## Task 4: Gate B — make-or-break backward re-profile (Stage 2)

The decisive Tier-1 measurement: does the charge-mask cluster shrink under the prototype? Trace-only, CPU, no XLA compile.

**Files:**
- Use: `examples/probe_backward_jaxpr_566.py` (u1sz arm; `backward_vjp_jaxpr`, `bucketize`)
- Use: `examples/u1sz_defrag_prototype_610.py`

- [ ] **Step 1: Capture the baseline backward op-histogram (u1sz, no prototype)**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py \
    --sym u1sz --D 3 --chi-factor 4 > probe_u1sz_baseline_610.txt
```
Expected: the bucketized histogram (buffer-pack / charge-mask / math clusters) printed. Record the **charge-mask cluster** absolute op count and total backward op count. (If the existing arm only runs D=2, run at the largest D it supports and note it; the cluster *shares* are the comparison, not absolute D.)

- [ ] **Step 2: Capture the under-prototype histogram**

The probe builds the site and traces the backward internally, so wrap its measurement call in the prototype context. Add a `--defrag` flag to `examples/probe_backward_jaxpr_566.py` that, when set, wraps the `backward_vjp_jaxpr` trace in `sector_dropping_derive_charges()`:
```python
# in main(), near the per-(sym,D,chi) loop:
import contextlib
ctx = contextlib.nullcontext()
if args.defrag:
    from examples.u1sz_defrag_prototype_610 import sector_dropping_derive_charges
    ctx = sector_dropping_derive_charges()
with ctx:
    counts = backward_vjp_jaxpr(A, chi, on=on)   # match the existing call site
```
Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py \
    --sym u1sz --D 3 --chi-factor 4 --defrag > probe_u1sz_defrag_610.txt
```
Expected: a histogram with **fewer** charge-mask ops. Record the charge-mask cluster count and total.

- [ ] **Step 3: Compute the charge-mask reduction**

Compute `1 - (charge_mask_defrag / charge_mask_baseline)` and the total-op reduction. Record both.

- [ ] **Step 4: GATE B decision (make-or-break, Tier 1)**

**Pass criterion:** charge-mask cluster op count drops **≥ 25%** (well beyond #609's ~1% noise) under the prototype, with total backward ops dropping commensurately.
- If **≥25%** → record PASS, proceed to Task 5 (A100).
- If **<25%** → record **NO-GO** (the representation lever does not attack the charge-mask third), jump to Task 6, skip Task 5. **Do not spend A100.**

- [ ] **Step 5: Commit**

```bash
git add examples/probe_backward_jaxpr_566.py probe_u1sz_baseline_610.txt probe_u1sz_defrag_610.txt
git commit -m "study(#610): Gate B — backward charge-mask re-profile under prototype"
```

---

## Task 5: Gate C — A100 end-to-end + energy (Stage 3, conditional)

Only if Gate B passed. The worth-it tier: does the runtime reach ≥ parity-with-dense, and is the energy within 1% of `E_frag`? Run on the A100 box (see memory `a100-gpu-env`: `uv sync --extra cuda13`, `JAX_PLATFORMS=cuda,cpu`, x64).

**Files:**
- Use: `examples/profile_ctm_ad_wall_566.py` (u1sz arm)
- Use: `examples/u1sz_defrag_prototype_610.py`

- [ ] **Step 1: Baseline + dense reference grid at D=3 χ=12 (A100)**

Run:
```bash
JAX_PLATFORMS=cuda,cpu uv run python examples/profile_ctm_ad_wall_566.py \
    --sym u1sz dense --D 3 --chi 12 --json profile_d3_baseline_610.json
```
Record `vg_cmp` and `warm_ms` for u1sz (fragmented) and dense. The dense `warm_ms`/`vg_cmp` define the parity-with-dense target.

- [ ] **Step 2: Under-prototype grid (A100)**

Add the same `--defrag` flag wrapping to `examples/profile_ctm_ad_wall_566.py`'s u1sz measurement call (mirror Task 4 Step 2). Run:
```bash
JAX_PLATFORMS=cuda,cpu uv run python examples/profile_ctm_ad_wall_566.py \
    --sym u1sz --D 3 --chi 12 --defrag --json profile_d3_defrag_610.json
```
Record `vg_cmp` and `warm_ms` under the prototype.

- [ ] **Step 3: Energy under the prototype vs `E_frag`**

Run:
```bash
JAX_PLATFORMS=cuda,cpu uv run python -c "
import jax; jax.config.update('jax_enable_x64', True)
from examples.u1sz_defrag_prototype_610 import sector_dropping_derive_charges
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair
A,_=heisenberg_u1sz_init_pair(D=3,key=jax.random.PRNGKey(0))
with sector_dropping_derive_charges():
    env,_=ctm_tensor(A,chi=12,max_iter=20,conv_tol=1e-7)
    print('E_uniform =', float(compute_energy_ctm_tensor(A,env,heisenberg_gate_u1sz())))
"
```
Record `E_uniform`. Compute `abs(E_uniform - E_frag) / abs(E_frag)`.

- [ ] **Step 4: GATE C decision (worth-it, Tier 2)**

**Pass criterion (both):**
1. `vg_cmp` **and/or** `warm_ms` under the prototype reaches **≥ parity with dense** (≈3× faster than the fragmented u1sz baseline), **and**
2. `abs(E_uniform - E_frag) / abs(E_frag) ≤ 0.01`.

- Both pass → **GO**.
- Runtime passes but energy fails (or vice versa) → **documented partial** (mechanism real, accuracy/worth-it cost too high).
- Neither → **NO-GO**.

- [ ] **Step 5: Commit**

```bash
git add examples/profile_ctm_ad_wall_566.py profile_d3_baseline_610.json profile_d3_defrag_610.json
git commit -m "study(#610): Gate C — A100 end-to-end + energy under prototype"
```

---

## Task 6: Findings, memory, and follow-up (always run)

Reached on any terminal gate (PASS-through to GO, or any NO-GO). Write the finding the way #609 did.

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md`
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/` (new memory + `MEMORY.md` pointer)

- [ ] **Step 1: Write the findings handoff**

Create `docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md` with: the verdict (GO / NO-GO / partial), which gate was binding, the recorded numbers (baseline census + `E_frag`; Gate-A reduction; Gate-B charge-mask drop; Gate-C runtime + energy if reached), the candidate that was prototyped (C sector-dropping), and the recommendation. Mirror the structure of `2026-06-16-u1sz-stacked-spike-findings.md`.

- [ ] **Step 2: Write a memory file + MEMORY.md pointer**

Create `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/610-u1sz-env-defrag.md` (type `project`) capturing the verdict, the binding gate, and the non-obvious takeaway (e.g. "charge-mask third is/ isn't reducible by sector-dropping"). Link `[[566-u1sz-stacking-nogo]]` and `[[u1sz-perf-study-d3-findings]]`. Add a one-line pointer to `MEMORY.md`.

- [ ] **Step 3: If GO — open the follow-up implementation issue**

Only if Gate C = GO:
```bash
gh issue create --title "feat(#566): implement U(1)-Sz CTM env de-fragmentation (sector-dropping truncation)" \
  --body "Spike #610 GO: the sector-dropping chi-bond truncation prototype cut the backward charge-mask cluster by <X>% and reached parity-with-dense at D=3 chi=12 within 1% energy. Implement it as a committed src/ change behind the accuracy spine. See docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md and spec 2026-06-16-u1sz-env-defrag-design.md."
```

- [ ] **Step 4: Commit the findings + memory pointer**

```bash
git add docs/superpowers/handoffs/2026-06-17-u1sz-env-defrag-findings.md
git commit -m "docs(#610): U(1)-Sz CTM env de-fragmentation spike — <GO|NO-GO|partial> finding"
```

- [ ] **Step 5: Open the spike PR**

```bash
git push -u origin spike/610-u1sz-env-defrag
gh pr create --title "study(#610): U(1)-Sz CTM env de-fragmentation spike — <verdict>" \
  --body "Measure-first GO/NO-GO spike per docs/superpowers/specs/2026-06-16-u1sz-env-defrag-design.md. Three-gate staircase; no committed src/. Verdict: <...>. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Note: the throwaway `examples/_e_frag_610.py` and prototype examples stay on the branch as spike artifacts (mirroring #609, which kept its census/probe tools); **no `src/` file is modified by this spike.**

---

## Self-Review

**Spec coverage:**
- §3 Stage 0 baseline lock → Task 1. ✓
- §3 Stage 1 static sector census + Gate A → Task 2. ✓
- §3 Stage 2 prototype + faithfulness guard + Gate B re-profile → Tasks 3–4. ✓
- §3 Stage 3 A100 end-to-end + energy + Gate C → Task 5. ✓
- §4 candidate C lead / B cramped-at-D=3 noted → Task 2 Step 5. ✓
- §5 faithfulness guard before profiling → Task 3 Step 4 (gates Task 4). ✓
- §7 deliverables (census, prototype, re-profile, findings, memory, follow-up issue) → Tasks 2,3,4,6. ✓
- §8 GO definition (A and B and C) → gate-decision steps in Tasks 2/4/5. ✓
- §10 non-goal "no committed src/" → enforced in every task (monkeypatch only); restated in Task 6 Step 5. ✓

**Placeholder scan:** No TBD/TODO. The `<X>%`/`<verdict>` tokens in Task 6 are intentional fill-from-measurement values in human-authored prose, not code placeholders.

**Type consistency:** `predict_sectors_under_keep(block_keys, chi_axes, keep)` defined in Task 2 Step 3, used with the same signature in the Task 2 Step 1 test. `sector_dropping_derive_charges(keep=...)` context manager defined in Task 3 Step 3, used identically in Tasks 3/4/5. `_derive_charges(base_charges, target_dim)` matches the real signature in `src/tenax/algorithms/_ctm_utils.py:45`.
