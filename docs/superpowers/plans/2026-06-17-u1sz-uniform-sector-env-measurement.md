# U(1)-Sz uniform-sector env — flag-gated Gate-B measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal opt-in (default-off) structural change that makes the 2×2 multisite U(1)-Sz CTM carry a uniform chi-bond sector set, so the AD backward becomes cold-traceable under the `|Sz|≤1` sector drop — then measure Gate B (charge-mask op reduction) and emit a documented GO/NO-GO finding.

**Architecture:** A `keep_sectors` knob, **off by default**. The public entry points (`ctm_tensor`, `ctm_energy_implicit`) expose a `keep_sectors: frozenset[int] | None = None` parameter and, internally, activate a **process-local context** (`keep_sectors_context`) — read at *trace time* — that the symmetric env-init and the projector truncation consult. This avoids threading the flag through ~8 projector layers and mirrors the codebase's existing module-toggle pattern (`set_implicit_ad_norm_diagnostics`, `_batch_blocksparse_enabled()`). When the context is inactive (`None`), every touched function takes its original path verbatim. Tactic **A** (uniform-by-construction: env-init seeds `keep` + truncation outputs `keep`); tactic **C** (conform at the `_build_enlarged_corner` boundary) is the scoped fallback if per-sector multiplicities mismatch; tactic **B** (sweep rewrite) is deferred to the full feature.

**Tech Stack:** Python, JAX (x64), pytest, the Tenax block-sparse `SymmetricTensor` CTM-AD path. Reuses the #610 assets (merged via #614): `examples/probe_backward_jaxpr_566.py` (`--defrag`), `examples/u1sz_defrag_prototype_610.py` (the truncation reference — `_make_keep_filtered_traced_svd`), `tests/test_profiler_u1sz_arm.py`.

**Research-spike note (read before executing):** Tasks 0–3, 5–7 are deterministic and get full TDD. **Task 4 is the make-or-break gate** — whether tactic A alone makes the cold backward VJP *build*. Its acceptance is crisp (the cold VJP traces without `ValueError`), but the *mechanism* may need one or two iterations (A-consistency → tactic C). **Honor the gate:** if Task 4 cannot reach a building trace via A-consistency or tactic C cheaply, STOP, record "uniform-by-construction insufficient; needs tactic B sweep rewrite" as the finding (Task 7), and skip Tasks 5–6. The src change merges to `main` **only if Gate B (Task 6) passes** — until then it stays branch-local.

**Spec:** `docs/superpowers/specs/2026-06-17-u1sz-uniform-sector-env-measurement-design.md`

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/tenax/algorithms/_ctm_uniform_sector.py` | **Create** | The process-local `keep_sectors` context: `keep_sectors_context(keep)` (sets/restores) + `current_keep_sectors()` (trace-time reader) + `restrict_charges_to_keep(charges, keep)` helper. Default state `None`. |
| `src/tenax/algorithms/_ctm_tensor_init.py` | Modify | Env-init seed: in `_init_symmetric_standard_corner` and `_init_symmetric_standard_edge`, restrict chi-bond charges to the active keep set before `_grouped_chi_perm`. |
| `src/tenax/algorithms/_ctm_tensor_paired_moves.py` | Modify | `_get_base_charges`: filter extracted charges to the active keep set. |
| `src/tenax/algorithms/_ctm_tensor_convergence.py` | Modify | `_get_base_charges`: same keep-filter. `ctm_tensor`: add `keep_sectors` param wrapping the body in `keep_sectors_context`. |
| `src/tenax/algorithms/_ctm_energy_ad.py` | Modify | `ctm_energy_implicit` (+ its dispatch): add `keep_sectors` param wrapping the body in `keep_sectors_context`, so the implicit-AD backward honors the drop. |
| `src/tenax/linalg.py` | Modify | `_truncated_svd_symmetric_traced`: when keep is active, drop non-keep sectors in Phase-1 allocation and suppress the Phase-2 χ-backfill (promote the prototype's `_make_keep_filtered_traced_svd` logic into a guarded branch). |
| `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` | Modify | `_retruncate_by_base_charges`: mirror the keep-restriction (eager forward path). |
| `tests/test_u1sz_uniform_sector_615.py` | **Create** | Default-off identity test, env-init keep-seed test, faithfulness guard, and the cold-trace-builds regression lock. |
| `examples/probe_backward_jaxpr_566.py` | Modify | Repoint `--defrag` from the monkeypatch to `keep_sectors_context` (Gate B). |
| `docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md` | **Create (last task)** | The Gate-B measured number + GO/NO-GO finding. |

**Note on line anchors:** the `src/` files are unchanged by #610, so all `src/` line numbers below are valid on the current branch. The `examples/probe_backward_jaxpr_566.py --defrag` flag and `examples/u1sz_defrag_prototype_610.py` arrive via #614; Task 0 merges them in.

---

## Task 0: Sync the branch and lock the baseline

Bring in the #610 artifacts (probe `--defrag`, prototype reference, docs) once #614 lands, and confirm the D=3 χ=12 backward traces (flag-off) so later deltas are attributable.

**Files:** none modified (sync + smoke only).

- [ ] **Step 1: Merge `origin/main` once #614 has merged**

```bash
cd /home/yjkao/tenax
gh pr view 614 --json state -q .state    # expect: MERGED
git fetch origin
git merge origin/main
```
Expected: clean merge (this branch only adds a new spec doc; #614 adds `examples/`+`tests/`+`docs/`). If `gh` shows `OPEN`, wait for the merge queue and retry — do not proceed.

- [ ] **Step 2: Confirm the D=3 χ=12 U(1)-Sz forward runs (flag-off prerequisite)**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py::test_u1sz_ctm_forward_runs -v
```
Expected: PASS for the `[3-8]` case. If it fails, STOP — the prerequisite is broken; record and report.

- [ ] **Step 3: Confirm the flag-off backward traces and record the anchor**

```bash
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py --sym u1sz --D 3 --chi-factor 4 > /tmp/probe_baseline_615.txt
tail -20 /tmp/probe_baseline_615.txt
```
Expected: the bucketized histogram prints; total ≈ **63,612** ops, charge-mask ≈ **7,398** (the #610 anchor). Record both. No commit (no file changed).

---

## Task 1: The `keep_sectors` context + flag plumbing (default-off firewall)

Create the process-local context and wire `keep_sectors` onto the public entry points. With `keep_sectors=None`, behavior is byte-identical — proven by an identity test.

**Files:**
- Create: `src/tenax/algorithms/_ctm_uniform_sector.py`
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py` (`ctm_tensor`, ~line 571), `src/tenax/algorithms/_ctm_energy_ad.py` (`ctm_energy_implicit`, ~line 337; `_ctm_energy_implicit_dispatch`, ~line 723)
- Test: `tests/test_u1sz_uniform_sector_615.py`

- [ ] **Step 1: Write the failing identity test**

Create `tests/test_u1sz_uniform_sector_615.py`:
```python
"""#615 uniform-sector env: flag-gated Gate-B measurement scaffold."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair


def _env_block_signature(env):
    """A structural fingerprint: per-tensor sorted (block-key, shape) list."""
    sig = {}
    for name in env._fields:
        t = getattr(env, name)
        sig[name] = sorted(
            (tuple(int(q) for q in k), tuple(int(s) for s in b.shape))
            for k, b in t.blocks.items()
        )
    return sig


def test_keep_sectors_none_is_identity():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env_a, e_a = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4)
    env_b, e_b = ctm_tensor(A, chi=12, max_iter=4, conv_tol=1e-4, keep_sectors=None)
    assert _env_block_signature(env_a) == _env_block_signature(env_b)
    assert float(e_a) == float(e_b)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_keep_sectors_none_is_identity -v
```
Expected: FAIL with `TypeError: ctm_tensor() got an unexpected keyword argument 'keep_sectors'`.

- [ ] **Step 3: Create the context module**

Create `src/tenax/algorithms/_ctm_uniform_sector.py`:
```python
"""Process-local "keep sectors" context for the U(1)-Sz uniform-sector env (#615).

Default OFF. When a keep set is active, the symmetric CTM env-init and the
projector truncation restrict chi-bond charges to the keep set, so the 2x2
multisite backward carries a *uniform* sector set and becomes traceable under
the sector drop (#610 NO-GO-by-obstruction).

The keep set is a STATIC, structural choice (it changes block sectors/shapes),
so it is read at TRACE time as a plain Python value via ``current_keep_sectors``
— never threaded as a traced array. Mirrors the codebase's existing module-level
toggles (``set_implicit_ad_norm_diagnostics``, ``_batch_blocksparse_enabled``).
"""
from __future__ import annotations

import contextlib

import numpy as np

_KEEP_SECTORS: frozenset[int] | None = None


def current_keep_sectors() -> frozenset[int] | None:
    """The active keep set, or ``None`` (default: no restriction)."""
    return _KEEP_SECTORS


@contextlib.contextmanager
def keep_sectors_context(keep):
    """Activate a keep set for the duration of the ``with`` block.

    ``keep=None`` is a no-op pass-through (restores the default path), so callers
    can wrap unconditionally: ``with keep_sectors_context(keep_sectors): ...``.
    """
    global _KEEP_SECTORS
    prev = _KEEP_SECTORS
    _KEEP_SECTORS = None if keep is None else frozenset(int(q) for q in keep)
    try:
        yield
    finally:
        _KEEP_SECTORS = prev


def restrict_charges_to_keep(charges, keep) -> np.ndarray:
    """Return ``charges`` with entries outside ``keep`` removed.

    Degenerate guard: if filtering would empty the array, return the original
    (an env bond / SVD sector list must never be empty).
    """
    arr = np.asarray(charges, dtype=np.int32)
    if keep is None:
        return arr
    mask = np.array([int(q) in keep for q in arr], dtype=bool)
    return arr[mask] if mask.any() else arr
```

- [ ] **Step 4: Add the `keep_sectors` param to `ctm_tensor`**

In `src/tenax/algorithms/_ctm_tensor_convergence.py`, add the import near the other algorithm imports:
```python
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context
```
Change the `ctm_tensor` signature (~line 571) to add the parameter (keep all existing params; append the new one):
```python
def ctm_tensor(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "svd",
    qr_warmup_steps: int = 3,
    projector_backward: str = "auto",
    keep_sectors: frozenset[int] | None = None,
) -> tuple[CTMTensorEnv, float]:
```
Wrap the existing function body in the context. Concretely, immediately after the docstring, open the context and indent the existing body one level:
```python
    with keep_sectors_context(keep_sectors):
        # ... existing body unchanged (env-init, sweep loop, return) ...
```
(If indenting the whole body is unwieldy, instead set it explicitly at the top and restore in a `finally` — but the context manager is preferred and matches the test.)

- [ ] **Step 5: Run the identity test to verify it passes**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_keep_sectors_none_is_identity -v
```
Expected: PASS (with `keep_sectors=None` the context is a no-op, so nothing downstream changes).

- [ ] **Step 6: Thread `keep_sectors` onto the implicit-AD entry**

In `src/tenax/algorithms/_ctm_energy_ad.py`, add the import:
```python
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context
```
Add `keep_sectors: frozenset[int] | None = None` to the signatures of `ctm_energy_implicit` (~line 337) and `_ctm_energy_implicit_dispatch` (~line 723), and wrap each function body in `with keep_sectors_context(keep_sectors):` (so the forward env-init **and** the `jit_step_bwd` backward sweep both run under the active keep set). Pass `keep_sectors` from `ctm_energy_implicit` down to `_ctm_energy_implicit_dispatch` at its call site.

- [ ] **Step 7: Verify default-off across the core suite (no regression)**

```bash
JAX_PLATFORMS=cpu uv run pytest -m core -q
JAX_PLATFORMS=cpu uv run pytest tests/test_profiler_u1sz_arm.py -q
```
Expected: PASS (the new param defaults to `None`; no existing call changes).

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check src/ tests/
git add src/tenax/algorithms/_ctm_uniform_sector.py \
        src/tenax/algorithms/_ctm_tensor_convergence.py \
        src/tenax/algorithms/_ctm_energy_ad.py \
        tests/test_u1sz_uniform_sector_615.py
git commit -m "feat(#615): keep_sectors context + default-off flag plumbing"
```

---

## Task 2: Env-init seeds only the keep sectors

When a keep set is active, the freshly-initialized env chi bonds carry only `keep` charges. This is half of tactic A (the other half is Task 3).

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_init.py` (`_init_symmetric_standard_corner` ~line 390; `_init_symmetric_standard_edge` ~lines 334–337)
- Test: `tests/test_u1sz_uniform_sector_615.py`

- [ ] **Step 1: Write the failing env-init keep-seed test**

Append to `tests/test_u1sz_uniform_sector_615.py`:
```python
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context


def _chi_sectors(t):
    """Set of charges appearing on any 'chi'-labelled leg of tensor ``t``."""
    out = set()
    for ix in t.indices:
        if ix.label.lower().startswith("chi"):
            out |= {int(q) for q in ix.charges}
    return out


def test_env_init_seeds_only_keep_sectors():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    # Baseline: chi legs carry the full {-2..+2}.
    env0 = initialize_ctm_tensor_env(A, chi=12)
    base = set().union(*(_chi_sectors(getattr(env0, n)) for n in env0._fields))
    assert 2 in base or -2 in base, "baseline should carry |Sz|=2 chi sectors"
    # Under keep={-1,0,1}, no chi leg may carry |Sz|=2.
    with keep_sectors_context({-1, 0, 1}):
        env = initialize_ctm_tensor_env(A, chi=12)
    for n in env._fields:
        assert _chi_sectors(getattr(env, n)) <= {-1, 0, 1}, f"{n} kept |Sz|=2"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_env_init_seeds_only_keep_sectors -v
```
Expected: FAIL — the second assertion trips (env still carries `±2`).

- [ ] **Step 3: Restrict chi charges in the corner init**

In `src/tenax/algorithms/_ctm_tensor_init.py`, add at the top of the module:
```python
from tenax.algorithms._ctm_uniform_sector import (
    current_keep_sectors,
    restrict_charges_to_keep,
)
```
In `_init_symmetric_standard_corner`, replace the permutation block (currently lines ~390–391):
```python
    perm = _grouped_chi_perm(chi_charges)
    chi_charges = np.asarray(chi_charges)[perm]
```
with a keep-restriction *before* the perm, and apply the perm to the (possibly smaller) charge array. Because the corner data is rank-1 (`C_dense[0,0]=1`), the dense array must be re-sized to the kept length:
```python
    chi_charges = restrict_charges_to_keep(chi_charges, current_keep_sectors())
    chi_len = len(chi_charges)
    perm = _grouped_chi_perm(chi_charges)
    chi_charges = np.asarray(chi_charges)[perm]
```
and change the dense construction (currently lines ~397–399) to use `chi_len`:
```python
    idx_a = TensorIndex.from_charges(sym, chi_charges.copy(), flow_a, label=label_a)
    idx_b = TensorIndex.from_charges(sym, chi_charges.copy(), flow_b, label=label_b)
    C_dense = jnp.zeros((chi_len, chi_len), dtype=A.dtype).at[0, 0].set(1.0)
    C_dense = C_dense[perm][:, perm]
```

- [ ] **Step 4: Restrict chi charges in the edge init**

In `_init_symmetric_standard_edge`, replace the perm block (currently lines ~334–337):
```python
    perm1 = _grouped_chi_perm(chi1_charges)
    perm2 = _grouped_chi_perm(chi2_charges)
    chi1_charges = np.asarray(chi1_charges)[perm1]
    chi2_charges = np.asarray(chi2_charges)[perm2]
```
with a keep-restriction first, tracking the kept lengths for the dense δ-pattern:
```python
    keep = current_keep_sectors()
    chi1_charges = restrict_charges_to_keep(chi1_charges, keep)
    chi2_charges = restrict_charges_to_keep(chi2_charges, keep)
    chi1_len = len(chi1_charges)
    chi2_len = len(chi2_charges)
    perm1 = _grouped_chi_perm(chi1_charges)
    perm2 = _grouped_chi_perm(chi2_charges)
    chi1_charges = np.asarray(chi1_charges)[perm1]
    chi2_charges = np.asarray(chi2_charges)[perm2]
```
and change the dense δ-pattern construction (currently lines ~348–352) to use the kept lengths:
```python
    diag_idx = np.arange(D, dtype=np.int32) * (D + 1)
    T = jnp.zeros((chi1_len, D2, chi2_len), dtype=A.dtype)
    T = T.at[0, diag_idx, 0].set(jnp.ones(D, dtype=A.dtype))
    T = T[perm1][:, :, perm2]
```
(The `[0, ..., 0]` δ-write requires `chi1_len ≥ 1` and `chi2_len ≥ 1`, guaranteed by `restrict_charges_to_keep`'s never-empty guard, since charge `0` is always in `keep`.)

- [ ] **Step 5: Run the env-init test to verify it passes**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_env_init_seeds_only_keep_sectors -v
```
Expected: PASS. Also re-run the identity test (Step from Task 1) to confirm default-off is still byte-identical:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_keep_sectors_none_is_identity -v
```
Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src/ tests/
git add src/tenax/algorithms/_ctm_tensor_init.py tests/test_u1sz_uniform_sector_615.py
git commit -m "feat(#615): env-init seeds only keep sectors under active context"
```

---

## Task 3: Truncation outputs only the keep sectors (traced + eager) + base_charges filter

The other half of tactic A: every projector truncation produces a `keep`-only chi bond — dropping non-keep sectors and **never** backfilling χ from a non-keep sector. This is the prototype's patch (4)/(3)/(1) promoted into guarded branches.

**Files:**
- Modify: `src/tenax/linalg.py` (`_truncated_svd_symmetric_traced`, ~lines 583–864; backfill ~732–747)
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (`_retruncate_by_base_charges`, ~lines 709–801)
- Modify: `src/tenax/algorithms/_ctm_tensor_paired_moves.py` (`_get_base_charges`, ~line 38) and `src/tenax/algorithms/_ctm_tensor_convergence.py` (`_get_base_charges`, ~line 239)

- [ ] **Step 1: Filter both `_get_base_charges` to the active keep set**

In **both** `_ctm_tensor_paired_moves.py` and `_ctm_tensor_convergence.py`, import the reader and apply the filter just before each `_get_base_charges` returns `charges`:
```python
from tenax.algorithms._ctm_uniform_sector import current_keep_sectors, restrict_charges_to_keep
...
    # (existing: charges = np.asarray(a.indices[u2_pos].charges, dtype=np.int32))
    if np.all(charges == 0):
        return None
    keep = current_keep_sectors()
    if keep is not None:
        charges = restrict_charges_to_keep(charges, keep)
    return charges
```
This sets the per-sector allocation *intent* to keep-only on both the forward paired-moves path and the multisite/backward path.

- [ ] **Step 2: Constrain the traced SVD (the backward truncation)**

In `src/tenax/linalg.py`, in `_truncated_svd_symmetric_traced`, import the reader at module scope and apply two guarded edits inside the `base_charges is not None` allocation block (~lines 714–747), exactly mirroring `examples/u1sz_defrag_prototype_610.py::_make_keep_filtered_traced_svd` (the proven reference):
```python
from tenax.algorithms._ctm_uniform_sector import current_keep_sectors
...
        target_charges = _derive_charges(base_charges, max_singular_values)
        target_count = {}
        for tq in target_charges:
            target_count[int(tq)] = target_count.get(int(tq), 0) + 1
        _keep = current_keep_sectors()           # None ⇒ original behaviour
        k_per_sector = {
            q: (0 if (_keep is not None and int(q) not in _keep)
                else min(target_count.get(q, 0), r[5]))
            for q, r in sector_results.items()
        }
        remaining = max_singular_values - sum(k_per_sector.values())
        if remaining > 0:
            for q in sorted(sector_results.keys(),
                            key=lambda qq: (-(sector_results[qq][5] - k_per_sector.get(qq, 0)), qq)):
                if remaining <= 0:
                    break
                if _keep is not None and int(q) not in _keep:   # never backfill non-keep
                    continue
                capacity_left = sector_results[q][5] - k_per_sector.get(q, 0)
                take = min(remaining, capacity_left)
                if take > 0:
                    k_per_sector[q] = k_per_sector.get(q, 0) + take
                    remaining -= take
```
Consequence: with keep active, `chi_new` is the sum of kept-sector allocations and may be `< max_singular_values` (~10 at D=3 χ=12). With keep `None`, this is byte-identical to the current code. **Do not** change the concatenation / block-rebuild logic below — only the `k_per_sector` allocation.

- [ ] **Step 3: Mirror the restriction in the eager retruncation**

In `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`, `_retruncate_by_base_charges` (~lines 709–801): import `current_keep_sectors`, and in its greedy fill, apply the same two rules — a sector outside the active keep set gets `0`, and the leftover-budget refill skips non-keep sectors. (Match the exact local variable names in that function; the logic is identical to Step 2.)

- [ ] **Step 4: Write the failing "truncation drops sectors" test**

Append to `tests/test_u1sz_uniform_sector_615.py`:
```python
def test_forward_env_block_counts_drop_under_keep():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env0, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6)
    n0 = {n: len(getattr(env0, n).blocks) for n in env0._fields}
    env1, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6, keep_sectors={-1, 0, 1})
    n1 = {n: len(getattr(env1, n).blocks) for n in env1._fields}
    # #610 Gate-A: corners 5->3, edges 19->9; assert a real drop on every tensor.
    for name in n0:
        assert n1[name] < n0[name], f"{name}: {n1[name]} !< {n0[name]}"
    # And no chi leg carries |Sz|=2 in the converged env.
    for name in env1._fields:
        assert _chi_sectors(getattr(env1, name)) <= {-1, 0, 1}
```

- [ ] **Step 5: Run it (and the identity test) to verify pass**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py -v
```
Expected: `test_forward_env_block_counts_drop_under_keep` PASS, `test_keep_sectors_none_is_identity` still PASS, `test_env_init_seeds_only_keep_sectors` still PASS. If the forward energy is NaN/non-finite under keep, that's a faithfulness problem — debug before continuing (the converged env must be valid).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src/ tests/
git add src/tenax/linalg.py \
        src/tenax/algorithms/_ctm_tensor_projector_2x2.py \
        src/tenax/algorithms/_ctm_tensor_paired_moves.py \
        src/tenax/algorithms/_ctm_tensor_convergence.py \
        tests/test_u1sz_uniform_sector_615.py
git commit -m "feat(#615): keep-restricted projector truncation (traced + eager) + base_charges filter"
```

---

## Task 4: The make-or-break — cold backward VJP builds under the flag

The decisive structural gate: does tactic A (Tasks 2+3) make the 2×2 multisite **backward** trace without `ValueError`? This is the #610 obstruction test, inverted to PASS.

**Files:**
- Test: `tests/test_u1sz_uniform_sector_615.py`
- (Iteration only, if needed) `src/tenax/algorithms/_ctm_tensor_init.py`, `src/tenax/algorithms/_ctm_tensor_projector_2x2.py`

- [ ] **Step 1: Write the cold-trace-builds test**

Append to `tests/test_u1sz_uniform_sector_615.py`. Use the probe's backward tracer (the same unit the #610 obstruction hit):
```python
from examples.probe_backward_jaxpr_566 import backward_vjp_jaxpr  # from #614


def test_cold_backward_vjp_builds_under_keep():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    # COLD trace is load-bearing: a prior flag-off trace of the same jit unit
    # (identical avals) would be reused, masking the structural change (#610).
    jax.clear_caches()
    with keep_sectors_context({-1, 0, 1}):
        counts = backward_vjp_jaxpr(A, chi=12, on=True)  # raised ValueError pre-#615
    assert counts is not None and counts.get("total", 0) > 0
```
(If `backward_vjp_jaxpr`'s signature differs after #614, match its actual call — it returns the bucketized op counts and traces the 2×2 multisite VJP internally.)

- [ ] **Step 2: Run it — this is the gate**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_cold_backward_vjp_builds_under_keep -v
```
- **PASS** → tactic A is sufficient. Go to Step 4.
- **FAIL with `ValueError: Size of label ... (X) != (Y)`** → per-sector **multiplicity** mismatch (uniform sector *set* but unequal per-sector dims between an init-seeded leg and a truncated leg). Go to Step 3.

- [ ] **Step 3 (only if Step 2 mismatched): A-consistency, then tactic C**

Apply in order; re-run Step 2 after each; STOP at the first that makes the trace build:

- **A-consistency (preferred — stays in tactic A):** make env-init seed the SAME per-keep-sector multiplicities the truncation targets. Both already tile the D²-charge pattern; align them by deriving the corner/edge chi charges from `_derive_charges(restrict_charges_to_keep(base, keep), chi)` using the *identical* base-charge ordering the truncation consumes (the double-layer `u2` charges), so the `[:chi]` truncation picks the same per-sector counts. Edit in `_ctm_tensor_init.py`.
- **Tactic C (scoped fallback):** if multiplicities still diverge (data-availability under-fills a keep sector below the seeded count), conform the perpendicular legs to the truncated allocation **only at the `_build_enlarged_corner` boundary** in `_ctm_tensor_projector_2x2.py` (project/pad the un-refreshed leg to the contracted leg's per-sector dims). Smallest delta that closes the specific gap.
- **Escalation (record, do not build):** if neither builds the trace within ~2 iterations, STOP. Record in the Task-7 findings: *"uniform-by-construction (tactic A) + boundary-conform (tactic C) insufficient; the measurement requires tactic B (all-legs-per-sweep refresh)"* — a documented escalation, not silent scope growth. Skip Tasks 5–6.

- [ ] **Step 4: Confirm default-off backward is unchanged (regression)**

```bash
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py --sym u1sz --D 3 --chi-factor 4 > /tmp/probe_offcheck_615.txt
diff <(tail -20 /tmp/probe_baseline_615.txt) <(tail -20 /tmp/probe_offcheck_615.txt) && echo "FLAG-OFF UNCHANGED ✓"
```
Expected: identical (flag-off backward must match the Task-0 anchor).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/ tests/
git add tests/test_u1sz_uniform_sector_615.py src/tenax/algorithms/_ctm_tensor_init.py src/tenax/algorithms/_ctm_tensor_projector_2x2.py
git commit -m "feat(#615): cold backward VJP builds under keep (tactic A; +C fallback if needed)"
```

---

## Task 5: Faithfulness guard under the real flag

Before trusting Gate-B op counts, prove the keep-active CTM produces a *valid* env: converges, finite, energy sane at D=3 χ=12.

**Files:** Test: `tests/test_u1sz_uniform_sector_615.py`

- [ ] **Step 1: Write the faithfulness-guard test**

Append to `tests/test_u1sz_uniform_sector_615.py`:
```python
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz


def test_keep_env_is_faithful_at_d3_chi12():
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7, keep_sectors={-1, 0, 1})
    for name in env._fields:
        assert np.all(np.isfinite(np.asarray(getattr(env, name)._data))), f"{name} non-finite"
    e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))
    assert np.isfinite(e) and -2.0 < e < 0.0, f"energy {e} outside sane Heisenberg window"
```

- [ ] **Step 2: Run it**

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_uniform_sector_615.py::test_keep_env_is_faithful_at_d3_chi12 -v
```
Expected: PASS. If energy is out of window or non-finite, the truncation is producing an invalid charge-conserving env — debug (likely a degenerate keep allocation) before Gate B.

- [ ] **Step 3: Commit**

```bash
git add tests/test_u1sz_uniform_sector_615.py
git commit -m "test(#615): faithfulness guard — keep-active env converges, energy sane"
```

---

## Task 6: Gate B — measure the charge-mask reduction, then decide

Repoint the probe's `--defrag` to the real flag (no monkeypatch), measure flag-off vs flag-on at D=3 χ=12 with **cold** traces, and apply the Gate-B criterion.

**Files:** Modify: `examples/probe_backward_jaxpr_566.py`

- [ ] **Step 1: Repoint `--defrag` to the real context**

In `examples/probe_backward_jaxpr_566.py`, replace the `--defrag` monkeypatch wrapper (added in #610) with the real context, and clear caches so the on-trace is cold:
```python
import contextlib, jax
from tenax.algorithms._ctm_uniform_sector import keep_sectors_context

ctx = contextlib.nullcontext()
if args.defrag:
    jax.clear_caches()                      # cold-trace: do NOT reuse the flag-off jaxpr
    ctx = keep_sectors_context({-1, 0, 1})
with ctx:
    counts = backward_vjp_jaxpr(A, chi, on=on)   # match the existing call site
```

- [ ] **Step 2: Capture flag-off and flag-on histograms**

```bash
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py --sym u1sz --D 3 --chi-factor 4 > probe_u1sz_off_615.txt
JAX_PLATFORMS=cpu uv run python examples/probe_backward_jaxpr_566.py --sym u1sz --D 3 --chi-factor 4 --defrag > probe_u1sz_on_615.txt
```
Expected: the flag-on run **traces** (no `ValueError`) and shows fewer charge-mask ops. Record the charge-mask cluster count and total for each.

- [ ] **Step 3: Compute the reduction**

Compute `charge_mask_reduction = 1 - charge_mask_on / charge_mask_off` and `total_reduction = 1 - total_on / total_off`. (Anchor: off ≈ 7,398 / 63,612.) Record both.

- [ ] **Step 4: GATE B decision**

**Pass criterion:** `charge_mask_reduction ≥ 0.25` (well beyond #609's ~1% noise), with `total_reduction` commensurate.
- **≥25% → GO.** Proceed to Task 7 GO branch (PR to main + follow-up issue).
- **<25% → NO-GO.** Proceed to Task 7 NO-GO branch (document the measured number; do not merge to main).

- [ ] **Step 5: Commit the measurement artifacts**

```bash
uv run ruff check src/ tests/
git add examples/probe_backward_jaxpr_566.py probe_u1sz_off_615.txt probe_u1sz_on_615.txt
git commit -m "study(#615): Gate B — keep-active backward charge-mask re-profile"
```

---

## Task 7: Findings, memory, and the decision (always run)

Reached on any terminal outcome (GO, NO-GO, or Task-4 escalation). Record the *measured* number — this is the deliverable that converts #610's inference.

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md`
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/` (new memory + `MEMORY.md` pointer)

- [ ] **Step 1: Write the findings handoff**

Create `docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md` with: the verdict (GO / NO-GO / escalation), the binding outcome (cold-trace built? via tactic A or C?), the measured numbers (flag-off vs flag-on charge-mask + total; the reduction %), the faithfulness energy, and the recommendation. Mirror the #610 findings structure.

- [ ] **Step 2: Memory + MEMORY.md pointer**

Create `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/615-u1sz-uniform-sector-env.md` (type `project`): the verdict, the measured charge-mask reduction, and whether tactic A alone sufficed (the non-obvious takeaway). Link `[[610-u1sz-env-defrag]]`, `[[566-u1sz-stacking-nogo]]`, `[[u1sz-perf-study-d3-findings]]`. Add a one-line pointer to `MEMORY.md`.

- [ ] **Step 3a: If GO — open the follow-up full-feature issue**

```bash
gh issue create --title "feat(#566): productionize U(1)-Sz uniform-sector env (Gate C + accuracy spine + tactic B)" \
  --body "Spike #615 GO: keep-active backward cut the charge-mask cluster by <X>% at D=3 chi=12 (cold trace builds via tactic <A|C>). Productionize: Gate C (A100 end-to-end + |E_uniform-E_frag|/|E_frag|<=1% on an OPTIMIZED state), accuracy spine, and tactic B (all-legs sweep refresh). See docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md."
```

- [ ] **Step 3b: If NO-GO — record and do NOT merge src/ to main**

Note in the findings that the measured charge-mask reduction (<25%) does not clear the bar; the flag-gated `src/` change stays branch-local for the record. (No `gh issue`; no PR to main.)

- [ ] **Step 4: Commit the findings + memory pointer**

```bash
git add docs/superpowers/handoffs/2026-06-17-u1sz-uniform-sector-env-findings.md
git commit -m "docs(#615): U(1)-Sz uniform-sector env — Gate-B <GO|NO-GO> finding"
```

- [ ] **Step 5: If GO — open the PR to main**

```bash
git push -u origin feat/615-u1sz-uniform-sector-env
gh pr create --title "feat(#615): U(1)-Sz uniform-sector env (default-off) — Gate-B GO" \
  --body "Opt-in (default-off) keep_sectors flag makes the 2x2 multisite CTM carry a uniform chi-bond sector set; the AD backward is now cold-traceable under the |Sz|<=1 drop. Gate B measured a <X>% charge-mask reduction at D=3 chi=12. Default path byte-identical (identity test). See docs/superpowers/specs/2026-06-17-u1sz-uniform-sector-env-measurement-design.md. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
If NO-GO, skip — the branch documents the finding without merging.

---

## Self-Review

**Spec coverage:**
- §3 flag (default-off, `keep_sectors`) → Task 1 (context + public param + identity test). ✓
- §4 tactic A injection points: env-init seed → Task 2; truncation constraint (traced+eager) + base_charges filter → Task 3. ✓
- §4 make-or-break cold-trace + A-consistency/tactic-C fallback + tactic-B escalation → Task 4. ✓
- §5 faithfulness guard → Task 5. ✓
- §6 Gate B (repoint probe, cold trace, ≥25%) → Task 6. ✓
- §7 verification: default-off identity (Task 1), flag-on faithfulness (Task 5), cold-trace-builds lock (Task 4), flag-off backward unchanged (Task 4 Step 4). ✓
- §8 merge discipline (GO→PR+issue; NO-GO→no merge) → Task 6 Step 4 + Task 7. ✓
- §9 deliverables (src change, tests, Gate-B numbers, findings, memory, follow-up) → Tasks 1–7. ✓

**Placeholder scan:** No TBD/TODO. `<X>%` and `<A|C>`/`<GO|NO-GO>` in Task 7 are intentional fill-from-measurement values in human-authored issue/PR prose, not code placeholders. Line anchors marked `~` are flagged in the File Structure note as re-confirm-at-execution (src unchanged by #610, so they are current).

**Type/name consistency:** `keep_sectors_context(keep)` / `current_keep_sectors()` / `restrict_charges_to_keep(charges, keep)` defined in Task 1 Step 3, used identically in Tasks 2, 3, 6. `keep_sectors: frozenset[int] | None = None` param defined in Task 1 Steps 4/6, used in tests in Tasks 1/3/5. `backward_vjp_jaxpr(A, chi, on=...)` (from #609/#614) used in Tasks 4/6. `_env_block_signature`/`_chi_sectors` test helpers defined once (Tasks 1/2) and reused.

**Scope check:** single coherent measurement build (one plan); Gate C / accuracy spine / tactic B explicitly deferred to the GO follow-up issue.
