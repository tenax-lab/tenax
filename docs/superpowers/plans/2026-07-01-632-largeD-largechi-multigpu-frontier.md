# Large-D × large-χ multi-GPU frontier benchmark (phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the reachable (D, χ) frontier of one iPEPS `value_and_grad` step across split-CTM (1-GPU, χ²·D⁴), dense (1-GPU, χ²·D⁶), and dense multi-GPU (shard, shard+chunk), all at recipe=1×1, to answer whether split-1GPU reaches larger (D, χ) than dense-2GPU and where each wall is.

**Architecture:** A path-dispatching probe (`tests/_frontier_grad_probe.py`) wraps the two production AD entry points — `ctm_energy_implicit` (dense, recipe=1×1, optionally `device_mesh`-sharded and `ctm_chunk_size`-chunked) and `ctm_energy_split_implicit` (split, single-GPU) — behind one `jax.value_and_grad`, reusing the trusted rung-2 probe's 1-site iPEPS + gate + well-conditioned init so peaks are comparable to the shard-reach benchmark. A CLI harness (`examples/bench_ctm_frontier_grad.py`) runs one (path, D, χ) per process and reports per-device peak, distinguishing the two orthogonal walls (divisibility SKIP vs memory OOM). A GPU sweep on GPUs 1,2 fills a findings doc.

**Tech Stack:** Python, JAX (x64, `jax.value_and_grad`, `memory_stats()["peak_bytes_in_use"]`), Tenax `DenseTensor` / `ctm_energy_implicit` / `ctm_energy_split_implicit` / `build_ctm_mesh`, pytest (`-m core`), 2× A100 (GPUs 1,2).

**Spec:** `docs/superpowers/specs/2026-07-01-632-largeD-largechi-multigpu-frontier-design.md`

## Global Constraints

- **No `src/tenax/**` change.** Only `tests/`, `examples/`, `docs/` are touched. (`tests/conftest.py` marker-dict edits are test-infra, allowed.)
- **Branch/worktree:** `bench/632-frontier-multigpu`, worktree `/home/yjkao/tenax-632-frontier`, based on `origin/main` f9b8f6e (#668 — has `ctm_chunk_size` + `_shard_a`-in-1×1). Do all work here; leave `/home/yjkao/tenax` untouched.
- **Recipe fixed at `"1x1"`** for every dense config (split is intrinsically 1×1).
- **x64 always** (`jax.config.update("jax_enable_x64", True)`).
- **One (path, D, χ) per process** for a clean cumulative per-device peak; `XLA_PYTHON_CLIENT_PREALLOCATE=false` on GPU runs.
- **State:** 1-site dense iPEPS, well-conditioned, gate `diag(0.25,-0.25,-0.25,0.25).reshape(2,2,2,2)` — reuse `_rung2_grad_probe._indices` / `_init_data` verbatim.
- **Two orthogonal walls:** divisibility `D² % N == 0` (SKIP, decided before running) vs memory OOM (FAILED at run time). Never conflate them.
- **Commit messages end with:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Path-dispatching frontier probe (dense + split → value_and_grad)

**Files:**
- Create: `tests/_frontier_grad_probe.py`
- Create: `tests/test_frontier_probe.py`
- Modify: `tests/conftest.py` (add `"test_frontier_probe.py": "core"` to `_FILE_MARKERS`)

**Interfaces:**
- Consumes: `_rung2_grad_probe._indices(D) -> tuple[TensorIndex, ...]`, `_rung2_grad_probe._init_data(D, seed, well_conditioned) -> jax.Array (D,D,D,D,2)`; `tenax.algorithms._ctm_energy_ad.ctm_energy_implicit`; `tenax.algorithms._split_ctm_energy_ad.ctm_energy_split_implicit`; `tenax.algorithms._ctm_tensor_convergence.SINGLE_SITE_NEIGHBORS`; `tenax.core.tensor.DenseTensor`.
- Produces: `frontier_energy_and_grad(*, path, D, chi, chi_I=None, device_mesh=None, ctm_chunk_size=None, seed=0, well_conditioned=True, max_iter=30) -> tuple[float, np.ndarray]`. `path ∈ {"dense","split"}`. Raises `ValueError` on an unknown path, or on `device_mesh`/`ctm_chunk_size` passed to `path="split"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frontier_probe.py
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from _frontier_grad_probe import frontier_energy_and_grad


@pytest.mark.parametrize("path", ["dense", "split"])
def test_frontier_probe_finite(path):
    e, g = frontier_energy_and_grad(path=path, D=2, chi=6, max_iter=15)
    assert np.isfinite(e), e
    assert np.all(np.isfinite(g)), path
    assert g.shape == (2, 2, 2, 2, 2), g.shape


def test_frontier_split_rejects_mesh():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="split", D=2, chi=6, device_mesh=object())


def test_frontier_split_rejects_chunk():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="split", D=2, chi=6, ctm_chunk_size=4)


def test_frontier_unknown_path():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="nope", D=2, chi=6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/yjkao/tenax-632-frontier && JAX_PLATFORMS=cpu uv run pytest tests/test_frontier_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named '_frontier_grad_probe'`.

- [ ] **Step 3: Write the probe module**

```python
# tests/_frontier_grad_probe.py
"""Frontier benchmark probe: value_and_grad of the CTM-AD energy across paths.

Dispatches the dense (``ctm_energy_implicit``, recipe="1x1", optionally sharded
+ chunked) and split (``ctm_energy_split_implicit``, single-GPU, chi^2 * D^4)
paths to a common ``jax.value_and_grad`` for the large-D x large-chi multi-GPU
frontier study (phase 1, #632). Reuses the rung-2 probe's 1-site dense iPEPS +
gate + well-conditioned init so per-device peaks are directly comparable to the
shard-reach benchmark.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._split_ctm_energy_ad import ctm_energy_split_implicit
from tenax.core.tensor import DenseTensor

from _rung2_grad_probe import _indices, _init_data  # pristine helpers (do not edit)

_HEISENBERG = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)


def frontier_energy_and_grad(
    *,
    path,
    D,
    chi,
    chi_I=None,
    device_mesh=None,
    ctm_chunk_size=None,
    seed=0,
    well_conditioned=True,
    max_iter=30,
):
    """Return (energy: float, grad: (D,D,D,D,2) array) of one value_and_grad step.

    path="dense": ctm_energy_implicit(recipe="1x1", device_mesh=..., ctm_chunk_size=...)
    path="split": ctm_energy_split_implicit(chi_I=chi_I or chi)  # single-GPU only
    """
    idx = _indices(D)
    data0 = _init_data(D, seed, well_conditioned)

    if path == "dense":

        def loss(data):
            A = DenseTensor(data, idx)
            return ctm_energy_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _HEISENBERG,
                chi=chi,
                max_iter=max_iter,
                conv_tol=1e-10,
                forward_gauge="phase",
                adjoint_method="fixed_point",
                recipe="1x1",
                device_mesh=device_mesh,
                ctm_chunk_size=ctm_chunk_size,
            )

    elif path == "split":
        if device_mesh is not None:
            raise ValueError("split path is single-GPU only (no device_mesh)")
        if ctm_chunk_size is not None:
            raise ValueError("split path does not support ctm_chunk_size")

        def loss(data):
            A = DenseTensor(data, idx)
            return ctm_energy_split_implicit(
                {(0, 0): A},
                SINGLE_SITE_NEIGHBORS,
                _HEISENBERG,
                chi=chi,
                chi_I=chi_I or chi,
                max_iter=max_iter,
                conv_tol=1e-10,
            )

    else:
        raise ValueError(f"unknown path {path!r} (expected 'dense' or 'split')")

    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/yjkao/tenax-632-frontier && JAX_PLATFORMS=cpu uv run pytest tests/test_frontier_probe.py -q`
Expected: PASS (4 tests: 2 parametrized finite + 3 guard raises — note `test_frontier_probe_finite` yields 2).

- [ ] **Step 5: Mark the test file `core` in conftest**

In `tests/conftest.py`, inside the `_FILE_MARKERS` dict, next to the `test_ctm_chunk_backward_grad.py` entry (near the `#632 Increment` comments), add:

```python
    # #632 frontier benchmark (phase 1): tiny D=2 CPU value_and_grad finite +
    # path-guard checks. Fast; core so CI required checks run them.
    "test_frontier_probe.py": "core",
```

- [ ] **Step 6: Confirm the marker + core collection**

Run: `cd /home/yjkao/tenax-632-frontier && uv run pytest tests/test_frontier_probe.py -m core --collect-only -q`
Expected: all 5 test items collected under `-m core` (none deselected).

- [ ] **Step 7: Commit**

```bash
cd /home/yjkao/tenax-632-frontier
git add tests/_frontier_grad_probe.py tests/test_frontier_probe.py tests/conftest.py
git commit -m "$(printf 'feat(#632): frontier value_and_grad probe (dense/split path dispatch)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: CLI harness + divisibility guard (the two-walls reporting)

**Files:**
- Create: `examples/bench_ctm_frontier_grad.py`
- Create: `tests/test_frontier_bench_guard.py`
- Modify: `tests/conftest.py` (add `"test_frontier_bench_guard.py": "core"`)

**Interfaces:**
- Consumes: `frontier_energy_and_grad` (Task 1); `tenax.algorithms.ctm_sharding.build_ctm_mesh`.
- Produces: `skip_reason(D, mesh_n, shard) -> str | None` (pure; SKIP reason iff a sharded config is un-shardable), `peak_gb() -> float`, and a `main()` CLI. `skip_reason` and `peak_gb` are importable **without** importing JAX (JAX/tenax imports live inside `main`/`peak_gb`).

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_frontier_bench_guard.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from bench_ctm_frontier_grad import skip_reason


def test_skip_divisibility_n2():
    # N=2: all even D^2 shard; odd D^2 does not.
    assert skip_reason(10, 2, True) is None   # 100 % 2 == 0
    assert skip_reason(12, 2, True) is None   # 144 % 2 == 0
    assert skip_reason(11, 2, True) is not None  # 121 % 2 != 0


def test_skip_divisibility_n3():
    # N=3: 144 shards; 64 and 100 do not.
    assert skip_reason(12, 3, True) is None      # 144 % 3 == 0
    assert skip_reason(8, 3, True) is not None   # 64 % 3 != 0
    assert skip_reason(10, 3, True) is not None  # 100 % 3 != 0


def test_no_shard_never_skips():
    assert skip_reason(11, 2, False) is None
    assert skip_reason(8, 1, False) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/yjkao/tenax-632-frontier && uv run pytest tests/test_frontier_bench_guard.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench_ctm_frontier_grad'`.

- [ ] **Step 3: Write the harness**

```python
# examples/bench_ctm_frontier_grad.py
"""#632 large-D x large-chi multi-GPU frontier benchmark — value_and_grad reach.

Per-device peak of ONE value_and_grad step across (all at recipe=1x1):
  --path split               1-GPU  ctm_energy_split_implicit (chi^2 * D^4)
  --path dense               1-GPU  ctm_energy_implicit recipe=1x1 (chi^2 * D^6)
  --path dense --shard       N-GPU  + GSPMD device_mesh
  --path dense --shard --chunk K   N-GPU  + device_mesh + ctm_chunk_size

One (path, D, chi) per process for a clean cumulative peak (shard-reach method).

Two orthogonal walls (see the design spec):
  divisibility : --shard needs D^2 % mesh_n == 0  -> SKIP (decided before running)
  memory (OOM) : a shardable config can still exceed RAM -> FAILED(RESOURCE_EXHAUSTED)

Usage (PREALLOCATE=false for a faithful peak; one config per process):
    CUDA_VISIBLE_DEVICES=1   XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path split --D 10 --chi 48
    CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/bench_ctm_frontier_grad.py --path dense --D 10 --chi 24 --shard --chunk 8
"""
import argparse
import os
import sys
import time


def skip_reason(D, mesh_n, shard):
    """SKIP reason iff a sharded config is un-shardable (D^2 not divisible by N)."""
    if shard and (D * D) % mesh_n != 0:
        return f"D^2={D * D} % mesh_n={mesh_n} != 0"
    return None


def peak_gb():
    import jax

    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["dense", "split"], required=True)
    ap.add_argument("--D", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--chi-I", type=int, default=None, dest="chi_I")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--chunk", type=int, default=0, help="ctm_chunk_size (0=off; dense only)")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    from tenax.algorithms.ctm_sharding import build_ctm_mesh

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from _frontier_grad_probe import frontier_energy_and_grad

    n = jax.device_count()
    shard = args.shard and args.path == "dense"
    if args.shard and args.path == "split":
        print("# WARN: --shard ignored on split path (single-GPU only)")
    chunk = args.chunk if (args.chunk > 0 and args.path == "dense") else None
    if args.chunk > 0 and args.path == "split":
        print("# WARN: --chunk ignored on split path")
    mesh = build_ctm_mesh() if shard else None
    mesh_n = n if shard else 1
    print(
        f"# path={args.path} devices={n} shard={shard} mesh_n={mesh_n} "
        f"chi={args.chi} chi_I={args.chi_I} chunk={chunk} max_iter={args.max_iter} "
        f"recipe=1x1 x64=True"
    )
    for D in args.D:
        reason = skip_reason(D, mesh_n, shard)
        if reason is not None:
            print(f"path={args.path} D={D} chi={args.chi}: SKIP ({reason})")
            continue
        t0 = time.perf_counter()
        try:
            e, g = frontier_energy_and_grad(
                path=args.path,
                D=D,
                chi=args.chi,
                chi_I=args.chi_I,
                device_mesh=mesh,
                ctm_chunk_size=chunk,
                seed=args.seed,
                well_conditioned=True,
                max_iter=args.max_iter,
            )
            gnorm = float((g ** 2).sum() ** 0.5)
            dt = time.perf_counter() - t0
            print(
                f"path={args.path} D={D} chi={args.chi} OK  E={e:.6f}  |g|={gnorm:.3e}  "
                f"per_device_peak={peak_gb():.2f} GB  wall={dt:.1f}s"
            )
        except Exception as ex:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(
                f"path={args.path} D={D} chi={args.chi} "
                f"FAILED({type(ex).__name__}: {str(ex)[:110]})  wall={dt:.1f}s"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the guard test to verify it passes**

Run: `cd /home/yjkao/tenax-632-frontier && uv run pytest tests/test_frontier_bench_guard.py -q`
Expected: PASS (3 tests). Fast — no JAX imported (guard + module import are JAX-free).

- [ ] **Step 5: CPU smoke of the full CLI (both paths run end-to-end)**

Run:
```bash
cd /home/yjkao/tenax-632-frontier
JAX_PLATFORMS=cpu uv run python examples/bench_ctm_frontier_grad.py --path split --D 2 --chi 6 --max-iter 10
JAX_PLATFORMS=cpu uv run python examples/bench_ctm_frontier_grad.py --path dense --D 2 --chi 6 --max-iter 10
```
Expected: each prints one `path=... D=2 chi=6 OK  E=...  |g|=...  per_device_peak=... GB  wall=...s` line (peak may be `nan` on CPU — acceptable; the smoke checks wiring, not memory).

- [ ] **Step 6: Mark the guard test `core` in conftest**

In `tests/conftest.py` `_FILE_MARKERS`, directly after the `test_frontier_probe.py` entry from Task 1, add:

```python
    "test_frontier_bench_guard.py": "core",
```

- [ ] **Step 7: Commit**

```bash
cd /home/yjkao/tenax-632-frontier
git add examples/bench_ctm_frontier_grad.py tests/test_frontier_bench_guard.py tests/conftest.py
git commit -m "$(printf 'feat(#632): frontier CLI harness + divisibility guard (two-walls reporting)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: GPU frontier sweep on GPUs 1,2 (the measurement)

**Files:** Run-only. Capture raw stdout to `/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt`.

This task is measurement, not TDD. Each command is one config/process (clean peak). The χ values below are a **starting grid**; for each config, step χ up until it OOMs to find that config's reach (the frontier), and step down one if the first χ already OOMs.

- [ ] **Step 1: Harness validation anchor (reproduce a trusted number)**

Run:
```bash
cd /home/yjkao/tenax-632-frontier
SCRATCH=/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_frontier_grad.py --path dense --D 8 --chi 24 | tee -a "$SCRATCH"
```
Expected: `path=dense D=8 chi=24 OK ... per_device_peak≈? GB`. Record the number. (This is the dense-1×1 anchor; note it may differ from shard-reach's recipe=2×2 10.66 GB — recipe differs. If it OOMs unexpectedly at D=8 χ=24 on an 80 GB card, stop and investigate before the full sweep.)

- [ ] **Step 2: Config 1 — split-CTM, 1-GPU (push χ high)**

Run (GPU 1; step χ up per D until OOM):
```bash
cd /home/yjkao/tenax-632-frontier
SCRATCH=/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt
for D in 6 8 10 12; do for CHI in 24 32 48 64 96; do
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python examples/bench_ctm_frontier_grad.py --path split --D $D --chi $CHI | tee -a "$SCRATCH"
done; done
```
Expected: `OK` lines with per-device peak until a χ OOMs (`FAILED(...RESOURCE_EXHAUSTED...)`); the last `OK` χ per D is split's reach at that D.

- [ ] **Step 3: Config 2 — dense, 1-GPU baseline**

Run (GPU 1):
```bash
cd /home/yjkao/tenax-632-frontier
SCRATCH=/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt
for D in 6 8 10 12; do for CHI in 16 24 32; do
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python examples/bench_ctm_frontier_grad.py --path dense --D $D --chi $CHI | tee -a "$SCRATCH"
done; done
```
Expected: dense OOMs at far smaller χ than split (χ²·D⁶ vs χ²·D⁴).

- [ ] **Step 4: Config 3 — dense, 2-GPU shard**

Run (GPUs 1,2):
```bash
cd /home/yjkao/tenax-632-frontier
SCRATCH=/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt
for D in 6 8 10 12; do for CHI in 16 24 32; do
  CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python examples/bench_ctm_frontier_grad.py --path dense --D $D --chi $CHI --shard | tee -a "$SCRATCH"
done; done
```
Expected: `OK`/`FAILED` per cell; SKIP never fires (all D² even → divisible by 2). Compare peak vs config 2 for the shard relief.

- [ ] **Step 5: Config 4 — dense, 2-GPU shard + chunk**

Run (GPUs 1,2; chunk K such that χ%K==0, e.g. K=8 for χ∈{16,24,32}):
```bash
cd /home/yjkao/tenax-632-frontier
SCRATCH=/tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt
for D in 6 8 10 12; do for CHI in 16 24 32; do
  CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python examples/bench_ctm_frontier_grad.py --path dense --D $D --chi $CHI --shard --chunk 8 | tee -a "$SCRATCH"
done; done
```
Expected: per-device peak vs config 3 (chunk may not help the `value_and_grad` peak — Inc2 found backward chunk is NO-GO; forward chunk is below the convergence waterline until large D. This measures whether that holds in the pipeline). Note in the findings whichever way it goes.

- [ ] **Step 6: Sanity-scan the raw file**

Run: `cat /tmp/claude-1007/-home-yjkao-tenax/25ff8739-c4e3-4407-adf4-d89bb2c91531/scratchpad/frontier_scan.txt`
Confirm every line is a `# header`, `OK`, `SKIP`, or `FAILED` line (no tracebacks leaked). If a config produced a Python traceback rather than a `FAILED(...)` line, the probe raised something the harness did not catch — investigate before writing findings.

---

### Task 4: Findings handoff + verdict

**Files:**
- Create: `docs/superpowers/handoffs/2026-07-01-632-largeD-largechi-multigpu-frontier-findings.md`

- [ ] **Step 1: Write the findings doc from the scratch file**

Fill this template with the measured numbers from Task 3 (`frontier_scan.txt`):

```markdown
# Large-D × large-χ multi-GPU frontier — phase 1 findings

**Date:** 2026-07-01
**Branch:** `bench/632-frontier-multigpu` (off #668)
**Harness:** `examples/bench_ctm_frontier_grad.py` + `tests/_frontier_grad_probe.py`
**Hardware:** 2× A100-80GB (GPUs 1,2), f64, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, one config/process.
**Spec:** `docs/superpowers/specs/2026-07-01-632-largeD-largechi-multigpu-frontier-design.md`
**Verdict: <split-1GPU vs dense-2GPU — which reaches larger (D, χ); one line>**

## Harness anchor
dense-1×1 D=8 χ=24 = <…> GB (vs shard-reach recipe=2×2 10.66 GB — recipe differs).

## Frontier (per-device peak GB; OK / OOM / SKIP), recipe=1×1, value_and_grad

### split-CTM, 1-GPU (χ²·D⁴)
| D | χ=24 | 32 | 48 | 64 | 96 | reach (max χ OK) |
|---|---|---|---|---|---|---|
| 6 | … | | | | | |
| 8 | … | | | | | |
| 10 | … | | | | | |
| 12 | … | | | | | |

### dense, 1-GPU / 2-GPU shard / 2-GPU shard+chunk (χ²·D⁶)
| D | χ | dense 1-GPU | dense 2-GPU shard | dense 2-GPU shard+chunk |
|---|---|---|---|---|
| 6 | 16 | … | … | … |
| … | | | | |

## Reads
1. **Split-1GPU vs dense-2GPU frontier:** <which reaches larger (D, χ); by how much>.
2. **Shard relief in the pipeline:** dense 2-GPU vs 1-GPU factor at each (D, χ) — <χ-gated? D-fading? matches shard-reach ~1.2–2×?>.
3. **Chunk in value_and_grad:** <did shard+chunk beat shard-only, or confirm Inc2's forward-below-waterline / backward-NO-GO?>.
4. **The composition gap made concrete:** split has no multi-GPU column; its 1-GPU reach is <above/below> dense's best 2-GPU reach.

## Recommendation (feeds phase 2)
<Is building split × multi-GPU (thread device_mesh into the split path) worth it —
i.e., is dense-2GPU ever ahead of split-1GPU on the frontier? If split-1GPU
dominates everywhere, multi-GPU is not the large-(D,χ) lever and phase 2 should
target the split single-GPU ceiling (e.g., chi_I<χ, split forward χ²·D⁴ at larger
χ) instead. State the call.>

## Reproduce
<paste the exact command block from the plan Task 3>
```

- [ ] **Step 2: Commit**

```bash
cd /home/yjkao/tenax-632-frontier
git add docs/superpowers/handoffs/2026-07-01-632-largeD-largechi-multigpu-frontier-findings.md
git commit -m "$(printf 'docs(#632): large-D x large-chi multi-GPU frontier findings (phase 1)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

- [ ] **Step 3: Full core-bucket regression (no CI break)**

Run: `cd /home/yjkao/tenax-632-frontier && uv run pytest -m core -q`
Expected: PASS — the two new test files are collected and green; nothing else regresses (no `src/` change).

---

## Self-Review

**Spec coverage:**
- 4-config matrix (split-1GPU, dense-1GPU, dense-2GPU shard, dense-2GPU shard+chunk), recipe=1×1 → Task 1 probe (dense/split dispatch, `device_mesh`, `ctm_chunk_size`) + Task 3 sweep (all four). ✓
- value_and_grad operation → `jax.value_and_grad` in the probe. ✓
- Two orthogonal walls (divisibility SKIP vs memory OOM) → `skip_reason` (Task 2) + try/except OOM reporting (Task 2 harness); tested in `test_frontier_bench_guard.py`. ✓
- Component 1 (`tests/_frontier_grad_probe.py`) / Component 2 (`examples/bench_ctm_frontier_grad.py`) → Tasks 1 / 2. ✓
- Validation: finite E + finite ‖g‖ both paths → `test_frontier_probe_finite`; split rejects mesh/chunk → guard tests. ✓ (The qualitative split-vs-dense energy approach is an *observation* recorded in the findings Reads, not a CI assertion — avoids a flaky tolerance; consistent with the spec's "not equality" framing.)
- Deliverable findings doc → Task 4. ✓
- No `src/` change; #668 branch base → Global Constraints + worktree. ✓
- Out of scope (4-GPU, 2×2, symmetric, chi_I<χ) → not in any task. ✓

**Placeholder scan:** The findings-doc template `<…>` are data slots filled from `frontier_scan.txt` at Task 4 (measured values unknown until run), not unspecified logic. All code and commands are complete and runnable. The χ grid in Task 3 is an explicit starting grid with a stated reach-search rule (step χ until OOM), not a vague "sweep as needed".

**Type/name consistency:** `frontier_energy_and_grad(path, D, chi, chi_I, device_mesh, ctm_chunk_size, seed, well_conditioned, max_iter)` — identical signature in Task 1 definition, Task 2 harness call, and tests. `skip_reason(D, mesh_n, shard)` — identical in Task 2 definition and `test_frontier_bench_guard.py`. `_indices` / `_init_data` / `SINGLE_SITE_NEIGHBORS` match `_rung2_grad_probe.py` and `_ctm_tensor_convergence.py` (verified on the #668 branch). `ctm_energy_implicit(..., recipe, device_mesh, ctm_chunk_size)` and `ctm_energy_split_implicit(..., chi, chi_I, max_iter, conv_tol)` match the merged signatures (`_ctm_energy_ad.py:367`, `_split_ctm_energy_ad.py:205`).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-632-largeD-largechi-multigpu-frontier.md`.
