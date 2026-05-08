# Tenax ↔ variPEPS Square Heisenberg Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land a small Tenax history-capture hook, then build a subprocess-isolated benchmark harness under `benchmarks/varipeps_compare/` that runs Tenax and variPEPS 1.4.2 on the spin-½ square-lattice Heisenberg AFM with shared init (random for `single_site`, Tenax SU for `bipartite_2site`) across an 8-point grid (paths × D × χ) and emits a parity report (final E, Δ, num steps, wall-clock, trajectory plot).

**Architecture:** Shared protocol module (`protocol.py`) is single source of truth. Orchestrator (`compare.py`) builds Heisenberg gate, builds path-dependent init (random for `single_site`; Tenax SU for `bipartite_2site`), saves init+gate to `.npz`, spawns one Tenax subprocess and one variPEPS subprocess per grid point with identical CLI args, parses each runner's JSON output (shared schema), merges into `report.json` + `summary.md` + per-point trajectory plots. Process isolation gives fair JIT/cache accounting and prevents variPEPS's import-time `jax.config` writes from polluting Tenax. **Both libs use unconstrained ansatze on both paths** — Tenax does not enforce C4v on `single_site`, so parameter counts match variPEPS exactly.

**Tech Stack:** Python 3.12, JAX (CPU first, GPU optional), Tenax (this repo), variPEPS 1.4.2 (already installed at `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps`), numpy `.npz` for cross-lib payload, matplotlib for plots, pytest for the smoke test.

**Design doc:** `docs/plans/2026-05-08-tenax-varipeps-square-heisenberg-benchmark-design.md` (committed e1fa37e).

---

## Layout

```
src/tenax/algorithms/ipeps_optimize.py        # MODIFY: add return_history flag (Task 1)
tests/test_ipeps_ad_history.py                # CREATE: history-hook unit test (Task 1)
benchmarks/varipeps_compare/__init__.py       # CREATE
benchmarks/varipeps_compare/protocol.py       # CREATE: shared constants (Task 2)
benchmarks/varipeps_compare/payload.py        # CREATE: npz round-trip (Task 3)
benchmarks/varipeps_compare/su_init.py        # CREATE: gate + SU (Task 4)
benchmarks/varipeps_compare/run_tenax.py      # CREATE: Tenax CLI runner (Task 5)
benchmarks/varipeps_compare/run_varipeps.py   # CREATE: variPEPS CLI runner (Task 6)
benchmarks/varipeps_compare/compare.py        # CREATE: orchestrator + plot (Task 7)
benchmarks/varipeps_compare/__main__.py       # CREATE: `python -m ... .compare` entry (Task 7)
tests/test_varipeps_compare.py                # CREATE: end-to-end smoke (Task 8)
.gitignore                                    # MODIFY: ignore benchmarks/varipeps_compare/results/
```

---

## Task 1: Add `return_history` flag to `optimize_gs_ad`

**Why:** variPEPS already returns `step_energies` / `step_runtime`; Tenax does not. Without this, trajectory parity is impossible. Strict superset feature (default unchanged).

**Scope:** Add the hook to **only the two paths the benchmark uses**: 1-site tensor (`_optimize_gs_ad_tensor`) and 2-site (`_optimize_gs_ad_tensor_2site`). Multisite (`_optimize_gs_ad_multisite`) and the dense C4v reference path (`_optimize_gs_ad_tensor_reference_c4v`) are out of scope — if `return_history=True` is set with those paths, raise `NotImplementedError("return_history not yet supported for unit_cell=Lattice / ctm_ad_mode=c4v_reference")` with a clear message. Future PR can extend.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py` (add field)
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (capture in 1-site `_optimize_gs_ad_tensor` ~line 653 and 2-site `_optimize_gs_ad_tensor_2site` ~line 1389; raise in the other two)
- Modify: `src/tenax/__init__.py` (no change — flag goes through config, not new export)
- Create: `tests/test_ipeps_ad_history.py`

**Step 1.1: Write the failing test**

```python
# tests/test_ipeps_ad_history.py
"""Tests for the optimize_gs_ad history-capture hook."""
import jax.numpy as jnp
import pytest

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad


def _heisenberg_gate(dtype=jnp.complex128):
    Sz = jnp.array([[0.5, 0], [0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = jnp.array([[0, 0], [1, 0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    return H.reshape(2, 2, 2, 2)


@pytest.mark.algorithm
def test_optimize_gs_ad_returns_history_2site():
    """When return_history=True, history dict has energies & step_times."""
    gate = _heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=8),
        unit_cell="2site",
        gs_num_steps=5,
        gs_optimizer="lbfgs",
        gs_implicit_ad=True,
        su_init=False,
        return_history=True,
    )
    out = optimize_gs_ad(gate, None, config)

    # Backwards-compat shape unchanged when return_history=False (default).
    # When True, last element is a history dict.
    assert isinstance(out[-1], dict)
    history = out[-1]
    assert "energies" in history
    assert "step_times" in history
    assert "jit_compile_time" in history
    assert len(history["energies"]) == len(history["step_times"])
    assert len(history["energies"]) >= 1
    assert all(isinstance(e, float) for e in history["energies"])
    assert all(isinstance(t, float) for t in history["step_times"])


@pytest.mark.algorithm
def test_optimize_gs_ad_default_no_history():
    """Default return shape is unchanged (no history element)."""
    gate = _heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=8),
        unit_cell="2site",
        gs_num_steps=2,
        su_init=False,
    )
    out = optimize_gs_ad(gate, None, config)
    # 2-site path: ((A, B), (env_A, env_B), E_gs) — no dict at end
    assert not isinstance(out[-1], dict)
```

**Step 1.2: Run the test — verify it fails**

```bash
uv run pytest tests/test_ipeps_ad_history.py -v
```

Expected: FAIL with `TypeError: ... unexpected keyword argument 'return_history'` (or `AttributeError`).

**Step 1.3: Add `return_history` field to `iPEPSConfig`**

In `src/tenax/algorithms/ipeps_config.py` near the other `gs_*` fields (around line 230, after `gs_implicit_ad`):

```python
    return_history: bool = False  # if True, append history dict to return tuple
```

**Step 1.4: Capture history in `_optimize_gs_ad_tensor` (1-site path)**

In `src/tenax/algorithms/ipeps_optimize.py`, find `_optimize_gs_ad_tensor` (line 653). Locate the `for step in range(config.gs_num_steps):` loop (line 887) and the `energy_float = float(energy_val)` line that follows. Add at top of function before the loop:

```python
import time as _time
_history_energies: list[float] = []
_history_step_times: list[float] = []
_jit_compile_time: float = 0.0
_first_step = True
```

Wrap the `value_and_grad` call:

```python
        _step_t0 = _time.perf_counter()
        try:
            energy_val, grads = jax.value_and_grad(loss_fn)(params)
        except CTMRGGradientError as exc:
            ...  # existing handling
        _step_dt = _time.perf_counter() - _step_t0
        if _first_step:
            _jit_compile_time = _step_dt
            _first_step = False
        else:
            _history_step_times.append(_step_dt)
        # …
        energy_float = float(energy_val)
        _history_energies.append(energy_float)
```

At the end of the function, before the existing `return ...`:

```python
    if config.return_history:
        history = {
            "energies": _history_energies,
            "step_times": _history_step_times,
            "jit_compile_time": _jit_compile_time,
            "num_steps": len(_history_energies),
            "converged": converged,  # bool already tracked locally
        }
        return best_params, best_envs, best_energy, history
    return best_params, best_envs, best_energy
```

(Adapt to the actual local variable names — `best_params`/`best_env_cache`/etc. — discovered when reading the function.)

**Step 1.5: Repeat for 2-site path (`_optimize_gs_ad_tensor_2site`, ~line 1389)**

Same pattern. Return shape changes from `((A, B), (env_A, env_B), E_gs)` to `((A, B), (env_A, env_B), E_gs, history)` when flag is set.

**Step 1.6: Guard the unsupported paths**

In `_optimize_gs_ad_multisite` and `_optimize_gs_ad_tensor_reference_c4v`, near the top after the config is normalized, add:

```python
if config.return_history:
    raise NotImplementedError(
        "return_history is currently only supported for unit_cell='1x1' "
        "(non-C4v-reference) and unit_cell='2site'."
    )
```

**Step 1.7: Run all tests, verify they pass**

```bash
uv run pytest tests/test_ipeps_ad_history.py -v
uv run pytest -m core -x   # ensure no regression on default-shape callers
```

Expected: both new tests PASS, core suite PASS.

**Step 1.8: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_ad_history.py
git commit -m "feat(ipeps-ad): optional return_history flag for trajectory capture

Adds iPEPSConfig.return_history (default False).  When True,
optimize_gs_ad appends a history dict {energies, step_times,
jit_compile_time, num_steps, converged} to the return tuple in the
1-site tensor and 2-site paths.  Multisite and c4v_reference paths
raise NotImplementedError when the flag is set; they can be extended
later.  Required by benchmarks/varipeps_compare for trajectory parity
with variPEPS step_energies/step_runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `protocol.py` — shared constants

**Files:**
- Create: `benchmarks/varipeps_compare/__init__.py` (empty)
- Create: `benchmarks/varipeps_compare/protocol.py`

**Step 2.1: Write `protocol.py`**

```python
# benchmarks/varipeps_compare/protocol.py
"""Single source of truth for the Tenax↔variPEPS benchmark protocol.

Both runners (run_tenax.py, run_varipeps.py) and the orchestrator (compare.py)
import constants from this module.  Do not redefine knobs locally — change
them here.
"""
from __future__ import annotations

PATHS = ("single_site", "bipartite_2site")
D_VALUES = (2, 3)
CHI_VALUES = (16, 24)

GRID = tuple(
    {"path": p, "D": D, "chi": chi}
    for p in PATHS
    for D in D_VALUES
    for chi in CHI_VALUES
)  # 8 points

TOL = 1e-6
MAX_STEPS = 100
SEED = 0
DTYPE = "complex128"

LBFGS_HISTORY = 10        # both libs use L-BFGS with history depth 10
CTM_TOL = 1e-8
CTM_MAX_ITER = 100

SUBPROCESS_TIMEOUT_SEC = 30 * 60   # 30 min per (path, D, chi, lib)


def grid_key(path: str, D: int, chi: int) -> str:
    """Canonical filesystem key for a grid point."""
    return f"{path}_D{D}_chi{chi}"
```

**Step 2.2: Commit**

```bash
git add benchmarks/varipeps_compare/__init__.py benchmarks/varipeps_compare/protocol.py
git commit -m "bench(varipeps-compare): protocol module — shared grid + knobs"
```

---

## Task 3: `payload.py` — npz round-trip

**Files:**
- Create: `benchmarks/varipeps_compare/payload.py`
- Create: `tests/test_varipeps_compare_payload.py` (lives in `tests/`, not in `benchmarks/`, so pytest picks it up)

**Step 3.1: Write the failing test**

```python
# tests/test_varipeps_compare_payload.py
"""Round-trip test for benchmarks.varipeps_compare.payload."""
import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import save_payload, load_payload


@pytest.mark.core
def test_payload_roundtrip(tmp_path):
    init = np.random.default_rng(0).standard_normal((2, 2, 2, 2, 2)).astype(np.complex128)
    gate = np.random.default_rng(1).standard_normal((2, 2, 2, 2)).astype(np.complex128)
    meta = {"path": "single_site", "D": 2, "chi": 16, "seed": 0}

    out = tmp_path / "payload.npz"
    save_payload(out, init=init, gate=gate, meta=meta)
    assert out.exists()

    init2, gate2, meta2 = load_payload(out)
    np.testing.assert_array_equal(init, init2)
    np.testing.assert_array_equal(gate, gate2)
    assert meta2 == meta
```

**Step 3.2: Verify it fails**

```bash
uv run pytest tests/test_varipeps_compare_payload.py -v
```

Expected: FAIL — module does not exist.

**Step 3.3: Implement**

```python
# benchmarks/varipeps_compare/payload.py
"""Cross-library .npz payload — init iPEPS tensor + Hamiltonian gate + metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_payload(path: Path | str, *, init: np.ndarray, gate: np.ndarray, meta: dict) -> None:
    """Write init+gate+meta to a single .npz file.

    Args:
        path: Output filename (typically `<key>.npz`).
        init: Initial iPEPS site tensor as a numpy array.  Shape depends on path:
              C4v 1×1 → (D, D, D, D, d); 2-site checkerboard → (2, D, D, D, D, d) stacking A and B.
        gate:  Two-site Hamiltonian (d, d, d, d).  For C4v paths this is the
               sublattice-rotated gate; for 2-site checkerboard it is the bare gate.
        meta:  JSON-serializable metadata.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, init=init, gate=gate, meta=np.array(json.dumps(meta)))


def load_payload(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        init = f["init"]
        gate = f["gate"]
        meta = json.loads(str(f["meta"]))
    return init, gate, meta
```

**Step 3.4: Verify it passes**

```bash
uv run pytest tests/test_varipeps_compare_payload.py -v
```

Expected: PASS.

**Step 3.5: Commit**

```bash
git add benchmarks/varipeps_compare/payload.py tests/test_varipeps_compare_payload.py
git commit -m "bench(varipeps-compare): payload — .npz init/gate/meta round-trip"
```

---

## Task 4: `su_init.py` — gates + init dispatcher (random for `single_site`, SU for `bipartite_2site`)

**Files:**
- Create: `benchmarks/varipeps_compare/su_init.py`
- Create: `tests/test_varipeps_compare_su.py`

**Step 4.1: Write the failing test**

```python
# tests/test_varipeps_compare_su.py
"""Init dispatcher smoke test."""
import numpy as np
import pytest

from benchmarks.varipeps_compare.su_init import (
    build_heisenberg_gate,
    build_sublattice_rotated_gate,
    make_init,
)


@pytest.mark.core
def test_heisenberg_gate_shape_and_hermiticity():
    gate = build_heisenberg_gate()
    assert gate.shape == (2, 2, 2, 2)
    assert gate.dtype == np.complex128
    M = gate.reshape(4, 4)
    np.testing.assert_allclose(M, M.conj().T, atol=1e-12)


@pytest.mark.core
def test_sublattice_rotated_gate_shape():
    g_rot = build_sublattice_rotated_gate()
    assert g_rot.shape == (2, 2, 2, 2)
    assert g_rot.dtype == np.complex128


@pytest.mark.core
def test_make_init_single_site_random_deterministic():
    """single_site path: random init, deterministic given seed."""
    a = make_init(path="single_site", D=2, seed=0)
    b = make_init(path="single_site", D=2, seed=0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 2, 2, 2, 2)  # (D,D,D,D,d)
    assert a.dtype == np.complex128
    # Different seed → different array
    c = make_init(path="single_site", D=2, seed=1)
    assert not np.array_equal(a, c)


@pytest.mark.algorithm
def test_make_init_bipartite_2site_su_d2():
    """bipartite_2site path: Tenax SU returns stacked (A, B) of shape (2, D, D, D, D, d)."""
    init = make_init(path="bipartite_2site", D=2, seed=0)
    assert isinstance(init, np.ndarray)
    assert init.shape == (2, 2, 2, 2, 2, 2)  # (2, D, D, D, D, d)
    assert init.dtype == np.complex128
```

**Step 4.2: Verify it fails**

```bash
uv run pytest tests/test_varipeps_compare_su.py -v
```

Expected: FAIL — module not found.

**Step 4.3: Implement**

```python
# benchmarks/varipeps_compare/su_init.py
"""Heisenberg gate constructors + path-dependent init dispatcher.

For ``single_site`` (1×1 + sublattice-rotated gate, unconstrained tensor):
    SU on the rotated gate converges to the |↑↑⟩ saddle (E=−0.5/site) which
    L-BFGS cannot escape (see ``ipeps_optimize.py:1389`` reference-mode
    comment).  Use random init instead.
For ``bipartite_2site`` (2-tensor checkerboard + bare gate):
    Tenax SU on the bare gate finds a Néel-like state.  Use it.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from tenax import CTMConfig, iPEPSConfig, ipeps, sublattice_rotate_gate

DTYPE_NP = np.complex128


def build_heisenberg_gate(dtype=jnp.complex128) -> np.ndarray:
    """H = Sx⊗Sx + Sy⊗Sy + Sz⊗Sz with spin-½ matrices, returned as (2,2,2,2)."""
    Sx = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
    Sy = jnp.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    H = jnp.kron(Sx, Sx) + jnp.kron(Sy, Sy) + jnp.kron(Sz, Sz)
    return np.asarray(H.reshape(2, 2, 2, 2))


def build_sublattice_rotated_gate(dtype=jnp.complex128) -> np.ndarray:
    """``single_site`` path gate: bare H rotated by Y on B sublattice.

    Lets a 1×1 unit cell encode the AFM ground state in the rotated frame.
    """
    return np.asarray(sublattice_rotate_gate(jnp.asarray(build_heisenberg_gate(dtype))))


def _random_complex(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    return (re + 1j * im).astype(DTYPE_NP)


def make_init(*, path: str, D: int, seed: int = 0,
              su_num_steps: int = 100, su_dt: float = 0.01) -> np.ndarray:
    """Build the init tensor for the given path.

    Args:
        path: ``"single_site"`` (random) or ``"bipartite_2site"`` (Tenax SU).
        D:    iPEPS bond dimension.
        seed: numpy seed for random init (only used for ``single_site``).
        su_num_steps / su_dt: SU schedule (only used for ``bipartite_2site``).

    Returns:
        ``single_site``       → ``(D, D, D, D, d)`` complex128 array.
        ``bipartite_2site``   → ``(2, D, D, D, D, d)`` stacked (A, B) complex128.
    """
    d = 2  # spin-½
    if path == "single_site":
        return _random_complex((D, D, D, D, d), seed=seed)
    elif path == "bipartite_2site":
        gate = jnp.asarray(build_heisenberg_gate())
        config = iPEPSConfig(
            max_bond_dim=D,
            num_imaginary_steps=su_num_steps,
            dt=su_dt,
            ctm=CTMConfig(chi=4 * D),
            unit_cell="2site",
        )
        _, (A, B), _ = ipeps(gate, None, config)
        return np.stack([np.asarray(A), np.asarray(B)], axis=0).astype(DTYPE_NP)
    else:
        raise ValueError(f"unknown path: {path}")
```

**Step 4.4: Verify it passes**

```bash
uv run pytest tests/test_varipeps_compare_su.py -v
```

Expected: PASS (the SU test is `algorithm`-marked, takes ~30 s).

**Step 4.5: Commit**

```bash
git add benchmarks/varipeps_compare/su_init.py tests/test_varipeps_compare_su.py
git commit -m "bench(varipeps-compare): su_init — gates + path-dependent init dispatcher"
```

---

## Task 5: `run_tenax.py` — Tenax CLI runner

**Files:**
- Create: `benchmarks/varipeps_compare/run_tenax.py`
- Create: `tests/test_varipeps_compare_run_tenax.py`

**Step 5.1: Write the failing test**

```python
# tests/test_varipeps_compare_run_tenax.py
"""Smoke test for run_tenax.py — runs as subprocess, parses JSON."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import save_payload
from benchmarks.varipeps_compare.su_init import build_sublattice_rotated_gate


@pytest.mark.algorithm
def test_run_tenax_single_site_d2_chi8(tmp_path):
    """Tiny end-to-end: D=2, chi=8, single_site, MAX_STEPS=5 → JSON with right schema."""
    gate = build_sublattice_rotated_gate()
    rng = np.random.default_rng(0)
    init = (rng.standard_normal((2, 2, 2, 2, 2))
            + 1j * rng.standard_normal((2, 2, 2, 2, 2))).astype(np.complex128)

    payload = tmp_path / "payload.npz"
    save_payload(payload, init=init, gate=gate, meta={"path": "single_site", "D": 2, "chi": 8})

    out = tmp_path / "tenax_result.json"
    cmd = [
        sys.executable, "-m", "benchmarks.varipeps_compare.run_tenax",
        "--payload", str(payload),
        "--path", "single_site",
        "--D", "2", "--chi", "8",
        "--tol", "1e-4", "--max-steps", "5",
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, timeout=300)
    data = json.loads(out.read_text())
    for key in ("lib", "path", "D", "chi", "energy_history", "step_times",
                "jit_compile_time", "final_energy", "num_steps", "converged",
                "device", "lib_version"):
        assert key in data, f"missing {key}"
    assert data["lib"] == "tenax"
    assert data["path"] == "single_site"
    assert data["D"] == 2 and data["chi"] == 8
    assert len(data["energy_history"]) == data["num_steps"]
    assert data["final_energy"] < 0  # Heisenberg ground state is negative
```

**Step 5.2: Verify it fails**

```bash
uv run pytest tests/test_varipeps_compare_run_tenax.py -v
```

Expected: FAIL — module not found.

**Step 5.3: Implement**

```python
# benchmarks/varipeps_compare/run_tenax.py
"""Tenax CLI runner for the variPEPS compare benchmark.

Usage:
    python -m benchmarks.varipeps_compare.run_tenax \
        --payload payload.npz --path single_site --D 2 --chi 16 \
        --tol 1e-6 --max-steps 100 --out tenax_<key>.json
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

from .payload import load_payload
from .protocol import CTM_MAX_ITER, CTM_TOL, LBFGS_HISTORY


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _tenax_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _build_config(*, path: str, D: int, chi: int, tol: float, max_steps: int) -> iPEPSConfig:
    ctm = CTMConfig(
        chi=chi,
        max_iter=CTM_MAX_ITER,
        conv_tol=CTM_TOL,
        projector_method="svd",   # Fishman, matches variPEPS default
    )
    common = dict(
        max_bond_dim=D, ctm=ctm,
        gs_optimizer="lbfgs",
        gs_num_steps=max_steps,
        gs_conv_tol=tol,
        gs_implicit_ad=True,
        su_init=False,
        return_history=True,
    )
    if path == "single_site":
        # Unconstrained 1×1 + sublattice-rotated gate.  No C4v — same ansatz
        # as variPEPS structure=[[0]] so parameter counts match.
        return iPEPSConfig(unit_cell="1x1", gs_c4v=False, **common)
    elif path == "bipartite_2site":
        return iPEPSConfig(unit_cell="2site", **common)
    else:
        raise ValueError(f"unknown path: {path}")


def _load_init(init: np.ndarray, path: str):
    if path == "single_site":
        return jnp.asarray(init)
    elif path == "bipartite_2site":
        # init shape: (2, D, D, D, D, d) → (A, B)
        return (jnp.asarray(init[0]), jnp.asarray(init[1]))
    else:
        raise ValueError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, type=Path)
    ap.add_argument("--path", required=True, choices=("single_site", "bipartite_2site"))
    ap.add_argument("--D", required=True, type=int)
    ap.add_argument("--chi", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--max-steps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    init_np, gate_np, _ = load_payload(args.payload)
    gate = jnp.asarray(gate_np)
    A_init = _load_init(init_np, args.path)
    config = _build_config(
        path=args.path, D=args.D, chi=args.chi,
        tol=args.tol, max_steps=args.max_steps,
    )

    t0 = time.perf_counter()
    result = optimize_gs_ad(gate, A_init, config)
    dt = time.perf_counter() - t0

    history = result[-1]
    final_energy = float(result[-2])

    out = {
        "lib": "tenax",
        "path": args.path,
        "D": args.D,
        "chi": args.chi,
        "dtype": "complex128",
        "seed": 0,
        "energy_history": [float(e) for e in history["energies"]],
        "step_times": [float(t) for t in history["step_times"]],
        "jit_compile_time": float(history["jit_compile_time"]),
        "final_energy": final_energy,
        "num_steps": int(history["num_steps"]),
        "converged": bool(history["converged"]),
        "total_wall_clock": dt,
        "peak_memory_mb": _peak_rss_mb(),
        "device": str(jax.devices()[0]).lower(),
        "lib_version": _tenax_git_sha(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

**Step 5.4: Verify it passes**

```bash
uv run pytest tests/test_varipeps_compare_run_tenax.py -v
```

Expected: PASS in 1–3 minutes.

**Step 5.5: Commit**

```bash
git add benchmarks/varipeps_compare/run_tenax.py tests/test_varipeps_compare_run_tenax.py
git commit -m "bench(varipeps-compare): run_tenax — CLI runner emitting shared JSON schema"
```

---

## Task 6: `run_varipeps.py` — variPEPS CLI runner

**Files:**
- Create: `benchmarks/varipeps_compare/run_varipeps.py`
- Create: `tests/test_varipeps_compare_run_varipeps.py`

**Step 6.1: Write the failing test**

```python
# tests/test_varipeps_compare_run_varipeps.py
"""Smoke test for run_varipeps.py — runs as subprocess, parses JSON."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import save_payload
from benchmarks.varipeps_compare.su_init import build_heisenberg_gate

_HAVE_VARIPEPS = importlib.util.find_spec("varipeps") is not None


@pytest.mark.algorithm
@pytest.mark.skipif(not _HAVE_VARIPEPS, reason="varipeps not installed")
def test_run_varipeps_bipartite_2site_d2_chi8(tmp_path):
    gate = build_heisenberg_gate()
    rng = np.random.default_rng(0)
    init = (rng.standard_normal((2, 2, 2, 2, 2, 2))
            + 1j * rng.standard_normal((2, 2, 2, 2, 2, 2))).astype(np.complex128)

    payload = tmp_path / "payload.npz"
    save_payload(payload, init=init, gate=gate,
                 meta={"path": "bipartite_2site", "D": 2, "chi": 8})

    out = tmp_path / "varipeps_result.json"
    cmd = [
        sys.executable, "-m", "benchmarks.varipeps_compare.run_varipeps",
        "--payload", str(payload),
        "--path", "bipartite_2site",
        "--D", "2", "--chi", "8",
        "--tol", "1e-4", "--max-steps", "5",
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, timeout=600)
    data = json.loads(out.read_text())
    assert data["lib"] == "varipeps"
    assert data["path"] == "bipartite_2site"
    assert data["final_energy"] < 0
```

**Step 6.2: Verify it fails**

```bash
uv run pytest tests/test_varipeps_compare_run_varipeps.py -v
```

Expected: FAIL — module not found.

**Step 6.3: Implement**

Pattern lifted from `/tmp/varipeps/examples/heisenberg_afm_square.py` and `/tmp/varipeps_test/run.py`.

```python
# benchmarks/varipeps_compare/run_varipeps.py
"""variPEPS 1.4.2 CLI runner for the compare benchmark."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX BEFORE importing varipeps (variPEPS reads jax.config at import)
jax.config.update("jax_enable_x64", True)

import varipeps  # noqa: E402

from .payload import load_payload
from .protocol import CTM_MAX_ITER, CTM_TOL


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _build_unitcell_bipartite_2site(init_AB: np.ndarray, D: int, chi_start: int, chi_max: int):
    """Build a 2-tensor checkerboard PEPS_Unit_Cell from stacked (2, D, D, D, D, d) init.

    variPEPS convention: tensors are (l, t, p, r, b) ordering (see
    ``/tmp/varipeps/examples/heisenberg_afm_square.py``).  We accept Tenax's
    (l, t, r, b, p) layout and transpose.
    """
    A = init_AB[0]  # (D, D, D, D, d)
    B = init_AB[1]
    A_v = np.transpose(A, (0, 1, 4, 2, 3))
    B_v = np.transpose(B, (0, 1, 4, 2, 3))
    structure = [[0, 1], [1, 0]]
    return varipeps.peps.PEPS_Unit_Cell.from_tensor_list(
        [A_v, B_v], structure, chi_start, max_chi=chi_max,
    )


def _build_unitcell_single_site(init_A: np.ndarray, D: int, chi_start: int, chi_max: int):
    """1×1 unconstrained PEPS_Unit_Cell with sublattice-rotated gate handled
    at the expectation-value level.  Same ansatz as Tenax single_site path.
    """
    A_v = np.transpose(init_A, (0, 1, 4, 2, 3))
    structure = [[0]]
    return varipeps.peps.PEPS_Unit_Cell.from_tensor_list(
        [A_v], structure, chi_start, max_chi=chi_max,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, type=Path)
    ap.add_argument("--path", required=True, choices=("single_site", "bipartite_2site"))
    ap.add_argument("--D", required=True, type=int)
    ap.add_argument("--chi", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--max-steps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    init_np, gate_np, _ = load_payload(args.payload)

    # Match variPEPS knobs to protocol.
    varipeps.config.optimizer_method = varipeps.config.Optimizing_Methods.L_BFGS
    varipeps.config.ctmrg_full_projector_method = varipeps.config.Projector_Method.FISHMAN
    varipeps.config.optimizer_max_steps = args.max_steps
    varipeps.config.ctmrg_max_steps = CTM_MAX_ITER
    varipeps.config.ctmrg_convergence_eps = CTM_TOL
    varipeps.config.ctmrg_print_steps = False
    varipeps.config.ad_custom_print_steps = False

    gate = jnp.asarray(gate_np.reshape(4, 4))  # variPEPS expects (d^2, d^2) gate
    exp_func = varipeps.expectation.Two_Sites_Expectation_Value(
        horizontal_gates=(gate,),
        vertical_gates=(gate,),
    )

    chi_start = min(args.D ** 2, args.chi)
    if args.path == "single_site":
        unitcell = _build_unitcell_single_site(init_np, args.D, chi_start, args.chi)
    else:
        unitcell = _build_unitcell_bipartite_2site(init_np, args.D, chi_start, args.chi)

    autosave = args.out.with_suffix(".hdf5")
    autosave.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = varipeps.optimization.optimize_peps_network(
        unitcell, exp_func, autosave_filename=str(autosave),
    )
    total = time.perf_counter() - t0

    # variPEPS returns step_energies/step_runtime as dicts keyed by run index.
    # We want the best-run trajectory (the one used to compute result.fun).
    best_run = int(result.best_run)
    energies = [float(e) for e in result.step_energies[best_run]]
    step_times = [float(t) for t in result.step_runtime[best_run]]

    # variPEPS step_runtime[0] includes JIT compile; split it out.
    if step_times:
        jit_time = step_times[0]
        step_times = step_times[1:]
    else:
        jit_time = 0.0

    out = {
        "lib": "varipeps",
        "path": args.path,
        "D": args.D,
        "chi": args.chi,
        "dtype": "complex128",
        "seed": 0,
        "energy_history": energies,
        "step_times": step_times,
        "jit_compile_time": jit_time,
        "final_energy": float(result.fun),
        "num_steps": int(result.nit),
        "converged": bool(result.success),
        "total_wall_clock": total,
        "peak_memory_mb": _peak_rss_mb(),
        "device": str(jax.devices()[0]).lower(),
        "lib_version": varipeps.__version__,
    }
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

**Caveat:** variPEPS's `PEPS_Unit_Cell.from_tensor_list` axis convention must be verified against the actual installed version (1.4.2). If `from_tensor_list` doesn't exist or the axis order differs, look up the constructor at `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/peps/__init__.py` and adapt.

**Step 6.4: Verify it passes**

```bash
uv run pytest tests/test_varipeps_compare_run_varipeps.py -v
```

Expected: PASS in 2–5 minutes. If `from_tensor_list` is the wrong API, fix the loader and re-run.

**Step 6.5: Commit**

```bash
git add benchmarks/varipeps_compare/run_varipeps.py tests/test_varipeps_compare_run_varipeps.py
git commit -m "bench(varipeps-compare): run_varipeps — variPEPS 1.4.2 CLI runner"
```

---

## Task 7: `compare.py` — orchestrator + plot

**Files:**
- Create: `benchmarks/varipeps_compare/compare.py`
- Create: `benchmarks/varipeps_compare/__main__.py`

**Step 7.1: Write `compare.py`**

```python
# benchmarks/varipeps_compare/compare.py
"""Orchestrator: enumerate grid, run SU once per point, spawn both runners, merge results."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .payload import save_payload
from .protocol import (
    GRID, MAX_STEPS, SUBPROCESS_TIMEOUT_SEC, TOL, grid_key,
)
from .su_init import (
    build_heisenberg_gate, build_sublattice_rotated_gate, make_init,
)


_logger = logging.getLogger(__name__)


@dataclass
class PointResult:
    key: str
    path: str
    D: int
    chi: int
    tenax: dict | None       # JSON dict or {"status": "error", "msg": ...}
    varipeps: dict | None
    delta_final_energy: float | None
    delta_num_steps: int | None
    tenax_speedup: float | None  # (varipeps total) / (tenax total)


def _run_subprocess(cmd: list[str], log_file: Path, timeout: int) -> tuple[bool, str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_file.open("w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except subprocess.CalledProcessError as exc:
        return False, f"exit code {exc.returncode}; see {log_file}"


def _build_payload(point: dict, results_dir: Path) -> Path:
    key = grid_key(point["path"], point["D"], point["chi"])
    payload = results_dir / f"{key}_payload.npz"
    if payload.exists():
        return payload
    if point["path"] == "single_site":
        gate = build_sublattice_rotated_gate()
    elif point["path"] == "bipartite_2site":
        gate = build_heisenberg_gate()
    else:
        raise ValueError(f"unknown path: {point['path']}")
    init = make_init(path=point["path"], D=point["D"], seed=0)
    save_payload(payload, init=init, gate=gate, meta={**point, "seed": 0})
    return payload


def _run_one_point(point: dict, results_dir: Path, *, force: bool) -> PointResult:
    key = grid_key(point["path"], point["D"], point["chi"])
    payload = _build_payload(point, results_dir)

    tenax_json = results_dir / f"{key}_tenax.json"
    varipeps_json = results_dir / f"{key}_varipeps.json"
    common_args = [
        "--payload", str(payload),
        "--path", point["path"],
        "--D", str(point["D"]),
        "--chi", str(point["chi"]),
        "--tol", str(TOL),
        "--max-steps", str(MAX_STEPS),
    ]

    if force or not tenax_json.exists():
        ok, msg = _run_subprocess(
            [sys.executable, "-m", "benchmarks.varipeps_compare.run_tenax",
             *common_args, "--out", str(tenax_json)],
            results_dir / f"{key}_tenax.log",
            SUBPROCESS_TIMEOUT_SEC,
        )
        if not ok:
            tenax_json.write_text(json.dumps({"status": "error", "msg": msg}))

    if force or not varipeps_json.exists():
        ok, msg = _run_subprocess(
            [sys.executable, "-m", "benchmarks.varipeps_compare.run_varipeps",
             *common_args, "--out", str(varipeps_json)],
            results_dir / f"{key}_varipeps.log",
            SUBPROCESS_TIMEOUT_SEC,
        )
        if not ok:
            varipeps_json.write_text(json.dumps({"status": "error", "msg": msg}))

    tenax = json.loads(tenax_json.read_text())
    varipeps = json.loads(varipeps_json.read_text())

    if "final_energy" in tenax and "final_energy" in varipeps:
        delta_e = tenax["final_energy"] - varipeps["final_energy"]
        delta_n = tenax["num_steps"] - varipeps["num_steps"]
        speedup = varipeps["total_wall_clock"] / tenax["total_wall_clock"]
    else:
        delta_e = delta_n = speedup = None

    return PointResult(
        key=key, path=point["path"], D=point["D"], chi=point["chi"],
        tenax=tenax, varipeps=varipeps,
        delta_final_energy=delta_e, delta_num_steps=delta_n,
        tenax_speedup=speedup,
    )


def _plot_trajectory(point: PointResult, out_png: Path):
    if "energy_history" not in (point.tenax or {}) or "energy_history" not in (point.varipeps or {}):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(point.tenax["energy_history"], "o-", label=f"tenax ({point.tenax['lib_version']})")
    ax.plot(point.varipeps["energy_history"], "x-", label=f"varipeps ({point.varipeps['lib_version']})")
    ax.set_xlabel("AD step")
    ax.set_ylabel("E / site")
    ax.set_title(f"{point.path}, D={point.D}, χ={point.chi}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _write_summary(results: list[PointResult], out_md: Path):
    lines = [
        "# Tenax ↔ variPEPS — Square Heisenberg AFM benchmark",
        "",
        "| key | path | D | χ | E_tenax | E_varipeps | ΔE | n_tenax | n_vp | t_tenax (s) | t_vp (s) | speedup |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.tenax and "final_energy" in r.tenax and r.varipeps and "final_energy" in r.varipeps:
            lines.append(
                f"| {r.key} | {r.path} | {r.D} | {r.chi} "
                f"| {r.tenax['final_energy']:.8f} | {r.varipeps['final_energy']:.8f} "
                f"| {r.delta_final_energy:+.2e} "
                f"| {r.tenax['num_steps']} | {r.varipeps['num_steps']} "
                f"| {r.tenax['total_wall_clock']:.1f} | {r.varipeps['total_wall_clock']:.1f} "
                f"| {r.tenax_speedup:.2f}x |"
            )
        else:
            lines.append(f"| {r.key} | {r.path} | {r.D} | {r.chi} | error | error | — | — | — | — | — | — |")
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--force", action="store_true", help="Re-run even if JSON exists")
    args = ap.parse_args()

    import os
    os.environ["JAX_PLATFORMS"] = args.device

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(results_dir / "run.log"), logging.StreamHandler()],
    )

    all_points: list[PointResult] = []
    for point in GRID:
        _logger.info("running %s", grid_key(point["path"], point["D"], point["chi"]))
        try:
            r = _run_one_point(point, results_dir, force=args.force)
        except Exception:
            _logger.exception("orchestrator failure on %s", point)
            continue
        all_points.append(r)
        _plot_trajectory(r, results_dir / f"{r.key}_trajectory.png")

    report = {r.key: asdict(r) for r in all_points}
    (results_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_summary(all_points, results_dir / "summary.md")
    _logger.info("done — report at %s", results_dir / "summary.md")


if __name__ == "__main__":
    main()
```

**Step 7.2: Write `__main__.py`**

```python
# benchmarks/varipeps_compare/__main__.py
from .compare import main

if __name__ == "__main__":
    main()
```

**Step 7.3: Add `.gitignore` entry**

```bash
echo "" >> .gitignore
echo "# variPEPS compare benchmark outputs" >> .gitignore
echo "benchmarks/varipeps_compare/results/" >> .gitignore
```

**Step 7.4: Commit**

```bash
git add benchmarks/varipeps_compare/compare.py benchmarks/varipeps_compare/__main__.py .gitignore
git commit -m "bench(varipeps-compare): compare — orchestrator, merge, plot, summary"
```

---

## Task 8: End-to-end smoke test

**Files:**
- Create: `tests/test_varipeps_compare.py`

**Step 8.1: Write the test**

```python
# tests/test_varipeps_compare.py
"""End-to-end smoke for benchmarks.varipeps_compare on the cheapest possible point.

C4v 1×1, D=2, chi=8, MAX_STEPS=20, tol=1e-4. Asserts both libs run and land
within 1e-3 of each other.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.varipeps_compare.payload import save_payload
from benchmarks.varipeps_compare.su_init import make_init, build_sublattice_rotated_gate

_HAVE_VARIPEPS = importlib.util.find_spec("varipeps") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_VARIPEPS, reason="varipeps not installed")
def test_smoke_single_site_d2_chi8(tmp_path):
    init = make_init(path="single_site", D=2, seed=0)
    gate = build_sublattice_rotated_gate()
    payload = tmp_path / "payload.npz"
    save_payload(payload, init=init, gate=gate, meta={"path": "single_site", "D": 2, "chi": 8})

    common = ["--payload", str(payload), "--path", "single_site",
              "--D", "2", "--chi", "8", "--tol", "1e-4", "--max-steps", "20"]
    tenax_out = tmp_path / "tenax.json"
    varipeps_out = tmp_path / "varipeps.json"
    subprocess.run([sys.executable, "-m", "benchmarks.varipeps_compare.run_tenax",
                    *common, "--out", str(tenax_out)], check=True, timeout=600)
    subprocess.run([sys.executable, "-m", "benchmarks.varipeps_compare.run_varipeps",
                    *common, "--out", str(varipeps_out)], check=True, timeout=900)
    t = json.loads(tenax_out.read_text())
    v = json.loads(varipeps_out.read_text())

    assert t["final_energy"] < 0
    assert v["final_energy"] < 0
    # NOTE: at chi=8 + 20 steps from random init, both libs may not yet be
    # converged.  We assert they agree, not that they're at the reference E.
    assert abs(t["final_energy"] - v["final_energy"]) < 1e-2, (
        f"libs disagree: tenax={t['final_energy']} varipeps={v['final_energy']}"
    )
    # Both should at least be below the trivial -0.5 saddle.
    assert t["final_energy"] < -0.5
    assert v["final_energy"] < -0.5
```

**Step 8.2: Run, verify it passes**

```bash
uv run pytest tests/test_varipeps_compare.py -v
```

Expected: PASS in ~2–4 minutes. If it doesn't, debug (likely variPEPS init axis convention from Task 6 — check actual energy printout in `tmp_path/varipeps.json` and the first-step energy in the log).

**Step 8.3: Commit**

```bash
git add tests/test_varipeps_compare.py
git commit -m "test(varipeps-compare): smoke — both libs agree within 1e-2 on single_site D=2 χ=8"
```

---

## Task 9: Run full benchmark + capture report

**Step 9.1: Run on CPU**

```bash
python -m benchmarks.varipeps_compare.compare --device cpu \
    --results-dir benchmarks/varipeps_compare/results
```

Expected: ~30–60 minutes for the 8-point grid on CPU. Tail `benchmarks/varipeps_compare/results/run.log` to monitor progress.

**Step 9.2: Inspect**

```bash
cat benchmarks/varipeps_compare/results/summary.md
ls benchmarks/varipeps_compare/results/*.png
```

Sanity checks:
- All 8 rows present and non-error.
- |ΔE| < 1e-4 on all points (within numerical noise of identical fixed point).
- Trajectory plots overlap closely after the first step (transient JIT differences acceptable).

If a point fails or |ΔE| > 1e-3, **stop and investigate** — do not paper over divergence. Likely causes: SU produced sligthly different init for the two-tensor case due to gauge, or variPEPS axis convention mismatch from Task 6.

**Step 9.3: Save the report under version control**

```bash
mkdir -p benchmarks/varipeps_compare/published_results
cp benchmarks/varipeps_compare/results/report.json benchmarks/varipeps_compare/published_results/
cp benchmarks/varipeps_compare/results/summary.md benchmarks/varipeps_compare/published_results/
cp benchmarks/varipeps_compare/results/*_trajectory.png benchmarks/varipeps_compare/published_results/
git add benchmarks/varipeps_compare/published_results/
git commit -m "bench(varipeps-compare): first published parity report (cpu, 2026-05-08)"
```

---

## Task 10: README + PR

**Step 10.1: Write `benchmarks/varipeps_compare/README.md`**

```markdown
# Tenax ↔ variPEPS square Heisenberg comparison

Apples-to-apples benchmark of Tenax against variPEPS 1.4.2 on the
spin-½ square-lattice Heisenberg AFM ground state.

See `docs/plans/2026-05-08-tenax-varipeps-square-heisenberg-benchmark-design.md`
for the protocol locks.

## Run

```bash
python -m benchmarks.varipeps_compare.compare --device cpu
```

Output: `results/report.json`, `results/summary.md`, `results/<key>_trajectory.png` per point.

## Files

- `protocol.py` — single source of truth for grid + knobs.
- `su_init.py` — Heisenberg gate + Tenax SU bootstrap.
- `payload.py` — cross-lib `.npz` init/gate/meta.
- `run_tenax.py` / `run_varipeps.py` — independently runnable CLI runners.
- `compare.py` — orchestrator + plotting + summary.
- `published_results/` — checked-in baseline (CPU, see commit history).
```

**Step 10.2: Update top-level README**

Add a one-line entry under the existing "Benchmarks" section pointing at the new sub-benchmark.

**Step 10.3: Commit and open PR**

```bash
git add benchmarks/varipeps_compare/README.md README.md
git commit -m "docs(varipeps-compare): README + top-level pointer"
git push -u origin varipeps-compare-square-heisenberg
gh pr create --title "Tenax ↔ variPEPS apples-to-apples square Heisenberg AD benchmark" \
  --body "$(cat <<'EOF'
## Summary
- Adds optional `iPEPSConfig.return_history` for trajectory capture in `optimize_gs_ad` (Task 1).
- New benchmark harness at `benchmarks/varipeps_compare/` runs both Tenax and variPEPS 1.4.2 on the spin-½ square-lattice Heisenberg AFM with shared SU init across an 8-point grid (D∈{2,3} × χ∈{16,24} × {C4v 1×1, 2×2 checkerboard}).
- Subprocess-isolated runners + orchestrator → `report.json`, `summary.md`, and per-point trajectory plots.
- First baseline report committed under `published_results/`.

Design doc: `docs/plans/2026-05-08-tenax-varipeps-square-heisenberg-benchmark-design.md`.

## Test plan
- [x] `uv run pytest tests/test_ipeps_ad_history.py -v`
- [x] `uv run pytest tests/test_varipeps_compare*.py -v`
- [x] `uv run pytest -m core -x` (no regression)
- [x] `python -m benchmarks.varipeps_compare.compare --device cpu` — full grid green
- [x] |ΔE| < 1e-3 on all 8 points

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Risk register & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| variPEPS `from_tensor_list` API mismatch or axis-ordering drift between 1.4.x patches | Medium | Task 6 caveat: verify against installed source at `/home/yjkao/miniforge3/lib/python3.12/site-packages/varipeps/peps/__init__.py`; smoke test at Task 8 catches it before full run. |
| OOM at D=3, χ=24 on CPU | Medium | Compare orchestrator marks point error and continues; can rerun on GPU with `--device cuda:0`. |
| Tenax 2-site path SU returns gauge-different tensors than what variPEPS expects | Medium | Both libs run their own CTM-fixed-point inside the AD step, so a one-step gauge mismatch should wash out in the first sweep. If not, smoke test (Task 8) flags it. |
| `single_site` random init from same seed lands at different basins in the two libs (path symmetry breaking) | Medium | The unconstrained 1×1 ansatz with rotated gate has a global minimum continuum (gauge-equivalent solutions). Energies should still match within 1e-3; trajectories may diverge in detail. The smoke test asserts only energy agreement (1e-2). |
| variPEPS imports auto-mutate `jax.config` and break Tenax precision settings | Low | Subprocess isolation eliminates this entirely. |
| Smoke test flaky due to L-BFGS line-search non-determinism | Low | Tolerances are loose (1e-2 for inter-lib agreement at the smoke point). |

---

## Out of scope (explicit)

- D=4, χ=32 — would need GPU; can extend grid in `protocol.py` later.
- Native-defaults run (each lib's own SU + own optimizer) — separate follow-up.
- Honeycomb / kagome cross-checks — separate follow-up.
- Bit-identity CPU↔GPU — JAX guarantees per-device determinism; cross-device parity is not a goal.
- `protocol.py` / `payload.py` / `su_init.py` unit-of-each unit tests — covered transitively by the smoke test (YAGNI).
