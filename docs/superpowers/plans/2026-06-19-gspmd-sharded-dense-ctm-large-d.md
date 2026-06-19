# GSPMD-sharded dense CTM (large-D, rung 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the **dense** iPEPS CTM forward + energy fit at D=6–8 / χ≈24 on 2–4 NVLink GPUs (where one GPU OOMs) by sharding the `D²` virtual axis of the env/double-layer tensors via JAX GSPMD, replicating the small projector SVD.

**Architecture:** A 1-D `jax.sharding.Mesh`; `NamedSharding`s shard the `D²` axis of edges + double-layer `a` and replicate the tiny corners. Because `DenseTensor`/`CTMTensorEnv` are registered pytrees and GSPMD propagates shardings from **committed inputs**, we shard the *initial* envs + site tensors with `jax.device_put` and the existing jitted CTM step partitions automatically — no change to the step math.

**Tech Stack:** Python, JAX (`jax.sharding`, GSPMD, `jax.device_put`), pytest (multi-device via `XLA_FLAGS=--xla_force_host_platform_device_count=N` in a subprocess).

**Spec:** `docs/superpowers/specs/2026-06-19-gspmd-sharded-dense-ctm-large-d-design.md`

---

## Background the implementer must know

- Env type: `CTMTensorEnv` = `NamedTuple` (`src/tenax/algorithms/_ctm_tensor_init.py:38`) with corners `C1..C4` `(χ,χ)` and edges `T1..T4` `(χ, D², χ)`. Each field is a `Tensor` (here `DenseTensor`).
- `DenseTensor` is a registered pytree (`src/tenax/core/tensor.py:446`): `tree_flatten` → leaf is one `jax.Array` (`.data`), aux is the `TensorIndex` tuple. So an env flattens to 8 arrays.
- Double-layer tensor `a`: `(D², D², D², D²)`, built by `_build_double_layer_tensor(A)` (`_ctm_tensor_init.py:84`).
- Forward loop: `python_loop_ctm_converge(...)` (`_ctm_python_loop.py:126`) builds `jit_step = _make_jit_ctm_step(neighbors, recipe)` (`:228`) and iterates; envs are a `dict[Coord, CTMTensorEnv]`.
- GSPMD rule we rely on: `jax.jit` applied to **committed** (sharded) input arrays partitions the computation and returns sharded outputs **without** explicit `in_shardings`. So sharding the inputs suffices.

## File structure

- **Create** `src/tenax/algorithms/ctm_sharding.py` — mesh + sharding-spec helpers (one responsibility: "given a mesh, produce shardings / commit tensors").
- **Create** `examples/spike_ctm_sharding.py` — throwaway GSPMD feasibility spike (parity + sharded-intermediate proof).
- **Create** `examples/bench_ctm_sharding_memory.py` — throwaway GPU memory benchmark (single-GPU OOM ceiling vs N-GPU fit).
- **Modify** `src/tenax/algorithms/ipeps_config.py` — add opt-in `device_mesh` field to `CTMConfig`.
- **Create** `tests/test_ctm_sharding.py` — helper unit tests + subprocess parity tests.
- **Create** `tests/_ctm_sharding_parity_subproc.py` — script the parity test runs under fake devices.

---

## Task 1: Sharding helpers (`ctm_sharding.py`)

**Files:**
- Create: `src/tenax/algorithms/ctm_sharding.py`
- Test: `tests/test_ctm_sharding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ctm_sharding.py`:

```python
import numpy as np
import pytest
import jax

from tenax.algorithms.ctm_sharding import (
    build_ctm_mesh,
    edge_partition_spec,
    corner_partition_spec,
    double_layer_partition_spec,
)


def test_mesh_and_specs():
    devs = jax.devices()
    mesh = build_ctm_mesh(devs)
    assert mesh.axis_names == ("d",)
    assert mesh.devices.size == len(devs)
    # edge (chi, D2, chi): shard axis 1 (D2) over "d"
    assert tuple(edge_partition_spec()) == (None, "d", None)
    # corner (chi, chi): replicated
    assert tuple(corner_partition_spec()) == (None, None)
    # double-layer (D2, D2, D2, D2): shard axis 0 over "d"
    assert tuple(double_layer_partition_spec()) == ("d", None, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_sharding.py::test_mesh_and_specs -v`
Expected: FAIL — `ModuleNotFoundError: ... ctm_sharding`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tenax/algorithms/ctm_sharding.py`:

```python
"""GSPMD sharding helpers for the dense CTM path (large-D, single-node multi-GPU).

Shards the D² virtual axis of the double-layer tensor and CTM edges over a 1-D
device mesh; corners (tiny, ~χ²) stay replicated. Used by the forward CTM to keep
per-device peak memory ≈1/N at large D. See the rung-1 design spec.
"""
from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_AXIS = "d"


def build_ctm_mesh(devices=None) -> Mesh:
    """1-D mesh named ``"d"`` over the given devices (default: all local devices)."""
    devs = list(devices) if devices is not None else jax.devices()
    return Mesh(np.asarray(devs), axis_names=(_AXIS,))


def edge_partition_spec() -> PartitionSpec:
    """Edge ``(χ, D², χ)`` → shard the D² axis."""
    return PartitionSpec(None, _AXIS, None)


def corner_partition_spec() -> PartitionSpec:
    """Corner ``(χ, χ)`` → replicated."""
    return PartitionSpec(None, None)


def double_layer_partition_spec() -> PartitionSpec:
    """Double-layer ``(D², D², D², D²)`` → shard the first D² axis."""
    return PartitionSpec(_AXIS, None, None, None)
```

Add `import numpy as np` at the top (used by `build_ctm_mesh`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_sharding.py::test_mesh_and_specs -v`
Expected: PASS (single-device mesh of size 1 is fine for the spec assertions).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ctm_sharding.py tests/test_ctm_sharding.py
git commit -m "feat(ctm): GSPMD sharding-spec helpers for dense CTM (large-D rung 1)"
```

---

## Task 2: Env/tensor commit helpers + sharded-shape unit test (fake devices)

**Files:**
- Modify: `src/tenax/algorithms/ctm_sharding.py`
- Test: `tests/test_ctm_sharding.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_sharding.py`:

```python
def test_commit_double_layer_is_sharded():
    # Requires >=2 devices; this test is run by the subprocess harness under
    # XLA_FLAGS=--xla_force_host_platform_device_count=4 (Task 4 driver). When
    # run on a single device it asserts the no-op (replicated) fallback.
    import jax.numpy as jnp
    from tenax.algorithms.ctm_sharding import build_ctm_mesh, commit_double_layer

    mesh = build_ctm_mesh()
    D2 = 4
    a = jnp.ones((D2, D2, D2, D2))
    a_sharded = commit_double_layer(a, mesh)
    n = mesh.devices.size
    # first axis is split across n devices (or 1 device → full shard = whole axis)
    shard_shape = a_sharded.sharding.shard_shape(a_sharded.shape)
    assert shard_shape[0] == D2 // n
    assert shard_shape[1:] == (D2, D2, D2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_sharding.py::test_commit_double_layer_is_sharded -v`
Expected: FAIL — `ImportError: cannot import name 'commit_double_layer'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/tenax/algorithms/ctm_sharding.py`:

```python
def commit_double_layer(a: jax.Array, mesh: Mesh) -> jax.Array:
    """device_put the double-layer tensor onto its D²-sharded layout."""
    return jax.device_put(a, NamedSharding(mesh, double_layer_partition_spec()))


def commit_env(env, mesh: Mesh):
    """device_put a CTMTensorEnv: edges D²-sharded, corners replicated.

    Operates on the ``DenseTensor`` leaves and rebuilds the env via the pytree so
    the wrapper/indices are preserved.
    """
    corner_sh = NamedSharding(mesh, corner_partition_spec())
    edge_sh = NamedSharding(mesh, edge_partition_spec())
    fields = {}
    for name in env._fields:
        t = getattr(env, name)
        sh = corner_sh if name.startswith("C") else edge_sh
        leaves, treedef = jax.tree_util.tree_flatten(t)
        leaves = [jax.device_put(x, sh) for x in leaves]
        fields[name] = jax.tree_util.tree_unflatten(treedef, leaves)
    return env._replace(**fields)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_sharding.py::test_commit_double_layer_is_sharded -v`
Expected: PASS (on 1 device, `shard_shape[0] == D2 // 1 == D2`).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ctm_sharding.py tests/test_ctm_sharding.py
git commit -m "feat(ctm): device_put commit helpers for sharded envs + double-layer"
```

---

## Task 3: GSPMD feasibility spike (de-risk before wiring)

**Files:**
- Create: `examples/spike_ctm_sharding.py`

This is a standalone script (no unit test) that proves GSPMD shards the dominant dense contraction. It is the empirical gate the spec front-loaded.

- [ ] **Step 1: Write the spike script**

Create `examples/spike_ctm_sharding.py`:

```python
"""Spike: does GSPMD shard the dominant dense-CTM contraction across devices?

Run on N fake CPU devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        uv run python examples/spike_ctm_sharding.py

Asserts: (1) the sharded enlarged-corner-style contraction equals the
single-device result to 1e-10; (2) the output is sharded (shard_shape < full).
"""
import jax
import jax.numpy as jnp

from tenax.algorithms.ctm_sharding import (
    build_ctm_mesh, commit_double_layer, edge_partition_spec, corner_partition_spec,
)
from jax.sharding import NamedSharding


def enlarged_corner(C, T_h, T_v, a):
    # toy stand-in for the enlarged-corner contraction: C(χ,χ), T_h(χ,D²,χ),
    # T_v(χ,D²,χ), a(D²,D²,D²,D²) → carries the D² legs GSPMD must partition.
    x = jnp.einsum("ij,jkl->ikl", C, T_h)        # (χ, D², χ)
    x = jnp.einsum("ikl,lmn->ikmn", x, T_v)      # (χ, D², D², χ)
    x = jnp.einsum("ikmn,kmpq->ipqn", x, a)      # (χ, D², D², χ)
    return x


def main():
    n = jax.device_count()
    chi, D2 = 8, 8
    key = jax.random.PRNGKey(0)
    k = jax.random.split(key, 4)
    C = jax.random.normal(k[0], (chi, chi))
    T_h = jax.random.normal(k[1], (chi, D2, chi))
    T_v = jax.random.normal(k[2], (chi, D2, chi))
    a = jax.random.normal(k[3], (D2, D2, D2, D2))

    single = jax.jit(enlarged_corner)(C, T_h, T_v, a)

    mesh = build_ctm_mesh()
    edge_sh = NamedSharding(mesh, edge_partition_spec())
    Cs = jax.device_put(C, NamedSharding(mesh, corner_partition_spec()))
    T_hs = jax.device_put(T_h, edge_sh)
    T_vs = jax.device_put(T_v, edge_sh)
    a_s = commit_double_layer(a, mesh)
    sharded = jax.jit(enlarged_corner)(Cs, T_hs, T_vs, a_s)

    err = float(jnp.max(jnp.abs(single - sharded)))
    shard0 = sharded.sharding.shard_shape(sharded.shape)
    print(f"devices={n}  max|single-sharded|={err:.2e}  out_shard_shape={shard0}")
    assert err < 1e-10, err
    if n > 1:
        assert shard0 != sharded.shape, "output not sharded"
    print("SPIKE OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the spike on 1 device (parity sanity)**

Run: `uv run python examples/spike_ctm_sharding.py`
Expected: prints `devices=1 ... SPIKE OK` (parity holds; no sharding assertion on 1 device).

- [ ] **Step 3: Run the spike on 4 fake devices (the real check)**

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python examples/spike_ctm_sharding.py`
Expected: `devices=4  max|single-sharded|=<1e-10  out_shard_shape=(8, 2, ...)` and `SPIKE OK`. If parity fails or the output is not sharded, STOP and report — the D²-axis sharding choice needs revisiting before wiring (this is the spec's flagged risk).

- [ ] **Step 4: Commit**

```bash
git add examples/spike_ctm_sharding.py
git commit -m "spike(ctm): GSPMD shards the dense enlarged-corner contraction (4 fake devices)"
```

---

## Task 4: Opt-in `CTMConfig.device_mesh` (default off, no behavior change)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py` (`CTMConfig`)
- Test: `tests/test_ctm_sharding.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ctm_sharding.py`:

```python
def test_ctmconfig_device_mesh_defaults_none():
    from tenax.algorithms.ipeps_config import CTMConfig
    cfg = CTMConfig(chi=8)
    assert cfg.device_mesh is None  # default off → single-device path
    mesh = build_ctm_mesh()
    cfg2 = CTMConfig(chi=8, device_mesh=mesh)
    assert cfg2.device_mesh is mesh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_sharding.py::test_ctmconfig_device_mesh_defaults_none -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'device_mesh'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tenax/algorithms/ipeps_config.py`, add a field to the `CTMConfig` dataclass (place it after the existing fields; use `field` import already present or add `from typing import Any`):

```python
    # Optional jax.sharding.Mesh for GSPMD-sharded dense CTM (large-D, multi-GPU).
    # None → single-device (default). See ctm_sharding.py and the rung-1 spec.
    device_mesh: Any = None
```

Add `from typing import Any` to the imports if not present. Update the class docstring's Attributes list with one line:
```
        device_mesh:        Optional jax.sharding.Mesh; when set, the dense CTM
                            shards env/double-layer tensors across it (multi-GPU
                            large-D). None = single-device.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_sharding.py::test_ctmconfig_device_mesh_defaults_none -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_ctm_sharding.py
git commit -m "feat(ctm): opt-in CTMConfig.device_mesh for GSPMD-sharded dense CTM"
```

---

## Task 5: Wire sharding into the forward CTM + parity test (fake devices, subprocess)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_python_loop.py` (env init in `python_loop_ctm_converge`)
- Create: `tests/_ctm_sharding_parity_subproc.py`
- Test: `tests/test_ctm_sharding.py`

The integration point: when a mesh is provided, commit the double-layer/site tensors and the initial envs to their shardings before the loop; GSPMD propagates through the existing `jit_step`.

- [ ] **Step 1: Write the parity subprocess script**

Create `tests/_ctm_sharding_parity_subproc.py`:

```python
"""Run a small dense CTM single-device vs sharded and assert energy parity.

Invoked by tests/test_ctm_sharding.py under
XLA_FLAGS=--xla_force_host_platform_device_count=2. Exits 0 on parity (<1e-8),
nonzero otherwise.
"""
import sys
import numpy as np
import jax

from tenax.algorithms.ctm_sharding import build_ctm_mesh
from tenax.algorithms.ipeps_config import CTMConfig
# Use the smallest end-to-end dense CTM energy helper available; build a D=2
# Heisenberg iPEPS site tensor and converge CTM both ways.
from tenax.algorithms.ipeps import _heisenberg_dense_probe_energy  # see Step 3


def main() -> int:
    chi, D = 8, 2
    e_single = _heisenberg_dense_probe_energy(D=D, chi=chi, device_mesh=None, seed=0)
    mesh = build_ctm_mesh()  # 2 fake devices
    e_sharded = _heisenberg_dense_probe_energy(D=D, chi=chi, device_mesh=mesh, seed=0)
    err = abs(float(e_single) - float(e_sharded))
    print(f"devices={jax.device_count()} e_single={e_single:.10f} "
          f"e_sharded={e_sharded:.10f} |Δ|={err:.2e}")
    return 0 if err < 1e-8 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing test (drives the subprocess)**

Append to `tests/test_ctm_sharding.py`:

```python
import os
import subprocess
import sys


def test_sharded_forward_matches_single_device():
    """Dense CTM energy under a 2-device GSPMD mesh equals the single-device
    result to <1e-8 (run in a subprocess with fake CPU devices)."""
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2")
    r = subprocess.run(
        [sys.executable, "tests/_ctm_sharding_parity_subproc.py"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_sharding.py::test_sharded_forward_matches_single_device -v`
Expected: FAIL — subprocess errors with `ImportError: cannot import name '_heisenberg_dense_probe_energy'` (the helper and the mesh wiring don't exist yet).

- [ ] **Step 4: Write minimal implementation**

(a) In `src/tenax/algorithms/_ctm_python_loop.py`, give `python_loop_ctm_converge` an optional `device_mesh=None` keyword. Immediately after the envs are initialized (`envs = env_init if ... else {...}`, around line 258), add:

```python
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import commit_env
        envs = {coord: commit_env(env, device_mesh) for coord, env in envs.items()}
```

Also commit the site tensors when a mesh is set, right after `site_tensors` is available at the top of the function:

```python
    if device_mesh is not None:
        from tenax.algorithms.ctm_sharding import commit_double_layer
        # site_tensors carry the iPEPS A; the double-layer is built inside the
        # step. Commit A's underlying array so the built double-layer inherits a
        # D²-compatible sharding via GSPMD propagation.
        site_tensors = {
            c: t.with_data(commit_double_layer_compatible(t.data, device_mesh))
            for c, t in site_tensors.items()
        }
```

If `with_data` / a single-array commit on the raw site tensor is not a clean fit, commit only the envs (the dominant memory objects) in this rung and note it — the env commit alone drives the GSPMD propagation through the step. Keep this minimal: **env commit is the required path; site-tensor commit is best-effort.** (Implementer: prefer env-only commit if the site-tensor layout fights the D² spec; record which you did.)

(b) Add the probe helper used by the subprocess. In `src/tenax/algorithms/ipeps.py` (or wherever the dense CTM energy entry is cleanest), add:

```python
def _heisenberg_dense_probe_energy(*, D: int, chi: int, device_mesh=None, seed: int = 0) -> float:
    """Tiny dense iPEPS Heisenberg energy via one CTM convergence (test probe).

    Builds a random D-bond 1-site iPEPS A, converges the dense CTM (optionally on
    ``device_mesh``), and returns the NN Heisenberg energy. Used only by tests to
    compare single-device vs GSPMD-sharded forward CTM.
    """
    # Reuse the existing dense forward-CTM + energy path used by the 1-site
    # optimizer; thread device_mesh into python_loop_ctm_converge. Keep χ/iter
    # small (max_iter=30) for a fast, deterministic probe.
    ...  # implementer: assemble from optimize_gs_ad's dense building blocks
```

The implementer composes this from the existing 1-site dense building blocks (`_ctm_tensor_init`, `python_loop_ctm_converge(..., device_mesh=device_mesh)`, `compute_energy_ctm_tensor`), with a fixed seed and `max_iter=30`. It must accept `device_mesh` and pass it to `python_loop_ctm_converge`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_sharding.py::test_sharded_forward_matches_single_device -v`
Expected: PASS — subprocess prints `devices=2 ... |Δ|=<1e-8` and exits 0.

- [ ] **Step 6: Run the full sharding test module + a dense-CTM regression**

Run: `uv run pytest tests/test_ctm_sharding.py -v && uv run pytest -m core -q -k ctm`
Expected: all PASS; the flag-off dense CTM path is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_python_loop.py src/tenax/algorithms/ipeps.py tests/_ctm_sharding_parity_subproc.py tests/test_ctm_sharding.py
git commit -m "feat(ctm): thread device_mesh through forward CTM; sharded==single parity"
```

---

## Task 6: No-regression guard (flag-off identical)

**Files:**
- Test: `tests/test_ctm_sharding.py`

- [ ] **Step 1: Write the test**

Append:

```python
def test_flag_off_is_unchanged():
    """device_mesh=None must give the exact same energy as not passing it."""
    from tenax.algorithms.ipeps import _heisenberg_dense_probe_energy
    e_default = _heisenberg_dense_probe_energy(D=2, chi=8, seed=1)
    e_none = _heisenberg_dense_probe_energy(D=2, chi=8, device_mesh=None, seed=1)
    assert e_default == e_none
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_ctm_sharding.py::test_flag_off_is_unchanged -v`
Expected: PASS (bit-identical; `device_mesh=None` is a no-op).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ctm_sharding.py
git commit -m "test(ctm): flag-off device_mesh path is bit-identical"
```

---

## Task 7: GPU memory feasibility benchmark (throwaway, manual)

**Files:**
- Create: `examples/bench_ctm_sharding_memory.py`

This is the headline deliverable: show a D that OOMs on 1 GPU fits on N. Manual (needs real GPUs); not a CI test.

- [ ] **Step 1: Write the benchmark**

Create `examples/bench_ctm_sharding_memory.py`:

```python
"""Memory feasibility: dense CTM at large D, single-GPU vs N-GPU GSPMD mesh.

Usage (real GPUs):
    # 1) find the single-GPU OOM ceiling:
    CUDA_VISIBLE_DEVICES=0 uv run python examples/bench_ctm_sharding_memory.py --D 6 8 --chi 24
    # 2) show it fits on N GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python examples/bench_ctm_sharding_memory.py --D 6 8 --chi 24 --shard

Reports per-device peak memory (jax.device.memory_stats peak_bytes_in_use) and
whether the run completed or OOM'd, for each D.
"""
import argparse
import jax

from tenax.algorithms.ctm_sharding import build_ctm_mesh
from tenax.algorithms.ipeps import _heisenberg_dense_probe_energy


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--shard", action="store_true")
    args = ap.parse_args()
    mesh = build_ctm_mesh() if args.shard else None
    n = jax.device_count() if args.shard else 1
    print(f"# devices={jax.device_count()} shard={args.shard} mesh_n={n} chi={args.chi}")
    for D in args.D:
        try:
            e = _heisenberg_dense_probe_energy(D=D, chi=args.chi, device_mesh=mesh, seed=0)
            print(f"D={D}: OK  E={float(e):.6f}  per_device_peak={peak_gb():.2f} GB")
        except Exception as ex:  # noqa: BLE001 - benchmark reports OOM and continues
            print(f"D={D}: FAILED ({type(ex).__name__}: {str(ex)[:80]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on CPU (small D, correctness of the harness)**

Run: `uv run python examples/bench_ctm_sharding_memory.py --D 2 3 --chi 8`
Expected: prints `D=2: OK ...` and `D=3: OK ...` (peak may be `nan` on CPU — acceptable; this only checks the harness runs).

- [ ] **Step 3: Commit**

```bash
git add examples/bench_ctm_sharding_memory.py
git commit -m "bench(ctm): single-GPU-vs-N-GPU memory feasibility for sharded dense CTM"
```

---

## Self-Review notes

- **Spec coverage:** sharding scheme (D²/corner/SVD) → Tasks 1–2; GSPMD risk micro-benchmark → Task 3; opt-in surface → Task 4; sharded forward CTM + correctness parity → Task 5; no-regression → Task 6; memory feasibility benchmark → Task 7. AD backward / distributed SVD / multisite are explicitly out of scope (rungs 2–3).
- **Type consistency:** `build_ctm_mesh`, `commit_double_layer`, `commit_env`, `*_partition_spec`, `CTMConfig.device_mesh`, `_heisenberg_dense_probe_energy(*, D, chi, device_mesh, seed)` are used identically across tasks.
- **Honest-uncertainty flags (not placeholders):** Task 3 is a gating spike with concrete pass/fail; Task 5 Step 4 documents the env-commit-required / site-commit-best-effort decision and the probe-helper assembly, because the exact site-tensor commit depends on GSPMD behavior the spike establishes. If the spike (Task 3) fails the sharded-output assertion, revisit the D²-axis choice before Task 5.
