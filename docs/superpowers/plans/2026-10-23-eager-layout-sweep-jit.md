# Eager Layout Sweep Compile Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eager #1098 chi-layout sweep compile a few dozen XLA programs per new layout instead of thousands of per-primitive executables. The layout it chooses and the jitted CTM step must not change.

**Architecture:** One eager sweep (`_ctm_python_loop._eager_ctm_sweep`) needs concrete values at one point only: the M′ truncation in the symmetric 2x2 projector (`_retruncate_chi_bond`, global top-chi). Every other piece has static shapes once the input layout is fixed. A small helper, `_ctm_eager_jit.call(fn, ...)`, runs a function under `jax.jit` when its inputs are concrete and calls it directly when they are traced. The eager sweep routes these pieces through the helper:

- the enlarged corners
- the projector up to M′
- the projector after the cut
- the four absorb moves

The M′ SVD and the truncation stay eager. The traced path (the jitted `_step`, AD) still calls the same functions directly, so its HLO does not change.

**Tech Stack:** JAX 0.10.2 (`jax.jit`, `jax.monitoring`), tenax block-sparse `SymmetricTensor` (a pytree: one flat data leaf; block keys, shapes, offsets and indices in the treedef), pytest.

**Spec:** None. This plan argues from the measurements in "Problem record" below, so rulings made without a spec are provisional.

## Problem record (measured 2026-10-23)

- The S5 chi=16 run (`xxz03_u1_D3_chi16_s0`, harness `a00c38c`) stayed in S2 for more than 8 h.
  - GPU 0 at 0%; one CPU core at about 100%.
  - Two py-spy dumps (21:07, 21:15) showed the main thread in `backend_compile_and_load`, under `dot_general` ← `_contract_symmetric` ← `_ctm_tensor_absorb_{right,bottom}_2plaq` ← `_eager_ctm_sweep` ← `_adapt_symmetric_layout` ← `s2` → `_chi_binding`. `_chi_binding` runs CTM at chi+4 = 20.
  - The persistent cache, which keeps compiles of 1 s or more, got no new entry for 4 h.
- Per eager op, measured with a microbenchmark: cold 178 ms on GPU and 40 ms on CPU; warm 0.35 ms on GPU and 0.10 ms on CPU.
- Spike results, `python_loop_ctm_converge` at D=3, U(1) XXZ Δ=0.3, SU 100 steps, chi=8, GPU 1:

| Arm | Eager sweep 1: compiles / compile s / wall s | Eager sweep 2: compiles / wall s | Converge wall s |
|---|---|---|---|
| Baseline, GPU | 3249 / 89.7 / 125.4 | 0 / 10.4 | 348.4 |
| Baseline, CPU | 3249 / 52.1 / 75.2 | 0 / 4.1 | 248.2 |
| `TENAX_BATCH_BLOCKSPARSE=1`, GPU | 4487 / 143.3 / 201.3 | 16 / 22.3 | 439.2 |
| Only `G.contract` jitted, GPU | 1169 / 67.9 / 85.0 | 118 / 39.5 | 161.4 |

  - Running on the CPU and the #568 batched mode do not fix the cost.
  - Jitting only `contract` halves it, so the remaining compiles come from the other block ops. `SymmetricTensor.blocks` slices one flat buffer at static offsets, so every block access is another eager executable.

## Global Constraints

- The traced path must not change. The lowered HLO text of the jitted `_step` (`_make_jit_ctm_step`) for a symmetric fixture must be byte-identical before and after each task that edits projector or move code (Task 3, Task 4).
- The dense path must not change. `DenseTensor` input never enters the eager layout phase (`_needs_eager_layout_phase`). Do not touch the dense branches.
- Turning the feature off must give the old behaviour exactly. With `_ctm_eager_jit._ENABLED = False`, the eager sweep must run the same Python code as before this plan.
- Use the CUDA plugin that is already in the worktree venv. For GPU timing, use only GPU 1 (`CUDA_VISIBLE_DEVICES=1`); GPU 0 runs the S5 job.
- Run test runs one at a time, with `taskset -c 0-15`. Set `JAX_PLATFORMS=cpu` for tests.
- Use the worktree's own `.venv/bin/python`. Do not pipe a long run to `tail`; redirect it to a file.
- Mutation-test every new test. Commit first, use a unique anchor, set `PYTHONDONTWRITEBYTECODE=1`, and delete `__pycache__` between mutants.
- One concern per branch: `perf/eager-layout-sweep-jit`. No S5 harness change in this plan.

## Review Focus

1. **A tie at the cut.** Where two singular values at the chi boundary are equal, the jitted M′ can differ from the eager one in the last bits, so `_select_chi_slots` can keep the other tied slot. Expected behaviour: the energy agrees and the layout may differ. Test: U(1) D=2 chi=8, the known three-fold degenerate cut; compare energies, not layouts (Task 4, Step 6).
2. **Inputs that are already traced** (AD, the jitted `_step`, `jax.grad` through `python_loop_ctm_converge` callers). The helper must call the raw function and must not build its own `jit` (Task 2 test).
3. **A function patched in a test** (`test_enlarged_corner_flow_invariant_834.py` patches `moves._build_enlarged_corner`). The helper takes the function object at call time and keys its jit cache on it, so a patch takes effect (Task 2 test).
4. **Fermionic (graded Z2) `SymmetricTensor` input.** The graded `G.contract` and `G.svd` must give the same sweep through the jitted pieces (Task 4, Step 6).
5. **The #907 rank-0 SVD diagnosis** (`_require_svd_connected`). It must still raise the same `ValueError` from inside the jitted projector stage (Task 4, Step 6).

---

### Task 1: Benchmark the eager sweep compile cost

**Files:**
- Create: `benchmarks/ctm_eager_layout_compile.py`

**Interfaces:**
- Produces: a CLI `python -m benchmarks.ctm_eager_layout_compile --D 3 --chi 8 [--su-steps 100] [--max-iter 40]`. It prints one line per eager sweep, `eager sweep N: wall W s, compiles C, compile_s S`, then `converge: wall W s, compiles C, eager_sweeps N`. Tasks 3–5 use it for before/after numbers.

- [ ] **Step 1: Write the benchmark**

```python
"""Compile count and wall time of the #1098 eager chi-layout sweeps.

Fixture: U(1) XXZ Delta=0.3 2-site iPEPS from a short simple update, then a
cold ``python_loop_ctm_converge``.  Each eager sweep is timed and the XLA
backend compiles inside it are counted (``jax.monitoring``).

Run on one GPU (``CUDA_VISIBLE_DEVICES=1``) or with ``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import jax

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._ctm_python_loop as cpl  # noqa: E402
from tenax import CTMConfig, ipeps, iPEPSConfig  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS  # noqa: E402
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate  # noqa: E402
from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs  # noqa: E402

_COMPILES = [0, 0.0]


def _listener(name, dur, **_):
    if name == "/jax/core/compile/backend_compile_duration":
        _COMPILES[0] += 1
        _COMPILES[1] += dur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--chi", type=int, default=8)
    ap.add_argument("--su-steps", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=40)
    a = ap.parse_args()
    jax.monitoring.register_event_duration_secs_listener(_listener)
    print("devices", jax.devices(), flush=True)

    A0, B0 = heisenberg_u1sz_init_pair(a.D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=a.D, num_imaginary_steps=a.su_steps, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))

    n_sweeps = [0]
    raw = cpl._eager_ctm_sweep

    def timed(*args, **kwargs):
        c0, s0, t0 = _COMPILES[0], _COMPILES[1], time.time()
        out = raw(*args, **kwargs)
        jax.block_until_ready(jax.tree.leaves(out[0]))
        n_sweeps[0] += 1
        print(
            f"eager sweep {n_sweeps[0]}: wall {time.time() - t0:.1f} s, "
            f"compiles {_COMPILES[0] - c0}, compile_s {_COMPILES[1] - s0:.1f}",
            flush=True,
        )
        return out

    cpl._eager_ctm_sweep = timed
    cfg = dataclasses.replace(
        CTMConfig(chi=a.chi),
        conv_tol=1e-10,
        max_iter=a.max_iter,
        min_iter=5,
        plateau_patience=None,
        on_unconverged="warn",
    )
    c0, t0 = _COMPILES[0], time.time()
    cpl.python_loop_ctm_converge(
        {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
    )
    print(
        f"converge: wall {time.time() - t0:.1f} s, compiles {_COMPILES[0] - c0}, "
        f"eager_sweeps {n_sweeps[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the baseline at chi=8 on GPU 1**

Run: `CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_enable_command_buffer= taskset -c 0-15 .venv/bin/python -m benchmarks.ctm_eager_layout_compile --D 3 --chi 8 > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/bench_base_chi8.log 2>&1`

Expected: within about 10% of the spike numbers: sweep 1 has 3249 compiles; converge wall is about 350 s. Record the lines in the ledger.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/ctm_eager_layout_compile.py
git commit -m "bench(ctm): count compiles per eager #1098 layout sweep"
```

The chi=16 baseline is in Task 5. It takes hours, so run it only after the fix numbers at chi=8 look right.

---

### Task 2: The `jit when concrete` helper

**Files:**
- Create: `src/tenax/algorithms/_ctm_eager_jit.py`
- Create: `tests/test_ctm_eager_jit.py`
- Modify: `tests/conftest.py` (add `"test_ctm_eager_jit.py": "core"` to `_FILE_MARKERS`, with a one-line comment: pure mechanism, no CTM)

**Interfaces:**
- Produces:
  - `call(fn, *args, static_argnames: tuple[str, ...] = (), **kwargs)`: returns `fn(*args, **kwargs)`. It runs through a cached `jax.jit(fn, static_argnames=...)` when `_ENABLED` is true and no leaf of `(args, kwargs)` is a `jax.core.Tracer`; otherwise it calls `fn` directly.
  - `_ENABLED: bool = True`, the module switch that tests set with `monkeypatch.setattr`.
  - `_jitted(fn, static_argnames)`: the `functools.cache` that holds the jit wrappers, keyed by the function object.

- [ ] **Step 1: Write the failing tests**

```python
"""``_ctm_eager_jit.call``: jit when the inputs are concrete, pass through when traced."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import tenax.algorithms._ctm_eager_jit as ej


def _double(x):
    return x * 2.0


def _scale(x, *, k):
    return x * k


def test_concrete_call_runs_through_one_cached_jit():
    ej._jitted.cache_clear()
    out = ej.call(_double, jnp.ones(3))
    assert jnp.allclose(out, 2.0)
    jitted = ej._jitted(_double, ())
    assert jitted._cache_size() == 1
    ej.call(_double, jnp.zeros(3))  # same shape: no new trace
    assert jitted._cache_size() == 1


def test_traced_call_bypasses_the_jit():
    ej._jitted.cache_clear()
    jax.jit(lambda x: ej.call(_double, x))(jnp.ones(3))
    assert ej._jitted.cache_info().currsize == 0


def test_disabled_calls_the_raw_function(monkeypatch):
    ej._jitted.cache_clear()
    monkeypatch.setattr(ej, "_ENABLED", False)
    out = ej.call(_double, jnp.ones(3))
    assert jnp.allclose(out, 2.0)
    assert ej._jitted.cache_info().currsize == 0


def test_static_argnames_are_static():
    ej._jitted.cache_clear()
    a = ej.call(_scale, jnp.ones(2), k=3.0, static_argnames=("k",))
    b = ej.call(_scale, jnp.ones(2), k=5.0, static_argnames=("k",))
    assert jnp.allclose(a, 3.0) and jnp.allclose(b, 5.0)
    assert ej._jitted(_scale, ("k",))._cache_size() == 2


def test_a_patched_function_gets_its_own_jit():
    """Review Focus 3: the jit is keyed by the function object at call time."""
    ej._jitted.cache_clear()

    def f1(x):
        return x * 2.0

    def f2(x):
        return x + 10.0

    assert jnp.allclose(ej.call(f1, jnp.ones(2)), 2.0)
    assert jnp.allclose(ej.call(f2, jnp.ones(2)), 11.0)
    assert ej._jitted.cache_info().currsize == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_jit.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_eager_jit'`

- [ ] **Step 3: Write the helper**

```python
"""Jit the static-shape pieces of an eager CTM sweep (#1098 compile cost).

Outside ``jax.jit`` every JAX primitive compiles one executable per new
shape.  An eager symmetric CTM sweep runs thousands of block-level
primitives, so at chi=16 the #1098 layout phase spends hours compiling
(measured 2026-10-23: 3249 compiles in one D=3 chi=8 sweep).  Only the M'
truncation needs concrete values; :func:`call` runs every other piece under
one ``jax.jit`` per function and input structure.

Under an enclosing trace (the jitted ``_step``, AD) :func:`call` calls the
function directly, so the traced program is the one it was before.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax

# Module switch: ``False`` runs every piece eagerly, as before this module.
_ENABLED = True


@functools.cache
def _jitted(fn: Callable, static_argnames: tuple[str, ...]) -> Callable:
    return jax.jit(fn, static_argnames=static_argnames)


def _has_tracer(tree) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def call(fn: Callable, *args, static_argnames: tuple[str, ...] = (), **kwargs):
    """``fn(*args, **kwargs)``, under a cached ``jax.jit`` when the inputs are concrete."""
    if not _ENABLED or _has_tracer((args, kwargs)):
        return fn(*args, **kwargs)
    return _jitted(fn, tuple(static_argnames))(*args, **kwargs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_jit.py -q`

Expected: 5 passed

- [ ] **Step 5: Commit, then mutation-test**

```bash
git add src/tenax/algorithms/_ctm_eager_jit.py tests/test_ctm_eager_jit.py tests/conftest.py
git commit -m "perf(ctm): helper that jits a function only for concrete inputs"
```

Mutants (assert each anchor occurs once in the file; set `PYTHONDONTWRITEBYTECODE=1`; delete `__pycache__`; restore with `git checkout` after each one):

- M1: `if not _ENABLED or _has_tracer((args, kwargs)):` → `if not _ENABLED:`. Expect `test_traced_call_bypasses_the_jit` to fail.
- M2: `return _jitted(fn, tuple(static_argnames))(*args, **kwargs)` → `return fn(*args, **kwargs)`. Expect `test_concrete_call_runs_through_one_cached_jit` to fail.
- M3: `if not _ENABLED or` → `if False and not _ENABLED or`. Expect `test_disabled_calls_the_raw_function` to fail.

Record each mutant's result in the ledger.

---

### Task 3: Jit the absorb moves and the double layer in the eager sweep

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py`. The four `_ctm_tensor_absorb_*_2plaq(...)` calls in the 2x2 Phase 2 (about lines 534–610) become `_ej.call(_ctm_tensor_absorb_*_2plaq, ...)`.
- Modify: `src/tenax/algorithms/_ctm_python_loop.py:_ctm_sweep`. `_build_double_layer_tensor(A)` becomes `_ej.call(_build_double_layer_tensor, A)`.
- Create: `tests/test_ctm_eager_sweep_jit.py`
- Modify: `tests/conftest.py` (add `"test_ctm_eager_sweep_jit.py": "algorithm"`)

**Interfaces:**
- Consumes: `tenax.algorithms._ctm_eager_jit.call`, `_ENABLED` (Task 2).
- Produces: the module-scoped `sites` fixture and the `_sweeps` helper in `tests/test_ctm_eager_sweep_jit.py`. Task 4 adds tests to this file.

- [ ] **Step 1: Write the failing equivalence test**

```python
"""Eager #1098 layout sweeps jit their static-shape pieces; the result is unchanged."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tenax.algorithms._ctm_eager_jit as ej
import tenax.algorithms._ctm_python_loop as cpl
from tenax import CTMConfig, ipeps, iPEPSConfig
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair, xxz_gate

jax.config.update("jax_enable_x64", True)

D = 2
CHI = 6  # U(1) chi=6 cuts the eager C1 spectrum in a clean gap (#1098 test)


def _sweep_kwargs(chi):
    return dict(
        neighbors=CHECKERBOARD_NEIGHBORS,
        recipe="2x2",
        device_mesh=None,
        projector_method="svd",
        renormalize=False,
        projector_backward="auto",
        chunk_size=None,
        chi=chi,
    )


@pytest.fixture(scope="module")
def sites():
    A0, B0 = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    return {
        (0, 0): A * (1.0 / (A.norm() + 1e-10)),
        (1, 0): B * (1.0 / (B.norm() + 1e-10)),
    }


def _sweeps(sites, chi, n, enabled, monkeypatch):
    monkeypatch.setattr(ej, "_ENABLED", enabled)
    envs = {c: initialize_ctm_tensor_env(A, chi) for c, A in sites.items()}
    for _ in range(n):
        envs, _, _ = cpl._eager_ctm_sweep(sites, envs, **_sweep_kwargs(chi))
        envs = cpl._charge_sorted_envs(envs)
    return envs


def _assert_same_envs(a, b, atol=1e-10):
    assert cpl._chi_layout(a) == cpl._chi_layout(b)
    for c in a:
        for f in a[c]._fields:
            x = np.asarray(getattr(a[c], f).todense())  # chi=6, D^2=4: small
            y = np.asarray(getattr(b[c], f).todense())
            np.testing.assert_allclose(x, y, atol=atol, err_msg=f"{c}.{f}")


def test_jitted_eager_sweeps_match_plain_eager_sweeps(sites, monkeypatch):
    plain = _sweeps(sites, CHI, 3, False, monkeypatch)
    jitted = _sweeps(sites, CHI, 3, True, monkeypatch)
    _assert_same_envs(plain, jitted)


def test_eager_sweep_runs_the_absorb_moves_under_jit(sites, monkeypatch):
    seen = []
    raw_jitted = ej._jitted

    def spy(fn, static_argnames):
        seen.append(fn.__name__)
        return raw_jitted(fn, static_argnames)

    monkeypatch.setattr(ej, "_jitted", spy)
    _sweeps(sites, CHI, 1, True, monkeypatch)
    for d in ("left", "right", "top", "bottom"):
        assert f"_ctm_tensor_absorb_{d}_2plaq" in seen
    assert "_build_double_layer_tensor" in seen
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_sweep_jit.py -q > /tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/t3_red.log 2>&1`

Expected: `test_jitted_eager_sweeps_match_plain_eager_sweeps` PASSES, because nothing is jitted yet and the two runs are the same code. `test_eager_sweep_runs_the_absorb_moves_under_jit` FAILS with `AssertionError` on `_ctm_tensor_absorb_left_2plaq`.

- [ ] **Step 3: Record the traced-step HLO before the change**

Create `/tmp/claude-1000/-home-yjkao-tenax/eac5c580-d979-44d1-a285-3b31137975ef/scratchpad/step_hlo.py`:

```python
"""Print a hash of the lowered jitted CTM _step HLO for a U(1) D=2 chi=6 fixture."""
import hashlib, sys
import jax
jax.config.update("jax_enable_x64", True)
import tenax.algorithms._ctm_python_loop as cpl
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

A, B = heisenberg_u1sz_init_pair(2, jax.random.PRNGKey(0))
sites = {(0, 0): A, (1, 0): B}
envs = {c: initialize_ctm_tensor_env(t, 6) for c, t in sites.items()}
step = cpl._make_jit_ctm_step(CHECKERBOARD_NEIGHBORS, "2x2")
text = step.lower(sites, envs, chi=6).as_text()
print(hashlib.sha256(text.encode()).hexdigest(), len(text))
```

Run it on the Task 2 commit: `JAX_PLATFORMS=cpu .venv/bin/python <that path> > .../hlo_before.txt`.

- [ ] **Step 4: Route the pieces through the helper**

In `src/tenax/algorithms/_ctm_tensor_convergence.py`, add `from tenax.algorithms import _ctm_eager_jit as _ej` next to the other `tenax.algorithms` imports. Then change each of the four absorb calls the same way. For LEFT:

```python
                    C1_new, T4_new, C4_new = _ej.call(
                        _ctm_tensor_absorb_left_2plaq,
                        envs_old[s_src],
                        _shard_a(double_layers[s_src], "left"),
                        P_top_above,
                        P_bot_above,
                        P_top_curr,
                        P_bot_curr,
                    )
```

Make the same change for `_ctm_tensor_absorb_right_2plaq` (`"right"`, `P_top_above, P_bot_above, P_top_curr, P_bot_curr`), `_ctm_tensor_absorb_top_2plaq` (`"top"`, `P_top_left, P_bot_left, P_top_curr, P_bot_curr`) and `_ctm_tensor_absorb_bottom_2plaq` (`"bottom"`, `P_top_left, P_bot_left, P_top_curr, P_bot_curr`). Keep the arguments in their current order.

In `src/tenax/algorithms/_ctm_python_loop.py:_ctm_sweep`:

```python
    from tenax.algorithms import _ctm_eager_jit as _ej

    double_layers = {
        c: _ej.call(_build_double_layer_tensor, A) for c, A in site_tensors.items()
    }
```

(Put the import at module top with the other `tenax.algorithms` imports, not inside the function.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_sweep_jit.py -q > .../t3_green.log 2>&1`

Expected: 2 passed

- [ ] **Step 6: Check that the traced-step HLO is unchanged**

Run the Step 3 script again into `hlo_after.txt`, then `diff hlo_before.txt hlo_after.txt`.

Expected: no difference (the same sha256 and length). Under the `_step` trace, `call` sees tracers and calls each function directly.

- [ ] **Step 7: Run the #1098 regression files**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_symmetric_layout_adapts_1098.py tests/test_enlarged_corner_flow_invariant_834.py tests/test_ctm_traced_chi_inventory_929.py -q -m "not slow" > .../t3_regress.log 2>&1`

Expected: all pass (compare the count with the same command run on `origin/main`).

- [ ] **Step 8: Commit, then mutation-test**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py src/tenax/algorithms/_ctm_python_loop.py tests/test_ctm_eager_sweep_jit.py tests/conftest.py
git commit -m "perf(ctm): jit the absorb moves of an eager layout sweep"
```

Mutant M4: in the RIGHT branch, `_ej.call(\n                        _ctm_tensor_absorb_right_2plaq,` → a direct call `_ctm_tensor_absorb_right_2plaq(`, with the argument list adjusted. Expect `test_eager_sweep_runs_the_absorb_moves_under_jit` to fail.

- [ ] **Step 9: Measure**

Run the Task 1 chi=8 benchmark into `bench_t3_chi8.log`. Record the sweep 1 compile count and the converge wall in the ledger next to the baseline.

---

### Task 4: Split the symmetric projector around the M′ cut

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:_compute_2x2_projector_symmetric` (lines 1228–1543). Split it into three functions; the old function becomes their composition.
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py:_compute_plaquette_projector_pair` (lines 387–463). Route the corners and the two projector stages through `_ej.call` on the eager symmetric branch.
- Test: `tests/test_ctm_eager_sweep_jit.py` (add tests)

**Interfaces:**
- Consumes: `_ej.call` (Task 2); the fixture and helpers in `tests/test_ctm_eager_sweep_jit.py` (Task 3).
- Produces, in `_ctm_tensor_projector_2x2.py`:
  - `_symmetric_projector_prime(Q_TL, Q_TR, Q_BL, Q_BR, *, direction: str) -> tuple[SymmetricTensor, SymmetricTensor, SymmetricTensor]` returns `(M_prime_T, first_half, second_half)`. This is stages 1–3 and the M′ contraction of stage 4, line for line from the current body.
  - `_prime_meta(direction: str) -> dict`: the static labels. Keys: `prime_order`, `mp_left_labels`, `mp_right_labels`, `first_outer_labels`, `second_outer_labels`.
  - `_symmetric_projector_cut(M_prime_T, *, direction, chi, base_charges, incoming_chi_charges, Q_TL, Q_TR, Q_BL, Q_BR) -> tuple[SymmetricTensor, jax.Array, SymmetricTensor]` returns `(U_Mp_T, S_Mp, Vh_Mp_T)` before the gauge fix. This is the stage-4 SVD branch (`base_charges is None`, traced, eager plus `_retruncate_chi_bond`) and `_require_svd_connected` on M′, unchanged. The Q arguments are still needed for `_incoming_chi_charges` on the traced branch.
  - `_symmetric_projector_finish(M_prime_T, first_half, second_half, U_Mp_T, S_Mp, Vh_Mp_T, *, direction, projector_backward) -> tuple[SymmetricTensor, SymmetricTensor, jax.Array, jax.Array]`: the M′ gauge fix, eps_T and smallest_S, stage 5 and stage 6, unchanged.
  - `_compute_2x2_projector_symmetric` becomes `finish(prime, cut(...))`, with the same signature and return value.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_ctm_eager_sweep_jit.py`)

```python
def test_eager_sweep_runs_the_projector_stages_under_jit(sites, monkeypatch):
    seen = []
    raw_jitted = ej._jitted

    def spy(fn, static_argnames):
        seen.append(fn.__name__)
        return raw_jitted(fn, static_argnames)

    monkeypatch.setattr(ej, "_jitted", spy)
    _sweeps(sites, CHI, 1, True, monkeypatch)
    for name in (
        "_build_enlarged_corner",
        "_symmetric_projector_prime",
        "_symmetric_projector_finish",
    ):
        assert name in seen, name


def test_jitted_eager_sweep_compiles_far_less(sites, monkeypatch):
    """Mechanism: count XLA backend compiles in one cold eager sweep."""
    n = [0]

    def listener(name, dur, **_):
        if name == "/jax/core/compile/backend_compile_duration":
            n[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        jax.clear_caches()
        ej._jitted.cache_clear()
        _sweeps(sites, CHI, 1, True, monkeypatch)
        on = n[0]
        jax.clear_caches()
        n[0] = 0
        _sweeps(sites, CHI, 1, False, monkeypatch)
        off = n[0]
    finally:
        jax.monitoring._unregister_event_duration_listener_by_callback(listener)
    assert on * 5 < off, (on, off)
```

Before relying on `jax.monitoring._unregister_event_duration_listener_by_callback`, check that it exists in JAX 0.10.2: `.venv/bin/python -c "import jax.monitoring as m; print([x for x in dir(m) if 'unregister' in x])"`. If it does not, use the name that exists, or leave the listener registered with a module-level counter that the test resets.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_sweep_jit.py -q > .../t4_red.log 2>&1`

Expected: `test_eager_sweep_runs_the_projector_stages_under_jit` FAILS on `_build_enlarged_corner`. `test_jitted_eager_sweep_compiles_far_less` FAILS with the `(on, off)` pair in the message; record that pair in the ledger, because it shows how much Task 3 alone saves.

- [ ] **Step 3: Split `_compute_2x2_projector_symmetric`**

Move code; do not rewrite it. In the same order:

1. `_prime_meta(direction)` returns the label tuples that the current body assigns in its `if direction in ("left", "right")` / `else` arms and its `prime_order` arms (`m1_left_labels`, …, `mp_left_labels`, `mp_right_labels`, `first_outer_labels`, `second_outer_labels`, `prime_order`). They are pure functions of `direction`.
2. `_symmetric_projector_prime` has the body from `# ---- Stage 1` down to the `M_prime_T = G.contract(...)` lines of stage 4, including both `_require_svd_connected` calls for M1 and M2. It returns `(M_prime_T, first_half, second_half)`.
3. `_symmetric_projector_cut` has the `if base_charges is None:` … `else:` block of stage 4 (traced and eager branches, `_LAYOUT_PROBE` append included) and the `_require_svd_connected(M_prime_T, ...)` call after it. It returns `(U_Mp_T, S_Mp, Vh_Mp_T)`.
4. `_symmetric_projector_finish` has `U_Mp_T, Vh_Mp_T = _gauge_fix_symmetric_svd(U_Mp_T, Vh_Mp_T)` down to the final `return`.
5. `_compute_2x2_projector_symmetric` keeps its docstring and its `direction` check, then:

```python
    M_prime_T, first_half, second_half = _symmetric_projector_prime(
        Q_TL, Q_TR, Q_BL, Q_BR, direction=direction
    )
    U_Mp_T, S_Mp, Vh_Mp_T = _symmetric_projector_cut(
        M_prime_T,
        direction=direction,
        chi=chi,
        base_charges=base_charges,
        incoming_chi_charges=incoming_chi_charges,
        Q_TL=Q_TL,
        Q_TR=Q_TR,
        Q_BL=Q_BL,
        Q_BR=Q_BR,
    )
    return _symmetric_projector_finish(
        M_prime_T,
        first_half,
        second_half,
        U_Mp_T,
        S_Mp,
        Vh_Mp_T,
        direction=direction,
        projector_backward=projector_backward,
    )
```

The traced branch of `_symmetric_projector_cut` reads `is_traced_inputs` from the Q blocks, as it does now. Keep that test on the Q tensors, not on M′, so that behaviour does not change.

- [ ] **Step 4: Check that the traced-step HLO is unchanged after the split**

Run the Task 3 `step_hlo.py` and diff it against `hlo_before.txt`.

Expected: no difference. If it differs, the split changed the op order: fix the split and do not go on.

- [ ] **Step 5: Route the eager symmetric branch through the helper**

At the top of `_ctm_tensor_moves.py`, add `from tenax.algorithms import _ctm_eager_jit as _ej` and `from tenax.algorithms import _ctm_tensor_projector_2x2 as _p2x2` next to the other `tenax.algorithms` imports. Check first with `grep -n "_ctm_tensor_moves" src/tenax/algorithms/_ctm_tensor_projector_2x2.py` that this makes no import cycle; if it does, import `_p2x2` inside the function and ledger why.

In `_ctm_tensor_moves.py:_compute_plaquette_projector_pair`, build the four corners through the helper, with the same arguments as now:

```python
    Q_TL = _ej.call(
        _build_enlarged_corner, env_TL.C1, env_TL.T1, env_TL.T4, a_TL,
        position="top_left", static_argnames=("position",),
    )
    Q_TR = _ej.call(
        _build_enlarged_corner, env_TR.C2, env_TR.T1, env_TR.T2, a_TR,
        position="top_right", static_argnames=("position",),
    )
    Q_BL = _ej.call(
        _build_enlarged_corner, env_BL.C4, env_BL.T3, env_BL.T4, a_BL,
        position="bottom_left", static_argnames=("position",),
    )
    Q_BR = _ej.call(
        _build_enlarged_corner, env_BR.C3, env_BR.T3, env_BR.T2, a_BR,
        position="bottom_right", static_argnames=("position",),
    )
```

Then, when every Q is a `SymmetricTensor`, `base_charges is not None` and the Q blocks are concrete (the existing eager case), replace the `_compute_2x2_projector(...)` call with:

```python
        M_prime_T, first_half, second_half = _ej.call(
            _p2x2._symmetric_projector_prime,
            Q_TL, Q_TR, Q_BL, Q_BR,
            direction=direction,
            static_argnames=("direction",),
        )
        U_Mp_T, S_Mp, Vh_Mp_T = _p2x2._symmetric_projector_cut(  # eager: reads S
            M_prime_T,
            direction=direction,
            chi=chi,
            base_charges=base_charges,
            incoming_chi_charges=incoming_chi_charges,
            Q_TL=Q_TL, Q_TR=Q_TR, Q_BL=Q_BL, Q_BR=Q_BR,
        )
        P_top_raw, P_bot_raw, eps_T, smallest_S = _ej.call(
            _p2x2._symmetric_projector_finish,
            M_prime_T, first_half, second_half, U_Mp_T, S_Mp, Vh_Mp_T,
            direction=direction,
            projector_backward=projector_backward,
            static_argnames=("direction", "projector_backward"),
        )
```

Leave every other case (dense, traced, `base_charges is None`, `projector_method != "svd"` if that dispatch exists in `_compute_2x2_projector`) on the existing `_compute_2x2_projector(...)` call. Read `_compute_2x2_projector` lines 640–760 first and copy its exact dispatch conditions into a helper `_is_eager_symmetric_svd_case(...)`, so the eager-jit branch is taken exactly where `_compute_2x2_projector_symmetric` would have taken its eager branch. Also run the two `_half_to_chi_new_*` calls through `_ej.call`.

- [ ] **Step 6: Add the Review Focus tests**

```python
def test_degenerate_cut_gives_the_same_energy(monkeypatch):
    """Review Focus 1: U(1) D=2 chi=8 cuts a three-fold degenerate set (#1098
    test).  The jitted M' can keep another tied slot, so compare energies."""
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site

    A0, B0 = heisenberg_u1sz_init_pair(D, jax.random.PRNGKey(0))
    su = iPEPSConfig(
        max_bond_dim=D, num_imaginary_steps=200, dt=0.05, ctm=CTMConfig(chi=8)
    )
    _, (A, B), _ = ipeps(xxz_gate(0.3), (A0, B0), su, compute_energy=False)
    A = A * (1.0 / (A.norm() + 1e-10))
    B = B * (1.0 / (B.norm() + 1e-10))
    g = xxz_gate(0.3).todense()
    energies = []
    for enabled in (False, True):
        monkeypatch.setattr(ej, "_ENABLED", enabled)
        from tenax.algorithms.ipeps_ad_policy import ctm_converge_kwargs
        import dataclasses

        cfg = dataclasses.replace(
            CTMConfig(chi=8), conv_tol=1e-12, max_iter=200, min_iter=10,
            plateau_patience=None, on_unconverged="warn",
        )
        envs, _ = cpl.python_loop_ctm_converge(
            {(0, 0): A, (1, 0): B}, CHECKERBOARD_NEIGHBORS, **ctm_converge_kwargs(cfg)
        )
        energies.append(
            float(compute_energy_ctm_tensor_2site(A, B, envs[(0, 0)], envs[(1, 0)], g, 2))
        )
    assert abs(energies[0] - energies[1]) < 1e-9, energies


def test_fermionic_eager_sweeps_match(monkeypatch):
    """Review Focus 4: a graded Z2 pair through the jitted pieces."""
    sites = _fermionic_pair()
    plain = _sweeps(sites, 8, 2, False, monkeypatch)
    jitted = _sweeps(sites, 8, 2, True, monkeypatch)
    _assert_same_envs(plain, jitted)


def test_rank0_projector_still_raises_the_907_diagnosis(monkeypatch):
    """Review Focus 5: the #907 ValueError from inside the jitted prime stage."""
    sites = _confined_pair()
    for enabled in (False, True):
        with pytest.raises(ValueError, match="rank-0 SVD bond"):
            _sweeps(sites, 4, 1, enabled, monkeypatch)
```

Build `_fermionic_pair()` and `_confined_pair()` from fixtures that already exist; do not write new tensor constructions:

- `_fermionic_pair`: copy the graded Z2 random pair from `tests/test_ctm_674_fermionic_fused.py` (grep `def _` in that file for the pair builder) and normalise each tensor.
- `_confined_pair`: copy the #905/#907 confined fixture. Find it with `grep -rln "rank-0 SVD bond" tests/` and reuse its construction.

If either fixture is not in a reusable form, ledger a ruling with the file and line you copied from.

- [ ] **Step 7: Run all the tests in the file**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_eager_sweep_jit.py -q > .../t4_green.log 2>&1`

Expected: 7 passed.

- [ ] **Step 8: Run the projector and #1098 regression files**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest tests/test_ctm_2x2_projector_symmetric.py tests/test_ctm_670_symmetric_2x2.py tests/test_ctm_674_fermionic_fused.py tests/test_ctm_traced_chi_inventory_929.py tests/test_enlarged_corner_flow_invariant_834.py tests/test_ctm_symmetric_layout_adapts_1098.py tests/test_ipeps_u1sz.py -q -m "not slow" > .../t4_regress.log 2>&1`

Expected: the same pass count as on `origin/main`. Run the same command there first and record both counts.

- [ ] **Step 9: Commit, then mutation-test**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py src/tenax/algorithms/_ctm_tensor_moves.py tests/test_ctm_eager_sweep_jit.py
git commit -m "perf(ctm): jit the symmetric projector around the eager M' cut"
```

Mutants:

- M5: in `_symmetric_projector_finish`, delete the line `U_Mp_T, Vh_Mp_T = _gauge_fix_symmetric_svd(U_Mp_T, Vh_Mp_T)`. Expect `test_jitted_eager_sweeps_match_plain_eager_sweeps` to fail. If it does not fail, the test cannot see the gauge, and the ledger must say so.
- M6: in `_compute_plaquette_projector_pair`, route the eager case back to `_compute_2x2_projector(...)`. Expect `test_eager_sweep_runs_the_projector_stages_under_jit` and `test_jitted_eager_sweep_compiles_far_less` to fail.
- M7: in `_symmetric_projector_cut`, call `_retruncate_chi_bond(..., chi=chi - 1)`. Expect the equivalence test to fail on the layout assertion.

---

### Task 5: Measure, then open the PR

**Files:**
- Modify: none in `src/`. The PR body carries the numbers.

- [ ] **Step 1: Measure at chi=8 on GPU 1**

Run the Task 1 benchmark (`--D 3 --chi 8`) into `bench_final_chi8.log`.

Acceptance:
- Sweep 1 compiles are at most 10% of the Task 1 baseline (≤ 325).
- Converge wall is at most 50% of the baseline.

If either fails, find out which pieces still compile eagerly. Rerun with `JAX_LOG_COMPILES=1` and count the compile log lines by function name. Ledger the residual, then decide whether to wrap `_charge_sorted_envs` or `_retruncate_chi_bond` slices next. Each one is a new task with its own test, and neither is done without that evidence.

- [ ] **Step 2: Measure at chi=16 on GPU 1 (before and after)**

Run the benchmark at `--chi 16 --max-iter 40` on `origin/main` (in a separate worktree with its own venv) and on this branch, one after the other, never in parallel. Each run is long, so start it in the background, with output redirected to a file and a fail-closed watcher.

Record for each run: sweep 1 and sweep 2 compiles, sweep walls, converge wall, and peak RSS (`/usr/bin/time -v`, or `ps` sampling), because a jitted stage can raise peak memory.

- [ ] **Step 3: Run the core gate**

Run: `JAX_PLATFORMS=cpu taskset -c 0-15 .venv/bin/python -m pytest -m core -q > .../core.log 2>&1`

Expected: all pass.

- [ ] **Step 4: Push and open the PR**

- Push `perf/eager-layout-sweep-jit`. Open a PR to `main` with the 🤖 marker, the Problem record table, the before/after table from Steps 1–2, the HLO-unchanged evidence (Task 3 Step 6, Task 4 Step 4) and the mutation table.
- Do not arm `--auto`; arming is the user's call.
- Read Codex with `--paginate` before asking.

---

## Out of scope (separate concerns, separate branches)

- **Restarting the S5 chi=16 run with this fix.** The harness worktree is on `a00c38c`, which predates the merged stack (#1130–#1135), so using the fix there means rebuilding the harness on `main`. That is S5 tuning and needs the user.
- **Padding block shapes to fixed buckets,** so different layouts share compiles. This is a larger design change; only consider it if Task 5's chi=16 numbers show per-layout compile time still dominates.
- **The #1135 per-optimizer-step layout recheck** goes through `_eager_ctm_sweep` and gets the fix for free. No change there.
