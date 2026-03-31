# Multi-GPU Sharded DMRG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Shard the dense JIT DMRG sweep across multiple GPUs via JAX GSPMD to enable chi=1000-2000.

**Architecture:** Add a thin sharding wrapper around the existing `_jit_sweep_loop` that creates a device mesh, applies `NamedSharding` to shard the chi dimension across devices, and lets XLA handle communication. A new `accelerator="sharded"` dispatch option routes through this wrapper. No changes to the core sweep internals.

**Tech Stack:** JAX `jax.sharding.Mesh`, `NamedSharding`, `PartitionSpec`; existing `_jit_sweep_loop` from `_jit_sweep.py`.

---

### Task 1: Sharded Sweep Wrapper

**Files:**
- Modify: `src/tenax/algorithms/_jit_sweep.py:688-790`
- Test: `tests/test_jit_sweep.py`

**Step 1: Write the failing test**

Add to `tests/test_jit_sweep.py`:

```python
class TestShardedSweep:
    """Test multi-device sharded DMRG sweep."""

    def test_sharded_sweep_matches_single_device(self):
        """Sharded sweep must produce same energy as single-device JIT."""
        from tenax.algorithms._jit_sweep import (
            jit_dmrg_sweep_dense,
            jit_dmrg_sweep_dense_sharded,
        )

        L = 6
        chi_max = 8
        num_sweeps = 4

        mpo_tn = _build_dense_heisenberg(L)
        mpo_raw = [mpo_tn.get_tensor(i).todense() for i in range(L)]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mps_tn = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        mps_raw = [mps_tn.get_tensor(i).todense() for i in range(L)]

        # Single-device JIT
        energies_jit, _ = jit_dmrg_sweep_dense(
            mps_raw, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=20
        )

        # Sharded (may use 1 or more devices)
        import warnings as w2
        with w2.catch_warnings():
            w2.simplefilter("ignore", DeprecationWarning)
            mps_tn2 = build_random_mps(L=L, physical_dim=2, bond_dim=chi_max, seed=42)
        mps_raw2 = [mps_tn2.get_tensor(i).todense() for i in range(L)]

        energies_sharded, _ = jit_dmrg_sweep_dense_sharded(
            mps_raw2, mpo_raw, chi_max, num_sweeps=num_sweeps, lanczos_max_iter=20
        )

        # Energies must match within numerical precision
        for i in range(len(energies_jit)):
            np.testing.assert_allclose(
                energies_sharded[i], energies_jit[i], atol=1e-6,
                err_msg=f"Sweep {i}: sharded={energies_sharded[i]:.8f} vs jit={energies_jit[i]:.8f}",
            )
```

**Step 2: Run test to verify it fails**

Run: `cd /home/yjkao/tenax-tpu-gpu && uv run pytest tests/test_jit_sweep.py::TestShardedSweep::test_sharded_sweep_matches_single_device -xvs`
Expected: FAIL with `ImportError: cannot import name 'jit_dmrg_sweep_dense_sharded'`

**Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/_jit_sweep.py` after the `jit_dmrg_sweep_dense` function (~line 738):

```python
def jit_dmrg_sweep_dense_sharded(
    mps_tensors: list[jax.Array],
    mpo_tensors: list[jax.Array],
    chi_max: int,
    num_sweeps: int = 10,
    lanczos_max_iter: int = 20,
) -> tuple[list[float], list[jax.Array]]:
    """Run DMRG sweeps sharded across multiple devices via GSPMD.

    Same interface as :func:`jit_dmrg_sweep_dense` but distributes the
    MPS bond dimension across all available JAX devices using
    ``NamedSharding``.  XLA's GSPMD compiler automatically inserts
    communication (all-reduce, all-gather) as needed for einsums and SVD.

    Falls back to single-device JIT if only one device is available.

    Args:
        mps_tensors:  List of L MPS site tensors, each ``(chi_l, d, chi_r)``.
        mpo_tensors:  List of L MPO site tensors, each ``(D_w_l, d, d, D_w_r)``.
        chi_max:      Maximum bond dimension for MPS.
        num_sweeps:   Number of full (L->R + R->L) sweeps.
        lanczos_max_iter: Number of Lanczos iterations per site update.

    Returns:
        Tuple of ``(energies, mps_out)`` where *energies* is a list of
        floats (one per sweep) and *mps_out* is a list of L JAX arrays.
    """
    devices = jax.devices()
    if len(devices) < 2:
        # Single device — fall back to non-sharded JIT
        return jit_dmrg_sweep_dense(
            mps_tensors, mpo_tensors, chi_max, num_sweeps, lanczos_max_iter
        )

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    L = len(mps_tensors)
    d = mps_tensors[0].shape[1]
    D_w_max = max(max(W.shape[0], W.shape[3]) for W in mpo_tensors)
    dtype = mps_tensors[0].dtype

    # Pad and stack (same as jit_dmrg_sweep_dense)
    mps_stack = jnp.zeros((L, chi_max, d, chi_max), dtype=dtype)
    for i, M in enumerate(mps_tensors):
        chi_l, _, chi_r = M.shape
        mps_stack = mps_stack.at[i, :chi_l, :, :chi_r].set(M)

    W_stack = jnp.zeros((L, D_w_max, d, d, D_w_max), dtype=dtype)
    for i, W in enumerate(mpo_tensors):
        dw_l, _, _, dw_r = W.shape
        W_stack = W_stack.at[i, :dw_l, :, :, :dw_r].set(W)

    # Create mesh and shard MPS along chi dimensions
    mesh = Mesh(jnp.array(devices), axis_names=("chi",))
    mps_sharding = NamedSharding(mesh, P(None, "chi", None, "chi"))
    w_sharding = NamedSharding(mesh, P())  # replicated
    out_sharding = NamedSharding(mesh, P())  # energies replicated

    # Shard inputs
    mps_stack = jax.device_put(mps_stack, mps_sharding)
    W_stack = jax.device_put(W_stack, w_sharding)

    # JIT with sharding constraints
    @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
    def _sharded_sweep(mps, W, L, chi, D_w, d, n_sweeps, lanczos_iter):
        return _jit_sweep_loop.__wrapped__(
            mps, W, L, chi, D_w, d, n_sweeps, lanczos_iter
        )

    energies, mps_final = _sharded_sweep(
        mps_stack, W_stack, L, chi_max, D_w_max, d, num_sweeps, lanczos_max_iter
    )

    mps_out = [mps_final[i] for i in range(L)]
    return [float(e) for e in energies], mps_out
```

**Important note for the implementer:** The `_jit_sweep_loop` is already decorated with `@jax.jit`. We cannot re-jit a jitted function with different sharding. There are two approaches:

**Option A:** Call the unwrapped function via `_jit_sweep_loop.__wrapped__` (if using `functools.partial(jax.jit, ...)`).

**Option B:** Refactor `_jit_sweep_loop` to separate the pure function from the `@jit` decorator — extract the body into `_sweep_loop_impl` and have `_jit_sweep_loop = jax.jit(_sweep_loop_impl, static_argnums=...)`.

Option B is cleaner. Refactor like this:

```python
def _sweep_loop_impl(
    mps_stack: jax.Array,
    W_stack: jax.Array,
    L: int,
    chi_max: int,
    D_w: int,
    d: int,
    num_sweeps: int,
    lanczos_max_iter: int,
) -> jax.Array:
    """Pure sweep loop implementation (no @jit decorator)."""
    # ... existing body of _jit_sweep_loop ...

# Single-device JIT version (unchanged behavior)
_jit_sweep_loop = functools.partial(
    jax.jit, static_argnums=(2, 3, 4, 5, 6, 7)
)(_sweep_loop_impl)
```

Then `jit_dmrg_sweep_dense_sharded` can re-jit `_sweep_loop_impl` with sharding.

**Step 4: Run test to verify it passes**

Run: `cd /home/yjkao/tenax-tpu-gpu && uv run pytest tests/test_jit_sweep.py::TestShardedSweep::test_sharded_sweep_matches_single_device -xvs`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/yjkao/tenax-tpu-gpu
git add src/tenax/algorithms/_jit_sweep.py tests/test_jit_sweep.py
git commit -m "feat: add multi-GPU sharded DMRG sweep via GSPMD"
```

---

### Task 2: DMRGConfig Dispatch for `accelerator="sharded"`

**Files:**
- Modify: `src/tenax/algorithms/dmrg.py:96-131` (DMRGConfig docstring)
- Modify: `src/tenax/algorithms/dmrg.py:236-265` (dispatch block)
- Test: `tests/test_jit_sweep.py`

**Step 1: Write the failing test**

Add to `TestShardedSweep` in `tests/test_jit_sweep.py`:

```python
    def test_sharded_dispatch_via_dmrg(self):
        """accelerator='sharded' produces correct energy via dmrg() entry point."""
        L = 6
        chi_max = 8
        num_sweeps = 6

        mpo_tn = _build_dense_heisenberg(L)

        mps = FiniteMPS.random(L, d=2, chi=chi_max, key=jax.random.PRNGKey(42))
        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            accelerator="sharded",
        )
        result = dmrg(mpo_tn, mps, config)

        assert np.isfinite(result.energy)
        assert result.energy < 0.0

        # Compare with Python path
        mps2 = FiniteMPS.random(L, d=2, chi=chi_max, key=jax.random.PRNGKey(42))
        config_py = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=num_sweeps,
            lanczos_max_iter=20,
            accelerator="off",
        )
        result_py = dmrg(mpo_tn, mps2, config_py)

        np.testing.assert_allclose(
            result.energy, result_py.energy, atol=1e-4,
            err_msg=f"Sharded={result.energy:.8f} vs Python={result_py.energy:.8f}",
        )

    def test_sharded_fallback_single_device(self):
        """accelerator='sharded' with 1 device falls back to JIT silently."""
        # This test verifies the code path works; on a multi-GPU machine
        # it still exercises the sharded path (which is fine).
        L = 4
        chi_max = 4
        mpo_tn = _build_dense_heisenberg(L)
        mps = FiniteMPS.random(L, d=2, chi=chi_max, key=jax.random.PRNGKey(0))
        config = DMRGConfig(
            max_bond_dim=chi_max,
            num_sweeps=3,
            lanczos_max_iter=10,
            accelerator="sharded",
        )
        result = dmrg(mpo_tn, mps, config)
        assert np.isfinite(result.energy)
        assert result.energy < 0.0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/yjkao/tenax-tpu-gpu && uv run pytest tests/test_jit_sweep.py::TestShardedSweep -xvs`
Expected: FAIL with `ValueError: accelerator must be 'auto', 'jit', or 'off', got 'sharded'`

**Step 3: Implement dispatch**

In `src/tenax/algorithms/dmrg.py`:

**3a.** Update the `accelerator` docstring in `DMRGConfig` (around line 102):

```python
        accelerator:        Backend dispatch mode for the DMRG sweep:
                            ``"auto"`` (default) — GPU/TPU uses JIT path; CPU with
                            symmetric tensors uses numpy/Cython path; CPU with
                            dense tensors uses JIT path.
                            ``"jit"`` — force JIT-compiled ``lax.scan`` sweep
                            (requires dense tensors; silently falls back for
                            symmetric or 1-site DMRG).
                            ``"sharded"`` — like ``"jit"`` but shards the bond
                            dimension across all available devices via GSPMD.
                            Falls back to ``"jit"`` if only one device is present.
                            ``"off"`` — always use the existing Python sweep loop.
```

**3b.** Update the accelerator validation (line 237):

```python
    if config.accelerator not in ("auto", "jit", "sharded", "off"):
        raise ValueError(
            f"accelerator must be 'auto', 'jit', 'sharded', or 'off', "
            f"got {config.accelerator!r}"
        )
```

**3c.** Add sharded dispatch before the existing JIT block. Insert right before line 252 (`if use_jit and config.two_site and not use_symmetric:`):

```python
    use_sharded = config.accelerator == "sharded"

    if use_sharded and config.two_site and not use_symmetric:
        from tenax.algorithms._jit_sweep import jit_dmrg_sweep_dense_sharded

        # Same warmup logic as JIT path: check if bonds are saturated
        all_saturated = all(
            mps_tensors[idx].todense().shape[-1] >= config.max_bond_dim
            for idx in range(L - 1)
        )

        if all_saturated:
            # All bonds at chi_max — go straight to sharded JIT
            raw_mps = [t.todense() for t in mps_tensors]
            raw_mpo = [t.todense() for t in mpo_tensors]
            energies, mps_out_raw = jit_dmrg_sweep_dense_sharded(
                raw_mps, raw_mpo,
                chi_max=config.max_bond_dim,
                num_sweeps=config.num_sweeps,
                lanczos_max_iter=config.lanczos_max_iter,
            )

            sym = U1Symmetry()
            result_mps_tensors = []
            for i, orig_t in enumerate(mps_tensors):
                new_indices = []
                for leg_idx, orig_idx in enumerate(orig_t.indices):
                    padded_dim = mps_out_raw[i].shape[leg_idx]
                    if padded_dim == orig_idx.dim:
                        new_indices.append(orig_idx)
                    else:
                        new_charges = np.zeros(padded_dim, dtype=np.int32)
                        new_indices.append(
                            TensorIndex(sym, new_charges, orig_idx.flow, label=orig_idx.label)
                        )
                result_mps_tensors.append(DenseTensor(mps_out_raw[i], tuple(new_indices)))
            result_mps = FiniteMPS.from_tensors(result_mps_tensors)

            converged = (
                len(energies) >= 2
                and abs(energies[-1] - energies[-2]) < config.convergence_tol
            )
            return DMRGResult(
                energy=energies[-1] if energies else 0.0,
                energies_per_sweep=energies,
                mps=result_mps,
                truncation_errors=[],
                converged=converged,
            )
        else:
            # Needs warmup — fall through to the existing JIT warmup block
            # by setting use_jit = True (the warmup block will handle it,
            # then we could switch to sharded for the JIT phase, but for
            # simplicity we use regular JIT for warmup+JIT since warmup
            # grows chi and sharded only helps at large chi)
            use_jit = True
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/yjkao/tenax-tpu-gpu && uv run pytest tests/test_jit_sweep.py::TestShardedSweep -xvs`
Expected: All 3 tests PASS

**Step 5: Run full core tests for regressions**

Run: `cd /home/yjkao/tenax-tpu-gpu && uv run pytest -m core -x -q`
Expected: All pass, no regressions

**Step 6: Commit**

```bash
cd /home/yjkao/tenax-tpu-gpu
git add src/tenax/algorithms/dmrg.py tests/test_jit_sweep.py
git commit -m "feat: add accelerator='sharded' dispatch for multi-GPU DMRG"
```

---

### Task 3: Benchmark Script

**Files:**
- Create: `bench_sharded_dmrg.py` (project root, not in src/)

**Step 1: Write the benchmark**

```python
"""Benchmark: single-GPU JIT vs multi-GPU sharded DMRG."""

import time
import warnings

import jax
import numpy as np

warnings.filterwarnings("ignore")

from tenax.algorithms.auto_mpo import build_auto_mpo
from tenax.algorithms.dmrg import DMRGConfig, dmrg
from tenax.core.mps import FiniteMPS

print(f"JAX backend: {jax.default_backend()}")
devices = jax.devices()
for d in devices:
    print(f"  {d}")
print()


def _build_dense_heisenberg(L):
    terms = []
    for i in range(L - 1):
        terms.append((1.0, "Sz", i, "Sz", i + 1))
        terms.append((0.5, "Sp", i, "Sm", i + 1))
        terms.append((0.5, "Sm", i, "Sp", i + 1))
    return build_auto_mpo(terms, L=L, symmetric=False)


results = []

for L in [20, 40]:
    for chi in [1000, 1500, 2000]:
        num_sweeps = 2  # few sweeps, just timing
        mpo = _build_dense_heisenberg(L)

        for mode in ["jit", "sharded"]:
            # Warmup run (JIT compilation)
            mps_w = FiniteMPS.random(L, d=2, chi=chi, key=jax.random.PRNGKey(0))
            _ = dmrg(
                mpo, mps_w,
                DMRGConfig(max_bond_dim=chi, num_sweeps=1, accelerator=mode),
            )

            # Timed run
            mps = FiniteMPS.random(L, d=2, chi=chi, key=jax.random.PRNGKey(42))
            config = DMRGConfig(
                max_bond_dim=chi,
                num_sweeps=num_sweeps,
                lanczos_max_iter=20,
                convergence_tol=0.0,
                accelerator=mode,
            )
            t0 = time.perf_counter()
            result = dmrg(mpo, mps, config)
            t_elapsed = time.perf_counter() - t0

            results.append((L, chi, mode, t_elapsed, result.energy))
            print(
                f"L={L:3d}  chi={chi:5d}  {mode:>8s}  "
                f"time={t_elapsed:8.2f}s  E={result.energy:.6f}"
            )
        print()

print("--- Summary ---")
print(f"{'L':>4s}  {'chi':>5s}  {'Mode':>8s}  {'Time':>8s}")
for L, chi, mode, t, _ in results:
    print(f"{L:4d}  {chi:5d}  {mode:>8s}  {t:8.2f}s")
```

**Step 2: Run it (manual, not in CI)**

Run: `cd /home/yjkao/tenax-tpu-gpu && PYTHONPATH=src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 python -u bench_sharded_dmrg.py`

This is a manual benchmark, not an automated test. Results will show whether sharding provides speedup at chi=1000-2000.

**Step 3: Commit**

```bash
cd /home/yjkao/tenax-tpu-gpu
git add bench_sharded_dmrg.py
git commit -m "bench: add multi-GPU sharded vs single-GPU DMRG benchmark"
```

---

### Task 4: Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `src/tenax/algorithms/dmrg.py` (DMRGConfig docstring — already done in Task 2)

**Step 1: Update README**

Add to the GPU/TPU DMRG bullet in `README.md`:

```markdown
- **GPU/TPU-accelerated DMRG** — JIT-compiled sweeps via `jax.lax.scan` for dense tensors and per-operation JIT for block-sparse symmetric tensors; automatic warmup-to-JIT transition when bond dimensions are growing; multi-GPU sharding via GSPMD for large bond dimensions (`DMRGConfig(accelerator="sharded")`)
```

**Step 2: Commit**

```bash
cd /home/yjkao/tenax-tpu-gpu
git add README.md
git commit -m "docs: document multi-GPU sharded DMRG in README"
```
