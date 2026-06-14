# U(1)-Sz Heisenberg Enablement + Feasibility Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a U(1)-Sz–charged Heisenberg gate + symmetric 2-site init, then run a feasibility spike (smoke + symmetric-vs-dense correctness + perf/block signal) that returns a GO/NO-GO verdict on running U(1)-Sz Heisenberg through tenax's iPEPS-AD path.

**Architecture:** `heisenberg_gate()` is already the unrotated `SzSz + ½(S+S− + S−S+)` with U(1) indices carrying *trivial* charges. The novelty is *non-trivial* charges (physical `[+1,−1]` = 2·Sz; non-trivially-blocked site tensors), which is what exercises the documented "U(1) non-trivial charges fail in the production absorb step" coverage gap (`examples/bench_symmetric_ad_batching_566.py:57`). We build the charged gate + a non-trivially-blocked 2-site init as production code (with tests), then a study script that runs the symmetric path against a dense baseline from the *same* densified init. If the absorb step crashes, a conditional task fixes it properly with a regression test.

**Tech Stack:** Python, JAX (x64), tenax (`SymmetricTensor`, `U1Symmetry`, `TensorIndex.from_charges`, `optimize_gs_ad` 2-site). Tests: pytest (`-m core`).

**Spec:** `docs/superpowers/specs/2026-06-14-u1sz-heisenberg-enablement-design.md`

---

## File Structure

- `src/tenax/algorithms/ipeps.py` — add `heisenberg_gate_u1sz()` (charged gate) and `heisenberg_u1sz_init_pair()` (symmetric 2-site init). These are small, self-contained constructors that belong next to `heisenberg_gate()`.
- `src/tenax/__init__.py` — export the two new public functions (`__all__`).
- `README.md` — mention the U(1)-Sz Heisenberg helpers in the features/example list.
- `tests/test_ipeps_u1sz.py` — unit tests for the gate + init (charge structure, dense round-trip, non-trivial blocking, one optimization step runs).
- `examples/bench_heisenberg_u1sz_spike.py` — the feasibility-spike study script.
- *(Conditional)* `src/tenax/algorithms/<absorb-step file>` + `tests/test_<...>.py` — absorb-step fix + regression test, only if the gap blocks.
- `docs/superpowers/handoffs/2026-06-14-u1sz-heisenberg-spike.md` — verdict writeup.

---

## Task 1: U(1)-Sz charged Heisenberg gate

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py` (add `heisenberg_gate_u1sz`, after `heisenberg_gate` at line 63)
- Modify: `src/tenax/__init__.py` (`__all__` + import)
- Modify: `README.md`
- Test: `tests/test_ipeps_u1sz.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipeps_u1sz.py`:

```python
"""Tests for U(1)-Sz–symmetric Heisenberg helpers (issue #570 follow-up)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax import heisenberg_gate
from tenax.algorithms.ipeps import heisenberg_gate_u1sz
from tenax.core.tensor import SymmetricTensor


class TestHeisenbergGateU1Sz:
    def test_returns_symmetric_tensor(self):
        gate = heisenberg_gate_u1sz()
        assert isinstance(gate, SymmetricTensor)

    def test_dense_roundtrip_matches_plain_gate(self):
        """Same physics as heisenberg_gate(); only the charges differ."""
        gate_u1 = heisenberg_gate_u1sz()
        gate_plain = heisenberg_gate()
        np.testing.assert_allclose(
            np.asarray(gate_u1.todense()),
            np.asarray(gate_plain.todense()),
            atol=1e-12,
        )

    def test_physical_charges_are_sz(self):
        """Physical legs carry Sz charges [+1, -1] (units of 2*Sz)."""
        gate = heisenberg_gate_u1sz()
        # si leg (index 0) charges, in original basis order
        charges = np.asarray(gate.indices[0].charges)
        assert sorted(charges.tolist()) == [-1, 1]

    def test_is_nontrivially_blocked(self):
        """Sz conservation splits H into more than one charge block."""
        gate = heisenberg_gate_u1sz()
        assert len(gate.blocks) > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py -x -q`
Expected: FAIL with `ImportError: cannot import name 'heisenberg_gate_u1sz'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tenax/algorithms/ipeps.py`, immediately after `heisenberg_gate` (line 63), add:

```python
def heisenberg_gate_u1sz(dtype=jnp.float64) -> SymmetricTensor:
    """Build the 2-site Heisenberg Hamiltonian as a U(1)-Sz SymmetricTensor.

    Identical numerics to :func:`heisenberg_gate`
    (``H = Sz Sz + 0.5 (S+ S- + S- S+)``) but the physical legs carry
    U(1) charges ``[+1, -1]`` for ``{up, down}`` (units of ``2*Sz``,
    matching the ``S+``/``S-`` charge-(+/-)2 convention in
    ``tests/test_observables.py``). Sz conservation makes the gate
    block-sparse. Returned as a 4-leg ``SymmetricTensor`` with labels
    ``(si, sj, si_out, sj_out)``.
    """
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    sym = U1Symmetry()
    charges = np.array([1, -1], dtype=np.int32)  # Sz = +1/2, -1/2 -> 2*Sz
    indices = (
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="si"),
        TensorIndex.from_charges(sym, charges.copy(), FlowDirection.IN, label="sj"),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="si_out"
        ),
        TensorIndex.from_charges(
            sym, charges.copy(), FlowDirection.OUT, label="sj_out"
        ),
    )
    return SymmetricTensor.from_dense(H.reshape(2, 2, 2, 2), indices)
```

Add `SymmetricTensor` to the existing tensor import at the top of the file. The line is currently:

```python
from tenax.core.tensor import DenseTensor, Tensor
```

Change it to:

```python
from tenax.core.tensor import DenseTensor, SymmetricTensor, Tensor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py -x -q`
Expected: PASS (4 tests). If `from_dense` raises a ValueError about elements outside valid sectors, the charge assignment is wrong — the dense `H` must be exactly Sz-block-diagonal under charges `[+1,−1]` (it is, mathematically); investigate before proceeding.

- [ ] **Step 5: Export the new public API**

In `src/tenax/__init__.py`, find where `heisenberg_gate` is imported and re-exported, and add `heisenberg_gate_u1sz` alongside it (both the `from tenax.algorithms.ipeps import (...)` line and the `__all__` list).

Run: `JAX_PLATFORMS=cpu uv run python -c "from tenax import heisenberg_gate_u1sz; print(type(heisenberg_gate_u1sz()).__name__)"`
Expected: `SymmetricTensor`

- [ ] **Step 6: Update README**

In `README.md`, find the features/helpers list that mentions `heisenberg_gate` and add a one-line entry for `heisenberg_gate_u1sz` (U(1)-Sz–symmetric Heisenberg gate). Keep wording consistent with the surrounding entries.

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps.py src/tenax/__init__.py README.md tests/test_ipeps_u1sz.py
git commit -m "feat(#570): U(1)-Sz Heisenberg gate (charged SymmetricTensor)"
```

---

## Task 2: Symmetric 2-site init with non-trivial Sz charges

**Files:**
- Modify: `src/tenax/algorithms/ipeps.py` (add `heisenberg_u1sz_init_pair`)
- Modify: `src/tenax/__init__.py` (`__all__` + import)
- Test: `tests/test_ipeps_u1sz.py` (add a class)

**Why:** The gate alone does not exercise the absorb-step gap — *site tensors with non-trivial charge blocks* do. This helper builds the `(A, B)` pair the spike feeds to 2-site `optimize_gs_ad`. Both tensors are target-0 Sz-conserving (physical `[+1,−1]`, virtual `[+1,−1]` per bond at D=2). A Sz-symmetric ansatz does not show a staggered moment (z-Néel breaks U(1)) but captures the Heisenberg energy in the Sz=0 sector; 2 independent tensors give the variational freedom.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ipeps_u1sz.py`:

```python
class TestHeisenbergU1SzInit:
    def test_pair_are_symmetric_tensors(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert isinstance(A, SymmetricTensor)
        assert isinstance(B, SymmetricTensor)

    def test_pair_have_five_legs_and_nontrivial_blocks(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, B = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        assert len(A.indices) == 5  # u, d, l, r, phys
        assert len(A.blocks) > 1    # non-trivially blocked -> exercises absorb step
        assert len(B.blocks) > 1

    def test_physical_leg_is_sz_charged(self):
        from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair

        A, _ = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        phys = np.asarray(A.indices[4].charges)  # phys is the 5th leg
        assert sorted(phys.tolist()) == [-1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py::TestHeisenbergU1SzInit -x -q`
Expected: FAIL with `ImportError: cannot import name 'heisenberg_u1sz_init_pair'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tenax/algorithms/ipeps.py`, after `heisenberg_gate_u1sz`, add:

```python
def heisenberg_u1sz_init_pair(D: int, key: jax.Array):
    """Build a random U(1)-Sz–symmetric 2-site iPEPS pair ``(A, B)``.

    Each site tensor has 5 legs ``(u, d, l, r, phys)`` with flows
    ``u=OUT, d=IN, l=OUT, r=IN, phys=IN`` (matching
    ``_build_initial_fpeps_tensor``). Physical charges are ``[+1, -1]``
    (2*Sz); virtual charges alternate ``+1, -1, +1, ...`` over the bond
    dimension ``D``. Both tensors are Sz-conserving (target 0); the AFM
    correlations emerge from optimization within the Sz=0 sector.

    Args:
        D:   Virtual bond dimension.
        key: JAX random key (split internally for A and B).

    Returns:
        Tuple ``(A, B)`` of SymmetricTensors.
    """
    sym = U1Symmetry()
    virt_charges = np.array([1 if i % 2 == 0 else -1 for i in range(D)], dtype=np.int32)
    phys_charges = np.array([1, -1], dtype=np.int32)

    indices = (
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, virt_charges.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )

    kA, kB = jax.random.split(key)
    A = SymmetricTensor.random_normal(indices, kA)
    B = SymmetricTensor.random_normal(indices, kB)
    return A, B
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py::TestHeisenbergU1SzInit -x -q`
Expected: PASS (3 tests). If `len(A.blocks)` is 1 or 0, the virtual/physical charge combination does not produce multiple conserving sectors — print `A._block_keys` and adjust `virt_charges` so `d+r+phys == u+l` has multiple solutions (it does for `{+1,−1}` legs).

- [ ] **Step 5: Export + commit**

Add `heisenberg_u1sz_init_pair` to `src/tenax/__init__.py` (`__all__` + import).

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py -q`
Expected: PASS (all 7 tests).

```bash
git add src/tenax/algorithms/ipeps.py src/tenax/__init__.py tests/test_ipeps_u1sz.py
git commit -m "feat(#570): U(1)-Sz symmetric 2-site iPEPS init pair"
```

---

## Task 3: One-step symmetric-vs-dense smoke test (the GO gate, as a unit test)

**Files:**
- Test: `tests/test_ipeps_u1sz.py` (add a class)

**Why:** This is checks 1 (runs) + 2a (contraction correctness) + 2b (right basin) of the GO gate, encoded as a fast test. From the *same* densified init, one symmetric and one dense optimization step must produce matching energies (~1e-8) — proving the block-sparse CTM/energy path is numerically correct against dense. This is the task most likely to surface the absorb-step gap. **If it raises inside the CTM absorb step, STOP and go to Task 5 (conditional fix) before continuing.**

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ipeps_u1sz.py`:

```python
class TestU1SzSymmetricMatchesDense:
    def test_one_step_symmetric_matches_dense(self):
        """Symmetric and dense agree from the same init after 1 step (~1e-8)."""
        from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad
        from tenax.algorithms.ipeps import (
            heisenberg_gate,
            heisenberg_u1sz_init_pair,
        )

        A_sym, B_sym = heisenberg_u1sz_init_pair(D=2, key=jax.random.PRNGKey(0))
        A_dense = A_sym.todense()
        B_dense = B_sym.todense()
        gate = heisenberg_gate().todense()  # dense gate, identical numerics

        config = iPEPSConfig(
            max_bond_dim=2,
            ctm=CTMConfig(chi=8, max_iter=20),
            gs_num_steps=1,
            unit_cell="2site",
        )

        # Symmetric run — this exercises the non-trivial-charge absorb step.
        (_, _), _, E_sym = optimize_gs_ad(gate, (A_sym, B_sym), config)
        # Dense run from the densified same init.
        (_, _), _, E_dense = optimize_gs_ad(gate, (A_dense, B_dense), config)

        assert np.isfinite(E_sym)
        np.testing.assert_allclose(float(E_sym), float(E_dense), atol=1e-8)
```

- [ ] **Step 2: Run the test**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py::TestU1SzSymmetricMatchesDense -x -q`

Three possible outcomes:
- **PASS** → checks 1 + 2a + 2b satisfied at one step. Proceed to Step 3, then Task 4.
- **RAISES in the CTM absorb step** (e.g. shape/charge mismatch, KeyError on a block) → the documented gap. Capture the full traceback into `docs/superpowers/handoffs/2026-06-14-u1sz-absorb-repro.txt`, then go to **Task 5**. Return here after the fix.
- **FAILS the allclose** (runs, energies differ) → check-2 failure: a charge-assignment / normalization / gauge bug. Diagnose before any perf claim (compare `A_sym.todense()` vs `A_dense`, verify the gate round-trip, check flows). Do not skip.

- [ ] **Step 3: Commit (once passing)**

```bash
git add tests/test_ipeps_u1sz.py
git commit -m "test(#570): U(1)-Sz symmetric path matches dense at one step"
```

---

## Task 4: Feasibility-spike study script

**Files:**
- Create: `examples/bench_heisenberg_u1sz_spike.py`

**Why:** The operational artifact: runs D=2, χ∈{8,16}, symmetric and dense from the same init, to a small multi-step budget, recording energies, the symmetric-vs-dense gap, timings, and the block-count structural proxy (the perf signal). JSON-checkpointed like `examples/bench_heisenberg_largeD.py`.

- [ ] **Step 1: Write the script**

Create `examples/bench_heisenberg_u1sz_spike.py`:

```python
#!/usr/bin/env python3
"""U(1)-Sz Heisenberg feasibility spike (issue #570 follow-up).

For each (D, chi) cell, runs a 2-site iPEPS-AD optimization twice from the
SAME densified init: once with U(1)-Sz SymmetricTensor site tensors
(block-sparse) and once dense. Records final energies, the
symmetric-vs-dense gap (correctness), timings (perf), and the symmetric
block count (structural proxy). This is a GO/NO-GO feasibility probe, not
a scaling sweep.

Usage::

    JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_u1sz_spike.py \\
        --D 2 --chi-list 8 16 --gs-steps 30 \\
        --json examples/heisenberg_u1sz_spike.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad  # noqa: E402
from tenax.algorithms.ipeps import (  # noqa: E402
    heisenberg_gate,
    heisenberg_u1sz_init_pair,
)

REF_ENERGY = -0.6694430


def _history(result):
    """Unpack optimize_gs_ad result with return_history=True (2site)."""
    (_, _), _, E_gs, history = result
    energies = history["energies"]
    step_times = history["step_times"]
    return {
        "E_final": float(min(energies)) if energies else float("nan"),
        "jit_compile_s": float(history["jit_compile_time"]),
        "warm_step_s": (
            statistics.median(step_times) if step_times else float("nan")
        ),
        "num_steps": int(history["num_steps"]),
        "converged": bool(history["converged"]),
    }


def run_cell(D: int, chi: int, gs_steps: int) -> dict:
    A_sym, B_sym = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    A_dense, B_dense = A_sym.todense(), B_sym.todense()
    gate = heisenberg_gate().todense()
    num_blocks = len(A_sym.blocks)

    config = iPEPSConfig(
        max_bond_dim=D,
        ctm=CTMConfig(chi=chi, max_iter=100, conv_tol=1e-8),
        gs_num_steps=gs_steps,
        gs_conv_criterion="grad_norm",
        gs_grad_norm_tol=1e-5,
        unit_cell="2site",
        return_history=True,
    )

    t0 = time.perf_counter()
    res_sym = optimize_gs_ad(gate, (A_sym, B_sym), config)
    sym_wall = time.perf_counter() - t0
    sym = _history(res_sym)

    t0 = time.perf_counter()
    res_dense = optimize_gs_ad(gate, (A_dense, B_dense), config)
    dense_wall = time.perf_counter() - t0
    dense = _history(res_dense)

    return {
        "D": D,
        "chi": chi,
        "num_blocks_sym": num_blocks,
        "E_sym": sym["E_final"],
        "E_dense": dense["E_final"],
        "dE_sym_vs_dense": sym["E_final"] - dense["E_final"],
        "dE_sym_vs_ref": sym["E_final"] - REF_ENERGY,
        "sym_total_wall_s": sym_wall,
        "dense_total_wall_s": dense_wall,
        "sym_warm_step_s": sym["warm_step_s"],
        "dense_warm_step_s": dense["warm_step_s"],
        "sym_jit_compile_s": sym["jit_compile_s"],
        "dense_jit_compile_s": dense["jit_compile_s"],
        "num_steps": sym["num_steps"],
        "converged": sym["converged"],
    }


def _write_json(path: str, meta: dict, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    tmp.rename(p)


def _load_rows(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open() as f:
            return json.load(f).get("rows", [])
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="U(1)-Sz Heisenberg spike (#570).")
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--chi-list", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--gs-steps", type=int, default=30)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    try:
        device_kind = jax.devices()[0].device_kind
    except Exception:
        device_kind = "unknown"

    meta = {
        "platform": platform.node(),
        "device_kind": device_kind,
        "x64": True,
        "ref_energy": REF_ENERGY,
        "D": args.D,
        "chi_list": args.chi_list,
        "gs_steps": args.gs_steps,
    }

    rows = _load_rows(args.json) if args.json else []
    done = {(r["D"], r["chi"]) for r in rows if "error" not in r}

    print(f"\nU(1)-Sz Heisenberg spike | ref E = {REF_ENERGY} | device {device_kind}")
    for chi in args.chi_list:
        if (args.D, chi) in done:
            continue
        try:
            row = run_cell(args.D, chi, args.gs_steps)
        except Exception as exc:
            row = {"D": args.D, "chi": chi, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps(row, indent=2))
        if args.json:
            _write_json(args.json, meta, rows)

    if args.json:
        print(f"\nResults saved to: {args.json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses/imports**

Run: `JAX_PLATFORMS=cpu uv run python -c "import ast; ast.parse(open('examples/bench_heisenberg_u1sz_spike.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
git add examples/bench_heisenberg_u1sz_spike.py
git commit -m "bench(#570): U(1)-Sz Heisenberg feasibility-spike script"
```

---

## Task 5 (CONDITIONAL): Fix the U(1) non-trivial-charge absorb-step gap

**Run this task ONLY if Task 3 raised inside the CTM absorb step.** Skip entirely if Task 3 passed.

**Files:**
- Read first: the CTM absorb/move code under `src/tenax/algorithms/` (start from the traceback frames; the absorb step lives in the CTM tensor-move/renormalization path — grep `absorb` and follow from `ctm_2site` / `_ctm_tensor_*`).
- Modify: the offending production file.
- Test: a regression test in `tests/` reproducing the captured failure.

- [ ] **Step 1: Capture a minimal repro**

Reduce the Task 3 failure to the smallest script that raises (single CTM move on the non-trivially-blocked symmetric pair, no optimization). Save as `tests/test_u1sz_absorb_regression.py` as an `xfail`-free test that currently raises. Record the exact exception + frame.

- [ ] **Step 2: Localize bounded vs unbounded**

Re-run the repro with a *bounded* charge proxy (rebuild the pair with `virt_charges`/`phys_charges` mapped through a `ZnSymmetry`-style cap, or clamp to `{0,1}`) to determine whether the bug is specific to unbounded U(1) charges or general to non-trivial charges. Note the finding in the handoff. This decides the fix's shape.

- [ ] **Step 3: Root-cause + fix**

Fix the absorb step so non-trivial U(1) charge blocks are handled (likely a block-key alignment / missing-sector / zero-block assumption). Keep the change minimal and within the symmetric path; do not alter the dense path. Follow the block-sparse coding rule (no `todense()` on the symmetric path).

- [ ] **Step 4: Make the regression test pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_u1sz_absorb_regression.py -x -q`
Expected: PASS.

- [ ] **Step 5: Re-run Task 3's gate**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py::TestU1SzSymmetricMatchesDense -x -q`
Expected: PASS (or a check-2 diagnosis if it now runs-but-differs).

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/ tests/test_u1sz_absorb_regression.py
git commit -m "fix(#570): U(1) non-trivial-charge CTM absorb step + regression test"
```

---

## Task 6: CPU smoke run + GO/NO-GO writeup

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-14-u1sz-heisenberg-spike.md`
- Create: `examples/heisenberg_u1sz_spike.json` (produced by the run)

- [ ] **Step 1: Run the spike on CPU**

Run:
```bash
JAX_PLATFORMS=cpu uv run python examples/bench_heisenberg_u1sz_spike.py \
    --D 2 --chi-list 8 16 --gs-steps 30 \
    --json examples/heisenberg_u1sz_spike.json
```
Expected: two rows, each with finite `E_sym`, `|dE_sym_vs_dense|` ≲ 1e-3 (looser at 30 steps from the same init; the tight 1e-8 check lives in Task 3), `E_sym` near −0.66, and `num_blocks_sym > 1`.

- [ ] **Step 2: Evaluate the GO/NO-GO gate**

Apply the spec's three checks to the JSON:
1. **Runs** — no `error` rows.
2. **Correct** — Task 3 passed (tight 2a) AND `E_sym ≈ E_dense` here AND `E_sym ≈ −0.66`.
3. **Perf signal** — compare `sym_warm_step_s` vs `dense_warm_step_s` and report `num_blocks_sym`. Note honestly if symmetric is slower at D=2 (block overhead can dominate at tiny D — a yellow flag, not an automatic NO-GO).

- [ ] **Step 3: Write the verdict**

Create `docs/superpowers/handoffs/2026-06-14-u1sz-heisenberg-spike.md` with: setup, the results table (both energies, the gap, timings, block count), whether the absorb-step gap was hit and (if so) how it was fixed, the explicit **GO / NO-GO** verdict against the three checks, and — **on GO** — a recommended (D,χ) characterization-sweep design (the apples-to-apples symmetric-vs-dense scaling study at larger D/χ, the next study). Reference `[[570-dense-largeD-study]]` framing: dense is runtime-bound; does block-sparsity move that wall.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/handoffs/2026-06-14-u1sz-heisenberg-spike.md examples/heisenberg_u1sz_spike.json
git commit -m "docs(#570): U(1)-Sz Heisenberg spike — GO/NO-GO verdict + data"
```

---

## Final verification

- [ ] Run core tests touching the new code:
  `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_u1sz.py -q` → all PASS.
- [ ] `JAX_PLATFORMS=cpu uv run pytest -m core -q` → no regressions from the new src/ functions.
- [ ] Then proceed to `superpowers:finishing-a-development-branch` (PR into main; merge queue runs CI).
