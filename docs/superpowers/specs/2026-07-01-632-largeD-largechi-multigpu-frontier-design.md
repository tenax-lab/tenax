# Large-D × large-χ multi-GPU frontier benchmark (phase 1) — Design

**Date:** 2026-07-01
**Issue:** #632 (multi-GPU dense CTM-AD reach)
**Status:** design approved, pending spec review → writing-plans

## Goal

Measure the reachable **(D, χ)** frontier of one iPEPS `value_and_grad` step
(energy + gradient) across the paths and GPU counts tenax has **today**, to
answer one question:

> Does split-CTM on **1 GPU** reach larger (D, χ) than dense-CTM on **2 GPUs**,
> and where is each wall?

This is a **characterization** study — **no `src/` change**. It establishes the
current ground truth (and confirms the path-composition gap, below) before any
decision to build a new capability in a later phase.

## Motivating crux (why this study exists)

tenax has two relevant CTM paths, and they **do not compose**:

- **split-CTM** (`_split_ctm_*`, `ctm_energy_split_implicit`): χ²·D⁴ absorption —
  the memory-efficient large-χ/large-D forward+AD lever (productized as
  `--path split` / `fuse_virtual_legs=False`, #650). It has **no `device_mesh`**
  anywhere → **single-GPU only**, single-site (`recipe="1x1"`).
- **dense CTM** (`ctm_energy_implicit`): χ²·D⁶ absorption. Has the multi-GPU
  machinery (GSPMD `device_mesh` sharding + the `ctm_chunk_size` knob from Inc1,
  #668), but the sharding relief is weak — SVD-replication-bound, ~1.2–2× and
  D-fading (the shard-reach benchmark, `2026-07-01-632-shard-only-reach-grad-D10-12`).

So "large D + large χ on multiple GPUs" today forces a choice of **one** lever,
not both: split's χ²·D⁴ savings on 1 GPU, *or* dense's weak multi-GPU sharding on
the χ²·D⁶ path. This benchmark quantifies that trade so phase 2 can decide
whether building split × multi-GPU is worth it.

## Scope

- **Operation:** `value_and_grad` — one optimize step (energy + gradient of the
  implicit-AD CTM energy). This is the "running a ground-state calculation"
  metric and where the backward wall lives.
- **Recipe:** fixed at **1×1** for *all* configs — split IS 1×1, and
  `ctm_chunk_size` (Inc1) is 1×1-only, so 1×1 is the common ground that makes the
  four configs apples-to-apples.
- **GPU count:** 1-GPU and 2-GPU, on the two clean cards (GPUs 1,2). 4-GPU is
  **deferred** to a follow-up (needs GPUs 0 and 4 freed).
- **State:** 1-site dense iPEPS, Heisenberg gate, well-conditioned tensor (same
  init as the trusted rung-2 / shard-reach probe), x64.

## Prerequisites / branch base

Phase 1 **must branch off fresh `origin/main` (f9b8f6e, #668 "Increment 1 —
chunked dense-CTM absorption")**, not the current `bench/632-shard-reach-D10-12`
branch, which predates #668. Two of the four configs depend on #668:

- **config 4 (dense + chunk):** `ctm_energy_implicit` gains the
  `ctm_chunk_size: int | None = None` param (`_ctm_energy_ad.py:367`), threaded
  into the forward as `partial(jit_step_raw, chunk_size=ctm_chunk_size)` (`:577`).
  Merged behavior is **forward-chunk + monolith-backward** (exact grads; Inc2
  backward-chunk was gated NO-GO and reverted).
- **config 3 (dense-1×1 + shard):** #668 hoisted `_shard_a` into the **1×1** sweep
  branch (previously 2×2-only), so `device_mesh` sharding only affects the 1×1
  forward *after* #668. Before it, a 1×1 `device_mesh` run was silently unsharded.

`ctm_energy_split_implicit` (config 1) and the base dense path (config 2) predate
#668 and are already present. The current branch's one uncommitted edit (the
shard-reach handoff's buffer-level OOM section) belongs to the shard-reach PR, not
phase 1 — handle it separately so it is not lost.

## Comparison matrix

All at recipe=1×1; **one config per process** (peak memory is cumulative
in-process, so isolation gives a clean per-device peak — the shard-reach method).

| # | config                    | path / entry                             | GPUs | knobs                          |
|---|---------------------------|------------------------------------------|------|--------------------------------|
| 1 | split-CTM, 1-GPU          | `ctm_energy_split_implicit` (χ²·D⁴)      | 1    | `chi_I = chi`                  |
| 2 | dense, 1-GPU              | `ctm_energy_implicit` recipe=1×1 (χ²·D⁶) | 1    | —                              |
| 3 | dense, 2-GPU shard        | dense + `device_mesh`                     | 2    | GSPMD (N=2)                    |
| 4 | dense, 2-GPU shard+chunk  | dense + `device_mesh` + `ctm_chunk_size`  | 2    | GSPMD (N=2) + chunk (Inc1)     |

**The gap this documents:** split has **no column 3/4** — it is 1-GPU only. The
study makes that concrete by showing where split-1GPU lands relative to
dense-2GPU on the same (D, χ).

**Metric per cell:** per-device peak (GB), outcome (`OK` / `OOM` / autotuner-fail),
wall time, and an energy / ‖g‖ sanity value. The frontier = the largest (D, χ)
that returns `OK` for each config.

**Grid:** D ∈ {6, 8, 10, 12}; a per-D χ list, **pushed higher on the split path**
(χ²·D⁴ ≪ χ²·D⁶, so split should reach much larger χ before OOM). Concrete χ values
are chosen at run time to bracket each config's wall (start where the prior
shard-reach numbers land and step χ up until OOM).

## Components

### Component 1 — `tests/_frontier_grad_probe.py` (new)

A single dispatcher that leaves the trusted `_rung2_grad_probe.py` **pristine**
(it is still used by the rung-2 gate tests). It reuses that probe's `_indices` /
`_init_data` helpers (imported) for an identical 1-site dense iPEPS + Heisenberg
gate + well-conditioned init.

```python
def frontier_energy_and_grad(
    *, path, D, chi, chi_I=None, device_mesh=None, ctm_chunk_size=None,
    seed=0, well_conditioned=True, max_iter=30,
):
    """Return (energy: float, grad: (D,D,D,D,2) array) of one value_and_grad step.

    path="dense": ctm_energy_implicit(..., recipe="1x1", device_mesh=device_mesh,
                  ctm_chunk_size=ctm_chunk_size,
                  forward_gauge="phase", adjoint_method="fixed_point")
    path="split": ctm_energy_split_implicit(..., chi_I=chi_I or chi)   # no mesh (1-GPU)
    """
    ...
    e, g = jax.value_and_grad(loss)(data0)
    return float(e), np.asarray(g)
```

- **dense** branch mirrors the rung-2 probe but at `recipe="1x1"` and additionally
  threads `ctm_chunk_size`.
- **split** branch calls `ctm_energy_split_implicit({(0,0): A}, SINGLE_SITE_NEIGHBORS,
  gate, chi=chi, chi_I=chi_I or chi, max_iter=..., conv_tol=..., renormalize=True)`.
  It **does not** accept `device_mesh` (single-GPU); passing one is a caller error
  the harness guards against.

Peak memory is ~independent of `max_iter` (implicit-AD stores the fixed point, not
the unrolled trajectory), so `max_iter=30` is enough for a faithful peak.

### Component 2 — `examples/bench_ctm_frontier_grad.py` (new)

CLI harness in the shard-reach style. Args:
`--path {dense,split}  --D <ints...>  --chi <int>  --chi-I <int|None>
 --shard  --chunk <int>  --max-iter <int>  --seed <int>`.

Builds the mesh via `build_ctm_mesh()` when `--shard`. Runs **one (path, D, χ) per
process** for a clean peak. Reports one line per D:

```
path=split D=10 chi=48 OK  E=-0.601234  |g|=1.2e-02  per_device_peak=NN.NN GB  wall=NN.Ns
path=dense D=10 chi=24 FAILED(XlaRuntimeError: RESOURCE_EXHAUSTED ... 30.5GiB)  wall=NN.Ns
```

`peak_gb()` reads `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` (the
per-device peak; on a 2-GPU sharded run every device holds the same replicated +
sharded mix, so device 0's peak is representative — matches shard-reach).

## Guards (two orthogonal walls: divisibility vs memory)

There are **two independent** reasons a sharded config can fail to run. The
harness distinguishes them explicitly.

1. **Divisibility (the shard guard).** GSPMD shards the **D² leg** of the
   double-layer tensor and the CTM edges (`ctm_sharding.py`:
   `edge_partition_spec()` → `PartitionSpec(None, "d", None)`;
   `double_layer_partition_spec()` → shard a D² axis). So a config is *shardable
   at all* iff **`D² % N == 0`**. The harness **skips** (prints `SKIP`) a `--shard`
   config when `D² % mesh_n != 0`, before ever building the computation.

   At **N=2** all four grid D values pass (D² ∈ {36, 64, 100, 144} are all even):
   - D=10 → 100 → 100 % 2 = 0 ✓
   - D=12 → 144 → 144 % 2 = 0 ✓

   D²=100 and 144 are **not** impossible to shard — they are fine at N=2. They
   would only be blocked at an N that does not divide them (e.g., N=3 divides 144
   but not 100), or for odd D² such as D=11 → 121 (un-shardable by any even N).
   For the deferred 4-GPU follow-up, **N=4** also passes all four D (36, 64, 100,
   144 are all divisible by 4); it is N=3 that would break D²=64 and 100.

2. **Memory (OOM).** Passing the divisibility guard does **not** mean the config
   fits. A shardable config can still exceed device RAM — e.g., shard-reach's
   D=10/12 were *attempted* (they passed the guard) and then hit the memory wall
   at those χ. This is caught at run time as `RESOURCE_EXHAUSTED` and reported as
   `FAILED(...)`, distinct from `SKIP`.

The `--chunk` guard: `ctm_chunk_size` is dense-1×1 only, so `--chunk` on
`--path split` is a caller error → the harness warns and ignores it. `--shard` on
`--path split` likewise warns and is ignored (split has no `device_mesh`).

## Data flow / the sweep

A documented bash matrix (in the findings doc's Reproduce block) drives the four
configs across the grid, one process each, appending stdout to a scratch file;
the frontier table is assembled from those lines. Per-process isolation is the
trusted method (in-process peak is cumulative and would contaminate a multi-config
loop). Example rows:

```bash
# split, 1-GPU, push chi high:
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_frontier_grad.py --path split --D 10 --chi 48
# dense, 2-GPU shard+chunk:
CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_frontier_grad.py --path dense --D 10 --chi 24 --shard --chunk 8
```

## Error handling

`value_and_grad` is wrapped in try/except. Catch `RESOURCE_EXHAUSTED` (OOM) and
XLA autotuner failures; print `FAILED(<ExcType>: <message[:110]>)` plus the
attempted-allocation size when the BFC dump exposes it. A `SKIP` (divisibility) is
printed *without* running. Everything else re-raises (a real bug should not be
silently swallowed as `FAILED`).

## Validation

Before the GPU sweep, a **CPU sanity check** at D=2 χ=6 for both paths:

- assert finite `E` and finite `‖g‖` for `path="dense"` and `path="split"`;
- assert that at fixed D the split and dense energies **move toward each other as
  χ grows** (they truncate differently, so they are *not* equal at small χ — that
  is expected physics, not a bug; the check is monotone-approach, not equality).

No `src/` is touched, so the default dense path stays byte-identical and existing
CI (`-m core`) continues to cover it.

## Deliverable

`docs/superpowers/handoffs/2026-07-01-632-largeD-largechi-multigpu-frontier-findings.md`:

- the 4-config × grid frontier table (per-device peak + OK/OOM/SKIP per cell);
- the reachable frontier per config (largest (D, χ) that fits);
- the headline verdict: **split-1GPU vs dense-2GPU** — which reaches larger (D, χ);
- the confirmed composition gap (split has no multi-GPU column);
- a recommendation feeding a possible **phase 2** (build split × multi-GPU, or not).

## Out of scope (phase 1)

- **4-GPU** runs (deferred until GPUs 0 and 4 are free).
- Any **`src/` change** (this is a measurement study).
- The **2×2** recipe (chunk is 1×1-only; split is 1×1-only).
- **Symmetric / fermionic** envs (large-D symmetric is YASTN territory).
- Reducing **`chi_I` below χ** on the split path (held `chi_I = chi` for a clean
  frontier; `chi_I < χ` is a further split-only memory lever, measured later).
- **Forward-only** reach (this study is `value_and_grad`; forward-only is a cheaper
  secondary map that can be added later).

## Reproduce (validated methodology anchor)

The dense-1×1 probe should reproduce the shard-reach anchor within recipe
differences; the harness prints the same `per_device_peak` field the shard-reach
benchmark used, so numbers are directly comparable across the two studies.
