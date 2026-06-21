# #632 rung-1 — Multi-GPU GSPMD-sharded dense forward CTM at large D — Findings

**Date:** 2026-06-21
**Parent:** #566 (large-D iPEPS is a tenax/JAX wall). **Implements:** the rung-1 plan's
**Task 7** (GPU memory-feasibility benchmark, never run on real hardware until now).
**Spec:** `docs/superpowers/specs/2026-06-19-gspmd-sharded-dense-ctm-large-d-design.md`.
**Hardware:** single node, 4× A100-SXM4-80GB (NVLink) + 1 DGX-Display GPU (excluded).
**Verdict:** **GO on correctness, WEAK as a large-D lever (measured).** GSPMD sharding of
the dense forward CTM is numerically correct and does fit a D that OOMs on one GPU onto
four — but the per-device memory reduction is only **~1.6×** (not ~4×), and dense-CTM
memory scales as **~D⁶**, so N GPUs buy only **N^(1/6)** in D. Net: 4 GPUs lift the forward
ceiling from **D=10 → D=12** (χ=24), i.e. **+2 in D**. Per-step throughput is unchanged.
This is not a route to the genuinely large-D regime, and it does not change the
[fermionic-large-D] strategic picture (eager/YASTN for large D).

## Question

Does single-node multi-GPU (GSPMD D²-axis sharding, #632 rung-1) "really scale" the dense
iPEPS forward CTM to large D? Concretely: (1) does a D that OOMs on one A100-80GB fit on
2–4? (2) by how much does per-device memory drop? (3) is the sharded result correct on real
GPUs? (4) what is the throughput cost?

## Setup

Dense forward CTM + energy, **f64** (tenax enables x64 on import — all numbers are
double-precision), 1-site Heisenberg probe (`tests/_ctm_sharding_probe.py`), GSPMD per-move
D²-axis resharding (corners + projector SVD replicated). Benchmark
`examples/bench_ctm_sharding_memory.py`; I added a `--max-iter` passthrough + wall-time
(peak memory is reached in CTM sweep 1, so `max_iter=2` measures the true peak cheaply).
`peak_bytes_in_use` from `jax.Device.memory_stats()`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`
so OOM and peak are real. **Device note:** CUDA enumerates the DGX-Display as device 4 (not
nvidia-smi's index 3); use `CUDA_VISIBLE_DEVICES=0,1,2,3` for the four A100s — `0,1,2,4`
wrongly includes the 4 GB display and the executable fails to launch.

## Answer

### 1. Memory: the OOM crossover is real (the headline) — but the reduction is only ~1.6×

Cleanest single-config measurements (each D in its own process; 4-GPU peak is reproducible
to the byte across runs, single-GPU peak has ~25% allocator/autotune variance):

| D (χ=24, f64) | 1 GPU peak | 4-GPU per-device | reduction | outcome |
|---|---:|---:|---|---|
| 8  | 6.7 GB | 2.2 GB | — (small-D, replicated floor dominates) | both fit |
| 10 | **20.3 GB** | **12.8 GB** (stable ×3) | **1.6×** | both fit |
| 12 | **OOM** (>80 GB; +45 GiB alloc failed) | **55 GB → FITS** | — | ✅ **crossover** |
| 14 | autotuner failure (see §5) | ~143 GB (est.) → OOM | — | fails on both |

- **Single-GPU forward ceiling (χ=24) = D=10.** D=12 OOMs.
- **4-GPU forward ceiling (χ=24) = D=12.** Net reach gain **+2 in D**.
- The per-device reduction is **~1.6×, far below the ideal 4×.** Peak memory is pinned by
  **partially-replicated intermediates** — the projector-SVD all-gather is replicated *by
  design* (spec §"sharding scheme"), plus per-move `a` reshard buffers and unsharded
  reductions — not by the cleanly-sharded χ²D⁶ term. So 4 GPUs do **not** give a 4×
  memory headroom.

> ⚠️ **Correction to the in-session interim table.** An earlier sweep reported reductions
> "2.1× → 2.8× → 3.1× rising with D" from a *multi-D loop* (D=4,6,8 in one process). That was
> **single-GPU allocator high-water inflation** (the loop's peak carries fragmentation from
> earlier D's), which oversold the win. The controlled single-config number is **1.6×** and is
> the one to trust.

### 2. Throughput: unchanged (not slower, not faster)

Compile-subtracted per-CTM-iteration at D=10 χ=24 (two `max_iter` points, difference out the
one-time compile):

| | wall @ mi=2 | wall @ mi=5 | per-iter | 
|---|---:|---:|---:|
| 1 GPU | 16.1 s | 39.6 s | **7.83 s/iter** |
| 4 GPU | 17.1 s | 40.2 s | **7.70 s/iter** |

Per-step compute is **essentially equal** (4-GPU 1.7% faster = noise). The spec feared the
replicated-SVD all-gathers would make it slower; they don't, but the sharded D⁶ compute
doesn't make it faster either. **Multi-GPU here is a pure memory play.** (An in-session
"16× faster" observation was a single cold-compile outlier — a fresh-process first compile of
D=10 took 116 s once but 16 s on repeat; debunked by the compile-subtracted numbers above.)

### 3. Correctness: SETTLED — sharded == single to machine precision on well-conditioned states

The benchmark's *random* probe states showed sharded-vs-single energy gaps of ~1e-4, larger
than CI's <1e-8 parity claim. I traced this to **floating-point reassociation of the sharded
contraction axis, NOT a logic bug.** Two decisive diagnostics (fake CPU devices so no GPU FP
confound):

| state (D=4 χ=8) | 2 devices | 4 devices |
|---|---:|---:|
| random (ill-conditioned CTM fixed point) | Δ = 2.7e-5 | Δ = 5.6e-5 |
| well-conditioned (near-product) | Δ = **6.9e-18** | Δ = **4.9e-17** |

The discrepancy **scales with device count** (more shards → more partial-sum regrouping →
different rounding) and **collapses to machine epsilon when the fixed point is
well-conditioned.** That is the exact signature of non-associative floating-point summation,
amplified by the κ of a *random* iPEPS CTM fixed point. Real optimization targets
well-conditioned (physical) states, where the sharded CTM equals single-device to ~1e-17.

**Two regimes of bit-exactness.** (i) At D=2, D²=4 → 1 element/device, so the contracted sum
is never regrouped → bit-exact for *any* state (CPU 2-dev and 4-dev both Δ~1e-17).
(ii) At D=4/χ=4 with N=4 (4 elements/device), the sum *is* regrouped, yet the random probe is
still Δ~1.4e-17 — because the small-χ fixed point happens to be well-conditioned. Push the same
random state to χ=8 and it diverges to ~1e-4. So bit-exactness at D≥4 is **conditioning-, not
structure-, dependent**: the reassociation always happens; whether it shows depends on κ.

### 4. Why it does NOT reach truly large D

1. **D⁶ memory scaling ⇒ N^(1/6) gain in D.** 4 GPUs = 4^(1/6) ≈ 1.26× → +2 in D. Reaching
   D≈16 would need ~64 GPUs; D≈20, ~4000. Structural, not tunable.
2. **Sharding efficiency ~1.6×, not 4×** (replicated SVD + reshard buffers, §1). The realized
   headroom is even smaller than the device count suggests.

### 5. Scope caveats

- **Forward CTM only.** Full `optimize_gs_ad` at large D still needs **rung-2 AD-backward
  sharding (UNBUILT)**. Backward memory is ~2–4× forward, so the *optimization* ceiling is
  **below D=12** until rung 2 lands.
- **The spec's premise was wrong for forward-only f64.** "D=6–8 χ=24 OOMs on 1 GPU" is false —
  D=8 is 6.65 GB and fits one A100 comfortably. Multi-GPU only matters at **D≥11**.
- **D=14 hits a separate XLA failure on BOTH single and multi-GPU:** `INTERNAL: Failed to get
  configs for: 6 out of 30 instructions` (cuDNN/cuBLAS autotuner cannot allocate workspace at
  that size). This is an XLA-autotuner issue distinct from the sharding and would gate D≥14
  regardless of device count.

## Recommendation

**Land rung-1 as correct, but record it as a weak large-D lever; do not over-promise.** The
mechanism works and is numerically sound, and the D=12 crossover is genuine — but +2 in D per
4× GPUs against a D⁶ wall means multi-GPU dense in tenax/JAX is **not** the path to the
large-D regime where symmetric/eager frameworks matter. For genuinely large D, the
[fermionic-large-D] verdict stands (YASTN/peps-torch eager).

**If rung-2 is pursued anyway** (to make the win usable for actual optimization, not just
forward energy), weigh it against this ceiling first: even a perfect rung-2 inherits the
~1.6×-per-4-GPU / N^(1/6) economics, so the optimization ceiling lands around D=11–12 on 4
GPUs — a marginal gain over the D=10 single-GPU forward ceiling.

**CI parity gap (closed in this PR).** `tests/_ctm_sharding_parity.py` covers D=2 (N=2) and
D=4/χ=4 (N=4). Both pass bit-exact — but only D=4/χ=4 even reassociates, and it stays bit-exact
*because its small-χ random fixed point happens to be well-conditioned*, not by construction. No
case exercised a larger-χ regime where a random state diverges (~1e-4) but a physical one stays
tight. This PR adds `test_sharded_well_conditioned_tight_parity` (D=4/χ=8, N=4, near-product
state, asserts <1e-10) — verified meaningful: the *random* state at the same (D,χ) fails 1e-10,
the well-conditioned one passes. This guards the property that matters: sharding is exact up to
FP reassociation, which stays at machine precision on physical (well-conditioned) states.

## Artifacts (this PR, off `main`)

- `docs/.../2026-06-21-632-multigpu-dense-ctm-findings.md` — this doc.
- `tests/test_ctm_sharding_parity.py` — new well-conditioned tight-parity case (the CI gap fix).
- `tests/_ctm_sharding_probe.py` — added `max_iter` kwarg (default 20, backward-compatible) and a
  `well_conditioned` near-product option (used by the new parity case + the bench).
- `tests/_ctm_sharding_parity_subproc.py` — accepts `well_conditioned` + threshold args.
- `examples/bench_ctm_sharding_memory.py` — added `--max-iter`, wall-time, x64-in-header; removed
  a misleading `reset_peak()` no-op and documented the cumulative-peak / one-D-per-process caveat.

## Risks / caveats recorded

- **`peak_bytes_in_use` is a cumulative high-water mark** — JAX has no reset API, so multi-D
  sweeps report the running max (inflating single-GPU peaks with earlier-D fragmentation; the
  bench's old `reset_peak()` was a silent no-op, now removed). **Run one D per process** for
  trustworthy per-config memory (as the headline numbers above do).
- **`CUDA_VISIBLE_DEVICES` ordering ≠ nvidia-smi ordering** — the DGX-Display is CUDA index 4;
  use `0,1,2,3` for the four A100s.
- The D=12 4-GPU peak (55 GB) is close to the 80 GB ceiling, so D=12 is near the 4-GPU forward
  limit at χ=24 — there is little headroom for rung-2 backward at that D.
