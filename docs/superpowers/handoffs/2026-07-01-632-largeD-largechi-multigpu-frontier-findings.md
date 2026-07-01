# Large-D × large-χ multi-GPU frontier — phase 1 findings

**Date:** 2026-07-01
**Branch:** `bench/632-frontier-multigpu` (off #668)
**Harness:** `examples/bench_ctm_frontier_grad.py` + `tests/_frontier_grad_probe.py`
**Hardware:** 2× A100-80GB (GPUs 1,2), f64, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, one config/process, recipe=1×1, `value_and_grad` (energy + gradient), well-conditioned 1-site state, max_iter=30.
**Spec:** `docs/superpowers/specs/2026-07-01-632-largeD-largechi-multigpu-frontier-design.md`

**Verdict: split-CTM on 1 GPU reaches far larger (D, χ) than dense-CTM on 2 GPUs — the memory-efficient *algorithm* (χ²·D⁴, single-GPU) dominates the multi-GPU *dense* apparatus (χ²·D⁶ + shard + chunk). Multi-GPU dense is NOT the large-(D,χ) lever; split-CTM is.**

## Harness anchor

dense-1×1 D=8 χ=24 = **11.01 GB** (shard-reach recipe=2×2 was 10.66 GB — recipe differs; same order, path trusted). Split and dense energies/‖g‖ agree to ~4 digits at every D (D6 both E=+0.000044; D8 both −0.000405; D10 both −0.000226) — cross-path validation of the probe. (Energies are tiny because the well-conditioned init is a near-product state; this is a memory/reach study, not a ground-state energy.)

## Frontier — per-device peak (GB), recipe=1×1, value_and_grad

### Config 1 — split-CTM, 1-GPU (χ²·D⁴)
| D | χ=24 | χ=48 | χ=96 | χ=128 | reach (max χ OK) |
|---|------|------|------|-------|------------------|
| 6 | 0.28 | 0.98 | 3.78 | 6.42 | **≥128** (grid cap, not a wall) |
| 8 | 0.71 | 2.58 | 9.68 | 16.93 | **≥128** (grid cap) |
| 10 | 1.60 | 5.71 | 22.43 | 39.89 | **≥128** (grid cap) |
| 12 | 3.20 | 11.59 | 46.35 | **autotuner-fail** | **96** (χ128 = compile wall, not OOM) |

### Configs 2–4 — dense (χ²·D⁶): 1-GPU / 2-GPU shard / 2-GPU shard+chunk8
| D | χ | dense 1-GPU | 2-GPU shard | 2-GPU shard+chunk8 |
|---|---|-------------|-------------|--------------------|
| 6 | 16 | 1.20 | 1.07 | 1.07 |
| 6 | 24 | 2.69 | 1.27 | 1.27 |
| 6 | 32 | 4.13 | 2.91 | 2.91 |
| 8 | 16 | 5.10 | 3.95 | 3.95 |
| 8 | 24 | 11.01 | 9.78 | 9.78 |
| 8 | 32 | 18.37 | 16.50 | 16.50 |
| 10 | 16 | 28.01 | 19.64 | 19.64 |
| 10 | 24 | **OOM** (29.0 GiB) | **OOM** (34.5 GiB) | **OOM** (34.5 GiB) |
| 12 | 16 | **autotuner-fail** | **autotuner-fail** | **autotuner-fail** |

Dense reach (max χ OK): D6 ≥32, D8 ≥32, D10 =16, D12 = none (χ16 fails XLA autotuning on the `f64[144,12,…]` transpose — the giant-gemm compile wall, not memory), for **all three** dense configs.

## Reads

1. **Split-1GPU vs dense-2GPU — split wins by a wide margin.**
   - **D=10:** split reaches **χ≥128 (39.89 GB) on 1 GPU**; dense-2GPU caps at **χ16 (19.64 GB)**, OOMs at χ24. Split does ~8× the χ at 1 GPU vs dense at 2 GPUs.
   - **D=8:** split χ128 = 16.93 GB (1 GPU) ≈ dense-2GPU χ32 = 16.50 GB — same peak, 4× the χ.
   - **D=12:** split runs to **χ96 (46 GB, 1 GPU)**; dense (1- or 2-GPU) **cannot compile even χ16**. Split is the *only* path that runs D=12.

2. **Shard relief is weak and χ-gated: ~1.1–1.4×, no D-reach extension.**
   - D8 χ24: 11.01→9.78 = 1.13×; D8 χ32: 18.37→16.50 = 1.11×; D10 χ16: 28.01→19.64 = **1.43×** (best); D6 χ24: 2.69→1.27 = 2.1× (small D, small absolute).
   - Critically, 2-GPU does **not** extend the reach: both 1- and 2-GPU cap at **D10 χ16** (χ24 OOMs on both). Confirms `632-shard-only-reach-grad-D10-12` (SVD-replication-bound; the χ²·D⁶ backward intermediate stays replicated per device).

3. **Chunk adds nothing to the `value_and_grad` peak.** Config 4 (shard+chunk8) peaks are **identical** to config 3 (shard-only) at every cell — chunk gives 0 GB relief in the pipeline. Confirms the Increment-2 NO-GO from the pipeline side: the forward absorb chunk is below the CTM-convergence memory waterline and the backward is (correctly) not chunked. (Config 4's shorter wall times are just JAX persistent-compile-cache warmth from config 3's identical shapes, not a memory effect.)

4. **The composition gap, made concrete.** split-1GPU (χ²·D⁴) reaches D=12 χ96 and D≤10 χ≥128 on **one** GPU; the entire dense multi-GPU apparatus (shard + chunk on **two** GPUs) caps at D10 χ16 / D8 χ32. The memory-efficient algorithm on 1 GPU dominates the multi-GPU dense path everywhere on the frontier.

5. **Two walls both observed and distinct.** *Memory OOM* (dense D10 χ24 on 1- and 2-GPU). *Autotuner compile-fail* on the giant transpose (`f64[144,12,…]` for dense D12; `f64[12,128,128,12,12,12,2]` for split D12 χ128) — a compile wall independent of per-device memory. Split D12 χ128 fails on the autotuner, not OOM, so multi-GPU sharding would not rescue that specific cell.

## Recommendation (feeds phase 2)

**Building split × multi-GPU (thread `device_mesh` into the split path) is the only lever that could push past split's *memory* wall — but multi-GPU dense is a dead end.** The data is unambiguous: multi-GPU on the *dense* path buys ~1.1–1.4× and never extends the D reach, so it is not worth further investment as a large-(D,χ) enabler. The open large-(D,χ) question is whether sharding the *split* χ²·D⁴ path across GPUs extends its already-large reach:

- Split-1GPU already reaches **D=12 χ96 = 46 GB** and **D≤10 χ≥128** (grid cap, not measured to OOM) on one 80 GB card. Its memory headroom is large, so its reach is even higher than measured for D≤10.
- Sharding split across 2 GPUs would help where split hits a **memory** wall — i.e., very large χ at D≤10 (χ ≫ 128) or reaching **D=14**. It would *not* help split's D=12 χ128 cell, which is an **autotuner compile** wall, not memory.
- **Suggested phase 2:** (a) first push split-1GPU χ to its true memory wall at D=8/10 (grid capped at 128 here) to quantify the single-GPU split ceiling; (b) then gate whether `device_mesh` on the split path shards the χ²·D⁴ intermediate for a real per-device ÷N — the same GSPMD question, but on a path where the dominant intermediate may shard better than dense's SVD-replicated one. If split's dominant backward intermediate is *also* SVD-replication-bound, multi-GPU split will fade like dense and phase 2 should instead target `chi_I < χ` (the split-only interlayer-bond memory lever, held = χ here) and the D=12/14 autotuner compile wall.

## Reproduce

```bash
cd /home/yjkao/tenax-632-frontier   # worktree, cuda13 env
# anchor:
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_frontier_grad.py --path dense --D 8 --chi 24
# split reach (1-GPU), dense 1-GPU, dense 2-GPU shard, dense 2-GPU shard+chunk:
CUDA_VISIBLE_DEVICES=1   ... --path split --D 10 --chi 128
CUDA_VISIBLE_DEVICES=1   ... --path dense --D 10 --chi 16
CUDA_VISIBLE_DEVICES=1,2 ... --path dense --D 10 --chi 16 --shard
CUDA_VISIBLE_DEVICES=1,2 ... --path dense --D 10 --chi 16 --shard --chunk 8
```
Full sweep driver: `scratchpad/run_frontier_sweep.sh` (anchor-gated, per-D early-exit on OOM, 900s/run timeout).

## Limitations / open

- Split reach for D≤10 is a **grid cap (χ=128), not a measured wall** — split's true single-GPU χ ceiling at D=8/10 is higher and unmeasured (phase-2 item a).
- **2 GPUs only** (GPUs 1,2 clean; 0 held 61 GiB, 4 busy). A 4-GPU dense run would raise shard relief modestly (N=4 shards all of D²∈{36,64,100,144}) but the ~1.1–1.4× / no-D-reach pattern predicts it stays a "fit a bit bigger" lever, not a D-enabler.
- Well-conditioned near-product state (clean adjoint); a random/critical state would have a harder backward but similar peak scaling (D⁶ dense / D⁴ split).
