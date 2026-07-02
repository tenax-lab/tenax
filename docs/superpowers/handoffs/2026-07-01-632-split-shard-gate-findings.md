# Split-CTM × multi-GPU shard — feasibility gate findings (phase 2)

**Date:** 2026-07-01
**Branch:** `docs/632-split-shard-gate` (off main)
**Harness:** `examples/spike_split_shard_ctm_gate.py` (throwaway monkeypatch spike — NO `src/` change; shards the split χ²·D⁴ grown corner via `with_sharding_constraint`, reuses the frontier probe's state).
**Hardware:** 2× A100-80GB (GPUs 1,2), f64, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, one config/process.
**Parent:** phase-1 frontier study (`2026-07-01-632-largeD-largechi-multigpu-frontier-findings.md`, PR #672), which recommended gating whether sharding the *split* path escapes the SVD-replication wall that killed multi-GPU *dense*.

**Verdict: GATE = NO-GO.** Sharding the split-CTM `value_and_grad` gives only ~1.1–1.2× per-device relief (flat in χ) and is *slower* in wall time. HLO confirms the projector SVD forces the sharded intermediate fully replicated — the **same** SVD-replication wall as dense. The wall is path-agnostic: XLA lowers the CTM projector SVD as unshardable, so it all-gathers whatever GSPMD sharded, whether the dominant intermediate is dense χ²·D⁶ or split χ²·D⁴.

## What was tested

Shard the split path's dominant χ²·D⁴ intermediate — `Cg = (χ, D, D, χ)` from `_doublelayer_grown_corner` (`_split_ctm_tensor_moves.py:63`) — by a `with_sharding_constraint` on its `u_ket` (dim-D) leg on a 2-GPU mesh (axis `"d"`), plus `device_put` of the input site tensor `A` sharded. The split forward is a `custom_vjp` implicit-AD fixed point whose backward reuses the same move functions via `jax.vjp`, so the constraint propagates to the backward automatically (verified). Measured `jax.value_and_grad` per-device peak, 1-GPU vs 2-GPU, at D=10.

## Correctness (this is NOT a false NO-GO)

- **Grad parity:** max\|grad_shard − grad_noshard\| / \|grad_noshard\| = **8.0e-19** (≤ 1e-10). The sharded computation is numerically identical.
- **Positive control:** the constraint *genuinely* shards `Cg` — all captured evidence entries report `is_fully_replicated=False` across 2 real devices. So a flat peak reflects SVD replication, not a constraint that silently did nothing.

## Result (D=10, per-device peak GB, eager production pipeline)

| χ | 1-GPU | 2-GPU sharded | relief | 2-GPU wall |
|---|-------|---------------|--------|------------|
| 48 | 5.42 | 4.88 | 1.11× | 2–4× slower |
| 64 | 10.07 | 8.44 | 1.19× | 2–4× slower |
| 96 | 21.0 | 18.7 | 1.12× | 2–4× slower |

Relief is ~1.1–1.2×, **flat in χ** (not χ-gated), and 2-GPU is 2–4× worse in wall time (collective overhead).

## Mechanism (HLO smoking gun)

One compiled sweep step, D=10 χ=48, 2 devices: **12 all-gather + 12 all-reduce** collectives. **All 8 projector SVD custom-calls** (4 moves × [corner χ×χ SVD + edge-split (χD)×(χD) SVD]) receive a **fully-replicated** operand:
- one `all-gather` explicitly reconstructs the exact `(χ, D, D, χ)` `Cg` tensor we sharded, right before the big contraction;
- the edge-split SVDs are fed by an `all-reduce` (partial-sum combine) or `all-gather`, always landing on a full **480×480** (χD = 48·10) replicated matrix.

`_compute_projector_tensor` builds `M = C1g^T @ C4g` (χD² × χD²) and runs `truncated_svd_ad(M)` on the AD path; XLA requires an un-sharded SVD operand, so the χD² fused leg is gathered — replicating the χ²·D⁴-scale operand. Identical to the dense root cause (`632-gspmd-svd-replication-rootcause`); the split path merely has a smaller (χ²·D⁴ vs χ²·D⁶) — but equally replicated — SVD operand.

## Consequences for #632

- **Multi-GPU is dead for BOTH dense and split large-D CTM AD.** The SVD-replication wall is not a dense-path artifact; it is intrinsic to differentiating CTM through an XLA-lowered SVD. Consistent with no external library offering multi-GPU CTM AD (variPEPS single-device; YASTN lists distributed as a *future* need; only Ace-TN does multi-GPU CTMRG, and it uses imaginary-time evolution, **not** AD — see `yastn-varipeps-sym-vs-dense-benchmark`).
- **The large-(D,χ) levers are all single-GPU / algorithmic** (phase-1's fallback, now the main line):
  1. push split-1GPU χ to its true memory wall (phase-1 grid capped at 128; real ceiling higher — measured in the addendum below / a follow-up);
  2. `chi_I < χ` — the split-only interlayer-bond memory lever (held = χ throughout phases 1–2);
  3. the D≥12 autotuner **compile** wall (the giant transpose that blocks split χ128 / dense χ16) — a compile-model limit, not memory.

## Addendum: split-1GPU χ-ceiling (2026-07-01) — the real wall is COMPILE, not memory

Follow-up to phase-1 lever #1 ("push split-1GPU χ to its true memory wall"). Pushed χ on split 1-GPU (GPU 1, `bench_ctm_frontier_grad.py --path split`) until failure:

| D | reach | per-device peak at reach | first failure | failure type |
|---|-------|--------------------------|---------------|--------------|
| 8 | **χ=224** | 51.2 GB (χ128 16.7 / 160 26.3 / 192 37.7 / 224 51.2) | χ256 | **autotuner compile-fail** `f64[2048,2048]` (χD=2048) |
| 10 | **χ=128** | 37.3 GB | χ160 | **autotuner compile-fail** `f64[1600,1600]` (χD=1600) |

**The single-GPU ceiling is an XLA autotuner _compile_ wall on the (χD)×(χD) projector gemm/SVD — NOT memory.** 30+ GB of card headroom remains at the failure point (D8 χ256 ≈ 65 GB, D10 χ160 ≈ 58 GB both fit an 80 GB A100). The wall lands at **χD ≈ 1700** (D8 χ224→1792 OK, χ256→2048 fail; D10 χ128→1280 OK, χ160→1600 fail) — the same autotuner failure that blocked dense D12 χ16 and split D12 χ128 in phase 1.

**Consequence:** at default settings split-1GPU is **compile-bound, not memory-bound** — the (χD)×(χD) f64 gemm autotuner fails while 30+ GB of card sits idle. The lever is therefore the autotuner wall, not a memory trick. That wall turns out to be **soft** (next section).

## Autotuner-wall probe (2026-07-02) — the wall is SOFT (`--xla_gpu_autotune_level=0` bypasses it)

Retried the failing configs (split D=10 χ160, D=8 χ256) under XLA flags:

| flag | D=10 χ160 | D=8 χ256 |
|------|-----------|----------|
| baseline | FAIL (Triton `gemm_fusion_dot` autotune) | FAIL (Triton autotune) |
| `--xla_gpu_enable_triton_gemm=false` | FAIL (**cuBLAS `custom-call` autotune**) | FAIL (cuBLAS autotune) |
| `--xla_gpu_autotune_level=0` | **OK, 57.82 GB** | OOM (512 MiB short) |

- **Forcing cuBLAS does NOT help** — the failure just moves to the cuBLAS custom-call autotuning. It is the *autotuning* that fails at this f64 size, not Triton specifically.
- **`--xla_gpu_autotune_level=0` (disable autotuning) bypasses the wall.** D=10 χ160 (57.82 GB) — which **fits** an 80 GB card — was being blocked purely by the autotuner; disabling it recovers the config. Reach D=10 **χ128 → χ160**; the next step (χ192) then OOMs (memory).

**Net:** the autotuner compile wall is soft. `--xla_gpu_autotune_level=0` recovers configs that fit in memory but were autotuner-blocked (a modest reach gain) and **converts the wall from a hard compile failure to a graceful memory limit** (~64 GB usable). **Practical recommendation: run large-χ split with `XLA_FLAGS=--xla_gpu_autotune_level=0`.**

## D=12 recovery probe (2026-07-02) — NOT recovered (memory-gated)

`--xla_gpu_autotune_level=0` does **not** re-open D=12:
- **dense D12 χ16:** compiles past the autotuner but OOMs on a single **61 GiB** buffer — memory-infeasible on one 80 GB card (D⁶ scaling).
- **split D12 χ96** (fit at 46 GB with *default* autotuning): **OOMs under autotune0** (2.85 GiB short) — untuned codegen has a higher memory footprint, so at D=12's tight budget autotune0 *hurts*.

So D=12 is **memory-gated**, and `autotune_level=0` is a compile/memory-**boundary** lever (helps at D=8/D=10 where memory is slack), **not** a D=12 enabler. The only remaining D=12 lever is `chi_I < χ` (split-only, untested); multi-GPU stays NO-GO.

## Reproduce

```bash
cd <worktree with cuda13 env + the frontier harness>
# grad parity (CPU): shard vs no-shard
JAX_PLATFORMS=cpu uv run python examples/spike_split_shard_ctm_gate.py --verify --D 2 --chi 6
# 1-GPU vs 2-GPU per-device peak at D=10:
CUDA_VISIBLE_DEVICES=1   XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/spike_split_shard_ctm_gate.py --peak --D 10 --chi 48
CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/spike_split_shard_ctm_gate.py --peak --D 10 --chi 48 --shard
# HLO all-gather / SVD-operand replication check:
CUDA_VISIBLE_DEVICES=1,2 uv run python examples/spike_split_shard_ctm_gate.py --hlo --D 10 --chi 48 --shard
```

_The `examples/spike_split_shard_ctm_gate.py` here is the throwaway monkeypatch gate (imports split internals, monkeypatches `_doublelayer_grown_corner`); it is a reproducibility artifact, not production code._
