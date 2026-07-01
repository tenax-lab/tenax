# #632 shard-only reach benchmark — dense CTM-AD value_and_grad at D=10–12

**Date:** 2026-07-01
**Branch:** `feat/632-chunk-ctm-absorb-inc1`
**Harness:** `examples/bench_ctm_shard_reach_grad.py` (uses the rung-2 probe
`tests/_rung2_grad_probe.py::implicit_energy_and_grad`, recipe="2x2",
`forward_gauge="phase"`, `adjoint_method="fixed_point"`, well-conditioned state).
**Hardware:** 2× A100-SXM4-80GB (GPUs 1,2; ~64 GiB usable/device),
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, one D per process (clean per-device peak).
**Context:** ordered after the Increment-2 chunked-backward NO-GO
(`2026-07-01-chunk-ctm-absorb-increment2-backward-gate.md`) — the large-D backward
memory lever is GSPMD sharding (rung-2), not chunking. This measures how far that
lever reaches at D=10–12.

**Verdict: on 2 GPUs, GSPMD sharding gives NO reach extension at D=10–12.** Every
D≥10 config either already fits on 1 GPU or OOMs on both. Sharding relief is
χ-gated *and* D-fading; it confirms the SVD-replication wall
(`632-gspmd-svd-replication-rootcause`). A definitive 4-GPU test (rung-2's config,
where +2-in-D was claimed) needs 4 clean A100s — unavailable at run time.

## Harness validation (reproduces rung-2 exactly)

Single-GPU D=8 χ=24 = **10.66 GB** — identical to the rung-2 gate 2B number. The
probe is the rung-2 probe; the measurement path is trusted.

## Results

Per-device peak of the full `value_and_grad` (forward CTM + implicit-AD backward):

### χ = 24
| D | 1-GPU | 2-GPU sharded | relief |
|---|---|---|---|
| 6 | 2.16 GB | 1.15 GB | **1.88×** |
| 8 | 10.66 GB | 6.85 GB | **1.56×** |
| 10 | OOM (tried +29 GiB) | OOM (tried +31 GiB) | — |
| 12 | autotuner-fail (`f64[144,24,144,144,24]` transpose) | OOM (tried +45 GiB) | — |

### χ = 16
| D | 1-GPU | 2-GPU sharded | relief |
|---|---|---|---|
| 6 | 0.81 GB | 0.91 GB | **0.89×** (overhead > relief) |
| 8 | 5.10 GB | 4.84 GB | **1.05×** |
| 10 | 23.21 GB | 23.20 GB | **1.00×** (none) |
| 12 | OOM (tried +30.7 GiB) | OOM (tried **+77 GiB**) | — |

## Reads

1. **Relief is χ-gated.** At χ=24 sharding pays off (1.5–1.9× at D=6–8); at χ=16 it
   is ~1.0× or worse (D=6 is 0.89× — the all-gather overhead exceeds the ÷N relief).
   The sharded axis is the D² double-layer leg; at small χ the replicated part +
   comm buffers dominate, so ÷N buys nothing.
2. **Relief D-fades at fixed χ=24:** 1.88× (D6) → 1.56× (D8). The SVD-forced
   replicated χ²·D⁶ intermediate grows as D⁶ while the shardable contraction relief
   is fixed at ÷N, so the replicated fraction → 1 and relief → 1× as D grows.
3. **No reach on 2 GPUs at D=10–12.** There is no config where 2-GPU fits and 1-GPU
   OOMs: D=10 χ=16 fits on both; D=10 χ=24 and all D=12 OOM on both.
4. **D=12 sharding is counterproductive.** 2-GPU allocates *more* than 1-GPU
   (χ=16: +77 vs +30.7 GiB) — all-gather buffers for the replicated intermediate.

## Mechanism (confirms `632-gspmd-svd-replication-rootcause`)

The projector SVD is unshardable in XLA, forcing the dominant χ²·D⁶ backward
intermediate **replicated** across devices. Sharding partitions the absorption
*contraction* (÷N) but not the replicated SVD-bound part; as D grows (D⁶) or χ
shrinks, the replicated part + all-gather overhead dominate → relief collapses to
1× (and below). This is the *same* wall the chunked-backward NO-GO hit from the
other side: neither chunk nor shard addresses the SVD-replicated backward
intermediate — that is the structural D≥10 dense CTM-AD wall.

## Limitation / open

- **Only 2 clean A100s at run time** (GPU 0 held 61 GiB; GPU 4 at 96% util). rung-2
  reported 4-GPU extends the single ceiling D=8 → D=10 at χ=24 (+2 in D, 21.45
  GB/device). 2-GPU is insufficient to reach D=10 χ=24 (both OOM). A 4-GPU rerun
  (`CUDA_VISIBLE_DEVICES=0,1,2,4` or any 4 free A100s) would test whether the +2-in-D
  reproduces and whether it reaches into D=12. The relief-fade + D=12 counterproductive
  overhead predict **D=12 is out of reach even at 4-GPU**, but that is untested here.
- Practical takeaway (unchanged from `fermionic-large-d-tooling` / dense-pragmatic):
  the large-D **forward** lever is split-CTM (χ²·D⁴, #650); the **backward** remains
  SVD-replication-bound at D≥10. Multi-GPU dense CTM-AD is a modest "fit a bit
  bigger" lever (χ≥24, D≤~10, needs ≥4 GPUs), not a D=12 enabler.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=1  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_shard_reach_grad.py --D 8 --chi 24          # 1-GPU 10.66 GB
CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python examples/bench_ctm_shard_reach_grad.py --D 8 --chi 24 --shard  # 2-GPU 6.85 GB (1.56x)
```
