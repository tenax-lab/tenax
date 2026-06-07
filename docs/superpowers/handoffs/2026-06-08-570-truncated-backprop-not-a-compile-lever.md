# Truncated backprop is accurate but NOT a compile lever for #570

**Date:** 2026-06-08
**Tool:** `examples/probe_truncated_backprop_570.py` (CPU, trace-only op-count +
eager gradient parity at D=2).
**Context:** After SVD-via-eigh (a) and batched-decomposition (b) were both
falsified and the wall was re-localized to per-sector STRUCTURAL emission, this
tests the third #570 lever: truncated backprop (TBPTT, `ctm_energy_explicit(
backward_steps=K)`, issue #506 — differentiate only the last K CTM sweeps).

## Result

### Backward op-count (compile proxy), fermionic D=2 χ=6
| K | TBPTT backward ops | per added sweep | vs implicit (63,543) |
|---:|---:|---:|---:|
| 1 | 103,448 | — | 1.63× |
| 2 | 138,043 | +34,595 | 2.17× |
| 4 | 207,233 | +34,595/sweep | 3.26× |
| 8 | 345,613 | +34,595/sweep | 5.44× |

Each differentiated sweep adds ~34,600 ops (one sweep-VJP — the #566 per-sector
structural unit). The **implicit** fixed-point backward compiles ~one sweep-VJP
**in a `while_loop`** (depth-flat, exact gradient via iteration). Explicit-TBPTT
compiles **K sweep-VJPs unrolled**, so it is **strictly ≥ implicit** for compile —
even K=1 is 1.6× (it also carries the energy VJP), and it only grows. Truncated
backprop does **not** reduce the per-sweep VJP, which is the wall.

### Gradient parity (eager, full backprop = reference), D=2
| K | dE vs full | rel‖g − g_full‖ |
|---:|---:|---:|
| 1 | 0 | 3.03e-2 |
| 2 | 0 | 5.52e-3 |
| 4 | 0 | 1.85e-4 |
| 8 | 0 | 1.89e-7 |

The truncated gradient converges geometrically (ρ≈0.16/sweep here); by K=4–8 it is
essentially exact. (Energy is K-independent — only the forward determines it.) So
**TBPTT is correct and accurate** — it is simply a runtime/robustness lever, not a
compile one.

## Consolidated #570 conclusion (three falsifications)

All three decomposition/truncation levers leave the compile wall intact, because
the wall is the **per-sweep VJP op count = #566 per-block STRUCTURAL emission** in
the symmetric SVD/projector wrapper — none of them touch it:

1. **Cheaper decomposition (SVD→eigh):** same `1/(sᵢ²−sⱼ²)` F-matrix; no win.
2. **Batched decomposition (`#572`):** kernels 48→24 but svd_vjp ops −0.7%, total +2.8%.
3. **Truncated backprop:** strictly ≥ implicit for compile (K sweep-VJPs unrolled).

**The only compile lever is the #566 representation restructure:** extend the
stacked-block representation (PR #586, contraction-only) to the SVD/projector
wrapper so the per-sector pack/unpack + gauge-fix batch ACROSS sectors. High blast
radius; the bigger build; but the data says it is the *only* thing that shrinks the
per-sweep VJP.

### What #570's levers ARE good for (not compile)
- **Truncated backprop:** accurate gradient without the implicit adjoint solve →
  avoids the contractivity precheck / `CTMRGGradientError`, and fewer sweep-VJP
  *executions* per step than implicit Neumann. A **runtime/robustness** lever —
  measure warm-step (implicit ~31 s at D=4/χ=12) on A100 to quantify.
- **QR projector + truncated backprop (Yang/Corboz):** a **large-χ GPU/TPU runtime**
  program (QR batches cleanly on accelerators), not a compile-wall fix.

### Practical recommendation
The compile wall is a **one-time** cost. Given all cheap levers are exhausted, the
pragmatic split is: (i) treat compile as amortized and pursue **runtime** via TBPTT
(+ QR on GPU) where it genuinely helps; or (ii) commit to the **#566 stacked-
projector restructure** if the one-time compile minutes are themselves the blocker.
That is a scope decision, not a measurement question — the measurements are in.

## Reproduce
```bash
JAX_PLATFORMS=cpu uv run python examples/probe_truncated_backprop_570.py \
    --D 2 --depth 12 --K 1 2 4 8 --parity
```
