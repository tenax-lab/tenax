# iPEPS Code Paths Overview

This document is the **architectural map** of Tenax's iPEPS stack: which
entry points dispatch to which CTM variant, where AD happens, and where
the known-broken or experimental paths live. For **user-facing
recommendations** (what config to use for a production AD run) see
[`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md).

Paths marked **BROKEN** are known non-functional. Paths marked
**EXPERIMENTAL** have not been validated at production scale.

## Pipeline Graph

```
                         ┌─────────────────────┐
                         │  optimize_gs_ad()    │
                         │  ipeps_optimize.py   │
                         └─────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                   │
          ┌─────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
          │ su_init    │    │  random     │    │ A_init /    │
          │ (SU→AD)    │    │ (fallback)  │    │ AB_init     │
          │ (default)  │    │             │    │ (explicit)  │
          └─────┬─────┘    └──────┬──────┘    └──────┬──────┘
                └──────────────────┼──────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                                     │
       ┌────────▼────────┐                 ┌──────────▼──────────┐
       │ unit_cell="1x1" │                 │ unit_cell="2site"   │
       │ SINGLE_SITE     │                 │ CHECKERBOARD        │
       │                 │                 │                     │
       │ gs_c4v=True:    │                 │ Neel init for       │
       │  symmetrize_c4v │                 │ AFM (user provides) │
       │  + sublattice   │                 │                     │
       │  rotation       │                 │                     │
       └────────┬────────┘                 └──────────┬──────────┘
                └──────────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
     ┌────────▼──────────┐                  ┌───────────▼───────────┐
     │ gs_optimizer       │                  │ gs_optimizer          │
     │ = "lbfgs"          │                  │ = "cg" (default)     │
     │ + hager_zhang LS   │                  │ Polak-Ribiere+       │
     │ + metric_precond   │                  │ + metric_precond     │
     │                    │                  │   (natural gradient) │
     │ (recommended)      │                  │                      │
     └────────┬──────────┘                  └───────────┬───────────┘
              └────────────────────┬────────────────────┘
                                   │
                             ┌─────▼─────┐
                             │ loss_fn() │
                             │ E(A) via  │
                             │ CTM       │
                             └─────┬─────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                                                 │
 gs_explicit_ad=True                           gs_explicit_ad=False
 (default, RECOMMENDED)                        (implicit, EXPERIMENTAL)
          │                                                 │
 ┌────────▼─────────────────────┐                ┌──────────▼─────────┐
 │ ctm_tensor_converge_explicit  │                │ ctm_tensor_        │
 │ (ad_utils.py)                 │                │ converge()         │
 │                               │                │ custom_vjp         │
 │ warmup (stop_gradient)        │                │                    │
 │ ──── W steps ────              │                └──────────┬─────────┘
 │                               │                           │
 │ auto-phase gauge promotion:   │                   ┌───────┼───────┐
 │  qr → phase when user left    │                   │               │
 │  forward_gauge="qr"           │                   ▼               ▼
 │                               │                 "vjp"        xxxxxxxxxxx
 │ backprop (jax.checkpoint)     │                 (supported)  x "gmres" x
 │ ──── B steps ────              │                               x BROKEN x
 │                               │                               x issue  x
 │ gauge modes honored:          │                               x #292   x
 │   qr / phase / sigma / none   │                               xxxxxxxxxxx
 └────────┬──────────────────────┘
          │
          │ (both paths share the forward CTM stack below)
          │
          ▼
     ┌─────────────────┐
     │ Standard CTM    │
     │ _ctm_tensor_*   │
     │ double-layer a  │
     │ O(chi^3 D^6)    │
     └────────┬────────┘
              │
     ┌────────┼──────────────────────┐
     │        │                      │
 ┌───▼──────────────┐  ┌────────────▼─────────────────┐
 │ Standard CTM     │  │ C4v CTM                      │
 │ 4 moves/sweep    │  │ 1 move/sweep                 │
 │ general unit     │  │ ctm_tensor_c4v()             │
 │ cells            │  │                              │
 └───┬──────────────┘  │ projector="qr":              │
     │                 │  QR-CTMRG (2505.00494)       │
     │                 │  O(chi^2 D^2) per projector  │
     │                 └────────────┬─────────────────┘
     └──────────┬───────────────────┘
                │
       ┌────────┼────────────────────┐
       │        │                    │
 ┌─────▼────┐ ┌─▼────────┐  ┌────────▼────┐
 │ eigh     │ │ qr       │  │ svd         │
 │ ρ=CC†+…  │ │ QR+eigh  │  │ Fishman     │
 │ O(χ³ D⁶) │ │ O(χ³ D²) │  │ two-proj    │
 └──────────┘ └──────────┘  └─────────────┘
       │        │                    │
       └────────┼────────────────────┘
                │
     ┌──────────┼──────────┐
     │                     │
 ┌───▼─────────────┐  ┌───▼───────────────┐
 │ SymmetricTensor │  │ DenseTensor       │
 │ block-sparse    │  │ (or AD-traced)    │
 │ per-sector eigh │  │                   │
 │ stop_gradient   │  │ AD: regularized   │
 │                 │  │   SVD backward    │
 └─────────────────┘  └───────────────────┘

Legend:
  ┌────────┐  working path
  └────────┘

  xxxxxxxxxxxxxx
  x BROKEN    x  known broken / incomplete (issue #292)
  xxxxxxxxxxxxxx
```

## Recommended Path (post-PR-#291)

For AFM models on the square lattice, the post-PR-#291 recommended path is
sublattice rotation + C4v + explicit AD + QR projectors. The optimizer
transparently promotes the conservative default ``forward_gauge="qr"`` to
``"phase"`` (variPEPS-style Frobenius + phase fix) for the unrolled CTM
sweeps, which is 6–9× faster than the historical sigma gauge with equal or
better energy.

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad, sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(H)        # map AFM → ferromagnetic

config = iPEPSConfig(
    unit_cell="1x1",
    gs_c4v=True,                         # enforce C4v symmetry
    gs_optimizer="lbfgs",                # L-BFGS + Hager-Zhang line search
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_explicit_ad=True,                 # default; unrolled + checkpointed
    gs_explicit_ad_steps=20,
    gs_explicit_ad_warmup=2,
    gs_projector_method="qr",
    ctm=CTMConfig(chi=16, max_iter=80, projector_method="qr"),
    su_init=True,
)

A_opt, env, E = optimize_gs_ad(H_rot, A_init=None, config=config)
```

See [`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md)
for the full benchmark table, the forward-gauge mode matrix, and the
``gs_ctm_conv_tol_schedule`` tuning knob.

## Status Summary

| Path                              | Status           | Notes                                                              |
|-----------------------------------|------------------|--------------------------------------------------------------------|
| Explicit AD (standard CTM)        | **Working**      | Default. Warmup + checkpoint, auto-phase gauge for explicit AD.    |
| Implicit diff + VJP backward      | **Working**      | Regression-covered; use with ``forward_gauge="sigma"``.            |
| Implicit diff + GMRES backward    | **BROKEN**       | Spectral radius > 1; ``xfail`` regression. Tracked by issue #292.  |
| C4v + sublattice rotation         | **Working**      | Recommended Zhang/Corboz-style path.                               |
| QR-CTMRG (C4v)                    | **Working**      | Best-scaling projector at D=2; recommended for chi ≥ 16.           |
| Phase gauge (``forward_gauge``)   | **Working**      | variPEPS-style Frobenius + phase fix; default for explicit AD.     |
| Sigma gauge (``forward_gauge``)   | **Working**      | Historical; still required for the implicit-diff path.             |
| ``forward_gauge="none"``          | **Working**      | Diagnostic / benchmark mode; honored on the explicit-AD path.     |
| ``forward_gauge="none"`` on JIT   | **EXPERIMENTAL** | JIT ``while_loop`` kernel falls back to ``"qr"``; known limitation.|
| ``gs_ctm_conv_tol_schedule``      | **Working**      | Loose-to-tight CTM tolerance ramp; optional tuning knob.           |
| Metric preconditioning            | **Working**      | Natural-gradient preconditioner for CG / L-BFGS.                   |
| Split CTM forward (SU)            | **Working**      | Used by simple update.                                             |
| Split CTM + implicit diff         | **BROKEN**       | Not wired into optimizer.                                          |
| Split CTM + explicit diff         | **Working**      | No ``jax.checkpoint``, high memory.                                |
| Fermionic iPEPS AD                | **EXPERIMENTAL** | Wraps ``SymmetricTensor`` as ``DenseTensor``.                      |

### Benchmark highlights (2D Heisenberg AFM)

| D | chi | Path                         | E (best)   | Literature / exact |
|---|-----|------------------------------|------------|--------------------|
| 2 | 16  | qr + phase + explicit AD     | -0.6628    | -0.6548 (Corboz D=2)|
| 2 | 8   | qr + phase + explicit AD     | -0.6610    | —                  |
| — | —   | QMC exact                    | -0.66944   | Sandvik, PRB 56, 11678 (1997) |

See [`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md)
for the full benchmark table and the projector × gauge comparison matrix.

## Config Cheat Sheet

| Setting               | Flag                        | Default          | Options                              |
|-----------------------|-----------------------------|------------------|--------------------------------------|
| Init                  | ``su_init``                 | ``True``         | ``False`` = random init              |
| Unit cell             | ``unit_cell``               | ``"1x1"``        | ``"2site"`` = checkerboard           |
| Optimizer             | ``gs_optimizer``            | ``"cg"``         | ``"lbfgs"`` (recommended for AD)     |
| C4v symmetry          | ``gs_c4v``                  | ``False``        | ``True`` = enforce C4v               |
| AD method             | ``gs_explicit_ad``          | ``True``         | ``False`` = implicit diff            |
| Warmup steps          | ``gs_explicit_ad_warmup``   | ``3``            | stop_gradient CTM steps              |
| Backprop steps        | ``gs_explicit_ad_steps``    | ``20``           | differentiable CTM steps             |
| AD projector override | ``gs_projector_method``     | ``None``         | ``"qr"`` (recommended)               |
| Backward              | ``ad_backward_method``      | ``"vjp"``        | ``"gmres"`` (BROKEN — issue #292)    |
| Projector             | ``projector_method``        | ``"eigh"``       | ``"qr"`` (recommended) / ``"svd"``   |
| Forward gauge         | ``forward_gauge``           | ``"qr"``         | ``"phase"`` / ``"sigma"`` / ``"none"`` |
| Conv tol schedule     | ``gs_ctm_conv_tol_schedule``| ``None``         | ``[(frac, tol), ...]``               |
| Metric precond        | ``gs_metric_precond``       | ``True``         | ``False`` = standard grad            |
| CTM variant           | (function choice)           | standard         | split, C4v                           |

The static default ``forward_gauge="qr"`` is kept conservative so that
callers who construct a ``CTMConfig`` directly (forward-only CTM,
notebooks, diagnostics) see predictable behavior.  ``optimize_gs_ad``
auto-promotes to ``"phase"`` at runtime when ``gs_explicit_ad=True`` and
the user has not opted into a different gauge.

## Key Files

| Responsibility                          | File                                          |
|-----------------------------------------|-----------------------------------------------|
| Main optimizer entry                    | ``src/tenax/algorithms/ipeps_optimize.py``    |
| Config dataclasses                      | ``src/tenax/algorithms/ipeps_config.py``      |
| Sublattice rotation, C4v symmetrization | ``src/tenax/algorithms/ipeps.py``             |
| CTM fixed-point AD (implicit/explicit)  | ``src/tenax/algorithms/ad_utils.py``          |
| Standard CTM (Tensor protocol)          | ``src/tenax/algorithms/_ctm_tensor*.py``      |
| C4v CTM + QR-CTMRG                     | ``src/tenax/algorithms/_ctm_tensor_c4v.py``   |
| Split CTM (Tensor protocol)             | ``src/tenax/algorithms/_split_ctm_tensor*.py``|
| CTM projectors (eigh/QR/SVD)            | ``src/tenax/algorithms/_ctm_projector.py``    |
| Metric preconditioning                  | ``src/tenax/algorithms/_metric_precond.py``   |
| Legacy CTM (dense arrays, SU only)      | ``src/tenax/algorithms/ipeps_ctm*.py``        |
| Simple update                           | ``src/tenax/algorithms/ipeps_simple_update.py``, ``ipeps.py`` |
| Energy computation                      | ``src/tenax/algorithms/_ctm_tensor_energy.py``, ``_split_ctm_tensor_energy.py`` |
| Fermionic variant                       | ``src/tenax/algorithms/fermionic_ipeps.py``   |

## Related Documents

- [`docs/guide/algorithms/ipeps.md`](guide/algorithms/ipeps.md) — user guide
  for the iPEPS algorithm.
- [`docs/guide/algorithms/ctm.md`](guide/algorithms/ctm.md) — user guide
  for the CTM environment computation.
- [`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md)
  — benchmarked recommendation for iPEPS AD (config, gauge mode matrix,
  critical components, known limitations).
