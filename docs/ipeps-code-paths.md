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
                         │  optimize_gs_ad()   │
                         │  ipeps_optimize.py  │
                         └─────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                   │
          ┌─────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
          │ su_init   │    │  random     │    │ A_init /    │
          │ (SU→AD)   │    │ (fallback)  │    │ AB_init     │
          │ (default) │    │             │    │ (explicit)  │
          └─────┬─────┘    └──────┬──────┘    └──────┬──────┘
                └──────────────────┼──────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                                     │
       ┌────────▼────────┐                 ┌──────────▼──────────┐
       │ unit_cell="1x1" │                 │ unit_cell="2site"   │
       │ SINGLE_SITE     │                 │ CHECKERBOARD        │
       │                 │                 │                     │
       │ gs_c4v=True:    │                 │ gs_c4v=True:        │
       │  symmetrize_c4v │                 │  shared-tensor C4v  │
       │  + sublattice   │                 │  B = σ^y · A        │
       │  rotation       │                 │  (spin-1/2 only)    │
       │                 │                 │ gs_c4v=False:       │
       │                 │                 │  independent A,B    │
       │                 │                 │  (EXPERIMENTAL)     │
       └────────┬────────┘                 └──────────┬──────────┘
                └──────────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
     ┌────────▼──────────┐                  ┌───────────▼───────────┐
     │ gs_optimizer      │                  │ gs_optimizer          │
     │ = "lbfgs"         │                  │ = "cg" (default)      │
     │ + hager_zhang LS  │                  │ Polak-Ribiere+        │
     │ + metric_precond  │                  │ + metric_precond      │
     │                   │                  │   (natural gradient)  │
     │ (recommended)     │                  │                       │
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
 gs_implicit_ad=False                           gs_implicit_ad=True
 (default, RECOMMENDED)                        (implicit)
          │                                                 │
 ┌────────▼─────────────────────┐                ┌──────────▼───────────────┐
 │ ctm_tensor_converge_explicit │                │ ctm_ad_mode selects      │
 │ (ad_utils.py)                │                │ the backward path:       │
 │                              │                └──────────┬───────────────┘
 │ warmup (stop_gradient)       │                           │
 │ ──── W steps ────            │                ┌──────────┴─────────┐
 │                              │                │                    │
 │ auto-phase gauge promotion:  │                ▼                    ▼
 │  qr → phase when user left   │         ctm_ad_mode=None      ctm_ad_mode=
 │  forward_gauge="qr"          │         (custom_vjp           "c4v_reference"
 │                              │          ctm_tensor_converge) (Francuz et al.,
 │ backprop (jax.checkpoint)    │                               dense 1-site
 │ ──── B steps ────            │          ┌───────┼───────┐    C4v, opt-in)
 │                              │          │               │
 │ gauge modes honored:         │          ▼               ▼    adjoint_solver:
 │   qr / phase / sigma / none  │        "vjp"        xxxxxxxxxxx  bicgstab /
 │                              │        (supported)  x "gmres" x  gmres
 │                              │                     x BROKEN  x
 │                              │                     x issue   x
 │                              │                     x #292    x
 │                              │                     xxxxxxxxxxx
 └────────┬─────────────────────┘
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
     ┌────────┼─────────────────────┐
     │        │                     │
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
    # gs_implicit_ad=False is the default (explicit AD, recommended)
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

## 2-site Shared-Tensor C4v (PR #304)

For AFM checkerboard unit cells the 2-site optimizer supports a
**shared-tensor C4v** mode, enabled by ``unit_cell="2site"`` together
with ``gs_c4v=True``. A single C4v-parameterized tensor ``A`` is
optimized and the second sublattice is derived as
``B = einsum("luRDs,sS->luRDS", A, U_sub)`` with ``U_sub = e^{iπσ^y/2}``.
This ties the two sublattices together and avoids the drift into
non-variational CTM artifacts that affects the unconstrained 2-site
path (issue #299).

Constraints:

- spin-1/2 only (physical dim ``d=2``); other dims raise ``ValueError``;
- ``gs_stall_recovery="noise"`` is rejected (the noise branch requires
  an ``(A, B)`` tuple — use ``"reset"`` or leave the default auto);
- metric preconditioning is skipped because ``params`` is a flat
  coefficient vector, not an ``(A, B)`` DenseTensor pair.

The unconstrained 2-site path (``gs_c4v=False``) still exists for
diagnostics but emits an ``experimental`` warning on entry; prefer the
shared-tensor path for AFM models.

## Reference-Mode C4v AD (PR #304, renamed in #306)

A reference implicit-AD path that follows the stable
fixed-point-differentiation construction of Francuz, Schuch, Vanhecke,
*PRR* **7**, 013237 (2025) — stable truncated-eigh backward plus an
implicit linear solve at the CTM fixed point. It is available for
**dense 1-site C4v** runs only, lives in
``src/tenax/algorithms/_ctm_tensor_c4v_reference_ad.py``, and is opt-in
via ``CTMConfig(ctm_ad_mode="c4v_reference")``.

```
Forward:  A (C4v-projected) → dense C4v CTM fixed point (single C/T)
Backward: stable truncated-eigh backward
        + implicit solve  (I - J^T) λ = g
          via bicgstab with gmres fallback
```

Strict gate — all of the following must hold, or the optimizer falls
back to the standard path:

- ``unit_cell="1x1"``,
- ``gs_c4v=True``,
- ``gs_implicit_ad=True``,
- ``ctm.ctm_ad_mode="c4v_reference"``.

Additional knobs on ``CTMConfig``: ``adjoint_solver`` (``"bicgstab"`` or
``"gmres"``), ``adjoint_maxiter``, ``adjoint_tol``. On solver failure
the Krylov adjoint raises rather than silently returning a
non-solution, and records a ``converged`` flag in the backward meta
dict. Dense tensors only for now — no SymmetricTensor path yet.

## Stall Recovery (issue #298)

When the L-BFGS / CG line search cannot make progress the optimizer runs
``gs_stall_recovery``. The knob is auto-defaulted per unit cell at
dispatch time:

- **1-site** (``_optimize_gs_ad_tensor``) → ``"noise"``: inject a
  ``gs_noise_amplitude`` Frobenius perturbation and reset the L-BFGS
  history. Required for the C4v production path to break out of the
  SU-init plateau, where gradient norms ≈ ``1e-10`` would otherwise
  trip ``gs_conv_tol``.
- **2-site** (``_optimize_gs_ad_tensor_2site``) → ``"reset"``: clear
  the L-BFGS ``(s, y)`` history and CG beta state so the next step is
  a plain (preconditioned) steepest descent step from the current
  iterate. No randomness, no rollback. Needed because the 10 % noise
  kick in the 32-dim D=2 space teleports the state into non-variational
  CTM regions and produces unphysical "best" energies.

An explicit ``gs_stall_recovery`` setting is never overridden by the
dispatcher. For extra safety on 2-site runs, set ``gs_energy_floor`` to
a value below the expected variational minimum — any in-loop candidate
energy at or below the floor is rejected as a non-variational CTM
artifact. Both knobs are off / auto by default.

The 2-site L-BFGS path still has a separate convergence gap at
χ=8 (reaches only ``E ≈ -0.558`` vs literature ``-0.6548``) that is
**not** a stall-recovery problem — tracked by issue #299.

## Status Summary

| Path                              | Status           | Notes                                                              |
|-----------------------------------|------------------|--------------------------------------------------------------------|
| Explicit AD (standard CTM)        | **Working**      | Default. Warmup + checkpoint, auto-phase gauge for explicit AD.    |
| Implicit diff + VJP backward      | **Working**      | Regression-covered; use with ``forward_gauge="sigma"``.            |
| Implicit diff + GMRES backward    | **BROKEN**       | Spectral radius > 1; ``xfail`` regression. Tracked by issue #292.  |
| C4v + sublattice rotation         | **Working**      | Recommended Zhang/Corboz-style path.                               |
| 2-site shared-tensor C4v          | **Working**      | ``unit_cell="2site"`` + ``gs_c4v=True``; spin-1/2 only (PR #304).  |
| 2-site independent A/B            | **EXPERIMENTAL** | Unconstrained 2-site AD; drifts to unphysical energies (issue #299).|
| Reference-mode C4v AD             | **Working**      | ``ctm_ad_mode="c4v_reference"``; dense 1-site C4v only (PR #304).  |
| QR-CTMRG (C4v)                    | **Working**      | Best-scaling projector at D=2; recommended for chi ≥ 16.           |
| Phase gauge (``forward_gauge``)   | **Working**      | variPEPS-style Frobenius + phase fix; default for explicit AD.     |
| Sigma gauge (``forward_gauge``)   | **Working**      | Historical; still required for the implicit-diff path.             |
| ``forward_gauge="none"``          | **Working**      | Diagnostic / benchmark mode; honored on the explicit-AD path.     |
| ``forward_gauge="none"`` on JIT   | **EXPERIMENTAL** | JIT ``while_loop`` kernel falls back to ``"qr"``; known limitation.|
| ``gs_ctm_conv_tol_schedule``      | **Working**      | Loose-to-tight CTM tolerance ramp; optional tuning knob.           |
| Metric preconditioning            | **Working**      | Natural-gradient preconditioner for CG / L-BFGS.                   |
| Stall recovery (1-site)           | **Working**      | ``gs_stall_recovery="noise"`` auto-default; required by C4v path.  |
| Stall recovery (2-site)           | **Working**      | ``gs_stall_recovery="reset"`` auto-default since #298.             |
| 2-site L-BFGS at χ=8              | **Working**      | ``E_best ≈ -0.6602`` at D=2 with Lorentzian projector backward (issue #299 closed; post-convergence re-eval tracked separately by #317). |
| Lorentzian projector backward     | **Working**      | Auto-default when ``gs_implicit_ad=False`` + ``projector_method="eigh"``; 1-site + 2-site, dense only (SymmetricTensor deferred to Approach B). |
| Split CTM forward (SU)            | **Working**      | Used by simple update.                                             |
| Split CTM + implicit diff         | **BROKEN**       | Not wired into optimizer.                                          |
| Split CTM + explicit diff         | **Working**      | No ``jax.checkpoint``, high memory.                                |
| Fermionic iPEPS AD                | **EXPERIMENTAL** | Wraps ``SymmetricTensor`` as ``DenseTensor``.                      |

### Benchmark highlights (2D Heisenberg AFM)

| D | chi | Path                         | E (best)   | Literature / exact |
|---|-----|------------------------------|------------|--------------------|
| 2 | 16  | qr + phase + explicit AD     | -0.6628    | -0.6548 (Corboz D=2)|
| 2 | 8   | qr + phase + explicit AD     | -0.6610    | —                  |
| 2 | 8   | eigh + phase + lorentzian + 2-site C4v | -0.6602 | -0.6548 (Corboz D=2 χ=16) |
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
| AD method             | ``gs_implicit_ad``          | ``True``         | ``False`` = explicit AD (recommended) |
| Implicit AD mode      | ``ctm_ad_mode``             | ``None``         | ``"c4v_reference"`` (Francuz et al., 1-site C4v) |
| Implicit adjoint      | ``adjoint_solver``          | ``"bicgstab"``   | ``"gmres"`` (reference-mode only)    |
| Adjoint max iters     | ``adjoint_maxiter``         | ``50``           | reference-mode only                  |
| Adjoint tol           | ``adjoint_tol``             | ``1e-8``         | reference-mode only                  |
| Warmup steps          | ``gs_explicit_ad_warmup``   | ``3``            | stop_gradient CTM steps              |
| Backprop steps        | ``gs_explicit_ad_steps``    | ``20``           | differentiable CTM steps             |
| AD projector override | ``gs_projector_method``     | ``None``         | ``"qr"`` (recommended)               |
| Backward              | ``ad_backward_method``      | ``"vjp"``        | ``"gmres"`` (BROKEN — issue #292)    |
| Projector             | ``projector_method``        | ``"eigh"``       | ``"qr"`` (recommended) / ``"svd"``   |
| Projector backward    | ``projector_backward``      | ``"auto"``       | ``"standard"`` / ``"lorentzian"`` (auto-promoted to lorentzian when ``gs_implicit_ad=False`` and ``projector_method="eigh"``) |
| Forward gauge         | ``forward_gauge``           | ``"qr"``         | ``"phase"`` / ``"sigma"`` / ``"none"`` |
| Conv tol schedule     | ``gs_ctm_conv_tol_schedule``| ``None``         | ``[(frac, tol), ...]``               |
| Metric precond        | ``gs_metric_precond``       | ``True``         | ``False`` = standard grad            |
| Stall recovery        | ``gs_stall_recovery``       | ``None``         | ``"noise"`` / ``"reset"`` (auto)     |
| Energy floor          | ``gs_energy_floor``         | ``None``         | ``float`` = reject below as non-variational |
| CTM variant           | (function choice)           | standard         | split, C4v                           |

The static default ``forward_gauge="qr"`` is kept conservative so that
callers who construct a ``CTMConfig`` directly (forward-only CTM,
notebooks, diagnostics) see predictable behavior.  ``optimize_gs_ad``
auto-promotes to ``"phase"`` at runtime when ``gs_implicit_ad=False`` and
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
| Reference-mode C4v AD (Francuz et al.)  | ``src/tenax/algorithms/_ctm_tensor_c4v_reference_ad.py`` |
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
