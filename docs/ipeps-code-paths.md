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
     │ = "lbfgs"         │                  │ = "cg"                │
     │ (default)         │                  │ Polak-Ribiere+        │
     │ + hager_zhang LS  │                  │ + metric_precond      │
     │ + metric_precond  │                  │   (natural gradient)  │
     │                   │                  │                       │
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
 (explicit AD)                                  (default, RECOMMENDED)
          │                                                 │
 ┌────────▼─────────────────────┐                ┌──────────▼───────────────┐
 │ ctm_tensor_converge_explicit │                │ ctm_ad_mode selects      │
 │ (ad_utils.py)                │                │ the backward path:       │
 │                              │                └──────────┬───────────────┘
 │ warmup (stop_gradient)       │                           │
 │ ──── W steps ────            │                ┌──────────┴─────────┐
 │                              │                │                    │
 │ forward_gauge: passed         │                ▼                    ▼
 │   through unchanged           │         ctm_ad_mode=None      ctm_ad_mode=
 │   (no silent promotion)       │         (custom_vjp           "c4v_reference"
 │                              │          ctm_tensor_converge) (Francuz et al.,
 │ backprop (jax.checkpoint)    │                               dense 1-site
 │ ──── B steps ────            │          ┌───────┼───────┐    C4v, opt-in)
 │                              │          │               │
 │ gauge modes honored:         │          ▼               ▼    adjoint_solver:
 │   qr / phase / sigma / none  │        "vjp"        xxxxxxxxxxx  bicgstab /
 │                              │        (default,    x "gmres" x  gmres
 │                              │         w/ phase)   x BROKEN  x
 │                              │                     x issue   x  resolve_projector
 │                              │                     x #292    x  _backward
 │                              │                     xxxxxxxxxxx  enforces
 │                              │                                  svd + phase +
 │                              │                                  elementwise
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

## Recommended Path (post-PR-#341)

For AFM models on the square lattice, the recommended path is
**implicit AD with the SVD (Fishman) projector + phase gauge +
element-wise CTM convergence**, the "trifecta" combination
empirically validated in PR #341.  ``resolve_projector_backward`` in
``ipeps_ad_policy.py`` enforces this combination at dispatch time and
raises ``ValueError`` if any of the three components is set
otherwise.  Explicit AD with QR projectors remains supported as a
secondary path (see below).

The defaults already match the recommended combination:
``gs_implicit_ad=True``, ``ctm.projector_method="svd"``,
``ctm.forward_gauge="phase"``, ``ctm.ctm_conv_method="elementwise"``,
``gs_optimizer="lbfgs"``, ``gs_metric_precond=True``.  No silent gauge
or projector promotion is applied.

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad, sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(H)        # map AFM → ferromagnetic

config = iPEPSConfig(
    unit_cell="1x1",
    gs_c4v=True,                         # enforce C4v symmetry
    # Defaults below are already correct for the recommended path:
    #   gs_implicit_ad=True
    #   gs_optimizer="lbfgs" + hager_zhang line search
    #   gs_metric_precond=True
    ctm=CTMConfig(
        chi=16,
        max_iter=80,
        # All three are defaults — shown here for visibility:
        projector_method="svd",          # Fishman two-projector
        forward_gauge="phase",           # Frobenius + first-above-threshold phase fix
        ctm_conv_method="elementwise",
    ),
    su_init=True,
)

A_opt, env, E = optimize_gs_ad(H_rot, A_init=None, config=config)
```

For the **explicit-AD + QR-CTMRG** path that was the recommendation
through PR #291 (best-scaling projector at D=2 for ``chi ≥ 16``):

```python
config = iPEPSConfig(
    unit_cell="1x1",
    gs_c4v=True,
    gs_implicit_ad=False,                # explicit AD path
    gs_explicit_ad_steps=20,
    gs_explicit_ad_warmup=2,
    gs_projector_method="qr",
    ctm=CTMConfig(chi=16, max_iter=80, projector_method="qr"),
    su_init=True,
)
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

## Coarse-Grained iPEPS (PR #352 / #353)

For non-square lattices (honeycomb, kagome) Tenax provides a
**coarse-grained iPEPS (CG iPEPS)** path that maps the lattice onto the
1-site square pipeline by grouping a unit cell of physical sites into a
single tensor with effective physical dimension ``d_eff = d ** n_sites``.
The optimizer reuses the standard ``optimize_gs_ad`` square machinery —
no new CTM or AD path — only the gate construction and energy
contraction differ.

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, optimize_gs_ad, honeycomb_cg_gates

cg = honeycomb_cg_gates(J=1.0)        # d_eff = 4 (two spin-1/2 sub-sites)
config = iPEPSConfig(
    unit_cell="1x1",
    cg_gates=cg,
    su_init=False,                     # cg_gates requires random init
    # Standard implicit-AD trifecta otherwise.
)
# The active Hamiltonian lives on cg_gates; optimize_gs_ad still reads
# gate.shape[0] for its dispatch / random A_init shape, so we pass a
# d_eff-shaped placeholder of the correct rank.
dummy_gate = jnp.zeros((4, 4, 4, 4))   # (d_eff, d_eff, d_eff, d_eff)
A_opt, env, E = optimize_gs_ad(dummy_gate, A_init=None, config=config)
```

``honeycomb_cg_gates`` builds 2 sub-sites per cell (``d_eff = 4``);
``kagome_cg_gates`` builds 3 sub-sites per up-triangle (``d_eff = 8``).
``CGGates.map_fn`` and ``CGGates.init_fn`` let the user parameterize the
CG tensor as a contraction of raw site tensors (e.g. for variational
restrictions or staggered initialization); when both are ``None`` the
optimizer treats the CG tensor itself as the variational parameter.

Constraints (enforced by ``iPEPSConfig.__post_init__``):

- requires ``unit_cell="1x1"``;
- incompatible with ``su_init=True`` (no SU on coarse-grained tensors).

This is the recommended path for honeycomb/kagome AFM ground states at
moderate ``D``; for native honeycomb topology see the next section.

## Native Honeycomb CTM (PR #347)

A separate, **native rank-4 honeycomb CTM** lives under
``src/tenax/algorithms/_ctm_honeycomb_*.py`` (re-exported via
``tenax.algorithms.honeycomb_ctm``). It works directly with the
two-sublattice honeycomb topology — 6 corners, 3 directions, 2
sublattice CTM tensors — rather than fusing the two sites into a CG
tensor. Public entry points: ``honeycomb_ctm_run`` (forward),
``honeycomb_ctm_energy_implicit`` (with implicit-AD backward),
``HoneycombCTMEnv`` (env container), ``compute_honeycomb_energy`` and
``compute_honeycomb_triangle_energy``.

The native path is more faithful to honeycomb geometry (smaller
effective bond dimension at the same ``chi``) but does **not** plug into
``optimize_gs_ad`` directly yet — wire-up is tracked separately. CG
iPEPS remains the easiest production-AD entry point for honeycomb until
that lands.

> **Note (PR #387 — kagome iPESS).** The native honeycomb CTM was
> originally motivated by serving kagome iPESS via a 2-sublattice
> honeycomb supersite construction (the design doc's "Convention A"
> + the M2b across-sublattice 2-site RDM follow-up to PR #347). PR #387
> instead delivered kagome iPESS through the **CG iPEPS** path
> (``kagome_cg_gates`` + ``optimize_gs_ad``, "Convention C" — square
> 1-site supersite with a dummy-bond), reusing the validated
> 1-site square AD pipeline. The kagome use case therefore no longer
> depends on the native honeycomb CTM. The native path remains
> useful for **honeycomb-native lattice models** (honeycomb Heisenberg,
> Kitaev, Lukin–Sotnikov bipartite) where the dummy-bond detour to a
> square iPEPS introduces SVD-degenerate-spectrum issues at AD time
> that the native path avoids by construction.

## Auto-χ_E Bump (variPEPS §2.8.2)

When the CTM truncation error `ε_T = ‖discarded SVs‖₂ / ‖SVs‖₂` exceeds
`chi_auto_bump_eps`, `optimize_gs_ad` increases `chi` by `chi_auto_bump_step`
between L-BFGS steps and zero-pads the cached environment as a warm start.
Disabled by default (opt-in). Mutually exclusive with `chi_ramp`.

The bump fires *between* optimizer steps — the implicit-AD GMRES linearisation
sees a fixed χ within each gradient evaluation, so AD correctness within a
single backward solve is unaffected.

> **Variational caveat (issue #511).** Implicit AD's variational guarantee
> requires the CTM env to be a *converged* fixed point at the current χ.
> End-of-outer-step `chi_auto_bump` (and scheduled `chi_ramp`) zero-pads
> env rows for newly-active χ indices: the first few gradient evaluations
> after a bump see a non-fixed-point env, during which the optimizer can
> descend to a non-physical "ghost minimum" below the true variational
> floor (v8b D=3 bipartite: E_best = −0.844 vs QMC ≈ −0.669).  For
> χ-grown runs **prefer `ctmrg_heuristic_increase_chi=True`** (in-CTM
> bump, see #492/#514), which grows χ *during* CTM convergence and
> never returns a partial env to the optimizer.  The end-of-outer-step
> `chi_auto_bump` path is retained for explicit-AD runs and for
> backwards-compatibility; see also memory note
> `feedback_drop_chi_schedule_protocol.md`.

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    ctm=CTMConfig(
        chi=10,
        chi_auto_bump=True,
        chi_auto_bump_eps=1e-5,   # variPEPS §2.8.2 default
        chi_auto_bump_step=2,
        chi_max=40,               # hard ceiling; None = unbounded
    ),
    ...
)
A_opt, env, E = optimize_gs_ad(gate, A_init, config)
```

Reference: Naumann, Weerda, Rizzi, Eisert, Schmoll, *SciPost Phys. Lect. Notes* **86** (2024), §2.8.2.

> **Scope:** dense single-site (`unit_cell="1x1"`) path only.
> Block-sparse (SymmetricTensor) and multisite paths are tracked as follow-up issues.

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
| Implicit AD (svd + phase + elementwise) | **Working** | **Default and recommended.** Enforced by ``resolve_projector_backward`` (PR #341). |
| Explicit AD (standard CTM)        | **Working**      | ``gs_implicit_ad=False``. Warmup + checkpoint, ``forward_gauge`` honored as set. |
| Implicit diff + VJP backward      | **Working**      | Regression-covered. Default ``ad_backward_method``; uses ``forward_gauge="phase"`` (the only value the implicit path accepts). |
| Implicit diff + GMRES backward    | **BROKEN**       | ``ad_backward_method="gmres"`` user knob still flagged unstable (spectral radius > 1); ``xfail`` regression, issue #292. The internal Python-loop CTM AD calls JAX's ``gmres_pytree_jax`` directly and is *not* gated by this knob. |
| C4v + sublattice rotation         | **Working**      | Recommended Zhang/Corboz-style path.                               |
| 2-site shared-tensor C4v          | **Working**      | ``unit_cell="2site"`` + ``gs_c4v=True``; spin-1/2 only (PR #304).  |
| 2-site independent A/B            | **EXPERIMENTAL** | Unconstrained 2-site AD; needs ``complex128`` site tensors to stay variational (real ``float64`` drifts non-variationally — see ``project_complex_tensors_variational`` notes / PR #341). |
| Reference-mode C4v AD             | **Working**      | ``ctm_ad_mode="c4v_reference"``; dense 1-site C4v only (PR #304).  |
| QR-CTMRG (C4v)                    | **Working**      | Best-scaling projector at D=2 for explicit AD; recommended for ``chi ≥ 16`` on the explicit path. Implicit AD requires ``"svd"``. |
| Phase gauge (``forward_gauge``)   | **Working**      | variPEPS-style Frobenius + first-above-threshold phase fix per absorption. Default for both implicit and explicit AD. |
| Sigma gauge (``forward_gauge``)   | **Working**      | Transfer-matrix eigenvector alignment, **1-site only**. Breaks 2-site (inconsistent A/B alignment causes gradient explosion). Not auto-promoted; opt-in for 1-site users mirroring YASTN. |
| ``forward_gauge="none"``          | **Working**      | Diagnostic / benchmark mode; honored on the explicit-AD path.     |
| ``forward_gauge="none"`` on JIT   | **EXPERIMENTAL** | JIT ``while_loop`` kernel falls back to ``"qr"``; known limitation.|
| ``gs_ctm_conv_tol_schedule``      | **Working**      | Loose-to-tight CTM tolerance ramp; optional tuning knob.           |
| Metric preconditioning            | **Working**      | Natural-gradient preconditioner for CG / L-BFGS.                   |
| Stall recovery (1-site)           | **Working**      | ``gs_stall_recovery="noise"`` auto-default; required by C4v path.  |
| Stall recovery (2-site)           | **Working**      | ``gs_stall_recovery="reset"`` auto-default since #298.             |
| 2-site L-BFGS at χ=8              | **Working**      | ``E_best ≈ -0.6602`` at D=2 with Lorentzian projector backward (issue #299 closed; post-convergence re-eval tracked separately by #317). |
| Lorentzian projector backward     | **Aspirational** | ``CTMConfig`` documents auto-promotion of ``"auto"`` to ``"lorentzian"`` when ``gs_implicit_ad=False`` + ``projector_method="eigh"``, but the ``"auto"`` resolver is **not yet implemented** (see ``_ctm_projector.py`` "Task 8 will resolve 'auto'"); ``"auto"`` currently behaves as ``"standard"``. Setting ``projector_backward="lorentzian"`` explicitly works. |
| Adjoint Arnoldi precheck          | **Working**      | ``adjoint_arnoldi_precheck=True`` (default) probes ``J^T``'s spectral radius before the Krylov solve; falls back to a regularized solve when ``> adjoint_arnoldi_threshold`` (default 5.0). |
| Adjoint Tikhonov damping          | **Working**      | ``adjoint_tikhonov`` (default ``1e-6``) adds ``+τI`` to ``(I − J^T)`` to prevent Krylov stalls near a well-converged GS. |
| Auto-χ_E bump (variPEPS §2.8.2)  | **Working**      | ``CTMConfig(chi_auto_bump=True, chi_auto_bump_eps=1e-5, chi_auto_bump_step=2, chi_max=N)``. Dense single-site path only; SymmetricTensor + multisite are follow-up issues. |
| Split CTM forward (SU)            | **Working**      | Used by simple update.                                             |
| Split CTM + implicit diff         | **Working** (single-site, dense) | ``CTMConfig(fuse_virtual_legs=False)`` + ``unit_cell="1x1"`` routes the whole 1-site ``optimize_gs_ad`` through the split χ²·D⁴ forward, on either ``gs_recipe`` (the default ``"2x2"`` since #746, or ``"1x1"`` -- which collapses the environment to rank-1 corners and is bisection-only, see #726) (warm-start, line-search probe, final env all split-aware; returns a ``SplitCTMTensorEnv``). Γ-gauge-fixed fixed-point ``custom_vjp`` with Neumann backward (``ctm_energy_split_implicit``); env_init warm-start carries zero cotangent. Implicit grad matches the trusted explicit grad to ~1e-12. Dense ``DenseTensor`` only (SymmetricTensor/fermionic = later phase); fixed χ (``chi_ramp`` / ``chi_auto_bump`` / ``ctmrg_heuristic_increase_chi`` rejected via ``validate_split_ctm_config``); memory win is large-D (D≳16). PR #648/#651/#652. |
| Split CTM + explicit diff         | **Working** (single-site, dense) | ``ctm_energy_split_explicit`` (unrolled warmup+backprop). Also dispatched through ``optimize_gs_ad`` via ``fuse_virtual_legs=False`` + ``gs_implicit_ad=False``; the unrolled forward re-initializes each eval (no env_init warm-start yet). No ``jax.checkpoint``, higher memory than implicit. |
| Fermionic iPEPS AD                | **EXPERIMENTAL** | Wraps ``SymmetricTensor`` as ``DenseTensor``. Fermionic Koszul twist (`bar_super()`) added in PR #361 fixes super-algebra sign issues that previously broke fermionic AD with non-trivial parity sectors. |
| Coarse-grained iPEPS (honeycomb)  | **Working**      | ``cg_gates=honeycomb_cg_gates()`` + ``unit_cell="1x1"``; reuses square 1-site AD pipeline at ``d_eff=4`` (PR #352 / #353). |
| Coarse-grained iPEPS (kagome)     | **Working**      | ``cg_gates=kagome_cg_gates()`` + ``unit_cell="1x1"``; reuses square 1-site AD pipeline at ``d_eff=8`` (PR #352 / #353). |
| Kagome iPESS AD (Convention C)    | **Working**      | ``kagome_xxz_pess_cg_gates()`` + ``optimize_pess_ad``; spin-½ and spin-1 XXZ via the same CG iPEPS path (PR #387). Optimizes ``(R_a, R_b, R_c, T_u, lambdas)`` on the dummy-bond square supersite. |
| Native honeycomb CTM (rank-4)     | **Working** (forward + implicit-AD energy); not wired into ``optimize_gs_ad`` | 6-corner / 3-direction / 2-sublattice (PR #347). Originally motivated by kagome iPESS but the kagome use case ships through the CG path (PR #387) instead; native CTM now applies primarily to honeycomb-native models (Kitaev, hc Heisenberg). |

### Benchmark highlights (2D Heisenberg AFM)

| D | chi | Path                                        | E (best)   | Literature / exact |
|---|-----|---------------------------------------------|------------|--------------------|
| 2 | 16  | qr + phase + explicit AD (1-site C4v)       | -0.6628    | -0.6548 (Corboz D=2) |
| 2 | 8   | qr + phase + explicit AD (1-site C4v)       | -0.6610    | —                  |
| 2 | 8   | svd + phase + implicit AD (2-site C4v)      | -0.6602    | -0.6548 (Corboz D=2 χ=16) |
| 3 | 16  | svd + phase + implicit AD (2-site C4v)      | -0.6521    | (50 steps, monotonic, 7.9 s/step) |
| 2 | 16  | svd + phase + implicit AD (2-site, complex128, non-C4v) | -0.6406 | variational; ≈1 min/step on GPU |
| — | —   | QMC exact                                   | -0.66944   | Sandvik, PRB 56, 11678 (1997) |

The 2-site non-C4v entry requires ``complex128`` site tensors —
``float64`` drifts non-variationally because the implicit-diff linear
system ``(I − J^T)λ = g`` is ill-conditioned in the real subspace.

See [`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md)
for the full benchmark table and the projector × gauge comparison matrix.

## Config Cheat Sheet

| Setting               | Flag                        | Default          | Options                              |
|-----------------------|-----------------------------|------------------|--------------------------------------|
| Init                  | ``su_init``                 | ``True``         | ``False`` = random init              |
| Unit cell             | ``unit_cell``               | ``"1x1"``        | ``"2site"`` = checkerboard           |
| Optimizer             | ``gs_optimizer``            | ``"lbfgs"``      | ``"cg"`` / ``"adam"``                |
| Line search           | ``gs_line_search_method``   | ``"hager_zhang"``| ``"armijo"``                         |
| C4v symmetry          | ``gs_c4v``                  | ``False``        | ``True`` = enforce C4v               |
| AD method             | ``gs_implicit_ad``          | ``True``         | ``False`` = explicit AD              |
| Implicit AD mode      | ``ctm_ad_mode``             | ``None``         | ``"c4v_reference"`` (Francuz et al., 1-site C4v) |
| Implicit adjoint      | ``adjoint_solver``          | ``"bicgstab"``   | ``"gmres"`` (reference-mode only)    |
| Adjoint max iters     | ``adjoint_maxiter``         | ``50``           | reference-mode only                  |
| Adjoint tol           | ``adjoint_tol``             | ``1e-8``         | reference-mode only                  |
| Adjoint Tikhonov      | ``adjoint_tikhonov``        | ``1e-6``         | ``0.0`` for strictly exact adjoint; raise to 1e-4…1e-3 near a well-converged GS |
| Adjoint Arnoldi precheck | ``adjoint_arnoldi_precheck`` | ``True``    | spectral-radius probe before Krylov solve |
| Adjoint Arnoldi threshold | ``adjoint_arnoldi_threshold`` | ``5.0``   | fall back to regularized solve when ρ(J^T) exceeds this |
| Warmup steps          | ``gs_explicit_ad_warmup``   | ``3``            | stop_gradient CTM steps (explicit AD) |
| Backprop steps        | ``gs_explicit_ad_steps``    | ``20``           | differentiable CTM steps (explicit AD) |
| AD projector override | ``gs_projector_method``     | ``None``         | ``"qr"`` (recommended for explicit AD); implicit AD requires ``"svd"`` |
| Backward              | ``ad_backward_method``      | ``"vjp"``        | ``"gmres"`` (BROKEN — issue #292)    |
| Projector             | ``projector_method``        | ``"svd"``        | ``"eigh"`` / ``"qr"`` (recommended for explicit AD only)  |
| Projector backward    | ``projector_backward``      | ``"auto"``       | ``"standard"`` / ``"lorentzian"`` (``"auto"`` resolver not yet implemented; behaves as ``"standard"``) |
| Forward gauge         | ``forward_gauge``           | ``"phase"``      | ``"qr"`` / ``"sigma"`` / ``"none"`` (no silent promotion; implicit AD requires ``"phase"``) |
| CTM conv method       | ``ctm_conv_method``         | ``"elementwise"``| ``"sv"`` (singular-value); implicit AD requires ``"elementwise"`` |
| Conv tol schedule     | ``gs_ctm_conv_tol_schedule``| ``None``         | ``[(frac, tol), ...]``               |
| Metric precond        | ``gs_metric_precond``       | ``True``         | ``False`` = standard grad            |
| Stall recovery        | ``gs_stall_recovery``       | ``None``         | ``"noise"`` / ``"reset"`` (auto)     |
| Energy floor          | ``gs_energy_floor``         | ``None``         | ``float`` = reject below as non-variational |
| CG iPEPS gates        | ``cg_gates``                | ``None``         | ``honeycomb_cg_gates()`` / ``kagome_cg_gates()`` / custom ``CGGates`` (requires ``unit_cell="1x1"``, rejects ``su_init=True``) |
| Auto-χ_E bump         | ``chi_auto_bump``           | ``False``        | ``True`` = opt-in (§2.8.2)          |
| Bump threshold        | ``chi_auto_bump_eps``       | ``1e-5``         | variPEPS §2.8.2 default             |
| Bump step             | ``chi_auto_bump_step``      | ``2``            | additive increment per trigger      |
| Max χ ceiling         | ``chi_max``                 | ``None``         | ``int`` = hard cap on bumped χ      |
| CTM variant           | (function choice)           | standard         | split, C4v                          |

The static defaults
(``projector_method="svd"``, ``forward_gauge="phase"``, ``ctm_conv_method="elementwise"``)
are the AD-correct choices for both implicit and explicit AD.  For
the implicit-AD path these three values are *enforced* by
``resolve_projector_backward`` — any other combination raises
``ValueError`` at dispatch.  For the explicit-AD path
``optimize_gs_ad`` passes the user's configured gauge and projector
through unchanged.  ``build_ad_ctm_config`` performs **no silent
promotion** of any value: only ``gs_projector_method`` (when set)
overrides ``ctm.projector_method``.  Direct ``CTMConfig()`` users get
the same behavior the optimizer uses.

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
| Coarse-grained iPEPS (honeycomb/kagome) | ``src/tenax/algorithms/coarse_grain.py``      |
| Kagome iPESS AD                         | ``src/tenax/algorithms/pess.py``, ``pess_optimize.py`` |
| Native honeycomb CTM (rank-4)           | ``src/tenax/algorithms/honeycomb_ctm.py``, ``_ctm_honeycomb_*.py`` |
| Fermionic Koszul twist (super-algebra)  | ``src/tenax/core/tensor.py`` (``bar_super()``, PR #361) |

## Related Documents

- [`docs/guide/algorithms/ipeps.md`](guide/algorithms/ipeps.md) — user guide
  for the iPEPS algorithm.
- [`docs/guide/algorithms/ctm.md`](guide/algorithms/ctm.md) — user guide
  for the CTM environment computation.
- [`docs/guide/algorithms/ipeps_ad_paths.md`](guide/algorithms/ipeps_ad_paths.md)
  — benchmarked recommendation for iPEPS AD (config, gauge mode matrix,
  critical components, known limitations).
