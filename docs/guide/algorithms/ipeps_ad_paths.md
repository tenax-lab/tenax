# iPEPS AD Optimization Paths

This document summarizes the successful automatic differentiation (AD) paths
for iPEPS ground-state optimization in Tenax, based on extensive benchmarking
against the AFM Heisenberg model on the square lattice.

## Recommended Configuration

```python
from tenax import (
    CTMConfig,
    iPEPSConfig,
    heisenberg_gate,
    optimize_gs_ad,
    optimize_gs_ad_chi_schedule,
    sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(heisenberg_gate())

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(
        chi=16,
        max_iter=80,
        conv_tol=1e-8,
        projector_method="qr",  # fastest, best energy, scales to chi=64+
        # forward_gauge defaults to "phase" (AD-correct for 1-site and 2-site)
    ),
    gs_implicit_ad=False,
    gs_explicit_ad_steps=30,
    gs_explicit_ad_warmup=10,
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_c4v=True,
    su_init=True,
)

# Chi-ramping for progressive refinement (recommended for chi > 16)
chi_schedule = [(8, 30), (16, 30), (32, 30)]
A_opt, env, E = optimize_gs_ad_chi_schedule(H_rot, None, config, chi_schedule)
```

## Benchmark Results

**Model**: AFM Heisenberg, D=2, C4v, explicit AD, 30 L-BFGS steps, RTX 4070 Ti GPU, float64

### Phase gauge (recommended) + projector comparison at D=2

| Projector | chi=8 | chi=16 | chi=32 | chi=48 | chi=64 |
|-----------|-------|--------|--------|--------|--------|
| **qr+phase** | -0.6610 (170s) | **-0.6628 (172s)** | -0.6602 (207s) | -0.6622 (259s) | -0.6541 (424s) |
| eigh+phase | -0.6602 (176s) | -0.6602 (836s) | -0.6599 (452s) | — | — |
| svd+phase | -0.6602 (195s) | -0.6602 (805s) | -0.6599 (1109s) | — | — |

**qr+phase** is the new recommended path: best energy at chi=16 (-0.6628),
scales well to chi=64 (2.5x slower than chi=8), and never NaNs.

### Sigma gauge (historical, slower)

| Path | Best E | Time | Notes |
|------|--------|------|-------|
| eigh + sigma (explicit AD) | -0.6601 | 1234s | Slower than phase |
| svd + sigma (two-proj) | -0.6623 | 1124s | Slower than phase |
| eigh + sigma (GMRES implicit) | -0.6601 | — | Implicit AD path |
| Literature (chi=8) | -0.6625 | — | — |
| Exact (QMC, chi→∞) | -0.6694 | — | — |

Phase gauge is **6-9x faster** than sigma gauge for explicit AD with equal
or better energy. Sigma gauge is still needed for implicit AD (GMRES backward).

## Working AD Paths

**Implicit AD (Path 2) is the recommended path and the one you get by
default** — `gs_implicit_ad` is `True` in `iPEPSConfig`. Explicit AD (Path 1)
is opt-in via `gs_implicit_ad=False`. Both paths share the same forward CTM
stack; they differ only in how gradients are computed.

The stability gap tracked by issue #292 is **specific to
`ad_backward_method="gmres"`**, not to implicit differentiation as such. The
default backward is `"vjp"` — the iterative Neumann-series solve — which is
regression-covered and is what the default configuration runs. These headings
previously read "Path 1 (Recommended)" / "Path 2 (Experimental)" while the code
defaulted to Path 2, so the guide recommended against its own default (#808).

### Path 1: Explicit AD (opt-in)

**Architecture**: Warmup CTM sweeps (no gradient) → N backprop sweeps with
phase-gauge fixing and `jax.checkpoint` → gradients accumulate through the
unrolled graph.

```
Forward:  A → warmup sweeps (stop_gradient) → N CTM sweeps (phase gauge, checkpointed) → energy
Backward: dE/dA via backprop through all N sweeps
```

**Strengths**: Best reported energy at chi=16 (-0.6628 with qr+phase),
scales cleanly to chi=64+, never NaNs, and is 6–9× faster than sigma
gauge for equal or better energy.

**Configuration**:
- `gs_implicit_ad=False` — backprop through unrolled steps (explicit AD; opt-in, the default is `True` / implicit diff).
- `gs_projector_method="qr"` — QR projectors (recommended for explicit AD).
- `forward_gauge="phase"` (config default, AD-correct).  Users can
  override with `forward_gauge="sigma"` (historical path), `"qr"`
  (legacy), or `"none"` (diagnostic); see the mode table below.  No
  silent promotion — explicit user choice is preserved.
- `projector_backward="auto"` (config default) — when `projector_method="eigh"`
  and `gs_implicit_ad=False`, `optimize_gs_ad` auto-promotes to `"lorentzian"`,
  routing the projector VJP through the Francuz–Schuch–Vanhecke
  truncated-eigh kernel (`_lorentzian_eigh.py`) instead of the legacy
  `regularized_eigh` path. This closes the 2-site χ=8 convergence gap
  tracked in issue #299 (E_best improves from ≈ −0.558 to ≈ −0.6602 at
  D=2, χ=8). Set to `"standard"` to force the legacy backward, or
  `"lorentzian"` to opt in even when using `projector_method="qr"/"svd"`
  (the flag is a no-op on non-eigh projectors). Currently dense-only;
  U(1) `SymmetricTensor` support is deferred to Approach B of the plan.
- `gs_explicit_ad_steps=30` — number of backprop CTM sweeps.
- `gs_explicit_ad_warmup=10` — warmup sweeps (stop_gradient).

**Why phase gauge works in backprop**: Each phase-gauge step is a
Frobenius normalization plus a differentiable global-phase fix on each
environment tensor. Applied inside the checkpointed unrolled graph, it
removes the gauge ambiguity that causes element-wise CTM convergence to
drift without introducing the power-iteration cost of sigma gauge. The
Frobenius + phase fix is what variPEPS uses in `_post_process_CTM_tensors`.

**Sigma gauge as a fallback**: ``forward_gauge="sigma"`` is still a
first-class mode **on this path**. It is slower (~40% per sweep from power
iteration) but remains available when you want the exact transfer-matrix
alignment. It is *not* available on the implicit path below, which refuses
every value but ``"phase"`` — this paragraph used to say the opposite (#808).
There is no silent promotion either way: ``optimize_gs_ad`` passes
``ctm.forward_gauge`` through unchanged.

### Path 2: Implicit AD (Recommended — the default)

**Architecture**: CTM converges to fixed point → backward solves the
implicit-differentiation linear system at the fixed point → gradients flow
back to the tensor without unrolling.

```
Forward:  A → CTM sweeps (phase gauge) → converged env → energy
Backward: dE/dA via (I - J^T) λ = g  (VJP iteration or GMRES)
```

**Strengths on paper**: Fast convergence, memory-efficient (no unrolled
graph in memory).

**Current status**:
- `ad_backward_method="vjp"` (default) — iterative Neumann-series backward,
  YASTN-style. This is the regression-covered implicit backward.
- `ad_backward_method="gmres"` — direct Krylov solve. **Documented unstable
  without further stabilization**: the `test_gmres_path_agrees_with_vjp`
  regression is currently marked `xfail` in `tests/test_ad_utils.py`. The
  GMRES spectral radius exceeds 1 without careful sigma-gauge alignment and
  a tighter CTM fixed point. Tracked by issue #292.

**Configuration (for the VJP path only)**:
- `forward_gauge="phase"` — the only value this path accepts, and the
  `CTMConfig` default. `validate_ctm_for_implicit_ad`
  (`ipeps_ad_policy.py:30`) raises `ValueError` for anything else, `"sigma"`
  included; there is no `sigma` branch in the check at all.
- `ad_backward_method="vjp"` — the supported implicit backward.
- `gs_implicit_ad=True` — use implicit differentiation. This is the
  `iPEPSConfig` default, so Path 2 is what you get without asking.

> This block used to read `forward_gauge="sigma"` — "required for stable
> element-wise convergence". That was stale rather than a second supported
> mode: transcribed verbatim it raises `ValueError` before the first CTM
> sweep (#808). Sigma gauge remains reachable on the **explicit** AD path
> (Path 1).

**Arnoldi spectral-radius precheck** (enabled by default):

When `adjoint_arnoldi_precheck=True` (default), the backward pass runs 20
Arnoldi iterations on J^T before solving the adjoint system.  If ρ(J^T) ≥ 1,
a `CTMRGGradientError` is raised and the optimizer triggers stall recovery
(noise kick or L-BFGS reset).  This prevents wasting iterations on a
divergent Neumann/GMRES solve.

Inspired by variPEPS (Naumann et al., arXiv:2308.12358) which uses the same
Arnoldi precheck → GMRES fallback pattern.  Set
`adjoint_arnoldi_precheck=False` to disable (e.g. for benchmarking the raw
solver).

This is the YASTN approach (arXiv:2311.11894), adapted for JAX. It is what
runs unless you set `gs_implicit_ad=False`. Reach for Path 1 when you want
backprop through the unrolled sweeps — for a gauge other than `"phase"`, or
to avoid the implicit backward entirely — not as a general default.

### Path 3: 2-site Shared-Tensor C4v (Checkerboard AFM)

For antiferromagnetic models on the square lattice where the Néel order is
explicit in the unit cell, the 2-site shared-tensor C4v path optimizes a
**single** C4v-parameterized tensor ``A`` and derives ``B`` from ``A`` via
sublattice rotation on the physical leg:

```
B = einsum("luRDs,sS->luRDS", A, U_sub)     U_sub = e^{i π σ^y / 2}
```

This ties the two sublattices together, eliminating the A/B drift that
causes the unconstrained 2-site AD path to collapse into non-variational
CTM artifacts. The rotation is spin-1/2 specific — the path raises
``ValueError`` for physical dimension ``d ≠ 2``.

**Enable it with:**

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
    gs_optimizer="lbfgs",
    gs_num_steps=50,
    gs_line_search=True,
    gs_explicit_ad_steps=10,
    gs_explicit_ad_warmup=2,
    su_init=True,
    num_imaginary_steps=100,
    dt=0.3,
    unit_cell="2site",
    gs_c4v=True,
)
_, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
```

**Constraints:**

- spin-1/2 only (``d=2``) — the physical-leg rotation is hard-coded
  to ``e^{i π σ^y/2}``,
- ``gs_stall_recovery="noise"`` is rejected with a clear error (the noise
  branch requires ``(A, B)`` tuple params; use the ``"reset"`` default or
  the ``"reset"`` auto-default for 2-site),
- metric preconditioning is skipped on this path because ``params`` is
  a flat coefficient vector rather than a ``(A, B)`` DenseTensor pair.

Compared to the legacy independent 2-site path (each sublattice parameterized
separately), the shared-tensor approach is stable across chi=8/16/20/24 at
D=2 Heisenberg, while the independent path diverges to unphysical
energies at chi>16.

### Path 4: Reference-Mode Dense C4v (Opt-In)

This opt-in path follows the stable fixed-point differentiation
construction of Francuz, Schuch, Vanhecke, *PRR* **7**, 013237 (2025)
for **dense 1-site C4v** runs:

```
Forward:  A (C4v-projected) -> dense C4v CTM fixed point (single C/T representation)
Backward: stable truncated-eigh backward + implicit solve
          (I - J^T) lambda = g  via bicgstab with gmres fallback
```

**Enable it with:**

```python
config = iPEPSConfig(
    gs_implicit_ad=True,
    gs_c4v=True,
    unit_cell="1x1",
    ctm=CTMConfig(
        chi=16,
        ctm_ad_mode="c4v_reference",
        adjoint_solver="bicgstab",  # or "gmres"
        adjoint_maxiter=50,
        adjoint_tol=1e-8,
    ),
)
```

**Current scope and constraints:**
- dense tensors only (no SymmetricTensor path yet),
- strict gate: `unit_cell="1x1"`, `gs_c4v=True`, `gs_implicit_ad=True`,
  `ctm.ctm_ad_mode="c4v_reference"`,
- supports `gs_num_steps>0` optimization with implicit gradients.

### Path 5: Root Implicit AD (Opt-In, dense 1×1)

Root implicit differentiation of the CTMRG fixed point, following
Burgelman et al., [arXiv:2607.15030](https://arxiv.org/abs/2607.15030).
Instead of back-propagating the CTM sweep, the environment is
characterised by a system of *characteristic equations* `F(y, p) = 0` in
modified variables, and the gradient comes from the implicit function
theorem:

```
Forward:  A -> CTM fixed point -> root parametrisation y* with F(y*, p) = 0
Backward: solve the adjoint of F -- no SVD or eigh backward anywhere in
          the gradient path
```

**This is an accuracy/stability lever, not a speed one.** The paper's own
§VI.3 shows implicit differentiation *losing* to fixed-point iteration at
the bond dimensions we run. The reason to reach for it is #566 and #687:
the block-sparse SVD/eigh VJP compile wall, and the accuracy floor of the
SVD backward on a degenerate discarded spectrum. At `D=3, chi=4` explicit
back-propagation returns **NaN for every entry** — the SVD backward
divides by `s_i^2 - s_j^2` — while this path stays finite and
finite-difference-correct.

**Enable it with:**

```python
config = iPEPSConfig(
    max_bond_dim=2,
    unit_cell="1x1",
    su_init=True,
    gs_num_steps=40,
    gs_optimizer="adam",
    gs_learning_rate=1e-2,
    gs_line_search=False,
    gs_metric_precond=False,          # unsupported here; warns and falls back
    ctm=CTMConfig(chi=6, max_iter=100, conv_tol=1e-10,
                  ctm_ad_mode="root_implicit"),
)
```

**Current scope:**
- **dense 1×1 (asymmetric) only.** `unit_cell` other than `"1x1"` and
  `ctm_ad_mode="root_implicit_symmetric"` both raise `NotImplementedError`
  with the reason; a `SymmetricTensor` input raises `TypeError`. The
  multisite and symmetric engines exist and are tested, but are not wired
  to the optimizer — the symmetric one is blocked on #731 (8.4 GB peak in
  the GMRES solve).
- **Rejected rather than silently ignored:** `chi_auto_bump`, `chi_ramp`,
  `ctmrg_heuristic_increase_chi`, `fuse_virtual_legs=False`,
  `gs_checkpoint_path`, `cg_gates`. These knobs depend on warm-start
  behaviour this path does not have, so it refuses them instead of
  quietly dropping them.

#### The rank clamp, and what the residual gate does and does not mean

The covariant equations carry `S^-1` on both corner legs plus the Eq. 73
quartic roots, so their dependence on the retained spectrum is *cubic*: a
direction whose weight is below `eps^(1/3)` relative to the largest cannot
be resolved in float64, and retaining one makes `‖F(y*)‖` explode.
`_rank_capped_spectrum` clamps those directions (#772).

Physical states routinely trip this. A `D=2` simple-update Heisenberg
state has a `usable_rank` of **3 at every chi from 4 to 24** — its
environment simply does not support more — so every production chi
over-provisions and the clamp always fires. Once it does, `y*` cannot
satisfy the equations below the clamp level however long it is polished,
and the residual floor grows as `sqrt(chi - usable_rank)`. The gate
follows that floor and stays strict (`1e-6`) only where the clamp is
inert, i.e. where a well-conditioned state genuinely reaches ~1e-16
(#779).

> **The root residual is not a measure of gradient accuracy.** Measured
> against a converged directional finite difference, it mispredicts in
> both directions: it rejects a simple-update state whose gradient is
> correct to `3.1e-08` and accepts a poorly-conditioned one whose gradient
> is off by `4.4e-05`. What it reports is whether `y*` solves the
> equations. Tracking a real gradient-quality signal is #785.

#### Do not over-provision chi on this path

**Raising `chi` past the number of directions the state's environment
actually supports makes the gradient *worse*.** This inverts the usual
"a larger `chi` is at worst wasted work" intuition, and it is specific to
this path — it follows from the rank clamp above, which is part of the map
being differentiated.

Measured on one state, changing nothing but `chi`:

| | `usable_rank` | gradient error |
|---|---|---|
| `chi=4` | **4 / 4** (clamp inert) | **7.7e-10** |
| `chi=8` | 4 / 8 (clamp fires) | **2.1e-08** |

**27x worse from raising `chi` alone.** The state supports four directions
either way; at `chi=8` the extra four are noise, the clamp fires on them,
and `y*` stops being an exact root. Every full-rank cut measured
differentiates to ~1e-9, and every clamped one to 3e-8 or worse.

Practical consequence: pick `chi` for this path from where the corner
spectrum actually decays, not by the usual "raise it until the energy stops
moving". A χ-scan that looks converged in *energy* can still be losing
gradient accuracy, because the energy is far less sensitive to the clamp
than the gradient is.

Note this does not make clamped runs unusable — the physical simple-update
state is clamped at *every* chi (`usable_rank=3` from 4 to 24) and its
gradient is accurate to 3.1e-08, which drove the 40-step optimization
below. It means the accuracy band widens, and that `usable_rank < chi` is
the only signal available for it (#785).

#### Benchmark

`D=2`, `chi=6`, square-lattice AFM Heisenberg, Adam at `lr=1e-2` from a
simple-update start:

| step | E/site |
|------|--------|
| 1 (SU start) | -0.481981694887 |
| 5 | -0.525497737892 |
| 10 | -0.579427060195 |
| 20 | -0.623884726814 |
| 30 | -0.636958101476 |
| 40 | **-0.642224027865** |

Monotone at every step, and above the exact ground state (-0.669437) as a
`D=2` state must be.

**Cross-check against explicit AD**, same start, same 40 Adam steps:

| path | E/site | wall-clock |
|------|--------|------------|
| root implicit | -0.642224022008 | 926.4 s |
| explicit AD (Path 1) | -0.643015471158 | 14.8 s |
| difference | 7.9e-04 | **63x slower** |

The two land 7.9e-04 apart, and explicit AD is still descending at
`dE = 2.4e-04` per step at step 40 — so that gap is roughly three steps of
remaining progress, not a disagreement between the gradients.

The 63x is the point to take seriously, and it is not a defect to be tuned
away: it is the same conclusion as the paper's §VI.3. **Do not choose this
path for speed.** Choose it when explicit back-propagation cannot produce a
gradient at all — at `D=3, chi=4` it returns NaN for every entry while this
path is finite and finite-difference-correct — or to avoid the block-sparse
SVD/eigh VJP compile wall (#566, #687).

## Forward Gauge Mode Matrix

Tenax supports four ``forward_gauge`` modes. Their intended use is
summarized below:

| Mode | Explicit AD (Path 1) | Implicit AD (Path 2, VJP) | Notes |
|------|----------------------|----------------------------|-------|
| ``"phase"`` (default) | **Recommended** | **The only accepted value** | Cheapest gauge fix; Frobenius + differentiable phase fix. Works for 1-site and 2-site. |
| ``"qr"`` | Legacy QR gauge | Refused (`ValueError`) | Forward-only CTM, notebooks, diagnostics. |
| ``"sigma"`` | Historical — still correct but ~6–9× slower than phase | Refused (`ValueError`) | Power iteration (30 steps) per sweep. |
| ``"none"`` | Benchmark / diagnostic only | Refused (`ValueError`) | Isolates gauge-fix cost from projector cost. |

The Path 2 column is a hard gate, not a preference: `validate_ctm_for_implicit_ad`
accepts `forward_gauge="phase"` and nothing else, and it narrows by neither
`chi` nor unit cell — so the older "at large chi (1-site only)" qualifier on
the `sigma` row described a configuration that never ran (#808).

**No silent gauge promotion**: ``optimize_gs_ad`` passes
``ctm.forward_gauge`` through unchanged.  An explicit user choice
(``"qr"``, ``"sigma"``, ``"phase"``, or ``"none"``) is always
respected.  This was previously achieved through an auto-promotion of
``"qr"`` → ``"phase"``; the promotion was removed in PR #343 in favor
of a sensible static default.

The GMRES backward (``ad_backward_method="gmres"``) is tracked as an open
gap — see issue #292 and the ``xfail``-marked regression test in
``tests/test_ad_utils.py``.

## Critical Components

### 1. Phase Gauge Fixing (default for explicit AD)

The phase gauge fix is two differentiable steps applied to every
corner and edge after each CTM sweep:

1. **Frobenius normalization** — divides each tensor by its Frobenius norm
   so that absorbed-layer singular values cannot exponentially grow or
   shrink across sweeps.
2. **Global phase fix** — picks the first sufficiently large element of
   each tensor and rotates the global U(1) phase so that this element is
   real-positive (variPEPS ``_post_process_CTM_tensors`` convention).

Together they remove the dominant gauge ambiguity at negligible cost —
no power iteration, no eigensolve, fully differentiable — and are the
reason the qr+phase path scales to chi=64 without NaNs.

### 2. Sigma Gauge Fixing (implicit-diff path)

Sigma gauge aligns each iteration's environment to the previous one using
transfer matrix eigenvectors, making element-wise convergence monotonic.
This is required for the implicit-diff backward, where a well-conditioned
fixed-point environment is needed for the ``(I - J^T) λ = g`` solve to
behave well.

**Implementation**: Power iteration (30 iterations) computes the leading
eigenvector of the double-layer transfer matrix. This is fully
JAX-differentiable, unlike `jnp.linalg.eig`.

**Sweep mutation fix (PR #291)**: The multisite CTM sweep used to mutate
the environment dict in-place. A shallow copy (`envs = dict(envs)`) at
the start of each sweep ensures callers that saved a reference to the
input dict (for sigma gauge comparison) still see the pre-sweep
environments. Before this fix, sigma gauge in the Python convergence
loop silently degenerated into a no-op.

### 3. Projector Methods

**qr** (recommended for explicit AD): QR-factored small eigenproblem. Best
benchmarked energy at D=2 with the phase gauge and scales cleanly to
chi=64+. Fails for AD **without** a gauge fix (phase or sigma).

**eigh**: Forms density matrix ρ = C1g·C1g† + C4g·C4g† and eigendecomposes.
Best energy with sigma gauge; slower than qr for large chi. Block-sparse
path available for SymmetricTensor.

**svd** (Fishman): Cross-product M = C1g†·C4g, SVD, projector P = C4g·V·S^{-1/2}.
Works with sigma gauge (E=-0.6624). The S^{-1/2} weighting is differentiable
(no stop_gradient), allowing gradient flow through singular values.

### 4. SVD Projector: Two-Projector Fishman

The two-projector Fishman formulation (arXiv:2502.10298) computes a
bi-orthogonal pair:

- P_1 = C4g·V·S^{-1/2} (applied to C1g side)
- P_2 = C1g·U·S^{-1/2} (applied to C4g side)

satisfying P_1†·P_2 = I. Both corners get clean projections (S^{1/2}·U†
and S^{1/2}·V†), and gradients flow through all SVD factors (U, S, V).

For eigh/qr projectors, P_1 = P_2 = P (standard isometric projector).

This matches the variPEPS and YASTN implementations. The remaining
differences are in numerical conditioning: variPEPS uses 2×2 enlarged
corners with Fishman low-rank pre-truncation; YASTN uses QR pre-factoring
of half-corners; Tenax uses neither (QR backward is unstable for
rank-deficient matrices during AD).

### 5. L-BFGS + Hager-Zhang Line Search

Second-order optimization with approximate Wolfe conditions. Metric
preconditioning (Rader et al., arxiv:2511.09546) uses the environment metric
tensor as a natural gradient preconditioner.

### 6. C4v Symmetry Enforcement

For the square lattice with C4v-symmetric Hamiltonians, enforcing C4v symmetry
on the site tensor reduces the parameter space from D²d to ~D²d/8, improving
optimization stability and speed.

## Known Limitations

1. **GMRES implicit backward**: ``ad_backward_method="gmres"`` is documented
   unstable (spectral radius > 1 without tight sigma-gauge alignment) and
   its regression test is marked ``xfail`` in ``tests/test_ad_utils.py``.
   Tracked by issue #292. Prefer the explicit-AD path (Path 1) or
   ``ad_backward_method="vjp"`` until the GMRES path is stabilized.

2. **``forward_gauge="none"``**: The ``"none"`` gauge is honored
   end-to-end as a benchmark / diagnostic mode. It skips all gauge
   fixing in the CTM loop.

3. **2-site explicit AD**: The 2-site optimizer supports the same
   auto-phase promotion as the 1-site C4v path. For antiferromagnetic
   models, prefer the 2-site **shared-tensor C4v path** (Path 3) over
   the unconstrained 2-site optimizer, which is known to drift apart
   and produce unphysical energies. The unconstrained path emits an
   ``experimental`` warning on entry.

4. **SymmetricTensor**: Block-sparse tensors fall back to the Python loop
   (not JIT-traceable). Both phase and sigma gauge work on this path, but
   with dense fallbacks for gauge fixing.

5. **Sigma gauge cost**: ~40% overhead from power iteration per sweep.
   Phase gauge is the recommended replacement for explicit AD; reserve
   sigma gauge for the implicit-diff path.

6. **Root implicit AD has no gradient-quality signal**: the root residual
   it reports says whether `y*` solves the characteristic equations, and
   that is measurably *not* the same as whether the gradient is accurate —
   it mispredicts in both directions (see Path 5). `usable_rank` does not
   separate the cases either. Gradient accuracy on this path currently has
   to be established by finite differences offline, not by anything the
   library reports at run time. Tracked by issue #785.

## Stall recovery (`gs_stall_recovery`)

When the L-BFGS / CG line search fails to make progress, the optimizer
runs a stall-recovery routine. Two modes are supported:

- ``"noise"`` — inject a ``gs_noise_amplitude`` (default 10 %) Frobenius
  perturbation on the current params and reset the L-BFGS history.
  **Required for the 1-site C4v production path**, which sits on an
  SU-init plateau with gradient norms around ``1e-10`` that would
  otherwise trip ``gs_conv_tol`` before the first real step.
- ``"reset"`` — clear the L-BFGS ``(s, y)`` history and the CG beta
  state so the next iteration is a plain (preconditioned) steepest
  descent step from the current iterate. No rollback, no randomness.
  **Default for the 2-site path** because the 10 % noise kick in the
  ~32-dimensional D=2 parameter space lands in non-variational CTM
  regions and drives the optimizer into unphysical "best" energies
  (see issue #298).

Leaving ``gs_stall_recovery=None`` (the default) auto-selects the
right mode for the unit cell at dispatch time. An explicit user
setting is never overridden.

For extra safety on 2-site runs, set ``gs_energy_floor`` to a value a
bit below the expected variational minimum (e.g. ``2 * E_literature``).
Any in-loop candidate energy at or below the floor is rejected as a
non-variational CTM artifact — this catches pathological "best"
states arising from the ``_rdm2x1_tensor_2site`` trace-normalization
at near-zero trace. The check is off by default
(``gs_energy_floor=None``).

## Known open problems

- **2-site L-BFGS convergence gap at χ=8** — The 2-site AD path with
  L-BFGS + Hager-Zhang + metric precond + SU init reaches only
  ``E/site ≈ -0.56`` at D=2 χ=8 in 20 steps, vs. the ≈-0.65 literature
  value documented in issue #298's trajectory study. The gap is not
  a stall-recovery problem (the reset branch never fires on this
  trajectory); root cause is under investigation. Tracked by issue
  #299.

## References

- Francuz, Schuch, Vanhecke, *PRR* **7**, 013237 (2025) — Stable AD through CTM
- Rader et al., arXiv:2511.09546 (2025) — Metric preconditioning
- Zhang, Yang, Corboz, arXiv:2505.00494 (2025) — Chi-ramping schedule
- Naumann et al., arXiv:2502.10298 (2025) — Split CTMRG with two-projector formulation
- Fishman et al., *PRB* **98**, 235148 (2018) — SVD (Fishman) projectors
