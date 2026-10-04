# Corner Transfer Matrix (CTM)

The Corner Transfer Matrix (CTM) method computes the environment of an
infinite 2D tensor network (iPEPS) by iteratively absorbing rows and columns
until convergence.

## Background

CTM represents the infinite environment using 4 corner tensors (C1–C4) and
4 edge tensors (T1–T4). Each sweep grows the corners/edges by absorbing the
double-layer tensor, then truncates back to bond dimension χ via a projector.

Tenax provides three CTM variants:

### Standard CTM (``ctm_tensor``)

The general-purpose CTM using the Tensor protocol (works with both
``DenseTensor`` and ``SymmetricTensor``). Uses 4 directional moves per sweep.

- Supports ``"eigh"`` and ``"qr"`` projector methods.
- For fermionic ``SymmetricTensor``, automatically uses paired moves
  (``_ctm_tensor_paired_moves``) to prevent charge-sector divergence.
  Falls back to ``DenseTensor`` when paired moves cannot be applied.

### C4v CTM (``ctm_tensor_c4v``)

Exploits C4v point-group symmetry for 1-site unit cells. Stores only one
corner and one edge, performing a **single move per sweep**. This eliminates
the charge-distribution divergence entirely.

- Only valid for models without sublattice structure.
- Returns a full ``CTMTensorEnv`` by expanding via C4v symmetry relations.

### 2-site CTM (``ctm_tensor_2site``)

For checkerboard (A/B sublattice) unit cells. Maintains separate environments
for each sublattice.

- Supports Heisenberg antiferromagnet, Néel order, and other models with
  2-sublattice structure.
- Also available for general multi-site unit cells via ``ctm_multisite``.

## Configuration

CTM is configured via ``CTMConfig`` (used in iPEPS pipelines) or directly
via function arguments:

```python
from tenax import CTMConfig

ctm_cfg = CTMConfig(
    chi=32,                      # environment bond dimension
    max_iter=100,                # maximum CTM iterations
    conv_tol=1e-10,              # convergence tolerance on corner singular values
    forward_gauge="auto",       # "auto" (default), "phase", "bond_phase", "qr", "sigma", or "none"
    projector_method="svd",     # "svd" (Fishman, default), "eigh", or "qr"
    ad_backward_method="vjp",   # "vjp" (default) or "gmres" (experimental)
)
```

### Growing chi inside CTM convergence

The recommended way to let the environment bond dimension grow is the in-CTM
χ-bump of variPEPS §2.8.2 (Naumann et al., SciPost Phys. Lect. Notes 86, 2024):
set ``ctmrg_heuristic_increase_chi=True`` together with a ``chi_max`` ceiling.
CTM convergence then raises ``chi`` by ``ctmrg_heuristic_increase_chi_step_size``
whenever the smallest kept singular value (relative to the largest) of a
projector SVD exceeds ``ctmrg_heuristic_increase_chi_threshold``, so the
environment is always converged at the new χ before the optimizer sees it.

```python
ctm_cfg = CTMConfig(chi=9, chi_max=16, ctmrg_heuristic_increase_chi=True)
```

This avoids the zero-padded-environment cliff-edge artifact that the legacy
end-of-outer-step ``chi_auto_bump`` and the scheduled ``chi_ramp`` (below)
introduce between L-BFGS steps. Both legacy knobs still work but emit a
``DeprecationWarning`` (issue #512) and will be removed in a future release.

### Chi ramping (deprecated)

```{warning}
``chi_ramp`` is deprecated and emits a ``DeprecationWarning`` (#512). Replace
``chi_ramp=[(9, 20), (12, 20), (16, 20)]`` with
``chi=9, chi_max=16, ctmrg_heuristic_increase_chi=True``.
```

The ``chi_ramp`` field runs CTM convergence in stages at increasing chi,
reducing total cost by doing cheap sweeps at small chi before the final
convergence:

```python
ctm_cfg = CTMConfig(
    chi=32,
    chi_ramp=[(8, 10), (16, 10), (32, None)],  # (chi, num_sweeps)
)
```

Each tuple is ``(chi, num_sweeps)``. The last entry (or any with
``num_sweeps=None``) runs to convergence. Environments are
re-initialized when chi changes between stages. Benchmarks show
1.2–2.1× speedup on GPU with identical energies.

### Forward gauge

The ``forward_gauge`` parameter controls how gauge ambiguity is resolved
after each CTM sweep. Five modes are supported, plus the ``"auto"`` default:

| Value | Description |
|-------|-------------|
| ``"auto"`` (default) | Resolved per path (``resolve_forward_gauge``): ``"bond_phase"`` on the fused implicit-AD path with no ``chi_ramp`` and ``ctm_ad_mode=None``; ``"phase"`` everywhere else (explicit AD, split CTM, ``chi_ramp``, ``ctm_ad_mode`` engines, legacy ``ad_utils`` paths). Never warns. |
| ``"phase"`` | variPEPS-style Frobenius normalization + phase fixing. Cheapest gauge fix. Applied by ``ctm_energy_implicit`` (accepted on the implicit path, 1-site and 2-site) and the legacy ``ad_utils`` paths. What ``"auto"`` resolves to off the implicit path, but ``optimize_gs_ad``'s explicit and split energies apply no forward gauge, so there it has no effect (#1074). |
| ``"bond_phase"`` | Implicit-AD path only (``ctm_energy_implicit``): ``"phase"`` plus one sign/phase per chi index of every bond family, aligned to the previous environment (#841). An exact gauge transform that pins the per-bond-index signs the projector SVD re-draws each sweep. What ``"auto"`` runs on the implicit path; set explicitly elsewhere, it raises. |
| ``"qr"`` | Legacy QR decomposition on corners with sign-fixed diagonal. Fast and stable for simple update and forward-only CTM. |
| ``"sigma"`` | Transfer-matrix eigenvector alignment via power iteration. Required for element-wise CTM convergence at large chi (1-site path). Under ``optimize_gs_ad`` it is refused on implicit AD and has no effect on explicit AD (#1074). |
| ``"none"`` | No gauge fix. Diagnostic / benchmark mode only — isolates the cost of gauge fixing from the rest of the sweep. Not recommended for production runs. |

Without a gauge fix, the ``eigh`` projector CTM converges spectrally (corner
singular values stabilize) but is chaotic element-wise — the individual
tensor entries keep fluctuating between iterations, which makes implicit
differentiation ill-conditioned. ``forward_gauge="phase"`` fixes this with
negligible overhead, while ``forward_gauge="sigma"`` is appropriate when
strict element-wise convergence is needed at large chi.

**No silent gauge promotion.** Only the ``"auto"`` default is resolved;
``optimize_gs_ad`` passes an explicitly configured ``forward_gauge`` through
unchanged.  Set it explicitly (e.g. ``"phase"``) to override the default.

### Projector methods

``projector_method`` selects how each sweep truncates back to χ:

| Value | Description |
|-------|-------------|
| ``"svd"`` (default) | Fishman two-projector SVD with safe singular-value handling. |
| ``"eigh"`` | Hermitian eigendecomposition of the corner density matrix. |
| ``"qr"`` | Reduced-corner QR-CTMRG isometry (Zhang, Yang & Corboz, arXiv:2505.00494). |

```{warning}
**``projector_method`` is consulted only on the ``"1x1"`` recipe, and that
recipe is deprecated (#911).** The ``"2x2"`` default hardcodes Fishman SVD and
ignores this parameter entirely -- ``svd``/``eigh``/``qr`` give bit-identical
``2x2`` energies.

This page used to say "set ``gs_recipe="1x1"`` + ``gs_projector_method="qr"`` to
run reduced-corner QR-CTMRG under the implicit-diff AD optimizer". Do not: #911
measured that recipe reaching **no** fixed point in any reachable
configuration, for any state with D > 1. (D=1 is the one exception -- rank 1 is
the maximum reachable corner rank there, so the collapse is vacuous and ``1x1``
matches ``2x2`` exactly -- but the recipe is still being removed.)
Under ``svd`` it collapses the corner to rank 1; under ``eigh``/``qr`` it holds
full rank but limit-cycles, with the energy ranging over 3.4e-3--4.9e-3 across
the last 40 of 240 sweeps.

For ``qr`` or ``eigh`` on a C4v-symmetric state, use ``ctm_tensor_c4v`` -- a
different function that calls the same projector, runs all three methods at full
rank, and agrees with ``recipe="2x2"`` to 1e-12. The SymmetricTensor/block-sparse
``"qr"`` path is a later phase.
```

### AD backward method

| Value | Description |
|-------|-------------|
| ``"vjp"`` (default) | Iterative VJP (Neumann series). Robust; the only implicit-diff backward that is currently regression-covered end-to-end. |
| ``"gmres"`` | Direct Krylov solve of ``(I - J^T) λ = g``. **Experimental / documented unstable** — the GMRES backward is tracked as an open gap (see issue #292) and its regression test is currently marked ``xfail``. |

For new code prefer the explicit-AD path (``gs_implicit_ad=False``), which
does not exercise the implicit backward at all and does not require GMRES.
See {doc}`ipeps_ad_paths` for the complete recommended configuration.

## Example — standalone CTM

```python
from tenax import ctm_tensor, ctm_tensor_c4v, compute_energy_ctm_tensor

# A is an iPEPS site tensor (DenseTensor or SymmetricTensor)
# with 5 legs (u, d, l, r, phys)
env = ctm_tensor(A, chi=32, max_iter=100, conv_tol=1e-10)
E = compute_energy_ctm_tensor(A, env, hamiltonian_gate, d=2)

# Or with C4v symmetry (1-site, no sublattice)
env_c4v = ctm_tensor_c4v(A, chi=32, max_iter=100, conv_tol=1e-10)
```

(ctm-convergence-check)=
## Checking whether the CTM actually converged

`ctm`, `ctm_2site`, `ctm_split` and `ctm_tensor` return an environment whether
or not the sweep met `conv_tol` — running out of `max_iter` is not an error.
Pass `return_meta=True` for a `CTMConvergenceInfo` saying which happened, rather
than inferring it from an energy that silently moves with `max_iter` (#839):

```python
from tenax import CTMConfig, ctm_2site

env_A, env_B, info = ctm_2site(A, B, CTMConfig(chi=16), return_meta=True)
if not bool(info.converged):
    print(f"stopped at max_iter after {int(info.n_iter)} sweeps, "
          f"criterion still {float(info.diff):.2e}")
```

`info.diff` is the convergence criterion — the change in the corner singular
values, not in the energy. `ipeps()` performs this check itself and warns.

`ctm_tensor` takes the same flag, and returns the info as a *third* element
after `(env, max_truncation_error)`:

```python
from tenax import ctm_tensor
from tenax.algorithms._ctm_diagnostics import env_is_collapsed

env, eps_T, info = ctm_tensor(A, chi=16, max_iter=100, return_meta=True)
if not info.converged:
    # inf means the criterion never produced a value: either fewer than two
    # sweeps ran, or the corner collapsed to rank 1 and the criterion refused
    # to certify it (#898).  Only the second is unfixable by more sweeps.
    reason = "collapsed" if env_is_collapsed(env) else "budget"
    print(f"not a fixed point ({reason}): {info.n_iter} sweeps, diff {info.diff:.2e}")
```

(ctm-hold-test)=
## Mixing for two-state cycles

`CTMConfig(ctm_mixing=β)` (default `0.0`, off) damps the element-wise CTM
forward, `E ← (1−β)·F(E) + β·E`. That maps a fixed-point multiplier λ to
`(1−β)λ + β`, so a sweep stuck in an exact two-state cycle (λ ≈ −1, the χ=20
case of #1060) converges with β ≈ 0.3. The price is slower convergence of
modes with λ near +1.

- Fixed points are unchanged. Convergence is still certified on the
  undamped residual `|F(E) − E|`, and the implicit backward linearises the
  undamped `F`.
- Requires `forward_gauge="bond_phase"` and `ctm_conv_method="elementwise"`:
  averaging is only meaningful between environments in one fixed gauge.
- `CTMConvergeInfo.step_multiplier` reports the signed λ estimate of the last
  two sweeps: near −1 is a flip cycle mixing cures, near +1 a slow
  contraction it slows further, NaN fewer than two comparable sweeps. A log
  therefore names which failure an unconverged forward hit.
- Mixing does not cure a forward that wanders without a sign pattern (the
  χ=14 case of #1060).

## A converged CTM can be a saddle: the hold test

`conv_tol` compares *successive* sweeps, and that cannot tell an attractor
from a saddle: at a saddle successive sweeps agree to 1e-10 while a small
displacement grows every sweep (#1035 measured one on a fermionic D=3 χ=12
state, escaping at ×1.041/sweep to a stable fixed point 1.4e-2 away).
`ctm_tensor_2site` and `ctm_multisite` can therefore run a **hold test**
once the criterion passes (opt-in: pass `hold_sweeps=40`; the default `0` is
off): two copies perturbed by
`hold_perturbation` (relative, default 1e-6; independent deterministic
directions) and the point itself are stepped side by side, and each
displacement is measured in a gauge-invariant metric (per-leg, per-sector
singular values of every environment tensor, blind to χ-bond order and
signs) and renormalised whenever it shrinks 1e-3 — a power iteration, so a
weakly excited unstable direction still surfaces. The point is accepted only
if every direction's fitted growth rate over the last `hold_sweeps // 2`
sweeps (default window at 40) is below 1 — never on early contraction alone;
a fit that still grows is re-tested on later windows up to
`3 * hold_sweeps`, since a stable point can amplify a perturbation for a
while before contracting it. On a saddle the loop keeps iterating from the
perturbed point and walks on to the attractor. If `max_iter` runs out first
— hold steps count toward it, three per hold sweep — it warns that the
environment is not converged; if the criterion passes with fewer than
`3 * hold_sweeps` steps left, it stops, reports the sweeps actually run, and
warns that the point is unverified. A pass means no growth was seen
within the window, not a proof: a weakly excited unstable mode that grows
only slightly faster than slowly decaying stable modes can need more sweeps
than the window to show (e.g. ×1.01 against ×0.99 takes ~230). The metric is
also per tensor, so a mode that rotates one side of a shared χ bond relative
to the other is invisible to it. With the default `hold_sweeps=0` the loop uses successive-sweep
agreement alone, as before.

`ctm_hold_test` runs the same test on any environment, e.g. a seed before it
goes to `optimize_gs_ad(envs_init=...)` or the environment an implicit-AD
forward returned:

```python
from tenax import ctm_hold_test

held = ctm_hold_test({(0, 0): A, (1, 0): B}, {(0, 0): env_A, (1, 0): env_B}, chi=12)
print(held.passed, held.rate)  # rate: fitted per-sweep growth, < 1 = attractor
```

(split-ctmrg)=
## Split-CTMRG

```python
from tenax import CTMConfig, ctm_split, compute_energy_split_ctm

# Split-CTMRG keeps ket/bra layers separate for O(χ³D³) projector cost
# instead of O(χ³D⁶). That is a projector-cost bound, not a peak-memory one:
# measured against the fused path it buys ~1.5x in chi at D=8 and ~2x at D=12
# on one GPU, and nothing at D=10 (#825).
config = CTMConfig(chi=20, max_iter=100, chi_I=10)
env = ctm_split(A, config)
E = compute_energy_split_ctm(A, env, gate, d=2)
```

Split-CTMRG keeps the ket/bra layers as separate CTM environment tensors, for
O(χ³D³) *projector* cost instead of O(χ³D⁶), and works with both
`DenseTensor` and `SymmetricTensor` via the Tensor protocol (Naumann et al.,
arXiv:2502.10298). This is a projector **cost** bound, not a peak-memory one:
the realized `value_and_grad` peak is 1.02–2.7× below the fused path depending
on χ, and converges to ~1× at the memory ceiling (#825).

Energy entry points: `compute_energy_split_ctm_tensor_2site` and
`compute_energy_split_ctm_tensor_multisite` cover 2-site checkerboard and
multisite unit cells (kagome PESS, etc.) at large D; ``ctm_split_tensor`` /
``compute_energy_split_ctm_tensor`` are the single-site Tensor-protocol
versions (see {doc}`ipeps`).

### Split-CTM AD ground-state optimization

`optimize_gs_ad` with `CTMConfig(fuse_virtual_legs=False)` drives the
single-site optimizer (`unit_cell="1x1"`) **and** the 2-site checkerboard
optimizer (`unit_cell="2site"`) through the split χ²·D⁴ forward instead of the
fused χ²·D⁶ double layer.

- **Recipe.** Both run on the default `gs_recipe="2x2"` (single-site since
  #746); `gs_recipe="1x1"` remains reachable but collapses the environment to
  rank-1 corners and is bisection-only — see #726.
- **Gradient.** Implicit AD via a Γ-gauge-fixed fixed-point `custom_vjp`
  (Neumann backward; the 2-site case differentiates the coupled
  `(env_A, env_B)` fixed point), with the line-search probe, warm-start, and
  final environment all routed through the same split forward (returns
  `SplitCTMTensorEnv`). The implicit gradient matches the trusted explicit-AD
  gradient to machine precision in the non-degenerate regime (~1e-15; the
  SU(2)-symmetric Heisenberg point carries a degenerate-SV SVD-backward floor
  on the explicit reference).
- **Scope.** The split CTM is dense-bosonic and experimental (frozen by
  design): `DenseTensor` and bosonic `SymmetricTensor` (U(1)/Z_n) both run on
  the eager forward and on single-site (`unit_cell="1x1"`) AD, but a *traced*
  2-site `SymmetricTensor` multisite sweep is refused rather than fixed — use
  `fuse_virtual_legs=True` for symmetric multisite AD (#1048). Fermionic input
  is refused outright (#1035).
- **Fixed χ.** The χ-changing knobs are rejected on this path.
- **Memory.** The win over fused is a large-D effect (D≳16). Measured at
  `recipe="2x2"` on one A100-80GB it reaches χ=96/48/32 at D=8/10/12 against
  the fused path's χ=64/48/16, i.e. 1.5× / 1.0× / 2.0× in χ, and the per-cell
  peak advantage shrinks from 2.66× at χ=16 to 1.02× at the ceiling (#825).

References: Naumann et al., arXiv:2502.10298.

## API

**1-site:**

- ``ctm_tensor(A, chi, ...)`` — general 4-move CTM.
- ``ctm_tensor_c4v(A, chi, ...)`` — C4v single-move CTM.
- ``compute_energy_ctm_tensor(A, env, H, d)`` — energy from CTM environment.

**2-site:**

- ``ctm_tensor_2site(A, B, chi, ...)`` — checkerboard CTM.
- ``compute_energy_ctm_tensor_2site(A, B, env_A, env_B, H, d)`` — 2-site energy.

**Multi-site:**

- ``ctm_multisite(site_tensors, lattice, chi, ...)`` — general unit cell.

## Implementation Details

The ``CTMTensorEnv`` is a ``NamedTuple`` with 8 fields:

```
C1(c1_d, c1_r)  T1(t1_l, u2, t1_r)  C2(c2_l, c2_d)
T4(t4_d, l2, t4_u)      a(u2,d2,l2,r2)      T2(t2_u, r2, t2_d)
C4(c4_r, c4_u)  T3(t3_r, d2, t3_l)  C3(c3_u, c3_l)
```

Edges carry the fused double-layer (dimension D²). Corners are χ × χ.

## References

- Nishino & Okunishi, *J. Phys. Soc. Jpn.* **65**, 891 (1996) -- CTM method.
- Corboz et al., *Phys. Rev. B* **90**, 195114 (2014) -- CTM for iPEPS.
- Francuz et al., *Phys. Rev. Research* **7**, 013237 (2025) --
  Stable AD of CTM (sigma gauge, custom SVD VJP, implicit differentiation).
