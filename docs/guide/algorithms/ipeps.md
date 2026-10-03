# iPEPS

Infinite Projected Entangled Pair States (iPEPS) is a variational ansatz for
2D quantum lattice models. Tenax implements the **simple update** for
optimisation and the **Corner Transfer Matrix (CTM)** method for computing
observables.

## Background

An iPEPS represents a 2D quantum state as a tensor network where each site has
a local tensor $A[u,d,l,r,s]$ with four virtual bonds and one physical index.
For translationally invariant states, a single-site unit cell suffices.

### Simple update

Fast imaginary time evolution:

1. For each nearest-neighbour bond, apply $\exp(-\delta\tau\, H_{\text{bond}})$.
2. SVD to restore tensor-product form; truncate to bond dimension $D$.
3. Update diagonal $\lambda$ matrices that approximate the bond environment.

### CTM environment

The Corner Transfer Matrix method approximates the infinite environment of a
PEPS site using 8 tensors (4 corners + 4 edges):

```
C1 --- T1 --- C2
|             |
T4    [A]    T2
|             |
C4 --- T3 --- C3
```

CTM iteratively absorbs rows and columns until the corner singular values
converge. The projectors used to truncate the enlarged corners are built
from an eigendecomposition (`eigh`) of the half-row/half-column density
matrices. For AD-based optimization, use `truncated_svd_ad` instead
(see {doc}`ad_excitations`).

## Configuration

```python
from tenax import iPEPSConfig, CTMConfig

ctm_config = CTMConfig(
    chi=20,              # CTM environment bond dimension
    max_iter=100,        # maximum CTM iterations
    conv_tol=1e-8,       # convergence tolerance on corner singular values
    renormalize=True,
    forward_gauge="auto",   # "auto" (default), "phase", "bond_phase", "qr", "sigma", or "none"
)

config = iPEPSConfig(
    max_bond_dim=2,            # PEPS virtual bond dimension D
    num_imaginary_steps=100,   # imaginary time evolution steps
    dt=0.05,                   # time step size
    ctm=ctm_config,
    gate_order="sequential",
)
```

### Forward gauge

The ``forward_gauge`` option in ``CTMConfig`` controls how gauge ambiguity is
fixed after each CTM sweep during the forward pass. Five modes are supported,
plus the ``"auto"`` default that picks one per path:

| Value | Description |
|-------|-------------|
| ``"auto"`` (default) | Resolved per path: ``"bond_phase"`` on the fused implicit-AD path (no ``chi_ramp``, ``ctm_ad_mode=None``), ``"phase"`` everywhere else. |
| ``"phase"`` | variPEPS-style Frobenius normalization + phase fixing. Cheapest gauge fix. What ``"auto"`` resolves to on explicit AD, where ``optimize_gs_ad``'s explicit energy applies no forward gauge, so it has no effect (#1074). |
| ``"bond_phase"`` | ``"phase"`` plus a per-chi-index sign/phase aligned to the previous environment (#841); removes the per-index Z2 sign 2-cycle the SVD projectors re-draw each sweep. Implicit AD only; what ``"auto"`` runs there. |
| ``"qr"`` | Legacy QR decomposition on each corner with sign-fixed diagonal. Fast and stable for simple update and forward-only CTM. |
| ``"sigma"`` | Transfer-matrix eigenvector alignment via power iteration. Required for element-wise convergence at large chi (1-site path). |
| ``"none"`` | No gauge fix. Diagnostic / benchmark mode only. |

**Forward gauge default**: ``forward_gauge`` defaults to ``"auto"``, which
runs ``"bond_phase"`` on the implicit-AD path and resolves to ``"phase"`` on
the explicit path, where ``optimize_gs_ad``'s explicit energy applies no
forward gauge (#1074) — the implicit-AD
path in fact *requires* one of those two and validates it
(``projector_method`` in ``("svd", "qr")``, ``forward_gauge`` in
``("phase", "bond_phase")``, ``ctm_conv_method="elementwise"``). There is **no
silent gauge promotion**: if you set ``forward_gauge="phase"``, ``"sigma"`` or
``"none"`` explicitly, that choice is passed through as-is (the implicit path
then refuses ``"sigma"`` and ``"none"``; the explicit-AD energy applies none
of them, #1074).

See {doc}`ipeps_ad_paths` for the complete post-PR-#291 recommended
configuration, benchmark results, and the split between the explicit-AD
and implicit-diff paths.

### Choosing `dt`

Larger time steps (`dt=0.1`–`0.3`) converge faster but can overshoot;
smaller steps (`dt=0.01`) are safer but need more iterations. A good
strategy is to start with `dt=0.1` for quick exploration and reduce it
for final production runs. The 2-site unit cell often benefits from
larger `dt` because the two independent tensors converge more slowly.

## Example -- 2D Heisenberg model

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, ipeps

# Heisenberg gate: H = Sz Sz + 0.5 (S+ S- + S- S+)
Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)
Sp = jnp.array([[0, 1], [0, 0]], dtype=jnp.float32)
Sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.float32)
I2 = jnp.eye(2, dtype=jnp.float32)

H_bond = (
    jnp.kron(Sz, Sz)
    + 0.5 * jnp.kron(Sp, Sm)
    + 0.5 * jnp.kron(Sm, Sp)
).reshape(2, 2, 2, 2)

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.01,
    ctm=CTMConfig(chi=10, max_iter=50),
)

energy, peps, env = ipeps(H_bond, initial_peps=None, config=config)
print(f"Energy per site: {energy:.6f}")
```

## Result

`ipeps()` returns a 3-tuple:

| Element | Type | Description |
|---------|------|-------------|
| `energy` | `float` | Energy per site |
| `peps` | `TensorNetwork` | Optimised PEPS (1x1 unit cell) |
| `env` | `CTMEnvironment` | Converged CTM environment tensors |

## CTMEnvironment

The `CTMEnvironment` named tuple contains the 8 environment tensors:

- **Corners** (`C1`, `C2`, `C3`, `C4`): shape `(chi, chi)`
- **Edges** (`T1`, `T2`, `T3`, `T4`): shape `(chi, D^2, chi)`

## Using CTM standalone

The `ctm()` function can be called independently to compute the
environment for an existing PEPS tensor:

```python
from tenax import ctm, CTMConfig

env = ctm(A_tensor, CTMConfig(chi=20, max_iter=100))
```

## 2-site checkerboard unit cell

A single-site unit cell cannot capture antiferromagnetic (Néel) order
because both sublattices share the same tensor. Setting
`unit_cell="2site"` in `iPEPSConfig` uses a 2-site checkerboard unit cell
with independent tensors $A$ (sublattice 0) and $B$ (sublattice 1).

On the checkerboard every neighbour of $A$ is $B$ and vice versa, which
is the minimal unit cell for Néel-ordered states.

### Simple update on the checkerboard

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, ipeps

# Build a 2-site Heisenberg gate
Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) + 0.5 * (
    jnp.einsum("ij,kl->ikjl", Sp, Sm) + jnp.einsum("ij,kl->ikjl", Sm, Sp)
)

# 2-site checkerboard iPEPS — captures Neel order
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, (env_A, env_B) = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")  # ~ -0.63
```

The checkerboard has **four** inequivalent bonds — `A.r<->B.l`, `B.r<->A.l`,
`A.d<->B.u`, `B.d<->A.u` — and by default each pair shares one Schmidt
spectrum. On a translation-invariant Hamiltonian that is exact at the fixed
point (the paired bonds agree to ~1e-6), and it is the more robust choice: it
constrains the two horizontal bonds to be equal, which projects out a
dimerising direction that four free bonds can follow. Measured at D=3 from a
random start, four free bonds converged to a dimerised state on 3 of 8 seeds
against 1 of 8 when shared.

Give each bond its own spectrum when the *state* may genuinely break the
AB↔BA symmetry — a spontaneously dimerised or valence-bond phase, where two
spectra cannot represent the answer — and prefer a physical initial state with
it:

```python
config = iPEPSConfig(..., su_independent_bond_lambdas=True)
```

This does **not** make the bonds inequivalent in the *Hamiltonian*: `ipeps()`
takes a single `hamiltonian_gate` and applies it to all four bonds, so an
anisotropic model (`Jx != Jy`) cannot be expressed today regardless of this
flag — setting it would silently evolve the uniform model. Per-bond
gates are #883.

The energy `ipeps()` reports comes from the legacy 2-site CTM, which does not
converge on a genuinely entangled state — it sits ~0.02 above the truth. For an
accurate number, measure the returned state with `ctm_tensor(recipe="2x2")`
(D=2 gives −0.65933, χ-converged).

When you want only the simple-update state — as a warm start or fixture — skip
that measurement entirely:

```python
_, (A, B), _ = ipeps(gate, None, config, compute_energy=False)
# returns (None, (A, B), None): no CTM is run, no energy is computed
```

Simple update itself was fixed in #667; if
you have results from before that, note it converged to the product state and
that *smaller* `dt` made it worse — see the changelog.

See `examples/heisenberg_ipeps_su.py` for 1-site and 2-site unit cell examples.

(bp-gauge)=
### Belief-propagation gauge (correct bond weights)

Simple update stores each bond's Schmidt spectrum straight from the SVD that
produced it. A *non-unitary* gate on a neighbouring bond changes this bond's
Schmidt values, and they are never recomputed, so the stored weights drift away
from the spectra they are taken to be. `bp_gauge_checkerboard` re-derives all
four of them by solving the belief-propagation fixed point (bond weights on a
PEPS *are* BP messages) and re-gauges the tensors to match:

```python
from tenax import BondWeights, bp_gauge_checkerboard

# A, B are bare Vidal Gamma tensors; lam_h, lam_v are the weights they carry.
stored = BondWeights(h_AB=lam_h, h_BA=lam_h, v_AB=lam_v, v_BA=lam_v)
A, B, weights, info = bp_gauge_checkerboard(A, B, stored)
print(info.converged, info.iterations)
print(weights.h_AB, weights.h_BA)   # the two horizontal bonds, resolved separately
```

The weights are required, and are not an initial guess: in Vidal form the state
is `... Γ_A λ Γ_B ...`, so `λ` is half of what you are handing over. A fresh
random pair whose bonds really are unweighted passes `BondWeights.ones(D, D)`.

Every step is a gauge transformation, so the physical state is unchanged to
machine precision — only the weights move. Measured on simple update's own
converged D=3 output, the stored spectrum is `[1, 0.16586, 0.01564]` where the
BP-consistent one is `[1, 0.14243, 0.01130]`: 15% off on the second Schmidt
value and ~35% on the tail. Use this before reading `lambda` as a Schmidt
spectrum — entanglement entropy, truncation-error estimates, or the symmetric
gauge handed to a CTM.

This corrects the *weights*, not simple update's dynamics; it does not change
the state `ipeps()` converges to.

### `ctm_2site()` -- standalone 2-site CTM

Compute CTM environments for an existing 2-site iPEPS:

```python
from tenax import ctm_2site, CTMConfig

env_A, env_B = ctm_2site(A, B, CTMConfig(chi=20, max_iter=100))
```

> **Note:** ``ctm_2site()`` is the legacy dense CTM used internally by
> simple update (``ipeps()``).  For AD-based optimization, use
> ``optimize_gs_ad()`` with ``unit_cell="2site"`` — it routes through
> the Tensor-protocol multisite CTM which supports both ``DenseTensor``
> and ``SymmetricTensor``.

| Argument | Type | Description |
|----------|------|-------------|
| `A` | `jax.Array` | Site tensor for sublattice A, shape `(D, D, D, D, d)` |
| `B` | `jax.Array` | Site tensor for sublattice B, shape `(D, D, D, D, d)` |
| `config` | `CTMConfig` | CTM configuration |

Returns a tuple `(env_A, env_B)` of `CTMEnvironment` named tuples.

### `compute_energy_ctm_2site()` -- 2-site energy

Compute the energy per site for a 2-site checkerboard iPEPS given
converged environments:

```python
from tenax import compute_energy_ctm_2site

energy = compute_energy_ctm_2site(A, B, env_A, env_B, H_bond, d=2)
```

The energy includes one horizontal and one vertical bond per site:
$E/\text{site} = E_h + E_v$.

### AD ground-state optimization

`optimize_gs_ad()` uses automatic differentiation through the CTM
fixed-point equation to compute exact gradients of the energy with
respect to the site tensor, then optimises with optax:

```python
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=20, max_iter=100),
    gs_optimizer="adam",
    gs_learning_rate=1e-3,
    gs_num_steps=200,
    gs_verbose=True,      # print optimization progress
    gs_log_interval=10,   # print every 10 AD steps
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

Set ``gs_verbose=False`` (default) to disable console output.

#### Simple update initialization

Starting AD optimization from a random tensor can cause large gradients
and slow convergence.  Setting ``su_init=True`` runs simple update first
(using the ``num_imaginary_steps`` and ``dt`` already in the config) to
produce a physically reasonable starting point:

```python
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.01,
    ctm=CTMConfig(chi=20, max_iter=100),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, A_init=None, config=config)
```

When ``A_init`` is provided explicitly, ``su_init`` is ignored.

#### Optimizer selection

The AD optimizer is chosen via ``gs_optimizer`` in ``iPEPSConfig``:

| Optimizer | Setting | Best for |
|-----------|---------|----------|
| L-BFGS | ``gs_optimizer="lbfgs"`` (default) | Fast convergence near minimum |
| Adam | ``gs_optimizer="adam"`` | Stable convergence, noisy gradients |
| Conjugate gradient | ``gs_optimizer="cg"`` | Memory-efficient alternative to L-BFGS |

L-BFGS and CG run a line search by default (``gs_line_search=None`` resolves
to ``True`` for them): **Hager-Zhang** (``gs_line_search_method="hager_zhang"``,
the default) or Armijo backtracking (``"armijo"``). Each trial step runs a
fresh CTM convergence to evaluate the energy, avoiding stale-environment
artifacts. Metric preconditioning (``gs_metric_precond``, Rader et al.,
arXiv:2511.09546) uses the environment metric as a natural-gradient
preconditioner.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_optimizer="lbfgs",
    gs_num_steps=30,
    gs_line_search_max_steps=8,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, None, config)
```

For Adam, a **cosine learning rate schedule** (lr → lr/10) is automatically
applied when ``gs_num_steps > 20``.

#### Explicit CTM differentiation

Set ``gs_implicit_ad=False`` to backpropagate through unrolled CTM
iterations instead of using implicit differentiation (the default
``gs_implicit_ad=True`` uses implicit diff). The forward pass runs
``gs_explicit_ad_warmup`` CTM sweeps without gradient tracking, then
``gs_explicit_ad_steps`` sweeps with full backpropagation.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50, projector_method="qr"),
    gs_implicit_ad=False,      # opt into explicit AD (default is implicit)
    gs_explicit_ad_steps=20,   # CTM steps with gradient tracking
    gs_explicit_ad_warmup=3,   # warmup steps (no gradient)
    gs_projector_method="qr",  # QR projectors scale cleanly to chi >= 16
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_num_steps=50,
)
A_opt, env, E_gs = optimize_gs_ad(H_bond, None, config)
```

Explicit AD is the **recommended** AD path on the 1-site C4v workflow
(``gs_c4v=True``). Each CTM sweep is a single move, the unrolled graph stays
manageable, and the backward pass avoids the implicit-diff linear solve
entirely.

```{note}
``forward_gauge`` defaults to ``"auto"``, which resolves to ``"phase"`` on
this explicit path. Under ``optimize_gs_ad`` the explicit energy
(``ctm_energy_explicit``) applies no forward gauge, so the setting has no
effect here; only the legacy ``ad_utils`` entry points apply it. The
phase-vs-sigma benchmark in {doc}`ipeps_ad_paths` predates this routing
(#1074).
```

#### CTM convergence tolerance schedule

``iPEPSConfig.gs_ctm_conv_tol_schedule`` ramps the CTM convergence tolerance
from loose to tight across the AD optimization. It accepts a list of
``(step_fraction, conv_tol)`` pairs: at each AD step the optimizer looks up
the tolerance corresponding to the current ``step_index / gs_num_steps``
fraction and rebuilds the CTM config accordingly.

```python
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=80, conv_tol=1e-7),
    gs_num_steps=50,
    gs_ctm_conv_tol_schedule=[(0.0, 1e-5), (0.5, 1e-6), (0.8, 1e-7)],
)
```

This is an advanced tuning knob — leaving it at ``None`` (the default) uses
``ctm.conv_tol`` throughout, which is fine for most runs.

#### Chi-ramping schedule

Starting AD optimization at a large chi can be slow and unstable.
``optimize_gs_ad_chi_schedule`` runs optimization at progressively
increasing chi values, using the converged tensor from each level to
warm-start the next:

```python
from tenax import optimize_gs_ad_chi_schedule, iPEPSConfig, CTMConfig

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=8, projector_method="qr"),
    gs_projector_method="qr",
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_c4v=True,
    su_init=True,
)

# (chi, num_steps) pairs — chi and gs_num_steps are overridden per stage
A_opt, env, E_gs = optimize_gs_ad_chi_schedule(
    H_bond, None, config, [(8, 30), (16, 20)]
)
```

This follows the approach of Zhang, Yang & Corboz (arXiv:2505.00494).
The base ``config`` provides all other settings (optimizer, line search,
metric preconditioning, etc.); only ``chi`` and ``gs_num_steps`` change
per stage.

#### Backward method selection

The backward pass for CTM implicit differentiation (``gs_implicit_ad=True``)
has two options:

| Method | Setting | Description |
|--------|---------|-------------|
| Iterative VJP | ``ad_backward_method="vjp"`` (default) | Neumann series accumulation of VJP (YASTN-style). The regression-covered backward for the implicit path. |
| GMRES | ``ad_backward_method="gmres"`` | Direct linear solve of ``(I - J^T) λ = g``. **Experimental / documented unstable** — the GMRES backward is currently tracked as an open gap and its regression test is marked ``xfail`` (see issue #292). |

**To avoid the implicit-diff linear solve entirely**: set
``gs_implicit_ad=False`` (explicit AD is opt-in; the default
``gs_implicit_ad=True`` uses implicit diff). Explicit AD does not use the
``(I - J^T)`` solve at all and is the fastest path on the 1-site C4v
workflow. If you use the implicit path, prefer ``ad_backward_method="vjp"``
(the default) until the GMRES backward is stabilized.

```python
# Explicit-AD configuration — explicit AD + QR projectors (forward gauge not applied, #1074)
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=100, projector_method="qr"),
    gs_implicit_ad=False,  # opt into explicit AD (default is implicit)
    gs_projector_method="qr",
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_c4v=True,
    gs_num_steps=100,
    su_init=True,
)
```

For AD-based excitation spectra on top of an optimised iPEPS, see
{doc}`ad_excitations`. For the full benchmarked recommendation (including
when to reach for ``forward_gauge="sigma"``, ``forward_gauge="none"``, or
the ``gs_ctm_conv_tol_schedule`` knob) see {doc}`ipeps_ad_paths`.

## Model gates

Pre-built 2-site Hamiltonian tensors:

- `heisenberg_gate` — dense `DenseTensor` with trivial charges.
- `heisenberg_gate_u1sz` — U(1)-Sz block-sparse `SymmetricTensor` with charges
  `[+1, −1]` for spin-↑/↓.
- `xxz_gate` — XXZ anisotropy.
- `spinless_fermion_gate` — fPEPS hopping + interaction + chemical potential
  (`FPEPSConfig.mu`), with `FermionParity` symmetry; see {doc}`fpeps`.

For honeycomb and kagome lattices see {doc}`honeycomb_kagome`.

## Split-CTMRG with Tensor protocol

The `ctm_split_tensor()` function provides a polymorphic split-CTMRG that
works with both `DenseTensor` and `SymmetricTensor` iPEPS site tensors.
It uses `bar()` (conjugate + flip flows, no charge dual) for the bra layer
instead of `dagger()`, which ensures correct physical-trace block matching
for nontrivial U(1) or fermionic charges.

```python
from tenax import ctm_split_tensor, compute_energy_split_ctm_tensor

env = ctm_split_tensor(A, chi=20, max_iter=100, chi_I=10)
E = compute_energy_split_ctm_tensor(A, env, H_bond, d=2)
```

`A` can be either a `DenseTensor` or `SymmetricTensor` with 5 legs
`(u, d, l, r, phys)`. For the 2-site/multisite entry points, the
projector-cost vs memory trade-off and split-CTM AD, see {ref}`split-ctmrg`.

## Fermionic iPEPS (fPEPS)

Tenax supports fermionic PEPS using `SymmetricTensor` with `FermionParity`
symmetry. All contractions and decompositions automatically handle Koszul
signs (fermionic anticommutation).

### Spinless fermion example

```python
import jax
from tenax import FPEPSConfig, spinless_fermion_gate, fpeps, sublattice_gap

config = FPEPSConfig(D=2, ctm_chi=8, num_imaginary_steps=200, dt=0.05, V=4.0)
gate = spinless_fermion_gate(config)
energy, (A, B), (env_A, env_B) = fpeps(gate, config, key=jax.random.PRNGKey(0))
print(f"Energy per site: {energy:.6f}")
print(f"CDW gap: {sublattice_gap(A, B, env_A, env_B):.4f}")
```

The `spinless_fermion_gate()` builds $H = -t \sum (c^\dagger_i c_j + \text{h.c.}) + V \sum n_i n_j$
as a `SymmetricTensor` with `FermionParity` charges. The simple update uses
`contract()` (sign-free) and `svd()`, whose matricization carries the
Koszul signs; see {ref}`fermionic-sign-convention`.

The state and environment are **pairs** (#878): the t-V ground state at finite
`V` is a checkerboard charge-density wave, which is inherently two-site.
`sublattice_gap` measures that charge order — it is a one-body probe, so a
nonzero value is evidence of a CDW but a zero does *not* certify that a single
tensor would suffice. See [Fermionic iPEPS (fPEPS)](fpeps.md) for the full
statement, the warm-restart form, and the two standing caveats (seed
dependence, and #392's uncertified energy).
