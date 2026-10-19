# Fermionic iPEPS (fPEPS)

Fermionic iPEPS extends the iPEPS algorithm to systems with fermionic
statistics using graded tensors that automatically handle anticommutation
(Koszul signs).

## Background

Standard tensor networks assume bosonic statistics — contracting two tensors
does not depend on the order of legs. For fermions, exchanging two legs picks
up a minus sign when both carry odd parity. Tenax's ``SymmetricTensor`` with
``FermionParity`` or ``FermionicU1`` symmetry gets these signs from the graded
tensor algebra rather than from hand-placed swap gates.

(fermionic-sign-convention)=
### Sign convention

The convention (#555, #994): the graded `transpose` and the matricization
inside `svd`/`qr`/`eigh` carry Koszul signs; label-based `contract` is
sign-free (correct for the planar networks every tenax algorithm uses), and
`permute_legs` reorders leg *storage* with no sign — it, not `transpose`, is
how code restores an axis order after `contract`. Non-planar diagrams are the
exception and need the explicit twist ({ref}`the-twist`); swap gates are
described in {doc}`/guide/symmetry`.

Key properties:

- **Graded tensor formalism**: Koszul signs come from the graded
  ``transpose`` and the decompositions' matricization (see the sign
  convention above). No explicit Jordan-Wigner strings needed.
- **Spinless fermion gate** (``spinless_fermion_gate``): pre-built 2-site
  Hamiltonian for the t-V model with ``FermionParity`` symmetry.
- **Simple update**: imaginary-time evolution on the square lattice, identical
  to the bosonic path but using ``SymmetricTensor`` throughout.
- **CTM environment**: uses ``ctm_tensor`` (with automatic densify workaround
  for fermionic symmetries in the general 4-move path) or ``ctm_tensor_c4v``
  (single-move, no workaround needed).

## Configuration

```python
from tenax import FPEPSConfig

config = FPEPSConfig(
    D=2,                    # virtual bond dimension
    t=1.0,                  # hopping amplitude
    V=0.5,                  # nearest-neighbor interaction
    dt=0.05,                # imaginary time step
    num_imaginary_steps=200,
    ctm_chi=16,             # CTM bond dimension
    ctm_max_iter=50,
    ctm_conv_tol=1e-8,
)
```

## Example — spinless fermion t-V model

```python
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate, sublattice_gap
import jax

config = FPEPSConfig(D=2, t=1.0, V=4.0, dt=0.05, num_imaginary_steps=200,
                     ctm_chi=8, ctm_max_iter=60, ctm_conv_tol=1e-8)
H = spinless_fermion_gate(config)
energy, (A, B), (env_A, env_B) = fpeps(H, config, key=jax.random.PRNGKey(0))
print(f"E/site = {energy:.8f}")
print(f"CDW gap = {sublattice_gap(A, B, env_A, env_B):.4f}")
```

**The state and environment are pairs** (#878). The t-V ground state at finite
``V`` is a checkerboard charge-density wave, which no single tensor can
represent, and the 1-site ansatz that preceded this made ``A`` both ends of
every bond — its update kept only ``U`` from each SVD, so ``A`` received the
left/top half of every gate and never the right/bottom half, and the state
collapsed to a product state regardless of ``dt``, and then to exactly
``0.0``.

``sublattice_gap(A, B, env_A, env_B)`` measures **charge order** between the two
sublattices: the trace distance between their one-site reduced density matrices,
traced out of the two-site RDM the energy already uses. For spinless fermions
``FermionParity`` forbids the off-diagonal entries, so this is exactly
``|<n_A> - <n_B>|`` — the CDW order parameter, ~0 at ``V=0`` and 1 for the fully
polarised occupied/empty checkerboard.

```{warning}
It is a **one-body** probe, and a zero does not mean one tensor would do. A
``0`` says the two *one-site* RDMs coincide; it says nothing about two-site
structure. A columnar-dimer or bond-ordered state has identical on-site
densities on both sublattices, reads ``0``, and is still genuinely two-site. A
nonzero value is positive evidence of charge order; the converse does not hold.
To rule out two-site order in general, compare a two-site observable — e.g. the
horizontal against the vertical bond energy of the pair.
```

A value above 1 means the environment's RDM is not PSD (#854) — measured up to
1.07 at χ=4 on a deliberately under-converged environment, against a few `1e-4`
once the CTM has settled. It is not clipped: the excess tells you χ or the sweep
count is too small, and clipping would hide that inside a plausible-looking 1.0.

Do **not** compare the two sublattices with `||A - B||`, or with any fingerprint
built from `T T†` on a virtual leg. A simple-update tensor is defined only up to
a bond gauge `T -> G T`, under which that matrix goes to `G M G†` — its spectrum
moves unless `G` is unitary, and simple update's gauge is not. Measured on a
provably uniform pair, `||A - B||` sits at ~1.7. A reduced density matrix has no
such freedom.

The returned pair is in physical (CTM-contractable) form, which is also the form
`initial_tensor` takes for a warm restart:

```python
energy, pair, envs = fpeps(H, config, initial_tensor=pair)   # continues
energy, pair, envs = fpeps(H, config, initial_tensor=A)      # both sites from A
```

A restart is not a continuation. The sweep always begins from
`BondWeights.ones`, so its first cycle treats the outer legs as unweighted while
the tensors you hand back already carry `sqrt(λ)`. `fpeps(N)` is therefore not
`fpeps(N/2)` fed back for another `N/2` — use a restart to continue annealing,
not to reproduce a longer single run.

### Caveats

Two standing caveats. Simple update on this path is **seed-dependent**: over
seeds 0–4 at 600 steps, the fraction whose bond spectrum survives is 4/5 at D=2,
2/5 at D=3, 4/5 at D=4 and 4/5 at D=6 — every bond dimension has both surviving
and dying seeds, so check the result rather than assuming it (#869 is the same
basin behaviour on the bosonic path). And the **absolute energy is not
certified** (#392): at the default `FPEPSConfig.mu = 0.0`, `H` carries no
chemical potential, so both the empty state and the fully polarised
checkerboard are `E = 0` eigenstates, and the sweep is observed to settle on
them — measured at 200 steps, D=2, `E ≈ -6e-05` at `V=0` where the half-filled
answer is ≈ `-1.6t`. `sublattice_gap` tells you *which* state you landed on;
it does not tell you it is the ground state.

(fpeps-chemical-potential)=
### Chemical potential

`FPEPSConfig.mu` adds a chemical potential to `H`, distributed over the bonds
as `-(mu / 4) * (n_i + n_j)` per bond (4 bonds per square-lattice site, so the
per-site total is `-mu * n_i`). `mu = 2 * V` is the particle-hole-symmetric
half-filling point of the t-V model, where the CDW energy anchor
`E = -V` per site holds:

```python
config = FPEPSConfig(D=2, t=1.0, V=1.0, mu=2.0, dt=0.05, num_imaginary_steps=200)
H = spinless_fermion_gate(config)  # -t(c†c+h.c.) + V n_i n_j - (mu/4)(n_i+n_j)
```

## API

- ``fpeps(hamiltonian_gate, config, initial_tensor=None, key=None)`` — full
  pipeline: simple update + CTM + energy. Returns
  ``(energy, (A, B), (env_A, env_B))``. ``initial_tensor`` takes either an
  ``(A, B)`` pair — the form this returns, so its own output restarts it — or a
  single tensor, which starts both sublattices from the same place. A restart
  is **not** a continuation: the sweep always begins from ``BondWeights.ones``,
  so its first cycle treats the outer legs as unweighted while the tensors
  handed in already carry ``sqrt(lambda)``. ``fpeps(N)`` is not ``fpeps(N/2)``
  restarted for another ``N/2``.
- ``sublattice_gap(A, B, env_A, env_B)`` — charge-order diagnostic, above; a
  one-body probe, so a zero does not certify that one tensor would suffice.
- ``spinless_fermion_gate(config)`` — build the t-V model gate from an
  ``FPEPSConfig`` (it reads ``t``, ``V`` and ``mu``).
- ``compute_energy_ctm_tensor_2site(..., nan_on_invalid_rdm=..., psd_tol=...)``
  — 2-site energy with the opt-in invalid-RDM gate, below.
- ``su_grow_layout(H, cfg, key=key)`` / ``bond_layout(A, B)`` — grow and read the
  χ-sector layout for a frozen-layout AD seed, below.
- ``FPEPSConfig`` — configuration dataclass.
- ``optimize_fpeps_ad(hamiltonian_gate, A_init, config, fpeps_config=None, *,
  envs_init=None)`` — AD-based ground-state optimization. It is
  ``optimize_gs_ad`` with a fermionic initial state: ``A_init=None`` builds
  ``FermionParity`` tensors from ``fpeps_config`` (one for
  ``unit_cell="1x1"``, an ``(A, B)`` pair for ``"2site"``; a ``Lattice`` needs
  an explicit dict). It returns ``optimize_gs_ad``'s shape for the unit cell:
  ``(A, env, E)`` for 1x1, ``((A, B), (env_A, env_B), E)`` for 2-site.
  ``envs_init`` is the frozen-layout seed from ``su_grow_layout`` (2-site
  only). The default 1x1 cell is a uniform state and cannot hold a CDW; use
  ``"2site"`` for one. ``gs_c4v=True`` and
  ``ctm_ad_mode="root_implicit_symmetric"`` have no fermionic signs; do not
  use them with fermions (#1059).

(fpeps-invalid-rdm)=
## Refusing an energy built from an invalid RDM

`Σ_bonds tr(ρ H)` is bounded by `H`'s spectrum only when every `ρ` is a genuine
density matrix. When a bond RDM is not one — non-finite, trace collapsed, or
badly non-PSD — the CTM energy path returned the number anyway: finite, and
an unphysical lie (#879). The 2-site energy functions (fused
`compute_energy_ctm_tensor_2site`, which `fpeps()` uses, and the split
`compute_energy_split_ctm_tensor_2site`) take an opt-in gate:

```python
from tenax import compute_energy_ctm_tensor_2site

E = compute_energy_ctm_tensor_2site(
    A, B, env_A, env_B, gate, d=2,
    nan_on_invalid_rdm=True,   # default False
    psd_tol=None,              # default None -> RDM_PSD_TOL = 1e-8
)
```

- **`nan_on_invalid_rdm`** (default `False`) — re-check every bond RDM with
  `check_rdm(strict=True)` and return `NaN` if any bond fails.
- **`psd_tol`** (default `None` → `RDM_PSD_TOL = 1e-8`) — negativity tolerance
  for the **PSD arm only**, relative to the spectral radius. The non-finite and
  trace-collapse arms keep their own tolerances, so a collapse is refused at
  *any* `psd_tol`.

The gate is **eager-only**: it is skipped on tracers, so `jit`, `grad` and every
optimizer path are bit-for-bit unchanged, and leaving it off is the identity on
existing callers.

`fpeps()` turns the gate on and loosens the PSD arm to `1e-2`. A low-χ CTM leaves
~1e-3 relative negativity that is convergence noise rather than a collapse, so
those runs still return a number and only *gross* non-PSD is refused (the #853
case sits ~80× higher, its smallest eigenvalue 0.8 of the spectral radius below
zero). **This changes `fpeps()` behaviour**: on such a state it now returns
`NaN` rather than a finite unphysical energy.

(fpeps-frozen-layout)=
## Seeding the 2-site AD optimizer with a frozen environment layout

`optimize_gs_ad`'s 2-site implicit-AD path re-derives its CTM environment
from a cold tiled/identity seed unless told otherwise. On a fermionic
(block-sparse `SymmetricTensor`) state, a cold-started *traced* CTM can
settle on a different χ-sector layout than an eager `ctm_tensor_2site` run
on the same tensors finds — a different point to differentiate through, and
possibly extra retraces. `envs_init` seeds the first forward CTM (and the
warm-start refresh between steps) with an already-converged environment;
under tracing the CTM keeps the χ-sector layout it is given (#1035), so the
layout stays fixed for the run:

```python
import jax
from tenax import (
    CTMConfig,
    FPEPSConfig,
    ctm_tensor_2site,
    iPEPSConfig,
    optimize_gs_ad,
    spinless_fermion_gate,
    su_grow_layout,
)

cfg = FPEPSConfig(D=2, t=1.0, V=0.0, dt=0.05)
H = spinless_fermion_gate(cfg)
su = su_grow_layout(H, cfg, key=jax.random.PRNGKey(4))  # eager SU grows the sectors
eA, eB = ctm_tensor_2site(su.A, su.B, 8, max_iter=300, conv_tol=1e-10,
                          hold_sweeps=40)  # eager CTM picks the chi layout; hold: not a saddle

(A, B), (env_A, env_B), E = optimize_gs_ad(              # traced AD keeps both layouts
    H, (su.A, su.B),
    iPEPSConfig(max_bond_dim=2, unit_cell="2site", su_init=False,
                gs_implicit_ad=True, gs_num_steps=5,
                ctm=CTMConfig(chi=8, max_iter=50, conv_tol=1e-9)),
    envs_init={(0, 0): eA, (1, 0): eB},
)
```

The eager CTM's hold test (opt-in via `hold_sweeps=40`, see
{ref}`ctm-hold-test`) is what makes this seed an attractor rather than a point the
CTM merely paused at; budget `max_iter` for it — a hold spends `3 *
hold_sweeps` extra steps at an attractor (up to `9 * hold_sweeps` to reject
a saddle), and walking off a saddle takes as many sweeps as it takes: about
1100 in total on the #1035 D=3 χ=12 V=1 state (use `max_iter=1500`), where
`max_iter=260` returned the saddle before the hold existed.

`su_grow_layout` tracks the sector split with
`bond_layout(A: SymmetricTensor, B: SymmetricTensor) -> tuple[tuple[int, int], ...]`
internally; call it directly on any checkerboard pair to read the same
`(n_even, n_odd)` count per bond leg (`u, d, l, r` of `A` then `B`) that
`su.layout` above already reports.

`envs_init` is refused (`ValueError`) in eleven cases: `unit_cell` other than
`"2site"`; `gs_c4v=True` (the C4v path rebuilds the sites as `DenseTensor`);
the root-implicit AD path (`ctm_ad_mode="root_implicit"`/
`"root_implicit_symmetric"`); the split CTM (`fuse_virtual_legs=False`);
`chi_auto_bump`; `ctmrg_heuristic_increase_chi`; a `chi_ramp`; a χ schedule
(`gs_chi_schedule_steps`); a chi that does not match `CTMConfig.chi`;
keys other than `{(0, 0), (1, 0)}`; and edge D² legs whose charges do not
match the double layers of the tensors passed in (a seed built for a
different virtual charge layout) — each one changes, bypasses, or is
inconsistent with the layout `envs_init` is meant to freeze. The
optimizer's own *final* returned environment is always a fresh, cold CTM
evaluation on the optimized tensors (issue #899) — `envs_init` fixes the
layout used *during* optimization, not this last re-check.

## Performance: AD compile cost on symmetric tensors

The AD path differentiates through the CTM fixed point. The backward is
traced and XLA-compiled **once per optimizer run** (then reused across
steps), but for block-sparse ``SymmetricTensor`` site tensors that single
trace+compile scales with the **number of charge blocks** — hence with the
symmetry's sector count and with ``D``/``chi``. This is *not* specific to
fermions: any charge-conserving iPEPS AD (U(1), Zₙ, FermionParity) is
affected; fermionic tensors simply always carry non-trivial parity sectors.

Practical consequence: the **first** gradient step can take from seconds (a
single block) to many minutes (large ``D``/``chi`` with many blocks). With
``gs_verbose=True`` a one-time notice is printed before step 1 so the wait is
not mistaken for a hang. Subsequent steps reuse the compiled backward and are
fast. If the first step seems stuck, it is almost always compiling, not
deadlocked. The underlying compile-time scaling — and the plan to fix it via
sweep-level block batching — is tracked in
[issue #566](https://github.com/tenax-lab/tenax/issues/566).

## References

- Corboz et al., *Phys. Rev. B* **81**, 165104 (2010) — fermionic PEPS formalism.
- Barthel et al., *Phys. Rev. A* **80**, 042333 (2009) — graded tensor networks.
