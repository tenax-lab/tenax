# Honeycomb and Kagome iPEPS

Tenax has a native CTM for honeycomb iPEPS and a differentiable iPESS pipeline
for the kagome lattice. Both build on the square-lattice machinery described in
{doc}`ipeps`, {doc}`ctm` and {doc}`ipeps_ad_paths`.

## Honeycomb iPEPS CTM (native rank-4)

Native rank-4 CTMRG for honeycomb iPEPS — six corners, three edge
directions, two sublattices — without the dummy-bond brick-wall hack.
Custom `jax.custom_vjp` forward with a JIT-fused GMRES backward. It replaces the
dummy-bond brick-wall workaround; the public entry
`honeycomb_ctm_energy_implicit` uses the Corboz biorthogonal projector + a
per-column phase fix by default, and takes a configurable `energy_fn` hook for
kagome iPESS triangle energies. References: Lukin & Sotnikov, PRB 107, 054424
(2023) for the 6-corner CTMRG and the bipartite extension in PRE 109, 045305
(2024) §II.C.

```python
import jax
import jax.numpy as jnp
import numpy as np
from tenax import (
    HONEYCOMB_DIRECTIONS,
    honeycomb_ctm_energy_implicit,
    honeycomb_ctm_run,
)
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor


def _make_site(D=2, d=2, key=jax.random.PRNGKey(0)):
    sym = U1Symmetry()
    virt = np.zeros(D, dtype=np.int32)
    phys = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e0"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e1"),
        TensorIndex.from_charges(sym, virt.copy(), FlowDirection.OUT, label="e2"),
        TensorIndex.from_charges(sym, phys.copy(), FlowDirection.IN, label="phys"),
    )
    re = jax.random.normal(key, (D, D, D, d))
    im = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d))
    return DenseTensor((re + 1j * im).astype(jnp.complex128), indices)


# Spin-1/2 Heisenberg bond operator (4×4)
sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_bond = jnp.asarray(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))

# Honeycomb iPEPS uses two rank-4 sites at coords (0,0) and (1,0); legs
# (e0, e1, e2, phys). All virtuals OUT, phys IN.
A = _make_site(D=2, d=2, key=jax.random.PRNGKey(0))
B = _make_site(D=2, d=2, key=jax.random.PRNGKey(1))
sites = {(0, 0): A, (1, 0): B}

# Forward only: returns the converged per-sublattice env dict + info.
envs, info = honeycomb_ctm_run(
    sites, chi=8, max_iter=80, conv_tol=1e-8,
    projector_method="biorthogonal",  # default; eigh/svd are A=B opt-ins
    forward_gauge="phase",            # default; sigma reserved for A=B opt-in
)

# Implicit-AD energy: takes jax.grad through the CTM fixed point via
# JIT-fused GMRES on (I - dF/denv) lambda = dE/denv.
energy = honeycomb_ctm_energy_implicit(
    sites, H_bond, chi=8, max_iter=80, conv_tol=1e-8,
)
grad_fn = jax.grad(
    lambda Ad: honeycomb_ctm_energy_implicit(
        {(0, 0): DenseTensor(Ad, A.indices), (1, 0): B},
        H_bond, chi=8, max_iter=40,
    )
)
gA = grad_fn(A.todense())
```

The default energy is the 3-edge nearest-neighbor bond sum
`Σ_α Tr(ρ_α · H_bond)`. Pass `energy_fn=compute_honeycomb_triangle_energy`
for the kagome iPESS use case where each site is a 3-spin triangle and
the Hamiltonian is the intra-triangle 3-spin operator.

## Kagome iPESS with AD

Differentiable iPESS pipeline for kagome XXZ ground states (Liao et al.,
PRX 9, 031041, 2019). Two simplex tensors `T_u`, `T_d` and three site
tensors `R_a`, `R_b`, `R_c` define the variational state; triangle
simple update gives the SU warm start, then L-BFGS through the exact
single-supersite CTM (`loss_builder="exact"`, the #991 blocking: `T_d`
contracted explicitly, no dummy leg) refines all five primitives.
`T_d` is a real wavefunction tensor in this blocking, so it is
optimized alongside the rest.

```python
import jax
from tenax import (
    CTMConfig,
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates_exact,
    pess_simple_update,
    optimize_pess_ad,
)

D, d = 2, 3  # spin-1
H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=d)
cg_gates = kagome_xxz_pess_cg_gates_exact(delta=1.0, d=d)

state = IPESSState.random(D=D, d=d, key=jax.random.PRNGKey(0))
state = pess_simple_update(state, H,
                           dt_schedule=[(0.1, 200), (0.01, 200), (0.001, 100)],
                           D_max=D)

config = CTMConfig(chi=8, max_iter=30, conv_tol=1e-7,
                   projector_method="svd", forward_gauge="phase",
                   ctm_conv_method="elementwise")
state, e_per_site = optimize_pess_ad(state, cg_gates, config, max_iter=30,
                                     loss_builder="exact")
print(f"E/site = {e_per_site:.6f}")  # spin-1 D=2 lands around -1.27
```

`loss_builder` defaults to `"convc"` — the legacy Convention-C loss
(`kagome_xxz_pess_cg_gates` gates, `T_d` frozen) kept only for backward
compatibility. **Do not use it for physics results (#1002):** on
SU-converged states its CTM collapses to rank-1 corners, so the
converged readout is backend-dependent and is not the kagome energy
(the spin-1 D=2 "around -1.0" quoted here before #1002 came through
that broken probe; the exact-path value is -1.270). Each
`loss_builder` requires its matching gate builder, as above —
mismatched pairings encode different inter-cell sub-site pairings and
are rejected at entry.

The full kagome Hamiltonian (3 up-triangle bonds + 3 down-triangle
bonds per unit cell) is reconstructed via `compute_energy_cg`'s
intra-cell + horizontal/vertical/diagonal inter-cell 2-site RDMs; see
`examples/kagome_spin12_pess_ad_benchmark.py` and
`examples/kagome_spin1_pess_ad_benchmark.py` for full sweeps.

### Exact supersite loss readout

`pess_to_kagome_supersite_exact` blocks all five iPESS primitives
(`R_a, R_b, R_c, T_u, T_d`) into one rank-5 supersite with four real
virtual legs and no dummy — the same single-PEPS-site mapping variPEPS
uses for kagome 3-PESS. `build_pess_loss_exact` runs it through the
single-site CTM (forward + implicit AD); on control states it agrees
with variPEPS to 1e-9 and with exact cylinder oracles to ~2e-4 at D=2
and D=4 (issue #991). This is the loss `optimize_pess_ad(...,
loss_builder="exact")` optimizes; call it directly for a standalone
energy readout of an existing state:

```python
from tenax import (
    build_pess_loss_exact,
    kagome_xxz_pess_cg_gates_exact,
)

loss = build_pess_loss_exact(kagome_xxz_pess_cg_gates_exact(delta=1.0, d=d),
                             config)
e_per_site = float(loss(state).real)
```

### Multisite path (3-site kagome on a square unit cell)

For the multisite encoding `pess_to_kagome_3site_multisite`, where the
kagome unit cell maps to three sites `(u, v, w)` on a square lattice and
the energy uses 4 NN bonds + 2 marginalised-3-site contributions, use
`build_pess_loss_3site_multisite` and `optimize_pess_3site_multisite_ad`.
**Caution (#991):** the multisite encoding places dim-1 bonds on the CTM
lattice, where the plaquette environment's fixed point rank-truncates and
biases per-site energies by ~2.5e-3 in the non-variational direction —
prefer `build_pess_loss_exact` above for any quantitative energy readout:

```python
from tenax import (
    build_pess_loss_3site_multisite,
    optimize_pess_3site_multisite_ad,
    pess_to_kagome_3site_multisite,
)
from tenax.algorithms._pess_multisite_energy import kagome_3site_bond_gates

bond_gates = kagome_3site_bond_gates(delta=1.0, d=d)
state, e_per_site = optimize_pess_3site_multisite_ad(
    state, bond_gates, config, max_iter=30,
)
```

The optimizer warm-starts CTM envs across L-BFGS steps via an internal
`env_cache`, returns the best-seen energy across the trajectory, and
gates `CTMConfig` at entry on the implicit-AD invariants
(`projector_method='svd'`, `forward_gauge` in `('phase', 'bond_phase')`,
`ctm_conv_method='elementwise'`).
