# HOTRG

Higher-Order Tensor Renormalization Group (HOTRG) improves upon TRG by using
Higher-Order SVD (HOSVD) to compute optimal truncation isometries.

## Background

Instead of pairwise SVD splits (as in TRG), HOTRG constructs the truncation
isometry from the **environment tensor** -- formed by contracting two adjacent
tensors over their shared bonds -- and computing its HOSVD. This produces
a globally better approximation at each coarse-graining step.

Algorithm (horizontal step):

1. Form $M[u, u', d, d'] = \sum_{l,r} T[u,d,l,r] \cdot T[u',d',r,l]$
2. HOSVD of $M$: compute truncated isometries $U_u$, $U_d$
3. Compress $T$ using the isometries and contract to form $T_{\text{new}}$

The vertical step is analogous with left/right bonds.

Reference: Xie et al., PRB 86, 045139 (2012).

## Configuration

```python
from tenax import HOTRGConfig

config = HOTRGConfig(
    max_bond_dim=16,               # maximum chi
    num_steps=20,                  # RG iterations
    direction_order="alternating", # "alternating" or "horizontal_first"
    svd_trunc_err=None,            # optional truncation error threshold
)
```

## Example -- 2D Ising model

```python
import math
from tenax import HOTRGConfig, hotrg, compute_ising_tensor, ising_free_energy_exact

beta_c = math.log(1 + math.sqrt(2)) / 2
tensor = compute_ising_tensor(beta_c)

config = HOTRGConfig(max_bond_dim=16, num_steps=20)
log_Z_per_site = hotrg(tensor, config)

exact = ising_free_energy_exact(beta_c)
print(f"HOTRG log(Z)/N = {float(log_Z_per_site):.8f}")
print(f"Exact          = {exact:.8f}")
```

## TRG vs HOTRG

At the same bond dimension, HOTRG typically achieves better accuracy because
the HOSVD-based isometries account for the full tensor environment rather than
a single pairwise split.

| Method | `max_bond_dim=16` relative error |
|--------|----------------------------------|
| TRG | ~1e-5 |
| HOTRG | ~1e-7 |

The trade-off is that HOTRG is more expensive per step (additional SVDs for
the environment tensor).

## Direction order

- `"alternating"` (default): alternate horizontal and vertical coarse-graining
  steps. This preserves the square-lattice symmetry at each step.
- `"horizontal_first"`: perform both horizontal and vertical coarse-graining
  within each step. May converge faster for anisotropic systems.

(hotrg-multigpu)=
## Multi-GPU sharding

For large-χ dense HOTRG, set `HOTRGConfig(device_mesh=mesh)` (a 1-D
`jax.sharding.Mesh`) to shard the dominant χ⁶ intermediate across multiple GPUs —
~1/N per-device peak memory and a higher reachable χ, at the same free energy.
Since HOTRG is forward-only there is no autodiff-through-SVD barrier, so GSPMD
sharding is effective here (unlike the CTM-AD path). See
`examples/probe_hotrg_multigpu.py`.

(gilt-hotrg)=
## Gilt-HOTRG

`gilt_hotrg` applies the GILT filter of {ref}`gilt-tnr` before every **HOTRG** move (a
drop-in counterpart of `hotrg`; `gilt_eps=0.0` recovers plain HOTRG exactly).
Because HOTRG's HOSVD already suppresses most corner-double-line entanglement,
GILT does not improve the smooth *free energy* here — its payoff shows in the
*critical data*: the estimated `beta_c` lands closer to Onsager than plain
HOTRG at the same bond dimension. It reuses the χ⁶ sharding above via
`GiltHOTRGConfig(device_mesh=mesh)`. See `examples/gilt_hotrg_ising.py`.

```python
from tenax import GiltConfig, GiltHOTRGConfig, gilt_hotrg, compute_ising_tensor

T = compute_ising_tensor(0.44068679350977147, symmetric=True)
config = GiltHOTRGConfig(max_bond_dim=16, num_steps=18, gilt=GiltConfig(gilt_eps=1e-3))
log_z_per_n = gilt_hotrg(T, config)
```

(potts)=
## q-state Potts model

The same coarse-graining works for the **q-state Potts model**
(`compute_potts_tensor` produces any `q >= 2`; `q = 2` reduces to Ising):

```python
from tenax import HOTRGConfig, hotrg, compute_potts_tensor, potts_critical_beta

q = 3
beta_c = potts_critical_beta(q)  # ln(1 + sqrt(q)), the self-dual critical point
T = compute_potts_tensor(beta_c, q=q)

config = HOTRGConfig(max_bond_dim=16, num_steps=20)
log_z_per_n = hotrg(T, config)
print(f"Potts q={q} at beta_c={beta_c:.5f}:  ln(Z)/N = {float(log_z_per_n):.6f}")
```
