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
    forward_gauge="qr",         # "qr" (default) or "sigma"
    projector_method="eigh",    # "eigh" (default), "qr", or "svd" (Fishman)
    ad_backward_method="vjp",   # "vjp" (default) or "gmres"
)
```

### Forward gauge

The ``forward_gauge`` parameter controls how gauge ambiguity is resolved
after each CTM sweep:

| Value | Description |
|-------|-------------|
| ``"qr"`` | QR decomposition on corners; fast, default for simple update. |
| ``"sigma"`` | Transfer-matrix eigenvector alignment via power iteration. Ensures element-wise convergence. **Required for stable AD optimization.** |

Without sigma gauge, the ``eigh`` projector CTM converges spectrally
(corner singular values stabilize) but is chaotic element-wise -- the
individual tensor entries keep fluctuating between iterations. This
makes implicit differentiation ill-conditioned and causes GMRES to
diverge. Setting ``forward_gauge="sigma"`` fixes this.

### AD backward method

| Value | Description |
|-------|-------------|
| ``"vjp"`` (default) | Iterative VJP (Neumann series). Safer without sigma gauge. |
| ``"gmres"`` | Direct Krylov solve. Faster and recommended with ``forward_gauge="sigma"``. |

### Recommended AD configuration

```python
ctm_cfg = CTMConfig(
    chi=16,
    max_iter=100,
    conv_tol=1e-10,
    forward_gauge="sigma",
    ad_backward_method="gmres",
)
```

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
