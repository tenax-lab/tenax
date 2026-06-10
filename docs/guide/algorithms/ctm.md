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
    forward_gauge="phase",      # "phase" (default), "qr", "sigma", or "none"
    projector_method="eigh",    # "eigh" (default), "qr", or "svd" (Fishman)
    ad_backward_method="vjp",   # "vjp" (default) or "gmres" (experimental)
)
```

### Chi ramping

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
after each CTM sweep. Four modes are supported:

| Value | Description |
|-------|-------------|
| ``"phase"`` (default) | variPEPS-style Frobenius normalization + phase fixing. Cheapest gauge fix that still stabilizes unrolled AD. **Recommended for both implicit and explicit AD** (1-site and 2-site). |
| ``"qr"`` | Legacy QR decomposition on corners with sign-fixed diagonal. Fast and stable for simple update and forward-only CTM. |
| ``"sigma"`` | Transfer-matrix eigenvector alignment via power iteration. Required for element-wise CTM convergence at large chi (1-site path). |
| ``"none"`` | No gauge fix. Diagnostic / benchmark mode only — isolates the cost of gauge fixing from the rest of the sweep. Not recommended for production runs. |

Without a gauge fix, the ``eigh`` projector CTM converges spectrally (corner
singular values stabilize) but is chaotic element-wise — the individual
tensor entries keep fluctuating between iterations, which makes implicit
differentiation ill-conditioned. ``forward_gauge="phase"`` fixes this with
negligible overhead, while ``forward_gauge="sigma"`` is appropriate when
strict element-wise convergence is needed at large chi.

**No silent gauge promotion.** ``optimize_gs_ad`` passes the configured
``forward_gauge`` through unchanged.  Set it explicitly to override the
``"phase"`` default.

### Projector methods

``projector_method`` selects how each sweep truncates back to χ:

| Value | Description |
|-------|-------------|
| ``"svd"`` (default) | Fishman two-projector SVD with safe singular-value handling. |
| ``"eigh"`` | Hermitian eigendecomposition of the corner density matrix. |
| ``"qr"`` | Reduced-corner QR-CTMRG isometry (Zhang, Yang & Corboz, arXiv:2505.00494). On the dense single-site (``recipe="1x1"``) path this is a real reduced-corner QR projector. As of Phase 2 it is also usable under AD ground-state optimization: set ``gs_recipe="1x1"`` + ``gs_projector_method="qr"`` to run reduced-corner QR-CTMRG under the implicit-diff AD optimizer (dense). The SymmetricTensor/block-sparse ``"qr"`` path is a later phase; the default remains ``"svd"`` with ``gs_recipe="2x2"``. |

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
