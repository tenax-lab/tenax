# Symmetric & Fermionic AD for iPEPS

## Goal

Enable AD-based variational optimization of fermionic iPEPS through the
existing CTM implicit-differentiation pipeline, and harden the
block-sparse AD path with Lorentzian-regularized per-sector SVD gradients.

## Background

Tenax already has:
- `optimize_gs_ad` for bosonic iPEPS (dense jax.Array or DenseTensor)
- `ctm_tensor_converge` with implicit differentiation via GMRES (works
  polymorphically with SymmetricTensor)
- Fermionic iPEPS simple update with FermionParity graded tensors
  (Koszul signs handled automatically, no explicit swap gates)
- `truncated_svd_ad` with Lorentzian-regularized backward (dense only)

Missing:
- No AD optimization path for fPEPS
- `todense()` round-trips in the AD path may break gradient flow
- Per-sector SVD in SymmetricTensor has no regularization for
  degenerate singular values

Reference: YASTN (yastn/yastn) implements fermionic iPEPS AD with
block-sparse PyTorch autograd, fixed-point implicit differentiation, and
Lorentzian SVD regularization per charge sector. Our approach follows
the same strategy adapted to JAX.

## Architecture

Three changes in dependency order:

### Change 1: Fermionic AD optimization

New entry point `optimize_fpeps_ad` in `ipeps_optimize.py`:

- Takes fPEPS site tensor(s) (SymmetricTensor with FermionParity) +
  model parameters (t, V for t-V model)
- Optional simple-update warm start
- Calls `ctm_tensor_converge` (implicit diff) with the fermionic tensor
- Energy via `compute_energy_ctm_tensor` (Tensor-protocol, already
  handles SymmetricTensor)
- Optimizer: optax adam + gradient clipping (same as bosonic path)

The 2-site Hamiltonian gate from `spinless_fermion_gate` must be wrapped
as a Tensor with FermionParity indices for the RDM energy trace.

**Key assumption:** Koszul signs are automatic in the graded tensor
formalism, so the standard CTM + implicit diff pipeline works for
fermionic tensors without explicit swap gates. The simple update
already validates this for the forward pass.

**Test:** t-V model at V=0 (free fermion limit) — compare AD-optimized
energy against exact result.

### Change 2: Differentiable todense() round-trip

The energy RDM path does:
```
SymmetricTensor contractions -> .todense() -> reshape/normalize -> trace
```

`todense()` produces a `jax.Array` that should already be in the JAX
trace (differentiable). Verify this; if gradient flow breaks, add
`jax.custom_vjp` at the specific call sites.

The gauge-fixing path does `todense()` -> QR -> `from_dense()`. The
`from_dense()` reconstruction may use non-differentiable index ops.
If so, wrap the full round-trip in a `custom_vjp`:
- Forward: `todense()` -> QR -> `from_dense()` (unchanged)
- Backward: dense QR backward, scatter to block positions

Since RDMs are small (chi^2 * D^2), dense materialization is fine for
performance. We only need unbroken gradient chains.

### Change 3: Lorentzian SVD per charge sector

Factor out the backward logic from `_truncated_svd_ad_bwd` into a
reusable function:

```python
def _svd_sector_backward(U, s, Vh, dU, ds, dVh, eps=1e-12):
    """Lorentzian-regularized SVD backward for one dense sector."""
    # F-matrix, truncation correction (existing logic from ad_utils.py)
    ...
    return dM
```

New `_truncated_svd_symmetric_ad`:
- Forward: same as `_truncated_svd_symmetric` (per-sector SVD, global
  truncation across sectors)
- Backward: `_svd_sector_backward` applied per sector, gradients
  scattered to correct sectors based on global truncation sort

Wire into CTM projector path gated by `CTMConfig.ad_regularize_svd`
(default True). Non-AD callers are unaffected.

Test: SymmetricTensor with known degenerate singular values within a
sector. Verify gradients are finite with regularization, NaN without.

## Implementation Order

1. **Change 1 first** — get fermionic AD working end-to-end using
   existing todense fallback. This validates the pipeline.
2. **Changes 2 + 3** — harden the AD path. Change 2 ensures gradient
   flow through block-sparse operations. Change 3 prevents NaN from
   degenerate singular values.

## Files

| File | Change |
|------|--------|
| `ipeps_optimize.py` | New `optimize_fpeps_ad` entry point |
| `fermionic_ipeps.py` | Tensor-wrapped gate, energy helper |
| `ad_utils.py` | Factored `_svd_sector_backward`, gauge-fix VJP if needed |
| `linalg.py` | New `_truncated_svd_symmetric_ad` with custom VJP |
| `ipeps_config.py` | `ad_regularize_svd` flag on CTMConfig |
| `tests/test_fpeps_ad.py` | Fermionic AD tests (t-V free fermion) |
| `tests/test_ad_utils.py` | Lorentzian SVD per-sector tests |

## Success Criteria

- fPEPS AD optimization converges on spinless t-V (V=0) to within 1%
  of exact free-fermion energy at chi=16, D=2
- No gradient NaN/inf with degenerate singular values (regularization on)
- All existing iPEPS tests still pass
- No performance regression on bosonic AD path
