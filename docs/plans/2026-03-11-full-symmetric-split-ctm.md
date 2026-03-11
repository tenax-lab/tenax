# Full Symmetric Split CTM Design

**Date:** 2026-03-11
**Status:** Approved
**Reference:** arXiv:2502.10298 (Naumann et al.), variPEPS implementation

## Goal

Eliminate all `todense()` calls from split CTM sweeps and energy computation so
SymmetricTensor iPEPS runs the full CTM iteration in block-sparse mode.

Current state: 20 todense calls per sweep, ~12 in energy, 1 in convergence.
Target state: 0 todense calls (except JAX-level operations inside block-sparse
SVD/eigh which are per-charge-sector).

## Background

The split-CTMRG keeps ket and bra layers separate, defining 12 environment
tensors (4 corners + 4 ket edges + 4 bra edges) connected by an interlayer
bond χ_I. Per-layer projectors are computed via eigh on half-system density
matrices, then combined via contraction over the interlayer index.

The standard CTM (`_ctm_tensor.py`) already runs fully in Tensor protocol with
block-sparse projectors (`_eigh_projector_symmetric` in `_ctm_projector.py`)
and block-sparse SVD (`linalg.svd`). The split CTM has the same building blocks
available but currently densifies at every projector computation and interlayer
contraction.

Previous attempt failed because Tensor-protocol projectors were mixed with
dense `_wrap_corner_dense` functions that enforce hardcoded charge conventions.
The charges from the projector drifted from the wrap conventions across sweeps.
**Fix**: stay fully in Tensor protocol — corners and edges come out of
`contract()` and `svd()`, never wrapped. Charges are self-consistent by
construction.

## Per-move data flow (left move example)

### Current (dense)

```
C1g_ket = contract(C1, T1_ket) → fuse → todense()
P_ket = _compute_projector_dense(C1g_ket_dense, C4g_ket_dense, chi)
C1_mid = P_ket† @ dense
C1g_bra = einsum(C1_mid, T1_bra.todense()) → reshape
P_bra = _compute_projector_dense(C1g_bra_dense, C4g_bra_dense, chi)
C1_new = P_bra† @ dense → _wrap_corner_dense()
P_full = reshape+einsum(P_ket, P_bra) → dense sandwich on T_grown.todense()
SVD split → _wrap_edge_*_dense()
```

### Proposed (tensor protocol)

```
C1g_ket = contract(C1, T1_ket) → fuse_indices         [Tensor]
P_ket = _compute_projector_tensor(C1g_ket, C4g_ket)   [SymmetricTensor]
C1_mid = contract(P_ket.bar(), C1g_ket)                [Tensor]
C1g_bra = contract(C1_mid, T1_bra) → fuse_indices     [Tensor]
P_bra = _compute_projector_tensor(C1g_bra, C4g_bra)   [SymmetricTensor]
C1_new = contract(P_bra.bar(), C1g_bra)                [Tensor, no wrap]

P_ket_3d = unfuse_indices(P_ket, ...)                  [Tensor]
P_bra_3d = unfuse_indices(P_bra, ...)                  [Tensor]
P_full = contract(P_ket_3d, P_bra_3d) → fuse_indices  [Tensor]

T_grown = _grow_edge_no_double_layer(...)              [Tensor, stop before todense]
T_projected = contract(P_full.bar(), T_grown, P_full)  [Tensor]
T_ket_new, T_bra_new = linalg.svd(T_projected, chi_I) [block-sparse SVD]
```

## Energy computation

Replace `_split_env_to_dense_standard` + `compute_energy_ctm` with:

1. Merge split edges via `contract()` over interlayer index + `fuse_indices`
   to produce standard double-layer edges as Tensors.
2. Construct `CTMTensorEnv` from corners (already chi×chi Tensors) and
   merged edges.
3. Call existing `compute_energy_ctm_tensor` which uses tensor-protocol RDMs.

## Convergence check

Replace `jnp.linalg.svd(env.C1.todense(), compute_uv=False)` with
`linalg.svd(env.C1, ...)` which dispatches to block-sparse SVD for
SymmetricTensor.

## Functions deleted

- `_compute_projector_dense` — replaced by `_compute_projector_tensor`
- `_svd_split_edge_dense` — replaced by `linalg.svd`
- `_wrap_corner_dense` — no longer needed (Tensor protocol preserves charges)
- `_wrap_edge_ket_dense` — no longer needed
- `_wrap_edge_bra_dense` — no longer needed
- `_split_env_to_dense_standard` — replaced by tensor-protocol merge

## Functions modified

- `_split_ctm_move_left/right/top/bottom` — rewritten to use Tensor protocol
- `_grow_edge_no_double_layer` — return Tensor instead of todense+reshape
- `compute_energy_split_ctm_tensor` — rewritten to merge env and delegate
- `ctm_split_tensor` — convergence check uses block-sparse SVD

## Key risk: combined projector unfuse/fuse

The combined projector requires:
```
P_ket: (fused[chi,D], chi_new) → unfuse → (chi, D, chi_new)
P_bra: (fused[chi_k,D], chi) → unfuse → (chi_k, D, chi)
P_full = contract over chi_k → (chi, D, D, chi) → fuse(D,D) → (chi*D², chi)
```

`unfuse_indices` must correctly recover the original two indices from the fused
index. This works because `fuse_indices` stores the original structure in the
fused TensorIndex metadata.

## Testing

- `test_symmetric_sweep_no_todense` (currently xfail strict) flips to pass
- All existing tests continue passing (DenseTensor dispatches to same ops)
- FermionicU1 charge preservation test validates block-sparse correctness
- Energy tests validate tensor-protocol merge matches dense conversion
