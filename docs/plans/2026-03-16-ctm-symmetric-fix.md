# Fix SymmetricTensor CTM: C4v Mode + General Charge Consistency

**Date**: 2026-03-16
**Status**: Draft

## Problem

The SymmetricTensor CTM has a charge-distribution divergence bug:
each directional move independently truncates chi via its own projector,
which can split chi across charge sectors differently (e.g. {q=0: 2, q=1: 2}
vs {q=0: 1, q=1: 3}). After 2-3 sweeps, connected chi legs have
incompatible block sizes, causing `ValueError` in `_contract_symmetric`.

This affects **all** SymmetricTensor symmetries (FermionParity, U1, Zn),
not just fermionic cases. Currently worked around by densifying fermionic
tensors before the CTM loop.

## Root Cause

In `_ctm_tensor_moves.py`, four independent moves (left/right/top/bottom)
each compute a projector via `_compute_projector_tensor` and truncate to
chi. The per-sector dimension split in the projector output depends on
the eigenvalue spectrum of the grown corner density matrix, which varies
between moves. After relabeling, chi legs that must later contract
(e.g. C2.c2_d with T2.t2_u) may have different per-sector sizes.

Secondary issue: the D² leg flow direction is flipped during relabeling
(e.g. d2→u2 changes the label but not the FlowDirection). Partial fix
already in place (`_flip_leg_flow`).

## Design

### Part 1: C4v CTM (new, for 1-site unit cell)

For a 1-site translationally invariant iPEPS with C4v point-group
symmetry, all four corners are identical (up to rotation) and all four
edges are identical. Only **one move per sweep** is needed.

Following YASTN's approach:

**Environment representation** (C4v):
- Store only one corner `C` and one edge `T`
- All eight environment tensors derived by rotation/transposition:
  `C1 = C`, `C2 = C.T`, `C3 = C`, `C4 = C.T`,
  `T1 = T`, `T2 = T.flip`, `T3 = T`, `T4 = T.flip`

**Single move per sweep**:
1. Grow corner: `Cg = C @ T` contracted with `T` and double-layer `a`
2. Compute projector from Cg via SVD (one projector, one charge distribution)
3. Update `C_new = project(Cg)`, `T_new = sandwich(T, a, projectors)`
4. Apply `flip_charges` on T_new's chi legs for correct flow convention
5. Normalize

**Advantages**:
- Only one projector → no charge distribution divergence (problem eliminated)
- 4× fewer projector computations per sweep
- Simpler code
- Natural convergence check on single corner's singular values

**Limitations**:
- Only works for 1-site unit cells with C4v symmetry
- Not applicable to 2-site (checkerboard), rectangular, or distorted lattices

**New file**: `src/tenax/algorithms/_ctm_tensor_c4v.py`

**API**:
```python
def ctm_tensor_c4v(
    A: Tensor,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    **kwargs,
) -> CTMTensorEnv:
    """CTM for 1-site C4v-symmetric iPEPS using single-move sweeps."""
```

### Part 2: General CTM charge consistency fix

For multi-site unit cells or non-C4v cases, we need four moves but
must keep charge distributions consistent.

**Approach: paired horizontal/vertical moves** (following YASTN):

Instead of four independent moves (l, r, t, b), use two paired moves:
- **Horizontal move**: compute one projector pair from left AND right
  grown corners together. Apply to update C1, T4, C4 (left side) and
  C2, T2, C3 (right side) using the SAME projector pair.
- **Vertical move**: similarly for top/bottom.

This ensures that chi legs produced by a single projector pair are
always consistent when later contracted.

**Implementation**:

Modify `_ctm_tensor_sweep` to call two paired moves instead of four
independent moves:

```python
def _ctm_tensor_sweep(env, a, chi, ...):
    env = _ctm_tensor_move_horizontal(env, a, chi, ...)  # updates l+r
    env = _ctm_tensor_move_vertical(env, a, chi, ...)    # updates t+b
    return env
```

Each paired move:
1. Grow corners for both directions
2. Compute ONE projector from the combined corners
3. Apply projector to update all affected tensors
4. Fix D² leg flows via `_flip_leg_flow`

**Alternative**: Use the same projector for all four moves within a sweep.
Compute the projector once from the "worst" corner pair, then apply it
everywhere. Simpler but may be less accurate.

### Part 3: Remove densify workaround

Once Parts 1 and 2 are implemented and tested:
1. Remove the `BraidingStyle.FERMIONIC` check in `ctm_tensor()`
2. Remove the DenseTensor fallback in `compute_energy_fermionic_ctm()`
3. Add regression tests ensuring SymmetricTensor CTM matches DenseTensor
   results for FermionParity, U1, and Zn symmetries

## File Changes

| File | Change |
|------|--------|
| Create `_ctm_tensor_c4v.py` | C4v single-move CTM |
| Modify `_ctm_tensor_moves.py` | Add paired horizontal/vertical moves |
| Modify `_ctm_tensor_convergence.py` | Wire up C4v path, remove densify workaround |
| Modify `_ctm_tensor.py` | Export `ctm_tensor_c4v` |
| Modify `__init__.py` | Export `ctm_tensor_c4v` |
| Create `tests/test_ctm_tensor_c4v.py` | C4v CTM tests |
| Modify `tests/test_ctm_tensor.py` | Symmetric CTM consistency tests |

## Testing

1. **C4v correctness**: C4v CTM energy matches general CTM energy for
   1-site Heisenberg iPEPS (DenseTensor)
2. **C4v symmetric**: Same test with U1 SymmetricTensor
3. **C4v fermionic**: Same test with FermionParity SymmetricTensor
4. **General symmetric**: General CTM with U1 SymmetricTensor produces
   same energy as DenseTensor path (no charge divergence)
5. **General fermionic**: Same with FermionParity
6. **Multi-site**: 2-site checkerboard with SymmetricTensor works
7. **Convergence**: C4v converges in fewer sweeps than general CTM

## Implementation Order

1. Part 1 (C4v CTM) — standalone, no existing code changes
2. Part 3 tests — validate C4v against existing dense CTM
3. Part 2 (general fix) — modify existing move structure
4. Remove densify workaround
