# Fermionic Implementation Audit

## Audit History

- **PR #94** (2026-03-10): Initial audit and fixes (fermionic-fixes branch)
- **PR #103** (2026-03-10): Fixed stale audit framing and dagger() docstring
- **Current audit** (2026-03-10): Full re-audit of fermionic support

## Current State Summary

Fermionic support in Tenax is **production-ready** across all core operations.
Koszul signs are correctly applied in contraction, SVD, QR, and transpose.
Three symmetry classes support fermionic statistics: `FermionParity`,
`FermionicU1`, and `ProductSymmetry` (when either factor is fermionic).

## Component Status

### Core Tensor Operations — Complete

| Operation | Fermionic Support | Location |
|-----------|:-:|----------|
| `contract()` | Yes | contractor.py:362-470 |
| `truncated_svd()` | Yes | linalg.py:171-183 |
| `qr_decompose()` | Yes | linalg.py:424-437 |
| `transpose()` | Yes | tensor.py:1022-1026 |
| `dagger()` | Yes | tensor.py:975-984 |
| `bar()` | Yes (no twist) | tensor.py:991-1000 |
| `conj()` | Yes (data only) | tensor.py:944-952 |
| `eigh` | No (intentional) | Hermiticity incompatible with fermionic reordering |

### Symmetry Classes — Complete

| Class | `is_fermionic` | `exchange_sign` | `parity` | Location |
|-------|:-:|:-:|:-:|----------|
| `FermionParity` | Yes | `1-2*(p_a*p_b)` | `q % 2` | symmetry.py:311-365 |
| `FermionicU1` | Yes | Same formula | Configurable grading | symmetry.py:374-449 |
| `ProductSymmetry` | OR of factors | Sum of parities mod 2 | Sum mod 2 | symmetry.py:452-582 |

### AutoMPO — Complete

- `fermion_site_ops()`: C, Cd, N, F, Id operators (auto_mpo.py:70-88)
- `_insert_jw_strings()`: Pairwise JW string insertion (auto_mpo.py:570-623)
- `build_auto_mpo()`: `fermionic_ops` kwarg passthrough (auto_mpo.py:766)
- Public export: `from tenax import fermion_site_ops`

### fPEPS (fermionic iPEPS) — Complete

- Simple update: `_fpeps_simple_update_horizontal/vertical()` using
  label-based `contract()` and `truncated_svd()` (fermionic_ipeps.py:208-362)
- CTM dispatch: `fermionic_ctm()` routes Tensor inputs to `ctm_tensor()`
  (fermionic_ipeps.py:408-433)
- Comments correctly describe graded tensor formalism (fermionic_ipeps.py:401-404)

### Test Coverage

| Test Area | File | Coverage |
|-----------|------|----------|
| Symmetry classes | test_fermionic.py:82-331 | FermionParity, FermionicU1, ProductSymmetry |
| Koszul signs | test_fermionic.py:338-370 | Identity, swaps, multi-element perms |
| Fermionic transpose | test_fermionic.py:377-441 | Roundtrip, bosonic control |
| Fermionic contraction | test_fermionic.py:448-490 | Charge conservation |
| Fermionic SVD | test_fermionic.py:497-569 | Roundtrip reconstruction |
| Fermionic QR | test_fermionic.py:576-606 | Roundtrip reconstruction |
| Cross-validation (dense) | test_fermionic.py:869-958 | FermionParity, FermionicU1, ProductSymmetry |
| Dagger involution | test_fermionic.py:832-866 | dagger(dagger(T)) == T |
| Bar (fermionic) | test_tensor.py | FermionParity, FermionicU1 todense == conj |
| AutoMPO JW strings | test_fermionic.py:699-770 | NN, NNN, long-range, mixed, bosonic control |
| Free fermion DMRG | test_fermionic.py:799-830 | Energy matches exact result |
| fPEPS simple update | test_fermionic_ipeps.py | 31 tests across 7 classes |
| Fermionic TRG | test_trg.py:TestTRGFermionic | 3 tests |
| Fermionic HOTRG | test_hotrg.py:TestHOTRGFermionic | 3 tests |

## Issues Found in Previous Audits (all resolved)

1. AutoMPO lacked fermionic operators and JW strings → Fixed in PR #94
2. Dagger twist used wrong formula → Fixed in PR #94, docstring fixed in PR #103
3. fPEPS comments referenced swap gates → Fixed in PR #94
4. No dense cross-validation tests → Fixed in PR #94

## Issues Found in Current Audit

1. **`compute_energy_fermionic_ctm()` docstring overstated coverage.**
   Claimed support for `SplitCTMTensorEnv` but no code path existed.
   Fixed: docstring corrected (fermionic_ipeps.py:448-450).

2. **No fermionic `bar()` tests.**
   `bar()` is used in CTM for fermionic tensors but was only tested with
   bosonic U(1). Fixed: added FermionParity and FermionicU1 bar() tests
   (test_tensor.py:TestBar).

## Known Limitations (not bugs)

- **Split CTM block-sparse projectors**: `_split_ctm_tensor.py` uses
  `todense()` fallbacks for SymmetricTensor projectors due to fused index
  charge mismatch. Standard CTM (`_ctm_tensor.py`) handles symmetric
  tensors natively.

- **AD-based fPEPS optimization**: Not yet implemented. Current fPEPS
  uses simple update only. Requires fermionic CTM inside the gradient
  loop with explicit swap-gate handling.

- **`eigh` has no fermionic sign handling**: Intentional — eigendecomposition
  assumes Hermiticity, which is incompatible with fermionic leg reordering.

## Comparison with Other Libraries

For a detailed comparison of Tenax's fermionic implementation with
TensorKit.jl, ITensor, TeNPy, Cytnx, and quimb, see the
[migration guides](https://tenax-lab.github.io/migration/#tensorkitjl).

Key difference with TensorKit.jl: TensorKit encodes fermionic statistics
via abstract category theory (R-symbols, fusion trees, ribbon twists),
which generalises to non-Abelian and anyonic symmetries. Tenax applies
Koszul signs explicitly in each operation, which is simpler to audit but
limited to Abelian fermionic systems. Tenax's `contract()` handles
fermionic signs automatically; TensorKit's `@tensor` fermionic contraction
is still TODO.
