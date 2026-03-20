# 3-Leg Boundary Tensor Refactor

**Date:** 2026-03-20
**Status:** Approved
**Branch:** Fold into FiniteMPS worktree branch (combined PR)
**Motivation:** Ian McCulloch pointed out that 2-leg boundary tensors can't represent non-zero charge states. 3-leg boundaries with explicit trivial bonds fix this and eliminate ~23 special-case locations across 5 files.

## Core Change

All MPS site tensors become uniformly 3-leg `(chi_l, d, chi_r)`:

| Site | Shape | Labels | Notes |
|------|-------|--------|-------|
| 0 | `(1, d, chi)` | `v_-1_0, p0, v0_1` | Left bond = trivial (charge 0) |
| bulk | `(chi_l, d, chi_r)` | `v{i-1}_{i}, p{i}, v{i}_{i+1}` | Unchanged |
| L-1 | `(chi, d, 1)` | `v{L-2}_{L-1}, p{L-1}, v{L-1}_{L}` | Right bond = trivial (charge = 0); target charge encoded via block selection |

## `target_charge` Field

Add explicit `target_charge: int | None` field to `FiniteMPS`:

```python
@dataclass
class FiniteMPS:
    tensors: list[Tensor]
    orth_center: int | None = None
    singular_values: list[jnp.ndarray | None] = field(default_factory=list)
    log_norm: float = 0.0
    target_charge: int | None = None  # None = dense (no symmetry)
```

- `FiniteMPS.random(..., target_charge=0)` produces 3-leg boundaries with charge on the right trivial bond.
- `compute_mps_sector()` reads `mps.target_charge`.
- `validate_mps_sector()` checks right boundary bond charge matches `target_charge`.
- Follows Ian McCulloch's convention: structural info as explicit fields, not implicit in tensor structure.

## Files Changed

| File | Changes |
|------|---------|
| `core/mps.py` | Add `target_charge` field; make `random()` produce 3-leg boundaries |
| `algorithms/dmrg.py` | Remove all `ndim==2` padding/unpadding, delete `_pad_boundary_symmetric` / `_unpad_boundary_symmetric`, simplify `_one_site_update` / `_two_site_update` (both dense and symmetric paths) |
| `algorithms/tdvp.py` | Delete `_site_to_3d` / `_make_site_tensor` boundary logic, simplify `_identity_mpo_site` |
| `algorithms/observables.py` | Remove boundary branch in `_contract_sandwich` |
| `algorithms/cbe.py` | Remove boundary skip — CBE can expand at all bonds uniformly |
| `algorithms/auto_mpo.py` | No change needed — MPO boundaries are legitimately D=1 |

## Deletions

- `_pad_boundary_symmetric()` and `_unpad_boundary_symmetric()` (dmrg.py)
- `_site_to_3d()` boundary detection logic (tdvp.py)
- All `ndim == 2` + `startswith("p")` guard clauses (~23 locations)

## Tests

Update `test_mps.py`, `test_dmrg.py`, `test_tdvp.py` to construct and assert 3-leg boundaries.
