# 2-Site Symmetric Simple Update Design

**Date:** 2026-03-28
**Branch:** `feat/symmetric-simple-update`

## Goal

Remove the 1-site SU code path (physically limited, can't represent AFM Neel order) and implement a unified 2-site Tensor-protocol SU that works with both DenseTensor and SymmetricTensor. Add sublattice rotation utility for transitioning from 2-site SU to 1-site C4v AD.

## Decisions

- **Unify:** Single Tensor-protocol path replaces both the dense JAX-array and Tensor-protocol 1-site paths. No more duplicate logic.
- **Lambdas stay as plain JAX arrays.** `truncated_svd` returns singular values as a 1D array regardless of tensor type.
- **`ipeps()` always does 2-site SU**, returns `(A, B)` tuple. `unit_cell` stays in config for `optimize_gs_ad` dispatch but `ipeps()` ignores it.
- **Sublattice rotation** is a public utility, not wired into `ipeps()` automatically.

## Section 1: Removals

**`ipeps_simple_update.py`:**
- `_simple_update_1x1()`, `_simple_update_3leg()`, `_simple_update_bond()`
- `_simple_update_horizontal()` / `_simple_update_vertical()` (1-site wrappers)
- `_simple_update_horizontal_tensor()` / `_simple_update_vertical_tensor()` (1-site Tensor-protocol)
- `_absorb_lambdas_tensor()` (1-site lambda helper)

**`ipeps.py`:**
- `_build_1x1_peps()`
- Dense 1x1 path in `ipeps()`
- `_ipeps_tensor()` (1-site Tensor-protocol path)

**`ipeps_config.py`:**
- `unit_cell` stays (still used by `optimize_gs_ad` and multisite CTM)

**`test_ipeps.py`:**
- `TestSimpleUpdate1x1` class
- `test_1x1_backward_compatible`

**Example:**
- Remove 1x1 run from `heisenberg_ipeps_su.py`

## Section 2: New 2-Site Tensor-Protocol SU

Two new functions in `ipeps_simple_update.py`:

```python
def _simple_update_2site_horizontal_tensor(
    A: Tensor, B: Tensor,
    gate: Tensor,
    lam_h: jax.Array, lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:

def _simple_update_2site_vertical_tensor(
    A: Tensor, B: Tensor,
    gate: Tensor,
    lam_h: jax.Array, lam_v: jax.Array,
    max_D: int,
) -> tuple[Tensor, Tensor, jax.Array]:
```

**Algorithm (horizontal):**
1. Absorb outer lambdas onto A (u, d, l) and B (u, d, r) via diagonal scaling
2. Absorb shared lambda onto A.r
3. Contract A and B on shared r-l bond
4. Apply gate on physical legs
5. `truncated_svd(theta, max_singular_values=max_D)` splitting A legs from B legs
6. Distribute sqrt(sigma) into both A_new and B_new
7. Remove outer lambdas by dividing back out
8. Normalize each tensor by max element
9. Return `(A_new, B_new, sigma / max(sigma))`

## Section 3: Entry Point and Sublattice Rotation

**`ipeps()` signature:**

```python
def ipeps(
    hamiltonian_gate: Tensor | jax.Array,
    initial_peps: tuple[Tensor, Tensor] | None,
    config: iPEPSConfig,
) -> tuple[float, tuple[Tensor, Tensor], object]:
```

1. If `initial_peps is None`, initialize random A, B (DenseTensor or SymmetricTensor matching gate type)
2. Build Trotter gate via `_make_trotter_gate_tensor()`
3. Run 2-site SU loop alternating horizontal/vertical
4. Compute energy via 2-site CTM
5. Return `(energy, (A, B), env)`

**Sublattice rotation:**

```python
def sublattice_rotate(A: Tensor, B: Tensor) -> Tensor:
    """Average A and rotated-B into a single C4v-symmetric tensor for 1-site AD."""
```

Apply pi rotation to B (permute u<->d, l<->r), average `(A + B_rot) / 2`.

## Section 4: Testing

**Remove:** `TestSimpleUpdate1x1`, `test_1x1_backward_compatible`

**Adapt:** `TestSimpleUpdate2Site`, `TestIPEPS2Site` to new signatures

**New tests:**
- `test_2site_symmetric_tensor_runs` — U(1) SymmetricTensor SU completes
- `test_2site_symmetric_heisenberg_energy` — energy < -0.5 (better than product state)
- `test_sublattice_rotate` — output is C4v-symmetric
- `test_dense_tensor_matches_old_path` — no regression vs old dense 2-site
