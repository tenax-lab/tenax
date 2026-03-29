# NumPy-Only DMRG/iDMRG Fast Path

**Date:** 2026-03-29
**Status:** Approved

## Problem

Symmetric DMRG at chi=32 takes 38.9s (L=20, 10 sweeps). Profiling shows 83% of
the time is JAX overhead — JIT compilation for per-block SVD/QR/eigh (~18s first
sweep), JAX dispatch for array ops, and pytree management. The symmetric path
gets zero benefit from JAX: no AD needed, no JIT benefit (1000s of small blocks
make compilation overhead dominate), and GPU uses cuTENSOR separately.

TeNPy achieves ~1.3s for the same problem by using pure C/Cython with BLAS.

## Design

Add parallel numpy versions of all symmetric DMRG/iDMRG functions. Keep existing
JAX versions as reference implementations. A config flag selects the path.

### Config Flag

```python
@dataclass
class DMRGConfig:
    numpy_blockwise: bool = True   # default: use fast numpy path

@dataclass
class iDMRGConfig:
    numpy_blockwise: bool = True
```

When `True`, the symmetric DMRG/iDMRG path uses numpy throughout — zero JAX
calls. When `False`, falls back to existing JAX path (useful for debugging or
comparing results).

### Data Type: BlockArray

Lightweight numpy-backed block-sparse array for the DMRG hot loop. Lives in
`dmrg.py` (not exported).

```python
@dataclass
class BlockArray:
    blocks: dict[tuple[int, ...], np.ndarray]
    indices: tuple[TensorIndex, ...]
```

Free functions for arithmetic (no method dispatch overhead):

- `ba_scale(ba, scalar) -> BlockArray` — multiply all blocks by scalar
- `ba_add(a, b) -> BlockArray` — add corresponding blocks
- `ba_inner(a, b) -> float` — sum of element-wise products (Frobenius inner product)
- `ba_norm(ba) -> float` — sqrt(ba_inner(ba, ba))
- `ba_conj(ba) -> BlockArray` — conjugate all blocks
- `symmetric_to_ba(t: SymmetricTensor) -> BlockArray` — extract blocks + indices
- `ba_to_symmetric(ba: BlockArray) -> SymmetricTensor` — reconstruct via `_init_flat_buffer`

Conversion to/from `SymmetricTensor` happens only at sweep boundaries, not in
the Lanczos inner loop.

### New Linalg Functions (linalg.py)

Parallel numpy versions of the symmetric decomposition functions. Same algorithm,
same block-grouping logic, but `np.linalg.*` instead of `jnp.linalg.*` and
direct numpy indexing (`matrix[slice] = block`) instead of JAX functional
updates (`.at[].set()`).

```python
def _truncated_svd_symmetric_np(
    blocks: dict, indices: tuple[TensorIndex, ...],
    left_labels: tuple[str, ...], right_labels: tuple[str, ...],
    max_bond_dim: int | None, cutoff: float,
) -> tuple[BlockArray, np.ndarray, BlockArray, np.ndarray]:
    """NumPy version of _truncated_svd_symmetric."""
    # Same sector-grouping logic
    # np.linalg.svd per sector
    # numpy scatter: matrix[r0:r1, c0:c1] = block
    # Returns BlockArray for U, Vh; np.ndarray for s, s_full

def _qr_symmetric_np(
    blocks: dict, indices: tuple[TensorIndex, ...],
    left_labels: tuple[str, ...], right_labels: tuple[str, ...],
) -> tuple[BlockArray, BlockArray]:
    """NumPy version of _qr_symmetric."""

def _eigh_symmetric_np(
    blocks: dict, indices: tuple[TensorIndex, ...],
    max_bond_dim: int | None, cutoff: float,
) -> tuple[np.ndarray, BlockArray]:
    """NumPy version of _eigh_symmetric."""
```

Public API dispatch:

```python
def truncated_svd(tensor, ..., _force_jax=False):
    if isinstance(tensor, SymmetricTensor) and not _force_jax:
        return _truncated_svd_symmetric_np(...)
    ...
```

The `_force_jax` parameter (private, for testing) selects the JAX reference path.

### New DMRG Functions (dmrg.py)

```python
def _lanczos_solve_np(
    matvec: Callable[[BlockArray], BlockArray],
    initial: BlockArray,
    num_steps: int, tol: float,
) -> tuple[float, BlockArray]:
    """Lanczos eigensolver on BlockArray with numpy tridiagonal eigh."""
    # Same algorithm as _lanczos_solve_tensor but with ba_* arithmetic
    # np.linalg.eigh for tridiagonal matrix (replaces jnp.linalg.eigh)

def _two_site_update_symmetric_np(
    site_l, site_r, left_env, mpo_l, mpo_r, right_env, config,
) -> tuple[BlockArray, float]:
    """2-site DMRG update using numpy-only path."""
    # Convert inputs to BlockArray
    # matvec via _blockwise_contract (already numpy)
    # Lanczos via _lanczos_solve_np
    # SVD via _truncated_svd_symmetric_np

def _one_site_update_symmetric_np(
    site, left_env, mpo_site, right_env, config,
) -> tuple[BlockArray, float]:
    """1-site DMRG update using numpy-only path."""
```

### Dispatch in Sweep Logic

```python
def _symmetric_ops(config: DMRGConfig) -> SweepOps:
    if config.numpy_blockwise:
        return SweepOps(
            two_site_update=_two_site_update_symmetric_np,
            one_site_update=_one_site_update_symmetric_np,
            update_left_env=_update_left_env_symmetric,   # already numpy
            update_right_env=_update_right_env_symmetric,  # already numpy
            ...
        )
    else:
        return SweepOps(
            two_site_update=_two_site_update_symmetric,    # JAX reference
            ...
        )
```

### What Stays Unchanged

- `SymmetricTensor` class — no changes
- All existing JAX functions — kept as reference
- Dense DMRG path — keeps JAX (benefits from JIT)
- iPEPS, CTM, TRG — untouched
- Contraction engine — untouched
- `_blockwise_contract` — already numpy, used by both paths

### Expected Performance

| Component | Before (JAX) | After (NumPy) |
|-----------|-------------|--------------|
| SVD per block | jnp.linalg.svd + JIT | np.linalg.svd (no JIT) |
| QR per block | jnp.linalg.qr + JIT | np.linalg.qr |
| Lanczos arithmetic | SymmetricTensor ops (JAX dispatch) | BlockArray ops (numpy, ~0 overhead) |
| Tridiagonal eigh | jnp.linalg.eigh | np.linalg.eigh |
| Contraction | BLAS plan (already numpy) | Same |
| JIT compilation | ~18s first sweep | 0s |

Estimated total: 38.9s → ~8-12s (3-5x improvement). Remaining gap to
TeNPy (~1.3s) is from per-block Python loop overhead (amenable to future
Cython optimization of SVD/QR loops).

## Testing

- All existing symmetric DMRG/iDMRG tests run with `numpy_blockwise=True`
  (new default) and `numpy_blockwise=False` (JAX reference).
- Energy values must match between paths to machine precision.
- Add a parametrized fixture: `@pytest.mark.parametrize("numpy_blockwise", [True, False])`.
