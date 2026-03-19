# MPS Classes with Canonical Form Contracts

**Date:** 2026-03-19
**Status:** Proposed
**Motivation:** [PR #161 review](https://github.com/tenax-lab/tenax/pull/161#issuecomment-4084577041) — Ian McCulloch identified that 1-site DMRG defensively canonicalizes input via `todense()`, when the real fix is output contracts on MPS-producing functions.

## Problem

No canonical form contract exists between algorithms. Each function independently handles (or doesn't) canonicalization:

- `build_random_symmetric_mps()` returns uncanonicalized MPS; consumers must guess.
- 1-site DMRG defensively right-canonicalizes input via `todense()` (lines 175-213 of `dmrg.py`), breaking the block-sparse symmetric tensor path.
- 2-site DMRG output has no defined canonical form.
- CBE's `expand_bond()` silently assumes left-canonical input with no enforcement.
- TDVP also defensively canonicalizes via `todense()`.
- `_qr_left_canonical()` and `_rq_right_canonical()` helpers in `dmrg.py` call `todense()` internally, so every QR/RQ step during sweeps densifies SymmetricTensor.

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Canonical form tracking | Single `orth_center: int \| None` field | Sufficient for A/B/C convention used by all Tenax algorithms. Per-site labels (TeNPy-style) unnecessary. |
| Singular values | Stored at bonds (`list[jax.Array \| None]`) | Every major library (MPToolkit, TeNPy, quimb) stores them. Enables entanglement entropy without re-doing SVD. |
| Canonicalization method | QR sweeps + SVD at center bond | Fast O(chi^2 d) QR sweeps; single SVD at center gives singular values where they matter. |
| Type system | Single class per MPS type (`FiniteMPS`, `InfiniteMPS`) | More Pythonic than MPToolkit's type-per-canonical-form. Matches ITensor/TeNPy/quimb convention. |
| Mutability | `canonicalize()` returns new instance; `__setitem__` invalidates `orth_center` | Prevents stale canonical form claims. Algorithms that mutate during sweeps work with `.tensors` directly. |
| JAX pytree | Not registered (Python-level container) | Sufficient for DMRG/TDVP where outer loop is Python. Pytree registration is a future extension for AD-through-MPS. |
| MPO interface | Unchanged (`TensorNetwork` for finite, single `Tensor` for iDMRG bulk) | MPOs don't have canonical form or singular values to track. |
| Block-sparse path | Canonicalization uses `tenax.linalg.qr` throughout, never `todense()` | Preserves sparsity, quantum numbers, and performance at large bond dimension. |

## `FiniteMPS`

New module: `src/tenax/core/mps.py`

### Data

```python
@dataclass
class FiniteMPS:
    tensors: list[Tensor]                       # L site tensors
    orth_center: int | None = None              # None = unknown canonical form
    singular_values: list[jax.Array | None]     # L-1 bonds, None if not computed
```

### Construction

```python
@staticmethod
def random(L, d, chi, key, *, symmetric=False, symmetry=None,
           target_charge=None) -> FiniteMPS:
    """Random MPS, right-canonicalized (orth_center=0).
    Replaces build_random_symmetric_mps() and build_random_mps()."""

@staticmethod
def from_tensors(tensors, orth_center=None) -> FiniteMPS:
    """Wrap existing tensors. orth_center=None means unknown form."""

@staticmethod
def product_state(local_states, *, symmetric=False) -> FiniteMPS:
    """MPS from a product state (e.g. Neel state)."""
```

### Access and properties

```python
def __getitem__(self, i) -> Tensor: ...
def __setitem__(self, i, tensor): ...  # sets orth_center = None
def __len__(self) -> int: ...
def __iter__(self) -> Iterator[Tensor]: ...

@property
def L(self) -> int: ...
@property
def bond_dims(self) -> list[int]: ...       # len = L-1
@property
def phys_dims(self) -> list[int]: ...       # len = L
@property
def max_bond_dim(self) -> int: ...
@property
def is_symmetric(self) -> bool: ...         # all tensors are SymmetricTensor
```

### Canonicalization

All methods return a new `FiniteMPS` instance. Use `tenax.linalg.qr` (block-sparse for SymmetricTensor).

```python
def canonicalize(self, center: int) -> FiniteMPS:
    """Mixed canonical form: QR sweep from both ends toward center,
    SVD at center bond to populate singular_values[center]."""

def left_canonicalize(self) -> FiniteMPS:
    """Equivalent to canonicalize(center=L-1)."""

def right_canonicalize(self) -> FiniteMPS:
    """Equivalent to canonicalize(center=0)."""
```

### Observables

```python
def norm(self) -> float: ...
def overlap(self, other: FiniteMPS) -> complex: ...
def entanglement_entropy(self, bond: int) -> float:
    """Von Neumann entropy. Uses cached singular_values if available."""
def expectation_value(self, op, site: int) -> float: ...
def correlation(self, op_a, site_a, op_b, site_b) -> float: ...
```

## `InfiniteMPS`

Same module: `src/tenax/core/mps.py`

### Data

```python
@dataclass
class InfiniteMPS:
    tensors: list[Tensor]                   # Unit cell tensors
    singular_values: list[jax.Array]        # Unit cell bonds (always populated)
```

### Construction

```python
@staticmethod
def from_tensors(tensors, singular_values) -> InfiniteMPS: ...

@staticmethod
def random(d, chi, key, *, unit_cell_size=2, symmetric=False,
           symmetry=None) -> InfiniteMPS: ...
```

### Properties and access

```python
@property
def unit_cell_size(self) -> int: ...
@property
def bond_dims(self) -> list[int]: ...
@property
def phys_dims(self) -> list[int]: ...
@property
def is_symmetric(self) -> bool: ...

def __getitem__(self, i) -> Tensor: ...     # modular indexing
def __len__(self) -> int: ...
```

### Canonicalization

```python
def canonicalize(self) -> InfiniteMPS:
    """Standard iMPS canonicalization via transfer matrix diagonalization."""
```

## Algorithm contract changes

| Function | Current signature | New signature |
|----------|------------------|---------------|
| `dmrg()` | `dmrg(hamiltonian: TensorNetwork, initial_mps: TensorNetwork, config) -> DMRGResult` | `dmrg(hamiltonian: TensorNetwork, initial_mps: FiniteMPS, config) -> DMRGResult` |
| `DMRGResult` | `.mps: TensorNetwork` | `.mps: FiniteMPS` (with `orth_center` set) |
| `tdvp()` | `tdvp(mps: TensorNetwork, hamiltonian: TensorNetwork, config, ...)` | `tdvp(mps: FiniteMPS, hamiltonian: TensorNetwork, config, ...)` |
| `tdvp_step()` | returns `TensorNetwork` | returns `FiniteMPS` |
| `idmrg()` | returns `iDMRGResult` with raw tensors | `iDMRGResult.mps: InfiniteMPS` |
| `expand_bond()` | `expand_bond(site, right_tensor, ...)` | `expand_bond(mps: FiniteMPS, bond: int, ...)` — validates `orth_center` |
| `expectation_value()` | `expectation_value(mps_tensors, op, site)` | `FiniteMPS.expectation_value(op, site)` (method) |
| `correlation()` | `correlation(mps_tensors, ...)` | `FiniteMPS.correlation(...)` (method) |
| `build_random_symmetric_mps()` | returns `TensorNetwork` | deprecated, replaced by `FiniteMPS.random()` |
| `build_mpo_*` | returns `TensorNetwork` | unchanged |

## Removals after migration

- `dmrg.py` lines 175-213: defensive input canonicalization (1-site path)
- `dmrg.py` `_qr_left_canonical()`, `_rq_right_canonical()`: replaced by `tenax.linalg.qr`-based canonicalization inside `FiniteMPS.canonicalize()`
- `dmrg.py` `_absorb_r_into_next()`, `_absorb_l_into_prev()`: absorbed into `FiniteMPS` canonicalization
- `dmrg.py` `build_random_symmetric_mps()`, `build_random_mps()`: replaced by `FiniteMPS.random()`
- `tdvp.py` `_right_canonicalize_dense()`: replaced by `FiniteMPS.right_canonicalize()`
- Standalone `expectation_value()`, `correlation()` in `observables.py`: become `FiniteMPS` methods

## Migration strategy

1. Add `FiniteMPS` and `InfiniteMPS` to `src/tenax/core/mps.py`
2. Implement `canonicalize()` using `tenax.linalg.qr` (block-sparse path, no `todense()`)
3. Implement `FiniteMPS.random()` replacing `build_random_symmetric_mps()` / `build_random_mps()`
4. Update `dmrg()` to accept/return `FiniteMPS`; remove defensive canonicalization
5. Replace `_qr_left_canonical` / `_rq_right_canonical` sweep helpers with `FiniteMPS.canonicalize()` for init and `tenax.linalg.qr` for during-sweep steps
6. Update `tdvp()` to accept/return `FiniteMPS`; remove `_right_canonicalize_dense()`
7. Update `idmrg()` to return `InfiniteMPS`
8. Update CBE (`expand_bond`) to take `FiniteMPS` and validate canonical form
9. Move observables to `FiniteMPS` methods
10. Export `FiniteMPS`, `InfiniteMPS` from `__init__.py`; deprecate old functions
11. Update all tests
12. Update `README.md` examples
