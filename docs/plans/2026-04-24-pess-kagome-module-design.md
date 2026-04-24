# PESS Kagome Module Design

**Goal:** Extract the Kagome PESS simple update from example code into a library module, reusing existing iPEPS CTM infrastructure for energy measurement.

**Status:** Design approved, implementation deferred.

## New files
- `src/tenax/algorithms/pess.py` — PESS state, SU, coarse-graining
- `tests/test_pess.py` — unit tests

## Public API

```python
from tenax import pess_kagome, PESSState

# gate_3site: (8, 8) or (2,2,2,2,2,2) triangle Hamiltonian
energy, pess_state = pess_kagome(gate_3site, config)

# pess_state.to_ipeps() returns (D_eff, D_eff, D_eff, D_eff, d_eff) array
# that feeds into existing ctm() / optimize_gs_ad()
A_ipeps = pess_state.to_ipeps()
```

## Components
1. `PESSState` — frozen dataclass holding `(site_tensors, simplex_tensors, lambdas)`
2. `pess_simple_update_step()` — one HOSVD-based SU step on a triangle
3. `pess_simple_update()` — full SU loop alternating up/down triangles
4. `PESSState.to_ipeps()` — coarse-grain to standard iPEPS super-site tensor
5. `pess_kagome()` — entry point: init → SU → coarse-grain → CTM energy
6. `kagome_triangle_hamiltonian()` — build 3-site XXZ/Heisenberg gate

## Config
Reuse `iPEPSConfig` (already has `max_bond_dim`, `dt`, `num_imaginary_steps`, `ctm`). No new config class.

## Exports
Add `pess_kagome`, `PESSState`, `kagome_triangle_hamiltonian` to `__init__.py`.

## Scope
- Kagome-only (not generic simplex lattices)
- Simple update only (no AD optimization of PESS tensors)
- AD can be done on the coarse-grained iPEPS tensor via existing `optimize_gs_ad`
- Existing example code (`kagome_xxz_pess.py`, `kagome_xxz_spin1_pess.py`) provides the reference implementation
