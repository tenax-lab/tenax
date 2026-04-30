# General Lattice Abstraction — Design

**Status:** COMPLETED — Lattice, Bond, ctm_multisite, and built-in factories (square, checkerboard, honeycomb, triangular, kagome) all merged.

## Goal

Add a `Lattice` class to `tenax.core` that describes 2D periodic lattice geometries declaratively, with built-in factories for common lattices and a new `ctm_multisite()` entry point for 3+ site unit cells.

## Architecture

A frozen `Lattice` dataclass encodes unit cell sites, bonds, and a neighbor map compatible with the existing multisite CTM sweep machinery. Factory functions provide pre-configured lattices (square, checkerboard, honeycomb, triangular, kagome). A new `ctm_multisite()` function in the algorithms layer accepts a `Lattice` + dict of site tensors and runs CTMRG to convergence.

## Components

### `tenax.core.lattice`

```python
@dataclass(frozen=True)
class Bond:
    site_i: str
    site_j: str
    direction: str  # "horizontal", "vertical", "diagonal", etc.

@dataclass(frozen=True)
class Lattice:
    sites: tuple[str, ...]
    bonds: tuple[Bond, ...]
    neighbor_map: dict[str, dict[str, str]]  # site → {left/right/top/bottom → neighbor_site}
```

`neighbor_map` uses the 4 cardinal directions that the existing CTM move functions expect: `"left"`, `"right"`, `"top"`, `"bottom"`.

### Factory functions

All in `tenax.core.lattice`:

- `square()` → 1 site, self-neighbors in all directions
- `checkerboard()` → 2 sites (a, b) alternating
- `honeycomb()` → 2 sites mapped to square lattice
- `triangular()` → 1 site with diagonal bonds encoded via neighbor map
- `kagome()` → 3+ sites mapped to rectangular unit cell on square lattice

### CTM integration

New public function in `ipeps_ctm_convergence.py`:

```python
def ctm_multisite(
    site_tensors: dict[str, jax.Array],
    lattice: Lattice,
    config: CTMConfig,
) -> dict[str, CTMEnvironment]:
```

Translates `Lattice.neighbor_map` (string-keyed) to the coordinate-based `neighbors` dict that `_ctm_tensor_sweep_multisite` expects, then delegates to the existing sweep machinery.

`ctm()` and `ctm_2site()` remain unchanged. `ctm_multisite()` is for 3+ site unit cells.

### Hamiltonian

Gates are separate from the lattice — the user passes gates externally, matching tenax's current convention. The `Lattice` only defines geometry.

## File layout

- `src/tenax/core/lattice.py` — `Lattice`, `Bond`, 5 factory functions
- `src/tenax/core/__init__.py` — export `Lattice`, `Bond`, factories
- `src/tenax/__init__.py` — re-export
- `src/tenax/algorithms/ipeps_ctm_convergence.py` — `ctm_multisite()`
- `src/tenax/algorithms/__init__.py` — export `ctm_multisite`

## Testing

- Unit tests for each factory: correct sites, bonds, neighbor_map structure
- Integration: `ctm_multisite()` with `checkerboard()` matches `ctm_2site()`
- Integration: `ctm_multisite()` with `kagome()` runs to completion
