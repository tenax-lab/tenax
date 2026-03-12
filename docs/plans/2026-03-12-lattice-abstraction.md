# Lattice Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `Lattice` class to `tenax.core` that describes 2D periodic lattice geometries declaratively, with built-in factories for common lattices and a new `ctm_multisite()` entry point for 3+ site unit cells.

**Architecture:** A frozen `Lattice` dataclass encodes unit cell sites, bonds, and a neighbor map compatible with the existing multisite CTM sweep machinery. Factory functions provide pre-configured lattices (square, checkerboard, honeycomb, triangular, kagome). A new `ctm_multisite()` function in `_ctm_tensor_convergence.py` accepts a `Lattice` + dict of site tensors and runs CTMRG to convergence via the existing `_ctm_tensor_multisite()`.

**Tech Stack:** Python dataclasses, JAX, existing tenax CTM infrastructure (Tensor protocol)

---

### Task 1: `Bond` and `Lattice` Dataclasses

**Files:**
- Create: `src/tenax/core/lattice.py`
- Test: `tests/test_lattice.py`

**Step 1: Write the failing test**

In `tests/test_lattice.py`:

```python
"""Tests for tenax.core.lattice — Lattice geometry abstraction."""

import pytest

from tenax.core.lattice import Bond, Lattice


class TestBondDataclass:
    def test_creation(self):
        b = Bond(site_i="a", site_j="b", direction="horizontal")
        assert b.site_i == "a"
        assert b.site_j == "b"
        assert b.direction == "horizontal"

    def test_frozen(self):
        b = Bond(site_i="a", site_j="b", direction="horizontal")
        with pytest.raises(AttributeError):
            b.site_i = "c"


class TestLatticeDataclass:
    def test_creation(self):
        bond = Bond("a", "b", "horizontal")
        lattice = Lattice(
            sites=("a", "b"),
            bonds=(bond,),
            neighbor_map={
                "a": {"left": "b", "right": "b", "top": "b", "bottom": "b"},
                "b": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
            },
        )
        assert lattice.sites == ("a", "b")
        assert len(lattice.bonds) == 1
        assert lattice.neighbor_map["a"]["left"] == "b"

    def test_frozen(self):
        lattice = Lattice(
            sites=("a",),
            bonds=(),
            neighbor_map={"a": {"left": "a", "right": "a", "top": "a", "bottom": "a"}},
        )
        with pytest.raises(AttributeError):
            lattice.sites = ("b",)

    def test_neighbor_map_has_four_directions(self):
        """Every site must map all 4 cardinal directions."""
        lattice = Lattice(
            sites=("a",),
            bonds=(),
            neighbor_map={"a": {"left": "a", "right": "a", "top": "a", "bottom": "a"}},
        )
        for site in lattice.sites:
            assert set(lattice.neighbor_map[site].keys()) == {
                "left", "right", "top", "bottom"
            }
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tenax.core.lattice'"

**Step 3: Write minimal implementation**

In `src/tenax/core/lattice.py`:

```python
"""2D periodic lattice geometry for iPEPS unit cells."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Bond", "Lattice"]


@dataclass(frozen=True)
class Bond:
    """A bond between two sites in the unit cell.

    Attributes:
        site_i:    Name of the first site.
        site_j:    Name of the second site.
        direction: Bond orientation (``"horizontal"``, ``"vertical"``,
                   ``"diagonal"``, etc.).
    """

    site_i: str
    site_j: str
    direction: str


@dataclass(frozen=True)
class Lattice:
    """Declarative description of a 2D periodic lattice unit cell.

    The ``neighbor_map`` encodes which site is reached from each site
    in each of the four cardinal directions used by CTM moves:
    ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.

    Attributes:
        sites:        Tuple of site names in the unit cell.
        bonds:        Tuple of Bond objects describing connectivity.
        neighbor_map: ``{site_name: {"left": ..., "right": ...,
                      "top": ..., "bottom": ...}}``
    """

    sites: tuple[str, ...]
    bonds: tuple[Bond, ...]
    neighbor_map: dict[str, dict[str, str]]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/tenax/core/lattice.py tests/test_lattice.py
git commit -m "feat: add Bond and Lattice dataclasses"
```

---

### Task 2: Factory Functions — `square()` and `checkerboard()`

**Files:**
- Modify: `src/tenax/core/lattice.py`
- Modify: `tests/test_lattice.py`

**Step 1: Write the failing tests**

Append to `tests/test_lattice.py`:

```python
from tenax.core.lattice import checkerboard, square


class TestSquareFactory:
    def test_single_site(self):
        lat = square()
        assert lat.sites == ("a",)

    def test_self_neighbors(self):
        lat = square()
        for direction in ("left", "right", "top", "bottom"):
            assert lat.neighbor_map["a"][direction] == "a"

    def test_no_bonds(self):
        """Square 1-site: all bonds are self-loops, so bonds tuple is empty."""
        lat = square()
        assert lat.bonds == ()


class TestCheckerboardFactory:
    def test_two_sites(self):
        lat = checkerboard()
        assert set(lat.sites) == {"a", "b"}

    def test_alternating_neighbors(self):
        lat = checkerboard()
        for direction in ("left", "right", "top", "bottom"):
            assert lat.neighbor_map["a"][direction] == "b"
            assert lat.neighbor_map["b"][direction] == "a"

    def test_bonds(self):
        lat = checkerboard()
        assert len(lat.bonds) == 2  # horizontal + vertical
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py::TestSquareFactory -v`
Expected: FAIL with "ImportError: cannot import name 'square'"

**Step 3: Write minimal implementation**

Append to `src/tenax/core/lattice.py`:

```python
__all__ = ["Bond", "Lattice", "square", "checkerboard"]


def square() -> Lattice:
    """1-site square lattice (self-neighbors in all directions)."""
    return Lattice(
        sites=("a",),
        bonds=(),
        neighbor_map={
            "a": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )


def checkerboard() -> Lattice:
    """2-site checkerboard lattice (A/B alternating)."""
    return Lattice(
        sites=("a", "b"),
        bonds=(
            Bond("a", "b", "horizontal"),
            Bond("a", "b", "vertical"),
        ),
        neighbor_map={
            "a": {"left": "b", "right": "b", "top": "b", "bottom": "b"},
            "b": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/tenax/core/lattice.py tests/test_lattice.py
git commit -m "feat: add square() and checkerboard() lattice factories"
```

---

### Task 3: Factory Functions — `honeycomb()`, `triangular()`, `kagome()`

**Files:**
- Modify: `src/tenax/core/lattice.py`
- Modify: `tests/test_lattice.py`

**Step 1: Write the failing tests**

Append to `tests/test_lattice.py`:

```python
from tenax.core.lattice import honeycomb, kagome, triangular


class TestHoneycombFactory:
    def test_two_sites(self):
        lat = honeycomb()
        assert set(lat.sites) == {"a", "b"}

    def test_neighbor_map_complete(self):
        lat = honeycomb()
        for site in lat.sites:
            assert set(lat.neighbor_map[site].keys()) == {
                "left", "right", "top", "bottom"
            }

    def test_has_bonds(self):
        lat = honeycomb()
        assert len(lat.bonds) >= 1


class TestTriangularFactory:
    def test_single_site(self):
        lat = triangular()
        assert lat.sites == ("a",)

    def test_self_neighbors(self):
        lat = triangular()
        for direction in ("left", "right", "top", "bottom"):
            assert lat.neighbor_map["a"][direction] == "a"

    def test_has_diagonal_bond(self):
        lat = triangular()
        directions = [b.direction for b in lat.bonds]
        assert "diagonal" in directions


class TestKagomeFactory:
    def test_three_sites(self):
        lat = kagome()
        assert len(lat.sites) == 3

    def test_neighbor_map_complete(self):
        lat = kagome()
        for site in lat.sites:
            assert set(lat.neighbor_map[site].keys()) == {
                "left", "right", "top", "bottom"
            }

    def test_all_neighbors_valid(self):
        lat = kagome()
        for site in lat.sites:
            for direction, nb in lat.neighbor_map[site].items():
                assert nb in lat.sites, f"{site}.{direction} -> {nb} not in sites"

    def test_has_bonds(self):
        lat = kagome()
        assert len(lat.bonds) >= 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py::TestHoneycombFactory -v`
Expected: FAIL with "ImportError: cannot import name 'honeycomb'"

**Step 3: Write minimal implementation**

Add to `src/tenax/core/lattice.py` (update `__all__` to include all 5 factories):

```python
__all__ = [
    "Bond",
    "Lattice",
    "square",
    "checkerboard",
    "honeycomb",
    "triangular",
    "kagome",
]


def honeycomb() -> Lattice:
    """2-site honeycomb lattice mapped to a square unit cell.

    Sites a and b alternate on a brick-wall pattern.  The neighbor map
    encodes the effective square-lattice connectivity.
    """
    return Lattice(
        sites=("a", "b"),
        bonds=(
            Bond("a", "b", "horizontal"),
            Bond("a", "b", "vertical"),
        ),
        neighbor_map={
            "a": {"left": "b", "right": "b", "top": "b", "bottom": "b"},
            "b": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )


def triangular() -> Lattice:
    """1-site triangular lattice with diagonal bonds.

    Mapped to a square lattice with an extra diagonal bond encoded
    via the bond list.  The neighbor map uses the standard 4 cardinal
    directions (all self-loops for a 1-site cell).
    """
    return Lattice(
        sites=("a",),
        bonds=(
            Bond("a", "a", "horizontal"),
            Bond("a", "a", "vertical"),
            Bond("a", "a", "diagonal"),
        ),
        neighbor_map={
            "a": {"left": "a", "right": "a", "top": "a", "bottom": "a"},
        },
    )


def kagome() -> Lattice:
    """3-site Kagome lattice mapped to a rectangular unit cell.

    Three sites (u, v, w) per unit cell, mapped onto a square-lattice
    CTM framework.  The neighbor map encodes which site is reached from
    each site in each cardinal direction for the CTM sweep.
    """
    return Lattice(
        sites=("u", "v", "w"),
        bonds=(
            Bond("u", "v", "horizontal"),
            Bond("u", "w", "vertical"),
            Bond("v", "w", "diagonal"),
        ),
        neighbor_map={
            "u": {"left": "w", "right": "v", "top": "w", "bottom": "v"},
            "v": {"left": "u", "right": "w", "top": "u", "bottom": "w"},
            "w": {"left": "v", "right": "u", "top": "v", "bottom": "u"},
        },
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/tenax/core/lattice.py tests/test_lattice.py
git commit -m "feat: add honeycomb, triangular, and kagome lattice factories"
```

---

### Task 4: Export `Lattice`, `Bond`, and Factories

**Files:**
- Modify: `src/tenax/core/__init__.py`
- Modify: `src/tenax/__init__.py`
- Modify: `src/tenax/algorithms/__init__.py` (only if `ctm_multisite` added later)

**Step 1: Write the failing test**

Append to `tests/test_lattice.py`:

```python
class TestExports:
    def test_importable_from_core(self):
        from tenax.core import Bond, Lattice

    def test_importable_from_tenax(self):
        from tenax import Bond, Lattice, checkerboard, kagome, square

    def test_factories_from_tenax(self):
        from tenax import checkerboard, honeycomb, kagome, square, triangular
        # Verify they return Lattice instances
        from tenax.core.lattice import Lattice as L
        assert isinstance(square(), L)
        assert isinstance(checkerboard(), L)
        assert isinstance(honeycomb(), L)
        assert isinstance(triangular(), L)
        assert isinstance(kagome(), L)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py::TestExports -v`
Expected: FAIL with "ImportError: cannot import name 'Bond' from 'tenax.core'"

**Step 3: Write minimal implementation**

In `src/tenax/core/__init__.py`, add imports and `__all__` entries:

```python
from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)
```

Add to `__all__`: `"Bond"`, `"Lattice"`, `"square"`, `"checkerboard"`, `"honeycomb"`, `"triangular"`, `"kagome"`

In `src/tenax/__init__.py`, add imports:

```python
from tenax.core.lattice import (
    Bond,
    Lattice,
    checkerboard,
    honeycomb,
    kagome,
    square,
    triangular,
)
```

Add to `__all__`: `"Bond"`, `"Lattice"`, `"square"`, `"checkerboard"`, `"honeycomb"`, `"triangular"`, `"kagome"`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/tenax/core/__init__.py src/tenax/__init__.py tests/test_lattice.py
git commit -m "feat: export Lattice, Bond, and factory functions from tenax"
```

---

### Task 5: `ctm_multisite()` — Lattice-Based CTM Entry Point

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py`
- Test: `tests/test_lattice.py`

**Context:** The existing `_ctm_tensor_multisite()` function (line 190 of `_ctm_tensor_convergence.py`) accepts coordinate-keyed dicts: `dict[Coord, Tensor]` for site tensors and `dict[Coord, dict[str, Coord]]` for neighbors. The new `ctm_multisite()` wraps this with a string-keyed `Lattice` interface.

The coordinate assignment is: enumerate `lattice.sites` and assign `(i, 0)` to each site, matching the pattern used by `CHECKERBOARD_NEIGHBORS` where `(0, 0)` and `(1, 0)` are the two sites.

**Step 1: Write the failing test**

Append to `tests/test_lattice.py`:

```python
import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
from tenax.core.lattice import Lattice, checkerboard
from tenax.core.tensor import DenseTensor
from tenax.core.index import TensorIndex, FlowDirection
from tenax.core.symmetry import U1Symmetry
import numpy as np


def _make_random_site_tensor(key, D=2, d=2):
    """Helper: build a random DenseTensor site tensor with 5 legs (u,d,l,r,phys)."""
    sym = U1Symmetry()
    charges = np.zeros(D, dtype=np.int32)
    phys_charges = np.zeros(d, dtype=np.int32)
    indices = (
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="u"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="d"),
        TensorIndex(sym, charges.copy(), FlowDirection.IN, label="l"),
        TensorIndex(sym, charges.copy(), FlowDirection.OUT, label="r"),
        TensorIndex(sym, phys_charges.copy(), FlowDirection.IN, label="phys"),
    )
    data = jax.random.normal(key, shape=(D, D, D, D, d))
    return DenseTensor(data, indices)


class TestCtmMultisite:
    def test_checkerboard_returns_dict(self):
        """ctm_multisite with checkerboard returns envs keyed by site name."""
        lat = checkerboard()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        A = _make_random_site_tensor(k1)
        B = _make_random_site_tensor(k2)
        envs = ctm_multisite(
            {"a": A, "b": B},
            lat,
            chi=4,
            max_iter=5,
            conv_tol=1e-6,
        )
        assert set(envs.keys()) == {"a", "b"}

    def test_checkerboard_matches_ctm_tensor_2site(self):
        """ctm_multisite with checkerboard should match ctm_tensor_2site."""
        from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        A = _make_random_site_tensor(k1)
        B = _make_random_site_tensor(k2)

        lat = checkerboard()
        envs_multi = ctm_multisite(
            {"a": A, "b": B},
            lat,
            chi=4,
            max_iter=30,
            conv_tol=1e-10,
        )

        env_A, env_B = ctm_tensor_2site(
            A, B,
            chi=4,
            max_iter=30,
            conv_tol=1e-10,
        )

        # Compare corner singular values (environment-independent measure)
        sv_multi_a = jnp.linalg.svd(envs_multi["a"].C1.todense(), compute_uv=False)
        sv_direct_a = jnp.linalg.svd(env_A.C1.todense(), compute_uv=False)
        sv_multi_a = sv_multi_a / jnp.sum(sv_multi_a)
        sv_direct_a = sv_direct_a / jnp.sum(sv_direct_a)
        assert jnp.allclose(sv_multi_a, sv_direct_a, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py::TestCtmMultisite::test_checkerboard_returns_dict -v`
Expected: FAIL with "ImportError: cannot import name 'ctm_multisite'"

**Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/_ctm_tensor_convergence.py`:

```python
from tenax.core.lattice import Lattice
```

Then add the function (after `ctm_tensor_2site`):

```python
def ctm_multisite(
    site_tensors: dict[str, Tensor],
    lattice: Lattice,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    renormalize: bool = True,
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
) -> dict[str, CTMTensorEnv]:
    """Run multisite CTM to convergence for an arbitrary lattice.

    Translates the string-keyed ``Lattice.neighbor_map`` to the
    coordinate-based format expected by ``_ctm_tensor_multisite()``,
    then maps the results back to site names.

    Use this for unit cells with 3+ sites.  For 1- or 2-site cells,
    prefer ``ctm_tensor()`` or ``ctm_tensor_2site()`` which are
    optimized for those cases.

    Args:
        site_tensors:      ``{site_name: Tensor}`` for each site in
                           ``lattice.sites``.
        lattice:           A :class:`~tenax.core.lattice.Lattice` describing
                           the unit cell geometry.
        chi:               Environment bond dimension.
        max_iter:          Maximum CTM iterations.
        conv_tol:          Convergence tolerance on corner singular values.
        renormalize:       Renormalize environment at each step.
        projector_method:  ``"eigh"`` or ``"qr"``.
        qr_warmup_steps:   Number of eigh warm-up sweeps before QR kicks in.

    Returns:
        ``{site_name: CTMTensorEnv}`` — converged environments.
    """
    # Map site names to coordinates: site_i -> (i, 0)
    name_to_coord: dict[str, Coord] = {
        name: (i, 0) for i, name in enumerate(lattice.sites)
    }
    coord_to_name: dict[Coord, str] = {v: k for k, v in name_to_coord.items()}

    # Translate site_tensors to coordinate keys
    coord_tensors: dict[Coord, Tensor] = {
        name_to_coord[name]: t for name, t in site_tensors.items()
    }

    # Translate neighbor_map to coordinate keys
    coord_neighbors: dict[Coord, dict[str, Coord]] = {
        name_to_coord[name]: {
            direction: name_to_coord[nb_name]
            for direction, nb_name in neighbors.items()
        }
        for name, neighbors in lattice.neighbor_map.items()
    }

    # Delegate to existing multisite CTM
    coord_envs = _ctm_tensor_multisite(
        coord_tensors,
        coord_neighbors,
        chi,
        max_iter,
        conv_tol,
        renormalize,
        projector_method,
        qr_warmup_steps,
    )

    # Map results back to site names
    return {coord_to_name[c]: env for c, env in coord_envs.items()}
```

Add `"ctm_multisite"` to the `__all__` list in `_ctm_tensor_convergence.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py::TestCtmMultisite -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_lattice.py
git commit -m "feat: add ctm_multisite() lattice-based CTM entry point"
```

---

### Task 6: Export `ctm_multisite` and Update Re-Export Shims

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor.py` (re-export shim)
- Modify: `src/tenax/algorithms/__init__.py`
- Modify: `src/tenax/__init__.py`
- Modify: `tests/test_lattice.py`

**Step 1: Write the failing test**

Append to `tests/test_lattice.py`:

```python
class TestCtmMultisiteExports:
    def test_importable_from_algorithms(self):
        from tenax.algorithms import ctm_multisite  # noqa: F401

    def test_importable_from_tenax(self):
        from tenax import ctm_multisite  # noqa: F401

    def test_importable_from_ctm_tensor_shim(self):
        from tenax.algorithms._ctm_tensor import ctm_multisite  # noqa: F401
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lattice.py::TestCtmMultisiteExports -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

In `src/tenax/algorithms/__init__.py`:

Add import:
```python
from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
```

Add `"ctm_multisite"` to `__all__`.

In `src/tenax/__init__.py`:

Add import:
```python
from tenax.algorithms._ctm_tensor_convergence import ctm_multisite
```

Add `"ctm_multisite"` to `__all__` under the `# Standard CTM (Tensor protocol)` section.

In `src/tenax/algorithms/_ctm_tensor.py` (the re-export shim):

The shim uses `from tenax.algorithms._ctm_tensor_convergence import *` which will pick up `ctm_multisite` automatically since it's in `__all__`. No change needed.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lattice.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/__init__.py src/tenax/__init__.py tests/test_lattice.py
git commit -m "feat: export ctm_multisite from tenax and algorithms"
```

---

### Task 7: Documentation Updates

**Files:**
- Modify: `docs/api/algorithms.rst`
- Modify: `README.md`

**Step 1: Update Sphinx docs**

In `docs/api/algorithms.rst`, add under the iPEPS section (after the `ipeps_ctm.ctm_2site` entry):

```rst
.. autofunction:: tenax.algorithms._ctm_tensor_convergence.ctm_multisite
```

Also add:

```rst
Lattice
-------

.. autoclass:: tenax.core.lattice.Bond
   :members:
   :no-index:

.. autoclass:: tenax.core.lattice.Lattice
   :members:
   :no-index:

.. autofunction:: tenax.core.lattice.square

.. autofunction:: tenax.core.lattice.checkerboard

.. autofunction:: tenax.core.lattice.honeycomb

.. autofunction:: tenax.core.lattice.triangular

.. autofunction:: tenax.core.lattice.kagome
```

**Step 2: Update README.md**

Add a bullet under the iPEPS features section mentioning `Lattice` abstraction with built-in geometries. Example:

```
- **Lattice geometries**: Declarative `Lattice` class with built-in square, checkerboard, honeycomb, triangular, and kagome factories; `ctm_multisite()` for 3+ site unit cells
```

**Step 3: Verify docs build**

Run: `cd docs && make html`
Expected: Build succeeds without warnings for the new entries.

**Step 4: Commit**

```bash
git add docs/api/algorithms.rst README.md
git commit -m "docs: add Lattice and ctm_multisite to API docs and README"
```

---

### Task 8: Final Verification

**Step 1: Run all core tests**

Run: `uv run pytest -m core -v`
Expected: All tests pass, including all new lattice tests.

**Step 2: Run full test suite (non-slow)**

Run: `uv run pytest -m "not slow" -v`
Expected: All tests pass.

**Step 3: Verify imports work end-to-end**

Run:
```bash
uv run python -c "
from tenax import Lattice, Bond, square, checkerboard, honeycomb, triangular, kagome, ctm_multisite
print('Lattice:', Lattice)
print('square():', square())
print('kagome():', kagome())
print('ctm_multisite:', ctm_multisite)
print('All imports OK')
"
```
Expected: All imports succeed, objects print correctly.
