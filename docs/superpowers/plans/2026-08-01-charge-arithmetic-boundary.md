# Charge Arithmetic Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BaseSymmetry` the only place in Tenax that inverts or combines a charge, fixing Z_n bond mislabelling (#733), Z_n `contract` silently returning zeros, and `ProductSymmetry` being unusable.

**Architecture:** Add `flow_charge` and `canonicalize` to `BaseSymmetry`, plus a `net_charge(indices, key)` helper in `core/index.py` seeded with `identity()`. `flow_charge` is an involution on canonical representatives; `TensorIndex.__post_init__` guarantees canonical representatives by canonicalising and merging duplicate sectors. Then replace all eight hand-rolled `int(flow) * q` sites with these.

**Tech Stack:** Python 3.11+, NumPy (charge algebra is host-side, never traced), JAX (block data only), pytest with `core`/`algorithm`/`slow` markers auto-applied by `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-08-01-charge-arithmetic-boundary-design.md`
**Issue:** #734 (umbrella); #733 closed by Task 2.

---

## Background an implementer needs

A `SymmetricTensor` stores blocks keyed by one charge per leg. A block is valid when the
flow-weighted charges of all legs fuse to the symmetry's identity. Existing code writes
that as:

```python
total = sum(int(idx.flow) * int(q) for idx, q in zip(tensor.indices, key))
if total != sym.identity(): ...
```

This encodes two assumptions that are true **only for U(1)**:

1. the group inverse is integer negation (`int(flow) * q` where `flow` is `+1`/`-1`)
2. the group operation is integer addition (the bare `sum`)

For `Z_n` they are accidentally true whenever at least two legs fuse afterwards, because
`ZnSymmetry.fuse` applies `% n`. They fail exactly when nothing fuses — which is the
single-leg case in #733. For `ProductSymmetry`, charges are bit-packed
(`encode = (q2 << 16) | (q1 & 0xFFFF)`), so negating the packed integer is meaningless and
the assumptions never hold.

`FlowDirection` is an `IntEnum` with `IN = 1` and `OUT = -1`.

**Charge labels are not observable.** Every physical observable is a scalar obtained by
collapsing all legs. A stored charge label is a basis/sector convention, so rewriting the
Z₂ partner of `1` from `-1` to `1` changes nothing physical. What must hold is that
operations **compose** — and today mismatched labels make `contract` silently zero the
non-matching sectors rather than raising.

### Verification commands

```bash
# fast gate (what required CI runs)
JAX_PLATFORMS=cpu uv run pytest -m core -q

# a single file
JAX_PLATFORMS=cpu uv run pytest tests/test_symmetry.py -q
```

All five test files touched by this plan (`test_symmetry.py`, `test_index.py`,
`test_tensor.py`, `test_linalg.py`, `test_contraction.py`) are already registered as
`core` in `tests/conftest.py:_FILE_MARKERS`, so new tests land in the required CI gate
without a conftest change. **Do not create new test files** — add to these.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/tenax/core/symmetry.py` | Owns all charge algebra. Gains `flow_charge`, `canonicalize`; `is_conserved` rewritten on top of them. | 1 |
| `src/tenax/core/index.py` | Gains `net_charge(indices, key)` — the one way to evaluate a block's conservation law. `TensorIndex.__post_init__` canonicalises and merges. | 1, 2 |
| `src/tenax/core/tensor.py` | `_validate` and `_compute_valid_blocks` consume `net_charge` / `flow_charge`. | 2 |
| `src/tenax/linalg.py` | Four sites (80, 121, 605, 2109) consume `net_charge`. | 2 |
| `src/tenax/contraction/contractor.py` | Target inference consumes `net_charge` and fuses targets instead of adding them. | 3 |
| `src/tenax/algorithms/dmrg.py` | Target inference consumes `net_charge`. | 3 |
| `src/tenax/algorithms/_tensor_utils.py` | `_compute_fused_sectors` drops its hand-rolled `% n`. | 4 |

`core/index.py` imports only `core/symmetry.py`; `core/tensor.py` imports `core/index.py`.
So `net_charge` in `index.py` is importable by `tensor.py`, `linalg.py`, `contractor.py`
and `dmrg.py` with no import cycle. **Do not put `net_charge` in `core/_tensor_utils.py`**
— that module imports `core/tensor.py`, which would make `_validate` circular.

---

## Task 1: The boundary API (PR 1 — behaviourally inert)

**Files:**
- Modify: `src/tenax/core/symmetry.py` (add two methods to `BaseSymmetry`; override in `U1Symmetry` and `FermionicU1`; rewrite `is_conserved` at lines 156-179)
- Modify: `src/tenax/core/index.py` (add module-level `net_charge`)
- Test: `tests/test_symmetry.py`, `tests/test_index.py`

- [ ] **Step 1: Write the failing tests for the symmetry boundary**

Append to `tests/test_symmetry.py`. Note `ProductSymmetry` is included in every case list —
it is the symmetry the current code gets wrong, so excluding it defeats the purpose.

```python
import numpy as np
import pytest

from tenax.core.symmetry import (
    FermionicU1,
    FermionParity,
    ProductSymmetry,
    U1Symmetry,
    ZnSymmetry,
)

_BOUNDARY_CASES = [
    ("U1", U1Symmetry(), [-2, -1, 0, 1, 2]),
    ("Z2", ZnSymmetry(2), [0, 1]),
    ("Z3", ZnSymmetry(3), [0, 1, 2]),
    ("Z4", ZnSymmetry(4), [0, 1, 2, 3]),
    ("FermionParity", FermionParity(), [0, 1]),
    ("FermionicU1", FermionicU1(), [-2, -1, 0, 1, 2]),
    (
        "Prod(Z2,U1)",
        ProductSymmetry(ZnSymmetry(2), U1Symmetry()),
        [ProductSymmetry.encode(a, b) for a in (0, 1) for b in (-2, -1, 0, 1, 2)],
    ),
    (
        "Prod(Z2,Z3)",
        ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
        [ProductSymmetry.encode(a, b) for a in (0, 1) for b in (0, 1, 2)],
    ),
]
_BOUNDARY_IDS = [c[0] for c in _BOUNDARY_CASES]


@pytest.mark.parametrize("name,sym,sectors", _BOUNDARY_CASES, ids=_BOUNDARY_IDS)
def test_canonicalize_fixes_canonical_sectors_and_is_idempotent(name, sym, sectors):
    secs = np.array(sectors, dtype=np.int32)
    once = sym.canonicalize(secs)
    assert np.array_equal(once, secs), f"{name}: canonical input was rewritten"
    assert np.array_equal(sym.canonicalize(once), once), f"{name}: not idempotent"


@pytest.mark.parametrize("name,sym,sectors", _BOUNDARY_CASES, ids=_BOUNDARY_IDS)
def test_flow_charge_is_an_involution_on_canonical_sectors(name, sym, sectors):
    secs = np.array(sectors, dtype=np.int32)
    twice = sym.flow_charge(-1, sym.flow_charge(-1, secs))
    assert np.array_equal(twice, secs), f"{name}: OUT twice is not the identity map"
    assert np.array_equal(sym.flow_charge(1, secs), secs), f"{name}: IN altered charges"


@pytest.mark.parametrize("name,sym,sectors", _BOUNDARY_CASES, ids=_BOUNDARY_IDS)
def test_fuse_with_dual_gives_identity(name, sym, sectors):
    secs = np.array(sectors, dtype=np.int32)
    assert np.all(sym.fuse(secs, sym.dual(secs)) == sym.identity()), name


@pytest.mark.parametrize("name,sym,sectors", _BOUNDARY_CASES, ids=_BOUNDARY_IDS)
def test_closed_form_solves_the_last_leg_charge(name, sym, sectors):
    """``q = flow_charge(flow, fuse(target, dual(partial)))`` inverts conservation.

    This is the property that lets ``_compute_valid_blocks`` drop its
    ``n_values() is None`` split: it holds for every abelian group, not just U(1).
    """
    secs = np.array(sectors, dtype=np.int32)
    for flow in (1, -1):
        for partial in secs:
            for target in secs:
                p = np.array([partial], dtype=np.int32)
                t = np.array([target], dtype=np.int32)
                q_last = sym.flow_charge(flow, sym.fuse(t, sym.dual(p)))
                got = int(sym.fuse(p, sym.flow_charge(flow, q_last))[0])
                assert got == int(t[0]), f"{name}: flow={flow} partial={partial}"


def test_canonicalize_maps_negative_zn_representatives():
    sym = ZnSymmetry(3)
    got = sym.canonicalize(np.array([-1, -2, 0], dtype=np.int32))
    assert np.array_equal(got, np.array([2, 1, 0], dtype=np.int32))


def test_is_conserved_accepts_a_conserving_product_symmetry_block():
    """The IN/OUT pair that ``int(flow) * q`` rejects for bit-packed charges."""
    ps = ProductSymmetry(ZnSymmetry(2), U1Symmetry())
    q = ProductSymmetry.encode(1, 1)
    assert ps.is_conserved([q, q], [1, -1])
    assert not ps.is_conserved([q, ProductSymmetry.encode(0, 1)], [1, -1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_symmetry.py -k "canonicalize or flow_charge or closed_form or is_conserved" -q`

Expected: FAIL — `AttributeError: 'U1Symmetry' object has no attribute 'canonicalize'` for the
new-method tests, and `test_is_conserved_accepts_a_conserving_product_symmetry_block` failing
on the first assert.

- [ ] **Step 3: Add `flow_charge` and `canonicalize` to `BaseSymmetry`**

In `src/tenax/core/symmetry.py`, insert both methods into `BaseSymmetry` immediately after
`fuse_many` (which ends at line 87, before the `braiding_style` property):

```python
    def flow_charge(self, flow: int, charges: np.ndarray) -> np.ndarray:
        """Return the flow-weighted effective charge of a leg.

        An IN leg (flow ``+1``) contributes its charge unchanged; an OUT leg
        (flow ``-1``) contributes the group inverse.

        This is the only sanctioned way to weight a charge by a flow.
        ``int(flow) * charge`` hard-codes "the group inverse is integer
        negation": true for U(1), true for ``Z_n`` only because ``fuse``
        reduces mod ``n`` afterwards, and meaningless for the bit-packed
        charges of :class:`ProductSymmetry` (#734).

        On canonical representatives this is an involution, which is what makes
        ``q = flow_charge(flow, fuse(target, dual(partial)))`` a valid closed
        form for the last leg of a conservation law in any abelian group.

        Args:
            flow:    ``+1`` (IN) or ``-1`` (OUT); a ``FlowDirection`` works too.
            charges: Integer charge array.

        Returns:
            Effective charge array of the same shape.
        """
        charges = np.asarray(charges, dtype=np.int32)
        return charges if int(flow) > 0 else self.dual(charges)

    def canonicalize(self, charges: np.ndarray) -> np.ndarray:
        """Return the canonical representative of each charge.

        Fusing against the identity applies whatever reduction the group
        defines — ``% n`` for ``Z_n``, component-wise reduction for
        :class:`ProductSymmetry`, nothing for U(1).  Charge representatives are
        a basis convention and not observable, so rewriting them is free; what
        is *not* free is letting two representatives of one sector coexist,
        because label equality is how blocks are paired during contraction.

        Args:
            charges: Integer charge array.

        Returns:
            Canonical charge array of the same shape.
        """
        charges = np.asarray(charges, dtype=np.int32)
        return self.fuse(np.full_like(charges, self.identity()), charges)
```

- [ ] **Step 4: Override `canonicalize` in the two symmetries where it is a no-op**

`TensorIndex.__post_init__` will call `canonicalize` on every index construction (Task 2),
and the codebase builds millions of them. For U(1)-style groups every integer already *is*
its own canonical representative, so skip the allocation.

Add to `U1Symmetry` (after `identity`, before `n_values`):

```python
    def canonicalize(self, charges: np.ndarray) -> np.ndarray:
        # Every integer is its own canonical U(1) representative.
        return np.asarray(charges, dtype=np.int32)
```

Add the identical method to `FermionicU1` (after its `identity`, before `n_values`).

- [ ] **Step 5: Rewrite `is_conserved` on top of the new primitives**

Replace the body of `BaseSymmetry.is_conserved` (currently `src/tenax/core/symmetry.py:156-180`,
the version ending `return net == target`) with:

```python
    def is_conserved(
        self,
        charges_per_leg: list[np.ndarray],
        flows: list[int],
        target: int | None = None,
    ) -> bool:
        """Check if a single charge combination satisfies conservation.

        Args:
            charges_per_leg: List of scalar charge values per leg.
            flows: List of +1 (IN) or -1 (OUT) per leg.
            target: Required net charge; defaults to identity().

        Returns:
            True if the net charge equals target.
        """
        if target is None:
            target = self.identity()
        effective = [np.array([self.identity()], dtype=np.int32)]
        effective.extend(
            self.flow_charge(f, np.array([int(q)], dtype=np.int32))
            for f, q in zip(flows, charges_per_leg)
        )
        net = int(self.fuse_many(effective)[0])
        want = int(self.canonicalize(np.array([int(target)], dtype=np.int32))[0])
        return net == want
```

The old `net % n_values()` reduction is gone: `fuse` already reduces, and `% n` was wrong
for `ProductSymmetry`, whose `n_values()` is a cardinality (`2 * 3 == 6`) rather than a
modulus for the packed integer.

- [ ] **Step 6: Run the symmetry tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_symmetry.py -q`

Expected: PASS, all cases including both `ProductSymmetry` variants.

- [ ] **Step 7: Write the failing test for `net_charge`**

Append to `tests/test_index.py`:

```python
import numpy as np

from tenax.core.index import FlowDirection, TensorIndex, net_charge
from tenax.core.symmetry import ProductSymmetry, U1Symmetry, ZnSymmetry


def test_net_charge_reduces_a_single_out_leg():
    """The #733 case: one OUT leg, nothing to fuse against, so no reduction ran."""
    sym = ZnSymmetry(2)
    idx = TensorIndex.from_charges(
        sym, np.array([0, 1], dtype=np.int32), FlowDirection.OUT, label="a"
    )
    assert net_charge((idx,), (1,)) == 1  # not -1


def test_net_charge_agrees_between_one_leg_and_two_legs():
    sym = ZnSymmetry(3)
    out = TensorIndex.from_charges(
        sym, np.array([0, 1, 2], dtype=np.int32), FlowDirection.OUT, label="a"
    )
    trivial = TensorIndex.from_charges(
        sym, np.array([0], dtype=np.int32), FlowDirection.IN, label="b"
    )
    for q in (0, 1, 2):
        assert net_charge((out,), (q,)) == net_charge((out, trivial), (q, 0))


def test_net_charge_is_identity_for_a_conserving_product_symmetry_block():
    ps = ProductSymmetry(ZnSymmetry(2), U1Symmetry())
    q = ProductSymmetry.encode(1, 1)
    secs = np.array([ProductSymmetry.encode(0, 0), q], dtype=np.int32)
    a = TensorIndex.from_charges(ps, secs, FlowDirection.IN, label="a")
    b = TensorIndex.from_charges(ps, secs, FlowDirection.OUT, label="b")
    assert net_charge((a, b), (q, q)) == ps.identity()


def test_net_charge_matches_plain_summation_for_u1():
    """U(1) is the case the old hand-rolled arithmetic got right; keep it right."""
    sym = U1Symmetry()
    a = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN, label="a"
    )
    b = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.OUT, label="b"
    )
    for qa in (-1, 0, 1):
        for qb in (-1, 0, 1):
            assert net_charge((a, b), (qa, qb)) == qa - qb
```

- [ ] **Step 8: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_index.py -k net_charge -q`

Expected: FAIL with `ImportError: cannot import name 'net_charge' from 'tenax.core.index'`.

- [ ] **Step 9: Add `net_charge` to `core/index.py`**

Append at module level in `src/tenax/core/index.py` (after the `TensorIndex` class):

```python
def net_charge(
    indices: Sequence[TensorIndex],
    key: Sequence[int],
) -> int:
    """Return the net fused charge of a block key, as a Python int.

    This is the only sanctioned way to evaluate a block's conservation law: a
    block is valid exactly when ``net_charge(indices, key) == symmetry.identity()``.

    ``sum(int(idx.flow) * int(q) for ...)`` is **not** equivalent. It assumes the
    group inverse is integer negation and the group operation is integer
    addition — true for U(1), accidentally true for ``Z_n`` whenever two or more
    legs fuse afterwards, and false for the bit-packed charges of
    :class:`~tenax.core.symmetry.ProductSymmetry` (#734).

    The fusion is seeded with ``identity()`` so that a rank-1 tensor is reduced
    exactly like a rank-N one. Without the seed, ``fuse_many`` of a single array
    returns it unreduced and a lone OUT leg yields a non-canonical
    representative (#733).

    Args:
        indices: Tensor indices, one per leg.
        key:     One charge per leg, in the same order.

    Returns:
        The fused net charge.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    if len(indices) == 0:
        raise ValueError("net_charge requires at least one index")
    sym = indices[0].symmetry
    effective = [np.array([sym.identity()], dtype=np.int32)]
    effective.extend(
        sym.flow_charge(idx.flow, np.array([int(q)], dtype=np.int32))
        for idx, q in zip(indices, key)
    )
    return int(sym.fuse_many(effective)[0])
```

Add `Sequence` to the imports at the top of `src/tenax/core/index.py`:

```python
from collections.abc import Sequence
```

- [ ] **Step 10: Run the index tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_index.py -q`

Expected: PASS.

- [ ] **Step 11: Confirm Task 1 changed no existing behaviour**

Run: `JAX_PLATFORMS=cpu uv run pytest -m core -q`

Expected: PASS with no new failures. Task 1 only *adds* callable surface — `is_conserved`
had zero call sites before this change, so rewriting it cannot move any existing behaviour.
If anything fails here, it is a genuine regression; stop and diagnose before continuing.

- [ ] **Step 12: Commit**

```bash
git add src/tenax/core/symmetry.py src/tenax/core/index.py tests/test_symmetry.py tests/test_index.py
git commit -m "feat(#734): charge-arithmetic boundary on BaseSymmetry

Add flow_charge and canonicalize to BaseSymmetry plus a net_charge helper
in core/index.py, seeded with identity so a rank-1 tensor reduces exactly
like a rank-N one. Rewrite the never-called is_conserved on top of them,
dropping its % n_values() reduction (wrong for the bit-packed
ProductSymmetry, whose n_values is a cardinality, not a modulus).

flow_charge is an involution on canonical representatives; verified across
U1, Z2, Z3, Z4, FermionParity, FermionicU1 and both ProductSymmetry
variants, along with the closed form that Task 2 relies on.

Behaviourally inert: pure addition plus a dead method.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Canonical indices, `_validate`, `_compute_valid_blocks`, `linalg.py` (PR 2 — closes #733)

**Prerequisite:** PR #729 must be merged and `origin/main` merged into this branch, so that
`tests/test_ctm_root_implicit_sym_sectors.py::test_zn_same_flow_bonds_carry_the_library_s_own_representative`
exists to be updated in Step 11.

**Files:**
- Modify: `src/tenax/core/index.py` (`TensorIndex.__post_init__`)
- Modify: `src/tenax/core/tensor.py` (`_compute_valid_blocks` lines 162-274; `_validate` lines 727-744)
- Modify: `src/tenax/linalg.py` (lines 80-92, 95-130, 601-607, 2105-2111)
- Modify: `tests/test_ctm_root_implicit_sym_sectors.py:417-470`
- Test: `tests/test_index.py`, `tests/test_linalg.py`

- [ ] **Step 1: Write the failing tests for index canonicalisation**

Append to `tests/test_index.py`:

```python
def test_index_canonicalises_and_merges_duplicate_representatives():
    """Z2 ``-1`` and ``1`` are one sector written two ways; they must merge."""
    sym = ZnSymmetry(2)
    idx = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN, label="x"
    )
    assert np.array_equal(idx.sectors, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(idx.multiplicities, np.array([1, 2], dtype=np.int32))
    assert idx.dim == 3
    assert idx.n_sectors == 2
    assert idx.multiplicity(1) == 2
    assert not idx.has_sector(-1)


def test_index_charges_stay_consistent_with_sectors_after_canonicalisation():
    """``charges`` must not desynchronise from ``sectors``/``multiplicities``."""
    sym = ZnSymmetry(2)
    idx = TensorIndex.from_charges(
        sym, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN, label="x"
    )
    charges = idx.charges
    assert len(charges) == idx.dim
    for q in idx.sectors:
        assert int(np.sum(charges == q)) == idx.multiplicity(int(q))


def test_index_dual_is_an_involution():
    """``dual()`` used to sort without merging, emitting sectors=[0, 1, 1]."""
    sym = ZnSymmetry(3)
    idx = TensorIndex.from_charges(
        sym, np.array([0, 1, 2], dtype=np.int32), FlowDirection.IN, label="x"
    )
    back = idx.dual().dual()
    assert np.array_equal(back.sectors, idx.sectors)
    assert np.array_equal(back.multiplicities, idx.multiplicities)
    assert back.flow == idx.flow


def test_index_sectors_are_always_sorted_and_unique():
    sym = ZnSymmetry(2)
    idx = TensorIndex(
        symmetry=sym,
        sectors=np.array([1, -1, 0], dtype=np.int32),
        multiplicities=np.array([2, 3, 4], dtype=np.int32),
        flow=FlowDirection.IN,
        label="x",
    )
    assert np.array_equal(idx.sectors, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(idx.multiplicities, np.array([4, 5], dtype=np.int32))
```

- [ ] **Step 2: Run to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_index.py -k "canonicalises or consistent or involution or sorted_and_unique" -q`

Expected: FAIL — `sectors` is still `[-1, 0, 1]`, `multiplicities` still `[1, 1, 1]`.

- [ ] **Step 3: Canonicalise and merge in `TensorIndex.__post_init__`**

In `src/tenax/core/index.py`, append to the end of `__post_init__` (after the existing
`multiplicities` dtype coercion):

```python
        # Canonicalise sectors through the symmetry and merge duplicates.
        #
        # Charge representatives are a basis convention, not physics, so
        # rewriting them is free.  But ``flow_charge`` is an involution only on
        # canonical representatives, and label *equality* is how blocks get
        # paired during contraction -- so two representatives of one sector
        # (Z2 ``-1`` and ``1``) must never coexist on a leg (#734).
        #
        # Every construction path funnels through __post_init__, so this also
        # repairs ``dual()``, which sorted without merging and could emit
        # sectors=[0, 1, 1] in violation of the sorted-unique invariant.
        canon = self.symmetry.canonicalize(self.sectors)
        uniq, inverse = np.unique(canon, return_inverse=True)
        if len(uniq) != len(canon) or not np.array_equal(canon, self.sectors):
            merged = np.zeros(len(uniq), dtype=np.int32)
            np.add.at(merged, inverse, self.multiplicities)
            object.__setattr__(self, "sectors", uniq.astype(np.int32))
            object.__setattr__(self, "multiplicities", merged)
```

`np.unique` returns sorted unique values plus an `inverse` mapping each original position
to its slot, so `np.add.at` sums the multiplicities of merged sectors. This handles
canonicalisation, duplicate merging and re-sorting in one path.

- [ ] **Step 4: Canonicalise the cached dense charges too**

`from_charges` stores the original dense charges in `_charges_cache` to preserve basis
ordering. If it is not canonicalised, `charges` desynchronises from `sectors` and
`np.where(idx.charges == q)` finds nothing.

In `src/tenax/core/index.py`, in `from_charges`, replace:

```python
        charges = np.asarray(charges, dtype=np.int32)
        if charges.ndim != 1:
            raise ValueError(f"charges must be 1-D, got shape {charges.shape}")
        sectors, multiplicities = np.unique(charges, return_counts=True)
```

with:

```python
        charges = np.asarray(charges, dtype=np.int32)
        if charges.ndim != 1:
            raise ValueError(f"charges must be 1-D, got shape {charges.shape}")
        # Canonicalise before deriving sectors so the cached dense array and the
        # sector table agree on representatives (#734).  Element-wise, so basis
        # ordering within the leg is preserved.
        charges = symmetry.canonicalize(charges)
        sectors, multiplicities = np.unique(charges, return_counts=True)
```

And in `dual`, replace `object.__setattr__(obj, "_charges_cache", self.symmetry.dual(self._charges_cache))` with:

```python
            object.__setattr__(
                obj,
                "_charges_cache",
                self.symmetry.canonicalize(self.symmetry.dual(self._charges_cache)),
            )
```

- [ ] **Step 5: Run the index tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_index.py -q`

Expected: PASS.

- [ ] **Step 6: Check the hot-path cost of canonicalisation**

`TensorIndex` is constructed millions of times, so confirm the added work is not material.

Run:

```bash
JAX_PLATFORMS=cpu uv run python -c "
import numpy as np, timeit
from tenax.core.index import TensorIndex, FlowDirection
from tenax.core.symmetry import U1Symmetry, ZnSymmetry
for name, sym in [('U1', U1Symmetry()), ('Z2', ZnSymmetry(2))]:
    secs = np.array([0,1,2,3], dtype=np.int32); mult = np.array([4,4,4,4], dtype=np.int32)
    t = timeit.timeit(lambda: TensorIndex(symmetry=sym, sectors=secs,
                                          multiplicities=mult, flow=FlowDirection.IN,
                                          label='a'), number=20000)
    print(f'{name}: {t/20000*1e6:.2f} us per construction')
"
```

Expected: single-digit microseconds. U(1) should be the cheaper of the two because
`U1Symmetry.canonicalize` is overridden to return its argument. If Z2 exceeds ~10 µs,
record the number in the PR description rather than optimising speculatively.

- [ ] **Step 7: Write the failing test for `svd` bond-label symmetry**

This is the regression test #733 asks for. Append to `tests/test_linalg.py`:

```python
import numpy as np
import pytest

import tenax.linalg as tl
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity, U1Symmetry, ZnSymmetry
from tenax.core.tensor import SymmetricTensor

_BOND_LABEL_CASES = [
    ("U1", U1Symmetry(), [0, 1]),
    ("Z2", ZnSymmetry(2), [0, 1]),
    ("Z3", ZnSymmetry(3), [0, 1, 2]),
    ("FermionParity", FermionParity(), [0, 1]),
]


@pytest.mark.parametrize(
    "name,sym,sectors", _BOND_LABEL_CASES, ids=[c[0] for c in _BOND_LABEL_CASES]
)
def test_svd_bond_sectors_do_not_depend_on_left_leg_flow(name, sym, sectors):
    """Same decomposition, mirrored flows -> same bond sectors.

    Before #734 a single OUT left leg skipped the modular reduction, so Z2 gave
    ``[-1, 0]`` where the IN orientation gave ``[0, 1]`` -- self-consistent
    labels that silently fail to pair with canonically-built tensors.
    """

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array(sectors, dtype=np.int32),
            multiplicities=np.array([2] * len(sectors), dtype=np.int32),
            flow=flow,
            label=lbl,
        )

    seen = {}
    for left_flow, right_flow in (
        (FlowDirection.IN, FlowDirection.OUT),
        (FlowDirection.OUT, FlowDirection.IN),
    ):
        t = SymmetricTensor.random_normal_np(
            (leg(left_flow, "a"), leg(right_flow, "b")), np.random.RandomState(0)
        )
        U = tl.svd(t, left_labels=["a"], right_labels=["b"])[0]
        bond = [i for i in U.indices if i.label != "a"][0]
        seen[left_flow.name] = sorted(int(q) for q in bond.sectors)

    assert seen["IN"] == seen["OUT"], f"{name}: bond labels depend on flow: {seen}"
    canonical = sorted(
        int(sym.canonicalize(np.array([q], dtype=np.int32))[0]) for q in seen["IN"]
    )
    assert seen["IN"] == canonical, f"{name}: bond labels are not canonical: {seen}"
```

- [ ] **Step 8: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_linalg.py -k svd_bond_sectors -q`

Expected: FAIL for `Z2`, `Z3` and `FermionParity` with e.g. `Z2: bond labels depend on flow:
{'IN': [0, 1], 'OUT': [-1, 0]}`. `U1` passes — `-1` is a genuine U(1) charge.

- [ ] **Step 9: Route the four `linalg.py` sites through `net_charge`**

Add the import near the top of `src/tenax/linalg.py`, alongside the existing
`tenax.core.index` import:

```python
from tenax.core.index import net_charge
```

**9a.** Replace the body of `_has_nonstandard_blocks` (lines 80-92):

```python
def _has_nonstandard_blocks(tensor: SymmetricTensor) -> bool:
    """Return True if any block violates standard conservation."""
    if not tensor.blocks:
        return False
    identity = tensor.indices[0].symmetry.identity()
    for key in tensor.blocks:
        if net_charge(tensor.indices, key) != identity:
            return True
    return False
```

The old raw `sum` reported `True` for standard `Z3` tensors that `_validate` accepts — a
3-leg all-IN block `(1, 1, 1)` sums to `3` but fuses to `0`. It gates the `_bypass`
branch at line 1516.

**9b.** Replace the charge computation inside `_group_blocks_by_bond_charge` (lines 116-124):

```python
    grouped: dict[int, list[tuple[BlockKey, BlockKey, jax.Array]]] = {}
    left_indices_for_charge = tuple(tensor.indices[i] for i in left_leg_positions)

    for key, block in tensor.blocks.items():
        left_subkey = tuple(key[i] for i in left_leg_positions)
        right_subkey = tuple(key[i] for i in right_leg_positions)
        q = net_charge(left_indices_for_charge, left_subkey)
        grouped.setdefault(q, []).append((left_subkey, right_subkey, block))

    return grouped
```

Delete the now-unused `sym = tensor.indices[0].symmetry` line at the top of the function if
nothing else in it uses `sym`.

**9c.** Replace the `input_target` computation at lines 601-607 **and** the identical one at
lines 2105-2111 with:

```python
    input_target = 0
    if tensor.blocks:
        key0 = next(iter(tensor.blocks))
        input_target = net_charge(tensor.indices, key0)

    if input_target != tensor.indices[0].symmetry.identity():
```

(the following line is the existing `if input_target != 0:` — replace that whole line as
shown, keeping its indented body unchanged).

- [ ] **Step 10: Route `core/tensor.py` through the boundary**

**10a.** Replace `_validate` (lines 727-744) in `src/tenax/core/tensor.py`:

```python
    def _validate(self) -> None:
        """Verify all block keys satisfy the symmetry conservation law."""
        if not self._indices:
            return
        identity = self._indices[0].symmetry.identity()

        for key in self._block_keys:
            fused_val = net_charge(self._indices, key)
            if fused_val != identity:
                raise ValueError(
                    f"Block {key} violates charge conservation: "
                    f"fused={fused_val}, expected identity={identity}"
                )
```

Add `net_charge` to the existing `from tenax.core.index import ...` line at
`src/tenax/core/tensor.py:28`.

**10b.** Replace `_compute_valid_blocks` (lines 162-274). The `is_infinite` split
disappears: with `flow_charge` an involution, the closed form works for every abelian
group.

```python
def _compute_valid_blocks(
    indices: tuple[TensorIndex, ...],
    target: int | None = None,
) -> list[BlockKey]:
    """Find all charge-sector tuples satisfying the symmetry conservation law.

    Uses incremental fused-sector propagation: builds up partial fused charges
    one leg at a time, then solves the last leg in closed form via
    ``q = flow_charge(flow, fuse(target, dual(partial)))``.  That inversion is
    valid for any abelian group because ``flow_charge`` is an involution on
    canonical representatives, so there is no longer a separate U(1) branch
    (#734 -- the old ``n_values() is None`` split ran U(1)-only integer algebra
    on ``ProductSymmetry``'s bit-packed charges).

    Args:
        indices: Tuple of TensorIndex objects, one per tensor leg.
        target:  Target charge for the conservation law. If None, uses the
                 symmetry identity (standard conservation).  Setting target=Q
                 selects blocks whose net charge is Q instead of the identity;
                 used at MPS boundaries to enforce a specific quantum number.

    Returns:
        List of BlockKey tuples (one charge per leg) for valid sectors.
    """
    if not indices:
        return [()]

    sym = indices[0].symmetry
    identity_arr = np.array([sym.identity()], dtype=np.int32)
    raw_target = target if target is not None else sym.identity()
    effective_target = int(
        sym.canonicalize(np.array([int(raw_target)], dtype=np.int32))[0]
    )

    # Sectors are canonical, sorted and unique (guaranteed by TensorIndex).
    unique_charges_per_leg = [idx.sectors.tolist() for idx in indices]
    n_legs = len(indices)

    def _effective(leg_i: int, q: int) -> np.ndarray:
        return sym.flow_charge(indices[leg_i].flow, np.array([q], dtype=np.int32))

    if n_legs == 1:
        # Seed with identity so the lone leg is still reduced (#733).
        return [
            (q,)
            for q in unique_charges_per_leg[0]
            if int(sym.fuse(identity_arr, _effective(0, q))[0]) == effective_target
        ]

    partial: dict[int, list[tuple[int, ...]]] = {}
    for q in unique_charges_per_leg[0]:
        fused = int(sym.fuse(identity_arr, _effective(0, q))[0])
        partial.setdefault(fused, []).append((q,))

    last_leg_idx = n_legs - 1
    for leg_i in range(1, last_leg_idx):
        next_partial: dict[int, list[tuple[int, ...]]] = {}
        for q in unique_charges_per_leg[leg_i]:
            eff_q = _effective(leg_i, q)
            for prev_fused, prev_combos in partial.items():
                new_fused = int(
                    sym.fuse(np.array([prev_fused], dtype=np.int32), eff_q)[0]
                )
                extended = [combo + (q,) for combo in prev_combos]
                if new_fused in next_partial:
                    next_partial[new_fused].extend(extended)
                else:
                    next_partial[new_fused] = extended
        partial = next_partial

    # Closed form for the last leg. In an abelian group the solution is unique,
    # so this replaces enumeration for finite groups too.
    last_charge_set = set(unique_charges_per_leg[last_leg_idx])
    flow_last = indices[last_leg_idx].flow
    target_arr = np.array([effective_target], dtype=np.int32)
    valid_keys: list[BlockKey] = []

    for prev_fused, prev_combos in partial.items():
        needed = sym.fuse(target_arr, sym.dual(np.array([prev_fused], dtype=np.int32)))
        q_last = int(sym.flow_charge(flow_last, needed)[0])
        if q_last in last_charge_set:
            for combo in prev_combos:
                valid_keys.append(combo + (q_last,))

    return valid_keys
```

- [ ] **Step 11: Update the test that pinned the old convention**

`tests/test_ctm_root_implicit_sym_sectors.py:417` is
`test_zn_same_flow_bonds_carry_the_library_s_own_representative`, whose docstring says it is
the test that should fail and be updated when #733 is fixed. Rename it and invert its
assertion. Replace the assertion at line 454:

```python
    assert sorted(int(q) for q in lib_bond.sectors) == [-1, 0]
```

with:

```python
    assert sorted(int(q) for q in lib_bond.sectors) == [0, 1]
```

Rename the function to `test_zn_same_flow_bonds_carry_the_canonical_representative` and
replace its docstring with:

```python
    """Same-flow legs label ``Z_n`` partners canonically, matching ``tenax.linalg``.

    This is the orientation the CTM cut actually uses (both projectors are built
    with row and col flowing the same way).  Before #734 the partner of charge 1
    was written ``-1`` here and by ``tenax.linalg.svd``, because
    ``_group_blocks_by_bond_charge`` fused a single flow-weighted charge and
    ``fuse_many`` of one array skipped the ``% n``.  Both now go through
    ``net_charge``, which seeds the fusion with the identity, so the single-leg
    and multi-leg paths agree by construction.
    """
```

Then read the rest of the function body (lines 455-470) and update any further assertion
that assumes a negative representative, so the module's convention and the library's stay
pinned together.

- [ ] **Step 12: Run the targeted tests**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_linalg.py tests/test_index.py tests/test_tensor.py -q
JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -q
```

Expected: PASS.

- [ ] **Step 13: Run the full core gate**

Run: `JAX_PLATFORMS=cpu uv run pytest -m core -q`

Expected: PASS. This is the step most likely to surface fallout — 15 test files touch block
keys or sectors. Any failure asserting a negative Z_n representative is an expected update;
any failure about *values* is a real regression, so diagnose rather than re-baseline.

- [ ] **Step 14: Run the broader suite**

Run: `JAX_PLATFORMS=cpu uv run pytest -m "not slow" -q`

Expected: PASS.

- [ ] **Step 15: Commit**

```bash
git add src/tenax/core/index.py src/tenax/core/tensor.py src/tenax/linalg.py \
        tests/test_index.py tests/test_linalg.py tests/test_ctm_root_implicit_sym_sectors.py
git commit -m "fix(#733): canonical charge representatives through the boundary

TensorIndex now canonicalises sectors and merges duplicate representatives,
so Z2 [-1, 0, 1] becomes sectors [0, 1] with multiplicities [1, 2]. That
guarantees the precondition flow_charge needs, and repairs dual(), which
sorted without merging and could emit sectors=[0, 1, 1] against its own
sorted-unique invariant.

_validate and the four linalg.py sites consume net_charge.
_compute_valid_blocks loses its n_values() is None split entirely: with
flow_charge an involution, the closed form for the last leg is valid for
any abelian group, so the U(1)-only integer algebra that ran on
ProductSymmetry's packed charges is gone rather than patched.

svd bond sectors no longer depend on the left leg's flow: Z2/Z3/
FermionParity gave [-1, 0] / [-2, -1, 0] with the left leg OUT and
[0, 1] / [0, 1, 2] with it IN. Updates the #729 test that pinned the old
convention, as its docstring anticipated.

Closes #733. Refs #734.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Contraction and DMRG target inference (PR 3 — fixes the silent zeros)

**Files:**
- Modify: `src/tenax/contraction/contractor.py:420-435`
- Modify: `src/tenax/algorithms/dmrg.py:3112-3131`
- Test: `tests/test_contraction.py`

- [ ] **Step 1: Write the failing regression test**

This is the measured repro: the contraction should have norm ≈ 5.824 and returns 0.0.
Append to `tests/test_contraction.py`:

```python
import jax.numpy as jnp
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import ZnSymmetry
from tenax.core.tensor import SymmetricTensor


def test_zn_contract_does_not_drop_blocks_on_uniform_raw_sum():
    """Every block of A has raw sum(flow*q) == 3 while its fused charge is 0.

    The old target inference added flow-weighted charges as plain integers, so
    it read a target of 3, and _compute_valid_blocks then admitted no output
    block at all -- an all-zero result with no error raised (#734).
    """
    sym = ZnSymmetry(3)

    def leg(sectors, flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array(sectors, dtype=np.int32),
            multiplicities=np.array([2] * len(sectors), dtype=np.int32),
            flow=flow,
            label=lbl,
        )

    A = SymmetricTensor.random_normal_np(
        (
            leg([1], FlowDirection.IN, "a"),
            leg([1], FlowDirection.IN, "b"),
            leg([1], FlowDirection.IN, "c"),
        ),
        np.random.RandomState(0),
    )
    B = SymmetricTensor.random_normal_np(
        (leg([1], FlowDirection.OUT, "c"), leg([0, 1, 2], FlowDirection.IN, "d")),
        np.random.RandomState(1),
    )

    out = contract(A, B)
    ref = jnp.einsum("abc,cd->abd", A.todense(), B.todense())

    # Compare against a dense reference in norm: asserting only that the
    # contraction "succeeds" would pass on the all-zero result.
    assert float(jnp.linalg.norm(ref)) > 1.0, "reference is degenerate, test is vacuous"
    assert float(jnp.max(jnp.abs(out.todense() - ref))) < 1e-10
```

- [ ] **Step 2: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_contraction.py -k uniform_raw_sum -q`

Expected: FAIL — the max-abs-difference assertion fails at roughly `3.55`, because
`out.todense()` is all zeros while `ref` has norm ≈ 5.824.

- [ ] **Step 3: Fix target inference in `contractor.py`**

Add the import near the other `tenax.core` imports in
`src/tenax/contraction/contractor.py`:

```python
from tenax.core.index import net_charge
```

Replace lines 421-433 (the `output_target: int | None = None` block through
`output_target = total_target`) with:

```python
    output_target: int | None = None
    sym = None
    for tensor in tensors:
        if getattr(tensor, "indices", None):
            sym = tensor.indices[0].symmetry
            break

    if sym is not None:
        # Accumulate with fuse, not +=: adding targets as plain integers is the
        # same category error as weighting a charge by int(flow) (#734).
        total = np.array([sym.identity()], dtype=np.int32)
        for tensor in tensors:
            if getattr(tensor, "_block_keys", None):
                targets = {net_charge(tensor.indices, key) for key in tensor._block_keys}
                if len(targets) == 1:
                    total = sym.fuse(
                        total, np.array([targets.pop()], dtype=np.int32)
                    )
        total_target = int(total[0])
        if total_target != sym.identity():
            output_target = total_target
```

Keep the surrounding comment block above it (lines 413-420) and the
`valid_output_set = set(_compute_valid_blocks(...))` line below it unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_contraction.py -k uniform_raw_sum -q`

Expected: PASS.

- [ ] **Step 5: Verify `BlockArray` exposes `.indices` before touching dmrg.py**

`dmrg.py:3115` accepts both `SymmetricTensor` and `BlockArray`, and `net_charge` needs
`.indices`.

Run:
```bash
JAX_PLATFORMS=cpu uv run python -c "
from tenax.core._block_array import BlockArray
print('indices' in dir(BlockArray))
"
```

Expected: `True`. If it prints `False`, do not proceed with Step 6 — instead keep the
`BlockArray` branch on its existing code path and note the gap in the PR description.

- [ ] **Step 6: Fix target inference in `dmrg.py`**

Add `from tenax.core.index import net_charge` to the imports at the top of
`src/tenax/algorithms/dmrg.py`. Replace lines 3120-3131:

```python
        sectors: set[int] = set()
        for key in site.blocks:
            total = 0
            for idx, q in zip(site.indices, key):
                total += int(idx.flow) * q
            sectors.add(total)

        if len(sectors) != 1:
            return None
        charge = sectors.pop()
        if charge != 0:
            return charge
```

with:

```python
        sectors = {net_charge(site.indices, key) for key in site.blocks}

        if len(sectors) != 1:
            return None
        charge = sectors.pop()
        if charge != site.indices[0].symmetry.identity():
            return charge
```

- [ ] **Step 7: Run contraction and DMRG tests**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_contraction.py -q
JAX_PLATFORMS=cpu uv run pytest tests/test_dmrg.py -q
```

Expected: PASS.

- [ ] **Step 8: Run the full core gate and the broader suite**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest -m core -q
JAX_PLATFORMS=cpu uv run pytest -m "not slow" -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/tenax/contraction/contractor.py src/tenax/algorithms/dmrg.py \
        tests/test_contraction.py
git commit -m "fix(#734): fuse contraction targets instead of adding them

contractor.py inferred each tensor's target from a raw integer sum, so a
Z_n tensor whose blocks share a nonzero raw sum got a spurious
output_target and _compute_valid_blocks then admitted nothing -- an
all-zero contraction with no error raised. Measured on Z3: reference norm
5.824, contract result 0.0.

Targets are now read with net_charge and accumulated with fuse rather
than +=; adding them as plain integers was the same category error as
weighting a charge by int(flow). dmrg.py's target inference gets the same
treatment.

The regression test compares against a dense einsum reference in norm:
asserting only that the contraction succeeds would have passed on the
all-zero result.

Refs #734.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: ProductSymmetry enablement (PR 4)

**Files:**
- Modify: `src/tenax/algorithms/_tensor_utils.py:180-206`
- Modify: `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py` (lift the refusal near line 290)
- Test: `tests/test_symmetry.py`, `tests/test_tensor.py`

- [ ] **Step 1: Write the failing end-to-end ProductSymmetry test**

Append to `tests/test_tensor.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.linalg as tl
from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import ProductSymmetry, U1Symmetry, ZnSymmetry
from tenax.core.tensor import SymmetricTensor

_PRODUCT_CASES = [
    (
        "Prod(Z2,U1)",
        ProductSymmetry(ZnSymmetry(2), U1Symmetry()),
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
    ),
    (
        "Prod(Z2,Z3)",
        ProductSymmetry(ZnSymmetry(2), ZnSymmetry(3)),
        [ProductSymmetry.encode(0, 0), ProductSymmetry.encode(1, 1)],
    ),
]


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_mixed_flow_tensor_constructs(name, sym, sectors):
    """Before #734 this raised: fused=-65536 for the conserving block (q, q)."""

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array(sectors, dtype=np.int32),
            multiplicities=np.array([2, 2], dtype=np.int32),
            flow=flow,
            label=lbl,
        )

    t = SymmetricTensor.random_normal_np(
        (leg(FlowDirection.IN, "a"), leg(FlowDirection.OUT, "b")),
        np.random.RandomState(0),
    )
    assert t.blocks, f"{name}: no valid blocks found"
    t._validate()


@pytest.mark.parametrize(
    "name,sym,sectors", _PRODUCT_CASES, ids=[c[0] for c in _PRODUCT_CASES]
)
def test_product_symmetry_svd_reconstructs(name, sym, sectors):
    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array(sectors, dtype=np.int32),
            multiplicities=np.array([2, 2], dtype=np.int32),
            flow=flow,
            label=lbl,
        )

    t = SymmetricTensor.random_normal_np(
        (leg(FlowDirection.IN, "a"), leg(FlowDirection.OUT, "b")),
        np.random.RandomState(0),
    )
    U, s, Vh, _ = tl.svd(t, left_labels=["a"], right_labels=["b"])
    rebuilt = contract(U, Vh)
    ref = t.todense()
    assert float(jnp.linalg.norm(ref)) > 1.0, "reference is degenerate"
    # U and Vh carry the singular values folded in per sector; compare the
    # reconstruction against the dense original in norm.
    assert float(jnp.max(jnp.abs(rebuilt.todense() - ref))) < 1e-8
```

Note: if `tl.svd` returns `U`, `s`, `Vh` with `s` held out rather than folded in, the
reconstruction needs `s` reapplied. Run Step 2 first and read the failure before adjusting;
mirror whatever the existing `Z2` round-trip test in `tests/test_linalg.py` does.

- [ ] **Step 2: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_tensor.py -k product_symmetry -q`

Expected: after Tasks 1-3 the construction test may already PASS (`_validate` and
`_compute_valid_blocks` are fixed). The `svd` test is the one expected to fail, in
`_compute_fused_sectors`. Record which of the two fails — that determines whether Step 3
is needed at all.

- [ ] **Step 3: Remove the hand-rolled `% n` from `_compute_fused_sectors`**

In `src/tenax/algorithms/_tensor_utils.py`, replace the loop body at lines 193-204:

```python
    for i in range(da):
        for j in range(db):
            # Raw charge contribution: flow_a * q_a + flow_b * q_b
            raw = flow_a_sign * int(idx_a.charges[i]) + flow_b_sign * int(
                idx_b.charges[j]
            )
            # Map to fused charge: q_f such that fused_flow * q_f = raw
            q_f = raw * fused_sign  # since fused_sign^2 = 1
            # For Zn: reduce mod n
            n = sym.n_values() if hasattr(sym, "n_values") else None
            if n is not None:
                q_f = q_f % n
            fused[i * db + j] = q_f
```

with:

```python
    for i in range(da):
        for j in range(db):
            # Effective charge contributed by each parent leg, then fused.
            # int(flow) * q plus a hand-rolled "% n_values()" was wrong for
            # ProductSymmetry twice over: negating a bit-packed charge is not
            # the component-wise inverse, and n_values() is a cardinality
            # (2 * 3 == 6) rather than a modulus for the packed integer (#734).
            raw = sym.fuse(
                sym.flow_charge(flow_a_sign, np.array([idx_a.charges[i]], np.int32)),
                sym.flow_charge(flow_b_sign, np.array([idx_b.charges[j]], np.int32)),
            )
            # q_f such that flow_charge(fused_flow, q_f) == raw; flow_charge is
            # an involution on canonical representatives.
            fused[i * db + j] = int(sym.flow_charge(fused_sign, raw)[0])
```

Delete the now-unused `fused_sign = int(fused_flow)` only if nothing else references it —
it is still used above, so keep it.

- [ ] **Step 4: Run the ProductSymmetry tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_tensor.py -k product_symmetry -q`

Expected: PASS.

- [ ] **Step 5: Lift the ProductSymmetry refusal in the root-implicit module**

Read `src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py` around lines 280-315. It
raises for `ProductSymmetry` with a message about `-q` not being the group inverse, and
builds keys with `key[col_axis] = -int(q) * col_flow`.

Replace the key construction at lines 310-311:

```python
        key[row_axis] = int(q) * row_flow
        key[col_axis] = -int(q) * col_flow
```

with:

```python
        q_arr = np.array([int(q)], dtype=np.int32)
        key[row_axis] = int(sym.flow_charge(row_flow, q_arr)[0])
        key[col_axis] = int(sym.flow_charge(col_flow, sym.dual(q_arr))[0])
```

binding `sym = row_index.symmetry` above the loop if it is not already in scope. Then delete
the `ProductSymmetry` refusal block and update the surrounding docstring (lines 245-295),
which describes the old `-q` convention at length, to state that keys now go through
`flow_charge` and that `ProductSymmetry` is supported.

- [ ] **Step 6: Run the root-implicit sector tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_sym_sectors.py -q`

Expected: PASS. If the refusal had a dedicated test asserting `ProductSymmetry` raises,
convert it into a test asserting the round-trip now succeeds.

- [ ] **Step 7: Run the full core gate and the broader suite**

Run:
```bash
JAX_PLATFORMS=cpu uv run pytest -m core -q
JAX_PLATFORMS=cpu uv run pytest -m "not slow" -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/tenax/algorithms/_tensor_utils.py \
        src/tenax/algorithms/_ctm_root_implicit_sym_sectors.py tests/test_tensor.py
git commit -m "feat(#734): mixed-flow ProductSymmetry tensors

_compute_fused_sectors no longer negates a bit-packed charge and no
longer applies % n_values() to it -- n_values() is a cardinality (2*3==6),
not a modulus for the packed integer. Both go through flow_charge/fuse.

With the boundary in place, mixed-flow ProductSymmetry tensors construct,
decompose and contract for the first time, so the root-implicit sector
module's refusal is lifted and its keys built with flow_charge.

Refs #734.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Final verification (before opening each PR)

- [ ] `JAX_PLATFORMS=cpu uv run pytest -m core -q` passes
- [ ] `pre-commit run --all-files` passes (ruff, ruff-format)
- [ ] No `int(idx.flow) *` or `int(...flow) *` remains outside `core/symmetry.py`:

```bash
grep -rn "flow) \* \|flow\.value \* " src/tenax --include=*.py
```

Expected after Task 4: no hits in executable code. Prose in docstrings that *describes* the
old convention is fine only where it explains why the convention changed.

- [ ] Open the PR with `gh pr create`, then `gh pr merge <n> --squash --auto`.
      **Never pass `--delete-branch`** — the merge queue deletes the head branch itself, and
      the flag deletes it the moment the PR *enters* the queue, closing the PR and dropping
      it from the queue.

---

## Self-review notes

**Spec coverage.** §1 boundary API → Task 1 Steps 3-5, 9. §2 invariant and the
`_compute_valid_blocks` unification → Task 1 Step 1 (`test_closed_form_solves_the_last_leg_charge`)
and Task 2 Step 10b. §3 canonicalisation at construction, including the four sub-steps and
`_charges_cache` → Task 2 Steps 3-4. §4 staging → Tasks 1-4 in order. Testing section →
Task 1 Steps 1, 7; Task 2 Steps 1, 7; Task 3 Step 1; Task 4 Step 1. Risks: regression
surface → Task 2 Steps 13-14; silent-zero weak evidence → Task 3 Step 1 asserts against a
dense reference in norm and guards the reference is non-degenerate; `_charges_cache` →
Task 2 Step 4 plus its dedicated test; fermionic paths → `FermionParity` appears in every
parametrised case list.

**Naming consistency.** `flow_charge(flow, charges)`, `canonicalize(charges)`,
`net_charge(indices, key)` — used identically in every task.

**Known judgement calls left to the implementer**, each with a verification step rather than
a guess: whether `BlockArray` exposes `.indices` (Task 3 Step 5), whether `tl.svd` folds the
singular values into `U`/`Vh` (Task 4 Step 1 note, resolved by Task 4 Step 2), and whether
the `ProductSymmetry` refusal has a dedicated test (Task 4 Step 6).
