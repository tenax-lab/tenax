# Charge arithmetic must go through the symmetry

**Date:** 2026-08-01
**Issue:** #734 (umbrella); #733 is one symptom, closed by PR 2 below
**Branch:** `fix/733-charge-arithmetic-boundary`

## Guiding principle

The stored elements and charge labels of a symmetric tensor are not physically
observable. Every observable is a scalar with respect to the symmetry group, obtained by
collapsing all legs. A single stored coordinate is a *coefficient in a chosen
basis/sector*, not a scalar; that a fused-all-legs scalar coincides with a stored element
is a special case, true only for abelian symmetries with trivial fusion coefficients.

Two consequences drive this design:

1. Charge **representatives** are convention. Rewriting the Z₂ partner of `1` from `-1` to
   `1` breaks nothing observable, so it is free to canonicalise.
2. What *must* hold is that operations **compose**. Labels that disagree make `contract`
   silently keep the charge-matching components and zero the rest, rather than raising.

## Problem

Eight sites in `src/tenax` hand-roll group arithmetic on charges, encoding one of two
assumptions that are true only for U(1):

- **"the group inverse is integer negation"** — `int(flow) * q`
- **"the group operation is integer addition"** — a raw `sum(...)` compared to `identity()`

For Z_n these are accidentally true whenever at least two legs fuse afterwards (`fuse`
reduces mod n), and fail exactly when nothing fuses. For `ProductSymmetry`, whose charges
are bit-packed (`encode = (q2 << 16) | (q1 & 0xFFFF)`), they are never true.

### Verified findings

Severity 1 — silent wrong numbers, no error raised:

| Site | Defect | Evidence |
|---|---|---|
| `contraction/contractor.py:427` | Target inferred from a raw integer sum; a Z_n tensor whose blocks share a nonzero raw sum gets a spurious `output_target` and `_compute_valid_blocks` then admits nothing | Z3 contraction: `norm(ref)=5.824`, `norm(contract)=0.0`, every block dropped |
| `linalg.py:121` `_group_blocks_by_bond_charge` | Single left leg ⇒ `fuse_many` of one array skips reduction ⇒ non-canonical bond label (#733) | Z2/Z3/FermionParity left-OUT bonds label `[-1,0]` / `[-2,-1,0]` where left-IN gives `[0,1]` / `[0,1,2]` |

Severity 2 — wrong control-flow decisions:

| Site | Defect | Evidence |
|---|---|---|
| `linalg.py:80` `_has_nonstandard_blocks` | Raw sum with no modular reduction; gates the `_bypass` branch at `linalg.py:1516` | Returns `True` for a standard Z3 tensor `_validate` accepts; 8 of 9 blocks misjudged |
| `dmrg.py:3124`, `linalg.py:605`, `linalg.py:2109` | Same raw-sum pattern, used to infer target charge and select the validation-bypass path | same class |

Severity 3 — `ProductSymmetry` is systemically non-functional, failing well before #733's site:

| Site | Defect | Evidence |
|---|---|---|
| `core/tensor.py:736` `_validate` | Raw negation of a packed charge rejects conserving blocks at construction | `Block (65537, 65537) violates charge conservation: fused=-65536, expected identity=0` |
| `core/tensor.py:255` | `ProductSymmetry(Z2, U1).n_values()` is `None`, so the U(1)-only closed form `q_last = needed * flow_last` runs on packed charges | branch verified |
| `core/tensor.py:205` | Single-leg enumeration, raw and unreduced | same class as #733 |
| `algorithms/_tensor_utils.py:204`, `core/symmetry.py:178` | `ProductSymmetry(Z2, Z3).n_values()` is `6`, so both apply `% 6` to a bit-packed charge | branch verified |

Severity 4 — the API invites the violation:

- `n_values()` reads as *"the modulus"*. Every caller that does `% n` is correct only when
  the charge is literally an integer in Z_n.
- `BaseSymmetry.is_conserved` — the one method that reduces properly — has **zero call
  sites**. A sanctioned abstraction exists, is dead, and eight sites reimplement a worse
  version of it. (It also negates raw, so it is still wrong for `ProductSymmetry`.)

Severity 5 — index invariant already violated:

- `TensorIndex` documents `sectors` as *sorted unique*, but `dual()` sorts without merging.
  A Z2 index `[-1, 0, 1]` duals to `sectors=[0, 1, 1]`, after which `multiplicity(1)`
  undercounts because `searchsorted` returns only the first match. `dual` is therefore not
  an involution: `[-1,0,1] → [0,1,1] → [0,1,1]`.

Element-as-scalar, the principle's most direct subject, is respected: **zero hits** for raw
element access on symmetric tensors across `src/tenax`, and README never teaches users to
reach for `.blocks`. The one exception is `algorithms/tdvp.py:251` and `:270`,
`float(left_env.todense().real.ravel()[0])` — a genuine scalar only because every leg is
dim-1 after contracting the full chain; `.ravel()[0]` would silently return a basis
coefficient rather than raising if that stopped holding.

## Design

### 1. The boundary

Nothing outside `BaseSymmetry` may invert or combine a charge. Two additions to
`BaseSymmetry`, both with working defaults so no subclass needs to override:

```python
def flow_charge(self, flow, charges):   # charges if IN else self.dual(charges)
def canonicalize(self, charges):        # self.fuse(identity, charges)
```

One helper in `core/`, used by every conservation site:

```python
def net_charge(indices, key) -> int:    # fuse_many of flow_charge per leg, seeded with identity
```

Seeding with `identity()` is what fixes #733: there is no longer a path where zero fusions
happen, so the single-leg and multi-leg cases agree by construction.

`is_conserved` is rewritten in terms of these and thereby becomes correct for
`ProductSymmetry`. Its `% n_values()` reduction is removed.

### 2. The invariant

**`flow_charge` is an involution on canonical representatives.** This is the load-bearing
property. It makes the following valid for *any* abelian group, not just U(1):

```python
q_last = flow_charge(flow_last, fuse(target, dual(partial)))
```

Consequently `core/tensor.py:_compute_valid_blocks` no longer needs its
`is_infinite = sym.n_values() is None` split: the closed-form branch generalises to every
abelian symmetry and the two branches **collapse into one**. The
`ProductSymmetry(Z2, U1) → None → U(1) formula on packed charges` defect disappears rather
than being patched. The same substitution removes the two `% n_values()` misuses in
`_tensor_utils.py:204` and `symmetry.py:178`.

`n_values()` keeps its name and meaning as a **cardinality** — used for sizing, never as a
modulus. This is documented on the method.

### 3. Canonicalisation at construction

The involution holds only if stored sectors are canonical. `TensorIndex.__post_init__`
therefore canonicalises `sectors` and **merges duplicates**, so the precondition holds by
construction rather than by discipline. Every construction path — `from_charges`, `dual`,
`flip_flow`, direct construction — funnels through `__post_init__`, so this repairs the
`dual()` non-uniqueness in Severity 5 as a side effect.

Merging is not optional: canonicalising Z2 `[-1, 0, 1]` yields `[1, 0, 1]`, which must
become sectors `[0, 1]` with multiplicities `[1, 2]`. The steps are:

1. `sectors ← symmetry.canonicalize(sectors)`
2. group equal sectors, summing their multiplicities
3. re-sort ascending
4. canonicalise `_charges_cache` the same way when present, so `charges`,
   `from_dense`/`todense` orderings stay consistent with `sectors`

Under the guiding principle this rewrite is free: it changes representatives, not physics.
It is the only part of this design that touches a constructor.

### 4. Staging

Four PRs, each independently landable, each with its own regression gate.

| PR | Scope | Behaviour change | Gate |
|---|---|---|---|
| 1 | `flow_charge`, `canonicalize`, `net_charge`; rewrite the dead `is_conserved` to use them | none (pure addition) | unit tests per symmetry, including `ProductSymmetry`: `flow_charge` involution, `fuse(q, dual(q)) == identity`, `canonicalize` idempotent |
| 2 | `core/tensor.py` (`_validate`, `_compute_valid_blocks`, `TensorIndex.__post_init__`) and `linalg.py` (lines 80, 121, 605, 2109) | **Z_n bond labels canonicalise**; updates `test_zn_same_flow_bonds_carry_the_library_s_own_representative` | full symmetric-tensor suite; new test asserting `svd` gives identical bond sectors with the left leg IN and OUT, for U(1), Z2, Z3 |
| 3 | `contraction/contractor.py:427`, `dmrg.py:3124` | **fixes Z_n `contract` returning all zeros** | the verified 5.824 → 0.0 repro becomes a regression test |
| 4 | `ProductSymmetry` enablement; `_tensor_utils.py:204` | mixed-flow `ProductSymmetry` tensors work for the first time | construction, `svd`, `contract` round-trips on `ProductSymmetry(Z2, U1)` and `(Z2, Z3)`; lift the refusal in `_ctm_root_implicit_sym_sectors.py` |

Ordering rationale: PR 1 is behaviourally inert, making it a safe merge-queue warm-up. PR 2
must land `_validate` and `_group_blocks_by_bond_charge` together — fixing either alone
leaves the two conventions inconsistent. PRs 2 and 3 each carry a Severity-1 fix, so
neither waits on PR 4.

## Testing

Per-symmetry parametrised tests over `U1Symmetry`, `ZnSymmetry(2)`, `ZnSymmetry(3)`,
`FermionParity`, `FermionicU1`, `ProductSymmetry(Z2, U1)`, `ProductSymmetry(Z2, Z3)`:

- `flow_charge(OUT, flow_charge(OUT, q)) == q` for canonical `q`
- `fuse(q, dual(q)) == identity()` for every sector
- `canonicalize` is idempotent, and `canonicalize(sectors)` equals `sectors` for any index
  built through `TensorIndex`
- `svd` produces the same bond sectors with the left leg IN and with it OUT (the
  regression test named in #733)
- `contract` of a Z3 network matches the dense `einsum` reference in norm — the 5.824/0.0
  repro
- `TensorIndex` with duplicate representatives merges: Z2 `[-1,0,1]` gives
  `sectors=[0,1]`, `multiplicities=[1,2]`, `dim=3`
- `TensorIndex.dual()` is an involution

## Risks

- **Regression surface.** 15 test files touch block keys or sectors; 17 assertions pin
  them. Only one pins a negative Z_n representative
  (`tests/test_ctm_root_implicit_sym_sectors.py:454`), which anticipates its own update.
- **Silent-zero class of bug.** Because the failure mode being fixed is *silent*, absence
  of new test failures is weak evidence. PR 3's gate must compare against a dense
  reference in norm, not merely assert that a contraction succeeds.
- **`_charges_cache` consistency.** Canonicalising `sectors` without canonicalising the
  cached dense charges would desynchronise `charges` from `sectors`. Step 4 of §3 is
  load-bearing, not cosmetic.
- **Fermionic paths.** `FermionParity` is self-dual (`dual(q) == q`), so `flow_charge` is
  the identity map there and Koszul-sign logic is untouched. This is asserted, not assumed.

## Out of scope

- `tdvp.py:251/270` element-as-scalar. Correct today; tracked separately as a robustness
  note, not a bug.
- Renaming or deprecating `n_values()`.
- Making `.blocks` private. The principle argues for it, but nothing in `src/tenax` or the
  README currently abuses it, so there is no defect to fix.
- Non-abelian symmetries. `flow_charge` gives them a place to hook in later; no F/R-symbol
  machinery is introduced here.
