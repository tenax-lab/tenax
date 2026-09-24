# Graded contractor, Phase 1 (reference + go/no-go) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reference graded contractor beside the production path, prove it against the exact Fock oracle (#1038) on `SymmetricTensor`s, and run the design's go/no-go (SVD under graded semantics plus a cost measurement), with **no change to any production code path**.

**Architecture:** One new private module, `src/tenax/core/_graded.py`, is built only from existing primitives: graded `transpose` (rule 1), a per-block parity twist (rule 2) and `bar()` times the full-reversal phase (rule 3). Plain `contract` does the actual block work. Test helpers rebuild #1038's oracle clusters out of `SymmetricTensor`s, so every claim in design §3.4 is re-checked on tenax's own tensors. The result is a go/no-go record appended to the design doc; phases 2+ (design §5 steps 3–7) get their own plans only if it is GO.

**Tech Stack:** Python ≥3.11, JAX (x64), numpy ≥1.26, pytest, `uv`, ruff pre-commit.

**Spec:** `docs/plans/2026-09-23-1035-fermionic-sign-convention-design.md` (PR #1036, revision 2 at `165dd04`). This plan implements §5 steps 1–2 and the D0 cost measurement. Step 0 (the oracle, #1038) is already merged as `b6fad3d`.

**Scope note:** Design §5 has seven steps, and steps 3–7 depend on this phase's go/no-go and touch ~10 production modules (CTM moves, fusion, DMRG, TRG/HOTRG). Following the one-plan-per-subsystem rule, this plan covers only the gate. Write Phase 2's plan (step 3: fusion + double layer on the graded path) after Task 6 records GO.

**Pre-validated:** Every code block below was run on `origin/main` @ `b6fad3d` in a scratch worktree before this plan was written. Results: 23 tests passed (core subset 20 passed in 28 s), and the mutants listed in each task fail the tests stated. Expect the same results. If a result differs, stop and investigate rather than adjusting the test.

## Global Constraints

- **No production behaviour change.** The only file under `src/` this plan touches is the new `src/tenax/core/_graded.py`, and nothing in `src/` imports it.
- **The module is private** (leading underscore), so there is no `src/tenax/__init__.py` `__all__` or `README.md` change. CLAUDE.md requires those only for public API.
- **numpy ≥1.26 is supported** (`pyproject.toml`): no `np.bitwise_count` or other NumPy-2-only calls.
- **The required gate stays fast:** new tests run in `pytest -m core`. Each new 2×3 case carries an explicit `@pytest.mark.slow` (the conftest rule withholds `core` from it).
- **Run everything from the worktree with `uv run`.** Bare `python` imports the main clone's tenax. Use `JAX_PLATFORMS=cpu`.
- **Git:** a branch off `origin/main`, **one PR** for this phase, never push to `main`, **never pass `--delete-branch`**, **don't arm `--auto`** (the user decides). Read Codex's review before anything is armed.
- **Commit trailers** (every commit):
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap`.
- **Any GitHub text an agent posts** carries `> 🤖 **AI-generated comment** — written by Claude Code, posted by @yingjerkao.` (hook-enforced).
- **Mutation discipline:** commit before mutating, assert the anchor is unique, restore with `git checkout <file>`, and name which test failed.

## Review Focus

Inputs the design implies but its §3.4 prototype never exercised. Each has a test in the task that owns the code:

1. **Complex tensors.** Energy must stay real and match Fock. The earlier oracle PR hit exactly this twice (bilinear forms, `float(complex)`). Task 2: `test_graded_energy_matches_fock_for_complex_tensors`.
2. **Tracing under `jax.jit`.** Signs come from host-side block keys, and blocks are traced. jit must equal eager bit-for-bit. Task 1: `test_it_traces_under_jit`.
3. **`FermionicU1`** (parity derived from a U(1) charge, multiplicity > 1 per sector). The rules must not assume FermionParity's charge set. Task 1: both algebra tests are parametrized over it.
4. **A `DenseTensor` operand next to a fermionic one.** `DenseTensor` has no grading, and silently contracting it would drop every sign. It must raise. Task 1: `test_a_dense_operand_is_refused_when_either_is_fermionic`.
5. **SVD of a tensor whose storage order isn't `left + right`.** `linalg.svd` reorders sign-free internally. That's the raw-reorder hazard Codex raised on #1036 (`cb44090`). Task 4: `test_graded_svd_reconstructs_when_storage_order_is_not_left_plus_right`, plus its regime twin.

## File Structure

| File | Responsibility |
|---|---|
| `src/tenax/core/_graded.py` (new) | Rules 1–3 as functions: `graded_reorder`, `twist_legs`, `graded_bar`, `graded_contract`; plus `graded_svd` (Task 4). |
| `tests/test_graded_contract.py` (new, core) | Algebra: commutativity, associativity, the regime against sign-free contraction, bosonic fallback, outer product, the rule-3 identity, involution, jit, refusal of dense operands, and graded SVD reconstruction. |
| `tests/_fermionic_fock_oracle.py` (modify) | Adds `z_gauge` (maps rule 2's pairing onto the oracle's projector) and `fock_psi_ordered` (defines the state when a site tensor is odd). |
| `tests/_graded_cluster.py` (new) | Builds the oracle clusters as `SymmetricTensor`s (ket sites, graded bras, hop operator, optional odd auxiliary leg) and contracts them in global or per-site order. |
| `tests/test_graded_contract_oracle.py` (new, core + slow) | The graded contractor against Fock: even (both orders, complex), the rule-3 regime, odd tensors on an auxiliary leg, and the SVD-regauge go/no-go. |
| `tests/conftest.py` (modify) | Registers the two new test files as `core`. |
| `examples/profile_graded_contract_1035.py` (new) | Cost: graded vs sign-free on a CTM-shaped contraction, eager and jit compile. |
| `docs/plans/2026-09-23-1035-fermionic-sign-convention-design.md` (modify, on the #1036 branch) | §8: the Phase-1 go/no-go record. |

---

### Task 0: Worktree and branch

- [ ] **Step 1: Create the worktree off `origin/main`**

```bash
cd /home/yjkao/tenax
git fetch origin
git worktree add ../tenax-1035-phase1 -b feat/1035-graded-contractor-phase1 origin/main
cd ../tenax-1035-phase1
pre-commit install
git log --oneline -1   # expect b6fad3d or later; tests/_fermionic_fock_oracle.py must exist
```

- [ ] **Step 2: Confirm the oracle baseline is green**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_fermionic_fock_oracle.py -m core -q -p no:cacheprovider --no-cov`
Expected: `8 passed, ... 1 xfailed` (the strict xfail is #1037's marker).

---

### Task 1: The graded primitives and their algebra

**Files:**
- Create: `src/tenax/core/_graded.py`
- Create: `tests/test_graded_contract.py`
- Modify: `tests/conftest.py` (the `_FILE_MARKERS` dict, after the `"test_fermionic_fock_oracle.py": "core",` entry)

**Interfaces:**
- Consumes: `SymmetricTensor.transpose` (graded, Koszul), `.bar()`, `.permute_legs`, `._from_blocks_unchecked`, `.blocks`, `.labels()`, `.indices`; `tenax.contraction.contractor.contract(*t, output_labels=...)`; `BaseSymmetry.parity`, `.is_fermionic`.
- Produces (used by Tasks 2–5):
  - `graded_reorder(t: SymmetricTensor, labels: Sequence[str]) -> SymmetricTensor`
  - `twist_legs(t: SymmetricTensor, labels: Iterable[str]) -> SymmetricTensor`
  - `graded_bar(t: SymmetricTensor) -> SymmetricTensor` (same leg order and labels as `t.bar()`)
  - `graded_contract(a: SymmetricTensor, b: SymmetricTensor) -> SymmetricTensor`. It contracts the shared labels; the output legs are `a`'s free legs then `b`'s, in storage order; it raises `TypeError` if either operand is fermionic and either is not a `SymmetricTensor`.

- [ ] **Step 1: Register the test file as core**

In `tests/conftest.py`, directly after the line `    "test_fermionic_fock_oracle.py": "core",` add:

```python
    "test_graded_contract.py": "core",
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_graded_contract.py`:

```python
"""Algebra of the reference graded contractor (#1035, design §3.4).

The physics check against the exact Fock oracle is in
``test_graded_contract_oracle.py``; this file pins the algebraic properties a
graded contraction must have and the ones a sign-free one lacks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.contraction.contractor import contract
from tenax.core._graded import graded_bar, graded_contract, graded_reorder
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, FermionParity, U1Symmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT
SYMS = [
    pytest.param(FermionParity(), [0, 1, 0, 1], id="FermionParity"),
    pytest.param(FermionicU1(), [-1, 0, 1, 0, 1], id="FermionicU1"),
]


def _idx(sym, charges, flow, label):
    return TensorIndex.from_charges(
        sym, np.asarray(charges, np.int32), flow, label=label
    )


def _rand(indices, seed):
    return SymmetricTensor.random_normal(
        indices=tuple(indices), key=jax.random.PRNGKey(seed)
    )


def _triple(sym, ch):
    A = _rand(
        [
            _idx(sym, ch, IN, "a"),
            _idx(sym, ch, OUT, "x"),
            _idx(sym, ch, IN, "y"),
            _idx(sym, ch, OUT, "b"),
        ],
        0,
    )
    B = _rand(
        [_idx(sym, ch, IN, "x"), _idx(sym, ch, OUT, "c"), _idx(sym, ch, OUT, "y")], 1
    )
    C = _rand([_idx(sym, ch, IN, "c"), _idx(sym, ch, IN, "d")], 2)
    return A, B, C


def _maxdiff(s, t):
    return float(jnp.max(jnp.abs(s.todense() - t.todense())))


@pytest.mark.parametrize("sym,ch", SYMS)
def test_even_tensors_commute_up_to_a_graded_reorder(sym, ch):
    A, B, _ = _triple(sym, ch)
    ab, ba = graded_contract(A, B), graded_contract(B, A)
    assert ab.labels() == ("a", "b", "c")
    assert _maxdiff(graded_reorder(ba, ab.labels()), ab) < 1e-12


@pytest.mark.parametrize("sym,ch", SYMS)
def test_contraction_is_associative(sym, ch):
    A, B, C = _triple(sym, ch)
    left = graded_contract(graded_contract(A, B), C)
    right = graded_contract(A, graded_contract(B, C))
    assert _maxdiff(graded_reorder(right, left.labels()), left) < 1e-12


def test_regime_the_graded_result_is_not_the_sign_free_one():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    ab = graded_contract(A, B)
    assert _maxdiff(contract(A, B, output_labels=ab.labels()), ab) > 1e-3


def test_bosonic_operands_fall_back_to_plain_contract():
    u1 = U1Symmetry()
    q = [-1, 0, 1]
    A = _rand([_idx(u1, q, IN, "a"), _idx(u1, q, OUT, "x")], 3)
    B = _rand([_idx(u1, q, IN, "x"), _idx(u1, q, OUT, "b")], 4)
    assert (
        _maxdiff(graded_contract(A, B), contract(A, B, output_labels=("a", "b"))) == 0.0
    )


def test_no_shared_labels_is_the_ordered_outer_product():
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    z = _rand([_idx(FermionParity(), [0, 1], IN, "z")], 9)
    assert graded_contract(A, z).labels() == ("a", "x", "y", "b", "z")


def test_graded_bar_is_the_full_reversal_kept_in_storage_order():
    """Rule 3: conj + flip + reverse the generators, then reorder back with
    the Koszul sign -- which is the (-1)^{sum_{i<j} p_i p_j} phase."""
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    rev = tuple(reversed(range(A.ndim)))
    reversed_form = A.bar().permute_legs(rev)  # generators in reverse order
    assert _maxdiff(graded_bar(A), graded_reorder(reversed_form, A.labels())) < 1e-12
    assert (
        _maxdiff(graded_bar(A), A.bar()) > 1e-3
    )  # regime: the phase is not trivial here


def test_graded_bar_is_an_involution():
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    assert _maxdiff(graded_bar(graded_bar(A)), A) == 0.0


def test_it_traces_under_jit():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    jitted = jax.jit(lambda a, b: graded_contract(a, b).todense())
    assert (
        float(jnp.max(jnp.abs(jitted(A, B) - graded_contract(A, B).todense()))) == 0.0
    )


def test_a_dense_operand_is_refused_when_either_is_fermionic():
    A, B, _ = _triple(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(TypeError, match="DenseTensor"):
        graded_contract(A, DenseTensor(B.todense(), B.indices))

```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -q -p no:cacheprovider --no-cov`
Expected: collection error `ModuleNotFoundError: No module named 'tenax.core._graded'`.

- [ ] **Step 4: Write the implementation** — create `src/tenax/core/_graded.py`:

```python
"""Reference graded (Grassmann) contraction for fermionic SymmetricTensors (#1035).

Storage order is the Grassmann generator order.  Three rules:

1. reorder with the Koszul sign (``SymmetricTensor.transpose``);
2. a contracted pair is +1 when the left operand's leg is OUT and
   ``(-1)**p`` when it is IN;
3. the bra is ``graded_bar``: ``bar()`` times ``(-1)**sum_{i<j} p_i p_j``
   (a full order reversal, kept in the original storage order).

Beside the production path; nothing calls it yet.
"""

from __future__ import annotations

import numpy as np

from tenax.contraction.contractor import contract
from tenax.core.index import FlowDirection
from tenax.core.tensor import SymmetricTensor


def _is_graded(t) -> bool:
    """True for a tensor on a fermionic (Z2-graded) symmetry."""
    return bool(t.indices) and t.indices[0].symmetry.is_fermionic


def _parities(t: SymmetricTensor, key) -> list[int]:
    sym = t.indices[0].symmetry
    return [int(sym.parity(np.array([q]))[0]) for q in key]


def _scale_blocks(t: SymmetricTensor, exponent) -> SymmetricTensor:
    """Multiply each block by ``(-1)**exponent(parities_of_its_key)``."""
    blocks = {}
    for key, block in t.blocks.items():
        blocks[key] = -block if exponent(_parities(t, key)) % 2 else block
    return SymmetricTensor._from_blocks_unchecked(blocks, t.indices)


def twist_legs(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Multiply each block by ``(-1)**(sum of parities on the named legs)``."""
    labels = set(labels)
    axes = [i for i, lab in enumerate(t.labels()) if lab in labels]
    if not axes or not _is_graded(t):
        return t
    return _scale_blocks(t, lambda p: sum(p[i] for i in axes))


def graded_bar(t: SymmetricTensor) -> SymmetricTensor:
    """Rule 3: the Grassmann conjugate, in the original storage order."""
    b = t.bar()
    if not _is_graded(t):
        return b
    n = t.ndim
    return _scale_blocks(
        b, lambda p: sum(p[i] * p[j] for i in range(n) for j in range(i + 1, n))
    )


def graded_reorder(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Rule 1: reorder legs to ``labels`` with the Koszul sign."""
    current = t.labels()
    return t.transpose(tuple(current.index(lab) for lab in labels))


def graded_contract(a: SymmetricTensor, b: SymmetricTensor) -> SymmetricTensor:
    """Contract the labels ``a`` and ``b`` share, graded.  Output legs are
    ``a``'s free legs then ``b``'s, each in its own storage order."""
    if (_is_graded(a) or _is_graded(b)) and not (
        isinstance(a, SymmetricTensor) and isinstance(b, SymmetricTensor)
    ):
        raise TypeError(
            "graded_contract needs SymmetricTensor operands when either is "
            "fermionic: DenseTensor carries no parity grading"
        )
    b_labels = set(b.labels())
    shared = [lab for lab in a.labels() if lab in b_labels]
    free_a = [lab for lab in a.labels() if lab not in b_labels]
    free_b = [lab for lab in b.labels() if lab not in set(shared)]
    out = tuple(free_a + free_b)
    if not _is_graded(a):
        return contract(a, b, output_labels=out)
    # nest the pairs: free_a s1..sk | sk..s1 free_b -- each pair adjacent, a's leg first
    a2 = graded_reorder(a, free_a + shared)
    b2 = graded_reorder(b, list(reversed(shared)) + free_b)
    flows = dict(zip(a2.labels(), (idx.flow for idx in a2.indices)))
    a2 = twist_legs(
        a2, [lab for lab in shared if flows[lab] == FlowDirection.IN]
    )  # rule 2
    return contract(a2, b2, output_labels=out)

```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -q -p no:cacheprovider --no-cov`
Expected: `11 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/core/_graded.py tests/test_graded_contract.py tests/conftest.py
git commit -m "feat(#1035): reference graded contractor beside the production path

Rules 1-3 of design 3.4 as graded_reorder / twist_legs / graded_bar /
graded_contract, built on existing primitives. Nothing in src/ imports
it yet.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

- [ ] **Step 7: Mutation-check each rule** (committed, so `git checkout` restores)

```bash
F=src/tenax/core/_graded.py
T="JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -q -p no:cacheprovider --no-cov"
# rule 2 dropped
grep -c "flows\[lab\] == FlowDirection.IN" $F        # must print 1
sed -i 's/flows\[lab\] == FlowDirection.IN/False/' $F; eval $T | tail -3; git checkout $F
# rule 1 dropped (sign-free reorder)
grep -c "return t.transpose(" $F                         # must print 1
sed -i 's/return t.transpose(/return t.permute_legs(/' $F; eval $T | tail -3; git checkout $F
# rule 3 dropped (no reversal phase)
grep -c "for j in range(i + 1, n)" $F                    # must print 1
sed -i 's/for j in range(i + 1, n)/for j in range(0)/' $F; eval $T | tail -3; git checkout $F
git status --short                                       # must be clean
```

Expected:
- rule 2 dropped → both `test_even_tensors_commute_up_to_a_graded_reorder` params fail (by 6.6 on FermionParity);
- rule 1 dropped → the commute tests fail;
- rule 3 dropped → `test_graded_bar_is_the_full_reversal_kept_in_storage_order` fails.

---

### Task 2: Oracle helpers, the cluster builder, and the even-state check against Fock

**Files:**
- Modify: `tests/_fermionic_fock_oracle.py` (insert two functions immediately before `def hop_energy(`)
- Create: `tests/_graded_cluster.py`
- Create: `tests/test_graded_contract_oracle.py`
- Modify: `tests/conftest.py` (add one entry after the Task 1 entry)

**Interfaces:**
- Consumes: Task 1's `graded_bar`, `graded_contract`; the oracle's `_H2`, `_create`, `_annihilate`, `bonds_of`, `sites_of`, `fock_psi`, `hop_energy`, `plain_amplitudes`, `random_even_tensors`.
- Produces:
  - `z_gauge(R, C, As) -> dict[site, ndarray]`
  - `fock_psi_ordered(R, C, As) -> ndarray`
  - `ket_site(R, C, s, A, *, aux=False) -> SymmetricTensor` (labels `b{bond}` for bonds and `p{n}` for the physical leg; `x` is the odd auxiliary leg)
  - `bra_site(ket, n, *, touched) -> SymmetricTensor`
  - `hop_operator(ns, nt) -> SymmetricTensor`
  - `cluster_value(R, C, bra_As, ket_As, *, op_bond=None, order="global"|"per_site", ket_aux=None, bra_aux=None, override=None) -> complex`
  - `cluster_energy(R, C, As, **kw) -> tuple[float, float]` (energy, norm)

- [ ] **Step 1: Add the two oracle helpers** — in `tests/_fermionic_fock_oracle.py`, insert immediately before `def hop_energy(`:

```python
def z_gauge(R: int, C: int, As: dict) -> dict:
    """Z on every bond's t-side leg (u, l).  Rule 2's "OUT leg first" pairing
    equals the oracle's ``(1 + a_t a_s)`` projector in this gauge.  The sign
    is per bond, so the s-side legs would do equally well."""
    out = {}
    for s, A in As.items():
        A = np.array(A, copy=True)
        for ax in (0, 2):
            if A.shape[ax] == 2:
                idx = [slice(None)] * A.ndim
                idx[ax] = 1
                A[tuple(idx)] *= -1
        out[s] = A
    return out


def fock_psi_ordered(R: int, C: int, As: dict) -> np.ndarray:
    """``O_1 O_2 ... O_N |0>`` (row-major, left to right).  Equal to
    ``fock_psi`` for even tensors; defines the order when some are odd."""
    sites = sites_of(R, C)
    sid = {s: n for n, s in enumerate(sites)}
    mode, M = {}, len(sites)
    for s, x, t, y in bonds_of(R, C):
        mode[(s, x)], mode[(t, y)] = M, M + 1
        M += 2
    vec = np.zeros(1 << M, dtype=np.result_type(*As.values()))
    vec[0] = 1.0
    for s in reversed(sites):  # O_N acts first
        A, new = As[s], np.zeros_like(vec)
        for k in itertools.product(*[range(n) for n in A.shape]):
            if A[k] == 0:
                continue
            v = vec
            for leg, bit in reversed(list(zip("udlrp", k))):
                if bit:
                    v = _create(v, sid[s] if leg == "p" else mode[(s, leg)])
            new = new + A[k] * v
        vec = new
    for s, x, t, y in bonds_of(R, C):
        vec = vec + _annihilate(_annihilate(vec, mode[(s, x)]), mode[(t, y)])
    return vec[: 1 << len(sites)]


```

- [ ] **Step 2: Create `tests/_graded_cluster.py`:**

```python
"""Build #1038's oracle clusters out of SymmetricTensors and contract them with
tenax.core._graded, so the graded contractor can be checked against Fock."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _fermionic_fock_oracle import _H2, bonds_of, sites_of

from tenax.core._graded import graded_bar, graded_contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor

IN, OUT = FlowDirection.IN, FlowDirection.OUT
FLOW = {"u": OUT, "d": IN, "l": OUT, "r": IN}
SYM = FermionParity()


def _index(n: int, flow, label) -> TensorIndex:
    return TensorIndex.from_charges(
        SYM, np.arange(n, dtype=np.int32) % 2, flow, label=label
    )


def ket_site(R: int, C: int, s, A: np.ndarray, *, aux: bool = False) -> SymmetricTensor:
    """``A[u,d,l,r,p]`` as a SymmetricTensor over its bond legs and ``p``.

    Boundary legs (dimension 1, parity 0) are dropped.  ``aux=True`` prepends
    a dimension-1 **odd** leg ``x`` (flow IN), which is how an odd-parity
    ``A`` is stored as an even tensor; its bra partner is contracted with it.
    """
    name = {}
    for b, (a, x, t, y) in enumerate(bonds_of(R, C)):
        name[(a, x)] = f"b{b}"
        name[(t, y)] = f"b{b}"
    n = sites_of(R, C).index(s)
    keep = [k for k, x in enumerate("udlr") if (s, x) in name]
    arr = A[tuple(slice(None) if k in keep else 0 for k in range(4))]
    idx = [_index(2, FLOW["udlr"[k]], name[(s, "udlr"[k])]) for k in keep]
    idx.append(_index(2, IN, f"p{n}"))
    if aux:
        arr = arr[None]
        idx.insert(
            0, TensorIndex.from_charges(SYM, np.array([1], np.int32), IN, label="x")
        )
    return SymmetricTensor.from_dense(jnp.asarray(arr), tuple(idx))


def bra_site(ket: SymmetricTensor, n: int, *, touched: bool) -> SymmetricTensor:
    """``graded_bar`` of a ket site, legs renamed so they pair with the bra
    network (``b*`` -> ``B*``), the ket (``p`` if untouched) or the operator."""
    m = {lab: "B" + lab[1:] for lab in ket.labels() if lab.startswith("b")}
    m[f"p{n}"] = f"P{n}" if touched else f"p{n}"
    if "x" in ket.labels():
        m["x"] = "x"
    return graded_bar(ket).relabels(m)


def hop_operator(ns: int, nt: int) -> SymmetricTensor:
    """``-(c_s^+ c_t + h.c.)``: legs ``(P_s, P_t, p_t, p_s)`` -- the creation
    half (s, t) then the annihilation half (t, s)."""
    data = np.einsum("ABab->ABba", _H2)
    idx = (
        _index(2, IN, f"P{ns}"),
        _index(2, IN, f"P{nt}"),
        _index(2, OUT, f"p{nt}"),
        _index(2, OUT, f"p{ns}"),
    )
    return SymmetricTensor.from_dense(jnp.asarray(data), idx)


def cluster_value(
    R,
    C,
    bra_As,
    ket_As,
    *,
    op_bond=None,
    order="global",
    ket_aux=None,
    bra_aux=None,
    override=None,
):
    """``<bra| O |ket>`` by graded contraction.

    ``override`` maps a site to a prebuilt ket SymmetricTensor (used as both
    ket and bra) -- e.g. after an SVD regauge.
    ``order="global"``: ``bar(K_N) ... bar(K_1) [O] K_1 ... K_N``.
    ``order="per_site"``: ``[bar(K_s) K_s]`` site by site (what a CTM builds),
    then ``O``.  ``ket_aux`` / ``bra_aux`` name the one site whose ket / bra
    tensor is odd; it is stored with the auxiliary leg (see ``ket_site``).
    """
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    touched = set() if op_bond is None else {n_of[op_bond[0]], n_of[op_bond[1]]}
    override = override or {}
    kets = {
        s: override.get(s) or ket_site(R, C, s, ket_As[s], aux=s == ket_aux)
        for s in sites
    }
    bras = {
        s: bra_site(
            override.get(s) or ket_site(R, C, s, bra_As[s], aux=s == bra_aux),
            n_of[s],
            touched=n_of[s] in touched,
        )
        for s in sites
    }
    op = None if op_bond is None else hop_operator(n_of[op_bond[0]], n_of[op_bond[1]])
    if order == "global":
        seq = (
            [bras[s] for s in reversed(sites)]
            + ([op] if op is not None else [])
            + [kets[s] for s in sites]
        )
    elif order == "per_site":
        seq = [graded_contract(bras[s], kets[s]) for s in sites] + (
            [op] if op is not None else []
        )
    else:
        raise ValueError(order)
    t = seq[0]
    for u in seq[1:]:
        t = graded_contract(t, u)
    assert t.ndim == 0, t.labels()
    return complex(np.asarray(t.todense()).reshape(-1)[0])


def cluster_energy(R, C, As, **kw):
    norm = cluster_value(R, C, As, As, **kw).real
    e = sum(
        cluster_value(R, C, As, As, op_bond=(s, t), **kw).real
        for s, _, t, _ in bonds_of(R, C)
    )
    return e / norm, norm
```

- [ ] **Step 3: Register and write the failing tests.** In `tests/conftest.py`, after `    "test_graded_contract.py": "core",` add `    "test_graded_contract_oracle.py": "core",`. Then create `tests/test_graded_contract_oracle.py`:

```python
"""The reference graded contractor against #1038's exact Fock oracle (#1035).

Design §3.4 showed, in plain numpy, that three rules reproduce Fock with
nothing fitted.  These tests make the same claim about tenax's own
``SymmetricTensor`` + ``tenax.core._graded`` -- including the per-site order
a CTM builds, odd tensors, and an SVD regauge (the design's go/no-go).
"""

from __future__ import annotations

import _graded_cluster
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    fock_psi,
    fock_psi_ordered,
    hop_energy,
    plain_amplitudes,
    random_even_tensors,
    z_gauge,
)
from _graded_cluster import cluster_energy

CLUSTERS = [(2, 2), pytest.param(2, 3, marks=pytest.mark.slow)]


def test_the_ordered_fock_state_is_the_oracle_state_for_even_tensors():
    As = random_even_tensors(2, 2, np.random.default_rng(0))
    np.testing.assert_allclose(
        fock_psi_ordered(2, 2, As), fock_psi(2, 2, As), atol=1e-14
    )


@pytest.mark.parametrize("order", ["global", "per_site"])
@pytest.mark.parametrize("R,C", CLUSTERS)
def test_graded_energy_matches_fock(R, C, order):
    rng = np.random.default_rng(7)
    for _ in range(2):
        As = random_even_tensors(R, C, rng)
        E, norm = cluster_energy(R, C, As, order=order)
        E_fock = hop_energy(R, C, fock_psi(R, C, z_gauge(R, C, As)), fermion=True)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert norm > 0
        assert E == pytest.approx(E_fock, abs=1e-12)
        assert abs(E - E_hcb) > 1e-3  # regime: the bosonic answer is different


def test_graded_energy_matches_fock_for_complex_tensors():
    rng = np.random.default_rng(4)
    re, im = random_even_tensors(2, 2, rng), random_even_tensors(2, 2, rng)
    As = {s: re[s] + 1j * im[s] for s in re}
    E, norm = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert norm > 0
    assert E == pytest.approx(E_fock, abs=1e-12)


def test_regime_an_ungraded_bar_misses_the_oracle(monkeypatch):
    """Rule 3 is load-bearing: today's ``bar()`` in place of ``graded_bar``."""
    monkeypatch.setattr(_graded_cluster, "graded_bar", lambda t: t.bar())
    As = random_even_tensors(2, 2, np.random.default_rng(7))
    E, _ = cluster_energy(2, 2, As)
    E_fock = hop_energy(2, 2, fock_psi(2, 2, z_gauge(2, 2, As)), fermion=True)
    assert abs(E - E_fock) > 1e-3
```

- [ ] **Step 4: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -q -p no:cacheprovider --no-cov`
Expected: `7 passed`: the ordered-state identity, 2×2 global and per-site, 2×3 global and per-site [slow], complex, and the rule-3 regime. There is no red phase here: Task 1's implementation already exists, so these tests *certify* it against physics. The red evidence is Step 6's mutant.

- [ ] **Step 5: Commit**

```bash
git add tests/_fermionic_fock_oracle.py tests/_graded_cluster.py tests/test_graded_contract_oracle.py tests/conftest.py
git commit -m "test(#1035): the graded contractor matches the Fock oracle on SymmetricTensors

Even states in global and per-site (CTM) order, complex tensors, and
the rule-3 regime (today's bar() misses the oracle). z_gauge and
fock_psi_ordered join the oracle module.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

- [ ] **Step 6: Mutation-check the gauge claim**

```bash
F=tests/_fermionic_fock_oracle.py
grep -c "for ax in (0, 2):" $F          # must print 1
sed -i 's/for ax in (0, 2):/for ax in ():/' $F   # drop the gauge entirely
JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -m core -q -p no:cacheprovider --no-cov | tail -3
git checkout $F && git status --short   # clean
```

Expected: three failures: `test_graded_energy_matches_fock[2-2-global]`, `[2-2-per_site]` and `test_graded_energy_matches_fock_for_complex_tensors`. **Do not use "move the Z to the s-side legs `(1, 3)`" as the mutant: it is an equivalent mutant** (verified: still 3 passed). The sign is a per-bond (−1)^n, so either end of the bond carries it.

---

### Task 3: Odd tensors on an auxiliary leg (design §5 step 7's representation, finite clusters)

**Files:**
- Modify: `tests/test_graded_contract_oracle.py` (replace the import block; append a helper and a test)

**Interfaces:**
- Consumes: Task 2's `cluster_value(..., ket_aux=, bra_aux=, order=)`, `fock_psi_ordered`, `z_gauge`; the oracle's `hop_energy_matvec`, `leg_dims`, `bonds_of`, `sites_of`.
- Produces: none (a test only).

- [ ] **Step 1: Replace the import block** (everything from `from __future__ import annotations` up to, but not including, `CLUSTERS = [`) with:

```python
from __future__ import annotations

import itertools

import _graded_cluster
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    bonds_of,
    fock_psi,
    fock_psi_ordered,
    hop_energy,
    hop_energy_matvec,
    leg_dims,
    plain_amplitudes,
    random_even_tensors,
    sites_of,
    z_gauge,
)
from _graded_cluster import cluster_energy, cluster_value

```

- [ ] **Step 2: Append the test** to the end of `tests/test_graded_contract_oracle.py`:

```python


def _odd(shape, rng):
    B = rng.standard_normal(shape)
    for k in itertools.product(*[range(n) for n in shape]):
        if sum(k) % 2 == 0:
            B[k] = 0.0
    return B


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_odd_tensors_on_an_auxiliary_leg_match_fock_in_every_order(R, C):
    """Design §5 step 7's representation: an odd site tensor carries its
    parity on a dimension-1 odd leg, contracted bra-to-ket.  Every tensor is
    then even, so the per-site (CTM) order is as good as the global one."""
    rng = np.random.default_rng(11)
    sites = sites_of(R, C)
    bg = random_even_tensors(R, C, rng)
    Bs = {s: _odd([leg_dims(R, C, *s)[x] for x in "udlr"] + [2], rng) for s in sites}
    bonds = bonds_of(R, C)
    for x in sites:
        for y in sites:
            kx, by = dict(bg), dict(bg)
            kx[x], by[y] = Bs[x], Bs[y]
            px = fock_psi_ordered(R, C, z_gauge(R, C, kx))
            py = fock_psi_ordered(R, C, z_gauge(R, C, by))
            N_f = py @ px
            H_f = py @ hop_energy_matvec(R, C, px, fermion=True)
            for order in ("global", "per_site"):
                kw = dict(ket_aux=x, bra_aux=y, order=order)
                N = cluster_value(R, C, by, kx, **kw).real
                H = sum(
                    cluster_value(R, C, by, kx, op_bond=(s, t), **kw).real
                    for s, _, t, _ in bonds
                )
                assert N == pytest.approx(N_f, abs=1e-12), (x, y, order)
                assert H == pytest.approx(H_f, abs=1e-12), (x, y, order)
```

- [ ] **Step 3: Run it**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -k odd -q -p no:cacheprovider --no-cov`
Expected: `2 passed` (2×2 in about 12 s; 2×3 is slow, about 190 s).

- [ ] **Step 4: Commit, then mutation-check the auxiliary leg's orientation**

```bash
git add tests/test_graded_contract_oracle.py
git commit -m "test(#1035): odd site tensors on an auxiliary leg match Fock in per-site order

With the parity carried on a dimension-1 odd leg contracted bra-to-ket,
every tensor is even, and the per-site (CTM) order reproduces the
non-local string for all (x, y).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
F=tests/_graded_cluster.py
grep -c 'IN, label="x")' $F             # must print 1
sed -i 's/IN, label="x")/OUT, label="x")/' $F
JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -k "odd and 2-2" -q -p no:cacheprovider --no-cov | tail -3
git checkout $F && git status --short
```

Expected: the mutant fails `test_odd_tensors_on_an_auxiliary_leg_match_fock_in_every_order[2-2]` (the norm flips sign). The auxiliary leg must be IN on the ket so that its bra partner, OUT, comes first.

---

### Task 4: `graded_svd` and the go/no-go

**Files:**
- Modify: `src/tenax/core/_graded.py` (append a function)
- Modify: `tests/test_graded_contract.py` (append tests)
- Modify: `tests/test_graded_contract_oracle.py` (replace the import block; append the go/no-go)

**Interfaces:**
- Consumes: `tenax.linalg.svd(t, left, right, new_bond_label=..., max_singular_values=...) -> (U, S, Vh, _)`; `tenax.algorithms._ctm_tensor_projector_2x2._scale_bond_by_diag(T, diag, bond_label)`; Task 2's `ket_site`, `cluster_energy(..., override=)`.
- Produces: `graded_svd(t, left_labels, right_labels, new_bond_label, **svd_kwargs) -> (U, S, Vh, info)`, with `U` as `(left..., bond)` and `Vh` as `(bond, right...)`.

- [ ] **Step 1: Write the failing unit tests** — append to `tests/test_graded_contract.py`:

```python


def _scaled(U, S, bond):
    from tenax.algorithms._ctm_tensor_projector_2x2 import _scale_bond_by_diag

    return _scale_bond_by_diag(U, S, bond)


def test_graded_svd_reconstructs_when_storage_order_is_not_left_plus_right():
    from tenax.core._graded import graded_svd

    fp, ch = FermionParity(), [0, 1, 0, 1]
    T = _rand(
        [
            _idx(fp, ch, IN, "p"),
            _idx(fp, ch, OUT, "q"),
            _idx(fp, ch, IN, "r"),
            _idx(fp, ch, OUT, "s"),
        ],
        5,
    )
    left, right = ["p", "r"], ["q", "s"]
    U, S, Vh, _ = graded_svd(T, left, right, "k")
    assert U.labels() == ("p", "r", "k") and Vh.labels() == ("k", "q", "s")
    target = graded_reorder(T, left + right)
    assert _maxdiff(graded_contract(_scaled(U, S, "k"), Vh), target) < 1e-12


def test_regime_plain_svd_reorders_sign_free_and_does_not_reconstruct():
    from tenax.linalg import svd

    fp, ch = FermionParity(), [0, 1, 0, 1]
    T = _rand(
        [
            _idx(fp, ch, IN, "p"),
            _idx(fp, ch, OUT, "q"),
            _idx(fp, ch, IN, "r"),
            _idx(fp, ch, OUT, "s"),
        ],
        5,
    )
    U, S, Vh, _ = svd(T, ["p", "r"], ["q", "s"], new_bond_label="k")
    target = graded_reorder(T, ["p", "r", "q", "s"])
    assert _maxdiff(graded_contract(_scaled(U, S, "k"), Vh), target) > 1e-3
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -k svd -q -p no:cacheprovider --no-cov`
Expected: `test_graded_svd_reconstructs...` FAILS with `ImportError: cannot import name 'graded_svd'`. `test_regime_plain_svd...` PASSES: it documents today's hazard (wrong by ~5.1).

- [ ] **Step 3: Implement** — append to `src/tenax/core/_graded.py`:

```python


def graded_svd(
    t: SymmetricTensor, left_labels, right_labels, new_bond_label: str, **svd_kwargs
):
    """``tenax.linalg.svd`` under graded semantics.

    ``svd`` matricizes with a sign-free reorder, which is only correct when
    the legs are already in ``left + right`` order, so reorder them first with
    the Koszul sign.  ``U`` is returned as ``(left..., bond)`` and ``Vh`` as
    ``(bond, right...)``, so ``graded_contract(U * S, Vh)`` reconstructs
    ``graded_reorder(t, left + right)``; ``svd``'s bond orientation (``U``'s
    bond leg OUT) is exactly rule 2's +1 pairing, so no twist is needed.
    """
    from tenax.linalg import svd

    left, right = list(left_labels), list(right_labels)
    return svd(
        graded_reorder(t, left + right),
        left,
        right,
        new_bond_label=new_bond_label,
        **svd_kwargs,
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -q -p no:cacheprovider --no-cov`
Expected: `13 passed`.

- [ ] **Step 5: Add the go/no-go test.** In `tests/test_graded_contract_oracle.py`, replace the import block (from `from __future__ import annotations` up to, not including, `CLUSTERS = [`) with:

```python
from __future__ import annotations

import itertools

import _graded_cluster
import jax.numpy as jnp
import numpy as np
import pytest
from _fermionic_fock_oracle import (
    bonds_of,
    fock_psi,
    fock_psi_ordered,
    hop_energy,
    hop_energy_matvec,
    leg_dims,
    plain_amplitudes,
    random_even_tensors,
    sites_of,
    z_gauge,
)
from _graded_cluster import cluster_energy, cluster_value, ket_site

from tenax.algorithms._ctm_tensor_projector_2x2 import _scale_bond_by_diag
from tenax.core._graded import graded_contract, graded_reorder, graded_svd

```

Then append:

```python


def _regauge_bond0(As, *, reorder):
    """Merge sites (0,0)-(0,1) over bond b0, SVD back with sqrt(S) on each
    side (exact rank), and restore each site's leg order with ``reorder``."""
    s, t = (0, 0), (0, 1)
    Ks, Kt = ket_site(2, 2, s, As[s]), ket_site(2, 2, t, As[t])
    M = graded_contract(Ks, Kt)
    left = [lab for lab in Ks.labels() if lab != "b0"]
    right = [lab for lab in Kt.labels() if lab != "b0"]
    _, S, _, _ = graded_svd(M, left, right, "b0")
    keep = int(np.sum(np.asarray(S) > 1e-12 * float(np.max(S))))
    U, S, Vh, _ = graded_svd(M, left, right, "b0", max_singular_values=keep)
    r = jnp.sqrt(S)
    return {
        s: reorder(_scale_bond_by_diag(U, r, "b0"), list(Ks.labels())),
        t: reorder(_scale_bond_by_diag(Vh, r, "b0"), list(Kt.labels())),
    }


def _sign_free(t, labels):
    return t.permute_legs(tuple(t.labels().index(lab) for lab in labels))


def test_go_no_go_an_svd_regauge_leaves_the_state_unchanged():
    """Design §5 step 2: linalg under graded semantics.  Splitting a bond and
    re-absorbing sqrt(S) is a gauge move; the energy must not change."""
    As = random_even_tensors(2, 2, np.random.default_rng(3))
    E0, n0 = cluster_energy(2, 2, As)
    E1, n1 = cluster_energy(
        2, 2, As, override=_regauge_bond0(As, reorder=graded_reorder)
    )
    assert E1 == pytest.approx(E0, abs=1e-12)
    assert n1 == pytest.approx(n0, rel=1e-12)
    E_bad, _ = cluster_energy(2, 2, As, override=_regauge_bond0(As, reorder=_sign_free))
    assert abs(E_bad - E0) > 1e-3  # regime: a sign-free reorder around the SVD is wrong
```

- [ ] **Step 6: Run the whole phase**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py tests/test_graded_contract_oracle.py tests/test_fermionic_fock_oracle.py -m core -q -p no:cacheprovider --no-cov`
Expected: `28 passed, 9 deselected, 1 xfailed`, in about 35 s.

- [ ] **Step 7: Commit, then mutation-check the reorder-first**

```bash
git add src/tenax/core/_graded.py tests/test_graded_contract.py tests/test_graded_contract_oracle.py
git commit -m "feat(#1035): graded_svd; go/no-go -- an SVD regauge leaves the state unchanged

linalg.svd matricizes with a sign-free reorder, so graded_svd reorders
to left+right with the Koszul sign first. svd's bond orientation (U's
bond leg OUT) is rule 2's +1 pairing: no twist is needed. A sqrt(S)
regauge of a 2x2 bond keeps the energy and norm to 1e-12; a sign-free
reorder around it moves the energy by ~0.37.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
F=src/tenax/core/_graded.py
grep -c "        graded_reorder(t, left + right)," $F   # must print 1
sed -i 's/        graded_reorder(t, left + right),/        t,/' $F
JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -k svd -q -p no:cacheprovider --no-cov | tail -3
git checkout $F && git status --short
```

Expected: the mutant fails `test_graded_svd_reconstructs_when_storage_order_is_not_left_plus_right`.

---

### Task 5: Measure the cost (design D0: "unmeasured")

**Files:**
- Create: `examples/profile_graded_contract_1035.py`

**Interfaces:**
- Consumes: Task 1's `graded_contract`; `contract`.
- Produces: a CLI that prints one line per arm and a ratio line.

- [ ] **Step 1: Create the script:**

```python
"""Cost of the reference graded contractor vs today's sign-free ``contract``
(#1035 design §4 D0: B's cost is unmeasured until this runs).

A CTM-shaped pairwise contraction on FermionParity tensors: an edge tensor
``T(l, m, r)`` (chi, D^2, chi) against a double-layer-shaped ``a(m, u, d, s)``.
Reports eager wall time (median of repeats, after a warm-up) and jit compile
time (``jax.clear_caches()`` before each arm, so each compile is fresh).

    JAX_PLATFORMS=cpu uv run python examples/profile_graded_contract_1035.py --chi 16 --d2 4
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import numpy as np

from tenax.contraction.contractor import contract
from tenax.core._graded import graded_contract
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionParity
from tenax.core.tensor import SymmetricTensor


def _idx(n, flow, label):
    return TensorIndex.from_charges(
        FermionParity(), np.arange(n, dtype=np.int32) % 2, flow, label=label
    )


def _operands(chi: int, d2: int):
    IN, OUT = FlowDirection.IN, FlowDirection.OUT
    T = SymmetricTensor.random_normal(
        indices=(_idx(chi, IN, "l"), _idx(d2, OUT, "m"), _idx(chi, OUT, "r")),
        key=jax.random.PRNGKey(0),
    )
    a = SymmetricTensor.random_normal(
        indices=(
            _idx(d2, IN, "m"),
            _idx(d2, OUT, "u"),
            _idx(d2, IN, "d"),
            _idx(d2, OUT, "s"),
        ),
        key=jax.random.PRNGKey(1),
    )
    return T, a


def _arms(T, a):
    out = tuple(lab for lab in T.labels() + a.labels() if lab != "m")
    return {
        "sign_free": lambda x, y: contract(x, y, output_labels=out).todense(),
        "graded": lambda x, y: graded_contract(x, y).todense(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--d2", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    T, a = _operands(args.chi, args.d2)
    res = {}
    for name, fn in _arms(T, a).items():
        fn(T, a).block_until_ready()  # warm-up
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn(T, a).block_until_ready()
            times.append(time.perf_counter() - t0)
        jax.clear_caches()
        jitted = jax.jit(fn)
        t0 = time.perf_counter()
        jitted(T, a).block_until_ready()
        compile_s = time.perf_counter() - t0
        res[name] = (statistics.median(times), compile_s)
        print(
            f"{name:10s} eager median {res[name][0] * 1e3:8.2f} ms   jit first call {compile_s:7.3f} s",
            flush=True,
        )
    (e0, c0), (e1, c1) = res["sign_free"], res["graded"]
    print(
        f"ratio graded/sign_free: eager {e1 / e0:.2f}x   compile {c1 / c0:.2f}x   (chi={args.chi}, D^2={args.d2})"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it at the two sizes and keep the output**

```bash
JAX_PLATFORMS=cpu uv run python examples/profile_graded_contract_1035.py --chi 16 --d2 4 2>&1 | tee /tmp/graded_cost_chi16.txt
JAX_PLATFORMS=cpu uv run python examples/profile_graded_contract_1035.py --chi 32 --d2 4 2>&1 | tee /tmp/graded_cost_chi32.txt
```

Expected shape: two arm lines and one `ratio graded/sign_free:` line per run. A pre-validation run at χ=16 printed `eager 1.20x   compile 0.92x`. Record whatever the run prints, not these numbers.

- [ ] **Step 3: Commit**

```bash
git add examples/profile_graded_contract_1035.py
git commit -m "bench(#1035): graded vs sign-free contraction cost, eager and jit compile

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

---

### Task 6: Full check, the go/no-go record, and the PR

**Files:**
- Modify (on branch `design/1035-sign-convention`, PR #1036): `docs/plans/2026-09-23-1035-fermionic-sign-convention-design.md`, appending `## 8. Phase 1 result`.

- [ ] **Step 1: Run the whole phase, including slow**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py tests/test_graded_contract_oracle.py tests/test_fermionic_fock_oracle.py -q -p no:cacheprovider --no-cov`
Expected: all pass, except `test_tenax_double_layer_matches_the_fermionic_oracle` stays xfailed (it flips in Phase 2, not here).

- [ ] **Step 2: Confirm production code is untouched**

Run: `git diff --stat origin/main -- src/ | tail -1`
Expected: `1 file changed`, and `git diff --name-only origin/main -- src/` prints only `src/tenax/core/_graded.py`. Also `git grep -n "_graded" -- src/ ':!src/tenax/core/_graded.py'` prints nothing, because no production module imports it.

- [ ] **Step 3: Push and open the PR (don't arm)**

```bash
git push -u origin feat/1035-graded-contractor-phase1
gh pr create --base main --title "feat(#1035): reference graded contractor + go/no-go (phase 1 of #1036)" --body-file /tmp/pr_phase1.md
```

`/tmp/pr_phase1.md` must contain: the 🤖 marker line; one paragraph saying what the PR adds (the private module, three test files, the benchmark) and that no production path changes; the go/no-go results (the regauge Δ, the odd-tensor per-site result, and Task 5's two ratio lines verbatim); the mutants and which test each broke; `Related: #1035, #1036, #1037, #1038`; and the PR attribution footer (`🤖 Generated with [Claude Code](https://claude.com/claude-code)` plus the session URL).

- [ ] **Step 4: Record the decision on the design PR.** In the #1036 worktree (`design/1035-sign-convention`), append to the design doc:

```markdown
## 8. Phase 1 result (go/no-go)

Implemented in PR #<phase-1 PR number> on `origin/main` @ `<sha>`. The reference graded contractor (`tenax.core._graded`, beside the production path) was checked against the #1038 oracle on tenax's own `SymmetricTensor`s:

| Check | Result |
|---|---|
| even states, global and per-site order, 2×2 / 2×3 | matches Fock to ≤1e−12 |
| complex tensors (2×2) | matches Fock |
| odd tensors on an auxiliary leg, every (x, y), per-site order | matches Fock (§5 step 7's representation holds on finite clusters) |
| SVD regauge of a bond (go/no-go, §5 step 2) | energy and norm unchanged to 1e−12; a sign-free reorder around the SVD moves the energy by ≈0.37 |
| cost, χ=16 / χ=32 (Task 5) | <paste the two ratio lines verbatim> |

**Verdict:** GO on correctness: linalg needs no change of its own beyond a graded reorder *into* `left + right` order (`graded_svd`), since `svd`'s bond orientation is rule 2's +1 pairing. The cost verdict is the user's call, from the table above.
```

Replace the two `<...>` fields with the real PR number and SHA, and paste the ratio lines verbatim. Then commit with the trailers, push, and post a 🤖-marked `@codex review` comment on #1036.

- [ ] **Step 5: Hand back.** Report the PR link, the four go/no-go numbers and the cost ratios to the user. Phase 2's plan (§5 step 3) waits for their GO.
