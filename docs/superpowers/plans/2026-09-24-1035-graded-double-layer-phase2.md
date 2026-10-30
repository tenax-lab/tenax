# Graded double layer, Phase 2 (design §5 step 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put fusion and the double layer on the graded path. A graded fuse of ket/bra leg pairs, and a graded twin of production's double-layer builders with the same labels, flows and fused legs, are certified against the exact Fock oracle on finite clusters for norm, energy and every element of the two-site RDM. **No production code path changes.**

**Architecture:** One new private module, `src/tenax/algorithms/_graded_double_layer.py`. It lives in `algorithms/` because `fuse_indices` does. It provides:
- `graded_fuse_pair` and `graded_split_pair`, which do a graded reorder, then a one-end pair sign, then plain `fuse_indices` / `split_index`;
- `build_graded_double_layer`, which composes `graded_bar` (rule 3), `graded_contract` and `graded_fuse_pair`.

`src/tenax/core/_graded.py` gains Phase 1's two outstanding guards. The oracle tests contract the production-shaped graded double layers site by site (the order a CTM builds) and compare against Phase 1's certified state.

**Tech Stack:** Python ≥3.11, JAX (x64), numpy ≥1.26, pytest, `uv`, ruff pre-commit.

**Spec:** `docs/plans/2026-09-23-1035-fermionic-sign-convention-design.md` (PR #1036 @ `3fb24c9`). This plan implements §5 step 3 and §8's Phase 2 entry criteria 2 and 3, on the Phase 1 code (PR #1039, merged as `13d59c2`).

**Scope note:**
- **Step 3 only.** Step 4 (re-deriving the 9 CTM reorders, and with it an RDM on a CTM environment) touches the production CTM and gets its own plan.
- **Finite clusters stand in for the environment.** Here "RDM and energy on the graded path" means the finite-cluster RDM and energy from graded double layers. That is the part the oracle can certify exactly.

**Pre-validated:** Every code block below was run on PR #1039's head `7695282` in a scratch worktree before this plan was written. The results:
- the core subset of the three graded test files gives 48 passed and 4 deselected, in about 70 s. The same three files plus #1038's oracle file give 65 passed and 2 xfailed;
- the mutants listed in Task 5 fail the tests stated.

Expect the same. If a result differs, stop and investigate rather than adjusting the test.

## Rulings this plan makes against the design text (for the reviewer)

1. **"Emit the pairs interleaved so the fuse is a pure reshape" (§5 step 3) is not enough.** A fused ket⊗bra leg under rule 2 is twisted by (−1)^(p_ket+p_bra) when it is IN on the left. The unfused pair needs (−1)^(p_ket). Also, the fused basis has to be enumerated in the same `(ket, bra)` order on both ends of the bond, while nesting wants `(bra, ket)` on one of them. Both are fixed by one sign on exactly one end of every bond, the end whose ket leg is IN: **(−1)^(p_k·p_K + p_k)**. This was found by prototype and matches a derivation on paper.
   - **Exact:** it reproduces the unfused graded contraction to 1e−15 on FermionParity and FermionicU1, in both operand orders.
   - **Wrong variants miss:** a swap sign alone misses by 7–18, and no sign misses by 7.
   - **Either end would do:** the sign depends only on a contracted pair's parity. Tying it to the ket leg's flow makes the rule local.
2. **The oracle state is the ket-level one, `Fock(z_gauge(As))`.** That's the state Phase 1's `graded_contract` on kets defines, and the one gates and `graded_svd` act on. A double layer must measure that same state, or the CTM measures a different state from the one simple update optimizes.
   - The same fuse with a bra-parity twist instead reproduces the *un-gauged* `Fock(As)`, off by 1.9 from the ket-level state. #1038's test harness would also point there, because its `ident` closure of the physical legs adds a pairing that Phase 1's direct bra–ket contraction does not.
   - Both are legitimate states related by a bond gauge. Consistency with the ket level decides between them.
3. **#1038's strict xfail and its #1037 characterization tests stay as they are.** They test production (`_build_double_layer_*` with sign-free `contract`), which this phase does not change. §5 step 3's "the strict xfail flips" moves to the step that routes production through the graded double layer (step 4 for the CTM, step 6 for the default). This phase adds the passing graded twin instead.
4. **§8 entry criterion 1 (two odd tensors in one ket or bra) moves to step 7**, where odd insertions enter the CTM. Every tensor here is even. Criteria 2 and 3 are Task 1.

## Global Constraints

- **No production behaviour change.** The only `src/` files touched are the new `src/tenax/algorithms/_graded_double_layer.py` and `src/tenax/core/_graded.py`, and nothing else in `src/` imports either.
- **Private modules** (leading underscore): no `src/tenax/__init__.py` `__all__` or `README.md` change.
- **numpy ≥1.26:** no NumPy-2-only calls.
- **The required gate stays fast:** new tests run in `pytest -m core`. 2×3 cases carry `@pytest.mark.slow` (through `CLUSTERS`). `tests/test_graded_double_layer.py` is registered as `core` in `tests/conftest.py`. Keep FermionicU1 double-layer fixtures at bond dimension 2: at dimension 5 one structure test costs about 2.5 min of block-sparse compile.
- **Run everything with `uv run`** from the worktree, with `JAX_PLATFORMS=cpu`.
- **Git:**
  - branch off `origin/main`, which has Phase 1 (#1039, `13d59c2`); if it falls behind, `git merge origin/main`, never rebase;
  - one PR, never push to `main`, **never pass `--delete-branch`**;
  - **don't arm `--auto`** (the user decides), and read Codex's review before anything is armed.
- **Commit trailers** (every commit):
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap`.
- **Any GitHub text an agent posts** carries `> 🤖 **AI-generated comment** — written by Claude Code, posted by @yingjerkao.` (hook-enforced).
- **Mutation discipline:** commit before mutating, assert each anchor is unique, restore with `git checkout <file>`, and name the test that failed.

## Review Focus

1. **A ket/bra pair given with equal flows** must be refused, not fused into a leg that cannot contract: `test_a_pair_with_equal_flows_is_refused` (Task 2).
2. **A misspelt label** in `graded_fuse_pair`, `graded_split_pair` or `twist_legs` must raise, not silently drop a sign: `test_an_unknown_label_is_refused` (Task 2) and `test_twist_legs_refuses_a_label_the_tensor_does_not_have` (Task 1).
3. **Splitting a leg that was never fused** must raise, not scramble data: `test_splitting_a_leg_that_was_never_fused_is_refused` (Task 2).
4. **Bosonic tensors** through the graded fuse must behave exactly like production: `test_a_bosonic_pair_fuses_as_production_does` (Task 2).
5. **Under `jax.jit`** the graded double layer must trace and agree with eager: `test_the_graded_double_layer_traces_under_jit` (Task 3).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/tenax/core/_graded.py` (modify) | Refuse mixed fermionic/bosonic operands in `graded_contract`; refuse unknown labels in `twist_legs` | 1 |
| `src/tenax/algorithms/_graded_double_layer.py` (create) | `graded_fuse_pair`, `graded_split_pair` (Task 2); `build_graded_double_layer` (Task 3) | 2, 3 |
| `tests/test_graded_contract.py` (modify) | Tests for the two guards | 1 |
| `tests/test_graded_double_layer.py` (create) | Algebra: fuse commutes with graded contraction, split inverts fuse, guards, production's structure, jit | 2, 3 |
| `tests/conftest.py` (modify) | Register `test_graded_double_layer.py` as `core` | 2 |
| `tests/_graded_cluster.py` (modify) | `bond_operator` made public; `production_site`, `double_layer_value`, `double_layer_energy` | 4 |
| `tests/test_graded_contract_oracle.py` (modify) | Graded double layer vs Fock: energy (2×2, 2×3 slow, complex), every two-site RDM element, pair-sign regime | 4 |

---

### Task 0: Worktree and branch

- [ ] **Step 1: Pick the base and create the worktree**

```bash
cd /home/yjkao/tenax && git fetch origin
git log origin/main --oneline | grep -m1 "#1039"   # Phase 1 merged as 13d59c2
git worktree add -b feat/1035-graded-double-layer-phase2 /home/yjkao/tenax-1035-phase2 origin/main
cd /home/yjkao/tenax-1035-phase2 && pre-commit install && uv sync -q
```

- [ ] **Step 2: Baseline**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py tests/test_graded_contract_oracle.py -m core -q -p no:cacheprovider --no-cov`
Expected: `23 passed` (13 algebra + 10 oracle core cases), no failures.

### Task 1: The two Phase 2 entry guards (design §8, criteria 2 and 3)

**Files:**
- Modify: `src/tenax/core/_graded.py` (`twist_legs`, `graded_contract`)
- Test: `tests/test_graded_contract.py`

**Interfaces:**
- Consumes: `graded_contract(a, b)` and `twist_legs(t, labels)` from Phase 1.
- Produces:
  - `graded_contract` raises `TypeError` matching `"one fermionic and one bosonic"` when exactly one operand is fermionic.
  - `twist_legs` raises `ValueError` matching `"no leg labelled"` for any label that isn't a leg of `t`.

- [ ] **Step 1: Write the failing tests.** Change the `_graded` import line to

```python
from tenax.core._graded import graded_bar, graded_contract, graded_reorder, twist_legs
```

and append to `tests/test_graded_contract.py`:

```python
@pytest.mark.parametrize("order", ["fermionic_first", "bosonic_first"])
def test_a_mixed_fermionic_and_bosonic_pair_is_refused(order):
    """Design §8 Phase 2 entry criterion 2: production ``contract`` accepts
    this pair silently; the graded contractor must not."""
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    u1 = U1Symmetry()
    z = _rand([_idx(u1, [-1, 0, 1], IN, "z")], 9)
    a, b = (A, z) if order == "fermionic_first" else (z, A)
    with pytest.raises(TypeError, match="one fermionic and one bosonic"):
        graded_contract(a, b)


def test_twist_legs_refuses_a_label_the_tensor_does_not_have():
    """Design §8 Phase 2 entry criterion 3: an unknown label used to be
    ignored, dropping its sign silently."""
    A, _, _ = _triple(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="no leg labelled"):
        twist_legs(A, ["a", "nope"])
    assert _maxdiff(twist_legs(A, []), A) == 0.0
```

- [ ] **Step 2: Run them and see them fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -k "mixed or refuses_a_label" -q -p no:cacheprovider --no-cov`
Expected: 3 FAILED. The mixed pair does not raise, since today a fermionic-then-bosonic pair is twisted and a bosonic-then-fermionic one falls back to `contract`. `twist_legs` ignores `"nope"`.

- [ ] **Step 3: Implement.** In `src/tenax/core/_graded.py`, replace `twist_legs`'s docstring and first line with:

```python
def twist_legs(t: SymmetricTensor, labels) -> SymmetricTensor:
    """Multiply each block by ``(-1)**(sum of parities on the named legs)``.

    Every label must name a leg of ``t``: a misspelt label would otherwise
    drop its sign silently."""
    labels = set(labels)
    unknown = labels - set(t.labels())
    if unknown:
        raise ValueError(f"twist_legs: no leg labelled {sorted(unknown, key=str)}")
    axes = [i for i, lab in enumerate(t.labels()) if lab in labels]
```

and in `graded_contract`, directly after the existing `DenseTensor` `TypeError` block and before `b_labels = set(b.labels())`:

```python
    if _is_graded(a) != _is_graded(b):
        raise TypeError(
            "graded_contract got one fermionic and one bosonic operand; a "
            "bosonic leg has no parity, so the graded sign is undefined"
        )
```

- [ ] **Step 4: Run the file**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py -q -p no:cacheprovider --no-cov`
Expected: `16 passed`.

- [ ] **Step 5: Run the oracle file** (every existing `twist_legs` caller passes real labels)

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -m core -q -p no:cacheprovider --no-cov`
Expected: `10 passed, 3 deselected`.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/core/_graded.py tests/test_graded_contract.py
git commit -m "feat(#1035): graded_contract refuses mixed operands, twist_legs unknown labels

Design §8's Phase 2 entry criteria 2 and 3.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

---

### Task 2: The graded fuse and split

**Files:**
- Create: `src/tenax/algorithms/_graded_double_layer.py`
- Create: `tests/test_graded_double_layer.py`
- Modify: `tests/conftest.py` (register the new test file as `core`)

**Interfaces:**
- Consumes:
  - `graded_reorder`, `_scale_blocks(t, exponent)` and `_is_graded` from `tenax.core._graded`;
  - `fuse_indices(t, axis_a, axis_b, fused_label, fused_flow)` and `split_index(t, axis)` from `tenax.algorithms._tensor_utils`;
  - production's `_fuse_pair_by_label` (for the regime test).
- Produces:
  - `graded_fuse_pair(t, ket, bra, fused_label) -> SymmetricTensor`: the fused leg sits where the first of the two legs was, with the bra leg's flow.
  - `graded_split_pair(t, fused_label) -> SymmetricTensor`: ket then bra, where the fused leg was.
  - The private `_pair_sign(t, k)`: Task 4's regime test monkeypatches it by this name.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_graded_double_layer.py`:

```python
"""Graded fusion and the graded double layer (#1035, design §5 step 3).

The physics check against the exact Fock oracle is in
``test_graded_contract_oracle.py``; this file pins the algebra: a graded
fuse must commute with graded contraction, invert under split, and build a
double layer with production's structure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_tensor_init import _fuse_pair_by_label
from tenax.algorithms._graded_double_layer import graded_fuse_pair, graded_split_pair
from tenax.core._graded import graded_contract, graded_reorder
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import FermionicU1, FermionParity, U1Symmetry
from tenax.core.tensor import SymmetricTensor

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


def _maxdiff(s, t):
    return float(jnp.max(jnp.abs(s.todense() - t.todense())))


def _bond_pair(sym, ch):
    """``T1`` holds ket leg ``k`` (IN) and bra leg ``K`` (OUT) with a free leg
    between them, so the fuse's reorder is a real permutation; ``T2`` holds
    their partners (flows flipped) in the opposite order."""
    T1 = _rand(
        [
            _idx(sym, ch, IN, "k"),
            _idx(sym, ch, OUT, "a"),
            _idx(sym, ch, OUT, "K"),
            _idx(sym, ch, IN, "b"),
        ],
        0,
    )
    T2 = _rand(
        [_idx(sym, ch, IN, "c"), _idx(sym, ch, IN, "K"), _idx(sym, ch, OUT, "k")], 1
    )
    return T1, T2


@pytest.mark.parametrize("sym,ch", SYMS)
@pytest.mark.parametrize("first", ["T1", "T2"])
def test_contracting_fused_legs_equals_contracting_the_pairs(sym, ch, first):
    T1, T2 = _bond_pair(sym, ch)
    F1 = graded_fuse_pair(T1, "k", "K", "f")
    F2 = graded_fuse_pair(T2, "k", "K", "f")
    x, y, fx, fy = (T1, T2, F1, F2) if first == "T1" else (T2, T1, F2, F1)
    ref = graded_contract(x, y)
    got = graded_contract(fx, fy)
    assert _maxdiff(graded_reorder(got, ref.labels()), ref) < 1e-12


def test_regime_a_sign_free_fuse_does_not_commute_with_graded_contraction():
    """Production's fuse (``fuse_indices`` after a sign-free move) is what the
    graded fuse replaces; it must fail the property above."""
    T1, T2 = _bond_pair(FermionParity(), [0, 1, 0, 1])
    ref = graded_contract(T1, T2)
    F1 = _fuse_pair_by_label(T1, "k", "K", "f", OUT)
    F2 = _fuse_pair_by_label(T2, "k", "K", "f", IN)
    got = graded_contract(F1, F2)
    assert _maxdiff(graded_reorder(got, ref.labels()), ref) > 1e-3


@pytest.mark.parametrize("sym,ch", SYMS)
def test_split_inverts_fuse(sym, ch):
    for T in _bond_pair(sym, ch):
        F = graded_fuse_pair(T, "k", "K", "f")
        back = graded_split_pair(F, "f")
        assert set(back.labels()) == set(T.labels())
        assert _maxdiff(graded_reorder(back, T.labels()), T) == 0.0


def test_the_fused_leg_takes_the_bra_legs_flow_and_place():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    F = graded_fuse_pair(T1, "k", "K", "f")
    assert F.labels() == ("f", "a", "b")
    assert F.indices[0].flow == OUT


def test_a_pair_with_equal_flows_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="same flow"):
        graded_fuse_pair(T1, "k", "b", "f")


def test_an_unknown_label_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="no leg labelled"):
        graded_fuse_pair(T1, "k", "nope", "f")
    with pytest.raises(ValueError, match="no leg labelled"):
        graded_split_pair(T1, "nope")


def test_splitting_a_leg_that_was_never_fused_is_refused():
    T1, _ = _bond_pair(FermionParity(), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="not a fused leg"):
        graded_split_pair(T1, "a")


def test_a_bosonic_pair_fuses_as_production_does():
    u1 = U1Symmetry()
    q = [-1, 0, 1]
    T = _rand([_idx(u1, q, IN, "k"), _idx(u1, q, OUT, "a"), _idx(u1, q, OUT, "K")], 3)
    graded = graded_fuse_pair(T, "k", "K", "f")
    plain = _fuse_pair_by_label(graded_reorder(T, ["k", "K", "a"]), "k", "K", "f", OUT)
    assert graded.labels() == plain.labels()
    assert _maxdiff(graded, plain) == 0.0
```

In `tests/conftest.py`, add the new file to `_FILE_MARKERS` right after `"test_graded_contract_oracle.py": "core",`:

```python
    "test_graded_double_layer.py": "core",
```

- [ ] **Step 2: Run and see them fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_double_layer.py -q -p no:cacheprovider --no-cov`
Expected: collection error `ModuleNotFoundError: No module named 'tenax.algorithms._graded_double_layer'`.

- [ ] **Step 3: Implement.** Create `src/tenax/algorithms/_graded_double_layer.py`:

```python
"""Graded fusion of ket/bra leg pairs and the graded double layer (#1035,
design §5 step 3).

Beside the production path: nothing in ``src/`` imports this module yet.

**The fuse rule.**  A double-layer leg fuses a ket leg ``k`` with its bra
partner ``K`` (opposite flows).  Bringing them together is a graded reorder
(rule 1), and the fused leg takes the bra leg's flow, as production's
``_build_double_layer_tensor`` does.  That is not yet enough: rule 2 twists
a contracted leg by the parity of the whole leg when it is IN on the left
operand, but the unfused pair needs ``(-1)^{p_k}`` alone, and the fused
basis must be enumerated in the same ``(k, K)`` order on both ends of the
bond.  Both are repaired by one sign on exactly one end of every bond --
the end whose ket leg is IN:

    (-1)^{p_k p_K + p_k}

``p_k p_K`` is the Koszul swap between the ``(K, k)`` order nesting needs
and the ``(k, K)`` order the fused basis uses; ``p_k`` converts the fused
leg's twist into the ket leg's.  The sign depends only on the parity of a
contracted pair, so either end would do; tying it to the ket leg's flow
makes the rule local.  With it, contracting two fused tensors equals
contracting them unfused, in either operand order.
"""

from __future__ import annotations

from tenax.algorithms._tensor_utils import fuse_indices, split_index
from tenax.core._graded import _is_graded, _scale_blocks, graded_reorder
from tenax.core.index import FlowDirection
from tenax.core.tensor import SymmetricTensor

IN = FlowDirection.IN


def _pair_sign(t: SymmetricTensor, k: int) -> SymmetricTensor:
    """``(-1)^{p_k p_K + p_k}`` for the ket leg at axis ``k`` and its bra
    partner at ``k + 1``."""
    return _scale_blocks(t, lambda p: p[k] * p[k + 1] + p[k])


def graded_fuse_pair(t: SymmetricTensor, ket, bra, fused_label) -> SymmetricTensor:
    """Fuse ket leg ``ket`` and bra leg ``bra`` into ``fused_label``, graded.

    The fused leg sits where the first of the two legs was and takes the
    bra leg's flow.  On a bosonic tensor this is plain ``fuse_indices``
    after bringing the legs together.
    """
    if not isinstance(t, SymmetricTensor):
        raise TypeError("graded_fuse_pair needs a SymmetricTensor")
    labels = list(t.labels())
    missing = [lab for lab in (ket, bra) if lab not in labels]
    if missing:
        raise ValueError(f"graded_fuse_pair: no leg labelled {missing}")
    flow_k = t.indices[labels.index(ket)].flow
    flow_b = t.indices[labels.index(bra)].flow
    if flow_k == flow_b:
        raise ValueError(
            f"graded_fuse_pair: {ket!r} and {bra!r} have the same flow; a "
            "ket/bra pair has opposite flows"
        )
    rest = [lab for lab in labels if lab not in (ket, bra)]
    i = min(labels.index(ket), labels.index(bra))
    t = graded_reorder(t, rest[:i] + [ket, bra] + rest[i:])
    if _is_graded(t) and flow_k == IN:
        t = _pair_sign(t, i)
    return fuse_indices(t, i, i + 1, fused_label, flow_b)


def graded_split_pair(t: SymmetricTensor, fused_label) -> SymmetricTensor:
    """Inverse of :func:`graded_fuse_pair`: the ket and bra legs come back,
    in that order, where the fused leg was."""
    labels = list(t.labels())
    if fused_label not in labels:
        raise ValueError(f"graded_split_pair: no leg labelled {fused_label!r}")
    i = labels.index(fused_label)
    info = t.indices[i].fuse_info
    if info is None:
        raise ValueError(f"graded_split_pair: {fused_label!r} is not a fused leg")
    out = split_index(t, i)
    if _is_graded(out) and info.parent_indices[0].flow == IN:
        out = _pair_sign(out, i)  # the sign is its own inverse
    return out
```

- [ ] **Step 4: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_double_layer.py -q -p no:cacheprovider --no-cov`
Expected: `12 passed`. The sign-free regime test passes because it asserts a difference greater than 1e−3.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_graded_double_layer.py tests/test_graded_double_layer.py tests/conftest.py
git commit -m "feat(#1035): graded fuse and split of ket/bra leg pairs

The fused leg takes the bra leg's flow; the end of each bond whose ket
leg is IN carries (-1)^(p_k p_K + p_k), so contracting fused legs equals
contracting the pairs, in either operand order.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

---

### Task 3: The graded double layer, with production's structure

**Files:**
- Modify: `src/tenax/algorithms/_graded_double_layer.py` (append `build_graded_double_layer`; widen the `_graded` import)
- Test: `tests/test_graded_double_layer.py`

**Interfaces:**
- Consumes:
  - `graded_fuse_pair` (Task 2), and `graded_bar` and `graded_contract` (Phase 1);
  - production's `_build_double_layer_tensor` and `_build_double_layer_open_tensor` (structure reference).
- Produces: `build_graded_double_layer(A, *, phys_bra=None) -> SymmetricTensor`, with labels `(u2, d2, l2, r2)`, plus `(phys, phys_bra)` when `phys_bra` is given. Its flows and charges are production's. `A` has labels `(u, d, l, r, phys)`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_graded_double_layer.py`, replace the two production-import lines with

```python
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_open_tensor,
    _build_double_layer_tensor,
    _fuse_pair_by_label,
)
from tenax.algorithms._graded_double_layer import (
    build_graded_double_layer,
    graded_fuse_pair,
    graded_split_pair,
)
```

and append:

```python
def _site(sym, ch, seed):
    """A site tensor in production's convention: ``(u, d, l, r, phys)``
    with flows ``(OUT, IN, OUT, IN, IN)``."""
    flows = (OUT, IN, OUT, IN)
    legs = [_idx(sym, ch, f, x) for f, x in zip(flows, "udlr")]
    legs.append(_idx(sym, [0, 1], IN, "phys"))
    return _rand(legs, seed)


# Dimension-2 bonds: a double layer squares every bond, and FermionicU1 at
# dimension 5 costs minutes of block-sparse compile for a structure check.
SITE_SYMS = [
    pytest.param(FermionParity(), [0, 1], id="FermionParity"),
    pytest.param(FermionicU1(), [0, 1], id="FermionicU1"),
]


@pytest.mark.parametrize("sym,ch", SITE_SYMS)
@pytest.mark.parametrize("open_phys", [False, True])
def test_the_graded_double_layer_has_productions_structure(sym, ch, open_phys):
    A = _site(sym, ch, 5)
    if open_phys:
        prod = _build_double_layer_open_tensor(A)
        graded = build_graded_double_layer(A, phys_bra="phys_bra")
    else:
        prod = _build_double_layer_tensor(A)
        graded = build_graded_double_layer(A)
    assert graded.labels() == prod.labels()
    for g, p in zip(graded.indices, prod.indices):
        assert g.flow == p.flow
        assert list(g.charges) == list(p.charges)


def test_regime_the_graded_double_layer_differs_from_productions():
    A = _site(FermionParity(), [0, 1], 5)
    assert _maxdiff(build_graded_double_layer(A), _build_double_layer_tensor(A)) > 1e-3


def test_the_graded_double_layer_traces_under_jit():
    A = _site(FermionParity(), [0, 1], 5)
    eager = build_graded_double_layer(A, phys_bra="phys_bra")
    jitted = jax.jit(lambda t: build_graded_double_layer(t, phys_bra="phys_bra"))(A)
    assert _maxdiff(jitted, eager) < 1e-12
```

- [ ] **Step 2: Run and see them fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_double_layer.py -q -p no:cacheprovider --no-cov`
Expected: collection error `ImportError: cannot import name 'build_graded_double_layer'`.

- [ ] **Step 3: Implement.** In `src/tenax/algorithms/_graded_double_layer.py`, replace the `_graded` import with

```python
from tenax.core._graded import (
    _is_graded,
    _scale_blocks,
    graded_bar,
    graded_contract,
    graded_reorder,
)
```

and append:

```python


def build_graded_double_layer(A: SymmetricTensor, *, phys_bra=None) -> SymmetricTensor:
    """Graded twin of ``_build_double_layer_tensor`` (``phys_bra=None``: the
    physical leg is contracted) and ``_build_double_layer_open_tensor``
    (``phys_bra="phys_bra"``: it stays open under that label).

    Same labels, order and flows as production -- ``(u2, d2, l2, r2)`` then
    ``phys, phys_bra`` when open -- built with ``graded_bar`` (rule 3),
    ``graded_contract`` and :func:`graded_fuse_pair`.  ``A`` has labels
    ``(u, d, l, r, phys)``.
    """
    rename = {x: x.upper() for x in "udlr"}
    if phys_bra is not None:
        rename["phys"] = phys_bra
    a = graded_contract(A, graded_bar(A).relabels(rename))
    for x in "udlr":
        a = graded_fuse_pair(a, x, x.upper(), f"{x}2")
    return a
```

- [ ] **Step 4: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_double_layer.py -q -p no:cacheprovider --no-cov --durations=3`
Expected: `18 passed` in about 30 s. No test should take more than about 10 s; if one does, a fixture has grown past bond dimension 2.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_graded_double_layer.py tests/test_graded_double_layer.py
git commit -m "feat(#1035): build_graded_double_layer, production's structure on the graded path

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

---

### Task 4: The graded double layer against Fock (the step 3 gate)

**Files:**
- Modify: `tests/_graded_cluster.py`:
  - rename `_bond_operator` to `bond_operator`, including its two callers;
  - add an import;
  - append `production_site`, `double_layer_value` and `double_layer_energy`.
- Modify: `tests/test_graded_contract_oracle.py` (imports; append the Phase 2 section)

**Interfaces:**
- Consumes:
  - `build_graded_double_layer` (Task 3) and `_graded_double_layer._pair_sign` (Task 2, monkeypatched);
  - `graded_contract` (Phase 1);
  - the oracle's `fock_psi`, `z_gauge`, `hop_energy`, `_create`, `_annihilate`, `_H2`.
- Produces: `double_layer_value(R, C, As, *, op=None) -> complex` and `double_layer_energy(R, C, As) -> (E, norm)`. Step 4 will reuse them as its finite-patch reference.

- [ ] **Step 1: The cluster helpers.** In `tests/_graded_cluster.py`:
  - rename `_bond_operator` to `bond_operator` in its definition and in `hop_operator` and `gate_operator`. Check with `grep -c "_bond_operator" tests/_graded_cluster.py`, which must print `0`;
  - add `from tenax.algorithms._graded_double_layer import build_graded_double_layer` above the `tenax.core._graded` import;
  - append:

```python
def production_site(A: np.ndarray) -> SymmetricTensor:
    """``A[u,d,l,r,p]`` in production's convention: labels
    ``(u, d, l, r, phys)``, flows ``FLOW`` then ``phys`` IN, boundary legs
    kept (dimension 1, parity 0) -- the input of the double-layer builders."""
    idx = [
        TensorIndex.from_charges(
            SYM, np.arange(n, dtype=np.int32) % 2, FLOW[x], label=x
        )
        for n, x in zip(A.shape[:4], "udlr")
    ]
    idx.append(_index(2, IN, "phys"))
    return SymmetricTensor.from_dense(jnp.asarray(A), tuple(idx))


def double_layer_value(R, C, As, *, op=None):
    """``<psi| O |psi>`` from graded double layers
    (:func:`build_graded_double_layer`), contracted site by site with
    ``graded_contract`` -- the per-site order a CTM builds.

    ``op=None`` is the norm.  ``op=((s, t), h2)`` puts the two-site operator
    ``h2[P_s, P_t, p_s, p_t]`` (the local ``(s, t)`` basis, as
    :func:`bond_operator`) on sites ``s, t``, whose double layers keep their
    physical legs open.
    """
    sites = sites_of(R, C)
    n_of = {s: n for n, s in enumerate(sites)}
    rename = {s: {} for s in sites}
    for b, (s, x, t, y) in enumerate(bonds_of(R, C)):
        rename[s][f"{x}2"] = rename[t][f"{y}2"] = f"b{b}"
    on = () if op is None else op[0]
    seq = []
    for s in sites:
        n = n_of[s]
        if s in on:
            a = build_graded_double_layer(production_site(As[s]), phys_bra=f"P{n}")
            m = {"phys": f"p{n}"}
        else:
            a = build_graded_double_layer(production_site(As[s]))
            m = {}
        m |= {f"{x}2": f"open_{n}_{x}" for x in "udlr"} | rename[s]
        seq.append(a.relabels(m))
    if op is not None:
        (s, t), h2 = op
        seq.append(bond_operator(h2, n_of[s], n_of[t]))
    out = seq[0]
    for u in seq[1:]:
        out = graded_contract(out, u)
    arr = np.asarray(out.todense()).reshape(-1)  # only dimension-1 legs remain
    assert arr.size == 1, out.labels()
    return complex(arr[0])


def double_layer_energy(R, C, As):
    norm = double_layer_value(R, C, As).real
    e = sum(
        double_layer_value(R, C, As, op=((s, t), _H2)).real
        for s, _, t, _ in bonds_of(R, C)
    )
    return e / norm, norm
```

- [ ] **Step 2: Write the oracle tests.** In `tests/test_graded_contract_oracle.py`:
  - add `double_layer_energy,` and `double_layer_value,` to the `from _graded_cluster import (...)` list, alphabetically;
  - add `import tenax.algorithms._graded_double_layer as _gdl` as the first line of the `tenax` import block (ruff's isort places it there);
  - append:

```python
# ------------------------------------------------------------------ #
# Phase 2: the graded double layer (design §5 step 3)                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("R,C", CLUSTERS)
def test_the_graded_double_layer_energy_matches_fock(R, C):
    """Production-shaped double layers (``build_graded_double_layer``: same
    labels, flows and fused legs as ``_build_double_layer_tensor``),
    contracted site by site, give the energy of the ket-level state that
    ``graded_contract`` defines -- the state gates and SVDs act on."""
    rng = np.random.default_rng(2)
    for _ in range(2):
        As = random_even_tensors(R, C, rng)
        psi = fock_psi(R, C, z_gauge(R, C, As))
        E, norm = double_layer_energy(R, C, As)
        E_hcb = hop_energy(R, C, plain_amplitudes(R, C, As), fermion=False)
        assert norm == pytest.approx(np.vdot(psi, psi).real, rel=1e-12)
        assert E == pytest.approx(hop_energy(R, C, psi, fermion=True), abs=1e-12)
        assert abs(E - E_hcb) > 1e-3  # regime: the bosonic answer is different


def test_the_graded_double_layer_energy_matches_fock_for_complex_tensors():
    rng = np.random.default_rng(4)
    re, im = random_even_tensors(2, 2, rng), random_even_tensors(2, 2, rng)
    As = {s: re[s] + 1j * im[s] for s in re}
    psi = fock_psi(2, 2, z_gauge(2, 2, As))
    E, norm = double_layer_energy(2, 2, As)
    assert norm == pytest.approx(np.vdot(psi, psi).real, rel=1e-12)
    assert E == pytest.approx(hop_energy(2, 2, psi, fermion=True), abs=1e-12)


def _fock_bond_element(psi, a, b, P_a, P_b, p_a, p_b):
    """``<psi| (c_a^+)^P_a (c_b^+)^P_b |0><0|_ab (c_b)^p_b (c_a)^p_a |psi>``:
    the operator ``|P_a P_b><p_a p_b|`` in the local ``(a, b)`` basis."""
    v = psi
    if p_a:
        v = _annihilate(v, a)
    if p_b:
        v = _annihilate(v, b)
    idx = np.arange(v.size)
    v = np.where(((idx >> a) & 1) | ((idx >> b) & 1), 0, v)  # |0><0| on a, b
    if P_b:
        v = _create(v, b)
    if P_a:
        v = _create(v, a)
    return np.vdot(psi, v)


def test_every_two_site_rdm_element_matches_fock():
    """The two-site reduced density matrix of every bond, element by
    element: all eight parity-even operators ``|P_s P_t><p_s p_t|``,
    including the pairing ones (the random tensors conserve parity, not
    number) and the bond whose sites are not neighbours in Jordan-Wigner
    order, (0,0)-(1,0)."""
    R, C = 2, 2
    As = random_even_tensors(R, C, np.random.default_rng(6))
    psi = fock_psi(R, C, z_gauge(R, C, As))
    n_of = {s: n for n, s in enumerate(sites_of(R, C))}
    checked = 0
    for s, _, t, _ in bonds_of(R, C):
        for k in itertools.product((0, 1), repeat=4):
            if sum(k) % 2:
                continue
            h2 = np.zeros((2, 2, 2, 2))
            h2[k] = 1.0
            got = double_layer_value(R, C, As, op=((s, t), h2))
            want = _fock_bond_element(psi, n_of[s], n_of[t], *k)
            assert got == pytest.approx(want, abs=1e-12), (s, t, k)
            checked += abs(want) > 1e-3
    assert checked >= 16  # regime: most elements are not trivially zero


def test_regime_the_fuse_needs_its_pair_sign(monkeypatch):
    """Without the pair sign, the fused double layer is a different state."""
    monkeypatch.setattr(_gdl, "_pair_sign", lambda t, k: t)
    As = random_even_tensors(2, 2, np.random.default_rng(2))
    psi = fock_psi(2, 2, z_gauge(2, 2, As))
    E, _ = double_layer_energy(2, 2, As)
    assert abs(E - hop_energy(2, 2, psi, fermion=True)) > 1e-3
```

- [ ] **Step 3: Run**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract_oracle.py -k "double_layer or rdm or pair_sign" -q -p no:cacheprovider --no-cov`
Expected: `5 passed` (the 2×3 energy case runs here because `-k` selects it; `-m core` would deselect it).

This task has no red step of its own: the code under test landed in Tasks 2–3. What makes these tests meaningful is Task 5's mutants, each of which must turn at least one of them red.

- [ ] **Step 4: Commit**

```bash
git add tests/_graded_cluster.py tests/test_graded_contract_oracle.py
git commit -m "test(#1035): the graded double layer against Fock (design §5 step 3)

Production-shaped graded double layers, contracted site by site, give
the norm, the hopping energy (2x2, 2x3, complex) and all eight
parity-even elements of every bond's two-site RDM of the ket-level state
Fock(z_gauge(As)) -- the state graded gates and SVDs act on -- to 1e-12.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014EB82CpbPJGcdPnAEU46Ap"
```

---

### Task 5: Mutants, full check, the record, and the PR

- [ ] **Step 1: Mutants.** Commit first, which Task 4 already did. For each row below:
  - assert the anchor occurs exactly once;
  - apply the mutation;
  - run `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py tests/test_graded_double_layer.py tests/test_graded_contract_oracle.py -m core -q -p no:cacheprovider --no-cov`;
  - record which tests fail;
  - restore with `git checkout -- <file>`.

Pre-validated results:

Every one of these must be caught. M4 is **equivalent for the fuse by design**: the pair sign depends only on a contracted pair's parity, so either end is correct (ruling 1). Only the fuse→split round trip catches it, because `graded_split_pair` still undoes the sign on the IN end. That makes it a test of fuse/split consistency, not of the physics.

| # | File | Anchor → mutation | Fails (core run of the three files) |
|---|---|---|---|
| M1 | `_graded_double_layer.py` | `lambda p: p[k] * p[k + 1] + p[k])` → `... + p[k + 1])` (twist on the bra parity) | fused-contraction property, 2×2 energy, complex energy, every RDM element (7 cases) |
| M2 | same | `lambda p: p[k] * p[k + 1] + p[k])` → `lambda p: p[k])` (no swap sign) | same four tests (7 cases) |
| M3 | same | `if _is_graded(t) and flow_k == IN:` → `if _is_graded(t):` (sign on both ends) | the above plus split round trip (9 cases) |
| M4 | same | `flow_k == IN` → `flow_k != IN` in the fuse (sign on the other end) | split round trip only (2 cases): an equivalent mutant for the fuse, see above |
| M5 | same | split's `if _is_graded(out) and info.parent_indices[0].flow == IN:` → `if False:` | split round trip (2 cases) |
| M6 | same | `graded_bar(A)` → `A.bar()` in `build_graded_double_layer` (rule 3 dropped) | 2×2 energy, complex energy, every RDM element (3 cases) |
| M7 | same | `fused_label, flow_b)` → `fused_label, flow_k)` (fused leg takes the ket flow) | structure, fused-leg flow, property, energies, RDM (12 cases) |
| M8 | same | the fuse's `graded_reorder(...)` → `t.permute_legs(...)` (sign-free move) | property, split, energies, RDM (9 cases) |
| M9 | `core/_graded.py` | `if _is_graded(a) != _is_graded(b):` → `if False:` | mixed-operand guard (2 cases) |
| M10 | same | `    if unknown:` → `    if False:` | unknown-label guard (1 case) |

- [ ] **Step 2: Full check**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_graded_contract.py tests/test_graded_double_layer.py tests/test_graded_contract_oracle.py tests/test_fermionic_fock_oracle.py -q -p no:cacheprovider --no-cov`
Expected: `65 passed, 2 xfailed` in about 6 min. The 2 xfails are #1038's production markers (ruling 3).

Run: `JAX_PLATFORMS=cpu uv run pytest -m core -q -p no:cacheprovider --no-cov -x` (the required gate). Expected: no failures.

- [ ] **Step 3: The record.** Append a §9 to the design doc on branch `design/1035-sign-convention` (PR #1036), in the same shape as §8. It covers:
  - what was certified (the Task 4 table);
  - the fuse rule and why the design's "pure reshape" was not enough (ruling 1);
  - the oracle-state choice (ruling 2);
  - rulings 3 and 4;
  - the mutant table;
  - the step 4 entry criteria: the 9 CTM reorders, the environment's own fused legs going through `graded_fuse_pair`, and `double_layer_value` as the finite-patch reference.

Commit with the trailers and push.

- [ ] **Step 4: The PR.** Push the branch and open a PR against `main`:
  - Title: `feat(#1035): graded double layer (Phase 2, design §5 step 3)`.
  - Body: what changed, the Task 4 results, the rulings, the mutant table, the exact test counts, and "no production code path changes".
  - End the body with the attribution lines.
  - Post `@codex review` only if Codex does not auto-review.
  - Do **not** arm auto-merge.
