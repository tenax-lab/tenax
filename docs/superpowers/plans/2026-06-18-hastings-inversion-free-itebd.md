# Inversion-free (Hastings) 1D iTEBD reference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone dense `numpy` reference for Hastings' inversion-free iTEBD update (arXiv:0903.3253) alongside the existing Vidal pseudo-inverse path, validated to reproduce the same Heisenberg ground-state energy.

**Architecture:** A new left-canonical state (`ITEBDStateLeft`), a symmetric inversion-free bond update (`_update_bond_hastings`) whose only "trick" is `B' = X†·Θ` (contracting the *unweighted* gated block, so `λ⁻¹` is never formed), a left-canonical energy helper, and a driver `itebd_groundstate_hastings` mirroring the existing `itebd_groundstate`. All live in `src/tenax/algorithms/itebd.py` and reuse the existing `_trotter_gate` / `heisenberg_2site_h`.

**Tech Stack:** Python, `numpy` (dense reference, no JAX), `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-18-hastings-inversion-free-itebd-design.md`

---

## File Structure

- **Modify** `src/tenax/algorithms/itebd.py` — add `_update_bond_hastings`, `_bond_energy_left`, `ITEBDStateLeft`, `itebd_groundstate_hastings`. (Currently 148 lines; ~90 added. Reuses existing `_trotter_gate`, `heisenberg_2site_h`.)
- **Modify** `tests/test_itebd.py` — add unit tests (update invariants, energy) and parity integration tests.
- **Modify** `src/tenax/algorithms/__init__.py` — register `itebd_groundstate_hastings`, `ITEBDStateLeft`.
- **Modify** `src/tenax/__init__.py` — register the same two symbols.
- **Modify** `README.md` — one-line note on the inversion-free variant.

**Index/shape conventions (match existing file):** site tensors are `(chiL, d, chiR)`; left-canonical means `reshape(chiL*d, chiR)` has orthonormal columns (`A†A = I`). `ITEBDStateLeft.A` is `(chiB, d, chiA)`, `B` is `(chiA, d, chiB)`; `lab` (dim `chiA`) is the Schmidt weight on the A→B bond, `lba` (dim `chiB`) on the B→A bond. The update of a bond is closed on the right by the *other* bond's weight.

---

## Task 1: Inversion-free bond update `_update_bond_hastings`

**Files:**
- Modify: `src/tenax/algorithms/itebd.py` (add after `_bond_energy`, near line 110)
- Test: `tests/test_itebd.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_itebd.py` (add `import numpy as np` at top if not already module-level; it is currently imported inside one test — add a top-level `import numpy as np`):

```python
def _random_left_canonical(chiL, d, chiR, seed):
    """Random left-canonical tensor (chiL, d, chiR): reshape(chiL*d, chiR) isometry."""
    import numpy as np
    rng = np.random.RandomState(seed)
    m = rng.standard_normal((chiL * d, chiR))
    q, _ = np.linalg.qr(m)  # (chiL*d, chiR) with orthonormal columns
    return q.reshape(chiL, d, chiR)


class TestHastingsUpdate:
    def test_identity_gate_is_inversion_free_and_exact(self):
        """With the identity gate and no truncation, the Hastings update must
        return a left-canonical A_new and reproduce the original weighted block
        A·B·diag(l_right) exactly (A_new·B_new·diag(l_right))."""
        import numpy as np
        from tenax.algorithms.itebd import _update_bond_hastings

        chi, d = 3, 2
        A = _random_left_canonical(chi, d, chi, seed=1)
        B = _random_left_canonical(chi, d, chi, seed=2)
        l_right = np.abs(np.random.RandomState(3).standard_normal(chi))
        l_right = l_right / np.linalg.norm(l_right)

        # identity gate: G[i,j,k,l] = delta_ik delta_jl, so einsum("aijc,ijkl->aklc") is identity
        ident = np.einsum("ik,jl->ijkl", np.eye(d), np.eye(d))

        A_new, S, B_new = _update_bond_hastings(A, B, l_right, ident, chi_max=chi * d)

        # A_new is left-canonical (isometry)
        k = A_new.shape[2]
        flat = A_new.reshape(-1, k)
        assert np.allclose(flat.conj().T @ flat, np.eye(k), atol=1e-10)

        # new bond weights normalized
        assert abs(np.linalg.norm(S) - 1.0) < 1e-10

        # reconstruction: A_new·B_new·diag(l_right) == A·B·diag(l_right)
        recon = np.einsum("aik,kjc,c->aijc", A_new, B_new, l_right)
        orig = np.einsum("aik,kjc,c->aijc", A, B, l_right)
        assert np.allclose(recon, orig, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_itebd.py::TestHastingsUpdate -v`
Expected: FAIL with `ImportError: cannot import name '_update_bond_hastings'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/itebd.py` (after `_bond_energy`, before `itebd_groundstate`):

```python
def _update_bond_hastings(A, B, l_right, gate, chi_max):
    """Hastings' inversion-free iTEBD bond update (arXiv:0903.3253).

    ``A``, ``B`` are left-canonical ``(chiL, d, chiR)`` tensors sharing the bond
    to be updated; ``l_right`` is the Schmidt weight on the bond to the *right*
    of the ``B`` tensor. Applies ``gate`` across the A-B bond and returns updated
    left-canonical ``(A_new, l_AB_new, B_new)``. No ``lambda^-1`` is ever formed:
    ``B_new = X^dagger . Theta`` contracts the *unweighted* gated block.
    """
    chiL, d, _ = A.shape
    _, _, chiR = B.shape
    C = np.einsum("aik,kjc->aijc", A, B)  # (chiL, d, d, chiR), both left-canonical
    theta = np.einsum("aijc,ijkl->aklc", C, gate)  # apply gate: ket (ij) -> bra (kl)
    M = np.einsum("aijc,c->aijc", theta, l_right)  # weight the right bond (no inverse)
    mat = M.reshape(chiL * d, d * chiR)
    X, S, _Yh = np.linalg.svd(mat, full_matrices=False)
    k = min(chi_max, int(np.sum(S > 1e-14)))
    k = max(k, 1)
    X, S = X[:, :k], S[:k]
    S = S / (np.linalg.norm(S) + 1e-300)
    A_new = X.reshape(chiL, d, k)  # exact isometry (left-canonical)
    # B_new = X^dagger . theta  (uses unweighted theta -> no l_right^-1)
    B_new = np.einsum("aiK,aijc->Kjc", A_new.conj(), theta)
    return A_new, S, B_new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_itebd.py::TestHastingsUpdate -v`
Expected: PASS (2 assertions in one test).

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/itebd.py tests/test_itebd.py
git commit -m "feat(itebd): inversion-free Hastings bond update (dense reference)"
```

---

## Task 2: Left-canonical energy helper `_bond_energy_left`

**Files:**
- Modify: `src/tenax/algorithms/itebd.py` (add after `_update_bond_hastings`)
- Test: `tests/test_itebd.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_itebd.py`:

```python
class TestHastingsEnergy:
    def test_product_state_energies(self):
        """In left-canonical form, <H> on a chi=1 product state matches the
        analytic Heisenberg diagonal: |up up> -> +0.25, |up down> -> -0.25."""
        import numpy as np
        from tenax.algorithms.itebd import _bond_energy_left, heisenberg_2site_h

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        up = np.array([1.0, 0.0])
        down = np.array([0.0, 1.0])
        l_right = np.array([1.0])

        def site(vec):
            return vec.reshape(1, 2, 1)  # (chiL=1, d=2, chiR=1), already normalized

        e_uu = _bond_energy_left(site(up), site(up), l_right, H)
        e_ud = _bond_energy_left(site(up), site(down), l_right, H)
        assert abs(e_uu - 0.25) < 1e-12, f"e_uu={e_uu}"
        assert abs(e_ud - (-0.25)) < 1e-12, f"e_ud={e_ud}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_itebd.py::TestHastingsEnergy -v`
Expected: FAIL with `ImportError: cannot import name '_bond_energy_left'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tenax/algorithms/itebd.py` (after `_update_bond_hastings`):

```python
def _bond_energy_left(A, B, l_right, H):
    """<H> across the A-B bond for left-canonical A, B closed on the right by
    ``l_right`` (left tensors are isometries => identity left env; ``l_right**2``
    is the right env). ``Theta = A . B . diag(l_right)``."""
    theta = np.einsum("aik,kjc,c->aijc", A, B, l_right)  # (chiL, d, d, chiR)
    norm = np.einsum("aijc,aijc->", theta, theta.conj()).real
    Ht = np.einsum("aijc,ijkl->aklc", theta, H)
    e = np.einsum("aklc,aklc->", Ht, theta.conj()).real
    return e / (norm + 1e-300)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_itebd.py::TestHastingsEnergy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/itebd.py tests/test_itebd.py
git commit -m "feat(itebd): left-canonical bond-energy helper for Hastings path"
```

---

## Task 3: Driver `itebd_groundstate_hastings` + `ITEBDStateLeft` (parity)

**Files:**
- Modify: `src/tenax/algorithms/itebd.py` (add `ITEBDStateLeft` near `ITEBDState` ~line 63; add driver at end of file)
- Test: `tests/test_itebd.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_itebd.py` (top of file already imports `itebd_groundstate`, `heisenberg_2site_h`; extend the import to add `itebd_groundstate_hastings`):

```python
class TestHastingsGroundState:
    def test_matches_bethe_ansatz(self):
        """Hastings iTEBD reaches e_0 = 1/4 - ln 2 for the Heisenberg chain."""
        import math
        from tenax.algorithms.itebd import heisenberg_2site_h, itebd_groundstate_hastings

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        e, _ = itebd_groundstate_hastings(H, chi_max=16, steps_per_dt=600)
        e_exact = 0.25 - math.log(2)
        assert abs(e - e_exact) < 5e-3, f"e={e}, exact={e_exact}"

    def test_matches_pseudoinverse(self):
        """The inversion-free path reproduces the validated Vidal path."""
        from tenax.algorithms.itebd import (
            heisenberg_2site_h,
            itebd_groundstate,
            itebd_groundstate_hastings,
        )

        H = heisenberg_2site_h(Jz=1.0, Jxy=1.0)
        kw = dict(chi_max=16, steps_per_dt=600, seed=0)
        e_vidal, _ = itebd_groundstate(H, **kw)
        e_hast, _ = itebd_groundstate_hastings(H, **kw)
        assert abs(e_vidal - e_hast) < 1e-3, f"vidal={e_vidal}, hastings={e_hast}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_itebd.py::TestHastingsGroundState -v`
Expected: FAIL with `ImportError: cannot import name 'itebd_groundstate_hastings'`.

- [ ] **Step 3: Write minimal implementation**

Add the dataclass to `src/tenax/algorithms/itebd.py` immediately after the existing `ITEBDState` dataclass (after line ~70):

```python
@dataclass
class ITEBDStateLeft:
    """2-site unit cell in LEFT-canonical form: ... A B A B ...

    ``A``, ``B`` are left-canonical site tensors; ``lab``/``lba`` are the Schmidt
    weights on the A->B and B->A bonds respectively.
    """

    A: np.ndarray  # (chiB, d, chiA) left-canonical
    B: np.ndarray  # (chiA, d, chiB) left-canonical
    lab: np.ndarray  # (chiA,) Schmidt weights on the A->B bond
    lba: np.ndarray  # (chiB,) Schmidt weights on the B->A bond
```

Add the driver at the end of `src/tenax/algorithms/itebd.py`:

```python
def itebd_groundstate_hastings(
    H2: np.ndarray,
    chi_max: int = 16,
    dts=(0.1, 0.03, 0.01, 0.003, 0.001),
    steps_per_dt: int = 2000,
    seed: int = 0,
) -> tuple[float, ITEBDStateLeft]:
    """Inversion-free (Hastings) imaginary-time iTEBD ground state for a 2-site
    nearest-neighbour ``H2``. No ``lambda^-1`` is formed (cf. arXiv:0903.3253).

    Returns ``(energy_per_site, state)``.
    """
    d = H2.shape[0]
    rng = np.random.RandomState(seed)
    chi = 1
    # chi=1 left-canonical tensors: each is a normalized d-vector
    A = rng.standard_normal((chi, d, chi))
    A = A / np.linalg.norm(A)
    B = rng.standard_normal((chi, d, chi))
    B = B / np.linalg.norm(B)
    lab = np.ones(chi)  # weight on A->B bond
    lba = np.ones(chi)  # weight on B->A bond

    e_prev = np.inf
    for dt in dts:
        gate = _trotter_gate(H2, dt)
        for _ in range(steps_per_dt):
            # bond A-B (closed on the right by lba)
            A, lab, B = _update_bond_hastings(A, B, lba, gate, chi_max)
            # bond B-A (translate one site; closed on the right by lab)
            B, lba, A = _update_bond_hastings(B, A, lab, gate, chi_max)
        eAB = _bond_energy_left(A, B, lba, H2)
        eBA = _bond_energy_left(B, A, lab, H2)
        e = 0.5 * (eAB + eBA)
        if abs(e - e_prev) < 1e-10:
            e_prev = e
            break
        e_prev = e
    return float(e_prev), ITEBDStateLeft(A, B, lab, lba)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_itebd.py::TestHastingsGroundState -v`
Expected: PASS (both tests). This takes ~tens of seconds (dense, small χ).

- [ ] **Step 5: Run the full iTEBD test module to confirm no regressions**

Run: `uv run pytest tests/test_itebd.py -v`
Expected: all existing + new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/itebd.py tests/test_itebd.py
git commit -m "feat(itebd): inversion-free Hastings ground-state driver + parity tests"
```

---

## Task 4: Public API exports + README

**Files:**
- Modify: `src/tenax/algorithms/__init__.py` (lazy-import map ~line 38, `__all__` ~line 175)
- Modify: `src/tenax/__init__.py` (lazy-import map ~line 220, `__all__` ~line 426)
- Modify: `README.md` (line 18)
- Test: `tests/test_itebd.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_itebd.py`:

```python
class TestHastingsExports:
    def test_public_imports_resolve(self):
        """itebd_groundstate_hastings and ITEBDStateLeft are exported from both
        tenax and tenax.algorithms (per CLAUDE.md new-public-API rule)."""
        import tenax
        import tenax.algorithms as ta

        assert tenax.itebd_groundstate_hastings is ta.itebd_groundstate_hastings
        assert tenax.ITEBDStateLeft is ta.ITEBDStateLeft
        assert "itebd_groundstate_hastings" in tenax.__all__
        assert "ITEBDStateLeft" in tenax.__all__
        assert "itebd_groundstate_hastings" in ta.__all__
        assert "ITEBDStateLeft" in ta.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_itebd.py::TestHastingsExports -v`
Expected: FAIL with `AttributeError: module 'tenax' has no attribute 'itebd_groundstate_hastings'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tenax/algorithms/__init__.py`, after the line:
```python
    "heisenberg_2site_h": ("tenax.algorithms.itebd", "heisenberg_2site_h"),
```
add:
```python
    "itebd_groundstate_hastings": ("tenax.algorithms.itebd", "itebd_groundstate_hastings"),
    "ITEBDStateLeft": ("tenax.algorithms.itebd", "ITEBDStateLeft"),
```
and in the `__all__` list, after the `"heisenberg_2site_h",` entry, add:
```python
    "itebd_groundstate_hastings",
    "ITEBDStateLeft",
```

In `src/tenax/__init__.py`, after the line:
```python
    "heisenberg_2site_h": ("tenax.algorithms.itebd", "heisenberg_2site_h"),
```
add:
```python
    "itebd_groundstate_hastings": ("tenax.algorithms.itebd", "itebd_groundstate_hastings"),
    "ITEBDStateLeft": ("tenax.algorithms.itebd", "ITEBDStateLeft"),
```
and in its `__all__` list, after the `"heisenberg_2site_h",` entry, add:
```python
    "itebd_groundstate_hastings",
    "ITEBDStateLeft",
```

In `README.md` line 18, change the iTEBD clause:
```
iTEBD (numerically stable infinite TEBD),
```
to:
```
iTEBD (numerically stable infinite TEBD, incl. inversion-free Hastings update),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_itebd.py::TestHastingsExports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/__init__.py src/tenax/__init__.py README.md tests/test_itebd.py
git commit -m "docs(itebd): export inversion-free Hastings API + README note"
```

---

## Task 5: Final verification

- [ ] **Step 1: Run the full iTEBD module**

Run: `uv run pytest tests/test_itebd.py -v`
Expected: all tests PASS (existing 3 + new: TestHastingsUpdate, TestHastingsEnergy, TestHastingsGroundState ×2, TestHastingsExports).

- [ ] **Step 2: Confirm no `lambda^-1` leaked into the Hastings path**

Run: `grep -n "_safe_inv\|inv_lo\|1.0 /" src/tenax/algorithms/itebd.py`
Expected: matches appear only inside `_safe_inv` and the existing `_update_bond` (the Vidal path), NOT inside `_update_bond_hastings`, `_bond_energy_left`, or `itebd_groundstate_hastings`. Visually confirm.

- [ ] **Step 3: Run core marker subset to confirm nothing else broke**

Run: `uv run pytest -m core -q`
Expected: PASS (or unchanged from baseline).

---

## Self-Review notes

- **Spec coverage:** Component 1 (state) → Task 3; Component 2 (update) → Task 1; Component 3 (energy) → Task 2; Component 4 (driver) → Task 3; Component 5 (placement/exports) → Task 4; Component 6 (tests: Bethe-ansatz + pseudo-inverse parity) → Task 3. All covered.
- **Type/name consistency:** `_update_bond_hastings(A, B, l_right, gate, chi_max)` returns `(A_new, S, B_new)` used identically in Task 3; `_bond_energy_left(A, B, l_right, H)` signature matches its calls; `ITEBDStateLeft(A, B, lab, lba)` field order matches construction in the driver.
- **No placeholders:** every code/edit step shows the exact content.
