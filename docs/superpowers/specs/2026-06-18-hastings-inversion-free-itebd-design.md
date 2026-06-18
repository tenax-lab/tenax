# Design: Inversion-free (Hastings) 1D iTEBD reference

**Date:** 2026-06-18
**Issue/PR context:** Follow-up to PR #583 review by @ianmccul; attribution fix in #616.
**Status:** Approved (brainstorming), pending implementation plan.

## Motivation

PR #583 added a numerically stable 1D iTEBD but mislabeled its stabilization as
"Hastings' trick." As Ian McCulloch pointed out, the implemented algorithm is
the standard Vidal Γ-λ iTEBD with a *regularized* `λ⁻¹` (thresholded
pseudo-inverse, `_safe_inv`). Hastings' stable iTEBD (arXiv:0903.3253) is the
opposite idea: it **avoids forming `λ⁻¹` entirely** by working in
left-/right-canonical form and re-orthogonalizing via SVD, using no
pseudo-inverse at all. #616 corrected the comments. This spec adds an actual
inversion-free Hastings update as a **standalone dense reference** so the two
algorithms can be cross-validated.

## Scope (decided during brainstorming)

- **Role:** a new standalone reference function alongside the existing
  `itebd_groundstate`, not a replacement and not a flag/mode. Keeps the
  validated Vidal path intact and enables a parity cross-check.
- **Implementation style:** dense `numpy` reference, matching the existing
  `itebd.py` (small χ, `numpy` linear algebra). A block-sparse / JAX version is
  explicitly out of scope.
- **Validation:** parity with the existing path only — reach the same
  Heisenberg ground-state energy and agree with the pseudo-inverse path. No
  constructed stability-edge case, no structural no-inverse assertion.
- **1D only.** Hastings' trick has no 2D canonical-form analogue for the PEPS
  simple-update path; this work does not touch fermionic/PEPS code.

## Approach (chosen: A — faithful Hastings, left-canonical state)

Rejected alternatives:
- **B — QR re-orthogonalization:** inversion-free but a different
  (VUMPS-flavored) algorithm, not "Hastings' trick."
- **C — keep Vidal state, Hastings internally:** converting the Hastings result
  back to Γ-λ for storage reintroduces a `λ⁻¹`, defeating the purpose.

## Component 1 — State representation

New dataclass alongside the existing Vidal `ITEBDState`:

```python
@dataclass
class ITEBDStateLeft:
    """2-site unit cell in LEFT-canonical form: ...A B A B..."""
    A: np.ndarray    # (chiB, d, chiA) left-canonical
    B: np.ndarray    # (chiA, d, chiB) left-canonical
    lab: np.ndarray  # (chiA,) Schmidt weights on the A->B bond (right of A)
    lba: np.ndarray  # (chiB,) Schmidt weights on the B->A bond (right of B)
```

"Left-canonical" means each tensor reshaped to `(left·d, right)` has orthonormal
columns (`A†A = I` over the `(left, d)` indices). The right environment of any
cut is therefore `diag(λ²)`; no right-canonical tensors are stored. This is the
gauge Hastings' trick operates in.

## Component 2 — The inversion-free update (core)

One symmetric helper, reused for both bonds of the 2-site cell:

```python
def _update_bond_hastings(A, B, l_right, gate, chi_max):
    # 1. C  = A·B                      shared bond -> (chiL, d, d, chiR), both left-canonical
    # 2. Θ  = gate·C                   apply Trotter gate to the two physical legs
    # 3. M  = Θ·diag(l_right)          weight the RIGHT bond (no inverse)
    # 4. X, S, Yh = svd(M, full_matrices=False)
    #    k = min(chi_max, rank(S > 1e-14));  k = max(k, 1);  S = S/‖S‖
    #    X, S = X[:, :k], S[:k]
    # 5. A_new = X.reshape(chiL, d, k)                 exact isometry (left-canonical)
    #    B_new = einsum('aiK, aijc -> Kjc', conj(X3), Θ)   ==  X†·Θ
    #    return A_new, S, B_new
```

Where `X3 = X.reshape(chiL, d, k)`.

**Why it is inversion-free.** Algebraically `X†·Θ == S·Yh·diag(l_right)⁻¹` (the
exact left-canonical second tensor), but because step 5 contracts the
**unweighted** `Θ` (= `gate·A·B`, built directly from the left-canonical
tensors) rather than the weighted `M`, `l_right⁻¹` is never formed. No
`_safe_inv`, no `1/λ`. Truncation (dropping all but `k` columns of `X`) is the
sole source of `B_new` being a slightly imperfect isometry — the controlled cost
Hastings describes, which self-corrects under imaginary-time evolution.

Driver invokes it symmetrically (translation by one site):

```python
A, lab, B = _update_bond_hastings(A, B, lba, gate, chi_max)   # bond A–B, closed by lba
B, lba, A = _update_bond_hastings(B, A, lab, gate, chi_max)   # bond B–A, closed by lab
```

## Component 3 — Energy

```python
def _bond_energy_left(A, B, l_right, H):
    # Θ = A·B·diag(l_right)  -> (chiL, d, d, chiR), the 2-site reduced wavefunction
    # (left tensors are isometries => identity left env; l_right² is the right env)
    # <H> = Θ†HΘ / Θ†Θ
```

Same contraction structure as the existing `_bond_energy`; the only difference is
the source gauge (left-canonical block weighted on the right vs. Vidal `θ`).

## Component 4 — Driver

```python
def itebd_groundstate_hastings(
    H2, chi_max=16, dts=(0.1, 0.03, 0.01, 0.003, 0.001),
    steps_per_dt=2000, seed=0,
) -> tuple[float, ITEBDStateLeft]:
    ...
```

Identical structure and signature to `itebd_groundstate`:
- χ=1 random init, normalized to left-canonical (trivial at χ=1).
- Loop over `dts`; per step apply both bond updates above.
- Per-`dt` energy `e = 0.5·(e_AB + e_BA)`; early-stop on `abs(e - e_prev) < 1e-10`.
- Returns `(energy_per_site, ITEBDStateLeft)`.
- Reuses existing `_trotter_gate` and `heisenberg_2site_h`.

## Component 5 — Placement & exports

- All new code lives in the **existing `src/tenax/algorithms/itebd.py`** (~80
  added lines to a 148-line file). Co-locating both variants makes the
  cross-check obvious and reuses the gate/Trotter helpers.
- Per CLAUDE.md (new public API): add `itebd_groundstate_hastings` and
  `ITEBDStateLeft` to the lazy-import maps and `__all__` in both
  `src/tenax/__init__.py` and `src/tenax/algorithms/__init__.py`; add a one-line
  note to the README iTEBD entry.

## Component 6 — Tests (parity only)

Add to `tests/test_itebd.py`:

1. **`test_hastings_matches_bethe_ansatz`** — Heisenberg
   `e₀ = ¼ − ln2 ≈ −0.4431`, tolerance `< 5e-3` (mirrors the existing
   `test_energy_matches_bethe_ansatz`).
2. **`test_hastings_matches_pseudoinverse`** — run both `itebd_groundstate` and
   `itebd_groundstate_hastings` at the same χ/dt schedule; assert the two
   energies agree to `< 1e-3`. Load-bearing test: proves the inversion-free path
   reproduces the validated Vidal path.

Tests are dense/small-χ and complete within the existing iTEBD test budget
(file is `algorithm`-marked by `conftest.py`).

## Out of scope / follow-ups

- Block-sparse / JAX implementation.
- De-duplication of the `_safe_inv` helper across `itebd.py`,
  `_tensor_utils.safe_inv_lambda`, and `pess._safe_inv` (separate concern noted
  on #583).
- Any 2D / PEPS canonical-form work.
