# Spin-1 XXZ iPESS on Kagome with AD — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land a Liao-2019-style differentiable iPESS pipeline for spin-1 XXZ on the kagome lattice, optimizing the PESS parameterization (T_u, T_d, R_a, R_b, R_c) directly via reverse-mode AD through CTM.

**Architecture:** PESS tensors stored as dense complex128. Triangle simple update (HOSVD-based) provides the warm start. AD path coarse-grains kagome → honeycomb supersites (Convention A from the design doc) and feeds the supersites into Tenax's existing implicit-diff CTM AD pipeline (sigma gauge, GMRES backward, SVD projectors). L-BFGS optimizer.

**Tech Stack:** JAX (jit, grad), Tenax (ipeps_optimize, honeycomb CTM AD, SVD projectors, sigma gauge, spin_one_ops).

**Reference design:** `docs/plans/2026-04-25-spin1-xxz-kagome-ipess-design.md`

**Reference example (existing, partial):** `examples/kagome_xxz_spin1_pess.py` — contains correct triangle Hamiltonian, init, HOSVD truncation, and triangle SU; lift these into the library module. Replace its `pess_to_ipeps` (Convention C, dummy-bond) with new honeycomb supersites coarse-graining (Convention A).

**Branching:** All work on a feature branch `feat/spin1-xxz-pess-ad`. Each task is a separate commit. Open a single PR at the end including the design doc.

---

## Task 0: Set up branch

**Step 1:** From `main`, create branch:
```bash
git checkout -b feat/spin1-xxz-pess-ad
```

**Step 2:** Stage the design doc only (created during brainstorming, currently untracked):
```bash
git add docs/plans/2026-04-25-spin1-xxz-kagome-ipess-design.md docs/plans/2026-04-25-spin1-xxz-kagome-ipess-plan.md
git commit -m "docs: design + plan for spin-1 XXZ iPESS AD on kagome"
```

---

## Task 1: Module skeleton + IPESSState dataclass

**Files:**
- Create: `src/tenax/algorithms/pess.py`
- Create: `tests/test_pess.py`

**Step 1: Write the failing test**

```python
# tests/test_pess.py
import jax
import jax.numpy as jnp
from tenax.algorithms.pess import IPESSState

def test_ipess_state_shapes():
    D, d = 4, 3
    key = jax.random.PRNGKey(0)
    state = IPESSState.random(D=D, d=d, key=key)
    assert state.R_a.shape == (D, D, d)
    assert state.R_b.shape == (D, D, d)
    assert state.R_c.shape == (D, D, d)
    assert state.T_u.shape == (D, D, D)
    assert state.T_d.shape == (D, D, D)
    assert all(lam.shape == (D,) for lam in state.lambdas)
    assert state.R_a.dtype == jnp.complex128
    assert state.T_u.dtype == jnp.complex128
```

**Step 2: Verify it fails**

```bash
uv run pytest tests/test_pess.py -v
```
Expected: ImportError or ModuleNotFoundError on `tenax.algorithms.pess`.

**Step 3: Implement IPESSState**

In `src/tenax/algorithms/pess.py`:
```python
"""iPESS (infinite Projected Entangled Simplex State) on the kagome lattice."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

D_PHYS_DEFAULT = 3  # spin-1


@dataclass(frozen=True)
class IPESSState:
    """Kagome iPESS parameters.

    R_a, R_b, R_c: rank-3 site tensors, shape (D, D, d). Index order is
        (leg-to-T_u, leg-to-T_d, physical).
    T_u, T_d: rank-3 simplex tensors, shape (D, D, D). Index order is
        (leg-to-R_a, leg-to-R_b, leg-to-R_c).
    lambdas: 6 bond singular-value vectors of length D, ordered
        (a-up, b-up, c-up, a-down, b-down, c-down).
    """
    R_a: jax.Array
    R_b: jax.Array
    R_c: jax.Array
    T_u: jax.Array
    T_d: jax.Array
    lambdas: Tuple[jax.Array, ...]

    @classmethod
    def random(cls, D: int, d: int = D_PHYS_DEFAULT, key: jax.Array | None = None,
               scale: float = 0.1) -> "IPESSState":
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        def cmplx(k, shape):
            re = jax.random.normal(k, shape) * scale
            im = jax.random.normal(jax.random.fold_in(k, 1), shape) * scale
            return (re + 1j * im).astype(jnp.complex128)
        return cls(
            R_a=cmplx(keys[0], (D, D, d)),
            R_b=cmplx(keys[1], (D, D, d)),
            R_c=cmplx(keys[2], (D, D, d)),
            T_u=cmplx(keys[3], (D, D, D)),
            T_d=cmplx(keys[4], (D, D, D)),
            lambdas=tuple(jnp.ones(D) for _ in range(6)),
        )
```

**Step 4: Verify tests pass**

```bash
uv run pytest tests/test_pess.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/pess.py tests/test_pess.py
git commit -m "feat(pess): IPESSState dataclass with random init"
```

---

## Task 2: Triangle XXZ Hamiltonian builder

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:** `kagome_triangle_xxz_hamiltonian(delta, d=3)` returns a hermitian (d^3, d^3) numpy array equal to the sum of the three pair-XXZ Hamiltonians around a triangle. Reuse `tenax.spin_one_ops` for d=3 and `tenax.spin_half_ops` for d=2 (verify the latter exists; if not, build inline).

**Step 1: Failing test**

```python
import numpy as np
from tenax.algorithms.pess import kagome_triangle_xxz_hamiltonian

def test_triangle_hamiltonian_hermitian_spin1():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    assert H.shape == (27, 27)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

def test_triangle_hamiltonian_hermitian_spin_half():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    assert H.shape == (8, 8)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

def test_triangle_hamiltonian_xy_isotropic():
    H1 = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    H0 = kagome_triangle_xxz_hamiltonian(delta=0.0, d=3)
    diff = H1 - H0
    # Difference should be only the Sz Sz couplings
    assert np.linalg.norm(diff) > 0
```

**Step 2:** Run, verify failure (function not defined).

**Step 3:** Lift `kagome_triangle_hamiltonian_spin1` from `examples/kagome_xxz_spin1_pess.py:40-68` into `pess.py`, generalized over `d` via a `_site_ops(d)` helper that dispatches to `spin_one_ops()` / `spin_half_ops()`.

**Step 4:** Run tests; expect PASS.

**Step 5:** Commit:
```bash
git commit -am "feat(pess): kagome triangle XXZ Hamiltonian builder"
```

---

## Task 3: Trotter gate

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:** `make_triangle_gate(H, dt, d)` returns the (d,d,d,d,d,d) reshape of `expm(-dt H)` cast to complex128.

**Step 1: Test**

```python
def test_trotter_gate_unitarity_real_time():
    from tenax.algorithms.pess import make_triangle_gate, kagome_triangle_xxz_hamiltonian
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    gate = make_triangle_gate(H, dt=1j * 0.05, d=3)  # real time
    G = np.asarray(gate).reshape(27, 27)
    np.testing.assert_allclose(G @ G.conj().T, np.eye(27), atol=1e-10)

def test_trotter_gate_imag_time_decreases_norm_on_excited_state():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    gate = make_triangle_gate(H, dt=0.1, d=3)
    G = np.asarray(gate).reshape(27, 27)
    eigvals = np.linalg.eigvalsh(H)
    # Largest singular value of e^{-dt H} = e^{-dt * lambda_min}
    assert np.max(np.linalg.svd(G, compute_uv=False)) == \
           pytest.approx(np.exp(-0.1 * eigvals[0]), rel=1e-8)
```

**Step 3:** Lift `make_trotter_gate_3site` from example. Cast result to complex128.

**Step 5:** Commit: `feat(pess): triangle Trotter gate`.

---

## Task 4: HOSVD truncation primitive

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:** Lift `hosvd_truncate` from `examples/kagome_xxz_spin1_pess.py:126-185` verbatim, but ensure complex-safe (use `.conj().T` not `.T` where needed in the projection step) and dimension-agnostic over `d`.

**Step 1: Test**

```python
def test_hosvd_truncate_idempotent_no_truncation():
    """If D_max >= input dim, theta should round-trip."""
    from tenax.algorithms.pess import hosvd_truncate
    D, d = 3, 3
    key = jax.random.PRNGKey(7)
    theta = (jax.random.normal(key, (D, D, D, d, d, d)) +
             1j * jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, d, d, d))).astype(jnp.complex128)
    S_a, S_b, S_c, core, lams = hosvd_truncate(theta, D_max=D * d, d=d)
    # Reconstruct theta and compare; reconstruction should match within numerical tol.
    # (Implementation detail: theta = einsum(core, S_a, S_b, S_c))
    ...
```

**Step 3:** Implement.

**Step 5:** Commit: `feat(pess): HOSVD truncation`.

---

## Task 5: Triangle simple-update step

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:** Lift `pess_simple_update_triangle` from example, made complex-safe and operating on `IPESSState`. Public API:

```python
def pess_simple_update_triangle(state: IPESSState, gate: jax.Array,
                                triangle: str, D_max: int) -> IPESSState
```
where `triangle` is `"up"` or `"down"`.

**Step 1: Test**

```python
def test_su_step_identity_gate_preserves_norm():
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    gate = jnp.eye(27, dtype=jnp.complex128).reshape(3, 3, 3, 3, 3, 3)
    new_state = pess_simple_update_triangle(state, gate, triangle="up", D_max=2)
    # Within numerical tolerance, the contracted state should be the same up to
    # gauge — easiest check: energies of identical observables match.
    ...
```

**Step 5:** Commit: `feat(pess): triangle simple-update step`.

---

## Task 6: Full SU loop with dt schedule

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:**

```python
def pess_simple_update(state: IPESSState, hamiltonian: np.ndarray,
                       dt_schedule: list[tuple[float, int]],
                       D_max: int) -> IPESSState:
    """Run alternating up/down triangle SU.

    dt_schedule: list of (dt, num_steps). E.g. [(0.1, 200), (0.01, 200), (0.001, 100)].
    """
```

**Step 1: Test** — at small D=2 and Δ=1, run 100 steps with dt=0.05 and verify ⟨H⟩ decreases. Use a quick energy estimator (contract supersites with environment-free local triangle) for the test only — a full CTM is overkill here.

```python
def test_su_decreases_energy_d2():
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    state0 = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(1))
    state1 = pess_simple_update(state0, H, dt_schedule=[(0.05, 100)], D_max=2)
    e0 = _local_triangle_energy(state0, H)
    e1 = _local_triangle_energy(state1, H)
    assert e1 < e0
```

**Step 5:** Commit: `feat(pess): full simple-update loop`.

---

## Task 7: Honeycomb supersites coarse-graining

**Files:**
- Modify: `src/tenax/algorithms/pess.py`
- Modify: `tests/test_pess.py`

**Behavior:**

```python
def pess_to_honeycomb_supersites(state: IPESSState) -> tuple[jax.Array, jax.Array]:
    """Coarse-grain kagome iPESS → 2-sublattice honeycomb iPEPS supersites.

    Returns (A_u, A_d), each of shape (d, d, d, D, D, D), where physical legs
    are (R_a, R_b, R_c) and virtual legs go to the three neighboring opposite-
    sublattice supersites.

    A_u[p_a, p_b, p_c, l_a, l_b, l_c] =
        sum_{i,j,k} R_a[i, l_a, p_a] * R_b[j, l_b, p_b] * R_c[k, l_c, p_c] * T_u[i, j, k]
    A_d analogous with T_d, but with the (i,j,k) legs being the *down*-facing
    legs of the R-tensors.
    """
```

The site tensors R_x have leg index 0 → T_u, leg index 1 → T_d (per IPESSState docstring), so:
- A_u contracts R_x[:, l_x, p_x] (axis 0) with T_u.
- A_d contracts R_x[l_x, :, p_x] (axis 1) with T_d.

**Step 1: Test**

```python
def test_supersite_shapes():
    state = IPESSState.random(D=4, d=3, key=jax.random.PRNGKey(0))
    A_u, A_d = pess_to_honeycomb_supersites(state)
    assert A_u.shape == (3, 3, 3, 4, 4, 4)
    assert A_d.shape == (3, 3, 3, 4, 4, 4)

def test_supersite_grad_flows():
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    def loss(s):
        A_u, A_d = pess_to_honeycomb_supersites(s)
        return jnp.real(jnp.vdot(A_u.ravel(), A_u.ravel())) + \
               jnp.real(jnp.vdot(A_d.ravel(), A_d.ravel()))
    g = jax.grad(loss)(state)
    # All five primitive tensors should receive a finite gradient.
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u, g.T_d):
        assert jnp.all(jnp.isfinite(arr))
        assert jnp.linalg.norm(arr) > 0
```

(`jax.grad` over a frozen dataclass needs `IPESSState` to be a JAX pytree — register it via `jax.tree_util.register_pytree_node` in the module. Add this in Task 1 if not already; otherwise add it here and adjust Task 1's tests accordingly.)

**Step 5:** Commit: `feat(pess): honeycomb supersite coarse-graining`.

---

## Task 8: AD loss closure + integration with existing CTM

**Files:**
- Create: `src/tenax/algorithms/pess_optimize.py`
- Create: `tests/test_pess_ad.py`

**Behavior:** Build the loss closure that:
1. Takes an `IPESSState` (or its flat parameter pytree).
2. Calls `pess_to_honeycomb_supersites` to produce `(A_u, A_d)`.
3. Hands `(A_u, A_d)` to the existing honeycomb 2-sublattice iPEPS CTM AD path (`tenax.algorithms.ipeps_ctm.ctm` configured for honeycomb with two sublattices, or the `_split_ctm_tensor.py` path).
4. Computes triangle energy via `compute_energy_ctm` extended to evaluate a 3-site triangle expectation value (the existing pair-energy measurement may not cover this; if not, add a `compute_triangle_energy_ctm` helper that uses the converged environment to compute ⟨H_triangle⟩).
5. Returns the real scalar energy.

**Investigation step (do this first, before TDD):** Inspect `src/tenax/algorithms/ipeps_ctm.py`, `ipeps_rdm.py`, and `_split_ctm_tensor*.py` to determine:
   - Which entry point handles 2-sublattice honeycomb supersites with rank-3 physical legs (d, d, d). The existing path may assume one physical leg per supersite — if so, decide whether to (i) pre-fuse the three physical legs into d³ during the CTM (still cleaner than Convention C since each supersite has a clean rank-3+rank-3 split) or (ii) extend the CTM machinery to support multi-physical-leg supersites.
   - If pre-fusion is the answer, this is a notable design refinement: the AD parameterization stays (T_u, T_d, R_a, R_b, R_c), only the supersites carry a fused d³ leg into CTM. This still differs from Convention C because the *virtual* manifold is honeycomb with 3 D-bonds, not square with a dummy bond.

Document the answer in a short comment block at the top of `pess_optimize.py`. If the investigation surfaces unexpected gaps, pause and update this plan before continuing.

**Step 1: Test**

```python
# tests/test_pess_ad.py
def test_pess_loss_is_differentiable():
    state = IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0))
    H = kagome_triangle_xxz_hamiltonian(delta=1.0, d=3)
    config = make_test_config(chi=8)
    loss_fn = build_pess_loss(H, config)
    e0 = loss_fn(state)
    g = jax.grad(loss_fn)(state)
    assert jnp.isfinite(e0)
    for arr in (g.R_a, g.R_b, g.R_c, g.T_u, g.T_d):
        assert jnp.all(jnp.isfinite(arr))
```

**Step 5:** Commit: `feat(pess): AD loss closure with honeycomb CTM`.

---

## Task 9: optimize_pess_ad entry point

**Files:**
- Modify: `src/tenax/algorithms/pess_optimize.py`
- Modify: `tests/test_pess_ad.py`

**Behavior:**

```python
def optimize_pess_ad(initial_state: IPESSState,
                     hamiltonian: np.ndarray,
                     config: iPEPSConfig,
                     max_iter: int = 100) -> tuple[IPESSState, float]:
    """L-BFGS optimization of PESS tensors to minimize triangle energy."""
```

Wrap the existing L-BFGS plumbing from `ipeps_optimize.py`. The optimization variable is the flat pytree of (R_a, R_b, R_c, T_u, T_d) — exclude `lambdas` (those are SU bookkeeping, not optimized at AD time).

**Step 1: Test**

```python
def test_ad_improves_over_su_small():
    """At D=2, χ=8, AD should match or beat the SU energy."""
    state_su = pess_simple_update(IPESSState.random(D=2, d=3, key=jax.random.PRNGKey(0)),
                                  H, dt_schedule=[(0.05, 200), (0.005, 200)], D_max=2)
    e_su = compute_pess_energy_via_ctm(state_su, H, chi=8)
    state_ad, e_ad = optimize_pess_ad(state_su, H, config=test_config(chi=8), max_iter=20)
    assert e_ad <= e_su + 1e-4  # allow tiny numerical slack
```

**Step 5:** Commit: `feat(pess): optimize_pess_ad entry point`.

---

## Task 10: Validation script — spin-½ reduction

**Files:**
- Create: `examples/kagome_spin12_pess_ad_benchmark.py`

**Behavior:** Run AD optimization at d=2, Δ=1, D ∈ {2, 4, 6}, χ=2D². Save energies to JSON. Compare against Liao 2019 Table I (spin-½ kagome AFM): expect E/N improving toward −0.4378 with D.

This is a runnable script, not a test. Add a smoke-test version that runs at D=2 in <60s and asserts the energy is in [−0.45, −0.40].

**Step 1: Smoke test**

```python
# tests/test_pess_validation.py
@pytest.mark.slow
def test_kagome_spin12_d2_smoke():
    state, e = run_kagome_spin12_benchmark(D=2, chi=8, max_iter=30)
    assert -0.45 < e < -0.40
```

**Step 5:** Commit: `feat(pess): spin-½ kagome AD benchmark + smoke test`.

---

## Task 11: Validation script — spin-1 Heisenberg

**Files:**
- Create: `examples/kagome_spin1_pess_ad_benchmark.py`

**Behavior:** d=3, Δ=1, D ∈ {2, 3, 4}, χ=2D². Compare against Picot 2015 (E/N ≈ −1.41 at large D). Smoke test at D=2: expect E/N in [−1.45, −1.30].

**Step 5:** Commit: `feat(pess): spin-1 Heisenberg kagome AD benchmark`.

---

## Task 12: Validation script — anisotropy sweep

**Files:**
- Create: `examples/kagome_spin1_xxz_anisotropy_sweep.py`

**Behavior:** Δ ∈ {0.0, 0.5, 1.0, 1.5, 2.0} at D=3, χ=12. Output JSON with energy and per-site ⟨S^z⟩. Smoke test omitted (this is research output, not a test).

**Step 5:** Commit: `feat(pess): XXZ anisotropy sweep on kagome`.

---

## Task 13: Public API + docs

**Files:**
- Modify: `src/tenax/algorithms/__init__.py`
- Modify: `src/tenax/__init__.py` (`__all__`)
- Modify: `README.md` (features list, example usage block)
- Modify: `docs/source/algorithms.rst` (or whichever Sphinx page lists iPEPS algorithms)

**Behavior:** Export `IPESSState`, `pess_simple_update`, `optimize_pess_ad`, `kagome_triangle_xxz_hamiltonian`, `pess_to_honeycomb_supersites`. README gets a short "Kagome iPESS with AD" subsection mirroring the existing iPEPS examples.

**Step 5:** Commit: `docs: public API + README/Sphinx for kagome iPESS AD`.

---

## Task 14: Open PR

**Step 1:** Push and open PR:
```bash
git push -u origin feat/spin1-xxz-pess-ad
gh pr create --title "feat(pess): differentiable iPESS for spin-1 XXZ on kagome" \
    --body "$(cat <<'EOF'
## Summary
- New `IPESSState` parameterization and `optimize_pess_ad` entry point implementing Liao-2019-style differentiable iPESS on the kagome lattice (Convention A: kagome → honeycomb supersites).
- Triangle simple update lifted from the existing example into the library; replaces the dummy-bond `pess_to_ipeps` coarse-graining with honeycomb supersites for clean AD.
- Validated against spin-½ kagome AFM (Liao 2019), spin-1 Heisenberg (Picot 2015), and an anisotropy sweep.

## Test plan
- [ ] `uv run pytest tests/test_pess.py tests/test_pess_ad.py -v` passes
- [ ] `uv run pytest -m core` passes (existing CI)
- [ ] `python examples/kagome_spin12_pess_ad_benchmark.py --D 2 --chi 8` reproduces published spin-½ trend
- [ ] `python examples/kagome_spin1_pess_ad_benchmark.py --D 2 --chi 8` lands in [-1.45, -1.30]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 2:** After CI passes:
```bash
gh pr merge --squash --delete-branch --auto
```

---

## Open questions to resolve during execution (not blockers)

1. **Multi-physical-leg supersite handling in CTM** (Task 8 investigation step). If the existing CTM path requires a single physical leg per site, decide on (i) fuse d×d×d → d³ at the supersite boundary, vs (ii) extend CTM to multi-physical-leg supersites. Most likely (i); document the tradeoff.
2. **Triangle energy from converged environment**: confirm `compute_energy_ctm` covers 3-site triangle ops on honeycomb supersites, or add `compute_triangle_energy_ctm`.
3. **L-BFGS hyperparameters**: start with the same defaults as the existing iPEPS optimizer; tune only if convergence stalls in Task 9.
