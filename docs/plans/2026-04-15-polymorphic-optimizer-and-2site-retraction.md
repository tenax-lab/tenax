# Polymorphic Optimizer Shell + 2-Site Retraction Implementation Plan

> **Status:** COMPLETED (#297 merged as PR #329). #328 retraction (Option A) was superseded by Option C (shared-C4v + implicit AD, PR #332).

**Goal:** Resolve issues #297 (polymorphic optimizer shell for SymmetricTensor AD) and #328 (2-site joint drift via explicit retraction).

**Architecture:** #297 replaces hard-coded `DenseTensor(...)` / `type(X)(dense, indices)` rebuilds in `ipeps_optimize.py` with the existing `ad_utils._wrap_tensor` helper, and normalizes via the Tensor-protocol `.norm()` method instead of `.todense()` + `jnp.linalg.norm()` + re-wrap. #328 adds Option A retraction: after each accepted L-BFGS step in `_optimize_gs_ad_tensor_2site`, project A and B back to unit norm and reset curvature history so the pre-retraction `(s_k, y_k)` pairs do not corrupt the Hessian approximation.

**Tech Stack:** JAX, Python, Tenax (DenseTensor / SymmetricTensor), pytest.

---

## Issue #297 — Polymorphic Optimizer Shell

### Task 1: Replace normalization-in-loss-fn with polymorphic `.norm()` (1-site)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:655, 716` (inside `_optimize_gs_ad_tensor.loss_fn` and `loss_fn_fwd`)
- Modify: `src/tenax/algorithms/ipeps_optimize.py` around `_eval_fresh` (same pattern)

**Step 1: Change pattern**

Old:
```python
A_data = params.todense()
A_norm_data = A_data / (jnp.linalg.norm(A_data) + 1e-10)
A_norm = DenseTensor(A_norm_data, A.indices)
```

New:
```python
A_norm = params * (1.0 / (params.norm() + 1e-10))
```

`DenseTensor.__mul__` and `SymmetricTensor.__mul__` both accept a scalar. Both return the same tensor type. `.norm()` exists on both.

**Step 2: Verify existing tests still pass**

Run: `uv run pytest -m core tests/test_ipeps.py -x -q`

### Task 2: Replace `type(X)(dense, indices)` sites with `_wrap_tensor` (1-site)

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:811` — CG metric precond
- Modify: `src/tenax/algorithms/ipeps_optimize.py:881` — L-BFGS metric precond
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1666` — noise recovery

Add `from tenax.algorithms.ad_utils import _wrap_tensor` at the top of the file (or import locally inside each function, matching the existing style).

Replace `type(X)(dense_data, X.indices)` with `_wrap_tensor(dense_data, X)`.

Replace `type(p)(noisy / (jnp.linalg.norm(noisy) + 1e-10), p.indices)` with `_wrap_tensor(noisy, p) * (1.0 / (_wrap_tensor(noisy, p).norm() + 1e-10))`. (Or keep dense computation then wrap once.)

### Task 3: Remove 1-site dense-wrap band-aid

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:601-602`

Delete:
```python
if not isinstance(A, DenseTensor):
    A = DenseTensor(A.todense(), A.indices)
```

Run: `uv run pytest -m core tests/test_ipeps.py::TestADSymmetric -xvs`

### Task 4: Do the same for 2-site path

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1233-1236` — remove 2-site band-aid
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1474-1475` — CG metric precond (`z_dict` wrap)
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1553-1558, 1576-1577` — L-BFGS h0_matvec and direction wraps

Same replacement strategy: `_wrap_tensor(dense, orig)`.

### Task 5: Non-trivial U(1) charge regression test

**Files:**
- Modify: `tests/test_ipeps.py` — add to `TestADSymmetric`

```python
def test_optimize_gs_ad_symmetric_nontrivial_u1_charges(self):
    """Regression: U(1) charges [0,1] on virtual legs should not be
    silently downgraded to DenseTensor by the optimizer shell."""
    from tenax.core.tensor import SymmetricTensor, TensorIndex, FlowDirection
    import jax.numpy as jnp

    D, d = 2, 2
    # Virtual leg index with charges [0, 1]
    v_in = TensorIndex(charges=(0, 1), flow=FlowDirection.IN)
    v_out = TensorIndex(charges=(0, 1), flow=FlowDirection.OUT)
    phys = TensorIndex(charges=(0, 1), flow=FlowDirection.OUT)
    indices = (v_in, v_in, v_out, v_out, phys)

    A = SymmetricTensor.from_dense(
        jnp.ones((D, D, D, D, d)), indices, tol=float("inf")
    )
    hamiltonian = _make_xxz_gate()  # or whatever exists
    A_opt, E, info = tenax.optimize_gs_ad(
        A, hamiltonian, chi=4, n_iter=3, gs_c4v=False,
    )
    assert isinstance(A_opt, SymmetricTensor), (
        "optimizer must preserve SymmetricTensor input type"
    )
```

Skip if building TensorIndex API is too onerous — use the simplest construction the repo currently has.

### Task 6: Remove fermionic_ipeps dense-wrap workaround

**Files:**
- Modify: `src/tenax/algorithms/fermionic_ipeps.py:486-493`

Delete the `isinstance(env.C1, DenseTensor) and isinstance(A, SymmetricTensor)` fallback. Run fermionic tests: `uv run pytest -m core tests/ -k fermionic -xvs`.

### Task 7: Open PR for #297

```bash
gh pr create --title "fix(ipeps): polymorphic optimizer shell for SymmetricTensor AD (#297)" \
    --body "..."
gh pr merge <num> --squash --delete-branch --auto
```

---

## Issue #328 — 2-Site Joint Drift (Option A: Retraction)

### Task 8: Add retraction after accepted L-BFGS step

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` — inside `_optimize_gs_ad_tensor_2site`, in the accepted-step branch of the L-BFGS loop.

After the L-BFGS update is accepted (new `A`, `B` assigned), add:

```python
# Option A retraction (issue #328): project onto ||A||=||B||=1
# to kill the joint A→λA, B→(1/λ)B flat direction. Reset curvature
# history because pre-retraction (s_k, y_k) pairs would otherwise
# corrupt the Hessian approximation.
A_norm = A.norm()
B_norm = B.norm()
A = A * (1.0 / (A_norm + 1e-12))
B = B * (1.0 / (B_norm + 1e-12))
if A_norm_prev is None or abs(A_norm - A_norm_prev) > retraction_reset_tol:
    s_history.clear()
    y_history.clear()
A_norm_prev = A_norm
```

(Adapt variable names to match the actual loop. The critical bit is: project, and reset the L-BFGS history whenever retraction moved the iterate by a non-negligible amount.)

### Task 9: Regression tests for #328

**Files:**
- Modify: `tests/test_ipeps.py`

```python
def test_ad_2site_general_noc4v_is_variational():
    """gs_c4v=False on non-Heisenberg model must be variational:
    E_gs above physical lower bound, not below."""
    # XXZ with Jz/Jxy=2.0, gs_c4v=False
    A, B = _make_2site_init(D=2)
    ham = _make_xxz_gate(Jz=2.0, Jxy=1.0)
    (A_opt, B_opt), E, info = tenax.optimize_gs_ad(
        (A, B), ham, chi=8, n_iter=20, gs_c4v=False,
    )
    assert E > -1.5, f"AD energy {E} below physical lower bound for XXZ"

def test_ad_2site_no_joint_drift():
    """||A|| and ||B|| should not drift by more than 10x across steps."""
    norms = []
    def callback(step, A, B, E):
        norms.append((float(A.norm()), float(B.norm())))
    # ... run optimizer with callback
    A_norms = [n[0] for n in norms]
    assert max(A_norms) / min(A_norms) < 10.0
```

If the optimizer doesn't expose a callback, track norms by running fewer iterations and checking final A/B norms only.

### Task 10: Open PR for #328

```bash
gh pr create --title "fix(ipeps): Option A retraction for 2-site AD (#328)" \
    --body "Closes #328 with minimal Option A (explicit retraction onto ||A||=||B||=1 after each accepted L-BFGS step, with curvature history reset). Options B (joint multi-site metric) and C (full Riemannian L-BFGS) tracked as follow-up issues."
gh pr merge <num> --squash --delete-branch --auto
```

---

## Testing Strategy

After each task: `uv run pytest -m core tests/test_ipeps.py -x -q`

Before each PR:
- `uv run pytest -m core` (full core suite)
- `uv run pytest tests/test_ipeps.py::TestADSymmetric -xvs` (SymmetricTensor coverage)
- `uv run pytest tests/ -k fermionic -xvs` (fermionic path)

## DRY / YAGNI Notes

- Reuse the existing `ad_utils._wrap_tensor` helper. Do NOT define a new one.
- Do NOT touch `precondition_gradient` / `precondition_gradient_multisite` in this PR — they internally `.todense()` the gradient, which is correct (if sub-optimal) for SymmetricTensor inputs as long as the result is wrapped back. Optimizing the metric precond to preserve block-sparsity is a separate optimization.
- Do NOT attempt Option B or C for #328 in this session. Option A alone: is it enough? Bench data will tell. If not, open follow-up.
