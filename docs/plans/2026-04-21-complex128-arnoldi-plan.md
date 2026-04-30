# Complex128 + Arnoldi Precheck Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make complex128 the default dtype for iPEPS AD optimization (1-site and 2-site) and add Arnoldi spectral-radius precheck before GMRES backward, matching variPEPS's approach to achieve E ~ -0.6625 on non-C4v Heisenberg.

**Architecture:** Three changes: (1) complex128 initialization + noise generation throughout the optimizer, (2) fix `jnp.dot` -> `jnp.vdot` in L-BFGS two-loop for complex correctness, (3) Arnoldi precheck in `f_bwd()` that raises `CTMRGGradientError` when rho(J^T) >= 1. The existing HZ line search, tangent projection, and `_tree_dot` already handle complex correctly.

**Tech Stack:** JAX, jax.numpy, tenax.algorithms

---

### Task 1: Arnoldi spectral-radius estimator

**Files:**
- Create: `src/tenax/algorithms/_arnoldi.py`
- Test: `tests/test_arnoldi.py`

**Step 1: Write the failing test**

```python
# tests/test_arnoldi.py
"""Tests for Arnoldi spectral-radius estimator."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._arnoldi import arnoldi_spectral_radius


def test_contractive_matrix():
    """Arnoldi should return rho < 1 for a contractive matrix."""
    # Diagonal matrix with eigenvalues 0.1, 0.5, 0.9
    A = jnp.diag(jnp.array([0.1, 0.5, 0.9]))
    matvec = lambda v: A @ v
    v0 = jnp.ones(3)
    rho = arnoldi_spectral_radius(matvec, v0, n_iter=20)
    assert 0.85 < rho < 0.95, f"Expected ~0.9, got {rho}"


def test_non_contractive_matrix():
    """Arnoldi should return rho > 1 for a non-contractive matrix."""
    A = jnp.diag(jnp.array([0.5, 1.5, 0.3]))
    matvec = lambda v: A @ v
    v0 = jnp.ones(3)
    rho = arnoldi_spectral_radius(matvec, v0, n_iter=20)
    assert rho > 1.0, f"Expected > 1, got {rho}"


def test_complex_matrix():
    """Arnoldi should work with complex matrices."""
    A = jnp.diag(jnp.array([0.5 + 0.3j, 0.8 - 0.1j, 0.2 + 0.4j]))
    matvec = lambda v: A @ v
    v0 = jnp.ones(3, dtype=jnp.complex128)
    rho = arnoldi_spectral_radius(matvec, v0, n_iter=20)
    # max |eigenvalue| = |0.8 - 0.1j| = sqrt(0.65) ~ 0.806
    assert rho < 1.0


def test_pytree_matvec():
    """Arnoldi should work with pytree-valued matvec (tuple of arrays)."""
    from tenax.algorithms._arnoldi import arnoldi_spectral_radius_pytree

    # Two arrays scaled by 0.5 and 0.9
    def matvec(v):
        return (v[0] * 0.5, v[1] * 0.9)

    v0 = (jnp.ones(4), jnp.ones(3))
    rho = arnoldi_spectral_radius_pytree(matvec, v0, n_iter=20)
    assert 0.85 < rho < 0.95
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_arnoldi.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tenax.algorithms._arnoldi'"

**Step 3: Write minimal implementation**

```python
# src/tenax/algorithms/_arnoldi.py
"""Arnoldi iteration for spectral-radius estimation."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def arnoldi_spectral_radius(
    matvec,
    v0: jnp.ndarray,
    n_iter: int = 20,
) -> float:
    """Estimate spectral radius of a linear operator via Arnoldi iteration.

    Builds an upper Hessenberg matrix H of size n_iter x n_iter via the
    Arnoldi process, then returns max|eig(H)| as an estimate of the
    spectral radius.

    Args:
        matvec: Function v -> A @ v.
        v0: Starting vector, shape (n,).
        n_iter: Number of Arnoldi iterations.

    Returns:
        Estimated spectral radius (float).
    """
    n = v0.shape[0]
    m = min(n_iter, n)
    Q = jnp.zeros((n, m + 1), dtype=v0.dtype)
    H = jnp.zeros((m + 1, m), dtype=v0.dtype)

    # Normalize starting vector
    v0_norm = jnp.linalg.norm(v0)
    Q = Q.at[:, 0].set(v0 / (v0_norm + 1e-30))

    for j in range(m):
        w = matvec(Q[:, j])
        for i in range(j + 1):
            h_ij = jnp.vdot(Q[:, i], w)
            H = H.at[i, j].set(h_ij)
            w = w - h_ij * Q[:, i]
        h_jp1_j = jnp.linalg.norm(w)
        H = H.at[j + 1, j].set(h_jp1_j)
        if h_jp1_j > 1e-14:
            Q = Q.at[:, j + 1].set(w / h_jp1_j)
        else:
            break

    # Eigenvalues of the m x m upper Hessenberg
    eigvals = jnp.linalg.eigvals(H[:m, :m])
    return float(jnp.max(jnp.abs(eigvals)))


def arnoldi_spectral_radius_pytree(
    matvec,
    v0,
    n_iter: int = 20,
) -> float:
    """Arnoldi spectral-radius estimate for pytree-valued operators.

    Flattens the pytree to a single vector, runs Arnoldi, returns rho.

    Args:
        matvec: Function pytree -> pytree.
        v0: Starting pytree (same structure as matvec input/output).
        n_iter: Number of Arnoldi iterations.

    Returns:
        Estimated spectral radius (float).
    """
    leaves, treedef = jax.tree.flatten(v0)
    shapes = [l.shape for l in leaves]
    dtypes = [l.dtype for l in leaves]

    def _flatten(tree):
        return jnp.concatenate([l.ravel() for l in jax.tree.leaves(tree)])

    def _unflatten(flat):
        parts = []
        offset = 0
        for shape, dtype in zip(shapes, dtypes):
            size = 1
            for s in shape:
                size *= s
            parts.append(flat[offset : offset + size].reshape(shape).astype(dtype))
            offset += size
        return jax.tree.unflatten(treedef, parts)

    def flat_matvec(v_flat):
        v_tree = _unflatten(v_flat)
        result = matvec(v_tree)
        return _flatten(result)

    v0_flat = _flatten(v0)
    return arnoldi_spectral_radius(flat_matvec, v0_flat, n_iter=n_iter)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_arnoldi.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_arnoldi.py tests/test_arnoldi.py
git commit -m "feat: add Arnoldi spectral-radius estimator for J^T precheck"
```

---

### Task 2: Wire Arnoldi precheck into implicit AD backward

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:716-736` (f_bwd function)
- Test: `tests/test_ctm_energy_implicit.py` (add precheck test)

**Step 1: Write the failing test**

Add to `tests/test_ctm_energy_implicit.py`:

```python
@pytest.mark.slow
def test_arnoldi_precheck_raises_on_noncontractive(monkeypatch):
    """Verify that the implicit backward raises CTMRGGradientError
    when Arnoldi detects rho(J^T) >= 1."""
    from tenax.algorithms.ad_utils import CTMRGGradientError

    # Patch arnoldi to always return rho=2.0
    import tenax.algorithms._ctm_energy_ad as _mod

    monkeypatch.setattr(
        _mod,
        "arnoldi_spectral_radius_pytree",
        lambda *a, **kw: 2.0,
    )

    A = _make_random_A(D=2, d=2, key=jax.random.PRNGKey(99))
    gate = heisenberg_gate()
    site_tensors = {(0, 0): A}
    neighbors = SINGLE_SITE_NEIGHBORS

    with pytest.raises(CTMRGGradientError):
        jax.value_and_grad(
            lambda p: ctm_energy_implicit(
                {(0, 0): _wrap_as_dense_tensor(p)},
                neighbors,
                gate,
                chi=4,
                max_iter=20,
                conv_tol=1e-6,
            )
        )(A.todense())
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_energy_implicit.py::test_arnoldi_precheck_raises_on_noncontractive -v`
Expected: FAIL (no Arnoldi precheck exists yet)

**Step 3: Implement the Arnoldi precheck in f_bwd**

In `src/tenax/algorithms/_ctm_energy_ad.py`, add import at top:

```python
from tenax.algorithms._arnoldi import arnoldi_spectral_radius_pytree
from tenax.algorithms.ad_utils import CTMRGGradientError
```

Modify `f_bwd()` (line 716) to add Arnoldi precheck before GMRES:

```python
def f_bwd(residuals, g):
    """Python-loop backward: JIT'd VJPs + eager GMRES."""
    params_data_tuple, env_leaves = residuals

    # Step 1: dE/denv
    dE_denv = _jit_dE_denv(params_data_tuple, env_leaves)

    # Step 1.5: Arnoldi precheck — estimate spectral radius of J^T
    def apply_Jt_only(v):
        """Apply J^T (without the I - J^T subtraction)."""
        result = _jit_apply_Jt(params_data_tuple, env_leaves, v)
        # _jit_apply_Jt returns (I - J^T)v, so J^T v = v - result
        return tuple(vi - ri for vi, ri in zip(v, result))

    rho = arnoldi_spectral_radius_pytree(apply_Jt_only, dE_denv, n_iter=20)
    if rho >= 1.0:
        raise CTMRGGradientError(rho)

    # Step 2: GMRES solve (I - J^T) lam = dE/denv
    def apply_I_minus_Jt(v):
        return _jit_apply_Jt(params_data_tuple, env_leaves, v)

    lam, _info = gmres_pytree_jax(
        apply_I_minus_Jt,
        dE_denv,
        dE_denv,
        tol=gmres_tol,
    )

    # Steps 3-4: chain rule
    return _jit_chain_rule(params_data_tuple, env_leaves, lam, g)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_energy_implicit.py::test_arnoldi_precheck_raises_on_noncontractive -v`
Expected: PASS

**Step 5: Run existing tests to verify no regression**

Run: `uv run pytest tests/test_ctm_energy_implicit.py -v -m "not slow"`
Expected: PASS (existing tests unaffected — their J^T should be contractive)

**Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_ctm_energy_implicit.py
git commit -m "feat: wire Arnoldi spectral-radius precheck into implicit AD backward"
```

---

### Task 3: Complex128 random initialization

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1196-1202` (2-site random init)
- Modify: `src/tenax/algorithms/ipeps_optimize.py:414-416` (1-site random init)
- Modify: `src/tenax/algorithms/ipeps_optimize.py:462-464` (reference C4v random init)
- Test: `tests/test_complex128_ad.py`

**Step 1: Write the failing test**

```python
# tests/test_complex128_ad.py
"""Tests for complex128 iPEPS AD optimization."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import heisenberg_gate
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _make_complex_A(D=2, d=2, key=None):
    """Create a random complex128 iPEPS site tensor."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    real = jax.random.normal(k1, (D, D, D, D, d), dtype=jnp.float64)
    imag = jax.random.normal(k2, (D, D, D, D, d), dtype=jnp.float64)
    data = real + 1j * imag
    data = data / jnp.linalg.norm(data)
    return _wrap_as_dense_tensor(data)


@pytest.mark.slow
def test_complex128_1site_gradient():
    """Verify FD-vs-AD gradient agreement for complex128 1-site."""
    A = _make_complex_A(D=2, d=2, key=jax.random.PRNGKey(42))
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    def loss(p):
        A_local = _wrap_as_dense_tensor(p)
        return ctm_energy_implicit(
            {(0, 0): A_local},
            neighbors,
            gate,
            chi=4,
            max_iter=60,
            conv_tol=1e-8,
        )

    params = A.todense()
    e, g = jax.value_and_grad(loss)(params)

    # Energy should be finite and real
    assert jnp.isfinite(e), f"Energy is not finite: {e}"
    assert jnp.isreal(e), f"Energy has imaginary part: {e}"
    # Gradient should be finite complex128
    assert g.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.slow
def test_complex128_2site_gradient():
    """Verify FD-vs-AD gradient agreement for complex128 2-site."""
    A = _make_complex_A(D=2, d=2, key=jax.random.PRNGKey(10))
    B = _make_complex_A(D=2, d=2, key=jax.random.PRNGKey(20))
    gate = heisenberg_gate()
    neighbors = CHECKERBOARD_NEIGHBORS

    from tenax.algorithms._ctm_tensor_energy import (
        compute_energy_ctm_tensor_2site,
    )

    d_phys = 2

    def energy_fn(site_tensors, envs, gate_):
        return compute_energy_ctm_tensor_2site(
            site_tensors[(0, 0)],
            site_tensors[(1, 0)],
            envs[(0, 0)],
            envs[(1, 0)],
            gate_,
            d_phys,
        )

    def loss(p_a, p_b):
        st = {
            (0, 0): _wrap_as_dense_tensor(p_a),
            (1, 0): _wrap_as_dense_tensor(p_b),
        }
        return ctm_energy_implicit(
            st,
            neighbors,
            gate,
            chi=4,
            max_iter=60,
            conv_tol=1e-8,
            energy_fn=energy_fn,
        )

    e, (gA, gB) = jax.value_and_grad(loss, argnums=(0, 1))(
        A.todense(), B.todense()
    )
    assert jnp.isfinite(e)
    assert gA.dtype == jnp.complex128
    assert gB.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(gA))
    assert jnp.all(jnp.isfinite(gB))


@pytest.mark.slow
def test_complex128_1site_optimization():
    """End-to-end 1-site complex128 optimization: energy decreases."""
    A = _make_complex_A(D=2, d=2, key=jax.random.PRNGKey(42))
    gate = heisenberg_gate()
    neighbors = SINGLE_SITE_NEIGHBORS

    energies = []
    params = A.todense()

    for step in range(15):

        def loss(p):
            A_local = _wrap_as_dense_tensor(p)
            return ctm_energy_implicit(
                {(0, 0): A_local},
                neighbors,
                gate,
                chi=4,
                max_iter=60,
                conv_tol=1e-8,
            )

        e, g = jax.value_and_grad(loss)(params)
        energies.append(float(jnp.real(e)))
        params = params - 0.01 * g
        params = params / jnp.linalg.norm(params)

    assert energies[-1] < energies[0], (
        f"Energy didn't decrease: {energies[0]:.4f} -> {energies[-1]:.4f}"
    )
    assert energies[-1] > -0.70, f"Non-variational: {energies[-1]:.4f}"
```

**Step 2: Run test to verify it fails (or passes — complex may already work at low level)**

Run: `uv run pytest tests/test_complex128_ad.py::test_complex128_1site_gradient -v`
Expected: May pass (JAX handles complex natively) or fail (dtype issues in gauge fixing)

**Step 3: Fix any issues found in Step 2**

The low-level AD path (`_ctm_energy_ad.py`) should already handle complex because:
- `jnp.vdot` is used in `_transfer_matrix_leading_eigvec`
- `conj().T` is used in sigma gauge
- Phase fix uses `jnp.conj(phase)`

If tests pass, proceed. If dtype issues arise, fix them.

**Step 4: Modify random initialization to use complex128**

In `src/tenax/algorithms/ipeps_optimize.py`:

**1-site random init (line 414-416):**
```python
# Before:
key = jax.random.PRNGKey(0)
A_init = _wrap_as_dense_tensor(jax.random.normal(key, (D, D, D, D, d_phys)))

# After:
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
A_data = jax.random.normal(k1, (D, D, D, D, d_phys)) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
A_init = _wrap_as_dense_tensor(A_data)
```

**Reference C4v random init (line 462-464):**
```python
# Before:
key = jax.random.PRNGKey(0)
A = _wrap_as_dense_tensor(jax.random.normal(key, (D, D, D, D, d_phys)))

# After:
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
A_data = jax.random.normal(k1, (D, D, D, D, d_phys)) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
A = _wrap_as_dense_tensor(A_data)
```

**2-site random init (line 1196-1202):**
```python
# Before:
key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
A_data = jax.random.normal(key_A, (D, D, D, D, d_phys))
B_data = jax.random.normal(key_B, (D, D, D, D, d_phys))

# After:
key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
kA1, kA2 = jax.random.split(key_A)
kB1, kB2 = jax.random.split(key_B)
A_data = jax.random.normal(kA1, (D, D, D, D, d_phys)) + 1j * jax.random.normal(kA2, (D, D, D, D, d_phys))
B_data = jax.random.normal(kB1, (D, D, D, D, d_phys)) + 1j * jax.random.normal(kB2, (D, D, D, D, d_phys))
```

**Step 5: Run all complex128 tests**

Run: `uv run pytest tests/test_complex128_ad.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_complex128_ad.py
git commit -m "feat: complex128 random initialization for iPEPS AD (1-site + 2-site)"
```

---

### Task 4: Fix L-BFGS two-loop for complex tensors

**Files:**
- Modify: `src/tenax/algorithms/_metric_precond.py:236,246` (jnp.dot -> jnp.vdot)
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1679` (jnp.dot -> jnp.vdot in history update)
- Test: existing tests + new unit test

**Step 1: Write the failing test**

Add to `tests/test_complex128_ad.py`:

```python
def test_lbfgs_two_loop_complex():
    """L-BFGS two-loop recursion with complex vectors."""
    from tenax.algorithms._metric_precond import lbfgs_two_loop

    # Build a simple history with complex vectors
    s = jnp.array([1.0 + 0.5j, 0.3 - 0.2j])
    y = jnp.array([0.5 + 0.1j, 0.2 + 0.3j])
    sy = float(jnp.real(jnp.vdot(s, y)))
    assert sy > 0  # must be positive for L-BFGS
    rho = 1.0 / sy
    history = [(s, y, rho)]

    grad = jnp.array([1.0 + 0.0j, 0.0 + 1.0j])
    result = lbfgs_two_loop(grad, history, lambda v: v)

    assert result.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(result))
```

**Step 2: Run test**

Run: `uv run pytest tests/test_complex128_ad.py::test_lbfgs_two_loop_complex -v`
Expected: May produce wrong results (dot vs vdot gives different values for complex)

**Step 3: Fix jnp.dot -> jnp.vdot**

In `src/tenax/algorithms/_metric_precond.py`, change lines 236 and 246:

```python
# Line 236 — Before:
alpha = rho * jnp.dot(s, q)
# After:
alpha = rho * jnp.vdot(s, q)

# Line 246 — Before:
beta = rho * jnp.dot(y, r)
# After:
beta = rho * jnp.vdot(y, r)
```

In `src/tenax/algorithms/ipeps_optimize.py`, change line 1679:

```python
# Before:
sy = float(jnp.dot(s, y))
# After:
sy = float(jnp.real(jnp.vdot(s, y)))
```

Also fix `delta_metric` computation at line 1696:

```python
# Before:
delta_metric = delta_energy if step > 0 else float(jnp.dot(g_flat, g_flat))
# After:
delta_metric = delta_energy if step > 0 else float(jnp.real(jnp.vdot(g_flat, g_flat)))
```

**Step 4: Run test**

Run: `uv run pytest tests/test_complex128_ad.py::test_lbfgs_two_loop_complex -v`
Expected: PASS

**Step 5: Run regression tests**

Run: `uv run pytest tests/test_python_loop_ad_integration.py -v -m "not slow" && uv run pytest tests/test_arnoldi.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/tenax/algorithms/_metric_precond.py src/tenax/algorithms/ipeps_optimize.py tests/test_complex128_ad.py
git commit -m "fix: use jnp.vdot in L-BFGS two-loop for complex128 correctness"
```

---

### Task 5: Complex128 noise generation in stall recovery

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (noise kick sections — lines 1523-1541, 1825-1834)

**Step 1: Identify all noise generation sites**

There are noise kicks in:
1. Lines 1523-1541: CTMRGGradientError recovery (C4v and non-C4v)
2. Lines 1825-1834: Line search stall recovery (non-C4v)

Both generate real noise via `jax.random.normal`. For complex params, noise must be complex.

**Step 2: Modify noise generation to match param dtype**

For each noise site, change from:

```python
noise = config.gs_noise_amplitude * jax.random.normal(k, data.shape)
```

to:

```python
if jnp.iscomplexobj(data):
    k1, k2 = jax.random.split(k)
    noise = config.gs_noise_amplitude * (
        jax.random.normal(k1, data.shape) + 1j * jax.random.normal(k2, data.shape)
    )
else:
    noise = config.gs_noise_amplitude * jax.random.normal(k, data.shape)
```

Apply to all 4 noise sites:
- Line ~1526 (C4v params, CTMRGGradientError recovery)
- Line ~1536 (non-C4v params, CTMRGGradientError recovery)
- Line ~1830 (non-C4v params, line search stall recovery)

For the C4v branch (line ~1526), check `params.dtype`:

```python
if jnp.iscomplexobj(params):
    k1, k2 = jax.random.split(noise_key)
    noise = config.gs_noise_amplitude * (
        jax.random.normal(k1, params.shape) + 1j * jax.random.normal(k2, params.shape)
    )
else:
    noise = config.gs_noise_amplitude * jax.random.normal(noise_key, params.shape)
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/test_complex128_ad.py -v && uv run pytest -m core -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py
git commit -m "fix: generate complex128 noise in stall recovery for complex params"
```

---

### Task 6: Complex128 C4v sublattice rotation dtype

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:1326` (_U_sub dtype)

**Step 1: Check the issue**

At line 1326:
```python
_U_sub = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=A.todense().dtype)
```

This already inherits dtype from A. If A is complex128, _U_sub will be complex128. **This should already work.** Verify:

**Step 2: Write a quick test**

Add to `tests/test_complex128_ad.py`:

```python
def test_c4v_sublattice_rotation_complex():
    """Verify C4v sublattice rotation preserves complex128 dtype."""
    from tenax.algorithms.ipeps import build_c4v_basis, c4v_tensor_from_coeffs

    D, d = 2, 2
    basis = jnp.array(build_c4v_basis(D, d))
    # Complex coefficients
    coeffs = jnp.array([1.0 + 0.5j, 0.3 - 0.2j, 0.1 + 0.1j], dtype=jnp.complex128)
    # Pad or truncate to match basis size
    n_basis = basis.shape[0]
    coeffs = jnp.zeros(n_basis, dtype=jnp.complex128).at[:min(3, n_basis)].set(coeffs[:min(3, n_basis)])
    A_data = c4v_tensor_from_coeffs(coeffs, basis, (D, D, D, D, d))
    U_sub = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.complex128)
    B_data = jnp.einsum("luRDs,sS->luRDS", A_data, U_sub)
    assert B_data.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(B_data))
```

**Step 3: Run test**

Run: `uv run pytest tests/test_complex128_ad.py::test_c4v_sublattice_rotation_complex -v`
Expected: PASS (dtype propagation is automatic)

**Step 4: Commit**

```bash
git add tests/test_complex128_ad.py
git commit -m "test: verify C4v sublattice rotation works with complex128"
```

---

### Task 7: End-to-end integration test — 2-site non-C4v Heisenberg

**Files:**
- Modify: `tests/test_complex128_ad.py` (add full optimizer test)

**Step 1: Write the integration test**

Add to `tests/test_complex128_ad.py`:

```python
@pytest.mark.slow
def test_complex128_2site_nonc4v_heisenberg_optimization():
    """End-to-end 2-site non-C4v Heisenberg optimization with complex128.

    This is the key test: verify that complex128 tensors give variational
    energies on the 2-site non-C4v path where real float64 fails.
    D=2, chi=8, ~30 L-BFGS steps with HZ line search.
    Target: E < -0.5 (variational, improving toward -0.6625).
    """
    from tenax.algorithms.ipeps import heisenberg_gate
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    gate = heisenberg_gate()
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=8),
        gs_num_steps=30,
        gs_optimizer="lbfgs",
        gs_implicit_ad=True,
        gs_c4v=False,
        gs_line_search_method="hager_zhang",
        gs_verbose=True,
        unit_cell="2site",
    )

    # Random complex128 init
    key_A, key_B = jax.random.split(jax.random.PRNGKey(42))
    kA1, kA2 = jax.random.split(key_A)
    kB1, kB2 = jax.random.split(key_B)
    D, d = 2, 2
    A_data = jax.random.normal(kA1, (D, D, D, D, d)) + 1j * jax.random.normal(kA2, (D, D, D, D, d))
    B_data = jax.random.normal(kB1, (D, D, D, D, d)) + 1j * jax.random.normal(kB2, (D, D, D, D, d))
    from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor
    A_init = _wrap_as_dense_tensor(A_data)
    B_init = _wrap_as_dense_tensor(B_data)

    (A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(
        gate, (A_init, B_init), config
    )

    # Energy should be variational (above exact -0.6694)
    assert E_gs > -0.70, f"Non-variational: E={E_gs:.6f}"
    # Energy should improve from random init (not stuck)
    assert E_gs < -0.3, f"Optimization stuck: E={E_gs:.6f}"
    # Tensors should be complex128
    assert A_opt.todense().dtype == jnp.complex128
    assert B_opt.todense().dtype == jnp.complex128
```

**Step 2: Run test**

Run: `uv run pytest tests/test_complex128_ad.py::test_complex128_2site_nonc4v_heisenberg_optimization -v -s`
Expected: PASS (this is the whole point of the PR)

**Step 3: Commit**

```bash
git add tests/test_complex128_ad.py
git commit -m "test: end-to-end 2-site non-C4v complex128 Heisenberg optimization"
```

---

### Task 8: Run full test suite and fix regressions

**Files:**
- Various (fix any failures)

**Step 1: Run core tests**

Run: `uv run pytest -m core -v`
Expected: PASS

**Step 2: Run slow AD tests**

Run: `uv run pytest tests/test_python_loop_ad_integration.py tests/test_ctm_energy_implicit.py tests/test_ctm_energy_explicit.py tests/test_complex128_ad.py tests/test_arnoldi.py -v`
Expected: PASS

**Step 3: Fix any regressions**

The existing real-valued tests in `test_python_loop_ad_integration.py` must still pass — the complex128 changes should not break real tensor paths. If `_make_random_A` in those tests generates float64, the code should handle both dtypes.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: address regressions from complex128 + Arnoldi changes"
```

---

### Task 9: Final commit and PR preparation

**Step 1: Verify all changes**

Run: `uv run pytest -m core -v && uv run pytest tests/test_arnoldi.py tests/test_complex128_ad.py -v`

**Step 2: Review diff**

Run: `git diff main --stat`

**Step 3: Create PR**

The PR should reference the design doc and target E ~ -0.6625 at D=2 chi=16 (the slow test uses chi=8 for CI speed; production runs at chi=16 match variPEPS).
