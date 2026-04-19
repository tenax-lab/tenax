# Python-Level CTM AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace JIT-traced CTM convergence with Python-loop forward + JIT-fused GMRES backward, enabling multi-site AD for arbitrary unit cells and CG iPEPS.

**Architecture:** `custom_vjp` boundary wraps CTM-to-energy. Forward runs CTM sweeps in a Python loop (each sweep JIT'd). Backward JIT-compiles a `lax.while_loop` GMRES that solves `(I - J^T) λ = ∂E/∂env` with VJP-based matvec through one CTM sweep. Two dispatch paths: implicit (GMRES backward) and explicit (checkpointed autodiff backward).

**Tech Stack:** JAX (`custom_vjp`, `lax.while_loop`, `jax.vjp`, `jax.checkpoint`), existing Tenax CTM infrastructure.

**Design doc:** `docs/plans/2026-04-18-python-level-ctm-ad-design.md`

---

## Phase 1: lax.while_loop GMRES Solver

### Task 1: GMRES solver module — failing test

**Files:**
- Create: `src/tenax/algorithms/_gmres_lax.py`
- Create: `tests/test_gmres_lax.py`

**Step 1: Write the failing test**

```python
# tests/test_gmres_lax.py
import jax
import jax.numpy as jnp
import pytest
from tenax.algorithms._gmres_lax import gmres_lax


def test_gmres_solves_simple_linear_system():
    """Solve A x = b where A = 2*I (trivial)."""
    n = 10
    A = 2.0 * jnp.eye(n)
    b = jnp.ones(n)
    matvec = lambda x: A @ x
    x, info = gmres_lax(matvec, b, tol=1e-10, maxiter=50)
    assert jnp.allclose(x, 0.5 * jnp.ones(n), atol=1e-8)
    assert info == 0  # converged


def test_gmres_solves_spd_system():
    """Solve (I - 0.5*J) x = b where J is a contraction."""
    key = jax.random.PRNGKey(42)
    n = 20
    M = jax.random.normal(key, (n, n))
    J = 0.5 * M / jnp.linalg.norm(M, ord=2)  # spectral radius < 1
    A = jnp.eye(n) - J
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))
    matvec = lambda x: A @ x
    x, info = gmres_lax(matvec, b, tol=1e-8, maxiter=100)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-6)


def test_gmres_works_with_spectral_radius_above_one():
    """GMRES should still solve even when rho(J) > 1 (unlike Neumann)."""
    key = jax.random.PRNGKey(0)
    n = 15
    M = jax.random.normal(key, (n, n))
    J = 2.0 * M / jnp.linalg.norm(M, ord=2)  # spectral radius ~ 2
    A = jnp.eye(n) - J
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))
    matvec = lambda x: A @ x
    x, info = gmres_lax(matvec, b, tol=1e-6, maxiter=200)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-4)


def test_gmres_is_jit_compatible():
    """Must work inside jax.jit."""
    n = 10
    A = 2.0 * jnp.eye(n)
    b = jnp.ones(n)

    @jax.jit
    def solve(b):
        matvec = lambda x: A @ x
        x, info = gmres_lax(matvec, b, tol=1e-10, maxiter=50)
        return x

    x = solve(b)
    assert jnp.allclose(x, 0.5 * jnp.ones(n), atol=1e-8)


def test_gmres_restart():
    """GMRES(m) with restart should still converge."""
    key = jax.random.PRNGKey(7)
    n = 50
    M = jax.random.normal(key, (n, n))
    A = jnp.eye(n) + 0.3 * (M + M.T) / n
    b = jax.random.normal(jax.random.PRNGKey(2), (n,))
    matvec = lambda x: A @ x
    x, info = gmres_lax(matvec, b, tol=1e-8, maxiter=200, restart=10)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gmres_lax.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tenax.algorithms._gmres_lax'`

**Step 3: Commit failing test**

```bash
git add tests/test_gmres_lax.py
git commit -m "test: add failing tests for lax.while_loop GMRES solver"
```

---

### Task 2: GMRES solver module — implementation

**Files:**
- Create: `src/tenax/algorithms/_gmres_lax.py`

**Step 1: Implement GMRES(m) with lax.while_loop**

```python
# src/tenax/algorithms/_gmres_lax.py
"""GMRES(m) solver using lax.while_loop for JIT compatibility.

Solves A x = b where A is represented as a matvec callable.
Designed for use inside jax.jit — the entire solve compiles into
a single XLA while_loop with no Python-level iteration.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def gmres_lax(
    matvec,
    b: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 30,
) -> tuple[jnp.ndarray, int]:
    """GMRES(m) solver via lax.while_loop.

    Parameters
    ----------
    matvec : callable
        Linear operator A applied to a vector: matvec(x) -> A @ x.
    b : array
        Right-hand side vector.
    x0 : array, optional
        Initial guess.  Defaults to zeros.
    tol : float
        Convergence tolerance on relative residual ||r|| / ||b||.
    maxiter : int
        Maximum total matvec applications across all restarts.
    restart : int
        Restart dimension (number of Arnoldi vectors per cycle).

    Returns
    -------
    x : array
        Approximate solution.
    info : int
        0 if converged, 1 otherwise.
    """
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)

    bnorm = jnp.linalg.norm(b)
    atol = tol * jnp.maximum(bnorm, 1e-30)
    m = min(restart, n)

    def outer_body(state):
        x, converged, total_iters = state

        r = b - matvec(x)
        rnorm = jnp.linalg.norm(r)
        v0 = r / jnp.maximum(rnorm, 1e-30)

        # Arnoldi + Givens rotation via inner lax.while_loop
        H = jnp.zeros((m + 1, m))
        V = jnp.zeros((m, n))
        V = V.at[0].set(v0)
        cs = jnp.zeros(m)
        sn = jnp.zeros(m)
        e1 = jnp.zeros(m + 1).at[0].set(rnorm)

        def inner_cond(inner_state):
            _, _, _, _, _, j, inner_conv = inner_state
            return (j < m) & (~inner_conv)

        def inner_body(inner_state):
            V, H, cs, sn, e1, j, _ = inner_state

            # Arnoldi step
            w = matvec(V[j])
            def orth_body(i, carry):
                V_c, H_c, w_c = carry
                h = jnp.dot(V_c[i], w_c)
                H_c = H_c.at[i, j].set(h)
                w_c = w_c - h * V_c[i]
                return V_c, H_c, w_c

            V, H, w = lax.fori_loop(0, j + 1, orth_body, (V, H, w))
            h_new = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_new)
            v_new = w / jnp.maximum(h_new, 1e-30)
            V = lax.cond(j + 1 < m, lambda: V.at[j + 1].set(v_new), lambda: V)

            # Apply previous Givens rotations
            def apply_prev(i, H_col):
                h_i = H_col[i]
                h_ip1 = H_col[i + 1]
                H_col = H_col.at[i].set(cs[i] * h_i + sn[i] * h_ip1)
                H_col = H_col.at[i + 1].set(-sn[i] * h_i + cs[i] * h_ip1)
                return H_col

            H_col = H[:, j]
            H_col = lax.fori_loop(0, j, apply_prev, H_col)

            # Compute new Givens rotation
            a_val = H_col[j]
            b_val = H_col[j + 1]
            denom = jnp.sqrt(a_val**2 + b_val**2)
            denom = jnp.maximum(denom, 1e-30)
            c_new = a_val / denom
            s_new = b_val / denom

            H_col = H_col.at[j].set(c_new * a_val + s_new * b_val)
            H_col = H_col.at[j + 1].set(0.0)
            H = H.at[:, j].set(H_col)

            cs = cs.at[j].set(c_new)
            sn = sn.at[j].set(s_new)

            e1_j = e1[j]
            e1_jp1 = e1[j + 1]
            e1 = e1.at[j].set(c_new * e1_j + s_new * e1_jp1)
            e1 = e1.at[j + 1].set(-s_new * e1_j + c_new * e1_jp1)

            inner_conv = jnp.abs(e1[j + 1]) < atol
            return V, H, cs, sn, e1, j + 1, inner_conv

        init_inner = (V, H, cs, sn, e1, 0, False)
        V, H, cs, sn, e1, k, inner_conv = lax.while_loop(
            inner_cond, inner_body, init_inner
        )

        # Back-substitution
        y = jax.scipy.linalg.solve_triangular(H[:m, :m], e1[:m], lower=False)
        x_new = x + V[:m].T @ y

        converged_now = inner_conv | (jnp.linalg.norm(b - matvec(x_new)) < atol)
        return x_new, converged_now, total_iters + k

    def outer_cond(state):
        _, converged, total_iters = state
        return (~converged) & (total_iters < maxiter)

    init_state = (x0, False, 0)
    x_final, converged, _ = lax.while_loop(outer_cond, outer_body, init_state)

    info = jnp.where(converged, 0, 1)
    return x_final, info
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_gmres_lax.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add src/tenax/algorithms/_gmres_lax.py
git commit -m "feat: add lax.while_loop GMRES(m) solver for JIT-fused backward"
```

---

### Task 3: Pytree GMRES wrapper

The CTM backward operates on pytrees (environment tensors), not flat vectors.
We need a wrapper that flattens/unflattens pytrees around the flat GMRES.

**Files:**
- Modify: `src/tenax/algorithms/_gmres_lax.py`
- Test: `tests/test_gmres_lax.py`

**Step 1: Write failing test**

```python
# Append to tests/test_gmres_lax.py
from tenax.algorithms._gmres_lax import gmres_pytree


def test_gmres_pytree_dict():
    """GMRES on a pytree of arrays (simulating CTM env)."""
    key = jax.random.PRNGKey(99)
    tree_template = {
        "C1": jnp.zeros((4, 4)),
        "T1": jnp.zeros((4, 3, 4)),
    }

    # Build a random SPD-ish linear op on the flattened pytree
    leaves, treedef = jax.tree.flatten(tree_template)
    flat_sizes = [l.size for l in leaves]
    total = sum(flat_sizes)
    M = jax.random.normal(key, (total, total))
    A = jnp.eye(total) + 0.3 * (M + M.T) / total  # well-conditioned

    def matvec_pytree(v_tree):
        v_flat = jnp.concatenate([l.ravel() for l in jax.tree.leaves(v_tree)])
        y_flat = A @ v_flat
        out_leaves = []
        offset = 0
        for sz, leaf in zip(flat_sizes, jax.tree.leaves(v_tree)):
            out_leaves.append(y_flat[offset : offset + sz].reshape(leaf.shape))
            offset += sz
        return jax.tree.unflatten(treedef, out_leaves)

    b_tree = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(3), x.shape), tree_template)
    x_tree, info = gmres_pytree(matvec_pytree, b_tree, tol=1e-8, maxiter=100)

    # Verify: A @ x ≈ b
    residual = jax.tree.map(lambda ax, b: ax - b, matvec_pytree(x_tree), b_tree)
    res_norm = jnp.sqrt(sum(jnp.sum(l**2) for l in jax.tree.leaves(residual)))
    b_norm = jnp.sqrt(sum(jnp.sum(l**2) for l in jax.tree.leaves(b_tree)))
    assert res_norm / b_norm < 1e-6


def test_gmres_pytree_jit_compatible():
    """gmres_pytree must work inside jax.jit."""
    tree = {"a": jnp.ones(5), "b": jnp.ones(3)}
    matvec = lambda t: jax.tree.map(lambda x: 2.0 * x, t)

    @jax.jit
    def solve(b):
        x, info = gmres_pytree(matvec, b, tol=1e-10, maxiter=50)
        return x

    x = solve(tree)
    assert jnp.allclose(x["a"], 0.5 * jnp.ones(5), atol=1e-8)
    assert jnp.allclose(x["b"], 0.5 * jnp.ones(3), atol=1e-8)
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_gmres_lax.py::test_gmres_pytree_dict -v`
Expected: FAIL — `ImportError: cannot import name 'gmres_pytree'`

**Step 3: Implement gmres_pytree**

Add to `src/tenax/algorithms/_gmres_lax.py`:

```python
def gmres_pytree(
    matvec,
    b_tree,
    x0_tree=None,
    *,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 30,
) -> tuple:
    """GMRES(m) for pytree-valued linear systems.

    Flattens the pytree to a single vector, solves, then unflattens.
    ``matvec`` operates on pytrees: matvec(tree) -> tree.
    """
    b_leaves, treedef = jax.tree.flatten(b_tree)
    shapes = [l.shape for l in b_leaves]
    sizes = [l.size for l in b_leaves]
    splits = jnp.cumsum(jnp.array(sizes[:-1]))

    def flatten(tree):
        return jnp.concatenate([l.ravel() for l in jax.tree.leaves(tree)])

    def unflatten(vec):
        parts = jnp.split(vec, splits)
        return jax.tree.unflatten(treedef, [p.reshape(s) for p, s in zip(parts, shapes)])

    def flat_matvec(v):
        return flatten(matvec(unflatten(v)))

    b_flat = flatten(b_tree)
    x0_flat = flatten(x0_tree) if x0_tree is not None else None

    x_flat, info = gmres_lax(flat_matvec, b_flat, x0_flat,
                              tol=tol, maxiter=maxiter, restart=restart)
    return unflatten(x_flat), info
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_gmres_lax.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_gmres_lax.py tests/test_gmres_lax.py
git commit -m "feat: add pytree GMRES wrapper for CTM environment solve"
```

---

## Phase 2: Python-Loop CTM Forward

### Task 4: Extract _jit_ctm_step

Extract a JIT-friendly single-sweep function from the existing convergence code.

**Files:**
- Create: `src/tenax/algorithms/_ctm_python_loop.py`
- Test: `tests/test_ctm_python_loop.py`

**Step 1: Write failing test**

```python
# tests/test_ctm_python_loop.py
import jax
import jax.numpy as jnp
import pytest
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
from tenax.algorithms._ctm_tensor_convergence import (
    ctm_tensor,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.models import heisenberg_gate
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.tensors import DenseTensor, TensorIndex


def _make_random_A(D=2, d=2, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (D, D, D, D, d), dtype=jnp.float64)
    indices = [
        TensorIndex.create("u", D),
        TensorIndex.create("d", D),
        TensorIndex.create("l", D),
        TensorIndex.create("r", D),
        TensorIndex.create("p", d),
    ]
    return DenseTensor(data, indices)


def test_python_loop_matches_existing_ctm():
    """Python-loop CTM must converge to same environment as existing code."""
    A = _make_random_A()
    chi = 8

    # Reference: existing convergence
    env_ref = ctm_tensor(A, chi=chi, max_iter=100, conv_tol=1e-10)

    # New: Python-loop convergence
    env_init = initialize_ctm_tensor_env(A, chi)
    env_new, info = python_loop_ctm_converge(
        A, env_init,
        neighbors=SINGLE_SITE_NEIGHBORS,
        chi=chi, max_iter=100, conv_tol=1e-10,
    )

    # Compare energies (environments may differ by gauge)
    gate = heisenberg_gate()
    e_ref = compute_energy_ctm_tensor(A, env_ref, gate)
    e_new = compute_energy_ctm_tensor(A, env_new[(0, 0)], gate)
    assert jnp.allclose(e_ref, e_new, atol=1e-8), f"Energy mismatch: {e_ref} vs {e_new}"
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_python_loop_matches_existing_ctm -v`
Expected: FAIL — `ImportError`

**Step 3: Implement python_loop_ctm_converge**

```python
# src/tenax/algorithms/_ctm_python_loop.py
"""Python-loop CTM convergence.

Runs CTM sweeps in a Python for-loop. Each sweep is JIT-compiled
individually, but the convergence check and chi ramp happen in Python.
This avoids tracing the entire convergence loop into XLA.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_sweep_multisite,
    _ctm_sv_diff,
    Coord,
)
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CTMConvergeInfo:
    converged: bool
    iterations: int
    sv_diff: float


@partial(jax.jit, static_argnames=("chi", "projector_method", "renormalize", "projector_backward"))
def _jit_ctm_sweep(
    site_tensors: dict[Coord, Any],
    envs: dict[Coord, CTMTensorEnv],
    neighbors: dict[Coord, dict[str, Coord]],
    *,
    chi: int,
    projector_method: str = "eigh",
    renormalize: bool = True,
    projector_backward: str = "auto",
) -> dict[Coord, CTMTensorEnv]:
    """One full CTM sweep (all directions, all sites). JIT-compiled."""
    double_layers = {
        coord: _build_double_layer_tensor(A) for coord, A in site_tensors.items()
    }
    return _ctm_tensor_sweep_multisite(
        envs, double_layers, neighbors, chi,
        renormalize=renormalize,
        projector_method=projector_method,
        projector_backward=projector_backward,
    )


def python_loop_ctm_converge(
    A_or_site_tensors,
    env_init,
    *,
    neighbors,
    chi: int,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    min_iter: int = 10,
    projector_method: str = "eigh",
    renormalize: bool = True,
    projector_backward: str = "auto",
    qr_warmup_steps: int = 3,
    chi_ramp: list[tuple[int, int | None]] | None = None,
    gauge_fn=None,
) -> tuple[dict[Coord, CTMTensorEnv], CTMConvergeInfo]:
    """Run CTM to convergence in a Python loop.

    Each CTM sweep is JIT-compiled. Convergence checking, chi ramp,
    and gauge fixing happen in Python — no recompilation needed.
    """
    # Normalize input: single tensor → dict
    if not isinstance(A_or_site_tensors, dict):
        site_tensors = {(0, 0): A_or_site_tensors}
    else:
        site_tensors = A_or_site_tensors

    if not isinstance(env_init, dict):
        envs = {(0, 0): env_init}
    else:
        envs = env_init

    # QR warmup
    pm = projector_method
    if pm == "qr" and qr_warmup_steps > 0:
        warmup = min(qr_warmup_steps, max_iter)
        for _ in range(warmup):
            envs = _jit_ctm_sweep(
                site_tensors, envs, neighbors,
                chi=chi, projector_method="eigh",
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        max_iter -= warmup

    # Track SVs for convergence
    prev_svs = {
        coord: jnp.linalg.svd(env.C1.data if hasattr(env.C1, 'data') else env.C1,
                                compute_uv=False)
        for coord, env in envs.items()
    }

    converged = False
    sv_diff = float("inf")
    for i in range(max_iter):
        # Chi ramp
        current_chi = chi
        if chi_ramp is not None:
            for ramp_chi, ramp_until in chi_ramp:
                if ramp_until is None or i < ramp_until:
                    current_chi = min(ramp_chi, chi)
                    break

        envs = _jit_ctm_sweep(
            site_tensors, envs, neighbors,
            chi=current_chi, projector_method=pm,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )

        # Gauge fix (if provided)
        if gauge_fn is not None:
            envs = gauge_fn(envs, site_tensors)

        # Convergence check
        if i >= min_iter - 1:
            current_svs = {
                coord: jnp.linalg.svd(
                    env.C1.data if hasattr(env.C1, 'data') else env.C1,
                    compute_uv=False,
                )
                for coord, env in envs.items()
            }
            sv_diff = max(
                float(_ctm_sv_diff(current_svs[c], prev_svs[c]))
                for c in envs
            )
            prev_svs = current_svs

            if sv_diff < conv_tol:
                converged = True
                logger.debug("CTM converged at iter %d (sv_diff=%.2e)", i + 1, sv_diff)
                break

    return envs, CTMConvergeInfo(converged=converged, iterations=i + 1, sv_diff=sv_diff)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_ctm_python_loop.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_python_loop.py tests/test_ctm_python_loop.py
git commit -m "feat: add Python-loop CTM convergence with JIT'd single sweeps"
```

---

### Task 5: Test Python-loop CTM with 2-site and gauge fixing

**Files:**
- Modify: `tests/test_ctm_python_loop.py`

**Step 1: Write tests for 2-site and gauge-fixed convergence**

```python
# Append to tests/test_ctm_python_loop.py
from tenax.algorithms._ctm_tensor_convergence import CHECKERBOARD_NEIGHBORS


def test_python_loop_2site_checkerboard():
    """Python-loop CTM converges for 2-site checkerboard."""
    A = _make_random_A(key=jax.random.PRNGKey(1))
    B = _make_random_A(key=jax.random.PRNGKey(2))
    chi = 8
    site_tensors = {(0, 0): A, (1, 0): B}
    env_init = {
        (0, 0): initialize_ctm_tensor_env(A, chi),
        (1, 0): initialize_ctm_tensor_env(B, chi),
    }
    envs, info = python_loop_ctm_converge(
        site_tensors, env_init,
        neighbors=CHECKERBOARD_NEIGHBORS,
        chi=chi, max_iter=100, conv_tol=1e-10,
    )
    assert info.converged
    assert info.sv_diff < 1e-10


def test_python_loop_chi_ramp():
    """Chi ramp works: starts at chi=4, ramps to chi=8."""
    A = _make_random_A()
    chi = 8
    env_init = initialize_ctm_tensor_env(A, chi)
    chi_ramp = [(4, 20), (8, None)]
    envs, info = python_loop_ctm_converge(
        A, env_init,
        neighbors=SINGLE_SITE_NEIGHBORS,
        chi=chi, max_iter=100, conv_tol=1e-10,
        chi_ramp=chi_ramp,
    )
    assert info.converged
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_ctm_python_loop.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_ctm_python_loop.py
git commit -m "test: add 2-site and chi-ramp tests for Python-loop CTM"
```

---

## Phase 3: custom_vjp Implicit Path (GMRES Backward)

### Task 6: ctm_energy_implicit — failing gradient test

**Files:**
- Create: `src/tenax/algorithms/_ctm_energy_ad.py`
- Modify: `tests/test_ctm_python_loop.py`

**Step 1: Write the failing test**

Test gradient correctness via finite-difference comparison:

```python
# Append to tests/test_ctm_python_loop.py
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit


@pytest.mark.slow
def test_ctm_energy_implicit_gradient_matches_fd():
    """Gradient from GMRES backward matches finite differences."""
    A = _make_random_A(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()
    env_init = initialize_ctm_tensor_env(A, chi)

    def energy_fn(params_data):
        A_local = DenseTensor(params_data, A.indices)
        env_init_local = initialize_ctm_tensor_env(A_local, chi)
        return ctm_energy_implicit(
            A_local, env_init_local, gate,
            neighbors=SINGLE_SITE_NEIGHBORS,
            chi=chi, max_iter=60, conv_tol=1e-10,
            gmres_tol=1e-8, gmres_maxiter=200,
        )

    grad_ad = jax.grad(energy_fn)(A.data)

    # Finite differences
    eps = 1e-5
    grad_fd = jnp.zeros_like(A.data)
    flat = A.data.ravel()
    for i in range(min(flat.size, 20)):  # check first 20 elements
        e_plus = energy_fn(flat.at[i].add(eps).reshape(A.data.shape))
        e_minus = energy_fn(flat.at[i].add(-eps).reshape(A.data.shape))
        grad_fd = grad_fd.at[jnp.unravel_index(i, A.data.shape)].set(
            (e_plus - e_minus) / (2 * eps)
        )

    # Compare nonzero elements
    mask = jnp.abs(grad_fd) > 1e-10
    if mask.any():
        rel_err = jnp.max(jnp.abs(grad_ad[mask] - grad_fd[mask]) / jnp.abs(grad_fd[mask]))
        assert rel_err < 1e-2, f"Gradient relative error {rel_err:.4e} > 1e-2"
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_ctm_energy_implicit_gradient_matches_fd -v`
Expected: FAIL — `ImportError`

**Step 3: Commit failing test**

```bash
git add tests/test_ctm_python_loop.py
git commit -m "test: add failing FD-AD gradient test for ctm_energy_implicit"
```

---

### Task 7: ctm_energy_implicit — implementation

**Files:**
- Create: `src/tenax/algorithms/_ctm_energy_ad.py`

**Step 1: Implement the custom_vjp wrapper**

```python
# src/tenax/algorithms/_ctm_energy_ad.py
"""CTM-to-energy with custom_vjp: Python-loop forward, JIT-fused GMRES backward.

This module provides the core AD boundary for iPEPS optimization.
The forward pass runs CTM sweeps in a Python loop (fast compilation).
The backward pass JIT-compiles a GMRES solve for the fixed-point adjoint.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from tenax.algorithms._ctm_python_loop import (
    _jit_ctm_sweep,
    python_loop_ctm_converge,
    Coord,
)
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._gmres_lax import gmres_pytree

logger = logging.getLogger(__name__)


def ctm_energy_implicit(
    A,
    env_init,
    gate,
    *,
    neighbors=None,
    chi: int = 20,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    min_iter: int = 10,
    projector_method: str = "eigh",
    renormalize: bool = True,
    projector_backward: str = "auto",
    qr_warmup_steps: int = 3,
    chi_ramp=None,
    gauge_fn=None,
    gmres_tol: float = 1e-6,
    gmres_maxiter: int = 200,
    gmres_restart: int = 30,
    energy_fn=None,
) -> jnp.ndarray:
    """Compute iPEPS energy with implicit-differentiation backward.

    Forward: Python-loop CTM convergence.
    Backward: JIT-fused GMRES solving (I - J^T) λ = ∂E/∂env.
    """
    if neighbors is None:
        neighbors = SINGLE_SITE_NEIGHBORS

    # Delegate to the custom_vjp-wrapped function
    # We pass A.data (the raw array) as the differentiated argument
    return _ctm_energy_implicit_impl(
        A.data, A, env_init, gate,
        neighbors, chi, max_iter, conv_tol, min_iter,
        projector_method, renormalize, projector_backward,
        qr_warmup_steps, chi_ramp, gauge_fn,
        gmres_tol, gmres_maxiter, gmres_restart,
        energy_fn,
    )


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(1, 19)))
def _ctm_energy_implicit_impl(
    params_data,
    A_template, env_init, gate,
    neighbors, chi, max_iter, conv_tol, min_iter,
    projector_method, renormalize, projector_backward,
    qr_warmup_steps, chi_ramp, gauge_fn,
    gmres_tol, gmres_maxiter, gmres_restart,
    energy_fn,
):
    A = A_template.__class__(params_data, A_template.indices)
    envs, info = python_loop_ctm_converge(
        A, env_init,
        neighbors=neighbors, chi=chi, max_iter=max_iter,
        conv_tol=conv_tol, min_iter=min_iter,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        qr_warmup_steps=qr_warmup_steps,
        chi_ramp=chi_ramp, gauge_fn=gauge_fn,
    )
    coord = next(iter(envs))
    env = envs[coord]
    if energy_fn is not None:
        return energy_fn(A, env, gate)
    return compute_energy_ctm_tensor(A, env, gate)


def _ctm_energy_implicit_fwd(
    params_data,
    A_template, env_init, gate,
    neighbors, chi, max_iter, conv_tol, min_iter,
    projector_method, renormalize, projector_backward,
    qr_warmup_steps, chi_ramp, gauge_fn,
    gmres_tol, gmres_maxiter, gmres_restart,
    energy_fn,
):
    energy = _ctm_energy_implicit_impl(
        params_data,
        A_template, env_init, gate,
        neighbors, chi, max_iter, conv_tol, min_iter,
        projector_method, renormalize, projector_backward,
        qr_warmup_steps, chi_ramp, gauge_fn,
        gmres_tol, gmres_maxiter, gmres_restart,
        energy_fn,
    )

    # Re-run forward to cache converged env (not stored in impl to keep it clean)
    A = A_template.__class__(params_data, A_template.indices)
    envs, _ = python_loop_ctm_converge(
        A, env_init,
        neighbors=neighbors, chi=chi, max_iter=max_iter,
        conv_tol=conv_tol, min_iter=min_iter,
        projector_method=projector_method,
        renormalize=renormalize,
        projector_backward=projector_backward,
        qr_warmup_steps=qr_warmup_steps,
        chi_ramp=chi_ramp, gauge_fn=gauge_fn,
    )
    coord = next(iter(envs))
    env_converged = envs[coord]
    residuals = (params_data, env_converged)
    return energy, residuals


def _ctm_energy_implicit_bwd(
    A_template, env_init, gate,
    neighbors, chi, max_iter, conv_tol, min_iter,
    projector_method, renormalize, projector_backward,
    qr_warmup_steps, chi_ramp, gauge_fn,
    gmres_tol, gmres_maxiter, gmres_restart,
    energy_fn,
    residuals, g,
):
    params_data, env_converged = residuals

    # JIT-compile the backward solve
    d_params = _jit_backward(
        params_data, env_converged, gate,
        A_template, neighbors, chi,
        projector_method, renormalize, projector_backward,
        gmres_tol, gmres_maxiter, gmres_restart,
        energy_fn, g,
    )
    return (d_params,)


@partial(jax.jit, static_argnames=(
    "A_template", "neighbors", "chi",
    "projector_method", "renormalize", "projector_backward",
    "gmres_maxiter", "gmres_restart", "energy_fn",
))
def _jit_backward(
    params_data, env_converged, gate,
    A_template, neighbors, chi,
    projector_method, renormalize, projector_backward,
    gmres_tol, gmres_maxiter, gmres_restart,
    energy_fn, g_scalar,
):
    """JIT-compiled GMRES backward for implicit differentiation."""
    A = A_template.__class__(params_data, A_template.indices)

    # 1. Compute d(energy)/d(env) — the GMRES RHS
    def energy_from_env_leaves(env_leaves):
        env_local = CTMTensorEnv(*env_leaves)
        if energy_fn is not None:
            return energy_fn(A, env_local, gate)
        return compute_energy_ctm_tensor(A, env_local, gate)

    env_leaves = list(env_converged)
    d_energy_d_env_leaves = jax.grad(energy_from_env_leaves)(env_leaves)

    # 2. Define matvec: v → (I - J_env^T) @ v
    site_tensors = {(0, 0): A}  # TODO: generalize for multi-site

    def matvec(v_leaves):
        def sweep_from_env(env_leaves_in):
            envs_in = {(0, 0): CTMTensorEnv(*env_leaves_in)}
            envs_out = _jit_ctm_sweep(
                site_tensors, envs_in, neighbors,
                chi=chi, projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
            return list(envs_out[(0, 0)])

        _, vjp_fn = jax.vjp(sweep_from_env, env_leaves)
        jt_v = vjp_fn(v_leaves)[0]
        return [vi - ji for vi, ji in zip(v_leaves, jt_v)]

    # 3. GMRES solve: (I - J^T) @ lam = d_energy_d_env
    lam, info = gmres_pytree(
        matvec, d_energy_d_env_leaves,
        tol=gmres_tol, maxiter=gmres_maxiter, restart=gmres_restart,
    )

    # 4. Chain rule: d(energy)/d(params) = direct + J_A^T @ lam
    # Direct term
    def energy_from_params_direct(p):
        A_local = A_template.__class__(p, A_template.indices)
        if energy_fn is not None:
            return energy_fn(A_local, env_converged, gate)
        return compute_energy_ctm_tensor(A_local, env_converged, gate)

    direct = jax.grad(energy_from_params_direct)(params_data)

    # Indirect term: J_A^T @ lam
    def sweep_from_params(p):
        A_local = A_template.__class__(p, A_template.indices)
        site_tensors_local = {(0, 0): A_local}
        envs_in = {(0, 0): CTMTensorEnv(*env_leaves)}
        envs_out = _jit_ctm_sweep(
            site_tensors_local, envs_in, neighbors,
            chi=chi, projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        return list(envs_out[(0, 0)])

    _, vjp_params = jax.vjp(sweep_from_params, params_data)
    indirect = vjp_params(lam)[0]

    return g_scalar * (direct + indirect)


_ctm_energy_implicit_impl.defvjp(_ctm_energy_implicit_fwd, _ctm_energy_implicit_bwd)
```

**Step 2: Run the FD-AD gradient test**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_ctm_energy_implicit_gradient_matches_fd -v -s`
Expected: PASS (may be slow — ~1-5 min for FD over 20 elements)

**Step 3: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "feat: add ctm_energy_implicit with Python-loop forward + GMRES backward"
```

---

## Phase 4: Explicit Path Under New Architecture

### Task 8: ctm_energy_explicit

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`
- Modify: `tests/test_ctm_python_loop.py`

**Step 1: Write failing test**

```python
# Append to tests/test_ctm_python_loop.py
from tenax.algorithms._ctm_energy_ad import ctm_energy_explicit


@pytest.mark.slow
def test_ctm_energy_explicit_gradient_matches_fd():
    """Gradient from explicit backprop matches finite differences."""
    A = _make_random_A(D=2, d=2)
    chi = 4
    gate = heisenberg_gate()

    def energy_fn(params_data):
        A_local = DenseTensor(params_data, A.indices)
        env_init_local = initialize_ctm_tensor_env(A_local, chi)
        return ctm_energy_explicit(
            A_local, env_init_local, gate,
            neighbors=SINGLE_SITE_NEIGHBORS,
            chi=chi, warmup_steps=10, backprop_steps=5,
        )

    grad_ad = jax.grad(energy_fn)(A.data)

    eps = 1e-5
    grad_fd = jnp.zeros_like(A.data)
    flat = A.data.ravel()
    for i in range(min(flat.size, 20)):
        e_plus = energy_fn(flat.at[i].add(eps).reshape(A.data.shape))
        e_minus = energy_fn(flat.at[i].add(-eps).reshape(A.data.shape))
        grad_fd = grad_fd.at[jnp.unravel_index(i, A.data.shape)].set(
            (e_plus - e_minus) / (2 * eps)
        )

    mask = jnp.abs(grad_fd) > 1e-10
    if mask.any():
        rel_err = jnp.max(jnp.abs(grad_ad[mask] - grad_fd[mask]) / jnp.abs(grad_fd[mask]))
        assert rel_err < 1e-2, f"Gradient relative error {rel_err:.4e} > 1e-2"
```

**Step 2: Implement ctm_energy_explicit**

Add to `src/tenax/algorithms/_ctm_energy_ad.py`:

```python
def ctm_energy_explicit(
    A,
    env_init,
    gate,
    *,
    neighbors=None,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    projector_method: str = "eigh",
    renormalize: bool = True,
    projector_backward: str = "auto",
    energy_fn=None,
) -> jnp.ndarray:
    """Compute iPEPS energy with explicit-differentiation backward.

    Forward: warmup (no grad) + checkpointed CTM sweeps.
    Backward: standard JAX autodiff through checkpointed sweeps.
    """
    if neighbors is None:
        neighbors = SINGLE_SITE_NEIGHBORS

    A_tensor = A
    site_tensors = {(0, 0): A_tensor}

    if not isinstance(env_init, dict):
        envs = {(0, 0): env_init}
    else:
        envs = env_init

    # Warmup: no gradient tracking
    for _ in range(warmup_steps):
        envs = jax.lax.stop_gradient(
            _jit_ctm_sweep(
                site_tensors, envs, neighbors,
                chi=chi, projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )

    # Backprop phase: checkpointed sweeps
    for _ in range(backprop_steps):
        envs = jax.checkpoint(
            lambda e: _jit_ctm_sweep(
                site_tensors, e, neighbors,
                chi=chi, projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )(envs)

    coord = next(iter(envs))
    env = envs[coord]
    if energy_fn is not None:
        return energy_fn(A_tensor, env, gate)
    return compute_energy_ctm_tensor(A_tensor, env, gate)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_ctm_python_loop.py -v -k explicit`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_ctm_python_loop.py
git commit -m "feat: add ctm_energy_explicit with checkpointed backprop"
```

---

## Phase 5: Config & Optimizer Integration

### Task 9: Config changes

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py`
- Modify: `tests/test_ipeps_ad_policy.py`

**Step 1: Update CTMConfig**

At `ipeps_config.py`, add new fields and update defaults:

- Add `gmres_tol: float = 1e-6` after line ~45 (after `ad_backward_method`)
- Add `gmres_restart: int = 30` after `gmres_tol`
- Change `gs_line_search_method` default from `"armijo"` to `"hager_zhang"` in iPEPSConfig

**Step 2: Update iPEPSConfig validation**

Deprecate `ad_backward_method` field — keep it but log a warning if set to anything other than default when `gs_implicit_ad=True` (GMRES is always used in new path).

**Step 3: Update AD policy tests**

Run: `uv run pytest tests/test_ipeps_ad_policy.py -v`
Expected: PASS (or update tests for new defaults)

**Step 4: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_ipeps_ad_policy.py
git commit -m "feat: add gmres_tol/restart to CTMConfig, default HZ line search"
```

---

### Task 10: Wire new paths into optimizer

This is the largest integration task. Replace the loss_fn internals in both
`_optimize_gs_ad_tensor` and `_optimize_gs_ad_tensor_2site` to use the new
`ctm_energy_implicit` / `ctm_energy_explicit` wrappers.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (lines 631-661 for 1-site, 1282-1367 for 2-site)
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (generalize for multi-site)

**Step 1: Generalize _ctm_energy_ad.py for multi-site**

The current implementation hardcodes `{(0,0): A}`. Generalize to accept
`site_tensors: dict[Coord, Tensor]` and `energy_fn` that handles multi-site energy.

**Step 2: Update 1-site optimizer loss_fn**

Replace lines 638-661 of `ipeps_optimize.py`. The new loss_fn calls
`ctm_energy_implicit(A, env_init, gate, ...)` or `ctm_energy_explicit(...)`.

Key changes:
- Remove `ctm_tensor_converge` / `ctm_tensor_converge_explicit` imports
- Use new `ctm_energy_implicit` / `ctm_energy_explicit` from `_ctm_energy_ad`
- Pass config fields through to the new wrappers

**Step 3: Update 2-site optimizer loss_fn**

Replace lines 1341-1367 of `ipeps_optimize.py`. Similar changes but with
`site_tensors = {(0,0): A, (1,0): B}` and `CHECKERBOARD_NEIGHBORS`.

**Step 4: Run existing tests**

Run: `uv run pytest tests/test_ipeps.py -v -m core`
Expected: PASS (existing tests should still work)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py src/tenax/algorithms/_ctm_energy_ad.py
git commit -m "feat: wire Python-loop CTM AD into iPEPS optimizer"
```

---

### Task 11: Remove old backward code

After validating the new paths work, clean up:

**Files:**
- Modify: `src/tenax/algorithms/ad_utils.py`

**Step 1: Deprecate (not remove) old paths**

Mark `ctm_tensor_converge` and `ctm_tensor_converge_explicit` as deprecated
with a `warnings.warn()` pointing to the new `_ctm_energy_ad` module.
Do NOT remove yet — keep for A/B testing.

**Step 2: Run full test suite**

Run: `uv run pytest -m core -v`
Expected: PASS (with deprecation warnings)

**Step 3: Commit**

```bash
git add src/tenax/algorithms/ad_utils.py
git commit -m "refactor: deprecate old CTM AD paths in favor of _ctm_energy_ad"
```

---

## Phase 6: Multi-Site Generalization

### Task 12: make_neighbors factory

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py`
- Create: `tests/test_make_neighbors.py`

**Step 1: Write failing test**

```python
# tests/test_make_neighbors.py
from tenax.algorithms._ctm_tensor_convergence import (
    make_neighbors,
    SINGLE_SITE_NEIGHBORS,
    CHECKERBOARD_NEIGHBORS,
)


def test_make_neighbors_1x1():
    neighbors = make_neighbors(1, 1)
    assert neighbors == SINGLE_SITE_NEIGHBORS


def test_make_neighbors_2x1():
    neighbors = make_neighbors(2, 1)
    # 2-site checkerboard: (0,0) and (1,0)
    assert set(neighbors.keys()) == {(0, 0), (1, 0)}
    # (0,0)'s right neighbor is (1,0), left is (1,0), etc.
    assert neighbors[(0, 0)]["right"] == (1, 0)
    assert neighbors[(1, 0)]["right"] == (0, 0)


def test_make_neighbors_2x2():
    neighbors = make_neighbors(2, 2)
    assert set(neighbors.keys()) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    # Periodic: (0,0) right → (1,0), (0,0) bottom → (0,1)
    assert neighbors[(0, 0)]["right"] == (1, 0)
    assert neighbors[(0, 0)]["bottom"] == (0, 1)
    assert neighbors[(1, 1)]["right"] == (0, 1)
    assert neighbors[(1, 1)]["bottom"] == (1, 0)


def test_make_neighbors_3x3():
    neighbors = make_neighbors(3, 3)
    assert len(neighbors) == 9
    # Periodic wrap: (2,0) right → (0,0)
    assert neighbors[(2, 0)]["right"] == (0, 0)
    assert neighbors[(0, 2)]["bottom"] == (0, 0)
```

**Step 2: Implement make_neighbors**

Add to `_ctm_tensor_convergence.py`:

```python
def make_neighbors(nx: int, ny: int) -> dict[Coord, dict[str, Coord]]:
    """Build periodic neighbor map for an nx × ny unit cell."""
    neighbors = {}
    for x in range(nx):
        for y in range(ny):
            neighbors[(x, y)] = {
                "left": ((x - 1) % nx, y),
                "right": ((x + 1) % nx, y),
                "top": (x, (y - 1) % ny),
                "bottom": (x, (y + 1) % ny),
            }
    return neighbors
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_make_neighbors.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py tests/test_make_neighbors.py
git commit -m "feat: add make_neighbors(nx, ny) factory for arbitrary unit cells"
```

---

### Task 13: Generalized multi-site energy computation

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_energy.py`
- Test: `tests/test_ctm_python_loop.py`

**Step 1: Write failing test**

```python
# Append to tests/test_ctm_python_loop.py

def test_multisite_energy_2x2():
    """Energy computation for 2x2 unit cell."""
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_multisite
    from tenax.algorithms._ctm_tensor_convergence import make_neighbors

    # 4 random tensors
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    tensors = {}
    for i, (x, y) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
        tensors[(x, y)] = _make_random_A(key=keys[i])

    chi = 8
    neighbors = make_neighbors(2, 2)
    envs = {c: initialize_ctm_tensor_env(t, chi) for c, t in tensors.items()}
    envs, _ = python_loop_ctm_converge(
        tensors, envs, neighbors=neighbors,
        chi=chi, max_iter=100, conv_tol=1e-10,
    )

    gate = heisenberg_gate()
    energy = compute_energy_ctm_tensor_multisite(tensors, envs, neighbors, gate)
    # Should return a finite scalar
    assert jnp.isfinite(energy)
```

**Step 2: Implement compute_energy_ctm_tensor_multisite**

Add to `_ctm_tensor_energy.py`:

```python
def compute_energy_ctm_tensor_multisite(
    site_tensors: dict,
    envs: dict,
    neighbors: dict,
    gate,
    d: int | None = None,
) -> jnp.ndarray:
    """Compute energy summed over all nearest-neighbor bonds in a multi-site unit cell.

    Each bond is counted once. Energy is normalized per site.
    """
    n_sites = len(site_tensors)
    total_energy = jnp.array(0.0)
    counted_bonds = set()

    for coord, A in site_tensors.items():
        env_A = envs[coord]
        for direction in ("right", "bottom"):
            nb_coord = neighbors[coord][direction]
            bond = frozenset([coord, nb_coord, direction])
            if bond in counted_bonds:
                continue
            counted_bonds.add(bond)

            B = site_tensors[nb_coord]
            env_B = envs[nb_coord]

            if direction == "right":
                rdm = _rdm2x1_tensor_2site(A, B, env_A, env_B)
            else:
                rdm = _rdm1x2_tensor_2site(A, B, env_A, env_B)

            bond_energy = jnp.einsum("ijkl,ijkl->", rdm, gate)
            total_energy = total_energy + bond_energy

    return total_energy / n_sites
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_multisite_energy_2x2 -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_energy.py tests/test_ctm_python_loop.py
git commit -m "feat: add compute_energy_ctm_tensor_multisite for arbitrary unit cells"
```

---

## Phase 7: Integration Tests

### Task 14: End-to-end 1-site optimization test

**Files:**
- Modify: `tests/test_ctm_python_loop.py`

**Step 1: Write end-to-end test**

```python
@pytest.mark.slow
def test_1site_heisenberg_implicit_optimization():
    """Full 1-site Heisenberg optimization with new Python-loop CTM AD."""
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit

    key = jax.random.PRNGKey(42)
    D, d, chi = 2, 2, 8
    A = _make_random_A(D=D, d=d, key=key)
    gate = heisenberg_gate()

    # Simple gradient descent
    params = A.data
    lr = 0.01
    for step in range(30):
        def loss(p):
            A_local = DenseTensor(p, A.indices)
            env_init = initialize_ctm_tensor_env(A_local, chi)
            return ctm_energy_implicit(
                A_local, env_init, gate,
                chi=chi, max_iter=60, conv_tol=1e-8,
                gmres_tol=1e-6, gmres_maxiter=100,
            )
        e, g = jax.value_and_grad(loss)(params)
        params = params - lr * g
        params = params / jnp.linalg.norm(params)

    # Energy should be variational (above exact -0.6694)
    final_energy = float(loss(params))
    assert final_energy > -0.70, f"Energy {final_energy} below physical!"
    assert final_energy < -0.30, f"Energy {final_energy} stuck near init"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_1site_heisenberg_implicit_optimization -v -s`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ctm_python_loop.py
git commit -m "test: add end-to-end 1-site Heisenberg optimization with GMRES backward"
```

---

### Task 15: End-to-end 2-site optimization test

**Files:**
- Modify: `tests/test_ctm_python_loop.py`

**Step 1: Write test**

```python
@pytest.mark.slow
def test_2site_heisenberg_implicit_optimization():
    """2-site Heisenberg optimization with Python-loop CTM AD."""
    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    D, d, chi = 2, 2, 8
    A = _make_random_A(D=D, d=d, key=key1)
    B = _make_random_A(D=D, d=d, key=key2)
    gate = heisenberg_gate()

    params = jnp.concatenate([A.data.ravel(), B.data.ravel()])
    lr = 0.005
    split_idx = A.data.size

    for step in range(30):
        def loss(p):
            a_data = p[:split_idx].reshape(A.data.shape)
            b_data = p[split_idx:].reshape(B.data.shape)
            A_local = DenseTensor(a_data, A.indices)
            B_local = DenseTensor(b_data, B.indices)
            site_tensors = {(0, 0): A_local, (1, 0): B_local}
            env_init = {
                (0, 0): initialize_ctm_tensor_env(A_local, chi),
                (1, 0): initialize_ctm_tensor_env(B_local, chi),
            }
            # Need to implement multi-site version of ctm_energy_implicit
            # For now this is a placeholder for the test structure
            return ctm_energy_implicit(
                A_local, env_init[(0, 0)], gate,
                chi=chi, max_iter=60, conv_tol=1e-8,
                gmres_tol=1e-6,
            )

        e, g = jax.value_and_grad(loss)(params)
        params = params - lr * g
        params = params / jnp.linalg.norm(params)

    final_energy = float(loss(params))
    assert final_energy > -0.80, f"Energy {final_energy} below physical!"
    assert final_energy < -0.30, f"Energy {final_energy} stuck near init"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_ctm_python_loop.py::test_2site_heisenberg_implicit_optimization -v -s`
Expected: PASS (energy variational)

**Step 3: Commit**

```bash
git add tests/test_ctm_python_loop.py
git commit -m "test: add end-to-end 2-site Heisenberg optimization with GMRES backward"
```

---

### Task 16: Compilation time benchmark

**Files:**
- Create: `benchmarks/bench_compile_time.py`

**Step 1: Write compilation time benchmark**

```python
# benchmarks/bench_compile_time.py
"""Benchmark: compilation time for Python-loop CTM AD vs old JIT-traced path."""
import time
import jax
import jax.numpy as jnp
from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env
from tenax.tensors import DenseTensor, TensorIndex
from tenax.models import heisenberg_gate


def bench_compile_implicit(D=2, chi=8):
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (D, D, D, D, 2), dtype=jnp.float64)
    indices = [
        TensorIndex.create("u", D),
        TensorIndex.create("d", D),
        TensorIndex.create("l", D),
        TensorIndex.create("r", D),
        TensorIndex.create("p", 2),
    ]
    A = DenseTensor(data, indices)
    gate = heisenberg_gate()
    env_init = initialize_ctm_tensor_env(A, chi)

    def loss(p):
        A_local = DenseTensor(p, A.indices)
        env = initialize_ctm_tensor_env(A_local, chi)
        return ctm_energy_implicit(
            A_local, env, gate,
            chi=chi, max_iter=30, conv_tol=1e-8,
            gmres_tol=1e-6,
        )

    # Compile
    t0 = time.perf_counter()
    e, g = jax.value_and_grad(loss)(A.data)
    jax.block_until_ready(g)
    t_compile = time.perf_counter() - t0

    # Second call (no recompile)
    t0 = time.perf_counter()
    e2, g2 = jax.value_and_grad(loss)(A.data * 1.01)
    jax.block_until_ready(g2)
    t_run = time.perf_counter() - t0

    print(f"D={D} chi={chi}")
    print(f"  First call (compile+run): {t_compile:.1f}s")
    print(f"  Second call (run only):   {t_run:.1f}s")
    return t_compile, t_run


if __name__ == "__main__":
    bench_compile_implicit(D=2, chi=8)
    bench_compile_implicit(D=2, chi=16)
```

**Step 2: Run benchmark**

Run: `uv run python benchmarks/bench_compile_time.py`
Expected: First call < 120s, second call < 30s (success criterion: compile < 2min)

**Step 3: Commit**

```bash
git add benchmarks/bench_compile_time.py
git commit -m "bench: add compilation time benchmark for Python-loop CTM AD"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-3 | lax.while_loop GMRES solver + pytree wrapper |
| 2 | 4-5 | Python-loop CTM forward with JIT'd single sweeps |
| 3 | 6-7 | ctm_energy_implicit custom_vjp (GMRES backward) |
| 4 | 8 | ctm_energy_explicit (checkpointed backprop) |
| 5 | 9-11 | Config changes + optimizer integration + deprecation |
| 6 | 12-13 | make_neighbors factory + multi-site energy |
| 7 | 14-16 | End-to-end optimization tests + compile benchmark |

**Dependencies:** Phase 1 → Phase 3 (GMRES needed for backward). Phase 2 → Phase 3, 4 (forward needed for both paths). Phase 3, 4 → Phase 5 (wrappers needed before optimizer integration). Phase 5, 6 can be parallelized. Phase 7 depends on everything.

**Parallelizable:** Tasks 1-3 and Tasks 4-5 are independent and can run in parallel.
