# F3: Fused-Backward `@jax.jit` for Tenax Implicit-AD CTM

> **For Claude (new session):** This is the F3 follow-up to PR #415 (squash 1af2ee8).
> Start here. The diagnosis doc at
> `docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md` §F3 has the full
> background; the F2 plan at
> `docs/plans/2026-05-09-ipeps-ad-fixed-point-adjoint-plan.md` is the
> immediate predecessor. Read both — you don't need to re-derive any of it.
>
> **Branch:** off `main` (NOT off the deleted F2 branch). Suggested name:
> `ipeps-ad-fused-backward-jit`. A worktree is recommended — see Task 0.
>
> **REQUIRED SUB-SKILL:** Use `superpowers:executing-plans` to implement
> this plan task-by-task.

## Goal

Replace the three separate JIT'd backward helpers (`_jit_dE_denv`,
`_jit_apply_Jt`, `_jit_chain_rule`) plus the eager Python adjoint loop
with **one** `@jax.jit` function whose body fuses all three steps and
runs the adjoint fixed-point inside `jax.lax.while_loop` — variPEPS
`_ctmrg_rev_workhorse` style.

Expected outcome:
- Cold-compile time at D=2 χ=8 single_site (current F2 baseline ~16 s)
  drops by ≥ 30% — only one trace+lower+compile pass instead of three.
- Per-step wall-clock at D=2 χ=16 single_site drops by ≥ 20% — the
  adjoint loop body becomes a cached XLA dispatch instead of a Python
  `_jit_apply_Jt` call per iter.
- The variPEPS-compare benchmark at single_site D=2 χ=16 fits inside
  the 30-min subprocess budget on CPU (Definition of Done #4 from F2,
  still unmet after PR #415).

## Architecture

One `@jax.jit`'d closure built per `_VJP_CACHE` entry. Inside the JIT:
construct the env-side and params-side `jax.vjp` closures **once**,
then iterate `λ_{k+1} = b + vjp_env(λ_k)` as a `lax.while_loop` whose
carry is `(lam, prev_diff, k, diverged)`. After the loop, apply the
params-side VJP and the direct `dE/dparams` chain. Divergence inside
the loop sets a scalar bool flag that the caller reads to decide
whether to fall back to eager GMRES.

The `arnoldi_precheck` and the GMRES fallback both stay **outside**
the fused JIT — Arnoldi raises `CTMRGGradientError` (Python-level
control flow), and the eager-GMRES path is the existing `"gmres"`
branch reused verbatim.

## Tech Stack

- `jax.jit` + `jax.lax.while_loop` (only JAX primitives; no
  `jax.experimental` features).
- `jax.vjp` constructed inside the traced JIT body (legal — produces
  a callable that traces alongside the parent JIT).
- Existing helpers: `_make_jit_ctm_step`, `_apply_gauge_fix`,
  `gmres_pytree_jax`, `arnoldi_spectral_radius_pytree`.
- Tests: `pytest -m core` regression + new gradient-parity test +
  microbench script.

## Code change scope

**Files modified:**
- `src/tenax/algorithms/_ctm_energy_ad.py` — replace `_jit_dE_denv`,
  `_jit_apply_Jt`, `_jit_chain_rule`, and the Python adjoint loop in
  `f_bwd` (lines ~684–895) with one `_jit_fused_fixed_point_bwd`.
  The `"gmres"` opt-out branch is **untouched** — it keeps calling
  `_jit_apply_Jt` (which is retained for that branch only, see below)
  or its own JIT'd `gmres_pytree_jax`.

**Files created:**
- `tests/test_ipeps_ad_f3_fused_bwd.py` — gradient-parity tests
  (vs F2 fixed_point and vs gmres) and an in-loop divergence smoke
  test.
- `benchmarks/varipeps_compare/microbench_f3.py` — small standalone
  cold/warm-step microbench to attribute the F3 win.

**Out of scope:**
- `_run_forward` is untouched.
- `CTMConfig.adjoint_method` API surface is untouched. F3 is an
  internal refactor of the `"fixed_point"` path. If users want the
  pre-F3 behaviour, `git revert` is the answer — not a new knob.
- `_VJP_CACHE` keying is unchanged.
- F4 (`jax.experimental.implicit_diff.custom_root`) is still out of
  scope.

## Design notes

### Why one fused JIT helps when three small JITs don't

PR #415 verified empirically (`jax.log_compiles()` capture at χ=4)
that the three small JITs together total ~7 s of compile time.
Consolidating their compile time alone is not the win. The win comes
from **the VJP closure being constructed once and reused** for all
adjoint iterations:

- Currently `_jit_apply_Jt` is called once per fixed-point iter from
  Python. Each call dispatches a cached XLA program (cheap), but
  each call is a separate JAX dispatch with its own pytree
  flatten/unflatten roundtrip and Python overhead.
- variPEPS's `_ctmrg_rev_workhorse` builds `vjp_env` once, then the
  `lax.while_loop` body invokes that closure inside the same XLA
  program. Dispatch overhead → zero.

At chi=16 with ~20–30 fixed-point iters per backward, the
per-iter Python+dispatch overhead adds up. The diagnosis doc's
empirical attribution at chi=4 underweights this because both
the per-iter cost and the iter count grow with chi.

### Carry shape for `lax.while_loop`

The carry pytree is fixed-shape (jit requirement):
```
carry = (lam, prev_diff, k, diverged)
  lam:       same pytree shape as env_leaves (tuple of jax arrays)
  prev_diff: scalar f64 (or f32 to match env dtype)
  k:         scalar int32
  diverged:  scalar bool
```

`prev_diff` is initialised to `jnp.inf` so the first iter never
triggers the divergence check. `k` increments each iter. The cond
function exits when any of: `k >= maxiter`, `diverged`,
`prev_diff < tol`.

### In-loop divergence check

Replicate the F2 in-loop guard (`if diff > prev_diff and k > 5`) as
`lax.cond` — a scalar bool comparison inside the body. When it
fires, the body sets `diverged = True` and the cond function exits
on the next iter. The caller (in `f_bwd` outside the JIT) then runs
the eager GMRES fallback. Since the JIT returns a scalar bool, this
is a clean Python branch.

### `apply_Jt` semantic is raw `J^T`, not `(I - J^T)`

The new helper's inner `apply_Jt(v)` must return the **raw** `J^T v`
(i.e. `vjp_env_fn(v)[0]` directly), not the `(I - J^T) v` wrapping
that the surviving F2 `_jit_apply_Jt` returns. The Neumann iteration
`λ_{k+1} = b + J^T λ_k` uses `J^T` directly. A `(I - J^T)` apply_Jt
here would silently diverge within ~7 steps and rely on the GMRES
fallback for every backward call — defeating F3's purpose without
failing any gradient-correctness test.

### Where the F2-only `_jit_apply_Jt` survives

Keep the standalone `_jit_apply_Jt` definition for two callers:
1. `arnoldi_precheck` — needs an eager closure.
2. The `adjoint_method == "gmres"` opt-out branch — eager GMRES
   needs an eager matvec closure.

So the file contains both the new fused JIT and the existing small
JIT. This is fine: `_jit_apply_Jt` doesn't get traced unless one of
those two paths runs.

### Why g (cotangent) is passed as a JIT arg, not closed over

`custom_vjp` always passes `g = jnp.ones(())` for our scalar energy,
so it's effectively constant. But passing it as a JIT arg (rather
than baking it in at trace time) costs one extra dispatch arg and
keeps the JIT signature stable across hypothetical future callers
that pass non-unit cotangents. Trivial cost, more general.

---

## Tasks

### Task 0: Set up the branch + worktree

**Files:** none — git plumbing.

**Step 1: Create the worktree off `main`**

Run:
```bash
cd /home/yjkao/tenax
git fetch origin
git worktree add ../tenax-f3-fused-bwd origin/main -b ipeps-ad-fused-backward-jit
cd ../tenax-f3-fused-bwd
```

Expected: new worktree at `../tenax-f3-fused-bwd` checked out at the
new branch.

**Step 2: Verify branch state**

Run:
```bash
git status
git log --oneline -5
```

Expected: branch `ipeps-ad-fused-backward-jit`; HEAD matches
`origin/main` (the F2 squash commit `1af2ee8` should be in the log).

**Step 3: Install hooks**

Run:
```bash
uv run pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

(Memory `feedback_precommit.md`: hooks must be installed before any
commit on a fresh worktree.)

---

### Task 1: Add the F2 baseline microbench

Before changing code, lock in measurable F2 numbers so the F3 win is
quantified, not asserted.

**Files:**
- Create: `benchmarks/varipeps_compare/microbench_f3.py`

**Step 1: Write the microbench**

```python
"""F3 microbench: cold compile + 1 backward, warm 2 backwards.

Run twice — once on F2 (baseline), once on F3 (this branch). Report
JSON to stdout for easy diff.
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp

# Force CPU complex128 for apples-to-apples with variPEPS reference.
os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heisenberg_gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def main() -> None:
    H = _heisenberg_gate()
    key = jax.random.PRNGKey(2026)
    A = jax.random.normal(key, (2, 2, 2, 2, 2), dtype=jnp.complex128)
    A = _wrap_as_dense_tensor(A / jnp.linalg.norm(A))

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=8,
            max_iter=40,
            conv_tol=1e-6,
            forward_gauge="phase",
            gmres_tol=1e-6,
            gmres_maxiter=200,
            arnoldi_precheck=False,
            adjoint_method="fixed_point",
        )

    grad_fn = jax.grad(loss)

    t0 = time.perf_counter()
    g0 = grad_fn(A)
    jax.block_until_ready(g0)
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(2):
        t0 = time.perf_counter()
        g = grad_fn(A)
        jax.block_until_ready(g)
        warm.append(time.perf_counter() - t0)

    print(json.dumps({"cold_s": cold, "warm_s": warm}, indent=2))


if __name__ == "__main__":
    main()
```

**Step 2: Run on F2 baseline**

Since the worktree is at `origin/main` (F2 already merged), the
current state IS F2. Run:
```bash
uv run python benchmarks/varipeps_compare/microbench_f3.py
```

Expected: a JSON like `{"cold_s": ~16, "warm_s": [~1.3, ~2.1]}`
(per the F2 memory entry). Record the actual numbers — they go in
the F3 PR description.

**Step 3: Commit the microbench**

```bash
git add benchmarks/varipeps_compare/microbench_f3.py
git commit -m "bench(ipeps-ad): microbench harness for F3 cold/warm bwd"
```

---

### Task 2: Write the failing F3 gradient-parity test

**Files:**
- Create: `tests/test_ipeps_ad_f3_fused_bwd.py`

**Step 1: Write the test**

```python
"""Tests for the F3 fused-backward refactor of implicit-AD CTM.

The fused JIT must produce element-wise identical gradients to the
F2 Python-loop fixed-point path within float roundoff.  It must also
match the GMRES path within the existing solver tolerance, since the
underlying linear system is the same.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heisenberg_gate():
    d = 2
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(d, d, d, d)


def _random_peps(seed=2026, D=2, d=2):
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (D, D, D, D, d))
    return A / (jnp.linalg.norm(A) + 1e-10)


def _grad_for_method(method: str):
    H = _heisenberg_gate()
    A = _wrap_as_dense_tensor(_random_peps())

    def loss(A_):
        return ctm_energy_implicit(
            {(0, 0): A_},
            SINGLE_SITE_NEIGHBORS,
            H,
            chi=8,
            max_iter=40,
            conv_tol=1e-6,
            forward_gauge="phase",
            gmres_tol=1e-6,
            gmres_maxiter=200,
            arnoldi_precheck=False,
            adjoint_method=method,
        )

    return jax.grad(loss)(A)


@pytest.mark.algorithm
def test_fused_bwd_matches_gmres_at_chi8():
    """F3 fused JIT must match eager GMRES gradient within solver tol."""
    g_fp = _grad_for_method("fixed_point")
    g_gm = _grad_for_method("gmres")

    g_fp_arr = np.asarray(g_fp.todense() if hasattr(g_fp, "todense") else g_fp)
    g_gm_arr = np.asarray(g_gm.todense() if hasattr(g_gm, "todense") else g_gm)
    np.testing.assert_allclose(
        g_fp_arr, g_gm_arr, rtol=1e-5, atol=1e-7,
        err_msg="F3 fused fixed_point and GMRES gradients diverged",
    )
```

**Step 2: Run it on the F2 baseline (this branch's HEAD)**

Run:
```bash
uv run pytest tests/test_ipeps_ad_f3_fused_bwd.py -v
```

Expected: PASS. (At this point the test exercises the F2 fixed-point
implementation, so it should pass — this baseline confirms the test
itself is correct *before* we change the implementation.)

**Step 3: Commit the test**

```bash
git add tests/test_ipeps_ad_f3_fused_bwd.py
git commit -m "test(ipeps-ad): F3 fused-backward gradient parity test (F2 baseline pass)"
```

---

### Task 3: Implement the fused JIT — body skeleton

This is the core refactor. Done as one task because the body's
internal structure is tightly coupled — splitting it would create
an intermediate non-compiling state.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:684-895` (replace
  the three small JITs and the Python adjoint loop)

**Step 1: Add the new fused JIT next to the existing helpers**

Insert the new function **after** `_jit_chain_rule` (around line
782) and **before** `_jit_gmres_solve` (around line 784). Do not
delete anything yet:

```python
@jax.jit
def _jit_fused_fixed_point_bwd(
    params_data_tuple,
    env_leaves,
    g_scalar,
):
    """F3: fused dE/denv + adjoint fixed-point + chain rule.

    One @jax.jit boundary in place of the F2 trio
    (_jit_dE_denv, _jit_apply_Jt, _jit_chain_rule) + Python adjoint
    loop. The adjoint runs as lax.while_loop so the loop body and
    the J^T VJP closure trace once and reuse on every iter.

    Returns
    -------
    grads : tuple of pytrees
        Same shape as ``params_data_tuple``.
    diverged : bool scalar
        Set when the in-loop check (diff > prev_diff after step 5)
        fires. Caller falls back to eager GMRES when True.
    converged : bool scalar
        Set when ``diff < gmres_tol``. Diagnostic only.
    n_iter : int32 scalar
        Diagnostic only.
    """
    gate_ = mutables["gate"]
    energy_fn_ = mutables["energy_fn"]
    site_tensors = dict(zip(coords, params_data_tuple))
    env_treedef = _cached["env_treedef"]
    envs = jax.tree.unflatten(env_treedef, env_leaves)

    # --- 1. dE/denv via VJP through the energy function ---
    def energy_from_env(env_leaves_flat):
        e = jax.tree.unflatten(env_treedef, env_leaves_flat)
        if energy_fn_ is not None:
            return energy_fn_(site_tensors, e, gate_)
        return _default_energy(site_tensors, e, gate_, coords, neighbors)

    _, vjp_energy_env = jax.vjp(energy_from_env, env_leaves)
    dE_denv = vjp_energy_env(jnp.ones(()))[0]

    # --- 2. Build the J^T matvec closure ONCE inside this JIT ---
    def gauge_fixed_sweep_from_env(env_leaves_flat):
        e = jax.tree.unflatten(env_treedef, env_leaves_flat)
        e_ref = jax.tree.map(jax.lax.stop_gradient, e)
        e_out, _eps = jit_step_bwd(
            site_tensors,
            e,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        e_fixed = _apply_gauge_fix(e_out, e_ref)
        return tuple(jax.tree.leaves(e_fixed))

    _, vjp_env_fn = jax.vjp(gauge_fixed_sweep_from_env, env_leaves)

    def apply_Jt(v):
        # vjp_env_fn(v)[0] is the raw VJP through gauge_fixed_sweep_from_env,
        # which equals J^T v. Note this differs from the surviving F2
        # helper _jit_apply_Jt — that one wraps the VJP as (I - J^T) v
        # because GMRES needs the matvec for (I - J^T)·λ = b. The Neumann
        # iteration here uses J^T directly: λ_{k+1} = b + J^T λ_k.
        return vjp_env_fn(v)[0]

    # --- 3. Adjoint fixed-point inside lax.while_loop ---
    real_dtype = jnp.real(dE_denv[0]).dtype if dE_denv else jnp.float64
    init_lam = dE_denv  # lambda_0 = b
    init_diff = jnp.array(jnp.inf, dtype=real_dtype)
    init_k = jnp.array(0, dtype=jnp.int32)
    init_diverged = jnp.array(False)

    tol_arr = jnp.array(gmres_tol, dtype=real_dtype)
    maxiter_arr = jnp.array(gmres_maxiter, dtype=jnp.int32)

    def cond_fn(carry):
        _lam, prev_diff, k, diverged = carry
        return (
            (k < maxiter_arr) & (~diverged) & (prev_diff > tol_arr)
        )

    def body_fn(carry):
        lam, prev_diff, k, _diverged = carry
        jt_lam = apply_Jt(lam)
        new_lam = tuple(b + j for b, j in zip(dE_denv, jt_lam))
        diff = sum(
            jnp.linalg.norm(n - p) for n, p in zip(new_lam, lam)
        ).astype(real_dtype)
        # Divergence guard: same trigger as F2 Python loop.
        new_diverged = (diff > prev_diff) & (k > jnp.array(5, jnp.int32))
        return (new_lam, diff, k + jnp.array(1, jnp.int32), new_diverged)

    lam_final, final_diff, n_iter, diverged = jax.lax.while_loop(
        cond_fn, body_fn,
        (init_lam, init_diff, init_k, init_diverged),
    )
    converged = final_diff <= tol_arr

    # --- 4. Chain rule: direct dE/dparams + indirect J_p^T @ lam ---
    def energy_from_params(p_tuple):
        st = dict(zip(coords, p_tuple))
        if energy_fn_ is not None:
            return energy_fn_(st, envs, gate_)
        return _default_energy(st, envs, gate_, coords, neighbors)

    _, vjp_energy_params = jax.vjp(energy_from_params, params_data_tuple)
    direct = vjp_energy_params(jnp.ones(()))[0]

    def gauge_fixed_sweep_from_params(p_tuple):
        st = dict(zip(coords, p_tuple))
        e_out, _eps = jit_step_bwd(
            st,
            envs,
            chi=chi,
            projector_method=projector_method,
            renormalize=renormalize,
            projector_backward=projector_backward,
        )
        e_fixed = _apply_gauge_fix(e_out, envs)
        return tuple(jax.tree.leaves(e_fixed))

    _, vjp_sweep_params = jax.vjp(gauge_fixed_sweep_from_params, params_data_tuple)
    indirect = vjp_sweep_params(lam_final)[0]

    total = jax.tree.map(lambda d, ind: g_scalar * (d + ind), direct, indirect)
    return (total,), diverged, converged, n_iter
```

**Step 2: Quick syntax check**

Run:
```bash
uv run python -c "from tenax.algorithms import _ctm_energy_ad; print('ok')"
```

Expected: `ok`. (No actual JIT trace happens until first call.)

**Step 3: Do NOT commit yet** — the new helper is unwired. Continue
to Task 4.

---

### Task 4: Wire `f_bwd` to the new fused JIT (fixed_point branch only)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py:824-895` (`f_bwd`
  body — only the `adjoint_method == "fixed_point"` branch).

**Step 1: Replace the fixed_point branch**

Find the block that starts with `if adjoint_method == "fixed_point":`
(currently line 850) and ends just before `else:` (line 881).
Replace with:

```python
        if adjoint_method == "fixed_point":
            # F3 fused JIT: dE/denv + adjoint fixed-point + chain rule
            # in one trace. Returns final grads directly; if the in-loop
            # divergence guard fired we fall back to eager GMRES below
            # (same path as adjoint_method=="gmres").
            grads_tuple, diverged, _converged, _n_iter = (
                _jit_fused_fixed_point_bwd(params_data_tuple, env_leaves, g)
            )
            if bool(jax.device_get(diverged)):
                # Diverged inside the JIT loop — fall back to eager GMRES
                # using the surviving _jit_apply_Jt closure.
                lam, _info = gmres_pytree_jax(
                    _eager_apply_I_minus_Jt,
                    dE_denv,
                    dE_denv,
                    tol=gmres_tol,
                    maxiter=gmres_maxiter,
                    restart=gmres_restart,
                )
                lam_leaves = tuple(jax.tree.leaves(lam))
                return _jit_chain_rule(
                    params_data_tuple, env_leaves, lam_leaves, g
                )
            return grads_tuple
```

**Step 2: Verify the gmres branch is untouched**

The `else:` branch starting `# adjoint_method == "gmres"` (current
line 882) and the trailing `lam_leaves = ... ; return _jit_chain_rule
(...)` (current lines 892–895) must remain. The fixed_point branch
now `return`s early when it succeeds; the gmres branch falls through
to the existing chain_rule call.

**Step 3: Run the gradient-parity test**

```bash
uv run pytest tests/test_ipeps_ad_f3_fused_bwd.py -v
```

Expected: `test_fused_bwd_matches_gmres_at_chi8` PASS. If it fails
with a tolerance miss, inspect the diff — likely a missing
`stop_gradient` or pytree-leaf-order mismatch between
`gauge_fixed_sweep_from_env` here vs. the F2 `_jit_apply_Jt`.

**Step 4: Run the F2 regression tests**

```bash
uv run pytest tests/test_ipeps_ad_adjoint_methods.py -v
```

Expected: both F2 tests PASS (gradient parity + Arnoldi precheck
rejection — neither path is affected by F3 because Arnoldi runs
outside the JIT and the gmres branch is untouched).

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_ipeps_ad_f3_fused_bwd.py
git commit -m "feat(ipeps-ad): F3 fused @jax.jit backward with lax.while_loop adjoint"
```

---

### Task 5: Run the core regression suite

**Files:** none — verification only.

**Step 1: Core suite**

```bash
uv run pytest -m core -x
```

Expected: same pass count as PR #415 (789 passed last known).

**Step 2: Full implicit-AD suite**

```bash
uv run pytest tests/test_ipeps_excitations.py::TestOptimizeGsAd -v
```

Expected: all four `TestOptimizeGsAd` tests PASS:
`test_runs_without_error`, `test_energy_decreases`,
`test_heisenberg_negative_energy`, `test_heisenberg_excitation_dispersion`.

**Step 3: If any test fails — diagnose, do NOT push the band-aid**

Most likely failure modes (in order of likelihood):
1. Pytree leaf order mismatch in `gauge_fixed_sweep_from_env` vs the
   F2 `_jit_apply_Jt` — JAX pytree leaves are order-sensitive but
   the env_treedef capture in `_cached` should make this identical.
   Fix: ensure the tuple wrapping matches verbatim.
2. The `dE_denv[0]` dtype probe in Task 3 fails on an empty tuple
   (shouldn't happen — the env always has at least one leaf — but
   defensive: fall back to `jnp.float64` if `len(dE_denv) == 0`).
3. `lax.while_loop` rejects the `diff > prev_diff` comparison
   because `prev_diff` is `inf` and the body's `diff` is concrete —
   this is fine in JAX, but if it errors, replace `init_diff =
   jnp.inf` with a large concrete float (e.g. `1e30`).

**Step 4: Commit any fixes as separate commits**

Do NOT amend the Task 4 commit — keep the history bisectable.

---

### Task 6: Re-run the microbench, record F3 numbers

**Files:**
- Update: `benchmarks/varipeps_compare/microbench_f3.py` if needed
  (no change expected).

**Step 1: Re-run**

```bash
uv run python benchmarks/varipeps_compare/microbench_f3.py
```

Expected: cold time drops from F2's ~16 s to ≤ ~11 s. Warm-step
times drop modestly (~10–20%) since the warm path now uses the
fused dispatch instead of three separate ones.

**Step 2: Diff vs F2 baseline**

Compare to the JSON recorded in Task 1, Step 2. If cold-time savings
are < 30%, the F3 win didn't materialise — investigate before
proceeding to the full benchmark (which costs hours). Likely
culprits:
- The JIT is being re-traced per call because of an unhashable
  arg → check `_VJP_CACHE` is being hit (add a print, then remove).
- The VJP closure isn't actually being reused inside the
  `lax.while_loop` body → `jax.make_jaxpr(_jit_fused_fixed_point_bwd)
  (...)` should show one CTM-step VJP, not N copies. If it shows N
  copies, the body_fn is closing over the closure incorrectly
  (rare; would need to factor `apply_Jt` differently).

**Step 3: Record numbers in the PR description (later)**

Save the JSON output for the PR body. Do not commit it.

---

### Task 7: Run the variPEPS-compare benchmark (Definition of Done #4)

**Files:**
- Update: `benchmarks/varipeps_compare/published_results/STATUS.md`
  (after the run completes).

**Step 1: Run the orchestrator**

```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.varipeps_compare.compare \
    --device cpu \
    --results-dir benchmarks/varipeps_compare/results
```

Expected: Tenax `single_site D=2 chi=16` finishes inside the 30-min
subprocess budget. variPEPS data is already cached, so this run
only adds the Tenax side.

**Step 2: Inspect the result JSON**

Run:
```bash
ls benchmarks/varipeps_compare/results/
cat benchmarks/varipeps_compare/results/tenax_single_site_D2_chi16.json
```

Expected: a JSON with final energy ≈ −0.6625 (matching variPEPS to
~1e-4) and total wall-clock < 1800 s.

**Step 3: Update STATUS.md**

Add a section documenting F3 numbers, mirroring the F2 entry.
Keep it terse — the diagnosis doc and this plan have the why.

**Step 4: Commit**

```bash
git add benchmarks/varipeps_compare/results/ \
        benchmarks/varipeps_compare/published_results/STATUS.md
git commit -m "bench: F3 closes variPEPS-compare 30-min budget at single_site D=2 chi=16"
```

---

### Task 8: Verification gate (use `superpowers:verification-before-completion`)

**Files:** none — verification.

**Step 1: Re-run all of these in sequence**

```bash
uv run pytest -m core -x
uv run pytest tests/test_ipeps_ad_adjoint_methods.py -v
uv run pytest tests/test_ipeps_ad_f3_fused_bwd.py -v
uv run pytest tests/test_ipeps_excitations.py::TestOptimizeGsAd -v
```

Expected: all PASS, no skips except `slow`-marked tests outside core.

**Step 2: Confirm Definition-of-Done line items**

Walk through the list in §"Definition of done" below. Anything not
green → loop back, do not push.

---

### Task 9: Open the PR

**Files:** none — git/gh plumbing.

**Step 1: Push the branch**

```bash
git push -u origin ipeps-ad-fused-backward-jit
```

**Step 2: Create the PR**

```bash
gh pr create --title "feat(ctm): F3 fused @jax.jit backward for implicit-AD CTM" \
    --body "$(cat <<'EOF'
## Summary

- Fuses `_jit_dE_denv` + adjoint fixed-point loop + `_jit_chain_rule`
  into one `@jax.jit` with `lax.while_loop` for the adjoint —
  variPEPS `_ctmrg_rev_workhorse` style.
- Internal refactor of `CTMConfig.adjoint_method="fixed_point"`; no
  API surface change. The `"gmres"` opt-out path is untouched.
- Microbench (D=2 χ=8 single_site CPU complex128): cold compile
  drops from <F2 number> to <F3 number> s; warm bwd from <F2> to
  <F3> s.
- variPEPS-compare benchmark (single_site D=2 χ=16, CPU): Tenax now
  fits inside the 30-min subprocess budget — closes Definition of
  Done #4 from PR #415.

Spec: `docs/plans/2026-05-09-f3-fused-backward-jit.md`
Diagnosis: `docs/plans/2026-05-09-ipeps-ad-jit-cost-diagnosis.md` §F3

## Test plan

- [x] `uv run pytest -m core -x`
- [x] `uv run pytest tests/test_ipeps_ad_adjoint_methods.py -v`
- [x] `uv run pytest tests/test_ipeps_ad_f3_fused_bwd.py -v`
- [x] `uv run pytest tests/test_ipeps_excitations.py::TestOptimizeGsAd -v`
- [x] microbench cold/warm numbers recorded above
- [x] `published_results/STATUS.md` updated with chi=16 single_site
      Tenax data
EOF
)"
```

**Step 3: Enable auto-merge**

```bash
gh pr merge --squash --delete-branch --auto
```

(Per CLAUDE.md.)

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pytree leaf order in `vjp_env_fn` differs from F2 `_jit_apply_Jt`, gradient mismatch | Medium | Task 4 Step 3 catches this. The treedef capture in `_cached["env_treedef"]` is shared with F2, so tuple roundtripping is identical — but verify. |
| `lax.while_loop` carry shape inferred wrong because `lam` pytree depends on `env_leaves` shape | Low | The carry is initialised at `dE_denv` which has the same pytree shape as `env_leaves` by construction (energy VJP cotangent shape). `lax.while_loop` infers shape from the init carry. |
| In-loop divergence rate is materially different vs. the F2 Python loop because of float promotion in `prev_diff` | Low | Both paths compute `sum(jnp.linalg.norm(n - p))`. The only difference is the F3 path keeps it as a JAX scalar instead of `float()`-converting — same value, same comparison. |
| Compile time grows because the fused JIT body is larger than each F2 helper, even though there are fewer of them | Low | The diagnosis doc empirical attribution showed the F2 trio totals 7 s; the fused body should land at most around the larger of those (`_jit_apply_Jt` at 2.6 s) plus a small VJP-of-VJP overhead. If cold time goes up, revert. |
| `bool(jax.device_get(diverged))` blocks dispatch on a tiny scalar each backward call | Negligible | One `device_get` per backward is < 1 ms. Compare to the seconds-per-step regime we're in. |
| The `arnoldi_precheck` matvec still needs `_jit_apply_Jt` separately and that JIT now compiles even when unused | Low | Already the case (`_jit_apply_Jt` is defined unconditionally). When `arnoldi_precheck=False` and `adjoint_method="fixed_point"`, `_jit_apply_Jt` is never called → never traced. Verified by `jax.log_compiles()` later if needed. |
| Future maintainers don't realise the fused JIT closes over `mutables["gate"]` and assume `gate` can change without recompile | Low | Same contract as F2; documented in the function docstring and in this plan §"Design notes / mutables dict" (implicit by reuse of existing pattern). |

## Estimate

- Task 0 (worktree + branch): 5 min
- Task 1 (microbench harness + F2 baseline run): 30 min (run is the
  long part — ~25 s cold + ~5 s warm)
- Task 2 (write parity test, baseline pass): 30 min
- Task 3 (write fused JIT body): 1 hour
- Task 4 (wire f_bwd, get test green): 1 hour (likely some
  iteration on pytree-leaf ordering)
- Task 5 (regression suite): 30 min wall-clock (mostly waiting)
- Task 6 (re-run microbench): 10 min
- Task 7 (variPEPS-compare benchmark): 1 hour wall-clock
- Task 8 (verification gate): 15 min
- Task 9 (PR): 10 min

End-to-end: ~half a day if it goes smoothly, full day with one
debug iteration on Task 4. The main unknown is the pytree-ordering
risk in Task 4 Step 3.

## Definition of done

1. `uv run pytest -m core -x` clean (≥ 789 passed, no new fails).
2. `uv run pytest tests/test_ipeps_ad_f3_fused_bwd.py -v` PASS.
3. `uv run pytest tests/test_ipeps_ad_adjoint_methods.py -v` PASS
   (F2 regression).
4. `uv run pytest tests/test_ipeps_excitations.py::TestOptimizeGsAd -v`
   PASS.
5. Microbench shows cold-compile time at D=2 χ=8 drops by ≥ 30%
   vs. F2 baseline; warm bwd by ≥ 10%.
6. `python -m benchmarks.varipeps_compare.compare --device cpu`
   produces a Tenax JSON for `single_site D=2 chi=16` inside the
   30-min subprocess budget.
7. `published_results/STATUS.md` updated with F3 numbers.
8. PR opened, auto-merge enabled with `--squash --delete-branch
   --auto`.
