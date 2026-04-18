# Issue #328 Option B: Joint Multi-Site Metric Preconditioner

> **Status:** SUPERSEDED — #328 resolved via Option C (shared-C4v + implicit AD, PR #332). Joint metric not needed for C4v path. Non-C4v 2-site must use implicit AD (`gs_implicit_ad=True`); explicit is non-variational.

**Goal:** Close the remaining 2-site gs_c4v=False AD drift (post-#330 bench: χ=16 still lands at E≈-1.16 below physical -0.669) by replacing the per-site block-diagonal metric with a **joint 2-site QGT matvec** that includes cross-site coupling ∂A↔∂B through the shared CTM environment. Acts via GMRES inside the L-BFGS / CG metric-preconditioned direction.

**Architecture:** Two-phase approach. Phase 1 uses JAX forward-mode AD through a `norm_squared_2site(A, B, env_A, env_B)` scalar function to get the joint matvec for free via `jax.jvp(jax.grad(...))` — zero new CTM contractions, ~80 lines. Phase 2 (only if Phase 1 has unacceptable overhead) hand-writes the 4-block contraction by generalizing `norm_environment_matvec` to a "double-hole" environment. Phase 1 alone should be sufficient to close the bench.

**Tech Stack:** JAX (`jvp`, `grad`, `scipy.sparse.linalg.gmres`), existing `compute_energy_ctm_tensor_2site` infrastructure in `_ctm_tensor_energy.py`, `_metric_precond.py`.

---

## Design

### The math

For a 2-site iPEPS with site tensors A at (0,0), B at (1,0) and shared CTM environments (env_A, env_B), the quantum geometric tensor (QGT) is a 2×2 block operator:

```
N = [ ⟨∂A|∂A⟩  ⟨∂A|∂B⟩ ]
    [ ⟨∂B|∂A⟩  ⟨∂B|∂B⟩ ]
```

Its action on a direction `(v_A, v_B)` is:

```
(N v)_A = ⟨∂A|∂A⟩ v_A + ⟨∂A|∂B⟩ v_B
(N v)_B = ⟨∂B|∂A⟩ v_A + ⟨∂B|∂B⟩ v_B
```

The existing `precondition_gradient_multisite` computes only the diagonal blocks (the A-only and B-only metrics). The flat direction `A → λA, B → (1/λ)B` lies in `null(N_diag)` but **not** in `null(N)` — the cross block `⟨∂A|∂B⟩` kills it. That's why the current optimizer follows it into non-variational territory, and that's what Option B fixes.

### The AD identity

For `N(A, B) := ⟨ψ(A,B) | ψ(A,B)⟩` (a real scalar function of the two tensors), the QGT is the Hessian at fixed conjugate variables:

```
⟨∂X|∂Y⟩ = ∂²N / ∂X̄ ∂Y
```

In JAX, for real tensors, we compute this as a Jacobian-vector product (JVP) of the gradient:

```python
def grad_N(A, B):
    return jax.grad(norm_squared, argnums=(0, 1))(A, B)

# (N v)_A, (N v)_B = JVP of grad_N in direction (v_A, v_B)
_, (NvA, NvB) = jax.jvp(grad_N, (A, B), (v_A, v_B))
```

This is a **single JVP** and costs ~2× a single grad of `norm_squared`. No new contraction code.

### Why this is cheap

- `norm_squared(A, B, env_A, env_B)` is ~the same cost as `compute_energy_ctm_tensor_2site` (reuse it with `gate = identity`, or write a trimmed version that skips the gate leg). One forward contraction.
- `grad(norm_squared)` = 2× forward cost (backward pass).
- `jvp(grad)` = ~2-4× cost of grad = ~4-8× cost of the forward norm contraction.
- GMRES inside precondition typically needs 5-20 matvecs per solve; total joint-precondition cost ~50-200× forward contraction per optimizer step.
- Existing block-diagonal precondition is roughly half this. So 2-3× overhead per precondition call.

The issue's estimate was "~4× per-matvec relative to block-diagonal" — consistent.

### Config knob

Add `gs_joint_metric: bool = False` to `iPEPSConfig`. Defaulting to `False` keeps existing behavior (per-site block-diagonal metric). Users opt in for 2-site gs_c4v=False cases that need the joint metric. Once the bench shows it's stable and Pareto-improving, flip the default to `True`.

Register the knob in `tuning/registry.py` as a `BOOLEAN` entry.

---

## Task 1: Add `norm_squared_2site` scalar function

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_energy.py` (or create a new helper in `_metric_precond.py`, whichever keeps related code colocated — prefer `_metric_precond.py` to avoid polluting the energy module)

**Step 1: Write the helper**

In `_metric_precond.py`, add:

```python
def _norm_squared_2site(
    A: Tensor,
    B: Tensor,
    env_A: "CTMTensorEnv",
    env_B: "CTMTensorEnv",
) -> jax.Array:
    """Compute ``⟨ψ(A,B) | ψ(A,B)⟩`` for the 2-site checkerboard iPEPS
    with CTM environments ``env_A``, ``env_B``.

    This reuses ``compute_energy_ctm_tensor_2site`` with an identity
    2-site gate — ``<ψ|I|ψ> = <ψ|ψ>``.  Real-valued and differentiable
    in A and B.
    """
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_2site

    d = A.todense().shape[-1]
    identity_gate = jnp.eye(d * d, dtype=A.todense().dtype).reshape(d, d, d, d)
    # compute_energy_ctm_tensor_2site returns <ψ|H|ψ> / <ψ|ψ>; we need
    # the denominator.  The function's internal trace of the 2-site RDM
    # against `identity_gate` gives <ψ|I|ψ> / <ψ|ψ> = 1, so this approach
    # doesn't directly expose norm². Instead, read the intermediate
    # numerator by calling the (private) RDM contraction, or just write
    # the contraction inline here — whichever is cleaner.
    raise NotImplementedError("see Step 2")
```

**NOTE on implementation path:** Read `compute_energy_ctm_tensor_2site` first and decide whether (a) it exposes an intermediate `<ψ|ψ>` we can grab, (b) we can pass an identity-gate AND an un-normalized variant, or (c) we write a direct ~15-line RDM contraction here. Option (c) is most likely cleanest and the contraction is standard (the same 10-tensor network as the energy, minus the gate and minus the normalization).

**Step 2: Implement the contraction**

Hand-write the 2-site unnormalized norm contraction. The network is: for the `(0,0)` / `(1,0)` checkerboard cell, contract `A`, `A*`, `B`, `B*`, the two per-site corner-edge environments (`env_A`, `env_B`), and fuse the two sites through their shared horizontal virtual bond.

```python
def _norm_squared_2site(A, B, env_A, env_B):
    # TODO: write the 10-tensor contraction.  Cross-reference
    # src/tenax/algorithms/_ctm_tensor_energy.py::compute_energy_ctm_
    # tensor_2site for the leg labels and ordering, but return the
    # un-traced <ψ|ψ> before the gate and before the 1/<ψ|ψ> division.
    ...
```

**Step 3: Test: norm_squared is positive and matches a reference**

Add `tests/test_metric_precond.py::TestNormSquared2Site::test_matches_reference`:

```python
def test_matches_reference_via_energy():
    """<ψ|ψ> from _norm_squared_2site should match the denominator
    implicit in compute_energy_ctm_tensor_2site when called with a
    non-normalizing code path."""
    # set up small random A, B, converge CTM, compute norm²
    # via _norm_squared_2site.  Separately, compute <ψ|H|ψ> and
    # <ψ|H|ψ>/<ψ|ψ> via the existing energy function; verify the
    # ratio implied by the two matches our standalone norm².
```

**Step 4: Commit**

```bash
git add src/tenax/algorithms/_metric_precond.py tests/test_metric_precond.py
git commit -m "feat(ipeps): add _norm_squared_2site helper for joint metric (#328)"
```

---

## Task 2: JVP-based joint metric matvec

**Files:**
- Modify: `src/tenax/algorithms/_metric_precond.py`

**Step 1: Add the joint matvec**

```python
def norm_environment_matvec_joint_2site(
    A: Tensor,
    B: Tensor,
    env_A: "CTMTensorEnv",
    env_B: "CTMTensorEnv",
    v_A: Tensor,
    v_B: Tensor,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Joint 2-site QGT matvec:

    ``(N v)_A = ⟨∂A|∂A⟩ v_A + ⟨∂A|∂B⟩ v_B``
    ``(N v)_B = ⟨∂B|∂A⟩ v_A + ⟨∂B|∂B⟩ v_B``

    Computed via JVP through ``jax.grad(_norm_squared_2site)`` — no
    new explicit contractions needed.
    """
    def grad_N(A_, B_):
        return jax.grad(_norm_squared_2site, argnums=(0, 1))(A_, B_, env_A, env_B)

    _, (NvA, NvB) = jax.jvp(grad_N, (A, B), (v_A, v_B))
    return NvA.todense(), NvB.todense()
```

**Step 2: Test Hermiticity**

```python
def test_joint_matvec_hermitian():
    """⟨v1, N v2⟩ == ⟨v2, N v1⟩ for any v1, v2 (conjugate transpose)."""
    # Build A, B, env_A, env_B; draw two random direction pairs;
    # compute the inner products and assert they match to ~1e-10.
```

**Step 3: Test positive definiteness on random directions**

```python
def test_joint_matvec_positive():
    """⟨v, N v⟩ >= 0 (N is PSD for valid physical state)."""
```

**Step 4: Test cancellation of the flat direction**

```python
def test_joint_matvec_kills_flat_direction():
    """For the direction v_A = A, v_B = -B (infinitesimal A→λA, B→(1/λ)B),
    the **sum** ⟨v, N v⟩ should be much smaller than on a random direction
    — the joint metric knows this is a gauge mode."""
```

This is the critical behavioral test — if it fails, Option B doesn't actually kill the flat direction and the approach is wrong.

**Step 5: Commit**

```bash
git commit -m "feat(ipeps): joint 2-site QGT matvec via jvp(grad(norm²)) (#328)"
```

---

## Task 3: GMRES solver wrapping the joint matvec

**Files:**
- Modify: `src/tenax/algorithms/_metric_precond.py`

**Step 1: Add `precondition_gradient_joint_2site`**

```python
def precondition_gradient_joint_2site(
    A: Tensor, B: Tensor,
    env_A: "CTMTensorEnv", env_B: "CTMTensorEnv",
    grad_A: Tensor, grad_B: Tensor,
    delta: float,
    config: "iPEPSConfig",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve ``(N + δI) [z_A, z_B] = [g_A, g_B]`` via GMRES, where N is
    the joint 2-site QGT matvec."""
    from jax.scipy.sparse.linalg import gmres

    n_A = grad_A.todense().size
    n_B = grad_B.todense().size
    shape_A = grad_A.todense().shape
    shape_B = grad_B.todense().shape

    def op(v_flat):
        v_A_flat = v_flat[:n_A]
        v_B_flat = v_flat[n_A:]
        v_A_dense = v_A_flat.reshape(shape_A)
        v_B_dense = v_B_flat.reshape(shape_B)
        # _wrap_tensor needed because jvp needs a Tensor-typed tangent
        from tenax.algorithms.ad_utils import _wrap_tensor
        v_A_t = _wrap_tensor(v_A_dense, A)
        v_B_t = _wrap_tensor(v_B_dense, B)
        NvA, NvB = norm_environment_matvec_joint_2site(
            A, B, env_A, env_B, v_A_t, v_B_t
        )
        out = jnp.concatenate([NvA.reshape(-1), NvB.reshape(-1)])
        return out + delta * v_flat  # Tikhonov regularization

    b = jnp.concatenate([grad_A.todense().reshape(-1), grad_B.todense().reshape(-1)])
    z_flat, info = gmres(
        op, b,
        tol=config.metric_gmres_tol,
        maxiter=config.metric_gmres_maxiter,
    )
    z_A = z_flat[:n_A].reshape(shape_A)
    z_B = z_flat[n_A:].reshape(shape_B)
    return z_A, z_B
```

**Step 2: Test that joint precond recovers the gradient at large δ**

At large `delta`, `(N + δI)⁻¹ g ≈ g / δ` — the solve should give back a scaled copy of the gradient.

**Step 3: Commit**

```bash
git commit -m "feat(ipeps): GMRES wrapper for joint 2-site QGT precond (#328)"
```

---

## Task 4: Config knob

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py`
- Modify: `src/tenax/tuning/registry.py`

**Step 1: Add the knob**

In `iPEPSConfig`:

```python
gs_joint_metric: bool = False  # issue #328: joint 2-site QGT in metric-precond
```

Only active when `gs_metric_precond=True` AND `unit_cell="2site"` AND `gs_c4v=False`. Document this clearly in the docstring.

In `tuning/registry.py`, register as a new `BOOLEAN` entry under `iPEPSConfig`:

```python
RegistryEntry(
    path="iPEPSConfig.gs_joint_metric",
    scale=Scale.BOOLEAN,
    description="Use joint 2-site QGT (cross-site coupling) in metric precond. "
                "Only active for gs_metric_precond=True, unit_cell='2site', "
                "gs_c4v=False. Closes #328 joint-flat-direction drift.",
    applies_when={
        "iPEPSConfig.gs_metric_precond": True,
        "iPEPSConfig.unit_cell": "2site",
        "iPEPSConfig.gs_c4v": False,
    },
)
```

**Step 2: Drift check**

```bash
uv run pytest tests/test_tuning_registry_drift.py -x -q
```

**Step 3: Commit**

```bash
git commit -m "feat(ipeps): register gs_joint_metric config knob (#328)"
```

---

## Task 5: Wire into the 2-site optimizer

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py::_optimize_gs_ad_tensor_2site`

**Step 1: Switch CG metric-precond path**

At the existing callsite in the CG branch (~line 1470 per the pre-#330 survey; adjust for current line numbers):

```python
if config.gs_metric_precond and not use_c4v:
    if config.gs_joint_metric:
        from tenax.algorithms._metric_precond import precondition_gradient_joint_2site
        z_A_dense, z_B_dense = precondition_gradient_joint_2site(
            A, B, env_A_m, env_B_m, A_g, B_g, delta_metric, config
        )
        z = (_wrap_tensor(z_A_dense, A_g), _wrap_tensor(z_B_dense, B_g))
    else:
        # existing per-site block-diagonal path
        z_dict = precondition_gradient_multisite(...)
        z = (_wrap_tensor(z_dict[(0, 0)], A_g), _wrap_tensor(z_dict[(1, 0)], B_g))
```

**Step 2: Same switch in the L-BFGS metric-precond branch**

The L-BFGS path uses `h0_matvec` that calls `precondition_gradient_multisite` per site. Replace with a single call to the joint matvec that returns the concatenated flat vector.

**Step 3: Smoke test**

```bash
uv run pytest tests/test_ipeps.py::TestOptimizeGsAd2Site::test_2site_ad_runs \
    tests/test_ipeps.py::TestOptimizeGsAd2Site::test_2site_ad_c4v_runs \
    tests/test_ipeps.py::TestOptimizeGsAd2Site::test_2site_noc4v_ad_stays_variational_issue_328 \
    -xvs --no-cov
```

All three should still pass with `gs_joint_metric=False` (default).

**Step 4: Commit**

```bash
git commit -m "feat(ipeps): wire gs_joint_metric into 2-site AD optimizer (#328)"
```

---

## Task 6: Regression test — 2-site Heisenberg χ=16 with joint metric

**Files:**
- Modify: `tests/test_ipeps.py`

**Step 1: New test**

```python
def test_2site_noc4v_joint_metric_closes_drift_issue_328(self, heisenberg_gate):
    """With gs_joint_metric=True, 2-site gs_c4v=False Heisenberg at D=2
    χ=8 should land close to the physical ground state -0.6694 instead
    of drifting below it.

    Pre-Option-B bench (post-#330): E ≈ -1.16 at χ=16 (still below
    physical).  With joint metric the cross-site coupling ⟨∂A|∂B⟩
    should cancel the joint-rescaling flat direction that caused the
    drift.
    """
    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=8, max_iter=30, min_iter=10),
        gs_num_steps=20,
        gs_optimizer="lbfgs",
        gs_line_search=True,
        gs_metric_precond=True,
        gs_joint_metric=True,   # issue #328 opt-in
        unit_cell="2site",
        gs_c4v=False,
    )
    _, _, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
    # Variational lower bound: physical ground state is -0.6694.
    # We assert the energy is close to physical, not below.
    # Tolerance 0.05 absorbs finite-chi + finite-D errors.
    assert E_gs > -0.72, f"E={E_gs:.4f} still below physical -0.669"
    assert E_gs < -0.50, f"E={E_gs:.4f} not reasonably optimized"
```

**Step 2: Run**

```bash
uv run pytest tests/test_ipeps.py::TestOptimizeGsAd2Site::test_2site_noc4v_joint_metric_closes_drift_issue_328 -xvs --no-cov
```

**Step 3: If it fails**, run the full bench script (`/tmp/bench_issue_328.py` with `gs_joint_metric=True` added to each non-C4v row) to see whether the joint metric helps at all or whether the drift has another root cause. Do **not** commit a passing test that's only passing by luck — either fix the underlying issue or escalate to Option C (full Riemannian L-BFGS).

**Step 4: Commit**

```bash
git commit -m "test(ipeps): joint metric closes 2-site drift at chi=8 (#328)"
```

---

## Task 7: Re-run the post-#330 bench with joint metric

**Files:**
- Create: `/tmp/bench_issue_328_optb.py` (copy of the existing `/tmp/bench_issue_328.py` with `gs_joint_metric=True` flipped on for all non-C4v rows, run against the joint-metric build)

**Step 1: Run**

```bash
JAX_PLATFORMS=cpu uv run python /tmp/bench_issue_328_optb.py > /tmp/bench_issue_328_optb.log 2>&1 &
```

**Step 2: Compare**

Build a side-by-side table of:
- Pre-fix (issue body, original bench)
- Post-#330 (tangent projection only, current bench)
- Post-Option-B (joint metric)

For each of (c4v control, noc4v-chi8, noc4v-chi16-armijo, noc4v-chi16-hz, noc4v-chi16-implicit).

**Step 3: Decide**

- If joint metric closes the drift: flip `gs_joint_metric` default to `True` and close #328. Keep Option C as a far-future research item.
- If joint metric partially helps: keep it as an opt-in knob, update #328 recommending joint metric + decaying L-BFGS restart, and open a new issue for Option C.
- If joint metric doesn't help: the drift has another source. Re-investigate the CTM forward at finite χ for 2-site gs_c4v=False.

---

## Task 8: Open PR

```bash
gh pr create --title "feat(ipeps): joint 2-site QGT metric preconditioner (#328 Option B)" \
    --body "$(cat docs/plans/2026-04-15-issue-328-option-b-joint-metric.md | head -50)..."
gh pr merge <num> --squash --delete-branch --auto
```

---

## Scoping notes

- **Strictly 2-site gs_c4v=False non-C4v.** The 1-site path doesn't need a cross block. The C4v path enforces A and B to be spin-rotated copies of each other, so the cross block is already implicit in the parameterization.
- **Don't touch the 1-site `precondition_gradient` / `norm_environment_matvec`.** Those are in production use.
- **Don't couple Phase 1 (AD-based) and Phase 2 (hand-contracted) in the same PR.** Ship AD-based as an opt-in knob first, bench it, and only hand-write the contraction if profiling shows the JVP overhead is prohibitive.
- **Complex-tensor support**: once `_norm_squared_2site` is written, it must work for `complex128` tensors too (fermionic iPEPS, chiral models). Use `jnp.vdot`-style conjugation where applicable; test with a complex-tensor regression like PR #331 did.

## DRY / YAGNI

- **Do not** write a parallel "double-hole environment" contraction in Phase 1. The whole point of the JVP-through-grad approach is that JAX synthesizes it from the scalar norm². Only hand-write it if benchmarks justify it.
- **Do not** extend to ≥3-site unit cells in the same PR. That's a separate feature — depends on this one landing cleanly first.
- **Do not** change the default `gs_joint_metric` to `True` in the initial PR. Opt-in until the bench matrix confirms it's Pareto-stable.

## Out of scope

- Option C (Riemannian L-BFGS with parallel transport of curvature).
- Joint metric for ≥3-site unit cells.
- Joint metric for the fermionic CTM fallback path in `fermionic_ipeps.py`.
- Making `gs_joint_metric` work with `use_c4v=True` (not needed — C4v already fixes the flat direction at the parameterization level).
