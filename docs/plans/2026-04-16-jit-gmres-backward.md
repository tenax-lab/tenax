# JIT GMRES Backward Implementation Plan

> **Status:** COMPLETED (mechanically) — implemented but `jit_ctm` path removed in PR #337. Non-C4v 2-site ρ(J^T) ≫ 1 means GMRES doesn't help. #328 resolved via shared-C4v + implicit AD.

**Goal:** JIT-compile the GMRES backward solve in `_ctm_tensor_converge_bwd` so the entire adjoint system fuses into a single XLA program, matching variPEPS performance.

**Architecture:** When `jit_ctm=True` and `ad_backward_method="gmres"`, wrap the GMRES solve + site projection in `@jax.jit`. The VJP closures from `jax.vjp` are captured and traced symbolically. Tikhonov damping stabilizes the `(I - J^T + τI)` operator. Arnoldi precheck is skipped in JIT mode.

**Tech Stack:** JAX (`jax.scipy.sparse.linalg.gmres`, `jax.vjp`, `@jax.jit`)

---

### Task 1: Wire adjoint config fields through config_tuple

The backward function receives config as a tuple via `_config_to_tuple` /
`_config_from_tuple`.  Currently `adjoint_maxiter`, `adjoint_tol`, and
`adjoint_tikhonov` are NOT serialized — the backward can't see them.

**Files:**
- Modify: `src/tenax/algorithms/ad_utils.py:537-594` (`_config_to_tuple`, `_config_from_tuple`)
- Test: `tests/test_ad_utils.py` (existing config round-trip tests)

**Step 1: Write the failing test**

In `tests/test_ad_utils.py`, add at the end of the file (or in a new class):

```python
class TestConfigTupleAdjointFields:
    def test_adjoint_fields_round_trip(self):
        """adjoint_maxiter, adjoint_tol, adjoint_tikhonov survive config_tuple round-trip."""
        config = CTMConfig(
            chi=8, max_iter=50, conv_tol=1e-7,
            adjoint_maxiter=200, adjoint_tol=1e-9, adjoint_tikhonov=1e-4,
        )
        ct = _config_to_tuple(config)
        rebuilt = _config_from_tuple(ct)
        assert rebuilt.adjoint_maxiter == 200
        assert rebuilt.adjoint_tol == pytest.approx(1e-9)
        assert rebuilt.adjoint_tikhonov == pytest.approx(1e-4)

    def test_adjoint_fields_default_round_trip(self):
        """Default adjoint fields survive round-trip."""
        config = CTMConfig(chi=4)
        ct = _config_to_tuple(config)
        rebuilt = _config_from_tuple(ct)
        assert rebuilt.adjoint_maxiter == 50
        assert rebuilt.adjoint_tol == pytest.approx(1e-8)
        assert rebuilt.adjoint_tikhonov == pytest.approx(1e-6)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ad_utils.py::TestConfigTupleAdjointFields -v`
Expected: FAIL — `_config_from_tuple` doesn't set adjoint fields.

**Step 3: Implement — extend `_config_to_tuple` and `_config_from_tuple`**

In `_config_to_tuple` (line ~537), append three fields after index 13:

```python
    # index 14: adjoint_maxiter (int)
    getattr(config, "adjoint_maxiter", 50),
    # index 15: adjoint_tol (float)
    getattr(config, "adjoint_tol", 1e-8),
    # index 16: adjoint_tikhonov (float)
    getattr(config, "adjoint_tikhonov", 1e-6),
```

In `_config_from_tuple` (line ~559), decode after index 13:

```python
    adjoint_maxiter = int(config_tuple[14]) if len(config_tuple) > 14 else 50
    adjoint_tol = float(config_tuple[15]) if len(config_tuple) > 15 else 1e-8
    adjoint_tikhonov = float(config_tuple[16]) if len(config_tuple) > 16 else 1e-6
```

And add to the CTMConfig constructor call:

```python
    adjoint_maxiter=adjoint_maxiter,
    adjoint_tol=adjoint_tol,
    adjoint_tikhonov=adjoint_tikhonov,
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ad_utils.py::TestConfigTupleAdjointFields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ad_utils.py tests/test_ad_utils.py
git commit -m "feat(ad): wire adjoint_maxiter/tol/tikhonov through config_tuple"
```

---

### Task 2: Implement JIT GMRES backward path

**Files:**
- Modify: `src/tenax/algorithms/ad_utils.py:1186-1375` (`_ctm_tensor_converge_bwd`)

**Step 1: Write the failing test**

In `tests/test_ad_utils.py`, add to the existing `TestGMRESBackward` class
(or create a new class after it):

```python
class TestJitGMRESBackward:
    def test_jit_gmres_backward_finite_gradient(self):
        """JIT GMRES backward produces finite, nonzero gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        config = CTMConfig(
            chi=4, max_iter=10, conv_tol=1e-6,
            ad_backward_method="gmres",
            jit_ctm=True,
            adjoint_maxiter=50,
            adjoint_tol=1e-6,
            adjoint_tikhonov=1e-6,
            adjoint_arnoldi_precheck=False,
        )
        config_tuple = _config_to_tuple(config)
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def energy_fn(A_in):
            A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
            env_leaves = ctm_tensor_converge(
                {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, config_tuple
            )
            env = jax.tree.unflatten(
                jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                list(env_leaves),
            )
            return compute_energy_ctm_tensor(A_norm, env, gate)

        grad = jax.grad(energy_fn)(A)
        assert jnp.all(jnp.isfinite(grad.todense())), "JIT GMRES: NaN/Inf"
        assert grad.norm() > 1e-15, "JIT GMRES: gradient is all zeros"

    def test_jit_gmres_agrees_with_eager_gmres(self):
        """JIT and eager GMRES backward produce similar gradients."""
        A = _make_dense_tensor(jax.random.PRNGKey(55))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with(jit_ctm: bool):
            config = CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-6,
                ad_backward_method="gmres",
                jit_ctm=jit_ctm,
                adjoint_maxiter=50,
                adjoint_tol=1e-6,
                adjoint_tikhonov=1e-6,
                adjoint_arnoldi_precheck=False,
            )
            ct = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, ct
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_jit = _grad_with(jit_ctm=True)
        grad_eager = _grad_with(jit_ctm=False)
        diff = float(jnp.max(jnp.abs(grad_jit.todense() - grad_eager.todense())))
        assert diff < 1e-4, f"JIT vs eager GMRES gradient diff = {diff}"

    def test_jit_gmres_tikhonov_damps_operator(self):
        """Tikhonov damping should change the gradient (nonzero effect)."""
        A = _make_dense_tensor(jax.random.PRNGKey(42))
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)

        def _grad_with_tikhonov(tau):
            config = CTMConfig(
                chi=4, max_iter=10, conv_tol=1e-6,
                ad_backward_method="gmres",
                jit_ctm=True,
                adjoint_maxiter=50,
                adjoint_tol=1e-6,
                adjoint_tikhonov=tau,
                adjoint_arnoldi_precheck=False,
            )
            ct = _config_to_tuple(config)

            def energy_fn(A_in):
                A_norm = A_in * (1.0 / (A_in.norm() + 1e-10))
                env_leaves = ctm_tensor_converge(
                    {(0, 0): A_norm}, None, SINGLE_SITE_NEIGHBORS, ct
                )
                env = jax.tree.unflatten(
                    jax.tree.structure(initialize_ctm_tensor_env(A_in, 4)),
                    list(env_leaves),
                )
                return compute_energy_ctm_tensor(A_norm, env, gate)

            return jax.grad(energy_fn)(A)

        grad_small = _grad_with_tikhonov(1e-10)
        grad_large = _grad_with_tikhonov(1e-1)
        diff = float(jnp.max(jnp.abs(grad_small.todense() - grad_large.todense())))
        assert diff > 1e-6, f"Tikhonov damping had no effect: diff = {diff}"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ad_utils.py::TestJitGMRESBackward -v`
Expected: FAIL — JIT GMRES path not implemented yet.

**Step 3: Implement the JIT GMRES backward**

In `_ctm_tensor_converge_bwd`, replace lines 1265-1302 (the `max_fp_iter`
assignment through the end of the GMRES branch) with:

```python
    jit_bwd = getattr(config, "jit_ctm", False)
    adjoint_maxiter = getattr(config, "adjoint_maxiter", 50)
    adjoint_tol = getattr(config, "adjoint_tol", 1e-8)
    adjoint_tikhonov = getattr(config, "adjoint_tikhonov", 1e-6)
    max_fp_iter = adjoint_maxiter if config.ad_backward_method == "gmres" else min(config.max_iter, 50)

    # --- Arnoldi spectral-radius precheck (skip in JIT mode) ---
    if not jit_bwd and getattr(config, "adjoint_arnoldi_precheck", True):
        ... (keep existing Arnoldi block unchanged)

    if config.ad_backward_method == "gmres" and jit_bwd:
        # --- JIT GMRES path ---
        # Wrap the entire GMRES solve + site projection in @jax.jit.
        # The vjp closures are captured and traced symbolically, so
        # the GMRES while_loop + all vjp_env_fn applications compile
        # into a single XLA program.
        tikhonov = adjoint_tikhonov

        @jax.jit
        def _jit_gmres_solve(env_leaves_in, site_leaves_in, g_in):
            if use_sigma:
                def _step(s, e):
                    e_ref = tuple(jax.lax.stop_gradient(x) for x in e)
                    return _ctm_tensor_step_multisite(
                        s, e, neighbors, config.chi, config.renormalize,
                        config.projector_method, site_treedefs, env_treedef,
                        n_env_per_site, sigma_gauge_ref_leaves=e_ref,
                        projector_backward=getattr(config, "projector_backward", "auto"),
                    )
            else:
                def _step(s, e):
                    return _ctm_tensor_step_multisite(
                        s, e, neighbors, config.chi, config.renormalize,
                        config.projector_method, site_treedefs, env_treedef,
                        n_env_per_site, skip_gauge=True,
                        projector_backward=getattr(config, "projector_backward", "auto"),
                    )

            _, vjp_e = jax.vjp(lambda e: _step(site_leaves_in, e), env_leaves_in)
            _, vjp_s = jax.vjp(lambda s: _step(s, env_leaves_in), site_leaves_in)

            def apply_I_minus_Jt_damped(v):
                Jt_v = vjp_e(v)[0]
                return tuple(
                    vi - ji + tikhonov * vi for vi, ji in zip(v, Jt_v)
                )

            lam, info = jax_gmres(
                apply_I_minus_Jt_damped, g_in, x0=g_in,
                tol=adjoint_tol, maxiter=adjoint_maxiter,
            )
            d_site = vjp_s(lam)[0]
            return d_site, info

        d_site_leaves, gmres_info = _jit_gmres_solve(env_leaves, site_leaves, g)
        gmres_info_val = int(gmres_info)
        if gmres_info_val != 0:
            _logger.warning("JIT GMRES backward did not converge (info=%d)", gmres_info_val)

    elif config.ad_backward_method == "gmres":
        # --- Eager GMRES path (original) ---
        ... (keep existing eager GMRES code unchanged)
    else:
        # --- Neumann path (unchanged) ---
        ...
```

Key design notes for the implementor:
- `_jit_gmres_solve` takes explicit `env_leaves_in`, `site_leaves_in`,
  `g_in` as arguments (JAX arrays), NOT closures.
- Inside the JIT, `jax.vjp` is called fresh — this traces the step
  function symbolically and produces JIT-compatible VJP closures.
- `neighbors`, `config.*`, `site_treedefs`, `env_treedef`, `n_env_per_site`
  are captured from the enclosing scope as static/hashable values.
- `use_sigma` is a Python bool captured from outer scope — JIT traces
  only the selected branch.
- The Tikhonov term `+ tikhonov * vi` adds `τI` to `(I - J^T)`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ad_utils.py::TestJitGMRESBackward -v`
Expected: PASS

Also run existing GMRES tests to verify no regression:

Run: `uv run pytest tests/test_ad_utils.py::TestGMRESBackward -v`
Expected: PASS (eager path unchanged)

**Step 5: Commit**

```bash
git add src/tenax/algorithms/ad_utils.py tests/test_ad_utils.py
git commit -m "feat(ad): JIT-compiled GMRES backward for CTM implicit diff"
```

---

### Task 3: Integration test — 2-site non-C4v with JIT GMRES

This is the actual #328 scenario. Slow test (minutes), marks as `slow`.

**Files:**
- Modify: `tests/test_ipeps.py` (add to `TestIPEPSTensor2Site` or new class)

**Step 1: Write the test**

```python
@pytest.mark.slow
def test_2site_noc4v_jit_gmres_is_variational(self, heisenberg_gate):
    """Non-C4v 2-site with JIT GMRES backward stays variational (#328)."""
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=8,
            max_iter=30,
            conv_tol=1e-7,
            min_iter=5,
            ad_backward_method="gmres",
            jit_ctm=True,
            adjoint_maxiter=200,
            adjoint_tol=1e-7,
            adjoint_tikhonov=1e-6,
            forward_gauge="phase",
            adjoint_arnoldi_precheck=False,
        ),
        gs_num_steps=15,
        gs_optimizer="lbfgs",
        unit_cell="2site",
        gs_c4v=False,
        gs_explicit_ad=False,
        su_init=False,
        gs_verbose=True,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        result = optimize_gs_ad(heisenberg_gate, None, config)
    E = float(result.energy)
    # Heisenberg exact: -0.6694. Must be variational (above exact).
    assert E > -0.70, f"Non-variational energy {E} (below exact -0.6694)"
    # Should descend to a reasonable value
    assert E < -0.30, f"Energy {E} stuck near initial random state"
```

**Step 2: Run to verify behavior**

Run: `uv run pytest tests/test_ipeps.py -k "test_2site_noc4v_jit_gmres" -v -s`
Expected: either PASS (variational) or informative failure.

**Step 3: Commit**

```bash
git add tests/test_ipeps.py
git commit -m "test(ipeps): integration test for 2-site non-C4v JIT GMRES (#328)"
```

---

### Task 4: Benchmark script

Write a standalone benchmark that reproduces the #328 scenario with the
JIT GMRES path and compares wall time + energy against the eager baseline.

**Files:**
- Create: `/tmp/bench_328_jit_gmres.py`

**Step 1: Write the benchmark**

```python
"""Benchmark: JIT GMRES backward vs eager for 2-site non-C4v (#328)."""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_optimize import optimize_gs_ad
from tenax.models.quantum import heisenberg_gate

gate = heisenberg_gate(d_phys=2)

configs = {
    "eager_neumann": iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=16, max_iter=30, conv_tol=1e-7, min_iter=5,
            ad_backward_method="vjp", jit_ctm=False,
            forward_gauge="phase",
        ),
        gs_num_steps=30, gs_optimizer="lbfgs", unit_cell="2site",
        gs_c4v=False, gs_explicit_ad=False, su_init=False, gs_verbose=True,
    ),
    "jit_gmres": iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(
            chi=16, max_iter=30, conv_tol=1e-7, min_iter=5,
            ad_backward_method="gmres", jit_ctm=True,
            adjoint_maxiter=200, adjoint_tol=1e-7,
            adjoint_tikhonov=1e-6,
            forward_gauge="phase",
            adjoint_arnoldi_precheck=False,
        ),
        gs_num_steps=30, gs_optimizer="lbfgs", unit_cell="2site",
        gs_c4v=False, gs_explicit_ad=False, su_init=False, gs_verbose=True,
    ),
}

for name, cfg in configs.items():
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")
    with jax.default_device(jax.devices("cpu")[0]):
        t0 = time.time()
        result = optimize_gs_ad(gate, None, cfg)
        wall = time.time() - t0
    E = float(result.energy)
    print(f"  E = {E:.6f}  wall = {wall:.1f}s  variational = {E > -0.6694}")
```

**Step 2: Run it**

Run: `JAX_ENABLE_X64=1 uv run python /tmp/bench_328_jit_gmres.py 2>&1 | tee /tmp/bench_328_jit_gmres.log`

Record results. Success = JIT GMRES variational AND faster than eager.

---

### Task 5: Update GMRES docstring and xfail status

If Task 4 shows JIT GMRES working, update the config docstring and
potentially un-xfail the GMRES test.

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py:34-42` (ad_backward_method docstring)
- Modify: `tests/test_ipeps.py` (`test_gmres_implicit_ad_is_stable` — un-xfail or update)

**Step 1: Update docstring**

Replace the `ad_backward_method` docstring to note GMRES is stable with
`jit_ctm=True`:

```python
    ad_backward_method: Backward method for the implicit-diff path.
                        ``"vjp"`` (default) is the regression-covered
                        Neumann-series backward.  ``"gmres"`` uses
                        GMRES to solve `(I - J^T + τI) λ = g`;
                        combine with ``jit_ctm=True`` for XLA-fused
                        backward (recommended for non-C4v 2-site).
```

**Step 2: Update xfail test**

If benchmark is variational, replace the placeholder in
`test_gmres_implicit_ad_is_stable` with a real test or remove xfail.

**Step 3: Commit**

```bash
git add src/tenax/algorithms/ipeps_config.py tests/test_ipeps.py
git commit -m "docs(ad): update GMRES backward status after JIT stabilization"
```
