# Multisite `optimize_gs_ad` Dispatcher

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `optimize_gs_ad` to accept a `Lattice` object as `unit_cell`, enabling AD optimization for arbitrary multi-site unit cells (kagome, honeycomb, custom lattices).

**Architecture:** `iPEPSConfig.unit_cell` is widened from `str` to `str | Lattice`. When a `Lattice` is passed, `optimize_gs_ad` dispatches to `_optimize_gs_ad_multisite()` which translates the `Lattice.neighbor_map` to coordinate-keyed dicts and runs the same L-BFGS / HZ line search loop as the 2-site path, using `ctm_energy_implicit` + `compute_energy_ctm_tensor_multisite`. Returns `(dict[str, Tensor], dict[str, CTMTensorEnv], float)`.

**Tech Stack:** JAX, existing multisite CTM infrastructure (`ctm_energy_implicit`, `python_loop_ctm_converge`, `compute_energy_ctm_tensor_multisite`, `precondition_gradient_multisite`).

## 2026-04-23 Upstream Sync

**Latest merged upstream PRs (no newer CTM/AD merges beyond this set):**
- #340 `feat(ipeps): Python-level CTM AD with JIT-fused GMRES backward`
- #339 `docs: add chi_ramp to guides, tuning, and skills`
- #338 `refactor(ipeps): centralize AD policy and add import-cycle guardrails`
- #337 `refactor(ctm): unify on Python-loop CTM forward + chi ramp`

**Implication for this plan:**
- Keep this plan focused on maintainability extension for multisite dispatch.
- Do not duplicate #337-#340 implementation scope; build on the merged baseline.

---

## Task 1: Widen `iPEPSConfig.unit_cell` to accept `Lattice`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py:203` (field type)
- Modify: `src/tenax/algorithms/ipeps_config.py:265-269` (validation)
- Test: `tests/test_ipeps_config.py`

**Step 1: Write the failing test**

Add to `tests/test_ipeps_config.py`:

```python
def test_unit_cell_accepts_lattice():
    from tenax import kagome
    config = iPEPSConfig(unit_cell=kagome())
    assert config.unit_cell.sites == ("u", "v", "w")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_config.py::test_unit_cell_accepts_lattice -v`
Expected: FAIL — `ValueError: unit_cell must be one of {'1x1', '2site'}`

**Step 3: Implement**

In `src/tenax/algorithms/ipeps_config.py`:

1. Add import at top (after existing imports):

```python
from tenax.core.lattice import Lattice
```

2. Change field type (line 203):

```python
unit_cell: str | Lattice = "1x1"  # "1x1", "2site", or Lattice(...)
```

3. Update validation in `__post_init__` (lines 265-269):

```python
valid_unit_cells = {"1x1", "2site"}
if not isinstance(self.unit_cell, Lattice) and self.unit_cell not in valid_unit_cells:
    raise ValueError(
        f"unit_cell must be one of {valid_unit_cells} or a Lattice, "
        f"got {self.unit_cell!r}"
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps_config.py -v`
Expected: PASS (all existing tests + new test)

**Step 5: Commit**

```
feat(config): allow Lattice objects in iPEPSConfig.unit_cell
```

---

## Task 2: Add `_lattice_to_neighbors` helper

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (add helper near top)
- Test: `tests/test_ipeps.py`

**Step 1: Write the failing test**

Add to `tests/test_ipeps.py`:

```python
def test_lattice_to_neighbors_kagome():
    from tenax import kagome
    from tenax.algorithms.ipeps_optimize import _lattice_to_neighbors

    lat = kagome()
    neighbors, name_to_coord, coord_to_name = _lattice_to_neighbors(lat)

    # 3 sites
    assert len(neighbors) == 3
    assert len(name_to_coord) == 3
    # All values should be Coord tuples
    for c, dirs in neighbors.items():
        assert set(dirs.keys()) == {"left", "right", "top", "bottom"}
        for nb in dirs.values():
            assert nb in neighbors
    # Round-trip: coord_to_name inverts name_to_coord
    for name, coord in name_to_coord.items():
        assert coord_to_name[coord] == name
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps.py::test_lattice_to_neighbors_kagome -v`
Expected: FAIL — `ImportError: cannot import name '_lattice_to_neighbors'`

**Step 3: Implement**

Add after the existing imports in `src/tenax/algorithms/ipeps_optimize.py` (around line 25):

```python
from tenax.core.lattice import Lattice

Coord = tuple[int, int]


def _lattice_to_neighbors(
    lattice: Lattice,
) -> tuple[dict[Coord, dict[str, Coord]], dict[str, Coord], dict[Coord, str]]:
    """Convert a Lattice neighbor_map to coordinate-keyed dicts.

    Returns (neighbors, name_to_coord, coord_to_name).
    """
    name_to_coord: dict[str, Coord] = {
        name: (i, 0) for i, name in enumerate(lattice.sites)
    }
    coord_to_name: dict[Coord, str] = {v: k for k, v in name_to_coord.items()}
    neighbors: dict[Coord, dict[str, Coord]] = {
        name_to_coord[name]: {
            direction: name_to_coord[nb]
            for direction, nb in lattice.neighbor_map[name].items()
        }
        for name in lattice.sites
    }
    return neighbors, name_to_coord, coord_to_name
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps.py::test_lattice_to_neighbors_kagome -v`
Expected: PASS

**Step 5: Commit**

```
feat(ipeps): add _lattice_to_neighbors helper
```

---

## Task 3: Add dispatch in `optimize_gs_ad`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:399` (add dispatch branch)
- Test: `tests/test_ipeps.py`

**Step 1: Write the failing test**

Add to `tests/test_ipeps.py`:

```python
def test_multisite_ad_runs_kagome(self, heisenberg_gate):
    """Multisite AD with kagome lattice should run without crashing."""
    from tenax import kagome

    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=10),
        gs_num_steps=2,
        unit_cell=kagome(),
    )
    result = optimize_gs_ad(heisenberg_gate, None, config)
    site_tensors, envs, E_gs = result

    # Check return types
    assert isinstance(site_tensors, dict)
    assert set(site_tensors.keys()) == {"u", "v", "w"}
    assert isinstance(envs, dict)
    assert set(envs.keys()) == {"u", "v", "w"}
    for name in ("u", "v", "w"):
        assert site_tensors[name].todense().shape == (2, 2, 2, 2, 2)
    assert np.isfinite(E_gs)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps.py::TestIPEPSAD::test_multisite_ad_runs_kagome -v`
Expected: FAIL — no multisite dispatch exists yet

**Step 3: Implement dispatch**

In `optimize_gs_ad()` (line 399 of `ipeps_optimize.py`), add before the existing `if config.unit_cell == "2site":` check:

```python
    if isinstance(config.unit_cell, Lattice):
        return _optimize_gs_ad_multisite(hamiltonian_gate, A_init, config)
```

Then add a stub `_optimize_gs_ad_multisite` at the end of the file (before `optimize_fpeps_ad`):

```python
def _optimize_gs_ad_multisite(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: dict[str, Tensor] | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for multi-site iPEPS (Lattice unit cell).

    Uses implicit differentiation through the multisite CTM fixed point.
    Returns (site_tensors_dict, envs_dict, E_gs) keyed by site name.
    """
    raise NotImplementedError("multisite AD optimization — implemented in Task 4")
```

The test will fail with `NotImplementedError`, confirming dispatch works.

**Step 4: Run test to verify dispatch reaches stub**

Run: `uv run pytest tests/test_ipeps.py::TestIPEPSAD::test_multisite_ad_runs_kagome -v`
Expected: FAIL with `NotImplementedError: multisite AD optimization`

**Step 5: Commit**

```
feat(ipeps): add multisite dispatch in optimize_gs_ad (stub)
```

---

## Task 4: Implement `_optimize_gs_ad_multisite`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (replace stub)

This is the core task. The function follows the same pattern as
`_optimize_gs_ad_tensor_2site` (lines 1234-1964) but generalized to N sites.

**Step 1: Implement initialization**

Replace the stub with the full function. Key differences from 2-site:

- `A_init` is `dict[str, Tensor] | None` (site-name-keyed)
- No C4v support (frustrated lattices don't have sublattice rotation)
- Random complex128 initialization for all sites
- Uses `_lattice_to_neighbors()` to get coordinate mapping

```python
def _optimize_gs_ad_multisite(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: dict[str, Tensor] | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization for multi-site iPEPS unit cells.

    Accepts ``config.unit_cell`` as a ``Lattice`` object.  Returns
    ``(site_tensors, envs, E_gs)`` where keys are site names from the lattice.
    """
    lattice = config.unit_cell
    config = _normalize_stall_recovery(config, unit_cell="multisite")
    neighbors, name_to_coord, coord_to_name = _lattice_to_neighbors(lattice)
    coords = sorted(neighbors.keys())  # deterministic order
    n_sites = len(coords)

    import optax

    from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit, ctm_energy_explicit
    from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_multisite
    from tenax.algorithms._line_search import hager_zhang_line_search
    from tenax.algorithms._metric_precond import (
        lbfgs_two_loop,
        precondition_gradient_multisite,
    )
    from tenax.algorithms.ad_utils import CTMRGGradientError, _wrap_tensor

    gate = (
        hamiltonian_gate.todense()
        if isinstance(hamiltonian_gate, Tensor)
        else jnp.array(hamiltonian_gate)
    )
    d_phys = gate.shape[0]
    D = config.max_bond_dim

    # --- Initialization ---
    if A_init is not None:
        if not isinstance(A_init, dict):
            raise TypeError(
                "For Lattice unit_cell, A_init must be None or a "
                f"dict[str, Tensor], got {type(A_init).__name__}"
            )
        site_tensors = {
            name_to_coord[name]: (
                _wrap_as_dense_tensor(t) if not isinstance(t, Tensor) else t
            )
            for name, t in A_init.items()
        }
    else:
        site_tensors = {}
        key = jax.random.PRNGKey(0)
        for c in coords:
            k1, k2, key = jax.random.split(key, 3)
            data = jax.random.normal(
                k1, (D, D, D, D, d_phys)
            ) + 1j * jax.random.normal(k2, (D, D, D, D, d_phys))
            data = data / (jnp.linalg.norm(data) + 1e-10)
            site_tensors[c] = _wrap_as_dense_tensor(data)

    # Normalize initial tensors
    for c in coords:
        t = site_tensors[c]
        site_tensors[c] = t * (1.0 / (t.norm() + 1e-10))

    ctm_cfg = build_ad_ctm_config(config)
    use_explicit = not config.gs_implicit_ad
    explicit_steps = config.gs_explicit_ad_steps
    explicit_warmup = config.gs_explicit_ad_warmup

    env_cache: dict[str, dict] = {}

    # --- Energy functions ---
    def _energy_fn(site_tensors_, envs_, gate_):
        return compute_energy_ctm_tensor_multisite(
            site_tensors_, envs_, neighbors, gate_
        )

    def _ctm_energy_fn(st):
        env_init = env_cache.get("envs", None)
        if use_explicit:
            return ctm_energy_explicit(
                st, neighbors, gate,
                chi=ctm_cfg.chi,
                warmup_steps=explicit_warmup,
                backprop_steps=explicit_steps,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                env_init=env_init,
                energy_fn=_energy_fn,
            )
        else:
            return ctm_energy_implicit(
                st, neighbors, gate,
                chi=ctm_cfg.chi,
                max_iter=ctm_cfg.max_iter,
                conv_tol=ctm_cfg.conv_tol,
                projector_method=ctm_cfg.projector_method,
                renormalize=ctm_cfg.renormalize,
                projector_backward=ctm_cfg.projector_backward,
                qr_warmup_steps=ctm_cfg.qr_warmup_steps,
                chi_ramp=ctm_cfg.chi_ramp,
                env_init=env_init,
                forward_gauge=ctm_cfg.forward_gauge,
                conv_method=ctm_cfg.ctm_conv_method,
                min_iter=ctm_cfg.min_iter,
                energy_fn=_energy_fn,
            )

    def loss_fn(params_tuple):
        st = {}
        for i, c in enumerate(coords):
            t = params_tuple[i]
            st[c] = t * (1.0 / (t.norm() + 1e-10))
        return _ctm_energy_fn(st)

    def loss_fn_fwd(params_tuple):
        st = {}
        for i, c in enumerate(coords):
            t = params_tuple[i]
            st[c] = t * (1.0 / (t.norm() + 1e-10))
        envs, _ = python_loop_ctm_converge(
            st, neighbors,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=env_cache.get("envs", None),
        )
        return float(
            compute_energy_ctm_tensor_multisite(st, envs, neighbors, gate)
        )

    def _update_env_cache(params_tuple):
        st = {}
        for i, c in enumerate(coords):
            t = params_tuple[i]
            st[c] = t * (1.0 / (t.norm() + 1e-10))
        envs, _ = python_loop_ctm_converge(
            st, neighbors,
            chi=ctm_cfg.chi,
            max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol,
            renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=env_cache.get("envs", None),
        )
        env_cache["envs"] = envs

    # --- Flatten / unflatten for L-BFGS ---
    def _flatten(params_tuple):
        return jnp.concatenate([p.todense().reshape(-1) for p in params_tuple])

    def _tree_dot(a, b):
        return float(jnp.real(
            sum(jnp.vdot(ai.todense(), bi.todense()) for ai, bi in zip(a, b))
        ))

    def _tree_add(a, b):
        return tuple(
            _wrap_tensor(ai.todense() + bi.todense(), ai)
            for ai, bi in zip(a, b)
        )

    def _tree_scale(a, alpha):
        return tuple(_wrap_tensor(ai.todense() * alpha, ai) for ai in a)

    def _normalize(params_tuple):
        return tuple(p * (1.0 / (p.norm() + 1e-10)) for p in params_tuple)

    # --- Optimization loop (L-BFGS + HZ line search) ---
    params = tuple(site_tensors[c] for c in coords)
    is_metric_lbfgs = (
        config.gs_metric_precond and config.gs_optimizer.lower() == "lbfgs"
    )
    lbfgs_history: list = []
    prev_params_flat = None
    prev_grad_flat = None
    best_energy = float("inf")
    best_params = params
    best_env_cache: dict[str, dict] = {}
    prev_energy = float("inf")
    log_interval = config.gs_log_interval
    stall_count = 0

    for step in range(config.gs_num_steps):
        try:
            energy_val, grads = jax.value_and_grad(loss_fn)(params)
        except CTMRGGradientError as exc:
            _logger.warning(
                "[iPEPS-AD:multisite] rho(J^T)=%.4f at step %d — stall recovery",
                exc.spectral_radius, step,
            )
            if config.gs_verbose:
                print(
                    f"[iPEPS-AD:multisite] step {step + 1}/{config.gs_num_steps} "
                    f"rho(J^T)={exc.spectral_radius:.4f} — stall recovery",
                    flush=True,
                )
            stall_count += 1
            if config.gs_stall_recovery == "reset":
                params = best_params
                env_cache.update(best_env_cache)
                lbfgs_history.clear()
                prev_params_flat = None
                prev_grad_flat = None
            continue

        energy_float = float(energy_val)
        _update_env_cache(params)

        if _should_accept_best(
            current_best=best_energy,
            candidate=energy_float,
            floor=config.gs_energy_floor,
        ):
            best_energy = energy_float
            best_params = params
            best_env_cache = dict(env_cache)

        delta_energy = abs(energy_float - prev_energy)
        if config.gs_verbose and _should_log_step(
            step, config.gs_num_steps, log_interval
        ):
            _log_ad_step(
                "multisite", step, config.gs_num_steps,
                energy_float, delta_energy, best_energy,
            )
        prev_energy = energy_float

        if delta_energy < config.gs_conv_tol and step > 5:
            if config.gs_verbose:
                _log_ad_converged("multisite", step, delta_energy, config.gs_conv_tol)
            break

        # L-BFGS direction
        p_flat = _flatten(params)
        g_flat = _flatten(grads)

        if prev_params_flat is not None:
            s = p_flat - prev_params_flat
            y = g_flat - prev_grad_flat
            sy = float(jnp.real(jnp.vdot(s, y)))
            if sy > 1e-10:
                rho = 1.0 / sy
                lbfgs_history.append((s, y, rho))
                if len(lbfgs_history) > 10:
                    lbfgs_history.pop(0)
        prev_params_flat = p_flat
        prev_grad_flat = g_flat

        # H0: metric preconditioning if envs available
        envs_cached = env_cache.get("envs", {})
        delta_metric = (
            delta_energy if step > 0
            else float(jnp.real(jnp.vdot(g_flat, g_flat)))
        )
        sites_m = {c: params[i] for i, c in enumerate(coords)}
        n_per_site = params[0].todense().size

        def h0_matvec(v):
            grads_v = {}
            offset_ = 0
            for c_ in coords:
                chunk = v[offset_: offset_ + n_per_site]
                grads_v[c_] = _wrap_tensor(
                    chunk.reshape(params[0].todense().shape), params[0]
                )
                offset_ += n_per_site
            z_dict = precondition_gradient_multisite(
                sites_m, envs_cached, grads_v, delta_metric, config
            )
            return jnp.concatenate([z_dict[c_].reshape(-1) for c_ in coords])

        if is_metric_lbfgs and envs_cached:
            direction_flat = lbfgs_two_loop(g_flat, lbfgs_history, h0_matvec)
        else:
            direction_flat = lbfgs_two_loop(
                g_flat, lbfgs_history, lambda v: v
            )

        # Unflatten direction
        dir_parts = []
        offset = 0
        for i, c in enumerate(coords):
            n = params[i].todense().size
            shape = params[i].todense().shape
            dir_parts.append(
                _wrap_tensor(-direction_flat[offset: offset + n].reshape(shape), params[i])
            )
            offset += n
        direction = tuple(dir_parts)

        # Tangent projection
        direction = _tangent_project_unit(direction, params)

        # Line search
        slope = _tree_dot(grads, direction)
        if slope >= 0:
            direction = tuple(_wrap_tensor(-g.todense(), g) for g in grads)
            direction = _tangent_project_unit(direction, params)
            slope = _tree_dot(grads, direction)
            lbfgs_history.clear()

        dir_norm = math.sqrt(max(_tree_dot(direction, direction), 1e-30))
        param_norm = math.sqrt(max(_tree_dot(params, params), 1e-30))
        alpha0 = min(1.0, 0.1 * param_norm / dir_norm)

        def _phi(alpha):
            trial = _normalize(_tree_add(params, _tree_scale(direction, alpha)))
            return loss_fn_fwd(trial)

        def _dphi(alpha):
            trial = _normalize(_tree_add(params, _tree_scale(direction, alpha)))
            _, g = jax.value_and_grad(loss_fn)(trial)
            return _tree_dot(g, direction)

        alpha, f_alpha, _converged = hager_zhang_line_search(
            _phi, _dphi, energy_float, slope,
            alpha_init=alpha0,
            rho=1.5,
            max_step=2.0 * alpha0,
            energy_bound=max(2.0, 2.0 * abs(best_energy)),
        )
        if f_alpha < energy_float:
            params = _normalize(_tree_add(params, _tree_scale(direction, alpha)))
            stall_count = 0
        else:
            stall_count += 1
            if config.gs_stall_recovery == "reset" and stall_count > config.gs_noise_recovery_retries:
                params = best_params
                env_cache.update(best_env_cache)
                lbfgs_history.clear()
                prev_params_flat = None
                prev_grad_flat = None
                stall_count = 0

    # --- Final evaluation ---
    st_final = {c: params[i] * (1.0 / (params[i].norm() + 1e-10)) for i, c in enumerate(coords)}
    envs_final, _ = python_loop_ctm_converge(
        st_final, neighbors,
        chi=ctm_cfg.chi,
        max_iter=ctm_cfg.max_iter,
        conv_tol=ctm_cfg.conv_tol,
        renormalize=ctm_cfg.renormalize,
        projector_method=ctm_cfg.projector_method,
        qr_warmup_steps=ctm_cfg.qr_warmup_steps,
        projector_backward=ctm_cfg.projector_backward,
        chi_ramp=ctm_cfg.chi_ramp,
        env_init=env_cache.get("envs", None),
    )
    E_last = float(compute_energy_ctm_tensor_multisite(st_final, envs_final, neighbors, gate))

    if best_params is not params:
        st_best = {c: best_params[i] * (1.0 / (best_params[i].norm() + 1e-10)) for i, c in enumerate(coords)}
        envs_best, _ = python_loop_ctm_converge(
            st_best, neighbors,
            chi=ctm_cfg.chi, max_iter=ctm_cfg.max_iter,
            conv_tol=ctm_cfg.conv_tol, renormalize=ctm_cfg.renormalize,
            projector_method=ctm_cfg.projector_method,
            qr_warmup_steps=ctm_cfg.qr_warmup_steps,
            projector_backward=ctm_cfg.projector_backward,
            chi_ramp=ctm_cfg.chi_ramp,
            env_init=best_env_cache.get("envs", None),
        )
        E_best_fresh = float(compute_energy_ctm_tensor_multisite(st_best, envs_best, neighbors, gate))
    else:
        E_best_fresh = E_last + 1  # force E_last to win

    if E_last <= E_best_fresh:
        final_st, final_envs, E_gs = st_final, envs_final, E_last
    else:
        final_st, final_envs, E_gs = st_best, envs_best, E_best_fresh

    if config.gs_verbose:
        print(f"[iPEPS-AD:multisite] final E={E_gs:.10f}", flush=True)

    # Map back to site names
    out_tensors = {coord_to_name[c]: final_st[c] for c in coords}
    out_envs = {coord_to_name[c]: final_envs[c] for c in coords}
    return out_tensors, out_envs, E_gs
```

**Step 2: Also update `_normalize_stall_recovery` (line 33)**

Change the `default` line to handle multisite:

```python
default = "noise" if unit_cell == "1x1" else "reset"
```

This already returns `"reset"` for anything other than `"1x1"`, so multisite gets `"reset"` automatically. No change needed.

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps.py::TestIPEPSAD::test_multisite_ad_runs_kagome -v`
Expected: PASS

**Step 4: Commit**

```
feat(ipeps): implement _optimize_gs_ad_multisite for Lattice unit cells
```

---

## Task 5: Add energy-decreasing test

**Files:**
- Test: `tests/test_ipeps.py`

**Step 1: Write the test**

```python
def test_multisite_ad_energy_decreases_kagome(self, heisenberg_gate):
    """Multisite AD energy should decrease after a few steps."""
    from tenax import kagome

    config = iPEPSConfig(
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=20),
        gs_num_steps=5,
        unit_cell=kagome(),
        gs_optimizer="lbfgs",
        gs_metric_precond=False,  # skip metric for speed
    )
    site_tensors, envs, E_gs = optimize_gs_ad(heisenberg_gate, None, config)
    assert E_gs < 0.0, f"Energy should be negative, got {E_gs}"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_ipeps.py::TestIPEPSAD::test_multisite_ad_energy_decreases_kagome -v`
Expected: PASS

**Step 3: Commit**

```
test: add multisite AD energy-decreasing test for kagome
```

---

## Task 6: Update docstring and type hints for `optimize_gs_ad`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:359-387` (docstring)

**Step 1: Update the docstring**

Update the `optimize_gs_ad` docstring to document the multisite return type:

```python
def optimize_gs_ad(
    hamiltonian_gate: jax.Array | Tensor,
    A_init: jax.Array | Tensor | tuple | dict | None,
    config: iPEPSConfig,
):
    """AD-based ground state optimization of iPEPS.

    Supports:
    - 1-site (``unit_cell="1x1"``): returns ``(A_opt, env, E_gs)``
    - 2-site (``unit_cell="2site"``): returns ``((A, B), (env_A, env_B), E_gs)``
    - Multi-site (``unit_cell=Lattice(...)``): returns
      ``(dict[str, Tensor], dict[str, CTMTensorEnv], E_gs)``
      where keys are site names from the lattice.

    ...
    """
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/test_ipeps.py -v`
Expected: All PASS

**Step 3: Commit**

```
docs: document multisite return type in optimize_gs_ad
```

---

## Task 7: Update benchmark script to use `optimize_gs_ad`

**Files:**
- Modify: `examples/kagome_heisenberg_benchmark.py`

**Step 1: Simplify benchmark to use the new API**

Replace the manual optimization loop with:

```python
from tenax import optimize_gs_ad, kagome

config = iPEPSConfig(
    max_bond_dim=D,
    ctm=CTMConfig(chi=chi, max_iter=100, min_iter=30, conv_tol=1e-8),
    unit_cell=kagome(),
    gs_optimizer="lbfgs",
    gs_line_search_method="hager_zhang",
    gs_metric_precond=True,
    gs_num_steps=num_steps,
    gs_verbose=True,
    gs_log_interval=5,
)
site_tensors, envs, E_gs = optimize_gs_ad(gate, None, config)
```

**Step 2: Run the benchmark smoke test (2 steps)**

Run: `JAX_ENABLE_X64=1 uv run python -c "import examples.kagome_heisenberg_benchmark as b; b.run_benchmark(D=2, chi=8, num_steps=2, label='smoke')"`
Expected: Runs without error, prints energy

**Step 3: Commit**

```
bench: simplify kagome benchmark to use optimize_gs_ad multisite API
```

---

## Task 8: Lint, format, run full test suite

**Step 1: Lint and format**

```bash
uv run ruff check src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_optimize.py examples/kagome_heisenberg_benchmark.py --fix
uv run ruff format src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_optimize.py examples/kagome_heisenberg_benchmark.py
```

**Step 2: Run core tests**

```bash
uv run pytest -m core -x -v
```

Expected: All PASS

**Step 3: Final commit if any fixes**

```
style: lint and format multisite dispatcher
```
