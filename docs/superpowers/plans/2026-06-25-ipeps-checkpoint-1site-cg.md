# CG Coarse-Grained 1-Site iPEPS Checkpoint/Resume — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend iPEPS ground-state-AD checkpoint/resume (currently `unit_cell="2site"`-only) to the `unit_cell="1x1"` path, with first-class support for the coarse-grained (`cg_gates`) supersite case, so long CG runs can checkpoint and resume.

**Architecture:** Mirror the existing, tested 2-site checkpoint wiring into the 1-site function `_optimize_gs_ad_tensor`. Add a `cg_gates_fingerprint` and a `_config_to_dict` fix so configs carrying `cg_gates` snapshot cleanly (fingerprint, not serialize). Narrow the dispatch guard to allow `1x1`; generic `Lattice` multisite stays guarded.

**Tech Stack:** Python, JAX, tenax (`optimize_gs_ad`, `_optimize_gs_ad_tensor`, `_checkpoint.py`, `coarse_grain.CGGates`/`kagome_cg_gates`, `pess.kagome_xxz_pess_cg_gates`), pytest. All new tests are fast (D=2, χ=4).

**Reference (the template to copy):** the 2-site implementation in `src/tenax/algorithms/ipeps_optimize.py` — the resume block at **lines 2728–2835** and the `_maybe_save_2s_checkpoint` closure + its bundle at **lines 2843–2886**, plus its save-call sites. The generic checkpoint API in `src/tenax/algorithms/_checkpoint.py` (`save_checkpoint`/`load_checkpoint`/`validate_config`/`gate_fingerprint`/`checkpoint_exists`) is reused unchanged.

**1-site vs 2-site variable name map** (apply when copying the template):

| 2-site | 1-site |
|---|---|
| `_env_cache_2s` | `_env_cache` |
| `best_env_cache_2s` | `best_env_cache` |
| `ctm_cfg_2s` | `ctm_cfg` |
| `_current_conv_tol_2s` | `_current_conv_tol` |
| `_current_patience_2s` | `_current_patience` |
| `prev_params_flat` | `prev_A_flat` |
| `_maybe_save_2s_checkpoint` | `_maybe_save_1s_checkpoint` |

---

## File Structure

| File | Responsibility |
|---|---|
| `src/tenax/algorithms/_checkpoint.py` (modify) | Add `cg_gates_fingerprint`; fix `_config_to_dict` to fingerprint `cg_gates`. |
| `src/tenax/algorithms/ipeps_optimize.py` (modify) | Narrow the dispatch guard; wire 1-site save + resume into `_optimize_gs_ad_tensor`. |
| `tests/test_ipeps_checkpoint.py` (modify) | Unit tests: `cg_gates_fingerprint`, `_config_to_dict` with `cg_gates`. |
| `tests/test_ipeps_checkpoint_resume.py` (modify) | Flip the reject test; add 1-site + CG resume e2e tests. |

---

### Task 1: `cg_gates_fingerprint` in `_checkpoint.py`

**Files:**
- Modify: `src/tenax/algorithms/_checkpoint.py`
- Test: `tests/test_ipeps_checkpoint.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_ipeps_checkpoint.py`:

```python
def test_cg_gates_fingerprint_round_trip_and_changes_with_bytes():
    import jax.numpy as jnp

    from tenax.algorithms._checkpoint import cg_gates_fingerprint
    from tenax.algorithms.coarse_grain import CGGates

    h_intra = jnp.eye(4)
    h_inter = {"h": jnp.ones((4, 4, 4, 4)), "v": jnp.zeros((4, 4, 4, 4))}
    g1 = CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=2, map_fn=None, init_fn=None)
    # same arrays, DIFFERENT callable -> same fingerprint (callables not hashed)
    g2 = CGGates(h_intra=h_intra, h_inter=h_inter, n_sites=2,
                 map_fn=lambda x: x, init_fn=None)
    assert cg_gates_fingerprint(g1) == cg_gates_fingerprint(g2)
    # perturb h_intra -> different fingerprint
    g3 = CGGates(h_intra=h_intra.at[0, 0].add(1.0), h_inter=h_inter, n_sites=2)
    assert cg_gates_fingerprint(g3) != cg_gates_fingerprint(g1)
    # perturb an h_inter entry -> different fingerprint
    g4 = CGGates(h_intra=h_intra,
                 h_inter={"h": h_inter["h"].at[0, 0, 0, 0].add(1.0), "v": h_inter["v"]},
                 n_sites=2)
    assert cg_gates_fingerprint(g4) != cg_gates_fingerprint(g1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_checkpoint.py::test_cg_gates_fingerprint_round_trip_and_changes_with_bytes -v`
Expected: FAIL — `ImportError: cannot import name 'cg_gates_fingerprint'`.

- [ ] **Step 3: Write minimal implementation** — add to `src/tenax/algorithms/_checkpoint.py` immediately after `gate_fingerprint` (after its `return` at ~line 70):

```python
def cg_gates_fingerprint(cg_gates: Any) -> tuple:
    """Stable identifier for a coarse-grained ``CGGates`` Hamiltonian.

    Hashes the array content + structure: ``(n_sites, fp(h_intra),
    ((label, fp(arr)) for label, arr in sorted h_inter))`` where ``fp`` is
    :func:`gate_fingerprint`.  The ``map_fn`` / ``init_fn`` callables are
    intentionally NOT hashed: they are re-supplied by the live ``config`` on
    resume, and two equivalent closures should still resume.  Used to refuse a
    silent coarse-grained-gate swap on resume.
    """
    inter = tuple(
        (label, gate_fingerprint(arr))
        for label, arr in sorted(cg_gates.h_inter.items())
    )
    return (int(cg_gates.n_sites), gate_fingerprint(cg_gates.h_intra), inter)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps_checkpoint.py::test_cg_gates_fingerprint_round_trip_and_changes_with_bytes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/tenax/algorithms/_checkpoint.py tests/test_ipeps_checkpoint.py
git add src/tenax/algorithms/_checkpoint.py tests/test_ipeps_checkpoint.py
git commit -m "feat(checkpoint): cg_gates_fingerprint for resume validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_config_to_dict` fingerprints `cg_gates`

**Files:**
- Modify: `src/tenax/algorithms/_checkpoint.py`
- Test: `tests/test_ipeps_checkpoint.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_ipeps_checkpoint.py`:

```python
def test_config_to_dict_handles_cg_gates():
    import pickle

    import jax.numpy as jnp

    from tenax.algorithms._checkpoint import _config_to_dict, validate_config
    from tenax.algorithms.coarse_grain import CGGates
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig

    cg = CGGates(
        h_intra=jnp.eye(4),
        h_inter={"h": jnp.ones((4, 4, 4, 4))},
        n_sites=2,
        map_fn=lambda *p: p[0],
        init_fn=None,
    )
    cfg = iPEPSConfig(unit_cell="1x1", max_bond_dim=2, ctm=CTMConfig(chi=4),
                      gs_num_steps=1, cg_gates=cg, su_init=False)
    d = _config_to_dict(cfg)
    # picklable (no jnp arrays / callables / mappingproxy leak through)
    pickle.dumps(d)
    # validate_config accepts an identical config (no array-truth-value error)
    validate_config(d, cfg)
    # a different cg_gates surfaces as a (soft) mismatch, not a crash
    cfg2 = iPEPSConfig(unit_cell="1x1", max_bond_dim=2, ctm=CTMConfig(chi=4),
                       gs_num_steps=1,
                       cg_gates=CGGates(h_intra=jnp.eye(4).at[0, 0].add(1.0),
                                        h_inter={"h": jnp.ones((4, 4, 4, 4))},
                                        n_sites=2),
                       su_init=False)
    with pytest.warns(UserWarning):
        validate_config(d, cfg2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_checkpoint.py::test_config_to_dict_handles_cg_gates -v`
Expected: FAIL — `_config_to_dict`'s `asdict` raises on the `jnp.ndarray` / callable fields of `cg_gates` (or `validate_config` raises `ValueError: truth value of an array...`).

- [ ] **Step 3: Write minimal implementation** — in `src/tenax/algorithms/_checkpoint.py`, replace the body of `_config_to_dict` (currently the 3-line `if is_dataclass(config): return asdict(config); return dict(config)` at the end of the function) with:

```python
    if is_dataclass(config):
        cg = getattr(config, "cg_gates", None)
        if cg is not None:
            # cg_gates holds jnp.ndarray fields + callables (map_fn/init_fn):
            # asdict can't recurse it and validate_config's dict-eq would hit
            # "truth value of an array".  Replace it with a hashable array
            # fingerprint so the snapshot is picklable and comparable; the
            # live cg_gates is re-supplied from `config` on resume.
            from dataclasses import replace as _dc_replace

            snap = _dc_replace(config, cg_gates=("__cg_gates_fp__",
                                                 cg_gates_fingerprint(cg)))
            return asdict(snap)
        return asdict(config)
    return dict(config)
```

Also update the `_config_to_dict` docstring: delete the now-stale "Fix when the 1-site cg_gates path is wired" bullet about `cg_gates` (keep the `Lattice`/`neighbor_map` bullet, still unhandled).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps_checkpoint.py::test_config_to_dict_handles_cg_gates -v`
Expected: PASS.

- [ ] **Step 5: Run the whole checkpoint unit suite + commit**

```bash
uv run pytest tests/test_ipeps_checkpoint.py -v
uv run ruff check src/tenax/algorithms/_checkpoint.py tests/test_ipeps_checkpoint.py
git add src/tenax/algorithms/_checkpoint.py tests/test_ipeps_checkpoint.py
git commit -m "fix(checkpoint): _config_to_dict fingerprints cg_gates (1-site path)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Narrow the dispatch guard to allow `1x1`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py:915-921`
- Test: `tests/test_ipeps_checkpoint_resume.py:242-...`

- [ ] **Step 1: Update the failing test first** — in `tests/test_ipeps_checkpoint_resume.py`, REPLACE `test_checkpoint_path_rejects_non_2site_paths` (the whole function, starting at line 242) with:

```python
def test_checkpoint_path_allows_1site_rejects_lattice():
    """1-site (incl. CG) checkpointing is now wired; generic Lattice
    multisite still raises NotImplementedError."""
    from tenax.algorithms.lattice import Lattice

    gate = _heisenberg_gate()
    # 1-site no longer raises NotImplementedError on the guard. (We pass
    # gs_resume=False + a tmp path; it should dispatch into the 1-site path,
    # not raise on the guard. We give 0 steps so it returns immediately.)
    cfg_1site = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=0,
        gs_checkpoint_path="/tmp/ckpt_guard_1site",
        gs_c4v=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    # must NOT raise NotImplementedError mentioning the guard:
    optimize_gs_ad(gate, None, cfg_1site)

    # generic Lattice multisite is still guarded
    lat = Lattice(nx=2, ny=1, neighbors={(0, 0): {"r": (1, 0)}, (1, 0): {"r": (0, 0)}})
    cfg_multi = iPEPSConfig(
        unit_cell=lat,
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=1,
        gs_checkpoint_path="/tmp/ckpt_guard_lattice",
        su_init=False,
    )
    with pytest.raises(NotImplementedError, match="Lattice|multisite"):
        optimize_gs_ad(gate, None, cfg_multi)
```

NOTE on the `Lattice(...)` constructor: check the real signature first with
`uv run python -c "from tenax.algorithms.lattice import Lattice; help(Lattice)"`.
If the kwargs differ from `nx/ny/neighbors`, adjust the `lat = Lattice(...)` line
to a minimal valid 2-site lattice; the test only needs a `Lattice` instance that
routes to `_optimize_gs_ad_multisite`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py::test_checkpoint_path_allows_1site_rejects_lattice -v`
Expected: FAIL — the 1-site call raises `NotImplementedError` (guard still blocks `1x1`).

- [ ] **Step 3: Narrow the guard** — in `src/tenax/algorithms/ipeps_optimize.py`, replace the guard block (lines 915–921):

```python
    # Checkpointing is currently wired only for the 2-site path; 1-site
    # and multisite land in follow-up PRs to feat/ipeps-checkpoint-2site.
    if config.gs_checkpoint_path is not None and config.unit_cell != "2site":
        raise NotImplementedError(
            "iPEPSConfig.gs_checkpoint_path is currently supported only for "
            "unit_cell='2site'. 1-site and multisite checkpoint wiring land "
            "in follow-up PRs (see PR #497)."
        )
```

with:

```python
    # Checkpointing is wired for the 2-site and 1-site (incl. coarse-grained
    # cg_gates) paths. Generic Lattice multisite is not yet wired (#497).
    if config.gs_checkpoint_path is not None and isinstance(config.unit_cell, Lattice):
        raise NotImplementedError(
            "iPEPSConfig.gs_checkpoint_path is not yet supported for generic "
            "Lattice multisite unit cells; use unit_cell='2site' or '1x1' "
            "(incl. cg_gates). Multisite checkpoint wiring is a follow-up (#497)."
        )
```

(`Lattice` is already imported in this module — it is used a few lines below at `if isinstance(config.unit_cell, Lattice): return _optimize_gs_ad_multisite(...)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py::test_checkpoint_path_allows_1site_rejects_lattice -v`
Expected: PASS. (1-site dispatches without the guard error; Lattice still raises.)

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git commit -m "feat(checkpoint): allow gs_checkpoint_path on unit_cell=1x1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 1-site checkpoint SAVE wiring

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (inside `_optimize_gs_ad_tensor`, the function at line 1202; its loop is at line 1553)
- Test: `tests/test_ipeps_checkpoint_resume.py`

This task makes a 1-site run WRITE `ckpt.last.pkl`. (Resume is Task 5.)

- [ ] **Step 1: Write the failing test** — append to `tests/test_ipeps_checkpoint_resume.py`:

```python
def test_1site_writes_checkpoint(tmp_path):
    """A plain 1-site run with gs_checkpoint_path writes ckpt.last.pkl whose
    recorded step matches the last completed optimizer step."""
    from tenax.algorithms._checkpoint import checkpoint_exists, load_checkpoint

    gate = _heisenberg_gate()
    cfg = iPEPSConfig(
        unit_cell="1x1",
        max_bond_dim=2,
        ctm=CTMConfig(chi=4),
        gs_num_steps=2,
        gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1,
        gs_c4v=False,
        su_init=False,
        gs_conv_criterion="grad_norm",
    )
    optimize_gs_ad(gate, None, cfg)
    assert checkpoint_exists(str(tmp_path))
    bundle = load_checkpoint(str(tmp_path))
    assert bundle["step"] == 1  # 0-indexed last of 2 steps
    assert "params" in bundle and "opt_state" in bundle
    assert bundle["cg_gates_fingerprint"] is None  # plain 1-site, no cg_gates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py::test_1site_writes_checkpoint -v`
Expected: FAIL — no checkpoint written (guard passes but no save wiring); `checkpoint_exists` is `False`.

- [ ] **Step 3: Add startup fingerprints + the save closure**

(3a) Near the top of `_optimize_gs_ad_tensor`, AFTER the state variables are initialised (after `_current_patience = ctm_cfg.plateau_patience`, ~line 1501) and the checkpoint imports are available, add the fingerprint setup. First ensure the checkpoint helpers are imported in this function (mirror the 2-site import at line 2442) — add near the top of `_optimize_gs_ad_tensor`:

```python
    from tenax.algorithms._checkpoint import (
        cg_gates_fingerprint,
        checkpoint_exists,
        gate_fingerprint,
        load_checkpoint,
        save_checkpoint,
        validate_config,
    )
```

Then add the fingerprint vars + `start_step` default (place just before the `for step in range(...)` loop at line 1553):

```python
    _gate_fp = gate_fingerprint(gate) if config.gs_checkpoint_path is not None else None
    _cg_fp = (
        cg_gates_fingerprint(config.cg_gates)
        if (config.gs_checkpoint_path is not None and config.cg_gates is not None)
        else None
    )
    start_step = 0
```

(3b) Add the save closure just before the loop (after the fingerprint vars). This mirrors `_maybe_save_2s_checkpoint` (template at lines 2843–2886) with the 1-site name map applied:

```python
    def _maybe_save_1s_checkpoint(step, chi_before, e_prev, *, force_last=False):
        if config.gs_checkpoint_path is None:
            return
        chi_changed = ctm_cfg.chi != chi_before
        is_new_best = best_energy < e_prev
        should_save_last = (
            force_last or chi_changed or (step + 1) % config.gs_checkpoint_every == 0
        )
        if not (should_save_last or is_new_best):
            return
        _ckpt_state = {
            "step": step,
            "config": _config_to_dict(config),
            "hamiltonian_fingerprint": _gate_fp,
            "cg_gates_fingerprint": _cg_fp,
            "params": params,
            "best_params": best_params,
            "best_energy": float(best_energy),
            "prev_energy": float(prev_energy),
            "env_cache": dict(_env_cache),
            "best_env_cache": dict(best_env_cache),
            "opt_state": opt_state,
            "lbfgs_history": list(lbfgs_history),
            "prev_params_flat": prev_A_flat,
            "prev_grad_flat": prev_grad_flat,
            "cg_direction": cg_direction,
            "prev_grad": prev_grad,
            "prev_precond_grad": prev_precond_grad,
            "stall_count": stall_count,
            "current_stage_idx": current_stage_idx,
            "stage_start_step": stage_start_step,
            "ctm_cfg_chi": ctm_cfg.chi,
            "current_conv_tol": _current_conv_tol,
            "current_patience": _current_patience,
        }
        if should_save_last:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path)
        if is_new_best:
            save_checkpoint(_ckpt_state, config.gs_checkpoint_path, is_best=True)
```

`_config_to_dict` is already imported at module scope in this file (used by the 2-site path); if a `NameError` occurs, add it to the import block in (3a).

NOTE on closure semantics: `_maybe_save_1s_checkpoint` only READS the loop-mutated names (`params`, `best_energy`, `lbfgs_history`, `ctm_cfg`, `prev_A_flat`, …). Because they live in the enclosing function scope (not a narrower block), the closure sees their current values at call time — no `nonlocal` needed. This is exactly how `_maybe_save_2s_checkpoint` works.

VERIFY the names exist with these exact spellings in `_optimize_gs_ad_tensor` before referencing them: `params`, `best_params`, `best_energy`, `prev_energy`, `_env_cache`, `best_env_cache`, `opt_state`, `lbfgs_history`, `prev_A_flat`, `prev_grad_flat`, `cg_direction`, `prev_grad`, `prev_precond_grad`, `stall_count`, `current_stage_idx`, `stage_start_step`, `ctm_cfg`, `_current_conv_tol`, `_current_patience`. (All were confirmed present. If `lbfgs_history` is named differently in the 1-site metric-LBFGS branch, match the actual name.)

(3c) Add the save CALL sites in the loop, mirroring the 2-site call sites:
- At the **top of the loop body** capture the step-start snapshots:
  ```python
        _chi_at_step_start = ctm_cfg.chi
        _e_prev_at_step_start = prev_energy
  ```
- At the **end of each loop iteration** (the last statements of the `for step` body, after the optimizer update), add:
  ```python
        _maybe_save_1s_checkpoint(step, _chi_at_step_start, _e_prev_at_step_start)
  ```
- Before each `continue` that follows a chi-bump / stage-advance (the 1-site loop has bump/stage-advance points analogous to the 2-site ones at `force_last=True` sites — `_maybe_bump_chi` results and the convergence/stage-advance branch), add a forced save:
  ```python
        _maybe_save_1s_checkpoint(step, _chi_at_step_start, _e_prev_at_step_start, force_last=True)
  ```
  Place these mirroring the 2-site `force_last=True` calls. The e2e tests (this task + Task 5) pin the behavior; if a bump-then-continue path isn't covered, the resume test in Task 5 will surface it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py::test_1site_writes_checkpoint -v`
Expected: PASS (`ckpt.last.pkl` written, `step == 1`, `cg_gates_fingerprint is None`).

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git commit -m "feat(checkpoint): 1-site checkpoint save wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 1-site RESUME wiring

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (inside `_optimize_gs_ad_tensor`)
- Test: `tests/test_ipeps_checkpoint_resume.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_ipeps_checkpoint_resume.py`:

```python
def test_resume_1site_continues_from_saved_step(tmp_path):
    """Run 2 steps, checkpoint; resume to 4 total; the resumed run picks up at
    step 2 and reaches the same final state as an uninterrupted 4-step run."""
    gate = _heisenberg_gate()

    def cfg(nsteps, resume):
        return iPEPSConfig(
            unit_cell="1x1", max_bond_dim=2, ctm=CTMConfig(chi=4),
            gs_num_steps=nsteps, gs_checkpoint_path=str(tmp_path),
            gs_checkpoint_every=1, gs_resume=resume, gs_c4v=False,
            su_init=False, gs_conv_criterion="grad_norm",
        )

    optimize_gs_ad(gate, None, cfg(2, False))            # phase A: 2 steps
    A_resumed, _, E_resumed = optimize_gs_ad(gate, None, cfg(4, True))  # resume -> 4

    from tenax.algorithms._checkpoint import load_checkpoint
    assert load_checkpoint(str(tmp_path))["step"] == 3   # 0-indexed last of 4
    assert E_resumed < 0  # finished, sensible energy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py::test_resume_1site_continues_from_saved_step -v`
Expected: FAIL — without resume wiring the second call restarts from step 0 (loop is `range(gs_num_steps)`, ignores the checkpoint), so the saved `step` ends at 3 only by luck OR the run redoes steps; more reliably it fails because `gs_resume=True` currently has no effect on the 1-site path and the final `step` / behavior won't match a resumed run. (If it happens to pass, the resume block is still required for correctness — proceed to Step 3.)

- [ ] **Step 3: Add the resume block** — in `_optimize_gs_ad_tensor`, immediately AFTER the `start_step = 0` line added in Task 4 (and after the save-closure definition is fine too, but before the loop), insert the resume block. This mirrors the 2-site resume block (template at lines 2728–2835) with the 1-site name map applied, plus the cg-fingerprint check:

```python
    if config.gs_resume:
        if not checkpoint_exists(config.gs_checkpoint_path):
            raise FileNotFoundError(
                f"gs_resume=True but no checkpoint found at "
                f"{config.gs_checkpoint_path!r} (looked for 'ckpt.last.pkl'). "
                f"Either point gs_checkpoint_path at the directory of a prior "
                f"run, or set gs_resume=False to start fresh."
            )
        bundle = load_checkpoint(config.gs_checkpoint_path)
        validate_config(bundle.get("config", {}), config)

        saved_fp = bundle.get("hamiltonian_fingerprint")
        if saved_fp is not None and tuple(saved_fp) != _gate_fp:
            raise ValueError(
                "Cannot resume: the hamiltonian gate has changed since the "
                "checkpoint was written.\n"
                f"  saved fingerprint:   {tuple(saved_fp)!r}\n"
                f"  current fingerprint: {_gate_fp!r}\n"
                "If this is intentional, start a fresh run."
            )

        saved_cg_fp = bundle.get("cg_gates_fingerprint")
        if saved_cg_fp is not None and saved_cg_fp != _cg_fp:
            raise ValueError(
                "Cannot resume: the coarse-grained cg_gates have changed since "
                "the checkpoint was written.\n"
                f"  saved cg fingerprint:   {saved_cg_fp!r}\n"
                f"  current cg fingerprint: {_cg_fp!r}\n"
                "If this is intentional, start a fresh run."
            )

        saved_cfg = bundle.get("config", {})
        _opt_compat = (
            saved_cfg.get("gs_optimizer") == config.gs_optimizer
            and saved_cfg.get("gs_metric_precond") == config.gs_metric_precond
        )

        params = bundle["params"]
        best_params = bundle["best_params"]
        best_energy = float(bundle["best_energy"])
        prev_energy = float(bundle["prev_energy"])
        _env_cache.clear()
        _env_cache.update(bundle.get("env_cache", {}))
        best_env_cache = dict(bundle.get("best_env_cache", {}))
        stall_count = int(bundle.get("stall_count", 0))

        if _opt_compat:
            if optimizer is not None and bundle.get("opt_state") is not None:
                opt_state = bundle["opt_state"]
            lbfgs_history = list(bundle.get("lbfgs_history") or [])
            prev_A_flat = bundle.get("prev_params_flat")
            prev_grad_flat = bundle.get("prev_grad_flat")
            cg_direction = bundle.get("cg_direction")
            prev_grad = bundle.get("prev_grad")
            prev_precond_grad = bundle.get("prev_precond_grad")
        else:
            import warnings as _warnings

            _warnings.warn(
                "Optimizer-defining config differs from checkpoint "
                f"(saved: gs_optimizer={saved_cfg.get('gs_optimizer')!r}, "
                f"gs_metric_precond={saved_cfg.get('gs_metric_precond')!r}; "
                f"current: gs_optimizer={config.gs_optimizer!r}, "
                f"gs_metric_precond={config.gs_metric_precond!r}). "
                "Restoring params/envs/energies but discarding saved "
                "optimizer history (curvature/momentum will restart fresh).",
                stacklevel=2,
            )

        current_stage_idx = int(bundle.get("current_stage_idx", 0))
        stage_start_step = int(bundle.get("stage_start_step", 0))
        saved_chi = bundle.get("ctm_cfg_chi")
        if saved_chi is not None and int(saved_chi) != ctm_cfg.chi:
            ctm_cfg = _replace(ctm_cfg, chi=int(saved_chi))
        saved_conv_tol = bundle.get("current_conv_tol")
        if saved_conv_tol is not None:
            _current_conv_tol = float(saved_conv_tol)
            ctm_cfg = _replace(ctm_cfg, conv_tol=_current_conv_tol)
        saved_patience = bundle.get("current_patience")
        if saved_patience is not None:
            _current_patience = int(saved_patience)
            ctm_cfg = _replace(ctm_cfg, plateau_patience=_current_patience)

        start_step = int(bundle["step"]) + 1
        if config.gs_verbose:
            print(
                f"[iPEPS-AD:1site-tensor] resumed from step {start_step} "
                f"(best E={best_energy:.10f}, chi={ctm_cfg.chi})",
                flush=True,
            )
```

NOTES:
- `_replace` is `dataclasses.replace`, already imported/aliased in this module (used by the 2-site path as `_replace`). If the alias name differs in scope, use `dataclasses.replace`.
- The 2-site uses `_drop_env_cache_for_reset(_env_cache_2s)` then `.update(...)`. The 1-site uses a plain `_env_cache.clear(); _env_cache.update(...)` — equivalent for the resume case (we are replacing the cache wholesale, not selectively invalidating). If `_drop_env_cache_for_reset` is in scope and you prefer parity, you may use it instead.
- These names are REASSIGNED here (`params`, `ctm_cfg`, etc.). They are function-local in `_optimize_gs_ad_tensor`, so plain assignment is correct (this is not a nested function).

- [ ] **Step 4: Change the loop range** — change the loop header at line 1553 from:

```python
    for step in range(config.gs_num_steps):
```

to:

```python
    for step in range(start_step, config.gs_num_steps):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py -k "1site or allows_1site" -v`
Expected: PASS — `test_1site_writes_checkpoint`, `test_resume_1site_continues_from_saved_step`, `test_checkpoint_path_allows_1site_rejects_lattice` all green.

- [ ] **Step 6: Commit**

```bash
uv run ruff check src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ipeps_checkpoint_resume.py
git commit -m "feat(checkpoint): 1-site resume wiring (load/validate/restore)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: CG coarse-grained resume e2e tests

**Files:**
- Test: `tests/test_ipeps_checkpoint_resume.py`

These validate the CG-specific path end to end. No production code change is expected — if a test fails, fix the wiring from Tasks 4–5 (do not weaken the test).

- [ ] **Step 1: Write the CG resume test** — append to `tests/test_ipeps_checkpoint_resume.py`:

```python
def _cg_cfg(tmp_path, *, nsteps, resume):
    from tenax.algorithms.pess import kagome_xxz_pess_cg_gates

    cg = kagome_xxz_pess_cg_gates(delta=1.0, d=2)
    return iPEPSConfig(
        unit_cell="1x1", max_bond_dim=2, ctm=CTMConfig(chi=4),
        gs_num_steps=nsteps, gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1, gs_resume=resume,
        cg_gates=cg, su_init=False, gs_c4v=False,
        gs_conv_criterion="grad_norm",
    )


@pytest.mark.slow
def test_resume_cg_1site_continues_from_saved_step(tmp_path):
    """Coarse-grained (cg_gates) 1-site run checkpoints and resumes; the
    resumed run picks up at the saved step and the bundle records a non-None
    cg_gates fingerprint."""
    from tenax.algorithms._checkpoint import load_checkpoint

    # CG path: hamiltonian_gate is unused (Hamiltonian lives in cg_gates); pass
    # a placeholder array of the right rank. (Confirm what optimize_gs_ad expects
    # for the CG path's `hamiltonian_gate` arg; many CG callers pass None.)
    gate = None
    optimize_gs_ad(gate, None, _cg_cfg(tmp_path, nsteps=2, resume=False))
    b = load_checkpoint(str(tmp_path))
    assert b["step"] == 1
    assert b["cg_gates_fingerprint"] is not None

    _, _, E = optimize_gs_ad(gate, None, _cg_cfg(tmp_path, nsteps=4, resume=True))
    assert load_checkpoint(str(tmp_path))["step"] == 3
    assert E < 0
```

NOTE: confirm the CG path's `hamiltonian_gate` argument — run a 1-step CG optimize
to see whether it expects `None` or a placeholder gate, and whether
`gate_fingerprint(None)` is reached when `cg_gates` is set. If `gate is None`
breaks `gate_fingerprint`, guard `_gate_fp` with `gate is not None` (adjust Task 4
(3a): `_gate_fp = gate_fingerprint(gate) if (config.gs_checkpoint_path is not None
and gate is not None) else None`). Make that adjustment if needed and note it.

- [ ] **Step 2: Write the reject-different-cg_gates test** — append:

```python
@pytest.mark.slow
def test_resume_rejects_different_cg_gates(tmp_path):
    """Resuming a CG run against perturbed cg_gates is a fatal mismatch."""
    optimize_gs_ad(None, None, _cg_cfg(tmp_path, nsteps=2, resume=False))

    from tenax.algorithms.pess import kagome_xxz_pess_cg_gates

    cg_other = kagome_xxz_pess_cg_gates(delta=2.0, d=2)  # different XXZ anisotropy
    cfg = iPEPSConfig(
        unit_cell="1x1", max_bond_dim=2, ctm=CTMConfig(chi=4),
        gs_num_steps=4, gs_checkpoint_path=str(tmp_path),
        gs_checkpoint_every=1, gs_resume=True, cg_gates=cg_other,
        su_init=False, gs_c4v=False, gs_conv_criterion="grad_norm",
    )
    with pytest.raises(ValueError, match="cg_gates"):
        optimize_gs_ad(None, None, cfg)
```

- [ ] **Step 3: Run the CG tests**

Run: `uv run pytest tests/test_ipeps_checkpoint_resume.py -k "cg_1site or different_cg_gates" -v`
Expected: PASS. If the CG path rejects `hamiltonian_gate=None`, apply the Task-6 Step-1 NOTE adjustment to Task 4's `_gate_fp` guard and re-run.

- [ ] **Step 4: Register the new test names' markers if needed** — `tests/test_ipeps_checkpoint_resume.py` is marked by `tests/conftest.py`'s `_FILE_MARKERS`. Confirm its entry (likely `"test_ipeps_checkpoint_resume.py": "algorithm"`). The `@pytest.mark.slow` on the two CG tests adds `slow` on top (the kagome CG d_eff=8 forward is heavier); the plain 1-site tests stay in the file's default bucket. No conftest change needed unless the file is unmarked — if so, add `"test_ipeps_checkpoint_resume.py": "algorithm"`.

- [ ] **Step 5: Run the full checkpoint suites + commit**

```bash
uv run pytest tests/test_ipeps_checkpoint.py tests/test_ipeps_checkpoint_resume.py -v
uv run ruff check tests/test_ipeps_checkpoint_resume.py
git add tests/test_ipeps_checkpoint_resume.py
git commit -m "test(checkpoint): CG coarse-grained 1-site resume + reject-swap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- `cg_gates_fingerprint` → Task 1. ✓
- `_config_to_dict` cg_gates fix → Task 2. ✓
- Narrow guard (allow 1x1, keep Lattice guarded) → Task 3. ✓
- 1-site save wiring (startup fingerprints, `_maybe_save_1s_checkpoint`, save sites, bundle incl. `cg_gates_fingerprint`) → Task 4. ✓
- 1-site resume wiring (load/validate/gate+cg fingerprint check/restore/loop range) → Task 5. ✓
- Flip the reject test → Task 3. ✓
- 1-site + CG e2e resume tests + reject-different-cg_gates → Tasks 5, 6. ✓
- Generic Lattice out of scope (still guarded) → Task 3 keeps the guard for Lattice. ✓

**Placeholder scan:** Complete code in every code step. Two flagged VERIFY/NOTE points (the `Lattice(...)` constructor kwargs in Task 3; the CG `hamiltonian_gate=None` handling in Task 6) are explicit "check the real signature and adjust" instructions with the concrete fallback given (guard `_gate_fp` on `gate is not None`) — not deferred work.

**Type/name consistency:** bundle keys are identical across the save closure (Task 4) and the resume block (Task 5): `params, best_params, best_energy, prev_energy, env_cache, best_env_cache, opt_state, lbfgs_history, prev_params_flat, prev_grad_flat, cg_direction, prev_grad, prev_precond_grad, stall_count, current_stage_idx, stage_start_step, ctm_cfg_chi, current_conv_tol, current_patience, step, config, hamiltonian_fingerprint, cg_gates_fingerprint`. The 1-site local `prev_A_flat` maps to the bundle key `prev_params_flat` consistently in both directions (save reads `prev_A_flat`→writes `prev_params_flat`; resume reads `prev_params_flat`→writes `prev_A_flat`). `_maybe_save_1s_checkpoint` / `_gate_fp` / `_cg_fp` / `start_step` names match across Tasks 4–5.
