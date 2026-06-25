# Design: CG Coarse-Grained 1-Site iPEPS Checkpoint/Resume

**Date:** 2026-06-25
**Branch:** `feat/ipeps-checkpoint-1site-cg` (off `origin/main`)
**Issue:** closes part of #497 (1-site checkpoint wiring), targeting the
coarse-grained (`cg_gates`) path.
**Status:** Approved design → ready for implementation plan.

## Purpose

iPEPS ground-state-AD checkpoint/resume is currently wired **only for
`unit_cell="2site"`** (a dispatch guard in `optimize_gs_ad` raises
`NotImplementedError` for any other unit cell). This blocks long, crash-/
preemption-resilient runs on the **1-site path** — including the **coarse-grained
(CG) supersite** path (`cg_gates`, e.g. kagome / honeycomb PESS), which is
*physically* multi-site but *computationally* runs on `unit_cell="1x1"`
(`cg_gates` requires `unit_cell="1x1"`, enforced in `iPEPSConfig.__post_init__`).

This work extends checkpoint/resume to the **1-site path**, with first-class
support for the **CG coarse-grained** case, so a multi-hour CG production run can
checkpoint and resume. It mirrors the existing, tested 2-site implementation.

## Scope

**In scope**
- 1-site checkpoint/resume wiring in `_optimize_gs_ad_tensor` (the
  `unit_cell="1x1"` path). This covers BOTH plain 1-site and CG-1-site, since
  they share that function.
- CG-specific handling: `cg_gates` fingerprinting (resume validation) + a
  `_config_to_dict` fix so a config carrying `cg_gates` can be snapshotted.
- Narrow the dispatch guard from "only 2site" to "2site **or** 1x1".
- Tests mirroring the 2-site checkpoint tests, plus CG-specific cases.

**Out of scope** (separate follow-ups)
- Generic `Lattice` multisite (`_optimize_gs_ad_multisite`) + its
  `Lattice.neighbor_map` (`MappingProxyType`) serialization fix — stays guarded.
- C4v 1-site (`_optimize_gs_ad_tensor_reference_c4v`).
- Extracting a shared checkpoint-wiring helper across 2-site/1-site (see
  Approach).

## Approach (decisions)

**Wiring — COPY the 2-site template (not extract a shared helper).** The 2-site
path has a proven ~23-key bundle + resume block + a `_maybe_save_*_checkpoint`
closure + save sites. We replicate that into `_optimize_gs_ad_tensor` rather than
refactor the working/tested 2-site path into a shared helper. The two loops
differ in small ways (params shape, stall-recovery default, CG params); a
premature abstraction would leak. A shared `_checkpoint_wiring` helper is a
reasonable *later* refactor once both paths exist and the common shape is
obvious.

**CGGates — FINGERPRINT, don't serialize.** `cg_gates` (`CGGates`) holds
`jnp.ndarray` fields (`h_intra`, `h_inter`) and callables (`map_fn`, `init_fn`).
`dataclasses.asdict` on it produces nested arrays that break `validate_config`'s
dict-equality, and callables/mappingproxy break pickling. On resume the user
re-passes `config` (with live `cg_gates`), so the bundle never needs to serialize
`cg_gates` — it only needs to **validate** that the resumed CG Hamiltonian
matches the checkpointed one. We therefore replace `cg_gates` in the config
snapshot with an **array fingerprint** and validate that on resume.

## Architecture / components

### 1. `src/tenax/algorithms/_checkpoint.py`

**`cg_gates_fingerprint(cg_gates) -> tuple`** — new, mirrors `gate_fingerprint`.
Hashes a stable digest of the array content + structure:
`(n_sites, gate_fingerprint(h_intra), tuple(sorted (label, gate_fingerprint(arr)) for label, arr in h_inter.items()))`.
Callables (`map_fn`, `init_fn`) are intentionally NOT hashed — they come from the
fresh config on resume, and two configs with the same gate arrays but different
(equivalent) closures should still resume.

**`_config_to_dict` fix** — before `asdict`, if `config.cg_gates is not None`,
snapshot a shallow copy with the `cg_gates` field replaced by
`("__cg_gates_fingerprint__", cg_gates_fingerprint(cg_gates))`. This makes the
config dict picklable and `validate_config`-comparable (the fingerprint is a
tuple of hashables). The existing `_FATAL_CONFIG_FIELDS` (`max_bond_dim`,
`unit_cell`, `gs_c4v`, `gs_implicit_ad`) are unaffected; a `cg_gates` change
surfaces as a soft-field mismatch in the snapshot AND is caught fatally by the
explicit fingerprint check (below).

### 2. `src/tenax/algorithms/ipeps_optimize.py`

**Narrow the guard** (currently ~line 916): raise `NotImplementedError` only when
`gs_checkpoint_path is not None` and `unit_cell` is neither `"2site"` nor
`"1x1"`. Update the message to name `Lattice` multisite as the remaining gap.

**Wire `_optimize_gs_ad_tensor`** (the 1-site function), mirroring 2-site:
- **Startup:** compute `_gate_fp = gate_fingerprint(gate)` and, if
  `config.cg_gates is not None`, `_cg_fp = cg_gates_fingerprint(config.cg_gates)`
  — both only when `gs_checkpoint_path is not None`.
- **Resume block** (`if config.gs_resume:`): fail-fast on missing checkpoint
  (`FileNotFoundError`); `load_checkpoint`; `validate_config`; reject gate
  fingerprint mismatch (`ValueError`); reject `cg_gates` fingerprint mismatch
  (`ValueError`, only when a CG fingerprint was saved); restore the optimizer
  state with the same opt-compat logic as 2-site (discard `opt_state`/history if
  `gs_optimizer`/`gs_metric_precond` changed); restore `params`, `best_params`,
  `best_energy`, `prev_energy`, `_env_cache`, `best_env_cache`, `stall_count`,
  schedule/stage state (`current_stage_idx`, `stage_start_step`, chi, conv_tol,
  patience); set `start_step = bundle["step"] + 1`; loop `for step in
  range(start_step, config.gs_num_steps)`.
- **`_maybe_save_1s_checkpoint(step, chi_before, e_prev, *, force_last=False)`** —
  a closure mirroring `_maybe_save_2s_checkpoint`, building the bundle from the
  1-site state. The bundle adds `"cg_gates_fingerprint": _cg_fp` (or `None`).
  `params` is stored as-is — a single `Tensor`, a CG supersite tensor, or the
  raw-params tuple when `cg_gates.map_fn` is set (all picklable).
- **Save sites:** end-of-step, plus the stage-advance / bump intercepts that the
  1-site loop already has (mirror the 2-site `force_last=True` calls).

**Bundle keys** (1-site) — the 2-site key set minus the B-tensor specifics, plus
the CG fingerprint:
`step, config, hamiltonian_fingerprint, cg_gates_fingerprint, params, best_params,
best_energy, prev_energy, env_cache, best_env_cache, opt_state, lbfgs_history,
prev_params_flat, prev_grad_flat, cg_direction, prev_grad, prev_precond_grad,
stall_count, current_stage_idx, stage_start_step, ctm_cfg_chi, current_conv_tol,
current_patience`.

### 3. Config

No new fields — `gs_checkpoint_path`, `gs_checkpoint_every`, `gs_resume` already
exist and validate in `__post_init__`.

## Data flow

```
optimize_gs_ad(gate, A_init, config)            config.cg_gates = kagome_cg_gates(...)
  guard: 1x1 allowed -> _optimize_gs_ad_tensor   unit_cell="1x1", gs_checkpoint_path=DIR
    startup: _gate_fp, _cg_fp
    if gs_resume: load -> validate_config -> check _gate_fp,_cg_fp -> restore state, start_step
    for step in range(start_step, N):
        value_and_grad -> update env_cache -> track best -> optimizer step
        _maybe_save_1s_checkpoint(step, ...)   # writes ckpt.last.pkl every K; ckpt.best.pkl on improve
  return (A_opt, env, E_gs[, history])
```

## Error handling / edge cases

- **Missing checkpoint on resume:** `FileNotFoundError` (fail fast), as 2-site.
- **Gate or `cg_gates` swap:** `ValueError` with a clear message — refuse to
  resume a CG run against different coarse-grained gates.
- **Optimizer/precond change:** discard `opt_state` + curvature history, warn,
  continue (2-site semantics).
- **`cg_gates.map_fn` set:** `params` is the raw-params tuple; the bundle pickles
  it directly; resume rebuilds the CG tensor via the fresh `config.cg_gates.map_fn`.
- **Fatal config mismatch** (`max_bond_dim` etc.): `validate_config` raises, as
  2-site.
- **Atomic writes / schema version:** unchanged — reuse `save_checkpoint`'s
  `.tmp`→`os.replace` and the schema guard.

## Testing

Mirror the existing checkpoint tests; all fast (D=2, χ=4, 1-site).

**`tests/test_ipeps_checkpoint.py`** (core, unit):
- `test_cg_gates_fingerprint_round_trip_and_changes_with_bytes` — same arrays →
  same FP; perturbed `h_intra`/`h_inter` → different FP; callable differences do
  NOT change FP.
- `test_config_to_dict_handles_cg_gates` — a config with `cg_gates` round-trips
  through `_config_to_dict` (picklable, no array-truth-value error) and
  `validate_config` accepts identical / flags changed `cg_gates`.

**`tests/test_ipeps_checkpoint_resume.py`** (algorithm, end-to-end):
- Update `test_checkpoint_path_rejects_non_2site_paths` → rename/split:
  `unit_cell="1x1"` no longer raises; `unit_cell=Lattice(...)` still raises
  `NotImplementedError`.
- `test_resume_1site_continues_from_saved_step` — plain 1-site: run 2 steps,
  resume to 4, assert final step index + monotone energy (mirror 2-site).
- `test_resume_cg_1site_continues_from_saved_step` — CG path
  (`kagome_xxz_pess_cg_gates` / `map_fn`), run→checkpoint→resume, assert the
  resumed trajectory matches an uninterrupted run to the same step.
- `test_resume_rejects_different_cg_gates` — checkpoint with one `cg_gates`,
  resume with a different one (perturbed arrays) → `ValueError`.
- `test_resume_1site_with_missing_checkpoint_raises` — `FileNotFoundError`.

## Reference

The 2-site implementation is the template: `_optimize_gs_ad_2site` /
`_optimize_gs_ad_tensor_2site` in `ipeps_optimize.py` (resume block, the
`_maybe_save_2s_checkpoint` closure + its 4 save sites, the 23-key bundle), the
generic `_checkpoint.py` API (`save_checkpoint`/`load_checkpoint`/`validate_config`/
`gate_fingerprint`/`checkpoint_exists`), and the test pattern in
`tests/test_ipeps_checkpoint_resume.py`.
