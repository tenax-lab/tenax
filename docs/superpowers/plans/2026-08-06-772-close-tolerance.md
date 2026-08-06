# Close #772: recalibrate the residual gate onto the clamped spectrum

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close #772 by setting `root_residual_warn` from the clamp that #778
introduced, and finish the surface around that clamp so the signals it computes
actually reach a caller.

**Architecture:** PR #778 (`2cb2441`) fixed the NaN gradients with a derived
`eps^(1/3)` rank cap in `_rank_capped_spectrum`. It left three loose ends: the
`rel_floor` knob is unreachable from any public entry point, the `usable_rank`
it computes is discarded, and the `root_residual_warn=1e-6` gate rejects the
now-healthy state. This plan closes all three.

**Tech Stack:** Python 3.11+, JAX (x64), pytest, `uv`.

**Worktree:** `/home/yjkao/tenax-772`, branch `fix/772-close-tolerance`, based on
`origin/main` at `2cb2441`. A GPU job is running in the primary worktree
(`/home/yjkao/tenax`) — do not touch it, and do not run GPU tests.

## Global Constraints

- Run everything with `JAX_PLATFORMS=cpu uv run ...` from `/home/yjkao/tenax-772`.
  GPU 0 is occupied by an unrelated job.
- Parameter name is exactly `rel_floor` — matching `_rank_capped_spectrum`'s
  existing keyword. Do NOT introduce a second name.
- Scope is `_ctm_root_implicit_asym.py`, `ipeps_config.py`,
  `ipeps_optimize_root_implicit.py`, and their tests. Do NOT change
  `_ctm_root_implicit_multisite.py`'s behaviour — it shares
  `_rank_capped_spectrum` and must keep working unchanged.
- pre-commit (ruff, ruff-format) must pass before each commit. 18 pre-existing
  ruff errors in `examples/` are unrelated — leave them.
- Do NOT use background jobs, Monitor, or `&`. Foreground only.
- Commit after every task.

## The measurements this plan rests on

All taken in this worktree on `2cb2441`, physical D=2 Heisenberg SU fixture,
`max_iter=300, conv_tol=1e-13`, default clamp `eps^(1/3) = 6.0555e-06`:

| state | χ | `root_residual` | `covariant_residual` | max‖g‖ |
|---|---|---|---|---|
| physical | 4 | 1.255e-06 | 8.493e-06 | 1.953e-02 |
| physical | 6 | 2.808e-06 | 1.911e-05 | 1.953e-02 |
| physical | 8 | 3.767e-06 | 2.567e-05 | 1.953e-02 |
| physical | 12 | 5.178e-06 | 3.529e-05 | 1.953e-02 |
| random | 4 | 2.794e-16 | 2.298e-14 | 1.044e+00 |
| random | 12 | 9.261e-16 | 1.809e-13 | 1.041e+00 |

χ=6 `root_residual` = 2.808e-06 reproduces #778's reported 2.8e-06 exactly.

**The gate does not measure gradient error.** Against a directional finite
difference on the same fixture:

| χ | `covariant_residual` | rel err vs FD |
|---|---|---|
| 4 | 8.493e-06 | 2.94e-07 |
| 12 | 3.529e-05 | 2.80e-07 |

The residual grows 4.2× while the gradient error is flat (marginally *better*
at the worse residual). Under a clamped spectrum the residual measures the
clamp's designed-in inconsistency, not error. That is why the gate is
recalibrated rather than tightened, and the docstring must say so.

---

### Task 1: Make `rel_floor` reachable and surface `usable_rank`

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_asym.py`
- Test: `tests/test_ctm_root_implicit_asym.py`

**Interfaces produced:**
- `all_projectors(env, a, chi, prev=None, *, rel_floor=None)` — returns 4 tuples
  of `(P_top, P_bot, U, S_keep, Vh, usable_rank)` **NO** — see Step 3; the rank
  is returned via a separate accessor, not by widening the projector tuple.
- `sweep(env, a, chi, prev=None, *, rel_floor=None)`
- `converge(A, chi, *, ..., rel_floor=None)`
- `asym_root_parametrize(env, a, chi, *, ..., rel_floor=None)`
- `asym_root_implicit_energy_and_grad(..., rel_floor=None)`
- `retained_rank_report(env, a, chi, rel_floor=None) -> dict` with keys
  `usable_rank` (int, min over the four directions) and `retained_smin_rtol`
  (float, min over directions of `s_min/s_max` measured **before** clamping).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ctm_root_implicit_asym.py`:

```python
def test_rel_floor_reaches_the_projectors():
    """#778 exposed rel_floor on _rank_capped_spectrum but no caller can pass it."""
    A = _site_tensor(D=2)
    env, a = M._init_env(A, 4)
    projs = M.all_projectors(env, a, 4, rel_floor=1e-2)
    for _pt, _pb, _U, S_keep, _Vh in projs:
        d = np.abs(np.diag(np.asarray(S_keep)))
        assert d.min() / d.max() >= 1e-2 * (1 - 1e-9)


def test_rel_floor_reaches_both_the_forward_and_the_root_paths():
    """converge and asym_root_parametrize build projectors independently.

    If only one honours the knob, y* is not a root of the environment it was
    extracted from and the root residual is meaningless.
    """
    A = _site_tensor(D=2)
    env, a, _meta, projs = M.converge(
        A, 4, max_iter=60, conv_tol=1e-12, return_projectors=True, rel_floor=1e-2
    )
    root, _resid = M.asym_root_parametrize(env, a, 4, prev_projs=projs, rel_floor=1e-2)
    for S in root.s:
        d = np.abs(np.diag(np.asarray(S)))
        assert d.min() / d.max() >= 1e-2 * (1 - 1e-9)


def test_usable_rank_is_reported_not_discarded():
    """#778 computes usable_rank and drops it on the floor at all_projectors."""
    A = _site_tensor(D=2)
    env, a = M._init_env(A, 4)
    rep = M.retained_rank_report(env, a, 4)
    assert rep["usable_rank"] == 4
    assert rep["retained_smin_rtol"] > 1e-5
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_asym.py -k "rel_floor_reaches or usable_rank_is_reported" -v
```

Expected: FAIL — `TypeError: all_projectors() got an unexpected keyword argument
'rel_floor'` and `AttributeError: ... has no attribute 'retained_rank_report'`.

- [ ] **Step 3: Thread `rel_floor`**

Add `*, rel_floor: float | None = None` to `all_projectors`, `sweep`, `converge`,
`asym_root_parametrize` and `asym_root_implicit_energy_and_grad`, forwarding it
at every internal call site. `None` means "use `_rank_capped_spectrum`'s derived
`eps^(1/3)`", so passing `None` through is correct and changes nothing.

`all_projectors` line ~517 currently reads:

```python
        s_k, _rank = _rank_capped_spectrum(s, chi)
```

Change to:

```python
        s_k, _rank = _rank_capped_spectrum(s, chi, rel_floor=rel_floor)
```

**Do NOT widen the tuple `all_projectors` appends to `out`.** Several call sites
unpack it as a 5-tuple (`_pt, _pb, U, S_keep, Vh`), including
`asym_root_parametrize` and the existing tests. Widening it is a broad breaking
change for a diagnostic. The rank is exposed via `retained_rank_report` instead.

`asym_root_parametrize` has **two** internal projector-building call sites — an
`all_projectors` call and a `sweep` call, both in its polish loop. Both must
forward `rel_floor`. Missing the second is what
`test_rel_floor_reaches_both_the_forward_and_the_root_paths` catches.

- [ ] **Step 4: Add `retained_rank_report`**

Immediately after `all_projectors`:

```python
def retained_rank_report(
    env: AsymEnv, a: jax.Array, chi: int, rel_floor: float | None = None
) -> dict:
    """How much of the retained spectrum carries real weight, before clamping.

    ``usable_rank`` is the minimum over the four directions of the count
    :func:`_rank_capped_spectrum` would leave above the clamp; when it is below
    ``chi`` the extra directions are numerically empty and ``chi`` exceeds what
    the state's environment supports.  ``retained_smin_rtol`` is the true
    ``s_min/s_max`` of the *unclamped* cut, so it describes the state rather
    than reading back ``rel_floor``.

    #778 computes the rank inside :func:`all_projectors` and discards it, which
    left the signal it describes as "the honest signal that chi exceeds what
    the state's environment supports" unreachable.  This is that accessor.
    """
    smin_rtol = float("inf")
    rank = chi
    env_k, a_k = env, a
    for _k in range(4):
        M_k = half_infinite_environment(env_k, a_k)
        s = jnp.linalg.svd(M_k, compute_uv=False)
        _capped, usable = _rank_capped_spectrum(s, chi, rel_floor=rel_floor)
        s_k = s[:chi]
        smin_rtol = min(smin_rtol, float(s_k[-1]) / (float(s_k[0]) + 1e-300))
        rank = min(rank, int(usable))
        env_k, a_k = rotate_env(env_k), rotate_a(a_k)
    return {"usable_rank": rank, "retained_smin_rtol": smin_rtol}
```

- [ ] **Step 5: Wire it into the diagnostics**

In `asym_root_implicit_energy_and_grad`, after the `converge` call that produces
`env, a_arr, meta, forward_projs`:

```python
    rank_report = retained_rank_report(env, a_arr, chi, rel_floor)
    if rank_report["usable_rank"] < chi:
        warnings.warn(
            f"Asymmetric root implicit AD: only "
            f"{rank_report['usable_rank']} of {chi} retained directions carry "
            f"weight above the rank clamp (smallest is "
            f"{rank_report['retained_smin_rtol']:.2e} relative), so chi={chi} "
            "exceeds this state's usable environment rank. The gradient is "
            "sound, but the energy will stop moving with chi -- which looks "
            "like the #723/#726 rank-1 collapse and is not. Reduce chi, or "
            "increase D.",
            RuntimeWarning,
            stacklevel=2,
        )
```

and merge `**rank_report` into the `return_diagnostics` dict alongside `**meta`.

- [ ] **Step 6: Run to verify they pass, then the whole file**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_asym.py -v
```

Expected: all pass. `rel_floor=None` everywhere means no number moves; a failure
here means a threading step passed the wrong value.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(#772): make rel_floor reachable and report usable_rank

#778 exposed rel_floor on _rank_capped_spectrum but no public entry point can
pass it, and computed usable_rank at all_projectors only to discard it -- so
the signal its own docstring calls 'the honest signal that chi exceeds what the
state's environment supports' reached no caller.

Thread rel_floor through both paths that build projectors (converge/sweep and
asym_root_parametrize; they must agree or y* is not a root of the environment
it came from), and add retained_rank_report as the accessor.  Defaults are
None throughout, so no number moves.

Refs #772."
```

---

### Task 2: Add the physical-state fixture

`tests/_su_fixtures.py` and `scripts/gen_su_fixture.py` are **already present in
the worktree, untracked** — ported from the superseded branch. They need no
edits. `main` has no physical-state fixture at all: #778's tests use random
tensors and unit-test the clamp function directly, which is the same gap that
let #772 reach the wiring stage.

**Files:**
- Add (already on disk): `tests/_su_fixtures.py`, `scripts/gen_su_fixture.py`
- Test: `tests/test_ctm_root_implicit_asym.py`

- [ ] **Step 1: Write the tests**

```python
from tests._su_fixtures import PHYSICAL_SU_D2_E_SU, physical_su_d2


@pytest.mark.parametrize("chi", [4, 6, 8])
def test_the_physical_state_exhausts_its_usable_rank(chi):
    """The fixture's whole point: a real state whose environment is rank-poor.

    The random tensor the rest of this file uses never gets near the clamp,
    which is why #772 reached the wiring stage undetected.
    """
    A = physical_su_d2()
    env, a, _meta, _projs = M.converge(
        A, chi, max_iter=100, conv_tol=1e-11, return_projectors=True
    )
    rep = M.retained_rank_report(env, a, chi)
    assert rep["usable_rank"] < chi
    assert rep["retained_smin_rtol"] < 1e-7


@pytest.mark.parametrize("chi", [4, 6, 8])
def test_the_random_fixture_keeps_its_full_rank(chi):
    """The contrast that hid the bug."""
    A = _site_tensor(D=2)
    env, a, _meta, _projs = M.converge(
        A, chi, max_iter=100, conv_tol=1e-11, return_projectors=True
    )
    rep = M.retained_rank_report(env, a, chi)
    assert rep["usable_rank"] == chi
    assert rep["retained_smin_rtol"] > 1e-5


@pytest.mark.slow
def test_the_frozen_fixture_still_matches_simple_update():
    """The frozen literal must stay the state it claims to be.

    Compares *physically*: a simple-update tensor is defined only up to a bond
    gauge, so an element-wise diff would be wrong even when nothing drifted.
    """
    import sys

    sys.path.insert(0, "scripts")
    from gen_su_fixture import build

    E_live, A_live = build()
    assert E_live == pytest.approx(PHYSICAL_SU_D2_E_SU, abs=1e-6)
    env_l, a_l, _m, _p = M.converge(
        A_live, 4, max_iter=100, conv_tol=1e-11, return_projectors=True
    )
    env_f, a_f, _m, _p = M.converge(
        physical_su_d2(), 4, max_iter=100, conv_tol=1e-11, return_projectors=True
    )
    live = M.retained_rank_report(env_l, a_l, 4)["retained_smin_rtol"]
    frozen = M.retained_rank_report(env_f, a_f, 4)["retained_smin_rtol"]
    assert live == pytest.approx(frozen, rel=1e-3)
```

- [ ] **Step 2: Run them**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_asym.py -k "physical_state_exhausts or random_fixture_keeps or frozen_fixture" -v
```

Expected: 7 passed. If `test_the_physical_state_exhausts_its_usable_rank` fails,
the fixture is not the state #772 measured — regenerate with
`scripts/gen_su_fixture.py` and check `E_su ≈ -0.5517095758652025`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test(#772): add the physical-state fixture main lacks

#778's tests use random tensors and unit-test the clamp directly, so nothing on
main exercises a real physical state -- the same gap that let the NaN gradients
reach the wiring stage.  Freeze a D=2 Heisenberg simple-update ground state and
pin the contrast: it exhausts its usable rank at every chi, the random fixture
never does.

Frozen rather than regenerated so the guard costs seconds, with a slow test
comparing live-vs-frozen physically (energy + spectral pathology) rather than
element-wise, since the tensor is defined only up to a bond gauge.

Refs #772."
```

---

### Task 3: Recalibrate the gate and close #772

**Files:**
- Modify: `src/tenax/algorithms/_ctm_root_implicit_asym.py`
- Modify: `tests/test_root_implicit_wiring.py`
- Test: `tests/test_ctm_root_implicit_asym.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("chi", [4, 6, 8, 12])
def test_the_default_gate_admits_a_healthy_clamped_state(chi):
    """#772: the gate rejected the state the clamp had just made healthy."""
    A = physical_su_d2()
    _E, _g, d = M.asym_root_implicit_energy_and_grad(
        A, _gate(), chi=chi, max_iter=300, conv_tol=1e-13,
        return_diagnostics=True, on_root_residual="raise",
    )
    assert np.all(np.isfinite(np.asarray(_g)))


def test_the_gate_tightens_back_when_the_clamp_is_lowered():
    """The threshold is tied to the clamp, not a loosened constant.

    A caller running an unclamped spectrum should still get the 1e-6 gate,
    where residuals genuinely are ~1e-13.
    """
    assert M._default_root_residual_warn(None) == pytest.approx(6.0555e-04, rel=1e-3)
    assert M._default_root_residual_warn(1e-12) == 1e-6
    assert M._default_root_residual_warn(1e-2) == pytest.approx(1.0)


def test_the_gate_still_rejects_a_genuinely_broken_root():
    """Recalibrated, not disabled: the pre-#778 failure was 1.9e-02."""
    A = physical_su_d2()
    with pytest.raises(M.RootResidualError):
        M.asym_root_implicit_energy_and_grad(
            A, _gate(), chi=6, max_iter=300, conv_tol=1e-13,
            rel_floor=1e-12, on_root_residual="raise",
        )
```

If `RootResidualError` is not exported from this module, import it from wherever
`_report_root_residual` raises it and adjust the reference — do not weaken the
assertion to a bare `Exception`.

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_asym.py -k "default_gate_admits or gate_tightens_back or gate_still_rejects" -v
```

Expected: the first fails with `RootResidualError` (covariant residual 8.5e-06
against a 1e-06 gate); the second fails with `AttributeError`.

- [ ] **Step 3: Implement**

Add above `asym_root_implicit_energy_and_grad`:

```python
def _default_root_residual_warn(rel_floor: float | None) -> float:
    """The residual gate, set from the clamp that produces the residual.

    Clamping the numerically-null tail (#778) leaves an inconsistency of order
    ``rel_floor`` in the characteristic equations *by construction* -- the
    stored ``S`` no longer matches the contraction that reproduces the true
    singular values.  A gate at a fixed ``1e-6`` therefore rejects exactly the
    states the clamp has just made healthy, which is what kept #772 open after
    the NaN gradients were fixed.

    **This gate is a sanity check on the equations, not a proxy for gradient
    accuracy.**  Measured on the physical simple-update fixture, the covariant
    residual grows 4.2x from chi=4 to chi=12 (8.49e-06 -> 3.53e-05) while the
    gradient's error against a directional finite difference is flat, and
    marginally smaller at the worse residual (2.94e-07 -> 2.80e-07).  Do not
    tighten this onto gradient quality; it does not measure it.

    The 100x headroom over ``rel_floor`` covers the observed growth in chi with
    room to spare, and still rejects the pre-#778 failure (1.9e-02 at chi=4) by
    31x.  Tied to the clamp rather than raised to a constant so that lowering
    ``rel_floor`` restores the tight gate, where residuals genuinely are
    ~1e-13.
    """
    if rel_floor is None:
        rel_floor = float(jnp.finfo(jnp.float64).eps ** (1.0 / 3.0))
    return max(1e-6, 100.0 * rel_floor)
```

Change the signature default to `root_residual_warn: float | None = None`, and
resolve it at the top of the function body:

```python
    if root_residual_warn is None:
        root_residual_warn = _default_root_residual_warn(rel_floor)
```

- [ ] **Step 4: Run to verify they pass**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_root_implicit_asym.py -v
```

- [ ] **Step 5: Flip the #772 xfail**

In `tests/test_root_implicit_wiring.py`, delete the `@pytest.mark.xfail(...)`
decorator from `test_production_heisenberg_run_through_optimize_gs_ad`, keep
`@pytest.mark.slow`, and replace its docstring:

```python
    """The production case this wiring exists for: a real Heisenberg state.

    This carried #772 as an xfail through two rounds.  First the covariant
    equations NaN'd the gradient on a physical simple-update state; #778's rank
    clamp fixed that.  What remained was the *gate*: clamping leaves an
    O(rel_floor) inconsistency in the equations by construction, and the fixed
    1e-06 threshold rejected it.  The gate is now set from the clamp.
    """
```

- [ ] **Step 6: Run the wiring tests**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_root_implicit_wiring.py -v
```

Expected: all pass, including the previously-xfailing production run. If it
fails on the asserted energy rather than on a residual or a NaN, report the
actual value — `gs_num_steps=3` from a simple-update start is a weak
optimization and the `abs=5e-3` tolerance may be the wrong assertion, but that
is a judgement call to raise, not to silently widen.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "fix(#772): set the residual gate from the clamp that produces the residual

Clamping the numerically-null tail (#778) leaves an inconsistency of order
rel_floor in the characteristic equations by construction, so the fixed 1e-06
gate rejected exactly the states the clamp had just made healthy.  That, not
the gradient, is what kept #772 open: measured covariant residual 8.49e-06 at
chi=4 rising to 3.53e-05 at chi=12, against a 1e-06 threshold.

The gate is now max(1e-6, 100*rel_floor).  Tied to the clamp rather than raised
to a constant, so lowering rel_floor restores the tight gate where residuals
genuinely are ~1e-13.  It still rejects the pre-#778 failure by 31x.

The docstring records what the measurement showed: this gate is a sanity check
on the equations and NOT a proxy for gradient accuracy.  The covariant residual
grows 4.2x from chi=4 to chi=12 while the gradient's error against a
directional finite difference is flat and marginally smaller at the worse
residual (2.94e-07 -> 2.80e-07).

Closes #772."
```

---

### Task 4: Config surface and the silent NaN mask

**Files:**
- Modify: `src/tenax/algorithms/ipeps_config.py`
- Modify: `src/tenax/algorithms/ipeps_optimize_root_implicit.py`
- Test: `tests/test_root_implicit_wiring.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_ctm_config_carries_the_rank_clamp():
    assert CTMConfig(chi=6).rel_floor is None


def test_ctm_config_rejects_a_nonsensical_rank_clamp():
    with pytest.raises(ValueError, match="rel_floor"):
        CTMConfig(chi=6, rel_floor=0.0)
    with pytest.raises(ValueError, match="rel_floor"):
        CTMConfig(chi=6, rel_floor=1.5)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_root_implicit_wiring.py -k ctm_config_ -v
```

- [ ] **Step 3: Add the field**

Append **after** `ctm_chunk_size` in `CTMConfig` — the tail is where fields go,
to preserve the positional ABI:

```python
    # Relative clamp on the retained CTM spectrum for the root-implicit path.
    # ``None`` (default) uses the derived ``eps**(1/3)``: the covariant
    # characteristic equations depend on ``S`` cubically, so a retained
    # direction below that cannot be resolved in working precision (#772/#778).
    # Two-sided -- raising it above genuinely-weighted directions breaks
    # well-conditioned states.  Consulted by ``ctm_ad_mode="root_implicit"``
    # only.  Appended at the end to preserve positional CTMConfig ABI.
    rel_floor: float | None = None
```

and validate in `__post_init__`, after the `ctm_chunk_size` check:

```python
        if self.rel_floor is not None and not 0.0 < self.rel_floor < 1.0:
            raise ValueError(
                "rel_floor must be None or lie in (0, 1) -- it is a relative "
                f"clamp on the retained spectrum; got {self.rel_floor}"
            )
```

- [ ] **Step 4: Forward it, and make the NaN mask loud**

In `ipeps_optimize_root_implicit.py`, add `rel_floor=ctm_cfg.rel_floor` to the
`asym_root_implicit_energy_and_grad` call inside `_energy_and_grad`.

Then replace line ~274:

```python
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
```

with:

```python
        n_nonfinite = int(jnp.sum(~jnp.isfinite(grads)))
        if n_nonfinite:
            nonfinite_grad_steps += 1
            warnings.warn(
                f"ctm_ad_mode='root_implicit': step {step} produced "
                f"{n_nonfinite} non-finite gradient entries, masked to zero so "
                "the best-so-far state survives. The step is a no-op, so "
                "apparent convergence here is not convergence. Check "
                "usable_rank -- this is the #772 failure shape.",
                RuntimeWarning,
                stacklevel=2,
            )
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
```

Initialise `nonfinite_grad_steps = 0` alongside `prev_energy = float("inf")`, and
add `import warnings` if absent.

`optimize_gs_ad_root_implicit` returns the bare `(A_opt, env, energy)` 3-tuple
that `optimize_gs_ad` uses on every path, so there is **no** info dict to carry
a count and widening the return would break that shared contract. Emit a summary
immediately before the final `A_opt, env = _final_env(best_params)`:

```python
    if nonfinite_grad_steps:
        warnings.warn(
            f"ctm_ad_mode='root_implicit': {nonfinite_grad_steps} of "
            f"{config.gs_num_steps} optimizer steps made no progress because "
            "their gradient was masked. The reported energy is from the best "
            "state actually reached, not from a converged optimization.",
            RuntimeWarning,
            stacklevel=2,
        )
```

- [ ] **Step 5: Run the wiring tests and commit**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest tests/test_root_implicit_wiring.py -v
```

```bash
git add -A && git commit -m "feat(#772): surface rel_floor on CTMConfig, unmask NaN gradients

The rank clamp is a documented policy rather than a constant reachable only by
monkeypatching _rank_capped_spectrum.

The optimizer masked non-finite gradients to zero silently, turning a NaN into
a no-op step reported as progress -- the reason a recurrence of #772 would be
quiet.  Keep the mask (aborting would discard the best-so-far state) but warn
per step and summarise at the end.

Refs #772, #715."
```

---

### Task 5: Documentation and the closing note

**Files:**
- Add (already on disk): `scripts/probe_772_tolerance.py`,
  `scripts/probe_772_gate_vs_gradient.py`
- Modify: `docs/ipeps_ad_paths.md` if it documents `ctm_ad_mode`

- [ ] **Step 1: Check whether the knob needs documenting**

```bash
cd /home/yjkao/tenax-772 && grep -n "ctm_ad_mode\|root_implicit" README.md docs/ipeps_ad_paths.md
```

If `ctm_ad_mode="root_implicit"` is documented in either, add a sentence naming
`rel_floor` and `root_residual_warn`, stating that the gate is set from the
clamp and is not a gradient-accuracy proxy. If neither mentions it, skip — this
is a new field on an already-documented config object, not a new entry point.

- [ ] **Step 2: Run the full non-slow suite**

```bash
cd /home/yjkao/tenax-772 && JAX_PLATFORMS=cpu uv run pytest -m "not slow" -q 2>&1 | tail -20
```

Note any pre-existing failures explicitly rather than assuming they are yours.

- [ ] **Step 3: Commit and open the PR**

Do NOT pass `--delete-branch` — `main` uses a merge queue that deletes the head
branch itself, and the flag closes the PR the moment it enters the queue.
