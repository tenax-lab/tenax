# Flow-gradient slope check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject a flowing-projector gradient on a non-stationary forward when one extra forward energy evaluation along its descent direction disagrees with what the gradient predicts (#1111).

**Architecture:**
- `f_fwd` adds the forward energy `E0` and a `nonstationary` flag to the custom_vjp residuals.
- After `f_bwd` accepts a flow gradient by the existing #1028 rules, a new closure `_flow_slope_check` runs on non-stationary forwards. It evaluates `E(x - h G/|G|)` through a module-level helper `_slope_probe_energy`, isolated from diagnostics, warnings and run-state latches, and computes `r = ΔE / (-h|G|)`.
- If `r` is outside `[0.5, 2.0]`, or the check is inconclusive, the existing frozen fallback runs with a new reason.

**Tech Stack:** Python 3.11+, JAX, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-flow-slope-check-design.md`

## Global Constraints

- Branch `fix/1111-flow-slope-check` in `~/tenax-1111`, built on `feat/gmres-single-matvec`. Never touch `/home/yjkao/tenax`.
- Acceptance band: keep flow iff `0.5 <= r <= 2.0` (`_FLOW_SLOPE_RATIO_MIN = 0.5`, `_FLOW_SLOPE_RATIO_MAX = 2.0`). Step: `h = 1e-4 * ||x||` (`_FLOW_SLOPE_STEP = 1e-4`).
- The check runs only for `projector_backward == "flow"`, after the existing acceptance tests pass, and only when the forward was non-stationary (`not (forward_stationarity_residual <= forward_stationarity_threshold)`, which fails closed on NaN).
- Inconclusive cases fall back to frozen: the extra forward raises, `E_h` or `r` is non-finite, the extra forward's `chi_post` differs, or `|G| == 0` while `g != 0`.
- If `g == 0`, keep flow and do not run the check.
- The extra forward leaves the diagnostics restored exactly, emits no warnings, and leaves the run-state `*_warned` latches unchanged.
- Stationary forwards and `projector_backward != "flow"` behave exactly as before, bit for bit.
- Run tests with: `cd ~/tenax-1111 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider <files>`
- pre-commit: `cd ~/tenax-1111 && PATH=$HOME/.local/bin:$PATH pre-commit run --files <files>`
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg`.
- Do not push and do not open a PR.

## Review Focus

1. **A backward called twice for the same forward.** Under `jax.grad` this does not happen, but `jax.vjp` lets a caller apply the pullback twice. The residuals carry `E0` and the flag, so a second call must give the same verdict. Pinned in Task 1 by `test_pullback_twice_gives_the_same_verdict`.
2. **The diagnostics a user reads after `jax.grad`.** They must describe the real forward, not the probe forward. Pinned by `test_forward_diagnostics_survive_the_check`.
3. **A stale `flow_slope_ratio`.** It must be cleared at the start of every `f_bwd`, or a stationary call after a checked one would report the old ratio. Pinned by `test_stationary_forward_skips_the_check` (it runs a checked call first, then a stationary one).
4. **Complex parameters.** Norms use `abs(leaf)**2`, and the energy is taken as `jnp.real`. Covered by the code; the existing complex-gradient PESS tests in the regression set exercise the flow path.
5. **`optimize_gs_ad` end to end.** One optimizer step on the #1028 fixture must not crash or warn more than once. Pinned by `test_optimize_gs_ad_step_on_the_1028_fixture`.

---

### Task 1: The slope check and its tests

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`. Edit these places:
  - the module constants near `_PROJECTOR_BACKWARD_FLOW` (about line 241);
  - a new module-level `_slope_probe_energy`;
  - `f_fwd` (about line 1624);
  - `_bwd_core`'s residual unpack (about line 1991);
  - a new closure `_flow_slope_check` next to `f_bwd`;
  - `f_bwd` (about lines 2295-2390);
  - the diagnostics key list (about line 1110).
- Modify: `tests/test_projector_backward_default_1028.py`. Replace the strict xfail `test_flow_gradient_on_the_1028_fixture_is_wrong_issue_1111`.
- Create: `tests/test_flow_slope_check_1111.py`
- Modify: `tests/conftest.py`. Add `"test_flow_slope_check_1111.py": "algorithm",` next to `"test_projector_backward_default_1028.py"`. If that file is marked `"core"`, use `"core"` to match.

**Interfaces:**
- Produces, at module level in `_ctm_energy_ad.py`:
  - `_FLOW_SLOPE_RATIO_MIN = 0.5`, `_FLOW_SLOPE_RATIO_MAX = 2.0`, `_FLOW_SLOPE_STEP = 1e-4`;
  - `_slope_probe_energy(run_forward, compute_energy, site_tensors) -> tuple[float, int]`, which returns `(real energy, chi_post)`. Tests monkeypatch it.
- Residuals tuple: `(params_data_tuple, env_leaves, chi_post, run_token, energy0, nonstationary)`.
- Diagnostics: new key `flow_slope_ratio` (float). It is present only when the check ran and gave a finite `r`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_flow_slope_check_1111.py`:

```python
"""#1111: a solved flowing-projector adjoint on a non-stationary forward can
give a gradient orders of magnitude off finite differences (1497x on the
#1028 fixture, 86x at D=3 chi=12).  The backward now checks it with one extra
forward energy evaluation along the gradient's own descent direction:
r = dE_measured / dE_predicted must lie in [0.5, 2].  Measured: garbage flow
r <= 0.002, good flow 0.97-1.24, frozen 1.00-1.54."""

from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_energy_ad as cea
from tenax.algorithms._ctm_energy_ad import (
    ctm_energy_implicit,
    get_last_implicit_ad_diagnostics,
)
from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
)
from tenax.algorithms.ipeps import heisenberg_gate, heisenberg_u1sz_init_pair
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor


def _heis():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = Sp.T
    H = jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


_H1 = _heis()
_G2 = heisenberg_gate().todense()


def _pair_params(D=2, seed=0):
    A, B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(seed))
    return (jnp.asarray(A.todense()), jnp.asarray(B.todense()))


def _pair_energy(p, *, chi=8, max_iter=20, pb="flow"):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(p[0]), (1, 0): _wrap_as_dense_tensor(p[1])},
            CHECKERBOARD_NEIGHBORS,
            _G2,
            chi=chi,
            max_iter=max_iter,
            projector_method="svd",
            forward_gauge="bond_phase",
            conv_method="elementwise",
            projector_backward=pb,
        )
    )


def _single_params(seed=2026):
    A = jax.random.normal(jax.random.PRNGKey(seed), (2, 2, 2, 2, 2))
    return (A / jnp.linalg.norm(A),)


def _single_energy(p, *, chi=4, max_iter=20, conv_tol=1e-10, pb="flow"):
    return jnp.real(
        ctm_energy_implicit(
            {(0, 0): _wrap_as_dense_tensor(p[0])},
            SINGLE_SITE_NEIGHBORS,
            _H1,
            chi=chi,
            max_iter=max_iter,
            conv_tol=conv_tol,
            forward_gauge="bond_phase",
            projector_backward=pb,
        )
    )


def _grad(energy, params, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(lambda p: energy(p, **kw))(params)
    return g, dict(get_last_implicit_ad_diagnostics())


def _all_latches():
    out = {}
    for key, entry in cea._VJP_CACHE.items():
        if not isinstance(entry, tuple):
            continue
        _f, mutables = entry
        for rkey, (token, state, _objs) in mutables.get("_runs", {}).items():
            out[(id(key), token)] = {k: v for k, v in state.items() if k.endswith("_warned")}
    return out


def test_slope_check_rejects_the_1028_flow_gradient():
    p = _pair_params()
    g, d = _grad(_pair_energy, p)
    assert d["forward_stationarity_residual"] > d["forward_stationarity_threshold"], "premise"
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert "slope check" in d["flow_fallback_reason"], d["flow_fallback_reason"]
    assert d["flow_slope_ratio"] < 0.5, d["flow_slope_ratio"]
    g_frozen, _ = _grad(_pair_energy, p, pb="auto")
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_frozen)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_slope_check_keeps_a_good_flow_gradient():
    """The chi=4 limit cycle: non-stationary (0.15) but flow is the better
    gradient (FD error 1.4% / 0.5% against frozen's 6% / 19%), r = 1.002."""
    _, d = _grad(_single_energy, _single_params())
    assert d["forward_stationarity_residual"] > d["forward_stationarity_threshold"], "premise"
    assert d["projector_backward_used"] == "flow", d
    assert 0.5 <= d["flow_slope_ratio"] <= 2.0, d["flow_slope_ratio"]


def test_stationary_forward_skips_the_check(monkeypatch):
    # A checked call first, so a stale flow_slope_ratio would be visible.
    _grad(_single_energy, _single_params())
    assert "flow_slope_ratio" in get_last_implicit_ad_diagnostics()

    def must_not_run(*a, **k):
        raise AssertionError("the slope check ran on a stationary forward")

    monkeypatch.setattr(cea, "_slope_probe_energy", must_not_run)
    _, d = _grad(_single_energy, _single_params(), chi=8, max_iter=400, conv_tol=1e-11)
    assert d["forward_stationarity_residual"] <= d["forward_stationarity_threshold"], "premise"
    assert d["projector_backward_used"] == "flow"
    assert "flow_slope_ratio" not in d


def test_forward_diagnostics_survive_the_check():
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _single_energy(p)  # forward only
    before = dict(get_last_implicit_ad_diagnostics())
    _, after = _grad(_single_energy, p)
    assert "flow_slope_ratio" in after, "premise: the check ran"
    for k in ("forward_stationarity_residual", "forward_iterations", "forward_converged"):
        assert after[k] == before[k], (k, before[k], after[k])


def test_inconclusive_check_falls_back_to_frozen(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe forward failed")

    monkeypatch.setattr(cea, "_slope_probe_energy", boom)
    _, d = _grad(_single_energy, _single_params())
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert "could not evaluate" in d["flow_fallback_reason"], d["flow_fallback_reason"]
    assert "probe forward failed" in d["flow_fallback_reason"]
    assert "flow_slope_ratio" not in d


def test_the_check_spends_no_latch_and_emits_no_warning(monkeypatch):
    real = cea._slope_probe_energy

    def noisy(*a, **k):
        warnings.warn("probe-forward warning", RuntimeWarning)
        return real(*a, **k)

    monkeypatch.setattr(cea, "_slope_probe_energy", noisy)
    p = _single_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _single_energy(p)  # make sure the cache entry and run exist
    latches_before = _all_latches()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.grad(lambda q: _single_energy(q))(p)
    assert "flow_slope_ratio" in get_last_implicit_ad_diagnostics(), "premise: the check ran"
    assert not any("probe-forward warning" in str(w.message) for w in caught)
    latches_after = _all_latches()
    for k, v in latches_after.items():
        if k in latches_before:
            assert v == latches_before[k], (k, latches_before[k], v)


def test_pullback_twice_gives_the_same_verdict():
    p = _pair_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pullback = jax.vjp(lambda q: _pair_energy(q), p)
        g1 = pullback(jnp.ones(()))
        d1 = dict(get_last_implicit_ad_diagnostics())
        g2 = pullback(jnp.ones(()))
        d2 = dict(get_last_implicit_ad_diagnostics())
    assert d1["projector_backward_used"] == d2["projector_backward_used"] == "frozen_fallback"
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_optimize_gs_ad_step_on_the_1028_fixture():
    from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
    from tenax.algorithms.ipeps_optimize import optimize_gs_ad

    A, B = _pair_params()
    config = iPEPSConfig(
        max_bond_dim=2,
        unit_cell="2site",
        gs_num_steps=1,
        ctm=CTMConfig(chi=8, max_iter=20, projector_method="svd", conv_method="elementwise"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize_gs_ad(heisenberg_gate(), (A, B), config)
    flow_warnings = [w for w in caught if "flowing-projector backward" in str(w.message)]
    assert len(flow_warnings) <= 1
    energy = result[0] if isinstance(result, tuple) else getattr(result, "energy", None)
    assert energy is None or np.isfinite(float(energy))


@pytest.mark.slow
def test_slope_check_rejects_the_d3_chi12_flow_gradient():
    """The hole #1111 found on main: D=3 chi=12 flow gradient 86x off FD,
    r = 0.002."""
    _, d = _grad(_pair_energy, _pair_params(D=3), chi=12)
    assert d["projector_backward_used"] == "frozen_fallback", d
    assert d["flow_slope_ratio"] < 0.5
```

In `tests/test_projector_backward_default_1028.py`, delete the whole
`test_flow_gradient_on_the_1028_fixture_is_wrong_issue_1111` function and its
xfail decorator. Its replacement is
`test_slope_check_rejects_the_1028_flow_gradient` in the new file. A
finite-difference assertion cannot pass on that fixture, because frozen is
itself 1.5x off on that direction (the #1028 bias).

Register the new file in `tests/conftest.py` with the marker used by
`test_projector_backward_default_1028.py`.

Before running, check `optimize_gs_ad`'s real signature and its `iPEPSConfig`
fields: `grep -n "def optimize_gs_ad" -A30 src/tenax/algorithms/ipeps_optimize.py`.
Adapt `test_optimize_gs_ad_step_on_the_1028_fixture` to that signature. The
gate argument, the initial-state form and the result type may differ from the
sketch above. Keep its two assertions: at most one flowing-projector warning,
and a finite energy.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/tenax-1111 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider -m "not slow" tests/test_flow_slope_check_1111.py`

Expected:
- `test_slope_check_rejects_the_1028_flow_gradient` FAILS (`used == "flow"`).
- The monkeypatch tests ERROR or FAIL (`_slope_probe_energy` does not exist).
- `test_slope_check_keeps_a_good_flow_gradient` FAILS on the missing `flow_slope_ratio` key.

- [ ] **Step 3: Module constants and the probe helper**

In `src/tenax/algorithms/_ctm_energy_ad.py`, directly after
`_PROJECTOR_BACKWARD_FROZEN = "auto"`:

```python
# #1111: a flowing-projector gradient on a non-stationary forward is checked
# against one extra forward energy evaluation along its own descent
# direction.  r = (E(x - h G/|G|) - E(x)) / (-h |G|) must lie in
# [MIN, MAX].  Measured on 15 fixtures (h = 1e-4, CPU): garbage flow
# gradients (1497x / 7128x / 86x off finite differences, the last at D=3
# chi=12) gave r = 0.000 / 0.001 / 0.002; good flow 0.97-1.24; frozen
# 1.00-1.54.  The upper bound catches a gradient that is too small, the
# symmetric failure.
_FLOW_SLOPE_RATIO_MIN = 0.5
_FLOW_SLOPE_RATIO_MAX = 2.0
_FLOW_SLOPE_STEP = 1e-4  # h = _FLOW_SLOPE_STEP * ||x||


def _slope_probe_energy(run_forward, compute_energy, site_tensors):
    """``(real energy, chi_post)`` of the implicit-AD forward at ``site_tensors``.

    The #1111 slope check's one extra forward.  Module level so tests can
    replace it; it is the same ``_run_forward`` + ``_compute_energy`` pair the
    forward itself uses, so the check evaluates exactly the differentiated
    function.
    """
    envs, chi_post = run_forward(site_tensors)
    return float(jnp.real(compute_energy(site_tensors, envs))), int(chi_post)
```

Add `"flow_slope_ratio",` to the tuple of diagnostics keys near
`"forward_stationarity_threshold",` (about line 1142), keeping that tuple's
ordering style. Add a bullet to the diagnostics docstring near the
`forward_stationarity_*` bullets:
`* ``flow_slope_ratio`` -- #1111: measured/predicted energy change along the flowing gradient's descent direction; present only when the slope check ran (non-stationary forward, flow accepted by the other tests) and gave a finite value.`

- [ ] **Step 4: Residuals from `f_fwd`**

Replace the body of `f_fwd` after `energy = _compute_energy(site_tensors, envs)` with:

```python
        _cached["env_treedef"] = jax.tree.structure(envs)
        env_leaves = tuple(jax.tree.leaves(envs))
        # #1111: the slope check needs this forward's energy and whether it was
        # stationary.  Read now, right after _run_forward wrote them, and carry
        # them in the residuals -- another forward can overwrite the
        # module-global diagnostics before this backward runs.  Written as
        # ``not (r <= t)`` so a NaN residual counts as non-stationary.
        nonstationary = not (
            _F3_LAST_DIAGNOSTICS.get("forward_stationarity_residual", float("nan"))
            <= _F3_LAST_DIAGNOSTICS.get("forward_stationarity_threshold", 0.0)
        )
        # The run token travels with the residuals, so a backward finds its
        # own run even if another forward ran in between.
        residuals = (
            params_data_tuple,
            env_leaves,
            chi_post,
            mutables["run_token"],
            energy,
            nonstationary,
        )
        return energy, residuals
```

In `_bwd_core`, change
`params_data_tuple, env_leaves, chi_post, run_token = residuals` to
`params_data_tuple, env_leaves, chi_post, run_token = residuals[:4]`.

Then grep the file for any other unpacking of `residuals`:
`grep -n "= residuals\b\|residuals\[" src/tenax/algorithms/_ctm_energy_ad.py`.
`f_bwd` uses `residuals[3]`, which is unchanged. Fix any other full unpack the
same way.

- [ ] **Step 5: The check closure and its use in `f_bwd`**

Directly above `def f_bwd(residuals, g):`, add:

```python
    def _flow_slope_check(residuals, g, grads, run):
        """#1111: ``None`` to keep the flowing gradient, else why to discard it.

        Runs only on a non-stationary forward.  One extra forward energy
        evaluation along the gradient's own descent direction must change the
        energy by what the gradient predicts: r = dE_measured / dE_predicted
        in [_FLOW_SLOPE_RATIO_MIN, _FLOW_SLOPE_RATIO_MAX].  An inconclusive
        check (the probe raised, a non-finite value, a chi bump) discards flow
        too -- frozen is biased (#1028) but bounded.  The probe leaves the
        diagnostics, the warnings and the run's once-per-run latches as it
        found them.
        """
        params_data_tuple, _env_leaves, chi_post, _token, energy0, nonstationary = (
            residuals
        )
        if not nonstationary:
            return None
        g_arr = jnp.asarray(g)
        if float(jax.device_get(jnp.abs(g_arr))) == 0.0:
            return None

        def _norm(tree):
            return float(
                jax.device_get(
                    jnp.sqrt(
                        sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree.leaves(tree))
                    )
                )
            )

        G = jax.tree.map(lambda leaf: leaf / g_arr, grads[0])
        g_norm = _norm(G)
        if not g_norm > 0.0:
            return "the slope check could not evaluate (the energy gradient is zero)"
        h = _FLOW_SLOPE_STEP * _norm(params_data_tuple)
        x_h = jax.tree.map(lambda a, b: a - (h / g_norm) * b, params_data_tuple, G)

        diag_snapshot = dict(_F3_LAST_DIAGNOSTICS)
        latch_snapshot = {k: v for k, v in run.items() if k.endswith("_warned")}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e_h, chi_h = _slope_probe_energy(
                    _run_forward, _compute_energy, dict(zip(coords, x_h))
                )
        except Exception as exc:  # noqa: BLE001 -- any failure is inconclusive
            return (
                "the slope check could not evaluate (its forward raised "
                f"{type(exc).__name__}: {exc})"
            )
        finally:
            _F3_LAST_DIAGNOSTICS.clear()
            _F3_LAST_DIAGNOSTICS.update(diag_snapshot)
            run.update(latch_snapshot)

        if chi_h != int(chi_post):
            return (
                "the slope check could not evaluate (its forward changed chi "
                f"{int(chi_post)} -> {chi_h})"
            )
        e0 = float(jax.device_get(jnp.real(energy0)))
        r = (e_h - e0) / (-h * g_norm)
        if not math.isfinite(r):
            return f"the slope check could not evaluate (non-finite ratio {r})"
        _F3_LAST_DIAGNOSTICS["flow_slope_ratio"] = r
        if _FLOW_SLOPE_RATIO_MIN <= r <= _FLOW_SLOPE_RATIO_MAX:
            return None
        return (
            f"its gradient failed the slope check (measured/predicted dE = "
            f"{r:.3g}, accepted range [{_FLOW_SLOPE_RATIO_MIN:g}, "
            f"{_FLOW_SLOPE_RATIO_MAX:g}])"
        )
```

`math` is already imported at the top of the module.

In `f_bwd`:
- Next to the two existing `pop`s at its top, add
  `_F3_LAST_DIAGNOSTICS.pop("flow_slope_ratio", None)`.
- Directly before the existing block
  `if reason is None:\n            _F3_LAST_DIAGNOSTICS["projector_backward_used"] = _PROJECTOR_BACKWARD_FLOW\n            return grads`,
  insert:

```python
        if reason is None:
            reason = _flow_slope_check(residuals, g, grads, run)
```

The existing fallback code below it (discard, latch-guarded warning, frozen
retry, diagnostics) handles the new reasons unchanged.

- [ ] **Step 6: Run the new tests and the #1028/policy set**

Run:
`cd ~/tenax-1111 && JAX_PLATFORMS=cpu PYTHONPATH=src ~/tenax-bench/.venv/bin/python -m pytest -q -o addopts="" -p no:cacheprovider -m "not slow" tests/test_flow_slope_check_1111.py tests/test_projector_backward_default_1028.py tests/test_ctm_unconverged_policy.py tests/test_adjoint_convergence_gate.py tests/test_gmres_single_matvec_wiring.py tests/test_shared_chain_rule_bwd.py`

Expected: all PASS.

If an existing test fails because its fixture's outcome changed (flow is now rejected by the slope check, or vice versa):
- run the same test on `feat/gmres-single-matvec` (`~/tenax-gmres1`) to confirm the change comes from the check;
- record the measured `flow_slope_ratio`;
- re-point the assertion to the new outcome with that number in a comment.

Never starve a solve or loosen a tolerance to keep an old outcome. Report every such change.

Then run the slow D=3 test once:
`... -m slow tests/test_flow_slope_check_1111.py::test_slope_check_rejects_the_d3_chi12_flow_gradient`.
It can take an hour or more on this host. Run it in the background and report the result.

- [ ] **Step 7: Commit**

```bash
cd ~/tenax-1111 && PATH=$HOME/.local/bin:$PATH pre-commit run --files src/tenax/algorithms/_ctm_energy_ad.py tests/test_flow_slope_check_1111.py tests/test_projector_backward_default_1028.py tests/conftest.py
git add src/tenax/algorithms/_ctm_energy_ad.py tests/test_flow_slope_check_1111.py tests/test_projector_backward_default_1028.py tests/conftest.py
git commit -m "fix(#1111): slope-check flowing-projector gradients on non-stationary forwards

One extra forward energy evaluation along the flow gradient's descent
direction; reject (frozen fallback) unless measured/predicted dE is in
[0.5, 2].  Garbage flow r <= 0.002, good flow 0.97-1.24, frozen 1.00-1.54.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg"
```

---

### Task 2: Docs and the regression sweep

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`, the `f_bwd` docstring only.
- Modify: `CHANGELOG.md`, under `## Unreleased` → `### Behavior Changes`.
- Modify: `docs/guide/algorithms/ipeps_ad_paths.md`, the projector-flow section. Find it with `grep -n "projector_backward" docs/guide/algorithms/ipeps_ad_paths.md`.

**Interfaces:**
- Consumes: Task 1's constants and the diagnostics key `flow_slope_ratio`.

- [ ] **Step 1: `f_bwd` docstring**

After the paragraph that begins "Deliberately NOT gated on the forward
stationarity residual", append this paragraph. Keep the existing text.

```
        A *solved* flowing adjoint is still not enough on a non-stationary
        forward (#1111): the single-matvec GMRES solves adjoints JAX's GMRES
        stalled on, and on the #1028 fixture the solved flowing gradient was
        1497x off finite differences (86x at D=3 chi=12, already on main).
        Neither stationarity nor the adjoint amplification separates those
        from good flowing gradients; one extra forward evaluation along the
        gradient's own descent direction does.  ``_flow_slope_check`` keeps
        flow only if measured/predicted dE lies in [0.5, 2] (garbage
        <= 0.002, good 0.97-1.24, frozen 1.00-1.54), and only spends that
        forward when the forward was not stationary.
```

- [ ] **Step 2: CHANGELOG**

Insert at the top of `### Behavior Changes` under `## Unreleased`:

```markdown
- **Flowing-projector gradients are slope-checked on non-stationary forwards
  (#1111).** A solved flowing adjoint can still give a gradient orders of
  magnitude off on a forward that is not a fixed point (measured 1497x on the
  #1028 fixture, 86x at D=3 chi=12). When the forward is not stationary, the
  backward now evaluates the energy once more along the gradient's descent
  direction and keeps the flowing gradient only if the measured change is
  0.5-2x the predicted one; otherwise it falls back to frozen projectors with
  the reason in `flow_fallback_reason` and the ratio in `flow_slope_ratio`.
  Measured: garbage flowing gradients 0.000-0.002, good ones 0.97-1.24. Cost:
  one extra forward CTM evaluation on non-stationary flowing steps; stationary
  forwards are unchanged.
```

- [ ] **Step 3: Guide**

In `docs/guide/algorithms/ipeps_ad_paths.md`, at the end of the
`#### Projector response in the gradient (projector_backward="flow")`
subsection, add:

```markdown
On a forward that is not a fixed point, a solved flowing adjoint can still
give a gradient that is badly wrong (#1111). The backward therefore checks a
flowing gradient there: it evaluates the energy once more along the
gradient's descent direction and keeps the gradient only if the measured
change is 0.5-2x the predicted one. Otherwise it falls back to frozen
projectors. `get_last_implicit_ad_diagnostics()` reports the ratio as
`flow_slope_ratio` and the reason as `flow_fallback_reason`. The check costs
one extra forward CTM evaluation, and only on non-stationary forwards.
```

- [ ] **Step 4: Regression sweep**

Run the not-slow tests of every file that touches `_ctm_energy_ad`, eight files in parallel on cores 64-127, one log per file:

```bash
S=/tmp/claude-1007/-home-yjkao-tenax/9d5185cf-8f92-4cd5-aae3-5ba5b2d21977/scratchpad/gmres
mkdir -p $S/sweep_1111
cd ~/tenax-1111 && grep -l "ctm_energy_implicit\|_ctm_energy_ad" tests/*.py | sort | xargs -P 8 -I{} taskset -c 64-127 $S/sweep_one.sh $HOME/tenax-1111 $S/sweep_1111 {}
for f in $S/sweep_1111/*.log; do echo "$(basename $f): $(tail -1 $f) | $(grep -E '[0-9]+ (passed|failed)' $f | tail -1)"; done
```

`sweep_one.sh` already exists in that directory and refuses a log that already exists.

Expected: rc=0 everywhere, except files where every test is slow (rc=5). For any failure, run the same test on `~/tenax-gmres1` (the base branch) and report both results.

- [ ] **Step 5: Commit**

```bash
cd ~/tenax-1111 && PATH=$HOME/.local/bin:$PATH pre-commit run --files src/tenax/algorithms/_ctm_energy_ad.py CHANGELOG.md docs/guide/algorithms/ipeps_ad_paths.md
git add src/tenax/algorithms/_ctm_energy_ad.py CHANGELOG.md docs/guide/algorithms/ipeps_ad_paths.md
git commit -m "docs(#1111): document the flowing-gradient slope check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg"
```
