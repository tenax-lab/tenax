# CTM Unconverged-Forward Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every CTM forward that feeds a gradient or a reported energy in the 2-site and 1-site iPEPS-AD optimizers (and `ctm_tensor_2site`) acts on `converged=False`. The default is to fail loudly; an explicit `"warn"` opt-out keeps today's control flow.

**Architecture:** A new `_ctm_convergence_policy.py` holds one exception, one warning and one checker. It reads a `CTMConfig.on_unconverged` switch. Five call sites apply site-specific rules:

| Site | Where | Rule |
|---|---|---|
| 1 | gradient forward, read from the implicit-AD diagnostics after `value_and_grad` | raise |
| 2 | warm-start cache | refuse to cache |
| 3 | line-search φ | `+inf` |
| 4 | fresh final evaluation | warm-env fallback, else raise |
| 5 | `ctm_tensor_2site(strict=)` | raise when strict |

The optimizers catch the new error next to `CTMRGGradientError`. They restore the best env (not clear it) and checkpoint-then-raise when recovery is impossible.

**Tech Stack:** Python 3.11+, JAX, pytest. Run via `uv run`; on headless or macOS use `JAX_PLATFORMS=cpu`.

**Spec:** `docs/superpowers/specs/2026-10-01-ctm-unconverged-policy-design.md`

## Global Constraints

- **New config field:** `CTMConfig.on_unconverged: Literal["raise", "warn"] = "raise"`. Any other value raises `ValueError` in `__post_init__`.
- **New public names:** `CTMNotConvergedError(RuntimeError)` and `CTMNotConvergedWarning(UserWarning)`, exported in `src/tenax/__init__.py` (lazy map **and** `__all__`) and mentioned in `README.md`.
- **Error message:** contains site, step, iterations, sv_diff, conv_tol, chi and the step multiplier. The multiplier is `getattr(info, "step_multiplier", None)`; `None` or NaN prints as `n/a`. Missing iterations, sv_diff or conv_tol print as `n/a`.
- **History:** when `return_history=True`, gains parallel lists `ctm_converged`, `ctm_sv_diff`, `ctm_step_multiplier` (one entry per successful step) and the key `final_env_source` (`"fresh"` or `"warm_fallback"`).
- **Reset log line:** `[iPEPS-AD] CTM forward not converged at step k (sweeps n, sv_diff x vs tol y, step multiplier m|n/a) — reset to best (#n/N)`.
- **Warn mode:** under `"warn"`, control flow is unchanged everywhere except site 2, which refuses to cache an unconverged env in **both** modes.
- **Out of scope:** split-CTM paths (`use_split`, `use_split_2s`), multisite, PESS and root-implicit are untouched.
- **New test file:** `tests/test_ctm_unconverged_policy.py`, registered as `"core"` in `tests/conftest.py` `_FILE_MARKERS`.
- **Git and repo rules:** commit with pre-commit (ruff, ruff-format) passing. Never use `todense()` on a symmetric path except for small tensors. Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QJU38nnL8t1KQvWs6VTTVg
  ```

## Review Focus

1. **Line-search probes with `probe_max_iter` / `probe_conv_tol` set (#503).** These probes are *designed* to stop early. A user who set them expects truncated probes, not every probe rejected as `+inf`. Site 3 applies `+inf` only when both probe overrides are `None`; with an override it keeps legacy behaviour and does not warn. Pinned by a test in Task 4.
2. **A forward that did not run.** If `value_and_grad` raised before the forward finished, or the explicit-AD / split path never writes the implicit-AD diagnostics, then `forward_converged` is absent. Site 1 must then skip the check, not crash on a missing key or reuse a stale value from the previous call. The optimizer pops the key before each `value_and_grad`. Pinned in Task 2.
3. **Unconverged at `best_params` itself.** The reset target cannot be converged, so recovery would loop. The optimizer must checkpoint and raise on the first occurrence at the best point, not burn the whole stall budget. Pinned in Task 2.
4. **χ changed since the best snapshot** (reactive or scheduled bump). Restoring a stale-χ best env would crash the next forward on a shape mismatch, so it must clear instead (the #518 path). Pinned in Task 2.
5. **`"warn"` mode on a fully unconverged run.** It must complete and return, with every history entry `ctm_converged=False`, rather than raising from a site that forgot to consult the policy. Pinned in Task 2 (site 1) and Task 5 (site 4).

---

### Task 1: Policy module, config field, exports

**Files:**
- Create: `src/tenax/algorithms/_ctm_convergence_policy.py`
- Modify: `src/tenax/algorithms/ipeps_config.py` (`CTMConfig` field block near `plateau_patience: int | None = 20`, line ~273, and `CTMConfig.__post_init__`, line ~324)
- Modify: `src/tenax/__init__.py` (lazy-import map near `"CTMConfig": (...)` line ~289; `__all__` near `"CTMConfig",` line ~508)
- Modify: `README.md` (iPEPS section; one short paragraph)
- Modify: `tests/conftest.py` (`_FILE_MARKERS`)
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Produces:
  - `class CTMNotConvergedError(RuntimeError)`: `__init__(self, info, site: str, step: int | None = None, *, conv_tol: float | None = None, chi: int | None = None)`; attributes `info`, `site`, `step`.
  - `class CTMNotConvergedWarning(UserWarning)`.
  - `check_ctm_converged(info, *, site: str, policy: str, step: int | None = None, conv_tol: float | None = None, chi: int | None = None) -> bool`.
  - `format_step_multiplier(info) -> str`.
  - `CTMConfig.on_unconverged`.
- `info` is any object with `.converged`; optional `.iterations`, `.sv_diff`, `.step_multiplier` are read with `getattr`.

- [ ] **Step 1: Register the test file and write the failing tests**

In `tests/conftest.py`, add to `_FILE_MARKERS` (keep alphabetical placement if the map is sorted; otherwise add next to `"test_ctm_hold.py"`):

```python
    "test_ctm_unconverged_policy.py": "core",
```

Create `tests/test_ctm_unconverged_policy.py`:

```python
"""CTM unconverged-forward policy (#1059 hotspot 8, #1060).

Spec: docs/superpowers/specs/2026-10-01-ctm-unconverged-policy-design.md
"""

from __future__ import annotations

import math
import warnings

import pytest

from tenax.algorithms._ctm_convergence_policy import (
    CTMNotConvergedError,
    CTMNotConvergedWarning,
    check_ctm_converged,
    format_step_multiplier,
)
from tenax.algorithms._ctm_python_loop import CTMConvergeInfo
from tenax.algorithms.ipeps_config import CTMConfig


def _info(converged, iterations=500, sv_diff=3.9e-7, **extra):
    info = CTMConvergeInfo(converged=converged, iterations=iterations, sv_diff=sv_diff)
    if not extra:
        return info

    class _WithExtra:
        pass

    obj = _WithExtra()
    for f in info._fields:
        setattr(obj, f, getattr(info, f))
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def test_check_passes_when_converged():
    assert check_ctm_converged(_info(True), site="gradient", policy="raise") is True


def test_check_raises_with_diagnostics():
    with pytest.raises(CTMNotConvergedError) as ei:
        check_ctm_converged(
            _info(False), site="gradient", policy="raise", step=7, conv_tol=1e-10, chi=12
        )
    msg = str(ei.value)
    for needle in ("gradient", "step 7", "500", "3.9e-07", "1e-10", "chi=12"):
        assert needle in msg, (needle, msg)
    assert ei.value.site == "gradient" and ei.value.step == 7


def test_check_warns_under_warn():
    with pytest.warns(CTMNotConvergedWarning, match="gradient"):
        out = check_ctm_converged(_info(False), site="gradient", policy="warn", step=3)
    assert out is False


def test_message_shows_numeric_step_multiplier():
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier -0\.99"):
        check_ctm_converged(_info(False, step_multiplier=-0.99), site="g", policy="raise")


@pytest.mark.parametrize("value", [None, float("nan")])
def test_message_shows_na_when_multiplier_missing_or_nan(value):
    info = _info(False) if value is None else _info(False, step_multiplier=value)
    assert format_step_multiplier(info) == "n/a"
    with pytest.raises(CTMNotConvergedError, match=r"step multiplier n/a"):
        check_ctm_converged(info, site="g", policy="raise")


def test_ctmconfig_default_and_validation():
    assert CTMConfig().on_unconverged == "raise"
    assert CTMConfig(on_unconverged="warn").on_unconverged == "warn"
    with pytest.raises(ValueError, match="on_unconverged"):
        CTMConfig(on_unconverged="bogus")


def test_public_exports():
    import tenax

    assert tenax.CTMNotConvergedError is CTMNotConvergedError
    assert tenax.CTMNotConvergedWarning is CTMNotConvergedWarning
    assert "CTMNotConvergedError" in tenax.__all__
    assert "CTMNotConvergedWarning" in tenax.__all__
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_convergence_policy'`.

- [ ] **Step 3: Implement the module**

Create `src/tenax/algorithms/_ctm_convergence_policy.py`:

```python
"""What to do when a CTM forward reports ``converged=False`` (#1059, #1060).

The implicit-AD gradient is only valid at an element-wise fixed point, and an
unconverged environment can report an energy below the variational bound
(measured: spinless t-V V=2, E+V = -0.3024 vs ED -0.2918).  Two distinct
non-convergence modes exist (a period-2 cycle with step multiplier ~ -1 and a
slow wander), and no cheap pre-check predicts either, so every consumer of a
forward checks the flag.  ``CTMConfig.on_unconverged`` picks the response.
"""

from __future__ import annotations

import math
import warnings

_VALID_POLICIES = ("raise", "warn")


def format_step_multiplier(info) -> str:
    """Signed slow-mode multiplier from #1061, or ``"n/a"``.

    ~ -1: flip (period-2) cycle, damping helps; ~ +1: slow monotone
    convergence, raise max_iter; otherwise a wander.
    """
    m = getattr(info, "step_multiplier", None)
    if m is None:
        return "n/a"
    try:
        m = float(m)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(m) else f"{m:.3g}"


def _fmt(x, spec: str) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(xf) or (isinstance(x, int) and x < 0):
        return "n/a"
    return format(x, spec)


def _describe(info, site, step, conv_tol, chi) -> str:
    where = f"site={site}" + (f", step {step}" if step is not None else "")
    return (
        f"CTM forward did not converge ({where}): sweeps "
        f"{_fmt(getattr(info, 'iterations', None), 'd')}, sv_diff "
        f"{_fmt(getattr(info, 'sv_diff', None), '.3g')} vs conv_tol "
        f"{_fmt(conv_tol, 'g')}, chi={_fmt(chi, 'd')}, step multiplier "
        f"{format_step_multiplier(info)}. An unconverged environment invalidates "
        f"the implicit-AD gradient and can report a sub-variational energy. Set "
        f"CTMConfig(on_unconverged='warn') to continue anyway."
    )


class CTMNotConvergedError(RuntimeError):
    """A CTM forward feeding a gradient or a reported energy did not converge."""

    def __init__(self, info, site: str, step: int | None = None, *,
                 conv_tol: float | None = None, chi: int | None = None):
        self.info = info
        self.site = site
        self.step = step
        super().__init__(_describe(info, site, step, conv_tol, chi))


class CTMNotConvergedWarning(UserWarning):
    """Emitted instead of CTMNotConvergedError under on_unconverged='warn'."""


def check_ctm_converged(info, *, site: str, policy: str, step: int | None = None,
                        conv_tol: float | None = None, chi: int | None = None) -> bool:
    """Return ``info.converged``; raise or warn per ``policy`` when False."""
    if policy not in _VALID_POLICIES:
        raise ValueError(f"on_unconverged must be one of {_VALID_POLICIES}, got {policy!r}")
    if bool(getattr(info, "converged", False)):
        return True
    if policy == "raise":
        raise CTMNotConvergedError(info, site, step, conv_tol=conv_tol, chi=chi)
    warnings.warn(_describe(info, site, step, conv_tol, chi), CTMNotConvergedWarning,
                  stacklevel=2)
    return False
```

In `src/tenax/algorithms/ipeps_config.py`, add to `CTMConfig` directly after `plateau_patience: int | None = 20` (keep the surrounding comment style):

```python
    # What to do when a CTM forward feeding a gradient or a reported energy
    # returns converged=False (#1059/#1060): "raise" (default) fails loudly;
    # "warn" keeps the legacy control flow and emits CTMNotConvergedWarning.
    # See tenax.algorithms._ctm_convergence_policy.
    on_unconverged: str = "raise"
```

At the top of `CTMConfig.__post_init__` (line ~324, before `valid_modes = {`):

```python
        if self.on_unconverged not in ("raise", "warn"):
            raise ValueError(
                f"CTMConfig.on_unconverged must be 'raise' or 'warn', "
                f"got {self.on_unconverged!r}"
            )
```

In `src/tenax/__init__.py`, add to the lazy map after the `"CTMConfig"` entry:

```python
    "CTMNotConvergedError": (
        "tenax.algorithms._ctm_convergence_policy",
        "CTMNotConvergedError",
    ),
    "CTMNotConvergedWarning": (
        "tenax.algorithms._ctm_convergence_policy",
        "CTMNotConvergedWarning",
    ),
```

and to `__all__` after `"CTMConfig",`:

```python
    "CTMNotConvergedError",
    "CTMNotConvergedWarning",
```

In `README.md`, inside the iPEPS section (next to the `CTMConfig` description), add:

```markdown
**Unconverged CTM forwards fail loudly.** When a CTM forward that feeds an AD gradient or a reported energy does not converge, `optimize_gs_ad` raises `CTMNotConvergedError`; line-search trials that do not converge are rejected. Set `CTMConfig(on_unconverged="warn")` to keep going with a `CTMNotConvergedWarning` instead.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_convergence_policy.py src/tenax/algorithms/ipeps_config.py \
  src/tenax/__init__.py README.md tests/conftest.py tests/test_ctm_unconverged_policy.py
git commit -m "feat(#1059): CTMNotConvergedError/Warning, check_ctm_converged, CTMConfig.on_unconverged"
```

---

### Task 2: Site 1 (gradient forward) and optimizer recovery, 2-site and 1-site

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (add `reset_forward_diagnostics()` next to `get_last_implicit_ad_diagnostics`, line ~817)
- Modify: `src/tenax/algorithms/ipeps_optimize.py`:
  - 2-site loop: the `try: energy_val, grads = jax.value_and_grad(loss_fn)(params)` block at line ~3484 and its `except CTMRGGradientError` branch; `_reset_env_cache_2s` at ~3130; history dict at ~4398.
  - 1-site loop: the matching `try`/`except CTMRGGradientError` at ~1790; history dict at ~2580.
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Consumes: from Task 1, `check_ctm_converged`, `CTMNotConvergedError`, `format_step_multiplier`, `CTMConfig.on_unconverged`.
- Produces:
  - `_ctm_energy_ad.reset_forward_diagnostics() -> None`: pops `"forward_converged"` and `"forward_stationarity_residual"` from `_F3_LAST_DIAGNOSTICS`.
  - Optimizer history lists `ctm_converged`, `ctm_sv_diff`, `ctm_step_multiplier`.

**Behaviour:**

- **Before each value_and_grad.** Right before `energy_val, grads = jax.value_and_grad(loss_fn)(params)` in each loop, call `reset_forward_diagnostics()` (only when `config.gs_implicit_ad` is true and the path is not split).
- **Checking the flag after it.** Inside the same `try`, after `grads = _euclidean_grads(grads)`:
  - read `d = get_last_implicit_ad_diagnostics()`;
  - if `"forward_converged" in d`, build `CTMConvergeInfo(converged=bool(d["forward_converged"]), iterations=-1, sv_diff=float(d.get("forward_stationarity_residual", float("nan"))))`. `-1` prints as `n/a`;
  - call `check_ctm_converged(info, site="gradient", policy=<ctm cfg>.on_unconverged, step=step + 1, conv_tol=<ctm cfg>.conv_tol, chi=<ctm cfg>.chi)`;
  - append to the history lists.
- **Catching the error.** Change `except CTMRGGradientError as exc:` to `except (CTMRGGradientError, CTMNotConvergedError) as exc:`. Inside, branch on `isinstance(exc, CTMNotConvergedError)`:
  - log the spec's reset line via `_logger.warning`, and print it when `gs_verbose`;
  - if `params is best_params`, or `config.gs_stall_recovery != "reset"`, or `stall_count + 1 > config.gs_stall_recovery_retries`: write the checkpoint (`_maybe_save_2s_checkpoint(step, ctm_cfg_2s.chi, best_energy, force_last=True)` in 2-site; the 1-site equivalent `_maybe_save_checkpoint(...)` at ~1752 with `force_last=True`), then `raise`;
  - otherwise: `stall_count += 1`; `params = best_params`; restore the env with the helper below; clear L-BFGS/CG state exactly as the existing reset branch does; `continue`.
  - The existing `CTMRGGradientError` handling stays byte-identical in its own branch.
- **Restore helper, 2-site.** New closure next to `_reset_env_cache_2s`:

  ```python
      def _restore_best_env_2s():
          """Reset for CTMNotConvergedError: restore the converged best env
          when its chi matches; otherwise fall back to _reset_env_cache_2s."""
          from tenax.algorithms.ad_utils import _env_chi

          best = best_env_cache_2s.get("envs") if best_env_cache_2s else None
          if best is not None and _env_chi(best) == ctm_cfg_2s.chi:
              _drop_env_cache_for_reset(_env_cache_2s)
              _env_cache_2s.update(best_env_cache_2s)
          else:
              _reset_env_cache_2s()
  ```

- **Restore helper, 1-site.** Same logic with `best_env_cache`, `_env_cache`, `ctm_cfg` and `_drop_env_cache_for_reset(_env_cache)` as the fallback.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ctm_unconverged_policy.py`:

```python
import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_energy_ad as _cea
import tenax.algorithms.ipeps_optimize as _opt
from tenax.algorithms.ipeps_config import iPEPSConfig


def _heisenberg_gate():
    sx = 0.5 * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * jnp.array([[0.0, -1j], [1j, 0.0]])
    sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        jnp.einsum("ij,kl->ikjl", sx, sx)
        + jnp.einsum("ij,kl->ikjl", sy, sy)
        + jnp.einsum("ij,kl->ikjl", sz, sz)
    ).real


def _rand(D, d, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((D, D, D, D, d)) + 1j * rng.standard_normal((D, D, D, D, d))
    return jnp.asarray(a / np.linalg.norm(a))


def _cfg(unit_cell, policy, *, max_iter=3, steps=3, ckpt=None, retries=2):
    return iPEPSConfig(
        unit_cell=unit_cell,
        max_bond_dim=2,
        ctm=CTMConfig(chi=4, max_iter=max_iter, min_iter=1, conv_tol=1e-14,
                      on_unconverged=policy),
        gs_num_steps=steps,
        gs_stall_recovery="reset",
        gs_stall_recovery_retries=retries,
        su_init=False,
        gs_conv_criterion="grad_norm",
        return_history=True,
        gs_checkpoint_path=ckpt,
    )


def _init(unit_cell):
    return (_rand(2, 2, 0), _rand(2, 2, 1)) if unit_cell == "2site" else _rand(2, 2, 0)


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site1_raises_and_checkpoints(unit_cell, tmp_path):
    cfg = _cfg(unit_cell, "raise", ckpt=str(tmp_path / "ck"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError) as ei:
            _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    assert ei.value.site == "gradient"
    assert (tmp_path / "ck" / "ckpt.last.pkl").exists()


@pytest.mark.parametrize("unit_cell", ["2site", "1x1"])
def test_site1_warn_completes_and_records(unit_cell):
    cfg = _cfg(unit_cell, "warn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init(unit_cell), cfg)
    history = out[-1]
    assert history["ctm_converged"], "no per-step convergence recorded"
    assert all(c is False for c in history["ctm_converged"])
    assert len(history["ctm_sv_diff"]) == len(history["ctm_converged"])
    assert len(history["ctm_step_multiplier"]) == len(history["ctm_converged"])


def test_site1_missing_diagnostic_skips_check(monkeypatch):
    """Review Focus 2: no forward_converged key -> no check, no stale reuse."""
    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", lambda: {})
    cfg = _cfg("2site", "raise", max_iter=200, steps=2)
    cfg.ctm.conv_tol  # noqa: B018  (config is frozen-ish; just exercise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), _cfg(
            "2site", "raise", max_iter=200, steps=2))
    assert out[-1]["ctm_converged"] == []


def test_reset_forward_diagnostics_pops_keys():
    _cea._F3_LAST_DIAGNOSTICS["forward_converged"] = False
    _cea._F3_LAST_DIAGNOSTICS["forward_stationarity_residual"] = 1.0
    _cea.reset_forward_diagnostics()
    d = _cea.get_last_implicit_ad_diagnostics()
    assert "forward_converged" not in d and "forward_stationarity_residual" not in d


def test_unconverged_at_best_raises_immediately(monkeypatch, tmp_path):
    """Review Focus 3: failure at best_params must not burn the stall budget."""
    calls = {"n": 0}
    real = _cea.get_last_implicit_ad_diagnostics

    def fake():
        d = real()
        if "forward_converged" in d:
            calls["n"] += 1
            d["forward_converged"] = False  # every gradient forward "fails"
        return d

    monkeypatch.setattr(_cea, "get_last_implicit_ad_diagnostics", fake)
    cfg = _cfg("2site", "raise", max_iter=200, steps=10, retries=5,
               ckpt=str(tmp_path / "ck"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CTMNotConvergedError):
            _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    # step 1 fails at the initial params, which are best_params -> raise at once
    assert calls["n"] == 1
```

> **Implementer note (Review Focus 4, χ mismatch):** add `test_restore_best_env_chi_mismatch_clears` by testing the helper logic directly. Call `_env_chi` on an env built with `chi=4` against a config `chi=6`, and assert the fallback path is chosen. Factor the 2-site helper's decision into a module-level pure function `_should_restore_best_env(best_envs, chi) -> bool` in `ipeps_optimize.py`, used by both closures, and unit-test it:
>
> ```python
> def test_should_restore_best_env():
>     from tenax.algorithms.ipeps_optimize import _should_restore_best_env
>     from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site
>     A, B = _rand(2, 2, 0), _rand(2, 2, 1)
>     eA, eB = ctm_tensor_2site(A, B, chi=4, max_iter=5)
>     envs = {(0, 0): eA, (1, 0): eB}
>     assert _should_restore_best_env(envs, 4) is True
>     assert _should_restore_best_env(envs, 6) is False
>     assert _should_restore_best_env(None, 4) is False
> ```
>
> `ctm_tensor_2site` takes dense or Tensor inputs. If the dense `jax.Array` is rejected, wrap it with `tenax.DenseTensor` exactly as `tests/test_ctm_hold.py` does.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v -k "site1 or reset_forward or at_best or should_restore"`
Expected: FAIL. `reset_forward_diagnostics` is missing, the history has no `ctm_converged` key, and no `CTMNotConvergedError` is raised.

- [ ] **Step 3: Implement**

In `_ctm_energy_ad.py`, after `get_last_implicit_ad_diagnostics`:

```python
def reset_forward_diagnostics() -> None:
    """Drop the forward-verdict keys so a later read cannot see a stale
    value from a previous call (the optimizer calls this before each
    value_and_grad; a missing key afterwards means no forward ran)."""
    _F3_LAST_DIAGNOSTICS.pop("forward_converged", None)
    _F3_LAST_DIAGNOSTICS.pop("forward_stationarity_residual", None)
```

In `ipeps_optimize.py`, module level near `_drop_env_cache_for_reset` (line ~69):

```python
def _should_restore_best_env(best_envs, chi) -> bool:
    """CTMNotConvergedError reset: restore the converged best env only when it
    exists and its chi matches the current chi (else the #518 clear path)."""
    if not best_envs:
        return False
    from tenax.algorithms.ad_utils import _env_chi

    return _env_chi(best_envs) == chi
```

Then apply the **Behaviour** bullets above in both loops:
- Initialize `_hist_ctm_converged: list[bool] = []`, `_hist_ctm_sv_diff: list[float] = []`, `_hist_ctm_mult: list = []` next to `_history_energies`.
- Add `"ctm_converged": _hist_ctm_converged, "ctm_sv_diff": _hist_ctm_sv_diff, "ctm_step_multiplier": _hist_ctm_mult` to the returned history dict.
- `ctm_step_multiplier` entries are `None` (the gradient-forward diagnostics carry no multiplier).
- Import `CTMNotConvergedError`, `check_ctm_converged` and `CTMConvergeInfo` inside the optimizer functions, next to the existing `from tenax.algorithms.ad_utils import CTMRGGradientError`.

- [ ] **Step 4: Run the tests**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: all PASS.

Then run the existing reset/cap/history tests:
`JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_stall_recovery_cap.py tests/test_ipeps_ad_history.py tests/test_frozen_layout_ad.py -v`
Expected: PASS. If one fails because it runs with an unconverged forward, do **not** pin it yet; record its name for Task 6.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_energy_ad.py src/tenax/algorithms/ipeps_optimize.py tests/test_ctm_unconverged_policy.py
git commit -m "feat(#1059): fail loudly on an unconverged gradient forward; restore best env on reset"
```

---

### Task 3: Site 2: never cache an unconverged warm-start env

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py`: `_update_env_cache_2s` (line ~3038, the `envs, info = python_loop_ctm_converge(...)` / `_env_cache_2s["envs"] = envs` pair) and 1-site `_update_env_cache` (line ~1332, the `envs, info = ...` / `_env_cache["envs"] = envs` pair).
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Consumes: `check_ctm_converged`, `CTMNotConvergedWarning` (Task 1).
- Produces: nothing new.

**Behaviour:**
- Replace the unconditional `_env_cache_2s["envs"] = envs` with:

  ```python
          if info.converged:
              _env_cache_2s["envs"] = envs
          else:
              _logger.warning(
                  "[iPEPS-AD] warm-start CTM refresh did not converge "
                  "(sweeps %d, sv_diff %.3g); keeping the previous env",
                  info.iterations, info.sv_diff,
              )
              if ctm_cfg_2s.on_unconverged == "warn":
                  check_ctm_converged(info, site="env_cache", policy="warn",
                                      conv_tol=ctm_cfg_2s.conv_tol, chi=ctm_cfg_2s.chi)
  ```

- Keep the `max_truncation_error` / `max_smallest_S` captures unchanged; they read `info`.
- Make the same change in 1-site `_update_env_cache` with `_env_cache` / `ctm_cfg`.
- **Do not raise here in either mode.** Site 1 is authoritative.

- [ ] **Step 1: Write the failing test**

```python
import tenax.algorithms._ctm_python_loop as _cpl


@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_site2_refuses_to_cache_unconverged(monkeypatch, policy):
    """Spy on the env each python_loop call is SEEDED with. An unconverged
    warm-start refresh must not become the next call's env_init."""
    real = _cpl.python_loop_ctm_converge
    seen = []
    poisoned = {"env": None, "armed": True}

    def spy(*a, **k):
        seen.append(k.get("env_init"))
        envs, info = real(*a, **k)
        if poisoned["armed"] and k.get("env_init") is not None:
            # first warm refresh: report it as unconverged
            poisoned["armed"] = False
            poisoned["env"] = envs
            return envs, info._replace(converged=False)
        return envs, info

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", spy)
    cfg = _cfg("2site", policy, max_iter=300, steps=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert poisoned["env"] is not None, "spy never saw a warm refresh"
    assert all(e is not poisoned["env"] for e in seen), "unconverged env was used as a seed"
```

> **Implementer note:** this test relies on `python_loop_ctm_converge` being imported inside `_optimize_gs_ad_tensor_2site` at call time (`from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge`, line ~1154), so patching the source module takes effect. If the poisoned refresh happens to be a `loss_fn_fwd` probe call rather than `_update_env_cache_2s`, tighten the spy: arm it only when the call stack contains `_update_env_cache_2s`, via `inspect.stack()` with a function-name check.

- [ ] **Step 2: Run to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v -k site2`
Expected: FAIL with "unconverged env was used as a seed".

- [ ] **Step 3: Implement** the **Behaviour** above in both functions.

- [ ] **Step 4: Run to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ctm_unconverged_policy.py
git commit -m "feat(#1059): never cache an unconverged warm-start CTM env"
```

---

### Task 4: Site 3: line-search φ returns +inf for an unconverged trial

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py`: 2-site `loss_fn_fwd` (line ~3209; the `envs, _ = python_loop_ctm_converge(... for_probe=True)` call) and 1-site `loss_fn_fwd` (line ~1477).
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Consumes: `check_ctm_converged` (Task 1).

**Behaviour:**
- Capture `info` instead of discarding it.
- `probe_overridden = ctm_cfg_2s.probe_max_iter is not None or ctm_cfg_2s.probe_conv_tol is not None` (Review Focus 1).
- If `not info.converged and not probe_overridden`:
  - under `"raise"`: do **not** write `_env_cache_2s["envs"]`; return `float("inf")`;
  - under `"warn"`: call `check_ctm_converged(..., site="line_search", policy="warn")`, then keep the legacy path (write the cache, return the energy).
- With a probe override: legacy path, no warning.
- Same in 1-site with `ctm_cfg` / `_env_cache`.

- [ ] **Step 1: Write the failing tests**

```python
def _probe_loss_2site(monkeypatch, policy, *, probe_max_iter=None):
    """Run a tiny 2-site optimization whose probe forwards report unconverged,
    capturing what the line search's phi returned."""
    import tenax.algorithms._line_search as _ls

    real_cpl = _cpl.python_loop_ctm_converge

    def unconverged_probes(*a, **k):
        envs, info = real_cpl(*a, **k)
        if k.get("max_iter") == (probe_max_iter or 300) and k.get("env_init") is not None:
            return envs, info._replace(converged=False)
        return envs, info

    phis = []
    real_hz = _ls.hager_zhang_line_search

    def spy_hz(phi, dphi, phi0, slope, **kw):
        phis.append(phi(1e-3))
        return 0.0, phi0, False

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", unconverged_probes)
    monkeypatch.setattr(_ls, "hager_zhang_line_search", spy_hz)
    ctm = CTMConfig(chi=4, max_iter=300, min_iter=1, conv_tol=1e-14,
                    on_unconverged=policy, probe_max_iter=probe_max_iter)
    cfg = iPEPSConfig(unit_cell="2site", max_bond_dim=2, ctm=ctm, gs_num_steps=1,
                      su_init=False, gs_conv_criterion="grad_norm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
        except CTMNotConvergedError:
            pass  # site 1 may raise on the real (truncated) gradient forward
    return phis, real_hz


def test_site3_unconverged_probe_is_inf(monkeypatch):
    phis, _ = _probe_loss_2site(monkeypatch, "raise")
    assert phis and all(math.isinf(p) for p in phis)


def test_site3_warn_returns_energy(monkeypatch):
    phis, _ = _probe_loss_2site(monkeypatch, "warn")
    assert phis and all(math.isfinite(p) for p in phis)


def test_site3_probe_override_keeps_legacy(monkeypatch):
    """Review Focus 1: probe_max_iter set -> truncated probes are expected."""
    phis, _ = _probe_loss_2site(monkeypatch, "raise", probe_max_iter=7)
    assert phis and all(math.isfinite(p) for p in phis)
```

> **Implementer note:** the spy marks a call as a probe by `max_iter` equal to the probe budget plus a non-None `env_init`. If that heuristic also catches `_update_env_cache_2s`, switch the probe test to call `loss_fn_fwd` through the line-search spy. The `phi` passed to `hager_zhang_line_search` *is* `loss_fn_fwd` composed with the trial retraction, so the spy's `phi(1e-3)` already exercises site 3 directly. In that case drop the `max_iter` condition and mark every forward made inside `spy_hz` as unconverged with a flag set around the `phi(...)` call.

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v -k site3`
Expected: `test_site3_unconverged_probe_is_inf` FAILS (finite φ); the other two PASS already.

- [ ] **Step 3: Implement** the **Behaviour** in both `loss_fn_fwd`s.

- [ ] **Step 4: Run to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ctm_unconverged_policy.py
git commit -m "feat(#1059): reject unconverged line-search trials (phi=+inf) unless probe overrides are set"
```

---

### Task 5: Site 4: fresh final evaluation falls back to the converged warm env

**Files:**
- Modify: `src/tenax/algorithms/ipeps_optimize.py`: `_eval_fresh_2site` (line ~4353) and its two call sites below it; 1-site `_eval_fresh` (line ~2531) and its call sites; both history dicts.
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Consumes: `check_ctm_converged`, `CTMNotConvergedError` (Task 1); `best_env_cache_2s` / `best_env_cache` and `_env_cache_2s` / `_env_cache` (existing).
- Produces: history key `final_env_source: str` (`"fresh"` | `"warm_fallback"`).

**Behaviour (2-site; the 1-site version is identical with its own names):**

- `_eval_fresh_2site(p, env_init=None, warm=None)` captures `info`. If `info.converged`, return as today, tagged `"fresh"`.
- If unconverged and `warm` is not None:
  - `warm` is the env dict converged at exactly these params: `best_env_cache_2s.get("envs")` for `best_params`, the post-step cache for `params` only if it was written by a converged `_update_env_cache_2s` (Task 3 guarantees that);
  - and `_should_restore_best_env(warm, ctm_cfg_2s.chi)`;
  - then recompute `E_` from `warm` with `compute_energy_ctm_tensor_2site` and return tagged `"warm_fallback"`, logging `[iPEPS-AD] fresh final CTM did not converge; reporting the converged warm env's energy`.
- Otherwise: `check_ctm_converged(info, site="final_energy", policy=ctm_cfg_2s.on_unconverged, conv_tol=ctm_cfg_2s.conv_tol, chi=ctm_cfg_2s.chi)`. Under `"raise"` this raises; under `"warn"` it warns and returns the legacy result tagged `"fresh"`.
- **Which warm env goes with which params:**
  - For `best_params` pass `warm=best_env_cache_2s.get("envs")`.
  - For `params` (the last iterate) pass `warm=_env_cache_2s.get("envs")` only when `params is best_params`. After the line search the cache was restored to the previous params (#502/#899), so it is not params' env otherwise; pass `None` in that case.
- Set `history["final_env_source"]` from whichever evaluation produced `E_gs`.

- [ ] **Step 1: Write the failing tests**

```python
def _final_eval_unconverged(monkeypatch):
    """Make every COLD (env_init=None) forward after the loop report unconverged."""
    real = _cpl.python_loop_ctm_converge
    state = {"in_final": False}

    def spy(*a, **k):
        envs, info = real(*a, **k)
        if state["in_final"] and k.get("env_init") is None:
            return envs, info._replace(converged=False)
        return envs, info

    monkeypatch.setattr(_cpl, "python_loop_ctm_converge", spy)
    return state


def test_site4_falls_back_to_warm(monkeypatch):
    state = _final_eval_unconverged(monkeypatch)
    real_save = _opt.save_checkpoint if hasattr(_opt, "save_checkpoint") else None
    # flip into "final" mode when the loop ends: the 2-site loop flushes a
    # force_last checkpoint right before the fresh evaluation, so hook that
    import tenax.algorithms._checkpoint as _ck
    orig = _ck.save_checkpoint

    def hook(*a, **k):
        state["in_final"] = True
        return orig(*a, **k)

    monkeypatch.setattr(_ck, "save_checkpoint", hook)
    cfg = _cfg("2site", "raise", max_iter=300, steps=1)
    cfg = cfg.__class__(**{**cfg.__dict__, "gs_checkpoint_path": None})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _opt.optimize_gs_ad(_heisenberg_gate(), _init("2site"), cfg)
    assert out[-1]["final_env_source"] == "warm_fallback"
    assert math.isfinite(out[2])
```

> **Implementer note:** the "enter final mode" hook above is fragile (it depends on the checkpoint flush ordering). Prefer a direct seam: factor the end-of-run block into a nested helper and expose a module-level flag. Or, simpler: spy on `_cpl.python_loop_ctm_converge` and flip `in_final` the first time `env_init is None` occurs **after** at least one call with `env_init is not None` has happened (the loop's warm calls precede the cold final ones; #899 removed `env_init` from the final evaluation only). Write it that way and delete the checkpoint hook.
>
> Also add:
> - `test_site4_raises_without_converged_warm`: patch so that the best-env snapshot is empty (`steps=0` is invalid; use `gs_num_steps=1` with a monkeypatched `_should_restore_best_env` returning `False`), and assert `CTMNotConvergedError` with `site == "final_energy"`;
> - `test_site4_warn_legacy`: same as above under `"warn"`; assert `pytest.warns(CTMNotConvergedWarning)` and `final_env_source == "fresh"`.

- [ ] **Step 2: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v -k site4`
Expected: FAIL with `KeyError: 'final_env_source'`.

- [ ] **Step 3: Implement** the **Behaviour** in both optimizers.

- [ ] **Step 4: Run to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py tests/test_ipeps_final_energy_is_fresh_899.py -v`
Expected: all PASS. The #899 test asserts the final evaluation is cold. A converged cold evaluation is still the first choice, so it must keep passing.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/ipeps_optimize.py tests/test_ctm_unconverged_policy.py
git commit -m "feat(#1059): final energy falls back to the converged warm env, else fails loudly"
```

---

### Task 6: Site 5 (`ctm_tensor_2site(strict=)`) and the regression inventory

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py`:
  - `_ctm_tensor_multisite` (def line ~1372; final `return envs` line ~1731): add keyword `_return_status: bool = False`; when true, `return envs, CTMConvergeInfo(converged=converged, iterations=budget if not converged else -1, sv_diff=final_diff)`.
  - `ctm_tensor_2site` (line ~1763): add `strict: bool = False` after `hold_perturbation`, and document it in the docstring.
- Create: `docs/superpowers/plans/2026-10-01-ctm-unconverged-policy.pinned.md` (the inventory; it becomes the PR description section)
- Modify: any test file the inventory flags (pin with `on_unconverged="warn"` plus a comment)
- Test: `tests/test_ctm_unconverged_policy.py`

**Interfaces:**
- Consumes: `CTMNotConvergedError` (Task 1); `CTMConvergeInfo` (existing).
- Produces: `ctm_tensor_2site(..., strict: bool = False)`.

- [ ] **Step 1: Write the failing test**

```python
def test_site5_ctm_tensor_2site_strict():
    from tenax.algorithms._ctm_tensor_convergence import ctm_tensor_2site

    A, B = _rand(2, 2, 0), _rand(2, 2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eA, eB = ctm_tensor_2site(A, B, chi=4, max_iter=2, conv_tol=1e-14)  # default: no raise
        with pytest.raises(CTMNotConvergedError) as ei:
            ctm_tensor_2site(A, B, chi=4, max_iter=2, conv_tol=1e-14, strict=True)
    assert ei.value.site == "ctm_tensor_2site"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctm_tensor_2site(A, B, chi=4, max_iter=400, conv_tol=1e-6, strict=True)  # converges: no raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v -k site5`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'strict'`.

- [ ] **Step 3: Implement**

In `_ctm_tensor_multisite`, add the keyword-only parameter `_return_status: bool = False` and replace the final `return envs` with:

```python
    if _return_status:
        from tenax.algorithms._ctm_python_loop import CTMConvergeInfo

        return envs, CTMConvergeInfo(
            converged=bool(converged),
            iterations=budget if not converged else -1,
            sv_diff=float(final_diff),
        )
    return envs
```

In `ctm_tensor_2site`, pass `_return_status=True`, unpack `envs, info`, and before returning:

```python
    if strict and not info.converged:
        from tenax.algorithms._ctm_convergence_policy import CTMNotConvergedError

        raise CTMNotConvergedError(info, "ctm_tensor_2site", conv_tol=conv_tol, chi=chi)
    return envs[(0, 0)], envs[(1, 0)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ctm_unconverged_policy.py -v`
Expected: all PASS.

- [ ] **Step 5: Regression inventory**

Run the core bucket, then the rest:
- `JAX_PLATFORMS=cpu uv run pytest -m core -q -p no:cacheprovider 2>&1 | tail -40`
- `JAX_PLATFORMS=cpu uv run pytest -m "not slow" -q 2>&1 | tail -60`

Compare failures against `main` (`git stash` is not available, because this branch only adds code; run the same command in `~/tenax-review` at c606103). Only failures that are new on this branch count. Per the memory "main fast-other chronic red", six CTM/QR tests fail bit-identically on main; ignore those.

Triage each new failure into exactly one bucket:
- **(a) Needs an unconverged forward on purpose** (e.g. it tests truncated budgets): pin with `CTMConfig(..., on_unconverged="warn")` and a one-line comment `# needs an unconverged forward: <why>; see #1059`.
- **(b) Was silently passing on an unconverged forward and asserting a number:** do NOT pin. Raise its `max_iter` until the forward converges, or fix the test's expectation. Note it as "encoded the bug".
- **(c) A genuine regression:** fix the implementation.

Record every (a) and (b) test (file::name, bucket, one-line reason) in `docs/superpowers/plans/2026-10-01-ctm-unconverged-policy.pinned.md`.

- [ ] **Step 6: Commit**

```bash
git add -A src tests docs/superpowers/plans/2026-10-01-ctm-unconverged-policy.pinned.md
git commit -m "feat(#1059): ctm_tensor_2site(strict=); regression inventory for the raise default"
```

---

## Self-review notes (plan author)

- **Spec coverage.**
  - Section 1 (helper, config, exports, multiplier): Task 1.
  - Section 2: site 1 → Task 2, site 2 → Task 3, site 3 → Task 4, site 4 → Task 5, site 5 → Task 6.
  - Section 3 (restore best env, checkpoint-and-raise, history, log line): Task 2.
  - Section 4: tests 1–2 → Task 1, 3–4 → Task 2, 5 → Task 3, 6 → Task 4, 7 → Task 5, 8 → Task 6. The inventory is Task 6, Step 5.
- **Deviation from the spec, to confirm in review.** Site 3 exempts runs that set `probe_max_iter` / `probe_conv_tol` (Review Focus 1). The spec did not consider #503's deliberately truncated probes. Without the exemption, every such user's line search would reject every trial.
- **Known soft spots.** The site-3 and site-4 tests identify probe and final calls by heuristics on `python_loop_ctm_converge` kwargs. Each has an implementer note naming a sturdier seam. A reviewer should check that the chosen seam really isolates the site under test.
